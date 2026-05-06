"""
ocr_utils.py
------------
Utilities for:
  1. Converting uploaded PDF files to PIL Images (one per page).
  2. Pre-processing images for better OCR quality (especially handwriting).
  3. Running EasyOCR to extract raw text as a supplementary signal.

EasyOCR is used as a *transparency layer* — its raw dump is shown to the
user and also passed as additional context to the Gemini vision model.

Preprocessing pipeline (v2):
  - Adaptive binarization (Otsu-like via numpy) for pen-stroke separation
  - Stronger contrast (2.5) and sharpness (2.5)
  - Deskewing to handle tilted scans
  - Unsharp mask for crisp edges
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy EasyOCR reader — only initialised on first call (expensive import).
# ---------------------------------------------------------------------------
_easyocr_reader = None


def _get_reader():
    """Return a cached EasyOCR Reader for English (and Bengali fallback)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr  # type: ignore

            # English only — Bengali support requires extra model download.
            # For Bengali+English mixed docs, Gemini vision handles it better.
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("EasyOCR reader initialised (CPU, English).")
        except ImportError as exc:
            raise ImportError(
                "easyocr is not installed. Run: pip install easyocr"
            ) from exc
    return _easyocr_reader


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------
def _deskew(gray: Image.Image) -> Image.Image:
    """
    Estimate and correct skew angle of a grayscale image using
    the projection profile method. Corrects up to ±15 degrees.
    """
    try:
        arr = np.array(gray)
        # Binarize
        thresh = arr.mean()
        binary = (arr < thresh).astype(np.uint8)

        best_angle = 0
        best_score = -1
        for angle in range(-15, 16):
            rotated = Image.fromarray(binary * 255).rotate(
                angle, resample=Image.BICUBIC, expand=False
            )
            rot_arr = np.array(rotated) // 255
            row_sums = rot_arr.sum(axis=1).astype(float)
            score = float(np.var(row_sums))
            if score > best_score:
                best_score = score
                best_angle = angle

        if best_angle != 0:
            gray = gray.rotate(best_angle, resample=Image.BICUBIC, expand=False)
            logger.debug("Deskewed by %d degrees.", best_angle)
    except Exception as exc:
        logger.warning("Deskew failed: %s", exc)
    return gray


def _adaptive_binarize(gray: Image.Image) -> Image.Image:
    """
    Apply a simple block-wise adaptive threshold to separate dark pen
    strokes from varying backgrounds (works better than global threshold
    on uneven lighting / wrinkled paper).
    """
    try:
        arr = np.array(gray, dtype=np.float32)
        # Blur to get local background estimate
        blurred = Image.fromarray(arr.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=15)
        )
        bg = np.array(blurred, dtype=np.float32)
        # Subtract background; pixels darker than bg become dark
        diff = bg - arr
        # Threshold: pixels with positive diff (darker than BG) → black
        binary = np.where(diff > 10, 0, 255).astype(np.uint8)
        return Image.fromarray(binary)
    except Exception as exc:
        logger.warning("Adaptive binarize failed: %s", exc)
        return gray


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Apply a sequence of PIL / numpy operations that improve OCR accuracy
    on handwritten / low-contrast prescription scans.

    Steps (v2):
      1. Convert to RGB (safety — handles RGBA, P, etc.)
      2. Normalize brightness via ImageOps.autocontrast
      3. Convert to grayscale
      4. Deskew (projection-profile, ±15°)
      5. Adaptive binarization (local-background subtraction)
      6. Boost contrast (factor 2.5)
      7. Boost sharpness (factor 2.5)
      8. Unsharp mask to bring out pen strokes
      9. Convert back to RGB (EasyOCR & Gemini both accept RGB)

    Returns a new PIL Image; the original is not modified.
    """
    img = image.convert("RGB")

    # Auto-level brightness so overexposed / dim scans are normalised
    img = ImageOps.autocontrast(img, cutoff=1)

    gray = img.convert("L")

    # Deskew
    gray = _deskew(gray)

    # Adaptive binarization for better pen-stroke separation
    gray = _adaptive_binarize(gray)

    # Stronger contrast enhancement
    contrast_enhancer = ImageEnhance.Contrast(gray)
    gray = contrast_enhancer.enhance(2.5)

    # Stronger sharpness enhancement
    sharp_enhancer = ImageEnhance.Sharpness(gray)
    gray = sharp_enhancer.enhance(2.5)

    # Unsharp mask to bring out pen strokes
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=2))

    # Convert back to RGB so downstream tools receive consistent input
    result = gray.convert("RGB")
    return result


def resize_for_api(image: Image.Image, max_side: int = 1600) -> Image.Image:
    """
    Resize image so that neither dimension exceeds *max_side* pixels,
    preserving aspect ratio. Keeps memory and API payload reasonable.
    """
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# PDF → images
# ---------------------------------------------------------------------------
def pdf_to_images(file_bytes: bytes) -> list[Image.Image]:
    """
    Convert a PDF (supplied as raw bytes) to a list of PIL Images,
    one per page. Uses pdf2image (wraps poppler).

    Falls back gracefully if pdf2image / poppler is unavailable and
    tries pypdf's page-rendering as a last resort (lower quality).

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the PDF file.

    Returns
    -------
    list[Image.Image]
        One PIL Image per page.

    Raises
    ------
    RuntimeError
        If neither pdf2image nor pypdf can handle the file.
    """
    # --- Primary: pdf2image -----------------------------------------------
    try:
        from pdf2image import convert_from_bytes  # type: ignore

        images = convert_from_bytes(file_bytes, dpi=200)
        logger.info("pdf2image: converted %d page(s).", len(images))
        return images
    except ImportError:
        logger.warning("pdf2image not available; falling back to pypdf.")
    except Exception as exc:
        logger.warning("pdf2image failed (%s); falling back to pypdf.", exc)

    # --- Fallback: pypdf (text+image extraction, lower quality) -----------
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        images: list[Image.Image] = []
        for page in reader.pages:
            for img_ref in page.images:
                images.append(Image.open(io.BytesIO(img_ref.data)).convert("RGB"))
        if images:
            logger.info("pypdf fallback: extracted %d image(s).", len(images))
            return images
        raise RuntimeError("pypdf found no embedded images in the PDF.")
    except ImportError as exc:
        raise RuntimeError(
            "Neither pdf2image nor pypdf is installed. "
            "Run: pip install pdf2image pypdf"
        ) from exc


# ---------------------------------------------------------------------------
# EasyOCR text extraction
# ---------------------------------------------------------------------------
def run_easyocr(image: Image.Image) -> str:
    """
    Run EasyOCR on *image* and return all detected text as a single string.

    Each detected text block is joined by a newline. Low-confidence results
    (confidence < 0.3) are filtered to reduce noise.

    Parameters
    ----------
    image : PIL.Image.Image
        The image to process (RGB preferred).

    Returns
    -------
    str
        Extracted raw text, or an empty string on failure.
    """
    try:
        reader = _get_reader()
        img_np = np.array(image.convert("RGB"))

        # paragraph=True groups nearby words into lines for better readability
        results = reader.readtext(
            img_np,
            paragraph=False,       # keep per-word bboxes for confidence filtering
            min_size=10,           # skip tiny noise blobs
            contrast_ths=0.1,      # detect low-contrast ink
            adjust_contrast=0.5,   # internal EasyOCR contrast boost
            text_threshold=0.6,    # text detection confidence
            low_text=0.3,          # text map sensitivity
        )

        lines: list[str] = []
        for (_bbox, text, confidence) in results:
            # Lowered from 0.25 → 0.15 to catch more handwritten words
            if confidence >= 0.15 and text.strip():
                lines.append(text.strip())

        raw = "\n".join(lines)
        logger.info("EasyOCR extracted %d tokens (conf ≥ 0.15).", len(lines))
        return raw
    except Exception as exc:
        logger.error("EasyOCR failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Unified loader: bytes → (PIL Image, raw_ocr_text)
# ---------------------------------------------------------------------------
def load_and_extract(
    file_bytes: bytes,
    filename: str,
    run_ocr: bool = True,
) -> tuple[Image.Image, str]:
    """
    High-level helper that:
      1. Decodes the uploaded file (image or PDF first page).
      2. Preprocesses the image for optimal quality.
      3. Optionally runs EasyOCR and returns the raw text.

    Parameters
    ----------
    file_bytes : bytes
        Raw file contents.
    filename : str
        Original filename (used to determine if the file is a PDF).
    run_ocr : bool
        Whether to run EasyOCR (can be disabled for speed in batch mode).

    Returns
    -------
    tuple[Image.Image, str]
        (preprocessed_image, raw_ocr_text)  — raw_ocr_text is "" if skipped.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        pages = pdf_to_images(file_bytes)
        # Use only the first page for single-mode; caller handles multi-page
        raw_image = pages[0] if pages else Image.new("RGB", (400, 600), "white")
    else:
        raw_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    processed = preprocess_image(raw_image)
    processed = resize_for_api(processed)

    raw_ocr_text = run_easyocr(processed) if run_ocr else ""
    return processed, raw_ocr_text
