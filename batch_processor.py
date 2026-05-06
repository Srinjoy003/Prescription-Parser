"""
batch_processor.py
------------------
Tier-2 batch processing for the Smart Prescription Parser.

Accepts a list of uploaded file-like objects (name + bytes), processes each
sequentially (respecting Gemini free-tier rate limits), and returns:
  1. A list of Prescription objects.
  2. A pandas DataFrame (medications exploded — one row per med).
  3. A combined JSON list.

Rate limiting:
  - 1-second delay between API calls (stays within free-tier ~15 RPM).
  - Each call already has its own back-off in gemini_utils.extract_prescription.
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import pandas as pd

from gemini_utils import extract_prescription, flag_unknown_medicines
from ocr_utils import load_and_extract
from schema import Prescription

logger = logging.getLogger(__name__)

# Seconds to wait between successive Gemini calls
INTER_CALL_DELAY: float = 1.5


# ---------------------------------------------------------------------------
# Result dataclass for a single file
# ---------------------------------------------------------------------------
@dataclass
class BatchResult:
    filename: str
    prescription: Optional[Prescription] = None
    error: Optional[str] = None
    unknown_medicines: List[str] = field(default_factory=list)
    processing_time_s: float = 0.0

    @property
    def success(self) -> bool:
        return self.prescription is not None and self.error is None


# ---------------------------------------------------------------------------
# Main batch processor
# ---------------------------------------------------------------------------
def process_batch(
    uploaded_files: list,          # List of objects with .name and .read()
    api_key: str,
    run_ocr: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[list[BatchResult], pd.DataFrame, str]:
    """
    Process multiple prescription files.

    Parameters
    ----------
    uploaded_files : list
        Streamlit UploadedFile objects (or any obj with .name and .read()).
    api_key : str
        Google Gemini API key.
    run_ocr : bool
        Whether to run EasyOCR on each file (adds ~3-5s per file).
    progress_callback : callable(current, total, filename) | None
        Optional callback for progress reporting (used by Streamlit progress bar).

    Returns
    -------
    tuple[list[BatchResult], pd.DataFrame, str]
        - results: per-file BatchResult list
        - df: combined DataFrame (one row per medication)
        - json_str: combined JSON array of all prescriptions
    """
    results: list[BatchResult] = []
    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        filename = getattr(uploaded_file, "name", f"file_{i+1}")
        logger.info("Batch processing [%d/%d]: %s", i + 1, total, filename)

        if progress_callback:
            progress_callback(i, total, filename)

        start_time = time.time()
        result = BatchResult(filename=filename)

        try:
            # Read file bytes
            file_bytes = (
                uploaded_file.read()
                if hasattr(uploaded_file, "read")
                else uploaded_file.getvalue()
            )

            # Load image + optional OCR
            image, raw_ocr_text = load_and_extract(
                file_bytes, filename, run_ocr=run_ocr
            )

            # Gemini extraction
            prescription = extract_prescription(image, raw_ocr_text, api_key)
            result.prescription = prescription
            result.unknown_medicines = flag_unknown_medicines(prescription)

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process %s: %s", filename, exc)
            result.error = str(exc)

        result.processing_time_s = time.time() - start_time
        results.append(result)

        # Rate limiting between calls
        if i < total - 1:
            time.sleep(INTER_CALL_DELAY)

    # Final progress callback
    if progress_callback:
        progress_callback(total, total, "Done")

    # Build combined DataFrame
    df = _build_dataframe(results)

    # Build combined JSON
    json_str = _build_json(results)

    return results, df, json_str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_dataframe(results: list[BatchResult]) -> pd.DataFrame:
    """
    Construct a pandas DataFrame from batch results.
    One row per medication per file. Includes a 'source_file' column.
    """
    rows: list[dict] = []
    for result in results:
        if result.success and result.prescription is not None:
            for row in result.prescription.to_csv_rows():
                rows.append({
                    "source_file": result.filename,
                    "processing_time_s": round(result.processing_time_s, 2),
                    "unknown_medicine_flag": (
                        row.get("med_name", "") in result.unknown_medicines
                    ),
                    **row,
                })
        else:
            rows.append({
                "source_file": result.filename,
                "processing_time_s": round(result.processing_time_s, 2),
                "error": result.error or "Unknown error",
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _build_json(results: list[BatchResult]) -> str:
    """
    Serialise all successful prescriptions to a pretty-printed JSON array.
    Failed files are included with an error marker.
    """
    output: list[dict] = []
    for result in results:
        if result.success and result.prescription is not None:
            entry = result.prescription.model_dump()
            entry["_source_file"] = result.filename
            entry["_processing_time_s"] = round(result.processing_time_s, 2)
            output.append(entry)
        else:
            output.append({
                "_source_file": result.filename,
                "_error": result.error or "Unknown error",
                "_processing_time_s": round(result.processing_time_s, 2),
            })

    return json.dumps(output, indent=2, ensure_ascii=False)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return the DataFrame as UTF-8 CSV bytes (for st.download_button)."""
    return df.to_csv(index=False, encoding="utf-8").encode("utf-8")
