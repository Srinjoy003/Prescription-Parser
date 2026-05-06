"""
gemini_utils.py
---------------
Integrates Google Gemini Flash (multimodal) for structured prescription extraction.

Key features:
  - Sends the prescription image + EasyOCR text together to Gemini.
  - Uses Gemini's native JSON-schema enforcement
    (response_mime_type="application/json" + response_schema=Prescription).
  - Implements exponential back-off (up to 3 retries) on 429 rate-limit errors.
  - Falls back to a best-effort JSON parse if schema enforcement fails.
  - Returns a validated Pydantic `Prescription` object.
"""

from __future__ import annotations

import io
import json
import logging
import time
from typing import Optional

from PIL import Image

from medicine_names import get_known_medicines_prompt_text
from schema import Medication, Prescription

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "gemini-flash-lite-latest"   # confirmed working on free tier
FALLBACK_MODEL = "gemini-flash-latest"     # fallback if primary unavailable
MAX_RETRIES = 4
BASE_BACKOFF = 2.0  # seconds; doubles each retry


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _build_system_prompt(raw_ocr_text: str) -> str:
    """
    Build a detailed system-level extraction prompt.
    EasyOCR raw text is embedded as additional context.
    Known medicine names are listed to guide the model.
    """
    known_meds = get_known_medicines_prompt_text()

    ocr_section = (
        f"\n\n## Supplementary OCR Text (EasyOCR)\n"
        f"The following raw text was automatically extracted from the image. "
        f"It may contain errors. Use it as a secondary signal only.\n"
        f"```\n{raw_ocr_text}\n```"
        if raw_ocr_text.strip()
        else "\n\n## Supplementary OCR Text\nNo OCR text was extracted."
    )

    return f"""You are a highly accurate medical document parser specialising in \
handwritten prescription analysis. Your task is to carefully examine the uploaded \
prescription image and extract all legible information into the required structured \
JSON format.

## Important Guidelines
1. The prescription may contain printed and/or handwritten text in any language.
2. Doctor's name and qualifications are usually printed at the top.
3. Patient details (name, age, sex, weight) appear below the header.
4. Rx section contains the list of medicines with dosage, frequency, and duration.
5. Investigations (lab tests) may appear separately or at the bottom.
6. Follow-up and advice are typically at the very bottom.
7. If a field is completely illegible or absent, set it to null.
8. For the `date` field, try to parse to YYYY-MM-DD; if uncertain, preserve the original.
9. For `confidence_notes`, describe ANY portion of the prescription that is unclear or \
illegible so the user knows to verify those fields manually.

## Known Pharmaceutical Terms
The following medicine names are commonly found in prescriptions. \
Use them as reference when deciphering handwritten drug names:
{known_meds}

## Required JSON Schema
Return ONLY a valid JSON object (no markdown fences, no extra text) with this exact structure:
{{
  "doctor_name": string or null,
  "doctor_qualification": string or null,
  "doctor_registration_no": string or null,
  "clinic_name": string or null,
  "clinic_address": string or null,
  "clinic_contact": string or null,
  "patient_name": string or null,
  "patient_age": string or null,
  "patient_sex": "Male" | "Female" | "Other" | null,
  "patient_weight": string or null,
  "patient_address": string or null,
  "date": string or null,
  "chief_complaint": string or null,
  "diagnosis": string or null,
  "known_allergies": string or null,
  "medications": [
    {{
      "name": string,
      "strength": string or null,
      "dosage_form": string or null,
      "frequency": string or null,
      "duration": string or null,
      "route": string or null,
      "special_instructions": string or null
    }}
  ],
  "investigations": [string],
  "follow_up": string or null,
  "advice": string or null,
  "confidence_notes": string or null
}}{ocr_section}"""


# ---------------------------------------------------------------------------
# Image → bytes helper
# ---------------------------------------------------------------------------
def _pil_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------
def extract_prescription(
    image: Image.Image,
    raw_ocr_text: str,
    api_key: str,
) -> Prescription:
    """
    Send *image* + *raw_ocr_text* to Gemini Flash and return a validated
    `Prescription` Pydantic object.

    Parameters
    ----------
    image : PIL.Image.Image
        The pre-processed prescription image.
    raw_ocr_text : str
        Raw text from EasyOCR (may be empty).
    api_key : str
        Google Gemini API key.

    Returns
    -------
    Prescription
        Validated Pydantic model. Fields that could not be extracted are None.

    Raises
    ------
    RuntimeError
        If extraction fails after all retries.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "google-genai is not installed. "
            "Run: pip install google-genai"
        ) from exc

    client = genai.Client(api_key=api_key)

    # Build prompt & image payload
    system_prompt = _build_system_prompt(raw_ocr_text)
    image_bytes = _pil_to_bytes(image)

    # ---------------------------------------------------------------------------
    # Retry loop with exponential back-off
    # ---------------------------------------------------------------------------
    last_error: Optional[Exception] = None
    raw_json_text = ""

    # Try primary model first, fall back to alternate if model-not-found
    models_to_try = [MODEL_NAME, FALLBACK_MODEL]
    active_model = MODEL_NAME

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Gemini extraction attempt %d/%d (model: %s) …", attempt, MAX_RETRIES, active_model)

            response = client.models.generate_content(
                model=active_model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text="Extract the prescription data as JSON.")
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    # NOTE: response_schema removed — it's unreliable with nested
                    # Optional Pydantic models; we parse JSON manually instead.
                    temperature=0.1,
                    top_p=0.9,
                )
            )

            raw_json_text = response.text.strip() if response.text else ""
            logger.info("Gemini raw response (first 200 chars): %s", raw_json_text[:200])

            if not raw_json_text:
                raise ValueError("Gemini returned an empty response.")

            # Strip accidental markdown fences if present
            if raw_json_text.startswith("```"):
                lines = raw_json_text.splitlines()
                raw_json_text = "\n".join(
                    l for l in lines if not l.strip().startswith("```")
                ).strip()

            data = json.loads(raw_json_text)

            # Validate via Pydantic
            prescription = Prescription.model_validate(data)

            # Attach raw OCR text for transparency
            if not prescription.raw_ocr_text:
                prescription.raw_ocr_text = raw_ocr_text or "Not extracted"

            logger.info("Extraction succeeded on attempt %d.", attempt)
            return prescription

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            err_str = str(exc).lower()
            logger.error("Gemini attempt %d failed: %s", attempt, exc)

            # Check for invalid API key
            is_auth_error = (
                "api_key_invalid" in err_str
                or "api key not valid" in err_str
                or "invalid api key" in err_str
                or "401" in err_str
                or "403" in err_str
                or "permission_denied" in err_str
                or "unauthenticated" in err_str
            )
            if is_auth_error:
                raise RuntimeError(
                    f"❌ Invalid or unauthorized API key. "
                    f"Please check your Gemini API key at https://aistudio.google.com/apikey\n"
                    f"Details: {exc}"
                ) from exc

            # Check for model-not-found — switch to fallback model
            is_model_error = (
                "not found" in err_str
                or "model" in err_str and "404" in err_str
                or "not_found" in err_str
            )
            if is_model_error and active_model == MODEL_NAME:
                logger.warning("Model %s not found, switching to %s", MODEL_NAME, FALLBACK_MODEL)
                active_model = FALLBACK_MODEL
                continue

            # Check for server-busy (503) — wait longer and retry
            is_server_busy = "503" in err_str or "unavailable" in err_str
            if is_server_busy and attempt < MAX_RETRIES:
                wait = 10.0 * attempt  # 10s, 20s, 30s
                logger.warning(
                    "Server busy 503 (attempt %d). Waiting %.1fs …", attempt, wait
                )
                time.sleep(wait)
                continue

            # Check for rate-limit signals
            is_rate_limit = (
                "429" in err_str
                or "resource exhausted" in err_str
                or "quota" in err_str
            )
            if is_rate_limit and attempt < MAX_RETRIES:
                wait = BASE_BACKOFF ** attempt
                logger.warning(
                    "Rate limit hit (attempt %d). Waiting %.1fs …", attempt, wait
                )
                time.sleep(wait)
                continue

            # If it's a JSON parse error on the last attempt, try best-effort recovery
            if attempt == MAX_RETRIES:
                try:
                    return _best_effort_parse(raw_json_text, raw_ocr_text)
                except Exception:  # noqa: BLE001
                    pass
            elif not is_rate_limit and not is_model_error:
                break  # Non-retriable error, stop immediately

    raise RuntimeError(
        f"Prescription extraction failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Best-effort fallback parser
# ---------------------------------------------------------------------------
def _best_effort_parse(raw_text: str, raw_ocr_text: str) -> Prescription:
    """
    Try to parse whatever JSON-like text Gemini returned, even if incomplete.
    Returns a Prescription with available fields and an error note.
    """
    logger.warning("Attempting best-effort JSON recovery …")
    try:
        # Locate first '{' … last '}'
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        fragment = raw_text[start:end]
        data = json.loads(fragment)
        prescription = Prescription.model_validate(data)
        prescription.raw_ocr_text = raw_ocr_text
        prescription.confidence_notes = (
            (prescription.confidence_notes or "")
            + " [WARNING: Partial extraction — some fields may be missing.]"
        )
        return prescription
    except Exception as exc:
        logger.error("Best-effort parse also failed: %s", exc)
        # Return a minimal prescription with OCR text at least
        return Prescription(
            raw_ocr_text=raw_ocr_text,
            confidence_notes=(
                "Extraction failed. Please review the raw OCR text and fill in fields manually."
            ),
        )


# ---------------------------------------------------------------------------
# Validation helper: flag unknown medicine names
# ---------------------------------------------------------------------------
def flag_unknown_medicines(prescription: Prescription) -> list[str]:
    """
    Return a list of medicine names in *prescription* that are NOT in the
    known medicine reference list.
    """
    from medicine_names import is_known_medicine  # lazy import

    unknown: list[str] = []
    for med in prescription.medications:
        if med.name and med.name not in ("Unknown", "") and not is_known_medicine(med.name):
            unknown.append(med.name)
    return unknown
