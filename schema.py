"""
schema.py
---------
Pydantic v2 models defining the structured JSON schema for a parsed
medical prescription. Covers handwritten prescriptions and any general
prescription format.

Export helpers:
  - Prescription.to_flat_dict()  → flat {column: value} dict (for CSV)
  - Prescription.model_json()    → pretty-printed JSON string
"""

from __future__ import annotations

import json
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-model: individual medication line
# ---------------------------------------------------------------------------
class Medication(BaseModel):
    """A single medicine entry on the prescription."""

    name: str = Field(
        default="Unknown",
        description="Generic or brand name of the prescribed medication.",
    )
    strength: Optional[str] = Field(
        default=None,
        description="Dosage strength, e.g. '500 mg', '250 mg/5 ml'.",
    )
    dosage_form: Optional[str] = Field(
        default=None,
        description="Physical form of the drug, e.g. 'tablet', 'capsule', 'syrup', 'injection'.",
    )
    frequency: Optional[str] = Field(
        default=None,
        description="How often to take, e.g. 'twice daily', 'TDS', 'once at night'.",
    )
    duration: Optional[str] = Field(
        default=None,
        description="Duration of the course, e.g. '5 days', '1 week', '1 month'.",
    )
    route: Optional[str] = Field(
        default=None,
        description="Route of administration, e.g. 'oral', 'topical', 'IV'.",
    )
    special_instructions: Optional[str] = Field(
        default=None,
        description=(
            "Any special patient instructions, e.g. 'after meals', "
            "'avoid sunlight', 'shake well before use'."
        ),
    )

    def to_flat_dict(self) -> dict:
        """Return a flat dictionary with 'med_' prefixed keys for CSV export."""
        return {
            "med_name": self.name,
            "med_strength": self.strength or "",
            "med_dosage_form": self.dosage_form or "",
            "med_frequency": self.frequency or "",
            "med_duration": self.duration or "",
            "med_route": self.route or "",
            "med_special_instructions": self.special_instructions or "",
        }


# ---------------------------------------------------------------------------
# Top-level model: full prescription
# ---------------------------------------------------------------------------
class Prescription(BaseModel):
    """Structured representation of a complete medical prescription."""

    # ── Doctor / Clinic Info ──────────────────────────────────────────────
    doctor_name: Optional[str] = Field(
        default=None,
        description="Full name of the prescribing doctor.",
    )
    doctor_qualification: Optional[str] = Field(
        default=None,
        description="Doctor's academic/professional qualifications, e.g. 'MBBS, FCPS'.",
    )
    doctor_registration_no: Optional[str] = Field(
        default=None,
        description="Medical council registration number if present.",
    )
    clinic_name: Optional[str] = Field(
        default=None,
        description="Name of the hospital, clinic, or chamber.",
    )
    clinic_address: Optional[str] = Field(
        default=None,
        description="Address of the clinic or hospital.",
    )
    clinic_contact: Optional[str] = Field(
        default=None,
        description="Phone number or contact info of the clinic.",
    )

    # ── Patient Info ──────────────────────────────────────────────────────
    patient_name: Optional[str] = Field(
        default=None,
        description="Full name of the patient.",
    )
    patient_age: Optional[str] = Field(
        default=None,
        description="Patient age, e.g. '35 years', '8 months'.",
    )
    patient_sex: Optional[str] = Field(
        default=None,
        description="Patient gender/sex: 'Male', 'Female', or 'Other'.",
    )
    patient_weight: Optional[str] = Field(
        default=None,
        description="Patient body weight, e.g. '65 kg'.",
    )
    patient_address: Optional[str] = Field(
        default=None,
        description="Patient's home address if written on the prescription.",
    )

    # ── Prescription Date ─────────────────────────────────────────────────
    date: Optional[str] = Field(
        default=None,
        description=(
            "Date the prescription was written. "
            "Return in YYYY-MM-DD format when possible; "
            "otherwise preserve the original text."
        ),
    )

    # ── Clinical Info ─────────────────────────────────────────────────────
    chief_complaint: Optional[str] = Field(
        default=None,
        description="Primary complaint(s) or presenting symptoms noted on the prescription.",
    )
    diagnosis: Optional[str] = Field(
        default=None,
        description="Doctor's diagnosis or clinical impression.",
    )
    known_allergies: Optional[str] = Field(
        default=None,
        description="Any documented drug or food allergies.",
    )

    # ── Medications ───────────────────────────────────────────────────────
    medications: List[Medication] = Field(
        default_factory=list,
        description="List of prescribed medications with full details.",
    )

    # ── Investigations / Labs ─────────────────────────────────────────────
    investigations: Optional[List[str]] = Field(
        default_factory=list,
        description=(
            "Laboratory tests or investigations ordered, "
            "e.g. ['CBC', 'Chest X-Ray', 'Blood Sugar Fasting']."
        ),
    )

    # ── Follow-up / Misc ──────────────────────────────────────────────────
    follow_up: Optional[str] = Field(
        default=None,
        description="Follow-up instruction, e.g. 'Review after 7 days'.",
    )
    advice: Optional[str] = Field(
        default=None,
        description="General health advice written on the prescription.",
    )

    # ── Meta ──────────────────────────────────────────────────────────────
    raw_ocr_text: Optional[str] = Field(
        default=None,
        description="Raw text extracted by EasyOCR, preserved for transparency.",
    )
    confidence_notes: Optional[str] = Field(
        default=None,
        description=(
            "Model's own notes about uncertain or illegible portions of the prescription."
        ),
    )

    # ── Helpers ───────────────────────────────────────────────────────────
    def model_json(self, indent: int = 2) -> str:
        """Return a pretty-printed JSON string of this prescription."""
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)

    def to_csv_rows(self) -> list[dict]:
        """
        Flatten into a list of row-dicts suitable for a CSV (one row per medication).
        If there are no medications, returns a single row with empty med fields.
        """
        base = {
            "doctor_name": self.doctor_name or "",
            "doctor_qualification": self.doctor_qualification or "",
            "doctor_registration_no": self.doctor_registration_no or "",
            "clinic_name": self.clinic_name or "",
            "clinic_address": self.clinic_address or "",
            "clinic_contact": self.clinic_contact or "",
            "patient_name": self.patient_name or "",
            "patient_age": self.patient_age or "",
            "patient_sex": self.patient_sex or "",
            "patient_weight": self.patient_weight or "",
            "patient_address": self.patient_address or "",
            "date": self.date or "",
            "chief_complaint": self.chief_complaint or "",
            "diagnosis": self.diagnosis or "",
            "known_allergies": self.known_allergies or "",
            "investigations": "; ".join(self.investigations or []),
            "follow_up": self.follow_up or "",
            "advice": self.advice or "",
        }
        if not self.medications:
            return [{**base, **Medication().to_flat_dict()}]
        return [{**base, **med.to_flat_dict()} for med in self.medications]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def empty_prescription() -> Prescription:
    """Return a Prescription instance with all fields set to None / empty."""
    return Prescription()
