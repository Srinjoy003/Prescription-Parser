"""
medicine_names.py
-----------------
Reference list of commonly prescribed pharmaceutical names.

Used for:
  1. Enriching the Gemini prompt with domain pharmaceutical context.
  2. Post-extraction validation — flagging any extracted medicine name
     that doesn't match a known term (low-confidence warning).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Known medicine names used as a reference for extraction validation
# ---------------------------------------------------------------------------
KNOWN_MEDICINES: list[str] = [
    "Aceclofenac",
    "Amlodipine",
    "Amoxicillin",
    "Amoxiclav",
    "Ampicillin",
    "Antacid",
    "Atorvastatin",
    "Azithromycin",
    "Bromhexine",
    "Calcium",
    "Cetirizine",
    "Cefixime",
    "Cefuroxime",
    "Ceftriaxone",
    "Chlorpheniramine",
    "Ciprofloxacin",
    "Clindamycin",
    "Clonazepam",
    "Clopidogrel",
    "Cotrimoxazole",
    "Dexamethasone",
    "Diclofenac",
    "Domperidone",
    "Doxycycline",
    "Enalapril",
    "Erythromycin",
    "Escitalopram",
    "Esomeprazole",
    "Fexofenadine",
    "Fluconazole",
    "Folic Acid",
    "Furosemide",
    "Gabapentin",
    "Glibenclamide",
    "Gliclazide",
    "Hydrocortisone",
    "Hydroxychloroquine",
    "Ibuprofen",
    "Insulin",
    "Isosorbide",
    "Ivermectin",
    "Ketoconazole",
    "Levocetirizine",
    "Levofloxacin",
    "Lisinopril",
    "Losartan",
    "Mebendazole",
    "Metformin",
    "Metronidazole",
    "Montelukast",
    "Naproxen",
    "Nifedipine",
    "Nitrofurantoin",
    "Omeprazole",
    "Ondansetron",
    "Paracetamol",
    "Pantoprazole",
    "Prednisolone",
    "Propranolol",
    "Rabeprazole",
    "Ranitidine",
    "Rosuvastatin",
    "Salbutamol",
    "Sertraline",
    "Simvastatin",
    "Spironolactone",
    "Sucralfate",
    "Tamsulosin",
    "Tenofovir",
    "Terbinafine",
    "Tetracycline",
    "Tramadol",
    "Triamcinolone",
    "Valacyclovir",
    "Valsartan",
    "Vitamin B Complex",
    "Vitamin C",
    "Vitamin D",
    "Zinc",
]

# Lower-cased set for fast O(1) lookup
_KNOWN_LOWER: set[str] = {m.lower() for m in KNOWN_MEDICINES}


def is_known_medicine(name: str) -> bool:
    """Return True if *name* matches a known medicine name (case-insensitive)."""
    return name.strip().lower() in _KNOWN_LOWER


def get_known_medicines_prompt_text() -> str:
    """
    Return a compact, comma-separated string of all known medicines for
    inclusion in a Gemini prompt to improve extraction accuracy.
    """
    return ", ".join(KNOWN_MEDICINES)
