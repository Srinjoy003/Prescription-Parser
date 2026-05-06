"""
app.py
------
Smart Prescription Parser — Streamlit application.

Features:
  • Single mode: Upload one image/PDF → extract structured prescription →
    editable form → download JSON/CSV.
  • Batch mode: Upload multiple files → bulk processing with progress bar →
    combined table → download combined JSON/CSV.
  • Dark-themed, premium UI with card layout, gradient header, animations.
  • EasyOCR raw text shown in an expandable section for transparency.
  • Unknown medicine name warnings (flagged against a pharmaceutical reference list).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RxParser — Smart Prescription Analyzer",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark premium theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* ── Root / Global ── */
  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }
  .stApp {
      background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #12122a 100%);
      color: #e8eaf6;
  }

  /* ── Gradient Header ── */
  .rx-header {
      background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
      padding: 2.5rem 2rem 2rem;
      border-radius: 16px;
      margin-bottom: 1.5rem;
      box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
      text-align: center;
  }
  .rx-header h1 {
      color: #ffffff;
      font-size: 2.4rem;
      font-weight: 700;
      margin: 0;
      letter-spacing: -0.5px;
  }
  .rx-header p {
      color: rgba(255,255,255,0.85);
      margin: 0.4rem 0 0;
      font-size: 1.05rem;
  }

  /* ── Cards ── */
  .rx-card {
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      padding: 1.4rem 1.6rem;
      margin-bottom: 1.2rem;
      backdrop-filter: blur(10px);
      box-shadow: 0 4px 24px rgba(0,0,0,0.3);
      transition: box-shadow 0.2s ease;
  }
  .rx-card:hover {
      box-shadow: 0 8px 32px rgba(102,126,234,0.2);
  }
  .rx-card h3 {
      color: #a5b4fc;
      font-size: 1.05rem;
      font-weight: 600;
      margin: 0 0 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
  }

  /* ── Section Labels ── */
  .section-label {
      font-size: 0.75rem;
      font-weight: 600;
      color: #818cf8;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 0.8rem 0 0.2rem;
  }

  /* ── Pill / Badge ── */
  .pill {
      display: inline-block;
      background: rgba(102, 126, 234, 0.2);
      color: #a5b4fc;
      border: 1px solid rgba(102, 126, 234, 0.4);
      border-radius: 999px;
      padding: 0.15rem 0.7rem;
      font-size: 0.78rem;
      font-weight: 500;
      margin: 0.15rem;
  }
  .pill-warn {
      background: rgba(251, 191, 36, 0.15);
      color: #fbbf24;
      border-color: rgba(251, 191, 36, 0.4);
  }
  .pill-ok {
      background: rgba(52, 211, 153, 0.15);
      color: #34d399;
      border-color: rgba(52, 211, 153, 0.4);
  }
  .pill-error {
      background: rgba(248, 113, 113, 0.15);
      color: #f87171;
      border-color: rgba(248, 113, 113, 0.4);
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background: rgba(15, 12, 41, 0.95);
      border-right: 1px solid rgba(255,255,255,0.08);
  }
  [data-testid="stSidebar"] .stMarkdown h2 {
      color: #a5b4fc;
  }

  /* ── Buttons ── */
  .stButton > button {
      background: linear-gradient(90deg, #667eea, #764ba2);
      color: white;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      padding: 0.55rem 1.5rem;
      transition: opacity 0.2s, transform 0.15s;
  }
  .stButton > button:hover {
      opacity: 0.88;
      transform: translateY(-1px);
  }

  /* ── Text inputs ── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea,
  .stSelectbox > div > div {
      background: rgba(255,255,255,0.05) !important;
      border-color: rgba(255,255,255,0.15) !important;
      color: #e8eaf6 !important;
      border-radius: 8px !important;
  }

  /* ── Data editor ── */
  .stDataFrame, [data-testid="stDataFrameResizable"] {
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.1);
  }

  /* ── Expander ── */
  .streamlit-expanderHeader {
      background: rgba(255,255,255,0.04) !important;
      border-radius: 8px !important;
      color: #a5b4fc !important;
      font-weight: 500;
  }

  /* ── Download buttons ── */
  .stDownloadButton > button {
      background: rgba(52, 211, 153, 0.15);
      color: #34d399;
      border: 1px solid rgba(52, 211, 153, 0.4);
      border-radius: 8px;
      font-weight: 600;
      transition: all 0.2s;
  }
  .stDownloadButton > button:hover {
      background: rgba(52, 211, 153, 0.25);
      transform: translateY(-1px);
  }

  /* ── Progress ── */
  .stProgress > div > div {
      background: linear-gradient(90deg, #667eea, #f093fb) !important;
      border-radius: 4px;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
      background: rgba(255,255,255,0.04);
      border-radius: 10px;
      padding: 0.2rem;
  }
  .stTabs [data-baseweb="tab"] {
      border-radius: 8px;
      color: #9ca3af;
      font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
      background: rgba(102, 126, 234, 0.25) !important;
      color: #a5b4fc !important;
  }

  /* ── Alerts ── */
  .stAlert {
      border-radius: 10px !important;
  }

  /* ── Divider ── */
  hr { border-color: rgba(255,255,255,0.1); }

  /* ── Spinner text ── */
  .stSpinner > div > div {
      border-top-color: #667eea !important;
  }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Local imports (after page config)
# ---------------------------------------------------------------------------
from schema import Medication, Prescription
from gemini_utils import extract_prescription, flag_unknown_medicines
from ocr_utils import load_and_extract

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "prescription": None,
        "ocr_text": "",
        "source_image": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[str, bool]:
    """
    Render sidebar controls.
    Returns (api_key, run_ocr).
    """
    with st.sidebar:
        st.markdown("## 💊 RxParser")
        st.markdown("*Smart Prescription Analyzer*")
        st.divider()

        # API Key
        st.markdown("### 🔑 Gemini API Key")
        api_key = st.text_input(
            "Enter your Google Gemini API key",
            type="password",
            placeholder="AIza...",
            help="Get your free key at https://aistudio.google.com/",
            key="api_key_input",
        )

        # Try Streamlit Cloud secrets first, then local .env fallback.
        if not api_key:
            try:
                api_key = st.secrets.get("GEMINI_API_KEY", "")
            except Exception:
                api_key = ""

        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("GEMINI_API_KEY", "")
            except ImportError:
                api_key = os.getenv("GEMINI_API_KEY", "")

        if api_key:
            st.success("✅ API key loaded", icon="🔓")
        else:
            st.warning("Please enter your Gemini API key to proceed.", icon="⚠️")

        st.divider()

        # OCR toggle
        run_ocr = st.toggle(
            "Enable EasyOCR",
            value=True,
            help=(
                "Run EasyOCR as a supplementary text extractor. "
                "Adds ~3-5s per image but gives Gemini additional context."
            ),
        )

        st.divider()

        # Info
        st.markdown("""
**About this app**

Extracts structured data from handwritten medical prescriptions using:
- 🤖 **Gemini Flash** (multimodal vision)
- 👁️ **EasyOCR** (supplementary text)
- ✅ **Pydantic v2** (schema validation)

Supports handwritten and printed prescriptions.
        """)

    return api_key, run_ocr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _prescription_to_json_bytes(prescription: Prescription) -> bytes:
    return prescription.model_json(indent=2).encode("utf-8")


def _prescription_to_csv_bytes(prescription: Prescription) -> bytes:
    rows = prescription.to_csv_rows()
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def _display_unknown_medicine_warnings(unknown: list[str]):
    if unknown:
        warn_pills = " ".join(
            f'<span class="pill pill-warn">⚠️ {m}</span>' for m in unknown
        )
        st.markdown(
            f"**Unknown medicine(s) flagged** (not in reference list): {warn_pills}",
            unsafe_allow_html=True,
        )
        st.caption(
            "These names were extracted but don't match the known pharmaceutical "
            "terms. Please verify manually."
        )


# ---------------------------------------------------------------------------
# Single-file form renderer
# ---------------------------------------------------------------------------
def render_editable_form(prescription: Prescription) -> Prescription:
    """
    Render an editable form from a Prescription object and return the
    (potentially user-edited) Prescription back.
    """
    updated = prescription.model_dump()

    # ── Doctor / Clinic ────────────────────────────────────────────────────
    st.markdown('<div class="rx-card">', unsafe_allow_html=True)
    st.markdown("### 🩺 Doctor & Clinic Information")
    c1, c2 = st.columns(2)
    with c1:
        updated["doctor_name"] = st.text_input(
            "Doctor Name", value=prescription.doctor_name or "", key="f_doc_name"
        )
        updated["doctor_qualification"] = st.text_input(
            "Qualification", value=prescription.doctor_qualification or "", key="f_doc_qual"
        )
        updated["doctor_registration_no"] = st.text_input(
            "Registration No.", value=prescription.doctor_registration_no or "", key="f_doc_reg"
        )
    with c2:
        updated["clinic_name"] = st.text_input(
            "Clinic / Hospital", value=prescription.clinic_name or "", key="f_clinic_name"
        )
        updated["clinic_address"] = st.text_input(
            "Clinic Address", value=prescription.clinic_address or "", key="f_clinic_addr"
        )
        updated["clinic_contact"] = st.text_input(
            "Contact", value=prescription.clinic_contact or "", key="f_clinic_contact"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Patient ────────────────────────────────────────────────────────────
    st.markdown('<div class="rx-card">', unsafe_allow_html=True)
    st.markdown("### 👤 Patient Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        updated["patient_name"] = st.text_input(
            "Patient Name", value=prescription.patient_name or "", key="f_pat_name"
        )
        updated["patient_address"] = st.text_input(
            "Address", value=prescription.patient_address or "", key="f_pat_addr"
        )
    with c2:
        updated["patient_age"] = st.text_input(
            "Age", value=prescription.patient_age or "", key="f_pat_age"
        )
        updated["patient_sex"] = st.selectbox(
            "Sex",
            options=["", "Male", "Female", "Other"],
            index=["", "Male", "Female", "Other"].index(
                prescription.patient_sex or ""
            ) if prescription.patient_sex in ["", "Male", "Female", "Other", None] else 0,
            key="f_pat_sex",
        )
    with c3:
        updated["patient_weight"] = st.text_input(
            "Weight", value=prescription.patient_weight or "", key="f_pat_wt"
        )
        updated["date"] = st.text_input(
            "Date", value=prescription.date or "", key="f_date"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Clinical ───────────────────────────────────────────────────────────
    st.markdown('<div class="rx-card">', unsafe_allow_html=True)
    st.markdown("### 🔬 Clinical Details")
    c1, c2 = st.columns(2)
    with c1:
        updated["chief_complaint"] = st.text_area(
            "Chief Complaint / Symptoms",
            value=prescription.chief_complaint or "",
            height=80,
            key="f_complaint",
        )
        updated["diagnosis"] = st.text_area(
            "Diagnosis",
            value=prescription.diagnosis or "",
            height=80,
            key="f_diagnosis",
        )
    with c2:
        updated["known_allergies"] = st.text_input(
            "Known Allergies", value=prescription.known_allergies or "", key="f_allergies"
        )
        updated["follow_up"] = st.text_input(
            "Follow-up", value=prescription.follow_up or "", key="f_followup"
        )
        updated["advice"] = st.text_area(
            "General Advice",
            value=prescription.advice or "",
            height=80,
            key="f_advice",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Investigations ─────────────────────────────────────────────────────
    st.markdown('<div class="rx-card">', unsafe_allow_html=True)
    st.markdown("### 🧪 Investigations / Lab Tests")
    inv_text = st.text_area(
        "Lab tests ordered (one per line)",
        value="\n".join(prescription.investigations or []),
        height=80,
        key="f_investigations",
        help="Enter each investigation on a new line.",
    )
    updated["investigations"] = [
        line.strip() for line in inv_text.splitlines() if line.strip()
    ]
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Medications table ──────────────────────────────────────────────────
    st.markdown('<div class="rx-card">', unsafe_allow_html=True)
    st.markdown("### 💊 Medications")
    st.caption(
        "You can edit the table directly. Add/remove rows using the ± controls."
    )

    med_data = [
        {
            "Name": m.name,
            "Strength": m.strength or "",
            "Form": m.dosage_form or "",
            "Frequency": m.frequency or "",
            "Duration": m.duration or "",
            "Route": m.route or "",
            "Special Instructions": m.special_instructions or "",
        }
        for m in (prescription.medications or [])
    ]

    if not med_data:
        # Start with one empty row
        med_data = [{
            "Name": "",
            "Strength": "",
            "Form": "",
            "Frequency": "",
            "Duration": "",
            "Route": "",
            "Special Instructions": "",
        }]

    edited_df = st.data_editor(
        pd.DataFrame(med_data),
        num_rows="dynamic",
        use_container_width=True,
        key="f_medications_table",
        column_config={
            "Name": st.column_config.TextColumn("Medicine Name", required=True),
            "Strength": st.column_config.TextColumn("Strength (e.g. 500mg)"),
            "Form": st.column_config.SelectboxColumn(
                "Dosage Form",
                options=["", "Tablet", "Capsule", "Syrup", "Injection",
                         "Ointment", "Drops", "Inhaler", "Suppository", "Other"],
            ),
            "Frequency": st.column_config.TextColumn("Frequency"),
            "Duration": st.column_config.TextColumn("Duration"),
            "Route": st.column_config.SelectboxColumn(
                "Route",
                options=["", "Oral", "IV", "IM", "SC", "Topical",
                         "Sublingual", "Inhalation", "Other"],
            ),
            "Special Instructions": st.column_config.TextColumn("Special Instructions"),
        },
    )

    # Reconstruct Medication list from edited DataFrame
    new_meds: list[dict] = []
    for _, row in edited_df.iterrows():
        if str(row.get("Name", "")).strip():
            new_meds.append({
                "name": str(row.get("Name", "")),
                "strength": str(row.get("Strength", "")) or None,
                "dosage_form": str(row.get("Form", "")) or None,
                "frequency": str(row.get("Frequency", "")) or None,
                "duration": str(row.get("Duration", "")) or None,
                "route": str(row.get("Route", "")) or None,
                "special_instructions": str(row.get("Special Instructions", "")) or None,
            })
    updated["medications"] = new_meds

    st.markdown("</div>", unsafe_allow_html=True)

    # Keep transparency fields
    updated["raw_ocr_text"] = prescription.raw_ocr_text
    updated["confidence_notes"] = prescription.confidence_notes

    return Prescription.model_validate(updated)


# ---------------------------------------------------------------------------
# Single-file mode
# ---------------------------------------------------------------------------
def render_single_mode(api_key: str, run_ocr: bool):
    st.markdown("""
    <div class="rx-card">
      <h3>📤 Upload Prescription</h3>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload prescription image or PDF",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "pdf"],
        key="single_uploader",
        help="Supports JPG, PNG, BMP, TIFF, WEBP, and PDF.",
    )

    if uploaded is None and st.session_state.source_image is None:
        st.info(
            "📋 Upload a prescription image above to try the parser.",
            icon="ℹ️",
        )
        return

    # If new file uploaded, reset state
    if uploaded is not None:
        file_bytes = uploaded.read()
        with st.spinner("🔍 Pre-processing image …"):
            try:
                image, ocr_text = load_and_extract(
                    file_bytes, uploaded.name, run_ocr=run_ocr
                )
            except Exception as e:
                st.error(f"Failed to load image: {e}", icon="❌")
                return
        st.session_state.source_image = image
        st.session_state.ocr_text = ocr_text
        st.session_state.prescription = None  # Reset previous result

    image = st.session_state.source_image
    ocr_text = st.session_state.ocr_text

    # Display image + OCR in columns
    img_col, info_col = st.columns([1, 1])
    with img_col:
        st.markdown('<div class="rx-card"><h3>🖼️ Input Image</h3>', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="Uploaded Prescription")
        st.markdown("</div>", unsafe_allow_html=True)

    with info_col:
        if ocr_text:
            st.markdown('<div class="rx-card"><h3>👁️ EasyOCR Raw Text</h3>', unsafe_allow_html=True)
            with st.expander("View extracted raw text", expanded=False):
                st.text_area(
                    "Raw OCR output",
                    value=ocr_text,
                    height=250,
                    disabled=True,
                    key="raw_ocr_display",
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="rx-card" style="opacity:0.6;">'
                '<h3>👁️ EasyOCR</h3>EasyOCR is disabled or produced no output.</div>',
                unsafe_allow_html=True,
            )

    # Parse button
    st.markdown("---")
    parse_col, _ = st.columns([1, 3])
    with parse_col:
        parse_clicked = st.button(
            "🚀 Parse Prescription",
            key="btn_parse",
            disabled=not api_key,
            use_container_width=True,
        )

    if not api_key and parse_clicked:
        st.error("Please enter your Gemini API key in the sidebar.", icon="🔑")
        return

    if parse_clicked and api_key:
        with st.spinner("🤖 Gemini is reading the prescription …"):
            try:
                prescription = extract_prescription(image, ocr_text, api_key)
                st.session_state.prescription = prescription
                st.success(
                    "✅ Prescription parsed successfully! Review and edit below.",
                    icon="🎉",
                )
            except Exception as e:
                st.error(f"Extraction failed: {e}", icon="❌")
                return

    # Show editable form if we have a result
    if st.session_state.prescription is not None:
        prescription: Prescription = st.session_state.prescription

        st.markdown("---")
        st.markdown("## 📋 Extracted Prescription — Review & Edit")

        # Confidence notes
        if prescription.confidence_notes:
            st.info(
                f"**Model Note:** {prescription.confidence_notes}",
                icon="🤖",
            )

        # Unknown medicine warnings
        unknown = flag_unknown_medicines(prescription)
        _display_unknown_medicine_warnings(unknown)

        # Editable form
        edited_prescription = render_editable_form(prescription)

        # Download section
        st.markdown("---")
        st.markdown("## 💾 Download Results")
        dl_c1, dl_c2, dl_c3 = st.columns(3)

        with dl_c1:
            st.download_button(
                label="⬇️ Download JSON",
                data=_prescription_to_json_bytes(edited_prescription),
                file_name="prescription.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl_c2:
            st.download_button(
                label="⬇️ Download CSV",
                data=_prescription_to_csv_bytes(edited_prescription),
                file_name="prescription.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_c3:
            # Preview JSON
            with st.expander("👀 Preview JSON"):
                st.json(json.loads(edited_prescription.model_json()))





# ---------------------------------------------------------------------------
# Sample prescription loader
# ---------------------------------------------------------------------------
def _load_sample_image():
    """
    Load a built-in sample prescription image.
    Generates a synthetic demo image if no sample files are available.
    """
    sample_dir = Path(__file__).parent / "sample_prescriptions"
    sample_files = []
    if sample_dir.exists():
        sample_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))

    if sample_files:
        img = Image.open(sample_files[0]).convert("RGB")
    else:
        # Generate a synthetic blank prescription image
        img = _generate_synthetic_prescription()

    from ocr_utils import preprocess_image, resize_for_api, run_easyocr
    processed = preprocess_image(img)
    processed = resize_for_api(processed)
    ocr_text = run_easyocr(processed)

    st.session_state.source_image = processed
    st.session_state.ocr_text = ocr_text
    st.session_state.prescription = None
    st.success("Sample prescription loaded! Click 'Parse Prescription' to extract.", icon="✅")


def _generate_synthetic_prescription() -> Image.Image:
    """
    Generate a simple synthetic prescription image using PIL for demo purposes.
    This is used when no sample prescription image files are available.
    """
    from PIL import ImageDraw, ImageFont

    width, height = 600, 800
    bg_color = (255, 255, 255)
    text_color = (0, 0, 0)
    blue_color = (0, 80, 160)

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Try to use a default font
    try:
        font_large = ImageFont.truetype("arial.ttf", 18)
        font_med = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_large = ImageFont.load_default()
        font_med = font_large
        font_small = font_large

    # Header block
    draw.rectangle([(0, 0), (width, 110)], fill=(230, 240, 255))
    draw.text((20, 15), "Dr. Ahmed Rahman", fill=blue_color, font=font_large)
    draw.text((20, 42), "MBBS, FCPS (Medicine)", fill=text_color, font=font_med)
    draw.text((20, 62), "Dhaka Medical College & Hospital", fill=text_color, font=font_small)
    draw.text((20, 80), "Phone: +880-1712-345678", fill=text_color, font=font_small)
    draw.text((20, 96), "Reg: 45678", fill=text_color, font=font_small)
    draw.line([(0, 110), (width, 110)], fill=(100, 140, 200), width=2)

    y = 125
    draw.text((20, y), "Patient: Rahim Uddin", fill=text_color, font=font_med)
    draw.text((300, y), "Age: 45 yrs", fill=text_color, font=font_med)
    y += 22
    draw.text((20, y), "Sex: Male", fill=text_color, font=font_med)
    draw.text((300, y), "Wt: 68 kg", fill=text_color, font=font_med)
    y += 22
    draw.text((20, y), "Date: 15/01/2024", fill=text_color, font=font_med)

    y += 30
    draw.line([(20, y), (580, y)], fill=(180, 180, 180), width=1)
    y += 15

    draw.text((20, y), "Diagnosis: Acute Respiratory Tract Infection", fill=text_color, font=font_med)
    y += 22
    draw.text((20, y), "C/C: Fever, cough, sore throat × 3 days", fill=text_color, font=font_med)

    y += 30
    draw.line([(20, y), (580, y)], fill=(180, 180, 180), width=1)
    y += 10

    draw.text((20, y), "℞", fill=blue_color, font=font_large)
    y += 30

    meds = [
        ("1. Amoxicillin 500mg Cap", "1+0+1 × 7 days (after meals)"),
        ("2. Paracetamol 500mg Tab", "0+1+1 × 5 days (if fever >99°F)"),
        ("3. Cetirizine 10mg Tab", "0+0+1 × 7 days (at bedtime)"),
        ("4. Omeprazole 20mg Cap", "1+0+0 × 7 days (before breakfast)"),
    ]
    for med_name, med_inst in meds:
        draw.text((40, y), med_name, fill=text_color, font=font_med)
        y += 20
        draw.text((60, y), med_inst, fill=(80, 80, 80), font=font_small)
        y += 28

    y += 10
    draw.line([(20, y), (580, y)], fill=(180, 180, 180), width=1)
    y += 15
    draw.text((20, y), "Investigations: CBC, CXR (PA view)", fill=text_color, font=font_med)
    y += 25
    draw.text((20, y), "Advice: Take plenty of fluids. Rest.", fill=text_color, font=font_med)
    y += 25
    draw.text((20, y), "Follow-up: After 7 days or if worsening", fill=text_color, font=font_med)

    y += 60
    draw.line([(350, y), (560, y)], fill=text_color, width=1)
    draw.text((380, y + 8), "Signature", fill=text_color, font=font_small)

    return img


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    # Header
    st.markdown("""
    <div class="rx-header">
      <h1>💊 RxParser</h1>
      <p>Smart Handwritten Prescription Analyzer · Powered by Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    api_key, run_ocr = render_sidebar()

    # Main content tabs
    tab_single, tab_about = st.tabs(
        ["📄 Single File", "ℹ️ About"]
    )

    with tab_single:
        render_single_mode(api_key, run_ocr)

    with tab_about:
        render_about()


# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
def render_about():
    st.markdown("""
    <div class="rx-card">
      <h3>📖 About RxParser</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🔬 What is RxParser?

    **RxParser** is a Smart Document Parser specialized for **medical prescriptions**.
    It uses state-of-the-art AI to extract structured, validated data from handwritten
    or printed prescription images.

    ### 🏗️ Architecture

    ```
    Upload Image/PDF
         │
         ▼
    Pre-processing (PIL: contrast/sharpness)
         │
    ┌────┴──────────────────────┐
    │                           │
    EasyOCR               Gemini 2.0 Flash
    (raw text)          (multimodal vision)
                               │
                    Pydantic v2 Validation
                               │
                   Editable Form · JSON · CSV
    ```

    ### 📋 Extracted Fields

    | Category | Fields |
    |---|---|
    | **Doctor** | Name, Qualifications, Reg. No. |
    | **Clinic** | Name, Address, Contact |
    | **Patient** | Name, Age, Sex, Weight, Address |
    | **Prescription** | Date, Chief Complaint, Diagnosis, Allergies |
    | **Medications** | Name, Strength, Form, Frequency, Duration, Route, Instructions |
    | **Investigations** | Lab tests ordered |
    | **Follow-up** | Review instructions, General advice |

    ### 🌏 Pharmaceutical Reference

    The app validates extracted medicine names against a reference list of common
    pharmaceutical terms and flags any names not found for manual review.

    ### ⚙️ Tech Stack

    - **[Streamlit](https://streamlit.io/)** — UI framework
    - **[Google Gemini 2.0 Flash](https://aistudio.google.com/)** — Multimodal LLM
    - **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** — Supplementary OCR
    - **[Pydantic v2](https://docs.pydantic.dev/)** — Schema validation
    - **[pypdf](https://github.com/py-pdf/pypdf)** — PDF processing
    - **[Pillow](https://python-pillow.org/)** — Image processing

    ### 🔒 Privacy Notice

    This app sends prescription images to the Google Gemini API for processing.
    On the **free tier**, your data may be used to improve Google's models.
    For production healthcare applications, use the paid/enterprise tier with a
    **Business Associate Agreement (BAA)** for HIPAA compliance.

    ---

    _Built as part of SMAI Assignment 3 (T13 — Smart Document Parser)._
    """)


if __name__ == "__main__":
    main()
