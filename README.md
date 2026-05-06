# 💊 RxParser — Smart Prescription Parser

> **T13 — Smart Document Parser** | SMAI Assignment 3  
> Handwritten medical prescription → structured JSON/CSV, powered by **Gemini 2.0 Flash + EasyOCR + Pydantic**.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows note for PDF support**: Install Poppler (needed by `pdf2image`):
>
> ```bash
> choco install poppler
> ```
>
> Or download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases).

### 2. Set your Gemini API Key

Get a free key at [aistudio.google.com](https://aistudio.google.com/).

Either:

- Enter it directly in the app's sidebar, **or**
- Add `GEMINI_API_KEY` in Streamlit Cloud's Secrets panel, **or**
- Copy `.env.example` → `.env` and paste your key locally:
  ```
  GEMINI_API_KEY=AIza...
  ```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. On Streamlit Cloud, create a new app from the repo.
3. Set the main file path to `app.py`.
4. Add `GEMINI_API_KEY` in the app Secrets panel.
5. Let Streamlit install `requirements.txt` and `packages.txt` automatically.

---

## 📁 Project Structure

```
Assignment3/
├── app.py                  ← Main Streamlit application
├── schema.py               ← Pydantic v2 models (Prescription + Medication)
├── gemini_utils.py         ← Gemini Flash multimodal extraction
├── ocr_utils.py            ← EasyOCR wrapper + PDF→image + preprocessing
├── batch_processor.py      ← Tier 2: bulk processing with rate-limit handling
├── medicine_names.py       ← 78 pharmaceutical reference terms
├── packages.txt            ← Streamlit Cloud system package list
├── sample_prescriptions/   ← (optional) Place sample .jpg files here
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🗂️ Features

### Tier 1 — Single File Mode

1. Upload any prescription image (JPG, PNG, BMP, TIFF, WEBP) or PDF
2. Preview the uploaded image alongside EasyOCR raw text
3. Click **Parse Prescription** → Gemini extracts structured data
4. Review/edit all extracted fields in an interactive form
5. Download as **JSON** or **CSV**

### Tier 2 — Batch Mode

1. Upload multiple files at once
2. Processing progress bar with per-file status
3. Combined medications table with `source_file` column
4. Download combined **JSON** array and **CSV**

---

## 📋 Extracted Schema

| Category       | Fields                                                                 |
| -------------- | ---------------------------------------------------------------------- |
| Doctor         | Name, Qualifications, Registration No.                                 |
| Clinic         | Name, Address, Contact                                                 |
| Patient        | Name, Age, Sex, Weight, Address                                        |
| Clinical       | Date, Chief Complaint, Diagnosis, Allergies                            |
| Medications    | Name, Strength, Form, Frequency, Duration, Route, Special Instructions |
| Investigations | Lab tests / imaging ordered                                            |
| Follow-up      | Review schedule, General advice                                        |
| Meta           | Raw OCR text, Confidence notes                                         |

---

## 🌏 Dataset

Optimized for the [Doctors' Handwritten Prescription Dataset](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset):

- 4,680 word-level prescription images
- 78 pharmaceutical term classes
- Supports English + Bengali mixed handwriting

The 78 known medicine names are used as a reference list:

- Injected into the Gemini prompt for better accuracy
- Used to **flag unknown medicines** post-extraction (yellow warning badges)

---

## 🏗️ Architecture

```
Upload Image/PDF
      │
      ▼
Image Preprocessing (PIL)
  - Grayscale + Contrast × 1.8
  - Sharpness × 2.0
  - Unsharp mask on pen strokes
  - Resize to max 1600px
      │
 ┌────┴────────────────────────┐
 │                             │
 ▼                             ▼
EasyOCR               Gemini 2.0 Flash
(raw text dump)    (multimodal vision)
                              │
               ┌──────────────▼──────────────┐
               │   JSON Schema Enforcement    │
               │   response_mime="app/json"   │
               │   Pydantic v2 Validation     │
               └──────────────┬──────────────┘
                              │
               ┌──────────────▼──────────────┐
               │   Unknown Medicine Warning   │
               │   (vs. 78-name reference list) │
               └──────────────┬──────────────┘
                              │
              ┌───────────────┴──────────────┐
              ▼                              ▼
    Editable Streamlit Form          Batch Table View
              │                              │
     ┌────────┴────────┐          ┌──────────┴────────┐
     ▼                 ▼          ▼                    ▼
  JSON download   CSV download  Combined JSON    Combined CSV
```

---

## ⚙️ Technical Notes

### Rate Limiting

- Free Gemini tier: ~15 RPM
- Batch mode automatically adds 1.5s delay between calls
- Exponential backoff (up to 3 retries) on `429` errors

### PDF Support

- Primary: `pdf2image` (requires Poppler, best quality)
- Fallback: `pypdf` embedded image extraction

### EasyOCR

- CPU mode by default (GPU auto-detected)
- English only (Gemini handles Bengali natively)
- Results filtered at confidence ≥ 0.25

---

## 🔒 Privacy

This app sends images to the **Google Gemini API**. On the free tier, data may be used by Google to improve models. For production healthcare use, upgrade to a paid plan with a **BAA** for HIPAA compliance.

---

## 📦 Dependencies

```
streamlit >= 1.35.0
google-genai >= 0.7.0
easyocr >= 1.7.1
pypdf >= 4.0.0
pdf2image >= 1.17.0
Pillow >= 10.0.0
pydantic >= 2.0.0
pandas >= 2.0.0
python-dotenv >= 1.0.0
numpy >= 1.24.0
```
