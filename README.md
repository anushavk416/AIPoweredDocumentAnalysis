# AI Powered Document Analysis

An AI-based system to automate the evaluation of bidder documents for large-scale tenders using OCR, vector search, and LLM-based query response — with Hindi & English support.

---

## Use Case

**Scenario:** Evaluating 1800+ documents from 200+ bidders in a tender (e.g., Housekeeping Staff Tender via GeM Portal)  
**Goal:** Automatically extract, understand, and verify qualification documents against predefined criteria such as turnover, experience, and geographic eligibility.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| OCR | Tesseract, PyMuPDF |
| Embeddings | DeepSeek via Ollama |
| Vector Store | In-Memory Vector DB |
| LLM Processing | LangChain |
| UI | Streamlit |
| Deployment | Conda + Localhost |
| Language Support | English (Latin), Hindi (Devanagari) |

---

## Setup & Operation Steps

### STEP 1: OCR Processing (in Colab or Locally)

1. Clone this repo and open the OCR script (`.ipynb`) in Google Colab or Jupyter.
2. Install necessary packages (Tesseract, OpenCV, etc.).
3. Upload the scanned/input PDF.
4. The script:
   - Performs OCR on text/image PDFs.
   - Supports **English & Hindi** script detection.
   - Outputs a cleaned, text-searchable **PDF with extracted content**.
5. Download the output file (timestamp-named PDF).

---

### STEP 2: Backend Setup (LLM & Embeddings)

#### Prerequisites

- [Install Python](https://www.python.org/downloads/)
- [Install Ollama](https://ollama.com/download)
- [Install Conda](https://docs.conda.io/en/latest/miniconda.html)

#### 🛠 Environment Setup

```bash
# Pull the DeepSeek model
ollama pull deepseek-r1:1.5b

# Create virtual environment
conda create -n rag python=3.10 -y
conda activate rag

# Install Python packages
pip install -r requirements.txt

# Start Ollama backend
ollama run deepseek-r1:1.5b   # or ollama serve
```

---

### STEP 3: Frontend (Streamlit UI)

```bash
# Launch the app
streamlit run app.py
```

---

### STEP 4: Using the App

1. Upload the **OCR-extracted PDF** via Streamlit UI.
2. Ask any **qualification-related query** (e.g., _"Is the bidder MSME?"_, _"What is their annual turnover?"_).
3. The model will:
   - Retrieve relevant document chunks using vector search.
   - Answer the query using DeepSeek.
   - Highlight supporting evidence from the PDF.

---

## Future Scope

- LLM pipelining to refine OCR extracted text
- Multi-shot prompting for complex queries across documents.
- Rule-based logic layer for auto-qualification (e.g., Geolocation, Financial Limits).
- Multi-document upload & simultaneous evaluation.
- Auto-flagging of missing or non-compliant documents.

---

- Anusha V Kumar
