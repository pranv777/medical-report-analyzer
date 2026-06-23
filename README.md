# Medical Report Understanding System

An AI-powered medical report analyzer that extracts clinical insights from blood test reports using BioBERT NER, LangChain RAG, and local LLM inference — with zero external API costs.

---

## What It Does

Upload a blood test report (PDF or TXT) and the system will:

1. **Extract** all lab values, test names, units, and reference ranges
2. **Classify** each result as Normal, Low, High, or Critical
3. **Identify** medical entities (diseases, medications, procedures) using BioBERT + SciSpacy
4. **Search** a medical knowledge base using semantic RAG
5. **Generate** a clinical summary, patient-friendly explanation, and actionable recommendations

---

## Sample Output

```
Detection complete: 22 normal, 13 abnormal, 0 critical

Lab Findings:
┌─────────────────────────┬───────┬────────┬────────┬──────────────────────────────────┐
│ Test                    │ Value │ Unit   │ Status │ Interpretation                   │
├─────────────────────────┼───────┼────────┼────────┼──────────────────────────────────┤
│ Haemoglobin             │ 10.5  │ g/dL   │ Low    │ Reference range: 13.5 - 17.5     │
│ Serum Iron              │ 45.0  │ µg/dL  │ Low    │ Reference range: 60.0 - 170.0    │
│ TIBC                    │ 410.0 │ µg/dL  │ High   │ Reference range: 250.0 - 370.0   │
│ Fasting Blood Glucose   │ 118.0 │ mg/dL  │ High   │ Reference range: 70.0 - 100.0    │
│ Triglycerides           │ 188.0 │ mg/dL  │ High   │ High result for triglycerides    │
│ Vitamin D (25-OH)       │ 14.5  │ ng/mL  │ Low    │ Reference range: 30.0 - 100.0    │
│ Platelet Count          │ 210.0 │ 10^3   │ Normal │ Reference range: 150.0 - 400.0   │
└─────────────────────────┴───────┴────────┴────────┴──────────────────────────────────┘
```

---

## Architecture

```
Input (PDF / TXT)
       │
       ▼
┌─────────────────┐
│   Ingestion     │  PyMuPDF + pdfplumber + OCR fallback
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NER Extraction │  BioBERT (chunked) + SciSpacy en_core_sci_md
│  Lab Parsing    │  Tabular regex parser with inline ref range detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Detection     │  Classifies Normal / Low / High / Critical
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────┐
│  Summarization  │────▶│  LangChain RAG        │
│  (Ollama LLM)   │     │  ChromaDB + BGE-large │
└────────┬────────┘     └──────────────────────┘
         │
         ▼
  JSON + CLI Output
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| NER | BioBERT (`dmis-lab/biobert-base-cased-v1.2`) + SciSpacy (`en_core_sci_md`) |
| RAG | LangChain 0.1 + ChromaDB + `BAAI/bge-large-en-v1.5` embeddings |
| LLM | Ollama (Llama3, runs fully locally) |
| PDF Processing | PyMuPDF + pdfplumber + Tesseract OCR |
| Config | Pydantic Settings + YAML |
| Testing | Pytest (160+ tests) |
| Language | Python 3.11 |

---

## Project Structure

```
medical-report-analyzer/
├── configs/
│   ├── config.yaml               # model and pipeline config
│   └── prompts.yaml              # LLM prompt templates
├── data/
│   ├── reference_ranges.json     # lab reference range database
│   ├── knowledge_base/           # ChromaDB vector store
│   └── sample_reports/           # sample blood test reports
├── scripts/
│   ├── analyze_report.py         # main CLI entrypoint
│   ├── build_knowledge_base.py   # builds ChromaDB from medical docs
│   ├── generate_sample_report.py # generates synthetic test reports
│   └── simple_analyze.py         # simplified analysis script
├── src/
│   ├── pipeline.py               # orchestrates full pipeline
│   ├── ingestion/                # PDF + TXT ingestion
│   ├── extraction/               # BioBERT + SciSpacy + lab parser
│   ├── detection/                # abnormality detection
│   ├── rag/                      # LangChain RAG pipeline
│   ├── knowledge_base/           # ChromaDB vector store builder
│   ├── summarization/            # Ollama LLM summarization
│   └── utils/                    # config, logging, helpers
├── tests/                        # 160+ unit tests
└── docs/
    └── resume_description.md
```

---

## Quick Start

### Prerequisites

- Python 3.11
- [Ollama](https://ollama.ai) installed and running

### Installation

```bash
# Clone the repo
git clone https://github.com/pranv777/medical-report-analyzer.git
cd medical-report-analyzer

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install SciSpacy medical model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

### Run

```bash
# Terminal 1 — Start Ollama
ollama serve
ollama pull llama3

# Terminal 2 — Build knowledge base (first time only)
python scripts/build_knowledge_base.py --reset

# Analyze a report
python scripts/analyze_report.py --report data/sample_reports/sample_blood_test.txt

# Generate a synthetic report and analyze it
python scripts/generate_sample_report.py --profile anemic --gender male --age 40 --format pdf
python scripts/analyze_report.py --report data/sample_reports/synthetic_anemic_male_40.pdf
```

---

## Key Technical Decisions

### BioBERT Chunking
BioBERT has a 512-token limit. Long medical reports are split into 200-word chunks with 20-word overlap, processed independently, and merged — ensuring no clinical context is lost.

### Inline Reference Range Detection
Instead of relying solely on a static reference range database, the lab parser extracts reference ranges directly from the report itself (e.g. `13.5 - 17.5` from the tabular format). This means the system works with any lab's report format without needing pre-configured ranges.

### Raw Text Preservation
The ingestion pipeline preserves original whitespace for tabular lab value parsing, while passing cleaned text to NER models. This dual-path approach ensures both accurate regex matching and clean NLP input.

### Local LLM (No API Costs)
Ollama runs Llama3 entirely on-device. No OpenAI or Anthropic API keys required — the system works fully offline after initial model download.

---

## Evaluation Results

| Metric | Result |
|---|---|
| Lab values extracted | 39 / 39 |
| Normal results detected | 22 |
| Abnormal results detected | 13 |
| Embedding Precision@1 | 1.000 |
| Embedding MRR | 1.000 |
| Unit tests passing | 160+ |

---

## Known Limitations

- Lab parser is optimised for tabular formats with multiple-space column separators
- BioBERT base model is not fine-tuned on NER — used for general entity detection
- Ollama summaries take 1–3 minutes per report on CPU inference
- Cholesterol sub-fractions (LDL, HDL) marked Unknown when report uses text thresholds (e.g. "Below 200") instead of numeric ranges

---

## Future Improvements

- [ ] Fine-tune BioBERT on medical NER datasets (i2b2, BC5CDR)
- [ ] FastAPI deployment for REST API access
- [ ] Support HL7 / FHIR report formats
- [ ] Add critical alert notifications
- [ ] Multi-language medical report support
- [ ] Docker containerization

---

## Research

Built alongside a research paper on **Wi-Fi RSSI-based Indoor Positioning Systems** accepted at **ICICA 2026**.

---

## License

MIT
