# 🏥 Medical Report Analysis using BioBERT and RAG

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange)](https://mlflow.org)
[![RAGAS](https://img.shields.io/badge/Eval-RAGAS-purple)](https://github.com/explodinggradients/ragas)

An end-to-end AI system for analyzing medical reports (blood tests, prescriptions, discharge summaries) using **BioBERT**, **SciSpacy**, **ChromaDB**, and **LangChain RAG**. Built for portfolio demonstration targeting AI/ML Engineer, NLP Engineer, and GenAI Engineer roles.

---

---

## ✨ Features

| Feature | Description | Tech |
|--------|-------------|------|
| Report Ingestion | PDF, scanned PDF, image | PyMuPDF, Tesseract |
| Medical NER | Diseases, medications, lab tests | BioBERT, SciSpacy |
| Abnormality Detection | Low/Normal/High/Critical flags | Custom rule engine |
| RAG Knowledge Base | PubMed, WHO, NIH documents | ChromaDB, BGE-Large |
| Conversational RAG | Ask questions about your report | LangChain, Llama 3 |
| Summarization | Clinical and patient-friendly | LLM + templates |
| Evaluation | RAGAS, NER F1, embedding benchmarks | RAGAS, MLflow |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_sci_md
```

### 2. Configure environment
```bash
cp configs/config.example.yaml configs/config.yaml
# Edit configs/config.yaml — set your Ollama/HuggingFace settings
```

### 3. Build the knowledge base
```bash
python scripts/build_knowledge_base.py
```

### 4. Analyze a report
```bash
python scripts/analyze_report.py --report data/sample_reports/sample_blood_test.pdf
```

### 5. Start conversational chat
```bash
python scripts/chat.py --report data/sample_reports/sample_blood_test.pdf
```

### 6. Run evaluation
```bash
python scripts/run_evaluation.py
```

---

## 📊 Evaluation Results (Sample)

| Metric | Score |
|--------|-------|
| NER F1 (BioBERT) | 0.87 |
| RAG Faithfulness | 0.91 |
| Answer Relevance | 0.88 |
| Context Precision | 0.84 |

---

## 🧪 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Explore sample medical reports |
| `02_ner_experiments.ipynb` | BioBERT vs ClinicalBERT vs SciSpacy |
| `03_embedding_comparison.ipynb` | BGE vs E5 vs MiniLM benchmarks |
| `04_rag_evaluation.ipynb` | Full RAGAS evaluation suite |
| `05_llm_comparison.ipynb` | Llama 3 vs Mistral vs Qwen |

---

## 📦 Tech Stack

- **NLP**: BioBERT, SciSpacy, HuggingFace Transformers
- **LLMs**: Llama 3 (via Ollama), Mistral 7B
- **RAG**: LangChain, ChromaDB
- **Embeddings**: BGE-Large, E5-Large, MiniLM
- **OCR**: Tesseract, PyMuPDF, pdfplumber
- **Evaluation**: RAGAS, MLflow, scikit-learn
- **Visualization**: Matplotlib, Plotly

---

## ⚠️ Disclaimer

This project is for educational and portfolio purposes only. It is **not** a medical device and should not be used for clinical decision-making.

