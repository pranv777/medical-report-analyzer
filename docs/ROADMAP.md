# 🗓 8-Week Development Roadmap

## Overview

| Week | Phase | Focus |
|------|-------|-------|
| 1 | Foundation | Environment, ingestion, basic NER |
| 2 | Extraction | BioBERT fine-tuning prep, entity pipeline |
| 3 | Detection | Abnormality engine, reference database |
| 4 | Knowledge Base | ChromaDB, embedding, document ingestion |
| 5 | RAG Pipeline | LangChain RAG, LLM integration, chat |
| 6 | Summarization | Clinical & patient summaries, follow-up Q gen |
| 7 | Evaluation | RAGAS, NER eval, embedding benchmarks, MLflow |
| 8 | Polish | Testing, documentation, GitHub, demo video |

---

## Week 1 — Foundation & Ingestion

**Goals:** Working environment, able to extract text from any report format.

### Tasks
- [ ] Set up Python virtual environment and install dependencies
- [ ] Configure `configs/config.yaml` for your system
- [ ] Install Tesseract OCR (`sudo apt install tesseract-ocr` or Homebrew)
- [ ] Pull Ollama + `llama3` model (`ollama pull llama3`)
- [ ] Implement and test `ReportIngestion` on sample PDFs and images
- [ ] Run `01_data_exploration.ipynb` on your own sample reports
- [ ] Write unit tests for ingestion (file not found, bad format, empty pages)

**Milestone:** `python scripts/analyze_report.py --report sample.pdf --no-summary` prints extracted text.

---

## Week 2 — Entity Extraction

**Goals:** BioBERT NER pipeline extracts all major medical entities reliably.

### Tasks
- [ ] Test `MedicalNERPipeline` on sample text — inspect entity types returned
- [ ] Tune `confidence_threshold` in config (start 0.75, lower if missing entities)
- [ ] Install SciSpacy model (`en_core_sci_md`)
- [ ] Compare BioBERT vs SciSpacy outputs side-by-side in `02_ner_experiments.ipynb`
- [ ] Validate lab-value regex on 5+ report formats
- [ ] (Optional) Fine-tune BioBERT on i2b2 2010 NER dataset for better coverage
- [ ] Write NER unit tests for common entity types

**Milestone:** `pipeline.run(text)` returns structured entities + lab values JSON for the sample report.

---

## Week 3 — Abnormality Detection

**Goals:** Every lab value is classified with correct status and clinical interpretation.

### Tasks
- [ ] Review `data/reference_ranges.json` — add any tests missing for your use case
- [ ] Test `AbnormalityDetector` against `LAB_VALUES` in test suite
- [ ] Validate gender-aware ranges (HDL, Hgb differ by sex)
- [ ] Add paediatric ranges where relevant (age-aware logic)
- [ ] Write tests for critical flags (glucose 510, Hgb 5.0)
- [ ] Confirm `DetectionReport.to_dict()` serializes cleanly to JSON
- [ ] Print formatted findings table with `pipeline.print_summary()`

**Milestone:** `make test` passes all detection tests. Report shows correct Low/High/Critical flags.

---

## Week 4 — Knowledge Base

**Goals:** ChromaDB populated with medical knowledge; retriever returns relevant chunks.

### Tasks
- [ ] Run `make build-kb` — verify 8+ built-in docs are indexed
- [ ] Run `scripts/fetch_pubmed.py` to download real PubMed abstracts
- [ ] Add 2–3 WHO/NIH PDF guidelines to `data/knowledge_base/custom/`
- [ ] Tune `chunk_size` and `chunk_overlap` — test retrieval quality manually
- [ ] Compare 2 embedding models in `03_embedding_comparison.ipynb`
- [ ] Validate `MedicalRetriever.retrieve()` returns relevant chunks for 10 queries
- [ ] Track embedding model choice in MLflow

**Milestone:** `retriever.retrieve("What causes iron deficiency anemia?")` returns 5 relevant chunks.

---

## Week 5 — RAG Pipeline & Chat

**Goals:** Full conversational RAG working end-to-end with report context injection.

### Tasks
- [ ] Verify Ollama is running (`ollama serve`) and `llama3` responds
- [ ] Test `MedicalRAGPipeline.ask()` on 10 medical questions
- [ ] Test multi-turn conversation with `_history` — confirm context is maintained
- [ ] Test `set_report_context()` — answers should reference patient findings
- [ ] Run `make analyze-chat` on the sample blood test report
- [ ] (Optional) Try `mistral` and compare response quality to `llama3`
- [ ] Log RAG latency per query in MLflow

**Milestone:** `make chat --report data/sample_reports/sample_blood_test.txt` gives accurate, context-aware answers.

---

## Week 6 — Summarization & Follow-up Questions

**Goals:** Clinical and patient summaries generated from detection results.

### Tasks
- [ ] Test `MedicalSummarizer.summarize()` on the sample report's detection output
- [ ] Evaluate clinical summary — is it accurate and concise?
- [ ] Evaluate patient summary — is it genuinely understandable for a non-medical person?
- [ ] Test follow-up question generation — are questions clinically relevant?
- [ ] Test recommendations list — actionable and specific?
- [ ] Tune prompts in `summarizer.py` for better output quality
- [ ] End-to-end: `make analyze` produces full JSON output including summaries

**Milestone:** `result.summary` contains a 3-sentence patient summary and 5 follow-up questions.

---

## Week 7 — Evaluation & Benchmarking

**Goals:** All evaluation metrics computed, logged in MLflow, visualized in notebooks.

### Tasks
- [ ] Run `make eval-ner` — record baseline F1 scores per entity type
- [ ] Run `make eval-embed` with at least 2 embedding models
- [ ] Run `make eval-ragas` — record faithfulness, relevancy, context metrics
- [ ] Open `make mlflow` and review experiment dashboard
- [ ] Run `04_rag_evaluation.ipynb` — generate RAGAS radar chart
- [ ] Run `05_llm_comparison.ipynb` — compare at least 2 LLMs
- [ ] Save all output charts to `outputs/` for README
- [ ] Write `outputs/evaluation_report.json` summary

**Milestone:** `outputs/evaluation_report.json` contains NER F1 > 0.75 and RAGAS faithfulness > 0.80.

---

## Week 8 — Polish, Testing & Documentation

**Goals:** Production-ready code, comprehensive tests, polished GitHub repo.

### Tasks
- [ ] Run full test suite: `make test-cov` — aim for >80% coverage
- [ ] Fix any failing tests
- [ ] Run `make lint` and fix all warnings
- [ ] Finalize `README.md` with actual evaluation numbers and screenshots
- [ ] Add architecture diagram to README (use `diagrams` library or draw.io)
- [ ] Record a 2-minute demo video showing CLI analysis + chat
- [ ] Create GitHub repo and push all code
- [ ] Tag `v1.0.0` release
- [ ] Update your resume with this project (see `docs/resume_description.md`)

**Milestone:** Public GitHub repo with passing tests, README with screenshots, and demo video.

---

## Key Decisions & Trade-offs

| Decision | Default Choice | Alternatives |
|----------|---------------|--------------|
| LLM inference | Ollama (local) | HuggingFace Hub, OpenAI API |
| Primary embedder | BGE-Large | E5-Large, MiniLM |
| Vector store | ChromaDB | FAISS, Weaviate, Qdrant |
| NER primary | BioBERT | ClinicalBERT, PubMedBERT |
| Chunking | Recursive (512/64) | Sentence-level, semantic |

---

## Stretch Goals (Post Week 8)

- [ ] FastAPI REST API wrapper (`/analyze`, `/chat`, `/summary` endpoints)
- [ ] Streamlit web UI for non-technical users
- [ ] Docker containerization (`docker-compose.yml`)
- [ ] Fine-tune BioBERT on i2b2 / n2c2 NER datasets
- [ ] UMLS entity normalization (link entities to standard medical codes)
- [ ] Multi-report longitudinal comparison ("your Hgb has improved from last visit")
- [ ] Export to PDF summary report using `reportlab`
