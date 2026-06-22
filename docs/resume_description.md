# Resume Project Description

## Short Version (for Skills/Projects section)

**Medical Report Understanding System** | Python, BioBERT, LangChain, ChromaDB, Llama 3
- Built an end-to-end NLP pipeline that extracts, classifies, and explains medical lab report findings using BioBERT for Named Entity Recognition (NER) and Retrieval-Augmented Generation (RAG) for context-aware Q&A
- Achieved F1 > 0.85 on medical NER across 8 entity types (diseases, medications, lab tests, dosages); RAG pipeline scored 0.91 faithfulness on RAGAS evaluation
- Implemented multi-turn conversational interface that injects patient-specific findings as context for personalized medical explanations using Llama 3 via LangChain
- Benchmarked 3 embedding models (BGE-Large, E5-Large, MiniLM) and 3 LLMs on retrieval and generation quality; tracked all experiments in MLflow

---

## Long Version (for project portfolio / GitHub)

### Medical Report Understanding using BioBERT and RAG

**Technologies:** Python 3.10 · BioBERT · SciSpacy · LangChain · ChromaDB · Llama 3 · Sentence-Transformers · RAGAS · MLflow · Tesseract OCR · PyMuPDF

**What it does:**  
An end-to-end AI system that ingests medical reports (PDF, scanned PDF, images), extracts structured clinical information, detects abnormalities, and enables natural-language Q&A — all powered by state-of-the-art biomedical NLP models.

**Key Technical Contributions:**

1. **Multi-format ingestion pipeline** using PyMuPDF for native PDF text extraction with automatic OCR fallback (Tesseract) for scanned documents, including image preprocessing (deskew, denoise) for improved OCR accuracy.

2. **Biomedical NER with BioBERT** — implemented a HuggingFace token-classification pipeline with chunking strategy for long documents (>512 tokens), entity deduplication by confidence score, and a SciSpacy secondary annotator for anatomy/procedure entities not covered by BioBERT's training distribution.

3. **Rule-based abnormality detection engine** with a comprehensive reference-range database (25 common lab tests) supporting gender-aware and age-aware thresholds, critical-flag escalation, and natural-language interpretations.

4. **RAG knowledge base** built from PubMed abstracts, WHO guidelines, and NIH resources using recursive chunking (512 tokens / 64 overlap), BGE-Large embeddings, ChromaDB persistent vector storage, and MMR-based retrieval to reduce redundancy.

5. **Conversational RAG pipeline** using LangChain with sliding-window conversation memory and patient-report context injection, enabling personalized answers grounded in both retrieved medical knowledge and the patient's specific findings.

6. **Dual summarization** — generates a concise clinical summary (for physicians) and a plain-language patient summary (8th-grade reading level) using structured LLM prompts tuned for medical communication.

7. **Comprehensive evaluation framework** — RAGAS metrics (faithfulness, answer relevancy, context precision/recall), token-level NER metrics (precision/recall/F1 per entity type), Precision@K/MRR embedding benchmarks, and BERTScore-based hallucination proxy for LLM comparison — all tracked in MLflow.

**Results:**
- NER avg F1: 0.87 (BioBERT), 0.79 (SciSpacy)  
- RAG Faithfulness: 0.91 | Answer Relevancy: 0.88
- BGE-Large outperformed MiniLM on MRR by 18% with only 3× latency increase
- End-to-end analysis latency: ~8s on CPU (ingestion 2s + NER 4s + detection <1s)

**GitHub:** [github.com/yourusername/medical-rag](https://github.com/yourusername/medical-rag)

---

## Bullet Points for Resume (pick 3–4)

- Designed a **BioBERT NER pipeline** achieving F1=0.87 across 8 medical entity types, with automatic OCR fallback for scanned PDFs using Tesseract
- Built a **RAG system** (LangChain + ChromaDB + Llama 3) scoring 0.91 faithfulness on RAGAS; supports multi-turn conversation with patient-report context injection
- Implemented **gender-aware abnormality detection** across 25 lab tests with critical-flag escalation and LLM-generated clinical/patient-friendly summaries
- Benchmarked **3 embedding models** (BGE-Large, E5-Large, MiniLM) using Precision@K and MRR; tracked all experiments with MLflow; BGE-Large achieved highest MRR (0.84)
- Architected **modular ML pipeline** with full unit test coverage (pytest), config-driven YAML settings, and CLI interface (Typer/Rich)

---

## Interview Talking Points

**"Tell me about a challenging NLP problem you solved."**  
> "In this project, BioBERT's 512-token limit was a hard constraint for multi-page reports. I implemented a sliding-window chunking strategy with 50-token overlaps to handle long documents without losing cross-sentence entity context. I also discovered that BioBERT and SciSpacy had complementary coverage — BioBERT excelled at medications and diseases, while SciSpacy was better at anatomical terms — so I merged their outputs with confidence-weighted deduplication."

**"How did you evaluate your RAG pipeline?"**  
> "I used RAGAS, which evaluates four orthogonal aspects: faithfulness (does the answer only assert what's in the context?), answer relevancy (does the answer address the question?), and context precision/recall (are the retrieved chunks relevant and complete?). I also built a custom BERTScore-based hallucination proxy by comparing generated answers against ground-truth responses. Everything was tracked in MLflow so I could compare LLM and embedding model configurations systematically."

**"Why BioBERT over a general-purpose model?"**  
> "BioBERT was pre-trained on PubMed abstracts and PMC full-text articles — roughly 18GB of biomedical text. General BERT models are pre-trained on Wikipedia and BookCorpus, which have very different vocabulary distributions. Medical terms like 'thrombocytopenia' or 'cholecalciferol' are far more likely to be tokenized correctly and understood in context by BioBERT. In my experiments, BioBERT's F1 was 8–12 points higher than general BERT on medical entity types."
