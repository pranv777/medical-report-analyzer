"""
src/utils/config.py
Loads and validates the YAML configuration, exposes a typed Settings object.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Nested config models
# ─────────────────────────────────────────────

class OCRConfig(BaseModel):
    engine: str = "tesseract"
    language: str = "eng"
    dpi: int = 300
    preprocess: bool = True


class IngestionConfig(BaseModel):
    supported_formats: List[str] = ["pdf", "png", "jpg", "jpeg", "tiff"]
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    fallback_to_ocr: bool = True
    min_text_length: int = 50


class ExtractionConfig(BaseModel):
    primary_model: str = "dmis-lab/biobert-base-cased-v1.2"
    fallback_model: str = "allenai/scibert_scivocab_uncased"
    spacy_model: str = "en_core_sci_md"
    entity_types: List[str] = Field(default_factory=list)
    confidence_threshold: float = 0.75
    batch_size: int = 16
    max_seq_length: int = 512
    use_gpu: bool = False


class DetectionConfig(BaseModel):
    reference_db_path: str = "data/reference_ranges.json"
    critical_alert_threshold: float = 0.2
    gender_aware: bool = True
    age_aware: bool = True


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64


class RetrievalConfig(BaseModel):
    top_k: int = 5
    score_threshold: float = 0.4
    search_type: str = "mmr"


class KnowledgeBaseConfig(BaseModel):
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    vector_store: str = "chromadb"
    chromadb_path: str = "data/knowledge_base/chroma_db"
    sources: dict = {  # ← CORRECT (dict, not list)
        'pubmed': True,
        'who_guidelines': True,
        'custom': 'data/knowledge_base/custom/'
    }
    collection_name: str = "medical_knowledge"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


class RAGConfig(BaseModel):
    llm_provider: str = "ollama"
    llm_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 1024
    system_prompt: str = ""


class SummarizationConfig(BaseModel):
    clinical_max_tokens: int = 600
    patient_max_tokens: int = 400
    reading_level: str = "8th grade"
    include_recommendations: bool = True
    include_follow_up_questions: bool = True
    num_follow_up_questions: int = 5


class MLflowConfig(BaseModel):
    enabled: bool = True
    tracking_uri: str = "logs/mlflow"
    experiment_name: str = "medical_rag_eval"


class EvaluationConfig(BaseModel):
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


class ProjectConfig(BaseModel):
    name: str = "Medical Report Understanding"
    version: str = "1.0.0"
    log_level: str = "INFO"
    output_dir: str = "outputs/"
    log_dir: str = "logs/"


class Settings(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# ─────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache(maxsize=1)
def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load settings from YAML, merge with environment overrides, return Settings.
    Cached — call once per process.
    """
    root = Path(__file__).resolve().parents[2]  # project root
    path = Path(config_path) if config_path else root / "configs" / "config.yaml"

    raw: Dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    # Allow environment variable overrides like MEDICAL_RAG__RAG__LLM_MODEL
    env_overrides = _env_overrides()
    merged = _deep_merge(raw, env_overrides)

    return Settings(**merged)


def _env_overrides() -> Dict[str, Any]:
    """
    Scan environment variables prefixed with MEDICAL_RAG__ and convert to
    nested dict. e.g. MEDICAL_RAG__RAG__LLM_MODEL=mistral
      → {"rag": {"llm_model": "mistral"}}
    """
    prefix = "MEDICAL_RAG__"
    overrides: Dict[str, Any] = {}
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        node = overrides
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return overrides
