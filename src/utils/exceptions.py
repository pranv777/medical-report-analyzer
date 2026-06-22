"""
src/utils/exceptions.py
Custom exception hierarchy for the medical RAG project.

Raise these instead of generic exceptions so callers can catch
specific failure modes without matching on string messages.
"""
from __future__ import annotations


# ─────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────

class MedicalRAGError(Exception):
    """Base exception for all medical-rag errors."""


# ─────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────

class IngestionError(MedicalRAGError):
    """Raised when report ingestion fails."""


class UnsupportedFormatError(IngestionError):
    """File format is not supported by the ingestion pipeline."""

    def __init__(self, fmt: str, supported: list):
        self.fmt = fmt
        self.supported = supported
        super().__init__(
            f"Unsupported format '{fmt}'. Supported: {supported}"
        )


class OCRError(IngestionError):
    """OCR processing failed or returned empty text."""


# ─────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────

class ExtractionError(MedicalRAGError):
    """Raised when NER or lab-value extraction fails."""


class ModelLoadError(ExtractionError):
    """Failed to load an NLP model."""

    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        super().__init__(f"Failed to load model '{model_name}': {reason}")


# ─────────────────────────────────────────────
# Knowledge base
# ─────────────────────────────────────────────

class KnowledgeBaseError(MedicalRAGError):
    """Raised for knowledge base / vector store errors."""


class EmptyKnowledgeBaseError(KnowledgeBaseError):
    """Knowledge base has no documents — retrieval is impossible."""


class EmbeddingError(KnowledgeBaseError):
    """Failed to generate embeddings for a batch of documents."""


# ─────────────────────────────────────────────
# RAG / LLM
# ─────────────────────────────────────────────

class RAGError(MedicalRAGError):
    """Raised when the RAG pipeline fails to produce an answer."""


class LLMUnavailableError(RAGError):
    """LLM endpoint is unreachable or not responding."""

    def __init__(self, provider: str, url: str):
        self.provider = provider
        self.url = url
        super().__init__(
            f"LLM provider '{provider}' unavailable at {url}. "
            "Check that Ollama is running or your API key is valid."
        )


class LLMResponseError(RAGError):
    """LLM returned an unexpected or empty response."""


# ─────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────

class DetectionError(MedicalRAGError):
    """Raised when abnormality detection fails."""


class ReferenceDBNotFoundError(DetectionError):
    """Reference range database file not found."""

    def __init__(self, path: str):
        super().__init__(
            f"Reference range database not found: {path}. "
            "Ensure data/reference_ranges.json exists."
        )


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

class EvaluationError(MedicalRAGError):
    """Raised when an evaluation step fails."""


class RAGASError(EvaluationError):
    """RAGAS evaluation failed — likely missing ground truth or LLM issue."""


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

class ConfigError(MedicalRAGError):
    """Invalid or missing configuration."""
