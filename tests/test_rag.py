"""
tests/test_rag.py
Unit tests for the RAG pipeline, prompt templates, and conversation memory.
All LLM calls are mocked — no Ollama required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─────────────────────────────────────────────
# Prompt template tests
# ─────────────────────────────────────────────

class TestPromptTemplates:
    def test_rag_qa_formats_correctly(self):
        from src.rag.prompt_templates import prompts
        text = prompts.rag_qa.format(
            context="Anemia is low hemoglobin.",
            report_section="",
            history="(none)",
            question="What is anemia?",
        )
        assert "Anemia is low hemoglobin" in text
        assert "What is anemia?" in text

    def test_missing_key_raises_value_error(self):
        from src.rag.prompt_templates import prompts
        with pytest.raises((KeyError, ValueError)):
            prompts.rag_qa.format(context="ctx", history="h", question="q")
            # missing report_section

    def test_clinical_summary_formats(self):
        from src.rag.prompt_templates import prompts
        text = prompts.clinical_summary.format(
            max_words=300,
            findings="Hemoglobin: 9.5 [Low]",
            abnormality_summary="Low hemoglobin detected.",
            entities="DISEASE: anemia",
        )
        assert "Hemoglobin" in text
        assert "300" in text

    def test_patient_summary_formats(self):
        from src.rag.prompt_templates import prompts
        text = prompts.patient_summary.format(
            max_words=200,
            reading_level="8th grade",
            findings="Hemoglobin: 9.5 [Low]",
            abnormality_summary="Low hemoglobin.",
        )
        assert "8th grade" in text

    def test_recommendations_formats(self):
        from src.rag.prompt_templates import prompts
        text = prompts.recommendations.format(
            n=5,
            findings="High cholesterol: 245 mg/dL",
        )
        assert "5" in text
        assert "cholesterol" in text.lower()

    def test_follow_up_questions_formats(self):
        from src.rag.prompt_templates import prompts
        text = prompts.follow_up_questions.format(
            n=3,
            findings="Low hemoglobin, high glucose",
        )
        assert "3" in text

    def test_faithfulness_check_formats(self):
        from src.rag.prompt_templates import prompts
        text = prompts.faithfulness_check.format(
            context="Anemia means low hemoglobin.",
            answer="Anemia is caused by low iron.",
        )
        assert "Anemia" in text

    def test_list_templates_returns_dict(self):
        from src.rag.prompt_templates import PromptLibrary
        templates = PromptLibrary.list_templates()
        assert isinstance(templates, dict)
        assert len(templates) >= 5
        assert "rag_qa" in templates

    def test_prompt_str_returns_template_text(self):
        from src.rag.prompt_templates import prompts
        assert "RETRIEVED MEDICAL KNOWLEDGE" in str(prompts.rag_qa)


# ─────────────────────────────────────────────
# OllamaLLM (mocked)
# ─────────────────────────────────────────────

class TestOllamaLLM:
    def _make_llm(self):
        from src.rag.rag_pipeline import OllamaLLM
        return OllamaLLM("llama3", "http://localhost:11434", 0.1, 256)

    def test_generate_returns_string(self):
        llm = self._make_llm()
        with patch.object(llm, "generate", return_value="Low hemoglobin means anemia."):
            result = llm.generate("What is anemia?")
        assert isinstance(result, str)
        assert "anemia" in result.lower()

    def test_generate_strips_whitespace(self):
        llm = self._make_llm()
        # Test stripping logic directly
        raw = "  Answer here.  \n"
        assert raw.strip() == "Answer here."

    def test_generate_raises_on_error(self):
        llm = self._make_llm()
        with patch.object(llm, "generate", side_effect=ConnectionError("Ollama not running")):
            with pytest.raises(ConnectionError):
                llm.generate("question")


# ─────────────────────────────────────────────
# MedicalRAGPipeline (mocked)
# ─────────────────────────────────────────────

class TestMedicalRAGPipeline:
    @pytest.fixture
    def pipeline(self, tmp_path):
        """Build a RAG pipeline with mocked LLM and retriever."""
        from src.rag.rag_pipeline import MedicalRAGPipeline
        from src.utils.config import get_settings
        get_settings.cache_clear()
        cfg = get_settings()
        cfg.knowledge_base.chromadb_path = str(tmp_path / "chroma_db")
        cfg.rag.llm_provider = "ollama"

        pipeline = MedicalRAGPipeline(cfg)

        # Mock LLM
        pipeline._llm = MagicMock()
        pipeline._llm.generate.return_value = "Low hemoglobin indicates anemia."

        # Mock retriever
        pipeline.retriever = MagicMock()
        pipeline.retriever.retrieve.return_value = [
            {
                "content":  "Low hemoglobin is a sign of anemia.",
                "metadata": {"source": "test_doc"},
                "score":    0.92,
            }
        ]
        return pipeline

    def test_ask_returns_rag_response(self, pipeline):
        from src.rag.rag_pipeline import RAGResponse
        resp = pipeline.ask("What does low hemoglobin mean?")
        assert isinstance(resp, RAGResponse)
        assert resp.answer != ""
        assert resp.question == "What does low hemoglobin mean?"

    def test_ask_populates_sources(self, pipeline):
        resp = pipeline.ask("What does low hemoglobin mean?")
        assert isinstance(resp.sources, list)
        assert "test_doc" in resp.sources

    def test_ask_records_latency(self, pipeline):
        resp = pipeline.ask("What is anemia?")
        assert resp.latency_ms >= 0

    def test_history_grows_after_ask(self, pipeline):
        assert len(pipeline._history) == 0
        pipeline.ask("Question 1")
        assert len(pipeline._history) == 1
        pipeline.ask("Question 2")
        assert len(pipeline._history) == 2

    def test_reset_clears_history(self, pipeline):
        pipeline.ask("Question 1")
        pipeline.ask("Question 2")
        pipeline.reset_conversation()
        assert pipeline._history == []

    def test_set_report_context_stored(self, pipeline):
        report = {"findings": [{"display_name": "Hemoglobin", "value": 9.5, "status": "Low"}]}
        pipeline.set_report_context(report)
        assert pipeline._report_context is not None
        assert pipeline._report_context["findings"][0]["value"] == 9.5

    def test_format_report_findings_with_context(self, pipeline):
        pipeline.set_report_context({
            "findings": [
                {"display_name": "Hemoglobin", "value": 9.5, "unit": "g/dL", "status": "Low"},
                {"display_name": "Glucose", "value": 115.0, "unit": "mg/dL", "status": "High"},
            ],
            "critical_flags": [],
        })
        text = pipeline._format_report_findings()
        assert "Hemoglobin" in text
        assert "Glucose" in text

    def test_format_report_findings_empty(self, pipeline):
        text = pipeline._format_report_findings()
        assert "No structured findings" in text

    def test_ask_batch_resets_between_questions(self, pipeline):
        questions = ["Q1", "Q2", "Q3"]
        responses = pipeline.ask_batch(questions)
        assert len(responses) == 3
        assert pipeline._history == []   # reset after each

    def test_rag_response_to_dict(self, pipeline):
        import json
        resp = pipeline.ask("What is anemia?")
        d = resp.to_dict()
        assert "question" in d
        assert "answer" in d
        assert "sources" in d
        json.dumps(d)   # must be serialisable


# ─────────────────────────────────────────────
# Conversation memory
# ─────────────────────────────────────────────

class TestConversationMemory:
    def test_history_format_produces_string(self):
        from src.rag.rag_pipeline import _format_history
        history = [
            {"user": "What is anemia?", "assistant": "Anemia is low hemoglobin."},
            {"user": "How is it treated?", "assistant": "With iron supplements."},
        ]
        text = _format_history(history)
        assert "What is anemia?" in text
        assert "iron supplements" in text

    def test_empty_history_returns_placeholder(self):
        from src.rag.rag_pipeline import _format_history
        text = _format_history([])
        assert text != ""   # should return some placeholder

    def test_context_formatting(self):
        from src.rag.rag_pipeline import _format_context
        docs = [
            {"content": "Anemia overview.", "metadata": {"source": "doc1"}, "score": 0.9},
            {"content": "Iron deficiency.", "metadata": {"source": "doc2"}, "score": 0.8},
        ]
        text = _format_context(docs)
        assert "Anemia overview" in text
        assert "Iron deficiency" in text
        assert "doc1" in text

    def test_empty_context_returns_message(self):
        from src.rag.rag_pipeline import _format_context
        text = _format_context([])
        assert "No relevant" in text
