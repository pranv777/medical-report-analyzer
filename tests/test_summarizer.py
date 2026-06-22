"""
tests/test_summarizer.py
Unit tests for MedicalSummarizer and report formatter.
All LLM calls are mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLE_DETECTION = {
    "findings": [
        {
            "test_name":    "hemoglobin",
            "display_name": "Hemoglobin",
            "value":        9.5,
            "unit":         "g/dL",
            "status":       "Low",
            "reference_low":  12.0,
            "reference_high": 15.5,
            "critical_low":   7.0,
            "critical_high":  20.0,
            "deviation_pct":  20.8,
            "interpretation": "May indicate anemia.",
        },
        {
            "test_name":    "glucose",
            "display_name": "Blood Glucose",
            "value":        118.0,
            "unit":         "mg/dL",
            "status":       "High",
            "reference_low":  70.0,
            "reference_high": 100.0,
            "critical_low":   40.0,
            "critical_high":  500.0,
            "deviation_pct":  18.0,
            "interpretation": "Elevated fasting glucose.",
        },
        {
            "test_name":    "tsh",
            "display_name": "TSH",
            "value":        3.1,
            "unit":         "mIU/L",
            "status":       "Normal",
            "reference_low":  0.4,
            "reference_high": 4.0,
            "critical_low":   0.01,
            "critical_high":  100.0,
            "deviation_pct":  0.0,
            "interpretation": "Thyroid function appears normal.",
        },
    ],
    "critical_flags": [],
    "abnormal_count": 2,
    "normal_count":   1,
    "unknown_count":  0,
    "summary": "Analysed 3 results. 2 abnormal. 1 normal.",
}


# ─────────────────────────────────────────────
# Static formatting helpers
# ─────────────────────────────────────────────

class TestSummarizerFormatters:
    def test_format_findings_marks_abnormals(self):
        from src.summarization.summarizer import MedicalSummarizer
        text = MedicalSummarizer._format_findings(SAMPLE_DETECTION)
        assert "Hemoglobin" in text
        assert "Blood Glucose" in text
        assert "Low" in text or "High" in text

    def test_format_findings_none_returns_placeholder(self):
        from src.summarization.summarizer import MedicalSummarizer
        text = MedicalSummarizer._format_findings(None)
        assert "No structured findings" in text

    def test_extract_key_findings_excludes_normal(self):
        from src.summarization.summarizer import MedicalSummarizer
        findings = MedicalSummarizer._extract_key_findings(SAMPLE_DETECTION)
        labels = " ".join(findings)
        assert "Hemoglobin" in labels
        assert "Blood Glucose" in labels
        assert "TSH" not in labels   # TSH is Normal

    def test_extract_key_findings_empty_report(self):
        from src.summarization.summarizer import MedicalSummarizer
        assert MedicalSummarizer._extract_key_findings(None) == []
        assert MedicalSummarizer._extract_key_findings({}) == []

    def test_format_entities_with_result(self):
        from src.summarization.summarizer import MedicalSummarizer
        from src.extraction.ner_extractor import ExtractionResult, MedicalEntity
        ner = ExtractionResult(raw_text="test")
        ner.entity_summary = {
            "DISEASE":    ["anemia", "diabetes"],
            "LAB_TEST":   ["hemoglobin", "glucose"],
            "MEDICATION": ["metformin"],
        }
        text = MedicalSummarizer._format_entities(ner)
        assert "anemia" in text
        assert "hemoglobin" in text

    def test_format_entities_none(self):
        from src.summarization.summarizer import MedicalSummarizer
        text = MedicalSummarizer._format_entities(None)
        assert "No entities" in text


# ─────────────────────────────────────────────
# MedicalSummarizer (mocked LLM)
# ─────────────────────────────────────────────

class TestMedicalSummarizer:
    @pytest.fixture
    def summarizer(self):
        from src.summarization.summarizer import MedicalSummarizer
        from src.utils.config import get_settings
        get_settings.cache_clear()
        cfg = get_settings()
        s = MedicalSummarizer(cfg)
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "This is a generated summary."
        s._llm = mock_llm
        return s

    def test_summarize_returns_summary_result(self, summarizer):
        from src.summarization.summarizer import SummaryResult
        result = summarizer.summarize(detection_report=SAMPLE_DETECTION)
        assert isinstance(result, SummaryResult)

    def test_clinical_summary_populated(self, summarizer):
        result = summarizer.summarize(detection_report=SAMPLE_DETECTION)
        assert result.clinical_summary != ""

    def test_patient_summary_populated(self, summarizer):
        result = summarizer.summarize(detection_report=SAMPLE_DETECTION)
        assert result.patient_summary != ""

    def test_recommendations_populated(self, summarizer):
        summarizer._llm.generate.return_value = (
            "1. Repeat CBC in 3 months.\n"
            "2. Start iron supplementation.\n"
            "3. Follow up fasting glucose."
        )
        result = summarizer.summarize(detection_report=SAMPLE_DETECTION)
        assert len(result.recommendations) >= 1

    def test_follow_up_questions_populated(self, summarizer):
        summarizer._llm.generate.return_value = (
            "1. Should I repeat this test?\n"
            "2. Could diet be causing this?\n"
            "3. Do I need medication?"
        )
        result = summarizer.summarize(detection_report=SAMPLE_DETECTION)
        assert len(result.follow_up_questions) >= 1

    def test_key_findings_only_abnormals(self, summarizer):
        result = summarizer.summarize(detection_report=SAMPLE_DETECTION)
        combined = " ".join(result.key_findings)
        assert "Hemoglobin" in combined
        assert "Blood Glucose" in combined
        assert "TSH" not in combined

    def test_summary_to_dict_serialisable(self, summarizer):
        import json
        result = summarizer.summarize(detection_report=SAMPLE_DETECTION)
        json.dumps(result.to_dict())   # must not raise

    def test_summarize_without_detection_report(self, summarizer):
        result = summarizer.summarize()
        assert isinstance(result.clinical_summary, str)

    def test_generate_list_parses_numbered_lines(self, summarizer):
        summarizer._llm.generate.return_value = (
            "1. First item\n2. Second item\n3. Third item"
        )
        items = summarizer._generate_list("some prompt")
        assert len(items) == 3
        assert items[0] == "First item"
        assert items[2] == "Third item"

    def test_generate_list_strips_prefixes(self, summarizer):
        summarizer._llm.generate.return_value = (
            "- Item A\n- Item B\n• Item C"
        )
        items = summarizer._generate_list("prompt")
        assert all(not i.startswith("-") for i in items)
        assert all(not i.startswith("•") for i in items)
