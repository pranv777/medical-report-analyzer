"""
tests/test_pipeline.py
Unit and integration tests for all pipeline modules.

Run with:
    pytest tests/ -v --tb=short
    pytest tests/ -v -k "detection"     # run specific group
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

SAMPLE_TEXT = """
Patient: Jane Smith, Age: 45, Female

Haemoglobin: 9.5 g/dL  [Low]
WBC: 7.8 10^3/µL        [Normal]
Fasting Glucose: 115 mg/dL [High]
HbA1c: 6.1%             [High]
Total Cholesterol: 225 mg/dL [High]
LDL: 145 mg/dL          [High]
HDL: 48 mg/dL           [Normal]
Serum Ferritin: 6 ng/mL [Low]
Vitamin D: 16 ng/mL     [Low]
TSH: 3.1 mIU/L          [Normal]
Creatinine: 0.88 mg/dL  [Normal]
"""

LAB_VALUES = [
    {"test_name": "hemoglobin", "value": "9.5", "unit": "g/dL"},
    {"test_name": "glucose", "value": "115", "unit": "mg/dL"},
    {"test_name": "hba1c", "value": "6.1", "unit": "%"},
    {"test_name": "total_cholesterol", "value": "225", "unit": "mg/dL"},
    {"test_name": "ldl", "value": "145", "unit": "mg/dL"},
    {"test_name": "ferritin", "value": "6", "unit": "ng/mL"},
    {"test_name": "vitamin_d", "value": "16", "unit": "ng/mL"},
    {"test_name": "tsh", "value": "3.1", "unit": "mIU/L"},
    {"test_name": "creatinine", "value": "0.88", "unit": "mg/dL"},
]


# ─────────────────────────────────────────────
# Utils tests
# ─────────────────────────────────────────────

class TestHelpers:
    def test_clean_text_removes_control_chars(self):
        from src.utils.helpers import clean_text
        result = clean_text("Hello\x00World\r\n  test  ")
        assert "\x00" not in result
        assert result.strip() == result

    def test_extract_numeric_from_string(self):
        from src.utils.helpers import extract_numeric
        assert extract_numeric("10.5 g/dL") == 10.5
        assert extract_numeric("145") == 145.0
        assert extract_numeric("N/A") is None

    def test_flatten_entities_deduplicates(self):
        from src.utils.helpers import flatten_entities
        entities = [
            {"text": "anemia", "label": "DISEASE", "score": 0.85},
            {"text": "anemia", "label": "DISEASE", "score": 0.92},
            {"text": "ferritin", "label": "LAB_TEST", "score": 0.78},
        ]
        flat = flatten_entities(entities)
        assert len(flat) == 2
        # Should keep the higher-score duplicate
        anemia_entry = next(e for e in flat if e["text"] == "anemia")
        assert anemia_entry["score"] == 0.92

    def test_save_and_load_json(self, tmp_path):
        from src.utils.helpers import save_json, load_json
        data = {"key": "value", "number": 42}
        path = tmp_path / "test.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data


# ─────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────

class TestConfig:
    def test_default_settings_loads(self):
        from src.utils.config import get_settings
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.project.name != ""
        assert settings.ingestion.fallback_to_ocr is True
        assert settings.extraction.confidence_threshold > 0

    def test_settings_have_expected_fields(self):
        from src.utils.config import get_settings
        get_settings.cache_clear()
        s = get_settings()
        assert hasattr(s, "rag")
        assert hasattr(s, "knowledge_base")
        assert hasattr(s, "evaluation")


# ─────────────────────────────────────────────
# Abnormality detector tests
# ─────────────────────────────────────────────

class TestAbnormalityDetector:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.detection.abnormality_detector import AbnormalityDetector
        from src.utils.config import get_settings
        get_settings.cache_clear()
        self.detector = AbnormalityDetector()

    def test_low_hemoglobin_detected(self):
        report = self.detector.detect(
            [{"test_name": "hemoglobin", "value": "9.5", "unit": "g/dL"}],
            gender="female",
        )
        finding = report.findings[0]
        assert finding.status.value in ("Low", "Critical Low")
        assert finding.is_abnormal

    def test_normal_creatinine(self):
        report = self.detector.detect(
            [{"test_name": "creatinine", "value": "0.88", "unit": "mg/dL"}],
            gender="female",
        )
        finding = report.findings[0]
        assert finding.status.value == "Normal"
        assert not finding.is_abnormal

    def test_high_cholesterol(self):
        report = self.detector.detect(
            [{"test_name": "total_cholesterol", "value": "245", "unit": "mg/dL"}]
        )
        finding = report.findings[0]
        assert finding.status.value == "High"

    def test_critical_glucose(self):
        report = self.detector.detect(
            [{"test_name": "glucose", "value": "510", "unit": "mg/dL"}]
        )
        finding = report.findings[0]
        assert finding.status.value == "Critical High"
        assert finding.is_critical
        assert len(report.critical_flags) == 1

    def test_unknown_test_handled(self):
        report = self.detector.detect(
            [{"test_name": "some_unknown_test_xyz", "value": "42", "unit": "units"}]
        )
        assert report.unknown_count == 1

    def test_non_numeric_skipped(self):
        report = self.detector.detect(
            [{"test_name": "hemoglobin", "value": "not a number", "unit": "g/dL"}]
        )
        assert report.unknown_count == 1
        assert not report.findings  # no valid findings

    def test_full_sample_lab_values(self):
        report = self.detector.detect(LAB_VALUES, gender="female", age=45)
        assert report.abnormal_count > 0
        assert report.normal_count > 0
        assert report.summary != ""

    def test_detection_report_serialisable(self):
        report = self.detector.detect(LAB_VALUES)
        data = report.to_dict()
        json_str = json.dumps(data)  # should not raise
        assert "findings" in json_str

    def test_gender_aware_hdl_range(self):
        # Female HDL normal >= 50, Male >= 40
        report_female = self.detector.detect(
            [{"test_name": "hdl", "value": "45", "unit": "mg/dL"}], gender="female"
        )
        report_male = self.detector.detect(
            [{"test_name": "hdl", "value": "45", "unit": "mg/dL"}], gender="male"
        )
        # 45 should be Low for female but Normal for male
        assert report_female.findings[0].status.value == "Low"
        assert report_male.findings[0].status.value == "Normal"


# ─────────────────────────────────────────────
# Reference DB tests
# ─────────────────────────────────────────────

class TestReferenceRangeDB:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.detection.abnormality_detector import ReferenceRangeDB
        self.db = ReferenceRangeDB("data/reference_ranges.json")

    def test_lookup_by_canonical_key(self):
        result = self.db.lookup("hemoglobin", gender="male")
        assert result is not None
        assert result["low"] == 13.5

    def test_lookup_by_alias(self):
        result = self.db.lookup("hgb", gender="male")
        assert result is not None
        assert result["key"] == "hemoglobin"

    def test_lookup_nonexistent_returns_none(self):
        result = self.db.lookup("some_fake_test_that_doesnt_exist")
        assert result is None

    def test_fallback_to_default_gender(self):
        result = self.db.lookup("wbc", gender="nonbinary")
        assert result is not None  # Should fall back to "default"


# ─────────────────────────────────────────────
# Lab value regex extractor tests
# ─────────────────────────────────────────────

class TestLabValueExtractor:
    def test_extracts_from_structured_text(self):
        from src.extraction.ner_extractor import extract_lab_values
        text = "Haemoglobin: 10.5 g/dL\nGlucose: 115 mg/dL\nWBC: 7.2"
        results = extract_lab_values(text)
        assert len(results) >= 2
        values = [r["value"] for r in results]
        assert "10.5" in values or "115" in values

    def test_handles_colon_separator(self):
        from src.extraction.ner_extractor import extract_lab_values
        text = "TSH: 3.2 mIU/L"
        results = extract_lab_values(text)
        assert len(results) >= 1
        assert results[0]["value"] == "3.2"

    def test_empty_text_returns_empty(self):
        from src.extraction.ner_extractor import extract_lab_values
        assert extract_lab_values("") == []
        assert extract_lab_values("No numbers here at all.") == []


# ─────────────────────────────────────────────
# Ingestion tests (mocked)
# ─────────────────────────────────────────────

class TestReportIngestion:
    def test_unsupported_format_raises(self):
        from src.ingestion.report_ingestion import ReportIngestion
        ingestion = ReportIngestion()
        with pytest.raises(ValueError, match="Unsupported format"):
            ingestion.ingest("report.docx")

    def test_missing_file_raises(self):
        from src.ingestion.report_ingestion import ReportIngestion
        ingestion = ReportIngestion()
        with pytest.raises(FileNotFoundError):
            ingestion.ingest("/nonexistent/path/report.pdf")

    def test_ingest_text_file(self, tmp_path):
        """Test ingestion of a plain-text report saved as .txt (read directly)."""
        from src.utils.helpers import clean_text
        txt = tmp_path / "report.txt"
        txt.write_text(SAMPLE_TEXT)
        content = clean_text(txt.read_text())
        assert "Haemoglobin" in content or "hemoglobin" in content.lower()


# ─────────────────────────────────────────────
# Evaluation tests
# ─────────────────────────────────────────────

class TestNEREvaluator:
    def test_perfect_predictions(self):
        from src.evaluation.evaluator import NEREvaluator
        evaluator = NEREvaluator()
        preds = [[{"text": "anemia", "label": "DISEASE"}]]
        golds = [[{"text": "anemia", "label": "DISEASE"}]]
        metrics = evaluator.evaluate(preds, golds)
        assert len(metrics) == 1
        assert metrics[0].f1 == pytest.approx(1.0)
        assert metrics[0].precision == pytest.approx(1.0)

    def test_no_predictions(self):
        from src.evaluation.evaluator import NEREvaluator
        evaluator = NEREvaluator()
        preds = [[]]
        golds = [[{"text": "anemia", "label": "DISEASE"}]]
        metrics = evaluator.evaluate(preds, golds)
        disease_m = next((m for m in metrics if m.entity_type == "DISEASE"), None)
        assert disease_m is not None
        assert disease_m.recall == 0.0

    def test_partial_overlap(self):
        from src.evaluation.evaluator import NEREvaluator
        evaluator = NEREvaluator()
        preds = [[
            {"text": "anemia", "label": "DISEASE"},
            {"text": "false_positive", "label": "DISEASE"},
        ]]
        golds = [[
            {"text": "anemia", "label": "DISEASE"},
            {"text": "diabetes", "label": "DISEASE"},
        ]]
        metrics = evaluator.evaluate(preds, golds)
        disease_m = next(m for m in metrics if m.entity_type == "DISEASE")
        assert 0 < disease_m.f1 < 1.0


# ─────────────────────────────────────────────
# Full pipeline smoke test (no LLM needed)
# ─────────────────────────────────────────────

class TestPipelineIntegration:
    def test_detection_only_pipeline(self):
        """Integration test: NER regex → detection (no LLM or BioBERT needed)."""
        from src.extraction.ner_extractor import extract_lab_values
        from src.detection.abnormality_detector import AbnormalityDetector

        lab_values = extract_lab_values(SAMPLE_TEXT)
        assert len(lab_values) > 0

        detector = AbnormalityDetector()
        report   = detector.detect(lab_values, gender="female", age=45)

        assert isinstance(report.findings, list)
        assert isinstance(report.summary, str)
        assert report.summary != ""

        # At least some findings from our rich sample text
        all_statuses = {f.status.value for f in report.findings}
        assert len(all_statuses) > 0

    def test_detection_report_has_correct_types(self):
        from src.detection.abnormality_detector import AbnormalityDetector, Status
        detector = AbnormalityDetector()
        report = detector.detect(LAB_VALUES, gender="female")

        for finding in report.findings:
            assert isinstance(finding.status, Status)
            assert isinstance(finding.value, float)
            assert isinstance(finding.interpretation, str)
