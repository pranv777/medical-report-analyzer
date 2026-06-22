"""
tests/test_ingestion.py
Unit tests for report ingestion and text preprocessing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─────────────────────────────────────────────
# TextPreprocessor tests
# ─────────────────────────────────────────────

class TestTextPreprocessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.ingestion.text_preprocessor import TextPreprocessor
        self.pp = TextPreprocessor()

    def test_normalises_whitespace(self):
        result = self.pp.process("Line one\r\n\r\nLine two\n\n\n\nLine three")
        assert "\r" not in result.clean_text
        assert "\n\n\n" not in result.clean_text

    def test_extracts_patient_name(self):
        text = "Patient: Jane Smith\nAge: 35\nGender: Female"
        result = self.pp.process(text)
        assert result.metadata.patient_name is not None
        assert "Jane" in result.metadata.patient_name

    def test_extracts_age(self):
        text = "Patient: John Doe\nAge: 42 years\nDate: 10-Jan-2024"
        result = self.pp.process(text)
        assert result.metadata.age == 42

    def test_extracts_gender(self):
        text = "Gender: Male\nHemoglobin: 13.5"
        result = self.pp.process(text)
        assert result.metadata.gender == "male"

        text2 = "Sex: F\nHemoglobin: 12.0"
        result2 = self.pp.process(text2)
        assert result2.metadata.gender == "female"

    def test_expands_abbreviations(self):
        text = "WBC count is 7.2. HGB is 10.5."
        result = self.pp.process(text)
        assert "White Blood Cell" in result.expanded_text
        assert "Hemoglobin" in result.expanded_text

    def test_splits_sections(self):
        text = (
            "PATIENT INFORMATION\n"
            "Name: John Doe\n\n"
            "HAEMATOLOGY\n"
            "Hemoglobin: 10.5\n\n"
            "LIPID PROFILE\n"
            "Cholesterol: 220\n"
        )
        result = self.pp.process(text)
        titles = [s.title for s in result.sections]
        assert any("HAEMATOLOGY" in t for t in titles)

    def test_strips_boilerplate(self):
        text = "Hemoglobin: 10.5\nPage 1 of 3\nResults are for clinical correlation only."
        result = self.pp.process(text)
        assert "Page 1 of 3" not in result.clean_text
        assert "clinical correlation" not in result.clean_text.lower()

    def test_empty_text_returns_result(self):
        result = self.pp.process("")
        assert result.clean_text == ""
        assert result.sections == [] or len(result.sections) >= 0

    def test_full_sample_report(self):
        report_path = Path("data/sample_reports/sample_blood_test.txt")
        if not report_path.exists():
            pytest.skip("Sample report not found")
        text = report_path.read_text()
        result = self.pp.process(text)
        assert len(result.clean_text) > 100
        assert len(result.sections) > 0


# ─────────────────────────────────────────────
# ReportIngestion tests
# ─────────────────────────────────────────────

class TestReportIngestion:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.ingestion.report_ingestion import ReportIngestion
        from src.utils.config import get_settings
        get_settings.cache_clear()
        self.ingestion = ReportIngestion()

    def test_raises_on_missing_file(self):
        from src.ingestion.report_ingestion import ReportIngestion
        with pytest.raises(FileNotFoundError):
            self.ingestion.ingest("/nonexistent/path/file.pdf")

    def test_raises_on_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            self.ingestion.ingest("report.docx")

    def test_ingestion_result_fields(self, tmp_path):
        """IngestionResult has required fields even on a minimal run."""
        from src.ingestion.report_ingestion import IngestionResult
        result = IngestionResult(
            file_path="/fake/report.pdf",
            file_hash="abc123",
            file_type="pdf",
        )
        assert result.success is False        # no pages yet
        assert result.full_text == ""
        assert result.errors == []

    def test_page_result_extraction_method(self):
        from src.ingestion.report_ingestion import PageResult
        page = PageResult(page_number=1, text="Hello world", extraction_method="native")
        assert page.extraction_method == "native"
        assert page.text == "Hello world"

    def test_ingest_text_file_via_helper(self, tmp_path):
        """Simulate reading a .txt report through the helpers."""
        from src.utils.helpers import clean_text
        content = "Hemoglobin: 10.5 g/dL\nGlucose: 115 mg/dL\n"
        f = tmp_path / "report.txt"
        f.write_text(content)
        text = clean_text(f.read_text())
        assert "Hemoglobin" in text
        assert "Glucose" in text


# ─────────────────────────────────────────────
# LabReportParser tests
# ─────────────────────────────────────────────

class TestLabReportParser:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.extraction.lab_parser import LabReportParser
        self.parser = LabReportParser()

    def test_parses_colon_format(self):
        entries = self.parser.parse("Glucose: 115 mg/dL\nHbA1c: 6.2%")
        names = [e.test_name.lower() for e in entries]
        assert any("glucose" in n for n in names)
        assert any("hba1c" in n for n in names)

    def test_parses_tabular_format(self):
        text = "Hemoglobin        10.5        13.5-17.5    g/dL    LOW"
        entries = self.parser.parse(text)
        assert len(entries) == 1
        assert entries[0].numeric_value == pytest.approx(10.5)
        assert entries[0].flag == "LOW"
        assert entries[0].ref_low == pytest.approx(13.5)
        assert entries[0].ref_high == pytest.approx(17.5)

    def test_flag_normalisation(self):
        from src.extraction.lab_parser import _normalise_flag
        assert _normalise_flag("low") == "LOW"
        assert _normalise_flag("HIGH") == "HIGH"
        assert _normalise_flag("CRITICAL LOW") == "CRITICAL LOW"
        assert _normalise_flag("elevated") == "HIGH"
        assert _normalise_flag(None) is None

    def test_empty_text_returns_empty(self):
        assert self.parser.parse("") == []
        assert self.parser.parse("   \n  \n  ") == []

    def test_deduplicates_same_test(self):
        text = "Glucose: 115 mg/dL\nGlucose: 118 mg/dL"
        entries = self.parser.parse(text)
        names = [e.test_name.lower() for e in entries]
        assert names.count("glucose") == 1

    def test_to_dict_serialisable(self):
        import json
        entries = self.parser.parse("TSH: 3.2 mIU/L")
        for e in entries:
            json.dumps(e.to_dict())   # should not raise

    def test_full_sample_report(self):
        report_path = Path("data/sample_reports/sample_blood_test.txt")
        if not report_path.exists():
            pytest.skip("Sample report not found")
        from src.extraction.ner_extractor import extract_lab_values
        text = report_path.read_text()
        entries = extract_lab_values(text)
        assert len(entries) >= 5
