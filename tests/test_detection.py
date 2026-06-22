"""
tests/test_detection.py
Focused unit tests for the abnormality detection engine.
Covers edge cases, boundary values, gender-awareness, and serialisation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─────────────────────────────────────────────
# Status enum
# ─────────────────────────────────────────────

class TestStatus:
    def test_status_values(self):
        from src.detection.abnormality_detector import Status
        assert Status.NORMAL.value == "Normal"
        assert Status.LOW.value == "Low"
        assert Status.HIGH.value == "High"
        assert Status.CRITICAL_LOW.value == "Critical Low"
        assert Status.CRITICAL_HIGH.value == "Critical High"
        assert Status.UNKNOWN.value == "Unknown"

    def test_is_abnormal_true_for_low(self):
        from src.detection.abnormality_detector import LabFinding, Status
        f = LabFinding(
            test_name="hgb", display_name="Hemoglobin",
            value=9.5, unit="g/dL", status=Status.LOW,
            reference_low=12.0, reference_high=15.5,
            critical_low=7.0, critical_high=20.0,
        )
        assert f.is_abnormal is True
        assert f.is_critical is False

    def test_is_critical_true(self):
        from src.detection.abnormality_detector import LabFinding, Status
        f = LabFinding(
            test_name="hgb", display_name="Hemoglobin",
            value=5.0, unit="g/dL", status=Status.CRITICAL_LOW,
            reference_low=12.0, reference_high=15.5,
            critical_low=7.0, critical_high=20.0,
        )
        assert f.is_critical is True
        assert f.is_abnormal is True

    def test_normal_is_not_abnormal(self):
        from src.detection.abnormality_detector import LabFinding, Status
        f = LabFinding(
            test_name="tsh", display_name="TSH",
            value=3.0, unit="mIU/L", status=Status.NORMAL,
            reference_low=0.4, reference_high=4.0,
            critical_low=0.01, critical_high=100.0,
        )
        assert f.is_abnormal is False
        assert f.is_critical is False


# ─────────────────────────────────────────────
# Classifier logic
# ─────────────────────────────────────────────

class TestClassifier:
    def setup_method(self):
        from src.detection.abnormality_detector import _classify_value
        self.classify = _classify_value

    def test_normal_value(self):
        status, _ = self.classify(13.0, low=12.0, high=15.5, critical_low=7.0, critical_high=20.0)
        from src.detection.abnormality_detector import Status
        assert status == Status.NORMAL

    def test_low_value(self):
        status, _ = self.classify(10.0, low=12.0, high=15.5, critical_low=7.0, critical_high=20.0)
        from src.detection.abnormality_detector import Status
        assert status == Status.LOW

    def test_high_value(self):
        status, _ = self.classify(17.0, low=12.0, high=15.5, critical_low=7.0, critical_high=20.0)
        from src.detection.abnormality_detector import Status
        assert status == Status.HIGH

    def test_critical_low(self):
        status, _ = self.classify(5.0, low=12.0, high=15.5, critical_low=7.0, critical_high=20.0)
        from src.detection.abnormality_detector import Status
        assert status == Status.CRITICAL_LOW

    def test_critical_high(self):
        status, _ = self.classify(25.0, low=12.0, high=15.5, critical_low=7.0, critical_high=20.0)
        from src.detection.abnormality_detector import Status
        assert status == Status.CRITICAL_HIGH

    def test_exactly_at_boundary_low(self):
        # Value == low boundary should be NORMAL (not LOW)
        status, _ = self.classify(12.0, low=12.0, high=15.5, critical_low=7.0, critical_high=20.0)
        from src.detection.abnormality_detector import Status
        assert status == Status.NORMAL

    def test_exactly_at_boundary_high(self):
        status, _ = self.classify(15.5, low=12.0, high=15.5, critical_low=7.0, critical_high=20.0)
        from src.detection.abnormality_detector import Status
        assert status == Status.NORMAL

    def test_no_critical_bounds(self):
        status, _ = self.classify(5.0, low=12.0, high=15.5, critical_low=None, critical_high=None)
        from src.detection.abnormality_detector import Status
        assert status == Status.LOW

    def test_deviation_computed_for_low(self):
        _, dev = self.classify(9.0, low=12.0, high=15.5, critical_low=7.0, critical_high=20.0)
        assert dev is not None
        assert dev > 0


# ─────────────────────────────────────────────
# ReferenceRangeDB
# ─────────────────────────────────────────────

class TestReferenceRangeDB:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.detection.abnormality_detector import ReferenceRangeDB
        self.db = ReferenceRangeDB("data/reference_ranges.json")

    def test_lookup_hemoglobin_male(self):
        r = self.db.lookup("hemoglobin", gender="male")
        assert r is not None
        assert r["low"] == pytest.approx(13.5)
        assert r["high"] == pytest.approx(17.5)

    def test_lookup_hemoglobin_female(self):
        r = self.db.lookup("hemoglobin", gender="female")
        assert r is not None
        assert r["low"] == pytest.approx(12.0)

    def test_lookup_by_alias_hgb(self):
        r = self.db.lookup("hgb")
        assert r is not None
        assert r["key"] == "hemoglobin"

    def test_lookup_by_alias_sgpt(self):
        r = self.db.lookup("sgpt")
        assert r is not None
        assert r["key"] == "alt"

    def test_lookup_nonexistent_returns_none(self):
        assert self.db.lookup("totally_unknown_test_xyz") is None

    def test_lookup_case_insensitive(self):
        assert self.db.lookup("HEMOGLOBIN") is not None
        assert self.db.lookup("Hemoglobin") is not None
        assert self.db.lookup("hemoglobin") is not None

    def test_lookup_default_gender_fallback(self):
        # WBC has only "default" range
        r = self.db.lookup("wbc", gender="male")
        assert r is not None
        assert r["low"] is not None

    def test_display_name_present(self):
        r = self.db.lookup("tsh")
        assert "display_name" in r
        assert r["display_name"] != ""

    def test_critical_bounds_present(self):
        r = self.db.lookup("glucose")
        assert r["critical_low"] is not None
        assert r["critical_high"] is not None


# ─────────────────────────────────────────────
# AbnormalityDetector
# ─────────────────────────────────────────────

class TestAbnormalityDetectorExtended:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.detection.abnormality_detector import AbnormalityDetector
        self.detector = AbnormalityDetector()

    def test_empty_lab_values(self):
        report = self.detector.detect([])
        assert report.findings == []
        assert report.abnormal_count == 0
        assert report.normal_count == 0

    def test_multiple_critical_flags(self):
        labs = [
            {"test_name": "hemoglobin", "value": "5.0", "unit": "g/dL"},
            {"test_name": "glucose",    "value": "510", "unit": "mg/dL"},
        ]
        report = self.detector.detect(labs)
        assert len(report.critical_flags) == 2

    def test_unknown_tests_counted(self):
        labs = [
            {"test_name": "mystery_biomarker_x", "value": "42", "unit": "units"},
        ]
        report = self.detector.detect(labs)
        assert report.unknown_count == 1

    def test_non_numeric_value_counted_as_unknown(self):
        labs = [{"test_name": "hemoglobin", "value": "pending", "unit": "g/dL"}]
        report = self.detector.detect(labs)
        assert report.unknown_count == 1

    def test_gender_affects_hdl_threshold(self):
        # HDL = 45: Low for female (threshold 50), Normal for male (threshold 40)
        labs = [{"test_name": "hdl", "value": "45", "unit": "mg/dL"}]

        report_f = self.detector.detect(labs, gender="female")
        report_m = self.detector.detect(labs, gender="male")

        assert report_f.findings[0].status.value == "Low"
        assert report_m.findings[0].status.value == "Normal"

    def test_summary_reflects_counts(self):
        labs = [
            {"test_name": "hemoglobin", "value": "9.5", "unit": "g/dL"},
            {"test_name": "tsh",        "value": "3.1", "unit": "mIU/L"},
        ]
        report = self.detector.detect(labs, gender="female")
        assert report.summary != ""
        assert str(report.normal_count) in report.summary or str(report.abnormal_count) in report.summary

    def test_finding_interpretation_set(self):
        labs = [{"test_name": "hemoglobin", "value": "9.5", "unit": "g/dL"}]
        report = self.detector.detect(labs, gender="female")
        assert report.findings[0].interpretation != ""

    def test_deviation_pct_set_for_abnormal(self):
        labs = [{"test_name": "hemoglobin", "value": "9.5", "unit": "g/dL"}]
        report = self.detector.detect(labs, gender="female")
        f = report.findings[0]
        assert f.deviation_pct is not None
        assert f.deviation_pct > 0

    def test_to_dict_is_json_serialisable(self):
        labs = [
            {"test_name": "hemoglobin", "value": "9.5",  "unit": "g/dL"},
            {"test_name": "glucose",    "value": "115",  "unit": "mg/dL"},
            {"test_name": "tsh",        "value": "3.1",  "unit": "mIU/L"},
        ]
        report = self.detector.detect(labs)
        json.dumps(report.to_dict())   # must not raise

    def test_all_normal_report(self):
        labs = [
            {"test_name": "tsh",        "value": "2.5",  "unit": "mIU/L"},
            {"test_name": "creatinine", "value": "0.9",  "unit": "mg/dL"},
            {"test_name": "sodium",     "value": "140",  "unit": "mEq/L"},
        ]
        report = self.detector.detect(labs)
        assert report.abnormal_count == 0
        assert len(report.critical_flags) == 0
