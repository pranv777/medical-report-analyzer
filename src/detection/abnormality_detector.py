"""
src/detection/abnormality_detector.py
Classifies extracted lab values as Normal / Low / High / Critical Low / Critical High
by comparing them against a reference-range database.

Also produces a structured findings report with clinical interpretation hints.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.config import get_settings
from src.utils.helpers import extract_numeric, normalize_unit
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
# Enums & data models
# ─────────────────────────────────────────────

class Status(str, Enum):
    CRITICAL_LOW  = "Critical Low"
    LOW           = "Low"
    NORMAL        = "Normal"
    HIGH          = "High"
    CRITICAL_HIGH = "Critical High"
    UNKNOWN       = "Unknown"


@dataclass
class LabFinding:
    test_name: str
    display_name: str
    value: float
    unit: str
    status: Status
    reference_low: Optional[float]
    reference_high: Optional[float]
    critical_low: Optional[float]
    critical_high: Optional[float]
    deviation_pct: Optional[float] = None   # % above/below range boundary
    interpretation: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @property
    def is_abnormal(self) -> bool:
        return self.status != Status.NORMAL and self.status != Status.UNKNOWN

    @property
    def is_critical(self) -> bool:
        return self.status in (Status.CRITICAL_LOW, Status.CRITICAL_HIGH)


@dataclass
class DetectionReport:
    findings: List[LabFinding] = field(default_factory=list)
    critical_flags: List[str]  = field(default_factory=list)
    abnormal_count: int = 0
    normal_count: int   = 0
    unknown_count: int  = 0
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "critical_flags": self.critical_flags,
            "abnormal_count": self.abnormal_count,
            "normal_count": self.normal_count,
            "unknown_count": self.unknown_count,
            "summary": self.summary,
        }


# ─────────────────────────────────────────────
# Reference range loader
# ─────────────────────────────────────────────

class ReferenceRangeDB:
    """Loads reference_ranges.json and provides fast lookup by test name / alias."""

    def __init__(self, db_path: str):
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference range DB not found: {path}")
        with open(path) as f:
            self._db: Dict = json.load(f)

        # Build alias → canonical key index
        self._alias_index: Dict[str, str] = {}
        for key, entry in self._db.items():
            self._alias_index[key.lower()] = key
            for alias in entry.get("aliases", []):
                self._alias_index[alias.lower()] = key

    def lookup(
        self, test_name: str, gender: str = "default", age: int = 30
    ) -> Optional[Dict]:
        """
        Return reference range dict for a test name, or None if not found.
        gender: 'male' | 'female' | 'child' | 'default'
        """
        canon = self._alias_index.get(test_name.lower().strip())
        if canon is None:
            # Fuzzy: require the query to be at least 5 chars AND a substring match
            # in both directions to avoid spurious hits on short tokens like "xyz"
            tl = test_name.lower().strip()
            if len(tl) >= 5:
                for alias, key in self._alias_index.items():
                    if len(alias) >= 5 and (tl in alias and len(tl) / len(alias) > 0.5):
                        canon = key
                        break

        if canon is None:
            return None

        entry = self._db[canon]
        ranges = entry.get("ranges", {})

        # Pick the most specific range tier
        gender_key = gender if gender in ranges else "default"
        if gender_key not in ranges:
            gender_key = next(iter(ranges))  # fallback to first available

        range_data = ranges[gender_key]
        return {
            "key": canon,
            "display_name": entry.get("display_name", canon),
            "unit": entry.get("unit", ""),
            "low": range_data.get("low"),
            "high": range_data.get("high"),
            "critical_low": range_data.get("critical_low"),
            "critical_high": range_data.get("critical_high"),
        }


# ─────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────

def _classify_value(
    value: float,
    low: Optional[float],
    high: Optional[float],
    critical_low: Optional[float],
    critical_high: Optional[float],
) -> Tuple[Status, Optional[float]]:
    """
    Returns (Status, deviation_pct).
    deviation_pct is the percentage outside the normal boundary (positive = above high).
    """
    if critical_low is not None and value <= critical_low:
        dev = ((critical_low - value) / critical_low * 100) if critical_low else None
        return Status.CRITICAL_LOW, dev

    if critical_high is not None and value >= critical_high:
        dev = ((value - critical_high) / critical_high * 100) if critical_high else None
        return Status.CRITICAL_HIGH, dev

    if low is not None and value < low:
        dev = ((low - value) / low * 100) if low else None
        return Status.LOW, dev

    if high is not None and value > high:
        dev = ((value - high) / high * 100) if high else None
        return Status.HIGH, dev

    return Status.NORMAL, 0.0


# ─────────────────────────────────────────────
# Interpretation hints
# ─────────────────────────────────────────────

_INTERPRETATIONS: Dict[str, Dict[str, str]] = {
    "hemoglobin": {
        "Low":          "May indicate anemia; causes include iron deficiency, B12/folate deficiency, or chronic disease.",
        "Critical Low": "Severely low hemoglobin — urgent clinical evaluation required.",
        "High":         "Elevated hemoglobin can indicate dehydration, polycythemia, or living at high altitude.",
        "Normal":       "Hemoglobin within normal range.",
    },
    "glucose": {
        "Low":          "Hypoglycemia — may cause dizziness, sweating, confusion.",
        "Critical Low": "Severe hypoglycemia — requires immediate treatment.",
        "High":         "Elevated fasting glucose — may indicate pre-diabetes or diabetes. Repeat testing advised.",
        "Critical High":"Dangerously high glucose — risk of diabetic ketoacidosis.",
        "Normal":       "Fasting glucose within normal range.",
    },
    "total_cholesterol": {
        "High":         "Elevated cholesterol increases cardiovascular risk. Lifestyle modification and further lipid workup recommended.",
        "Normal":       "Total cholesterol within desirable range.",
    },
    "ldl": {
        "High":         "High LDL ('bad cholesterol') raises risk of atherosclerosis and heart disease.",
        "Normal":       "LDL within optimal range.",
    },
    "tsh": {
        "Low":          "Low TSH may indicate hyperthyroidism. Free T4/T3 testing recommended.",
        "High":         "Elevated TSH may indicate hypothyroidism. Free T4 testing recommended.",
        "Normal":       "Thyroid function appears normal.",
    },
    "creatinine": {
        "High":         "Elevated creatinine may indicate impaired kidney function. GFR calculation recommended.",
        "Normal":       "Kidney function marker within normal range.",
    },
    "vitamin_d": {
        "Low":          "Vitamin D deficiency — associated with bone loss, immune dysfunction, and fatigue.",
        "Critical Low": "Severe vitamin D deficiency requiring prompt supplementation.",
        "Normal":       "Vitamin D level is sufficient.",
    },
}

def _get_interpretation(key: str, status: Status) -> str:
    hints = _INTERPRETATIONS.get(key, {})
    return hints.get(status.value, f"{status.value} result for {key}.")


# ─────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────

class AbnormalityDetector:
    """
    Takes a list of raw lab values (from NER/regex extraction) and produces
    a structured DetectionReport with per-test status and interpretation.

    Usage::

        detector = AbnormalityDetector()
        report = detector.detect(lab_values, gender="female", age=35)
        print(report.to_dict())
    """

    def __init__(self, config=None):
        self.cfg = config or get_settings().detection
        self.db  = ReferenceRangeDB(self.cfg.reference_db_path)

    def detect(
        self,
        lab_values: List[Dict],
        gender: str = "default",
        age: int = 30,
    ) -> DetectionReport:
        report = DetectionReport()

        for lv in lab_values:
            test_name = lv.get("test_name", "")
            raw_value = lv.get("value", "")

            numeric = extract_numeric(str(raw_value))
            if numeric is None:
                log.debug(f"Skipping non-numeric value for '{test_name}': {raw_value}")
                report.unknown_count += 1
                continue

            ref = self.db.lookup(test_name, gender=gender, age=age)
            if ref is None:
                log.debug(f"No reference range found for: '{test_name}'")
                report.unknown_count += 1
                # Still record the finding as UNKNOWN
                finding = LabFinding(
                    test_name=test_name,
                    display_name=test_name,
                    value=numeric,
                    unit=lv.get("unit", ""),
                    status=Status.UNKNOWN,
                    reference_low=None,
                    reference_high=None,
                    critical_low=None,
                    critical_high=None,
                    interpretation="No reference range available for this test.",
                )
                report.findings.append(finding)
                continue

            status, deviation = _classify_value(
                numeric,
                ref["low"],
                ref["high"],
                ref["critical_low"],
                ref["critical_high"],
            )

            interp = _get_interpretation(ref["key"], status)

            finding = LabFinding(
                test_name=test_name,
                display_name=ref["display_name"],
                value=numeric,
                unit=ref["unit"] or lv.get("unit", ""),
                status=status,
                reference_low=ref["low"],
                reference_high=ref["high"],
                critical_low=ref["critical_low"],
                critical_high=ref["critical_high"],
                deviation_pct=round(deviation, 1) if deviation else None,
                interpretation=interp,
            )
            report.findings.append(finding)

            if finding.is_critical:
                report.critical_flags.append(
                    f"{finding.display_name}: {status.value} ({numeric} {finding.unit})"
                )
                report.abnormal_count += 1
            elif finding.is_abnormal:
                report.abnormal_count += 1
            else:
                report.normal_count += 1

        report.summary = self._build_summary(report)
        log.info(
            f"Detection complete: {report.normal_count} normal, "
            f"{report.abnormal_count} abnormal, "
            f"{len(report.critical_flags)} critical"
        )
        return report

    @staticmethod
    def _build_summary(report: DetectionReport) -> str:
        total = report.normal_count + report.abnormal_count + report.unknown_count
        if total == 0:
            return "No lab values could be evaluated."

        parts = [f"Analysed {total} test result(s)."]
        if report.critical_flags:
            parts.append(
                f"⚠️  CRITICAL: {len(report.critical_flags)} result(s) require immediate attention: "
                + "; ".join(report.critical_flags)
            )
        if report.abnormal_count:
            abnormal_names = [
                f.display_name for f in report.findings if f.is_abnormal and not f.is_critical
            ]
            parts.append(f"Abnormal (non-critical): {', '.join(abnormal_names)}.")
        if report.normal_count:
            parts.append(f"{report.normal_count} result(s) within normal range.")
        return " ".join(parts)
