"""
src/extraction/lab_parser.py
Structured parser for tabular lab report lines.

Handles formats:
  Hemoglobin       10.5     13.5-17.5    g/dL    LOW
  Glucose: 115 mg/dL
  HbA1c = 6.2%
  TSH              3.2 mIU/L   [Normal]

Produces a normalised list of LabEntry objects with test name,
numeric value, unit, reference range, and flag.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.utils.helpers import extract_numeric, normalize_unit


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class LabEntry:
    test_name: str
    raw_value: str
    numeric_value: Optional[float]
    unit: str
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    flag: Optional[str] = None          # "LOW" | "HIGH" | "NORMAL" | "CRITICAL" | None
    raw_line: str = ""

    def to_dict(self) -> dict:
        return {
            "test_name":     self.test_name,
            "value":         self.raw_value,
            "numeric_value": self.numeric_value,
            "unit":          self.unit,
            "ref_low":       self.ref_low,
            "ref_high":      self.ref_high,
            "flag":          self.flag,
        }


# ─────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────

# Tabular: "Haemoglobin  10.5  13.5 - 17.5  g/dL  LOW"
_TABULAR = re.compile(
    r"^(?P<test>[A-Za-z][A-Za-z0-9 \-/\(\)]{1,50}?)"
    r"\s{2,}"
    r"(?P<value>[\d\.]+)"
    r"\s{2,}"
    r"(?P<ref>[\d\.]+ \s* - \s* [\d\.]+)?"
    r"(?:\s{2,}(?P<unit>[a-zA-Z/%µ\^][A-Za-z/%µ0-9\.\-\^/]*))?"
    r"(?:\s{2,}(?P<flag>LOW|HIGH|NORMAL|ABNORMAL|CRITICAL))?"
    r"\s*$",
    re.IGNORECASE,
)

# Colon: "Glucose: 115 mg/dL"
_COLON = re.compile(
    r"^(?P<test>[A-Za-z][A-Za-z0-9 \-/\(\)]{1,45}?)"
    r"\s*[:=]\s*"
    r"(?P<value>[\d\.]+)"
    r"\s*(?P<unit>[a-zA-Z/%µ][A-Za-z/%µ0-9\.\-]*)?"
    r"(?:\s*\[(?P<flag>[A-Za-z ]+)\])?"
    r"\s*$",
    re.IGNORECASE,
)

# Reference range "13.5-17.5" or "13.5 - 17.5"
_REF_RANGE = re.compile(r"([\d\.]+)\s*-\s*([\d\.]+)")

# Flag keywords
_FLAG_WORDS = re.compile(
    r"\b(critical\s*(?:low|high)?|low|high|abnormal|normal|elevated|decreased)\b",
    re.IGNORECASE,
)


def _parse_ref_range(ref_str: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not ref_str:
        return None, None
    m = _REF_RANGE.search(ref_str)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass
    return None, None


def _normalise_flag(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    r = raw.strip().upper()
    if "CRITICAL" in r and "LOW" in r:
        return "CRITICAL LOW"
    if "CRITICAL" in r and "HIGH" in r:
        return "CRITICAL HIGH"
    if "CRITICAL" in r:
        return "CRITICAL"
    if r in ("LOW", "DECREASED"):
        return "LOW"
    if r in ("HIGH", "ELEVATED", "ABNORMAL"):
        return "HIGH"
    if r == "NORMAL":
        return "NORMAL"
    return r


# ─────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────

class LabReportParser:
    """
    Parses structured lab report text line-by-line into LabEntry objects.

    Usage::

        parser = LabReportParser()
        entries = parser.parse(report_text)
        for e in entries:
            print(e.test_name, e.numeric_value, e.unit, e.flag)
    """
    
    def parse(self, text: str) -> List[LabEntry]:
        entries: List[LabEntry] = []
        seen: set = set()
        
        # Valid test names that should be extracted (first word of test)
        valid_starts = [
            'haemoglobin', 'hematocrit', 'pcv', 'rbc', 'wbc', 'platelet', 'mcv', 'mch', 'mchc',
            'neutrophil', 'lymphocyte', 'eosinophil', 'monocyte', 'basophil',
            'serum', 'tibc', 'transferrin', 'ferritin',
            'fasting', 'glucose', 'hba1c', 'alkaline', 'vitamin',
            'sodium', 'potassium', 'calcium', 'albumin', 'protein',
            'ast', 'alt', 'ggt', 'ldh', 'creatinine', 'bun',
        ]

        for raw_line in text.splitlines():
            line = raw_line.strip()
            
            # Skip empty/short lines
            if not line or len(line) < 10:
                continue
            
            # ONLY match lines that start with a valid test name
            line_lower = line.lower()
            if not any(line_lower.startswith(test) for test in valid_starts):
                continue
            
            # Only process tabular format (multiple spaces)
            if '  ' not in line:
                continue
            
            entry = self._parse_tabular(line)
            if entry is None:
                continue

            # Dedup by test name
            key = entry.test_name.lower()
            if key in seen:
                continue
            seen.add(key)

            entry.raw_line = raw_line
            entries.append(entry)

        return entries

    # ── line parsers ──────────────────────────

    def _parse_tabular(self, line: str) -> Optional[LabEntry]:
        m = _TABULAR.match(line)
        if not m:
            return None
        ref_low, ref_high = _parse_ref_range(m.group("ref"))
        numeric = extract_numeric(m.group("value"))
        return LabEntry(
            test_name=m.group("test").strip(),
            raw_value=m.group("value"),
            numeric_value=numeric,
            unit=normalize_unit(m.group("unit") or ""),
            ref_low=ref_low,
            ref_high=ref_high,
            flag=_normalise_flag(m.group("flag")),
        )

    def _parse_colon(self, line: str) -> Optional[LabEntry]:
        m = _COLON.match(line)
        if not m:
            return None
        numeric = extract_numeric(m.group("value"))
        if numeric is None:
            return None
        return LabEntry(
            test_name=m.group("test").strip(),
            raw_value=m.group("value"),
            numeric_value=numeric,
            unit=normalize_unit(m.group("unit") or ""),
            flag=_normalise_flag(m.group("flag")),
        )

    # ── inline flag scan ─────────────────────

    @staticmethod
    def scan_inline_flags(text: str) -> List[Tuple[str, str]]:
        """
        Fallback: scan free-text for (value, flag) pairs
        e.g. "hemoglobin 9.5 g/dL [LOW]" → ("9.5", "LOW")
        """
        results = []
        for m in _FLAG_WORDS.finditer(text):
            flag = _normalise_flag(m.group(1))
            # grab the nearest number before the flag
            snippet = text[max(0, m.start() - 30): m.start()]
            num_m = re.search(r"[\d\.]+", snippet[::-1])
            if num_m:
                value = num_m.group()[::-1]
                results.append((value, flag))
        return results
