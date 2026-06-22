"""
src/ingestion/text_preprocessor.py
Cleans and normalizes raw text extracted from medical reports.

Steps:
  1. Remove boilerplate headers/footers (page numbers, lab watermarks)
  2. Normalize medical abbreviations to full forms
  3. Split report into labeled sections (HAEMATOLOGY, LIPID PROFILE, etc.)
  4. Detect report metadata (patient name, date, physician)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# Common abbreviation expansions
# ─────────────────────────────────────────────

ABBREVIATIONS: Dict[str, str] = {
    r"\bHgb\b": "Hemoglobin",
    r"\bHGB\b": "Hemoglobin",
    r"\bHb\b":  "Hemoglobin",
    r"\bRBC\b": "Red Blood Cell count",
    r"\bWBC\b": "White Blood Cell count",
    r"\bPlt\b": "Platelets",
    r"\bPLT\b": "Platelets",
    r"\bHct\b": "Hematocrit",
    r"\bHCT\b": "Hematocrit",
    r"\bFBS\b": "Fasting Blood Sugar",
    r"\bPBS\b": "Peripheral Blood Smear",
    r"\bBUN\b": "Blood Urea Nitrogen",
    r"\bCr\b":  "Creatinine",
    r"\bCR\b":  "Creatinine",
    r"\bNa\b":  "Sodium",
    r"\bNA\b":  "Sodium",
    r"\bK\b":   "Potassium",
    r"\bCa\b":  "Calcium",
    r"\bCA\b":  "Calcium",
    r"\bTC\b":  "Total Cholesterol",
    r"\bTG\b":  "Triglycerides",
    r"\bHDL\b": "HDL Cholesterol",
    r"\bLDL\b": "LDL Cholesterol",
    r"\bVLDL\b":"VLDL Cholesterol",
    r"\bALT\b": "Alanine Aminotransferase",
    r"\bAST\b": "Aspartate Aminotransferase",
    r"\bALP\b": "Alkaline Phosphatase",
    r"\bTSH\b": "Thyroid Stimulating Hormone",
    r"\bT3\b":  "Triiodothyronine",
    r"\bT4\b":  "Thyroxine",
    r"\bBID\b": "twice daily",
    r"\bTID\b": "three times daily",
    r"\bQD\b":  "once daily",
    r"\bPRN\b": "as needed",
    r"\bPO\b":  "by mouth",
    r"\bIV\b":  "intravenous",
    r"\bIM\b":  "intramuscular",
}

# Patterns that look like headers/footers to strip
_BOILERPLATE_PATTERNS = [
    r"Page\s+\d+\s+of\s+\d+",
    r"CONFIDENTIAL\s+PATIENT\s+REPORT",
    r"This report is generated electronically.*",
    r"Validated by:.*",
    r"Laboratory certified.*",
    r"Results are for clinical correlation only.*",
    r"(?i)printed\s+on\s+\d{2}[/-]\d{2}[/-]\d{4}",
]

# Section header detection
_SECTION_PATTERN = re.compile(
    r"^[=\-]{3,}[\s\S]*?[=\-]{3,}$|"
    r"^([A-Z][A-Z\s\(\)&/]{4,}):?\s*$",
    re.MULTILINE,
)

# Metadata patterns
_META_PATTERNS: Dict[str, re.Pattern] = {
    "patient_name": re.compile(
        r"(?i)(?:patient|name)\s*[:]\s*([A-Za-z ,\.]+?)(?:\n|DOB|Age|$)"
    ),
    "dob": re.compile(
        r"(?i)(?:DOB|date\s+of\s+birth)\s*[:]\s*(\d{1,2}[/-]\w+[/-]\d{2,4})"
    ),
    "age": re.compile(r"(?i)age\s*[:]\s*(\d{1,3})\s*(?:years?|yrs?)?"),
    "gender": re.compile(r"(?i)(?:gender|sex)\s*[:]\s*(male|female|m|f)\b"),
    "report_date": re.compile(
        r"(?i)(?:date|report date|collection date)\s*[:]\s*(\d{1,2}[/-]\w+[/-]\d{2,4})"
    ),
    "physician": re.compile(
        r"(?i)(?:physician|doctor|referred by|ordering)\s*[:]\s*(Dr\.?\s+[A-Za-z ,\.]+?)(?:\n|,|$)"
    ),
    "lab_id": re.compile(
        r"(?i)(?:lab(?:oratory)?\s*(?:no|id|ref|#)|ref(?:erence)?\s*(?:no|#))\s*[:]\s*([\w\-]+)"
    ),
}


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class ReportSection:
    title: str
    content: str
    start_pos: int
    end_pos: int


@dataclass
class ReportMetadata:
    patient_name: Optional[str] = None
    dob: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    report_date: Optional[str] = None
    physician: Optional[str] = None
    lab_id: Optional[str] = None


@dataclass
class PreprocessedReport:
    raw_text: str
    clean_text: str
    expanded_text: str
    sections: List[ReportSection] = field(default_factory=list)
    metadata: ReportMetadata = field(default_factory=ReportMetadata)


# ─────────────────────────────────────────────
# Preprocessor
# ─────────────────────────────────────────────

class TextPreprocessor:
    """
    Cleans and structures raw text extracted from a medical report.

    Usage::

        preprocessor = TextPreprocessor()
        result = preprocessor.process(raw_text)
        print(result.metadata.patient_name)
        for section in result.sections:
            print(section.title, "→", len(section.content), "chars")
    """

    def process(self, raw_text: str) -> PreprocessedReport:
        clean  = self._strip_boilerplate(raw_text)
        clean  = self._normalize_whitespace(clean)
        expanded = self._expand_abbreviations(clean)
        sections = self._split_sections(clean)
        metadata = self._extract_metadata(raw_text)

        return PreprocessedReport(
            raw_text=raw_text,
            clean_text=clean,
            expanded_text=expanded,
            sections=sections,
            metadata=metadata,
        )

    # ── boilerplate removal ───────────────────

    def _strip_boilerplate(self, text: str) -> str:
        for pattern in _BOILERPLATE_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text

    # ── whitespace normalisation ─────────────

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ── abbreviation expansion ───────────────

    @staticmethod
    def _expand_abbreviations(text: str) -> str:
        for abbr_pattern, full_form in ABBREVIATIONS.items():
            text = re.sub(abbr_pattern, full_form, text)
        return text

    # ── section detection ─────────────────────

    @staticmethod
    def _split_sections(text: str) -> List[ReportSection]:
        sections: List[ReportSection] = []
        lines = text.splitlines()
        current_title = "HEADER"
        current_start = 0
        current_lines: List[str] = []

        section_re = re.compile(r"^(?:[=\-]{3,}|([A-Z][A-Z\s\(\)&/]{4,}))$")

        for i, line in enumerate(lines):
            stripped = line.strip()
            if section_re.match(stripped) and len(stripped) > 4:
                # Save previous section
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        sections.append(ReportSection(
                            title=current_title,
                            content=content,
                            start_pos=current_start,
                            end_pos=i,
                        ))
                current_title = stripped.strip("=- ")
                current_start = i
                current_lines = []
            else:
                current_lines.append(line)

        # Last section
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(ReportSection(
                    title=current_title,
                    content=content,
                    start_pos=current_start,
                    end_pos=len(lines),
                ))

        return sections if sections else [
            ReportSection(title="FULL_REPORT", content=text, start_pos=0, end_pos=len(lines))
        ]

    # ── metadata extraction ──────────────────

    @staticmethod
    def _extract_metadata(text: str) -> ReportMetadata:
        meta = ReportMetadata()
        for field_name, pattern in _META_PATTERNS.items():
            m = pattern.search(text)
            if m:
                value = m.group(1).strip()
                if field_name == "age":
                    try:
                        setattr(meta, field_name, int(value))
                    except ValueError:
                        pass
                elif field_name == "gender":
                    setattr(meta, field_name, "male" if value.lower() in ("m", "male") else "female")
                else:
                    setattr(meta, field_name, value)
        return meta
