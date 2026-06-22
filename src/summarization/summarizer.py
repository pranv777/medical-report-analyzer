"""
src/summarization/summarizer.py
Generates two types of medical report summaries:

  1. Clinical Summary   — concise, technical, doctor-oriented
  2. Patient Summary    — plain-language explanation for the patient

Also generates:
  • Recommended follow-up questions
  • Actionable recommendations list
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
# Output model
# ─────────────────────────────────────────────

@dataclass
class SummaryResult:
    clinical_summary: str   = ""
    patient_summary: str    = ""
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "clinical_summary":     self.clinical_summary,
            "patient_summary":      self.patient_summary,
            "key_findings":         self.key_findings,
            "recommendations":      self.recommendations,
            "follow_up_questions":  self.follow_up_questions,
        }


# ─────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────

_CLINICAL_PROMPT = """\
You are a clinical documentation assistant. Write a concise clinical summary \
(maximum {max_tokens} words) based on the following medical report findings.

Use formal medical terminology. Be factual. Do not speculate beyond the data.

LAB FINDINGS:
{findings}

ABNORMALITY SUMMARY:
{abnormality_summary}

EXTRACTED ENTITIES:
{entities}

Write the clinical summary now:"""

_PATIENT_PROMPT = """\
You are a compassionate healthcare educator explaining medical results to a patient.
Write a patient-friendly summary (maximum {max_tokens} words, {reading_level} reading level).

Use simple, everyday language. Avoid jargon. Be reassuring but honest.
Explain what each abnormal result means in plain terms.

LAB FINDINGS:
{findings}

ABNORMALITY SUMMARY:
{abnormality_summary}

Write the patient-friendly summary now:"""

_RECOMMENDATIONS_PROMPT = """\
Based on these medical findings, list {n} specific, actionable recommendations \
the patient should discuss with their doctor. Be concise (one sentence each).

FINDINGS:
{findings}

Provide only a numbered list, no preamble:"""

_FOLLOW_UP_PROMPT = """\
Based on these medical findings, generate {n} important follow-up questions \
a patient should ask their doctor at their next visit.

FINDINGS:
{findings}

Provide only a numbered list, no preamble:"""


# ─────────────────────────────────────────────
# Summarizer
# ─────────────────────────────────────────────

class MedicalSummarizer:
    """
    Generates clinical and patient-friendly summaries using the configured LLM.

    Usage::

        summarizer = MedicalSummarizer()
        result = summarizer.summarize(
            detection_report=report.to_dict(),
            extraction_result=ner_result,
        )
        print(result.clinical_summary)
        print(result.patient_summary)
    """

    def __init__(self, config=None):
        self.cfg    = config or get_settings()
        self.sum_cfg = self.cfg.summarization
        self._llm   = None

    def _get_llm(self):
        if self._llm is None:
            from src.rag.rag_pipeline import OllamaLLM, HuggingFaceLLM
            rag_cfg = self.cfg.rag
            if rag_cfg.llm_provider == "ollama":
                self._llm = OllamaLLM(
                    model=rag_cfg.llm_model,
                    base_url=rag_cfg.ollama_base_url,
                    temperature=0.2,
                    max_tokens=800,
                )
            else:
                self._llm = HuggingFaceLLM(
                    model=rag_cfg.llm_model,
                    temperature=0.2,
                    max_tokens=800,
                )
        return self._llm

    # ── public ───────────────────────────────

    def summarize(
        self,
        detection_report: Optional[Dict[str, Any]] = None,
        extraction_result: Optional[Any] = None,
        raw_text: str = "",
    ) -> SummaryResult:
        result = SummaryResult()

        findings_str        = self._format_findings(detection_report)
        abnormality_summary = detection_report.get("summary", "") if detection_report else ""
        entities_str        = self._format_entities(extraction_result)

        log.info("Generating clinical summary…")
        result.clinical_summary = self._generate_clinical(
            findings_str, abnormality_summary, entities_str
        )

        log.info("Generating patient-friendly summary…")
        result.patient_summary = self._generate_patient(
            findings_str, abnormality_summary
        )

        result.key_findings = self._extract_key_findings(detection_report)

        if self.sum_cfg.include_recommendations:
            log.info("Generating recommendations…")
            result.recommendations = self._generate_list(
                _RECOMMENDATIONS_PROMPT.format(
                    n=self.sum_cfg.num_follow_up_questions,
                    findings=findings_str,
                )
            )

        if self.sum_cfg.include_follow_up_questions:
            log.info("Generating follow-up questions…")
            result.follow_up_questions = self._generate_list(
                _FOLLOW_UP_PROMPT.format(
                    n=self.sum_cfg.num_follow_up_questions,
                    findings=findings_str,
                )
            )

        log.info("Summarization complete")
        return result

    # ── private generators ───────────────────

    def _generate_clinical(
        self, findings: str, abnormality: str, entities: str
    ) -> str:
        prompt = _CLINICAL_PROMPT.format(
            max_tokens=self.sum_cfg.clinical_max_tokens,
            findings=findings,
            abnormality_summary=abnormality,
            entities=entities,
        )
        return self._get_llm().generate(prompt)

    def _generate_patient(self, findings: str, abnormality: str) -> str:
        prompt = _PATIENT_PROMPT.format(
            max_tokens=self.sum_cfg.patient_max_tokens,
            reading_level=self.sum_cfg.reading_level,
            findings=findings,
            abnormality_summary=abnormality,
        )
        return self._get_llm().generate(prompt)

    def _generate_list(self, prompt: str) -> List[str]:
        raw = self._get_llm().generate(prompt)
        items = []
        for line in raw.splitlines():
            line = line.strip().lstrip("0123456789.-) •").strip()
            if line:
                items.append(line)
        return items

    # ── formatting helpers ───────────────────

    @staticmethod
    def _format_findings(report: Optional[Dict]) -> str:
        if not report:
            return "No structured findings available."
        parts = []
        for f in report.get("findings", []):
            status = f.get("status", "Unknown")
            marker = "⚠️ " if status not in ("Normal", "Unknown") else "✓  "
            parts.append(
                f"{marker}{f.get('display_name', f.get('test_name'))}: "
                f"{f.get('value')} {f.get('unit', '')} [{status}]"
                + (f" — {f.get('interpretation', '')}" if f.get("interpretation") else "")
            )
        return "\n".join(parts) if parts else "No lab findings."

    @staticmethod
    def _format_entities(extraction_result: Optional[Any]) -> str:
        if not extraction_result:
            return "No entities extracted."
        summary = getattr(extraction_result, "entity_summary", {})
        parts = []
        for label, items in summary.items():
            if items:
                parts.append(f"{label}: {', '.join(items[:5])}")
        return "\n".join(parts) if parts else "No entities."

    @staticmethod
    def _extract_key_findings(report: Optional[Dict]) -> List[str]:
        if not report:
            return []
        findings = []
        for f in report.get("findings", []):
            if f.get("status") not in ("Normal", "Unknown"):
                findings.append(
                    f"{f.get('display_name', f.get('test_name'))}: "
                    f"{f.get('value')} {f.get('unit', '')} — {f.get('status')}"
                )
        return findings
