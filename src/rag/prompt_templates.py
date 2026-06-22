"""
src/rag/prompt_templates.py
Centralised prompt templates for all LLM calls in the project.

All templates use Python .format() substitution with named keys.
Import the PromptLibrary singleton rather than the raw strings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# ─────────────────────────────────────────────
# Template definitions
# ─────────────────────────────────────────────

# ── RAG Q&A ──────────────────────────────────

RAG_QA = """\
You are a knowledgeable medical AI assistant. Your role is to help users \
understand their medical reports and general health questions.

INSTRUCTIONS:
- Answer ONLY using the provided context. Do not fabricate information.
- If the answer is not in the context, say so clearly.
- Be accurate, clear, and compassionate.
- Always remind users to consult a qualified healthcare professional.

RETRIEVED MEDICAL KNOWLEDGE:
{context}

{report_section}\
CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

ANSWER:"""

RAG_REPORT_SECTION = """\
PATIENT REPORT FINDINGS:
{report_findings}

"""

# ── Clinical summary ─────────────────────────

CLINICAL_SUMMARY = """\
You are a clinical documentation assistant.
Write a concise clinical summary (maximum {max_words} words) based on the \
lab findings below. Use formal medical terminology. Be factual. \
Do not speculate beyond the provided data.

LAB FINDINGS:
{findings}

ABNORMALITY SUMMARY:
{abnormality_summary}

EXTRACTED CLINICAL ENTITIES:
{entities}

CLINICAL SUMMARY:"""

# ── Patient-friendly summary ─────────────────

PATIENT_SUMMARY = """\
You are a compassionate healthcare educator explaining test results to a patient \
with no medical background.

Write a patient-friendly summary (maximum {max_words} words, {reading_level} reading level).
- Use simple everyday language, no jargon.
- Explain what each abnormal result means in plain terms.
- Be reassuring but honest.
- End with one sentence reminding them to discuss results with their doctor.

LAB FINDINGS:
{findings}

ABNORMALITY OVERVIEW:
{abnormality_summary}

PATIENT-FRIENDLY SUMMARY:"""

# ── Recommendations ───────────────────────────

RECOMMENDATIONS = """\
Based on the following medical findings, list {n} specific, actionable \
recommendations the patient should discuss with their doctor. \
One sentence per recommendation. Be concrete and practical.

FINDINGS:
{findings}

Numbered list only, no preamble:"""

# ── Follow-up questions ───────────────────────

FOLLOW_UP_QUESTIONS = """\
Based on the following medical findings, generate {n} important follow-up questions \
a patient should ask their doctor at their next visit. \
Focus on clarification, next steps, and lifestyle changes.

FINDINGS:
{findings}

Numbered list only, no preamble:"""

# ── Doctor Q&A ────────────────────────────────

DOCTOR_QA = """\
You are assisting a physician reviewing a patient's lab report. \
Provide a concise, technically accurate response using clinical language.

PATIENT FINDINGS:
{findings}

PHYSICIAN QUESTION: {question}

CLINICAL RESPONSE:"""

# ── Report chat system message ────────────────

CHAT_SYSTEM = """\
You are a helpful medical AI assistant. You help users understand their \
medical reports. You have access to general medical knowledge and \
the patient's specific report findings.

Key rules:
1. Only state facts supported by the provided context or retrieved knowledge.
2. Never diagnose, prescribe, or make treatment decisions.
3. Always advise consulting a qualified healthcare professional.
4. Be compassionate, clear, and avoid unnecessary medical jargon.
5. If you are unsure, say so rather than guessing."""

# ── Hallucination check ───────────────────────

FAITHFULNESS_CHECK = """\
Given the context below and the generated answer, identify any claims in \
the answer that are NOT supported by the context.

CONTEXT:
{context}

GENERATED ANSWER:
{answer}

List unsupported claims (one per line). \
If all claims are supported, reply with "ALL SUPPORTED"."""


# ─────────────────────────────────────────────
# Prompt library
# ─────────────────────────────────────────────

@dataclass
class PromptTemplate:
    name: str
    template: str
    required_keys: list

    def format(self, **kwargs) -> str:
        missing = [k for k in self.required_keys if k not in kwargs]
        if missing:
            raise ValueError(f"Prompt '{self.name}' missing keys: {missing}")
        return self.template.format(**kwargs)

    def __str__(self) -> str:
        return self.template


class PromptLibrary:
    """
    Central registry for all prompt templates.

    Usage::

        from src.rag.prompt_templates import prompts

        text = prompts.rag_qa.format(
            context="...",
            report_section="",
            history="(none)",
            question="What does low hemoglobin mean?",
        )
    """

    rag_qa = PromptTemplate(
        name="rag_qa",
        template=RAG_QA,
        required_keys=["context", "report_section", "history", "question"],
    )

    clinical_summary = PromptTemplate(
        name="clinical_summary",
        template=CLINICAL_SUMMARY,
        required_keys=["max_words", "findings", "abnormality_summary", "entities"],
    )

    patient_summary = PromptTemplate(
        name="patient_summary",
        template=PATIENT_SUMMARY,
        required_keys=["max_words", "reading_level", "findings", "abnormality_summary"],
    )

    recommendations = PromptTemplate(
        name="recommendations",
        template=RECOMMENDATIONS,
        required_keys=["n", "findings"],
    )

    follow_up_questions = PromptTemplate(
        name="follow_up_questions",
        template=FOLLOW_UP_QUESTIONS,
        required_keys=["n", "findings"],
    )

    doctor_qa = PromptTemplate(
        name="doctor_qa",
        template=DOCTOR_QA,
        required_keys=["findings", "question"],
    )

    chat_system = PromptTemplate(
        name="chat_system",
        template=CHAT_SYSTEM,
        required_keys=[],
    )

    faithfulness_check = PromptTemplate(
        name="faithfulness_check",
        template=FAITHFULNESS_CHECK,
        required_keys=["context", "answer"],
    )

    @classmethod
    def list_templates(cls) -> Dict[str, PromptTemplate]:
        return {
            k: v for k, v in vars(cls).items()
            if isinstance(v, PromptTemplate)
        }


# Singleton for easy import
prompts = PromptLibrary()
