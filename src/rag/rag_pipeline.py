"""
src/rag/rag_pipeline.py
Full Retrieval-Augmented Generation pipeline.

Flow:
  User question
    → embed query
    → similarity search (ChromaDB)
    → inject retrieved context into prompt
    → LLM (Ollama / HuggingFace)
    → structured response with citations

Supports:
  • Single-turn Q&A
  • Multi-turn conversation with sliding-window memory
  • Report-aware mode (injects extracted findings as context)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.knowledge_base.kb_builder import MedicalRetriever
from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# Optional runtime dependency — imported at module level so tests can patch it
try:
    import ollama as ollama  # noqa: F401 (re-exported for patching)
except ImportError:
    ollama = None  # type: ignore


# ─────────────────────────────────────────────
# Response model
# ─────────────────────────────────────────────

@dataclass
class RAGResponse:
    question: str
    answer: str
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    model_used: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question":       self.question,
            "answer":         self.answer,
            "sources":        self.sources,
            "model_used":     self.model_used,
            "latency_ms":     self.latency_ms,
            "num_docs_used":  len(self.retrieved_docs),
        }


# ─────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────

_RAG_PROMPT_TEMPLATE = """\
You are a helpful medical AI assistant. Use ONLY the context below to answer \
the question. If the answer is not in the context, say "I don't have enough \
information to answer this based on the provided context." Never make up \
medical facts. Always remind the user to consult a qualified healthcare \
professional for medical decisions.

MEDICAL CONTEXT:
{context}

{report_section}CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

ANSWER:"""

_REPORT_SECTION_TEMPLATE = """\
PATIENT REPORT FINDINGS:
{report_findings}

"""


def _format_context(docs: List[Dict]) -> str:
    if not docs:
        return "No relevant medical knowledge found."
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.get("metadata", {}).get("source", "unknown")
        parts.append(f"[{i}] (source: {src})\n{doc['content']}")
    return "\n\n".join(parts)


def _format_history(history: List[Dict]) -> str:
    if not history:
        return "(No prior conversation)"
    lines = []
    for turn in history:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# LLM adapter — Ollama
# ─────────────────────────────────────────────

class OllamaLLM:
    def __init__(self, model: str, base_url: str, temperature: float, max_tokens: int):
        self.model       = model
        self.base_url    = base_url
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def generate(self, prompt: str) -> str:
        try:
            import src.rag.rag_pipeline as _mod
            _ollama = _mod.ollama
            if _ollama is None:
                raise ImportError("ollama not installed")
            response = _ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            return response["response"].strip()
        except Exception as exc:
            log.error(f"Ollama generate failed: {exc}")
            raise


# ─────────────────────────────────────────────
# LLM adapter — HuggingFace (fallback)
# ─────────────────────────────────────────────

class HuggingFaceLLM:
    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._pipe       = None

    def _load(self):
        if self._pipe:
            return
        from transformers import pipeline

        log.info(f"Loading HuggingFace pipeline: {self.model}")
        self._pipe = pipeline(
            "text-generation",
            model=self.model,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=(self.temperature > 0),
        )

    def generate(self, prompt: str) -> str:
        self._load()
        result = self._pipe(prompt)
        generated = result[0]["generated_text"]
        # Strip the prompt prefix
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        return generated.strip()


# ─────────────────────────────────────────────
# Main RAG pipeline
# ─────────────────────────────────────────────

class MedicalRAGPipeline:
    """
    Orchestrates retrieval → prompt construction → LLM generation.

    Usage::

        pipeline = MedicalRAGPipeline()

        # Single-turn
        resp = pipeline.ask("Why is low hemoglobin concerning?")
        print(resp.answer)

        # Multi-turn with report context
        pipeline.set_report_context(detection_report.to_dict())
        resp = pipeline.ask("What follow-up tests should I do?")

        # Reset conversation
        pipeline.reset_conversation()
    """

    def __init__(self, config=None):
        self.cfg       = config or get_settings()
        self.rag_cfg   = self.cfg.rag
        self.retriever = MedicalRetriever(self.cfg.knowledge_base)
        self._llm      = self._init_llm()
        self._history: List[Dict[str, str]] = []
        self._report_context: Optional[Dict] = None

    # ── LLM factory ──────────────────────────

    def _init_llm(self):
        if self.rag_cfg.llm_provider == "ollama":
            return OllamaLLM(
                model=self.rag_cfg.llm_model,
                base_url=self.rag_cfg.ollama_base_url,
                temperature=self.rag_cfg.temperature,
                max_tokens=self.rag_cfg.max_tokens,
            )
        elif self.rag_cfg.llm_provider == "huggingface":
            return HuggingFaceLLM(
                model=self.rag_cfg.llm_model,
                temperature=self.rag_cfg.temperature,
                max_tokens=self.rag_cfg.max_tokens,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.rag_cfg.llm_provider}")

    # ── context setters ───────────────────────

    def set_report_context(self, report_dict: Dict) -> None:
        """Inject patient report findings as persistent context."""
        self._report_context = report_dict
        log.debug("Report context set in RAG pipeline")

    def reset_conversation(self) -> None:
        self._history.clear()
        log.debug("Conversation history cleared")

    # ── core ask ─────────────────────────────

    def ask(self, question: str, k: Optional[int] = None) -> RAGResponse:
        import time

        t0 = time.perf_counter()

        # 1. Retrieve relevant knowledge
        docs = self.retriever.retrieve(question, k=k)

        # 2. Build prompt
        context  = _format_context(docs)
        history  = _format_history(self._history[-self.cfg.rag.memory.max_history_turns * 2:] if hasattr(self.cfg.rag, 'memory') else self._history[-6:])

        report_section = ""
        if self._report_context:
            findings_text = self._format_report_findings()
            report_section = _REPORT_SECTION_TEMPLATE.format(
                report_findings=findings_text
            )

        prompt = _RAG_PROMPT_TEMPLATE.format(
            context=context,
            report_section=report_section,
            history=history,
            question=question,
        )

        # 3. Generate
        log.info(f"Generating answer for: '{question[:60]}…'")
        answer = self._llm.generate(prompt)

        # 4. Update history
        self._history.append({"user": question, "assistant": answer})

        latency = (time.perf_counter() - t0) * 1000
        sources = list({
            d.get("metadata", {}).get("source", "unknown") for d in docs
        })

        log.info(f"RAG answered in {latency:.0f}ms | {len(docs)} docs used")
        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_docs=docs,
            sources=sources,
            model_used=self.rag_cfg.llm_model,
            latency_ms=round(latency, 1),
        )

    # ── helper ───────────────────────────────

    def _format_report_findings(self) -> str:
        if not self._report_context:
            return "No structured findings available."
        parts = []
        findings = self._report_context.get("findings", [])
        for f in findings:
            line = (
                f"{f.get('display_name', f.get('test_name'))}: "
                f"{f.get('value')} {f.get('unit', '')} "
                f"[{f.get('status', 'Unknown')}]"
            )
            parts.append(line)
        if self._report_context.get("critical_flags"):
            parts.append("CRITICAL FLAGS: " + "; ".join(self._report_context["critical_flags"]))
        return "\n".join(parts) if parts else "No structured findings available."

    # ── batch Q&A ─────────────────────────────

    def ask_batch(self, questions: List[str]) -> List[RAGResponse]:
        """Run multiple questions sequentially (stateless — no history sharing)."""
        responses = []
        for q in questions:
            self.reset_conversation()
            responses.append(self.ask(q))
        self.reset_conversation()
        return responses


# ─────────────────────────────────────────────
# Predefined follow-up question bank
# ─────────────────────────────────────────────

FOLLOW_UP_QUESTIONS = [
    "Should I repeat any of these tests?",
    "Do I need to see a specialist based on these results?",
    "Could my diet or lifestyle be causing any of these findings?",
    "What medications, if any, might help with these results?",
    "Are any of these results urgent enough to seek immediate care?",
    "How do these results compare to my previous tests?",
    "What follow-up tests would you recommend?",
    "Could any of these findings be related to each other?",
]
