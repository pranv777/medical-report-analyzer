"""
src/pipeline.py
End-to-end orchestration pipeline.

Chains:
  ReportIngestion
    → MedicalNERPipeline
    → AbnormalityDetector
    → MedicalSummarizer
    → MedicalRAGPipeline  (optional: conversational follow-up)

Single entry point for both CLI scripts and notebook usage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import get_settings
from src.utils.helpers import ensure_dir, save_json
from src.utils.logger import get_logger, setup_logger

log = get_logger(__name__)


@dataclass
class FullAnalysisResult:
    file_path: str
    ingestion: Dict[str, Any]   = field(default_factory=dict)
    entities: Dict[str, Any]    = field(default_factory=dict)
    detection: Dict[str, Any]   = field(default_factory=dict)
    summary: Dict[str, Any]     = field(default_factory=dict)
    success: bool = False
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "success":   self.success,
            "errors":    self.errors,
            "ingestion": self.ingestion,
            "entities":  self.entities,
            "detection": self.detection,
            "summary":   self.summary,
        }


class MedicalReportPipeline:
    """
    Usage::

        pipeline = MedicalReportPipeline()
        result   = pipeline.analyze("path/to/report.pdf")
        pipeline.save_result(result, "outputs/")

        # Conversational follow-up
        rag_resp = pipeline.ask("What does low hemoglobin mean for me?")
    """

    def __init__(self, config=None):
        cfg = config or get_settings()
        setup_logger(
            log_dir=cfg.project.log_dir,
            log_level=cfg.project.log_level,
        )
        self.cfg = cfg
        self._rag_pipeline = None  # lazy

        # Lazy-loaded sub-pipelines
        from src.ingestion.report_ingestion import ReportIngestion
        from src.extraction.ner_extractor import MedicalNERPipeline
        from src.detection.abnormality_detector import AbnormalityDetector
        from src.summarization.summarizer import MedicalSummarizer

        self.ingestion  = ReportIngestion(cfg.ingestion)
        self.ner        = MedicalNERPipeline(cfg.extraction)
        self.detector   = AbnormalityDetector(cfg.detection)
        self.summarizer = MedicalSummarizer(cfg)
        self._last_result: Optional[FullAnalysisResult] = None

    # ── core analyze ──────────────────────────

    def analyze(
        self,
        file_path: str,
        gender: str = "default",
        age: int = 30,
        skip_summarization: bool = False,
    ) -> FullAnalysisResult:
        result = FullAnalysisResult(file_path=file_path)
        log.info(f"=== Starting analysis: {Path(file_path).name} ===")

        try:
            # 1. Ingest
            log.info("Step 1/4 — Ingesting report…")
            ing = self.ingestion.ingest(file_path)
            if not ing.success:
                result.errors.extend(ing.errors)
                return result

            result.ingestion = {
                "file_hash":   ing.file_hash,
                "file_type":   ing.file_type,
                "page_count":  len(ing.pages),
                "char_count":  len(ing.full_text),
                "metadata":    ing.metadata,
            }

            # 2. NER
            log.info("Step 2/4 — Extracting medical entities…")
            ner_result = self.ner.run(ing.full_text)
            result.entities = {
                "entity_count": len(ner_result.entities),
                "lab_value_count": len(ner_result.lab_values),
                "entities": [e.to_dict() for e in ner_result.entities],
                "lab_values": ner_result.lab_values,
                "entity_summary": ner_result.entity_summary,
            }

            # 3. Abnormality detection
            log.info("Step 3/4 — Detecting abnormalities…")
            det_report = self.detector.detect(
                ner_result.lab_values, gender=gender, age=age
            )
            result.detection = det_report.to_dict()

            # 4. Summarization
            if not skip_summarization:
                log.info("Step 4/4 — Generating summaries…")
                summary = self.summarizer.summarize(
                    detection_report=det_report.to_dict(),
                    extraction_result=ner_result,
                )
                result.summary = summary.to_dict()

            result.success = True
            self._last_result = result
            log.info("=== Analysis complete ===")

        except Exception as exc:
            log.exception(f"Pipeline error: {exc}")
            result.errors.append(str(exc))

        return result

    # ── conversational RAG ────────────────────

    def ask(self, question: str) -> str:
        if self._rag_pipeline is None:
            from src.rag.rag_pipeline import MedicalRAGPipeline
            self._rag_pipeline = MedicalRAGPipeline(self.cfg)

        if self._last_result and self._last_result.detection:
            self._rag_pipeline.set_report_context(self._last_result.detection)

        resp = self._rag_pipeline.ask(question)
        return resp.answer

    def reset_chat(self) -> None:
        if self._rag_pipeline:
            self._rag_pipeline.reset_conversation()

    # ── output helpers ───────────────────────

    def save_result(
        self,
        result: FullAnalysisResult,
        output_dir: str = "outputs/",
    ) -> str:
        ensure_dir(output_dir)
        stem = Path(result.file_path).stem
        out  = Path(output_dir) / f"{stem}_analysis.json"
        save_json(result.to_dict(), out)
        log.info(f"Result saved → {out}")
        return str(out)

    def print_summary(self, result: FullAnalysisResult) -> None:
        """Pretty-print key findings to stdout."""
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        console.rule("[bold blue]Medical Report Analysis")

        # Detection summary
        detection = result.detection
        if detection.get("summary"):
            console.print(f"\n[bold]Summary:[/bold] {detection['summary']}\n")

        # Findings table
        findings = detection.get("findings", [])
        if findings:
            table = Table(
                title="Lab Findings",
                box=box.ROUNDED,
                show_lines=True,
            )
            table.add_column("Test",   style="cyan", no_wrap=True)
            table.add_column("Value",  style="white")
            table.add_column("Unit",   style="white")
            table.add_column("Status", style="white")
            table.add_column("Interpretation", style="dim")

            for f in findings:
                status = f.get("status", "Unknown")
                color  = {
                    "Normal":        "green",
                    "Low":           "yellow",
                    "High":          "yellow",
                    "Critical Low":  "red",
                    "Critical High": "red",
                }.get(status, "white")
                table.add_row(
                    f.get("display_name", f.get("test_name", "")),
                    str(f.get("value", "")),
                    f.get("unit", ""),
                    f"[{color}]{status}[/{color}]",
                    f.get("interpretation", "")[:70] + ("…" if len(f.get("interpretation", "")) > 70 else ""),
                )
            console.print(table)

        # Patient summary
        patient_summary = result.summary.get("patient_summary", "")
        if patient_summary:
            console.print("\n[bold]Patient Summary:[/bold]")
            console.print(patient_summary)

        # Follow-up questions
        follow_ups = result.summary.get("follow_up_questions", [])
        if follow_ups:
            console.print("\n[bold]Suggested Follow-up Questions:[/bold]")
            for i, q in enumerate(follow_ups, 1):
                console.print(f"  {i}. {q}")
