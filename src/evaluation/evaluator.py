"""
src/evaluation/evaluator.py
Comprehensive evaluation suite:

  A. RAG Evaluation (RAGAS)
       — Faithfulness, Answer Relevance, Context Precision, Context Recall

  B. NER Evaluation
       — Token-level Precision / Recall / F1 across entity types

  C. Embedding Model Comparison
       — Precision@K, Recall@K, MRR across BGE / E5 / MiniLM

  D. LLM Comparison
       — Relevance, Faithfulness, Hallucination proxy

All results are tracked in MLflow.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.config import get_settings
from src.utils.helpers import load_json, save_json
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
# Result data models
# ─────────────────────────────────────────────

@dataclass
class NERMetrics:
    entity_type: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EmbeddingMetrics:
    model_name: str
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_5: float
    mrr: float
    latency_ms: float


@dataclass
class RAGASMetrics:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


@dataclass
class EvaluationReport:
    ner_metrics: List[NERMetrics] = field(default_factory=list)
    embedding_metrics: List[EmbeddingMetrics] = field(default_factory=list)
    ragas_metrics: Optional[RAGASMetrics] = None
    run_id: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "ner_metrics": [asdict(m) for m in self.ner_metrics],
            "embedding_metrics": [asdict(m) for m in self.embedding_metrics],
            "ragas_metrics": asdict(self.ragas_metrics) if self.ragas_metrics else None,
        }


# ─────────────────────────────────────────────
# NER Evaluator
# ─────────────────────────────────────────────

class NEREvaluator:
    """
    Evaluates NER model against a gold-standard annotated test set.

    Test set format (JSON list)::

        [
          {
            "text": "Patient has hemoglobin 10.5 g/dL and high cholesterol.",
            "entities": [
              {"text": "hemoglobin", "label": "LAB_TEST", "start": 12, "end": 22},
              {"text": "high cholesterol", "label": "DISEASE", "start": 35, "end": 51}
            ]
          }
        ]
    """

    def evaluate(
        self,
        predictions: List[List[Dict]],
        ground_truth: List[List[Dict]],
    ) -> List[NERMetrics]:
        from collections import defaultdict

        tp: Dict[str, int] = defaultdict(int)
        fp: Dict[str, int] = defaultdict(int)
        fn: Dict[str, int] = defaultdict(int)

        for pred_ents, gold_ents in zip(predictions, ground_truth):
            pred_set = {(e["text"].lower(), e["label"]) for e in pred_ents}
            gold_set = {(e["text"].lower(), e["label"]) for e in gold_ents}

            for item in pred_set & gold_set:
                tp[item[1]] += 1
            for item in pred_set - gold_set:
                fp[item[1]] += 1
            for item in gold_set - pred_set:
                fn[item[1]] += 1

        metrics = []
        all_labels = set(tp) | set(fp) | set(fn)
        for label in sorted(all_labels):
            p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
            r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            support = tp[label] + fn[label]
            metrics.append(NERMetrics(
                entity_type=label,
                precision=round(p, 4),
                recall=round(r, 4),
                f1=round(f, 4),
                support=support,
            ))

        return metrics


# ─────────────────────────────────────────────
# Embedding Model Evaluator
# ─────────────────────────────────────────────

class EmbeddingEvaluator:
    """
    Compares multiple embedding models on a retrieval benchmark dataset.

    Benchmark format::

        [
          {
            "query": "causes of anemia",
            "relevant_doc_ids": ["doc_001", "doc_005"]
          }
        ]
    """

    def __init__(self, corpus: List[Dict[str, str]], k_values: List[int] = None):
        """
        corpus: list of {"id": ..., "content": ...} dicts
        """
        self.corpus   = corpus
        self.k_values = k_values or [1, 3, 5]

    def evaluate_model(
        self, model_name: str, queries: List[Dict]
    ) -> EmbeddingMetrics:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        log.info(f"Evaluating embedding model: {model_name}")
        t0 = time.perf_counter()
        model = SentenceTransformer(model_name)

        corpus_texts = [d["content"] for d in self.corpus]
        corpus_ids   = [d["id"] for d in self.corpus]
        corpus_vecs  = model.encode(corpus_texts, show_progress_bar=False)

        precision_at = {k: [] for k in self.k_values}
        recall_at_5  = []
        rr_list      = []

        for q_item in queries:
            q_vec = model.encode([q_item["query"]])[0]
            scores = corpus_vecs @ q_vec / (
                np.linalg.norm(corpus_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-9
            )
            ranked_ids = [corpus_ids[i] for i in np.argsort(-scores)]
            relevant   = set(q_item.get("relevant_doc_ids", []))

            for k in self.k_values:
                top_k = set(ranked_ids[:k])
                precision_at[k].append(len(top_k & relevant) / k if k > 0 else 0)

            top5 = set(ranked_ids[:5])
            recall_at_5.append(len(top5 & relevant) / len(relevant) if relevant else 0)

            # MRR
            rr = 0.0
            for rank, doc_id in enumerate(ranked_ids[:10], 1):
                if doc_id in relevant:
                    rr = 1.0 / rank
                    break
            rr_list.append(rr)

        latency_ms = (time.perf_counter() - t0) * 1000 / max(len(queries), 1)

        return EmbeddingMetrics(
            model_name=model_name,
            precision_at_1=round(sum(precision_at[1]) / len(precision_at[1]), 4) if 1 in precision_at else 0,
            precision_at_3=round(sum(precision_at[3]) / len(precision_at[3]), 4) if 3 in precision_at else 0,
            precision_at_5=round(sum(precision_at[5]) / len(precision_at[5]), 4) if 5 in precision_at else 0,
            recall_at_5=round(sum(recall_at_5) / len(recall_at_5), 4),
            mrr=round(sum(rr_list) / len(rr_list), 4),
            latency_ms=round(latency_ms, 1),
        )


# ─────────────────────────────────────────────
# RAGAS Evaluator
# ─────────────────────────────────────────────

class RAGASEvaluator:
    """
    Evaluates RAG pipeline quality using the RAGAS framework.

    Expects a dataset of (question, answer, contexts, ground_truth) tuples.
    """

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> RAGASMetrics:
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
        except ImportError:
            raise ImportError("Install ragas: pip install ragas")

        log.info(f"Running RAGAS evaluation on {len(questions)} samples…")

        dataset = Dataset.from_dict({
            "question":      questions,
            "answer":        answers,
            "contexts":      contexts,
            "ground_truth":  ground_truths,
        })

        result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        df = result.to_pandas()
        return RAGASMetrics(
            faithfulness=      round(df["faithfulness"].mean(), 4),
            answer_relevancy=  round(df["answer_relevancy"].mean(), 4),
            context_precision= round(df["context_precision"].mean(), 4),
            context_recall=    round(df["context_recall"].mean(), 4),
        )


# ─────────────────────────────────────────────
# MLflow tracker
# ─────────────────────────────────────────────

class MLflowTracker:
    def __init__(self, config=None):
        cfg = config or get_settings().evaluation.mlflow
        self.enabled         = cfg.enabled
        self.tracking_uri    = cfg.tracking_uri
        self.experiment_name = cfg.experiment_name
        self._run            = None

    def start_run(self, run_name: str = "eval") -> Optional[str]:
        if not self.enabled:
            return None
        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        log.info(f"MLflow run started: {self._run.info.run_id}")
        return self._run.info.run_id

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if not self.enabled or self._run is None:
            return
        import mlflow
        mlflow.log_metrics(metrics)

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.enabled or self._run is None:
            return
        import mlflow
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_artifact(self, path: str) -> None:
        if not self.enabled or self._run is None:
            return
        import mlflow
        mlflow.log_artifact(path)

    def end_run(self) -> None:
        if not self.enabled or self._run is None:
            return
        import mlflow
        mlflow.end_run()
        log.info("MLflow run ended")


# ─────────────────────────────────────────────
# Unified evaluation orchestrator
# ─────────────────────────────────────────────

class EvaluationOrchestrator:
    """
    Runs all evaluation components and produces a unified report.

    Usage::

        orch = EvaluationOrchestrator()
        report = orch.run_full_evaluation(
            ner_test_path="data/processed/ner_test.json",
            rag_test_path="data/processed/rag_test.json",
        )
        save_json(report.to_dict(), "outputs/evaluation_report.json")
    """

    def __init__(self, config=None):
        self.cfg     = config or get_settings()
        self.tracker = MLflowTracker(self.cfg.evaluation.mlflow)

    def run_full_evaluation(
        self,
        ner_test_path: Optional[str] = None,
        rag_test_path: Optional[str] = None,
        output_path: str = "outputs/evaluation_report.json",
    ) -> EvaluationReport:
        from datetime import datetime

        run_id = self.tracker.start_run(run_name="full_eval")
        report = EvaluationReport(
            run_id=run_id or "local",
            timestamp=datetime.utcnow().isoformat(),
        )

        # ── NER evaluation ───────────────────
        if ner_test_path and Path(ner_test_path).exists():
            log.info("Running NER evaluation…")
            test_data = load_json(ner_test_path)
            from src.extraction.ner_extractor import MedicalNERPipeline

            pipeline = MedicalNERPipeline(self.cfg.extraction)
            predictions = []
            ground_truths_ner = []
            for sample in test_data:
                res = pipeline.run(sample["text"])
                predictions.append([e.to_dict() for e in res.entities])
                ground_truths_ner.append(sample.get("entities", []))

            ner_eval = NEREvaluator()
            report.ner_metrics = ner_eval.evaluate(predictions, ground_truths_ner)

            avg_f1 = sum(m.f1 for m in report.ner_metrics) / max(len(report.ner_metrics), 1)
            self.tracker.log_metrics({"ner_avg_f1": avg_f1})
            log.info(f"NER avg F1: {avg_f1:.4f}")

        # ── RAGAS evaluation ─────────────────
        if rag_test_path and Path(rag_test_path).exists():
            log.info("Running RAGAS evaluation…")
            rag_data = load_json(rag_test_path)
            from src.rag.rag_pipeline import MedicalRAGPipeline

            rag = MedicalRAGPipeline(self.cfg)
            questions, answers, contexts, ground_truths = [], [], [], []

            for sample in rag_data:
                resp = rag.ask(sample["question"])
                questions.append(sample["question"])
                answers.append(resp.answer)
                contexts.append([d["content"] for d in resp.retrieved_docs])
                ground_truths.append(sample.get("ground_truth", ""))

            ragas_eval = RAGASEvaluator()
            try:
                report.ragas_metrics = ragas_eval.evaluate(
                    questions, answers, contexts, ground_truths
                )
                self.tracker.log_metrics(asdict(report.ragas_metrics))
                log.info(f"RAGAS faithfulness: {report.ragas_metrics.faithfulness}")
            except Exception as exc:
                log.warning(f"RAGAS evaluation failed: {exc}")

        # ── Save report ──────────────────────
        save_json(report.to_dict(), output_path)
        if Path(output_path).exists():
            self.tracker.log_artifact(output_path)

        self.tracker.end_run()
        log.info(f"Evaluation complete — report saved to {output_path}")
        return report
