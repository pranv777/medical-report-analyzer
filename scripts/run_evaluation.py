#!/usr/bin/env python3
"""
scripts/run_evaluation.py
Runs the full evaluation suite:
  - NER model comparison (BioBERT vs SciSpacy)
  - Embedding model benchmarks
  - RAGAS RAG evaluation
  - MLflow tracking

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --mode ner
    python scripts/run_evaluation.py --mode embedding
    python scripts/run_evaluation.py --mode ragas
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
app = typer.Typer(help="Run evaluation experiments.")

# ── Built-in NER test samples ──────────────────────────────────────────────
NER_TEST_SAMPLES = [
    {
        "text": "Patient presents with hemoglobin of 9.5 g/dL, indicating microcytic anemia. "
                "Currently on ferrous sulfate 325 mg twice daily.",
        "entities": [
            {"text": "hemoglobin", "label": "LAB_TEST"},
            {"text": "microcytic anemia", "label": "DISEASE"},
            {"text": "ferrous sulfate", "label": "MEDICATION"},
            {"text": "325 mg", "label": "DOSAGE"},
        ],
    },
    {
        "text": "Fasting glucose 126 mg/dL, HbA1c 7.2%. Diagnosis: Type 2 diabetes mellitus. "
                "Prescribed metformin 500 mg twice daily.",
        "entities": [
            {"text": "glucose", "label": "LAB_TEST"},
            {"text": "HbA1c", "label": "LAB_TEST"},
            {"text": "Type 2 diabetes mellitus", "label": "DISEASE"},
            {"text": "metformin", "label": "MEDICATION"},
            {"text": "500 mg", "label": "DOSAGE"},
        ],
    },
    {
        "text": "TSH 8.2 mIU/L, free T4 0.6 ng/dL. Hypothyroidism confirmed. "
                "Initiated levothyroxine 50 mcg daily.",
        "entities": [
            {"text": "TSH", "label": "LAB_TEST"},
            {"text": "free T4", "label": "LAB_TEST"},
            {"text": "Hypothyroidism", "label": "DISEASE"},
            {"text": "levothyroxine", "label": "MEDICATION"},
            {"text": "50 mcg", "label": "DOSAGE"},
        ],
    },
]

# ── Built-in RAG test samples ──────────────────────────────────────────────
RAG_TEST_SAMPLES = [
    {
        "question": "What does low hemoglobin indicate?",
        "ground_truth": (
            "Low hemoglobin indicates anemia. It may be caused by iron deficiency, "
            "vitamin B12 or folate deficiency, chronic disease, or blood loss. "
            "Symptoms include fatigue, weakness, and pale skin."
        ),
    },
    {
        "question": "What is a normal range for HbA1c?",
        "ground_truth": (
            "A normal HbA1c is below 5.7%. Pre-diabetes is 5.7-6.4%. "
            "Diabetes is diagnosed at 6.5% or higher."
        ),
    },
    {
        "question": "Why is high LDL cholesterol concerning?",
        "ground_truth": (
            "High LDL cholesterol raises the risk of atherosclerosis, heart disease, "
            "and stroke by contributing to plaque buildup in arteries."
        ),
    },
]


@app.command()
def evaluate(
    mode: str = typer.Option(
        "all", "--mode", "-m",
        help="Evaluation mode: all | ner | embedding | ragas"
    ),
    output: str = typer.Option("outputs/", "--output", "-o"),
    config: str = typer.Option(None, "--config"),
):
    """Run evaluation experiments and log results."""
    from src.utils.config import get_settings
    from src.utils.logger import setup_logger
    from src.utils.helpers import save_json, ensure_dir
    import json

    cfg = get_settings(config) if config else get_settings()
    setup_logger(log_dir=cfg.project.log_dir, log_level=cfg.project.log_level)
    ensure_dir(output)

    # Write test data files
    ner_path = Path(output) / "ner_test.json"
    rag_path = Path(output) / "rag_test.json"
    save_json(NER_TEST_SAMPLES, ner_path)
    save_json(RAG_TEST_SAMPLES, rag_path)

    all_results = {}

    # ── NER evaluation ───────────────────────
    if mode in ("all", "ner"):
        console.rule("[bold cyan]NER Evaluation")
        try:
            from src.extraction.ner_extractor import MedicalNERPipeline
            from src.evaluation.evaluator import NEREvaluator

            pipeline  = MedicalNERPipeline(cfg.extraction)
            evaluator = NEREvaluator()
            preds, golds = [], []

            for sample in NER_TEST_SAMPLES:
                res = pipeline.run(sample["text"])
                preds.append([e.to_dict() for e in res.entities])
                golds.append(sample["entities"])

            metrics = evaluator.evaluate(preds, golds)

            table = Table(title="NER Metrics", box=box.ROUNDED)
            table.add_column("Entity Type", style="cyan")
            table.add_column("Precision", justify="right")
            table.add_column("Recall", justify="right")
            table.add_column("F1", justify="right")
            table.add_column("Support", justify="right")

            for m in metrics:
                table.add_row(
                    m.entity_type,
                    f"{m.precision:.3f}",
                    f"{m.recall:.3f}",
                    f"[bold]{m.f1:.3f}[/bold]",
                    str(m.support),
                )
            console.print(table)
            all_results["ner"] = [vars(m) for m in metrics]
        except Exception as exc:
            console.print(f"[red]NER evaluation error: {exc}[/red]")

    # ── Embedding comparison ─────────────────
    if mode in ("all", "embedding"):
        console.rule("[bold cyan]Embedding Model Comparison")
        try:
            from src.evaluation.evaluator import EmbeddingEvaluator

            # Build a tiny retrieval benchmark from built-in KB docs
            from scripts.build_knowledge_base import SAMPLE_DOCUMENTS
            corpus = [
                {"id": f"doc_{i}", "content": d["content"]}
                for i, d in enumerate(SAMPLE_DOCUMENTS)
            ]
            queries = [
                {"query": "low hemoglobin anemia treatment", "relevant_doc_ids": ["doc_0", "doc_6", "doc_7"]},
                {"query": "diabetes HbA1c glucose management", "relevant_doc_ids": ["doc_1"]},
                {"query": "cholesterol LDL cardiovascular risk", "relevant_doc_ids": ["doc_2"]},
                {"query": "thyroid TSH hypothyroidism", "relevant_doc_ids": ["doc_3"]},
                {"query": "vitamin D deficiency supplementation", "relevant_doc_ids": ["doc_4"]},
            ]

            evaluator = EmbeddingEvaluator(corpus)
            models = cfg.evaluation.embedding_comparison.models if hasattr(cfg.evaluation, 'embedding_comparison') else [
                "sentence-transformers/all-MiniLM-L6-v2",
            ]

            table = Table(title="Embedding Comparison", box=box.ROUNDED)
            table.add_column("Model", style="cyan")
            table.add_column("P@1", justify="right")
            table.add_column("P@3", justify="right")
            table.add_column("P@5", justify="right")
            table.add_column("R@5", justify="right")
            table.add_column("MRR", justify="right")
            table.add_column("Latency(ms)", justify="right")

            emb_results = []
            for model_name in models:
                with console.status(f"Evaluating {model_name}…"):
                    try:
                        m = evaluator.evaluate_model(model_name, queries)
                        table.add_row(
                            model_name.split("/")[-1],
                            f"{m.precision_at_1:.3f}",
                            f"{m.precision_at_3:.3f}",
                            f"{m.precision_at_5:.3f}",
                            f"{m.recall_at_5:.3f}",
                            f"[bold]{m.mrr:.3f}[/bold]",
                            f"{m.latency_ms:.1f}",
                        )
                        emb_results.append(vars(m))
                    except Exception as exc:
                        console.print(f"[yellow]Skipped {model_name}: {exc}[/yellow]")

            console.print(table)
            all_results["embedding"] = emb_results
        except Exception as exc:
            console.print(f"[red]Embedding evaluation error: {exc}[/red]")

    # ── RAGAS evaluation ─────────────────────
    if mode in ("all", "ragas"):
        console.rule("[bold cyan]RAGAS Evaluation")
        try:
            from src.rag.rag_pipeline import MedicalRAGPipeline
            from src.evaluation.evaluator import RAGASEvaluator

            rag = MedicalRAGPipeline(cfg)
            questions, answers, contexts, ground_truths = [], [], [], []

            console.print("[dim]Running RAG pipeline on test questions…[/dim]")
            for sample in RAG_TEST_SAMPLES:
                rag.reset_conversation()
                resp = rag.ask(sample["question"])
                questions.append(sample["question"])
                answers.append(resp.answer)
                contexts.append([d["content"] for d in resp.retrieved_docs])
                ground_truths.append(sample["ground_truth"])

            evaluator = RAGASEvaluator()
            with console.status("[bold green]Running RAGAS metrics…"):
                metrics = evaluator.evaluate(questions, answers, contexts, ground_truths)

            table = Table(title="RAGAS Metrics", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Score", justify="right")
            table.add_row("Faithfulness",      f"[bold]{metrics.faithfulness:.4f}[/bold]")
            table.add_row("Answer Relevancy",  f"[bold]{metrics.answer_relevancy:.4f}[/bold]")
            table.add_row("Context Precision", f"[bold]{metrics.context_precision:.4f}[/bold]")
            table.add_row("Context Recall",    f"[bold]{metrics.context_recall:.4f}[/bold]")
            console.print(table)
            all_results["ragas"] = vars(metrics)
        except Exception as exc:
            console.print(f"[red]RAGAS evaluation error: {exc}[/red]")
            console.print("[dim]Hint: ensure ragas is installed and Ollama/LLM is running.[/dim]")

    # ── Save results ─────────────────────────
    out_file = Path(output) / "evaluation_results.json"
    save_json(all_results, out_file)
    console.print(f"\n[green]✓[/green] Results saved → [cyan]{out_file}[/cyan]")


if __name__ == "__main__":
    app()
