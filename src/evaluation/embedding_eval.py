"""
src/evaluation/embedding_eval.py
Retrieval benchmark utilities for comparing embedding models.

Provides:
  - RetrievalBenchmark  : builds corpus + query sets from KB documents
  - precision_at_k()    : standard IR metric
  - recall_at_k()       : standard IR metric
  - mean_reciprocal_rank(): MRR across queries
  - ndcg_at_k()         : nDCG for graded relevance
  - run_benchmark()     : evaluates one embedding model end-to-end
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────
# Result model
# ─────────────────────────────────────────────

@dataclass
class RetrievalResult:
    model_name: str
    precision_at_1:  float = 0.0
    precision_at_3:  float = 0.0
    precision_at_5:  float = 0.0
    recall_at_5:     float = 0.0
    mrr:             float = 0.0
    ndcg_at_5:       float = 0.0
    avg_latency_ms:  float = 0.0
    num_queries:     int   = 0

    def to_dict(self) -> dict:
        return {
            "model":            self.model_name,
            "precision@1":      round(self.precision_at_1, 4),
            "precision@3":      round(self.precision_at_3, 4),
            "precision@5":      round(self.precision_at_5, 4),
            "recall@5":         round(self.recall_at_5, 4),
            "mrr":              round(self.mrr, 4),
            "ndcg@5":           round(self.ndcg_at_5, 4),
            "latency_ms/query": round(self.avg_latency_ms, 1),
            "num_queries":      self.num_queries,
        }


# ─────────────────────────────────────────────
# IR metrics
# ─────────────────────────────────────────────

def precision_at_k(ranked_ids: List[str], relevant_ids: set, k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    top_k = ranked_ids[:k]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: List[str], relevant_ids: set, k: int) -> float:
    """Fraction of relevant docs retrieved in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = ranked_ids[:k]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(ranked_ids: List[str], relevant_ids: set, max_k: int = 10) -> float:
    """Reciprocal rank of the first relevant result."""
    for rank, doc_id in enumerate(ranked_ids[:max_k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: List[str], relevant_ids: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at k.
    Binary relevance: 1 if relevant, 0 otherwise.
    """
    def dcg(ids: List[str]) -> float:
        return sum(
            (1.0 / math.log2(i + 2)) if doc_id in relevant_ids else 0.0
            for i, doc_id in enumerate(ids[:k])
        )

    actual_dcg  = dcg(ranked_ids)
    # Ideal: all relevant docs ranked first
    ideal_ids   = list(relevant_ids) + [f"__pad_{i}__" for i in range(k)]
    ideal_dcg   = dcg(ideal_ids)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ─────────────────────────────────────────────
# Corpus & query container
# ─────────────────────────────────────────────

@dataclass
class RetrievalBenchmark:
    """
    Holds a corpus of documents and a set of queries with known-relevant doc IDs.

    corpus: list of {"id": str, "content": str}
    queries: list of {"query": str, "relevant_doc_ids": list[str]}
    """
    corpus:  List[Dict] = field(default_factory=list)
    queries: List[Dict] = field(default_factory=list)

    def add_doc(self, doc_id: str, content: str) -> None:
        self.corpus.append({"id": doc_id, "content": content})

    def add_query(self, query: str, relevant_ids: List[str]) -> None:
        self.queries.append({"query": query, "relevant_doc_ids": relevant_ids})

    @property
    def corpus_size(self) -> int:
        return len(self.corpus)

    @property
    def num_queries(self) -> int:
        return len(self.queries)


# ─────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────

def run_benchmark(
    model_name: str,
    benchmark: RetrievalBenchmark,
    k_values: Optional[List[int]] = None,
    batch_size: int = 64,
) -> RetrievalResult:
    """
    Evaluate one SentenceTransformer embedding model on the benchmark.

    Steps:
      1. Embed entire corpus (batched)
      2. For each query: embed → cosine rank → compute metrics
      3. Aggregate and return RetrievalResult
    """
    from sentence_transformers import SentenceTransformer

    k_values = k_values or [1, 3, 5]
    max_k    = max(k_values)

    model       = SentenceTransformer(model_name)
    corpus_ids  = [d["id"] for d in benchmark.corpus]
    corpus_text = [d["content"] for d in benchmark.corpus]

    # Encode corpus
    corpus_vecs = model.encode(
        corpus_text,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    p_at: Dict[int, List[float]] = {k: [] for k in k_values}
    recalls: List[float] = []
    mrrs:    List[float] = []
    ndcgs:   List[float] = []
    latencies: List[float] = []

    for q_item in benchmark.queries:
        relevant = set(q_item["relevant_doc_ids"])
        t0 = time.perf_counter()

        q_vec   = model.encode([q_item["query"]], normalize_embeddings=True)[0]
        scores  = corpus_vecs @ q_vec                        # cosine (already normalised)
        ranking = [corpus_ids[i] for i in np.argsort(-scores)]

        latencies.append((time.perf_counter() - t0) * 1000)

        for k in k_values:
            p_at[k].append(precision_at_k(ranking, relevant, k))

        recalls.append(recall_at_k(ranking, relevant, max_k))
        mrrs.append(mean_reciprocal_rank(ranking, relevant))
        ndcgs.append(ndcg_at_k(ranking, relevant, max_k))

    def avg(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return RetrievalResult(
        model_name=model_name,
        precision_at_1=avg(p_at.get(1, [])),
        precision_at_3=avg(p_at.get(3, [])),
        precision_at_5=avg(p_at.get(5, [])),
        recall_at_5=avg(recalls),
        mrr=avg(mrrs),
        ndcg_at_5=avg(ndcgs),
        avg_latency_ms=avg(latencies),
        num_queries=benchmark.num_queries,
    )


def compare_models(
    model_names: List[str],
    benchmark: RetrievalBenchmark,
    k_values: Optional[List[int]] = None,
) -> List[RetrievalResult]:
    """Run run_benchmark for multiple models and return sorted results."""
    results = []
    for name in model_names:
        result = run_benchmark(name, benchmark, k_values=k_values)
        results.append(result)
    return sorted(results, key=lambda r: r.mrr, reverse=True)
