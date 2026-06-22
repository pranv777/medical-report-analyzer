"""
src/evaluation/ner_eval.py
Standalone NER evaluation utilities — token-level and span-level metrics.

Provides:
  - span_f1()        : exact span match (text + label must match)
  - token_f1()       : token-overlap F1 (partial credit)
  - per_type_report(): DataFrame with P/R/F1 per entity type
  - confusion_matrix(): label confusion for error analysis
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ─────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────

@dataclass
class SpanAnnotation:
    text: str
    label: str
    start: Optional[int] = None
    end: Optional[int] = None

    def key(self) -> Tuple[str, str]:
        """Normalised (text, label) key for exact match."""
        return (self.text.lower().strip(), self.label.upper())


# ─────────────────────────────────────────────
# Span-level F1
# ─────────────────────────────────────────────

def span_f1(
    predictions: List[List[SpanAnnotation]],
    ground_truth: List[List[SpanAnnotation]],
) -> Dict[str, float]:
    """
    Compute overall micro-averaged Precision, Recall, F1 using exact span match.

    Args:
        predictions:  List of per-sample prediction spans.
        ground_truth: List of per-sample gold spans.

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    tp = fp = fn = 0
    for preds, golds in zip(predictions, ground_truth):
        pred_keys = {s.key() for s in preds}
        gold_keys = {s.key() for s in golds}
        tp += len(pred_keys & gold_keys)
        fp += len(pred_keys - gold_keys)
        fn += len(gold_keys - pred_keys)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ─────────────────────────────────────────────
# Per-type report
# ─────────────────────────────────────────────

def per_type_report(
    predictions: List[List[SpanAnnotation]],
    ground_truth: List[List[SpanAnnotation]],
) -> pd.DataFrame:
    """
    Compute Precision / Recall / F1 / Support for each entity type.

    Returns a DataFrame indexed by entity type.
    """
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)

    for preds, golds in zip(predictions, ground_truth):
        pred_map: Dict[str, set] = defaultdict(set)
        gold_map: Dict[str, set] = defaultdict(set)

        for s in preds:
            pred_map[s.label.upper()].add(s.text.lower().strip())
        for s in golds:
            gold_map[s.label.upper()].add(s.text.lower().strip())

        all_labels = set(pred_map) | set(gold_map)
        for label in all_labels:
            p_set = pred_map[label]
            g_set = gold_map[label]
            tp[label] += len(p_set & g_set)
            fp[label] += len(p_set - g_set)
            fn[label] += len(g_set - p_set)

    rows = []
    for label in sorted(set(tp) | set(fp) | set(fn)):
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = tp[label] + fn[label]
        rows.append({
            "entity_type": label,
            "precision":   round(p, 4),
            "recall":      round(r, 4),
            "f1":          round(f, 4),
            "support":     support,
            "tp": tp[label], "fp": fp[label], "fn": fn[label],
        })

    df = pd.DataFrame(rows).set_index("entity_type")

    # Macro average row
    if not df.empty:
        macro = df[["precision", "recall", "f1"]].mean().round(4)
        macro_row = pd.DataFrame(
            [{**macro.to_dict(), "support": df["support"].sum(),
              "tp": df["tp"].sum(), "fp": df["fp"].sum(), "fn": df["fn"].sum()}],
            index=["MACRO AVG"],
        )
        df = pd.concat([df, macro_row])

    return df


# ─────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────

def label_confusion(
    predictions: List[List[SpanAnnotation]],
    ground_truth: List[List[SpanAnnotation]],
) -> pd.DataFrame:
    """
    Build a confusion matrix: rows = gold labels, cols = predicted labels.
    Counts how often a gold entity of type A was predicted as type B.
    Useful for finding systematic label errors.
    """
    counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for preds, golds in zip(predictions, ground_truth):
        pred_text_to_label: Dict[str, str] = {
            s.text.lower().strip(): s.label.upper() for s in preds
        }
        for gold in golds:
            gold_text  = gold.text.lower().strip()
            gold_label = gold.label.upper()
            pred_label = pred_text_to_label.get(gold_text, "MISSED")
            counts[(gold_label, pred_label)] += 1

    all_labels = sorted({k[0] for k in counts} | {k[1] for k in counts} - {"MISSED"})
    all_labels_with_missed = all_labels + ["MISSED"]

    matrix = pd.DataFrame(0, index=all_labels, columns=all_labels_with_missed)
    for (gold, pred), count in counts.items():
        if gold in matrix.index and pred in matrix.columns:
            matrix.loc[gold, pred] += count

    return matrix


# ─────────────────────────────────────────────
# Token-level F1 (partial credit)
# ─────────────────────────────────────────────

def token_f1_score(pred_text: str, gold_text: str) -> float:
    """
    Compute token-level F1 between a predicted span and a gold span.
    Gives partial credit for overlapping tokens (used in QA-style evaluation).
    """
    pred_tokens = set(pred_text.lower().split())
    gold_tokens = set(gold_text.lower().split())
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)
