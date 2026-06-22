"""
tests/test_ner.py
Unit tests for NER extraction, lab value parsing, and entity merging.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLE_TEXT = (
    "Patient has hemoglobin 10.5 g/dL indicating iron deficiency anemia. "
    "Started on ferrous sulfate 325 mg twice daily. "
    "Fasting glucose 126 mg/dL. HbA1c 7.2%. "
    "TSH 3.2 mIU/L. Total cholesterol 235 mg/dL."
)


# ─────────────────────────────────────────────
# extract_lab_values (regex)
# ─────────────────────────────────────────────

class TestExtractLabValues:
    def test_extracts_colon_separated(self):
        from src.extraction.ner_extractor import extract_lab_values
        text = "Hemoglobin: 10.5 g/dL\nGlucose: 115 mg/dL"
        results = extract_lab_values(text)
        assert len(results) >= 2

    def test_extracts_numeric_value(self):
        from src.extraction.ner_extractor import extract_lab_values
        results = extract_lab_values("TSH: 3.2 mIU/L")
        assert any(r["value"] == "3.2" for r in results)

    def test_extracts_unit(self):
        from src.extraction.ner_extractor import extract_lab_values
        results = extract_lab_values("Creatinine: 0.88 mg/dL")
        assert results[0]["unit"].lower() in ("mg/dl", "mg/dl")

    def test_skips_non_numeric(self):
        from src.extraction.ner_extractor import extract_lab_values
        results = extract_lab_values("Patient name: John Doe")
        assert all(r["value"].replace(".", "").isdigit() for r in results)

    def test_empty_returns_empty(self):
        from src.extraction.ner_extractor import extract_lab_values
        assert extract_lab_values("") == []

    def test_multiple_on_same_line(self):
        from src.extraction.ner_extractor import extract_lab_values
        text = "Glucose: 115 mg/dL  HbA1c: 6.2%"
        results = extract_lab_values(text)
        assert len(results) >= 1


# ─────────────────────────────────────────────
# MedicalEntity
# ─────────────────────────────────────────────

class TestMedicalEntity:
    def test_to_dict_has_required_keys(self):
        from src.extraction.ner_extractor import MedicalEntity
        ent = MedicalEntity(
            text="anemia", label="DISEASE",
            start=0, end=6, score=0.92, source="biobert"
        )
        d = ent.to_dict()
        for key in ["text", "label", "start", "end", "score", "source"]:
            assert key in d

    def test_to_dict_serialisable(self):
        import json
        from src.extraction.ner_extractor import MedicalEntity
        ent = MedicalEntity("hemoglobin", "LAB_TEST", 0, 10, 0.88, "biobert")
        json.dumps(ent.to_dict())  # should not raise


# ─────────────────────────────────────────────
# ExtractionResult
# ─────────────────────────────────────────────

class TestExtractionResult:
    def test_default_fields(self):
        from src.extraction.ner_extractor import ExtractionResult
        result = ExtractionResult(raw_text=SAMPLE_TEXT)
        assert result.entities == []
        assert result.lab_values == []
        assert result.entity_summary == {}

    def test_lab_values_populated_by_regex(self):
        from src.extraction.ner_extractor import extract_lab_values
        # extract_lab_values requires colon/whitespace-separated format
        text = "Hemoglobin: 10.5 g/dL\nGlucose: 126 mg/dL\nTSH: 3.2 mIU/L"
        result = extract_lab_values(text)
        assert len(result) > 0


# ─────────────────────────────────────────────
# flatten_entities
# ─────────────────────────────────────────────

class TestFlattenEntities:
    def test_keeps_highest_score_duplicate(self):
        from src.utils.helpers import flatten_entities
        entities = [
            {"text": "diabetes", "label": "DISEASE", "score": 0.80},
            {"text": "diabetes", "label": "DISEASE", "score": 0.95},
            {"text": "insulin", "label": "MEDICATION", "score": 0.88},
        ]
        flat = flatten_entities(entities)
        assert len(flat) == 2
        diab = next(e for e in flat if e["text"] == "diabetes")
        assert diab["score"] == 0.95

    def test_different_labels_not_merged(self):
        from src.utils.helpers import flatten_entities
        entities = [
            {"text": "glucose", "label": "LAB_TEST", "score": 0.90},
            {"text": "glucose", "label": "DISEASE", "score": 0.75},
        ]
        flat = flatten_entities(entities)
        assert len(flat) == 2

    def test_empty_input(self):
        from src.utils.helpers import flatten_entities
        assert flatten_entities([]) == []

    def test_single_entity_unchanged(self):
        from src.utils.helpers import flatten_entities
        ent = {"text": "anemia", "label": "DISEASE", "score": 0.88}
        flat = flatten_entities([ent])
        assert flat[0]["text"] == "anemia"


# ─────────────────────────────────────────────
# NER evaluation metrics
# ─────────────────────────────────────────────

class TestNEREvalMetrics:
    def test_span_f1_perfect(self):
        from src.evaluation.ner_eval import span_f1, SpanAnnotation
        preds  = [[SpanAnnotation("anemia", "DISEASE")]]
        golds  = [[SpanAnnotation("anemia", "DISEASE")]]
        result = span_f1(preds, golds)
        assert result["f1"] == pytest.approx(1.0)

    def test_span_f1_zero(self):
        from src.evaluation.ner_eval import span_f1, SpanAnnotation
        preds  = [[SpanAnnotation("diabetes", "DISEASE")]]
        golds  = [[SpanAnnotation("anemia", "DISEASE")]]
        result = span_f1(preds, golds)
        assert result["f1"] == pytest.approx(0.0)

    def test_span_f1_partial(self):
        from src.evaluation.ner_eval import span_f1, SpanAnnotation
        preds = [[SpanAnnotation("anemia", "DISEASE"), SpanAnnotation("wrong", "DISEASE")]]
        golds = [[SpanAnnotation("anemia", "DISEASE"), SpanAnnotation("glucose", "LAB_TEST")]]
        result = span_f1(preds, golds)
        assert 0.0 < result["f1"] < 1.0

    def test_per_type_report_returns_dataframe(self):
        from src.evaluation.ner_eval import per_type_report, SpanAnnotation
        import pandas as pd
        preds = [[SpanAnnotation("anemia", "DISEASE"), SpanAnnotation("glucose", "LAB_TEST")]]
        golds = [[SpanAnnotation("anemia", "DISEASE"), SpanAnnotation("insulin", "MEDICATION")]]
        df = per_type_report(preds, golds)
        assert isinstance(df, pd.DataFrame)
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1" in df.columns

    def test_token_f1_exact_match(self):
        from src.evaluation.ner_eval import token_f1_score
        assert token_f1_score("iron deficiency anemia", "iron deficiency anemia") == pytest.approx(1.0)

    def test_token_f1_no_overlap(self):
        from src.evaluation.ner_eval import token_f1_score
        assert token_f1_score("diabetes mellitus", "iron deficiency") == pytest.approx(0.0)

    def test_token_f1_partial(self):
        from src.evaluation.ner_eval import token_f1_score
        score = token_f1_score("iron deficiency anemia", "iron deficiency")
        assert 0.0 < score < 1.0
