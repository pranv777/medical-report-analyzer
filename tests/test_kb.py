"""
tests/test_kb.py
Unit tests for the knowledge base builder, document loader, and retriever.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLE_DOCS = [
    {
        "content": "Iron deficiency anemia is caused by low ferritin and low hemoglobin. "
                   "Treatment includes oral ferrous sulfate supplementation.",
        "source":   "test_doc_1",
        "doc_type": "text",
        "metadata": {"title": "Anemia Overview"},
    },
    {
        "content": "Diabetes mellitus is diagnosed when HbA1c exceeds 6.5%. "
                   "Metformin is the first-line treatment.",
        "source":   "test_doc_2",
        "doc_type": "text",
        "metadata": {"title": "Diabetes Guide"},
    },
    {
        "content": "Hypothyroidism is characterised by elevated TSH and low free T4. "
                   "Levothyroxine is the standard treatment.",
        "source":   "test_doc_3",
        "doc_type": "text",
        "metadata": {"title": "Thyroid Disorders"},
    },
]


# ─────────────────────────────────────────────
# Document loader tests
# ─────────────────────────────────────────────

class TestDocumentLoader:
    def test_load_txt_file(self, tmp_path):
        from src.knowledge_base.document_loader import load_txt
        f = tmp_path / "test.txt"
        f.write_text("Anemia is a low hemoglobin condition.")
        docs = load_txt(f)
        assert len(docs) == 1
        assert "Anemia" in docs[0]["content"]
        assert docs[0]["doc_type"] == "text"

    def test_load_txt_empty_file_returns_empty(self, tmp_path):
        from src.knowledge_base.document_loader import load_txt
        f = tmp_path / "empty.txt"
        f.write_text("")
        docs = load_txt(f)
        assert docs == []

    def test_load_json_pubmed(self, tmp_path):
        from src.knowledge_base.document_loader import load_json_pubmed
        records = [
            {"pmid": "12345", "title": "Anemia review", "abstract": "Low hemoglobin.", "year": "2023"},
            {"pmid": "67890", "title": "Diabetes", "abstract": "High glucose.", "year": "2022"},
        ]
        f = tmp_path / "pubmed.json"
        f.write_text(json.dumps(records))
        docs = load_json_pubmed(f)
        assert len(docs) == 2
        assert all(d["doc_type"] == "pubmed" for d in docs)
        assert docs[0]["metadata"]["pmid"] == "12345"

    def test_load_json_skips_records_without_abstract(self, tmp_path):
        from src.knowledge_base.document_loader import load_json_pubmed
        records = [
            {"pmid": "111", "title": "Title only"},   # no abstract
            {"pmid": "222", "title": "Has abstract", "abstract": "Some content"},
        ]
        f = tmp_path / "partial.json"
        f.write_text(json.dumps(records))
        docs = load_json_pubmed(f)
        assert len(docs) == 1
        assert docs[0]["metadata"]["pmid"] == "222"

    def test_load_directory_txt(self, tmp_path):
        from src.knowledge_base.document_loader import load_directory
        (tmp_path / "a.txt").write_text("Content A about anemia.")
        (tmp_path / "b.txt").write_text("Content B about diabetes.")
        (tmp_path / "skip.csv").write_text("col1,col2")  # unsupported
        docs = load_directory(tmp_path)
        assert len(docs) == 2

    def test_load_directory_nonexistent_returns_empty(self):
        from src.knowledge_base.document_loader import load_directory
        docs = load_directory("/nonexistent/path/xyz")
        assert docs == []

    def test_stream_directory_yields_docs(self, tmp_path):
        from src.knowledge_base.document_loader import stream_directory
        (tmp_path / "doc1.txt").write_text("Medical content 1.")
        (tmp_path / "doc2.txt").write_text("Medical content 2.")
        streamed = list(stream_directory(tmp_path))
        assert len(streamed) == 2


# ─────────────────────────────────────────────
# KnowledgeBaseBuilder tests
# ─────────────────────────────────────────────

class TestKnowledgeBaseBuilder:
    @pytest.fixture
    def tmp_kb(self, tmp_path):
        """Return a KnowledgeBaseBuilder pointing at a temp directory."""
        pytest.importorskip("langchain", reason="langchain not installed")
        pytest.importorskip("chromadb", reason="chromadb not installed")
        pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
        from src.knowledge_base.kb_builder import KnowledgeBaseBuilder
        from src.utils.config import get_settings
        get_settings.cache_clear()
        cfg = get_settings()
        # Override paths for isolation
        cfg.knowledge_base.chromadb_path = str(tmp_path / "chroma_db")
        cfg.knowledge_base.collection_name = "test_collection"
        return KnowledgeBaseBuilder(cfg.knowledge_base)

    def test_add_documents_returns_count(self, tmp_kb):
        added = tmp_kb.add_documents(SAMPLE_DOCS)
        assert added > 0

    def test_add_same_documents_twice_skips(self, tmp_kb):
        added1 = tmp_kb.add_documents(SAMPLE_DOCS)
        added2 = tmp_kb.add_documents(SAMPLE_DOCS)
        assert added1 > 0
        assert added2 == 0   # already indexed

    def test_collection_stats_has_required_fields(self, tmp_kb):
        tmp_kb.add_documents(SAMPLE_DOCS)
        stats = tmp_kb.get_collection_stats()
        assert "total_documents" in stats
        assert "collection_name" in stats
        assert stats["total_documents"] > 0

    def test_collection_grows_with_new_docs(self, tmp_kb):
        tmp_kb.add_documents(SAMPLE_DOCS[:1])
        stats1 = tmp_kb.get_collection_stats()

        new_doc = [{
            "content": "Vitamin D deficiency causes bone loss and fatigue.",
            "source": "new_unique_doc",
            "doc_type": "text",
        }]
        tmp_kb.add_documents(new_doc)
        stats2 = tmp_kb.get_collection_stats()
        assert stats2["total_documents"] > stats1["total_documents"]


# ─────────────────────────────────────────────
# MedicalRetriever tests
# ─────────────────────────────────────────────

class TestMedicalRetriever:
    @pytest.fixture
    def retriever(self, tmp_path):
        pytest.importorskip("langchain", reason="langchain not installed")
        pytest.importorskip("chromadb", reason="chromadb not installed")
        pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
        from src.knowledge_base.kb_builder import KnowledgeBaseBuilder, MedicalRetriever
        from src.utils.config import get_settings
        get_settings.cache_clear()
        cfg = get_settings()
        cfg.knowledge_base.chromadb_path = str(tmp_path / "chroma_db")
        cfg.knowledge_base.collection_name = "retrieval_test"

        # Pre-populate
        builder = KnowledgeBaseBuilder(cfg.knowledge_base)
        builder.add_documents(SAMPLE_DOCS)
        return MedicalRetriever(cfg.knowledge_base)

    def test_retrieve_returns_list(self, retriever):
        results = retriever.retrieve("hemoglobin iron anemia", k=2)
        assert isinstance(results, list)

    def test_retrieve_respects_k(self, retriever):
        results = retriever.retrieve("hemoglobin", k=1)
        assert len(results) <= 1

    def test_retrieve_has_required_fields(self, retriever):
        results = retriever.retrieve("diabetes HbA1c", k=2)
        for r in results:
            assert "content" in r
            assert "metadata" in r
            assert "score" in r

    def test_retrieve_scores_in_range(self, retriever):
        results = retriever.retrieve("thyroid TSH", k=3)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_retrieve_relevant_doc_for_anemia_query(self, retriever):
        results = retriever.retrieve("ferritin low hemoglobin iron deficiency")
        contents = " ".join(r["content"] for r in results).lower()
        assert "anemia" in contents or "hemoglobin" in contents or "ferritin" in contents


# ─────────────────────────────────────────────
# Embedding eval IR metrics
# ─────────────────────────────────────────────

class TestIRMetrics:
    def test_precision_at_k_perfect(self):
        from src.evaluation.embedding_eval import precision_at_k
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)

    def test_precision_at_k_zero(self):
        from src.evaluation.embedding_eval import precision_at_k
        assert precision_at_k(["x", "y", "z"], {"a", "b"}, k=3) == pytest.approx(0.0)

    def test_precision_at_k_partial(self):
        from src.evaluation.embedding_eval import precision_at_k
        score = precision_at_k(["a", "x", "b"], {"a", "b"}, k=3)
        assert score == pytest.approx(2 / 3)

    def test_recall_at_k(self):
        from src.evaluation.embedding_eval import recall_at_k
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=2) == pytest.approx(1.0)
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == pytest.approx(0.5)

    def test_mrr_first_position(self):
        from src.evaluation.embedding_eval import mean_reciprocal_rank
        assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_mrr_second_position(self):
        from src.evaluation.embedding_eval import mean_reciprocal_rank
        assert mean_reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)

    def test_mrr_not_found(self):
        from src.evaluation.embedding_eval import mean_reciprocal_rank
        assert mean_reciprocal_rank(["x", "y", "z"], {"a"}) == pytest.approx(0.0)

    def test_ndcg_perfect(self):
        from src.evaluation.embedding_eval import ndcg_at_k
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_ndcg_zero(self):
        from src.evaluation.embedding_eval import ndcg_at_k
        assert ndcg_at_k(["x", "y"], {"a", "b"}, k=2) == pytest.approx(0.0)
