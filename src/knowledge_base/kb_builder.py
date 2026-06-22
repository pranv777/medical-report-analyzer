"""
src/knowledge_base/kb_builder.py
Builds and manages the medical RAG knowledge base.

Responsibilities:
  1. Load source documents (PubMed abstracts, WHO/NIH text files, custom PDFs)
  2. Chunk documents using LangChain text splitters
  3. Generate embeddings with a Sentence-Transformer model
  4. Persist to a ChromaDB vector store
  5. Expose a retriever for the RAG pipeline

Supports incremental updates — already-embedded docs are skipped via hash check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import get_settings
from src.utils.helpers import ensure_dir, load_json, save_json
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
# Document loader helpers
# ─────────────────────────────────────────────

def _load_text_file(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{"content": text, "source": str(path), "doc_type": "text"}]


def _load_json_documents(path: Path) -> List[Dict[str, Any]]:
    """
    Expects a JSON list of objects with at least {"title": ..., "abstract": ...}
    as produced by the PubMed fetcher script.
    """
    records = load_json(path)
    docs = []
    for rec in records:
        content = "\n".join(filter(None, [rec.get("title"), rec.get("abstract")]))
        if content.strip():
            docs.append({
                "content": content,
                "source": rec.get("pmid", str(path)),
                "doc_type": "pubmed",
                "metadata": {
                    "title":   rec.get("title", ""),
                    "authors": rec.get("authors", []),
                    "year":    rec.get("year", ""),
                    "pmid":    rec.get("pmid", ""),
                },
            })
    return docs


def _doc_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


# ─────────────────────────────────────────────
# Knowledge Base Builder
# ─────────────────────────────────────────────

class KnowledgeBaseBuilder:
    """
    Builds the ChromaDB vector store from medical source documents.

    Usage::

        builder = KnowledgeBaseBuilder()
        builder.build(source_dirs=["data/knowledge_base/custom/"])
        # Later — incremental add
        builder.add_documents([{"content": "...", "source": "who_2024.txt"}])
    """

    def __init__(self, config=None):
        self.cfg     = config or get_settings().knowledge_base
        self._client = None
        self._collection = None
        self._embedder   = None
        self._splitter   = None
        self._hashes_path = Path(self.cfg.chromadb_path) / "indexed_hashes.json"

    # ── lazy loaders ─────────────────────────

    def _get_client(self):
        if self._client is None:
            import chromadb

            ensure_dir(self.cfg.chromadb_path)
            self._client = chromadb.PersistentClient(path=self.cfg.chromadb_path)
            log.info(f"ChromaDB client at: {self.cfg.chromadb_path}")
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.cfg.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            log.info(
                f"Collection '{self.cfg.collection_name}' "
                f"({self._collection.count()} docs)"
            )
        return self._collection

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            log.info(f"Loading embedding model: {self.cfg.embedding_model}")
            self._embedder = SentenceTransformer(self.cfg.embedding_model)
        return self._embedder

    def _get_splitter(self):
        if self._splitter is None:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            c = self.cfg.chunking
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=c.chunk_size,
                chunk_overlap=c.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        return self._splitter

    # ── indexed hash tracking ─────────────────

    def _load_hashes(self) -> set:
        if self._hashes_path.exists():
            return set(load_json(self._hashes_path))
        return set()

    def _save_hashes(self, hashes: set) -> None:
        ensure_dir(self._hashes_path.parent)
        save_json(list(hashes), self._hashes_path)

    # ── public API ────────────────────────────

    def build(self, source_dirs: Optional[List[str]] = None) -> int:
        """
        Scan source directories and index all new documents.
        Returns number of new chunks added.
        """
        dirs = source_dirs or [self.cfg.sources.get("custom_documents_path", "data/knowledge_base/custom/")]
        documents: List[Dict] = []

        for dir_str in dirs:
            d = Path(dir_str)
            if not d.exists():
                log.warning(f"Source directory not found: {d}")
                continue
            for file in d.rglob("*"):
                if file.suffix == ".txt":
                    documents.extend(_load_text_file(file))
                elif file.suffix == ".json":
                    documents.extend(_load_json_documents(file))

        log.info(f"Found {len(documents)} source document(s)")
        return self.add_documents(documents)

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Chunk, embed, and add documents to ChromaDB.
        Skips documents already indexed (by content hash).
        Returns number of new chunks added.
        """
        splitter    = self._get_splitter()
        embedder    = self._get_embedder()
        collection  = self._get_collection()
        known       = self._load_hashes()

        ids, texts, embeddings, metadatas = [], [], [], []

        for doc in documents:
            content  = doc.get("content", "")
            source   = doc.get("source", "unknown")
            doc_meta = doc.get("metadata", {})

            chunks = splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                h = _doc_hash(chunk)
                if h in known:
                    continue
                chunk_id = f"{h}_{i}"
                ids.append(chunk_id)
                texts.append(chunk)
                metadatas.append({
                    "source":   source,
                    "chunk_idx": i,
                    "doc_type": doc.get("doc_type", "custom"),
                    **{k: str(v) for k, v in doc_meta.items()},
                })
                known.add(h)

        if not ids:
            log.info("No new documents to index.")
            return 0

        log.info(f"Embedding {len(ids)} new chunk(s)…")
        vecs = embedder.encode(texts, show_progress_bar=True, batch_size=32).tolist()

        # ChromaDB batch upsert (max 5000 per call)
        batch = 500
        for start in range(0, len(ids), batch):
            collection.upsert(
                ids=ids[start : start + batch],
                documents=texts[start : start + batch],
                embeddings=vecs[start : start + batch],
                metadatas=metadatas[start : start + batch],
            )

        self._save_hashes(known)
        log.info(f"Indexed {len(ids)} new chunk(s). Total: {collection.count()}")
        return len(ids)

    def get_collection_stats(self) -> Dict:
        col = self._get_collection()
        return {
            "collection_name": self.cfg.collection_name,
            "total_documents": col.count(),
            "chromadb_path":   self.cfg.chromadb_path,
            "embedding_model": self.cfg.embedding_model,
        }


# ─────────────────────────────────────────────
# LangChain-compatible retriever wrapper
# ─────────────────────────────────────────────

class MedicalRetriever:
    """
    Wraps ChromaDB collection into a simple retriever usable both standalone
    and inside the LangChain RAG pipeline.

    Usage::

        retriever = MedicalRetriever()
        docs = retriever.retrieve("What causes low hemoglobin?", k=5)
    """

    def __init__(self, config=None):
        self.cfg      = config or get_settings().knowledge_base
        self._builder = KnowledgeBaseBuilder(config)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = k or self.cfg.retrieval.top_k
        embedder   = self._builder._get_embedder()
        collection = self._builder._get_collection()

        query_vec = embedder.encode([query])[0].tolist()
        results   = collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1 - dist  # cosine similarity from cosine distance
            if score >= self.cfg.retrieval.score_threshold:
                docs.append({"content": text, "metadata": meta, "score": round(score, 4)})

        log.debug(f"Retrieved {len(docs)} chunk(s) for query: '{query[:60]}…'")
        return docs

    def as_langchain_retriever(self):
        """Return a LangChain BaseRetriever-compatible object."""
        from langchain.schema import BaseRetriever, Document
        from langchain.callbacks.manager import CallbackManagerForRetrieverRun

        outer = self

        class _LCRetriever(BaseRetriever):
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                raw = outer.retrieve(query)
                return [
                    Document(page_content=r["content"], metadata=r["metadata"])
                    for r in raw
                ]

        return _LCRetriever()
