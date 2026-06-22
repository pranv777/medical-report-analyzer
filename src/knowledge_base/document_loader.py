"""
src/knowledge_base/document_loader.py
Loads medical documents from various sources into a unified format
ready for chunking and embedding.

Supported sources:
  - Plain text files (.txt)
  - JSON files with PubMed-style records
  - PDF files (using PyMuPDF with OCR fallback)
  - Directory scan (recursive)

Each document is returned as a dict:
  {
    "content":  str,          # full text content
    "source":   str,          # file path or URL
    "doc_type": str,          # "pubmed" | "guideline" | "text" | "pdf"
    "metadata": dict,         # title, authors, year, etc.
  }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
# Individual loaders
# ─────────────────────────────────────────────

def load_txt(path: Path) -> List[Dict[str, Any]]:
    """Load a plain text file as a single document."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            return []
        return [{
            "content":  content,
            "source":   str(path),
            "doc_type": "text",
            "metadata": {"filename": path.name},
        }]
    except Exception as exc:
        log.warning(f"Failed to load {path}: {exc}")
        return []


def load_json_pubmed(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSON file containing PubMed-style records.

    Expected format::

        [
          {
            "pmid": "12345678",
            "title": "...",
            "abstract": "...",
            "authors": ["Smith J", ...],
            "year": "2023"
          }
        ]
    """
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning(f"Failed to parse JSON {path}: {exc}")
        return []

    if not isinstance(records, list):
        records = [records]

    docs = []
    for rec in records:
        title    = rec.get("title", "")
        abstract = rec.get("abstract", "")
        if not abstract.strip():   # skip records with no abstract
            continue
        content  = "\n\n".join(filter(None, [title, abstract])).strip()
        docs.append({
            "content":  content,
            "source":   rec.get("pmid", str(path)),
            "doc_type": "pubmed",
            "metadata": {
                "title":   title,
                "authors": rec.get("authors", [])[:3],
                "year":    rec.get("year", ""),
                "pmid":    rec.get("pmid", ""),
            },
        })
    return docs


def load_pdf(path: Path, ocr_fallback: bool = True) -> List[Dict[str, Any]]:
    """
    Load a PDF using PyMuPDF; falls back to OCR per-page if needed.
    Returns one document per page.
    """
    try:
        import fitz
    except ImportError:
        log.warning("PyMuPDF not installed — skipping PDF: %s", path)
        return []

    docs = []
    try:
        pdf = fitz.open(str(path))
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if len(text) < 50 and ocr_fallback:
                text = _ocr_page(page)
            if text.strip():
                docs.append({
                    "content":  text,
                    "source":   f"{path}#page{page_num}",
                    "doc_type": "pdf",
                    "metadata": {
                        "filename":   path.name,
                        "page_number": page_num,
                        "total_pages": len(pdf),
                    },
                })
        pdf.close()
    except Exception as exc:
        log.warning(f"Failed to load PDF {path}: {exc}")
    return docs


def _ocr_page(page) -> str:
    try:
        import pytesseract
        from PIL import Image
        import io
        import fitz

        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang="eng")
    except Exception:
        return ""


# ─────────────────────────────────────────────
# Directory scanner
# ─────────────────────────────────────────────

_LOADER_MAP = {
    ".txt":  load_txt,
    ".json": load_json_pubmed,
    ".pdf":  load_pdf,
}


def load_directory(
    directory: str | Path,
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Scan a directory and load all supported documents.

    Args:
        directory:  Path to scan.
        recursive:  Whether to recurse into subdirectories.
        extensions: Limit to these extensions (e.g. [".txt", ".json"]).
                    Defaults to all supported types.

    Returns:
        List of document dicts ready for chunking.
    """
    exts = extensions or list(_LOADER_MAP.keys())
    root = Path(directory)
    if not root.exists():
        log.warning(f"Directory not found: {root}")
        return []

    glob = root.rglob("*") if recursive else root.glob("*")
    docs: List[Dict[str, Any]] = []
    file_count = 0

    for path in glob:
        if path.suffix.lower() not in exts:
            continue
        loader = _LOADER_MAP.get(path.suffix.lower())
        if loader is None:
            continue
        loaded = loader(path)
        docs.extend(loaded)
        file_count += 1

    log.info(f"Loaded {len(docs)} document(s) from {file_count} file(s) in {root}")
    return docs


# ─────────────────────────────────────────────
# Streaming loader (memory-efficient for large corpora)
# ─────────────────────────────────────────────

def stream_directory(
    directory: str | Path,
    recursive: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator version of load_directory — yields one document at a time.
    Use for large corpora where loading all at once would exhaust RAM.
    """
    root = Path(directory)
    glob = root.rglob("*") if recursive else root.glob("*")

    for path in glob:
        loader = _LOADER_MAP.get(path.suffix.lower())
        if loader is None:
            continue
        for doc in loader(path):
            yield doc
