"""
src/utils/helpers.py
Shared utility functions used across modules.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise whitespace, remove control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 4096) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "… [truncated]"


def split_into_sentences(text: str) -> List[str]:
    """Simple sentence splitter (NLTK-free fallback)."""
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    return [s.strip() for s in sentence_endings.split(text) if s.strip()]


# ─────────────────────────────────────────────
# Number / unit parsing
# ─────────────────────────────────────────────

def extract_numeric(value_str: str) -> Optional[float]:
    """Parse numeric value from strings like '10.5 g/dL' → 10.5."""
    match = re.search(r"[-+]?\d*\.?\d+", str(value_str))
    return float(match.group()) if match else None


def normalize_unit(unit: str) -> str:
    """Lowercase and strip whitespace from unit strings."""
    return unit.strip().lower()


# ─────────────────────────────────────────────
# File utilities
# ─────────────────────────────────────────────

def file_hash(path: str | Path) -> str:
    """SHA-256 hash of a file (for caching & deduplication)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Timing decorator
# ─────────────────────────────────────────────

def timeit(func):
    """Decorator — logs wall-clock time of a function call."""
    import functools
    from src.utils.logger import get_logger
    _log = get_logger(__name__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _log.debug(f"{func.__qualname__} finished in {elapsed:.3f}s")
        return result

    return wrapper


# ─────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────

def flatten_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate entity list by (text, label) keeping highest confidence.
    """
    seen: Dict[tuple, Dict] = {}
    for ent in entities:
        key = (ent.get("text", "").lower(), ent.get("label", ""))
        existing = seen.get(key)
        if existing is None or ent.get("score", 0) > existing.get("score", 0):
            seen[key] = ent
    return list(seen.values())
