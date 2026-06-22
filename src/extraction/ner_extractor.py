"""
src/extraction/ner_extractor.py
Medical Named Entity Recognition using BioBERT (HuggingFace Transformers)
with SciSpacy as a secondary annotator / disambiguation layer.

Entity types extracted:
  DISEASE, SYMPTOM, MEDICATION, DOSAGE, LAB_TEST,
  LAB_VALUE, PROCEDURE, ANATOMY, MEDICAL_CONDITION

Output is a structured list of MedicalEntity objects that can be serialised
to JSON for downstream modules.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from src.utils.config import get_settings
from src.utils.helpers import clean_text, flatten_entities
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class MedicalEntity:
    text: str
    label: str
    start: int
    end: int
    score: float
    source: str                      # "biobert" | "spacy" | "merged"
    normalized: Optional[str] = None # UMLS / MeSH normalised form (future)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractionResult:
    raw_text: str
    entities: List[MedicalEntity] = field(default_factory=list)
    lab_values: List[Dict] = field(default_factory=list)   # parsed numeric findings
    entity_summary: Dict[str, List[str]] = field(default_factory=dict)


# ─────────────────────────────────────────────
# BioBERT NER wrapper
# ─────────────────────────────────────────────

class BioBERTExtractor:
    """
    Wraps a HuggingFace token-classification pipeline for BioBERT-based NER.
    Lazy-loads the model on first call so startup is fast.
    """

    _LABEL_MAP: Dict[str, str] = {
        "B-DISEASE": "DISEASE",  "I-DISEASE": "DISEASE",
        "B-CHEMICAL": "MEDICATION", "I-CHEMICAL": "MEDICATION",
        "B-GENE_OR_GENE_PRODUCT": "LAB_TEST",
        # Add full BC5CDR / i2b2 label map here
    }

    def __init__(self, config=None):
        self.cfg = config or get_settings().extraction
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline as hf_pipeline, AutoTokenizer

        log.info(f"Loading BioBERT model: {self.cfg.primary_model}")
        device = 0 if self.cfg.use_gpu else -1
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.primary_model,
            model_max_length=512,
        )
        
        try:
            self._pipeline = hf_pipeline(
                "ner",
                model=self.cfg.primary_model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=device,
            )
            log.info("BioBERT model loaded successfully")
        except Exception as exc:
            log.warning(
                f"Failed to load primary model ({exc}), "
                f"trying fallback: {self.cfg.fallback_model}"
            )
            fallback_tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.fallback_model,
                model_max_length=512,
            )
            self._pipeline = hf_pipeline(
                "ner",
                model=self.cfg.fallback_model,
                tokenizer=fallback_tokenizer,
                aggregation_strategy="simple",
                device=device,
            )

    def extract(self, text: str) -> List[MedicalEntity]:
        self._load_pipeline()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.primary_model)
        entities: List[MedicalEntity] = []

        # Tokenize and split into 512-token chunks with overlap
        words = text.split()
        chunk_size = 200   # words per chunk (safe under 512 tokens)
        overlap = 20

        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i: i + chunk_size])
            chunks.append((chunk, i))

        for chunk_text, word_offset in chunks:
            try:
                raw = self._pipeline(chunk_text)
                for item in raw:
                    score = float(item.get("score", 0.0))
                    if score < self.cfg.confidence_threshold:
                        continue
                    label = self._normalise_label(item["entity_group"])
                    if label not in self.cfg.entity_types:
                        continue
                    entities.append(
                        MedicalEntity(
                            text=item["word"].strip(),
                            label=label,
                            start=item["start"],
                            end=item["end"],
                            score=score,
                            source="biobert",
                        )
                    )
            except Exception as exc:
                log.warning(f"BioBERT chunk failed: {exc}")

        return entities

    def _normalise_label(self, raw_label: str) -> str:
        return self._LABEL_MAP.get(raw_label, raw_label.upper())

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into ~400 word chunks with 50-word overlap."""
        words = text.split()
        size, overlap = 400, 50
        chunks = []
        for i in range(0, len(words), size - overlap):
            chunks.append(" ".join(words[i : i + size]))
        return chunks or [text]


# ─────────────────────────────────────────────
# SciSpacy secondary annotator
# ─────────────────────────────────────────────

class SciSpacyExtractor:
    """Uses SciSpacy for complementary NER (anatomy, procedures, conditions)."""

    def __init__(self, config=None):
        self.cfg = config or get_settings().extraction
        self._nlp = None

    def _load(self):
        if self._nlp is not None:
            return
        import spacy

        log.info(f"Loading SciSpacy model: {self.cfg.spacy_model}")
        try:
            self._nlp = spacy.load(self.cfg.spacy_model)
        except OSError:
            raise OSError(
                f"SciSpacy model '{self.cfg.spacy_model}' not found. "
                "Install it with:\n"
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                "releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz"
            )

    def extract(self, text: str) -> List[MedicalEntity]:
        self._load()
        doc = self._nlp(text[:100_000])  # spacy limit guard
        entities = []
        for ent in doc.ents:
            entities.append(
                MedicalEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=1.0,
                    source="spacy",
                )
            )
        return entities


# ─────────────────────────────────────────────
# Lab value pattern extractor
# ─────────────────────────────────────────────

_LAB_PATTERN = re.compile(
    r"^(?P<test>[A-Za-z][A-Za-z0-9 \(\)/\-]{2,40}?)"
    r"\s{2,}"
    r"(?P<value>\d+\.?\d*)"
    r"\s{2,}"
    r"(?P<ref>\d+\.?\d*\s*-\s*\d+\.?\d*)?"
    r"\s*(?P<unit>[a-zA-Z/%µ\^][A-Za-z/%µ0-9\.\-\^/]*)?"
    r"(?:\s+(?P<flag>LOW|HIGH|NORMAL|ABNORMAL|CRITICAL))?",
    re.IGNORECASE,
)

_SKIP_LINES = {
    'patient', 'dob', 'gender', 'age', 'date', 'ref. no',
    'ordering', 'physician', 'test', 'result', 'reference',
    'units', 'flag', 'recommended',
}


def extract_lab_values(text: str) -> List[Dict]:
    results = []
    seen = set()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 10:
            continue

        # Skip metadata/header lines
        if any(skip in stripped.lower() for skip in _SKIP_LINES):
            continue

        # Must have multiple spaces (tabular format)
        if '  ' not in stripped:
            continue

        m = _LAB_PATTERN.match(stripped)
        if not m:
            continue

        test_name = m.group("test").strip()
        value = m.group("value")
        unit = (m.group("unit") or "").strip()
        flag = (m.group("flag") or "").strip().upper() or None

        # Parse reference range
        ref_low, ref_high = None, None
        ref_str = m.group("ref")
        if ref_str:
            parts = re.split(r'\s*-\s*', ref_str.strip())
            try:
                ref_low = float(parts[0])
                ref_high = float(parts[1])
            except (ValueError, IndexError):
                pass

        # Skip if test name looks like metadata
        key = test_name.lower()
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "test_name": test_name,
            "value": value,
            "numeric_value": float(value) if value else None,
            "unit": unit,
            "ref_low": ref_low,
            "ref_high": ref_high,
            "flag": flag,
            "span": (m.start(), m.end()),
        })

    return results


# ─────────────────────────────────────────────
# Unified NER pipeline
# ─────────────────────────────────────────────

class MedicalNERPipeline:
    """
    Orchestrates BioBERT + SciSpacy + regex lab extraction.

    Usage::

        pipeline = MedicalNERPipeline()
        result = pipeline.run(text)
        # result.entities — list of MedicalEntity
        # result.lab_values — list of dicts with test/value/unit
        # result.entity_summary — grouped by label
    """

    def __init__(self, config=None):
        self.cfg = config or get_settings().extraction
        self.biobert = BioBERTExtractor(config)
        self.scispacy = SciSpacyExtractor(config)

    def run(self, text: str) -> ExtractionResult:
        log.info("Running medical NER pipeline")
        raw_text = text  # preserve original for lab parsing
        text = clean_text(text)
        result = ExtractionResult(raw_text=text)

        # 1. BioBERT entities biobert_ents = self.biobert.extract(text)
        # 1. BioBERT entities
        try:
            biobert_ents = self.biobert.extract(text)
            log.debug(f"BioBERT: {len(biobert_ents)} entities")
        except Exception as exc:
            log.warning(f"BioBERT extraction skipped: {exc}")
            biobert_ents = []
            log.debug(f"BioBERT: {len(biobert_ents)} entities")

        # 2. SciSpacy entities
        try:
            spacy_ents = self.scispacy.extract(text)
            log.debug(f"SciSpacy: {len(spacy_ents)} entities")
        except Exception as exc:
            log.warning(f"SciSpacy extraction skipped: {exc}")
            spacy_ents = []

        # 3. Merge + deduplicate
        all_ents = [e.to_dict() for e in biobert_ents + spacy_ents]
        merged = flatten_entities(all_ents)
        result.entities = [MedicalEntity(**e) for e in merged]

        # 4. Regex lab values
        result.lab_values = extract_lab_values(raw_text)
        log.debug(f"Lab values extracted: {len(result.lab_values)}")

        # 5. Summary grouped by label
        summary: Dict[str, List[str]] = {}
        for ent in result.entities:
            summary.setdefault(ent.label, []).append(ent.text)
        result.entity_summary = {k: list(set(v)) for k, v in summary.items()}

        log.info(
            f"NER complete: {len(result.entities)} entities, "
            f"{len(result.lab_values)} lab values"
        )
        return result
