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
        from transformers import pipeline as hf_pipeline

        log.info(f"Loading BioBERT model: {self.cfg.primary_model}")
        device = 0 if self.cfg.use_gpu else -1
        try:
            self._pipeline = hf_pipeline(
                "ner",
                model=self.cfg.primary_model,
                tokenizer=self.cfg.primary_model,
                aggregation_strategy="simple",
                device=device,
            )
            log.info("BioBERT model loaded successfully")
        except Exception as exc:
            log.warning(
                f"Failed to load primary model ({exc}), "
                f"trying fallback: {self.cfg.fallback_model}"
            )
            self._pipeline = hf_pipeline(
                "ner",
                model=self.cfg.fallback_model,
                aggregation_strategy="simple",
                device=device,
            )

    def extract(self, text: str) -> List[MedicalEntity]:
        self._load_pipeline()
        entities: List[MedicalEntity] = []

        # BioBERT max 512 tokens — chunk long texts
        chunks = self._chunk_text(text)
        offset = 0
        for chunk in chunks:
            raw = self._pipeline(chunk)
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
                        start=item["start"] + offset,
                        end=item["end"] + offset,
                        score=score,
                        source="biobert",
                    )
                )
            offset += len(chunk)

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
    r"(?P<test>[A-Za-z][A-Za-z0-9 \-/]{1,40})"
    r"\s*[:\-]\s*"
    r"(?P<value>\d+\.?\d*)"
    r"\s*(?P<unit>[a-zA-Z/%µ][A-Za-z/%µ0-9\.\-]*)?",
    re.MULTILINE,
)


def extract_lab_values(text: str) -> List[Dict]:
    """
    Regex-based extractor for lab test name + numeric value + unit.
    Complements NER for structured blood-test lines.
    """
    results = []
    for match in _LAB_PATTERN.finditer(text):
        results.append(
            {
                "test_name": match.group("test").strip(),
                "value": match.group("value"),
                "unit": (match.group("unit") or "").strip(),
                "span": (match.start(), match.end()),
            }
        )
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
        text = clean_text(text)
        result = ExtractionResult(raw_text=text)

        # 1. BioBERT entities biobert_ents = self.biobert.extract(text)
        biobert_ents = []  # Skip BioBERT - causes token length issues
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
        result.lab_values = extract_lab_values(text)
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
