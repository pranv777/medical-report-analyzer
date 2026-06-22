"""
src/ingestion/report_ingestion.py
Extracts clean text from PDF reports (native text layer or OCR fallback)
and image-based reports.

Supported inputs:
  • Digitally-born PDFs   — PyMuPDF fast path
  • Scanned PDFs          — pdf2image + Tesseract OCR
  • Images (PNG/JPG/TIFF) — Pillow + Tesseract OCR
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.utils.config import get_settings
from src.utils.helpers import clean_text, file_hash
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class PageResult:
    page_number: int
    text: str
    extraction_method: str          # "native" | "ocr"
    confidence: Optional[float] = None


@dataclass
class IngestionResult:
    file_path: str
    file_hash: str
    file_type: str
    pages: List[PageResult] = field(default_factory=list)
    full_text: str = ""
    metadata: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.full_text) and not self.errors


# ─────────────────────────────────────────────
# Ingestion engine
# ─────────────────────────────────────────────

class ReportIngestion:
    """
    Entry point for report ingestion.

    Usage::

        ingestion = ReportIngestion()
        result = ingestion.ingest("path/to/report.pdf")
        print(result.full_text)
    """

    def __init__(self, config=None):
        self.cfg = config or get_settings().ingestion
        self._check_dependencies()

    # ── public ──────────────────────────────

    def ingest(self, file_path: str | Path) -> IngestionResult:
        path = Path(file_path).resolve()

        suffix = path.suffix.lower().lstrip(".")

        # Handle plain text files directly
        if suffix == "txt":
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            return IngestionResult(
                file_path=str(path),
                file_hash=file_hash(path),
                file_type="txt",
                pages=[PageResult(page_number=1, text=text, extraction_method="native")],
                full_text=text,
                metadata={"page_count": 1, "char_count": len(text)},
            )

        if suffix not in self.cfg.supported_formats:
            raise ValueError(
                f"Unsupported format '{suffix}'. "
                f"Supported: {self.cfg.supported_formats}"
            )

        if not path.exists():
            raise FileNotFoundError(f"Report not found: {path}")

        log.info(f"Ingesting report: {path.name}")
        result = IngestionResult(
            file_path=str(path),
            file_hash=file_hash(path),
            file_type=suffix,
        )

        try:
            if suffix == "pdf":
                self._ingest_pdf(path, result)
            else:
                self._ingest_image(path, result)

            result.full_text = self._assemble_text(result.pages)
            log.info(
                f"Ingested {len(result.pages)} page(s), "
                f"{len(result.full_text)} chars"
            )
        except Exception as exc:
            log.error(f"Ingestion failed: {exc}")
            result.errors.append(str(exc))

        return result

    # ── PDF path ────────────────────────────

    def _ingest_pdf(self, path: Path, result: IngestionResult) -> None:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        result.metadata.update(
            {
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
            }
        )

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            if len(text) >= self.cfg.min_text_length:
                # Good native text layer
                result.pages.append(
                    PageResult(
                        page_number=page_num,
                        text=clean_text(text),
                        extraction_method="native",
                    )
                )
            elif self.cfg.fallback_to_ocr:
                # Scanned page — render to image and OCR
                log.debug(f"Page {page_num}: native text too short, running OCR")
                ocr_text = self._ocr_pdf_page(page)
                result.pages.append(
                    PageResult(
                        page_number=page_num,
                        text=clean_text(ocr_text),
                        extraction_method="ocr",
                    )
                )
            else:
                log.warning(f"Page {page_num}: skipped (no text, OCR disabled)")

        doc.close()

    def _ocr_pdf_page(self, page) -> str:
        """Render a PyMuPDF page to an image and OCR it."""
        import pytesseract
        from PIL import Image
        import io

        dpi = self.cfg.ocr.dpi
        mat = __import__("fitz").Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        if self.cfg.ocr.preprocess:
            img = self._preprocess_image(img)

        return pytesseract.image_to_string(img, lang=self.cfg.ocr.language)

    # ── Image path ──────────────────────────

    def _ingest_image(self, path: Path, result: IngestionResult) -> None:
        import pytesseract
        from PIL import Image

        img = Image.open(path)
        result.metadata["mode"] = img.mode
        result.metadata["size"] = img.size

        if self.cfg.ocr.preprocess:
            img = self._preprocess_image(img)

        text = pytesseract.image_to_string(img, lang=self.cfg.ocr.language)
        result.pages.append(
            PageResult(page_number=1, text=clean_text(text), extraction_method="ocr")
        )

    # ── Image preprocessing ─────────────────

    @staticmethod
    def _preprocess_image(img):
        """Convert to grayscale and threshold for better OCR accuracy."""
        from PIL import Image, ImageFilter

        if img.mode != "L":
            img = img.convert("L")
        img = img.filter(ImageFilter.SHARPEN)
        return img

    # ── Text assembly ────────────────────────

    @staticmethod
    def _assemble_text(pages: List[PageResult]) -> str:
        parts = []
        for p in pages:
            if p.text:
                parts.append(f"[Page {p.page_number}]\n{p.text}")
        return "\n\n".join(parts)

    # ── Dependency check ─────────────────────

    @staticmethod
    def _check_dependencies() -> None:
        try:
            import fitz  # noqa
        except ImportError:
            log.warning("PyMuPDF not installed — PDF ingestion unavailable: pip install PyMuPDF")
        try:
            import pytesseract  # noqa
        except ImportError:
            log.warning("pytesseract not installed — OCR will be unavailable")
