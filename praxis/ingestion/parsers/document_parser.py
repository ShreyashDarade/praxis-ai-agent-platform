# praxis/ingestion/parsers/document_parser.py
"""Document parser for PDF/DOCX/HTML (spec §5 step 2).

**Deviation from the spec's `unstructured`-for-everything table entry,
verified and documented, not guessed:** `from unstructured.partition.auto
import partition` segfaults on import on this machine (Windows,
Python 3.11) - a *second*, deeper native-compatibility problem in
`unstructured`'s own dependency chain, beyond the `python-magic`
segfault already found and fixed (see pyproject.toml's comment) by
uninstalling `python-magic`. Reproduced directly:

    python -c "from unstructured.partition.auto import partition"
    -> Segmentation fault (exit 139)

`unstructured` remains listed in pyproject.toml (its own docstring
elsewhere may still reference it) but is not imported by this module.
Chasing a second native crash inside a heavy auto-detection stack, on a
platform `unstructured` itself already restricts real PDF layout
partitioning on (see pyproject.toml), is not worth the platform-specific
fragility - three focused, pure-Python-safe libraries per format are
both simpler and more predictable:

- PDF -> `pypdfium2` (already a dependency, for OCR page rasterization;
  it also does plain text extraction directly, no OCR needed for a
  text-based PDF)
- DOCX -> `python-docx` (pure Python XML parsing, no native extension)
- HTML -> `BeautifulSoup` (`beautifulsoup4`, pure Python)

Both `python-docx` and `beautifulsoup4` were already present as
transitive dependencies; this module makes that dependency direct and
explicit rather than accidental (see pyproject.toml).
"""
from __future__ import annotations

import asyncio
import io

import pypdfium2 as pdfium
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

from praxis.core.interfaces import Parser
from praxis.ingestion.parsers.registry import register_parser


class DocumentParser(Parser):
    """PDF / DOCX / HTML -> plain text, one dedicated extractor per format."""

    supported_mime_types = (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/html",
    )

    async def parse(self, data: bytes, mime_type: str) -> str:
        # Each extractor is sync/CPU-bound - run off the event loop
        # thread, same discipline as SentenceTransformerEmbedder.embed().
        return await asyncio.to_thread(self._extract, data, mime_type)

    @staticmethod
    def _extract(data: bytes, mime_type: str) -> str:
        if mime_type == "application/pdf":
            return DocumentParser._extract_pdf(data)
        if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return DocumentParser._extract_docx(data)
        if mime_type == "text/html":
            return DocumentParser._extract_html(data)
        raise ValueError(f"DocumentParser cannot handle mime type '{mime_type}'")

    @staticmethod
    def _extract_pdf(data: bytes) -> str:
        pdf = pdfium.PdfDocument(data)
        try:
            pages_text = []
            for page in pdf:
                text_page = page.get_textpage()
                try:
                    pages_text.append(text_page.get_text_range())
                finally:
                    text_page.close()
                page.close()
            return "\n".join(pages_text)
        finally:
            pdf.close()

    @staticmethod
    def _extract_docx(data: bytes) -> str:
        document = DocxDocument(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs if p.text)

    @staticmethod
    def _extract_html(data: bytes) -> str:
        soup = BeautifulSoup(data, "html.parser")
        return soup.get_text(separator="\n", strip=True)


register_parser(DocumentParser())
