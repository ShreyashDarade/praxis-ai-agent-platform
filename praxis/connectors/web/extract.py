# praxis/connectors/web/extract.py
"""Extracts plain text from guarded-fetched web content, by content
type (spec §6.1/§5's "extract" step of guarded-fetch -> extract ->
distill).

Reuses Phase 3's own per-format extractors
(`praxis.ingestion.parsers.document_parser.DocumentParser`) rather than
reimplementing HTML/PDF parsing a second time. `DocumentParser._extract_html`/
`_extract_pdf` are already pure, synchronous, dependency-free static
methods - `DocumentParser.parse()` is the only thing that wraps them in
`asyncio.to_thread` for its own async `Parser` interface, so calling the
static methods directly here needs no such wrapping of its own. This is
a deliberate, judgment-call reuse across `praxis.connectors` ->
`praxis.ingestion.parsers` (a one-directional dependency - nothing in
`praxis.ingestion` imports `praxis.connectors`, so this introduces no
import cycle) rather than duplicating two already-working parsers, or
inventing an awkward shared base module purely to avoid one cross-
package import.

`extract_text` is deliberately synchronous (not `async def`): both
reused extractors are fast, pure-Python, CPU-only, no-I/O calls -
nothing to `await`. A caller wanting this off the event-loop thread (the
discipline `DocumentParser.parse` applies for its own, potentially
larger, ingestion-pipeline documents) can wrap this call in
`asyncio.to_thread` itself.
"""
from __future__ import annotations

from praxis.ingestion.parsers.document_parser import DocumentParser


def extract_text(content: bytes, content_type: str) -> str:
    """Extracts plain text from `content`, dispatching on `content_type`.

    Raises `ValueError` for a content type this function doesn't
    understand - reachable only if a caller bypasses
    `guarded_fetch.fetch`'s own content-type allow-list (which every
    real call site in this subpackage goes through first), so failing
    loudly here is a defensive backstop, not the primary enforcement
    point.
    """
    base_type = content_type.split(";")[0].strip().lower()

    if base_type == "application/pdf":
        return DocumentParser._extract_pdf(content)
    if base_type in ("text/html", "application/xhtml+xml"):
        return DocumentParser._extract_html(content)
    if base_type.startswith("text/"):
        return content.decode("utf-8", errors="replace")

    raise ValueError(f"extract_text cannot handle content type '{content_type}'")
