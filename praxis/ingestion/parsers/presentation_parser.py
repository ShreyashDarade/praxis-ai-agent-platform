# praxis/ingestion/parsers/presentation_parser.py
"""PPTX parser (spec §5 step 2).

Kept out of `document_parser.py` deliberately. That module's three
formats (PDF/DOCX/HTML) all reduce to "one linear stream of prose";
a deck does not - its unit is the *slide*, and losing the slide
boundary loses the only structure a deck has. So this parser emits an
explicitly slide-delimited text, and lives in its own module rather
than growing DocumentParser's dispatch chain with a format whose
output shape is different.

`python-pptx` is the extractor (pure Python over the OOXML package via
`lxml`; no native binary beyond `lxml`'s own, which this venv already
depends on for other reasons). It is added to pyproject.toml as a
direct, pinned dependency - same discipline as every other pin there.
`unstructured` is *not* used here for the same reason
`document_parser.py` documents: its auto-partition import segfaults on
this machine.

Extraction covers the three places text actually lives in a .pptx:
autoshape/placeholder text frames, table cells, and - because a deck's
argument frequently lives there rather than on the slide itself -
speaker notes. Group shapes are walked recursively, since PowerPoint
nests shapes arbitrarily deep and a flat `slide.shapes` loop silently
misses everything inside a group.

What this does NOT extract, stated plainly rather than implied by
omission: text baked into embedded *images* (that is OCR's job - a
caller wanting it should route the image bytes to `ImageOcrParser`),
text inside embedded OLE objects or charts, WordArt rendered as
vector geometry, and slide-master/layout boilerplate (deliberately
skipped - repeating a template's footer on every slide is noise for
semantic search, not signal).
"""
from __future__ import annotations

import asyncio
import io
from typing import Iterator

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from praxis.core.interfaces import Parser
from praxis.ingestion.parsers.registry import register_parser

_PPTX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


class PresentationParser(Parser):
    """PPTX -> slide-delimited plain text, including tables and notes."""

    supported_mime_types = (_PPTX_MIME_TYPE,)

    async def parse(self, data: bytes, mime_type: str) -> str:
        # python-pptx parses the whole OOXML package eagerly and is
        # sync/CPU-bound - run it off the event loop thread, same
        # discipline as DocumentParser and TabularParser.
        return await asyncio.to_thread(self._extract, data)

    @staticmethod
    def _extract(data: bytes) -> str:
        presentation = Presentation(io.BytesIO(data))

        slides: list[str] = []
        for index, slide in enumerate(presentation.slides, start=1):
            lines = [f"Slide {index}:"]
            for shape in PresentationParser._walk_shapes(slide.shapes):
                lines.extend(PresentationParser._shape_lines(shape))
            notes = PresentationParser._notes_text(slide)
            if notes:
                lines.append(f"Notes: {notes}")
            slides.append("\n".join(lines))

        return "\n\n".join(slides)

    @staticmethod
    def _walk_shapes(shapes) -> Iterator[object]:
        """Yield every shape, descending into groups.

        PowerPoint lets a group contain groups; a non-recursive loop
        over `slide.shapes` returns the group container itself (which
        has no text frame of its own) and silently drops all the text
        inside it.
        """
        for shape in shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                yield from PresentationParser._walk_shapes(shape.shapes)
            else:
                yield shape

    @staticmethod
    def _shape_lines(shape) -> list[str]:
        if shape.has_table:
            return PresentationParser._table_lines(shape.table)
        if shape.has_text_frame:
            text = shape.text_frame.text.strip()
            return [text] if text else []
        return []

    @staticmethod
    def _table_lines(table) -> list[str]:
        """Render a table row-per-line with ` | `-joined cells.

        The same separator TabularParser-adjacent output uses, chosen so
        a row stays one line: the chunkers split on line and blank-line
        boundaries, so a cell-per-line rendering would scatter one
        logical row across several chunks.
        """
        lines = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))
        return lines

    @staticmethod
    def _notes_text(slide) -> str:
        # `slide.notes_slide` *creates* a notes slide as a side effect if
        # one does not exist, so the `has_notes_slide` guard is load
        # bearing, not defensive noise: without it, parsing would mutate
        # the in-memory package for every slide in the deck.
        if not slide.has_notes_slide:
            return ""
        notes_frame = slide.notes_slide.notes_text_frame
        if notes_frame is None:
            return ""
        return notes_frame.text.strip()


register_parser(PresentationParser())
