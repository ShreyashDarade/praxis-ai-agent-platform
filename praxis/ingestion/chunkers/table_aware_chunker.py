# praxis/ingestion/chunkers/table_aware_chunker.py
"""Row/section-aware chunker for tabular summaries (spec §5 step 3).

`RecursiveCharacterTextSplitter` is tuned for prose (paragraph ->
sentence -> word fallback) and would happily cut a table row or a
column-list entry in half. `TabularParser`'s output is instead a small
number of logical sections (shape / column list / numeric summary),
separated by blank lines - this chunker splits on exactly those blank
line boundaries so each chunk is a whole section, falling back to whole
*lines* (never a partial line/row) only when a section itself is bigger
than `chunk_size`.
"""
from __future__ import annotations

from praxis.core.interfaces import Chunker

_DEFAULT_CHUNK_SIZE = 1000


class TableAwareChunker(Chunker):
    """Splits tabular-summary text at blank-line section boundaries, then at line boundaries."""

    def __init__(self, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        chunks: list[str] = []
        for section in self._split_into_sections(text):
            if len(section) <= self._chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self._split_by_line(section))
        return chunks

    @staticmethod
    def _split_into_sections(text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n")
        return [section.strip("\n") for section in normalized.split("\n\n") if section.strip()]

    def _split_by_line(self, section: str) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        def flush() -> None:
            if current:
                chunks.append("\n".join(current))
                current.clear()

        for line in section.split("\n"):
            if len(line) > self._chunk_size:
                # A single row/line wider than chunk_size on its own -
                # flush what's pending, then hard-slice this one line so
                # "no chunk exceeds chunk_size" always holds, even for a
                # pathologically wide table row.
                flush()
                current_len = 0
                for start in range(0, len(line), self._chunk_size):
                    chunks.append(line[start : start + self._chunk_size])
                continue

            extra = len(line) + (1 if current else 0)
            if current and current_len + extra > self._chunk_size:
                flush()
                current_len = 0
                extra = len(line)

            current.append(line)
            current_len += extra

        flush()
        return chunks
