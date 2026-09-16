# praxis/ingestion/chunkers/recursive_chunker.py
"""Prose chunker wrapping LangChain's `RecursiveCharacterTextSplitter` (spec §5 step 3, §2).

The general-purpose default for anything that isn't tabular: text,
markdown, code, and `unstructured`-extracted document text. Per spec
§2's table: "LangChain text splitters (open-source,
`RecursiveCharacterTextSplitter` + semantic)".
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from praxis.core.interfaces import Chunker

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 200


class RecursiveChunker(Chunker):
    """Splits prose recursively (paragraph -> sentence -> word) to stay under `chunk_size`."""

    def __init__(self, chunk_size: int = _DEFAULT_CHUNK_SIZE, chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return self._splitter.split_text(text)
