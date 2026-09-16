# praxis/ingestion/parsers/text_parser.py
"""Direct-extraction parser for already-textual content (spec §5 step 2)."""
from __future__ import annotations

from praxis.core.interfaces import Parser
from praxis.ingestion.parsers.registry import register_parser


class TextParser(Parser):
    """Plain-text-shaped content: decode the bytes as UTF-8, no library needed."""

    supported_mime_types = ("text/plain", "text/markdown", "text/x-python", "application/json")

    async def parse(self, data: bytes, mime_type: str) -> str:
        return data.decode("utf-8")


register_parser(TextParser())
