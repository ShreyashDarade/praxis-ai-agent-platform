# praxis/ingestion/parsers/text_parser.py
"""Direct-extraction parser for already-textual content (spec §5 step 2).

`text/markdown` and `application/json` were originally handled here and
have since been handed to `markdown_parser.py` and `json_parser.py`.
Both were formats whose *source text* is a poor stand-in for their
*content* - raw markdown spends its tokens on syntax markers and URLs,
and minified JSON is one unbreakable line of punctuation. The registry
allows exactly one parser per mime type, so those two moved out rather
than being shadowed; see each new module's docstring for the reasoning.

What is left is the set of formats where the bytes really are the best
available text: plain text, and Python source (code is its own clearest
representation - reformatting it would destroy meaning, not reveal it).
"""
from __future__ import annotations

from praxis.core.interfaces import Parser
from praxis.ingestion.parsers.registry import register_parser


class TextParser(Parser):
    """Plain-text-shaped content: decode the bytes as UTF-8, no library needed."""

    supported_mime_types = ("text/plain", "text/x-python")

    async def parse(self, data: bytes, mime_type: str) -> str:
        return data.decode("utf-8")


register_parser(TextParser())
