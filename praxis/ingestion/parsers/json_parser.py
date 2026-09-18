# praxis/ingestion/parsers/json_parser.py
"""JSON parser (spec §5 step 2).

**This mime type was taken over from `TextParser`, not double-claimed.**
`application/json` used to sit in `TextParser.supported_mime_types`,
which meant a JSON attachment was ingested as its own raw source text -
braces, quotes, escapes and all. That is actively bad for the thing
ingestion exists to feed: the chunkers split on blank lines and
sentence-ish boundaries, and minified JSON has neither, so a single
one-line document became one undifferentiated chunk of punctuation, and
the embedding of `{"user":{"name":"Alice"}}` is dominated by syntax
rather than by "Alice". The registry forbids two parsers claiming one
mime type (deliberately - see `register_parser`), so taking this over
meant removing `application/json` from `TextParser`, which this change
does. `text/x-python` stays with `TextParser`: source code *is* its own
best textual representation, JSON is not.

Output is an indented, YAML-shaped outline: keys and scalars on their
own lines, list items bulleted, nesting expressed by indentation.
Structure is preserved (a reader can still see what nests inside what)
while the syntax noise is gone.

Rendering is iterative over an explicit stack rather than recursive.
That is not style preference: a hostile or merely machine-generated
document can nest thousands of levels deep, and a recursive renderer
would die with `RecursionError` on input `json.loads` itself accepted.
What this does NOT bound: total input size. `json.loads` allocates a
Python object graph proportional to (and several times larger than) the
input bytes, and this parser imposes no ceiling of its own - callers
handing over untrusted, unbounded-size uploads need that limit at the
upload boundary, not here. `json.loads`'s own C scanner does enforce
the interpreter recursion limit while *parsing*, so absurdly deep input
fails as a clean `ValueError` below rather than exhausting the C stack.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from praxis.core.interfaces import Parser
from praxis.ingestion.parsers.registry import register_parser

_INDENT = "  "


class JsonParser(Parser):
    """JSON -> an indented outline that keeps structure but drops syntax."""

    supported_mime_types = ("application/json",)

    async def parse(self, data: bytes, mime_type: str) -> str:
        # Decoding plus building the whole object graph is sync and
        # CPU-bound for any non-trivial document.
        return await asyncio.to_thread(self._render, data)

    @staticmethod
    def _render(data: bytes) -> str:
        try:
            document = json.loads(data.decode("utf-8"))
        except RecursionError:
            raise ValueError(
                "JSON document nests deeper than the interpreter recursion limit; refusing to parse"
            ) from None

        lines: list[str] = []
        # Each stack frame is (value, indent_level, label, is_list_item).
        # `label` is the object key this value hangs off (None at the
        # root and for list members); `is_list_item` is what earns the
        # `-` bullet. Children are pushed in reverse so siblings come
        # back off the stack in source order.
        stack: list[tuple[Any, int, str | None, bool]] = [(document, 0, None, False)]
        while stack:
            value, level, label, is_list_item = stack.pop()
            pad = _INDENT * level
            prefix = f"{label}: " if label is not None else ("- " if is_list_item else "")

            if isinstance(value, (dict, list)):
                empty_text = "(empty object)" if isinstance(value, dict) else "(empty list)"
                if not value:
                    lines.append(f"{pad}{prefix}{empty_text}".rstrip())
                    continue
                if label is not None:
                    lines.append(f"{pad}{label}:")
                    level += 1
                elif is_list_item:
                    # A container inside a list gets a bare bullet line
                    # of its own; without it, two adjacent objects in an
                    # array would run together into one flat key list
                    # with no visible boundary between the items.
                    lines.append(f"{pad}-")
                    level += 1
                children = (
                    [(value[key], level, str(key), False) for key in value]
                    if isinstance(value, dict)
                    else [(item, level, None, True) for item in value]
                )
                stack.extend(reversed(children))
                continue

            lines.append(f"{pad}{prefix}{JsonParser._scalar_text(value)}")

        return "\n".join(lines)

    @staticmethod
    def _scalar_text(value: Any) -> str:
        # `str(True)` is "True" and `str(None)` is "None" - neither is
        # what the document said. Round-trip JSON's own spelling so the
        # extracted text still reads like the source it came from.
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)


register_parser(JsonParser())
