# praxis/ingestion/parsers/xml_parser.py
"""XML parser (spec §5 step 2), with explicit, bounded untrusted-input handling.

XML is the one text format in this package that is a genuine attack
surface, so the bounds below are stated precisely - including what they
do *not* cover.

What is actually mitigated here, and by what
--------------------------------------------
1. **Entity-expansion bombs ("billion laughs")** - two independent
   defences, because one of them is a *scan* and scans can be evaded:

   a. The prolog is scanned before parsing and any `<!DOCTYPE`
      declaration is refused outright. An internal general entity must
      be declared in a DTD internal subset, so no DOCTYPE means no
      declared entities means no expansion. The cost of this bound,
      stated plainly: perfectly legitimate XML that carries a DTD is
      also refused. That is a deliberate trade - a text-extraction
      parser gains nothing from a DTD.
   b. `_BoundedTreeBuilder` counts cumulative *character data* as expat
      delivers it and aborts past `MAX_TEXT_CHARS`. This is the backstop
      that does not depend on the scan being right, because an entity
      bomb has to materialise as character data to do any damage, and
      expat delivers expanded content through the data handler
      incrementally.

   Modern libexpat (2.4+) additionally ships its own amplification
   guard, on by default. That is a nice-to-have, not something this
   module relies on: it is a property of whichever libexpat this
   interpreter happens to be linked against, not of this code.

2. **Unbounded structure** - `MAX_ELEMENTS` caps element count and
   `MAX_DEPTH` caps nesting, both enforced during tree building rather
   than after, so a hostile document is abandoned partway instead of
   being fully materialised first. Rendering then walks the tree with an
   explicit stack, never recursion, so a deeply nested (but
   under-the-cap) document cannot blow the Python stack.

3. **Unbounded input** - `MAX_INPUT_BYTES` rejects the payload before a
   parser is even constructed.

What is NOT mitigated, stated honestly
--------------------------------------
- **XXE / external entities.** `xml.etree.ElementTree` does not resolve
  external entities and raises on undefined ones, so the common XXE
  payload fails - but that is *expat's default behaviour*, not an
  enforcement this module makes. Nothing here would stop a future change
  that passed in a parser with an `ExternalEntityRefHandler` installed.
  If untrusted XML ever needs stronger guarantees than "the library's
  default happens to be safe", the answer is `defusedxml`, not this
  file.
- **External DTD / schema fetches** are likewise simply not performed by
  default; no network egress control is implemented here.
- **Encoding-hidden DOCTYPEs.** The prolog scan understands BOM-marked
  UTF-8/16/32 and BOM-less UTF-16 (the null-byte pattern expat itself
  sniffs) and treats everything else as ASCII-compatible. A document in
  some exotic non-ASCII-compatible encoding could in principle hide a
  DOCTYPE from the scan. Defence 1b, not 1a, is what covers that case.
- **Content trust.** Extracted text is returned verbatim. This parser
  makes no claim that the *content* is safe to interpolate into a
  prompt, a query, or markup downstream - that is the caller's problem,
  and `praxis/safety/` is where it belongs.
- **Well-formedness only.** No schema/DTD validation is performed, so
  "parsed successfully" says nothing about the document being the shape
  its producer intended.
"""
from __future__ import annotations

import asyncio
import codecs
import xml.etree.ElementTree as ET

from praxis.core.interfaces import Parser
from praxis.ingestion.parsers.registry import register_parser

# 8 MiB of XML is already an enormous document for a text-extraction
# pipeline; the point of the cap is that every other bound below is then
# reasoning about a payload of known, modest maximum size.
MAX_INPUT_BYTES = 8 * 1024 * 1024
# 8M characters is ~10x the text a 8 MiB ASCII document could legitimately
# contain, so this never fires on honest input - it only catches content
# that appeared from *somewhere other than the input bytes*, which is
# exactly the signature of entity expansion.
MAX_TEXT_CHARS = 8_000_000
MAX_ELEMENTS = 200_000
# 100 levels is far past any hand-authored or tool-generated schema;
# beyond it the document is either machine-generated garbage or hostile.
MAX_DEPTH = 100

_INDENT = "  "


class XmlSafetyError(ValueError):
    """Raised when an XML document is refused on safety grounds.

    A `ValueError` subclass on purpose: the registry and the rest of the
    parser package already signal "this input cannot be handled" with
    `ValueError`, so existing callers catch this without changes, while
    code that cares about the distinction can still catch the specific
    type. `bound` names which rule fired (e.g. `"doctype"`,
    `"text_chars"`) so a log line or test can assert on the specific
    defence rather than on message wording.
    """

    def __init__(self, message: str, *, bound: str) -> None:
        super().__init__(message)
        self.bound = bound


class _BoundedTreeBuilder(ET.TreeBuilder):
    """A `TreeBuilder` that refuses to keep building past the caps above.

    Enforcing inside the builder (rather than inspecting the finished
    tree) is the whole point: by the time a finished tree exists, the
    memory a bomb wanted to consume has already been consumed.
    """

    def __init__(self) -> None:
        super().__init__()
        self._depth = 0
        self._elements = 0
        self._characters = 0

    def start(self, tag, attrs):  # type: ignore[override]
        self._elements += 1
        self._depth += 1
        if self._elements > MAX_ELEMENTS:
            raise XmlSafetyError(
                f"XML document exceeds the {MAX_ELEMENTS} element limit; refusing to parse",
                bound="elements",
            )
        if self._depth > MAX_DEPTH:
            raise XmlSafetyError(
                f"XML document nests deeper than {MAX_DEPTH} levels; refusing to parse",
                bound="depth",
            )
        return super().start(tag, attrs)

    def end(self, tag):  # type: ignore[override]
        self._depth -= 1
        return super().end(tag)

    def data(self, data):  # type: ignore[override]
        self._characters += len(data)
        if self._characters > MAX_TEXT_CHARS:
            raise XmlSafetyError(
                f"XML character data exceeds {MAX_TEXT_CHARS} characters "
                "(possible entity-expansion bomb); refusing to parse",
                bound="text_chars",
            )
        return super().data(data)


class XmlParser(Parser):
    """XML -> an indented outline of elements, attributes and text."""

    supported_mime_types = ("application/xml", "text/xml")

    async def parse(self, data: bytes, mime_type: str) -> str:
        # Parsing and rendering are both sync and CPU-bound; the safety
        # bounds above are what keep that thread's work finite.
        return await asyncio.to_thread(self._render, data)

    @staticmethod
    def _render(data: bytes) -> str:
        if len(data) > MAX_INPUT_BYTES:
            raise XmlSafetyError(
                f"XML input is {len(data)} bytes, above the {MAX_INPUT_BYTES} byte limit; "
                "refusing to parse",
                bound="input_bytes",
            )
        _refuse_doctype(_decode_for_scan(data))

        parser = ET.XMLParser(target=_BoundedTreeBuilder())
        parser.feed(data)
        root = parser.close()

        lines: list[str] = []
        # (element, depth), pushed in reverse so children come back out
        # in document order. Explicit stack, not recursion - see the
        # module docstring's point 2.
        stack: list[tuple[ET.Element, int]] = [(root, 0)]
        while stack:
            element, depth = stack.pop()
            lines.append(_element_line(element, depth))
            tail = (element.tail or "").strip()
            if tail:
                lines.append(f"{_INDENT * depth}{tail}")
            stack.extend((child, depth + 1) for child in reversed(list(element)))

        return "\n".join(lines)


def _element_line(element: ET.Element, depth: int) -> str:
    parts = [f"{_INDENT * depth}{_local_name(element.tag)}"]
    if element.attrib:
        rendered = ", ".join(
            f"{_local_name(key)}={value}" for key, value in element.attrib.items()
        )
        parts.append(f" [{rendered}]")
    text = (element.text or "").strip()
    if text:
        parts.append(f": {text}")
    return "".join(parts)


def _local_name(tag: str) -> str:
    """Strip the `{namespace-uri}` prefix ElementTree prepends to names.

    Namespace URIs are pure noise for the downstream consumer here
    (embedding + semantic search): they are long, near-identical across
    every element of a document, and would dominate the extracted text.
    The cost is real and worth naming: two same-named elements from
    different namespaces become indistinguishable in the output.
    """
    if tag.startswith("{"):
        return tag.partition("}")[2]
    return tag


def _decode_for_scan(data: bytes) -> str:
    """Decode just far enough to read the prolog reliably.

    The bytes themselves are still handed to expat afterwards, which
    honours the XML declaration's own encoding properly. This decode
    exists only so `_refuse_doctype` is looking at characters rather
    than guessing at bytes.
    """
    for bom, encoding in (
        # UTF-32 first: BOM_UTF32_LE starts with the two bytes of
        # BOM_UTF16_LE, so checking UTF-16 first would mis-detect it.
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
        (codecs.BOM_UTF8, "utf-8"),
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
    ):
        if data.startswith(bom):
            return data[len(bom) :].decode(encoding, errors="replace")

    # BOM-less UTF-16: expat sniffs this from the null-byte pattern of
    # the `<` that must open a well-formed document, so the scan has to
    # as well - otherwise a UTF-16 DOCTYPE parses fine but reads as
    # mojibake to the scan and slips through.
    if data[:2] == b"\x00<":
        return data.decode("utf-16-be", errors="replace")
    if data[:2] == b"<\x00":
        return data.decode("utf-16-le", errors="replace")

    return data.decode("utf-8", errors="replace")


def _refuse_doctype(text: str) -> None:
    """Walk the prolog's constructs and refuse if a DOCTYPE appears.

    Deliberately not a bare substring search for `<!DOCTYPE`: that would
    also refuse a document that merely *mentions* the string inside a
    comment or a text node, which is a false positive on honest input.
    A DOCTYPE may only legally appear in the prolog, so stepping over
    the prolog's three legal construct kinds (whitespace, processing
    instructions, comments) until the root element starts is both
    precise and cheap.
    """
    index = 0
    length = len(text)
    while index < length:
        if text[index].isspace() or text[index] == "﻿":
            index += 1
            continue
        if not text.startswith("<", index):
            # Character data before the root element is malformed XML.
            # Say nothing and let expat raise the real, specific
            # ParseError rather than inventing a worse message here.
            return
        if text.startswith("<?", index):
            end = text.find("?>", index + 2)
            if end == -1:
                return
            index = end + 2
            continue
        if text.startswith("<!--", index):
            end = text.find("-->", index + 4)
            if end == -1:
                return
            index = end + 3
            continue
        if text[index : index + 9].upper() == "<!DOCTYPE":
            raise XmlSafetyError(
                "XML document declares a DOCTYPE; refusing to parse because a DTD internal "
                "subset is where entity-expansion bombs are declared",
                bound="doctype",
            )
        return


register_parser(XmlParser())
