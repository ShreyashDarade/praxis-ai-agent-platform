# praxis/ingestion/parsers/markdown_parser.py
"""Markdown parser (spec §5 step 2).

**Decision: `text/markdown` is taken over from `TextParser`, cleanly.**
The registry forbids two parsers claiming one mime type, so this was a
real fork in the road, not a formality. `TextParser` listed
`text/markdown` and returned the raw source verbatim - `# Heading`,
`**bold**`, `[label](https://example.com/some/long/path)` and all. The
third of those is the reason to change: a link-dense document
(changelogs, runbooks, READMEs - exactly what gets ingested) embeds
mostly URL fragments rather than the words a human wrote, and every
`#`/`*`/`` ` `` marker is a token spent on punctuation. Stripping
formatting to prose is a strict improvement for the one thing this
output feeds: chunking and embedding.

So `text/markdown` was *removed* from `TextParser.supported_mime_types`
and claimed here. It is not double-registered, and markdown is not
"left alone". The one thing genuinely lost by the change: the raw
markdown source is no longer recoverable from `parse()` output. That is
acceptable because the original bytes are still stored on the
attachment row - `parse()` feeds search, it is not the system of
record.

Extraction is a real CommonMark parse (`markdown-it-py`) rendered to
HTML and then reduced to text with `BeautifulSoup`, rather than a pile
of regexes. A regex stripper gets fenced code blocks, nested emphasis,
setext headings and reference-style links wrong in ways that are quiet
and hard to notice; a compliant parser gets them right by
construction. Both libraries were already installed in this venv;
`markdown-it-py` is now declared directly in pyproject.toml rather than
relied on transitively, same as `document_parser.py` did for
`beautifulsoup4`.

Limitations worth naming: GFM tables and strikethrough are enabled but
`linkify` is not (it needs `linkify-it-py`, which is not a dependency
here), so bare URLs in text stay as written. Link *targets* are dropped
in favour of link text - a document whose meaning lives in its URLs
loses that here. Struck-through text keeps its words and loses only the
`~~` markers, because deleting the content would be this parser
deciding that a formatting gesture meant "not part of the document".
Raw HTML embedded in the markdown is flattened to its
text, not preserved. Front matter (`---` delimited YAML) is not treated
specially: CommonMark reads the opening `---` as a thematic break, so
front-matter keys land in the output as ordinary text.
"""
from __future__ import annotations

import asyncio
import re

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

from praxis.core.interfaces import Parser
from praxis.ingestion.parsers.registry import register_parser

_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


class MarkdownParser(Parser):
    """Markdown -> readable prose, formatting markers removed."""

    supported_mime_types = ("text/markdown",)

    async def parse(self, data: bytes, mime_type: str) -> str:
        return await asyncio.to_thread(self._strip_formatting, data)

    @staticmethod
    def _strip_formatting(data: bytes) -> str:
        # A fresh MarkdownIt per call rather than a module-level
        # singleton: `parse()` runs on an arbitrary worker thread via
        # `asyncio.to_thread`, and markdown-it-py makes no thread-safety
        # guarantee about a shared instance. Construction is cheap
        # relative to parsing a real document.
        renderer = MarkdownIt("commonmark").enable("table").enable("strikethrough")
        soup = BeautifulSoup(renderer.render(data.decode("utf-8")), "html.parser")

        # `<img>` carries its words in an attribute, so `get_text()`
        # would drop alt text entirely. Substituting the alt text in
        # place keeps a diagram's caption in the extracted prose.
        for image in soup.find_all("img"):
            alt = image.get("alt")
            image.replace_with(alt if isinstance(alt, str) else "")

        # Block-by-block rather than one `get_text(separator="\n")` over
        # the whole tree: that separator would break *inline* elements
        # onto their own lines too, turning "Some **bold** text" into
        # three lines and destroying the sentence. Taking each top-level
        # block's text whole keeps sentences intact, and joining blocks
        # with a blank line gives the chunkers the paragraph boundaries
        # they split on.
        blocks = []
        for node in soup.children:
            text = node.get_text() if hasattr(node, "get_text") else str(node)
            text = _EXCESS_BLANK_LINES.sub("\n\n", text.strip())
            if text:
                blocks.append(text)

        return "\n\n".join(blocks)


register_parser(MarkdownParser())
