# praxis/ingestion/parsers/log_parser.py
"""Log file parser (spec §5 step 2).

A log file is plain text, so `TextParser` would technically "work" -
this parser exists because what makes a log searchable is not its
characters but its *shape*, and two properties of that shape are worth
spending a parser on.

First, line structure is the document structure. A log has no
paragraphs; one line is one event. The lines are therefore reproduced
verbatim (only `\\r\\n` normalised to `\\n`), never re-flowed, so an
event never gets merged into its neighbour.

Second, the levels present are the single most useful fact about a log
before you read it - "does this file contain any ERRORs at all" is the
question almost every ingestion of a log is really asking. A level
census is emitted as the first line so it lands in the first chunk and
stays retrievable by semantic search, instead of being a fact you could
only recover by reading every chunk.

Level detection is a documented heuristic, not a parse, and its rules
are deliberately conservative:
- Bare level words are matched **case-sensitively in upper case only**
  (`ERROR`, not `error`). Lower-case matching would count the ordinary
  English word "error" in a message body as a level and make the census
  lie, which is worse than under-reporting.
- The two structured forms that *do* carry the level unambiguously -
  logfmt `level=warn` and JSON `"level":"warn"` - are matched
  case-insensitively, because there the key names the field and there
  is no ambiguity to protect against.
- At most one level is counted per line, first match wins. The level
  field appears near the start of essentially every log format, so this
  stops a message that quotes another level ("retrying after ERROR")
  from being double-counted.
- Aliases are **not** collapsed: `WARN` and `WARNING` are reported as
  the distinct tokens they are in the source. Merging them would be
  this parser guessing at the producer's vocabulary.

Consequences of the heuristic, stated plainly: an all-lower-case log
format reports no levels at all unless it uses one of the two
structured forms; a custom level name outside the table below is not
recognised; and a multi-line event (a stack trace) counts as many
lines, only the first of which carries a level.

Decoding uses `errors="replace"`. Logs are routinely truncated
mid-character by rotation, or carry a stray byte from one misbehaving
process, and refusing to ingest a 200 MB log over one bad byte is the
wrong trade for this format specifically.
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter

from praxis.core.interfaces import Parser
from praxis.ingestion.parsers.registry import register_parser

_LEVEL_WORDS = (
    "TRACE",
    "DEBUG",
    "INFO",
    "NOTICE",
    "WARN",
    "WARNING",
    "ERROR",
    "ERR",
    "CRITICAL",
    "FATAL",
    "SEVERE",
    "PANIC",
    "ALERT",
    "EMERGENCY",
)
# Longest-first so `WARNING` is preferred over the `WARN` prefix; `\b`
# alone would not decide between two alternatives that both match at the
# same position, and Python's alternation takes the first that matches.
_BARE_LEVEL = re.compile(r"\b(" + "|".join(sorted(_LEVEL_WORDS, key=len, reverse=True)) + r")\b")
_STRUCTURED_LEVEL = re.compile(
    r"""(?:"level"|'level'|\blevel|\blvl|\bseverity)\s*[:=]\s*["']?([A-Za-z]+)""",
    re.IGNORECASE,
)


class LogParser(Parser):
    """Log files -> a level census followed by the lines, verbatim."""

    supported_mime_types = ("text/x-log",)

    async def parse(self, data: bytes, mime_type: str) -> str:
        return await asyncio.to_thread(self._summarize, data)

    @staticmethod
    def _summarize(data: bytes) -> str:
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")
        # A trailing newline is a line terminator, not an empty final
        # event - dropping it keeps the line count honest.
        if lines and lines[-1] == "":
            lines.pop()

        levels: Counter[str] = Counter()
        for line in lines:
            level = _level_of(line)
            if level is not None:
                levels[level] += 1

        header = f"Log file: {len(lines)} lines"
        if levels:
            census = ", ".join(
                f"{level} ({count})" for level, count in sorted(levels.items())
            )
            header += f"; levels present: {census}"
        else:
            header += "; no recognised log levels"

        # Blank line between the census and the body so the two are
        # separate units to the chunkers, exactly as in TabularParser.
        return "\n".join([header, "", *lines])


def _level_of(line: str) -> str | None:
    """The level this line declares, upper-cased, or None."""
    structured = _STRUCTURED_LEVEL.search(line)
    if structured is not None:
        candidate = structured.group(1).upper()
        # The key said "level", but the value still has to be one we
        # recognise - `level=production` is a deployment tag, not a
        # severity, and counting it would pollute the census.
        if candidate in _LEVEL_WORDS:
            return candidate
    bare = _BARE_LEVEL.search(line)
    if bare is not None:
        return bare.group(1)
    return None


register_parser(LogParser())
