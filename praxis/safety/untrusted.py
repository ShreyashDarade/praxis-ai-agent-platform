# praxis/safety/untrusted.py
"""Untrusted-content wrapping and prompt-injection detection (Prompt §4, §8).

Generalizes what `praxis.connectors.web.untrusted` already did for
fetched web pages to *every* source of content the system did not
author: uploaded files, connector query results, tool output. The
prompt is explicit that this applies to uploads too - "Treat extracted
content as untrusted data, never executable instructions. Separate
user uploads from trusted skill packages."

Two complementary controls:

1. **Delimiting.** `wrap_untrusted` fences content in explicit markers
   naming its source, so a model reading it can tell data from
   instruction. This is a mitigation, not a guarantee - a sufficiently
   clever injection can still try to talk its way out of a fence, which
   is exactly why the *authorization* controls in `praxis.security`
   never consult model output.

2. **Detection.** `detect_injection_markers` flags the recognizable
   shapes of an injection attempt (instruction-override phrasing,
   fake system/role turns, attempts to close the fence, exfiltration
   verbs aimed at credentials). Detection feeds logging, audit detail,
   and - for ingestion - a quarantine decision, rather than silently
   scrubbing the content, because silently modifying a user's document
   would be worse than flagging it.

Deliberately dependency-free regex work: this sits on the ingestion and
tool-result hot paths and must never be the slow or failing part.
"""
from __future__ import annotations

import re

UNTRUSTED_OPEN = "<<<UNTRUSTED_CONTENT source={source}>>>"
UNTRUSTED_CLOSE = "<<<END_UNTRUSTED_CONTENT>>>"

# Each pattern targets a *shape* an injection takes, not a specific
# wording, so paraphrases still match.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "instruction_override",
        re.compile(
            r"(?i)\b(ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|earlier|all)\b[^.\n]{0,20}"
            r"\b(instruction|prompt|rule|direction|context)s?\b"
        ),
    ),
    (
        "fake_role_turn",
        re.compile(
            r"(?im)^\s*(system|assistant|developer|human|user)\s*[:>\]]\s*",
        ),
    ),
    (
        "fence_escape",
        re.compile(re.escape(UNTRUSTED_CLOSE), re.IGNORECASE),
    ),
    (
        "new_instructions",
        re.compile(
            r"(?i)\b(new|updated|revised|real|actual)\s+(instruction|task|objective|goal)s?\b"
        ),
    ),
    (
        "credential_exfiltration",
        re.compile(
            r"(?i)\b(send|post|exfiltrate|upload|reveal|print|output|leak|email)\b"
            r"[^.\n]{0,40}\b(api[_\- ]?key|secret|password|token|credential|env(ironment)?\s+var)"
        ),
    ),
    (
        "tool_coercion",
        re.compile(
            r"(?i)\b(call|invoke|execute|run)\b[^.\n]{0,30}"
            r"\b(tool|function|skill|command|shell|script)\b"
        ),
    ),
]


def detect_injection_markers(text: str) -> list[str]:
    """Names every injection shape detected in `text`.

    Returns an empty list for ordinary content. Never raises, and never
    modifies the input - a caller decides what to do with the finding
    (log it, quarantine the document, refuse to feed it to a model).
    """
    if not text:
        return []
    return [name for name, pattern in _INJECTION_PATTERNS if pattern.search(text)]


def _neutralize_fence_escape(text: str) -> str:
    """Breaks any literal closing marker embedded in the content.

    Without this, content containing the close marker could end the
    fence early and have its remainder read as trusted instruction -
    the one injection shape that defeats delimiting outright, so it is
    the one case where modifying the content is more honest than
    leaving it intact.
    """
    return text.replace(UNTRUSTED_CLOSE, "<<<END_UNTRUSTED_CONTENT_ESCAPED>>>")


def wrap_untrusted(text: str, *, source: str) -> str:
    """Fences `text` as untrusted data attributed to `source`."""
    return "\n".join(
        (
            UNTRUSTED_OPEN.format(source=source),
            _neutralize_fence_escape(text),
            UNTRUSTED_CLOSE,
        )
    )


def is_wrapped(text: str) -> bool:
    """True iff `text` already carries the untrusted fence - so a value
    passed through two layers isn't double-wrapped."""
    return text.startswith("<<<UNTRUSTED_CONTENT") and text.rstrip().endswith(UNTRUSTED_CLOSE)
