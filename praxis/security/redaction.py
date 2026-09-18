# praxis/security/redaction.py
"""PII/secret detection and redaction (Prompt §8's "PII detection/
redaction", §9's "Redact sensitive payloads in tracing").

Used in three places, all of them *outbound* from the system:
- every `AuditLog.detail` payload (`praxis.security.audit`),
- every structlog event (`praxis.observability.logging`'s redaction
  processor),
- any value a caller explicitly routes through `redact_text`.

Deliberately regex-based and dependency-free. This is a defence-in-depth
control that must never itself fail, be slow enough to matter on a hot
path, or need a model call; it is *not* a claim of perfect recall. The
detectors below cover the categories that actually show up in this
system's payloads: credentials in DSNs, bearer/API tokens, emails,
phone numbers, national IDs, and payment card numbers.
"""
from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"

# Keys whose *value* is always redacted regardless of its shape - the
# cheapest and most reliable signal available, since this system names
# its own secrets consistently (Settings fields, connector kwargs).
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key"
    r"|authorization|auth|credential|dsn|database_url|connection_string|cookie|session)",
    re.IGNORECASE,
)

# Each entry: (name, compiled pattern, replacement). Order matters -
# DSN credentials are matched before the bare-email detector, so
# "postgresql://user:pw@host/db" redacts the credential pair rather
# than being partially mangled as an address.
_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "dsn_credentials",
        re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)(?P<creds>[^/\s:@]+:[^/\s:@]+)@"),
        rf"\g<scheme>{REDACTED}@",
    ),
    (
        "bearer_token",
        re.compile(r"(?i)\b(bearer|token|apikey|api[_-]key)\s+[A-Za-z0-9._\-]{8,}"),
        rf"\1 {REDACTED}",
    ),
    # Provider-prefixed keys (Anthropic, OpenAI, Slack, GitHub, Tavily,
    # AWS) - all share a "known prefix + long opaque tail" shape.
    (
        "provider_api_key",
        re.compile(
            r"\b(sk-ant-[A-Za-z0-9._\-]{8,}|sk-[A-Za-z0-9]{16,}|xox[baprs]-[A-Za-z0-9\-]{8,}"
            r"|gh[pousr]_[A-Za-z0-9]{16,}|tvly-[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{12,})"
        ),
        REDACTED,
    ),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        REDACTED,
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
        REDACTED,
    ),
    (
        "us_ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        REDACTED,
    ),
    (
        "india_aadhaar",
        re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
        REDACTED,
    ),
    (
        "phone",
        re.compile(r"(?<![\w.])\+\d{1,3}[\s\-]?\d{6,14}(?![\w.])"),
        REDACTED,
    ),
]

# The credit-card detector's digit run would also swallow ordinary long
# integers (row counts, epoch millis in a list, ids). Luhn-check any
# candidate before redacting it, so genuine numeric data survives.
_CARD_PATTERN_NAME = "credit_card"


def _luhn_valid(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _redact_card(match: re.Match[str]) -> str:
    raw = match.group(0)
    digits = re.sub(r"[ \-]", "", raw)
    if len(digits) < 13 or not _luhn_valid(digits):
        return raw
    return REDACTED


def detect(text: str) -> list[str]:
    """Returns the names of every detector that fires on `text`.

    Exposed separately from `redact_text` so a caller can *decide* based
    on presence (e.g. refuse to send a payload to an external provider)
    rather than only sanitizing it.
    """
    found: list[str] = []
    for name, pattern, _replacement in _PATTERNS:
        if name == _CARD_PATTERN_NAME:
            if any(_luhn_valid(re.sub(r"[ \-]", "", m.group(0))) for m in pattern.finditer(text)):
                found.append(name)
            continue
        if pattern.search(text):
            found.append(name)
    return found


def redact_text(text: str) -> str:
    """Replaces every detected secret/PII span in `text` with `[REDACTED]`."""
    redacted = text
    for name, pattern, replacement in _PATTERNS:
        if name == _CARD_PATTERN_NAME:
            redacted = pattern.sub(_redact_card, redacted)
            continue
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact(value: Any, *, _depth: int = 0) -> Any:
    """Recursively redacts a JSON-ish structure.

    - A dict key matching `_SENSITIVE_KEY_PATTERN` has its whole value
      replaced, whatever its type (so a nested credential object is
      dropped entirely rather than walked into).
    - Every string is passed through `redact_text`.
    - Lists/tuples/sets are walked elementwise; other scalars pass
      through unchanged.

    `_depth` bounds recursion at 12 levels - a defensive control must
    never be the thing that blows the stack on a pathological payload.
    """
    if _depth > 12:
        return "[TRUNCATED]"

    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _SENSITIVE_KEY_PATTERN.search(key):
                result[key] = REDACTED
            else:
                result[key] = redact(item, _depth=_depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item, _depth=_depth + 1) for item in value]
    if isinstance(value, set):
        return [redact(item, _depth=_depth + 1) for item in sorted(value, key=str)]
    return value
