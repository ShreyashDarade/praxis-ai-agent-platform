# praxis/observability/logging.py
"""Structured JSON logging (spec §10): "structured JSON logs at
Orchestrator, subagent, tool-call, and Factory-synthesis granularity,
every entry carrying the task's `correlation_id` so a full request
trace is reconstructable."

`configure_logging()` sets structlog up to render every log call as one
JSON object on stdout - called once, at `praxis.api.main` import time.
`bind_correlation_id(correlation_id)` is a contextvar-based bind (via
`structlog.contextvars`): every log call made *anywhere* during that
`with` block - in any module that does its own
`structlog.get_logger(__name__)` (the same pattern every real call site
in this codebase uses - `praxis.core.orchestrator`,
`praxis.agents.capability_factory`, `praxis.connectors.retry`,
`praxis.api.main`) - automatically carries `correlation_id`, with no
need to thread it through every function's parameters. This module
deliberately doesn't also re-export a `get_logger()` wrapper around
`structlog.get_logger`: every real call site already imports `structlog`
directly, so a wrapper here would be unused indirection, not a
convenience.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

import structlog

from praxis.security.redaction import redact

_CONFIGURED = False


def _redaction_processor(
    _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Redacts PII/secrets out of every log event before it is rendered
    (Prompt §9: "Redact sensitive payloads in tracing").

    Sits immediately before the JSON renderer so it sees the fully
    merged event - contextvars, bound values, and call-site kwargs
    alike - and therefore cannot be bypassed by binding a secret
    earlier in the chain.

    Never raises: a redaction failure must degrade to *dropping the
    offending value*, never to losing the log line or breaking the
    caller that emitted it.
    """
    try:
        return redact(dict(event_dict))
    except Exception:  # noqa: BLE001 - see docstring
        return {"event": event_dict.get("event", "unknown"), "redaction_error": True}


def configure_logging(*, level: int = logging.INFO) -> None:
    """Configures structlog to emit one JSON line per log call.

    Idempotent - a second call is a no-op - mirrors
    `praxis.agents.skill_registry.discover_skills()`'s own idempotence
    posture, so it's safe to call at `praxis.api.main` import time and
    again from a test with no double-configuration surprises.

    Deliberately does *not* pass an explicit `file=` to
    `structlog.PrintLoggerFactory()`: structlog's own `PrintLogger`
    resolves `sys.stdout` dynamically at each write when none is given
    (verified directly against the installed version - "switched to use
    `print` for better monkeypatchability") - which is exactly what lets
    a test capture real output via pytest's `capsys` even though
    `capsys` replaces `sys.stdout` *after* this function may already
    have run.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redaction_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


@contextmanager
def bind_correlation_id(correlation_id: str) -> Iterator[None]:
    """Binds `correlation_id` into every log call made anywhere during
    this `with` block (structlog's `contextvars` machinery - genuinely
    per-asyncio-task, since `asyncio.Task` copies the current
    `contextvars.Context` at creation, so concurrently-gathered subtasks
    started underneath this block each still carry it).

    Always unbinds on the way out - including on an exception - so one
    task's correlation id can never leak into whatever runs next in the
    same context after this block ends.
    """
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars("correlation_id")
