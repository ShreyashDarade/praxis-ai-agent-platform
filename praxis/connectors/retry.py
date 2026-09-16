# praxis/connectors/retry.py
"""Bounded retry for a real `Connector` call (spec §12: `ConnectorError`
- "a connector call fails; caught at the connector wrapper, logged,
bounded retry, then surfaced as a step failure").

`call_with_retry` is that wrapper: a small async function (not a
decorator - a decorator would need to wrap a whole method definition
even for the one-off call sites that actually need this, whereas a
plain function wraps exactly the one call that needs bounding, with the
connector name/operation supplied right there for the log lines and the
eventual `ConnectorError`) that runs a zero-arg async callable up to
`max_attempts` times, logging every failed attempt, and raising
`ConnectorError` - carrying the real last attempt's error, never a
generic message - only once every attempt has failed.

Wired into the one real `Connector` call site that exists outside a
connector's own implementation today:
`praxis.agents.capability_factory.CapabilityFactory._connector_schema`'s
`connector.describe()` call. This phase's other candidate call site,
`praxis.agents.skills.retrieve_documents.RetrieveDocumentsSkill`, was
checked directly and goes through `praxis.ingestion.pipeline.retrieve`
against a `VectorStore`, never a `Connector` - there is genuinely
nothing to wire there today. This module stays real, usable
infrastructure for whichever future connector-calling skill needs it
next, not speculative code guessing at a call site that doesn't exist.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

import structlog

from praxis.core.exceptions import ConnectorError

T = TypeVar("T")

_logger = structlog.get_logger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 0.0


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    connector_name: str,
    operation: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> T:
    """Calls `fn()` (a zero-arg async callable wrapping one connector
    call, e.g. `lambda: connector.describe()` or `connector.describe`
    itself) up to `max_attempts` times.

    Returns the first successful attempt's result immediately - no
    retry happens once one attempt succeeds. Every failed attempt is
    logged (structured, via `structlog` - automatically carrying
    whatever `correlation_id` is bound via
    `praxis.observability.logging.bind_correlation_id` in the calling
    context, with zero extra plumbing here). When every attempt fails,
    raises `ConnectorError` carrying `connector_name`, `operation`,
    the real attempt count, and the *last* attempt's real error detail.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure of this one call is retried up to the bound, then surfaced as ConnectorError - never silently swallowed
            last_error = exc
            _logger.warning(
                "connector_call_failed",
                connector=connector_name,
                operation=operation,
                attempt=attempt,
                max_attempts=max_attempts,
                error=str(exc),
            )
            if attempt < max_attempts and retry_delay_seconds > 0:
                await asyncio.sleep(retry_delay_seconds)

    raise ConnectorError(
        f"connector '{connector_name}' operation '{operation}' failed after {max_attempts} attempt(s)",
        connector_name=connector_name,
        operation=operation,
        attempts=max_attempts,
        detail=str(last_error),
    )
