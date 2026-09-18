# praxis/connectors/resilience.py
"""Per-connector rate limiting and circuit breaking (brief §6).

The brief requires each connector to carry "rate limiting, retries,
circuit breakers". `retry.py` already does the middle one. This module
adds the other two and composes all three in the order that is the only
sensible one:

    rate limit  ->  circuit breaker  ->  retry  ->  the call

**Why that order.** The rate limiter is outermost because a call that
is about to be throttled should wait *before* consuming a circuit
breaker slot - throttling is not failure. The breaker sits outside the
retry because retrying into a known-dead dependency is precisely the
behaviour a breaker exists to stop: three retries against an outage
turn one user's failure into three units of load on a service that is
already struggling, and N concurrent users into 3N.

## Rate limiting: `aiolimiter`

A leaky bucket, async-native, and about a hundred lines. Reimplementing
it would mean reimplementing the part that is actually subtle - waking
waiters fairly as capacity returns, without a busy loop.

**Scope, stated honestly:** the limiter is per process. Several API
processes each get their own bucket, so the effective limit is
`processes x rate`. That is the right trade for protecting *Praxis*
from a runaway loop, and the wrong tool for honouring a vendor's global
quota. A cross-process limit needs shared state (the `limits` package
over Redis would be the way), and this module does not pretend to
provide one.

## Circuit breaking: written here, after checking

The library-first rule applies, and the check was made rather than
assumed - both candidates failed empirically on this environment:

- `pybreaker` 1.4.1: `CircuitBreaker.call_async` raises
  `NameError: name 'gen' is not defined` on every call, and the circuit
  never opens. A breaker that silently does not break is worse than
  none, because it is trusted.
- `purgatory-circuitbreaker` 0.7.2: imports `pkg_resources` at package
  import, which is absent here, so it cannot be imported at all.

So this is the one piece written by hand, and it is a plain three-state
machine - closed, open, half-open - which is a well-specified thing,
not a novel invention.

**Half-open lets exactly one call through.** The subtlety worth naming:
when the reset timeout expires, a naive breaker flips to closed and
admits everything at once, so a dependency that is still down gets the
full thundering herd it just recovered from. Here, the first caller
after the timeout is the *only* one admitted; everyone else is still
rejected until that trial call reports back.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import structlog
from aiolimiter import AsyncLimiter

from praxis.core.exceptions import ConnectorError
from praxis.core.interfaces import Connector, ConnectorDescription, HealthStatus

T = TypeVar("T")

_logger = structlog.get_logger(__name__)

DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_RESET_TIMEOUT_SECONDS = 30.0


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(ConnectorError):
    """The call was rejected without being attempted.

    A `ConnectorError` subclass so every existing `except ConnectorError`
    at a step boundary keeps working, while a caller that wants to
    distinguish "the dependency is known-down" from "this call failed"
    can catch this specific type - they warrant different messages to a
    user and different alerting.
    """

    def __init__(
        self, connector: str, *, operation: str = "call", retry_after_seconds: float
    ) -> None:
        super().__init__(
            f"circuit for connector '{connector}' is open",
            connector_name=connector,
            operation=operation,
            # Zero, and meaningfully so: the call was rejected, never
            # attempted. A caller reading `attempts` to decide whether
            # the dependency was actually touched gets the right answer.
            attempts=0,
            detail=f"retry in {retry_after_seconds:.0f}s",
        )
        self.connector = connector
        self.retry_after_seconds = retry_after_seconds


@dataclass
class CircuitBreaker:
    """A three-state breaker for one connector.

    Not thread-safe by design - it guards an asyncio-concurrent
    resource and is protected by its own `asyncio.Lock`, which is
    cheaper and more correct here than a threading primitive that would
    block the event loop.
    """

    name: str
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    reset_timeout_seconds: float = DEFAULT_RESET_TIMEOUT_SECONDS

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _half_open_in_flight: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def _retry_after(self) -> float:
        elapsed = time.monotonic() - self._opened_at
        return max(0.0, self.reset_timeout_seconds - elapsed)

    async def _admit(self, operation: str) -> None:
        """Decides whether this call may proceed, or raises."""
        async with self._lock:
            if self._state is CircuitState.CLOSED:
                return

            if self._state is CircuitState.OPEN:
                if self._retry_after() > 0:
                    raise CircuitOpenError(
                        self.name,
                        operation=operation,
                        retry_after_seconds=self._retry_after(),
                    )
                # Timeout elapsed: promote to half-open and let *this*
                # caller be the single trial.
                self._state = CircuitState.HALF_OPEN
                self._half_open_in_flight = True
                _logger.info("circuit half-open; admitting one trial call", connector=self.name)
                return

            # HALF_OPEN: exactly one trial at a time (see module
            # docstring on the thundering herd).
            if self._half_open_in_flight:
                raise CircuitOpenError(
                    self.name,
                    operation=operation,
                    retry_after_seconds=self.reset_timeout_seconds,
                )
            self._half_open_in_flight = True

    async def _record_success(self) -> None:
        async with self._lock:
            was = self._state
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._half_open_in_flight = False
            if was is not CircuitState.CLOSED:
                _logger.info("circuit closed after successful trial", connector=self.name)

    async def _record_failure(self) -> None:
        async with self._lock:
            self._half_open_in_flight = False
            self._consecutive_failures += 1

            # A failed trial re-opens immediately and restarts the
            # clock: the dependency is demonstrably still down, so
            # counting back up to the threshold would admit
            # `failure_threshold` more doomed calls for no information.
            if self._state is CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                _logger.warning("circuit re-opened; trial call failed", connector=self.name)
                return

            if self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                _logger.warning(
                    "circuit opened",
                    connector=self.name,
                    consecutive_failures=self._consecutive_failures,
                    reset_timeout_seconds=self.reset_timeout_seconds,
                )

    async def call(
        self, operation: Callable[[], Awaitable[T]], *, operation_name: str = "call"
    ) -> T:
        """Runs `operation` under the breaker.

        Raises `CircuitOpenError` without calling `operation` when the
        circuit is open. Any other exception propagates unchanged after
        being counted - the breaker observes failures, it does not
        translate them, so a caller still sees the connector's real
        error.
        """
        await self._admit(operation_name)
        try:
            result = await operation()
        except Exception:
            await self._record_failure()
            raise
        await self._record_success()
        return result

    def reset(self) -> None:
        """Forces the circuit closed. For operator intervention and tests."""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._half_open_in_flight = False


@dataclass(frozen=True)
class RateLimit:
    """How many calls per period one connector may make."""

    max_calls: int
    period_seconds: float = 1.0


class ConnectorResilience:
    """One rate limiter and one circuit breaker per connector name.

    Held per registry rather than globally: two tenants' registries
    should not share a breaker, or one tenant's outage would reject the
    other's calls to a connector that is working fine for them.
    """

    def __init__(
        self,
        *,
        default_rate_limit: RateLimit | None = None,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        reset_timeout_seconds: float = DEFAULT_RESET_TIMEOUT_SECONDS,
    ) -> None:
        self._default_rate_limit = default_rate_limit
        self._failure_threshold = failure_threshold
        self._reset_timeout_seconds = reset_timeout_seconds
        self._limiters: dict[str, AsyncLimiter] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self._rate_limits: dict[str, RateLimit] = {}

    def configure(self, connector: str, rate_limit: RateLimit) -> None:
        """Sets a per-connector rate limit, replacing any default."""
        self._rate_limits[connector] = rate_limit
        self._limiters.pop(connector, None)

    def breaker(self, connector: str) -> CircuitBreaker:
        breaker = self._breakers.get(connector)
        if breaker is None:
            breaker = CircuitBreaker(
                name=connector,
                failure_threshold=self._failure_threshold,
                reset_timeout_seconds=self._reset_timeout_seconds,
            )
            self._breakers[connector] = breaker
        return breaker

    def _limiter(self, connector: str) -> AsyncLimiter | None:
        limit = self._rate_limits.get(connector, self._default_rate_limit)
        if limit is None:
            return None
        limiter = self._limiters.get(connector)
        if limiter is None:
            limiter = AsyncLimiter(limit.max_calls, limit.period_seconds)
            self._limiters[connector] = limiter
        return limiter

    async def call(
        self,
        connector: str,
        operation: Callable[[], Awaitable[T]],
        *,
        operation_name: str = "call",
    ) -> T:
        """Runs `operation` rate-limited and breaker-guarded.

        Composition order is the one argued for in the module docstring:
        throttle first (waiting is not failure), then the breaker, then
        the operation - which the caller may already have wrapped in
        `call_with_retry`, keeping retries *inside* the breaker.
        """
        limiter = self._limiter(connector)
        if limiter is None:
            return await self.breaker(connector).call(operation, operation_name=operation_name)
        async with limiter:
            return await self.breaker(connector).call(operation, operation_name=operation_name)

    def states(self) -> dict[str, str]:
        """Every known connector's circuit state - for the health endpoint."""
        return {name: breaker.state.value for name, breaker in self._breakers.items()}


class ResilientConnector(Connector):
    """Wraps a `Connector` so its calls are throttled and breaker-guarded.

    A proxy rather than a base class, so it composes with every existing
    connector without any of them knowing it exists - which is the whole
    point: resilience is a property of *calling* a connector, not
    something each of the thirteen has to remember to implement.

    **`health()` deliberately bypasses both.** A health check exists to
    report the dependency's real state; routing it through an open
    circuit would make the connector report "unhealthy: circuit is open"
    forever, since the only thing that could close the circuit is a
    successful call the breaker is refusing to make. Health checks are
    also exactly what an operator hits *during* an outage, so they must
    not be rate limited away.
    """

    def __init__(self, inner: Connector, resilience: ConnectorResilience) -> None:
        self._inner = inner
        self._resilience = resilience
        self.name = inner.name
        self.read_only = inner.read_only

    @property
    def inner(self) -> Connector:
        """The wrapped connector.

        Needed because a proxy is not an `isinstance` of what it wraps:
        a caller that genuinely needs the concrete type (a test
        asserting `build_registry` produced an `MCPConnector`, say)
        unwraps explicitly rather than having the proxy pretend.
        """
        return self._inner

    def __getattr__(self, attribute: str) -> Any:
        """Forwards anything this proxy does not define to the wrapped
        connector.

        Connectors carry their own configuration and helpers beyond the
        four ABC methods, and a proxy that hid them would turn "add
        resilience" into "break every caller that touches a connector's
        own surface". Only called for attributes normal lookup missed,
        so it never shadows `read`/`write`/`describe`/`health`.
        """
        return getattr(self._inner, attribute)

    async def describe(self) -> ConnectorDescription:
        return await self._resilience.call(
            self.name, self._inner.describe, operation_name="describe"
        )

    async def read(self, query: str, **params: Any) -> Any:
        return await self._resilience.call(
            self.name, lambda: self._inner.read(query, **params), operation_name="read"
        )

    async def write(self, action: str, **params: Any) -> Any:
        return await self._resilience.call(
            self.name, lambda: self._inner.write(action, **params), operation_name="write"
        )

    async def health(self) -> HealthStatus:
        # See the class docstring: never guarded.
        return await self._inner.health()
