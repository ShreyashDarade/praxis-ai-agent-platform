# praxis/cache/redis_cache.py
"""`RedisCache`: the distributed `Cache` backend (brief §10's
"distributed cache invalidation").

`InMemoryCache` is correct and fast and is wrong the moment there are
two processes: each worker warms its own copy, a hit rate is divided by
the worker count, and there is no way to invalidate an entry another
process is holding. This is the same `Cache` interface backed by a
shared Redis keyspace, so the decision between them is which object a
component is constructed with and nothing else.

**Degrading instead of failing.** Every Redis call here is wrapped, and
a connection failure, timeout or protocol error is logged and reported
as a cache miss. That is the whole point: a cache exists to avoid work,
so a cache that is down must cost the deployment its speedup and
nothing else. A `RedisCache` that raised on `get()` would turn an
optional dependency into a hard one and take the platform down with
Redis - strictly worse than having no cache at all. The failures are
counted (`degraded_operations`) and logged rather than swallowed
silently, so "Redis is down" shows up as a fact rather than as an
unexplained collapse in hit rate.

**What JSON serialization costs.** Values are stored as JSON, which is
what makes an entry readable by a differently-versioned worker, by a
different language, and by a human debugging a keyspace - a pickle
would also be a remote-code-execution primitive on a shared store
anyone can write to. The honest limitations that follow, none of which
apply to `InMemoryCache`:

- A tuple comes back as a list, and a dict with non-string keys comes
  back with string keys. JSON has no other option.
- A value JSON cannot represent at all (a dataclass, a datetime, a set)
  is *not* stored: `set()` logs and returns rather than raising, so a
  caller written against `InMemoryCache` keeps working and simply stops
  getting hits. Callers store plain JSON-shaped data - which every
  current call site already does.
- `None` is indistinguishable from "absent", because `Cache.get`'s own
  contract already collapses the two.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from praxis.core.interfaces import Cache

_logger = structlog.get_logger(__name__)

DEFAULT_REDIS_URL = "redis://localhost:6379/0"

# Short on purpose. Every caller of a cache is on a latency-sensitive
# path and has a working fallback (do the real work), so waiting on an
# unresponsive Redis is strictly worse than declaring a miss and moving
# on. A longer timeout would convert "Redis is slow" into "Praxis is
# slow", which is exactly the coupling this class exists to avoid.
DEFAULT_TIMEOUT_SECONDS = 2.0

# The errors that mean "the cache is unavailable right now", as opposed
# to a bug in the caller. `RedisError` covers the library's own
# connection/timeout/protocol hierarchy; `OSError` catches socket- and
# DNS-level failures raised before the library wraps them.
# `asyncio.TimeoutError` is listed separately because it is not an
# `OSError` and is what surfaces if a call is bounded outside the
# client. `asyncio.CancelledError` is deliberately absent - it is a
# `BaseException` and cancelling a request must cancel it, not be
# reported as a cache miss.
_UNAVAILABLE = (RedisError, OSError, asyncio.TimeoutError)


class RedisCache(Cache):
    """A `Cache` over a shared Redis keyspace.

    Construction never connects: `Redis.from_url` builds a lazy
    connection pool, so an unreachable Redis is discovered on first use
    and handled as degradation there, rather than making the whole
    process fail to start over an optional dependency.

    `namespace` is prefixed to every key so several deployments (or a
    test run and a live process) can share one Redis instance without
    colliding. It is not a security boundary - tenancy lives inside the
    key itself, via `praxis.cache.keys.CacheKey`.

    `client` lets a caller inject an already-configured `Redis` (a
    cluster client, one with TLS or auth options this constructor does
    not expose); when supplied, `url` and `timeout_seconds` are ignored
    because they are the injected client's own concern.
    """

    def __init__(
        self,
        url: str = DEFAULT_REDIS_URL,
        *,
        namespace: str = "praxis",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: Redis | None = None,
    ) -> None:
        self._url = url
        self._namespace = namespace
        self._timeout_seconds = timeout_seconds
        self._client: Redis = client if client is not None else Redis.from_url(
            url,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
            # Values are JSON text, so decoding in the client saves
            # every call site a `.decode()` and makes a corrupt entry
            # surface as a `UnicodeDecodeError` here rather than as a
            # confusing bytes object three frames away.
            decode_responses=True,
            # A retry would multiply the timeout above by the retry
            # count while the caller waits for a value it can compute
            # itself. Fail fast, miss, move on.
            retry_on_timeout=False,
        )
        # Counted rather than merely logged: a test can assert that a
        # dead Redis produced a real degraded operation instead of a
        # hit, and an operator can alert on the counter without parsing
        # log lines. Same rationale as `LLMCatalogue.real_api_calls`.
        self.degraded_operations = 0
        self.last_error: str | None = None

    def _namespaced(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    def _degrade(self, operation: str, key: str, exc: BaseException) -> None:
        self.degraded_operations += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        _logger.warning(
            "redis_cache_unavailable",
            operation=operation,
            key=key,
            error=str(exc),
            error_type=type(exc).__name__,
        )

    async def get(self, key: str) -> Any | None:
        """Returns the cached value, or `None` for an absent, expired,
        corrupt or unreachable entry.

        All four collapse to `None` because all four mean the same thing
        to a caller: do the work yourself. They are distinguished in the
        logs and in `degraded_operations`, which is where the difference
        actually matters.
        """
        try:
            raw = await self._client.get(self._namespaced(key))
        except _UNAVAILABLE as exc:
            self._degrade("get", key, exc)
            return None
        except UnicodeDecodeError as exc:
            # `decode_responses=True` makes the client decode the value
            # as UTF-8 while parsing, so a binary value written by
            # something other than this class fails there rather than in
            # `json.loads` below. Still a miss, never an exception.
            _logger.warning("redis_cache_undecodable_entry", key=key, error=str(exc))
            return None

        if raw is None:
            return None

        try:
            return json.loads(raw)
        except (TypeError, ValueError) as exc:
            # Written by something that did not go through this class,
            # or truncated. Treated as a miss and reported, never as an
            # exception to the caller.
            _logger.warning("redis_cache_undecodable_entry", key=key, error=str(exc))
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Stores `value` as JSON, optionally expiring after
        `ttl_seconds`.

        `ttl_seconds=None` means no expiry, matching `InMemoryCache`. A
        non-positive TTL means "already expired", which `InMemoryCache`
        supports by storing an entry nobody can read; Redis rejects
        `EX 0` outright, so it is implemented here as a delete - the
        observable behavior a caller depends on (a subsequent `get`
        misses) is identical, and any previously cached value is
        correctly displaced rather than left in place.
        """
        try:
            payload = json.dumps(value)
        except (TypeError, ValueError) as exc:
            _logger.warning(
                "redis_cache_unserializable_value",
                key=key,
                value_type=type(value).__name__,
                error=str(exc),
            )
            return

        namespaced = self._namespaced(key)
        try:
            if ttl_seconds is None:
                await self._client.set(namespaced, payload)
            elif ttl_seconds <= 0:
                await self._client.delete(namespaced)
            else:
                await self._client.set(namespaced, payload, ex=int(ttl_seconds))
        except _UNAVAILABLE as exc:
            self._degrade("set", key, exc)

    async def invalidate_prefix(self, prefix: str) -> int:
        """Deletes every key under `prefix` and returns how many were
        removed; `0` when Redis is unreachable.

        This is the distributed half of invalidation the brief asks for,
        and pairs with `CacheKey.tenant_prefix` - "drop everything this
        tenant has cached for dashboards" is one call here rather than a
        per-process sweep that can only ever reach one worker.

        Uses `scan_iter`, never `KEYS`: `KEYS` blocks the whole server
        for the duration of the scan, which on a shared cache means
        blocking every other tenant's reads to invalidate one tenant's
        writes.
        """
        pattern = f"{self._namespaced(prefix)}*"
        removed = 0
        try:
            async for found in self._client.scan_iter(match=pattern, count=500):
                removed += await self._client.delete(found)
        except _UNAVAILABLE as exc:
            self._degrade("invalidate_prefix", prefix, exc)
            return removed
        return removed

    async def ping(self) -> bool:
        """Whether Redis is currently reachable.

        Separate from `get`/`set` precisely because those must never
        report unavailability to their caller: this is how a health
        check or a startup probe asks the question directly.
        """
        try:
            await self._client.ping()
        except _UNAVAILABLE as exc:
            self._degrade("ping", "", exc)
            return False
        return True

    async def close(self) -> None:
        """Releases the connection pool. Safe to call more than once,
        and safe when nothing ever connected."""
        try:
            await self._client.aclose()
        except _UNAVAILABLE as exc:
            _logger.warning("redis_cache_close_failed", error=str(exc))


def redis_cache_from_settings(settings: Any) -> RedisCache | None:
    """Builds a `RedisCache` from `Settings`, or `None` when no Redis
    URL is configured.

    `None` rather than an error for an unset URL, matching how every
    other optional backend in `praxis.config` behaves: not configuring
    a distributed cache means the deployment uses the in-process one,
    which is a supported configuration, not a misconfiguration.
    """
    url = getattr(settings, "redis_url", None)
    if not url:
        return None
    return RedisCache(url, namespace=getattr(settings, "redis_namespace", "praxis"))
