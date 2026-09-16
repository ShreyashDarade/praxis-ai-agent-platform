# praxis/cache/memory_cache.py
"""`InMemoryCache` (spec §11): the one `Cache` implementation for this
MVP - "one `Cache` interface, in-process TTL/LRU for MVP" - backing all
four caching scopes spec §11 names (LLM response, embedding,
connector/schema-introspection, tool-result). Each scope owns its own
instance (a fresh, task-scoped one for the tool-result scope; longer-
lived, component-owned ones for the other three) - never one cache
shared across unrelated scopes, so a policy tuned for one scope (e.g.
the tool-result cache's "lives exactly as long as one task") never
leaks into another.
"""
from __future__ import annotations

import time
from typing import Any

from praxis.core.interfaces import Cache


class InMemoryCache(Cache):
    """A real TTL-aware in-process cache: a plain dict keyed on the
    caller's own cache key, storing `(value, expires_at)`.

    `get()` returns `None` for an absent OR expired key, evicting an
    expired entry the moment it's noticed - so a cache never grows
    unboundedly from expired entries nobody ever reads again. `set()`
    with `ttl_seconds=None` (the default) means no expiry at all: the
    entry lives until process end or an explicit overwrite.

    Not thread-safe by design - single-process asyncio is this MVP's
    only concurrency model (spec: "Thread-safety isn't a concern").
    `get`/`set` are genuinely `async def` purely to satisfy the `Cache`
    ABC's contract; a future backend (e.g. Redis) swapped in behind the
    same interface is exactly where a real `await` would start to
    matter.
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, float | None]] = {}

    async def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and expires_at <= time.monotonic():
            del self._data[key]
            return None
        return value

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        expires_at = time.monotonic() + ttl_seconds if ttl_seconds is not None else None
        self._data[key] = (value, expires_at)

    def __len__(self) -> int:
        return len(self._data)
