# praxis/llm/catalogue.py
"""LLM Catalogue: purpose -> model registry (spec §7).

"Orchestrator/Factory select by purpose, never hardcode a model name -
swapping providers is a config change." This module is the one place a
model id string is allowed to appear; every other caller in the
codebase asks for a *purpose* (``"routing"``, ``"planning"``,
``"code_synthesis"``, ``"vision"``) and gets the right model back.

A plain dict is the "config-driven registry" for this MVP - no external
config file is required by the spec, only that model selection isn't
hardcoded inline at every call site. The mapping is still constructor-
injectable so a caller (or a test) can override it, e.g. to force every
purpose onto the cheapest model regardless of its default tier.

**LLM response cache (Phase 7, spec §11)**: `complete()` is fronted by a
`Cache` keyed on `(purpose, model, prompt, max_tokens, sorted **kwargs)`
- an identical call within the cache's TTL returns the cached response
text with zero real API call. `real_api_calls` is a plain counter
attribute, incremented only immediately before an actual
`client.messages.create` call - real, load-bearing testability (not
guessed from wall-clock timing, which is flaky): a test asserting
`catalogue.real_api_calls == 1` after two identical `complete()` calls
is asserting on the one thing that actually matters, "did a second real
network call happen," not on how long either call took.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import anthropic

from praxis.cache.memory_cache import InMemoryCache
from praxis.core.interfaces import Cache

# Verified-available Claude model ids as of this phase. Do not invent
# other model name strings - these are the only three in use.
DEFAULT_MODEL_MAPPING: dict[str, str] = {
    "routing": "claude-haiku-4-5-20251001",
    "planning": "claude-sonnet-5",
    "code_synthesis": "claude-sonnet-5",
    "vision": "claude-sonnet-5",
}

# No expiry-by-default is deliberately too aggressive for an LLM whose
# provider-side weights/behavior can change; a bounded TTL means a
# long-running process eventually re-asks rather than serving a
# same-process-lifetime-stale response forever.
DEFAULT_CACHE_TTL_SECONDS = 3600


def _cache_key(purpose: str, model: str, prompt: str, max_tokens: int, kwargs: dict[str, Any]) -> str:
    payload = json.dumps(
        {"purpose": purpose, "model": model, "prompt": prompt, "max_tokens": max_tokens, "kwargs": kwargs},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMCatalogue:
    """Maps a purpose string to a model id, and runs completions against it."""

    def __init__(
        self,
        model_mapping: dict[str, str] | None = None,
        *,
        cache: Cache | None = None,
        cache_ttl_seconds: int | None = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self._model_mapping: dict[str, str] = (
            dict(model_mapping) if model_mapping is not None else dict(DEFAULT_MODEL_MAPPING)
        )
        self._cache: Cache = cache if cache is not None else InMemoryCache()
        self._cache_ttl_seconds = cache_ttl_seconds
        # Real, load-bearing testability (see module docstring) - counts
        # actual `client.messages.create` calls, never cache hits.
        self.real_api_calls = 0

    def model_for(self, purpose: str) -> str:
        """Returns the model id registered for `purpose`.

        Raises `ValueError` (with the known purposes listed) for
        anything not in the mapping - a typo'd purpose must fail loudly,
        not silently fall back to some default model.
        """
        try:
            return self._model_mapping[purpose]
        except KeyError:
            known = ", ".join(sorted(self._model_mapping)) or "(none registered)"
            raise ValueError(f"unknown LLM purpose '{purpose}' - known purposes: {known}") from None

    async def complete(self, purpose: str, prompt: str, **kwargs: Any) -> str:
        """Runs one completion against the model registered for `purpose`.

        A call with the same `(purpose, prompt, model, max_tokens,
        **kwargs)` within the cache's TTL returns the cached response
        text - no client is constructed and no API call is made at all
        on a cache hit. On a miss, constructs a fresh
        `anthropic.AsyncAnthropic()` client with no `api_key` kwarg -
        the SDK reads `ANTHROPIC_API_KEY` from the environment by
        default - and caches the concatenated text of every text
        content block in the response before returning it.
        """
        model = self.model_for(purpose)
        max_tokens = kwargs.pop("max_tokens", 1024)
        cache_key = _cache_key(purpose, model, prompt, max_tokens, kwargs)

        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        client = anthropic.AsyncAnthropic()
        self.real_api_calls += 1
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        await self._cache.set(cache_key, text, ttl_seconds=self._cache_ttl_seconds)
        return text
