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
text with zero real API call. The key is built through
`praxis.cache.keys.CacheKey`, so when a caller passes a `principal` it
also carries that principal's tenant and effective permissions, and the
tenant is rechecked on the hit before the cached text is returned
(brief §10: "Cache keys must include tenant, effective permissions ...
Recheck authorization on cache hits"). `principal` is optional: a
caller that has none - the auth-disabled single-operator deployment,
and every call site written before tenancy existed - gets an
untenanted key in a keyspace that cannot collide with any tenant's.
The model id is part of the key, which is what makes a model swap a
different entry rather than a stale one. `real_api_calls` is a plain counter
attribute, incremented only immediately before an actual
`client.messages.create` call - real, load-bearing testability (not
guessed from wall-clock timing, which is flaky): a test asserting
`catalogue.real_api_calls == 1` after two identical `complete()` calls
is asserting on the one thing that actually matters, "did a second real
network call happen," not on how long either call took.
"""
from __future__ import annotations

from typing import Any

from praxis.agents.budget import BudgetTracker
from praxis.cache import scopes
from praxis.cache.keys import CacheKey
from praxis.cache.memory_cache import InMemoryCache
from praxis.core.interfaces import Cache
from praxis.llm import providers
from praxis.observability.metrics import (
    LLM_LATENCY,
    metrics,
    record_cache,
    record_llm_call,
)
from praxis.security.principal import Principal

# Rough prompt-size estimate for the PRE-call budget check only. It
# deliberately does not try to be accurate: the real token counts come
# from the provider's own usage report after the call. This exists so
# a budget can refuse an obviously-too-large call before paying for
# it, and ~4 characters per token is the standard rule of thumb for
# English prose.
_CHARS_PER_TOKEN = 4


def _estimate_tokens(prompt: str) -> int:
    return max(1, len(prompt) // _CHARS_PER_TOKEN)

# Verified-available Claude model ids as of this phase. Do not invent
# other model name strings - these are the only ones in use.
DEFAULT_MODEL_MAPPING: dict[str, str] = {
    "routing": "claude-haiku-4-5-20251001",
    "planning": "claude-sonnet-5",
    "code_synthesis": "claude-sonnet-5",
    "vision": "claude-sonnet-5",
    # Phase 11 (spec §16.1's "Diagnosis subagent" step): a real
    # reasoning task over a real fetched metric, warranting the same
    # strong-tier model as planning/code_synthesis, not the cheap
    # routing tier.
    "diagnosis": "claude-sonnet-5",
    # Composing the assistant's reply from what actually ran
    # (`praxis.agents.answer`). Strong tier deliberately: this is
    # the text a user reads and acts on, and the failure mode of a
    # weak model here is a fluent answer that misreads its own
    # evidence - which is worse than no answer, because it looks
    # exactly like a good one.
    "answering": "claude-sonnet-5",
}

# No expiry-by-default is deliberately too aggressive for an LLM whose
# provider-side weights/behavior can change; a bounded TTL means a
# long-running process eventually re-asks rather than serving a
# same-process-lifetime-stale response forever.
# Read from `praxis.cache.scopes` rather than restated here, so the TTL
# and the reasoning for it stay in one place.
DEFAULT_CACHE_TTL_SECONDS = scopes.ttl_for(scopes.LLM_RESPONSE)


def _cache_key(
    purpose: str,
    model: str,
    prompt: str,
    max_tokens: int | None,
    kwargs: dict[str, Any],
    principal: Principal | None = None,
) -> CacheKey:
    """The `llm_response` key for one completion.

    The model id is named explicitly rather than left implicit in the
    purpose: `DEFAULT_MODEL_MAPPING` is config, so the same purpose can
    resolve to a different model after a redeploy, and a key that
    omitted the model would serve the old model's answer as the new
    one's.
    """
    return CacheKey.build(
        scopes.LLM_RESPONSE,
        principal,
        purpose=purpose,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        kwargs=kwargs,
    )


def _configured_overrides() -> dict[str, str]:
    """`Settings.llm_model_overrides`, or nothing if unreadable.

    Never raises. `LLMCatalogue()` is constructed in six places
    including at import time, and several of them predate `Settings`
    being constructible in that context; a deployment with no database
    URL configured must still be able to build a catalogue with its
    defaults rather than fail on an unrelated missing setting.
    """
    try:
        from praxis.config import Settings

        return dict(Settings().llm_model_overrides)
    except Exception:  # noqa: BLE001 - see docstring
        return {}


class LLMCatalogue:
    """Maps a purpose string to a model id, and runs completions against it."""

    def __init__(
        self,
        model_mapping: dict[str, str] | None = None,
        *,
        cache: Cache | None = None,
        cache_ttl_seconds: int | None = DEFAULT_CACHE_TTL_SECONDS,
        budget: BudgetTracker | None = None,
    ) -> None:
        # An explicit mapping wins outright (a test forcing every
        # purpose onto one model must not have deployment config
        # quietly reintroduced underneath it). Otherwise the defaults
        # are taken and `Settings.llm_model_overrides` applied on top,
        # which is how a deployment moves one purpose to another
        # provider without touching code.
        if model_mapping is not None:
            self._model_mapping: dict[str, str] = dict(model_mapping)
        else:
            self._model_mapping = dict(DEFAULT_MODEL_MAPPING)
            self._model_mapping.update(_configured_overrides())
        self._cache: Cache = cache if cache is not None else InMemoryCache()
        self._cache_ttl_seconds = cache_ttl_seconds
        # Phase 14: optional, so every existing caller is unbudgeted
        # exactly as before. When supplied, the budget is checked
        # before each call and charged after it.
        self._budget = budget
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

    async def complete(
        self, purpose: str, prompt: str, *, principal: Principal | None = None, **kwargs: Any
    ) -> str:
        """Runs one completion against the model registered for `purpose`.

        A call with the same `(purpose, prompt, model, max_tokens,
        **kwargs)` - and the same tenant and effective permissions,
        when a `principal` is supplied - within the cache's TTL returns
        the cached response text: no client is constructed and no API
        call is made at all on a cache hit. On a miss, the call is
        dispatched by `praxis.llm.providers` to whichever provider the
        model id belongs to, each SDK reading its own key from the
        environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`), and the
        response text is cached before it is returned.

        `principal` is keyword-only and is never forwarded to the
        provider: it identifies who is asking, which decides which
        cache entry may be read, not what the model is asked.
        """
        model = self.model_for(purpose)
        # `None` means uncapped, and it is the default deliberately.
        # The previous default of 1024 silently truncated: on a
        # reasoning model the cap covers thinking as well as the
        # reply, so a long planning prompt spent the whole allowance
        # reasoning and returned an empty string, which surfaced two
        # layers away as an unparseable plan. A caller that genuinely
        # wants a short answer still passes `max_tokens`.
        max_tokens = kwargs.pop("max_tokens", None)
        cache_key = _cache_key(purpose, model, prompt, max_tokens, kwargs, principal)

        cached = await self._cache.get(cache_key.value)
        if cached is not None:
            # Rechecked before the value is handed back, not merely
            # implied by the key having matched (brief §10). With a
            # shared Redis keyspace the key is not proof of who wrote
            # the entry.
            cache_key.authorize(principal)
            record_cache(True, scope=scopes.LLM_RESPONSE)
            return cached
        record_cache(False, scope=scopes.LLM_RESPONSE)

        # Phase 14: enforce the budget BEFORE the call, so an overrun
        # is prevented rather than merely recorded afterwards. No
        # tracker configured means unbudgeted, which is the historical
        # behavior for every caller that does not supply one.
        if self._budget is not None:
            self._budget.check_llm_call(model, estimated_tokens=_estimate_tokens(prompt))

        # Which provider serves this model is derived from the model
        # id (`praxis.llm.providers`), so a purpose can be pointed at
        # an OpenAI model with no code change - which is what the
        # "swapping providers is a config change" promise at the top
        # of this module actually requires.
        self.real_api_calls += 1
        with metrics.time(LLM_LATENCY, {"model": model, "purpose": purpose}):
            completion = await providers.complete(
                model, prompt, max_tokens=max_tokens, **kwargs
            )
        text = completion.text

        # Phase 17: real token and cost accounting, read off the
        # provider's own usage report rather than estimated - an
        # estimate would make the cost guardrail systematically wrong
        # in whichever direction the estimator is biased.
        tokens_in = completion.tokens_in
        tokens_out = completion.tokens_out
        cost_usd = 0.0
        if self._budget is not None:
            cost_usd = self._budget.record_llm_call(model, tokens_in, tokens_out)
        else:
            cost_usd = BudgetTracker().price_of(model, tokens_in, tokens_out)
        record_llm_call(
            model=model,
            purpose=purpose,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )

        await self._cache.set(cache_key.value, text, ttl_seconds=self._cache_ttl_seconds)
        return text
