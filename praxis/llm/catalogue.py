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
"""
from __future__ import annotations

from typing import Any

import anthropic

# Verified-available Claude model ids as of this phase. Do not invent
# other model name strings - these are the only three in use.
DEFAULT_MODEL_MAPPING: dict[str, str] = {
    "routing": "claude-haiku-4-5-20251001",
    "planning": "claude-sonnet-5",
    "code_synthesis": "claude-sonnet-5",
    "vision": "claude-sonnet-5",
}


class LLMCatalogue:
    """Maps a purpose string to a model id, and runs completions against it."""

    def __init__(self, model_mapping: dict[str, str] | None = None) -> None:
        self._model_mapping: dict[str, str] = (
            dict(model_mapping) if model_mapping is not None else dict(DEFAULT_MODEL_MAPPING)
        )

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

        Constructs a fresh `anthropic.AsyncAnthropic()` client with no
        `api_key` kwarg - the SDK reads `ANTHROPIC_API_KEY` from the
        environment by default. Returns the concatenated text of every
        text content block in the response.
        """
        model = self.model_for(purpose)
        max_tokens = kwargs.pop("max_tokens", 1024)
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return "".join(block.text for block in response.content if block.type == "text")
