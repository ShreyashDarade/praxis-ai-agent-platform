# praxis/llm/providers.py
"""Which provider serves a model id, and the call-shape each one wants.

`LLMCatalogue`'s premise is that a model is chosen by *purpose* and
that swapping providers is a config change. That premise only holds if
the provider is derived from the model id rather than configured
alongside it: two knobs that can disagree (`model="gpt-5"`,
`provider="anthropic"`) produce a deployment that fails at the first
call with a confusing error. So the id is the single source of truth
and this module reads the provider off it.

The two SDKs are not interchangeable, and the differences are not
cosmetic:

- **Anthropic** takes `max_tokens` and returns `content` blocks.
- **OpenAI chat completions** returns `choices[].message.content`, and
  its *reasoning* models (gpt-5, o1/o3/o4) reject `max_tokens` in
  favour of `max_completion_tokens` and reject any `temperature` other
  than the default.

That last point is the one that silently produces empty answers rather
than errors, and it is why **no output cap is sent unless a caller
asks for one**. A reasoning model spends tokens thinking before it
emits any visible text, and one allowance covers both: gpt-5 given
`max_completion_tokens=1024` on a long planning prompt spends all 1024
reasoning and returns `""` with `finish_reason="length"` - not a
truncated plan, an empty one. Downstream that surfaced as "Planner
response was not valid JSON ... raw response: ''", which points at
the parser rather than at the cap that actually caused it.

So `max_tokens=None` means uncapped: the parameter is omitted and the
model's own maximum applies. A caller that passes an explicit cap
still gets it, and if that cap truncates the reply to nothing, the
call raises `OutputTruncatedError` naming the cause rather than
returning an empty string for something else to misdiagnose.

"Uncapped" means different things to the two SDKs, and neither is a
hardcoded number here. OpenAI simply omits the parameter. Anthropic
*requires* `max_tokens`, so the ceiling is discovered from the API
itself: the request asks for more than any model allows, and the 400
that comes back names the real maximum ("max_tokens: 1000000 > 128000,
which is the maximum allowed number of output tokens for
claude-sonnet-5"), which is parsed, cached per model and used. One
extra request per model per process buys a limit that is correct by
construction and cannot go stale the way a table of per-model ceilings
would the next time a model ships.

Anthropic calls are streamed throughout, because a large `max_tokens`
makes the SDK refuse a non-streaming request outright ("Streaming is
required for operations that may take longer than 10 minutes"), and
streaming is equally correct for small ones.

Credentials are read from the environment by each SDK, as before - the
Anthropic client already worked this way, and keeping it uniform means
a key never has to travel through Praxis's own config objects or logs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

ANTHROPIC = "anthropic"
OPENAI = "openai"

# Deliberately above any model's real ceiling: the point is to be
# refused, so the refusal can be read for the true maximum.
_ANTHROPIC_CEILING_PROBE = 1_000_000

# "max_tokens: 1000000 > 128000, which is the maximum allowed number of
# output tokens for claude-sonnet-5"
_CEILING_RE = re.compile(
    r">\s*(\d+),\s*which is the maximum allowed number of output tokens"
)

# Discovered ceilings, per model, for this process.
_anthropic_ceilings: dict[str, int] = {}

_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
# Reasoning-family ids, which take `max_completion_tokens` and refuse a
# non-default temperature.
_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class UnknownProviderError(ValueError):
    """A model id no configured provider recognises."""


class OutputTruncatedError(RuntimeError):
    """A cap cut the reply off before any usable text was produced.

    Raised instead of returning the empty string, so the error names
    the cap that caused it rather than leaving the next layer to
    report whatever it failed to parse.
    """


def provider_for(model: str) -> str:
    """The provider that serves `model`, from the id alone.

    Raises rather than defaulting: a typo'd id that quietly fell
    through to one provider would be reported by that provider as an
    unknown *model*, which sends whoever is debugging it to the wrong
    place entirely.
    """
    name = model.strip().lower()
    if name.startswith("claude-"):
        return ANTHROPIC
    if name.startswith(_OPENAI_PREFIXES):
        return OPENAI
    raise UnknownProviderError(
        f"no provider is registered for model '{model}'; ids beginning 'claude-' route to "
        f"Anthropic and {list(_OPENAI_PREFIXES)} to OpenAI"
    )


def is_reasoning_model(model: str) -> bool:
    return model.strip().lower().startswith(_OPENAI_REASONING_PREFIXES)


@dataclass(frozen=True)
class Completion:
    """One provider response, normalised.

    Token counts come from the provider's own usage report rather than
    an estimate, because they are what the budget is charged against.
    """

    text: str
    tokens_in: int
    tokens_out: int


async def complete(
    model: str, prompt: str, *, max_tokens: int | None = None, **kwargs: Any
) -> Completion:
    """Runs one completion against whichever provider serves `model`.

    `max_tokens=None` (the default) means uncapped - see the module
    docstring on why a default cap is actively harmful for reasoning
    models.
    """
    which = provider_for(model)
    if which == ANTHROPIC:
        return await _complete_anthropic(model, prompt, max_tokens=max_tokens, **kwargs)
    return await _complete_openai(model, prompt, max_tokens=max_tokens, **kwargs)


async def _complete_anthropic(
    model: str, prompt: str, *, max_tokens: int | None, **kwargs: Any
) -> Completion:
    if max_tokens is not None:
        return await _anthropic_call(model, prompt, max_tokens, **kwargs)

    known = _anthropic_ceilings.get(model)
    if known is not None:
        return await _anthropic_call(model, prompt, known, **kwargs)

    import anthropic

    try:
        result = await _anthropic_call(model, prompt, _ANTHROPIC_CEILING_PROBE, **kwargs)
    except anthropic.BadRequestError as exc:
        found = _CEILING_RE.search(str(exc))
        if found is None:
            # A 400 about something else entirely - a bad model id, a
            # malformed prompt. Re-raised untouched rather than
            # reported as a token-limit problem it is not.
            raise
        ceiling = int(found.group(1))
        _anthropic_ceilings[model] = ceiling
        return await _anthropic_call(model, prompt, ceiling, **kwargs)

    # The probe was accepted, so this model has no ceiling below it.
    _anthropic_ceilings[model] = _ANTHROPIC_CEILING_PROBE
    return result


async def _anthropic_call(
    model: str, prompt: str, max_tokens: int, **kwargs: Any
) -> Completion:
    """One streamed Anthropic call. See the module docstring on why
    every call is streamed rather than only the large ones."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    ) as stream:
        message = await stream.get_final_message()

    text = "".join(block.text for block in message.content if block.type == "text")
    usage = getattr(message, "usage", None)
    return Completion(
        text=text,
        tokens_in=int(getattr(usage, "input_tokens", 0) or 0),
        tokens_out=int(getattr(usage, "output_tokens", 0) or 0),
    )


async def _complete_openai(
    model: str, prompt: str, *, max_tokens: int | None, **kwargs: Any
) -> Completion:
    import openai

    client = openai.AsyncOpenAI()
    call: dict[str, Any] = dict(kwargs)
    reasoning = is_reasoning_model(model)

    if max_tokens is not None:
        # Only sent when a caller actually asked for a cap. Omitting
        # the parameter lets the model's own maximum apply, which is
        # what keeps a long plan from being truncated to nothing.
        call["max_completion_tokens" if reasoning else "max_tokens"] = max_tokens

    if reasoning:
        # Dropped rather than passed through: these models reject any
        # non-default value outright, so forwarding a caller's 0.2
        # turns a working call into a 400. Dropping it costs nothing -
        # the model does not honour it either way - while passing it
        # breaks the call.
        call.pop("temperature", None)
        call.pop("top_p", None)

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **call,
    )
    choice = response.choices[0] if response.choices else None
    text = (getattr(choice.message, "content", None) if choice else None) or ""
    usage = getattr(response, "usage", None)

    if not text.strip() and choice is not None and choice.finish_reason == "length":
        details = getattr(usage, "completion_tokens_details", None)
        spent = getattr(details, "reasoning_tokens", None) if details else None
        raise OutputTruncatedError(
            f"model '{model}' hit its output cap of {max_tokens} tokens before producing "
            f"any text"
            + (f" ({spent} of them spent on reasoning)" if spent else "")
            + "; raise max_tokens or omit it to leave the reply uncapped"
        )

    return Completion(
        text=text,
        tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
        tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
    )
