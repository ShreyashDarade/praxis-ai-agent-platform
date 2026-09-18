# praxis/agents/budget.py
"""Budgets and cost guardrails (Prompt §1, §9).

Required by the prompt in two places: *"allocate budgets ... total
descendant cost limits"* and *"budget limits and cost guardrails"*.
Before this there was no notion of cost anywhere in the platform - a
task could make unbounded LLM calls and nothing would notice.

Three design decisions worth stating, because each rules out a
tempting-but-wrong alternative:

1. **Spend is tracked on a shared `BudgetTracker`, not on the
   immutable `Budget`.** The budget is the *allowance* (a value object,
   safely shared and copied); the tracker is the *ledger*. Keeping
   them separate is what lets a parent and all its descendants charge
   against one pooled allowance without any of them being able to
   rewrite the allowance itself.

2. **Descendant spend rolls up to the parent.** A child tracker holds
   a reference to its parent and charges both. Without this, "total
   descendant cost limit" would be unenforceable - each child would
   individually stay under budget while collectively blowing past it.

3. **The check is before the spend, and exhaustion raises.** A budget
   that is only observed after the fact is a report, not a guardrail.
   `BudgetExhaustedError` is raised at the point of the call that
   would exceed it, so the caller fails cleanly with a stated reason
   instead of discovering an overrun later.

Token pricing is deliberately *configurable per model* rather than
hardcoded to current list prices: prices change, and a stale constant
buried in code produces confidently wrong cost numbers, which is worse
than no numbers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


class BudgetExhaustedError(Exception):
    """A budgeted resource would be exceeded by the operation attempted.

    `resource` names which one (`tokens`, `cost_usd`, `llm_calls`,
    `wall_clock_seconds`, `descendants`), with the limit and the
    would-be total, so the failure message states the real arithmetic
    rather than a generic "over budget".
    """

    def __init__(
        self, message: str, *, resource: str, limit: float, attempted: float
    ) -> None:
        super().__init__(message)
        self.resource = resource
        self.limit = limit
        self.attempted = attempted


@dataclass(frozen=True)
class Budget:
    """An allowance. Immutable - the ledger lives in `BudgetTracker`.

    Every field is optional (`None` = unlimited for that dimension) so
    a caller can bound exactly the dimensions it cares about rather
    than being forced to invent numbers for all five.
    """

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_llm_calls: int | None = None
    max_wall_clock_seconds: float | None = None
    max_descendants: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "max_cost_usd": self.max_cost_usd,
            "max_llm_calls": self.max_llm_calls,
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
            "max_descendants": self.max_descendants,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> Budget | None:
        if not payload:
            return None
        return cls(
            max_tokens=payload.get("max_tokens"),
            max_cost_usd=payload.get("max_cost_usd"),
            max_llm_calls=payload.get("max_llm_calls"),
            max_wall_clock_seconds=payload.get("max_wall_clock_seconds"),
            max_descendants=payload.get("max_descendants"),
        )


# Cost per 1M tokens, (input, output), for the models this deployment
# actually routes to (`praxis.llm.catalogue.DEFAULT_MODEL_MAPPING`).
# Configurable rather than authoritative: a deployment overrides this
# via `BudgetTracker(pricing=...)` rather than editing code, precisely
# because published prices change and a stale hardcoded number
# produces confidently wrong cost reporting.
DEFAULT_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-5": (15.0, 75.0),
    # OpenAI, reachable since `praxis.llm.providers` began routing by
    # model id. Listed for the same reason the Claude tiers are: an
    # unpriced model falls back to the most expensive tier below, so
    # an OpenAI deployment without these entries would report costs
    # roughly an order of magnitude too high and trip budgets that
    # were never actually exceeded.
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
}

# What an unrecognized model costs. Zero would silently under-report a
# newly-added model to nothing; using the most expensive known tier
# instead means an unknown model is conservatively over-estimated,
# which fails safe for a guardrail.
_UNKNOWN_MODEL_PRICING = (15.0, 75.0)


@dataclass
class Spend:
    """The running ledger for one tracker."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    descendants: int = 0

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 6),
            "llm_calls": self.llm_calls,
            "descendants": self.descendants,
        }


class BudgetTracker:
    """Enforces a `Budget`, rolling spend up to a parent tracker.

    Not thread-safe, matching this codebase's single-process asyncio
    concurrency model (see `praxis.cache.memory_cache.InMemoryCache`'s
    own note on the same choice).
    """

    def __init__(
        self,
        budget: Budget | None = None,
        *,
        parent: BudgetTracker | None = None,
        pricing: dict[str, tuple[float, float]] | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        self._budget = budget if budget is not None else Budget()
        self._parent = parent
        self._pricing = pricing if pricing is not None else dict(DEFAULT_PRICING_PER_MTOK)
        self._clock = clock
        self._started_at = clock()
        self.spend = Spend()

    @property
    def budget(self) -> Budget:
        return self._budget

    @property
    def elapsed_seconds(self) -> float:
        return self._clock() - self._started_at

    def price_of(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Cost in USD for one call, per the configured pricing table."""
        rate_in, rate_out = self._pricing.get(model, _UNKNOWN_MODEL_PRICING)
        return (tokens_in / 1_000_000) * rate_in + (tokens_out / 1_000_000) * rate_out

    # ---------------------------------------------------------------- #
    # Checks. Each raises rather than returning a bool, because a
    # guardrail a caller can forget to consult is not a guardrail.
    # ---------------------------------------------------------------- #

    def check_deadline(self) -> None:
        """Raises if the wall-clock allowance is already spent."""
        limit = self._budget.max_wall_clock_seconds
        if limit is not None and self.elapsed_seconds > limit:
            raise BudgetExhaustedError(
                f"wall-clock budget of {limit}s exhausted ({self.elapsed_seconds:.1f}s elapsed)",
                resource="wall_clock_seconds",
                limit=limit,
                attempted=self.elapsed_seconds,
            )
        if self._parent is not None:
            self._parent.check_deadline()

    def check_llm_call(self, model: str, estimated_tokens: int = 0) -> None:
        """Raises if one more LLM call would breach any allowance.

        Checked *before* the call, using an estimate, so an overrun is
        prevented rather than merely recorded.
        """
        self.check_deadline()

        calls_limit = self._budget.max_llm_calls
        if calls_limit is not None and self.spend.llm_calls + 1 > calls_limit:
            raise BudgetExhaustedError(
                f"LLM call budget of {calls_limit} exhausted",
                resource="llm_calls",
                limit=calls_limit,
                attempted=self.spend.llm_calls + 1,
            )

        tokens_limit = self._budget.max_tokens
        if tokens_limit is not None and self.spend.tokens + estimated_tokens > tokens_limit:
            raise BudgetExhaustedError(
                f"token budget of {tokens_limit} would be exceeded",
                resource="tokens",
                limit=tokens_limit,
                attempted=self.spend.tokens + estimated_tokens,
            )

        cost_limit = self._budget.max_cost_usd
        if cost_limit is not None:
            projected = self.spend.cost_usd + self.price_of(model, estimated_tokens, 0)
            if projected > cost_limit:
                raise BudgetExhaustedError(
                    f"cost budget of ${cost_limit} would be exceeded (projected ${projected:.4f})",
                    resource="cost_usd",
                    limit=cost_limit,
                    attempted=projected,
                )

        if self._parent is not None:
            self._parent.check_llm_call(model, estimated_tokens)

    def check_descendant(self) -> None:
        """Raises if spawning one more child would breach the fan-out
        allowance - the prompt's "total descendant cost limits"."""
        limit = self._budget.max_descendants
        if limit is not None and self.spend.descendants + 1 > limit:
            raise BudgetExhaustedError(
                f"descendant budget of {limit} exhausted",
                resource="descendants",
                limit=limit,
                attempted=self.spend.descendants + 1,
            )
        if self._parent is not None:
            self._parent.check_descendant()

    # ---------------------------------------------------------------- #
    # Recording. Always rolls up, so a parent's ledger reflects every
    # descendant's spend - otherwise a descendant limit is unenforceable.
    # ---------------------------------------------------------------- #

    def record_llm_call(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Records one real call; returns its cost in USD."""
        cost = self.price_of(model, tokens_in, tokens_out)
        self.spend.llm_calls += 1
        self.spend.tokens_in += tokens_in
        self.spend.tokens_out += tokens_out
        self.spend.cost_usd += cost
        if self._parent is not None:
            self._parent.record_llm_call(model, tokens_in, tokens_out)
        return cost

    def record_descendant(self) -> None:
        self.spend.descendants += 1
        if self._parent is not None:
            self._parent.record_descendant()

    def child(self, budget: Budget | None = None) -> BudgetTracker:
        """A tracker for delegated work, charging this one too.

        The child's own `budget` may be tighter than the parent's but
        cannot escape it: every check and every record walks up the
        chain, so the parent's limits bind regardless of what the child
        was given.
        """
        self.check_descendant()
        self.record_descendant()
        return BudgetTracker(
            budget if budget is not None else self._budget,
            parent=self,
            pricing=self._pricing,
            clock=self._clock,
        )

    def snapshot(self) -> dict[str, Any]:
        """Spend plus allowance, for audit detail and task results."""
        return {
            "spend": self.spend.to_dict(),
            "budget": self._budget.to_dict(),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
