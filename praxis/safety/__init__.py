# praxis/safety/__init__.py
"""Execution-time safety controls (Prompt §8).

Distinct from `praxis.security`, and the split is deliberate:

- `praxis.security` answers **"is this actor allowed to ask for this?"**
  - identity, permissions, tenancy, approval, audit.
- `praxis.safety` answers **"is this specific action safe to perform,
  and was its result sound?"** - SQL validation, query cost bounds,
  output validation, untrusted-content handling, and reversibility.

An action can pass every authorization check and still be unsafe (an
authorized analyst asking for an unbounded cross join), and a perfectly
safe action can be unauthorized. Keeping them in separate modules keeps
each one auditable on its own terms.

Every control here is deterministic, non-LLM code. Per Prompt §8, SQL
parsing in particular is "an additional check, not the security
boundary" - the database's own permissions remain the real boundary,
and `sql_guard` says so in its own docstring rather than overclaiming.
"""
from praxis.safety.sql_guard import (
    QueryCostLimits,
    SqlGuard,
    SqlGuardError,
    SqlStatementKind,
    UnsafeQueryError,
)
from praxis.safety.output_validation import (
    OutputValidationError,
    validate_skill_output,
)
from praxis.safety.untrusted import (
    UNTRUSTED_CLOSE,
    UNTRUSTED_OPEN,
    detect_injection_markers,
    wrap_untrusted,
)

__all__ = [
    "QueryCostLimits",
    "SqlGuard",
    "SqlGuardError",
    "SqlStatementKind",
    "UnsafeQueryError",
    "OutputValidationError",
    "validate_skill_output",
    "UNTRUSTED_OPEN",
    "UNTRUSTED_CLOSE",
    "detect_injection_markers",
    "wrap_untrusted",
]
