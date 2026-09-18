# praxis/semantic/__init__.py
"""The semantic / business layer (Prompt §2, §6).

Prompt §2 requires the platform to *"create a semantic/business
layer"* and §6 specifies its contents: *"metrics, dimensions, joins,
entity definitions, business glossary, data lineage, freshness, and
trust scores"*.

**Why this exists at all**, rather than letting the Planner write raw
SQL against an introspected schema every time: a raw schema says
`orders.amount` is a numeric column. It does not say that "revenue"
means `SUM(amount) WHERE status='completed'`, that it is denominated
in minor units, that it must never be summed across currencies
without conversion, or that joining it to `sessions` double-counts
because the grain differs. Those are business facts, not schema facts,
and they are exactly where LLM-generated analytics goes quietly wrong:
the SQL runs, returns a number, and the number is meaningless.

So the layer's job is not convenience - it is to make a class of
silent wrong answers into loud errors. `validation.py` refuses a
metric/dimension combination whose grain does not line up, or that
mixes currencies or units, rather than returning a plausible number.
"""
from praxis.semantic.model import (
    Dimension,
    Entity,
    GlossaryTerm,
    Grain,
    Join,
    JoinType,
    Metric,
    MetricType,
    SemanticLayer,
)
from praxis.semantic.validation import (
    MetricValidationError,
    ValidationIssue,
    validate_query_request,
)

__all__ = [
    "Dimension",
    "Entity",
    "GlossaryTerm",
    "Grain",
    "Join",
    "JoinType",
    "Metric",
    "MetricType",
    "SemanticLayer",
    "MetricValidationError",
    "ValidationIssue",
    "validate_query_request",
]
