# praxis/semantic/model.py
"""Semantic-layer entities: metrics, dimensions, joins, glossary.

Every field here exists to encode a business fact a database schema
cannot express, and each one corresponds to a specific way an
LLM-authored analytics query goes silently wrong:

- `Metric.grain` - the level a metric is defined at. Summing a
  per-customer metric across a per-session dimension double-counts.
  Recorded so `validation.py` can refuse the combination instead of
  returning an inflated number.
- `Metric.unit` / `Metric.currency` - "revenue" in minor units summed
  with "revenue" in major units produces a number 100x wrong that
  looks entirely plausible.
- `Metric.is_additive` - averages, ratios, and distinct counts cannot
  be re-aggregated by summing partial results. This is the single most
  common silent error in generated SQL.
- `Metric.filters` - the business definition of "completed order" is
  a filter, not a table. Leaving it to the model to remember means it
  is sometimes applied and sometimes not, producing two different
  "revenue" numbers on the same dashboard.
- `freshness_sla_seconds` / `trust_score` - whether an answer should
  be *believed*, surfaced alongside it rather than discovered later.

Everything is a frozen dataclass: a semantic definition is a
declaration, and a caller mutating one at runtime would change what a
metric means underneath other callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MetricType(str, Enum):
    """How a metric aggregates - which determines whether partial
    results can be combined."""

    SUM = "sum"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    RATIO = "ratio"

    @property
    def is_additive(self) -> bool:
        """Whether partial results may be summed to form the whole.

        The three false cases are exactly the ones that produce
        confidently wrong dashboards: you cannot average averages,
        you cannot sum distinct counts (the same entity appears in
        several partitions), and you cannot sum ratios.
        """
        return self in (MetricType.SUM, MetricType.COUNT, MetricType.MIN, MetricType.MAX)


class JoinType(str, Enum):
    ONE_TO_ONE = "one_to_one"
    MANY_TO_ONE = "many_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"

    @property
    def fans_out(self) -> bool:
        """Whether this join can multiply rows.

        A fan-out join before an additive aggregation is the classic
        double-counting bug: joining orders to line-items then summing
        `order.total` counts each order once per line item.
        """
        return self in (JoinType.ONE_TO_MANY, JoinType.MANY_TO_MANY)


@dataclass(frozen=True)
class Grain:
    """The level of detail a metric or dimension is defined at.

    `entity` is what one row represents ("customer", "order",
    "session"); `time_unit` is the temporal granularity when there is
    one ("day", "week", "month"). Two grains are compatible when their
    entities match, or when one is a declared roll-up of the other.
    """

    entity: str
    time_unit: str | None = None

    def __str__(self) -> str:
        return f"{self.entity}" + (f"/{self.time_unit}" if self.time_unit else "")


@dataclass(frozen=True)
class Metric:
    """A business measure with everything needed to compute it safely."""

    name: str
    expression: str
    metric_type: MetricType
    grain: Grain
    description: str = ""
    # The table/entity the expression is rooted in.
    source: str = ""
    # Business-definition filters that are part of what the metric
    # MEANS, not optional refinements a caller may drop.
    filters: tuple[str, ...] = ()
    unit: str | None = None
    currency: str | None = None
    # Dimensions this metric may legitimately be sliced by. Empty means
    # "unconstrained" - the honest default when nobody has declared it,
    # rather than silently forbidding everything.
    allowed_dimensions: tuple[str, ...] = ()
    freshness_sla_seconds: int | None = None
    trust_score: float = 1.0
    owner: str = ""

    @property
    def is_additive(self) -> bool:
        return self.metric_type.is_additive

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "type": self.metric_type.value,
            "grain": str(self.grain),
            "description": self.description,
            "source": self.source,
            "filters": list(self.filters),
            "unit": self.unit,
            "currency": self.currency,
            "additive": self.is_additive,
            "allowed_dimensions": list(self.allowed_dimensions),
            "freshness_sla_seconds": self.freshness_sla_seconds,
            "trust_score": self.trust_score,
            "owner": self.owner,
        }


@dataclass(frozen=True)
class Dimension:
    """An attribute a metric can be sliced by."""

    name: str
    expression: str
    grain: Grain
    description: str = ""
    source: str = ""
    data_type: str = "string"
    # For a time dimension, the unit it buckets into.
    time_unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "grain": str(self.grain),
            "description": self.description,
            "source": self.source,
            "data_type": self.data_type,
            "time_unit": self.time_unit,
        }


@dataclass(frozen=True)
class Join:
    """A declared relationship between two sources."""

    left: str
    right: str
    join_type: JoinType
    on: str
    description: str = ""

    @property
    def fans_out(self) -> bool:
        """Whether this join can multiply rows - delegated to the type,
        so callers ask the join (which is what they hold) rather than
        having to reach through to its type."""
        return self.join_type.fans_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "type": self.join_type.value,
            "on": self.on,
            "fans_out": self.join_type.fans_out,
            "description": self.description,
        }


@dataclass(frozen=True)
class Entity:
    """A business object: what one row of a source represents."""

    name: str
    source: str
    primary_key: str
    description: str = ""
    # Coarser entities this one rolls up into ("order" -> "customer"),
    # which is what makes two different grains compatible.
    rolls_up_to: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "primary_key": self.primary_key,
            "description": self.description,
            "rolls_up_to": list(self.rolls_up_to),
        }


@dataclass(frozen=True)
class GlossaryTerm:
    """A business term and what it actually means here.

    Distinct from a metric: "churn" may be a glossary term that three
    different metrics each partially capture, and writing down that
    ambiguity is more useful than pretending one metric owns the word.
    """

    term: str
    definition: str
    synonyms: tuple[str, ...] = ()
    related_metrics: tuple[str, ...] = ()
    owner: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "definition": self.definition,
            "synonyms": list(self.synonyms),
            "related_metrics": list(self.related_metrics),
            "owner": self.owner,
        }


class SemanticLayer:
    """A registry of metrics, dimensions, joins, entities, and glossary.

    In-memory and per-tenant. Deliberately NOT a global singleton: two
    tenants will define "revenue" differently, and a shared registry
    would let one tenant's definition silently answer the other's
    question - the same class of cross-tenant leak the storage layer
    already guards against.
    """

    def __init__(self, tenant_id: str = "") -> None:
        self.tenant_id = tenant_id
        self._metrics: dict[str, Metric] = {}
        self._dimensions: dict[str, Dimension] = {}
        self._joins: list[Join] = []
        self._entities: dict[str, Entity] = {}
        self._glossary: dict[str, GlossaryTerm] = {}
        self._last_refreshed: dict[str, datetime] = {}

    # -- registration ------------------------------------------------ #

    def add_metric(self, metric: Metric) -> None:
        if metric.name in self._metrics:
            raise ValueError(f"metric '{metric.name}' is already defined")
        self._metrics[metric.name] = metric

    def add_dimension(self, dimension: Dimension) -> None:
        if dimension.name in self._dimensions:
            raise ValueError(f"dimension '{dimension.name}' is already defined")
        self._dimensions[dimension.name] = dimension

    def add_join(self, join: Join) -> None:
        self._joins.append(join)

    def add_entity(self, entity: Entity) -> None:
        if entity.name in self._entities:
            raise ValueError(f"entity '{entity.name}' is already defined")
        self._entities[entity.name] = entity

    def add_term(self, term: GlossaryTerm) -> None:
        self._glossary[term.term.lower()] = term

    # -- lookup ------------------------------------------------------- #

    def metric(self, name: str) -> Metric:
        try:
            return self._metrics[name]
        except KeyError:
            raise KeyError(
                f"no metric named '{name}'; defined: {sorted(self._metrics)}"
            ) from None

    def dimension(self, name: str) -> Dimension:
        try:
            return self._dimensions[name]
        except KeyError:
            raise KeyError(
                f"no dimension named '{name}'; defined: {sorted(self._dimensions)}"
            ) from None

    def entity(self, name: str) -> Entity | None:
        return self._entities.get(name)

    def metrics(self) -> list[Metric]:
        return list(self._metrics.values())

    def dimensions(self) -> list[Dimension]:
        return list(self._dimensions.values())

    def joins(self) -> list[Join]:
        return list(self._joins)

    def glossary(self) -> list[GlossaryTerm]:
        return list(self._glossary.values())

    def lookup_term(self, text: str) -> GlossaryTerm | None:
        """Resolves a business word to its definition, including via
        synonyms - which is how a user's "sales" reaches the "revenue"
        definition without the Planner having to guess."""
        needle = text.strip().lower()
        direct = self._glossary.get(needle)
        if direct is not None:
            return direct
        for term in self._glossary.values():
            if needle in {synonym.lower() for synonym in term.synonyms}:
                return term
        return None

    def join_between(self, left: str, right: str) -> Join | None:
        """The declared join between two sources, in either direction."""
        for join in self._joins:
            if {join.left, join.right} == {left, right}:
                return join
        return None

    # -- freshness ---------------------------------------------------- #

    def record_refresh(self, source: str, when: datetime | None = None) -> None:
        self._last_refreshed[source] = when or datetime.now(timezone.utc)

    def freshness_seconds(self, source: str) -> float | None:
        """Seconds since `source` was last refreshed, or `None` if
        never recorded.

        `None` means genuinely unknown, and callers must treat it as
        such rather than as "fresh" - an unmeasured source is not a
        current one.
        """
        last = self._last_refreshed.get(source)
        if last is None:
            return None
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last).total_seconds()

    def is_stale(self, metric_name: str) -> bool | None:
        """Whether a metric's source has breached its freshness SLA.

        Returns `None` when the answer is unknown (no SLA declared, or
        no refresh ever recorded) - deliberately three-valued, because
        reporting "fresh" for something never measured would be a
        false assurance.
        """
        metric = self.metric(metric_name)
        if metric.freshness_sla_seconds is None:
            return None
        age = self.freshness_seconds(metric.source)
        if age is None:
            return None
        return age > metric.freshness_sla_seconds

    def describe(self) -> dict[str, Any]:
        """The whole layer, JSON-safe - what gets rendered into a
        planning prompt so the model works from real definitions
        instead of guessing at column semantics."""
        return {
            "tenant_id": self.tenant_id,
            "metrics": [metric.to_dict() for metric in self._metrics.values()],
            "dimensions": [dim.to_dict() for dim in self._dimensions.values()],
            "joins": [join.to_dict() for join in self._joins],
            "entities": [entity.to_dict() for entity in self._entities.values()],
            "glossary": [term.to_dict() for term in self._glossary.values()],
        }
