# praxis/semantic/loader.py
"""Builds a `SemanticLayer` from plain JSON-shaped data.

The model in `praxis.semantic.model` is deliberately made of frozen
dataclasses with typed enums and a structured `Grain`. That is right
for the code that reasons over it, and wrong for every boundary where
the layer actually arrives: an HTTP request body, a plan step's
arguments, a tenant's checked-in definition file. Those all speak
dicts and strings.

This module is that one boundary, in one place. Two properties matter:

- **It rejects rather than coerces.** An unknown metric type or a
  missing expression raises `SemanticLayerError` naming the offending
  metric. Defaulting an unrecognised type to `SUM` would produce a
  layer that computes confidently wrong numbers - a non-additive
  measure summed across partitions is exactly the double-counting bug
  `MetricType.is_additive` exists to prevent.
- **It is tenant-scoped like the model it builds.** `tenant_id` is a
  required argument rather than an optional one, because a layer built
  without it is a layer that can answer another tenant's question.
"""
from __future__ import annotations

from typing import Any

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


class SemanticLayerError(ValueError):
    """A semantic layer definition that cannot be built as written."""


def _grain(payload: Any, *, what: str) -> Grain:
    """Reads a grain from either `"customer/day"` or `{...}` form.

    The string form exists because `Grain.__str__` emits it, so a layer
    that has been serialized and sent back round-trips without the
    caller having to re-expand it by hand.
    """
    if isinstance(payload, str):
        entity, _, time_unit = payload.partition("/")
        if not entity.strip():
            raise SemanticLayerError(f"{what}: grain '{payload}' has no entity")
        return Grain(entity=entity.strip(), time_unit=time_unit.strip() or None)
    if isinstance(payload, dict):
        entity = str(payload.get("entity") or "").strip()
        if not entity:
            raise SemanticLayerError(f"{what}: grain is missing 'entity'")
        time_unit = payload.get("time_unit")
        return Grain(entity=entity, time_unit=str(time_unit) if time_unit else None)
    raise SemanticLayerError(
        f"{what}: grain must be a string like 'customer/day' or an object, got "
        f"{type(payload).__name__}"
    )


def _metric_type(raw: Any, *, what: str) -> MetricType:
    try:
        return MetricType(str(raw).lower())
    except ValueError:
        raise SemanticLayerError(
            f"{what}: unknown metric type '{raw}'; valid types are "
            f"{[t.value for t in MetricType]}"
        ) from None


def metric_from_dict(payload: dict[str, Any]) -> Metric:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise SemanticLayerError("a metric is missing 'name'")
    what = f"metric '{name}'"

    expression = str(payload.get("expression") or "").strip()
    if not expression:
        raise SemanticLayerError(f"{what}: 'expression' is required")
    if "grain" not in payload:
        raise SemanticLayerError(
            f"{what}: 'grain' is required - a metric with no declared grain cannot be "
            "checked for double-counting"
        )

    return Metric(
        name=name,
        expression=expression,
        metric_type=_metric_type(payload.get("type") or payload.get("metric_type"), what=what),
        grain=_grain(payload["grain"], what=what),
        description=str(payload.get("description") or ""),
        source=str(payload.get("source") or ""),
        filters=tuple(str(f) for f in payload.get("filters") or ()),
        unit=payload.get("unit"),
        currency=payload.get("currency"),
        allowed_dimensions=tuple(str(d) for d in payload.get("allowed_dimensions") or ()),
        freshness_sla_seconds=payload.get("freshness_sla_seconds"),
        trust_score=float(payload.get("trust_score", 1.0)),
        owner=str(payload.get("owner") or ""),
    )


def dimension_from_dict(payload: dict[str, Any]) -> Dimension:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise SemanticLayerError("a dimension is missing 'name'")
    what = f"dimension '{name}'"

    expression = str(payload.get("expression") or "").strip()
    if not expression:
        raise SemanticLayerError(f"{what}: 'expression' is required")
    if "grain" not in payload:
        raise SemanticLayerError(f"{what}: 'grain' is required")

    return Dimension(
        name=name,
        expression=expression,
        grain=_grain(payload["grain"], what=what),
        description=str(payload.get("description") or ""),
        source=str(payload.get("source") or ""),
        data_type=str(payload.get("data_type") or "string"),
        time_unit=payload.get("time_unit"),
    )


def join_from_dict(payload: dict[str, Any]) -> Join:
    left = str(payload.get("left") or "").strip()
    right = str(payload.get("right") or "").strip()
    if not left or not right:
        raise SemanticLayerError("a join needs both 'left' and 'right'")
    what = f"join {left}->{right}"
    try:
        join_type = JoinType(str(payload.get("type") or payload.get("join_type")).lower())
    except ValueError:
        raise SemanticLayerError(
            f"{what}: unknown join type '{payload.get('type')}'; valid types are "
            f"{[t.value for t in JoinType]}"
        ) from None
    return Join(
        left=left,
        right=right,
        join_type=join_type,
        on=str(payload.get("on") or ""),
    )


def layer_from_dict(payload: dict[str, Any], *, tenant_id: str) -> SemanticLayer:
    """Builds a tenant-scoped `SemanticLayer` from a definition document.

    Accepts the shape `to_dict()` emits, so a layer read back out of an
    API and posted again produces the same layer.
    """
    if not isinstance(payload, dict):
        raise SemanticLayerError(
            f"a semantic layer definition must be an object, got {type(payload).__name__}"
        )

    layer = SemanticLayer(tenant_id=tenant_id)

    for raw in payload.get("entities") or ():
        name = str(raw.get("name") or "").strip()
        if not name:
            raise SemanticLayerError("an entity is missing 'name'")
        layer.add_entity(
            Entity(
                name=name,
                primary_key=str(raw.get("primary_key") or ""),
                source=str(raw.get("source") or ""),
                description=str(raw.get("description") or ""),
                rolls_up_to=tuple(str(r) for r in raw.get("rolls_up_to") or ()),
            )
        )

    for raw in payload.get("dimensions") or ():
        layer.add_dimension(dimension_from_dict(raw))

    for raw in payload.get("metrics") or ():
        layer.add_metric(metric_from_dict(raw))

    for raw in payload.get("joins") or ():
        layer.add_join(join_from_dict(raw))

    for raw in payload.get("glossary") or ():
        term = str(raw.get("term") or "").strip()
        if not term:
            raise SemanticLayerError("a glossary entry is missing 'term'")
        layer.add_term(
            GlossaryTerm(
                term=term,
                definition=str(raw.get("definition") or ""),
                synonyms=tuple(str(s) for s in raw.get("synonyms") or ()),
                related_metrics=tuple(str(m) for m in raw.get("related_metrics") or ()),
                owner=str(raw.get("owner") or ""),
            )
        )

    return layer


def layer_to_dict(layer: SemanticLayer) -> dict[str, Any]:
    """The inverse of `layer_from_dict`, for serving a layer back."""
    return {
        "tenant_id": layer.tenant_id,
        "metrics": [m.to_dict() for m in layer.metrics()],
        "dimensions": [d.to_dict() for d in layer.dimensions()],
        "joins": [j.to_dict() for j in layer.joins()],
        "glossary": [t.to_dict() for t in layer.glossary()],
    }
