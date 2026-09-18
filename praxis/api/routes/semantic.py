# praxis/api/routes/semantic.py
"""Validating an aggregation against declared metric definitions.

`praxis.semantic` models metrics, dimensions, grains and joins, and
knows the three ways a correct-looking aggregate is wrong: averaging
an average, summing a distinct count, and summing across a fan-out
join. None of it was reachable, so the checks ran nowhere and the
knowledge changed no answer.

The endpoint is stateless by design, and that is worth being explicit
about rather than apologising for. A tenant posts the layer it wants
checked against together with the aggregation in question. Praxis has
no store of per-tenant metric definitions, so the alternative would be
a server-side registry this deployment cannot yet populate - and an
endpoint that read from an empty registry would answer "no problems
found" to everything, which is worse than not having it. Passing the
definitions in makes the basis of the verdict explicit and auditable.

The same layer can be handed to the `metric_validator` specialist
through `delegate_to_specialist`, which is how a *plan* gets a metric
checked mid-task; this endpoint is for checking one directly.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from praxis.api.dependencies import require
from praxis.security.policy import Permission
from praxis.security.principal import Principal
from praxis.semantic.loader import SemanticLayerError, layer_from_dict, layer_to_dict
from praxis.semantic.validation import validate_query_request

router = APIRouter(prefix="/semantic", tags=["semantic"])


class SemanticLayerRequest(BaseModel):
    """A metric-definition document, in the shape `layer_to_dict` emits."""

    layer: dict[str, Any] = Field(
        description="Object with 'metrics', 'dimensions', 'joins', 'entities', 'glossary'"
    )


class ValidateRequest(SemanticLayerRequest):
    metrics: list[str] = Field(default_factory=list, description="metric names being computed")
    dimensions: list[str] = Field(
        default_factory=list, description="dimension names being grouped by"
    )
    # Joins are NOT passed separately: fan-out is checked from the
    # joins declared in the layer itself, so a caller cannot get a
    # clean verdict by omitting the join that causes the problem.


def _build(payload: dict[str, Any], tenant_id: str):
    try:
        return layer_from_dict(payload, tenant_id=tenant_id)
    except SemanticLayerError as exc:
        # The definition itself is malformed - the caller's document,
        # not their aggregation, so a 400 naming the metric.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/describe")
async def describe_layer(
    body: SemanticLayerRequest,
    principal: Annotated[Principal, Depends(require(Permission.TASK_READ))],
) -> dict[str, Any]:
    """Parses a layer and echoes back what Praxis understood.

    Round-trips: what comes back can be posted again unchanged. Useful
    for confirming a grain or an additivity flag was read the way the
    author meant, before anyone relies on a verdict computed from it.
    """
    layer = _build(body.layer, principal.tenant_id)
    described = layer_to_dict(layer)
    described["non_additive_metrics"] = [
        m.name for m in layer.metrics() if not m.is_additive
    ]
    described["fan_out_joins"] = [
        f"{j.left}->{j.right}" for j in layer.joins() if j.fans_out
    ]
    return described


@router.post("/validate")
async def validate(
    body: ValidateRequest,
    principal: Annotated[Principal, Depends(require(Permission.TASK_READ))],
) -> dict[str, Any]:
    """Checks one aggregation against the declared definitions.

    Issues come back split into blocking and advisory. A blocking
    issue means the number would be wrong, not merely questionable -
    summing a `count_distinct` across partitions, or aggregating
    additively over a join that multiplies rows.

    An unknown metric is itself a blocking issue rather than a 400:
    the caller asked whether this aggregation is sound, and "one of
    these metrics is not defined" is an answer to that question.
    """
    layer = _build(body.layer, principal.tenant_id)
    # `raise_on_error=False`: this endpoint's job is to report every
    # issue, and the library default raises on the first blocking set.
    # A caller asking "is this aggregation sound?" wants the list, not
    # an exception.
    issues = validate_query_request(
        layer,
        metric_names=body.metrics,
        dimension_names=body.dimensions,
        raise_on_error=False,
    )

    blocking = [i.to_dict() for i in issues if i.blocks]
    advisory = [i.to_dict() for i in issues if not i.blocks]
    return {
        "valid": not blocking,
        "blocking": blocking,
        "advisory": advisory,
        "checked": {
            "metrics": body.metrics,
            "dimensions": body.dimensions,
            "joins_declared_in_layer": [
                f"{j.left}->{j.right}" for j in layer.joins()
            ],
        },
    }
