# praxis/semantic/validation.py
"""Metric validation: catching silently-wrong analytics (Prompt §2).

Prompt §2: *"Validate joins, metric grain, units, currencies, time
zones, missing data, and double-counting before rendering."*

Every check here targets a failure that **produces a number rather
than an error**, which is what makes them worth doing at all. A query
that crashes gets noticed; a query that returns 4,182,000 when the
true answer is 41,820 gets put on a slide.

The six classes of silent wrongness checked:

1. **Grain mismatch** - slicing a per-customer metric by a per-session
   dimension. The SQL joins fine and returns a plausible, inflated
   number.
2. **Fan-out double counting** - an additive metric aggregated across
   a one-to-many join counts each parent row once per child.
3. **Non-additive re-aggregation** - summing averages, ratios, or
   distinct counts. Always wrong, always plausible-looking.
4. **Currency mixing** - summing amounts in different currencies.
5. **Unit mixing** - the same, for non-monetary units.
6. **Staleness** - a correct number computed from data that breached
   its freshness SLA, presented as current.

Severity matters: `ERROR` issues make the request invalid and must
block rendering, while `WARNING` issues are surfaced alongside the
result. Staleness is a warning (the number is right, just old);
double-counting is an error (the number is wrong).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from praxis.semantic.model import Metric, SemanticLayer


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found with a requested metric/dimension combination."""

    code: str
    severity: Severity
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def blocks(self) -> bool:
        return self.severity is Severity.ERROR

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "detail": self.detail,
        }


class MetricValidationError(Exception):
    """Raised when a request has blocking issues.

    Carries every issue, not just the first: an analyst fixing a query
    should see all of what is wrong in one pass rather than
    rediscovering problems one at a time.
    """

    def __init__(self, message: str, *, issues: list[ValidationIssue]) -> None:
        super().__init__(message)
        self.issues = issues


def _grains_compatible(layer: SemanticLayer, metric_entity: str, dim_entity: str) -> bool:
    """Whether a metric at `metric_entity` may be sliced by a dimension
    at `dim_entity`.

    Compatible when the entities match outright, or when the metric's
    entity declares a roll-up path to the dimension's (an order-grain
    metric can legitimately be sliced by customer, because orders roll
    up to customers - but not the reverse).
    """
    if metric_entity == dim_entity:
        return True
    entity = layer.entity(metric_entity)
    return entity is not None and dim_entity in entity.rolls_up_to


def _check_grain(
    layer: SemanticLayer, metric: Metric, dimension_names: list[str]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name in dimension_names:
        try:
            dimension = layer.dimension(name)
        except KeyError as exc:
            issues.append(
                ValidationIssue(
                    code="unknown_dimension",
                    severity=Severity.ERROR,
                    message=str(exc),
                    detail={"dimension": name},
                )
            )
            continue

        if not _grains_compatible(layer, metric.grain.entity, dimension.grain.entity):
            issues.append(
                ValidationIssue(
                    code="grain_mismatch",
                    severity=Severity.ERROR,
                    message=(
                        f"metric '{metric.name}' is defined at {metric.grain} grain but "
                        f"dimension '{name}' is at {dimension.grain} grain; slicing across "
                        "these would produce a misleading number rather than an error"
                    ),
                    detail={
                        "metric_grain": str(metric.grain),
                        "dimension_grain": str(dimension.grain),
                    },
                )
            )

        if metric.allowed_dimensions and name not in metric.allowed_dimensions:
            issues.append(
                ValidationIssue(
                    code="dimension_not_allowed",
                    severity=Severity.ERROR,
                    message=(
                        f"metric '{metric.name}' declares it may only be sliced by "
                        f"{list(metric.allowed_dimensions)}, not '{name}'"
                    ),
                    detail={"dimension": name},
                )
            )
    return issues


def _check_fan_out(
    layer: SemanticLayer, metric: Metric, dimension_names: list[str]
) -> list[ValidationIssue]:
    """An additive metric aggregated across a fan-out join double-counts."""
    if not metric.is_additive or not metric.source:
        return []

    issues: list[ValidationIssue] = []
    for name in dimension_names:
        try:
            dimension = layer.dimension(name)
        except KeyError:
            continue  # already reported by the grain check
        if not dimension.source or dimension.source == metric.source:
            continue

        join = layer.join_between(metric.source, dimension.source)
        if join is None:
            issues.append(
                ValidationIssue(
                    code="undeclared_join",
                    severity=Severity.ERROR,
                    message=(
                        f"no declared join between '{metric.source}' and "
                        f"'{dimension.source}'; refusing to guess a join condition"
                    ),
                    detail={"left": metric.source, "right": dimension.source},
                )
            )
            continue

        if join.fans_out:
            issues.append(
                ValidationIssue(
                    code="double_counting",
                    severity=Severity.ERROR,
                    message=(
                        f"joining '{metric.source}' to '{dimension.source}' is "
                        f"{join.join_type.value}, so aggregating additive metric "
                        f"'{metric.name}' across it would count rows more than once"
                    ),
                    detail={"join": join.to_dict()},
                )
            )
    return issues


def _check_multi_metric_compatibility(metrics: list[Metric]) -> list[ValidationIssue]:
    """Currency/unit mixing, and non-additive re-aggregation."""
    issues: list[ValidationIssue] = []

    currencies = {metric.currency for metric in metrics if metric.currency}
    if len(currencies) > 1:
        issues.append(
            ValidationIssue(
                code="currency_mismatch",
                severity=Severity.ERROR,
                message=(
                    f"metrics span multiple currencies {sorted(currencies)}; they cannot be "
                    "combined without an explicit conversion"
                ),
                detail={"currencies": sorted(currencies)},
            )
        )

    units = {metric.unit for metric in metrics if metric.unit}
    if len(units) > 1:
        issues.append(
            ValidationIssue(
                code="unit_mismatch",
                severity=Severity.WARNING,
                message=(
                    f"metrics use different units {sorted(units)}; they are shown together "
                    "but must not be summed or compared directly"
                ),
                detail={"units": sorted(units)},
            )
        )

    return issues


def _check_non_additive(metric: Metric, dimension_names: list[str]) -> list[ValidationIssue]:
    """Slicing a non-additive metric is fine; *rolling it back up* is not.

    Reported as a warning rather than an error because the request
    itself is legitimate - what would be wrong is a consumer later
    summing the sliced results, so the caller is told rather than
    blocked.
    """
    if metric.is_additive or not dimension_names:
        return []
    return [
        ValidationIssue(
            code="non_additive_metric",
            severity=Severity.WARNING,
            message=(
                f"metric '{metric.name}' is {metric.metric_type.value}, which is not "
                "additive: its per-slice values must not be summed to obtain the total"
            ),
            detail={"metric": metric.name, "type": metric.metric_type.value},
        )
    ]


def _check_freshness(layer: SemanticLayer, metric: Metric) -> list[ValidationIssue]:
    stale = layer.is_stale(metric.name)
    if stale is None:
        if metric.freshness_sla_seconds is not None:
            return [
                ValidationIssue(
                    code="freshness_unknown",
                    severity=Severity.WARNING,
                    message=(
                        f"metric '{metric.name}' declares a freshness SLA but no refresh has "
                        "ever been recorded for its source, so its currency is unknown"
                    ),
                    detail={"metric": metric.name, "source": metric.source},
                )
            ]
        return []
    if stale:
        age = layer.freshness_seconds(metric.source)
        return [
            ValidationIssue(
                code="stale_data",
                severity=Severity.WARNING,
                message=(
                    f"metric '{metric.name}' is computed from data {age:.0f}s old, past its "
                    f"{metric.freshness_sla_seconds}s freshness SLA"
                ),
                detail={"metric": metric.name, "age_seconds": age},
            )
        ]
    return []


def validate_query_request(
    layer: SemanticLayer,
    *,
    metric_names: list[str],
    dimension_names: list[str] | None = None,
    raise_on_error: bool = True,
) -> list[ValidationIssue]:
    """Validates a metric/dimension request against the semantic layer.

    Returns every issue found. With `raise_on_error=True` (the
    default), any `ERROR`-severity issue raises
    `MetricValidationError` carrying the full list - the defaulting
    choice matters, because a validator whose findings are easy to
    ignore is one that will be ignored.
    """
    dimensions = list(dimension_names or [])
    issues: list[ValidationIssue] = []
    resolved: list[Metric] = []

    for name in metric_names:
        try:
            resolved.append(layer.metric(name))
        except KeyError as exc:
            issues.append(
                ValidationIssue(
                    code="unknown_metric",
                    severity=Severity.ERROR,
                    message=str(exc),
                    detail={"metric": name},
                )
            )

    for metric in resolved:
        issues.extend(_check_grain(layer, metric, dimensions))
        issues.extend(_check_fan_out(layer, metric, dimensions))
        issues.extend(_check_non_additive(metric, dimensions))
        issues.extend(_check_freshness(layer, metric))

    if len(resolved) > 1:
        issues.extend(_check_multi_metric_compatibility(resolved))

    blocking = [issue for issue in issues if issue.blocks]
    if blocking and raise_on_error:
        raise MetricValidationError(
            f"{len(blocking)} blocking validation issue(s): "
            + "; ".join(issue.message for issue in blocking),
            issues=issues,
        )
    return issues
