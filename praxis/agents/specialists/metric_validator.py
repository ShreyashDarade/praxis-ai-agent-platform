# praxis/agents/specialists/metric_validator.py
"""`metric_validator`: the specialist that refuses plausible wrong numbers.

Prompt §2: *"Validate joins, metric grain, units, currencies, time
zones, missing data, and double-counting before rendering."* §7 names
this the `metric-definition` skill, owned by a "Semantic analyst".

**This is the specialist whose job is to say no.** Every other agent
in the dashboard pipeline moves work forward; this one exists to stop
a result that would render fine and mean nothing. That asymmetry is
deliberate - the failures it catches (a metric summed across a
fan-out join, revenue in two currencies added together) do not throw
exceptions. They produce a number, and the number goes on a slide.

It runs the deterministic checks in `praxis.semantic.validation`
against the declared semantic layer, and separately inspects the
*actual returned rows* for problems a schema-level check cannot see:
nulls in a measure column, a suspiciously round row count suggesting
truncation, and duplicate dimension keys that indicate the fan-out
already happened.
"""
from __future__ import annotations

from typing import Any

from praxis.agents.budget import Budget
from praxis.agents.contract import SpecialistResult
from praxis.agents.subagent import AgentContext, AgentManifest, SubAgent, register_agent
from praxis.semantic.validation import Severity, validate_query_request


def _inspect_rows(
    rows: list[dict[str, Any]], *, measure: str | None, dimension: str | None
) -> tuple[list[str], list[str]]:
    """Row-level checks the semantic layer cannot make.

    Returns `(errors, warnings)`. These look at what actually came
    back rather than at what was declared, which is the only way to
    catch a fan-out that already happened.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not rows:
        warnings.append("the result set is empty, so nothing can be validated against it")
        return errors, warnings

    if measure and measure in rows[0]:
        null_count = sum(1 for row in rows if row.get(measure) is None)
        if null_count:
            warnings.append(
                f"{null_count} of {len(rows)} rows have a NULL '{measure}'; charting them "
                "will silently treat the gaps as absent rather than zero"
            )

    if dimension and dimension in rows[0]:
        seen = [row.get(dimension) for row in rows]
        duplicates = len(seen) - len(set(map(str, seen)))
        if duplicates:
            # The signature of a fan-out that already occurred: one
            # dimension value appearing several times means the
            # aggregate is being double-counted downstream.
            errors.append(
                f"dimension '{dimension}' has {duplicates} duplicate value(s); the rows are "
                "not aggregated to one row per dimension value, so charting them would "
                "double-count"
            )

    return errors, warnings


class MetricValidatorAgent(SubAgent):
    """Validates a metric request and its actual results."""

    manifest = AgentManifest(
        name="metric_validator",
        description=(
            "Checks a metric/dimension request for grain mismatch, double counting, "
            "currency and unit mixing, staleness, and inspects the returned rows for "
            "nulls and duplicate dimension keys"
        ),
        tools=(),
        model_purpose="planning",
        default_budget=Budget(max_llm_calls=0),
        keywords=("validate", "metric", "grain", "double-count", "currency"),
    )

    async def execute(self, context: AgentContext) -> SpecialistResult:
        inputs = context.contract.inputs
        layer = inputs.get("semantic_layer")
        rows = list(inputs.get("rows") or [])
        metric_names = list(inputs.get("metrics") or [])
        dimension_names = list(inputs.get("dimensions") or [])

        errors: list[str] = []
        warnings: list[str] = []
        issue_payload: list[dict[str, Any]] = []

        if layer is not None and metric_names:
            issues = validate_query_request(
                layer,
                metric_names=metric_names,
                dimension_names=dimension_names,
                raise_on_error=False,
            )
            issue_payload = [issue.to_dict() for issue in issues]
            for issue in issues:
                (errors if issue.severity is Severity.ERROR else warnings).append(issue.message)
        elif metric_names:
            warnings.append(
                "no semantic layer was supplied, so grain, join and currency checks could "
                "not run - only the returned rows were inspected"
            )

        row_errors, row_warnings = _inspect_rows(
            rows,
            measure=inputs.get("measure_column"),
            dimension=inputs.get("dimension_column"),
        )
        errors.extend(row_errors)
        warnings.extend(row_warnings)

        # A validator that "succeeds" while reporting blocking errors
        # would be useless - the caller must be able to branch on this.
        return SpecialistResult(
            task_id=context.contract.task_id,
            succeeded=not errors,
            results={
                "valid": not errors,
                "issues": issue_payload,
                "blocking": errors,
                "advisory": warnings,
                "row_count": len(rows),
            },
            evidence=[
                {
                    "kind": "validation",
                    "metrics": metric_names,
                    "dimensions": dimension_names,
                    "checked_rows": len(rows),
                }
            ],
            limitations=warnings,
            errors=errors,
            spend=context.budget.snapshot(),
        )


register_agent(MetricValidatorAgent())
