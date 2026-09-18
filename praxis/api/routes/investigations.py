# praxis/api/routes/investigations.py
"""The agentic half of the brief, reachable at last.

The product brief asks for two execution models: *"Deterministic
workflows for high-risk operations; agentic workflows for ambiguous
research/analysis."* Praxis had both built - `Orchestrator` runs a
plan, `AgentLoop` runs an observe/decide/act cycle - but only the
first had a route. `AgentLoop` also had no `Policy` implementation
outside its tests, so half the brief's execution model could not be
reached at all. `praxis.agents.policy.ModelPolicy` supplies the
decision function and this endpoint supplies the door.

**Why a separate endpoint rather than another `ExecutionMode`.** The
modes answer "how far may this task go" - plan only, read only, dry
run, execute - and every one of them runs the same plan-then-execute
engine. A loop is a different engine, not another permission level,
and folding it into that enum would put an engine switch inside the
type the mutation and visibility rules are keyed on. `AgentLoop`'s own
docstring makes the same separation deliberately, declining to migrate
the Orchestrator onto it because the approval path's argument-hash
binding, tenancy and audit semantics are the security core.

**Read-only, and structurally so.** An investigation is exploratory:
the model picks each next action from its own previous observations,
which is exactly the situation where a mutating call should not be
one of the options. The contract is granted only skills the registry
reports as `read_only`, and `AgentLoop._act_skill` re-checks the
contract on every call, so the restriction is enforced twice and
depends on the registry's own risk tier rather than on this module's
say-so. A mutating action belongs in a planned task, where it meets
the approval gate.

**Bounded by construction.** `max_iterations` caps the cycle, stall
detection stops a loop repeating itself, and a `BudgetTracker` caps
spend - a loop that cannot terminate by construction needs all three,
and the response reports which one stopped it.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from praxis.agents.budget import Budget, BudgetTracker
from praxis.agents.contract import TaskContract
from praxis.agents.loop import AgentLoop
from praxis.agents.policy import ModelPolicy
from praxis.agents.skill_registry import all_skills, discover_skills
from praxis.agents.subagent import all_agents, all_manifests, discover_agents
from praxis.api.dependencies import require
from praxis.llm.catalogue import LLMCatalogue
from praxis.security.policy import Permission
from praxis.security.principal import Principal

router = APIRouter(prefix="/investigations", tags=["investigations"])

# Deliberately below `AgentLoop`'s own default of 12. An investigation
# is answered synchronously, and each iteration is a real model call
# plus a real tool call; a lower ceiling keeps the request bounded
# while the caller can still ask for more.
_DEFAULT_MAX_ITERATIONS = 6
_MAX_ALLOWED_ITERATIONS = 20


class InvestigationRequest(BaseModel):
    objective: str = Field(
        min_length=1,
        description="The question to investigate, in one sentence",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Starting facts the loop should work from, e.g. an attachment id",
    )
    max_iterations: int = Field(
        default=_DEFAULT_MAX_ITERATIONS, ge=1, le=_MAX_ALLOWED_ITERATIONS
    )
    max_llm_calls: int | None = Field(
        default=None, ge=1, description="Spend ceiling for the whole investigation"
    )


def _read_only_tools() -> list[Any]:
    discover_skills()
    return [skill for skill in all_skills() if skill.risk == "read_only"]


@router.post("", status_code=200)
async def investigate(
    body: InvestigationRequest,
    principal: Annotated[Principal, Depends(require(Permission.TASK_CREATE))],
) -> dict[str, Any]:
    """Runs one bounded investigation and returns its full trace.

    The trace is the point, not a debugging extra: an agentic answer
    is only as trustworthy as the steps that produced it, so every
    iteration's action, arguments and observation come back with the
    answer. `stop_reason` says whether the loop finished because it
    answered the question, ran out of iterations, stalled, or
    exhausted its budget - four outcomes a single `succeeded` flag
    would flatten into one.
    """
    discover_agents()
    tools = _read_only_tools()
    tool_names = tuple(skill.name for skill in tools)

    contract = TaskContract(
        objective=body.objective,
        tenant_id=principal.tenant_id,
        inputs=dict(body.inputs),
        authorized_tools=tool_names,
    )

    tracker = BudgetTracker(
        Budget(max_llm_calls=body.max_llm_calls) if body.max_llm_calls else None
    )

    policy = ModelPolicy(
        # The tracker is handed to the *catalogue*, not just to the
        # loop. Without this the budget was decorative: the loop
        # charges tool calls and delegations against it, but the
        # policy's own per-iteration model call is the dominant cost
        # of an investigation, and a `max_llm_calls` ceiling that
        # never saw those calls could not stop anything.
        catalogue=LLMCatalogue(budget=tracker),
        tools=[
            {
                "name": skill.name,
                "risk": skill.risk,
                "inputs": ", ".join(f"{k} ({v})" for k, v in skill.inputs.items()),
                "outputs": ", ".join(skill.outputs),
            }
            for skill in tools
        ],
        agents=[
            {
                "name": manifest.name,
                "description": manifest.description,
                "inputs": ", ".join(manifest.tools) or "see description",
            }
            for manifest in all_manifests()
        ],
    )

    loop = AgentLoop(
        policy=policy,
        skills={skill.name: skill for skill in tools},
        agents={agent.manifest.name: agent for agent in all_agents()},
        max_iterations=body.max_iterations,
    )

    try:
        result = await loop.run(contract, budget=tracker)
    except Exception as exc:  # noqa: BLE001 - reported, never a 500 with no detail
        raise HTTPException(
            status_code=500, detail=f"investigation failed: {type(exc).__name__}: {exc}"
        ) from exc

    payload = result.to_dict()
    payload["spend"] = tracker.snapshot()
    payload["authorized_tools"] = list(tool_names)
    return payload
