# praxis/api/routes/agents.py
"""The specialist catalogue and the delegation limits that bound it.

Praxis shipped five specialists, task contracts, per-agent budgets, a
delegation registry and a critic - and no way to see any of it from
outside the process. A capability nobody can enumerate is a capability
nobody can plan around, review, or safely rely on.

This is read-only on purpose. A specialist is *invoked* by planning a
`delegate_to_specialist` step through `POST /intent`, so that every
delegation goes through the same approval gate, risk tiering, audit
trail and re-plan loop as any other skill. An endpoint that ran a
specialist directly would be a second execution path around all of
those controls, which is precisely the kind of side door the rest of
this codebase is built to avoid.

What the catalogue is for: a caller (or a person writing an intent)
can see which specialists exist, what each is for, which tools each is
permitted, and what its default budget is - the same manifest the
supervisor routes on.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from praxis.agents.delegation import DelegationLimits
from praxis.agents.subagent import all_manifests, discover_agents, get_agent
from praxis.api.dependencies import require
from praxis.security.policy import Permission
from praxis.security.principal import Principal

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("")
async def list_agents(
    principal: Annotated[Principal, Depends(require(Permission.SKILL_READ))],
) -> dict[str, Any]:
    """Every registered specialist, with the limits delegation enforces.

    `limits` travels with the list because they are what makes a plan
    that fans out widely fail predictably rather than at an arbitrary
    point: a caller can see the ceiling before planning against it.
    """
    del principal  # authorization is the dependency's job; the list is not tenant-scoped
    discover_agents()
    limits = DelegationLimits()
    return {
        "agents": [manifest.to_dict() for manifest in all_manifests()],
        "limits": {
            "max_depth": limits.max_depth,
            "max_concurrency": limits.max_concurrency,
            "max_children_per_task": limits.max_children_per_task,
        },
        "invoked_via": (
            "POST /intent with a plan step naming the 'delegate_to_specialist' skill"
        ),
    }


@router.get("/{name}")
async def get_agent_manifest(
    name: str,
    principal: Annotated[Principal, Depends(require(Permission.SKILL_READ))],
) -> dict[str, Any]:
    del principal
    discover_agents()
    try:
        agent = get_agent(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return agent.manifest.to_dict()
