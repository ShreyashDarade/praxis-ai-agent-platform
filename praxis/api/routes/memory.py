# praxis/api/routes/memory.py
"""Reading back what the system remembers.

Every task already wrote an episode (`Orchestrator._record_episode`)
and nothing ever read one, which made episodic memory write-only: rows
accumulated, no planner consulted them, and no operator could see
them. Memory you cannot read is storage, not memory.

Two things changed together, and both matter:

- the planner now recalls the prior runs of an intent before planning
  it (`Orchestrator._recall_episodes`), so memory affects behavior;
- this endpoint makes the same records inspectable, so a wrong or
  stale memory can be found and corrected rather than silently
  steering plans.

Scoping is not optional here. Every read is bound to the calling
principal's tenant and user, pushed down as SQL predicates by
`MemoryStore._scoped_query` rather than filtered afterwards - so a
caller cannot page through another tenant's history, and cannot infer
its size from a result count. `DELETE` is likewise scoped, and really
does delete every version - it is the right-to-be-forgotten path, not
a correction, and leaving a tombstone holding the value would defeat
the point of asking.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from praxis.api.dependencies import get_settings, require
from praxis.config import Settings
from praxis.memory.db import PostgresStore
from praxis.memory.memory_types import MemoryKind, MemoryScope, MemoryStore
from praxis.security.policy import Permission
from praxis.security.principal import Principal

router = APIRouter(prefix="/memory", tags=["memory"])


def _scope(principal: Principal) -> MemoryScope:
    return MemoryScope(tenant_id=principal.tenant_id, user_id=principal.user_id)


@router.get("")
async def recall_memories(
    principal: Annotated[Principal, Depends(require(Permission.TASK_READ))],
    settings: Annotated[Settings, Depends(get_settings)],
    kind: Annotated[
        str | None, Query(description="episodic, procedural, preference, workspace, ...")
    ] = None,
    key: Annotated[str | None, Query(description="exact memory key")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    trusted_only: Annotated[bool, Query()] = False,
) -> dict[str, Any]:
    """Live, in-scope memories, newest first.

    `trusted_only` is the switch for a caller about to *state* a
    memory as fact rather than merely consider it: it drops anything
    an agent inferred but nothing verified.
    """
    parsed_kind: MemoryKind | None = None
    if kind is not None:
        try:
            parsed_kind = MemoryKind(kind)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown memory kind '{kind}'; valid kinds are "
                    f"{[k.value for k in MemoryKind]}"
                ),
            ) from exc

    store = PostgresStore(settings)
    try:
        memories = await MemoryStore(store).recall(
            scope=_scope(principal),
            kind=parsed_kind,
            key=key,
            limit=limit,
            trusted_only=trusted_only,
        )
    finally:
        await store.dispose()

    return {"memories": [memory.to_dict() for memory in memories], "count": len(memories)}


@router.get("/history")
async def memory_history(
    principal: Annotated[Principal, Depends(require(Permission.TASK_READ))],
    settings: Annotated[Settings, Depends(get_settings)],
    kind: Annotated[str, Query(description="episodic, procedural, preference, ...")],
    key: Annotated[str, Query(description="exact memory key")],
) -> dict[str, Any]:
    """Every version of one memory, superseded ones included.

    This is what makes correcting a memory auditable rather than
    destructive - and, for an episode key, it is literally the run
    history of one intent.
    """
    try:
        parsed_kind = MemoryKind(kind)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"unknown memory kind '{kind}'; valid kinds are "
            f"{[k.value for k in MemoryKind]}",
        ) from exc

    store = PostgresStore(settings)
    try:
        versions = await MemoryStore(store).history(
            scope=_scope(principal), kind=parsed_kind, key=key
        )
    finally:
        await store.dispose()

    return {"key": key, "versions": [m.to_dict() for m in versions], "count": len(versions)}


@router.delete("", status_code=200)
async def forget_memory(
    principal: Annotated[Principal, Depends(require(Permission.TASK_CREATE))],
    settings: Annotated[Settings, Depends(get_settings)],
    kind: Annotated[str, Query(description="episodic, procedural, preference, ...")],
    key: Annotated[str, Query(description="exact memory key")],
) -> dict[str, Any]:
    """Forgets every version of one memory.

    Identified by kind and key rather than by row id, because that is
    what `MemoryStore.forget` is scoped on and because a memory is
    conceptually one thing with a history, not a row.

    Returns the number of versions removed; zero is reported as a 404
    rather than a silent success, so a caller can tell "forgotten"
    apart from "never existed here" - and a key belonging to another
    tenant is indistinguishable from one that does not exist, which is
    the point.
    """
    try:
        parsed_kind = MemoryKind(kind)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"unknown memory kind '{kind}'; valid kinds are "
            f"{[k.value for k in MemoryKind]}",
        ) from exc

    store = PostgresStore(settings)
    try:
        removed = await MemoryStore(store).forget(
            scope=_scope(principal), kind=parsed_kind, key=key
        )
    finally:
        await store.dispose()

    if not removed:
        raise HTTPException(
            status_code=404, detail=f"no {kind} memory with key '{key}' in this scope"
        )
    return {"forgotten": key, "versions_removed": removed}
