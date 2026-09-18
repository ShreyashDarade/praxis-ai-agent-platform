# praxis/api/routes/skills.py
"""The skills catalogue, and adding a capability as markdown.

The gap this closes, stated plainly: there was no endpoint to add a
skill. `SkillManifest` could parse a SKILL.md and nothing called that
parser outside a unit test, so the three documents under `skills/` were
documentation of a format the running system did not read. Adding a
capability meant writing a Python module and redeploying.

`POST /skills` takes the markdown itself. A procedure that composes
tools which already exist and are already approved needs no new code,
which is the brief's preferred path for a new capability and its main
lever for "less code".

What the endpoint refuses, and why it refuses at submit time rather
than at run time:

- **Markdown that does not parse**, or is missing `name`/`description`.
- **A step naming a tool that does not exist.** Accepting it would
  hand the author a success and whoever ran it a failure.
- **Under-declared risk.** A procedure that says `read_only` while
  calling a mutating tool is rejected outright, not corrected -
  silently fixing it would let a mutating action slip past the
  Orchestrator's approval gate, and would hide that the author
  misunderstood what they wrote.
- **An unwirable chain**, such as a step reading an output no earlier
  step produces.

Approval applies. A procedure cannot execute arbitrary code, so it is
genuinely safer than generated Python - but it still chooses which
permitted tools run with which arguments, so it is catalogued as
`pending_approval` under the same `require_skill_approval` setting and
is not callable until someone with `skill:approve` publishes it.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from praxis.agents.manifest import ManifestError
from praxis.agents.procedure import ProcedureError
from praxis.agents.procedure_registry import (
    activate_procedure,
    describe_loaded,
    register_procedure,
)
from praxis.agents.publication import SkillPublisher, SkillStatus
from praxis.api.dependencies import get_settings, require
from praxis.config import Settings
from praxis.memory.db import PostgresStore
from praxis.memory.models import SkillRecord
from praxis.security.policy import Permission
from praxis.security.principal import Principal

router = APIRouter(prefix="/skills", tags=["skills"])


class CreateSkillRequest(BaseModel):
    """A whole SKILL.md document, as text.

    Markdown rather than a JSON object on purpose: the format is the
    brief's, it is what the three sample packages already use, and it
    is reviewable as a diff by whoever has to approve it.
    """

    markdown: str = Field(
        min_length=1,
        description="A complete SKILL.md: '---' frontmatter with name, description, "
        "risk and steps, followed by the human-readable procedure",
    )


class ApproveSkillRequest(BaseModel):
    version: int = Field(default=1, ge=1)
    expected_code_hash: str | None = Field(
        default=None,
        description="Binds the approval to exactly the document that was reviewed",
    )


def _serialize(record: SkillRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "name": record.name,
        "version": record.version,
        "status": record.status,
        "risk": record.risk,
        "synthesized": record.synthesized,
        "description": record.description,
        "tools": list(record.dependencies or []),
        "inputs": dict(record.inputs_schema or {}),
        "outputs": dict(record.outputs_schema or {}),
        "code_hash": record.code_hash,
        "created_at": record.created_at.isoformat(),
    }


@router.post("", status_code=201)
async def create_skill(
    body: CreateSkillRequest,
    principal: Annotated[Principal, Depends(require(Permission.SKILL_SYNTHESIZE))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Adds a capability from a SKILL.md document.

    Returns 201 with the catalogue row. `active` says whether it is
    callable yet: under the default approval policy it is not, and
    `POST /skills/{name}/approve` is what publishes it.
    """
    store = PostgresStore(settings)
    try:
        manifest, record, active = await register_procedure(
            store,
            body.markdown,
            tenant_id=principal.tenant_id,
            created_by_user_id=principal.user_id,
            require_approval=settings.require_skill_approval,
        )
    except ManifestError as exc:
        # The document itself is malformed - a different problem from a
        # well-formed document that cannot be wired, and worth telling
        # the author apart.
        raise HTTPException(status_code=400, detail=f"malformed SKILL.md: {exc}") from exc
    except ProcedureError as exc:
        # A conflict is not the author's mistake, so it is not a 400.
        status = 409 if exc.reason == "version_conflict" else 400
        raise HTTPException(
            status_code=status,
            detail={"message": str(exc), "reason": exc.reason, "skill": exc.skill},
        ) from exc
    finally:
        await store.dispose()

    return {
        **_serialize(record),
        "active": active,
        "keywords": manifest.keywords,
    }


@router.get("")
async def list_skills(
    principal: Annotated[Principal, Depends(require(Permission.SKILL_READ))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """The catalogue for this tenant, plus what is loaded right now.

    Both, because they answer different questions: the catalogue says
    what exists and what its status is, and `loaded` says what the
    running process will actually accept a call for.
    """
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            rows = (
                (
                    await session.execute(
                        select(SkillRecord)
                        .where(SkillRecord.tenant_id == principal.tenant_id)
                        .order_by(SkillRecord.created_at.desc())
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await store.dispose()

    return {
        "skills": [_serialize(row) for row in rows],
        "loaded": describe_loaded(),
    }


@router.post("/{name}/approve")
async def approve_skill(
    name: str,
    body: ApproveSkillRequest,
    principal: Annotated[Principal, Depends(require(Permission.SKILL_APPROVE))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Publishes a pending capability, making it callable.

    This is the surface the audit found missing: `SkillPublisher` had
    approve/reject/rollback and no caller, so approving anything meant
    running Python by hand.
    """
    store = PostgresStore(settings)
    try:
        publisher = SkillPublisher(store)
        try:
            record = await publisher.approve(
                name=name,
                version=body.version,
                approver=principal,
                expected_code_hash=body.expected_code_hash,
            )
        except PermissionError:
            raise
        except Exception as exc:  # noqa: BLE001 - PublicationError and friends
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # A procedure's "code" is its markdown, which lives in the file
        # it came from. Re-compiling and registering here is what makes
        # approval actually publish it, mirroring how an approved
        # generated skill is moved out of quarantine and imported.
        if not record.synthesized and record.definition:
            # Re-compiled from the stored document rather than from an
            # object held since submission: that is what binds the
            # published capability to the bytes that were reviewed.
            activate_procedure(record.definition)
    finally:
        await store.dispose()

    return {**_serialize(record), "active": SkillStatus(record.status).is_callable}
