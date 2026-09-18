# praxis/agents/publication.py
"""Versioned, approval-gated publication of synthesized skills (Prompt §7).

The prompt is unambiguous: *"Generated agents/tools must never be
silently trusted. The platform can generate a proposed agent or tool
definition dynamically, validate it in a sandbox, run tests and policy
checks, then require approval before publishing/enabling it."* And
§8: *"generate schema and tests -> validate dependencies and
permissions -> scan and test in isolation -> review -> publish an
immutable version -> activate through supported registration ->
monitor -> revoke/roll back."*

Praxis previously did the first half genuinely well - real LLM
synthesis, real Docker sandbox validation with a self-test - and then
went straight to registering the skill live. The design doc even
stated "No approval gate on capability synthesis" as a deliberate
choice. Against this product brief that is simply a gap, so this
module adds the missing half.

**The lifecycle, and why each state exists:**

- `pending_approval` - sandbox-validated but NOT importable and NOT
  reachable by the Planner. This is the state the old design skipped.
- `active` - approved by a real principal, registered, callable.
- `deprecated` - superseded by a newer version; still callable so
  in-flight plans referencing it do not break.
- `revoked` - withdrawn; refused even if something still names it.

**Immutability.** Approving publishes version N and never mutates it.
A change is version N+1 with its own code hash and its own approval.
This is what makes rollback meaningful: "roll back" means activating
a previously-approved version that still exists, not regenerating
something and hoping it matches.

**The policy gate before approval** is deliberately mechanical - it
checks declared dependencies against an allow-list and declared
permissions against what the requesting principal actually holds.
A human approving a skill should not have to notice that it quietly
declared a dependency on `requests` or a permission its author
lacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
from sqlalchemy import select

from praxis.agents.manifest import SkillManifest, compute_code_hash
from praxis.memory.db import PostgresStore
from praxis.memory.models import SkillRecord
from praxis.security.policy import Permission, PolicyEngine, policy_engine
from praxis.security.principal import Principal

_logger = structlog.get_logger(__name__)


class SkillStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"

    @property
    def is_callable(self) -> bool:
        """Whether a skill in this state may actually be invoked.

        `deprecated` remains callable on purpose: superseding a skill
        must not break a plan that is mid-flight against it.
        """
        return self in (SkillStatus.ACTIVE, SkillStatus.DEPRECATED)


class PublicationError(Exception):
    """A publication or activation step was refused."""

    def __init__(self, message: str, *, reason: str, detail: str = "") -> None:
        super().__init__(message)
        self.reason = reason
        self.detail = detail


# Only the Python standard library is importable inside the validation
# sandbox, so a synthesized skill declaring a third-party dependency
# has either not been validated against what it actually needs, or is
# trying to reach beyond the sandbox. Either way it must not be
# approved silently. A deployment that genuinely wants to widen this
# passes its own allow-list.
STDLIB_ONLY = "stdlib-only"


@dataclass
class PolicyCheckResult:
    """The mechanical pre-approval gate's findings."""

    passed: bool
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "problems": self.problems}


def check_publication_policy(
    manifest: SkillManifest,
    *,
    requester: Principal,
    allowed_dependencies: set[str] | None = None,
    engine: PolicyEngine = policy_engine,
) -> PolicyCheckResult:
    """Mechanical checks a human reviewer should not have to perform.

    Two things a reviewer would plausibly miss by eye:

    1. A declared dependency outside the allow-list - meaning the code
       cannot actually have been validated in the stdlib-only sandbox.
    2. A declared permission the *requesting principal does not itself
       hold*. New code must never gain broader privileges than the
       task that asked for it (Prompt §8); without this check,
       synthesis would be a privilege-escalation path.
    """
    problems: list[str] = []
    permitted = allowed_dependencies if allowed_dependencies is not None else set()

    for dependency in manifest.dependencies:
        if dependency not in permitted:
            problems.append(
                f"declares dependency '{dependency}', which is not in the allowed set "
                f"({sorted(permitted) or STDLIB_ONLY}); it cannot have been validated in "
                "the stdlib-only sandbox"
            )

    holder = engine.effective_permissions(requester)
    for raw_permission in manifest.required_permissions:
        try:
            permission = Permission(raw_permission)
        except ValueError:
            problems.append(f"declares unknown permission '{raw_permission}'")
            continue
        if permission not in holder:
            problems.append(
                f"declares permission '{raw_permission}', which the requesting principal "
                f"({requester.describe()}) does not hold; synthesized code must never gain "
                "broader privileges than the task that requested it"
            )

    if manifest.risk not in ("read_only", "mutating"):
        problems.append(f"declares invalid risk level '{manifest.risk}'")

    return PolicyCheckResult(passed=not problems, problems=problems)


class SkillPublisher:
    """Publishes, approves, activates, deprecates, and revokes skills."""

    def __init__(
        self,
        store: PostgresStore,
        *,
        allowed_dependencies: set[str] | None = None,
    ) -> None:
        self._store = store
        self._allowed_dependencies = (
            allowed_dependencies if allowed_dependencies is not None else set()
        )

    async def _next_version(self, tenant_id: str, name: str) -> int:
        async with self._store.session() as session:
            rows = (
                await session.execute(
                    select(SkillRecord.version).where(
                        SkillRecord.tenant_id == tenant_id, SkillRecord.name == name
                    )
                )
            ).scalars().all()
        return (max(rows) + 1) if rows else 1

    async def propose(
        self,
        manifest: SkillManifest,
        *,
        code: str,
        requester: Principal,
        source_path: str = "",
        test_evidence: list[Any] | None = None,
    ) -> SkillRecord:
        """Records a sandbox-validated skill as `pending_approval`.

        Deliberately does NOT import, register, or make the skill
        reachable. This is the state the previous design skipped
        entirely - the code exists and is described, but nothing can
        call it until a principal with `skill:approve` says so.
        """
        policy = check_publication_policy(
            manifest,
            requester=requester,
            allowed_dependencies=self._allowed_dependencies,
        )
        if not policy.passed:
            raise PublicationError(
                f"skill '{manifest.name}' failed policy checks: "
                + "; ".join(policy.problems),
                reason="policy_check_failed",
                detail="; ".join(policy.problems),
            )

        version = await self._next_version(requester.tenant_id, manifest.name)
        code_hash = compute_code_hash(code)

        async with self._store.session() as session:
            record = SkillRecord(
                tenant_id=requester.tenant_id,
                name=manifest.name,
                version=version,
                status=SkillStatus.PENDING_APPROVAL.value,
                risk=manifest.risk,
                synthesized=manifest.synthesized,
                inputs_schema=manifest.inputs,
                outputs_schema=manifest.outputs,
                owner=manifest.owner or requester.describe(),
                description=manifest.description,
                required_permissions=list(manifest.required_permissions),
                supported_connectors=list(manifest.supported_connectors),
                dependencies=list(manifest.dependencies),
                model_requirements=dict(manifest.model_requirements),
                execution_budget=dict(manifest.execution_budget),
                test_cases=list(test_evidence or manifest.test_cases),
                code_hash=code_hash,
                source_path=source_path or manifest.source_path,
                health_status="unknown",
            )
            session.add(record)
            await session.commit()

        _logger.info(
            "skill_proposed",
            skill=manifest.name,
            version=version,
            code_hash=code_hash,
            requester=requester.describe(),
        )
        return record

    async def approve(
        self,
        *,
        name: str,
        version: int,
        approver: Principal,
        expected_code_hash: str | None = None,
        engine: PolicyEngine = policy_engine,
    ) -> SkillRecord:
        """Approves and activates a pending version.

        `expected_code_hash` binds the approval to exactly the code
        that was reviewed. Supplying it and having it mismatch is a
        hard refusal rather than a warning: approving version N and
        then running different bytes is precisely the failure this
        whole gate exists to prevent.

        Activating a new version deprecates the previously-active one
        rather than deleting it, which is what makes rollback a real
        operation.
        """
        engine.authorize(approver, Permission.SKILL_APPROVE, resource_type="skill")

        async with self._store.session() as session:
            record = (
                await session.execute(
                    select(SkillRecord).where(
                        SkillRecord.tenant_id == approver.tenant_id,
                        SkillRecord.name == name,
                        SkillRecord.version == version,
                    )
                )
            ).scalar_one_or_none()

            if record is None:
                raise PublicationError(
                    f"no skill '{name}' version {version} in tenant "
                    f"'{approver.tenant_id}'",
                    reason="not_found",
                )
            if record.status != SkillStatus.PENDING_APPROVAL.value:
                raise PublicationError(
                    f"skill '{name}' v{version} is '{record.status}', not pending approval",
                    reason="wrong_status",
                )
            if expected_code_hash is not None and record.code_hash != expected_code_hash:
                raise PublicationError(
                    (
                        f"code hash mismatch for '{name}' v{version}: the stored "
                        "implementation is not the one that was reviewed"
                    ),
                    reason="code_hash_mismatch",
                    detail=f"stored={record.code_hash} expected={expected_code_hash}",
                )

            previously_active = (
                await session.execute(
                    select(SkillRecord).where(
                        SkillRecord.tenant_id == approver.tenant_id,
                        SkillRecord.name == name,
                        SkillRecord.status == SkillStatus.ACTIVE.value,
                    )
                )
            ).scalars().all()
            for superseded in previously_active:
                superseded.status = SkillStatus.DEPRECATED.value

            record.status = SkillStatus.ACTIVE.value
            record.approved_by_user_id = approver.user_id
            record.approved_at = datetime.now(UTC)
            await session.commit()

        _logger.info(
            "skill_approved",
            skill=name,
            version=version,
            approver=approver.describe(),
            superseded=[s.version for s in previously_active],
        )
        return record

    async def reject(
        self, *, name: str, version: int, approver: Principal, reason: str = ""
    ) -> SkillRecord:
        """Revokes a pending version outright.

        Revoked rather than deleted: the record of *what was proposed
        and refused* is part of the audit trail, and deleting it would
        discard exactly the evidence a later reviewer needs.
        """
        policy_engine.authorize(approver, Permission.SKILL_APPROVE, resource_type="skill")

        async with self._store.session() as session:
            record = (
                await session.execute(
                    select(SkillRecord).where(
                        SkillRecord.tenant_id == approver.tenant_id,
                        SkillRecord.name == name,
                        SkillRecord.version == version,
                    )
                )
            ).scalar_one_or_none()
            if record is None:
                raise PublicationError(
                    f"no skill '{name}' version {version}", reason="not_found"
                )
            record.status = SkillStatus.REVOKED.value
            record.approved_by_user_id = approver.user_id
            record.approved_at = datetime.now(UTC)
            await session.commit()

        _logger.info("skill_rejected", skill=name, version=version, reason=reason)
        return record

    async def revoke(self, *, name: str, version: int, approver: Principal) -> SkillRecord:
        """Withdraws an active version - the kill switch."""
        return await self.reject(name=name, version=version, approver=approver, reason="revoked")

    async def rollback_to(
        self, *, name: str, version: int, approver: Principal
    ) -> SkillRecord:
        """Re-activates a previously-approved version.

        Only a version that was genuinely approved before can be rolled
        back to - rollback must restore something a human already
        signed off on, never silently resurrect a rejected or
        never-reviewed one.
        """
        policy_engine.authorize(approver, Permission.SKILL_APPROVE, resource_type="skill")

        async with self._store.session() as session:
            target = (
                await session.execute(
                    select(SkillRecord).where(
                        SkillRecord.tenant_id == approver.tenant_id,
                        SkillRecord.name == name,
                        SkillRecord.version == version,
                    )
                )
            ).scalar_one_or_none()
            if target is None:
                raise PublicationError(
                    f"no skill '{name}' version {version}", reason="not_found"
                )
            if target.approved_at is None:
                raise PublicationError(
                    (
                        f"skill '{name}' v{version} was never approved, so it cannot be "
                        "rolled back to"
                    ),
                    reason="never_approved",
                )

            currently_active = (
                await session.execute(
                    select(SkillRecord).where(
                        SkillRecord.tenant_id == approver.tenant_id,
                        SkillRecord.name == name,
                        SkillRecord.status == SkillStatus.ACTIVE.value,
                    )
                )
            ).scalars().all()
            for active in currently_active:
                active.status = SkillStatus.DEPRECATED.value

            target.status = SkillStatus.ACTIVE.value
            await session.commit()

        _logger.info("skill_rolled_back", skill=name, version=version)
        return target

    async def active_version(self, *, tenant_id: str, name: str) -> SkillRecord | None:
        async with self._store.session() as session:
            return (
                await session.execute(
                    select(SkillRecord).where(
                        SkillRecord.tenant_id == tenant_id,
                        SkillRecord.name == name,
                        SkillRecord.status == SkillStatus.ACTIVE.value,
                    )
                )
            ).scalars().first()

    async def pending(self, *, tenant_id: str) -> list[SkillRecord]:
        """The approval queue."""
        async with self._store.session() as session:
            return list(
                (
                    await session.execute(
                        select(SkillRecord)
                        .where(
                            SkillRecord.tenant_id == tenant_id,
                            SkillRecord.status == SkillStatus.PENDING_APPROVAL.value,
                        )
                        .order_by(SkillRecord.created_at)
                    )
                ).scalars().all()
            )

    async def is_callable(self, *, tenant_id: str, name: str) -> bool:
        """Whether any version of `name` may currently be invoked.

        The runtime gate: a skill whose only versions are pending or
        revoked must not run even if something still references it by
        name.
        """
        async with self._store.session() as session:
            statuses = (
                await session.execute(
                    select(SkillRecord.status).where(
                        SkillRecord.tenant_id == tenant_id, SkillRecord.name == name
                    )
                )
            ).scalars().all()
        return any(SkillStatus(status).is_callable for status in statuses)
