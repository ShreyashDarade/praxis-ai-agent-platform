"""SQLAlchemy models for Praxis's relational + vector + graph store (spec §2, §9).

**Phase 12 (multi-tenancy)**: every tenant-scoped row now carries a
`tenant_id`. `DEFAULT_TENANT_ID` below is a real, always-present tenant
row (created by migration `0003`), not a sentinel meaning "no tenant" -
a single-operator deployment with `PRAXIS_AUTH_ENABLED=false` simply
runs everything inside that one tenant, so the isolation code path is
identical whether or not auth is switched on. Nothing is ever written
with a NULL tenant.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# The one tenant every pre-Phase-12 row is backfilled into, and the one
# an auth-disabled deployment operates as. A fixed, well-known UUID (not
# randomly generated at migration time) so application code, migrations,
# and tests can all refer to the same tenant without a lookup.
DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_TENANT_SLUG = "default"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    """An isolation boundary (spec §20/Prompt §11's "multi-tenancy, tenant
    data isolation"). Every tenant-scoped table below carries a
    `tenant_id` FK-by-convention to this table's `id`."""

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(128), unique=True)
    name: Mapped[str] = mapped_column(String(256))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class User(Base):
    """A principal that can act inside exactly one tenant.

    `roles` is a JSON list of role names resolved against
    `praxis.security.policy.ROLE_PERMISSIONS` - roles are the RBAC half;
    the tenant match on each resource is the ABAC half (see
    `praxis.security.policy.PolicyEngine`).
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    email: Mapped[str] = mapped_column(String(320), index=True)
    display_name: Mapped[str] = mapped_column(String(256), default="")
    roles: Mapped[list] = mapped_column(JSON, default=list)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ApiKey(Base):
    """A credential binding an inbound request to a `User` (and therefore
    a tenant). Only the SHA-256 hash of the key is ever stored - the
    plaintext is shown exactly once, at creation
    (`praxis.security.api_key.generate_api_key`).

    `scopes` optionally *narrows* the owning user's permissions for calls
    made with this key (never widens them - see
    `PolicyEngine.authorize`), which is what makes a
    read-only/CI-scoped key meaningful.
    """

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    name: Mapped[str] = mapped_column(String(256), default="")
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    scopes: Mapped[list | None] = mapped_column(JSON, nullable=True, default=None)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditLog(Base):
    """An append-only record of every authorization decision and every
    mutating action (Prompt §9's "audit log", §8's "audit origin, code
    hash, test evidence, approver, and version").

    Deliberately stores *what was decided and why*, never model
    chain-of-thought or raw sensitive payloads - `detail` is redacted
    through `praxis.security.redaction.redact` before it is written.
    """

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    actor_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(128), index=True)
    resource_type: Mapped[str] = mapped_column(String(64), default="")
    resource_id: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    decision: Mapped[str] = mapped_column(String(32), default="allowed")
    reason: Mapped[str] = mapped_column(Text, default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    correlation_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class ApprovalRecord(Base):
    """A durable, identity-bound record of one human approval decision
    (Prompt §8: "Approval records must bind approver, tenant, exact
    action/arguments, expiry, and idempotency key").

    `action_hash` is a SHA-256 over the exact `(skill_name, resolved
    args)` the operator was shown - re-checked at resume time, so an
    approval can never be replayed against *different* arguments than
    the ones a human actually saw (`praxis.security.approval`).
    """

    __tablename__ = "approval_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    task_id: Mapped[str] = mapped_column(String(36), index=True)
    step_index: Mapped[int] = mapped_column(Integer, default=0)
    action: Mapped[str] = mapped_column(String(256), default="")
    action_hash: Mapped[str] = mapped_column(String(64), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    approver_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    approved: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=None)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Task(Base):
    """A task moving through the Orchestrator (Phase 5, spec §3/§7/§8/§9/§14).

    `status` stays a plain string column (no DB-level enum, matching the
    rest of this schema's style) but its value set is now:
    `pending` (row created, not yet planned) -> `planning` (Planner is
    decomposing the intent) -> `running` (executing the plan) ->
    `awaiting_approval` | `awaiting_clarification` (paused on a
    `PendingInput`, see `praxis.core.risk_policy.PendingInput`) ->
    `completed` | `failed` (terminal).

    `checklist` is the materialized plan (spec §7's "every step becomes
    a visible checklist item"): a list of
    `{"content": str, "status": "pending"|"in_progress"|"completed"|"skipped",
    "reason": str | None}` dicts, one per `PlanStep`, mutated (via
    reassignment, never in-place - see `Orchestrator._mark_item`) by the
    Orchestrator as execution proceeds - this is what `GET /tasks/{id}`
    (§14) renders as real progress, never a heuristic percentage. A
    mutating step's item also transiently reads `"awaiting_approval"`
    while the task itself is paused on it, mirroring the task-level
    status below (`Orchestrator._process_level`).

    `pending_input` is `None` except while the task is paused, in which
    case it holds the `PendingInput` this task is waiting on, serialized
    as `{"kind": "approval"|"clarification", "detail": str,
    "options": list[str] | None}`.

    `result` is `None` until the task reaches a terminal status, then
    holds the final delivered output (a short structured summary of what
    happened plus each step's output for `completed`, or a clear failure
    message for `failed`).
    """

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    created_by_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    correlation_id: Mapped[str] = mapped_column(String(36), default=_uuid)
    intent_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    # Phase 13: which execution mode this task runs under
    # (`praxis.core.execution_mode.ExecutionMode`) - `execute` is the
    # historical behavior and stays the default.
    mode: Mapped[str] = mapped_column(String(16), default="execute")
    checklist: Mapped[list] = mapped_column(JSON, default=list)
    pending_input: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Attachment(Base):
    __tablename__ = "attachments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    uploaded_by_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    mime_type: Mapped[str] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(512))
    size_bytes: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SkillRecord(Base):
    """The relational skills catalogue (spec §9).

    **Phase 16** widens this well past the original five columns, so a
    skill carries the full manifest metadata Prompt §7 requires
    (version, owner, permissions, supported connectors, budget, model
    requirements, test cases, health, dependencies, audit trail) and can
    be published as an *immutable version* that requires approval before
    activation - hence `(name, version)` uniqueness rather than `name`
    alone, plus the `status` lifecycle
    (`pending_approval` -> `active` -> `deprecated` | `revoked`).
    """

    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    name: Mapped[str] = mapped_column(String(256), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    risk: Mapped[str] = mapped_column(String(16))  # "read_only" | "mutating"
    synthesized: Mapped[bool] = mapped_column(Boolean, default=False)
    inputs_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    outputs_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    # Phase 16 manifest metadata (Prompt §7's required field list).
    owner: Mapped[str] = mapped_column(String(256), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    required_permissions: Mapped[list] = mapped_column(JSON, default=list)
    supported_connectors: Mapped[list] = mapped_column(JSON, default=list)
    dependencies: Mapped[list] = mapped_column(JSON, default=list)
    model_requirements: Mapped[dict] = mapped_column(JSON, default=dict)
    execution_budget: Mapped[dict] = mapped_column(JSON, default=dict)
    test_cases: Mapped[list] = mapped_column(JSON, default=list)
    code_hash: Mapped[str] = mapped_column(String(64), default="")
    source_path: Mapped[str] = mapped_column(String(512), default="")
    approved_by_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    health_status: Mapped[str] = mapped_column(String(32), default="unknown")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class HealthRecord(Base):
    __tablename__ = "health_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    component: Mapped[str] = mapped_column(String(128))
    healthy: Mapped[bool] = mapped_column(Boolean)
    detail: Mapped[str] = mapped_column(Text, default="")
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class VectorChunk(Base):
    __tablename__ = "vector_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    doc_id: Mapped[str] = mapped_column(String(512), index=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))  # all-MiniLM-L6-v2 dimension
    chunk_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class GraphEdge(Base):
    __tablename__ = "graph_edges"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    source: Mapped[str] = mapped_column(String(512), index=True)
    relation: Mapped[str] = mapped_column(String(128))
    target: Mapped[str] = mapped_column(String(512), index=True)
    edge_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
