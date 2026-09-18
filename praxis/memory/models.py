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
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# The one tenant every pre-Phase-12 row is backfilled into, and the one
# an auth-disabled deployment operates as. A fixed, well-known UUID (not
# randomly generated at migration time) so application code, migrations,
# and tests can all refer to the same tenant without a lookup.
DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_TENANT_SLUG = "default"


def _now() -> datetime:
    return datetime.now(UTC)


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
    # The conversation turn that produced this task, when it came
    # from chat rather than a standalone `POST /intent`. Nullable so
    # every existing caller is unchanged.
    conversation_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    # Which connector this task runs against, persisted rather than
    # held in the starting process's memory - a task resumed after a
    # restart must reach the same data source, not lose it.
    connector_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    # Phase 13: which execution mode this task runs under
    # (`praxis.core.execution_mode.ExecutionMode`) - `execute` is the
    # historical behavior and stays the default.
    mode: Mapped[str] = mapped_column(String(16), default="execute")
    # Phase 21: the materialized plan (list of
    # `{skill_name, args, depends_on}`). Durable task data, not
    # ephemeral execution state - which is why it lives here rather
    # than in the LangGraph checkpoint. Resuming a task in a fresh
    # process needs the plan to rebuild the same graph; the
    # checkpoint then supplies how far it got.
    plan: Mapped[list] = mapped_column(JSON, default=list)
    checklist: Mapped[list] = mapped_column(JSON, default=list)
    pending_input: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class LangGraphCheckpoint(Base):
    """One LangGraph state snapshot (Phase 21).

    Written by `praxis.core.checkpoint.PraxisCheckpointSaver`, which
    implements LangGraph's `BaseCheckpointSaver` over this table
    rather than using its official `AsyncPostgresSaver` - that one
    requires `psycopg`, whose async mode cannot run on Windows'
    ProactorEventLoop. Implementing the library's own extension point
    keeps one driver and one connection pool.

    Retaining every snapshot (rather than upserting one row per task,
    as the previous hand-rolled checkpointer did) is what makes
    time-travel and replay possible: `seq` orders them, because
    checkpoint ids are UUIDs and are not lexically time-ordered.
    """

    __tablename__ = "langgraph_checkpoints"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    thread_id: Mapped[str] = mapped_column(String(128), index=True)
    checkpoint_ns: Mapped[str] = mapped_column(String(256), default="")
    checkpoint_id: Mapped[str] = mapped_column(String(128), index=True)
    parent_checkpoint_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    checkpoint: Mapped[bytes] = mapped_column(LargeBinary)
    checkpoint_type: Mapped[str] = mapped_column(String(64), default="json")
    checkpoint_metadata: Mapped[bytes] = mapped_column(LargeBinary)
    metadata_type: Mapped[str] = mapped_column(String(64), default="json")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class LangGraphWrite(Base):
    """Pending channel writes for a partially-completed superstep.

    What lets an interrupted step resume without discarding the work
    its siblings already finished in the same superstep.
    """

    __tablename__ = "langgraph_writes"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    thread_id: Mapped[str] = mapped_column(String(128), index=True)
    checkpoint_ns: Mapped[str] = mapped_column(String(256), default="")
    checkpoint_id: Mapped[str] = mapped_column(String(128), index=True)
    task_id: Mapped[str] = mapped_column(String(128))
    task_path: Mapped[str] = mapped_column(String(256), default="")
    idx: Mapped[int] = mapped_column(Integer, default=0)
    channel: Mapped[str] = mapped_column(String(256))
    value: Mapped[bytes] = mapped_column(LargeBinary)
    value_type: Mapped[str] = mapped_column(String(64), default="json")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


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


class DashboardRecord(Base):
    """A saved dashboard (Phase 15, Prompt §2's "save the dashboard
    configuration and allow scheduled refreshes").

    The whole dashboard is stored as one validated `spec` JSON document
    (`praxis.analytics.dashboard.DashboardSpec`) rather than being
    shredded across panel/filter/encoding tables. That is deliberate:
    the spec is a *document* that is always read and written whole, is
    versioned by `spec_version` so a later release can migrate it, and
    has no query pattern that would benefit from relational
    decomposition. Shredding it would buy nothing and make round-trip
    fidelity - the property that makes a dashboard exportable and
    diffable - much harder to guarantee.
    """

    __tablename__ = "dashboards"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    title: Mapped[str] = mapped_column(String(512))
    description: Mapped[str] = mapped_column(Text, default="")
    owner_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    spec_version: Mapped[int] = mapped_column(Integer, default=1)
    spec: Mapped[dict] = mapped_column(JSON, default=dict)
    # Scheduled-refresh wiring; `None` means this dashboard is only
    # refreshed on load or on demand.
    refresh_interval_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_refreshed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class ScheduleRecord(Base):
    """A user-defined recurring task (Phase 17, Prompt §9).

    Stores `intent_text` rather than a frozen plan deliberately: the
    plan is produced fresh at each fire time, because the data, the
    available skills, and the connectors may all have changed since
    the schedule was created. A frozen plan would go quietly stale.

    `principal_user_id` is a *reference*, never a stored credential -
    credentials are resolved fresh at fire time so a revoked key stops
    the schedule at its next run instead of it continuing on a stale
    copy (see `praxis.agents.schedules`).
    """

    __tablename__ = "schedules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    name: Mapped[str] = mapped_column(String(256))
    intent_text: Mapped[str] = mapped_column(Text)
    principal_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    interval_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cron: Mapped[str | None] = mapped_column(String(128), nullable=True)
    timezone_name: Mapped[str] = mapped_column(String(64), default="UTC")
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    overlap_policy: Mapped[str] = mapped_column(String(32), default="skip")
    missed_run_policy: Mapped[str] = mapped_column(String(32), default="run_once")
    connector_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    last_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    next_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class ScheduleRun(Base):
    """One execution of a schedule.

    `idempotency_key` is unique and derived from
    `(schedule_id, scheduled_for)`, so two workers racing on the same
    slot cannot both execute it - deduplication by construction rather
    than by locking.
    """

    __tablename__ = "schedule_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    schedule_id: Mapped[str] = mapped_column(String(36), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    scheduled_for: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    task_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class DeadLetterEntry(Base):
    """Failed background work that needs a human (Phase 17, Prompt §9).

    `failures` keeps the WHOLE attempt history, not just the last
    error: "three timeouts then an auth error" is a materially
    different diagnosis from "an auth error", and only the history
    distinguishes them.
    """

    __tablename__ = "dead_letter"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    job_type: Mapped[str] = mapped_column(String(128), index=True)
    job_key: Mapped[str] = mapped_column(String(256), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    failures: Mapped[list] = mapped_column(JSON, default=list)
    state: Mapped[str] = mapped_column(String(32), default="pending_retry", index=True)
    next_retry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class MemoryEntry(Base):
    """One typed, scoped memory (Phase 20, Prompt §5).

    Scope is stored as four nullable columns rather than one opaque
    blob so it can be *queried* - `praxis.memory.memory_types`
    turns each into a SQL predicate, which is what makes isolation
    between users/workspaces/agents real rather than advisory.

    `superseded_by` is what makes correction auditable: updating a
    memory writes a new row and points the old one at it, so the
    previous belief stays on record instead of being overwritten.
    """

    __tablename__ = "memory_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    workspace_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    agent_name: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    key: Mapped[str] = mapped_column(String(512), index=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)
    source: Mapped[str] = mapped_column(String(32), default="agent_inferred")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    superseded_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True
    )


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


class Conversation(Base):
    """A continuing exchange between one user and Praxis.

    The unit a follow-up attaches to. Without it every request starts
    from zero, so "now group that by region" is a brand-new plan with no
    idea what "that" refers to - which is the single largest gap between
    a task runner and the conversational commander the brief describes.

    Deliberately thin: the conversation owns identity, ownership and
    lifecycle, while everything said lives in `Message` rows. That keeps
    a long exchange from rewriting one ever-growing row on every turn,
    and lets history be paged rather than loaded whole.
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    # Who owns it. A conversation is private to its creator: two users
    # in one tenant do not share each other's threads.
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(512), default="")
    # Which connector this thread is about, when the user picked one, so
    # follow-ups do not have to name it again.
    connector_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Message(Base):
    """One turn: what the user said, or what Praxis answered.

    `role` is `user` | `assistant` | `system`, matching the shape every
    chat client already expects.

    An assistant message carries more than prose. `evidence` records
    what the answer was actually derived from (which skill, which step,
    which artifact), and `limitations` records what it could not
    establish - both required by the brief, and both the difference
    between an answer a reader can check and one they must simply
    believe. They are columns rather than prose inside `content` so a
    client can render citations without parsing English.
    """

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, default=DEFAULT_TENANT_ID)
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text, default="")
    # The task this turn produced (user message) or reported on
    # (assistant message), when there was one.
    task_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    # Attachment ids the user explicitly scoped this turn to, so "these
    # invoices" means those files rather than a similarity search over
    # everything the tenant has ever uploaded.
    attachment_ids: Mapped[list] = mapped_column(JSON, default=list)
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    limitations: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True
    )
