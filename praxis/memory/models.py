"""SQLAlchemy models for Praxis's relational + vector + graph store (spec §2, §9)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


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
    correlation_id: Mapped[str] = mapped_column(String(36), default=_uuid)
    intent_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    checklist: Mapped[list] = mapped_column(JSON, default=list)
    pending_input: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Attachment(Base):
    __tablename__ = "attachments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    mime_type: Mapped[str] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(512))
    size_bytes: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SkillRecord(Base):
    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    risk: Mapped[str] = mapped_column(String(16))  # "read_only" | "mutating"
    synthesized: Mapped[bool] = mapped_column(Boolean, default=False)
    inputs_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    outputs_schema: Mapped[dict] = mapped_column(JSON, default=dict)
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
    doc_id: Mapped[str] = mapped_column(String(512), index=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))  # all-MiniLM-L6-v2 dimension
    chunk_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class GraphEdge(Base):
    __tablename__ = "graph_edges"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    source: Mapped[str] = mapped_column(String(512), index=True)
    relation: Mapped[str] = mapped_column(String(128))
    target: Mapped[str] = mapped_column(String(512), index=True)
    edge_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
