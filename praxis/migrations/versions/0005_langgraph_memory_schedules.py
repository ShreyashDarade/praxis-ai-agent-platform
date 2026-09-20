"""LangGraph checkpoints, typed memory, schedules, DLQ, dashboards (Phase 14-21)

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-18

Adds:

- `tasks.plan` - the materialized plan. Durable task data rather than
  ephemeral execution state, which is why it lives on the task and not
  in the checkpoint: resuming in a fresh process needs the plan to
  rebuild the same graph, and the checkpoint then supplies how far it
  got.
- `langgraph_checkpoints` / `langgraph_writes` - LangGraph's state
  snapshots and pending superstep writes, written by
  `praxis.core.checkpoint.PraxisCheckpointSaver`. These REPLACE
  `task_checkpoints` from revision 0004, which is dropped: keeping two
  checkpoint mechanisms would be duplicated state, and the earlier one
  degraded unserializable values to `repr()` where LangGraph's
  serializer handles them properly.
- `memory_entries` - the typed memory subsystem (preference,
  workspace, agent, team, episodic).
- `schedules` / `schedule_runs` - user-defined recurring tasks.
- `dead_letter` - failed background work awaiting a human.
- `dashboards` - saved dashboard specs.

Inspection-guarded throughout, for the same reason revisions 0003 and
0004 are: this project supports both Alembic and
`Base.metadata.create_all` as bootstrap paths, so a database created
the second way already has some of these while its `alembic_version`
points at an older revision.
"""
from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001"


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _has_table(name: str) -> bool:
    return name in _inspector().get_table_names()


def _has_column(table: str, column: str) -> bool:
    inspector = _inspector()
    if table not in inspector.get_table_names():
        return False
    return column in {col["name"] for col in inspector.get_columns(table)}


def upgrade() -> None:
    if not _has_column("tasks", "plan"):
        op.add_column(
            "tasks", sa.Column("plan", sa.JSON, nullable=False, server_default="[]")
        )

    if not _has_table("langgraph_checkpoints"):
        op.create_table(
            "langgraph_checkpoints",
            sa.Column("seq", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "tenant_id", sa.String(36), nullable=False, server_default=DEFAULT_TENANT_ID
            ),
            sa.Column("thread_id", sa.String(128), nullable=False),
            sa.Column("checkpoint_ns", sa.String(256), nullable=False, server_default=""),
            sa.Column("checkpoint_id", sa.String(128), nullable=False),
            sa.Column("parent_checkpoint_id", sa.String(128), nullable=True),
            sa.Column("checkpoint", sa.LargeBinary, nullable=False),
            sa.Column("checkpoint_type", sa.String(64), nullable=False, server_default="json"),
            sa.Column("checkpoint_metadata", sa.LargeBinary, nullable=False),
            sa.Column("metadata_type", sa.String(64), nullable=False, server_default="json"),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )
        op.create_index(
            "ix_langgraph_checkpoints_thread", "langgraph_checkpoints", ["thread_id"]
        )
        op.create_index(
            "ix_langgraph_checkpoints_cid", "langgraph_checkpoints", ["checkpoint_id"]
        )
        op.create_index(
            "ix_langgraph_checkpoints_tenant", "langgraph_checkpoints", ["tenant_id"]
        )

    if not _has_table("langgraph_writes"):
        op.create_table(
            "langgraph_writes",
            sa.Column("seq", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "tenant_id", sa.String(36), nullable=False, server_default=DEFAULT_TENANT_ID
            ),
            sa.Column("thread_id", sa.String(128), nullable=False),
            sa.Column("checkpoint_ns", sa.String(256), nullable=False, server_default=""),
            sa.Column("checkpoint_id", sa.String(128), nullable=False),
            sa.Column("task_id", sa.String(128), nullable=False),
            sa.Column("task_path", sa.String(256), nullable=False, server_default=""),
            sa.Column("idx", sa.Integer, nullable=False, server_default="0"),
            sa.Column("channel", sa.String(256), nullable=False),
            sa.Column("value", sa.LargeBinary, nullable=False),
            sa.Column("value_type", sa.String(64), nullable=False, server_default="json"),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )
        op.create_index("ix_langgraph_writes_thread", "langgraph_writes", ["thread_id"])
        op.create_index("ix_langgraph_writes_cid", "langgraph_writes", ["checkpoint_id"])
        op.create_index("ix_langgraph_writes_tenant", "langgraph_writes", ["tenant_id"])

    # Superseded by the LangGraph tables above - see module docstring.
    if _has_table("task_checkpoints"):
        op.drop_table("task_checkpoints")

    if not _has_table("memory_entries"):
        op.create_table(
            "memory_entries",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "tenant_id", sa.String(36), nullable=False, server_default=DEFAULT_TENANT_ID
            ),
            sa.Column("user_id", sa.String(36), nullable=True),
            sa.Column("workspace_id", sa.String(128), nullable=True),
            sa.Column("agent_name", sa.String(128), nullable=True),
            sa.Column("kind", sa.String(32), nullable=False),
            sa.Column("key", sa.String(512), nullable=False),
            sa.Column("value", sa.JSON, nullable=False, server_default="{}"),
            sa.Column(
                "source", sa.String(32), nullable=False, server_default="agent_inferred"
            ),
            sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
            sa.Column("detail", sa.JSON, nullable=False, server_default="{}"),
            sa.Column("superseded_by", sa.String(36), nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )
        for column in ("tenant_id", "user_id", "workspace_id", "agent_name", "kind", "key"):
            op.create_index(f"ix_memory_entries_{column}", "memory_entries", [column])
        op.create_index("ix_memory_entries_expires_at", "memory_entries", ["expires_at"])
        op.create_index("ix_memory_entries_created_at", "memory_entries", ["created_at"])

    if not _has_table("schedules"):
        op.create_table(
            "schedules",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "tenant_id", sa.String(36), nullable=False, server_default=DEFAULT_TENANT_ID
            ),
            sa.Column("name", sa.String(256), nullable=False),
            sa.Column("intent_text", sa.Text, nullable=False),
            sa.Column("principal_user_id", sa.String(36), nullable=True),
            sa.Column("interval_seconds", sa.Integer, nullable=True),
            sa.Column("cron", sa.String(128), nullable=True),
            sa.Column("timezone_name", sa.String(64), nullable=False, server_default="UTC"),
            sa.Column("state", sa.String(32), nullable=False, server_default="active"),
            sa.Column(
                "overlap_policy", sa.String(32), nullable=False, server_default="skip"
            ),
            sa.Column(
                "missed_run_policy", sa.String(32), nullable=False, server_default="run_once"
            ),
            sa.Column("connector_name", sa.String(256), nullable=True),
            sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )
        op.create_index("ix_schedules_tenant_id", "schedules", ["tenant_id"])
        op.create_index("ix_schedules_state", "schedules", ["state"])
        op.create_index("ix_schedules_next_run_at", "schedules", ["next_run_at"])

    if not _has_table("schedule_runs"):
        op.create_table(
            "schedule_runs",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "tenant_id", sa.String(36), nullable=False, server_default=DEFAULT_TENANT_ID
            ),
            sa.Column("schedule_id", sa.String(36), nullable=False),
            sa.Column("idempotency_key", sa.String(64), nullable=False, unique=True),
            sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("task_id", sa.String(36), nullable=True),
            sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
            sa.Column("detail", sa.Text, nullable=False, server_default=""),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )
        op.create_index("ix_schedule_runs_tenant_id", "schedule_runs", ["tenant_id"])
        op.create_index("ix_schedule_runs_schedule_id", "schedule_runs", ["schedule_id"])
        op.create_index(
            "ix_schedule_runs_idempotency", "schedule_runs", ["idempotency_key"]
        )

    if not _has_table("dead_letter"):
        op.create_table(
            "dead_letter",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "tenant_id", sa.String(36), nullable=False, server_default=DEFAULT_TENANT_ID
            ),
            sa.Column("job_type", sa.String(128), nullable=False),
            sa.Column("job_key", sa.String(256), nullable=False),
            sa.Column("payload", sa.JSON, nullable=False, server_default="{}"),
            sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
            sa.Column("failures", sa.JSON, nullable=False, server_default="[]"),
            sa.Column(
                "state", sa.String(32), nullable=False, server_default="pending_retry"
            ),
            sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )
        op.create_index("ix_dead_letter_tenant_id", "dead_letter", ["tenant_id"])
        op.create_index("ix_dead_letter_job_type", "dead_letter", ["job_type"])
        op.create_index("ix_dead_letter_job_key", "dead_letter", ["job_key"])
        op.create_index("ix_dead_letter_state", "dead_letter", ["state"])
        op.create_index("ix_dead_letter_next_retry", "dead_letter", ["next_retry_at"])

    if not _has_table("dashboards"):
        op.create_table(
            "dashboards",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "tenant_id", sa.String(36), nullable=False, server_default=DEFAULT_TENANT_ID
            ),
            sa.Column("title", sa.String(512), nullable=False),
            sa.Column("description", sa.Text, nullable=False, server_default=""),
            sa.Column("owner_user_id", sa.String(36), nullable=True),
            sa.Column("spec_version", sa.Integer, nullable=False, server_default="1"),
            sa.Column("spec", sa.JSON, nullable=False, server_default="{}"),
            sa.Column("refresh_interval_seconds", sa.Integer, nullable=True),
            sa.Column("last_refreshed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )
        op.create_index("ix_dashboards_tenant_id", "dashboards", ["tenant_id"])


def downgrade() -> None:
    for table in (
        "dashboards",
        "dead_letter",
        "schedule_runs",
        "schedules",
        "memory_entries",
        "langgraph_writes",
        "langgraph_checkpoints",
    ):
        if _has_table(table):
            op.drop_table(table)
    if _has_column("tasks", "plan"):
        op.drop_column("tasks", "plan")

