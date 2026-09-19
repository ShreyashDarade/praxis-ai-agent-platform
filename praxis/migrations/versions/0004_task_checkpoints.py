"""durable task checkpoints (Phase 14)

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-18

Adds `task_checkpoints`, which is what makes a task paused on approval
survive a process restart. Before this, execution-graph state lived
only in memory, so a restart left a paused task permanently
unresumable - the database knew what it was waiting on, but the graph
needed to continue it was gone.

One row per task (`task_id` unique), upserted rather than appended:
the resumable state is the *current* state, and retaining every
intermediate version would grow without bound for no operational
benefit.

Inspection-guarded for the same reason revision 0003 is - this project
supports both Alembic and `Base.metadata.create_all` as bootstrap
paths, so a database created the second way already has this table
while its `alembic_version` still points at an older revision.
"""
from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "task_checkpoints" in inspector.get_table_names():
        return

    op.create_table(
        "task_checkpoints",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id", sa.String(36), nullable=False, server_default=DEFAULT_TENANT_ID
        ),
        sa.Column("task_id", sa.String(36), nullable=False, unique=True),
        sa.Column("state", sa.JSON, nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_task_checkpoints_task_id", "task_checkpoints", ["task_id"])
    op.create_index("ix_task_checkpoints_tenant_id", "task_checkpoints", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("ix_task_checkpoints_tenant_id", table_name="task_checkpoints")
    op.drop_index("ix_task_checkpoints_task_id", table_name="task_checkpoints")
    op.drop_table("task_checkpoints")
