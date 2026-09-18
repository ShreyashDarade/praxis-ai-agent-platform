"""Conversations and messages (the chat surface)

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-18

Adds the two tables a continuing exchange needs, plus the link from a
task back to the turn that produced it:

- `conversations` - identity, ownership and lifecycle of one thread.
- `messages` - what was said, by whom, and for an assistant turn the
  evidence it rests on and the limitations it admits to. Those are
  columns rather than prose inside `content` so a client can render
  citations without parsing English.
- `tasks.conversation_id` - nullable, so every existing standalone
  `POST /intent` task is unaffected.

Inspection-guarded throughout, for the same reason revisions 0003-0005
are: this project supports both Alembic and `Base.metadata.create_all`
as bootstrap paths, so a database created the second way already has
these while its `alembic_version` points at an older revision.
"""
from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def _inspector():
    return sa.inspect(op.get_bind())


def _has_table(name: str) -> bool:
    return name in _inspector().get_table_names()


def _has_column(table: str, column: str) -> bool:
    if not _has_table(table):
        return False
    return column in {c["name"] for c in _inspector().get_columns(table)}


def upgrade() -> None:
    if not _has_table("conversations"):
        op.create_table(
            "conversations",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("tenant_id", sa.String(36), nullable=False, index=True),
            sa.Column("user_id", sa.String(36), nullable=True, index=True),
            sa.Column("title", sa.String(512), nullable=False, server_default=""),
            sa.Column("connector_name", sa.String(256), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )

    if not _has_table("messages"):
        op.create_table(
            "messages",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("tenant_id", sa.String(36), nullable=False, index=True),
            sa.Column("conversation_id", sa.String(36), nullable=False, index=True),
            sa.Column("role", sa.String(16), nullable=False),
            sa.Column("content", sa.Text(), nullable=False, server_default=""),
            sa.Column("task_id", sa.String(36), nullable=True, index=True),
            sa.Column("attachment_ids", sa.JSON(), nullable=False),
            sa.Column("evidence", sa.JSON(), nullable=False),
            sa.Column("limitations", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
        )

    if not _has_column("tasks", "conversation_id"):
        op.add_column("tasks", sa.Column("conversation_id", sa.String(36), nullable=True))
        op.create_index("ix_tasks_conversation_id", "tasks", ["conversation_id"])

    if not _has_column("tasks", "connector_name"):
        op.add_column("tasks", sa.Column("connector_name", sa.String(256), nullable=True))


def downgrade() -> None:
    if _has_column("tasks", "connector_name"):
        op.drop_column("tasks", "connector_name")
    if _has_column("tasks", "conversation_id"):
        op.drop_index("ix_tasks_conversation_id", table_name="tasks")
        op.drop_column("tasks", "conversation_id")
    if _has_table("messages"):
        op.drop_table("messages")
    if _has_table("conversations"):
        op.drop_table("conversations")
