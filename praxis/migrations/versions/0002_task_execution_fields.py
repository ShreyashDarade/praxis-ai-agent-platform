"""task execution fields (checklist, pending_input, result)

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-16

Adds the three columns Phase 5's Orchestrator needs on `tasks`
(`praxis/memory/models.py`'s `Task` model docstring has the full
shape/lifecycle):
- `checklist` (JSON, default `[]`) - the materialized plan.
- `pending_input` (JSON, nullable) - the `PendingInput` a paused task is
  waiting on, `null` otherwise.
- `result` (JSON, nullable) - the final delivered output once terminal.

No change to `status`'s column type (still `String(32)`) - only its
value set grows (see the model docstring); a plain string keeps this a
non-breaking, additive migration.
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column("checklist", sa.JSON, nullable=False, server_default="[]"),
    )
    op.add_column(
        "tasks",
        sa.Column("pending_input", sa.JSON, nullable=True),
    )
    op.add_column(
        "tasks",
        sa.Column("result", sa.JSON, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tasks", "result")
    op.drop_column("tasks", "pending_input")
    op.drop_column("tasks", "checklist")
