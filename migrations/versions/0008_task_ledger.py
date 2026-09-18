"""Persist a task's evidence ledger

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-19

`praxis.core.harness.Ledger` records every attempt a task made, each
step as a claim with a verdict and its evidence, the computed
justification for each retry, and the blocker when the harness gave
up. That record is durable task data in the same sense the plan is:
it is what a person reads to understand why a task stopped, and what
a retry across a restart must not lose. It lives on the row rather
than in the LangGraph checkpoint for the same reason `plan` does -
checkpoints carry execution state and are pruned; this is history.

Inspection-guarded, like every revision since 0003: this project
supports both Alembic and `Base.metadata.create_all` as bootstrap
paths.
"""
from alembic import op
import sqlalchemy as sa

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def _has_column(table: str, column: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return False
    return column in {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    if not _has_column("tasks", "ledger"):
        op.add_column("tasks", sa.Column("ledger", sa.JSON(), nullable=True))


def downgrade() -> None:
    if _has_column("tasks", "ledger"):
        op.drop_column("tasks", "ledger")
