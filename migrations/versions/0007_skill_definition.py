"""Store a capability's own source on its catalogue row

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-18

A declarative procedure submitted through `POST /skills` has no file
behind it, so its SKILL.md has to live on the row or it is lost. It is
also what approval re-compiles from, which is what binds the published
capability to exactly the document that was reviewed.

Inspection-guarded, like every revision since 0003: this project
supports both Alembic and `Base.metadata.create_all` as bootstrap
paths.
"""
from alembic import op
import sqlalchemy as sa

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def _has_column(table: str, column: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return False
    return column in {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    if not _has_column("skills", "definition"):
        op.add_column(
            "skills",
            sa.Column("definition", sa.Text(), nullable=False, server_default=""),
        )


def downgrade() -> None:
    if _has_column("skills", "definition"):
        op.drop_column("skills", "definition")
