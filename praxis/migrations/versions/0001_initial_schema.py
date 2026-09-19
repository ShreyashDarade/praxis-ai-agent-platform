"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-09-15

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("correlation_id", sa.String(36), nullable=False),
        sa.Column("intent_text", sa.Text, nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "attachments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("mime_type", sa.String(128), nullable=False),
        sa.Column("source", sa.String(512), nullable=False),
        sa.Column("size_bytes", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False, server_default="uploaded"),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "skills",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(256), nullable=False, unique=True),
        sa.Column("risk", sa.String(16), nullable=False),
        sa.Column("synthesized", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("inputs_schema", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("outputs_schema", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "health_history",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("component", sa.String(128), nullable=False),
        sa.Column("healthy", sa.Boolean, nullable=False),
        sa.Column("detail", sa.Text, nullable=False, server_default=""),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "vector_chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("doc_id", sa.String(512), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("chunk_metadata", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_vector_chunks_doc_id", "vector_chunks", ["doc_id"])

    op.create_table(
        "graph_edges",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("source", sa.String(512), nullable=False),
        sa.Column("relation", sa.String(128), nullable=False),
        sa.Column("target", sa.String(512), nullable=False),
        sa.Column("edge_metadata", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_graph_edges_source", "graph_edges", ["source"])
    op.create_index("ix_graph_edges_target", "graph_edges", ["target"])


def downgrade() -> None:
    op.drop_table("graph_edges")
    op.drop_table("vector_chunks")
    op.drop_table("health_history")
    op.drop_table("skills")
    op.drop_table("attachments")
    op.drop_table("tasks")
