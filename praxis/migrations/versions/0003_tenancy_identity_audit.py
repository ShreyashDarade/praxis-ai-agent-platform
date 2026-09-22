"""tenancy, identity, audit, approval records (Phase 12)

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-18

Adds the identity/isolation substrate Prompt §8/§11 require:

- `tenants`, `users`, `api_keys` - who exists and who may act.
- `audit_log` - every authorization decision and mutating action.
- `approval_records` - identity-bound, expiring, idempotent,
  argument-bound human approvals.
- A `tenant_id` column on every previously-unscoped tenant-owned table
  (`tasks`, `attachments`, `skills`, `vector_chunks`, `graph_edges`).

The default tenant row is inserted *before* the `tenant_id` columns are
added, so every pre-existing row backfills into a real tenant (via the
column's `server_default`) rather than a dangling id. That ordering is
what makes this migration safe to run against a populated database.

Also widens `skills` with the Phase 16 manifest metadata and swaps its
`name` unique constraint for a `(tenant_id, name, version)` one, so a
skill can be published as an immutable version per tenant instead of a
single globally-unique row that later versions overwrite.

**Why every step is inspection-guarded.** This project supports two
legitimate ways to get a schema: Alembic (`praxis init`, production)
and `Base.metadata.create_all` (the test suite, and quick local
scratch databases). A database bootstrapped the second way already has
these tables/columns while its `alembic_version` still points at an
older revision, so a naive `create_table` here would abort with
"relation already exists" on a schema that is in fact already correct.
Guarding each step on real introspection makes this migration
convergent - it brings any starting state up to the target schema and
is safe to re-run - rather than only working from one exact prior
state.
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001"

_TENANT_SCOPED_TABLES = ("tasks", "attachments", "skills", "vector_chunks", "graph_edges")


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _has_table(name: str) -> bool:
    return name in _inspector().get_table_names()


def _has_column(table: str, column: str) -> bool:
    inspector = _inspector()
    if table not in inspector.get_table_names():
        return False
    return column in {col["name"] for col in inspector.get_columns(table)}


def _has_index(table: str, index: str) -> bool:
    inspector = _inspector()
    if table not in inspector.get_table_names():
        return False
    return index in {idx["name"] for idx in inspector.get_indexes(table)}


def _add_column_if_missing(table: str, column: sa.Column) -> None:
    if not _has_column(table, column.name):
        op.add_column(table, column)


def _create_index_if_missing(name: str, table: str, columns: list[str]) -> None:
    if _has_table(table) and not _has_index(table, name):
        op.create_index(name, table, columns)


def upgrade() -> None:
    if not _has_table("tenants"):
        _create_tenants_table()

    # Idempotent regardless of which branch above ran: a create_all
    # database has the table but not this row.
    op.execute(
        sa.text(
            "INSERT INTO tenants (id, slug, name, active, settings, created_at) "
            "VALUES (:id, 'default', 'Default Tenant', true, '{}', now()) "
            "ON CONFLICT (id) DO NOTHING"
        ).bindparams(id=DEFAULT_TENANT_ID)
    )

    if not _has_table("users"):
        _create_users_table()
    if not _has_table("api_keys"):
        _create_api_keys_table()
    if not _has_table("audit_log"):
        _create_audit_log_table()
    if not _has_table("approval_records"):
        _create_approval_records_table()

    for table in _TENANT_SCOPED_TABLES:
        _add_column_if_missing(
            table,
            sa.Column(
                "tenant_id", sa.String(36), nullable=False, server_default=DEFAULT_TENANT_ID
            ),
        )
        _create_index_if_missing(f"ix_{table}_tenant_id", table, ["tenant_id"])

    _add_column_if_missing("tasks", sa.Column("created_by_user_id", sa.String(36), nullable=True))
    _add_column_if_missing(
        "tasks", sa.Column("mode", sa.String(16), nullable=False, server_default="execute")
    )
    _add_column_if_missing(
        "attachments", sa.Column("uploaded_by_user_id", sa.String(36), nullable=True)
    )

    for column in _SKILL_MANIFEST_COLUMNS:
        _add_column_if_missing("skills", column)

    # The old `name` UNIQUE constraint must go: a versioned catalogue
    # keeps every published version, and two tenants may legitimately
    # own same-named skills.
    op.execute(sa.text("ALTER TABLE skills DROP CONSTRAINT IF EXISTS skills_name_key"))
    _create_index_if_missing("ix_skills_name", "skills", ["name"])
    _create_index_if_missing("ix_skills_status", "skills", ["status"])

    existing_constraints = {
        constraint["name"] for constraint in _inspector().get_unique_constraints("skills")
    }
    if "uq_skills_tenant_name_version" not in existing_constraints:
        op.create_unique_constraint(
            "uq_skills_tenant_name_version", "skills", ["tenant_id", "name", "version"]
        )

    _create_index_if_missing(
        "ix_vector_chunks_tenant_doc", "vector_chunks", ["tenant_id", "doc_id"]
    )
    _create_index_if_missing(
        "ix_graph_edges_tenant_source", "graph_edges", ["tenant_id", "source"]
    )


_SKILL_MANIFEST_COLUMNS = [
    sa.Column("version", sa.Integer, nullable=False, server_default="1"),
    sa.Column("status", sa.String(32), nullable=False, server_default="active"),
    sa.Column("owner", sa.String(256), nullable=False, server_default=""),
    sa.Column("description", sa.Text, nullable=False, server_default=""),
    sa.Column("required_permissions", sa.JSON, nullable=False, server_default="[]"),
    sa.Column("supported_connectors", sa.JSON, nullable=False, server_default="[]"),
    sa.Column("dependencies", sa.JSON, nullable=False, server_default="[]"),
    sa.Column("model_requirements", sa.JSON, nullable=False, server_default="{}"),
    sa.Column("execution_budget", sa.JSON, nullable=False, server_default="{}"),
    sa.Column("test_cases", sa.JSON, nullable=False, server_default="[]"),
    sa.Column("code_hash", sa.String(64), nullable=False, server_default=""),
    sa.Column("source_path", sa.String(512), nullable=False, server_default=""),
    sa.Column("approved_by_user_id", sa.String(36), nullable=True),
    sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("health_status", sa.String(32), nullable=False, server_default="unknown"),
]


def _create_tenants_table() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("slug", sa.String(128), nullable=False, unique=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("settings", sa.JSON, nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def _create_users_table() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), nullable=False, index=True),
        sa.Column("email", sa.String(320), nullable=False, index=True),
        sa.Column("display_name", sa.String(256), nullable=False, server_default=""),
        sa.Column("roles", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def _create_api_keys_table() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), nullable=False, index=True),
        sa.Column("user_id", sa.String(36), nullable=False, index=True),
        sa.Column("name", sa.String(256), nullable=False, server_default=""),
        sa.Column("key_hash", sa.String(64), nullable=False, unique=True, index=True),
        sa.Column("scopes", sa.JSON, nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def _create_audit_log_table() -> None:
    op.create_table(
        "audit_log",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), nullable=False, index=True),
        sa.Column("actor_user_id", sa.String(36), nullable=True, index=True),
        sa.Column("action", sa.String(128), nullable=False, index=True),
        sa.Column("resource_type", sa.String(64), nullable=False, server_default=""),
        sa.Column("resource_id", sa.String(256), nullable=True, index=True),
        sa.Column("decision", sa.String(32), nullable=False, server_default="allowed"),
        sa.Column("reason", sa.Text, nullable=False, server_default=""),
        sa.Column("detail", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("correlation_id", sa.String(36), nullable=True, index=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            index=True,
        ),
    )


def _create_approval_records_table() -> None:
    op.create_table(
        "approval_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), nullable=False, index=True),
        sa.Column("task_id", sa.String(36), nullable=False, index=True),
        sa.Column("step_index", sa.Integer, nullable=False, server_default="0"),
        sa.Column("action", sa.String(256), nullable=False, server_default=""),
        sa.Column("action_hash", sa.String(64), nullable=False, index=True),
        sa.Column("idempotency_key", sa.String(128), nullable=False, unique=True, index=True),
        sa.Column("approver_user_id", sa.String(36), nullable=True),
        sa.Column("approved", sa.Boolean, nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def downgrade() -> None:
    op.drop_index("ix_graph_edges_tenant_source", table_name="graph_edges")
    op.drop_index("ix_vector_chunks_tenant_doc", table_name="vector_chunks")
    op.drop_constraint("uq_skills_tenant_name_version", "skills", type_="unique")
    op.drop_index("ix_skills_status", table_name="skills")
    op.drop_index("ix_skills_name", table_name="skills")
    op.create_unique_constraint("skills_name_key", "skills", ["name"])

    for column in (
        "health_status",
        "approved_at",
        "approved_by_user_id",
        "source_path",
        "code_hash",
        "test_cases",
        "execution_budget",
        "model_requirements",
        "dependencies",
        "supported_connectors",
        "required_permissions",
        "description",
        "owner",
        "status",
        "version",
    ):
        op.drop_column("skills", column)

    op.drop_column("attachments", "uploaded_by_user_id")
    op.drop_column("tasks", "mode")
    op.drop_column("tasks", "created_by_user_id")

    for table in _TENANT_SCOPED_TABLES:
        op.drop_index(f"ix_{table}_tenant_id", table_name=table)
        op.drop_column(table, "tenant_id")

    op.drop_table("approval_records")
    op.drop_table("audit_log")
    op.drop_table("api_keys")
    op.drop_table("users")
    op.drop_table("tenants")

