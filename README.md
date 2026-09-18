# Praxis

An autonomous enterprise data and operations intelligence platform: it
turns a natural-language intent into a validated plan, executes that
plan against real systems under real permissions, pauses for human
approval before anything mutating, and shows its working.

It is three things that usually ship separately — an agent runtime, a
retrieval/knowledge layer, and a dashboard engine — built on one
orchestrator so a single task can read a warehouse, consult an uploaded
spreadsheet, and produce a dashboard without leaving the audit trail.

## What is actually true about this build

Two documents exist specifically so this README does not have to
oversell:

- **[`docs/coverage-matrix.md`](docs/coverage-matrix.md)** — every major
  requirement, its status (*Built* / *Partial* / *Not built*), and the
  test that is the evidence. Gaps are stated, not softened.
- **[`docs/connector-matrix.md`](docs/connector-matrix.md)** — which
  connectors are live-tested, which are protocol-tested, and which have
  never executed a query against a real server. "Extensible to any
  database" is a statement about cost, not coverage, and that document
  keeps the two apart.

Also: **[`docs/dependency-table.md`](docs/dependency-table.md)** (verified
versions, licenses, alternatives, and a known conflict stated plainly)
and **[`docs/decisions/`](docs/decisions/)** (architecture decision
records).

## Quick start

Requires Python 3.11, Docker, and a POSIX-ish shell.

```bash
# 1. Postgres (with pgvector) + Prometheus + Pushgateway
docker compose up -d

# 2. Install. Add extras only if you want them (see below).
python -m venv .venv
./.venv/bin/pip install -e ".[dev]"        # Windows: ./.venv/Scripts/pip

# 3. Configure
export PRAXIS_DATABASE_URL="postgresql+asyncpg://praxis:praxis@localhost:5433/praxis"
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. Idempotent bootstrap: validate config, apply schema, check connectivity
python -m praxis.cli init

# 5. Run
uvicorn praxis.api.main:app --reload
```

`GET /health` reports every registered component. The OpenAPI schema is
at `/docs`.

### Optional extras

The base install needs no managed service — the defaults are an
in-process cache and filesystem blob storage, both real implementations.

| Extra | Adds |
|---|---|
| `praxis[redis]` | Distributed cache |
| `praxis[s3]` | S3-compatible object storage |
| `praxis[mongodb]` | MongoDB connector |
| `praxis[neo4j]` | Neo4j connector |
| `praxis[all]` | All of the above |

A connector whose driver is not installed reports itself unavailable and
the rest still register — it is never a startup error.

## How a task runs

```
intent → plan → [approval?] → execute → validate → result + audit
```

1. **Plan.** The Planner turns intent into a DAG of steps, choosing from
   the skills the caller's execution mode and permissions actually allow
   it to see.
2. **Approve.** Any `mutating` step pauses on a real LangGraph
   `interrupt()`. The approval record binds approver, tenant, argument
   hash, expiry and an idempotency key, so it cannot be replayed against
   different arguments.
3. **Execute.** Independent steps run concurrently as LangGraph
   supersteps. Every step is checkpointed, so a pause survives a full
   process restart — not just a request.
4. **Account for it.** Structured logs with correlation ids,
   OpenTelemetry traces, token/cost metrics, and an audit row for every
   authorization decision, allow *and* deny.

Execution modes (`read_only`, `plan_only`, `dry_run`, `execute`) are
enforced twice: the Planner never sees a forbidden skill, and the
executor refuses it anyway if one appears.

## Extending it

Adding a connector or a skill is one new file and no core edits. That is
not a claim on trust —
[`tests/e2e/test_extension_proof.py`](tests/e2e/test_extension_proof.py)
writes a real new connector *and* a real new skill, finds them through
the real discovery, runs them end to end, and hashes the orchestration
modules before and after to prove none of them changed.

```python
# praxis/connectors/acme/acme_connector.py — the whole integration
register_connector_factory(
    ConnectorFactory(
        name="acme",
        is_configured=lambda s: bool(s.acme_token),
        build=lambda s: AcmeConnector(s.acme_token),
    )
)
```

The connector still needs a `Settings` field, and the image must be
rebuilt to ship the file. Those are the honest costs; the orchestration
layer is untouched.

## Exact answers vs. plausible ones

Uploaded spreadsheets are kept as **typed tables** and queried with real
SQL (DuckDB), not summarized to prose and retrieved by similarity.
Embeddings retrieve text that *resembles* an answer, which is not the
same as the correct total — so `retrieve_documents` handles "what does
this say" and `query_table` handles "what is the number".

## Safety

- **Multi-tenancy** with RBAC × ABAC. A cross-tenant reference is a 404,
  never a 403 — a 403 would confirm the resource exists.
- **SQL** is parsed with sqlglot (dialect-aware), validated read-only,
  and row-bounded. The guard never regenerates the query.
- **Synthesized skills** are written by an LLM, validated in a hardened
  Docker sandbox (no network, read-only root, non-root user, dropped
  capabilities, pid/CPU bounds) and refused at execution until a human
  approves their exact code hash. This defaults **on**.
- **Untrusted content** — web pages, uploaded documents — is wrapped and
  flagged, never silently rewritten.

## Tests

```bash
pytest                                   # everything
pytest tests/acceptance                  # the brief's nine scenarios
pytest tests/e2e                         # full walkthroughs
```

Tests run against a real Postgres, real Docker, and real parsers. Where
a live third-party server was unavailable, the connector matrix says so
rather than implying coverage.

Running suites in parallel against the same database will collide on
`create_all`/`drop_all`; use a separate database per runner
(`PRAXIS_TEST_DATABASE_URL`).

## Layout

| Path | What lives there |
|---|---|
| `praxis/core/` | Orchestrator, graph engine, checkpointing, interfaces |
| `praxis/agents/` | Planner, skills, specialists, delegation, budgets, schedules |
| `praxis/connectors/` | One package per connector, self-registering |
| `praxis/ingestion/` | Parsers, chunkers, embedders, typed tables |
| `praxis/memory/` | Postgres, pgvector, graph and blob stores |
| `praxis/security/` | Principals, policy, approvals, audit, redaction |
| `praxis/safety/` | SQL guard, output validation, untrusted content |
| `praxis/analytics/` | Dashboard spec, validation, rendering, refresh |
| `praxis/semantic/` | Metrics, dimensions, joins, grain validation |
| `migrations/` | Alembic revisions |
