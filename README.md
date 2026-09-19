# Praxis

An autonomous enterprise data and operations intelligence platform. It
turns a natural-language intent into a validated plan, executes that
plan against real systems under real permissions, pauses for human
approval before anything mutating, and — when it gets something wrong —
diagnoses why and corrects itself rather than reporting a failure.

Three things that usually ship separately — an agent runtime, a
retrieval/knowledge layer, and a dashboard engine — on one orchestrator,
so a single task can query a warehouse, consult an uploaded spreadsheet,
and produce a dashboard without leaving the audit trail.

---

## Verified against a running server, not just a test suite

Every claim below was produced by driving the real HTTP API of a running
instance — real Postgres, real Docker, real MCP server, real third-party
endpoints, real LLM provider. Nothing here is asserted from a unit test
with mocks.

| | Result |
|---|---|
| Live subsystem sweep (35 checks over HTTP) | **35 / 35** |
| Live agent-harness verification (10 checks) | **10 / 10** |
| Unit + integration suite | **1438 passed, 0 failed** |
| LLM provider driving the sweep | OpenAI `gpt-5` (planning, synthesis, answering) |

### The distinction that made the sweep worth running

A subsystem can import cleanly, pass its unit tests, and still be
unreachable by any user. Those are different claims, and only the second
one is a product. The sweep reports them separately and **never counts an
in-process demonstration as an API pass**:

```
REACH=yes   a caller can drive this through the HTTP API
REACH=no    it works, but only in-process — reported, never scored
```

Applying that rule to this codebase found **five subsystems with complete
implementations, full unit tests, and zero API surface**: specialists,
delegation, the critic, budgets, and the agentic loop. They were not
broken. They were unreachable — which, from a user's position, is the
same thing. All five are reachable now (`/agents`, `delegate_to_specialist`,
`/investigations`), and the sweep proves it over HTTP.

### What the live environment actually contained

```bash
PRAXIS_DATABASE_URL=postgresql+asyncpg://…/praxis_live   # real Postgres + pgvector
PRAXIS_AUTH_ENABLED=true                                 # real API keys, RBAC enforced
PRAXIS_REQUIRE_SKILL_APPROVAL=true                       # synthesis gate ON
DOCKER_HOST=tcp://localhost:2375                         # real sandbox for synthesis

# Real connectors — no stubs, no recorded fixtures
PRAXIS_PROMETHEUS_URL=http://localhost:9090
PRAXIS_REST_BASE_URL=https://jsonplaceholder.typicode.com
PRAXIS_GRAPHQL_URL=https://countries.trevorblades.com/graphql
PRAXIS_MCP_SERVERS='[{"name":"everything",
                      "command":"npx",
                      "args":["-y","@modelcontextprotocol/server-everything"]}]'

# Provider selected by model id — see "Model providers" below
PRAXIS_LLM_MODEL_OVERRIDES='{"planning":"gpt-5","routing":"gpt-4.1-mini", … }'
```

The MCP connector spawns the public
[`@modelcontextprotocol/server-everything`](https://github.com/modelcontextprotocol/servers)
over stdio and round-trips a real `echo` call. Prometheus, REST and
GraphQL are live endpoints. Capability synthesis runs generated code in a
real hardened Docker container.

### What the sweep covers

All 35 checks run over HTTP against the live server:

| Area | Verified live |
|---|---|
| **agents / subagents** | 5 specialists catalogued; a plan delegates to two of them in one run; concurrent sibling delegations; unknown specialist fails loudly |
| **delegation** | depth, fan-out (refused at exactly 20 children) and recursive-objective limits enforced |
| **critic** | every delegation reviewed; a specialist reporting success whose result fails its criteria is recorded as a failure |
| **budget** | per-delegation token/cost accounting; `max_llm_calls=1` halts an investigation with `budget_exhausted` |
| **agentic loop** | answers from its own findings, cites the iterations it rests on, each action carries a hypothesis |
| **connectors** | 6 healthy; real MCP round-trip; Prometheus query |
| **ingestion** | spreadsheet → typed table; text → embeddings |
| **LLM** | real planning + **exact** arithmetic (a total the model cannot guess) |
| **memory** | episodic recall over HTTP, scoped per principal, bad kind rejected |
| **semantic** | layer round-trips; flags a non-additive rollup; refuses a malformed layer |
| **procedure** | declarative `SKILL.md` loaded and a new capability added by API |
| **safety / sandbox** | `plan_only` executes nothing; unapproved synthesized code never runs |
| **security** | 401 unauthenticated, 403 cross-role, cross-tenant is 404 |
| **observability** | audit rows for every decision; `/health` aggregates every component |

### Defects that only live execution found

Each of these passed unit tests and failed against a real server:

- **`gpt-5` returned an empty string.** Reasoning models spend tokens
  *thinking* against the same allowance as the reply, so the old default
  cap of 1024 was consumed before any visible output. It surfaced two
  layers away as `Planner response was not valid JSON … raw response: ''`,
  blaming the parser for the cap. Output is now uncapped everywhere.
- **Conversation turns ran as `SYSTEM_PRINCIPAL`** — a privilege
  escalation invisible to every in-process test.
- **A budget that counted nothing.** An investigation reported `0` LLM
  calls after making two, because the policy built its own catalogue; a
  `max_llm_calls` ceiling could not have stopped anything.
- **The evidence ledger was overwritten on resume**, silently discarding
  the attempts recorded before a task paused for approval.
- **A new skill diverted the planner** off the synthesis path onto a
  connector that did not exist.

---

## The agent harness

The orchestrator already retried a failed plan with the failure text as
context. That is necessary and not sufficient: fed prose about what went
wrong, a planner can emit a plan that *reads* differently and *is*
identical — the same skill, the same arguments, re-worded — and the loop
spends another model call learning nothing.

Praxis now runs every task under an **evidence ledger** and a
**completion gate**.

### Findings are claims with a verdict

Each step becomes a claim whose verdict is decided by what the step
*did*, never by model prose:

| Verdict | Meaning |
|---|---|
| `confirmed` | it ran and produced a result (the output is the evidence) |
| `refuted` | it failed (the real error is the evidence) |
| `unresolved` | it never ran, because a dependency failed first |

The third state matters. A step starved by an upstream failure has been
tested by *nothing* — calling it refuted would steer the next plan away
from an approach that may be exactly right.

### A retry must differ, structurally

Plans are fingerprinted on the skills they call and their arguments,
normalised for whitespace and case. A retry matching an already-refuted
attempt is **refused before it runs**. Its justification is a *computed
diff* of the two plans, not a statement from the model — a model asked
what it changed can answer anything; a diff cannot.

### "Every step ran" is not "the question was answered"

The completion gate judges a finished, error-free run before it is called
complete:

1. **Machine checks, always** — any step error; no output at all; a step
   whose own output reports `succeeded: false` (exactly how a rejected
   delegation looks from outside, and previously reported as success).
2. **A model judgement, opt-in** — a reviewer that did not produce the
   result decides whether it answers the intent.

A refuted gate is a failure the harness re-plans from. The diagnosis gets
checked, not just the patch.

### A blocker, not a stack of symptoms

When the harness stops — stalled, attempts exhausted — the task carries a
blocker naming **what was tried, what each attempt established, and what
was never reached**, alongside the full ledger on `GET /tasks/{id}`.

A real run, from the live verification:

```
attempt 0  failed   [query_table, select_revenue_column…, query_table]
  confirmed  step 0  query_table — ran, returned the schema
  refuted    step 1  synthesized skill quarantined pending approval
  unresolved step 2  never ran: depends on step 1, which failed first

attempt 1  completed  [query_table]
  justification: step 0 (query_table): changed argument(s) ['sql'];
                 dropped step(s) ['select_revenue_column…', 'query_table']
  confirmed  step 0  → total_revenue: 5771.9

verification: model / accepted — "includes a row with total_revenue: 5771.9"
stop_cause: goal_reached          blocker: none
```

The first attempt over-decomposed a simple sum and invented a capability;
synthesis correctly quarantined it; the second attempt — shown the
verdicts, not just the error text — collapsed to a single query and got
the exact number.

---

## Quick start

Requires Python 3.11, Docker, and a POSIX-ish shell.

```bash
# 1. Postgres (with pgvector) + Prometheus + Pushgateway
docker compose up -d

# 2. Install
python -m venv .venv
./.venv/bin/pip install -e ".[dev]"        # Windows: ./.venv/Scripts/pip

# 3. Configure
export PRAXIS_DATABASE_URL="postgresql+asyncpg://praxis:praxis@localhost:5433/praxis"
export ANTHROPIC_API_KEY="sk-ant-..."      # or OPENAI_API_KEY — see below

# 4. Idempotent bootstrap: validate config, apply schema, check connectivity
python -m praxis.cli init

# 5. Run
uvicorn praxis.api.main:app --reload
```

`GET /health` reports every registered component; OpenAPI is at `/docs`.

### Model providers

The provider is derived from the **model id**, so switching is a config
change and nothing can disagree about who serves a request:

| Model id prefix | Provider |
|---|---|
| `claude-*` | Anthropic |
| `gpt-*`, `o1`, `o3`, `o4` | OpenAI |

```bash
PRAXIS_LLM_MODEL_OVERRIDES='{"planning":"gpt-5","code_synthesis":"gpt-5"}'
```

Output tokens are **uncapped by default**. On reasoning models a cap
covers thinking as well as the reply, so a fixed ceiling silently
truncates valid work to an empty string. Anthropic requires a number, so
its real per-model ceiling is discovered from the API once and cached;
cost is bounded by `praxis.agents.budget`, not by truncation.

### Optional extras

The base install needs no managed service — the defaults are an
in-process cache and filesystem blob storage, both real.

| Extra | Adds |
|---|---|
| `praxis[redis]` | Distributed cache |
| `praxis[s3]` | S3-compatible object storage |
| `praxis[mongodb]` | MongoDB connector |
| `praxis[neo4j]` | Neo4j connector |
| `praxis[all]` | All of the above |

A connector whose driver is missing reports itself unavailable and the
rest still register — never a startup error.

---

## How a task runs

```
intent → plan → [approval?] → execute → verify → result + ledger + audit
```

1. **Plan.** The Planner turns intent into a DAG, choosing only from
   skills the caller's execution mode and permissions allow it to see.
   It is also shown what previous attempts established, and what this
   deployment learned from earlier runs of the same intent.
2. **Approve.** Any `mutating` step pauses on a real LangGraph
   `interrupt()`. The approval record binds approver, tenant, argument
   hash, expiry and an idempotency key, so it cannot be replayed against
   different arguments.
3. **Execute.** Independent steps run concurrently as LangGraph
   supersteps, each checkpointed — a pause survives a process restart.
4. **Verify.** The completion gate decides whether the result answers the
   intent. A refutation re-plans.
5. **Account for it.** Correlation-id logs, OpenTelemetry traces,
   token/cost metrics, and an audit row for every authorization
   decision — allow *and* deny.

Execution modes (`plan_only`, `read_only`, `dry_run`, `execute`) are
enforced twice: the Planner never sees a forbidden skill, and the
executor refuses it anyway if one appears.

### Two execution models

The brief asks for deterministic workflows for high-risk operations and
agentic ones for ambiguous research. Both are reachable:

- **`POST /intent`** — plan-then-execute. A DAG decided up front, with
  approval gates. Right when the steps are knowable in advance.
- **`POST /investigations`** — an observe/decide/act loop for questions
  that cannot be planned ahead ("which region earned most?" requires
  looking at the schema first). Bounded by iterations, stall detection
  and budget; read-only by construction; returns its full trace.

---

## Exact answers vs. plausible ones

Uploaded spreadsheets are kept as **typed tables** and queried with real
SQL (DuckDB), not summarised to prose and retrieved by similarity.
Embeddings retrieve text that *resembles* an answer, which is not the
same as the correct total — so `retrieve_documents` answers "what does
this say" and `query_table` answers "what is the number".

## Safety

- **Multi-tenancy** with RBAC × ABAC. A cross-tenant reference is a 404,
  never a 403 — a 403 would confirm the resource exists.
- **SQL** is parsed with sqlglot (dialect-aware), validated read-only and
  row-bounded. The guard never regenerates the query.
- **Synthesized skills** are written by an LLM, validated in a hardened
  Docker sandbox (no network, read-only root, non-root user, dropped
  capabilities, pid/CPU bounds), and refused at execution until a human
  approves their exact code hash. Defaults **on**.
- **Delegation cannot launder a mutating call.** `delegate_to_specialist`
  declares itself read-only and enforces it against the live registry, so
  a mutating action must be planned as its own step and meet the approval
  gate.
- **Untrusted content** — web pages, uploaded documents — is wrapped and
  flagged, never silently rewritten.

## Extending it

A connector or skill is one new file and no core edits:

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

The connector still needs a `Settings` field and the image must be
rebuilt to ship the file. Those are the honest costs; the orchestration
layer is untouched.

## Layout

| Path | What lives there |
|---|---|
| `praxis/core/` | Orchestrator, graph engine, checkpointing, evidence ledger, completion gate |
| `praxis/agents/` | Planner, skills, specialists, delegation, critic, budgets, agentic loop, schedules |
| `praxis/connectors/` | One package per connector, self-registering |
| `praxis/ingestion/` | Parsers, chunkers, embedders, typed tables |
| `praxis/memory/` | Postgres, pgvector, graph and blob stores |
| `praxis/security/` | Principals, policy, approvals, audit, redaction |
| `praxis/safety/` | SQL guard, output validation, untrusted content |
| `praxis/analytics/` | Dashboard spec, validation, rendering, refresh |
| `praxis/semantic/` | Metrics, dimensions, joins, grain validation |
| `praxis/llm/` | Provider routing, catalogue, prompts |
| `praxis/migrations/` | Alembic revisions, shipped as package data |

> The test suite, developer docs and demo seeders live outside the
> published tree (`tests/`, `docs/`, `demo/` are gitignored in this
> repository), which is why this README states results rather than
> linking to them.
