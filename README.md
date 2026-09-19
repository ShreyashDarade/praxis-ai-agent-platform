# Praxis

An autonomous enterprise data and operations intelligence platform.

You give it a question in plain language. It plans the work, runs that
plan against your real systems under your real permissions, pauses for
your approval before anything that changes state, and hands back an
answer with the evidence behind it. If a step fails, it reads the
error, re-plans, and tries a genuinely different approach.

## What it does

- **Answers questions over your data.** Point it at a warehouse, an
  API, a metrics server or a spreadsheet you just uploaded. It works
  out which to use and how.
- **Computes exact numbers.** Uploaded spreadsheets become typed tables
  queried with real SQL, so "what is the total" returns arithmetic, not
  something that reads like an answer.
- **Builds new capabilities when none fits.** It writes the tool it
  needs, validates it in a locked-down container, and refuses to run it
  until a human approves that exact code.
- **Investigates open questions.** For asks that cannot be planned
  ahead ("which region is underperforming and why"), it runs an
  observe/decide/act loop and shows every step it took.
- **Never changes anything behind your back.** Mutating actions stop
  and wait for a person.

## How this helps you

| You want to | Praxis gives you |
|---|---|
| Ask across systems without writing glue | One intent; it picks the connectors and composes the steps |
| Trust the number | Exact SQL over typed data, plus the query and rows it used |
| Let agents act, safely | Approval gates bound to the exact arguments and code approved |
| Know what happened | Per-step evidence, an audit row per decision, full replay |
| Not babysit failures | It diagnoses the error and re-plans instead of stopping |
| Avoid lock-in | Swap Anthropic/OpenAI models with one env var; add a connector in one file |

---

## Running it

Requires Python 3.11, Docker, and a POSIX-ish shell.

### 1. Start the infrastructure

```bash
docker compose up -d          # Postgres + pgvector, Prometheus, Pushgateway
```

### 2. Install

```bash
python -m venv .venv
./.venv/bin/pip install -e ".[dev]"        # Windows: ./.venv/Scripts/pip
```

### 3. Configure

```bash
export PRAXIS_DATABASE_URL="postgresql+asyncpg://praxis:praxis@localhost:5433/praxis"
export ANTHROPIC_API_KEY="sk-ant-..."      # or OPENAI_API_KEY, see Model providers
```

### 4. Create the schema

```bash
python -m praxis.cli init                  # validates config, applies migrations, checks connectivity
```

### 5. Create a tenant, a user, and an API key

Skip this if you leave `PRAXIS_AUTH_ENABLED` off — every request then
runs as a single local operator. To exercise auth:

```bash
export PRAXIS_AUTH_ENABLED=true

python -m praxis.cli tenant-create acme --name "Acme"
python -m praxis.cli user-create you@acme.com --role admin      # prints a user id
python -m praxis.cli key-issue <user-id> --name "local"         # prints the key ONCE
```

Roles are `admin`, `operator`, `analyst`, `viewer`. Issue a second
lower-privilege key if you want to try the permission checks below.

### 6. Run the server

```bash
uvicorn praxis.api.main:app --host 127.0.0.1 --port 8077
```

```bash
export KEY="praxis_sk_..."                 # the key printed above
export API="http://127.0.0.1:8077"
```

Interactive API docs: `http://127.0.0.1:8077/docs`.

### Connecting real systems

Every connector is optional and registers only when configured. Set any
of these before starting the server:

```bash
export PRAXIS_PROMETHEUS_URL="http://localhost:9090"
export PRAXIS_REST_BASE_URL="https://jsonplaceholder.typicode.com"
export PRAXIS_GRAPHQL_URL="https://countries.trevorblades.com/graphql"
export PRAXIS_SLACK_BOT_TOKEN="xoxb-..."
export PRAXIS_GITHUB_TOKEN="ghp_..."

# Any MCP server, stdio or HTTP. This one is a public reference server:
export PRAXIS_MCP_SERVERS='[{"name":"everything","command":"npx",
  "args":["-y","@modelcontextprotocol/server-everything"]}]'
```

A connector whose driver or credential is missing reports itself
unavailable; the rest still start.

### Model providers

The provider is chosen by the **model id**, so switching is config only:

| Model id | Provider |
|---|---|
| `claude-*` | Anthropic (`ANTHROPIC_API_KEY`) |
| `gpt-*`, `o1`, `o3`, `o4` | OpenAI (`OPENAI_API_KEY`) |

```bash
export PRAXIS_LLM_MODEL_OVERRIDES='{"planning":"gpt-5","code_synthesis":"gpt-5","answering":"gpt-5"}'
```

Purposes you can point at a model: `planning`, `routing`,
`code_synthesis`, `answering`, `diagnosis`, `vision`.

### Useful settings

```bash
export PRAXIS_REQUIRE_SKILL_APPROVAL=true   # generated code needs sign-off (default on)
export PRAXIS_MAX_REPLAN_ATTEMPTS=2         # retries after a failed plan
export PRAXIS_VERIFY_COMPLETION=true        # have a model check the answer fits the question
export DOCKER_HOST=tcp://localhost:2375     # if the sandbox can't find your Docker socket
```

---

## Exercising each part on the live server

Every example below is a real call against the running server.

### Health and connectors

```bash
curl -s $API/health | jq
```

Lists every registered component and whether it is reachable.

### Ingestion — upload data

```bash
printf 'week,region,revenue\nW1,North,1200.50\nW1,South,1400.00\nW2,North,1600.40\n' > sales.csv

curl -s -X POST $API/attachments -H "X-API-Key: $KEY" \
  -F "file=@sales.csv;type=text/csv" | jq
```

Returns an `attachment_id`. CSV and XLSX become typed tables; PDFs,
DOCX and text are chunked and embedded.

### Planning and exact answers

```bash
TASK=$(curl -s -X POST $API/intent -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d "{\"text\":\"What is the total revenue in spreadsheet $ATTACHMENT? Use the exact sum.\",
       \"mode\":\"execute\"}" | jq -r .task_id)

curl -s $API/tasks/$TASK -H "X-API-Key: $KEY" | jq
```

Poll until `status` is `completed`, `failed` or `awaiting_approval`.
The response carries the answer, the per-step results, and a `ledger`
showing what each attempt tried and why.

### Execution modes

```bash
# plan only — produces the plan, runs nothing
-d '{"text":"...","mode":"plan_only"}'
# read-only — real reads, refuses any mutating step
-d '{"text":"...","mode":"read_only"}'
# dry run — reads for real, simulates mutations
-d '{"text":"...","mode":"dry_run"}'
# execute — full run, pauses for approval on mutations
-d '{"text":"...","mode":"execute"}'
```

### Approving a mutating action

Ask for something that changes state (posting to Slack, writing a
dashboard). The task reaches `awaiting_approval` with a `pending_input`
describing the exact call:

```bash
curl -s -X POST $API/tasks/$TASK/approve -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{"approved": true}' | jq
```

Send `false` to reject. The approval is bound to those arguments — it
cannot be replayed against different ones.

### Specialists and delegation

```bash
curl -s $API/agents -H "X-API-Key: $KEY" | jq            # catalogue + delegation limits
curl -s $API/agents/sql_analyst -H "X-API-Key: $KEY" | jq
```

Specialists are invoked by a plan, through the `delegate_to_specialist`
skill:

```bash
curl -s -X POST $API/intent -H "X-API-Key: $KEY" -H 'Content-Type: application/json' -d '{
  "text": "Use delegate_to_specialist to hand chart_designer the objective '\''chart revenue by week'\'' with inputs rows [{\"week\":\"W1\",\"revenue\":2600.5},{\"week\":\"W2\",\"revenue\":3171.4}], and acceptance criteria [{\"check\":\"succeeded\"},{\"check\":\"has_evidence\"}].",
  "mode": "execute"}' | jq
```

The step result includes the specialist's output, the reviewer's
verdict on each acceptance criterion, and what the delegation spent.

### The agentic loop — open-ended investigation

For questions that cannot be planned up front:

```bash
curl -s -X POST $API/investigations -H "X-API-Key: $KEY" -H 'Content-Type: application/json' -d "{
  \"objective\": \"Which region produced the most total revenue in spreadsheet $ATTACHMENT, and how much? Inspect the data before answering.\",
  \"inputs\": {\"attachment_id\": \"$ATTACHMENT\"},
  \"max_iterations\": 6,
  \"max_llm_calls\": 20}" | jq
```

Returns the answer, which iterations it rests on, every action and
observation, why it stopped, and what it spent. Lower `max_llm_calls`
to watch the budget stop it.

### Conversations

```bash
CONV=$(curl -s -X POST $API/conversations -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{"title":"analysis"}' | jq -r .id)

curl -s -X POST $API/conversations/$CONV/messages -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{"text":"What is the total revenue?"}'

curl -s $API/conversations/$CONV/messages -H "X-API-Key: $KEY" | jq
```

Context carries across turns; each turn runs as the calling user.

### Memory

```bash
curl -s "$API/memory?kind=episodic&limit=5" -H "X-API-Key: $KEY" | jq
curl -s "$API/memory/history?kind=episodic&key=episode:<intent>" -H "X-API-Key: $KEY" | jq
curl -s -X DELETE "$API/memory?kind=episodic&key=episode:<intent>" -H "X-API-Key: $KEY"
```

Run the same intent twice and watch its history grow — the planner
reads it before planning.

### Semantic layer

```bash
LAYER='{"layer":{
  "metrics":[{"name":"revenue","expression":"sum(amount)","type":"sum","grain":"order/day","source":"orders"},
             {"name":"buyers","expression":"count(distinct customer_id)","type":"count_distinct","grain":"order/day","source":"orders"}],
  "dimensions":[{"name":"week","expression":"date_trunc(week, ts)","grain":"order/week","source":"orders"}],
  "joins":[{"left":"orders","right":"line_items","type":"one_to_many","on":"orders.id = line_items.order_id"}]}}'

curl -s -X POST $API/semantic/describe -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d "$LAYER" | jq

curl -s -X POST $API/semantic/validate -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d "$(echo $LAYER | jq '. + {metrics:["buyers"], dimensions:["week"]}')" | jq
```

Catches the aggregations that are wrong but look right — summing a
distinct count, rolling up across a fan-out join.

### Adding a capability as markdown

No code, no redeploy — compose tools you already have:

```bash
curl -s -X POST $API/skills -H "X-API-Key: $KEY" -H 'Content-Type: application/json' -d '{
 "markdown": "---\nname: total_revenue\ndescription: Totals the revenue column of a spreadsheet\nrisk: read_only\nsteps:\n  - tool: query_table\n    args:\n      attachment_id: $.attachment_id\n      sql: SELECT SUM(revenue) AS total FROM data\n---\nTotals revenue.\n"}' | jq

curl -s $API/skills -H "X-API-Key: $KEY" | jq
curl -s -X POST $API/skills/total_revenue/approve -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{"version":1}' | jq
```

### Capability synthesis and the sandbox

Ask for something no existing skill covers:

```bash
curl -s -X POST $API/intent -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"text":"Compute the 11th Fibonacci number using a new capability called fib_eleven.","mode":"execute"}' | jq
```

Praxis writes the code, runs its self-test in a hardened container, and
records it as `pending_approval`. The task refuses to run it. Approve
it via `POST /skills/{name}/approve`, then re-run the intent.

### Querying a connector directly

```bash
curl -s -X POST $API/intent -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"text":"Use query_connector on the prometheus connector with query up.","mode":"execute"}' | jq

# An MCP tool: query is the tool name, params are its arguments
curl -s -X POST $API/intent -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"text":"Use query_connector on the everything connector: query is the tool name echo, params is {\"message\":\"hello\"}.","connector":"everything","mode":"execute"}' | jq
```

### Security

```bash
curl -s -o /dev/null -w "%{http_code}\n" $API/admin/users                         # 401
curl -s -o /dev/null -w "%{http_code}\n" $API/admin/users -H "X-API-Key: $ANALYST" # 403
curl -s -o /dev/null -w "%{http_code}\n" $API/tasks/<other-tenants-task> -H "X-API-Key: $KEY"  # 404
```

Another tenant's resource is a 404, never a 403 — a 403 would confirm
it exists.

### Observability

```bash
curl -s $API/admin/audit -H "X-API-Key: $KEY" | jq '.entries[:5]'
curl -s $API/health | jq
```

One audit row per authorization decision, allowed and denied. Logs
carry a correlation id per task; OpenTelemetry traces and token/cost
metrics are emitted if you point them somewhere.

### Dashboards, schedules, alerts

```bash
curl -s -X POST $API/dashboards -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"title":"Revenue","panels":[]}' | jq
curl -s -X POST $API/dashboards/<id>/refresh -H "X-API-Key: $KEY" | jq
curl -s $API/artifacts/<key> -H "X-API-Key: $KEY" --output chart.png

curl -s -X POST $API/schedules -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"name":"daily","cron":"0 9 * * *","intent_text":"Summarise yesterday'\''s revenue"}' | jq

curl -s -X POST $API/webhook/alert -H 'Content-Type: application/json' \
  -d '{"alerts":[{"labels":{"alertname":"HighLatency"}}]}' | jq
```

---

## Extending it

A connector or a skill is one new file, no core edits:

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

It still needs a `Settings` field, and the image must be rebuilt to
ship the file. The orchestration layer is untouched.

## Optional extras

```bash
pip install -e ".[redis]"     # distributed cache
pip install -e ".[s3]"        # S3-compatible blob storage
pip install -e ".[mongodb]"   # MongoDB connector
pip install -e ".[neo4j]"     # Neo4j connector
pip install -e ".[all]"
```

Defaults need no managed service: in-process cache, filesystem blobs.

## Safety

- Multi-tenant, RBAC × ABAC; cross-tenant reads are 404.
- SQL is parsed with sqlglot, validated read-only, and row-bounded.
- Generated code runs in a container with no network, read-only root,
  a non-root user, dropped capabilities and CPU/pid limits — and
  cannot execute until a human approves its exact hash.
- Delegation is read-only and enforced, so it cannot be used to slip a
  mutating call past the approval gate.
- Untrusted content (web pages, uploaded documents) is wrapped and
  flagged, never silently rewritten.

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
