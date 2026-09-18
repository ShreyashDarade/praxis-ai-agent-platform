---
name: dashboard-composition
description: >-
  Turn a natural-language dashboard request into a validated, saveable
  dashboard specification, by profiling the source, writing bounded SQL,
  validating the metric semantics, and choosing charts from the data's shape.
version: "1.0.0"
owner: analytics-platform@example.com
risk: read_only
keywords: [dashboard, chart, metric, sql, analytics]
supported_connectors: [sql, postgres, customer-db]
required_permissions:
  - connector:read
  - dashboard:write
dependencies: []
model_requirements:
  purpose: planning
  structured_output: false
execution_budget:
  max_llm_calls: 4
  max_wall_clock_seconds: 180
inputs:
  request: The user's dashboard request, in their own words
  connector: Name of the registered connector holding the data
outputs:
  dashboard_id: Identifier of the saved dashboard
  spec: The validated DashboardSpec document
test_cases:
  - name: weekly revenue line chart
    given: A sales table with week and revenue columns
    expect: A line panel encoded x=week, y=revenue, with provenance recorded
  - name: refuses un-aggregated rows
    given: Rows with duplicate values in the chosen dimension
    expect: Refusal citing double-counting, no panel produced
---

## When to use this skill

Use it when someone asks for a dashboard, chart, or recurring report over
data reachable through a registered connector.

Do **not** use it for a one-off question with a single numeric answer —
that is a query, not a dashboard, and `safe-query-analysis` answers it
more cheaply.

## Procedure

1. **Resolve the source.** Confirm the named connector exists and the
   caller holds `connector:read` for it. If the request names no
   connector and more than one is configured, ask which — do not guess.

2. **Profile the schema** with the `schema_analyst` specialist. Take from
   it: the real SQL dialect, the candidate measures, and the candidate
   time axes. The dialect it reports is authoritative — it overrides
   whatever the request's wording implies. A request saying "our Postgres"
   against a SQLite-backed connector is common and must not produce
   `date_trunc`.

3. **Resolve business meaning** before writing SQL. If a semantic layer is
   declared, map the request's words to defined metrics through the
   glossary (a request for "sales" may mean the `revenue` metric). If a
   term cannot be resolved, ask rather than inventing a definition — a
   plausible wrong metric is worse than a clarifying question.

4. **Write and validate the query** with `sql_analyst`. It parses against
   the real dialect, refuses stacked statements and hidden mutations, and
   applies a row bound. Aggregate to exactly one row per dimension value.

5. **Validate the result** with `metric_validator`. Treat its blocking
   issues as fatal: grain mismatch, fan-out double counting, and currency
   mixing all produce numbers that look right and are wrong. Report its
   advisory warnings alongside the dashboard rather than dropping them.

6. **Choose the visualization** with `chart_designer`. It decides from the
   data's shape, deterministically, so the same data always yields the
   same chart. Override it only if the user asked for a specific type.

7. **Assemble and save** with `dashboard_builder`, then `POST /dashboards`.
   Every panel must carry `alt_text` and its query provenance; a panel
   without either is rejected by validation.

## What this skill cannot do

- It cannot join across two connectors. Cross-source analysis needs an
  explicit data-movement decision that is out of scope here.
- It cannot define a new metric. It resolves against metrics someone has
  declared; inventing one silently would make two dashboards disagree
  about what "revenue" means.
- It cannot render client-side interactivity. The output is a declarative
  spec, deliberately — never generated browser JavaScript.
