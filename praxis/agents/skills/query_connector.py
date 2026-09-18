# praxis/agents/skills/query_connector.py
"""`query_connector`: a generic, hand-written, read-only skill that runs
a query against any already-registered `Connector` by name (spec §6,
§16.1's "fetch-metrics"/"fetch-logs" steps).

**Why hand-written, not synthesized (Phase 11)**: every *synthesized*
skill's sandbox self-test can only ever import the Python standard
library, with no network at all (see
`praxis.agents.capability_factory`'s own docstring) - which structurally
rules out a synthesized skill genuinely reaching a real networked
connector like Prometheus for its own self-test (there's no fixture it
could stand up in-sandbox that would prove anything about reaching a
*real*, already-running external service). A generic "call
connector.read() by name" tool sidesteps that entirely: it is
hand-written (never sandbox-validated - reviewed once, here), yet every
call it makes at run time is 100% real - the actual, already-configured
`Connector` object's actual `read()` method, including whatever real
read-only enforcement that connector itself applies (spec §6's "safety
net independent of LLM-authored code"). This is squarely in scope as
spec §16.1's own text allows: "existing skill ... or synthesizes, first
time seen" - this phase's SRE walkthrough takes the existing-skill path
for the metric-fetch step, reserving the (LLM-code-generation-dependent)
synthesis proof for the dashboard walkthrough, which needs it for a
different reason (no existing skill is even shaped right for "query an
arbitrary customer's SQL schema").

**Deliberately NOT advertised for SQL/schema-aware querying** (see
`inputs["query"]`'s own description below): this skill's real
`build_registry(Settings())` lookup never includes Phase 11's demo-only
`"customer-db"` connector at all (see the docstring paragraph below), so
a plan that tried to route the dashboard walkthrough's "chart weekly
signup counts" ask through this skill would simply fail with an unknown-
connector error - the Planner must synthesize a bespoke, schema-aware
capability for that instead (spec §16.2). The `query` input's wording
exists specifically to steer a real Planner call away from reaching for
this generic tool where a purpose-built one is actually needed.

`connector.read()` is structurally read-only on every `Connector` (the
ABC keeps mutation strictly behind the separate `write()` method - see
`praxis.core.interfaces.Connector`), so declaring this skill
`risk = "read_only"` is correct regardless of which connector name it's
pointed at.

Dependencies are constructed fresh per call, exactly like every other
hand-written skill in this package (`post_slack_message.py`,
`retrieve_documents.py`, `create_chart.py`): a fresh `Settings()` +
`build_registry(Settings())` per call, so this skill always reflects
whatever connectors this deployment currently has configured, with
nothing cached or shared across calls to go stale. Note this
deliberately does NOT see Phase 11's demo-only `"customer-db"`
connector (`praxis.api.main`'s own module-level registration, not part
of `praxis.connectors.bootstrap.build_registry`) - that connector is
reached only via the Capability Factory's connector-aware synthesis
path (spec §16.2), never via this generic tool.
"""
from __future__ import annotations

from typing import Any

from praxis.agents.skill import Skill
from praxis.agents.skill_registry import register_skill
from praxis.config import Settings
from praxis.connectors.bootstrap import build_registry


class QueryConnectorSkill(Skill):
    name = "query_connector"
    risk = "read_only"
    inputs = {
        "connector_name": "name of an already-registered connector to query (e.g. 'prometheus')",
        "query": (
            "a connector-specific query string for a metrics/observability or "
            "messaging connector - e.g. a PromQL expression for a Prometheus "
            "connector, or a channel id for Slack. NOT for a connector needing "
            "bespoke, schema-aware SQL against an arbitrary customer database - "
            "a dedicated capability should be synthesized for that instead. "
            "For an MCP connector this is the TOOL NAME, and the tool's own "
            "arguments go in `params`."
        ),
        "params": (
            "optional object of argument names to values for connectors whose "
            "query is a named call - an MCP tool's own arguments, for example "
            "{\"text\": \"hello\"}. Omit for query-string connectors."
        ),
    }
    outputs = {"result": "the connector's raw read() result for this query"}

    async def run(self, **kwargs: Any) -> Any:
        connector_name = kwargs["connector_name"]
        query = kwargs["query"]
        # Arguments for connectors whose "query" is a named call rather
        # than a query string - an MCP tool above all, where `query` is
        # the tool name and the arguments are a separate object.
        #
        # Without this an MCP tool taking any argument was unreachable
        # through this skill, so a planner asked to call one had no
        # choice but to invent a new capability and have it synthesized:
        # generating code to do something a registered tool already
        # does, which is exactly the outcome the brief's "prefer reuse"
        # ordering exists to avoid.
        params = kwargs.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(
                "'params' must be an object of argument names to values, got "
                f"{type(params).__name__}"
            )

        registry = build_registry(Settings())
        connector = registry.get(connector_name)  # KeyError is already clear/typed (spec §12)
        result = await connector.read(query, **params)
        return {"result": result}


register_skill(QueryConnectorSkill())
