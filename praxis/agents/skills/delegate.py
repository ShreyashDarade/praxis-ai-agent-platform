# praxis/agents/skills/delegate.py
"""`delegate_to_specialist`: the supervisor's hand-off to a subagent.

Praxis had a complete specialist stack - manifests, task contracts,
delegation limits, per-agent budgets, a critic - and nothing that
called it. Every one of those modules imported and passed its unit
tests while being unreachable from an actual request, which is a
different thing from working.

This skill is the missing edge. Making delegation *a skill* rather
than a new orchestrator branch is deliberate: the planner already
chooses among skills, so a specialist becomes something a plan can
reach for the same way it reaches for `query_connector`, and every
control already wrapped around skill execution - the approval gate,
the risk tiering, the result cache, the audit trail - applies to a
delegation without being reimplemented.

Four things are enforced here rather than assumed:

- **Tools narrow, never widen.** The child contract is built with
  `TaskContract.child_of`, whose intersection rule means a specialist
  can only ever be handed tools its parent already held.
- **The declared risk is true.** This skill says `read_only`, so it
  refuses to delegate to any specialist whose manifest names a
  mutating tool. Without that check the declaration would be a claim
  rather than a fact, and delegation would be a way to launder a
  mutating call past the approval gate.
- **Structural limits are consulted.** Depth, fan-out and recursive
  objectives go through the task's own `DelegationRegistry`, shared
  across sibling steps so the limits count the whole tree rather than
  one step at a time.
- **The result is reviewed before it is believed.** A specialist's
  own `succeeded=True` is not taken at face value; the `Critic`
  re-checks it against the contract's acceptance criteria, and a
  rejected result fails the step. That failure text is what the
  orchestrator's re-plan loop then reads.

The registry is keyed by task id at module scope rather than carried
on `_TaskContext`, because a `Skill` is handed only its arguments and
threading a mutable orchestrator object through every skill's kwargs
to serve one skill would leak supervisor state into all of them.
`release_delegation_state` is called when a task settles.
"""
from __future__ import annotations

import uuid
from typing import Any

import structlog

from praxis.agents.budget import BudgetTracker
from praxis.agents.contract import AcceptanceCriterion, TaskContract
from praxis.agents.critic import Critic
from praxis.agents.delegation import DelegationRegistry
from praxis.agents.skill import Skill
from praxis.agents.skill_registry import discover_skills, get_skill, register_skill
from praxis.agents.subagent import AgentContext, all_manifests, discover_agents, get_agent
from praxis.config import Settings
from praxis.connectors.bootstrap import build_registry
from praxis.llm.catalogue import LLMCatalogue
from praxis.llm.prompt_manager import PromptManager
from praxis.semantic.loader import layer_from_dict

_logger = structlog.get_logger(__name__)

# One delegation tree per parent task. See the module docstring for why
# this lives here rather than on the orchestrator's task context.
_REGISTRIES: dict[str, DelegationRegistry] = {}
_ROOTS: dict[str, TaskContract] = {}


def _tree_for(task_id: str, tenant_id: str, objective: str) -> tuple[DelegationRegistry, TaskContract]:
    """The registry and root contract for `task_id`, created on first use.

    `setdefault` rather than a lock: two sibling steps delegating at
    the same time are two coroutines on one event loop, and there is
    no await between the check and the insert.
    """
    registry = _REGISTRIES.setdefault(task_id, DelegationRegistry())
    root = _ROOTS.get(task_id)
    if root is None:
        root = TaskContract(objective=objective, task_id=task_id, tenant_id=tenant_id)
        _ROOTS[task_id] = root
        registry.register_root(root)
    return registry, root


def release_delegation_state(task_id: str) -> None:
    """Drops a settled task's delegation tree.

    Called by the orchestrator when a task reaches a terminal state.
    Without it a long-lived process would retain one node per
    delegation for the life of the process.
    """
    _REGISTRIES.pop(task_id, None)
    _ROOTS.pop(task_id, None)


def _criteria_from(payload: Any) -> list[AcceptanceCriterion]:
    """Reads acceptance criteria from a plan step's arguments.

    Accepts a bare string (a judgement criterion), or an object naming
    a machine check. An unknown `check` name is left as-is rather than
    dropped: `Critic` fails a criterion whose check it cannot find,
    and silently discarding it here would turn a typo into a criterion
    that appears to pass.
    """
    criteria: list[AcceptanceCriterion] = []
    for raw in payload or ():
        if isinstance(raw, str):
            criteria.append(AcceptanceCriterion(description=raw))
        elif isinstance(raw, dict):
            criteria.append(
                AcceptanceCriterion(
                    description=str(raw.get("description") or raw.get("check") or ""),
                    check=raw.get("check"),
                    expected=raw.get("expected"),
                )
            )
        else:
            raise ValueError(
                "each acceptance criterion must be a string or an object, got "
                f"{type(raw).__name__}"
            )
    return criteria


async def _resolve_inputs(supplied: dict[str, Any], tenant_id: str) -> dict[str, Any]:
    """Turns JSON-shaped plan arguments into the live objects the
    specialists actually read.

    Two inputs are not plain data: `connector` is a live `Connector`
    (a specialist calls `.read()` on it) and `semantic_layer` is a
    built `SemanticLayer`. A plan can only ever supply a name and a
    definition document respectively, so the translation happens here
    rather than in five specialists.
    """
    resolved = dict(supplied)

    connector = resolved.get("connector")
    if isinstance(connector, str):
        registry = build_registry(Settings())
        try:
            resolved["connector"] = registry.get(connector)
        except KeyError:
            # The registry's own KeyError names the connector that is
            # missing but not the ones that exist, which leaves a
            # re-plan guessing a second name. Listing them turns a
            # dead end into a correctable error.
            raise KeyError(
                f"no connector named '{connector}' is registered; available connectors "
                f"are {sorted(existing.name for existing in registry.all())}"
            ) from None

    layer = resolved.get("semantic_layer")
    if isinstance(layer, dict):
        resolved["semantic_layer"] = layer_from_dict(layer, tenant_id=tenant_id)

    return resolved


class DelegateToSpecialistSkill(Skill):
    name = "delegate_to_specialist"
    risk = "read_only"
    inputs = {
        "agent": (
            "name of the specialist to hand this sub-task to. One of: "
            "'sql_analyst' (validates and runs one bounded read query), "
            "'schema_analyst' (profiles a connector's tables and columns), "
            "'chart_designer' (chooses a chart type for a set of rows and renders it), "
            "'dashboard_builder' (assembles rendered panels into a saved dashboard), "
            "'metric_validator' (checks an aggregation against semantic-layer "
            "metric definitions for double-counting)"
        ),
        "objective": "one sentence stating what the specialist must accomplish",
        "inputs": (
            "object of inputs for the specialist. 'connector' may be given as a "
            "registered connector NAME and is resolved to the live connector; "
            "'semantic_layer' may be given as a metric-definition object. "
            "NOTE: a 'connector' name is resolved against the deployment's "
            "configured connector registry only. A bespoke or "
            "environment-specific database that is not in that registry cannot "
            "be reached through a "
            "specialist at all, and a purpose-built capability must be "
            "synthesized for it instead - delegating is not a way around that. "
            "sql_analyst takes {connector, query}; schema_analyst takes {connector}; "
            "chart_designer takes {rows}; metric_validator takes "
            "{rows, semantic_layer, metrics}"
        ),
        "acceptance_criteria": (
            "optional list of conditions the result must meet before it is accepted. "
            "Each is either a sentence, or an object naming a machine check: "
            "'succeeded', 'no_errors', 'has_evidence', 'non_empty_result', "
            "'min_rows' (with 'expected'), 'output_keys' (with 'expected'), "
            "'has_artifact'. NOTE: 'output_keys' names keys the specialist RETURNS "
            "(metric_validator returns valid/issues/blocking/advisory; sql_analyst "
            "returns rows/row_count/columns/query; chart_designer returns "
            "chart_type/encoding/alt_text), never the inputs you passed it - a "
            "criterion demanding an input name back will always fail."
        ),
    }
    outputs = {
        "agent": "the specialist that ran",
        "objective": "the objective this delegation was given",
        "inputs_given": "names of the inputs handed to the specialist",
        "succeeded": "whether the specialist reported success AND the critic accepted it",
        "results": "the specialist's own structured output",
        "evidence": "what the specialist did, as verifiable records",
        "limitations": "caveats the specialist attached to its own result",
        "errors": "why it failed, empty when it did not",
        "review": "the critic's verdict on each acceptance criterion",
        "spend": "tokens, cost and wall-clock this delegation consumed",
    }

    async def run(self, **kwargs: Any) -> Any:
        agent_name = str(kwargs.get("agent") or "").strip()
        objective = str(kwargs.get("objective") or "").strip()
        if not agent_name:
            raise ValueError("delegate_to_specialist requires an 'agent'")
        if not objective:
            raise ValueError("delegate_to_specialist requires an 'objective'")

        tenant_id = str(kwargs.get("tenant_id") or "")
        # Injected by the orchestrator's step runner. Absent only when a
        # skill is driven directly (a test, the CLI), where a synthetic
        # root is the honest thing to use.
        parent_task_id = str(kwargs.get("task_id") or uuid.uuid4())

        # Both registries are self-registering and idempotent. This
        # skill is reached through the orchestrator, which has already
        # discovered skills - but it is also reachable from the CLI and
        # from a test, and a specialist's tool lookup below must not
        # depend on who called first.
        discover_agents()
        discover_skills()
        try:
            agent = get_agent(agent_name)
        except KeyError:
            raise KeyError(
                f"no specialist named '{agent_name}'; registered specialists are "
                f"{sorted(m.name for m in all_manifests())}"
            ) from None

        manifest = agent.manifest

        # The declared risk of THIS skill is read_only, so a specialist
        # that could mutate must not be reachable through it. Checked
        # against the live registry rather than the manifest's word.
        for tool_name in manifest.tools:
            try:
                tool_skill = get_skill(tool_name)
            except KeyError:
                raise KeyError(
                    f"specialist '{agent_name}' declares tool '{tool_name}', which is not "
                    "registered in this deployment"
                ) from None
            if tool_skill.risk != "read_only":
                raise PermissionError(
                    f"specialist '{agent_name}' declares the mutating tool '{tool_name}'; "
                    "delegate_to_specialist is a read-only skill and will not run it. A "
                    "mutating action must be planned as its own step so it passes the "
                    "approval gate."
                )

        registry, root = _tree_for(
            parent_task_id, tenant_id, objective=f"supervise task {parent_task_id}"
        )

        child = TaskContract.child_of(
            root,
            objective=objective,
            authorized_tools=manifest.tools,
            inputs=await _resolve_inputs(kwargs.get("inputs") or {}, tenant_id),
            acceptance_criteria=_criteria_from(kwargs.get("acceptance_criteria")),
            budget=manifest.default_budget,
        )
        # Raises DelegationLimitError on depth, fan-out or a recursive
        # objective. Not caught: the step fails with that message, and
        # the re-plan loop gets a specific reason rather than a generic
        # one.
        registry.register_child(child)

        tracker = BudgetTracker(child.budget)
        context = AgentContext(
            contract=child,
            budget=tracker,
            tools={name: get_skill(name) for name in child.authorized_tools},
            tenant_id=tenant_id,
        )

        _logger.info(
            "delegation_started",
            agent=agent_name,
            parent_task_id=parent_task_id,
            child_task_id=child.task_id,
            depth=child.depth,
        )
        result = await registry.run_bounded(lambda: agent.execute(context))

        # A critic WITH a reviewer model. Constructed bare, the critic
        # marks every judgement-based criterion unverified-and-failed -
        # honest, but it meant any prose criterion a planner wrote made
        # the delegation fail unconditionally. Machine criteria never
        # reach the model, so this costs a call only when a judgement
        # was actually asked for.
        review = await Critic(LLMCatalogue(), PromptManager()).review(child, result)
        _logger.info(
            "delegation_completed",
            agent=agent_name,
            child_task_id=child.task_id,
            succeeded=result.succeeded,
            accepted=review.accepted,
        )

        # A specialist that says it succeeded but whose result the
        # critic rejects is a failure, and the step must say so -
        # reporting it as success is exactly the "wrap a failure as a
        # result" pattern the brief forbids.
        rejections = [
            f"{verdict.description}: {verdict.reason}" for verdict in review.failures
        ]
        errors = list(result.errors)
        if rejections:
            errors.append(
                "the critic rejected this result against its acceptance criteria: "
                + "; ".join(rejections)
            )

        return {
            "agent": agent_name,
            # What this delegation was asked to do, on the result itself.
            # A reviewer reading the output alone otherwise cannot tell
            # which sub-task it answers, and a trace with several
            # delegations to one specialist is unreadable without it.
            "objective": objective,
            "inputs_given": sorted(str(k) for k in (kwargs.get("inputs") or {})),
            "succeeded": bool(result.succeeded and review.accepted),
            "results": result.results,
            "evidence": result.evidence,
            "limitations": result.limitations,
            "errors": errors,
            "review": review.to_dict(),
            "spend": tracker.snapshot(),
        }


register_skill(DelegateToSpecialistSkill())
