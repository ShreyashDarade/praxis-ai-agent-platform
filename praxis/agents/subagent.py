# praxis/agents/subagent.py
"""Specialist sub-agents and their registry (Prompt §1, §2).

Praxis previously had exactly one level of abstraction below the
Orchestrator: flat `Skill` objects. That is enough to call a tool, but
not to model what the prompt actually asks for - *"dynamically selected
specialist sub-agents"* with *"per-agent scoped context, permissions,
budgets, tools, and memory"*, each accepting a typed task contract and
returning results with evidence and limitations.

**A sub-agent is not a bigger skill.** The distinction that matters:

- A `Skill` is one deterministic tool call. Its contract is
  `inputs -> outputs`. It has no budget, no tool selection, no
  judgement.
- A `SubAgent` owns an *objective*. It receives a `TaskContract`,
  decides which of its authorized tools to use and in what order,
  spends against a budget, and returns a `SpecialistResult` that
  reports evidence and limitations alongside the answer.

The registry mirrors the self-registering pattern used everywhere else
in this codebase (`skill_registry`, `parser registry`, connector
factories): one new file with a `register_agent(...)` call at import
time adds a specialist, with zero edits to orchestration code - which
is the prompt's central "write once" architectural requirement.
"""
from __future__ import annotations

import abc
import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any

from praxis.agents.budget import Budget, BudgetTracker
from praxis.agents.contract import SpecialistResult, TaskContract


@dataclass(frozen=True)
class AgentManifest:
    """Declared metadata for one specialist (Prompt §7's manifest fields).

    Deliberately declarative and inspectable *without* instantiating or
    running the agent: the supervisor routes on this, and a routing
    decision must not require executing candidate agents to discover
    what they can do.
    """

    name: str
    description: str
    version: str = "1.0.0"
    owner: str = ""
    # Which registered skills this specialist may call. Becomes the
    # contract's `authorized_tools` when the supervisor delegates,
    # and is enforced - not advisory.
    tools: tuple[str, ...] = ()
    required_permissions: tuple[str, ...] = ()
    supported_connectors: tuple[str, ...] = ()
    # The purpose string handed to `LLMCatalogue` when this agent needs
    # a model, so model choice stays a catalogue concern.
    model_purpose: str = "planning"
    default_budget: Budget | None = None
    # Free-form routing hints the supervisor matches an intent against.
    keywords: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "owner": self.owner,
            "tools": list(self.tools),
            "required_permissions": list(self.required_permissions),
            "supported_connectors": list(self.supported_connectors),
            "model_purpose": self.model_purpose,
            "default_budget": self.default_budget.to_dict() if self.default_budget else None,
            "keywords": list(self.keywords),
        }


@dataclass
class AgentContext:
    """Everything a specialist is given for one execution.

    Scoped per the prompt's "per-agent scoped context, permissions,
    budgets, tools, and memory": the specialist sees only the tools its
    contract authorized, spends only against `budget`, and reads/writes
    only its own `memory` slice - it is handed these rather than
    reaching for globals, which is what makes the scoping real rather
    than nominal.
    """

    contract: TaskContract
    budget: BudgetTracker
    tools: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    tenant_id: str = ""

    def tool(self, name: str) -> Any:
        """Fetches an authorized tool by name.

        Raises `PermissionError` for a tool this contract did not
        authorize - the enforcement point for `authorized_tools`. A
        specialist cannot reach past its grant even by name.
        """
        if not self.contract.authorizes(name):
            raise PermissionError(
                f"task '{self.contract.task_id}' is not authorized to use tool '{name}'; "
                f"authorized: {list(self.contract.authorized_tools)}"
            )
        try:
            return self.tools[name]
        except KeyError:
            raise KeyError(f"tool '{name}' is not available to this agent") from None


class SubAgent(abc.ABC):
    """A specialist that fulfils a `TaskContract`."""

    manifest: AgentManifest

    @abc.abstractmethod
    async def execute(self, context: AgentContext) -> SpecialistResult:
        """Does the work and reports back.

        Implementations must return a `SpecialistResult` even on
        failure (with `succeeded=False` and populated `errors`) rather
        than raising for ordinary, expected failure - a specialist that
        raises gives its supervisor nothing to aggregate. Genuinely
        exceptional conditions (a budget exhausted, an unauthorized
        tool) still raise, because those are contract violations rather
        than work outcomes.
        """

    @property
    def name(self) -> str:
        return self.manifest.name


# --------------------------------------------------------------------- #
# Registry - same self-registering pattern as every other extension
# point in this codebase.
# --------------------------------------------------------------------- #

_AGENTS: dict[str, SubAgent] = {}


def register_agent(agent: SubAgent) -> None:
    """Registers `agent` by its manifest name.

    A duplicate name raises rather than overwriting: two specialists
    answering to one name is a programming error, and silently
    preferring whichever imported last would make routing
    nondeterministic.
    """
    name = agent.manifest.name
    if name in _AGENTS:
        raise ValueError(f"an agent named '{name}' is already registered")
    _AGENTS[name] = agent


def get_agent(name: str) -> SubAgent:
    try:
        return _AGENTS[name]
    except KeyError:
        raise KeyError(f"no agent named '{name}' is registered") from None


def all_agents() -> list[SubAgent]:
    return list(_AGENTS.values())


def all_manifests() -> list[AgentManifest]:
    return [agent.manifest for agent in _AGENTS.values()]


_DISCOVERED = False


def discover_agents() -> None:
    """Imports every module in `praxis.agents.specialists`, triggering
    each one's own `register_agent(...)` call.

    Idempotent, matching `discover_skills()`/`discover_parsers()`.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    try:
        import praxis.agents.specialists as specialists_pkg
    except ModuleNotFoundError:  # pragma: no cover - package always ships
        _DISCOVERED = True
        return

    for module_info in pkgutil.iter_modules(specialists_pkg.__path__):
        if module_info.ispkg:
            continue
        importlib.import_module(f"praxis.agents.specialists.{module_info.name}")
    _DISCOVERED = True
