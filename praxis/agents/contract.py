# praxis/agents/contract.py
"""The typed task contract for delegation (Prompt §1).

The prompt is specific about what delegation must carry: *"Represent
delegation as a typed task contract: task ID, parent ID, objective,
bounded input/artifact references, expected output schema, authorized
tools, deadline, budget, and acceptance criteria. Specialist outputs
must include results, evidence, limitations, errors, and artifact
references."*

Before this, a `PlanStep` carried only `skill_name`, `args`, and
`depends_on` - enough to execute one tool call, nowhere near enough to
delegate work to a semi-autonomous specialist and then judge whether
what came back is acceptable.

The two halves here are deliberately asymmetric:

- `TaskContract` is what the **supervisor promises and constrains** -
  it is authored before the work starts and never mutated by the
  specialist. `authorized_tools` in particular is an authorization
  input, not a hint: `praxis.agents.delegation` enforces it, so a
  specialist cannot reach a tool its contract did not grant.
- `SpecialistResult` is what the **specialist reports back** - and it
  has required slots for `limitations` and `errors` precisely so that
  "it worked" and "it worked, with these caveats" are structurally
  different answers. A specialist that hit a wall must say so in a
  field, not bury it in prose that the supervisor then has to
  re-interpret.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from praxis.agents.budget import Budget


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class AcceptanceCriterion:
    """One checkable condition the result must satisfy.

    `check` names a machine-evaluable rule
    (`praxis.agents.critic.CRITERION_CHECKS`) where one exists, so a
    criterion is verified deterministically rather than by asking a
    model whether it feels satisfied. `description` is the
    human/LLM-readable statement used for the criteria a machine
    cannot check (e.g. "the diagnosis cites the metric it is based
    on"), which the critic agent evaluates.
    """

    description: str
    check: str | None = None
    expected: Any = None

    @property
    def is_machine_checkable(self) -> bool:
        return self.check is not None


@dataclass(frozen=True)
class ArtifactRef:
    """A pointer to a stored artifact, never its bytes.

    Delegation passes references rather than payloads (the prompt's
    "bounded input/artifact references"): inlining a rendered chart or
    a parsed document into a task contract would blow out context and
    duplicate storage for no benefit.
    """

    key: str
    mime_type: str = "application/octet-stream"
    description: str = ""


@dataclass
class TaskContract:
    """What a supervisor hands a specialist.

    `deadline` is absolute rather than a duration so it survives being
    persisted and resumed - a relative "30s from now" would silently
    restart its clock on every checkpoint restore, which is exactly the
    bug that makes deadlines useless across restarts.
    """

    objective: str
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    tenant_id: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    output_schema: dict[str, str] = field(default_factory=dict)
    authorized_tools: tuple[str, ...] = ()
    acceptance_criteria: list[AcceptanceCriterion] = field(default_factory=list)
    budget: Budget | None = None
    deadline: datetime | None = None
    depth: int = 0

    @classmethod
    def child_of(
        cls,
        parent: TaskContract,
        *,
        objective: str,
        authorized_tools: tuple[str, ...] | None = None,
        inputs: dict[str, Any] | None = None,
        output_schema: dict[str, str] | None = None,
        acceptance_criteria: list[AcceptanceCriterion] | None = None,
        budget: Budget | None = None,
    ) -> TaskContract:
        """Derives a child contract that can never exceed its parent.

        Three inheritance rules, each closing a real escalation route:

        - **Tools narrow, never widen.** A child's `authorized_tools`
          is intersected with the parent's, so a subagent cannot grant
          its own child a tool it was not itself given.
        - **The deadline is inherited and never extended.** A child may
          finish sooner; it may not outlive the work it serves.
        - **Depth increments.** `praxis.agents.delegation` bounds it,
          which is what stops unbounded recursive delegation.
        """
        requested_tools = (
            tuple(authorized_tools) if authorized_tools is not None else parent.authorized_tools
        )
        if parent.authorized_tools:
            granted = tuple(
                tool for tool in requested_tools if tool in parent.authorized_tools
            )
        else:
            granted = requested_tools

        return cls(
            objective=objective,
            parent_id=parent.task_id,
            tenant_id=parent.tenant_id,
            inputs=dict(inputs or {}),
            artifacts=list(parent.artifacts),
            output_schema=dict(output_schema or {}),
            authorized_tools=granted,
            acceptance_criteria=list(acceptance_criteria or []),
            budget=budget if budget is not None else parent.budget,
            deadline=parent.deadline,
            depth=parent.depth + 1,
        )

    def with_deadline_in(self, seconds: float) -> TaskContract:
        """Sets an absolute deadline `seconds` from now."""
        self.deadline = _utcnow() + timedelta(seconds=seconds)
        return self

    @property
    def is_expired(self) -> bool:
        if self.deadline is None:
            return False
        deadline = self.deadline
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=UTC)
        return deadline <= _utcnow()

    def authorizes(self, tool_name: str) -> bool:
        """Whether this contract permits `tool_name`.

        An EMPTY `authorized_tools` means unrestricted - the historical
        behavior for every plan that predates contracts. This is a
        deliberate compatibility choice, and the reason
        `praxis.agents.delegation.enforce_authorized_tools` exists as a
        separate explicit call rather than being implied: a caller that
        wants restriction states it, rather than restriction appearing
        by accident when a field is left unset.
        """
        if not self.authorized_tools:
            return True
        return tool_name in self.authorized_tools

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form, for checkpointing and audit detail."""
        return {
            "task_id": self.task_id,
            "parent_id": self.parent_id,
            "tenant_id": self.tenant_id,
            "objective": self.objective,
            "inputs": self.inputs,
            "artifacts": [
                {"key": a.key, "mime_type": a.mime_type, "description": a.description}
                for a in self.artifacts
            ],
            "output_schema": self.output_schema,
            "authorized_tools": list(self.authorized_tools),
            "acceptance_criteria": [
                {"description": c.description, "check": c.check, "expected": c.expected}
                for c in self.acceptance_criteria
            ],
            "budget": self.budget.to_dict() if self.budget is not None else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "depth": self.depth,
        }


@dataclass
class SpecialistResult:
    """What a specialist reports back.

    `limitations` and `errors` are required slots rather than optional
    prose so that partial success is structurally visible: a supervisor
    aggregating several specialists can tell "complete", "complete but
    caveated", and "partially failed" apart without parsing sentences.
    """

    task_id: str
    succeeded: bool
    results: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    spend: dict[str, Any] = field(default_factory=dict)

    @property
    def is_partial(self) -> bool:
        """Succeeded, but with caveats worth surfacing upward."""
        return self.succeeded and bool(self.limitations or self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "succeeded": self.succeeded,
            "partial": self.is_partial,
            "results": self.results,
            "evidence": self.evidence,
            "limitations": self.limitations,
            "errors": self.errors,
            "artifacts": [
                {"key": a.key, "mime_type": a.mime_type, "description": a.description}
                for a in self.artifacts
            ],
            "spend": self.spend,
        }
