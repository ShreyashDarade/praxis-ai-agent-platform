# praxis/core/exceptions.py
"""The exception taxonomy from spec §12.

`SynthesisValidationError` (Phase 6): the Capability Factory's bounded
retry (2 extra attempts, 3 total) feeds the sandbox's own failure
output back into the next synthesis attempt (see
`praxis.agents.capability_factory.CapabilityFactory.synthesize`); when
every attempt is exhausted, this is raised - carrying the final
attempt's generated code and the sandbox's real error detail - so the
caller (the Orchestrator) can surface a clear, specific failure rather
than a generic exception string. Per spec §12/§19/§20: "a bad synthesis
is never silently registered" - this is what makes that true; nothing
upstream of this raise ever wrote a skill file, imported it, or touched
the `skills` catalogue.

This phase (7) adds the three remaining types spec §12 names, each for
a real call site, not speculatively:

- `ConnectorError`: raised by `praxis.connectors.retry.call_with_retry`
  once every bounded retry attempt of a real connector call has failed
  - the "connector wrapper" spec §12 describes. Wired into the one real
  `Connector` call site that exists outside a connector's own
  implementation today: `CapabilityFactory._connector_schema`'s
  `connector.describe()` call.
- `SandboxViolationError`: raised by
  `praxis.sandbox.executor.DockerSandboxExecutor.run` when Docker itself
  reports the container was OOM-killed (exit code 137 and/or
  `OOMKilled: true` in the container's real inspect result) - a genuine
  resource-bound violation, never merely "the user's code returned
  non-zero" (that stays a normal `SandboxResult`, per spec §12's
  distinction between a failure and a violation).
- `ApprovalTimeoutError`: raised (its `str()` recorded on the `Task` row,
  not raised as a live exception, since the raiser is a scheduled sweep
  with no in-flight caller to catch it) by
  `praxis.api.main.sweep_stale_approvals` for any task left
  `awaiting_approval`/`awaiting_clarification` past
  `Settings.approval_timeout_seconds`.
"""
from __future__ import annotations


class SynthesisValidationError(Exception):
    """Raised when Capability Factory-generated code fails sandbox
    validation on every attempt (spec §12).

    `code` is the last attempt's full generated module source; `detail`
    is the sandbox's own stdout/stderr/exit-code detail (or a parsing
    error, if the model's response didn't even parse into the expected
    MODULE/SELF_TEST shape) - both kept as attributes, not just folded
    into the message string, so a caller (a test, a log line, the
    Orchestrator's failure message) can inspect exactly what was tried
    and why it was rejected.
    """

    def __init__(self, message: str, *, code: str, detail: str) -> None:
        super().__init__(f"{message}: {detail}")
        self.code = code
        self.detail = detail


class ConnectorError(Exception):
    """Raised when every bounded retry attempt of a real connector call
    has failed (spec §12: "a connector call fails; caught at the
    connector wrapper, logged, bounded retry, then surfaced as a step
    failure").

    Carries the connector's name, the operation attempted (e.g.
    `"describe"`, `"read"`), how many attempts were actually made, and
    the *last* attempt's real error detail as attributes - never
    discarded into a generic "connector failed" string - so a caller can
    inspect exactly which connector/operation failed and why.
    """

    def __init__(
        self,
        message: str,
        *,
        connector_name: str,
        operation: str,
        attempts: int,
        detail: str,
    ) -> None:
        super().__init__(f"{message}: {detail}")
        self.connector_name = connector_name
        self.operation = operation
        self.attempts = attempts
        self.detail = detail


class SandboxViolationError(Exception):
    """Raised when a sandboxed execution exceeds its resource/network
    bounds - a genuine violation of the sandbox's own limits, distinct
    from the code under test simply returning a non-zero exit code
    (spec §12: "a sandboxed execution exceeds its resource/network
    bounds; execution killed, task step marked failed").

    `violation` names which bound was exceeded (e.g. `"oom_killed"`);
    `exit_code` is the container's real reported exit code when one was
    observed (Docker reports 137 for an OOM kill); `detail` is the
    real, concrete evidence for why this was classified as a violation
    rather than an ordinary failure (e.g. the container's own
    `OOMKilled` inspect field, plus its stdout/stderr) - kept as
    attributes, not folded away, so a caller can tell a violation apart
    from a normal `SandboxResult` failure and inspect exactly what was
    detected.
    """

    def __init__(
        self, message: str, *, violation: str, exit_code: int | None, detail: str
    ) -> None:
        super().__init__(f"{message}: {detail}")
        self.violation = violation
        self.exit_code = exit_code
        self.detail = detail


class ApprovalTimeoutError(Exception):
    """Represents a mutating task left `awaiting_approval` (or
    `awaiting_clarification`) past `Settings.approval_timeout_seconds`
    (spec §12: "surfaced to the user, not silently dropped").

    Raised (its `str()` captured, since the scheduled sweep that detects
    this has no live caller awaiting the paused task to actually catch
    an exception from) by `praxis.api.main.sweep_stale_approvals`.
    `task_id` and `waited_seconds` are kept as attributes - never folded
    away - so the resulting `Task.result["error"]` string, and any log
    line recording this, both trace back to exactly which task and how
    far past its timeout it was found.
    """

    def __init__(self, message: str, *, task_id: str, waited_seconds: float, detail: str) -> None:
        super().__init__(f"{message}: {detail}")
        self.task_id = task_id
        self.waited_seconds = waited_seconds
        self.detail = detail
