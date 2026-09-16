# praxis/core/exceptions.py
"""Phase 6 exception types (spec §12).

`SynthesisValidationError` is the one new type this phase adds: the
Capability Factory's bounded retry (2 extra attempts, 3 total) feeds
the sandbox's own failure output back into the next synthesis attempt
(see `praxis.agents.capability_factory.CapabilityFactory.synthesize`);
when every attempt is exhausted, this is raised - carrying the final
attempt's generated code and the sandbox's real error detail - so the
caller (the Orchestrator) can surface a clear, specific failure rather
than a generic exception string. Per spec §12/§19/§20: "a bad synthesis
is never silently registered" - this is what makes that true; nothing
upstream of this raise ever wrote a skill file, imported it, or touched
the `skills` catalogue.

No other exception type is added here: `ConnectorError`,
`SandboxViolationError`, and `ApprovalTimeoutError` (also named in spec
§12) belong to connector-call retry and approval-timeout concerns this
phase doesn't touch - adding unused exception classes "for completeness"
would be exactly the kind of speculative code this project avoids.
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
