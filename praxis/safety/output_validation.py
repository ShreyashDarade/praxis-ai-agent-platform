# praxis/safety/output_validation.py
"""Skill output validation (Prompt §8's "output validation").

A skill declares `outputs: dict[str, str]` - its contract with the rest
of the plan. Nothing previously checked that `run()` actually honoured
it, which matters most for *synthesized* skills, where the
implementation is LLM-authored and the Planner has already committed
downstream steps to reading specific keys (`"$0.rows"`).

The check is deliberately a **contract** check, not a type system:

- Every declared output key must be present in the returned dict.
- Extra keys are allowed and pass through untouched. A skill returning
  more than it promised is not a failure, and rejecting it would break
  legitimate evolution.
- A non-dict return is accepted only when the skill declares exactly
  one output - the common "just return the value" shape - and is then
  normalized into `{the_one_key: value}` so downstream `$n.key`
  resolution works uniformly. A non-dict return from a skill declaring
  several outputs is a genuine contract violation.

Validation failures raise rather than being silently repaired: spec
§12's "never wrap failure as if it were success" applies exactly here -
a skill that didn't return what it promised has failed, and the step
should fail with that stated plainly.
"""
from __future__ import annotations

from typing import Any


class OutputValidationError(Exception):
    """A skill's return value did not satisfy its declared `outputs`.

    Carries the skill name, what was expected, and what was actually
    returned, so the resulting step failure names the real mismatch
    rather than a generic "invalid output".
    """

    def __init__(
        self,
        message: str,
        *,
        skill_name: str,
        expected: list[str],
        actual: list[str] | str,
    ) -> None:
        super().__init__(message)
        self.skill_name = skill_name
        self.expected = expected
        self.actual = actual


def validate_skill_output(
    skill_name: str, declared_outputs: dict[str, str], result: Any
) -> Any:
    """Validates `result` against `declared_outputs`; returns the
    (possibly normalized) value to record for this step.

    A skill declaring no outputs at all is unconstrained - some skills
    genuinely exist for their side effect - so its result passes
    through untouched.
    """
    expected = list(declared_outputs or {})
    if not expected:
        return result

    if not isinstance(result, dict):
        if len(expected) == 1:
            # The "just return the value" shape - normalize it so
            # downstream `$n.<key>` resolution works uniformly.
            return {expected[0]: result}
        raise OutputValidationError(
            (
                f"skill '{skill_name}' declares {len(expected)} outputs {expected} but returned "
                f"a bare {type(result).__name__}; a dict is required"
            ),
            skill_name=skill_name,
            expected=expected,
            actual=type(result).__name__,
        )

    missing = [key for key in expected if key not in result]
    if missing:
        raise OutputValidationError(
            (
                f"skill '{skill_name}' declared output(s) {expected} but its result is missing "
                f"{missing}; returned keys: {sorted(result)}"
            ),
            skill_name=skill_name,
            expected=expected,
            actual=sorted(result),
        )

    return result
