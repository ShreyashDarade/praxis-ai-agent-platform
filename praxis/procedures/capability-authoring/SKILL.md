---
name: capability-authoring
description: >-
  Propose a new executable capability when no existing tool, MCP tool, or
  composition can do the job - generate it, validate it in an isolated
  sandbox, run policy checks, and submit it for human approval before it
  can ever execute.
version: "1.0.0"
owner: platform-extensions@example.com
risk: read_only
keywords: [synthesis, capability, skill, tool, authoring, approval]
supported_connectors: []
required_permissions:
  - skill:synthesize
dependencies: []
model_requirements:
  purpose: code_synthesis
  structured_output: false
execution_budget:
  max_llm_calls: 3
  max_wall_clock_seconds: 300
inputs:
  need_description: What the missing capability must do, including exact argument names
  connector: Optional - a connector the capability must reach
outputs:
  skill_name: Name of the proposed skill
  status: pending_approval, or the reason synthesis failed
  code_hash: SHA-256 of the exact proposed implementation
test_cases:
  - name: pure computation
    given: A need describable with stdlib only
    expect: Sandbox self-test passes, skill catalogued as pending_approval
  - name: unvalidatable code
    given: A need whose generated code fails its own self-test
    expect: Synthesis exhausts its retries and registers nothing at all
  - name: privilege escalation attempt
    given: A proposal declaring a permission the requester lacks
    expect: Policy check fails, nothing is recorded as proposable
---

## Reach for this LAST

There are four ways to get a capability, and this is the most expensive
and highest-risk of them. Exhaust the others first:

1. **Reuse an existing skill.** Check the registry. A skill whose
   description sounds adjacent is usually the answer.
2. **Discover an approved MCP tool.** A configured MCP server may
   already expose exactly this.
3. **Compose existing tools** with a `ToolChain`. If the need is "do A,
   then feed its output to B", that is wiring, not new code — and
   wiring is validated before it runs and has nothing to review.
4. **Only then** generate new executable code.

Generating code creates something nobody has read, that must be
sandboxed, reviewed and approved. The first three create nothing.

## Procedure

1. **State the need precisely**, including the exact keyword argument
   names the caller will pass and the exact output keys a downstream
   step will read. Vagueness here is the single biggest cause of a
   synthesized skill that works but has the wrong interface.

2. **Supply the connector's real schema** if the capability touches
   data, so the generated code is written against actual tables and the
   actual SQL dialect rather than a guess.

3. **Generate and validate.** The Capability Factory writes both the
   module and a self-test, then runs them in a network-isolated,
   read-only-root, non-root, resource-bounded container. The self-test
   must build its own fixtures and assert a genuine expected answer —
   "it didn't raise" is not validation. Only the standard library is
   importable in there.

4. **Retries feed back the real error.** A failed attempt passes the
   sandbox's actual stdout/stderr into the next prompt. Three attempts,
   then it stops. An exhausted synthesis registers **nothing** — no
   file, no catalogue row, no partial state.

5. **Policy checks run before anything is proposable.** A declared
   dependency outside the allow-list means the code cannot have been
   validated in the stdlib-only sandbox. A declared permission the
   requesting principal does not itself hold is refused outright —
   synthesis must never become a privilege-escalation path.

6. **Submit for approval.** The skill is catalogued as
   `pending_approval` with its code hash, source path and test
   evidence. The Orchestrator **refuses to execute it** in this state.

7. **A human approves the exact code hash.** Approval binds to those
   precise bytes; approving version N and then running different code
   is what the hash check exists to prevent. Approval requires
   `skill:approve`, which the requester deliberately does not have.

## What a skill definition cannot conjure

A generated skill can express a new *procedure* over capabilities that
already exist. It cannot manufacture:

- a driver or native library that is not installed;
- a credential, permission, or network route it was not given;
- a model capability the catalogue does not offer.

If the need requires one of those, synthesis is the wrong answer and
will fail in the sandbox. Say so plainly rather than retrying — the
fix is an adapter, a dependency, or a configuration change made by a
person, not another generation attempt.

## After approval

Published versions are immutable. A change is a new version with its
own hash and its own approval; rollback re-activates a previously
approved version rather than regenerating anything. Revocation takes
effect at the next execution attempt.
