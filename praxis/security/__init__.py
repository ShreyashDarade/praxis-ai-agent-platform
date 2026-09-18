# praxis/security/__init__.py
"""Identity, authorization, audit, approval, and redaction (Prompt §8, §9, §11).

Every module here is deliberately **deterministic, non-LLM policy code**
(Prompt §1: "Use deterministic policy checks outside the LLM"). No
module in this package ever calls a model, and nothing an LLM produces
can widen a permission, forge a principal, or approve an action:

- `principal.py` - who is acting (`Principal`: tenant + user + roles +
  optional key-narrowed scopes).
- `policy.py` - the `PolicyEngine`: RBAC (role -> permissions) plus ABAC
  (the resource's `tenant_id` must match the principal's) producing a
  `PolicyDecision` that is always logged.
- `api_key.py` - generating/verifying API keys; only hashes are stored.
- `redaction.py` - PII/secret detection + redaction, applied to
  everything written to logs, traces, and the audit trail.
- `approval.py` - identity-bound, expiring, idempotent, argument-bound
  approval records (Prompt §8), re-verified when a paused task resumes.
- `audit.py` - the append-only `AuditLog` writer.
"""
from praxis.security.principal import (
    Principal,
    SYSTEM_PRINCIPAL,
    system_principal_for_tenant,
)
from praxis.security.policy import (
    Permission,
    PolicyDecision,
    PolicyEngine,
    Role,
    PermissionDeniedError,
    TenantIsolationError,
)

__all__ = [
    "Principal",
    "SYSTEM_PRINCIPAL",
    "system_principal_for_tenant",
    "Permission",
    "PolicyDecision",
    "PolicyEngine",
    "Role",
    "PermissionDeniedError",
    "TenantIsolationError",
]
