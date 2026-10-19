# praxis/security/principal.py
"""`Principal`: who is acting, for every authorization decision.

A `Principal` is always constructed server-side - from a verified API
key (`praxis.api.dependencies.current_principal`), from an explicit
operator action, or as the internal `SYSTEM_PRINCIPAL` used by
scheduled jobs that have no human caller. It is never built from
anything an LLM produced or from a client-supplied header other than
the API key itself (Prompt §3: "bind each call to authenticated
tenant/user context rather than LLM-supplied identity").
"""
from __future__ import annotations

from dataclasses import dataclass

from praxis.memory.models import DEFAULT_TENANT_ID


@dataclass(frozen=True)
class Principal:
    """An authenticated actor, scoped to exactly one tenant.

    `roles` drives RBAC (`praxis.security.policy.ROLE_PERMISSIONS`).

    `scopes` is an optional *narrowing* set attached to the specific API
    key used for this call: when present, the effective permission set
    is the intersection of the role-derived permissions and `scopes`,
    never the union (see `PolicyEngine.effective_permissions`). This is
    what makes a read-only CI key genuinely read-only even when it
    belongs to an admin user.

    `is_system` marks the internal principal used by scheduled jobs and
    bootstrap paths; it is never reachable from an inbound HTTP request.
    """

    tenant_id: str
    user_id: str | None = None
    email: str = ""
    roles: tuple[str, ...] = ()
    scopes: tuple[str, ...] | None = None
    api_key_id: str | None = None
    is_system: bool = False

    def with_scopes(self, scopes: tuple[str, ...] | None) -> Principal:
        """Returns a copy carrying `scopes` - used when a request's API
        key narrows what that call may do."""
        return Principal(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            email=self.email,
            roles=self.roles,
            scopes=scopes,
            api_key_id=self.api_key_id,
            is_system=self.is_system,
        )

    def describe(self) -> str:
        """A short, log-safe identity string (never includes a key)."""
        if self.is_system:
            return f"system@{self.tenant_id}"
        return f"{self.email or self.user_id or 'anonymous'}@{self.tenant_id}"


# The internal principal for work with no human caller: the scheduled
# health scan, the approval-timeout sweep, CLI bootstrap, and the
# auth-disabled single-operator deployment mode. It carries the
# `system` role, which `praxis.security.policy` grants every permission
# - deliberately explicit rather than a magic "skip all checks" branch,
# so even system actions flow through the same `PolicyEngine` and land
# in the same audit log.
SYSTEM_PRINCIPAL = Principal(
    tenant_id=DEFAULT_TENANT_ID,
    user_id=None,
    email="system@praxis",
    roles=("system",),
    is_system=True,
)


def system_principal_for_tenant(tenant_id: str) -> Principal:
    """The system principal, scoped to a specific tenant - used by
    scheduled/background work that must act inside one tenant rather
    than the default one."""
    return Principal(
        tenant_id=tenant_id,
        user_id=None,
        email="system@praxis",
        roles=("system",),
        is_system=True,
    )
