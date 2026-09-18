# praxis/security/policy.py
"""The `PolicyEngine`: deterministic RBAC + ABAC, outside the LLM.

Two independent checks, both of which must pass (Prompt §8's "policy
engine, role/permission checks" and §11's "multi-tenancy, RBAC/ABAC,
tenant data isolation"):

1. **RBAC** - does the principal's effective permission set contain the
   required `Permission`? Effective permissions are the union of its
   roles' permissions, *intersected* with the API key's `scopes` when
   the key narrows them (a key can only ever remove permissions).
2. **ABAC** - does the resource being touched belong to the principal's
   own tenant? A cross-tenant reference is never a 404-by-accident; it
   is an explicit, audited `TenantIsolationError`.

Both produce a `PolicyDecision` rather than a bare bool, because every
decision - allow *and* deny - is written to the audit log with its
reason (`praxis.security.audit`).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from praxis.security.principal import Principal


class Permission(str, Enum):
    """Every distinct capability the API and orchestrator gate on.

    A `str` Enum so a permission round-trips through JSON (API key
    `scopes`, role definitions, audit detail) without conversion.
    """

    # Tasks / orchestration
    TASK_CREATE = "task:create"
    TASK_READ = "task:read"
    TASK_APPROVE = "task:approve"
    TASK_CANCEL = "task:cancel"
    # Attachments / knowledge
    ATTACHMENT_UPLOAD = "attachment:upload"
    ATTACHMENT_READ = "attachment:read"
    ATTACHMENT_DELETE = "attachment:delete"
    # Connectors
    CONNECTOR_READ = "connector:read"
    CONNECTOR_WRITE = "connector:write"
    # Skills / capability authoring
    SKILL_READ = "skill:read"
    SKILL_SYNTHESIZE = "skill:synthesize"
    SKILL_APPROVE = "skill:approve"
    # Dashboards / analytics
    DASHBOARD_READ = "dashboard:read"
    DASHBOARD_WRITE = "dashboard:write"
    # Schedules
    SCHEDULE_READ = "schedule:read"
    SCHEDULE_WRITE = "schedule:write"
    # Administration
    TENANT_ADMIN = "tenant:admin"
    AUDIT_READ = "audit:read"


class Role(str, Enum):
    """The built-in roles. Deliberately few and coarse - a deployment
    needing finer grain narrows an API key's `scopes` rather than
    inventing a role per endpoint."""

    SYSTEM = "system"
    ADMIN = "admin"
    OPERATOR = "operator"
    ANALYST = "analyst"
    VIEWER = "viewer"


# The one reason string used for *every* "you cannot have this
# resource" outcome - another tenant owns it, or it does not exist at
# all. Shared as a constant rather than written twice because the whole
# point is that the two are indistinguishable: a reader of their own
# tenant's audit trail must not be able to tell which happened, and two
# separately-maintained strings would eventually diverge and leak it.
#
# `praxis.api.dependencies.authorize_resource` uses this for the
# not-found case; `PolicyEngine.check` uses it for the cross-tenant one.
UNAVAILABLE_REASON = "resource is not available to this principal"

_ALL_PERMISSIONS: frozenset[Permission] = frozenset(Permission)

_VIEWER_PERMISSIONS: frozenset[Permission] = frozenset(
    {
        Permission.TASK_READ,
        Permission.ATTACHMENT_READ,
        Permission.CONNECTOR_READ,
        Permission.SKILL_READ,
        Permission.DASHBOARD_READ,
        Permission.SCHEDULE_READ,
    }
)

_ANALYST_PERMISSIONS: frozenset[Permission] = _VIEWER_PERMISSIONS | {
    Permission.TASK_CREATE,
    Permission.TASK_CANCEL,
    Permission.ATTACHMENT_UPLOAD,
    Permission.DASHBOARD_WRITE,
    Permission.SKILL_SYNTHESIZE,
    Permission.SCHEDULE_WRITE,
}

# An operator can additionally approve a paused mutating action and let
# a connector write - the two genuinely privileged runtime powers -
# but still cannot administer the tenant or approve new capabilities.
_OPERATOR_PERMISSIONS: frozenset[Permission] = _ANALYST_PERMISSIONS | {
    Permission.TASK_APPROVE,
    Permission.CONNECTOR_WRITE,
    Permission.ATTACHMENT_DELETE,
}

ROLE_PERMISSIONS: dict[str, frozenset[Permission]] = {
    Role.SYSTEM.value: _ALL_PERMISSIONS,
    Role.ADMIN.value: _ALL_PERMISSIONS,
    Role.OPERATOR.value: _OPERATOR_PERMISSIONS,
    Role.ANALYST.value: _ANALYST_PERMISSIONS,
    Role.VIEWER.value: _VIEWER_PERMISSIONS,
}


class PermissionDeniedError(PermissionError):
    """RBAC denial: the principal's effective permissions lack `permission`.

    A `PermissionError` subclass so any existing `except PermissionError`
    (e.g. `Connector.write`'s read-only guard) keeps behaving, while
    callers that care can catch this specific type and surface a 403
    with the real missing permission named.
    """

    def __init__(self, message: str, *, permission: str, principal: str) -> None:
        super().__init__(message)
        self.permission = permission
        self.principal = principal


class TenantIsolationError(PermissionError):
    """ABAC denial: the resource belongs to a different tenant.

    Deliberately distinct from `PermissionDeniedError` - "you may not do
    this at all" and "this is not yours" are different facts, and the
    audit log records which one occurred. The API surface maps this to
    404 (never leaking another tenant's resource existence) while the
    audit row records the real reason.
    """

    def __init__(self, message: str, *, principal_tenant: str, resource_tenant: str) -> None:
        super().__init__(message)
        self.principal_tenant = principal_tenant
        self.resource_tenant = resource_tenant


@dataclass(frozen=True)
class PolicyDecision:
    """The outcome of one authorization check - always produced, always
    auditable, never reduced to a bare bool at the call site."""

    allowed: bool
    permission: str
    principal: str
    tenant_id: str
    resource_type: str = ""
    resource_id: str | None = None
    reason: str = ""

    @property
    def decision(self) -> str:
        return "allowed" if self.allowed else "denied"


class PolicyEngine:
    """Stateless, deterministic authorization. No I/O, no LLM, no config
    lookups at decision time - so a decision is trivially testable and
    can never fail open because a dependency was unreachable."""

    @staticmethod
    def effective_permissions(principal: Principal) -> frozenset[Permission]:
        """Role-derived permissions, narrowed by the API key's `scopes`.

        An unknown role contributes nothing (rather than raising) - a
        role removed from `ROLE_PERMISSIONS` in a later version must
        degrade to *fewer* permissions, never to an error that a caller
        might mistakenly treat as a transient failure and retry past.
        """
        granted: set[Permission] = set()
        for role in principal.roles:
            granted |= ROLE_PERMISSIONS.get(role, frozenset())

        if principal.scopes is None:
            return frozenset(granted)

        scoped: set[Permission] = set()
        for scope in principal.scopes:
            try:
                scoped.add(Permission(scope))
            except ValueError:
                # An unrecognized scope string narrows nothing and grants
                # nothing - it simply doesn't match any permission.
                continue
        return frozenset(granted & scoped)

    def check(
        self,
        principal: Principal,
        permission: Permission,
        *,
        resource_type: str = "",
        resource_id: str | None = None,
        resource_tenant_id: str | None = None,
    ) -> PolicyDecision:
        """Evaluates RBAC + ABAC and returns the decision (never raises)."""
        if resource_tenant_id is not None and resource_tenant_id != principal.tenant_id:
            return PolicyDecision(
                allowed=False,
                permission=permission.value,
                principal=principal.describe(),
                tenant_id=principal.tenant_id,
                resource_type=resource_type,
                resource_id=resource_id,
                # Deliberately does NOT name the owning tenant, and is
                # deliberately the same string used when the resource
                # does not exist at all (see `UNAVAILABLE_REASON`).
                #
                # This reason is audited, and `GET /admin/audit` serves a
                # tenant its own rows - so naming the owner here handed
                # an admin who probes ids exactly the existence oracle
                # the 404 response exists to deny, plus the other
                # tenant's id. "Not yours" is the whole decision; whose
                # it is, and whether it exists, are not the asker's
                # business.
                #
                # `TenantIsolationError` below still carries both ids as
                # attributes for a platform operator debugging in
                # process; they just do not reach a tenant-readable row.
                reason=UNAVAILABLE_REASON,
            )

        if permission not in self.effective_permissions(principal):
            scope_note = (
                f"narrowed by key scopes {list(principal.scopes)} "
                if principal.scopes is not None
                else ""
            )
            return PolicyDecision(
                allowed=False,
                permission=permission.value,
                principal=principal.describe(),
                tenant_id=principal.tenant_id,
                resource_type=resource_type,
                resource_id=resource_id,
                reason=(
                    f"principal roles {list(principal.roles)} "
                    f"{scope_note}"
                    f"do not grant '{permission.value}'"
                ),
            )

        return PolicyDecision(
            allowed=True,
            permission=permission.value,
            principal=principal.describe(),
            tenant_id=principal.tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            reason="granted",
        )

    def authorize(
        self,
        principal: Principal,
        permission: Permission,
        *,
        resource_type: str = "",
        resource_id: str | None = None,
        resource_tenant_id: str | None = None,
    ) -> PolicyDecision:
        """`check()`, but raises the specific typed error on denial.

        Raises `TenantIsolationError` for a cross-tenant resource and
        `PermissionDeniedError` for a missing permission - two different
        facts, never collapsed into one generic error (spec §12).
        """
        decision = self.check(
            principal,
            permission,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_tenant_id=resource_tenant_id,
        )
        if decision.allowed:
            return decision

        if resource_tenant_id is not None and resource_tenant_id != principal.tenant_id:
            raise TenantIsolationError(
                decision.reason,
                principal_tenant=principal.tenant_id,
                resource_tenant=resource_tenant_id,
            )
        raise PermissionDeniedError(
            decision.reason,
            permission=permission.value,
            principal=principal.describe(),
        )


# One shared, stateless instance - constructing a new engine per call
# would allocate for no reason; it holds nothing.
policy_engine = PolicyEngine()
