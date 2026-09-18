# praxis/cache/keys.py
"""Tenant- and permission-aware cache keys, plus the authorization
recheck that has to happen on a cache *hit*.

The product brief states the requirement in one sentence: *"Cache keys
must include tenant, effective permissions, source versions, and
relevant model/prompt/tool versions. Recheck authorization on cache
hits."* This module is both halves of that.

**Why permissions belong in the key at all.** A cache in front of an
authorized read is an authorization bypass waiting to happen: the
uncached path filters what it returns by who is asking, and the cached
path returns whatever the first caller happened to see. Putting the
asker's effective permissions into the key means two callers can only
ever share an entry when the uncached path would have produced the same
answer for both.

**Why *effective* permissions and not roles.** `Principal.roles` is the
wrong granularity in both directions. Two deployments can name the same
permission set "analyst" and "data-analyst" and would then never share
an entry that is identical by construction. Worse, an API key's
`scopes` *narrows* a principal's permissions
(`PolicyEngine.effective_permissions` intersects rather than unions),
so two principals with the same role and different key scopes are not
interchangeable even though their roles match. Hashing the resolved
permission set - the exact input the `PolicyEngine` would use - is the
only composition where "same key" means "same authorization outcome".

**Why the tenant is in the key AND rechecked on read.** The key already
segregates tenants, so under correct use the recheck can never fire.
That stops being true the moment the cache is distributed: a Redis
keyspace outlives the process, is shared by every worker, and is
reachable by anything with the connection string. `authorize_cache_hit`
is the second, independent guard for that - it compares the reading
principal's tenant against the tenant recorded on the entry being
consumed, and refuses loudly rather than returning a value. A
cross-tenant hit is never a cache problem to degrade around; it is an
isolation failure, and a `TenantIsolationError` is the only honest
response.

**Backwards compatibility.** `principal` is optional everywhere. A
caller with no principal (the auth-disabled single-operator deployment,
a scheduled job before tenancy is resolved, every pre-Phase-12 call
site) gets a key in the same shape with `~` in both the tenant and
permission positions. That key can never collide with a real tenant's,
because `~` is not a legal encoded tenant id (see `_encode`), so the
untenanted and tenanted keyspaces stay disjoint without either caller
knowing about the other.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from praxis.security.policy import PolicyEngine, TenantIsolationError
from praxis.security.principal import Principal

# The single position in a key that means "no principal was supplied".
# Deliberately a character `_encode` can never emit, so an untenanted
# key and a tenant literally named "~" are different keys. Also chosen
# to have no meaning in a Redis `SCAN` glob, unlike `*` or `?`, so a
# prefix sweep cannot match it by accident.
NO_PRINCIPAL = "~"

_KEY_SEPARATOR = ":"

# Everything outside this set is replaced before a tenant id goes into
# a key: a tenant id is free-form text from a database row, and a `:`
# in it would silently re-partition the key into different fields. `-`
# is inside the set because tenant ids here are UUIDs
# (`praxis.memory.models.DEFAULT_TENANT_ID`), and forcing every one of
# them through the escaping branch below would make every key in the
# system carry a digest suffix for no benefit.
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.\-]")

# Long enough that a collision between two distinct permission sets is
# not a practical concern (there are 18 permissions, so 2^18 possible
# sets against a 64-bit digest), short enough that a key stays readable
# in a log line or a Redis `SCAN` listing.
_PERMISSIONS_DIGEST_CHARS = 16


def _encode(component: str) -> str:
    """Makes `component` safe to place between key separators.

    Substituting unsafe characters alone would map two distinct tenant
    ids onto one (`a:b` and `a.b` both to `a.b`), so whenever a
    substitution actually happened a digest of the original is appended
    - the encoding stays readable for the overwhelmingly common case of
    an already-safe id, and stays injective for the rest.
    """
    if not component:
        raise ValueError("cache key component must not be empty")
    safe = _SAFE_COMPONENT_RE.sub("_", component)
    if safe == component:
        return safe
    suffix = hashlib.sha256(component.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{suffix}"


def permissions_digest(principal: Principal | None) -> str:
    """A stable digest of `principal`'s effective permission set.

    Resolved through `PolicyEngine.effective_permissions`, so it is the
    same computation an authorization check would run - roles unioned,
    then intersected with the API key's `scopes` when the key narrows
    them. Sorted before hashing, so the digest depends on the *set* and
    not on the order roles happened to be listed in.

    Returns `NO_PRINCIPAL` for `None`. An authenticated principal with
    *zero* effective permissions (an admin holding a key scoped to
    nothing recognizable) still gets a real digest, which is distinct
    from `NO_PRINCIPAL` - "nobody asked" and "somebody who may do
    nothing asked" must not share cache entries.
    """
    if principal is None:
        return NO_PRINCIPAL
    permissions = PolicyEngine.effective_permissions(principal)
    payload = "|".join(sorted(permission.value for permission in permissions))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_PERMISSIONS_DIGEST_CHARS]


def _parts_digest(parts: dict[str, Any]) -> str:
    """A stable digest of the scope-specific parts of a key.

    `sort_keys=True` so keyword order at the call site is irrelevant,
    and `default=str` so an unserializable part degrades to its repr
    rather than raising - a cache key builder that can throw would make
    the cache a source of failures instead of a way to avoid work. The
    honest limitation: two objects with the same `str()` and different
    identity collapse to one key, so callers pass the *version string*
    of a model or prompt rather than the object itself.
    """
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheKey:
    """One cache key, kept as its four composed fields rather than as a
    finished string.

    Holding the fields is what lets a caller do the two things a flat
    string cannot: recheck the tenant on a hit (`authorize`) and sweep a
    tenant's entries by prefix (`tenant_prefix`). `str(key)` renders the
    string the `Cache` actually sees.

    `tenant_id` is the raw id, `None` when no principal was supplied;
    the rendered key holds its encoded form.
    """

    scope: str
    tenant_id: str | None
    permissions_digest: str
    parts_digest: str

    @classmethod
    def build(cls, scope: str, principal: Principal | None = None, **parts: Any) -> CacheKey:
        """Composes a key from a scope, an optional principal, and the
        scope-specific parts.

        `parts` is where the brief's "source versions, and relevant
        model/prompt/tool versions" live: this builder cannot know which
        versions matter to a given scope, so each call site names them
        explicitly (`model=`, `prompt_version=`, `source_version=`, ...).
        Anything omitted there is a real staleness bug at that call
        site, not something this function can catch - which is exactly
        why `praxis.cache.scopes` documents the invalidation rule per
        scope alongside the constant.
        """
        return cls(
            scope=scope,
            tenant_id=principal.tenant_id if principal is not None else None,
            permissions_digest=permissions_digest(principal),
            parts_digest=_parts_digest(parts),
        )

    @property
    def value(self) -> str:
        """The key string handed to `Cache.get`/`Cache.set`."""
        tenant = _encode(self.tenant_id) if self.tenant_id is not None else NO_PRINCIPAL
        return _KEY_SEPARATOR.join(
            (self.scope, tenant, self.permissions_digest, self.parts_digest)
        )

    def __str__(self) -> str:
        return self.value

    def authorize(self, principal: Principal | None) -> None:
        """Rechecks that `principal` may consume an entry stored under
        this key; see `authorize_cache_hit`."""
        authorize_cache_hit(principal, self.tenant_id)

    @staticmethod
    def tenant_prefix(scope: str, tenant_id: str) -> str:
        """The key prefix covering every entry of one scope for one
        tenant - what a distributed invalidation sweep matches on.

        This is the reason the tenant is a readable field in the key
        rather than being folded into the digest: "drop everything this
        tenant has cached for dashboards, their warehouse just
        reloaded" is a prefix scan, and would otherwise require reading
        and parsing every value in the keyspace.
        """
        return f"{scope}{_KEY_SEPARATOR}{_encode(tenant_id)}{_KEY_SEPARATOR}"


def build_key(scope: str, principal: Principal | None = None, **parts: Any) -> CacheKey:
    """Module-level shorthand for `CacheKey.build`."""
    return CacheKey.build(scope, principal, **parts)


def authorize_cache_hit(principal: Principal | None, entry_tenant_id: str | None) -> None:
    """Re-checks tenancy before a cached value is returned to a caller.

    Returns `None` when the read is permitted and raises
    `TenantIsolationError` otherwise - the same exception
    `PolicyEngine.authorize` raises for a cross-tenant resource, so a
    cache hit and an uncached read fail identically and land in the
    audit log the same way.

    The four cases, all decided by failing closed:

    - principal and entry in the same tenant: allowed.
    - principal and entry in different tenants: refused. Under correct
      use the keys differ so this is unreachable; reaching it means a
      key was crafted, reused across identities, or collided in a shared
      keyspace, and every one of those is a bug that must surface rather
      than quietly serve another tenant's data.
    - an authenticated principal reading an untenanted entry: refused.
      Nothing about such an entry records which permission set produced
      it, so it cannot be shown to be one this principal was entitled
      to see.
    - no principal reading a tenanted entry: refused, for the same
      reason from the other direction - there is no identity to check
      it against.

    Raising rather than degrading to a miss is deliberate. Treating an
    isolation failure as a miss would paper over the bug and recompute
    the value, leaving the poisoned entry in place for the next reader.
    """
    principal_tenant = principal.tenant_id if principal is not None else None
    if principal_tenant == entry_tenant_id:
        return

    raise TenantIsolationError(
        "cache hit refused: entry belongs to tenant "
        f"{entry_tenant_id!r}, reader is in tenant {principal_tenant!r}",
        principal_tenant=principal_tenant or NO_PRINCIPAL,
        resource_tenant=entry_tenant_id or NO_PRINCIPAL,
    )
