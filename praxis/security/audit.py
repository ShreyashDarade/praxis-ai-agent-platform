# praxis/security/audit.py
"""The append-only audit trail (Prompt §9's "audit log").

Every authorization decision (allow *and* deny) and every mutating
action writes one `AuditLog` row. Two properties make this trustworthy
rather than decorative:

1. **Redacted at write time.** `detail` goes through
   `praxis.security.redaction.redact` before it is persisted, so the
   audit trail can never itself become the place a credential leaks.
2. **Never able to break the caller.** `record()` swallows and logs its
   own storage failures. An audit write failing must not turn a
   successful action into a 500, nor a denial into an allow - the
   decision has already been made by the time we get here.

Writes go through their own short-lived session so an audit row is
committed independently of whatever transaction the caller is in - a
denied action whose surrounding transaction rolls back still leaves its
audit row behind, which is the entire point.
"""
from __future__ import annotations

from typing import Any

import structlog

from praxis.memory.db import PostgresStore
from praxis.memory.models import AuditLog
from praxis.security.policy import PolicyDecision
from praxis.security.principal import Principal
from praxis.security.redaction import redact

_logger = structlog.get_logger(__name__)


class AuditLogger:
    """Writes `AuditLog` rows. Construct with a live `PostgresStore`."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    async def record(
        self,
        *,
        principal: Principal,
        action: str,
        resource_type: str = "",
        resource_id: str | None = None,
        decision: str = "allowed",
        reason: str = "",
        detail: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Appends one audit row. Never raises."""
        try:
            async with self._store.session() as session:
                session.add(
                    AuditLog(
                        tenant_id=principal.tenant_id,
                        actor_user_id=principal.user_id,
                        action=action,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        decision=decision,
                        reason=reason[:4000],
                        detail=redact(detail or {}),
                        correlation_id=correlation_id,
                    )
                )
                await session.commit()
        except Exception as exc:  # noqa: BLE001 - see module docstring
            _logger.error(
                "audit_write_failed",
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                error=str(exc),
            )

    async def record_decision(
        self,
        *,
        principal: Principal,
        decision: PolicyDecision,
        action: str | None = None,
        detail: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Convenience wrapper recording a `PolicyDecision` verbatim."""
        await self.record(
            principal=principal,
            action=action or decision.permission,
            resource_type=decision.resource_type,
            resource_id=decision.resource_id,
            decision=decision.decision,
            reason=decision.reason,
            detail=detail,
            correlation_id=correlation_id,
        )
