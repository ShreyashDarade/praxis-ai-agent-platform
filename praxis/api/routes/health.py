# praxis/api/routes/health.py
"""`GET /health` (spec §14) - split out of `praxis.api.main`, which still
owns every underlying health-check registration/aggregation this route
calls into (`register_health_check`, `HEALTH_CHECKS`,
`run_health_checks`) - this module is just the route surface.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    # Imported inside the handlers, not at module scope: `main` imports
    # this module (to mount the router), so a module-level import here
    # is a genuine cycle that resolves today only because of import
    # ordering. Deferring it to call time removes the cycle without
    # moving any shared state out of `main` - several tests reset
    # `main._orchestrator` directly, and relocating it would make that
    # reset silently stop working.
    from praxis.api import main

    results = await main.run_health_checks()

    overall = all(r.healthy for r in results)
    body = {
        "healthy": overall,
        "components": [
            {"name": r.name, "healthy": r.healthy, "detail": r.detail} for r in results
        ],
    }
    return JSONResponse(content=body, status_code=200 if overall else 503)
