# praxis/api/routes/health.py
"""`GET /health` (spec §14) - split out of `praxis.api.main`, which still
owns every underlying health-check registration/aggregation this route
calls into (`register_health_check`, `HEALTH_CHECKS`,
`run_health_checks`) - this module is just the route surface.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from praxis.api import main

router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    results = await main.run_health_checks()

    overall = all(r.healthy for r in results)
    body = {
        "healthy": overall,
        "components": [
            {"name": r.name, "healthy": r.healthy, "detail": r.detail} for r in results
        ],
    }
    return JSONResponse(content=body, status_code=200 if overall else 503)
