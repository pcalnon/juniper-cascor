"""Health check endpoints for container orchestration.

Provides three health check endpoints:
- /v1/health: Combined health check (backward compatible)
- /v1/health/live: Liveness probe - is the process running?
- /v1/health/ready: Readiness probe - is the service ready to accept traffic?
"""

from fastapi import APIRouter, Request

from api.models.common import success_response

_API_VERSION: str = "0.4.0"

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """Combined health check endpoint (backward compatible)."""
    return success_response({"status": "ok", "version": _API_VERSION})


@router.get("/health/live")
async def liveness_probe() -> dict:
    """Liveness probe for container orchestration."""
    return success_response({"status": "alive"})


@router.get("/health/ready")
async def readiness_probe(request: Request) -> dict:
    """Readiness probe for container orchestration.

    Reports ready when the lifecycle manager is available.
    """
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        return success_response({"status": "ready", "version": _API_VERSION, "network_loaded": False})

    return success_response(
        {
            "status": "ready",
            "version": _API_VERSION,
            "network_loaded": lifecycle.has_network(),
        }
    )
