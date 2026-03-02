"""Health check endpoints for container orchestration.

Provides three health check endpoints:
- /v1/health: Combined health check (backward compatible)
- /v1/health/live: Liveness probe - is the process running?
- /v1/health/ready: Readiness probe - is the service ready to accept traffic?

Health endpoints return flat JSON (not wrapped in ResponseEnvelope) for
compatibility with Docker healthcheck and Kubernetes httpGet probes that
expect a top-level ``status`` field.
"""

import os

from fastapi import APIRouter, Request

from api.models.health import ReadinessResponse, probe_dependency

_API_VERSION: str = "0.4.0"

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """Combined health check endpoint (backward compatible)."""
    return {"status": "ok", "version": _API_VERSION}


@router.get("/health/live")
async def liveness_probe() -> dict:
    """Liveness probe for container orchestration."""
    return {"status": "alive"}


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_probe(request: Request) -> ReadinessResponse:
    """Readiness probe for container orchestration.

    Reports ready when the lifecycle manager is available. Probes the
    JuniperData service health endpoint when ``JUNIPER_DATA_URL`` is set.
    Returns a flat ReadinessResponse (no envelope).
    """
    lifecycle = getattr(request.app.state, "lifecycle", None)
    network_loaded = lifecycle.has_network() if lifecycle else False

    training_state = "unknown"
    if lifecycle is not None:
        try:
            status = lifecycle.get_status()
            training_state = status.get("training_state", "unknown")
        except Exception:
            pass

    # Probe JuniperData
    data_url = os.getenv("JUNIPER_DATA_URL")
    dependencies: dict = {}
    if data_url:
        data_dep = probe_dependency("JuniperData Service", f"{data_url.rstrip('/')}/v1/health/live")
        dependencies["juniper_data"] = data_dep

    overall = "ready"
    for dep in dependencies.values():
        if dep.status == "unhealthy":
            overall = "degraded"
            break

    return ReadinessResponse(
        status=overall,
        version=_API_VERSION,
        service="juniper-cascor",
        dependencies=dependencies,
        details={"network_loaded": network_loaded, "training_state": training_state},
    )
