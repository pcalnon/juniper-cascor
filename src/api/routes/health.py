"""Health check endpoints for container orchestration.

Provides three health check endpoints:

- /v1/health: Combined health check (backward compatible)
- /v1/health/live: Liveness probe — runs an in-process tick within a strict
  budget; returns 503 if the lifecycle heartbeat is stale or no lifecycle
  is bound, so the orchestrator can restart wedged pods.
- /v1/health/ready: Readiness probe — returns 200 when all required
  dependencies are healthy, 200 with status "degraded" when only optional
  dependencies are unhealthy, and 503 when a required dependency is
  unhealthy so load balancers can shed traffic without parsing the body.

Health endpoints return flat JSON (not wrapped in ResponseEnvelope) for
compatibility with Docker healthcheck and Kubernetes httpGet probes that
expect a top-level ``status`` field.

See ``notes/code-review/METRICS_MONITORING_R1.2_PROBE_DESIGN_2026-04-27.md``
in juniper-ml for the cross-repo contract this implements (R1.2 / seed-02
and seed-03).
"""

import os
import time

from fastapi import APIRouter, Request, Response

from api.models.health import DependencyStatus, ReadinessResponse, probe_dependency

_API_VERSION: str = "0.4.0"

# R1.2: liveness tick budget. The tick is purely in-process (consults the
# lifecycle heartbeat counter); 250 ms catches event-loop stalls and CPU
# starvation. Helm timeoutSeconds (5–10) wraps this with headroom.
LIVENESS_TICK_BUDGET_MS = 250

# R1.2: header surfaces readiness state to ``kubectl describe pod`` /
# ``curl -I`` without requiring body parsing.
READINESS_HEADER = "X-Juniper-Readiness"

# R1.2: heartbeat staleness threshold. Lifecycle daemon thread bumps every
# 1 second; a staleness > 30 s reliably indicates a wedged process.
LIVENESS_STALENESS_SECONDS = 30.0

router = APIRouter(tags=["health"])


def _liveness_tick(request: Request) -> None:
    """Run the juniper-cascor liveness tick.

    Pure in-process work: consults ``app.state.lifecycle.is_alive()``.
    Raises if the lifecycle is missing or the heartbeat is stale.
    """
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise RuntimeError("lifecycle manager not bound on app.state")
    if not lifecycle.is_alive(stale_after_seconds=LIVENESS_STALENESS_SECONDS):
        raise RuntimeError(f"lifecycle heartbeat stale (> {LIVENESS_STALENESS_SECONDS}s)")


@router.get("/health")
async def health_check() -> dict:
    """Combined health check endpoint (backward compatible).

    Always returns 200 while the process can respond. Reserved for legacy
    integrations; new probes should use ``/health/live`` or
    ``/health/ready``.
    """
    return {"status": "ok", "version": _API_VERSION}


@router.get("/health/live")
async def liveness_probe(request: Request, response: Response) -> dict:
    """Liveness probe — runs an in-process tick within a strict budget.

    Returns 200 with ``{"status": "alive", "tick": "juniper-cascor",
    "duration_ms": N}`` when the tick succeeds within
    ``LIVENESS_TICK_BUDGET_MS``. Returns 503 with ``{"status":
    "unresponsive", ...}`` otherwise.
    """
    started = time.perf_counter()
    try:
        _liveness_tick(request)
    except Exception as exc:  # noqa: BLE001 — health probe must surface every failure
        duration_ms = int((time.perf_counter() - started) * 1000)
        response.status_code = 503
        return {
            "status": "unresponsive",
            "tick": "juniper-cascor",
            "error": str(exc),
            "duration_ms": duration_ms,
        }

    duration_ms = int((time.perf_counter() - started) * 1000)
    if duration_ms > LIVENESS_TICK_BUDGET_MS:
        response.status_code = 503
        return {
            "status": "unresponsive",
            "tick": "juniper-cascor",
            "error": f"tick exceeded budget: {duration_ms}ms > {LIVENESS_TICK_BUDGET_MS}ms",
            "duration_ms": duration_ms,
        }

    return {
        "status": "alive",
        "tick": "juniper-cascor",
        "duration_ms": duration_ms,
    }


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_probe(request: Request, response: Response) -> ReadinessResponse:
    """Readiness probe — drives orchestrator traffic decisions via status code.

    Status code semantics:

    - 200, body status="ready"     — lifecycle bound and (when configured)
      JuniperData reachable.
    - 200, body status="degraded"  — required deps healthy, an optional
      dep unhealthy. juniper-cascor has no optional deps; this branch is
      unreachable for this service.
    - 503, body status="not_ready" — lifecycle missing OR (when
      ``JUNIPER_DATA_URL`` set) JuniperData unreachable.

    Sets ``X-Juniper-Readiness`` header to mirror body status.
    """
    lifecycle = getattr(request.app.state, "lifecycle", None)
    network_loaded = lifecycle.has_network() if lifecycle else False

    training_state = "unknown"
    if lifecycle is not None:
        try:
            status = lifecycle.get_status()
            training_state = status.get("training_state", "unknown")
        except Exception:  # nosec B110 - intentional pass for health check resilience
            pass

    dependencies: dict = {}

    # Required dep: lifecycle manager.
    dependencies["lifecycle"] = DependencyStatus(
        name="Lifecycle Manager",
        status="healthy" if lifecycle is not None else "unhealthy",
        message="bound" if lifecycle is not None else "not bound on app.state",
    )

    # Required-when-configured dep: JuniperData. URL unset → dep skipped
    # entirely (collapses to ready); URL set + unhealthy → not_ready.
    data_url = os.getenv("JUNIPER_DATA_URL")
    if data_url:
        dependencies["juniper_data"] = probe_dependency("JuniperData Service", f"{data_url.rstrip('/')}/v1/health/live")

    # Required deps for cascor: lifecycle (always) and juniper_data when
    # configured. ``not_configured`` is treated as healthy for readiness.
    required_unhealthy = any(dep.status == "unhealthy" for dep_name, dep in dependencies.items() if dep_name in {"lifecycle", "juniper_data"})

    if required_unhealthy:
        overall = "not_ready"
        response.status_code = 503
    else:
        overall = "ready"

    response.headers[READINESS_HEADER] = overall

    return ReadinessResponse(
        status=overall,
        version=_API_VERSION,
        service="juniper-cascor",
        dependencies=dependencies,
        details={"network_loaded": network_loaded, "training_state": training_state},
    )
