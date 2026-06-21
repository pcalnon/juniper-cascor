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

import importlib.metadata
import time

from fastapi import APIRouter, Request, Response

# METRICS-MON R2.1.4: liveness/readiness contract constants now live in the
# shared juniper-observability package so all three Juniper servers cannot
# drift from the R1.2 cross-service contract.
from juniper_observability import LIVENESS_STALENESS_SECONDS, LIVENESS_TICK_BUDGET_MS, READINESS_HEADER

from api import provenance
from api.models.health import DependencyStatus, ReadinessResponse, probe_dependency
from api.settings import Settings

# Single source of truth: the installed distribution's metadata (OQ-1 of the
# build-provenance effort — juniper-ml notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md).
# Falls back to the literal only in a bare source checkout where the package is
# not installed, so this constant can no longer drift from pyproject's version.
try:
    _API_VERSION: str = importlib.metadata.version("juniper-cascor")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover - source checkout
    _API_VERSION = "0.5.0"

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

    Response schema (API-02 shared base):

    - ``status``  — always ``"ok"`` on success.
    - ``version`` — the package version of this service.
    - ``service`` — canonical service identifier (``"juniper-cascor"``);
      matches the cross-service ``{status, version, service}`` base
      shared by juniper-data and juniper-canopy so monitoring tools can
      tell health responses apart without inspecting the URL.
    """
    return {
        "status": "ok",
        "version": _API_VERSION,
        "service": "juniper-cascor",
        # Build provenance (juniper-ml notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md):
        # source git SHA + ISO-8601 build date baked into the image. ``None``
        # outside a provenance-stamped image; lets ``make doctor`` detect drift.
        "git_sha": provenance.git_sha(),
        "build_date": provenance.build_date(),
    }


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
    network_loaded = lifecycle.has_model() if lifecycle else False

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
    # CFG-04: Settings field consolidates the env-var lookup. Fresh
    # ``Settings()`` per request (not the cached ``get_settings()``) so
    # tests that patch ``JUNIPER_DATA_URL`` per-test pick up the change
    # without needing ``get_settings.cache_clear()`` plumbing — pydantic
    # construction is sub-millisecond and this route is rarely on a hot
    # path.
    data_url = Settings().juniper_data_url
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
        git_sha=provenance.git_sha(),
        build_date=provenance.build_date(),
        dependencies=dependencies,
        details={"network_loaded": network_loaded, "training_state": training_state},
    )
