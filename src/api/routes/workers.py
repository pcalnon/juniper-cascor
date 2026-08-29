"""REST endpoints for querying remote worker status.

Exposes the WorkerRegistry data as read-only REST endpoints for monitoring
dashboards and client tooling.
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from api.models.common import ResponseEnvelope, success_response

logger = logging.getLogger("juniper_cascor.api.routes.workers")

router = APIRouter(prefix="/workers", tags=["workers"])


def _get_registry(request: Request):
    registry = getattr(request.app.state, "worker_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Worker registry not initialized")
    return registry


def _serialize_worker(worker) -> dict:
    """Serialize a WorkerRegistration to a JSON-safe dict.

    METRICS-MON R1.3 / seed-04: includes ``in_flight_tasks``,
    ``last_task_completed_at``, and ``rss_mb`` from R1.3-aware workers'
    enriched heartbeats.

    METRICS-MON R4.4: includes ``last_task_duration_seconds``,
    ``recent_task_durations_seconds``, and ``gpu_utilization_pct`` from
    R4.4-aware workers' training-loop instrumentation.

    Workers running older images report missing fields as ``0`` / ``None``
    / ``[]`` defaults until they upgrade.
    """
    return {
        "worker_id": worker.worker_id,
        "capabilities": worker.capabilities,
        "connected_at": worker.connected_at,
        "last_heartbeat": worker.last_heartbeat,
        "tasks_completed": worker.tasks_completed,
        "tasks_failed": worker.tasks_failed,
        "active_task_id": worker.active_task_id,
        "health_score": worker.health_score,
        "idle": worker.idle,
        # METRICS-MON R1.3 / seed-04: enriched fields.
        "in_flight_tasks": worker.in_flight_tasks,
        "last_task_completed_at": worker.last_task_completed_at,
        "rss_mb": worker.rss_mb,
        # METRICS-MON R4.4: training-loop instrumentation fields.
        "last_task_duration_seconds": worker.last_task_duration_seconds,
        "recent_task_durations_seconds": worker.recent_task_durations_seconds,
        "gpu_utilization_pct": worker.gpu_utilization_pct,
    }


@router.get("", operation_id="list_workers", response_model=ResponseEnvelope)
async def list_workers(request: Request) -> dict:
    """List all registered workers with status."""
    registry = _get_registry(request)
    workers = registry.get_all_workers()
    return success_response(
        {
            "workers": [_serialize_worker(w) for w in workers],
            "count": len(workers),
        }
    )


@router.get("/stats", operation_id="get_worker_stats", response_model=ResponseEnvelope)
async def get_worker_stats(request: Request) -> dict:
    """Aggregate worker statistics."""
    registry = _get_registry(request)
    all_workers = registry.get_all_workers()
    idle_workers = registry.get_idle_workers()
    stale_workers = registry.get_stale_workers()

    total_completed = sum(w.tasks_completed for w in all_workers)
    total_failed = sum(w.tasks_failed for w in all_workers)
    avg_health = sum(w.health_score for w in all_workers) / len(all_workers) if all_workers else 0.0

    return success_response(
        {
            "total": len(all_workers),
            "idle": len(idle_workers),
            "busy": len(all_workers) - len(idle_workers) - len(stale_workers),
            "stale": len(stale_workers),
            "total_tasks_completed": total_completed,
            "total_tasks_failed": total_failed,
            "average_health_score": round(avg_health, 4),
            "timestamp": time.time(),
        }
    )


@router.get("/{worker_id}", operation_id="get_worker", response_model=ResponseEnvelope)
async def get_worker(request: Request, worker_id: str) -> dict:
    """Get details for a specific worker."""
    registry = _get_registry(request)
    worker = registry.get(worker_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    return success_response(_serialize_worker(worker))
