"""Metrics routes for training metrics retrieval."""

from fastapi import APIRouter, HTTPException, Query, Request

from api.models.common import ResponseEnvelope, success_response

router = APIRouter(prefix="/metrics", tags=["metrics"])


def _get_lifecycle(request: Request):
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise HTTPException(status_code=503, detail="Lifecycle manager not initialized")
    return lifecycle


@router.get("", operation_id="get_metrics", response_model=ResponseEnvelope)
async def get_metrics(request: Request) -> dict:
    """Get current training metrics snapshot."""
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_model():
        raise HTTPException(status_code=404, detail="No network created")
    return success_response(lifecycle.get_metrics())


@router.get("/history", operation_id="get_metrics_history", response_model=ResponseEnvelope)
async def get_metrics_history(
    request: Request,
    count: int = Query(None, ge=1, description="Number of recent metrics to return"),
) -> dict:
    """Get training metrics history."""
    lifecycle = _get_lifecycle(request)
    return success_response(lifecycle.get_metrics_history(count=count))


@router.get("/transport", operation_id="get_transport_stats", response_model=ResponseEnvelope)
async def get_transport_stats(request: Request) -> dict:
    """GAP-WS-16: cumulative WebSocket transport counters.

    Diagnostic endpoint for validating the bandwidth delta from REST polling
    once GAP-WS-16 lands. All counters are cumulative since process start.
    Returns a payload with bytes/messages sent (totals + per type), connection
    counts, and replay-buffer state. Returns 503 if the WebSocket manager is
    not initialized (shouldn't happen post-startup, but kept defensive).
    """
    ws_manager = getattr(request.app.state, "ws_manager", None)
    if ws_manager is None:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    return success_response(ws_manager.transport_stats())
