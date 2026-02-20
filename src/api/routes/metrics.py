"""Metrics routes for training metrics retrieval."""

from fastapi import APIRouter, HTTPException, Query, Request

from api.models.common import success_response

router = APIRouter(prefix="/metrics", tags=["metrics"])


def _get_lifecycle(request: Request):
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise HTTPException(status_code=503, detail="Lifecycle manager not initialized")
    return lifecycle


@router.get("")
async def get_metrics(request: Request) -> dict:
    """Get current training metrics snapshot."""
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_network():
        raise HTTPException(status_code=404, detail="No network created")
    return success_response(lifecycle.get_metrics())


@router.get("/history")
async def get_metrics_history(
    request: Request,
    count: int = Query(None, ge=1, description="Number of recent metrics to return"),
) -> dict:
    """Get training metrics history."""
    lifecycle = _get_lifecycle(request)
    return success_response(lifecycle.get_metrics_history(count=count))
