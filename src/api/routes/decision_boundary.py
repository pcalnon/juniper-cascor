"""Decision boundary routes for 2D visualization."""

import asyncio

from fastapi import APIRouter, HTTPException, Query, Request

from api.models.common import success_response
from cascor_constants.constants_api import _PROJECT_API_DECISION_BOUNDARY_RESOLUTION_DEFAULT, _PROJECT_API_DECISION_BOUNDARY_RESOLUTION_MAX, _PROJECT_API_DECISION_BOUNDARY_RESOLUTION_MIN, _PROJECT_API_HTTP_404_NOT_FOUND, _PROJECT_API_HTTP_500_INTERNAL_SERVER_ERROR, _PROJECT_API_HTTP_503_SERVICE_UNAVAILABLE

router = APIRouter(prefix="/decision-boundary", tags=["decision-boundary"])


def _get_lifecycle(request: Request):
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise HTTPException(status_code=_PROJECT_API_HTTP_503_SERVICE_UNAVAILABLE, detail="Lifecycle manager not initialized")
    return lifecycle


@router.get("")
async def get_decision_boundary(
    request: Request,
    resolution: int = Query(_PROJECT_API_DECISION_BOUNDARY_RESOLUTION_DEFAULT, ge=_PROJECT_API_DECISION_BOUNDARY_RESOLUTION_MIN, le=_PROJECT_API_DECISION_BOUNDARY_RESOLUTION_MAX, description="Grid resolution for boundary computation"),
) -> dict:
    """Get decision boundary data for 2D visualization.

    Computes a grid of network predictions over the input space.
    Requires a network with 2D input and loaded training data.
    Computation is offloaded to a thread to avoid blocking the event loop.
    """
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_network():
        raise HTTPException(status_code=_PROJECT_API_HTTP_404_NOT_FOUND, detail="No network created")
    if not lifecycle.has_training_data():
        raise HTTPException(status_code=_PROJECT_API_HTTP_404_NOT_FOUND, detail="No training data loaded")
    boundary = await asyncio.to_thread(lifecycle.get_decision_boundary, resolution=resolution)
    if boundary is None:
        raise HTTPException(status_code=_PROJECT_API_HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to compute decision boundary")
    return success_response(boundary)
