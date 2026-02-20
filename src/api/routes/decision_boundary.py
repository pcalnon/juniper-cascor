"""Decision boundary routes for 2D visualization."""

from fastapi import APIRouter, HTTPException, Query, Request

from api.models.common import success_response

router = APIRouter(prefix="/decision-boundary", tags=["decision-boundary"])


def _get_lifecycle(request: Request):
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise HTTPException(status_code=503, detail="Lifecycle manager not initialized")
    return lifecycle


@router.get("")
async def get_decision_boundary(
    request: Request,
    resolution: int = Query(50, ge=5, le=200, description="Grid resolution for boundary computation"),
) -> dict:
    """Get decision boundary data for 2D visualization.

    Computes a grid of network predictions over the input space.
    Requires a network with 2D input and loaded training data.
    """
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_network():
        raise HTTPException(status_code=404, detail="No network created")
    if not lifecycle.has_training_data():
        raise HTTPException(status_code=404, detail="No training data loaded")
    boundary = lifecycle.get_decision_boundary(resolution=resolution)
    if boundary is None:
        raise HTTPException(status_code=500, detail="Failed to compute decision boundary")
    return success_response(boundary)
