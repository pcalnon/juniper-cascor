"""Dataset routes for training data metadata."""

from fastapi import APIRouter, HTTPException, Request

from api.models.common import ResponseEnvelope, success_response

router = APIRouter(prefix="/dataset", tags=["dataset"])


def _get_lifecycle(request: Request):
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise HTTPException(status_code=503, detail="Lifecycle manager not initialized")
    return lifecycle


@router.get("", operation_id="get_dataset", response_model=ResponseEnvelope)
async def get_dataset(request: Request) -> dict:
    """Get dataset metadata."""
    lifecycle = _get_lifecycle(request)
    return success_response(lifecycle.get_dataset())


@router.get("/data", operation_id="get_dataset_data", response_model=ResponseEnvelope)
async def get_dataset_data(request: Request) -> dict:
    """Get dataset arrays for visualization."""
    lifecycle = _get_lifecycle(request)
    data = lifecycle.get_dataset_data()
    if data is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    return success_response(data)
