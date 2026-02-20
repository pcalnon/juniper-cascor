"""Dataset routes for training data metadata."""

from fastapi import APIRouter, HTTPException, Request

from api.models.common import success_response

router = APIRouter(prefix="/dataset", tags=["dataset"])


def _get_lifecycle(request: Request):
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise HTTPException(status_code=503, detail="Lifecycle manager not initialized")
    return lifecycle


@router.get("")
async def get_dataset(request: Request) -> dict:
    """Get dataset metadata."""
    lifecycle = _get_lifecycle(request)
    return success_response(lifecycle.get_dataset())
