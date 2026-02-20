"""Training control routes."""

from fastapi import APIRouter, HTTPException, Request

from api.models.common import success_response

router = APIRouter(prefix="/training", tags=["training"])


def _get_lifecycle(request: Request):
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise HTTPException(status_code=503, detail="Lifecycle manager not initialized")
    return lifecycle


@router.post("/start")
async def start_training(request: Request) -> dict:
    """Start network training."""
    lifecycle = _get_lifecycle(request)
    try:
        result = lifecycle.start_training()
        return success_response(result)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/stop")
async def stop_training(request: Request) -> dict:
    """Stop network training."""
    lifecycle = _get_lifecycle(request)
    result = lifecycle.stop_training()
    return success_response(result)


@router.post("/pause")
async def pause_training(request: Request) -> dict:
    """Pause network training."""
    lifecycle = _get_lifecycle(request)
    try:
        result = lifecycle.pause_training()
        return success_response(result)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/resume")
async def resume_training(request: Request) -> dict:
    """Resume paused training."""
    lifecycle = _get_lifecycle(request)
    try:
        result = lifecycle.resume_training()
        return success_response(result)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/reset")
async def reset_training(request: Request) -> dict:
    """Reset training state."""
    lifecycle = _get_lifecycle(request)
    result = lifecycle.reset()
    return success_response(result)


@router.get("/status")
async def get_status(request: Request) -> dict:
    """Get current training status."""
    lifecycle = _get_lifecycle(request)
    return success_response(lifecycle.get_status())


@router.get("/params")
async def get_params(request: Request) -> dict:
    """Get current training parameters."""
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_network():
        raise HTTPException(status_code=404, detail="No network created")
    return success_response(lifecycle.get_training_params())
