"""Training control routes."""

import torch
from fastapi import APIRouter, HTTPException, Request

from api.models.common import success_response
from api.models.training import TrainingStartRequest

router = APIRouter(prefix="/training", tags=["training"])


def _get_lifecycle(request: Request):
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise HTTPException(status_code=503, detail="Lifecycle manager not initialized")
    return lifecycle


@router.post("/start")
async def start_training(request: Request, body: TrainingStartRequest = None) -> dict:
    """Start network training.

    Accepts an optional request body with:
    - inline_data: Direct training data (train_x, train_y, val_x, val_y)
    - dataset: Dataset source specification (juniper-data or generator)
    - params: Training parameter overrides
    - epochs: Max epochs override (shorthand)
    """
    lifecycle = _get_lifecycle(request)

    kwargs = {}
    x = None
    y = None
    x_val = None
    y_val = None

    if body is not None:
        # Handle inline dataset
        if body.inline_data is not None:
            x = torch.tensor(body.inline_data.train_x, dtype=torch.float32)
            y = torch.tensor(body.inline_data.train_y, dtype=torch.float32)
            if body.inline_data.val_x is not None:
                x_val = torch.tensor(body.inline_data.val_x, dtype=torch.float32)
                y_val = torch.tensor(body.inline_data.val_y, dtype=torch.float32)

        # Handle dataset source (juniper-data generator)
        if body.dataset is not None and body.dataset.generator == "spiral":
            x, y = _generate_spiral_data(body.dataset.params or {})

        # Handle training params
        if body.params:
            kwargs.update(body.params)

        if body.epochs is not None:
            kwargs["max_epochs"] = body.epochs

    try:
        result = lifecycle.start_training(x=x, y=y, x_val=x_val, y_val=y_val, **kwargs)
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


def _generate_spiral_data(params: dict):
    """Generate spiral dataset for training."""
    import numpy as np

    n_per_spiral = params.get("n_per_spiral", 100)
    n_spirals = params.get("n_spirals", 2)

    x_data = []
    y_data = []

    for i in range(n_spirals):
        t = np.linspace(0, 4 * np.pi, n_per_spiral)
        angle_offset = 2 * np.pi * i / n_spirals

        x_spiral = t * np.cos(t + angle_offset) / (4 * np.pi)
        y_spiral = t * np.sin(t + angle_offset) / (4 * np.pi)

        x_data.append(np.stack([x_spiral, y_spiral], axis=1))

        y_one_hot = np.zeros((n_per_spiral, n_spirals))
        y_one_hot[:, i] = 1
        y_data.append(y_one_hot)

    x = torch.tensor(np.concatenate(x_data, axis=0), dtype=torch.float32)
    y = torch.tensor(np.concatenate(y_data, axis=0), dtype=torch.float32)

    return x, y
