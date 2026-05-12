"""Training control routes."""

import logging

import torch
from fastapi import APIRouter, HTTPException, Request

from api.lifecycle.manager import (
    InvalidCandidatePoolError,
    NoSwapInProgressError,
    SwapCancelledError,
    SwapInProgressError,
)
from api.models.common import success_response
from api.models.training import StageDatasetRequest, SwapDatasetLiveRequest, TrainingParamUpdateRequest, TrainingStartRequest

logger = logging.getLogger("juniper_cascor.api.routes.training")

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

    # SEC-07: the ``params`` field is now a typed ``TrainingParams`` model
    # with ``extra="forbid"``; Pydantic rejects unknown keys with 422 at
    # the request boundary, and per-field validators enforce numeric
    # ranges. The old hand-maintained ``_ALLOWED_TRAINING_PARAMS`` set and
    # silent-drop behavior are gone because they left values unchecked.
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

        # Handle training params — typed TrainingParams model rejects
        # unknown keys via Pydantic; forward only explicitly-set fields.
        if body.params is not None:
            kwargs.update(body.params.model_dump(exclude_none=True))

        if body.epochs is not None:
            kwargs["max_epochs"] = body.epochs

    try:
        result = lifecycle.start_training(x=x, y=y, x_val=x_val, y_val=y_val, **kwargs)
        return success_response(result)
    except (RuntimeError, ValueError) as e:
        logger.debug("Start training failed: %s", e)
        raise HTTPException(status_code=409, detail="Training cannot be started in the current state")


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
        logger.debug("Pause training failed: %s", e)
        raise HTTPException(status_code=409, detail="Training cannot be paused in the current state")


@router.post("/resume")
async def resume_training(request: Request) -> dict:
    """Resume paused training."""
    lifecycle = _get_lifecycle(request)
    try:
        result = lifecycle.resume_training()
        return success_response(result)
    except RuntimeError as e:
        logger.debug("Resume training failed: %s", e)
        raise HTTPException(status_code=409, detail="Training cannot be resumed in the current state")


@router.post("/reset")
async def reset_training(request: Request) -> dict:
    """Reset training state."""
    lifecycle = _get_lifecycle(request)
    result = lifecycle.reset()
    return success_response(result)


@router.get("/status")
async def get_status(request: Request) -> dict:
    """Get current training status with atomic snapshot_seq."""
    lifecycle = _get_lifecycle(request)
    status = lifecycle.get_status()

    ws_manager = getattr(request.app.state, "ws_manager", None)
    if ws_manager is not None:
        with ws_manager._seq_lock:
            status["snapshot_seq"] = ws_manager._next_seq - 1
            status["server_instance_id"] = ws_manager.server_instance_id

    return success_response(status)


@router.get("/params")
async def get_params(request: Request) -> dict:
    """Get current training parameters."""
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_network():
        raise HTTPException(status_code=404, detail="No network created")
    return success_response(lifecycle.get_training_params())


@router.patch("/params")
async def update_training_params(request: Request, body: TrainingParamUpdateRequest) -> dict:
    """Update runtime-modifiable training parameters.

    Modifies parameters on the running network without requiring a restart.
    All fields are optional — only provided fields are updated (PATCH semantics).
    """
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_network():
        raise HTTPException(status_code=404, detail="No network created")
    try:
        updated = lifecycle.update_params(body.model_dump(exclude_none=True))
        return success_response(updated)
    except InvalidCandidatePoolError as e:
        # FRONTEND_ISSUES_PLAN_2026-05-09 §1.5 C2.1 — surface the violation
        # string in the JSON body so the canopy adapter can route it through
        # the same `mismatches`/`skipped` toast machinery from C3.
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# FRONTEND_ISSUES_PLAN_2026-05-09 §3.5.1 + §3.5.2 P1 — Issue #3 Phase 1 dataset
# staging endpoints. Stage a dataset config now, restart training to apply
# (cold-swap), or DELETE the staged config to cancel before restart.


@router.post("/dataset")
async def stage_dataset(request: Request, body: StageDatasetRequest) -> dict:
    """Stage a dataset-config change for the next ``start_training``.

    Returns the staged config in ``data``. An empty body clears any prior
    staging (idempotent with DELETE for that case).
    """
    lifecycle = _get_lifecycle(request)
    cfg = body.model_dump(exclude_none=True)
    return success_response(lifecycle.stage_dataset_config(**cfg))


@router.delete("/dataset")
async def cancel_dataset_stage(request: Request) -> dict:
    """Discard any staged dataset change — Phase 1 Cancel button target."""
    lifecycle = _get_lifecycle(request)
    return success_response(lifecycle.clear_pending_dataset_config())


@router.get("/dataset/pending")
async def get_pending_dataset(request: Request) -> dict:
    """Return the staged dataset config (or null) — drives the canopy banner."""
    lifecycle = _get_lifecycle(request)
    return success_response({"pending": lifecycle.get_pending_dataset_config()})


# ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09 §3.3 — Phase 2 P2-1a live-swap
# entry point. Initiates an in-flight dataset swap without stopping training.
# Pre-conditions and failure modes per §3.2:
#   403 experimental_functions_disabled — gate is closed (F2.10)
#   422 training_not_running             — no active training to swap into
#   422 dim_change_unsupported           — P2-1a equal-dim only; P2-1c/1d will lift
#   409 swap_already_in_progress         — concurrent swap rejected (idempotency)
#   504 pause_timeout                    — training thread did not reach pause boundary
#   5xx                                  — juniper-data fetch / arch-adapt failure (rolled back)


@router.post("/dataset/live")
async def swap_dataset_live(request: Request, body: SwapDatasetLiveRequest) -> dict:
    """Initiate an in-flight dataset swap (P2-1a skeleton; P2-1b cancel-aware)."""
    lifecycle = _get_lifecycle(request)
    cfg = body.model_dump(exclude_none=True)
    try:
        return success_response(lifecycle.swap_dataset_live(**cfg))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except SwapInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SwapCancelledError:
        # P2-1b: a DELETE arrived mid-swap, the §3.8 rollback already restored
        # pre-swap state. Return 200 with cancelled status — the request was
        # serviced cleanly; not a failure path.
        return success_response({"status": "cancelled"})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TimeoutError as exc:
        # Future.result(timeout=10) raises TimeoutError on §3.7 guardrail #2.
        raise HTTPException(status_code=504, detail=f"pause_timeout: {exc}") from exc
    except RuntimeError as exc:
        # juniper-data fetch failure (raised by _reload_dataset) or other
        # operational error. Rollback already happened inside swap_dataset_live.
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.delete("/dataset/live")
async def cancel_swap_dataset_live(request: Request) -> dict:
    """Cancel an in-flight live dataset swap (P2-1b).

    Sets the cancellation signal observed by ``swap_dataset_live`` at its
    post-fetch checkpoint. The actual swap may take up to one fetch RTT to
    observe the flag and unwind — this route only confirms the signal was
    delivered. The originating ``POST`` returns 200 with
    ``{"status": "cancelled"}`` once the rollback completes.

    Status codes:
      200 — cancel signal accepted; an in-flight swap will roll back.
      403 — experimental-functions gate closed (F2.10).
      404 — no swap currently in progress.
    """
    lifecycle = _get_lifecycle(request)
    # Gate check mirrors the POST: a closed gate hides the existence of the
    # entire feature surface, including its cancel side. Avoids a probe vector
    # where a 404 vs 403 leaks "swap is enabled and idle".
    if not lifecycle.get_experimental_functions():
        raise HTTPException(status_code=403, detail="experimental_functions_disabled")
    try:
        return success_response(lifecycle.request_swap_cancel())
    except NoSwapInProgressError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
