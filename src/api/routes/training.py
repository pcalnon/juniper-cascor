"""Training control routes."""

import logging

import torch
from fastapi import APIRouter, HTTPException, Request

from api.lifecycle.manager import (
    InvalidCandidatePoolError,
    NoMetricsUndoError,
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

    If no network exists yet, one is created automatically from the training
    data's dims (inline/staged/pending dataset) — ``_auto_start_training``
    parity, so the first user-initiated start works on a fresh service
    (training-start diagnosis 2026-07-09, PR-B). A start with neither data nor
    a staged dataset still 409s with "Training data not provided".
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
    # C5 (Q4 use-case 2 / U-1): start-fresh toggle (default off). Forwarded to
    # the lifecycle, which discards the model + retained metrics/history before
    # a fresh run (snapshots preserved). Omitted / False continues the current
    # model with its retained metrics/history (Q4 use-case 1).
    start_fresh = False

    if body is not None:
        start_fresh = bool(body.start_fresh)
        # Handle inline dataset
        if body.inline_data is not None:
            x = torch.tensor(body.inline_data.train_x, dtype=torch.float32)
            y = torch.tensor(body.inline_data.train_y, dtype=torch.float32)
            # InlineDataset's model_validator already rejects a half-specified
            # validation split; require both here so a future model change cannot
            # reintroduce ``torch.tensor(None)`` on a lone val_x/val_y.
            if body.inline_data.val_x is not None and body.inline_data.val_y is not None:
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
        result = lifecycle.start_training(X=x, y=y, X_val=x_val, y_val=y_val, start_fresh=start_fresh, **kwargs)
        return success_response(result)
    except (RuntimeError, ValueError) as e:
        # Surface the specific reason (training already in progress / no dataset
        # staged ("Training data not provided") / juniper-data fetch failed /
        # investigating|replaying a snapshot) rather than a generic message, so API and
        # Canopy callers can tell *why* the start was rejected and act on it. The generic
        # string previously masked a juniper-data fetch failure as a bogus "state" error.
        # See notes/CASCOR_STARTUP_SECRET_INDIRECTION_INVESTIGATION_2026-06-14.md (3.4).
        # ("No network created" left this list in PR-B — start now creates the
        # network from the dataset dims when one is missing.)
        logger.warning("Start training failed: %s", e)
        raise HTTPException(status_code=409, detail=f"Training cannot be started: {e}") from e


@router.post("/stop")
async def stop_training(request: Request) -> dict:
    """Stop network training."""
    lifecycle = _get_lifecycle(request)
    try:
        result = lifecycle.stop_training()
        return success_response(result)
    except RuntimeError as e:
        logger.debug("Stop training failed: %s", e)
        raise HTTPException(status_code=409, detail="Training cannot be stopped in the current state") from e


@router.post("/pause")
async def pause_training(request: Request) -> dict:
    """Pause network training."""
    lifecycle = _get_lifecycle(request)
    try:
        result = lifecycle.pause_training()
        return success_response(result)
    except RuntimeError as e:
        logger.debug("Pause training failed: %s", e)
        raise HTTPException(status_code=409, detail="Training cannot be paused in the current state") from e


@router.post("/resume")
async def resume_training(request: Request) -> dict:
    """Resume paused training."""
    lifecycle = _get_lifecycle(request)
    try:
        result = lifecycle.resume_training()
        return success_response(result)
    except RuntimeError as e:
        logger.debug("Resume training failed: %s", e)
        raise HTTPException(status_code=409, detail="Training cannot be resumed in the current state") from e


@router.post("/reset")
async def reset_training(request: Request) -> dict:
    """Reset training state."""
    lifecycle = _get_lifecycle(request)
    result = lifecycle.reset()
    return success_response(result)


@router.post("/metrics/clear")
async def clear_metrics(request: Request) -> dict:
    """Clear the retained training metrics/history, with undo (C5 / Q4 use-case 1).

    Retention is now the default across run boundaries (Q4/U-1), so this is the
    explicit control that empties the metrics/history buffer between runs. The
    clear is reversible via ``POST /v1/training/metrics/clear/undo`` at any
    point until the next training run starts (starting a run finalizes the
    clear and drops the undo snapshot). Unlike ``POST /v1/training/reset`` this
    touches metrics/history only — it does not reset the FSM, counters, or the
    model.

    Returns ``{"status": "cleared", "cleared_count": <int>, "undo_available":
    true}``.
    """
    lifecycle = _get_lifecycle(request)
    return success_response(lifecycle.clear_metrics_with_undo())


@router.post("/metrics/clear/undo")
async def undo_clear_metrics(request: Request) -> dict:
    """Undo the most recent metrics/history clear (C5 / Q4 use-case 1 fallback).

    Restores the rows removed by the last ``POST /v1/training/metrics/clear``.
    Valid only until the next training run starts; returns 409 when there is no
    clear to undo (nothing cleared, or a run has started since).

    Returns ``{"status": "restored", "restored_count": <int>, "undo_available":
    false}``.
    """
    lifecycle = _get_lifecycle(request)
    try:
        return success_response(lifecycle.undo_clear_metrics())
    except NoMetricsUndoError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


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
    if not lifecycle.has_model():
        raise HTTPException(status_code=404, detail="No network created")
    return success_response(lifecycle.get_training_params())


@router.patch("/params")
async def update_training_params(request: Request, body: TrainingParamUpdateRequest) -> dict:
    """Update runtime-modifiable training parameters.

    Modifies parameters on the running network without requiring a restart.
    All fields are optional — only provided fields are updated (PATCH semantics).

    C2a (I-4 / T3): the success ``data`` accounts for every requested key — the
    full params echo plus additive ``applied: [key, ...]`` and
    ``skipped: [{"key", "reason"}, ...]`` fields, so a whitelisted key the live
    network object lacks is reported (``no-such-attribute``) instead of being
    silently dropped. Bound violations remain atomic 422 rejections at
    request-model validation (no partial apply).
    """
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_model():
        raise HTTPException(status_code=404, detail="No network created")
    try:
        updated = lifecycle.update_params(body.model_dump(exclude_none=True))
        return success_response(updated)
    except InvalidCandidatePoolError as e:
        # FRONTEND_ISSUES_PLAN_2026-05-09 §1.5 C2.1 — surface the violation
        # string in the JSON body so the canopy adapter can route it through
        # the same `mismatches`/`skipped` toast machinery from C3.
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        # CAN-015c: update_params rejects REPLAYING — surface as 409 so
        # Canopy can distinguish conflict from missing-network 404.
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


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
