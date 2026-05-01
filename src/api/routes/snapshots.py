"""Snapshot management routes."""

import asyncio
import logging
import re

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from api.models.common import success_response

logger = logging.getLogger("juniper_cascor.api.routes.snapshots")

router = APIRouter(prefix="/snapshots", tags=["snapshots"])

# SEC-17: allowlist for snapshot identifiers. The lifecycle manager already
# matches on file stems via ``glob("*.h5")`` so a crafted path with ``..``
# or a slash cannot escape the snapshots directory today, but we enforce a
# strict regex at the route boundary so (a) attempts are rejected with 400
# before reaching the lifecycle layer, (b) any future code path that uses
# ``snapshot_id`` to build a path directly inherits the defense, and
# (c) traversal attempts are audit-logged.
_SNAPSHOT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _validate_snapshot_id(snapshot_id: str, client: str | None = None) -> None:
    """Reject ``snapshot_id`` values that are not pure alphanumerics/_/-.

    Raises ``HTTPException(400)`` with a fixed detail. Also logs the bad
    identifier at WARNING so operators can spot traversal probing in the
    access logs.
    """
    if not _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id or ""):
        logger.warning(
            "Rejected snapshot_id (invalid format): %r client=%s",
            snapshot_id,
            client or "unknown",
        )
        raise HTTPException(status_code=400, detail="Invalid snapshot_id format")


class SnapshotCreateRequest(BaseModel):
    """Request body for creating a snapshot."""

    description: str = ""


def _get_lifecycle(request: Request):
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise HTTPException(status_code=503, detail="Lifecycle manager not initialized")
    return lifecycle


@router.post("")
async def save_snapshot(request: Request, body: SnapshotCreateRequest = None) -> dict:
    """Save a snapshot of the current network state."""
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_network():
        raise HTTPException(status_code=404, detail="No network created")
    description = body.description if body else ""
    # PERF-CC-01: serializer.save_network is synchronous HDF5 I/O. Run it
    # off the event loop so concurrent requests aren't blocked while the
    # snapshot is being written.
    result = await asyncio.to_thread(lifecycle.save_snapshot, description=description)
    if result is None:
        raise HTTPException(status_code=404, detail="No network available to snapshot")
    return success_response(result)


@router.get("")
async def list_snapshots(request: Request) -> dict:
    """List all available snapshots."""
    lifecycle = _get_lifecycle(request)
    return success_response(lifecycle.list_snapshots())


@router.get("/{snapshot_id}")
async def get_snapshot(request: Request, snapshot_id: str) -> dict:
    """Get metadata for a specific snapshot."""
    _validate_snapshot_id(snapshot_id, client=request.client.host if request.client else None)
    lifecycle = _get_lifecycle(request)
    result = lifecycle.get_snapshot(snapshot_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found")
    return success_response(result)


@router.post("/{snapshot_id}/restore")
async def restore_snapshot(request: Request, snapshot_id: str) -> dict:
    """Restore a network from a snapshot.

    CAN-014 (Phase 6E Sprint A-5): the response now includes the
    post-restore ``training_params`` so the client can verify the
    round-trip without making a second ``GET /v1/training/params``
    call. The serializer round-trips every field listed in
    ``update_params``' whitelist (PR #163 / CAN-014); surfacing them
    here lets a tuning UI immediately reconcile its local state.
    """
    _validate_snapshot_id(snapshot_id, client=request.client.host if request.client else None)
    lifecycle = _get_lifecycle(request)
    # PERF-CC-01: serializer.load_network is synchronous HDF5 I/O. Run it
    # off the event loop so concurrent requests aren't blocked while the
    # snapshot is being read.
    success = await asyncio.to_thread(lifecycle.load_snapshot, snapshot_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found or failed to load")
    # ``get_training_params`` is cheap (synchronous attribute reads under
    # the training lock) and it would be surprising to surface restore
    # without surfacing what the client actually got back.
    try:
        params = lifecycle.get_training_params()
    except Exception:
        # Defensive — if the params extraction fails, the restore itself
        # already succeeded; fall back to the original minimal response
        # rather than making the round-trip look failed.
        logger.exception("restore_snapshot: get_training_params failed after successful restore")
        params = None
    payload: dict = {"snapshot_id": snapshot_id, "status": "restored"}
    if params is not None:
        payload["training_params"] = params
    return success_response(payload)


@router.post("/{snapshot_id}/retrain")
async def retrain_from_snapshot(request: Request, snapshot_id: str) -> dict:
    """Restore a snapshot and reset history for a fresh training run.

    CAN-015a (Phase 6E Sprint B B-1). The lifecycle loads the snapshot
    identically to ``/restore`` (preserving weights, topology, and all
    A-5 meta-parameters) then resets the training history arrays,
    counters, FSM state, and auto-snap-best ratchet so the next
    ``POST /v1/training/start`` call begins at epoch 0 with empty
    metric curves. The user benefits from the snapshot's prior
    training as a starting point but the new run is judged on its own
    merits.

    See ``juniper-ml/notes/PHASE_6E_SPRINT_B_DESIGN.md`` §2.4 for the
    full reset scope and §9 for the field-by-field table.

    Response shape mirrors ``/restore`` (snapshot_id + training_params),
    differing only in the ``operation`` field. The unified response
    shape with ``fsm_state`` and ``time_index`` lands in B-4 alongside
    the Investigating-state work; B-1 stays close to the existing
    ``/restore`` shape so clients written against either endpoint can
    transition between them with minimal changes.
    """
    _validate_snapshot_id(snapshot_id, client=request.client.host if request.client else None)
    lifecycle = _get_lifecycle(request)
    # PERF-CC-01: HDF5 I/O off the event loop, same as the other
    # snapshot routes. The reset side-effects (FSM, monitor, training_state)
    # are quick attribute writes — no need to fan out further.
    success = await asyncio.to_thread(lifecycle.restore_for_retrain, snapshot_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found or failed to load")
    try:
        params = lifecycle.get_training_params()
    except Exception:
        logger.exception("retrain_from_snapshot: get_training_params failed after successful restore_for_retrain")
        params = None
    payload: dict = {"snapshot_id": snapshot_id, "operation": "retrain", "status": "ready"}
    if params is not None:
        payload["training_params"] = params
    return success_response(payload)


@router.post("/{snapshot_id}/resume")
async def resume_snapshot(request: Request, snapshot_id: str) -> dict:
    """Restore a snapshot and prepare to continue training (CAN-015b).

    Phase 6E Sprint B B-2. The lifecycle loads the snapshot identically
    to ``/restore`` (preserving weights, topology, all A-5
    meta-parameters, AND the training history) then transitions to the
    ``RESUME_READY`` FSM state. The next ``POST /v1/training/start``
    extends the existing history arrays from the snapshot's terminal
    epoch rather than starting fresh, and the auto-snap-best ratchet
    keeps its prior accuracy ceiling.

    Response includes ``resume_point_epoch`` — the snapshot's terminal
    epoch count — so a tuning UI can render a visual boundary in the
    metrics-curve component (vertical dashed line separating the
    pre-resume read-only history from the new training that appends
    past it).

    Rejected with 409 if training is currently active (Started /
    Paused) — the user must stop training before resuming.

    See ``juniper-ml/notes/PHASE_6E_SPRINT_B_DESIGN.md`` §2.3.
    """
    _validate_snapshot_id(snapshot_id, client=request.client.host if request.client else None)
    lifecycle = _get_lifecycle(request)
    # Pre-flight check on FSM state. The lifecycle method also checks
    # but we surface the conflict as 409 at the route boundary for a
    # clean client error rather than a generic 404.
    if lifecycle.state_machine.is_started() or lifecycle.state_machine.is_paused():
        raise HTTPException(status_code=409, detail=f"Cannot resume from snapshot while training is {lifecycle.state_machine.status.name}")
    # PERF-CC-01: HDF5 I/O off the event loop, same as the other snapshot routes.
    success = await asyncio.to_thread(lifecycle.resume_from_snapshot, snapshot_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found or failed to load")
    try:
        params = lifecycle.get_training_params()
    except Exception:
        logger.exception("resume_snapshot: get_training_params failed after successful resume_from_snapshot")
        params = None
    payload: dict = {
        "snapshot_id": snapshot_id,
        "operation": "resume",
        "status": "ready",
        "resume_point_epoch": lifecycle._resume_point_epoch,
    }
    if params is not None:
        payload["training_params"] = params
    return success_response(payload)
