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


def _compute_snapshot_window(lifecycle) -> dict:
    """CAN-015d (Phase 6E Sprint B B-4): compute the snapshot window
    metadata for the unified response shape's ``time_index`` block.

    The window is derived from the longest history array on the loaded
    network — same source as ``resume_from_snapshot``'s
    ``_resume_point_epoch`` calculation. ``start_epoch`` is always 0
    (snapshots can't yet represent a model that has skipped epochs);
    ``end_epoch`` is the count of epochs the loaded network has trained
    for. Falls back to ``{0, 0}`` when the network has no history.
    """
    network = getattr(lifecycle, "network", None)
    if network is None:
        return {"start_epoch": 0, "end_epoch": 0}
    history = getattr(network, "history", None)
    end = 0
    if isinstance(history, dict):
        # Match the lifecycle's _NETWORK_HISTORY_KEYS scope so the route
        # surface stays consistent with what Resume / Retrain operate on.
        for key in ("train_loss", "value_loss", "train_accuracy", "value_accuracy"):
            series = history.get(key, ())
            try:
                end = max(end, len(series))
            except TypeError:
                continue
    return {"start_epoch": 0, "end_epoch": end}


def _build_unified_payload(
    lifecycle,
    snapshot_id: str,
    operation: str,
    time_index_default,
    *,
    extra: dict | None = None,
) -> dict:
    """CAN-015d (Phase 6E Sprint B B-4): assemble the unified response
    body for the four snapshot operation endpoints.

    Per ``notes/PHASE_6E_SPRINT_B_DESIGN.md`` §3, every snapshot
    operation returns the same shape — ``snapshot_id`` + ``operation``
    + ``fsm_state`` + ``time_index`` + ``training_params``. Fields like
    ``status`` and ``resume_point_epoch`` (added pre-B-4 by B-1 and B-2
    respectively) are preserved as a strict superset for backward
    compatibility — existing canopy clients keying off them keep
    working.

    ``time_index_default`` is the operation-specific default position
    in the snapshot's narrative: ``"end"`` for Restore / Resume,
    ``0`` for Retrain (history reset), ``"start"`` for Replay (B-3,
    not yet shipped).
    """
    fsm_state = lifecycle.state_machine.status.name
    snapshot_window = _compute_snapshot_window(lifecycle)
    payload: dict = {
        "snapshot_id": snapshot_id,
        "operation": operation,
        "fsm_state": fsm_state,
        "time_index": {
            "default": time_index_default,
            "snapshot_window": snapshot_window,
        },
    }
    try:
        payload["training_params"] = lifecycle.get_training_params()
    except Exception:
        # Defensive: surfacing params is best-effort. A failure here
        # should not undo a successful snapshot operation.
        logger.exception("snapshots: get_training_params failed after successful operation %r", operation)
    if extra:
        payload.update(extra)
    return payload


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
    """Restore a network from a snapshot for inspection and modification.

    CAN-015d (Phase 6E Sprint B B-4): Restore is now an explicit
    inspection / modification mode rather than an implicit "load + can
    train next" shortcut. The lifecycle transitions to the new
    ``Investigating`` FSM state which rejects ``start_training`` /
    ``pause_training`` / ``resume_training`` until the user explicitly
    invokes ``/retrain`` or ``/resume``. The user can edit
    meta-params via ``PATCH /v1/training/params``, replace the dataset,
    and re-snapshot the modified state — all of which are permitted in
    ``Investigating``.

    Rejected with 409 if training is currently active (Started /
    Paused) — same FSM-guard contract as Resume / Retrain.

    CAN-014 (Sprint A-5): response includes the post-restore
    ``training_params`` so a tuning UI can reconcile state. CAN-015d
    further unifies the response shape with ``operation`` /
    ``fsm_state`` / ``time_index``. The pre-existing ``status:
    "restored"`` field is retained as a strict superset so existing
    canopy clients keying off it keep working.

    See ``juniper-ml/notes/PHASE_6E_SPRINT_B_DESIGN.md`` §2.1, §3.
    """
    _validate_snapshot_id(snapshot_id, client=request.client.host if request.client else None)
    lifecycle = _get_lifecycle(request)
    # Pre-flight FSM check — surface 409 at the route boundary rather
    # than letting the lifecycle's own check map to a generic 404.
    if lifecycle.state_machine.is_started() or lifecycle.state_machine.is_paused():
        raise HTTPException(status_code=409, detail=f"Cannot restore from snapshot while training is {lifecycle.state_machine.status.name}")
    # PERF-CC-01: serializer.load_network is synchronous HDF5 I/O. Run it
    # off the event loop so concurrent requests aren't blocked while the
    # snapshot is being read.
    success = await asyncio.to_thread(lifecycle.load_snapshot, snapshot_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found or failed to load")
    payload = _build_unified_payload(
        lifecycle,
        snapshot_id,
        operation="restore",
        time_index_default="end",
        # Strict-superset backward-compatibility: keep the pre-B-4
        # ``status: "restored"`` field so existing clients keep working.
        extra={"status": "restored"},
    )
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
    payload = _build_unified_payload(
        lifecycle,
        snapshot_id,
        operation="retrain",
        # Retrain resets history / counters to 0, so the time index is 0.
        time_index_default=0,
        # Strict-superset backward-compatibility for the B-1 response shape.
        extra={"status": "ready"},
    )
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
    payload = _build_unified_payload(
        lifecycle,
        snapshot_id,
        operation="resume",
        # Resume lands the user at the snapshot's terminal epoch.
        time_index_default="end",
        # Strict-superset backward-compatibility for the B-2 response shape.
        extra={
            "status": "ready",
            "resume_point_epoch": lifecycle._resume_point_epoch,
        },
    )
    return success_response(payload)
