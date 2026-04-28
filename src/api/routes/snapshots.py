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
    """Restore a network from a snapshot."""
    _validate_snapshot_id(snapshot_id, client=request.client.host if request.client else None)
    lifecycle = _get_lifecycle(request)
    # PERF-CC-01: serializer.load_network is synchronous HDF5 I/O. Run it
    # off the event loop so concurrent requests aren't blocked while the
    # snapshot is being read.
    success = await asyncio.to_thread(lifecycle.load_snapshot, snapshot_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found or failed to load")
    return success_response({"snapshot_id": snapshot_id, "status": "restored"})
