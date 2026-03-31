"""Snapshot management routes."""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from api.models.common import success_response

logger = logging.getLogger("juniper_cascor.api.routes.snapshots")

router = APIRouter(prefix="/snapshots", tags=["snapshots"])


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
    result = lifecycle.save_snapshot(description=description)
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
    lifecycle = _get_lifecycle(request)
    result = lifecycle.get_snapshot(snapshot_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found")
    return success_response(result)


@router.post("/{snapshot_id}/restore")
async def restore_snapshot(request: Request, snapshot_id: str) -> dict:
    """Restore a network from a snapshot."""
    lifecycle = _get_lifecycle(request)
    success = lifecycle.load_snapshot(snapshot_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found or failed to load")
    return success_response({"snapshot_id": snapshot_id, "status": "restored"})
