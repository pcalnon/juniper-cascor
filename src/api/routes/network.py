"""Network management routes."""

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger("juniper_cascor.api.routes.network")

from api.models.common import success_response
from api.models.network import NetworkCreateRequest

router = APIRouter(prefix="/network", tags=["network"])


def _get_lifecycle(request: Request):
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise HTTPException(status_code=503, detail="Lifecycle manager not initialized")
    return lifecycle


@router.post("")
async def create_network(request: Request, body: NetworkCreateRequest) -> dict:
    """Create a new CasCor network."""
    lifecycle = _get_lifecycle(request)
    try:
        info = lifecycle.create_network(**body.model_dump())
        return success_response(info)
    except RuntimeError as e:
        logger.debug("Create network failed: %s", e)
        raise HTTPException(status_code=409, detail="Network cannot be created in the current state")


@router.get("")
async def get_network(request: Request) -> dict:
    """Get current network info."""
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_network():
        raise HTTPException(status_code=404, detail="No network created")
    return success_response(lifecycle.get_network_info())


@router.delete("")
async def delete_network(request: Request) -> dict:
    """Delete the current network."""
    lifecycle = _get_lifecycle(request)
    try:
        lifecycle.delete_network()
        return success_response({"deleted": True})
    except RuntimeError as e:
        logger.debug("Delete network failed: %s", e)
        raise HTTPException(status_code=409, detail="Network cannot be deleted in the current state")


@router.get("/topology")
async def get_topology(request: Request) -> dict:
    """Get network topology for visualization."""
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_network():
        raise HTTPException(status_code=404, detail="No network created")
    topology = lifecycle.get_topology()
    if topology is None:
        raise HTTPException(status_code=500, detail="Failed to extract topology")
    return success_response(topology)


@router.get("/stats")
async def get_stats(request: Request) -> dict:
    """Get network weight statistics."""
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_network():
        raise HTTPException(status_code=404, detail="No network created")
    return success_response(lifecycle.get_statistics())
