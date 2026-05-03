"""Network management routes."""

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger("juniper_cascor.api.routes.network")

from api.models.common import success_response
from api.models.network import NetworkCreateRequest, PatchWeightsRequest

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


@router.patch("/weights")
async def patch_weights(request: Request, body: PatchWeightsRequest) -> dict:
    """CAN-015h-1: surgically rewrite a single parameter group.

    FSM-gated to ``Investigating`` (entered via ``/restore``).
    Returns 200 + the post-patch network info on success; status
    codes per the design plan §"Endpoint design / 1. PATCH":

    - 200 — patch applied
    - 400 — shape mismatch / unknown target / unknown field
    - 404 — no network created / hidden_unit_index out of range
    - 409 — FSM not Investigating
    - 422 — NaN/Inf or untensorable values

    Plan: ``notes/PHASE_6E_DEFERRED_CAN-015GH_DESIGN.md`` §"Endpoint
    design / 1. PATCH /v1/network/weights" (juniper-ml).
    """
    lifecycle = _get_lifecycle(request)
    result = lifecycle.patch_weights(
        target=body.target,
        field=body.field,
        values=body.values,
        hidden_unit_index=body.hidden_unit_index,
        dtype=body.dtype,
    )
    status = result.get("status")
    detail = result.get("detail", "patch failed")

    if status == lifecycle._PATCH_OK:
        info = lifecycle.get_network_info()
        info["operation"] = "patch_weights"
        info["fsm_state"] = lifecycle.state_machine.status.name
        return success_response(info)
    if status == lifecycle._PATCH_NO_NETWORK:
        raise HTTPException(status_code=404, detail=detail)
    if status == lifecycle._PATCH_FSM_REJECTED:
        raise HTTPException(status_code=409, detail=detail)
    if status == lifecycle._PATCH_HIDDEN_UNIT_OUT_OF_RANGE:
        raise HTTPException(status_code=404, detail=detail)
    if status == lifecycle._PATCH_NAN_INF:
        raise HTTPException(status_code=422, detail=detail)
    if status in (lifecycle._PATCH_BAD_TARGET, lifecycle._PATCH_SHAPE_MISMATCH):
        raise HTTPException(status_code=400, detail=detail)
    # Defensive — unmapped sentinel from the lifecycle layer.
    logger.error("patch_weights: unmapped status %r", status)
    raise HTTPException(status_code=500, detail="patch_weights returned an unexpected status")
