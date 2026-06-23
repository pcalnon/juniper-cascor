"""Network management routes."""

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger("juniper_cascor.api.routes.network")

from api.models.common import success_response
from api.models.network import AddHiddenUnitRequest, NetworkCreateRequest, PatchWeightsRequest

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
        raise HTTPException(status_code=409, detail="Network cannot be created in the current state") from e


@router.get("")
async def get_network(request: Request) -> dict:
    """Get current network info."""
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_model():
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
        raise HTTPException(status_code=409, detail="Network cannot be deleted in the current state") from e


@router.get("/topology")
async def get_topology(request: Request) -> dict:
    """Get network topology for visualization."""
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_model():
        raise HTTPException(status_code=404, detail="No network created")
    topology = lifecycle.get_topology()
    if topology is None:
        raise HTTPException(status_code=500, detail="Failed to extract topology")
    return success_response(topology)


@router.get("/stats")
async def get_stats(request: Request) -> dict:
    """Get network weight statistics."""
    lifecycle = _get_lifecycle(request)
    if not lifecycle.has_model():
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


@router.post("/hidden-units")
async def add_hidden_unit(request: Request, body: AddHiddenUnitRequest) -> dict:
    """CAN-015h-2: append a fresh hidden unit at the cascade tail.

    FSM-gated to ``Investigating``. The new unit's output-layer
    column is initialized to **zero** so it contributes nothing
    until the user re-trains or patches the output layer.

    Status codes:

    - 200 — unit appended; response body carries the new
      ``unit_index``, ``num_hidden_units``, ``operation``,
      ``fsm_state``.
    - 400 — bad shape (weight vector length does not match
      input_size + num_existing_hidden_units).
    - 404 — no network created.
    - 409 — FSM not Investigating, or network at
      ``max_hidden_units`` cap.
    - 422 — NaN/Inf in weights or bias / unknown activation.

    Plan: ``notes/PHASE_6E_DEFERRED_CAN-015GH_DESIGN.md`` §"Endpoint
    design / 2. POST /v1/network/hidden-units" (juniper-ml).
    """
    lifecycle = _get_lifecycle(request)
    result = lifecycle.add_hidden_unit_manual(
        weights=body.weights,
        bias=body.bias,
        activation=body.activation,
    )
    status = result.get("status")
    detail = result.get("detail", "add failed")

    if status == lifecycle._ADD_OK:
        info = lifecycle.get_network_info()
        info.update(
            {
                "operation": "add_hidden_unit",
                "fsm_state": lifecycle.state_machine.status.name,
                "unit_index": result.get("unit_index"),
                "num_hidden_units": result.get("num_hidden_units"),
            }
        )
        return success_response(info)
    if status == lifecycle._ADD_NO_NETWORK:
        raise HTTPException(status_code=404, detail=detail)
    if status in (lifecycle._ADD_FSM_REJECTED, lifecycle._ADD_AT_CAP):
        raise HTTPException(status_code=409, detail=detail)
    if status in (lifecycle._ADD_NAN_INF, lifecycle._ADD_BAD_ACTIVATION):
        raise HTTPException(status_code=422, detail=detail)
    if status == lifecycle._ADD_BAD_SHAPE:
        raise HTTPException(status_code=400, detail=detail)
    logger.error("add_hidden_unit_manual: unmapped status %r", status)
    raise HTTPException(status_code=500, detail="add_hidden_unit_manual returned an unexpected status")


@router.delete("/hidden-units/{idx}")
async def delete_hidden_unit(request: Request, idx: int) -> dict:
    """CAN-015h-3: remove the hidden unit at ``idx`` with cascade rebuild.

    FSM-gated to ``Investigating``. Subsequent units shift down by
    one, and each has its weight at the deleted unit's input
    position dropped (so the cascade-input width matches the new
    cascade position). The output layer's corresponding column is
    removed; the optimizer is dropped.

    Status codes:

    - 200 — unit removed; response body carries
      ``removed_index``, ``num_hidden_units``, ``operation``,
      ``fsm_state``.
    - 404 — no network created OR ``idx`` out of range.
    - 409 — FSM not Investigating.

    Plan: ``notes/PHASE_6E_DEFERRED_CAN-015GH_DESIGN.md`` §"Endpoint
    design / 3. DELETE /v1/network/hidden-units/{idx}" (juniper-ml).
    """
    lifecycle = _get_lifecycle(request)
    result = lifecycle.remove_hidden_unit_manual(idx=idx)
    status = result.get("status")
    detail = result.get("detail", "remove failed")

    if status == lifecycle._REMOVE_OK:
        info = lifecycle.get_network_info()
        info.update(
            {
                "operation": "remove_hidden_unit",
                "fsm_state": lifecycle.state_machine.status.name,
                "removed_index": result.get("removed_index"),
                "num_hidden_units": result.get("num_hidden_units"),
            }
        )
        return success_response(info)
    if status in (lifecycle._REMOVE_NO_NETWORK, lifecycle._REMOVE_OUT_OF_RANGE):
        raise HTTPException(status_code=404, detail=detail)
    if status == lifecycle._REMOVE_FSM_REJECTED:
        raise HTTPException(status_code=409, detail=detail)
    logger.error("remove_hidden_unit_manual: unmapped status %r", status)
    raise HTTPException(status_code=500, detail="remove_hidden_unit_manual returned an unexpected status")
