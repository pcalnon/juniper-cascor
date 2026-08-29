"""Admin routes.

Currently contains only the experimental-functions gate that controls Phase 2
features like ``swap_dataset_live`` (ISSUE_3_PHASE_2_LIVE_DATASET_SWAP §3.3).
The server-side state of this gate is authoritative per F2.10 — a stale
canopy frontend toggle alone cannot bypass it.

Authorisation: the admin route is access-controlled separately via the
existing ``JUNIPER_DATA_API_KEY`` middleware (the same gate that protects
other state-mutating endpoints). The lifecycle method itself does not
re-validate, so any future direct caller must ensure equivalent gating.
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from api.models.common import ResponseEnvelope, success_response
from api.models.training import ExperimentalFunctionsToggleRequest

logger = logging.getLogger("juniper_cascor.api.routes.admin")

router = APIRouter(prefix="/admin", tags=["admin"])


def _get_lifecycle(request: Request):
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise HTTPException(status_code=503, detail="Lifecycle manager not initialized")
    return lifecycle


@router.get("/experimental_functions", operation_id="get_experimental_functions", response_model=ResponseEnvelope)
async def get_experimental_functions(request: Request) -> dict:
    """Report whether the experimental-functions gate is currently open."""
    lifecycle = _get_lifecycle(request)
    return success_response({"enabled": lifecycle.get_experimental_functions()})


@router.post("/experimental_functions", operation_id="set_experimental_functions", response_model=ResponseEnvelope)
async def set_experimental_functions(request: Request, body: ExperimentalFunctionsToggleRequest) -> dict:
    """Open or close the experimental-functions gate.

    Canopy's ``Enable Experimental Functions`` toggle POSTs here; if this
    returns ``enabled=false`` the UI must revert the local toggle (F2.10).
    """
    lifecycle = _get_lifecycle(request)
    return success_response(lifecycle.set_experimental_functions(body.enabled))
