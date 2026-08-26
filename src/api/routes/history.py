"""Training history read routes (P2-2 Follow-up B, Issue #3).

Exposes the network's training-history event lists over REST so canopy can
render timeline markers without fetching a full HDF5 snapshot. Today only
the ``dataset_swap`` event list is surfaced — other history keys
(``train_loss``, ``hidden_units_added``, etc.) are still snapshot-only.
Future history-fetch routes belong in this module.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from api.models.common import ResponseEnvelope, success_response

logger = logging.getLogger("juniper_cascor.api.routes.history")

router = APIRouter(prefix="/history", tags=["history"])


def _get_lifecycle(request: Request):
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise HTTPException(status_code=503, detail="Lifecycle manager not initialized")
    return lifecycle


@router.get("/dataset_swaps", operation_id="get_dataset_swap_events", response_model=ResponseEnvelope)
async def get_dataset_swap_events(
    request: Request,
    since: Optional[str] = Query(
        default=None,
        description="Optional ISO-8601 timestamp. When set, only events with timestamp strictly greater than this value are returned. Lexicographic compare is correct for the canonical UTC format the recorder uses.",
    ),
) -> dict:
    """Return the network's ``dataset_swap`` history events.

    Canopy P2-7 uses this to render timeline markers + paired-diff
    affordances without joining against the full HDF5 snapshot. The
    ``?since=`` filter lets long-running clients poll only for new
    events without redownloading the full list each tick.

    Status codes:
      200 — list (possibly empty) returned in chronological order.
    """
    lifecycle = _get_lifecycle(request)
    events = lifecycle.get_dataset_swap_events(since=since)
    return success_response({"events": events})
