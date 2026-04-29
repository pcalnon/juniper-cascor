"""Standardized WebSocket message builders.

All WebSocket messages follow the format:
{
    "type": "<message_type>",
    "timestamp": <unix_timestamp>,
    "data": { ... }
}

Broadcast messages additionally include:
- "seq": monotonically increasing sequence number (assigned by manager)
- "emitted_at_monotonic": monotonic clock timestamp for latency instrumentation

Compatible with juniper-canopy's WebSocket message protocol.
"""

import time
from typing import Any, Dict, Optional


def _build_envelope(
    msg_type: str,
    data: Dict[str, Any],
    *,
    seq: Optional[int] = None,
    emitted_at_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Build standard message envelope with optional seq and monotonic timestamp.

    Args:
        msg_type: Message type identifier.
        data: Message payload.
        seq: Sequence number assigned by WebSocketManager (broadcast messages only).
        emitted_at_monotonic: Monotonic clock timestamp for latency instrumentation.

    Returns:
        Envelope dict. Fields ``seq`` and ``emitted_at_monotonic`` are only
        present when explicitly provided (backward-compatible with legacy clients).
    """
    envelope: Dict[str, Any] = {
        "type": msg_type,
        "timestamp": time.time(),
        "data": data,
    }
    if seq is not None:
        envelope["seq"] = seq
    if emitted_at_monotonic is not None:
        envelope["emitted_at_monotonic"] = emitted_at_monotonic
    return envelope


def create_metrics_message(
    data: Dict[str, Any],
    *,
    seq: Optional[int] = None,
    emitted_at_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a metrics update message."""
    return _build_envelope("metrics", data, seq=seq, emitted_at_monotonic=emitted_at_monotonic)


def create_state_message(
    data: Dict[str, Any],
    *,
    seq: Optional[int] = None,
    emitted_at_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a training state update message."""
    return _build_envelope("state", data, seq=seq, emitted_at_monotonic=emitted_at_monotonic)


def create_topology_message(
    data: Dict[str, Any],
    *,
    seq: Optional[int] = None,
    emitted_at_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a network topology message."""
    return _build_envelope("topology", data, seq=seq, emitted_at_monotonic=emitted_at_monotonic)


def create_event_message(
    data: Dict[str, Any],
    *,
    seq: Optional[int] = None,
    emitted_at_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a training event message."""
    return _build_envelope("event", data, seq=seq, emitted_at_monotonic=emitted_at_monotonic)


def create_cascade_add_message(
    data: Dict[str, Any],
    *,
    seq: Optional[int] = None,
    emitted_at_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a cascade unit addition message."""
    return _build_envelope("cascade_add", data, seq=seq, emitted_at_monotonic=emitted_at_monotonic)


def create_initial_metrics_message(
    metrics: list,
    *,
    current_seq: Optional[int] = None,
) -> Dict[str, Any]:
    """GAP-WS-16: build the initial_metrics burst sent on fresh WS connect.

    Carries up to N most-recent metrics so a freshly-connected client can
    paint its time-series chart without an immediate REST poll.

    The envelope is a personal (non-broadcast) message — it carries no
    ``seq`` of its own, but ``data.current_seq`` reflects the last broadcast
    seq the manager had assigned at send time so the client knows where the
    live stream picks up.
    """
    return _build_envelope(
        "initial_metrics",
        {
            "metrics": metrics,
            "count": len(metrics),
            "current_seq": current_seq if current_seq is not None else 0,
        },
    )


def create_candidate_progress_message(
    data: Dict[str, Any],
    *,
    seq: Optional[int] = None,
    emitted_at_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a candidate training progress message."""
    return _build_envelope("candidate_progress", data, seq=seq, emitted_at_monotonic=emitted_at_monotonic)


def create_control_ack_message(
    command: str,
    status: str,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    *,
    command_id: Optional[str] = None,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a control command acknowledgment message.

    Note: command_response messages have NO ``seq`` field (D-03 canonical).
    The /ws/control channel has no replay buffer.
    """
    msg: Dict[str, Any] = {
        "type": "command_response",
        "timestamp": time.time(),
        "data": {
            "command": command,
            "status": status,
        },
    }
    if command_id is not None:
        msg["data"]["command_id"] = command_id
    if data:
        msg["data"]["result"] = data
    if error:
        msg["data"]["error"] = error
    if code is not None:
        msg["data"]["code"] = code
    return msg
