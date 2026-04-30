"""Standardized WebSocket message builders.

METRICS-MON R2.2.2 / seed-05: the canonical envelope schemas live in
:mod:`juniper_cascor_protocol.envelope`. This module preserves the
existing ``create_*_message`` helper signatures for backwards
compatibility — every helper now constructs the corresponding
Pydantic envelope and dumps it to a wire-format dict via
``model.model_dump(exclude_none=True)``. Output is byte-for-byte
identical to the pre-migration implementation; the wire-compat
snapshot test in ``src/tests/unit/api/test_messages_wire_compat.py``
pins this contract.

All WebSocket messages follow the format:

    {
        "type": "<message_type>",
        "timestamp": <unix_timestamp>,
        "data": { ... }
    }

Broadcast messages additionally include:

- ``"seq"``: monotonically increasing sequence number (assigned by manager)
- ``"emitted_at_monotonic"``: monotonic clock timestamp for latency instrumentation

Compatible with juniper-canopy's WebSocket message protocol.

New code should prefer ``from juniper_cascor_protocol.envelope import …``
to make the dependency on the shared schema package explicit.
"""

import time
from typing import Any, Dict, Optional

from juniper_cascor_protocol.envelope import CandidateProgressEnvelope, CascadeAddEnvelope, ChunkedMessageEnvelope, CommandResponseEnvelope, EventEnvelope, InitialMetricsEnvelope, MetricsEnvelope, StateEnvelope, TopologyEnvelope


def _dump_envelope(envelope: Any) -> Dict[str, Any]:
    """Serialize a Pydantic envelope to a wire-format dict.

    ``exclude_none=True`` matches the pre-migration behaviour where
    optional ``seq`` / ``emitted_at_monotonic`` fields were omitted
    when the manager had not assigned them.
    """
    return envelope.model_dump(exclude_none=True)


def create_metrics_message(
    data: Dict[str, Any],
    *,
    seq: Optional[int] = None,
    emitted_at_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a metrics update message."""
    return _dump_envelope(MetricsEnvelope(timestamp=time.time(), data=data, seq=seq, emitted_at_monotonic=emitted_at_monotonic))


def create_state_message(
    data: Dict[str, Any],
    *,
    seq: Optional[int] = None,
    emitted_at_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a training state update message."""
    return _dump_envelope(StateEnvelope(timestamp=time.time(), data=data, seq=seq, emitted_at_monotonic=emitted_at_monotonic))


def create_topology_message(
    data: Dict[str, Any],
    *,
    seq: Optional[int] = None,
    emitted_at_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a network topology message."""
    return _dump_envelope(TopologyEnvelope(timestamp=time.time(), data=data, seq=seq, emitted_at_monotonic=emitted_at_monotonic))


def create_event_message(
    data: Dict[str, Any],
    *,
    seq: Optional[int] = None,
    emitted_at_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a training event message."""
    return _dump_envelope(EventEnvelope(timestamp=time.time(), data=data, seq=seq, emitted_at_monotonic=emitted_at_monotonic))


def create_cascade_add_message(
    data: Dict[str, Any],
    *,
    seq: Optional[int] = None,
    emitted_at_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a cascade unit addition message."""
    return _dump_envelope(CascadeAddEnvelope(timestamp=time.time(), data=data, seq=seq, emitted_at_monotonic=emitted_at_monotonic))


def create_chunked_message(
    *,
    chunk_id: str,
    chunk_index: int,
    total_chunks: int,
    original_type: str,
    payload: str,
) -> Dict[str, Any]:
    """GAP-WS-18: build one chunk of a fragmented oversized message.

    Each chunk is itself a normal envelope and gets its own ``seq`` from
    the manager — chunks of one logical message land on consecutive seqs
    so the replay buffer reorders them naturally on resume.

    Args:
        chunk_id: UUID4 identifying the logical message all chunks belong to.
        chunk_index: 0-based position of this chunk within the message.
        total_chunks: Total number of chunks the logical message was split into.
        original_type: ``type`` field of the pre-chunked message (e.g., "topology").
        payload: A slice of the JSON-serialized original message as a string.
            The client concatenates payloads in chunk_index order and parses
            the result as JSON to reconstruct the original envelope.

    Returns:
        Envelope with type "chunked_message".
    """
    return _dump_envelope(
        ChunkedMessageEnvelope(
            timestamp=time.time(),
            data={
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "original_type": original_type,
                "payload": payload,
            },
        )
    )


def create_initial_metrics_message(
    metrics: list,
    *,
    current_seq: Optional[int] = None,
) -> Dict[str, Any]:
    """GAP-WS-16: build the initial_metrics burst sent on fresh WS connect.

    Carries up to N most-recent metrics so a freshly-connected client
    can paint its time-series chart without an immediate REST poll.

    The envelope is a personal (non-broadcast) message — it carries no
    ``seq`` of its own, but ``data.current_seq`` reflects the last broadcast
    seq the manager had assigned at send time so the client knows where the
    live stream picks up.
    """
    return _dump_envelope(
        InitialMetricsEnvelope(
            timestamp=time.time(),
            data={
                "metrics": metrics,
                "count": len(metrics),
                "current_seq": current_seq if current_seq is not None else 0,
            },
        )
    )


def create_candidate_progress_message(
    data: Dict[str, Any],
    *,
    seq: Optional[int] = None,
    emitted_at_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a candidate training progress message."""
    return _dump_envelope(CandidateProgressEnvelope(timestamp=time.time(), data=data, seq=seq, emitted_at_monotonic=emitted_at_monotonic))


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
    payload: Dict[str, Any] = {"command": command, "status": status}
    if command_id is not None:
        payload["command_id"] = command_id
    if data:
        payload["result"] = data
    if error:
        payload["error"] = error
    if code is not None:
        payload["code"] = code

    envelope = CommandResponseEnvelope(timestamp=time.time(), data=payload)
    # ``CommandResponseData`` declares ``command_id``/``result``/``error``/``code``
    # as ``Optional`` with ``None`` defaults; ``model_dump(exclude_none=True)``
    # drops the unset ones, matching the pre-migration dict-builder which only
    # included keys the caller had explicitly set.
    return _dump_envelope(envelope)
