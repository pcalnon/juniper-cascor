"""Pydantic models for ``/ws/training`` envelopes.

Eight typed envelopes mirror :mod:`juniper-cascor/src/api/websocket/messages`:

- :class:`MetricsEnvelope`              ``type = "metrics"``
- :class:`StateEnvelope`                ``type = "state"``
- :class:`TopologyEnvelope`             ``type = "topology"``
- :class:`EventEnvelope`                ``type = "event"``
- :class:`CascadeAddEnvelope`           ``type = "cascade_add"``
- :class:`CandidateProgressEnvelope`    ``type = "candidate_progress"``
- :class:`InitialMetricsEnvelope`       ``type = "initial_metrics"``
- :class:`ChunkedMessageEnvelope`       ``type = "chunked_message"``

The free-form ``data`` payloads (metrics/state/topology/event/cascade_add/
candidate_progress) keep the existing ``dict[str, Any]`` shape so the
protocol package does not couple to cascor's training internals. The
two structured payloads (``initial_metrics``, ``chunked_message``) are
strongly-typed because their wire format is part of the cross-repo
contract (GAP-WS-16, GAP-WS-18).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from juniper_cascor_protocol.envelope.base import BaseEnvelope


class MetricsEnvelope(BaseEnvelope):
    """``/ws/training`` ``metrics`` frame — periodic training metrics broadcast."""

    type: Literal["metrics"] = "metrics"
    data: dict[str, Any] = Field(default_factory=dict)


class StateEnvelope(BaseEnvelope):
    """``/ws/training`` ``state`` frame — training state snapshot."""

    type: Literal["state"] = "state"
    data: dict[str, Any] = Field(default_factory=dict)


class TopologyEnvelope(BaseEnvelope):
    """``/ws/training`` ``topology`` frame — network topology update."""

    type: Literal["topology"] = "topology"
    data: dict[str, Any] = Field(default_factory=dict)


class EventEnvelope(BaseEnvelope):
    """``/ws/training`` ``event`` frame — generic training event."""

    type: Literal["event"] = "event"
    data: dict[str, Any] = Field(default_factory=dict)


class CascadeAddEnvelope(BaseEnvelope):
    """``/ws/training`` ``cascade_add`` frame — cascade unit addition."""

    type: Literal["cascade_add"] = "cascade_add"
    data: dict[str, Any] = Field(default_factory=dict)


class CandidateProgressEnvelope(BaseEnvelope):
    """``/ws/training`` ``candidate_progress`` frame — candidate training progress."""

    type: Literal["candidate_progress"] = "candidate_progress"
    data: dict[str, Any] = Field(default_factory=dict)


class InitialMetricsData(BaseModel):
    """Strongly-typed payload for :class:`InitialMetricsEnvelope`.

    Carries up to N most-recent metrics so a freshly-connected client
    can paint its time-series chart without an immediate REST poll.
    See GAP-WS-16 in juniper-cascor.
    """

    metrics: list[Any] = Field(default_factory=list, description="Recent metric snapshots, oldest first.")
    count: int = Field(..., description="Length of ``metrics`` (denormalized for clients on small wire budgets).")
    current_seq: int = Field(default=0, description="Last broadcast seq the manager had assigned at send time.")


class InitialMetricsEnvelope(BaseEnvelope):
    """``/ws/training`` ``initial_metrics`` frame — burst on fresh connect (GAP-WS-16)."""

    type: Literal["initial_metrics"] = "initial_metrics"
    data: InitialMetricsData


class ChunkedMessageData(BaseModel):
    """Strongly-typed payload for :class:`ChunkedMessageEnvelope`.

    One slice of a fragmented oversized broadcast. See GAP-WS-18 in
    juniper-cascor for the reassembly contract.
    """

    chunk_id: str = Field(..., description="UUID4 identifying the logical message all chunks belong to.")
    chunk_index: int = Field(..., ge=0, description="0-based position of this chunk within the message.")
    total_chunks: int = Field(..., ge=1, description="Total number of chunks the logical message was split into.")
    original_type: str = Field(..., description="``type`` field of the pre-chunked message.")
    payload: str = Field(..., description="A slice of the JSON-serialized original message as a string.")


class ChunkedMessageEnvelope(BaseEnvelope):
    """``/ws/training`` (or ``/ws/control``) ``chunked_message`` frame — GAP-WS-18 fragmentation."""

    type: Literal["chunked_message"] = "chunked_message"
    data: ChunkedMessageData
