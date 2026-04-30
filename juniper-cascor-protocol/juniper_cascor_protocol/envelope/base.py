"""Base envelope shape shared by every juniper-cascor broadcast/control frame.

The envelope is the ``{type, timestamp, data, seq?, emitted_at_monotonic?}``
JSON object emitted by the cascor server on ``/ws/training`` and
``/ws/control``. ``seq`` and ``emitted_at_monotonic`` are present only on
broadcast messages where the WebSocket manager assigns a monotonic
sequence number; control-channel messages omit both.

This module exposes:

- :class:`BaseEnvelope` — the abstract base every typed frame derives from.
- :class:`UnknownEnvelope` — a fallback used by :func:`validate_envelope`
  when a frame's ``type`` doesn't match any known typed envelope, so
  consumers can still observe the frame's wire shape.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BaseEnvelope(BaseModel):
    """Abstract base shared by every juniper-cascor JSON envelope.

    Concrete subclasses pin ``type`` to a literal value so a Pydantic
    discriminated union (see :func:`validate_envelope`) can dispatch by
    type. Field order matches the wire format the cascor server emits.
    """

    # ``extra="allow"`` keeps unknown top-level keys around so consumers can
    # surface server-side additions before the schema bumps; tests pin the
    # known set so silent additions still surface in CI.
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: str = Field(..., description="Message type identifier (e.g. 'metrics', 'command_response').")
    timestamp: float = Field(..., description="Unix epoch seconds when the server emitted the frame.")
    data: dict[str, Any] = Field(default_factory=dict, description="Type-specific payload.")
    seq: int | None = Field(default=None, description="Broadcast sequence number; absent on control-channel frames.")
    emitted_at_monotonic: float | None = Field(default=None, description="Server monotonic clock at emit time; broadcast-only.")


class UnknownEnvelope(BaseEnvelope):
    """Returned by :func:`validate_envelope` for frames whose ``type`` is
    not one of the known typed envelopes.

    Distinguished from :class:`BaseEnvelope` by class identity so consumers
    can match on ``isinstance(env, UnknownEnvelope)`` to drive the
    ``unrecognized_ws_frames_total`` counter without re-parsing the type
    string.
    """
