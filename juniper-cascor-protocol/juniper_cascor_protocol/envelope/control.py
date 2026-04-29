"""Pydantic models for ``/ws/control`` envelopes.

Two typed envelopes (plus :class:`ChunkedMessageEnvelope` which is shared
with ``/ws/training``):

- :class:`CommandResponseEnvelope`        ``type = "command_response"``
- :class:`ConnectionEstablishedEnvelope`  ``type = "connection_established"``

``command_response`` carries the result of a control command (start /
stop / pause / resume / set_params / etc.) and is the only frame in the
ecosystem with a flat ``data`` whose own fields are part of the wire
contract (``command``, ``status``, optional ``command_id`` / ``result``
/ ``error`` / ``code``). Per D-03 it never carries ``seq`` — the
control channel has no replay buffer.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from juniper_cascor_protocol.envelope.base import BaseEnvelope


class CommandResponseData(BaseModel):
    """Strongly-typed payload for :class:`CommandResponseEnvelope`."""

    command: str = Field(..., description="Command name being acknowledged (start/stop/pause/resume/set_params/etc.).")
    status: str = Field(..., description="Outcome ('success' / 'error' / 'timeout' / 'queued').")
    command_id: str | None = Field(default=None, description="Caller-supplied correlation id.")
    result: dict[str, Any] | None = Field(default=None, description="Command-specific result payload on success.")
    error: str | None = Field(default=None, description="Human-readable error string on failure.")
    code: str | None = Field(default=None, description="Stable error code (e.g. 'unknown_command') for programmatic clients.")


class CommandResponseEnvelope(BaseEnvelope):
    """``/ws/control`` ``command_response`` frame.

    No ``seq`` field — the control channel has no replay buffer.
    Per-command timeouts (Phase D §S10) emit this envelope with
    ``data.status = "error"`` and ``data.error`` populated.
    """

    type: Literal["command_response"] = "command_response"
    data: CommandResponseData


class ConnectionEstablishedData(BaseModel):
    """Strongly-typed payload for :class:`ConnectionEstablishedEnvelope`.

    The handshake message a server sends when a new control connection
    completes auth. ``protocol_version`` is reserved for future schema
    negotiation but currently fixed; consumers should accept any value
    they recognize and ignore unknown extras (handled by Pydantic's
    ``extra="allow"`` on the base envelope).
    """

    server_version: str | None = Field(default=None, description="Cascor server semver string.")
    protocol_version: str | None = Field(default=None, description="Reserved for future protocol negotiation.")


class ConnectionEstablishedEnvelope(BaseEnvelope):
    """``/ws/control`` ``connection_established`` frame — handshake.

    Sent by the server when a new control connection completes auth.
    Body fields are advisory and not load-bearing for the current
    protocol; future versions may pin them.
    """

    type: Literal["connection_established"] = "connection_established"
    data: ConnectionEstablishedData = Field(default_factory=ConnectionEstablishedData)
