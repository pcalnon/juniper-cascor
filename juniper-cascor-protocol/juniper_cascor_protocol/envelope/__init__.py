"""Pydantic envelope schemas for ``/ws/training`` and ``/ws/control``.

Importing this subpackage triggers Pydantic model construction. The
worker subpackage (:mod:`juniper_cascor_protocol.worker`) is structured
so it never imports this module — workers stay Pydantic-free at runtime
per the METRICS-MON R2 exit-gate decision.
"""

from juniper_cascor_protocol.envelope.base import BaseEnvelope, UnknownEnvelope
from juniper_cascor_protocol.envelope.control import (
    CommandResponseData,
    CommandResponseEnvelope,
    ConnectionEstablishedData,
    ConnectionEstablishedEnvelope,
)
from juniper_cascor_protocol.envelope.training import (
    CandidateProgressEnvelope,
    CascadeAddEnvelope,
    ChunkedMessageData,
    ChunkedMessageEnvelope,
    EventEnvelope,
    InitialMetricsData,
    InitialMetricsEnvelope,
    MetricsEnvelope,
    StateEnvelope,
    TopologyEnvelope,
)
from juniper_cascor_protocol.envelope.validate import (
    KNOWN_ENVELOPES,
    UNKNOWN_TYPE_BUDGET,
    UNMATCHED_TYPE_LABEL,
    KnownEnvelope,
    reset_unknown_label_state,
    validate_envelope,
)

__all__ = [
    # Base
    "BaseEnvelope",
    "UnknownEnvelope",
    # Training envelopes
    "MetricsEnvelope",
    "StateEnvelope",
    "TopologyEnvelope",
    "EventEnvelope",
    "CascadeAddEnvelope",
    "CandidateProgressEnvelope",
    "InitialMetricsEnvelope",
    "InitialMetricsData",
    "ChunkedMessageEnvelope",
    "ChunkedMessageData",
    # Control envelopes
    "CommandResponseEnvelope",
    "CommandResponseData",
    "ConnectionEstablishedEnvelope",
    "ConnectionEstablishedData",
    # Validation entrypoint
    "validate_envelope",
    "KnownEnvelope",
    "KNOWN_ENVELOPES",
    "UNMATCHED_TYPE_LABEL",
    "UNKNOWN_TYPE_BUDGET",
    "reset_unknown_label_state",
]
