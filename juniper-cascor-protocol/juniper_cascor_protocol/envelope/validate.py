"""Envelope validation entrypoint with R1.1 cardinality bound.

:func:`validate_envelope` is the consumer-facing helper: pass it a
``dict`` parsed from ``json.loads(raw_ws_frame)`` and it returns either
the typed Pydantic envelope (one of :data:`KNOWN_ENVELOPES`) or an
:class:`UnknownEnvelope` whose ``type`` has been collapsed to the
literal ``"_unmatched"`` once the per-process distinct-unknown-type
budget is exhausted.

The cardinality bound mirrors METRICS-MON R1.1 (see the cross-service
``UNMATCHED_ENDPOINT_LABEL`` in juniper-observability): unknown ``type``
strings are attacker-influenceable, so an unbounded label set on the
``unrecognized_ws_frames_total`` counter would blow up Prometheus
storage. The first :data:`UNKNOWN_TYPE_BUDGET` distinct unknown values
seen by a process are tracked with their original ``type`` string; all
subsequent unknowns collapse to ``"_unmatched"``.
"""

import threading
from typing import Any, Union

from pydantic import ValidationError

from juniper_cascor_protocol.envelope.base import BaseEnvelope, UnknownEnvelope
from juniper_cascor_protocol.envelope.control import CommandResponseEnvelope, ConnectionEstablishedEnvelope
from juniper_cascor_protocol.envelope.training import (
    CandidateProgressEnvelope,
    CascadeAddEnvelope,
    ChunkedMessageEnvelope,
    EventEnvelope,
    InitialMetricsEnvelope,
    MetricsEnvelope,
    StateEnvelope,
    TopologyEnvelope,
)

# Sentinel value used by :func:`validate_envelope` for unknown ``type``
# strings once the per-process budget is exhausted. Same character-for-
# character literal as juniper-observability's ``UNMATCHED_ENDPOINT_LABEL``
# so dashboards built off the HTTP cardinality bound work for the WS
# counter without relabeling.
UNMATCHED_TYPE_LABEL = "_unmatched"

# Per-process budget for distinct unknown type strings tracked verbatim.
# After this many distinct unknowns are seen, every further new unknown
# type collapses to :data:`UNMATCHED_TYPE_LABEL`. 16 is generous enough
# that legitimate typos (e.g. server-side rename) still surface in logs
# while bounded enough that an attacker spamming N distinct frame types
# cannot inflate the unrecognized-frame counter's label set.
UNKNOWN_TYPE_BUDGET = 16

# Mapping of known ``type`` literal strings to their Pydantic models.
# Iteration order is stable (dict preserves insertion order) so a
# reviewer can spot a missing type by reading the dict.
KNOWN_ENVELOPES: dict[str, type[BaseEnvelope]] = {
    # /ws/training
    "metrics": MetricsEnvelope,
    "state": StateEnvelope,
    "topology": TopologyEnvelope,
    "event": EventEnvelope,
    "cascade_add": CascadeAddEnvelope,
    "candidate_progress": CandidateProgressEnvelope,
    "initial_metrics": InitialMetricsEnvelope,
    "chunked_message": ChunkedMessageEnvelope,
    # /ws/control
    "command_response": CommandResponseEnvelope,
    "connection_established": ConnectionEstablishedEnvelope,
}

# Discriminated-union type alias for IDE / type-checker introspection.
KnownEnvelope = Union[
    MetricsEnvelope,
    StateEnvelope,
    TopologyEnvelope,
    EventEnvelope,
    CascadeAddEnvelope,
    CandidateProgressEnvelope,
    InitialMetricsEnvelope,
    ChunkedMessageEnvelope,
    CommandResponseEnvelope,
    ConnectionEstablishedEnvelope,
]


# ---------------------------------------------------------------------------
# Cardinality-bounded label tracking — per process, thread-safe
# ---------------------------------------------------------------------------

_unknown_seen: set[str] = set()
_unknown_lock = threading.Lock()


def _bound_unknown_label(type_str: str) -> str:
    """Return ``type_str`` while there is budget; collapse to ``_unmatched`` after.

    The first :data:`UNKNOWN_TYPE_BUDGET` distinct unknown values are
    tracked verbatim and returned as-is. Once the budget is reached,
    only the previously-tracked values are returned verbatim; all new
    values collapse to :data:`UNMATCHED_TYPE_LABEL`.

    Thread-safe: protected by :data:`_unknown_lock` so the
    ``unknown_seen`` set can be safely mutated from multiple consumer
    threads (matters in canopy where the WS subscriber and Dash callback
    threads share a process).
    """
    with _unknown_lock:
        if type_str in _unknown_seen:
            return type_str
        if len(_unknown_seen) < UNKNOWN_TYPE_BUDGET:
            _unknown_seen.add(type_str)
            return type_str
        return UNMATCHED_TYPE_LABEL


def reset_unknown_label_state() -> None:
    """Clear the per-process distinct-unknown-type tracker.

    Intended for tests; production callers should never invoke this.
    """
    with _unknown_lock:
        _unknown_seen.clear()


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def validate_envelope(frame: dict[str, Any]) -> BaseEnvelope:
    """Validate an inbound WS frame and return the typed envelope.

    Args:
        frame: The dict produced by ``json.loads(raw_ws_message)``.

    Returns:
        One of the :data:`KNOWN_ENVELOPES` instances when ``frame["type"]``
        matches a known type and the payload validates; otherwise an
        :class:`UnknownEnvelope` whose ``type`` field is the bounded
        label (the original string, or :data:`UNMATCHED_TYPE_LABEL` once
        the per-process budget is exhausted).

    Raises:
        TypeError: If ``frame`` is not a dict (caller bug; consumers
            should guard with ``isinstance(frame, dict)`` before calling
            so they can attribute the failure to JSON-decode rather
            than schema mismatch).
    """
    if not isinstance(frame, dict):
        raise TypeError(f"validate_envelope expects a dict, got {type(frame).__name__}")

    type_str = frame.get("type", "")
    model_cls = KNOWN_ENVELOPES.get(type_str)

    if model_cls is None:
        # Unknown type — bound the label and wrap in UnknownEnvelope so
        # callers can detect via ``isinstance(env, UnknownEnvelope)``.
        bounded = _bound_unknown_label(type_str)
        # Reconstruct a minimal envelope; keep the original payload
        # under ``data`` so debug log lines surface what was received.
        # Coerce ``timestamp`` to float when possible to keep the
        # BaseEnvelope contract; fall back to 0.0 if missing/invalid.
        ts_raw = frame.get("timestamp", 0.0)
        try:
            ts = float(ts_raw)
        except (TypeError, ValueError):
            ts = 0.0
        return UnknownEnvelope(
            type=bounded,
            timestamp=ts,
            data=frame.get("data", {}) if isinstance(frame.get("data"), dict) else {},
            seq=frame.get("seq") if isinstance(frame.get("seq"), int) else None,
            emitted_at_monotonic=frame.get("emitted_at_monotonic") if isinstance(frame.get("emitted_at_monotonic"), (int, float)) else None,
        )

    try:
        return model_cls.model_validate(frame)
    except ValidationError:
        # Schema-mismatch on a known type — treat as unknown for the
        # counter's purposes (the wire contract was violated even though
        # ``type`` matched). Re-raise would crash the consumer, which
        # the chaos test explicitly forbids.
        bounded = _bound_unknown_label(type_str)
        ts_raw = frame.get("timestamp", 0.0)
        try:
            ts = float(ts_raw)
        except (TypeError, ValueError):
            ts = 0.0
        return UnknownEnvelope(
            type=bounded,
            timestamp=ts,
            data=frame.get("data", {}) if isinstance(frame.get("data"), dict) else {},
            seq=frame.get("seq") if isinstance(frame.get("seq"), int) else None,
            emitted_at_monotonic=frame.get("emitted_at_monotonic") if isinstance(frame.get("emitted_at_monotonic"), (int, float)) else None,
        )
