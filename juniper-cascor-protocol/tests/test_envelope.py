"""Round-trip + wire-compat tests for the envelope schemas."""

import json

import pytest
from pydantic import ValidationError

from juniper_cascor_protocol.envelope import (
    KNOWN_ENVELOPES,
    UNKNOWN_TYPE_BUDGET,
    UNMATCHED_TYPE_LABEL,
    CandidateProgressEnvelope,
    CascadeAddEnvelope,
    ChunkedMessageEnvelope,
    CommandResponseEnvelope,
    ConnectionEstablishedEnvelope,
    EventEnvelope,
    InitialMetricsEnvelope,
    MetricsEnvelope,
    StateEnvelope,
    TopologyEnvelope,
    UnknownEnvelope,
    reset_unknown_label_state,
    validate_envelope,
)


# ---------------------------------------------------------------------------
# Per-envelope round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,type_str,extra_data",
    [
        (MetricsEnvelope, "metrics", {"loss": 0.1, "accuracy": 0.9}),
        (StateEnvelope, "state", {"phase": "candidate", "epoch": 42}),
        (TopologyEnvelope, "topology", {"hidden_units": 3, "edges": []}),
        (EventEnvelope, "event", {"event": "training_started"}),
        (CascadeAddEnvelope, "cascade_add", {"unit_id": 7, "correlation": 0.5}),
        (CandidateProgressEnvelope, "candidate_progress", {"candidate_index": 2, "epoch": 10}),
    ],
)
def test_freeform_data_envelope_roundtrip(cls, type_str, extra_data):
    """Free-form-data envelopes preserve arbitrary ``data`` dicts byte-for-byte."""
    env = cls(timestamp=1716_000_000.0, data=extra_data, seq=42, emitted_at_monotonic=123.456)
    dumped = env.model_dump(exclude_none=True)
    assert dumped == {
        "type": type_str,
        "timestamp": 1716_000_000.0,
        "data": extra_data,
        "seq": 42,
        "emitted_at_monotonic": 123.456,
    }
    # Round-trip through json
    decoded = json.loads(json.dumps(dumped))
    rebuilt = cls.model_validate(decoded)
    assert rebuilt == env


def test_initial_metrics_typed_payload_roundtrip():
    env = InitialMetricsEnvelope(
        timestamp=1716_000_000.0,
        data={"metrics": [{"loss": 0.1}, {"loss": 0.2}], "count": 2, "current_seq": 17},
    )
    assert env.data.count == 2
    assert env.data.current_seq == 17
    assert env.data.metrics[0]["loss"] == 0.1
    rebuilt = InitialMetricsEnvelope.model_validate_json(env.model_dump_json())
    assert rebuilt == env


def test_chunked_message_typed_payload_roundtrip():
    env = ChunkedMessageEnvelope(
        timestamp=1716_000_000.0,
        data={
            "chunk_id": "abc-123",
            "chunk_index": 0,
            "total_chunks": 3,
            "original_type": "topology",
            "payload": "{\"foo\":\"bar\"}",
        },
        seq=99,
    )
    assert env.data.chunk_index == 0
    assert env.data.total_chunks == 3
    rebuilt = ChunkedMessageEnvelope.model_validate_json(env.model_dump_json())
    assert rebuilt == env


def test_chunked_message_rejects_negative_chunk_index():
    with pytest.raises(ValidationError):
        ChunkedMessageEnvelope(
            timestamp=0.0,
            data={"chunk_id": "x", "chunk_index": -1, "total_chunks": 1, "original_type": "metrics", "payload": ""},
        )


def test_chunked_message_rejects_zero_total_chunks():
    with pytest.raises(ValidationError):
        ChunkedMessageEnvelope(
            timestamp=0.0,
            data={"chunk_id": "x", "chunk_index": 0, "total_chunks": 0, "original_type": "metrics", "payload": ""},
        )


def test_command_response_minimal():
    env = CommandResponseEnvelope(
        timestamp=1716_000_000.0,
        data={"command": "start", "status": "success"},
    )
    dumped = env.model_dump(exclude_none=True)
    assert dumped == {
        "type": "command_response",
        "timestamp": 1716_000_000.0,
        "data": {"command": "start", "status": "success"},
    }


def test_command_response_with_correlation_and_error():
    env = CommandResponseEnvelope(
        timestamp=1716_000_000.0,
        data={
            "command": "set_params",
            "status": "error",
            "command_id": "uuid-1",
            "error": "validation failed",
            "code": "invalid_params",
        },
    )
    assert env.data.code == "invalid_params"
    assert env.data.error == "validation failed"


def test_connection_established_default_data():
    """The handshake message accepts an empty ``data`` and still validates."""
    env = ConnectionEstablishedEnvelope(timestamp=1716_000_000.0, data={})
    assert env.data.server_version is None
    assert env.data.protocol_version is None


def test_connection_established_with_versions():
    env = ConnectionEstablishedEnvelope(
        timestamp=1716_000_000.0,
        data={"server_version": "0.4.0", "protocol_version": "0.1"},
    )
    assert env.data.server_version == "0.4.0"


# ---------------------------------------------------------------------------
# validate_envelope
# ---------------------------------------------------------------------------


def test_known_envelopes_dict_covers_all_types():
    """The dispatch dict must include every typed envelope class."""
    expected_types = {
        "metrics",
        "state",
        "topology",
        "event",
        "cascade_add",
        "candidate_progress",
        "initial_metrics",
        "chunked_message",
        "command_response",
        "connection_established",
    }
    assert set(KNOWN_ENVELOPES.keys()) == expected_types


def test_validate_envelope_known_type():
    frame = {"type": "metrics", "timestamp": 1.0, "data": {"loss": 0.1}}
    env = validate_envelope(frame)
    assert isinstance(env, MetricsEnvelope)
    assert env.data == {"loss": 0.1}


def test_validate_envelope_unknown_type_returns_unknown_envelope():
    reset_unknown_label_state()
    frame = {"type": "totally_made_up_type", "timestamp": 1.0, "data": {}}
    env = validate_envelope(frame)
    assert isinstance(env, UnknownEnvelope)
    assert env.type == "totally_made_up_type"


def test_validate_envelope_unknown_type_collapses_after_budget():
    """METRICS-MON R1.1: cardinality bound on unknown labels."""
    reset_unknown_label_state()
    # Fill the budget with distinct unknowns.
    for i in range(UNKNOWN_TYPE_BUDGET):
        env = validate_envelope({"type": f"unknown_{i}", "timestamp": 0.0, "data": {}})
        assert env.type == f"unknown_{i}"
    # The next distinct unknown collapses.
    env = validate_envelope({"type": "another_unknown", "timestamp": 0.0, "data": {}})
    assert env.type == UNMATCHED_TYPE_LABEL
    # A previously-seen unknown still returns its original label.
    env_repeat = validate_envelope({"type": "unknown_0", "timestamp": 0.0, "data": {}})
    assert env_repeat.type == "unknown_0"


def test_validate_envelope_known_type_with_invalid_payload_falls_back_to_unknown():
    """Schema-mismatch on a known type does NOT raise — callers stay alive."""
    reset_unknown_label_state()
    # initial_metrics requires count to be int; pass a wildly invalid value.
    bad = {"type": "initial_metrics", "timestamp": 1.0, "data": {"metrics": [], "count": "not-an-int", "current_seq": 0}}
    env = validate_envelope(bad)
    assert isinstance(env, UnknownEnvelope)
    assert env.type == "initial_metrics"  # original label preserved (was budget-tracked because the type itself is recognized)


def test_validate_envelope_rejects_non_dict():
    with pytest.raises(TypeError):
        validate_envelope("not a dict")  # type: ignore[arg-type]


def test_validate_envelope_handles_missing_timestamp():
    """An unknown frame missing ``timestamp`` falls back to 0.0."""
    reset_unknown_label_state()
    env = validate_envelope({"type": "garbage", "data": {}})
    assert isinstance(env, UnknownEnvelope)
    assert env.timestamp == 0.0


def test_validate_envelope_handles_non_numeric_timestamp():
    reset_unknown_label_state()
    env = validate_envelope({"type": "garbage", "timestamp": "not-a-float", "data": {}})
    assert isinstance(env, UnknownEnvelope)
    assert env.timestamp == 0.0


def test_validate_envelope_preserves_seq_when_int():
    reset_unknown_label_state()
    env = validate_envelope({"type": "garbage", "timestamp": 1.0, "data": {}, "seq": 42})
    assert env.seq == 42


def test_validate_envelope_drops_non_int_seq():
    reset_unknown_label_state()
    env = validate_envelope({"type": "garbage", "timestamp": 1.0, "data": {}, "seq": "not-an-int"})
    assert env.seq is None


def test_validate_envelope_drops_non_dict_data():
    reset_unknown_label_state()
    env = validate_envelope({"type": "garbage", "timestamp": 1.0, "data": "not-a-dict"})
    assert env.data == {}


def test_validate_envelope_preserves_emitted_at_monotonic():
    reset_unknown_label_state()
    env = validate_envelope({"type": "garbage", "timestamp": 1.0, "data": {}, "emitted_at_monotonic": 123.456})
    assert env.emitted_at_monotonic == 123.456


def test_validate_envelope_drops_non_numeric_monotonic():
    reset_unknown_label_state()
    env = validate_envelope({"type": "garbage", "timestamp": 1.0, "data": {}, "emitted_at_monotonic": "bad"})
    assert env.emitted_at_monotonic is None


# ---------------------------------------------------------------------------
# Wire-compat snapshot — pre-R2.2 byte-for-byte shapes
# ---------------------------------------------------------------------------


def test_metrics_envelope_byte_compat_with_pre_migration_dict_builder():
    """Snapshot from juniper-cascor/src/api/websocket/messages.py::create_metrics_message.

    The dict-builder produced ``{"type": "metrics", "timestamp": <float>, "data": {...},
    "seq": <int>, "emitted_at_monotonic": <float>}``. Our Pydantic dump must match.
    """
    env = MetricsEnvelope(timestamp=1716_000_000.0, data={"x": 1}, seq=5, emitted_at_monotonic=100.0)
    assert env.model_dump(exclude_none=True) == {
        "type": "metrics",
        "timestamp": 1716_000_000.0,
        "data": {"x": 1},
        "seq": 5,
        "emitted_at_monotonic": 100.0,
    }


def test_metrics_envelope_byte_compat_no_seq_no_monotonic():
    """``create_metrics_message`` omits ``seq`` and ``emitted_at_monotonic`` when None."""
    env = MetricsEnvelope(timestamp=1716_000_000.0, data={"x": 1})
    assert env.model_dump(exclude_none=True) == {
        "type": "metrics",
        "timestamp": 1716_000_000.0,
        "data": {"x": 1},
    }


def test_initial_metrics_envelope_byte_compat():
    """``create_initial_metrics_message`` packs ``metrics`` + ``count`` + ``current_seq``."""
    env = InitialMetricsEnvelope(
        timestamp=1716_000_000.0,
        data={"metrics": [{"a": 1}, {"a": 2}], "count": 2, "current_seq": 7},
    )
    assert env.model_dump(exclude_none=True) == {
        "type": "initial_metrics",
        "timestamp": 1716_000_000.0,
        "data": {"metrics": [{"a": 1}, {"a": 2}], "count": 2, "current_seq": 7},
    }


def test_chunked_message_envelope_byte_compat():
    """``create_chunked_message`` shape from GAP-WS-18."""
    env = ChunkedMessageEnvelope(
        timestamp=1716_000_000.0,
        data={
            "chunk_id": "uuid-1",
            "chunk_index": 0,
            "total_chunks": 2,
            "original_type": "topology",
            "payload": "<json-slice>",
        },
    )
    assert env.model_dump(exclude_none=True) == {
        "type": "chunked_message",
        "timestamp": 1716_000_000.0,
        "data": {
            "chunk_id": "uuid-1",
            "chunk_index": 0,
            "total_chunks": 2,
            "original_type": "topology",
            "payload": "<json-slice>",
        },
    }


def test_command_response_envelope_byte_compat_no_seq():
    """``command_response`` carries no ``seq`` per D-03 (control channel has no replay buffer)."""
    env = CommandResponseEnvelope(
        timestamp=1716_000_000.0,
        data={"command": "stop", "status": "success", "command_id": "id-1", "result": {"ok": True}},
    )
    dumped = env.model_dump(exclude_none=True)
    assert "seq" not in dumped
    assert dumped == {
        "type": "command_response",
        "timestamp": 1716_000_000.0,
        "data": {"command": "stop", "status": "success", "command_id": "id-1", "result": {"ok": True}},
    }
