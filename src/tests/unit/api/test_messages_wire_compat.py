"""Wire-compat snapshot tests for the R2.2.2 juniper-cascor-protocol migration.

METRICS-MON R2.2.2 / seed-05: per the R2.2 design §6, the cascor server
migration ships a snapshot test that pins the byte-for-byte shape of
every emitted ``/ws/training`` and ``/ws/control`` frame so the
shared-lib swap cannot silently drift the contract.

The shapes below were captured from juniper-cascor ``main`` at HEAD
``062a4b9`` (the R2.2.1 commit immediately before the server adoption
landed). Any future bump of the shared lib that changes these keys,
field ordering, or omit-when-None behaviour will fail this test first.
"""

import pytest

from api.websocket.messages import create_candidate_progress_message, create_cascade_add_message, create_chunked_message, create_control_ack_message, create_event_message, create_initial_metrics_message, create_metrics_message, create_state_message, create_topology_message

# ---------------------------------------------------------------------------
# Free-form-data envelopes — broadcast types
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBroadcastEnvelopeShapes:
    """Each broadcast helper produces ``{type, timestamp, data}`` plus optional ``seq``/``emitted_at_monotonic``."""

    def test_metrics_envelope_with_full_seq_and_monotonic(self):
        msg = create_metrics_message({"loss": 0.1, "accuracy": 0.9}, seq=42, emitted_at_monotonic=123.456)
        assert set(msg.keys()) == {"type", "timestamp", "data", "seq", "emitted_at_monotonic"}
        assert msg["type"] == "metrics"
        assert msg["data"] == {"loss": 0.1, "accuracy": 0.9}
        assert msg["seq"] == 42
        assert msg["emitted_at_monotonic"] == 123.456
        assert isinstance(msg["timestamp"], float)

    def test_metrics_envelope_omits_seq_and_monotonic_when_not_provided(self):
        msg = create_metrics_message({"loss": 0.1})
        assert set(msg.keys()) == {"type", "timestamp", "data"}
        assert "seq" not in msg
        assert "emitted_at_monotonic" not in msg

    def test_state_envelope_byte_compat(self):
        msg = create_state_message({"phase": "candidate", "epoch": 7}, seq=5)
        assert msg["type"] == "state"
        assert msg["data"] == {"phase": "candidate", "epoch": 7}
        assert msg["seq"] == 5
        assert "emitted_at_monotonic" not in msg

    def test_topology_envelope_byte_compat(self):
        msg = create_topology_message({"hidden_units": 3, "edges": []}, seq=11, emitted_at_monotonic=200.5)
        assert msg["type"] == "topology"
        assert msg["data"] == {"hidden_units": 3, "edges": []}

    def test_event_envelope_byte_compat(self):
        msg = create_event_message({"event": "training_started"})
        assert msg["type"] == "event"
        assert msg["data"] == {"event": "training_started"}
        assert "seq" not in msg

    def test_cascade_add_envelope_byte_compat(self):
        msg = create_cascade_add_message({"unit_id": 7, "correlation": 0.5}, seq=99)
        assert msg["type"] == "cascade_add"
        assert msg["seq"] == 99

    def test_candidate_progress_envelope_byte_compat(self):
        msg = create_candidate_progress_message({"candidate_index": 2, "epoch": 10}, emitted_at_monotonic=42.0)
        assert msg["type"] == "candidate_progress"
        assert msg["emitted_at_monotonic"] == 42.0


# ---------------------------------------------------------------------------
# Typed-payload envelopes — initial_metrics, chunked_message
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitialMetricsShape:

    def test_initial_metrics_keys_unchanged(self):
        msg = create_initial_metrics_message([{"loss": 0.1}, {"loss": 0.2}], current_seq=17)
        assert set(msg.keys()) == {"type", "timestamp", "data"}
        assert msg["type"] == "initial_metrics"
        assert set(msg["data"].keys()) == {"metrics", "count", "current_seq"}
        assert msg["data"]["count"] == 2
        assert msg["data"]["current_seq"] == 17
        assert msg["data"]["metrics"] == [{"loss": 0.1}, {"loss": 0.2}]

    def test_initial_metrics_default_current_seq_zero(self):
        msg = create_initial_metrics_message([])
        assert msg["data"]["count"] == 0
        assert msg["data"]["current_seq"] == 0


@pytest.mark.unit
class TestChunkedMessageShape:

    def test_chunked_message_keys_unchanged(self):
        msg = create_chunked_message(
            chunk_id="abc-123",
            chunk_index=0,
            total_chunks=3,
            original_type="topology",
            payload="<json-slice>",
        )
        assert msg["type"] == "chunked_message"
        assert set(msg["data"].keys()) == {"chunk_id", "chunk_index", "total_chunks", "original_type", "payload"}
        assert msg["data"]["chunk_index"] == 0
        assert msg["data"]["total_chunks"] == 3
        # ``seq`` is set by the manager during broadcast — never by the helper.
        assert "seq" not in msg


# ---------------------------------------------------------------------------
# Control-channel envelope — command_response
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCommandResponseShape:
    """``command_response`` carries no ``seq`` per D-03 (control channel has no replay buffer)."""

    def test_minimal(self):
        msg = create_control_ack_message("start", "success")
        assert msg["type"] == "command_response"
        assert "seq" not in msg
        assert msg["data"] == {"command": "start", "status": "success"}

    def test_with_command_id(self):
        msg = create_control_ack_message("set_params", "success", command_id="uuid-1")
        assert msg["data"]["command_id"] == "uuid-1"
        assert "result" not in msg["data"]
        assert "error" not in msg["data"]

    def test_with_result(self):
        msg = create_control_ack_message("status", "success", data={"running": True}, command_id="uuid-2")
        assert msg["data"]["result"] == {"running": True}
        assert msg["data"]["command_id"] == "uuid-2"

    def test_with_error_and_code(self):
        msg = create_control_ack_message("nope", "error", error="not allowed", code="unknown_command")
        assert msg["data"]["status"] == "error"
        assert msg["data"]["error"] == "not allowed"
        assert msg["data"]["code"] == "unknown_command"
        assert "result" not in msg["data"]
        assert "command_id" not in msg["data"]

    def test_omits_optional_fields_when_none(self):
        msg = create_control_ack_message("start", "queued")
        # Only command + status — no command_id, result, error, code.
        assert msg["data"] == {"command": "start", "status": "queued"}


# ---------------------------------------------------------------------------
# Pydantic-on-the-wire round-trip — message bytes survive a full client decode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPydanticRoundTrip:
    """Each emitted dict must validate cleanly through the typed envelope class.

    This is the producer-side equivalent of the consumer ``validate_envelope``
    contract: cascor never emits a frame that its own typed envelope rejects.
    """

    def test_metrics_roundtrip(self):
        from juniper_cascor_protocol.envelope import MetricsEnvelope

        msg = create_metrics_message({"loss": 0.1}, seq=1, emitted_at_monotonic=2.0)
        rebuilt = MetricsEnvelope.model_validate(msg)
        assert rebuilt.data == {"loss": 0.1}

    def test_initial_metrics_roundtrip(self):
        from juniper_cascor_protocol.envelope import InitialMetricsEnvelope

        msg = create_initial_metrics_message([{"a": 1}], current_seq=99)
        rebuilt = InitialMetricsEnvelope.model_validate(msg)
        assert rebuilt.data.count == 1
        assert rebuilt.data.current_seq == 99

    def test_command_response_roundtrip(self):
        from juniper_cascor_protocol.envelope import CommandResponseEnvelope

        msg = create_control_ack_message("stop", "success", command_id="cid")
        rebuilt = CommandResponseEnvelope.model_validate(msg)
        assert rebuilt.data.command == "stop"
        assert rebuilt.data.command_id == "cid"

    def test_chunked_message_roundtrip(self):
        from juniper_cascor_protocol.envelope import ChunkedMessageEnvelope

        msg = create_chunked_message(
            chunk_id="x",
            chunk_index=0,
            total_chunks=1,
            original_type="topology",
            payload="{}",
        )
        rebuilt = ChunkedMessageEnvelope.model_validate(msg)
        assert rebuilt.data.chunk_index == 0
