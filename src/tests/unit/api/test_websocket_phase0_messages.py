"""Tests for Phase 0-cascor message envelope extensions."""

import pytest

from api.websocket.messages import create_candidate_progress_message, create_cascade_add_message, create_control_ack_message, create_event_message, create_metrics_message, create_state_message, create_topology_message


@pytest.mark.unit
class TestMessageEnvelopeSeq:
    """Test seq and emitted_at_monotonic on broadcast messages."""

    @pytest.mark.parametrize(
        "builder",
        [
            create_metrics_message,
            create_state_message,
            create_topology_message,
            create_event_message,
            create_cascade_add_message,
            create_candidate_progress_message,
        ],
    )
    def test_seq_included_when_provided(self, builder):
        """Broadcast builders include seq when explicitly passed."""
        msg = builder({"key": "val"}, seq=42, emitted_at_monotonic=100.5)
        assert msg["seq"] == 42
        assert msg["emitted_at_monotonic"] == 100.5

    @pytest.mark.parametrize(
        "builder",
        [
            create_metrics_message,
            create_state_message,
            create_topology_message,
            create_event_message,
            create_cascade_add_message,
            create_candidate_progress_message,
        ],
    )
    def test_seq_absent_when_none(self, builder):
        """Broadcast builders omit seq when not provided (backward compat)."""
        msg = builder({"key": "val"})
        assert "seq" not in msg
        assert "emitted_at_monotonic" not in msg

    @pytest.mark.parametrize(
        "builder",
        [
            create_metrics_message,
            create_state_message,
            create_topology_message,
            create_event_message,
            create_cascade_add_message,
            create_candidate_progress_message,
        ],
    )
    def test_envelope_structure_preserved(self, builder):
        """All builders produce {type, timestamp, data} envelope."""
        msg = builder({"key": "val"})
        assert "type" in msg
        assert "timestamp" in msg
        assert "data" in msg
        assert msg["data"]["key"] == "val"


@pytest.mark.unit
class TestControlAckCommandId:
    """Test command_id echo on command_response (D-02/D-03)."""

    def test_control_ack_command_id_echo(self):
        """command_id is echoed in command_response when provided."""
        msg = create_control_ack_message("stop", "success", command_id="req-123")
        assert msg["data"]["command_id"] == "req-123"

    def test_control_ack_no_command_id_when_absent(self):
        """command_id is omitted when not provided."""
        msg = create_control_ack_message("stop", "success")
        assert "command_id" not in msg["data"]

    def test_ws_control_command_response_has_no_seq(self):
        """command_response never has a seq field (D-03 canonical)."""
        msg = create_control_ack_message("stop", "success", command_id="abc")
        assert "seq" not in msg
        assert msg["type"] == "command_response"

    def test_control_ack_preserves_result_and_error(self):
        """command_id works alongside existing data and error fields."""
        msg = create_control_ack_message(
            "set_params",
            "success",
            data={"applied": True},
            command_id="cmd-456",
        )
        assert msg["data"]["command_id"] == "cmd-456"
        assert msg["data"]["result"] == {"applied": True}
        assert msg["data"]["command"] == "set_params"
        assert msg["data"]["status"] == "success"
