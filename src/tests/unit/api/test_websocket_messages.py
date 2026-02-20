"""Tests for WebSocket message builders."""

import time

import pytest

from api.websocket.messages import create_cascade_add_message, create_control_ack_message, create_event_message, create_metrics_message, create_state_message, create_topology_message


@pytest.mark.unit
class TestMessageBuilders:
    """Test WebSocket message builders."""

    def test_create_metrics_message(self):
        """Metrics message has correct type and structure."""
        data = {"epoch": 10, "loss": 0.5}
        msg = create_metrics_message(data)
        assert msg["type"] == "metrics"
        assert "timestamp" in msg
        assert msg["data"] == data

    def test_create_state_message(self):
        """State message has correct type and structure."""
        data = {"status": "Started", "phase": "Output"}
        msg = create_state_message(data)
        assert msg["type"] == "state"
        assert "timestamp" in msg
        assert msg["data"] == data

    def test_create_topology_message(self):
        """Topology message has correct type and structure."""
        data = {"input_size": 2, "output_size": 2, "hidden_units": []}
        msg = create_topology_message(data)
        assert msg["type"] == "topology"
        assert msg["data"] == data

    def test_create_event_message(self):
        """Event message has correct type and structure."""
        data = {"event": "training_complete"}
        msg = create_event_message(data)
        assert msg["type"] == "event"
        assert msg["data"] == data

    def test_create_cascade_add_message(self):
        """Cascade add message has correct type."""
        data = {"hidden_unit_index": 0, "correlation": 0.95}
        msg = create_cascade_add_message(data)
        assert msg["type"] == "cascade_add"
        assert msg["data"] == data

    def test_create_control_ack_success(self):
        """Control ack message with success."""
        msg = create_control_ack_message("start", "success", data={"training": True})
        assert msg["type"] == "command_response"
        assert msg["data"]["command"] == "start"
        assert msg["data"]["status"] == "success"
        assert msg["data"]["result"]["training"] is True

    def test_create_control_ack_error(self):
        """Control ack message with error."""
        msg = create_control_ack_message("pause", "error", error="Not running")
        assert msg["type"] == "command_response"
        assert msg["data"]["command"] == "pause"
        assert msg["data"]["status"] == "error"
        assert msg["data"]["error"] == "Not running"

    def test_create_control_ack_minimal(self):
        """Control ack message with no extra data or error."""
        msg = create_control_ack_message("stop", "success")
        assert msg["type"] == "command_response"
        assert msg["data"]["command"] == "stop"
        assert msg["data"]["status"] == "success"
        assert "error" not in msg["data"]

    def test_timestamp_is_recent(self):
        """All messages have timestamps near current time."""
        before = time.time()
        msg = create_metrics_message({"epoch": 1})
        after = time.time()
        assert before <= msg["timestamp"] <= after

    def test_data_passthrough(self):
        """Message data is passed through unmodified."""
        complex_data = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "float": 0.123,
        }
        msg = create_metrics_message(complex_data)
        assert msg["data"] is complex_data
