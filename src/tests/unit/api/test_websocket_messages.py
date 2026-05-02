"""Tests for WebSocket message builders."""

import time

import pytest

from api.websocket.messages import create_candidate_progress_message, create_cascade_add_message, create_chunked_message, create_control_ack_message, create_event_message, create_initial_metrics_message, create_metrics_message, create_state_message, create_topology_message


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

    def test_create_candidate_progress_message(self):
        """Candidate progress message has correct type."""
        data = {"candidate_id": 2, "epoch": 51, "total_epochs": 100, "correlation": 0.73}
        msg = create_candidate_progress_message(data)
        assert msg["type"] == "candidate_progress"
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
        # ``model_dump`` always returns a fresh dict, so identity (``is``)
        # comparison is unreliable; we only require value-equality for the
        # passthrough contract.
        assert msg["data"] == complex_data


@pytest.mark.unit
class TestInitialMetricsMessage:
    """GAP-WS-16: initial_metrics envelope sent on fresh /ws/training connect."""

    def test_envelope_shape(self):
        metrics = [{"epoch": 1}, {"epoch": 2}]
        msg = create_initial_metrics_message(metrics, current_seq=42)
        assert msg["type"] == "initial_metrics"
        assert "timestamp" in msg
        # ``model_dump`` always returns fresh containers; assert value-equality
        # rather than identity (``is``).
        assert msg["data"]["metrics"] == metrics
        assert msg["data"]["count"] == 2
        assert msg["data"]["current_seq"] == 42

    def test_empty_metrics_count_is_zero(self):
        msg = create_initial_metrics_message([])
        assert msg["data"]["count"] == 0
        assert msg["data"]["metrics"] == []
        assert msg["data"]["current_seq"] == 0

    def test_no_seq_field(self):
        """initial_metrics is a personal message, never carries its own seq."""
        msg = create_initial_metrics_message([{"epoch": 1}], current_seq=10)
        assert "seq" not in msg

    def test_current_seq_defaults_when_omitted(self):
        msg = create_initial_metrics_message([{"epoch": 1}])
        assert msg["data"]["current_seq"] == 0


@pytest.mark.unit
class TestChunkedMessageBuilder:
    """GAP-WS-18: chunked_message envelope shape."""

    def test_envelope_shape(self):
        msg = create_chunked_message(
            chunk_id="abc-123",
            chunk_index=0,
            total_chunks=3,
            original_type="topology",
            payload="<json-fragment>",
        )
        assert msg["type"] == "chunked_message"
        assert "timestamp" in msg
        assert msg["data"]["chunk_id"] == "abc-123"
        assert msg["data"]["chunk_index"] == 0
        assert msg["data"]["total_chunks"] == 3
        assert msg["data"]["original_type"] == "topology"
        assert msg["data"]["payload"] == "<json-fragment>"

    def test_no_seq_field_at_construction(self):
        """Builder does not embed seq — manager assigns one per chunk on broadcast."""
        msg = create_chunked_message(chunk_id="x", chunk_index=0, total_chunks=1, original_type="t", payload="")
        assert "seq" not in msg

    def test_payload_can_be_empty(self):
        """An empty payload is legal (e.g., the tail chunk of an exact-multiple split)."""
        msg = create_chunked_message(chunk_id="x", chunk_index=2, total_chunks=3, original_type="t", payload="")
        assert msg["data"]["payload"] == ""
