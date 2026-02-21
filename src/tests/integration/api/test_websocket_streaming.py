"""Integration tests for WebSocket streaming.

Tests real-time WebSocket connections with the training lifecycle.
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings


@pytest.fixture
def client():
    """Create a test client with lifecycle manager."""
    settings = Settings()
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
    # Signal training threads to stop (see test_api_full_lifecycle.py)
    lifecycle = getattr(app.state, "lifecycle", None)
    if lifecycle:
        lifecycle._stop_requested.set()
        if getattr(lifecycle, "_executor", None):
            lifecycle._executor.shutdown(wait=False, cancel_futures=True)


_TRAIN_X = [
    [-1.0, -1.0],
    [-0.8, -0.9],
    [1.0, 1.0],
    [0.8, 0.9],
]
_TRAIN_Y = [
    [1.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
]


@pytest.mark.integration
class TestWebSocketStreaming:
    """Test WebSocket streaming with training."""

    def test_training_stream_connect_sequence(self, client):
        """Connect to /ws/training and receive the 3-message connect sequence."""
        with client.websocket_connect("/ws/training") as ws:
            msg1 = ws.receive_json()
            assert msg1["type"] == "connection_established"

            msg2 = ws.receive_json()
            assert msg2["type"] == "initial_status"

            msg3 = ws.receive_json()
            assert msg3["type"] == "state"

    def test_control_stream_send_commands(self, client):
        """Send commands via /ws/control and get responses."""
        with client.websocket_connect("/ws/control") as ws:
            # Drain connection established
            ws.receive_json()

            # Send stop (valid even without training)
            ws.send_text(json.dumps({"command": "stop"}))
            resp = ws.receive_json()
            assert resp["type"] == "command_response"
            assert resp["data"]["command"] == "stop"
            assert resp["data"]["status"] == "success"

            # Send reset
            ws.send_text(json.dumps({"command": "reset"}))
            resp = ws.receive_json()
            assert resp["type"] == "command_response"
            assert resp["data"]["command"] == "reset"
            assert resp["data"]["status"] == "success"

    def test_control_stream_invalid_command(self, client):
        """Invalid commands return structured errors."""
        with client.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established

            ws.send_text(json.dumps({"command": "fly"}))
            resp = ws.receive_json()
            assert resp["data"]["status"] == "error"
            assert "Unknown command" in resp["data"]["error"]

    def test_control_start_requires_network(self, client):
        """Start command via WebSocket fails without network."""
        with client.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established

            ws.send_text(json.dumps({"command": "start"}))
            resp = ws.receive_json()
            assert resp["data"]["status"] == "error"

    def test_multiple_training_stream_clients(self, client):
        """Multiple clients can connect to /ws/training simultaneously."""
        with client.websocket_connect("/ws/training") as ws1:
            msg1 = ws1.receive_json()
            assert msg1["type"] == "connection_established"

            with client.websocket_connect("/ws/training") as ws2:
                msg2 = ws2.receive_json()
                assert msg2["type"] == "connection_established"
                # Both connected — connection count should be 2
                assert msg2["data"]["connections"] == 2

    def test_training_stream_after_network_create(self, client):
        """Training stream shows network_loaded=True after creation."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2, "candidate_epochs": 2, "output_epochs": 2, "patience": 1})

        with client.websocket_connect("/ws/training") as ws:
            ws.receive_json()  # connection_established
            status = ws.receive_json()  # initial_status
            assert status["data"]["network_loaded"] is True
