"""Tests for /ws/training WebSocket handler."""

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


@pytest.mark.unit
class TestTrainingStreamHandler:
    """Test /ws/training WebSocket handler."""

    def test_connect_receives_established(self, client):
        """Connecting to /ws/training receives connection_established message."""
        with client.websocket_connect("/ws/training") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connection_established"

    def test_connect_receives_initial_status(self, client):
        """After connection_established, receives initial_status."""
        with client.websocket_connect("/ws/training") as ws:
            # 1. connection_established
            msg1 = ws.receive_json()
            assert msg1["type"] == "connection_established"

            # 2. initial_status
            msg2 = ws.receive_json()
            assert msg2["type"] == "initial_status"
            assert "data" in msg2
            assert "network_loaded" in msg2["data"]

    def test_connect_receives_state(self, client):
        """After initial_status, receives current state."""
        with client.websocket_connect("/ws/training") as ws:
            ws.receive_json()  # connection_established
            ws.receive_json()  # initial_status
            msg3 = ws.receive_json()
            assert msg3["type"] == "state"
            assert "data" in msg3

    def test_connect_with_network(self, client):
        """Training stream reports network_loaded when network exists."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})

        with client.websocket_connect("/ws/training") as ws:
            ws.receive_json()  # connection_established
            status = ws.receive_json()  # initial_status
            assert status["data"]["network_loaded"] is True
