"""Tests for /ws/control WebSocket handler."""

import json

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings


@pytest.fixture
def client():
    """Create a test client with lifecycle manager."""
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_network(client):
    """Test client with a network already created via POST /v1/network."""
    client.post("/v1/network", json={"input_size": 2, "output_size": 2})
    return client


@pytest.mark.unit
class TestControlStreamHandler:
    """Test /ws/control WebSocket handler."""

    def test_connect_receives_established(self, client):
        """Connecting to /ws/control receives connection_established."""
        with client.websocket_connect("/ws/control") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connection_established"
            assert msg["data"]["channel"] == "control"

    def test_invalid_json(self, client):
        """Sending invalid JSON returns error."""
        with client.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established
            ws.send_text("not json")
            response = ws.receive_json()
            assert response["type"] == "command_response"
            assert response["data"]["status"] == "error"
            assert "Invalid JSON" in response["data"]["error"]

    def test_unknown_command(self, client):
        """Unknown command returns error."""
        with client.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established
            ws.send_text(json.dumps({"command": "unknown_cmd"}))
            response = ws.receive_json()
            assert response["type"] == "command_response"
            assert response["data"]["status"] == "error"
            assert "Unknown command" in response["data"]["error"]

    def test_stop_command(self, client):
        """Stop command returns success."""
        with client.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established
            ws.send_text(json.dumps({"command": "stop"}))
            response = ws.receive_json()
            assert response["type"] == "command_response"
            assert response["data"]["command"] == "stop"
            assert response["data"]["status"] == "success"

    def test_reset_command(self, client):
        """Reset command returns success."""
        with client.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established
            ws.send_text(json.dumps({"command": "reset"}))
            response = ws.receive_json()
            assert response["type"] == "command_response"
            assert response["data"]["command"] == "reset"
            assert response["data"]["status"] == "success"

    def test_pause_no_training(self, client):
        """Pause command without active training returns error."""
        with client.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established
            ws.send_text(json.dumps({"command": "pause"}))
            response = ws.receive_json()
            assert response["type"] == "command_response"
            assert response["data"]["status"] == "error"

    def test_resume_no_training(self, client):
        """Resume command without paused training returns error."""
        with client.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established
            ws.send_text(json.dumps({"command": "resume"}))
            response = ws.receive_json()
            assert response["type"] == "command_response"
            assert response["data"]["status"] == "error"

    def test_start_no_network(self, client):
        """Start command without network returns error."""
        with client.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established
            ws.send_text(json.dumps({"command": "start"}))
            response = ws.receive_json()
            assert response["type"] == "command_response"
            assert response["data"]["status"] == "error"

    def test_missing_command_field(self, client):
        """Message without command field returns error."""
        with client.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established
            ws.send_text(json.dumps({"action": "start"}))
            response = ws.receive_json()
            assert response["type"] == "command_response"
            assert response["data"]["status"] == "error"

    def test_set_params_updates_live_network(self, client_with_network):
        """set_params command applies whitelisted params to the live network (CR-008)."""
        with client_with_network.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established
            ws.send_text(
                json.dumps(
                    {
                        "command": "set_params",
                        "params": {"learning_rate": 0.007, "max_iterations": 23},
                    }
                )
            )
            response = ws.receive_json()
            assert response["type"] == "command_response"
            assert response["data"]["command"] == "set_params"
            assert response["data"]["status"] == "success"

        lifecycle = client_with_network.app.state.lifecycle
        assert lifecycle.network.learning_rate == pytest.approx(0.007)
        assert lifecycle.network.max_iterations == 23

    def test_set_params_without_params_dict_errors(self, client_with_network):
        """set_params without a 'params' dict returns error."""
        with client_with_network.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established
            ws.send_text(json.dumps({"command": "set_params"}))
            response = ws.receive_json()
            assert response["type"] == "command_response"
            assert response["data"]["command"] == "set_params"
            assert response["data"]["status"] == "error"

    def test_set_params_without_network_errors(self, client):
        """set_params without a network returns error (update_params raises)."""
        with client.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established
            ws.send_text(
                json.dumps(
                    {
                        "command": "set_params",
                        "params": {"learning_rate": 0.01},
                    }
                )
            )
            response = ws.receive_json()
            assert response["type"] == "command_response"
            assert response["data"]["command"] == "set_params"
            assert response["data"]["status"] == "error"
