"""Tests for WebSocket authentication in training and control streams."""

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.app import create_app
from api.security import reset_security_state
from api.settings import Settings


@pytest.fixture(autouse=True)
def _reset_security():
    """Reset security state before each test."""
    reset_security_state()
    yield
    reset_security_state()


@pytest.fixture
def auth_client():
    """Create a test client with API key auth enabled."""
    settings = Settings(api_keys=["test-key"])
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def noauth_client():
    """Create a test client with API key auth disabled."""
    settings = Settings(api_keys=None)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.mark.unit
class TestWebSocketTrainingAuth:
    """Test /ws/training WebSocket authentication."""

    def test_rejected_when_auth_enabled_no_key(self, auth_client):
        """Connection should be closed when auth enabled and no key provided."""
        with pytest.raises((WebSocketDisconnect, Exception)):
            with auth_client.websocket_connect("/ws/training") as ws:
                ws.receive_json()

    def test_rejected_when_auth_enabled_invalid_key(self, auth_client):
        """Connection should be closed when auth enabled and invalid key."""
        with pytest.raises((WebSocketDisconnect, Exception)):
            with auth_client.websocket_connect(
                "/ws/training",
                headers={"X-API-Key": "wrong-key"},
            ) as ws:
                ws.receive_json()

    def test_accepted_when_auth_enabled_valid_key(self, auth_client):
        """Connection should be accepted when auth enabled and valid key provided."""
        with auth_client.websocket_connect(
            "/ws/training",
            headers={"X-API-Key": "test-key"},
        ) as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connection_established"

    def test_accepted_when_auth_disabled(self, noauth_client):
        """Connection should be accepted when auth is disabled."""
        with noauth_client.websocket_connect("/ws/training") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connection_established"


@pytest.mark.unit
class TestWebSocketControlAuth:
    """Test /ws/control WebSocket authentication."""

    def test_rejected_when_auth_enabled_no_key(self, auth_client):
        """Connection should be closed when auth enabled and no key provided."""
        with pytest.raises((WebSocketDisconnect, Exception)):
            with auth_client.websocket_connect("/ws/control") as ws:
                ws.receive_json()

    def test_rejected_when_auth_enabled_invalid_key(self, auth_client):
        """Connection should be closed when auth enabled and invalid key."""
        with pytest.raises((WebSocketDisconnect, Exception)):
            with auth_client.websocket_connect(
                "/ws/control",
                headers={"X-API-Key": "wrong-key"},
            ) as ws:
                ws.receive_json()

    def test_accepted_when_auth_enabled_valid_key(self, auth_client):
        """Connection should be accepted when auth enabled and valid key provided."""
        with auth_client.websocket_connect(
            "/ws/control",
            headers={"X-API-Key": "test-key"},
        ) as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connection_established"

    def test_accepted_when_auth_disabled(self, noauth_client):
        """Connection should be accepted when auth is disabled."""
        with noauth_client.websocket_connect("/ws/control") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connection_established"
