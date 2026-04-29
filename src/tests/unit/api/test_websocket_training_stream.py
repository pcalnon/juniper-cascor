"""Tests for /ws/training WebSocket handler."""

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
def fast_client():
    """Test client with a 0.1s resume handshake timeout for fast tests."""
    settings = Settings(auto_start=False, ws_resume_handshake_timeout_s=0.1)
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


@pytest.mark.unit
class TestInitialMetricsBurst:
    """GAP-WS-16: fresh /ws/training connect emits an initial_metrics burst."""

    def test_initial_metrics_follows_state(self, fast_client):
        """After connection_established + initial_status + state, an initial_metrics
        envelope is delivered to fresh connects."""
        with fast_client.websocket_connect("/ws/training") as ws:
            ws.receive_json()  # connection_established
            ws.receive_json()  # initial_status
            ws.receive_json()  # state
            metrics_msg = ws.receive_json()
            assert metrics_msg["type"] == "initial_metrics"
            assert "metrics" in metrics_msg["data"]
            assert "count" in metrics_msg["data"]
            assert "current_seq" in metrics_msg["data"]
            assert metrics_msg["data"]["count"] == 0
            assert metrics_msg["data"]["metrics"] == []

    def test_initial_metrics_disabled_when_count_zero(self):
        """ws_initial_metrics_count=0 suppresses the burst entirely."""
        settings = Settings(auto_start=False, ws_initial_metrics_count=0, ws_resume_handshake_timeout_s=0.1)
        app = create_app(settings)
        with TestClient(app) as c:
            with c.websocket_connect("/ws/training") as ws:
                ws.receive_json()  # connection_established
                ws.receive_json()  # initial_status
                state_msg = ws.receive_json()
                assert state_msg["type"] == "state"
                # No initial_metrics burst — but the dispatcher should still
                # be alive. Send a subscribe_metrics request to verify.
                ws.send_json({"type": "subscribe_metrics", "data": {"max_count": 50}})
                resp = ws.receive_json()
                assert resp["type"] == "initial_metrics"


@pytest.mark.unit
class TestSubscribeMetrics:
    """GAP-WS-16: client-initiated subscribe_metrics request."""

    def test_subscribe_metrics_returns_initial_metrics_envelope(self, fast_client):
        with fast_client.websocket_connect("/ws/training") as ws:
            for _ in range(4):
                ws.receive_json()  # drain handshake (connection_established + initial_status + state + initial_metrics)
            ws.send_json({"type": "subscribe_metrics", "data": {"max_count": 25}})
            resp = ws.receive_json()
            assert resp["type"] == "initial_metrics"
            assert resp["data"]["count"] == 0  # No training yet

    def test_subscribe_metrics_clamps_max_count(self, fast_client):
        """max_count above the server cap is silently clamped (no error)."""
        with fast_client.websocket_connect("/ws/training") as ws:
            for _ in range(4):
                ws.receive_json()
            ws.send_json({"type": "subscribe_metrics", "data": {"max_count": 999999}})
            resp = ws.receive_json()
            assert resp["type"] == "initial_metrics"

    def test_unknown_message_type_is_ignored(self, fast_client):
        """Unknown frame types must not crash the recv loop."""
        with fast_client.websocket_connect("/ws/training") as ws:
            for _ in range(4):
                ws.receive_json()
            ws.send_json({"type": "this_is_not_a_real_type", "data": {}})
            ws.send_json({"type": "subscribe_metrics", "data": {"max_count": 10}})
            resp = ws.receive_json()
            assert resp["type"] == "initial_metrics"
