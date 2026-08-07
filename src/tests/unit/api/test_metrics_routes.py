"""Tests for metrics API routes."""

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


@pytest.mark.unit
class TestMetricsRoutes:
    """Test metrics retrieval routes."""

    def test_get_metrics_no_network(self, client):
        """GET /v1/metrics returns 404 when no network created."""
        response = client.get("/v1/metrics")
        assert response.status_code == 404

    def test_get_metrics_with_network(self, client):
        """GET /v1/metrics returns metrics after network creation."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        response = client.get("/v1/metrics")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert "epoch" in body["data"]

    def test_get_metrics_history_empty(self, client):
        """GET /v1/metrics/history returns empty list initially."""
        response = client.get("/v1/metrics/history")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["data"] == []

    def test_get_metrics_history_with_count(self, client):
        """GET /v1/metrics/history respects count parameter."""
        response = client.get("/v1/metrics/history?count=5")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert isinstance(body["data"], list)

    def test_get_metrics_history_invalid_count(self, client):
        """GET /v1/metrics/history rejects invalid count."""
        response = client.get("/v1/metrics/history?count=0")
        assert response.status_code == 422  # Pydantic validation error

    def test_get_metrics_history_negative_count(self, client):
        """GET /v1/metrics/history rejects negative count."""
        response = client.get("/v1/metrics/history?count=-1")
        assert response.status_code == 422


@pytest.mark.unit
class TestTransportEndpoint:
    """GAP-WS-16: GET /v1/metrics/transport surfaces WS bandwidth counters."""

    def test_transport_endpoint_returns_zeroed_stats_at_startup(self, client):
        response = client.get("/v1/metrics/transport")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        data = body["data"]
        assert data["bytes_sent_total"] == 0
        assert data["messages_sent_total"] == 0
        assert data["active_connections"] == 0
        assert data["pending_connections"] == 0
        assert "messages_sent_by_type" in data
        assert "bytes_sent_by_type" in data
        assert "uptime_seconds" in data

    def test_transport_endpoint_reflects_ws_activity(self, client):
        """A WS connect drives the counters above zero."""
        with client.websocket_connect("/ws/training") as ws:
            # Drain the handshake so we know sends have completed
            for _ in range(4):
                ws.receive_json()

            response = client.get("/v1/metrics/transport")
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["messages_sent_total"] >= 4
            assert data["bytes_sent_total"] > 0
            assert data["messages_sent_by_type"]["connection_established"] >= 1
            assert data["messages_sent_by_type"]["initial_metrics"] >= 1

    def test_transport_endpoint_503_when_ws_manager_missing(self, client):
        """GET /v1/metrics/transport returns 503 when ws_manager is uninitialized."""
        client.app.state.ws_manager = None
        response = client.get("/v1/metrics/transport")
        assert response.status_code == 503
        assert "WebSocket manager not initialized" in response.json()["error"]["message"]


@pytest.mark.unit
class TestMetricsRoutesUninitialized:
    """Defensive 503 branches when lifespan state is incomplete."""

    def test_get_metrics_503_when_lifecycle_missing(self, client):
        client.app.state.lifecycle = None
        response = client.get("/v1/metrics")
        assert response.status_code == 503
        assert "Lifecycle manager not initialized" in response.json()["error"]["message"]

    def test_get_metrics_history_503_when_lifecycle_missing(self, client):
        client.app.state.lifecycle = None
        response = client.get("/v1/metrics/history")
        assert response.status_code == 503
        assert "Lifecycle manager not initialized" in response.json()["error"]["message"]
