"""Tests for metrics API routes."""

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
