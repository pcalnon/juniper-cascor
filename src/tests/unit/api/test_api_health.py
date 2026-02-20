"""Tests for API health check endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings


@pytest.fixture
def client():
    """Create a test client for the API (lifespan runs)."""
    settings = Settings()
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.mark.unit
class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client):
        """Test GET /v1/health returns ok."""
        response = client.get("/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["data"]["status"] == "ok"
        assert body["data"]["version"] == "0.4.0"
        assert "meta" in body
        assert "timestamp" in body["meta"]
        assert body["meta"]["version"] == "0.4.0"

    def test_liveness_probe(self, client):
        """Test GET /v1/health/live returns alive."""
        response = client.get("/v1/health/live")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["data"]["status"] == "alive"

    def test_readiness_probe_no_lifecycle(self, client):
        """Test GET /v1/health/ready when no lifecycle manager."""
        response = client.get("/v1/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["data"]["status"] == "ready"
        assert body["data"]["network_loaded"] is False

    def test_readiness_probe_with_lifecycle(self, client):
        """Test GET /v1/health/ready with lifecycle manager that has a network."""

        class MockLifecycle:
            def has_network(self):
                return True

            def shutdown(self):
                pass

        client.app.state.lifecycle = MockLifecycle()
        response = client.get("/v1/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["network_loaded"] is True


@pytest.mark.unit
class TestResponseEnvelope:
    """Test response envelope format."""

    def test_success_response_structure(self, client):
        """Test that all responses follow envelope format."""
        response = client.get("/v1/health")
        body = response.json()

        # Verify envelope structure
        assert "status" in body
        assert "data" in body
        assert "meta" in body
        assert "timestamp" in body["meta"]
        assert "version" in body["meta"]

    def test_error_response_for_invalid_route(self, client):
        """Test 404 for non-existent route."""
        response = client.get("/v1/nonexistent")
        assert response.status_code == 404
