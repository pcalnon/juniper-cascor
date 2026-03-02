"""Tests for API health check endpoints."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.models.health import DependencyStatus, ReadinessResponse, probe_dependency
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
        """Test GET /v1/health returns flat ok (no envelope)."""
        response = client.get("/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"] == "0.4.0"

    def test_liveness_probe(self, client):
        """Test GET /v1/health/live returns alive."""
        response = client.get("/v1/health/live")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "alive"

    def test_readiness_probe_default(self, client):
        """Test GET /v1/health/ready with default lifecycle state."""
        response = client.get("/v1/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("ready", "degraded")
        assert body["version"] == "0.4.0"
        assert body["service"] == "juniper-cascor"
        assert "timestamp" in body
        assert body["details"]["network_loaded"] is False
        assert "training_state" in body["details"]

    def test_readiness_probe_with_lifecycle(self, client):
        """Test GET /v1/health/ready with lifecycle manager that has a network."""

        class MockLifecycle:
            def has_network(self):
                return True

            def get_status(self):
                return {"training_state": "idle"}

            def shutdown(self):
                pass

        client.app.state.lifecycle = MockLifecycle()
        response = client.get("/v1/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["details"]["network_loaded"] is True
        assert body["details"]["training_state"] == "idle"

    @patch.dict("os.environ", {"JUNIPER_DATA_URL": "http://fake-data:8100"})
    def test_readiness_probe_data_unhealthy(self, client):
        """Test degraded status when JuniperData is unreachable."""
        response = client.get("/v1/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["dependencies"]["juniper_data"]["status"] == "unhealthy"

    def test_readiness_probe_no_data_url(self, client):
        """Test readiness without JUNIPER_DATA_URL set."""
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("JUNIPER_DATA_URL", None)
            response = client.get("/v1/health/ready")
            assert response.status_code == 200
            body = response.json()
            # No juniper_data dependency when URL not set
            assert "juniper_data" not in body.get("dependencies", {})


@pytest.mark.unit
class TestProbeDependency:
    """Test the probe_dependency utility function."""

    def test_probe_healthy_service(self):
        """Test probing a healthy service."""
        with patch("api.models.health.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__ = lambda s: s
            mock_urlopen.return_value.__exit__ = lambda s, *a: None
            result = probe_dependency("Test Service", "http://localhost:8100/v1/health/live")
            assert result.status == "healthy"
            assert result.latency_ms is not None
            assert result.latency_ms >= 0
            assert result.name == "Test Service"

    def test_probe_unhealthy_service(self):
        """Test probing an unreachable service."""
        with patch("api.models.health.urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
            result = probe_dependency("Test Service", "http://localhost:9999/v1/health/live", timeout=1.0)
            assert result.status == "unhealthy"
            assert result.latency_ms is not None
            assert "ConnectionRefusedError" in result.message

    def test_probe_timeout(self):
        """Test probing a service that times out."""
        from urllib.error import URLError

        with patch("api.models.health.urllib.request.urlopen", side_effect=URLError("timeout")):
            result = probe_dependency("Slow Service", "http://localhost:8100/v1/health/live", timeout=0.1)
            assert result.status == "unhealthy"
            assert "URLError" in result.message


@pytest.mark.unit
class TestHealthModels:
    """Test Pydantic health models."""

    def test_dependency_status_healthy(self):
        dep = DependencyStatus(name="Test", status="healthy", latency_ms=2.5, message="ok")
        assert dep.model_dump()["status"] == "healthy"

    def test_dependency_status_not_configured(self):
        dep = DependencyStatus(name="Optional", status="not_configured")
        data = dep.model_dump()
        assert data["latency_ms"] is None
        assert data["message"] is None

    def test_readiness_response_serialization(self):
        dep = DependencyStatus(name="Data", status="healthy", latency_ms=1.0, message="ok")
        resp = ReadinessResponse(
            status="ready",
            version="0.4.0",
            service="juniper-cascor",
            dependencies={"juniper_data": dep},
            details={"network_loaded": True},
        )
        data = resp.model_dump()
        assert data["service"] == "juniper-cascor"
        assert data["dependencies"]["juniper_data"]["status"] == "healthy"
        assert data["details"]["network_loaded"] is True


@pytest.mark.unit
class TestResponseFormat:
    """Test that health responses use flat format (not envelope)."""

    def test_health_check_flat_response(self, client):
        """Health check returns flat JSON, no envelope wrapper."""
        response = client.get("/v1/health")
        body = response.json()
        # Flat format: top-level status field
        assert body["status"] == "ok"
        # Should NOT have envelope fields
        assert "data" not in body
        assert "meta" not in body

    def test_liveness_flat_response(self, client):
        """Liveness returns flat JSON."""
        response = client.get("/v1/health/live")
        body = response.json()
        assert body["status"] == "alive"
        assert "data" not in body

    def test_readiness_flat_response(self, client):
        """Readiness returns ReadinessResponse directly."""
        response = client.get("/v1/health/ready")
        body = response.json()
        # Flat ReadinessResponse fields at top level
        assert "status" in body
        assert "version" in body
        assert "service" in body
        assert "dependencies" in body
        # Should NOT have envelope
        assert "data" not in body
        assert "meta" not in body
