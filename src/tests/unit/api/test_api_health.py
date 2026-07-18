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
    settings = Settings(auto_start=False)
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
        assert body["version"] == "0.6.0"

    def test_health_includes_service_identifier(self, client):
        """API-02: /v1/health includes the ``service`` field naming this service.

        Part of the shared ``{status, version, service}`` base schema across
        juniper-data, juniper-cascor, and juniper-canopy so cross-service
        monitoring tools can tell health responses apart without parsing
        the URL.
        """
        response = client.get("/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["service"] == "juniper-cascor"

    def test_liveness_probe(self, client):
        """R1.2: GET /v1/health/live runs in-process tick + returns 200 with tick metadata."""
        response = client.get("/v1/health/live")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "alive"
        assert body["tick"] == "juniper-cascor"
        assert isinstance(body["duration_ms"], int)

    def test_liveness_503_when_lifecycle_missing(self, client):
        """R1.2 / seed-03: tick raises when lifecycle not bound → 503."""
        # Clear the lifecycle the lifespan installed.
        original = client.app.state.lifecycle
        client.app.state.lifecycle = None
        try:
            response = client.get("/v1/health/live")
            assert response.status_code == 503
            body = response.json()
            assert body["status"] == "unresponsive"
            assert body["tick"] == "juniper-cascor"
            assert "lifecycle" in body["error"]
        finally:
            client.app.state.lifecycle = original

    def test_liveness_503_when_heartbeat_stale(self, client):
        """R1.2 / seed-03: stale heartbeat → 503."""
        from api.routes import health as health_module

        # Force the lifecycle's last-tick timestamp to long-past so is_alive() is False.
        lifecycle = client.app.state.lifecycle
        with lifecycle._liveness_lock:
            lifecycle._liveness_last_tick_at = lifecycle._liveness_last_tick_at - (health_module.LIVENESS_STALENESS_SECONDS + 5)
        # Stop the heartbeat thread so it can't recover before our check.
        lifecycle.stop_liveness_heartbeat()

        response = client.get("/v1/health/live")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unresponsive"
        assert "stale" in body["error"]

    def test_readiness_probe_default(self, client):
        """Test GET /v1/health/ready with lifecycle bound → 200 + ready."""
        response = client.get("/v1/health/ready")
        assert response.status_code == 200
        assert response.headers.get("X-Juniper-Readiness") == "ready"
        body = response.json()
        assert body["status"] == "ready"
        assert body["version"] == "0.6.0"
        assert body["service"] == "juniper-cascor"
        assert "timestamp" in body
        assert body["details"]["network_loaded"] is False
        assert "training_state" in body["details"]
        assert body["dependencies"]["lifecycle"]["status"] == "healthy"

    def test_readiness_503_when_lifecycle_missing(self, client):
        """R1.2 / seed-02: lifecycle unbound → 503 + status=not_ready."""
        original = client.app.state.lifecycle
        client.app.state.lifecycle = None
        try:
            response = client.get("/v1/health/ready")
            assert response.status_code == 503
            assert response.headers.get("X-Juniper-Readiness") == "not_ready"
            body = response.json()
            assert body["status"] == "not_ready"
            assert body["dependencies"]["lifecycle"]["status"] == "unhealthy"
        finally:
            client.app.state.lifecycle = original

    def test_readiness_probe_with_lifecycle(self, client):
        """Test GET /v1/health/ready details surface mock lifecycle state."""

        class MockLifecycle:
            def has_model(self):
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
    def test_readiness_503_when_juniper_data_unhealthy(self, client):
        """R1.2 / seed-02: when JUNIPER_DATA_URL set + dep unhealthy → 503 not_ready."""
        response = client.get("/v1/health/ready")
        assert response.status_code == 503
        assert response.headers.get("X-Juniper-Readiness") == "not_ready"
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["dependencies"]["juniper_data"]["status"] == "unhealthy"

    def test_readiness_probe_no_data_url(self, client):
        """When JUNIPER_DATA_URL unset, juniper_data dep is skipped entirely → ready."""
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("JUNIPER_DATA_URL", None)
            response = client.get("/v1/health/ready")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "ready"
            assert "juniper_data" not in body.get("dependencies", {})


@pytest.mark.unit
class TestProbeDependency:
    """Test the probe_dependency utility function."""

    def test_probe_healthy_service(self):
        """Test probing a healthy service."""
        with patch("juniper_observability.health.probe.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__ = lambda s: s
            mock_urlopen.return_value.__exit__ = lambda s, *a: None
            result = probe_dependency("Test Service", "http://localhost:8100/v1/health/live")
            assert result.status == "healthy"
            assert result.latency_ms is not None
            assert result.latency_ms >= 0
            assert result.name == "Test Service"

    def test_probe_unhealthy_service(self):
        """Test probing an unreachable service."""
        with patch("juniper_observability.health.probe.urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
            result = probe_dependency("Test Service", "http://localhost:9999/v1/health/live", timeout=1.0)
            assert result.status == "unhealthy"
            assert result.latency_ms is not None
            assert "ConnectionRefusedError" in result.message

    def test_probe_timeout(self):
        """Test probing a service that times out."""
        from urllib.error import URLError

        with patch("juniper_observability.health.probe.urllib.request.urlopen", side_effect=URLError("timeout")):
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
            version="0.6.0",
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
        assert body["tick"] == "juniper-cascor"
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


@pytest.mark.unit
class TestBuildProvenance:
    """Build provenance on /v1/health + /v1/health/ready (stale-image detection).

    juniper-ml notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md — the image stamps
    ``JUNIPER_CASCOR_GIT_SHA`` / ``JUNIPER_CASCOR_BUILD_DATE`` env vars at build
    time (from build-args); the health endpoints surface them and the
    ``provenance`` accessor reads them so ``make doctor`` can detect when a
    running container has fallen behind its source.
    """

    def test_health_includes_provenance_null_outside_image(self, client, monkeypatch):
        """Outside a provenance-stamped image the fields are present but null."""
        monkeypatch.delenv("JUNIPER_CASCOR_GIT_SHA", raising=False)
        monkeypatch.delenv("JUNIPER_CASCOR_BUILD_DATE", raising=False)
        body = client.get("/v1/health").json()
        assert body["git_sha"] is None
        assert body["build_date"] is None

    def test_health_surfaces_baked_provenance(self, client, monkeypatch):
        """When the image baked the env vars, /v1/health reports them."""
        monkeypatch.setenv("JUNIPER_CASCOR_GIT_SHA", "abc1234")
        monkeypatch.setenv("JUNIPER_CASCOR_BUILD_DATE", "2026-06-14T00:00:00Z")
        body = client.get("/v1/health").json()
        assert body["git_sha"] == "abc1234"
        assert body["build_date"] == "2026-06-14T00:00:00Z"

    def test_readiness_surfaces_baked_provenance(self, client, monkeypatch):
        """The shared ReadinessResponse also carries git_sha/build_date."""
        monkeypatch.setenv("JUNIPER_CASCOR_GIT_SHA", "def5678")
        monkeypatch.setenv("JUNIPER_CASCOR_BUILD_DATE", "2026-06-14T01:02:03Z")
        body = client.get("/v1/health/ready").json()
        assert body["git_sha"] == "def5678"
        assert body["build_date"] == "2026-06-14T01:02:03Z"

    def test_readiness_provenance_null_outside_image(self, client, monkeypatch):
        """Readiness fields default to null with no provenance env present."""
        monkeypatch.delenv("JUNIPER_CASCOR_GIT_SHA", raising=False)
        monkeypatch.delenv("JUNIPER_CASCOR_BUILD_DATE", raising=False)
        body = client.get("/v1/health/ready").json()
        assert body["git_sha"] is None
        assert body["build_date"] is None

    def test_accessor_returns_none_when_unset(self, monkeypatch):
        from api import provenance

        monkeypatch.delenv("JUNIPER_CASCOR_GIT_SHA", raising=False)
        monkeypatch.delenv("JUNIPER_CASCOR_BUILD_DATE", raising=False)
        assert provenance.git_sha() is None
        assert provenance.build_date() is None

    def test_accessor_empty_string_is_none(self, monkeypatch):
        """A bare ``docker build`` leaves the env var empty-string → None."""
        from api import provenance

        monkeypatch.setenv("JUNIPER_CASCOR_GIT_SHA", "")
        monkeypatch.setenv("JUNIPER_CASCOR_BUILD_DATE", "")
        assert provenance.git_sha() is None
        assert provenance.build_date() is None

    def test_accessor_returns_value_when_set(self, monkeypatch):
        from api import provenance

        monkeypatch.setenv("JUNIPER_CASCOR_GIT_SHA", "deadbee")
        monkeypatch.setenv("JUNIPER_CASCOR_BUILD_DATE", "2026-06-14T12:00:00Z")
        assert provenance.git_sha() == "deadbee"
        assert provenance.build_date() == "2026-06-14T12:00:00Z"
