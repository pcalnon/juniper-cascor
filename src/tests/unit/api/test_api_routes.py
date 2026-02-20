"""Tests for network and training API routes."""

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings


@pytest.fixture
def client():
    """Create a test client with lifecycle manager (lifespan runs)."""
    settings = Settings()
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.mark.unit
class TestNetworkRoutes:
    """Test network CRUD routes."""

    def test_create_network(self, client):
        """POST /v1/network creates a network."""
        response = client.post(
            "/v1/network",
            json={
                "input_size": 2,
                "output_size": 2,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["data"]["input_size"] == 2
        assert body["data"]["output_size"] == 2

    def test_get_network(self, client):
        """GET /v1/network returns network info."""
        client.post("/v1/network", json={"input_size": 3, "output_size": 2})
        response = client.get("/v1/network")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["input_size"] == 3

    def test_get_network_not_created(self, client):
        """GET /v1/network returns 404 when no network."""
        response = client.get("/v1/network")
        assert response.status_code == 404

    def test_delete_network(self, client):
        """DELETE /v1/network deletes the network."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        response = client.delete("/v1/network")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["deleted"] is True

    def test_delete_network_not_created(self, client):
        """DELETE /v1/network when no network doesn't crash."""
        response = client.delete("/v1/network")
        # delete_network should work even without a network (reset handles it)
        assert response.status_code == 200

    def test_get_topology(self, client):
        """GET /v1/network/topology returns topology."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        response = client.get("/v1/network/topology")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["input_size"] == 2
        assert "output_weights" in body["data"]

    def test_get_topology_not_created(self, client):
        """GET /v1/network/topology returns 404 when no network."""
        response = client.get("/v1/network/topology")
        assert response.status_code == 404

    def test_get_stats(self, client):
        """GET /v1/network/stats returns statistics."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        response = client.get("/v1/network/stats")
        assert response.status_code == 200
        body = response.json()
        assert "total_hidden_units" in body["data"]

    def test_get_stats_not_created(self, client):
        """GET /v1/network/stats returns 404 when no network."""
        response = client.get("/v1/network/stats")
        assert response.status_code == 404

    def test_create_network_with_defaults(self, client):
        """POST /v1/network with no body uses defaults."""
        response = client.post("/v1/network", json={})
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["input_size"] == 2
        assert body["data"]["output_size"] == 2


@pytest.mark.unit
class TestTrainingRoutes:
    """Test training control routes."""

    def test_get_status_no_network(self, client):
        """GET /v1/training/status works without network."""
        response = client.get("/v1/training/status")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["network_loaded"] is False

    def test_get_status_with_network(self, client):
        """GET /v1/training/status reflects network state."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        response = client.get("/v1/training/status")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["network_loaded"] is True

    def test_start_training_no_network(self, client):
        """POST /v1/training/start fails without network."""
        response = client.post("/v1/training/start")
        assert response.status_code == 409

    def test_stop_training(self, client):
        """POST /v1/training/stop returns success."""
        response = client.post("/v1/training/stop")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["status"] == "stop_requested"

    def test_pause_training_not_active(self, client):
        """POST /v1/training/pause fails when not active."""
        response = client.post("/v1/training/pause")
        assert response.status_code == 409

    def test_resume_training_not_paused(self, client):
        """POST /v1/training/resume fails when not paused."""
        response = client.post("/v1/training/resume")
        assert response.status_code == 409

    def test_reset_training(self, client):
        """POST /v1/training/reset returns success."""
        response = client.post("/v1/training/reset")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["status"] == "reset"

    def test_get_params_no_network(self, client):
        """GET /v1/training/params returns 404 without network."""
        response = client.get("/v1/training/params")
        assert response.status_code == 404

    def test_get_params_with_network(self, client):
        """GET /v1/training/params returns parameters."""
        client.post(
            "/v1/network",
            json={
                "input_size": 2,
                "output_size": 2,
                "learning_rate": 0.05,
            },
        )
        response = client.get("/v1/training/params")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["learning_rate"] == 0.05
