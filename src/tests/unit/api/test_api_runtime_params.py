"""Tests for PATCH /v1/training/params endpoint."""

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.lifecycle.manager import TrainingLifecycleManager
from api.settings import Settings

pytestmark = pytest.mark.unit


@pytest.fixture
def test_client():
    """Create a test client with lifecycle manager (lifespan runs)."""
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def test_client_with_network(test_client):
    """Create a test client with a network already created."""
    test_client.post("/v1/network", json={"input_size": 2, "output_size": 2})
    return test_client


# Use the existing test fixtures/client patterns from the repo's conftest.py


class TestUpdateTrainingParams:
    """Tests for the PATCH /v1/training/params endpoint."""

    def test_update_params_returns_404_without_network(self, test_client):
        """PATCH /v1/training/params returns 404 when no network exists."""
        response = test_client.patch("/v1/training/params", json={"learning_rate": 0.01})
        assert response.status_code == 404

    def test_update_learning_rate(self, test_client_with_network):
        """PATCH /v1/training/params updates learning_rate successfully."""
        response = test_client_with_network.patch("/v1/training/params", json={"learning_rate": 0.005})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["learning_rate"] == pytest.approx(0.005)

    def test_update_multiple_params(self, test_client_with_network):
        """PATCH /v1/training/params updates multiple parameters."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"learning_rate": 0.003, "correlation_threshold": 0.15},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["learning_rate"] == pytest.approx(0.003)
        assert data["data"]["correlation_threshold"] == pytest.approx(0.15)

    def test_update_params_empty_body_is_noop(self, test_client_with_network):
        """PATCH with empty body returns current params unchanged."""
        response = test_client_with_network.patch("/v1/training/params", json={})
        assert response.status_code == 200

    def test_patch_semantics_null_fields_ignored(self, test_client_with_network):
        """PATCH semantics: null/missing fields are not applied."""
        # Get current learning_rate
        before = test_client_with_network.get("/v1/training/params").json()["data"]["learning_rate"]
        # PATCH with only correlation_threshold
        test_client_with_network.patch("/v1/training/params", json={"correlation_threshold": 0.2})
        # learning_rate should be unchanged
        after = test_client_with_network.get("/v1/training/params").json()["data"]["learning_rate"]
        assert before == after

    def test_update_init_output_weights(self, test_client_with_network):
        """PATCH /v1/training/params updates init_output_weights on live network."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"init_output_weights": "random"},
        )
        assert response.status_code == 200
        assert test_client_with_network.app.state.lifecycle.network.init_output_weights == "random"

    def test_update_init_output_weights_rejects_invalid_value(self, test_client_with_network):
        """PATCH /v1/training/params rejects unsupported init_output_weights values."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"init_output_weights": "invalid"},
        )
        assert response.status_code == 422
