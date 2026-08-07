"""Tests for decision boundary API route."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.lifecycle.manager import TrainingLifecycleManager
from api.settings import Settings


@pytest.fixture
def client():
    """Create a test client with lifecycle manager."""
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.mark.unit
class TestDecisionBoundaryRoute:
    """Test decision boundary route."""

    def test_decision_boundary_no_network(self, client):
        """GET /v1/decision-boundary returns 404 when no network."""
        response = client.get("/v1/decision-boundary")
        assert response.status_code == 404

    def test_decision_boundary_no_data(self, client):
        """GET /v1/decision-boundary returns 404 when no training data."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        response = client.get("/v1/decision-boundary")
        assert response.status_code == 404

    def test_decision_boundary_with_data(self, client):
        """GET /v1/decision-boundary returns grid data after training data loaded."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        # Load data via training start, mock _run_training to prevent
        # a background thread that outlives the test and blocks process exit.
        train_x = [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
        train_y = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        with patch.object(TrainingLifecycleManager, "_run_training"):
            client.post(
                "/v1/training/start",
                json={"inline_data": {"train_x": train_x, "train_y": train_y}},
            )
        client.post("/v1/training/stop")
        client.post("/v1/training/reset")

        response = client.get("/v1/decision-boundary?resolution=10")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["data"]["resolution"] == 10
        assert "x_range" in body["data"]
        assert "y_range" in body["data"]
        assert "predictions" in body["data"]

    def test_decision_boundary_resolution_param(self, client):
        """GET /v1/decision-boundary validates resolution parameter."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        # Invalid resolution (too low)
        response = client.get("/v1/decision-boundary?resolution=2")
        assert response.status_code == 422

    def test_decision_boundary_resolution_too_high(self, client):
        """GET /v1/decision-boundary validates max resolution."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        response = client.get("/v1/decision-boundary?resolution=300")
        assert response.status_code == 422

    def test_decision_boundary_lifecycle_missing_returns_503(self, client):
        """GET /v1/decision-boundary returns 503 when lifecycle is uninitialized."""
        lifecycle = client.app.state.lifecycle
        del client.app.state.lifecycle
        try:
            response = client.get("/v1/decision-boundary")
            assert response.status_code == 503
            body = response.json()
            assert "Lifecycle manager not initialized" in body["error"]["message"]
        finally:
            client.app.state.lifecycle = lifecycle

    def test_decision_boundary_compute_failure_returns_500(self, client):
        """GET /v1/decision-boundary returns 500 when boundary computation yields None."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        train_x = [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
        train_y = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        with patch.object(TrainingLifecycleManager, "_run_training"):
            client.post(
                "/v1/training/start",
                json={"inline_data": {"train_x": train_x, "train_y": train_y}},
            )
        client.post("/v1/training/stop")
        client.post("/v1/training/reset")

        with patch.object(client.app.state.lifecycle, "get_decision_boundary", return_value=None):
            response = client.get("/v1/decision-boundary?resolution=10")
        assert response.status_code == 500
        assert "Failed to compute decision boundary" in response.json()["error"]["message"]
