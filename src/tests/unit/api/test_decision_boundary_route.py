"""Tests for decision boundary API route."""

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
        # Load data via training start then stop
        train_x = [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
        train_y = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        client.post(
            "/v1/training/start",
            json={"inline_data": {"train_x": train_x, "train_y": train_y}},
        )
        import time

        time.sleep(0.2)
        client.post("/v1/training/stop")
        time.sleep(0.1)
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
