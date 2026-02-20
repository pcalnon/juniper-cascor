"""Tests for dataset API route."""

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
class TestDatasetRoute:
    """Test dataset metadata route."""

    def test_get_dataset_no_data(self, client):
        """GET /v1/dataset returns loaded=False when no data."""
        response = client.get("/v1/dataset")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["data"]["loaded"] is False

    def test_get_dataset_with_data(self, client):
        """GET /v1/dataset returns metadata after training data is set."""
        # Create network and start training with inline data
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        # Provide inline data via training start
        train_x = [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
        train_y = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        client.post(
            "/v1/training/start",
            json={
                "inline_data": {"train_x": train_x, "train_y": train_y},
            },
        )
        # Wait briefly for training to start then stop it
        import time

        time.sleep(0.2)
        client.post("/v1/training/stop")
        time.sleep(0.1)

        response = client.get("/v1/dataset")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["loaded"] is True
        assert body["data"]["train_samples"] == 4
        assert body["data"]["input_features"] == 2
        assert body["data"]["output_features"] == 2
