"""Integration tests for the full API lifecycle.

Tests the complete workflow: create network -> load data -> train -> monitor -> stop -> cleanup.
These tests use real CasCor network instances (via TestClient, no external server).
"""

import time

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


# Simple linearly separable data for fast training
_TRAIN_X = [
    [-1.0, -1.0],
    [-0.8, -0.9],
    [-0.9, -0.7],
    [-1.1, -0.8],
    [1.0, 1.0],
    [0.8, 0.9],
    [0.9, 0.7],
    [1.1, 0.8],
]
_TRAIN_Y = [
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
]


@pytest.mark.integration
class TestFullLifecycle:
    """Test complete API lifecycle."""

    def test_create_train_stop_lifecycle(self, client):
        """Create network -> start training with inline data -> stop -> check metrics."""
        # 1. Create network
        resp = client.post(
            "/v1/network",
            json={"input_size": 2, "output_size": 2, "epochs_max": 5},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["input_size"] == 2

        # 2. Start training with inline data
        resp = client.post(
            "/v1/training/start",
            json={"inline_data": {"train_x": _TRAIN_X, "train_y": _TRAIN_Y}},
        )
        assert resp.status_code == 200

        # 3. Wait for training to progress
        time.sleep(1.0)

        # 4. Check status
        resp = client.get("/v1/training/status")
        assert resp.status_code == 200
        status = resp.json()["data"]
        assert status["network_loaded"] is True

        # 5. Stop training
        resp = client.post("/v1/training/stop")
        assert resp.status_code == 200
        time.sleep(0.5)

        # 6. Dataset should be loaded
        resp = client.get("/v1/dataset")
        assert resp.status_code == 200
        assert resp.json()["data"]["loaded"] is True
        assert resp.json()["data"]["train_samples"] == 8

    def test_create_train_wait_for_completion(self, client):
        """Create network with tiny params -> train to completion."""
        # Create network with very small training parameters
        client.post(
            "/v1/network",
            json={
                "input_size": 2,
                "output_size": 2,
                "epochs_max": 3,
                "output_epochs": 2,
                "candidate_epochs": 2,
                "max_hidden_units": 1,
                "patience": 1,
            },
        )

        # Start training
        client.post(
            "/v1/training/start",
            json={"inline_data": {"train_x": _TRAIN_X, "train_y": _TRAIN_Y}},
        )

        # Wait for training to complete (tiny params = fast completion)
        for _ in range(30):
            time.sleep(0.5)
            resp = client.get("/v1/training/status")
            sm_status = resp.json()["data"]["state_machine"]["status"]
            if sm_status.upper() in ("COMPLETED", "FAILED", "STOPPED"):
                break

        # Verify training ended
        resp = client.get("/v1/training/status")
        assert resp.status_code == 200
        sm_status = resp.json()["data"]["state_machine"]["status"]
        assert sm_status.upper() in ("COMPLETED", "FAILED", "STOPPED")

    def test_metrics_endpoint_after_training(self, client):
        """Metrics endpoint returns data after training starts."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2, "epochs_max": 3})
        client.post(
            "/v1/training/start",
            json={"inline_data": {"train_x": _TRAIN_X, "train_y": _TRAIN_Y}},
        )
        time.sleep(0.5)
        client.post("/v1/training/stop")
        time.sleep(0.3)

        resp = client.get("/v1/metrics")
        assert resp.status_code == 200

    def test_decision_boundary_after_data_loaded(self, client):
        """Decision boundary works after data is loaded."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2, "epochs_max": 3})
        client.post(
            "/v1/training/start",
            json={"inline_data": {"train_x": _TRAIN_X, "train_y": _TRAIN_Y}},
        )
        time.sleep(0.5)
        client.post("/v1/training/stop")
        time.sleep(0.3)
        client.post("/v1/training/reset")

        resp = client.get("/v1/decision-boundary?resolution=10")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["resolution"] == 10

    def test_reset_clears_state(self, client):
        """Reset clears training state."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        client.post(
            "/v1/training/start",
            json={"inline_data": {"train_x": _TRAIN_X, "train_y": _TRAIN_Y}},
        )
        time.sleep(0.3)
        client.post("/v1/training/stop")
        time.sleep(0.2)

        resp = client.post("/v1/training/reset")
        assert resp.status_code == 200

        resp = client.get("/v1/training/status")
        training_state = resp.json()["data"]["training_state"]
        assert training_state["status"] == "Stopped"
        assert training_state["current_epoch"] == 0

    def test_delete_network_cleanup(self, client):
        """Deleting network cleans up everything."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})

        resp = client.delete("/v1/network")
        assert resp.status_code == 200

        resp = client.get("/v1/network")
        assert resp.status_code == 404

    def test_spiral_data_generator(self, client):
        """Training with spiral data generator."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2, "epochs_max": 3})
        resp = client.post(
            "/v1/training/start",
            json={
                "dataset": {
                    "source": "inline",
                    "generator": "spiral",
                    "params": {"n_per_spiral": 20, "n_spirals": 2},
                },
            },
        )
        assert resp.status_code == 200
        time.sleep(0.5)
        client.post("/v1/training/stop")
        time.sleep(0.2)

        resp = client.get("/v1/dataset")
        assert resp.json()["data"]["loaded"] is True
        assert resp.json()["data"]["train_samples"] == 40  # 20 * 2 spirals
