#!/usr/bin/env python
"""
Unit tests for api/routes/training.py to improve code coverage.

Covers:
- _get_lifecycle: HTTPException when lifecycle not initialized
- start_training: inline_data, dataset generator, params, epochs, error paths
- stop_training, pause_training, resume_training: error paths
- get_params: no network path
- _generate_spiral_data: default and custom parameters
"""

import os
import sys
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.app import create_app
from api.settings import Settings

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    """Create a test client with lifecycle manager (lifespan runs)."""
    settings = Settings()
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_network(client):
    """Create a test client with a network already created."""
    client.post("/v1/network", json={"input_size": 2, "output_size": 2})
    return client


class TestGetLifecycle:
    """Tests for _get_lifecycle helper."""

    def test_lifecycle_not_initialized_returns_503(self):
        """Should return 503 when lifecycle is not on app.state."""
        settings = Settings()
        app = create_app(settings)

        # Remove lifecycle from app state
        with TestClient(app) as c:
            # Temporarily remove lifecycle
            lifecycle = c.app.state.lifecycle
            del c.app.state.lifecycle

            response = c.post("/v1/training/start")
            assert response.status_code == 503
            assert "Lifecycle manager not initialized" in response.json()["detail"]

            # Restore lifecycle for clean shutdown
            c.app.state.lifecycle = lifecycle


class TestStartTraining:
    """Tests for POST /training/start."""

    def test_start_training_with_inline_data(self, client_with_network):
        """start_training should accept inline_data in request body."""
        response = client_with_network.post(
            "/v1/training/start",
            json={
                "inline_data": {
                    "train_x": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                    "train_y": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                }
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["status"] == "training_started"

        # Wait for background training to settle
        import time

        time.sleep(0.5)

    def test_start_training_with_inline_data_and_validation(self, client_with_network):
        """start_training should accept inline_data with validation data."""
        response = client_with_network.post(
            "/v1/training/start",
            json={
                "inline_data": {
                    "train_x": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                    "train_y": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                    "val_x": [[0.9, 1.0], [1.1, 1.2]],
                    "val_y": [[1.0, 0.0], [0.0, 1.0]],
                }
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["status"] == "training_started"

        import time

        time.sleep(0.5)

    def test_start_training_with_dataset_generator(self, client_with_network):
        """start_training should accept dataset generator specification."""
        response = client_with_network.post(
            "/v1/training/start",
            json={
                "dataset": {
                    "generator": "spiral",
                    "params": {
                        "n_per_spiral": 20,
                        "n_spirals": 2,
                    },
                }
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["status"] == "training_started"

        import time

        time.sleep(0.5)

    def test_start_training_with_params(self, client_with_network):
        """start_training should accept training parameter overrides."""
        response = client_with_network.post(
            "/v1/training/start",
            json={
                "inline_data": {
                    "train_x": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                    "train_y": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                },
                "params": {"max_hidden_units": 5},
            },
        )
        assert response.status_code == 200

        import time

        time.sleep(0.5)

    def test_start_training_with_epochs_override(self, client_with_network):
        """start_training should accept epochs override."""
        response = client_with_network.post(
            "/v1/training/start",
            json={
                "inline_data": {
                    "train_x": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                    "train_y": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                },
                "epochs": 5,
            },
        )
        assert response.status_code == 200

        import time

        time.sleep(0.5)

    def test_start_training_without_network_returns_error(self, client):
        """start_training should return 409 when no network exists."""
        response = client.post(
            "/v1/training/start",
            json={
                "inline_data": {
                    "train_x": [[0.1, 0.2]],
                    "train_y": [[1.0, 0.0]],
                }
            },
        )
        assert response.status_code == 409

    def test_start_training_no_body(self, client_with_network):
        """start_training without body should fail with state error."""
        response = client_with_network.post("/v1/training/start")
        assert response.status_code == 409
        assert "cannot be started" in response.json()["detail"].lower()


class TestPauseTraining:
    """Tests for POST /training/pause."""

    def test_pause_training_not_active_returns_409(self, client):
        """Pause should return 409 when training is not active."""
        response = client.post("/v1/training/pause")
        assert response.status_code == 409
        assert "cannot be paused" in response.json()["detail"].lower()


class TestResumeTraining:
    """Tests for POST /training/resume."""

    def test_resume_training_not_paused_returns_409(self, client):
        """Resume should return 409 when training is not paused."""
        response = client.post("/v1/training/resume")
        assert response.status_code == 409
        assert "cannot be resumed" in response.json()["detail"].lower()


class TestGetParams:
    """Tests for GET /training/params."""

    def test_get_params_no_network_returns_404(self, client):
        """get_params should return 404 when no network created."""
        response = client.get("/v1/training/params")
        assert response.status_code == 404
        assert "No network" in response.json()["detail"]

    def test_get_params_with_network(self, client_with_network):
        """get_params should return training parameters when network exists."""
        response = client_with_network.get("/v1/training/params")
        assert response.status_code == 200
        body = response.json()
        assert "data" in body


class TestGenerateSpiralData:
    """Tests for _generate_spiral_data function."""

    def test_generate_spiral_data_default_params(self):
        """_generate_spiral_data should generate data with default parameters."""
        from api.routes.training import _generate_spiral_data

        x, y = _generate_spiral_data({})
        assert x.shape[0] == 200  # 100 per spiral * 2 spirals
        assert x.shape[1] == 2
        assert y.shape[0] == 200
        assert y.shape[1] == 2

    def test_generate_spiral_data_custom_params(self):
        """_generate_spiral_data should respect custom parameters."""
        from api.routes.training import _generate_spiral_data

        x, y = _generate_spiral_data({"n_per_spiral": 50, "n_spirals": 3})
        assert x.shape[0] == 150  # 50 * 3
        assert x.shape[1] == 2
        assert y.shape[0] == 150
        assert y.shape[1] == 3  # one-hot for 3 classes

    def test_generate_spiral_data_single_spiral(self):
        """_generate_spiral_data should work with a single spiral."""
        from api.routes.training import _generate_spiral_data

        x, y = _generate_spiral_data({"n_per_spiral": 30, "n_spirals": 1})
        assert x.shape[0] == 30
        assert y.shape[1] == 1

    def test_generate_spiral_data_returns_float32_tensors(self):
        """_generate_spiral_data should return float32 tensors."""
        import torch

        from api.routes.training import _generate_spiral_data

        x, y = _generate_spiral_data({})
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32
