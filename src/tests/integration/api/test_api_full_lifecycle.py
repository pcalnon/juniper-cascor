"""Integration tests for the full API lifecycle.

Tests the complete workflow: create network -> load data -> train -> monitor -> stop -> cleanup.
These tests use real CasCor network instances (via TestClient, no external server).
"""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings


def wait_for_state(client, expected_states, *, timeout=5.0, poll_interval=0.1):
    """Poll training status until state machine reaches one of expected_states.

    Args:
        client: TestClient instance
        expected_states: Tuple of state strings to wait for (e.g., ("STARTED", "COMPLETED", "FAILED"))
        timeout: Maximum wait time in seconds
        poll_interval: Time between polls in seconds

    Returns:
        The final status response JSON

    Raises:
        TimeoutError: If expected state not reached within timeout
    """
    deadline = time.monotonic() + timeout
    sm_status = None
    while time.monotonic() < deadline:
        resp = client.get("/v1/training/status")
        data = resp.json()
        sm_status = data["data"]["state_machine"]["status"].upper()
        if sm_status in {s.upper() for s in expected_states}:
            return data
        time.sleep(poll_interval)
    raise TimeoutError(f"Training state did not reach {expected_states} within {timeout}s. " f"Last status: {sm_status}")


@pytest.fixture
def client():
    """Create a test client with lifecycle manager.

    Uses a daemon-thread approach for TestClient exit to prevent the 60s hang
    from anyio's blocking portal waiting on the ASGI event loop to shut down.
    The event loop blocks because the training thread (in ThreadPoolExecutor)
    doesn't respond to cancellation -- network.fit() has no cooperative
    cancellation check.
    """
    settings = Settings()
    app = create_app(settings)
    tc = TestClient(app)
    tc.__enter__()
    yield tc

    # Signal all background components to stop
    lifecycle = getattr(app.state, "lifecycle", None)
    if lifecycle:
        lifecycle._stop_event.set()
        if getattr(lifecycle, "_executor", None):
            lifecycle._executor.shutdown(wait=False, cancel_futures=True)

    coord = getattr(app.state, "worker_coordinator", None)
    if coord:
        coord.shutdown()

    # Exit TestClient in a daemon thread with a short timeout.
    # TestClient.__exit__ blocks indefinitely on anyio portal thread join
    # when the ASGI event loop has pending work (training thread).
    # The session-scoped _force_clean_exit fixture handles final cleanup.
    exit_thread = threading.Thread(target=lambda: tc.__exit__(None, None, None), daemon=True)
    exit_thread.start()
    exit_thread.join(timeout=5)


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
            json={"input_size": 2, "output_size": 2, "epochs_max": 5, "candidate_epochs": 2, "output_epochs": 2, "patience": 1},
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
        data = wait_for_state(client, ("STARTED", "COMPLETED", "FAILED"), timeout=5.0)

        # 4. Check status
        status = data["data"]
        assert status["network_loaded"] is True

        # 5. Stop training
        resp = client.post("/v1/training/stop")
        assert resp.status_code == 200
        wait_for_state(client, ("STOPPED", "COMPLETED", "FAILED"), timeout=5.0)

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
        data = wait_for_state(client, ("COMPLETED", "FAILED", "STOPPED"), timeout=15.0)

        # Verify training ended
        sm_status = data["data"]["state_machine"]["status"]
        assert sm_status.upper() in ("COMPLETED", "FAILED", "STOPPED")

    def test_metrics_endpoint_after_training(self, client):
        """Metrics endpoint returns data after training starts."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2, "epochs_max": 3, "candidate_epochs": 2, "output_epochs": 2, "patience": 1})
        client.post(
            "/v1/training/start",
            json={"inline_data": {"train_x": _TRAIN_X, "train_y": _TRAIN_Y}},
        )
        wait_for_state(client, ("STARTED", "COMPLETED", "FAILED"), timeout=5.0)
        client.post("/v1/training/stop")
        wait_for_state(client, ("STOPPED", "COMPLETED", "FAILED"), timeout=5.0)

        resp = client.get("/v1/metrics")
        assert resp.status_code == 200

    def test_decision_boundary_after_data_loaded(self, client):
        """Decision boundary works after data is loaded."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2, "epochs_max": 3, "candidate_epochs": 2, "output_epochs": 2, "patience": 1})
        client.post(
            "/v1/training/start",
            json={"inline_data": {"train_x": _TRAIN_X, "train_y": _TRAIN_Y}},
        )
        wait_for_state(client, ("STARTED", "COMPLETED", "FAILED"), timeout=5.0)
        client.post("/v1/training/stop")
        wait_for_state(client, ("STOPPED", "COMPLETED", "FAILED"), timeout=5.0)
        client.post("/v1/training/reset")

        resp = client.get("/v1/decision-boundary?resolution=10")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["resolution"] == 10

    def test_reset_clears_state(self, client):
        """Reset clears training state."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2, "candidate_epochs": 2, "output_epochs": 2, "patience": 1})
        client.post(
            "/v1/training/start",
            json={"inline_data": {"train_x": _TRAIN_X, "train_y": _TRAIN_Y}},
        )
        # Wait for training to COMPLETE (not just start) to avoid race
        # between the training thread updating state and the reset clearing it.
        wait_for_state(client, ("COMPLETED", "FAILED"), timeout=10.0)

        resp = client.post("/v1/training/reset")
        assert resp.status_code == 200

        resp = client.get("/v1/training/status")
        training_state = resp.json()["data"]["training_state"]
        assert training_state["status"] == "Stopped"
        assert training_state["current_epoch"] == 0

    def test_delete_network_cleanup(self, client):
        """Deleting network cleans up everything."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2, "candidate_epochs": 2, "output_epochs": 2, "patience": 1})

        resp = client.delete("/v1/network")
        assert resp.status_code == 200

        resp = client.get("/v1/network")
        assert resp.status_code == 404

    def test_spiral_data_generator(self, client):
        """Training with spiral data generator."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2, "epochs_max": 3, "candidate_epochs": 2, "output_epochs": 2, "patience": 1})
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
        wait_for_state(client, ("STARTED", "COMPLETED", "FAILED"), timeout=5.0)
        client.post("/v1/training/stop")
        wait_for_state(client, ("STOPPED", "COMPLETED", "FAILED"), timeout=5.0)

        resp = client.get("/v1/dataset")
        assert resp.json()["data"]["loaded"] is True
        assert resp.json()["data"]["train_samples"] == 40  # 20 * 2 spirals
