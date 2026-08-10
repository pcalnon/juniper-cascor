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
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_network(client):
    """Create a test client with a network already created."""
    client.post("/v1/network", json={"input_size": 2, "output_size": 2})
    yield client
    # Stop any running training before client teardown triggers lifespan shutdown.
    # Without this, the background training thread blocks TestClient exit.
    client.post("/v1/training/stop")


class TestGetLifecycle:
    """Tests for _get_lifecycle helper."""

    def test_lifecycle_not_initialized_returns_503(self):
        """Should return 503 when lifecycle is not on app.state."""
        settings = Settings(auto_start=False)
        app = create_app(settings)

        # Remove lifecycle from app state
        with TestClient(app) as c:
            # Temporarily remove lifecycle
            lifecycle = c.app.state.lifecycle
            del c.app.state.lifecycle

            response = c.post("/v1/training/start")
            assert response.status_code == 503
            assert "Lifecycle manager not initialized" in response.json()["error"]["message"]

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
                },
                "epochs": 1,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["status"] == "training_started"

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
                },
                "epochs": 1,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["status"] == "training_started"

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
                },
                "epochs": 1,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["status"] == "training_started"

    def test_start_training_unsupported_generator_rejected(self, client_with_network):
        """W-1: a non-spiral dataset.generator is rejected 422 at the request boundary.

        Pre-W-1 the route materialized only ``generator == "spiral"`` and silently
        IGNORED every other value, so this request fell through to a downstream 409
        only because the fixture holds no staged/retained data. Post-W-1 the route
        rejects it up front with guidance naming the staging endpoint, and the
        spiral generator is never invoked.
        """
        with patch("api.routes.training._generate_spiral_data") as spiral_gen:
            response = client_with_network.post(
                "/v1/training/start",
                json={
                    "dataset": {
                        "generator": "xor",
                        "params": {"n_samples": 20},
                    },
                    "epochs": 1,
                },
            )
        assert response.status_code == 422
        detail = response.json()["error"]["message"]
        assert "not supported" in detail
        assert "/v1/training/dataset" in detail
        spiral_gen.assert_not_called()

    def test_start_training_nonspiral_rejected_even_with_retained_data(self, client_with_network):
        """W-1's sharp arm: the silent-wrong-data class is closed.

        Seed the lifecycle with retained inline data (a completed start), then post a
        start naming ``generator: "xor"``. Pre-W-1 the generator was silently dropped
        and training STARTED on the retained inline data — succeeding on the wrong
        dataset. Post-W-1 the request 422s before the lifecycle is touched and no new
        training run begins.
        """
        seed = client_with_network.post(
            "/v1/training/start",
            json={
                "inline_data": {
                    "train_x": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                    "train_y": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                },
                "epochs": 1,
            },
        )
        assert seed.status_code == 200
        client_with_network.post("/v1/training/stop")

        response = client_with_network.post(
            "/v1/training/start",
            json={
                "dataset": {"generator": "xor", "params": {"n_samples": 20}},
                "epochs": 1,
            },
        )
        assert response.status_code == 422
        assert "/v1/training/dataset" in response.json()["error"]["message"]
        status = client_with_network.get("/v1/training/status").json()["data"]
        assert status["training_active"] is False

    def test_start_training_generator_none_keeps_prior_meaning(self, client_with_network):
        """W-1 scope guard: ``dataset`` present with ``generator: null`` is NOT the
        rejected class — it keeps its pre-W-1 fall-through (no generator requested,
        inline/staged/retained data decide; here none exists, so the downstream 409)."""
        response = client_with_network.post(
            "/v1/training/start",
            json={"dataset": {"source": "juniper-data", "params": {}}, "epochs": 1},
        )
        assert response.status_code == 409
        assert "Training cannot be started" in response.json()["error"]["message"]

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
                "epochs": 1,
            },
        )
        assert response.status_code == 200

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

    def test_start_training_without_network_creates_it_from_inline_data(self, client):
        """PR-B (training-start diagnosis 2026-07-09): a start with data but no
        network creates the network from the data dims (``_auto_start_training``
        parity) instead of returning 409 'No network created'."""
        response = client.post(
            "/v1/training/start",
            json={
                "inline_data": {
                    "train_x": [[0.1, 0.2], [0.3, 0.4]],
                    "train_y": [[1.0, 0.0], [0.0, 1.0]],
                },
                "epochs": 2,
            },
        )
        assert response.status_code == 200
        net = client.get("/v1/network")
        assert net.status_code == 200
        info = net.json()["data"]
        assert info["input_size"] == 2
        assert info["output_size"] == 2
        # Stop the background run before client teardown triggers lifespan shutdown.
        client.post("/v1/training/stop")

    def test_start_training_no_body(self, client_with_network):
        """start_training without body should fail with state error."""
        response = client_with_network.post("/v1/training/start")
        assert response.status_code == 409
        assert "cannot be started" in response.json()["error"]["message"].lower()

    def test_start_training_409_surfaces_specific_reason(self, client_with_network):
        """The 409 detail must name *why* the start was rejected (here: no dataset
        loaded), not a generic 'current state' string. The generic message previously
        masked real causes — e.g. a juniper-data fetch failure surfaced as a bogus state
        error. See notes/CASCOR_STARTUP_SECRET_INDIRECTION_INVESTIGATION_2026-06-14.md (3.4)."""
        response = client_with_network.post("/v1/training/start")
        assert response.status_code == 409
        msg = response.json()["error"]["message"]
        assert "Training data not provided" in msg, msg

    def test_start_training_rejects_unknown_params(self, client_with_network):
        """SEC-07 regression (supersedes CR-023): unknown keys in ``params``
        are now rejected by Pydantic at the request boundary (422) instead
        of silently dropped. A whitelisted key plus any unknown key must
        fail-closed so callers learn about the typo / attack attempt."""
        response = client_with_network.post(
            "/v1/training/start",
            json={
                "inline_data": {
                    "train_x": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                    "train_y": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                },
                "params": {
                    "learning_rate": 0.01,  # allowed
                    "evil_injection_key": "pwned",  # rejected
                },
                "epochs": 1,
            },
        )
        assert response.status_code == 422
        body = response.json()
        assert any("evil_injection_key" in str(err) for err in body.get("detail", [])), body

    def test_start_training_accepts_all_known_params(self, client_with_network):
        """SEC-07: TrainingParams model forwards every known key unchanged."""
        lifecycle = client_with_network.app.state.lifecycle
        with patch.object(lifecycle, "start_training", wraps=lifecycle.start_training) as spy:
            response = client_with_network.post(
                "/v1/training/start",
                json={
                    "inline_data": {
                        "train_x": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                        "train_y": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                    },
                    "params": {"learning_rate": 0.01, "patience": 5},
                    "epochs": 1,
                },
            )
        assert response.status_code == 200
        forwarded_kwargs = spy.call_args.kwargs
        assert forwarded_kwargs.get("learning_rate") == 0.01
        assert forwarded_kwargs.get("patience") == 5

    def test_start_training_while_investigating_returns_409(self, client_with_network):
        """CAN-015d: start while Investigating must 409 with the specific reason string.

        Manager-level RuntimeError is covered in lifecycle unit tests; this pins the
        HTTP mapping so Canopy sees a actionable 409 (not a generic 500).
        Orthogonal to create_network-while-Investigating and network PAUSED/REPLAYING guards.
        """
        lifecycle = client_with_network.app.state.lifecycle
        assert lifecycle.state_machine.mark_investigating()
        response = client_with_network.post(
            "/v1/training/start",
            json={
                "inline_data": {
                    "train_x": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                    "train_y": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                },
            },
        )
        assert response.status_code == 409
        msg = response.json()["error"]["message"]
        assert "Investigating" in msg
        assert lifecycle.state_machine.is_investigating()

    def test_start_training_while_replaying_returns_409(self, client_with_network):
        """CAN-015c: start while Replaying must 409 telling the client to stop replay first."""
        lifecycle = client_with_network.app.state.lifecycle
        assert lifecycle.state_machine.mark_replaying()
        response = client_with_network.post(
            "/v1/training/start",
            json={
                "inline_data": {
                    "train_x": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                    "train_y": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                },
            },
        )
        assert response.status_code == 409
        msg = response.json()["error"]["message"]
        assert "replaying" in msg.lower()
        assert lifecycle.state_machine.is_replaying()


class TestStopTraining:
    """Tests for POST /training/stop."""

    def test_stop_training_while_investigating_returns_409(self, client):
        """Stop while Investigating must 409 — not silently desync FSM."""
        lifecycle = client.app.state.lifecycle
        lifecycle.state_machine.mark_investigating()
        try:
            response = client.post("/v1/training/stop")
            assert response.status_code == 409
            assert "cannot be stopped" in response.json()["error"]["message"].lower()
            assert lifecycle.state_machine.is_investigating()
        finally:
            from api.lifecycle.state_machine import Command

            lifecycle.state_machine.handle_command(Command.RESET)

    def test_stop_training_while_replaying_returns_409(self, client):
        """Stop while Replaying must 409 and leave FSM in REPLAYING."""
        lifecycle = client.app.state.lifecycle
        lifecycle.state_machine.mark_replaying()
        try:
            response = client.post("/v1/training/stop")
            assert response.status_code == 409
            assert "cannot be stopped" in response.json()["error"]["message"].lower()
            assert lifecycle.state_machine.is_replaying()
        finally:
            from api.lifecycle.state_machine import Command

            lifecycle.state_machine.handle_command(Command.RESET)


class TestPauseTraining:
    """Tests for POST /training/pause."""

    def test_pause_training_not_active_returns_409(self, client):
        """Pause should return 409 when training is not active."""
        response = client.post("/v1/training/pause")
        assert response.status_code == 409
        assert "cannot be paused" in response.json()["error"]["message"].lower()


class TestResumeTraining:
    """Tests for POST /training/resume."""

    def test_resume_training_not_paused_returns_409(self, client):
        """Resume should return 409 when training is not paused."""
        response = client.post("/v1/training/resume")
        assert response.status_code == 409
        assert "cannot be resumed" in response.json()["error"]["message"].lower()


class TestGetParams:
    """Tests for GET /training/params."""

    def test_get_params_no_network_returns_404(self, client):
        """get_params should return 404 when no network created."""
        response = client.get("/v1/training/params")
        assert response.status_code == 404
        assert "No network" in response.json()["error"]["message"]

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

    # ------------------------------------------------------------------ #
    # F-P4-1 regression arms: param fidelity + the radius-10 scale fix.
    # The unit-radius fallback pinned candidate correlation at ~2.7e-4
    # (tanh linear regime vs an x-orthogonal residual), terminating every
    # service spiral run below_threshold with zero hidden units.
    # ------------------------------------------------------------------ #

    def test_generate_spiral_data_canonical_param_name(self):
        """F-P4-1: the juniper-data name n_points_per_spiral is honored (previously silently ignored)."""
        from api.routes.training import _generate_spiral_data

        x, y = _generate_spiral_data({"n_points_per_spiral": 40, "n_spirals": 2})
        assert x.shape[0] == 80
        assert y.shape[1] == 2

    def test_generate_spiral_data_legacy_param_name_still_accepted(self):
        """The legacy n_per_spiral key keeps working (canonical wins when both are present)."""
        from api.routes.training import _generate_spiral_data

        x, _ = _generate_spiral_data({"n_per_spiral": 30})
        assert x.shape[0] == 60
        x, _ = _generate_spiral_data({"n_points_per_spiral": 20, "n_per_spiral": 30})
        assert x.shape[0] == 40

    def test_generate_spiral_data_default_radius_10_scale(self):
        """F-P4-1: default scale is the classic radius 10 (unit radius was degenerate for candidate training)."""
        from api.routes.training import _generate_spiral_data

        x, _ = _generate_spiral_data({})
        assert 9.5 <= float(x.abs().max()) <= 10.5

    def test_generate_spiral_data_custom_radius(self):
        """The radius param scales the spiral."""
        from api.routes.training import _generate_spiral_data

        x, _ = _generate_spiral_data({"radius": 3.0})
        assert 2.8 <= float(x.abs().max()) <= 3.2

    def test_generate_spiral_data_seeded_noise_reproducible(self):
        """Seeded noise is reproducible; a different seed moves the points."""
        import torch

        from api.routes.training import _generate_spiral_data

        x1, _ = _generate_spiral_data({"noise": 0.25, "seed": 7})
        x2, _ = _generate_spiral_data({"noise": 0.25, "seed": 7})
        x3, _ = _generate_spiral_data({"noise": 0.25, "seed": 8})
        assert torch.equal(x1, x2)
        assert not torch.equal(x1, x3)

    def test_generate_spiral_data_noiseless_default_is_deterministic(self):
        """The default (noise 0) stays deterministic without a seed, as before."""
        import torch

        from api.routes.training import _generate_spiral_data

        x1, _ = _generate_spiral_data({})
        x2, _ = _generate_spiral_data({})
        assert torch.equal(x1, x2)

    def test_generate_spiral_data_rotations_param(self):
        """n_rotations stretches the parameter sweep (arc length grows with rotations)."""
        from api.routes.training import _generate_spiral_data

        x2, _ = _generate_spiral_data({"n_rotations": 2.0})
        x4, _ = _generate_spiral_data({"n_rotations": 4.0})
        # Same radius envelope either way; the 4-rotation spiral winds tighter,
        # so consecutive-point spacing near the rim differs. Just pin the
        # envelope and shape here — the winding count is visual.
        assert x2.shape == x4.shape
        assert 9.5 <= float(x4.abs().max()) <= 10.5
