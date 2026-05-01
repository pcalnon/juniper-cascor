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

    def test_update_max_iterations_updates_live_network(self, test_client_with_network):
        """PATCH /v1/training/params applies max_iterations to the live network."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"max_iterations": 17},
        )
        assert response.status_code == 200
        lifecycle = test_client_with_network.app.state.lifecycle
        assert lifecycle.network.max_iterations == 17

    def test_update_max_iterations_rejects_non_positive_value(self, test_client_with_network):
        """PATCH /v1/training/params enforces max_iterations >= 1."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"max_iterations": 0},
        )
        assert response.status_code == 422

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

    # CAS-002 (Phase 6E Sprint A-1): output_epochs is a per-output-training-phase
    # epoch budget, distinct from the global ``epochs_max``. The network already
    # exposes ``self.output_epochs``; this PR surfaces it on the param-update
    # surface (TrainingParamUpdateRequest) and the get/set lifecycle path.
    def test_update_output_epochs_updates_live_network(self, test_client_with_network):
        """PATCH /v1/training/params applies output_epochs to the live network."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"output_epochs": 250},
        )
        assert response.status_code == 200
        lifecycle = test_client_with_network.app.state.lifecycle
        assert lifecycle.network.output_epochs == 250

    def test_update_output_epochs_rejects_non_positive(self, test_client_with_network):
        """PATCH /v1/training/params enforces output_epochs >= 1 (matches the
        Pydantic validator on TrainingParamUpdateRequest)."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"output_epochs": 0},
        )
        assert response.status_code == 422

    def test_get_training_params_includes_output_epochs(self, test_client_with_network):
        """GET /v1/training/params returns output_epochs alongside the other fields,
        so a reconnecting client can reconcile UI state without falling back to
        stale defaults."""
        response = test_client_with_network.get("/v1/training/params")
        assert response.status_code == 200
        data = response.json()["data"]
        assert "output_epochs" in data
        # Live network has a non-zero default — verify it's a positive int.
        assert isinstance(data["output_epochs"], int)
        assert data["output_epochs"] >= 1

    # CAN-010 / ENH-006 (Phase 6E Sprint A-2): optimizer_type lives in a
    # nested config (network.config.optimizer_config.optimizer_type) rather
    # than directly on the network — runtime patching goes through a
    # special-cased setter in update_params (_write_optimizer_type).
    def test_update_optimizer_type_updates_nested_config(self, test_client_with_network):
        """PATCH /v1/training/params applies optimizer_type to the nested config."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"optimizer_type": "AdamW"},
        )
        assert response.status_code == 200
        lifecycle = test_client_with_network.app.state.lifecycle
        assert lifecycle.network.config.optimizer_config.optimizer_type == "AdamW"

    def test_update_optimizer_type_rejects_unsupported(self, test_client_with_network):
        """Pydantic Literal rejects optimizer names outside the supported set."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"optimizer_type": "NotARealOptimizer"},
        )
        assert response.status_code == 422

    def test_update_optimizer_type_accepts_full_registry(self, test_client_with_network):
        """Each Literal-allowed optimizer is accepted by the API."""
        for name in ("Adam", "AdamW", "SGD", "RMSprop", "NAdam", "RAdam", "Adamax", "Adagrad"):
            response = test_client_with_network.patch(
                "/v1/training/params",
                json={"optimizer_type": name},
            )
            assert response.status_code == 200, f"PATCH rejected supported optimizer {name!r}"

    def test_get_training_params_includes_optimizer_type(self, test_client_with_network):
        """GET surfaces optimizer_type so UI clients can reconcile after reconnect."""
        response = test_client_with_network.get("/v1/training/params")
        assert response.status_code == 200
        data = response.json()["data"]
        assert "optimizer_type" in data
        assert data["optimizer_type"] in {
            "Adam",
            "AdamW",
            "SGD",
            "RMSprop",
            "NAdam",
            "RAdam",
            "Adamax",
            "Adagrad",
            "Adadelta",
            "Adafactor",
            "ASGD",
            "LBFGS",
            "Rprop",
            "Muon",
        }

    def test_update_optimizer_type_round_trip(self, test_client_with_network):
        """PATCH then GET — the optimizer_type round-trips correctly."""
        test_client_with_network.patch("/v1/training/params", json={"optimizer_type": "SGD"})
        data = test_client_with_network.get("/v1/training/params").json()["data"]
        assert data["optimizer_type"] == "SGD"

    # CAN-011 (Phase 6E Sprint A-3): activation_function_name surface.
    # Lives on the network directly but requires re-running
    # ``_init_activation_function`` to refresh ``activation_fn`` /
    # ``activation_fn_no_diff`` from the registry — the runtime PATCH path
    # goes through ``_write_activation_function_name`` which sets
    # ``config.activation_function_name`` and re-runs the init routine.
    def test_update_activation_function_updates_live_network(self, test_client_with_network):
        """PATCH /v1/training/params applies activation_function_name and re-inits the activation."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"activation_function_name": "ReLU"},
        )
        assert response.status_code == 200
        lifecycle = test_client_with_network.app.state.lifecycle
        assert lifecycle.network.activation_function_name == "ReLU"
        # The re-init also refreshes ``activation_fn`` / ``activation_fn_no_diff``
        # from the registry, so the network is actually using the new function
        # rather than the surface attribute drifting from the live module.
        assert lifecycle.network.activation_fn_no_diff is not None
        assert lifecycle.network.config.activation_function_name == "ReLU"

    def test_update_activation_function_rejects_unsupported(self, test_client_with_network):
        """Pydantic Literal rejects activation names outside the supported registry."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"activation_function_name": "NotARealActivation"},
        )
        assert response.status_code == 422

    def test_update_activation_function_accepts_full_registry(self, test_client_with_network):
        """Each Literal-allowed activation function is accepted by the API."""
        for name in (
            "Identity",
            "Tanh",
            "Sigmoid",
            "ReLU",
            "LeakyReLU",
            "ELU",
            "SELU",
            "GELU",
            "Softmax",
            "Softplus",
            "Hardtanh",
            "Softshrink",
            "Tanhshrink",
            "tanh",
            "sigmoid",
            "relu",
        ):
            response = test_client_with_network.patch(
                "/v1/training/params",
                json={"activation_function_name": name},
            )
            assert response.status_code == 200, f"PATCH rejected supported activation {name!r}"

    def test_get_training_params_includes_activation_function(self, test_client_with_network):
        """GET surfaces activation_function_name so UI clients can reconcile after reconnect."""
        response = test_client_with_network.get("/v1/training/params")
        assert response.status_code == 200
        data = response.json()["data"]
        assert "activation_function_name" in data
        assert data["activation_function_name"] in {
            "Identity",
            "Tanh",
            "Sigmoid",
            "ReLU",
            "LeakyReLU",
            "ELU",
            "SELU",
            "GELU",
            "Softmax",
            "Softplus",
            "Hardtanh",
            "Softshrink",
            "Tanhshrink",
            "tanh",
            "sigmoid",
            "relu",
        }

    def test_update_activation_function_round_trip(self, test_client_with_network):
        """PATCH then GET — the activation_function_name round-trips correctly."""
        test_client_with_network.patch("/v1/training/params", json={"activation_function_name": "GELU"})
        data = test_client_with_network.get("/v1/training/params").json()["data"]
        assert data["activation_function_name"] == "GELU"

    # CAS-006 (Phase 6E Sprint A-4): auto_snap_best is a lifecycle-level
    # toggle (not a network attribute). Runtime PATCH lands on
    # ``self._auto_snap_best`` / ``self._auto_snap_min_epochs`` on the
    # lifecycle manager, and ``get_training_params`` surfaces the
    # current values. The functional behavior — the epoch_end callback
    # actually saving a snapshot when accuracy beats the best — is
    # exercised by ``TestAutoSnapBestCallback`` further down.
    def test_update_auto_snap_best_updates_lifecycle_flag(self, test_client_with_network):
        """PATCH /v1/training/params toggles auto_snap_best on the lifecycle."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"auto_snap_best": True},
        )
        assert response.status_code == 200
        lifecycle = test_client_with_network.app.state.lifecycle
        assert lifecycle._auto_snap_best is True

    def test_update_auto_snap_min_epochs_updates_lifecycle(self, test_client_with_network):
        """PATCH /v1/training/params updates auto_snap_min_epochs on the lifecycle."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"auto_snap_min_epochs": 200},
        )
        assert response.status_code == 200
        lifecycle = test_client_with_network.app.state.lifecycle
        assert lifecycle._auto_snap_min_epochs == 200

    def test_update_auto_snap_min_epochs_rejects_negative(self, test_client_with_network):
        """PATCH /v1/training/params enforces ``auto_snap_min_epochs >= 0``."""
        response = test_client_with_network.patch(
            "/v1/training/params",
            json={"auto_snap_min_epochs": -1},
        )
        assert response.status_code == 422

    def test_get_training_params_includes_auto_snap_fields(self, test_client_with_network):
        """GET surfaces both auto_snap_* fields so a reconnecting client can reconcile."""
        response = test_client_with_network.get("/v1/training/params")
        assert response.status_code == 200
        data = response.json()["data"]
        assert "auto_snap_best" in data
        assert "auto_snap_min_epochs" in data
        assert isinstance(data["auto_snap_best"], bool)
        assert isinstance(data["auto_snap_min_epochs"], int)
        assert data["auto_snap_min_epochs"] >= 0

    def test_update_auto_snap_round_trip(self, test_client_with_network):
        """PATCH then GET — both auto_snap_* fields round-trip correctly."""
        test_client_with_network.patch(
            "/v1/training/params",
            json={"auto_snap_best": True, "auto_snap_min_epochs": 75},
        )
        data = test_client_with_network.get("/v1/training/params").json()["data"]
        assert data["auto_snap_best"] is True
        assert data["auto_snap_min_epochs"] == 75

    def test_update_auto_snap_best_resets_metric_tracker_on_toggle_on(self, test_client_with_network):
        """Toggling auto_snap_best from False -> True resets the best-metric tracker.

        Otherwise a previous run's accuracy ceiling would suppress every
        snapshot in the new run — an obvious footgun if the user enables
        the feature mid-session after some earlier experiment.
        """
        lifecycle = test_client_with_network.app.state.lifecycle
        # Simulate a prior run leaving a stale tracker (without disturbing
        # any lifecycle thread state).
        with lifecycle._auto_snap_lock:
            lifecycle._auto_snap_best_metric = 0.95
        # Toggle on via the API.
        test_client_with_network.patch("/v1/training/params", json={"auto_snap_best": True})
        assert lifecycle._auto_snap_best is True
        assert lifecycle._auto_snap_best_metric is None


@pytest.mark.unit
class TestAutoSnapBestCallback:
    """Functional tests for the CAS-006 epoch_end callback.

    The route-level tests above cover the wiring (PATCH lands the flag,
    GET surfaces it, validation rejects bad values). These tests cover
    the actual behavior: when ``training_monitor.on_epoch_end`` fires,
    does the lifecycle save a snapshot at the right moments and skip
    the wrong ones?
    """

    def test_callback_skips_when_disabled(self):
        """Default state: feature off → no snapshot saved no matter what."""
        from unittest.mock import MagicMock

        from api.lifecycle.manager import TrainingLifecycleManager

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.save_snapshot = MagicMock()  # type: ignore[method-assign]
        # auto_snap_best defaults to False.
        mgr._maybe_auto_snap_callback(metrics={"validation_accuracy": 0.99}, epoch=1000, loss=0.01, accuracy=0.99)
        mgr.save_snapshot.assert_not_called()
        mgr.shutdown()

    def test_callback_skips_during_warmup(self):
        """Even with feature on, snapshots are suppressed until ``auto_snap_min_epochs`` is reached."""
        from unittest.mock import MagicMock

        from api.lifecycle.manager import TrainingLifecycleManager

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.save_snapshot = MagicMock()  # type: ignore[method-assign]
        with mgr._auto_snap_lock:
            mgr._auto_snap_best = True
            mgr._auto_snap_min_epochs = 100
        mgr._maybe_auto_snap_callback(metrics={"validation_accuracy": 0.5}, epoch=10, loss=0.5, accuracy=0.5)
        mgr._maybe_auto_snap_callback(metrics={"validation_accuracy": 0.9}, epoch=99, loss=0.1, accuracy=0.9)
        mgr.save_snapshot.assert_not_called()
        mgr.shutdown()

    def test_callback_saves_on_first_eligible_epoch(self):
        """First eligible epoch (post-warmup, accuracy > None) saves a snapshot."""
        from unittest.mock import MagicMock

        from api.lifecycle.manager import TrainingLifecycleManager

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.save_snapshot = MagicMock(return_value={"id": "snap_x"})  # type: ignore[method-assign]
        with mgr._auto_snap_lock:
            mgr._auto_snap_best = True
            mgr._auto_snap_min_epochs = 50
            mgr._auto_snap_best_metric = None
        mgr._maybe_auto_snap_callback(metrics={"validation_accuracy": 0.7}, epoch=50, loss=0.3, accuracy=0.6)
        assert mgr.save_snapshot.call_count == 1
        # Should track the validation_accuracy, not the (lower) training accuracy.
        assert mgr._auto_snap_best_metric == 0.7
        # Description carries the epoch + accuracy so the snapshot list is human-readable.
        kwargs = mgr.save_snapshot.call_args.kwargs
        assert "epoch=50" in kwargs.get("description", "")
        assert "0.700000" in kwargs.get("description", "")
        mgr.shutdown()

    def test_callback_only_snaps_on_strict_improvement(self):
        """Tied or worse accuracy after the first save → no additional snapshots."""
        from unittest.mock import MagicMock

        from api.lifecycle.manager import TrainingLifecycleManager

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.save_snapshot = MagicMock(return_value={"id": "snap_x"})  # type: ignore[method-assign]
        with mgr._auto_snap_lock:
            mgr._auto_snap_best = True
            mgr._auto_snap_min_epochs = 0
        mgr._maybe_auto_snap_callback(metrics={"validation_accuracy": 0.8}, epoch=10, loss=0.2, accuracy=0.8)
        mgr._maybe_auto_snap_callback(metrics={"validation_accuracy": 0.8}, epoch=11, loss=0.2, accuracy=0.8)  # tie
        mgr._maybe_auto_snap_callback(metrics={"validation_accuracy": 0.79}, epoch=12, loss=0.21, accuracy=0.79)  # worse
        mgr._maybe_auto_snap_callback(metrics={"validation_accuracy": 0.85}, epoch=13, loss=0.15, accuracy=0.85)  # better → snap
        assert mgr.save_snapshot.call_count == 2
        assert mgr._auto_snap_best_metric == 0.85
        mgr.shutdown()

    def test_callback_falls_back_to_training_accuracy_without_validation(self):
        """When ``validation_accuracy`` is None, the callback uses ``accuracy`` instead."""
        from unittest.mock import MagicMock

        from api.lifecycle.manager import TrainingLifecycleManager

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.save_snapshot = MagicMock(return_value={"id": "snap_x"})  # type: ignore[method-assign]
        with mgr._auto_snap_lock:
            mgr._auto_snap_best = True
            mgr._auto_snap_min_epochs = 0
        mgr._maybe_auto_snap_callback(metrics={"validation_accuracy": None}, epoch=5, loss=0.3, accuracy=0.7)
        assert mgr.save_snapshot.call_count == 1
        assert mgr._auto_snap_best_metric == 0.7
        mgr.shutdown()

    def test_callback_swallows_save_snapshot_exceptions(self):
        """A failing ``save_snapshot`` must not crash the training loop."""
        from unittest.mock import MagicMock

        from api.lifecycle.manager import TrainingLifecycleManager

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.save_snapshot = MagicMock(side_effect=RuntimeError("disk full"))  # type: ignore[method-assign]
        with mgr._auto_snap_lock:
            mgr._auto_snap_best = True
            mgr._auto_snap_min_epochs = 0
        # Should NOT raise — the exception is caught and logged.
        mgr._maybe_auto_snap_callback(metrics={"validation_accuracy": 0.9}, epoch=5, loss=0.1, accuracy=0.9)
        assert mgr.save_snapshot.call_count == 1
        mgr.shutdown()

    def test_callback_subscribed_to_epoch_end_event(self):
        """The callback is registered on ``training_monitor.epoch_end`` at __init__."""
        from unittest.mock import MagicMock

        from api.lifecycle.manager import TrainingLifecycleManager

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        # Stub the actual snapshot so we can observe firing.
        mgr.save_snapshot = MagicMock(return_value={"id": "snap_x"})  # type: ignore[method-assign]
        with mgr._auto_snap_lock:
            mgr._auto_snap_best = True
            mgr._auto_snap_min_epochs = 0
        # Trigger the public monitor event — the lifecycle's callback
        # registration must wire through.
        mgr.training_monitor.on_epoch_end(epoch=10, loss=0.2, accuracy=0.8, learning_rate=0.01, validation_accuracy=0.85)
        assert mgr.save_snapshot.call_count == 1
        mgr.shutdown()
