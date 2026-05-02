"""Tests for TrainingLifecycleManager."""

import time

import pytest
import torch

from api.lifecycle.manager import TrainingLifecycleManager
from api.lifecycle.state_machine import Command


@pytest.mark.unit
class TestLifecycleManagerNetwork:
    """Test network management operations."""

    def test_initial_state(self):
        """Manager starts with no network."""
        mgr = TrainingLifecycleManager()
        assert not mgr.has_network()
        assert mgr.get_network_info() == {}

    def test_create_network(self):
        """Create network returns info dict."""
        mgr = TrainingLifecycleManager()
        info = mgr.create_network(input_size=2, output_size=2)
        assert mgr.has_network()
        assert info["input_size"] == 2
        assert info["output_size"] == 2
        assert info["hidden_units"] == 0
        assert "uuid" in info

    def test_delete_network(self):
        """Delete network removes it."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.delete_network()
        assert not mgr.has_network()

    def test_get_network_info(self):
        """Get network info returns expected fields."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=3, output_size=2, learning_rate=0.05)
        info = mgr.get_network_info()
        assert info["input_size"] == 3
        assert info["output_size"] == 2
        assert info["learning_rate"] == 0.05

    def test_create_network_updates_training_state(self):
        """Creating a network updates training state."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2, learning_rate=0.02)
        state = mgr.training_state.get_state()
        assert state["status"] == "Stopped"
        assert state["learning_rate"] == 0.02
        assert "CasCor" in state["network_name"]

    def test_create_network_keeps_max_epochs_and_max_iterations_separate(self):
        """Creating a network keeps epoch and growth limits independent."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(
            input_size=2,
            output_size=2,
            epochs_max=11,
            max_iterations=4,
        )
        state = mgr.training_state.get_state()
        assert state["max_epochs"] == 11
        assert state["max_iterations"] == 4

    def test_create_network_training_limit_defaults_aligned_with_canopy(self):
        """Phase 1 deferred item (revised 2026-04-10): training-limit defaults
        must match the canopy / requirements-spec values:

        - epochs_max          = 100,000,000,000  (1e11, raised from 1e6)
        - max_iterations      =       1,000,000  (1e6, raised from 1e3)
        - max_hidden_units    =          10,000  (1e4, raised from 1e3)

        These caps are intentionally large so the user, not the API model,
        chooses when to stop. See juniper-ml/notes/code-review/
        CANOPY_CASCOR_INTERFACE_ROADMAP_2026-04-08.md §3.5.
        """
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        state = mgr.training_state.get_state()
        assert state["max_epochs"] == 100000000000, f"epochs_max default should be 1e11; got {state['max_epochs']}"
        assert state["max_iterations"] == 1000000, f"max_iterations default should be 1e6; got {state['max_iterations']}"
        assert state["max_hidden_units"] == 10000, f"max_hidden_units default should be 1e4; got {state['max_hidden_units']}"

    def test_get_training_params_no_network(self):
        """Training params returns empty dict without network."""
        mgr = TrainingLifecycleManager()
        assert mgr.get_training_params() == {}

    def test_get_training_params(self):
        """Training params returns network params."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(
            input_size=2,
            output_size=2,
            learning_rate=0.01,
            max_hidden_units=10,
        )
        params = mgr.get_training_params()
        assert params["learning_rate"] == 0.01
        assert params["max_hidden_units"] == 10

    def test_get_training_params_returns_all_updatable_keys(self):
        """Every key in update_params' updatable_keys whitelist must be present in
        the get_training_params response so clients reconciling after reconnect
        observe live values instead of stale defaults (NEW-03)."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        params = mgr.get_training_params()
        expected_keys = {
            "learning_rate",
            "candidate_learning_rate",
            "correlation_threshold",
            "candidate_pool_size",
            "max_hidden_units",
            "epochs_max",
            "max_iterations",
            "patience",
            "convergence_threshold",
            "candidate_convergence_threshold",
            "candidate_patience",
            "candidate_epochs",
            "init_output_weights",
        }
        missing = expected_keys - params.keys()
        assert not missing, f"get_training_params missing updatable keys: {missing}"

    def test_shutdown(self):
        """Shutdown cleans up resources."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.shutdown()
        # Should not raise


@pytest.mark.unit
class TestLifecycleHeartbeat:
    """R1.2 / seed-03: liveness heartbeat counter and is_alive() accessor."""

    def test_heartbeat_initially_alive(self):
        """Fresh manager: heartbeat just bumped → is_alive() True."""
        mgr = TrainingLifecycleManager()
        try:
            assert mgr.is_alive() is True
            assert mgr._liveness_counter >= 1  # bumped at least by the daemon thread tick
        finally:
            mgr.stop_liveness_heartbeat()

    def test_bump_advances_counter_and_timestamp(self):
        """bump_liveness() increments counter and updates monotonic timestamp."""
        mgr = TrainingLifecycleManager()
        try:
            mgr.stop_liveness_heartbeat()  # avoid races with the daemon
            before = mgr._liveness_counter
            t_before = mgr._liveness_last_tick_at
            time.sleep(0.001)
            mgr.bump_liveness()
            assert mgr._liveness_counter == before + 1
            assert mgr._liveness_last_tick_at > t_before
        finally:
            mgr.stop_liveness_heartbeat()

    def test_is_alive_false_when_stale(self):
        """is_alive() returns False when last tick is older than the staleness window."""
        mgr = TrainingLifecycleManager()
        try:
            mgr.stop_liveness_heartbeat()
            with mgr._liveness_lock:
                mgr._liveness_last_tick_at = time.monotonic() - 100
            assert mgr.is_alive(stale_after_seconds=30) is False
        finally:
            mgr.stop_liveness_heartbeat()

    def test_monitor_event_callback_bumps_heartbeat(self):
        """TrainingMonitor event callbacks bump the heartbeat (training-thread liveness signal)."""
        mgr = TrainingLifecycleManager()
        try:
            mgr.stop_liveness_heartbeat()
            before = mgr._liveness_counter
            mgr.training_monitor.on_phase_change("output")
            assert mgr._liveness_counter > before
        finally:
            mgr.stop_liveness_heartbeat()

    def test_daemon_thread_bumps_periodically(self):
        """Daemon heartbeat thread bumps the counter at ~1s cadence."""
        mgr = TrainingLifecycleManager()
        try:
            before = mgr._liveness_counter
            time.sleep(1.5)
            assert mgr._liveness_counter > before
        finally:
            mgr.stop_liveness_heartbeat()


@pytest.mark.unit
class TestLifecycleManagerTrainingControl:
    """Test training start/stop/pause/resume/reset."""

    def test_start_training_without_network(self):
        """Start fails without network."""
        mgr = TrainingLifecycleManager()
        with pytest.raises(RuntimeError, match="No network created"):
            mgr.start_training(x=torch.randn(10, 2), y=torch.randn(10, 2))

    def test_start_training_without_data(self):
        """Start fails without training data."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        with pytest.raises(ValueError, match="Training data not provided"):
            mgr.start_training()

    def test_start_training(self):
        """Start training returns success dict."""
        from unittest.mock import patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2, epochs_max=2, candidate_pool_size=2, candidate_epochs=2, output_epochs=2, patience=1)
        x = torch.randn(20, 2)
        y = torch.zeros(20, 2)
        y[:10, 0] = 1
        y[10:, 1] = 1
        # Mock the network's fit() to avoid actual training overhead (~4s)
        with patch.object(mgr.network, "fit", return_value={"train_loss": [0.5]}):
            result = mgr.start_training(x=x, y=y)
            assert result["status"] == "training_started"
            assert "timestamp" in result
            # Wait for background training to actually complete before shutdown
            if mgr._training_future is not None:
                mgr._training_future.result(timeout=10)
        mgr.shutdown()

    def test_start_training_routes_network_kwargs_through_update_params(self):
        """``start_training(learning_rate=...)`` mutates the network in place
        without leaking the kwarg to ``fit()`` (closes CASCOR_FIT_KWARGS_LATENT_BUG).

        Pre-fix behavior: the POST /v1/training/start handler forwarded the
        full TrainingParams body to ``network.fit(**kwargs)`` whose narrow
        signature only accepts ``{max_epochs, epochs, max_iterations,
        early_stopping}``. Passing ``learning_rate`` (or any other valid
        TrainingParams field) caused ``TypeError: fit() got an unexpected
        keyword argument`` on the background training thread, while the
        HTTP response had already returned 200 — silent failure.

        Post-fix: lifecycle splits kwargs into fit-shaped vs.
        network-attribute, applies network-attribute kwargs via the
        ``_apply_params_unlocked`` helper (same whitelist + atomic-rollback
        as ``update_params``), and forwards only fit-shaped kwargs to
        ``fit()``.
        """
        from unittest.mock import patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2, learning_rate=0.01, epochs_max=2, candidate_pool_size=2, candidate_epochs=2, output_epochs=2, patience=1)
        assert mgr.network.learning_rate == 0.01

        x = torch.randn(20, 2)
        y = torch.zeros(20, 2)
        y[:10, 0] = 1
        y[10:, 1] = 1

        with patch.object(mgr.network, "fit", return_value={"train_loss": [0.5]}) as mock_fit:
            mgr.start_training(x=x, y=y, learning_rate=0.005, max_iterations=3)
            if mgr._training_future is not None:
                mgr._training_future.result(timeout=10)

        # Network attribute was applied in place.
        assert mgr.network.learning_rate == 0.005
        # fit() received only its narrow-signature kwargs — never learning_rate.
        assert mock_fit.call_count == 1
        fit_kwargs = mock_fit.call_args.kwargs
        assert "learning_rate" not in fit_kwargs
        assert fit_kwargs.get("max_iterations") == 3
        mgr.shutdown()

    def test_start_training_does_not_typeerror_on_full_training_params_body(self):
        """Regression: passing every TrainingParams field that previously
        broke fit() must now succeed (covers learning_rate, output_epochs,
        optimizer_type, patience — the four examples called out in the
        latent-bug doc as pre-fix TypeError triggers)."""
        from unittest.mock import patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2, epochs_max=2, candidate_pool_size=2, candidate_epochs=2, output_epochs=2, patience=1)
        x = torch.randn(20, 2)
        y = torch.zeros(20, 2)
        y[:10, 0] = 1
        y[10:, 1] = 1

        # Body covers every previously-broken category: numeric tunable,
        # int budget, Literal-validated string, and a fit-shaped kwarg.
        with patch.object(mgr.network, "fit", return_value={"train_loss": [0.5]}) as mock_fit:
            mgr.start_training(
                x=x,
                y=y,
                learning_rate=0.003,
                output_epochs=7,
                optimizer_type="AdamW",
                patience=15,
                max_iterations=2,
            )
            if mgr._training_future is not None:
                mgr._training_future.result(timeout=10)

        # All four network-attribute kwargs landed on the live network.
        assert mgr.network.learning_rate == 0.003
        assert mgr.network.output_epochs == 7
        assert mgr.network.config.optimizer_config.optimizer_type == "AdamW"
        assert mgr.network.patience == 15
        # fit() got only the fit-shaped kwarg.
        fit_kwargs = mock_fit.call_args.kwargs
        for non_fit in ("learning_rate", "output_epochs", "optimizer_type", "patience"):
            assert non_fit not in fit_kwargs, f"{non_fit} leaked into fit() kwargs"
        assert fit_kwargs.get("max_iterations") == 2
        mgr.shutdown()

    def test_stop_training(self):
        """Stop training returns success dict."""
        mgr = TrainingLifecycleManager()
        result = mgr.stop_training()
        assert result["status"] == "stop_requested"

    def test_pause_training_not_active(self):
        """Pause fails when training not active."""
        mgr = TrainingLifecycleManager()
        with pytest.raises(RuntimeError, match="Training is not active"):
            mgr.pause_training()

    def test_resume_training_not_paused(self):
        """Resume fails when not paused."""
        mgr = TrainingLifecycleManager()
        with pytest.raises(RuntimeError, match="Training is not paused"):
            mgr.resume_training()

    def test_reset(self):
        """Reset returns success and clears state."""
        mgr = TrainingLifecycleManager()
        result = mgr.reset()
        assert result["status"] == "reset"
        state = mgr.training_state.get_state()
        assert state["current_epoch"] == 0
        assert state["current_step"] == 0


@pytest.mark.unit
class TestLifecycleManagerStatus:
    """Test status and metrics retrieval."""

    def test_get_status(self):
        """Get status returns all expected sections."""
        mgr = TrainingLifecycleManager()
        status = mgr.get_status()
        assert "state_machine" in status
        assert "monitor" in status
        assert "training_state" in status
        assert "network_loaded" in status
        assert "training_active" in status
        assert status["network_loaded"] is False
        assert status["training_active"] is False

    def test_get_status_with_network(self):
        """Get status reflects network presence."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        status = mgr.get_status()
        assert status["network_loaded"] is True

    def test_get_metrics_no_network(self):
        """Get metrics returns empty dict without network."""
        mgr = TrainingLifecycleManager()
        assert mgr.get_metrics() == {}

    def test_get_metrics_history_empty(self):
        """Metrics history is empty initially."""
        mgr = TrainingLifecycleManager()
        assert mgr.get_metrics_history() == []

    def test_get_metrics_history_with_count(self):
        """Metrics history respects count param."""
        mgr = TrainingLifecycleManager()
        # Directly add to monitor
        mgr.training_monitor.on_epoch_end(
            epoch=1,
            loss=0.5,
            accuracy=0.75,
            learning_rate=0.01,
            hidden_units=0,
        )
        mgr.training_monitor.on_epoch_end(
            epoch=2,
            loss=0.4,
            accuracy=0.80,
            learning_rate=0.01,
            hidden_units=0,
        )
        history = mgr.get_metrics_history(count=1)
        assert len(history) == 1

    def test_get_topology_no_network(self):
        """Topology returns None without network."""
        mgr = TrainingLifecycleManager()
        assert mgr.get_topology() is None

    def test_get_topology_with_network(self):
        """Topology returns dict with network."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        topology = mgr.get_topology()
        assert topology is not None
        assert topology["input_size"] == 2
        assert topology["output_size"] == 2
        assert "output_weights" in topology
        assert "hidden_units" in topology

    def test_get_statistics_no_network(self):
        """Statistics returns empty dict without network."""
        mgr = TrainingLifecycleManager()
        assert mgr.get_statistics() == {}

    def test_get_statistics_with_network(self):
        """Statistics returns dict with network."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        stats = mgr.get_statistics()
        assert "total_hidden_units" in stats
        assert "output_weight_mean" in stats
        assert "output_weight_std" in stats


@pytest.mark.unit
class TestLifecycleManagerMonitoringHooks:
    """Test monitoring hook installation."""

    def test_hooks_installed_on_create(self):
        """Monitoring hooks are installed when network is created."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        assert mgr._monitoring_active is True
        assert "fit" in mgr._original_methods

    def test_hooks_restored_on_delete(self):
        """Monitoring hooks are restored when network is deleted."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.delete_network()
        assert mgr._monitoring_active is False
        assert len(mgr._original_methods) == 0


@pytest.mark.unit
class TestLifecycleWorkerCoordinator:
    """Test worker coordinator injection into lifecycle manager and network."""

    def test_set_coordinator_before_network(self):
        """Coordinator set before network creation is injected into new network."""
        from unittest.mock import MagicMock

        mgr = TrainingLifecycleManager()
        mock_coord = MagicMock()
        mgr.set_worker_coordinator(mock_coord)
        assert mgr._worker_coordinator is mock_coord

        mgr.create_network(input_size=2, output_size=2)
        assert mgr.network._worker_coordinator is mock_coord
        assert mgr.network._remote_workers_enabled is True

    def test_set_coordinator_after_network(self):
        """Coordinator set after network creation is injected into existing network."""
        from unittest.mock import MagicMock

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        assert mgr.network._worker_coordinator is None

        mock_coord = MagicMock()
        mgr.set_worker_coordinator(mock_coord)
        assert mgr.network._worker_coordinator is mock_coord

    def test_no_coordinator_leaves_network_default(self):
        """Without coordinator, network uses local-only dispatch."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        assert mgr.network._worker_coordinator is None
        assert mgr.network._remote_workers_enabled is False


@pytest.mark.unit
class TestUpdateParamsAtomicity:
    """GAP-WS-28: update_params applies all keys or none — never half.

    The race itself is closed by ``_training_lock`` (one writer at a time);
    these tests cover the all-or-nothing semantics for the case where a
    property setter rejects a value mid-loop. No setter currently raises,
    so we drive the path with a fake network that raises on a chosen key.
    """

    class _FakeNetwork:
        """Minimal stand-in with the attributes update_params() touches.

        ``failing_key`` (when set) makes setattr for that key raise ValueError,
        modeling a future property setter that validates input.
        """

        def __init__(self, failing_key: str | None = None):
            self.learning_rate = 0.01
            self.candidate_learning_rate = 0.02
            self.correlation_threshold = 0.1
            self.candidate_pool_size = 8
            self.max_hidden_units = 50
            self.epochs_max = 100
            self.max_iterations = 200
            self.patience = 5
            self.convergence_threshold = 1e-4
            self.candidate_convergence_threshold = 1e-4
            self.candidate_patience = 3
            self.candidate_epochs = 10
            self.init_output_weights = "zero"
            self._failing_key = failing_key

        def __setattr__(self, name, value):
            failing = self.__dict__.get("_failing_key")
            if failing is not None and name == failing:
                raise ValueError(f"setter rejected: {name}={value!r}")
            object.__setattr__(self, name, value)

    def _mgr_with_network(self, network):
        mgr = TrainingLifecycleManager()
        mgr.network = network
        return mgr

    def test_happy_path_applies_all_keys(self):
        """All updatable keys are applied when no setter raises."""
        net = self._FakeNetwork()
        mgr = self._mgr_with_network(net)
        mgr.update_params({"learning_rate": 0.005, "correlation_threshold": 0.2, "patience": 10})
        assert net.learning_rate == pytest.approx(0.005)
        assert net.correlation_threshold == pytest.approx(0.2)
        assert net.patience == 10

    def test_unrecognized_keys_silently_skipped(self):
        """Unknown keys don't raise and don't change state."""
        net = self._FakeNetwork()
        mgr = self._mgr_with_network(net)
        before = net.learning_rate
        mgr.update_params({"this_is_not_a_real_key": 99, "learning_rate": before + 0.1})
        assert net.learning_rate == pytest.approx(before + 0.1)

    def test_setter_failure_rolls_back_earlier_keys(self):
        """If patience setter raises, learning_rate and correlation_threshold
        (applied before patience in iteration order) must be reverted."""
        net = self._FakeNetwork(failing_key="patience")
        mgr = self._mgr_with_network(net)

        original_lr = net.learning_rate
        original_threshold = net.correlation_threshold
        original_patience = net.patience

        with pytest.raises(ValueError, match="setter rejected: patience"):
            # Dict order matters: in Python 3.7+ dicts preserve insertion order,
            # so learning_rate and correlation_threshold are applied before patience.
            mgr.update_params(
                {
                    "learning_rate": 0.999,
                    "correlation_threshold": 0.999,
                    "patience": 999,
                }
            )

        # GAP-WS-28: all three must be at their original values.
        assert net.learning_rate == pytest.approx(original_lr), "learning_rate not rolled back"
        assert net.correlation_threshold == pytest.approx(original_threshold), "correlation_threshold not rolled back"
        assert net.patience == original_patience, "patience never advanced (correct)"

    def test_setter_failure_on_first_key_no_state_change(self):
        """If the first key's setter raises, nothing was applied — nothing to revert."""
        net = self._FakeNetwork(failing_key="learning_rate")
        mgr = self._mgr_with_network(net)
        original_lr = net.learning_rate
        original_threshold = net.correlation_threshold

        with pytest.raises(ValueError):
            mgr.update_params({"learning_rate": 0.999, "correlation_threshold": 0.999})

        assert net.learning_rate == pytest.approx(original_lr)
        assert net.correlation_threshold == pytest.approx(original_threshold)

    def test_no_network_raises_value_error(self):
        """Pre-existing contract preserved: no network → ValueError."""
        mgr = TrainingLifecycleManager()
        with pytest.raises(ValueError, match="No network exists"):
            mgr.update_params({"learning_rate": 0.005})


@pytest.mark.unit
class TestRestoreForRetrain:
    """CAN-015a (Phase 6E Sprint B B-1): tests for the new
    ``restore_for_retrain`` lifecycle path.

    The method shares the load step with ``load_snapshot`` but adds the
    full reset scope from ``PHASE_6E_SPRINT_B_DESIGN.md`` §9. These tests
    pin each piece of the reset contract individually plus the
    return-False-on-missing-snapshot behavior.
    """

    def test_returns_false_when_snapshot_missing(self, tmp_path):
        """Snapshot ID with no matching .h5 file → False, no state changes."""
        from unittest.mock import patch

        mgr = TrainingLifecycleManager()
        # Empty snapshots dir — no matching file.
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path):
            assert mgr.restore_for_retrain("nonexistent-snapshot") is False
        # No network was created — verify the call didn't accidentally make one.
        assert mgr.network is None
        mgr.shutdown()

    def test_returns_false_when_deserializer_fails(self, tmp_path):
        """Deserializer returning None → False, original network preserved."""
        from unittest.mock import patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        original_network = mgr.network
        fake_file = tmp_path / "broken.h5"
        fake_file.write_bytes(b"not actually an hdf5 file")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=None):
            assert mgr.restore_for_retrain("broken") is False
        # Original network is still installed — failed retrain doesn't
        # leave the lifecycle in a half-replaced state.
        assert mgr.network is original_network
        mgr.shutdown()

    def test_resets_network_history_arrays(self, tmp_path):
        """After successful restore_for_retrain, network.history arrays are empty."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        loaded = MagicMock()
        loaded.history = {
            "train_loss": [0.5, 0.4, 0.3],
            "value_loss": [0.6, 0.5, 0.4],
            "train_accuracy": [0.7, 0.8, 0.85],
            "value_accuracy": [0.65, 0.75, 0.8],
        }
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            assert mgr.restore_for_retrain("snap") is True

        for key in ("train_loss", "value_loss", "train_accuracy", "value_accuracy"):
            assert loaded.history[key] == [], f"history[{key!r}] not cleared by retrain"
        mgr.shutdown()

    def test_resets_training_state_counters(self, tmp_path):
        """current_epoch / current_step go to 0; status is Stopped, phase is Idle."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.training_state.update_state(current_epoch=42, current_step=999, status="Started", phase="Output")
        loaded = MagicMock()
        loaded.history = {}
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            mgr.restore_for_retrain("snap")

        state = mgr.training_state.get_state()
        assert state["current_epoch"] == 0
        assert state["current_step"] == 0
        assert state["status"] == "Stopped"
        assert state["phase"] == "Idle"
        mgr.shutdown()

    def test_resets_auto_snap_best_metric(self, tmp_path):
        """The auto-snap-best ratchet is reset so the next run starts
        from a fresh accuracy ceiling."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        with mgr._auto_snap_lock:
            mgr._auto_snap_best_metric = 0.95
        loaded = MagicMock()
        loaded.history = {}
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            mgr.restore_for_retrain("snap")

        assert mgr._auto_snap_best_metric is None
        mgr.shutdown()

    def test_resets_last_emitted_history_len(self, tmp_path):
        """The history-emission cursor is reset so the next training run
        re-emits from epoch 0 rather than skipping to the loaded snapshot's
        end."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr._last_emitted_history_len = 100
        loaded = MagicMock()
        loaded.history = {}
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            mgr.restore_for_retrain("snap")

        assert mgr._last_emitted_history_len == 0
        mgr.shutdown()

    def test_clears_training_monitor_metrics(self, tmp_path):
        """The training_monitor's metrics buffer is cleared."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        loaded = MagicMock()
        loaded.history = {}
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"), patch.object(mgr.training_monitor, "clear_metrics") as mock_clear:
            mgr.restore_for_retrain("snap")
            mock_clear.assert_called_once()
        mgr.shutdown()

    def test_tolerates_missing_history_attr(self, tmp_path):
        """A loaded network without a ``history`` attribute doesn't crash
        the retrain — best-effort consistency mirroring A-5's
        legacy-snapshot tolerance."""
        from unittest.mock import patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        class NetworkWithoutHistory:
            input_size = 2
            output_size = 2

        loaded = NetworkWithoutHistory()
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            assert mgr.restore_for_retrain("snap") is True
        mgr.shutdown()

    def test_tolerates_non_dict_history(self, tmp_path):
        """A loaded network whose ``history`` is not a dict (e.g. None) is
        also tolerated — same defensive reasoning."""
        from unittest.mock import patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        class NetworkWithNoneHistory:
            input_size = 2
            output_size = 2
            history = None

        loaded = NetworkWithNoneHistory()
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            assert mgr.restore_for_retrain("snap") is True
        mgr.shutdown()

    def test_load_snapshot_preserves_history_and_counters(self, tmp_path):
        """Regression: ``load_snapshot`` (Restore semantics) preserves
        history arrays and ``training_state`` counters. B-4 added the FSM
        transition to INVESTIGATING but the data-side preservation
        contract is unchanged from B-1."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.training_state.update_state(current_epoch=42, current_step=999)
        loaded = MagicMock()
        loaded.history = {"train_loss": [0.1, 0.2], "value_loss": [], "train_accuracy": [], "value_accuracy": []}
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            assert mgr.load_snapshot("snap") is True

        # History on the loaded network is preserved — Restore is a load,
        # not a reset.
        assert loaded.history["train_loss"] == [0.1, 0.2]
        # Counters on training_state are NOT reset by Restore (only Retrain
        # resets these). status / phase ARE updated by B-4 to keep
        # training_state in sync with the FSM transition to Investigating,
        # but the epoch/step counters reflect the loaded snapshot.
        state = mgr.training_state.get_state()
        assert state["current_epoch"] == 42
        assert state["current_step"] == 999
        mgr.shutdown()


@pytest.mark.unit
class TestResumeFromSnapshot:
    """CAN-015b (Phase 6E Sprint B B-2): tests for the new
    ``resume_from_snapshot`` lifecycle method.

    Resume preserves history (in contrast to Retrain which resets it),
    transitions the FSM to ``RESUME_READY``, and records
    ``_resume_point_epoch`` so canopy can render a visual boundary.
    The auto-snap-best ratchet is also preserved so a re-snap only
    fires when the resumed training beats the prior run's best.
    """

    def test_returns_false_when_snapshot_missing(self, tmp_path):
        """Snapshot ID with no matching .h5 file → False, no FSM transition."""
        from unittest.mock import patch

        mgr = TrainingLifecycleManager()
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path):
            assert mgr.resume_from_snapshot("nonexistent") is False
        assert mgr.state_machine.is_stopped()
        assert not mgr.state_machine.is_resume_ready()
        mgr.shutdown()

    def test_returns_false_when_training_active(self, tmp_path):
        """Resume rejected while training is Started; FSM unchanged."""
        from unittest.mock import patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.state_machine.handle_command(Command.START)
        assert mgr.state_machine.is_started()
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path):
            assert mgr.resume_from_snapshot("anything") is False
        assert mgr.state_machine.is_started()
        assert not mgr.state_machine.is_resume_ready()
        mgr.shutdown()

    def test_preserves_network_history_arrays(self, tmp_path):
        """After successful resume_from_snapshot, network.history arrays are intact."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        loaded = MagicMock()
        loaded.history = {
            "train_loss": [0.5, 0.4, 0.3],
            "value_loss": [0.6, 0.5, 0.4],
            "train_accuracy": [0.7, 0.8, 0.85],
            "value_accuracy": [0.65, 0.75, 0.8],
        }
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            assert mgr.resume_from_snapshot("snap") is True

        assert loaded.history["train_loss"] == [0.5, 0.4, 0.3]
        assert loaded.history["value_loss"] == [0.6, 0.5, 0.4]
        assert loaded.history["train_accuracy"] == [0.7, 0.8, 0.85]
        assert loaded.history["value_accuracy"] == [0.65, 0.75, 0.8]
        mgr.shutdown()

    def test_preserves_auto_snap_best_metric(self, tmp_path):
        """Resume preserves the auto-snap-best ratchet (in contrast to Retrain)."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        with mgr._auto_snap_lock:
            mgr._auto_snap_best_metric = 0.95
        loaded = MagicMock()
        loaded.history = {"train_loss": [], "value_loss": [], "train_accuracy": [], "value_accuracy": []}
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            mgr.resume_from_snapshot("snap")

        assert mgr._auto_snap_best_metric == 0.95
        mgr.shutdown()

    def test_transitions_fsm_to_resume_ready(self, tmp_path):
        """FSM goes to RESUME_READY after a successful resume."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        loaded = MagicMock()
        loaded.history = {"train_loss": [0.1], "value_loss": [], "train_accuracy": [], "value_accuracy": []}
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            mgr.resume_from_snapshot("snap")

        assert mgr.state_machine.is_resume_ready()
        mgr.shutdown()

    def test_records_resume_point_epoch(self, tmp_path):
        """``_resume_point_epoch`` reflects the longest history array's length."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        loaded = MagicMock()
        # TODO: migrate hard-coded values into constants
        loaded.history = {
            "train_loss": [0.1, 0.2, 0.3, 0.4, 0.5],  # 5
<<<<<<< HEAD
            "value_loss": [0.2, 0.3],  # 2
            "train_accuracy": [0.6, 0.7, 0.8],  # 3
            "value_accuracy": [],  # 0
=======
            "value_loss": [0.2, 0.3],                 # 2
            "train_accuracy": [0.6, 0.7, 0.8],        # 3
            "value_accuracy": [],                     # 0
>>>>>>> 29f1aec16a2903ab4fd6fbb4c93e18e10e76909b
        }
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            mgr.resume_from_snapshot("snap")

        # Longest array (train_loss) = 5.
        assert mgr._resume_point_epoch == 5
        mgr.shutdown()

    def test_resume_point_zero_for_empty_history(self, tmp_path):
        """An empty history dict produces resume_point_epoch=0."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        loaded = MagicMock()
        loaded.history = {}
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            assert mgr.resume_from_snapshot("snap") is True

        assert mgr._resume_point_epoch == 0
        mgr.shutdown()

    def test_resume_point_zero_when_history_missing(self, tmp_path):
        """A network without a ``history`` attribute produces resume_point_epoch=0."""
        from unittest.mock import patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        class NetworkWithoutHistory:
            input_size = 2
            output_size = 2

        loaded = NetworkWithoutHistory()
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            assert mgr.resume_from_snapshot("snap") is True

        assert mgr._resume_point_epoch == 0
        mgr.shutdown()

    def test_retrain_after_resume_clears_marker(self, tmp_path):
        """A Retrain after a Resume clears the resume marker — Retrain is a clean slate."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        loaded = MagicMock()
        loaded.history = {"train_loss": [0.1, 0.2, 0.3], "value_loss": [], "train_accuracy": [], "value_accuracy": []}
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            mgr.resume_from_snapshot("snap")
            assert mgr._resume_point_epoch == 3
            mgr.restore_for_retrain("snap")
            assert mgr._resume_point_epoch is None
        mgr.shutdown()

    def test_start_training_after_resume_preserves_auto_snap_baseline(self, tmp_path):
        """When start_training fires from RESUME_READY, the auto-snap baseline
        is preserved (the existing reset-on-start applies only to non-resume
        starts). The resume marker is consumed (cleared) once start_training
        runs."""
        from unittest.mock import MagicMock, patch

        import torch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2, candidate_pool_size=2, candidate_epochs=2, output_epochs=2, patience=1)
        loaded = MagicMock()
        loaded.history = {"train_loss": [0.5, 0.4], "value_loss": [], "train_accuracy": [], "value_accuracy": []}
        # Carry the network attributes start_training reads.
        for attr in ("input_size", "output_size"):
            setattr(loaded, attr, getattr(mgr.network, attr, 2))

        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            mgr.resume_from_snapshot("snap")

        # Set ratchet to a known value AFTER resume to isolate this test.
        with mgr._auto_snap_lock:
            mgr._auto_snap_best_metric = 0.85
        assert mgr.state_machine.is_resume_ready()
        assert mgr._resume_point_epoch == 2

        x = torch.randn(20, 2)
        y = torch.zeros(20, 2)
        y[:10, 0] = 1
        y[10:, 1] = 1
        with patch.object(mgr.network, "fit", return_value={"train_loss": [0.3]}):
            mgr.start_training(x=x, y=y)
            if mgr._training_future is not None:
                mgr._training_future.result(timeout=10)

        # Ratchet preserved across the resume-to-start transition.
        assert mgr._auto_snap_best_metric == 0.85
        # Resume marker consumed.
        assert mgr._resume_point_epoch is None
        mgr.shutdown()

    def test_start_training_from_stopped_resets_auto_snap_baseline(self, tmp_path):
        """Regression: a normal start_training (FSM = Stopped, not RESUME_READY)
        still resets the auto-snap ratchet."""
        from unittest.mock import patch

        import torch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2, candidate_pool_size=2, candidate_epochs=2, output_epochs=2, patience=1)
        with mgr._auto_snap_lock:
            mgr._auto_snap_best_metric = 0.85
        assert mgr.state_machine.is_stopped()

        x = torch.randn(20, 2)
        y = torch.zeros(20, 2)
        y[:10, 0] = 1
        y[10:, 1] = 1
        with patch.object(mgr.network, "fit", return_value={"train_loss": [0.3]}):
            mgr.start_training(x=x, y=y)
            if mgr._training_future is not None:
                mgr._training_future.result(timeout=10)

        assert mgr._auto_snap_best_metric is None
        mgr.shutdown()


@pytest.mark.unit
class TestLoadSnapshotInvestigating:
    """CAN-015d (Phase 6E Sprint B B-4): tests for the new ``load_snapshot``
    Investigating contract.

    Restore now transitions the FSM to ``INVESTIGATING`` so the user
    can edit / re-snapshot but cannot start training directly.
    Pre-flight check rejects when training is currently active.
    """

    def test_returns_false_when_training_active(self, tmp_path):
        """load_snapshot rejected while training is Started."""
        from unittest.mock import patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.state_machine.handle_command(Command.START)
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path):
            assert mgr.load_snapshot("anything") is False
        # FSM still in Started — Restore didn't sneak through.
        assert mgr.state_machine.is_started()
        mgr.shutdown()

    def test_transitions_fsm_to_investigating(self, tmp_path):
        """Successful load_snapshot transitions FSM to INVESTIGATING."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        loaded = MagicMock()
        loaded.history = {"train_loss": [0.1, 0.2], "value_loss": [], "train_accuracy": [], "value_accuracy": []}
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            assert mgr.load_snapshot("snap") is True

        assert mgr.state_machine.is_investigating()
        mgr.shutdown()

    def test_clears_resume_marker(self, tmp_path):
        """A Restore over a previously-resumed snapshot clears the
        resume marker — Restore is the inspection-only entry point."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        # Simulate a prior Resume having set the marker.
        mgr._resume_point_epoch = 5
        loaded = MagicMock()
        loaded.history = {"train_loss": [], "value_loss": [], "train_accuracy": [], "value_accuracy": []}
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            mgr.load_snapshot("snap")

        assert mgr._resume_point_epoch is None
        mgr.shutdown()

    def test_start_training_rejected_when_investigating(self):
        """``start_training`` raises RuntimeError when FSM is INVESTIGATING.

        The user must invoke /retrain or /resume to transition out of
        Investigating before training can begin. Failing fast at the API
        boundary is much clearer than letting the future submit and the
        FSM transition fail silently inside monitored_fit.
        """
        import torch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        # Force FSM to Investigating without actually loading a snapshot.
        mgr.state_machine.mark_investigating()
        x = torch.randn(20, 2)
        y = torch.zeros(20, 2)
        with pytest.raises(RuntimeError, match="Investigating"):
            mgr.start_training(x=x, y=y)
        # FSM unchanged — failed start didn't accidentally transition.
        assert mgr.state_machine.is_investigating()
        mgr.shutdown()

    def test_retrain_after_restore_clears_investigating(self, tmp_path):
        """Retrain after Restore transitions out of Investigating to
        Stopped — Retrain explicitly moves to a training-ready state."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        loaded = MagicMock()
        loaded.history = {"train_loss": [0.1], "value_loss": [], "train_accuracy": [], "value_accuracy": []}
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            mgr.load_snapshot("snap")
            assert mgr.state_machine.is_investigating()
            mgr.restore_for_retrain("snap")
            # After Retrain the FSM is Stopped (Retrain calls Command.RESET).
            assert mgr.state_machine.is_stopped()
            assert not mgr.state_machine.is_investigating()
        mgr.shutdown()

    def test_resume_after_restore_clears_investigating(self, tmp_path):
        """Resume after Restore transitions Investigating -> ResumeReady."""
        from unittest.mock import MagicMock, patch

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        loaded = MagicMock()
        loaded.history = {"train_loss": [0.1, 0.2, 0.3], "value_loss": [], "train_accuracy": [], "value_accuracy": []}
        fake_file = tmp_path / "snap.h5"
        fake_file.write_bytes(b"")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.load_network", return_value=loaded), patch.object(mgr, "_restore_original_methods"), patch.object(mgr, "_install_monitoring_hooks"):
            mgr.load_snapshot("snap")
            assert mgr.state_machine.is_investigating()
            mgr.resume_from_snapshot("snap")
            assert mgr.state_machine.is_resume_ready()
            assert not mgr.state_machine.is_investigating()
        mgr.shutdown()
