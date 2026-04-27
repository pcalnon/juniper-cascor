"""Tests for TrainingLifecycleManager."""

import time

import pytest
import torch

from api.lifecycle.manager import TrainingLifecycleManager


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
