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

    def test_shutdown(self):
        """Shutdown cleans up resources."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.shutdown()
        # Should not raise


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
