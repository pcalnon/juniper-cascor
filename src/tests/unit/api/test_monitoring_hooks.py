"""Tests for lifecycle manager monitoring hooks."""

from unittest.mock import MagicMock, patch

import pytest

from api.lifecycle.manager import TrainingLifecycleManager


@pytest.mark.unit
class TestMonitoringHooks:
    """Test monitoring hooks and WebSocket wiring."""

    def test_install_hooks_wraps_fit(self):
        """Monitoring hooks wrap network.fit method."""
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)

        # fit should be wrapped
        assert manager._monitoring_active is True
        assert "fit" in manager._original_methods

    def test_install_hooks_wraps_grow_network(self):
        """Monitoring hooks wrap grow_network if available."""
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)

        if hasattr(manager.network, "grow_network"):
            assert "grow_network" in manager._original_methods

    def test_install_hooks_wraps_train_output_layer(self):
        """Monitoring hooks wrap train_output_layer if available."""
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)

        if hasattr(manager.network, "train_output_layer"):
            assert "train_output_layer" in manager._original_methods

    def test_restore_original_methods(self):
        """Restoring original methods clears monitoring state."""
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)
        assert manager._monitoring_active is True

        manager._restore_original_methods()
        assert manager._monitoring_active is False
        assert len(manager._original_methods) == 0

    def test_set_ws_manager(self):
        """set_ws_manager stores reference and registers callbacks."""
        manager = TrainingLifecycleManager()
        ws_mgr = MagicMock()
        ws_mgr.broadcast_from_thread = MagicMock()

        manager.set_ws_manager(ws_mgr)

        assert manager._ws_manager is ws_mgr
        # Verify callbacks were registered
        assert len(manager.training_monitor.callbacks["epoch_end"]) > 0
        assert len(manager.training_monitor.callbacks["cascade_add"]) > 0
        assert len(manager.training_monitor.callbacks["training_start"]) > 0
        assert len(manager.training_monitor.callbacks["training_end"]) > 0

    def test_ws_callbacks_broadcast_on_epoch_end(self):
        """Epoch end callback broadcasts metrics via WebSocket."""
        manager = TrainingLifecycleManager()
        ws_mgr = MagicMock()
        manager.set_ws_manager(ws_mgr)

        # Trigger epoch_end callback
        manager.training_monitor.on_epoch_end(epoch=1, loss=0.5, accuracy=0.8, learning_rate=0.01)

        ws_mgr.broadcast_from_thread.assert_called()
        call_args = ws_mgr.broadcast_from_thread.call_args[0][0]
        assert call_args["type"] == "metrics"

    def test_ws_callbacks_broadcast_on_training_start(self):
        """Training start callback broadcasts state via WebSocket."""
        manager = TrainingLifecycleManager()
        ws_mgr = MagicMock()
        manager.set_ws_manager(ws_mgr)

        manager.training_monitor.on_training_start()

        ws_mgr.broadcast_from_thread.assert_called()
        call_args = ws_mgr.broadcast_from_thread.call_args[0][0]
        assert call_args["type"] == "state"

    def test_ws_callbacks_broadcast_on_training_end(self):
        """Training end callback broadcasts event via WebSocket."""
        manager = TrainingLifecycleManager()
        ws_mgr = MagicMock()
        manager.set_ws_manager(ws_mgr)

        manager.training_monitor.on_training_end()

        ws_mgr.broadcast_from_thread.assert_called()
        call_args = ws_mgr.broadcast_from_thread.call_args[0][0]
        assert call_args["type"] == "event"

    def test_ws_callbacks_broadcast_on_cascade_add(self):
        """Cascade add callback broadcasts cascade_add via WebSocket."""
        manager = TrainingLifecycleManager()
        ws_mgr = MagicMock()
        manager.set_ws_manager(ws_mgr)

        manager.training_monitor.on_cascade_add(hidden_unit_index=0, correlation=0.95)

        ws_mgr.broadcast_from_thread.assert_called()
        call_args = ws_mgr.broadcast_from_thread.call_args[0][0]
        assert call_args["type"] == "cascade_add"

    def test_get_dataset_no_data(self):
        """get_dataset returns loaded=False when no data."""
        manager = TrainingLifecycleManager()
        result = manager.get_dataset()
        assert result["loaded"] is False

    def test_has_training_data(self):
        """has_training_data returns correct boolean."""
        manager = TrainingLifecycleManager()
        assert manager.has_training_data() is False

    def test_get_decision_boundary_no_network(self):
        """get_decision_boundary returns None without network."""
        manager = TrainingLifecycleManager()
        assert manager.get_decision_boundary() is None

    def test_hooks_not_reinstalled(self):
        """Calling _install_monitoring_hooks twice doesn't double-wrap."""
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)

        original_fit = manager.network.fit
        manager._install_monitoring_hooks()  # Should be no-op (already active)
        assert manager.network.fit is original_fit
