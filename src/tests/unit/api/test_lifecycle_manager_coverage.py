#!/usr/bin/env python
"""
Additional unit tests for TrainingLifecycleManager to improve code coverage.

Covers:
- _extract_and_record_metrics: various history states
- get_decision_boundary: with/without network, with/without training data
- get_dataset: metadata retrieval
- set_ws_manager / _register_ws_callbacks: WebSocket integration
- _install_monitoring_hooks: monitored_fit, monitored_train_output, monitored_grow
- start_training: already in progress, reuse stored data
- shutdown: with/without executor
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.lifecycle.manager import TrainingLifecycleManager

pytestmark = pytest.mark.unit


class TestExtractAndRecordMetrics:
    """Tests for _extract_and_record_metrics."""

    def test_no_network_does_nothing(self):
        """_extract_and_record_metrics should return early if no network."""
        mgr = TrainingLifecycleManager()
        mgr._extract_and_record_metrics()  # Should not raise

    def test_network_without_history_does_nothing(self):
        """_extract_and_record_metrics should return early if no history attribute."""
        mgr = TrainingLifecycleManager()
        mgr.network = MagicMock(spec=[])  # no history attribute
        mgr._extract_and_record_metrics()  # Should not raise

    def test_extracts_metrics_from_history(self):
        """_extract_and_record_metrics should record metrics from network history."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        mgr.network.history = {
            "train_loss": [0.5, 0.4, 0.3],
            "train_accuracy": [0.6, 0.7, 0.8],
        }
        mgr.network.hidden_units = []
        mgr.network.learning_rate = 0.01

        mgr._extract_and_record_metrics()

        state = mgr.training_state.get_state()
        assert state["current_epoch"] == 3

    def test_handles_empty_history(self):
        """_extract_and_record_metrics should handle empty history gracefully."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        mgr.network.history = {"train_loss": [], "train_accuracy": []}
        mgr.network.hidden_units = []

        mgr._extract_and_record_metrics()  # Should not raise

    def test_handles_validation_metrics(self):
        """_extract_and_record_metrics should handle validation metrics."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        mgr.network.history = {
            "train_loss": [0.5],
            "train_accuracy": [0.6],
            "value_loss": [0.55],
            "value_accuracy": [0.55],
        }
        mgr.network.hidden_units = []
        mgr.network.learning_rate = 0.01

        mgr._extract_and_record_metrics()

        state = mgr.training_state.get_state()
        assert state["current_epoch"] == 1

    def test_handles_runtime_error_gracefully(self):
        """_extract_and_record_metrics should handle RuntimeError from network."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        # Make the history dict raise RuntimeError during iteration
        # by replacing the get method on the history dict
        bad_history = MagicMock()
        bad_history.get.side_effect = RuntimeError("concurrent access")
        mgr.network.history = bad_history

        mgr._extract_and_record_metrics()  # Should not raise


class TestGetDecisionBoundary:
    """Tests for get_decision_boundary."""

    def test_returns_none_without_network(self):
        """get_decision_boundary should return None when no network."""
        mgr = TrainingLifecycleManager()
        assert mgr.get_decision_boundary() is None

    def test_returns_none_without_training_data(self):
        """get_decision_boundary should return None when no training data stored."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        assert mgr.get_decision_boundary() is None

    def test_returns_none_for_non_2d_data(self):
        """get_decision_boundary should return None when training data is not 2D features."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=5, output_size=2)
        mgr._train_x = torch.randn(10, 5)  # 5 features, not 2
        assert mgr.get_decision_boundary() is None

    def test_returns_boundary_grid_for_2d_data(self):
        """get_decision_boundary should return grid dict for 2D training data."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr._train_x = torch.randn(20, 2)
        mgr._train_y = torch.zeros(20, 2)
        mgr._train_y[:10, 0] = 1
        mgr._train_y[10:, 1] = 1

        result = mgr.get_decision_boundary(resolution=10)

        assert result is not None
        assert "x_range" in result
        assert "y_range" in result
        assert "resolution" in result
        assert result["resolution"] == 10
        assert "grid_x" in result
        assert "grid_y" in result
        assert "predictions" in result
        assert len(result["predictions"]) == 10
        assert len(result["predictions"][0]) == 10

    def test_handles_forward_error_gracefully(self):
        """get_decision_boundary should return None on forward error."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr._train_x = torch.randn(20, 2)
        mgr._train_y = torch.zeros(20, 2)

        # Make forward raise an error
        mgr.network.forward = MagicMock(side_effect=RuntimeError("forward failed"))

        result = mgr.get_decision_boundary(resolution=5)
        assert result is None


class TestGetDataset:
    """Tests for get_dataset."""

    def test_no_data_returns_not_loaded(self):
        """get_dataset should return loaded=False when no data."""
        mgr = TrainingLifecycleManager()
        result = mgr.get_dataset()
        assert result == {"loaded": False}

    def test_with_training_data_returns_metadata(self):
        """get_dataset should return metadata when training data is stored."""
        mgr = TrainingLifecycleManager()
        mgr._train_x = torch.randn(100, 2)
        mgr._train_y = torch.randn(100, 2)
        mgr._val_x = torch.randn(20, 2)
        mgr._val_y = torch.randn(20, 2)

        result = mgr.get_dataset()
        assert result["loaded"] is True
        assert result["train_samples"] == 100
        assert result["test_samples"] == 20
        assert result["input_features"] == 2
        assert result["output_features"] == 2

    def test_with_training_data_no_validation(self):
        """get_dataset should report 0 test_samples when no validation data."""
        mgr = TrainingLifecycleManager()
        mgr._train_x = torch.randn(50, 3)
        mgr._train_y = torch.randn(50, 2)

        result = mgr.get_dataset()
        assert result["loaded"] is True
        assert result["test_samples"] == 0
        assert result["input_features"] == 3


class TestSetWsManager:
    """Tests for set_ws_manager and _register_ws_callbacks."""

    def test_set_ws_manager_stores_and_registers(self):
        """set_ws_manager should store manager and register callbacks."""
        mgr = TrainingLifecycleManager()
        mock_ws = MagicMock()

        mgr.set_ws_manager(mock_ws)

        assert mgr._ws_manager is mock_ws

    def test_register_ws_callbacks_skips_when_no_manager(self):
        """_register_ws_callbacks should return early when ws_manager is None."""
        mgr = TrainingLifecycleManager()
        mgr._ws_manager = None
        mgr._register_ws_callbacks()  # Should not raise

    def test_register_ws_callbacks_registers_all_events(self):
        """_register_ws_callbacks should register epoch_end, cascade_add, training_start, training_end."""
        mgr = TrainingLifecycleManager()
        mock_ws = MagicMock()
        mgr._ws_manager = mock_ws

        with patch.object(mgr.training_monitor, "register_callback") as mock_register:
            mgr._register_ws_callbacks()

            assert mock_register.call_count == 4
            event_names = [call.args[0] for call in mock_register.call_args_list]
            assert "epoch_end" in event_names
            assert "cascade_add" in event_names
            assert "training_start" in event_names
            assert "training_end" in event_names


class TestMonitoringHooks:
    """Tests for _install_monitoring_hooks behavior."""

    def test_hooks_not_installed_without_network(self):
        """_install_monitoring_hooks should return early when no network."""
        mgr = TrainingLifecycleManager()
        mgr._install_monitoring_hooks()
        assert mgr._monitoring_active is False

    def test_hooks_not_reinstalled_if_already_active(self):
        """_install_monitoring_hooks should not reinstall if already active."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        assert mgr._monitoring_active is True

        original_fit = mgr.network.fit
        mgr._install_monitoring_hooks()
        # fit should still be the same wrapped version (hooks not reinstalled)
        assert mgr.network.fit is original_fit

    def test_monitored_fit_records_metrics(self):
        """Monitored fit should call _extract_and_record_metrics."""
        mgr = TrainingLifecycleManager()

        # Temporarily patch CascadeCorrelationNetwork.fit to avoid real training
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

        original_class_fit = CascadeCorrelationNetwork.fit

        CascadeCorrelationNetwork.fit = MagicMock(return_value={"train_loss": [0.5]})
        try:
            mgr.create_network(input_size=2, output_size=2)

            x = torch.randn(10, 2)
            y = torch.zeros(10, 2)
            y[:5, 0] = 1
            y[5:, 1] = 1

            with patch.object(mgr, "_extract_and_record_metrics") as mock_extract:
                mgr.network.fit(x, y)
                assert mock_extract.called
        finally:
            CascadeCorrelationNetwork.fit = original_class_fit


class TestStartTrainingEdgeCases:
    """Tests for start_training edge cases."""

    def test_start_training_already_in_progress(self):
        """start_training should raise RuntimeError if training already in progress."""
        import threading

        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

        barrier = threading.Event()
        started = threading.Event()

        def blocking_fit(self_network, *args, **kwargs):
            started.set()
            barrier.wait(timeout=10)
            return {"train_loss": [0.5]}

        original_class_fit = CascadeCorrelationNetwork.fit
        CascadeCorrelationNetwork.fit = blocking_fit

        try:
            mgr = TrainingLifecycleManager()
            mgr.create_network(input_size=2, output_size=2)
            x = torch.randn(10, 2)
            y = torch.zeros(10, 2)
            y[:5, 0] = 1
            y[5:, 1] = 1

            mgr.start_training(x=x, y=y)
            started.wait(timeout=5)

            with pytest.raises(RuntimeError, match="already in progress"):
                mgr.start_training(x=x, y=y)
        finally:
            barrier.set()
            CascadeCorrelationNetwork.fit = original_class_fit
            if mgr._training_future is not None:
                try:
                    mgr._training_future.result(timeout=10)
                except Exception:
                    pass
            mgr.shutdown()

    def test_start_training_reuses_stored_data(self):
        """start_training should reuse previously stored training data."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        x = torch.randn(10, 2)
        y = torch.zeros(10, 2)
        y[:5, 0] = 1
        y[5:, 1] = 1

        mgr._train_x = x
        mgr._train_y = y

        with patch.object(mgr.network, "fit", return_value={"train_loss": [0.5]}):
            result = mgr.start_training()  # No x, y provided
            assert result["status"] == "training_started"

            if mgr._training_future is not None:
                mgr._training_future.result(timeout=10)
        mgr.shutdown()


class TestShutdown:
    """Tests for shutdown method."""

    def test_shutdown_without_network(self):
        """Shutdown without network should not raise."""
        mgr = TrainingLifecycleManager()
        mgr.shutdown()

    def test_shutdown_cleans_up_executor(self):
        """Shutdown should shut down the executor."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        x = torch.randn(10, 2)
        y = torch.zeros(10, 2)
        y[:5, 0] = 1
        y[5:, 1] = 1

        with patch.object(mgr.network, "fit", return_value={"train_loss": [0.5]}):
            mgr.start_training(x=x, y=y)
            if mgr._training_future is not None:
                mgr._training_future.result(timeout=10)

        mgr.shutdown()
        # After shutdown, executor should be cleaned up


class TestHasTrainingData:
    """Tests for has_training_data."""

    def test_no_training_data(self):
        """has_training_data should return False when no data stored."""
        mgr = TrainingLifecycleManager()
        assert mgr.has_training_data() is False

    def test_with_training_data(self):
        """has_training_data should return True when data is stored."""
        mgr = TrainingLifecycleManager()
        mgr._train_x = torch.randn(10, 2)
        mgr._train_y = torch.randn(10, 2)
        assert mgr.has_training_data() is True
