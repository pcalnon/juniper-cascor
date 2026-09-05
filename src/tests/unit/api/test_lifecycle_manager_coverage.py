#!/usr/bin/env python
"""
Additional unit tests for TrainingLifecycleManager to improve code coverage.

Covers:
- _extract_and_record_metrics: various history states
- get_decision_boundary: with/without network, with/without training data
- get_dataset: metadata retrieval
- set_ws_manager / _register_ws_callbacks: WebSocket integration
- start_training: already in progress, reuse stored data
- shutdown: with/without executor
"""

import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.lifecycle.manager import _SHUTDOWN_TRAINING_JOIN_TIMEOUT_SECONDS, TrainingInterrupted, TrainingLifecycleManager
from api.lifecycle.state_machine import Command

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
        """get_dataset reports one count PER PARTITION, and does not conflate two.

        Sized distinctly (100 / 20 / 35) on purpose: while ``test_samples`` was read off
        ``_val_x`` -- correct before cascor#620, when the two were the same rows -- this
        assertion passed with the validation count under the test name. Equal sizes would
        let it pass again.
        """
        mgr = TrainingLifecycleManager()
        mgr._train_x = torch.randn(100, 2)
        mgr._train_y = torch.randn(100, 2)
        mgr._val_x = torch.randn(20, 2)
        mgr._val_y = torch.randn(20, 2)
        mgr._test_x = torch.randn(35, 2)
        mgr._test_y = torch.randn(35, 2)

        result = mgr.get_dataset()
        assert result["loaded"] is True
        assert result["train_samples"] == 100
        assert result["val_samples"] == 20
        assert result["test_samples"] == 35
        assert result["input_features"] == 2
        assert result["output_features"] == 2

    def test_val_and_test_counts_are_not_the_same_field(self):
        """The regression itself: a val-only manager must NOT report those rows as test."""
        mgr = TrainingLifecycleManager()
        mgr._train_x = torch.randn(10, 2)
        mgr._train_y = torch.randn(10, 2)
        mgr._val_x = torch.randn(7, 2)
        mgr._val_y = torch.randn(7, 2)

        result = mgr.get_dataset()
        assert result["val_samples"] == 7
        assert result["test_samples"] == 0, "with no _test_x there is no reported partition to count"

    def test_dataset_data_carries_the_reported_partition_too(self):
        """``get_dataset_data`` returned val arrays and no test arrays.

        A visualiser drawing "the held-out data" from that payload was drawing the in-loop
        split, or nothing at all.
        """
        mgr = TrainingLifecycleManager()
        mgr._train_x = torch.randn(4, 2)
        mgr._train_y = torch.randn(4, 2)
        mgr._val_x = torch.randn(3, 2)
        mgr._val_y = torch.randn(3, 2)
        mgr._test_x = torch.randn(2, 2)
        mgr._test_y = torch.randn(2, 2)

        data = mgr.get_dataset_data()
        assert len(data["val_x"]) == 3
        assert len(data["test_x"]) == 2
        assert len(data["train_x"]) == 4

    def test_with_training_data_no_validation(self):
        """Both optional partitions absent -> both counts are 0, neither is invented."""
        mgr = TrainingLifecycleManager()
        mgr._train_x = torch.randn(50, 3)
        mgr._train_y = torch.randn(50, 2)

        result = mgr.get_dataset()
        assert result["loaded"] is True
        assert result["val_samples"] == 0
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

        with patch.object(mgr.monitor, "register_callback") as mock_register:
            mgr._register_ws_callbacks()

            assert mock_register.call_count == 5
            event_names = [call.args[0] for call in mock_register.call_args_list]
            assert "epoch_end" in event_names
            assert "cascade_add" in event_names
            assert "training_start" in event_names
            assert "training_end" in event_names
            assert "candidate_progress" in event_names


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

            mgr.start_training(X=x, y=y)
            started.wait(timeout=5)

            with pytest.raises(RuntimeError, match="already in progress"):
                mgr.start_training(X=x, y=y)
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
            mgr.start_training(X=x, y=y)
            if mgr._training_future is not None:
                mgr._training_future.result(timeout=10)

        mgr.shutdown()
        # After shutdown, executor should be cleaned up

    # ------------------------------------------------------------------
    # 2026-08-25 stop-during-training fix (juniper-ml
    # notes/JUNIPER_2026-08-25_JUNIPER-CASCOR_DEV-SHM-LEAK-CHARACTERISATION.md §6).
    # On a SIGTERM stop the lifespan's shutdown stanza is the last Python that runs
    # (uvicorn re-raises the signal with the default disposition once the lifespan
    # returns; atexit never fires), so shutdown() itself must unwind training and
    # release the run's resources before it returns.
    # ------------------------------------------------------------------

    def test_shutdown_joins_a_live_training_thread_before_returning(self):
        """shutdown() must not return while the training thread is still inside fit.

        Pre-fix it set ``_stop_event`` and returned at once; the process then died with
        the thread live (the /dev/shm ``juniper_train_*`` + 9 ``sem.mp-*`` ledger). The
        fake fit mirrors the engine's callback-driven interrupt: it observes the stop
        event and raises ``TrainingInterrupted``, which ``_run_training`` records as a
        clean stop.
        """
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        fit_entered = threading.Event()

        def fit_until_stopped(*_args, **_kwargs):
            fit_entered.set()
            while not mgr._stop_event.is_set():
                time.sleep(0.005)
            raise TrainingInterrupted("stop_requested")

        future = None
        try:
            with patch.object(mgr.network, "fit", side_effect=fit_until_stopped):
                mgr.start_training(X=torch.randn(10, 2), y=torch.randn(10, 2))
                assert fit_entered.wait(timeout=10), "fake fit never started"
                future = mgr._training_future
                assert future is not None and not future.done()
                started = time.monotonic()
                mgr.shutdown()
                elapsed = time.monotonic() - started
            assert future.done(), "shutdown() returned with the training future still running"
            assert elapsed < _SHUTDOWN_TRAINING_JOIN_TIMEOUT_SECONDS, f"join took {elapsed:.2f}s: the interrupt path did not fire"
            assert mgr._training_future is None
            assert mgr._executor is None
            assert mgr.training_state.get_state()["status"] == "Stopped"
        finally:
            mgr._stop_event.set()
            if future is not None:
                future.result(timeout=10)

    def test_shutdown_is_bounded_and_still_releases_resources_when_training_ignores_the_stop(self, caplog):
        """A training thread that never observes ``_stop_event`` (a stop landing
        mid-candidate-round) must not hold shutdown() hostage: the join is bounded, the
        outcome is logged, and the network's pool + shared-memory release hooks still run
        from the shutdown thread -- they are what the dead ``atexit`` path used to do.
        """
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        fit_entered = threading.Event()
        release_fit = threading.Event()

        def fit_ignoring_stop(*_args, **_kwargs):
            fit_entered.set()
            release_fit.wait(timeout=30)
            return {"train_loss": [0.5]}

        pool_release = MagicMock()
        shm_cleanup = MagicMock()
        future = None
        try:
            with (
                patch.object(mgr.network, "fit", side_effect=fit_ignoring_stop),
                patch.object(mgr.network, "_release_candidate_worker_pool", pool_release),
                patch.object(mgr.network, "_cleanup_shared_memory", shm_cleanup),
                patch("api.lifecycle.manager._SHUTDOWN_TRAINING_JOIN_TIMEOUT_SECONDS", 0.2),
                caplog.at_level("WARNING", logger="api.lifecycle.manager"),
            ):
                mgr.start_training(X=torch.randn(10, 2), y=torch.randn(10, 2))
                assert fit_entered.wait(timeout=10), "fake fit never started"
                future = mgr._training_future
                started = time.monotonic()
                mgr.shutdown()
                elapsed = time.monotonic() - started
                assert not future.done(), "the fake fit was supposed to outlive the join"
                assert elapsed < 2.0, f"shutdown() blocked {elapsed:.2f}s on a training thread that ignored the stop"
            pool_release.assert_called_once()
            shm_cleanup.assert_called_once()
            assert mgr._training_future is None
            assert mgr._executor is None
            assert any("did not unwind within" in rec.getMessage() for rec in caplog.records), "the timed-out join must be logged"
        finally:
            release_fit.set()
            if future is not None:
                future.result(timeout=10)

    def test_shutdown_releases_pool_and_shared_memory_when_idle(self):
        """No training in flight: both release hooks still run (a pool can outlive its
        run), and both are safe no-ops on a fresh network."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        with patch.object(mgr.network, "_release_candidate_worker_pool") as pool_release, patch.object(mgr.network, "_cleanup_shared_memory") as shm_cleanup:
            mgr.shutdown()
        pool_release.assert_called_once()
        shm_cleanup.assert_called_once()

    def test_shutdown_swallows_a_failing_release_hook_and_runs_the_next(self):
        """Cleanup must never mask shutdown: a wedged pool release is logged, and the
        shared-memory unlink still runs after it."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        with patch.object(mgr.network, "_release_candidate_worker_pool", side_effect=RuntimeError("pool wedged")), patch.object(mgr.network, "_cleanup_shared_memory") as shm_cleanup:
            mgr.shutdown()  # must not raise
        shm_cleanup.assert_called_once()


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
