"""Tests for lifecycle manager monitoring hooks."""

import queue
from unittest.mock import MagicMock, patch

import pytest
import torch

from api.lifecycle.manager import TrainingLifecycleManager
from api.lifecycle.state_machine import Command


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

    def test_install_hooks_wraps_validate_training(self):
        """Monitoring hooks wrap validate_training if available."""
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)

        if hasattr(manager.network, "validate_training"):
            assert "validate_training" in manager._original_methods

    def test_train_output_layer_not_hooked(self):
        """train_output_layer should NOT be hooked (fires before history update)."""
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)

        assert "train_output_layer" not in manager._original_methods

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
        assert len(manager.training_monitor.callbacks["candidate_progress"]) > 0

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

    def test_ws_callbacks_broadcast_on_candidate_progress(self):
        """Candidate progress callback broadcasts candidate_progress via WebSocket."""
        manager = TrainingLifecycleManager()
        ws_mgr = MagicMock()
        manager.set_ws_manager(ws_mgr)

        progress = {
            "candidate_id": 3,
            "candidate_uuid": "uuid-3",
            "epoch": 50,
            "total_epochs": 200,
            "correlation": 0.51,
        }
        manager.training_monitor.on_candidate_progress(progress)

        ws_mgr.broadcast_from_thread.assert_called()
        call_args = ws_mgr.broadcast_from_thread.call_args[0][0]
        assert call_args["type"] == "candidate_progress"
        assert call_args["data"] == progress

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

    def test_extract_metrics_no_new_data(self):
        """_extract_and_record_metrics does nothing when history hasn't grown."""
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)

        # No history data — should be a no-op
        manager._extract_and_record_metrics()
        assert manager._last_emitted_history_len == 0
        assert manager.training_monitor.get_current_state()["total_metrics"] == 0

    def test_extract_metrics_emits_new_entries(self):
        """_extract_and_record_metrics emits only new history entries."""
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)

        # Simulate history populated by the network
        manager.network.history["train_loss"].append(0.5)
        manager.network.history["train_accuracy"].append(0.6)

        manager._extract_and_record_metrics()
        assert manager._last_emitted_history_len == 1
        assert manager.training_monitor.get_current_state()["total_metrics"] == 1

        # Call again — should be no-op (no new data)
        manager._extract_and_record_metrics()
        assert manager._last_emitted_history_len == 1
        assert manager.training_monitor.get_current_state()["total_metrics"] == 1

        # Add another entry
        manager.network.history["train_loss"].append(0.3)
        manager.network.history["train_accuracy"].append(0.8)

        manager._extract_and_record_metrics()
        assert manager._last_emitted_history_len == 2
        assert manager.training_monitor.get_current_state()["total_metrics"] == 2

    def test_extract_metrics_missing_accuracy_emits_none(self):
        """Missing train_accuracy should emit metrics with accuracy=None."""
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)

        manager.network.history["train_loss"].append(0.42)
        # Intentionally omit train_accuracy entry for this epoch.

        manager._extract_and_record_metrics()

        metrics = manager.training_monitor.get_recent_metrics(1)
        assert len(metrics) == 1
        assert metrics[0]["loss"] == 0.42
        assert metrics[0]["accuracy"] is None

    def test_monitored_fit_injects_output_callback_and_updates_state(self):
        """Wrapped fit injects callback that records output-phase metrics."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

        original_fit = CascadeCorrelationNetwork.fit

        def fake_fit(self_network, x, y, x_val=None, y_val=None, **kwargs):
            assert hasattr(self_network, "_output_epoch_callback")
            self_network._output_epoch_callback(epoch=3, epochs=10, loss=0.123)
            return {"train_loss": [0.123]}

        CascadeCorrelationNetwork.fit = fake_fit
        try:
            manager = TrainingLifecycleManager()
            manager.create_network(input_size=2, output_size=2)
            x = torch.randn(8, 2)
            y = torch.randn(8, 2)

            manager.network.fit(x, y)

            state = manager.training_state.get_state()
            metrics = manager.training_monitor.get_recent_metrics(1)
            assert state["status"] == "Completed"
            assert state["phase"] == "Idle"
            assert state["current_epoch"] == 3
            assert state["phase_detail"] == "training_output"
            assert state["phase_started_at"] != ""
            assert metrics[0]["epoch"] == 3
            assert metrics[0]["accuracy"] is None
            assert metrics[0]["phase"] == "output"
        finally:
            CascadeCorrelationNetwork.fit = original_fit

    def test_monitored_grow_injects_iteration_callback_and_restores_output_phase(self):
        """Wrapped grow_network updates grow metadata and returns to output phase."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

        original_grow = CascadeCorrelationNetwork.grow_network

        def fake_grow(self_network, *args, **kwargs):
            assert hasattr(self_network, "_grow_iteration_callback")
            self_network._grow_iteration_callback(
                iteration=2,
                max_iterations=8,
                best_correlation=0.77,
                candidates_trained=4,
                candidates_total=12,
                phase_detail="adding_candidate",
                best_candidate_id=3,
                best_candidate_uuid="cand-uuid-3",
                second_candidate_id=7,
                second_candidate_correlation=0.65,
                all_correlations=[0.77, 0.65, 0.42],
            )
            self_network.hidden_units.append({})
            return {"ok": True}

        CascadeCorrelationNetwork.grow_network = fake_grow
        try:
            manager = TrainingLifecycleManager()
            manager.create_network(input_size=2, output_size=2)
            manager.state_machine.handle_command(Command.START)

            x = torch.randn(8, 2)
            y = torch.randn(8, 2)
            manager.network.grow_network(x, y, max_iterations=1)

            state = manager.training_state.get_state()
            sm_state = manager.state_machine.get_state_summary()
            assert state["phase"] == "Output"
            assert state["phase_detail"] == ""
            assert state["grow_iteration"] == 2
            assert state["grow_max"] == 8
            assert state["best_correlation"] == 0.77
            assert state["candidates_trained"] == 4
            assert state["candidates_total"] == 12
            assert state["phase_started_at"] != ""
            assert state["best_candidate_id"] == 3
            assert state["best_candidate_uuid"] == "cand-uuid-3"
            assert state["second_candidate_id"] == 7
            assert state["second_candidate_correlation"] == 0.65
            assert state["all_correlations"] == [0.77, 0.65, 0.42]
            assert manager.training_monitor.current_phase == "output"
            assert manager.training_monitor.get_current_state()["current_hidden_units"] == 1
            assert sm_state["phase"] == "OUTPUT"
        finally:
            CascadeCorrelationNetwork.grow_network = original_grow

    def test_monitored_grow_drains_progress_queue_and_resets_candidate_progress(self):
        """Wrapped grow_network drains worker progress queue into lifecycle state.

        The drain thread uses deferred queue discovery: the progress queue is
        created lazily inside grow_network (via _ensure_worker_pool), so the
        drain thread must poll for it. This test simulates that by having
        fake_grow create the queue during execution, just like real training.
        """
        from queue import Queue

        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

        original_grow = CascadeCorrelationNetwork.grow_network

        def fake_grow(self_network, *args, **kwargs):
            # Simulate _ensure_worker_pool creating the queue during grow_network
            self_network._persistent_progress_queue = Queue()
            self_network._persistent_progress_queue.put(
                {
                    "candidate_id": 1,
                    "candidate_uuid": "q-1",
                    "epoch": 75,
                    "total_epochs": 100,
                    "correlation": 0.66,
                }
            )
            # Give the drain thread a chance to discover the queue and consume progress
            import time

            time.sleep(0.15)
            return {"ok": True}

        CascadeCorrelationNetwork.grow_network = fake_grow
        try:
            manager = TrainingLifecycleManager()
            manager.create_network(input_size=2, output_size=2)
            manager.state_machine.handle_command(Command.START)

            # Queue starts as None — drain thread must discover it dynamically
            manager.network._persistent_progress_queue = None
            progress_events = []
            manager.training_monitor.register_callback("candidate_progress", lambda **kw: progress_events.append(kw["progress"]))

            x = torch.randn(8, 2)
            y = torch.randn(8, 2)
            manager.network.grow_network(x, y, max_iterations=1)

            state = manager.training_state.get_state()
            assert len(progress_events) >= 1
            assert progress_events[0]["epoch"] == 75
            assert progress_events[0]["total_epochs"] == 100
            assert state["phase"] == "Output"
            assert state["phase_detail"] == ""
            assert state["candidate_epoch"] == 0
            assert state["candidate_total_epochs"] == 0
        finally:
            CascadeCorrelationNetwork.grow_network = original_grow

    def test_drain_thread_starts_without_progress_queue(self):
        """Drain thread starts and exits cleanly when no queue is ever created."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

        original_grow = CascadeCorrelationNetwork.grow_network

        def fake_grow(self_network, *args, **kwargs):
            # Do NOT create a progress queue — simulate no worker pool
            import time

            time.sleep(0.15)
            return {"ok": True}

        CascadeCorrelationNetwork.grow_network = fake_grow
        try:
            manager = TrainingLifecycleManager()
            manager.create_network(input_size=2, output_size=2)
            manager.state_machine.handle_command(Command.START)

            # Ensure queue is None — drain thread should poll harmlessly
            manager.network._persistent_progress_queue = None

            x = torch.randn(8, 2)
            y = torch.randn(8, 2)
            # Should not raise or hang — drain thread exits via stop_event
            manager.network.grow_network(x, y, max_iterations=1)

            state = manager.training_state.get_state()
            assert state["phase"] == "Output"
            assert state["candidate_epoch"] == 0
        finally:
            CascadeCorrelationNetwork.grow_network = original_grow

    def test_high_water_mark_reset_on_reset(self):
        """reset() clears the high-water-mark."""
        manager = TrainingLifecycleManager()
        manager._last_emitted_history_len = 5
        manager.reset()
        assert manager._last_emitted_history_len == 0
