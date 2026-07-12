"""Tests for lifecycle-manager monitoring (WS-6 PR-B3.3 on_event model).

Monitoring is driven by ``CascorModel.fit``'s ``on_event`` sink (``_handle_event``) and the
``_run_training`` session lifecycle — *not* by monkey-patching ``network.fit`` /
``grow_network``. The former install/restore/``monitored_*`` tests are therefore replaced by
``_handle_event`` (per-event projection) and ``_run_training`` (session lifecycle + candidate-
progress drain + terminal FSM) tests below. The WebSocket wiring and ``_extract_and_record_metrics``
tests are unchanged.
"""

import time
from queue import Queue
from unittest.mock import MagicMock

import pytest
import torch
from juniper_model_core.events import TrainingEvent

from api.lifecycle.manager import TrainingInterrupted, TrainingLifecycleManager
from api.lifecycle.state_machine import Command


def _event(event_type, payload):
    """A model-core TrainingEvent (seq is irrelevant to _handle_event dispatch)."""
    return TrainingEvent(event_type, payload, 0)


@pytest.mark.unit
class TestWebSocketWiring:
    """set_ws_manager stores the manager and registers the monitor broadcast callbacks."""

    def test_set_ws_manager(self):
        """set_ws_manager stores reference and registers callbacks."""
        manager = TrainingLifecycleManager()
        ws_mgr = MagicMock()
        ws_mgr.broadcast_from_thread = MagicMock()

        manager.set_ws_manager(ws_mgr)

        assert manager._ws_manager is ws_mgr
        # Verify callbacks were registered
        assert len(manager.monitor.callbacks["epoch_end"]) > 0
        assert len(manager.monitor.callbacks["cascade_add"]) > 0
        assert len(manager.monitor.callbacks["training_start"]) > 0
        assert len(manager.monitor.callbacks["training_end"]) > 0
        assert len(manager.monitor.callbacks["candidate_progress"]) > 0

    def test_ws_callbacks_broadcast_on_epoch_end(self):
        """Epoch end callback broadcasts metrics via WebSocket."""
        manager = TrainingLifecycleManager()
        ws_mgr = MagicMock()
        manager.set_ws_manager(ws_mgr)

        # Trigger epoch_end callback
        manager.monitor.on_epoch_end(epoch=1, loss=0.5, accuracy=0.8, learning_rate=0.01)

        ws_mgr.broadcast_from_thread.assert_called()
        call_args = ws_mgr.broadcast_from_thread.call_args[0][0]
        assert call_args["type"] == "metrics"

    def test_ws_callbacks_broadcast_on_training_start(self):
        """Training start callback broadcasts state via WebSocket."""
        manager = TrainingLifecycleManager()
        ws_mgr = MagicMock()
        manager.set_ws_manager(ws_mgr)

        manager.monitor.on_training_start()

        ws_mgr.broadcast_from_thread.assert_called()
        call_args = ws_mgr.broadcast_from_thread.call_args[0][0]
        assert call_args["type"] == "state"

    def test_ws_callbacks_broadcast_on_training_end(self):
        """Training end callback broadcasts event via WebSocket."""
        manager = TrainingLifecycleManager()
        ws_mgr = MagicMock()
        manager.set_ws_manager(ws_mgr)

        manager.monitor.on_training_end()

        ws_mgr.broadcast_from_thread.assert_called()
        call_args = ws_mgr.broadcast_from_thread.call_args[0][0]
        assert call_args["type"] == "event"

    def test_ws_callbacks_broadcast_on_cascade_add(self):
        """Cascade add callback broadcasts cascade_add via WebSocket."""
        manager = TrainingLifecycleManager()
        ws_mgr = MagicMock()
        manager.set_ws_manager(ws_mgr)

        manager.monitor.on_cascade_add(hidden_unit_index=0, correlation=0.95)

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
        manager.monitor.on_candidate_progress(progress)

        ws_mgr.broadcast_from_thread.assert_called()
        call_args = ws_mgr.broadcast_from_thread.call_args[0][0]
        assert call_args["type"] == "candidate_progress"
        assert call_args["data"] == progress


@pytest.mark.unit
class TestExtractAndRecordMetrics:
    """_extract_and_record_metrics high-water-mark behaviour + misc read helpers."""

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

    def test_extract_metrics_no_new_data(self):
        """_extract_and_record_metrics does nothing when history hasn't grown."""
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)

        # No history data — should be a no-op
        manager._extract_and_record_metrics()
        assert manager._last_emitted_history_len == 0
        assert manager.monitor.get_current_state()["total_metrics"] == 0

    def test_extract_metrics_emits_new_entries(self):
        """_extract_and_record_metrics emits only new history entries."""
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)

        # Simulate history populated by the network
        manager.network.history["train_loss"].append(0.5)
        manager.network.history["train_accuracy"].append(0.6)

        manager._extract_and_record_metrics()
        assert manager._last_emitted_history_len == 1
        assert manager.monitor.get_current_state()["total_metrics"] == 1

        # Call again — should be no-op (no new data)
        manager._extract_and_record_metrics()
        assert manager._last_emitted_history_len == 1
        assert manager.monitor.get_current_state()["total_metrics"] == 1

        # Add another entry
        manager.network.history["train_loss"].append(0.3)
        manager.network.history["train_accuracy"].append(0.8)

        manager._extract_and_record_metrics()
        assert manager._last_emitted_history_len == 2
        assert manager.monitor.get_current_state()["total_metrics"] == 2

    def test_extract_metrics_missing_accuracy_emits_none(self):
        """Missing train_accuracy should emit metrics with accuracy=None."""
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)

        manager.network.history["train_loss"].append(0.42)
        # Intentionally omit train_accuracy entry for this epoch.

        manager._extract_and_record_metrics()

        metrics = manager.monitor.get_recent_metrics(1)
        assert len(metrics) == 1
        assert metrics[0]["loss"] == 0.42
        assert metrics[0]["accuracy"] is None

    def test_high_water_mark_reset_on_reset(self):
        """reset() clears the high-water-mark."""
        manager = TrainingLifecycleManager()
        manager._last_emitted_history_len = 5
        manager.reset()
        assert manager._last_emitted_history_len == 0


@pytest.mark.unit
class TestHandleEvent:
    """The on_event sink: ``_handle_event`` projects CascorModel.fit's coarse events
    (training_start / epoch_end / phase_change / unit_added / training_end) onto the
    TrainingMonitor + TrainingState the read routes serialize."""

    def test_epoch_end_updates_state_and_records_metrics(self):
        """epoch_end records the within-pass output-epoch on the monitor and updates the live state.

        C2b (I-1c): the inner output-epoch no longer writes ``current_epoch`` (that
        field is the training-step counter, owned by the history drain) — it lands
        in the dedicated ``output_epoch`` / ``output_total_epochs`` pair, and the
        buffered row is tagged ``kind="output_epoch"``."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr._step_timer_prev = None

        mgr._handle_event(_event("epoch_end", {"epoch": 3, "metrics": {"loss": 0.123}, "epochs": 10}))

        state = mgr.training_state.get_state()
        assert state["output_epoch"] == 3
        assert state["output_total_epochs"] == 10
        assert state["current_epoch"] == 0, "inner output-epoch must not clobber the training-step counter"
        assert state["phase_detail"] == "training_output"
        metrics = mgr.monitor.get_recent_metrics(1)
        assert metrics[0]["epoch"] == 3
        assert metrics[0]["kind"] == "output_epoch"
        assert metrics[0]["loss"] == 0.123
        assert metrics[0]["accuracy"] is None

    def test_epoch_end_raises_on_stop_request(self):
        """A stop request observed at an epoch boundary aborts via TrainingInterrupted."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr._stop_event.set()

        with pytest.raises(TrainingInterrupted):
            mgr._handle_event(_event("epoch_end", {"epoch": 1, "metrics": {"loss": 0.5}}))

    def test_phase_change_enters_candidate_and_updates_grow_state(self):
        """phase_change (a grow iteration) enters the Candidate phase once and projects the
        per-iteration candidate-pool detail from payload['detail']."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.state_machine.handle_command(Command.START)
        mgr._grow_phase_entered = False

        detail = {
            "grow_iteration": 2,
            "max_iterations": 8,
            "best_correlation": 0.77,
            "candidates_trained": 4,
            "candidates_total": 12,
            "phase_detail": "adding_candidate",
            "best_candidate_id": 3,
            "best_candidate_uuid": "cand-uuid-3",
            "second_candidate_id": 7,
            "second_candidate_correlation": 0.65,
            "all_correlations": [0.77, 0.65, 0.42],
        }
        mgr._handle_event(_event("phase_change", {"phase": "candidate", "detail": detail}))

        state = mgr.training_state.get_state()
        assert mgr._grow_phase_entered is True
        assert state["phase"] == "Candidate"
        assert state["phase_started_at"] != ""
        assert state["grow_iteration"] == 2
        assert state["grow_max"] == 8
        assert state["best_correlation"] == 0.77
        assert state["candidates_trained"] == 4
        assert state["candidates_total"] == 12
        assert state["best_candidate_id"] == 3
        assert state["best_candidate_uuid"] == "cand-uuid-3"
        assert state["second_candidate_id"] == 7
        assert state["second_candidate_correlation"] == 0.65
        assert state["all_correlations"] == [0.77, 0.65, 0.42]

    def test_phase_change_raises_on_stop_request(self):
        """A stop request observed at a grow-iteration boundary aborts."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr._stop_event.set()

        with pytest.raises(TrainingInterrupted):
            mgr._handle_event(_event("phase_change", {"phase": "candidate", "detail": {}}))

    def test_unit_added_emits_cascade_add_via_cursor(self):
        """unit_added emits one cascade_add per newly-installed hidden unit, advancing the
        cursor (so a retrain only emits the units grown this run)."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr._cascade_emitted_count = 0

        unit = MagicMock()
        unit.best_correlation = 0.91
        mgr.network.hidden_units.append(unit)

        events = []
        mgr.monitor.register_callback("cascade_add", lambda *a, **kw: events.append((a, kw)))

        mgr._handle_event(_event("unit_added", {"n_units": 1, "unit_id": "h0", "score": 0.91}))

        assert mgr._cascade_emitted_count == 1
        assert len(events) == 1

    def test_unit_added_noop_when_cursor_past_units(self):
        """A spurious unit_added beyond the installed units is a harmless no-op."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr._cascade_emitted_count = 0  # but no hidden units installed

        events = []
        mgr.monitor.register_callback("cascade_add", lambda *a, **kw: events.append((a, kw)))

        mgr._handle_event(_event("unit_added", {"n_units": 1, "unit_id": "h0", "score": 0.5}))

        assert mgr._cascade_emitted_count == 0
        assert events == []

    def test_training_end_broadcasts_full_topology_and_returns_to_output(self):
        """After growth, training_end broadcasts the full serialized topology (not a count-only
        stub — BUG-CC-01/02 regression) and returns the live state to the Output phase.

        Regression: a count-only ``hidden_units: int`` stub made canopy's _transform_topology
        render 0 hidden units. The fix broadcasts ``manager.get_topology()`` directly.
        """
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        ws_mgr = MagicMock()
        mgr.set_ws_manager(ws_mgr)
        mgr._grow_phase_entered = True

        # A structurally-valid hidden unit so get_topology() serializes a list-shaped payload.
        mgr.network.hidden_units.append(
            {
                "weights": torch.tensor([0.10, 0.20, 0.30]),
                "bias": torch.tensor(0.05),
                "activation_fn": torch.sigmoid,
            }
        )

        mgr._handle_event(_event("training_end", {"metrics": {}}))

        state = mgr.training_state.get_state()
        assert state["phase"] == "Output"
        assert state["phase_detail"] == ""
        assert state["candidate_epoch"] == 0
        assert state["candidate_total_epochs"] == 0

        topology_calls = [c[0][0] for c in ws_mgr.broadcast_from_thread.call_args_list if isinstance(c[0][0], dict) and c[0][0].get("type") == "topology"]
        assert len(topology_calls) >= 1, "expected a full-topology broadcast at training_end after growth"
        payload = topology_calls[-1]["data"]
        hidden_units = payload.get("hidden_units")
        assert isinstance(hidden_units, list), f"hidden_units must be a list, was {type(hidden_units).__name__}"
        assert len(hidden_units) >= 1
        unit = hidden_units[0]
        for required in ("id", "weights", "bias", "activation"):
            assert required in unit, f"hidden unit missing {required!r}: {unit}"
        assert "output_weights" in payload
        assert "output_bias" in payload

    def test_training_end_without_growth_is_quiet(self):
        """training_end with no growth this run does not broadcast a topology or flip phase."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        ws_mgr = MagicMock()
        mgr.set_ws_manager(ws_mgr)
        mgr._grow_phase_entered = False

        mgr._handle_event(_event("training_end", {"metrics": {}}))

        topology_calls = [c[0][0] for c in ws_mgr.broadcast_from_thread.call_args_list if isinstance(c[0][0], dict) and c[0][0].get("type") == "topology"]
        assert topology_calls == []


@pytest.mark.unit
class TestRunTraining:
    """_run_training: session start/terminal FSM, the OBS active-session gauge pair, and the
    candidate-progress drain thread — all driven through CascorModel.fit's on_event sink."""

    @staticmethod
    def _toy():
        return torch.randn(8, 2), torch.randn(8, 2)

    def test_success_path_completes(self):
        """A clean fit transitions Started -> Completed/Idle and dispatches the events."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        def fake_fit(x, y, *, X_val=None, y_val=None, on_event=None, **kw):
            if on_event is not None:
                on_event(TrainingEvent("training_start", {"n_samples": int(x.shape[0])}, 0))
                on_event(TrainingEvent("epoch_end", {"epoch": 1, "metrics": {"loss": 0.5}}, 1))
                on_event(TrainingEvent("training_end", {"metrics": {}}, 2))

        mgr.model.fit = fake_fit
        x, y = self._toy()
        mgr._run_training(x, y, x, y)

        state = mgr.training_state.get_state()
        assert state["status"] == "Completed"
        assert state["phase"] == "Idle"

    def test_stop_event_marks_stopped(self):
        """A stop requested before fit returns yields a Stopped (cancelled) terminal state."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.model.fit = lambda *a, **k: None  # no-op fit
        mgr._stop_event.set()

        x, y = self._toy()
        mgr._run_training(x, y, x, y)

        assert mgr.training_state.get_state()["status"] == "Stopped"

    def test_interrupt_is_clean_cancellation(self):
        """TrainingInterrupted (raised from a callback on stop) is a clean stop, not a failure;
        _run_training swallows it and does not propagate."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        def fake_fit(*a, on_event=None, **k):
            raise TrainingInterrupted("stop_requested")

        mgr.model.fit = fake_fit
        x, y = self._toy()
        mgr._run_training(x, y, x, y)  # must NOT raise

        assert mgr.training_state.get_state()["status"] == "Stopped"

    def test_exception_marks_failed_and_reraises(self):
        """An unexpected error transitions to Failed and propagates to the training future."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        def fake_fit(*a, on_event=None, **k):
            raise RuntimeError("boom")

        mgr.model.fit = fake_fit
        x, y = self._toy()
        with pytest.raises(RuntimeError):
            mgr._run_training(x, y, x, y)

        assert mgr.training_state.get_state()["status"] == "Failed"

    def test_drains_candidate_progress_queue(self):
        """The retained drain thread discovers the lazily-created progress queue during fit and
        projects per-candidate frames onto the candidate_progress callback (the 50 Hz
        /ws/training side-channel)."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.network._persistent_progress_queue = None  # discovered dynamically

        def fake_fit(x, y, *, X_val=None, y_val=None, on_event=None, **kw):
            # Simulate _ensure_worker_pool creating the queue mid-fit.
            mgr.network._persistent_progress_queue = Queue()
            mgr.network._persistent_progress_queue.put({"candidate_id": 1, "candidate_uuid": "q-1", "epoch": 75, "total_epochs": 100, "correlation": 0.66})
            # Give the drain thread (~50 ms poll) time to discover + consume.
            time.sleep(0.5)

        mgr.model.fit = fake_fit
        progress_events = []
        mgr.monitor.register_callback("candidate_progress", lambda **kw: progress_events.append(kw["progress"]))

        x, y = self._toy()
        mgr._run_training(x, y, x, y)

        assert len(progress_events) >= 1
        assert progress_events[0]["epoch"] == 75
        assert progress_events[0]["total_epochs"] == 100

    def test_drain_exits_cleanly_without_progress_queue(self):
        """When no worker pool / queue is ever created, the drain thread polls harmlessly and
        _run_training still completes."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.network._persistent_progress_queue = None

        def fake_fit(x, y, *, X_val=None, y_val=None, on_event=None, **kw):
            time.sleep(0.15)

        mgr.model.fit = fake_fit
        x, y = self._toy()
        mgr._run_training(x, y, x, y)  # must not hang or raise

        assert mgr.training_state.get_state()["status"] == "Completed"
