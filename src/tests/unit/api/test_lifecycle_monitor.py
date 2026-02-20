"""Tests for training monitor and training state."""

import json
import time

import pytest

from api.lifecycle.monitor import TrainingMonitor, TrainingState


@pytest.mark.unit
class TestTrainingState:
    """Test thread-safe training state."""

    def test_initial_state(self):
        """Initial state has expected defaults."""
        state = TrainingState()
        s = state.get_state()
        assert s["status"] == "Stopped"
        assert s["phase"] == "Idle"
        assert s["learning_rate"] == 0.0
        assert s["current_epoch"] == 0
        assert s["current_step"] == 0
        assert "timestamp" in s

    def test_update_state(self):
        """Can update individual fields."""
        state = TrainingState()
        state.update_state(status="Started", phase="Output", learning_rate=0.01)
        s = state.get_state()
        assert s["status"] == "Started"
        assert s["phase"] == "Output"
        assert s["learning_rate"] == 0.01

    def test_update_ignores_unknown_fields(self):
        """Unknown fields are silently ignored."""
        state = TrainingState()
        state.update_state(nonexistent_field="value", status="Started")
        s = state.get_state()
        assert s["status"] == "Started"
        assert "nonexistent_field" not in s

    def test_update_ignores_none_values(self):
        """None values leave fields unchanged."""
        state = TrainingState()
        state.update_state(status="Started")
        state.update_state(status=None, phase="Output")
        s = state.get_state()
        assert s["status"] == "Started"
        assert s["phase"] == "Output"

    def test_timestamp_auto_updates(self):
        """Timestamp updates automatically on state change."""
        state = TrainingState()
        s1 = state.get_state()
        time.sleep(0.01)
        state.update_state(current_epoch=1)
        s2 = state.get_state()
        assert s2["timestamp"] > s1["timestamp"]

    def test_to_json(self):
        """State serializes to valid JSON."""
        state = TrainingState()
        state.update_state(status="Started", current_epoch=5)
        j = state.to_json()
        parsed = json.loads(j)
        assert parsed["status"] == "Started"
        assert parsed["current_epoch"] == 5


@pytest.mark.unit
class TestTrainingMonitor:
    """Test training metrics monitor."""

    def test_initial_state(self):
        """Monitor starts in non-training state."""
        monitor = TrainingMonitor()
        current = monitor.get_current_state()
        assert current["is_training"] is False
        assert current["current_epoch"] == 0
        assert current["total_metrics"] == 0

    def test_on_training_start(self):
        """Training start sets is_training flag."""
        monitor = TrainingMonitor()
        monitor.on_training_start()
        current = monitor.get_current_state()
        assert current["is_training"] is True

    def test_on_training_end(self):
        """Training end clears is_training flag."""
        monitor = TrainingMonitor()
        monitor.on_training_start()
        monitor.on_training_end()
        current = monitor.get_current_state()
        assert current["is_training"] is False

    def test_on_epoch_end_records_metrics(self):
        """Epoch end records metrics to buffer."""
        monitor = TrainingMonitor()
        monitor.on_epoch_end(
            epoch=1,
            loss=0.5,
            accuracy=0.75,
            learning_rate=0.01,
            hidden_units=0,
        )
        assert monitor.get_current_state()["total_metrics"] == 1
        metrics = monitor.get_recent_metrics(1)
        assert len(metrics) == 1
        assert metrics[0]["epoch"] == 1
        assert metrics[0]["loss"] == 0.5
        assert metrics[0]["accuracy"] == 0.75

    def test_on_epoch_end_with_validation(self):
        """Epoch end includes validation metrics when provided."""
        monitor = TrainingMonitor()
        monitor.on_epoch_end(
            epoch=1,
            loss=0.5,
            accuracy=0.75,
            learning_rate=0.01,
            hidden_units=0,
            validation_loss=0.6,
            validation_accuracy=0.7,
        )
        metrics = monitor.get_recent_metrics(1)
        assert metrics[0]["validation_loss"] == 0.6
        assert metrics[0]["validation_accuracy"] == 0.7

    def test_get_recent_metrics_limit(self):
        """get_recent_metrics respects count limit."""
        monitor = TrainingMonitor()
        for i in range(10):
            monitor.on_epoch_end(
                epoch=i,
                loss=1.0 / (i + 1),
                accuracy=0.5,
                learning_rate=0.01,
                hidden_units=0,
            )
        recent = monitor.get_recent_metrics(3)
        assert len(recent) == 3
        assert recent[0]["epoch"] == 7
        assert recent[-1]["epoch"] == 9

    def test_get_all_metrics(self):
        """get_all_metrics returns complete buffer."""
        monitor = TrainingMonitor()
        for i in range(5):
            monitor.on_epoch_end(
                epoch=i,
                loss=0.5,
                accuracy=0.5,
                learning_rate=0.01,
                hidden_units=0,
            )
        all_metrics = monitor.get_all_metrics()
        assert len(all_metrics) == 5

    def test_clear_metrics(self):
        """clear_metrics empties the buffer."""
        monitor = TrainingMonitor()
        monitor.on_epoch_end(
            epoch=1,
            loss=0.5,
            accuracy=0.5,
            learning_rate=0.01,
            hidden_units=0,
        )
        monitor.clear_metrics()
        assert monitor.get_current_state()["total_metrics"] == 0

    def test_on_cascade_add(self):
        """Cascade add increments hidden unit count."""
        monitor = TrainingMonitor()
        monitor.on_cascade_add(hidden_unit_index=0, correlation=0.8)
        assert monitor.get_current_state()["current_hidden_units"] == 1
        monitor.on_cascade_add(hidden_unit_index=1, correlation=0.7)
        assert monitor.get_current_state()["current_hidden_units"] == 2

    def test_register_callback(self):
        """Registered callbacks are called on events."""
        monitor = TrainingMonitor()
        called = []
        monitor.register_callback("epoch_end", lambda **kwargs: called.append(kwargs))
        monitor.on_epoch_end(
            epoch=1,
            loss=0.5,
            accuracy=0.75,
            learning_rate=0.01,
            hidden_units=0,
        )
        assert len(called) == 1
        assert called[0]["epoch"] == 1

    def test_register_unknown_callback(self):
        """Registering unknown event type logs warning but doesn't crash."""
        monitor = TrainingMonitor()
        monitor.register_callback("unknown_event", lambda: None)
        # Should not raise

    def test_metrics_buffer_bounded(self):
        """Metrics buffer respects maxlen."""
        monitor = TrainingMonitor()
        for i in range(10001):
            monitor.on_epoch_end(
                epoch=i,
                loss=0.5,
                accuracy=0.5,
                learning_rate=0.01,
                hidden_units=0,
            )
        assert monitor.get_current_state()["total_metrics"] == 10000

    def test_poll_metrics_queue(self):
        """Metrics queue receives epoch data."""
        monitor = TrainingMonitor()
        monitor.on_epoch_end(
            epoch=1,
            loss=0.5,
            accuracy=0.75,
            learning_rate=0.01,
            hidden_units=0,
        )
        metric = monitor.poll_metrics_queue(timeout=0.1)
        assert metric is not None
        assert metric["epoch"] == 1

    def test_poll_metrics_queue_empty(self):
        """Empty queue returns None."""
        monitor = TrainingMonitor()
        result = monitor.poll_metrics_queue(timeout=0.01)
        assert result is None

    def test_training_start_clears_buffer(self):
        """Starting training clears existing metrics."""
        monitor = TrainingMonitor()
        monitor.on_epoch_end(
            epoch=1,
            loss=0.5,
            accuracy=0.5,
            learning_rate=0.01,
            hidden_units=0,
        )
        monitor.on_training_start()
        assert monitor.get_current_state()["total_metrics"] == 0
