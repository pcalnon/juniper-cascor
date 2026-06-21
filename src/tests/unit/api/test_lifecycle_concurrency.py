"""Concurrency regression tests for TrainingLifecycleManager.

Track 3 / Phase 3B coverage:

- **CONC-02 / BUG-CC-16** — `_broadcast_training_state` reads/writes
  `_last_state_broadcast_time` outside any lock. Two callers can both read
  `now - last >= interval`, both pass the throttle, and both broadcast,
  defeating the GAP-WS-21 coalescer.

- **CONC-03 / BUG-CC-17** — `_extract_and_record_metrics` formerly held the
  metrics lock only briefly to snapshot history, released it across the
  per-entry `on_epoch_end` calls, and re-acquired it just to advance the
  high-water-mark. Two callers can both observe `last_emitted = 0`, both
  emit the same epoch-1 entry, and only then race on the write — so
  `TrainingMonitor.on_epoch_end` is invoked twice per epoch.

The throttle race is widened by patching `time.monotonic` to return identical
ticks under a barrier; the split-lock race is widened by patching
`TrainingMonitor.on_epoch_end` to sleep ~5 ms so the window between snapshot
and high-water-mark advance is observable to other threads. Both tests fail
reliably on the pre-fix code and pass after the lock-scope changes.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from api.lifecycle.manager import TrainingLifecycleManager


def _build_history(n_epochs: int) -> dict:
    return {
        "train_loss": [1.0 / (i + 1) for i in range(n_epochs)],
        "train_accuracy": [0.5 + 0.01 * i for i in range(n_epochs)],
        "value_loss": [1.5 / (i + 1) for i in range(n_epochs)],
        "value_accuracy": [0.4 + 0.01 * i for i in range(n_epochs)],
    }


@pytest.mark.unit
class TestBroadcastThrottleRace:
    """CONC-02 / BUG-CC-16 regression coverage."""

    def test_concurrent_broadcast_passes_throttle_only_once(self, monkeypatch):
        """Two near-simultaneous broadcasts must not both pass the throttle."""
        mgr = TrainingLifecycleManager()
        ws_manager = MagicMock()
        ws_manager.broadcast_from_thread = MagicMock()
        mgr._ws_manager = ws_manager
        mgr._state_throttle_interval = 1.0
        # Initial timestamp far enough in the past that the very first call
        # would otherwise pass the throttle.
        mgr._last_state_broadcast_time = 0.0

        # Force time.monotonic to return identical ticks for every caller so
        # the throttle window is observed simultaneously by every thread.
        fixed_now = 100.0
        monkeypatch.setattr("api.lifecycle.manager.time.monotonic", lambda: fixed_now)

        # Stub out training_state.get_state so a non-terminal status is reported.
        mgr.training_state = MagicMock()
        mgr.training_state.get_state.return_value = {"status": "Running"}

        n_threads = 16
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            mgr._broadcast_training_state(force=False)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Without the lock multiple threads pass the throttle and each calls
        # broadcast_from_thread. With the lock exactly one wins.
        assert ws_manager.broadcast_from_thread.call_count == 1, f"throttle race: {ws_manager.broadcast_from_thread.call_count} " "broadcasts emitted in a single throttle window"

    def test_terminal_broadcast_always_emits_under_lock(self, monkeypatch):
        """Terminal statuses bypass the throttle but still update the timestamp."""
        mgr = TrainingLifecycleManager()
        ws_manager = MagicMock()
        ws_manager.broadcast_from_thread = MagicMock()
        mgr._ws_manager = ws_manager
        mgr._state_throttle_interval = 60.0  # very long throttle
        mgr._last_state_broadcast_time = 100.0

        monkeypatch.setattr("api.lifecycle.manager.time.monotonic", lambda: 100.5)
        mgr.training_state = MagicMock()
        mgr.training_state.get_state.return_value = {"status": "Completed"}

        mgr._broadcast_training_state(force=False)

        assert ws_manager.broadcast_from_thread.call_count == 1
        assert mgr._last_state_broadcast_time == 100.5


@pytest.mark.unit
class TestExtractAndRecordMetricsRace:
    """CONC-03 / BUG-CC-17 regression coverage."""

    def test_concurrent_extract_does_not_double_emit(self, monkeypatch):
        """Two concurrent _extract_and_record_metrics calls must not both emit the same epoch."""
        mgr = TrainingLifecycleManager()
        # Plant a fake network with deterministic history.
        n_epochs = 3
        fake_network = MagicMock()
        fake_network.history = _build_history(n_epochs)
        fake_network.hidden_units = []
        fake_network.learning_rate = 0.01
        mgr.network = fake_network

        # Replace TrainingMonitor.on_epoch_end with a sleepy recording stub
        # so the per-entry loop is observable to other threads.
        emit_calls: list = []
        emit_lock = threading.Lock()

        def slow_on_epoch_end(epoch: int, **kwargs: Any) -> None:
            time.sleep(0.005)
            with emit_lock:
                emit_calls.append(epoch)

        monkeypatch.setattr(mgr.monitor, "on_epoch_end", slow_on_epoch_end)
        mgr.training_state = MagicMock()

        n_threads = 8
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            mgr._extract_and_record_metrics()

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly n_epochs emissions must happen in total — one per epoch.
        # Without the single-lock scope, multiple threads each emit all
        # n_epochs entries (n_threads * n_epochs total).
        emitted_epochs = sorted(emit_calls)
        expected = list(range(1, n_epochs + 1))
        assert emitted_epochs == expected, f"split-lock race: TrainingMonitor.on_epoch_end called for epochs " f"{emitted_epochs}, expected {expected}"
        assert mgr._last_emitted_history_len == n_epochs

    def test_extract_idempotent_when_no_new_history(self, monkeypatch):
        """A second call with no new entries must not re-emit anything."""
        mgr = TrainingLifecycleManager()
        fake_network = MagicMock()
        fake_network.history = _build_history(2)
        fake_network.hidden_units = []
        fake_network.learning_rate = 0.01
        mgr.network = fake_network

        calls: list = []
        monkeypatch.setattr(mgr.monitor, "on_epoch_end", lambda epoch, **kw: calls.append(epoch))
        mgr.training_state = MagicMock()

        mgr._extract_and_record_metrics()
        mgr._extract_and_record_metrics()

        assert sorted(calls) == [1, 2]
        assert mgr._last_emitted_history_len == 2
