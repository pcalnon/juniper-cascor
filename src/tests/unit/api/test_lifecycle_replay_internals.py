#!/usr/bin/env python
"""Unit coverage for the module-level replay helpers in
``api.lifecycle.manager`` (per-file coverage lift 4, C-5).

Targets the previously-uncovered branches of ``_ReplaySession`` (the
already-running ``start_thread`` guard, the ``_emit_frame`` monitor-None
and subscriber-raises guards, and the playing arm of the ``_run`` driver
loop) and the out-of-range per-unit slice branch of
``_WeightCache._build_payload``. Threaded paths are exercised via a bounded
poll and always torn down.
"""

import time
from unittest.mock import MagicMock

import pytest

from api.lifecycle.manager import _ReplaySession, _WeightCache

pytestmark = pytest.mark.unit


def _history(n: int = 5) -> dict:
    return {
        "train_loss": [1.0 / (i + 1) for i in range(n)],
        "train_accuracy": [i / (n or 1) for i in range(n)],
        "value_loss": [1.0 / (i + 1) for i in range(n)],
        "value_accuracy": [i / (n or 1) for i in range(n)],
    }


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class TestReplaySessionThread:
    """``start_thread`` idempotency + the ``_run`` playing arm."""

    def test_start_thread_idempotent_when_alive(self):
        session = _ReplaySession("snap-idem", _history(0), MagicMock())
        try:
            session.start_thread()
            first = session._thread
            assert first is not None
            # Second call sees a live thread and returns without spawning a new one.
            session.start_thread()
            assert session._thread is first
        finally:
            session.stop()

    def test_run_playing_advances_to_boundary(self):
        monitor = MagicMock()
        session = _ReplaySession("snap-run", _history(5), monitor)
        try:
            session.start_thread()
            session.set_speed(10.0)
            session.play()
            # The driver advances the time index while playing and auto-pauses
            # at the range boundary (length - 1).
            reached = _wait_until(lambda: session.time_index >= session.length - 1)
            assert reached
            # It emitted synthetic frames through the monitor while playing.
            assert monitor._trigger_callbacks.call_count >= 1
        finally:
            session.stop()

    def test_run_reverse_direction_pauses_at_lower_boundary(self):
        monitor = MagicMock()
        session = _ReplaySession("snap-rev", _history(5), monitor)
        try:
            session.seek(3)
            session.start_thread()
            session.set_speed(-10.0)
            session.play()
            reached = _wait_until(lambda: session.time_index <= session.range_start and session.paused)
            assert reached
        finally:
            session.stop()


class TestEmitFrameGuards:
    """``_emit_frame`` short-circuits."""

    def test_monitor_none_returns(self):
        session = _ReplaySession("snap-none", _history(3), None)
        # No monitor → emission is a no-op (must not raise).
        session._emit_frame(0)

    def test_index_out_of_range_returns(self):
        monitor = MagicMock()
        session = _ReplaySession("snap-oob", _history(3), monitor)
        session._emit_frame(99)  # index >= length → no callback
        monitor._trigger_callbacks.assert_not_called()

    def test_subscriber_exception_is_swallowed(self):
        monitor = MagicMock()
        monitor._trigger_callbacks.side_effect = RuntimeError("subscriber blew up")
        session = _ReplaySession("snap-raise", _history(3), monitor)
        # A raising subscriber must not crash the (best-effort) emit path.
        session._emit_frame(0)
        monitor._trigger_callbacks.assert_called_once()


class TestWeightCacheBuildPayload:
    """``_WeightCache._build_payload`` out-of-range per-unit slice branch."""

    def test_unit_slice_shorter_than_sample_index_is_skipped(self):
        # Three samples, but the single hidden unit only carries one per-sample
        # weight entry. Requesting sample 2 → local_idx (2) >= len(unit weights)
        # (1) → the unit is skipped in the payload.
        weight_history = {
            "sampling_strategy": "adaptive",
            "sampling_interval": 1,
            "sample_indices": [0, 1, 2],
            "output_weights": [[0.0], [1.0], [2.0]],
            "output_bias": [[0.0], [0.0], [0.0]],
            "hidden_units": [
                {"first_sample_index": 0, "activation": "tanh", "weights": [[9.0]], "bias": [0.5]},
            ],
        }
        cache = _WeightCache(weight_history)
        payload = cache.get(2)
        assert payload is not None
        assert payload["sample_index"] == 2
        # The under-length unit was skipped, leaving no hidden-unit slice.
        assert payload["hidden_units"] == []

    def test_unit_slice_present_is_included(self):
        weight_history = {
            "sampling_strategy": "adaptive",
            "sampling_interval": 1,
            "sample_indices": [0, 1],
            "output_weights": [[0.0], [1.0]],
            "output_bias": [[0.0], [0.0]],
            "hidden_units": [
                {"first_sample_index": 0, "activation": "tanh", "weights": [[9.0], [8.0]], "bias": [0.5, 0.6]},
            ],
        }
        cache = _WeightCache(weight_history)
        payload = cache.get(1)
        assert payload is not None
        assert len(payload["hidden_units"]) == 1
        assert payload["hidden_units"][0]["bias"] == pytest.approx(0.6)
