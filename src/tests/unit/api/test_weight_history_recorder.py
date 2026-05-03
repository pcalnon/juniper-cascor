#!/usr/bin/env python
"""Unit tests for the training-loop weight history recorder (CAN-015g g-6).

Covers:
- Initialization: pre-g-6 networks (no ``weight_history`` attribute)
  pick up the V2 capture surface seamlessly.
- Trigger ordering: every-Nth-epoch, cascade-grow events, terminal
  capture all populate the history correctly. Dedup by epoch when
  multiple triggers fire at the same epoch.
- Config tunables: ``sampling_interval=0`` disables the periodic
  trigger; ``max_samples=0`` disables decimation.
- Decimation: cascade-add samples are retained; inter-cascade are
  decimated 2× when the cap is exceeded.
- Hidden-unit slicing: ``first_sample_index`` is a sample-list index
  (matches the g-2 cache convention).

The recorder is exercised against a synthetic ``network`` mock so
the tests don't drag in a full CascadeCorrelationNetwork instance —
keeps the tests fast and isolates the recorder logic from any
training-loop side effects.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Add parent directories for imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.lifecycle.manager import _WeightHistoryRecorder  # noqa: E402
from api.lifecycle.monitor import TrainingMonitor  # noqa: E402

pytestmark = pytest.mark.unit


def _mock_network(num_hidden=0, in_size=2, out_size=1, sampling_interval=10, max_samples=100):
    """Build a stand-in for a CascadeCorrelationNetwork.

    Just exposes the attributes the recorder reads:
    ``output_weights``, ``output_bias``, ``hidden_units``,
    ``config.weight_history_sampling_interval``,
    ``config.weight_history_max_samples``.
    """
    config = SimpleNamespace(
        weight_history_sampling_interval=sampling_interval,
        weight_history_max_samples=max_samples,
    )
    network = SimpleNamespace(
        config=config,
        output_weights=torch.zeros(in_size + num_hidden, out_size, dtype=torch.float32),
        output_bias=torch.zeros(out_size, dtype=torch.float32),
        hidden_units=[{"weights": torch.zeros(in_size + i, dtype=torch.float32), "bias": torch.zeros(1, dtype=torch.float32)} for i in range(num_hidden)],
    )
    return network


def _real_monitor():
    """Use the real TrainingMonitor so callbacks fire through ``_trigger_callbacks``."""
    return TrainingMonitor()


# =============================================================================
# Initialization
# =============================================================================


class TestWeightHistoryRecorderInit:
    def test_pre_g6_network_gets_weight_history_initialized(self):
        net = _mock_network()
        assert not hasattr(net, "weight_history")
        recorder = _WeightHistoryRecorder(net, _real_monitor())
        assert hasattr(net, "weight_history")
        assert net.weight_history["sample_indices"] == []
        assert net.weight_history["sampling_strategy"] == "adaptive"
        assert net.weight_history["sampling_interval"] == 10
        assert recorder.sampling_interval == 10
        assert recorder.max_samples == 100

    def test_existing_weight_history_preserved(self):
        net = _mock_network()
        net.weight_history = {
            "sampling_strategy": "every_n",
            "sampling_interval": 7,
            "sample_indices": [0, 7],
            "output_weights": [np.zeros((2, 1), dtype=np.float32)] * 2,
            "output_bias": [np.zeros(1, dtype=np.float32)] * 2,
            "hidden_units": [],
            "_captured_epochs": [0, 7],
            "_cascade_epochs": set(),
        }
        recorder = _WeightHistoryRecorder(net, _real_monitor())
        # Existing samples are preserved; only sampling_interval is
        # refreshed to the recorder's value (matches runtime PATCH).
        assert net.weight_history["sample_indices"] == [0, 7]
        assert recorder.sampling_interval == 10
        assert net.weight_history["sampling_interval"] == 10  # refreshed

    def test_constructor_overrides(self):
        net = _mock_network(sampling_interval=10, max_samples=100)
        recorder = _WeightHistoryRecorder(net, _real_monitor(), sampling_interval=5, max_samples=50)
        assert recorder.sampling_interval == 5
        assert recorder.max_samples == 50

    def test_register_idempotent(self):
        net = _mock_network()
        monitor = _real_monitor()
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.register()
        before = len(monitor.callbacks.get("epoch_end", []))
        recorder.register()  # second call should NOT re-register
        after = len(monitor.callbacks.get("epoch_end", []))
        assert before == after


# =============================================================================
# Periodic trigger
# =============================================================================


class TestPeriodicTrigger:
    def test_nth_epoch_captures(self):
        net = _mock_network(sampling_interval=5)
        monitor = _real_monitor()
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.register()
        for epoch in range(0, 12):
            monitor.on_epoch_end(epoch=epoch, loss=1.0, accuracy=0.5, learning_rate=0.1)
        # Captures at 0, 5, 10
        assert net.weight_history["sample_indices"] == [0, 5, 10]

    def test_sampling_interval_zero_disables_periodic(self):
        net = _mock_network(sampling_interval=0)
        monitor = _real_monitor()
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.register()
        for epoch in range(0, 20):
            monitor.on_epoch_end(epoch=epoch, loss=1.0, accuracy=0.5, learning_rate=0.1)
        assert net.weight_history["sample_indices"] == []

    def test_capture_records_independent_copy(self):
        net = _mock_network(sampling_interval=1)
        monitor = _real_monitor()
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.register()
        monitor.on_epoch_end(epoch=0, loss=1.0, accuracy=0.5, learning_rate=0.1)
        # Mutate the network's weights — the captured sample must NOT change.
        net.output_weights = net.output_weights + 999.0
        np.testing.assert_array_equal(net.weight_history["output_weights"][0], np.zeros((2, 1), dtype=np.float32))


# =============================================================================
# Cascade-grow trigger
# =============================================================================


class TestCascadeAddTrigger:
    def test_cascade_add_captures_at_current_epoch(self):
        net = _mock_network(sampling_interval=0)  # only cascade-add fires
        monitor = _real_monitor()
        # Set monitor's current_epoch via on_epoch_end.
        monitor.on_epoch_end(epoch=17, loss=1.0, accuracy=0.5, learning_rate=0.1)
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.register()
        monitor.on_cascade_add(hidden_unit_index=0, correlation=0.9)
        assert net.weight_history["sample_indices"] == [17]
        assert 17 in net.weight_history["_cascade_epochs"]

    def test_cascade_add_dedup_with_periodic(self):
        net = _mock_network(sampling_interval=10)
        monitor = _real_monitor()
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.register()
        monitor.on_epoch_end(epoch=10, loss=1.0, accuracy=0.5, learning_rate=0.1)
        monitor.on_cascade_add(hidden_unit_index=0, correlation=0.9)
        # Single sample at epoch 10 — cascade marker set.
        assert net.weight_history["sample_indices"] == [10]
        assert 10 in net.weight_history["_cascade_epochs"]


# =============================================================================
# Terminal capture
# =============================================================================


class TestCaptureTerminal:
    def test_terminal_captures_at_current_epoch(self):
        net = _mock_network(sampling_interval=0)
        monitor = _real_monitor()
        monitor.on_epoch_end(epoch=99, loss=0.1, accuracy=0.95, learning_rate=0.1)
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.register()
        recorder.capture_terminal()
        assert net.weight_history["sample_indices"] == [99]
        # Terminal samples are decimation-exempt (marked as cascade).
        assert 99 in net.weight_history["_cascade_epochs"]


# =============================================================================
# Decimation
# =============================================================================


class TestDecimation:
    def test_decimation_keeps_cascade_samples(self):
        net = _mock_network(sampling_interval=1, max_samples=4)
        monitor = _real_monitor()
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.register()
        # Epochs 0..4 = 5 samples; cap is 4. Mark epoch 2 as cascade-add.
        for epoch in range(0, 5):
            monitor.on_epoch_end(epoch=epoch, loss=1.0, accuracy=0.5, learning_rate=0.1)
            if epoch == 2:
                monitor.on_cascade_add(hidden_unit_index=0, correlation=0.9)
        # After exceeding cap at epoch 4: decimation keeps cascade
        # epoch 2 unconditionally; non-cascade samples halve.
        assert 2 in net.weight_history["sample_indices"]
        # Final length must be ≤ original since decimation just ran.
        assert len(net.weight_history["sample_indices"]) < 5

    def test_max_samples_zero_disables_decimation(self):
        net = _mock_network(sampling_interval=1, max_samples=0)
        monitor = _real_monitor()
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.register()
        for epoch in range(0, 50):
            monitor.on_epoch_end(epoch=epoch, loss=1.0, accuracy=0.5, learning_rate=0.1)
        assert len(net.weight_history["sample_indices"]) == 50

    def test_decimation_doubles_recorded_interval(self):
        net = _mock_network(sampling_interval=1, max_samples=2)
        monitor = _real_monitor()
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.register()
        for epoch in range(0, 4):
            monitor.on_epoch_end(epoch=epoch, loss=1.0, accuracy=0.5, learning_rate=0.1)
        # Decimation fired at least once — recorded interval doubled.
        assert net.weight_history["sampling_interval"] >= 2


# =============================================================================
# Hidden-unit slicing
# =============================================================================


class TestHiddenUnitSlicing:
    def test_first_sample_index_uses_sample_list_index(self):
        net = _mock_network(num_hidden=0, sampling_interval=5)
        monitor = _real_monitor()
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.register()
        # Sample 0: no hidden units
        monitor.on_epoch_end(epoch=0, loss=1.0, accuracy=0.5, learning_rate=0.1)
        # Add a hidden unit, then sample 1 captures it.
        net.hidden_units.append({"weights": torch.tensor([1.0, 2.0], dtype=torch.float32), "bias": torch.tensor([0.5], dtype=torch.float32)})
        monitor.on_epoch_end(epoch=5, loss=0.9, accuracy=0.6, learning_rate=0.1)
        # Cache convention: ``first_sample_index`` is an index into
        # ``sample_indices``, NOT an epoch number. Unit appeared at
        # sample list index 1 (epoch 5).
        assert len(net.weight_history["hidden_units"]) == 1
        unit = net.weight_history["hidden_units"][0]
        assert unit["first_sample_index"] == 1
        assert len(unit["weights"]) == 1
        np.testing.assert_array_equal(unit["weights"][0], np.array([1.0, 2.0], dtype=np.float32))

    def test_unit_bias_recorded_as_python_float(self):
        net = _mock_network(num_hidden=1, sampling_interval=1)
        net.hidden_units[0]["bias"] = torch.tensor([0.42], dtype=torch.float32)
        monitor = _real_monitor()
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.register()
        monitor.on_epoch_end(epoch=0, loss=1.0, accuracy=0.5, learning_rate=0.1)
        unit = net.weight_history["hidden_units"][0]
        assert isinstance(unit["bias"][0], float)
        assert unit["bias"][0] == pytest.approx(0.42)


# =============================================================================
# Robustness
# =============================================================================


class TestRobustness:
    def test_callback_swallows_exceptions(self):
        net = _mock_network(sampling_interval=1)
        monitor = _real_monitor()
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.register()
        # Sabotage the network's output_weights so _tensor_to_numpy
        # fails; recorder must not crash the monitor's emission path.
        net.output_weights = "not a tensor"
        monitor.on_epoch_end(epoch=0, loss=1.0, accuracy=0.5, learning_rate=0.1)
        # Sample was appended (output_weights for it is None),
        # the training thread did not raise.
        assert net.weight_history["sample_indices"] == [0]

    def test_terminal_without_monitor_state_is_noop(self):
        net = _mock_network()
        monitor = MagicMock()
        monitor.current_epoch = None
        recorder = _WeightHistoryRecorder(net, monitor)
        recorder.capture_terminal()
        assert net.weight_history["sample_indices"] == []
