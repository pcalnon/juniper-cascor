#!/usr/bin/env python
"""Unit coverage for the module-level weight-history helpers in
``api.lifecycle.manager`` (per-file coverage lift 4, C-5).

Targets the previously-uncovered branches of ``_WeightHistoryRecorder``
(trigger-callback guards, dedupe rewrite, per-unit slicing, tensor-copy
edge cases, activation-name extraction, and cascade-aware decimation) and
the ``_PreSwapSnapshot`` frozen container. All fast unit tests driving the
helpers directly against lightweight fake networks — no training, no live
sockets, no I/O.
"""

import logging
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from api.lifecycle.manager import _PreSwapSnapshot, _WeightHistoryRecorder

pytestmark = pytest.mark.unit


def _act_tanh(x):  # pragma: no cover - identity stand-in; only its __name__ is read
    return x


_act_tanh.__name__ = "tanh"


def _make_unit(n_in: int = 3, bias: float = 0.25, activation=_act_tanh) -> dict:
    """A hidden-unit dict shaped like ``cascade_correlation`` produces."""
    return {
        "weights": torch.arange(float(n_in)).reshape(n_in),
        "bias": torch.tensor(bias),
        "activation_fn": activation,
    }


def _make_network(*, hidden_units=None, config=None) -> types.SimpleNamespace:
    net = types.SimpleNamespace()
    net.config = config
    net.hidden_units = list(hidden_units) if hidden_units is not None else []
    net.output_weights = torch.ones(2, 2)
    net.output_bias = torch.zeros(2)
    return net


def _make_recorder(*, hidden_units=None, sampling_interval=1, max_samples=1000, config=None):
    net = _make_network(hidden_units=hidden_units, config=config)
    monitor = MagicMock()
    monitor.current_epoch = 0
    return _WeightHistoryRecorder(net, monitor, sampling_interval=sampling_interval, max_samples=max_samples), net, monitor


class TestPreSwapSnapshot:
    """The frozen ``_PreSwapSnapshot`` container assigns every slot."""

    def test_all_slots_assigned(self):
        ow = torch.ones(2, 2)
        ob = torch.zeros(2)
        huw = [torch.ones(3)]
        snap = _PreSwapSnapshot(
            train_x=torch.ones(4, 2),
            train_y=torch.zeros(4, 2),
            val_x=None,
            val_y=None,
            state_dict={"k": 1},
            input_size=2,
            output_size=2,
            dataset_config={"dataset_type": "spiral"},
            active_output_dim=2,
            output_weights=ow,
            output_bias=ob,
            hidden_unit_weights=huw,
        )
        assert snap.input_size == 2
        assert snap.output_size == 2
        assert snap.dataset_config == {"dataset_type": "spiral"}
        assert snap.active_output_dim == 2
        assert snap.output_weights is ow
        assert snap.output_bias is ob
        assert snap.hidden_unit_weights is huw
        assert snap.state_dict == {"k": 1}


class TestWeightRecorderInit:
    """``__init__`` / ``_init_weight_history`` idempotency + config reads."""

    def test_init_creates_weight_history_dict(self):
        rec, net, _ = _make_recorder(sampling_interval=7)
        assert isinstance(net.weight_history, dict)
        assert net.weight_history["sampling_interval"] == 7
        assert net.weight_history["sample_indices"] == []

    def test_init_reads_config_defaults(self):
        cfg = types.SimpleNamespace(weight_history_sampling_interval=13, weight_history_max_samples=77)
        net = _make_network(config=cfg)
        rec = _WeightHistoryRecorder(net, MagicMock())
        assert rec.sampling_interval == 13
        assert rec.max_samples == 77

    def test_init_weight_history_is_idempotent_on_existing_dict(self):
        rec, net, _ = _make_recorder(sampling_interval=5)
        net.weight_history["sample_indices"].append(0)
        rec.sampling_interval = 9
        rec._init_weight_history()
        # Existing samples preserved; interval refreshed.
        assert net.weight_history["sample_indices"] == [0]
        assert net.weight_history["sampling_interval"] == 9


class TestOnEpochEnd:
    """``_on_epoch_end`` periodic-trigger guards."""

    def test_disabled_interval_is_noop(self):
        rec, net, _ = _make_recorder(sampling_interval=0)
        rec._on_epoch_end(epoch=10)
        assert net.weight_history["sample_indices"] == []

    def test_missing_epoch_returns(self):
        rec, net, _ = _make_recorder(sampling_interval=1)
        rec._on_epoch_end()  # no epoch kwarg
        assert net.weight_history["sample_indices"] == []

    def test_non_integer_epoch_returns(self):
        rec, net, _ = _make_recorder(sampling_interval=1)
        rec._on_epoch_end(epoch="not-an-int")
        assert net.weight_history["sample_indices"] == []

    def test_non_multiple_epoch_skipped(self):
        rec, net, _ = _make_recorder(sampling_interval=50)
        rec._on_epoch_end(epoch=7)
        assert net.weight_history["sample_indices"] == []

    def test_capture_exception_is_swallowed(self):
        rec, net, _ = _make_recorder(sampling_interval=1)
        with patch.object(rec, "_capture", side_effect=RuntimeError("boom")):
            rec._on_epoch_end(epoch=2)  # multiple of interval → capture attempted

    def test_capture_happy_path_records_sample(self):
        rec, net, _ = _make_recorder(hidden_units=[_make_unit()], sampling_interval=1)
        rec._on_epoch_end(epoch=4)
        assert net.weight_history["sample_indices"] == [4]


class TestOnCascadeAdd:
    """``_on_cascade_add`` uses the monitor's ``current_epoch``."""

    def test_none_epoch_returns(self):
        rec, net, monitor = _make_recorder()
        monitor.current_epoch = None
        rec._on_cascade_add()
        assert net.weight_history["sample_indices"] == []

    def test_non_integer_epoch_returns(self):
        rec, net, monitor = _make_recorder()
        monitor.current_epoch = object()  # int() raises TypeError
        rec._on_cascade_add()
        assert net.weight_history["sample_indices"] == []

    def test_capture_exception_swallowed(self):
        rec, net, monitor = _make_recorder()
        monitor.current_epoch = 3
        with patch.object(rec, "_capture", side_effect=RuntimeError("boom")):
            rec._on_cascade_add()

    def test_missing_monitor_returns(self):
        net = _make_network()
        rec = _WeightHistoryRecorder(net, None, sampling_interval=1)
        rec._on_cascade_add()  # monitor is None → early return
        assert net.weight_history["sample_indices"] == []


class TestCaptureTerminal:
    """``capture_terminal`` marks the final sample cascade-equivalent."""

    def test_non_integer_epoch_returns(self):
        rec, net, monitor = _make_recorder()
        monitor.current_epoch = "x"
        rec.capture_terminal()
        assert net.weight_history["sample_indices"] == []

    def test_capture_exception_swallowed(self):
        rec, net, monitor = _make_recorder()
        monitor.current_epoch = 9
        with patch.object(rec, "_capture", side_effect=ValueError("boom")):
            rec.capture_terminal()

    def test_happy_path_marks_cascade_epoch(self):
        rec, net, monitor = _make_recorder(hidden_units=[_make_unit()])
        monitor.current_epoch = 12
        rec.capture_terminal()
        assert 12 in net.weight_history["_cascade_epochs"]


class TestCaptureMechanics:
    """``_capture`` append + dedupe-rewrite paths."""

    def test_append_then_dedupe_rewrite(self):
        unit = _make_unit()
        rec, net, _ = _make_recorder(hidden_units=[unit], sampling_interval=1)
        # First capture at epoch 5 appends a fresh sample.
        rec._capture(5, is_cascade_add=False)
        assert net.weight_history["_captured_epochs"] == [5]
        first_len = len(net.weight_history["output_weights"])
        # Mutate the unit's weights, then re-capture the SAME epoch: the
        # dedupe path (_write_sample_at) overwrites in place rather than
        # appending, and the cascade flag is recorded.
        unit["weights"] = torch.full((3,), 9.0)
        rec._capture(5, is_cascade_add=True)
        assert net.weight_history["_captured_epochs"] == [5]
        assert len(net.weight_history["output_weights"]) == first_len
        assert 5 in net.weight_history["_cascade_epochs"]

    def test_rewrite_backfills_unit_added_after_first_sample(self):
        """A unit added between a sample's first capture and its rewrite gets a slot."""
        rec, net, _ = _make_recorder(hidden_units=[], sampling_interval=1)
        rec._capture(5, is_cascade_add=False)  # 0 units → no hidden slices
        # Grow a unit, then rewrite the same epoch → _append_hidden_unit_slices
        # runs with rewrite=True and first_sample_index for the new unit.
        net.hidden_units.append(_make_unit())
        rec._capture(5, is_cascade_add=True)
        assert len(net.weight_history["hidden_units"]) == 1

    def test_capture_earlier_sample_after_later_unit_skips_negative_local(self):
        """Rewriting an earlier sample whose index precedes a unit's first sample skips it."""
        rec, net, _ = _make_recorder(hidden_units=[], sampling_interval=1)
        rec._capture(1, is_cascade_add=False)  # sample index 0, no units
        net.hidden_units.append(_make_unit())
        rec._capture(2, is_cascade_add=False)  # sample index 1, unit first_sample_index=1
        # Rewrite epoch 1 (sample index 0): unit's first_sample_index (1) > 0
        # → local < 0 → the per-unit slice loop `continue`s.
        rec._capture(1, is_cascade_add=True)
        assert net.weight_history["hidden_units"][0]["first_sample_index"] == 1


class TestTensorCopyHelpers:
    """``_copy_unit`` / ``_tensor_to_numpy`` / ``_unit_activation_name`` edges."""

    def test_copy_unit_out_of_range(self):
        rec, net, _ = _make_recorder(hidden_units=[_make_unit()])
        assert rec._copy_unit(99) == (None, None)

    def test_copy_unit_none_bias_returns_zero(self):
        unit = {"weights": torch.ones(3), "bias": None}
        rec, net, _ = _make_recorder(hidden_units=[unit])
        w, b = rec._copy_unit(0)
        assert b == 0.0
        assert w is not None

    def test_copy_unit_object_attribute_access(self):
        unit = types.SimpleNamespace(weights=torch.ones(3), bias=torch.tensor(0.5))
        rec, net, _ = _make_recorder(hidden_units=[unit])
        w, b = rec._copy_unit(0)
        assert w is not None
        assert b == pytest.approx(0.5)

    def test_tensor_to_numpy_none(self):
        assert _WeightHistoryRecorder._tensor_to_numpy(None) is None

    def test_tensor_to_numpy_from_list(self):
        arr = _WeightHistoryRecorder._tensor_to_numpy([1.0, 2.0, 3.0])
        assert arr is not None
        assert list(arr) == [1.0, 2.0, 3.0]

    def test_unit_activation_name_out_of_range(self):
        rec, net, _ = _make_recorder(hidden_units=[_make_unit()])
        assert rec._unit_activation_name(99) == ""

    def test_unit_activation_name_from_dict(self):
        rec, net, _ = _make_recorder(hidden_units=[_make_unit(activation=_act_tanh)])
        assert rec._unit_activation_name(0) == "tanh"

    def test_unit_activation_name_non_dict_unit(self):
        unit = types.SimpleNamespace(weights=torch.ones(3), bias=torch.tensor(0.0))
        rec, net, _ = _make_recorder(hidden_units=[unit])
        assert rec._unit_activation_name(0) == ""


class TestDecimation:
    """``_decimate`` enforces the soft cap while retaining cascade samples."""

    def test_decimate_retains_cascade_and_reindexes_units(self):
        unit = _make_unit()
        rec, net, _ = _make_recorder(hidden_units=[unit], sampling_interval=1, max_samples=3)
        # Capture epoch 0 as a cascade-add (retained), then non-cascade epochs
        # 1..4. On the 4th append len(captured) > max_samples → decimation fires
        # with a hidden unit present (exercises the per-unit re-index loop).
        rec._capture(0, is_cascade_add=True)
        for e in (1, 2, 3, 4):
            rec._capture(e, is_cascade_add=False)
        captured = net.weight_history["_captured_epochs"]
        # Cascade sample (epoch 0) always survives decimation.
        assert 0 in captured
        # The soft cap dropped at least one non-cascade sample.
        assert len(captured) < 5
        # sampling_interval was bumped to reflect the decimation.
        assert net.weight_history["sampling_interval"] >= 2
