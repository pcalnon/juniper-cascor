#!/usr/bin/env python
"""Unit tests for synthetic-event weight emission (CAN-015g g-3).

Covers:
- Sample-boundary epochs emit ``is_sample_boundary=True`` + a
  base64-encoded ``weights`` block.
- Sub-sample epochs emit ``is_sample_boundary=False`` and **no**
  ``weights`` field.
- V1 snapshots (no weight cache) emit no ``is_sample_boundary``
  field at all — preserves backward compatibility for canopy
  clients pre-dating the V2 protocol.
- The encoded tensor envelope round-trips correctly: base64 decode
  → reshape via ``shape`` → matches the source array.
- Hidden units are emitted with their per-sample slices and the
  scalar bias is shipped as a float (no base64 overhead for
  single-value fields).
"""

import base64
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add parent directories for imports (matches sibling test files).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.lifecycle.manager import _ReplaySession  # noqa: E402

pytestmark = pytest.mark.unit


def _history(num_epochs=10):
    return {
        "train_loss": [1.0 - 0.05 * i for i in range(num_epochs)],
        "value_loss": [1.1 - 0.05 * i for i in range(num_epochs)],
        "train_accuracy": [0.5 + 0.04 * i for i in range(num_epochs)],
        "value_accuracy": [0.5 + 0.04 * i for i in range(num_epochs)],
    }


def _weight_history(num_samples=3, num_hidden=2, in_size=2, out_size=1, sampling_interval=2):
    """Build a synthetic weight_history.

    ``sample_indices`` maps sample-index → epoch (matches the
    convention captured in g-2's docs: epoch numbers are 0-based and
    line up with the metric-array index used by ``_emit_frame``).
    """
    sample_indices = [i * sampling_interval for i in range(num_samples)]
    output_weights = []
    output_bias = []
    for i in range(num_samples):
        hid_at_sample = min(i, num_hidden)
        output_weights.append(np.random.RandomState(i).randn(in_size + hid_at_sample, out_size).astype(np.float32))
        output_bias.append(np.random.RandomState(i + 100).randn(out_size).astype(np.float32))
    hidden_units = []
    for unit_idx in range(num_hidden):
        first_sample_index = unit_idx + 1  # sample-index, NOT epoch (g-2 convention)
        unit_weights = []
        unit_bias = []
        for j in range(num_samples - first_sample_index):
            unit_weights.append(np.random.RandomState(unit_idx * 10 + j).randn(in_size + unit_idx).astype(np.float32))
            unit_bias.append(float(np.random.RandomState(unit_idx * 10 + j + 1).randn()))
        hidden_units.append(
            {
                "first_sample_index": first_sample_index,
                "activation": "tanh",
                "weights": unit_weights,
                "bias": unit_bias,
            }
        )
    return {
        "sampling_strategy": "adaptive",
        "sampling_interval": sampling_interval,
        "sample_indices": sample_indices,
        "output_weights": output_weights,
        "output_bias": output_bias,
        "hidden_units": hidden_units,
    }


def _make_session(weight_history=None, num_epochs=10):
    monitor = MagicMock()
    monitor._trigger_callbacks = MagicMock()
    session = _ReplaySession("snap-test", _history(num_epochs), monitor, weight_history=weight_history)
    return session, monitor


# =============================================================================
# V1 backward-compat
# =============================================================================


class TestV1EmissionBackwardCompat:
    """Snapshots without weight history must emit the pre-g-3 shape."""

    def test_no_is_sample_boundary_field_on_v1(self):
        session, monitor = _make_session(weight_history=None)
        session._emit_frame(0)
        assert monitor._trigger_callbacks.call_count == 1
        metrics = monitor._trigger_callbacks.call_args.kwargs["metrics"]
        assert "is_sample_boundary" not in metrics
        assert "weights" not in metrics

    def test_v1_metrics_unchanged(self):
        session, monitor = _make_session(weight_history=None)
        session._emit_frame(2)
        metrics = monitor._trigger_callbacks.call_args.kwargs["metrics"]
        assert metrics["epoch"] == 3  # 1-indexed
        assert metrics["replay"] is True
        assert metrics["snapshot_id"] == "snap-test"


# =============================================================================
# V2 sample-boundary detection
# =============================================================================


class TestSampleBoundaryDetection:
    def test_boundary_epoch_emits_weights(self):
        wh = _weight_history(num_samples=3, sampling_interval=2)  # samples at epochs 0, 2, 4
        session, monitor = _make_session(weight_history=wh)
        session._emit_frame(2)  # epoch index 2 == sample 1
        metrics = monitor._trigger_callbacks.call_args.kwargs["metrics"]
        assert metrics["is_sample_boundary"] is True
        assert "weights" in metrics
        assert metrics["weights"]["sample_index"] == 1

    def test_non_boundary_epoch_marks_false_no_weights(self):
        wh = _weight_history(num_samples=3, sampling_interval=2)  # samples at 0, 2, 4
        session, monitor = _make_session(weight_history=wh)
        session._emit_frame(1)  # not a sample boundary
        metrics = monitor._trigger_callbacks.call_args.kwargs["metrics"]
        assert metrics["is_sample_boundary"] is False
        assert "weights" not in metrics

    def test_first_epoch_boundary_emits(self):
        # sample_indices[0] == 0 — the very first emission lands on a boundary.
        wh = _weight_history(num_samples=3, sampling_interval=2)
        session, monitor = _make_session(weight_history=wh)
        session._emit_frame(0)
        metrics = monitor._trigger_callbacks.call_args.kwargs["metrics"]
        assert metrics["is_sample_boundary"] is True
        assert metrics["weights"]["sample_index"] == 0

    def test_emit_frame_skips_out_of_range(self):
        wh = _weight_history(num_samples=3, sampling_interval=2)
        session, monitor = _make_session(weight_history=wh)
        session._emit_frame(99)
        assert monitor._trigger_callbacks.call_count == 0


# =============================================================================
# Tensor envelope
# =============================================================================


class TestEncodedTensorEnvelope:
    def test_output_weights_round_trip(self):
        wh = _weight_history(num_samples=3, sampling_interval=2)
        session, monitor = _make_session(weight_history=wh)
        session._emit_frame(4)  # sample index 2
        metrics = monitor._trigger_callbacks.call_args.kwargs["metrics"]
        ow = metrics["weights"]["output_weights"]
        assert ow["dtype"] == "float32"
        assert ow["shape"] == list(wh["output_weights"][2].shape)
        decoded = np.frombuffer(base64.b64decode(ow["data"]), dtype=np.float32).reshape(ow["shape"])
        np.testing.assert_array_equal(decoded, wh["output_weights"][2])

    def test_output_bias_round_trip(self):
        wh = _weight_history(num_samples=3, sampling_interval=2)
        session, monitor = _make_session(weight_history=wh)
        session._emit_frame(2)  # sample index 1
        metrics = monitor._trigger_callbacks.call_args.kwargs["metrics"]
        ob = metrics["weights"]["output_bias"]
        assert ob["dtype"] == "float32"
        decoded = np.frombuffer(base64.b64decode(ob["data"]), dtype=np.float32).reshape(ob["shape"])
        np.testing.assert_array_equal(decoded, wh["output_bias"][1])

    def test_hidden_unit_weights_round_trip(self):
        wh = _weight_history(num_samples=3, sampling_interval=2, num_hidden=2)
        session, monitor = _make_session(weight_history=wh)
        session._emit_frame(4)  # sample index 2 — both units present
        metrics = monitor._trigger_callbacks.call_args.kwargs["metrics"]
        hu = metrics["weights"]["hidden_units"]
        assert len(hu) == 2
        for emitted, source in zip(hu, wh["hidden_units"]):
            # local_idx = sample_index(2) - first_sample_index of unit
            local_idx = 2 - source["first_sample_index"]
            arr = source["weights"][local_idx]
            assert emitted["weights"]["dtype"] == "float32"
            assert emitted["weights"]["shape"] == list(arr.shape)
            decoded = np.frombuffer(base64.b64decode(emitted["weights"]["data"]), dtype=np.float32).reshape(emitted["weights"]["shape"])
            np.testing.assert_array_equal(decoded, arr)
            # Bias is a plain float, not an envelope
            assert isinstance(emitted["bias"], float)
            assert emitted["bias"] == pytest.approx(source["bias"][local_idx])

    def test_units_not_yet_added_excluded(self):
        wh = _weight_history(num_samples=3, sampling_interval=2, num_hidden=2)
        # Unit 0 first_sample_index=1, unit 1 first_sample_index=2.
        # At sample 0 neither unit is present.
        session, monitor = _make_session(weight_history=wh)
        session._emit_frame(0)
        metrics = monitor._trigger_callbacks.call_args.kwargs["metrics"]
        assert metrics["weights"]["hidden_units"] == []

    def test_encode_tensor_handles_none_gracefully(self):
        # Defensive: if the cache somehow returns a payload with None
        # tensors (e.g. ragged storage in a future PR), the encoder
        # must not crash the playback thread.
        encoded = _ReplaySession._encode_tensor(None)
        assert encoded is None


# =============================================================================
# Empty weight history
# =============================================================================


class TestEmptyWeightHistory:
    """A weight_history dict with empty sample_indices behaves like V1."""

    def test_empty_sample_indices_treated_as_v1(self):
        wh = {
            "sampling_strategy": "adaptive",
            "sampling_interval": 2,
            "sample_indices": [],
            "output_weights": [],
            "output_bias": [],
            "hidden_units": [],
        }
        session, monitor = _make_session(weight_history=wh)
        session._emit_frame(0)
        metrics = monitor._trigger_callbacks.call_args.kwargs["metrics"]
        # Cache.available is False when sample_indices is empty, so the
        # emitter falls through the V1 branch — no is_sample_boundary
        # field at all.
        assert "is_sample_boundary" not in metrics
        assert "weights" not in metrics
