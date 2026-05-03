#!/usr/bin/env python
"""Unit tests for per-sample weight history persistence (CAN-015g g-1).

Covers:
- Round-trip of a synthetic ``network.weight_history`` payload.
- V1 backward-compat: snapshots without a ``history/weights/`` group
  load successfully and leave ``network.weight_history`` absent.
- Schema-version gate: a snapshot written by a future schema rejects
  on load without breaking the rest of the file.
- Length-mismatch validation in the writer.
- Size-regression assertion: a small synthetic snapshot with the
  weight history attached is no larger than 1.5× the same snapshot
  without it (bounds the worst case the parent design committed to).
"""

import os
import sys
import tempfile

import h5py
import numpy as np
import pytest

# Add parent directories for imports (matches sibling test files).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from snapshots.snapshot_serializer import CascadeHDF5Serializer

pytestmark = pytest.mark.unit


@pytest.fixture
def temp_file():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def serializer():
    return CascadeHDF5Serializer()


@pytest.fixture
def simple_network():
    config = CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=1,
        learning_rate=0.1,
        max_hidden_units=3,
        random_seed=42,
    )
    return CascadeCorrelationNetwork(config=config)


def _make_weight_history(num_samples=4, in_size=2, out_size=1, num_hidden=2, sampling_interval=10):
    """Build a synthetic weight_history payload.

    The output-layer width grows by one for each unit added so the
    round-trip exercises the variable-shape per-sample dataset path.
    """
    sample_indices = [i * sampling_interval for i in range(num_samples)]
    output_weights = []
    output_bias = []
    for i in range(num_samples):
        # Hidden units appear progressively: at sample i, min(i, num_hidden) units exist.
        hid_at_sample = min(i, num_hidden)
        ow = np.random.RandomState(i).randn(in_size + hid_at_sample, out_size).astype(np.float32)
        ob = np.random.RandomState(i + 100).randn(out_size).astype(np.float32)
        output_weights.append(ow)
        output_bias.append(ob)

    hidden_units = []
    for unit_idx in range(num_hidden):
        first_sample_index = (unit_idx + 1) * sampling_interval
        unit_weights = []
        unit_bias = []
        for j in range(num_samples - (unit_idx + 1)):
            uw = np.random.RandomState(unit_idx * 10 + j).randn(in_size + unit_idx).astype(np.float32)
            ub = float(np.random.RandomState(unit_idx * 10 + j + 1).randn())
            unit_weights.append(uw)
            unit_bias.append(ub)
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


# =============================================================================
# Round-trip
# =============================================================================


class TestWeightHistoryRoundTrip:
    """Verify save → load returns an equivalent payload."""

    def test_meta_attrs_persist(self, serializer, simple_network, temp_file):
        simple_network.history = {"train_loss": [1.0, 0.5], "value_loss": [], "train_accuracy": [], "value_accuracy": [], "hidden_units_added": []}
        simple_network.weight_history = _make_weight_history()
        assert serializer.save_network(simple_network, temp_file, include_training_state=True)

        with h5py.File(temp_file, "r") as f:
            assert "history/weights" in f
            meta = f["history/weights/meta"]
            assert int(meta.attrs["schema_version"]) == 2
            assert int(meta.attrs["sampling_interval"]) == 10
            # str attrs come back as numpy bytes — decode for comparison.
            sampling_strategy = meta.attrs["sampling_strategy"]
            if isinstance(sampling_strategy, (bytes, np.bytes_)):
                sampling_strategy = sampling_strategy.decode("utf-8")
            assert sampling_strategy == "adaptive"

    def test_sample_indices_round_trip(self, serializer, simple_network, temp_file):
        wh = _make_weight_history(num_samples=5, sampling_interval=20)
        simple_network.history = {"train_loss": [], "value_loss": [], "train_accuracy": [], "value_accuracy": [], "hidden_units_added": []}
        simple_network.weight_history = wh
        serializer.save_network(simple_network, temp_file, include_training_state=True)

        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        assert hasattr(loaded, "weight_history")
        assert loaded.weight_history is not None
        assert loaded.weight_history["sample_indices"] == [0, 20, 40, 60, 80]
        assert loaded.weight_history["sampling_interval"] == 20
        assert loaded.weight_history["sampling_strategy"] == "adaptive"

    def test_output_tensors_round_trip(self, serializer, simple_network, temp_file):
        wh = _make_weight_history(num_samples=3, in_size=2, out_size=1, num_hidden=2)
        simple_network.history = {"train_loss": [], "value_loss": [], "train_accuracy": [], "value_accuracy": [], "hidden_units_added": []}
        simple_network.weight_history = wh
        serializer.save_network(simple_network, temp_file, include_training_state=True)

        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        for orig, got in zip(wh["output_weights"], loaded.weight_history["output_weights"]):
            np.testing.assert_array_equal(orig, got)
        for orig, got in zip(wh["output_bias"], loaded.weight_history["output_bias"]):
            np.testing.assert_array_equal(orig, got)

    def test_hidden_unit_tensors_round_trip(self, serializer, simple_network, temp_file):
        wh = _make_weight_history(num_samples=4, num_hidden=2)
        simple_network.history = {"train_loss": [], "value_loss": [], "train_accuracy": [], "value_accuracy": [], "hidden_units_added": []}
        simple_network.weight_history = wh
        serializer.save_network(simple_network, temp_file, include_training_state=True)

        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        assert len(loaded.weight_history["hidden_units"]) == 2
        for orig, got in zip(wh["hidden_units"], loaded.weight_history["hidden_units"]):
            assert orig["first_sample_index"] == got["first_sample_index"]
            assert orig["activation"] == got["activation"]
            for ow, gw in zip(orig["weights"], got["weights"]):
                np.testing.assert_array_equal(ow, gw)
            for ob, gb in zip(orig["bias"], got["bias"]):
                # bias entries come back as 0-d arrays — both should be equal scalars
                assert float(ob) == pytest.approx(float(gb))


# =============================================================================
# Backward compat
# =============================================================================


class TestV1BackwardCompat:
    """Snapshots without ``history/weights/`` keep loading unchanged."""

    def test_v1_snapshot_loads_without_weight_history(self, serializer, simple_network, temp_file):
        simple_network.history = {"train_loss": [1.0, 0.9], "value_loss": [], "train_accuracy": [], "value_accuracy": [], "hidden_units_added": []}
        # Note: no weight_history attribute. Mirrors a pre-g-1 network.
        serializer.save_network(simple_network, temp_file, include_training_state=True)

        with h5py.File(temp_file, "r") as f:
            assert "history" in f
            assert "history/weights" not in f

        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        # Loader must not fabricate a weight_history when the group is absent.
        assert getattr(loaded, "weight_history", None) is None or loaded.weight_history == {}

    def test_empty_weight_history_writes_meta_only(self, serializer, simple_network, temp_file):
        simple_network.history = {"train_loss": [], "value_loss": [], "train_accuracy": [], "value_accuracy": [], "hidden_units_added": []}
        simple_network.weight_history = {
            "sampling_strategy": "every_n",
            "sampling_interval": 50,
            "sample_indices": [],
            "output_weights": [],
            "output_bias": [],
            "hidden_units": [],
        }
        serializer.save_network(simple_network, temp_file, include_training_state=True)

        with h5py.File(temp_file, "r") as f:
            assert "history/weights/meta" in f
            assert "history/weights/sample_indices" not in f

        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        assert loaded.weight_history["sample_indices"] == []
        assert loaded.weight_history["sampling_interval"] == 50


# =============================================================================
# Validation
# =============================================================================


class TestWeightHistoryValidation:
    def test_length_mismatch_output_weights_rejected(self, serializer, simple_network, temp_file):
        wh = _make_weight_history(num_samples=3)
        wh["output_weights"] = wh["output_weights"][:2]  # mismatch with sample_indices length
        simple_network.history = {"train_loss": [], "value_loss": [], "train_accuracy": [], "value_accuracy": [], "hidden_units_added": []}
        simple_network.weight_history = wh
        # save_network catches and logs but returns False; assert that.
        assert serializer.save_network(simple_network, temp_file, include_training_state=True) is False

    def test_length_mismatch_hidden_unit_rejected(self, serializer, simple_network, temp_file):
        wh = _make_weight_history(num_samples=4, num_hidden=2)
        wh["hidden_units"][0]["bias"] = wh["hidden_units"][0]["bias"][:1]
        simple_network.history = {"train_loss": [], "value_loss": [], "train_accuracy": [], "value_accuracy": [], "hidden_units_added": []}
        simple_network.weight_history = wh
        assert serializer.save_network(simple_network, temp_file, include_training_state=True) is False

    def test_unsupported_schema_version_degrades_gracefully(self, serializer, simple_network, temp_file):
        # Write a V1 snapshot, then synthesize a forward-incompatible
        # weights/meta group with schema_version=99 and confirm the
        # loader degrades to V1 behaviour rather than raising.
        simple_network.history = {"train_loss": [1.0], "value_loss": [], "train_accuracy": [], "value_accuracy": [], "hidden_units_added": []}
        serializer.save_network(simple_network, temp_file, include_training_state=True)
        with h5py.File(temp_file, "a") as f:
            weights_group = f["history"].create_group("weights")
            meta_group = weights_group.create_group("meta")
            meta_group.attrs["schema_version"] = 99
            meta_group.attrs["sampling_interval"] = 50
            meta_group.attrs["num_samples"] = 0
            meta_group.attrs["sampling_strategy"] = "adaptive"

        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        # Loader caught the unsupported version, degraded to None.
        assert getattr(loaded, "weight_history", None) is None


# =============================================================================
# Size regression (parent design §10.1 risk)
# =============================================================================


class TestWeightHistorySize:
    """Bound the file-size impact at the toy scale.

    The parent design committed to a 1.5× ceiling vs. the V1 snapshot
    for adaptive subsampling at default ``N``. This test exercises a
    small synthetic 100-sample run and asserts the ratio.
    """

    def test_weight_history_under_size_ceiling(self, serializer, simple_network, tmp_path):
        # V1 baseline: same metric arrays, no weight history.
        simple_network.history = {
            "train_loss": np.linspace(1.0, 0.1, 100).tolist(),
            "value_loss": np.linspace(1.0, 0.2, 100).tolist(),
            "train_accuracy": np.linspace(0.5, 0.95, 100).tolist(),
            "value_accuracy": np.linspace(0.5, 0.93, 100).tolist(),
            "hidden_units_added": [],
        }
        v1_path = str(tmp_path / "v1.h5")
        assert serializer.save_network(simple_network, v1_path, include_training_state=True)
        v1_size = os.path.getsize(v1_path)

        # V2 with adaptive subsampling at N=50 → 2 samples for a 100-epoch run.
        # Tiny network (in=2, out=1, hidden=3) means each sample is a few hundred bytes.
        simple_network.weight_history = _make_weight_history(num_samples=2, in_size=2, out_size=1, num_hidden=2, sampling_interval=50)
        v2_path = str(tmp_path / "v2.h5")
        assert serializer.save_network(simple_network, v2_path, include_training_state=True)
        v2_size = os.path.getsize(v2_path)

        ratio = v2_size / v1_size
        # Toy scale; the absolute byte difference is small but the ratio
        # must stay under the design's 1.5× ceiling.
        assert ratio <= 1.5, f"weight history blew V1 size by {ratio:.2f}× (ceiling 1.5×); v1={v1_size}, v2={v2_size}"
