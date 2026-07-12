#!/usr/bin/env python
"""
Unit tests for snapshot_serializer.py to improve test coverage.

Tests focus on:
- save_object() method
- Error handling paths
- Edge cases in serialization
- Training data serialization
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

# Add parent directories for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from snapshots.snapshot_serializer import CascadeHDF5Serializer

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        filepath = f.name
    yield filepath
    if os.path.exists(filepath):
        os.unlink(filepath)


@pytest.fixture
def serializer():
    """Create a serializer instance."""
    return CascadeHDF5Serializer()


@pytest.fixture
def simple_network():
    """Create a simple network for testing."""
    config = CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=1,
        learning_rate=0.1,
        max_hidden_units=3,
        random_seed=42,
    )
    return CascadeCorrelationNetwork(config=config)


class TestSerializerInit:
    """Tests for serializer initialization."""

    def test_serializer_creates_with_defaults(self):
        """Test that serializer initializes with default values."""
        serializer = CascadeHDF5Serializer()
        assert serializer.version == "2.0.0"
        assert serializer.format_version == "2"
        assert serializer.format_name == "juniper.cascor"

    def test_serializer_has_logger(self):
        """Test that serializer has a logger."""
        serializer = CascadeHDF5Serializer()
        assert serializer.logger is not None


class TestSaveObject:
    """Tests for save_object() method."""

    def test_save_object_creates_file(self, serializer, simple_network, temp_file):
        """Test that save_object creates an HDF5 file."""
        result = serializer.save_object(simple_network, temp_file)
        assert result is True
        assert os.path.exists(temp_file)

    def test_save_object_with_compression(self, serializer, simple_network, temp_file):
        """Test save_object with different compression settings."""
        result = serializer.save_object(simple_network, temp_file, compression="gzip", compression_opts=9)
        assert result is True
        assert os.path.exists(temp_file)

    def test_save_object_creates_parent_directories(self, serializer, simple_network):
        """Test that save_object creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "nested", "dir", "file.h5")
            result = serializer.save_object(simple_network, nested_path)
            assert result is True
            assert os.path.exists(nested_path)


class TestSaveNetwork:
    """Tests for save_network() method."""

    def test_save_network_basic(self, serializer, simple_network, temp_file):
        """Test basic network saving."""
        result = serializer.save_network(simple_network, temp_file)
        assert result is True
        assert os.path.exists(temp_file)

    def test_save_network_with_training_state(self, serializer, simple_network, temp_file):
        """Test saving network with training state."""
        result = serializer.save_network(simple_network, temp_file, include_training_state=True)
        assert result is True

    def test_save_network_with_training_data(self, serializer, simple_network, temp_file):
        """Test saving network with training data."""
        # Set up some training data on the network
        simple_network._training_data_x = torch.randn(10, 2)
        simple_network._training_data_y = torch.randn(10, 1)

        result = serializer.save_network(
            simple_network,
            temp_file,
            include_training_state=True,
            include_training_data=True,
        )
        assert result is True


class TestLoadNetwork:
    """Tests for load_network() method."""

    def test_load_network_roundtrip(self, serializer, simple_network, temp_file):
        """Test saving and loading a network."""
        # Save
        serializer.save_network(simple_network, temp_file)

        # Load
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert loaded is not None
        assert loaded.input_size == simple_network.input_size
        assert loaded.output_size == simple_network.output_size

    def test_load_network_preserves_uuid(self, serializer, simple_network, temp_file):
        """Test that loading preserves UUID."""
        original_uuid = simple_network.get_uuid()
        serializer.save_network(simple_network, temp_file)

        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert str(loaded.get_uuid()) == str(original_uuid)

    def test_load_network_preserves_weights(self, serializer, simple_network, temp_file):
        """Test that loading preserves output weights."""
        original_weights = simple_network.output_weights.clone()
        serializer.save_network(simple_network, temp_file)

        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert torch.allclose(loaded.output_weights, original_weights)


class TestVerifyFile:
    """Tests for verify_saved_network() method."""

    def test_verify_valid_file(self, serializer, simple_network, temp_file):
        """Test verification of a valid file."""
        serializer.save_network(simple_network, temp_file)
        result = serializer.verify_saved_network(temp_file)
        assert result.get("valid") is True

    def test_verify_nonexistent_file(self, serializer):
        """Test verification of a non-existent file."""
        result = serializer.verify_saved_network("/nonexistent/path/file.h5")
        assert result.get("valid") is False
        assert "error" in result

    def test_verify_returns_format_info(self, serializer, simple_network, temp_file):
        """Test that verification returns format information."""
        serializer.save_network(simple_network, temp_file)
        result = serializer.verify_saved_network(temp_file)
        assert "format" in result
        assert "format_version" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_save_to_invalid_path(self, serializer, simple_network):
        """Saving to an invalid path raises SnapshotSaveError (C1 / I-3).

        Pre-C1 the failure was swallowed into ``False`` and a failed save was
        indistinguishable from a missing network at the API route."""
        from snapshots.snapshot_errors import SnapshotSaveError

        with pytest.raises(SnapshotSaveError) as excinfo:
            serializer.save_network(simple_network, "/nonexistent/readonly/path.h5")
        assert excinfo.value.__cause__ is not None

    def test_load_from_invalid_path(self, serializer):
        """Test loading from invalid path returns None."""
        result = serializer.load_network("/nonexistent/file.h5", CascadeCorrelationNetwork)
        assert result is None

    def test_save_network_with_hidden_units(self, serializer, temp_file):
        """Test saving network with hidden units."""
        config = CascadeCorrelationConfig.create_simple_config(
            input_size=3,
            output_size=2,
            learning_rate=0.1,
            max_hidden_units=5,
            random_seed=123,
        )
        network = CascadeCorrelationNetwork(config=config)

        # Add a mock hidden unit
        hidden_unit = {
            "weights": torch.randn(3),
            "bias": torch.tensor([0.1]),
            "activation_fn": torch.tanh,
            "correlation": 0.5,
        }
        network.hidden_units.append(hidden_unit)

        result = serializer.save_network(network, temp_file)
        assert result is True

        # Verify it can be loaded back
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert len(loaded.hidden_units) == 1


class TestRandomStatePreservation:
    """Tests for random state preservation during serialization."""

    def test_random_seed_preserved(self, serializer, simple_network, temp_file):
        """Test that random seed is preserved."""
        serializer.save_network(simple_network, temp_file, include_training_state=True)
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert loaded.random_seed == simple_network.random_seed

    def test_deterministic_after_load(self, serializer, simple_network, temp_file):
        """Test that network is deterministic after load with same inputs."""
        x = torch.randn(5, 2)

        # Get output before save
        output_before = simple_network.forward(x)

        # Save and load
        serializer.save_network(simple_network, temp_file, include_training_state=True)
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)

        # Get output after load
        output_after = loaded.forward(x)

        assert torch.allclose(output_before, output_after)


class TestConfigSerialization:
    """Tests for configuration serialization."""

    def test_config_roundtrip(self, serializer, temp_file):
        """Test that configuration survives roundtrip."""
        config = CascadeCorrelationConfig.create_simple_config(
            input_size=5,
            output_size=3,
            learning_rate=0.05,
            max_hidden_units=10,
            random_seed=999,
            activation_function_name="sigmoid",
        )
        network = CascadeCorrelationNetwork(config=config)

        serializer.save_network(network, temp_file)
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)

        assert loaded.input_size == 5
        assert loaded.output_size == 3
        assert loaded.learning_rate == 0.05
        assert loaded.max_hidden_units == 10


class TestPhase6EA5SnapshotRoundTrip:
    """Phase 6E Sprint A-5 (CAN-014): runtime-tunable training params
    must survive snapshot save → load.

    Pre-A-5, the serializer persisted ``learning_rate`` /
    ``candidate_learning_rate`` / ``max_hidden_units`` / etc., but the
    fields wired through the API surface in PRs #157 / #158 / #162
    (``output_epochs``, ``optimizer_type``, ``activation_function_name``)
    plus a handful of older tunables (``epochs_max``, ``max_iterations``,
    ``candidate_patience``, ``candidate_epochs``,
    ``convergence_threshold``, ``candidate_convergence_threshold``,
    ``init_output_weights``) were silently dropped on load — the
    rehydrated network reverted to construction-time defaults from
    ``CascadeCorrelationConfig``.

    These tests pin the contract: every field listed in
    ``update_params``' whitelist round-trips cleanly through
    ``save_network`` → ``load_network``.
    """

    def test_epochs_max_roundtrip(self, serializer, temp_file):
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, max_hidden_units=3, epochs_max=777)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, temp_file)
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert loaded.epochs_max == 777

    def test_max_iterations_roundtrip(self, serializer, temp_file):
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, max_hidden_units=3, max_iterations=42)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, temp_file)
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert loaded.max_iterations == 42

    def test_output_epochs_roundtrip(self, serializer, temp_file):
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, max_hidden_units=3, output_epochs=99)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, temp_file)
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert loaded.output_epochs == 99

    def test_candidate_patience_roundtrip(self, serializer, temp_file):
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, max_hidden_units=3, candidate_patience=17)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, temp_file)
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert loaded.candidate_patience == 17

    def test_candidate_epochs_roundtrip(self, serializer, temp_file):
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, max_hidden_units=3, candidate_epochs=33)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, temp_file)
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert loaded.candidate_epochs == 33

    def test_convergence_threshold_roundtrip(self, serializer, temp_file):
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, max_hidden_units=3, convergence_threshold=0.0042)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, temp_file)
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert loaded.convergence_threshold == pytest.approx(0.0042)

    def test_candidate_convergence_threshold_roundtrip(self, serializer, temp_file):
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, max_hidden_units=3, candidate_convergence_threshold=0.0033)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, temp_file)
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert loaded.candidate_convergence_threshold == pytest.approx(0.0033)

    def test_init_output_weights_roundtrip(self, serializer, temp_file):
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, max_hidden_units=3, init_output_weights="random")
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, temp_file)
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert loaded.init_output_weights == "random"

    def test_full_tunable_set_roundtrip(self, serializer, temp_file):
        """Every newly-persisted field at once — guards against any single
        field shadowing another (e.g. a name collision between an arch
        attr and a config attr) during load."""
        config = CascadeCorrelationConfig.create_simple_config(
            input_size=2,
            output_size=1,
            max_hidden_units=5,
            epochs_max=1234,
            max_iterations=78,
            output_epochs=66,
            candidate_patience=21,
            candidate_epochs=55,
            convergence_threshold=0.0011,
            candidate_convergence_threshold=0.0012,
            init_output_weights="random",
        )
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, temp_file)
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert loaded.epochs_max == 1234
        assert loaded.max_iterations == 78
        assert loaded.output_epochs == 66
        assert loaded.candidate_patience == 21
        assert loaded.candidate_epochs == 55
        assert loaded.convergence_threshold == pytest.approx(0.0011)
        assert loaded.candidate_convergence_threshold == pytest.approx(0.0012)
        assert loaded.init_output_weights == "random"

    def test_load_tolerates_legacy_snapshot_missing_new_fields(self, serializer, simple_network, temp_file):
        """A snapshot that pre-dates the A-5 expansion (no new attrs in
        the config group) must still load — the missing fields fall
        back to whatever the freshly-constructed network already has,
        rather than crashing on a missing-key access. Simulate by
        deleting the new attrs after save."""
        import h5py

        serializer.save_network(simple_network, temp_file)
        # Strip the A-5-added attrs to simulate an older snapshot.
        with h5py.File(temp_file, "r+") as f:
            for key in (
                "epochs_max",
                "max_iterations",
                "output_epochs",
                "candidate_patience",
                "candidate_epochs",
                "convergence_threshold",
                "candidate_convergence_threshold",
                "init_output_weights",
            ):
                if key in f["config"].attrs:
                    del f["config"].attrs[key]
        # Load should still succeed — missing fields just don't get
        # re-applied (the constructor's defaults stand).
        loaded = serializer.load_network(temp_file, CascadeCorrelationNetwork)
        assert loaded is not None
        # Non-A-5 fields still round-trip (the rest of the config block
        # was untouched).
        assert loaded.learning_rate == simple_network.learning_rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
