#!/usr/bin/env python
"""
Extended unit tests for snapshot_serializer.py to improve test coverage.

Focuses on uncovered lines: 127-128, 182, 329-330, 441-454, 458-459, 470-471, 479, 490, 499-503, etc.
"""

import json
import os
import sys
from unittest.mock import MagicMock, PropertyMock, patch

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from snapshots.snapshot_serializer import CascadeHDF5Serializer

pytestmark = pytest.mark.unit


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


class TestSaveObjectErrorHandling:
    """Tests for save_object error handling (lines 127-128)."""

    def test_save_object_with_invalid_object_returns_false(self, serializer, tmp_path):
        """Test save_object with an object that raises exception during serialization."""
        filepath = tmp_path / "test.h5"

        mock_obj = MagicMock()
        mock_obj.get_uuid.side_effect = Exception("UUID error")

        result = serializer.save_object(mock_obj, str(filepath))
        assert result is False

    def test_save_object_to_readonly_directory(self, serializer, simple_network):
        """Test save_object to read-only path returns False (exercises exception handler)."""
        result = serializer.save_object(simple_network, "/nonexistent/readonly/path.h5")
        assert result is False


class TestVerifyInvalidFormat:
    """Tests for verify_saved_network with invalid format (line 182)."""

    def test_verify_file_with_invalid_format_attribute(self, serializer, tmp_path):
        """Test verification fails when format attribute is invalid."""
        filepath = tmp_path / "invalid_format.h5"

        with h5py.File(filepath, "w") as f:
            f.attrs["format"] = "invalid_format_name"
            f.attrs["format_version"] = "1"

        result = serializer.verify_saved_network(str(filepath))
        assert result.get("valid") is False
        assert "Invalid format" in result.get("error", "")


class TestSaveParametersOptimizerError:
    """Tests for _save_parameters optimizer error handling (lines 329-330)."""

    def test_save_parameters_with_broken_optimizer(self, serializer, tmp_path):
        """Test that broken optimizer state_dict is handled gracefully."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)

        mock_optimizer = MagicMock()
        mock_optimizer.state_dict.side_effect = RuntimeError("Optimizer error")
        network.output_optimizer = mock_optimizer

        result = serializer.save_network(network, str(filepath))
        assert result is True
        assert filepath.exists()


class TestSaveRandomStateCUDA:
    """Tests for CUDA random state saving (lines 441-454, 458-459)."""

    def test_save_random_state_cuda_available_with_error(self, serializer, tmp_path, simple_network):
        """Test CUDA state saving when CUDA is 'available' but get_rng_state_all fails."""
        filepath = tmp_path / "test.h5"

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_rng_state_all", side_effect=RuntimeError("CUDA error")):
                result = serializer.save_network(simple_network, str(filepath))

        assert result is True
        assert filepath.exists()

    def test_save_random_state_with_pickle_error(self, serializer, tmp_path):
        """Test random state saving when pickle.dumps fails."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)

        with patch("pickle.dumps", side_effect=Exception("Pickle error")):
            result = serializer.save_network(network, str(filepath))

        assert result is True


class TestSaveMultiprocessingState:
    """Tests for multiprocessing state saving (lines 470-471, 479, 490, 499-503)."""

    def test_save_mp_state_as_server(self, serializer, tmp_path):
        """Test saving multiprocessing state when network is server role (line 479)."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)
        network.candidate_training_manager = MagicMock()

        result = serializer.save_network(network, str(filepath))
        assert result is True

    def test_save_mp_state_as_client(self, serializer, tmp_path):
        """Test saving multiprocessing state when network is client role."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)
        network.candidate_training_queue_address = ("127.0.0.1", 5000)

        result = serializer.save_network(network, str(filepath))
        assert result is True

    def test_save_mp_state_without_context(self, serializer, tmp_path):
        """Test saving MP state when candidate_training_context is missing (line 490)."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)

        if hasattr(network, "candidate_training_context"):
            delattr(network, "candidate_training_context")

        result = serializer.save_network(network, str(filepath))
        assert result is True

    def test_save_mp_state_with_non_tuple_address(self, serializer, tmp_path):
        """Test saving MP state when address is not a proper tuple (lines 499-503)."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)
        network.candidate_training_queue_address = "invalid_address"

        result = serializer.save_network(network, str(filepath))
        assert result is True

    def test_save_mp_state_with_authkey_bytes(self, serializer, tmp_path):
        """Test saving MP state with bytes authkey."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)
        network.candidate_training_queue_authkey = b"secret_key_123"

        result = serializer.save_network(network, str(filepath))
        assert result is True

    def test_save_mp_state_with_string_authkey(self, serializer, tmp_path):
        """Test saving MP state with string authkey."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)
        network.candidate_training_queue_authkey = "string_key"

        result = serializer.save_network(network, str(filepath))
        assert result is True

    def test_save_mp_state_exception_handling(self, serializer, tmp_path):
        """Test MP state save exception handling (lines 470-471)."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)

        mock_manager = MagicMock()
        type(mock_manager).get_start_method = PropertyMock(side_effect=Exception("MP error"))
        network.candidate_training_context = mock_manager

        result = serializer.save_network(network, str(filepath))
        assert result is True


class TestTrainingHistorySave:
    """Tests for training history save with hidden units data."""

    def test_save_training_history_with_hidden_units(self, serializer, tmp_path):
        """Test saving training history that includes hidden units data."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)

        network.history = {
            "train_loss": [0.5, 0.4, 0.3],
            "value_loss": [0.6, 0.5, 0.4],
            "train_accuracy": [0.7, 0.8, 0.9],
            "value_accuracy": [0.65, 0.75, 0.85],
            "hidden_units_added": [
                {"correlation": 0.8, "weights": np.array([0.1, 0.2]), "bias": np.array([0.05])},
                {"correlation": 0.9},
            ],
        }

        result = serializer.save_network(network, str(filepath), include_training_state=True)
        assert result is True


class TestTrainingDataSave:
    """Tests for training data save functionality."""

    def test_save_training_data_with_dict_format(self, serializer, tmp_path):
        """Test saving training data stored as dictionary."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)

        network._training_data = {
            "x_train": torch.randn(10, 2),
            "y_train": torch.randn(10, 1),
            "x_val": np.random.randn(5, 2),
        }

        result = serializer.save_network(network, str(filepath), include_training_data=True)
        assert result is True


class TestValidateFormat:
    """Tests for format validation edge cases."""

    def test_validate_format_incompatible_version(self, serializer, tmp_path):
        """Test validation fails with incompatible major version."""
        filepath = tmp_path / "future_version.h5"

        with h5py.File(filepath, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "99.0.0"
            f.create_group("meta")
            f.create_group("config")
            f.create_group("params")
            f.create_group("arch")
            f.create_group("random")

        result = serializer.verify_saved_network(str(filepath))
        assert result.get("valid") is False

    def test_validate_format_missing_output_weights(self, serializer, tmp_path):
        """Test validation fails when output weights are missing."""
        filepath = tmp_path / "missing_weights.h5"

        with h5py.File(filepath, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "2"
            f.create_group("meta")
            f.create_group("config")
            params = f.create_group("params")
            params.create_group("output_layer")
            f.create_group("arch")
            f.create_group("random")

        result = serializer.verify_saved_network(str(filepath))
        assert result.get("valid") is False

    def test_validate_format_hidden_units_count_mismatch(self, serializer, tmp_path):
        """Test validation fails when hidden unit count doesn't match."""
        filepath = tmp_path / "mismatch_units.h5"

        with h5py.File(filepath, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "2"
            f.create_group("meta")
            f.create_group("config")
            params = f.create_group("params")
            output_layer = params.create_group("output_layer")
            output_layer.create_dataset("weights", data=np.zeros((2, 1)))
            output_layer.create_dataset("bias", data=np.zeros(1))
            f.create_group("arch")
            f.create_group("random")
            hidden = f.create_group("hidden_units")
            hidden.attrs["num_units"] = 5
            hidden.create_group("unit_0")

        result = serializer.verify_saved_network(str(filepath))
        assert result.get("valid") is False


class TestLoadNetworkEdgeCases:
    """Tests for load_network edge cases."""

    def test_load_network_with_legacy_format(self, serializer, tmp_path):
        """Test loading file with legacy format name."""
        filepath = tmp_path / "legacy.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            f.attrs["format"] = "cascor_hdf5_v1"

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestValidateShapes:
    """Tests for shape validation."""

    def test_validate_shapes_mismatch_output_weights(self, serializer, tmp_path):
        """Test shape validation detects output weights mismatch."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)

        loaded.output_weights = torch.randn(10, 10)
        result = serializer._validate_shapes(loaded)
        assert result is False

    def test_validate_shapes_mismatch_output_bias(self, serializer, tmp_path):
        """Test shape validation detects output bias mismatch."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)

        loaded.output_bias = torch.randn(5)
        result = serializer._validate_shapes(loaded)
        assert result is False

    def test_validate_shapes_hidden_unit_weights_mismatch(self, serializer, tmp_path):
        """Test shape validation detects hidden unit weight shape mismatch."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)

        network.hidden_units.append(
            {
                "weights": torch.randn(2),
                "bias": torch.tensor([0.1]),
                "activation_fn": torch.tanh,
            }
        )
        network.output_weights = torch.randn(3, 1)

        serializer.save_network(network, str(filepath))
        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)

        loaded.hidden_units[0]["weights"] = torch.randn(10)
        result = serializer._validate_shapes(loaded)
        assert result is False

    def test_validate_shapes_hidden_unit_bias_mismatch(self, serializer, tmp_path):
        """Test shape validation detects hidden unit bias shape mismatch."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)

        network.hidden_units.append(
            {
                "weights": torch.randn(2),
                "bias": torch.tensor([0.1]),
                "activation_fn": torch.tanh,
            }
        )
        network.output_weights = torch.randn(3, 1)

        serializer.save_network(network, str(filepath))
        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)

        loaded.hidden_units[0]["bias"] = torch.randn(5)
        result = serializer._validate_shapes(loaded)
        assert result is False


class TestConfigToDict:
    """Tests for _config_to_dict conversion."""

    def test_config_to_dict_with_path_attribute(self, serializer):
        """Test config conversion with pathlib.Path attribute."""
        import pathlib

        class ConfigWithPath:
            some_path = pathlib.Path("/some/path")
            regular_attr = "value"

        config = ConfigWithPath()
        result = serializer._config_to_dict(config)
        assert result["some_path"] == "/some/path"
        assert result["regular_attr"] == "value"

    def test_config_to_dict_skips_callables(self, serializer):
        """Test that callable attributes are skipped."""

        class ConfigWithCallable:
            regular_attr = "value"

            def callable_method(self):
                pass

        config = ConfigWithCallable()
        result = serializer._config_to_dict(config)
        assert "callable_method" not in result
        assert result["regular_attr"] == "value"

    def test_config_to_dict_handles_list_attribute(self, serializer):
        """Test that list attributes with primitives are included."""

        class ConfigWithList:
            list_attr = [1, 2, 3]
            tuple_attr = ("a", "b")
            dict_attr = {"key": "value"}

        config = ConfigWithList()
        result = serializer._config_to_dict(config)
        assert result["list_attr"] == [1, 2, 3]
        assert result["tuple_attr"] == ["a", "b"]
        assert result["dict_attr"] == {"key": "value"}


class TestLoadOptimizerState:
    """Tests for optimizer state loading."""

    def test_load_network_without_optimizer(self, serializer, tmp_path):
        """Test loading network that was saved without optimizer state."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)
        network.output_optimizer = None

        serializer.save_network(network, str(filepath))
        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestRestoreMultiprocessingState:
    """Tests for multiprocessing state restoration."""

    def test_restore_mp_state_with_invalid_authkey_hex(self, serializer, tmp_path):
        """Test restoring MP state when authkey_hex is not valid hex."""
        filepath = tmp_path / "test.h5"

        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1, learning_rate=0.1, max_hidden_units=3, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            mp_group = f["mp"]
            if "authkey_hex" in mp_group.attrs:
                del mp_group.attrs["authkey_hex"]
            mp_group.attrs["authkey_hex"] = np.bytes_("not_valid_hex_zzz")

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestLogExceptionStacktrace:
    """Tests for _log_exception_stacktrace helper."""

    def test_log_exception_returns_arg2(self, serializer):
        """Test that _log_exception_stacktrace returns arg2."""
        result = serializer._log_exception_stacktrace("Error: ", ValueError("test"), "return_value")
        assert result == "return_value"

    def test_log_exception_returns_none(self, serializer):
        """Test that _log_exception_stacktrace can return None."""
        result = serializer._log_exception_stacktrace("Error: ", ValueError("test"), None)
        assert result is None

    def test_log_exception_returns_false(self, serializer):
        """Test that _log_exception_stacktrace can return False."""
        result = serializer._log_exception_stacktrace("Error: ", ValueError("test"), False)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
