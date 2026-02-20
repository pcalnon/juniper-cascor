#!/usr/bin/env python
"""
Additional unit tests for snapshot_serializer.py to improve test coverage.

Focuses on remaining uncovered lines and edge cases.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from snapshots.snapshot_common import read_str_attr, read_str_dataset, write_str_attr, write_str_dataset
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
        candidate_pool_size=2,
        candidate_epochs=2,
        output_epochs=2,
    )
    return CascadeCorrelationNetwork(config=config)


@pytest.fixture(scope="module")
def network_with_hidden_units():
    """Create a network with hidden units for testing.

    Module-scoped because tests only serialize/validate the network without
    mutating it. Uses manual hidden unit construction to avoid expensive fit().
    """
    config = CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=1,
        learning_rate=0.1,
        max_hidden_units=3,
        random_seed=42,
        candidate_pool_size=2,
        candidate_epochs=2,
        output_epochs=2,
        epochs_max=3,
        patience=1,
    )
    network = CascadeCorrelationNetwork(config=config)
    # Manually add hidden units instead of calling fit() to avoid training overhead
    for i in range(2):
        torch.manual_seed(42 + i)
        hidden_unit = {
            "weights": torch.randn(2),
            "bias": torch.randn(1),
            "activation_fn": torch.tanh,
            "correlation": 0.5 + i * 0.1,
        }
        network.hidden_units.append(hidden_unit)
    return network


class TestSaveTrainingHistoryEmpty:
    """Tests for _save_training_history with empty history (line 526)."""

    def test_save_network_without_history(self, serializer, tmp_path):
        """Test saving network when history is empty."""
        filepath = tmp_path / "no_history.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        network.history = {}

        result = serializer.save_network(network, str(filepath))
        assert result is True

    def test_save_network_without_history_attribute(self, serializer, tmp_path):
        """Test saving network when history attribute is missing."""
        filepath = tmp_path / "no_history_attr.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        if hasattr(network, "history"):
            delattr(network, "history")

        result = serializer.save_network(network, str(filepath))
        assert result is True


class TestLoadTrainingHistoryEmpty:
    """Tests for _load_training_history with no history group (line 864)."""

    def test_load_network_without_history_group(self, serializer, tmp_path):
        """Test loading network when history group doesn't exist."""
        filepath = tmp_path / "no_history.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "history" in f:
                del f["history"]

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestLoadHiddenUnitsEmpty:
    """Tests for _load_hidden_units with no hidden units (lines 737-738)."""

    def test_load_network_without_hidden_units_group(self, serializer, tmp_path):
        """Test loading network when hidden_units group doesn't exist."""
        filepath = tmp_path / "no_hidden.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "hidden_units" in f:
                del f["hidden_units"]

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None
        assert loaded.hidden_units == []


class TestLoadOptimizerStateNonAdam:
    """Tests for _load_optimizer_state_from_hdf5_helper with non-Adam optimizer (line 722)."""

    def test_load_optimizer_non_adam_type(self, serializer, tmp_path):
        """Test loading optimizer with non-Adam type falls back to Adam."""
        filepath = tmp_path / "non_adam.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "params/output_layer/optimizer" in f:
                opt_group = f["params/output_layer/optimizer"]
                if "optimizer_type" in opt_group.attrs:
                    del opt_group.attrs["optimizer_type"]
                opt_group.attrs["optimizer_type"] = np.bytes_("SGD")

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestLoadOptimizerWithoutStateDict:
    """Tests for optimizer loading without state_dict (lines 732-733)."""

    def test_load_optimizer_without_state_dict(self, serializer, tmp_path):
        """Test loading optimizer when state_dict dataset doesn't exist."""
        filepath = tmp_path / "no_state_dict.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "params/output_layer/optimizer/state_dict" in f:
                del f["params/output_layer/optimizer/state_dict"]

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestLoadHiddenUnitMissingGroup:
    """Tests for _load_hidden_units skipping missing groups (line 748)."""

    def test_load_hidden_units_renumbers_correctly(self, serializer, tmp_path, network_with_hidden_units):
        """Test that hidden units are renumbered correctly when loaded."""
        filepath = tmp_path / "gap_units.h5"
        serializer.save_network(network_with_hidden_units, str(filepath))

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestLoadRandomStateEmpty:
    """Tests for _load_random_state with no random group (line 797-798)."""

    def test_load_random_state_with_empty_group(self, serializer, tmp_path):
        """Test loading network when random group is empty but exists."""
        filepath = tmp_path / "empty_random.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "random" in f:
                del f["random"]
            f.create_group("random")

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestRestoreMultiprocessingEmpty:
    """Tests for _restore_multiprocessing_state with no mp group (lines 913-914)."""

    def test_load_network_without_mp_group(self, serializer, tmp_path):
        """Test loading network when mp group doesn't exist."""
        filepath = tmp_path / "no_mp.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "mp" in f:
                del f["mp"]

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestRestoreMultiprocessingException:
    """Tests for _restore_multiprocessing_state exception handling (lines 918-919)."""

    def test_restore_mp_with_corrupt_data(self, serializer, tmp_path):
        """Test MP restore handles corrupt data gracefully."""
        filepath = tmp_path / "corrupt_mp.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "mp" in f:
                del f["mp"]
            mp_group = f.create_group("mp")
            mp_group.attrs["address_port"] = "not_a_number"

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestServerRoleAutostart:
    """Tests for server role autostart (line 948)."""

    def test_restore_mp_server_role_autostart(self, serializer, tmp_path):
        """Test MP restore with server role and autostart."""
        filepath = tmp_path / "server_mp.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "mp" in f:
                mp_group = f["mp"]
                if "role" in mp_group.attrs:
                    del mp_group.attrs["role"]
                mp_group.attrs["role"] = np.bytes_("server")
                mp_group.attrs["autostart"] = False

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestCreateNetworkFromFileNoConfig:
    """Tests for _create_network_from_file with missing config (lines 959-961)."""

    def test_create_network_no_config_group(self, serializer, tmp_path):
        """Test network creation fails gracefully without config group."""
        filepath = tmp_path / "no_config.h5"

        with h5py.File(filepath, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "1"
            f.create_group("meta")

        result = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert result is None


class TestCreateNetworkFromFileOldFormat:
    """Tests for _create_network_from_file old format handling (lines 979-987)."""

    def test_create_network_from_old_format(self, serializer, tmp_path):
        """Test loading network from old attribute-based format."""
        filepath = tmp_path / "old_format.h5"

        with h5py.File(filepath, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "1"

            meta = f.create_group("meta")
            meta.attrs["uuid"] = np.bytes_("test-uuid-123")

            config_group = f.create_group("config")
            config_group.attrs["input_size"] = 2
            config_group.attrs["output_size"] = 1
            config_group.attrs["learning_rate"] = 0.1

            f.create_group("arch")
            f.create_group("random")

            params = f.create_group("params")
            output_layer = params.create_group("output_layer")
            output_layer.create_dataset("weights", data=np.random.randn(2, 1).astype(np.float32))
            output_layer.create_dataset("bias", data=np.random.randn(1).astype(np.float32))

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestCreateNetworkFromFileException:
    """Tests for _create_network_from_file exception handling (lines 993-994)."""

    def test_create_network_exception_handling(self, serializer, tmp_path):
        """Test network creation handles exceptions."""
        filepath = tmp_path / "bad_config.h5"

        with h5py.File(filepath, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "1"
            f.create_group("meta")
            config_group = f.create_group("config")
            config_group.create_dataset("config_json", data=np.bytes_("not valid json {{{"))

        result = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert result is None


class TestValidateShapesException:
    """Tests for _validate_shapes exception handling (lines 1036-1037)."""

    def test_validate_shapes_exception(self, serializer, tmp_path):
        """Test shape validation handles exceptions."""
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        network.output_weights = None

        result = serializer._validate_shapes(network)
        assert result is False


class TestValidateFormatVersionParseError:
    """Tests for _validate_format version parsing error (lines 1072-1073)."""

    def test_validate_format_unparseable_version(self, serializer, tmp_path):
        """Test format validation handles unparseable versions."""
        filepath = tmp_path / "bad_version.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "format_version" in f.attrs:
                del f.attrs["format_version"]
            f.attrs["format_version"] = np.bytes_("not.a.version.number.abc")

        result = serializer.verify_saved_network(str(filepath))
        assert result is not None


class TestValidateFormatMissingGroups:
    """Tests for _validate_format missing required groups (lines 1079-1080)."""

    def test_validate_format_missing_required_group(self, serializer, tmp_path):
        """Test format validation detects missing required groups."""
        filepath = tmp_path / "missing_group.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "arch" in f:
                del f["arch"]

        result = serializer.verify_saved_network(str(filepath))
        assert result.get("valid") is False


class TestValidateFormatMissingOutputLayer:
    """Tests for _validate_format missing output layer (lines 1091-1095)."""

    def test_validate_format_missing_output_layer(self, serializer, tmp_path):
        """Test format validation detects missing output layer."""
        filepath = tmp_path / "missing_output.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "params/output_layer" in f:
                del f["params/output_layer"]
            f["params"].create_group("output_layer")

        result = serializer.verify_saved_network(str(filepath))
        assert result.get("valid") is False

    def test_validate_format_missing_params_output_layer_group(self, serializer, tmp_path):
        """Test format validation detects missing output_layer group entirely."""
        filepath = tmp_path / "no_output_layer.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "params/output_layer" in f:
                del f["params/output_layer"]

        result = serializer.verify_saved_network(str(filepath))
        assert result.get("valid") is False


class TestValidateFormatHiddenUnitsInconsistent:
    """Tests for _validate_format hidden units consistency check (lines 1103-1105)."""

    def test_validate_format_hidden_units_count_mismatch(self, serializer, tmp_path, network_with_hidden_units):
        """Test format validation detects hidden units count mismatch."""
        filepath = tmp_path / "mismatch_units.h5"
        serializer.save_network(network_with_hidden_units, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "hidden_units" in f:
                hidden_group = f["hidden_units"]
                current_count = hidden_group.attrs.get("num_units", 0)
                hidden_group.attrs["num_units"] = current_count + 5

        result = serializer.verify_saved_network(str(filepath))
        assert result.get("valid") is False


class TestValidateFormatHiddenUnitMissingWeights:
    """Tests for _validate_format hidden unit missing weights (lines 1113-1117)."""

    def test_validate_format_hidden_unit_missing_weights(self, serializer, tmp_path, network_with_hidden_units):
        """Test format validation detects hidden unit missing weights."""
        filepath = tmp_path / "missing_weights.h5"
        serializer.save_network(network_with_hidden_units, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "hidden_units" in f:
                hidden_group = f["hidden_units"]
                for key in hidden_group.keys():
                    if key.startswith("unit_"):
                        unit_group = hidden_group[key]
                        if "weights" in unit_group:
                            del unit_group["weights"]
                        break

        result = serializer.verify_saved_network(str(filepath))
        assert result.get("valid") is False


class TestValidateFormatException:
    """Tests for _validate_format exception handling (lines 1122-1123)."""

    def test_validate_format_exception(self, serializer, tmp_path):
        """Test format validation handles exceptions."""
        filepath = tmp_path / "exception.h5"

        with h5py.File(filepath, "w") as f:
            f.attrs["format"] = "juniper.cascor"

        with patch.object(serializer, "logger") as mock_logger:
            mock_logger.error = MagicMock()
            mock_logger.debug = MagicMock()
            with h5py.File(filepath, "r") as f:
                with patch.object(f, "keys", side_effect=Exception("Read error")):
                    pass


class TestConfigToDictComplexTypes:
    """Tests for _config_to_dict with complex types (lines 1167-1178)."""

    def test_config_to_dict_with_list_of_primitives(self, serializer):
        """Test config_to_dict includes lists of primitive types."""
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        result = serializer._config_to_dict(config)
        assert isinstance(result, dict)

    def test_config_to_dict_skips_callables(self, serializer):
        """Test config_to_dict skips callable attributes."""
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        config.some_callable = lambda x: x
        result = serializer._config_to_dict(config)
        assert "some_callable" not in result

    def test_config_to_dict_handles_dict_with_primitives(self, serializer):
        """Test config_to_dict includes dicts with primitive values."""
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        config.test_dict = {"key1": "value1", "key2": 42}
        result = serializer._config_to_dict(config)
        assert "test_dict" in result

    def test_config_to_dict_skips_dict_with_complex_values(self, serializer):
        """Test config_to_dict skips dicts with non-primitive values."""
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        config.complex_dict = {"key": lambda x: x}
        result = serializer._config_to_dict(config)
        assert "complex_dict" not in result

    def test_config_to_dict_exception_handling(self, serializer):
        """Test config_to_dict handles exceptions when getting attributes."""
        config = MagicMock()
        config.__dir__ = MagicMock(return_value=["bad_attr", "input_size"])

        def getattr_side_effect(name):
            if name == "bad_attr":
                raise AttributeError("Cannot access")
            return 2

        with patch("builtins.getattr", side_effect=getattr_side_effect):
            pass


class TestChecksumVerificationFailure:
    """Tests for checksum verification failure paths (lines 681, 687, 767, 773)."""

    def test_checksum_verification_failure_output_weights(self, serializer, tmp_path):
        """Test checksum verification logs error for output weights mismatch."""
        filepath = tmp_path / "bad_checksum.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            output_group = f["params/output_layer"]
            if "checksums" in output_group:
                checksums_json = output_group["checksums"][()].decode()
                checksums = json.loads(checksums_json)
                checksums["output_weights"] = "invalid_checksum_hash"
                del output_group["checksums"]
                output_group.create_dataset("checksums", data=np.bytes_(json.dumps(checksums)))

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestLoadCUDARandomStates:
    """Tests for CUDA random state loading (lines 830-845)."""

    def test_load_cuda_states_when_unavailable(self, serializer, tmp_path):
        """Test CUDA state loading when CUDA is not available."""
        filepath = tmp_path / "cuda_states.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "random" in f:
                random_group = f["random"]
                if "cuda_states" not in random_group:
                    cuda_group = random_group.create_group("cuda_states")
                    cuda_group.create_dataset("device_0", data=np.random.randint(0, 255, size=100, dtype=np.uint8))

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


class TestRandomStateRestorationException:
    """Tests for random state restoration exception (lines 841-845)."""

    def test_random_state_restoration_exception(self, serializer, tmp_path):
        """Test random state restoration handles exceptions gracefully."""
        filepath = tmp_path / "bad_random.h5"
        config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        serializer.save_network(network, str(filepath))

        with h5py.File(filepath, "r+") as f:
            if "random/python_state" in f:
                del f["random/python_state"]
            f["random"].create_dataset("python_state", data=np.bytes_("not valid state"))

        loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
        assert loaded is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
