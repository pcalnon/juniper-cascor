#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# File Name:     test_snapshot_serializer_coverage_deep.py
# Author:        Paul Calnon
# Version:       0.1.0
#
# Date Created:  2026-03-12
# Last Modified: 2026-03-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Deep coverage tests for CascadeHDF5Serializer targeting uncovered lines.
#    Covers: get_summary optional sections, save/load round trips with edge cases,
#    _validate_format with corrupted files, save/load branch coverage,
#    _config_to_dict type filtering.
#
#####################################################################################################################################################################################################
import json
import os
import sys
import tempfile

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from snapshots.snapshot_serializer import CascadeHDF5Serializer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def serializer():
    """Create a CascadeHDF5Serializer instance."""
    return CascadeHDF5Serializer()


@pytest.fixture
def simple_config():
    """Create a simple CascadeCorrelationConfig."""
    return CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=2,
        learning_rate=0.01,
        candidate_learning_rate=0.005,
        max_hidden_units=2,
        candidate_pool_size=2,
        correlation_threshold=0.01,
        patience=1,
        candidate_epochs=3,
        output_epochs=3,
        epochs_max=5,
    )


@pytest.fixture
def simple_network(simple_config):
    """Create a simple cascade correlation network."""
    return CascadeCorrelationNetwork(config=simple_config)


@pytest.fixture
def trained_network(simple_config):
    """Create a briefly-trained network with hidden units."""
    network = CascadeCorrelationNetwork(config=simple_config)
    torch.manual_seed(42)
    x = torch.randn(20, 2)
    y = torch.cat([torch.tensor([[1, 0]] * 10), torch.tensor([[0, 1]] * 10)], dim=0).float()
    network.fit(x, y, max_epochs=3)
    return network


@pytest.fixture
def temp_hdf5_path():
    """Provide a temporary HDF5 file path with cleanup."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        filepath = f.name
    yield filepath
    if os.path.exists(filepath):
        os.unlink(filepath)


def _create_valid_hdf5_skeleton(filepath, serializer, network):
    """Helper to create a minimal valid HDF5 file from a network."""
    serializer.save_network(network, filepath)


# ===========================================================================
# 1. verify_saved_network / get_summary optional sections
# ===========================================================================


class TestVerifySavedNetworkOptionalSections:
    """Test verify_saved_network with HDF5 files that have/miss optional groups."""

    @pytest.mark.unit
    def test_summary_with_all_optional_sections(self, serializer, trained_network, temp_hdf5_path):
        """Lines 196-220: File with history, mp, data groups."""
        # Save with training state and data
        trained_network._training_data = {
            "x_train": torch.randn(10, 2),
            "y_train": torch.randn(10, 2),
        }
        serializer.save_network(
            trained_network,
            temp_hdf5_path,
            include_training_state=True,
            include_training_data=True,
        )

        summary = serializer.verify_saved_network(temp_hdf5_path)
        assert summary["valid"] is True
        assert "format" in summary
        assert "format_version" in summary
        assert "serializer_version" in summary
        assert "created" in summary
        assert "file_size" in summary
        assert "network_uuid" in summary
        assert summary.get("has_mp") is True

    @pytest.mark.unit
    def test_summary_without_optional_sections(self, serializer, simple_network, temp_hdf5_path):
        """Lines 211-213: File without history/data groups."""
        serializer.save_network(simple_network, temp_hdf5_path)

        summary = serializer.verify_saved_network(temp_hdf5_path)
        assert summary["valid"] is True
        assert summary.get("has_history") is False
        assert summary.get("has_data") is False
        assert summary.get("has_mp") is True  # mp is always saved

    @pytest.mark.unit
    def test_summary_with_arch_section(self, serializer, simple_network, temp_hdf5_path):
        """Lines 203-208: File with arch group."""
        serializer.save_network(simple_network, temp_hdf5_path)

        summary = serializer.verify_saved_network(temp_hdf5_path)
        assert summary["valid"] is True
        assert summary.get("input_size") == 2
        assert summary.get("output_size") == 2
        assert summary.get("num_hidden_units", -1) >= 0

    @pytest.mark.unit
    def test_summary_with_meta_section(self, serializer, simple_network, temp_hdf5_path):
        """Lines 196-200: File with meta group."""
        serializer.save_network(simple_network, temp_hdf5_path)

        summary = serializer.verify_saved_network(temp_hdf5_path)
        assert summary["valid"] is True
        assert "network_uuid" in summary
        assert "python_version" in summary
        assert "torch_version" in summary

    @pytest.mark.unit
    def test_summary_invalid_file(self, serializer, temp_hdf5_path):
        """Lines 217-218: Invalid file returns error dict."""
        # Write garbage to the file
        with open(temp_hdf5_path, "wb") as f:
            f.write(b"not an hdf5 file")

        summary = serializer.verify_saved_network(temp_hdf5_path)
        assert summary["valid"] is False
        assert "error" in summary


# ===========================================================================
# 2. Save/Load round trips for edge cases
# ===========================================================================


class TestSaveLoadRoundTrips:
    """Test save/load round trips with various options."""

    @pytest.mark.unit
    def test_round_trip_basic(self, serializer, simple_network, temp_hdf5_path):
        """Basic save/load round trip without hidden units."""
        assert serializer.save_network(simple_network, temp_hdf5_path) is True

        loaded = serializer.load_network(temp_hdf5_path)
        assert loaded is not None
        assert loaded.input_size == simple_network.input_size
        assert loaded.output_size == simple_network.output_size

    @pytest.mark.unit
    def test_round_trip_with_hidden_units(self, serializer, trained_network, temp_hdf5_path):
        """Save a network with hidden units, verify loaded state matches."""
        num_hidden_before = len(trained_network.hidden_units)
        assert serializer.save_network(trained_network, temp_hdf5_path) is True

        loaded = serializer.load_network(temp_hdf5_path)
        assert loaded is not None
        assert len(loaded.hidden_units) == num_hidden_before

        # Verify hidden unit weights match
        for i, (orig, loaded_unit) in enumerate(zip(trained_network.hidden_units, loaded.hidden_units)):
            if "weights" in orig and "weights" in loaded_unit:
                torch.testing.assert_close(orig["weights"], loaded_unit["weights"])
            if "bias" in orig and "bias" in loaded_unit:
                torch.testing.assert_close(orig["bias"], loaded_unit["bias"])

    @pytest.mark.unit
    def test_round_trip_include_training_state_true(self, serializer, simple_network, temp_hdf5_path):
        """Save with include_training_state=True."""
        # Explicitly set history with correct format
        simple_network.history = {
            "train_loss": [0.5, 0.4, 0.3],
            "value_loss": [0.6, 0.5, 0.4],
            "train_accuracy": [0.6, 0.7, 0.8],
            "value_accuracy": [0.5, 0.6, 0.7],
            "hidden_units_added": [],
        }

        assert serializer.save_network(simple_network, temp_hdf5_path, include_training_state=True) is True

        loaded = serializer.load_network(temp_hdf5_path)
        assert loaded is not None
        if hasattr(loaded, "history") and loaded.history:
            assert "train_loss" in loaded.history

    @pytest.mark.unit
    def test_round_trip_include_training_data_true(self, serializer, simple_network, temp_hdf5_path):
        """Save with include_training_data=True."""
        simple_network._training_data = {
            "x_train": torch.randn(10, 2),
            "y_train": torch.randn(10, 2),
        }

        assert serializer.save_network(simple_network, temp_hdf5_path, include_training_data=True) is True

        # Verify data group exists in file
        with h5py.File(temp_hdf5_path, "r") as f:
            assert "data" in f

    @pytest.mark.unit
    def test_round_trip_include_training_data_false(self, serializer, simple_network, temp_hdf5_path):
        """Save with include_training_data=False (default)."""
        simple_network._training_data = {
            "x_train": torch.randn(10, 2),
        }

        assert serializer.save_network(simple_network, temp_hdf5_path, include_training_data=False) is True

        with h5py.File(temp_hdf5_path, "r") as f:
            assert "data" not in f

    @pytest.mark.unit
    def test_load_network_file_not_found(self, serializer):
        """Line 609: load_network with nonexistent file returns None."""
        result = serializer.load_network("/nonexistent/path/file.h5")
        assert result is None

    @pytest.mark.unit
    def test_load_network_no_multiprocessing_restore(self, serializer, simple_network, temp_hdf5_path):
        """Line 623: load_network with restore_multiprocessing=False."""
        serializer.save_network(simple_network, temp_hdf5_path)
        loaded = serializer.load_network(temp_hdf5_path, restore_multiprocessing=False)
        assert loaded is not None


# ===========================================================================
# 3. _validate_format with corrupted files
# ===========================================================================


class TestValidateFormatCorrupted:
    """Test _validate_format with corrupted/incomplete HDF5 files."""

    @pytest.mark.unit
    def test_missing_format_attribute(self, serializer, temp_hdf5_path):
        """Lines 1060-1067: File with wrong format name."""
        with h5py.File(temp_hdf5_path, "w") as f:
            f.attrs["format"] = "wrong_format"
            f.attrs["format_version"] = "2"

        with h5py.File(temp_hdf5_path, "r") as f:
            assert serializer._validate_format(f) is False

    @pytest.mark.unit
    def test_incompatible_format_version(self, serializer, temp_hdf5_path):
        """Lines 1075-1077: File with future format version."""
        with h5py.File(temp_hdf5_path, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "99"

        with h5py.File(temp_hdf5_path, "r") as f:
            assert serializer._validate_format(f) is False

    @pytest.mark.unit
    def test_unparseable_format_version(self, serializer, temp_hdf5_path):
        """Lines 1078-1079: Unparseable format version string."""
        with h5py.File(temp_hdf5_path, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "not_a_number"
            # Add required groups to pass remaining checks
            f.create_group("meta")
            f.create_group("config")
            params = f.create_group("params")
            output = params.create_group("output_layer")
            output.create_dataset("weights", data=np.zeros((2, 2)))
            output.create_dataset("bias", data=np.zeros(2))
            f.create_group("arch")
            f.create_group("random")

        with h5py.File(temp_hdf5_path, "r") as f:
            # Should not crash, just warn and continue
            result = serializer._validate_format(f)
            # The format is valid if required groups exist (version parse warns but continues)
            assert isinstance(result, bool)

    @pytest.mark.unit
    def test_missing_required_group_meta(self, serializer, temp_hdf5_path):
        """Lines 1082-1086: Missing 'meta' required group."""
        with h5py.File(temp_hdf5_path, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "2"
            f.create_group("config")
            f.create_group("params")
            f.create_group("arch")
            f.create_group("random")

        with h5py.File(temp_hdf5_path, "r") as f:
            assert serializer._validate_format(f) is False

    @pytest.mark.unit
    def test_missing_required_group_config(self, serializer, temp_hdf5_path):
        """Lines 1082-1086: Missing 'config' required group."""
        with h5py.File(temp_hdf5_path, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "2"
            f.create_group("meta")
            f.create_group("params")
            f.create_group("arch")
            f.create_group("random")

        with h5py.File(temp_hdf5_path, "r") as f:
            assert serializer._validate_format(f) is False

    @pytest.mark.unit
    def test_missing_output_layer_group(self, serializer, temp_hdf5_path):
        """Lines 1099-1101: Missing output_layer group in params."""
        with h5py.File(temp_hdf5_path, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "2"
            f.create_group("meta")
            f.create_group("config")
            f.create_group("params")  # No output_layer subgroup
            f.create_group("arch")
            f.create_group("random")

        with h5py.File(temp_hdf5_path, "r") as f:
            assert serializer._validate_format(f) is False

    @pytest.mark.unit
    def test_missing_output_layer_weights(self, serializer, temp_hdf5_path):
        """Lines 1093-1095: Missing weights in output_layer."""
        with h5py.File(temp_hdf5_path, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "2"
            f.create_group("meta")
            f.create_group("config")
            params = f.create_group("params")
            output = params.create_group("output_layer")
            output.create_dataset("bias", data=np.zeros(2))  # No weights
            f.create_group("arch")
            f.create_group("random")

        with h5py.File(temp_hdf5_path, "r") as f:
            assert serializer._validate_format(f) is False

    @pytest.mark.unit
    def test_missing_output_layer_bias(self, serializer, temp_hdf5_path):
        """Lines 1096-1098: Missing bias in output_layer."""
        with h5py.File(temp_hdf5_path, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "2"
            f.create_group("meta")
            f.create_group("config")
            params = f.create_group("params")
            output = params.create_group("output_layer")
            output.create_dataset("weights", data=np.zeros((2, 2)))  # No bias
            f.create_group("arch")
            f.create_group("random")

        with h5py.File(temp_hdf5_path, "r") as f:
            assert serializer._validate_format(f) is False

    @pytest.mark.unit
    def test_hidden_units_count_mismatch(self, serializer, temp_hdf5_path):
        """Lines 1109-1111: Hidden units count attr != actual groups."""
        with h5py.File(temp_hdf5_path, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "2"
            f.create_group("meta")
            f.create_group("config")
            params = f.create_group("params")
            output = params.create_group("output_layer")
            output.create_dataset("weights", data=np.zeros((2, 2)))
            output.create_dataset("bias", data=np.zeros(2))
            f.create_group("arch")
            f.create_group("random")
            hidden = f.create_group("hidden_units")
            hidden.attrs["num_units"] = 3  # Says 3 but has only 1
            unit = hidden.create_group("unit_0")
            unit.create_dataset("weights", data=np.zeros(2))
            unit.create_dataset("bias", data=np.zeros(1))

        with h5py.File(temp_hdf5_path, "r") as f:
            assert serializer._validate_format(f) is False

    @pytest.mark.unit
    def test_hidden_unit_missing_weights(self, serializer, temp_hdf5_path):
        """Lines 1118-1120: Hidden unit missing weights dataset."""
        with h5py.File(temp_hdf5_path, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "2"
            f.create_group("meta")
            f.create_group("config")
            params = f.create_group("params")
            output = params.create_group("output_layer")
            output.create_dataset("weights", data=np.zeros((2, 2)))
            output.create_dataset("bias", data=np.zeros(2))
            f.create_group("arch")
            f.create_group("random")
            hidden = f.create_group("hidden_units")
            hidden.attrs["num_units"] = 1
            unit = hidden.create_group("unit_0")
            unit.create_dataset("bias", data=np.zeros(1))  # No weights

        with h5py.File(temp_hdf5_path, "r") as f:
            assert serializer._validate_format(f) is False

    @pytest.mark.unit
    def test_hidden_unit_missing_bias(self, serializer, temp_hdf5_path):
        """Lines 1121-1123: Hidden unit missing bias dataset."""
        with h5py.File(temp_hdf5_path, "w") as f:
            f.attrs["format"] = "juniper.cascor"
            f.attrs["format_version"] = "2"
            f.create_group("meta")
            f.create_group("config")
            params = f.create_group("params")
            output = params.create_group("output_layer")
            output.create_dataset("weights", data=np.zeros((2, 2)))
            output.create_dataset("bias", data=np.zeros(2))
            f.create_group("arch")
            f.create_group("random")
            hidden = f.create_group("hidden_units")
            hidden.attrs["num_units"] = 1
            unit = hidden.create_group("unit_0")
            unit.create_dataset("weights", data=np.zeros(2))  # No bias

        with h5py.File(temp_hdf5_path, "r") as f:
            assert serializer._validate_format(f) is False

    @pytest.mark.unit
    def test_valid_format_passes(self, serializer, simple_network, temp_hdf5_path):
        """Verify a properly saved file passes validation."""
        serializer.save_network(simple_network, temp_hdf5_path)
        with h5py.File(temp_hdf5_path, "r") as f:
            assert serializer._validate_format(f) is True


# ===========================================================================
# 4. Branch coverage for save/load methods
# ===========================================================================


class TestSaveLoadBranches:
    """Test branch coverage in save/load methods."""

    @pytest.mark.unit
    def test_save_configuration_attributes(self, serializer, simple_network, temp_hdf5_path):
        """Lines 224-259: _save_configuration saves key attrs."""
        serializer.save_network(simple_network, temp_hdf5_path)
        with h5py.File(temp_hdf5_path, "r") as f:
            assert "config" in f
            config_group = f["config"]
            assert config_group.attrs["input_size"] == 2
            assert config_group.attrs["output_size"] == 2
            assert "config_json" in config_group

    @pytest.mark.unit
    def test_save_architecture_with_hidden_units(self, serializer, trained_network, temp_hdf5_path):
        """Lines 261-287: _save_architecture with hidden units."""
        serializer.save_network(trained_network, temp_hdf5_path)
        with h5py.File(temp_hdf5_path, "r") as f:
            assert "arch" in f
            arch = f["arch"]
            assert arch.attrs["input_size"] == 2
            assert arch.attrs["output_size"] == 2
            assert "connectivity" in arch

    @pytest.mark.unit
    def test_save_hidden_units_with_data(self, serializer, trained_network, temp_hdf5_path):
        """Lines 347-394: _save_hidden_units with populated units."""
        if len(trained_network.hidden_units) == 0:
            # Force at least one hidden unit for branch coverage
            trained_network.hidden_units.append(
                {
                    "weights": torch.randn(2),
                    "bias": torch.randn(1),
                    "correlation": 0.5,
                    "activation_fn": torch.tanh,
                }
            )
        serializer.save_network(trained_network, temp_hdf5_path)

        with h5py.File(temp_hdf5_path, "r") as f:
            assert "hidden_units" in f

    @pytest.mark.unit
    def test_save_load_no_hidden_units(self, serializer, simple_network, temp_hdf5_path):
        """Lines 349-350: _save_hidden_units returns early when no hidden units."""
        simple_network.hidden_units = []
        serializer.save_network(simple_network, temp_hdf5_path)

        with h5py.File(temp_hdf5_path, "r") as f:
            assert "hidden_units" not in f

    @pytest.mark.unit
    def test_load_no_arch_group(self, serializer, simple_network, temp_hdf5_path):
        """Lines 634-635: _load_architecture returns early when 'arch' is missing."""
        serializer.save_network(simple_network, temp_hdf5_path)

        # Remove arch group from the file
        with h5py.File(temp_hdf5_path, "r+") as f:
            del f["arch"]

        # Loading should still work but skip architecture loading
        # (it will fail at _validate_format since arch is required, so test the method directly)
        loaded = CascadeCorrelationNetwork(config=CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=2))
        with h5py.File(temp_hdf5_path, "r") as f:
            serializer._load_architecture(f, loaded)  # arch missing -> returns early

    @pytest.mark.unit
    def test_load_no_params_group(self, serializer, simple_network, temp_hdf5_path):
        """Lines 660-661: _load_parameters returns early when 'params' is missing."""
        serializer.save_network(simple_network, temp_hdf5_path)

        loaded = CascadeCorrelationNetwork(config=CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=2))
        with h5py.File(temp_hdf5_path, "r") as f:
            # Create a file without params to test the early return
            pass  # This branch is tested by calling _load_parameters directly

        # Test directly with a file that has no params
        with h5py.File(temp_hdf5_path, "r+") as f:
            del f["params"]
        with h5py.File(temp_hdf5_path, "r") as f:
            serializer._load_parameters(f, loaded)

    @pytest.mark.unit
    def test_load_no_random_group(self, serializer, simple_network, temp_hdf5_path):
        """Lines 799-800: _load_random_state returns early when 'random' is missing."""
        serializer.save_network(simple_network, temp_hdf5_path)
        loaded = CascadeCorrelationNetwork(config=CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=2))

        with h5py.File(temp_hdf5_path, "r+") as f:
            del f["random"]
        with h5py.File(temp_hdf5_path, "r") as f:
            serializer._load_random_state(f, loaded)

    @pytest.mark.unit
    def test_load_no_hidden_units_group(self, serializer, simple_network, temp_hdf5_path):
        """Lines 739-741: _load_hidden_units sets empty list when group is missing."""
        serializer.save_network(simple_network, temp_hdf5_path)
        loaded = CascadeCorrelationNetwork(config=CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=2))

        with h5py.File(temp_hdf5_path, "r") as f:
            serializer._load_hidden_units(f, loaded)
            assert isinstance(loaded.hidden_units, list)

    @pytest.mark.unit
    def test_save_training_history_with_hidden_units_added(self, serializer, simple_network, temp_hdf5_path):
        """Lines 545-567: _save_training_history with hidden_units_added."""
        simple_network.history = {
            "train_loss": [0.5, 0.4],
            "value_loss": [0.6, 0.5],
            "train_accuracy": [0.6, 0.7],
            "value_accuracy": [0.5, 0.6],
            "hidden_units_added": [
                {"correlation": 0.8, "weights": np.array([0.1, 0.2]), "bias": np.array([0.01])},
            ],
        }
        serializer.save_network(simple_network, temp_hdf5_path, include_training_state=True)

        with h5py.File(temp_hdf5_path, "r") as f:
            assert "history" in f
            assert "hidden_units_added" in f["history"]

    @pytest.mark.unit
    def test_save_training_data_with_numpy_arrays(self, serializer, simple_network, temp_hdf5_path):
        """Lines 571-592: _save_training_data with numpy arrays."""
        simple_network._training_data = {
            "x_np": np.random.randn(10, 2).astype(np.float32),
            "x_torch": torch.randn(10, 2),
        }
        serializer.save_network(simple_network, temp_hdf5_path, include_training_data=True)

        with h5py.File(temp_hdf5_path, "r") as f:
            assert "data" in f

    @pytest.mark.unit
    def test_save_training_data_no_data(self, serializer, simple_network, temp_hdf5_path):
        """Lines 579-580: _save_training_data returns early when no _training_data."""
        # Ensure no _training_data attribute
        if hasattr(simple_network, "_training_data"):
            delattr(simple_network, "_training_data")
        serializer.save_network(simple_network, temp_hdf5_path, include_training_data=True)

        with h5py.File(temp_hdf5_path, "r") as f:
            assert "data" not in f

    @pytest.mark.unit
    def test_save_training_history_empty(self, serializer, simple_network, temp_hdf5_path):
        """Lines 526-527: _save_training_history returns early when no history."""
        simple_network.history = {}
        serializer.save_network(simple_network, temp_hdf5_path, include_training_state=True)

        with h5py.File(temp_hdf5_path, "r") as f:
            assert "history" not in f

    @pytest.mark.unit
    def test_save_object_method(self, serializer, simple_network, temp_hdf5_path):
        """Lines 96-129: save_object method."""
        result = serializer.save_object(objectify=simple_network, filepath=temp_hdf5_path)
        assert result is True

    @pytest.mark.unit
    def test_load_training_history(self, serializer, simple_network, temp_hdf5_path):
        """Lines 866-915: _load_training_history with various key formats."""
        simple_network.history = {
            "train_loss": [0.5, 0.4, 0.3],
            "value_loss": [0.6, 0.5, 0.4],
            "train_accuracy": [0.6, 0.7, 0.8],
            "value_accuracy": [0.5, 0.6, 0.7],
            "hidden_units_added": [
                {"correlation": 0.75},
            ],
        }
        serializer.save_network(simple_network, temp_hdf5_path, include_training_state=True)

        loaded = serializer.load_network(temp_hdf5_path)
        assert loaded is not None
        assert len(loaded.history.get("train_loss", [])) == 3


# ===========================================================================
# 5. _config_to_dict type filtering
# ===========================================================================


class TestConfigToDict:
    """Test _config_to_dict with various attribute types."""

    @pytest.mark.unit
    def test_primitive_types_included(self, serializer, simple_config):
        """Lines 1166-1167: String, int, float, bool, None are included."""
        result = serializer._config_to_dict(simple_config)
        assert isinstance(result, dict)
        # Should contain at least input_size, output_size, learning_rate
        assert "input_size" in result
        assert "output_size" in result

    @pytest.mark.unit
    def test_callable_excluded(self, serializer):
        """Lines 1162-1163: Callable attributes are excluded."""

        class FakeConfig:
            input_size = 2
            output_size = 2

            def some_method(self):
                pass

        config = FakeConfig()
        result = serializer._config_to_dict(config)
        assert "some_method" not in result
        assert "input_size" in result

    @pytest.mark.unit
    def test_excluded_attrs_filtered(self, serializer):
        """Lines 1148-1152: Excluded attributes are filtered."""

        class FakeConfig:
            input_size = 2
            activation_functions_dict = {"tanh": torch.tanh}
            log_config = "complex_object"
            logger = "runtime_logger"

        config = FakeConfig()
        result = serializer._config_to_dict(config)
        assert "activation_functions_dict" not in result
        assert "log_config" not in result
        assert "logger" not in result
        assert "input_size" in result

    @pytest.mark.unit
    def test_list_with_primitives_included(self, serializer):
        """Lines 1168-1172: Lists with primitive items included."""

        class FakeConfig:
            my_list = [1, 2, 3]
            my_complex_list = [torch.tanh]  # non-primitive, excluded

        config = FakeConfig()
        result = serializer._config_to_dict(config)
        assert "my_list" in result
        assert result["my_list"] == [1, 2, 3]
        assert "my_complex_list" not in result

    @pytest.mark.unit
    def test_dict_with_primitives_included(self, serializer):
        """Lines 1173-1177: Dicts with primitive values included."""

        class FakeConfig:
            my_dict = {"a": 1, "b": "two"}
            my_complex_dict = {"fn": torch.tanh}  # non-primitive value, excluded

        config = FakeConfig()
        result = serializer._config_to_dict(config)
        assert "my_dict" in result
        assert result["my_dict"] == {"a": 1, "b": "two"}
        assert "my_complex_dict" not in result

    @pytest.mark.unit
    def test_path_type_converted_to_string(self, serializer):
        """Lines 1178-1179: pathlib.Path attributes converted to string."""
        import pathlib

        class FakeConfig:
            my_path = pathlib.Path("/some/path")

        config = FakeConfig()
        result = serializer._config_to_dict(config)
        assert "my_path" in result
        assert result["my_path"] == "/some/path"

    @pytest.mark.unit
    def test_private_attrs_excluded(self, serializer):
        """Line 1156: Attributes starting with _ are excluded."""

        class FakeConfig:
            input_size = 2
            _private = "hidden"

        config = FakeConfig()
        result = serializer._config_to_dict(config)
        assert "_private" not in result

    @pytest.mark.unit
    def test_none_value_included(self, serializer):
        """Lines 1166-1167: None values are included as primitive type."""

        class FakeConfig:
            optional_field = None

        config = FakeConfig()
        result = serializer._config_to_dict(config)
        assert "optional_field" in result
        assert result["optional_field"] is None


# ===========================================================================
# 6. Additional edge cases
# ===========================================================================


class TestSerializerEdgeCases:
    """Test miscellaneous edge cases in the serializer."""

    @pytest.mark.unit
    def test_validate_shapes_mismatch(self, serializer, simple_network):
        """Lines 1011-1043: _validate_shapes with wrong output weights shape."""
        # Force a shape mismatch
        simple_network.output_weights = torch.randn(99, 2)
        result = serializer._validate_shapes(simple_network)
        assert result is False

    @pytest.mark.unit
    def test_validate_shapes_bias_mismatch(self, serializer, simple_network):
        """Lines 1029-1031: _validate_shapes with wrong output bias shape."""
        # Correct weight shape but wrong bias
        expected_input = simple_network.input_size + len(simple_network.hidden_units)
        simple_network.output_weights = torch.randn(expected_input, simple_network.output_size)
        simple_network.output_bias = torch.randn(99)  # Wrong shape
        result = serializer._validate_shapes(simple_network)
        assert result is False

    @pytest.mark.unit
    def test_validate_shapes_valid(self, serializer, simple_network):
        """Lines 1011-1041: _validate_shapes passes for valid network."""
        result = serializer._validate_shapes(simple_network)
        assert result is True

    @pytest.mark.unit
    def test_log_exception_stacktrace(self, serializer):
        """Lines 1131-1136: _log_exception_stacktrace returns arg2."""
        result = serializer._log_exception_stacktrace("Test error: ", ValueError("test"), False)
        assert result is False

        result_none = serializer._log_exception_stacktrace("Test error: ", ValueError("test"), None)
        assert result_none is None

    @pytest.mark.unit
    def test_save_network_exception_handling(self, serializer, simple_network):
        """Lines 93-94: save_network exception returns False."""
        # Try to save to an invalid path
        result = serializer.save_network(simple_network, "/proc/impossible/path/file.h5")
        assert result is False

    @pytest.mark.unit
    def test_create_network_from_file_no_config(self, serializer, temp_hdf5_path):
        """Lines 965-967: _create_network_from_file when 'config' is missing."""
        with h5py.File(temp_hdf5_path, "w") as f:
            f.attrs["format"] = "juniper.cascor"

        with h5py.File(temp_hdf5_path, "r") as f:
            result = serializer._create_network_from_file(f)
            assert result is None

    @pytest.mark.unit
    def test_restore_training_state_helper(self, serializer, simple_network, temp_hdf5_path):
        """Lines 1002-1009: _restore_training_state_helper sets counters."""
        serializer.save_network(simple_network, temp_hdf5_path)

        with h5py.File(temp_hdf5_path, "r") as f:
            network = serializer._create_network_from_file(f)
            assert network is not None
            assert hasattr(network, "snapshot_counter")

    @pytest.mark.unit
    def test_save_multiprocessing_state(self, serializer, simple_network, temp_hdf5_path):
        """Lines 464-474: _save_multiprocessing_state."""
        serializer.save_network(simple_network, temp_hdf5_path)
        with h5py.File(temp_hdf5_path, "r") as f:
            assert "mp" in f

    @pytest.mark.unit
    def test_version_attributes(self, serializer):
        """Verify serializer version attributes are set."""
        assert serializer.version == "2.0.0"
        assert serializer.format_version == "2"
        assert serializer.format_name == "juniper.cascor"
