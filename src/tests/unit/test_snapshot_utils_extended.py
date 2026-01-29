#!/usr/bin/env python
"""
Extended unit tests for snapshot_utils.py HDF5Utils class.

Tests focus on improving coverage for:
- create_backup()
- list_networks_in_directory()
- compare_networks()
- compress_hdf5_file()
- get_file_info()
- get_network_summary()
- cleanup_old_files()
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

# Add parent directories for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from snapshots.snapshot_utils import HDF5Utils

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


@pytest.fixture
def simple_hdf5_file(tmp_path):
    """Create a simple HDF5 file for testing."""
    filepath = tmp_path / "test_network.h5"
    with h5py.File(filepath, "w") as f:
        f.attrs["format"] = "juniper.cascor"
        f.attrs["format_version"] = "2"

        # Create a group with datasets
        grp = f.create_group("network")
        grp.attrs["uuid"] = "test-uuid-1234"
        grp.create_dataset("weights", data=np.random.rand(10, 5))
        grp.create_dataset("biases", data=np.random.rand(5))
    return str(filepath)


@pytest.fixture
def mock_serializer():
    """Create a mock CascadeHDF5Serializer."""
    with patch("snapshots.snapshot_utils.CascadeHDF5Serializer") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


class TestCreateBackup:
    """Tests for HDF5Utils.create_backup()."""

    def test_create_backup_success(self, simple_hdf5_file, tmp_path):
        """Test successful backup creation."""
        backup_path = HDF5Utils.create_backup(simple_hdf5_file)

        assert os.path.exists(backup_path)
        assert "_backup_" in backup_path
        assert backup_path.endswith(".h5")
        # Backup should be in same directory as original by default
        assert os.path.dirname(backup_path) == os.path.dirname(simple_hdf5_file)

    def test_create_backup_file_not_found(self, tmp_path):
        """Test backup with non-existent file raises FileNotFoundError."""
        nonexistent = str(tmp_path / "nonexistent.h5")

        with pytest.raises(FileNotFoundError) as excinfo:
            HDF5Utils.create_backup(nonexistent)

        assert "Original file not found" in str(excinfo.value)

    def test_create_backup_custom_backup_dir(self, simple_hdf5_file, tmp_path):
        """Test backup with custom backup directory."""
        backup_dir = tmp_path / "backups"

        backup_path = HDF5Utils.create_backup(simple_hdf5_file, str(backup_dir))

        assert os.path.exists(backup_path)
        assert str(backup_dir) in backup_path
        assert backup_dir.exists()

    def test_create_backup_creates_backup_dir(self, simple_hdf5_file, tmp_path):
        """Test that backup creates the backup directory if it doesn't exist."""
        backup_dir = tmp_path / "nested" / "backup" / "dir"
        assert not backup_dir.exists()

        backup_path = HDF5Utils.create_backup(simple_hdf5_file, str(backup_dir))

        assert backup_dir.exists()
        assert os.path.exists(backup_path)


class TestListNetworksInDirectory:
    """Tests for HDF5Utils.list_networks_in_directory()."""

    def test_list_empty_directory(self, tmp_path):
        """Test listing networks in an empty directory."""
        result = HDF5Utils.list_networks_in_directory(str(tmp_path))
        assert result == []

    def test_list_nonexistent_directory(self, tmp_path):
        """Test listing networks in a non-existent directory."""
        nonexistent = str(tmp_path / "nonexistent_dir")
        result = HDF5Utils.list_networks_in_directory(nonexistent)
        assert result == []

    def test_list_with_valid_network_files(self, tmp_path, mock_serializer):
        """Test listing with valid network files."""
        # Create test files
        (tmp_path / "network1.h5").touch()
        (tmp_path / "network2.hdf5").touch()
        (tmp_path / "other.txt").touch()  # Should be ignored

        # Mock verification to return valid for all files
        mock_serializer.verify_saved_network.return_value = {"valid": True, "input_size": 10, "output_size": 5}

        result = HDF5Utils.list_networks_in_directory(str(tmp_path))

        assert len(result) == 2
        assert all(r["valid"] for r in result)
        assert all("filename" in r for r in result)
        assert all("filepath" in r for r in result)

    def test_list_with_invalid_network_files(self, tmp_path, mock_serializer):
        """Test listing filters out invalid network files."""
        # Create test files
        (tmp_path / "valid.h5").touch()
        (tmp_path / "invalid.h5").touch()

        # First call returns valid, second returns invalid
        mock_serializer.verify_saved_network.side_effect = [{"valid": True, "input_size": 10}, {"valid": False, "error": "Invalid format"}]

        result = HDF5Utils.list_networks_in_directory(str(tmp_path))

        assert len(result) == 1
        assert result[0]["valid"] is True

    def test_list_handles_verification_exception(self, tmp_path, mock_serializer):
        """Test listing handles exceptions during verification gracefully."""
        (tmp_path / "broken.h5").touch()

        mock_serializer.verify_saved_network.side_effect = Exception("Read error")

        result = HDF5Utils.list_networks_in_directory(str(tmp_path))

        assert result == []


class TestCompareNetworks:
    """Tests for HDF5Utils.compare_networks()."""

    def test_compare_same_networks(self, tmp_path, mock_serializer):
        """Test comparing identical networks."""
        file1 = str(tmp_path / "network1.h5")
        file2 = str(tmp_path / "network2.h5")

        mock_serializer.verify_saved_network.return_value = {"valid": True, "input_size": 10, "output_size": 5, "num_hidden_units": 3, "activation_function": "tanh"}

        result = HDF5Utils.compare_networks(file1, file2)

        assert result["comparable"] is True
        assert result["same_architecture"] is True
        assert result["same_hidden_units"] is True
        assert result["same_activation"] is True

    def test_compare_different_networks(self, tmp_path, mock_serializer):
        """Test comparing different networks."""
        file1 = str(tmp_path / "network1.h5")
        file2 = str(tmp_path / "network2.h5")

        mock_serializer.verify_saved_network.side_effect = [{"valid": True, "input_size": 10, "output_size": 5, "num_hidden_units": 3, "activation_function": "tanh"}, {"valid": True, "input_size": 20, "output_size": 10, "num_hidden_units": 5, "activation_function": "sigmoid"}]

        result = HDF5Utils.compare_networks(file1, file2)

        assert result["comparable"] is True
        assert result["same_architecture"] is False
        assert result["same_hidden_units"] is False
        assert result["same_activation"] is False
        assert result["architecture_diff"]["input_size"] == (10, 20)
        assert result["architecture_diff"]["output_size"] == (5, 10)

    def test_compare_invalid_first_file(self, tmp_path, mock_serializer):
        """Test comparing when first file is invalid."""
        file1 = str(tmp_path / "invalid.h5")
        file2 = str(tmp_path / "valid.h5")

        mock_serializer.verify_saved_network.side_effect = [{"valid": False, "error": "Invalid format"}, {"valid": True, "input_size": 10}]

        result = HDF5Utils.compare_networks(file1, file2)

        assert result["comparable"] is False
        assert "error" in result
        assert result["file1_valid"] is False
        assert result["file2_valid"] is True

    def test_compare_invalid_second_file(self, tmp_path, mock_serializer):
        """Test comparing when second file is invalid."""
        file1 = str(tmp_path / "valid.h5")
        file2 = str(tmp_path / "invalid.h5")

        mock_serializer.verify_saved_network.side_effect = [{"valid": True, "input_size": 10}, {"valid": False, "error": "Corrupted file"}]

        result = HDF5Utils.compare_networks(file1, file2)

        assert result["comparable"] is False
        assert result["file1_valid"] is True
        assert result["file2_valid"] is False

    def test_compare_both_invalid(self, tmp_path, mock_serializer):
        """Test comparing when both files are invalid."""
        file1 = str(tmp_path / "invalid1.h5")
        file2 = str(tmp_path / "invalid2.h5")

        mock_serializer.verify_saved_network.return_value = {"valid": False}

        result = HDF5Utils.compare_networks(file1, file2)

        assert result["comparable"] is False
        assert result["file1_valid"] is False
        assert result["file2_valid"] is False


class TestCompressHDF5File:
    """Tests for HDF5Utils.compress_hdf5_file()."""

    def test_compress_successful(self, simple_hdf5_file, tmp_path):
        """Test successful compression."""
        output_path = str(tmp_path / "compressed.h5")

        result = HDF5Utils.compress_hdf5_file(simple_hdf5_file, output_path)

        assert result is True
        assert os.path.exists(output_path)

        # Verify compressed file has the data
        with h5py.File(output_path, "r") as f:
            assert "network" in f
            assert "weights" in f["network"]
            assert f["network"]["weights"].compression == "gzip"

    def test_compress_preserves_attributes(self, simple_hdf5_file, tmp_path):
        """Test that compression preserves attributes."""
        output_path = str(tmp_path / "compressed.h5")

        HDF5Utils.compress_hdf5_file(simple_hdf5_file, output_path)

        with h5py.File(output_path, "r") as f:
            assert f.attrs["format"] == "juniper.cascor"
            assert f.attrs["format_version"] == "2"
            assert f["network"].attrs["uuid"] == "test-uuid-1234"

    def test_compress_with_custom_settings(self, simple_hdf5_file, tmp_path):
        """Test compression with custom compression settings."""
        output_path = str(tmp_path / "compressed.h5")

        result = HDF5Utils.compress_hdf5_file(simple_hdf5_file, output_path, compression="gzip", compression_opts=4)

        assert result is True

    def test_compress_error_handling(self, tmp_path):
        """Test compression error handling with invalid input."""
        nonexistent = str(tmp_path / "nonexistent.h5")
        output_path = str(tmp_path / "output.h5")

        result = HDF5Utils.compress_hdf5_file(nonexistent, output_path)

        assert result is False


class TestGetFileInfo:
    """Tests for HDF5Utils.get_file_info()."""

    def test_get_file_info_not_found(self, tmp_path):
        """Test get_file_info with non-existent file."""
        nonexistent = str(tmp_path / "nonexistent.h5")

        result = HDF5Utils.get_file_info(nonexistent)

        assert result["exists"] is False
        assert "error" in result
        assert result["error"] == "File not found"

    def test_get_file_info_valid_file(self, simple_hdf5_file):
        """Test get_file_info with valid file."""
        result = HDF5Utils.get_file_info(simple_hdf5_file)

        assert result["exists"] is True
        assert result["filepath"] == simple_hdf5_file
        assert "size_bytes" in result
        assert "size_mb" in result
        assert "modified_time" in result
        assert isinstance(result["groups"], list)
        assert isinstance(result["datasets"], list)
        assert isinstance(result["attributes"], dict)

    def test_get_file_info_groups_and_datasets(self, simple_hdf5_file):
        """Test that get_file_info correctly identifies groups and datasets."""
        result = HDF5Utils.get_file_info(simple_hdf5_file)

        # Should have the "network" group
        group_paths = [g["path"] for g in result["groups"]]
        assert "network" in group_paths

        # Should have the datasets
        dataset_paths = [d["path"] for d in result["datasets"]]
        assert "network/weights" in dataset_paths
        assert "network/biases" in dataset_paths

        # Check dataset info
        weights_dataset = next(d for d in result["datasets"] if d["path"] == "network/weights")
        assert weights_dataset["shape"] == (10, 5)
        assert "dtype" in weights_dataset

    def test_get_file_info_root_attributes(self, simple_hdf5_file):
        """Test that get_file_info reads root attributes."""
        result = HDF5Utils.get_file_info(simple_hdf5_file)

        assert "format" in result["attributes"]
        assert "format_version" in result["attributes"]

    def test_get_file_info_handles_corrupted_file(self, tmp_path):
        """Test get_file_info with corrupted file."""
        corrupted = tmp_path / "corrupted.h5"
        corrupted.write_text("not valid hdf5 content")

        result = HDF5Utils.get_file_info(str(corrupted))

        assert result["exists"] is True
        assert "error" in result


class TestGetNetworkSummary:
    """Tests for HDF5Utils.get_network_summary()."""

    def test_get_network_summary_valid(self, simple_hdf5_file, mock_serializer):
        """Test get_network_summary with valid network."""
        mock_serializer.verify_saved_network.return_value = {"valid": True, "network_uuid": "test-uuid", "input_size": 10, "output_size": 5, "num_hidden_units": 3, "activation_function": "tanh", "format_version": "2", "has_history": True, "has_mp": False}

        result = HDF5Utils.get_network_summary(simple_hdf5_file)

        assert result is not None
        assert result["filepath"] == simple_hdf5_file
        assert result["uuid"] == "test-uuid"
        assert result["input_size"] == 10
        assert result["output_size"] == 5
        assert result["num_hidden_units"] == 3
        assert result["activation_function"] == "tanh"
        assert result["has_training_history"] is True
        assert result["has_multiprocessing"] is False

    def test_get_network_summary_invalid(self, tmp_path, mock_serializer):
        """Test get_network_summary with invalid network returns None."""
        invalid_file = str(tmp_path / "invalid.h5")

        mock_serializer.verify_saved_network.return_value = {"valid": False, "error": "Not a valid network file"}

        result = HDF5Utils.get_network_summary(invalid_file)

        assert result is None

    def test_get_network_summary_includes_file_info(self, simple_hdf5_file, mock_serializer):
        """Test that get_network_summary includes file size and modification time."""
        mock_serializer.verify_saved_network.return_value = {"valid": True, "input_size": 10, "output_size": 5}

        result = HDF5Utils.get_network_summary(simple_hdf5_file)

        assert "size_mb" in result
        assert "modified" in result
        assert "filename" in result


class TestCleanupOldFiles:
    """Tests for HDF5Utils.cleanup_old_files()."""

    def test_cleanup_nonexistent_directory(self, tmp_path):
        """Test cleanup on non-existent directory returns 0."""
        nonexistent = str(tmp_path / "nonexistent_dir")

        result = HDF5Utils.cleanup_old_files(nonexistent)

        assert result == 0

    def test_cleanup_empty_directory(self, tmp_path):
        """Test cleanup on empty directory returns 0."""
        result = HDF5Utils.cleanup_old_files(str(tmp_path))

        assert result == 0

    def test_cleanup_keeps_recent_files(self, tmp_path):
        """Test cleanup keeps the most recent files."""
        # Create 5 files
        for i in range(5):
            filepath = tmp_path / f"network_{i}.h5"
            filepath.touch()
            time.sleep(0.01)  # Ensure different modification times

        result = HDF5Utils.cleanup_old_files(str(tmp_path), keep_count=5)

        assert result == 0
        assert len(list(tmp_path.glob("*.h5"))) == 5

    def test_cleanup_deletes_old_files(self, tmp_path):
        """Test cleanup deletes old files beyond keep_count."""
        # Create 5 files with different modification times
        files = []
        for i in range(5):
            filepath = tmp_path / f"network_{i}.h5"
            filepath.touch()
            files.append(filepath)
            time.sleep(0.02)  # Ensure different modification times

        result = HDF5Utils.cleanup_old_files(str(tmp_path), keep_count=2)

        assert result == 3
        remaining = list(tmp_path.glob("*.h5"))
        assert len(remaining) == 2
        # The 2 most recent files should remain
        remaining_names = {f.name for f in remaining}
        assert "network_4.h5" in remaining_names
        assert "network_3.h5" in remaining_names

    def test_cleanup_only_affects_hdf5_files(self, tmp_path):
        """Test cleanup only deletes HDF5 files."""
        # Create mixed file types
        (tmp_path / "network.h5").touch()
        (tmp_path / "network.hdf5").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "data.json").touch()

        result = HDF5Utils.cleanup_old_files(str(tmp_path), keep_count=0)

        # Only HDF5 files should be deleted
        assert result == 2
        assert (tmp_path / "readme.txt").exists()
        assert (tmp_path / "data.json").exists()

    def test_cleanup_with_keep_count_zero(self, tmp_path):
        """Test cleanup with keep_count=0 deletes all HDF5 files."""
        for i in range(3):
            (tmp_path / f"network_{i}.h5").touch()

        result = HDF5Utils.cleanup_old_files(str(tmp_path), keep_count=0)

        assert result == 3
        assert len(list(tmp_path.glob("*.h5"))) == 0

    def test_cleanup_ignores_subdirectories(self, tmp_path):
        """Test cleanup ignores subdirectories with .h5 in name."""
        # Create a subdirectory (should be ignored even if named .h5)
        subdir = tmp_path / "subdir.h5"
        subdir.mkdir()
        (subdir / "file.txt").touch()

        # Create actual files
        (tmp_path / "network.h5").touch()

        result = HDF5Utils.cleanup_old_files(str(tmp_path), keep_count=0)

        assert result == 1
        assert subdir.exists()  # Directory should not be deleted


class TestValidateNetworkFile:
    """Tests for HDF5Utils.validate_network_file()."""

    def test_validate_delegates_to_serializer(self, tmp_path, mock_serializer):
        """Test that validate_network_file delegates to serializer."""
        filepath = str(tmp_path / "test.h5")
        expected_result = {"valid": True, "format": "juniper.cascor"}
        mock_serializer.verify_saved_network.return_value = expected_result

        result = HDF5Utils.validate_network_file(filepath)

        mock_serializer.verify_saved_network.assert_called_once_with(filepath)
        assert result == expected_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
