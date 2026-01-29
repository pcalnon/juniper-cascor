#!/usr/bin/env python
"""
Unit tests for snapshots/snapshot_cli.py module.
Tests CLI functions for HDF5 network snapshot management.
"""

import os
import sys

# Add parent directories to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from unittest.mock import MagicMock, mock_open, patch

import pytest


@pytest.mark.unit
class TestSaveNetworkSnapshot:
    """Tests for save_network_snapshot function."""

    def test_save_network_snapshot_success(self, tmp_path):
        """Test successful network snapshot save."""
        output_file = str(tmp_path / "test_snapshot.h5")

        with patch("snapshots.snapshot_cli._object_to_file") as mock_save:
            mock_save.return_value = True
            from snapshots.snapshot_cli import save_network_snapshot

            result = save_network_snapshot("network.pkl", output_file, include_training=False)

            assert result is True
            mock_save.assert_called_once_with(output_file, False)

    def test_save_network_snapshot_with_training(self, tmp_path):
        """Test save with training history included."""
        output_file = str(tmp_path / "test_snapshot.h5")

        with patch("snapshots.snapshot_cli._object_to_file") as mock_save:
            mock_save.return_value = True
            from snapshots.snapshot_cli import save_network_snapshot

            result = save_network_snapshot("network.pkl", output_file, include_training=True)

            mock_save.assert_called_once_with(output_file, True)

    def test_save_network_snapshot_failure(self, tmp_path):
        """Test save network snapshot with exception."""
        output_file = str(tmp_path / "test_snapshot.h5")

        with patch("snapshots.snapshot_cli._object_to_file") as mock_save:
            mock_save.side_effect = Exception("Save error")
            from snapshots.snapshot_cli import save_network_snapshot

            result = save_network_snapshot("network.pkl", output_file, include_training=False)

            assert result is False


@pytest.mark.unit
class TestObjectToFile:
    """Tests for _object_to_file helper function."""

    def test_object_to_file_success(self, tmp_path):
        """Test successful object to file save."""
        output_file = str(tmp_path / "test.h5")

        with patch("cascade_correlation.cascade_correlation.CascadeCorrelationNetwork") as mock_network_class, patch("cascade_correlation.cascade_correlation_config.cascade_correlation_config.CascadeCorrelationConfig") as mock_config_class, patch("snapshots.snapshot_cli._get_saved_snapshot_file_data") as mock_get_data:

            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            mock_network = MagicMock()
            mock_network.save_to_hdf5.return_value = True
            mock_network_class.return_value = mock_network

            mock_get_data.return_value = True

            from snapshots.snapshot_cli import _object_to_file

            result = _object_to_file(output_file, include_training=True)

            assert result is True

    def test_object_to_file_save_failure(self, tmp_path):
        """Test object to file when save fails."""
        output_file = str(tmp_path / "test.h5")

        with patch("cascade_correlation.cascade_correlation.CascadeCorrelationNetwork") as mock_network_class, patch("cascade_correlation.cascade_correlation_config.cascade_correlation_config.CascadeCorrelationConfig") as mock_config_class:

            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            mock_network = MagicMock()
            mock_network.save_to_hdf5.return_value = False
            mock_network_class.return_value = mock_network

            from snapshots.snapshot_cli import _object_to_file

            result = _object_to_file(output_file, include_training=False)

            assert result is False


@pytest.mark.unit
class TestGetSavedSnapshotFileData:
    """Tests for _get_saved_snapshot_file_data function."""

    def test_get_saved_snapshot_file_data(self, tmp_path):
        """Test get saved snapshot file data."""
        output_file = str(tmp_path / "test.h5")

        with patch("snapshots.snapshot_cli.HDF5Utils") as mock_utils:
            mock_utils.get_file_info.return_value = {
                "size_mb": 1.5,
                "groups": ["group1", "group2"],
                "datasets": ["ds1", "ds2", "ds3"],
            }

            from snapshots.snapshot_cli import _get_saved_snapshot_file_data

            result = _get_saved_snapshot_file_data(output_file, True)

            assert result is True
            mock_utils.get_file_info.assert_called_once_with(output_file)


@pytest.mark.unit
class TestLoadNetworkSnapshot:
    """Tests for load_network_snapshot function."""

    def test_load_network_snapshot_success(self, tmp_path):
        """Test successful network snapshot load."""
        snapshot_file = str(tmp_path / "test_snapshot.h5")

        with patch("snapshots.snapshot_cli.CascadeHDF5Serializer") as mock_serializer_class, patch("cascade_correlation.cascade_correlation.CascadeCorrelationNetwork") as mock_network_class:

            mock_serializer = MagicMock()
            mock_serializer.verify_saved_network.return_value = {
                "valid": True,
                "network_uuid": "test-uuid",
                "input_size": 2,
                "num_hidden_units": 5,
                "output_size": 1,
                "created": "2024-01-01",
                "format": "cascade_correlation",
                "format_version": "1.0",
            }
            mock_serializer_class.return_value = mock_serializer

            mock_network = MagicMock()
            mock_network.input_size = 2
            mock_network.output_size = 1
            mock_network.hidden_units = [1, 2, 3, 4, 5]
            mock_network.activation_function_name = "tanh"
            mock_network_class.load_from_hdf5.return_value = mock_network

            from snapshots.snapshot_cli import load_network_snapshot

            result = load_network_snapshot(snapshot_file)

            assert result is True

    def test_load_network_snapshot_invalid_file(self, tmp_path):
        """Test load with invalid snapshot file."""
        snapshot_file = str(tmp_path / "invalid.h5")

        with patch("snapshots.snapshot_cli.CascadeHDF5Serializer") as mock_serializer_class:
            mock_serializer = MagicMock()
            mock_serializer.verify_saved_network.return_value = {
                "valid": False,
                "error": "Invalid file format",
            }
            mock_serializer_class.return_value = mock_serializer

            from snapshots.snapshot_cli import load_network_snapshot

            result = load_network_snapshot(snapshot_file)

            assert result is False

    def test_load_network_snapshot_load_failure(self, tmp_path):
        """Test load when network load returns None."""
        snapshot_file = str(tmp_path / "test_snapshot.h5")

        with patch("snapshots.snapshot_cli.CascadeHDF5Serializer") as mock_serializer_class, patch("cascade_correlation.cascade_correlation.CascadeCorrelationNetwork") as mock_network_class:

            mock_serializer = MagicMock()
            mock_serializer.verify_saved_network.return_value = {
                "valid": True,
                "network_uuid": "test-uuid",
            }
            mock_serializer_class.return_value = mock_serializer

            mock_network_class.load_from_hdf5.return_value = None

            from snapshots.snapshot_cli import load_network_snapshot

            result = load_network_snapshot(snapshot_file)

            assert result is False

    def test_load_network_snapshot_with_output(self, tmp_path):
        """Test load with output file specified."""
        snapshot_file = str(tmp_path / "test_snapshot.h5")
        output_file = str(tmp_path / "output.h5")

        with patch("snapshots.snapshot_cli.CascadeHDF5Serializer") as mock_serializer_class, patch("cascade_correlation.cascade_correlation.CascadeCorrelationNetwork") as mock_network_class:

            mock_serializer = MagicMock()
            mock_serializer.verify_saved_network.return_value = {"valid": True}
            mock_serializer_class.return_value = mock_serializer

            mock_network = MagicMock()
            mock_network.input_size = 2
            mock_network.output_size = 1
            mock_network.hidden_units = []
            mock_network.activation_function_name = "tanh"
            mock_network.save_to_hdf5.return_value = True
            mock_network_class.load_from_hdf5.return_value = mock_network

            from snapshots.snapshot_cli import load_network_snapshot

            result = load_network_snapshot(snapshot_file, output_file)

            assert result is True
            mock_network.save_to_hdf5.assert_called_once_with(output_file)

    def test_load_network_snapshot_exception(self, tmp_path):
        """Test load with exception."""
        snapshot_file = str(tmp_path / "test.h5")

        with patch("snapshots.snapshot_cli.CascadeHDF5Serializer") as mock_serializer_class:
            mock_serializer_class.side_effect = Exception("Load error")

            from snapshots.snapshot_cli import load_network_snapshot

            result = load_network_snapshot(snapshot_file)

            assert result is False


@pytest.mark.unit
class TestListSnapshots:
    """Tests for list_snapshots function."""

    def test_list_snapshots_with_files(self, tmp_path):
        """Test listing snapshots with valid files."""
        directory = str(tmp_path)

        with patch("snapshots.snapshot_cli.HDF5Utils") as mock_utils:
            mock_utils.list_networks_in_directory.return_value = [
                {
                    "filename": "network1.h5",
                    "file_size": 1024 * 1024,
                    "input_size": 2,
                    "output_size": 1,
                    "num_hidden_units": 5,
                    "format_version": "1.0",
                    "created": "2024-01-01 12:00:00",
                },
                {
                    "filename": "network2.h5",
                    "file_size": 2 * 1024 * 1024,
                    "input_size": 4,
                    "output_size": 2,
                    "num_hidden_units": 10,
                    "format_version": "1.0",
                    "created": "2024-01-02 12:00:00",
                },
            ]

            from snapshots.snapshot_cli import list_snapshots

            list_snapshots(directory)

            mock_utils.list_networks_in_directory.assert_called_once_with(directory)

    def test_list_snapshots_empty_directory(self, tmp_path):
        """Test listing snapshots in empty directory."""
        directory = str(tmp_path)

        with patch("snapshots.snapshot_cli.HDF5Utils") as mock_utils:
            mock_utils.list_networks_in_directory.return_value = []

            from snapshots.snapshot_cli import list_snapshots

            list_snapshots(directory)

    def test_list_snapshots_exception(self, tmp_path):
        """Test listing snapshots with exception."""
        directory = str(tmp_path)

        with patch("snapshots.snapshot_cli.HDF5Utils") as mock_utils:
            mock_utils.list_networks_in_directory.side_effect = Exception("List error")

            from snapshots.snapshot_cli import list_snapshots

            list_snapshots(directory)


@pytest.mark.unit
class TestVerifySnapshot:
    """Tests for verify_snapshot function."""

    def test_verify_snapshot_valid(self, tmp_path):
        """Test verifying a valid snapshot."""
        filepath = str(tmp_path / "valid.h5")

        with patch("snapshots.snapshot_cli.CascadeHDF5Serializer") as mock_serializer_class:
            mock_serializer = MagicMock()
            mock_serializer.verify_saved_network.return_value = {
                "valid": True,
                "format": "cascade_correlation",
                "format_version": "1.0",
                "network_uuid": "test-uuid",
                "input_size": 2,
                "num_hidden_units": 5,
                "output_size": 1,
                "activation_function": "tanh",
                "file_size": 1024 * 1024,
                "created": "2024-01-01",
                "has_history": True,
                "has_mp": False,
                "has_data": True,
            }
            mock_serializer_class.return_value = mock_serializer

            from snapshots.snapshot_cli import verify_snapshot

            result = verify_snapshot(filepath)

            assert result is True

    def test_verify_snapshot_invalid(self, tmp_path):
        """Test verifying an invalid snapshot."""
        filepath = str(tmp_path / "invalid.h5")

        with patch("snapshots.snapshot_cli.CascadeHDF5Serializer") as mock_serializer_class:
            mock_serializer = MagicMock()
            mock_serializer.verify_saved_network.return_value = {
                "valid": False,
                "error": "Invalid format",
            }
            mock_serializer_class.return_value = mock_serializer

            from snapshots.snapshot_cli import verify_snapshot

            result = verify_snapshot(filepath)

            assert result is False

    def test_verify_snapshot_exception(self, tmp_path):
        """Test verifying with exception."""
        filepath = str(tmp_path / "test.h5")

        with patch("snapshots.snapshot_cli.CascadeHDF5Serializer") as mock_serializer_class:
            mock_serializer_class.side_effect = Exception("Verify error")

            from snapshots.snapshot_cli import verify_snapshot

            result = verify_snapshot(filepath)

            assert result is False


@pytest.mark.unit
class TestCompareSnapshots:
    """Tests for compare_snapshots function."""

    def test_compare_snapshots_comparable(self, tmp_path):
        """Test comparing two comparable snapshots."""
        filepath1 = str(tmp_path / "network1.h5")
        filepath2 = str(tmp_path / "network2.h5")

        with patch("snapshots.snapshot_cli.HDF5Utils") as mock_utils:
            mock_utils.compare_networks.return_value = {
                "comparable": True,
                "same_architecture": True,
                "same_hidden_units": True,
                "same_activation": True,
            }

            from snapshots.snapshot_cli import compare_snapshots

            result = compare_snapshots(filepath1, filepath2)

            assert result is True

    def test_compare_snapshots_different_architecture(self, tmp_path):
        """Test comparing snapshots with different architectures."""
        filepath1 = str(tmp_path / "network1.h5")
        filepath2 = str(tmp_path / "network2.h5")

        with patch("snapshots.snapshot_cli.HDF5Utils") as mock_utils:
            mock_utils.compare_networks.return_value = {
                "comparable": True,
                "same_architecture": False,
                "same_hidden_units": False,
                "same_activation": True,
                "architecture_diff": {
                    "input_size": (2, 4),
                    "output_size": (1, 2),
                    "num_hidden_units": (5, 10),
                    "activation_function": ("tanh", "tanh"),
                },
            }

            from snapshots.snapshot_cli import compare_snapshots

            result = compare_snapshots(filepath1, filepath2)

            assert result is True

    def test_compare_snapshots_not_comparable(self, tmp_path):
        """Test comparing non-comparable snapshots."""
        filepath1 = str(tmp_path / "network1.h5")
        filepath2 = str(tmp_path / "invalid.h5")

        with patch("snapshots.snapshot_cli.HDF5Utils") as mock_utils:
            mock_utils.compare_networks.return_value = {
                "comparable": False,
                "error": "Invalid file",
            }

            from snapshots.snapshot_cli import compare_snapshots

            result = compare_snapshots(filepath1, filepath2)

            assert result is False

    def test_compare_snapshots_exception(self, tmp_path):
        """Test comparing with exception."""
        filepath1 = str(tmp_path / "network1.h5")
        filepath2 = str(tmp_path / "network2.h5")

        with patch("snapshots.snapshot_cli.HDF5Utils") as mock_utils:
            mock_utils.compare_networks.side_effect = Exception("Compare error")

            from snapshots.snapshot_cli import compare_snapshots

            result = compare_snapshots(filepath1, filepath2)

            assert result is False


@pytest.mark.unit
class TestCleanupOldSnapshots:
    """Tests for cleanup_old_snapshots function."""

    def test_cleanup_old_snapshots_success(self, tmp_path):
        """Test successful cleanup of old snapshots."""
        directory = str(tmp_path)

        with patch("snapshots.snapshot_cli.HDF5Utils") as mock_utils:
            mock_utils.cleanup_old_files.return_value = 5

            from snapshots.snapshot_cli import cleanup_old_snapshots

            result = cleanup_old_snapshots(directory, keep_count=10)

            assert result is True
            mock_utils.cleanup_old_files.assert_called_once_with(directory, 10)

    def test_cleanup_old_snapshots_no_files_deleted(self, tmp_path):
        """Test cleanup when no files are deleted."""
        directory = str(tmp_path)

        with patch("snapshots.snapshot_cli.HDF5Utils") as mock_utils:
            mock_utils.cleanup_old_files.return_value = 0

            from snapshots.snapshot_cli import cleanup_old_snapshots

            result = cleanup_old_snapshots(directory, keep_count=10)

            assert result is True

    def test_cleanup_old_snapshots_exception(self, tmp_path):
        """Test cleanup with exception."""
        directory = str(tmp_path)

        with patch("snapshots.snapshot_cli.HDF5Utils") as mock_utils:
            mock_utils.cleanup_old_files.side_effect = Exception("Cleanup error")

            from snapshots.snapshot_cli import cleanup_old_snapshots

            result = cleanup_old_snapshots(directory, keep_count=10)

            assert result is False


@pytest.mark.unit
class TestMainCLI:
    """Tests for main CLI entry point."""

    def test_main_no_command(self):
        """Test main with no command shows help."""
        with patch("sys.argv", ["snapshot_cli.py"]):
            from snapshots.snapshot_cli import main

            result = main()

            assert result == 1

    def test_main_save_command(self, tmp_path):
        """Test main with save command."""
        output_file = str(tmp_path / "output.h5")

        with patch("sys.argv", ["snapshot_cli.py", "save", "network.pkl", output_file]), patch("snapshots.snapshot_cli.save_network_snapshot") as mock_save:
            mock_save.return_value = True

            from snapshots.snapshot_cli import main

            result = main()

            assert result == 0
            mock_save.assert_called_once_with("network.pkl", output_file, False)

    def test_main_save_command_with_training(self, tmp_path):
        """Test main with save command including training."""
        output_file = str(tmp_path / "output.h5")

        with patch("sys.argv", ["snapshot_cli.py", "save", "network.pkl", output_file, "--include-training"]), patch("snapshots.snapshot_cli.save_network_snapshot") as mock_save:
            mock_save.return_value = True

            from snapshots.snapshot_cli import main

            result = main()

            mock_save.assert_called_once_with("network.pkl", output_file, True)

    def test_main_load_command(self, tmp_path):
        """Test main with load command."""
        snapshot_file = str(tmp_path / "snapshot.h5")

        with patch("sys.argv", ["snapshot_cli.py", "load", snapshot_file]), patch("snapshots.snapshot_cli.load_network_snapshot") as mock_load:
            mock_load.return_value = True

            from snapshots.snapshot_cli import main

            result = main()

            assert result == 0
            mock_load.assert_called_once_with(snapshot_file, None)

    def test_main_load_command_with_output(self, tmp_path):
        """Test main with load command and output file."""
        snapshot_file = str(tmp_path / "snapshot.h5")
        output_file = str(tmp_path / "output.h5")

        with patch("sys.argv", ["snapshot_cli.py", "load", snapshot_file, "--output", output_file]), patch("snapshots.snapshot_cli.load_network_snapshot") as mock_load:
            mock_load.return_value = True

            from snapshots.snapshot_cli import main

            result = main()

            mock_load.assert_called_once_with(snapshot_file, output_file)

    def test_main_list_command(self, tmp_path):
        """Test main with list command."""
        directory = str(tmp_path)

        with patch("sys.argv", ["snapshot_cli.py", "list", directory]), patch("snapshots.snapshot_cli.list_snapshots") as mock_list:
            from snapshots.snapshot_cli import main

            result = main()

            assert result == 0
            mock_list.assert_called_once_with(directory)

    def test_main_verify_command(self, tmp_path):
        """Test main with verify command."""
        filepath = str(tmp_path / "snapshot.h5")

        with patch("sys.argv", ["snapshot_cli.py", "verify", filepath]), patch("snapshots.snapshot_cli.verify_snapshot") as mock_verify:
            mock_verify.return_value = True

            from snapshots.snapshot_cli import main

            result = main()

            assert result == 0
            mock_verify.assert_called_once_with(filepath)

    def test_main_compare_command(self, tmp_path):
        """Test main with compare command."""
        file1 = str(tmp_path / "network1.h5")
        file2 = str(tmp_path / "network2.h5")

        with patch("sys.argv", ["snapshot_cli.py", "compare", file1, file2]), patch("snapshots.snapshot_cli.compare_snapshots") as mock_compare:
            mock_compare.return_value = True

            from snapshots.snapshot_cli import main

            result = main()

            assert result == 0
            mock_compare.assert_called_once_with(file1, file2)

    def test_main_cleanup_command(self, tmp_path):
        """Test main with cleanup command."""
        directory = str(tmp_path)

        with patch("sys.argv", ["snapshot_cli.py", "cleanup", directory]), patch("snapshots.snapshot_cli.cleanup_old_snapshots") as mock_cleanup:
            mock_cleanup.return_value = True

            from snapshots.snapshot_cli import main

            result = main()

            assert result == 0
            mock_cleanup.assert_called_once_with(directory, 10)

    def test_main_cleanup_command_with_keep(self, tmp_path):
        """Test main with cleanup command and keep count."""
        directory = str(tmp_path)

        with patch("sys.argv", ["snapshot_cli.py", "cleanup", directory, "--keep", "5"]), patch("snapshots.snapshot_cli.cleanup_old_snapshots") as mock_cleanup:
            mock_cleanup.return_value = True

            from snapshots.snapshot_cli import main

            result = main()

            mock_cleanup.assert_called_once_with(directory, 5)

    def test_main_command_failure(self, tmp_path):
        """Test main when command returns failure."""
        filepath = str(tmp_path / "snapshot.h5")

        with patch("sys.argv", ["snapshot_cli.py", "verify", filepath]), patch("snapshots.snapshot_cli.verify_snapshot") as mock_verify:
            mock_verify.return_value = False

            from snapshots.snapshot_cli import main

            result = main()

            assert result == 1

    def test_main_unexpected_exception(self, tmp_path):
        """Test main with unexpected exception."""
        directory = str(tmp_path)

        with patch("sys.argv", ["snapshot_cli.py", "list", directory]), patch("snapshots.snapshot_cli.list_snapshots") as mock_list:
            mock_list.side_effect = Exception("Unexpected error")

            from snapshots.snapshot_cli import main

            result = main()

            assert result == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
