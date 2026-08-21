#!/usr/bin/env python
"""
Utility functions for HDF5 operations and file management.
Consolidated from hdf5_utilities.py with fixes for string handling.
"""

import datetime
import os
import pathlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py

from log_config.logger.logger import Logger

# from cascade_correlation.hdf5.common import read_str_attr
# from hdf5.common import read_str_attr
from snapshots.snapshot_common import read_str_attr

# from hdf5.serializer import CascadeHDF5Serializer
from snapshots.snapshot_serializer import CascadeHDF5Serializer


class HDF5Utils:
    """Utility functions for HDF5 file operations."""

    @staticmethod
    def create_backup(filepath: str, backup_dir: Optional[str] = None) -> str:
        """
        Create a backup of an HDF5 file.

        Args:
            filepath: Path to original file
            backup_dir: Directory for backup (default: same directory as original)

        Returns:
            Path to backup file

        Raises:
            FileNotFoundError: If original file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Original file not found: {filepath}")

        # Determine backup location
        if backup_dir is None:
            backup_dir = os.path.dirname(filepath)

        # Create backup filename with timestamp
        base_name = Path(filepath).stem
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{base_name}_backup_{timestamp}.h5"
        backup_path = os.path.join(backup_dir, backup_filename)

        # Ensure backup directory exists
        Path(backup_dir).mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(filepath, backup_path)

        Logger.info(f"HDF5Utils: Created backup at {backup_path}")
        return backup_path

    @staticmethod
    def list_hdf5_files(directory: Union[str, Path]) -> List[Path]:
        """
        List all HDF5 files (.h5 / .hdf5) in a directory.

        Args:
            directory: Directory to search

        Returns:
            Sorted list of HDF5 file paths (empty if the directory does not exist)
        """
        if not os.path.isdir(directory):
            return []
        return sorted(Path(directory, filename) for filename in os.listdir(directory) if filename.endswith((".h5", ".hdf5")))

    @staticmethod
    def list_networks_in_directory(directory: str) -> List[Dict[str, Any]]:
        """
        List all valid cascade correlation network files in a directory.

        Args:
            directory: Directory to search

        Returns:
            List of dictionaries with file information
        """
        networks = []
        serializer = CascadeHDF5Serializer()

        if not os.path.exists(directory):
            return networks

        for filename in os.listdir(directory):
            if filename.endswith((".h5", ".hdf5")):
                filepath = os.path.join(directory, filename)
                try:
                    verification = serializer.verify_saved_network(filepath)
                    if verification.get("valid", False):
                        verification["filename"] = filename
                        verification["filepath"] = filepath
                        networks.append(verification)
                except Exception as e:
                    Logger.warning(f"HDF5Utils: Could not verify {filepath}: {e}")

        return networks

    @staticmethod
    def compare_networks(filepath1: str, filepath2: str) -> Dict[str, Any]:
        """
        Compare two saved networks and return differences.

        Args:
            filepath1: Path to first network file
            filepath2: Path to second network file

        Returns:
            Dictionary with comparison results
        """
        serializer = CascadeHDF5Serializer()

        # Verify both files
        info1 = serializer.verify_saved_network(filepath1)
        info2 = serializer.verify_saved_network(filepath2)

        if not (info1.get("valid") and info2.get("valid")):
            return {"comparable": False, "error": "One or both files are invalid", "file1_valid": info1.get("valid", False), "file2_valid": info2.get("valid", False)}

        # Compare key attributes
        comparison = {
            "comparable": True,
            "same_architecture": (info1.get("input_size", 0) == info2.get("input_size", 0) and info1.get("output_size", 0) == info2.get("output_size", 0)),
            "same_hidden_units": info1.get("num_hidden_units", 0) == info2.get("num_hidden_units", 0),
            "same_activation": info1.get("activation_function", "") == info2.get("activation_function", ""),
            "architecture_diff": {"input_size": (info1.get("input_size", 0), info2.get("input_size", 0)), "output_size": (info1.get("output_size", 0), info2.get("output_size", 0)), "num_hidden_units": (info1.get("num_hidden_units", 0), info2.get("num_hidden_units", 0)), "activation_function": (info1.get("activation_function", ""), info2.get("activation_function", ""))},
            "file1_info": info1,
            "file2_info": info2,
        }

        Logger.debug(f"HDF5Utils: Compared networks - same architecture: {comparison['same_architecture']}")
        return comparison

    @staticmethod
    def compress_hdf5_file(input_filepath: str, output_filepath: str, compression: str = "gzip", compression_opts: int = 9) -> bool:
        """
        Recompress an HDF5 file with different compression settings.

        Args:
            input_filepath: Path to input file
            output_filepath: Path to output file
            compression: Compression method
            compression_opts: Compression level

        Returns:
            Success status
        """
        try:
            with h5py.File(input_filepath, "r") as input_file:
                with h5py.File(output_filepath, "w") as output_file:

                    def copy_group(src_group, dst_group):
                        """Recursively copy groups with compression."""
                        for key in src_group.keys():
                            if isinstance(src_group[key], h5py.Group):
                                new_group = dst_group.create_group(key)
                                # Copy attributes
                                for attr_key, attr_val in src_group[key].attrs.items():
                                    new_group.attrs[attr_key] = attr_val
                                copy_group(src_group[key], new_group)
                            else:
                                # Copy dataset with new compression
                                dataset = src_group[key]
                                new_dataset = dst_group.create_dataset(key, data=dataset[:], compression=compression, compression_opts=compression_opts)
                                # Copy attributes
                                for attr_key, attr_val in dataset.attrs.items():
                                    new_dataset.attrs[attr_key] = attr_val

                    # Copy root attributes
                    for attr_key, attr_val in input_file.attrs.items():
                        output_file.attrs[attr_key] = attr_val

                    # Copy all groups
                    copy_group(input_file, output_file)

            Logger.info(f"HDF5Utils: Successfully compressed {input_filepath} to {output_filepath}")
            return True

        except Exception as e:
            Logger.error(f"HDF5Utils: Error compressing file: {e}")
            return False

    @staticmethod
    def get_file_info(filepath: str) -> Dict[str, Any]:
        """
        Get detailed information about an HDF5 file.

        Args:
            filepath: Path to HDF5 file

        Returns:
            Dictionary with file information
        """
        if not os.path.exists(filepath):
            return {"exists": False, "error": "File not found"}

        try:
            info = {"exists": True, "filepath": filepath, "size_bytes": os.path.getsize(filepath), "size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2), "modified_time": datetime.datetime.fromtimestamp(os.path.getmtime(filepath)), "groups": [], "datasets": [], "attributes": {}}

            with h5py.File(filepath, "r") as hdf5_file:

                def explore_group(group, path=""):
                    """Recursively explore HDF5 structure."""
                    for key in group.keys():
                        full_path = f"{path}/{key}" if path else key

                        if isinstance(group[key], h5py.Group):
                            # Safely read group attributes
                            group_attrs = {}
                            for attr_key, attr_val in group[key].attrs.items():
                                group_attrs[attr_key] = read_str_attr(group[key], attr_key, str(attr_val))

                            info["groups"].append({"path": full_path, "attributes": group_attrs})
                            explore_group(group[key], full_path)
                        else:
                            dataset = group[key]

                            # Safely read dataset attributes
                            dataset_attrs = {}
                            for attr_key, attr_val in dataset.attrs.items():
                                dataset_attrs[attr_key] = read_str_attr(dataset, attr_key, str(attr_val))

                            info["datasets"].append({"path": full_path, "shape": dataset.shape, "dtype": str(dataset.dtype), "size_bytes": dataset.size * dataset.dtype.itemsize if hasattr(dataset.dtype, "itemsize") else dataset.nbytes, "compression": dataset.compression, "attributes": dataset_attrs})

                # Safely read root attributes
                root_attrs = {}
                for attr_key, attr_val in hdf5_file.attrs.items():
                    root_attrs[attr_key] = read_str_attr(hdf5_file, attr_key, str(attr_val))
                info["attributes"] = root_attrs

                # Explore structure
                explore_group(hdf5_file)

            return info

        except Exception as e:
            Logger.warning(f"HDF5Utils: Error reading file info for {filepath}: {e}")
            return {"exists": True, "error": str(e)}

    @staticmethod
    def validate_network_file(filepath: str) -> Dict[str, Any]:
        """
        Validate a network file and return detailed validation results.

        Args:
            filepath: Path to HDF5 file to validate

        Returns:
            Dictionary with validation results
        """
        serializer = CascadeHDF5Serializer()
        return serializer.verify_saved_network(filepath)

    @staticmethod
    def get_network_summary(filepath: str) -> Optional[Dict[str, Any]]:
        """
        Get a quick summary of a network file.

        Args:
            filepath: Path to HDF5 file

        Returns:
            Dictionary with network summary or None if invalid
        """
        validation = HDF5Utils.validate_network_file(filepath)
        if not validation.get("valid", False):
            return None

        file_info = HDF5Utils.get_file_info(filepath)

        summary = {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "size_mb": file_info.get("size_mb", 0),
            "modified": file_info.get("modified_time", "unknown"),
            "uuid": validation.get("network_uuid", "unknown"),
            "input_size": validation.get("input_size", 0),
            "output_size": validation.get("output_size", 0),
            "num_hidden_units": validation.get("num_hidden_units", 0),
            "activation_function": validation.get("activation_function", "unknown"),
            "format_version": validation.get("format_version", "unknown"),
            "has_training_history": validation.get("has_history", False),
            "has_multiprocessing": validation.get("has_mp", False),
        }
        Logger.debug(f"HDF5Utils: Network summary for {filepath}: {summary}")

        return summary

    @staticmethod
    def _is_shared_snapshot_root(directory: str) -> bool:
        """True when ``directory`` is the host's shared snapshot root.

        That root is a project ASSET store: it holds tens of thousands of models that
        outlive every stack, is shared by the direct CLI, the service and the container
        alike, and is captured by the whole-tree offline backup. Age-based retention over
        it is a policy decision the owner has explicitly reserved -- see the snapshot
        lifecycle design's Phase 4, which writes no retention policy until the archive can
        say what each file IS. Deleting the 27,886 oldest of 27,896 files because a CLI
        default said ``--keep 10`` is not that policy.
        """
        try:
            resolved = pathlib.Path(directory).expanduser().resolve()
        except (OSError, ValueError):
            return False
        override = os.environ.get("JUNIPER_CASCOR_SNAPSHOTS_DIR", "").strip()
        candidates = []
        if override:
            candidates.append(pathlib.Path(override).expanduser())
        # <repo>/cascor-snapshots -- this file is src/snapshots/snapshot_utils.py
        candidates.append(pathlib.Path(__file__).resolve().parents[2] / "cascor-snapshots")
        for candidate in candidates:
            try:
                if resolved == candidate.expanduser().resolve():
                    return True
            except (OSError, ValueError):  # pragma: no cover - defensive
                continue
        return False

    @staticmethod
    def cleanup_old_files(directory: str, keep_count: int = 10, *, dry_run: bool = True, allow_shared_root: bool = False) -> int:
        """
        Clean up old network files, keeping only the most recent ones.

        Args:
            directory: Directory containing network files
            keep_count: Number of most recent files to keep
            dry_run: When True (the DEFAULT), report what would be deleted and delete
                nothing. Deletion must be asked for explicitly.
            allow_shared_root: Permit operating on the host's shared snapshot root.
                Refused by default.

        Returns:
            Number of files deleted -- or, under ``dry_run``, the number that WOULD be.

        Two guards, both added because this function is reachable from a shipped CLI
        (``snapshot_cli cleanup <dir> [--keep N]``) with no confirmation of any kind:

        * **Dry-run by default.** This matches the safety contract the rest of the
          ecosystem's destructive tooling already uses (``--dry-run`` default, ``--yes``
          required). An age-ordered ``os.remove`` loop with a ``keep_count=10`` default is
          not something that should happen because an argument was omitted.
        * **Refuses the shared snapshot root.** ``mtime`` is not creation time in that
          archive -- a copy reset every timestamp, so files named 2025-10 carry 2026
          mtimes -- which makes "keep the 10 most recent" not merely dangerous but
          *wrong*: it sorts by a field that does not mean what it appears to mean.

        Neither guard restricts the intended use (pruning a per-run ``RUN_DIR/snapshots``);
        both require an explicit opt-in for the case that destroys research data.
        """
        if not os.path.exists(directory):
            return 0

        if HDF5Utils._is_shared_snapshot_root(directory) and not allow_shared_root:
            Logger.warning(f"HDF5Utils: refusing to clean up {directory}: it is the SHARED snapshot root, a " "project asset store protected from deletion. mtime is not creation time in that " "archive, so 'keep the N most recent' would not even select what it claims to. " "Pass allow_shared_root=True only with a ratified retention policy.")
            return 0

        # Find all HDF5 files
        hdf5_files = []
        for filename in os.listdir(directory):
            if filename.endswith((".h5", ".hdf5")):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    mtime = os.path.getmtime(filepath)
                    hdf5_files.append((filepath, mtime))

        # Sort by modification time (newest first)
        hdf5_files.sort(key=lambda x: x[1], reverse=True)

        # Delete old files
        deleted_count = 0
        for filepath, _ in hdf5_files[keep_count:]:
            if dry_run:
                deleted_count += 1
                Logger.info(f"HDF5Utils: [dry-run] would delete old file: {filepath}")
                continue
            try:
                os.remove(filepath)
                deleted_count += 1
                Logger.info(f"HDF5Utils: Deleted old file: {filepath}")
            except Exception as e:
                Logger.warning(f"HDF5Utils: Could not delete {filepath}: {e}")

        return deleted_count
