#!/usr/bin/env python
"""
HDF5 Serializer for CascadeCorrelationNetwork objects.
Provides comprehensive state capture and restoration with full multiprocessing support.
"""

import datetime
import inspect
import io
import json
import multiprocessing as mp
import os
import pathlib as pl
import pickle  # trunk-ignore(bandit/B403)
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# SEC-11: allowlist for the snapshot RNG-state unpickler. Only modules we
# know Python's ``random.getstate()`` payloads reference (plus their pickle
# helpers) and a tight set of builtin container types are permitted. Any
# other ``find_class`` lookup — including torch, numpy, or user code — is
# rejected so a crafted HDF5 file cannot smuggle an RCE-capable pickle
# payload into the snapshot restore path.
_SNAPSHOT_UNPICKLER_ALLOWED_MODULES = frozenset(
    {
        "random",
        "_random",
        "collections",
        "collections.abc",
        "_codecs",
        "copyreg",
    }
)
_SNAPSHOT_UNPICKLER_ALLOWED_BUILTINS = frozenset(
    {
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        "int",
        "float",
        "str",
        "bool",
        "bytes",
        "complex",
        "slice",
        "range",
        "type",
    }
)


class SnapshotUnpicklingError(pickle.UnpicklingError):
    """Raised when a snapshot pickle references a class outside the allowlist."""


class _SnapshotRestrictedUnpickler(pickle.Unpickler):
    """Unpickler used to restore Python RNG state from HDF5 snapshots.

    Overrides ``find_class`` to fail closed on anything outside the
    allowlists above. This is the last line of defense for snapshot
    integrity: even a legitimately-signed snapshot file must still pass
    this check, so a compromised signing key or a file swapped on disk
    cannot escalate to arbitrary code execution.
    """

    def find_class(self, module: str, name: str):  # type: ignore[override]
        if module in _SNAPSHOT_UNPICKLER_ALLOWED_MODULES:
            return super().find_class(module, name)
        if module == "builtins" and name in _SNAPSHOT_UNPICKLER_ALLOWED_BUILTINS:
            return super().find_class(module, name)
        raise SnapshotUnpicklingError(f"Blocked unpickling of {module}.{name} — not in snapshot allowlist")


def _snapshot_restricted_loads(data: bytes):
    """Run ``pickle.loads`` equivalent against ``_SnapshotRestrictedUnpickler``."""
    return _SnapshotRestrictedUnpickler(io.BytesIO(data)).load()


import h5py
import numpy as np
import torch

# Add parent directories for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cascor_constants.constants_hdf5.constants_hdf5 import _HDF5_FORMAT_NAME_CURRENT, _HDF5_FORMAT_NAME_LEGACY
from log_config.logger.logger import Logger
from utils.activation import ActivationWithDerivative

from .snapshot_common import calculate_tensor_checksum, load_numpy_array, load_tensor, read_str_attr, read_str_dataset, save_numpy_array, save_tensor, verify_tensor_checksum, write_str_attr, write_str_dataset
from .snapshot_errors import SnapshotSaveError
from .snapshot_load_status import SNAPSHOT_ARCH_MISMATCH, SNAPSHOT_CORRUPT, SnapshotLoadResult
from .snapshot_load_status import absent as snapshot_absent
from .snapshot_load_status import arch_mismatch
from .snapshot_load_status import corrupt as snapshot_corrupt
from .snapshot_load_status import loaded as snapshot_loaded
from .snapshot_provenance import read_provenance, write_provenance


class CascadeHDF5Serializer:
    """
    Comprehensive HDF5 serialization system for CascadeCorrelationNetwork objects.

    Captures complete state including:
    - Network architecture and weights
    - Training history and statistics
    - Configuration parameters
    - Multiprocessing state
    - Hidden units and candidate pools

    Format Version: 2.0
    """

    def __init__(self, logger: Logger = None):
        """Initialize the HDF5 serializer."""
        self.logger = logger or Logger
        self.version = "2.0.0"
        self.format_version = "2"
        self.format_name = _HDF5_FORMAT_NAME_CURRENT

    def save_network(
        self,
        network,
        filepath: Union[str, Path],
        include_training_state: bool = False,
        include_training_data: bool = False,
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> bool:
        """
        Save a CascadeCorrelationNetwork to HDF5 format.

        Args:
            network: CascadeCorrelationNetwork instance to serialize
            filepath: Target file path for HDF5 file
            include_training_state: Whether to include training history
            include_training_data: Whether to include training datasets (excluded by default)
            compression: HDF5 compression method
            compression_opts: Compression level (0-9)

        Returns:
            bool: True on success.

        Raises:
            SnapshotSaveError: when the write fails, chaining the underlying
                exception so callers can surface the real reason (C1 / I-3 —
                pre-C1 every failure was swallowed into ``False`` and a failed
                save was indistinguishable from a missing network at the API
                route). Callers that want bool semantics catch it.
        """
        try:
            self.logger.info(f"CascadeHDF5Serializer: Saving network to {filepath}")

            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(filepath, "w") as hdf5_file:
                self._save_network_objects_helper(hdf5_file, network, compression, compression_opts)
                # D-C: stamp which run produced this model, from the process env the
                # launcher exports. Writes nothing when the run is unidentified, so
                # absence keeps meaning "unknown" rather than "failed to record".
                if write_provenance(hdf5_file):
                    self.logger.debug("CascadeHDF5Serializer: Recorded run provenance")
                # Save training history if requested
                if include_training_state:
                    self._save_training_history(hdf5_file, network, compression, compression_opts)

                # Save training data if explicitly requested (normally excluded)
                if include_training_data:
                    self._save_training_data(hdf5_file, network, compression, compression_opts)

            self.logger.info(f"CascadeHDF5Serializer: Successfully saved network to {filepath}")
            return True

        except Exception as e:
            # C1 (I-3): keep the ERROR + stacktrace logging, then raise a typed
            # error carrying the reason instead of collapsing to ``False`` — a
            # failed save must not be indistinguishable from "no network".
            self._log_exception_stacktrace("CascadeHDF5Serializer: Error saving network: ", e, False)
            raise SnapshotSaveError(f"{type(e).__name__}: {e}") from e

    def save_object(
        self,
        # objectify: any = None,
        objectify: Any = None,
        filepath: str = "./snapshots/object.h5",
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> bool:
        """
        Save a generic object to HDF5 format.
        Args:
            objectify: Object to serialize (should have similar interface to CascadeCorrelationNetwork)
            filepath: Target file path for HDF5 file
            compression: HDF5 compression method
            compression_opts: Compression level (0-9)
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"CascadeHDF5Serializer: Saving object to {filepath}")

            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(filepath, "w") as hdf5_file:
                # CASCOR-P0-004 FIX: Changed from _save_root_attributes (wrong arg count) to _save_network_objects_helper
                # OLD (TypeError - 4 args passed to 2-arg method):
                # self._save_root_attributes(hdf5_file, objectify, compression, compression_opts)
                self._save_network_objects_helper(hdf5_file, objectify, compression, compression_opts)
            self.logger.info(f"CascadeHDF5Serializer: Successfully saved object to {filepath}")
            return True

        except Exception as e:
            return self._log_exception_stacktrace("CascadeHDF5Serializer: Error saving object: ", e, False)

    def _save_root_attributes(self, hdf5_file: h5py.File, network) -> None:
        """Save root-level file attributes."""
        write_str_attr(hdf5_file, "format", self.format_name)
        write_str_attr(hdf5_file, "format_version", self.format_version)
        write_str_attr(hdf5_file, "serializer_version", self.version)
        write_str_attr(hdf5_file, "created", datetime.datetime.now().isoformat())
        # BUG-CC-04: read juniper-cascor version at runtime from package metadata.
        try:
            import importlib.metadata as _ilmd

            _juniper_version = _ilmd.version("juniper-cascor")
        except Exception:
            _juniper_version = "0.0.0-dev"
        write_str_attr(hdf5_file, "juniper_version", _juniper_version)
        self.logger.debug("CascadeHDF5Serializer: _save_root_attributes: Saved root attributes")

    def _save_metadata(self, hdf5_file: h5py.File, network) -> None:
        """Save metadata information."""
        meta_group = hdf5_file.create_group("meta")

        # Object metadata
        write_str_attr(meta_group, "uuid", str(network.get_uuid()))
        write_str_attr(meta_group, "creation_timestamp", datetime.datetime.now().isoformat())

        # Training state counters for resuming training
        meta_group.attrs["snapshot_counter"] = getattr(network, "snapshot_counter", 0)
        meta_group.attrs["current_epoch"] = getattr(network, "current_epoch", 0)
        meta_group.attrs["patience_counter"] = getattr(network, "patience_counter", 0)
        meta_group.attrs["best_value_loss"] = getattr(network, "best_value_loss", float("inf"))

        # Environment metadata
        write_str_attr(meta_group, "python_version", sys.version)
        write_str_attr(meta_group, "torch_version", torch.__version__)
        write_str_attr(meta_group, "h5py_version", h5py.__version__)
        self.logger.debug("CascadeHDF5Serializer: _save_metadata: Saved metadata with training counters")

    def _save_network_objects_helper(self, hdf5_file, arg1, compression, compression_opts):
        self._save_root_attributes(hdf5_file, arg1)
        self._save_metadata(hdf5_file, arg1)
        self._save_configuration(hdf5_file, arg1, compression, compression_opts)
        self._save_architecture(hdf5_file, arg1)
        self._save_parameters(hdf5_file, arg1, compression, compression_opts)
        self._save_hidden_units(hdf5_file, arg1, compression, compression_opts)
        self._save_random_state(hdf5_file, arg1, compression, compression_opts)
        self._save_multiprocessing_state(hdf5_file, arg1)

    def verify_saved_network(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Verify a saved network file and return summary information.

        Args:
            filepath: Path to HDF5 file to verify

        Returns:
            Dictionary with verification results and network summary
        """
        try:
            with h5py.File(filepath, "r") as hdf5_file:
                # Report WHICH check failed. ``_validate_format`` covers far more than
                # the format string — required groups, hidden-unit consistency, param
                # datasets — so a flat "Invalid format" pointed the operator at the one
                # thing that was usually fine.
                if (format_detail := self._validate_format_detail(hdf5_file)) is not None:
                    return {"valid": False, "error": format_detail}

                # Extract summary information
                summary = {
                    "valid": True,
                    "format": read_str_attr(hdf5_file, "format", "unknown"),
                    "format_version": read_str_attr(hdf5_file, "format_version", "unknown"),
                    "serializer_version": read_str_attr(hdf5_file, "serializer_version", "unknown"),
                    "created": read_str_attr(hdf5_file, "created", "unknown"),
                    "file_size": os.path.getsize(filepath),
                }

                # Get metadata if available
                if "meta" in hdf5_file:
                    meta_group = hdf5_file["meta"]
                    summary["network_uuid"] = read_str_attr(meta_group, "uuid", "unknown")
                    summary["python_version"] = read_str_attr(meta_group, "python_version", "unknown")
                    summary["torch_version"] = read_str_attr(meta_group, "torch_version", "unknown")

                # Get architecture if available
                if "arch" in hdf5_file:
                    arch_group = hdf5_file["arch"]
                    summary["input_size"] = arch_group.attrs.get("input_size", 0)
                    summary["output_size"] = arch_group.attrs.get("output_size", 0)
                    summary["num_hidden_units"] = arch_group.attrs.get("num_hidden_units", 0)
                    summary["activation_function"] = read_str_attr(arch_group, "activation_function_name", "unknown")

                # Check for optional sections
                optional_sections = ["history", "mp", "data"]
                for section in optional_sections:
                    summary[f"has_{section}"] = section in hdf5_file

                return summary

        except Exception as e:
            return {"valid": False, "error": str(e)}

    # CASCOR-P0-004 FIX: Removed duplicate _save_root_attributes and _save_metadata definitions
    # that existed here (lines 236-270). The canonical definitions are at lines 147-174.
    # Python would silently use these later definitions, making the earlier ones dead code.

    def _save_configuration(
        self,
        hdf5_file: h5py.File,
        network,
        compression: str,
        compression_opts: int,
    ) -> None:
        """Save configuration parameters."""
        config_group = hdf5_file.create_group("config")

        # Serialize config object to JSON
        config_dict = self._config_to_dict(network.config)
        config_json = json.dumps(config_dict, indent=2, default=str)

        # Save as UTF-8 string dataset
        write_str_dataset(
            config_group,
            "config_json",
            config_json,
            compression=compression,
            compression_opts=compression_opts,
        )

        # Save key parameters as attributes for quick access
        write_str_attr(config_group, "activation_function_name", network.activation_function_name)
        config_group.attrs["input_size"] = network.input_size
        config_group.attrs["output_size"] = network.output_size
        config_group.attrs["learning_rate"] = network.learning_rate
        config_group.attrs["candidate_learning_rate"] = network.candidate_learning_rate
        config_group.attrs["max_hidden_units"] = network.max_hidden_units
        config_group.attrs["correlation_threshold"] = network.correlation_threshold
        config_group.attrs["candidate_pool_size"] = network.candidate_pool_size
        config_group.attrs["patience"] = network.patience
        config_group.attrs["random_seed"] = network.random_seed

        # CAN-014 (Phase 6E Sprint A-5): persist the runtime-tunable training
        # params that ``update_params``' whitelist exposes so a snapshot
        # restore brings back the values the operator actually trained with.
        # Without this, ``epochs_max`` / ``max_iterations`` / etc. silently
        # revert to construction-time defaults on restore — defeating the
        # whole point of the wire-throughs in PRs #157, #158, #162. Each
        # field is read with ``getattr(..., default)`` so older networks
        # (or a partially-initialized network from a corner case) still
        # serialize cleanly. The load path mirrors the same ``getattr``
        # pattern in ``_load_config_to_network``.
        config_group.attrs["epochs_max"] = getattr(network, "epochs_max", 0)
        config_group.attrs["max_iterations"] = getattr(network, "max_iterations", 0)
        config_group.attrs["output_epochs"] = getattr(network, "output_epochs", 0)
        config_group.attrs["candidate_patience"] = getattr(network, "candidate_patience", 0)
        config_group.attrs["candidate_epochs"] = getattr(network, "candidate_epochs", 0)
        config_group.attrs["convergence_threshold"] = getattr(network, "convergence_threshold", 0.0)
        config_group.attrs["candidate_convergence_threshold"] = getattr(network, "candidate_convergence_threshold", 0.0)
        write_str_attr(config_group, "init_output_weights", getattr(network, "init_output_weights", "zero"))

        self.logger.debug("CascadeHDF5Serializer: Saved configuration")

    def _save_architecture(self, hdf5_file: h5py.File, network) -> None:
        """Save network architecture information."""
        arch_group = hdf5_file.create_group("arch")

        # Basic architecture parameters
        arch_group.attrs["input_size"] = network.input_size
        arch_group.attrs["output_size"] = network.output_size
        arch_group.attrs["num_hidden_units"] = len(network.hidden_units)
        arch_group.attrs["max_hidden_units"] = network.max_hidden_units
        write_str_attr(arch_group, "activation_function_name", network.activation_function_name)

        # Save connectivity information if needed
        connectivity_group = arch_group.create_group("connectivity")
        connectivity_group.attrs["input_to_output_connections"] = network.input_size * network.output_size

        # Hidden unit connectivity
        for i, unit in enumerate(network.hidden_units):
            unit_info = connectivity_group.create_group(f"hidden_unit_{i}")
            unit_info.attrs["input_connections"] = len(unit["weights"]) if "weights" in unit else 0
            if "activation_fn" in unit:
                write_str_attr(
                    unit_info,
                    "activation_function",
                    getattr(unit["activation_fn"], "__name__", "unknown"),
                )

        self.logger.debug("CascadeHDF5Serializer: Saved architecture")

    def _save_parameters(
        self,
        hdf5_file: h5py.File,
        network,
        compression: str,
        compression_opts: int,
    ) -> None:
        """Save model weights and biases."""
        params_group = hdf5_file.create_group("params")

        # Save output layer parameters
        output_group = params_group.create_group("output_layer")

        if hasattr(network, "output_weights") and network.output_weights is not None:
            save_tensor(
                output_group,
                "weights",
                network.output_weights,
                compression,
                compression_opts,
            )

        if hasattr(network, "output_bias") and network.output_bias is not None:
            save_tensor(output_group, "bias", network.output_bias, compression, compression_opts)

        # Calculate and save checksums
        checksum_data = {}
        if hasattr(network, "output_weights") and network.output_weights is not None:
            checksum_data["output_weights"] = calculate_tensor_checksum(network.output_weights)
        if hasattr(network, "output_bias") and network.output_bias is not None:
            checksum_data["output_bias"] = calculate_tensor_checksum(network.output_bias)

        if checksum_data:
            write_str_dataset(output_group, "checksums", json.dumps(checksum_data))
            self.logger.debug("CascadeHDF5Serializer: Saved parameter checksums")

        # Save optimizer state if it exists
        if hasattr(network, "output_optimizer") and network.output_optimizer is not None:
            opt_group = output_group.create_group("optimizer")
            try:
                self._save_network_parameters_to_hdf5_helper(network, opt_group)
            except Exception as e:
                self.logger.warning(f"CascadeHDF5Serializer: Could not save optimizer state: {e}")
        elif hasattr(network, "output_optimizer"):
            # The attribute exists but is None -- e.g. a network whose topology was
            # edited, or one restored from a snapshot whose optimizer could not be
            # rebuilt. No optimizer group is written, so any optimizer state the
            # source snapshot carried is dropped on this save. That used to be
            # entirely silent; say it out loud.
            self.logger.warning("CascadeHDF5Serializer: network.output_optimizer is None - writing no optimizer group; optimizer state from any source snapshot is not carried forward")

        self.logger.debug("CascadeHDF5Serializer: Saved parameters")

    def _save_network_parameters_to_hdf5_helper(self, network, opt_group):
        opt_state = network.output_optimizer.state_dict()
        # Convert optimizer state to JSON-serializable format
        opt_state_serializable = {
            "state": {str(k): {inner_k: (inner_v.tolist() if hasattr(inner_v, "tolist") else inner_v) for inner_k, inner_v in v.items()} for k, v in opt_state.get("state", {}).items()},
            "param_groups": opt_state.get("param_groups", []),
        }
        write_str_dataset(opt_group, "state_dict", json.dumps(opt_state_serializable))
        write_str_attr(opt_group, "optimizer_type", type(network.output_optimizer).__name__)
        # Record the lr the OPTIMIZER is actually running with. ``network.learning_rate``
        # is a separate field that diverges from ``config.optimizer_config.learning_rate``
        # (what ``_create_optimizer`` reads) after a runtime params patch, so persisting it
        # here recorded a value the optimizer never used.
        actual_lr = getattr(network, "learning_rate", None)
        try:
            actual_lr = network.output_optimizer.param_groups[0]["lr"]
        except (AttributeError, IndexError, KeyError, TypeError):
            pass
        write_str_attr(opt_group, "learning_rate", actual_lr)
        self.logger.debug("CascadeHDF5Serializer: Saved optimizer state")

    def _save_hidden_units(self, hdf5_file: h5py.File, network, compression: str, compression_opts: int) -> None:
        """Save hidden units with integrity checksums."""
        if not network.hidden_units:
            return

        hidden_group = hdf5_file.create_group("hidden_units")
        hidden_group.attrs["num_units"] = len(network.hidden_units)

        for i, unit in enumerate(network.hidden_units):
            unit_group = hidden_group.create_group(f"unit_{i}")

            # Save weights and bias
            if "weights" in unit:
                save_tensor(
                    unit_group,
                    "weights",
                    unit["weights"],
                    compression,
                    compression_opts,
                )
            if "bias" in unit:
                save_tensor(unit_group, "bias", unit["bias"], compression, compression_opts)

            # Calculate and save checksums for integrity verification
            checksum_data = {}
            if "weights" in unit:
                checksum_data["weights"] = calculate_tensor_checksum(unit["weights"])
            if "bias" in unit:
                checksum_data["bias"] = calculate_tensor_checksum(unit["bias"])
            if checksum_data:
                write_str_dataset(unit_group, "checksums", json.dumps(checksum_data))

            # Save correlation
            if "correlation" in unit:
                unit_group.attrs["correlation"] = float(unit["correlation"])

            # Save activation function name (per unit, in case they differ)
            if "activation_fn" in unit:
                af_name = getattr(unit["activation_fn"], "_activation_name", network.activation_function_name)
                write_str_attr(unit_group, "activation_function_name", af_name)
            else:
                write_str_attr(
                    unit_group,
                    "activation_function_name",
                    network.activation_function_name,
                )

        self.logger.debug(f"CascadeHDF5Serializer: Saved {len(network.hidden_units)} hidden units with checksums")

    def _save_random_state(self, hdf5_file: h5py.File, network, compression: str, compression_opts: int) -> None:
        """Save random state for deterministic reproducibility."""
        random_group = hdf5_file.create_group("random")

        # Save random parameters
        random_group.attrs["seed"] = getattr(network, "random_seed", 0)
        random_group.attrs["max_value"] = getattr(network, "random_max_value", 1000000)
        random_group.attrs["sequence_max_value"] = getattr(network, "sequence_max_value", 1000000)
        random_group.attrs["value_scale"] = getattr(network, "random_value_scale", 0.1)

        # Save RNG states
        try:
            # Python random state (for candidate seeding, etc.)
            python_state = random.getstate()
            python_state_bytes = pickle.dumps(python_state)
            # Save as fixed-length byte array (not variable-length)
            python_state_array = np.frombuffer(python_state_bytes, dtype=np.uint8)
            save_numpy_array(
                random_group,
                "python_state",
                python_state_array,
                compression,
                compression_opts,
            )

            # NumPy random state
            np_state = np.random.get_state()
            np_group = random_group.create_group("numpy_state")
            write_str_attr(np_group, "state_type", np_state[0])
            save_numpy_array(np_group, "state_array", np_state[1], compression, compression_opts)
            np_group.attrs["pos"] = np_state[2]
            np_group.attrs["has_gauss"] = np_state[3]
            np_group.attrs["cached_gaussian"] = np_state[4]

            # PyTorch random state
            torch_state = torch.get_rng_state()
            save_numpy_array(
                random_group,
                "torch_state",
                torch_state.numpy(),
                compression,
                compression_opts,
            )

            # CUDA random state if available
            if torch.cuda.is_available():
                try:
                    cuda_states = torch.cuda.get_rng_state_all()
                    cuda_group = random_group.create_group("cuda_states")
                    for i, state in enumerate(cuda_states):
                        save_numpy_array(
                            cuda_group,
                            f"device_{i}",
                            state.cpu().numpy(),
                            compression,
                            compression_opts,
                        )

                except Exception as e:
                    self.logger.warning(f"Could not save CUDA random states: {e}")

            self.logger.debug("CascadeHDF5Serializer: Saved all random states (Python, NumPy, PyTorch, CUDA)")

        except Exception as e:
            self.logger.warning(f"Could not save random states: {e}")

        self.logger.debug("CascadeHDF5Serializer: Saved random state")

    def _save_multiprocessing_state(self, hdf5_file: h5py.File, network) -> None:
        """Save multiprocessing configuration for restoration."""
        mp_group = hdf5_file.create_group("mp")

        # Save MP configuration (not live objects)
        try:
            self._save_cascor_network_state_to_hdf5_helper(network, mp_group)
        except Exception as e:
            self.logger.warning(f"Could not save multiprocessing state: {e}")

        self.logger.debug("CascadeHDF5Serializer: Saved multiprocessing state")

    def _save_cascor_network_state_to_hdf5_helper(self, network, mp_group):
        # Determine role (server/client/none)
        role = "none"  # Default
        if hasattr(network, "candidate_training_manager") and network.candidate_training_manager:
            role = "server"
        elif hasattr(network, "candidate_training_queue_address") and network.candidate_training_queue_address:
            role = "client"

        write_str_attr(mp_group, "role", role)

        # Save multiprocessing context information
        if hasattr(network, "candidate_training_context"):
            ctx = network.candidate_training_context
            write_str_attr(mp_group, "start_method", ctx.get_start_method() if ctx else "spawn")
        else:
            write_str_attr(mp_group, "start_method", "spawn")

        # Save address and authentication
        if hasattr(network, "candidate_training_queue_address"):
            addr = network.candidate_training_queue_address
            if isinstance(addr, (list, tuple)) and len(addr) >= 2:
                write_str_attr(mp_group, "address_host", str(addr[0]))
                mp_group.attrs["address_port"] = int(addr[1])
            else:
                write_str_attr(mp_group, "address_host", "127.0.0.1")
                mp_group.attrs["address_port"] = 0
        else:
            write_str_attr(mp_group, "address_host", "127.0.0.1")
            mp_group.attrs["address_port"] = 0

        if hasattr(network, "candidate_training_queue_authkey"):
            authkey = network.candidate_training_queue_authkey
            authkey_hex = authkey.hex() if isinstance(authkey, bytes) else str(authkey)
            write_str_attr(mp_group, "authkey_hex", authkey_hex)

        # Save timeouts
        if hasattr(network, "candidate_training_tasks_queue_timeout"):
            mp_group.attrs["tasks_queue_timeout"] = float(network.candidate_training_tasks_queue_timeout)
        if hasattr(network, "candidate_training_shutdown_timeout"):
            mp_group.attrs["shutdown_timeout"] = float(network.candidate_training_shutdown_timeout)

        # Save queue configuration
        queues_config = {"task_queue": "BaseManager", "result_queue": "BaseManager"}
        write_str_dataset(mp_group, "queues_to_create", json.dumps(queues_config))

        # Save policy flags
        mp_group.attrs["autostart"] = True  # Default to autostart on restore

    @staticmethod
    def _snapshot_history_view(network) -> Dict[str, Any]:
        """Return a shallow, point-in-time copy of the network's history dict.

        C1 (I-3 write-isolation hardening): ``save_network`` can run
        concurrently with the training thread, which appends per-epoch
        entries to the history lists (``cascade_correlation.py`` — no lock is
        shared with the serializer, so no manager-side lock can exclude those
        appends). The HDF5 writes below are slow; iterating the live lists
        risks mid-iteration mutation. Copying the top-level dict and each
        list value is a handful of GIL-atomic operations taken up front,
        giving every subsequent write a stable iteration target. Per-element
        objects (floats, unit-metadata dicts, swap events) are appended
        fully built and at most attr-backfilled afterwards, so a shallow
        copy is sufficient to prevent crashes; deep consistency of element
        internals is not claimed.
        """
        history = getattr(network, "history", None)
        if not history:
            return {}
        view: Dict[str, Any] = {}
        for key in list(history.keys()):
            value = history.get(key)
            view[key] = list(value) if isinstance(value, list) else value
        return view

    def _save_training_history(self, hdf5_file: h5py.File, network, compression: str, compression_opts: int) -> None:  # noqa: C901
        """Save training history.

        C1 (I-3): serializes from ``_snapshot_history_view``'s point-in-time
        copy rather than the live history dict, so a mid-training snapshot
        cannot crash on concurrent list mutation by the training thread.
        """
        history = self._snapshot_history_view(network)
        if not history:
            return

        history_group = hdf5_file.create_group("history")

        # Save numeric arrays - use network's actual keys (value_* not val_*)
        key_mapping = {
            "train_loss": "train_loss",
            "value_loss": "value_loss",  # Match network history keys
            "train_accuracy": "train_accuracy",
            "value_accuracy": "value_accuracy",  # Match network history keys
        }

        for network_key, save_key in key_mapping.items():
            if network_key in history and history[network_key]:
                data = np.array(history[network_key])
                save_numpy_array(history_group, save_key, data, compression, compression_opts)

        # Save hidden units added history (metadata-only since CR-063;
        # legacy weight/bias arrays are still handled for backward compat)
        if "hidden_units_added" in history:
            units_group = history_group.create_group("hidden_units_added")
            for i, unit_data in enumerate(history["hidden_units_added"]):
                unit_group = units_group.create_group(f"unit_{i}")
                if isinstance(unit_data, dict):
                    if "correlation" in unit_data:
                        unit_group.attrs["correlation"] = float(unit_data["correlation"])
                    # New metadata-only fields (CR-063)
                    if "weight_shape" in unit_data:
                        unit_group.attrs["weight_shape"] = list(unit_data["weight_shape"])
                    if "unit_index" in unit_data:
                        unit_group.attrs["unit_index"] = int(unit_data["unit_index"])
                    # Legacy weight/bias arrays (kept for backward compat with old snapshots)
                    if "weights" in unit_data:
                        save_numpy_array(
                            unit_group,
                            "weights",
                            unit_data["weights"],
                            compression,
                            compression_opts,
                        )
                    if "bias" in unit_data:
                        save_numpy_array(
                            unit_group,
                            "bias",
                            unit_data["bias"],
                            compression,
                            compression_opts,
                        )

        # P2-2 (Issue #3): persist live-dataset-swap events. Each event is
        # a subgroup under ``history/dataset_swaps/`` named ``event_{i}``
        # with the schema:
        #   * ``timestamp``                (str attr, ISO-8601 UTC)
        #   * ``before_cfg``               (JSON-encoded str attr; ``"null"`` for None)
        #   * ``after_cfg``                (JSON-encoded str attr; ``"null"`` for None)
        #   * ``arch_changes``             (JSON-encoded str attr)
        #   * ``pre_swap_snapshot_id``     (str attr, optional — P2-3 backfill)
        #   * ``post_swap_snapshot_id``    (str attr, optional — P2-3 backfill)
        # Nested dicts go through JSON because HDF5 attrs are flat and the
        # ``arch_changes`` block has variable shape (``appended_nodes`` is
        # itself a dict). Missing snapshot-ID attrs decode back to None on
        # load so the §3.9 schema is faithfully reproduced.
        if "dataset_swaps" in history and history["dataset_swaps"]:
            swaps_group = history_group.create_group("dataset_swaps")
            for i, swap_event in enumerate(history["dataset_swaps"]):
                if not isinstance(swap_event, dict):
                    continue
                ev_group = swaps_group.create_group(f"event_{i}")
                if "timestamp" in swap_event and swap_event["timestamp"] is not None:
                    ev_group.attrs["timestamp"] = str(swap_event["timestamp"])
                ev_group.attrs["before_cfg"] = json.dumps(swap_event.get("before_cfg"))
                ev_group.attrs["after_cfg"] = json.dumps(swap_event.get("after_cfg"))
                ev_group.attrs["arch_changes"] = json.dumps(swap_event.get("arch_changes", {}))
                if swap_event.get("pre_swap_snapshot_id") is not None:
                    ev_group.attrs["pre_swap_snapshot_id"] = str(swap_event["pre_swap_snapshot_id"])
                if swap_event.get("post_swap_snapshot_id") is not None:
                    ev_group.attrs["post_swap_snapshot_id"] = str(swap_event["post_swap_snapshot_id"])

        # CAN-015g (Phase 6E follow-on, g-1): per-sample weight history
        # for Replay V2. Persisted only when the network exposes a
        # ``weight_history`` dict (populated by the lifecycle in g-2);
        # absence is the V1 / pre-g case and silently skipped so all
        # existing snapshots and the metric-only Replay V1 path keep
        # working unchanged.
        if hasattr(network, "weight_history") and network.weight_history:
            self._save_weight_history(history_group, network.weight_history, compression, compression_opts)

        self.logger.debug("CascadeHDF5Serializer: Saved training history")

    # ------------------------------------------------------------------
    # CAN-015g (Phase 6E follow-on): per-sample weight history
    # ------------------------------------------------------------------
    # Schema v2 layout under ``history/weights/``:
    #   meta/                               (subgroup, attrs only)
    #     schema_version       (int64 attr) — always 2 for this writer
    #     sampling_strategy    (str attr)   — "adaptive" | "every_n" | "trigger"
    #     sampling_interval    (int64 attr) — N (epochs); 0 == trigger-only
    #     num_samples          (int64 attr) — len(sample_indices)
    #   sample_indices         (int64 dataset, [num_samples])
    #   output_weights/                     (subgroup of per-sample datasets)
    #     0000  (float32 dataset, [in + hid_at_sample_0, out])
    #     0001  (float32 dataset, [in + hid_at_sample_1, out])
    #     ...
    #   output_bias/                        (subgroup of per-sample datasets)
    #     0000  (float32 dataset, [out])
    #     ...
    #   hidden_units/
    #     0000/                             (one subgroup per unit)
    #       first_sample_index (int64 attr)
    #       activation         (str attr)
    #       weights/                        (subgroup of per-sample datasets)
    #         0050  (float32 dataset, [in + cascade_index])
    #         ...
    #       bias/                           (subgroup of per-sample datasets)
    #         0050  (float32 dataset, [])   — scalar
    #         ...
    #
    # Per-sample subgroups are used (rather than a single 3D dataset) because
    # the output-layer width grows with each cascade-add event — there is no
    # single fixed shape. The numeric subgroup names are zero-padded so they
    # sort lexicographically (HDF5 returns keys in insertion order, but the
    # readers tolerate either).

    _WEIGHT_HISTORY_SCHEMA_VERSION = 2

    def _save_weight_history(self, history_group: h5py.Group, weight_history: Dict[str, Any], compression: str, compression_opts: int) -> None:
        """Persist per-sample weight tensors under ``history/weights/``."""
        weights_group = history_group.create_group("weights")
        meta_group = weights_group.create_group("meta")

        sample_indices = list(weight_history.get("sample_indices", []))
        meta_group.attrs["schema_version"] = self._WEIGHT_HISTORY_SCHEMA_VERSION
        meta_group.attrs["num_samples"] = len(sample_indices)
        write_str_attr(meta_group, "sampling_strategy", str(weight_history.get("sampling_strategy", "adaptive")))
        meta_group.attrs["sampling_interval"] = int(weight_history.get("sampling_interval", 0))

        if not sample_indices:
            self.logger.debug("CascadeHDF5Serializer: weight_history has no samples — wrote meta only")
            return

        save_numpy_array(weights_group, "sample_indices", np.asarray(sample_indices, dtype=np.int64), compression, compression_opts)

        output_weights = list(weight_history.get("output_weights", []))
        output_bias = list(weight_history.get("output_bias", []))
        if len(output_weights) != len(sample_indices) or len(output_bias) != len(sample_indices):
            raise ValueError(f"weight_history output arrays length mismatch: got {len(output_weights)} weights, {len(output_bias)} biases for {len(sample_indices)} samples")

        ow_group = weights_group.create_group("output_weights")
        ob_group = weights_group.create_group("output_bias")
        for i, (w, b) in enumerate(zip(output_weights, output_bias)):
            sample_key = f"{i:04d}"
            save_numpy_array(ow_group, sample_key, np.ascontiguousarray(w, dtype=np.float32), compression, compression_opts)
            save_numpy_array(ob_group, sample_key, np.ascontiguousarray(b, dtype=np.float32), compression, compression_opts)

        hidden_units = list(weight_history.get("hidden_units", []))
        if hidden_units:
            units_group = weights_group.create_group("hidden_units")
            for unit_idx, unit in enumerate(hidden_units):
                unit_group = units_group.create_group(f"{unit_idx:04d}")
                unit_group.attrs["first_sample_index"] = int(unit.get("first_sample_index", 0))
                write_str_attr(unit_group, "activation", str(unit.get("activation", "")))
                unit_w = unit.get("weights", [])
                unit_b = unit.get("bias", [])
                if len(unit_w) != len(unit_b):
                    raise ValueError(f"weight_history hidden unit {unit_idx} weights/bias length mismatch: {len(unit_w)} vs {len(unit_b)}")
                w_group = unit_group.create_group("weights")
                b_group = unit_group.create_group("bias")
                for j, (w, b) in enumerate(zip(unit_w, unit_b)):
                    sample_key = f"{j:04d}"
                    save_numpy_array(w_group, sample_key, np.ascontiguousarray(w, dtype=np.float32), compression, compression_opts)
                    save_numpy_array(b_group, sample_key, np.asarray(b, dtype=np.float32), compression, compression_opts)

        self.logger.debug(f"CascadeHDF5Serializer: Saved weight_history with {len(sample_indices)} samples, {len(hidden_units)} hidden units")

    def _save_training_data(
        self,
        hdf5_file: h5py.File,
        network,
        compression: str,
        compression_opts: int,
    ) -> None:
        """Save training data (normally excluded)."""
        if not hasattr(network, "_training_data"):
            return

        data_group = hdf5_file.create_group("data")

        # Save training datasets if present
        training_data = network._training_data
        if isinstance(training_data, dict):
            for key, dataset in training_data.items():
                if hasattr(dataset, "numpy"):  # PyTorch tensor
                    save_tensor(data_group, key, dataset, compression, compression_opts)
                elif isinstance(dataset, np.ndarray):
                    save_numpy_array(data_group, key, dataset, compression, compression_opts)
        self.logger.debug("CascadeHDF5Serializer: Saved training data")

    def load_network_result(self, filepath: Union[str, Path], restore_multiprocessing: bool = True) -> SnapshotLoadResult:
        """
        Load a CascadeCorrelationNetwork, reporting the reason on failure.

        D-B: the load itself still goes through :meth:`load_network`, which stays the
        seam every caller and test knows. Only when it comes back ``None`` does this
        run a second, cheap pass to work out WHY — because that is the one thing the
        bare ``None`` throws away, and the API then reported every cause as
        ``404 "not found or failed to load"``, fusing *pick a different snapshot* with
        *investigate data loss*.

        Classifying after the fact rather than threading a reason through the loader
        keeps the hot path and the public signature untouched; the extra file open
        happens only on the error path.

        Returns:
            SnapshotLoadResult — ``network`` set and ``status == SNAPSHOT_OK`` on
            success; otherwise ``SNAPSHOT_ABSENT`` or ``SNAPSHOT_CORRUPT`` with a
            human-readable ``detail``.
        """
        network = self.load_network(filepath, restore_multiprocessing)
        if network is not None:
            return snapshot_loaded(network)
        return self._classify_load_failure(filepath)

    def _classify_load_failure(self, filepath: Union[str, Path]) -> SnapshotLoadResult:
        """Work out why :meth:`load_network` returned ``None`` (D-B).

        Absent is unambiguous. Everything else is corrupt: the file is there but the
        deserializer could not turn it into a network — a rejected format, a missing
        group, an unreadable file. ``_validate_format_detail`` supplies the specific
        reason when it can, so the operator is told which check failed rather than
        just "failed to load".
        """
        if not os.path.exists(filepath):
            return snapshot_absent(f"no snapshot file at {filepath}")
        try:
            with h5py.File(filepath, "r") as hdf5_file:
                detail = self._validate_format_detail(hdf5_file)
        except Exception as exc:  # noqa: BLE001 - any read failure means corrupt
            return snapshot_corrupt(f"unreadable snapshot: {exc}")
        if detail is not None:
            return snapshot_corrupt(detail)

        # D-E: the format is fine, so the refusal came from an integrity gate. Re-load
        # permissively to recover WHICH gate — the checks need the built network, which
        # the failed load did not hand back. Only ever runs on the error path, and it
        # goes through ``load_network`` so a patched loader stays the seam.
        network = self.load_network(filepath, restore_multiprocessing=False, allow_invalid=True)
        if network is not None:
            try:
                with h5py.File(filepath, "r") as hdf5_file:
                    findings = self._check_integrity(hdf5_file, network)
            except Exception as exc:  # noqa: BLE001 - re-read failure means corrupt
                return snapshot_corrupt(f"unreadable snapshot: {exc}")
            if findings:
                # Report the first finding; the rest are already logged. Arch mismatch
                # wins if present, because it names a different investigation than
                # "something is damaged".
                status, first = next((f for f in findings if f[0] == SNAPSHOT_ARCH_MISMATCH), findings[0])
                return arch_mismatch(first) if status == SNAPSHOT_ARCH_MISMATCH else snapshot_corrupt(first)

        return snapshot_corrupt("snapshot could not be deserialized into a network")

    # def load_network(self, filepath: Union[str, Path], restore_multiprocessing: bool = True) -> Optional:  # Original - invalid Optional usage
    def load_network(self, filepath: Union[str, Path], restore_multiprocessing: bool = True, allow_invalid: bool = False) -> Optional[Any]:
        """
        Load a CascadeCorrelationNetwork from HDF5 format.

        Returns ``None`` on every failure. Callers that need to distinguish *absent*
        from *corrupt* (D-B) should use :meth:`load_network_result` instead.

        D-E: the load is now **fail-closed**. Six integrity gates ran here before and
        none of them stopped anything — the loader logged (two at ERROR), returned the
        network anyway, and then logged ``Successfully loaded network``. A snapshot that
        fails any gate is refused.

        Args:
            filepath: Path to HDF5 file
            restore_multiprocessing: Whether to restore multiprocessing state
            allow_invalid: Load a snapshot that FAILS an integrity gate anyway, for
                forensics. Deliberately library/CLI-only — no API surface reaches it,
                so a knowingly-broken network can never be put on the live lifecycle
                from a URL. Roughly 0.6% of the archive needs it (~170 files, measured).
                Do not train a network loaded this way.

        Returns:
            CascadeCorrelationNetwork instance or None if failed
        """
        try:
            self.logger.info(f"CascadeHDF5Serializer: Loading network from {filepath}")
            if not os.path.exists(filepath):
                self.logger.error(f"CascadeHDF5Serializer: File not found: {filepath}")
                return None
            with h5py.File(filepath, "r") as hdf5_file:
                if not self._validate_format(hdf5_file):
                    return None
                network = self._create_network_from_file(hdf5_file)
                if not network:
                    return None
                self._load_architecture(hdf5_file, network)
                # CAN-014 (Phase 6E Sprint A-5): restore the runtime-tunable
                # training params persisted in the ``config`` group so the
                # rehydrated network actually reflects the values the
                # operator trained with — not just construction-time
                # defaults from CascadeCorrelationConfig.
                self._load_config_to_network(hdf5_file, network)
                self._load_parameters(hdf5_file, network)
                self._load_hidden_units(hdf5_file, network)
                self._load_random_state(hdf5_file, network)
                if "history" in hdf5_file:
                    self._load_training_history(hdf5_file, network)
                if restore_multiprocessing and "mp" in hdf5_file:
                    self._restore_multiprocessing_state(hdf5_file, network)
                # D-C: attach run identity to the loaded network. ``None`` when the
                # snapshot predates provenance or came from an unidentified run —
                # that is a real answer, not a failure, and must never gate the load.
                network.provenance = read_provenance(hdf5_file)
                findings = self._check_integrity(hdf5_file, network)
            if findings:
                for status, detail in findings:
                    self.logger.error(f"CascadeHDF5Serializer: integrity gate failed [{status}]: {detail}")
                if not allow_invalid:
                    self.logger.error(f"CascadeHDF5Serializer: REFUSING to load {filepath}: {len(findings)} integrity finding(s). Pass allow_invalid=True to inspect it anyway.")
                    return None
                self.logger.warning(f"CascadeHDF5Serializer: loading {filepath} DESPITE {len(findings)} integrity finding(s) (allow_invalid=True) — inspect only, do not train this network")
            self.logger.info(f"CascadeHDF5Serializer: Successfully loaded network from {filepath}")
            return network
        except Exception as e:
            return self._log_exception_stacktrace("CascadeHDF5Serializer: Error loading network: ", e, None)

    def _load_architecture(self, hdf5_file: h5py.File, network) -> None:
        """Load network architecture."""
        if "arch" not in hdf5_file:
            return

        arch_group = hdf5_file["arch"]

        # Verify architecture matches
        saved_input_size = arch_group.attrs.get("input_size", network.input_size)
        saved_output_size = arch_group.attrs.get("output_size", network.output_size)

        if saved_input_size != network.input_size:
            self.logger.warning(f"Input size mismatch: {saved_input_size} != {network.input_size}")

        if saved_output_size != network.output_size:
            self.logger.warning(f"Output size mismatch: {saved_output_size} != {network.output_size}")

        # Load activation function name
        af_name = read_str_attr(arch_group, "activation_function_name", network.activation_function_name)
        network.activation_function_name = af_name

        # Reinitialize activation function with the loaded name--ensures activation_fn and activation_functions_dict are properly set
        network._init_activation_function()

        self.logger.debug(f"CascadeHDF5Serializer: Loaded architecture with activation function: {af_name}")

    def _load_config_to_network(self, hdf5_file: h5py.File, network) -> None:
        """CAN-014 (Phase 6E Sprint A-5): restore runtime-tunable params
        from the ``config`` HDF5 group onto the live network.

        ``_save_configuration`` persists every field listed in
        ``update_params``' whitelist. This method restores them, mirrored
        as direct attributes on ``network`` so subsequent
        ``get_training_params`` reflects the snapshot rather than
        whatever ``CascadeCorrelationConfig`` defaults the constructor
        used.

        Each load is gated on ``is_attr`` so older snapshots that
        pre-date a given field still load cleanly — the missing field
        simply falls back to whatever the freshly-constructed network
        already has. The list of fields here matches
        ``_save_configuration`` exactly; keep them in sync when adding
        new tunables.
        """
        if "config" not in hdf5_file:
            return
        config_group = hdf5_file["config"]

        # Numeric attributes — straight setattr, but only when the field
        # was actually persisted (older snapshots may be missing some).
        for key in (
            "learning_rate",
            "candidate_learning_rate",
            "max_hidden_units",
            "correlation_threshold",
            "candidate_pool_size",
            "patience",
            "epochs_max",
            "max_iterations",
            "output_epochs",
            "candidate_patience",
            "candidate_epochs",
            "convergence_threshold",
            "candidate_convergence_threshold",
        ):
            if key in config_group.attrs:
                value = config_group.attrs[key]
                # ``getattr`` rather than ``hasattr`` so a freshly-loaded
                # network missing the attribute still picks up the value
                # — none of cascor's tunables are read-only properties.
                setattr(network, key, value)

        # String attribute — ``init_output_weights`` round-trip.
        if "init_output_weights" in config_group.attrs:
            value = read_str_attr(config_group, "init_output_weights", getattr(network, "init_output_weights", "zero"))
            network.init_output_weights = value

        self.logger.debug("CascadeHDF5Serializer: Loaded runtime-tunable params from config group")

    def _load_parameters(self, hdf5_file: h5py.File, network) -> None:
        """Load model parameters."""
        if "params" not in hdf5_file:
            return

        params_group = hdf5_file["params"]

        # Load output layer parameters
        if "output_layer" in params_group:
            output_group = params_group["output_layer"]

            if "weights" in output_group:
                network.output_weights = load_tensor(output_group["weights"])

            if "bias" in output_group:
                network.output_bias = load_tensor(output_group["bias"])

            # D-E: checksum verification moved to ``_check_integrity``, which is the
            # single place that both verifies AND acts. It used to live here and only
            # logged at ERROR before continuing, so a mismatch — unambiguous evidence
            # of corruption — never stopped the load. Verifying in both places would
            # re-hash the tensors on every successful load for no benefit.

            # Load optimizer state if it exists
            if "optimizer" in output_group:
                opt_group = output_group["optimizer"]
                try:
                    self._load_optimizer_state_from_hdf5_helper(opt_group, network)
                except Exception as e:
                    self.logger.warning(f"CascadeHDF5Serializer: Could not restore optimizer: {e}")
                    network.output_optimizer = None

        self.logger.debug("CascadeHDF5Serializer: Loaded parameters")

    @staticmethod
    def _coerce_optimizer_lr(raw: Any, fallback: float) -> float:
        """Coerce a persisted ``learning_rate`` value to ``float``.

        ``learning_rate`` is written through ``write_str_attr``, which stores
        ``np.bytes_(str(value))``. Reading it back with a bare ``attrs.get`` and
        handing it to an optimizer raises ``TypeError: '<=' not supported between
        instances of 'float' and 'numpy.bytes_'`` inside torch's ``0.0 <= lr``
        validation -- and every snapshot in the corpus stores it as bytes, so that
        fires on essentially every load. Accepts bytes / str / numpy scalar /
        float so both the historical string form and a plain numeric attribute
        round-trip.
        """
        if raw is None:
            return float(fallback)
        if isinstance(raw, (bytes, bytearray, np.bytes_)):
            raw = raw.decode("utf-8", "replace")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(fallback)

    @staticmethod
    def _rehydrate_optimizer_state(opt_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return a JSON-round-tripped optimizer ``state_dict`` in torch form.

        ``_save_network_parameters_to_hdf5_helper`` stringifies the per-parameter
        keys and calls ``.tolist()`` on every tensor. **``load_state_dict`` ACCEPTS
        that raw JSON form without raising** -- and then keys the state by the
        strings ``"0"`` / ``"1"``, which match no ``Parameter``. The result is an
        optimizer that reports "restored" while carrying no usable state, training
        silently from a fresh start. Keys must return to ``int`` and buffers to
        tensors, or the restore is a lie.
        """
        state: Dict[Any, Any] = {}
        for key, entry in (opt_state_dict.get("state") or {}).items():
            try:
                key = int(key)
            except (TypeError, ValueError):
                pass
            state[key] = {inner_k: (torch.tensor(inner_v) if isinstance(inner_v, list) else inner_v) for inner_k, inner_v in (entry or {}).items()}
        return {"state": state, "param_groups": opt_state_dict.get("param_groups", [])}

    def _load_optimizer_state_from_hdf5_helper(self, opt_group, network):
        import torch.optim as optim

        opt_type = read_str_attr(opt_group, "optimizer_type", "Adam")

        # Parse the persisted state first: its ``param_groups[0]["lr"]`` is the
        # learning rate the optimizer actually ran with. The ``learning_rate``
        # attribute records ``network.learning_rate``, which diverges from
        # ``config.optimizer_config.learning_rate`` (what ``_create_optimizer``
        # reads) after any runtime params patch, so the attribute alone is not
        # authoritative.
        opt_state_dict = None
        if "state_dict" in opt_group:
            try:
                opt_state_dict = json.loads(read_str_dataset(opt_group, "state_dict"))
            except (ValueError, TypeError) as exc:
                self.logger.warning(f"CascadeHDF5Serializer: Could not parse optimizer state_dict: {exc}")

        learning_rate = self._coerce_optimizer_lr(opt_group.attrs.get("learning_rate"), network.learning_rate)
        if opt_state_dict:
            param_groups = opt_state_dict.get("param_groups") or []
            if param_groups and "lr" in param_groups[0]:
                learning_rate = self._coerce_optimizer_lr(param_groups[0]["lr"], learning_rate)

        # Rebuild the output layer the optimizer was bound to at save time.
        # ``nn.Linear`` stores weight as (out, in) -- the transpose of
        # ``output_weights`` -- which is the shape the persisted buffers carry.
        input_size = network.output_weights.shape[0]
        output_layer = torch.nn.Linear(input_size, network.output_size)
        with torch.no_grad():
            output_layer.weight.copy_(network.output_weights.t())
            output_layer.bias.copy_(network.output_bias)

        # Honour the recorded optimizer class. ``optimizer_type`` is NOT persisted
        # anywhere else in the snapshot (``OptimizerConfig`` is skipped by
        # ``_config_to_dict``), so this group is its only record -- rebuilding
        # everything as Adam makes ``GET`` training params report a metaparameter
        # the operator never chose.
        opt_cls = getattr(optim, opt_type, None)
        type_matched = isinstance(opt_cls, type) and issubclass(opt_cls, optim.Optimizer)
        if not type_matched:
            self.logger.warning(f"CascadeHDF5Serializer: Unrecognized optimizer type {opt_type!r}; rebuilding as Adam WITHOUT restoring state")
            opt_cls = optim.Adam

        network.output_optimizer = opt_cls(output_layer.parameters(), lr=learning_rate)

        if opt_state_dict and type_matched:
            try:
                network.output_optimizer.load_state_dict(self._rehydrate_optimizer_state(opt_state_dict))
                self.logger.debug(f"CascadeHDF5Serializer: Restored {opt_type} state ({len(opt_state_dict.get('state') or {})} parameter entries)")
            except (KeyError, ValueError, TypeError) as exc:
                # Buffers differ per optimizer family (Adam keeps exp_avg/step, SGD
                # momentum_buffer), so a type/shape mismatch must degrade rather
                # than propagate -- raising here would fail loads that succeed
                # today. The freshly-built optimizer is kept.
                self.logger.warning(f"CascadeHDF5Serializer: Could not restore {opt_type} state ({exc}); optimizer rebuilt without it")
        elif opt_state_dict is None:
            self.logger.debug("CascadeHDF5Serializer: Created optimizer without state dict")

    def _load_hidden_units(self, hdf5_file: h5py.File, network) -> None:
        """Load hidden units."""
        if "hidden_units" not in hdf5_file:
            network.hidden_units = []
            return

        hidden_group = hdf5_file["hidden_units"]
        num_units = hidden_group.attrs.get("num_units", 0)

        network.hidden_units = []
        for i in range(num_units):
            unit_group_name = f"unit_{i}"
            if unit_group_name not in hidden_group:
                continue

            unit_group = hidden_group[unit_group_name]
            unit = {}

            # Load weights and bias
            if "weights" in unit_group:
                unit["weights"] = load_tensor(unit_group["weights"])
            if "bias" in unit_group:
                unit["bias"] = load_tensor(unit_group["bias"])

            # D-E: hidden-unit checksum verification moved to ``_check_integrity``,
            # alongside the output-layer one. It used to live here and only log at
            # ERROR before continuing, so a same-shape tamper of a hidden unit's
            # weights loaded cleanly -- the very failure class D-E closes, missed on
            # the first pass because the gate inventory counted six and there are
            # eight.

            # Load correlation
            if "correlation" in unit_group.attrs:
                unit["correlation"] = float(unit_group.attrs["correlation"])

            # Load activation function (per unit), wrapped in ActivationWithDerivative for consistency with runtime-created units
            af_name = read_str_attr(unit_group, "activation_function_name", network.activation_function_name)
            if hasattr(network, "activation_functions_dict") and af_name in network.activation_functions_dict:
                unit["activation_fn"] = ActivationWithDerivative(network.activation_functions_dict[af_name])
            else:
                unit["activation_fn"] = ActivationWithDerivative(network.activation_fn) if not isinstance(network.activation_fn, ActivationWithDerivative) else network.activation_fn

            network.hidden_units.append(unit)

        self.logger.debug(f"CascadeHDF5Serializer: Loaded {num_units} hidden units")

    def _load_random_state(self, hdf5_file: h5py.File, network) -> None:
        """Load random state for deterministic reproducibility."""
        if "random" not in hdf5_file:
            return

        random_group = hdf5_file["random"]

        # Load random parameters
        network.random_seed = random_group.attrs.get("seed", network.random_seed)
        network.random_max_value = random_group.attrs.get("max_value", network.random_max_value)
        network.sequence_max_value = random_group.attrs.get("sequence_max_value", network.sequence_max_value)
        network.random_value_scale = random_group.attrs.get("value_scale", network.random_value_scale)

        # Restore RNG states
        try:
            # Python random state
            if "python_state" in random_group:
                python_state_array = load_numpy_array(random_group["python_state"])
                python_state_bytes = python_state_array.tobytes()
                # SEC-11: route RNG-state deserialization through the
                # restricted unpickler so even a tampered snapshot cannot
                # escalate to arbitrary code execution. ``find_class`` is
                # locked to the ``random``/``_random`` modules and a small
                # set of builtin container types; anything else raises
                # ``SnapshotUnpicklingError`` and aborts the restore.
                python_state = _snapshot_restricted_loads(python_state_bytes)
                random.setstate(python_state)
                self.logger.debug("CascadeHDF5Serializer: Restored Python random state")

            # NumPy random state
            if "numpy_state" in random_group:
                self._restore_np_random_state_helper(random_group)
            # PyTorch random state
            if "torch_state" in random_group:
                torch_state_array = load_numpy_array(random_group["torch_state"])
                torch_state = torch.from_numpy(torch_state_array).to(torch.uint8)
                torch.set_rng_state(torch_state)
                self.logger.debug("CascadeHDF5Serializer: Restored PyTorch random state")

            # CUDA random states
            if "cuda_states" in random_group and torch.cuda.is_available():
                cuda_group = random_group["cuda_states"]
                cuda_states = []
                i = 0
                while f"device_{i}" in cuda_group:
                    state_array = load_numpy_array(cuda_group[f"device_{i}"])
                    cuda_states.append(torch.from_numpy(state_array).to(torch.uint8))
                    i += 1
                if cuda_states:
                    torch.cuda.set_rng_state_all(cuda_states)
                    self.logger.debug(f"CascadeHDF5Serializer: Restored CUDA random states for {len(cuda_states)} devices")

        except Exception as e:
            self.logger.warning(f"Could not restore random states: {e}")
            import traceback

            self.logger.debug(traceback.format_exc())

        self.logger.debug("CascadeHDF5Serializer: Loaded random state")

    def _restore_np_random_state_helper(self, random_group):
        np_group = random_group["numpy_state"]
        state_type = read_str_attr(np_group, "state_type", "MT19937")
        state_array = load_numpy_array(np_group["state_array"])
        pos = np_group.attrs.get("pos", 0)
        has_gauss = np_group.attrs.get("has_gauss", 0)
        cached_gaussian = np_group.attrs.get("cached_gaussian", 0.0)

        np_state = (state_type, state_array, pos, has_gauss, cached_gaussian)
        np.random.set_state(np_state)
        self.logger.debug("CascadeHDF5Serializer: Restored NumPy random state")

    def _load_training_history(self, hdf5_file: h5py.File, network) -> None:  # noqa: C901
        """Load training history."""
        if "history" not in hdf5_file:
            return

        history_group = hdf5_file["history"]

        # Initialize history with network's actual keys. ``dataset_swaps``
        # is P2-2 (Issue #3) — empty list when loading a pre-P2-2 snapshot,
        # so the network attribute stays consistent with construction-time
        # defaults regardless of snapshot vintage.
        network.history = {
            "train_loss": [],
            "value_loss": [],  # Use value_* to match network.history
            "train_accuracy": [],
            "value_accuracy": [],  # Use value_* to match network.history
            "hidden_units_added": [],
            "dataset_swaps": [],
        }

        # Load numeric arrays - handle both old (val_*) and new (value_*) key formats
        key_mappings = [
            ("train_loss", "train_loss"),
            ("value_loss", "value_loss"),  # Prefer new format
            ("val_loss", "value_loss"),  # Fallback to old format
            ("train_accuracy", "train_accuracy"),
            ("value_accuracy", "value_accuracy"),  # Prefer new format
            ("val_accuracy", "value_accuracy"),  # Fallback to old format
        ]

        for save_key, network_key in key_mappings:
            if save_key in history_group and not network.history[network_key]:
                data = load_numpy_array(history_group[save_key])
                network.history[network_key] = data.tolist()
                # self.logger.debug(f"CascadeHDF5Serializer: Loaded history key '{save_key}' as '{network_key}'")  # B907
                self.logger.debug(f"CascadeHDF5Serializer: Loaded history key {save_key!r} as {network_key!r}")

        # Load hidden units history (supports both metadata-only and legacy weight/bias formats)
        if "hidden_units_added" in history_group:
            units_group = history_group["hidden_units_added"]
            for unit_name in sorted(units_group.keys()):
                unit_group = units_group[unit_name]
                unit_data = {}

                if "correlation" in unit_group.attrs:
                    unit_data["correlation"] = float(unit_group.attrs["correlation"])
                # New metadata-only fields (CR-063)
                if "weight_shape" in unit_group.attrs:
                    unit_data["weight_shape"] = tuple(unit_group.attrs["weight_shape"])
                if "unit_index" in unit_group.attrs:
                    unit_data["unit_index"] = int(unit_group.attrs["unit_index"])
                # Legacy weight/bias arrays (from old snapshots)
                if "weights" in unit_group:
                    unit_data["weights"] = load_numpy_array(unit_group["weights"])
                if "bias" in unit_group:
                    unit_data["bias"] = load_numpy_array(unit_group["bias"])

                network.history["hidden_units_added"].append(unit_data)

        # P2-2 (Issue #3): live dataset swap event history. Decoding lives
        # in ``_decode_dataset_swap_events_from_group`` so the P2-7 follow-up
        # endpoint (``GET /v1/snapshots/{id}/history/dataset_swaps``) can
        # read the same schema without instantiating a full network.
        if "dataset_swaps" in history_group:
            events = self._decode_dataset_swap_events_from_group(history_group["dataset_swaps"], logger=self.logger)
            network.history["dataset_swaps"].extend(events)
            self.logger.debug(f"CascadeHDF5Serializer: Loaded {len(events)} dataset_swap event(s)")

        # CAN-015g (g-1): per-sample weight history. Loaded into a sibling
        # ``weight_history`` attribute on the network so g-2's
        # ``_ReplaySession`` weight cache can consume it. Absent in V1
        # snapshots — silently no-op so V1 files load identically to
        # before this change.
        if "weights" in history_group:
            try:
                network.weight_history = self._load_weight_history(history_group["weights"])
            except Exception as e:
                # Don't fail the whole snapshot load if the weight history is
                # corrupt — degrade to V1 behaviour with a WARNING. The
                # replay session checks ``weights_available`` before using
                # this attribute.
                self.logger.warning(f"CascadeHDF5Serializer: Failed to load weight_history; degrading to V1 replay: {e}")
                network.weight_history = None

        self.logger.debug("CascadeHDF5Serializer: Loaded training history")

    @staticmethod
    def _decode_dataset_swap_events_from_group(swaps_group: h5py.Group, logger: Optional[Logger] = None) -> List[Dict[str, Any]]:
        """Decode the ``history/dataset_swaps`` event subgroup into a list.

        Schema and ordering match the writer in ``_save_training_history``:
        each ``event_<N>`` carries timestamp, before_cfg / after_cfg /
        arch_changes (JSON-encoded), and optional pre/post_swap_snapshot_id
        attrs. Events are returned in chronological order — sorted by the
        numeric ``<N>`` suffix, falling back to dictionary order when the
        suffix can't be parsed.

        A JSON decode error on one event degrades that event's bad field to
        its schema default (with a warning) so a single corrupt event can't
        kill the whole history load.
        """

        def _event_sort_key(name: str) -> int:
            try:
                return int(name.split("_", 1)[1])
            except (IndexError, ValueError):
                return -1

        events: List[Dict[str, Any]] = []
        for event_name in sorted(swaps_group.keys(), key=_event_sort_key):
            ev_group = swaps_group[event_name]
            event: Dict[str, Any] = {
                "timestamp": None,
                "before_cfg": None,
                "after_cfg": None,
                "arch_changes": {},
                "pre_swap_snapshot_id": None,
                "post_swap_snapshot_id": None,
            }
            if "timestamp" in ev_group.attrs:
                event["timestamp"] = read_str_attr(ev_group, "timestamp", None)
            for json_key, default in (("before_cfg", None), ("after_cfg", None), ("arch_changes", {})):
                if json_key in ev_group.attrs:
                    try:
                        event[json_key] = json.loads(read_str_attr(ev_group, json_key, "null"))
                    except (json.JSONDecodeError, ValueError) as exc:
                        if logger is not None:
                            logger.warning(f"CascadeHDF5Serializer: failed to decode {json_key!r} for {event_name!r}: {exc}; using default {default!r}")
                        event[json_key] = default
            if "pre_swap_snapshot_id" in ev_group.attrs:
                event["pre_swap_snapshot_id"] = read_str_attr(ev_group, "pre_swap_snapshot_id", None)
            if "post_swap_snapshot_id" in ev_group.attrs:
                event["post_swap_snapshot_id"] = read_str_attr(ev_group, "post_swap_snapshot_id", None)
            events.append(event)
        return events

    def read_dataset_swap_events(self, filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        """Read just the ``history/dataset_swaps`` events from a snapshot file.

        P2-7 follow-up (Issue #3): the ``GET /v1/snapshots/{id}/history/dataset_swaps``
        route uses this to surface a stored snapshot's own swap history to
        the canopy Replay timeline (parent spec §4.4 — markers on the
        loaded snapshot's timeline) without paying the cost of a full
        network restore.

        Returns ``[]`` when the snapshot has no ``history`` group or no
        ``dataset_swaps`` subgroup — both legitimate cases (pre-P2-2
        snapshots, or training runs with no live swap).
        """
        with h5py.File(filepath, "r") as hdf5_file:
            if "history" not in hdf5_file:
                return []
            history_group = hdf5_file["history"]
            if "dataset_swaps" not in history_group:
                return []
            return self._decode_dataset_swap_events_from_group(history_group["dataset_swaps"], logger=self.logger)

    def _load_weight_history(self, weights_group: h5py.Group) -> Dict[str, Any]:
        """Restore per-sample weight history written by ``_save_weight_history``.

        Returns the same dict shape that the writer expects, so the lifecycle
        layer can ``network.weight_history = serializer.load(...)`` round-trip
        without re-shaping.
        """
        meta_group = weights_group.get("meta")
        if meta_group is None:
            raise ValueError("weight_history is missing required 'meta' subgroup")

        schema_version = int(meta_group.attrs.get("schema_version", 0))
        if schema_version != self._WEIGHT_HISTORY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported weight_history schema_version: {schema_version} (expected {self._WEIGHT_HISTORY_SCHEMA_VERSION})")

        result: Dict[str, Any] = {
            "schema_version": schema_version,
            "sampling_strategy": read_str_attr(meta_group, "sampling_strategy", "adaptive"),
            "sampling_interval": int(meta_group.attrs.get("sampling_interval", 0)),
            "sample_indices": [],
            "output_weights": [],
            "output_bias": [],
            "hidden_units": [],
        }

        if "sample_indices" not in weights_group:
            return result

        result["sample_indices"] = load_numpy_array(weights_group["sample_indices"]).astype(np.int64).tolist()

        ow_group = weights_group.get("output_weights")
        ob_group = weights_group.get("output_bias")
        if ow_group is not None and ob_group is not None:
            for sample_key in sorted(ow_group.keys()):
                result["output_weights"].append(load_numpy_array(ow_group[sample_key]))
                result["output_bias"].append(load_numpy_array(ob_group[sample_key]))

        units_group = weights_group.get("hidden_units")
        if units_group is not None:
            for unit_key in sorted(units_group.keys()):
                unit_group = units_group[unit_key]
                w_subgroup = unit_group.get("weights")
                b_subgroup = unit_group.get("bias")
                unit_weights = []
                unit_bias = []
                if w_subgroup is not None and b_subgroup is not None:
                    for sample_key in sorted(w_subgroup.keys()):
                        unit_weights.append(load_numpy_array(w_subgroup[sample_key]))
                        unit_bias.append(load_numpy_array(b_subgroup[sample_key]))
                result["hidden_units"].append(
                    {
                        "first_sample_index": int(unit_group.attrs.get("first_sample_index", 0)),
                        "activation": read_str_attr(unit_group, "activation", ""),
                        "weights": unit_weights,
                        "bias": unit_bias,
                    }
                )

        return result

    def _restore_multiprocessing_state(self, hdf5_file: h5py.File, network) -> None:
        """Restore multiprocessing state."""
        if "mp" not in hdf5_file:
            return
        mp_group = hdf5_file["mp"]
        try:
            self._restore_multiprocessing_state_helper(mp_group, network)
        except Exception as e:
            self.logger.warning(f"Could not restore multiprocessing state: {e}")

    def _restore_multiprocessing_state_helper(self, mp_group, network):
        # Load MP configuration
        role = read_str_attr(mp_group, "role", "none")
        start_method = read_str_attr(mp_group, "start_method", "spawn")
        address_host = read_str_attr(mp_group, "address_host", "127.0.0.1")
        address_port = mp_group.attrs.get("address_port", 0)
        authkey_hex = read_str_attr(mp_group, "authkey_hex", "")
        autostart = mp_group.attrs.get("autostart", True)

        # Restore timeouts
        network.candidate_training_tasks_queue_timeout = mp_group.attrs.get("tasks_queue_timeout", 30.0)
        network.candidate_training_shutdown_timeout = mp_group.attrs.get("shutdown_timeout", 10.0)

        # Set multiprocessing context
        network.candidate_training_context = mp.get_context(start_method)

        # Restore address and authkey
        network.candidate_training_queue_address = (address_host, address_port)
        if authkey_hex:
            try:
                network.candidate_training_queue_authkey = bytes.fromhex(authkey_hex)
            except ValueError:
                network.candidate_training_queue_authkey = authkey_hex.encode("utf-8")

        # Recreate multiprocessing components based on role
        if role == "server" and autostart:
            # Reinitialize as server
            network._init_multiprocessing()

        self.logger.debug(f"CascadeHDF5Serializer: Restored multiprocessing state (role: {role})")

    def _sanitize_config_dict(self, config_dict: Dict[str, Any], config_class) -> Dict[str, Any]:
        """Make a snapshot's stored config safe to pass to ``config_class(**config_dict)``.

        TWO filters, and neither subsumes the other.

        1. **Named drops** -- fields the config class DOES accept, but whose serialized
           form is unusable. ``_save_configuration`` serialized with ``default=str``, so an
           activation-function map or a log config comes back as a repr string rather than
           the live object; passing those through builds a network holding junk. They must
           go by name, because an allowlist would keep them.

        2. **An allowlist from the class itself** -- drop any field the CURRENT class no
           longer accepts. Without this, ``config_class(**config_dict)`` raises TypeError on
           strict keyword matching and the whole load fails with the generic "snapshot could
           not be deserialized into a network" -- a message naming nothing, for a snapshot
           that is perfectly intact. Measured 2026-08-22: 14 archive snapshots written by
           cascor 0.3.2 carry ``optimizer_config``, a real field then, since removed.

           The named drops cannot cover this. A denylist has to be extended by hand at every
           field removal and nothing prompts anyone to do so, so each removal silently bricks
           whichever slice of the archive still carries that field. Snapshots are long-lived
           project assets; the config schema is not frozen.

        Guarded two ways, because a filter with a wrong allowlist drops REAL config silently
        -- worse than the TypeError it fixes. If ``__init__`` reports ``**kwargs`` (a
        decorator without ``functools.wraps`` is enough), every name is legal, TypeError
        cannot happen, and filtering would strip the entire config -- so that case skips the
        filter. An uninspectable constructor falls back to the pre-existing behaviour rather
        than making every snapshot unloadable.
        """
        config_dict.pop("activation_functions_dict", None)
        config_dict.pop("log_config", None)
        config_dict.pop("logger", None)
        # Remove runtime-only attributes not in CascadeCorrelationConfig
        config_dict.pop("candidates_per_layer", None)
        config_dict.pop("layer_selection_strategy", None)

        try:
            parameters = inspect.signature(config_class.__init__).parameters
        except (TypeError, ValueError) as exc:  # pragma: no cover - pure-Python class, but a bad allowlist must never win
            self.logger.warning(f"CascadeHDF5Serializer: could not introspect {config_class.__name__} ({exc}); passing config through unfiltered")
            return config_dict
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return config_dict

        accepted = {name for name in parameters if name != "self"}
        dropped = sorted(set(config_dict) - accepted)
        if dropped:
            self.logger.warning(f"CascadeHDF5Serializer: dropping {len(dropped)} config field(s) this version no longer accepts: {dropped}")
            config_dict = {key: value for key, value in config_dict.items() if key in accepted}
        return config_dict

    def _create_network_from_file(self, hdf5_file: h5py.File):
        """Create a network instance from HDF5 configuration."""
        try:
            from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
            from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

            if "config" not in hdf5_file:
                self.logger.error("No configuration found in file")
                return None
            config_group = hdf5_file["config"]
            if "config_json" in config_group:
                config_json = read_str_dataset(config_group, "config_json")
                config_dict = self._sanitize_config_dict(json.loads(config_json), CascadeCorrelationConfig)
                if "meta" in hdf5_file:
                    meta_group = hdf5_file["meta"]
                    if saved_uuid := read_str_attr(meta_group, "uuid", None):
                        config_dict["uuid"] = saved_uuid
                        self.logger.debug(f"CascadeHDF5Serializer: Injecting UUID {saved_uuid} into config")
                config = CascadeCorrelationConfig(**config_dict)
            else:
                config = CascadeCorrelationConfig()
                for attr_name in config_group.attrs.keys():
                    if hasattr(config, attr_name):
                        setattr(config, attr_name, config_group.attrs[attr_name])
                if "meta" in hdf5_file:
                    meta_group = hdf5_file["meta"]
                    if saved_uuid := read_str_attr(meta_group, "uuid", None):
                        config.uuid = saved_uuid
                        self.logger.debug(f"CascadeHDF5Serializer: Setting UUID {saved_uuid} on config")
            network = CascadeCorrelationNetwork(config=config)
            if "meta" in hdf5_file:
                self._restore_training_state_helper(hdf5_file, network)
            self.logger.debug(f"CascadeHDF5Serializer: Created network instance (UUID: {network.get_uuid()})")
            return network
        except Exception as e:
            return self._log_exception_stacktrace("Could not create network from file: ", e, None)

    def _restore_training_state_helper(self, hdf5_file, network):
        meta_group = hdf5_file["meta"]
        network.snapshot_counter = meta_group.attrs.get("snapshot_counter", 0)
        if "current_epoch" in meta_group.attrs:
            network.current_epoch = meta_group.attrs.get("current_epoch", 0)
        network.patience_counter = meta_group.attrs.get("patience_counter", 0)
        network.best_value_loss = meta_group.attrs.get("best_value_loss", float("inf"))
        self.logger.debug(f"CascadeHDF5Serializer: Restored training counters - snapshot: {network.snapshot_counter}, patience: {network.patience_counter}")

    def _verify_output_checksums(self, hdf5_file: h5py.File, network) -> List[Tuple[str, str]]:
        """Re-hash the output-layer tensors and compare against the stored checksums.

        A mismatch is positive evidence that the bytes on disk no longer describe the
        tensors that were saved. Returns one finding per failing tensor; an absent or
        unreadable checksum block yields none (nothing to compare against is not the
        same as evidence of damage).
        """
        findings: List[Tuple[str, str]] = []
        try:
            output_group = hdf5_file["params"]["output_layer"]
        except (KeyError, TypeError):
            return findings
        if "checksums" not in output_group:
            return findings
        try:
            checksums = json.loads(read_str_dataset(output_group, "checksums"))
        except (ValueError, TypeError) as exc:
            self.logger.warning(f"CascadeHDF5Serializer: Could not read checksums: {exc}")
            return findings

        for key, attr in (("output_weights", "output_weights"), ("output_bias", "output_bias")):
            expected = checksums.get(key)
            tensor = getattr(network, attr, None)
            if expected is None or tensor is None:
                continue
            if not verify_tensor_checksum(tensor, expected):
                findings.append((SNAPSHOT_CORRUPT, f"{attr} checksum mismatch — the stored tensor does not match its recorded checksum"))
        return findings

    def _verify_hidden_unit_checksums(self, hdf5_file: h5py.File, network) -> List[Tuple[str, str]]:
        """Re-hash each hidden unit's tensors against its stored checksums.

        The output layer is not the only thing with checksums. A same-shape tamper of a
        hidden unit's weights passes every shape check, so before this ran as a gate it
        loaded cleanly with nothing but an ERROR line in the log — measured, not
        assumed.
        """
        findings: List[Tuple[str, str]] = []
        if "hidden_units" not in hdf5_file:
            return findings
        hidden_group = hdf5_file["hidden_units"]
        for i, unit in enumerate(getattr(network, "hidden_units", []) or []):
            unit_group = hidden_group.get(f"unit_{i}")
            if unit_group is None or "checksums" not in unit_group:
                continue
            try:
                checksums = json.loads(read_str_dataset(unit_group, "checksums"))
            except (ValueError, TypeError) as exc:
                self.logger.warning(f"CascadeHDF5Serializer: Could not read checksums for hidden unit {i}: {exc}")
                continue
            for field in ("weights", "bias"):
                expected = checksums.get(field)
                tensor = unit.get(field) if isinstance(unit, dict) else None
                if expected is None or tensor is None:
                    continue
                if not verify_tensor_checksum(tensor, expected):
                    findings.append((SNAPSHOT_CORRUPT, f"hidden unit {i} {field} checksum mismatch — the stored tensor does not match its recorded checksum"))
        return findings

    def _check_integrity(self, hdf5_file: h5py.File, network) -> List[Tuple[str, str]]:
        """Run every load-time integrity gate and report what failed (D-E).

        EIGHT gates, not six: the first D-E pass counted the output-layer checksums and
        missed the per-hidden-unit pair, which meant a same-shape tamper of a hidden
        unit still loaded. Count them here, in one place, so the next addition cannot
        be missed the same way.

        These checks all existed before; none of them stopped anything. The loader ran
        six of them, logged (two at ERROR), and returned the network anyway — then
        logged ``Successfully loaded network`` on the next line. The gates are now
        collected here so ``load_network`` can act on them once.

        Why that mattered: three shape-violation classes raise later, but a
        hidden-unit weight vector of length 1 is broadcast-compatible with the slice it
        multiplies, so the network computes a *different answer* with no error anywhere
        — it trains, reports a plausible loss, and can be re-snapshotted, propagating
        the corruption.

        Returns:
            A list of ``(status, detail)`` findings; empty means every gate passed.
        """
        findings: List[Tuple[str, str]] = []

        # Gates 1-2 — the declared architecture vs the network that was actually built.
        # ``_create_network_from_file`` builds from the ``config`` group while the
        # tensors come from ``params``; when those disagree nothing reconciles them, so
        # a snapshot can describe two different networks at once.
        if "arch" in hdf5_file:
            arch_group = hdf5_file["arch"]
            for attr, actual in (("input_size", network.input_size), ("output_size", network.output_size)):
                saved = arch_group.attrs.get(attr, actual)
                if saved != actual:
                    findings.append((SNAPSHOT_ARCH_MISMATCH, f"{attr} disagrees: the snapshot's arch group says {saved}, the network built from its config is {actual}"))

        # Gates 3-6 — checksum verification, output layer AND every hidden unit.
        findings.extend(self._verify_output_checksums(hdf5_file, network))
        findings.extend(self._verify_hidden_unit_checksums(hdf5_file, network))

        # Gate 5 — tensor shapes against the network's declared dimensions.
        if not self._validate_shapes(network):
            findings.append((SNAPSHOT_CORRUPT, "tensor shapes are inconsistent with the network's declared dimensions (see the shape errors logged above)"))

        return findings

    def _validate_shapes(self, network) -> bool:
        """
        Validate tensor shapes match expected dimensions.

        Args:
            network: CascadeCorrelationNetwork instance to validate

        Returns:
            bool: True if all shapes are valid, False otherwise
        """
        try:
            expected_output_input = network.input_size + len(network.hidden_units)
            if network.output_weights.shape != (
                expected_output_input,
                network.output_size,
            ):
                self.logger.error(f"Output weights shape mismatch: {network.output_weights.shape} != ({expected_output_input}, {network.output_size})")
                return False
            if network.output_bias.shape != (network.output_size,):
                self.logger.error(f"Output bias shape mismatch: {network.output_bias.shape} != ({network.output_size},)")
                return False
            for i, unit in enumerate(network.hidden_units):
                expected_input_size = network.input_size + i
                if "weights" in unit and unit["weights"].shape[0] != expected_input_size:
                    self.logger.error(f"Hidden unit {i} weight shape mismatch: {unit['weights'].shape[0]} != {expected_input_size}")
                    return False
                if "bias" in unit and unit["bias"].numel() != 1:
                    self.logger.error(f"Hidden unit {i} bias shape mismatch: {unit['bias'].shape} should be scalar or (1,)")
                    return False
            self.logger.debug("CascadeHDF5Serializer: Shape validation passed")
            return True
        except Exception as e:
            return self._log_exception_stacktrace("Shape validation failed: ", e, False)

    def _validate_format(self, hdf5_file: h5py.File) -> bool:
        """
        Validate HDF5 file format with comprehensive checks.

        Thin wrapper over :meth:`_validate_format_detail`, kept because callers and
        tests treat format validation as a boolean gate.

        Returns:
            bool: True if file format is valid, False otherwise
        """
        return self._validate_format_detail(hdf5_file) is None

    def _reject_format(self, message: str) -> str:
        """Log a format rejection and return it as the reason string."""
        self.logger.error(message)
        return message

    def _validate_format_detail(self, hdf5_file: h5py.File) -> Optional[str]:  # noqa: C901 - validation requires multiple checks
        """
        Validate HDF5 file format, returning WHICH check failed.

        Validates:
        - Format name and version compatibility
        - Required groups and datasets
        - Hidden units consistency
        - Parameter dataset shapes

        Every branch already produced a precise error message; previously all of them
        collapsed into a bare ``False``, and ``verify_saved_network`` reported the
        result to the operator as ``'Invalid format'`` — pointing at the format string
        even when the real problem was, say, a missing ``params`` group.

        Returns:
            None if the format is valid, otherwise the reason it was rejected.
        """
        try:
            # Check format identifier
            format_name = read_str_attr(hdf5_file, "format")
            if format_name not in [
                self.format_name,
                _HDF5_FORMAT_NAME_LEGACY,
                _HDF5_FORMAT_NAME_CURRENT,
            ]:
                return self._reject_format(f"Invalid format: {format_name}")

            # Check format version compatibility
            format_version = read_str_attr(hdf5_file, "format_version", "1")
            try:
                file_major_version = int(format_version.split(".")[0] if "." in format_version else format_version)
                serializer_major_version = int(self.format_version.split(".")[0])

                if file_major_version > serializer_major_version:
                    return self._reject_format(f"Incompatible format version: file={format_version}, " f"serializer={self.format_version}")
            except (ValueError, IndexError):
                self.logger.warning(f"Could not parse format version: {format_version}")

            # Check for required groups
            required_groups = ["meta", "config", "params", "arch", "random"]
            for group in required_groups:
                if group not in hdf5_file:
                    return self._reject_format(f"Missing required group: {group}")

            # Check for required datasets in params group
            if "params" in hdf5_file:
                params_group = hdf5_file["params"]
                if "output_layer" in params_group:
                    output_group = params_group["output_layer"]
                    if "weights" not in output_group:
                        return self._reject_format("Missing output layer weights dataset")
                    if "bias" not in output_group:
                        return self._reject_format("Missing output layer bias dataset")
                else:
                    return self._reject_format("Missing output_layer group in params")

            # Verify hidden units consistency
            if "hidden_units" in hdf5_file:
                hidden_group = hdf5_file["hidden_units"]
                num_units_attr = hidden_group.attrs.get("num_units", 0)
                actual_units = len([k for k in hidden_group.keys() if k.startswith("unit_")])

                if num_units_attr != actual_units:
                    return self._reject_format(f"Hidden units count mismatch: num_units={num_units_attr}, " f"actual groups={actual_units}")

                # Verify each hidden unit has required datasets
                for i in range(num_units_attr):
                    unit_name = f"unit_{i}"
                    if unit_name in hidden_group:
                        unit_group = hidden_group[unit_name]
                        if "weights" not in unit_group:
                            return self._reject_format(f"Hidden unit {i} missing weights dataset")
                        if "bias" not in unit_group:
                            return self._reject_format(f"Hidden unit {i} missing bias dataset")

            self.logger.debug("CascadeHDF5Serializer: Format validation passed")
            return None

        except Exception as e:
            return self._log_exception_stacktrace("Format validation failed: ", e, f"format validation raised: {e}")

    def _log_exception_stacktrace(self, arg0, e, arg2):
        self.logger.error(f"{arg0}{e}")
        import traceback

        self.logger.debug(traceback.format_exc())
        return arg2

    def _config_to_dict(self, config) -> Dict[str, Any]:
        """
        Convert configuration object to dictionary.

        Excludes non-serializable objects like callables, log_config, and activation_functions_dict.
        Only includes primitive types and serializable containers that can be safely JSON-encoded.
        """
        config_dict = {}

        # Whitelist of safe serializable attributes
        excluded_attrs = {
            "activation_functions_dict",  # Contains callable functions
            "log_config",  # Complex logging object, will be recreated on load
            "logger",  # Runtime object
        }

        # Get all attributes from config object
        for attr_name in dir(config):
            if attr_name.startswith("_") or attr_name in excluded_attrs:
                continue
            try:
                attr_value = getattr(config, attr_name)

                # Skip callable attributes
                if callable(attr_value):
                    continue

                # Handle different types
                if isinstance(attr_value, (str, int, float, bool, type(None))):
                    config_dict[attr_name] = attr_value
                elif isinstance(attr_value, (list, tuple)):

                    # Only include if items are primitive types
                    if all(isinstance(item, (str, int, float, bool, type(None))) for item in attr_value):
                        config_dict[attr_name] = list(attr_value)
                elif isinstance(attr_value, dict):

                    # Only include if values are primitive types
                    if all(isinstance(v, (str, int, float, bool, type(None))) for v in attr_value.values()):
                        config_dict[attr_name] = dict(attr_value)
                elif isinstance(attr_value, pl.Path):
                    config_dict[attr_name] = str(attr_value)
                # Skip other complex types

            except Exception as e:
                self.logger.debug(f"Skipping attribute {attr_name}: {e}")
                continue

        return config_dict
