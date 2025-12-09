#!/usr/bin/env python
"""
HDF5 Serializer for CascadeCorrelationNetwork objects.
Provides comprehensive state capture and restoration with full multiprocessing support.
"""

import datetime
import json
import multiprocessing as mp
import os
import pathlib as pl
import pickle # trunk-ignore(bandit/B403)
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py
import numpy as np
import torch

# Add parent directories for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from log_config.logger.logger import Logger

from .snapshot_common import (
    calculate_tensor_checksum,
    load_numpy_array,
    load_tensor,
    read_str_attr,
    read_str_dataset,
    save_numpy_array,
    save_tensor,
    verify_tensor_checksum,
    write_str_attr,
    write_str_dataset,
)


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
        self.format_name = "juniper.cascor"

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
            bool: Success status
        """
        try:
            self.logger.info(f"CascadeHDF5Serializer: Saving network to {filepath}")

            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(filepath, "w") as hdf5_file:
                self._save_network_objects_helper(
                    hdf5_file, network, compression, compression_opts
                )
                # Save training history if requested
                if include_training_state:
                    self._save_training_history(
                        hdf5_file, network, compression, compression_opts
                    )

                # Save training data if explicitly requested (normally excluded)
                if include_training_data:
                    self._save_training_data(
                        hdf5_file, network, compression, compression_opts
                    )

            self.logger.info( f"CascadeHDF5Serializer: Successfully saved network to {filepath}")
            return True

        except Exception as e:
            return self._log_exception_stacktrace( 'CascadeHDF5Serializer: Error saving network: ', e, False)

    def save_object(
        self,
        objectify: any = None,
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
                self._save_root_attributes( hdf5_file, objectify, compression, compression_opts)
            self.logger.info( f"CascadeHDF5Serializer: Successfully saved object to {filepath}")
            return True

        except Exception as e:
            return self._log_exception_stacktrace( 'CascadeHDF5Serializer: Error saving object: ', e, False)

    def _save_root_attributes(self, hdf5_file: h5py.File, network) -> None:
        """Save root-level file attributes."""
        write_str_attr(hdf5_file, "format", self.format_name)
        write_str_attr(hdf5_file, "format_version", self.format_version)
        write_str_attr(hdf5_file, "serializer_version", self.version)
        write_str_attr(hdf5_file, "created", datetime.datetime.now().isoformat())
        write_str_attr(hdf5_file, "juniper_version", "0.3.2")
        self.logger.debug( "CascadeHDF5Serializer: _save_root_attributes: Saved root attributes")

    def _save_metadata(self, hdf5_file: h5py.File, network) -> None:
        """Save metadata information."""
        meta_group = hdf5_file.create_group("meta")

        # Object metadata
        write_str_attr(meta_group, "uuid", str(network.get_uuid()))
        write_str_attr( meta_group, "creation_timestamp", datetime.datetime.now().isoformat())

        # Training state counters for resuming training
        meta_group.attrs["snapshot_counter"] = getattr(network, "snapshot_counter", 0)
        meta_group.attrs["current_epoch"] = getattr(network, "current_epoch", 0)
        meta_group.attrs["patience_counter"] = getattr(network, "patience_counter", 0)
        meta_group.attrs["best_value_loss"] = getattr( network, "best_value_loss", float("inf"))

        # Environment metadata
        write_str_attr(meta_group, "python_version", sys.version)
        write_str_attr(meta_group, "torch_version", torch.__version__)
        write_str_attr(meta_group, "h5py_version", h5py.__version__)
        self.logger.debug( "CascadeHDF5Serializer: _save_metadata: Saved metadata with training counters")

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
                if not self._validate_format(hdf5_file):
                    return {"valid": False, "error": "Invalid format"}

                # Extract summary information
                summary = {
                    "valid": True,
                    "format": read_str_attr(hdf5_file, "format", "unknown"),
                    "format_version": read_str_attr( hdf5_file, "format_version", "unknown"),
                    "serializer_version": read_str_attr( hdf5_file, "serializer_version", "unknown"),
                    "created": read_str_attr(hdf5_file, "created", "unknown"),
                    "file_size": os.path.getsize(filepath),
                }

                # Get metadata if available
                if "meta" in hdf5_file:
                    meta_group = hdf5_file["meta"]
                    summary["network_uuid"] = read_str_attr( meta_group, "uuid", "unknown")
                    summary["python_version"] = read_str_attr( meta_group, "python_version", "unknown")
                    summary["torch_version"] = read_str_attr( meta_group, "torch_version", "unknown")

                # Get architecture if available
                if "arch" in hdf5_file:
                    arch_group = hdf5_file["arch"]
                    summary["input_size"] = arch_group.attrs.get("input_size", 0)
                    summary["output_size"] = arch_group.attrs.get("output_size", 0)
                    summary["num_hidden_units"] = arch_group.attrs.get( "num_hidden_units", 0)
                    summary["activation_function"] = read_str_attr( arch_group, "activation_function_name", "unknown")

                # Check for optional sections
                optional_sections = ["history", "mp", "data"]
                for section in optional_sections:
                    summary[f"has_{section}"] = section in hdf5_file

                return summary

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _save_root_attributes(self, hdf5_file: h5py.File, network) -> None:
        """Save root-level file attributes."""
        write_str_attr(hdf5_file, "format", self.format_name)
        write_str_attr(hdf5_file, "format_version", self.format_version)
        write_str_attr(hdf5_file, "serializer_version", self.version)
        write_str_attr(hdf5_file, "created", datetime.datetime.now().isoformat())
        write_str_attr(hdf5_file, "juniper_version", "0.3.2")

        self.logger.debug("CascadeHDF5Serializer: Saved root attributes")

    def _save_metadata(self, hdf5_file: h5py.File, network) -> None:
        """Save metadata information."""
        meta_group = hdf5_file.create_group("meta")

        # Network metadata
        write_str_attr(meta_group, "uuid", str(network.get_uuid()))
        write_str_attr(meta_group, "creation_timestamp", str(datetime.datetime.now()))

        # Environment metadata
        write_str_attr(meta_group, "python_version", sys.version)
        write_str_attr(meta_group, "torch_version", torch.__version__)
        write_str_attr(meta_group, "h5py_version", h5py.__version__)

        # Network statistics
        meta_group.attrs["num_hidden_units"] = len(network.hidden_units)
        meta_group.attrs["input_size"] = network.input_size
        meta_group.attrs["output_size"] = network.output_size

        # Training state counters for resuming training
        meta_group.attrs["snapshot_counter"] = getattr(network, "snapshot_counter", 0)
        meta_group.attrs["current_epoch"] = getattr(network, "current_epoch", 0)
        meta_group.attrs["patience_counter"] = getattr(network, "patience_counter", 0)
        meta_group.attrs["best_value_loss"] = getattr( network, "best_value_loss", float("inf"))

        self.logger.debug( "CascadeHDF5Serializer: Saved metadata with training counters")

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
        write_str_attr( config_group, "activation_function_name", network.activation_function_name)
        config_group.attrs["input_size"] = network.input_size
        config_group.attrs["output_size"] = network.output_size
        config_group.attrs["learning_rate"] = network.learning_rate
        config_group.attrs["candidate_learning_rate"] = network.candidate_learning_rate
        config_group.attrs["max_hidden_units"] = network.max_hidden_units
        config_group.attrs["correlation_threshold"] = network.correlation_threshold
        config_group.attrs["candidate_pool_size"] = network.candidate_pool_size
        config_group.attrs["patience"] = network.patience
        config_group.attrs["random_seed"] = network.random_seed

        self.logger.debug("CascadeHDF5Serializer: Saved configuration")

    def _save_architecture(self, hdf5_file: h5py.File, network) -> None:
        """Save network architecture information."""
        arch_group = hdf5_file.create_group("arch")

        # Basic architecture parameters
        arch_group.attrs["input_size"] = network.input_size
        arch_group.attrs["output_size"] = network.output_size
        arch_group.attrs["num_hidden_units"] = len(network.hidden_units)
        arch_group.attrs["max_hidden_units"] = network.max_hidden_units
        write_str_attr( arch_group, "activation_function_name", network.activation_function_name)

        # Save connectivity information if needed
        connectivity_group = arch_group.create_group("connectivity")
        connectivity_group.attrs["input_to_output_connections"] = ( network.input_size * network.output_size)

        # Hidden unit connectivity
        for i, unit in enumerate(network.hidden_units):
            unit_info = connectivity_group.create_group(f"hidden_unit_{i}")
            unit_info.attrs["input_connections"] = ( len(unit["weights"]) if "weights" in unit else 0)
            if "activation_fn" in unit:
                write_str_attr(
                    unit_info,
                    "activation_function",
                    getattr(unit["activation_fn"],"__name__", "unknown"),
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
            save_tensor( output_group, "bias", network.output_bias, compression, compression_opts)

        # Calculate and save checksums
        checksum_data = {}
        if hasattr(network, "output_weights") and network.output_weights is not None:
            checksum_data["output_weights"] = calculate_tensor_checksum( network.output_weights)
        if hasattr(network, "output_bias") and network.output_bias is not None:
            checksum_data["output_bias"] = calculate_tensor_checksum( network.output_bias)

        if checksum_data:
            write_str_dataset(output_group, "checksums", json.dumps(checksum_data))
            self.logger.debug("CascadeHDF5Serializer: Saved parameter checksums")

        # Save optimizer state if it exists
        if ( hasattr(network, "output_optimizer") and network.output_optimizer is not None):
            opt_group = output_group.create_group("optimizer")
            try:
                self._save_network_parameters_to_hdf5_helper(network, opt_group)
            except Exception as e:
                self.logger.warning( f"CascadeHDF5Serializer: Could not save optimizer state: {e}")

        self.logger.debug("CascadeHDF5Serializer: Saved parameters")

    def _save_network_parameters_to_hdf5_helper(self, network, opt_group):
        opt_state = network.output_optimizer.state_dict()
        # Convert optimizer state to JSON-serializable format
        opt_state_serializable = {
            "state": {str(k): {inner_k: (inner_v.tolist() if hasattr(inner_v, "tolist") else inner_v) for inner_k, inner_v in v.items()} for k, v in opt_state.get("state", {}).items()},
            "param_groups": opt_state.get("param_groups", []),
        }
        write_str_dataset( opt_group, "state_dict", json.dumps(opt_state_serializable))
        write_str_attr( opt_group, "optimizer_type", type(network.output_optimizer).__name__)
        write_str_attr(opt_group, "learning_rate", network.learning_rate)
        self.logger.debug("CascadeHDF5Serializer: Saved optimizer state")

    def _save_hidden_units(
        self, hdf5_file: h5py.File, network, compression: str, compression_opts: int
    ) -> None:
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
                save_tensor( unit_group, "bias", unit["bias"], compression, compression_opts)

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
                af_name = getattr( unit["activation_fn"], "__name__", network.activation_function_name)
                write_str_attr(unit_group, "activation_function_name", af_name)
            else:
                write_str_attr( unit_group, "activation_function_name", network.activation_function_name,)

        self.logger.debug( f"CascadeHDF5Serializer: Saved {len(network.hidden_units)} hidden units with checksums")

    def _save_random_state(
        self, hdf5_file: h5py.File, network, compression: str, compression_opts: int
    ) -> None:
        """Save random state for deterministic reproducibility."""
        random_group = hdf5_file.create_group("random")

        # Save random parameters
        random_group.attrs["seed"] = getattr(network, "random_seed", 0)
        random_group.attrs["max_value"] = getattr(network, "random_max_value", 1000000)
        random_group.attrs["sequence_max_value"] = getattr( network, "sequence_max_value", 1000000)
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
            save_numpy_array( np_group, "state_array", np_state[1], compression, compression_opts)
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

            self.logger.debug( "CascadeHDF5Serializer: Saved all random states (Python, NumPy, PyTorch, CUDA)")

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
        if ( hasattr(network, "candidate_training_manager") and network.candidate_training_manager):
            role = "server"
        elif ( hasattr(network, "candidate_training_queue_address") and network.candidate_training_queue_address):
            role = "client"

        write_str_attr(mp_group, "role", role)

        # Save multiprocessing context information
        if hasattr(network, "candidate_training_context"):
            ctx = network.candidate_training_context
            write_str_attr( mp_group, "start_method", ctx.get_start_method() if ctx else "spawn")
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
            authkey_hex = ( authkey.hex() if isinstance(authkey, bytes) else str(authkey))
            write_str_attr(mp_group, "authkey_hex", authkey_hex)

        # Save timeouts
        if hasattr(network, "candidate_training_tasks_queue_timeout"):
            mp_group.attrs["tasks_queue_timeout"] = float( network.candidate_training_tasks_queue_timeout)
        if hasattr(network, "candidate_training_shutdown_timeout"):
            mp_group.attrs["shutdown_timeout"] = float( network.candidate_training_shutdown_timeout)

        # Save queue configuration
        queues_config = {"task_queue": "BaseManager", "result_queue": "BaseManager"}
        write_str_dataset(mp_group, "queues_to_create", json.dumps(queues_config))

        # Save policy flags
        mp_group.attrs["autostart"] = True  # Default to autostart on restore

    def _save_training_history(
        self, hdf5_file: h5py.File, network, compression: str, compression_opts: int
    ) -> None:
        """Save training history."""
        if not hasattr(network, "history") or not network.history:
            return

        history_group = hdf5_file.create_group("history")

        # Save numeric arrays - use network's actual keys (value_* not val_*)
        key_mapping = {
            "train_loss": "train_loss",
            "value_loss": "value_loss",  # Match network.history keys
            "train_accuracy": "train_accuracy",
            "value_accuracy": "value_accuracy",  # Match network.history keys
        }

        for network_key, save_key in key_mapping.items():
            if network_key in network.history and network.history[network_key]:
                data = np.array(network.history[network_key])
                save_numpy_array( history_group, save_key, data, compression, compression_opts)

        # Save hidden units added history
        if "hidden_units_added" in network.history:
            units_group = history_group.create_group("hidden_units_added")
            for i, unit_data in enumerate(network.history["hidden_units_added"]):
                unit_group = units_group.create_group(f"unit_{i}")
                if isinstance(unit_data, dict):
                    if "correlation" in unit_data:
                        unit_group.attrs["correlation"] = float( unit_data["correlation"])
                    if "weights" in unit_data:
                        save_numpy_array( unit_group, "weights", unit_data["weights"], compression, compression_opts,)
                    if "bias" in unit_data:
                        save_numpy_array( unit_group, "bias", unit_data["bias"], compression, compression_opts,)

        self.logger.debug("CascadeHDF5Serializer: Saved training history")

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
                    save_numpy_array( data_group, key, dataset, compression, compression_opts)
        self.logger.debug("CascadeHDF5Serializer: Saved training data")

    def load_network(self, filepath: Union[str, Path], restore_multiprocessing: bool = True) -> Optional:
        """
        Load a CascadeCorrelationNetwork from HDF5 format.

        Args:
            filepath: Path to HDF5 file
            restore_multiprocessing: Whether to restore multiprocessing state

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
                self._load_parameters(hdf5_file, network)
                self._load_hidden_units(hdf5_file, network)
                self._load_random_state(hdf5_file, network)
                if "history" in hdf5_file:
                    self._load_training_history(hdf5_file, network)
                if restore_multiprocessing and "mp" in hdf5_file:
                    self._restore_multiprocessing_state(hdf5_file, network)
                if not self._validate_shapes(network):
                    self.logger.warning( "CascadeHDF5Serializer: Network loaded but shape validation found issues")
            self.logger.info( f"CascadeHDF5Serializer: Successfully loaded network from {filepath}")
            return network
        except Exception as e:
            return self._log_exception_stacktrace( 'CascadeHDF5Serializer: Error loading network: ', e, None)

    def _load_architecture(self, hdf5_file: h5py.File, network) -> None:
        """Load network architecture."""
        if "arch" not in hdf5_file:
            return

        arch_group = hdf5_file["arch"]

        # Verify architecture matches
        saved_input_size = arch_group.attrs.get("input_size", network.input_size)
        saved_output_size = arch_group.attrs.get("output_size", network.output_size)

        if saved_input_size != network.input_size:
            self.logger.warning( f"Input size mismatch: {saved_input_size} != {network.input_size}")

        if saved_output_size != network.output_size:
            self.logger.warning( f"Output size mismatch: {saved_output_size} != {network.output_size}")

        # Load activation function name
        af_name = read_str_attr( arch_group, "activation_function_name", network.activation_function_name)
        network.activation_function_name = af_name

        # Reinitialize activation function with the loaded name--ensures activation_fn and activation_functions_dict are properly set
        network._init_activation_function()

        self.logger.debug( f"CascadeHDF5Serializer: Loaded architecture with activation function: {af_name}")

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

            # Verify checksums if they exist
            if "checksums" in output_group:
                try:
                    checksums_json = read_str_dataset(output_group, "checksums")
                    checksums = json.loads(checksums_json)

                    if "output_weights" in checksums and hasattr( network, "output_weights"):
                        if not verify_tensor_checksum( network.output_weights, checksums["output_weights"]):
                            self.logger.error( "CascadeHDF5Serializer: Output weights checksum verification failed!")
                        else:
                            self.logger.debug( "CascadeHDF5Serializer: Output weights checksum verified")

                    if "output_bias" in checksums and hasattr(network, "output_bias"):
                        if not verify_tensor_checksum( network.output_bias, checksums["output_bias"]):
                            self.logger.error( "CascadeHDF5Serializer: Output bias checksum verification failed!")
                        else:
                            self.logger.debug( "CascadeHDF5Serializer: Output bias checksum verified")

                except Exception as e:
                    self.logger.warning( f"CascadeHDF5Serializer: Could not verify checksums: {e}")

            # Load optimizer state if it exists
            if "optimizer" in output_group:
                opt_group = output_group["optimizer"]
                try:
                    self._load_optimizer_state_from_hdf5_helper(opt_group, network)
                except Exception as e:
                    self.logger.warning( f"CascadeHDF5Serializer: Could not restore optimizer: {e}")
                    network.output_optimizer = None

        self.logger.debug("CascadeHDF5Serializer: Loaded parameters")

    def _load_optimizer_state_from_hdf5_helper(self, opt_group, network):
        import torch.optim as optim

        # Get optimizer type
        opt_type = read_str_attr(opt_group, "optimizer_type", "Adam")
        learning_rate = opt_group.attrs.get( "learning_rate", network.learning_rate)

        # Create temporary output layer to get parameters for optimizer
        input_size = network.output_weights.shape[0]
        output_layer = torch.nn.Linear(input_size, network.output_size)
        with torch.no_grad():
            output_layer.weight.copy_(network.output_weights.t())
            output_layer.bias.copy_(network.output_bias)

        # Create optimizer (currently only Adam supported)
        # TODO: Extend to support more optimizers
        if opt_type != "Adam":
            self.logger.warning( f"Unknown optimizer type: {opt_type}, using Adam")
        network.output_optimizer = optim.Adam( output_layer.parameters(), lr=learning_rate)

        # Load optimizer state if state_dict exists
        if "state_dict" in opt_group:
            opt_state_json = read_str_dataset(opt_group, "state_dict")
            opt_state_dict = json.loads(opt_state_json)
            # Note: State restoration is complex and may not fully restore momentum. This provides the structure but training may need a few warmup steps
            self.logger.debug( "CascadeHDF5Serializer: Loaded optimizer state")
            self.logger.debug( f"CascadeHDF5Serializer: Note that optimizer state restoration may be incomplete: {opt_state_dict.keys()}")
        else:
            self.logger.debug( "CascadeHDF5Serializer: Created optimizer without state dict")

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

            # Verify checksums if they exist
            if "checksums" in unit_group:
                try:
                    checksums_json = read_str_dataset(unit_group, "checksums")
                    checksums = json.loads(checksums_json)

                    if "weights" in checksums and "weights" in unit:
                        if not verify_tensor_checksum(unit["weights"], checksums["weights"]):
                            self.logger.error( f"CascadeHDF5Serializer: Hidden unit {i} weights checksum verification failed!")
                        else:
                            self.logger.debug( f"CascadeHDF5Serializer: Hidden unit {i} weights checksum verified")

                    if "bias" in checksums and "bias" in unit:
                        if not verify_tensor_checksum(unit["bias"], checksums["bias"]):
                            self.logger.error( f"CascadeHDF5Serializer: Hidden unit {i} bias checksum verification failed!")
                        else:
                            self.logger.debug( f"CascadeHDF5Serializer: Hidden unit {i} bias checksum verified")

                except Exception as e:
                    self.logger.warning( f"CascadeHDF5Serializer: Could not verify checksums for hidden unit {i}: {e}")

            # Load correlation
            if "correlation" in unit_group.attrs:
                unit["correlation"] = float(unit_group.attrs["correlation"])

            # Load activation function (per unit)
            af_name = read_str_attr( unit_group, "activation_function_name", network.activation_function_name)
            if ( hasattr(network, "activation_functions_dict") and af_name in network.activation_functions_dict):
                unit["activation_fn"] = network.activation_functions_dict[af_name]
            else:
                unit["activation_fn"] = network.activation_fn

            network.hidden_units.append(unit)

        self.logger.debug(f"CascadeHDF5Serializer: Loaded {num_units} hidden units")

    def _load_random_state(self, hdf5_file: h5py.File, network) -> None:
        """Load random state for deterministic reproducibility."""
        if "random" not in hdf5_file:
            return

        random_group = hdf5_file["random"]

        # Load random parameters
        network.random_seed = random_group.attrs.get("seed", network.random_seed)
        network.random_max_value = random_group.attrs.get( "max_value", network.random_max_value)
        network.sequence_max_value = random_group.attrs.get( "sequence_max_value", network.sequence_max_value)
        network.random_value_scale = random_group.attrs.get( "value_scale", network.random_value_scale)

        # Restore RNG states
        try:
            # Python random state
            if "python_state" in random_group:
                python_state_array = load_numpy_array(random_group["python_state"])
                python_state_bytes = python_state_array.tobytes()
                python_state = pickle.loads(python_state_bytes)  # trunk-ignore(bandit/B301)
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
                self.logger.debug( "CascadeHDF5Serializer: Restored PyTorch random state")

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
                    self.logger.debug( f"CascadeHDF5Serializer: Restored CUDA random states for {len(cuda_states)} devices")

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

    def _load_training_history(self, hdf5_file: h5py.File, network) -> None:
        """Load training history."""
        if "history" not in hdf5_file:
            return

        history_group = hdf5_file["history"]

        # Initialize history with network's actual keys
        network.history = {
            "train_loss": [],
            "value_loss": [],  # Use value_* to match network.history
            "train_accuracy": [],
            "value_accuracy": [],  # Use value_* to match network.history
            "hidden_units_added": [],
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
                self.logger.debug( f"CascadeHDF5Serializer: Loaded history key '{save_key}' as '{network_key}'")

        # Load hidden units history
        if "hidden_units_added" in history_group:
            units_group = history_group["hidden_units_added"]
            for unit_name in sorted(units_group.keys()):
                unit_group = units_group[unit_name]
                unit_data = {}

                if "correlation" in unit_group.attrs:
                    unit_data["correlation"] = float(unit_group.attrs["correlation"])
                if "weights" in unit_group:
                    unit_data["weights"] = load_numpy_array(unit_group["weights"])
                if "bias" in unit_group:
                    unit_data["bias"] = load_numpy_array(unit_group["bias"])

                network.history["hidden_units_added"].append(unit_data)

        self.logger.debug("CascadeHDF5Serializer: Loaded training history")

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
        network.candidate_training_tasks_queue_timeout = mp_group.attrs.get( "tasks_queue_timeout", 30.0)
        network.candidate_training_shutdown_timeout = mp_group.attrs.get( "shutdown_timeout", 10.0)

        # Set multiprocessing context
        network.candidate_training_context = mp.get_context(start_method)

        # Restore address and authkey
        network.candidate_training_queue_address = (address_host, address_port)
        if authkey_hex:
            try:
                network.candidate_training_queue_authkey = bytes.fromhex( authkey_hex)
            except ValueError:
                network.candidate_training_queue_authkey = authkey_hex.encode( "utf-8")

        # Recreate multiprocessing components based on role
        if role == "server" and autostart:
            # Reinitialize as server
            network._init_multiprocessing()

        self.logger.debug( f"CascadeHDF5Serializer: Restored multiprocessing state (role: {role})")

    def _create_network_from_file(self, hdf5_file: h5py.File):
        """Create a network instance from HDF5 configuration."""
        try:
            from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
            from cascade_correlation_config.cascade_correlation_config import ( CascadeCorrelationConfig,)
            if "config" not in hdf5_file:
                self.logger.error("No configuration found in file")
                return None
            config_group = hdf5_file["config"]
            if "config_json" in config_group:
                config_json = read_str_dataset(config_group, "config_json")
                config_dict = json.loads(config_json)
                config_dict.pop("activation_functions_dict", None)
                config_dict.pop("log_config", None)
                config_dict.pop("logger", None)
                # Remove runtime-only attributes not in CascadeCorrelationConfig
                config_dict.pop("candidates_per_layer", None)
                config_dict.pop("layer_selection_strategy", None)
                if "meta" in hdf5_file:
                    meta_group = hdf5_file["meta"]
                    if saved_uuid := read_str_attr(meta_group, "uuid", None):
                        config_dict["uuid"] = saved_uuid
                        self.logger.debug( f"CascadeHDF5Serializer: Injecting UUID {saved_uuid} into config")
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
                        self.logger.debug( f"CascadeHDF5Serializer: Setting UUID {saved_uuid} on config")
            network = CascadeCorrelationNetwork(config=config)
            if "meta" in hdf5_file:
                self._restore_training_state_helper(hdf5_file, network)
            self.logger.debug( f"CascadeHDF5Serializer: Created network instance (UUID: {network.get_uuid()})")
            return network
        except Exception as e:
            return self._log_exception_stacktrace( 'Could not create network from file: ', e, None)

    def _restore_training_state_helper(self, hdf5_file, network):
        meta_group = hdf5_file["meta"]
        network.snapshot_counter = meta_group.attrs.get("snapshot_counter", 0)
        if "current_epoch" in meta_group.attrs:
            network.current_epoch = meta_group.attrs.get("current_epoch", 0)
        network.patience_counter = meta_group.attrs.get("patience_counter", 0)
        network.best_value_loss = meta_group.attrs.get("best_value_loss", float("inf"))
        self.logger.debug( f"CascadeHDF5Serializer: Restored training counters - snapshot: {network.snapshot_counter}, patience: {network.patience_counter}")

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
            if network.output_weights.shape != ( expected_output_input, network.output_size,):
                self.logger.error( f"Output weights shape mismatch: {network.output_weights.shape} != ({expected_output_input}, {network.output_size})")
                return False
            if network.output_bias.shape != (network.output_size,):
                self.logger.error( f"Output bias shape mismatch: {network.output_bias.shape} != ({network.output_size},)")
                return False
            for i, unit in enumerate(network.hidden_units):
                expected_input_size = network.input_size + i
                if "weights" in unit and unit["weights"].shape[0] != expected_input_size:
                    self.logger.error( f"Hidden unit {i} weight shape mismatch: {unit['weights'].shape[0]} != {expected_input_size}")
                    return False
                if "bias" in unit and unit["bias"].numel() != 1:
                    self.logger.error( f"Hidden unit {i} bias shape mismatch: {unit['bias'].shape} should be scalar or (1,)")
                    return False
            self.logger.debug("CascadeHDF5Serializer: Shape validation passed")
            return True
        except Exception as e:
            return self._log_exception_stacktrace( 'Shape validation failed: ', e, False)

    def _validate_format(self, hdf5_file: h5py.File) -> bool:
        """
        Validate HDF5 file format with comprehensive checks.

        Validates:
        - Format name and version compatibility
        - Required groups and datasets
        - Hidden units consistency
        - Parameter dataset shapes

        Returns:
            bool: True if file format is valid, False otherwise
        """
        try:
            # Check format identifier
            format_name = read_str_attr(hdf5_file, "format")
            if format_name not in [ self.format_name, "cascor_hdf5_v1", "juniper.cascor", ]:
                self.logger.error(f"Invalid format: {format_name}")
                return False

            # Check format version compatibility
            format_version = read_str_attr(hdf5_file, "format_version", "1")
            try:
                file_major_version = int(format_version.split('.')[0] if '.' in format_version else format_version)
                serializer_major_version = int(self.format_version.split('.')[0])

                if file_major_version > serializer_major_version:
                    self.logger.error( f"Incompatible format version: file={format_version}, " f"serializer={self.format_version}")
                    return False
            except (ValueError, IndexError):
                self.logger.warning(f"Could not parse format version: {format_version}")

            # Check for required groups
            required_groups = ["meta", "config", "params", "arch", "random"]
            for group in required_groups:
                if group not in hdf5_file:
                    self.logger.error(f"Missing required group: {group}")
                    return False

            # Check for required datasets in params group
            if "params" in hdf5_file:
                params_group = hdf5_file["params"]
                if "output_layer" in params_group:
                    output_group = params_group["output_layer"]
                    if "weights" not in output_group:
                        self.logger.error("Missing output layer weights dataset")
                        return False
                    if "bias" not in output_group:
                        self.logger.error("Missing output layer bias dataset")
                        return False
                else:
                    self.logger.error("Missing output_layer group in params")
                    return False

            # Verify hidden units consistency
            if "hidden_units" in hdf5_file:
                hidden_group = hdf5_file["hidden_units"]
                num_units_attr = hidden_group.attrs.get("num_units", 0)
                actual_units = len([k for k in hidden_group.keys() if k.startswith("unit_")])

                if num_units_attr != actual_units:
                    self.logger.error( f"Hidden units count mismatch: num_units={num_units_attr}, " f"actual groups={actual_units}")
                    return False

                # Verify each hidden unit has required datasets
                for i in range(num_units_attr):
                    unit_name = f"unit_{i}"
                    if unit_name in hidden_group:
                        unit_group = hidden_group[unit_name]
                        if "weights" not in unit_group:
                            self.logger.error(f"Hidden unit {i} missing weights dataset")
                            return False
                        if "bias" not in unit_group:
                            self.logger.error(f"Hidden unit {i} missing bias dataset")
                            return False

            self.logger.debug("CascadeHDF5Serializer: Format validation passed")
            return True

        except Exception as e:
            return self._log_exception_stacktrace(
                'Format validation failed: ', e, False
            )

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
                    if all(
                        isinstance(item, (str, int, float, bool, type(None)))
                        for item in attr_value
                    ):
                        config_dict[attr_name] = list(attr_value)
                elif isinstance(attr_value, dict):

                    # Only include if values are primitive types
                    if all(
                        isinstance(v, (str, int, float, bool, type(None)))
                        for v in attr_value.values()
                    ):
                        config_dict[attr_name] = dict(attr_value)
                elif isinstance(attr_value, pl.Path):
                    config_dict[attr_name] = str(attr_value)
                # Skip other complex types

            except Exception as e:
                self.logger.debug(f"Skipping attribute {attr_name}: {e}")
                continue

        return config_dict
