#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     cascade_correlation.py
# Author:        Paul Calnon
#
# Date Created:  2025-06-11
# Last Modified: 2026-01-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This file contains the implementation of the Cascade Correlation Neural Network.
#
#####################################################################################################################################################################################################
# Notes:
#   - The Cascade Correlation Neural Network is designed to incrementally add hidden units to the network.
#   - The network uses a correlation-based approach to determine the relevance of each candidate unit.
#   - The network is trained using a combination of supervised and unsupervised learning techniques.
#   - The network is designed to handle large-scale and high-dimensional datasets efficiently.
#
#####################################################################################################################################################################################################
# References:
#
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#
#####################################################################################################################################################################################################
import atexit
import copy
import datetime
import io

# import logging
import logging.config
import multiprocessing as mp
import os
import pathlib as pl
import pickle
import random
import struct
import sys
import time
import uuid as uuid
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from multiprocessing.shared_memory import SharedMemory
from queue import Queue  # Use stdlib queue for manager-hosted objects
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from candidate_unit.candidate_unit import CandidateTrainingResult, CandidateUnit

#####################################################################################################################################################################################################
# Define custom manager class and server-owned queues for multiprocessing
#
# IMPORTANT: This implementation uses picklable factory functions instead of lambda functions
# to avoid PicklingError when starting worker processes with forkserver context.
# The lambda functions that were previously used cannot be pickled and would cause:
# "Can't pickle <function <lambda> at 0x...>: attribute lookup <lambda> on ... failed"
#
# Server-owned queues (live in Manager server process)
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import CandidateTrainingError, ConfigurationError, TrainingError, ValidationError  # CascadeCorrelationError,; NetworkInitializationError,
from cascor_constants.constants import (  # TODO: Commented out for F401 compliance - may be needed for future activation function selection; _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_RELU,; _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_SIGMOID,; _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_TANH,; _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_RELU,; _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_SIGMOID,; _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_TANH,APPROVED
    _CANDIDATE_UNIT_CONVERGENCE_THRESHOLD,
    _CANDIDATE_UNIT_PATIENCE,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_CONVERGENCE_THRESHOLD,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_DISPLAY_FREQUENCY,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_EPOCHS,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_LEARNING_RATE,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_PATIENCE,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_POOL_SIZE,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_CONTEXT,
    _CASCADE_CORRELATION_NETWORK_CONVERGENCE_THRESHOLD,
    _CASCADE_CORRELATION_NETWORK_DISPLAY_FREQUENCY,
    _CASCADE_CORRELATION_NETWORK_EPOCH_DISPLAY_FREQUENCY,
    _CASCADE_CORRELATION_NETWORK_EPOCHS_MAX,
    _CASCADE_CORRELATION_NETWORK_GENERATE_PLOTS,
    _CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR,
    _CASCADE_CORRELATION_NETWORK_INIT_OUTPUT_WEIGHTS,
    _CASCADE_CORRELATION_NETWORK_INPUT_SIZE,
    _CASCADE_CORRELATION_NETWORK_LEARNING_RATE,
    _CASCADE_CORRELATION_NETWORK_LOG_DATE_FORMAT,
    _CASCADE_CORRELATION_NETWORK_LOG_FILE_NAME,
    _CASCADE_CORRELATION_NETWORK_LOG_FILE_PATH,
    _CASCADE_CORRELATION_NETWORK_LOG_FORMATTER_STRING,
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_CUSTOM_NAMES_LIST,
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_METHODS_DICT,
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_METHODS_LIST,
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME,
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAMES_LIST,
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_DICT,
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_LIST,
    _CASCADE_CORRELATION_NETWORK_MAX_HIDDEN_UNITS,
    _CASCADE_CORRELATION_NETWORK_MAX_ITERATIONS,
    _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD,
    _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS,
    _CASCADE_CORRELATION_NETWORK_OUTPUT_SIZE,
    _CASCADE_CORRELATION_NETWORK_PATIENCE,
    _CASCADE_CORRELATION_NETWORK_RANDOM_MAX_VALUE,
    _CASCADE_CORRELATION_NETWORK_RANDOM_SEED,
    _CASCADE_CORRELATION_NETWORK_RANDOM_VALUE_SCALE,
    _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MAX_TIMEOUT,
    _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MIN_TIMEOUT,
    _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_SECONDS_PER_EPOCH,
    _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE,
    _CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT,
    _CASCADE_CORRELATION_NETWORK_STATUS_DISPLAY_FREQUENCY,
    _CASCADE_CORRELATION_NETWORK_TARGET_ACCURACY,
    _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT,
    _CASCADE_CORRELATION_NETWORK_WORKER_STANDBY_SLEEPYTIME,
)
from cascor_plotter.cascor_plotter import CascadeCorrelationPlotter
from log_config.log_config import LogConfig
from log_config.logger.logger import Logger

# P-1 / RC-4 ring buffer (gated by CASCOR_RC4_RING_BUFFER=1; no-op
# otherwise). See src/parallelism/rc4_ring_buffer.py and
# notes/P1_RC4_INVESTIGATION_PLAN_2026-05-03.md (juniper-ml).
from parallelism import rc4_ring_buffer as _rc4
from parallelism.task_distributor import TaskDistributor
from utils.utils import display_progress

# import traceback


#####################################################################################################################################################################################################
# Data classes for structured results
@dataclass
class TrainingResults:
    """Aggregated results from candidate training."""

    epochs_completed: int
    candidate_ids: List[int]
    candidate_uuids: List[str]
    correlations: List[float]
    candidate_objects: List[Any]
    best_candidate_id: int
    best_candidate_uuid: str
    best_correlation: float
    best_candidate: Optional[Any]
    success_count: int
    successful_candidates: int
    failed_count: int
    error_messages: List[str]
    max_correlation: float
    start_time: datetime.datetime
    end_time: datetime.datetime


@dataclass
class ValidateTrainingInputs:
    """Inputs required for validating training results."""

    iteration: int
    max_iterations: int
    patience_counter: int
    early_stopping: bool
    train_accuracy: float
    train_loss: float
    best_value_loss: float
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray


@dataclass
class ValidateTrainingResults:
    """Results from validating training outputs."""

    early_stop: bool
    patience_counter: int
    best_value_loss: float
    value_output: float
    value_loss: float
    value_accuracy: float


# Maximum queue size to prevent unbounded memory growth (DoS vector)
_QUEUE_MAXSIZE = 1024

# DUAL-PATH/TIMEOUT FIX (2026-05-30): default *inactivity* timeout for candidate
# result collection. ``_collect_training_results`` historically used a fixed 60s
# TOTAL deadline, which was shorter than multi-minute candidate rounds (a
# 40-candidate pool can take >2 min), so the collector returned ~0 results while
# the persistent workers were still training — yielding best_candidate=None and
# 0 hidden units grown. The value below is the maximum gap BETWEEN consecutive
# results (the deadline is reset on each result), backstopped by a worker-liveness
# early-exit, so a slow-but-progressing round is never abandoned while a genuinely
# stalled or dead pool still bails out promptly.
_CASCADE_CORRELATION_NETWORK_RESULT_COLLECTION_INACTIVITY_TIMEOUT = 300.0


# Server-owned queues (live in Manager server process)--created in the manager server process
_task_queue = None
_result_queue = None


def _create_task_queue():
    """
    Factory function to create or return the task queue in the manager server process.
    This function is picklable and will be executed in the server process.
    """
    global _task_queue
    if _task_queue is None:
        _task_queue = Queue(maxsize=_QUEUE_MAXSIZE)
    return _task_queue


def _create_result_queue():
    """
    Factory function to create or return the result queue in the manager server process.
    This function is picklable and will be executed in the server process.
    """
    global _result_queue
    if _result_queue is None:
        _result_queue = Queue(maxsize=_QUEUE_MAXSIZE)
    return _result_queue


#####################################################################################################################################################################################################
# OPT-5: Shared Memory Training Tensors — zero-copy tensor sharing across worker processes
#####################################################################################################################################################################################################
class SharedTrainingMemory:
    """Manages a POSIX shared memory block for training tensor sharing (OPT-5).

    Creates a named /dev/shm block containing training tensors that worker
    processes can attach to by name for zero-copy reads. Block layout:
      - 64-byte header (magic b"JNPR", version, n_tensors)
      - 32-byte descriptor per tensor (offset, nbytes, ndim, dtype, shape)
      - Contiguous tensor data
    """

    MAGIC = b"JNPR"
    VERSION = 1
    HEADER_SIZE = 64
    DESCRIPTOR_SIZE = 32
    DTYPE_MAP = {torch.float32: 0, torch.float64: 1, torch.int32: 2, torch.int64: 3}
    DTYPE_RMAP = {v: k for k, v in DTYPE_MAP.items()}
    NUMPY_DTYPE_MAP = {0: np.float32, 1: np.float64, 2: np.int32, 3: np.int64}

    def __init__(self, tensors: list, name_suffix: str):
        """Create SharedMemory block and copy tensors into it.

        Args:
            tensors: List of torch.Tensor to share (must be float32/64, int32/64).
            name_suffix: Unique suffix for the block name.
        """
        contiguous_tensors = []
        self._tensors_info = []
        for t in tensors:
            ct = t.contiguous() if not t.is_contiguous() else t
            contiguous_tensors.append(ct)
            dtype_code = self.DTYPE_MAP.get(ct.dtype)
            if dtype_code is None:
                raise ValueError(f"Unsupported tensor dtype: {ct.dtype}")
            if ct.ndim > 2:
                raise ValueError(f"SharedTrainingMemory only supports tensors up to 2 dimensions, " f"got tensor with {ct.ndim} dimensions (shape: {tuple(ct.shape)}). " f"Higher-dimensional tensor support is planned for a future release.")
            self._tensors_info.append(
                {
                    "nbytes": ct.nbytes,
                    "ndim": ct.ndim,
                    "dtype_code": dtype_code,
                    "shape": tuple(ct.shape),
                }
            )

        n_tensors = len(contiguous_tensors)
        descriptor_table_size = self.DESCRIPTOR_SIZE * n_tensors
        data_offset = self.HEADER_SIZE + descriptor_table_size
        total_data_bytes = sum(info["nbytes"] for info in self._tensors_info)
        total_size = data_offset + total_data_bytes

        self._name = f"juniper_train_{name_suffix}"
        self._shm = SharedMemory(name=self._name, create=True, size=total_size)
        self._closed = False
        self._unlinked = False

        buf = self._shm.buf
        # Write header: magic(4s) + version(B) + n_tensors(B) + reserved(58x) = 64 bytes
        struct.pack_into("<4sBB58x", buf, 0, self.MAGIC, self.VERSION, n_tensors)

        # Write descriptor table and copy tensor data
        current_offset = data_offset
        for i, (info, ct) in enumerate(zip(self._tensors_info, contiguous_tensors)):
            shape_0 = info["shape"][0] if info["ndim"] >= 1 else 0
            shape_1 = info["shape"][1] if info["ndim"] >= 2 else 0
            # Descriptor: offset(Q) + nbytes(Q) + ndim(B) + dtype_code(B) + shape0(I) + shape1(I) + reserved(6x) = 32 bytes
            # NOTE: Struct format only stores shape_0 and shape_1, limiting support to
            # tensors with ndim <= 2. Validated in __init__. Higher-dimensional support
            # would require a format change (e.g., variable-length shape encoding).
            struct.pack_into(
                "<QQBBII6x",
                buf,
                self.HEADER_SIZE + i * self.DESCRIPTOR_SIZE,
                current_offset,
                info["nbytes"],
                info["ndim"],
                info["dtype_code"],
                shape_0,
                shape_1,
            )
            tensor_bytes = ct.numpy().tobytes()
            buf[current_offset : current_offset + info["nbytes"]] = tensor_bytes
            current_offset += info["nbytes"]

    @property
    def name(self) -> str:
        return self._name

    def get_metadata(self) -> dict:
        """Return metadata dict for inclusion in lightweight tasks."""
        return {"shm_name": self._name}

    @staticmethod
    def reconstruct_tensors(metadata: dict) -> tuple:
        """Attach to SharedMemory by name and return zero-copy tensor views.

        Returns:
            (list_of_tensors, shm_handle). Caller MUST keep shm_handle alive
            until all tensor operations complete, then call shm_handle.close().
        """
        # ``track=False`` is Python 3.13+ on multiprocessing.shared_memory; on
        # 3.12 it raises ``TypeError``. Pass it conditionally so the test
        # matrix (which still includes 3.12) doesn't blow up — when the kwarg
        # isn't available, the resource_tracker registers the segment and
        # cleans it up on interpreter shutdown, which is the safer default
        # and matches the historical 3.12 behavior.
        if sys.version_info >= (3, 13):
            shm = SharedMemory(name=metadata["shm_name"], create=False, track=False)
        else:
            shm = SharedMemory(name=metadata["shm_name"], create=False)
        try:
            buf = shm.buf
            magic = bytes(buf[:4])
            if magic != SharedTrainingMemory.MAGIC:
                raise ValueError(f"Invalid SharedMemory block header: expected {SharedTrainingMemory.MAGIC!r}, got {magic!r}")

            _version, n_tensors = struct.unpack_from("<BB", buf, 4)

            tensors = []
            for i in range(n_tensors):
                desc_offset = SharedTrainingMemory.HEADER_SIZE + i * SharedTrainingMemory.DESCRIPTOR_SIZE
                offset, nbytes, ndim, dtype_code, shape_0, shape_1 = struct.unpack_from(
                    "<QQBBII6x",
                    buf,
                    desc_offset,
                )
                np_dtype = SharedTrainingMemory.NUMPY_DTYPE_MAP[dtype_code]
                if ndim == 1:
                    shape = (shape_0,)
                elif ndim == 2:
                    shape = (shape_0, shape_1)
                else:
                    shape = (shape_0,) if shape_0 > 0 else ()
                np_array = np.ndarray(shape=shape, dtype=np_dtype, buffer=buf[offset : offset + nbytes])
                tensors.append(torch.from_numpy(np_array))

            return tensors, shm
        except Exception:
            shm.close()
            raise

    def close(self):
        """Close the main process's handle without unlinking.

        Workers may still be reading from this block. Calling close() releases
        the file descriptor in the current process but keeps the /dev/shm entry
        alive for other processes. Call unlink() separately when all workers
        have finished reading.
        """
        if not self._closed:
            try:
                self._shm.close()
            except Exception:  # nosec B110 — cleanup must not propagate exceptions
                pass
            self._closed = True

    def unlink(self):
        """Unlink the SharedMemory block from /dev/shm.

        Only call this after all worker processes have finished reading from the block.
        """
        if not self._unlinked:
            try:
                self._shm.unlink()
            except Exception:  # nosec B110 — cleanup must not propagate exceptions
                pass
            self._unlinked = True

    def close_and_unlink(self):
        """Release and unlink the SharedMemory block."""
        self.close()
        self.unlink()


#####################################################################################################################################################################################################
# Security: Restricted unpickler for deserialization defense-in-depth
#####################################################################################################################################################################################################
class RestrictedUnpickler(pickle.Unpickler):
    """Restricts unpickling to known-safe types for CandidateTrainingResult deserialization.

    This class provides defense-in-depth against pickle-based deserialization attacks.
    It is used for any manual deserialization paths (e.g., network I/O). Note that
    multiprocessing.Queue uses its own internal unpickler that cannot be overridden;
    for that path, post-deserialization validation via _validate_training_result() is
    the primary defense.
    """

    ALLOWED_CLASSES = {
        # Application types
        ("candidate_unit.candidate_unit", "CandidateTrainingResult"),
        ("candidate_unit.candidate_unit", "CandidateUnit"),
        ("utils.activation", "ActivationWithDerivative"),
        # Python builtins
        ("builtins", "list"),
        ("builtins", "dict"),
        ("builtins", "set"),
        ("builtins", "tuple"),
        ("builtins", "float"),
        ("builtins", "int"),
        ("builtins", "bool"),
        ("builtins", "str"),
        ("builtins", "bytes"),
        # PyTorch tensor reconstruction
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch", "Tensor"),
        ("torch", "Size"),
        ("torch.storage", "TypedStorage"),
        ("torch.storage", "UntypedStorage"),
        ("torch.storage", "_load_from_bytes"),
        ("torch", "float32"),
        ("torch", "float64"),
        ("torch", "int64"),
        # PyTorch activation modules (used by CandidateUnit.activation_fn_base)
        # All activation types from _PROJECT_MODEL_ACTIVATION_FUNCTIONS_DICT must be listed
        # to allow deserialization of CandidateUnit objects using any supported activation.
        ("torch.nn.modules.linear", "Identity"),
        ("torch.nn.modules.activation", "Tanh"),
        ("torch.nn.modules.activation", "Sigmoid"),
        ("torch.nn.modules.activation", "ReLU"),
        ("torch.nn.modules.activation", "LeakyReLU"),
        ("torch.nn.modules.activation", "ELU"),
        ("torch.nn.modules.activation", "SELU"),
        ("torch.nn.modules.activation", "GELU"),
        ("torch.nn.modules.activation", "Softmax"),
        ("torch.nn.modules.activation", "Softplus"),
        ("torch.nn.modules.activation", "Hardtanh"),
        ("torch.nn.modules.activation", "Softshrink"),
        ("torch.nn.modules.activation", "Tanhshrink"),
        # Collections and codecs
        ("collections", "OrderedDict"),
        ("_codecs", "encode"),
        # Numpy
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),  # Python 3.14+ internal rename
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Blocked unpickling of {module}.{name}")

    @classmethod
    def loads(cls, data: bytes):
        """Deserialize bytes using the restricted unpickler."""
        return cls(io.BytesIO(data)).load()


# Define CandidateTrainingManager Class and global functions
class CandidateTrainingManager(BaseManager):
    """Custom manager for handling candidate training queues."""

    def start(self, method: str = None, initializer=None, initargs=()):
        """
          Start the manager server, optionally validating a requested start method.

          Args:
              method: Optional multiprocessing start method ('fork', 'spawn', 'forkserver').
                      If provided, validates the method is supported on this platform.
              initializer: Optional initializer function for worker processes.
              initargs: Arguments for the initializer function.
        for tr in remote_results:
              ValueError: If an invalid start method is provided.
              NotImplementedError: If the start method is not supported on this platform.

          Returns:
              Result from BaseManager.start()
        """
        if method is not None:
            valid_methods = {"fork", "spawn", "forkserver"}
            if method not in valid_methods:
                raise ValueError(f"Invalid start method: {method}")

            # Verify that the context exists on this platform
            try:
                mp.get_context(method)
            except Exception as exc:
                # raise NotImplementedError(f"Start method '{method}' not implemented on this platform") from exc
                raise NotImplementedError(f"Start method {method!r} not implemented on this platform") from exc

        # Delegate to BaseManager.start() with supported arguments
        return super().start(initializer=initializer, initargs=initargs)


# Register picklable factory functions instead of lambda functions
CandidateTrainingManager.register("get_task_queue", callable=_create_task_queue)
CandidateTrainingManager.register("get_result_queue", callable=_create_result_queue)


#####################################################################################################################################################################################################
# Module-level worker functions for plotting (must be picklable for multiprocessing)
#####################################################################################################################################################################################################
def _plot_decision_boundary_worker(network, x_data, y_data, title_str):
    """
    Worker function to create decision boundary plot in separate process.
    This function must be at module level to be picklable for multiprocessing.
    Args:
        network: CascadeCorrelationNetwork instance
        x_data: Input tensor
        y_data: Target tensor
        title_str: Plot title
    """
    from cascor_plotter.cascor_plotter import CascadeCorrelationPlotter

    plotter = CascadeCorrelationPlotter()
    plotter.plot_decision_boundary(network, x_data, y_data, title_str)


def _plot_training_history_worker(history_data):
    """
    Worker function to create training history plot in separate process.
    This function must be at module level to be picklable for multiprocessing.
    Args:
        history_data: Training history dictionary
    """
    from cascor_plotter.cascor_plotter import CascadeCorrelationPlotter

    plotter = CascadeCorrelationPlotter()
    plotter.plot_training_history(history_data)


#####################################################################################################################################################################################################
# Picklable Activation Function Wrapper
# CASCOR-P1-003: Multiprocessing Pickling Error Fix
# Extracted to utils.activation to eliminate duplication with candidate_unit.py
#####################################################################################################################################################################################################
from utils.activation import ActivationWithDerivative  # noqa: F401, E402


#####################################################################################################################################################################################################
# Class definition for the Cascade Correlation Network
#####################################################################################################################################################################################################
class CascadeCorrelationNetwork:
    """
    Cascade Correlation Neural Network implementation.

    This class implements the Cascade Correlation algorithm (Fahlman & Lebiere, 1990)
    for constructive learning with automatic network growth.

    Warning:
        **NOT THREAD-SAFE**: Do not share CascadeCorrelationNetwork instances between
        threads without proper synchronization. For concurrent training scenarios,
        create separate network instances per thread. The internal multiprocessing
        for candidate training is handled within the class and does not require
        external synchronization.
    """

    #################################################################################################################################################################################################
    # Constructor for the Cascade Correlation Network
    def __init__(
        self,
        config: CascadeCorrelationConfig = None,
        **kwargs,
    ):
        Logger.debug("CascadeCorrelationNetwork: __init__: Initializing Cascade Correlation Network")
        super().__init__()

        # Initialize configuration (forward kwargs so direct parameter passing works)
        self._init_config(config, **kwargs)

        # Initialize logging system
        self._init_logging_system()

        # Initialize network parameters
        self._init_network_parameters()

        # Initialize multiprocessing components
        self._init_multiprocessing()

        # Initialize display and plotting components
        self._init_display_components()
        Logger.info("CascadeCorrelationNetwork: __init__: Initialization completed")

    #################################################################################################################################################################################################
    # Define init methods called by the __init__ constructor method.
    #################################################################################################################################################################################################

    def _init_config(self, config: CascadeCorrelationConfig = None, **kwargs) -> None:
        """Initialize configuration and set UUID."""
        if hasattr(self, "logger") and self.logger is not None:
            logger = self.logger
        else:
            logger = Logger
        logger.debug("CascadeCorrelationNetwork: _init_config: Initializing configuration")
        if config is None:
            config = CascadeCorrelationConfig(**kwargs)
        self.config = config
        logger.debug(f"CascadeCorrelationNetwork: _init_config: Configuration set to: {self.config}")
        self.set_uuid(self.config.uuid)
        logger.debug(f"CascadeCorrelationNetwork: _init_config: UUID set to: {self.uuid}")

    def _init_logging_system(self) -> None:
        """Initialize the logging system with proper configuration."""
        Logger.debug("CascadeCorrelationNetwork: _init_logging_system: Initializing logging system")

        # Set up log parameters
        self.log_file_name = self.config.log_file_name or _CASCADE_CORRELATION_NETWORK_LOG_FILE_NAME or __name__
        self.log_file_path = self.config.log_file_path or _CASCADE_CORRELATION_NETWORK_LOG_FILE_PATH or str(os.path.join(os.getcwd(), "logs"))
        self.log_level_name = self.config.log_level_name or _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME

        # Create LogConfig object
        self.log_config = self.config.log_config or LogConfig(
            _LogConfig__log_config=logging.config,
            _LogConfig__log_file_name=self.log_file_name or _CASCADE_CORRELATION_NETWORK_LOG_FILE_NAME,
            _LogConfig__log_file_path=self.log_file_path or _CASCADE_CORRELATION_NETWORK_LOG_FILE_PATH,
            _LogConfig__log_level_name=self.log_level_name or _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME,
            _LogConfig__log_date_format=self.config.log_date_format or _CASCADE_CORRELATION_NETWORK_LOG_DATE_FORMAT,
            _LogConfig__log_format_string=self.config.log_format_string or _CASCADE_CORRELATION_NETWORK_LOG_FORMATTER_STRING,
            _LogConfig__log_level_custom_names_list=self.config.log_level_custom_names_list or _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_CUSTOM_NAMES_LIST,
            _LogConfig__log_level_methods_dict=self.config.log_level_methods_dict or _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_METHODS_DICT,
            _LogConfig__log_level_methods_list=self.config.log_level_methods_list or _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_METHODS_LIST,
            _LogConfig__log_level_names_list=self.config.log_level_names_list or _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAMES_LIST,
            _LogConfig__log_level_numbers_dict=self.config.log_level_numbers_dict or _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_DICT,
            _LogConfig__log_level_numbers_list=self.config.log_level_numbers_list or _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_LIST,
        )

        # Set up logger
        self.logger = self.log_config.get_logger()
        self.logger.level = self.log_config.get_log_level()
        self.logger.debug(f"CascadeCorrelationNetwork: _init_logging_system: Logger initialized with level: {self.logger.level}")

    def _init_network_parameters(self) -> None:
        """Initialize network parameters, randomness, and model components."""
        Logger.debug("CascadeCorrelationNetwork: _init_network_parameters: Initializing network parameters")

        # Initialize randomness
        self.random_seed = self.config.random_seed or _CASCADE_CORRELATION_NETWORK_RANDOM_SEED
        self.random_max_value = self.config.random_max_value or _CASCADE_CORRELATION_NETWORK_RANDOM_MAX_VALUE
        self.sequence_max_value = self.config.sequence_max_value or _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE
        self._initialize_randomness(
            seed=self.random_seed,
            sequence_max_value=self.sequence_max_value,
            random_max_value=self.random_max_value,
        )

        # Initialize network architecture parameters
        self.input_size = self.config.input_size or _CASCADE_CORRELATION_NETWORK_INPUT_SIZE
        self.output_size = self.config.output_size or _CASCADE_CORRELATION_NETWORK_OUTPUT_SIZE
        # P2-1d: loss-mask depth. Equals ``output_size`` by default — no
        # masking, full-tensor training. Lower than ``output_size`` after a
        # dataset shrink (the lifecycle's ``_pad_dataset_for_network`` sets
        # it to the dataset's real output dim before submitting training).
        # ``train_output_layer`` slices ``output[:, :active_output_dim]`` and
        # ``y[:, :active_output_dim]`` before computing loss;
        # ``calculate_residual_error`` zero-fills residual columns
        # ``[active_output_dim:]`` so candidate correlation never picks up
        # signal from zero-padded target columns.
        self.active_output_dim = self.output_size
        self.candidate_pool_size = self.config.candidate_pool_size or _CASCADE_CORRELATION_NETWORK_CANDIDATE_POOL_SIZE
        # FRONTEND_ISSUES_PLAN_2026-05-09 §1.5 C2 / Issue #1 — candidate-pool selection
        # knobs. Defaults preserve current behavior (single top-correlation candidate
        # per growth iteration). PR-4b will rewire ``_select_best_candidates`` to
        # honour the multi/top/random triple; today these attributes are storage-only.
        self.multi_candidate = bool(getattr(self.config, "multi_candidate", False) or False)
        self.candidate_selection = getattr(self.config, "candidate_selection", None) or "top"
        self.selected_candidates = getattr(self.config, "selected_candidates", None) or 1
        self.top_candidates = getattr(self.config, "top_candidates", None) if getattr(self.config, "top_candidates", None) is not None else 1
        self.random_candidates = getattr(self.config, "random_candidates", None) if getattr(self.config, "random_candidates", None) is not None else 0

        # Initialize training parameters
        self.learning_rate = self.config.learning_rate or _CASCADE_CORRELATION_NETWORK_LEARNING_RATE
        self.candidate_learning_rate = self.config.candidate_learning_rate or _CASCADE_CORRELATION_NETWORK_CANDIDATE_LEARNING_RATE
        self.max_hidden_units = self.config.max_hidden_units or _CASCADE_CORRELATION_NETWORK_MAX_HIDDEN_UNITS
        self.correlation_threshold = self.config.correlation_threshold or _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD
        self.patience = self.config.patience or _CASCADE_CORRELATION_NETWORK_PATIENCE
        self.convergence_threshold = self.config.convergence_threshold or _CASCADE_CORRELATION_NETWORK_CONVERGENCE_THRESHOLD

        # self.candidate_convergence_threshold = self.config.candidate_convergence_threshold or _CASCADE_CORRELATION_NETWORK_CANDIDATE_CONVERGENCE_THRESHOLD
        # self.candidate_patience = self.config.candidate_patience or _CASCADE_CORRELATION_NETWORK_CANDIDATE_PATIENCE
        self.candidate_patience = self.config.candidate_patience if hasattr(self.config, "candidate_patience") else _CASCADE_CORRELATION_NETWORK_CANDIDATE_PATIENCE
        self.candidate_convergence_threshold = self.config.candidate_convergence_threshold if hasattr(self.config, "candidate_convergence_threshold") else _CASCADE_CORRELATION_NETWORK_CANDIDATE_CONVERGENCE_THRESHOLD

        self.candidate_epochs = self.config.candidate_epochs or _CASCADE_CORRELATION_NETWORK_CANDIDATE_EPOCHS
        self.epochs_max = self.config.epochs_max or _CASCADE_CORRELATION_NETWORK_EPOCHS_MAX
        self.max_iterations = getattr(self.config, "max_iterations", None) or _CASCADE_CORRELATION_NETWORK_MAX_ITERATIONS
        self.output_epochs = self.config.output_epochs or _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS
        self.random_value_scale = self.config.random_value_scale or _CASCADE_CORRELATION_NETWORK_RANDOM_VALUE_SCALE
        self.init_output_weights = getattr(self.config, "init_output_weights", None) or _CASCADE_CORRELATION_NETWORK_INIT_OUTPUT_WEIGHTS
        self.target_accuracy = self.config.candidate_training_target_accuracy or _CASCADE_CORRELATION_NETWORK_TARGET_ACCURACY
        self.worker_standby_sleepytime = self.config.candidate_training_worker_standby_sleepytime or _CASCADE_CORRELATION_NETWORK_WORKER_STANDBY_SLEEPYTIME
        self.shutdown_timeout = self.config.candidate_training_shutdown_timeout or _CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT
        self.task_queue_timeout = self.config.candidate_training_task_queue_timeout or _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT

        # Initialize snapshot counter for HDF5 serialization
        self.snapshot_counter = 0

        # Initialize activation function
        self._init_activation_function()

        # Initialize network model parameters)
        self.hidden_units = []
        # Diagnosability (Issue #3 follow-up): which grow_network exit fired on
        # the most recent growth run. Surfaced via manager.get_status() as
        # "completion_reason" so canopy can distinguish a genuine convergence
        # from a stall instead of both showing a bare "Completed". None until
        # training runs.
        self._completion_reason: Optional[str] = None
        self._cached_candidate_input = None  # OPT-4: forward pass cache for candidate input reuse

        self.output_weights = torch.randn(self.config.input_size, self.config.output_size, requires_grad=True) * self.random_value_scale
        self.output_bias = torch.randn(self.config.output_size, requires_grad=True) * self.random_value_scale

        self.history = {
            "train_loss": [],
            "value_loss": [],
            "train_accuracy": [],
            "value_accuracy": [],
            "hidden_units_added": [],
            # P2-2 (Issue #3): structured record of every successful
            # ``swap_dataset_live`` event. One dict appended per swap; see
            # ``record_dataset_swap_event`` for the payload schema. Sibling
            # of ``hidden_units_added`` in the event-with-payload pattern.
            "dataset_swaps": [],
        }

        # Snapshot directory
        self.cascade_correlation_network_snapshots_dir = self.config.cascade_correlation_network_snapshots_dir or _CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR
        self.logger.debug("CascadeCorrelationNetwork: _init_network_parameters: Network parameters initialized")

    # #################################################################################################################################################################################################
    # # Network Resizing Methods
    # #################################################################################################################################################################################################
    # def _resize_network_for_dataset(self, input_size_new: int = 0, output_size_new: int = 0) -> None:
    #     """Resize the network for a new dataset."""
    #     try:
    #         self._resize_network_output_for_dataset(input_size_new=input_size_new, output_size_new=output_size_new)
    #         self._resize_network_hidden_for_dataset(input_size_new=input_size_new, output_size_new=output_size_new)
    #     except Exception as e:
    #         self.logger.error(f"CascadeCorrelationNetwork: _resize_network_for_dataset: Error resizing network for outputs: {e}")
    #         raise e

    # def _resize_network_output_for_dataset(self, input_size_new: int = 0, output_size_new: int = 0) -> None:
    #     """Resize the network output layer for a new dataset."""
    #     try:
    #         self.output_weights, self.output_bias = self._grow_network_for_dataset(weights_tensor=self.output_weights, bias_tensor=self.output_bias, input_size_current=self.config.input_size, input_size_new=input_size_new, output_size_current=self.config.output_size, output_size_new=output_size_new, dims=(1, 0))
    #     except Exception as e:
    #         self.logger.error(f"CascadeCorrelationNetwork: _resize_network_output_for_dataset: Error growing network for outputs: {e}")
    #         raise e

    # def _resize_network_hidden_for_dataset(self, input_size_new: int = 0, output_size_new: int = 0) -> None:
    #     """Resize the network hidden layer for a new dataset."""
    #     try:
    #         for hidden_node_index, hidden_node in enumerate(self.hidden_units):
    #             hidden_node["weights"], hidden_node["bias"] = self._grow_network_for_dataset(tensor=hidden_node["weights"], bias_tensor=hidden_node["bias"], dim0_size_current=self.config.input_size, dim0_size_new=input_size_new, dim1_size_current=self.config.output_size, dim1_size_new=output_size_new, dims=(1, 0))
    #             self.hidden_units[hidden_node_index] = hidden_node
    #     except Exception as e:
    #         self.logger.error(f"CascadeCorrelationNetwork: _resize_network_hidden_for_dataset: Error resizing network hidden nodes for dataset change: {e}")
    #         raise e

    # def _grow_network_for_dataset(self, weights_tensor: torch.Tensor, bias_tensor: torch.Tensor, input_size_current: int = 0, input_size_new: int = 0, output_size_current: int = 0, output_size_new: int = 0, dims: tuple[int, int] = (0, 0)) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Grow the network output weights layer for a new dataset with new output size.""    "
    #     try:
    #         # Grow the network output layer weights tensor
    #         weights_tensor = self._grow_network_layer(tensor=weights_tensor, dim0_size_current=input_size_current, dim0_size_new=input_size_new, dim1_size_current=output_size_current, dim1_size_new=output_size_new, dims=(1, 0))
    #         # Grow the network output layer bias tensor
    #         bias_tensor = self._grow_network_layer(tensor=bias_tensor, dim0_size_current=input_size_new, dim0_size_new=input_size_new, dim1_size_current=output_size_current, dim1_size_new=output_size_new, dims=(1, 0))
    #     except Exception as e:
    #         self.logger.error(f"CascadeCorrelationNetwork: _grow_network_for_outputs: Error growing network for outputs: {e}")
    #         raise e
    #     return weights_tensor, bias_tensor

    # def _grow_network_layer(self, tensor: torch.Tensor, dim0_size_current: int = 0, dim0_size_new: int = 0, dim1_size_current: int = 0, dim1_size_new: int = 0, dims: tuple[int, int] = (0, 0)) -> torch.Tensor:
    #     """Grow the network output weights layer for a new dataset with new input size and/or new output size."""
    #     # New Dim 0 size should increase or be the same as Dim 0 size current
    #     if dim0_size_new < dim0_size_current:
    #         raise ValueError(f"CascadeCorrelationNetwork: _grow_network_output_layer: Dim 0 size new {dim0_size_new} is less than Dim 0 size current {dim0_size_current}")
    #     # New Dim 1 size should increase or be the same as Dim 1 size current
    #     if dim1_size_new < dim1_size_current:
    #         raise ValueError(f"CascadeCorrelationNetwork: _grow_network_output_layer: Dim 1 size new {dim1_size_new} is less than Dim 1 size current {dim1_size_current}")
    #     # Determine Size Increase for the dimension that is growing
    #     padding_size = (dim0_size_current - dim0_size_new, dim1_size_current - dim1_size_new)
    #     # Grow the output weights tensor
    #     try:
    #         tensor = self._grow_network_tensor(tensor=tensor, padding_size=padding_size, dims=dims, gradient_tracking=True, random_value_scale=self.random_value_scale)
    #     except Exception as e:
    #         self.logger.error(f"CascadeCorrelationNetwork: _grow_network_output_layer: Error growing network output layer: {e}")
    #         raise e
    #     return tensor

    # def _grow_network_tensor(self, tensor: torch.Tensor, padding_size: tuple[int, int] = (0, 0), dims: tuple[int, int] = (0, 0), gradient_tracking: bool = True, random_value_scale: float = 1.0) -> torch.Tensor:
    #     """Grow the network tensor for a new dataset with new input size and/or new output size."""
    #     if padding_size[0] > 0:
    #         dim = dims[0]
    #         tensor = torch.cat([ tensor, (torch.randn(padding_size[0], tensor.size(1)) * random_value_scale)], dim=dim)
    #     if padding_size[1] > 0:
    #         dim = dims[1]
    #         tensor = torch.cat([ tensor, (torch.randn(tensor.size(0), padding_size[1]) * random_value_scale)], dim=dim)
    #     tensor.requires_grad_(gradient_tracking)
    #     return tensor
    # #################################################################################################################################################################################################

    # ------------------------------------------------------------------
    # P2-1d: live dataset-driven dim growth. See
    # ``notes/PHASE_2_P2_1D_DESIGN_2026-05-13.md`` for the full contract.
    #
    # The network is monotonically non-decreasing — input/output dims only
    # ever grow. Shrink is handled at the dataset layer (zero-padding the
    # dataset tensors up to the network's current dims) and is implemented
    # by ``TrainingLifecycleManager._pad_dataset_for_network``. Calling these
    # methods with smaller dims raises ``ValueError`` rather than silently
    # truncating, so an accidental shrink request surfaces loudly.
    #
    # New parameter values are random-init scaled by ``self.random_value_scale``
    # to match the construction-time pattern at ``__init__``. Hidden-unit
    # weights remain detached (cascade-correlation freezes them after
    # promotion); ``output_weights`` and ``output_bias`` are re-enabled
    # for grad tracking after every grow so the next ``train_output_layer``
    # call can build an optimizer on the expanded tensors.
    # ------------------------------------------------------------------

    def _sync_config_dimensions(self) -> None:
        """Copy the live structural dimensions back onto the config object.

        ``input_size`` / ``output_size`` live in two places -- on the network (authoritative
        at runtime, and what the tensors are built to) and on ``self.config`` (what
        ``_save_configuration`` serializes into ``config_json``). They are read from the
        config once at construction and then diverge the moment the network is resized.

        Keeping them in step is what stops a snapshot describing two different networks at
        once. Written defensively: a config without these attributes is left alone rather
        than gaining them, so an unusual or stubbed config cannot turn a resize into a
        crash on the training path.
        """
        config = getattr(self, "config", None)
        if config is None:
            return
        for field, value in (("input_size", self.input_size), ("output_size", self.output_size)):
            if hasattr(config, field):
                setattr(config, field, value)

    def _resize_network_for_dataset(self, input_size_new: int, output_size_new: int) -> Dict[str, int]:
        """Grow ``self.input_size`` / ``self.output_size`` to fit a new dataset.

        Returns ``{"input_delta", "output_delta", "hidden_preserved"}`` so the
        lifecycle layer can surface the values in the §3.3 swap response.
        Equal-dim is a no-op (returns zero deltas without touching tensors).

        Raises ``ValueError`` if either new dim is smaller than the current —
        the network never shrinks (see ``_pad_dataset_for_network``).

        Tensor mutations: ``output_weights`` row-insertion at index
        ``self.input_size`` (separates raw-input rows from hidden-unit rows
        so the layout consumed by ``forward()`` stays aligned), column-append
        for output dim; ``output_bias`` element-append for output dim; each
        ``hidden_units[i]["weights"]`` mirrors the row-insertion at index
        ``self.input_size``. Hidden-unit biases (scalars) are not touched.

        Bookkeeping (``self.input_size``, ``self.output_size``) is updated
        AFTER all tensor mutations succeed so a partial failure leaves the
        network self-consistent for the lifecycle's rollback path.
        """
        if input_size_new < self.input_size:
            raise ValueError(f"CascadeCorrelationNetwork: _resize_network_for_dataset: input_size cannot shrink ({self.input_size} → {input_size_new}); zero-pad the dataset to network dims instead")
        if output_size_new < self.output_size:
            raise ValueError(f"CascadeCorrelationNetwork: _resize_network_for_dataset: output_size cannot shrink ({self.output_size} → {output_size_new}); zero-pad the dataset targets to network dims instead")

        input_delta = input_size_new - self.input_size
        output_delta = output_size_new - self.output_size

        if input_delta == 0 and output_delta == 0:
            return {"input_delta": 0, "output_delta": 0, "hidden_preserved": len(self.hidden_units)}

        # Mutate output layer first (cheapest to roll back if hidden-unit grow
        # raises): all changes here are pure-additive on a single tensor.
        self._grow_output_layer(input_delta=input_delta, output_delta=output_delta)
        if input_delta > 0:
            self._grow_hidden_units_for_input(input_delta=input_delta)

        # Bookkeeping last — only flip the live size attributes once all
        # tensor mutations completed successfully. Reset ``active_output_dim``
        # to the new full output size: a grow always coincides with the
        # arrival of a dataset whose output dim equals the new ``output_size``,
        # so the active dim should match. The lifecycle then immediately
        # overrides this if the new dataset's output is smaller than
        # ``output_size_new`` (shouldn't happen post-resize, but defensive).
        self.input_size = input_size_new
        self.output_size = output_size_new
        self.active_output_dim = output_size_new
        # The config object is a SECOND copy of these two dimensions, and until 2026-08-23
        # nothing here updated it -- ``config.input_size`` / ``config.output_size`` were read
        # at construction and never assigned again anywhere in the tree. So a resized network
        # carried a permanently stale config, and every snapshot taken afterwards recorded
        # the contradiction: ``arch`` and the tensors at the NEW width, ``config_json`` at the
        # old one.
        #
        # That is not hypothetical. It made 239 of the archive's 27,908 snapshots unloadable
        # (juniper-ml#1254): the loader rebuilds from ``config_json``, so it produced a
        # network narrower than the tensors about to be loaded into it, and the arch
        # integrity gate correctly refused the file. juniper-cascor#560 taught the LOADER to
        # recover those by preferring ``arch`` when the tensors corroborate it; this is the
        # writer-side half, so newly-written snapshots stop needing that recovery at all.
        self._sync_config_dimensions()

        self.logger.debug(f"CascadeCorrelationNetwork: _resize_network_for_dataset: grew network to input={input_size_new} output={output_size_new} (deltas: input={input_delta}, output={output_delta}, hidden={len(self.hidden_units)} preserved)")
        return {
            "input_delta": input_delta,
            "output_delta": output_delta,
            "hidden_preserved": len(self.hidden_units),
        }

    def _grow_output_layer(self, input_delta: int, output_delta: int) -> None:
        """Expand ``output_weights`` (rows for input grow, cols for output grow)
        and ``output_bias`` (elements for output grow). New entries random-init
        × ``self.random_value_scale``.

        ``output_weights`` shape is ``[input_size + n_hidden, output_size]``.
        Rows ``0..input_size-1`` are raw-input → output contributions; rows
        ``input_size..input_size+n_hidden-1`` are hidden-unit → output
        contributions (see ``forward()`` line 1577). Input grow MUST insert at
        index ``self.input_size`` so the hidden-row block shifts down and the
        layout stays aligned with ``_compute_hidden_outputs`` (line 1539-1545).

        Output grow is a plain column-append on ``output_weights`` plus an
        element-append on ``output_bias`` — hidden units have no output-dim
        coupling, so they're untouched here.
        """
        w = self.output_weights.detach()

        if input_delta > 0:
            new_rows = torch.randn(input_delta, w.size(1)) * self.random_value_scale
            w = torch.cat(
                [
                    w[: self.input_size, :],  # preserve raw-input rows
                    new_rows,  # random-init new raw-input rows
                    w[self.input_size :, :],  # preserve hidden-unit rows (shifted down)
                ],
                dim=0,
            )

        if output_delta > 0:
            new_cols = torch.randn(w.size(0), output_delta) * self.random_value_scale
            w = torch.cat([w, new_cols], dim=1)

        w.requires_grad_(True)
        self.output_weights = w

        if output_delta > 0:
            b = self.output_bias.detach()
            new_b = torch.randn(output_delta) * self.random_value_scale
            b = torch.cat([b, new_b], dim=0)
            b.requires_grad_(True)
            self.output_bias = b

    def _grow_hidden_units_for_input(self, input_delta: int) -> None:
        """For each hidden unit, insert ``input_delta`` random-init entries
        into its weight vector at index ``self.input_size``.

        Hidden unit ``i`` has weight vector shape ``[self.input_size + i]`` with
        layout ``[raw_input_0..raw_input_{I-1}, prior_hidden_0..prior_hidden_{i-1}]``.
        The forward pass at line 1544 does ``torch.sum(unit_input * unit["weights"], dim=1)``
        on ``buffer[:, :col]`` where ``buffer`` is ``[raw_inputs | prior_hiddens]`` —
        so the weight vector must keep that ordering. Appending new entries
        at the end would put them after the prior-hidden block, breaking the
        positional alignment. Hence row-insertion at the boundary.

        Hidden-unit biases (per-unit scalars) carry no input-dim coupling and
        are not touched. Weights stay detached after the cat (cascade-correlation
        freezes hidden parameters after promotion; output training updates
        ``output_weights`` only).
        """
        for unit in self.hidden_units:
            w = unit["weights"].detach()
            new_entries = torch.randn(input_delta) * self.random_value_scale
            w = torch.cat(
                [
                    w[: self.input_size],  # preserve raw-input slots
                    new_entries,  # random-init new raw-input slots
                    w[self.input_size :],  # preserve prior-hidden slots (shifted)
                ],
                dim=0,
            )
            # Stays detached — matches the cascade-correlation convention at
            # ``_install_hidden_unit_helper`` (line ~3568: weights.clone().detach()).
            unit["weights"] = w

    # ------------------------------------------------------------------
    # P2-2 (Issue #3): live dataset swap history persistence.
    #
    # ``record_dataset_swap_event`` is the sole writer for
    # ``self.history["dataset_swaps"]``. Called from
    # ``swap_dataset_live`` in the lifecycle layer at the end of the
    # success path (post-broadcast, pre-return), so cancelled swaps and
    # rolled-back failures never produce a stale event.
    #
    # The payload schema mirrors the §3.3 swap response shape so canopy's
    # P2-7 timeline UI can render markers without joining against any
    # other state. Snapshot ID fields are placeholders that P2-3 will
    # populate once pre/post-swap auto-snapshots are wired.
    # ------------------------------------------------------------------

    def record_dataset_swap_event(
        self,
        *,
        before_cfg: Optional[Dict[str, Any]],
        after_cfg: Optional[Dict[str, Any]],
        arch_changes: Dict[str, Any],
        pre_swap_snapshot_id: Optional[str] = None,
        post_swap_snapshot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a structured ``dataset_swap`` event to ``self.history``.

        Returns the event dict that was appended so the caller can use it
        for logging or for forwarding to a future broadcast layer (see
        ``notes/PHASE_2_P2_2_FOLLOWUPS_2026-05-14.md`` Follow-up A).

        Each event has the schema:

        ```
        {
            "timestamp":              ISO-8601 UTC string,
            "before_cfg":             dict | None,
            "after_cfg":              dict | None,
            "arch_changes":           dict (the §3.3 response shape),
            "pre_swap_snapshot_id":   str | None  (P2-3 backfills),
            "post_swap_snapshot_id":  str | None  (P2-3 backfills),
        }
        ```

        ``before_cfg`` / ``after_cfg`` are shallow-copied so a later mutation
        of the caller's dataset_config dict can't ripple back into the
        history. ``arch_changes`` is deep-copied for the same reason
        (the nested ``appended_nodes`` / ``prepended_layers`` containers
        would otherwise alias).
        """
        event = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "before_cfg": dict(before_cfg) if before_cfg is not None else None,
            "after_cfg": dict(after_cfg) if after_cfg is not None else None,
            "arch_changes": copy.deepcopy(arch_changes),
            "pre_swap_snapshot_id": pre_swap_snapshot_id,
            "post_swap_snapshot_id": post_swap_snapshot_id,
        }
        self.history.setdefault("dataset_swaps", []).append(event)
        self.logger.debug(f"CascadeCorrelationNetwork: record_dataset_swap_event: recorded event #{len(self.history['dataset_swaps'])} timestamp={event['timestamp']}")
        return event

    def _init_activation_function(self):
        """Initialize activation function components."""
        self.activation_function_name = self.config.activation_function_name or _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME
        self.activation_functions_dict = self.config.activation_functions_dict or _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT
        self.activation_fn_no_diff = (
            self.activation_functions_dict.get(
                self.activation_function_name,
                _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,
            )
            or _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT
        )
        self.activation_fn = self._init_activation_with_derivative(self.activation_fn_no_diff)

    def _init_multiprocessing(self) -> None:
        """Initialize multiprocessing context and manager attributes."""
        self.logger.trace("CascadeCorrelationNetwork: _init_multiprocessing: Initializing multiprocessing components")

        # Initialize multiprocessing context using configured context type
        # self._mp_ctx = mp.get_context("forkserver")
        # This is unnecessary:  Changing Context type did not corrUse 'fork' context for better compatibility with BaseManager on Linux
        context_type = self.config.candidate_training_context_type or _CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_CONTEXT
        self._mp_ctx = mp.get_context(context_type)

        # Only set forkserver preload if using forkserver context
        if context_type == "forkserver":
            try:
                self._mp_ctx.set_forkserver_preload(
                    [
                        "os",
                        "uuid",
                        "torch",
                        "numpy",
                        "random",
                        "logging",
                        "datetime",
                    ]
                )
            except Exception as e:
                self.logger.warning(f"CascadeCorrelationNetwork: _init_multiprocessing: Failed to set forkserver preload: {e}")

        # Initialize manager attributes (retained for backward compatibility; no longer used in RC-2 path)
        self._manager = None
        self._task_queue = None
        self._result_queue = None

        # PARALLEL-FIX (RC-4): Persistent worker pool state. Workers are created once and reused
        # across training rounds, eliminating per-round process creation, PyTorch initialization,
        # and 4-phase shutdown overhead.
        self._persistent_workers = []
        self._persistent_task_queue = None
        self._persistent_result_queue = None
        self._persistent_progress_queue = None
        self._persistent_pool_size = 0

        # OPT-5: Track active SharedMemory blocks for cleanup on error/shutdown
        self._active_shm_blocks = []
        atexit.register(self._cleanup_shared_memory)
        # Issue #509: the RC-4 pool is released at the end of every fit(); this is the
        # belt-and-braces arm for a pool created outside a fit (a direct
        # train_candidates call) or left behind by an abrupt exit. Registering a bound
        # method pins the instance for the process lifetime — the same trade the
        # _cleanup_shared_memory registration above already makes.
        atexit.register(self._release_candidate_worker_pool)

        # Phase 1b: Remote worker coordinator reference (set via set_worker_coordinator)
        self._worker_coordinator = None
        self._remote_workers_enabled = getattr(self.config, "enable_remote_workers", False)

        # Phase 3: Unified TaskDistributor for local + remote scheduling
        self._task_distributor = TaskDistributor(dist_logger=self.logger)

        # Initialize multiprocessing config values
        self.candidate_training_queue_authkey = self.config.candidate_training_queue_authkey
        self.candidate_training_queue_address = self.config.candidate_training_queue_address
        self.candidate_training_tasks_queue_timeout = self.config.candidate_training_task_queue_timeout or _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT
        self.candidate_training_shutdown_timeout = self.config.candidate_training_shutdown_timeout or _CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT
        self.candidate_training_context = mp.get_context(self.config.candidate_training_context_type) or _CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_CONTEXT

        # PARALLEL-FIX (RC-1): Configure PyTorch thread count for the parent process.
        # When the parent process also runs with default thread count, it competes with workers
        # for CPU cores. Limit the parent to 2 threads (enough for its own forward passes and
        # result processing, without starving worker processes).
        parent_thread_count = max(2, getattr(self.config, "worker_thread_count", 1) * 2)
        torch.set_num_threads(parent_thread_count)
        self.logger.debug(f"CascadeCorrelationNetwork: _init_multiprocessing: Parent process PyTorch thread count set to {parent_thread_count}")

        self.logger.debug("CascadeCorrelationNetwork: _init_multiprocessing: Multiprocessing components initialized")

    def set_worker_coordinator(self, coordinator) -> None:
        """Set the remote worker coordinator for dual-path dispatch.

        Called by the API layer to inject the coordinator reference when
        remote workers are enabled. This allows the training thread to
        dispatch tasks to remote WebSocket workers.

        Args:
            coordinator: WorkerCoordinator instance from the API layer.
        """
        self._worker_coordinator = coordinator
        self._remote_workers_enabled = True
        self._task_distributor.set_coordinator(coordinator)
        self.logger.info("CascadeCorrelationNetwork: Remote worker coordinator set — dual-path dispatch enabled")

    def _remote_result_collection_timeout(self) -> float:
        """Seconds to wait for remote candidate-training results in a dual-path round.

        ISSUE-319 (defect #3): the remote leg previously reused
        ``candidate_training_shutdown_timeout`` (~10s — a *process teardown* budget) to
        wait for a *full candidate-training round* (tens of seconds for the default
        ``candidate_epochs``). The wait therefore expired on every cascade growth, the
        remote results were discarded as "late", and the tasks were retried on the
        already-saturated local pool — starving the network at a single hidden unit.

        Scale the budget to the training workload (``candidate_epochs``) with a floor and
        a hard ceiling. The ceiling, paired with the worker-liveness early-exit in
        ``WorkerCoordinator.collect_results``, bounds the worst case (all remote workers
        die mid-round): the round falls back to local retry promptly instead of blocking
        for the full budget. The wait returns as soon as all results arrive, so a healthy
        round is unaffected — the budget is only an upper bound.
        """
        epochs = int(getattr(self, "candidate_epochs", 0) or 0)
        estimate = epochs * _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_SECONDS_PER_EPOCH
        return max(
            _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MIN_TIMEOUT,
            min(estimate, _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MAX_TIMEOUT),
        )

    def _dispatch_to_remote_workers(
        self,
        tasks: list,
        candidate_input: torch.Tensor,
        y: torch.Tensor,
        residual_error: torch.Tensor,
    ) -> list:
        """Dispatch tasks to remote WebSocket workers and collect results.

        Converts internal task tuples into the wire protocol format and
        submits them to the WorkerCoordinator. Blocks until all results
        are received or timeout expires. Converts TaskResults back into
        CandidateTrainingResult objects for compatibility with the
        existing result processing pipeline.

        Args:
            tasks: Internal task tuples from _generate_candidate_tasks.
            candidate_input: Enhanced input tensor.
            y: Target tensor.
            residual_error: Residual error tensor.

        Returns:
            List of CandidateTrainingResult objects.
        """
        round_id = str(uuid.uuid4())
        self.logger.info(
            "CascadeCorrelationNetwork: _dispatch_to_remote_workers: Dispatching %d tasks to remote workers (round %s)",
            len(tasks),
            round_id,
        )

        # Convert tensors to numpy for wire protocol
        tensors = {
            "candidate_input": candidate_input.numpy().astype(np.float32),
            "y": y.numpy().astype(np.float32),
            "residual_error": residual_error.numpy().astype(np.float32),
        }

        # Convert internal task tuples to wire protocol task specs
        task_specs = []
        # ISSUE-319 FIX (the actual stall trigger): source the per-round candidate
        # training params from the network config, NOT from positional indexing into
        # ``training_inputs``. Under the OPT-5 SharedMemory path ``training_inputs`` is a
        # *dict* (shm metadata), so ``training_inputs[1]`` raised ``KeyError(1)`` (masked
        # as the bare "(1)") on every dual-path round, forcing the remote dispatch into a
        # starving local-retry fallback. These are per-round constants — the OPT-5 dict
        # itself is populated from these same ``self.candidate_*`` attributes.
        for _task_idx, candidate_data_tuple, _training_inputs in tasks:
            candidate_index, input_size, activation_name, random_value_scale, candidate_uuid, candidate_seed, random_max_value, sequence_max_value = candidate_data_tuple
            task_specs.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_data": {
                        "input_size": input_size,
                        "activation_name": activation_name,
                        "random_value_scale": float(random_value_scale),
                        "candidate_uuid": candidate_uuid,
                        "candidate_seed": candidate_seed,
                        # Remote candidate bounds are int-valued and feed random.randint()/
                        # range() in CandidateUnit (which reject floats). Send them as int so
                        # the JSON wire preserves integer-ness — float() here made the remote
                        # worker raise "'float' object cannot be interpreted as an integer".
                        "random_max_value": int(random_max_value),
                        "sequence_max_value": int(sequence_max_value),
                    },
                    "training_params": {
                        "epochs": int(self.candidate_epochs),
                        "learning_rate": float(self.candidate_learning_rate),
                        "display_frequency": int(self.candidate_display_frequency),
                    },
                }
            )

        # Submit to coordinator
        self._worker_coordinator.submit_tasks(round_id, task_specs, tensors)

        # Block until results arrive. ISSUE-319 (defect #3): budget the wait to the
        # candidate-training workload, NOT candidate_training_shutdown_timeout (~10s, a
        # process-teardown budget) which always expired before a remote round could
        # finish and forced a starving local-retry fallback on every cascade growth.
        timeout = self._remote_result_collection_timeout()
        remote_results = self._worker_coordinator.collect_results(timeout=timeout)

        # ISSUE-319 FIX: ``task_specs`` holds only the remote subset of this round's
        # candidates, but ``tr.candidate_id`` is the GLOBAL pool index (0..pool_size-1,
        # assigned in ``_generate_candidate_tasks``). The previous code did
        # ``task_specs[tr.candidate_id]`` — indexing the local subset list by a global
        # id — which raised for any remote candidate whose global id exceeded the remote
        # slice length (the common case: e.g. ids 15,16 in a 40-pool split 38-local /
        # 2-remote). That exception was swallowed by
        # ``TaskDistributor._execute_remote_with_fallback`` and forced a local-retry
        # fallback every cascade round, which then starved — pinning the network at one
        # hidden unit. Map results by ``candidate_index`` so the lookup is by identity.
        spec_by_candidate_id = {spec["candidate_index"]: spec for spec in task_specs}

        # Convert TaskResults back to CandidateTrainingResult
        results = []
        for tr in remote_results:
            # ISSUE-319 (defect #4): drop results from any round other than this one.
            # ``submit_tasks`` clears ``_results`` per round, but a late result for a
            # still-tracked stale ``_pending_tasks`` entry can be accepted into a later
            # round; the authoritative ``round_id`` (attached server-side in
            # ``submit_result`` from the dispatching ``PendingTask``) lets us round-isolate
            # the remote leg exactly as RC-5 does for local results in
            # ``_collect_training_results``. ``getattr`` keeps this safe for any
            # ``TaskResult`` constructed without the field.
            tr_round_id = getattr(tr, "round_id", None)
            if tr_round_id is not None and tr_round_id != round_id:
                self.logger.warning(
                    "CascadeCorrelationNetwork: _dispatch_to_remote_workers: " "Discarding remote result for candidate_id=%s from stale round %s (current round %s)",
                    tr.candidate_id,
                    tr_round_id,
                    round_id,
                )
                continue
            # A result whose candidate_id is not part of THIS dispatch is stale/leaked or
            # a protocol mismatch. Skip it rather than crash or mis-bind ``input_size``.
            spec = spec_by_candidate_id.get(tr.candidate_id)
            if spec is None:
                self.logger.warning(
                    "CascadeCorrelationNetwork: _dispatch_to_remote_workers: " "Discarding remote result with unknown candidate_id=%s " "(not among this round's %d dispatched task(s))",
                    tr.candidate_id,
                    len(task_specs),
                )
                continue

            # Reconstruct tensors as torch tensors
            weights = torch.tensor(tr.tensors.get("weights", np.array([])), dtype=torch.float32) if "weights" in tr.tensors else None
            bias = torch.tensor(tr.tensors.get("bias", np.array([])), dtype=torch.float32) if "bias" in tr.tensors else None
            norm_output = torch.tensor(tr.tensors["norm_output"], dtype=torch.float32) if "norm_output" in tr.tensors else None
            norm_error = torch.tensor(tr.tensors["norm_error"], dtype=torch.float32) if "norm_error" in tr.tensors else None

            # Create a CandidateUnit from the result data
            candidate = CandidateUnit(
                CandidateUnit__input_size=spec["candidate_data"]["input_size"],
                CandidateUnit__activation_function_name=tr.activation_name,
            )
            if weights is not None:
                candidate.weights = nn.Parameter(weights)
            if bias is not None:
                candidate.bias = nn.Parameter(bias)

            ctr = CandidateTrainingResult(
                candidate_id=tr.candidate_id,
                candidate_uuid=tr.candidate_uuid,
                correlation=tr.correlation,
                candidate=candidate,
                best_corr_idx=tr.best_corr_idx,
                all_correlations=tr.all_correlations,
                norm_output=norm_output,
                norm_error=norm_error,
                numerator=tr.numerator,
                denominator=tr.denominator,
                success=tr.success,
                epochs_completed=tr.epochs_completed,
                error_message=tr.error_message,
            )
            results.append(ctr)

        self.logger.info(
            "CascadeCorrelationNetwork: _dispatch_to_remote_workers: Collected %d/%d results from remote workers",
            len(results),
            len(tasks),
        )
        return results

    def _init_display_components(self) -> None:
        """Initialize display and plotting components."""
        self.logger.trace("CascadeCorrelationNetwork: _init_display_components: Initializing display components")

        # Initialize display parameters
        self.display_frequency = self.config.display_frequency or _CASCADE_CORRELATION_NETWORK_CANDIDATE_DISPLAY_FREQUENCY
        self.epoch_display_frequency = self.config.epoch_display_frequency or _CASCADE_CORRELATION_NETWORK_EPOCH_DISPLAY_FREQUENCY
        self.generate_plots = self.config.generate_plots or _CASCADE_CORRELATION_NETWORK_GENERATE_PLOTS
        self.status_display_frequency = self.config.status_display_frequency or _CASCADE_CORRELATION_NETWORK_STATUS_DISPLAY_FREQUENCY
        self.candidate_display_frequency = self.config.candidate_display_frequency or _CASCADE_CORRELATION_NETWORK_DISPLAY_FREQUENCY

        # Initialize display progress functions
        self._network_display_progress = display_progress(display_frequency=self.epoch_display_frequency)
        self._status_display_progress = display_progress(display_frequency=self.status_display_frequency)
        self._candidate_display_progress = display_progress(display_frequency=self.candidate_display_frequency)

        # Initialize plotter
        self.plotter = CascadeCorrelationPlotter(logger=self.logger)
        self.logger.debug("CascadeCorrelationNetwork: _init_display_components: Display components initialized")

    @classmethod
    def create_simple_network(
        cls,
        input_size: int = 2,
        output_size: int = 1,
        learning_rate: float = 0.1,
        max_hidden_units: int = 10,
        **kwargs,
    ):
        """
        Factory method to create a CascadeCorrelationNetwork with simplified configuration.
        Args:
            input_size: Number of input features
            output_size: Number of output classes
            learning_rate: Learning rate for training
            max_hidden_units: Maximum number of hidden units to add
            **kwargs: Additional configuration parameters
        Returns:
            CascadeCorrelationNetwork: Configured network instance
        """
        config = CascadeCorrelationConfig.create_simple_config(
            input_size=input_size,
            output_size=output_size,
            learning_rate=learning_rate,
            max_hidden_units=max_hidden_units,
            **kwargs,
        )
        return cls(config=config)

    #################################################################################################################################################################################################
    # Helper method to perform initialization tasks for the __init__ method
    def _initialize_randomness(
        self,
        seed: Optional[int] = None,
        sequence_max_value: Optional[int] = None,
        random_max_value: Optional[int] = None,
    ) -> None:
        """
        Description:
            Initialize randomness for the cascade correlation network.
        Args:
            seed: Optional seed for random number generation
            sequence_max_value: Optional maximum value for random sequence generation
            random_max_value: Optional maximum value for random number generation
        """
        self.logger.trace("CascadeCorrelationNetwork: _initialize_randomness: Initializing randomness for the cascade correlation network")
        seed = seed or _CASCADE_CORRELATION_NETWORK_RANDOM_SEED
        self.logger.verbose(f"CascadeCorrelationNetwork: _initialize_randomness: Random seed set to: {seed}")
        sequence_max_value = sequence_max_value or _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE
        self.logger.verbose(f"CascadeCorrelationNetwork: _initialize_randomness: Random sequence max value set to: {sequence_max_value}")
        random_max_value = random_max_value or _CASCADE_CORRELATION_NETWORK_RANDOM_MAX_VALUE
        self.logger.verbose(f"CascadeCorrelationNetwork: _initialize_randomness: Random max value set to: {random_max_value}")
        self._seed_random_generator(
            seed=seed,
            max_value=sequence_max_value,
            seeder=np.random.seed,
            generator=np.random.randint,
        )
        self.logger.trace("CascadeCorrelationNetwork: _initialize_randomness: Completed initialization of numpy random generator with seed and sequence for the cascade correlation network")
        self._seed_random_generator(
            seed=seed,
            max_value=sequence_max_value,
            seeder=random.seed,
            generator=random.randint,
        )
        self.logger.trace("CascadeCorrelationNetwork: _initialize_randomness: Completed initialization of random random generator with seed and sequence for the cascade correlation network")
        self._seed_random_generator(
            seed=seed,
            max_value=sequence_max_value,
            seeder=torch.manual_seed,
            generator=lambda min, max: torch.randint(min, max, ()),
        )
        self.logger.trace("CascadeCorrelationNetwork: _initialize_randomness: Completed initialization of torch random generator with seed and sequence for the cascade correlation network")
        self._seed_random_generator(
            seed=seed,
            max_value=sequence_max_value,
            seeder=self._seed_hash,
            generator=None,
        )
        # if torch.cuda.is_available():
        #     self.logger.trace("CascadeCorrelationNetwork: _initialize_randomness: CUDA is available, seeding CUDA random generator.")
        #     self._seed_random_generator(seed=seed, max_value=sequence_max_value, seeder=torch.cuda.manual_seed, generator=lambda min, max: torch.rand(1, device='cuda'))
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False

    def _seed_random_generator(
        self,
        seed: int = None,
        max_value: int = None,
        # seeder: callable = None,
        seeder: Callable[..., Any] = None,
        # generator: callable = None,
        generator: Callable[..., Any] = None,
    ) -> None:
        """
        Description:
            Seed the random generator for the cascade correlation network.
        Args:
            seed: The seed value for the random generator
            max_value: The maximum value for the random generator
            seeder: The seeder function for the random generator
            generator: The random number generator function
        Note:
            This method seeds the random generator using the provided seed and max value.
            It then rolls the random generator to a specific sequence number.
        Returns:
            None
        """
        self.logger.trace("CascadeCorrelationNetwork: _seed_random_generator: Seeding random module with seed and max value.")
        if seeder is None:
            self.logger.verbose("CascadeCorrelationNetwork: _seed_random_generator: No seeder function provided, skipping seeding of random generator.")
            return
        seeder(seed)
        self.logger.trace("CascadeCorrelationNetwork: _seed_random_generator: Random seed set for random module.")
        if generator is None:
            self.logger.verbose("CascadeCorrelationNetwork: _seed_random_generator: No generator function provided, skipping random sequence generation and rolling.")
            return
        random_sequence = random.randint(0, max_value)  # trunk-ignore(bandit/B311)
        self.logger.verbose(f"CascadeCorrelationNetwork: _seed_random_generator: Random sequence number rolled to: {random_sequence}")

        # TODO:  Enable CUDA random generator seeding and rolling when needed
        #     self._seed_random_generator(seed=seed, max_value=sequence_max_value, seeder=torch.cuda.manual_seed, generator=lambda min, max: torch.rand(1, device='cuda'))
        # File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 392, in _roll_sequence_number
        #     discard = [generator(0, max_value) for _ in range(sequence)]
        # TypeError: only integer tensors of a single element can be converted to an index
        self._roll_sequence_number(sequence=random_sequence, max_value=max_value, generator=generator)
        self.logger.trace("CascadeCorrelationNetwork: _seed_random_generator: Completed initialization of random generator with seed and sequence for the cascade correlation network")

    # def _roll_sequence_number(self, sequence: int = None, max_value: int = None, generator: callable = None) -> None:
    def _roll_sequence_number(self, sequence: int = None, max_value: int = None, generator: Callable[..., Any] = None) -> None:
        """
        Description:
            Roll the sequence number for the cascade correlation network.
        Args:
            sequence: The current sequence number
            max_value: The maximum value for the random number generator
            generator: The random number generator function
        Note:
            This method rolls the random generator discarding the first sequence number of integers for the cascade correlation network
        Returns:
            None
        """
        self.logger.trace("CascadeCorrelationNetwork: _roll_sequence_number: Rolling sequence number.")
        self.logger.debug(f"CascadeCorrelationNetwork: _roll_sequence_number: Rolling sequence number to: {sequence} with max value: {max_value} using generator: {generator}")
        if generator is not None:
            # CASCOR-P1-008 / CR-041 FIX: Replaced list comprehension with simple loop to avoid OOM
            # OLD (can cause OOM if sequence is large - up to 2^32-1):
            # discard = [generator(0, max_value) for _ in range(sequence)]
            # NEW: Loop without storing, and cap roll count to prevent excessive iterations
            MAX_ROLL_COUNT = 10000
            roll_count = min(sequence, MAX_ROLL_COUNT) if sequence else 0
            for _ in range(roll_count):
                generator(0, max_value)
            self.logger.verbose(f"CascadeCorrelationNetwork: _roll_sequence_number: Discarded {roll_count} random values to roll to the desired sequence.")
            if sequence and sequence > MAX_ROLL_COUNT:
                self.logger.warning(f"CascadeCorrelationNetwork: _roll_sequence_number: Sequence {sequence} exceeded MAX_ROLL_COUNT {MAX_ROLL_COUNT}, capped at {MAX_ROLL_COUNT}")
            self.logger.verbose(f"CascadeCorrelationNetwork: _roll_sequence_number: Random Generator rolled for sequence number: {sequence}")
        self.logger.trace("CascadeCorrelationNetwork: _roll_sequence_number: Completed rolling of sequence number.")

    def _seed_hash(self, seed: int = None) -> None:
        """
        Description:
            Seed the hash function for the cascade correlation network.
        Args:
            seed: The seed value for the hash function
        """
        os.environ["PYTHONHASHSEED"] = str(seed)

    # Helper method to add hidden units to the network
    # def _init_activation_with_derivative(self, activation_fn: callable = None) -> ActivationWithDerivative:
    def _init_activation_with_derivative(self, activation_fn: Callable[..., Any] = None) -> ActivationWithDerivative:
        """
        Description:
            Wrap activation function to also provide its derivative.
        Args:
            activation_fn: Base activation function
        Note:
            This method wraps the activation function to also provide its derivative.
            CASCOR-P1-003: Now returns ActivationWithDerivative class instance instead of local function.
        Returns:
            ActivationWithDerivative instance that can compute both activation and its derivative
        """
        # Validate the activation function
        self.logger.trace("CascadeCorrelationNetwork: _init_activation_with_derivative: Validating activation function")
        activation_fn = (
            activation_fn,
            _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,
        )[activation_fn is None]
        self.logger.debug(f"CascadeCorrelationNetwork: _init_activation_with_derivative: Using activation function: {activation_fn}")

        # CASCOR-P1-003: Use picklable ActivationWithDerivative class instead of local function
        # OLD: Local function - NOT picklable for multiprocessing!
        # self.logger.trace(
        #     "CascadeCorrelationNetwork: _init_activation_with_derivative: Wrapping activation function to provide its derivative."
        # )
        # def wrapped_activation(x, derivative: bool = False):
        #     if derivative:
        #         if activation_fn in [
        #             _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_TANH,
        #             _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_TANH,
        #         ]:  # For tanh, derivative is 1 - tanh^2(x)
        #             return 1.0 - activation_fn(x) ** 2
        #         elif activation_fn in [
        #             _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_SIGMOID,
        #             _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_SIGMOID,
        #         ]:  # For sigmoid, derivative is sigmoid(x) * (1 - sigmoid(x))
        #             y = activation_fn(x)
        #             return y * (1.0 - y)
        #         elif activation_fn in [
        #             _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_RELU,
        #             _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_RELU,
        #         ]:  # For ReLU, derivative is 1 for x > 0, 0 otherwise
        #             return (x > 0).float()
        #         else:  # Numerical approximation for other functions
        #             eps = 1e-6
        #             return (activation_fn(x + eps) - activation_fn(x - eps)) / (2 * eps)
        #     else:
        #         return activation_fn(x)
        # return wrapped_activation

        # NEW: Picklable class instance for multiprocessing compatibility
        self.logger.trace("CascadeCorrelationNetwork: _init_activation_with_derivative: Creating ActivationWithDerivative wrapper.")
        wrapped_activation = ActivationWithDerivative(activation_fn)
        self.logger.verbose(f"CascadeCorrelationNetwork: _init_activation_with_derivative: Returning wrapped activation function: {wrapped_activation}.")

        # Return the wrapped activation function
        self.logger.trace("CascadeCorrelationNetwork: _init_activation_with_derivative: Completed wrapping of activation function.")
        return wrapped_activation

    #################################################################################################################################################################################################
    # Input validation methods
    #################################################################################################################################################################################################
    def _validate_tensor_input(
        self,
        x: torch.Tensor,
        param_name: str = "x",
        allow_none: bool = False,
        allow_empty: bool = False,
    ) -> None:
        """
        Validate tensor input parameters.
        Args:
            x: Input tensor to validate
            param_name: Name of the parameter for error messages
            allow_none: Whether to allow None values
            allow_empty: Whether to allow empty (zero-batch) tensors
        Raises:
            ValidationError: If tensor is invalid
        """
        if allow_none and x is None:
            return
        if x is None:
            # raise ValidationError(f"Parameter '{param_name}' cannot be None")
            raise ValidationError(f"Parameter {param_name!r} cannot be None")
        if not isinstance(x, torch.Tensor):
            # raise ValidationError(f"Parameter '{param_name}' must be a torch.Tensor, got {type(x)}")
            raise ValidationError(f"Parameter {param_name!r} must be a torch.Tensor, got {type(x)}")
        if x.numel() == 0 and not allow_empty:
            # raise ValidationError(f"Parameter '{param_name}' cannot be an empty tensor")
            raise ValidationError(f"Parameter {param_name!r} cannot be an empty tensor")
        # Skip NaN/Inf checks for empty tensors
        if x.numel() > 0:
            if torch.isnan(x).any():
                # raise ValidationError(f"Parameter '{param_name}' contains NaN values")
                raise ValidationError(f"Parameter {param_name!r} contains NaN values")
            if torch.isinf(x).any():
                # raise ValidationError(f"Parameter '{param_name}' contains infinite values")
                raise ValidationError(f"Parameter {param_name!r} contains infinite values")

    def _validate_tensor_shapes(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        expected_input_features: int = None,
    ) -> None:
        """
        Validate tensor shapes for compatibility.
        Args:
            x: Input tensor
            y: Target tensor (optional)
            expected_input_features: Expected number of input features
        Raises:
            ValidationError: If shapes are incompatible
        """
        if len(x.shape) != 2:
            raise ValidationError(f"Input tensor must be 2D (batch_size, features), got shape {x.shape}")
        if expected_input_features is not None and x.shape[1] != expected_input_features:
            raise ValidationError(f"Expected {expected_input_features} input features, got {x.shape[1]}")
        if y is not None:
            if len(y.shape) != 2:
                raise ValidationError(f"Target tensor must be 2D (batch_size, classes), got shape {y.shape}")
            if x.shape[0] != y.shape[0]:
                raise ValidationError(f"Input and target batch sizes must match: {x.shape[0]} != {y.shape[0]}")

    def _validate_numeric_parameter(
        self,
        value,
        param_name: str,
        min_val=None,
        max_val=None,
        allow_none: bool = False,
    ) -> None:
        """
        Validate numeric parameters.
        Args:
            value: Value to validate
            param_name: Name of the parameter for error messages
            min_val: Minimum allowed value (optional)
            max_val: Maximum allowed value (optional)
            allow_none: Whether to allow None values
        Raises:
            ValidationError: If value is invalid
        """
        if allow_none and value is None:
            return
        if value is None:
            # raise ValidationError(f"Parameter '{param_name}' cannot be None")
            raise ValidationError(f"Parameter {param_name!r} cannot be None")
        if not isinstance(value, (int, float)):
            # raise ValidationError(f"Parameter '{param_name}' must be numeric, got {type(value)}")
            raise ValidationError(f"Parameter {param_name!r} must be numeric, got {type(value)}")
        if min_val is not None and value < min_val:
            # raise ValidationError(f"Parameter '{param_name}' must be >= {min_val}, got {value}")
            raise ValidationError(f"Parameter {param_name!r} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            # raise ValidationError(f"Parameter '{param_name}' must be <= {max_val}, got {value}")
            raise ValidationError(f"Parameter {param_name!r} must be <= {max_val}, got {value}")

    def _validate_positive_integer(self, value, param_name: str, allow_zero: bool = False) -> None:
        """
        Validate positive integer parameters.

        Accepts both Python ``int`` and ``numpy.integer`` subclasses (``int64``,
        ``int32``, etc.). h5py-restored snapshot attributes deserialize as
        numpy scalars; rejecting them broke every snapshot-restore →
        ``fit()`` path (BUG-CC-09 regression, 2026-05-19).

        Args:
            value: Value to validate
            param_name: Name of the parameter
            allow_zero: Whether to allow zero values
        Raises:
            ValidationError: If value is invalid
        """
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            # raise ValidationError(f"Parameter '{param_name}' must be an integer, got {type(value)}")
            raise ValidationError(f"Parameter {param_name!r} must be an integer, got {type(value)}")
        min_val = 0 if allow_zero else 1
        if value < min_val:
            # raise ValidationError(f"Parameter '{param_name}' must be >= {min_val}, got {value}")
            raise ValidationError(f"Parameter {param_name!r} must be >= {min_val}, got {value}")

    #################################################################################################################################################################################################
    # Define Public Methods for Training and Evaluation
    #################################################################################################################################################################################################

    def _create_candidate_unit(
        self,
        candidate_index: int,
        candidate_uuid: Optional[str] = None,
        input_size: Optional[int] = None,
        **kwargs,
    ) -> CandidateUnit:
        """
        Factory method to create candidate units with consistent parameters.
        Args:
            candidate_index: Index of candidate in pool
            candidate_uuid: UUID for candidate (generates if None)
            input_size: Input size (uses network input_size if None)
            **kwargs: Additional CandidateUnit parameters
        Returns:
            Configured CandidateUnit instance
        """
        self.logger.debug(f"CascadeCorrelationNetwork: _create_candidate_unit: Creating candidate unit {candidate_index}")
        return CandidateUnit(
            CandidateUnit__activation_function=kwargs.get("activation_fn", self.activation_fn),
            CandidateUnit__input_size=input_size or self.input_size,
            CandidateUnit__output_size=kwargs.get("output_size", self.output_size),
            CandidateUnit__learning_rate=kwargs.get("learning_rate", self.candidate_learning_rate),
            CandidateUnit__epochs=kwargs.get("epochs", self.candidate_epochs),
            CandidateUnit__candidate_index=candidate_index,
            CandidateUnit__uuid=candidate_uuid,
            CandidateUnit__random_seed=kwargs.get("random_seed", self.random_seed),
            CandidateUnit__random_value_scale=kwargs.get("random_value_scale", self.random_value_scale),
            CandidateUnit__display_frequency=kwargs.get("display_frequency", self.candidate_display_frequency),
            CandidateUnit__log_level_name=kwargs.get("log_level", "INFO"),
            CandidateUnit__sequence_max_value=kwargs.get("sequence_max_value", self.sequence_max_value),
            CandidateUnit__random_max_value=kwargs.get("random_value_max", self.random_max_value),
            CandidateUnit__patience=kwargs.get("patience", self.candidate_patience),
            CandidateUnit__early_stopping=kwargs.get("early_stopping", True),
            CandidateUnit__convergence_threshold=kwargs.get("convergence_threshold", self.candidate_convergence_threshold),
        )

    # TODO: DUPLICATE FUNCTION - This version was commented out because a more complete
    # implementation exists at line ~1945. The kept version supports 15 optimizers
    # (Adadelta, Adafactor, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, LBFGS,
    # Muon, NAdam, RAdam, RMSprop, Rprop, SGD) vs only 4 in this version.
    # This duplicate should be removed after verification. - 2026-01-29
    # def _create_optimizer(self, parameters, config=None):
    #     """
    #     Create optimizer based on configuration.
    #     Args:
    #         parameters: Model parameters to optimize
    #         config: OptimizerConfig instance (uses self.optimizer_config if None)
    #     Returns:
    #         Configured optimizer instance
    #     """
    #     from cascade_correlation_config.cascade_correlation_config import OptimizerConfig
    #
    #     config = config or getattr(self, "optimizer_config", OptimizerConfig(learning_rate=self.learning_rate))
    #     optimizer_map = {
    #         "Adam": lambda: optim.Adam(
    #             parameters,
    #             lr=config.learning_rate,
    #             betas=(config.beta1, config.beta2),
    #             eps=config.epsilon,
    #             weight_decay=config.weight_decay,
    #         ),
    #         "SGD": lambda: optim.SGD(
    #             parameters,
    #             lr=config.learning_rate,
    #             momentum=config.momentum,
    #             weight_decay=config.weight_decay,
    #         ),
    #         "RMSprop": lambda: optim.RMSprop(
    #             parameters,
    #             lr=config.learning_rate,
    #             momentum=config.momentum,
    #             eps=config.epsilon,
    #             weight_decay=config.weight_decay,
    #         ),
    #         "AdamW": lambda: optim.AdamW(
    #             parameters,
    #             lr=config.learning_rate,
    #             betas=(config.beta1, config.beta2),
    #             eps=config.epsilon,
    #             weight_decay=config.weight_decay,
    #         ),
    #     }
    #     if config.optimizer_type not in optimizer_map:
    #         self.logger.warning(f"Unknown optimizer {config.optimizer_type}, defaulting to Adam")
    #         config.optimizer_type = "Adam"
    #     self.logger.debug(f"CascadeCorrelationNetwork: _create_optimizer: Creating {config.optimizer_type} optimizer with LR={config.learning_rate}")
    #     return optimizer_map[config.optimizer_type]()

    #################################################################################################################################################################################################
    # Public Method to train, grow, and evaluate the network
    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        max_epochs: int = None,
        epochs: int = None,
        max_iterations: int = None,
        early_stopping: bool = True,
    ) -> Dict[str, List]:
        """
        Train the network using the cascade correlation algorithm.
        Args:
            x_train: Training input tensor (batch_size, input_features)
            y_train: Training target tensor (batch_size, output_features)
            x_val: Validation input tensor (batch_size, input_features), optional
            y_val: Validation target tensor (batch_size, output_features), optional
            max_epochs: Maximum number of output training epochs (default: from config)
            epochs: Backward-compatible alias for max_epochs
            max_iterations: Maximum number of cascade growth iterations (default: from self.max_iterations)
            early_stopping: Whether to use early stopping
        Raises:
            ValidationError: If input tensors are invalid or have wrong shapes
            TrainingError: If training fails due to configuration issues
            ValueError: If both epochs and max_epochs are provided with different values
        Returns:
            Training history dictionary containing losses and accuracies
        """
        # Handle epochs/max_epochs alias for backward compatibility
        if epochs is not None and max_epochs is not None and epochs != max_epochs:
            raise ValueError(f"CascadeCorrelationNetwork: fit: Conflicting values for epochs ({epochs}) and max_epochs ({max_epochs})")
        if max_epochs is None and epochs is not None:
            max_epochs = epochs

        # Validate training data
        self._validate_tensor_input(x_train, "x_train")
        self._validate_tensor_input(y_train, "y_train")
        self._validate_tensor_shapes(x_train, y_train, expected_input_features=self.input_size)

        # Validate that target tensor has correct output size
        if y_train.shape[1] != self.output_size:
            raise ValidationError(f"Target tensor must have {self.output_size} output features, got {y_train.shape[1]}")

        # Validate validation data if provided
        if x_val is not None:
            self._validate_tensor_input(x_val, "x_val")
            self._validate_tensor_shapes(x_val, expected_input_features=self.input_size)
        if y_val is not None:
            self._validate_tensor_input(y_val, "y_val")
            if x_val is None:
                raise ValidationError("CascadeCorrelationNetwork: fit: Cannot provide y_val without x_val")
            self._validate_tensor_shapes(x_val, y_val, expected_input_features=self.input_size)
            if y_val.shape[1] != self.output_size:
                raise ValidationError(f"CascadeCorrelationNetwork: fit: Validation target tensor must have {self.output_size} output features, got {y_val.shape[1]}")

        # Validate max_epochs
        if max_epochs is not None:
            self._validate_positive_integer(max_epochs, "max_epochs")
        if max_iterations is not None:
            self._validate_positive_integer(max_iterations, "max_iterations")

        # Validate early_stopping
        if not isinstance(early_stopping, bool):
            raise ValidationError(f"CascadeCorrelationNetwork: fit: Parameter 'early_stopping' must be boolean, got {type(early_stopping)}")
        if len(x_train) == 0:
            raise ValidationError("CascadeCorrelationNetwork: fit: Training dataset cannot be empty")

        # Initial training of the output layer
        self.logger.trace("CascadeCorrelationNetwork: fit: Starting initial training of the output layer.")
        self.logger.info("CascadeCorrelationNetwork: fit: Initial training of output layer")
        # L-2 (SETTLED 2026-08-21, owner decision): ``max_epochs`` governs THIS initial
        # output pass only, and ``output_epochs`` governs ``grow_network``'s per-round
        # passes. That split is INTENDED, not an oversight -- it is deliberately not
        # forwarded to ``grow_network`` below, whose per-round output passes read
        # ``self.output_epochs`` directly (three sites). The two therefore agree only
        # while a caller leaves ``max_epochs`` unset (falling back to ``self.output_epochs``
        # here) or passes a value equal to it, and ``max_epochs`` is in the service's
        # ``_FIT_KWARGS``, so an explicit PATCH can still split them.
        #
        # The question this note previously left open -- whether an explicit ``max_epochs``
        # should ALSO re-budget the per-round passes -- was answered NO. Forwarding it
        # would change service behaviour and is golden-suite-visible, and the split is the
        # semantics the two knobs are named for: one bounds the initial pass, the other
        # bounds each growth round. Do not "fix" this by forwarding ``max_epochs``.
        #
        # The residual footgun is real and is INSTRUMENTED rather than removed: a config
        # setting ``max_epochs`` without ``output_epochs`` runs the service at N-then-10000
        # (the ``output_epochs`` default) while the direct CLI aliases ``max_epochs`` onto
        # every pass, so the arms are not like-for-like. juniper-ml#1159 detects exactly
        # that and records a ``validation_warnings`` entry on the run manifest. Set BOTH
        # knobs for any CLI-vs-service comparison. See the F-P1-3 root-cause note,
        # finding L-2, and juniper-ml#1143 SS2.2.
        max_epochs = (max_epochs, self.output_epochs)[max_epochs is None]
        # BUG-CC-09: re-validate after the ``self.output_epochs`` fall-back —
        # the earlier check only ran when ``max_epochs`` came in non-None,
        # so a misconfigured ``self.output_epochs`` (0 or negative) would
        # otherwise reach ``train_output_layer`` and silently no-op the
        # training loop (``range(0)`` body never executes; weights stay at
        # whatever the previous iteration left them; the returned
        # ``final_loss`` reflects the unchanged forward pass).
        self._validate_positive_integer(max_epochs, "max_epochs (resolved)")
        train_loss = self.train_output_layer(x_train, y_train, max_epochs)
        self.history["train_loss"].append(train_loss)
        if x_val is not None and y_val is not None:
            with torch.no_grad():
                value_output = self.forward(x_val)
                value_loss = nn.MSELoss()(value_output, y_val).item()
            self.history["value_loss"].append(value_loss)
            self.logger.info(f"CascadeCorrelationNetwork: fit: Initial - Train Loss: {train_loss:.6f}, Val Loss: {value_loss:.6f}")
        else:
            self.logger.info(f"CascadeCorrelationNetwork: fit: Initial - Train Loss: {train_loss:.6f}")

        # Calculate initial accuracy
        train_accuracy = self.calculate_accuracy(x_train, y_train)
        self.history["train_accuracy"].append(train_accuracy)
        if x_val is not None and y_val is not None:
            value_accuracy = self.calculate_accuracy(x_val, y_val)
            self.history["value_accuracy"].append(value_accuracy)
            self.logger.info(f"CascadeCorrelationNetwork: fit: Initial - Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {value_accuracy:.4f}")
        else:
            self.logger.info(f"CascadeCorrelationNetwork: fit: Initial - Train Accuracy: {train_accuracy:.4f}")

        # Main training loop
        patience_counter = 0
        best_value_loss = float("inf") if x_val is not None else None
        # Resolve max_iterations: explicit parameter > self.max_iterations
        max_iterations = max_iterations if max_iterations is not None else self.max_iterations
        self._validate_positive_integer(max_iterations, "max_iterations")
        self.logger.info(f"CascadeCorrelationNetwork: fit: Starting main training loop with max_epochs: {max_epochs}, max_iterations: {max_iterations}, early stopping: {early_stopping}")
        try:
            self.grow_network(
                x_train=x_train,
                y_train=y_train,
                max_iterations=max_iterations,
                early_stopping=early_stopping,
                patience_counter=patience_counter,
                best_value_loss=best_value_loss,
                x_val=x_val,
                y_val=y_val,
            )
        finally:
            # Issue #509: the growth rounds are over, so the RC-4 pool's reason to
            # persist is too. In a ``finally`` because a failed run is exactly when
            # leaked GPU-holding children do the most damage to the next one.
            self._release_candidate_worker_pool()
        self.history["hidden_units_added"].append({"correlation": 0.0, "weight_shape": (), "unit_index": -1})
        self.logger.info("CascadeCorrelationNetwork: fit: Training completed.")
        self.logger.debug(f"CascadeCorrelationNetwork: fit: Final history: {len(self.history.get('train_loss', []))} epochs recorded")
        self.logger.trace("CascadeCorrelationNetwork: fit: Completed training of the network.")
        return self.history

    #################################################################################################################################################################################################
    # Private helper: compute hidden unit outputs using pre-allocated buffer (OPT-1)
    def _compute_hidden_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute all hidden unit outputs using the OPT-1 pre-allocated buffer pattern.

        Builds a tensor of shape (batch_size, input_size + n_hidden) by pre-allocating
        a buffer and filling hidden unit output columns incrementally.  This eliminates
        N+1 torch.cat() calls that the naive implementation would require.

        Args:
            x: Input tensor with shape (batch_size, input_features)
        Returns:
            Tensor with shape (batch_size, input_size + n_hidden) containing the
            original input features followed by all hidden unit outputs.  If there
            are no hidden units, returns x directly.
        """
        n_hidden = len(self.hidden_units)
        if n_hidden == 0:
            return x
        batch_size = x.shape[0]
        total_features = self.input_size + n_hidden
        buffer = torch.empty(batch_size, total_features)
        buffer[:, : self.input_size] = x
        for i, unit in enumerate(self.hidden_units):
            col = self.input_size + i
            unit_input = buffer[:, :col]
            buffer[:, col] = unit["activation_fn"](torch.sum(unit_input * unit["weights"], dim=1) + unit["bias"])
        return buffer

    #################################################################################################################################################################################################
    # Public Method that Performs a Forward pass through the network
    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Perform a forward pass through the network.
        Args:
            x: Input tensor (batch_size, input_features)
        Raises:
            ValidationError: If input tensor is invalid or has wrong shape
        Returns:
            Network output tensor (batch_size, output_features)
        """
        # Validate input (allow empty tensors for edge case handling)
        self._validate_tensor_input(x, "x", allow_empty=True)
        self._validate_tensor_shapes(x, expected_input_features=self.input_size)
        # Start with the input features
        self.logger.trace("CascadeCorrelationNetwork: forward: Starting forward pass through the network.")
        self.logger.verbose(f"CascadeCorrelationNetwork: forward: Starting forward pass with input shape: {x.shape}")
        features = x
        self.logger.debug(f"CascadeCorrelationNetwork: forward: Input shape: {features.shape}")

        # OPT-1: Compute hidden unit outputs via shared pre-allocated buffer helper.
        output_input = self._compute_hidden_outputs(x)
        self.logger.verbose(f"CascadeCorrelationNetwork: forward: Output input shape: {output_input.shape}")

        # OPT-4: Cache candidate input (output_input == candidate_input) for reuse by _prepare_candidate_input().
        # Keyed by input data pointer to prevent stale cache consumption with different inputs.
        self._cached_candidate_input = (x.data_ptr(), output_input.detach())

        # Output layer (linear combination)
        output = torch.matmul(output_input, self.output_weights) + self.output_bias
        self.logger.debug(f"CascadeCorrelationNetwork: forward: Output shape: {output.shape}")
        self.logger.trace("CascadeCorrelationNetwork: forward: Completed forward pass through the network.")
        return output

    #################################################################################################################################################################################################
    # Public Method to train the output layer of the network
    def train_output_layer(
        self,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        epochs: int = None,
        on_epoch_callback=None,
    ) -> float:
        """
        Description:
            This method updates the weights and biases of the output layer of the network.
            Training is only applied to the output layer of the network.
        Args:
            x: Input tensor
            y: Target tensor
            epochs: Number of training epochs
        Note:
            This method only trains the output layer of the network.
        Raises:
            ValueError: If the input tensor or target tensor is None.
        Returns:
            Final loss value
        """
        # Validate input
        self.logger.trace("CascadeCorrelationNetwork: train_output_layer: Starting training of the output layer.")
        self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Input shape: {x.shape if x is not None else 'None'}, Target shape: {y.shape if y is not None else 'None'}, Epochs: {epochs}")
        epochs = (epochs, _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS)[epochs is None]
        if x is None or y is None:
            raise ValueError("CascadeCorrelationNetwork: train_output_layer: Input (x) and target (y) tensors must be provided for training the output layer.")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"CascadeCorrelationNetwork: train_output_layer: Batch size mismatch. Input x has {x.shape[0]} samples but target y has {y.shape[0]} samples.")

        # Define loss function and optimizer
        criterion = nn.MSELoss()

        # Create a simple linear layer for the output
        input_size = x.shape[1]
        self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Input size for output layer: {input_size}, Output size: {self.output_size}")
        if self.hidden_units:
            input_size += len(self.hidden_units)
        self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Adjusted input size for output layer with hidden units: {input_size}")

        # INTENTIONAL: Recreate nn.Linear and optimizer on every call.
        # In Cascade Correlation, the output layer's parameter space changes each time
        # a hidden unit is added (input_size grows), so the previous nn.Linear and
        # optimizer object are invalid.  Rebuilding from the current output_weights
        # ensures the optimizer tracks the correct parameter tensors.
        #
        # R3: recreating the OBJECT is still correct; discarding the accumulated
        # optimizer STATE is not.  On a resume the parameter space is unchanged and
        # the moments restored from the snapshot are the whole point of resuming, so
        # ``_adopt_prior_output_optimizer_state`` below transfers them onto the new
        # optimizer when — and only when — they still apply.
        output_layer = nn.Linear(input_size, self.output_size)
        with torch.no_grad():
            output_layer.weight.copy_(self.output_weights.t())  # Transpose because nn.Linear expects (out_features, in_features)
            self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Output weights shape: {self.output_weights.shape}, Transposed weights shape: {output_layer.weight.shape}")
            output_layer.bias.copy_(self.output_bias)
            self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Output bias shape: {self.output_bias.shape}")

        # Use this layer for optimization (store as instance variable for HDF5 serialization)
        # Create or recreate optimizer using factory method (see INTENTIONAL note above)
        # R3: capture the outgoing optimizer BEFORE the rebuild so its state can be
        # carried over.  ``output_optimizer`` is assigned nowhere else and never in
        # ``__init__``, so a bare attribute access raises on the first ``fit``.
        prior_optimizer = getattr(self, "output_optimizer", None)
        self.output_optimizer = self._create_optimizer(output_layer.parameters())
        self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Created optimizer: {type(self.output_optimizer).__name__}")
        self._adopt_prior_output_optimizer_state(prior_optimizer, self.output_optimizer)
        optimizer = self.output_optimizer
        self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Learning Rate: {self.learning_rate}, Optimizer: {type(optimizer).__name__}")
        self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Output layer initialized with weights shape: {output_layer.weight.shape}, Bias shape: {output_layer.bias.shape}")

        # CR-060: Hoist hidden output computation above the epoch loop.
        # Hidden unit weights are frozen during output training, so their outputs
        # are constant across epochs.  Computing once eliminates N*E redundant
        # forward passes through the hidden cascade.
        output_input = self._compute_hidden_outputs(x)

        # P2-1d: when the live dataset has fewer output dims than the
        # network (post-shrink-via-padding), ``self.active_output_dim``
        # slices both the prediction and the target to only the meaningful
        # columns before computing loss. The zero-padded tail columns of
        # ``y`` carry no real signal and would otherwise train the network
        # to predict zero on those dead dims (bleeding into the active dims
        # via shared gradients). Defaults to ``self.output_size`` at
        # construction (no masking); set by the lifecycle's padding helper
        # when a dataset smaller than the network is installed.
        active_o = int(getattr(self, "active_output_dim", self.output_size) or self.output_size)
        active_o = min(active_o, self.output_size)

        # Output Layer Training loop
        for epoch in range(epochs):
            output = output_layer(output_input)
            self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Output shape: {output.shape}, Output Input shape: {output_input.shape}")
            self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Output: shape={output.shape}, dtype={output.dtype}")
            self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Target shape: {y.shape}, dtype={y.dtype}")
            loss = criterion(output[:, :active_o], y[:, :active_o])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self._network_display_progress(epoch):
                self.logger.info(f"CascadeCorrelationNetwork: train_output_layer: Output Layer Training - Epoch {epoch + 1}, Loss: {loss.item():.6f}")
            self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Output Layer Training - Epoch {epoch + 1}, Loss: {loss.item():.6f}")

            # Throttled callback for real-time metrics emission
            _cb = on_epoch_callback or getattr(self, "_output_epoch_callback", None)
            if _cb is not None and (epoch % 25 == 0 or epoch == epochs - 1):
                _cb(epoch=epoch + 1, epochs=epochs, loss=loss.item())

        # Update our model's weights with the trained values
        with torch.no_grad():
            self.output_weights = output_layer.weight.t().clone()  # Transpose back
            self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Output weights shape: {self.output_weights.shape}")
            self.output_bias = output_layer.bias.clone()
            self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Output bias shape: {self.output_bias.shape}")

        # Final loss
        with torch.no_grad():
            output = self.forward(x)
            self.logger.debug(f"CascadeCorrelationNetwork: train_output_layer: Final output shape: {output.shape}")
            final_loss = criterion(output[:, :active_o], y[:, :active_o]).item()
            self.logger.info(f"CascadeCorrelationNetwork: train_output_layer: Final output layer training loss: {final_loss:.6f}")
        try:
            if (snapshot_path := self.create_snapshot()) is not None:
                self.logger.info(f"CascadeCorrelationNetwork: train_output_layer: Created network snapshot at: {snapshot_path}")
                self.snapshot_counter += 1
        except Exception as snap_err:
            self.logger.warning(f"CascadeCorrelationNetwork: train_output_layer: Snapshot creation failed (non-fatal): {snap_err}")
        self.logger.trace("CascadeCorrelationNetwork: train_output_layer: Completed training of the output layer.")
        return final_loss

    ##################################################################################################################################################################################################
    # Public Method to update candidate units based on the residual error
    def train_candidates(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        residual_error: torch.Tensor,
    ) -> TrainingResults:
        """
        Train a pool of candidate units based on the residual error from the network.
        Args:
            x: Input tensor
            y: Target tensor
            residual_error: Residual error from the network
        Returns:
            Tuple containing (candidates_list, best_candidate_data, statistics)
        """
        self.logger.trace("CascadeCorrelationNetwork: train_candidates: Starting training of candidate units.")
        start_time = datetime.datetime.now()
        self.logger.verbose(f"CascadeCorrelationNetwork: train_candidates: Start time: {start_time}")

        # Step 1: Prepare candidate input incorporating existing hidden units
        candidate_input = self._prepare_candidate_input(x)
        self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Prepared candidate input shape: {candidate_input.shape}")

        # Step 2: Generate candidate training data and tasks
        tasks = self._generate_candidate_tasks(candidate_input, y, residual_error)
        self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Generated {len(tasks)} candidate training tasks.")

        # Step 3: Determine optimal process count for training
        process_count = self._calculate_optimal_process_count()
        self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Optimal process count for training: {process_count}")

        # Step 4: Execute training (parallel or sequential)
        self.logger.trace("CascadeCorrelationNetwork: train_candidates: Starting candidate training execution.")
        try:
            self.logger.info(f"CascadeCorrelationNetwork: train_candidates: Executing candidate training with {process_count} processes.")
            results = self._execute_candidate_training(
                tasks,
                process_count,
                candidate_input=candidate_input,
                y=y,
                residual_error=residual_error,
            )
            self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidate training results: length: {len(results)}, value: {results}")
        except CandidateTrainingError:
            # BUG-CC-18 / ROBUST-01: propagate the specific candidate-training failure
            # unchanged so callers can distinguish it from generic training errors and
            # apply appropriate handling (abort, fail the run, surface via API, etc.).
            import traceback

            self.logger.error("CascadeCorrelationNetwork: train_candidates: Candidate training failed irrecoverably; aborting network growth")
            self.logger.error(traceback.format_exc())
            raise
        except Exception as e:
            self.logger.error(f"CascadeCorrelationNetwork: train_candidates: Error during candidate training: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise TrainingError(f"Error during candidate training: {e}") from e
        self.logger.trace(f"CascadeCorrelationNetwork: train_candidates: Completed training of candidate units: Results: {results}.")

        # Step 5: Process and analyze results
        self.logger.trace("CascadeCorrelationNetwork: train_candidates: Starting processing of candidate training results.")
        training_stats = self._process_training_results(results, tasks, start_time)
        self.logger.trace(f"CascadeCorrelationNetwork: train_candidates: Completed processing of candidate training results: {training_stats}.")
        return training_stats

    ##################################################################################################################################################################################################
    # Define private helper methods for candidate training
    def _prepare_candidate_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare input for candidate units by incorporating outputs from existing hidden units.
        Args:
            x: Original input tensor
        Returns:
            Enhanced input tensor including hidden unit outputs
        """
        # OPT-4: Reuse cached candidate input from forward() if available and valid.
        # The cache is set by forward() during calculate_residual_error(), which runs
        # immediately before train_candidates() in grow_network()'s epoch loop.
        cached = self._cached_candidate_input
        if cached is not None and cached[0] == x.data_ptr():
            self._cached_candidate_input = None
            candidate_input = cached[1]
            self.logger.debug(f"CascadeCorrelationNetwork: _prepare_candidate_input: Using cached candidate input, shape: {candidate_input.shape}")
            self.logger.info(f"CascadeCorrelationNetwork: _prepare_candidate_input: Hidden units: {len(self.hidden_units)}")
            return candidate_input
        self._cached_candidate_input = None

        # OPT-1: Pre-allocated buffer (fallback when OPT-4 cache misses)
        candidate_input = self._compute_hidden_outputs(x)
        self.logger.debug(f"CascadeCorrelationNetwork: _prepare_candidate_input: Candidate input shape: {candidate_input.shape}")
        self.logger.info(f"CascadeCorrelationNetwork: _prepare_candidate_input: Hidden units: {len(self.hidden_units)}")
        return candidate_input

    def _generate_candidate_tasks(
        self,
        candidate_input: torch.Tensor,
        y: torch.Tensor,
        residual_error: torch.Tensor,
    ) -> list:
        """
        Generate training tasks for the candidate pool.
        Args:
            candidate_input: Enhanced input tensor
            y: Target tensor
            residual_error: Residual error tensor
        Returns:
            List of training tasks
        """
        input_size = candidate_input.shape[1]

        # OPT-5: Create shared memory block for training tensors (lightweight tasks)
        shm = None
        try:
            shm = SharedTrainingMemory(
                tensors=[candidate_input, y, residual_error],
                name_suffix=str(uuid.uuid4())[:8],
            )
            self._active_shm_blocks.append(shm)
            shm_metadata = shm.get_metadata()
            shm_metadata["candidate_epochs"] = self.candidate_epochs
            shm_metadata["candidate_learning_rate"] = self.candidate_learning_rate
            shm_metadata["candidate_display_frequency"] = self.candidate_display_frequency
            # CASCOR-505: the candidate early-stopping controls are network attributes set from
            # the API/config, but the pool workers construct their own CandidateUnit and cannot
            # see `self`. Carry them in the task payload or every candidate silently runs the
            # CandidateUnit module defaults.
            shm_metadata["candidate_patience"] = self.candidate_patience
            shm_metadata["candidate_convergence_threshold"] = self.candidate_convergence_threshold
            training_inputs = shm_metadata
            self.logger.debug(f"CascadeCorrelationNetwork: _generate_candidate_tasks: OPT-5 SharedMemory block created: {shm.name}")
        except Exception as shm_err:
            if shm is not None:
                if shm in self._active_shm_blocks:
                    self._active_shm_blocks.remove(shm)
                try:
                    shm.close_and_unlink()
                except Exception:  # nosec B110 — cleanup must not propagate exceptions
                    pass
            self.logger.warning(f"CascadeCorrelationNetwork: _generate_candidate_tasks: OPT-5 SharedMemory creation failed, falling back to full tasks: {shm_err}")
            training_inputs = (
                candidate_input,
                self.candidate_epochs,
                y,
                residual_error,
                self.candidate_learning_rate,
                self.candidate_display_frequency,
                # CASCOR-505: appended, not inserted, so a 6-element tuple built by an older
                # caller still unpacks (see the length-guarded unpack in _build_candidate_inputs).
                self.candidate_patience,
                self.candidate_convergence_threshold,
            )

        # Generate candidate metadata
        candidate_uuids = [str(uuid.uuid4()) for _ in range(self.candidate_pool_size)]
        # Candidate seeds come from a network-OWNED generator, not from the process-global `random`
        # stream.
        #
        # WHY. `random.randint` draws from the interpreter-wide stream, so a candidate's seed is a
        # function of how many times ANYTHING in the process has drawn from that stream before this
        # point -- not of the configured seed. Measured consequence: on an identical cell with
        # `network_seed=42` on both paths, the direct CLI's round-0 seed list begins at the
        # service's FOURTH element and stays three draws offset for the whole run, because
        # `SpiralProblem` (CLI-only) consumes three extra values first. The two entry points
        # therefore train different candidates on identical configuration, and part of the measured
        # CLI-vs-service work difference is that rather than any property of either path.
        #
        # It is also silently fragile: adding one `random` call anywhere -- a retry, a shuffle, a
        # log line that samples -- re-seeds every candidate in every later round. Nothing fails; the
        # numbers just change.
        #
        # A dedicated `random.Random` seeded from the network seed makes the k-th round's seeds a
        # function of (random_seed, k) alone. Created lazily so that a network restored from a
        # snapshot, or one whose seed is set after construction, still picks it up; it is a plain
        # picklable object, so __getstate__ carries its position and a resumed run continues the
        # sequence rather than replaying round 0's seeds.
        if getattr(self, "_candidate_seed_rng", None) is None:
            self._candidate_seed_rng = random.Random(self.random_seed)  # nosec B311 -- reproducibility, not secrecy  # trunk-ignore(bandit/B311)
            self.logger.debug(f"CascadeCorrelationNetwork: _generate_candidate_tasks: candidate-seed generator initialised from random_seed={self.random_seed}")
        candidate_seeds = [self._candidate_seed_rng.randint(0, self.random_max_value) for _ in range(self.candidate_pool_size)]
        candidate_data = [
            (
                i,
                input_size,
                self.activation_function_name,
                self.random_value_scale,
                candidate_uuids[i],
                candidate_seeds[i],
                self.random_max_value,
                self.sequence_max_value,
            )
            for i in range(self.candidate_pool_size)
        ]

        tasks = [(i, candidate_data[i], training_inputs) for i in range(self.candidate_pool_size)]
        self.logger.debug(f"CascadeCorrelationNetwork: _generate_candidate_tasks: Created {len(tasks)} training tasks")
        return tasks

    def _calculate_optimal_process_count(self) -> int:
        """
        Calculate the optimal number of processes for candidate training.
        Returns:
            Optimal process count
        """
        # Allow environment override (useful for testing and CI)
        env_override = os.environ.get("CASCOR_NUM_PROCESSES")
        if env_override is not None:
            count = max(1, int(env_override))
            self.logger.debug(f"CascadeCorrelationNetwork: _calculate_optimal_process_count: Using env override CASCOR_NUM_PROCESSES={count}")
            return count

        self.logger.debug(f"CascadeCorrelationNetwork: _calculate_optimal_process_count: CPU count: {os.cpu_count()}")
        self.logger.debug(f"CascadeCorrelationNetwork: _calculate_optimal_process_count: Candidate pool size: {self.candidate_pool_size}")

        # Get available CPU cores considering affinity if available
        if hasattr(os, "sched_getaffinity"):
            affinity_cores = len(os.sched_getaffinity(0))
            self.logger.debug(f"CascadeCorrelationNetwork: _calculate_optimal_process_count: Affinity CPU count: {affinity_cores}")
        else:
            affinity_cores = os.cpu_count()

        # Calculate available cores considering various constraints
        cpu_cores_available = min(
            self.candidate_pool_size,
            affinity_cores,
            (self.candidate_training_context.cpu_count() if hasattr(self, "candidate_training_context") else os.cpu_count()),
            os.cpu_count(),
        )

        # Leave one core free to keep the system responsive
        process_count = max(1, cpu_cores_available - 1)
        self.logger.debug(f"CascadeCorrelationNetwork: _calculate_optimal_process_count: Using {process_count} processes")
        return process_count

    def _execute_candidate_training(self, tasks: list, process_count: int, *, candidate_input: torch.Tensor = None, y: torch.Tensor = None, residual_error: torch.Tensor = None) -> list:
        """
        Execute candidate training via the TaskDistributor (Phase 3).

        The TaskDistributor handles local-first scheduling: local workers fill
        first, overflow goes to remote WebSocket workers. Failed remote tasks
        are automatically retried on the local pool.

        Sequential fallback (process_count <= 1, no remote workers) is handled
        directly here since it bypasses the distributor.

        Args:
            tasks: List of training tasks
            process_count: Number of processes to use
            candidate_input: Original candidate input tensor (for sequential fallback regeneration)
            y: Original target tensor (for sequential fallback regeneration)
            residual_error: Original residual error tensor (for sequential fallback regeneration)
        Returns:
            List of training results
        """
        self.logger.info(f"CascadeCorrelationNetwork: _execute_candidate_training: Training {len(tasks)} candidates with {process_count} processes")

        results = []
        try:
            remote_available = self._task_distributor.remote_worker_count > 0

            if process_count <= 1 and not remote_available:
                # Sequential: no local pool, no remote workers
                self.logger.debug("CascadeCorrelationNetwork: _execute_candidate_training: Using sequential processing")
                results = self._execute_sequential_training(tasks)
                self.logger.debug("CascadeCorrelationNetwork: _execute_candidate_training: Completed sequential processing")
            else:
                # Use TaskDistributor for local-first scheduling with optional remote overflow
                # Build remote dispatch callable that captures tensors from the tasks
                training_inputs = tasks[0][2]
                if isinstance(training_inputs, dict):
                    # OPT-5: Reconstruct tensors from SharedMemory for remote dispatch.
                    # Clone to get independent copies — remote workers can't access local /dev/shm.
                    tensors, shm_handle = SharedTrainingMemory.reconstruct_tensors(training_inputs)
                    candidate_input, y, residual_error = tensors[0].clone(), tensors[1].clone(), tensors[2].clone()
                    shm_handle.close()
                else:
                    candidate_input, _, y, residual_error = training_inputs[0], training_inputs[1], training_inputs[2], training_inputs[3]

                def remote_fn(remote_tasks):
                    return self._dispatch_to_remote_workers(remote_tasks, candidate_input, y, residual_error)

                def local_fn(local_tasks, pc):
                    if pc <= 1:
                        return self._execute_sequential_training(local_tasks)
                    return self._execute_parallel_training(local_tasks, pc)

                timeout = getattr(self, "candidate_training_shutdown_timeout", 120.0)
                results = self._task_distributor.distribute_and_collect(
                    tasks=tasks,
                    local_capacity=max(1, process_count),
                    local_fn=local_fn,
                    remote_fn=remote_fn,
                    timeout=timeout,
                )

                if not results:
                    self.logger.warning("CascadeCorrelationNetwork: _execute_candidate_training: TaskDistributor returned no results, falling back to sequential")
                    raise RuntimeError("CascadeCorrelationNetwork: _execute_candidate_training: TaskDistributor failed to return results")
                self.logger.debug("CascadeCorrelationNetwork: _execute_candidate_training: Completed distributed processing")
        except Exception as e:
            self.logger.error(f"CascadeCorrelationNetwork: _execute_candidate_training: Error in candidate node training: {e}")
            import traceback

            self.logger.error(f"CascadeCorrelationNetwork: _execute_candidate_training: Traceback: {traceback.format_exc()}")

            # Fall back to sequential training when parallel fails
            self.logger.warning("CascadeCorrelationNetwork: _execute_candidate_training: Parallel training failed, falling back to sequential training")
            try:
                if tasks and isinstance(tasks[0][2], dict):
                    legacy_inputs = (
                        candidate_input,
                        self.candidate_epochs,
                        y,
                        residual_error,
                        self.candidate_learning_rate,
                        self.candidate_display_frequency,
                    )
                    legacy_tasks = [(t[0], t[1], legacy_inputs) for t in tasks]
                    results = self._execute_sequential_training(legacy_tasks)
                else:
                    results = self._execute_sequential_training(tasks)
                self.logger.info("CascadeCorrelationNetwork: _execute_candidate_training: Sequential training completed successfully")
            except Exception as seq_error:
                # BUG-CC-18 / ROBUST-01: Previously, both-paths-failed silently installed
                # dummy zero-correlation results via ``_get_dummy_results``. That corrupted
                # the network with meaningless candidate data while hiding the underlying
                # failure. Raise an explicit error instead so the caller can abort training.
                self.logger.error(f"CascadeCorrelationNetwork: _execute_candidate_training: Sequential training also failed: {seq_error}")
                raise CandidateTrainingError("Both parallel and sequential candidate training failed. " "Training cannot continue — network integrity would be compromised.") from seq_error
        self.logger.debug(f"CascadeCorrelationNetwork: _execute_candidate_training: Obtained {len(results)} results")

        # BUG-CC-18 / ROBUST-01: Empty results after a non-exception return means every
        # candidate silently produced nothing. Refuse to install zero-correlation dummies;
        # surface the failure so the training loop can stop rather than corrupt the model.
        if not results:
            self.logger.error("CascadeCorrelationNetwork: _execute_candidate_training: No results obtained from either parallel or sequential processing")
            raise CandidateTrainingError("Candidate training produced no results after both parallel and sequential paths. " "Refusing to install zero-correlation dummy candidates — network integrity would be compromised.")
        return results

    def _drain_stale_results(self, result_queue) -> int:
        """Drain stale results from previous rounds before submitting new tasks.

        PARALLEL-FIX (RC-5): The persistent pool (RC-4) reuses result_queue across rounds.
        If a slow worker from round N completes after collection ends, its result stays in
        the queue and contaminates round N+1, potentially returning a candidate trained with
        a stale input_size.

        Returns:
            Number of stale results drained.
        """
        from queue import Empty as _QueueEmpty

        drained_count = 0
        while True:
            try:
                stale_result = result_queue.get_nowait()
                drained_count += 1
                _rc4.emit("parent.drain.stale", candidate_id=getattr(stale_result, "candidate_id", None), round_id=getattr(stale_result, "round_id", None))
                self.logger.warning(f"CascadeCorrelationNetwork: _drain_stale_results: " f"Drained stale result from previous round: " f"candidate_id={getattr(stale_result, 'candidate_id', '?')}, " f"correlation={getattr(stale_result, 'correlation', '?')}")
            except _QueueEmpty:
                break
        if drained_count:
            self.logger.warning(f"CascadeCorrelationNetwork: _drain_stale_results: " f"Drained {drained_count} stale result(s) from persistent result queue")
        return drained_count

    def _cleanup_pending_shared_memory(self) -> None:
        """Unlink SharedMemory blocks deferred from previous training rounds.

        OPT-5: By now, all workers have finished reading from these blocks
        (results were drained above). Safe to unlink.
        """
        pending = getattr(self, "_pending_shm_unlinks", [])
        if pending:
            for shm_block in pending:
                try:
                    shm_block.unlink()
                except Exception:  # nosec B110 — cleanup must not propagate exceptions
                    pass
            self._pending_shm_unlinks = []
            self.logger.debug(f"CascadeCorrelationNetwork: _cleanup_pending_shared_memory: OPT-5 unlinked {len(pending)} deferred SharedMemory block(s)")

    def _collect_worker_results(self, workers, result_queue, tasks, sleepytime, round_id) -> list:
        """Wait for persistent workers to finish and collect training results.

        CASCOR-P0-001 FIX: Uses bounded timeout with worker liveness checks instead
        of unreliable busy-wait using empty()/qsize().

        Returns:
            List of CandidateTrainingResult objects.
        """
        # P-1 RC-4 fix (H1 — dropped qsize early-exit):
        #
        # The previous implementation polled ``result_queue.qsize()`` every
        # ``sleepytime`` seconds and broke out of a wait loop as soon as
        # qsize reached ``len(tasks)`` — then called ``_collect_training_results``
        # to actually drain the queue.
        #
        # ``multiprocessing.Queue.put()`` is non-atomic with respect to
        # ``qsize()``: the worker's ``put()`` returns as soon as the
        # pickled item lands in the queue's in-process ``_buffer`` deque,
        # but ``qsize()`` only increments after a daemon "feeder" thread
        # writes the buffered item to the underlying ``Pipe`` and bumps the
        # queue's semaphore. Under sub-second per-candidate budgets, the
        # parent's qsize-based early-exit could race ahead of in-flight
        # feeder writes, leaving ``_collect_training_results`` to time out
        # on workers whose results the buffer thread had not yet flushed
        # — yielding silent ``best_candidate=None`` (V38 Phase A.2).
        #
        # The wait loop is also redundant: ``_collect_training_results``
        # already blocks on ``result_queue.get(timeout=request_timeout)``
        # which is correctly synchronized with the queue's semaphore and
        # the feeder thread, and has its own ``queue_timeout`` deadline.
        # Skipping the wait loop forces every result-collection path
        # through that single, correctly-synchronized ``get()`` call.
        #
        # We still want to log if workers crash (so an empty result set
        # has a clearer cause). That's now done as a one-shot snapshot at
        # entry, without polling.
        alive_at_entry = [w for w in workers if w.is_alive()]
        if not alive_at_entry:
            self.logger.warning("CascadeCorrelationNetwork: _collect_worker_results: No alive workers at collect entry — results will be whatever's already on the queue.")
        _rc4.emit("parent.collect.entry", alive_workers=len(alive_at_entry), expected=len(tasks), round_id=round_id)

        # Collect results, NOTE: results is of type list of data class: [candidate_training_result, ...]
        self.logger.debug("CascadeCorrelationNetwork: _collect_worker_results: Collecting results from workers")
        results = self._collect_training_results(result_queue, len(tasks), round_id=round_id, workers=workers)
        self.logger.debug(f"CascadeCorrelationNetwork: _collect_worker_results: Collected {len(results)} results")
        # ``sleepytime`` is now only consumed by callers that pass it
        # explicitly. Suppress the unused-arg warning at call sites.
        del sleepytime
        return results

    def _execute_parallel_training(
        self,
        tasks: list,
        process_count: int = -1,
        sleepytime: float = _CASCADE_CORRELATION_NETWORK_WORKER_STANDBY_SLEEPYTIME,
    ) -> list:
        """Execute training using multiprocessing.

        PARALLEL-FIX (RC-2): Replaced BaseManager-proxied queues with direct multiprocessing.Queue.
        Manager-proxied queues route every put/get through a single-threaded server process via
        IPC sockets, creating a serial bottleneck. Direct queues use OS pipes for true concurrent
        access without a proxy intermediary.

        PARALLEL-FIX (RC-3): Shared training data (tensors) is now passed as separate args to
        _worker_loop() instead of being duplicated in every task tuple. This eliminates N-fold
        redundant serialization of identical training data through the queue.

        PARALLEL-FIX (RC-4): Uses a persistent worker pool that survives across training rounds.
        Workers are created once and reused, eliminating per-round process creation, PyTorch
        initialization, module imports, and 4-phase shutdown overhead.
        """
        self.logger.debug("CascadeCorrelationNetwork: _execute_parallel_training: Using multiprocessing")

        # Adjust process count if invalid
        process_count = (process_count, self._calculate_optimal_process_count())[process_count < 1]

        # PARALLEL-FIX (RC-2): Use direct multiprocessing.Queue instead of BaseManager-proxied queues.
        # Direct queues use OS pipes for IPC, allowing concurrent put/get without routing through
        # a single-threaded manager server process.
        # Original manager-based queue creation (RC-2 replaced):
        # self.logger.debug("CascadeCorrelationNetwork: _execute_parallel_training: Starting the manager server")
        # self._start_manager()
        # self.logger.debug("CascadeCorrelationNetwork: _execute_parallel_training: Manager server started")
        # task_queue = self._task_queue
        # result_queue = self._result_queue

        # PARALLEL-FIX (RC-4): Use persistent worker pool instead of creating/destroying workers each round.
        # Workers stay alive between rounds; shared_training_inputs=None because training data changes
        # each round (residual_error evolves). Full tasks are sent through the queue each round.
        # The RC-2 direct queue improvement still applies (no manager proxy bottleneck).
        num_workers = max(1, min(process_count, len(tasks)))
        task_queue, result_queue = self._ensure_worker_pool(num_workers, shared_training_inputs=None)
        self.logger.debug(f"CascadeCorrelationNetwork: _execute_parallel_training: Using persistent pool of {num_workers} workers with direct queues")

        results = []
        # self.logger.trace("CascadeCorrelationNetwork: _execute_parallel_training: Created task and result queues")
        self.logger.debug("CascadeCorrelationNetwork: _execute_parallel_training: Created task and result queues")
        try:
            # PARALLEL-FIX (RC-5): Drain stale results from previous rounds before submitting new tasks.
            _rc4.emit("parent.before_drain", num_tasks=len(tasks), num_workers=num_workers)
            drained = self._drain_stale_results(result_queue)
            _rc4.emit("parent.after_drain", drained=drained)

            # OPT-5: Unlink SharedMemory blocks from previous rounds.
            self._cleanup_pending_shared_memory()
            _rc4.emit("parent.after_shm_cleanup")

            # Add full tasks to the queue. With persistent workers (RC-4), shared_training_inputs
            # cannot be passed at worker startup since it changes each round (residual_error evolves).
            # The RC-2 direct queue improvement still provides significant speedup over the original
            # BaseManager-proxied queues.
            # Original RC-3 approach (lightweight tasks) commented out for persistent pool compatibility:
            # shared_training_inputs = tasks[0][2] if tasks else None
            # for task in tasks:
            #     lightweight_task = (task[0], task[1])
            #     task_queue.put(lightweight_task)
            # PARALLEL-FIX (RC-5): Tag tasks with a round_id so stale results from previous
            # rounds can be identified and filtered during result collection.
            round_id = str(uuid.uuid4())
            self.logger.debug(f"CascadeCorrelationNetwork: _execute_parallel_training: Round ID: {round_id}")
            self.logger.debug("CascadeCorrelationNetwork: _execute_parallel_training: Adding tasks to persistent pool queue")
            _rc4.emit("parent.before_task_put", round_id=round_id, num_tasks=len(tasks))
            for task in tasks:
                tagged_task = (task[0], task[1], task[2], round_id)
                task_queue.put(tagged_task)
            _rc4.emit("parent.after_task_put", round_id=round_id, num_tasks=len(tasks))
            self.logger.debug(f"CascadeCorrelationNetwork: _execute_parallel_training: Added {len(tasks)} tasks to queue")

            # PARALLEL-FIX (RC-4): Workers are already running in the persistent pool.
            # Original per-round worker spawning (RC-4 replaced):
            # workers = []
            # for i in range(num_workers):
            #     worker = self._mp_ctx.Process(...)
            #     worker.start()
            #     workers.append(worker)
            workers = self._persistent_workers
            self.logger.debug(f"CascadeCorrelationNetwork: _execute_parallel_training: Using {len(workers)} persistent workers")

            # Wait for workers to process all tasks and collect results
            _rc4.emit("parent.before_collect", round_id=round_id)
            results = self._collect_worker_results(workers, result_queue, tasks, sleepytime, round_id)
            _rc4.emit("parent.after_collect", round_id=round_id, num_results=len(results))

            # PARALLEL-FIX (RC-4): Do NOT stop workers — they persist for the next training round.
            # Original per-round worker shutdown:
            # self._stop_workers(workers, task_queue)
            # self.logger.debug("CascadeCorrelationNetwork: _execute_parallel_training: Stopped all workers")
            self.logger.debug("CascadeCorrelationNetwork: _execute_parallel_training: Workers kept alive for next round (persistent pool)")
        finally:
            # PARALLEL-FIX (RC-2): Manager server no longer needed — direct queues don't require one.
            # Original manager shutdown:
            # self.logger.trace("CascadeCorrelationNetwork: _execute_parallel_training: Stopping manager server")
            # self._stop_manager()

            # OPT-5: Close main-process SharedMemory handles for this round.
            # Only close() — do NOT unlink yet. Persistent worker processes may still
            # hold references to the shared memory block for in-flight tasks. The blocks
            # are moved to _pending_shm_unlinks and unlinked at the START of the next
            # training round (after results are drained), or during pool shutdown.
            if not hasattr(self, "_pending_shm_unlinks"):
                self._pending_shm_unlinks = []
            for shm_block in list(self._active_shm_blocks):
                try:
                    shm_block.close()
                    self._pending_shm_unlinks.append(shm_block)
                except Exception as shm_e:
                    self.logger.warning(f"CascadeCorrelationNetwork: _execute_parallel_training: OPT-5 SharedMemory close error: {shm_e}")
            self._active_shm_blocks = []

            self.logger.trace("CascadeCorrelationNetwork: _execute_parallel_training: Parallel training round complete (persistent pool, no cleanup needed)")
        return results

    def _execute_sequential_training(self, tasks: list) -> list:
        """Execute training sequentially."""
        self.logger.debug("CascadeCorrelationNetwork: _execute_sequential_training: Using sequential processing")
        results = []
        for candidate_index, task in enumerate(tasks):
            self.logger.verbose(f"CascadeCorrelationNetwork: _execute_sequential_training: Training candidate {candidate_index + 1}/{len(tasks)}")
            try:
                candidate_training_result = self.train_candidate_worker(task_data_input=task, parallel=False)
                results.append(candidate_training_result)
            except Exception as task_e:
                self.logger.error(f"CascadeCorrelationNetwork: _execute_sequential_training: Task error: {task_e}")
                results.append((task[0], task[1][4] if len(task[1]) > 4 else None, 0.0, None))
        return results

    # Maximum absolute weight/bias magnitude allowed in training results (V-4 threat model)
    _RESULT_MAX_WEIGHT_MAGNITUDE = 1000.0

    def _validate_training_result(self, result) -> bool:
        """Validate a CandidateTrainingResult from the result queue.

        Checks type, field types, bounds, and tensor integrity to detect
        corrupted or malicious results. Returns True if valid, False otherwise.
        """
        if not isinstance(result, CandidateTrainingResult):
            self.logger.error(f"SECURITY: Result queue returned unexpected type: {type(result).__name__}")
            return False
        if not isinstance(result.correlation, (int, float)):
            self.logger.error(f"SECURITY: Invalid correlation type: {type(result.correlation)}")
            return False
        if not (0.0 <= result.correlation <= 1.0):
            self.logger.warning(f"CascadeCorrelationNetwork: _validate_training_result: correlation {result.correlation} out of bounds [0, 1]")
            return False
        if result.candidate is not None and not isinstance(result.candidate, CandidateUnit):
            self.logger.error(f"SECURITY: Invalid candidate type: {type(result.candidate).__name__}")
            return False
        if result.norm_output is not None and isinstance(result.norm_output, torch.Tensor):
            if torch.isnan(result.norm_output).any() or torch.isinf(result.norm_output).any():
                self.logger.warning("CascadeCorrelationNetwork: _validate_training_result: norm_output contains NaN or Inf")
                return False
        if result.norm_error is not None and isinstance(result.norm_error, torch.Tensor):
            if torch.isnan(result.norm_error).any() or torch.isinf(result.norm_error).any():
                self.logger.warning("CascadeCorrelationNetwork: _validate_training_result: norm_error contains NaN or Inf")
                return False
        # Validate candidate weight/bias magnitude (V-4: training poisoning defense)
        if result.candidate is not None and isinstance(result.candidate, CandidateUnit):
            for param_name in ("weights", "bias"):
                param = getattr(result.candidate, param_name, None)
                if param is not None and isinstance(param, torch.Tensor):
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        self.logger.warning(f"CascadeCorrelationNetwork: _validate_training_result: candidate {param_name} contains NaN or Inf")
                        return False
                    if param.abs().max().item() > 100.0:
                        self.logger.warning(f"CascadeCorrelationNetwork: _validate_training_result: candidate {param_name} magnitude {param.abs().max().item():.2f} is elevated (threshold {self._RESULT_MAX_WEIGHT_MAGNITUDE})")
                    if param.abs().max().item() > self._RESULT_MAX_WEIGHT_MAGNITUDE:
                        self.logger.warning(f"CascadeCorrelationNetwork: _validate_training_result: candidate {param_name} magnitude {param.abs().max().item():.2f} exceeds limit {self._RESULT_MAX_WEIGHT_MAGNITUDE}")
                        return False
        return True

    def _collect_training_results(
        self,
        result_queue: Queue,
        num_tasks: int,
        # DUAL-PATH/TIMEOUT FIX (2026-05-30): ``queue_timeout`` is now the maximum
        # *inactivity* gap (seconds with no new result), reset on each result —
        # NOT a fixed total deadline. See the module constant for the rationale.
        queue_timeout: float = _CASCADE_CORRELATION_NETWORK_RESULT_COLLECTION_INACTIVITY_TIMEOUT,
        request_timeout: float = 1.0,
        round_id: Optional[str] = None,
        workers: Optional[list] = None,
    ) -> list:
        """
        Description:
            Collect results from worker processes.
            This method retrieves results from the result queue until all expected results are collected or a timeout occurs.
        Args:
            result_queue: Queue to collect results from
            num_tasks: Number of expected results
            queue_timeout: Total timeout for collecting all results
            request_timeout: Timeout for each individual get request
            round_id: Expected round identifier. Results with a mismatched round_id are discarded (RC-5).
        Raises:
            Exception: If an error occurs during result collection
        Notes:
            This method blocks until all results are collected or a timeout occurs.
        Returns:
            List of collected results
        """
        from queue import Empty

        results = []
        collected_results = 0
        stale_discarded = 0
        self.logger.debug(f"CascadeCorrelationNetwork: _collect_training_results: Collecting {num_tasks} results (round_id={round_id})")
        self.logger.debug(f"CascadeCorrelationNetwork: _collect_training_results: Timeout set to {queue_timeout} seconds")
        self.logger.debug(f"CascadeCorrelationNetwork: _collect_training_results: Result Queue: Length: {result_queue.qsize()}, Contents: {list(result_queue.queue) if hasattr(result_queue, 'queue') else 'N/A'}")
        # DUAL-PATH/TIMEOUT FIX (2026-05-30): wait on an *inactivity* deadline that
        # is reset whenever a result is collected, plus a worker-liveness
        # early-exit, instead of a fixed total wall-clock deadline. Persistent
        # workers (RC-4) stay alive for the whole round, so a multi-minute round
        # streams results in over time; the previous fixed total deadline
        # (default 60s) abandoned collection mid-round — returning ~0 results and
        # producing best_candidate=None / 0 hidden units grown. We keep waiting as
        # long as results keep arriving (or a worker is alive and could still
        # deliver one), and stop only after ``queue_timeout`` seconds with NO new
        # result, once every worker has exited with the queue drained, or on error.
        inactivity_deadline = time.time() + queue_timeout
        get_attempts = 0
        empty_count = 0
        while collected_results < num_tasks:
            # Early-exit: no live worker remains to deliver the outstanding
            # results and the queue is already drained — waiting is pointless.
            if workers is not None and not any(w.is_alive() for w in workers) and result_queue.empty():
                self.logger.warning(f"CascadeCorrelationNetwork: _collect_training_results: All workers exited with " f"{collected_results}/{num_tasks} results collected and the result queue drained; stopping.")
                break
            # Inactivity guard: give up only after a full ``queue_timeout`` gap
            # with no new result (a genuinely stalled or hung pool).
            if time.time() > inactivity_deadline:
                self.logger.warning(f"CascadeCorrelationNetwork: _collect_training_results: No new result for {queue_timeout:.0f}s " f"(inactivity timeout) with {collected_results}/{num_tasks} collected; stopping.")
                break
            try:
                get_attempts += 1
                result = result_queue.get(timeout=request_timeout)
                self.logger.debug(f"CascadeCorrelationNetwork: _collect_training_results: Retrieved result: {result}")
                if not self._validate_training_result(result):
                    self.logger.error("CascadeCorrelationNetwork: _collect_training_results: Discarding invalid training result")
                    _rc4.emit("parent.collect.invalid_result")
                    continue
                # PARALLEL-FIX (RC-5): Discard results from stale training rounds.
                result_round = getattr(result, "round_id", None)
                if round_id is not None and result_round is not None and result_round != round_id:
                    stale_discarded += 1
                    self.logger.warning(f"CascadeCorrelationNetwork: _collect_training_results: " f"Discarding stale result from round {result_round} " f"(current round: {round_id}, candidate_id={result.candidate_id})")
                    _rc4.emit("parent.collect.stale", expected=round_id, got=result_round, candidate_id=getattr(result, "candidate_id", None))
                    continue
                results.append(result)
                collected_results += 1
                # Progress — extend the inactivity window past this result.
                inactivity_deadline = time.time() + queue_timeout
                _rc4.emit("parent.collect.got", candidate_id=getattr(result, "candidate_id", None), n=collected_results, expected=num_tasks)
                self.logger.verbose(f"CascadeCorrelationNetwork: _collect_training_results: Collected {collected_results}/{num_tasks}")
            except Empty as empty_e:
                empty_count += 1
                _rc4.emit("parent.collect.empty", attempt=get_attempts, empty_count=empty_count, deadline_remaining=max(0.0, inactivity_deadline - time.time()))
                self.logger.warning(f"CascadeCorrelationNetwork: _collect_training_results: Result queue empty, continuing: {empty_e}")
                continue
            except Exception as e:
                self.logger.error(f"CascadeCorrelationNetwork: _collect_training_results: Error collecting result: {e}")
                _rc4.emit("parent.collect.exception", error=type(e).__name__)
                import traceback

                self.logger.error(traceback.format_exc())
                break
        _rc4.emit("parent.collect.done", collected=collected_results, expected=num_tasks, empty_count=empty_count, stale_discarded=stale_discarded, get_attempts=get_attempts)
        if stale_discarded:
            self.logger.warning(f"CascadeCorrelationNetwork: _collect_training_results: " f"Discarded {stale_discarded} stale result(s) from previous training rounds")
        self.logger.debug(f"CascadeCorrelationNetwork: _collect_training_results: Collected {collected_results} results")
        return results

    def _stop_workers(self, workers: list, task_queue) -> None:
        """Stop worker processes with bounded total shutdown time.

        Uses a total deadline instead of per-worker timeouts to prevent
        N workers × timeout = long stalls when workers are unresponsive.
        """
        import signal
        import time

        if not workers:
            self.logger.debug("CascadeCorrelationNetwork: _stop_workers: No workers to stop")
            return
        self.logger.info(f"CascadeCorrelationNetwork: _stop_workers: Stopping {len(workers)} worker processes")

        # Phase 1: Send sentinel values (bounded to 2s total)
        sentinel_deadline = time.time() + 2.0
        for i in range(len(workers)):
            try:
                remaining = max(0.1, sentinel_deadline - time.time())
                task_queue.put(None, timeout=remaining)
                self.logger.debug(f"CascadeCorrelationNetwork: _stop_workers: Sent sentinel to worker {i}")
            except Exception as e:
                self.logger.error(f"CascadeCorrelationNetwork: _stop_workers: Failed to send sentinel to worker {i}: {e}")
                break

        # Phase 2: Wait gracefully with bounded TOTAL timeout (not per-worker)
        terminated_count = 0
        graceful_deadline = time.time() + 5.0
        for worker in workers:
            remaining = max(0.1, graceful_deadline - time.time())
            worker.join(timeout=remaining)
            if not worker.is_alive():
                terminated_count += 1
                self.logger.debug(f"CascadeCorrelationNetwork: _stop_workers: Worker {worker.name} stopped gracefully")
            else:
                self.logger.warning(f"CascadeCorrelationNetwork: _stop_workers: Worker {worker.name} (PID {worker.pid}) did not stop gracefully")
            if time.time() >= graceful_deadline:
                self.logger.warning("CascadeCorrelationNetwork: _stop_workers: Graceful shutdown deadline reached, moving to terminate")
                break

        # Phase 3: Terminate remaining workers
        for worker in workers:
            if worker.is_alive():
                self.logger.warning(f"CascadeCorrelationNetwork: _stop_workers: Terminating worker {worker.name}")
                worker.terminate()
                worker.join(timeout=1)

                # Phase 4: Force kill if still alive
                if worker.is_alive():
                    self.logger.error(f"CascadeCorrelationNetwork: _stop_workers: Worker {worker.name} still alive, sending SIGKILL")
                    try:
                        os.kill(worker.pid, signal.SIGKILL)
                        worker.join(timeout=0.5)
                    except Exception as e:
                        self.logger.error(f"CascadeCorrelationNetwork: _stop_workers: Failed to SIGKILL worker: {e}")

        if alive_workers := [w for w in workers if w.is_alive()]:
            self.logger.error(f"CascadeCorrelationNetwork: _stop_workers: ⚠️  {len(alive_workers)} workers still alive after cleanup!")
        else:
            self.logger.info(f"CascadeCorrelationNetwork: _stop_workers: ✓ All {len(workers)} workers stopped successfully ({terminated_count} gracefully)")

    def _process_training_results(self, results: list, tasks: list, start_time) -> TrainingResults:
        """
        Process and analyze training results.

        Args:
            results: Raw training results
            tasks: Original tasks
            start_time: Training start time

        Returns:
            Processed results tuple
        """
        end_time = datetime.datetime.now()
        self.logger.info(f"CascadeCorrelationNetwork: _process_training_results: Training duration: {end_time - start_time}")

        # Process results
        if not results:
            self.logger.warning("CascadeCorrelationNetwork: _process_training_results: No results obtained")
            self.logger.warning(f"CascadeCorrelationNetwork: _process_training_results: Unable to Process empty results list.  Building dummy results: {results}")
            results = self._get_dummy_results(len(tasks))
        elif len(results) != len(tasks):
            self.logger.warning(f"CascadeCorrelationNetwork: _process_training_results: Mismatch in results count: expected {len(tasks)}, got {len(results)}")
        self.logger.trace("CascadeCorrelationNetwork: _process_training_results: Completed pre-processing of training results.")

        # # Sort and extract candidate data
        # NOTE: results is a list of CandidateTrainingResult objects
        results.sort(
            key=lambda r: (r.correlation is not None, np.abs(r.correlation)),
            reverse=True,
        )
        self.logger.debug(f"CascadeCorrelationNetwork: _process_training_results: Sorted {len(results)} results")

        # Extract candidates data from results: list of CandidateTrainingResult objects
        valid_candidates = [r.candidate_id is not None and r.candidate_uuid is not None and r.correlation is not None and r.candidate is not None for r in results]

        # Identify best candidate - results are sorted by correlation descending, so first is best
        # Note: The first result after sorting has the highest correlation, so its candidate_id is the best
        best_candidate_id = results[0].candidate_id if results else -1

        # Compile statistics
        successful_candidates = self.get_candidates_data_count(results, "correlation", lambda c: c >= self.correlation_threshold)
        success_count = self.get_candidates_data_count(results, "success", lambda s: s)
        if success_count != successful_candidates:
            self.logger.info(f"CascadeCorrelationNetwork: _process_training_results: Of {success_count} successfully trained candidates, {successful_candidates} met the correlation threshold ({self.correlation_threshold})")

        # Building TrainingResults object
        # Note: results are sorted by correlation descending, so index 0 has the best candidate
        best_result = results[0] if results else None
        training_results = TrainingResults(
            epochs_completed=self.get_candidates_data(results, "epochs_completed"),
            candidate_ids=self.get_candidates_data(results, "candidate_id"),
            candidate_uuids=self.get_candidates_data(results, "candidate_uuid"),
            correlations=self.get_candidates_data(results, "correlation"),
            candidate_objects=self.get_candidates_data(results, "candidate"),
            best_candidate_id=best_candidate_id,
            best_candidate_uuid=(getattr(best_result, "candidate_uuid", None) if best_result else None),
            best_correlation=(getattr(best_result, "correlation", 0.0) if best_result else 0.0),
            best_candidate=(getattr(best_result, "candidate", None) if best_result else None),
            success_count=success_count,
            successful_candidates=successful_candidates,
            failed_count=len(results) - successful_candidates,
            error_messages=self.get_candidates_error_messages(results, valid_candidates),
            max_correlation=(getattr(best_result, "correlation", 0.0) if best_result else 0.0),
            start_time=start_time,
            end_time=end_time,
        )
        self.logger.debug(f"CascadeCorrelationNetwork: _process_training_results: Processed results: {training_results}")
        self.logger.trace("CascadeCorrelationNetwork: _process_training_results: Completed processing of training results.")
        return training_results

    # For empty results list, create an intelligently empty TrainingResults object
    def _get_dummy_results(self, num_results: int) -> list:
        """
        Generate dummy results for failed candidate training.
        Args:
            num_results: Number of dummy results to generate
        Returns:
            List of dummy CandidateTrainingResult objects
        """
        return [
            CandidateTrainingResult(
                candidate_id=id,
                # candidate_uuid=None,
                # correlation=0.0,
                # candidate=None,
                best_corr_idx=None,
                # all_correlations=None,
                # norm_output=None,
                # norm_error=None,
                # numerator=None,
                # denominator=None,
                success=False,
                # epochs_completed=0,
                error_message="No results obtained from candidate training. Using Dummy Data.",
            )
            for id in range(num_results)
        ]

    def get_candidates_data(self, results: list, field: str) -> list:
        """
        Get candidate data from results.
        Returns:
            List of candidate data for the specified field
        """
        return [getattr(r, field) for r in results if getattr(r, field) is not None]

    def get_single_candidate_data(self, results: list, candidate_id: int, field: str, default: Any) -> Any:
        """
        Get single candidate data field from results using getattr for dataclass objects.
        Returns:
            Field value from specified result or default
        """
        # self.logger.debug(f"CascadeCorrelationNetwork: get_single_candidate_data: Retrieving field '{field}' for candidate ID {candidate_id}")
        self.logger.debug(f"CascadeCorrelationNetwork: get_single_candidate_data: Retrieving field {field!r} for candidate ID {candidate_id}")
        self.logger.debug(f"CascadeCorrelationNetwork: get_single_candidate_data: Results type: {type(results)}, length: {len(results)}, Results: {results}")
        self.logger.debug(f"CascadeCorrelationNetwork: get_single_candidate_data: Field: {field}, Default: {default}")
        self.logger.debug(f"CascadeCorrelationNetwork: get_single_candidate_data: ID: type: {type(candidate_id)}, value: {candidate_id}")

        # TODO: need to check types and handle looping through tuple
        #  B=A[0] if isinstance(A, tuple) else A if isinstance(A, int) else None
        # if B is not None and 0 <= B and B <= len(A):
        #    print(f"B is: Type: {type(B)}, Value: {B}, A is: Type {type(A)}, Value: {A}")

        candidate_id = candidate_id[0] if isinstance(candidate_id, tuple) else candidate_id if isinstance(candidate_id, int) else None
        self.logger.debug(f"CascadeCorrelationNetwork: get_single_candidate_data: Processed Candidate ID: type: {type(candidate_id)}, value: {candidate_id}")
        if candidate_id is not None and 0 <= candidate_id < len(results):
            value = getattr(results[candidate_id], field, None)
            self.logger.debug(f"CascadeCorrelationNetwork: get_single_candidate_data: Retrieved value: {value}")
            return value if value is not None else default
        self.logger.debug(f"CascadeCorrelationNetwork: get_single_candidate_data: ID {candidate_id} is out of bounds, returning default: {default}")
        return default

    # def get_candidates_data_count(self, results: list, field: str, constraint: callable) -> int:
    def get_candidates_data_count(self, results: list, field: str, constraint: Callable[..., Any]) -> int:
        """
        Get count of candidate data from results.
        Args:
            results: Raw training results
            field: Field to count
        Returns:
            Count of candidate data for the specified field
        """
        # CASCOR-P1-009 FIX: Changed from sum(getattr(r, field)...) to sum(1...) to count items, not sum values
        # OLD (buggy - summed field values instead of counting):
        # return sum( getattr(r, field) for r in results if getattr(r, field) is not None and constraint(getattr(r, field)))
        # return sum(1 for r in results if getattr(r, field) is not None and constraint(getattr(r, field)))
        return sum(bool(getattr(r, field) is not None and constraint(getattr(r, field))) for r in results)

    def get_candidates_error_messages(self, results: list, valid_candidates: list) -> dict:
        """
        Get error messages for candidates.
        Returns:
            Dictionary of candidate error messages
        """
        return {
            # key: (f'Candidate ID {r.candidate_id} (UUID: {r.candidate_uuid}): "{r.error_message}"' if r.error_message and valid_candidates[i] else (f"Candidate ID {r.candidate_id} (UUID: {r.candidate_uuid}): No error message provided" if valid_candidates[i] else (f"Candidate ID {r.candidate_id} (UUID: {r.candidate_uuid}): Invalid candidate data"))) for i, r in enumerate(results) if r.candidate_id is not None or r.candidate_uuid is not None for key in [r.candidate_id, r.candidate_uuid] if key is not None
            key: (f"Candidate ID {r.candidate_id} (UUID: {r.candidate_uuid}): {r.error_message!r}" if r.error_message and valid_candidates[i] else (f"Candidate ID {r.candidate_id} (UUID: {r.candidate_uuid}): No error message provided" if valid_candidates[i] else (f"Candidate ID {r.candidate_id} (UUID: {r.candidate_uuid}): Invalid candidate data")))
            for i, r in enumerate(results)
            if r.candidate_id is not None or r.candidate_uuid is not None
            for key in [r.candidate_id, r.candidate_uuid]
            if key is not None
        }

    def __getstate__(self):
        """Remove non-picklable items for multiprocessing."""
        state = self.__dict__.copy()
        # Remove logger and display functions (not picklable)
        state.pop("logger", None)
        state.pop("plotter", None)
        state.pop("_network_display_progress", None)
        state.pop("_status_display_progress", None)
        state.pop("_candidate_display_progress", None)
        # Remove log_config (contains loggers that cannot be pickled)
        state.pop("log_config", None)
        # Remove activation functions (local closures cannot be pickled)
        state.pop("activation_fn", None)
        state.pop("activation_fn_no_diff", None)
        # Remove locks and other non-picklable objects
        state.pop("_thread.lock", None)
        # Remove large training data (should not be in snapshot anyway)
        state.pop("_training_data", None)
        state.pop("_validation_data", None)
        # Remove multiprocessing objects (cannot be pickled)
        state.pop("_manager", None)
        state.pop("_task_queue", None)
        state.pop("_result_queue", None)
        state.pop("_mp_ctx", None)
        state.pop("candidate_training_context", None)
        # PARALLEL-FIX (RC-4): Remove persistent worker pool state (not picklable)
        state.pop("_persistent_workers", None)
        state.pop("_persistent_task_queue", None)
        state.pop("_persistent_result_queue", None)
        state.pop("_persistent_pool_size", None)
        return state

    def __setstate__(self, state):
        """Restore state and reinitialize non-picklable objects."""
        self.__dict__.update(state)
        # Reinitialize logger
        from log_config.logger.logger import Logger

        Logger.set_level(self.log_level_name if hasattr(self, "log_level_name") else "INFO")
        self.logger = Logger
        # Set log_config to None - it was removed during pickling
        self.log_config = None
        # Reinitialize activation function
        self._init_activation_function()
        # Reinitialize plotter if needed
        if not hasattr(self, "plotter"):
            from cascor_plotter.cascor_plotter import CascadeCorrelationPlotter

            self.plotter = CascadeCorrelationPlotter(logger=self.logger)
        # Reinitialize display progress functions
        from utils.utils import display_progress

        if not hasattr(self, "_network_display_progress"):
            self._network_display_progress = display_progress(display_frequency=getattr(self, "epoch_display_frequency", 10))
        if not hasattr(self, "_status_display_progress"):
            self._status_display_progress = display_progress(display_frequency=getattr(self, "status_display_frequency", 100))
        if not hasattr(self, "_candidate_display_progress"):
            self._candidate_display_progress = display_progress(display_frequency=getattr(self, "candidate_display_frequency", 10))
        # Set default values for removed data
        if not hasattr(self, "_training_data"):
            self._training_data = None
        if not hasattr(self, "_validation_data"):
            self._validation_data = None

    def _create_optimizer(self, parameters, optimizer_config=None):
        """
        Create optimizer based on configuration.

        Args:
            parameters: Model parameters to optimize
            optimizer_config: OptimizerConfig instance (uses self.config.optimizer_config if None)

        Returns:
            Configured optimizer instance
        """
        from cascade_correlation.cascade_correlation_config.cascade_correlation_config import OptimizerConfig

        config = optimizer_config or getattr(self.config, "optimizer_config", OptimizerConfig())
        optimizer_map = {
            # "Adam": lambda: optim.Adam(
            #     parameters,
            #     lr=config.learning_rate,
            #     betas=(config.beta1, config.beta2),
            #     eps=config.epsilon,
            #     weight_decay=config.weight_decay,
            #     amsgrad=getattr(config, "amsgrad", False),
            # ),
            # "SGD": lambda: optim.SGD(
            #     parameters,
            #     lr=config.learning_rate,
            #     momentum=config.momentum,
            #     weight_decay=config.weight_decay,
            # ),
            # "RMSprop": lambda: optim.RMSprop(
            #     parameters,
            #     lr=config.learning_rate,
            #     momentum=config.momentum,
            #     eps=config.epsilon,
            #     weight_decay=config.weight_decay,
            # ),
            # "AdamW": lambda: optim.AdamW(
            #     parameters,
            #     lr=config.learning_rate,
            #     betas=(config.beta1, config.beta2),
            #     eps=config.epsilon,
            #     weight_decay=config.weight_decay,
            #     amsgrad=getattr(config, "amsgrad", False),
            # ),
            "Adadelta": lambda: optim.Adadelta(  # Implements Adadelta algorithm.
                parameters,
                lr=config.learning_rate,
                rho=config.rho,
                eps=config.epsilon,
            ),
            "Adafactor": lambda: optim.Adafactor(  # Implements Adafactor algorithm.
                parameters,
                lr=config.learning_rate,
                eps=config.epsilon,
            ),
            "Adagrad": lambda: optim.Adagrad(  # Implements Adagrad algorithm.
                parameters,
                lr=config.learning_rate,
                lr_decay=config.lr_decay,
                weight_decay=config.weight_decay,
            ),
            "Adam": lambda: optim.Adam(  # Implements Adam algorithm.
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
                amsgrad=getattr(config, "amsgrad", False),
            ),
            "AdamW": lambda: optim.AdamW(  # Implements AdamW algorithm, where weight decay does not accumulate in the momentum nor variance.
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
                amsgrad=getattr(config, "amsgrad", False),
            ),
            "SparseAdam": lambda: optim.SparseAdam(  # SparseAdam implements a masked version of the Adam algorithm suitable for sparse gradients.
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
            ),
            "Adamax": lambda: optim.Adamax(  # Implements Adamax algorithm (a variant of Adam based on infinity norm).
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
            ),
            "ASGD": lambda: optim.ASGD(  # Implements Averaged Stochastic Gradient Descent.
                parameters,
                lr=config.learning_rate,
                lambd=config.lambd,
                alpha=config.alpha,
                t0=config.t0,
                weight_decay=config.weight_decay,
            ),
            "LBFGS": lambda: optim.LBFGS(  # Implements L-BFGS algorithm.
                parameters,
                lr=config.learning_rate,
                max_iter=config.max_iter,
                max_eval=config.max_eval,
                tolerance_grad=config.tolerance_grad,
                tolerance_change=config.tolerance_change,
                history_size=config.history_size,
                line_search_fn=config.line_search_fn,
            ),
            "Muon": lambda: optim.Muon(  # Implements Muon algorithm.
                parameters,
                lr=config.learning_rate,
                eps=config.epsilon,
            ),
            "NAdam": lambda: optim.NAdam(  # Implements NAdam algorithm.
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            ),
            "RAdam": lambda: optim.RAdam(  # Implements RAdam algorithm.
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            ),
            "RMSprop": lambda: optim.RMSprop(  # Implements RMSprop algorithm.
                parameters,
                lr=config.learning_rate,
                momentum=config.momentum,
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            ),
            "Rprop": lambda: optim.Rprop(  # Implements the resilient backpropagation algorithm.
                parameters,
                lr=config.learning_rate,
                etas=(config.eta_min, config.eta_max),
                step_sizes=(config.step_size_min, config.step_size_max),
            ),
            "SGD": lambda: optim.SGD(  # Implements stochastic gradient descent (optionally with momentum).
                parameters,
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            ),
        }

        if config.optimizer_type not in optimizer_map:
            # self.logger.warning(f"CascadeCorrelationNetwork: _create_optimizer: Unknown optimizer type '{config.optimizer_type}', defaulting to Adam")  # B907
            self.logger.warning(f"CascadeCorrelationNetwork: _create_optimizer: Unknown optimizer type {config.optimizer_type!r}, defaulting to Adam")
            config.optimizer_type = "Adam"

        optimizer = optimizer_map[config.optimizer_type]()
        self.logger.debug(f"CascadeCorrelationNetwork: _create_optimizer: Created {config.optimizer_type} optimizer with lr={config.learning_rate}")
        return optimizer

    def _adopt_prior_output_optimizer_state(self, prior_optimizer, optimizer) -> bool:
        """Carry a prior output optimizer's accumulated state onto a freshly built one.

        R3 (resume follow-on). ``train_output_layer`` rebuilds the output ``nn.Linear``
        and its optimizer on every call because Cascade Correlation grows the output
        layer's parameter space each time a hidden unit is installed. That is right on
        a grow pass and wrong on a resume, where the space is unchanged and the
        optimizer restored from the snapshot is precisely what the operator asked to
        continue from — Adam's moments and its step counter were being discarded, so a
        resumed run silently restarted the output layer's optimization.

        Reusing the restored optimizer OBJECT is not an option, and fails silently if
        attempted. The snapshot loader binds it to a throwaway ``nn.Linear`` that it
        builds itself (``snapshots/snapshot_serializer.py``), whose ``Parameter``\\ s are
        not the ones ``train_output_layer`` just created. Shapes are identical by
        construction, so a shape check on the object PASSES — and then
        ``loss.backward()`` populates ``.grad`` on the new layer's parameters while
        ``optimizer.step()`` iterates the old ones, whose ``.grad`` is ``None``. The
        step is a no-op, the unchanged weights are copied back, loss is still logged
        and callbacks still fire: the output layer stops training with no error
        anywhere. Torch's optimizer ``state_dict`` is POSITIONALLY indexed, so routing
        the state through it re-binds it to the new parameters instead.

        Hyperparameters are deliberately NOT adopted. ``load_state_dict`` replaces
        ``param_groups`` wholesale, which would silently revert a live
        ``PATCH /v1/training/params`` learning-rate change back to the snapshot's
        value. The current config wins for hyperparameters; the snapshot wins for
        state.

        Returns:
            True when state was adopted. False — a no-op leaving the freshly built
            optimizer untouched — in every other case: a cold start with no prior
            optimizer, a topology change that grew the parameter space, an optimizer
            type change, a prior that never stepped, or a malformed prior state.
        """
        if prior_optimizer is None or optimizer is None:
            return False

        # Compare against the optimizer just built from the CURRENT config rather
        # than against the config's ``optimizer_type`` string: ``_create_optimizer``
        # silently falls back to Adam for an unrecognized type, so comparing the
        # strings would disagree with what was actually constructed.
        if type(prior_optimizer) is not type(optimizer):
            self.logger.debug(f"CascadeCorrelationNetwork: _adopt_prior_output_optimizer_state: optimizer type changed ({type(prior_optimizer).__name__} -> {type(optimizer).__name__}); starting from fresh state")
            return False

        prior_params = [p for group in getattr(prior_optimizer, "param_groups", ()) for p in group.get("params", ())]
        new_params = [p for group in optimizer.param_groups for p in group.get("params", ())]
        if len(prior_params) != len(new_params) or any(tuple(a.shape) != tuple(b.shape) for a, b in zip(prior_params, new_params)):
            # The normal grow-pass outcome: ``input_size`` grew with the new hidden
            # unit, so the prior moments describe a parameter space that no longer
            # exists. This is the branch that keeps the golden trajectory byte-stable.
            self.logger.debug("CascadeCorrelationNetwork: _adopt_prior_output_optimizer_state: output parameter space changed; starting from fresh optimizer state")
            return False

        if not getattr(prior_optimizer, "state", None):
            self.logger.debug("CascadeCorrelationNetwork: _adopt_prior_output_optimizer_state: prior optimizer carries no state; nothing to adopt")
            return False

        # Snapshot the current config's hyperparameters before ``load_state_dict``
        # overwrites ``param_groups`` with the persisted ones.
        hyperparameters = [{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups]
        try:
            optimizer.load_state_dict(prior_optimizer.state_dict())
        except (KeyError, ValueError, TypeError) as exc:
            # Torch validates group and parameter counts before mutating anything, so
            # the freshly built optimizer is still intact here. Degrade to fresh state
            # rather than failing the training pass.
            self.logger.warning(f"CascadeCorrelationNetwork: _adopt_prior_output_optimizer_state: could not adopt prior optimizer state ({exc}); starting from fresh state")
            return False

        for group, saved in zip(optimizer.param_groups, hyperparameters):
            group.update(saved)

        self.logger.debug(f"CascadeCorrelationNetwork: _adopt_prior_output_optimizer_state: adopted {type(optimizer).__name__} state ({len(optimizer.state)} parameter entries) at lr={optimizer.param_groups[0].get('lr')}")
        return True

    @staticmethod
    def train_candidate_worker(task_data_input: tuple = None, parallel: bool = True, progress_callback=None) -> None:
        """Entry point for a candidate-pool worker, with opt-in per-worker profiling.

        The candidate workers are FORKED, so nothing that profiles the parent ever sees
        their time -- and the candidate phase is where the cost is. `main.py --profile`
        instruments the parent only, and the service has no equivalent flag at all, so
        there was otherwise no way to look inside a worker from either entry point.

        Set ``JUNIPER_CASCOR_WORKER_PROFILE=<dir>`` and every worker invocation dumps a
        ``.prof`` there, named by pid. Unset -- the default, and the shipped behaviour --
        this costs one ``os.environ.get`` and a direct call, so it cannot perturb a
        measurement run.

        USE IT FOR ATTRIBUTION, NOT FOR TIMING. cProfile's per-call overhead is large
        relative to this workload, and it flattens the very effect it would be used to
        measure: during the CLI-vs-service investigation a 9.2 ms/epoch gap measured
        1.9 ms under cProfile, and the per-call ratio fell to 0.944 -- i.e. profiling
        destroyed the difference. Use this to find WHICH functions dominate a worker; use
        ``py-spy --native`` when the question is how big a difference is. (py-spy cannot
        combine ``--native`` with ``--nonblocking``, and under ``ptrace_scope=1`` it must
        be an ancestor of the process it samples, not an attacher.)
        """
        _prof_dir = os.environ.get("JUNIPER_CASCOR_WORKER_PROFILE", "").strip()
        if not _prof_dir:
            return CascadeCorrelationNetwork._train_candidate_worker_impl(task_data_input, parallel, progress_callback)

        import cProfile  # noqa: PLC0415 -- lazy on purpose: profiling is opt-in, keep the import off the default path

        _pr = cProfile.Profile()
        _pr.enable()
        try:
            return CascadeCorrelationNetwork._train_candidate_worker_impl(task_data_input, parallel, progress_callback)
        finally:
            # `finally`, so a worker that raises still leaves its profile behind -- an
            # exception path is exactly when the timing is most worth seeing.
            _pr.disable()
            try:
                pl.Path(_prof_dir).mkdir(parents=True, exist_ok=True)
                _pr.dump_stats(f"{_prof_dir}/worker-{os.getpid()}-{uuid.uuid4().hex[:8]}.prof")
            except Exception:  # nosec B110 -- profiling must never break a training run
                pass

    @staticmethod
    def _train_candidate_worker_impl(task_data_input: tuple = None, parallel: bool = True, progress_callback=None) -> None:
        # NOTE: the log records below deliberately keep the name "train_candidate_worker".
        # Log-parsing tooling anchors on message TEXT (a line-number anchor silently parsed
        # nothing after an earlier refactor shifted this file by ~90 lines), so renaming the
        # strings to match this method would break those parsers for no benefit.
        logger = Logger
        logger.info("CascadeCorrelationNetwork: train_candidate_worker: Starting training of Candidate Units in Pool.")
        try:  # Get task data for process worker
            worker_id, worker_uuid = (mp.current_process().pid, str(uuid.uuid4())) if parallel else (0, "None")
            logger.debug(f"CascadeCorrelationNetwork: train_candidate_worker: Retrieved worker ID and UUID: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        except Exception as e:
            logger.error(f"CascadeCorrelationNetwork: train_candidate_worker: Error retrieving worker ID and UUID: {e}")
            worker_id, worker_uuid = (0, "None")
        shm_handle = None  # OPT-5: track SharedMemory handle for deferred close
        candidate_inputs = None  # Guard: prevent UnboundLocalError if _build_candidate_inputs raises
        try:
            if task_data_input is None:
                logger.error("CascadeCorrelationNetwork: train_candidate_worker: No task data input provided.")
                return (None, None, 0.0, None)
            candidate_inputs = CascadeCorrelationNetwork._build_candidate_inputs(
                task_data_input=task_data_input,
                worker_id=worker_id,
                worker_uuid=worker_uuid,
            )
            if candidate_inputs is None or not isinstance(candidate_inputs, dict) or len(candidate_inputs) == 0:
                logger.error(f"CascadeCorrelationNetwork: train_candidate_worker: No candidate inputs built: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
                return (None, None, 0.0, None)
            shm_handle = candidate_inputs.pop("_shm_handle", None)  # OPT-5: extract handle for deferred close
            logger.debug(f"CascadeCorrelationNetwork: train_candidate_worker: Built candidate inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Keys: {list(candidate_inputs.keys()) if isinstance(candidate_inputs, dict) else type(candidate_inputs)}")

            # Instantiate a CandidateUnit using factory method (Note: needs network instance for factory)
            logger.debug(f"CascadeCorrelationNetwork: train_candidate_worker: Instantiate a CandidateUnit using factory method (Note: needs network instance for factory, candidate_inputs: {candidate_inputs}): Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
            logger.debug(f"CascadeCorrelationNetwork: train_candidate_worker: Candidate Inputs Key Values: {candidate_inputs.get('candidate_display_frequency')}, Candidate Index: {candidate_inputs.get('candidate_index')}, Candidate UUID: {candidate_inputs.get('candidate_uuid')}")
            try:
                logger.debug(f"CascadeCorrelationNetwork: train_candidate_worker: Instantiating CandidateUnit Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
                # CASCOR-P0-005 FIX: Corrected parameter key names to match _build_candidate_inputs dictionary
                # OLD (incorrect keys - returned None):
                # CandidateUnit__epochs=candidate_inputs.get("epochs"),
                # CandidateUnit__learning_rate=candidate_inputs.get("learning_rate"),
                # CandidateUnit__random_seed=candidate_inputs.get("random_seed"),
                # NEW (correct keys matching _build_candidate_inputs):
                candidate = CandidateUnit(
                    CandidateUnit__activation_function=candidate_inputs.get("activation_fn"),
                    CandidateUnit__display_frequency=candidate_inputs.get("candidate_display_frequency"),
                    CandidateUnit__epochs=candidate_inputs.get("candidate_epochs"),
                    CandidateUnit__input_size=candidate_inputs.get("input_size"),
                    CandidateUnit__learning_rate=candidate_inputs.get("candidate_learning_rate"),
                    CandidateUnit__log_level_name="INFO",
                    CandidateUnit__sequence_max_value=candidate_inputs.get("sequence_max_value"),
                    CandidateUnit__random_seed=candidate_inputs.get("candidate_seed"),
                    CandidateUnit__random_max_value=candidate_inputs.get("random_max_value"),
                    CandidateUnit__random_value_scale=candidate_inputs.get("random_value_scale"),
                    CandidateUnit__uuid=candidate_inputs.get("candidate_uuid"),
                    CandidateUnit__candidate_index=candidate_inputs.get("candidate_index"),
                    # CASCOR-505: without these two the pool ran the module defaults no matter
                    # what the API applied. Defaults are repeated here because a caller that
                    # patches _build_candidate_inputs may hand back a dict without the keys, and
                    # CandidateUnit does not coerce None for either field.
                    CandidateUnit__patience=candidate_inputs.get("candidate_patience", _CANDIDATE_UNIT_PATIENCE),
                    CandidateUnit__convergence_threshold=candidate_inputs.get("candidate_convergence_threshold", _CANDIDATE_UNIT_CONVERGENCE_THRESHOLD),
                )
                logger.debug(f"CascadeCorrelationNetwork: train_candidate_worker: Completed Instantiating CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate.get_uuid()}")
            except Exception as e:
                logger.error(f"CascadeCorrelationNetwork: train_candidate_worker: Caught Exception while instantiating CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_inputs.get('candidate_index')}, Candidate UUID: {candidate_inputs.get('candidate_uuid')}, Error during candidate instantiation:\nException:\n{e}")
                import traceback

                traceback.print_exc()
                raise
            logger.verbose(f"CascadeCorrelationNetwork: train_candidate_worker: Created CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_inputs.get('candidate_index')}, Candidate UUID: {candidate.get_uuid()}, Candidate Object: {candidate}")

            # Train the candidate unit
            result = CascadeCorrelationNetwork._train_candidate_unit(
                candidate=candidate,
                candidate_uuid=candidate_inputs.get("candidate_uuid"),
                candidate_index=candidate_inputs.get("candidate_index"),
                candidate_input=candidate_inputs.get("candidate_input"),
                candidate_epochs=candidate_inputs.get("candidate_epochs"),
                residual_error=candidate_inputs.get("residual_error"),
                candidate_learning_rate=candidate_inputs.get("candidate_learning_rate"),
                candidate_display_frequency=candidate_inputs.get("candidate_display_frequency"),
                worker_id=worker_id,
                worker_uuid=worker_uuid,
                progress_callback=progress_callback,
            )
            # PARALLEL-FIX (RC-5): Tag result with round_id for cross-round contamination detection
            result.round_id = candidate_inputs.get("round_id")
            logger.info(f"CascadeCorrelationNetwork: train_candidate_worker: Returning from Candidate Unit Training: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate ID: {result.candidate_id}, Candidate UUID: {result.candidate_uuid}, Candidate Correlation: {float(result.correlation):.6f}")
            return result

        except Exception as e:
            import traceback

            logger.error(f"CascadeCorrelationNetwork: train_candidate_worker: Caught Exception while training CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Error during candidate training:\nException:\n{e}")
            logger.error(f"CascadeCorrelationNetwork: train_candidate_worker: Error during Candidate Training: Worker ID: {worker_id}, Worker UUID: {worker_uuid}\nTraceback:\n{traceback.format_exc()}")
            candidate_index = candidate_inputs.get("candidate_index") if candidate_inputs else -1
            candidate_uuid = candidate_inputs.get("candidate_uuid") if candidate_inputs else None
            return CandidateTrainingResult(
                candidate_id=candidate_index,
                candidate_uuid=candidate_uuid,
                correlation=0.0,
                candidate=None,
                success=False,
                epochs_completed=0,
                error_message=str(e),
                round_id=candidate_inputs.get("round_id") if candidate_inputs else None,
            )
        finally:
            # OPT-5: Close SharedMemory handle after training completes (or on error)
            if shm_handle is not None:
                try:
                    shm_handle.close()
                except Exception:  # nosec B110 — cleanup must not propagate exceptions
                    pass

    @staticmethod
    def _build_candidate_inputs(
        task_data_input: tuple = None,
        worker_uuid: str = None,
        worker_id: int = None,
    ):
        logger = Logger
        logger.debug(f"CascadeCorrelationNetwork: _build_candidate_inputs: Building candidate inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")

        # Unpack task data
        # TODO: consider using data classes for task data, candidate data, and training inputs
        logger.debug(f"CascadeCorrelationNetwork: _build_candidate_inputs: Attempting to Unpack Task data, Candidate data, and Training inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        logger.verbose(f"CascadeCorrelationNetwork: _build_candidate_inputs: Task data: length: {len(task_data_input)}, Type: {type(task_data_input)}")
        logger.debug(f"CascadeCorrelationNetwork: _build_candidate_inputs: Task data unpacked: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        # PARALLEL-FIX (RC-5): Support optional round_id as 4th element in task tuple.
        # Backward compatible: 3-element tuples from sequential path have round_id=None.
        if len(task_data_input) >= 4:
            candidate_index, candidate_data, training_inputs, round_id = task_data_input
        else:
            candidate_index, candidate_data, training_inputs = task_data_input
            round_id = None
        logger.debug(f"CascadeCorrelationNetwork: _build_candidate_inputs: Successfully Unpacked Task data: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        logger.verbose(f"CascadeCorrelationNetwork: _build_candidate_inputs: Candidate Index: {candidate_index}, Type: {type(candidate_index)}, Value: {candidate_index}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        logger.verbose(f"CascadeCorrelationNetwork: _build_candidate_inputs: Candidate Inputs: Length: {len(training_inputs)}, Type: {type(training_inputs)}, Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        logger.verbose(f"CascadeCorrelationNetwork: _build_candidate_inputs: Candidate Data: length: {len(candidate_data)}, Type: {type(candidate_data)}, Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        (
            input_size,
            activation_name,
            random_value_scale,
            candidate_uuid,
            candidate_seed,
            random_max_value,
            sequence_max_value,
        ) = candidate_data[1:]
        logger.debug(f"CascadeCorrelationNetwork: _build_candidate_inputs: Successfully Unpacked Candidate Data: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}.")
        logger.verbose(f"CascadeCorrelationNetwork: _build_candidate_inputs: Candidate data unpacked: Candidate ID: {id}, Input Size: {input_size}, Activation Function Name: {activation_name}, Random Value Scale: {random_value_scale}, Candidate UUID: {candidate_uuid}, Random Seed: {candidate_seed}, Random Value Max: {random_max_value}, Sequence Max Value: {sequence_max_value}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}.")
        logger.debug(f"CascadeCorrelationNetwork: _build_candidate_inputs: Attempting to unpack Training inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}")
        logger.verbose(f"CascadeCorrelationNetwork: _build_candidate_inputs: Training inputs: Type: {type(training_inputs)}, Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}")
        # OPT-5: Handle both dict (SharedMemory metadata) and tuple (legacy) formats
        shm_handle = None
        if isinstance(training_inputs, dict):
            # OPT-5: Reconstruct training tensors from SharedMemory block
            try:
                logger.debug(f"CascadeCorrelationNetwork: _build_candidate_inputs: OPT-5 reconstructing tensors from SharedMemory: {training_inputs.get('shm_name')}")
                tensors, shm_handle = SharedTrainingMemory.reconstruct_tensors(training_inputs)
                # OPT-5 FIX: Clone tensors from SharedMemory into owned, writable buffers.
                # reconstruct_tensors returns zero-copy views into the shared memory block,
                # which are read-only.  PyTorch autograd and in-place candidate training
                # operations require writable tensors.  Cloning here preserves the OPT-5
                # benefit (training data is read once from /dev/shm, not serialized N times
                # through the task queue) while giving each worker its own mutable copy.
                candidate_input, y, residual_error = tensors[0].clone(), tensors[1].clone(), tensors[2].clone()
            except OSError as shm_err:
                logger.warning(f"CascadeCorrelationNetwork: _build_candidate_inputs: OPT-5 SharedMemory unavailable ({shm_err}), re-raising for sequential fallback")
                raise  # Let caller's exception handler trigger sequential fallback with fresh tasks
            candidate_epochs = training_inputs["candidate_epochs"]
            candidate_learning_rate = training_inputs["candidate_learning_rate"]
            candidate_display_frequency = training_inputs["candidate_display_frequency"]
            # CASCOR-505: absent keys mean a payload built before this fix — fall back to the
            # CandidateUnit module defaults, which is exactly what those workers already used.
            candidate_patience = training_inputs.get("candidate_patience", _CANDIDATE_UNIT_PATIENCE)
            candidate_convergence_threshold = training_inputs.get("candidate_convergence_threshold", _CANDIDATE_UNIT_CONVERGENCE_THRESHOLD)
        elif len(training_inputs) >= 8:
            # CASCOR-505: 8-element legacy tuple carries the candidate early-stopping controls.
            (
                candidate_input,
                candidate_epochs,
                y,
                residual_error,
                candidate_learning_rate,
                candidate_display_frequency,
                candidate_patience,
                candidate_convergence_threshold,
            ) = training_inputs
        else:
            (
                candidate_input,
                candidate_epochs,
                y,
                residual_error,
                candidate_learning_rate,
                candidate_display_frequency,
            ) = training_inputs
            candidate_patience = _CANDIDATE_UNIT_PATIENCE
            candidate_convergence_threshold = _CANDIDATE_UNIT_CONVERGENCE_THRESHOLD
        logger.debug(f"CascadeCorrelationNetwork: _build_candidate_inputs: Successfully Unpacked Training inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}.")
        logger.debug(f"CascadeCorrelationNetwork: _build_candidate_inputs: Unpacked Task data, Candidate data, and Training inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Training Inputs: x shape: {candidate_input.shape}, epochs: {candidate_epochs}, y shape: {y.shape}, residual_error shape: {residual_error.shape}, learning_rate: {candidate_learning_rate}, display_frequency: {candidate_display_frequency}")
        logger.verbose(f"CascadeCorrelationNetwork: _build_candidate_inputs: Training inputs: x shape: {candidate_input.shape}, epochs: {candidate_epochs}, y shape: {y.shape}, residual_error shape: {residual_error.shape}, learning_rate: {candidate_learning_rate}, display_frequency: {candidate_display_frequency}")
        activation_fn = CascadeCorrelationNetwork._get_activation_function(activation_name)
        logger.debug(f"CascadeCorrelationNetwork: _build_candidate_inputs: Retrieved wrapped activation function: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Activation Function: Name: {activation_name}, Function: {activation_fn}")

        # TODO: reference data values from input tuples?
        # Build candidate inputs dictionary
        candidate_inputs = {
            "task_data_input": task_data_input,
            "candidate_index": candidate_index,
            "candidate_data": candidate_data,
            "training_inputs": training_inputs,
            "input_size": input_size,
            "activation_name": activation_name,
            "random_value_scale": random_value_scale,
            "candidate_uuid": candidate_uuid,
            "candidate_seed": candidate_seed,
            "random_max_value": random_max_value,
            "sequence_max_value": sequence_max_value,
            "candidate_input": candidate_input,
            "candidate_epochs": candidate_epochs,
            "y": y,
            "residual_error": residual_error,
            "candidate_learning_rate": candidate_learning_rate,
            "candidate_display_frequency": candidate_display_frequency,
            "candidate_patience": candidate_patience,  # CASCOR-505
            "candidate_convergence_threshold": candidate_convergence_threshold,  # CASCOR-505
            "activation_fn": activation_fn,
            "round_id": round_id,
            "_shm_handle": shm_handle,  # OPT-5: SharedMemory handle for deferred close (None if legacy path)
        }
        logger.debug(f"CascadeCorrelationNetwork: _build_candidate_inputs: Successfully built candidate inputs: {len(candidate_inputs)} keys")
        return candidate_inputs

    @staticmethod
    def _train_candidate_unit(
        candidate: CandidateUnit = None,
        # candidate_uuid: uuid = None,  # Original - invalid type (uuid is module, not type)
        candidate_uuid: uuid.UUID = None,
        candidate_index: int = 0,
        candidate_input: tuple = None,
        candidate_epochs: int = 0,
        residual_error: float = 0.0,
        candidate_learning_rate: float = 0.0,
        candidate_display_frequency: int = 0,
        worker_id: int = 0,
        worker_uuid: str = "None",
        progress_callback=None,
    ) -> CandidateTrainingResult:
        # Train the candidate unit
        logger = Logger

        try:
            logger.debug(f"CascadeCorrelationNetwork: _train_candidate_unit: Training CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate.get_uuid()}, Candidate Object: {candidate}")
            # training_result = candidate.train(
            training_result = candidate.train_detailed(
                x=candidate_input,
                epochs=candidate_epochs,
                residual_error=residual_error,
                learning_rate=candidate_learning_rate,
                display_frequency=candidate_display_frequency,
                progress_callback=progress_callback,
            )
            logger.info(f"CascadeCorrelationNetwork: _train_candidate_unit: Completed Training CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Correlation: {float(training_result.correlation):.6f}")
            logger.debug(f"CascadeCorrelationNetwork: _train_candidate_unit: Clearing Display Progress and Display Status for Candidate Unit: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}")
            candidate.clear_display_progress()  # Clear display progress for candidate unit, to avoid issues with multiprocessing--nested functions are not pickleable
            logger.debug(f"CascadeCorrelationNetwork: _train_candidate_unit: Cleared Display Progress for Candidate Unit: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}")
            candidate.clear_display_status()  # Clear display status for candidate unit, to avoid issues with multiprocessing--nested functions are not pickleable
            logger.debug(f"CascadeCorrelationNetwork: _train_candidate_unit: Cleared Display Status for Candidate Unit: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}")

            # Return CandidateTrainingResult with updated values
            training_result.candidate_id = candidate_index
            training_result.candidate_uuid = candidate_uuid
            training_result.candidate = candidate
            return training_result
        except Exception as e:
            logger.error(f"CascadeCorrelationNetwork: _train_candidate_unit: Caught Exception while training CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Error during candidate training:\nException:\n{e}")
            import traceback

            traceback.print_exc()
            return CandidateTrainingResult(
                candidate_id=candidate_index if "candidate_index" in locals() else -1,
                candidate_uuid=candidate_uuid if "candidate_uuid" in locals() else None,
                correlation=0.0,
                candidate=None,
                success=False,
                epochs_completed=0,
                error_message=str(e),
            )

    @staticmethod
    # def _get_activation_function(activation_function_name: str = None, activation_functions_dict: dict = None) -> callable:  # Original - invalid type
    def _get_activation_function(activation_function_name: str = None, activation_functions_dict: dict = None) -> Callable[..., Any]:
        """
        Description:
            Get the activation function based on its name.
        Args:
            activation_function_name: Name of the activation function
            activation_functions_dict: Dictionary of available activation functions
        Note:
            This method retrieves the activation function from the provided dictionary based on its name.
        Returns:
            Activation function
        """
        if activation_functions_dict is None:
            activation_functions_dict = _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT
        if activation_function_name is None:
            activation_function_name = _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME
        return activation_functions_dict.get(
            activation_function_name,
            _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,
        )

    #################################################################################################################################################################################################
    # Multiprocessing Manager methods
    #################################################################################################################################################################################################
    def _start_manager(self):
        """Start the multiprocessing manager server in non-blocking mode."""
        self.logger.debug("CascadeCorrelationNetwork: _start_manager: Starting multiprocessing manager")
        if self._manager is not None:
            self.logger.warning("CascadeCorrelationNetwork: _start_manager: Manager already started")
            return
        address = self.candidate_training_queue_address
        authkey = self.candidate_training_queue_authkey
        if isinstance(authkey, str):
            authkey = authkey.encode("utf-8")
        try:
            self._manager = CandidateTrainingManager(address=address, authkey=authkey, ctx=self._mp_ctx)
            self._manager.start()  # Non-blocking - server runs in background

            # Obtain queue proxies
            self._task_queue = self._manager.get_task_queue()
            self._result_queue = self._manager.get_result_queue()
            self.logger.info(f"CascadeCorrelationNetwork: _start_manager: Manager started at {address}")
        except Exception as e:
            self.logger.error(f"CascadeCorrelationNetwork: _start_manager: Failed to start manager: {e}")
            raise

    def _stop_manager(self):
        """Stop the multiprocessing manager server."""
        self.logger.debug("CascadeCorrelationNetwork: _stop_manager: Stopping multiprocessing manager")
        if self._manager is not None:
            try:
                self._manager.shutdown()
                self.logger.info("CascadeCorrelationNetwork: _stop_manager: Manager shutdown completed")
            except Exception as e:
                self.logger.error(f"CascadeCorrelationNetwork: _stop_manager: Error shutting down manager: {e}")
            finally:
                self._manager = None
                self._task_queue = None
                self._result_queue = None

    #################################################################################################################################################################################################
    # PARALLEL-FIX (RC-4): Persistent worker pool management methods
    #################################################################################################################################################################################################
    def _ensure_worker_pool(self, num_workers: int, shared_training_inputs: tuple = None) -> tuple:
        """
        Description:
            Ensure a persistent worker pool of the requested size is running.
            If workers already exist and are alive, reuse them. If the pool size changed
            or workers died, shut down and recreate.
        Args:
            num_workers: Desired number of worker processes
            shared_training_inputs: Shared training data to pass to new workers
        Returns:
            Tuple of (task_queue, result_queue) for submitting work to the pool
        Notes:
            PARALLEL-FIX (RC-4): Workers are created once and reused across training rounds,
            eliminating per-round overhead of process creation, PyTorch initialization,
            module imports, and 4-phase shutdown.
        """
        # Check if existing pool is valid (right size, all workers alive)
        alive_count = sum(1 for w in self._persistent_workers if w.is_alive()) if self._persistent_workers else 0
        # DUAL-PATH FIX (2026-05-30): reuse a healthy persistent pool whenever it
        # already has at least as many live workers as requested. A larger pool
        # serves a smaller request (e.g. a remote-fallback retry batch of 2 tasks)
        # — surplus workers simply idle on the empty task queue. The previous
        # exact-size check (``_persistent_pool_size == num_workers``) tore the pool
        # down to resize it, which SIGKILLed in-flight workers and orphaned their
        # result queue mid-round — the root cause of "expected N, got 2" /
        # 0 hidden units grown under dual-path dispatch. Only rebuild when no
        # healthy pool exists or MORE workers are needed.
        pool_healthy = bool(self._persistent_workers) and self._persistent_task_queue is not None and self._persistent_result_queue is not None and alive_count == self._persistent_pool_size
        pool_valid = pool_healthy and num_workers <= self._persistent_pool_size

        if pool_valid:
            self.logger.debug(f"CascadeCorrelationNetwork: _ensure_worker_pool: Reusing existing pool of {alive_count} workers")
            return self._persistent_task_queue, self._persistent_result_queue

        # Pool is invalid or needs resizing — shut down existing and create new
        if self._persistent_workers:
            self.logger.debug(f"CascadeCorrelationNetwork: _ensure_worker_pool: Existing pool invalid (alive={alive_count}, expected={self._persistent_pool_size}), recreating")
            self._shutdown_worker_pool()

        # Create fresh queues and workers
        self._persistent_task_queue = self._mp_ctx.Queue(maxsize=_QUEUE_MAXSIZE)
        self._persistent_result_queue = self._mp_ctx.Queue(maxsize=_QUEUE_MAXSIZE)
        self._persistent_progress_queue = self._mp_ctx.Queue(maxsize=_QUEUE_MAXSIZE)
        self._persistent_pool_size = num_workers
        _worker_thread_count = getattr(self.config, "worker_thread_count", 1)

        # P-1 RC-4: lazily create the cross-process instrumentation queue
        # so workers can post events back to the parent. None when the
        # ring buffer is disabled — _worker_loop's set_worker_queue is
        # a no-op in that case.
        instrumentation_queue = _rc4.init_parent_queue(self._mp_ctx)

        self.logger.debug(f"CascadeCorrelationNetwork: _ensure_worker_pool: Creating persistent pool of {num_workers} workers")
        for i in range(num_workers):
            worker = self._mp_ctx.Process(
                target=CascadeCorrelationNetwork._worker_loop,
                args=(
                    self._persistent_task_queue,
                    self._persistent_result_queue,
                    True,
                    _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT,
                    _worker_thread_count,
                    shared_training_inputs,
                    self._persistent_progress_queue,
                    instrumentation_queue,
                ),
                daemon=True,
                name=f"CandidateWorker-{i}",
            )
            worker.start()
            self.logger.debug(f"CascadeCorrelationNetwork: _ensure_worker_pool: Started persistent worker {i} with PID {worker.pid}")
            self._persistent_workers.append(worker)

        self.logger.info(f"CascadeCorrelationNetwork: _ensure_worker_pool: Persistent pool created with {num_workers} workers")
        return self._persistent_task_queue, self._persistent_result_queue

    def _send_shutdown_sentinels(self) -> None:
        """Send None sentinels to the task queue to signal workers to exit.

        Each worker checks for None as a shutdown signal in its main loop.
        """
        if self._persistent_task_queue is not None:
            for i in range(len(self._persistent_workers)):
                try:
                    self._persistent_task_queue.put(None, timeout=2.0)
                except Exception as e:
                    self.logger.warning(f"CascadeCorrelationNetwork: _send_shutdown_sentinels: Failed to send sentinel {i}: {e}")

    def _terminate_workers(self) -> None:
        """Join workers with timeout, escalating to terminate and SIGKILL if needed."""
        import signal

        for worker in self._persistent_workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                self.logger.warning(f"CascadeCorrelationNetwork: _terminate_workers: Worker {worker.name} did not stop gracefully, terminating")
                worker.terminate()
                worker.join(timeout=1.0)
                if worker.is_alive():
                    try:
                        os.kill(worker.pid, signal.SIGKILL)
                        worker.join(timeout=0.5)
                    except Exception:  # nosec B110 — cleanup must not propagate exceptions
                        pass

    def _cleanup_shared_memory_blocks(self) -> None:
        """Clean up any outstanding SharedMemory blocks (active + pending unlink).

        OPT-5: Ensures all SharedMemory blocks are properly closed and unlinked
        during pool shutdown.
        """
        for shm_block in list(getattr(self, "_active_shm_blocks", [])):
            try:
                shm_block.close_and_unlink()
            except Exception:  # nosec B110 — cleanup must not propagate exceptions
                pass
        if hasattr(self, "_active_shm_blocks"):
            self._active_shm_blocks = []
        for shm_block in list(getattr(self, "_pending_shm_unlinks", [])):
            try:
                shm_block.unlink()
            except Exception:  # nosec B110 — cleanup must not propagate exceptions
                pass
        if hasattr(self, "_pending_shm_unlinks"):
            self._pending_shm_unlinks = []

    def _shutdown_worker_pool(self) -> None:
        """
        Description:
            Shut down the persistent worker pool by sending sentinels and joining workers.
        Notes:
            PARALLEL-FIX (RC-4): Only called when the pool needs to be recreated (size change,
            dead workers) or when the network is being serialized/destroyed. Normal training
            rounds do NOT call this — workers persist across rounds.
        """
        if not self._persistent_workers:
            return

        self.logger.debug(f"CascadeCorrelationNetwork: _shutdown_worker_pool: Shutting down {len(self._persistent_workers)} persistent workers")

        # Send sentinels to tell workers to exit
        self._send_shutdown_sentinels()

        # Join workers with timeout, escalating to terminate/SIGKILL if needed
        self._terminate_workers()

        self._persistent_workers = []
        self._persistent_task_queue = None
        self._persistent_result_queue = None
        self._persistent_progress_queue = None
        self._persistent_pool_size = 0

        # OPT-5: Clean up any outstanding SharedMemory blocks (active + pending unlink)
        self._cleanup_shared_memory_blocks()

        self.logger.debug("CascadeCorrelationNetwork: _shutdown_worker_pool: Persistent pool shut down")

    def _release_candidate_worker_pool(self) -> None:
        """Issue #509: end the RC-4 pool's persistence when the *run* ends.

        RC-4 deliberately keeps candidate workers alive **across growth rounds** —
        that is the optimization, and it is preserved. What was missing is the other
        end of that lifetime: nothing ever released the pool when the run finished.
        ``_shutdown_worker_pool``'s own docstring claims it runs "when the network is
        being serialized/destroyed", but its only production caller was
        ``_ensure_worker_pool`` recycling a stale pool, so a healthy pool simply
        outlived every run that created it.

        The consequence is not a tidy-up nit. The orphaned forkserver children keep
        their CUDA context (~116 MiB each) and reparent to ``systemd --user``; the
        forkserver's parent-death detection uses a pipe heartbeat that can take many
        minutes to fire, so they accumulate at roughly 285 MiB per experiment cell
        until the card is full. Once it is, ``CandidateUnit.__init__`` dies with
        ``AcceleratorError: CUDA error: out of memory`` for every candidate, and the
        run reports a plausible-looking result computed from nothing — degradation
        that tracks position in a campaign rather than any experimental variable.

        Cleanup must never mask the outcome of the work it follows, so every failure
        here is logged and swallowed: this runs from a ``finally`` in ``fit`` where a
        training exception may already be in flight, and from ``atexit``.
        """
        try:
            self._shutdown_worker_pool()
        except Exception:  # nosec B110 — cleanup must not propagate exceptions
            self.logger.warning("CascadeCorrelationNetwork: _release_candidate_worker_pool: Failed to release the candidate worker pool; forkserver children may survive this run", exc_info=True)

    def _cleanup_shared_memory(self):
        """Emergency cleanup of SharedMemory blocks on process exit (OPT-5)."""
        for shm in list(getattr(self, "_active_shm_blocks", [])):
            try:
                shm.close_and_unlink()
            except Exception:  # nosec B110 — cleanup must not propagate exceptions
                pass
        if hasattr(self, "_active_shm_blocks"):
            self._active_shm_blocks = []
        for shm in list(getattr(self, "_pending_shm_unlinks", [])):
            try:
                shm.unlink()
            except Exception:  # nosec B110 — cleanup must not propagate exceptions
                pass
        if hasattr(self, "_pending_shm_unlinks"):
            self._pending_shm_unlinks = []

    # TODO: maybe break this up
    @staticmethod
    def _worker_loop(
        task_queue: Queue,
        result_queue: Queue,
        parallel: bool = True,
        task_queue_timeout: float = _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT,
        # PARALLEL-FIX (RC-1): Configurable thread count per worker, default 1 to prevent oversubscription
        worker_thread_count: int = 1,
        # PARALLEL-FIX (RC-3): Shared training data passed once at worker startup instead of
        # being duplicated in every task through the queue. None for backward compatibility
        # with sequential training path or legacy callers.
        shared_training_inputs: tuple = None,
        progress_queue: Queue = None,
        # P-1 RC-4: optional cross-process instrumentation queue. None when the
        # ring buffer is disabled (default); set by ``_ensure_worker_pool``
        # via ``rc4_ring_buffer.init_parent_queue`` when CASCOR_RC4_RING_BUFFER=1.
        instrumentation_queue=None,
    ):
        """
        Description:
            Worker process loop that processes tasks with stand-by mode.
        Args:
            task_queue: Queue to get tasks from
            result_queue: Queue to put results into
            parallel: Whether running in parallel mode
            task_queue_timeout: Timeout for getting tasks from queue
            worker_thread_count: Number of PyTorch internal threads per worker (default 1)
            shared_training_inputs: Shared training data tuple (candidate_input, epochs, y,
                residual_error, learning_rate, display_frequency) passed once to avoid
                N-fold redundant serialization through the task queue
        Raises:
            TrainingError: If an error occurs during task processing
        Notes:
            - This function runs in a separate process and continuously checks for new tasks.
            - If no tasks are available, it enters a stand-by mode to save resources.
            - PARALLEL-FIX (RC-1): Each worker pins its PyTorch thread count to prevent
              N_workers * M_threads CPU oversubscription that serializes parallel execution.
            - PARALLEL-FIX (RC-3): Shared training data is received once at startup rather than
              being serialized/deserialized with every task through the queue.
        Returns:
            None
        """
        logger = Logger
        from queue import Empty

        # PARALLEL-FIX (RC-1): pin the PyTorch ATen thread pool per worker. Without this each
        # worker defaults to using ALL CPU cores for BLAS/autograd threads, causing
        # N_workers * M_threads oversubscription that effectively serializes execution. THIS is
        # the oversubscription guard, it works, and it runs identically on both entry points.
        #
        # The three environment variables are deliberately NOT set here (#531). They are read once
        # when BLAS first loads -- which, in a forkserver-created worker, happened in an ancestor
        # process long before this line -- so assigning them here could never resize this worker's
        # pool. It only made the code read as though the worker controlled its own budget, which
        # is what delayed finding the real, entry-point-level asymmetry. The pool size is settled
        # by parallelism.blas_threads at entry-point import time; see that module.
        import torch as _torch

        _thread_count_str = str(max(1, worker_thread_count))
        _torch.set_num_threads(max(1, worker_thread_count))
        logger.debug(f"CascadeCorrelationNetwork: _worker_loop: PyTorch thread count pinned to {_thread_count_str} for worker process isolation")

        logger.debug("CascadeCorrelationNetwork: _worker_loop: Worker process started")
        # P-1 RC-4: hook the worker into the parent's instrumentation queue so
        # ``rc4_ring_buffer.emit`` calls inside this process post events
        # cross-process instead of into a worker-local deque the parent will
        # never see. No-op when ``CASCOR_RC4_RING_BUFFER`` is unset.
        if instrumentation_queue is not None:
            from parallelism import rc4_ring_buffer as _rc4_worker

            _rc4_worker.set_worker_queue(instrumentation_queue)
            _rc4_worker.emit("worker.started")
        if shared_training_inputs is not None:
            logger.debug("CascadeCorrelationNetwork: _worker_loop: Received shared training inputs (RC-3 optimization active)")
        while True:
            try:
                # Get task from queue with timeout
                task = task_queue.get(timeout=task_queue_timeout)
            except Empty:

                # Stand-by mode: no task available, continue waiting
                logger.debug("CascadeCorrelationNetwork: _worker_loop: No task available, entering stand-by mode")
                time.sleep(0.1)
                continue
            except Exception as e:
                logger.critical(f"CascadeCorrelationNetwork: _worker_loop: Worker critical get error: {e}")
                import traceback

                logger.critical(f"CascadeCorrelationNetwork: _worker_loop: Traceback: {traceback.format_exc()}")
                break

            # Sentinel value to stop worker
            if task is None:
                logger.debug("CascadeCorrelationNetwork: _worker_loop: Received sentinel, stopping worker")
                break
            try:
                if instrumentation_queue is not None:
                    from parallelism import rc4_ring_buffer as _rc4_worker

                    _rc4_worker.emit("worker.task_received", candidate_id=task[0] if task else None, round_id=task[3] if task and len(task) > 3 else None)
                CascadeCorrelationNetwork._process_worker_task(task, shared_training_inputs, progress_queue, result_queue, parallel, logger)
                if instrumentation_queue is not None:
                    from parallelism import rc4_ring_buffer as _rc4_worker

                    _rc4_worker.emit("worker.task_done", candidate_id=task[0] if task else None)
            except Exception as e:
                logger.error(f"CascadeCorrelationNetwork: _worker_loop: Worker task error: {e}")
                import traceback

                logger.error(f"CascadeCorrelationNetwork: _worker_loop: Traceback: {traceback.format_exc()}")
                CascadeCorrelationNetwork._publish_failure_result(task, e, result_queue, logger)
        logger.debug("CascadeCorrelationNetwork: _worker_loop: Worker process ended")

    @staticmethod
    def _process_worker_task(task, shared_training_inputs, progress_queue, result_queue, parallel, logger):
        """Process a single candidate training task from the work queue."""
        # PARALLEL-FIX (RC-3): Reconstruct full task tuple from lightweight task + shared data.
        if shared_training_inputs is not None and len(task) == 2:
            full_task = (task[0], task[1], shared_training_inputs)
        else:
            full_task = task

        # Build progress callback from progress_queue if available
        _progress_cb = None
        if progress_queue is not None:
            from queue import Full as _FullQ

            def _make_progress_cb(pq):
                def _cb(**kwargs):
                    try:
                        pq.put_nowait(kwargs)
                    except _FullQ:
                        pass

                return _cb

            _progress_cb = _make_progress_cb(progress_queue)

        # Process the task
        logger.debug(f"CascadeCorrelationNetwork: _worker_loop: Processing task: {full_task[0] if full_task else 'None'}")
        # P-1 RC-4: bracket the actual training call so the dump shows
        # how long training took vs. how long the put took. The put is
        # the suspected race-window owner.
        from parallelism import rc4_ring_buffer as _rc4_worker_inner

        cid = full_task[0] if full_task else None
        rid = task[3] if task and len(task) > 3 else None
        _rc4_worker_inner.emit("worker.train_start", candidate_id=cid)
        result = CascadeCorrelationNetwork.train_candidate_worker(task_data_input=full_task, parallel=parallel, progress_callback=_progress_cb)
        _rc4_worker_inner.emit("worker.train_done", candidate_id=cid, correlation=getattr(result, "correlation", None), success=getattr(result, "success", None))
        logger.debug("CascadeCorrelationNetwork: _worker_loop: Task processed, putting result in queue")

        from queue import Full

        try:
            _rc4_worker_inner.emit("worker.put_start", candidate_id=cid, round_id=rid)
            result_queue.put(result, timeout=30)
            _rc4_worker_inner.emit("worker.put_done", candidate_id=cid)
            logger.debug("CascadeCorrelationNetwork: _worker_loop: Task completed successfully")
        except Full as fe:
            _rc4_worker_inner.emit("worker.put_full", candidate_id=cid)
            logger.error(f"CascadeCorrelationNetwork: _worker_loop: Result queue full, dropping result: {fe}")
            raise TrainingError from fe

    @staticmethod
    def _publish_failure_result(task, error, result_queue, logger):
        """Publish a failure result to the result queue after a task processing error."""
        from queue import Full

        try:
            candidate_index = task[0] if task and len(task) > 0 else 0
            candidate_uuid = task[1][4] if task and len(task) > 1 and len(task[1]) > 4 else None
            from candidate_unit.candidate_unit import CandidateTrainingResult

            failure_result = CandidateTrainingResult(
                candidate_id=candidate_index,
                candidate_uuid=candidate_uuid,
                correlation=0.0,
                candidate=None,
                success=False,
                error_message=str(error),
            )
            result_queue.put(failure_result, timeout=30)
            logger.debug("CascadeCorrelationNetwork: _worker_loop: Put failure result")
        except Full as fq_e:
            logger.error(f"CascadeCorrelationNetwork: _worker_loop: Failed to put failure result - queue full: {fq_e}")
        except Exception as put_e:
            logger.error(f"CascadeCorrelationNetwork: _worker_loop: Failed to put failure result: {put_e}")

    #################################################################################################################################################################################################
    # Public Method to calculate the residual error of the network
    def calculate_residual_error(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Description:
            Calculate the residual error of the network. P2-1d: zeros out the
            residual on output columns ``[self.active_output_dim:]`` when the
            active dim is less than ``self.output_size`` (post-shrink-via-
            padding) so downstream candidate training never correlates against
            signal from the zero-padded target tail. Masking the residual at
            source avoids threading ``active_output_dim`` through every
            multiprocess candidate worker.
        Args:
            x: Input tensor
            y: Target tensor
        Notes:
            - The input and target tensors must have the same shape.
        Returns:
            Residual error tensor
        """
        self.logger.debug(f"CascadeCorrelationNetwork: calculate_residual_error: Calculating residual error for input shape: {x.shape if isinstance(x, torch.Tensor) else 'None'}, target shape: {y.shape if isinstance(y, torch.Tensor) else 'None'}")
        x = (x, torch.empty(0, self.input_size))[x is None]
        self.logger.debug(f"CascadeCorrelationNetwork: calculate_residual_error: After defaulting, input shape: {x.shape if isinstance(x, torch.Tensor) else 'None'}, target shape: {y.shape if isinstance(y, torch.Tensor) else 'None'}")
        y = (y, torch.empty(0, self.output_size))[y is None]
        residual = torch.empty(0, self.output_size)
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            # raise TypeError("Input and target must be torch.Tensor")
            self.logger.debug(f"CascadeCorrelationNetwork: calculate_residual_error: Input and target must be torch.Tensor, x type: {type(x)}, y type: {type(y)}")
            x = torch.empty(0, self.input_size)
            y = torch.empty(0, self.output_size)
            self.logger.debug(f"CascadeCorrelationNetwork: calculate_residual_error: After defaulting, input shape: {x.shape}, target shape: {y.shape}")
        # Check batch size match (x and y must have same number of samples)
        if x.shape[0] != y.shape[0]:
            self.logger.debug(f"CascadeCorrelationNetwork: calculate_residual_error: Input and target must have the same batch size (dim 0), x shape: {x.shape}, y shape: {y.shape}")
            # Return empty residual for mismatched batch sizes
        elif y.shape[1] != self.output_size:
            self.logger.debug(f"CascadeCorrelationNetwork: calculate_residual_error: Target must have same output size as network, expected {self.output_size}, got {y.shape[1]}")
            # Return empty residual for mismatched output size
        else:
            # result = torch.empty(0, simple_network.input_size)
            self.logger.debug("CascadeCorrelationNetwork: calculate_residual_error: Forward pass to calculate output for residual error computation")
            with torch.no_grad():
                self.logger.debug("CascadeCorrelationNetwork: calculate_residual_error: Performing forward pass without gradient tracking")
                output = self.forward(x)
                self.logger.debug(f"CascadeCorrelationNetwork: calculate_residual_error: Forward pass completed, output shape: {output.shape}")
                residual = y - output
                self.logger.debug(f"CascadeCorrelationNetwork: calculate_residual_error: Calculated residual error, shape: {residual.shape}")
            self.logger.verbose(f"CascadeCorrelationNetwork: calculate_residual_error: Validating residual error, shape: {residual.shape}")
            residual = (residual, torch.empty(0, self.output_size))[residual is None]
            self.logger.debug(f"CascadeCorrelationNetwork: calculate_residual_error: Calculated residual error, shape: {residual.shape}")
        # P2-1d: zero out the inactive output dims so candidate correlation
        # against the residual never picks up signal from zero-padded target
        # columns. Mathematically equivalent to slicing residual + outputs
        # at every consumer site, but avoids threading active_output_dim
        # through the multiprocess candidate training pool.
        active_o = int(getattr(self, "active_output_dim", self.output_size) or self.output_size)
        if 0 <= active_o < residual.shape[1]:
            residual = residual.clone()
            residual[:, active_o:] = 0.0
            self.logger.debug(f"CascadeCorrelationNetwork: calculate_residual_error: Masked residual to active_output_dim={active_o}; dims [{active_o}:{self.output_size}] zeroed")
        self.logger.verbose(f"CascadeCorrelationNetwork: calculate_residual_error: Returning residual error, shape: {residual.shape}")
        return residual

    #################################################################################################################################################################################################
    # Public Method to add a new hidden unit based on the correlation
    def _emit_candidate_correlation(self, value) -> None:
        """OBS-WIRE-01 (A.2): emit the best-candidate-correlation gauge.

        Defensive: never let a metrics-emit failure break the training
        loop. Kept as a method (not a free function) so unit tests can
        ``patch.object`` it on a per-instance basis.
        """
        try:
            from api.observability import set_candidate_correlation as _set_candidate_correlation

            _set_candidate_correlation(float(value))
        except Exception:
            self.logger.debug("set_candidate_correlation emission failed", exc_info=True)

    def _raise_if_candidate_training_failed(self, training_results: Optional[TrainingResults]) -> None:
        """Issue #509: refuse to report an infrastructure failure as ``no_candidate``.

        ``grow_network`` reaches its ``no_candidate`` exit whenever the best candidate
        is ``None``. Two very different things produce that state:

        * **No candidate was good enough** — every candidate trained, none beat the
          correlation threshold. A real algorithmic outcome; the run legitimately
          completes with the units it already has.
        * **No candidate could be trained at all** — every candidate errored. The
          observed case is a full GPU, where ``CandidateUnit.__init__`` dies with
          ``AcceleratorError: CUDA error: out of memory``.

        The second case slips past the existing BUG-CC-18 / ROBUST-01 guards in
        ``_execute_candidate_training`` because those only fire when *both* training
        paths raise, or when the result list is empty. Here neither holds: the
        per-candidate handlers catch the error and **return** a
        ``CandidateTrainingResult(success=False, candidate=None)``, so a full set of
        all-failed results comes back normally. The run then reported ``succeeded`` /
        ``no_candidate`` / 1 hidden unit while having trained nothing — silently
        corrupting downstream experiment campaigns.

        ``success_count`` (candidates that trained *without erroring*) already
        distinguishes the two and was simply never consulted. Note that
        ``failed_count`` is **not** an error count — it is
        ``len(results) - successful_candidates``, i.e. candidates that missed the
        correlation threshold — so it cannot be used for this test.

        Raises:
            CandidateTrainingError: if candidates were attempted and none trained
                successfully. ``TrainingLifecycleManager`` already wraps
                ``model.fit`` in ``except Exception`` and marks the run Failed, so
                raising yields the correct terminal state, metric, and broadcast
                with no new plumbing. ``_completion_reason`` is set before the raise
                and survives it, because ``get_status`` reads it off the persisted
                network object.
        """
        if training_results is None:
            return
        attempted = len(training_results.candidate_ids or [])
        if not attempted or (training_results.success_count or 0) > 0:
            return

        # ``error_messages`` is annotated ``List[str]`` but ``get_candidates_error_messages``
        # returns a dict keyed by candidate id/uuid — accept either shape.
        raw_errors = training_results.error_messages or {}
        error_values = list(raw_errors.values()) if isinstance(raw_errors, dict) else list(raw_errors)
        sample = "; ".join(str(message) for message in error_values[:3]) or "no error message recorded"

        self._completion_reason = "candidate_training_failed"
        self.logger.error(f"CascadeCorrelationNetwork: _raise_if_candidate_training_failed: All {attempted} candidate(s) failed to train (0 trained successfully). This is an infrastructure failure, not a converged network. First errors: {sample}")
        raise CandidateTrainingError(f"All {attempted} candidate(s) failed to train (0 trained successfully). This is an infrastructure failure — commonly GPU exhaustion — not a converged network. Refusing to report it as a normal 'no_candidate' completion. First errors: {sample}")

    def _install_hidden_unit(
        self,
        weights: torch.Tensor,
        bias: torch.Tensor,
        activation_fn,
        correlation: float = 0.0,
    ) -> int:
        """CAN-015h-0: shared helper that appends a hidden unit to the network.

        Used by both the cascade-grow path (``add_unit`` /
        ``add_units_as_layer``) and the future
        ``POST /v1/network/hidden-units`` endpoint (h-2). Keeping
        the unit-installation logic in one place prevents the two
        callers from drifting apart on dict shape, history-record
        layout, or unit-index accounting.

        The helper is **install only** — it does NOT resize
        ``self.output_weights``. Call
        ``_resize_output_layer_for_new_units`` separately. The
        two-step split lets ``add_units_as_layer`` batch one resize
        for N units instead of N resizes.

        Args:
            weights: Cloned + detached weight tensor for the new unit.
            bias: Cloned + detached bias tensor for the new unit.
            activation_fn: Callable applied to the unit's pre-activation.
            correlation: Recorded in history. ``0.0`` is a sane
                default for manual inserts where correlation is
                undefined.

        Returns:
            int: 0-based index of the appended unit (i.e.
            ``len(self.hidden_units) - 1`` after the append).
        """
        new_unit = {
            "weights": weights.clone().detach(),
            "bias": bias.clone().detach(),
            "activation_fn": activation_fn,
            "correlation": correlation,
        }
        self.hidden_units.append(new_unit)
        new_index = len(self.hidden_units) - 1
        # Metadata-only history entry (CR-063 — full weight tensors
        # already live in self.hidden_units).
        self.history["hidden_units_added"].append(
            {
                "correlation": correlation,
                "weight_shape": tuple(weights.shape),
                "unit_index": new_index,
            }
        )
        return new_index

    def _resize_output_layer_for_new_units(self, num_added: int, prev_input_size: int) -> None:
        """CAN-015h-0: shared helper that widens ``self.output_weights``.

        Re-initializes the output-layer weight matrix with
        ``prev_input_size + num_added`` rows per
        ``self.init_output_weights`` (zero or random×0.1), copies the
        old weights into the ``[:prev_input_size, :]`` slice, and
        enables gradient tracking. ``self.output_bias`` is preserved
        untouched (its shape is ``[output_size]``, independent of
        the input width).

        No-op when ``num_added <= 0`` so callers don't have to
        special-case the empty-add case.

        **RNG-sensitive**: the random-init path consumes one
        ``torch.randn`` call. Callers in a deterministic context
        must invoke this at the same point in the RNG stream as the
        pre-refactor inline code did. The bit-identity test in
        ``test_install_hidden_unit_helper.py`` pins this.
        """
        if num_added <= 0:
            return
        old_output_weights = self.output_weights.clone().detach()
        # Pre-refactor (line ~3488 in add_unit, ~3592 in add_units_as_layer)
        # clones + detaches output_bias and reassigns it. Functionally a
        # ``requires_grad`` strip — preserve here so the refactor is bit-
        # identical from the optimizer's perspective.
        old_output_bias = self.output_bias.clone().detach()
        new_input_size = prev_input_size + num_added
        if self.init_output_weights == "zero":
            self.output_weights = torch.zeros(new_input_size, self.output_size)
        else:
            self.output_weights = torch.randn(new_input_size, self.output_size) * 0.1
        self.output_weights[:prev_input_size, :] = old_output_weights
        self.output_weights.requires_grad_(True)
        self.output_bias = old_output_bias

    def add_unit(
        self,
        candidate: CandidateUnit = None,
        x: torch.Tensor = None,
    ) -> None:
        """
        Description:
            Add a new hidden unit to the network.
            This method takes a candidate unit and an input tensor, and adds the candidate unit to the network.
            If no candidate unit is provided, a random candidate unit will be selected from the candidate pool.
        Args:
            candidate: Candidate unit to add
            x: Input tensor to calculate the units output
        Notes:
            This method updates the networks hidden units and output layer weights to include the new unit.
            If no candidate unit is provided, a random candidate unit will be selected from the candidate pool.
            The new hidden unit will be appended to the networks hidden units list.
            The output layer weights will be updated to include the new unit.
        Raises:
            ValueError: If the candidate unit is None or if the maximum number of hidden units has been reached.
            TypeError: If the input tensor is not a torch.Tensor.
        Returns:
            None
        """
        # Prepare input for the new unit (includes outputs from existing hidden units)
        self.logger.trace("CascadeCorrelationNetwork: add_unit: Starting to add a new hidden unit.")
        candidate_input = self._compute_hidden_outputs(x)
        self.logger.debug(f"CascadeCorrelationNetwork: add_unit: Candidate input shape: {candidate_input.shape}, Input size: {candidate_input.shape[1]}")

        # PARALLEL-FIX (RC-5): Validate that candidate weight dimensions match current input.
        # A stale candidate from a previous training round (result queue contamination) would
        # have weights sized for a different input dimension. Catch this before the cryptic
        # RuntimeError from element-wise multiplication.
        expected_weight_size = candidate_input.shape[1]
        actual_weight_size = candidate.weights.shape[0]
        if expected_weight_size != actual_weight_size:
            raise ValidationError(f"Candidate weight dimension mismatch in add_unit: " f"candidate_input has {expected_weight_size} features " f"(original_input={x.shape[1]}, hidden_units={len(self.hidden_units)}), " f"but candidate weights have {actual_weight_size} elements. " f"This indicates a stale candidate from a previous training round.")

        # CAN-015h-0: install the unit via the shared helper. The helper
        # appends to ``hidden_units`` and ``history``; we still need to
        # widen the output layer below.
        self._install_hidden_unit(
            weights=candidate.weights,
            bias=candidate.bias,
            activation_fn=self.activation_fn,
            correlation=candidate.correlation,
        )
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"CascadeCorrelationNetwork: add_unit: Current number of hidden units: {len(self.hidden_units)}, Hidden units: {self.hidden_units}")

        # Calculate the output of the new unit (logging only — does
        # not contribute to persisted state, kept for parity with
        # pre-refactor logging).
        if self.logger.isEnabledFor(logging.DEBUG):
            new_unit = self.hidden_units[-1]
            unit_output = self.activation_fn(torch.sum(candidate_input * new_unit["weights"], dim=1) + new_unit["bias"]).unsqueeze(1)
            self.logger.debug(f"CascadeCorrelationNetwork: add_unit: New unit output shape: {unit_output.shape}, New unit output: {unit_output}")

        # CAN-015h-0: widen the output layer via the shared helper.
        # Single-unit add ⇒ ``num_added=1`` against the pre-add input size.
        self._resize_output_layer_for_new_units(
            num_added=1,
            prev_input_size=candidate_input.shape[1],
        )
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"CascadeCorrelationNetwork: add_unit: New output weights shape: {self.output_weights.shape}, init_mode: {self.init_output_weights}")

        self.logger.info(f"CascadeCorrelationNetwork: add_unit: Added hidden unit with correlation: {candidate.correlation:.6f}")
        self.logger.info(f"CascadeCorrelationNetwork: add_unit: Current number of hidden units: {len(self.hidden_units)}")
        self.logger.trace("CascadeCorrelationNetwork: add_unit: Completed adding a new hidden unit.")

    def _select_best_candidates(
        self,
        results: list,
        num_candidates: int = 1,
        *,
        strategy: Optional[str] = None,
        top_count: Optional[int] = None,
        random_count: Optional[int] = None,
    ) -> list:
        """
        Description:
            Select N candidates for layer addition. Honors the strategy /
            counts written by the canopy adapter via the PR-4a PATCH surface
            (multi_candidate / candidate_selection / selected_candidates /
            top_candidates / random_candidates).
        Args:
            results: List of CandidateTrainingResult objects.
            num_candidates: Total candidates to select (S in the §1.5 C2.1
                triple). Defaults to 1, preserving the legacy single-best
                behavior when called without strategy kwargs.
            strategy: One of "top" / "random" / "mixed". If None, falls
                back to ``self.candidate_selection`` (default "top").
            top_count: T in the triple. When ``strategy=="mixed"``, the
                first T picks are by descending correlation. Defaults to
                ``self.top_candidates``.
            random_count: R in the triple. When ``strategy=="mixed"``, the
                last R picks are random from candidates remaining after the
                top-T selection. Defaults to ``self.random_candidates``.
        Notes:
            - Threshold filtering (``correlation_threshold``) is applied to
              the *pool* before selection — random/mixed picks never include
              below-threshold candidates.
            - Numerical determinism for ``random``/``mixed`` uses
              ``random.Random(self.random_seed)`` if a seed exists, else
              the module ``random`` singleton.
        Returns:
            List of selected CandidateTrainingResult objects (length <= S).
        """
        strategy = strategy or getattr(self, "candidate_selection", "top") or "top"
        if top_count is None:
            top_count = getattr(self, "top_candidates", num_candidates)
        if random_count is None:
            random_count = getattr(self, "random_candidates", 0)

        self.logger.debug(f"CascadeCorrelationNetwork: _select_best_candidates: strategy={strategy!r} S={num_candidates} T={top_count} R={random_count} pool={len(results)}")

        # Threshold filter first — every strategy honors it.
        threshold = getattr(self, "correlation_threshold", 0.0)
        eligible = [r for r in results if r.correlation is not None and abs(r.correlation) >= threshold]
        if not eligible:
            self.logger.info("CascadeCorrelationNetwork: _select_best_candidates: no candidates above correlation_threshold")
            return []

        # Sort by absolute correlation (descending) — used as the input order
        # for top-N and as the "remainder pool" for mixed.
        sorted_eligible = sorted(eligible, key=lambda r: abs(r.correlation), reverse=True)

        if strategy == "top":
            selected = sorted_eligible[:num_candidates]
        elif strategy == "random":
            # Uniform pick from the eligible pool. ``num_candidates`` is the
            # upper bound; if the pool has fewer eligible members we just
            # return all of them.
            rng = self._candidate_selection_rng()
            selected = list(rng.sample(sorted_eligible, min(num_candidates, len(sorted_eligible))))
        elif strategy == "mixed":
            # T picks by descending correlation, then R uniform random from
            # the remainder. The §1.5 C2.1 invariant guarantees T+R==S when
            # both nonzero, but we defensively bound by ``len(sorted_eligible)``.
            top_slice = sorted_eligible[:top_count]
            remainder = sorted_eligible[top_count:]
            rng = self._candidate_selection_rng()
            random_slice = list(rng.sample(remainder, min(random_count, len(remainder))))
            selected = top_slice + random_slice
        else:
            self.logger.warning(f"CascadeCorrelationNetwork: _select_best_candidates: unknown strategy {strategy!r}, falling back to 'top'")
            selected = sorted_eligible[:num_candidates]

        self.logger.info(f"CascadeCorrelationNetwork: _select_best_candidates: Selected {len(selected)} via {strategy!r} with correlations: {[round(r.correlation, 4) for r in selected]}")
        return selected

    def _candidate_selection_rng(self) -> random.Random:
        """Return a deterministic RNG seeded from the network's ``random_seed``
        if available, else the module ``random`` singleton.

        Local instance so a parallel candidate-training pass on another
        thread doesn't perturb this selection's draw sequence.
        """
        seed = getattr(self, "random_seed", None)
        return random.Random(seed) if seed is not None else random.Random()

    def _effective_candidate_count(self) -> int:
        """How many candidates ``grow_network`` should promote per iteration.

        Resolution order (first match wins):

          1. ``self.candidates_per_layer`` if > 1 — legacy attribute that
             scripted runs may have set; preserved for backward compatibility.
          2. ``self.selected_candidates`` if ``self.multi_candidate`` is True
             — the PR-4a PATCH-surface knobs.
          3. 1 — the single-best-candidate path.
        """
        legacy_count = getattr(self, "candidates_per_layer", None)
        if legacy_count is not None and legacy_count > 1:
            return int(legacy_count)
        if getattr(self, "multi_candidate", False):
            return int(getattr(self, "selected_candidates", 1))
        return 1

    def add_units_as_layer(self, candidates: list, x: torch.Tensor) -> None:
        """
        Description:
            Add multiple candidates as a new layer. Each of top N candidates is added as a separate hidden unit.
        Args:
            candidates: List of CandidateTrainingResult objects
            x: Input tensor for calculating outputs
        Notes:
            - All candidates were trained with the same candidate_input (same number of hidden units).
            - candidate_input is pre-computed ONCE before adding any units, so that all candidates
              are evaluated against the input state they were trained on (compute-then-mutate pattern).
            - The output layer weights are updated after all units are added.
        Returns:
            None
        """
        self.logger.info(f"CascadeCorrelationNetwork: add_units_as_layer: Adding layer with {len(candidates)} units")

        # Pre-compute candidate_input from current hidden units BEFORE adding any new units.
        # All candidates in this batch were trained with this exact input shape.
        candidate_input = self._compute_hidden_outputs(x)

        # CAN-015h-0: install via the shared helper. Each successful
        # candidate appends to ``hidden_units`` + ``history`` in
        # exactly the same order as the pre-refactor inline append.
        added_count = 0
        for candidate_result in candidates:
            candidate = candidate_result.candidate
            if candidate and hasattr(candidate, "weights"):
                # Validate weight dimensions match pre-computed input
                expected_size = candidate_input.shape[1]
                actual_size = candidate.weights.shape[0]
                if expected_size != actual_size:
                    self.logger.warning(f"CascadeCorrelationNetwork: add_units_as_layer: " f"Skipping candidate with mismatched weights: expected {expected_size}, got {actual_size}")
                    continue
                self._install_hidden_unit(
                    weights=candidate.weights,
                    bias=candidate.bias,
                    activation_fn=self.activation_fn,
                    correlation=candidate.correlation,
                )
                added_count += 1
            else:
                self.logger.warning(f"CascadeCorrelationNetwork: add_units_as_layer: Skipping invalid candidate: {candidate_result}")

        # CAN-015h-0: single batch resize for all newly-installed units.
        # The helper is a no-op when ``num_added == 0`` so we don't
        # have to special-case the all-skipped path.
        self._resize_output_layer_for_new_units(
            num_added=added_count,
            prev_input_size=candidate_input.shape[1],
        )

        self.logger.info(f"CascadeCorrelationNetwork: add_units_as_layer: Layer added ({added_count} units), total hidden units: {len(self.hidden_units)}")

    #################################################################################################################################################################################################
    # Public Method to grow the network by adding hidden units
    # This method is the core of the Cascade Correlation algorithm
    # It iteratively adds hidden units based on the residual error until stopping criteria are met
    def grow_network(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        max_iterations: int = 1000,
        early_stopping: bool = True,
        patience_counter: int = 0,
        best_value_loss: float = float("inf"),
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        on_grow_iteration_callback=None,
    ) -> ValidateTrainingResults:
        """
        Description:
            Grow the network by adding hidden units based on the residual error until stopping criteria are met.
        Args:
            x_train: Training input tensor
            y_train: Training target tensor
            max_iterations: Maximum number of iterations (network growth cycles) to perform
            early_stopping: Whether to use early stopping
            patience_counter: Counter for early stopping patience
            best_value_loss: Best validation loss seen so far
            x_val: Validation input tensor
            y_val: Validation target tensor
        Raises:
            TrainingError: If an error occurs during training
        Notes:
            - Candidate units are trained using the Cascade Correlation algorithm
            - Early stopping is used if specified
            - Validation loss and accuracy are calculated and tracked
            - Training history is tracked
            - Hidden units are added to the network using the Cascade Correlation algorithm
        Returns:
            ValidateTrainingResults dataclass object containing:
                - early_stop: Whether training was stopped early
                - patience_counter: Updated patience counter
                - best_value_loss: Best validation loss seen so far
                - value_output: Output on validation set
                - value_loss: Validation loss
                - value_accuracy: Validation accuracy
        """
        self.logger.trace("CascadeCorrelationNetwork: grow_network: Starting to grow the network by adding hidden units.")

        validate_training_results: Optional[ValidateTrainingResults] = None
        # Reset the completion-reason diagnostic for this growth run; set at
        # each loop exit below (and "max_iterations" via the for/else clause).
        self._completion_reason = None
        # epochs_completed = 0
        # for growth_iteration in range(max_epochs):
        iterations_completed = 0
        for iteration in range(max_iterations):

            # Calculate residual error
            residual_error = self._calculate_residual_error_safe(x_train=x_train, y_train=y_train)
            if residual_error is None:
                self.logger.warning("CascadeCorrelationNetwork: grow_network: Residual error is None, stopping growth of the network.")
                self._completion_reason = "residual_collapsed"
                break
            # self.logger.debug(f"CascadeCorrelationNetwork: grow_network: Growth Iteration {growth_iteration}, Residual Error: {residual_error.mean().item():.6f}")
            self.logger.debug(f"CascadeCorrelationNetwork: grow_network: Iteration {iteration}, Residual Error: {residual_error.mean().item():.6f}")

            # Train candidate units
            if not (training_results := self._get_training_results(x_train=x_train, y_train=y_train, residual_error=residual_error)) or not training_results.best_candidate:
                # Issue #509: "no candidate was good enough" and "no candidate could be
                # trained at all" both land here. Separate them before recording the
                # benign outcome — the infrastructure case raises instead of breaking.
                self._raise_if_candidate_training_failed(training_results)
                self.logger.warning("CascadeCorrelationNetwork: grow_network: Training results are None or best candidate is None, stopping growth of the network.")
                self._completion_reason = "no_candidate"
                break

            # Check if best candidate meets correlation threshold (adaptive)
            # As training progresses and residual errors shrink, candidates naturally
            # achieve lower correlations. Use an adaptive threshold that scales with
            # the residual error magnitude to prevent premature cascade termination.
            residual_magnitude = residual_error.abs().mean().item()
            adaptive_threshold = max(1e-6, min(self.correlation_threshold, residual_magnitude * 0.01))
            best_correlation = training_results.best_candidate.get_correlation()
            # OBS-WIRE-01 (A.2): emit the best-candidate correlation
            # gauge. Sampled once per grow-iteration after the
            # candidate pool finishes training. Extracted into a tiny
            # helper to keep the grow_network branch count under the
            # flake8 C901 ceiling.
            self._emit_candidate_correlation(best_correlation)
            if best_correlation < adaptive_threshold:
                self.logger.info(f"CascadeCorrelationNetwork: grow_network: No candidate met adaptive correlation threshold: {adaptive_threshold:.6f} (static: {self.correlation_threshold}, residual_mag: {residual_magnitude:.6f}), Best Correlation Achieved: {best_correlation:.6f}")
                self._completion_reason = "below_threshold"
                break
            self.logger.info(f"CascadeCorrelationNetwork: grow_network: Best Candidate: {best_correlation if training_results.best_candidate else None}, Met adaptive correlation threshold: {adaptive_threshold:.6f} (static: {self.correlation_threshold})")

            # Grow iteration callback for real-time state updates
            _grow_cb = on_grow_iteration_callback or getattr(self, "_grow_iteration_callback", None)
            if _grow_cb is not None:
                pool_size = getattr(self, "candidate_pool_size", 0)
                _candidate_ids = getattr(training_results, "candidate_ids", [])
                _correlations = getattr(training_results, "correlations", [])
                _grow_cb(
                    # iteration=growth_iteration,
                    # max_iterations=self.max_hidden_units,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    best_correlation=float(training_results.best_candidate.get_correlation()),
                    candidates_trained=len(getattr(training_results, "candidate_objects", [])),
                    candidates_total=pool_size,
                    phase_detail="adding_candidate",
                    best_candidate_id=getattr(training_results, "best_candidate_id", -1),
                    best_candidate_uuid=getattr(training_results, "best_candidate_uuid", ""),
                    second_candidate_id=_candidate_ids[1] if len(_candidate_ids) > 1 else None,
                    second_candidate_correlation=float(_correlations[1]) if len(_correlations) > 1 else 0.0,
                    all_correlations=list(_correlations),
                )

            # FRONTEND_ISSUES_PLAN_2026-05-09 §1.5 C2 / Issue #1 — multi-candidate
            # promotion plus the (S, T, R) selection-strategy triple, wired
            # through the PR-4a PATCH surface (multi_candidate /
            # candidate_selection / selected_candidates / top/random_candidates).
            effective_count = self._effective_candidate_count()

            # Add candidate(s) to the network and retrain the output layer
            if effective_count > 1:
                if selected_candidates := self._select_best_candidates(
                    training_results.candidate_objects,
                    num_candidates=effective_count,
                ):
                    self.add_units_as_layer([c for c in selected_candidates if c.candidate], x_train)
                    train_loss = self.train_output_layer(x_train, y_train, self.output_epochs)
                    train_accuracy = self.get_accuracy(x_train, y_train)
                    self.logger.info(f"CascadeCorrelationNetwork: grow_network: Added {len(selected_candidates)} candidates as layer (strategy={getattr(self, 'candidate_selection', 'top')!r})")
                else:
                    self.logger.warning("CascadeCorrelationNetwork: grow_network: No candidates met selection criteria")
                    self._completion_reason = "no_candidate"
                    break
            else:

                #    # Original behavior: Add single best candidate
                #    train_loss, train_accuracy = self._add_best_candidate(training_results.best_candidate, x_train, y_train, growth_iteration)
                # self.logger.debug(f"CascadeCorrelationNetwork: grow_network: After adding candidate(s), Training Loss: {train_loss:.6f}, Training Accuracy: {train_accuracy:.4f}, For Growth Iteration {growth_iteration}")
                #
                # # Prepare inputs for validation of training results
                # validate_training_inputs = ValidateTrainingInputs(
                #     epoch=growth_iteration,
                #     max_epochs=max_epochs,
                train_loss, train_accuracy = self._add_best_candidate(training_results.best_candidate, x_train, y_train, iteration)
            self.logger.debug(f"CascadeCorrelationNetwork: grow_network: After adding candidate(s), Training Loss: {train_loss:.6f}, Training Accuracy: {train_accuracy:.4f}, For Current Iteration {iteration}")

            # Prepare inputs for validation of training results
            validate_training_inputs = ValidateTrainingInputs(
                iteration=iteration,
                max_iterations=max_iterations,
                patience_counter=patience_counter,
                early_stopping=early_stopping,
                train_accuracy=train_accuracy,
                train_loss=train_loss,
                best_value_loss=best_value_loss,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            )
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"CascadeCorrelationNetwork: grow_network: Validate Training Inputs: {validate_training_inputs}")

            # Validation of training results
            try:
                validate_training_results = self.validate_training(validate_training_inputs)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"CascadeCorrelationNetwork: grow_network: Validation Results: {validate_training_results}")
            except Exception as e:
                # self.logger.error(f"CascadeCorrelationNetwork: grow_network: Caught Exception while validating training at growth iteration {growth_iteration + 1}/{max_epochs}:\nException:\n{e}")
                self.logger.error(f"CascadeCorrelationNetwork: grow_network: Caught Exception while validating training at iteration {iteration + 1}/{max_iterations}:\nException:\n{e}")
                import traceback

                traceback.print_exc()
                raise TrainingError from e

            # Update loop state from validation results (critical for early stopping convergence)
            patience_counter = validate_training_results.patience_counter
            best_value_loss = validate_training_results.best_value_loss
            # ...and publish onto the instance, which is what a SNAPSHOT records.
            #
            # Until 2026-08-23 these lived only as locals here. They drove early stopping
            # correctly, but ``self.best_value_loss`` and ``self.patience_counter`` were never
            # assigned anywhere in this class -- so ``_save_metadata``'s
            # ``getattr(network, "best_value_loss", float("inf"))`` fell through to its default
            # on every write. Measured over the live archive: ``best_value_loss`` is ``inf``
            # and ``patience_counter`` is ``0`` in **all 27,908** snapshots.
            #
            # That is worse than missing data, because it is INDISTINGUISHABLE from a real
            # measurement. A reader cannot tell "training never improved" from "nobody ever
            # wrote this down", and the archive read literally says no run ever converged.
            self.best_value_loss = best_value_loss
            self.patience_counter = patience_counter
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"CascadeCorrelationNetwork: grow_network: Iteration {iteration}, Early Stop: {validate_training_results.early_stop}, Patience Counter: {validate_training_results.patience_counter}, Best Value Loss: {validate_training_results.best_value_loss:.6f}, Value Output: {validate_training_results.value_output} Value Loss: {validate_training_results.value_loss:.6f}, Value Accuracy: {validate_training_results.value_accuracy:.4f}")
            if validate_training_results.early_stop:
                self.logger.info(f"CascadeCorrelationNetwork: grow_network: Early stopping triggered at iteration {iteration}.")
                self._completion_reason = "early_stopped"
                break
            self.logger.info(f"CascadeCorrelationNetwork: grow_network: Iteration {iteration} - Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}, Early stop: {validate_training_results.early_stop}")
            iterations_completed = iteration + 1
        else:
            # The for-loop ran every iteration without breaking → we hit the
            # iteration cap rather than converging or stalling.
            self._completion_reason = "max_iterations"
        if not validate_training_results:
            self.logger.warning(f"CascadeCorrelationNetwork: grow_network: No validation was performed (training loop exited early or did not execute). Iterations completed: {iterations_completed}/{max_iterations}.")
            validate_training_results = ValidateTrainingResults(
                early_stop=False,
                patience_counter=patience_counter,
                best_value_loss=best_value_loss,
                value_output=None,
                value_loss=float("inf"),
                value_accuracy=0.0,
            )
        self.logger.info(f"CascadeCorrelationNetwork: grow_network: Finished training after {iterations_completed} iterations. Total hidden units: {len(self.hidden_units)}. Completion reason: {self._completion_reason}")
        self.logger.debug(f"CascadeCorrelationNetwork: grow_network: Final history: {len(self.history.get('train_loss', []))} epochs recorded")
        self.logger.trace("CascadeCorrelationNetwork: grow_network: Completed training of the network.")
        return validate_training_results

    # Calculate residual error
    def _calculate_residual_error_safe(
        self,
        x_train: torch.Tensor = None,
        y_train: torch.Tensor = None,
        iteration: int = 0,
        max_iterations: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Description:
            Safely calculate the residual error between predicted and true values. Handles exceptions and logs progress.
        Args:
            x_train: Training input tensor
            y_train: Training target tensor
            iteration: Current iteration number
            max_iterations: Maximum number of iterations
        Raises:
            TrainingError: If an error occurs during calculation
        Notes:
            - Validates input tensors
            - Logs progress and errors
        Returns:
            Residual error tensor or None if an error occurred
        """
        # Validate method input parameters
        if x_train is None or y_train is None or x_train.shape[0] == 0 or y_train.shape[0] == 0:
            self.logger.warning("CascadeCorrelationNetwork: _calculate_residual_error_safe: Training data is None or empty, cannot calculate residual error")
            return None
        try:
            self.logger.debug(f"CascadeCorrelationNetwork: _calculate_residual_error_safe: Starting iteration {iteration + 1}/{max_iterations}")
            residual_error = self.calculate_residual_error(x_train, y_train)
            self.logger.debug(f"CascadeCorrelationNetwork: _calculate_residual_error_safe: Iteration {iteration}, Residual Error: {residual_error.mean().item():.6f}")
        except Exception as e:
            self.logger.error(f"CascadeCorrelationNetwork: _calculate_residual_error_safe: Caught Exception while calculating residual error at iteration {iteration + 1}/{max_iterations}:\nException:\n{e}")
            import traceback

            traceback.print_exc()
            raise TrainingError from e
        return residual_error

    # Train candidate units
    def _get_training_results(
        self,
        x_train: torch.Tensor = None,
        y_train: torch.Tensor = None,
        residual_error: torch.Tensor = None,
        iteration: int = 0,
        max_iterations: int = 0,
    ) -> TrainingResults:
        """
        Description:
            Get training results for candidate units
        Args:
            x_train: Training input tensor
            y_train: Training target tensor
            residual_error: Residual error tensor
            iteration: Current iteration number
            max_iterations: Maximum number of iterations
        Raises:
            TrainingError: If an error occurs during training
        Notes:
            - Validates input tensors
            - Logs progress and errors
        Returns:
            TrainingResults dataclass object
        """
        try:
            # Get training results as TrainingResults dataclass object
            training_results = self.train_candidates(x=x_train, y=y_train, residual_error=residual_error)
            self.logger.debug(f"CascadeCorrelationNetwork: _get_training_results: Training Results - Iteration {iteration}, Successful: {training_results.successful_candidates}, Failed: {training_results.failed_count}")
        except Exception as e:
            self.logger.error(f"CascadeCorrelationNetwork: _get_training_results: Caught Exception while training candidates at iteration {iteration + 1}/{max_iterations}:\nException:\n{e}")
            import traceback

            traceback.print_exc()
            raise TrainingError from e
        return training_results

    # Add the best candidate to the network and retrain the output layer
    def _add_best_candidate(
        self,
        best_candidate: CandidateUnit = None,
        x_train: torch.Tensor = None,
        y_train: torch.Tensor = None,
        iteration: int = 0,
        max_iterations: int = None,
    ) -> Optional[Tuple[float, float]]:
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"CascadeCorrelationNetwork: _add_best_candidate: Adding best candidate {best_candidate} at iteration {iteration}")
        if best_candidate is None:
            self.logger.warning("CascadeCorrelationNetwork: _add_best_candidate: Best candidate is None, cannot add to network")
            return None, None
        try:

            # Add best candidate to the network
            self.add_unit(best_candidate, x_train)
            self.logger.info("CascadeCorrelationNetwork: _add_best_candidate: Added best candidate to the network")
            train_loss = self._retrain_output_layer(x_train=x_train, y_train=y_train, epochs=self.output_epochs, epoch=iteration)
            self.logger.debug(f"CascadeCorrelationNetwork: _add_best_candidate: Training Loss: {train_loss}, For Current Iteration {iteration}")
            train_accuracy = self._calculate_train_accuracy(x_train=x_train, y_train=y_train, epoch=iteration)
            self.logger.debug(f"CascadeCorrelationNetwork: _add_best_candidate: Training Accuracy: {train_accuracy}, For Current Iteration {iteration}")
        except Exception as e:
            self.logger.error(f"CascadeCorrelationNetwork: _add_best_candidate: Caught Exception while adding unit and retraining output layer at iteration {iteration + 1}/{max_iterations}:\nException:\n{e}")
            import traceback

            traceback.print_exc()
            raise TrainingError from e
        return train_loss, train_accuracy

    # Calculate training accuracy
    def _calculate_train_accuracy(self, x_train: torch.Tensor = None, y_train: torch.Tensor = None, epoch: int = 0) -> float:
        # Validate method input parameters
        if x_train is None or y_train is None or x_train.shape[0] == 0 or y_train.shape[0] == 0:
            self.logger.warning("CascadeCorrelationNetwork: _calculate_train_accuracy: Training data is None or empty, cannot calculate accuracy")
            return 0.0
        if x_train.shape[0] != y_train.shape[0]:
            self.logger.warning(f"CascadeCorrelationNetwork: _calculate_train_accuracy: Training data and target have different number of samples, x_train shape: {x_train.shape}, y_train shape: {y_train.shape}, cannot calculate accuracy")
            return 0.0

        # Calculate accuracy
        train_accuracy = self.calculate_accuracy(x_train, y_train)
        self.logger.debug(f"CascadeCorrelationNetwork: _calculate_train_accuracy: For Current Epoch {epoch}, Train Accuracy: {train_accuracy:.4f}")

        # Update training history
        self.history["train_accuracy"].append(train_accuracy)
        self.logger.debug(f"CascadeCorrelationNetwork: _calculate_train_accuracy: For Current Epoch {epoch}, Accuracy: {train_accuracy}")
        return train_accuracy

    # Retrain the output layer after adding a new hidden unit
    def _retrain_output_layer(
        self,
        x_train: torch.Tensor = None,
        y_train: torch.Tensor = None,
        epochs: int = 0,
        epoch: int = 0,
    ) -> float:
        # Validate method input parameters
        if x_train is None or y_train is None or x_train.shape[0] == 0 or y_train.shape[0] == 0:
            self.logger.warning("CascadeCorrelationNetwork: _retrain_output_layer: Training data is None or empty, cannot retrain output layer")
            return float("inf")
        if x_train.shape[0] != y_train.shape[0]:
            self.logger.warning(f"CascadeCorrelationNetwork: _retrain_output_layer: Training data and target have different number of samples, x_train shape: {x_train.shape}, y_train shape: {y_train.shape}, cannot retrain output layer")
            return float("inf")
        if epochs <= 0:
            self.logger.warning(f"CascadeCorrelationNetwork: _retrain_output_layer: Number of epochs for retraining output layer is non-positive: {epochs}, skipping retraining")
            return float("inf")
        self.logger.info(f"CascadeCorrelationNetwork: _retrain_output_layer: Retraining output layer for {epochs} epochs after adding new hidden unit")

        # Retrain output layer
        train_loss = self.train_output_layer(x_train, y_train, self.output_epochs)
        self.logger.info(f"CascadeCorrelationNetwork: _retrain_output_layer: Full Network Training Loss after Epoch {epoch}, Train Loss: {train_loss:.6f}")
        self.logger.debug(f"CascadeCorrelationNetwork: _retrain_output_layer: For Current Epoch: {epoch}, Train Loss: {train_loss}")

        # Update training history
        self.history["train_loss"].append(train_loss)
        self.logger.debug(f"CascadeCorrelationNetwork: _retrain_output_layer: For Current Epoch: {epoch}, Training complete")
        return train_loss

    #################################################################################################################################################################################################
    # Define Snapshot and Recovery methods using hdf5 serialization
    def create_snapshot(self, snapshot_dir: Union[str, pl.Path] = None) -> Optional[pl.Path]:
        """
        Create a timestamped snapshot of the current network state.
        Args:
            snapshot_dir: Directory to save snapshots (defaults to ./snapshots)
        Returns:
            Path to created snapshot or None if failed
        """
        try:
            # Ensure snapshot directory exists
            if snapshot_dir is None:
                snapshot_dir = pl.Path(self.cascade_correlation_network_snapshots_dir) or pl.Path(_CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR)
            else:
                snapshot_dir = pl.Path(snapshot_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp and UUID
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            uuid = str(self.get_uuid())
            filename = f"cascor_snapshot_{timestamp}_{uuid}.h5"
            snapshot_path = pl.Path(snapshot_dir).joinpath(filename)

            # Save the snapshot
            if self._save_to_hdf5(
                snapshot_path,
                include_training_data=False,
                create_backup=False,
            ):
                self.logger.info(f"CascadeCorrelationNetwork: create_snapshot: Created snapshot at {snapshot_path}")
                return snapshot_path
            else:
                return None
        except Exception as e:
            self.logger.error(f"CascadeCorrelationNetwork: create_snapshot: Error: {e}")
            return None

    @classmethod
    def restore_snapshot(
        cls,
        snapshot_path: Union[str, pl.Path] = None,
        restore_multiprocessing: bool = True,
    ) -> Union["CascadeCorrelationNetwork", bool]:
        """
        Restore a network from a snapshot file.
        Args:
            snapshot_path: Path to the snapshot file
            restore_multiprocessing: Whether to restore multiprocessing state
        Returns:
            The restored CascadeCorrelationNetwork on success; False on failure
        """
        logger = Logger
        try:
            if snapshot_path is None:
                logger.error("CascadeCorrelationNetwork: restore_snapshot: No snapshot path provided")
                return False
            snapshot_path = pl.Path(snapshot_path)
            if not snapshot_path.exists():
                logger.error(f"CascadeCorrelationNetwork: restore_snapshot: Snapshot file does not exist: {snapshot_path}")
                return False
            loaded_network = cls._load_from_hdf5(
                filepath=snapshot_path,
                restore_multiprocessing=restore_multiprocessing,
                logger=logger,
            )
            if loaded_network is None:
                logger.error(f"CascadeCorrelationNetwork: restore_snapshot: Failed to load network from snapshot: {snapshot_path}")
                return False

            logger.info(f"CascadeCorrelationNetwork: restore_snapshot: Restored snapshot from {snapshot_path}")
            return loaded_network
        except Exception as e:
            logger.error(f"CascadeCorrelationNetwork: restore_snapshot: Error restoring snapshot: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return False

    #################################################################################################################################################################################################
    # methods to save provided object to hdf5
    def save_object(self, objectify: Any = None, snapshot_dir: Union[str, pl.Path] = None) -> Optional[pl.Path]:
        """
        Create a timestamped snapshot of the provided object state.
        Args:
            snapshot_dir: Directory to save snapshots (defaults to ./snapshots)
        Returns:
            Path to created snapshot or None if failed
        """
        try:
            # Ensure snapshot directory exists
            if snapshot_dir is None:
                snapshot_dir = pl.Path(self.cascade_correlation_network_snapshots_dir) or pl.Path(_CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR)
            else:
                snapshot_dir = pl.Path(snapshot_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with object's name, timestamp, and UUID
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            uuid = str(objectify.get_uuid())
            object_name = objectify.__name__
            filename = f"{object_name}_snapshot_{timestamp}_{uuid}.h5"
            snapshot_path = pl.Path(snapshot_dir).joinpath(filename)

            # Save the snapshot
            if self._save_to_hdf5(
                snapshot_path,
                create_backup=False,
            ):
                self.logger.info(f"CascadeCorrelationNetwork: create_snapshot: Created snapshot at {snapshot_path}")
                return snapshot_path
            else:
                return None
        except Exception as e:
            self.logger.error(f"CascadeCorrelationNetwork: create_snapshot: Error: {e}")
            return None

    def _save_object_hdf5(
        self,
        objectify: Any,
        filepath: Union[str, pl.Path],
        compression: str = "gzip",
        compression_opts: int = 4,
        create_backup: bool = True,
    ) -> bool:  # sourcery skip: class-extract-method
        """
        Save this network to HDF5 format.
        Args:
            filepath: Target file path for HDF5 file
            include_training_state: Whether to include training history
            include_training_data: Whether to include training datasets (excluded by default)
            compression: HDF5 compression method ('gzip', 'lzf', 'szip')
            compression_opts: Compression level (0-9 for gzip)
            create_backup: Whether to create a backup before saving
        Returns:
            bool: Success status
        """
        try:
            from snapshots.snapshot_serializer import CascadeHDF5Serializer
            from snapshots.snapshot_utils import HDF5Utils

            serializer = CascadeHDF5Serializer(logger=self.logger)

            # Create backup if requested and file already exists
            if create_backup and os.path.exists(filepath):
                backup_dir = pl.Path(filepath).parent / "backups"
                backup_path = HDF5Utils.create_backup(str(filepath), str(backup_dir))
                self.logger.info(f"CascadeCorrelationNetwork: Created backup at {backup_path}")

            # Save the current object
            if success := serializer.save_object(
                objectify=objectify,
                filepath=filepath,
                compression=compression,
                compression_opts=compression_opts,
            ):
                self.logger.info(f"CascadeCorrelationNetwork: save_to_hdf5: Successfully saved to {filepath}")
            else:
                self.logger.error(f"CascadeCorrelationNetwork: save_to_hdf5: Failed to save to {filepath}")
            self.logger.debug("CascadeCorrelationNetwork: save_to_hdf5: Verifying saved HDF5 file")
            checked_object = self.verify_hdf5_file(filepath)
            if not checked_object.get("valid", False):
                self.logger.error(f"CascadeCorrelationNetwork: save_to_hdf5: Verification failed for saved HDF5 file: {filepath}, Error: {checked_object.get('error', 'Unknown error')}")
                return False
            self.logger.info(f"CascadeCorrelationNetwork: save_to_hdf5: Verified saved HDF5 file is valid: {filepath}")
            return success
        except Exception as e:
            self.logger.error(f"CascadeCorrelationNetwork: save_to_hdf5: Error saving to HDF5: {e}")
            import traceback

            self.logger.debug(traceback.format_exc())
            return False

    #################################################################################################################################################################################################
    # Private helper methods for HDF5 serialization of self
    def save_to_hdf5(
        self,
        filepath: Union[str, pl.Path],
        include_training_state: bool = True,
        include_training_data: bool = False,
        compression: str = "gzip",
        compression_opts: int = 4,
        create_backup: bool = False,
    ) -> bool:
        """
        Public method to save network to HDF5 format.
        Args:
            filepath: Target file path for HDF5 file
            include_training_state: Whether to include training history (default: True)
            include_training_data: Whether to include training datasets (default: False)
            compression: HDF5 compression method
            compression_opts: Compression level (0-9 for gzip)
            create_backup: Whether to create backup before saving
        Returns:
            bool: Success status
        """
        return self._save_to_hdf5(
            filepath=filepath,
            include_training_state=include_training_state,
            include_training_data=include_training_data,
            compression=compression,
            compression_opts=compression_opts,
            create_backup=create_backup,
        )

    def _save_to_hdf5(
        self,
        filepath: Union[str, pl.Path],
        include_training_state: bool = False,
        include_training_data: bool = False,
        compression: str = "gzip",
        compression_opts: int = 4,
        create_backup: bool = True,
    ) -> bool:  # sourcery skip: class-extract-method
        """
        Internal method to save network to HDF5 format.
        Args:
            filepath: Target file path for HDF5 file
            include_training_state: Whether to include training history
            include_training_data: Whether to include training datasets (excluded by default)
            compression: HDF5 compression method ('gzip', 'lzf', 'szip')
            compression_opts: Compression level (0-9 for gzip)
            create_backup: Whether to create a backup before saving
        Returns:
            bool: Success status
        """
        try:
            from snapshots.snapshot_serializer import CascadeHDF5Serializer
            from snapshots.snapshot_utils import HDF5Utils

            serializer = CascadeHDF5Serializer(logger=self.logger)

            # Create backup if requested and file already exists
            if create_backup and os.path.exists(filepath):
                backup_dir = pl.Path(filepath).parent / "backups"
                backup_path = HDF5Utils.create_backup(str(filepath), str(backup_dir))
                self.logger.info(f"CascadeCorrelationNetwork: Created backup at {backup_path}")

            # Save the network
            success = serializer.save_network(
                network=self,
                filepath=filepath,
                include_training_state=include_training_state,
                include_training_data=include_training_data,
                compression=compression,
                compression_opts=compression_opts,
            )
            if success:
                self.logger.info(f"CascadeCorrelationNetwork: save_to_hdf5: Successfully saved to {filepath}")
            else:
                self.logger.error(f"CascadeCorrelationNetwork: save_to_hdf5: Failed to save to {filepath}")
            self.logger.debug("CascadeCorrelationNetwork: save_to_hdf5: Verifying saved HDF5 file")
            checked_network = self.verify_hdf5_file(filepath)
            if not checked_network.get("valid", False):
                self.logger.error(f"CascadeCorrelationNetwork: save_to_hdf5: Verification failed for saved HDF5 file: {filepath}, Error: {checked_network.get('error', 'Unknown error')}")
                return False
            self.logger.info(f"CascadeCorrelationNetwork: save_to_hdf5: Verified saved HDF5 file is valid: {filepath}")
            return success
        except Exception as e:
            self.logger.error(f"CascadeCorrelationNetwork: save_to_hdf5: Error saving to HDF5: {e}")
            import traceback

            self.logger.debug(traceback.format_exc())
            return False

    @classmethod
    def load_from_hdf5(cls, filepath: Union[str, pl.Path], restore_multiprocessing: bool = False):
        """
        Public classmethod to load network from HDF5 file.
        Args:
            filepath: Path to HDF5 file
            restore_multiprocessing: Whether to restore multiprocessing state (default: False)
        Returns:
            CascadeCorrelationNetwork instance or None if failed
        """
        return cls._load_from_hdf5(filepath=filepath, restore_multiprocessing=restore_multiprocessing)

    @classmethod
    def _load_from_hdf5(
        cls,
        filepath: Union[str, pl.Path],
        restore_multiprocessing: bool = True,
        logger: Logger = None,
    ) -> Optional["CascadeCorrelationNetwork"]:
        """
        Load a network from HDF5 format.
        Args:
            filepath: Path to HDF5 file
            restore_multiprocessing: Whether to restore multiprocessing state
            logger: Logger instance to use
        Returns:
            CascadeCorrelationNetwork instance or None if failed
        """
        logger = logger or Logger
        try:
            from snapshots.snapshot_serializer import CascadeHDF5Serializer

            serializer = CascadeHDF5Serializer(logger=logger)
            network = serializer.load_network(filepath=filepath, restore_multiprocessing=restore_multiprocessing)
            if network:
                network.logger.info(f"CascadeCorrelationNetwork: load_from_hdf5: Successfully loaded from {filepath}")
            else:
                logger.error(f"CascadeCorrelationNetwork: load_from_hdf5: Failed to load from {filepath}")
            return network
        except Exception as e:
            logger.error(f"CascadeCorrelationNetwork: load_from_hdf5: Error loading from HDF5: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None

    def list_hdf5_snapshots(self, directory: Union[str, pl.Path]) -> List[pl.Path]:
        # sourcery skip: extract-method
        """
        List all HDF5 snapshot files in a directory.
        Args:
            directory: Directory to search for HDF5 files
        Returns:
            List of HDF5 file paths
        """
        try:
            from snapshots.snapshot_utils import HDF5Utils

            directory = pl.Path(directory)
            if not directory.exists() or not directory.is_dir():
                self.logger.error(f"CascadeCorrelationNetwork: list_hdf5_snapshots: Directory does not exist: {directory}")
                return []
            hdf5_files = HDF5Utils.list_hdf5_files(directory)
            self.logger.info(f"CascadeCorrelationNetwork: list_hdf5_snapshots: Found {len(hdf5_files)} HDF5 files in {directory}")
            return hdf5_files
        except Exception as e:
            self.logger.error(f"CascadeCorrelationNetwork: list_hdf5_snapshots: Error listing HDF5 files: {e}")
            import traceback

            self.logger.debug(traceback.format_exc())
            return []

    def verify_hdf5_file(self, filepath: Union[str, pl.Path]) -> Dict[str, Any]:
        """
        Verify an HDF5 file and return summary information.
        Args:
            filepath: Path to HDF5 file to verify
        Returns:
            Dictionary with verification results
        """
        try:
            from snapshots.snapshot_serializer import CascadeHDF5Serializer

            serializer = CascadeHDF5Serializer(logger=self.logger)
            return serializer.verify_saved_network(filepath)
        except Exception as e:
            self.logger.error(f"CascadeCorrelationNetwork: Error verifying HDF5 file: {e}")
            import traceback

            self.logger.debug(traceback.format_exc())
            return {"valid": False, "error": str(e)}

    #################################################################################################################################################################################################
    # Public Method to validate the training process
    #################################################################################################################################################################################################
    # def validate_training(
    #     self,
    #     epoch: int = 0,
    #     max_epochs: int = 0,
    #     patience_counter: int = 0,
    #     early_stopping: bool = True,
    #     train_accuracy: float = 0.0,
    #     train_loss: float = float("inf"),
    #     best_value_loss: float = 9999999.9,
    #     x_train: torch.Tensor = None,
    #     y_train: torch.Tensor = None,
    #     x_val: torch.Tensor = None,
    #     y_val: torch.Tensor = None,
    # ) -> (bool, int, float, torch.Tensor, float, float):
    def validate_training(
        self,
        validate_training_inputs: ValidateTrainingInputs,
    ) -> ValidateTrainingResults:
        """
        Description:
            Validate the training process by checking the validation loss and accuracy.
        Args:
            validate_training_inputs: ValidateTrainingInputs dataclass containing:
                - iteration: Current iteration number
                - max_iterations: Maximum number of iterations
                - patience_counter: Counter for early stopping patience
                - early_stopping: Whether to use early stopping
                - train_accuracy: Training accuracy
                - train_loss: Training loss
                - best_value_loss: Best validation loss seen so far
                - x_train: Training input tensor
                - y_train: Training target tensor
                - x_val: Validation input tensor
                - y_val: Validation target tensor
        Raises:
            ValueError: If the input tensors are not valid
        Returns:
            ValidateTrainingResults dataclass containing:
                - early_stop: Whether to stop training early
                - patience_counter: Updated patience counter
                - best_value_loss: Updated best validation loss
                - value_output: Output from the validation set
                - value_loss: Validation loss
                - value_accuracy: Validation accuracy
        """
        # Unpack the dataclass fields
        iteration = validate_training_inputs.iteration
        max_iterations = validate_training_inputs.max_iterations
        patience_counter = validate_training_inputs.patience_counter
        early_stopping = validate_training_inputs.early_stopping
        train_accuracy = validate_training_inputs.train_accuracy
        train_loss = validate_training_inputs.train_loss
        best_value_loss = validate_training_inputs.best_value_loss
        x_train = validate_training_inputs.x_train
        y_train = validate_training_inputs.y_train
        x_val = validate_training_inputs.x_val
        y_val = validate_training_inputs.y_val

        self.logger.trace("CascadeCorrelationNetwork: validate_training: Starting validation of the training process.")
        early_stop_flag = False
        value_output = 0
        value_loss = float("inf")
        value_accuracy = 0.0
        best_value_loss = best_value_loss if best_value_loss is not None else 9999999.9
        self.logger.debug(f"CascadeCorrelationNetwork: validate_training: Iteration {iteration}, Max Iterations: {max_iterations}, Early Stopping: {early_stopping}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}, Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}")

        # Validate input tensors
        self.logger.debug(f"CascadeCorrelationNetwork: validate_training: X Train: {x_train}, Y Train: {y_train}, X Val: {x_val}, Y Val: {y_val}")
        if x_val is not None and y_val is not None:

            # Validate the model on the validation set
            with torch.no_grad():
                value_output = self.forward(x_val)
                value_loss = nn.MSELoss()(value_output, y_val).item()
            self.history["value_loss"].append(value_loss)

            # Calculate validation accuracy
            value_accuracy = self.calculate_accuracy(x_val, y_val)
            self.history["value_accuracy"].append(value_accuracy)
            self.logger.info("CascadeCorrelationNetwork: validate_training: " f"Iteration {iteration} - Train Loss: {train_loss:.6f}, Val Loss: {value_loss:.6f}, " f"Train Acc: {train_accuracy:.4f}, Val Acc: {value_accuracy:.4f}, " f"Units: {len(self.hidden_units)}")

            # Check for early stopping conditions
            # TODO: Consider using named tuple or dataclass for return values
            early_stop, patience_counter, best_value_loss = self.evaluate_early_stopping(
                iteration=iteration,
                max_iterations=max_iterations,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                early_stopping=early_stopping,
                value_loss=value_loss,
                best_value_loss=best_value_loss,
                patience_counter=patience_counter,
            )
            self.logger.verbose(f"CascadeCorrelationNetwork: validate_training: Early Stopping: {early_stopping}")
            self.logger.verbose(f"CascadeCorrelationNetwork: validate_training: Early Stop: {early_stop}")
            self.logger.verbose(f"CascadeCorrelationNetwork: validate_training: Iteration: {iteration}, Early Stop: {early_stop}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}")

            # early_stop_flag = True if early_stopping and early_stop else False
            early_stop_flag = early_stopping and early_stop
            self.logger.info(f"CascadeCorrelationNetwork: validate_training: Stop Training Early: {early_stop} and Early Stopping: {early_stopping}: {early_stopping and early_stop}")
            self.logger.info(f"CascadeCorrelationNetwork: validate_training: Early Stopping: {early_stop_flag}, Patience Counter: {patience_counter}, Best Val Loss: {best_value_loss:.6f}")
            self.logger.verbose(f"CascadeCorrelationNetwork: validate_training: Value Output: {value_output}, Value Loss: {value_loss:.6f}, Value Accuracy: {value_accuracy:.4f}")

        else:
            # No validation data — use training loss for early stopping
            if early_stopping:
                if train_loss < best_value_loss - self.convergence_threshold:
                    best_value_loss = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                patience_exhausted = patience_counter >= self.patience
                max_units_reached = self.check_hidden_units_max()
                train_accuracy_reached = self.check_training_accuracy(
                    train_accuracy=train_accuracy,
                    accuracy_target=self.target_accuracy,
                )
                early_stop_flag = patience_exhausted or max_units_reached or train_accuracy_reached
                self.logger.info(f"CascadeCorrelationNetwork: validate_training: " f"Iteration {iteration} (no val data) - Train Loss: {train_loss:.6f}, " f"Train Acc: {train_accuracy:.4f}, Patience: {patience_counter}/{self.patience}, " f"Early Stop: {early_stop_flag}")
        self.logger.verbose(f"CascadeCorrelationNetwork: validate_training: Iteration {iteration}, Early Stop: {early_stop_flag}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}, Value Output: {value_output}, Value Loss: {value_loss:.6f}, Value Accuracy: {value_accuracy:.4f}")
        self.logger.trace("CascadeCorrelationNetwork: validate_training: Completed validation of the training process.")

        return ValidateTrainingResults(
            early_stop=early_stop_flag,
            patience_counter=patience_counter,
            best_value_loss=best_value_loss,
            value_output=value_output,
            value_loss=value_loss,
            value_accuracy=value_accuracy,
        )

    #################################################################################################################################################################################################
    # Public Method to evaluate early stopping conditions
    # This method checks if the training should stop early based on validation loss, patience, and other criteria
    def evaluate_early_stopping(
        self,
        iteration: int = 0,
        max_iterations: int = 0,
        train_loss: float = float("inf"),
        train_accuracy: float = 0.0,
        early_stopping: bool = True,
        value_loss: float = float("inf"),
        best_value_loss: float = float("inf"),
        patience_counter: int = 0,
        # ) -> (bool, int, float):  # Original - invalid tuple syntax
    ) -> tuple[bool, int, float]:
        """
        Description:
            Evaluate early stopping conditions to determine if the training should stop.
        Args:
            iteration: Current iteration number
            max_iterations: Maximum number of iterations
            train_loss: Training loss
            train_accuracy: Training accuracy
            early_stopping: Whether to use early stopping
            value_loss: Validation loss
            best_value_loss: Best validation loss
            patience_counter: Patience counter
        Notes:
            - Early stopping is based on validation loss and patience
            - Training stops if patience is exhausted, maximum hidden units are reached, or perfect accuracy is achieved
            - If early stopping is not enabled, this method will always return (False, 0, float('inf'))
            - If early stopping is enabled, this method will return (True, updated_patience_counter, updated_best_value_loss)
            - This method does not update the models parameters
        Returns:
            bool: Whether early stopping should be triggered
            int: Updated patience counter
            float: Updated best validation loss
        """
        # Early stopping
        self.logger.trace("CascadeCorrelationNetwork: evaluate_early_stopping: Starting evaluation of early stopping conditions.")

        # Initialize variables
        patience_exhausted = False
        max_units_reached = False
        train_accuracy_reached = False
        if early_stopping:

            # Check if we've reached the end of our patience
            # TODO: Consider using named tuple or dataclass for return values
            patience_exhausted, patience_counter, best_value_loss = self.check_patience(
                patience_counter=patience_counter,
                value_loss=value_loss,
                best_value_loss=best_value_loss,
            )
            self.logger.info(f"CascadeCorrelationNetwork: evaluate_early_stopping: Iteration {iteration} - Patience Counter: {patience_counter}, Value Loss: {value_loss}, Best Val Loss: {best_value_loss:.6f}")
            if patience_exhausted:
                self.logger.info(f"CascadeCorrelationNetwork: evaluate_early_stopping: Patience Exhausted: {patience_exhausted}, Early stopping triggered after {iteration} iterations")
            else:
                self.logger.info(f"CascadeCorrelationNetwork: evaluate_early_stopping: Iteration {iteration} - Train Loss: {train_loss:.6f}, " f"Train Acc: {train_accuracy:.4f}, Units: {len(self.hidden_units)}")

            # Check if we've reached the maximum number of hidden units
            if max_units_reached := self.check_hidden_units_max():
                self.logger.info(f"CascadeCorrelationNetwork: evaluate_early_stopping: Reached maximum number of hidden units: {max_units_reached}, stopping training")

            # Check if we've achieved perfect accuracy
            if train_accuracy_reached := self.check_training_accuracy(
                train_accuracy=train_accuracy,
                accuracy_target=self.target_accuracy,
            ):
                self.logger.info(f"CascadeCorrelationNetwork: evaluate_early_stopping: Training accuracy reached target: {train_accuracy:.4f} >= 0.999")

        early_stop = early_stopping and (train_accuracy_reached or max_units_reached or patience_exhausted)
        self.logger.info(f"CascadeCorrelationNetwork: evaluate_early_stopping: Early Stopping: {early_stop}, Patience Counter: {patience_counter}, Best Val Loss: {best_value_loss:.6f}")
        self.logger.trace("CascadeCorrelationNetwork: evaluate_early_stopping: Completed evaluation of early stopping conditions.")

        # TODO: Consider using named tuple or dataclass for return values
        return (early_stop, patience_counter, best_value_loss)

    #################################################################################################################################################################################################
    # Public Method to check patience limit
    # This method checks if the patience limit is reached based on validation loss
    def check_patience(
        self,
        patience_counter: int = 0,
        value_loss: float = float("inf"),
        best_value_loss: float = float("inf"),
        # ) -> (bool, int, float):  # Original - invalid tuple syntax
    ) -> tuple[bool, int, float]:
        """
        Description:
            Check if patience limit is reached based on validation loss.
        Args:
            patience_counter: Patience counter
            value_loss: Validation loss
            best_value_loss: Best validation loss
        Notes:
            - Patience counter is incremented if validation loss does not improve
            - Patience counter is reset if validation loss improves
            - If patience counter exceeds the patience limit, training should stop
            - This method does not update the models parameters
        Returns:
            bool: Whether patience limit is reached
            int: Updated patience counter
            float: Best validation loss
        """
        # Check if validation loss improved
        self.logger.trace("CascadeCorrelationNetwork: check_patience: Starting to check patience limit.")
        self.logger.verbose(f"CascadeCorrelationNetwork: check_patience: Current Value Loss: {value_loss:.6f}, Best Value Loss: {best_value_loss:.6f}, Patience Counter: {patience_counter}")
        if value_loss < best_value_loss - self.convergence_threshold:
            best_value_loss = value_loss
            patience_counter = 0
        else:
            patience_counter += 1
        self.logger.info(f"CascadeCorrelationNetwork: check_patience: Patience counter: {patience_counter}, Best Validation Loss: {best_value_loss:.6f}")

        # Check if patience limit is reached
        if patience_exhausted := (patience_counter >= self.patience):
            self.logger.info(f"CascadeCorrelationNetwork: check_patience: Patience limit reached: {patience_counter} >= {self.patience}")
        self.logger.debug(f"CascadeCorrelationNetwork: check_patience: Patience Exhausted: {patience_exhausted}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}")
        self.logger.trace("CascadeCorrelationNetwork: check_patience: Completed checking patience limit.")

        # TODO: Consider using named tuple or dataclass for return values
        return (patience_exhausted, patience_counter, best_value_loss)

    #################################################################################################################################################################################################
    # Public Methods to check conditions for training
    def check_hidden_units_max(self) -> bool:
        """
        Description:
            Check if reached the maximum number of hidden units
        Args:
            None
        Notes:
            - This method checks the length of the hidden_units list against the max_hidden_units attribute
            - If the length of hidden_units is greater than or equal to max_hidden_units, the method returns True
            - If the length of hidden_units is less than max_hidden_units, the method returns False
        Returns:
            bool: Whether reached max hidden units
        """
        # Check if we've reached max hidden units
        self.logger.trace("CascadeCorrelationNetwork: check_hidden_units_max: Starting to check if max hidden units reached.")
        max_units_reached = len(self.hidden_units) >= self.max_hidden_units
        self.logger.info(f"CascadeCorrelationNetwork: check_hidden_units_max: Current hidden units: {max_units_reached}, Max allowed: {self.max_hidden_units}")
        if max_units_reached:
            self.logger.info(f"CascadeCorrelationNetwork: check_hidden_units_max: Reached maximum number of hidden units: {self.max_hidden_units}")
        self.logger.trace("CascadeCorrelationNetwork: check_hidden_units_max: Completed checking if max hidden units reached.")
        return max_units_reached

    #################################################################################################################################################################################################
    # Public Method to check if training accuracy has reached the target
    # This method checks if the training accuracy has reached the target accuracy
    def check_training_accuracy(
        self,
        train_accuracy: float = 0.0,
        accuracy_target: float = 0.999,
    ) -> bool:
        """
        Description:
            Check if training accuracy has reached the target.
            This method compares the current training accuracy with the target accuracy.
            If the training accuracy is greater than or equal to the target accuracy, the method returns True.
            If the training accuracy is not greater than or equal to the target accuracy, the method returns False.
        Args:
            train_accuracy: Current training accuracy
            accuracy_target: Target accuracy to reach
        Returns:
            bool: Whether target accuracy has been reached
        """
        self.logger.trace("CascadeCorrelationNetwork: check_training_accuracy: Starting to check if training accuracy has reached the target.")
        if train_accuracy_reached := (train_accuracy >= accuracy_target):
            self.logger.info(f"CascadeCorrelationNetwork: check_training_accuracy: Reached target training accuracy: {train_accuracy:.4f} >= {accuracy_target:.4f}")
        self.logger.debug(f"CascadeCorrelationNetwork: check_training_accuracy: Current Training Accuracy: {train_accuracy:.4f}, Target Accuracy: {accuracy_target:.4f}")
        self.logger.trace("CascadeCorrelationNetwork: check_training_accuracy: Completed checking if training accuracy has reached the target.")
        return train_accuracy_reached

    ##################################################################################################################################################################################################
    # Public Method to calculate classification accuracy
    # This method calculates the classification accuracy of the network
    # It compares the predicted output with the target output
    def calculate_accuracy(
        self,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
    ) -> float:
        """
        Designation:
            This method takes input and target tensors, passes them through the network, and then calculates the accuracy based on the predicted and target outputs.
            The accuracy is calculated as the percentage of correct predictions over the total number of predictions.
        Args:
            x: Input tensor
            y: Target tensor
        Notes:
            - The accuracy is calculated using the custom `_accuracy` method
        Returns:
            Classification accuracy: float
        """
        self.logger.trace("CascadeCorrelationNetwork: calculate_accuracy: Starting to calculate accuracy.")
        x = (x, torch.empty(0, self.input_size))[x is None]
        y = (y, torch.empty(0, self.output_size))[y is None]
        accuracy = 0.0

        # Validate input tensors
        if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            self.logger.error(f"CascadeCorrelationNetwork: calculate_accuracy: Input and target tensors must be of type torch.Tensor. Input (x): {type(x)}, Target (y): {type(y)}")
            raise ValueError("CascadeCorrelationNetwork: calculate_accuracy: Input and target tensors must be of type torch.Tensor.")
        # elif x.shape[-1] != y.shape[-1]:
        #     self.logger.error( f"CascadeCorrelationNetwork: calculate_accuracy: Input shape: {x.shape}, Target shape: {y.shape}")
        #     raise ValueError( "CascadeCorrelationNetwork: calculate_accuracy: Input and target tensors must have the same number of features.")
        elif x.shape[0] != y.shape[0]:
            self.logger.error(f"CascadeCorrelationNetwork: calculate_accuracy: Input and target tensors must have compatible shapes. Input (x): {x.shape}, Target (y): {y.shape}, input size: {self.input_size}, output size: {self.output_size}")
            raise ValueError("CascadeCorrelationNetwork: calculate_accuracy: Input and target tensors must have compatible shapes.")
        else:
            self.logger.debug(f"CascadeCorrelationNetwork: calculate_accuracy: Validated input shape: {x.shape}, Target shape: {y.shape}")

            # Calculating accuracy
            self.logger.debug(f"CascadeCorrelationNetwork: calculate_accuracy: Calculating accuracy for input shape: {x.shape}, target shape: {y.shape}")
            with torch.no_grad():
                output = self.forward(x)
                self.logger.debug(f"CascadeCorrelationNetwork: calculate_accuracy: Output shape: {output.shape}")

                # Validate Output Tensor
                if not isinstance(output, torch.Tensor):
                    self.logger.error(f"CascadeCorrelationNetwork: calculate_accuracy: Output tensor must be of type torch.Tensor. Output: Type: {type(output)}")
                    raise ValueError("CascadeCorrelationNetwork: calculate_accuracy: Output tensor must be of type torch.Tensor.")
                elif output.shape[-1] != y.shape[-1]:
                    self.logger.error(f"CascadeCorrelationNetwork: calculate_accuracy: Output shape: {output.shape}, Target shape: {y.shape}")
                    raise ValueError("CascadeCorrelationNetwork: calculate_accuracy: Output and target tensors must have the same number of features.")
                elif output.shape[0] != y.shape[0]:
                    self.logger.error(f"CascadeCorrelationNetwork: calculate_accuracy: Output and target tensors must have compatible shapes. Output Tensor: {output.shape}, Target (y): {y.shape}, Output size: {output.size()}, Target size: {self.output_size}")
                    raise ValueError("CascadeCorrelationNetwork: calculate_accuracy: Output and target tensors must have compatible shapes.")
                else:
                    self.logger.debug(f"CascadeCorrelationNetwork: calculate_accuracy: Validated Output shape: {output.shape}, Target shape: {y.shape}")
                accuracy = self._accuracy(y=y, output=output)
            self.logger.info(f"CascadeCorrelationNetwork: calculate_accuracy: Calculated accuracy: {accuracy:.4f}, Percentage: {accuracy * 100:.2f}%")

        # Returning accuracy
        self.logger.trace("CascadeCorrelationNetwork: calculate_accuracy: Completed calculating accuracy.")
        return accuracy

    #################################################################################################################################################################################################
    def _accuracy(
        self,
        y: torch.Tensor = None,
        output: torch.Tensor = None,
    ) -> float:
        """
        Description:
            Private method to calculate accuracy.
            This method is used internally to calculate the accuracy of the network.
        Args:
            target: Target output
            output: Raw output from the network
        Notes:
            - For multi-class (output_size > 1), assumes one-hot encoded tensors and uses argmax.
            - For binary classification (output_size == 1), uses a 0.5 threshold.
            - The accuracy is calculated as the percentage of correct predictions over the total number of predictions.
            - If either the target or output tensor is missing, an error is raised.
        Returns:
            Accuracy as a float
        """
        self.logger.trace("CascadeCorrelationNetwork: _accuracy: Starting to calculate accuracy.")

        # Validate input tensors
        if y is None or output is None:
            self.logger.error("CascadeCorrelationNetwork: _accuracy: Missing required tensors for accuracy calculation.")
            raise ValueError("CascadeCorrelationNetwork: _accuracy: Missing required tensors for accuracy calculation.")
        elif not (isinstance(y, torch.Tensor) and isinstance(output, torch.Tensor)):
            self.logger.error("CascadeCorrelationNetwork: _accuracy: All inputs must be torch tensors.")
            raise TypeError("CascadeCorrelationNetwork: _accuracy: All inputs must be torch tensors.")
        elif y.shape[0] != output.shape[0]:
            self.logger.error(f"CascadeCorrelationNetwork: _accuracy: Output and Target tensors must have the same number of samples. Got {y.shape[0]} and {output.shape[0]}.")
            raise ValueError("CascadeCorrelationNetwork: _accuracy: Output and Target tensors must have the same number of samples.")
        self.logger.debug(f"CascadeCorrelationNetwork: _accuracy: Input shape: {y.shape}, Output shape: {output.shape}")
        self.logger.verbose(f"CascadeCorrelationNetwork: _accuracy: Input shape: {y.shape}")
        self.logger.verbose(f"CascadeCorrelationNetwork: _accuracy: Output shape: {output.shape}")

        # Handle empty batch case
        if y.shape[0] == 0:
            self.logger.debug("CascadeCorrelationNetwork: _accuracy: Empty batch, returning NaN for accuracy")
            return float("nan")

        # Find predicted and target values
        if output.shape[1] == 1:
            # Binary classification with single output: threshold at 0.5
            predicted = (output.squeeze(1) > 0.5).long()
            self.logger.verbose(f"CascadeCorrelationNetwork: _accuracy: Predicted shape (binary): {predicted.shape}")
            target_binary = (y.squeeze(1) > 0.5).long()
            self.logger.verbose(f"CascadeCorrelationNetwork: _accuracy: Target shape (binary): {target_binary.shape}")
            correct = (predicted == target_binary).sum().item()
            self.logger.verbose(f"CascadeCorrelationNetwork: _accuracy: Number of correct predictions: {correct}, Total samples: {len(target_binary)}")
            accuracy = correct / len(target_binary)
        else:
            # Multi-class classification: use argmax on one-hot encoded tensors
            predicted = torch.argmax(output, dim=1)
            self.logger.verbose(f"CascadeCorrelationNetwork: _accuracy: Predicted shape: {predicted.shape}")
            target = torch.argmax(y, dim=1)
            self.logger.verbose(f"CascadeCorrelationNetwork: _accuracy: Target shape: {target.shape}")
            correct = (predicted == target).sum().item()
            self.logger.verbose(f"CascadeCorrelationNetwork: _accuracy: Number of correct predictions: {correct}, Total samples: {len(target)}")
            accuracy = correct / len(target)
        self.logger.info(f"CascadeCorrelationNetwork: _accuracy: Calculated accuracy: {accuracy:.4f}, Percentage: {accuracy * 100:.4f}%")
        self.logger.trace("CascadeCorrelationNetwork: _accuracy: Completed calculating accuracy.")
        return accuracy

    #################################################################################################################################################################################################
    # Public Method to make predictions
    # This method uses the forward method to get the output of the network
    # It is used to make predictions on new data
    def predict(self, x: torch.Tensor) -> torch.Tensor:  # sourcery skip: class-extract-method
        """
        Make predictions using the trained network.
        Args:
            x: Input tensor (batch_size, input_features)
        Raises:
            ValidationError: If input tensor is invalid or has wrong shape
        Returns:
            Predicted output tensor (batch_size, output_features)
        """
        # Validate input tensor
        self._validate_tensor_input(x, "x")
        self._validate_tensor_shapes(x, expected_input_features=self.input_size)

        # Return the predicted output
        self.logger.debug(f"CascadeCorrelationNetwork: predict: Input shape: {x.shape}")
        self.logger.trace("CascadeCorrelationNetwork: predict: Starting to make predictions.")
        with torch.no_grad():
            predicted_value = self.forward(x)
            self.logger.trace("CascadeCorrelationNetwork: predict: Finished making predictions.")
        self.logger.debug(f"CascadeCorrelationNetwork: predict: Predicted shape: {predicted_value.shape}, Predicted: {predicted_value}")
        return predicted_value

    #################################################################################################################################################################################################
    # Public Method to predict class labels
    # This method predicts the class labels for the input tensor
    # It uses the forward method to get the output and then applies argmax to get the class labels
    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels using the trained network.
        Args:
            x: Input tensor (batch_size, input_features)
        Raises:
            ValidationError: If input tensor is invalid or has wrong shape
        Returns:
            Predicted class labels tensor (batch_size,)
        """
        # Validate input tensor
        self._validate_tensor_input(x, "x")
        self._validate_tensor_shapes(x, expected_input_features=self.input_size)

        # Return the predicted class labels
        self.logger.debug(f"CascadeCorrelationNetwork: predict_classes: Input shape: {x.shape}")
        self.logger.trace("CascadeCorrelationNetwork: predict_classes: Starting to predict class labels.")
        with torch.no_grad():
            output = self.forward(x)
            prediction = torch.argmax(output, dim=1)
            self.logger.info(f"CascadeCorrelationNetwork: predict_classes: Predicted class labels shape: {prediction.shape}, Prediction: {prediction}")
        return prediction

    #################################################################################################################################################################################################
    # Public Method to print a summary of the network architecture
    # This method prints the input size, output size, number of hidden units, and training parameters
    # It also prints the details of each hidden unit including its weights, bias, and correlation
    def summary(self) -> None:
        """
        Description:
            Print a summary of the network architecture.
        Notes:
            - Displays input size, output size, number of hidden units, and training parameters
            - Displays details of each hidden unit including weights, bias, and correlation
        Args:
            None
        Returns:
            None
        """
        self.logger.trace("CascadeCorrelationNetwork: summary: Starting to print network summary.")
        self.logger.info("CascadeCorrelationNetwork: summary: Display Cascade Correlation Network Summary:")
        self.logger.info(f"CascadeCorrelationNetwork: summary: Input size: {self.input_size}")
        self.logger.info(f"CascadeCorrelationNetwork: summary: Output size: {self.output_size}")
        self.logger.info(f"CascadeCorrelationNetwork: summary: Number of hidden units: {len(self.hidden_units)}")

        # Display hidden unit info if present
        if self.hidden_units:
            self.logger.info("CascadeCorrelationNetwork: summary: Hidden Units:\n")
            for i, unit in enumerate(self.hidden_units):
                self.logger.info(f"CascadeCorrelationNetwork: summary:   Unit {i+1}:")
                self.logger.info(f"CascadeCorrelationNetwork: summary:     Input size: {len(unit['weights'])}")
                self.logger.info(f"CascadeCorrelationNetwork: summary:     Correlation: {unit['correlation']:.6f}")

        # Display Training Parameters
        self.logger.info("CascadeCorrelationNetwork: summary: Training Parameters:")
        self.logger.info(f"CascadeCorrelationNetwork: summary:   Learning rate: {self.learning_rate}")
        self.logger.info(f"CascadeCorrelationNetwork: summary:   Candidate pool size: {self.candidate_pool_size}")
        self.logger.info(f"CascadeCorrelationNetwork: summary:   Correlation threshold: {self.correlation_threshold}")

        # Display final training accuracy if attribute exists
        if self.history["train_accuracy"]:
            self.logger.info(f"CascadeCorrelationNetwork: summary: Final training accuracy:\n{self.history['train_accuracy'][-1]:.6f}")

        # Display final value accuracy if validation was used
        if "value_accuracy" in self.history and self.history["value_accuracy"]:
            self.logger.info(f"CascadeCorrelationNetwork: summary: Final validation accuracy:\n{self.history['value_accuracy'][-1]:.6f}")
        self.logger.trace("CascadeCorrelationNetwork: summary: Completed printing network summary.")

    #################################################################################################################################################################################################
    # Define public methods for plotting the dataset, decision boundary and training history
    # These methods now delegate to the CascadeCorrelationPlotter class
    #################################################################################################################################################################################################
    @staticmethod
    def plot_dataset(
        x: torch.Tensor,
        y: torch.Tensor,
        title: str = "Training Dataset",
    ) -> None:
        """
        Plot the training dataset (static method for backward compatibility).
        Args:
            x: Input tensor (must have 2 features for 2D plotting)
            y: Target tensor (one-hot encoded labels)
            title: Plot title
        Raises:
            ValidationError: If input tensors are not valid for plotting
        """
        CascadeCorrelationPlotter.plot_dataset(x, y, title)

    def plot_decision_boundary(
        self,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        title: str = "Decision Boundary",
        async_plot: bool = True,
    ) -> Optional[mp.Process]:
        """
        Plot the decision boundary of the network.
        Args:
            x: Input tensor (must have 2 features for 2D plotting)
            y: Target tensor (one-hot encoded labels)
            title: Plot title
            async_plot: If True, plot in separate process (non-blocking)
        Returns:
            Process object if async_plot=True, otherwise None
        Raises:
            ValidationError: If input tensors are not valid for plotting
        """
        if async_plot:
            # Use spawn context for plotting to avoid pickling issues, Forkserver context requires functions to be picklable at module level
            plot_ctx = mp.get_context("spawn")
            plot_process = plot_ctx.Process(
                target=_plot_decision_boundary_worker,
                args=(self, x, y, title),
                daemon=True,
                name="PlotDecisionBoundary",
            )
            plot_process.start()
            self.logger.info(f"CascadeCorrelationNetwork: plot_decision_boundary: Started plotting process PID: {plot_process.pid}")
            return plot_process
        else:
            self.plotter.plot_decision_boundary(self, x, y, title)
            return None

    def plot_training_history(self, async_plot: bool = True) -> Optional[mp.Process]:
        """
        Plot the training history of the network.
        Args:
            async_plot: If True, plot in separate process (non-blocking)
        Returns:
            Process object if async_plot=True, otherwise None
        Raises:
            ValidationError: If training history is empty or invalid
        """
        if async_plot:
            # Use spawn context for plotting to avoid pickling issues
            # Forkserver context requires functions to be picklable at module level
            plot_ctx = mp.get_context("spawn")
            plot_process = plot_ctx.Process(
                target=_plot_training_history_worker,
                args=(self.history,),
                daemon=True,
                name="PlotTrainingHistory",
            )
            plot_process.start()
            self.logger.info(f"CascadeCorrelationNetwork: plot_training_history: Started plotting process PID: {plot_process.pid}")
            return plot_process
        else:
            self.plotter.plot_training_history(self.history)
            return None

    #################################################################################################################################################################################################
    # Define private method to generate a new uuid for the CascadeCorrelationNetwork class
    def _generate_uuid(self) -> str:
        """
        Description:
            This method is used to generate a new UUID for the CascadeCorrelationNetwork class.
        Args:
            self: The instance of the class
        Notes:
            - This method uses the uuid4 function from the uuid module to generate a new UUID.
            - The generated UUID is stored in the `uuid` attribute of the class.
            - The generated UUID is then returned.
        Returns:
            str: The generated UUID.
        """
        logger = self.logger if hasattr(self, "logger") and self.logger is not None else Logger
        logger.trace("CascadeCorrelationNetwork: _generate_uuid: Inside the CascadeCorrelationNetwork class Generate UUID method")
        new_uuid = str(uuid.uuid4())
        logger.debug(f"CascadeCorrelationNetwork: _generate_uuid: UUID: {new_uuid}")
        logger.trace("CascadeCorrelationNetwork: _generate_uuid: Completed the CascadeCorrelationNetwork class Generate UUID method")
        return new_uuid

    ####################################################################################################################################
    # Define CascadeCorrelationNetwork class Setters
    ####################################################################################################################################
    def set_candidate_training_queue_authkey(self, candidate_training_queue_authkey: bytes = None):
        self.candidate_training_queue_authkey = candidate_training_queue_authkey

    def set_candidate_training_queue_address(self, candidate_training_queue_address: str = None):
        self.candidate_training_queue_address = candidate_training_queue_address

    def set_candidate_training_tasks_queue_timeout(self, candidate_training_tasks_queue_timeout: int = None):
        self.candidate_training_tasks_queue_timeout = candidate_training_tasks_queue_timeout

    def set_candidate_training_shutdown_timeout(self, candidate_training_shutdown_timeout: int = None):
        self.candidate_training_shutdown_timeout = candidate_training_shutdown_timeout

    def set_activation_fn(self, activation_fn: str = None):
        self.activation_fn = activation_fn

    def set_activation_fn_no_diff(self, activation_fn_no_diff: str = None):
        self.activation_fn_no_diff = activation_fn_no_diff

    def set_candidate_epochs(self, candidate_epochs: int = None):
        self.candidate_epochs = candidate_epochs

    def set_candidate_pool_size(self, candidate_pool_size: int = None):
        self.candidate_pool_size = candidate_pool_size

    def set_candidate_unit(self, candidate_unit: CandidateUnit = None):
        self.candidate_unit = candidate_unit

    def set_correlation_threshold(self, correlation_threshold: float = None):
        self.correlation_threshold = correlation_threshold

    def set_display_frequency_epoch(self, display_frequency_epoch: int = None):
        self.display_frequency_epoch = display_frequency_epoch

    def set_display_frequency_units(self, display_frequency_units: int = None):
        self.display_frequency_units = display_frequency_units

    def set_generate_plots(self, generate_plots: bool = None):
        self.generate_plots = generate_plots

    def set_hidden_units(self, hidden_units: list = None):
        self.hidden_units = hidden_units

    def set_history(self, history: dict = None):
        self.history = history

    def set_input_size(self, input_size: int = None):
        self.input_size = input_size

    def set_learning_rate(self, learning_rate: float = None):
        """Set learning rate with validation."""
        if learning_rate is not None:
            self._validate_numeric_parameter(learning_rate, "learning_rate", min_val=0.0, max_val=10.0)
        self.learning_rate = learning_rate

    def set_max_hidden_units(self, max_hidden_units: int = None):
        """Set maximum hidden units with validation."""
        if max_hidden_units is not None:
            self._validate_positive_integer(max_hidden_units, "max_hidden_units")
        self.max_hidden_units = max_hidden_units

    def set_output_bias(self, output_bias: float = None):
        """Set output bias with validation."""
        if output_bias is not None and not isinstance(output_bias, (int, float, torch.Tensor)):
            raise ValidationError(f"output_bias must be numeric or tensor, got {type(output_bias)}")
        self.output_bias = output_bias

    def set_output_epochs(self, output_epochs: int = None):
        """Set output epochs with validation."""
        if output_epochs is not None:
            self._validate_positive_integer(output_epochs, "output_epochs")
        self.output_epochs = output_epochs

    def set_output_size(self, output_size: int = None):
        self.output_size = output_size

    def set_output_weights(self, output_weights: list = None):
        self.output_weights = output_weights

    def set_patience(self, patience: int = None):
        self.patience = patience

    def set_random_value_scale(self, random_value_scale: float = None):
        self.random_value_scale = random_value_scale

    def set_status_display_frequency(self, status_display_frequency: int = None):
        self.status_display_frequency = status_display_frequency

    def set_uuid(self, uuid: str = None):
        """
        Description:
            This method sets the UUID for the CascadeCorrelationNetwork class.  If no UUID is provided, a new UUID will be generated.
        Args:
            uuid (str): The UUID to be set. If None, a new UUID will be generated.
        Returns:
            None
        """
        logger = self.logger if hasattr(self, "logger") and self.logger is not None else Logger

        logger.trace("CascadeCorrelationNetwork: set_uuid: Starting to set UUID for CascadeCorrelationNetwork class")
        logger.debug(f"CascadeCorrelationNetwork: set_uuid: Setting UUID to: {uuid}")
        if not hasattr(self, "uuid") or self.uuid is None:
            self.uuid = (uuid, self._generate_uuid())[uuid is None]  # Generate a new UUID if none is provided
        else:
            error_msg = f"UUID already set: {self.uuid}. Cannot change UUID after initialization."
            logger.fatal(f"CascadeCorrelationNetwork: set_uuid: Fatal Error: {error_msg}")
            raise ConfigurationError(error_msg)
        logger.debug(f"CascadeCorrelationNetwork: set_uuid: UUID set to: {self.uuid}")
        logger.trace("CascadeCorrelationNetwork: set_uuid: Completed setting UUID for CascadeCorrelationNetwork class")

    ####################################################################################################################################
    # Define CascadeCorrelationNetwork class Getters
    ####################################################################################################################################
    def get_uuid(self) -> str:
        """
        Description:
            This method returns the UUID for the CascadeCorrelationNetwork class.
        Args:
            self: The instance of the class.
        Notes:
            - If the UUID is not set, it will generate a new UUID using the set_uuid method.
            - The generated UUID is then returned.
        Returns:
            str: The UUID for the CascadeCorrelationNetwork class.
        """
        self.logger.trace("CascadeCorrelationNetwork: get_uuid: Starting to get UUID for CascadeCorrelationNetwork class")
        self.logger.debug(f"CascadeCorrelationNetwork: get_uuid: Current UUID: {getattr(self, 'uuid', None)}")

        # Ensure UUID is set:  if not, generate a new one
        if not hasattr(self, "uuid"):
            self.set_uuid()  # Ensure UUID is set if not already
            self.logger.debug("CascadeCorrelationNetwork: get_uuid: UUID was not set, generated a new one.")

        # Return the UUID
        self.logger.debug(f"CascadeCorrelationNetwork: get_uuid: Returning UUID: {self.uuid}")
        self.logger.trace("CascadeCorrelationNetwork: get_uuid: Completed getting UUID for CascadeCorrelationNetwork class")
        return self.uuid

    def get_candidate_training_queue_authkey(self):
        return self.candidate_training_queue_authkey if hasattr(self, "candidate_training_queue_authkey") else None

    def get_candidate_training_queue_address(self):
        return self.candidate_training_queue_address if hasattr(self, "candidate_training_queue_address") else None

    def get_candidate_training_tasks_queue_timeout(self):
        return self.candidate_training_tasks_queue_timeout if hasattr(self, "candidate_training_tasks_queue_timeout") else None

    def get_candidate_training_shutdown_timeout(self):
        return self.candidate_training_shutdown_timeout if hasattr(self, "candidate_training_shutdown_timeout") else None

    def get_activation_fn(self):
        return self.activation_fn if hasattr(self, "activation_fn") else None

    def get_activation_fn_no_diff(self):
        return self.activation_fn_no_diff if hasattr(self, "activation_fn_no_diff") else None

    def get_candidate_epochs(self):
        return self.candidate_epochs if hasattr(self, "candidate_epochs") else None

    def get_candidate_pool_size(self):
        return self.candidate_pool_size if hasattr(self, "candidate_pool_size") else None

    def get_candidate_unit(self) -> CandidateUnit:
        return self.candidate_unit if hasattr(self, "candidate_unit") else None

    def get_correlation_threshold(self):
        return self.correlation_threshold if hasattr(self, "correlation_threshold") else None

    def get_display_frequency_epoch(self):
        return self.display_frequency_epoch if hasattr(self, "display_frequency_epoch") else None

    def get_display_frequency_units(self):
        return self.display_frequency_units if hasattr(self, "display_frequency_units") else None

    def get_generate_plots(self):
        return self.generate_plots if hasattr(self, "generate_plots") else None

    def get_hidden_units(self):
        return self.hidden_units if hasattr(self, "hidden_units") else None

    def get_history(self):
        return self.history if hasattr(self, "history") else None

    def get_input_size(self):
        return self.input_size if hasattr(self, "input_size") else None

    def get_learning_rate(self):
        return self.learning_rate if hasattr(self, "learning_rate") else None

    def get_max_hidden_units(self):
        return self.max_hidden_units if hasattr(self, "max_hidden_units") else None

    def get_output_bias(self):
        return self.output_bias if hasattr(self, "output_bias") else None

    def get_output_epochs(self):
        return self.output_epochs if hasattr(self, "output_epochs") else None

    def get_output_size(self):
        return self.output_size if hasattr(self, "output_size") else None

    def get_output_weights(self):
        return self.output_weights if hasattr(self, "output_weights") else None

    def get_patience(self):
        return self.patience if hasattr(self, "patience") else None

    def get_random_value_scale(self):
        return self.random_value_scale if hasattr(self, "random_value_scale") else None

    def get_status_display_frequency(self):
        return self.status_display_frequency if hasattr(self, "status_display_frequency") else None
