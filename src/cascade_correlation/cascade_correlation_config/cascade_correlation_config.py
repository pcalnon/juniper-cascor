#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     cascade_correlation_config.py
# Author:        Paul Calnon
#
# Date Created:  2025-09-26
# Last Modified: 2026-01-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This file contains the config class for the Cascade Correlation Neural Network.
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
import pathlib
import uuid
from dataclasses import dataclass

from cascor_constants.constants import (  # _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT,
    _CASCADE_CORRELATION_NETWORK_AUTHKEY,
    _CASCADE_CORRELATION_NETWORK_BASE_MANAGER_ADDRESS,
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
    _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE,
    _CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT,
    _CASCADE_CORRELATION_NETWORK_STATUS_DISPLAY_FREQUENCY,
    _CASCADE_CORRELATION_NETWORK_TARGET_ACCURACY,
    _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT,
    _CASCADE_CORRELATION_NETWORK_WORKER_STANDBY_SLEEPYTIME,
    _CASCADE_CORRELATION_NETWORK_WORKER_THREAD_COUNT,
)
from log_config.log_config import LogConfig


#####################################################################################################################################################################################################
# Optimizer configuration dataclass
@dataclass
class OptimizerConfig:
    """Configuration for output layer optimizer."""

    optimizer_type: str = "Adam"  # Adam, SGD, RMSprop, AdamW, etc.
    learning_rate: float = 0.01
    momentum: float = 0.9  # For SGD, RMSprop
    beta1: float = 0.9  # For Adam variants
    beta2: float = 0.999  # For Adam variants
    weight_decay: float = 0.0
    epsilon: float = 1e-8
    # Adadelta
    rho: float = 0.9
    # Adagrad
    lr_decay: float = 0.0
    # Adam, AdamW
    amsgrad: bool = False
    # ASGD
    lambd: float = 1e-4
    alpha: float = 0.75
    t0: float = 1e6
    # LBFGS
    max_iter: int = 20
    max_eval: int = 25
    tolerance_grad: float = 1e-5
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: str = "strong_wolfe"
    # Rprop
    eta_min: float = 0.5
    eta_max: float = 1.2
    step_size_min: float = 1e-6
    step_size_max: float = 50.0


#####################################################################################################################################################################################################
# Configuration class for Cascade Correlation Network
class CascadeCorrelationConfig:
    """Configuration class for CascadeCorrelationNetwork to reduce constructor complexity."""

    def __init__(
        self,
        # Network architecture
        input_size: int = _CASCADE_CORRELATION_NETWORK_INPUT_SIZE,
        output_size: int = _CASCADE_CORRELATION_NETWORK_OUTPUT_SIZE,
        max_hidden_units: int = _CASCADE_CORRELATION_NETWORK_MAX_HIDDEN_UNITS,
        # Activation function
        activation_function_name: str = _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME,
        activation_functions_dict: dict = _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT,
        # Training parameters
        learning_rate: float = _CASCADE_CORRELATION_NETWORK_LEARNING_RATE,
        candidate_learning_rate: float = _CASCADE_CORRELATION_NETWORK_CANDIDATE_LEARNING_RATE,
        candidate_pool_size: int = _CASCADE_CORRELATION_NETWORK_CANDIDATE_POOL_SIZE,
        candidate_epochs: int = _CASCADE_CORRELATION_NETWORK_CANDIDATE_EPOCHS,
        epochs_max: int = _CASCADE_CORRELATION_NETWORK_EPOCHS_MAX,
        max_iterations: int = _CASCADE_CORRELATION_NETWORK_MAX_ITERATIONS,
        output_epochs: int = _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS,
        patience: int = _CASCADE_CORRELATION_NETWORK_PATIENCE,
        convergence_threshold: float = _CASCADE_CORRELATION_NETWORK_CONVERGENCE_THRESHOLD,
        candidate_convergence_threshold: float = _CASCADE_CORRELATION_NETWORK_CANDIDATE_CONVERGENCE_THRESHOLD,
        candidate_patience: int = _CASCADE_CORRELATION_NETWORK_CANDIDATE_PATIENCE,
        # Thresholds
        correlation_threshold: float = _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD,
        # Display and visualization
        display_frequency: int = _CASCADE_CORRELATION_NETWORK_DISPLAY_FREQUENCY,
        epoch_display_frequency: int = _CASCADE_CORRELATION_NETWORK_EPOCH_DISPLAY_FREQUENCY,
        candidate_display_frequency: int = _CASCADE_CORRELATION_NETWORK_CANDIDATE_DISPLAY_FREQUENCY,
        status_display_frequency: int = _CASCADE_CORRELATION_NETWORK_STATUS_DISPLAY_FREQUENCY,
        generate_plots: bool = _CASCADE_CORRELATION_NETWORK_GENERATE_PLOTS,
        # Random number generation
        random_seed: int = _CASCADE_CORRELATION_NETWORK_RANDOM_SEED,
        random_max_value: int = _CASCADE_CORRELATION_NETWORK_RANDOM_MAX_VALUE,
        sequence_max_value: int = _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE,
        random_value_scale: float = _CASCADE_CORRELATION_NETWORK_RANDOM_VALUE_SCALE,
        # Output weight initialization
        init_output_weights: str = _CASCADE_CORRELATION_NETWORK_INIT_OUTPUT_WEIGHTS,
        # CAN-010 / ENH-006 (Phase 6E Sprint A-2): output-layer optimizer
        # type. Mirrors to ``self.optimizer_config.optimizer_type`` so the
        # existing ``_create_optimizer`` registry dispatch picks it up at
        # the next output-training pass. Validation against the registry
        # happens inside ``_create_optimizer`` (warns + falls back to
        # "Adam" for unknown values). Pydantic Literal at the API boundary
        # already restricts the wire to the supported set.
        optimizer_type: str = "Adam",
        # Logging configuration
        log_config: LogConfig = None,
        log_file_name: str = _CASCADE_CORRELATION_NETWORK_LOG_FILE_NAME,
        log_file_path: str = _CASCADE_CORRELATION_NETWORK_LOG_FILE_PATH,
        log_level_name: str = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME,
        log_date_format: str = _CASCADE_CORRELATION_NETWORK_LOG_DATE_FORMAT,
        log_format_string: str = _CASCADE_CORRELATION_NETWORK_LOG_FORMATTER_STRING,
        log_level_custom_names_list: list = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_CUSTOM_NAMES_LIST,
        log_level_methods_dict: dict = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_METHODS_DICT,
        log_level_methods_list: list = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_METHODS_LIST,
        log_level_names_list: list = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAMES_LIST,
        log_level_numbers_dict: dict = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_DICT,
        log_level_numbers_list: list = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_LIST,
        # Multiprocessing configuration
        candidate_training_queue_authkey: str | None = _CASCADE_CORRELATION_NETWORK_AUTHKEY,
        candidate_training_queue_address: tuple = _CASCADE_CORRELATION_NETWORK_BASE_MANAGER_ADDRESS,
        candidate_training_worker_standby_sleepytime: float = _CASCADE_CORRELATION_NETWORK_WORKER_STANDBY_SLEEPYTIME,
        candidate_training_task_queue_timeout: float = _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT,
        candidate_training_shutdown_timeout: float = _CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT,
        candidate_training_target_accuracy: float = _CASCADE_CORRELATION_NETWORK_TARGET_ACCURACY,
        candidate_training_context_type: str = _CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_CONTEXT,
        # PARALLEL-FIX (RC-1): Configurable PyTorch thread count per worker process
        worker_thread_count: int = _CASCADE_CORRELATION_NETWORK_WORKER_THREAD_COUNT,
        # Remote worker configuration (Phase 1b — WebSocket-based distributed workers)
        enable_remote_workers: bool = False,
        ws_worker_token_secret: str = "",  # nosec B107 — default empty, set from env at runtime
        heartbeat_timeout: float = 30.0,
        task_reassignment_timeout: float = 120.0,
        # cascade_correlation_network_snapshots_dir: str = _HDF5_PROJECT_SNAPSHOTS_DIR,
        cascade_correlation_network_snapshots_dir: pathlib.Path = _CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR,
        # CAN-015g (g-6): per-sample weight history capture during
        # training. ``weight_history_sampling_interval`` is N for the
        # every-Nth-epoch trigger (50 by default; 1 captures every
        # epoch — Option A in the design's storage-strategy table; 0
        # disables the periodic trigger so only cascade-add events
        # are sampled — Option D). ``weight_history_max_samples`` is
        # a soft cap on the in-memory history length before
        # decimation kicks in (see plan §"Live capture during
        # training (g-6)" / "Memory ceiling"); 0 means unbounded
        # (use with care on long runs).
        weight_history_sampling_interval: int = 50,
        weight_history_max_samples: int = 1000,
        # UUID
        uuid: uuid.UUID = None,
    ):
        # Network architecture
        self.input_size = input_size
        self.output_size = output_size
        self.max_hidden_units = max_hidden_units

        # Activation function
        self.activation_function_name = activation_function_name
        self.activation_functions_dict = activation_functions_dict

        # Training parameters
        self.learning_rate = learning_rate
        self.candidate_learning_rate = candidate_learning_rate
        self.candidate_pool_size = candidate_pool_size
        self.candidate_epochs = candidate_epochs
        self.epochs_max = epochs_max
        self.max_iterations = max_iterations
        self.output_epochs = output_epochs
        self.patience = patience
        self.convergence_threshold = convergence_threshold
        self.candidate_convergence_threshold = candidate_convergence_threshold
        self.candidate_patience = candidate_patience

        # Thresholds
        self.correlation_threshold = correlation_threshold

        # N-best candidate selection
        self.candidates_per_layer = 1  # Set to N for layer-based addition
        self.layer_selection_strategy = "top_n"  # 'top_n', 'threshold', 'adaptive'

        # Display and visualization
        self.display_frequency = display_frequency
        self.epoch_display_frequency = epoch_display_frequency
        self.candidate_display_frequency = candidate_display_frequency
        self.status_display_frequency = status_display_frequency
        self.generate_plots = generate_plots

        # Random number generation
        self.random_seed = random_seed
        self.random_max_value = random_max_value
        self.sequence_max_value = sequence_max_value
        self.random_value_scale = random_value_scale

        # Output weight initialization
        self.init_output_weights = init_output_weights

        # Logging configuration
        self.log_config = log_config
        self.log_file_name = log_file_name
        self.log_file_path = log_file_path
        self.log_level_name = log_level_name
        self.log_date_format = log_date_format
        self.log_format_string = log_format_string
        self.log_level_custom_names_list = log_level_custom_names_list
        self.log_level_methods_dict = log_level_methods_dict
        self.log_level_methods_list = log_level_methods_list
        self.log_level_names_list = log_level_names_list
        self.log_level_numbers_dict = log_level_numbers_dict
        self.log_level_numbers_list = log_level_numbers_list

        # Multiprocessing configuration
        if candidate_training_queue_authkey is None:
            import secrets

            candidate_training_queue_authkey = secrets.token_hex(32)
        self.candidate_training_queue_authkey = candidate_training_queue_authkey
        self.candidate_training_queue_address = candidate_training_queue_address
        self.candidate_training_worker_standby_sleepytime = candidate_training_worker_standby_sleepytime
        self.candidate_training_task_queue_timeout = candidate_training_task_queue_timeout
        self.candidate_training_shutdown_timeout = candidate_training_shutdown_timeout
        self.candidate_training_target_accuracy = candidate_training_target_accuracy
        self.candidate_training_context_type = candidate_training_context_type
        # PARALLEL-FIX (RC-1): Worker thread count for PyTorch thread pinning in worker processes
        self.worker_thread_count = worker_thread_count

        # Remote worker configuration (Phase 1b)
        self.enable_remote_workers = enable_remote_workers
        self.ws_worker_token_secret = ws_worker_token_secret
        self.heartbeat_timeout = heartbeat_timeout
        self.task_reassignment_timeout = task_reassignment_timeout

        # Snapshot directory
        self.cascade_correlation_network_snapshots_dir = cascade_correlation_network_snapshots_dir

        # CAN-015g (g-6): per-sample weight history capture during training.
        self.weight_history_sampling_interval = weight_history_sampling_interval
        self.weight_history_max_samples = weight_history_max_samples

        # Optimizer configuration. CAN-010 / ENH-006 (A-2): forward
        # ``optimizer_type`` from the constructor kwargs so creation-time
        # selection (POST /v1/network with ``optimizer_type``) and
        # start-time override (TrainingParams) both reach the registry.
        self.optimizer_config = OptimizerConfig(learning_rate=learning_rate, optimizer_type=optimizer_type)

        # UUID
        self.uuid = uuid

    def __getstate__(self):
        """Remove non-picklable items for multiprocessing serialization."""
        state = self.__dict__.copy()
        # Remove non-serializable items (log_config contains loggers)
        state.pop("log_config", None)
        return state

    def __setstate__(self, state):
        """Restore instance from serialized state."""
        self.__dict__.update(state)
        # Set log_config to None - it will be recreated if needed
        self.log_config = None

    @classmethod
    def create_simple_config(cls, input_size: int = 2, output_size: int = 1, learning_rate: float = 0.1, max_hidden_units: int = 10, **kwargs):
        """
        Factory method to create a simplified configuration for common use cases.

        Args:
            input_size: Number of input features
            output_size: Number of output classes
            learning_rate: Learning rate for training
            max_hidden_units: Maximum number of hidden units to add
            **kwargs: Additional configuration parameters

        Returns:
            CascadeCorrelationConfig: Configured instance with sensible defaults
        """
        return cls(input_size=input_size, output_size=output_size, learning_rate=learning_rate, max_hidden_units=max_hidden_units, **kwargs)
