#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     cascade_correlation_config.py
# Author:        Paul Calnon
# Version:       0.3.2 (0.7.3)
#
# Date:          2025-09-26
# Last Modified: 2025-10-25 01:43:26 CDT
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
import uuid
import pathlib
from dataclasses import dataclass
from log_config.log_config import LogConfig


from constants.constants import (
    # _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_DISPLAY_FREQUENCY,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_EPOCHS,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_LEARNING_RATE,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_POOL_SIZE,
    _CASCADE_CORRELATION_NETWORK_DISPLAY_FREQUENCY,
    _CASCADE_CORRELATION_NETWORK_EPOCH_DISPLAY_FREQUENCY,
    _CASCADE_CORRELATION_NETWORK_EPOCHS_MAX,
    _CASCADE_CORRELATION_NETWORK_GENERATE_PLOTS,
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
    _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD,
    _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS,
    _CASCADE_CORRELATION_NETWORK_OUTPUT_SIZE,
    _CASCADE_CORRELATION_NETWORK_PATIENCE,
    _CASCADE_CORRELATION_NETWORK_RANDOM_MAX_VALUE,
    _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE,
    _CASCADE_CORRELATION_NETWORK_RANDOM_SEED,
    _CASCADE_CORRELATION_NETWORK_RANDOM_VALUE_SCALE,
    _CASCADE_CORRELATION_NETWORK_STATUS_DISPLAY_FREQUENCY,
    _CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR,

    _CASCADE_CORRELATION_NETWORK_WORKER_STANDBY_SLEEPYTIME,
    _CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT,
    _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT,
    _CASCADE_CORRELATION_NETWORK_TARGET_ACCURACY,

    _CASCADE_CORRELATION_NETWORK_AUTHKEY,
    _CASCADE_CORRELATION_NETWORK_BASE_MANAGER_ADDRESS,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_CONTEXT,
)


#####################################################################################################################################################################################################
# Optimizer configuration dataclass
@dataclass
class OptimizerConfig:
    """Configuration for output layer optimizer."""
    optimizer_type: str = 'Adam'  # Adam, SGD, RMSprop, AdamW, etc.
    learning_rate: float = 0.01
    momentum: float = 0.9  # For SGD
    beta1: float = 0.9  # For Adam
    beta2: float = 0.999  # For Adam
    weight_decay: float = 0.0
    epsilon: float = 1e-8


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
        output_epochs: int = _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS,
        patience: int = _CASCADE_CORRELATION_NETWORK_PATIENCE,

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
        candidate_training_queue_authkey: str = _CASCADE_CORRELATION_NETWORK_AUTHKEY,
        candidate_training_queue_address: tuple = _CASCADE_CORRELATION_NETWORK_BASE_MANAGER_ADDRESS,

        candidate_training_worker_standby_sleepytime: float = _CASCADE_CORRELATION_NETWORK_WORKER_STANDBY_SLEEPYTIME,
        candidate_training_task_queue_timeout: float = _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT,
        candidate_training_shutdown_timeout: float = _CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT,
        candidate_training_target_accuracy: float = _CASCADE_CORRELATION_NETWORK_TARGET_ACCURACY,
        candidate_training_context_type: str = _CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_CONTEXT,

        # cascade_correlation_network_snapshots_dir: str = _HDF5_PROJECT_SNAPSHOTS_DIR,
        cascade_correlation_network_snapshots_dir: pathlib.Path = _CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR,

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
        self.output_epochs = output_epochs
        self.patience = patience

        # Thresholds
        self.correlation_threshold = correlation_threshold

        # N-best candidate selection
        self.candidates_per_layer = 1  # Set to N for layer-based addition
        self.layer_selection_strategy = 'top_n'  # 'top_n', 'threshold', 'adaptive'

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
        self.candidate_training_queue_authkey = candidate_training_queue_authkey
        self.candidate_training_queue_address = candidate_training_queue_address
        self.candidate_training_worker_standby_sleepytime = candidate_training_worker_standby_sleepytime
        self.candidate_training_task_queue_timeout = candidate_training_task_queue_timeout
        self.candidate_training_shutdown_timeout = candidate_training_shutdown_timeout
        self.candidate_training_target_accuracy = candidate_training_target_accuracy
        self.candidate_training_context_type = candidate_training_context_type

        # Snapshot directory
        self.cascade_correlation_network_snapshots_dir = cascade_correlation_network_snapshots_dir

        # Optimizer configuration
        self.optimizer_config = OptimizerConfig(learning_rate=learning_rate)

        # UUID
        self.uuid = uuid

    @classmethod
    def create_simple_config(
        cls,
        input_size: int = 2,
        output_size: int = 1,
        learning_rate: float = 0.1,
        max_hidden_units: int = 10,
        **kwargs
    ):
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
        return cls(
            input_size=input_size,
            output_size=output_size,
            learning_rate=learning_rate,
            max_hidden_units=max_hidden_units,
            **kwargs
        )

