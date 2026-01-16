#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     spiral_problem.py
# Author:        Paul Calnon
# Version:       0.3.1  (0.7.3)
# 
# Date Created:  2025-07-29
# Last Modified: 2026-01-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This file contains the functions and code needed to solve the two spiral problem using a Cascade Correlation Neural Network.
#
#####################################################################################################################################################################################################
# Notes:
#  - This class represents the two spiral problem and provides methods to generate the dataset and solve it using a Cascade Correlation Neural Network.
#  - The Cascade Correlation Neural Network is used to solve the problem.
#  - The Cascade Correlation Network is implemented using PyTorch and carries out the training and evaluation of the network using the Cascade Correlation algorithm.
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
import logging
import logging.config
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import torch
import uuid
import multiprocessing as mp

# from inspect import currentframe, getframeinfo
from typing import Tuple

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

from constants.constants import (
    _SPIRAL_PROBLEM_ACTIVATION_FUNCTION,
    _SPIRAL_PROBLEM_CANDIDATE_DISPLAY_FREQUENCY,
    _SPIRAL_PROBLEM_CANDIDATE_EPOCHS,
    _SPIRAL_PROBLEM_CANDIDATE_LEARNING_RATE,
    _SPIRAL_PROBLEM_CANDIDATE_POOL_SIZE,
    _SPIRAL_PROBLEM_CLOCKWISE,
    _SPIRAL_PROBLEM_CORRELATION_THRESHOLD,
    _SPIRAL_PROBLEM_DEFAULT_ORIGIN,
    _SPIRAL_PROBLEM_DEFAULT_RADIUS,
    _SPIRAL_PROBLEM_DISPLAY_FREQUENCY,
    _SPIRAL_PROBLEM_DISTRIBUTION_FACTOR,
    _SPIRAL_PROBLEM_EPOCH_DISPLAY_FREQUENCY,
    _SPIRAL_PROBLEM_EPOCHS_MAX,
    _SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT,
    _SPIRAL_PROBLEM_INPUT_SIZE,
    _SPIRAL_PROBLEM_LEARNING_RATE,
    _SPIRAL_PROBLEM_LOG_DATE_FORMAT,
    _SPIRAL_PROBLEM_LOG_FILE_NAME,
    _SPIRAL_PROBLEM_LOG_FILE_PATH,
    _SPIRAL_PROBLEM_LOG_FORMATTER_STRING,
    _SPIRAL_PROBLEM_LOG_LEVEL_CUSTOM_NAMES_LIST,
    _SPIRAL_PROBLEM_LOG_LEVEL_METHODS_DICT,
    _SPIRAL_PROBLEM_LOG_LEVEL_METHODS_LIST,
    _SPIRAL_PROBLEM_LOG_LEVEL_NAME,
    _SPIRAL_PROBLEM_LOG_LEVEL_NAMES_LIST,
    _SPIRAL_PROBLEM_LOG_LEVEL_NUMBERS_DICT,
    _SPIRAL_PROBLEM_LOG_LEVEL_NUMBERS_LIST,
    _SPIRAL_PROBLEM_MAX_HIDDEN_UNITS,
    _SPIRAL_PROBLEM_MAX_NEW,
    _SPIRAL_PROBLEM_MAX_ORIG,
    _SPIRAL_PROBLEM_MIN_NEW,
    _SPIRAL_PROBLEM_MIN_ORIG,
    _SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT,
    _SPIRAL_PROBLEM_NUM_ROTATIONS,
    _SPIRAL_PROBLEM_NUM_SPIRALS,
    _SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL,
    _SPIRAL_PROBLEM_ORIG_POINTS,
    _SPIRAL_PROBLEM_OUTPUT_EPOCHS,
    _SPIRAL_PROBLEM_OUTPUT_SIZE,
    _SPIRAL_PROBLEM_PATIENCE,
    _SPIRAL_PROBLEM_RANDOM_SEED,
    _SPIRAL_PROBLEM_RANDOM_VALUE_SCALE,
    _SPIRAL_PROBLEM_STATUS_DISPLAY_FREQUENCY,
    _SPIRAL_PROBLEM_TEST_RATIO,
    _SPIRAL_PROBLEM_TRAIN_RATIO,
    _SPIRAL_PROBLEM_AUTHKEY,
    _SPIRAL_PROBLEM_BASE_MANAGER_ADDRESS_IP,
    _SPIRAL_PROBLEM_BASE_MANAGER_ADDRESS_PORT,
    # _PROJECT_MODEL_AUTHKEY,
    # _PROJECT_MODEL_BASE_MANAGER_ADDRESS,
    _PROJECT_MODEL_TASK_QUEUE_TIMEOUT,
    _PROJECT_MODEL_SHUTDOWN_TIMEOUT,
    _PROJECT_MODEL_CANDIDATE_TRAINING_CONTEXT,
)
from log_config.log_config import LogConfig
from log_config.logger.logger import Logger


#####################################################################################################################################################################################################
class SpiralProblem(object):
    """
    Class to represent the two spiral problem.
    This class is used to generate the two spiral problem dataset and solve it using a Cascade Correlation Neural Network.
    """

    #################################################################################################################################################################################################
    def __init__(
        self,
        _SpiralProblem__activation_function: str = _SPIRAL_PROBLEM_ACTIVATION_FUNCTION,
        _SpiralProblem__candidate_display_frequency: int = _SPIRAL_PROBLEM_CANDIDATE_DISPLAY_FREQUENCY,
        _SpiralProblem__candidate_epochs: int = _SPIRAL_PROBLEM_CANDIDATE_EPOCHS,
        _SpiralProblem__candidate_learning_rate: float = _SPIRAL_PROBLEM_CANDIDATE_LEARNING_RATE,
        _SpiralProblem__candidate_pool_size: int = _SPIRAL_PROBLEM_CANDIDATE_POOL_SIZE,
        _SpiralProblem__clockwise: bool = _SPIRAL_PROBLEM_CLOCKWISE,                # True for clockwise spirals, False for counter-clockwise
        _SpiralProblem__correlation_threshold: float = _SPIRAL_PROBLEM_CORRELATION_THRESHOLD,
        _SpiralProblem__epoch_display_frequency: int = _SPIRAL_PROBLEM_EPOCH_DISPLAY_FREQUENCY,
        _SpiralProblem__epochs_max: int = _SPIRAL_PROBLEM_EPOCHS_MAX,
        _SpiralProblem__generate_plots_default: bool = _SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT,
        _SpiralProblem__default_origin: float = _SPIRAL_PROBLEM_DEFAULT_ORIGIN,
        _SpiralProblem__default_radius: float = _SPIRAL_PROBLEM_DEFAULT_RADIUS,
        _SpiralProblem__display_frequency: int = _SPIRAL_PROBLEM_DISPLAY_FREQUENCY,
        _SpiralProblem__distribution: float = _SPIRAL_PROBLEM_DISTRIBUTION_FACTOR,
        _SpiralProblem__input_size: int = _SPIRAL_PROBLEM_INPUT_SIZE,
        _SpiralProblem__learning_rate: float = _SPIRAL_PROBLEM_LEARNING_RATE,
        _SpiralProblem__log_config: LogConfig | None = None,
        _SpiralProblem__log_date_format: str = _SPIRAL_PROBLEM_LOG_DATE_FORMAT,
        _SpiralProblem__log_file_name: str = _SPIRAL_PROBLEM_LOG_FILE_NAME,
        _SpiralProblem__log_file_path: str = _SPIRAL_PROBLEM_LOG_FILE_PATH,
        _SpiralProblem__log_format_string: str = _SPIRAL_PROBLEM_LOG_FORMATTER_STRING,
        _SpiralProblem__log_level_custom_names_list: list = _SPIRAL_PROBLEM_LOG_LEVEL_CUSTOM_NAMES_LIST,
        _SpiralProblem__log_level_methods_dict: dict = _SPIRAL_PROBLEM_LOG_LEVEL_METHODS_DICT,
        _SpiralProblem__log_level_methods_list: list = _SPIRAL_PROBLEM_LOG_LEVEL_METHODS_LIST,
        _SpiralProblem__log_level_name: str = _SPIRAL_PROBLEM_LOG_LEVEL_NAME,
        _SpiralProblem__log_level_names_list: list = _SPIRAL_PROBLEM_LOG_LEVEL_NAMES_LIST,
        _SpiralProblem__log_level_numbers_dict: dict = _SPIRAL_PROBLEM_LOG_LEVEL_NUMBERS_DICT,
        _SpiralProblem__log_level_numbers_list: list = _SPIRAL_PROBLEM_LOG_LEVEL_NUMBERS_LIST,
        _SpiralProblem__max_hidden_units: int = _SPIRAL_PROBLEM_MAX_HIDDEN_UNITS,
        _SpiralProblem__max_new: float = _SPIRAL_PROBLEM_MAX_NEW,                    # Maximum value for the new points
        _SpiralProblem__max_orig: float = _SPIRAL_PROBLEM_MAX_ORIG,                  # Maximum value for the original points
        _SpiralProblem__min_new: float = _SPIRAL_PROBLEM_MIN_NEW,                    # Minimum value for the new points
        _SpiralProblem__min_orig: float = _SPIRAL_PROBLEM_MIN_ORIG,                  # Minimum value for the original points
        _SpiralProblem__n_points: int = _SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL,
        _SpiralProblem__n_rotations: int = _SPIRAL_PROBLEM_NUM_ROTATIONS,
        _SpiralProblem__n_spirals: int = _SPIRAL_PROBLEM_NUM_SPIRALS,
        _SpiralProblem__noise: float = _SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT,
        _SpiralProblem__orig_points: int = _SPIRAL_PROBLEM_ORIG_POINTS,               # User provided data points or None
        _SpiralProblem__output_epochs: int = _SPIRAL_PROBLEM_OUTPUT_EPOCHS,
        _SpiralProblem__output_size: int = _SPIRAL_PROBLEM_OUTPUT_SIZE,
        _SpiralProblem__patience: int = _SPIRAL_PROBLEM_PATIENCE,
        _SpiralProblem__random_seed: int = _SPIRAL_PROBLEM_RANDOM_SEED,             # Default random seed for reproducibility
        _SpiralProblem__authkey: bytes = _SPIRAL_PROBLEM_AUTHKEY,
        _SpiralProblem__queue_address = _SPIRAL_PROBLEM_BASE_MANAGER_ADDRESS_IP,
        _SpiralProblem__queue_port = _SPIRAL_PROBLEM_BASE_MANAGER_ADDRESS_PORT,
        _SpiralProblem__task_queue_timeout = _PROJECT_MODEL_TASK_QUEUE_TIMEOUT,
        _SpiralProblem__shutdown_timeout = _PROJECT_MODEL_SHUTDOWN_TIMEOUT,
        _SpiralProblem__task_queue_context = _PROJECT_MODEL_CANDIDATE_TRAINING_CONTEXT,
        _SpiralProblem__random_value_scale: float = _SPIRAL_PROBLEM_RANDOM_VALUE_SCALE,
        _SpiralProblem__status_display_frequency: int = _SPIRAL_PROBLEM_STATUS_DISPLAY_FREQUENCY,
        _SpiralProblem__test_ratio: float = _SPIRAL_PROBLEM_TEST_RATIO,
        _SpiralProblem__train_ratio: float = _SPIRAL_PROBLEM_TRAIN_RATIO,
        _SpiralProblem__uuid: uuid.UUID | None = None,
        **kwargs,
    ):
        """
        Description:
            Initialize the SpiralProblem class with the specified parameters.
        Args:
            _SpiralProblem__activation_function: Activation function to use in the network.
            _SpiralProblem__candidate_display_frequency: Frequency of candidate display during training.
            _SpiralProblem__candidate_epochs: Number of epochs for candidate training.
            _SpiralProblem__candidate_pool_size: Size of the candidate pool for the network.
            _SpiralProblem__clockwise: True for clockwise spirals, False for counter-clockwise.
            _SpiralProblem__correlation_threshold: Correlation threshold for adding new hidden units.
            _SpiralProblem__display_frequency: Frequency of display updates during training.LogConfig
            _SpiralProblem__distribution: Factor to apply to the degrees of the spiral points.
            _SpiralProblem__default_origin: Default origin for the spirals.
            _SpiralProblem__default_radius: Default radius for the spirals.
            _SpiralProblem__epoch_display_frequency: Frequency of epoch display during training.
            _SpiralProblem__epochs_max: Maximum number of epochs for training.
            _SpiralProblem__generate_plots_default: Whether to generate plots by default.
            _SpiralProblem__input_size: Size of the input layer.
            _SpiralProblem__learning_rate: Learning rate for the network.
            _SpiralProblem__log_config: LogConfig object to use for logging configuration (if not provided, creates a new one).
            _SpiralProblem__log_date_format: Date format for log messages.
            _SpiralProblem__log_file_name: Name of the log file to use (if not provided, uses default).
            _SpiralProblem__log_file_path: Path to the log file (if not provided, uses default).
            _SpiralProblem__log_format: Format string for log messages.
            _SpiralProblem__log_level_custom_names_list: Custom list for log level names (if not provided, uses default).
            _SpiralProblem__log_level_methods_dict: Dictionary for log level methods (if not provided, uses default).
            _SpiralProblem__log_level_methods_list: List for log level methods (if not provided, uses default).
            _SpiralProblem__log_level_name: Name of the logging level to use (if not provided, uses default).
            _SpiralProblem__log_level_names_list: List for log level names (if not provided, uses default).
            _SpiralProblem__log_level_numbers_dict: Dictionary for log level numbers (if not provided, uses default).
            _SpiralProblem__log_level_numbers_list: List for log level numbers (if not provided, uses default).
            _SpiralProblem__max_hidden_units: Maximum number of hidden units in the network.
            _SpiralProblem__n_points: Number of points per spiral.
            _SpiralProblem__n_rotations: Number of rotations for each spiral.
            _SpiralProblem__n_spirals: Number of spirals to generate.
            _SpiralProblem__noise: Amount of noise to add to the spiral points.
            _SpiralProblem__output_epochs: Number of epochs for output training.
            _SpiralProblem__output_size: Size of the output layer.
            _SpiralProblem__patience: Patience for early stopping in training.
            _SpiralProblem__random_seed: Random seed for reproducibility.
            _SpiralProblem__random_value_scale: Scale for random values added to the spiral points.
            _SpiralProblem__status_display_frequency: Frequency of status display during training.
            _SpiralProblem__test_ratio: Ratio of test data to total data. Must be between 0.0 and 1.0.
            _SpiralProblem__train_ratio: Ratio of training data to total data. Must be between 0.0 and 1.0.
            _SpiralProblem__uuid: UUID for the Spiral Problem instance.
        Raises:
            ValueError: If the input parameters are invalid.
        Notes:
            This method initializes the logger using the provided parameters.
            If no logger is provided, it creates a logger using the log file path and logging level.
            If the logger already exists, it does not create a new one.
            This method also initializes the Spiral Problem (CCN) using the specified parameters.
            If the CCN already exists, it does not create a new one.
            If the CCN does not exist, it creates a new one using the specified parameters.
        Returns:
            None
        """
        super().__init__()

        # Initialize the SpiralProblem class logger
        self.log_file_name = _SpiralProblem__log_file_name or __name__
        self.log_file_path = _SpiralProblem__log_file_path or str(os.path.join(os.getcwd(), "logs"))
        Logger.debug(f"SpiralProblem: __init__: Log file name: {self.log_file_name}, Log file path: {self.log_file_path}")
        self.log_level_name = _SpiralProblem__log_level_name or _SPIRAL_PROBLEM_LOG_LEVEL_NAME
        Logger.debug(f"SpiralProblem: __init__: Logging level name: {self.log_level_name}")

        # Create LogConfig object if not provided
        Logger.debug(f"SpiralProblem: __init__: Log config object: {_SpiralProblem__log_config}, Type: {type(_SpiralProblem__log_config)}")
        self.log_config = _SpiralProblem__log_config or LogConfig(
            _LogConfig__log_config=logging.config,
            _LogConfig__log_file_name=self.log_file_name,
            _LogConfig__log_file_path=self.log_file_path,
            _LogConfig__log_level_name=self.log_level_name,
            _LogConfig__log_date_format = _SpiralProblem__log_date_format,
            _LogConfig__log_format_string = _SpiralProblem__log_format_string,
            _LogConfig__log_level_custom_names_list = _SpiralProblem__log_level_custom_names_list,
            _LogConfig__log_level_methods_dict = _SpiralProblem__log_level_methods_dict,
            _LogConfig__log_level_methods_list = _SpiralProblem__log_level_methods_list,
            _LogConfig__log_level_names_list = _SpiralProblem__log_level_names_list,
            _LogConfig__log_level_numbers_dict = _SpiralProblem__log_level_numbers_dict,
            _LogConfig__log_level_numbers_list = _SpiralProblem__log_level_numbers_list,
        )
        Logger.debug(f"SpiralProblem: __init__: Log config after conditional assignment: {self.log_config}, Type: {type(self.log_config)}")

        # Get logger from LogConfig
        self.logger = self.log_config.get_logger()
        Logger.debug(f"SpiralProblem: __init__: Logger created: {self.logger}, Type: {type(self.logger)}")
        Logger.debug(f"SpiralProblem: __init__: Logger after getLogger: Type: {type(self.logger)}, Value: {self.logger}")
        Logger.debug(f"SpiralProblem: __init__: Logger before config: {self.logger}, Level: {self.logger.level}, Handlers: {self.logger.handlers}")

        # = logging.getLevelName(logging.DEBUG)
        self.logger.level = self.log_config.get_log_level()
        Logger.debug(f"SpiralProblem: __init__: Logger after setLevel: {self.logger}, Level: {self.logger.level}, Handlers: {self.logger.handlers}")
        self.logger.debug(f"SpiralProblem: __init__: Logger initialized: {self.logger}")
        self.logger.trace("SpiralProblem: __init__: Inside SpiralProblem class __init__ method")
        self.logger.debug("SpiralProblem: __init__: Completed initialization of Spiral Problem Logger")

        # Initialize the Spiral Problem input parameters
        self.logger.info("SpiralProblem: __init__: Initializing Spiral Problem")
        self.n_spirals = _SpiralProblem__n_spirals
        self.logger.verbose(f"SpiralProblem: __init__: Number of spirals: {self.n_spirals}")
        self.n_points = _SpiralProblem__n_points
        self.logger.verbose(f"SpiralProblem: __init__: Number of points per spiral: {self.n_points}")
        self.n_rotations = _SpiralProblem__n_rotations
        self.logger.verbose(f"SpiralProblem: __init__: Number of rotations per spiral: {self.n_rotations}")
        self.clockwise = _SpiralProblem__clockwise
        self.logger.verbose(f"SpiralProblem: __init__: Clockwise spirals: {self.clockwise}")
        self.noise = _SpiralProblem__noise
        self.logger.verbose(f"SpiralProblem: __init__: Noise factor: {self.noise}")
        self.distribution = _SpiralProblem__distribution
        self.logger.verbose(f"SpiralProblem: __init__: Distribution: {self.distribution}")
        self.random_seed = _SpiralProblem__random_seed  # Default random seed for reproducibility
        self.logger.verbose(f"SpiralProblem: __init__: Random seed: {self.random_seed}")
        self.random_value_scale = _SpiralProblem__random_value_scale
        self.logger.verbose(f"SpiralProblem: __init__: Random value scale: {self.random_value_scale}")
        self.train_ratio = _SpiralProblem__train_ratio
        self.logger.verbose(f"SpiralProblem: __init__: Training ratio: {self.train_ratio}")
        self.test_ratio = _SpiralProblem__test_ratio
        self.logger.verbose(f"SpiralProblem: __init__: Testing ratio: {self.test_ratio}")
        self.min_new = _SpiralProblem__min_new
        self.logger.verbose(f"SpiralProblem: __init__: Minimum new value: {self.min_new}")
        self.max_new = _SpiralProblem__max_new
        self.logger.verbose(f"SpiralProblem: __init__: Maximum new value: {self.max_new}")
        self.min_orig = _SpiralProblem__min_orig
        self.logger.verbose(f"SpiralProblem: :__init__: Minimum original value: {self.min_orig}")
        self.max_orig = _SpiralProblem__max_orig
        self.logger.verbose(f"SpiralProblem: __init__: Maximum original value: {self.max_orig}")
        self.orig_points = _SpiralProblem__orig_points
        self.logger.verbose(f"SpiralProblem: __init__: Original points: {self.orig_points}")
        self.default_origin = _SpiralProblem__default_origin
        self.logger.verbose(f"SpiralProblem: __init__: Default origin: {self.default_origin}")
        self.default_radius = _SpiralProblem__default_radius
        self.logger.verbose(f"SpiralProblem: __init__: Default radius: {self.default_radius}")
        self.candidate_pool_size = _SpiralProblem__candidate_pool_size
        self.logger.verbose(f"SpiralProblem: __init__: Candidate pool size: {self.candidate_pool_size}")
        self.candidate_learning_rate = _SpiralProblem__candidate_learning_rate
        self.logger.verbose(f"SpiralProblem: __init__: Candidate learning rate: {self.candidate_learning_rate}")
        self.max_hidden_units = _SpiralProblem__max_hidden_units
        self.logger.verbose(f"SpiralProblem: __init__: Max hidden units: {self.max_hidden_units}")
        self.activation_function = _SpiralProblem__activation_function
        self.logger.verbose(f"SpiralProblem: __init__: Activation function: {self.activation_function}")
        self.input_size = _SpiralProblem__input_size
        self.logger.verbose(f"SpiralProblem: __init__: Input size: {self.input_size}")
        self.output_size = _SpiralProblem__output_size
        self.logger.verbose(f"SpiralProblem: __init__: Output size: {self.output_size}")
        self.learning_rate = _SpiralProblem__learning_rate
        self.logger.verbose(f"SpiralProblem: __init__: Learning rate: {self.learning_rate}")
        self.correlation_threshold = _SpiralProblem__correlation_threshold
        self.logger.verbose(f"SpiralProblem: __init__: Correlation threshold: {self.correlation_threshold}")
        self.patience = _SpiralProblem__patience
        self.logger.verbose(f"SpiralProblem: __init__: Patience: {self.patience}")
        self.candidate_epochs = _SpiralProblem__candidate_epochs
        self.logger.verbose(f"SpiralProblem: __init__: Candidate epochs: {self.candidate_epochs}")
        self.output_epochs = _SpiralProblem__output_epochs
        self.logger.verbose(f"SpiralProblem: __init__: Output epochs: {self.output_epochs}")
        self.epochs_max = _SpiralProblem__epochs_max
        self.logger.verbose(f"SpiralProblem: __init__: Max epochs: {self.epochs_max}")
        self.display_frequency = _SpiralProblem__display_frequency
        self.logger.verbose(f"SpiralProblem: __init__: Display frequency: {self.display_frequency}")
        self.epoch_display_frequency = _SpiralProblem__epoch_display_frequency
        self.logger.verbose(f"SpiralProblem: __init__: Epoch display frequency: {self.epoch_display_frequency}")
        self.status_display_frequency = _SpiralProblem__status_display_frequency
        self.logger.verbose(f"SpiralProblem: __init__: Status display frequency: {self.status_display_frequency}")
        self.candidate_display_frequency = _SpiralProblem__candidate_display_frequency
        self.logger.verbose(f"SpiralProblem: __init__: Candidate display frequency: {self.candidate_display_frequency}")
        # self.generate_plots_default = _SpiralProblem__generate_plots_default
        self.generate_plots = _SpiralProblem__generate_plots_default
        self.logger.verbose(f"SpiralProblem: __init__: Generate plots: {self.generate_plots}")
        self.logger.trace(f"SpiralProblem: __init__: Setting UUID for SpiralProblem class: {self}, uuid param: {_SpiralProblem__uuid}")
        self.log_date_format = _SpiralProblem__log_date_format
        self.logger.verbose(f"SpiralProblem: __init__: Log date format: {self.log_date_format}")
        self.log_format_string = _SpiralProblem__log_format_string
        self.logger.verbose(f"SpiralProblem: __init__: Log format string: {self.log_format_string}")
        self.log_level_custom_names_list = _SpiralProblem__log_level_custom_names_list
        self.logger.verbose(f"SpiralProblem: __init__: Log level custom names list: {self.log_level_custom_names_list}")
        self.log_level_methods_dict = _SpiralProblem__log_level_methods_dict
        self.logger.verbose(f"SpiralProblem: __init__: Log level methods dict: {self.log_level_methods_dict}")
        self.log_level_methods_list = _SpiralProblem__log_level_methods_list
        self.logger.verbose(f"SpiralProblem: __init__: Log level methods list: {self.log_level_methods_list}")
        self.log_level_names_list = _SpiralProblem__log_level_names_list
        self.logger.verbose(f"SpiralProblem: __init__: Log level names list: {self.log_level_names_list}")
        self.log_level_numbers_dict = _SpiralProblem__log_level_numbers_dict
        self.logger.verbose(f"SpiralProblem: __init__: Log level numbers dict: {self.log_level_numbers_dict}")
        self.log_level_numbers_list = _SpiralProblem__log_level_numbers_list
        self.logger.verbose(f"SpiralProblem: __init__: Log level numbers list: {self.log_level_numbers_list}")
        self.candidate_training_queue_authkey = _SpiralProblem__authkey
        self.logger.verbose(f"SpiralProblem: __init__: Candidate training queue authkey: {self.candidate_training_queue_authkey}")
        self.candidate_training_queue_address = _SpiralProblem__queue_address
        self.logger.verbose(f"SpiralProblem: __init__: Candidate training queue address: {self.candidate_training_queue_address}")
        self.candidate_training_task_queue_timeout = _SpiralProblem__task_queue_timeout
        self.logger.verbose(f"SpiralProblem: __init__: Candidate training task queue timeout: {self.candidate_training_task_queue_timeout}")
        self.candidate_training_shutdown_timeout = _SpiralProblem__shutdown_timeout
        self.logger.verbose(f"SpiralProblem: __init__: Candidate training shutdown timeout: {self.candidate_training_shutdown_timeout}")
        self.candidate_training_context_type = _SpiralProblem__task_queue_context
        self.logger.verbose(f"SpiralProblem: __init__: Candidate training context type: {self.candidate_training_context_type}")
        self.set_uuid(_SpiralProblem__uuid)
        self.logger.debug(f"SpiralProblem: __init__: UUID set to: {self.uuid}")
        self.logger.verbose(f"SpiralProblem: __init__: self.uuid: {self.get_uuid()}")
        self.logger.trace("SpiralProblem: __init__: Completed initialization of SpiralProblem class with input parameters.")

        # Set the random seed for reproducibility
        self.logger.trace("SpiralProblem: __init__: Setting random seed for reproducibility")
        torch.manual_seed(self.random_seed) # Set random seed for reproducibility with torch.manual_seed
        np.random.seed(self.random_seed)    # Set random seed for reproducibility with np.random.seed
        random.seed(self.random_seed)       # Set random seed for reproducibility with random.seed

        # Create the spiral problem object
        self.logger.trace("SpiralProblem: __init__: Creating the CascadeCorrelationNetwork instance")
        if ((network := CascadeCorrelationNetwork(
            config = CascadeCorrelationConfig(
                input_size=self.input_size,
                output_size=self.output_size,
                max_hidden_units=self.max_hidden_units,
                activation_function_name=self.activation_function,
                # activation_function_name=self.activation_function_name,
                # activation_functions_dict=self.activation_functions_dict,
                learning_rate=self.learning_rate,
                candidate_learning_rate=self.candidate_learning_rate,
                candidate_pool_size=self.candidate_pool_size,
                candidate_epochs=self.candidate_epochs,
                epochs_max=self.epochs_max,
                output_epochs=self.output_epochs,
                patience=self.patience,
                correlation_threshold=self.correlation_threshold,
                display_frequency=self.display_frequency,
                epoch_display_frequency=self.epoch_display_frequency,
                candidate_display_frequency=self.candidate_display_frequency,
                status_display_frequency=self.status_display_frequency,
                generate_plots=self.generate_plots,
                random_seed=self.random_seed,
                # random_max_value=self.random_max_value,
                # sequence_max_value=self.sequence_max_value,
                random_value_scale=self.random_value_scale,
                log_config=self.log_config,
                log_file_name=self.log_file_name,
                log_file_path=self.log_file_path,
                log_level_name=self.log_level_name,
                log_date_format=self.log_date_format,
                log_format_string=self.log_format_string,
                log_level_custom_names_list=self.log_level_custom_names_list,
                log_level_methods_dict=self.log_level_methods_dict,
                log_level_methods_list=self.log_level_methods_list,
                log_level_names_list=self.log_level_names_list,
                log_level_numbers_dict=self.log_level_numbers_dict,
                log_level_numbers_list=self.log_level_numbers_list,
                candidate_training_queue_authkey=self.candidate_training_queue_authkey,
                candidate_training_queue_address=self.candidate_training_queue_address,
                candidate_training_task_queue_timeout=self.candidate_training_task_queue_timeout,
                candidate_training_shutdown_timeout=self.candidate_training_shutdown_timeout,
                candidate_training_context_type=self.candidate_training_context_type,
                uuid=None,
            ),
        )) is None):
            self.logger.critical("SpiralProblem: __init__: Failed to create CascadeCorrelationNetwork")
            raise ValueError("SpiralProblem: solve_n_spiral_problem: Failed to create Spiral Problem")

        self.network = network
        self.logger.verbose(f"SpiralProblem: __init__: Created CascadeCorrelationNetwork with UUID: {self.network.get_uuid()}")
        self.logger.trace("SpiralProblem: __init__: Created CascadeCorrelationNetwork")
        self.logger.trace("SpiralProblem: __init__: Completed SpiralProblem class __init__ method")


    #####################################################################################################################################################################################################
    # Define function to generate the two spiral problem dataset.
    # TODO: Convert this to use spiral problem in Project Data Dir.
    def generate_n_spiral_dataset(
        self,
        min_new=_SPIRAL_PROBLEM_MIN_NEW,                              # Minimum value for the new poi,nts
        max_new=_SPIRAL_PROBLEM_MAX_NEW,                              # Maximum value for the new points
        min_orig=_SPIRAL_PROBLEM_MIN_ORIG,                            # Minimum value for the original points
        max_orig=_SPIRAL_PROBLEM_MAX_ORIG,                            # Maximum value for the original points
        orig_points=_SPIRAL_PROBLEM_ORIG_POINTS,                      # User provided data points or None
        train_ratio=_SPIRAL_PROBLEM_TRAIN_RATIO,
        test_ratio=_SPIRAL_PROBLEM_TEST_RATIO,
        clockwise=_SPIRAL_PROBLEM_CLOCKWISE,                          # True for clockwise spirals, False for counter-clockwise
        n_spirals=_SPIRAL_PROBLEM_NUM_SPIRALS,
        n_rotations=_SPIRAL_PROBLEM_NUM_ROTATIONS,
        n_points=_SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL,            # Number of points per spiral
        default_origin=_SPIRAL_PROBLEM_DEFAULT_ORIGIN,
        default_radius=_SPIRAL_PROBLEM_DEFAULT_RADIUS,
        noise_level=_SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT,
        distribution=_SPIRAL_PROBLEM_DISTRIBUTION_FACTOR,
    ):
        """
        Description:
            Generate a dataset of n spirals with specified parameters.  This function generates a dataset of n spirals,
            each with a specified number of points, noise level, and rotation direction.
        Args:
            orig_points: User provided data points or None. If None, random values in the range [0.0, 1.0] will be generated.
            min_new: Minimum value for the new points.
            max_new: Maximum value for the new points.
            min_orig: Minimum value for the original points.
            max_orig: Maximum value for the original points.
            train_ratio: Ratio of training data to total data.  Must be between 0.0 and 1.0.
            test_ratio: Ratio of test data to total data.  Must be between 0.0 and 1.0.
            noise: Amount of noise to add to the spiral points.
            clockwise: True for clockwise spirals, False for counter-clockwise.
            num_spirals: Number of spirals to generate.
            num_rotations: Number of rotations for each spiral.
            default_origin: Default origin for the spirals.
            default_radius: Default radius for the spirals.
            num_points: Number of points per spiral.
            distribution: Factor to apply to the degrees of the spiral points.
        Notes:
            - The function generates a dataset of n spirals, each with a specified number of points, noise level, and rotation direction.
            - The spirals are generated in a clockwise or counter-clockwise direction based on the `clockwise` parameter.
            - The points are scaled from the original range [min_orig, max_orig] to the new range [min_new, max_new].    <============ # TODO: Need to restore this functionality
            - The function returns the input features and one-hot encoded targets as PyTorch tensors.
            - The function uses the `default_origin` and `default_radius` to calculate the radius of the spirals.
            - The function uses the `distribution` to apply a degree of rotation to the spiral points.
            - The function generates the spirals using the `generate_spiral_data` function.
        Returns:
            tuple: A tuple containing the training and test sets, and the full dataset.
                - (x_train, y_train): Training set features and targets.
                - (x_test, y_test): Test set features and targets.
                - (x_full, y_full): Full dataset features and targets.
        """
        # Initialize Spiral Problem input parameters
        # TODO: Reconsider how these parameters are set.  How should preferred values be chosen? Should they be class attributes or passed in as parameters?
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Initializing Spiral Problem input parameters")
        self._initialize_spiral_problem_params(
            min_new=min_new,
            max_new=max_new,
            min_orig=min_orig,
            max_orig=max_orig,
            orig_points=orig_points,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            clockwise=clockwise,
            n_spirals=n_spirals,
            n_rotations=n_rotations,
            n_points=n_points,
            default_origin=default_origin,
            default_radius=default_radius,
            noise_level=noise_level,
            distribution=distribution,
        )
        self.logger.debug("SpiralProblem: generate_n_spiral_dataset: Initialized Spiral Problem input parameters")
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Generating {self.n_spirals} spirals with {self.n_points} points each, direction: {'clockwise' if self.clockwise else 'counter-clockwise'}, noise level: {self.noise}, distribution factor: {self.distribution}")
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Generating spiral coordinates")
        (spiral_x_coords, spiral_y_coords) = self._generate_spiral_coordinates() # Generate the spiral coordinates
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Generated spiral coordinates with {len(spiral_x_coords)} x-coordinates and {len(spiral_y_coords)} y-coordinates")
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Creating the spiral dataset")
        (x, y) = self._create_spiral_dataset( spiral_x_coords=spiral_x_coords, spiral_y_coords=spiral_y_coords,) # convert the spiral coordinates to the spiral dataset
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Generated spiral dataset with {len(x)} points and {len(y)} labels")
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Converting spiral dataset to PyTorch tensors")
        x_tensor, y_tensor = self._convert_to_tensors(x, y) # Convert to PyTorch tensors
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Converted spiral dataset to tensors with x: Shape: {x_tensor.shape}, Type: {type(x_tensor)}, y: Shape: {y_tensor.shape}, Type: {type(y_tensor)}")
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Shuffling the dataset")
        x_shuffled, y_shuffled = self._shuffle_dataset(x_tensor=x_tensor, y_tensor=y_tensor) # Shuffle the data
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Shuffled spiral dataset with x: Shape: {x_shuffled.shape}, Type: {type(x_shuffled)}, y: Shape: {y_shuffled.shape}, Type: {type(y_shuffled)}")
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Splitting the dataset into training and test sets")
        partitioned_dataset = self._partition_dataset(total_points=self.total_points, partitions=(train_ratio, test_ratio), x=x_shuffled, y=y_shuffled) # Split the dataset into training and test sets
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Partitioned dataset: {partitioned_dataset}")
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Unpacking the partitioned dataset")
        partition_train, partition_test = partitioned_dataset # Unpack the partitioned dataset into training and test sets
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Partitioned dataset: {partition_train}, {partition_test}")
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Unpacking the training and test sets")
        (x_train, y_train,) = partition_train # Unpack the training and test sets
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Training set x: Shape: {x_train.shape}, Type: {type(x_train)}, y: Shape: {y_train.shape}, Type: {type(y_train)}")
        (x_test, y_test) = partition_test
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Test set x: Shape: {x_test.shape}, Type: {type(x_test)}, y: Shape: {y_test.shape}, Type: {type(y_test)}")
        self.logger.trace("SpiralProblem: generate_n_spiral_data: Completed generation of n spiral dataset")
        return (x_train, y_train), (x_test, y_test), (x_shuffled, y_shuffled) # Return training and test sets, and the full dataset

    def _initialize_spiral_problem_params(
        self,
        min_new=_SPIRAL_PROBLEM_MIN_NEW,                              # Minimum value for the new points
        max_new=_SPIRAL_PROBLEM_MAX_NEW,                              # Maximum value for the new points
        min_orig=_SPIRAL_PROBLEM_MIN_ORIG,                            # Minimum value for the original points
        max_orig=_SPIRAL_PROBLEM_MAX_ORIG,                            # Maximum value for the original points
        orig_points=_SPIRAL_PROBLEM_ORIG_POINTS,                      # User provided data points or None
        train_ratio=_SPIRAL_PROBLEM_TRAIN_RATIO,
        test_ratio=_SPIRAL_PROBLEM_TEST_RATIO,
        clockwise=_SPIRAL_PROBLEM_CLOCKWISE,                          # True for clockwise spirals, False for counter-clockwise
        n_spirals=_SPIRAL_PROBLEM_NUM_SPIRALS,
        n_rotations=_SPIRAL_PROBLEM_NUM_ROTATIONS,
        n_points=_SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL,            # Number of points per spiral
        default_origin=_SPIRAL_PROBLEM_DEFAULT_ORIGIN,
        default_radius=_SPIRAL_PROBLEM_DEFAULT_RADIUS,
        noise_level=_SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT,
        distribution=_SPIRAL_PROBLEM_DISTRIBUTION_FACTOR,
    ) -> None:
        """
        Description:
            Initialize the parameters for the Spiral Problem.
        Args:
            min_new: Minimum value for the new points.
            max_new: Maximum value for the new points.
            min_orig: Minimum value for the original points.
            max_orig: Maximum value for the original points.
            orig_points: User provided data points or None. If None, random values in the range [0.0, 1.0] will be generated.
            train_ratio: Ratio of training data to total data. Must be between 0.0 and 1.0.
            test_ratio: Ratio of test data to total data. Must be between 0.0 and 1.0.
            clockwise: True for clockwise spirals, False for counter-clockwise.
            n_spirals: Number of spirals to generate.
            n_rotations: Number of rotations for each spiral.
            default_origin: Default origin for the spirals.
            default_radius: Default radius for the spirals.
            n_points: Number of points per spiral.
            noise_level: Amount of noise to add to the spiral points.
            distribution: Factor to apply to the degrees of the spiral points.
        Returns:
            None
        Notes:
            This method initializes the parameters for the Spiral Problem.
            It sets the parameters to the provided values or uses the class attributes if the parameters are None.
            This method is called by the `generate_n_spiral_dataset` method to initialize the parameters for generating the spiral dataset.
        """
        self.logger.trace("SpiralProblem: _initialize_spiral_problem_params: Initializing Spiral Problem parameters")
        # Set the parameters to the provided values or use the class attributes if the parameters are None
        self.min_new = min_new or self.min_new or _SPIRAL_PROBLEM_MIN_NEW   # Use class attribute if min_new is None
        self.max_new = max_new or self.max_new or _SPIRAL_PROBLEM_MAX_NEW # Use class attribute if max_new is None
        self.min_orig = min_orig or self.min_orig or _SPIRAL_PROBLEM_MIN_ORIG # Use class attribute if min_orig is None
        self.max_orig = max_orig or self.max_orig or _SPIRAL_PROBLEM_MAX_ORIG # Use class attribute if max_orig is None
        self.orig_points = orig_points or self.orig_points or _SPIRAL_PROBLEM_ORIG_POINTS # Use class attribute if orig_points is None
        self.train_ratio = train_ratio or self.train_ratio or _SPIRAL_PROBLEM_TRAIN_RATIO # Use class attribute if train_ratio is None
        self.test_ratio = test_ratio or self.test_ratio or _SPIRAL_PROBLEM_TEST_RATIO # Use class attribute if test_ratio is None
        self.clockwise = clockwise or self.clockwise or _SPIRAL_PROBLEM_CLOCKWISE # Use class attribute if clockwise is None
        self.n_spirals = n_spirals or self.n_spirals or _SPIRAL_PROBLEM_NUM_SPIRALS # Use class attribute if n_spirals is None
        self.n_rotations = n_rotations or self.n_rotations or _SPIRAL_PROBLEM_NUM_ROTATIONS # Use class attribute if n_rotations is None
        self.n_points = n_points or self.n_points or _SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL # Use class attribute if n_points is None
        self.default_origin = default_origin or self.default_origin or _SPIRAL_PROBLEM_DEFAULT_ORIGIN # Use class attribute if default_origin is None
        self.default_radius = default_radius or self.default_radius or _SPIRAL_PROBLEM_DEFAULT_RADIUS # Use class attribute if default_radius is None
        self.noise = noise_level or self.noise or _SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT # Use class attribute if noise is None
        self.distribution = distribution or self.distribution or _SPIRAL_PROBLEM_DISTRIBUTION_FACTOR # Use class attribute if distribution is None
        self.total_points = self.n_spirals * self.n_points
        self.logger.trace("SpiralProblem: _initialize_spiral_problem_params: Completed initialization")

    def _generate_spiral_coordinates(self,) -> Tuple[np.ndarray, np.ndarray]:
        """
        Description:
            Generate the coordinates for n spirals.
        Args:
            None
        Raises:
            ValueError: If n_points or n_rotations are not positive integers.
            ValueError: If n_distance is not a numpy array of the correct shape.
            ValueError: If distribution is not a positive float.
        Notes:
            This function generates the coordinates for n spirals based on the input parameters.
        Returns:
            Tuple[np.ndarray, np.ndarray]: x and y coordinates for the spirals.
        """
        # Generate the base radial distance
        self.logger.trace("SpiralProblem: generate_spiral_dataset: Generating base radial distance")
        n_distance = self._generate_base_radial_distance(n_points=self.n_points,)
        self.logger.debug(f"SpiralProblem: generate_spiral_dataset: Base radial distance generated: {n_distance}")
        direction = (1, -1)[self.clockwise]  # Determine the direction of the spiral based on clockwise parameter
        self.logger.debug(f"SpiralProblem: generate_spiral_dataset: Direction: ({direction}) of spiral: {'clockwise' if self.clockwise else 'counter-clockwise'}")
        self.logger.trace("SpiralProblem: generate_spiral_coordinates: Generating the angular_offset.")
        angular_offset = self._generate_angular_offset()
        self.logger.verbose(f"SpiralProblem: generate_spiral_coordinates: Angular offset for spiral: {angular_offset}")
        self.logger.trace("SpiralProblem: generate_spiral_coordinates: Generating spiral coordinates")
        (spiral_x_coords, spiral_y_coords) = self._generate_raw_spiral_coordinates(n_distance=n_distance, direction=direction, angular_offset=angular_offset)
        self.logger.debug(f"SpiralProblem: generate_spiral_dataset: Generated {len(spiral_x_coords)} spirals with {len(spiral_y_coords)} coordinates each")
        self.logger.trace("SpiralProblem: generate_spiral_dataset: Completed generation of spiral coordinates")
        # Return the spiral coordinates as numpy arrays
        return (spiral_x_coords, spiral_y_coords)

    def _generate_base_radial_distance(self, n_points: int = 0) -> float:
        """
        Description:
            Generate the base radial distance for the spiral.
        Args:
            n_points: Number of points to generate the radial distance for.
        Raises:
            ValueError: If n_points is not a positive integer.
        Notes:
            This function generates a radial distance based on a uniform distribution.
        Returns:
            float: Base radial distance for the spiral.
        """
        self.logger.trace(f"SpiralProblem: generate_base_radial_distance: Generating base radial distance for {n_points} points")
        if not isinstance(n_points, int) or n_points <= 0:
            raise ValueError(f"SpiralProblem: generate_base_radial_distance: n_points must be a positive integer, but got {n_points}.")
        # Generate radial distance based on a uniform distribution
        self.logger.debug(f"SpiralProblem: generate_base_radial_distance: Generating radial distance for {n_points} points")
        radial_distance = np.sqrt(np.random.rand(n_points)) * 780 * (2 * np.pi) / 360
        self.logger.debug(f"SpiralProblem: generate_base_radial_distance: Generated radial distance with shape: {radial_distance.shape}")
        self.logger.trace(f"SpiralProblem: generate_base_radial_distance: Completed generating base radial distance for {n_points} points")
        # Return the radial distance
        return radial_distance

    def _generate_angular_offset(self,) -> float:
        """
        Description:
            Generate an angular offset for the spiral based on its index.
        Args:
            None
        Notes:
            This function generates an angular offset for the spiral based on its index.
            The offset is calculated to evenly distribute the spirals around a circle.
        Returns:
            float: Angular offset for the spiral.
        """
        self.logger.trace("SpiralProblem: generate_angular_offset: Generating angular offset for spiral")
        # Calculate angular offset for this spiral (evenly distributed around circle)
        angular_offset = 2 * np.pi / self.n_spirals
        self.logger.debug(f"SpiralProblem: generate_angular_offset: Angular offset for spiral: {angular_offset}")
        self.logger.trace("SpiralProblem: generate_angular_offset: Completed generating angular offset for spiral")
        # Return the angular offset
        return angular_offset

    def _generate_raw_spiral_coordinates(
        self,
        n_distance: float = 0.0,
        direction: int = 0,
        angular_offset: float = 0.0,
    ) -> Tuple[list, list]:
        self.logger.trace("SpiralProblem: generate_raw_spiral_coordinates: Generating raw spiral coordinates")
        # Initialize lists to hold the coordinates for each spiral
        spiral_x_coords = []
        spiral_y_coords = []
        # Generate N spirals with evenly distributed angular offsets
        self.logger.trace("SpiralProblem: generate_raw_spiral_coordinates: Generate N spirals with evenly distributed angular offsets")
        for index in range(self.n_spirals):
            # Append the Current Spiral Coordinates to the lists
            self.logger.trace(f"SpiralProblem: generate_raw_spiral_coordinates: Generating spiral {index+1}/{self.n_spirals} coordinates")
            spiral_x, spiral_y = self._generate_xy_coordinates( n_distance=n_distance, direction=direction, angular_offset=angular_offset, index=index,)
            self.logger.debug(f"SpiralProblem: generate_raw_spiral_coordinates: Spiral {index+1}/{self.n_spirals} coordinates generated")
            self.logger.verbose(f"SpiralProblem: generate_spiral_dataset: Spiral {index+1} X coordinates: {spiral_x}, Y coordinates: {spiral_y}")
            spiral_x_coords.append(spiral_x)
            spiral_y_coords.append(spiral_y)
            self.logger.debug(f"SpiralProblem: generate_spiral_dataset: Appended spiral {index+1}/{self.n_spirals} coordinates to lists")
        self.logger.trace("SpiralProblem: generate_raw_spiral_coordinates: Completed generating raw spiral coordinates")
        self.logger.debug(f"SpiralProblem: generate_raw_spiral_coordinates: Generated {len(spiral_x_coords)} spirals with {len(spiral_y_coords)} coordinates each")
        # Return the spiral coordinates as numpy arrays
        self.logger.trace("SpiralProblem: generate_raw_spiral_coordinates: Returning spiral coordinates as numpy arrays")
        return (np.array(spiral_x_coords), np.array(spiral_y_coords))

    def _generate_xy_coordinates(
        self,
        n_distance: float = 0.0,
        direction: int = 0,
        angular_offset: float = 0,
        index: int = 0,
    ) -> Tuple[float, float]:
        """
        Description:
            Generate spiral coordinates based on the angular offset and direction.
        Args:
            angular_offset: Angular offset for the spiral.
            direction: Direction of the spiral (1 or -1).
            index: Index of the spiral.
        Raises:
            ValueError: If the direction is not 1 or -1.
            ValueError: If the index is not a non-negative integer.
        Notes:
            This function generates the x and y coordinates of the spiral based on the angular offset and direction.
            The coordinates are generated using the cosine and sine functions, respectively.
        Returns:
            Tuple[np.ndarray, np.ndarray]: x and y coordinates of the spiral.
        """
        self.logger.trace(f"SpiralProblem: generate_xy_coordinates: Generating spiral coordinates with angular_offset: {angular_offset}, direction: {direction}")
        if direction not in [1, -1]:
            raise ValueError(f"SpiralProblem: generate_xy_coordinates: Direction must be 1 or -1, but got {direction}.")
        self.logger.trace("SpiralProblem: generate_xy_coordinates: Generating spiral coordinates")
        # Generate the x and y coordinates of the spiral
        spiral_x = self._make_coords(index=index, n_distance=n_distance, angular_offset=angular_offset, direction=direction, trig_function=np.cos)
        self.logger.verbose(f"SpiralProblem: generate_xy_coordinates: Spiral X coordinates for index {index}: {spiral_x}")
        spiral_y = self._make_coords(index=index, n_distance=n_distance, angular_offset=angular_offset, direction=direction, trig_function=np.sin)
        self.logger.verbose(f"SpiralProblem: generate_xy_coordinates: Spiral Y coordinates for index {index}: {spiral_y}")
        self.logger.debug(f"SpiralProblem: generate_xy_coordinates: Generated spiral coordinates with angular_offset: {angular_offset}, direction: {direction}")
        return (spiral_x, spiral_y)

    def _make_coords(
        self,
        index: int = 0,
        n_distance: float = 0.0,
        angular_offset: float = 0.0,
        direction: int = 0,
        trig_function: callable = None,
    ) -> np.ndarray:
        self.logger.trace("SpiralProblem: make_coords: Generating coordinates for the spiral")
        self.logger.verbose(f"SpiralProblem: make_coords: Generating coordinates for spiral with index {index}, n_distance {n_distance}, angular_offset {angular_offset}, direction {direction}")
        self.logger.verbose(f"SpiralProblem: make_coords: Using trig_function: {trig_function.__name__ if trig_function else None}")
        if trig_function is None:
            raise ValueError("SpiralProblem: make_coords: trig_function must be provided.")
        if direction not in [1, -1]:
            raise ValueError(f"SpiralProblem: make_coords: Direction must be 1 or -1, but got {direction}.")
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"SpiralProblem: make_coords: Index must be a non-negative integer, but got {index}.")
        self.logger.trace("SpiralProblem: make_coords: Generating coordinates for the spiral")
        # Generate the coordinates of the spiral using the provided trig_function
        self.logger.trace("SpiralProblem: make_coords: Generating coordinates for the spiral using the provided trig_function")
        self.logger.debug(f"SpiralProblem: make_coords: Generating coordinates for spiral with index {index}, n_distance {n_distance}, angular_offset {angular_offset}, direction {direction}")
        self.logger.verbose(f"SpiralProblem: make_coords: Using trig_function: {trig_function.__name__ if trig_function else None}")
        coords = direction * trig_function(n_distance + angular_offset * index) * n_distance + self._make_noise(self.n_points, self.noise)
        self.logger.debug(f"SpiralProblem: make_coords: Generated coordinates for spiral with direction {direction}: {coords}")
        self.logger.trace("SpiralProblem: make_coords: Completed generating coordinates for the spiral")
        # Return the coordinates
        return coords

    def _make_noise(
        self,
        n_points: int = 0,
        noise: float = 0.0
    ) -> np.ndarray:
        """
        Description:
            Generate random noise for the spiral points. The noise is generated based on a uniform distribution.
        Args:
            n_points: Number of points to generate noise for.
            noise: Factor to scale the noise.
        Returns:
            np.ndarray: Array of random noise values.
        """
        return np.random.rand(n_points) * noise

    def _create_input_features(
        self,
        spiral_x_coords: np.ndarray = None,
        spiral_y_coords: np.ndarray = None,
    ) -> np.ndarray:
        """
        Description:
            Create input features from the spiral coordinates.
        Args:
            spiral_x_coords: List of x coordinates for each spiral.
            spiral_y_coords: List of y coordinates for each spiral.
        Returns:
            np.ndarray: Input features as a 2D numpy array.
        """
        self.logger.trace("SpiralProblem: create_input_features: Creating input features from spiral coordinates")
        # Stack the x and y coordinates vertically and transpose to get the correct shape
        self.logger.trace("SpiralProblem: create_input_features: Stacking the x and y coordinates vertically and transposing to get the correct shape")
        x = np.vstack([ np.hstack(spiral_x_coords), np.hstack(spiral_y_coords) ]).T
        self.logger.verbose(f"SpiralProblem: create_input_features: Created input features with shape: {x.shape}, Values:\n{x}")
        self.logger.trace("SpiralProblem: create_input_features: Completed creating input features")
        return x

    def _create_one_hot_targets(self, total_points: int, n_spirals: int, dtype: np.dtype) -> np.ndarray:
        """
        Description:
            Create one-hot encoded targets for the spiral dataset.
        Args:
            total_points: Total number of points in the dataset.
            n_spirals: Number of spirals in the dataset.
            dtype: Data type for the one-hot encoded targets.
        Returns:
            np.ndarray: One-hot encoded targets as a 2D numpy array.
        Raises:
            ValueError: If total_points or n_spirals are not positive integers.
        Notes:
            This function creates a one-hot encoded target array for the spirals, where each spiral corresponds to a column in the array.
            The one-hot encoding is done by setting the corresponding column to 1 for the points belonging to that spiral.
        """
        self.logger.trace("SpiralProblem: create_one_hot_targets: Creating one-hot encoded targets")
        y = np.zeros((total_points, n_spirals), dtype=dtype)  # Changed to float32 to match PyTorch tensor
        self.logger.debug(f"SpiralProblem: create_one_hot_targets: Created empty, one-hot encoded targets with shape: {y.shape}, dtype: {dtype}")
        # Set one-hot encoding for each spiral
        for i in range(self.n_spirals):
            start_idx = i * self.n_points
            end_idx = (i + 1) * self.n_points
            y[start_idx:end_idx, i] = 1
        self.logger.debug(f"SpiralProblem: create_one_hot_targets: Created one-hot encoded targets: Shape: {y.shape}, dtype: {dtype}")
        self.logger.trace("SpiralProblem: create_one_hot_targets: Completed creating one-hot encoded targets")
        # Return the one-hot encoded targets
        return y

    def _create_spiral_dataset(
        self,
        spiral_x_coords: tuple = None, # Tuple of x coordinates for the spirals
        spiral_y_coords: tuple = None, # Tuple of y coordinates for the spirals
    ):
        """
        Description:
            Create a dataset of n spirals from the given x and y coordinates.
        Args:
            spiral_x_coords: Tuple of x coordinates for the spirals.
            spiral_y_coords: Tuple of y coordinates for the spirals.
        Raises:
            ValueError: If spiral_x_coords or spiral_y_coords are None.
        Notes:
            This function creates a dataset of n spirals from the given x and y coordinates.
            It generates the input features and one-hot encoded targets for the spirals.
        Returns:
            tuple: A tuple containing the input features and one-hot encoded targets.
        """
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Creating spiral dataset from coordinates")
        # Create input features (same structure as original)
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Creating input features")
        x = self._create_input_features( spiral_x_coords=spiral_x_coords, spiral_y_coords=spiral_y_coords,)
        # Create targets (one-hot encoded for N classes)
        y = self._create_one_hot_targets(total_points=self.total_points, n_spirals=self.n_spirals, dtype=np.float32)
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Created input features x: Shape: {x.shape}, Type: {type(x)}")
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Created targets y: Shape: {y.shape}, Type: {type(y)}")
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Completed creation of spiral dataset")
        return (x, y)  # Return the input features and targets

    def _convert_to_tensors(self, x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Description:
            Convert numpy arrays to PyTorch tensors.
        Args:
            x: Input features as a numpy array.
            y: Target labels as a numpy array.
        Raises:
            ValueError: If x or y are not numpy arrays.
        Notes:
            This function converts the input features and target labels from numpy arrays to PyTorch tensors.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input features and target labels as PyTorch tensors.
        """
        self.logger.trace("SpiralProblem: convert_to_tensors: Converting data to PyTorch tensors")
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)  # Direct conversion, no scatter_ needed
        self.logger.debug(f"SpiralProblem: convert_to_tensors: Final shapes - x: {x_tensor.shape}, y: {y_tensor.shape}")
        self.logger.debug(f"SpiralProblem: convert_to_tensors: Final types - x: {type(x_tensor)}, y: {type(y_tensor)}")
        self.logger.trace("SpiralProblem: convert_to_tensors: Completed conversion to PyTorch tensors")
        return x_tensor, y_tensor

    def _shuffle_dataset(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Description:
            Shuffle the dataset.
        Args:
            x_tensor: Input features tensor.
            y_tensor: Target tensor.
        Raises:
            ValueError: If x_tensor and y_tensor are not of the same size.
        Notes:
            This function shuffles the dataset by randomly permuting the indices of the input features and targets.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Shuffled input features and targets tensors.
        """
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Shuffling the dataset")
        if x_tensor.size(0) != y_tensor.size(0):
            raise ValueError(f"SpiralProblem: generate_n_spiral_dataset: x_tensor and y_tensor must be of the same size, but got {x_tensor.size(0)} and {y_tensor.size(0)} respectively.")
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Shuffling dataset with {x_tensor.size(0)} points")
        # Shuffle the dataset by randomly permuting the  indices of the input features and targets
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Randomly permuting indices for shuffling")
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: x_tensor shape: {x_tensor.shape}, Type: {type(x_tensor)}, y_tensor shape: {y_tensor.shape}, Type: {type(y_tensor)}")
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Shuffling dataset with {x_tensor.size(0)} points")
        indices = torch.randperm(x_tensor.size(0))
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Generated random indices for shuffling: {indices}")
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Shuffling input features and targets using the generated indices")
        # Shuffle the input features and targets using the generated indices
        x_shuffled = x_tensor[indices]
        y_shuffled = y_tensor[indices]
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Shuffled x shape: {x_shuffled.shape}, Type: {type(x_shuffled)}, y shape: {y_shuffled.shape}, Type: {type(y_shuffled)}")
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Completed shuffling the dataset")
        # Return the shuffled input features and targets
        return (x_shuffled, y_shuffled)

    def _partition_dataset(
        self,
        total_points: int = 0,
        partitions: Tuple[float, float] = None,
        x: torch._C._TensorMeta = None,
        y: torch._C._TensorMeta = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Description:
            Partition the dataset into training and test sets based on the specified ratios.
        Args:
            total_points: Total number of points in the dataset.
            partitions: Tuple of ratios to split the dataset. Must sum to 1.0.
            x: Input features as a PyTorch tensor.
            y: One-hot encoded targets as a PyTorch tensor.
        Returns:
            tuple: A tuple containing the training and test sets.
                - (x_train, y_train): Training set features and targets.
                - (x_test, y_test): Test set features and targets.
        Raises:
            ValueError: If total_points or partitions are None, if x and y are not provided, or if partitions do not sum to 1.0.
        Notes:
            - The function splits the dataset into training and test sets based on the provided ratios.
            - The function returns the training and test sets as tuples of PyTorch tensors.
        """
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Starting to partition dataset into training and test sets")
        # Initialize train and test ratios for splitting the dataset
        self.train_ratio = partitions[0] if partitions[0] is not None else _SPIRAL_PROBLEM_TRAIN_RATIO if self.train_ratio is None else self.train_ratio  # Use class attribute if train_ratio is None
        self.test_ratio = partitions[1] if partitions[1] is not None else _SPIRAL_PROBLEM_TEST_RATIO if self.test_ratio is None else self.test_ratio  # Use class attribute if test_ratio is None
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Train Ratio: {self.train_ratio}, Test Ratio: {self.test_ratio}")
        # Validate train and test ratios
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Validating train and test ratios")
        if not np.isclose(self.train_ratio + self.test_ratio, 1.0):  # Validate ratios
            raise ValueError("SpiralProblem: generate_n_spiral_dataset: Train and test ratios must sum to 1.0")
        self.logger.debug(f"SpiralProblem: generate_n_spiral_dataset: Shuffled Tensor data: x: Shape: {x.shape}, Type: {type(x)}, y: Shape: {y.shape}, y: Type: {type(y)}")
        partitioned_dataset = self._split_dataset(total_points=self.total_points, partitions=(self.train_ratio, self.test_ratio), x=x, y=y)
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Partitioned dataset: Type: {type(partitioned_dataset)}, Length: {len(partitioned_dataset)}, Values:\n{partitioned_dataset}")
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Partitioned dataset Elements: element 1: Type: {type(partitioned_dataset[0])}, Length: {len(partitioned_dataset[0])}, Value:\n{partitioned_dataset[0]}")
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Partitioned dataset Elements: element 2: Type: {type(partitioned_dataset[1])}, Length: {len(partitioned_dataset[1])}, Value:\n{partitioned_dataset[1]}")
        partition_train, partition_test = partitioned_dataset
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Partitioned dataset: Train: Type: {type(partition_train)}, Length: {len(partition_train)}, Value:\n{partition_train}")
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Partitioned dataset: Test: Type: {type(partition_test)}, Length: {len(partition_test)}, Value:\n{partition_test}")
        self.logger.trace("SpiralProblem: generate_n_spiral_dataset: Completed partitioning dataset into training and test sets")
        return (partition_train, partition_test)

    def _split_dataset(
        self,
        total_points: int = None,
        partitions: tuple = None,
        x: torch._C._TensorMeta = None,
        y: torch._C._TensorMeta = None,
    ) -> tuple:
        """
        Description:
            Split the dataset into partitions based on the provided ratios. The dataset is split into training and test sets based on the given ratios.
        Args:
            total_points: Total number of points in the dataset.
            partitions: Tuple of ratios to split the dataset. Must sum to 1.0.
            x: Input features as a PyTorch tensor.
            y: One-hot encoded targets as a PyTorch tensor.
        Returns:
            tuple: A tuple containing the dataset partitions.
        Raises:
            ValueError: If total_points or partitions are None, if x and y are not provided, or if partitions do not sum to 1.0.
        """
        self.logger.trace("SpiralProblem: split_dataset: Starting to split dataset into partitions")
        if total_points is None or partitions is None: # Check if total_points and partitions are provided
            raise ValueError("SpiralProblem: split_dataset: total_points and partitions must be provided.")
        if x is None or y is None: # Check if x and y are provided
            raise ValueError("SpiralProblem: split_dataset: Torch Tensors, x and y must be provided.")
        if not isinstance(total_points, int) or total_points <= 0: # Check if total_points is a positive integer
            raise ValueError(f"SpiralProblem: split_dataset: total_points must be a positive integer, but got {total_points}.")
        if not isinstance(partitions, tuple) or not all(isinstance(p, (float, int)) for p in partitions): # Check if partitions is a tuple of floats
            raise ValueError(f"SpiralProblem: split_dataset: partitions must be a tuple of floats, but got {partitions}.")
        if not partitions: # Check if the length of partitions is greater than 0
            raise ValueError("SpiralProblem: split_dataset: partitions must contain at least one partition.")
        if x.size(0) != total_points or y.size(0) != total_points: # Check if x and y have the same length as total_points
            raise ValueError(f"SpiralProblem: split_dataset: x and y must have the same length as total_points, but got x: {x.size(0)}, y: {y.size(0)}, total_points: {total_points}.")
        partitions_sum = sum(partitions)
        if not np.isclose(partitions_sum, 1.0): # Check if partitions are valid
            raise ValueError(f"SpiralProblem: split_dataset: Partitions must sum to 1.0, but got {partitions_sum}.")
        self.logger.trace("SpiralProblem: split_dataset: Calculating dataset partitions based on provided ratios")
        self.logger.debug(f"SpiralProblem: split_dataset: Splitting dataset with total points: {total_points}, partitions: {partitions}, x shape: {x.shape}, y shape: {y.shape}.")
        dataset_partitions = ()
        partition_start = 0
        for partition in partitions:  # Calculate the dataset partitions based on the provided ratios
            if not (0.0 <= partition <= 1.0):
                raise ValueError(f"SpiralProblem: split_dataset: Partition {partition} must be between 0.0 and 1.0.")
            partition_end = self._find_partition_index_end( partition_start=partition_start, total_points=total_points, partition=partition)
            current_partition = x[partition_start:partition_end], y[partition_start:partition_end]
            self.logger.debug(f"SpiralProblem: split_dataset: Current Partition: Length: {len(current_partition)}, Value:\n{current_partition}")
            self.logger.debug(f"SpiralProblem: split_dataset: Pre-appended length of dataset_partitions: {len(dataset_partitions)}")
            dataset_partitions = (current_partition,) if len(dataset_partitions) == 0 else dataset_partitions + (current_partition,)
            self.logger.debug(f"SpiralProblem: split_dataset: Current Value of Dataset Partitions: Partition Start: {partition_start}, Partition End: {partition_end}, Partition Ratio: {partition}. Value:\n{dataset_partitions}")
            partition_start = partition_end
        self.logger.debug(f"SpiralProblem: split_dataset: Dataset partitions Indices created with {len(dataset_partitions)} partitions, Partition Index Values:\n{dataset_partitions}.")
        self.logger.trace("SpiralProblem: split_dataset: Completed splitting dataset into partitions.")
        return dataset_partitions

    def _find_partition_index_end(
        self,
        partition_start: int,
        total_points: int,
        partition: float
    ) -> int:
        """
        Description:
            Calculate the index end for a given partition based on the start index, total points, and partition ratio.
        Args:
            partition_start: Starting index for the partition.
            total_points: Total number of points in the dataset.
            partition: Ratio to split the dataset.
        Returns:
            int: Index end for the partition.
        """
        self.logger.trace("SpiralProblem: find_partition_index_end: Starting to calculate partition end index")
        self.logger.debug(f"SpiralProblem: find_partition_index_end: Calculating partition end index for partition start: {partition_start}, total points: {total_points}, partition ratio: {partition}")
        partition_end = partition_start + self._dataset_split_index_end( total_points=total_points, split_ratio=partition,)
        self.logger.debug(f"SpiralProblem: find_partition_index_end: Calculated partition end index: {partition_end}")
        self.logger.trace("SpiralProblem: find_partition_index_end: Completed calculating partition end index")
        return partition_end

    def _dataset_split_index_end(
        self,
        total_points: int=0,
        split_ratio: float = 0.0,
    ) -> int:
        """
        Description:
            Calculate the index end for splitting the dataset based on the split ratio.
        Args:
            total_points: Total number of points in the dataset.
            split_ratio: Ratio to split the dataset.
        Returns:
            int: Index end for splitting the dataset.
        Raises:
            ValueError: If the calculated index end is not within the valid range.
        """
        self.logger.trace("SpiralProblem: dataset_split_index_end: Starting to calculate dataset split index end")
        index_end = int(split_ratio * total_points)  # Calculate the center index for splitting
        self.logger.debug(f"SpiralProblem: dataset_split_index_end: Calculated index end for split ratio {split_ratio} and total points {total_points}: {index_end}")
        if index_end < 0 or index_end > total_points:
            raise ValueError(f"SpiralProblem: dataset_split_index_end: Invalid index end: {index_end}. Must be between 0 and {total_points}.")
        self.logger.debug(f"SpiralProblem: dataset_split_index_end: Valid index end: {index_end}")
        self.logger.trace("SpiralProblem: dataset_split_index_end: Completed calculating dataset split index end")
        return index_end


    #####################################################################################################################################################################################################
    # Define function to solve the two spiral problem using Spiral Problem.
    def solve_n_spiral_problem(
        self,
        n_points=None,
        n_spirals=None,
        n_rotations=None,
        clockwise=None,
        noise=None,
        distribution=None,
        test_ratio=None,
        train_ratio=None,
        plot=None,
    ) -> Tuple[tuple, tuple, tuple]:
        """
        Description:
            Solve the two spiral problem using Spiral Problem. The dataset is split into training and test sets based on the given ratios.
        Args:
            n_points: Number of points per spiral
            n_spirals: Number of spirals to generate
            n_rotations: Number of rotations for each spiral
            clockwise: True for clockwise spirals, False for counter-clockwise
            distribution: Factor to apply to the degrees of the spiral points
            test_ratio: Ratio of test data to total data. Must be between 0.0 and 1.0.
            train_ratio: Ratio of training data to total data. Must be
            noise: Amount of noise to add
            plot: Whether to plot the results
        Returns:
            Trained SpiralProblem
        """
        self.logger.trace("SpiralProblem: solve_n_spiral_problem: Starting to solve N Spiral Problem")

        # Initialize Spiral Problem input parameters
        self.logger.trace("SpiralProblem: solve_n_spiral_problem: Initializing Spiral Problem input parameters")
        self.n_points = n_points or self.n_points or _SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL  # Use class attribute if n_points is None
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Number of points per spiral: {self.n_points}")
        self.n_spirals = n_spirals or self.n_spirals or _SPIRAL_PROBLEM_NUM_SPIRALS  # Use class attribute if n_spirals is None
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Number of spirals to generate: {self.n_spirals}")
        self.n_rotations = n_rotations or self.n_rotations or _SPIRAL_PROBLEM_NUM_ROTATIONS  # Use class attribute if n_rotations is None
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Number of rotations for each spiral: {self.n_rotations}")
        self.clockwise = clockwise or self.clockwise or _SPIRAL_PROBLEM_CLOCKWISE  # Use class attribute if clockwise is None
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Clockwise spirals: {'Yes' if self.clockwise else 'No'}")
        self.noise = noise or self.noise or _SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT  # Use class attribute if noise is None
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Noise level: {self.noise}")
        self.distribution = distribution or self.distribution or _SPIRAL_PROBLEM_DISTRIBUTION_FACTOR  # Use class attribute if distribution is None
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Distribution factor: {self.distribution}")
        self.test_ratio = test_ratio or self.test_ratio or _SPIRAL_PROBLEM_TEST_RATIO  # Use class attribute if test_ratio is None
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Test Ratio: {self.test_ratio}")
        self.train_ratio = train_ratio or self.train_ratio or _SPIRAL_PROBLEM_TRAIN_RATIO  # Use class attribute if train_ratio is None
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Train Ratio: {self.train_ratio}")
        self.plot = plot or self.plot or _SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT  # Use class attribute if plot is None
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Plotting results: {'Enabled' if self.plot else 'Disabled'}")

        # Generate the n spiral dataset
        self.logger.trace("SpiralProblem: solve_n_spiral_problem: Generating N Spiral dataset")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Generating two spiral dataset with {self.n_points} points and noise level {self.noise}.")

        # Generate the full N Spiral dataset, new version
        self.logger.trace("SpiralProblem: solve_n_spiral_problem: Generating full N Spiral dataset, new version")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Initialized Values for Parameters: n_spirals: {self.n_spirals}, n_points: {self.n_points}, n_rotations: {self.n_rotations}, clockwise: {self.clockwise}, noise: {self.noise}, distribution: {self.distribution}, train_ratio: {self.train_ratio}, test_ratio: {self.test_ratio}")
        dataset_partitions = self.generate_n_spiral_dataset(
            n_spirals=self.n_spirals,
            n_points=self.n_points,
            n_rotations=self.n_rotations,
            clockwise=self.clockwise,
            noise_level=self.noise,
            distribution=self.distribution,
            train_ratio=self.train_ratio,
            test_ratio=self.test_ratio,
        )
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Partitioned dataset: Size: {len(dataset_partitions)}, Type: {type(dataset_partitions)}, Value:\n{dataset_partitions}")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Generated N spiral dataset with {self.n_points} points and noise level {self.noise}.")

        # Unpack the dataset partitions
        self.logger.trace("SpiralProblem: solve_n_spiral_problem: Unpacking dataset partitions")
        train_partition, test_partition, full_partition = dataset_partitions
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Train Partition: Type: {type(train_partition)}, Length: {len(train_partition)}, Value:\n{train_partition}")
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Test Partition: Type: {type(test_partition)}, Length: {len(test_partition)}, Value:\n{test_partition}")
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Full Partition: Type: {type(full_partition)}, Length: {len(full_partition)}, Value:\n{full_partition}")
        self.x_train, self.y_train = train_partition
        self.x_test, self.y_test = test_partition
        self.x_full, self.y_full = full_partition

        self.logger.info(f"SpiralProblem: solve_n_spiral_problem: Dataset x_full: Shape: {self.x_full.shape}, Type: {type(self.x_full)}, Value:\n{self.x_full}")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Dataset x_full: Shape: {self.x_full.shape}, Type: {type(self.x_full)}, Value:\n{self.x_full}")
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Dataset x_full: Shape: {self.x_full.shape}, Type: {type(self.x_full)}, Value:\n{self.x_full}")

        self.logger.info(f"SpiralProblem: solve_n_spiral_problem: Dataset y_full: Shape: {self.y_full.shape}, Type: {type(self.y_full)}, Value:\n{self.y_full}")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Dataset y_full: Shape: {self.y_full.shape}, Type: {type(self.y_full)}, Value:\n{self.y_full}")
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Dataset y_full: Shape: {self.y_full.shape}, Type: {type(self.y_full)}, Value:\n{self.y_full}")

        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Dataset x_train: Shape: {self.x_train.shape}, Type: {type(self.x_train)}, Value:\n{self.x_train}")
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Dataset y_train: Shape: {self.y_train.shape}, Type: {type(self.y_train)}, Value:\n{self.y_train}")
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Dataset x_test: Shape: {self.x_test.shape}, Type: {type(self.x_test)}, Value:\n{self.x_test}")
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Dataset y_test: Shape: {self.y_test.shape}, Type: {type(self.y_test)}, Value:\n{self.y_test}")

        self.x = self.x_train  # Use training data for fitting the network
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Full dataset x: Shape: {self.x.shape}, Type: {type(self.x)}, Value:\n{self.x}")
        self.y = self.y_train  # Use training labels for fitting the network
        self.logger.verbose(f"SpiralProblem: solve_n_spiral_problem: Full dataset y: Shape: {self.y.shape}, Type: {type(self.y)}, Value:\n{self.y}")

        # Perform initial plot of the 2 Spiral Problem dataset
        self.logger.trace("SpiralProblem: solve_n_spiral_problem: Performing initial plot of the N Spiral Problem dataset")
        if self.plot:
            self.logger.debug("SpiralProblem: solve_n_spiral_problem: Performing initial plot of the N Spiral Problem dataset")
            # self.network.plot_dataset(
            #     x=self.x_full,
            #     y=self.y_full,
            #     title=f"N Spiral Problem: {self.n_spirals} Spirals, {self.n_points} Points Each, Noise Factor: {self.noise}",
            # )
            # plotter = mp.Process(target=self.network.plot_dataset, args=(self.x_full, self.y_full), kwargs={"title":f"N Spiral Problem: {self.n_spirals} Spirals, {self.n_points} Points Each, Noise Factor: {self.noise}",})
            # plotter_data = (self.x_full, self.y_full, {"title":f"N Spiral Problem: {self.n_spirals} Spirals, {self.n_points} Points Each, Noise Factor: {self.noise}"})
            plotter_data = (self.x_full, self.y_full, f"N Spiral Problem: {self.n_spirals} Spirals, {self.n_points} Points Each, Noise Factor: {self.noise}")
            # Use spawn context to avoid forkserver module reimport issues
            spawn_ctx = mp.get_context("spawn")
            self.plotter = spawn_ctx.Process(target=CascadeCorrelationNetwork.plot_dataset, args=plotter_data)
            self.plotter.start()
            # plotter.join()
            self.logger.trace("SpiralProblem: solve_n_spiral_problem: Completed initial plot of the N Spiral Problem dataset")
            # plotter.terminate()

        # Train the network
        self.logger.trace("SpiralProblem: solve_n_spiral_problem: Training the network on the N Spiral Problem dataset")
        self.logger.debug("SpiralProblem: solve_n_spiral_problem: Created Spiral Problem...")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Spiral Problem: \n{self.network}")
        self.history = self.network.fit(self.x, self.y, max_epochs=_SPIRAL_PROBLEM_OUTPUT_EPOCHS,)
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Training history: {self.history}")

        # Print summary
        self.logger.trace("SpiralProblem: solve_n_spiral_problem: Printing network summary")
        self.network.summary()

        # Plot results
        self.logger.trace("SpiralProblem: solve_n_spiral_problem: Plotting decision boundary")
        if self.plot:
            self.network.plot_decision_boundary(self.x, self.y, "N Spiral Problem - Decision Boundary")
            self.network.plot_training_history()
        self.logger.trace("SpiralProblem: solve_n_spiral_problem: Completed solving N Spiral Problem")
        if self.plot:
            plt.show()
            self.plotter.join()

    def evaluate(
        self,
        n_points=None,
        n_spirals=None,
        n_rotations=None,
        clockwise=None,
        noise=None,
        distribution=None,
        plot=None,
        train_ratio=None,
        test_ratio=None,
        random_value_scale=None,
        default_origin=None,
        default_radius=None,
    ) -> None:
        """
        Description:
            Evaluate the two spiral problem using the Spiral Problem class. This function initializes the parameters for the two spiral problem,
            generates the dataset, trains the network, and evaluates the accuracy on the training and test sets.
        Args:
            n_points: Number of points per spiral
            n_spirals: Number of spirals to generate
            n_rotations: Number of rotations for each spiral
            clockwise: True for clockwise spirals, False for counter-clockwise
            noise: Amount of noise to add
            distribution: Factor to apply to the degrees of the spiral points
            plot: Whether to plot the results
            train_ratio: Ratio of training data to total data. Must be between 0.0 and 1.0.
            test_ratio: Ratio of test data to total data. Must be between 0.0 and 1.0.
            random_value_scale: Scale for random values
            default_origin: Default origin for the spirals
            default_radius: Default radius for the spirals
        Raises:
            ValueError: If the input parameters are invalid.
        Notes:
            - The function initializes the parameters for the two spiral problem.
            - It generates the dataset using the `generate_n_spiral_dataset` method.
            - It trains the network using the `fit` method.
            - It evaluates the accuracy on the training and test sets using the `calculate_accuracy` method.
            - It plots the decision boundary and training history if `plot` is True.
        Returns:
            None
        """
        self.logger.trace("SpiralProblem: evaluate: Starting evaluation of the two spiral problem")

        # Set parameters for the two spiral problem
        self.logger.trace("SpiralProblem: evaluate: Setting parameters for the two spiral problem")
        self.n_points = n_points or self.n_points or _SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL
        self.logger.verbose(f"SpiralProblem: evaluate: Number of points per spiral: {self.n_points}")
        self.n_spirals = n_spirals or self.n_spirals or _SPIRAL_PROBLEM_NUM_SPIRALS
        self.logger.verbose(f"SpiralProblem: evaluate: Number of spirals to generate: {self.n_spirals}")
        self.n_rotations = n_rotations or self.n_rotations or _SPIRAL_PROBLEM_NUM_ROTATIONS
        self.logger.verbose(f"SpiralProblem: evaluate: Number of rotations for each spiral: {self.n_rotations}")
        self.clockwise = clockwise or self.clockwise or _SPIRAL_PROBLEM_CLOCKWISE
        self.logger.verbose(f"SpiralProblem: evaluate: Clockwise spirals: {'Yes' if self.clockwise else 'No'}")
        self.noise = noise or self.noise or _SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT
        self.logger.verbose(f"SpiralProblem: evaluate: Noise level: {self.noise}")
        self.distribution = distribution or self.distribution or _SPIRAL_PROBLEM_DISTRIBUTION_FACTOR
        self.logger.verbose(f"SpiralProblem: evaluate: Distribution factor: {self.distribution}")
        self.plot = plot or self.plot or _SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT
        self.logger.verbose(f"SpiralProblem: evaluate: Plotting results: {'Enabled' if self.plot else 'Disabled'}")
        self.train_ratio = train_ratio or self.train_ratio or _SPIRAL_PROBLEM_TRAIN_RATIO
        self.logger.verbose(f"SpiralProblem: evaluate: Training data ratio: {self.train_ratio}")
        self.test_ratio = test_ratio or self.test_ratio or _SPIRAL_PROBLEM_TEST_RATIO
        self.logger.verbose(f"SpiralProblem: evaluate: Test data ratio: {self.test_ratio}")
        self.random_value_scale = random_value_scale or self.random_value_scale or _SPIRAL_PROBLEM_RANDOM_VALUE_SCALE
        self.logger.verbose(f"SpiralProblem: evaluate: Random value scale: {self.random_value_scale}")
        self.default_origin = default_origin or self.default_origin or _SPIRAL_PROBLEM_DEFAULT_ORIGIN
        self.logger.verbose(f"SpiralProblem: evaluate: Default origin for spirals: {self.default_origin}")
        self.default_radius = default_radius or self.default_radius or _SPIRAL_PROBLEM_DEFAULT_RADIUS
        self.logger.verbose(f"SpiralProblem: evaluate: Default radius for spirals: {self.default_radius}")

        # Solve the two spiral problem
        self.logger.trace("SpiralProblem: main: Solving the two spiral problem with Cascade Correlation...")
        self.solve_n_spiral_problem(
            n_points=self.n_points,
            noise=self.noise,
            n_spirals=self.n_spirals,
            n_rotations=self.n_rotations,
            clockwise=self.clockwise,
            distribution=self.distribution,
            test_ratio=self.test_ratio,
            train_ratio=self.train_ratio,
            plot=self.plot,
        )
        # self.logger.verbose(f"SpiralProblem: main: Training Data: Type: {type(self.train_data)}, Shape: {self.train_data.shape}, Value:\n{self.train_data}")
        # self.logger.verbose(f"SpiralProblem: main: Test Data: Type: {type(self.test_data)}, Shape: {self.test_data.shape}, Value:\n{self.test_data}")

        # Print training dataset shapes and types
        # self.logger.verbose(f"SpiralProblem: main: Train Data: X Shape: {self.train_x.shape}, X Type: {type(self.train_x)}, X Value:\n{self.train_x}")
        self.logger.verbose(f"SpiralProblem: main: Train Data: X Shape: {self.x_train.shape}, X Type: {type(self.x_train)}, X Value:\n{self.x_train}")
        # self.logger.verbose(f"SpiralProblem: main: Train Data: Y Shape: {self.train_y.shape}, Y Type: {type(self.train_y)}, Y Value:\n{self.train_y}")
        self.logger.verbose(f"SpiralProblem: main: Train Data: Y Shape: {self.y_train.shape}, Y Type: {type(self.y_train)}, Y Value:\n{self.y_train}")

        # Print test dataset shapes and types
        # self.logger.verbose(f"SpiralProblem: main: Test Data: X Shape: {self.test_x.shape}, X Type: {type(self.test_x)}, X Value:\n{self.test_x}")
        self.logger.verbose(f"SpiralProblem: main: Test Data: X Shape: {self.x_test.shape}, X Type: {type(self.x_test)}, X Value:\n{self.x_test}")
        # self.logger.verbose(f"SpiralProblem: main: Test Data: Y Shape: {self.test_y.shape}, Y Type: {type(self.test_y)}, Y Value:\n{self.test_y}")
        self.logger.verbose(f"SpiralProblem: main: Test Data: Y Shape: {self.y_test.shape}, Y Type: {type(self.y_test)}, Y Value:\n{self.y_test}")

        # Calculate and log the accuracy on the training and test sets
        self.train_accuracy = self.network.calculate_accuracy(x=self.x_train, y=self.y_train)
        self.logger.verbose(f"SpiralProblem: main: Train accuracy on the two spiral problem: {self.train_accuracy:.4f}")
        self.test_accuracy = self.network.calculate_accuracy(x=self.x_test, y=self.y_test)
        self.logger.verbose(f"SpiralProblem: main: Test accuracy on the two spiral problem: {self.test_accuracy:.4f}")

        # Evaluate the Final Accuracy Percentages
        self.train_accuracy_percent = self.train_accuracy * 100
        self.logger.verbose(f"SpiralProblem: main: Final Train accuracy on the two spiral problem: {self.train_accuracy_percent:.2f}%")
        self.test_accuracy_percent = self.test_accuracy * 100
        self.logger.verbose(f"SpiralProblem: main: Final Test accuracy on the two spiral problem: {self.test_accuracy_percent:.2f}%")

        # Print final accuracy
        self.logger.trace("SpiralProblem: main: Printing final accuracy")
        self.network.summary()
        self.logger.debug(f"SpiralProblem: main: Training History: {self.history}")
        self.logger.debug(f"SpiralProblem: main: Dataset: Train: x:\n{self.x_train}\ny:\n{self.y_train}\nTest: x:\n{self.x_test}\ny:\n{self.y_test}")
        self.logger.debug(f"SpiralProblem: main: Dataset shape: Train: x: {self.x_train.shape}, y: {self.y_train.shape}, Test: x: {self.x_test.shape}, y: {self.y_test.shape}")
        self.logger.debug(f"SpiralProblem: main: Final accuracy on the two spiral problem: Training: {self.train_accuracy_percent:.2f}%")
        self.logger.debug(f"SpiralProblem: main: Final accuracy on the two spiral problem: Testing: {self.test_accuracy_percent:.2f}%")
        self.logger.trace("SpiralProblem: evaluate: Completed evaluation of the two spiral problem")


    ####################################################################################################################################
    # Define private methods for the SpiralProblem class
    def _generate_uuid(self):
        """
        Description:
            Generate a new UUID for the SpiralProblem class.
        Args:
            self: The instance of the class.
        Returns:
            str: The generated UUID.
        """
        self.logger.trace("SpiralProblem: _generate_uuid: Inside the SpiralProblem class Generate UUID method")
        new_uuid = str(uuid.uuid4())
        self.logger.debug(f"SpiralProblem: _generate_uuid: UUID: {new_uuid}")
        self.logger.trace("SpiralProblem: _generate_uuid: Completed the SpiralProblem class Generate UUID method")
        return new_uuid


    ####################################################################################################################################
    # Define SpiralProblem class Setters
    def set_network(self, network: CascadeCorrelationNetwork = None):
        if network is not None:
            self.network = network

    def set_logger(self, logger: logging.Logger = None):
        if logger is not None:
            self.logger = logger

    def set_n_spirals(self, n_spirals: int = None):
        if n_spirals is not None:
            self.n_spirals = n_spirals

    def set_n_points(self, n_points: int = None):
        if n_points is not None:
            self.n_points = n_points

    def set_n_rotations(self, n_rotations: int = None):
        if n_rotations is not None:
            self.n_rotations = n_rotations

    def set_clockwise(self, clockwise: bool = None):
        if clockwise is not None:
            self.clockwise = clockwise

    def set_noise(self, noise: float = None):
        if noise is not None:
            self.noise = noise

    def set_distribution(self, distribution: str = None):
        if distribution is not None:
            self.distribution = distribution

    def set_random_seed(self, random_seed: int = None):
        if random_seed is not None:
            self.random_seed = random_seed

    def set_train_ratio(self, train_ratio: float = None):
        if train_ratio is not None:
            self.train_ratio = train_ratio

    def set_test_ratio(self, test_ratio: float = None):
        if test_ratio is not None:
            self.test_ratio = test_ratio

    def set_plot(self, plot: bool = None):
        if plot is not None:
            self.plot = plot

    def set_random_value_scale(self, random_value_scale: float = None):
        if random_value_scale is not None:
            self.random_value_scale = random_value_scale

    def set_default_origin(self, default_origin: tuple = None):
        if default_origin is not None:
            self.default_origin = default_origin

    def set_default_radius(self, default_radius: float = None):
        self.default_radius = default_radius

    def set_uuid(self, uuid: str = None):
        """
        Description:
            This method sets the UUID for the SpiralProblem class.  If no UUID is provided, a new UUID will be generated.
        Args:
            uuid (str): The UUID to be set. If None, a new UUID will be generated.
        Returns:
            None
        """
        self.logger.trace("SpiralProblem: set_uuid: Inside the SpiralProblem class Set UUID method")
        self.logger.debug("SpiralProblem: set_uuid: Starting to set UUID for SpiralProblem class: Provided UUID: {uuid}, uuid is None: {uuid is None}")
        if (not hasattr(self, 'uuid')) or self.uuid is None:
            new_uuid = (uuid, self._generate_uuid())[uuid is None]  # Generate a new UUID if none is provided
            self.logger.debug(f"SpiralProblem: set_uuid: New UUID generated: {new_uuid}")
            self.uuid = new_uuid  # Set the UUID for the SpiralProblem class
            self.logger.verbose(f"SpiralProblem: set_uuid: UUID set to: {self.get_uuid()}")
            self.logger.debug(f"SpiralProblem: set_uuid: UUID was not set, generated a new one: {self.uuid}")
        else:
            self.logger.fatal(f"SpiralProblem: set_uuid: Fatal Error: UUID already set: {self.uuid}. Changing UUID is bad Juju.  Exiting...")
            os._exit(1)
        self.logger.debug(f"SpiralProblem: set_uuid: Completed setting UUID to: {self.uuid}")
        self.logger.trace("SpiralProblem: set_uuid: Completed the SpiralProblem class Set UUID method")


    ####################################################################################################################################
    # Define SpiralProblem class Getters
    def get_uuid(self) -> str:
        """
        Description:
            This method returns the UUID for the SpiralProblem class.
        Args:
            self: The instance of the class.
        Returns:
            str: The UUID for the SpiralProblem class.
        """
        self.logger.trace("SpiralProblem: get_uuid: Inside the SpiralProblem class Get UUID method")
        if not hasattr(self, "uuid") or self.uuid is None:
            self.set_uuid()  # Ensure UUID is set if not already
            self.logger.debug("SpiralProblem: get_uuid: UUID was not set, generated a new one.")
        self.logger.debug(f"SpiralProblem: get_uuid: Returning UUID: {self.uuid}")
        self.logger.trace("SpiralProblem: get_uuid: Completed the SpiralProblem class Get UUID method")
        return self.uuid

    def get_network(self) -> CascadeCorrelationNetwork:
        return self.network

    def get_n_spirals(self) -> int:
        return self.n_spirals

    def get_n_points(self) -> int:
        return self.n_points

    def get_n_rotations(self) -> int:
        return self.n_rotations

    def get_clockwise(self) -> bool:
        return self.clockwise

    def get_noise(self) -> float:
        return self.noise

    def get_distribution(self) -> str:
        return self.distribution

    def get_random_seed(self) -> int:
        return self.random_seed

    def get_train_ratio(self) -> float:
        return self.train_ratio

    def get_test_ratio(self) -> float:
        return self.test_ratio

    def get_plot(self) -> bool:
        return self.plot

    def get_random_value_scale(self) -> float:
        return self.random_value_scale

    def get_default_origin(self) -> tuple:
        return self.default_origin

    def get_default_radius(self) -> float:
        return self.default_radius
