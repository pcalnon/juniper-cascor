#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     cascade_correlation.py
# Author:        Paul Calnon
# Version:       0.3.2 (0.7.3)
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
import datetime
import datetime as pd
import logging
import logging.config
import multiprocessing as mp
import os
import pathlib as pl
import random
import sys
import time
import uuid as uuid
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from queue import Queue  # Use stdlib queue for manager-hosted objects
from typing import Any, Dict, List, Optional, Tuple, Union

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import traceback


#####################################################################################################################################################################################################
# Add current and parent dir to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#####################################################################################################################################################################################################
# Define custom manager class and server-owned queues for multiprocessing
#
# IMPORTANT: This implementation uses picklable factory functions instead of lambda functions
# to avoid PicklingError when starting worker processes with forkserver context.
# The lambda functions that were previously used cannot be pickled and would cause:
# "Can't pickle <function <lambda> at 0x...>: attribute lookup <lambda> on ... failed"
#
# Server-owned queues (live in Manager server process)

from candidate_unit.candidate_unit import (
    CandidateTrainingResult,
    CandidateUnit,
)
from cascade_correlation_config.cascade_correlation_config import (
    CascadeCorrelationConfig,
)
from cascade_correlation_exceptions.cascade_correlation_exceptions import (  # CascadeCorrelationError,; NetworkInitializationError,
    ConfigurationError,
    TrainingError,
    ValidationError,
)
from cascor_plotter.cascor_plotter import CascadeCorrelationPlotter
from log_config.log_config import LogConfig
from log_config.logger.logger import Logger
from utils.utils import (
    display_progress,
)

from constants.constants import (  # _CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_SLEEPYTIME,; _CASCADE_CORRELATION_HDF5_PROJECT_HDF5_CONSTANTS_DIR,; _CASCADE_CORRELATION_HDF5_PROJECT_CONSTANTS_DIR,; _CASCADE_CORRELATION_HDF5_PROJECT_SOURCE_DIR,; _PROJECT_MODEL_TARGET_ACCURACY,; _HDF5_PROJECT_SNAPSHOTS_DIR,; _CASCADE_CORRELATION_HDF5_PROJECT_DIR,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_RELU,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_SIGMOID,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_TANH,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_RELU,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_SIGMOID,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_TANH,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_DISPLAY_FREQUENCY,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_EPOCHS,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_LEARNING_RATE,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_POOL_SIZE,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_CONTEXT,
    _CASCADE_CORRELATION_NETWORK_DISPLAY_FREQUENCY,
    _CASCADE_CORRELATION_NETWORK_EPOCH_DISPLAY_FREQUENCY,
    _CASCADE_CORRELATION_NETWORK_EPOCHS_MAX,
    _CASCADE_CORRELATION_NETWORK_GENERATE_PLOTS,
    _CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR,
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
    _CASCADE_CORRELATION_NETWORK_RANDOM_SEED,
    _CASCADE_CORRELATION_NETWORK_RANDOM_VALUE_SCALE,
    _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE,
    _CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT,
    _CASCADE_CORRELATION_NETWORK_STATUS_DISPLAY_FREQUENCY,
    _CASCADE_CORRELATION_NETWORK_TARGET_ACCURACY,
    _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT,
    _CASCADE_CORRELATION_NETWORK_WORKER_STANDBY_SLEEPYTIME,
)


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
    epoch: int
    max_epochs: int
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
        _task_queue = Queue()
    return _task_queue

def _create_result_queue():
    """
    Factory function to create or return the result queue in the manager server process.
    This function is picklable and will be executed in the server process.
    """
    global _result_queue
    if _result_queue is None:
        _result_queue = Queue()
    return _result_queue


# Define CandidateTrainingManager Class and global functions
class CandidateTrainingManager(BaseManager):
    """Custom manager for handling candidate training queues."""
    pass

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
        Logger.debug( "CascadeCorrelationNetwork: __init__: Initializing Cascade Correlation Network")
        super().__init__()

        # Initialize configuration
        self._init_config(config)

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

    def _init_config(self, config: CascadeCorrelationConfig = None) -> None:
        """Initialize configuration and set UUID."""
        if hasattr(self, "logger") and self.logger is not None:
            logger = self.logger
        else:
            logger = Logger
        logger.debug( "CascadeCorrelationNetwork: _init_config: Initializing configuration")
        if config is None:
            config = CascadeCorrelationConfig()
        self.config = config
        logger.debug( f"CascadeCorrelationNetwork: _init_config: Configuration set to: {self.config}")
        self.set_uuid(self.config.uuid)
        logger.debug( f"CascadeCorrelationNetwork: _init_config: UUID set to: {self.uuid}")

    def _init_logging_system(self) -> None:
        """Initialize the logging system with proper configuration."""
        Logger.debug( "CascadeCorrelationNetwork: _init_logging_system: Initializing logging system")

        # Set up log parameters
        self.log_file_name = ( self.config.log_file_name or _CASCADE_CORRELATION_NETWORK_LOG_FILE_NAME or __name__)
        self.log_file_path = ( self.config.log_file_path or _CASCADE_CORRELATION_NETWORK_LOG_FILE_PATH or str(os.path.join(os.getcwd(), "logs")))
        self.log_level_name = ( self.config.log_level_name or _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME)

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
        self.logger.debug( f"CascadeCorrelationNetwork: _init_logging_system: Logger initialized with level: {self.logger.level}")

    def _init_network_parameters(self) -> None:
        """Initialize network parameters, randomness, and model components."""
        Logger.debug( "CascadeCorrelationNetwork: _init_network_parameters: Initializing network parameters")

        # Initialize randomness
        self.random_seed = ( self.config.random_seed or _CASCADE_CORRELATION_NETWORK_RANDOM_SEED)
        self.random_max_value = ( self.config.random_max_value or _CASCADE_CORRELATION_NETWORK_RANDOM_MAX_VALUE)
        self.sequence_max_value = ( self.config.sequence_max_value or _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE)
        self._initialize_randomness( seed=self.random_seed, sequence_max_value=self.sequence_max_value, random_max_value=self.random_max_value,)

        # Initialize network architecture parameters
        self.input_size = ( self.config.input_size or _CASCADE_CORRELATION_NETWORK_INPUT_SIZE)
        self.output_size = ( self.config.output_size or _CASCADE_CORRELATION_NETWORK_OUTPUT_SIZE)
        self.candidate_pool_size = ( self.config.candidate_pool_size or _CASCADE_CORRELATION_NETWORK_CANDIDATE_POOL_SIZE)

        # Initialize training parameters
        self.learning_rate = ( self.config.learning_rate or _CASCADE_CORRELATION_NETWORK_LEARNING_RATE)
        self.candidate_learning_rate = ( self.config.candidate_learning_rate or _CASCADE_CORRELATION_NETWORK_CANDIDATE_LEARNING_RATE)
        self.max_hidden_units = ( self.config.max_hidden_units or _CASCADE_CORRELATION_NETWORK_MAX_HIDDEN_UNITS)
        self.correlation_threshold = ( self.config.correlation_threshold or _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD)
        self.patience = self.config.patience or _CASCADE_CORRELATION_NETWORK_PATIENCE
        self.candidate_epochs = ( self.config.candidate_epochs or _CASCADE_CORRELATION_NETWORK_CANDIDATE_EPOCHS)
        self.epochs_max = ( self.config.epochs_max or _CASCADE_CORRELATION_NETWORK_EPOCHS_MAX)
        self.output_epochs = ( self.config.output_epochs or _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS)
        self.random_value_scale = ( self.config.random_value_scale or _CASCADE_CORRELATION_NETWORK_RANDOM_VALUE_SCALE)
        self.target_accuracy = ( self.config.candidate_training_target_accuracy or _CASCADE_CORRELATION_NETWORK_TARGET_ACCURACY)
        self.worker_standby_sleepytime = ( self.config.candidate_training_worker_standby_sleepytime or _CASCADE_CORRELATION_NETWORK_WORKER_STANDBY_SLEEPYTIME)
        self.shutdown_timeout = ( self.config.candidate_training_shutdown_timeout or _CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT)
        self.task_queue_timeout = ( self.config.candidate_training_task_queue_timeout or _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT)

        # Initialize snapshot counter for HDF5 serialization
        self.snapshot_counter = 0

        # Initialize activation function
        self._init_activation_function()

        # Initialize network model parameters)
        self.hidden_units = []
        self.output_weights = ( torch.randn( self.config.input_size, self.config.output_size, requires_grad=True) * self.random_value_scale)
        self.output_bias = ( torch.randn(self.config.output_size, requires_grad=True) * self.random_value_scale)
        self.history = {
            "train_loss": [],
            "value_loss": [],
            "train_accuracy": [],
            "value_accuracy": [],
            "hidden_units_added": [],
        }

        # Initialize snapshot counter for HDF5 serialization
        self.snapshot_counter = 0

        # Snapshot directory
        self.cascade_correlation_network_snapshots_dir = ( self.config.cascade_correlation_network_snapshots_dir or _CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR)
        self.logger.debug( "CascadeCorrelationNetwork: _init_network_parameters: Network parameters initialized")

    def _init_activation_function(self):
        """Initialize activation function components."""
        self.activation_function_name = ( self.config.activation_function_name or _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME)
        self.activation_functions_dict = ( self.config.activation_functions_dict or _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT)
        self.activation_fn_no_diff = ( self.activation_functions_dict.get( self.activation_function_name, _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,) or _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT)
        self.activation_fn = self._init_activation_with_derivative( self.activation_fn_no_diff)

    def _init_multiprocessing(self) -> None:
        """Initialize multiprocessing context and manager attributes."""
        self.logger.trace( "CascadeCorrelationNetwork: _init_multiprocessing: Initializing multiprocessing components")

        # Initialize multiprocessing context
        self._mp_ctx = mp.get_context("forkserver")
        try:
            self._mp_ctx.set_forkserver_preload(
                "os",
                "uuid",
                "torch",
                "numpy",
                "random",
                "logging",
                "datetime",
                "typing.Optional",
                "utils.utils.display_progress",
                "log_config.logger.logger.Logger",
            )
        except Exception as e:
            self.logger.warning( f"CascadeCorrelationNetwork: _init_multiprocessing: Failed to set forkserver preload: {e}")

        # Initialize manager attributes
        self._manager = None
        self._task_queue = None
        self._result_queue = None

        # Initialize multiprocessing config values
        self.candidate_training_queue_authkey = ( self.config.candidate_training_queue_authkey)
        self.candidate_training_queue_address = ( self.config.candidate_training_queue_address)
        self.candidate_training_tasks_queue_timeout = ( self.config.candidate_training_task_queue_timeout or _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT)
        self.candidate_training_shutdown_timeout = ( self.config.candidate_training_shutdown_timeout or _CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT)
        self.candidate_training_context = ( mp.get_context(self.config.candidate_training_context_type) or _CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_CONTEXT)
        self.logger.debug( "CascadeCorrelationNetwork: _init_multiprocessing: Multiprocessing components initialized")

    def _init_display_components(self) -> None:
        """Initialize display and plotting components."""
        self.logger.trace( "CascadeCorrelationNetwork: _init_display_components: Initializing display components")

        # Initialize display parameters
        self.display_frequency = ( self.config.display_frequency or _CASCADE_CORRELATION_NETWORK_CANDIDATE_DISPLAY_FREQUENCY)
        self.epoch_display_frequency = ( self.config.epoch_display_frequency or _CASCADE_CORRELATION_NETWORK_EPOCH_DISPLAY_FREQUENCY)
        self.generate_plots = ( self.config.generate_plots or _CASCADE_CORRELATION_NETWORK_GENERATE_PLOTS)
        self.status_display_frequency = ( self.config.status_display_frequency or _CASCADE_CORRELATION_NETWORK_STATUS_DISPLAY_FREQUENCY)
        self.candidate_display_frequency = ( self.config.candidate_display_frequency or _CASCADE_CORRELATION_NETWORK_DISPLAY_FREQUENCY)

        # Initialize display progress functions
        self._network_display_progress = display_progress( display_frequency=self.epoch_display_frequency)
        self._status_display_progress = display_progress( display_frequency=self.status_display_frequency)
        self._candidate_display_progress = display_progress( display_frequency=self.candidate_display_frequency)

        # Initialize plotter
        self.plotter = CascadeCorrelationPlotter(logger=self.logger)
        self.logger.debug( "CascadeCorrelationNetwork: _init_display_components: Display components initialized")

        # Add current dir to Python path for imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        self.logger.trace( "CascadeCorrelationNetwork: _initialize_randomness: Initializing randomness for the cascade correlation network")
        seed = seed or _CASCADE_CORRELATION_NETWORK_RANDOM_SEED
        self.logger.verbose( f"CascadeCorrelationNetwork: _initialize_randomness: Random seed set to: {seed}")
        sequence_max_value = ( sequence_max_value or _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE)
        self.logger.verbose( f"CascadeCorrelationNetwork: _initialize_randomness: Random sequence max value set to: {sequence_max_value}")
        random_max_value = ( random_max_value or _CASCADE_CORRELATION_NETWORK_RANDOM_MAX_VALUE)
        self.logger.verbose( f"CascadeCorrelationNetwork: _initialize_randomness: Random max value set to: {random_max_value}")
        self._seed_random_generator(
            seed=seed,
            max_value=sequence_max_value,
            seeder=np.random.seed,
            generator=np.random.randint,
        )
        self.logger.trace( "CascadeCorrelationNetwork: _initialize_randomness: Completed initialization of numpy random generator with seed and sequence for the cascade correlation network")
        self._seed_random_generator(
            seed=seed,
            max_value=sequence_max_value,
            seeder=random.seed,
            generator=random.randint,
        )
        self.logger.trace( "CascadeCorrelationNetwork: _initialize_randomness: Completed initialization of random random generator with seed and sequence for the cascade correlation network")
        self._seed_random_generator(
            seed=seed,
            max_value=sequence_max_value,
            seeder=torch.manual_seed,
            generator=lambda min, max: torch.randint(min, max, ()),
        )
        self.logger.trace( "CascadeCorrelationNetwork: _initialize_randomness: Completed initialization of torch random generator with seed and sequence for the cascade correlation network")
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
        seeder: callable = None,
        generator: callable = None,
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
        self.logger.trace( "CascadeCorrelationNetwork: _seed_random_generator: Seeding random module with seed and max value.")
        if seeder is None:
            self.logger.verbose( "CascadeCorrelationNetwork: _seed_random_generator: No seeder function provided, skipping seeding of random generator.")
            return
        seeder(seed)
        self.logger.trace( "CascadeCorrelationNetwork: _seed_random_generator: Random seed set for random module.")
        if generator is None:
            self.logger.verbose( "CascadeCorrelationNetwork: _seed_random_generator: No generator function provided, skipping random sequence generation and rolling.")
            return
        random_sequence = random.randint(0, max_value)  # trunk-ignore(bandit/B311)
        self.logger.verbose( f"CascadeCorrelationNetwork: _seed_random_generator: Random sequence number rolled to: {random_sequence}")

        # TODO:  Enable CUDA random generator seeding and rolling when needed
        #     self._seed_random_generator(seed=seed, max_value=sequence_max_value, seeder=torch.cuda.manual_seed, generator=lambda min, max: torch.rand(1, device='cuda'))
        # File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 392, in _roll_sequence_number
        #     discard = [generator(0, max_value) for _ in range(sequence)]
        # TypeError: only integer tensors of a single element can be converted to an index
        self._roll_sequence_number( sequence=random_sequence, max_value=max_value, generator=generator)
        self.logger.trace( "CascadeCorrelationNetwork: _seed_random_generator: Completed initialization of random generator with seed and sequence for the cascade correlation network")

    def _roll_sequence_number( self, sequence: int = None, max_value: int = None, generator: callable = None) -> None:
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
        self.logger.trace( "CascadeCorrelationNetwork: _roll_sequence_number: Rolling sequence number.")
        self.logger.debug( f"CascadeCorrelationNetwork: _roll_sequence_number: Rolling sequence number to: {sequence} with max value: {max_value} using generator: {generator}")
        if generator is not None:
            discard = [generator(0, max_value) for _ in range(sequence)]
            self.logger.verbose( f"CascadeCorrelationNetwork: _roll_sequence_number: Discarded {len(discard)} random numbers to roll to sequence number: {sequence}")
            self.logger.verbose( f"CascadeCorrelationNetwork: _roll_sequence_number: Random Generator rolled for sequence number: {sequence}")
        self.logger.trace( "CascadeCorrelationNetwork: _roll_sequence_number: Completed rolling of sequence number.")

    def _seed_hash(self, seed: int = None) -> None:
        """
        Description:
            Seed the hash function for the cascade correlation network.
        Args:
            seed: The seed value for the hash function
        """
        os.environ["PYTHONHASHSEED"] = str(seed)

    # Helper method to add hidden units to the network
    def _init_activation_with_derivative(
        self, activation_fn: callable = None
    ) -> callable:
        """
        Description:
            Wrap activation function to also provide its derivative.
        Args:
            activation_fn: Base activation function
        Note:
            This method wraps the activation function to also provide its derivative.
        Returns:
            Function that can compute both activation and its derivative
        """
        # Validate the activation function
        self.logger.trace( "CascadeCorrelationNetwork: _init_activation_with_derivative: Validating activation function")
        activation_fn = ( activation_fn, _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,)[activation_fn is None]
        self.logger.debug( f"CascadeCorrelationNetwork: _init_activation_with_derivative: Using activation function: {activation_fn}")

        # Wrapping the activation function with its derivative
        self.logger.trace( "CascadeCorrelationNetwork: _init_activation_with_derivative: Wrapping activation function to provide its derivative.")

        def wrapped_activation(x, derivative: bool = False):
            if derivative:
                if activation_fn in [ _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_TANH, _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_TANH, ]:  # For tanh, derivative is 1 - tanh^2(x)
                    return 1.0 - activation_fn(x) ** 2
                elif activation_fn in [ _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_SIGMOID, _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_SIGMOID, ]:  # For sigmoid, derivative is sigmoid(x) * (1 - sigmoid(x))
                    y = activation_fn(x)
                    return y * (1.0 - y)
                elif activation_fn in [ _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_RELU, _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_RELU, ]:  # For ReLU, derivative is 1 for x > 0, 0 otherwise
                    return (x > 0).float()
                else:  # Numerical approximation for other functions
                    eps = 1e-6
                    return (activation_fn(x + eps) - activation_fn(x - eps)) / (2 * eps)
            else:
                return activation_fn(x)
        self.logger.verbose( f"CascadeCorrelationNetwork: _init_activation_with_derivative: Returning wrapped activation function: {wrapped_activation}.")

        # Return the wrapped activation function
        self.logger.trace( "CascadeCorrelationNetwork: _init_activation_with_derivative: Completed wrapping of activation function.")
        return wrapped_activation

    #################################################################################################################################################################################################
    # Input validation methods
    #################################################################################################################################################################################################
    def _validate_tensor_input( self, x: torch.Tensor, param_name: str = "x", allow_none: bool = False) -> None:
        """
        Validate tensor input parameters.
        Args:
            x: Input tensor to validate
            param_name: Name of the parameter for error messages
            allow_none: Whether to allow None values
        Raises:
            ValidationError: If tensor is invalid
        """
        if allow_none and x is None:
            return
        if x is None:
            raise ValidationError(f"Parameter '{param_name}' cannot be None")
        if not isinstance(x, torch.Tensor):
            raise ValidationError( f"Parameter '{param_name}' must be a torch.Tensor, got {type(x)}")
        if x.numel() == 0:
            raise ValidationError(f"Parameter '{param_name}' cannot be an empty tensor")
        if torch.isnan(x).any():
            raise ValidationError(f"Parameter '{param_name}' contains NaN values")
        if torch.isinf(x).any():
            raise ValidationError(f"Parameter '{param_name}' contains infinite values")

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
            raise ValidationError( f"Input tensor must be 2D (batch_size, features), got shape {x.shape}")
        if ( expected_input_features is not None and x.shape[1] != expected_input_features):
            raise ValidationError( f"Expected {expected_input_features} input features, got {x.shape[1]}")
        if y is not None:
            if len(y.shape) != 2:
                raise ValidationError( f"Target tensor must be 2D (batch_size, classes), got shape {y.shape}")
            if x.shape[0] != y.shape[0]:
                raise ValidationError( f"Input and target batch sizes must match: {x.shape[0]} != {y.shape[0]}")

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
            raise ValidationError(f"Parameter '{param_name}' cannot be None")
        if not isinstance(value, (int, float)):
            raise ValidationError( f"Parameter '{param_name}' must be numeric, got {type(value)}")
        if min_val is not None and value < min_val:
            raise ValidationError( f"Parameter '{param_name}' must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValidationError( f"Parameter '{param_name}' must be <= {max_val}, got {value}")

    def _validate_positive_integer(
        self, value, param_name: str, allow_zero: bool = False
    ) -> None:
        """
        Validate positive integer parameters.
        Args:
            value: Value to validate
            param_name: Name of the parameter
            allow_zero: Whether to allow zero values
        Raises:
            ValidationError: If value is invalid
        """
        if not isinstance(value, int):
            raise ValidationError( f"Parameter '{param_name}' must be an integer, got {type(value)}")
        min_val = 0 if allow_zero else 1
        if value < min_val:
            raise ValidationError( f"Parameter '{param_name}' must be >= {min_val}, got {value}")

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
        self.logger.debug( f"CascadeCorrelationNetwork: _create_candidate_unit: Creating candidate unit {candidate_index}")
        return CandidateUnit(
            CandidateUnit__activation_function=kwargs.get( "activation_fn", self.activation_fn),
            CandidateUnit__input_size=input_size or self.input_size,
            CandidateUnit__output_size=kwargs.get("output_size", self.output_size),
            CandidateUnit__learning_rate=kwargs.get( "learning_rate", self.candidate_learning_rate),
            CandidateUnit__epochs=kwargs.get("epochs", self.candidate_epochs),
            CandidateUnit__candidate_index=candidate_index,
            CandidateUnit__uuid=candidate_uuid,
            CandidateUnit__random_seed=kwargs.get("random_seed", self.random_seed),
            CandidateUnit__random_value_scale=kwargs.get( "random_value_scale", self.random_value_scale),
            CandidateUnit__display_frequency=kwargs.get( "display_frequency", self.candidate_display_frequency),
            CandidateUnit__log_level_name=kwargs.get("log_level", "INFO"),
            CandidateUnit__sequence_max_value=kwargs.get( "sequence_max_value", self.sequence_max_value),
            CandidateUnit__random_max_value=kwargs.get( "random_value_max", self.random_max_value),
        )

    def _create_optimizer(self, parameters, config=None):
        """
        Create optimizer based on configuration.
        Args:
            parameters: Model parameters to optimize
            config: OptimizerConfig instance (uses self.optimizer_config if None)
        Returns:
            Configured optimizer instance
        """
        from cascade_correlation_config.cascade_correlation_config import ( OptimizerConfig,)
        config = config or getattr( self, "optimizer_config", OptimizerConfig(learning_rate=self.learning_rate))
        optimizer_map = {
            "Adam": lambda: optim.Adam(
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            ),
            "SGD": lambda: optim.SGD(
                parameters,
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            ),
            "RMSprop": lambda: optim.RMSprop(
                parameters,
                lr=config.learning_rate,
                momentum=config.momentum,
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            ),
            "AdamW": lambda: optim.AdamW(
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            ),
        }
        if config.optimizer_type not in optimizer_map:
            self.logger.warning( f"Unknown optimizer {config.optimizer_type}, defaulting to Adam")
            config.optimizer_type = "Adam"
        self.logger.debug( f"CascadeCorrelationNetwork: _create_optimizer: Creating {config.optimizer_type} optimizer with LR={config.learning_rate}")
        return optimizer_map[config.optimizer_type]()

    #################################################################################################################################################################################################
    # Public Method to train, grow, and evaluate the network
    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        max_epochs: int = None,
        early_stopping: bool = True,
    ) -> Dict[str, List]:
        """
        Train the network using the cascade correlation algorithm.
        Args:
            x_train: Training input tensor (batch_size, input_features)
            y_train: Training target tensor (batch_size, output_features)
            x_val: Validation input tensor (batch_size, input_features), optional
            y_val: Validation target tensor (batch_size, output_features), optional
            max_epochs: Maximum number of epochs to train (default: from config)
            early_stopping: Whether to use early stopping
        Raises:
            ValidationError: If input tensors are invalid or have wrong shapes
            TrainingError: If training fails due to configuration issues
        Returns:
            Training history dictionary containing losses and accuracies
        """
        # Validate training data
        self._validate_tensor_input(x_train, "x_train")
        self._validate_tensor_input(y_train, "y_train")
        self._validate_tensor_shapes( x_train, y_train, expected_input_features=self.input_size)

        # Validate that target tensor has correct output size
        if y_train.shape[1] != self.output_size:
            raise ValidationError( f"Target tensor must have {self.output_size} output features, got {y_train.shape[1]}")

        # Validate validation data if provided
        if x_val is not None:
            self._validate_tensor_input(x_val, "x_val")
            self._validate_tensor_shapes(x_val, expected_input_features=self.input_size)
        if y_val is not None:
            self._validate_tensor_input(y_val, "y_val")
            if x_val is None:
                raise ValidationError( "CascadeCorrelationNetwork: fit: Cannot provide y_val without x_val")
            self._validate_tensor_shapes( x_val, y_val, expected_input_features=self.input_size)
            if y_val.shape[1] != self.output_size:
                raise ValidationError( f"CascadeCorrelationNetwork: fit: Validation target tensor must have {self.output_size} output features, got {y_val.shape[1]}")

        # Validate max_epochs
        if max_epochs is not None:
            self._validate_positive_integer(max_epochs, "max_epochs")

        # Validate early_stopping
        if not isinstance(early_stopping, bool):
            raise ValidationError( f"CascadeCorrelationNetwork: fit: Parameter 'early_stopping' must be boolean, got {type(early_stopping)}")
        if len(x_train) == 0:
            raise ValidationError( "CascadeCorrelationNetwork: fit: Training dataset cannot be empty")

        # Initial training of the output layer
        self.logger.trace( "CascadeCorrelationNetwork: fit: Starting initial training of the output layer.")
        self.logger.info( "CascadeCorrelationNetwork: fit: Initial training of output layer")
        max_epochs = (max_epochs, self.output_epochs)[max_epochs is None]
        train_loss = self.train_output_layer(x_train, y_train, max_epochs)
        self.history["train_loss"].append(train_loss)
        if x_val is not None and y_val is not None:
            with torch.no_grad():
                value_output = self.forward(x_val)
                value_loss = nn.MSELoss()(value_output, y_val).item()
            self.history["value_loss"].append(value_loss)
            self.logger.info( f"CascadeCorrelationNetwork: fit: Initial - Train Loss: {train_loss:.6f}, Val Loss: {value_loss:.6f}")
        else:
            self.logger.info( f"CascadeCorrelationNetwork: fit: Initial - Train Loss: {train_loss:.6f}")

        # Calculate initial accuracy
        train_accuracy = self.calculate_accuracy(x_train, y_train)
        self.history["train_accuracy"].append(train_accuracy)
        if x_val is not None and y_val is not None:
            value_accuracy = self.calculate_accuracy(x_val, y_val)
            self.history["value_accuracy"].append(value_accuracy)
            self.logger.info( f"CascadeCorrelationNetwork: fit: Initial - Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {value_accuracy:.4f}")
        else:
            self.logger.info( f"CascadeCorrelationNetwork: fit: Initial - Train Accuracy: {train_accuracy:.4f}")

        # Main training loop
        patience_counter = 0
        best_value_loss = float("inf") if x_val is not None else None
        # TODO:  this code is repeated in the train candidates method--refactor it into a common method
        self.logger.info( f"CascadeCorrelationNetwork: fit: Starting main training loop with max epochs: {max_epochs}, early stopping: {early_stopping}")
        self.grow_network(
            candidate=CandidateUnit(
                _CandidateUnit__activation_function=self.activation_fn,
                _CandidateUnit__candidate_pool_size=self.candidate_pool_size,
                _CandidateUnit__display_frequency=self.display_frequency,
                _CandidateUnit__epochs_max=self.epochs_max,
                _CandidateUnit__input_size=self.input_size,
                _CandidateUnit__learning_rate=self.learning_rate,
                _CandidateUnit__log_file_name=self.log_file_name,
                _CandidateUnit__log_file_path=self.log_file_path,
                _CandidateUnit__log_level_name=self.log_level_name,
                _CandidateUnit__output_size=self.output_size,
                _CandidateUnit__random_value_scale=self.random_value_scale,
            ),
            x_train=x_train,
            y_train=y_train,
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            patience_counter=patience_counter,
            best_value_loss=best_value_loss,
            x_val=x_val,
            y_val=y_val,
        )
        self.history["hidden_units_added"].append( {"correlation": 0.0, "weights": [], "bias": []})
        self.logger.info("CascadeCorrelationNetwork: fit: Training completed.")
        self.logger.debug( f"CascadeCorrelationNetwork: fit: Final history:\n{self.history}")
        self.logger.trace( "CascadeCorrelationNetwork: fit: Completed training of the network.")
        return self.history

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
        # Validate input
        self._validate_tensor_input(x, "x")
        self._validate_tensor_shapes(x, expected_input_features=self.input_size)
        # Start with the input features
        self.logger.trace( "CascadeCorrelationNetwork: forward: Starting forward pass through the network.")
        self.logger.verbose( f"CascadeCorrelationNetwork: forward: Starting forward pass with input shape: {x.shape}")
        features = x
        self.logger.debug( f"CascadeCorrelationNetwork: forward: Input shape: {features.shape}")

        # Pass through each hidden unit
        hidden_outputs = []
        for i, unit in enumerate(self.hidden_units):

            # Concatenate all previous outputs with the input
            unit_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x

            # Get output from this unit
            unit_output = unit["activation_fn"]( torch.sum(unit_input * unit["weights"], dim=1) + unit["bias"]).unsqueeze(1)
            hidden_outputs.append(unit_output)
            if self._status_display_progress(i):
                self.logger.info( f"CascadeCorrelationNetwork: forward: Hidden unit {i + 1} output shape: {unit_output.shape}")
            self.logger.debug( f"CascadeCorrelationNetwork: forward: Hidden unit {i + 1} output shape: {unit_output.shape}")

        # Prepare input for the output layer
        output_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
        self.logger.verbose( f"CascadeCorrelationNetwork: forward: Output input shape: {output_input.shape}, Value: {output_input}")

        # Output layer (linear combination)
        output = torch.matmul(output_input, self.output_weights) + self.output_bias
        self.logger.debug( f"CascadeCorrelationNetwork: forward: Output shape: {output.shape}")
        self.logger.trace( "CascadeCorrelationNetwork: forward: Completed forward pass through the network.")
        return output

    #################################################################################################################################################################################################
    # Public Method to train the output layer of the network
    def train_output_layer(
        self,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        epochs: int = None,
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
        self.logger.trace( "CascadeCorrelationNetwork: train_output_layer: Starting training of the output layer.")
        self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Input shape: {x.shape if x is not None else 'None'}, Target shape: {y.shape if y is not None else 'None'}, Epochs: {epochs}")
        epochs = (epochs, _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS)[epochs is None]
        if x is None or y is None:
            raise ValueError( "CascadeCorrelationNetwork: train_output_layer: Input (x) and target (y) tensors must be provided for training the output layer.")

        # Define loss function and optimizer
        criterion = nn.MSELoss()

        # Create a simple linear layer for the output
        input_size = x.shape[1]
        self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Input size for output layer: {input_size}, Output size: {self.output_size}")
        if self.hidden_units:
            input_size += len(self.hidden_units)
        self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Adjusted input size for output layer with hidden units: {input_size}")

        # Create a temporary linear layer with the same weights as our current output layer
        output_layer = nn.Linear(input_size, self.output_size)
        with torch.no_grad():
            output_layer.weight.copy_( self.output_weights.t())  # Transpose because nn.Linear expects (out_features, in_features)
            self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Output weights shape: {self.output_weights.shape}, Transposed weights shape: {output_layer.weight.shape}")
            output_layer.bias.copy_(self.output_bias)
            self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Output bias shape: {self.output_bias.shape}, Bias: {output_layer.bias}")

        # Use this layer for optimization (store as instance variable for HDF5 serialization)
        # Create or recreate optimizer using factory method
        self.output_optimizer = self._create_optimizer(output_layer.parameters())
        self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Created optimizer: {type(self.output_optimizer).__name__}")
        optimizer = self.output_optimizer
        self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Learning Rate: {self.learning_rate}, Optimizer:\n{optimizer}")
        self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Output layer initialized with weights shape: {output_layer.weight.shape}, Bias shape: {output_layer.bias.shape}")

        # Output Layer Training loop
        for epoch in range(epochs):

            # Get the input for the output layer (original input + hidden unit outputs)
            hidden_outputs = []
            for unit in self.hidden_units:
                unit_input = ( torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x)
                unit_output = unit["activation_fn"]( torch.sum(unit_input * unit["weights"], dim=1) + unit["bias"]).unsqueeze(1)
                hidden_outputs.append(unit_output)

            # Calculate Loss by Concatenating inputs with outputs from existing hidden units
            output_input = ( torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x)
            output = output_layer(output_input)
            self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Output shape: {output.shape}, Output Input shape: {output_input.shape}")
            self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Output:\n{output}")
            self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Target shape: {y.shape}, Target:\n{y}")
            loss = criterion(output, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self._network_display_progress(epoch):
                self.logger.info( f"CascadeCorrelationNetwork: train_output_layer: Output Layer Training - Epoch {epoch + 1}, Loss: {loss.item():.6f}")
            self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Output Layer Training - Epoch {epoch + 1}, Loss: {loss.item():.6f}")

        # Update our model's weights with the trained values
        with torch.no_grad():
            self.output_weights = output_layer.weight.t().clone()  # Transpose back
            self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Output weights shape: {self.output_weights.shape}, Weights:\n{self.output_weights}")
            self.output_bias = output_layer.bias.clone()
            self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Output bias shape: {self.output_bias.shape}, Bias:\n{self.output_bias}")

        # Final loss
        with torch.no_grad():
            output = self.forward(x)
            self.logger.debug( f"CascadeCorrelationNetwork: train_output_layer: Final output shape: {output.shape}, Output: {output}")
            final_loss = criterion(output, y).item()
            self.logger.info( f"CascadeCorrelationNetwork: train_output_layer: Final output layer training loss: {final_loss:.6f}")
        if snapshot_path := self.create_snapshot() is not None:
            self.logger.info( f"CascadeCorrelationNetwork: train_output_layer: Created network snapshot at: {snapshot_path}")
            self.snapshot_counter += 1
        self.logger.trace( "CascadeCorrelationNetwork: train_output_layer: Completed training of the output layer.")
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
        self.logger.trace( "CascadeCorrelationNetwork: train_candidates: Starting training of candidate units.")
        start_time = datetime.datetime.now()
        self.logger.verbose( f"CascadeCorrelationNetwork: train_candidates: Start time: {start_time}")

        # Step 1: Prepare candidate input incorporating existing hidden units
        candidate_input = self._prepare_candidate_input(x)
        self.logger.debug( f"CascadeCorrelationNetwork: train_candidates: Prepared candidate input shape: {candidate_input.shape}")

        # Step 2: Generate candidate training data and tasks
        tasks = self._generate_candidate_tasks(candidate_input, y, residual_error)
        self.logger.debug( f"CascadeCorrelationNetwork: train_candidates: Generated {len(tasks)} candidate training tasks.")

        # Step 3: Determine optimal process count for training
        process_count = self._calculate_optimal_process_count()
        self.logger.debug( f"CascadeCorrelationNetwork: train_candidates: Optimal process count for training: {process_count}")

        # Step 4: Execute training (parallel or sequential)
        self.logger.trace( "CascadeCorrelationNetwork: train_candidates: Starting candidate training execution.")
        try:
            self.logger.info( f"CascadeCorrelationNetwork: train_candidates: Executing candidate training with {process_count} processes.")
            results = self._execute_candidate_training(tasks, process_count)
            self.logger.debug( f"CascadeCorrelationNetwork: train_candidates: Candidate training results: length: {len(results)}, value: {results}")
        except Exception as e:
            self.logger.error( f"CascadeCorrelationNetwork: train_candidates: Error during candidate training: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise TrainingError(f"Error during candidate training: {e}") from e
        self.logger.trace( f"CascadeCorrelationNetwork: train_candidates: Completed training of candidate units: Results: {results}.")

        # Step 5: Process and analyze results
        self.logger.trace( "CascadeCorrelationNetwork: train_candidates: Starting processing of candidate training results.")
        training_stats = self._process_training_results(results, tasks, start_time)
        self.logger.trace( f"CascadeCorrelationNetwork: train_candidates: Completed processing of candidate training results: {training_stats}.")
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
        hidden_outputs = []
        for unit in self.hidden_units:
            unit_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
            unit_output = unit["activation_fn"](
                torch.sum(unit_input * unit["weights"], dim=1) + unit["bias"]
            )
            hidden_outputs.append(unit_output.unsqueeze(1))
            self.logger.debug( f"CascadeCorrelationNetwork: _prepare_candidate_input: Hidden unit output shape: {unit_output.shape}")
        candidate_input = ( torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x)
        self.logger.debug( f"CascadeCorrelationNetwork: _prepare_candidate_input: Candidate input shape: {candidate_input.shape}")
        self.logger.info( f"CascadeCorrelationNetwork: _prepare_candidate_input: Hidden units: {len(hidden_outputs)}")
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
        training_inputs = (
            candidate_input,
            self.candidate_epochs,
            y,
            residual_error,
            self.candidate_learning_rate,
            self.candidate_display_frequency,
        )

        # Generate candidate metadata
        candidate_uuids = [str(uuid.uuid4()) for _ in range(self.candidate_pool_size)]
        candidate_seeds = [ random.randint(0, self.random_max_value) for _ in range(self.candidate_pool_size) ]  # trunk-ignore(bandit/B311)
        candidate_data = [ ( i, input_size, self.activation_function_name, self.random_value_scale, candidate_uuids[i], candidate_seeds[i], self.random_max_value, self.sequence_max_value,) for i in range(self.candidate_pool_size) ]

        tasks = [ (i, candidate_data[i], training_inputs) for i in range(self.candidate_pool_size) ]
        self.logger.debug( f"CascadeCorrelationNetwork: _generate_candidate_tasks: Created {len(tasks)} training tasks")
        return tasks

    def _calculate_optimal_process_count(self) -> int:
        """
        Calculate the optimal number of processes for candidate training.
        Returns:
            Optimal process count
        """
        self.logger.debug( f"CascadeCorrelationNetwork: _calculate_optimal_process_count: CPU count: {os.cpu_count()}")
        self.logger.debug( f"CascadeCorrelationNetwork: _calculate_optimal_process_count: Candidate pool size: {self.candidate_pool_size}")

        # Get available CPU cores considering affinity if available
        if hasattr(os, "sched_getaffinity"):
            affinity_cores = len(os.sched_getaffinity(0))
            self.logger.debug( f"CascadeCorrelationNetwork: _calculate_optimal_process_count: Affinity CPU count: {affinity_cores}")
        else:
            affinity_cores = os.cpu_count()

        # Calculate available cores considering various constraints
        cpu_cores_available = min(
            self.candidate_pool_size,
            affinity_cores,
            ( self.candidate_training_context.cpu_count() if hasattr(self, "candidate_training_context") else os.cpu_count()),
            os.cpu_count(),
        )

        # Leave one core free to keep the system responsive
        process_count = max(1, cpu_cores_available - 1)
        self.logger.debug( f"CascadeCorrelationNetwork: _calculate_optimal_process_count: Using {process_count} processes")
        return process_count

    def _execute_candidate_training(self, tasks: list, process_count: int) -> list:
        """
        Execute candidate training using multiprocessing or sequential processing.
        Args:
            tasks: List of training tasks
            process_count: Number of processes to use
        Returns:
            List of training results
        """
        self.logger.info( f"CascadeCorrelationNetwork: _execute_candidate_training: Training {len(tasks)} candidates with {process_count} processes")
        results = []
        self.logger.debug( f"CascadeCorrelationNetwork: _execute_candidate_training: Adjusted process count to: {process_count}")
        try:
            if process_count > 1:
                self.logger.debug( f"CascadeCorrelationNetwork: _execute_candidate_training: Using {process_count} processes")
                results = self._execute_parallel_training(tasks, process_count)

                # Validate results were actually obtained
                if not results:
                    self.logger.warning( "CascadeCorrelationNetwork: _execute_candidate_training: Parallel processing returned no results, falling back to sequential")
                    raise RuntimeError( "CascadeCorrelationNetwork: _execute_candidate_training: Parallel processing failed to return results")
                self.logger.debug( "CascadeCorrelationNetwork: _execute_candidate_training: Completed parallel processing")
            else:
                self.logger.debug( "CascadeCorrelationNetwork: _execute_candidate_training: Using sequential processing")
                results = self._execute_sequential_training(tasks)
                self.logger.debug( "CascadeCorrelationNetwork: _execute_candidate_training: Completed sequential processing")
        except Exception as e:
            self.logger.error( f"CascadeCorrelationNetwork: _execute_candidate_training: Error in candidate node training: {e}")
            import traceback
            self.logger.error( f"CascadeCorrelationNetwork: _execute_candidate_training: Traceback: {traceback.format_exc()}")

            # Create dummy failure results for each task
            self.logger.warning( "CascadeCorrelationNetwork: _execute_candidate_training: Creating dummy results for failed training")
            results = self._get_dummy_results(len(tasks))
        self.logger.debug( f"CascadeCorrelationNetwork: _execute_candidate_training: Obtained {len(results)} results")

        # Ensure we have some results:  For empty results list, create an intelligently empty dummy results
        if not results:
            self.logger.error( "CascadeCorrelationNetwork: _execute_candidate_training: No results obtained from either parallel or sequential processing")
            results = self._get_dummy_results(len(tasks))
        return results

    def _execute_parallel_training(
        self,
        tasks: list,
        process_count: int = -1,
        sleepytime: float = _CASCADE_CORRELATION_NETWORK_WORKER_STANDBY_SLEEPYTIME,
    ) -> list:
        """Execute training using multiprocessing."""
        self.logger.debug( "CascadeCorrelationNetwork: _execute_parallel_training: Using multiprocessing")

        # Adjust process count if invalid
        process_count = (process_count, self._calculate_optimal_process_count())[ process_count < 1 ]

        # Start the manager server
        # self.logger.trace("CascadeCorrelationNetwork: _execute_parallel_training: Starting the manager server")
        self.logger.debug( "CascadeCorrelationNetwork: _execute_parallel_training: Starting the manager server")
        self._start_manager()
        # self.logger.trace("CascadeCorrelationNetwork: _execute_parallel_training: Manager server started")
        self.logger.debug( "CascadeCorrelationNetwork: _execute_parallel_training: Manager server started")
        task_queue = self._task_queue
        result_queue = self._result_queue
        results = []
        # self.logger.trace("CascadeCorrelationNetwork: _execute_parallel_training: Created task and result queues")
        self.logger.debug( "CascadeCorrelationNetwork: _execute_parallel_training: Created task and result queues")
        try:
            # Add tasks to the queue
            # self.logger.trace("CascadeCorrelationNetwork: _execute_parallel_training: Adding tasks to the queue")
            self.logger.debug( "CascadeCorrelationNetwork: _execute_parallel_training: Adding tasks to the queue")
            for task in tasks:
                task_queue.put(task)
            self.logger.debug( f"CascadeCorrelationNetwork: _execute_parallel_training: Added {len(tasks)} tasks to queue")

            # Start worker processes
            num_workers = max(1, min(process_count, len(tasks)))
            workers = []
            self.logger.debug( f"CascadeCorrelationNetwork: _execute_parallel_training: Starting {num_workers} workers")

            for i in range(num_workers):
                # self.logger.trace(f"CascadeCorrelationNetwork: _execute_parallel_training: Starting worker {i}")
                self.logger.debug( f"CascadeCorrelationNetwork: _execute_parallel_training: Starting worker {i}")
                worker = self._mp_ctx.Process(
                    target=CascadeCorrelationNetwork._worker_loop,
                    args=(task_queue, result_queue, True),
                    daemon=True,
                    name=f"CandidateWorker-{i}",
                )
                # self.logger.trace(f"CascadeCorrelationNetwork: _execute_parallel_training: Worker {i} instantiated.")
                self.logger.debug( f"CascadeCorrelationNetwork: _execute_parallel_training: Worker {i} instantiated.")
                # self.logger.trace("CascadeCorrelationNetwork: _execute_parallel_training: Starting worker process.")
                self.logger.debug( "CascadeCorrelationNetwork: _execute_parallel_training: Starting worker process.")
                worker.start()
                # self.logger.trace(f"CascadeCorrelationNetwork: _execute_parallel_training: Worker {i} started with PID {worker.pid}.")
                self.logger.debug( f"CascadeCorrelationNetwork: _execute_parallel_training: Worker {i} started with PID {worker.pid}.")
                # self.logger.trace("CascadeCorrelationNetwork: _execute_parallel_training: Adding worker to the Workers list.")
                self.logger.debug( "CascadeCorrelationNetwork: _execute_parallel_training: Adding worker to the Workers list.")
                workers.append(worker)
                # self.logger.trace(f"CascadeCorrelationNetwork: _execute_parallel_training: Completed adding worker to the Workers list: workers: length: {len(workers)}, value: {workers}.")
                self.logger.debug( f"CascadeCorrelationNetwork: _execute_parallel_training: Completed adding worker to the Workers list: workers: length: {len(workers)}, value: {workers}.")
            self.logger.debug( f"CascadeCorrelationNetwork: _execute_parallel_training: Successfully Started Workers: {len(workers)}.")
            self.logger.debug( f"CascadeCorrelationNetwork: _execute_parallel_training: Initial Queue Sizes: Task Queue: {task_queue.qsize()}, Result Queue: {result_queue.qsize()}")

            # Joining worker processes when training is complete
            self.logger.debug( "CascadeCorrelationNetwork: _execute_parallel_training: Waiting for workers to complete all tasks: {len(tasks)}.")
            while not task_queue.empty() or result_queue.qsize() < len(tasks):
                time.sleep(sleepytime)
            self.logger.debug( f"CascadeCorrelationNetwork: _execute_parallel_training: Completed Wait for workers to complete all tasks: Task Queue: {task_queue.qsize()}, Result Queue: {result_queue.qsize()}.")
            self.logger.debug( f"CascadeCorrelationNetwork: _execute_parallel_training: Successfully Completed Joining {len(workers)} workers")
            self.logger.debug( f"CascadeCorrelationNetwork: _execute_parallel_training: Final Queue Sizes: Task Queue: {task_queue.qsize()}, Result Queue: {result_queue.qsize()}")

            # Collect results, NOTE: results is of type list of data class: [candidate_training_result, ...]
            self.logger.debug( "CascadeCorrelationNetwork: _execute_parallel_training: Collecting results from workers")
            results = self._collect_training_results(result_queue, len(tasks))
            self.logger.debug( f"CascadeCorrelationNetwork: _execute_parallel_training: Collected {len(results)} results")

            # Stop workers
            self._stop_workers(workers, task_queue)
            self.logger.debug( "CascadeCorrelationNetwork: _execute_parallel_training: Stopped all workers")
        finally:
            self.logger.trace( "CascadeCorrelationNetwork: _execute_parallel_training: Stopping manager server")
            self._stop_manager()
        return results

    def _execute_sequential_training(self, tasks: list) -> list:
        """Execute training sequentially."""
        self.logger.debug( "CascadeCorrelationNetwork: _execute_sequential_training: Using sequential processing")
        results = []
        for candidate_index, task in enumerate(tasks):
            self.logger.verbose( f"CascadeCorrelationNetwork: _execute_sequential_training: Training candidate {candidate_index + 1}/{len(tasks)}")
            try:
                candidate_training_result = self.train_candidate_worker( task_data_input=task, parallel=False)
                results.append(candidate_training_result)
            except Exception as task_e:
                self.logger.error( f"CascadeCorrelationNetwork: _execute_sequential_training: Task error: {task_e}")
                results.append( (task[0], task[1][4] if len(task[1]) > 4 else None, 0.0, None))
        return results

    def _collect_training_results(
        self,
        result_queue: Queue,
        num_tasks: int,
        # TODO: Make these into proper constants
        queue_timeout: float = 60.0,
        request_timeout: float = 1.0,
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
        self.logger.debug( f"CascadeCorrelationNetwork: _collect_training_results: Collecting {num_tasks} results")
        self.logger.debug( f"CascadeCorrelationNetwork: _collect_training_results: Timeout set to {queue_timeout} seconds")
        self.logger.debug( f"CascadeCorrelationNetwork: _collect_training_results: Result Queue: Length: {result_queue.qsize()}, Contents: {list(result_queue.queue) if hasattr(result_queue, 'queue') else 'N/A'}")
        deadline = time.time() + queue_timeout
        while collected_results < num_tasks and time.time() < deadline:
            try:
                result = result_queue.get(timeout=request_timeout)
                self.logger.debug( f"CascadeCorrelationNetwork: _collect_training_results: Retrieved result: {result}")
                results.append(result)
                collected_results += 1
                self.logger.verbose( f"CascadeCorrelationNetwork: _collect_training_results: Collected {collected_results}/{num_tasks}")
            except Empty as empty_e:
                self.logger.warning( f"CascadeCorrelationNetwork: _collect_training_results: Result queue empty, continuing: {empty_e}")
                continue
            except Exception as e:
                self.logger.error( f"CascadeCorrelationNetwork: _collect_training_results: Error collecting result: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                break
        self.logger.debug( f"CascadeCorrelationNetwork: _collect_training_results: Collected {collected_results} results")
        return results

    def _stop_workers(self, workers: list, task_queue) -> None:
        """Stop worker processes with improved termination handling."""
        import signal
        if not workers:
            self.logger.debug("CascadeCorrelationNetwork: _stop_workers: No workers to stop")
            return
        self.logger.info( f"CascadeCorrelationNetwork: _stop_workers: Stopping {len(workers)} worker processes")

        # Phase 1: Send sentinel values
        for i in range(len(workers)):
            try:
                task_queue.put(None, timeout=5)
                self.logger.debug( f"CascadeCorrelationNetwork: _stop_workers: Sent sentinel to worker {i}")
            except Exception as e:
                self.logger.error( f"CascadeCorrelationNetwork: _stop_workers: Failed to send sentinel to worker {i}: {e}")

        # Phase 2: Wait gracefully with increased timeout
        terminated_count = 0
        for worker in workers:
            worker.join(timeout=15)  # Increased from 10
            if not worker.is_alive():
                terminated_count += 1
                self.logger.debug( f"CascadeCorrelationNetwork: _stop_workers: Worker {worker.name} stopped gracefully")
            else:
                self.logger.warning( f"CascadeCorrelationNetwork: _stop_workers: Worker {worker.name} (PID {worker.pid}) did not stop gracefully")

        # Phase 3: Terminate remaining workers
        for worker in workers:
            if worker.is_alive():
                self.logger.warning( f"CascadeCorrelationNetwork: _stop_workers: Terminating worker {worker.name}")
                worker.terminate()
                worker.join(timeout=2)

                # Phase 4: Force kill if still alive
                if worker.is_alive():
                    self.logger.error( f"CascadeCorrelationNetwork: _stop_workers: Worker {worker.name} still alive, sending SIGKILL")
                    try:
                        os.kill(worker.pid, signal.SIGKILL)
                        worker.join(timeout=1)
                    except Exception as e:
                        self.logger.error( f"CascadeCorrelationNetwork: _stop_workers: Failed to SIGKILL worker: {e}")

        if alive_workers := [w for w in workers if w.is_alive()]:
            self.logger.error( f"CascadeCorrelationNetwork: _stop_workers: ⚠️  {len(alive_workers)} workers still alive after cleanup!")
        else:
            self.logger.info( f"CascadeCorrelationNetwork: _stop_workers: ✓ All {len(workers)} workers stopped successfully ({terminated_count} gracefully)")

    def _process_training_results( self, results: list, tasks: list, start_time) -> TrainingResults:
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
        self.logger.info( f"CascadeCorrelationNetwork: _process_training_results: Training duration: {end_time - start_time}")

        # Process results
        if not results:
            self.logger.warning( "CascadeCorrelationNetwork: _process_training_results: No results obtained")
            self.logger.warning( f"CascadeCorrelationNetwork: _process_training_results: Unable to Process empty results list.  Building dummy results: {results}")
            results = self._get_dummy_results(len(tasks))
        elif len(results) != len(tasks):
            self.logger.warning( f"CascadeCorrelationNetwork: _process_training_results: Mismatch in results count: expected {len(tasks)}, got {len(results)}")
        self.logger.trace( "CascadeCorrelationNetwork: _process_training_results: Completed pre-processing of training results.")

        # # Sort and extract candidate data
        # NOTE: results is a list of CandidateTrainingResult objects
        results.sort( key=lambda r: (r.correlation is not None, np.abs(r.correlation)), reverse=True,)
        self.logger.debug( f"CascadeCorrelationNetwork: _process_training_results: Sorted {len(results)} results")

        # Extract candidates data from results: list of CandidateTrainingResult objects
        valid_candidates = [ r.candidate_id is not None and r.candidate_uuid is not None and r.correlation is not None and r.candidate is not None for r in results ]

        # Identify best candidate
        best_candidate_id = ( self.get_single_candidate_data(results, 0, "candidate_id", -1),)

        # Compile statistics
        successful_candidates = self.get_candidates_data_count( results, "correlation", lambda c: c >= self.correlation_threshold)
        success_count = self.get_candidates_data_count(results, "success", lambda s: s)
        if success_count != successful_candidates:
            self.logger.warning( f"CascadeCorrelationNetwork: _process_training_results: Mismatch in success counts: success_count: {success_count}, successful_candidates: {successful_candidates}")

        # Building TrainingResults object
        training_results = TrainingResults(
            epochs_completed=self.get_candidates_data(results, "epochs_completed"),
            candidate_ids=self.get_candidates_data(results, "candidate_id"),
            candidate_uuids=self.get_candidates_data(results, "candidate_uuid"),
            correlations=self.get_candidates_data(results, "correlation"),
            candidate_objects=self.get_candidates_data(results, "candidate"),
            best_candidate_id=best_candidate_id,
            best_candidate_uuid=self.get_single_candidate_data( results, best_candidate_id, "candidate_uuid", None),
            best_correlation=self.get_single_candidate_data( results, best_candidate_id, "correlation", 0.0),
            best_candidate=self.get_single_candidate_data( results, best_candidate_id, "candidate", None),
            success_count=success_count,
            successful_candidates=successful_candidates,
            failed_count=len(results) - successful_candidates,
            error_messages=self.get_candidates_error_messages( results, valid_candidates),
            max_correlation=self.get_single_candidate_data( results, 0, "correlation", 0.0),
            start_time=start_time,
            end_time=end_time,
        )
        self.logger.debug( f"CascadeCorrelationNetwork: _process_training_results: Processed results: {training_results}")
        self.logger.trace( "CascadeCorrelationNetwork: _process_training_results: Completed processing of training results.")
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
            ) for id in range(num_results)
        ]

    def get_candidates_data(self, results: list, field: str) -> list:
        """
        Get candidate data from results.
        Returns:
            List of candidate data for the specified field
        """
        return [getattr(r, field) for r in results if getattr(r, field) is not None]

    def get_single_candidate_data(
        self, results: list, candidate_id: int, field: str, default: Any
    ) -> Any:
        """
        Get single candidate data field from results using getattr for dataclass objects.
        Returns:
            Field value from specified result or default
        """
        self.logger.debug( f"CascadeCorrelationNetwork: get_single_candidate_data: Retrieving field '{field}' for candidate ID {candidate_id}")
        self.logger.debug( f"CascadeCorrelationNetwork: get_single_candidate_data: Results type: {type(results)}, length: {len(results)}, Results: {results}")
        self.logger.debug( f"CascadeCorrelationNetwork: get_single_candidate_data: Field: {field}, Default: {default}")
        self.logger.debug( f"CascadeCorrelationNetwork: get_single_candidate_data: ID: type: {type(candidate_id)}, value: {candidate_id}")

        # TODO: need to check types and handle looping through tuple
        #  B=A[0] if isinstance(A, tuple) else A if isinstance(A, int) else None
        # if B is not None and 0 <= B and B <= len(A):
        #    print(f"B is: Type: {type(B)}, Value: {B}, A is: Type {type(A)}, Value: {A}")

        candidate_id = ( candidate_id[0] if isinstance(candidate_id, tuple) else candidate_id if isinstance(candidate_id, int) else None)
        self.logger.debug( f"CascadeCorrelationNetwork: get_single_candidate_data: Processed Candidate ID: type: {type(candidate_id)}, value: {candidate_id}")
        if candidate_id is not None and 0 <= candidate_id < len(results):
            value = getattr(results[candidate_id], field, None)
            self.logger.debug( f"CascadeCorrelationNetwork: get_single_candidate_data: Retrieved value: {value}")
            return value if value is not None else default
        self.logger.debug( f"CascadeCorrelationNetwork: get_single_candidate_data: ID {candidate_id} is out of bounds, returning default: {default}")
        return default

    def get_candidates_data_count(
        self, results: list, field: str, constraint: callable
    ) -> int:
        """
        Get count of candidate data from results.
        Args:
            results: Raw training results
            field: Field to count
        Returns:
            Count of candidate data for the specified field
        """
        return sum( getattr(r, field) for r in results if getattr(r, field) is not None and constraint(getattr(r, field)))

    def get_candidates_error_messages( self, results: list, valid_candidates: list) -> dict:
        """
        Get error messages for candidates.
        Returns:
            Dictionary of candidate error messages
        """
        return {
            key: (
                f'Candidate ID {r.candidate_id} (UUID: {r.candidate_uuid}): "{r.error_message}"' if r.error_message and valid_candidates[i] else (
                    f"Candidate ID {r.candidate_id} (UUID: {r.candidate_uuid}): No error message provided" if valid_candidates[i] else (
                        f"Candidate ID {r.candidate_id} (UUID: {r.candidate_uuid}): Invalid candidate data"
                    )
                )
            ) for i, r in enumerate(results) if r.candidate_id is not None or r.candidate_uuid is not None for key in [r.candidate_id, r.candidate_uuid] if key is not None
        }

    def __getstate__(self):
        """Remove non-picklable items for multiprocessing."""
        state = self.__dict__.copy()
        # Remove logger and display functions (not picklable)
        state.pop('logger', None)
        state.pop('plotter', None)
        state.pop('_network_display_progress', None)
        state.pop('_status_display_progress', None)
        state.pop('_candidate_display_progress', None)
        # Remove log_config (contains loggers that cannot be pickled)
        state.pop('log_config', None)
        # Remove activation functions (local closures cannot be pickled)
        state.pop('activation_fn', None)
        state.pop('activation_fn_no_diff', None)
        # Remove locks and other non-picklable objects
        state.pop('_thread.lock', None)
        # Remove large training data (should not be in snapshot anyway)
        state.pop('_training_data', None)
        state.pop('_validation_data', None)
        # Remove multiprocessing objects (cannot be pickled)
        state.pop('_manager', None)
        state.pop('_task_queue', None)
        state.pop('_result_queue', None)
        state.pop('_mp_ctx', None)
        state.pop('candidate_training_context', None)
        return state

    def __setstate__(self, state):
        """Restore state and reinitialize non-picklable objects."""
        self.__dict__.update(state)
        # Reinitialize logger
        from log_config.logger.logger import Logger
        Logger.set_level(self.log_level_name if hasattr(self, 'log_level_name') else 'INFO')
        self.logger = Logger
        # Set log_config to None - it was removed during pickling
        self.log_config = None
        # Reinitialize activation function
        self._init_activation_function()
        # Reinitialize plotter if needed
        if not hasattr(self, 'plotter'):
            from cascor_plotter.cascor_plotter import CascadeCorrelationPlotter
            self.plotter = CascadeCorrelationPlotter(logger=self.logger)
        # Reinitialize display progress functions
        from utils.utils import display_progress
        if not hasattr(self, '_network_display_progress'):
            self._network_display_progress = display_progress( display_frequency=getattr(self, 'epoch_display_frequency', 10))
        if not hasattr(self, '_status_display_progress'):
            self._status_display_progress = display_progress( display_frequency=getattr(self, 'status_display_frequency', 100))
        if not hasattr(self, '_candidate_display_progress'):
            self._candidate_display_progress = display_progress( display_frequency=getattr(self, 'candidate_display_frequency', 10))
        # Set default values for removed data
        if not hasattr(self, '_training_data'):
            self._training_data = None
        if not hasattr(self, '_validation_data'):
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
        from cascade_correlation_config.cascade_correlation_config import OptimizerConfig
        config = optimizer_config or getattr(self.config, 'optimizer_config', OptimizerConfig())
        optimizer_map = {
            'Adam': lambda: optim.Adam(
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
                amsgrad=getattr(config, 'amsgrad', False)
            ),
            'SGD': lambda: optim.SGD(
                parameters,
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            ),
            'RMSprop': lambda: optim.RMSprop(
                parameters,
                lr=config.learning_rate,
                momentum=config.momentum,
                eps=config.epsilon,
                weight_decay=config.weight_decay
            ),
            'AdamW': lambda: optim.AdamW(
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
                amsgrad=getattr(config, 'amsgrad', False)
            ),
        }

        if config.optimizer_type not in optimizer_map:
            self.logger.warning( f"CascadeCorrelationNetwork: _create_optimizer: Unknown optimizer type '{config.optimizer_type}', defaulting to Adam")
            config.optimizer_type = 'Adam'

        optimizer = optimizer_map[config.optimizer_type]()
        self.logger.debug( f"CascadeCorrelationNetwork: _create_optimizer: Created {config.optimizer_type} optimizer with lr={config.learning_rate}")
        return optimizer

    @staticmethod
    def train_candidate_worker(
        task_data_input: tuple = None, parallel: bool = True
    ) -> None:
        logger = Logger
        logger.info( "CascadeCorrelationNetwork: train_candidate_worker: Starting training of Candidate Units in Pool.")
        try:  # Get task data for process worker
            (worker_id, worker_uuid) = ( (mp.current_process().pid, str(uuid.uuid4())) if parallel else (0, "None"))
            logger.debug( f"CascadeCorrelationNetwork: train_candidate_worker: Retrieved worker ID and UUID: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        except Exception as e:
            logger.error( f"CascadeCorrelationNetwork: train_candidate_worker: Error retrieving worker ID and UUID: {e}")
            (worker_id, worker_uuid) = (0, "None")
        try:
            if task_data_input is None:
                logger.error( "CascadeCorrelationNetwork: train_candidate_worker: No task data input provided.")
                return (None, None, 0.0, None)
            candidate_inputs = CascadeCorrelationNetwork._build_candidate_inputs(
                task_data_input=task_data_input,
                worker_id=worker_id,
                worker_uuid=worker_uuid,
            )
            if ( candidate_inputs is None or not isinstance(candidate_inputs, dict) or len(candidate_inputs) == 0):
                logger.error( f"CascadeCorrelationNetwork: train_candidate_worker: No candidate inputs built: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
                return (None, None, 0.0, None)
            logger.debug( f"CascadeCorrelationNetwork: train_candidate_worker: Built candidate inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Inputs: {str(candidate_inputs)}")

            # Instantiate a CandidateUnit using factory method (Note: needs network instance for factory)
            logger.debug( f"CascadeCorrelationNetwork: train_candidate_worker: Instantiate a CandidateUnit using factory method (Note: needs network instance for factory, candidate_inputs: {candidate_inputs}): Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
            logger.debug( f"CascadeCorrelationNetwork: train_candidate_worker: Candidate Inputs Key Values: {candidate_inputs.get('candidate_display_frequency')}, Candidate Index: {candidate_inputs.get('candidate_index')}, Candidate UUID: {candidate_inputs.get('candidate_uuid')}")
            try:
                logger.debug( f"CascadeCorrelationNetwork: train_candidate_worker: Instantiating CandidateUnit Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Inputs: {str(candidate_inputs)}")
                candidate = CandidateUnit(
                    CandidateUnit__activation_function=candidate_inputs.get( "activation_fn"),
                    CandidateUnit__display_frequency=candidate_inputs.get( "candidate_display_frequency"),
                    CandidateUnit__epochs=candidate_inputs.get("epochs"),
                    CandidateUnit__input_size=candidate_inputs.get("input_size"),
                    CandidateUnit__learning_rate=candidate_inputs.get("learning_rate"),
                    CandidateUnit__log_level_name="INFO",
                    CandidateUnit__sequence_max_value=candidate_inputs.get( "sequence_max_value"),
                    CandidateUnit__random_seed=candidate_inputs.get("random_seed"),
                    CandidateUnit__random_max_value=candidate_inputs.get( "random_value_max"),
                    CandidateUnit__random_value_scale=candidate_inputs.get( "random_value_scale"),
                    CandidateUnit__uuid=candidate_inputs.get("candidate_uuid"),
                    CandidateUnit__candidate_index=candidate_inputs.get( "candidate_index"),
                )
                logger.debug( f"CascadeCorrelationNetwork: train_candidate_worker: Completed Instantiating CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate.get_uuid()}")
            except Exception as e:
                logger.error( f"CascadeCorrelationNetwork: train_candidate_worker: Caught Exception while instantiating CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_inputs.get('candidate_index')}, Candidate UUID: {candidate_inputs.get('candidate_uuid')}, Error during candidate instantiation:\nException:\n{e}")
                import traceback
                traceback.print_exc()
                raise
            logger.verbose( f"CascadeCorrelationNetwork: train_candidate_worker: Created CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_inputs.get('candidate_index')}, Candidate UUID: {candidate.get_uuid()}, Candidate Object: {candidate}")

            # Train the candidate unit
            result = CascadeCorrelationNetwork._train_candidate_unit(
                candidate=candidate,
                candidate_uuid=candidate_inputs.get("candidate_uuid"),
                candidate_index=candidate_inputs.get("candidate_index"),
                candidate_input=candidate_inputs.get("candidate_input"),
                candidate_epochs=candidate_inputs.get("candidate_epochs"),
                residual_error=candidate_inputs.get("residual_error"),
                candidate_learning_rate=candidate_inputs.get("candidate_learning_rate"),
                candidate_display_frequency=candidate_inputs.get( "candidate_display_frequency"),
                worker_id=worker_id,
                worker_uuid=worker_uuid,
            )
            logger.info( f"CascadeCorrelationNetwork: train_candidate_worker: Returning from Candidate Unit Training: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate ID: {result.candidate_id}, Candidate UUID: {result.candidate_uuid}, Candidate Correlation: {float(result.correlation):.6f}")
            return result

        except Exception as e:
            import traceback
            logger.error( f"CascadeCorrelationNetwork: train_candidate_worker: Caught Exception while training CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Error during candidate training:\nException:\n{e}")
            logger.error( f"CascadeCorrelationNetwork: train_candidate_worker: Error during Candidate Training: Worker ID: {worker_id}, Worker UUID: {worker_uuid}\nTraceback:\n{traceback.format_exc()}")
            candidate_index = ( candidate_inputs.get("candidate_index") if candidate_inputs else -1)
            candidate_uuid = ( candidate_inputs.get("candidate_uuid") if candidate_inputs else None)
            return CandidateTrainingResult(
                candidate_id=candidate_index,
                candidate_uuid=candidate_uuid,
                correlation=0.0,
                candidate=None,
                success=False,
                epochs_completed=0,
                error_message=str(e),
            )

    @staticmethod
    def _build_candidate_inputs(
        task_data_input: tuple = None,
        worker_uuid: str = None,
        worker_id: int = None,
    ):
        logger = Logger
        logger.debug( f"CascadeCorrelationNetwork: _build_candidate_inputs: Building candidate inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")

        # Unpack task data
        # TODO: consider using data classes for task data, candidate data, and training inputs
        logger.debug( f"CascadeCorrelationNetwork: _build_candidate_inputs: Attempting to Unpack Task data, Candidate data, and Training inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        logger.verbose( f"CascadeCorrelationNetwork: _build_candidate_inputs: Task data: length: {len(task_data_input)}, Type: {type(task_data_input)}, Content:\n{task_data_input}")
        logger.debug( f"CascadeCorrelationNetwork: _build_candidate_inputs: Task data unpacked: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        (candidate_index, candidate_data, training_inputs) = task_data_input  # Unpack training task data
        logger.debug( f"CascadeCorrelationNetwork: _build_candidate_inputs: Successfully Unpacked Task data: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        logger.verbose( f"CascadeCorrelationNetwork: _build_candidate_inputs: Candidate Index: {candidate_index}, Type: {type(candidate_index)}, Value: {candidate_index}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        logger.verbose( f"CascadeCorrelationNetwork: _build_candidate_inputs: Candidate Inputs: Length: {len(training_inputs)}, Type: {type(training_inputs)}, Content:\n{training_inputs}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        logger.verbose( f"CascadeCorrelationNetwork: _build_candidate_inputs: Candidate Data: length: {len(candidate_data)}, Type: {type(candidate_data)}, Content:\n{candidate_data}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}")
        ( input_size, activation_name, random_value_scale, candidate_uuid, candidate_seed, random_max_value, sequence_max_value,) = candidate_data[1:]
        logger.debug( f"CascadeCorrelationNetwork: _build_candidate_inputs: Successfully Unpacked Candidate Data: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}.")
        logger.verbose( f"CascadeCorrelationNetwork: _build_candidate_inputs: Candidate data unpacked: Candidate ID: {id}, Input Size: {input_size}, Activation Function Name: {activation_name}, Random Value Scale: {random_value_scale}, Candidate UUID: {candidate_uuid}, Random Seed: {candidate_seed}, Random Value Max: {random_max_value}, Sequence Max Value: {sequence_max_value}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}.")
        logger.debug( f"CascadeCorrelationNetwork: _build_candidate_inputs: Attempting to unpack Training inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}")
        logger.verbose( f"CascadeCorrelationNetwork: _build_candidate_inputs: Training inputs: length: {len(training_inputs)}, Type: {type(training_inputs)}, Content:\n{training_inputs}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}")
        ( candidate_input, candidate_epochs, y, residual_error, candidate_learning_rate, candidate_display_frequency,) = training_inputs
        logger.debug( f"CascadeCorrelationNetwork: _build_candidate_inputs: Successfully Unpacked Training inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}.")
        logger.debug( f"CascadeCorrelationNetwork: _build_candidate_inputs: Unpacked Task data, Candidate data, and Training inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Training Inputs: x shape: {candidate_input.shape}, epochs: {candidate_epochs}, y shape: {y.shape}, residual_error shape: {residual_error.shape}, learning_rate: {candidate_learning_rate}, display_frequency: {candidate_display_frequency}")
        logger.verbose( f"CascadeCorrelationNetwork: _build_candidate_inputs: Training inputs: x shape: {candidate_input.shape}, epochs: {candidate_epochs}, y shape: {y.shape}, residual_error shape: {residual_error.shape}, learning_rate: {candidate_learning_rate}, display_frequency: {candidate_display_frequency}")
        activation_fn = CascadeCorrelationNetwork._get_activation_function( activation_name)
        logger.debug( f"CascadeCorrelationNetwork: _build_candidate_inputs: Retrieved wrapped activation function: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Activation Function: Name: {activation_name}, Function: {activation_fn}")

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
            "activation_fn": activation_fn,
        }
        logger.debug( f"CascadeCorrelationNetwork: _build_candidate_inputs: Successfully built candidate inputs: {candidate_inputs}")
        return candidate_inputs

    @staticmethod
    def _train_candidate_unit(
        candidate: CandidateUnit = None,
        candidate_uuid: uuid = None,
        candidate_index: int = 0,
        candidate_input: tuple = None,
        candidate_epochs: int = 0,
        residual_error: float = 0.0,
        candidate_learning_rate: float = 0.0,
        candidate_display_frequency: int = 0,
        worker_id: int = 0,
        worker_uuid: str = "None",
    ) -> CandidateTrainingResult:
        # Train the candidate unit
        global shared_object_dict
        logger = Logger

        try:
            logger.debug( f"CascadeCorrelationNetwork: _train_candidate_unit: Training CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate.get_uuid()}, Candidate Object: {candidate}")
            training_result = candidate.train(
                x=candidate_input,
                epochs=candidate_epochs,
                residual_error=residual_error,
                learning_rate=candidate_learning_rate,
                display_frequency=candidate_display_frequency,
            )
            logger.info( f"CascadeCorrelationNetwork: _train_candidate_unit: Completed Training CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Correlation: {float(training_result.correlation):.6f}")
            logger.debug( f"CascadeCorrelationNetwork: _train_candidate_unit: Clearing Display Progress and Display Status for Candidate Unit: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}")
            candidate.clear_display_progress()  # Clear display progress for candidate unit, to avoid issues with multiprocessing--nested functions are not pickleable
            logger.debug( f"CascadeCorrelationNetwork: _train_candidate_unit: Cleared Display Progress for Candidate Unit: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}")
            candidate.clear_display_status()  # Clear display status for candidate unit, to avoid issues with multiprocessing--nested functions are not pickleable
            logger.debug( f"CascadeCorrelationNetwork: _train_candidate_unit: Cleared Display Status for Candidate Unit: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}")

            # Return CandidateTrainingResult with updated values
            training_result.candidate_id = candidate_index
            training_result.candidate_uuid = candidate_uuid
            training_result.candidate = candidate
            return training_result
        except Exception as e:
            logger.error( f"CascadeCorrelationNetwork: _train_candidate_unit: Caught Exception while training CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Error during candidate training:\nException:\n{e}")
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
    def _get_activation_function( activation_function_name: str = None, activation_functions_dict: dict = None) -> callable:
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
            activation_functions_dict = ( _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT)
        if activation_function_name is None:
            activation_function_name = ( _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME)
        return activation_functions_dict.get( activation_function_name, _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,)

    #################################################################################################################################################################################################
    # Multiprocessing Manager methods
    #################################################################################################################################################################################################
    def _start_manager(self):
        """Start the multiprocessing manager server in non-blocking mode."""
        self.logger.debug( "CascadeCorrelationNetwork: _start_manager: Starting multiprocessing manager")
        if self._manager is not None:
            self.logger.warning( "CascadeCorrelationNetwork: _start_manager: Manager already started")
            return
        address = self.candidate_training_queue_address
        authkey = self.candidate_training_queue_authkey
        if isinstance(authkey, str):
            authkey = authkey.encode("utf-8")
        try:
            self._manager = CandidateTrainingManager( address=address, authkey=authkey, ctx=self._mp_ctx)
            self._manager.start()  # Non-blocking - server runs in background

            # Obtain queue proxies
            self._task_queue = self._manager.get_task_queue()
            self._result_queue = self._manager.get_result_queue()
            self.logger.info( f"CascadeCorrelationNetwork: _start_manager: Manager started at {address}")
        except Exception as e:
            self.logger.error( f"CascadeCorrelationNetwork: _start_manager: Failed to start manager: {e}")
            raise

    def _stop_manager(self):
        """Stop the multiprocessing manager server."""
        self.logger.debug( "CascadeCorrelationNetwork: _stop_manager: Stopping multiprocessing manager")
        if self._manager is not None:
            try:
                self._manager.shutdown()
                self.logger.info( "CascadeCorrelationNetwork: _stop_manager: Manager shutdown completed")
            except Exception as e:
                self.logger.error( f"CascadeCorrelationNetwork: _stop_manager: Error shutting down manager: {e}")
            finally:
                self._manager = None
                self._task_queue = None
                self._result_queue = None

    # TODO: maybe break this up
    @staticmethod
    def _worker_loop(
        task_queue: Queue,
        result_queue: Queue,
        parallel: bool = True,
        task_queue_timeout: float = _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT,
    ):
        """
        Description:
            Worker process loop that processes tasks with stand-by mode.
        Args:
            task_queue: Queue to get tasks from
            result_queue: Queue to put results into
            parallel: Whether running in parallel mode
            task_queue_timeout: Timeout for getting tasks from queue
        Raises:
            TrainingError: If an error occurs during task processing
        Notes:
            - This function runs in a separate process and continuously checks for new tasks.
            - If no tasks are available, it enters a stand-by mode to save resources.
        Returns:
            None
        """
        logger = Logger
        from queue import Empty
        logger.debug("CascadeCorrelationNetwork: _worker_loop: Worker process started")
        while True:
            try:
                # Get task from queue with timeout
                task = task_queue.get(timeout=task_queue_timeout)
            except Empty:

                # Stand-by mode: no task available, continue waiting
                logger.debug( "CascadeCorrelationNetwork: _worker_loop: No task available, entering stand-by mode")
                time.sleep(0.1)
                continue
            except Exception as e:
                logger.critical( f"CascadeCorrelationNetwork: _worker_loop: Worker critical get error: {e}")
                import traceback
                logger.critical( f"CascadeCorrelationNetwork: _worker_loop: Traceback: {traceback.format_exc()}")
                break

            # Sentinel value to stop worker
            if task is None:
                logger.debug( "CascadeCorrelationNetwork: _worker_loop: Received sentinel, stopping worker")
                break
            try:

                # Process the task
                logger.debug( f"CascadeCorrelationNetwork: _worker_loop: Processing task: {task[0] if task else 'None'}")
                result = CascadeCorrelationNetwork.train_candidate_worker( task_data_input=task, parallel=parallel)
                logger.debug( "CascadeCorrelationNetwork: _worker_loop: Task processed, putting result in queue")

                # Add timeout to prevent deadlock if queue is full
                from queue import (Full,)
                try:
                    result_queue.put(result, timeout=30)
                    logger.debug( "CascadeCorrelationNetwork: _worker_loop: Task completed successfully")
                except Full as fe:
                    logger.error( f"CascadeCorrelationNetwork: _worker_loop: Result queue full, dropping result: {fe}")
                    import traceback
                    logger.error( f"CascadeCorrelationNetwork: _worker_loop: Traceback: {traceback.format_exc()}")
                    raise TrainingError from fe  # Re-raise to trigger error handling
            except Exception as e:
                logger.error( f"CascadeCorrelationNetwork: _worker_loop: Worker task error: {e}")
                import traceback
                logger.error( f"CascadeCorrelationNetwork: _worker_loop: Traceback: {traceback.format_exc()}")

                # Publish failure result and continue running
                try:
                    candidate_index = task[0] if task and len(task) > 0 else 0
                    candidate_uuid = ( task[1][4] if task and len(task) > 1 and len(task[1]) > 4 else None)
                    from candidate_unit.candidate_unit import CandidateTrainingResult
                    failure_result = CandidateTrainingResult(
                        candidate_id=candidate_index,
                        candidate_uuid=candidate_uuid,
                        correlation=0.0,
                        candidate=None,
                        success=False,
                        error_message=str(e),
                    )
                    result_queue.put(failure_result, timeout=30)
                    logger.debug( "CascadeCorrelationNetwork: _worker_loop: Put failure result")
                except Full as fq_e:
                    logger.error( f"CascadeCorrelationNetwork: _worker_loop: Failed to put failure result - queue full: {fq_e}")
                    import traceback
                    logger.error( f"CascadeCorrelationNetwork: _worker_loop: Traceback: {traceback.format_exc()}")
                except Exception as put_e:
                    logger.error( f"CascadeCorrelationNetwork: _worker_loop: Failed to put failure result: {put_e}")
                    import traceback
                    logger.error( f"CascadeCorrelationNetwork: _worker_loop: Traceback: {traceback.format_exc()}")
        logger.debug("CascadeCorrelationNetwork: _worker_loop: Worker process ended")

    #################################################################################################################################################################################################
    # Public Method to calculate the residual error of the network
    def calculate_residual_error( self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Description:
            Calculate the residual error of the network.
        Args:
            x: Input tensor
            y: Target tensor
        Notes:
            - The input and target tensors must have the same shape.
        Returns:
            Residual error tensor
        """
        self.logger.debug( f"CascadeCorrelationNetwork: calculate_residual_error: Calculating residual error for input shape: {x.shape if isinstance(x, torch.Tensor) else 'None'}, target shape: {y.shape if isinstance(y, torch.Tensor) else 'None'}")
        x = (x, torch.empty(0, self.input_size))[x is None]
        self.logger.debug( f"CascadeCorrelationNetwork: calculate_residual_error: After defaulting, input shape: {x.shape if isinstance(x, torch.Tensor) else 'None'}, target shape: {y.shape if isinstance(y, torch.Tensor) else 'None'}")
        y = (y, torch.empty(0, self.output_size))[y is None]
        residual = torch.empty(0, self.output_size)
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            # raise TypeError("Input and target must be torch.Tensor")
            self.logger.debug( f"CascadeCorrelationNetwork: calculate_residual_error: Input and target must be torch.Tensor, x type: {type(x)}, y type: {type(y)}")
            x = torch.empty(0, self.input_size)
            y = torch.empty(0, self.output_size)
            self.logger.debug( f"CascadeCorrelationNetwork: calculate_residual_error: After defaulting, input shape: {x.shape}, target shape: {y.shape}")
        if x.shape[1] != y.shape[1]:
            self.logger.debug( f"CascadeCorrelationNetwork: calculate_residual_error: Input and target must have the same number of features for dim: 1, x shape: {x.shape}, y shape: {y.shape}")
            # raise ValueError("Input and target must have the same number of features")
        elif x.shape[0] != y.shape[0]:
            self.logger.debug( f"CascadeCorrelationNetwork: calculate_residual_error: Input and target must have the same number of features for dim: 0, x shape: {x.shape}, y shape: {y.shape}")
            # raise ValueError("Input and target must have the same number of samples")
        else:
            # result = torch.empty(0, simple_network.input_size)
            self.logger.debug( "CascadeCorrelationNetwork: calculate_residual_error: Forward pass to calculate output for residual error computation")
            with torch.no_grad():
                self.logger.debug( "CascadeCorrelationNetwork: calculate_residual_error: Performing forward pass without gradient tracking")
                output = self.forward(x)
                self.logger.debug( f"CascadeCorrelationNetwork: calculate_residual_error: Forward pass completed, output shape: {output.shape}, Output:\n{output}")
                residual = y - output
                self.logger.debug( f"CascadeCorrelationNetwork: calculate_residual_error: Calculated residual error, shape: {residual.shape}, Residual Error:\n{residual}")
            self.logger.verbose( f"CascadeCorrelationNetwork: calculate_residual_error: Validating residual error, shape: {residual.shape}, Residual Error:\n{residual}")
            residual = (residual, torch.empty(0, self.output_size))[residual is None]
            self.logger.debug( f"CascadeCorrelationNetwork: calculate_residual_error: Calculated residual error, shape: {residual.shape}, Residual Error:\n{residual}")
        self.logger.verbose( f"CascadeCorrelationNetwork: calculate_residual_error: Returning residual error, shape: {residual.shape}, Residual Error:\n{residual}")
        return residual

    #################################################################################################################################################################################################
    # Public Method to add a new hidden unit based on the correlation
    def add_unit( self, candidate: CandidateUnit = None, x: torch.Tensor = None,) -> None:
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
        self.logger.trace( "CascadeCorrelationNetwork: add_unit: Starting to add a new hidden unit.")
        hidden_outputs = []
        for unit in self.hidden_units:
            unit_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
            unit_output = unit["activation_fn"]( torch.sum(unit_input * unit["weights"], dim=1) + unit["bias"]).unsqueeze(1)
            self.logger.debug( f"CascadeCorrelationNetwork: add_unit: Unit output shape: {unit_output.shape}, Unit output: {unit_output}")
            hidden_outputs.append(unit_output)
        self.logger.debug( f"CascadeCorrelationNetwork: add_unit: Hidden outputs shape: {[h.shape for h in hidden_outputs]}")
        candidate_input = ( torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x)
        self.logger.debug( f"CascadeCorrelationNetwork: add_unit: Candidate input shape: {candidate_input.shape}, Input size: {candidate_input.shape[1]}, Candidate Input:\n{candidate_input}")

        # Create a new hidden unit
        new_unit = {
            "weights": candidate.weights.clone().detach(),
            "bias": candidate.bias.clone().detach(),
            "activation_fn": self.activation_fn,
            "correlation": candidate.correlation,
        }
        self.logger.debug( f"CascadeCorrelationNetwork: add_unit: Adding new hidden unit with weights: {new_unit['weights']}, bias: {new_unit['bias']}, correlation: {new_unit['correlation']:.6f}, Unit: {new_unit}")

        # Add the new unit to the network
        self.hidden_units.append(new_unit)
        self.logger.debug( f"CascadeCorrelationNetwork: add_unit: Current number of hidden units: {len(self.hidden_units)}, Hidden units: {self.hidden_units}")

        # Update output layer weights to include the new unit
        old_output_weights = self.output_weights.clone().detach()
        self.logger.debug( f"CascadeCorrelationNetwork: add_unit: Old output weights shape: {old_output_weights.shape}, Weights: {old_output_weights}")
        old_output_bias = self.output_bias.clone().detach()
        self.logger.debug( f"CascadeCorrelationNetwork: add_unit: Old output bias shape: {old_output_bias.shape}, Bias: {old_output_bias}")

        # Calculate the output of the new unit
        unit_output = self.activation_fn( torch.sum(candidate_input * new_unit["weights"], dim=1) + new_unit["bias"]).unsqueeze(1)
        self.logger.debug( f"CascadeCorrelationNetwork: add_unit: New unit output shape: {unit_output.shape}, New unit output: {unit_output}")

        # Create new output weights with an additional row for the new unit
        if hidden_outputs:
            new_input_size = x.shape[1] + len(hidden_outputs) + 1
        else:
            new_input_size = x.shape[1] + 1
        self.logger.debug( f"CascadeCorrelationNetwork: add_unit: New input size for output weights: {new_input_size}, Old input size: {old_output_weights.shape[0]}")

        # Ensure new weights have requires_grad=True
        self.output_weights = ( torch.randn(new_input_size, self.output_size, requires_grad=True) * 0.1)
        self.logger.debug( f"CascadeCorrelationNetwork: add_unit: New output weights shape: {self.output_weights.shape}, Weights: {self.output_weights}")

        # Copy old weights
        if hidden_outputs:
            input_size_before = x.shape[1] + len(hidden_outputs)
        else:
            input_size_before = x.shape[1]
        self.logger.debug( f"CascadeCorrelationNetwork: add_unit: Input size before adding new unit: {input_size_before}")

        # Copy old bias
        self.output_weights[:input_size_before, :] = old_output_weights
        self.logger.debug( f"CascadeCorrelationNetwork: add_unit: Updated output weights after copying old weights: {self.output_weights}")
        self.output_bias = old_output_bias
        self.logger.debug( f"CascadeCorrelationNetwork: add_unit: Updated output bias after copying old bias: {self.output_bias}")

        # Add new unit to the history
        self.logger.info( f"CascadeCorrelationNetwork: add_unit: Added hidden unit with correlation: {candidate.correlation:.6f}")
        self.history["hidden_units_added"].append(
            {
                "correlation": candidate.correlation,
                "weights": candidate.weights.clone().detach().numpy(),
                "bias": candidate.bias.clone().detach().numpy(),
            }
        )
        self.logger.info( f"CascadeCorrelationNetwork: add_unit: Current number of hidden units: {len(self.hidden_units)}")
        self.logger.debug( f"CascadeCorrelationNetwork: add_unit: Updated history with new hidden unit:\n{self.history['hidden_units_added'][-1]}\nHistory\n{self.history}")
        self.logger.trace( "CascadeCorrelationNetwork: add_unit: Completed adding a new hidden unit.")

    def _select_best_candidates(self, results: list, num_candidates: int = 1) -> list:
        """
        Description:
            Select top N candidates for layer addition.
        Args:
            results: List of CandidateTrainingResult objects
            num_candidates: Number of candidates to select
        Notes:
            - Candidates are sorted by absolute correlation.
            - Top N candidates are selected.
            - Candidates below a correlation threshold are filtered out.
        Returns:
            List of selected CandidateTrainingResult objects
        """
        self.logger.debug( f"CascadeCorrelationNetwork: _select_best_candidates: Selecting top {num_candidates} from {len(results)} candidates")

        # Sort by absolute correlation
        sorted_results = sorted(
            results,
            key=lambda r: abs(r.correlation) if r.correlation else 0,
            reverse=True,
        )

        # Select top N
        selected = sorted_results[:num_candidates]

        # Filter by threshold
        threshold = getattr(self, "correlation_threshold", 0.0)
        selected = [r for r in selected if abs(r.correlation) >= threshold]
        self.logger.info( f"CascadeCorrelationNetwork: _select_best_candidates: Selected {len(selected)} candidates with correlations: {[r.correlation for r in selected]}")
        return selected

    def add_units_as_layer(self, candidates: list, x: torch.Tensor) -> None:
        """
        Description:
            Add multiple candidates as a new layer. Each of top N candidates is added as a separate hidden unit.
        Args:
            candidates: List of CandidateTrainingResult objects
            x: Input tensor for calculating outputs
        Notes:
            - Each candidate in the list is added as a separate hidden unit.
            - The output layer weights are updated to include the new units.
        Returns:
            None
        """
        self.logger.info( f"CascadeCorrelationNetwork: add_units_as_layer: Adding layer with {len(candidates)} units")
        for candidate in candidates:
            if candidate.candidate and hasattr(candidate.candidate, "weights"):
                self.add_unit(candidate.candidate, x)
            else:
                self.logger.warning( f"CascadeCorrelationNetwork: add_units_as_layer: Skipping invalid candidate: {candidate}")
        self.logger.info( f"CascadeCorrelationNetwork: add_units_as_layer: Layer added, total hidden units: {len(self.hidden_units)}")

    #################################################################################################################################################################################################
    # Public Method to grow the network by adding hidden units
    # This method is the core of the Cascade Correlation algorithm
    # It iteratively adds hidden units based on the residual error until stopping criteria are met
    def grow_network(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        max_epochs: int = 1000,
        early_stopping: bool = True,
        patience_counter: int = 0,
        candidate: CandidateUnit = None,
        best_value_loss: float = float("inf"),
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
    ) -> ValidateTrainingResults:
        """
        Description:
            Grow the network by adding hidden units based on the residual error until stopping criteria are met.
        Args:
            x_train: Training input tensor
            y_train: Training target tensor
            max_epochs: Maximum number of epochs to train
            early_stopping: Whether to use early stopping
            patience_counter: Counter for early stopping patience
            candidate: Candidate unit for training
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
        self.logger.trace( "CascadeCorrelationNetwork: grow_network: Starting to grow the network by adding hidden units.")

        # TODO: validate_training_results bug: needs to be fixed

        # validate_training_results = ValidateTrainingResults()
        # 'early_stop', 'patience_counter', 'best_value_loss', 'value_output', 'value_loss', and 'value_accuracy'
        validate_training_results = None
        for epoch in range(max_epochs):

            # Calculate residual error
            residual_error = self._calculate_residual_error_safe( x_train=x_train, y_train=y_train)
            if residual_error is None:
                self.logger.warning( "CascadeCorrelationNetwork: grow_network: Residual error is None, stopping growth of the network.")
                break
            self.logger.debug( f"CascadeCorrelationNetwork: grow_network: Epoch {epoch}, Residual Error: {residual_error.mean().item():.6f}")

            # Train candidate units
            if ( not ( training_results := self._get_training_results( x_train=x_train, y_train=y_train, residual_error=residual_error)) or not training_results.best_candidate):
                self.logger.warning( "CascadeCorrelationNetwork: grow_network: Training results are None or best candidate is None, stopping growth of the network.")
                break

            # Check if best candidate meets correlation threshold
            elif ( training_results.best_candidate.get_correlation() < self.correlation_threshold):
                self.logger.info( f"CascadeCorrelationNetwork: grow_network: No candidate met correlation threshold: {self.correlation_threshold}, Best Correlation Achieved: {training_results.best_candidate.get_correlation():.6f}")
                break
            self.logger.info( f"CascadeCorrelationNetwork: grow_network: Best Candidate: {training_results.best_candidate.get_correlation() if training_results.best_candidate else None}, Met correlation threshold: {self.correlation_threshold}")

            # Determine number of candidates to add
            candidates_per_layer = getattr(self, "candidates_per_layer", 1)

            # Add candidate(s) to the network and retrain the output layer
            if candidates_per_layer > 1:
                if selected_candidates := self._select_best_candidates( training_results.candidate_objects, num_candidates=candidates_per_layer,):
                    self.add_units_as_layer( [c for c in selected_candidates if c.candidate], x_train)
                    train_loss = self.train_output_layer( x_train, y_train, self.output_epochs)
                    train_accuracy = self.get_accuracy(x_train, y_train)
                    self.logger.info( f"CascadeCorrelationNetwork: grow_network: Added {len(selected_candidates)} candidates as layer")
                else:
                    self.logger.warning( "CascadeCorrelationNetwork: grow_network: No candidates met selection criteria")
                    break
            else:

                # Original behavior: Add single best candidate
                train_loss, train_accuracy = self._add_best_candidate( training_results.best_candidate, x_train, y_train, epoch)
            self.logger.debug( f"CascadeCorrelationNetwork: grow_network: After adding candidate(s), Training Loss: {train_loss:.6f}, Training Accuracy: {train_accuracy:.4f}, For Current Epoch {epoch}, Post-Train History:\n{self.history}")

            # Prepare inputs for validation of training results
            validate_training_inputs = ValidateTrainingInputs(
                epoch=epoch,
                max_epochs=max_epochs,
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
            self.logger.debug( f"CascadeCorrelationNetwork: grow_network: Validate Training Inputs: {validate_training_inputs}")

            # Validation of training results
            try:
                validate_training_results: ValidateTrainingResults = ( self.validate_training(validate_training_inputs))
                self.logger.debug( f"CascadeCorrelationNetwork: grow_network: Validation Results: {validate_training_results}")
            except Exception as e:
                self.logger.error( f"CascadeCorrelationNetwork: grow_network: Caught Exception while validating training at epoch {epoch + 1}/{max_epochs}:\nException:\n{e}")
                import traceback
                traceback.print_exc()
                raise TrainingError from e

            # Update variables from validation results
            self.logger.debug( f"CascadeCorrelationNetwork: grow_network: Epoch {epoch}, Early Stop: {validate_training_results.early_stop}, Patience Counter: {validate_training_results.patience_counter}, Best Value Loss: {validate_training_results.best_value_loss:.6f}, Value Output: {validate_training_results.value_output} Value Loss: {validate_training_results.value_loss:.6f}, Value Accuracy: {validate_training_results.value_accuracy:.4f}")
            if validate_training_results.early_stop:
                self.logger.info( f"CascadeCorrelationNetwork: grow_network: Early stopping triggered at epoch {epoch}.")
                break
            self.logger.info( f"CascadeCorrelationNetwork: grow_network: Epoch {epoch} - Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}, Early stop: {validate_training_results.early_stop}")
        if not validate_training_results:
            self.logger.warning( f"CascadeCorrelationNetwork: grow_network: Validation failed at epoch {epoch + 1}/{max_epochs}.")
            validate_training_results = ValidateTrainingResults(
                early_stop=False,
                patience_counter=patience_counter,
                best_value_loss=best_value_loss,
                value_output=None,
                value_loss=float("inf"),
                value_accuracy=0.0,
            )
        self.logger.info( f"CascadeCorrelationNetwork: grow_network: Finished training after {epoch + 1} epochs. Total hidden units: {len(self.hidden_units)}")
        self.logger.debug( f"CascadeCorrelationNetwork: grow_network: Final history:\n{self.history}")
        self.logger.trace( "CascadeCorrelationNetwork: grow_network: Completed training of the network.")
        return validate_training_results

    # Calculate residual error
    def _calculate_residual_error_safe(
        self,
        x_train: torch.Tensor = None,
        y_train: torch.Tensor = None,
        epoch: int = 0,
        max_epochs: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Description:
            Safely calculate the residual error between predicted and true values. Handles exceptions and logs progress.
        Args:
            x_train: Training input tensor
            y_train: Training target tensor
            epoch: Current epoch number
            max_epochs: Maximum number of epochs
        Raises:
            TrainingError: If an error occurs during calculation
        Notes:
            - Validates input tensors
            - Logs progress and errors
        Returns:
            Residual error tensor or None if an error occurred
        """
        # Validate method input parameters
        if ( x_train is None or y_train is None or x_train.shape[0] == 0 or y_train.shape[0] == 0):
            self.logger.warning( "CascadeCorrelationNetwork: _calculate_residual_error_safe: Training data is None or empty, cannot calculate residual error")
            return None
        try:
            self.logger.debug( f"CascadeCorrelationNetwork: _calculate_residual_error_safe: Starting epoch {epoch + 1}/{max_epochs}")
            residual_error = self.calculate_residual_error(x_train, y_train)
            self.logger.debug( f"CascadeCorrelationNetwork: _calculate_residual_error_safe: Epoch {epoch}, Residual Error: {residual_error.mean().item():.6f}")
        except Exception as e:
            self.logger.error( f"CascadeCorrelationNetwork: _calculate_residual_error_safe: Caught Exception while logging epoch {epoch + 1}/{max_epochs}:\nException:\n{e}")
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
        epoch: int = 0,
        max_epochs: int = 0,
    ) -> TrainingResults:
        """
        Description:
            Get training results for candidate units
        Args:
            x_train: Training input tensor
            y_train: Training target tensor
            residual_error: Residual error tensor
            epoch: Current epoch number
            max_epochs: Maximum number of epochs
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
            training_results = self.train_candidates( x=x_train, y=y_train, residual_error=residual_error)
            self.logger.debug( f"CascadeCorrelationNetwork: _get_training_results: Training Results - Epoch {epoch}, Successful: {training_results.successful_candidates}, Failed: {training_results.failed_count}")
        except Exception as e:
            self.logger.error( f"CascadeCorrelationNetwork: _get_training_results: Caught Exception while training candidates at epoch {epoch + 1}/{max_epochs}:\nException:\n{e}")
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
        epoch: int = 0,
        max_epochs: int = None,
    ) -> Optional[Tuple[float, float]]:
        self.logger.info( f"CascadeCorrelationNetwork: _add_best_candidate: Adding best candidate {best_candidate} at epoch {epoch}")
        if best_candidate is None:
            self.logger.warning( "CascadeCorrelationNetwork: _add_best_candidate: Best candidate is None, cannot add to network")
            return None, None
        try:

            # Add best candidate to the network
            self.add_unit(best_candidate, x_train)
            self.logger.info( "CascadeCorrelationNetwork: _add_best_candidate: Added best candidate to the network")
            train_loss = self._retrain_output_layer( x_train=x_train, y_train=y_train, epochs=self.output_epochs, epoch=epoch)
            self.logger.debug( f"CascadeCorrelationNetwork: _add_best_candidate: Training Loss: {train_loss}, For Current Epoch {epoch}, Post-Train Loss History:\n{self.history}")
            train_accuracy = self._calculate_train_accuracy( x_train=x_train, y_train=y_train, epoch=epoch)
            self.logger.debug( f"CascadeCorrelationNetwork: _add_best_candidate: Training Accuracy: {train_accuracy}, For Current Epoch {epoch}, Post-Train Accuracy History:\n{self.history}")
        except Exception as e:
            self.logger.error( f"CascadeCorrelationNetwork: _add_best_candidate: Caught Exception while adding unit and retraining output layer at epoch {epoch + 1}/{max_epochs}:\nException:\n{e}")
            import traceback
            traceback.print_exc()
            raise TrainingError from e
        return train_loss, train_accuracy

    # Calculate training accuracy
    def _calculate_train_accuracy( self, x_train: torch.Tensor = None, y_train: torch.Tensor = None, epoch: int = 0) -> float:
        # Validate method input parameters
        if ( x_train is None or y_train is None or x_train.shape[0] == 0 or y_train.shape[0] == 0):
            self.logger.warning( "CascadeCorrelationNetwork: _calculate_train_accuracy: Training data is None or empty, cannot calculate accuracy")
            return 0.0
        if x_train.shape[0] != y_train.shape[0]:
            self.logger.warning( f"CascadeCorrelationNetwork: _calculate_train_accuracy: Training data and target have different number of samples, x_train shape: {x_train.shape}, y_train shape: {y_train.shape}, cannot calculate accuracy")
            return 0.0

        # Calculate accuracy
        train_accuracy = self.calculate_accuracy(x_train, y_train)
        self.logger.debug( f"CascadeCorrelationNetwork: _calculate_train_accuracy: For Current Epoch {epoch}, Train Accuracy: {train_accuracy:.4f}")

        # Update training history
        self.history["train_accuracy"].append(train_accuracy)
        self.logger.debug( f"CascadeCorrelationNetwork: _calculate_train_accuracy: For Current Epoch {epoch}, Post-Train Accuracy History:\n{self.history}")
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
        if ( x_train is None or y_train is None or x_train.shape[0] == 0 or y_train.shape[0] == 0):
            self.logger.warning( "CascadeCorrelationNetwork: _retrain_output_layer: Training data is None or empty, cannot retrain output layer")
            return float("inf")
        if x_train.shape[0] != y_train.shape[0]:
            self.logger.warning( f"CascadeCorrelationNetwork: _retrain_output_layer: Training data and target have different number of samples, x_train shape: {x_train.shape}, y_train shape: {y_train.shape}, cannot retrain output layer")
            return float("inf")
        if epochs <= 0:
            self.logger.warning( f"CascadeCorrelationNetwork: _retrain_output_layer: Number of epochs for retraining output layer is non-positive: {epochs}, skipping retraining")
            return float("inf")
        self.logger.info( f"CascadeCorrelationNetwork: _retrain_output_layer: Retraining output layer for {epochs} epochs after adding new hidden unit")

        # Retrain output layer
        train_loss = self.train_output_layer(x_train, y_train, self.output_epochs)
        self.logger.info( f"CascadeCorrelationNetwork: _retrain_output_layer: Full Network Training Loss after Epoch {epoch}, Train Loss: {train_loss:.6f}")
        self.logger.debug( f"CascadeCorrelationNetwork: _retrain_output_layer: For Current Epoch: {epoch}, Post-Train Loss History:\n{self.history}")

        # Update training history
        self.history["train_loss"].append(train_loss)
        self.logger.debug( f"CascadeCorrelationNetwork: _retrain_output_layer: For Current Epoch: {epoch}, Post-Trained History:\n{self.history}")
        return train_loss

    #################################################################################################################################################################################################
    # Define Snapshot and Recovery methods using hdf5 serialization
    def create_snapshot( self, snapshot_dir: Union[str, pl.Path] = None) -> Optional[pl.Path]:
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
                snapshot_dir = pl.Path( self.cascade_correlation_network_snapshots_dir) or pl.Path(_CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR)
            else:
                snapshot_dir = pl.Path(snapshot_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp and UUID
            timestamp = pd.datetime.now().strftime("%Y%m%d_%H%M%S")
            uuid = str(self.get_uuid())
            filename = f"cascor_snapshot_{timestamp}_{uuid}.h5"
            snapshot_path = pl.Path(snapshot_dir).joinpath(filename)

            # Save the snapshot
            if self._save_to_hdf5( snapshot_path, include_training_data=False, create_backup=False,):
                self.logger.info( f"CascadeCorrelationNetwork: create_snapshot: Created snapshot at {snapshot_path}")
                return snapshot_path
            else:
                return None
        except Exception as e:
            self.logger.error(f"CascadeCorrelationNetwork: create_snapshot: Error: {e}")
            return None

    @classmethod
    def restore_snapshot( cls, snapshot_path: Union[str, pl.Path] = None, restore_multiprocessing: bool = True,) -> bool:
        """
        Restore the network state from a snapshot file.
        Args:
            snapshot_path: Path to the snapshot file
            restore_multiprocessing: Whether to restore multiprocessing state
        Returns:
            bool: Success status
        """
        logger = Logger
        try:
            if snapshot_path is None:
                logger.error( "CascadeCorrelationNetwork: restore_snapshot: No snapshot path provided")
                return False
            snapshot_path = pl.Path(snapshot_path)
            if not snapshot_path.exists():
                logger.error( f"CascadeCorrelationNetwork: restore_snapshot: Snapshot file does not exist: {snapshot_path}")
                return False
            loaded_network = cls._load_from_hdf5( filepath=snapshot_path, restore_multiprocessing=restore_multiprocessing, logger=logger,)
            if loaded_network is None:
                logger.error( f"CascadeCorrelationNetwork: restore_snapshot: Failed to load network from snapshot: {snapshot_path}")
                return False

            # Copy loaded network state into current instance
            cls.__dict__.update(loaded_network.__dict__)
            logger.info( f"CascadeCorrelationNetwork: restore_snapshot: Restored snapshot from {snapshot_path}")
            return True
        except Exception as e:
            logger.error( f"CascadeCorrelationNetwork: restore_snapshot: Error restoring snapshot: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    #################################################################################################################################################################################################
    # methods to save provided object to hdf5
    def save_object( self, objectify: Any = None, snapshot_dir: Union[str, pl.Path] = None) -> Optional[pl.Path]:
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
                snapshot_dir = pl.Path( self.cascade_correlation_network_snapshots_dir) or pl.Path(_CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR)
            else:
                snapshot_dir = pl.Path(snapshot_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with object's name, timestamp, and UUID
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            uuid = str(objectify.get_uuid())
            object_name = objectify.__name__
            filename = f"{object_name}_snapshot_{timestamp}_{uuid}.h5"
            snapshot_path = pl.Path(snapshot_dir).joinpath(filename)

            # Save the snapshot
            if self._save_to_hdf5( snapshot_path, objectify=objectify, create_backup=False,):
                self.logger.info( f"CascadeCorrelationNetwork: create_snapshot: Created snapshot at {snapshot_path}")
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
                self.logger.info( f"CascadeCorrelationNetwork: Created backup at {backup_path}")

            # Save the current object
            if success := serializer.save_object( objectify=objectify, filepath=filepath, compression=compression, compression_opts=compression_opts,):
                self.logger.info( f"CascadeCorrelationNetwork: save_to_hdf5: Successfully saved to {filepath}")
            else:
                self.logger.error( f"CascadeCorrelationNetwork: save_to_hdf5: Failed to save to {filepath}")
            self.logger.debug( "CascadeCorrelationNetwork: save_to_hdf5: Verifying saved HDF5 file")
            checked_object = self.verify_hdf5_file(filepath)
            if not checked_object.get("valid", False):
                self.logger.error( f"CascadeCorrelationNetwork: save_to_hdf5: Verification failed for saved HDF5 file: {filepath}, Error: {checked_object.get('error', 'Unknown error')}")
                return False
            self.logger.info( f"CascadeCorrelationNetwork: save_to_hdf5: Verified saved HDF5 file is valid: {filepath}")
            return success
        except Exception as e:
            self.logger.error( f"CascadeCorrelationNetwork: save_to_hdf5: Error saving to HDF5: {e}")
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
                self.logger.info( f"CascadeCorrelationNetwork: Created backup at {backup_path}")

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
                self.logger.info( f"CascadeCorrelationNetwork: save_to_hdf5: Successfully saved to {filepath}")
            else:
                self.logger.error( f"CascadeCorrelationNetwork: save_to_hdf5: Failed to save to {filepath}")
            self.logger.debug( "CascadeCorrelationNetwork: save_to_hdf5: Verifying saved HDF5 file")
            checked_network = self.verify_hdf5_file(filepath)
            if not checked_network.get("valid", False):
                self.logger.error( f"CascadeCorrelationNetwork: save_to_hdf5: Verification failed for saved HDF5 file: {filepath}, Error: {checked_network.get('error', 'Unknown error')}")
                return False
            self.logger.info( f"CascadeCorrelationNetwork: save_to_hdf5: Verified saved HDF5 file is valid: {filepath}")
            return success
        except Exception as e:
            self.logger.error( f"CascadeCorrelationNetwork: save_to_hdf5: Error saving to HDF5: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    @classmethod
    def load_from_hdf5( cls, filepath: Union[str, pl.Path], restore_multiprocessing: bool = False):
        """
        Public classmethod to load network from HDF5 file.
        Args:
            filepath: Path to HDF5 file
            restore_multiprocessing: Whether to restore multiprocessing state (default: False)
        Returns:
            CascadeCorrelationNetwork instance or None if failed
        """
        return cls._load_from_hdf5( filepath=filepath, restore_multiprocessing=restore_multiprocessing)

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
            network = serializer.load_network( filepath=filepath, restore_multiprocessing=restore_multiprocessing)
            if network:
                network.logger.info( f"CascadeCorrelationNetwork: load_from_hdf5: Successfully loaded from {filepath}")
            else:
                logger.error( f"CascadeCorrelationNetwork: load_from_hdf5: Failed to load from {filepath}")
            return network
        except Exception as e:
            logger.error( f"CascadeCorrelationNetwork: load_from_hdf5: Error loading from HDF5: {e}")
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
                self.logger.error( f"CascadeCorrelationNetwork: list_hdf5_snapshots: Directory does not exist: {directory}")
                return []
            hdf5_files = HDF5Utils.list_hdf5_files(directory)
            self.logger.info( f"CascadeCorrelationNetwork: list_hdf5_snapshots: Found {len(hdf5_files)} HDF5 files in {directory}")
            return hdf5_files
        except Exception as e:
            self.logger.error( f"CascadeCorrelationNetwork: list_hdf5_snapshots: Error listing HDF5 files: {e}")
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
            self.logger.error( f"CascadeCorrelationNetwork: Error verifying HDF5 file: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {"valid": False, "error": str(e)}

    #################################################################################################################################################################################################
    # Public Method to validate the training process
    #################################################################################################################################################################################################
    def validate_training(
        self,
        epoch: int = 0,
        max_epochs: int = 0,
        patience_counter: int = 0,
        early_stopping: bool = True,
        train_accuracy: float = 0.0,
        train_loss: float = float("inf"),
        best_value_loss: float = 9999999.9,
        x_train: torch.Tensor = None,
        y_train: torch.Tensor = None,
        x_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
    ) -> (bool, int, float, torch.Tensor, float, float):
        """
        Description:
            Validate the training process by checking the validation loss and accuracy.
        Args:
            epoch: Current epoch number
            max_epochs: Maximum number of epochs
            patience_counter: Counter for early stopping patience
            early_stopping: Whether to use early stopping
            train_accuracy: Training accuracy
            train_loss: Training loss
            best_value_loss: Best validation loss seen so far
            x_train: Training input tensor
            y_train: Training target tensor
            x_val: Validation input tensor
            y_val: Validation target tensor
        Raises:
            ValueError: If the input tensors are not valid
        Returns:
            Tuple containing:
                - early_stop_flag: Whether to stop training early
                - patience_counter: Updated patience counter
                - best_value_loss: Updated best validation loss
                - value_output: Output from the validation set
                - value_loss: Validation loss
                - value_accuracy: Validation accuracy
        """
        self.logger.trace( "CascadeCorrelationNetwork: validate_training: Starting validation of the training process.")
        early_stop_flag = False
        value_output = 0
        value_loss = float("inf")
        value_accuracy = 0.0
        best_value_loss = best_value_loss if best_value_loss is not None else 9999999.9
        self.logger.debug( f"CascadeCorrelationNetwork: validate_training: Epoch {epoch}, Max Epochs: {max_epochs}, Early Stopping: {early_stopping}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}, Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}")

        # Validate input tensors
        self.logger.debug( f"CascadeCorrelationNetwork: validate_training: X Train: {x_train}, Y Train: {y_train}, X Val: {x_val}, Y Val: {y_val}")
        if x_val is not None and y_val is not None:

            # Validate the model on the validation set
            with torch.no_grad():
                value_output = self.forward(x_val)
                value_loss = nn.MSELoss()(value_output, y_val).item()
            self.history["value_loss"].append(value_loss)

            # Calculate validation accuracy
            value_accuracy = self.calculate_accuracy(x_val, y_val)
            self.history["value_accuracy"].append(value_accuracy)
            self.logger.info( "CascadeCorrelationNetwork: validate_training: " f"Epoch {epoch} - Train Loss: {train_loss:.6f}, Val Loss: {value_loss:.6f}, " f"Train Acc: {train_accuracy:.4f}, Val Acc: {value_accuracy:.4f}, " f"Units: {len(self.hidden_units)}")

            # Check for early stopping conditions
            # TODO: Consider using named tuple or dataclass for return values
            (early_stop, patience_counter, best_value_loss) = (
                self.evaluate_early_stopping(
                    epoch=epoch,
                    max_epochs=max_epochs,
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    early_stopping=early_stopping,
                    value_loss=value_loss,
                    best_value_loss=best_value_loss,
                    patience_counter=patience_counter,
                )
            )
            self.logger.verbose( f"CascadeCorrelationNetwork: validate_training: Early Stopping: {early_stopping}")
            self.logger.verbose( f"CascadeCorrelationNetwork: validate_training: Early Stop: {early_stop}")
            self.logger.verbose( f"CascadeCorrelationNetwork: validate_training: Epoch: {epoch}, Early Stop: {early_stop}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}")

            # early_stop_flag = True if early_stopping and early_stop else False
            early_stop_flag = early_stopping and early_stop
            self.logger.info( f"CascadeCorrelationNetwork: validate_training: Stop Training Early: {early_stop} and Early Stopping: {early_stopping}: {early_stopping and early_stop}")
            self.logger.info( f"CascadeCorrelationNetwork: validate_training: Early Stopping: {early_stop_flag}, Patience Counter: {patience_counter}, Best Val Loss: {best_value_loss:.6f}")
            self.logger.verbose( f"CascadeCorrelationNetwork: validate_training: Value Output: {value_output}, Value Loss: {value_loss:.6f}, Value Accuracy: {value_accuracy:.4f}")

        self.logger.verbose( f"CascadeCorrelationNetwork: validate_training: Epoch {epoch}, Early Stop: {early_stop_flag}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}, Value Output: {value_output}, Value Loss: {value_loss:.6f}, Value Accuracy: {value_accuracy:.4f}")
        self.logger.trace( "CascadeCorrelationNetwork: validate_training: Completed validation of the training process.")

        # TODO: Consider using named tuple or dataclass for return values
        return (
            early_stop_flag,
            patience_counter,
            best_value_loss,
            value_output,
            value_loss,
            value_accuracy,
        )

    #################################################################################################################################################################################################
    # Public Method to evaluate early stopping conditions
    # This method checks if the training should stop early based on validation loss, patience, and other criteria
    def evaluate_early_stopping(
        self,
        epoch: int = 0,
        max_epochs: int = 0,
        train_loss: float = float("inf"),
        train_accuracy: float = 0.0,
        early_stopping: bool = True,
        value_loss: float = float("inf"),
        best_value_loss: float = float("inf"),
        patience_counter: int = 0,
    ) -> (bool, int, float):
        """
        Description:
            Evaluate early stopping conditions to determine if the training should stop.
        Args:
            epoch: Current epoch number
            max_epochs: Maximum number of epochs
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
        self.logger.trace( "CascadeCorrelationNetwork: evaluate_early_stopping: Starting evaluation of early stopping conditions.")

        # Initialize variables
        patience_exhausted = False
        max_units_reached = False
        train_accuracy_reached = False
        if early_stopping:

            # Check if we've reached the end of our patience
            # TODO: Consider using named tuple or dataclass for return values
            (patience_exhausted, patience_counter, best_value_loss) = (
                self.check_patience(
                    patience_counter=patience_counter,
                    value_loss=value_loss,
                    best_value_loss=best_value_loss,
                )
            )
            self.logger.info( f"CascadeCorrelationNetwork: evaluate_early_stopping: Epoch {epoch} - Patience Counter: {patience_counter}, Value Loss: {value_loss}, Best Val Loss: {best_value_loss:.6f}")
            if patience_exhausted:
                self.logger.info( f"CascadeCorrelationNetwork: evaluate_early_stopping: Patience Exhausted: {patience_exhausted}, Early stopping triggered after {epoch} epochs")
            else:
                self.logger.info( f"CascadeCorrelationNetwork: evaluate_early_stopping: Epoch {epoch} - Train Loss: {train_loss:.6f}, " f"Train Acc: {train_accuracy:.4f}, Units: {len(self.hidden_units)}")

            # Check if we've reached the maximum number of hidden units
            if max_units_reached := self.check_hidden_units_max():
                self.logger.info( f"CascadeCorrelationNetwork: evaluate_early_stopping: Reached maximum number of hidden units: {max_units_reached}, stopping training")

            # Check if we've achieved perfect accuracy
            if train_accuracy_reached := self.check_training_accuracy( train_accuracy=train_accuracy, accuracy_target=self.target_accuracy,):
                self.logger.info( f"CascadeCorrelationNetwork: evaluate_early_stopping: Training accuracy reached target: {train_accuracy:.4f} >= 0.999")

        early_stop = early_stopping and ( train_accuracy_reached or max_units_reached or patience_exhausted)
        self.logger.info( f"CascadeCorrelationNetwork: evaluate_early_stopping: Early Stopping: {early_stop}, Patience Counter: {patience_counter}, Best Val Loss: {best_value_loss:.6f}")
        self.logger.trace( "CascadeCorrelationNetwork: evaluate_early_stopping: Completed evaluation of early stopping conditions.")

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
    ) -> (bool, int, float):
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
        self.logger.trace( "CascadeCorrelationNetwork: check_patience: Starting to check patience limit.")
        self.logger.verbose( f"CascadeCorrelationNetwork: check_patience: Current Value Loss: {value_loss:.6f}, Best Value Loss: {best_value_loss:.6f}, Patience Counter: {patience_counter}")
        if value_loss < best_value_loss:
            best_value_loss = value_loss
            patience_counter = 0
        else:
            patience_counter += 1
        self.logger.info( f"CascadeCorrelationNetwork: check_patience: Patience counter: {patience_counter}, Best Validation Loss: {best_value_loss:.6f}")

        # Check if patience limit is reached
        if patience_exhausted := (patience_counter >= self.patience):
            self.logger.info( f"CascadeCorrelationNetwork: check_patience: Patience limit reached: {patience_counter} >= {self.patience}")
        self.logger.debug( f"CascadeCorrelationNetwork: check_patience: Patience Exhausted: {patience_exhausted}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}")
        self.logger.trace( "CascadeCorrelationNetwork: check_patience: Completed checking patience limit.")

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
        self.logger.trace( "CascadeCorrelationNetwork: check_hidden_units_max: Starting to check if max hidden units reached.")
        max_units_reached = len(self.hidden_units) >= self.max_hidden_units
        self.logger.info( f"CascadeCorrelationNetwork: check_hidden_units_max: Current hidden units: {max_units_reached}, Max allowed: {self.max_hidden_units}")
        if max_units_reached:
            self.logger.info( f"CascadeCorrelationNetwork: check_hidden_units_max: Reached maximum number of hidden units: {self.max_hidden_units}")
        self.logger.trace( "CascadeCorrelationNetwork: check_hidden_units_max: Completed checking if max hidden units reached.")
        return max_units_reached

    #################################################################################################################################################################################################
    # Public Method to check if training accuracy has reached the target
    # This method checks if the training accuracy has reached the target accuracy
    def check_training_accuracy( self, train_accuracy: float = 0.0, accuracy_target: float = 0.999,) -> bool:
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
        self.logger.trace( "CascadeCorrelationNetwork: check_training_accuracy: Starting to check if training accuracy has reached the target.")
        if train_accuracy_reached := (train_accuracy >= accuracy_target):
            self.logger.info( f"CascadeCorrelationNetwork: check_training_accuracy: Reached target training accuracy: {train_accuracy:.4f} >= {accuracy_target:.4f}")
        self.logger.debug( f"CascadeCorrelationNetwork: check_training_accuracy: Current Training Accuracy: {train_accuracy:.4f}, Target Accuracy: {accuracy_target:.4f}")
        self.logger.trace( "CascadeCorrelationNetwork: check_training_accuracy: Completed checking if training accuracy has reached the target.")
        return train_accuracy_reached

    ##################################################################################################################################################################################################
    # Public Method to calculate classification accuracy
    # This method calculates the classification accuracy of the network
    # It compares the predicted output with the target output
    def calculate_accuracy( self, x: torch.Tensor = None, y: torch.Tensor = None,) -> float:
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
        self.logger.trace( "CascadeCorrelationNetwork: calculate_accuracy: Starting to calculate accuracy.")
        x = (x, torch.empty(0, self.input_size))[x is None]
        y = (y, torch.empty(0, self.output_size))[y is None]
        accuracy = 0.0

        # Validate input tensors
        if x is None or y is None:
            self.logger.error( "CascadeCorrelationNetwork: calculate_accuracy: Missing required tensors for accuracy calculation, using safe defaults.")
            self.logger.debug( f"CascadeCorrelationNetwork: calculate_accuracy: input size: {self.input_size}, output size: {self.output_size}")
            x = torch.empty(0, self.input_size)
            y = torch.empty(0, self.output_size)
            # raise ValueError("CascadeCorrelationNetwork: calculate_accuracy: Missing required tensors for accuracy calculation.")
        if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            self.logger.error( f"CascadeCorrelationNetwork: calculate_accuracy: Input and target tensors must be of type torch.Tensor. Input (x): {type(x)}, Target (y): {type(y)}")
            raise ValueError( "CascadeCorrelationNetwork: calculate_accuracy: Input and target tensors must be of type torch.Tensor.")
        # elif x.shape[-1] != y.shape[-1]:
        #     self.logger.error( f"CascadeCorrelationNetwork: calculate_accuracy: Input shape: {x.shape}, Target shape: {y.shape}")
        #     raise ValueError( "CascadeCorrelationNetwork: calculate_accuracy: Input and target tensors must have the same number of features.")
        elif x.shape[0] != y.shape[0]:
            self.logger.error( f"CascadeCorrelationNetwork: calculate_accuracy: Input and target tensors must have compatible shapes. Input (x): {x.shape}, Target (y): {y.shape}, input size: {self.input_size}, output size: {self.output_size}")
            raise ValueError( "CascadeCorrelationNetwork: calculate_accuracy: Input and target tensors must have compatible shapes.")
        else:
            self.logger.debug( f"CascadeCorrelationNetwork: calculate_accuracy: Validated input shape: {x.shape}, Target shape: {y.shape}")

            # Calculating accuracy
            self.logger.debug( f"CascadeCorrelationNetwork: calculate_accuracy: Calculating accuracy for input shape: {x.shape}, target shape: {y.shape}")
            with torch.no_grad():
                output = self.forward(x)
                self.logger.debug( f"CascadeCorrelationNetwork: calculate_accuracy: Output shape: {output.shape}, Output: {output}")

                # Validate Output Tensor
                if not isinstance(output, torch.Tensor):
                    self.logger.error( f"CascadeCorrelationNetwork: calculate_accuracy: Output tensor must be of type torch.Tensor. Output: Type: {type(output)}")
                    raise ValueError( "CascadeCorrelationNetwork: calculate_accuracy: Output tensor must be of type torch.Tensor.")
                elif output.shape[-1] != y.shape[-1]:
                    self.logger.error( f"CascadeCorrelationNetwork: calculate_accuracy: Output shape: {output.shape}, Target shape: {y.shape}")
                    raise ValueError( "CascadeCorrelationNetwork: calculate_accuracy: Output and target tensors must have the same number of features.")
                elif output.shape[0] != y.shape[0]:
                    self.logger.error( f"CascadeCorrelationNetwork: calculate_accuracy: Output and target tensors must have compatible shapes. Output Tensor: {output.shape}, Target (y): {y.shape}, Output size: {output.size()}, Target size: {self.output_size}")
                    raise ValueError( "CascadeCorrelationNetwork: calculate_accuracy: Output and target tensors must have compatible shapes.")
                else:
                    self.logger.debug( f"CascadeCorrelationNetwork: calculate_accuracy: Validated Output shape: {output.shape}, Target shape: {y.shape}")
                accuracy = self._accuracy(y=y, output=output)
            self.logger.info( f"CascadeCorrelationNetwork: calculate_accuracy: Calculated accuracy: {accuracy:.4f}, Percentage: {accuracy * 100:.2f}%")

        # Returning accuracy
        self.logger.trace( "CascadeCorrelationNetwork: calculate_accuracy: Completed calculating accuracy.")
        return accuracy

    #################################################################################################################################################################################################
    def _accuracy( self, y: torch.Tensor = None, output: torch.Tensor = None,) -> float:
        """
        Description:
            Private method to calculate accuracy.
            This method is used internally to calculate the accuracy of the network.
        Args:
            target: Target output
            output: Raw output from the network
        Notes:
            - This method assumes that the target and output tensors are one-hot encoded.
            - The accuracy is calculated as the percentage of correct predictions over the total number of predictions.
            - If either the target or output tensor is missing, an error is raised.
        Returns:
            Accuracy as a float
        """
        self.logger.trace( "CascadeCorrelationNetwork: _accuracy: Starting to calculate accuracy.")

        # Validate input tensors
        if y is None or output is None:
            self.logger.error( "CascadeCorrelationNetwork: _accuracy: Missing required tensors for accuracy calculation.")
            raise ValueError( "CascadeCorrelationNetwork: _accuracy: Missing required tensors for accuracy calculation.")
        elif not (isinstance(y, torch.Tensor) and isinstance(output, torch.Tensor)):
            self.logger.error( "CascadeCorrelationNetwork: _accuracy: All inputs must be torch tensors.")
            raise TypeError( "CascadeCorrelationNetwork: _accuracy: All inputs must be torch tensors.")
        elif y.shape[0] != output.shape[0]:
            self.logger.error( f"CascadeCorrelationNetwork: _accuracy: Output and Target tensors must have the same number of samples. Got {y.shape[0]} and {output.shape[0]}.")
            raise ValueError( "CascadeCorrelationNetwork: _accuracy: Output and Target tensors must have the same number of samples.")
        self.logger.debug( f"CascadeCorrelationNetwork: _accuracy: Input shape: {y.shape}, Output shape: {output.shape}")
        self.logger.verbose( f"CascadeCorrelationNetwork: _accuracy: Input shape: {y.shape}, Input: {y}")
        self.logger.verbose( f"CascadeCorrelationNetwork: _accuracy: Output shape: {output.shape}, Output: {output}")

        # Find predicted and target values
        predicted = torch.argmax(output, dim=1)
        self.logger.verbose( f"CascadeCorrelationNetwork: _accuracy: Predicted shape: {predicted.shape}, Predicted: {predicted}")
        target = torch.argmax(y, dim=1)
        self.logger.verbose( f"CascadeCorrelationNetwork: _accuracy: Target shape: {target.shape}, Target: {target}")
        correct = (predicted == target).sum().item()
        self.logger.verbose( f"CascadeCorrelationNetwork: _accuracy: Number of correct predictions: {correct}, Total samples: {len(target)}")
        accuracy = correct / len(target)
        self.logger.info( f"CascadeCorrelationNetwork: _accuracy: Calculated accuracy: {accuracy:.4f}, Percentage: {accuracy * 100:.4f}%")
        self.logger.trace( "CascadeCorrelationNetwork: _accuracy: Completed calculating accuracy.")
        return accuracy

    #################################################################################################################################################################################################
    # Public Method to make predictions
    # This method uses the forward method to get the output of the network
    # It is used to make predictions on new data
    def predict( self, x: torch.Tensor) -> torch.Tensor:  # sourcery skip: class-extract-method
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
        self.logger.trace( "CascadeCorrelationNetwork: predict: Starting to make predictions.")
        with torch.no_grad():
            predicted_value = self.forward(x)
            self.logger.trace( "CascadeCorrelationNetwork: predict: Finished making predictions.")
        self.logger.debug( f"CascadeCorrelationNetwork: predict: Predicted shape: {predicted_value.shape}, Predicted: {predicted_value}")
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
        self.logger.debug( f"CascadeCorrelationNetwork: predict_classes: Input shape: {x.shape}")
        self.logger.trace( "CascadeCorrelationNetwork: predict_classes: Starting to predict class labels.")
        with torch.no_grad():
            output = self.forward(x)
            prediction = torch.argmax(output, dim=1)
            self.logger.info( f"CascadeCorrelationNetwork: predict_classes: Predicted class labels shape: {prediction.shape}, Prediction: {prediction}")
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
        self.logger.trace( "CascadeCorrelationNetwork: summary: Starting to print network summary.")
        self.logger.info( "CascadeCorrelationNetwork: summary: Display Cascade Correlation Network Summary:")
        self.logger.info( f"CascadeCorrelationNetwork: summary: Input size: {self.input_size}")
        self.logger.info( f"CascadeCorrelationNetwork: summary: Output size: {self.output_size}")
        self.logger.info( f"CascadeCorrelationNetwork: summary: Number of hidden units: {len(self.hidden_units)}")

        # Display hidden unit info if present
        if self.hidden_units:
            self.logger.info("CascadeCorrelationNetwork: summary: Hidden Units:\n")
            for i, unit in enumerate(self.hidden_units):
                self.logger.info(f"CascadeCorrelationNetwork: summary:   Unit {i+1}:")
                self.logger.info( f"CascadeCorrelationNetwork: summary:     Input size: {len(unit['weights'])}")
                self.logger.info( f"CascadeCorrelationNetwork: summary:     Correlation: {unit['correlation']:.6f}")

        # Display Training Parameters
        self.logger.info("CascadeCorrelationNetwork: summary: Training Parameters:")
        self.logger.info( f"CascadeCorrelationNetwork: summary:   Learning rate: {self.learning_rate}")
        self.logger.info( f"CascadeCorrelationNetwork: summary:   Candidate pool size: {self.candidate_pool_size}")
        self.logger.info( f"CascadeCorrelationNetwork: summary:   Correlation threshold: {self.correlation_threshold}")

        # Display final training accuracy if attribute exists
        if self.history["train_accuracy"]:
            self.logger.info( f"CascadeCorrelationNetwork: summary: Final training accuracy:\n{self.history['train_accuracy'][-1]:.6f}")

        # Display final value accuracy if validation was used
        if "value_accuracy" in self.history and self.history["value_accuracy"]:
            self.logger.info( f"CascadeCorrelationNetwork: summary: Final validation accuracy:\n{self.history['value_accuracy'][-1]:.6f}")
        self.logger.trace( "CascadeCorrelationNetwork: summary: Completed printing network summary.")

    #################################################################################################################################################################################################
    # Define public methods for plotting the dataset, decision boundary and training history
    # These methods now delegate to the CascadeCorrelationPlotter class
    #################################################################################################################################################################################################
    @staticmethod
    def plot_dataset( x: torch.Tensor, y: torch.Tensor, title: str = "Training Dataset",) -> None:
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
            self.logger.info( f"CascadeCorrelationNetwork: plot_decision_boundary: Started plotting process PID: {plot_process.pid}")
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
            self.logger.info( f"CascadeCorrelationNetwork: plot_training_history: Started plotting process PID: {plot_process.pid}")
            return plot_process
        else:
            self.plotter.plot_training_history(self.history)
            return None

    #################################################################################################################################################################################################
    # Define private method to generate a new uuid for the CascadeCorrelationNetwork class
    def _generate_uuid(self) -> None:
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
        logger = ( self.logger if hasattr(self, "logger") and self.logger is not None else Logger)
        logger.trace( "CascadeCorrelationNetwork: _generate_uuid: Inside the CascadeCorrelationNetwork class Generate UUID method")
        new_uuid = str(uuid.uuid4())
        logger.debug(f"CascadeCorrelationNetwork: _generate_uuid: UUID: {new_uuid}")
        logger.trace( "CascadeCorrelationNetwork: _generate_uuid: Completed the CascadeCorrelationNetwork class Generate UUID method")
        return new_uuid


    ####################################################################################################################################
    # Define CascadeCorrelationNetwork class Setters
    ####################################################################################################################################
    def set_candidate_training_queue_authkey( self, candidate_training_queue_authkey: bytes = None):
        self.candidate_training_queue_authkey = candidate_training_queue_authkey

    def set_candidate_training_queue_address( self, candidate_training_queue_address: str = None):
        self.candidate_training_queue_address = candidate_training_queue_address

    def set_candidate_training_tasks_queue_timeout( self, candidate_training_tasks_queue_timeout: int = None):
        self.candidate_training_tasks_queue_timeout = ( candidate_training_tasks_queue_timeout)

    def set_candidate_training_shutdown_timeout( self, candidate_training_shutdown_timeout: int = None):
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
            self._validate_numeric_parameter( learning_rate, "learning_rate", min_val=0.0, max_val=10.0)
        self.learning_rate = learning_rate

    def set_max_hidden_units(self, max_hidden_units: int = None):
        """Set maximum hidden units with validation."""
        if max_hidden_units is not None:
            self._validate_positive_integer(max_hidden_units, "max_hidden_units")
        self.max_hidden_units = max_hidden_units

    def set_output_bias(self, output_bias: float = None):
        """Set output bias with validation."""
        if output_bias is not None and not isinstance( output_bias, (int, float, torch.Tensor)):
            raise ValidationError( f"output_bias must be numeric or tensor, got {type(output_bias)}")
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
        logger = ( self.logger if hasattr(self, "logger") and self.logger is not None else Logger)

        logger.trace( "CascadeCorrelationNetwork: set_uuid: Starting to set UUID for CascadeCorrelationNetwork class")
        logger.debug(f"CascadeCorrelationNetwork: set_uuid: Setting UUID to: {uuid}")
        if not hasattr(self, "uuid") or self.uuid is None:
            self.uuid = (uuid, self._generate_uuid())[ uuid is None ]  # Generate a new UUID if none is provided
        else:
            error_msg = f"UUID already set: {self.uuid}. Cannot change UUID after initialization."
            logger.fatal( f"CascadeCorrelationNetwork: set_uuid: Fatal Error: {error_msg}")
            raise ConfigurationError(error_msg)
        logger.debug(f"CascadeCorrelationNetwork: set_uuid: UUID set to: {self.uuid}")
        logger.trace( "CascadeCorrelationNetwork: set_uuid: Completed setting UUID for CascadeCorrelationNetwork class")

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
        self.logger.trace( "CascadeCorrelationNetwork: get_uuid: Starting to get UUID for CascadeCorrelationNetwork class")
        self.logger.debug( f"CascadeCorrelationNetwork: get_uuid: Current UUID: {getattr(self, 'uuid', None)}")

        # Ensure UUID is set:  if not, generate a new one
        if not hasattr(self, "uuid"):
            self.set_uuid()  # Ensure UUID is set if not already
            self.logger.debug( "CascadeCorrelationNetwork: get_uuid: UUID was not set, generated a new one.")

        # Return the UUID
        self.logger.debug( f"CascadeCorrelationNetwork: get_uuid: Returning UUID: {self.uuid}")
        self.logger.trace( "CascadeCorrelationNetwork: get_uuid: Completed getting UUID for CascadeCorrelationNetwork class")
        return self.uuid

    def get_candidate_training_queue_authkey(self):
        return (
            self.candidate_training_queue_authkey
            if hasattr(self, "candidate_training_queue_authkey")
            else None
        )

    def get_candidate_training_queue_address(self):
        return (
            self.candidate_training_queue_address
            if hasattr(self, "candidate_training_queue_address")
            else None
        )

    def get_candidate_training_tasks_queue_timeout(self):
        return ( self.candidate_training_tasks_queue_timeout if hasattr(self, "candidate_training_tasks_queue_timeout") else None)

    def get_candidate_training_shutdown_timeout(self):
        return ( self.candidate_training_shutdown_timeout if hasattr(self, "candidate_training_shutdown_timeout") else None)

    def get_activation_fn(self):
        return self.activation_fn if hasattr(self, "activation_fn") else None

    def get_activation_fn_no_diff(self):
        return ( self.activation_fn_no_diff if hasattr(self, "activation_fn_no_diff") else None)

    def get_candidate_epochs(self):
        return self.candidate_epochs if hasattr(self, "candidate_epochs") else None

    def get_candidate_pool_size(self):
        return ( self.candidate_pool_size if hasattr(self, "candidate_pool_size") else None)

    def get_candidate_unit(self) -> CandidateUnit:
        return self.candidate_unit if hasattr(self, "candidate_unit") else None

    def get_correlation_threshold(self):
        return ( self.correlation_threshold if hasattr(self, "correlation_threshold") else None)

    def get_display_frequency_epoch(self):
        return ( self.display_frequency_epoch if hasattr(self, "display_frequency_epoch") else None)

    def get_display_frequency_units(self):
        return ( self.display_frequency_units if hasattr(self, "display_frequency_units") else None)

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
        return ( self.status_display_frequency if hasattr(self, "status_display_frequency") else None)
