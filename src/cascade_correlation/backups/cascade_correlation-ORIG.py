#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     cascade_correlation.py
# Author:        Paul Calnon
# Version:       0.3.1 (0.7.3)
#
# Date:          2025-06-11
# Last Modified: 2025-09-05
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
import logging
import logging.config
import multiprocessing as mp
import os
import random
import sys
import traceback
import uuid as uuid
from math import inf

# from multiprocessing import current_process, Pool, Manager, Queue
# from multiprocessing import current_process, Pool
from multiprocessing import Queue, current_process
from multiprocessing.managers import BaseManager

# from queue import Queue
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt

# import multiprocess as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from cascade_correlation.candidate_unit.candidate_unit import CandidateUnit
from candidate_unit.candidate_unit import CandidateUnit

# from candidate_unit.candidate_unit import CandidateUnit
from constants.constants import (
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
    _CASCADE_CORRELATION_NETWORK_RANDOM_SEED,
    _CASCADE_CORRELATION_NETWORK_RANDOM_VALUE_SCALE,
    _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE,
    _CASCADE_CORRELATION_NETWORK_STATUS_DISPLAY_FREQUENCY,
    _PROJECT_MODEL_AUTHKEY,
    _PROJECT_MODEL_BASE_MANAGER_ADDRESS,
)
from log_config.log_config import LogConfig
from log_config.logger.logger import Logger
from utils.utils import display_progress

# Add current dir to Python path for imports
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))




# Add current dir to Python path for imports
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


#####################################################################################################################################################################################################
# Class definition for the Cascade Correlation Network
class CascadeCorrelationNetwork:

    # # Add current dir to Python path for imports
    # sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    #################################################################################################################################################################################################
    # Constructor for the Cascade Correlation Network
    def __init__(
        self,
        _CascadeCorrelationNetwork__activation_function_name: str = _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME,
        _CascadeCorrelationNetwork__activation_functions_dict: dict = _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT,
        _CascadeCorrelationNetwork__candidate_display_frequency: int = _CASCADE_CORRELATION_NETWORK_CANDIDATE_DISPLAY_FREQUENCY,
        _CascadeCorrelationNetwork__candidate_epochs: int = _CASCADE_CORRELATION_NETWORK_CANDIDATE_EPOCHS,
        _CascadeCorrelationNetwork__candidate_learning_rate: float = _CASCADE_CORRELATION_NETWORK_CANDIDATE_LEARNING_RATE,
        _CascadeCorrelationNetwork__candidate_pool_size: int = _CASCADE_CORRELATION_NETWORK_CANDIDATE_POOL_SIZE,
        _CascadeCorrelationNetwork__correlation_threshold: float = _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD,
        _CascadeCorrelationNetwork__display_frequency: int = _CASCADE_CORRELATION_NETWORK_DISPLAY_FREQUENCY,
        _CascadeCorrelationNetwork__epoch_display_frequency: int = _CASCADE_CORRELATION_NETWORK_EPOCH_DISPLAY_FREQUENCY,
        _CascadeCorrelationNetwork__epochs_max: int = _CASCADE_CORRELATION_NETWORK_EPOCHS_MAX,
        _CascadeCorrelationNetwork__generate_plots: bool = _CASCADE_CORRELATION_NETWORK_GENERATE_PLOTS,
        _CascadeCorrelationNetwork__input_size: int = _CASCADE_CORRELATION_NETWORK_INPUT_SIZE,
        _CascadeCorrelationNetwork__learning_rate: float = _CASCADE_CORRELATION_NETWORK_LEARNING_RATE,
        _CascadeCorrelationNetwork__log_config: LogConfig = None,
        _CascadeCorrelationNetwork__log_date_format: str = _CASCADE_CORRELATION_NETWORK_LOG_DATE_FORMAT,
        _CascadeCorrelationNetwork__log_file_name: str = _CASCADE_CORRELATION_NETWORK_LOG_FILE_NAME,
        _CascadeCorrelationNetwork__log_file_path: str = _CASCADE_CORRELATION_NETWORK_LOG_FILE_PATH,
        _CascadeCorrelationNetwork__log_format_string: str = _CASCADE_CORRELATION_NETWORK_LOG_FORMATTER_STRING,
        _CascadeCorrelationNetwork__log_level_custom_names_list: list = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_CUSTOM_NAMES_LIST,
        _CascadeCorrelationNetwork__log_level_methods_dict: dict = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_METHODS_DICT,
        _CascadeCorrelationNetwork__log_level_methods_list: list = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_METHODS_LIST,
        _CascadeCorrelationNetwork__log_level_name: str = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME,
        _CascadeCorrelationNetwork__log_level_names_list: list = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAMES_LIST,
        _CascadeCorrelationNetwork__log_level_numbers_dict: dict = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_DICT,
        _CascadeCorrelationNetwork__log_level_numbers_list: list = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_LIST,
        _CascadeCorrelationNetwork__max_hidden_units: int = _CASCADE_CORRELATION_NETWORK_MAX_HIDDEN_UNITS,
        _CascadeCorrelationNetwork__node_correlation_threshold: float = _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD,
        _CascadeCorrelationNetwork__output_epochs: int = _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS,
        _CascadeCorrelationNetwork__output_size: int = _CASCADE_CORRELATION_NETWORK_OUTPUT_SIZE,
        _CascadeCorrelationNetwork__patience: int = _CASCADE_CORRELATION_NETWORK_PATIENCE,
        _CascadeCorrelationNetwork__random_max_value: int = _CASCADE_CORRELATION_NETWORK_RANDOM_MAX_VALUE,
        _CascadeCorrelationNetwork__sequence_max_value: int = _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE,
        _CascadeCorrelationNetwork__random_seed: int = _CASCADE_CORRELATION_NETWORK_RANDOM_SEED,
        _CascadeCorrelationNetwork__random_value_scale: float = _CASCADE_CORRELATION_NETWORK_RANDOM_VALUE_SCALE,
        _CascadeCorrelationNetwork__candidate_training_queue_authkey: str = _PROJECT_MODEL_AUTHKEY,
        _CascadeCorrelationNetwork__candidate_training_queue_address: tuple = _PROJECT_MODEL_BASE_MANAGER_ADDRESS,
        _CascadeCorrelationNetwork__status_display_frequency: int = _CASCADE_CORRELATION_NETWORK_STATUS_DISPLAY_FREQUENCY,
        _CascadeCorrelationNetwork__uuid: uuid.UUID = None,
        **kwargs,
    ):
        Logger.debug(
            "CascadeCorrelationNetwork: __init__: Initializing Cascade Correlation Network with parameters:"
        )
        super().__init__()

        # Initialize the CascadeCorrelationNetwork class logger
        Logger.debug(
            "CascadeCorrelationNetwork: __init__: Initializing the CascadeCorrelationNetwork class logger."
        )
        self.log_file_name = _CascadeCorrelationNetwork__log_file_name or __name__
        self.log_file_path = _CascadeCorrelationNetwork__log_file_path or str(
            os.path.join(os.getcwd(), "logs")
        )
        self.log_level_name = (
            _CascadeCorrelationNetwork__log_level_name
            or _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME
        )

        # Create LogConfig object if not provided
        Logger.debug(
            "CascadeCorrelationNetwork: __init__: Create LogConfig object if not provided"
        )
        self.log_config = _CascadeCorrelationNetwork__log_config or LogConfig(
            _LogConfig__log_config=logging.config,
            _LogConfig__log_file_name=self.log_file_name,
            _LogConfig__log_file_path=self.log_file_path,
            _LogConfig__log_level_name=self.log_level_name,
            _LogConfig__log_date_format=_CascadeCorrelationNetwork__log_date_format,
            _LogConfig__log_format_string=_CascadeCorrelationNetwork__log_format_string,
            _LogConfig__log_level_custom_names_list=_CascadeCorrelationNetwork__log_level_custom_names_list,
            _LogConfig__log_level_methods_dict=_CascadeCorrelationNetwork__log_level_methods_dict,
            _LogConfig__log_level_methods_list=_CascadeCorrelationNetwork__log_level_methods_list,
            _LogConfig__log_level_names_list=_CascadeCorrelationNetwork__log_level_names_list,
            _LogConfig__log_level_numbers_dict=_CascadeCorrelationNetwork__log_level_numbers_dict,
            _LogConfig__log_level_numbers_list=_CascadeCorrelationNetwork__log_level_numbers_list,
        )
        Logger.debug(
            f"CascadeCorrelationNetwork: __init__: LogConfig object: {self.log_config}"
        )
        Logger.debug(
            "CascadeCorrelationNetwork: __init__: Get logger from LogConfig object"
        )
        self.logger = self.log_config.get_logger()
        Logger.debug(
            f"CascadeCorrelationNetwork: __init__: Logger object: {self.logger}"
        )

        # Set log level for Cascade Correlation Network logger
        Logger.debug(
            "CascadeCorrelationNetwork: __init__: Set log level for Cascade Correlation Network logger"
        )
        self.logger.level = self.log_config.get_log_level()
        self.logger.debug(
            f"CascadeCorrelationNetwork: __init__: Initialized Cascade Correlation Network Logger with log file name: {self.log_file_name}, log file path: {self.log_file_path}, log level name: {self.log_level_name}, log level: {self.logger.level}, Handlers: {self.logger.handlers}"
        )

        # Add current dir to Python path for imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        # Initialize randomness for the Cascade Correlation Network
        self.random_seed = (
            _CascadeCorrelationNetwork__random_seed
            or _CASCADE_CORRELATION_NETWORK_RANDOM_SEED
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Random seed: {self.random_seed}"
        )
        self.random_max_value = (
            _CascadeCorrelationNetwork__random_max_value
            or _CASCADE_CORRELATION_NETWORK_RANDOM_MAX_VALUE
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Random max value: {self.random_max_value}"
        )
        self.sequence_max_value = (
            _CascadeCorrelationNetwork__sequence_max_value
            or _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Random sequence max value: {self.sequence_max_value}"
        )
        self._initialize_randomness(
            seed=self.random_seed,
            sequence_max_value=self.sequence_max_value,
            random_max_value=self.random_max_value,
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: __init__: Completed initialization of randomness for the cascade correlation network"
        )

        # Initialize the Cascade Correlation Network
        self.logger.trace(
            "CascadeCorrelationNetwork: __init__: Initializing Cascade Correlation Network with parameters:"
        )
        self.input_size = _CascadeCorrelationNetwork__input_size
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Input size: {self.input_size}"
        )
        self.output_size = _CascadeCorrelationNetwork__output_size
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Output size: {self.output_size}"
        )
        self.candidate_pool_size = _CascadeCorrelationNetwork__candidate_pool_size
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Candidate pool size: {self.candidate_pool_size}"
        )
        self.display_frequency = _CascadeCorrelationNetwork__display_frequency
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Display frequency: {self.display_frequency}"
        )
        self.epoch_display_frequency = (
            _CascadeCorrelationNetwork__epoch_display_frequency
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Display frequency epoch: {self.epoch_display_frequency}"
        )
        self.generate_plots = _CascadeCorrelationNetwork__generate_plots
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Generate plots: {self.generate_plots}"
        )
        self.status_display_frequency = (
            _CascadeCorrelationNetwork__status_display_frequency
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Status display frequency: {self.status_display_frequency}"
        )
        self.candidate_display_frequency = (
            _CascadeCorrelationNetwork__candidate_display_frequency
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Candidate display frequency: {self.candidate_display_frequency}"
        )

        # Initialize the activation function for the Cascade Correlation Network
        self.activation_function_name = (
            _CascadeCorrelationNetwork__activation_function_name
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Activation function name: {self.activation_function_name}"
        )
        self.activation_functions_dict = (
            _CascadeCorrelationNetwork__activation_functions_dict
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Activation functions dictionary: {self.activation_functions_dict}"
        )
        self.activation_fn_no_diff = self.activation_functions_dict.get(
            self.activation_function_name,
            _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Original activation function, no derivative: {self.activation_fn_no_diff}"
        )
        self.activation_fn = self._init_activation_with_derivative(
            self.activation_fn_no_diff
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Activation function: {self.activation_fn}"
        )

        self.learning_rate = _CascadeCorrelationNetwork__learning_rate
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Learning rate: {self.learning_rate}"
        )
        self.candidate_learning_rate = (
            _CascadeCorrelationNetwork__candidate_learning_rate
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Candidate learning rate: {self.candidate_learning_rate}"
        )
        self.max_hidden_units = _CascadeCorrelationNetwork__max_hidden_units
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Max hidden units: {self.max_hidden_units}"
        )
        self.correlation_threshold = _CascadeCorrelationNetwork__correlation_threshold
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Correlation threshold: {self.correlation_threshold}"
        )
        self.patience = _CascadeCorrelationNetwork__patience
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Patience: {self.patience}"
        )
        self.candidate_epochs = _CascadeCorrelationNetwork__candidate_epochs
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Candidate epochs: {self.candidate_epochs}"
        )
        self.epochs_max = _CascadeCorrelationNetwork__epochs_max
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Max epochs: {self.epochs_max}"
        )
        self.output_epochs = _CascadeCorrelationNetwork__output_epochs
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Output epochs: {self.output_epochs}"
        )
        self.random_max_value = _CascadeCorrelationNetwork__random_max_value
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Random max value: {self.random_max_value}"
        )
        self.random_seed = _CascadeCorrelationNetwork__random_seed
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Random seed: {self.random_seed}"
        )
        self.random_value_scale = _CascadeCorrelationNetwork__random_value_scale
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Random value scale: {self.random_value_scale}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: __init__: Setting UUID for Cascade Correlation Network"
        )
        self.set_uuid(
            _CascadeCorrelationNetwork__uuid
        )  # Set UUID for the CascadeCorrelationNetwork class
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: UUID set to: {self.uuid}"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: __init__: self.uuid = {self.get_uuid()}"
        )

        # Initialize display progress function with cascade correlation network display frequency
        self.logger.trace(
            "CascadeCorrelationNetwork: __init__: Initializing display progress function with cascade correlation network display frequency"
        )
        self._network_display_progress = display_progress(
            display_frequency=self.epoch_display_frequency
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Network display progress function initialized with display frequency: {self.epoch_display_frequency}, _network_display_progress = {self._network_display_progress}"
        )

        # Initialize status display progress function with cascade correlation training status display frequency
        self.logger.trace(
            "CascadeCorrelationNetwork: __init__: Initializing status display progress function with cascade correlation training status display frequency"
        )
        self._status_display_progress = display_progress(
            display_frequency=self.status_display_frequency
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Status display progress function initialized with status display frequency: {self.status_display_frequency}, _status_display_progress = {self._status_display_progress}"
        )

        # Initialize candidate display progress function with cascade correlation training candidate display frequency
        self.logger.trace(
            "CascadeCorrelationNetwork: __init__: Initializing candidate display progress function with cascade correlation training candidate display frequency"
        )
        self._candidate_display_progress = display_progress(
            display_frequency=self.candidate_display_frequency
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Candidate display progress function initialized with display frequency: {self.candidate_display_frequency}, _candidate_display_progress = {self._candidate_display_progress}"
        )

        # Initialize multiprocessing manager config values for candidate training queue
        self.candidate_training_queue_authkey = (
            _CascadeCorrelationNetwork__candidate_training_queue_authkey
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Candidate training queue authkey: {self.candidate_training_queue_authkey}"
        )
        self.candidate_training_queue_address = (
            _CascadeCorrelationNetwork__candidate_training_queue_address
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: __init__: Candidate training queue address: {self.candidate_training_queue_address}"
        )

        # Initialize network Model Parameters
        self.logger.trace(
            "CascadeCorrelationNetwork: __init__: Initializing network Model Parameters"
        )
        self.hidden_units = []
        self.output_weights = (
            torch.randn(
                _CascadeCorrelationNetwork__input_size,
                _CascadeCorrelationNetwork__output_size,
                requires_grad=True,
            )
            * self.random_value_scale
        )
        self.output_bias = (
            torch.randn(_CascadeCorrelationNetwork__output_size, requires_grad=True)
            * self.random_value_scale
        )
        self.history = {
            "train_loss": [],
            "value_loss": [],
            "train_accuracy": [],
            "value_accuracy": [],
            "hidden_units_added": [],
        }
        self.logger.trace(
            "CascadeCorrelationNetwork: __init__: Completed initialization of network Model Parameters"
        )
        self.logger.info(
            "CascadeCorrelationNetwork: __init__: Cascade Correlation Network initialized with parameters"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: __init__: Cascade Correlation Network initialization complete"
        )

    #################################################################################################################################################################################################
    # Define init methods called by the __init__ constructor method.
    #################################################################################################################################################################################################

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
        self.logger.trace(
            "CascadeCorrelationNetwork: _initialize_randomness: Initializing randomness for the cascade correlation network"
        )
        seed = seed or _CASCADE_CORRELATION_NETWORK_RANDOM_SEED
        self.logger.verbose(
            f"CascadeCorrelationNetwork: _initialize_randomness: Random seed set to: {seed}"
        )
        sequence_max_value = (
            sequence_max_value or _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: _initialize_randomness: Random sequence max value set to: {sequence_max_value}"
        )
        random_max_value = (
            random_max_value or _CASCADE_CORRELATION_NETWORK_RANDOM_MAX_VALUE
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: _initialize_randomness: Random max value set to: {random_max_value}"
        )
        self._seed_random_generator(
            seed=seed,
            max_value=sequence_max_value,
            seeder=np.random.seed,
            generator=np.random.randint,
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: _initialize_randomness: Completed initialization of numpy random generator with seed and sequence for the cascade correlation network"
        )
        self._seed_random_generator(
            seed=seed,
            max_value=sequence_max_value,
            seeder=random.seed,
            generator=random.randint,
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: _initialize_randomness: Completed initialization of random random generator with seed and sequence for the cascade correlation network"
        )
        self._seed_random_generator(
            seed=seed,
            max_value=sequence_max_value,
            seeder=torch.manual_seed,
            generator=lambda min, max: torch.randint(min, max, ()),
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: _initialize_randomness: Completed initialization of torch random generator with seed and sequence for the cascade correlation network"
        )
        self._seed_random_generator(
            seed=seed,
            max_value=sequence_max_value,
            seeder=self._seed_hash,
            generator=None,
        )

        if torch.cuda.is_available():
            self.logger.trace(
                "CascadeCorrelationNetwork: _initialize_randomness: CUDA is available, seeding CUDA random generator."
            )
            self._seed_random_generator(
                seed=seed,
                max_value=sequence_max_value,
                seeder=torch.cuda.manual_seed,
                generator=lambda min, max: torch.rand(1, device="cuda"),
            )
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

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
        self.logger.trace(
            "CascadeCorrelationNetwork: _seed_random_generator: Seeding random module with seed and max value."
        )
        if seeder is None:
            self.logger.verbose(
                "CascadeCorrelationNetwork: _seed_random_generator: No seeder function provided, skipping seeding of random generator."
            )
            return

        seeder(seed)
        self.logger.trace(
            "CascadeCorrelationNetwork: _seed_random_generator: Random seed set for random module."
        )

        # random_sequence = generator.randint(0, max_value)
        if generator is None:
            self.logger.verbose(
                "CascadeCorrelationNetwork: _seed_random_generator: No generator function provided, skipping random sequence generation and rolling."
            )
            return

        # random_sequence = generator(0, max_value)
        # trunk-ignore(bandit/B311)
        random_sequence = random.randint(0, max_value)
        self.logger.verbose(
            f"CascadeCorrelationNetwork: _seed_random_generator: Random sequence number rolled to: {random_sequence}"
        )

        # TODO:  Enable CUDA random generator seeding and rolling when needed
        #     self._seed_random_generator(seed=seed, max_value=sequence_max_value, seeder=torch.cuda.manual_seed, generator=lambda min, max: torch.rand(1, device='cuda'))
        # File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 392, in _roll_sequence_number
        #     discard = [generator(0, max_value) for _ in range(sequence)]
        #                                                 ^^^^^^^^^^^^^^^
        # TypeError: only integer tensors of a single element can be converted to an index

        self._roll_sequence_number(
            sequence=random_sequence, max_value=max_value, generator=generator
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: _seed_random_generator: Completed initialization of random generator with seed and sequence for the cascade correlation network"
        )

    def _roll_sequence_number(
        self, sequence: int = None, max_value: int = None, generator: callable = None
    ) -> None:
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
        self.logger.trace(
            "CascadeCorrelationNetwork: _roll_sequence_number: Rolling sequence number."
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: _roll_sequence_number: Rolling sequence number to: {sequence} with max value: {max_value} using generator: {generator}"
        )
        if generator is not None:
            discard = [generator(0, max_value) for _ in range(sequence)]
            self.logger.verbose(
                f"CascadeCorrelationNetwork: _roll_sequence_number: Discarded {len(discard)} random numbers to roll to sequence number: {sequence}"
            )
            self.logger.verbose(
                f"CascadeCorrelationNetwork: _roll_sequence_number: Random Generator rolled for sequence number: {sequence}"
            )
        self.logger.trace(
            "CascadeCorrelationNetwork: _roll_sequence_number: Completed rolling of sequence number."
        )

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
        self.logger.trace(
            "CascadeCorrelationNetwork: _init_activation_with_derivative: Validating activation function"
        )
        activation_fn = (
            activation_fn,
            _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,
        )[activation_fn is None]
        self.logger.debug(
            f"CascadeCorrelationNetwork: _init_activation_with_derivative: Using activation function: {activation_fn}"
        )

        # Wrapping the activation function with its derivative
        self.logger.trace(
            "CascadeCorrelationNetwork: _init_activation_with_derivative: Wrapping activation function to provide its derivative."
        )

        def wrapped_activation(x, derivative: bool = False):
            if derivative:
                if activation_fn in [
                    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_TANH,
                    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_TANH,
                ]:  # For tanh, derivative is 1 - tanh^2(x)
                    return 1.0 - activation_fn(x) ** 2
                elif activation_fn in [
                    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_SIGMOID,
                    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_SIGMOID,
                ]:  # For sigmoid, derivative is sigmoid(x) * (1 - sigmoid(x))
                    y = activation_fn(x)
                    return y * (1.0 - y)
                elif activation_fn in [
                    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_RELU,
                    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_RELU,
                ]:  # For ReLU, derivative is 1 for x > 0, 0 otherwise
                    return (x > 0).float()
                else:  # Numerical approximation for other functions
                    eps = 1e-6
                    return (activation_fn(x + eps) - activation_fn(x - eps)) / (2 * eps)
            else:
                return activation_fn(x)

        self.logger.verbose(
            f"CascadeCorrelationNetwork: _init_activation_with_derivative: Returning wrapped activation function: {wrapped_activation}."
        )

        # Return the wrapped activation function
        self.logger.trace(
            "CascadeCorrelationNetwork: _init_activation_with_derivative: Completed wrapping of activation function."
        )
        return wrapped_activation

    #################################################################################################################################################################################################
    # Public Method that Performs a Forward pass through the network
    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Description:
            This method performs a forward pass through the network.
        Args:
            x: Input tensor
        Note:
            This method only performs a forward pass through the output layer of the network.
        Raises:
            ValueError: If the input tensor is None.
        Returns:
            Network output: torch.Tensor
        """
        # Start with the input features
        self.logger.trace(
            "CascadeCorrelationNetwork: forward: Starting forward pass through the network."
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: forward: Starting forward pass with input shape: {x.shape if x is not None else 'None'}"
        )
        features = x
        self.logger.debug(
            f"CascadeCorrelationNetwork: forward: Input shape: {features.shape}"
        )

        # Pass through each hidden unit
        hidden_outputs = []
        for i, unit in enumerate(self.hidden_units):

            # Concatenate all previous outputs with the input
            unit_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x

            # Get output from this unit
            unit_output = unit["activation_fn"](
                torch.sum(unit_input * unit["weights"], dim=1) + unit["bias"]
            ).unsqueeze(1)
            hidden_outputs.append(unit_output)
            if self._status_display_progress(i):
                self.logger.info(
                    f"CascadeCorrelationNetwork: forward: Hidden unit {i + 1} output shape: {unit_output.shape}"
                )
            self.logger.debug(
                f"CascadeCorrelationNetwork: forward: Hidden unit {i + 1} output shape: {unit_output.shape}"
            )

        # Prepare input for the output layer
        output_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
        self.logger.verbose(
            f"CascadeCorrelationNetwork: forward: Output input shape: {output_input.shape}, Value: {output_input}"
        )

        # Output layer (linear combination)
        output = torch.matmul(output_input, self.output_weights) + self.output_bias
        self.logger.debug(
            f"CascadeCorrelationNetwork: forward: Output shape: {output.shape}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: forward: Completed forward pass through the network."
        )
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
        self.logger.trace(
            "CascadeCorrelationNetwork: train_output_layer: Starting training of the output layer."
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_output_layer: Input shape: {x.shape if x is not None else 'None'}, Target shape: {y.shape if y is not None else 'None'}, Epochs: {epochs}"
        )
        epochs = (epochs, _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS)[epochs is None]
        if x is None or y is None:
            raise ValueError(
                "CascadeCorrelationNetwork: train_output_layer: Input (x) and target (y) tensors must be provided for training the output layer."
            )

        # Define loss function and optimizer
        criterion = nn.MSELoss()

        # Create a simple linear layer for the output
        input_size = x.shape[1]
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_output_layer: Input size for output layer: {input_size}, Output size: {self.output_size}"
        )
        if self.hidden_units:
            input_size += len(self.hidden_units)
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_output_layer: Adjusted input size for output layer with hidden units: {input_size}"
        )

        # Create a temporary linear layer with the same weights as our current output layer
        output_layer = nn.Linear(input_size, self.output_size)
        with torch.no_grad():
            output_layer.weight.copy_(
                self.output_weights.t()
            )  # Transpose because nn.Linear expects (out_features, in_features)
            self.logger.debug(
                f"CascadeCorrelationNetwork: train_output_layer: Output weights shape: {self.output_weights.shape}, Transposed weights shape: {output_layer.weight.shape}"
            )
            output_layer.bias.copy_(self.output_bias)
            self.logger.debug(
                f"CascadeCorrelationNetwork: train_output_layer: Output bias shape: {self.output_bias.shape}, Bias: {output_layer.bias}"
            )

        # Use this layer for optimization
        optimizer = optim.Adam(output_layer.parameters(), lr=self.learning_rate)
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_output_layer: Learning Rate: {self.learning_rate}, Optimizer:\n{optimizer}"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_output_layer: Output layer initialized with weights shape: {output_layer.weight.shape}, Bias shape: {output_layer.bias.shape}"
        )

        # Output Layer Training loop
        for epoch in range(epochs):

            # Get the input for the output layer (original input + hidden unit outputs)
            hidden_outputs = []
            for unit in self.hidden_units:
                unit_input = (
                    torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
                )
                unit_output = unit["activation_fn"](
                    torch.sum(unit_input * unit["weights"], dim=1) + unit["bias"]
                ).unsqueeze(1)
                hidden_outputs.append(unit_output)

            # Calculate Loss by Concatenating inputs with outputs from existing hidden units
            output_input = (
                torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
            )
            output = output_layer(output_input)
            self.logger.debug(
                f"CascadeCorrelationNetwork: train_output_layer: Output shape: {output.shape}, Output Input shape: {output_input.shape}"
            )
            self.logger.debug(
                f"CascadeCorrelationNetwork: train_output_layer: Output:\n{output}"
            )
            self.logger.debug(
                f"CascadeCorrelationNetwork: train_output_layer: Target shape: {y.shape}, Target:\n{y}"
            )
            loss = criterion(output, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self._network_display_progress(epoch):
                self.logger.info(
                    f"CascadeCorrelationNetwork: train_output_layer: Output Layer Training - Epoch {epoch + 1}, Loss: {loss.item():.6f}"
                )
            self.logger.debug(
                f"CascadeCorrelationNetwork: train_output_layer: Output Layer Training - Epoch {epoch + 1}, Loss: {loss.item():.6f}"
            )

        # Update our model's weights with the trained values
        with torch.no_grad():
            self.output_weights = output_layer.weight.t().clone()  # Transpose back
            self.logger.debug(
                f"CascadeCorrelationNetwork: train_output_layer: Output weights shape: {self.output_weights.shape}, Weights:\n{self.output_weights}"
            )
            self.output_bias = output_layer.bias.clone()
            self.logger.debug(
                f"CascadeCorrelationNetwork: train_output_layer: Output bias shape: {self.output_bias.shape}, Bias:\n{self.output_bias}"
            )

        # Final loss
        with torch.no_grad():
            output = self.forward(x)
            self.logger.debug(
                f"CascadeCorrelationNetwork: train_output_layer: Final output shape: {output.shape}, Output: {output}"
            )
            final_loss = criterion(output, y).item()
            self.logger.info(
                f"CascadeCorrelationNetwork: train_output_layer: Final output layer training loss: {final_loss:.6f}"
            )

        self.logger.trace(
            "CascadeCorrelationNetwork: train_output_layer: Completed training of the output layer."
        )
        return final_loss

    ##################################################################################################################################################################################################
    # Define a method to train a candidate unit
    def _get_serializable_activation_fn(self):
        """
        Description:
            Get a serializable version of the activation function for multiprocessing.
        Returns:
            Serializable activation function
        """
        self.logger.trace(
            "CascadeCorrelationNetwork: _get_serializable_activation_fn: Retrieving serializable activation function."
        )
        self.activation_functions_dict.get(
            self.activation_function_name,
            _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: _get_serializable_activation_fn: Serializable activation function: {self.activation_fn_no_diff}"
        )
        return self.activation_fn_no_diff

    @staticmethod
    def _get_activation_function(
        activation_function_name: str = None, activation_functions_dict: dict = None
    ) -> callable:
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
            activation_functions_dict = (
                _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT
            )
        if activation_function_name is None:
            activation_function_name = (
                _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME
            )
        return activation_functions_dict.get(
            activation_function_name,
            _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,
        )

    @staticmethod
    # def train_candidate_worker(task_data_input=None, args=None, parallel=True):
    def train_candidate_worker(
        task_data_input: tuple = None,
        task_queue: Queue = None,
        done_queue: Queue = None,
        parallel=True,
    ):
        logger = Logger
        logger.info(
            "CascadeCorrelationNetwork: train_candidate_worker: Starting training of Candidate Units in Pool."
        )

        # Get task data for process worker
        task_data = None
        try:
            if parallel:
                worker_id = mp.current_process().pid
                worker_uuid = str(uuid.uuid4())
                candidate_uuid = str(uuid.uuid4())
                logger.debug(
                    f"CascadeCorrelationNetwork: train_candidate_worker: Training Candidate Units in Pool in Parallel: Multiprocess Worker ID: {worker_id}, Multiprocess Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
                )
                task_data = (
                    task_data_input
                    if task_data_input is not None
                    and isinstance(task_data_input, tuple)
                    else (
                        task_queue.get()
                        if task_queue is not None and isinstance(task_queue, Queue)
                        else None
                    )
                )
                if task_data is None:
                    logger.error(
                        f"CascadeCorrelationNetwork: train_candidate_worker: No valid task data input or task queue provided for parallel processing: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
                    )
                    raise ValueError(
                        f"CascadeCorrelationNetwork: train_candidate_worker: No valid task data input or task queue provided for parallel processing: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
                    )
                logger.info(
                    f"CascadeCorrelationNetwork: train_candidate_worker: Retrieved task data for parallel processing: Type: {type(task_data)}, Length: {len(task_data)}, Value: {task_data}"
                )
            else:
                worker_id = 0
                worker_uuid = "None"
                candidate_uuid = "None"
                logger.debug(
                    f"CascadeCorrelationNetwork: train_candidate_worker: Training Candidate Units in Pool Sequentially: Sequential Worker ID: {worker_id}, Sequential Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
                )
                if isinstance(task_data_input, tuple):
                    task_data = task_data_input
                    logger.debug(
                        f"CascadeCorrelationNetwork: train_candidate_worker: Received task data input as tuple: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
                    )
                elif task_queue is not None:
                    task_data = task_queue.get()
                    logger.debug(
                        f"CascadeCorrelationNetwork: train_candidate_worker: Retrieved task data from task queue: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
                    )
                else:
                    logger.error(
                        f"CascadeCorrelationNetwork: train_candidate_worker: No valid task data input or task queue provided for sequential processing: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
                    )
                    raise ValueError(
                        f"CascadeCorrelationNetwork: train_candidate_worker: No valid task data input or task queue provided for sequential processing: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
                    )
            logger.verbose(
                f"CascadeCorrelationNetwork: train_candidate_worker: Task data: length: {len(task_data) if task_data is not None else 'None'}, Type: {type(task_data) if task_data is not None else 'None'}, Content:\n{task_data if task_data is not None else 'None'}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
            )
        except Exception as e:
            logger.error(
                f"CascadeCorrelationNetwork: train_candidate_worker: Error retrieving task data from queue or input: {e}"
            )
            raise e

        try:
            logger.debug(
                f"CascadeCorrelationNetwork: train_candidate_worker: Attempting to Unpack Task data, Candidate data, and Training inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
            )
            logger.verbose(
                f"CascadeCorrelationNetwork: train_candidate_worker: Task data: length: {len(task_data)}, Type: {type(task_data)}, Content:\n{task_data}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
            )
            candidate_index, candidate_data, training_inputs = (
                task_data  # Unpack training task data
            )
            logger.debug(
                f"CascadeCorrelationNetwork: train_candidate_worker: Successfully Unpacked Task data: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
            )
            logger.verbose(
                f"CascadeCorrelationNetwork: train_candidate_worker: Candidate Index: {candidate_index}, Type: {type(candidate_index)}, Value: {candidate_index}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
            )
            logger.verbose(
                f"CascadeCorrelationNetwork: train_candidate_worker: Candidate Inputs: Length: {len(training_inputs)}, Type: {type(training_inputs)}, Content:\n{training_inputs}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
            )
            logger.verbose(
                f"CascadeCorrelationNetwork: train_candidate_worker: Candidate Data: length: {len(candidate_data)}, Type: {type(candidate_data)}, Content:\n{candidate_data}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
            )
            # logger.verbose(f"CascadeCorrelationNetwork: train_candidate_worker: Candidate Data: Element 0: Length: {len(candidate_data[0])}, Type: {type(candidate_data[0])}, Content:\n{candidate_data[0]}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}")

            logger.debug(
                f"CascadeCorrelationNetwork: train_candidate_worker: Attempting to unpack Training inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
            )
            logger.verbose(
                f"CascadeCorrelationNetwork: train_candidate_worker: Training inputs: length: {len(training_inputs)}, Type: {type(training_inputs)}, Content:\n{training_inputs}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
            )
            # logger.verbose(f"CascadeCorrelationNetwork: train_candidate_worker: Training inputs: Element 0: Length: {len(training_inputs[0])}, Type: {type(training_inputs[0])}, Content:\n{training_inputs[0]}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}")
            x, epochs, y, residual_error, learning_rate, display_frequency = (
                training_inputs  # unpack training inputs
            )
            logger.debug(
                f"CascadeCorrelationNetwork: train_candidate_worker: Successfully Unpacked Training inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}."
            )
            logger.verbose(
                f"CascadeCorrelationNetwork: train_candidate_worker: Training inputs: x shape: {x.shape}, epochs: {epochs}, y shape: {y.shape}, residual_error shape: {residual_error.shape}, learning_rate: {learning_rate}, display_frequency: {display_frequency}"
            )

            logger.verbose(
                f"CascadeCorrelationNetwork: train_candidate_worker: Candidate data: length: {len(candidate_data)}, Type: {type(candidate_data)}, Content:\n{candidate_data}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}"
            )
            # logger.verbose(f"CascadeCorrelationNetwork: train_candidate_worker: Candidate data: Element 0: Length: {len(candidate_data[0])}, Type: {type(candidate_data[0])}, Content:\n{candidate_data[0]}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}")
            (
                id,
                input_size,
                activation_function_name,
                random_value_scale,
                candidate_uuid,
                random_seed,
                random_value_max,
                sequence_max_value,
            ) = candidate_data  # unpack candidate data
            logger.debug(
                f"CascadeCorrelationNetwork: train_candidate_worker: Successfully Unpacked Candidate Data: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}."
            )
            logger.verbose(
                f"CascadeCorrelationNetwork: train_candidate_worker: Candidate data unpacked: Candidate ID: {id}, Input Size: {input_size}, Activation Function Name: {activation_function_name}, Random Value Scale: {random_value_scale}, Candidate UUID: {candidate_uuid}, Random Seed: {random_seed}, Random Value Max: {random_value_max}, Sequence Max Value: {sequence_max_value}: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate UUID: {candidate_uuid}."
            )

            logger.debug(
                f"CascadeCorrelationNetwork: train_candidate_worker: Unpacked Task data, Candidate data, and Training inputs: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Training Inputs: x shape: {x.shape}, epochs: {epochs}, y shape: {y.shape}, residual_error shape: {residual_error.shape}, learning_rate: {learning_rate}, display_frequency: {display_frequency}"
            )

            activation_fn = CascadeCorrelationNetwork._get_activation_function(
                activation_function_name
            )
            logger.debug(
                f"CascadeCorrelationNetwork: train_candidate_worker: Retrieved wrapped activation function: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Activation Function: Name: {activation_function_name}, Function: {activation_fn}"
            )

            try:
                # from candidate_unit.candidate_unit import CandidateUnit
                # from cascade_correlation.candidate_unit.candidate_unit import CandidateUnit
                from candidate_unit.candidate_unit import CandidateUnit

                logger.debug(
                    "CascadeCorrelationNetwork: train_candidate_worker: Imported CandidateUnit from candidate_unit.candidate_unit."
                )
            except ImportError as e:
                logger.error(
                    f"CascadeCorrelationNetwork: train_candidate_worker: Failed to import candidate unit.  Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Exception Raised\n{e}"
                )
                # return (candidate_index, None, None, 0.0)
                e.add_note(
                    f"CascadeCorrelationNetwork: train_candidate_worker: Failed to import candidate unit.  Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}"
                )
                raise ValueError from e

            # Instantiate a CandidateUnit
            logger.debug(
                f"CascadeCorrelationNetwork: train_candidate_worker: Instantiating CandidateUnit object: Candidate ID: {id}, Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Input Size: {input_size}, Activation Function: {activation_function_name}, Random Seed: {random_seed}, Random Value Max: {random_value_max}, Random Value Scale: {random_value_scale}"
            )
            candidate = CandidateUnit(
                CandidateUnit__input_size=input_size,
                CandidateUnit__activation_function=activation_fn,
                CandidateUnit__display_frequency=display_frequency,
                CandidateUnit__epochs_max=epochs,
                CandidateUnit__learning_rate=learning_rate,
                CandidateUnit__log_level_name="INFO",
                CandidateUnit__random_seed=random_seed,
                CandidateUnit__sequence_max_value=sequence_max_value,
                CandidateUnit__random_value_max=random_value_max,
                CandidateUnit__random_value_scale=random_value_scale,
                CandidateUnit__uuid=candidate_uuid,
            )
            logger.debug(
                f"CascadeCorrelationNetwork: train_candidate_worker: Completed Instantiating CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate.get_uuid()}"
            )

            # Train the candidate unit
            (correlation, min_norm_error) = candidate.train(
                x=x,
                epochs=epochs,
                residual_error=residual_error,
                learning_rate=learning_rate,
                display_frequency=display_frequency,
            )
            # logger.debug(f"CascadeCorrelationNetwork: train_candidate_worker: Trained CandidateUnit object Returned Correlation: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Correlation: Type: {type(correlation)}, Value: {float(correlation):.6f}")
            logger.info(
                f"CascadeCorrelationNetwork: train_candidate_worker: Completed Training CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Correlation: Type: {type(correlation)}, Value: {float(correlation):.6f}"
            )
            # return (candidate_index, candidate_uuid, candidate, correlation)
            logger.debug(
                f"CascadeCorrelationNetwork: train_candidate_worker: Clearing Display Progress and Display Status for Candidate Unit: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}"
            )
            candidate.clear_display_progress()  # Clear display progress for candidate unit, to avoid issues with multiprocessing--nested functions are not pickleable
            logger.debug(
                f"CascadeCorrelationNetwork: train_candidate_worker: Cleared Display Progress for Candidate Unit: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}"
            )
            candidate.clear_display_status()  # Clear display status for candidate unit, to avoid issues with multiprocessing--nested functions are not pickleable
            logger.debug(
                f"CascadeCorrelationNetwork: train_candidate_worker: Cleared Display Status for Candidate Unit: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}"
            )
            logger.verbose(
                f"CascadeCorrelationNetwork: train_candidate_worker: Returning from Candidate Unit Training: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Correlation: Type: {type(correlation)}, Value: {float(correlation):.6f}, Candidate Object: {candidate}"
            )
            return (candidate_index, candidate_uuid, correlation, candidate)

        except Exception as e:
            logger.error(
                f"CascadeCorrelationNetwork: train_candidate_worker: Caught Exception while training CandidateUnit object: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}, Error during candidate training:\nException:\n{e}"
            )
            # import traceback
            logger.error(
                f"CascadeCorrelationNetwork: train_candidate_worker: Error during Candidate Training: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}\nTraceback:\n{traceback.format_exc()}"
            )
            # return (candidate_index, candidate_uuid if 'candidate_uuid' in locals() else None, None, 0.0)
            e.add_note(
                f"CascadeCorrelationNetwork: train_candidate_worker: Error during CandidateUnit object Training: train_candidate_worker: Worker ID: {worker_id}, Worker UUID: {worker_uuid}, Candidate Index: {candidate_index}, Candidate UUID: {candidate_uuid}."
            )
            raise ValueError from e

    ##################################################################################################################################################################################################
    # Public Method to update candidate units based on the residual error
    def train_candidates(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        residual_error: torch.Tensor,
    ) -> List[CandidateUnit]:  # sourcery skip: extract-duplicate-method, extract-method
        """
        Description:
            Train a pool of candidate units based on the residual error from the network, and select the best one.
        Args:
            x: Input tensor
            residual_error: Residual error from the network
        Note:
            This method trains a pool of candidate units and selects the one with the highest correlation to the residual error.
            The selected candidate unit can then be added to the network as a new hidden unit.
        Raises:
            ValueError: If the input tensor or residual error tensor is None.
        Returns:
            List of trained candidate units
        """
        # Prepare input for candidates (includes outputs from existing hidden units)
        self.logger.trace(
            "CascadeCorrelationNetwork: train_candidates: Starting training of candidate units."
        )
        hidden_outputs = []
        for unit in self.hidden_units:
            unit_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x

            # Concatenate input with outputs from existing hidden units
            unit_output = unit["activation_fn"](
                torch.sum(unit_input * unit["weights"], dim=1) + unit["bias"]
            ).unsqueeze(1)
            hidden_outputs.append(unit_output)

        self.logger.info(
            f"CascadeCorrelationNetwork: train_candidates: LOOK AT ME!!  Number of Hidden Nodes:  {len(hidden_outputs)}, Hidden Outputs shape:\n{[h.shape for h in hidden_outputs]}"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Hidden outputs: {hidden_outputs}"
        )

        # Prepare candidate input
        candidate_input = (
            torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
        )
        input_size = candidate_input.shape[1]
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Candidate input shape: {candidate_input.shape}, Input size: {input_size}"
        )

        # Create training inputs for workers
        training_inputs = (
            candidate_input,
            self.candidate_epochs,
            y,
            residual_error,
            self.candidate_learning_rate,
            self.candidate_display_frequency,
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Training inputs: {training_inputs}"
        )
        # activation_name = self._get_serializable_activation_fn()
        activation_name = self.activation_function_name
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Serializable activation function: {activation_name}"
        )

        candidate_uuids = [str(uuid.uuid4()) for _ in range(self.candidate_pool_size)]
        self.logger.verbose(
            f"CascadeCorrelationNetwork: train_candidates: Generated UUID List for candidates: Length: {len(candidate_uuids)},\nValues:\n{candidate_uuids}"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: UUID List for candidates: Length: {len(candidate_uuids)}"
        )
        candidate_seeds = [
            # trunk-ignore(bandit/B311)
            random.randint(0, self.random_max_value)
            for _ in range(self.candidate_pool_size)
        ]  # trunk-ignore(bandit/B311)
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Generated Random Seed List for candidates: Length: {len(candidate_seeds)}, Values:\n{candidate_seeds}"
        )
        candidate_data = [
            (
                i,
                input_size,
                activation_name,
                self.random_value_scale,
                candidate_uuids[i],
                candidate_seeds[i],
                self.random_max_value,
                self.sequence_max_value,
            )
            for i in range(self.candidate_pool_size)
        ]
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Created candidate data list: Length: {len(candidate_data)}, Value:\n{candidate_data}"
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: train_candidates: Created candidate data list, Element 0: Length: {len(candidate_data[0])}, Value:\n{candidate_data[0]}"
        )

        # tasks = [(i, candidate_data, training_inputs) for i in range(self.candidate_pool_size)]
        tasks = [
            (i, candidate_data[i], training_inputs)
            for i in range(self.candidate_pool_size)
        ]
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Created training tasks for candidates: Length {len(tasks)}, Value:\n{tasks}"
        )

        # Determine number of available CPU cores
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: CPU count: {os.cpu_count()}"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Affinity CPU count: {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else inf}"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Candidate pool size: {self.candidate_pool_size}"
        )
        sched_affinity = (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else inf
        )
        cpu_cores_available = min(
            self.candidate_pool_size, sched_affinity, mp.cpu_count(), os.cpu_count()
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Available CPU cores: {cpu_cores_available}"
        )

        # Use all available CPU cores minus one for training candidates in parallel; leave one core free to keep the system responsive
        process_count = max(1, cpu_cores_available - 1)
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Candidates training pool size: {process_count}"
        )

        # Train candidates in parallel using multiprocessing
        self.logger.info(
            f"CascadeCorrelationNetwork: train_candidates: Starting training of {len(tasks)} candidate units using {process_count} processes."
        )
        start_time = datetime.datetime.now()
        results = []

        # process_count = 1  # For debugging, force single process mode
        if process_count == 1:
            # Single process fallback
            self.logger.debug(
                "CascadeCorrelationNetwork: train_candidates: Single process mode; training candidates sequentially."
            )
            self.logger.debug(
                f"CascadeCorrelationNetwork: train_candidates: Training {len(tasks)} candidates sequentially."
            )
            self.logger.debug(
                f"CascadeCorrelationNetwork: train_candidates: Candidate uuid and tasks: Length: {len(list(zip(candidate_uuids, tasks, strict=False)))}, Values:\n{list(zip(candidate_uuids, tasks, strict=False))}"
            )
            for id, (can_uuid, task) in enumerate(
                zip(candidate_uuids, tasks, strict=False)
            ):
                candidate_start_time = datetime.datetime.now()
                self.logger.debug(
                    f"CascadeCorrelationNetwork: train_candidates: Preparing to train candidate {id}, UUID: {can_uuid} ({task[1][3]}), Task Num: {task[0] + 1}, with input size: {input_size}."
                )
                result = self.train_candidate_worker(task, parallel=False)
                self.logger.debug(
                    f"CascadeCorrelationNetwork: train_candidates: Training candidate: ID number: {id}, UUID: {can_uuid}, Task Num: {task[0] + 1}, with input size: {input_size}."
                )
                candidate_end_time = datetime.datetime.now()
                self.logger.debug(
                    f"CascadeCorrelationNetwork: train_candidates: Timing for Candidate training: ID number: {id}: duration: {candidate_end_time - candidate_start_time}"
                )
                self.logger.debug(
                    f"CascadeCorrelationNetwork: train_candidates: Candidate training result: {result}"
                )
                results.append(result)
                self.logger.debug(
                    f"CascadeCorrelationNetwork: train_candidates: Collected result for candidate {can_uuid}, Num: {task[0] + 1}. Results size: {len(results)}"
                )
        else:
            # Multiprocessing happy dance
            self.logger.debug(
                "CascadeCorrelationNetwork: train_candidates: Multiprocessing mode; training candidates in parallel."
            )
            self.logger.debug(
                f"CascadeCorrelationNetwork: train_candidates: Training {len(tasks)} candidates in parallel using {process_count} processes."
            )
            self.logger.debug(
                f"CascadeCorrelationNetwork: train_candidates: Candidates training pool size: {process_count}"
            )

            # Set start method to 'spawn' to avoid issues with forking in some environments (e.g., Jupyter, certain OS)
            # # TODO: consider this option first -- this works
            # mp.set_start_method('spawn', force=True)  # Use 'spawn' to start fresh processes
            # self.logger.debug("CascadeCorrelationNetwork: train_candidates: Multiprocessing start method set to 'spawn'.")
            # # with mp.Pool(processes=process_count) as candidates_training_pool:
            # with Pool(processes=process_count) as candidates_training_pool:
            #     self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates training pool has been created: {candidates_training_pool}.")
            #     results = candidates_training_pool.map(CascadeCorrelationNetwork.train_candidate_worker, tasks)
            #     self.logger.debug("CascadeCorrelationNetwork: train_candidates: Candidates training pool has completed all tasks.")
            #     # candidates_training_pool.join()
            #     # candidates_training_pool.close()
            #     # self.logger.debug("CascadeCorrelationNetwork: train_candidates: Candidates training pool has been joined and closed.")
            # self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates training pool completed and closed: collected {len(results)} results.")

            # # TODO: consider this option second
            # candidate_training_context = mp.get_context(method='spawn')  # Alternative context is 'spawn'
            # self.logger.debug("CascadeCorrelationNetwork: train_candidates: Multiprocessing context set to 'spawn'.")
            # with candidate_training_context.Pool(processes=process_count) as candidates_training_pool:
            #     self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates training pool has been created: {candidates_training_pool}.")
            #     results = candidates_training_pool.map(CascadeCorrelationNetwork.train_candidate_worker, tasks)
            #     self.logger.debug("CascadeCorrelationNetwork: train_candidates: Candidates training pool has completed all tasks.")
            #     # candidates_training_pool.join()
            #     # candidates_training_pool.close()
            #     # self.logger.debug("CascadeCorrelationNetwork: train_candidates: Candidates training pool has been joined and closed.")
            # self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates training pool completed and closed: collected {len(results)} results.")

            # # TODO: consider this option third
            # candidate_training_context = mp.get_context(method='forkserver')  # Alternative context is 'forkserver'
            # self.logger.debug("CascadeCorrelationNetwork: train_candidates: Multiprocessing context set to 'forkserver'.")
            # with candidate_training_context.Manager() as candidate_training_manager:
            #     self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates training manager has been created: {candidate_training_manager}.")
            #     candidate_training_queue = candidate_training_manager.Queue()
            #     self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates training queue has been created: {candidate_training_queue}.")
            #     with candidate_training_context.Pool(processes=process_count) as candidates_training_pool:
            #         self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates training pool has been created: {candidates_training_pool}.")
            #         results = candidates_training_pool.map(CascadeCorrelationNetwork.train_candidate_worker, tasks)
            #         self.logger.debug("CascadeCorrelationNetwork: train_candidates: Candidates training pool has completed all tasks.")
            #         # candidates_training_pool.join()
            #         # candidates_training_pool.close()
            #         # self.logger.debug("CascadeCorrelationNetwork: train_candidates: Candidates training pool has been joined and closed.")
            #     self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates training pool completed and closed: collected {len(results)} results.")
            # self.logger.debug("CascadeCorrelationNetwork: train_candidates: Candidates training manager has been closed.")

            # TODO: consider this option fourth - Fixed multiprocessing manager implementation
            # from multiprocessing.managers import BaseManager
            # from multiprocessing import Queue
            # import multiprocessing as mp

            # Create custom shared task and done queues first
            candidate_training_tasks_queue = Queue()
            candidate_training_done_queue = Queue()
            self.logger.debug(
                "CascadeCorrelationNetwork: train_candidates: Created custom candidate training tasks and done queues."
            )

            # Define a custom manager class to manage the queues
            class CandidateTrainingManager(BaseManager):
                pass

            # Register the queues getter methods with the manager server
            CandidateTrainingManager.register(
                "get_tasks_queue", callable=lambda: candidate_training_tasks_queue
            )
            CandidateTrainingManager.register(
                "get_done_queue", callable=lambda: candidate_training_done_queue
            )
            self.logger.debug(
                "CascadeCorrelationNetwork: train_candidates: Registered CandidateTrainingManager with tasks and done queues."
            )

            try:
                # Start the manager server
                # with CandidateTrainingManager( address=self.candidate_training_queue_address, authkey=self.candidate_training_queue_authkey.encode('utf-8')) as candidate_training_manager:
                candidate_training_manager = CandidateTrainingManager(
                    address=self.candidate_training_queue_address,
                    authkey=self.candidate_training_queue_authkey.encode("utf-8"),
                )
                candidate_training_manager.start()
                self.logger.debug(
                    f"CascadeCorrelationNetwork: train_candidates: Manager server started at {self.candidate_training_queue_address}"
                )

                # Get the shared queues
                candidate_training_shared_tasks_queue = (
                    candidate_training_manager.get_tasks_queue()
                )
                self.logger.verbose(
                    f"CascadeCorrelationNetwork: train_candidates: Candidate training shared tasks queue: {candidate_training_shared_tasks_queue}"
                )
                candidate_training_shared_done_queue = (
                    candidate_training_manager.get_done_queue()
                )
                self.logger.verbose(
                    f"CascadeCorrelationNetwork: train_candidates: Candidate training shared done queue: {candidate_training_shared_done_queue}"
                )

                # Add tasks to the shared tasks queue
                candidate_training_shared_tasks_queue.put(
                    tasks[i] for i in range(len(tasks))
                )
                self.logger.debug(
                    f"CascadeCorrelationNetwork: train_candidates: Added {len(tasks)} tasks to queue"
                )

                # Create worker processes using the standard context
                # candidate_training_context = mp.get_context(method='spawn')  # Use spawn for better compatibility
                candidate_training_context = mp.get_context(
                    method="forkserver"
                )  # Alternative context is 'forkserver'

                with candidate_training_context.Pool(
                    processes=process_count
                ) as candidates_training_pool:
                    self.logger.debug(
                        f"CascadeCorrelationNetwork: train_candidates: Created candidates training pool with {process_count} processes: {candidates_training_pool}."
                    )

                    # # Use the sequential training approach with proper error handling
                    # results = []
                    # for task in tasks:
                    #     try:
                    #         result = self.train_candidate_worker(task, parallel=False)
                    #         results.append(result)
                    #         self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Completed task {task[0]}")
                    #     except Exception as e:
                    #         self.logger.error(f"CascadeCorrelationNetwork: train_candidates: Error processing task {task[0]}: {e}")
                    #         # Add a default result for failed tasks
                    #         results.append((task[0], task[1][4], 0.0, None))  # candidate_index, uuid, correlation, candidate

                    try:
                        self.logger.debug(
                            f"CascadeCorrelationNetwork: train_candidates: Candidates training pool has been created: {candidates_training_pool}."
                        )
                        # results = candidates_training_pool.map(CascadeCorrelationNetwork.train_candidate_worker, tasks)
                        results = candidates_training_pool.apply_async(
                            CascadeCorrelationNetwork.train_candidate_workers,
                            # args=(candidate_training_shared_tasks_queue,candidate_training_shared_done_queue,),
                            args=(
                                None,
                                candidate_training_shared_tasks_queue,
                                candidate_training_shared_done_queue,
                                True,
                            ),
                        )
                        self.logger.debug(
                            "CascadeCorrelationNetwork: train_candidates: Candidates training pool has completed all tasks."
                        )
                    except Exception as e:
                        self.logger.error(
                            f"CascadeCorrelationNetwork: train_candidates: Error occurred while training candidates: {e}"
                        )
                        # Add a default result for failed tasks
                        # results.append((task[0], task[1][4], 0.0, None))  # candidate_index, uuid, correlation, candidate

                    self.logger.debug(
                        f"CascadeCorrelationNetwork: train_candidates: Collected {len(results)} results"
                    )

                self.logger.debug(
                    "CascadeCorrelationNetwork: train_candidates: Candidates training pool has been closed."
                )

            except Exception as e:
                self.logger.error(
                    f"CascadeCorrelationNetwork: train_candidates: Error occurred while starting candidate training manager: {e}"
                )
                # Fallback to sequential processing
                self.logger.warning(
                    "CascadeCorrelationNetwork: train_candidates: Process pool failed for parallel evaluation of candidate units.  Falling back to sequential processing"
                )
                results = []
                for task in tasks:
                    try:
                        result = self.train_candidate_worker(task, parallel=False)
                        results.append(result)
                    except Exception as task_e:
                        self.logger.error(
                            f"CascadeCorrelationNetwork: train_candidates: Task error: Error processing task in sequential fallback mode: {task_e}"
                        )
                        results.append((task[0], task[1][4], 0.0, None))

                self.logger.debug(
                    f"CascadeCorrelationNetwork: train_candidates: Completed sequential fallback processing of {len(results)} tasks."
                )
            finally:
                # Shutdown the manager server
                try:
                    if "candidate_training_manager" in locals():
                        self.logger.debug(
                            "CascadeCorrelationNetwork: train_candidates: Shutting down candidate training manager."
                        )
                        candidate_training_manager.shutdown()
                    self.logger.debug(
                        "CascadeCorrelationNetwork: train_candidates: Candidates training manager has been closed."
                    )
                except Exception as cleanup_e:
                    self.logger.error(
                        f"CascadeCorrelationNetwork: train_candidates: Error occurred while shutting down candidate training manager: {cleanup_e}"
                    )

            self.logger.debug(
                f"CascadeCorrelationNetwork: train_candidates: Training candidate Units with multiprocessing manager and shared queues or fallback completed with {len(results)} results."
            )

            # self.logger.debug("CascadeCorrelationNetwork: train_candidates: Multiprocessing context set to 'forkserver'.")
            # with candidate_training_context.CandidateTrainingManager(address=self.candidate_training_queue_address, authkey=self.candidate_training_queue_authkey) as candidate_training_manager:
            #     self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates training manager has been created: {candidate_training_manager}.")
            #     candidate_training_tasks_queue = candidate_training_manager.get_tasks_queue()
            #     self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates training tasks queue has been created: {candidate_training_tasks_queue}.")
            #     candidate_training_server = candidate_training_manager.get_server()
            #     self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates training server has been created: {candidate_training_server}.")
            #     candidate_training_server.serve_forever()
            #     self.logger.debug("CascadeCorrelationNetwork: train_candidates: Candidates training server has been started.")
            #     candidate_training_tasks_queue.put(tasks[i] for i in range(len(tasks)))
            #     self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Tasks have been added to the candidates training tasks queue: {candidate_training_tasks_queue.qsize()} tasks in queue.")
            #     with candidate_training_context.Pool(processes=process_count) as candidates_training_pool:
            #         self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates training pool has been created: {candidates_training_pool}.")
            #         # results = candidates_training_pool.map(CascadeCorrelationNetwork.train_candidate_worker, tasks)
            #         results = candidates_training_pool.apply_async(CascadeCorrelationNetwork.train_candidate_workers)
            #         self.logger.debug("CascadeCorrelationNetwork: train_candidates: Candidates training pool has completed all tasks.")
            #     self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates training pool completed and closed: collected {len(results)} results.")
            # self.logger.debug("CascadeCorrelationNetwork: train_candidates: Candidates training manager has been closed.")

        end_time = datetime.datetime.now()
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Training ended at time: {end_time}"
        )
        self.logger.info(
            f"CascadeCorrelationNetwork: train_candidates: Training duration: {end_time - start_time}"
        )

        # Sort candidate correlations
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Unsorted candidate training results: Type: {type(results)}, Size: {len(results)}, Results:\n{results}"
        )

        results.sort(
            key=lambda r: (r[2] is not None, r[2]), reverse=True
        )  # Sort by correlation, None correlations go to the end
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Sorted candidate training results: Type: {type(results)}, Size: {len(results)}, Results:\n{results}"
        )

        # Extract candidates and their correlations
        candidate_ids = [r[0] for r in results if r[0] is not None]
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Extracted candidate IDs from results: Type: {type(candidate_ids)}, Size: {len(candidate_ids)}, IDs:\n{candidate_ids}"
        )
        candidate_uuids = [r[1] for r in results if r[1] is not None]
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Extracted candidate UUIDs from results: Type: {type(candidate_uuids)}, Size: {len(candidate_uuids)}, UUIDs:\n{candidate_uuids}"
        )
        correlations = [r[2] for r in results if r[2] is not None]
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Extracted correlations from results: Type: {type(correlations)}, Size: {len(correlations)}, Correlations:\n{correlations}"
        )
        candidates = [r[3] for r in results if r[3] is not None]
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Extracted candidates from results: Type: {type(candidates)}, Size: {len(candidates)}, Candidates:\n{candidates}"
        )

        # Identify the best candidate based on maximum correlation
        max_correlation = max(correlations, default=0.0)
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Maximum correlation from candidates: {max_correlation:.6f}"
        )
        best_candidate_id = correlations.index(max_correlation) if correlations else -1
        if best_candidate_id != 0:
            self.logger.warning(
                "CascadeCorrelationNetwork: train_candidates: Expected best candidate ID to be 0 after sorting correlations tuple list."
            )
        elif best_candidate_id >= len(correlations) or best_candidate_id < 0:
            self.logger.warning(
                "CascadeCorrelationNetwork: train_candidates: Best candidate ID is out of range; setting best candidate ID to -1."
            )
            best_candidate_id = -1
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Best candidate index: {best_candidate_id}"
        )
        best_candidate_uuid = (
            candidate_uuids[best_candidate_id]
            if best_candidate_id >= 0 and best_candidate_id < len(candidate_uuids)
            else "None"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Best candidate UUID: {best_candidate_uuid}"
        )
        best_candidate_correlation = (
            correlations[best_candidate_id]
            if best_candidate_id >= 0 and best_candidate_id < len(correlations)
            else 0.0
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Best candidate correlation: {best_candidate_correlation:.6f}"
        )
        best_candidate = (
            candidates[best_candidate_id]
            if best_candidate_id >= 0 and best_candidate_id < len(candidates)
            else "None"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Best candidate object: {best_candidate}"
        )
        self.logger.info(
            f"CascadeCorrelationNetwork: train_candidates: Maximum correlation from candidates: {max_correlation:.6f}, Best Candidate ID: {best_candidate_id}, Best Candidate UUID: {best_candidate_uuid}"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Maximum correlation from candidates: {max_correlation:.6f}"
        )
        # self.logger.warning("CascadeCorrelationNetwork: train_candidates: No valid correlations found among candidates; setting maximum correlation to 0.0")

        # Complete compilation of training results
        # TODO: verify this doesn't need an abs() around correlation (r[2])
        successful_candidates = sum(
            map(
                lambda r: r[2] is not None and r[2] > self.correlation_threshold,
                results,
            )
        )
        failed_candidates = len(results) - successful_candidates
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Successful candidates count: {successful_candidates}, Failed candidates count: {failed_candidates}"
        )

        # Prepare attribute tuples to return all candidate data and attributes
        # TODO: Consider using named tuple or dataclass for return values
        candidates_attribute_list = (
            candidate_ids,
            candidate_uuids,
            correlations,
            candidates,
        )
        best_candidate_attributes = (
            best_candidate_id,
            best_candidate_uuid,
            best_candidate_correlation,
            best_candidate,
        )
        max_correlation_attributes = (
            max_correlation,
            successful_candidates,
            failed_candidates,
        )

        self.logger.info(
            f"CascadeCorrelationNetwork: train_candidates: Total Candidates Trained: {len(results)}, Successful: {successful_candidates}, Failed: {failed_candidates}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: train_candidates: Completed training of candidate units."
        )

        # TODO: Consider using named tuple or dataclass for return values
        return (
            candidates_attribute_list,
            best_candidate_attributes,
            max_correlation_attributes,
        )

        # # for candidate_index, candidate_uuid, candidate, correlation in results:
        # for candidate_index, candidate_uuid, correlation in results:
        #     # if candidate is not None:
        #     #     candidate.correlation = correlation
        #     if correlation:
        #         # candidate.correlation = correlation
        #         # candidates.append(candidate)
        #         successful_candidates += 1
        #         # self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidate index: {candidate_index + 1}, Candidate UUID: {candidate_uuid}, Correlation: {correlation:.6f}, Weights: {candidate.weights}, Bias: {candidate.bias}\n{candidate}")
        #         self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidate index: {candidate_index + 1}, Candidate UUID: {candidate_uuid}, Correlation: {correlation:.6f}")
        #     else:
        #         failed_candidates += 1
        #         self.logger.warning(f"CascadeCorrelationNetwork: train_candidates: Candidate index: {candidate_index + 1}, Candidate UUID: {candidate_uuid}, Correlation: {correlation:.6f}, Failed to train candidate.")

        # self.logger.info(f"CascadeCorrelationNetwork: train_candidates: Total Candidates Trained: {len(results)}, Successful: {successful_candidates}, Failed: {failed_candidates}")

        # if not candidates:
        #     self.logger.error("CascadeCorrelationNetwork: train_candidates: No successful candidates were trained.")
        #     # Create a dummy candidate to return
        #     dummy_candidate = type('DummyCandidate', (), {
        #         'correlation': 0.0,
        #         'get_uuid': lambda: 'dummy_candidate_uuid',
        #         # 'weights': torch.tensor([0.0]),
        #         'weights': torch.zeros(input_size),
        #         # 'bias': torch.tensor(0.0)
        #         'bias': torch.tensor(1)
        #     })()
        #     candidates.append(dummy_candidate)
        #     self.logger.warning("CascadeCorrelationNetwork: train_candidates: Returning a dummy candidate with zero correlation.")

        # # Check if we have any successful candidates
        # candidates.sort(key=lambda c: getattr(c, 'correlation', 0.0), reverse=True)
        # self.logger.debug(f"CascadeCorrelationNetwork: train_candidates: Candidates sorted by correlation:\n{[c.correlation for c in candidates]}")
        # # self.logger.info(f"CascadeCorrelationNetwork: train_candidates: Best Candidate Correlation: {candidates[0].correlation:.6f}, Best Candidate UUID: {candidates[0].get_uuid()}")
        # self.logger.info(f"CascadeCorrelationNetwork: train_candidates: Best Candidate Correlation: {candidates[0].correlation:.6f}")

        # self.logger.trace("CascadeCorrelationNetwork: train_candidates: Completed training of candidate units.")
        # return candidates
        # return (candidate_ids, candidate_uuids, correlations, candidates, best_candidate_id, best_candidate_uuid, best_candidate_correlation, max_correlation, best_candidate, successful_candidates, failed_candidates)
        # return ((candidate_ids, candidate_uuids, correlations, candidates,), (best_candidate_id, best_candidate_uuid, best_candidate_correlation, best_candidate,), (max_correlation, successful_candidates, failed_candidates))
        # return (candidates_attribute_list, best_candidate_attributes, max_correlation_attributes,)
        # candidates_attribute_list = (candidate_ids, candidate_uuids, correlations, candidates,)
        # best_candidate_attributes = (best_candidate_id, best_candidate_uuid, best_candidate_correlation, best_candidate,)
        # max_correlation_attributes = (max_correlation, successful_candidates, failed_candidates)

    #################################################################################################################################################################################################
    # Public Method to add a new hidden unit based on the correlation
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
            x: Input tensor to calculate the unit's output
        Notes:
            This method updates the network's hidden units and output layer weights to include the new unit.
            If no candidate unit is provided, a random candidate unit will be selected from the candidate pool.
            The new hidden unit will be appended to the network's hidden units list.
            The output layer weights will be updated to include the new unit.
        Raises:
            ValueError: If the candidate unit is None or if the maximum number of hidden units has been reached.
            TypeError: If the input tensor is not a torch.Tensor.
        Returns:
            None
        """
        # Prepare input for the new unit (includes outputs from existing hidden units)
        self.logger.trace(
            "CascadeCorrelationNetwork: add_unit: Starting to add a new hidden unit."
        )
        hidden_outputs = []
        for unit in self.hidden_units:
            unit_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
            unit_output = unit["activation_fn"](
                torch.sum(unit_input * unit["weights"], dim=1) + unit["bias"]
            ).unsqueeze(1)
            self.logger.debug(
                f"CascadeCorrelationNetwork: add_unit: Unit output shape: {unit_output.shape}, Unit output: {unit_output}"
            )
            hidden_outputs.append(unit_output)
        self.logger.debug(
            f"CascadeCorrelationNetwork: add_unit: Hidden outputs shape: {[h.shape for h in hidden_outputs]}"
        )
        candidate_input = (
            torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: add_unit: Candidate input shape: {candidate_input.shape}, Input size: {candidate_input.shape[1]}, Candidate Input:\n{candidate_input}"
        )

        # Create a new hidden unit
        new_unit = {
            "weights": candidate.weights.clone().detach(),
            "bias": candidate.bias.clone().detach(),
            "activation_fn": self.activation_fn,
            "correlation": candidate.correlation,
        }
        self.logger.debug(
            f"CascadeCorrelationNetwork: add_unit: Adding new hidden unit with weights: {new_unit['weights']}, bias: {new_unit['bias']}, correlation: {new_unit['correlation']:.6f}, Unit: {new_unit}"
        )

        # Add the new unit to the network
        self.hidden_units.append(new_unit)
        self.logger.debug(
            f"CascadeCorrelationNetwork: add_unit: Current number of hidden units: {len(self.hidden_units)}, Hidden units: {self.hidden_units}"
        )

        # Update output layer weights to include the new unit
        old_output_weights = self.output_weights.clone().detach()
        self.logger.debug(
            f"CascadeCorrelationNetwork: add_unit: Old output weights shape: {old_output_weights.shape}, Weights: {old_output_weights}"
        )
        old_output_bias = self.output_bias.clone().detach()
        self.logger.debug(
            f"CascadeCorrelationNetwork: add_unit: Old output bias shape: {old_output_bias.shape}, Bias: {old_output_bias}"
        )

        # Calculate the output of the new unit
        unit_output = self.activation_fn(
            torch.sum(candidate_input * new_unit["weights"], dim=1) + new_unit["bias"]
        ).unsqueeze(1)
        self.logger.debug(
            f"CascadeCorrelationNetwork: add_unit: New unit output shape: {unit_output.shape}, New unit output: {unit_output}"
        )

        # Create new output weights with an additional row for the new unit
        if hidden_outputs:
            new_input_size = x.shape[1] + len(hidden_outputs) + 1
        else:
            new_input_size = x.shape[1] + 1
        self.logger.debug(
            f"CascadeCorrelationNetwork: add_unit: New input size for output weights: {new_input_size}, Old input size: {old_output_weights.shape[0]}"
        )

        # Ensure new weights have requires_grad=True
        self.output_weights = (
            torch.randn(new_input_size, self.output_size, requires_grad=True) * 0.1
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: add_unit: New output weights shape: {self.output_weights.shape}, Weights: {self.output_weights}"
        )

        # Copy old weights
        if hidden_outputs:
            input_size_before = x.shape[1] + len(hidden_outputs)
        else:
            input_size_before = x.shape[1]
        self.logger.debug(
            f"CascadeCorrelationNetwork: add_unit: Input size before adding new unit: {input_size_before}"
        )

        # Copy old bias
        self.output_weights[:input_size_before, :] = old_output_weights
        self.logger.debug(
            f"CascadeCorrelationNetwork: add_unit: Updated output weights after copying old weights: {self.output_weights}"
        )
        self.output_bias = old_output_bias
        self.logger.debug(
            f"CascadeCorrelationNetwork: add_unit: Updated output bias after copying old bias: {self.output_bias}"
        )

        # Add new unit to the history
        self.logger.info(
            f"CascadeCorrelationNetwork: add_unit: Added hidden unit with correlation: {candidate.correlation:.6f}"
        )
        self.history["hidden_units_added"].append(
            {
                "correlation": candidate.correlation,
                "weights": candidate.weights.clone().detach().numpy(),
                "bias": candidate.bias.clone().detach().numpy(),
            }
        )
        self.logger.info(
            f"CascadeCorrelationNetwork: add_unit: Current number of hidden units: {len(self.hidden_units)}"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: add_unit: Updated history with new hidden unit:\n{self.history['hidden_units_added'][-1]}\nHistory\n{self.history}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: add_unit: Completed adding a new hidden unit."
        )

    #################################################################################################################################################################################################
    # Public Method to calculate the residual error of the network
    def calculate_residual_error(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
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
        with torch.no_grad():
            output = self.forward(x)
            residual = y - output
        return residual

    #################################################################################################################################################################################################
    # Public Method to calculate the accuracy of the network
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
        Description:
            Fit the network using the cascade correlation algorithm.
        Args:
            x_train: Training input tensor
            y_train: Training target tensor
            x_val: Validation input tensor
            y_val: Validation target tensor
            max_epochs: Maximum number of epochs to train
            early_stopping: Whether to use early stopping
        Notes:
            - The cascade correlation algorithm adds new hidden units to the network during training if the current performance is not satisfactory.
            - The training process continues until the maximum number of epochs is reached.
            - If early stopping is enabled, the training process stops if the validation loss does not improve for a certain number of epochs.
            - The training history will be stored in the 'history' attribute of the network.
            - The training history includes the training loss, validation loss, and accuracy for both the training and validation datasets.
        Returns:
            Training history: Dictionary containing:
                - lists of training and validation losses and accuracies over epochs.
        """
        # Initial training of the output layer
        self.logger.trace(
            "CascadeCorrelationNetwork: fit: Starting initial training of the output layer."
        )
        self.logger.info(
            "CascadeCorrelationNetwork: fit: Initial training of output layer"
        )
        max_epochs = (max_epochs, self.output_epochs)[max_epochs is None]
        train_loss = self.train_output_layer(x_train, y_train, max_epochs)
        self.history["train_loss"].append(train_loss)
        if x_val is not None and y_val is not None:
            with torch.no_grad():
                value_output = self.forward(x_val)
                value_loss = nn.MSELoss()(value_output, y_val).item()
            self.history["value_loss"].append(value_loss)
            self.logger.info(
                f"CascadeCorrelationNetwork: fit: Initial - Train Loss: {train_loss:.6f}, Val Loss: {value_loss:.6f}"
            )
        else:
            self.logger.info(
                f"CascadeCorrelationNetwork: fit: Initial - Train Loss: {train_loss:.6f}"
            )

        # Calculate initial accuracy
        train_accuracy = self.calculate_accuracy(x_train, y_train)
        self.history["train_accuracy"].append(train_accuracy)
        if x_val is not None and y_val is not None:
            value_accuracy = self.calculate_accuracy(x_val, y_val)
            self.history["value_accuracy"].append(value_accuracy)
            self.logger.info(
                f"CascadeCorrelationNetwork: fit: Initial - Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {value_accuracy:.4f}"
            )
        else:
            self.logger.info(
                f"CascadeCorrelationNetwork: fit: Initial - Train Accuracy: {train_accuracy:.4f}"
            )

        # Main training loop
        patience_counter = 0
        best_value_loss = float("inf") if x_val is not None else None
        # TODO:  this code is repeated in the train candidates method--refactor it into a common method
        self.logger.info(
            f"CascadeCorrelationNetwork: fit: Starting main training loop with max epochs: {max_epochs}, early stopping: {early_stopping}"
        )
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
        self.history["hidden_units_added"].append(
            {"correlation": 0.0, "weights": [], "bias": []}
        )

        self.logger.info("CascadeCorrelationNetwork: fit: Training completed.")
        self.logger.debug(
            f"CascadeCorrelationNetwork: fit: Final history:\n{self.history}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: fit: Completed training of the network."
        )
        return self.history

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
    ) -> (bool, int, float, torch.Tensor, float, float):
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
        Notes:
            - Candidate units are trained using the Cascade Correlation algorithm
            - Early stopping is used if specified
            - Validation loss and accuracy are calculated and tracked
            - Training history is tracked
            - Hidden units are added to the network using the Cascade Correlation algorithm
        Returns:
            Tuple containing:
                - early_stop: Whether training was stopped early
                - patience_counter: Updated patience counter
                - best_value_loss: Best validation loss seen so far
                - value_output: Output on validation set
                - value_loss: Validation loss
                - value_accuracy: Validation accuracy
        """
        self.logger.trace(
            "CascadeCorrelationNetwork: grow_network: Starting to grow the network by adding hidden units."
        )
        early_stop = False
        value_output = torch.zeros_like(y_train) if y_train is not None else None
        value_loss = float("inf") if y_train is not None else None
        value_accuracy = 0.0 if y_train is not None else None
        for epoch in range(max_epochs):

            # Calculate residual error
            residual_error = self.calculate_residual_error(x_train, y_train)
            self.logger.debug(
                f"CascadeCorrelationNetwork: grow_network: Epoch {epoch}, Residual Error: {residual_error.mean().item():.6f}"
            )

            # Train candidate units
            # NOTE: This method returns a list of candidates pre-sorted by correlation in descending order.  i.e., the best candidate is the first in the list
            # candidates = self.train_candidates(
            # TODO: Consider using named tuple or dataclass for return values
            (
                candidates_attribute_list,
                best_candidate_attributes,
                max_correlation_attributes,
            ) = self.train_candidates(
                x=x_train, y=y_train, residual_error=residual_error
            )

            # candidates_attribute_list = (candidate_ids, candidate_uuids, correlations, candidates,)
            # best_candidate_attributes = (best_candidate_id, best_candidate_uuid, best_candidate_correlation, best_candidate,)
            # max_correlation_attributes = (max_correlation, successful_candidates, failed_candidates)
            # return (candidates_attribute_list, best_candidate_attributes, max_correlation_attributes,)

            # TODO: Consider using named tuple or dataclass for return values
            (candidate_ids, candidate_uuids, correlations, candidates) = (
                candidates_attribute_list
            )
            (
                best_candidate_id,
                best_candidate_uuid,
                best_candidate_correlation,
                best_candidate,
            ) = best_candidate_attributes
            (max_correlation, successful_candidates, failed_candidates) = (
                max_correlation_attributes
            )

            self.logger.debug(
                f"CascadeCorrelationNetwork: grow_network: Trained Candidates: Epoch {epoch}, Number: {len(candidates)}, Values\n{candidates}"
            )
            # best_candidate = candidates[0]
            self.logger.debug(
                f"CascadeCorrelationNetwork: grow_network: Best Candidate: {best_candidate}"
            )
            self.logger.info(
                f"CascadeCorrelationNetwork: grow_network: Best Candidate Correlation: {best_candidate.correlation:.6f}, Weights: {best_candidate.weights}, Bias: {best_candidate.bias}"
            )

            # Check if best candidate meets correlation threshold
            # if best_candidate.correlation < self.correlation_threshold:
            if best_candidate.get_correlation() < self.correlation_threshold:
                self.logger.info(
                    f"CascadeCorrelationNetwork: grow_network: No candidate met correlation threshold: {self.correlation_threshold}, Best Correlation Achieved: {best_candidate.get_correlation():.6f}"
                )
                break
            self.logger.info(
                f"CascadeCorrelationNetwork: grow_network: Best Candidate: {best_candidate.get_correlation()}, Met correlation threshold: {self.correlation_threshold}"
            )

            # Add best candidate to the network
            self.add_unit(best_candidate, x_train)
            self.logger.info(
                "CascadeCorrelationNetwork: grow_network: Added best candidate to the network"
            )

            # Retrain output layer
            train_loss = self.train_output_layer(x_train, y_train, self.output_epochs)
            self.logger.info(
                f"CascadeCorrelationNetwork: grow_network: Full Network Training Loss after Epoch {epoch}, Train Loss: {train_loss:.6f}"
            )
            self.logger.debug(
                f"CascadeCorrelationNetwork: grow_network: For Current Epoch: {epoch}, Post-Train Loss History:\n{self.history}"
            )
            self.history["train_loss"].append(train_loss)
            self.logger.debug(
                f"CascadeCorrelationNetwork: grow_network: For Current Epoch: {epoch}, Post-Trained History:\n{self.history}"
            )

            # Calculate accuracy
            train_accuracy = self.calculate_accuracy(x_train, y_train)
            self.logger.debug(
                f"CascadeCorrelationNetwork: grow_network: For Current Epoch {epoch}, Train Accuracy: {train_accuracy:.4f}"
            )
            self.history["train_accuracy"].append(train_accuracy)
            self.logger.debug(
                f"CascadeCorrelationNetwork: grow_network: For Current Epoch {epoch}, Post-Train Accuracy History:\n{self.history}"
            )

            # Validation
            # TODO: Consider using named tuple or dataclass for return values
            (
                early_stop,
                patience_counter,
                best_value_loss,
                value_output,
                value_loss,
                value_accuracy,
            ) = self.validate_training(
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
            self.logger.debug(
                f"CascadeCorrelationNetwork: grow_network: Epoch {epoch}, Early Stop: {early_stop}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}, Value Output: {value_output} Value Loss: {value_loss:.6f}, Value Accuracy: {value_accuracy:.4f}"
            )
            if early_stop:
                self.logger.info(
                    f"CascadeCorrelationNetwork: grow_network: Early stopping triggered at epoch {epoch}."
                )
                break
            self.logger.info(
                f"CascadeCorrelationNetwork: grow_network: Epoch {epoch} - Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}, Early stop: {early_stop}"
            )

        self.logger.info(
            f"CascadeCorrelationNetwork: grow_network: Finished training after {epoch + 1} epochs. Total hidden units: {len(self.hidden_units)}"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: grow_network: Final history:\n{self.history}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: grow_network: Completed training of the network."
        )
        # TODO: Consider using named tuple or dataclass for return values
        return (
            early_stop,
            patience_counter,
            best_value_loss,
            value_output,
            value_loss,
            value_accuracy,
        )

    #################################################################################################################################################################################################
    # Public Method to validate the training process
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
        self.logger.trace(
            "CascadeCorrelationNetwork: validate_training: Starting validation of the training process."
        )
        early_stop_flag = False
        value_output = 0
        value_loss = float("inf")
        value_accuracy = 0.0
        best_value_loss = best_value_loss if best_value_loss is not None else 9999999.9
        self.logger.debug(
            f"CascadeCorrelationNetwork: validate_training: Epoch {epoch}, Max Epochs: {max_epochs}, Early Stopping: {early_stopping}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}, Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}"
        )

        # Validate input tensors
        self.logger.debug(
            f"CascadeCorrelationNetwork: validate_training: X Train: {x_train}, Y Train: {y_train}, X Val: {x_val}, Y Val: {y_val}"
        )
        if x_val is not None and y_val is not None:

            # Validate the model on the validation set
            with torch.no_grad():
                value_output = self.forward(x_val)
                value_loss = nn.MSELoss()(value_output, y_val).item()
            self.history["value_loss"].append(value_loss)

            # Calculate validation accuracy
            value_accuracy = self.calculate_accuracy(x_val, y_val)
            self.history["value_accuracy"].append(value_accuracy)
            self.logger.info(
                "CascadeCorrelationNetwork: validate_training: "
                f"Epoch {epoch} - Train Loss: {train_loss:.6f}, Val Loss: {value_loss:.6f}, "
                f"Train Acc: {train_accuracy:.4f}, Val Acc: {value_accuracy:.4f}, "
                f"Units: {len(self.hidden_units)}"
            )

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
            self.logger.verbose(
                f"CascadeCorrelationNetwork: validate_training: Early Stopping: {early_stopping}"
            )
            self.logger.verbose(
                f"CascadeCorrelationNetwork: validate_training: Early Stop: {early_stop}"
            )
            self.logger.verbose(
                f"CascadeCorrelationNetwork: validate_training: Epoch: {epoch}, Early Stop: {early_stop}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}"
            )

            # early_stop_flag = True if early_stopping and early_stop else False
            early_stop_flag = early_stopping and early_stop
            self.logger.info(
                f"CascadeCorrelationNetwork: validate_training: Stop Training Early: {early_stop} and Early Stopping: {early_stopping}: {early_stopping and early_stop}"
            )
            self.logger.info(
                f"CascadeCorrelationNetwork: validate_training: Early Stopping: {early_stop_flag}, Patience Counter: {patience_counter}, Best Val Loss: {best_value_loss:.6f}"
            )
            self.logger.verbose(
                f"CascadeCorrelationNetwork: validate_training: Value Output: {value_output}, Value Loss: {value_loss:.6f}, Value Accuracy: {value_accuracy:.4f}"
            )

        self.logger.verbose(
            f"CascadeCorrelationNetwork: validate_training: Epoch {epoch}, Early Stop: {early_stop_flag}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}, Value Output: {value_output}, Value Loss: {value_loss:.6f}, Value Accuracy: {value_accuracy:.4f}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: validate_training: Completed validation of the training process."
        )
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
            - This method does not update the model's parameters
        Returns:
            bool: Whether early stopping should be triggered
            int: Updated patience counter
            float: Updated best validation loss
        """
        # Early stopping
        self.logger.trace(
            "CascadeCorrelationNetwork: evaluate_early_stopping: Starting evaluation of early stopping conditions."
        )

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
            self.logger.info(
                f"CascadeCorrelationNetwork: evaluate_early_stopping: Epoch {epoch} - Patience Counter: {patience_counter}, Value Loss: {value_loss}, Best Val Loss: {best_value_loss:.6f}"
            )
            if patience_exhausted:
                self.logger.info(
                    f"CascadeCorrelationNetwork: evaluate_early_stopping: Patience Exhausted: {patience_exhausted}, Early stopping triggered after {epoch} epochs"
                )
            else:
                self.logger.info(
                    f"CascadeCorrelationNetwork: evaluate_early_stopping: Epoch {epoch} - Train Loss: {train_loss:.6f}, "
                    f"Train Acc: {train_accuracy:.4f}, Units: {len(self.hidden_units)}"
                )

            # Check if we've reached the maximum number of hidden units
            if max_units_reached := self.check_hidden_units_max():
                self.logger.info(
                    f"CascadeCorrelationNetwork: evaluate_early_stopping: Reached maximum number of hidden units: {max_units_reached}, stopping training"
                )

            # Check if we've achieved perfect accuracy
            if train_accuracy_reached := self.check_training_accuracy(
                train_accuracy=train_accuracy,
                accuracy_target=0.999,
            ):
                self.logger.info(
                    f"CascadeCorrelationNetwork: evaluate_early_stopping: Training accuracy reached target: {train_accuracy:.4f} >= 0.999"
                )

        early_stop = early_stopping and (
            train_accuracy_reached or max_units_reached or patience_exhausted
        )
        self.logger.info(
            f"CascadeCorrelationNetwork: evaluate_early_stopping: Early Stopping: {early_stop}, Patience Counter: {patience_counter}, Best Val Loss: {best_value_loss:.6f}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: evaluate_early_stopping: Completed evaluation of early stopping conditions."
        )

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
            - This method does not update the model's parameters
        Returns:
            bool: Whether patience limit is reached
            int: Updated patience counter
            float: Best validation loss
        """
        # Check if validation loss improved
        self.logger.trace(
            "CascadeCorrelationNetwork: check_patience: Starting to check patience limit."
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: check_patience: Current Value Loss: {value_loss:.6f}, Best Value Loss: {best_value_loss:.6f}, Patience Counter: {patience_counter}"
        )
        if value_loss < best_value_loss:
            best_value_loss = value_loss
            patience_counter = 0
        else:
            patience_counter += 1
        self.logger.info(
            f"CascadeCorrelationNetwork: check_patience: Patience counter: {patience_counter}, Best Validation Loss: {best_value_loss:.6f}"
        )

        # Check if patience limit is reached
        if patience_exhausted := (patience_counter >= self.patience):
            self.logger.info(
                f"CascadeCorrelationNetwork: check_patience: Patience limit reached: {patience_counter} >= {self.patience}"
            )
        self.logger.debug(
            f"CascadeCorrelationNetwork: check_patience: Patience Exhausted: {patience_exhausted}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: check_patience: Completed checking patience limit."
        )

        # TODO: Consider using named tuple or dataclass for return values
        return (patience_exhausted, patience_counter, best_value_loss)

    #################################################################################################################################################################################################
    # Public Methods to check conditions for training
    def check_hidden_units_max(self) -> bool:
        """
        Description:
            Check if we've reached the maximum number of hidden units
        Args:
            None
        Notes:
            - This method checks the length of the hidden_units list against the max_hidden_units attribute
            - If the length of hidden_units is greater than or equal to max_hidden_units, the method returns True
            - If the length of hidden_units is less than max_hidden_units, the method returns False
        Returns:
            bool: Whether we've reached max hidden units
        """
        # Check if we've reached max hidden units
        self.logger.trace(
            "CascadeCorrelationNetwork: check_hidden_units_max: Starting to check if max hidden units reached."
        )
        max_units_reached = len(self.hidden_units) >= self.max_hidden_units
        self.logger.info(
            f"CascadeCorrelationNetwork: check_hidden_units_max: Current hidden units: {max_units_reached}, Max allowed: {self.max_hidden_units}"
        )
        if max_units_reached:
            self.logger.info(
                f"CascadeCorrelationNetwork: check_hidden_units_max: Reached maximum number of hidden units: {self.max_hidden_units}"
            )
        self.logger.trace(
            "CascadeCorrelationNetwork: check_hidden_units_max: Completed checking if max hidden units reached."
        )
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
        self.logger.trace(
            "CascadeCorrelationNetwork: check_training_accuracy: Starting to check if training accuracy has reached the target."
        )
        if train_accuracy_reached := (train_accuracy >= accuracy_target):
            self.logger.info(
                f"CascadeCorrelationNetwork: check_training_accuracy: Reached target training accuracy: {train_accuracy:.4f} >= {accuracy_target:.4f}"
            )
        self.logger.debug(
            f"CascadeCorrelationNetwork: check_training_accuracy: Current Training Accuracy: {train_accuracy:.4f}, Target Accuracy: {accuracy_target:.4f}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: check_training_accuracy: Completed checking if training accuracy has reached the target."
        )
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
        self.logger.trace(
            "CascadeCorrelationNetwork: calculate_accuracy: Starting to calculate accuracy."
        )
        # Validate input tensors
        if x is None or y is None:
            self.logger.error(
                "CascadeCorrelationNetwork: calculate_accuracy: Missing required tensors for accuracy calculation."
            )
            raise ValueError(
                "CascadeCorrelationNetwork: calculate_accuracy: Missing required tensors for accuracy calculation."
            )
        self.logger.debug(
            f"CascadeCorrelationNetwork: calculate_accuracy: Input shape: {x.shape}, Target shape: {y.shape}"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: calculate_accuracy: input size: {self.input_size}, output size: {self.output_size}"
        )
        if (
            x.shape[-1] != self.input_size
            or y.shape[-1] != self.output_size
            or x.shape[0] != y.shape[0]
        ):
            self.logger.error(
                f"CascadeCorrelationNetwork: calculate_accuracy: Input and target tensors must have compatible shapes. Input (x): {x.shape}, Target (y): {y.shape}, input size: {self.input_size}, output size: {self.output_size}"
            )
            raise ValueError(
                "CascadeCorrelationNetwork: calculate_accuracy: Input and target tensors must have compatible shapes."
            )
        self.logger.debug(
            f"CascadeCorrelationNetwork: calculate_accuracy: Validated input shape: {x.shape}, Target shape: {y.shape}"
        )

        # Calculating accuracy
        self.logger.debug(
            f"CascadeCorrelationNetwork: calculate_accuracy: Calculating accuracy for input shape: {x.shape}, target shape: {y.shape}"
        )
        with torch.no_grad():
            output = self.forward(x)
            self.logger.debug(
                f"CascadeCorrelationNetwork: calculate_accuracy: Output shape: {output.shape}, Output: {output}"
            )
            accuracy = self._accuracy(y=y, output=output)
        self.logger.info(
            f"CascadeCorrelationNetwork: calculate_accuracy: Calculated accuracy: {accuracy:.4f}, Percentage: {accuracy * 100:.2f}%"
        )

        # Returning accuracy
        self.logger.trace(
            "CascadeCorrelationNetwork: calculate_accuracy: Completed calculating accuracy."
        )
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
            - This method assumes that the target and output tensors are one-hot encoded.
            - The accuracy is calculated as the percentage of correct predictions over the total number of predictions.
            - If either the target or output tensor is missing, an error is raised.
        Returns:
            Accuracy as a float
        """
        self.logger.trace(
            "CascadeCorrelationNetwork: _accuracy: Starting to calculate accuracy."
        )

        # Validate input tensors
        if y is None or output is None:
            self.logger.error(
                "CascadeCorrelationNetwork: _accuracy: Missing required tensors for accuracy calculation."
            )
            raise ValueError(
                "CascadeCorrelationNetwork: _accuracy: Missing required tensors for accuracy calculation."
            )
        elif not (isinstance(y, torch.Tensor) and isinstance(output, torch.Tensor)):
            self.logger.error(
                "CascadeCorrelationNetwork: _accuracy: All inputs must be torch tensors."
            )
            raise TypeError(
                "CascadeCorrelationNetwork: _accuracy: All inputs must be torch tensors."
            )
        if y.shape[0] != output.shape[0]:
            self.logger.error(
                f"CascadeCorrelationNetwork: _accuracy: Input and output tensors must have the same number of samples. Got {y.shape[0]} and {output.shape[0]}."
            )
            raise ValueError(
                "CascadeCorrelationNetwork: _accuracy: Input and output tensors must have the same number of samples."
            )
        self.logger.debug(
            f"CascadeCorrelationNetwork: _accuracy: Input shape: {y.shape}, Output shape: {output.shape}"
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: _accuracy: Input shape: {y.shape}, Input: {y}"
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: _accuracy: Output shape: {output.shape}, Output: {output}"
        )

        # Find predicted and target values
        predicted = torch.argmax(output, dim=1)
        self.logger.verbose(
            f"CascadeCorrelationNetwork: _accuracy: Predicted shape: {predicted.shape}, Predicted: {predicted}"
        )
        target = torch.argmax(y, dim=1)
        self.logger.verbose(
            f"CascadeCorrelationNetwork: _accuracy: Target shape: {target.shape}, Target: {target}"
        )
        correct = (predicted == target).sum().item()
        self.logger.verbose(
            f"CascadeCorrelationNetwork: _accuracy: Number of correct predictions: {correct}, Total samples: {len(target)}"
        )
        accuracy = correct / len(target)
        self.logger.info(
            f"CascadeCorrelationNetwork: _accuracy: Calculated accuracy: {accuracy:.4f}, Percentage: {accuracy * 100:.4f}%"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: _accuracy: Completed calculating accuracy."
        )
        return accuracy

    #################################################################################################################################################################################################
    # Public Method to make predictions
    # This method uses the forward method to get the output of the network
    # It is used to make predictions on new data
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            This method takes an input tensor, passes it through the network, and then returns the predicted output.
            The input tensor must be compatible with the input shape of the network.
            The output tensor has the same shape as the input tensor.
            If the input tensor is missing, an error is raised.
        Args:
            x: Input tensor
        Notes:
            - The input tensor must have the same number of channels as the model's input layer.
            - If the input tensor is incompatible with the input shape of the network, an error is raised.
            - If the input tensor has a different number of channels than the model's input layer, an error is raised.
            - If the input tensor has a different number of samples than the model's input layer, an error is raised.
            - If the input tensor is compatible with the input shape of the network, the method returns the predicted output.
        Raises:
            ValueError: If the input tensor is missing or has an incompatible shape.
            TypeError: If the input tensor is not a torch tensor.
        Returns:
            Predicted output: torch.Tensor
        """
        # Validate input tensor
        if x is None:
            self.logger.error(
                "CascadeCorrelationNetwork: predict: Input tensor is missing."
            )
            raise ValueError(
                "CascadeCorrelationNetwork: predict: Input tensor is required."
            )
        if not isinstance(x, torch.Tensor):
            self.logger.error(
                "CascadeCorrelationNetwork: predict: Input must be a torch tensor."
            )
            raise TypeError(
                "CascadeCorrelationNetwork: predict: Input must be a torch tensor."
            )
        if x.shape[0] != self.input_size:
            self.logger.error(
                f"CascadeCorrelationNetwork: predict: Input tensor has incompatible shape. Expected first dimension to be {self.input_size}, got {x.shape[0]}."
            )
            raise ValueError(
                f"CascadeCorrelationNetwork: predict: Input tensor has incompatible shape. Expected first dimension to be {self.input_size}, got {x.shape[0]}."
            )

        # Return the predicted output
        self.logger.debug(f"CascadeCorrelationNetwork: predict: Input shape: {x.shape}")
        self.logger.trace(
            "CascadeCorrelationNetwork: predict: Starting to make predictions."
        )
        with torch.no_grad():
            predicted_value = self.forward(x)
            self.logger.trace(
                "CascadeCorrelationNetwork: predict: Finished making predictions."
            )
        self.logger.debug(
            f"CascadeCorrelationNetwork: predict: Predicted shape: {predicted_value.shape}, Predicted: {predicted_value}"
        )
        return predicted_value

    #################################################################################################################################################################################################
    # Public Method to predict class labels
    # This method predicts the class labels for the input tensor
    # It uses the forward method to get the output and then applies argmax to get the class labels
    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            This method takes an input tensor, passes it through the network, and then applies argmax to get the class labels.
        Args:
            x: Input tensor
        Notes:
            - The input tensor must be compatible with the input shape of the network.
            - The output tensor contains the predicted class labels.
            - If the input tensor is missing, an error is raised.
            - If the input tensor is not a torch tensor, an error is raised.
            - If the input tensor has an incompatible shape, an error is raised.
            - If the output tensor contains the predicted class labels, the method returns the predicted class labels.
        Raises:
            ValueError: If the input tensor is missing or has an incompatible shape.
            TypeError: If the input tensor is not a torch tensor.
        Returns:
            Predicted class labels
        """
        # Validate input tensor
        if x is None:
            self.logger.error(
                "CascadeCorrelationNetwork: predict_classes: Input tensor is missing."
            )
            raise ValueError(
                "CascadeCorrelationNetwork: predict_classes: Input tensor is required."
            )
        if not isinstance(x, torch.Tensor):
            self.logger.error(
                "CascadeCorrelationNetwork: predict_classes: Input must be a torch tensor."
            )
            raise TypeError(
                "CascadeCorrelationNetwork: predict_classes: Input must be a torch tensor."
            )
        # if x.shape[0]!= self.input_size:
        if x.shape[1] != self.input_size:
            self.logger.error(
                f"CascadeCorrelationNetwork: predict_classes: Input tensor has incompatible shape. Expected first dimension to be {self.input_size}, got[0,1] {x.shape[0]}, {x.shape[1]}."
            )
            raise ValueError(
                f"CascadeCorrelationNetwork: predict_classes: Input tensor has incompatible shape. Expected first dimension to be {self.input_size}, got {x.shape[0]}."
            )

        # Return the predicted class labels
        self.logger.debug(
            f"CascadeCorrelationNetwork: predict_classes: Input shape: {x.shape}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: predict_classes: Starting to predict class labels."
        )
        with torch.no_grad():
            output = self.forward(x)
            prediction = torch.argmax(output, dim=1)
            self.logger.info(
                f"CascadeCorrelationNetwork: predict_classes: Predicted class labels shape: {prediction.shape}, Prediction: {prediction}"
            )
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
        self.logger.trace(
            "CascadeCorrelationNetwork: summary: Starting to print network summary."
        )
        self.logger.info(
            "CascadeCorrelationNetwork: summary: Display Cascade Correlation Network Summary:"
        )
        self.logger.info(
            f"CascadeCorrelationNetwork: summary: Input size: {self.input_size}"
        )
        self.logger.info(
            f"CascadeCorrelationNetwork: summary: Output size: {self.output_size}"
        )
        self.logger.info(
            f"CascadeCorrelationNetwork: summary: Number of hidden units: {len(self.hidden_units)}"
        )

        # Display hidden unit info if present
        if self.hidden_units:
            self.logger.info("CascadeCorrelationNetwork: summary: Hidden Units:\n")
            for i, unit in enumerate(self.hidden_units):
                self.logger.info(f"CascadeCorrelationNetwork: summary:   Unit {i+1}:")
                self.logger.info(
                    f"CascadeCorrelationNetwork: summary:     Input size: {len(unit['weights'])}"
                )
                self.logger.info(
                    f"CascadeCorrelationNetwork: summary:     Correlation: {unit['correlation']:.6f}"
                )

        # Display Training Parameters
        self.logger.info("CascadeCorrelationNetwork: summary: Training Parameters:")
        self.logger.info(
            f"CascadeCorrelationNetwork: summary:   Learning rate: {self.learning_rate}"
        )
        self.logger.info(
            f"CascadeCorrelationNetwork: summary:   Candidate pool size: {self.candidate_pool_size}"
        )
        self.logger.info(
            f"CascadeCorrelationNetwork: summary:   Correlation threshold: {self.correlation_threshold}"
        )

        # Display final training accuracy if attribute exists
        if self.history["train_accuracy"]:
            self.logger.info(
                f"CascadeCorrelationNetwork: summary: Final training accuracy:\n{self.history['train_accuracy'][-1]:.6f}"
            )

        # Display final value accuracy if validation was used
        if "value_accuracy" in self.history and self.history["value_accuracy"]:
            self.logger.info(
                f"CascadeCorrelationNetwork: summary: Final validation accuracy:\n{self.history['value_accuracy'][-1]:.6f}"
            )
        self.logger.trace(
            "CascadeCorrelationNetwork: summary: Completed printing network summary."
        )

    #################################################################################################################################################################################################
    # Define public methods for plotting the dataset, decision boundary and training history
    #################################################################################################################################################################################################
    # Public Method to plot the dataset used for training
    @staticmethod
    def plot_dataset(
        # self,
        x: torch.Tensor,
        y: torch.Tensor,
        title: str = "Training Dataset",
    ) -> None:
        """
        Describe:
            Plot the training dataset.
            This method plots the input and target tensors on a 2D scatter plot.
            It also displays the decision boundary of the network.
        Args:
            x: Input tensor
            y: Target tensor
            title: Plot title
        Note:
            - This method assumes that the input tensor has 2 features for 2D plotting.
            - If the input tensor does not have 2 features, an error is raised.
            - If the target tensor does not have one-hot encoded labels, an error is raised.
        Raises:
            ValueError: If the input tensors are not valid
        Returns:
            None
        """
        # self.logger.trace("CascadeCorrelationNetwork: plot_dataset: Starting to plot the dataset.")
        # # logger = Logger.get_instance()
        # logger = Logger.get_instance()
        logger = Logger
        logger.set_level("INFO")
        logger.trace(
            "CascadeCorrelationNetwork: plot_dataset: Starting to plot the dataset."
        )
        process_info = current_process()
        process = (
            process_info.pid,
            process_info.name,
        )
        logger.debug(
            f"CascadeCorrelationNetwork: plot_dataset: Process ID: {os.getpid()}, Process ID: {process[0]}, Process Name: {process[1]}"
        )

        # Convert to numpy for plotting
        logger.debug(
            f"CascadeCorrelationNetwork: plot_dataset: process {process[0]}: Converting input and target tensors to numpy arrays for plotting."
        )
        logger.debug(
            f"CascadeCorrelationNetwork: plot_dataset: process {process[0]}: Input shape: {x.shape}, Target shape: {y.shape},\nY Value:\n{y}"
        )
        x_np = x.numpy()
        y_np = torch.argmax(y, dim=1).numpy()

        # Plot the figure and labels
        # self.logger.info(f"CascadeCorrelationNetwork: plot_dataset: Plotting dataset with title: {title}")
        logger.info(
            f"CascadeCorrelationNetwork: plot_dataset: process {process[0]}: Plotting dataset with title: {title}"
        )
        plt.figure(figsize=(10, 8))
        for i in range(len(np.unique(y_np))):
            plt.scatter(x_np[y_np == i, 0], x_np[y_np == i, 1], label=f"Class {i}")
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        # plt.show(block=False)
        plt.show()
        # self.logger.trace("CascadeCorrelationNetwork: plot_dataset: Completed plotting the dataset.")
        logger.trace(
            f"CascadeCorrelationNetwork: plot_dataset: process {process[0]}: Completed plotting the dataset."
        )
        # self.logger.trace("CascadeCorrelationNetwork: plot_dataset: Completed plotting the dataset.")
        # self.logger.debug(f"CascadeCorrelationNetwork: plot_dataset: Process ID: {os.getpid()}, Process Name: {mp.current_process().name}, Process[0]: {process[0]}")

    #################################################################################################################################################################################################
    # Public Method to plot the decision boundary of the network. This method visualizes the decision boundary of the network in a 2D space. It uses matplotlib to create a contour plot of the decision boundary
    def plot_decision_boundary(
        self,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        title: str = "Decision Boundary",
    ) -> None:
        """
        Describe:
            Plot the decision boundary of the network.
            This method plots a 2D scatter plot of the input data, along with the decision boundary of the network.
            It uses matplotlib to create a contour plot of the decision boundary.
        Args:
            x: Input tensor
            y: Target tensor
            title: Plot title
        Note:
            - This method assumes that the input tensor has 2 features for 2D plotting.
            - If the input tensor does not have 2 features, an error is raised.
            - If the target tensor does not have one-hot encoded labels, an error is raised.
            - If the input data is not provided, a random sample from the training data is used.
        Raises:
            ValueError: If the input tensors are not valid
            ValueError: If the input tensors have incompatible shapes
            ValueError: If the input tensor does not have 2 features
            TypeError: If the input tensors are not torch.Tensor objects
        Returns:
            None
        """
        self.logger.trace(
            "CascadeCorrelationNetwork: plot_decision_boundary: Starting to plot the decision boundary."
        )

        # Verifying that Solution Plotting is enabled
        if not self.get_generate_plots():
            self.logger.warning(
                "CascadeCorrelation: plot_decision_boundary: Plotting solutions is disabled. Set 'generate_plots' to True to enable plotting."
            )
            return
        self.logger.debug(
            "CascadeCorrelation: plot_decision_boundary: Plotting Solutions is enabled. Proceeding to plot decision boundary."
        )

        # Validate input and target tensors
        self.logger.debug(
            "CascadeCorrelation: plot_decision_boundary: Verifying input and target tensors for plotting decision boundary."
        )
        if x is None or y is None:
            raise ValueError(
                "CascadeCorrelation: plot_decision_boundary: Input (x) and target (y) tensors must be provided for plotting the decision boundary."
            )
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError(
                "CascadeCorrelation: plot_decision_boundary: Input (x) and target (y) must be torch.Tensor objects."
            )
        if x.shape[1] != 2:
            raise ValueError(
                "CascadeCorrelation: plot_decision_boundary: Input tensor must have 2 features."
            )
        self.logger.debug(
            "CascadeCorrelation: plot_decision_boundary: Successfully Verified input and target tensors for plotting decision boundary."
        )
        self.logger.debug(
            f"CascadeCorrelation: plot_decision_boundary: Plotting decision boundary for input shape: {x.shape}, target shape: {y.shape}"
        )

        # Convert to numpy for plotting
        self.logger.debug(
            "CascadeCorrelation: plot_decision_boundary: Converting input and target tensors to numpy arrays for plotting."
        )
        x_np = x.numpy()
        y_np = torch.argmax(y, dim=1).numpy()

        # Create a mesh grid
        self.logger.debug(
            "CascadeCorrelation: plot_decision_boundary: Creating mesh grid for plotting decision boundary."
        )
        h = 0.02  # step size in the mesh
        x_min, x_max = x_np[:, 0].min() - 1, x_np[:, 0].max() + 1
        y_min, y_max = x_np[:, 1].min() - 1, x_np[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict class labels for each point in the mesh
        self.logger.debug(
            "CascadeCorrelation: plot_decision_boundary: Predicting class labels for each point in the mesh grid."
        )
        self.logger.debug(
            f"CascadeCorrelation: plot_decision_boundary: Mesh grid shape: {xx.shape}, {yy.shape}"
        )
        Z = self.predict_classes(
            torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        ).numpy()
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        self.logger.debug(
            "CascadeCorrelation: plot_decision_boundary: Plotting the decision boundary."
        )
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8)

        # Plot the training points
        self.logger.debug(
            "CascadeCorrelation: plot_decision_boundary: Plotting the training points."
        )
        for i in range(len(np.unique(y_np))):
            plt.scatter(x_np[y_np == i, 0], x_np[y_np == i, 1], label=f"Class {i}")
        self._plot_headings(plot=plt, title=title, x_label="X1", y_label="Y1")
        plt.show()
        self.logger.trace(
            "CascadeCorrelationNetwork: plot_decision_boundary: Completed plotting the decision boundary."
        )

    #################################################################################################################################################################################################
    # Public Method to plot the training history of the network. This method visualizes the training history including loss, accuracy, number of hidden units, and correlation of added units. It uses matplotlib to create subplots for each metric
    def plot_training_history(self):
        """
        Description:
            Plot the training history of the network.
        Args:
            self: The instance of the class.
        Notes:
            - This method visualizes the training history including loss, accuracy, number of hidden units, and correlation of added units.
            - It uses matplotlib to create subplots for each metric.
            - The training history is stored in the `history` attribute of the class.
            - If the training history is empty, an error is raised.
        Raises:
            ValueError: If the training history is empty.
            TypeError: If the training history is not a dictionary.
        Returns:
            None
        """
        self.logger.trace(
            "CascadeCorrelationNetwork: plot_training_history: Starting to plot the training history."
        )
        plt.figure(figsize=(15, 10))
        self.logger.debug(
            f"CascadeCorrelationNetwork: plot_training_history: History: {self.history}"
        )

        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        if "value_loss" in self.history and self.history["value_loss"]:
            plt.plot(self.history["value_loss"], label="Validation Loss")
        self._plot_headings(
            plot=plt, title="Loss During Training", x_label="Epochs", y_label="Loss"
        )

        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.history["train_accuracy"], label="Train Accuracy")
        if "value_accuracy" in self.history and self.history["value_accuracy"]:
            plt.plot(self.history["value_accuracy"], label="Validation Accuracy")
        self._plot_headings(
            plot=plt,
            title="Accuracy During Training",
            x_label="Epochs",
            y_label="Accuracy",
        )

        # Plot number of hidden units
        plt.subplot(2, 2, 3)
        plt.plot(
            range(len(self.history["hidden_units_added"]) + 1),
            [0] + [i + 1 for i in range(len(self.history["hidden_units_added"]))],
        )
        self._plot_headings(
            plot=plt,
            title="Number of Hidden Units",
            x_label="Epochs",
            y_label="Number of Units",
            legend=False,
        )

        # Plot correlation of added units
        if self.history["hidden_units_added"]:
            plt.subplot(2, 2, 4)
            correlations = [
                unit["correlation"] for unit in self.history["hidden_units_added"]
            ]
            plt.plot(correlations)
            self._plot_headings(
                plot=plt,
                title="Correlation of Added Units",
                x_label="Unit Number",
                y_label="Correlation",
                legend=False,
            )

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()
        self.logger.trace(
            "CascadeCorrelationNetwork: plot_training_history: Completed plotting the training history."
        )

    #################################################################################################################################################################################################
    # Define private methods for the CascadeCorrelationNetwork class
    #################################################################################################################################################################################################
    # Private Method to set headings for the plot This method sets the title, x-label, y-label, and legend for the plot
    def _plot_headings(
        self,
        plot: plt = None,
        title: str = "Headings",
        x_label: str = "X axis",
        y_label: str = "Y axis",
        legend: bool = True,
    ) -> None:
        """
        Description:
            Set headings for the plot.
        Args:
            plot: Matplotlib plot object
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            legend: Whether to display legend
        Notes:
            - This method sets the title, x-label, y-label, and legend for the plot
            - If the plot object is not provided, it defaults to matplotlib.pyplot
            - If the title, x_label, or y_label are not strings, an error is raised
            - If the plot object does not have title, xlabel, or ylabel methods, an error is raised
        Raises:
            ValueError: If the plot object is not valid or if title, x_label, or y_label are not strings
        Returns:
            None
        """
        self.logger.trace(
            "CascadeCorrelationNetwork: _plot_headings: Starting to set plot headings."
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: _plot_headings: Setting plot headings: Title: {title}, X Label: {x_label}, Y Label: {y_label}, Legend: {legend}"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: _plot_headings: Plot type: {type(plot)}, Plot: {plot}"
        )

        # Validate inputs
        if plot is None:
            plot = plt
        if not isinstance(plot, type(matplotlib.pyplot)):
            raise ValueError(
                "CascadeCorrelationNetwork: _plot_headings: plot must be a matplotlib.pyplot object"
            )
        if not isinstance(title, str):
            raise ValueError(
                "CascadeCorrelationNetwork: _plot_headings: title must be a string"
            )
        if not isinstance(x_label, str):
            raise ValueError(
                "CascadeCorrelationNetwork: _plot_headings: x_label must be a string"
            )
        if not isinstance(y_label, str):
            raise ValueError(
                "CascadeCorrelationNetwork: _plot_headings: y_label must be a string"
            )
        if (
            not hasattr(plot, "title")
            or not hasattr(plot, "xlabel")
            or not hasattr(plot, "ylabel")
        ):
            raise ValueError(
                "CascadeCorrelationNetwork: _plot_headings: plot must have title, xlabel, and ylabel methods"
            )

        # Set plot headings
        plot.title(title)
        plot.xlabel(x_label)
        plot.ylabel(y_label)

        # Display legend if requested
        if legend:
            plot.legend()
        self.logger.debug(
            f"CascadeCorrelationNetwork: _plot_headings: Plot headings set: Title: {title}, X Label: {x_label}, Y Label: {y_label}, Legend: {legend}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: _plot_headings: Completed setting plot headings."
        )

    #################################################################################################################################################################################################
    # Define private method to generate a new uuid for the CascadeCorrelationNetwork class
    def _generate_uuid(self):
        """
        Description:
            This method is used to generate a new UUID for the CascadeCorrelationNetwork class.
        Args:
            self: The instance of the class.
        Notes:
            - This method uses the uuid4 function from the uuid module to generate a new UUID.
            - The generated UUID is stored in the `uuid` attribute of the class.
            - The generated UUID is then returned.
        Returns:
            str: The generated UUID.
        """
        self.logger.trace(
            "CascadeCorrelationNetwork: _generate_uuid: Inside the CascadeCorrelationNetwork class Generate UUID method"
        )
        new_uuid = str(uuid.uuid4())
        self.logger.debug(
            f"CascadeCorrelationNetwork: _generate_uuid: UUID: {new_uuid}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: _generate_uuid: Completed the CascadeCorrelationNetwork class Generate UUID method"
        )
        return new_uuid

    ####################################################################################################################################
    # Define CascadeCorrelationNetwork class Setters
    ####################################################################################################################################
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
        self.learning_rate = learning_rate

    def set_max_hidden_units(self, max_hidden_units: int = None):
        self.max_hidden_units = max_hidden_units

    def set_output_bias(self, output_bias: float = None):
        self.output_bias = output_bias

    def set_output_epochs(self, output_epochs: int = None):
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
        self.logger.trace(
            "CascadeCorrelationNetwork: set_uuid: Starting to set UUID for CascadeCorrelationNetwork class"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: set_uuid: Setting UUID to: {uuid}"
        )
        if not hasattr(self, "uuid") or self.uuid is None:
            self.uuid = (uuid, self._generate_uuid())[
                uuid is None
            ]  # Generate a new UUID if none is provided
        else:
            self.logger.fatal(
                f"CascadeCorrelationNetwork: set_uuid: Fatal Error: UUID already set: {self.uuid}. Changing UUID is bad Juju.  Exiting..."
            )
            os._exit(1)
        self.logger.debug(
            f"CascadeCorrelationNetwork: set_uuid: UUID set to: {self.uuid}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: set_uuid: Completed setting UUID for CascadeCorrelationNetwork class"
        )

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
            - If the UUID is not set, it will generate a new UUID using the `set_uuid` method.
            - The generated UUID is then returned.
        Returns:
            str: The UUID for the CascadeCorrelationNetwork class.
        """
        self.logger.trace(
            "CascadeCorrelationNetwork: get_uuid: Starting to get UUID for CascadeCorrelationNetwork class"
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: get_uuid: Current UUID: {getattr(self, 'uuid', None)}"
        )

        # Ensure UUID is set:  if not, generate a new one
        if not hasattr(self, "uuid"):
            self.set_uuid()  # Ensure UUID is set if not already
            self.logger.debug(
                "CascadeCorrelationNetwork: get_uuid: UUID was not set, generated a new one."
            )

        # Return the UUID
        self.logger.debug(
            f"CascadeCorrelationNetwork: get_uuid: Returning UUID: {self.uuid}"
        )
        self.logger.trace(
            "CascadeCorrelationNetwork: get_uuid: Completed getting UUID for CascadeCorrelationNetwork class"
        )
        return self.uuid

    def get_activation_fn(self):
        return self.activation_fn if hasattr(self, "activation_fn") else None

    def get_activation_fn_no_diff(self):
        return (
            self.activation_fn_no_diff
            if hasattr(self, "activation_fn_no_diff")
            else None
        )

    def get_candidate_epochs(self):
        return self.candidate_epochs if hasattr(self, "candidate_epochs") else None

    def get_candidate_pool_size(self):
        return (
            self.candidate_pool_size if hasattr(self, "candidate_pool_size") else None
        )

    def get_candidate_unit(self) -> CandidateUnit:
        return self.candidate_unit if hasattr(self, "candidate_unit") else None

    def get_correlation_threshold(self):
        return (
            self.correlation_threshold
            if hasattr(self, "correlation_threshold")
            else None
        )

    def get_display_frequency_epoch(self):
        return (
            self.display_frequency_epoch if hasattr(self, "display_frequency_epoch") else None

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
