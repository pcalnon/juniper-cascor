#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     constants.py
# Author:        Paul Calnon
# Version:       0.3.1 (0.7.3)
#
# Date Created:  2025-06-11
# Last Modified: 2026-01-12
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This file contains constants used in the Cascade Correlation Neural Network implementation.
#
#####################################################################################################################################################################################################
# Notes:
#    - This file defines constants for the Cascade Correlation Neural Network, Spiral Problem, and related classes.
#    - Constants are organized into sections for better readability and maintainability.
#
#####################################################################################################################################################################################################
# References:
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################
import os
import pathlib
import sys

#####################################################################################################################################################################################################
# Add current and parent dir to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# from constants.constants_activation.constants_activation import (
#     _PROJECT_MODEL_INPUT_SIZE
# )

from constants.constants_activation.constants_activation import (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_ELU,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_GELU,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_HARDTANH,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_IDENTITY,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_LEAKY_RELU,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_RELU,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SELU,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SIGMOID,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SOFTMAX,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SOFTPLUS,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SOFTSHRINK,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_TANH,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_TANHSHRINK,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_RELU,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_SIGMOID,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_TANH,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTIONS_LIST,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTIONS_NAME_LIST,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_ACTIVATION_FUNCTION,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_DEFAULT,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_ELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_GELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_HARDTANH,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_IDENTITY,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_LEAKY_RELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_RELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SIGMOID,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SOFTMAX,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SOFTPLUS,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SOFTSHRINK,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_TANH,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_TANHSHRINK,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_RELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_SIGMOID,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_TANH,
    _PROJECT_MODEL_ACTIVATION_FUNCTIONS_DICT,
)
from constants.constants_candidates.constants_candidates import (
    _PROJECT_MODEL_CANDIDATE_DISPLAY_FREQUENCY,
    _PROJECT_MODEL_CANDIDATE_EARLY_STOPPING,
    _PROJECT_MODEL_CANDIDATE_EPOCHS,
    _PROJECT_MODEL_CANDIDATE_PATIENCE,
    _PROJECT_MODEL_CANDIDATE_POOL_SIZE,
    _PROJECT_MODEL_CANDIDATE_UNIT_LEARNING_RATE,
    _PROJECT_MODEL_DISPLAY_FREQUENCY,
)
from constants.constants_hdf5.constants_hdf5 import (
    _HDF5_PROJECT_CONSTANTS_DIR,  # trunk-ignore(ruff/F401)
    _HDF5_PROJECT_DIR,  # trunk-ignore(ruff/F401)
    _HDF5_PROJECT_HDF5_CONSTANTS_DIR,  # trunk-ignore(ruff/F401)
    _HDF5_PROJECT_SOURCE_DIR,  # trunk-ignore(ruff/F401)
    _HDF5_PROJECT_SNAPSHOTS_DIR,
)
from constants.constants_logging.constants_logging import (
    _LOGGER_CLASS_LOGGER_LOG_LEVEL_DEFAULT,  # trunk-ignore(ruff/F401)
    _LOGGER_CONTENT_FIELD_NAMES_CONSOLE,  # trunk-ignore(ruff/F401)
    _LOGGER_CONTENT_FIELD_NAMES_MESSAGE,
    _LOGGER_CONTENT_FIELD_NAMES_FILE,
    _LOGGER_CONTENT_FIELD_NAMES_LEVELNAME,  # trunk-ignore(ruff/F401)
    _LOGGER_CONTENT_FIELDS_CONSOLE,  # trunk-ignore(ruff/F401)
    _LOGGER_CONTENT_FIELDS_FILE,  # trunk-ignore(ruff/F401)
    _LOGGER_CONTENT_FORMAT_CONSOLE,  # trunk-ignore(ruff/F401)
    _LOGGER_CONTENT_FORMAT_FILE,  # trunk-ignore(ruff/F401)
    _LOGGER_CONTENT_LEVELNAME,  # trunk-ignore(ruff/F401)
    _LOGGER_CONTENT_MESSAGE,  # trunk-ignore(ruff/F401)
    _LOGGER_CONTENT_NAME,  # trunk-ignore(ruff/F401)
    _LOGGER_DATA_FIELD_NAMES,  # trunk-ignore(ruff/F401)
    _LOGGER_DESTINATION_NAME_CONSOLE,  # trunk-ignore(ruff/F401)
    _LOGGER_DESTINATION_NAME_FILE,  # trunk-ignore(ruff/F401)
    _LOGGER_FORMAT_FIELD_NAMES,  # trunk-ignore(ruff/F401)
    _LOGGER_LOG_FORMATTER_STRING_CONSOLE_CONTENT,  # trunk-ignore(ruff/F401)
    _LOGGER_LOG_FORMATTER_STRING_CONSOLE_PREFIX,  # trunk-ignore(ruff/F401)
    _LOGGER_LOG_FORMATTER_STRING_FILE_CONTENT,  # trunk-ignore(ruff/F401)
    _LOGGER_LOG_FORMATTER_STRING_FILE_PREFIX,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_FIELD_NAME_ASCTIME,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_FIELD_NAME_FILENAME,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_FIELD_NAME_FUNCTION_NAME,
    _LOGGER_PREFIX_FIELD_NAME_LINE_NUMBER,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_FIELD_NAMES_CONSOLE,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_FIELD_NAMES_FILE,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_FIELDS_CONSOLE,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_FIELDS_FILE,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_FORMAT_CONSOLE,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_FORMAT_FILE,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_FRAME_FILE_NAME,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_FRAME_FUNC_NAME,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_FRAME_LINE_NAME,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_FUNCTION_NAME,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_LINE_NUMBER,  # trunk-ignore(ruff/F401)
    _LOGGER_PREFIX_NAME,  # trunk-ignore(ruff/F401)
    _LOGGER_LOG_FORMATTER_STRING_CONSOLE,
    _LOGGER_LOG_FORMATTER_STRING_FILE,
)
from constants.constants_model.constants_model import (
    _PROJECT_MODEL_OUTPUT_SIZE,  # trunk-ignore(ruff/F401)
    _PROJECT_MODEL_AUTHKEY,
    _PROJECT_MODEL_BASE_MANAGER_ADDRESS,
    _PROJECT_MODEL_BASE_MANAGER_ADDRESS_IP,
    _PROJECT_MODEL_BASE_MANAGER_ADDRESS_PORT,
    _PROJECT_MODEL_CANDIDATE_TRAINING_CONTEXT,
    _PROJECT_MODEL_CORRELATION_THRESHOLD,
    _PROJECT_MODEL_EPOCH_DISPLAY_FREQUENCY,
    _PROJECT_MODEL_EPOCHS_MAX,
    _PROJECT_MODEL_INPUT_SIZE,
    _PROJECT_MODEL_LEARNING_RATE,
    _PROJECT_MODEL_MAX_HIDDEN_UNITS,
    _PROJECT_MODEL_OUTPUT_EPOCHS,
    _PROJECT_MODEL_PATIENCE,
    _PROJECT_MODEL_SHUTDOWN_TIMEOUT,
    _PROJECT_MODEL_STATUS_DISPLAY_FREQUENCY,
    _PROJECT_MODEL_TARGET_ACCURACY,
    _PROJECT_MODEL_TASK_QUEUE_TIMEOUT,
    _PROJECT_MODEL_WORKER_STANDBY_SLEEPYTIME,
)
from constants.constants_problem.constants_problem import (
    _SPIRAL_PROBLEM_CLOCKWISE,
    _SPIRAL_PROBLEM_DEFAULT_ORIGIN,
    _SPIRAL_PROBLEM_DEFAULT_RADIUS,
    _SPIRAL_PROBLEM_DISTRIBUTION_FACTOR,
    _SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT,
    _SPIRAL_PROBLEM_MAX_NEW,
    _SPIRAL_PROBLEM_MAX_ORIG,
    _SPIRAL_PROBLEM_MIN_NEW,
    _SPIRAL_PROBLEM_MIN_ORIG,
    _SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT,
    _SPIRAL_PROBLEM_NUM_ROTATIONS,
    _SPIRAL_PROBLEM_NUM_SPIRALS,
    _SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL,
    _SPIRAL_PROBLEM_ORIG_POINTS,
    _SPIRAL_PROBLEM_OUTPUT_SIZE,
    _SPIRAL_PROBLEM_RANDOM_VALUE_SCALE,
    _SPIRAL_PROBLEM_TEST_RATIO,
    _SPIRAL_PROBLEM_TRAIN_RATIO,
)

# ##########################################################################################################
# # Define Plotting class Constants to provide reasonable defaults
# # Define SpiralDataset Example Constants for logging
# _SPIRAL_DATASET_PLOTTING_LOG_DATE_FORMAT = _CASCOR_SPIRAL_DATASET_LOG_DATE_FORMAT
# _SPIRAL_DATASET_PLOTTING_LOG_FORMATTER_STRING = _CASCOR_SPIRAL_DATASET_LOG_FORMATTER_STRING
# _SPIRAL_DATASET_PLOTTING_LOG_LEVEL_DEFAULT = logging.INFO
# # _SPIRAL_DATASET_PLOTTING_LOG_LEVEL_DEFAULT = logging.DEBUG
# _SPIRAL_DATASET_PLOTTING_LOG_FILE_PATH_DEFAULT  = _CASCOR_SPIRAL_DATASET_LOG_FILE_PATH_DEFAULT
# #####################################################################################################################################################################################################
# # Define constants for the Cascade Correlation Network
# # Define SpiralDataset Example Constants for logging
# _CASCADE_CORRELATION_NETWORK_LOG_NAME = _CASCOR_SPIRAL_DATASET_LOG_NAME
# _CASCADE_CORRELATION_NETWORK_LOG_DATE_FORMAT = _CASCOR_SPIRAL_DATASET_LOG_DATE_FORMAT
# _CASCADE_CORRELATION_NETWORK_LOG_FORMATTER_STRING = _CASCOR_SPIRAL_DATASET_LOG_FORMATTER_STRING
# _CASCADE_CORRELATION_NETWORK_LOG_DIR = _CASCOR_SPIRAL_DATASET_LOG_DIR_DEFAULT
# _CASCADE_CORRELATION_NETWORK_LOG_FILE_PATH = _CASCOR_SPIRAL_DATASET_LOG_FILE_PATH
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL = _CASCOR_SPIRAL_DATASET_LOG_LEVEL_DEFAULT
# _CASCADE_CORRELATION_NETWORK_PLOT_SOLUTIONS = _CASCOR_SPIRAL_PLOT_SOLUTIONS
# _CASCADE_CORRELATION_NETWORK_INPUT_SIZE = _CASCOR_SPIRAL_DATASET_INPUT_SIZE
# _CASCADE_CORRELATION_NETWORK_OUTPUT_SIZE = _CASCOR_SPIRAL_DATASET_OUTPUT_SIZE
# _CASCADE_CORRELATION_NETWORK_CANDIDATE_POOL_SIZE = _CASCOR_SPIRAL_CANDIDATE_POOL_SIZE
# _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION = _CASCOR_SPIRAL_ACTIVATION_FUNCTION
# _CASCADE_CORRELATION_NETWORK_LEARNING_RATE = _CASCOR_SPIRAL_LEARNING_RATE
# _CASCADE_CORRELATION_NETWORK_MAX_HIDDEN_UNITS = _CASCOR_SPIRAL_MAX_HIDDEN_UNITS
# # _CASCADE_CORRELATION_NETWORK_CORRELATION_THRESHOLD = 0.4
# # _CASCADE_CORRELATION_NETWORK_CORRELATION_THRESHOLD = 0.125
# # _CASCADE_CORRELATION_NETWORK_CORRELATION_THRESHOLD = 0.05
# _CASCADE_CORRELATION_NETWORK_CORRELATION_THRESHOLD = _CASCOR_SPIRAL_CORRELATION_THRESHOLD
# # _CASCADE_CORRELATION_NETWORK_PATIENCE = 5
# # _CASCADE_CORRELATION_NETWORK_PATIENCE =10
# _CASCADE_CORRELATION_NETWORK_PATIENCE = _CASCOR_SPIRAL_PATIENCE
# _CASCADE_CORRELATION_NETWORK_CANDIDATE_EPOCHS = _CASCOR_SPIRAL_CANDIDATE_EPOCHS
# _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS = _CASCOR_SPIRAL_OUTPUT_EPOCHS
# #####################################################################################################################################################################################################
# # Define constants for the Cascade Correlation Candidate Unit class, Define SpiralDataset Example Constants for logging
# _CANDIDATE_UNIT_LOG_NAME = _CASCOR_SPIRAL_DATASET_LOG_NAME
# _CANDIDATE_UNIT_LOG_DATE_FORMAT = _CASCOR_SPIRAL_DATASET_LOG_DATE_FORMAT
# _CANDIDATE_UNIT_LOG_FORMATTER_STRING = _CASCOR_SPIRAL_DATASET_LOG_FORMATTER_STRING
# _CANDIDATE_UNIT_LOG_DIR = _CASCOR_SPIRAL_DATASET_LOG_DIR_DEFAULT
# _CANDIDATE_UNIT_LOG_FILE_PATH = _CASCOR_SPIRAL_DATASET_LOG_FILE_PATH
# _CANDIDATE_UNIT_LOG_LEVEL_DEFAULT = _CASCOR_SPIRAL_DATASET_LOG_LEVEL_DEFAULT
# _CANDIDATE_UNIT_INPUT_SIZE = _CASCOR_SPIRAL_DATASET_INPUT_SIZE
# _CANDIDATE_UNIT_OUTPUT_SIZE = _CASCOR_SPIRAL_DATASET_OUTPUT_SIZE
# _CANDIDATE_UNIT_ACTIVATION_FUNCTION = torch.tanh
# _CANDIDATE_UNIT_LEARNING_RATE_DEFAULT = _CANDIDATE_UNIT_LEARNING_RATE
# _CANDIDATE_UNIT_EPOCHS_DEFAULT = _CANDIDATE_UNIT_EPOCHS
# #####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Define the project directory structure
#####################################################################################################################################################################################################
_PROJECT_CONSTANTS_DIR = pathlib.Path(__file__).parent.resolve()
_PROJECT_SOURCE_DIR = _PROJECT_CONSTANTS_DIR.parent.resolve()
_PROJECT_DIR = _PROJECT_SOURCE_DIR.parent.resolve()

_PROJECT_LOG_DIR_NAME_DEFAULT = "logs"
_PROJECT_LOG_DIR_DEFAULT = pathlib.Path(_PROJECT_DIR, _PROJECT_LOG_DIR_NAME_DEFAULT)

_PROJECT_CONFIG_DIR_NAME_DEFAULT = "conf"
_PROJECT_CONFIG_DIR_DEFAULT = pathlib.Path(
    _PROJECT_DIR, _PROJECT_CONFIG_DIR_NAME_DEFAULT
)

_LOGGER_NAME = "juniper"


#####################################################################################################################################################################################################
# Define Global Constants for current Project
_PROJECT_RANDOM_SEED = 42
_PROJECT_RANDOM_MAX_VALUE = 2**32 - 1
_PROJECT_SEQUENCE_MAX_VALUE = 10

_PROJECT_GENERATE_PLOTS_DEFAULT = True
# _PROJECT_GENERATE_PLOTS_DEFAULT = False

_PROJECT_RESIDUAL_ERROR_DEFAULT = float(
    "inf"
)  # Default value for residual error in CandidateUnit

_PROJECT_CANDIDATE_TRAINING_CONTEXT = _PROJECT_MODEL_CANDIDATE_TRAINING_CONTEXT


#####################################################################################################################################################################################################
# Define the default logging configuration file and its path
_PROJECT_LOG_CONFIG_DIR_NAME_DEFAULT = _PROJECT_CONFIG_DIR_NAME_DEFAULT
_PROJECT_LOG_CONFIG_DIR_DEFAULT = _PROJECT_CONFIG_DIR_DEFAULT

_PROJECT_LOG_LOGGING_CONFIGURED_DEFAULT = False

_PROJECT_LOG_FORMATTER_STRING_FILE = _LOGGER_LOG_FORMATTER_STRING_FILE
_PROJECT_LOG_FORMATTER_STRING_CONSOLE = _LOGGER_LOG_FORMATTER_STRING_CONSOLE

_PROJECT_LOG_FORMATTER_STRING = _PROJECT_LOG_FORMATTER_STRING_CONSOLE
_PROJECT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_PROJECT_LOG_CONFIG_FILE_NAME = "logging_config.yaml"
# _PROJECT_LOG_CONFIG_FILE_PATH = pathlib.Path(_PROJECT_LOG_CONFIG_DIR_DEFAULT, _PROJECT_LOG_CONFIG_FILE_NAME)
_PROJECT_LOG_CONFIG_FILE_PATH = _PROJECT_LOG_CONFIG_DIR_DEFAULT
_PROJECT_LOG_MESSAGE_DEFAULT = "Default Message: Log Config Initialized"

_PROJECT_LOG_FILE_EXT = ".log"
_PROJECT_LOG_FILE_NAME_ROOT = "juniper_cascor"
_PROJECT_LOG_FILE_NAME = f"{_PROJECT_LOG_FILE_NAME_ROOT}{_PROJECT_LOG_FILE_EXT}"
# _PROJECT_LOG_FILE_PATH = pathlib.Path(_PROJECT_LOG_DIR_DEFAULT, _PROJECT_LOG_FILE_NAME)
_PROJECT_LOG_FILE_PATH = _PROJECT_LOG_DIR_DEFAULT

# Logger name for the project (used in YAML config)
# _PROJECT_LOGGER_NAME = "juniper"
# _PROJECT_LOG_LEVEL_REDEFINITION = False  # Do Not Allow redefining existing log levels
_PROJECT_LOG_LEVEL_REDEFINITION = True  # Allow redefining existing log levels


#####################################################################################################################################################################################################
# Define Testing status constants for the Juniper Cascor Prototype
#####################################################################################################################################################################################################
_PROJECT_TESTING_SKIPPED_TEST = "🚫"
_PROJECT_TESTING_FAILED_TEST = "❌"
_PROJECT_TESTING_PASSED_TEST = "✅"
_PROJECT_TESTING_UNKNOWN_TEST = "❓"
_PROJECT_TESTING_UNSTABLE_TEST = "✗"
_PROJECT_TESTING_PARTIAL_TEST = "✔"
_PROJECT_TESTING_SUCCESSFUL_TEST = "🎉"


#####################################################################################################################################################################################################
# Define the default logging level, name, and configuration
#############################################################################################S########################################################################################################

#############################################################################################S########################################################################################################
# Define the default logging level names
_PROJECT_LOG_LEVEL_NAME_FATAL = "FATAL"  # Custom level for fatal errors
_PROJECT_LOG_LEVEL_NAME_CRITICAL = "CRITICAL"
_PROJECT_LOG_LEVEL_NAME_ERROR = "ERROR"
_PROJECT_LOG_LEVEL_NAME_WARNING = "WARNING"
_PROJECT_LOG_LEVEL_NAME_INFO = "INFO"
_PROJECT_LOG_LEVEL_NAME_DEBUG = "DEBUG"
_PROJECT_LOG_LEVEL_NAME_VERBOSE = "VERBOSE"  # Custom level for verbose logging
_PROJECT_LOG_LEVEL_NAME_TRACE = "TRACE"  # Custom level for trace logging


#############################################################################################S########################################################################################################
# Define the logging level names list
_PROJECT_LOG_LEVEL_NAMES_LIST = [
    _PROJECT_LOG_LEVEL_NAME_FATAL,  # Custom level for fatal errors
    _PROJECT_LOG_LEVEL_NAME_CRITICAL,
    _PROJECT_LOG_LEVEL_NAME_ERROR,
    _PROJECT_LOG_LEVEL_NAME_WARNING,
    _PROJECT_LOG_LEVEL_NAME_INFO,
    _PROJECT_LOG_LEVEL_NAME_DEBUG,
    _PROJECT_LOG_LEVEL_NAME_VERBOSE,  # Custom level for verbose logging
    _PROJECT_LOG_LEVEL_NAME_TRACE,  # Custom level for trace logging
]


#############################################################################################S########################################################################################################
# Define level number constants for all logging levels
_PROJECT_LOG_LEVEL_NUMBER_FATAL = 60  # Custom level for fatal errors
_PROJECT_LOG_LEVEL_NUMBER_CRITICAL = 50
_PROJECT_LOG_LEVEL_NUMBER_ERROR = 40
_PROJECT_LOG_LEVEL_NUMBER_WARNING = 30
_PROJECT_LOG_LEVEL_NUMBER_INFO = 20
_PROJECT_LOG_LEVEL_NUMBER_DEBUG = 10
_PROJECT_LOG_LEVEL_NUMBER_VERBOSE = 5  # Custom level for verbose logging
_PROJECT_LOG_LEVEL_NUMBER_TRACE = 1  # Custom level for trace logging


#############################################################################################S########################################################################################################
# Define the logging level number list
_PROJECT_LOG_LEVEL_NUMBERS_LIST = [
    _PROJECT_LOG_LEVEL_NUMBER_FATAL,
    _PROJECT_LOG_LEVEL_NUMBER_CRITICAL,
    _PROJECT_LOG_LEVEL_NUMBER_ERROR,
    _PROJECT_LOG_LEVEL_NUMBER_WARNING,
    _PROJECT_LOG_LEVEL_NUMBER_INFO,
    _PROJECT_LOG_LEVEL_NUMBER_DEBUG,
    _PROJECT_LOG_LEVEL_NUMBER_VERBOSE,
    _PROJECT_LOG_LEVEL_NUMBER_TRACE,
]
#############################################################################################S########################################################################################################


#############################################################################################S########################################################################################################
# Define project logging data structure to enable custom logging levels
#############################################################################################S########################################################################################################
# Define the logging level methods list
_PROJECT_LOG_LEVEL_METHODS_LIST = [m.lower() for m in _PROJECT_LOG_LEVEL_NAMES_LIST]


#####################################################################################################################################################################################################
# Define the logging levels dictionaries
_PROJECT_LOG_LEVEL_METHODS_DICT = dict(
    zip(_PROJECT_LOG_LEVEL_NAMES_LIST, _PROJECT_LOG_LEVEL_METHODS_LIST, strict=False)
)
_PROJECT_LOG_LEVEL_NUMBERS_DICT = dict(
    zip(_PROJECT_LOG_LEVEL_NAMES_LIST, _PROJECT_LOG_LEVEL_NUMBERS_LIST, strict=False)
)


#####################################################################################################################################################################################################
# Define the custom logging level lists
_PROJECT_LOG_LEVEL_CUSTOM_NAMES_LIST = [
    _PROJECT_LOG_LEVEL_NAME_TRACE,
    _PROJECT_LOG_LEVEL_NAME_VERBOSE,
    _PROJECT_LOG_LEVEL_NAME_FATAL,
]

# _PROJECT_LOG_LEVEL_CUSTOM_METHODS_LIST = [_PROJECT_LOG_LEVEL_METHODS_DICT[n] for n in _PROJECT_LOG_LEVEL_CUSTOM_NAMES_LIST]
# _PROJECT_LOG_LEVEL_CUSTOM_NUMBERS_LIST = [_PROJECT_LOG_LEVEL_NUMBERS_DICT[n] for n in _PROJECT_LOG_LEVEL_CUSTOM_NAMES_LIST]


#####################################################################################################################################################################################################
# Define the default logging level Name for the Project.
#     NOTE: This must be set to an existing predefined logging level because it is used while logging is being initialized and configured and before custom logging levels are added.
# _PROJECT_LOG_LEVEL_NAME_DEFAULT = _PROJECT_LOG_LEVEL_NAME_FATAL #     NOTE: This must be set to an existing predefined logging level
# _PROJECT_LOG_LEVEL_NAME_DEFAULT = _PROJECT_LOG_LEVEL_NAME_CRITICAL #  NOTE: This must be set to an existing predefined logging level
# _PROJECT_LOG_LEVEL_NAME_DEFAULT = _PROJECT_LOG_LEVEL_NAME_ERROR #     NOTE: This must be set to an existing predefined logging level
# _PROJECT_LOG_LEVEL_NAME_DEFAULT = _PROJECT_LOG_LEVEL_NAME_WARNING #   NOTE: This must be set to an existing predefined logging level
_PROJECT_LOG_LEVEL_NAME_DEFAULT = _PROJECT_LOG_LEVEL_NAME_INFO  #      NOTE: This must be set to an existing predefined logging level
# _PROJECT_LOG_LEVEL_NAME_DEFAULT = _PROJECT_LOG_LEVEL_NAME_DEBUG #       NOTE: This must be set to an existing predefined logging level

_PROJECT_LOG_LEVEL_DEFAULT = _PROJECT_LOG_LEVEL_NUMBERS_DICT[
    _PROJECT_LOG_LEVEL_NAME_DEFAULT
]  # NOTE: This must be set to an existing predefined logging level


#####################################################################################################################################################################################################
# Define the default logging level for the Project.  This is the log level used when logging is being initialized and configured.
#     NOTE: This must be set to an existing predefined logging level
_PROJECT_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_DEFAULT  # NOTE: This must be set to an existing predefined logging level
#
_PROJECT_LOG_LEVEL_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NUMBERS_DICT[
    _PROJECT_LOG_LEVEL_NAME_LOGGING_CONFIG
]  # NOTE: This must be set to an existing predefined logging level
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Define constants for the Cascor class used to generate the Two Spiral Problem Dataset
#####################################################################################################################################################################################################
_CASCOR_ACTIVATION_FUNCTION = _PROJECT_MODEL_ACTIVATION_FUNCTION
_CASCOR_ACTIVATION_FUNCTION_DEFAULT = _PROJECT_MODEL_ACTIVATION_FUNCTION_DEFAULT
_CASCOR_ACTIVATION_FUNCTION_NAME = _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME
_CASCOR_ACTIVATION_FUNCTIONS_DICT = _PROJECT_MODEL_ACTIVATION_FUNCTIONS_DICT

_CASCOR_CANDIDATE_DISPLAY_FREQUENCY = _PROJECT_MODEL_CANDIDATE_DISPLAY_FREQUENCY
_CASCOR_CANDIDATE_EPOCHS = _PROJECT_MODEL_CANDIDATE_EPOCHS
_CASCOR_CANDIDATE_LEARNING_RATE = _PROJECT_MODEL_CANDIDATE_UNIT_LEARNING_RATE
_CASCOR_CANDIDATE_POOL_SIZE = _PROJECT_MODEL_CANDIDATE_POOL_SIZE
_CASCOR_CORRELATION_THRESHOLD = _PROJECT_MODEL_CORRELATION_THRESHOLD
_CASCOR_DEFAULT_ORIGIN = _SPIRAL_PROBLEM_DEFAULT_ORIGIN
_CASCOR_DEFAULT_RADIUS = _SPIRAL_PROBLEM_DEFAULT_RADIUS
_CASCOR_DISPLAY_FREQUENCY = _PROJECT_MODEL_DISPLAY_FREQUENCY
_CASCOR_DISTRIBUTION_FACTOR = _SPIRAL_PROBLEM_DISTRIBUTION_FACTOR
_CASCOR_EPOCH_DISPLAY_FREQUENCY = _PROJECT_MODEL_EPOCH_DISPLAY_FREQUENCY
_CASCOR_EPOCHS_MAX = _PROJECT_MODEL_EPOCHS_MAX
_CASCOR_GENERATE_PLOTS_DEFAULT = _SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT
_CASCOR_INPUT_SIZE = _PROJECT_MODEL_INPUT_SIZE
_CASCOR_LEARNING_RATE = _PROJECT_MODEL_LEARNING_RATE
_CASCOR_LOG_CONFIG_FILE_NAME = _PROJECT_LOG_CONFIG_FILE_NAME
_CASCOR_LOG_CONFIG_FILE_PATH = _PROJECT_LOG_CONFIG_FILE_PATH
_CASCOR_LOG_LEVEL_CUSTOM_NAMES_LIST = _PROJECT_LOG_LEVEL_CUSTOM_NAMES_LIST
_CASCOR_LOG_LEVEL_METHODS_DICT = _PROJECT_LOG_LEVEL_METHODS_DICT
_CASCOR_LOG_LEVEL_METHODS_LIST = _PROJECT_LOG_LEVEL_METHODS_LIST
_CASCOR_LOG_LEVEL_NAMES_LIST = _PROJECT_LOG_LEVEL_NAMES_LIST
_CASCOR_LOG_LEVEL_NUMBERS_DICT = _PROJECT_LOG_LEVEL_NUMBERS_DICT
_CASCOR_LOG_LEVEL_NUMBERS_LIST = _PROJECT_LOG_LEVEL_NUMBERS_LIST
_CASCOR_LOG_LEVEL_REDEFINITION = _PROJECT_LOG_LEVEL_REDEFINITION
_CASCOR_LOG_MESSAGE_DEFAULT = _PROJECT_LOG_MESSAGE_DEFAULT
_CASCOR_MAX_HIDDEN_UNITS = _PROJECT_MODEL_MAX_HIDDEN_UNITS
_CASCOR_MAX_NEW = _SPIRAL_PROBLEM_MAX_NEW
_CASCOR_MAX_ORIG = _SPIRAL_PROBLEM_MAX_ORIG
_CASCOR_MIN_NEW = _SPIRAL_PROBLEM_MIN_NEW
_CASCOR_MIN_ORIG = _SPIRAL_PROBLEM_MIN_ORIG
_CASCOR_NOISE_FACTOR_DEFAULT = _SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT
_CASCOR_NUM_ROTATIONS = _SPIRAL_PROBLEM_NUM_ROTATIONS
_CASCOR_NUM_SPIRALS = _SPIRAL_PROBLEM_NUM_SPIRALS
_CASCOR_CLOCKWISE = _SPIRAL_PROBLEM_CLOCKWISE
_CASCOR_NUMBER_POINTS_PER_SPIRAL = _SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL
_CASCOR_ORIG_POINTS = _SPIRAL_PROBLEM_ORIG_POINTS
_CASCOR_OUTPUT_EPOCHS = _PROJECT_MODEL_OUTPUT_EPOCHS
_CASCOR_OUTPUT_SIZE = _SPIRAL_PROBLEM_OUTPUT_SIZE
_CASCOR_PATIENCE = _PROJECT_MODEL_PATIENCE
_CASCOR_RANDOM_SEED = _PROJECT_RANDOM_SEED
_CASCOR_RANDOM_VALUE_SCALE = _SPIRAL_PROBLEM_RANDOM_VALUE_SCALE
_CASCOR_STATUS_DISPLAY_FREQUENCY = _PROJECT_MODEL_STATUS_DISPLAY_FREQUENCY
_CASCOR_TEST_RATIO = _SPIRAL_PROBLEM_TEST_RATIO
_CASCOR_TRAIN_RATIO = _SPIRAL_PROBLEM_TRAIN_RATIO


#####################################################################################################################################################################################################
# Define constants for Cascor class logging
_CASCOR_LOG_DATE_FORMAT = _PROJECT_LOG_DATE_FORMAT
_CASCOR_LOG_FILE_NAME = _PROJECT_LOG_FILE_NAME
_CASCOR_LOG_FILE_PATH = _PROJECT_LOG_FILE_PATH

_CASCOR_LOG_FORMATTER_STRING = _PROJECT_LOG_FORMATTER_STRING
_CASCOR_LOG_FORMATTER_STRING_CONSOLE = _PROJECT_LOG_FORMATTER_STRING_CONSOLE
_CASCOR_LOG_FORMATTER_STRING_FILE = _PROJECT_LOG_FORMATTER_STRING_FILE


#####################################################################################################################################################################################################
# Define the constants for the Cascor class during logging initialization and configuration
#     NOTE:  To adjust log level for the Cascor class during logging initialization and configuration, change the following line(s):
# _CASCOR_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_CRITICAL
# _CASCOR_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_ERROR
# _CASCOR_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_WARNING
# _CASCOR_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_INFO
# _CASCOR_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_DEBUG
#
_CASCOR_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_LOGGING_CONFIG
# _CASCOR_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_DEFAULT

_CASCOR_LOG_LEVEL_LOGGING_CONFIG = _CASCOR_LOG_LEVEL_NUMBERS_DICT[
    _CASCOR_LOG_LEVEL_NAME_LOGGING_CONFIG
]


#####################################################################################################################################################################################################
# Define the constants for the default logging level for the Cascor class
#     NOTE: To adjust log level for the Cascor class, change the following line(s):
# _CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_FATAL
# _CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_CRITICAL
# _CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_ERROR
# _CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_WARNING
# _CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_INFO
# _CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEBUG
# _CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_VERBOSE
# _CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_TRACE
#
_CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEFAULT  # WARNING: This default prevents other classes from defining custom log levels as defaults--unless explicitly set in these classes Logging Constants.

_CASCOR_LOG_LEVEL = _CASCOR_LOG_LEVEL_NUMBERS_DICT[_CASCOR_LOG_LEVEL_NAME]
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Define constants for the SpiralProblem class, constants_problem.py file
#####################################################################################################################################################################################################
_SPIRAL_PROBLEM_ACTIVATION_FUNCTION = _CASCOR_ACTIVATION_FUNCTION
_SPIRAL_PROBLEM_ACTIVATION_FUNCTION_DEFAULT = _CASCOR_ACTIVATION_FUNCTION_DEFAULT
_SPIRAL_PROBLEM_ACTIVATION_FUNCTION_NAME = _CASCOR_ACTIVATION_FUNCTION_NAME
_SPIRAL_PROBLEM_ACTIVATION_FUNCTIONS_DICT = _CASCOR_ACTIVATION_FUNCTIONS_DICT

_SPIRAL_PROBLEM_CANDIDATE_DISPLAY_FREQUENCY = _CASCOR_CANDIDATE_DISPLAY_FREQUENCY
_SPIRAL_PROBLEM_CANDIDATE_LEARNING_RATE = _CASCOR_CANDIDATE_LEARNING_RATE
_SPIRAL_PROBLEM_CANDIDATE_POOL_SIZE = _CASCOR_CANDIDATE_POOL_SIZE
_SPIRAL_PROBLEM_CANDIDATE_EPOCHS = _CASCOR_CANDIDATE_EPOCHS
_SPIRAL_PROBLEM_CORRELATION_THRESHOLD = _CASCOR_CORRELATION_THRESHOLD
_SPIRAL_PROBLEM_DISPLAY_FREQUENCY = _CASCOR_DISPLAY_FREQUENCY
_SPIRAL_PROBLEM_EPOCH_DISPLAY_FREQUENCY = _CASCOR_EPOCH_DISPLAY_FREQUENCY
_SPIRAL_PROBLEM_EPOCHS_MAX = _CASCOR_EPOCHS_MAX
# Define constants for the Cascade Correlation Network, generate plots
# _SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT = False
# _SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT = True
_SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT = _CASCOR_GENERATE_PLOTS_DEFAULT
# Define Input and Output Sizes for the Cascade Correlation Network Model
_SPIRAL_PROBLEM_INPUT_SIZE = _CASCOR_INPUT_SIZE
_SPIRAL_PROBLEM_INPUT_SIZE_DEFAULT = _CASCOR_INPUT_SIZE
_SPIRAL_PROBLEM_LEARNING_RATE = _CASCOR_LEARNING_RATE
_SPIRAL_PROBLEM_LOG_CONFIG_FILE_NAME = _PROJECT_LOG_CONFIG_FILE_NAME
_SPIRAL_PROBLEM_LOG_CONFIG_FILE_PATH = _PROJECT_LOG_CONFIG_FILE_PATH
_SPIRAL_PROBLEM_LOG_LEVEL_CUSTOM_NAMES_LIST = _CASCOR_LOG_LEVEL_CUSTOM_NAMES_LIST
_SPIRAL_PROBLEM_LOG_LEVEL_METHODS_DICT = _CASCOR_LOG_LEVEL_METHODS_DICT
_SPIRAL_PROBLEM_LOG_LEVEL_METHODS_LIST = _CASCOR_LOG_LEVEL_METHODS_LIST
_SPIRAL_PROBLEM_LOG_LEVEL_NAMES_LIST = _CASCOR_LOG_LEVEL_NAMES_LIST
_SPIRAL_PROBLEM_LOG_LEVEL_NUMBERS_DICT = _CASCOR_LOG_LEVEL_NUMBERS_DICT
_SPIRAL_PROBLEM_LOG_LEVEL_NUMBERS_LIST = _CASCOR_LOG_LEVEL_NUMBERS_LIST
_SPIRAL_PROBLEM_LOG_LEVEL_REDEFINITION = _CASCOR_LOG_LEVEL_REDEFINITION
_SPIRAL_PROBLEM_LOG_MESSAGE_DEFAULT = _CASCOR_LOG_MESSAGE_DEFAULT
_SPIRAL_PROBLEM_MAX_HIDDEN_UNITS = _CASCOR_MAX_HIDDEN_UNITS
_SPIRAL_PROBLEM_MAX_NEW = _CASCOR_MAX_NEW
_SPIRAL_PROBLEM_MAX_ORIG = _CASCOR_MAX_ORIG
_SPIRAL_PROBLEM_MIN_NEW = _CASCOR_MIN_NEW
_SPIRAL_PROBLEM_MIN_ORIG = _CASCOR_MIN_ORIG
_SPIRAL_PROBLEM_ORIG_POINTS = _CASCOR_ORIG_POINTS
_SPIRAL_PROBLEM_OUTPUT_EPOCHS = _CASCOR_OUTPUT_EPOCHS
_SPIRAL_PROBLEM_OUTPUT_SIZE = _CASCOR_OUTPUT_SIZE
_SPIRAL_PROBLEM_OUTPUT_SIZE_DEFAULT = _CASCOR_OUTPUT_SIZE
_SPIRAL_PROBLEM_PATIENCE = _CASCOR_PATIENCE
_SPIRAL_PROBLEM_RANDOM_SEED = _PROJECT_RANDOM_SEED
_SPIRAL_PROBLEM_AUTHKEY = _PROJECT_MODEL_AUTHKEY
_SPIRAL_PROBLEM_BASE_MANAGER_ADDRESS = _PROJECT_MODEL_BASE_MANAGER_ADDRESS
_SPIRAL_PROBLEM_BASE_MANAGER_ADDRESS_IP = _PROJECT_MODEL_BASE_MANAGER_ADDRESS_IP
_SPIRAL_PROBLEM_BASE_MANAGER_ADDRESS_PORT = _PROJECT_MODEL_BASE_MANAGER_ADDRESS_PORT
_SPIRAL_PROBLEM_STATUS_DISPLAY_FREQUENCY = _CASCOR_STATUS_DISPLAY_FREQUENCY
_SPIRAL_PROBLEM_TEST_RATIO = _CASCOR_TEST_RATIO
_SPIRAL_PROBLEM_TRAIN_RATIO = _CASCOR_TRAIN_RATIO


#####################################################################################################################################################################################################
# Define constants for the SpiralProblem class, logging
_SPIRAL_PROBLEM_LOG_DATE_FORMAT = _PROJECT_LOG_DATE_FORMAT
_SPIRAL_PROBLEM_LOG_FILE_NAME = _PROJECT_LOG_FILE_NAME
_SPIRAL_PROBLEM_LOG_FILE_PATH = _PROJECT_LOG_FILE_PATH

_SPIRAL_PROBLEM_LOG_FORMATTER_STRING = _CASCOR_LOG_FORMATTER_STRING
_SPIRAL_PROBLEM_LOG_FORMATTER_STRING_CONSOLE = _CASCOR_LOG_FORMATTER_STRING_CONSOLE
_SPIRAL_PROBLEM_LOG_FORMATTER_STRING_FILE = _CASCOR_LOG_FORMATTER_STRING_FILE


#####################################################################################################################################################################################################
# Define the constants for the SpiralProblem class during logging initialization and configuration
#     NOTE:  To adjust log level for the SpiralProblem class during logging initialization and configuration, change the following line(s):
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_CRITICAL
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_ERROR
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_WARNING
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_INFO
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_DEBUG
#
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_LOGGING_CONFIG
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_DEFAULT
_SPIRAL_PROBLEM_LOG_LEVEL_NAME_LOGGING_CONFIG = _CASCOR_LOG_LEVEL_NAME_LOGGING_CONFIG

_SPIRAL_PROBLEM_LOG_LEVEL_LOGGING_CONFIG = _SPIRAL_PROBLEM_LOG_LEVEL_NUMBERS_DICT[
    _SPIRAL_PROBLEM_LOG_LEVEL_NAME_LOGGING_CONFIG
]


#####################################################################################################################################################################################################
# Define the constants for the default logging level for the SpiralProblem class
#     NOTE:  To adjust log level for the SpiralProblem class, change the following line(s):
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_FATAL
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_CRITICAL
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_ERROR
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_WARNING
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_INFO
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEBUG
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_VERBOSE
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_TRACE
#
# _SPIRAL_PROBLEM_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEFAULT
_SPIRAL_PROBLEM_LOG_LEVEL_NAME = _CASCOR_LOG_LEVEL_NAME

_SPIRAL_PROBLEM_LOG_LEVEL = _SPIRAL_PROBLEM_LOG_LEVEL_NUMBERS_DICT[
    _SPIRAL_PROBLEM_LOG_LEVEL_NAME
]
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Define constants for the CascadeCorrelationNetwork class, cascade_correlation.py file
#####################################################################################################################################################################################################
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION = _SPIRAL_PROBLEM_ACTIVATION_FUNCTION
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT = (
    _SPIRAL_PROBLEM_ACTIVATION_FUNCTION_DEFAULT
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME = (
    _SPIRAL_PROBLEM_ACTIVATION_FUNCTION_NAME
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT = (
    _SPIRAL_PROBLEM_ACTIVATION_FUNCTIONS_DICT
)

_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_IDENTITY = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_IDENTITY
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_TANH = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_TANH
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_SIGMOID = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SIGMOID
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_RELU = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_RELU
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_LEAKY_RELU = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_LEAKY_RELU
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_ELU = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_ELU
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_SELU = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SELU
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_GELU = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_GELU
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_SOFTMAX = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SOFTMAX
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_SOFTPLUS = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SOFTPLUS
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_HARDTANH = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_HARDTANH
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_SOFTSHRINK = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SOFTSHRINK
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NN_TANHSHRINK = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_TANHSHRINK
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_TANH = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_TANH
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_SIGMOID = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_SIGMOID
)
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_RELU = (
    _PROJECT_MODEL_ACTIVATION_FUNCTION_RELU
)

_CASCADE_CORRELATION_NETWORK_CANDIDATE_DISPLAY_FREQUENCY = (
    _SPIRAL_PROBLEM_CANDIDATE_DISPLAY_FREQUENCY
)
_CASCADE_CORRELATION_NETWORK_CANDIDATE_EPOCHS = _SPIRAL_PROBLEM_CANDIDATE_EPOCHS
_CASCADE_CORRELATION_NETWORK_CANDIDATE_LEARNING_RATE = (
    _SPIRAL_PROBLEM_CANDIDATE_LEARNING_RATE
)
_CASCADE_CORRELATION_NETWORK_CANDIDATE_POOL_SIZE = _SPIRAL_PROBLEM_CANDIDATE_POOL_SIZE
_CASCADE_CORRELATION_NETWORK_DISPLAY_FREQUENCY = _SPIRAL_PROBLEM_DISPLAY_FREQUENCY
_CASCADE_CORRELATION_NETWORK_EPOCH_DISPLAY_FREQUENCY = (
    _SPIRAL_PROBLEM_EPOCH_DISPLAY_FREQUENCY
)
_CASCADE_CORRELATION_NETWORK_EPOCHS_MAX = _SPIRAL_PROBLEM_EPOCHS_MAX
_CASCADE_CORRELATION_NETWORK_GENERATE_PLOTS = _SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT
_CASCADE_CORRELATION_NETWORK_INPUT_SIZE = _SPIRAL_PROBLEM_INPUT_SIZE_DEFAULT
_CASCADE_CORRELATION_NETWORK_LEARNING_RATE = _SPIRAL_PROBLEM_LEARNING_RATE
_CASCADE_CORRELATION_NETWORK_LOG_LEVEL_CUSTOM_NAMES_LIST = (
    _SPIRAL_PROBLEM_LOG_LEVEL_CUSTOM_NAMES_LIST
)
_CASCADE_CORRELATION_NETWORK_LOG_LEVEL_METHODS_DICT = (
    _SPIRAL_PROBLEM_LOG_LEVEL_METHODS_DICT
)
_CASCADE_CORRELATION_NETWORK_LOG_LEVEL_METHODS_LIST = (
    _SPIRAL_PROBLEM_LOG_LEVEL_METHODS_LIST
)
_CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAMES_LIST = _SPIRAL_PROBLEM_LOG_LEVEL_NAMES_LIST
_CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_DICT = (
    _SPIRAL_PROBLEM_LOG_LEVEL_NUMBERS_DICT
)
_CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_LIST = (
    _SPIRAL_PROBLEM_LOG_LEVEL_NUMBERS_LIST
)
_CASCADE_CORRELATION_NETWORK_LOG_LEVEL_REDEFINITION = (
    _SPIRAL_PROBLEM_LOG_LEVEL_REDEFINITION
)
_CASCADE_CORRELATION_NETWORK_LOG_MESSAGE_DEFAULT = _SPIRAL_PROBLEM_LOG_MESSAGE_DEFAULT
_CASCADE_CORRELATION_NETWORK_MAX_HIDDEN_UNITS = _SPIRAL_PROBLEM_MAX_HIDDEN_UNITS
_CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD = (
    _SPIRAL_PROBLEM_CORRELATION_THRESHOLD
)
_CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS = _SPIRAL_PROBLEM_OUTPUT_EPOCHS
_CASCADE_CORRELATION_NETWORK_OUTPUT_SIZE = _SPIRAL_PROBLEM_OUTPUT_SIZE_DEFAULT
_CASCADE_CORRELATION_NETWORK_PATIENCE = _SPIRAL_PROBLEM_PATIENCE
_CASCADE_CORRELATION_NETWORK_RANDOM_MAX_VALUE = _PROJECT_RANDOM_MAX_VALUE
_CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE = _PROJECT_SEQUENCE_MAX_VALUE
_CASCADE_CORRELATION_NETWORK_RANDOM_SEED = _PROJECT_RANDOM_SEED
_CASCADE_CORRELATION_NETWORK_RANDOM_VALUE_SCALE = _SPIRAL_PROBLEM_RANDOM_VALUE_SCALE
_CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR = _HDF5_PROJECT_SNAPSHOTS_DIR
# _CASCADE_CORRELATION_NETWORK_BASE_MANAGER_AUTHKEY = _PROJECT_MODEL_AUTHKEY
_CASCADE_CORRELATION_NETWORK_AUTHKEY = _PROJECT_MODEL_AUTHKEY
# Use the full address tuple (IP, port) - port 0 means dynamic allocation
_CASCADE_CORRELATION_NETWORK_BASE_MANAGER_IP = _PROJECT_MODEL_BASE_MANAGER_ADDRESS_IP
_CASCADE_CORRELATION_NETWORK_BASE_MANAGER_PORT = (
    _PROJECT_MODEL_BASE_MANAGER_ADDRESS_PORT
)
_CASCADE_CORRELATION_NETWORK_BASE_MANAGER_ADDRESS = _PROJECT_MODEL_BASE_MANAGER_ADDRESS
# _CASCADE_CORRELATION_NETWORK_BASE_MANAGER_TIMEOUT = _PROJECT_MODEL_BASE_MANAGER_TIMEOUT
_CASCADE_CORRELATION_NETWORK_TARGET_ACCURACY = _PROJECT_MODEL_TARGET_ACCURACY
_CASCADE_CORRELATION_NETWORK_STATUS_DISPLAY_FREQUENCY = (
    _SPIRAL_PROBLEM_STATUS_DISPLAY_FREQUENCY
)
# _CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_SLEEPYTIME = _PROJECT_MODEL_CANDIDATE_TRAINING_SLEEPYTIME
_CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT = _PROJECT_MODEL_SHUTDOWN_TIMEOUT
_CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT = _PROJECT_MODEL_TASK_QUEUE_TIMEOUT
_CASCADE_CORRELATION_NETWORK_WORKER_STANDBY_SLEEPYTIME = (
    _PROJECT_MODEL_WORKER_STANDBY_SLEEPYTIME
)
_CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_CONTEXT = (
    _PROJECT_CANDIDATE_TRAINING_CONTEXT
)


#####################################################################################################################################################################################################
# Define constants for the CascadeCorrelationNetwork class, logging Configuration
_CASCADE_CORRELATION_NETWORK_LOG_DATE_FORMAT = _SPIRAL_PROBLEM_LOG_DATE_FORMAT
_CASCADE_CORRELATION_NETWORK_LOG_FILE_NAME = _SPIRAL_PROBLEM_LOG_FILE_NAME
_CASCADE_CORRELATION_NETWORK_LOG_FILE_PATH = _SPIRAL_PROBLEM_LOG_FILE_PATH

_CASCADE_CORRELATION_NETWORK_LOG_FORMATTER_STRING = _SPIRAL_PROBLEM_LOG_FORMATTER_STRING
_CASCADE_CORRELATION_NETWORK_LOG_FORMATTER_STRING_CONSOLE = (
    _SPIRAL_PROBLEM_LOG_FORMATTER_STRING_CONSOLE
)
_CASCADE_CORRELATION_NETWORK_LOG_FORMATTER_STRING_FILE = (
    _SPIRAL_PROBLEM_LOG_FORMATTER_STRING_FILE
)


#####################################################################################################################################################################################################
# Define constants for the CascadeCorrelationNetwork class during logging initialization and configuration
#     NOTE:  To adjust log level for the CascadeCorrelationNetwork class during logging initialization and configuration, change the following line(s):
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_CRITICAL
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_ERROR
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_WARNING
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_INFO
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_DEBUG
#
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_LOGGING_CONFIG
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_NAME_DEFAULT
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME_LOGGING_CONFIG = _CASCOR_LOG_LEVEL_NAME_LOGGING_CONFIG
_CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME_LOGGING_CONFIG = (
    _SPIRAL_PROBLEM_LOG_LEVEL_NAME_LOGGING_CONFIG
)

_CASCOR_CORRELATION_NETWORK_LOG_LEVEL_LOGGING_CONFIG = (
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_DICT[
        _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME_LOGGING_CONFIG
    ]
)


#####################################################################################################################################################################################################
# Define the default logging level for the CascadeCorrelationNetwork class
#     NOTE:  To adjust log level for the CascadeCorrelationNetwork class, change the following line(s):
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_FATAL
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_CRITICAL
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_ERROR
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_WARNING
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_INFO
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEBUG
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_VERBOSE
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_TRACE
#
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEFAULT
# _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME = _CASCOR_LOG_LEVEL_NAME
_CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME = _SPIRAL_PROBLEM_LOG_LEVEL_NAME

_CASCADE_CORRELATION_NETWORK_LOG_LEVEL = (
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_DICT[
        _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME
    ]
)
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Define constants for the CandidateUnit class, candidate_unit.py file
#####################################################################################################################################################################################################
_CANDIDATE_UNIT_ACTIVATION_FUNCTION = _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION
_CANDIDATE_UNIT_ACTIVATION_FUNCTION_DEFAULT = (
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT
)
_CANDIDATE_UNIT_ACTIVATION_FUNCTION_NAME = (
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME
)
_CANDIDATE_UNIT_ACTIVATION_FUNCTIONS_DICT = (
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT
)

# _CANDIDATE_UNIT_DISPLAY_FREQUENCY = _CASCADE_CORRELATION_NETWORK_DISPLAY_FREQUENCY
_CANDIDATE_UNIT_EPOCHS_MAX = _CASCADE_CORRELATION_NETWORK_EPOCHS_MAX
_CANDIDATE_UNIT_INPUT_SIZE = _CASCADE_CORRELATION_NETWORK_INPUT_SIZE
_CANDIDATE_UNIT_OUTPUT_SIZE = _CASCADE_CORRELATION_NETWORK_OUTPUT_SIZE
_CANDIDATE_UNIT_LEARNING_RATE = _CASCADE_CORRELATION_NETWORK_CANDIDATE_LEARNING_RATE
_CANDIDATE_UNIT_RANDOM_SEED = _CASCADE_CORRELATION_NETWORK_RANDOM_SEED
_CANDIDATE_UNIT_RANDOM_MAX_VALUE = _CASCADE_CORRELATION_NETWORK_RANDOM_MAX_VALUE
_CANDIDATE_UNIT_SEQUENCE_MAX_VALUE = _CASCADE_CORRELATION_NETWORK_SEQUENCE_MAX_VALUE
_CANDIDATE_UNIT_RANDOM_VALUE_SCALE = _CASCADE_CORRELATION_NETWORK_RANDOM_VALUE_SCALE
_CANDIDATE_UNIT_INITIAL_RESIDUAL_ERROR = _PROJECT_RESIDUAL_ERROR_DEFAULT

_CANDIDATE_UNIT_EARLY_STOPPING = _PROJECT_MODEL_CANDIDATE_EARLY_STOPPING
_CANDIDATE_UNIT_PATIENCE = _PROJECT_MODEL_CANDIDATE_PATIENCE
_CANDIDATE_UNIT_DISPLAY_FREQUENCY = _PROJECT_MODEL_CANDIDATE_DISPLAY_FREQUENCY
_CANDIDATE_UNIT_STATUS_FREQUENCY = _PROJECT_MODEL_DISPLAY_FREQUENCY
_CANDIDATE_UNIT_POOL_SIZE = _PROJECT_MODEL_CANDIDATE_POOL_SIZE
_CANDIDATE_UNIT_EPOCHS = _PROJECT_MODEL_CANDIDATE_EPOCHS
_CANDIDATE_UNIT_LEARNING_RATE = _PROJECT_MODEL_CANDIDATE_UNIT_LEARNING_RATE


#####################################################################################################################################################################################################
# Define constants for the  CandidateUnit class Log Level data structures
_CANDIDATE_UNIT_LOG_LEVEL_CUSTOM_NAMES_LIST = (
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_CUSTOM_NAMES_LIST
)
_CANDIDATE_UNIT_LOG_LEVEL_METHODS_DICT = (
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_METHODS_DICT
)
_CANDIDATE_UNIT_LOG_LEVEL_METHODS_LIST = (
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_METHODS_LIST
)
_CANDIDATE_UNIT_LOG_LEVEL_NAMES_LIST = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAMES_LIST
_CANDIDATE_UNIT_LOG_LEVEL_NUMBERS_DICT = (
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_DICT
)
_CANDIDATE_UNIT_LOG_LEVEL_NUMBERS_LIST = (
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NUMBERS_LIST
)
_CANDIDATE_UNIT_LOG_LEVEL_REDEFINITION = (
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_REDEFINITION
)
_CANDIDATE_UNIT_LOG_MESSAGE_DEFAULT = _CASCADE_CORRELATION_NETWORK_LOG_MESSAGE_DEFAULT


#####################################################################################################################################################################################################
# Define constants for the CandidateUnit class, logging Configuration
_CANDIDATE_UNIT_LOG_DATE_FORMAT = _CASCADE_CORRELATION_NETWORK_LOG_DATE_FORMAT
_CANDIDATE_UNIT_LOG_FILE_NAME = _CASCADE_CORRELATION_NETWORK_LOG_FILE_NAME
_CANDIDATE_UNIT_LOG_FILE_PATH = _CASCADE_CORRELATION_NETWORK_LOG_FILE_PATH

_CANDIDATE_UNIT_LOG_FORMATTER_STRING = _CASCADE_CORRELATION_NETWORK_LOG_FORMATTER_STRING
_CANDIDATE_UNIT_LOG_FORMATTER_STRING_CONSOLE = (
    _CASCADE_CORRELATION_NETWORK_LOG_FORMATTER_STRING_CONSOLE
)
_CANDIDATE_UNIT_LOG_FORMATTER_STRING_FILE = (
    _CASCADE_CORRELATION_NETWORK_LOG_FORMATTER_STRING_FILE
)


#####################################################################################################################################################################################################
# Define the default logging level for the CandidateUnit class during logging configuration.
#     NOTE:  To adjust log level for the CandidateUnit class during logging configuration, change the following line(s):
# _CANDIDATE_UNIT_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_CRITICAL
# _CANDIDATE_UNIT_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_ERROR
# _CANDIDATE_UNIT_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_WARNING
# _CANDIDATE_UNIT_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_INFO
# _CANDIDATE_UNIT_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_DEBUG
#
# _CANDIDATE_UNIT_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_LOGGING_CONFIG
# _CANDIDATE_UNIT_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_DEFAULT
# _CANDIDATE_UNIT_LOG_LEVEL_NAME_LOGGING_CONFIG = _CASCOR_LOG_LEVEL_NAME_LOGGING_CONFIG
# _CANDIDATE_UNIT_LOG_LEVEL_NAME_LOGGING_CONFIG = _SPIRAL_PROBLEM_LOG_LEVEL_NAME_LOGGING_CONFIG
_CANDIDATE_UNIT_LOG_LEVEL_NAME_LOGGING_CONFIG = (
    _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME_LOGGING_CONFIG
)

_CANDIDATE_UNIT_LOG_LEVEL_LOGGING_CONFIG = _CANDIDATE_UNIT_LOG_LEVEL_NUMBERS_DICT[
    _CANDIDATE_UNIT_LOG_LEVEL_NAME_LOGGING_CONFIG
]


#####################################################################################################################################################################################################
# Define the default logging level for the CandidateUnit class
#     NOTE:  To adjust log level for the CandidateUnit class, change the following line(s):
# _CANDIDATE_UNIT_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_FATAL
# _CANDIDATE_UNIT_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_CRITICAL
# _CANDIDATE_UNIT_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_ERROR
# _CANDIDATE_UNIT_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_WARNING
# _CANDIDATE_UNIT_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_INFO
# _CANDIDATE_UNIT_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEBUG
# _CANDIDATE_UNIT_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_VERBOSE
# _CANDIDATE_UNIT_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_TRACE
#
# _CANDIDATE_UNIT_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEFAULT
# _CANDIDATE_UNIT_LOG_LEVEL_NAME = _CASCOR_LOG_LEVEL_NAME
# _CANDIDATE_UNIT_LOG_LEVEL_NAME = _SPIRAL_PROBLEM_LOG_LEVEL_NAME
_CANDIDATE_UNIT_LOG_LEVEL_NAME = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME

_CANDIDATE_UNIT_LOG_LEVEL = _CANDIDATE_UNIT_LOG_LEVEL_NUMBERS_DICT[
    _CANDIDATE_UNIT_LOG_LEVEL_NAME
]
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Define constants for the LogConfig class
#####################################################################################################################################################################################################
_LOG_CONFIG_LOG_ALLOW_LOG_LEVEL_REDEFINITION = _PROJECT_LOG_LEVEL_REDEFINITION
_LOG_CONFIG_LOG_CONFIG_FILE_NAME = _PROJECT_LOG_CONFIG_FILE_NAME
_LOG_CONFIG_LOG_CONFIG_FILE_PATH = _PROJECT_LOG_CONFIG_FILE_PATH
# Define custom logging level lists
_LOG_CONFIG_LOG_LEVEL_CUSTOM_NAMES_LIST = _PROJECT_LOG_LEVEL_CUSTOM_NAMES_LIST
# Define list and dict data structures for all logging levels
_LOG_CONFIG_LOG_LEVEL_METHODS_DICT = _PROJECT_LOG_LEVEL_METHODS_DICT
_LOG_CONFIG_LOG_LEVEL_METHODS_LIST = _PROJECT_LOG_LEVEL_METHODS_LIST
_LOG_CONFIG_LOG_LEVEL_NAMES_LIST = _PROJECT_LOG_LEVEL_NAMES_LIST
_LOG_CONFIG_LOG_LEVEL_NUMBERS_DICT = _PROJECT_LOG_LEVEL_NUMBERS_DICT
_LOG_CONFIG_LOG_LEVEL_NUMBERS_LIST = _PROJECT_LOG_LEVEL_NUMBERS_LIST
_LOG_CONFIG_LOG_LEVEL_REDEFINITION = _PROJECT_LOG_LEVEL_REDEFINITION
_LOG_CONFIG_LOG_MESSAGE_DEFAULT = _PROJECT_LOG_MESSAGE_DEFAULT


#####################################################################################################################################################################################################
# Define constants for the LogConfig class, logging configuration
_LOG_CONFIG_LOG_DATE_FORMAT = _PROJECT_LOG_DATE_FORMAT
_LOG_CONFIG_LOG_FILE_NAME = _PROJECT_LOG_FILE_NAME
_LOG_CONFIG_LOG_FILE_PATH = _PROJECT_LOG_FILE_PATH

_LOG_CONFIG_LOG_FORMATTER_STRING = _CASCOR_LOG_FORMATTER_STRING
_LOG_CONFIG_LOG_FORMATTER_STRING_CONSOLE = _CASCOR_LOG_FORMATTER_STRING_CONSOLE
_LOG_CONFIG_LOG_FORMATTER_STRING_FILE = _CASCOR_LOG_FORMATTER_STRING_FILE


#####################################################################################################################################################################################################
# Define the default logging level for the LogConfig class during logging configuration.
#     NOTE:  To adjust log level for the LogConfig class during logging configuration, change the following line(s):
# _LOG_CONFIG_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_CRITICAL
# _LOG_CONFIG_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_ERROR
# _LOG_CONFIG_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_WARNING
# _LOG_CONFIG_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_INFO
# _LOG_CONFIG_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_DEBUG
#
# _LOG_CONFIG_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_LOGGING_CONFIG
# _LOG_CONFIG_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_DEFAULT
_LOG_CONFIG_LOG_LEVEL_NAME_LOGGING_CONFIG = _CASCOR_LOG_LEVEL_NAME_LOGGING_CONFIG
# _LOG_CONFIG_LOG_LEVEL_NAME_LOGGING_CONFIG = _SPIRAL_PROBLEM_LOG_LEVEL_NAME_LOGGING_CONFIG
# _LOG_CONFIG_LOG_LEVEL_NAME_LOGGING_CONFIG = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME_LOGGING_CONFIG
# _LOG_CONFIG_LOG_LEVEL_NAME_LOGGING_CONFIG = _CANDIDATE_UNIT_LOG_LEVEL_NAME_LOGGING_CONFIG

_LOG_CONFIG_LOG_LEVEL_LOGGING_CONFIG = _LOG_CONFIG_LOG_LEVEL_NUMBERS_DICT[
    _LOG_CONFIG_LOG_LEVEL_NAME_LOGGING_CONFIG
]


#####################################################################################################################################################################################################
# Define the default logging level for the LogConfig class
#     NOTE: To adjust log level for the LogConfig class, change the following line(s):
# _LOG_CONFIG_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_FATAL
# _LOG_CONFIG_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_CRITICAL
# _LOG_CONFIG_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_ERROR
# _LOG_CONFIG_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_WARNING
# _LOG_CONFIG_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_INFO
# _LOG_CONFIG_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEBUG
# _LOG_CONFIG_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_VERBOSE
# _LOG_CONFIG_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_TRACE
#
# _LOG_CONFIG_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEFAULT
_LOG_CONFIG_LOG_LEVEL_NAME = _CASCOR_LOG_LEVEL_NAME
# _LOG_CONFIG_LOG_LEVEL_NAME = _SPIRAL_PROBLEM_LOG_LEVEL_NAME
# _LOG_CONFIG_LOG_LEVEL_NAME = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME
# _LOG_CONFIG_LOG_LEVEL_NAME = _CANDIDATE_UNIT_LOG_LEVEL_NAME

_LOG_CONFIG_LOG_LEVEL = _LOG_CONFIG_LOG_LEVEL_NUMBERS_DICT[_LOG_CONFIG_LOG_LEVEL_NAME]
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Define constants for the Logger class
#####################################################################################################################################################################################################
_LOGGER_LOG_ALLOW_LOG_LEVEL_REDEFINITION = _LOG_CONFIG_LOG_ALLOW_LOG_LEVEL_REDEFINITION
_LOGGER_LOG_CONFIG_FILE_NAME = _LOG_CONFIG_LOG_CONFIG_FILE_NAME
_LOGGER_LOG_CONFIG_FILE_PATH = _LOG_CONFIG_LOG_CONFIG_FILE_PATH
_LOGGER_LOG_LEVEL_CUSTOM_NAMES_LIST = _LOG_CONFIG_LOG_LEVEL_CUSTOM_NAMES_LIST
_LOGGER_LOG_LEVEL_LOGGING_CONFIG = _LOG_CONFIG_LOG_LEVEL_LOGGING_CONFIG
_LOGGER_LOG_LEVEL_METHODS_DICT = _LOG_CONFIG_LOG_LEVEL_METHODS_DICT
_LOGGER_LOG_LEVEL_METHODS_LIST = _LOG_CONFIG_LOG_LEVEL_METHODS_LIST
_LOGGER_LOG_LEVEL_NAMES_LIST = _LOG_CONFIG_LOG_LEVEL_NAMES_LIST
_LOGGER_LOG_LEVEL_NUMBERS_DICT = _LOG_CONFIG_LOG_LEVEL_NUMBERS_DICT
_LOGGER_LOG_LEVEL_NUMBERS_LIST = _LOG_CONFIG_LOG_LEVEL_NUMBERS_LIST
_LOGGER_LOG_LEVEL_REDEFINITION = _LOG_CONFIG_LOG_LEVEL_REDEFINITION
_LOGGER_LOG_MESSAGE_DEFAULT = _LOG_CONFIG_LOG_MESSAGE_DEFAULT


#####################################################################################################################################################################################################
# Define constants for the Logger class, logging configuration
_LOGGER_LOG_LOGGING_CONFIGURED_DEFAULT = _PROJECT_LOG_LOGGING_CONFIGURED_DEFAULT
_LOGGER_LOG_DATE_FORMAT = _LOG_CONFIG_LOG_DATE_FORMAT
_LOGGER_LOG_FILE_NAME = _LOG_CONFIG_LOG_FILE_NAME
_LOGGER_LOG_FILE_PATH = _LOG_CONFIG_LOG_FILE_PATH

_LOGGER_LOG_FORMATTER_STRING = _LOG_CONFIG_LOG_FORMATTER_STRING
_LOGGER_LOG_FORMATTER_STRING_CONSOLE = _PROJECT_LOG_FORMATTER_STRING_CONSOLE
_LOGGER_LOG_FORMATTER_STRING_FILE = _PROJECT_LOG_FORMATTER_STRING_FILE
# _LOGGER_NAME = _PROJECT_LOGGER_NAME


#####################################################################################################################################################################################################
# Define the default logging level for the Logger class during logging configuration
#     NOTE: To adjust log level for the Logger class during logging configuration, change the following line(s):
# _LOGGER_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_CRITICAL
# _LOGGER_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_ERROR
# _LOGGER_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_WARNING
# _LOGGER_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_INFO
# _LOGGER_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_DEBUG
#
# _LOGGER_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_LOGGING_CONFIG
# _LOGGER_LOG_LEVEL_NAME_LOGGING_CONFIG = _PROJECT_LOG_LEVEL_NAME_DEFAULT
# _LOGGER_LOG_LEVEL_NAME_LOGGING_CONFIG = _CASCOR_LOG_LEVEL_NAME_LOGGING_CONFIG
# _LOGGER_LOG_LEVEL_NAME_LOGGING_CONFIG = _SPIRAL_PROBLEM_LOG_LEVEL_NAME_LOGGING_CONFIG
# _LOGGER_LOG_LEVEL_NAME_LOGGING_CONFIG = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME_LOGGING_CONFIG
# _LOGGER_LOG_LEVEL_NAME_LOGGING_CONFIG = _CANDIDATE_UNIT_LOG_LEVEL_NAME_LOGGING_CONFIG
_LOGGER_LOG_LEVEL_NAME_LOGGING_CONFIG = _LOG_CONFIG_LOG_LEVEL_NAME_LOGGING_CONFIG

_LOGGER_LOG_LEVEL_LOGGING_CONFIG = _LOGGER_LOG_LEVEL_NUMBERS_DICT[
    _LOGGER_LOG_LEVEL_NAME_LOGGING_CONFIG
]


#####################################################################################################################################################################################################
# Define the default logging level for the Logger class.
#     NOTE: To adjust log level for the Logger class, change the following line(s):
# _LOGGER_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_FATAL
# _LOGGER_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_CRITICAL
# _LOGGER_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_ERROR
# _LOGGER_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_WARNING
# _LOGGER_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_INFO
# _LOGGER_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEBUG
# _LOGGER_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_VERBOSE
# _LOGGER_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_TRACE
#
# _LOGGER_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEFAULT
# _LOGGER_LOG_LEVEL_NAME = _CASCOR_LOG_LEVEL_NAME
# _LOGGER_LOG_LEVEL_NAME = _SPIRAL_PROBLEM_LOG_LEVEL_NAME
# _LOGGER_LOG_LEVEL_NAME = _CASCADE_CORRELATION_NETWORK_LOG_LEVEL_NAME
# _LOGGER_LOG_LEVEL_NAME = _CANDIDATE_UNIT_LOG_LEVEL_NAME
_LOGGER_LOG_LEVEL_NAME = _LOG_CONFIG_LOG_LEVEL_NAME

_LOGGER_LOG_LEVEL = _LOGGER_LOG_LEVEL_NUMBERS_DICT[_LOGGER_LOG_LEVEL_NAME]
#####################################################################################################################################################################################################
