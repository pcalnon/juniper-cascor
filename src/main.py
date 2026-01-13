#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     cascor.py
# Author:        Paul Calnon
# Version:       0.3.1 (0.7.3)
#
# Date Created:  2025-06-11
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
#    - This file serves as the main entry point for the Cascade Correlation Neural Network project.
#    - It initializes logging, sets up configurations, and runs the main logic to solve the two spiral problem.
#    - It uses the `setup_logging` function to configure the logging system.
#    - It creates an instance of the `SpiralProblem` class and calls its `run` method to start the problem-solving process.
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
import os
# import columnar as col
# import torch
# import numpy as np
# import random
# import utils.utils

# from inspect import currentframe, getframeinfo

from constants.constants import (
    _CASCOR_ACTIVATION_FUNCTION,
    _CASCOR_CANDIDATE_DISPLAY_FREQUENCY,
    _CASCOR_CANDIDATE_EPOCHS,
    _CASCOR_CLOCKWISE,
    _CASCOR_CORRELATION_THRESHOLD,
    _CASCOR_DISTRIBUTION_FACTOR,
    _CASCOR_DEFAULT_ORIGIN,
    _CASCOR_DEFAULT_RADIUS,
    _CASCOR_EPOCHS_MAX,
    _CASCOR_GENERATE_PLOTS_DEFAULT,
    _CASCOR_INPUT_SIZE,
    _CASCOR_LEARNING_RATE,
    _CASCOR_MAX_HIDDEN_UNITS,
    _CASCOR_MAX_NEW, # trunk-ignore(ruff/F401)
    _CASCOR_MAX_ORIG, # trunk-ignore(ruff/F401)
    _CASCOR_MIN_NEW, # trunk-ignore(ruff/F401)
    _CASCOR_MIN_ORIG, # trunk-ignore(ruff/F401)
    _CASCOR_NOISE_FACTOR_DEFAULT,
    _CASCOR_NUM_ROTATIONS,
    _CASCOR_NUM_SPIRALS,
    _CASCOR_NUMBER_POINTS_PER_SPIRAL,
    _CASCOR_ORIG_POINTS, # trunk-ignore(ruff/F401)
    _CASCOR_OUTPUT_EPOCHS,
    _CASCOR_OUTPUT_SIZE,
    _CASCOR_PATIENCE,
    _CASCOR_RANDOM_VALUE_SCALE,
    _CASCOR_STATUS_DISPLAY_FREQUENCY,
    _CASCOR_TEST_RATIO,
    _CASCOR_TRAIN_RATIO,
    _CASCOR_LOG_CONFIG_FILE_NAME,
    _CASCOR_LOG_CONFIG_FILE_PATH,
    _CASCOR_LOG_DATE_FORMAT,
    _CASCOR_LOG_FILE_NAME,
    _CASCOR_LOG_FILE_PATH,
    _CASCOR_LOG_FORMATTER_STRING,
    _CASCOR_LOG_LEVEL,
    _CASCOR_LOG_LEVEL_CUSTOM_NAMES_LIST,
    _CASCOR_LOG_LEVEL_LOGGING_CONFIG,
    _CASCOR_LOG_LEVEL_METHODS_DICT,
    _CASCOR_LOG_LEVEL_METHODS_LIST,
    _CASCOR_LOG_LEVEL_NAME,
    _CASCOR_LOG_LEVEL_NAMES_LIST,
    _CASCOR_LOG_LEVEL_NUMBERS_DICT,
    _CASCOR_LOG_LEVEL_NUMBERS_LIST,
    _CASCOR_LOG_LEVEL_REDEFINITION,
    _CASCOR_LOG_MESSAGE_DEFAULT,
    _CASCOR_RANDOM_SEED,
)
from log_config.log_config import LogConfig
from log_config.logger.logger import Logger
from spiral_problem.spiral_problem import SpiralProblem

# TODO: don't think this is needed with Logger class implementing singleton pattern, and Class methods for initial logging
global logger
global log_config

def main():
    Logger.info("Cascor: main: Starting the Cascade Correlation Neural Network project")
    Logger.info(f"Cascor: main: Project constants: Log Level: {_CASCOR_LOG_LEVEL}, Log Level Name: {_CASCOR_LOG_LEVEL_NAME}")
    Logger.info(f"Cascor: main: Project constants: Log File Name: {_CASCOR_LOG_FILE_NAME}, Log File Path: {_CASCOR_LOG_FILE_PATH}")


    if (log_config := LogConfig(
        _LogConfig__log_config=logging.config,
        # _LogConfig__log_config=None,
        _LogConfig__log_config_file_name=_CASCOR_LOG_CONFIG_FILE_NAME,
        _LogConfig__log_config_file_path=_CASCOR_LOG_CONFIG_FILE_PATH,
        _LogConfig__log_date_format=_CASCOR_LOG_DATE_FORMAT,
        _LogConfig__log_file_name=_CASCOR_LOG_FILE_NAME,
        _LogConfig__log_file_path=_CASCOR_LOG_FILE_PATH,
        _LogConfig__log_formatter_string=_CASCOR_LOG_FORMATTER_STRING,
        _LogConfig__log_level_custom_names_list=_CASCOR_LOG_LEVEL_CUSTOM_NAMES_LIST,
        _LogConfig__log_level_logging_config=_CASCOR_LOG_LEVEL_LOGGING_CONFIG,
        _LogConfig__log_level_methods_dict=_CASCOR_LOG_LEVEL_METHODS_DICT,
        _LogConfig__log_level_methods_list=_CASCOR_LOG_LEVEL_METHODS_LIST,
        _LogConfig__log_level_names_list=_CASCOR_LOG_LEVEL_NAMES_LIST,
        _LogConfig__log_level_numbers_dict=_CASCOR_LOG_LEVEL_NUMBERS_DICT,
        _LogConfig__log_level_numbers_list=_CASCOR_LOG_LEVEL_NUMBERS_LIST,
        _LogConfig__log_level_redefinition=_CASCOR_LOG_LEVEL_REDEFINITION,
        _LogConfig__log_level=_CASCOR_LOG_LEVEL,
        _LogConfig__log_level_name=_CASCOR_LOG_LEVEL_NAME,
        _LogConfig__log_message_default=_CASCOR_LOG_MESSAGE_DEFAULT,
    )) is None:
        Logger.error("Cascor: main: Error: Failed to create LogConfig class")
        os._exit(1)
    elif (logger := log_config.get_logger()) is None:
        Logger.error("Cascor: main: Error: Failed to get Logger object from LogConfig class")
        os._exit(2)


    Logger.debug(f"Cascor: main: Successfully created LogConfig class and Logger object: Type: {type(log_config)}, Value:\n{log_config}")
    Logger.debug(f"Cascor: main: Successfully created LogConfig class and Logger object: Type: {type(logger)}, Value:\n{logger}")
    Logger.debug(f"Cascor: main: Successfully created LogConfig class & Logger object: log level: Type: {type(log_config.get_log_level())}, Value: {log_config.get_log_level()}")
    Logger.debug(f"Cascor: main: Successfully created LogConfig class & Logger object: log level name: Type: {type(log_config.get_log_level_name())}, Value: {log_config.get_log_level_name()}")
    Logger.debug(f"Cascor: main: Successfully created Logger object: logger level: Type: {type(logger.level)}, Value: {logger.level}")
    Logger.debug(f"Cascor: main: Successfully created Logger object: logger name: '{logger.name}', handlers: {len(logger.handlers)}")

    logger.verbose(f"Cascor: main: Successfully created LogConfig class: Type: {type(log_config)}, Value: {log_config}, and Logger object: Type: {type(logger)}, Value: {logger}")
    logger.debug(f"Cascor: main: Successfully created LogConfig class: {log_config} and Logger object: {logger}")
    logger.info(f"Cascor: main: Successfully created LogConfig class & Logger object: log level: {log_config.get_log_level()}")
    logger.info("Cascor: main: Inside Main function")
    logger.info("Cascor: main: Completed Initialization of Project Logger")





# #####################################################################################################################################################################################################

#     generated_datasets = GeneratedDatasets(
#         _GeneratedDatasets__spiral_config=config,
#         _GeneratedDatasets__dataset_tensors=None,
#         _GeneratedDatasets__dataset_file_info=None,
#         _GeneratedDatasets__num_spirals=_SPIRAL_DATASET_DEFAULT_NUM_SPIRALS,
#         _GeneratedDatasets__num_points_per_spiral=_SPIRAL_DATASET_DEFAULT_NUM_POINTS_PER_SPIRAL,
#         _GeneratedDatasets__noise_level=_SPIRAL_DATASET_DEFAULT_NOISE_LEVEL,
#         _GeneratedDatasets__num_rotations=_SPIRAL_DATASET_DEFAULT_NUM_ROTATIONS,
#         _GeneratedDatasets__min_radius=_SPIRAL_DATASET_DEFAULT_MIN_RADIUS,
#         _GeneratedDatasets__max_radius=_SPIRAL_DATASET_DEFAULT_MAX_RADIUS,
#         _GeneratedDatasets__clockwise_rotation=_SPIRAL_DATASET_DEFAULT_CLOCKWISE_ROTATION,
#         _GeneratedDatasets__seed_value=_SPIRAL_DATASET_DEFAULT_SEED_VALUE,
#         _GeneratedDatasets__dataset_dir=_SPIRAL_DATASET_DATASET_DIR_DEFAULT,
#         _GeneratedDatasets__visualization_dir=_SPIRAL_DATASET_VISUALIZATION_DIR_DEFAULT,
#         _GeneratedDatasets__log_file_name=_CASCOR_SPIRAL_DATASET_LOG_NAME,
#         _GeneratedDatasets__log_formatter_string=_CASCOR_SPIRAL_DATASET_LOG_FORMATTER_STRING,
#         _GeneratedDatasets__log_date_format=_CASCOR_SPIRAL_DATASET_LOG_DATE_FORMAT,
#         _GeneratedDatasets__log_file_path=_CASCOR_SPIRAL_DATASET_LOG_FILE_PATH,
#         _GeneratedDatasets__log_level=_CASCOR_SPIRAL_DATASET_LOG_LEVEL_DEFAULT,
#         _GeneratedDatasets__dataset_train_ratio=_SPIRAL_DATASET_DEFAULT_TRAIN_RATIO,
#         _GeneratedDatasets__dataset_test_ratio=_SPIRAL_DATASET_DEFAULT_TEST_RATIO,
#         _GeneratedDatasets__dataset_val_ratio=_SPIRAL_DATASET_DEFAULT_VAL_RATIO,
#     )


#     config = SpiralConfig(
#         _SpiralConfig__num_spirals=_SPIRAL_DATASET_DEFAULT_NUM_SPIRALS,
#         _SpiralConfig__num_points_per_spiral=_SPIRAL_DATASET_DEFAULT_NUM_POINTS_PER_SPIRAL,
#         _SpiralConfig__noise_level=_SPIRAL_DATASET_DEFAULT_NOISE_LEVEL,
#         _SpiralConfig__num_rotations=_SPIRAL_DATASET_DEFAULT_NUM_ROTATIONS,
#         _SpiralConfig__min_radius=_SPIRAL_DATASET_DEFAULT_MIN_RADIUS,
#         _SpiralConfig__max_radius=_SPIRAL_DATASET_DEFAULT_MAX_RADIUS,
#         _SpiralConfig__clockwise_rotation=_SPIRAL_DATASET_DEFAULT_CLOCKWISE_ROTATION,
#         _SpiralConfig__seed_value=_SPIRAL_DATASET_DEFAULT_SEED_VALUE,
#         _SpiralConfig__visualization_dir=_SPIRAL_DATASET_VISUALIZATION_DIR_DEFAULT,
#         _SpiralConfig__dataset_dir=_SPIRAL_DATASET_DATASET_DIR_DEFAULT,
#         _SpiralConfig__log_file_path=_CASCOR_SPIRAL_DATASET_LOG_FILE_PATH,
#         _SpiralConfig__log_name=_CASCOR_SPIRAL_DATASET_LOG_NAME,
#         _SpiralConfig__log_level=_CASCOR_SPIRAL_DATASET_LOG_LEVEL_DEFAULT,
#     )

# #####################################################################################################################################################################################################





    # Instantiate the SpiralProblem class
    logger.info("Cascor: main: Creating SpiralProblem instance")
    sp = SpiralProblem(
        _SpiralProblem__spiral_config=logging.config,
        _SpiralProblem__dataset_tensors=None,
        _SpiralProblem__dataset_file_info=None,
        _SpiralProblem__activation_function=_CASCOR_ACTIVATION_FUNCTION,
        _SpiralProblem__candidate_display_frequency=_CASCOR_CANDIDATE_DISPLAY_FREQUENCY,
        _SpiralProblem__candidate_epochs=_CASCOR_CANDIDATE_EPOCHS,
        _SpiralProblem__clockwise=_CASCOR_CLOCKWISE,
        _SpiralProblem__correlation_threshold=_CASCOR_CORRELATION_THRESHOLD,
        _SpiralProblem__default_origin=_CASCOR_DEFAULT_ORIGIN,
        _SpiralProblem__default_radius=_CASCOR_DEFAULT_RADIUS,
        _SpiralProblem__distribution=_CASCOR_DISTRIBUTION_FACTOR,
        _SpiralProblem__epochs_max=_CASCOR_EPOCHS_MAX,
        _SpiralProblem__generate_plots_default=_CASCOR_GENERATE_PLOTS_DEFAULT,
        _SpiralProblem__input_size=_CASCOR_INPUT_SIZE,
        _SpiralProblem__learning_rate=_CASCOR_LEARNING_RATE,
        _SpiralProblem__log_config=log_config,
        _SpiralProblem__log_file_name=_CASCOR_LOG_FILE_NAME,
        _SpiralProblem__log_file_path=_CASCOR_LOG_FILE_PATH,
        _SpiralProblem__log_level_name=_CASCOR_LOG_LEVEL_NAME,
        _SpiralProblem__max_hidden_units=_CASCOR_MAX_HIDDEN_UNITS,
        _SpiralProblem__n_points=_CASCOR_NUMBER_POINTS_PER_SPIRAL,
        _SpiralProblem__n_rotations=_CASCOR_NUM_ROTATIONS,
        _SpiralProblem__n_spirals=_CASCOR_NUM_SPIRALS,
        _SpiralProblem__noise=_CASCOR_NOISE_FACTOR_DEFAULT,
        _SpiralProblem__output_size=_CASCOR_OUTPUT_SIZE,
        _SpiralProblem__patience=_CASCOR_PATIENCE,
        _SpiralProblem__output_epochs=_CASCOR_OUTPUT_EPOCHS,
        _SpiralProblem__status_display_frequency=_CASCOR_STATUS_DISPLAY_FREQUENCY,
        _SpiralProblem__random_seed=_CASCOR_RANDOM_SEED,
        _SpiralProblem__train_ratio=_CASCOR_TRAIN_RATIO,
        _SpiralProblem__test_ratio=_CASCOR_TEST_RATIO,
    )
    logger.debug(f"Main: sp: Type: {type(sp)}, Value:\n{sp}")




# #####################################################################################################################################################################################################

#     complex_dataset_output = generated_datasets.generate_spiral_datasets(
#         num_spirals=_SPIRAL_DATASET_DEFAULT_NUM_SPIRALS,
#         num_points=_SPIRAL_DATASET_DEFAULT_NUM_POINTS_PER_SPIRAL,
#         num_rotations=_SPIRAL_DATASET_DEFAULT_NUM_ROTATIONS,
#         noise_level=_SPIRAL_DATASET_DEFAULT_NOISE_LEVEL,
#         min_radius=_SPIRAL_DATASET_DEFAULT_MIN_RADIUS,
#         max_radius=_SPIRAL_DATASET_DEFAULT_MAX_RADIUS,
#         clockwise_rotation=_SPIRAL_DATASET_DEFAULT_CLOCKWISE_ROTATION,
#         seed_value=_SPIRAL_DATASET_DEFAULT_SEED_VALUE,
#         visualization_dir=_SPIRAL_DATASET_VISUALIZATION_DIR_DEFAULT,
#         dataset_dir=_SPIRAL_DATASET_DATASET_DIR_DEFAULT,
#         log_file_path=_CASCOR_SPIRAL_DATASET_LOG_FILE_PATH,
#         dataset_train_ratio=_SPIRAL_DATASET_DEFAULT_TRAIN_RATIO,
#         dataset_test_ratio=_SPIRAL_DATASET_DEFAULT_TEST_RATIO,
#         dataset_val_ratio=_SPIRAL_DATASET_DEFAULT_VAL_RATIO,
#     )
#     logger.info("cascor_spiral: main: Creating DatasetTensors instance for the complex dataset...")
#     (complex_dataset_file_info, complex_dataset_tensors) = complex_dataset_output
#     logger.info(f"cascor_spiral: main: Generated complex dataset file info: {complex_dataset_file_info}")
#     logger.info(f"cascor_spiral: main: Generated complex dataset tensors: {complex_dataset_tensors}")
#     # Convert Complex dataset to tensors dict and send to CUDA if available
#     complex_dataset_tensors.to_cuda()

# #####################################################################################################################################################################################################




    # Solve the two spiral problem using the SpiralProblem instance
    logger.info("Main: Solving SpiralProblem instance")
    sp.evaluate(
        n_points=_CASCOR_NUMBER_POINTS_PER_SPIRAL,
        n_spirals=_CASCOR_NUM_SPIRALS,
        n_rotations=_CASCOR_NUM_ROTATIONS,
        clockwise=_CASCOR_CLOCKWISE,
        distribution=_CASCOR_DISTRIBUTION_FACTOR,
        random_value_scale=_CASCOR_RANDOM_VALUE_SCALE,
        default_origin=_CASCOR_DEFAULT_ORIGIN,
        default_radius=_CASCOR_DEFAULT_RADIUS,
        train_ratio=_CASCOR_TRAIN_RATIO,
        test_ratio=_CASCOR_TEST_RATIO,
        noise=_CASCOR_NOISE_FACTOR_DEFAULT,
        plot=_CASCOR_GENERATE_PLOTS_DEFAULT,
    )
    logger.info("Main: Completed solving SpiralProblem instance")

#####################################################################################################################################################################################################
# Main function to run the two spiral problem solution
# This is the entry point for the script.
if __name__ == "__main__":
    main()
