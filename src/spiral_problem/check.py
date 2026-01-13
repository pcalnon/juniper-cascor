#!/usr/bin/env python3
#####################################################################################################################################################################################################
# Project:       Cascade Correlation Neural Network
# File Name:     spiral_problem.py
# Author:        Paul Calnon
# Version:       1.0.1
# Date Created:  2025-07-29
# Last Modified: 2026-01-12
# License:       MIT License
#
# Description:
#    This file contains the functions and code needed to solve the two spiral problem using a Cascade Correlation Neural Network.
#
#
#####################################################################################################################################################################################################
# References:
#
#
#
#####################################################################################################################################################################################################
# TODO:
#
#
#
#####################################################################################################################################################################################################

import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from typing import List Tuple, Optional, Dict, Any
# from typing import List, Tuple, Dict, Any
from typing import Tuple
import logging
import random
# import math
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

from constants.constants import (
    _SPIRAL_PROBLEM_INPUT_SIZE,
    _SPIRAL_PROBLEM_OUTPUT_SIZE,
    _SPIRAL_PROBLEM_CANDIDATE_POOL_SIZE,
    _SPIRAL_PROBLEM_ACTIVATION_FUNCTION,
    _SPIRAL_PROBLEM_LEARNING_RATE,
    _SPIRAL_PROBLEM_MAX_HIDDEN_UNITS,
    _SPIRAL_PROBLEM_CORRELATION_THRESHOLD,
    _SPIRAL_PROBLEM_PATIENCE,
    _SPIRAL_PROBLEM_CANDIDATE_EPOCHS,
    _SPIRAL_PROBLEM_OUTPUT_EPOCHS,
    _SPIRAL_PROBLEM_STATUS_DISPLAY_FREQUENCY,
    _SPIRAL_PROBLEM_RANDOM_VALUE_SCALE,
    _SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT,
    _SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL,
    _SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT,
    _SPIRAL_PROBLEM_RANDOM_SEED,

    _SPIRAL_PROBLEM_NUM_SPIRALS,
    _SPIRAL_PROBLEM_NUM_ROTATIONS,
    _SPIRAL_PROBLEM_DEFAULT_ORIGIN,
    _SPIRAL_PROBLEM_DEFAULT_RADIUS,
    _SPIRAL_PROBLEM_CLOCKWISE,
    _SPIRAL_PROBLEM_DISTRIBUTION_FACTOR,
    _SPIRAL_PROBLEM_ORIG_POINTS,
    _SPIRAL_PROBLEM_MIN_NEW,
    _SPIRAL_PROBLEM_MAX_NEW,
    _SPIRAL_PROBLEM_MIN_ORIG,
    _SPIRAL_PROBLEM_MAX_ORIG,
    _SPIRAL_PROBLEM_TRAIN_RATIO,
    _SPIRAL_PROBLEM_TEST_RATIO,

    _SPIRAL_PROBLEM_LOG_FILE_NAME,
    _SPIRAL_PROBLEM_LOG_FILE_PATH,
    _SPIRAL_PROBLEM_LOG_FORMATTER_STRING,
    _SPIRAL_PROBLEM_LOG_DATE_FORMAT,
    _SPIRAL_PROBLEM_LOGLEVEL_DEFAULT,
)


class SpiralProblem(object):
    """
    Class to represent the two spiral problem.
    This class is used to generate the two spiral problem dataset and solve it using a Cascade Correlation Neural Network.
    """

    def __init__(
        self,
        _SpiralProblem__n_spirals: int = _SPIRAL_PROBLEM_NUM_SPIRALS,
        _SpiralProblem__n_points: int = _SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL,
        _SpiralProblem__n_rotations: int = _SPIRAL_PROBLEM_NUM_ROTATIONS,
        _SpiralProblem__clockwise: bool = _SPIRAL_PROBLEM_CLOCKWISE,                          # True for clockwise spirals, False for counter-clockwise
        _SpiralProblem__noise: float = _SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT,
        _SpiralProblem__distribution: float = _SPIRAL_PROBLEM_DISTRIBUTION_FACTOR,
        _SpiralProblem__random_seed: int = _SPIRAL_PROBLEM_RANDOM_SEED,  # Default random seed for reproducibility
        _SpiralProblem__train_ratio: float = _SPIRAL_PROBLEM_TRAIN_RATIO,
        _SpiralProblem__test_ratio: float = _SPIRAL_PROBLEM_TEST_RATIO,
        _SpiralProblem__log_file_name: str = _SPIRAL_PROBLEM_LOG_FILE_NAME,
        _SpiralProblem__log_file_path: str = _SPIRAL_PROBLEM_LOG_FILE_PATH,
        _SpiralProblem__logging_level: int = _SPIRAL_PROBLEM_LOGLEVEL_DEFAULT,
        _SpiralProblem__log_format: str = _SPIRAL_PROBLEM_LOG_FORMATTER_STRING,
        _SpiralProblem__log_date_format: str = _SPIRAL_PROBLEM_LOG_DATE_FORMAT,
    ):
        self.logger = logging.getLogger(_SpiralProblem__log_file_path)  # Use the log file path as the logger name
        self.logger.setLevel(_SpiralProblem__logging_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(fmt=_SpiralProblem__log_format, datefmt=_SpiralProblem__log_date_format)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.info("SpiralProblem: __init__: Completed initialization of Logging for Cascade Correlation Network.")

        self.n_spirals = _SpiralProblem__n_spirals
        self.n_points = _SpiralProblem__n_points
        self.n_rotations = _SpiralProblem__n_rotations
        self.clockwise = _SpiralProblem__clockwise
        self.noise = _SpiralProblem__noise
        self.distribution = _SpiralProblem__distribution
        self.random_seed = _SpiralProblem__random_seed  # Default random seed for reproducibility
        self.train_ratio = _SpiralProblem__train_ratio
        self.test_ratio = _SpiralProblem__test_ratio
        self.logger.debug(f"SpiralProblem: __init__: initialized with {self.n_spirals} spirals,  {self.n_rotations} rotations, and {self.n_points} points per spiral.")

        torch.manual_seed(self.random_seed) # Set random seed for reproducibility
        np.random.seed(self.random_seed)    # Set random seed for reproducibility
        random.seed(self.random_seed)       # Set random seed for reproducibility

        # Create the cascade correlation network
        self.network = CascadeCorrelationNetwork(
            _CascadeCorrelationNetwork__input_size=_SPIRAL_PROBLEM_INPUT_SIZE,
            _CascadeCorrelationNetwork__output_size=_SPIRAL_PROBLEM_OUTPUT_SIZE,
            _CascadeCorrelationNetwork__candidate_pool_size=_SPIRAL_PROBLEM_CANDIDATE_POOL_SIZE,
            _CascadeCorrelationNetwork__activation_function=_SPIRAL_PROBLEM_ACTIVATION_FUNCTION,
            _CascadeCorrelationNetwork__learning_rate=_SPIRAL_PROBLEM_LEARNING_RATE,
            _CascadeCorrelationNetwork__max_hidden_units=_SPIRAL_PROBLEM_MAX_HIDDEN_UNITS,
            _CascadeCorrelationNetwork__correlation_threshold=_SPIRAL_PROBLEM_CORRELATION_THRESHOLD,
            _CascadeCorrelationNetwork__patience=_SPIRAL_PROBLEM_PATIENCE,
            _CascadeCorrelationNetwork__candidate_epochs=_SPIRAL_PROBLEM_CANDIDATE_EPOCHS,
            _CascadeCorrelationNetwork__output_epochs=_SPIRAL_PROBLEM_OUTPUT_EPOCHS,
            _CascadeCorrelationNetwork__status_display_frequency=_SPIRAL_PROBLEM_STATUS_DISPLAY_FREQUENCY,
            _CascadeCorrelationNetwork__random_value_scale=_SPIRAL_PROBLEM_RANDOM_VALUE_SCALE,
            _CascadeCorrelationNetwork__log_format=_SPIRAL_PROBLEM_LOG_FORMATTER_STRING,
            _CascadeCorrelationNetwork__log_date_format=_SPIRAL_PROBLEM_LOG_DATE_FORMAT,
            _CascadeCorrelationNetwork__logging_level=_SPIRAL_PROBLEM_LOGLEVEL_DEFAULT,
        )
        self.logger.debug("SpiralProblem: solve_n_spiral_problem: Created Cascade Correlation Network")


    #####################################################################################################################################################################################################
    # Define function to generate the two spiral problem dataset.
    # TODO: Convert this to use spiral problem in Project Data Dir.
    def generate_n_spiral_dataset(
        self,
        min_new=_SPIRAL_PROBLEM_MIN_NEW,                              # Minimum value for the new points
        max_new=_SPIRAL_PROBLEM_MAX_NEW,                              # Maximum value for the new points
        min_orig=_SPIRAL_PROBLEM_MIN_ORIG,                            # Minimum value for the original points
        max_orig=_SPIRAL_PROBLEM_MAX_ORIG,                            # Maximum value for the original points
        orig_points=_SPIRAL_PROBLEM_ORIG_POINTS,                      # User provided data points or None
        train_ratio=_SPIRAL_PROBLEM_TRAIN_RATIO,
        test_ratio=_SPIRAL_PROBLEM_TEST_RATIO,
        clockwise=_SPIRAL_PROBLEM_CLOCKWISE,                          # True for clockwise spirals, False for counter-clockwise
        num_spirals=_SPIRAL_PROBLEM_NUM_SPIRALS,
        num_rotations=_SPIRAL_PROBLEM_NUM_ROTATIONS,
        num_points=_SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL,          # Number of points per spiral
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
        Raises:
            ValueError: If the input parameters are invalid.
            ValueError: If train_ratio and test_ratio do not sum to 1.0.
            ValueError: If train_ratio or test_ratio are not between 0.0 and 1.0.
        Notes:
            - The function generates a dataset of n spirals, each with a specified number of points, noise level, and rotation direction.
            - The spirals are generated in a clockwise or counter-clockwise direction based on the `clockwise` parameter.
            - The points are scaled from the original range [min_orig, max_orig] to the new range [min_new, max_new].
            - The function returns the input features and one-hot encoded targets as PyTorch tensors.
            - The function uses the `default_origin` and `default_radius` to calculate the radius of the spirals.
            - The function uses the `distribution` to apply a degree of rotation to the spiral points.
            - The function generates the spirals using the `generate_spiral_data` function.
        Returns:
            x: Input features as a PyTorch tensor of shape (total_points, 2).
            y: One-hot encoded targets as a PyTorch tensor of shape (total_points, num_spirals).


        """
        # Initialize Spiral Problem input parameters

        if not np.isclose(train_ratio + test_ratio, 1.0):  # Validate ratios
            raise ValueError("SpiralProblem: generate_n_spiral_data: Train and test ratios must sum to 1.0")

        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Generating {num_spirals} spirals with {num_points} points each, direction: {'clockwise' if clockwise else 'counter-clockwise'}, noise level: {noise_level}, distribution factor: {distribution}")
        direction = (1, -1)[clockwise]  # Determine the direction of the spiral based on clockwise parameter

        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Direction: {'clockwise' if direction == 1 else 'counter-clockwise'}")
        total_points = num_spirals * num_points

        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Total points to generate: {total_points}")
        radius = default_radius - default_origin  # Calculate the radius based on the default origin and radius

        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Default origin: {default_origin}, Default radius: {default_radius}, Calculated radius: {radius}")
        x = np.zeros((total_points, 2), dtype=np.float32)  # Initialize the input features array

        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Initialized input features array x with shape: {x.shape}, Type: {type(x)}, Value:\n{x}")
        y = np.zeros(total_points, dtype=int)

        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Initialized target labels array y with shape: {y.shape}, Type: {type(y)}, Value:\n{y}")
        n_samples = total_points  # Fix: should be total_points, not x.size

        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Total samples to generate: {n_samples}")
        train_end = int(train_ratio * n_samples)

        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Training set end index: {train_end}")
        test_end = train_end + int(test_ratio * n_samples)

        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Test set end index: {test_end}")

        for i in range(num_spirals):
            self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Generating spiral {i + 1}/{num_spirals} with {num_points} points...")
            indices = np.arange(i * num_points, (i + 1) * num_points, dtype=int)
            self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Indices for spiral {i + 1}: Type: {type(indices)}, Value:\n{indices}")
            points_orig = orig_points if orig_points is not None else np.random.rand(num_points)  # User provided data points or Generate random values in the range [0.0, 1.0] if orig_points is None
            points_unity = (points_orig - min_orig) / (max_orig - min_orig)  # Scale the original--or randomly generated--points to the unity range [0.0, 1.0]
            points_scaled = (points_unity * (max_new - min_new)) + min_new  # Scale the points from the unity range [0.0, 1.0] to the new range [min_new, max_new]
            self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Spiral {i + 1} points: Type: {type(points_scaled)}, Value:\n{points_scaled}")
            total_degrees = num_rotations * 360
            angle_offset = (direction * 2 * np.pi * i / num_spirals)  # Calculate the angle offset for this spiral
            random_degrees = (np.random.rand(num_points) * total_degrees)  # Fix: use num_points instead of points_scaled
            adjusted_degrees = np.power(random_degrees, distribution)  # Apply the transformation to the degrees
            radians = adjusted_degrees * (np.pi / 180.0)
            theta = radians + angle_offset

            # Calculate the x and y coordinates for the spiral points
            x[indices, 0] = radius * np.cos(direction * theta)  # x-coordinate
            x[indices, 1] = radius * np.sin(direction * theta)  # y-coordinate
            noise = np.random.normal(0, noise_level, (num_points, 2))
            x[indices] += noise
            self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Spiral {i + 1} points after noise: Type: {type(x[indices])}, Value:\n{x[indices]}")
            y[indices] = i  # Assign the current spiral index as the label
            self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Spiral {i + 1} labels: Type: {type(y[indices])}, Value:\n{y[indices]}")

        x_tensor = torch.tensor(x, dtype=torch.float32)  # Convert to PyTorch tensors
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Final input features x: Shape: {x_tensor.shape}, Type: {type(x_tensor)}, Value:\n{x_tensor}")
        y_one_hot = torch.zeros((total_points, num_spirals), dtype=torch.float32)  # Create one-hot encoded targets
        y_one_hot.scatter_(1, torch.tensor(y).unsqueeze(1), 1)
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Final one-hot encoded targets y: Shape: {y_one_hot.shape}, Type: {type(y_one_hot)}, Value:\n{y_one_hot}")

        # Shuffle the data
        indices = torch.randperm(x_tensor.size(0))
        x_shuffled = x_tensor[indices]
        y_shuffled = y_one_hot[indices]

        # Split the data
        # x_train, y_train = x_shuffled[:train_end], y_shuffled[:train_end]
        # x_test, y_test = x_shuffled[train_end:test_end], y_shuffled[train_end:test_end]
        partitioned_dataset = self.split_dataset(total_points=total_points, partitions=(train_ratio, test_ratio), x=x_shuffled, y=y_shuffled)
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Partitioned dataset: {partitioned_dataset}")
        (x_train, y_train), (x_test, y_test) = partitioned_dataset
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Training set x: Shape: {x_train.shape}, Type: {type(x_train)}, y: Shape: {y_train.shape}, Type: {type(y_train)}")
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Test set x: Shape: {x_test.shape}, Type: {type(x_test)}, y: Shape: {y_test.shape}, Type: {type(y_test)}")

        partitioned_dataset = partitioned_dataset + (x_shuffled, y_shuffled)  # Append the full dataset to the partitioned dataset
        self.logger.debug(f"SpiralProblem: generate_n_spiral_data: Full dataset x: Shape: {x_shuffled.shape}, Type: {type(x_shuffled)}, y: Shape: {y_shuffled.shape}, Type: {type(y_shuffled)}")

        # Return training and test sets, and the full dataset
        #return partitioned_dataset
        return (x_train, y_train), (x_test, y_test), (x_shuffled, y_shuffled)  # Return training and test sets, and the full dataset


    def make_noise(
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


    def split_dataset(
        self,
        total_points: int = None,
        partitions: tuple = None,
        # x: torch.Tensor[] = None,
        x = None,
        # y: torch.Tensor[] = None,
        y = None,
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
        # Check if total_points and partitions are provided
        if total_points is None or partitions is None:
            raise ValueError("SpiralProblem: split_dataset: total_points and partitions must be provided.")
        # Check if x and y are provided
        if x is None or y is None:
            raise ValueError("SpiralProblem: split_dataset: Torch Tensors, x and y must be provided.")
        # Check if total_points is a positive integer
        if not isinstance(total_points, int) or total_points <= 0:
            raise ValueError(f"SpiralProblem: split_dataset: total_points must be a positive integer, but got {total_points}.")
        # Check if partitions is a tuple of floats
        if not isinstance(partitions, tuple) or not all(isinstance(p, (float, int)) for p in partitions):
            raise ValueError(f"SpiralProblem: split_dataset: partitions must be a tuple of floats, but got {partitions}.")
        # Check if the length of partitions is greater than 0
        if not partitions:
            raise ValueError("SpiralProblem: split_dataset: partitions must contain at least one partition.")
        # Check if x and y have the same length as total_points
        if x.size(0) != total_points or y.size(0) != total_points:
            raise ValueError(f"SpiralProblem: split_dataset: x and y must have the same length as total_points, but got x: {x.size(0)}, y: {y.size(0)}, total_points: {total_points}.")
        # Check if partitions are valid
        partitions_sum = sum(partitions)
        if not np.isclose(partitions_sum, 1.0):
            raise ValueError(f"SpiralProblem: split_dataset: Partitions must sum to 1.0, but got {partitions_sum}.")
        # Calculate the dataset partitions based on the provided ratios
        self.logger.debug(f"SpiralProblem: split_dataset: Splitting dataset with total points: {total_points}, partitions: {partitions}, x shape: {x.shape}, y shape: {y.shape}.")
        dataset_partitions = ()
        partition_start = 0
        for partition in partitions:
            if not (0.0 <= partition <= 1.0):
                raise ValueError(f"SpiralProblem: split_dataset: Partition {partition} must be between 0.0 and 1.0.")
            # partition_end = partition_start + self.dataset_split_index_end( total_points=total_points, split_ratio=partition,)
            # partition_x, partition_y = x[partition_start:partition_end], y[partition_start:partition_end]
            # dataset_partitions.append(tuple(partition_x, partition_y))
            partition_end = self.find_partition_index_end( partition_start=partition_start, total_points=total_points, partition=partition)
            # dataset_partitions = dataset_partitions + tuple(x[partition_start:partition_end], y[partition_start:partition_end])
            # dataset_partitions = dataset_partitions + ((x[partition_start:partition_end], y[partition_start:partition_end]))
            current_partition = x[partition_start:partition_end], y[partition_start:partition_end]
            self.logger.debug(f"SpiralProblem: split_dataset: Current Partition: Length: {len(current_partition)}, Value:\n{current_partition}")

            self.logger.debug(f"SpiralProblem: split_dataset: Pre-appended length of dataset_partitions: {len(dataset_partitions)}")
            # dataset_partitions = (x[partition_start:partition_end], y[partition_start:partition_end],) if len(dataset_partitions) == 0 else (dataset_partitions,) + (x[partition_start:partition_end], y[partition_start:partition_end],)
            # dataset_partitions = (current_partition,) if len(dataset_partitions) == 0 else (dataset_partitions,) + (current_partition,) if len(dataset_partitions) == 1 else dataset_partitions + (current_partition,)
            # dataset_partitions = (current_partition,) if len(dataset_partitions) == 0 else dataset_partitions + (current_partition,) if len(dataset_partitions) == 1 else dataset_partitions + (current_partition,)
            dataset_partitions = (current_partition,) if len(dataset_partitions) == 0 else dataset_partitions + (current_partition,)
            self.logger.debug(f"SpiralProblem: split_dataset: Current Value of Dataset Partitions: Partition Start: {partition_start}, Partition End: {partition_end}, Partition Ratio: {partition}. Value:\n{dataset_partitions}")

            partition_start = partition_end
        self.logger.debug(f"SpiralProblem: split_dataset: Dataset partitions Indices created with {len(dataset_partitions)} partitions, Partition Index Values:\n{dataset_partitions}.")
        return dataset_partitions


    def find_partition_index_end(
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
        self.logger.debug(f"SpiralProblem: find_partition_index_end: Calculating partition end index for partition start: {partition_start}, total points: {total_points}, partition ratio: {partition}")
        partition_end = partition_start + self.dataset_split_index_end(
            total_points=total_points,
            split_ratio=partition,
        )
        self.logger.debug(f"SpiralProblem: find_partition_index_end: Calculated partition end index: {partition_end}")
        return partition_end

    def dataset_split_index_end(
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
        index_end = int(split_ratio * total_points)  # Calculate the center index for splitting
        self.logger.debug(f"SpiralProblem: dataset_split_index_end: Calculated index end for split ratio {split_ratio} and total points {total_points}: {index_end}")
        if index_end < 0 or index_end > total_points:
            raise ValueError(f"SpiralProblem: dataset_split_index_end: Invalid index end: {index_end}. Must be between 0 and {total_points}.")
        self.logger.debug(f"SpiralProblem: dataset_split_index_end: Valid index end: {index_end}")
        return index_end


    #####################################################################################################################################################################################################
    # Define function to solve the two spiral problem using Cascade Correlation Network.
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
            Solve the two spiral problem using Cascade Correlation Network. The dataset is split into training and test sets based on the given ratios.
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
        self.n_points = n_points if n_points is not None else _SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL if self.n_points is None else self.n_points  # Use class attribute if n_points is None
        self.n_spirals = n_spirals if n_spirals is not None else _SPIRAL_PROBLEM_NUM_SPIRALS if self.n_spirals is None else self.n_spirals  # Use class attribute if n_spirals is None
        self.n_rotations = n_rotations if n_rotations is not None else _SPIRAL_PROBLEM_NUM_ROTATIONS if self.n_rotations is None else self.n_rotations  # Use class attribute if n_rotations is None
        self.clockwise = clockwise if clockwise is not None else _SPIRAL_PROBLEM_CLOCKWISE if self.clockwise is None else self.clockwise  # Use class attribute if clockwise is None
        self.noise = noise if noise is not None else _SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT if self.noise is None else self.noise  # Use class attribute if noise is None
        self.distribution = distribution if distribution is not None else _SPIRAL_PROBLEM_DISTRIBUTION_FACTOR if self.distribution is None else self.distribution  # Use class attribute if distribution is None
        self.test_ratio = test_ratio if test_ratio is not None else _SPIRAL_PROBLEM_TEST_RATIO if self.test_ratio is None else self.test_ratio  # Use class attribute if test_ratio is None
        self.train_ratio = train_ratio if train_ratio is not None else _SPIRAL_PROBLEM_TRAIN_RATIO if self.train_ratio is None else self.train_ratio  # Use class attribute if train_ratio is None
        self.plot = plot if plot is not None else _SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT if self.plot is None else self.plot  # Use default if plot is None

        # Generate the n spiral dataset
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Generating two spiral dataset with {self.n_points} points and noise level {self.noise}.")

        # Generate the full N Spiral dataset, new version
        self.logger.debug("SpiralProblem: solve_n_spiral_problem: Generating full N Spiral dataset, new version")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Initialized Values for Parameters: n_spirals: {self.n_spirals}, n_points: {self.n_points}, n_rotations: {self.n_rotations}, clockwise: {self.clockwise}, noise: {self.noise}, distribution: {self.distribution}, train_ratio: {self.train_ratio}, test_ratio: {self.test_ratio}")
        # x_n, y_n = generate_n_spiral_dataset(
        # (self.x_train, self.y_train,), (self.x_test, self.y_test,), (self.x_full, self.y_full,) = self.generate_n_spiral_dataset(
        # (dataset_partitions,) = self.generate_n_spiral_dataset(
        dataset_partitions = self.generate_n_spiral_dataset(
            # orig_points=None,  # Use default random points if not provided
            # min_new=_SPIRAL_PROBLEM_MIN_NEW,
            # max_new=_SPIRAL_PROBLEM_MAX_NEW,
            # min_orig=_SPIRAL_PROBLEM_MIN_ORIG,
            # max_orig=_SPIRAL_PROBLEM_MAX_ORIG,
            # default_origin=_SPIRAL_PROBLEM_DEFAULT_ORIGIN,
            # default_radius=_SPIRAL_PROBLEM_DEFAULT_RADIUS,
            n_spirals=self.n_spirals,
            n_points=self.n_points,
            n_rotations=self.n_rotations,
            clockwise=self.clockwise,
            noise=self.noise,
            distribution=self.distribution,
            train_ratio=self.train_ratio,
            test_ratio=self.test_ratio,
        )
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Partitioned dataset: Size: {len(dataset_partitions)}, Type: {type(dataset_partitions)}, Value:\n{dataset_partitions}")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Generated N spiral dataset with {self.n_points} points and noise level {self.noise}.")

        # (self.x_train, self.y_train,), (self.x_test, self.y_test,), (self.x_full, self.y_full,) = dataset_partitions
        train_partition, test_partition, full_partition = dataset_partitions
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Train Partition: Type: {type(train_partition)}, Length: {len(train_partition)}, Value:\n{train_partition}")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Test Partition: Type: {type(test_partition)}, Length: {len(test_partition)}, Value:\n{test_partition}")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Full Partition: Type: {type(full_partition)}, Length: {len(full_partition)}, Value:\n{full_partition}")

        self.x_train, self.y_train = train_partition
        self.x_test, self.y_test = test_partition
        self.x_full, self.y_full = full_partition


        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Dataset x_full: Shape: {self.x_full.shape}, Type: {type(self.x_full)}, Value:\n{self.x_full}")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Dataset y_full: Shape: {self.y_full.shape}, Type: {type(self.y_full)}, Value:\n{self.y_full}")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Dataset x_train: Shape: {self.x_train.shape}, Type: {type(self.x_train)}, Value:\n{self.x_train}")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Dataset y_train: Shape: {self.y_train.shape}, Type: {type(self.y_train)}, Value:\n{self.y_train}")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Dataset x_test: Shape: {self.x_test.shape}, Type: {type(self.x_test)}, Value:\n{self.x_test}")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Dataset y_test: Shape: {self.y_test.shape}, Type: {type(self.y_test)}, Value:\n{self.y_test}")

        self.x = self.x_train  # Use training data for fitting the network
        self.y = self.y_train  # Use training labels for fitting the network
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Full dataset x: Shape: {self.x.shape}, Type: {type(self.x)}, Value:\n{self.x}")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Full dataset y: Shape: {self.y.shape}, Type: {type(self.y)}, Value:\n{self.y}")

        # Perform initial plot of the 2 Spiral Problem dataset
        if self.plot:
            self.logger.debug("SpiralProblem: solve_n_spiral_problem: Performing initial plot of the N Spiral Problem dataset")
            self.network.plot_dataset(
                x=self.x_full,
                y=self.y_full,
                title=f"N Spiral Problem: {self.n_spirals} Spirals, {self.n_points} Points Each, Noise Factor: {self.noise}",
            )

        # Train the network
        self.logger.debug("SpiralProblem: solve_n_spiral_problem: Created Cascade Correlation Network...")
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Cascade Correlation Network: \n{self.network}")
        self.history = self.network.fit(self.x, self.y, max_epochs=_SPIRAL_PROBLEM_OUTPUT_EPOCHS,)
        self.logger.debug(f"SpiralProblem: solve_n_spiral_problem: Training history: {self.history}")

        # Print summary
        self.network.summary()

        # Plot results
        if self.plot:
            self.network.plot_decision_boundary(self.x, self.y, "N Spiral Problem - Decision Boundary")
            self.network.plot_training_history()

        # return (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_full, self.y_full)
        # return (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_full, self.y_full)

    def evaluate(
        self,
        n_points=None,
        n_spirals=None,
        n_rotations=None,
        clockwise=None,
        noise=None,
        distribution=None,
        plot=None,
        test_ratio=None,
        train_ratio=None,
    ) -> None:
        # Set parameters for the two spiral problem
        self.n_points = n_points if n_points is not None else _SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL if self.n_points is None else self.n_points
        self.n_spirals = n_spirals if n_spirals is not None else _SPIRAL_PROBLEM_NUM_SPIRALS if self.n_spirals is None else self.n_spirals
        self.n_rotations = n_rotations if n_rotations is not None else _SPIRAL_PROBLEM_NUM_ROTATIONS if self.n_rotations is None else self.n_rotations
        self.clockwise = clockwise if clockwise is not None else _SPIRAL_PROBLEM_CLOCKWISE if self.clockwise is None else self.clockwise
        self.distribution = distribution if distribution is not None else _SPIRAL_PROBLEM_DISTRIBUTION_FACTOR if self.distribution is None else self.distribution
        self.noise = noise if noise is not None else _SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT if self.noise is None else self.noise
        self.plot = plot if plot is not None else _SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT if self.plot is None else self.plot
        self.train_ratio = train_ratio if train_ratio is not None else _SPIRAL_PROBLEM_TRAIN_RATIO if self.train_ratio is None else self.train_ratio
        self.test_ratio = test_ratio if test_ratio is not None else _SPIRAL_PROBLEM_TEST_RATIO if self.test_ratio is None else self.test_ratio
        # Solve the two spiral problem
        self.logger.debug("SpiralProblem: main: Solving the two spiral problem with Cascade Correlation...")
        # self.network, self.history, self.train_data, self.test_data = self.solve_n_spiral_problem(n_points=_SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL, noise=_SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT,)
        # self.train_data, self.test_data = self.solve_n_spiral_problem(
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
        self.logger.debug(f"SpiralProblem: main: Training Data: Type: {type(self.train_data)}, Shape: {self.train_data.shape}, Value:\n{self.train_data}")
        self.logger.debug(f"SpiralProblem: main: Test Data: Type: {type(self.test_data)}, Shape: {self.test_data.shape}, Value:\n{self.test_data}")

        # Print training dataset shapes and types
        # (self.train_x, self.train_y) = self.train_data
        self.logger.debug(f"SpiralProblem: main: Train Data: X Shape: {self.train_x.shape}, X Type: {type(self.train_x)}, X Value:\n{self.train_x}")
        self.logger.debug(f"SpiralProblem: main: Train Data: Y Shape: {self.train_y.shape}, Y Type: {type(self.train_y)}, Y Value:\n{self.train_y}")

        # Print test dataset shapes and types
        # (self.test_x, self.test_y) = self.test_data
        self.logger.debug(f"SpiralProblem: main: Test Data: X Shape: {self.test_x.shape}, X Type: {type(self.test_x)}, X Value:\n{self.test_x}")
        self.logger.debug(f"SpiralProblem: main: Test Data: Y Shape: {self.test_y.shape}, Y Type: {type(self.test_y)}, Y Value:\n{self.test_y}")

        # Calculate and log the accuracy on the training and test sets
        self.train_accuracy = self.network.calculate_accuracy(x=self.train_x, y=self.train_y)
        self.logger.debug(f"SpiralProblem: main: Train accuracy on the two spiral problem: {self.train_accuracy:.4f}")
        self.test_accuracy = self.network.calculate_accuracy(x=self.test_x, y=self.test_y)
        self.logger.debug(f"SpiralProblem: main: Test accuracy on the two spiral problem: {self.test_accuracy:.4f}")

        # Evaluate the Final Accuracy Percentages
        self.train_accuracy_percent = self.train_accuracy * 100
        self.logger.debug(f"SpiralProblem: main: Final Train accuracy on the two spiral problem: {self.train_accuracy_percent:.2f}%")
        self.test_accuracy_percent = self.test_accuracy * 100
        self.logger.debug(f"SpiralProblem: main: Final Test accuracy on the two spiral problem: {self.test_accuracy_percent:.2f}%")

        # Print final accuracy
        self.network.summary()
        self.logger.debug(f"SpiralProblem: main: Training History: {self.history}")
        self.logger.debug(f"SpiralProblem: main: Dataset: Train: x:\n{self.train_x}\ny:\n{self.train_y}\nTest: x:\n{self.test_x}\ny:\n{self.test_y}")
        self.logger.debug(f"SpiralProblem: main: Dataset shape: Train: x: {self.train_x.shape}, y: {self.train_y.shape}, Test: x: {self.test_x.shape}, y: {self.test_y.shape}")
        self.logger.debug(f"SpiralProblem: main: Final accuracy on the two spiral problem: Training: {self.train_accuracy_percent:.2f}%")
        self.logger.debug(f"SpiralProblem: main: Final accuracy on the two spiral problem: Testing: {self.test_accuracy_percent:.2f}%")