#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     candidate_unit.py
# Author:        Paul Calnon
#
# Date Created:  2025-06-11
# Last Modified: 2026-01-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This module implements a candidate unit for the Cascade Correlation Neural Network.
#
#####################################################################################################################################################################################################
# Notes:
#   - This file contains the CandidateUnit class, which represents a candidate unit in the Cascade Correlation Neural Network.
#   - The class uses PyTorch for tensor computations.
#   - The CandidateUnit class has a constructor that initializes the necessary components for the candidate unit.
#   - The class also includes methods for training and evaluating the candidate unit.
#   - The CandidateUnit class also includes a method for generating random weights and biases.
#   - The CandidateUnit class also includes a method for training the candidate unit using a given dataset.
#
#####################################################################################################################################################################################################
# References:
#
#
#####################################################################################################################################################################################################
# TODO:
#    - Consider selecting n-best candidate units from the pool based on correlation scores, for some constrained random value of n.
#      This would allow for the selected cohort of n candidate units to be added as a new layer in the cascor network.
#
#####################################################################################################################################################################################################
# COMPLETED:
#    - Integration of candidate unit code from cascor_spiral: candidate_unit.py
#
#####################################################################################################################################################################################################
import os

# import torch.nn as nn
import random
import sys
import uuid
from dataclasses import dataclass, field
from numbers import Integral, Real

# from typing import Optional
from typing import Any, Callable, Optional

import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]

from cascor_constants.constants import (  # _CANDIDATE_UNIT_POOL_SIZE,  # pyright: ignore[reportMissingImports]
    _CANDIDATE_UNIT_ACTIVATION_FUNCTION,
    _CANDIDATE_UNIT_CONVERGENCE_THRESHOLD,
    _CANDIDATE_UNIT_DISPLAY_FREQUENCY,
    _CANDIDATE_UNIT_EARLY_STOPPING,
    _CANDIDATE_UNIT_EPOCHS,
    _CANDIDATE_UNIT_EPOCHS_MAX,
    _CANDIDATE_UNIT_INPUT_SIZE,
    _CANDIDATE_UNIT_LEARNING_RATE,
    _CANDIDATE_UNIT_LOG_LEVEL_NAME,
    _CANDIDATE_UNIT_OUTPUT_SIZE,
    _CANDIDATE_UNIT_PATIENCE,
    _CANDIDATE_UNIT_RANDOM_MAX_VALUE,
    _CANDIDATE_UNIT_RANDOM_SEED,
    _CANDIDATE_UNIT_RANDOM_VALUE_SCALE,
    _CANDIDATE_UNIT_SEQUENCE_MAX_VALUE,
    _CANDIDATE_UNIT_STATUS_FREQUENCY,
)
from cascor_constants.constants_candidates.constants_candidates import _PROJECT_MODEL_CANDIDATE_MAX_ROLL_COUNT, _PROJECT_MODEL_CANDIDATE_RANDOM_MAX_VALUE  # pyright: ignore[reportMissingImports]
from log_config.logger.logger import Logger  # pyright: ignore[reportMissingImports]
from utils.utils import display_progress  # pyright: ignore[reportMissingImports]


#####################################################################################################################################################################################################
# Data classes for structured results
@dataclass
class CandidateTrainingResult:
    """Result from training a single candidate unit."""

    candidate_id: int = -1  # Changed from candidate_index for consistency
    candidate_uuid: Optional[str] = None
    correlation: float = 0.0  # Changed from best_correlation for consistency
    # candidate: Optional[any] = None  # ADDED - stores trained CandidateUnit object
    candidate: Optional[Any] = None  # ADDED - stores trained CandidateUnit object
    best_corr_idx: int = -1
    all_correlations: list[float] = field(default_factory=list)
    norm_output: Optional[torch.Tensor] = None
    norm_error: Optional[torch.Tensor] = None
    numerator: float = 0.0
    denominator: float = 1.0
    success: bool = True
    epochs_completed: int = 0
    error_message: Optional[str] = None
    round_id: Optional[str] = None  # PARALLEL-FIX (RC-5): Training round identifier for cross-round contamination detection


@dataclass
class CandidateParametersUpdate:
    x: torch.Tensor = None
    y: torch.Tensor = None
    residual_error: torch.Tensor = None
    learning_rate: float = _CANDIDATE_UNIT_LEARNING_RATE
    norm_output: torch.Tensor = None
    norm_error: torch.Tensor = None
    best_corr_idx: int = -1
    numerator: float = 0.0
    denominator: float = 1.0
    success: bool = True


@dataclass
class CandidateCorrelationCalculation:
    correlation: float = 0.0
    best_corr_idx: int = -1
    best_norm_output: torch.Tensor = None
    best_norm_error: torch.Tensor = None
    numerator: float = 0.0
    denominator: float = 0.0
    output: torch.Tensor = None
    residual_error: torch.Tensor = None


@dataclass
class EpochTrainedCandidate:
    candidate_id: int = -1
    epochs_completed: int = 0
    success: bool = True
    error_message: Optional[str] = None


#####################################################################################################################################################################################################
# Picklable Activation Function Wrapper
# CASCOR-P1-003: Multiprocessing Pickling Error Fix
# Extracted to utils.activation to eliminate duplication with cascade_correlation.py
#####################################################################################################################################################################################################
from utils.activation import ActivationWithDerivative  # noqa: F401, E402


#####################################################################################################################################################################################################
class CandidateUnit:
    @staticmethod
    def _coerce_int_like(value: Any, field_name: str) -> int:
        """Normalize JSON-decoded numeric payload fields used by seed/range APIs."""
        if isinstance(value, bool):
            raise TypeError(f"{field_name} must be an integer, not bool")
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real) and float(value).is_integer():
            return int(value)
        raise ValueError(f"{field_name} must be an integer-valued number, got {value!r}")

    #################################################################################################################################################################################################
    # Initialize a candidate unit.
    # This method initializes the weights and bias of the candidate unit, sets the activation function, and configures the logger.
    def __init__(
        self,
        # CandidateUnit__activation_function: callable = _CANDIDATE_UNIT_ACTIVATION_FUNCTION,
        CandidateUnit__activation_function: Callable[..., Any] = _CANDIDATE_UNIT_ACTIVATION_FUNCTION,
        CandidateUnit__display_frequency: int = _CANDIDATE_UNIT_DISPLAY_FREQUENCY,
        CandidateUnit__epochs: int = _CANDIDATE_UNIT_EPOCHS,
        CandidateUnit__epochs_max: int = _CANDIDATE_UNIT_EPOCHS_MAX,
        CandidateUnit__input_size: int = _CANDIDATE_UNIT_INPUT_SIZE,
        CandidateUnit__output_size: int = _CANDIDATE_UNIT_OUTPUT_SIZE,
        CandidateUnit__learning_rate: float = _CANDIDATE_UNIT_LEARNING_RATE,
        CandidateUnit__patience: int = _CANDIDATE_UNIT_PATIENCE,
        CandidateUnit__convergence_threshold: float = _CANDIDATE_UNIT_CONVERGENCE_THRESHOLD,
        CandidateUnit__early_stopping: bool = _CANDIDATE_UNIT_EARLY_STOPPING,
        CandidateUnit__status_frequency: int = _CANDIDATE_UNIT_STATUS_FREQUENCY,
        CandidateUnit__random_max_value: int = _CANDIDATE_UNIT_RANDOM_MAX_VALUE,
        CandidateUnit__sequence_max_value: int = _CANDIDATE_UNIT_SEQUENCE_MAX_VALUE,
        CandidateUnit__random_seed: int = _CANDIDATE_UNIT_RANDOM_SEED,
        CandidateUnit__random_value_scale: float = _CANDIDATE_UNIT_RANDOM_VALUE_SCALE,
        CandidateUnit__log_level_name: str = _CANDIDATE_UNIT_LOG_LEVEL_NAME,
        CandidateUnit__uuid: str = None,
        CandidateUnit__candidate_index: int = 0,
        **kwargs,
    ):
        # Call the superclass constructor
        super().__init__()

        # Initialize CandidateUnit class logger
        self.log_level_name = CandidateUnit__log_level_name or _CANDIDATE_UNIT_LOG_LEVEL_NAME
        self.logger = Logger
        self.logger.set_level(self.log_level_name)
        self.logger.info("CandidateUnit: __init__: Initializing Candidate Unit with Logger class.")

        # Initialize candidate index for unique seeding
        candidate_index = CandidateUnit__candidate_index if CandidateUnit__candidate_index is not None else 0
        self.candidate_index = self._coerce_int_like(candidate_index, "CandidateUnit__candidate_index")
        self.logger.verbose(f"CandidateUnit: __init__: Candidate index: {self.candidate_index}")

        # Initialize CandidateUnit class attributes for randomness
        random_seed = CandidateUnit__random_seed if CandidateUnit__random_seed is not None else _CANDIDATE_UNIT_RANDOM_SEED
        self.random_seed = self._coerce_int_like(random_seed, "CandidateUnit__random_seed")
        self.logger.verbose(f"CandidateUnit: __init__: Random seed: {self.random_seed}")
        random_max_value = CandidateUnit__random_max_value if CandidateUnit__random_max_value is not None else _CANDIDATE_UNIT_RANDOM_MAX_VALUE
        self.random_max_value = self._coerce_int_like(random_max_value, "CandidateUnit__random_max_value")
        self.logger.verbose(f"CandidateUnit: __init__: Random max value: {self.random_max_value}")
        sequence_max_value = CandidateUnit__sequence_max_value if CandidateUnit__sequence_max_value is not None else _CANDIDATE_UNIT_SEQUENCE_MAX_VALUE
        self.sequence_max_value = self._coerce_int_like(sequence_max_value, "CandidateUnit__sequence_max_value")
        self.logger.verbose(f"CandidateUnit: __init__: Random sequence max value: {self.sequence_max_value}")
        # Use unique seed per candidate to ensure different initializations
        unique_seed = self.random_seed + self.candidate_index if self.random_seed else self.candidate_index
        self._initialize_randomness(seed=unique_seed, max_value=self.sequence_max_value)

        # Initialize CandidateUnit class attributes with Input size, Output Size and Activation Function
        self.logger.trace("CandidateUnit: __init__: Initializing CandidateUnit class attributes with input parameters.")
        self.input_size = CandidateUnit__input_size
        self.logger.verbose(f"CandidateUnit: __init__: Input size: {self.input_size}")
        self.output_size = CandidateUnit__output_size
        self.logger.verbose(f"CandidateUnit: __init__: Output size: {self.output_size}")

        # Cache activation function wrapper to avoid recreating on every forward pass (P2 optimization)
        self.activation_fn = self._init_activation_with_derivative(CandidateUnit__activation_function)
        self.activation_fn_base = self.activation_fn.activation_fn
        self.logger.verbose(f"CandidateUnit: __init__: Base Activation function: {self.activation_fn_base}")
        self.logger.debug("CandidateUnit: __init__: Cached activation function wrapper")

        # Initialize CandidateUnit class attributes for training epochs
        self.epochs = CandidateUnit__epochs
        self.logger.verbose(f"CandidateUnit: __init__: Epochs: {self.epochs}")
        self.epochs_max = CandidateUnit__epochs_max
        self.logger.verbose(f"CandidateUnit: __init__: Max epochs: {self.epochs_max}")

        # Initialize CandidateUnit class attributes for learning rate and random value scale
        self.learning_rate = CandidateUnit__learning_rate
        self.logger.verbose(f"CandidateUnit: __init__: Learning rate: {self.learning_rate}")
        self.random_value_scale = CandidateUnit__random_value_scale
        self.logger.verbose(f"CandidateUnit: __init__: Random value scale: {self.random_value_scale}")

        # Initialize CandidateUnit class attributes for early stopping and patience
        self.early_stopping = CandidateUnit__early_stopping
        self.logger.verbose(f"CandidateUnit: __init__: Early stopping: {self.early_stopping}")
        self.patience = CandidateUnit__patience
        self.logger.verbose(f"CandidateUnit: __init__: Patience: {self.patience}")
        self.convergence_threshold = CandidateUnit__convergence_threshold
        self.logger.verbose(f"CandidateUnit: __init__: Convergence threshold: {self.convergence_threshold}")

        # Initialize candidate unit attributes with random weights and bias
        self.weights = torch.randn(self.input_size) * self.random_value_scale
        self.logger.verbose(f"CandidateUnit: __init__: Weights: {self.weights}")
        self.bias = torch.randn(1) * self.random_value_scale
        self.logger.verbose(f"CandidateUnit: __init__: Bias: {self.bias}")

        # Initialize candidate unit attributes with constants
        self.logger.trace("CandidateUnit: __init__: Initializing CandidateUnit Attributes with Constants (e.g., 0.0)")
        self.correlation = 0.0

        # # Initialize candidate display progress function with training candidate display frequency
        # self.logger.trace("CandidateUnit: __init__: Initializing candidate display progress function with training candidate display frequency")
        # self._candidate_display_progress = self._init_display_progress(display_frequency=self.display_frequency)
        # self.logger.verbose(f"CandidateUnit: __init__: Candidate display progress function initialized with display frequency: {self.display_frequency}, _candidate_display_progress = {self._candidate_display_progress}")

        # Initialize candidate unit UUID
        self.set_uuid(CandidateUnit__uuid)
        self.logger.verbose(f"CandidateUnit: __init__: UUID: {self.uuid}")

        # Initialize CandidateUnit class attributes for display frequency and status frequency
        self.display_frequency = CandidateUnit__display_frequency
        self.logger.verbose(f"CandidateUnit: __init__: Display frequency: {self.display_frequency}")
        self.status_frequency = CandidateUnit__status_frequency
        self.logger.verbose(f"CandidateUnit: __init__: Status frequency: {self.status_frequency}")

        # Initialize display progress frequency checker with candidate unit display frequency
        self._candidate_display_progress = self._init_display_progress(display_frequency=self.display_frequency)
        self.logger.verbose(f"CandidateUnit: __init__: Candidate display progress function initialized with display frequency: {self.display_frequency}, _candidate_display_progress = {self._candidate_display_progress}")
        self._candidate_display_status = self._init_display_status(display_status=self.status_frequency)
        self.logger.verbose(f"CandidateUnit: __init__: Candidate display status function initialized with status frequency: {self.status_frequency}, _candidate_display_status = {self._candidate_display_status}")

        self.logger.debug("CandidateUnit: __init__: Completed initialization of Candidate Unit")
        self.logger.trace("CandidateUnit: __init__: Completed the __init__ method for the Candidate Unit")

    #################################################################################################################################################################################################
    # Serialization support for multiprocessing
    def __getstate__(self):
        """Remove non-picklable items for multiprocessing and HDF5 serialization."""
        state = self.__dict__.copy()
        # Remove non-serializable/transient items
        state.pop("logger", None)
        state.pop("_candidate_display_progress", None)
        state.pop("_candidate_display_status", None)
        if "activation_fn" in state:
            state["activation_fn_base"] = state["activation_fn"]
        return state

    def __setstate__(self, state):
        """Restore instance from serialized state."""
        self.__dict__.update(state)
        if "activation_fn_base" not in state and hasattr(self, "activation_fn"):
            self.activation_fn_base = self.activation_fn
        # Recreate logger
        self.logger = Logger
        self.logger.set_level(self.log_level_name)
        # Display functions will be recreated lazily in train() when needed

    #################################################################################################################################################################################################
    # Helper method to perform initialization tasks for the __init__ method
    def _initialize_randomness(self, seed: Optional[int] = None, max_value: Optional[int] = None) -> None:
        """
        Description:
            Initialize randomness for the candidate unit.
        Args:
            seed: Optional seed for random number generation
            max_value: Optional maximum value for random number generation
        """
        self.logger.trace("CandidateUnit: _initialize_randomness: Initializing randomness for the candidate unit")
        seed = self._coerce_int_like(seed or _CANDIDATE_UNIT_RANDOM_SEED, "seed")
        self.logger.verbose(f"CandidateUnit: _initialize_randomness: Random seed set to: {seed}")
        # max_value = max_value or _CANDIDATE_UNIT_RANDOM_MAX_VALUE
        # max_value = 10000
        max_value = self._coerce_int_like(max_value or _PROJECT_MODEL_CANDIDATE_RANDOM_MAX_VALUE, "max_value")  # Using a small max value to limit the number of random calls needed to roll to the desired sequence
        self.logger.verbose(f"CandidateUnit: _initialize_randomness: Random max value set to: {max_value}")
        self._seed_random_generator(seed=seed, max_value=max_value, seeder=np.random.seed, generator=np.random.randint)
        self.logger.trace("CandidateUnit: _initialize_randomness: Completed initialization of numpy random generator with seed and sequence for the candidate unit")
        self._seed_random_generator(seed=seed, max_value=max_value, seeder=random.seed, generator=random.randint)
        self.logger.trace("CandidateUnit: _initialize_randomness: Completed initialization of random random generator with seed and sequence for the candidate unit")
        self._seed_random_generator(seed=seed, max_value=max_value, seeder=torch.manual_seed, generator=lambda min, max: torch.randint(min, max, ()))
        self.logger.trace("CandidateUnit: _initialize_randomness: Completed initialization of torch random generator with seed and sequence for the candidate unit")
        self._seed_random_generator(
            seed=seed,
            max_value=max_value,
            seeder=self._seed_hash,
            generator=None,
        )
        # Initialize CUDA random generator if available
        if torch.cuda.is_available():
            self.logger.trace("CandidateUnit: _initialize_randomness: CUDA is available, seeding CUDA random generator.")
            # self._seed_random_generator(seed=seed, max_value=max_value, seeder=torch.cuda.manual_seed, generator=lambda min, max: torch.cuda.randint(min, max, ()))
            self._seed_random_generator(seed=seed, max_value=max_value, seeder=torch.cuda.manual_seed, generator=lambda min, max: torch.rand(1, device="cuda"))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # def _seed_random_generator(self, seed: int = None, max_value: int = None, seeder: callable = None, generator: callable = None) -> None:
    def _seed_random_generator(self, seed: int = None, max_value: int = None, seeder: Callable[[int], Any] | None = None, generator: Callable[..., Any] | None = None) -> None:
        """
        Description:
            Seed the random generator for the candidate unit.
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
        self.logger.trace("CandidateUnit: _seed_random_generator: Seeding random module with seed and max value.")
        if seeder is None:
            self.logger.error("CandidateUnit: _seed_random_generator: Initialization of Random generator seed Failed: No seeder function provided.")
            return
        seeder(seed)
        self.logger.trace("CandidateUnit: _seed_random_generator: Random seed set for random module.")
        if generator is None:
            self.logger.trace("CandidateUnit: _seed_random_generator: No generator function provided, skipping random number generation and sequence rolling.")
            return
        # random_sequence = generator(0, max_value)
        # trunk-ignore(bandit/B311)
        random_sequence = random.randint(0, max_value)
        self.logger.verbose(f"CandidateUnit: _seed_random_generator: Random sequence number rolled to: {random_sequence}")
        self._roll_sequence_number(sequence=random_sequence, max_value=max_value, generator=generator)
        self.logger.trace("CandidateUnit: _seed_random_generator: Completed initialization of random generator with seed and sequence for the candidate unit")

    # def _roll_sequence_number(self, sequence: int = None, max_value: int = None, generator: callable = None) -> None:
    def _roll_sequence_number(self, sequence: int = None, max_value: int = None, generator: Callable[..., Any] | None = None) -> None:
        """
        Description:
            Roll the sequence number for the candidate unit.
        Args:
            sequence: The current sequence number
            max_value: The maximum value for the random number generator
            generator: The random number generator function
        Note:
            This method rolls the random generator discarding the first sequence number of integers for the candidate unit
        Returns:
            None
        """
        self.logger.trace("CandidateUnit: _roll_sequence_number: Rolling sequence number.")
        if generator is not None:
            # CASCOR-P1-008 FIX: Replaced list comprehension with simple loop to avoid OOM
            # OLD (can cause OOM if sequence is large - up to 2^32-1):
            # discard = [generator(0, max_value) for _ in range(sequence)]
            # NEW: Loop without storing, and cap roll count to prevent excessive iterations
            MAX_ROLL_COUNT = _PROJECT_MODEL_CANDIDATE_MAX_ROLL_COUNT
            roll_count = min(sequence, MAX_ROLL_COUNT) if sequence else 0
            for _ in range(roll_count):
                generator(0, max_value)
            self.logger.verbose(f"CandidateUnit: _roll_sequence_number: Discarded {roll_count} random values to roll to the desired sequence.")
            if sequence and sequence > MAX_ROLL_COUNT:
                self.logger.warning(f"CandidateUnit: _roll_sequence_number: Sequence {sequence} exceeded MAX_ROLL_COUNT {MAX_ROLL_COUNT}, capped at {MAX_ROLL_COUNT}")
            self.logger.verbose(f"CandidateUnit: _roll_sequence_number: Random Generator rolled for sequence number: {sequence}")
        self.logger.trace("CandidateUnit: _roll_sequence_number: Completed rolling of sequence number.")

    def _seed_hash(self, seed: int = None) -> None:
        """
        Description:
            Seed the hash function for the candidate unit.
        Args:
            seed: The seed value for the hash function
        """
        os.environ["PYTHONHASHSEED"] = str(seed)

    # def _init_activation_with_derivative(self, activation_fn: callable = None) -> ActivationWithDerivative:
    def _init_activation_with_derivative(self, activation_fn: Callable[..., Any] | None = None) -> ActivationWithDerivative:
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
        self.logger.trace("CandidateUnit: _init_activation_with_derivative: Validating activation function")
        activation_fn = (activation_fn, _CANDIDATE_UNIT_ACTIVATION_FUNCTION)[activation_fn is None]
        self.logger.debug(f"CandidateUnit: _init_activation_with_derivative: Using activation function: {activation_fn}")

        # CASCOR-P1-003: Use picklable ActivationWithDerivative class instead of local function
        # OLD: Local function - NOT picklable for multiprocessing!
        # self.logger.trace("CandidateUnit: _init_activation_with_derivative: Wrapping activation function to provide its derivative.")
        # def wrapped_activation(x, derivative: bool = False):
        #     if derivative:
        #         if activation_fn == torch.tanh:        # For tanh, derivative is 1 - tanh^2(x)
        #             return 1.0 - activation_fn(x)**2
        #         elif activation_fn == torch.sigmoid:   # For sigmoid, derivative is sigmoid(x) * (1 - sigmoid(x))
        #             y = activation_fn(x)
        #             return y * (1.0 - y)
        #         elif activation_fn == torch.relu:      # For ReLU, derivative is 1 for x > 0, 0 otherwise
        #             return (x > 0).float()
        #         else:                                  # Numerical approximation for other functions
        #             eps = 1e-6
        #             return (activation_fn(x + eps) - activation_fn(x - eps)) / (2 * eps)
        #     else:
        #         return activation_fn(x)
        # self.logger.verbose(f"CandidateUnit: _init_activation_with_derivative: Returning wrapped activation function: {wrapped_activation}.")
        # return wrapped_activation

        # NEW: Picklable class instance for multiprocessing compatibility
        self.logger.trace("CandidateUnit: _init_activation_with_derivative: Creating ActivationWithDerivative wrapper.")
        wrapped_activation = ActivationWithDerivative(activation_fn)
        self.logger.verbose(f"CandidateUnit: _init_activation_with_derivative: Returning wrapped activation function: {wrapped_activation}.")

        # Return the wrapped activation function
        self.logger.verbose(f"CandidateUnit: _init_activation_with_derivative: Returning wrapped activation function: Type: {type(wrapped_activation)}, Value: {wrapped_activation}.")
        self.logger.trace("CandidateUnit: _init_activation_with_derivative: Completed wrapping of activation function.")
        return wrapped_activation

    #################################################################################################################################################################################################
    # Calculate the correlation between the candidate unit output and the residual error.
    def forward(
        self,
        x: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Description:
            Forward pass through the candidate unit.
        Args:
            x: Input tensor
        Returns:
            Output of the candidate unit
        """
        self.logger.trace("CandidateUnit: forward: Starting forward pass through the candidate unit")
        self.logger.verbose(f"CandidateUnit: forward: Input shape: {x.shape}, Input length: {len(x)}")

        # Add shape guard for 1-D inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)
            self.logger.trace(f"CandidateUnit: forward: Reshaped 1-D input to 2-D: {x.shape}")

        self.logger.trace("CandidateUnit: forward: Calculating output using weights and bias")
        output = self.activation_fn(torch.sum(x * self.weights, dim=1) + self.bias)
        self.logger.debug(f"CandidateUnit: forward: Output shape: {output.shape}, Output length: {len(output)}")
        self.logger.trace("CandidateUnit: forward: Completed forward pass through the candidate unit")
        return output

    #################################################################################################################################################################################################
    # Define method to Train the candidate unit to maximize correlation with residual error.
    # NOTE:  residual error is calculated and provided by the Cascade Correlation Network, so this method or its child methods should not be generating it.
    # NOTE:      instead, the candidate unit should be trained so that its output minimizes the residual error provided by the Cascade Correlation Network.
    # NOTE:      this is also means that the residual error should remain the same for all training epochs for all candidate units in the pool.
    def train(
        self,
        x: torch.Tensor = None,
        epochs: int = _CANDIDATE_UNIT_EPOCHS_MAX,
        residual_error: torch.Tensor = None,
        learning_rate: float = _CANDIDATE_UNIT_LEARNING_RATE,
        display_frequency: int = _CANDIDATE_UNIT_DISPLAY_FREQUENCY,
    ) -> float:
        """
        Description:
            Train the candidate unit to maximize correlation with residual error.
            Backward-compatible API that returns only the final correlation value.
        Args:
            x: Input tensor
            residual_error: Residual error from the network
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            display_frequency: Frequency of displaying training progress
        Notes:
            This method delegates to train_detailed() but returns only the correlation float.
            For full training details including the CandidateTrainingResult object, use train_detailed().
            The full result is also stored in self.last_training_result for introspection.
        Raises:
            ValueError: If input tensor and residual error tensor have incompatible shapes
        Returns:
            Final correlation value as a float
        """
        result = self.train_detailed(
            x=x,
            epochs=epochs,
            residual_error=residual_error,
            learning_rate=learning_rate,
            display_frequency=display_frequency,
        )
        # Store for introspection
        self.last_training_result = result
        # Preserve original contract: return correlation as float
        return float(result.correlation)

    def train_detailed(
        self,
        x: torch.Tensor = None,
        epochs: int = _CANDIDATE_UNIT_EPOCHS_MAX,
        residual_error: torch.Tensor = None,
        learning_rate: float = _CANDIDATE_UNIT_LEARNING_RATE,
        display_frequency: int = _CANDIDATE_UNIT_DISPLAY_FREQUENCY,
        progress_callback=None,
    ) -> CandidateTrainingResult:
        """
        Description:
            Train the candidate unit to maximize correlation with residual error.
            Returns full training details as a CandidateTrainingResult dataclass.
        Args:
            x: Input tensor
            residual_error: Residual error from the network
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            display_frequency: Frequency of displaying training progress
        Notes:
            This method takes the input tensor, residual error, learning rate, and number of epochs.
            It performs a forward pass, calculates the correlation with the residual error, and updates the weights and bias accordingly.
            It handles both single-output and multi-output networks.
            For backward compatibility, use train() which returns only the correlation float.
        Raises:
            ValueError: If input tensor and residual error tensor have incompatible shapes
        Returns:
            CandidateTrainingResult containing full training details
        """
        self.logger.trace("CandidateUnit: train: Starting training of Candidate Unit")
        self.logger.debug(f"CandidateUnit: train: Input parameters - epochs: Type: {type(epochs)}, Value: {epochs}, learning_rate: Type: {type(learning_rate)}, Value: {learning_rate}, display_frequency: Type: {type(display_frequency)}, Value: {display_frequency}")
        self.logger.debug(f"CandidateUnit: train: Input Tensor (x): Shape: {x.shape}, Type: {type(x)}, Dtype: {x.dtype}, Dimensions: {x.dim()}, Length: {len(x)}")
        self.logger.debug(f"CandidateUnit: train: Residual Error Tensor: Shape: {residual_error.shape}, Type: {type(residual_error)}, Dtype: {residual_error.dtype}, Dimensions: {residual_error.dim()}, Length: {len(residual_error)}")
        self.logger.info(f"CandidateUnit: train: Training candidate unit for {epochs} epochs with learning rate {learning_rate} and display frequency {display_frequency}")

        # Initialize display progress frequency checker with candidate unit display frequency
        self._candidate_display_progress = self._init_display_progress(display_frequency=self.display_frequency)
        self.logger.verbose(f"CandidateUnit: train: Candidate display progress function initialized with display frequency: {self.display_frequency}, _candidate_display_progress = {self._candidate_display_progress}")
        self._candidate_display_status = self._init_display_status(display_status=self.status_frequency)
        self.logger.verbose(f"CandidateUnit: train: Candidate display status function initialized with status frequency: {self.status_frequency}, _candidate_display_status = {self._candidate_display_status}")

        # Initialize early stopping tracking variables
        best_correlation_so_far = 0.0
        epochs_without_improvement = 0
        early_stopped = False
        actual_epochs_completed = 0
        self.logger.debug(f"CandidateUnit: train: Early stopping enabled: {self.early_stopping}, Patience: {self.patience}")

        # Log constant metadata once before the loop (CR-062: hoist invariant values)
        _log_debug = self.logger.isEnabledFor(level=10)  # DEBUG
        _log_trace = self.logger.isEnabledFor(level=5)  # TRACE
        if _log_debug:
            self.logger.debug("CandidateUnit: train: Residual error shape: %s, dtype: %s, dims: %s", residual_error.shape, residual_error.dtype, residual_error.dim())

        # Train candidate unit for specified epochs to maximize correlation with residual error
        for epoch in range(epochs):
            actual_epochs_completed = epoch + 1
            if _log_trace:
                self.logger.trace("CandidateUnit: train: Starting training step for epoch %d", epoch + 1)
            self.logger.info("CandidateUnit: train: Forward pass: UUID: %s, epoch: %d", self.uuid, epoch + 1)
            output = self.forward(x)  # Forward pass for current input mini-batch (x)

            # Compute correlation with each output and use the maximum absolute correlation
            self.logger.info("CandidateUnit: train: Calculating correlation: UUID: %s, epoch: %d", self.uuid, epoch + 1)
            candidate_training_result = self._get_correlations(output=output, residual_error=residual_error)
            self.logger.info("CandidateUnit: train: Correlation result: UUID: %s, epoch: %d, correlation: %s", self.uuid, epoch + 1, candidate_training_result.correlation)
            if _log_debug:
                self.logger.debug("CandidateUnit: train: Correlation detail: self=%s, result=%s, all=%s", self.correlation, candidate_training_result.correlation, candidate_training_result.all_correlations)

            # Update weights and bias based on correlation
            candidate_parameters_update = CandidateParametersUpdate(
                x=x,
                y=output,
                residual_error=residual_error,
                learning_rate=learning_rate,
                norm_output=candidate_training_result.norm_output,
                norm_error=candidate_training_result.norm_error,
                best_corr_idx=candidate_training_result.best_corr_idx,
                numerator=candidate_training_result.numerator,
                denominator=candidate_training_result.denominator,
            )
            self._update_weights_and_bias(candidate_parameters_update=candidate_parameters_update)
            if _log_debug:
                self.logger.debug("CandidateUnit: train: Epoch %d: weights shape: %s", epoch + 1, self.weights.shape)

            # Update instance correlation for monitoring during training
            self.correlation = float(candidate_training_result.correlation)

            # Check for early stopping if enabled
            if self.early_stopping:
                current_abs_correlation = abs(candidate_training_result.correlation)
                if current_abs_correlation > abs(best_correlation_so_far) + self.convergence_threshold:
                    best_correlation_so_far = candidate_training_result.correlation
                    epochs_without_improvement = 0
                    self.logger.debug(f"CandidateUnit: train: Improved correlation to {best_correlation_so_far:.6f}, resetting patience counter")
                else:
                    epochs_without_improvement += 1
                    self.logger.debug(f"CandidateUnit: train: No improvement, patience counter: {epochs_without_improvement}/{self.patience}")

                if epochs_without_improvement >= self.patience:
                    self.logger.info(f"CandidateUnit: train: Early stopping at epoch {epoch + 1} - no improvement for {self.patience} epochs")
                    early_stopped = True
                    break

            # Throttled progress emission for real-time monitoring
            _pcb = progress_callback or getattr(self, "_progress_callback", None)
            if _pcb is not None and (epoch % 50 == 0 or epoch == epochs - 1):
                _pcb(
                    candidate_id=getattr(self, "candidate_index", 0),
                    candidate_uuid=str(getattr(self, "uuid", "")),
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    correlation=float(self.correlation),
                )

            # Display training progress at specified frequency
            try:
                self._display_training_progress(epoch, candidate_parameters_update, residual_error)
            except Exception as e:
                self.logger.error(f"CandidateUnit: train: Failed to display training progress: {str(e)}")
                import traceback

                self.logger.error(f"CandidateUnit: train: Traceback: {traceback.format_exc()}")
            self.logger.trace(f"CandidateUnit: train: Completed training step: For Epoch {epoch + 1}.")

        # Generate final output after training
        self.logger.trace("CandidateUnit: train: Calculating the final correlation after training")
        output = self.forward(x)
        self.logger.debug(f"CandidateUnit: train: Output Shape: {output.shape}, For Final Epoch")

        # Calculate final correlation
        self.logger.trace("CandidateUnit: train: Calculating the final correlation after training, For Final Epoch.")
        candidate_training_result = self._get_correlations(output=output, residual_error=residual_error)
        self.logger.debug(f"CandidateUnit: train: Final correlation: after training: {candidate_training_result.correlation}, For Final Epoch.")

        # Save actual epochs completed (account for early stopping)
        candidate_training_result.epochs_completed = actual_epochs_completed
        self.logger.debug(f"CandidateUnit: train: Completed epochs: {candidate_training_result.epochs_completed}, Early stopped: {early_stopped}, For Final Epoch.")

        # Extract the best correlation value and update instance variable
        if candidate_training_result and candidate_training_result.success and candidate_training_result.correlation != 0.0:
            self.correlation = float(candidate_training_result.correlation)
            self.logger.info(f"CandidateUnit: train: Final Correlation: UUID: {self.uuid}, Final correlation value: {self.correlation:.6f}, Best Corr Index: {candidate_training_result.best_corr_idx}")
        else:
            self.correlation = 0.0
            self.logger.warning("CandidateUnit: train: No valid correlations found, setting correlation to 0.0")

        self.logger.debug(f"CandidateUnit: train: Final correlation: Best correlation for Candidate Unit: {self.correlation:.6f}")
        self.logger.trace("CandidateUnit: train: Completed training of Candidate Unit")

        # Return the final correlation value
        return candidate_training_result

    #################################################################################################################################################################################################
    # Initialize display progress frequency checker with candidate unit display frequency
    def _display_training_progress(self, epoch, candidate_parameters_update, residual_error):
        """Display training progress at specified frequency intervals."""
        self.logger.debug("CandidateUnit: _display_training_progress: Checking if training progress should be displayed")
        self.logger.debug(f"CandidateUnit: _display_training_progress: Display frequency: {self.display_frequency}, Current Epoch: {epoch + 1}")
        # Reinitialize display function if needed
        if self._candidate_display_progress is None:
            self.logger.debug(f"CandidateUnit: _display_training_progress: Display function: Type: {type(self._candidate_display_progress)}, Value: {self._candidate_display_progress}")
            self.logger.debug(f"CandidateUnit: _display_training_progress: Display function is None, re-initializing with frequency: Type: {type(self.display_frequency)}, Value: {self.display_frequency}")
            self._candidate_display_progress = self._init_display_progress(display_frequency=self.display_frequency)

        # Display progress if epoch matches frequency
        if self._candidate_display_progress(epoch):
            self.logger.info(f"CandidateUnit: train: Epoch {epoch + 1} - Norm Output: {candidate_parameters_update.norm_output}, Norm Error: {candidate_parameters_update.norm_error}")

        self.logger.verbose(f"CandidateUnit: train: Epoch {epoch + 1} - Residual Error: Shape: {residual_error.shape}, Dtype: {residual_error.dtype}")

    #################################################################################################################################################################################################
    # Get the correlations for the candidate unit.
    def _get_correlations(self, output: torch.Tensor = None, residual_error: torch.Tensor = None) -> tuple[list, int, torch.Tensor, torch.Tensor]:
        """
        Description:
            Get the correlations for the candidate unit.
        Args:
            output: Output tensor from the candidate unit
            residual_error: Residual error tensor from the network
        Notes:
            This method calculates the correlations between the output of the candidate unit and the residual error.
            It handles both single-output and multi-output networks.
            For multi-output networks, it computes correlation with each output and uses the maximum absolute correlation.
        Returns:
            correlations: List of correlations for each output
            output_val: Normalized output value
            error_val: Normalized error value
        """
        # Compute correlation with each output and use the maximum absolute correlation
        # CR-062: Guard log calls that evaluate tensor/object repr to avoid expensive __repr__ in hot path
        _log_debug = self.logger.isEnabledFor(level=10)  # DEBUG
        _log_verbose = self.logger.isEnabledFor(level=8)  # VERBOSE
        _log_trace = self.logger.isEnabledFor(level=5)  # TRACE
        if _log_trace:
            self.logger.trace("CandidateUnit: _get_correlations: Getting Correlations for network, residual_error shape: %s", residual_error.shape)
        if _log_debug:
            self.logger.debug("CandidateUnit: _get_correlations: Input Params: Residual Error: Shape: %s, Type: %s, Dims: %s, Dtype: %s", residual_error.shape, type(residual_error), residual_error.dim(), residual_error.dtype)
            self.logger.debug("CandidateUnit: _get_correlations: Output shape: %s, Output length: %d", output.shape, len(output))

        # Calculate correlations for current candidate nodes
        candidate_correlations = self._multi_output_correlation(residual_error=residual_error, output=output)
        if _log_debug:
            self.logger.debug("CandidateUnit: _get_correlations: Multi-output correlation result: Cascade Correlations: Length %d", len(candidate_correlations))

        # Extract all correlations and find the best one
        correlations = [c.correlation for c in candidate_correlations]
        if _log_verbose:
            self.logger.verbose("CandidateUnit: _get_correlations: All correlations: %s", correlations)

        # Find the best correlation by maximum absolute value
        best_idx = int(np.argmax(np.abs(np.array(correlations)))) if correlations else -1
        self.logger.debug(f"CandidateUnit: _get_correlations: Best correlation index: {best_idx}")

        # Get the best correlation data
        if best_idx >= 0:
            best = candidate_correlations[best_idx]
            best_correlation = best.correlation
            best_corr_idx = best.best_corr_idx
            norm_output = best.best_norm_output
            norm_error = best.best_norm_error
            numerator = best.numerator
            denominator = best.denominator
        else:
            best_correlation = 0.0
            best_corr_idx = -1
            norm_output = None
            norm_error = None
            numerator = 0.0
            denominator = 1.0

        if _log_debug:
            self.logger.debug("CandidateUnit: _get_correlations: Best correlation: %s, Best corr idx: %s", best_correlation, best_corr_idx)

        # Create a CandidateTrainingResult data class to return all relevant training results
        candidate_training_result = CandidateTrainingResult(candidate_id=self.candidate_index, candidate_uuid=self.uuid, correlation=best_correlation, best_corr_idx=best_corr_idx, all_correlations=correlations, norm_output=norm_output, norm_error=norm_error, numerator=numerator, denominator=denominator, success=(best_idx >= 0))
        return candidate_training_result

    #################################################################################################################################################################################################
    def _multi_output_correlation(
        self,
        residual_error: torch.Tensor = None,
        output: torch.Tensor = None,
        # ) -> list(tuple[float, int, torch.Tensor, torch.Tensor, float, float]):
        # ) -> [CandidateCorrelationCalculation]:  # Original - invalid list syntax
    ) -> list[CandidateCorrelationCalculation]:
        """
        Description:
            Calculate the correlation for multi-output networks.
        Args:
            residual_error: Residual error tensor from the network
            output: Output tensor from the candidate unit
        Notes:
            This method calculates the correlation between the output of the candidate unit and the residual error for multi-output networks.
            It iterates through each output index and calculates the correlation.
            The method returns a list of correlations for each output.
        Returns:
            correlations: List of tuples containing correlation, output index, normalized output, normalized error, numerator, and denominator
        """
        # CR-062: Guard log calls to avoid expensive f-string evaluation in hot path
        _log_debug = self.logger.isEnabledFor(level=10)  # DEBUG
        _log_verbose = self.logger.isEnabledFor(level=8)  # VERBOSE

        if _log_debug:
            self.logger.debug("CandidateUnit: _multi_output_correlation: Residual error shape: %s, Output shape: %s", residual_error.shape, output.shape)

        # Initialize a list to store correlations for each output
        calculated_correlations = []

        # Get the max index for the 2nd dim (dim 1) of the residual error.  Dim 0 is number of batches, dim 1 is the number of error values--one for each output.
        max_index = residual_error.shape[1] if hasattr(residual_error, "shape") and len(residual_error.shape) > 1 else 1

        # Iterate through each output index and calculate correlation
        for i in range(max_index):
            # Handle both 1D and 2D residual_error tensors
            error_i = residual_error[:, i] if residual_error.dim() > 1 else residual_error

            # Calculate correlation for the current output index
            (correlation, norm_output, norm_error, numerator, denominator) = self._calculate_correlation(output=output, residual_error=error_i)
            if _log_verbose:
                self.logger.verbose("CandidateUnit: _multi_output_correlation: Correlation for output %d: %s", i, correlation)

            candidate_correlation_calculation = CandidateCorrelationCalculation(
                correlation=correlation,
                best_corr_idx=i,
                best_norm_output=norm_output,
                best_norm_error=norm_error,
                numerator=numerator,
                denominator=denominator,
                output=output,
                residual_error=error_i,
            )
            calculated_correlations.append(candidate_correlation_calculation)

        if _log_debug:
            self.logger.debug("CandidateUnit: _multi_output_correlation: Total correlations calculated: %d", len(calculated_correlations))
        self.logger.verbose(f"CandidateUnit: _multi_output_correlation: Correlations computed: {len(calculated_correlations)} outputs")
        self.logger.trace("CandidateUnit: _multi_output_correlation: Completed the _multi_output_correlation method")

        # Return the list of correlations
        # return correlations
        return calculated_correlations

    #################################################################################################################################################################################################
    def _get_correlation_abs_value(
        self,
        index: int = 0,
    ) -> float:
        """
        Description:
            Get the absolute value of the correlation at the specified index.
        Args:
            index: Index of the correlation to retrieve
        Notes:
            This method retrieves the correlation item at the specified index and calculates its absolute value.
            It handles both single-output and multi-output networks.
            The method also handles cases where the correlation is a tuple or list of tensors, and where the correlation is a float.
            The method also handles cases where the correlation is a torch.Tensor or a numpy array.
            If the correlation is not a recognized type, it defaults to using numpy's absolute function.
        Returns:
            Absolute value of the correlation at the specified index
        """
        self.logger.trace("CandidateUnit: _get_correlation_abs_value: Starting to get the absolute value of the correlation")
        self.logger.verbose(f"CandidateUnit: _get_correlation_abs_value: Index: {index}, Correlations: Type: {type(self.correlations)}, Length: {len(self.correlations)}")
        correlation = self.correlations[index]
        self.logger.debug(f"CandidateUnit: _get_correlation_abs_value: Correlation item at index {index}: Type: {type(correlation)}, Length: {len(correlation)}")

        # Calculate the absolute value of the correlation
        self.logger.trace("CandidateUnit: _get_correlation_abs_value: Calculating absolute value of correlation")
        if hasattr(correlation, "item") and callable(correlation.item) and isinstance(correlation.item(), torch.Tensor):
            correlation_abs = self._calculate_abs_value(correlation.item())
        elif isinstance(correlation, (tuple, list)) and len(correlation) > 0 and isinstance(correlation[0], (torch.Tensor, float, int)):
            correlation_abs = self._calculate_abs_value(correlation[0])
        elif (isinstance(correlation, np.ndarray) and len(correlation) > 0) or (isinstance(correlation, (torch.Tensor, float, int))):
            correlation_abs = self._calculate_abs_value(correlation)
        else:
            self.logger.warning(f"CandidateUnit: _get_correlation_abs_value: Unexpected correlation type: {type(correlation)}.  Trying to calculate absolute value using numpy.")
            correlation_abs = self._calculate_abs_value(correlation)

        # Log the calculated absolute value of the correlation
        self.logger.debug(f"CandidateUnit: _get_correlation_abs_value: Correlation: {correlation}, Correlation absolute value: {correlation_abs}")
        self.logger.trace(f"CandidateUnit: _get_correlation_abs_value: Returning absolute value of correlation: {correlation_abs}")
        self.logger.trace("CandidateUnit: _get_correlation_abs_value: Completed the _get_correlation_abs_value method")

        # Return the absolute value of the correlation
        return correlation_abs

    def _calculate_abs_value(self, value):
        """
        Description:
            Calculate the absolute value of the given value.
        Args:
            value: The value to calculate the absolute value for
        Notes:
            This method calculates the absolute value of the given value.
            It handles both torch.Tensor and numpy array types.
            If the value is not a recognized type, it defaults to using numpy's absolute function.
        Returns:
            Absolute value of the given value
        """
        if isinstance(value, torch.Tensor):
            return torch.abs(value)
        elif isinstance(value, (np.ndarray, float, int)):
            return np.abs(value)
        else:
            self.logger.warning(f"CandidateUnit: _calculate_abs_value: Unexpected value type: {type(value)}. Using numpy's absolute function as fallback.")
            return np.abs(value)

    #################################################################################################################################################################################################
    def _calculate_correlation(
        self,
        output: torch.Tensor = None,
        residual_error: torch.Tensor = None,
        # ) -> tuple([float, torch.Tensor, torch.Tensor, float, float]):  # Original - invalid syntax
    ) -> tuple[float, torch.Tensor, torch.Tensor, float, float]:
        """
        Description:
            Calculate the correlation between the candidate unit output and residual error for the entire minibatch.
            This is the core evaluation metric for candidate effectiveness in Cascade Correlation.
        Args:
            output: Output tensor from the candidate unit (shape: [batch_size, 1])
            residual_error: Residual error tensor from the network (shape: [batch_size, num_outputs])
        Notes:
            Calculates Pearson correlation coefficient between candidate output and residual error.
            For minibatch evaluation, this measures how well the candidate unit's activation
            correlates with the network's current error across all samples in the batch.
        Mathematical Formula:
            correlation = |Σ((o_i - ō)(e_i - ē))| / √(Σ(o_i - ō)² × Σ(e_i - ē)²)
            where o_i, e_i are output/error for sample i, and ō, ē are their means.
        Returns:
            correlation: Absolute correlation value (higher is better for candidate selection)
            norm_output: Mean-centered output tensor
            norm_error: Mean-centered residual error tensor
            numerator: Covariance component of correlation
            denominator: Normalization factor (product of standard deviations)
        """
        self.logger.trace("CandidateUnit: _calculate_correlation: Starting minibatch correlation calculation")
        self.logger.debug(f"CandidateUnit: _calculate_correlation: Output shape: {output.shape}, Residual error shape: {residual_error.shape}")

        # Validate the parameters for correlation calculation
        self._validate_correlation_params(output=output, residual_error=residual_error)

        # OPT-2: Fused correlation computation — uses torch.dot() and torch.linalg.norm()
        # instead of separate sum(a*b) and sqrt(sum(x^2)) to reduce kernel launches and
        # eliminate intermediate tensor allocations.

        # Flatten to 1D and mean-center in two operations
        output_flat = output.flatten()
        residual_error_flat = residual_error.flatten()
        norm_output = output_flat - output_flat.mean()
        norm_error = residual_error_flat - residual_error_flat.mean()

        # Fused covariance via torch.dot (single BLAS call, no intermediate tensor)
        numerator = torch.dot(norm_output, norm_error)

        # Fused denominator via torch.linalg.norm (single BLAS call each, avoids square + sum + sqrt)
        output_norm = torch.linalg.norm(norm_output)
        error_norm = torch.linalg.norm(norm_error)
        denominator = output_norm * error_norm + 1e-8

        # Compute Pearson correlation coefficient
        if denominator < 1e-7 or torch.isnan(denominator):
            self.logger.warning("CandidateUnit: _calculate_correlation: Denominator is zero or NaN, setting correlation to zero")
            correlation = 0.0
            numerator_val = 0.0
            denominator_val = 1e-8
        else:
            numerator_val = numerator.item()
            denominator_val = denominator.item()
            correlation = abs(numerator_val / denominator_val)
            # Clamp to [0, 1] — floating-point arithmetic with the epsilon denominator
            # can produce values marginally above 1.0 (e.g., 1.0000000149).  The Pearson
            # correlation coefficient is theoretically bounded to [0, 1] for abs values,
            # so clamping is mathematically correct.
            if correlation > 1.0:
                self.logger.debug(f"CandidateUnit: _calculate_correlation: Clamping correlation {correlation:.10f} to 1.0 (floating-point overshoot)")
                correlation = 1.0

        self.logger.info(f"CandidateUnit: _calculate_correlation: Final correlation effectiveness: {correlation:.6f}")

        # Return correlation and components for gradient computation
        return (correlation, norm_output, norm_error, numerator_val, denominator_val)

    #################################################################################################################################################################################################
    # Update weights and bias of the candidate unit based on correlation with residual error.
    # This method computes the gradient of the correlation with respect to the output and updates the weights and bias accordingly.
    # It handles both single-output and multi-output networks.
    # The method takes the input tensor, residual error, learning rate, and optionally normalized output and error tensors, as well as numerator and denominator for correlation.
    # It returns the gradients of correlation and output.
    def _update_weights_and_bias(
        self,
        candidate_parameters_update: CandidateParametersUpdate = None,
    ) -> EpochTrainedCandidate:
        """
        Description:
            Update weights and bias of the candidate unit based on absolute correlation with residual error using autograd.
        Args:
            x: Input tensor
            y: Current output tensor (not used in autograd version)
            residual_error: Residual error from the network
            learning_rate: Learning rate for weight updates
            norm_output: Normalized output tensor (full vector)
            norm_error: Normalized error tensor (full vector)
            numerator: Numerator of the correlation (not used in autograd)
            denominator: Denominator of the correlation (not used in autograd)
        Notes:
            Uses PyTorch autograd to compute gradients of absolute correlation objective.
            This correctly handles the derivative of mean-centering and absolute value.
        Returns:
            grad_corr: Gradient of correlation with respect to output
            grad_output: Gradient of output with respect to weights
        """
        # CR-062: Guard log calls to avoid expensive f-string evaluation in hot path
        _log_debug = self.logger.isEnabledFor(level=10)  # DEBUG

        if _log_debug:
            self.logger.debug("CandidateUnit: _update_weights_and_bias: Input shape: %s, Residual error shape: %s, Learning rate: %s", candidate_parameters_update.x.shape, candidate_parameters_update.residual_error.shape, candidate_parameters_update.learning_rate)

        # Convert weights and bias to require gradients temporarily
        weights_param = self.weights.clone().detach().requires_grad_(True)
        bias_param = self.bias.clone().detach().requires_grad_(True)

        # Forward pass with gradient tracking
        logits = torch.sum(candidate_parameters_update.x * weights_param, dim=1) + bias_param
        output = self.activation_fn(logits)

        # Extract the correct error slice based on best correlation index
        self.logger.debug("CandidateUnit: _update_weights_and_bias: Extracting error slice for best correlation index")

        # Determine which error slice to use based on residual_error shape
        if candidate_parameters_update.residual_error.dim() > 1 and candidate_parameters_update.residual_error.shape[1] > 1:
            # Multi-output: slice to the best output index
            if candidate_parameters_update.best_corr_idx >= 0:
                self.logger.debug(f"CandidateUnit: _update_weights_and_bias: Multi-output detected, using error slice at index {candidate_parameters_update.best_corr_idx}")
                error_slice = candidate_parameters_update.residual_error[:, candidate_parameters_update.best_corr_idx]
            else:
                self.logger.warning("CandidateUnit: _update_weights_and_bias: Invalid best_corr_idx, using first error column")
                error_slice = candidate_parameters_update.residual_error[:, 0]
        else:
            # Single output or 1-D error: use as-is
            self.logger.debug("CandidateUnit: _update_weights_and_bias: Single-output detected, using error as-is")
            error_slice = candidate_parameters_update.residual_error.flatten()

        # Compute mean-centered tensors with gradient tracking
        output_mean = output.mean()
        error_mean = error_slice.mean()
        output_centered = output - output_mean
        error_centered = error_slice - error_mean

        # OPT-2: Fused correlation with autograd-compatible ops (torch.dot, torch.linalg.norm)
        candidate_parameters_update.numerator = torch.dot(output_centered, error_centered)
        output_std = torch.linalg.norm(output_centered)
        error_std = torch.linalg.norm(error_centered)
        candidate_parameters_update.denominator = output_std * error_std + 1e-8
        correlation = candidate_parameters_update.numerator / candidate_parameters_update.denominator

        # Objective: maximize absolute correlation (minimize negative absolute correlation)
        loss = -torch.abs(correlation)
        if _log_debug:
            self.logger.debug("CandidateUnit: _update_weights_and_bias: Correlation: %.6f, Loss: %.6f", correlation.item(), loss.item())

        # Compute gradients
        loss.backward()

        # Extract gradients
        grad_w = weights_param.grad
        grad_b = bias_param.grad

        if grad_w is None or grad_b is None:
            self.logger.warning("CandidateUnit: _update_weights_and_bias: Gradients are None, skipping update")
            return (torch.zeros_like(output), torch.zeros_like(output))

        # Update weights and bias (use -= for gradient descent to minimize loss)
        with torch.no_grad():
            self.weights -= candidate_parameters_update.learning_rate * grad_w
            self.bias -= candidate_parameters_update.learning_rate * grad_b

        if _log_debug:
            self.logger.debug("CandidateUnit: _update_weights_and_bias: Weights shape: %s, norm: %.6f, Bias gradient: %.6f", self.weights.shape, grad_w.norm().item(), grad_b.item())

        # Return dummy gradients for compatibility (not used elsewhere)
        return (error_centered, output_centered)

    #################################################################################################################################################################################################
    def _validate_correlation_params(
        self,
        output: torch.Tensor = None,
        residual_error: torch.Tensor = None,
    ) -> None:
        """
        Description:
            Validate the parameters for the correlation calculation.
        Args:
            output: Output tensor from the candidate unit
            residual_error: Residual error tensor from the network
        Raises:
            ValueError: If output or residual error is None, or if they have incompatible shapes
            TypeError: If output or residual error is not a torch.Tensor type
        Notes:
            This method checks if the output and residual error tensors are valid for correlation calculation.
            It ensures that both tensors are not None, are of type torch.Tensor, have the same batch size, and have compatible dimensions.
        Returns:
            None
        """
        self.logger.trace("CandidateUnit: _validate_correlation_params: Starting validation of correlation parameters")

        # Check if output and residual error are not None
        # NOTE: shapes are logged after the type check below (line ~1107); logging
        # output.shape here would AttributeError on non-tensor input before the
        # TypeError can be raised. Entry-time type detail is logged after the None check.
        self.logger.trace("CandidateUnit: _validate_correlation_params: Validating output and residual error are not None")
        if output is None or residual_error is None:
            raise ValueError("CandidateUnit: _validate_correlation_params: Output and residual error must not be None.")

        # Check if output and residual error are torch.Tensor types
        self.logger.debug(f"CandidateUnit: _validate_correlation_params: Output type: {type(output)}, Residual error type: {type(residual_error)}")
        self.logger.trace("CandidateUnit: _validate_correlation_params: Validating output and residual error types")
        if not isinstance(output, torch.Tensor) or not isinstance(residual_error, torch.Tensor):
            raise TypeError("CandidateUnit: _validate_correlation_params: Output and residual error must be torch.Tensor types.")

        # Check if output and residual error have compatible shapes
        self.logger.debug(f"CandidateUnit: _validate_correlation_params: Output shape: {output.shape}, Residual error shape: {residual_error.shape}")
        self.logger.trace("CandidateUnit: _validate_correlation_params: Validating output and residual error shapes")
        if output.shape[0] != residual_error.shape[0]:
            raise ValueError("CandidateUnit: _validate_correlation_params: Output and residual error must have the same batch size.")

        # Check if output and residual error have compatible dimensions
        self.logger.debug(f"CandidateUnit: _validate_correlation_params: Output dimensions: {len(output.shape)}, Residual error dimensions: {len(residual_error.shape)}")
        self.logger.trace("CandidateUnit: _validate_correlation_params: Validating output and residual error dimensions")
        if len(output.shape) < 1 or len(residual_error.shape) < 1:
            raise ValueError("CandidateUnit: _validate_correlation_params: Output and residual error must have at least one dimension.")

        # Ensure that output and residual error have compatible dimensions
        self.logger.debug(f"CandidateUnit: _validate_correlation_params: Output shape: {output.shape}, Residual error shape: {residual_error.shape}")
        self.logger.trace("CandidateUnit: _validate_correlation_params: Validating output and residual error dimensions for multi-output networks")
        if len(output.shape) > 2 or len(residual_error.shape) > 2:
            raise ValueError("CandidateUnit: _validate_correlation_params: Output and residual error must have at most two dimensions.")

        # Ensure that output and residual error have the same number of features if residual_error has more than one dimension
        self.logger.debug(f"CandidateUnit: _validate_correlation_params: Output shape: {output.shape}, Residual error shape: {residual_error.shape}")
        self.logger.trace("CandidateUnit: _validate_correlation_params: Validating output and residual error features for multi-output networks")
        self.logger.verbose(f"CandidateUnit: _validate_correlation_params: Residual Error: Shape: {residual_error.shape}, Shape Length: {len(residual_error.shape)}, Type: {type(residual_error)}, Dimensions: {residual_error.dim()}, Dtype: {residual_error.dtype}")
        dimensions = residual_error.dim() if hasattr(residual_error, "dim") else len(residual_error.shape)
        index = dimensions - 1 if dimensions > 1 else 0
        self.logger.debug(f"CandidateUnit: _validate_correlation_params: Checking if output and residual error have the same number of features at Index {index}, Dimensions: {dimensions}")
        if output.shape[index] != residual_error.shape[index] and dimensions > 1:
            raise ValueError("CandidateUnit: _validate_correlation_params: Output and residual error must have the same number of features if residual_error has more than one dimension.")

        # If all validations pass, log success
        self.logger.debug("CandidateUnit: _validate_correlation_params: Completed validation of correlation parameters.")

        # Log completion of validation
        self.logger.trace("CandidateUnit: _validate_correlation_params: Validation complete.")

    ####################################################################################################################################
    # Define private methods for the CandidateUnit class
    def _generate_uuid(self):  # sourcery skip: class-extract-method
        """
        Description:
            Generate a new UUID for the CandidateUnit class.
        Args:
            self: The instance of the class.
        Returns:
            str: The generated UUID.
        """
        self.logger.trace("CandidateUnit: _generate_uuid: Inside the CandidateUnit class Generate UUID method")
        new_uuid = str(uuid.uuid4())
        self.logger.debug(f"CandidateUnit: _generate_uuid: UUID: {new_uuid}")
        self.logger.trace("CandidateUnit: _generate_uuid: Completed the CandidateUnit class Generate UUID method")
        return new_uuid

    # def _init_display_progress(self, display_frequency=None) -> callable:  # Original - invalid type
    def _init_display_progress(self, display_frequency=None) -> Callable[..., Any]:
        """
        Description:
            Initialize the display progress for the CandidateUnit class.
        Args:
            self: The instance of the class.
            display_frequency: The frequency at which to display progress.
        Returns:
            callable: The display progress function.
        """
        self.logger.trace("CandidateUnit: _init_display_progress: Inside the CandidateUnit class Initialize Display Progress method")
        candidate_display_progress = display_progress(display_frequency=display_frequency)
        self.logger.debug(f"CandidateUnit: _init_display_progress: Display frequency set to: {display_frequency}, Candidate Display Progress: {candidate_display_progress}")
        self.logger.trace("CandidateUnit: _init_display_progress: Completed the CandidateUnit class Initialize Display Progress method")
        return candidate_display_progress

    # def _init_display_status(self, display_status=None) -> callable:  # Original - invalid type
    def _init_display_status(self, display_status=None) -> Callable[..., Any]:
        """
        Description:
            Initialize the display status for the CandidateUnit class.
        Args:
            self: The instance of the class.
            display_status: The display status function.
        Returns:
            callable: The display status function.
        """
        self.logger.trace("CandidateUnit: _init_display_status: Inside the CandidateUnit class Initialize Display Status method")
        candidate_display_status = display_progress(display_frequency=display_status)
        self.logger.debug(f"CandidateUnit: _init_display_status: Candidate Display Status: {candidate_display_status}")
        self.logger.trace("CandidateUnit: _init_display_status: Completed the CandidateUnit class Initialize Display Status method")
        return candidate_display_status

    def clear_display_status(self) -> None:
        """
        Description:
            Clear the display status for the CandidateUnit class.
        Args:
            self: The instance of the class.
        Returns:
            None
        """
        self.logger.trace("CandidateUnit: clear_display_status: Inside the CandidateUnit class Clear Display Status method")
        if hasattr(self, "_candidate_display_status") and self._candidate_display_status is not None:
            self._candidate_display_status = None
            self.logger.debug("CandidateUnit: clear_display_status: Cleared the display status")
        else:
            self.logger.warning("CandidateUnit: clear_display_status: _candidate_display_status attribute not found or is already None")
        self.logger.trace("CandidateUnit: clear_display_status: Completed the CandidateUnit class Clear Display Status method")

    def clear_display_progress(self) -> None:
        """
        Description:
            Clear the display progress for the CandidateUnit class.
        Args:
            self: The instance of the class.
        Returns:
            None
        """
        self.logger.trace("CandidateUnit: clear_display_progress: Inside the CandidateUnit class Clear Display Progress method")
        if hasattr(self, "_candidate_display_progress") and self._candidate_display_progress is not None:
            self._candidate_display_progress = None
            self.logger.debug("CandidateUnit: clear_display_progress: Cleared the display progress")
        else:
            self.logger.warning("CandidateUnit: clear_display_progress: _candidate_display_progress attribute not found or is already None")
        self.logger.trace("CandidateUnit: clear_display_progress: Completed the CandidateUnit class Clear Display Progress method")

    #################################################################################################################################################################################################
    # Define Setters for candidate unit attributes.
    def set_correlation(self, correlation: float) -> None:
        """
        Set the correlation of the candidate unit.
        Args:
            correlation: Correlation value
        """
        self.correlation = correlation

    def set_activation_fn(self, activation_fn: torch.nn.Module) -> None:
        """
        Set the activation function of the candidate unit.
        Args:
            activation_fn: Activation function
        """
        self.activation_fn = activation_fn

    def set_activation_fn_base(self, activation_fn_base: torch.nn.Module) -> None:
        """
        Set the base activation function of the candidate unit.
        Args:
            activation_fn_base: Base activation function
        """
        self.activation_fn = self._init_activation_with_derivative(activation_fn_base)
        self.activation_fn_base = self.activation_fn.activation_fn

    def set_bias(self, bias: torch.Tensor) -> None:
        """
        Set the bias of the candidate unit.
        Args:
            bias: Bias tensor
        """
        self.bias = bias

    # def set_epochs(self, epochs: int) -> None:
    #     """
    #     Set the number of epochs for the candidate unit.
    #     Args:
    #         epochs: Number of epochs
    #     """
    #     self.epochs = epochs

    def set_display_frequency(self, display_frequency: int) -> None:
        """
        Set the display frequency of the candidate unit.
        Args:
            display_frequency: Display frequency
        """
        self.display_frequency = display_frequency

    def set_epochs_max(self, epochs_max: int) -> None:
        """
        Set the maximum number of epochs for the candidate unit.
        Args:
            epochs_max: Maximum number of epochs
        """
        self.epochs_max = epochs_max

    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Set the learning rate of the candidate unit.
        Args:
            learning_rate: Learning rate
        """
        self.learning_rate = learning_rate

    def set_logging_file_name(self, logging_file_name: str) -> None:
        """
        Set the logging file name of the candidate unit.
        Args:
            logging_file_name: Logging file name
        """
        self.logging_file_name = logging_file_name

    def set_logging_level(self, logging_level: int) -> None:
        """
        Set the logging level of the candidate unit.
        Args:
            logging_level: Logging level
        """
        self.logging_level = logging_level

    def set_random_value_scale(self, random_value_scale: float) -> None:
        """
        Set the random value scale of the candidate unit.
        Args:
            random_value_scale: Random value scale
        """
        self.random_value_scale = random_value_scale

    def set_weights(self, weights: torch.Tensor) -> None:
        """
        Set the weights of the candidate unit.
        Args:
            weights: Weights tensor
        """
        self.weights = weights

    def set_uuid(self, uuid: str = None):
        """
        Description:
            This method sets the UUID for the CandidateUnit class.  If no UUID is provided, a new UUID will be generated.
        Args:
            uuid (str): The UUID to be set. If None, a new UUID will be generated.
        Returns:
            None
        """
        self.logger.trace("CandidateUnit: set_uuid: Starting to set UUID for CandidateUnit class")
        if not hasattr(self, "uuid") or self.uuid is None:
            self.uuid = (uuid, self._generate_uuid())[uuid is None]  # Generate a new UUID if none is provided
        else:
            self.logger.fatal(f"CandidateUnit: set_uuid: Fatal Error: UUID already set: {self.uuid}. Changing UUID is bad Juju.  Exiting...")
            sys.exit(1)
        self.logger.debug(f"CandidateUnit: set_uuid: UUID set to: {self.uuid}")
        self.logger.trace("CandidateUnit: set_uuid: Completed setting UUID for CandidateUnit class")

    ####################################################################################################################################
    # Define CandidateUnit class Getters for class attributes
    def get_uuid(self) -> str:
        """
        Description:
            This method returns the UUID for the CandidateUnit class.
        Args:
            self: The instance of the class.
        Returns:
            str: The UUID for the CandidateUnit class.
        """
        self.logger.trace("CandidateUnit: get_uuid: Starting to get UUID for CandidateUnit class")
        if not hasattr(self, "uuid") or self.uuid is None:
            self.set_uuid()  # Ensure UUID is set if not already
            self.logger.trace("CandidateUnit: get_uuid: UUID was not set, generated a new one.")
        self.logger.debug(f"CandidateUnit: get_uuid: Returning UUID: {self.uuid}")
        self.logger.trace("CandidateUnit: get_uuid: Completed getting UUID for CandidateUnit class")
        return self.uuid

    def get_correlation(self) -> float:
        """
        Get the correlation of the candidate unit.
        Returns:
            Correlation value
        """
        return self.correlation

    def get_activation_fn(self) -> torch.nn.Module:
        """
        Get the activation function of the candidate unit.
        Returns:
            Activation function
        """
        return self.activation_fn

    def get_activation_fn_base(self) -> torch.nn.Module:
        """
        Get the base activation function of the candidate unit.
        Returns:
            Base activation function
        """
        return self.activation_fn_base

    def get_bias(self) -> torch.Tensor:
        """
        Get the bias of the candidate unit.
        Returns:
            Bias tensor
        """
        return self.bias

    def get_display_frequency(self) -> int:
        """
        Get the display frequency of the candidate unit.
        Returns:
            Display frequency
        """
        return self.display_frequency

    # def get_epochs(self) -> int:
    #     """
    #     Get the number of epochs for the candidate unit.
    #     Returns:
    #         Number of epochs
    #     """
    #     return self.epochs

    def get_epochs_max(self) -> int:
        """
        Get the maximum number of epochs for the candidate unit.
        Returns:
            Maximum number of epochs
        """
        return self.epochs_max

    def get_learning_rate(self) -> float:
        """
        Get the learning rate of the candidate unit.
        Returns:
            Learning rate
        """
        return self.learning_rate

    def get_logging_file_name(self) -> str:
        """
        Get the logging file name of the candidate unit.
        Returns:
            Logging file name
        """
        return self.logging_file_name

    def get_logging_level(self) -> int:
        """
        Get the logging level of the candidate unit.
        Returns:
            Logging level
        """
        return self.logging_level

    def get_random_value_scale(self) -> float:
        """
        Get the random value scale of the candidate unit.
        Returns:
            Random value scale
        """
        return self.random_value_scale

    def get_weights(self) -> torch.Tensor:
        """
        Get the weights of the candidate unit.
        Returns:
            Weights tensor
        """
        return self.weights
