#!/usr/bin/env python
"""
Project:       Juniper
Prototype:     Cascade Correlation Neural Network
File Name:     mock_candidate.py
Author:        Paul Calnon
Version:       0.1.0

Date:          2025-09-26
Last Modified: 2025-09-26

License:       MIT License
Copyright:     Copyright (c) 2024-2025 Paul Calnon

Description:
    Mock candidate units for testing cascade correlation algorithms.
    Provides controllable mock objects for testing without full training.
"""

import uuid

# from typing import Optional, Tuple, Callable
from typing import Optional, Tuple

import torch

# from unittest.mock import Mock, MagicMock


class MockCandidateUnit:
    """
    Mock candidate unit that simulates training behavior.
    Allows controlled testing of candidate selection and integration.
    """

    def __init__(
        self,
        input_size: int = 2,
        correlation: float = 0.5,
        weights: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        uuid_str: Optional[str] = None,
        training_behavior: str = "success",
    ):
        """
        Initialize mock candidate unit.

        Args:
            input_size: Input dimensionality
            correlation: Correlation to return from training
            weights: Predefined weights (random if None)
            bias: Predefined bias (random if None)
            uuid_str: UUID string (generated if None)
            training_behavior: "success", "failure", or "slow"
        """
        self.input_size = input_size
        self.correlation = correlation
        self.training_behavior = training_behavior
        self.uuid = uuid_str or str(uuid.uuid4())

        # Initialize weights and bias
        if weights is None:
            self.weights = torch.randn(input_size, requires_grad=True) * 0.1
        else:
            self.weights = weights.clone().detach().requires_grad_(True)

        if bias is None:
            self.bias = torch.randn(1, requires_grad=True) * 0.1
        else:
            self.bias = bias.clone().detach().requires_grad_(True)

        # Training state
        self.is_trained = False
        self.training_epochs = 0
        self.training_calls = 0

    def train(
        self,
        x: torch.Tensor,
        epochs: int = 10,
        residual_error: torch.Tensor = None,
        learning_rate: float = 0.01,
        display_frequency: int = 10,
    ) -> Tuple[float, int]:
        """
        Mock training method that simulates candidate training.

        Args:
            x: Input data
            epochs: Number of training epochs
            residual_error: Residual error from network
            learning_rate: Learning rate
            display_frequency: Display frequency

        Returns:
            (correlation, epochs_trained): Training results
        """
        self.training_calls += 1

        if self.training_behavior == "failure":
            # Simulate training failure
            self.correlation = 0.0
            self.training_epochs = 0
            return 0.0, 0

        elif self.training_behavior == "slow":
            # Simulate slow convergence
            self.training_epochs = epochs
            # Gradually improve correlation
            improvement = min(epochs / 100.0, 0.3)
            self.correlation = max(0.1, self.correlation + improvement)

        else:  # "success"
            # Simulate successful training
            self.training_epochs = min(epochs, 20)  # Early convergence
            # Simulate correlation improvement
            if residual_error is not None:
                # Use residual error magnitude to determine correlation
                error_magnitude = residual_error.abs().mean().item()
                self.correlation = min(0.9, max(0.1, 1.0 - error_magnitude))

        self.is_trained = True
        return self.correlation, self.training_epochs

    def get_correlation(self) -> float:
        """Get correlation coefficient."""
        return self.correlation

    def get_uuid(self) -> str:
        """Get UUID string."""
        return self.uuid

    def clear_display_progress(self):
        """Mock method to clear display progress."""
        pass

    def clear_display_status(self):
        """Mock method to clear display status."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through candidate unit.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return torch.tanh(torch.mm(x, self.weights.unsqueeze(1)) + self.bias).squeeze()


class MockCandidatePool:
    """
    Mock candidate pool that generates predictable candidates.
    """

    def __init__(
        self,
        pool_size: int = 8,
        input_size: int = 2,
        correlation_range: Tuple[float, float] = (0.1, 0.8),
        success_rate: float = 1.0,
    ):
        """
        Initialize mock candidate pool.

        Args:
            pool_size: Number of candidates in pool
            input_size: Input dimensionality
            correlation_range: Range of correlations to generate
            success_rate: Fraction of candidates that train successfully
        """
        self.pool_size = pool_size
        self.input_size = input_size
        self.correlation_range = correlation_range
        self.success_rate = success_rate

        self.candidates = []
        self._generate_candidates()

    def _generate_candidates(self):
        """Generate mock candidates with varying correlations."""
        n_successful = int(self.pool_size * self.success_rate)
        n_failed = self.pool_size - n_successful

        # Generate successful candidates with varying correlations
        correlations = torch.linspace(
            self.correlation_range[0], self.correlation_range[1], n_successful
        )

        # for i, corr in enumerate(correlations):
        for corr in correlations:
            candidate = MockCandidateUnit(
                input_size=self.input_size,
                correlation=corr.item(),
                training_behavior="success",
            )
            self.candidates.append(candidate)

        # Generate failed candidates
        # for i in range(n_failed):
        for _ in range(n_failed):
            candidate = MockCandidateUnit(
                input_size=self.input_size, correlation=0.0, training_behavior="failure"
            )
            self.candidates.append(candidate)

    def get_best_candidate(self) -> MockCandidateUnit:
        """Get candidate with highest correlation."""
        if trained_candidates := [c for c in self.candidates if c.is_trained]:
            return max(trained_candidates, key=lambda c: c.correlation)
        else:
            return None

    def get_candidates(self) -> list:
        """Get all candidates."""
        return self.candidates.copy()


def create_mock_candidate_with_correlation(
    correlation: float, input_size: int = 2, training_behavior: str = "success"
) -> MockCandidateUnit:
    """
    Create a mock candidate with specific correlation.

    Args:
        correlation: Desired correlation coefficient
        input_size: Input dimensionality
        training_behavior: Training behavior pattern

    Returns:
        Mock candidate unit
    """
    return MockCandidateUnit(
        input_size=input_size,
        correlation=correlation,
        training_behavior=training_behavior,
    )


def create_mock_failing_candidate(input_size: int = 2) -> MockCandidateUnit:
    """
    Create a mock candidate that always fails training.

    Args:
        input_size: Input dimensionality

    Returns:
        Mock candidate that fails training
    """
    return MockCandidateUnit(
        input_size=input_size, correlation=0.0, training_behavior="failure"
    )


def create_mock_perfect_candidate(input_size: int = 2) -> MockCandidateUnit:
    """
    Create a mock candidate with perfect correlation.

    Args:
        input_size: Input dimensionality

    Returns:
        Mock candidate with correlation = 1.0
    """
    return MockCandidateUnit(
        input_size=input_size, correlation=1.0, training_behavior="success"
    )


class MockCandidateTrainer:
    """
    Mock trainer for candidate units that simulates multiprocessing behavior.
    """

    def __init__(self, success_rate: float = 1.0):
        self.success_rate = success_rate
        self.training_calls = 0

    def train_candidates(
        self, candidate_pool: list, x: torch.Tensor, residual_error: torch.Tensor
    ) -> list:
        """
        Mock training of candidate pool.

        Args:
            candidate_pool: List of candidates to train
            x: Input data
            residual_error: Residual error

        Returns:
            List of training results
        """
        self.training_calls += 1
        results = []

        for i, candidate in enumerate(candidate_pool):
            if torch.rand(1).item() < self.success_rate:
                # Simulate successful training
                # correlation, epochs = candidate.train(x, residual_error=residual_error)
                correlation, _ = candidate.train(x, residual_error=residual_error)
                results.append((i, candidate.uuid, correlation, candidate))
            else:
                # Simulate failed training
                results.append((i, candidate.uuid, 0.0, None))

        # Sort by correlation (descending)
        results.sort(key=lambda r: r[2], reverse=True)
        return results


# Additional mock utilities for testing specific scenarios


def create_mock_training_scenario(
    scenario_name: str, pool_size: int = 8, input_size: int = 2
) -> dict:
    """
    Create predefined training scenarios for testing.

    Args:
        scenario_name: Name of scenario to create
        pool_size: Size of candidate pool
        input_size: Input dimensionality

    Returns:
        Dictionary with scenario configuration
    """
    scenarios = {
        "all_succeed": {
            "pool": MockCandidatePool(pool_size, input_size, success_rate=1.0),
            "expected_best_correlation": 0.8,
            "expected_failures": 0,
        },
        "all_fail": {
            "pool": MockCandidatePool(pool_size, input_size, success_rate=0.0),
            "expected_best_correlation": 0.0,
            "expected_failures": pool_size,
        },
        "mixed_success": {
            "pool": MockCandidatePool(pool_size, input_size, success_rate=0.5),
            "expected_best_correlation": 0.8,
            "expected_failures": pool_size // 2,
        },
        "single_success": {
            "pool": MockCandidatePool(
                pool_size, input_size, success_rate=1 / pool_size
            ),
            "expected_best_correlation": 0.8,
            "expected_failures": pool_size - 1,
        },
    }

    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    return scenarios[scenario_name]
