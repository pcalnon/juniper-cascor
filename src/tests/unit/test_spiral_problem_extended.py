#!/usr/bin/env python
"""
Extended unit tests for spiral_problem/spiral_problem.py module.
Tests getter/setter methods and solve/evaluate methods with mocked network.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def mock_network():
    """Create a mock CascadeCorrelationNetwork."""
    mock = MagicMock()
    mock.fit.return_value = {"loss": [0.5, 0.3, 0.1]}
    mock.summary.return_value = None
    mock.plot_dataset = MagicMock()
    mock.plot_decision_boundary.return_value = None
    mock.plot_training_history.return_value = None
    mock.get_accuracy.return_value = 0.85
    return mock


@pytest.fixture
def spiral_problem_instance(mock_network):
    """Create a SpiralProblem instance with mocked network."""
    with patch("spiral_problem.spiral_problem.CascadeCorrelationNetwork", return_value=mock_network):
        from spiral_problem.spiral_problem import SpiralProblem

        sp = SpiralProblem(
            _SpiralProblem__n_points=20,
            _SpiralProblem__n_spirals=2,
            _SpiralProblem__random_seed=42,
        )
        sp.network = mock_network
        return sp


@pytest.mark.unit
class TestSpiralProblemSetters:
    """Tests for SpiralProblem setter methods."""

    def test_set_network(self, spiral_problem_instance):
        """Test set_network method."""
        new_network = MagicMock()
        spiral_problem_instance.set_network(new_network)
        assert spiral_problem_instance.network == new_network

    def test_set_network_none(self, spiral_problem_instance):
        """Test set_network with None doesn't change network."""
        original = spiral_problem_instance.network
        spiral_problem_instance.set_network(None)
        assert spiral_problem_instance.network == original

    def test_set_logger(self, spiral_problem_instance):
        """Test set_logger method."""
        new_logger = logging.getLogger("test_logger")
        spiral_problem_instance.set_logger(new_logger)
        assert spiral_problem_instance.logger == new_logger

    def test_set_logger_none(self, spiral_problem_instance):
        """Test set_logger with None doesn't change logger."""
        original = spiral_problem_instance.logger
        spiral_problem_instance.set_logger(None)
        assert spiral_problem_instance.logger == original

    def test_set_n_spirals(self, spiral_problem_instance):
        """Test set_n_spirals method."""
        spiral_problem_instance.set_n_spirals(5)
        assert spiral_problem_instance.n_spirals == 5

    def test_set_n_spirals_none(self, spiral_problem_instance):
        """Test set_n_spirals with None doesn't change value."""
        original = spiral_problem_instance.n_spirals
        spiral_problem_instance.set_n_spirals(None)
        assert spiral_problem_instance.n_spirals == original

    def test_set_n_points(self, spiral_problem_instance):
        """Test set_n_points method."""
        spiral_problem_instance.set_n_points(100)
        assert spiral_problem_instance.n_points == 100

    def test_set_n_points_none(self, spiral_problem_instance):
        """Test set_n_points with None doesn't change value."""
        original = spiral_problem_instance.n_points
        spiral_problem_instance.set_n_points(None)
        assert spiral_problem_instance.n_points == original

    def test_set_n_rotations(self, spiral_problem_instance):
        """Test set_n_rotations method."""
        spiral_problem_instance.set_n_rotations(3)
        assert spiral_problem_instance.n_rotations == 3

    def test_set_n_rotations_none(self, spiral_problem_instance):
        """Test set_n_rotations with None doesn't change value."""
        original = spiral_problem_instance.n_rotations
        spiral_problem_instance.set_n_rotations(None)
        assert spiral_problem_instance.n_rotations == original

    def test_set_clockwise(self, spiral_problem_instance):
        """Test set_clockwise method."""
        spiral_problem_instance.set_clockwise(True)
        assert spiral_problem_instance.clockwise is True

    def test_set_clockwise_false(self, spiral_problem_instance):
        """Test set_clockwise with False."""
        spiral_problem_instance.set_clockwise(False)
        assert spiral_problem_instance.clockwise is False

    def test_set_clockwise_none(self, spiral_problem_instance):
        """Test set_clockwise with None doesn't change value."""
        original = spiral_problem_instance.clockwise
        spiral_problem_instance.set_clockwise(None)
        assert spiral_problem_instance.clockwise == original

    def test_set_noise(self, spiral_problem_instance):
        """Test set_noise method."""
        spiral_problem_instance.set_noise(0.15)
        assert spiral_problem_instance.noise == 0.15

    def test_set_noise_none(self, spiral_problem_instance):
        """Test set_noise with None doesn't change value."""
        original = spiral_problem_instance.noise
        spiral_problem_instance.set_noise(None)
        assert spiral_problem_instance.noise == original

    def test_set_distribution(self, spiral_problem_instance):
        """Test set_distribution method."""
        spiral_problem_instance.set_distribution(0.7)
        assert spiral_problem_instance.distribution == 0.7

    def test_set_distribution_none(self, spiral_problem_instance):
        """Test set_distribution with None doesn't change value."""
        original = spiral_problem_instance.distribution
        spiral_problem_instance.set_distribution(None)
        assert spiral_problem_instance.distribution == original

    def test_set_random_seed(self, spiral_problem_instance):
        """Test set_random_seed method."""
        spiral_problem_instance.set_random_seed(123)
        assert spiral_problem_instance.random_seed == 123

    def test_set_random_seed_none(self, spiral_problem_instance):
        """Test set_random_seed with None doesn't change value."""
        original = spiral_problem_instance.random_seed
        spiral_problem_instance.set_random_seed(None)
        assert spiral_problem_instance.random_seed == original

    def test_set_train_ratio(self, spiral_problem_instance):
        """Test set_train_ratio method."""
        spiral_problem_instance.set_train_ratio(0.7)
        assert spiral_problem_instance.train_ratio == 0.7

    def test_set_train_ratio_none(self, spiral_problem_instance):
        """Test set_train_ratio with None doesn't change value."""
        original = spiral_problem_instance.train_ratio
        spiral_problem_instance.set_train_ratio(None)
        assert spiral_problem_instance.train_ratio == original

    def test_set_test_ratio(self, spiral_problem_instance):
        """Test set_test_ratio method."""
        spiral_problem_instance.set_test_ratio(0.3)
        assert spiral_problem_instance.test_ratio == 0.3

    def test_set_test_ratio_none(self, spiral_problem_instance):
        """Test set_test_ratio with None doesn't change value."""
        original = spiral_problem_instance.test_ratio
        spiral_problem_instance.set_test_ratio(None)
        assert spiral_problem_instance.test_ratio == original

    def test_set_plot(self, spiral_problem_instance):
        """Test set_plot method."""
        spiral_problem_instance.set_plot(True)
        assert spiral_problem_instance.plot is True

    def test_set_plot_false(self, spiral_problem_instance):
        """Test set_plot with False."""
        spiral_problem_instance.set_plot(False)
        assert spiral_problem_instance.plot is False

    def test_set_plot_none(self, spiral_problem_instance):
        """Test set_plot with None doesn't change value."""
        spiral_problem_instance.plot = True  # Set initial value
        spiral_problem_instance.set_plot(None)
        assert spiral_problem_instance.plot is True

    def test_set_random_value_scale(self, spiral_problem_instance):
        """Test set_random_value_scale method."""
        spiral_problem_instance.set_random_value_scale(0.01)
        assert spiral_problem_instance.random_value_scale == 0.01

    def test_set_random_value_scale_none(self, spiral_problem_instance):
        """Test set_random_value_scale with None doesn't change value."""
        original = spiral_problem_instance.random_value_scale
        spiral_problem_instance.set_random_value_scale(None)
        assert spiral_problem_instance.random_value_scale == original

    def test_set_default_origin(self, spiral_problem_instance):
        """Test set_default_origin method."""
        spiral_problem_instance.set_default_origin((0.5, 0.5))
        assert spiral_problem_instance.default_origin == (0.5, 0.5)

    def test_set_default_origin_none(self, spiral_problem_instance):
        """Test set_default_origin with None doesn't change value."""
        original = spiral_problem_instance.default_origin
        spiral_problem_instance.set_default_origin(None)
        assert spiral_problem_instance.default_origin == original

    def test_set_default_radius(self, spiral_problem_instance):
        """Test set_default_radius method."""
        spiral_problem_instance.set_default_radius(2.0)
        assert spiral_problem_instance.default_radius == 2.0


@pytest.mark.unit
class TestSpiralProblemGetters:
    """Tests for SpiralProblem getter methods."""

    def test_get_network(self, spiral_problem_instance):
        """Test get_network method."""
        result = spiral_problem_instance.get_network()
        assert result == spiral_problem_instance.network

    def test_get_n_spirals(self, spiral_problem_instance):
        """Test get_n_spirals method."""
        result = spiral_problem_instance.get_n_spirals()
        assert result == spiral_problem_instance.n_spirals

    def test_get_n_points(self, spiral_problem_instance):
        """Test get_n_points method."""
        result = spiral_problem_instance.get_n_points()
        assert result == spiral_problem_instance.n_points

    def test_get_n_rotations(self, spiral_problem_instance):
        """Test get_n_rotations method."""
        result = spiral_problem_instance.get_n_rotations()
        assert result == spiral_problem_instance.n_rotations

    def test_get_clockwise(self, spiral_problem_instance):
        """Test get_clockwise method."""
        result = spiral_problem_instance.get_clockwise()
        assert result == spiral_problem_instance.clockwise

    def test_get_noise(self, spiral_problem_instance):
        """Test get_noise method."""
        result = spiral_problem_instance.get_noise()
        assert result == spiral_problem_instance.noise

    def test_get_distribution(self, spiral_problem_instance):
        """Test get_distribution method."""
        result = spiral_problem_instance.get_distribution()
        assert result == spiral_problem_instance.distribution

    def test_get_random_seed(self, spiral_problem_instance):
        """Test get_random_seed method."""
        result = spiral_problem_instance.get_random_seed()
        assert result == spiral_problem_instance.random_seed

    def test_get_train_ratio(self, spiral_problem_instance):
        """Test get_train_ratio method."""
        result = spiral_problem_instance.get_train_ratio()
        assert result == spiral_problem_instance.train_ratio

    def test_get_test_ratio(self, spiral_problem_instance):
        """Test get_test_ratio method."""
        result = spiral_problem_instance.get_test_ratio()
        assert result == spiral_problem_instance.test_ratio

    def test_get_plot(self, spiral_problem_instance):
        """Test get_plot method."""
        spiral_problem_instance.plot = True  # Set initial value
        result = spiral_problem_instance.get_plot()
        assert result is True

    def test_get_random_value_scale(self, spiral_problem_instance):
        """Test get_random_value_scale method."""
        result = spiral_problem_instance.get_random_value_scale()
        assert result == spiral_problem_instance.random_value_scale

    def test_get_default_origin(self, spiral_problem_instance):
        """Test get_default_origin method."""
        result = spiral_problem_instance.get_default_origin()
        assert result == spiral_problem_instance.default_origin

    def test_get_default_radius(self, spiral_problem_instance):
        """Test get_default_radius method."""
        result = spiral_problem_instance.get_default_radius()
        assert result == spiral_problem_instance.default_radius


@pytest.mark.unit
class TestSpiralProblemUUID:
    """Tests for UUID methods."""

    def test_get_uuid_generates_if_none(self, spiral_problem_instance):
        """Test get_uuid generates UUID if none set."""
        if hasattr(spiral_problem_instance, "uuid"):
            delattr(spiral_problem_instance, "uuid")
        result = spiral_problem_instance.get_uuid()
        assert result is not None
        assert isinstance(result, str)

    def test_set_uuid_with_value(self, spiral_problem_instance):
        """Test set_uuid with provided value."""
        if hasattr(spiral_problem_instance, "uuid"):
            delattr(spiral_problem_instance, "uuid")
        spiral_problem_instance.set_uuid("test-uuid-123")
        assert spiral_problem_instance.uuid == "test-uuid-123"

    def test_set_uuid_generates_if_none(self, spiral_problem_instance):
        """Test set_uuid generates UUID if None provided."""
        if hasattr(spiral_problem_instance, "uuid"):
            delattr(spiral_problem_instance, "uuid")
        spiral_problem_instance.set_uuid(None)
        assert spiral_problem_instance.uuid is not None


@pytest.mark.unit
class TestGenerateUUID:
    """Tests for _generate_uuid method."""

    def test_generate_uuid_returns_string(self, spiral_problem_instance):
        """Test _generate_uuid returns a string."""
        result = spiral_problem_instance._generate_uuid()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_uuid_unique(self, spiral_problem_instance):
        """Test _generate_uuid returns unique values."""
        uuid1 = spiral_problem_instance._generate_uuid()
        uuid2 = spiral_problem_instance._generate_uuid()
        assert uuid1 != uuid2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
