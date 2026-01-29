#!/usr/bin/env python
"""
Unit tests for spiral_problem/check.py module.
Tests the SpiralProblem class in check.py (separate from spiral_problem.py).
"""

import os
import sys

# Add parent directories to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


@pytest.mark.unit
class TestSpiralProblemCheckInit:
    """Tests for SpiralProblem.__init__ in check.py."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        with patch("spiral_problem.check.CascadeCorrelationNetwork") as mock_network:
            mock_network.return_value = MagicMock()
            from spiral_problem.check import SpiralProblem

            sp = SpiralProblem()

            assert sp.n_spirals == 2
            assert sp.n_points >= 100  # Default can vary, just check it's set
            assert sp.n_rotations is not None
            assert sp.clockwise is not None
            assert sp.noise is not None
            assert sp.distribution is not None
            assert sp.random_seed is not None
            assert sp.train_ratio is not None
            assert sp.test_ratio is not None
            assert hasattr(sp, "logger")
            assert hasattr(sp, "network")

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch("spiral_problem.check.CascadeCorrelationNetwork") as mock_network:
            mock_network.return_value = MagicMock()
            from spiral_problem.check import SpiralProblem

            sp = SpiralProblem(
                _SpiralProblem__n_spirals=3,
                _SpiralProblem__n_points=50,
                _SpiralProblem__n_rotations=2,
                _SpiralProblem__clockwise=False,
                _SpiralProblem__noise=0.05,
                _SpiralProblem__distribution=0.5,
                _SpiralProblem__random_seed=123,
                _SpiralProblem__train_ratio=0.7,
                _SpiralProblem__test_ratio=0.3,
            )

            assert sp.n_spirals == 3
            assert sp.n_points == 50
            assert sp.n_rotations == 2
            assert sp.clockwise is False
            assert sp.noise == 0.05
            assert sp.distribution == 0.5
            assert sp.random_seed == 123
            assert sp.train_ratio == 0.7
            assert sp.test_ratio == 0.3

    def test_init_sets_random_seeds(self):
        """Test that initialization sets random seeds for reproducibility."""
        with patch("spiral_problem.check.CascadeCorrelationNetwork") as mock_network:
            mock_network.return_value = MagicMock()
            from spiral_problem.check import SpiralProblem

            sp = SpiralProblem(_SpiralProblem__random_seed=42)

            assert sp.random_seed == 42


@pytest.mark.unit
class TestGenerateNSpiralDataset:
    """Tests for generate_n_spiral_dataset method."""

    @pytest.fixture
    def spiral_problem(self):
        """Create a SpiralProblem instance with mocked network."""
        with patch("spiral_problem.check.CascadeCorrelationNetwork") as mock_network:
            mock_network.return_value = MagicMock()
            from spiral_problem.check import SpiralProblem

            return SpiralProblem(_SpiralProblem__n_points=20, _SpiralProblem__random_seed=42)

    def test_generate_dataset_default_params(self, spiral_problem):
        """Test dataset generation with default parameters."""
        result = spiral_problem.generate_n_spiral_dataset(
            num_points=20,
            num_spirals=2,
            train_ratio=0.8,
            test_ratio=0.2,
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        (train_x, train_y), (test_x, test_y), (full_x, full_y) = result

        assert isinstance(train_x, torch.Tensor)
        assert isinstance(train_y, torch.Tensor)
        assert isinstance(test_x, torch.Tensor)
        assert isinstance(test_y, torch.Tensor)
        assert isinstance(full_x, torch.Tensor)
        assert isinstance(full_y, torch.Tensor)

    def test_generate_dataset_shapes(self, spiral_problem):
        """Test that generated dataset has correct shapes."""
        num_points = 20
        num_spirals = 2
        total_points = num_points * num_spirals
        train_ratio = 0.8
        test_ratio = 0.2

        result = spiral_problem.generate_n_spiral_dataset(
            num_points=num_points,
            num_spirals=num_spirals,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
        )

        (train_x, train_y), (test_x, test_y), (full_x, full_y) = result

        assert full_x.shape[0] == total_points
        assert full_x.shape[1] == 2
        assert full_y.shape[0] == total_points
        assert full_y.shape[1] == num_spirals

        expected_train_size = int(train_ratio * total_points)
        expected_test_size = int(test_ratio * total_points)

        assert train_x.shape[0] == expected_train_size
        assert test_x.shape[0] == expected_test_size

    def test_generate_dataset_invalid_ratios(self, spiral_problem):
        """Test that invalid train/test ratios raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            spiral_problem.generate_n_spiral_dataset(
                num_points=20,
                num_spirals=2,
                train_ratio=0.7,
                test_ratio=0.2,
            )

    def test_generate_dataset_with_noise(self, spiral_problem):
        """Test dataset generation with noise."""
        result = spiral_problem.generate_n_spiral_dataset(
            num_points=20,
            num_spirals=2,
            train_ratio=0.8,
            test_ratio=0.2,
            noise_level=0.1,
        )

        (train_x, train_y), (test_x, test_y), (full_x, full_y) = result
        assert full_x.shape[0] == 40

    def test_generate_dataset_clockwise(self, spiral_problem):
        """Test dataset generation with clockwise direction."""
        result = spiral_problem.generate_n_spiral_dataset(
            num_points=20,
            num_spirals=2,
            train_ratio=0.8,
            test_ratio=0.2,
            clockwise=True,
        )

        assert len(result) == 3

    def test_generate_dataset_counter_clockwise(self, spiral_problem):
        """Test dataset generation with counter-clockwise direction."""
        result = spiral_problem.generate_n_spiral_dataset(
            num_points=20,
            num_spirals=2,
            train_ratio=0.8,
            test_ratio=0.2,
            clockwise=False,
        )

        assert len(result) == 3

    def test_generate_dataset_three_spirals(self, spiral_problem):
        """Test dataset generation with three spirals."""
        result = spiral_problem.generate_n_spiral_dataset(
            num_points=15,
            num_spirals=3,
            train_ratio=0.8,
            test_ratio=0.2,
        )

        (train_x, train_y), (test_x, test_y), (full_x, full_y) = result
        assert full_y.shape[1] == 3

    def test_generate_dataset_with_orig_points(self, spiral_problem):
        """Test dataset generation with provided original points."""
        orig_points = np.linspace(0, 1, 20)
        result = spiral_problem.generate_n_spiral_dataset(
            num_points=20,
            num_spirals=2,
            train_ratio=0.8,
            test_ratio=0.2,
            orig_points=orig_points,
        )

        assert len(result) == 3

    def test_generate_dataset_one_hot_encoding(self, spiral_problem):
        """Test that targets are properly one-hot encoded."""
        result = spiral_problem.generate_n_spiral_dataset(
            num_points=10,
            num_spirals=2,
            train_ratio=0.8,
            test_ratio=0.2,
        )

        (train_x, train_y), (test_x, test_y), (full_x, full_y) = result

        # Check one-hot encoding: each row should sum to 1
        assert torch.allclose(full_y.sum(dim=1), torch.ones(full_y.shape[0]))

    def test_generate_dataset_custom_distribution(self, spiral_problem):
        """Test dataset generation with custom distribution factor."""
        result = spiral_problem.generate_n_spiral_dataset(
            num_points=20,
            num_spirals=2,
            train_ratio=0.8,
            test_ratio=0.2,
            distribution=0.7,
        )

        assert len(result) == 3


@pytest.mark.unit
class TestSplitDataset:
    """Tests for split_dataset method."""

    @pytest.fixture
    def spiral_problem(self):
        """Create a SpiralProblem instance."""
        with patch("spiral_problem.check.CascadeCorrelationNetwork") as mock_network:
            mock_network.return_value = MagicMock()
            from spiral_problem.check import SpiralProblem

            return SpiralProblem(_SpiralProblem__n_points=20)

    def test_split_dataset_basic(self, spiral_problem):
        """Test basic dataset splitting."""
        total_points = 100
        x = torch.randn(total_points, 2)
        y = torch.zeros(total_points, 2)
        y[:50, 0] = 1
        y[50:, 1] = 1

        result = spiral_problem.split_dataset(
            total_points=total_points,
            partitions=(0.8, 0.2),
            x=x,
            y=y,
        )

        assert len(result) == 2
        (train_x, train_y), (test_x, test_y) = result
        assert train_x.shape[0] == 80
        assert test_x.shape[0] == 20

    def test_split_dataset_none_total_points(self, spiral_problem):
        """Test that None total_points raises ValueError."""
        x = torch.randn(100, 2)
        y = torch.zeros(100, 2)

        with pytest.raises(ValueError, match="total_points and partitions must be provided"):
            spiral_problem.split_dataset(
                total_points=None,
                partitions=(0.8, 0.2),
                x=x,
                y=y,
            )

    def test_split_dataset_none_partitions(self, spiral_problem):
        """Test that None partitions raises ValueError."""
        x = torch.randn(100, 2)
        y = torch.zeros(100, 2)

        with pytest.raises(ValueError, match="total_points and partitions must be provided"):
            spiral_problem.split_dataset(
                total_points=100,
                partitions=None,
                x=x,
                y=y,
            )

    def test_split_dataset_none_x(self, spiral_problem):
        """Test that None x raises ValueError."""
        y = torch.zeros(100, 2)

        with pytest.raises(ValueError, match="x and y must be provided"):
            spiral_problem.split_dataset(
                total_points=100,
                partitions=(0.8, 0.2),
                x=None,
                y=y,
            )

    def test_split_dataset_none_y(self, spiral_problem):
        """Test that None y raises ValueError."""
        x = torch.randn(100, 2)

        with pytest.raises(ValueError, match="x and y must be provided"):
            spiral_problem.split_dataset(
                total_points=100,
                partitions=(0.8, 0.2),
                x=x,
                y=None,
            )

    def test_split_dataset_invalid_total_points_type(self, spiral_problem):
        """Test that non-integer total_points raises ValueError."""
        x = torch.randn(100, 2)
        y = torch.zeros(100, 2)

        with pytest.raises(ValueError, match="total_points must be a positive integer"):
            spiral_problem.split_dataset(
                total_points="100",
                partitions=(0.8, 0.2),
                x=x,
                y=y,
            )

    def test_split_dataset_negative_total_points(self, spiral_problem):
        """Test that negative total_points raises ValueError."""
        x = torch.randn(100, 2)
        y = torch.zeros(100, 2)

        with pytest.raises(ValueError, match="total_points must be a positive integer"):
            spiral_problem.split_dataset(
                total_points=-10,
                partitions=(0.8, 0.2),
                x=x,
                y=y,
            )

    def test_split_dataset_zero_total_points(self, spiral_problem):
        """Test that zero total_points raises ValueError."""
        x = torch.randn(100, 2)
        y = torch.zeros(100, 2)

        with pytest.raises(ValueError, match="total_points must be a positive integer"):
            spiral_problem.split_dataset(
                total_points=0,
                partitions=(0.8, 0.2),
                x=x,
                y=y,
            )

    def test_split_dataset_invalid_partitions_type(self, spiral_problem):
        """Test that non-tuple partitions raises ValueError."""
        x = torch.randn(100, 2)
        y = torch.zeros(100, 2)

        with pytest.raises(ValueError, match="partitions must be a tuple of floats"):
            spiral_problem.split_dataset(
                total_points=100,
                partitions=[0.8, 0.2],
                x=x,
                y=y,
            )

    def test_split_dataset_empty_partitions(self, spiral_problem):
        """Test that empty partitions raises ValueError."""
        x = torch.randn(100, 2)
        y = torch.zeros(100, 2)

        with pytest.raises(ValueError, match="partitions must contain at least one partition"):
            spiral_problem.split_dataset(
                total_points=100,
                partitions=(),
                x=x,
                y=y,
            )

    def test_split_dataset_mismatched_x_length(self, spiral_problem):
        """Test that mismatched x length raises ValueError."""
        x = torch.randn(80, 2)
        y = torch.zeros(100, 2)

        with pytest.raises(ValueError, match="x and y must have the same length"):
            spiral_problem.split_dataset(
                total_points=100,
                partitions=(0.8, 0.2),
                x=x,
                y=y,
            )

    def test_split_dataset_mismatched_y_length(self, spiral_problem):
        """Test that mismatched y length raises ValueError."""
        x = torch.randn(100, 2)
        y = torch.zeros(80, 2)

        with pytest.raises(ValueError, match="x and y must have the same length"):
            spiral_problem.split_dataset(
                total_points=100,
                partitions=(0.8, 0.2),
                x=x,
                y=y,
            )

    def test_split_dataset_partitions_not_sum_to_one(self, spiral_problem):
        """Test that partitions not summing to 1.0 raises ValueError."""
        x = torch.randn(100, 2)
        y = torch.zeros(100, 2)

        with pytest.raises(ValueError, match="Partitions must sum to 1.0"):
            spiral_problem.split_dataset(
                total_points=100,
                partitions=(0.6, 0.2),
                x=x,
                y=y,
            )

    def test_split_dataset_partition_out_of_range(self, spiral_problem):
        """Test that partition out of [0, 1] range raises ValueError."""
        x = torch.randn(100, 2)
        y = torch.zeros(100, 2)

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            spiral_problem.split_dataset(
                total_points=100,
                partitions=(1.5, -0.5),
                x=x,
                y=y,
            )

    def test_split_dataset_three_partitions(self, spiral_problem):
        """Test splitting with three partitions."""
        total_points = 100
        x = torch.randn(total_points, 2)
        y = torch.zeros(total_points, 2)

        result = spiral_problem.split_dataset(
            total_points=total_points,
            partitions=(0.6, 0.2, 0.2),
            x=x,
            y=y,
        )

        assert len(result) == 3


@pytest.mark.unit
class TestDatasetSplitIndexEnd:
    """Tests for dataset_split_index_end method."""

    @pytest.fixture
    def spiral_problem(self):
        """Create a SpiralProblem instance."""
        with patch("spiral_problem.check.CascadeCorrelationNetwork") as mock_network:
            mock_network.return_value = MagicMock()
            from spiral_problem.check import SpiralProblem

            return SpiralProblem(_SpiralProblem__n_points=20)

    def test_dataset_split_index_end_basic(self, spiral_problem):
        """Test basic index end calculation."""
        result = spiral_problem.dataset_split_index_end(
            total_points=100,
            split_ratio=0.8,
        )
        assert result == 80

    def test_dataset_split_index_end_zero_ratio(self, spiral_problem):
        """Test index end with zero ratio."""
        result = spiral_problem.dataset_split_index_end(
            total_points=100,
            split_ratio=0.0,
        )
        assert result == 0

    def test_dataset_split_index_end_full_ratio(self, spiral_problem):
        """Test index end with ratio of 1.0."""
        result = spiral_problem.dataset_split_index_end(
            total_points=100,
            split_ratio=1.0,
        )
        assert result == 100

    def test_dataset_split_index_end_half(self, spiral_problem):
        """Test index end with half ratio."""
        result = spiral_problem.dataset_split_index_end(
            total_points=100,
            split_ratio=0.5,
        )
        assert result == 50


@pytest.mark.unit
class TestFindPartitionIndexEnd:
    """Tests for find_partition_index_end method."""

    @pytest.fixture
    def spiral_problem(self):
        """Create a SpiralProblem instance."""
        with patch("spiral_problem.check.CascadeCorrelationNetwork") as mock_network:
            mock_network.return_value = MagicMock()
            from spiral_problem.check import SpiralProblem

            return SpiralProblem(_SpiralProblem__n_points=20)

    def test_find_partition_index_end_basic(self, spiral_problem):
        """Test basic partition index end calculation."""
        result = spiral_problem.find_partition_index_end(
            partition_start=0,
            total_points=100,
            partition=0.8,
        )
        assert result == 80

    def test_find_partition_index_end_from_middle(self, spiral_problem):
        """Test partition index end starting from middle."""
        result = spiral_problem.find_partition_index_end(
            partition_start=50,
            total_points=100,
            partition=0.3,
        )
        assert result == 80


@pytest.mark.unit
class TestMakeNoise:
    """Tests for make_noise method."""

    @pytest.fixture
    def spiral_problem(self):
        """Create a SpiralProblem instance."""
        with patch("spiral_problem.check.CascadeCorrelationNetwork") as mock_network:
            mock_network.return_value = MagicMock()
            from spiral_problem.check import SpiralProblem

            return SpiralProblem(_SpiralProblem__n_points=20)

    def test_make_noise_basic(self, spiral_problem):
        """Test basic noise generation."""
        result = spiral_problem.make_noise(n_points=10, noise=0.1)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert np.all(result >= 0)
        assert np.all(result <= 0.1)

    def test_make_noise_zero_noise(self, spiral_problem):
        """Test noise generation with zero noise factor."""
        result = spiral_problem.make_noise(n_points=10, noise=0.0)
        assert np.all(result == 0)

    def test_make_noise_zero_points(self, spiral_problem):
        """Test noise generation with zero points."""
        result = spiral_problem.make_noise(n_points=0, noise=0.1)
        assert len(result) == 0

    def test_make_noise_large_noise(self, spiral_problem):
        """Test noise generation with large noise factor."""
        result = spiral_problem.make_noise(n_points=100, noise=1.0)
        assert len(result) == 100
        assert np.all(result >= 0)
        assert np.all(result <= 1.0)


@pytest.mark.unit
class TestSolveNSpiralProblem:
    """Tests for solve_n_spiral_problem method."""

    @pytest.fixture
    def spiral_problem(self):
        """Create a SpiralProblem instance with mocked network."""
        with patch("spiral_problem.check.CascadeCorrelationNetwork") as mock_network:
            mock_instance = MagicMock()
            mock_instance.fit.return_value = {"loss": [0.5, 0.3, 0.1]}
            mock_instance.summary.return_value = None
            mock_instance.plot_dataset.return_value = None
            mock_instance.plot_decision_boundary.return_value = None
            mock_instance.plot_training_history.return_value = None
            mock_network.return_value = mock_instance

            from spiral_problem.check import SpiralProblem

            return SpiralProblem(
                _SpiralProblem__n_points=10,
                _SpiralProblem__random_seed=42,
            )

    def test_solve_n_spiral_problem_basic(self, spiral_problem):
        """Test basic spiral problem solving."""
        spiral_problem.solve_n_spiral_problem(
            n_points=10,
            n_spirals=2,
            train_ratio=0.8,
            test_ratio=0.2,
            plot=False,
        )

        assert hasattr(spiral_problem, "x_train")
        assert hasattr(spiral_problem, "y_train")
        assert hasattr(spiral_problem, "x_test")
        assert hasattr(spiral_problem, "y_test")
        assert hasattr(spiral_problem, "x_full")
        assert hasattr(spiral_problem, "y_full")
        assert spiral_problem.network.fit.called

    def test_solve_n_spiral_problem_with_plot(self, spiral_problem):
        """Test spiral problem solving with plotting enabled."""
        spiral_problem.solve_n_spiral_problem(
            n_points=10,
            n_spirals=2,
            train_ratio=0.8,
            test_ratio=0.2,
            plot=True,
        )

        assert spiral_problem.network.plot_dataset.called
        assert spiral_problem.network.plot_decision_boundary.called
        assert spiral_problem.network.plot_training_history.called

    def test_solve_n_spiral_problem_stores_history(self, spiral_problem):
        """Test that training history is stored."""
        spiral_problem.solve_n_spiral_problem(
            n_points=10,
            n_spirals=2,
            train_ratio=0.8,
            test_ratio=0.2,
            plot=False,
        )

        assert hasattr(spiral_problem, "history")
        assert spiral_problem.history is not None

    def test_solve_n_spiral_problem_uses_class_defaults(self, spiral_problem):
        """Test that method uses class defaults when params are None."""
        spiral_problem.n_points = 15
        spiral_problem.n_spirals = 3

        spiral_problem.solve_n_spiral_problem(
            n_points=None,
            n_spirals=None,
            train_ratio=0.8,
            test_ratio=0.2,
            plot=False,
        )

        assert spiral_problem.n_points == 15
        assert spiral_problem.n_spirals == 3


@pytest.mark.unit
class TestEvaluate:
    """Tests for evaluate method."""

    @pytest.fixture
    def spiral_problem(self):
        """Create a SpiralProblem instance with mocked network."""
        with patch("spiral_problem.check.CascadeCorrelationNetwork") as mock_network:
            mock_instance = MagicMock()
            mock_instance.fit.return_value = {"loss": [0.5, 0.3, 0.1]}
            mock_instance.summary.return_value = None
            mock_instance.plot_dataset.return_value = None
            mock_instance.plot_decision_boundary.return_value = None
            mock_instance.plot_training_history.return_value = None
            mock_instance.calculate_accuracy.return_value = 0.85
            mock_network.return_value = mock_instance

            from spiral_problem.check import SpiralProblem

            return SpiralProblem(
                _SpiralProblem__n_points=10,
                _SpiralProblem__random_seed=42,
            )

    def test_evaluate_basic(self, spiral_problem):
        """Test basic evaluation."""
        spiral_problem.evaluate(
            n_points=10,
            n_spirals=2,
            train_ratio=0.8,
            test_ratio=0.2,
            plot=False,
        )

        assert hasattr(spiral_problem, "train_accuracy")
        assert hasattr(spiral_problem, "test_accuracy")
        assert hasattr(spiral_problem, "train_accuracy_percent")
        assert hasattr(spiral_problem, "test_accuracy_percent")

    def test_evaluate_calculates_accuracy(self, spiral_problem):
        """Test that evaluate calculates accuracy correctly."""
        spiral_problem.evaluate(
            n_points=10,
            n_spirals=2,
            train_ratio=0.8,
            test_ratio=0.2,
            plot=False,
        )

        assert spiral_problem.train_accuracy == 0.85
        assert spiral_problem.test_accuracy == 0.85
        assert spiral_problem.train_accuracy_percent == 85.0
        assert spiral_problem.test_accuracy_percent == 85.0

    def test_evaluate_calls_network_summary(self, spiral_problem):
        """Test that evaluate calls network summary."""
        spiral_problem.evaluate(
            n_points=10,
            n_spirals=2,
            train_ratio=0.8,
            test_ratio=0.2,
            plot=False,
        )

        # summary is called twice - once in solve_n_spiral_problem and once in evaluate
        assert spiral_problem.network.summary.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
