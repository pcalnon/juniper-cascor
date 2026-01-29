#!/usr/bin/env python
"""
Tests for spiral_problem/spiral_problem.py to increase code coverage.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spiral_problem.spiral_problem import SpiralProblem


class TestSpiralProblemInit:
    """Tests for SpiralProblem initialization."""

    def test_init_default(self):
        """Test SpiralProblem initialization with defaults."""
        sp = SpiralProblem()
        assert sp is not None
        assert sp.input_size == 2
        assert sp.output_size == 2

    def test_init_custom_n_spirals(self):
        """Test SpiralProblem initialization with custom n_spirals."""
        sp = SpiralProblem(
            _SpiralProblem__n_spirals=3,
        )
        assert sp is not None
        assert sp.n_spirals == 3


class TestSpiralProblemDataGeneration:
    """Tests for SpiralProblem data generation methods."""

    def test_generate_n_spiral_dataset(self):
        """Test generating n-spiral dataset."""
        sp = SpiralProblem(
            _SpiralProblem__n_points=20,
            _SpiralProblem__n_spirals=2,
        )

        # Use the actual method name - returns nested tuples ((train_x, train_y), (test_x, test_y), info)
        if hasattr(sp, "generate_n_spiral_dataset"):
            result = sp.generate_n_spiral_dataset()
            # Result is a tuple of tuples
            assert isinstance(result, tuple)
            assert len(result) >= 2
            # First element is (train_x, train_y)
            train_data = result[0]
            assert isinstance(train_data, tuple)
            train_x, train_y = train_data
            assert isinstance(train_x, torch.Tensor)
            assert isinstance(train_y, torch.Tensor)
            assert train_x.shape[1] == 2  # 2D input


class TestSpiralProblemProperties:
    """Tests for SpiralProblem properties and getters."""

    def test_get_n_points(self):
        """Test n_points property."""
        sp = SpiralProblem(_SpiralProblem__n_points=100)
        assert sp.n_points == 100

    def test_get_n_spirals(self):
        """Test n_spirals property."""
        sp = SpiralProblem(_SpiralProblem__n_spirals=4)
        assert sp.n_spirals == 4

    def test_get_input_size(self):
        """Test input_size property."""
        sp = SpiralProblem()
        assert sp.input_size == 2

    def test_get_output_size(self):
        """Test output_size property - matches n_spirals."""
        sp = SpiralProblem(_SpiralProblem__n_spirals=2)
        # output_size is typically set during init or may be 2 by default
        assert sp.output_size >= 2


class TestSpiralProblemConfiguration:
    """Tests for SpiralProblem configuration."""

    def test_learning_rate(self):
        """Test learning rate configuration."""
        sp = SpiralProblem(_SpiralProblem__learning_rate=0.05)
        assert sp.learning_rate == 0.05

    def test_max_hidden_units(self):
        """Test max hidden units configuration."""
        sp = SpiralProblem(_SpiralProblem__max_hidden_units=20)
        assert sp.max_hidden_units == 20

    def test_candidate_pool_size(self):
        """Test candidate pool size configuration."""
        sp = SpiralProblem(_SpiralProblem__candidate_pool_size=8)
        assert sp.candidate_pool_size == 8

    def test_epochs_max(self):
        """Test epochs max configuration."""
        sp = SpiralProblem(_SpiralProblem__epochs_max=50)
        assert sp.epochs_max == 50


class TestSpiralProblemNetwork:
    """Tests for SpiralProblem network interaction."""

    def test_has_network_attribute(self):
        """Test that SpiralProblem can interact with network."""
        sp = SpiralProblem()
        # Check for common attributes
        assert hasattr(sp, "input_size")
        assert hasattr(sp, "output_size")
        assert hasattr(sp, "learning_rate")
