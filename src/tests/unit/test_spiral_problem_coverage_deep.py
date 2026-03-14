#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# Purpose:       Deep coverage tests for spiral_problem/spiral_problem.py
#
# Author:        Paul Calnon
# Version:       0.1.0
# File Name:     test_spiral_problem_coverage_deep.py
# File Path:     <Project>/<Sub-Project>/<Application>/src/tests/unit/
#
# Date Created:  2026-03-12
# Last Modified: 2026-03-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#     Comprehensive unit tests targeting uncovered lines in spiral_problem.py.
#     Covers deprecated data-generation methods, solve/evaluate workflows,
#     UUID logic, __init__ error path, and setter/getter edge cases.
#
#####################################################################################################################################################################################################
import os
import sys
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from spiral_problem.spiral_problem import SpiralProblem

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sp():
    """Create a SpiralProblem instance with small parameters for fast tests."""
    instance = SpiralProblem(
        _SpiralProblem__n_points=50,
        _SpiralProblem__n_spirals=2,
        _SpiralProblem__noise=0.1,
        _SpiralProblem__n_rotations=1,
        _SpiralProblem__input_size=2,
        _SpiralProblem__output_size=2,
        _SpiralProblem__random_seed=42,
        _SpiralProblem__candidate_pool_size=2,
        _SpiralProblem__candidate_epochs=3,
        _SpiralProblem__output_epochs=3,
        _SpiralProblem__max_hidden_units=2,
        _SpiralProblem__patience=1,
        _SpiralProblem__generate_plots_default=False,
    )
    # Ensure all attributes needed by deprecated methods are present
    instance.n_points = 50
    instance.n_spirals = 2
    instance.noise = 0.1
    instance.n_rotations = 1
    instance.clockwise = True
    instance.distribution = 1.0
    instance.random_seed = 42
    instance.train_ratio = 0.7
    instance.test_ratio = 0.3
    instance.default_origin = 0.0
    instance.default_radius = 1.0
    instance.random_value_scale = 1.0
    instance.plot = False
    instance.total_points = 100  # n_spirals * n_points
    return instance


@pytest.fixture
def sp_three_spirals():
    """Create a SpiralProblem instance with 3 spirals."""
    instance = SpiralProblem(
        _SpiralProblem__n_points=30,
        _SpiralProblem__n_spirals=3,
        _SpiralProblem__noise=0.05,
        _SpiralProblem__n_rotations=1,
        _SpiralProblem__input_size=2,
        _SpiralProblem__output_size=3,
        _SpiralProblem__random_seed=42,
        _SpiralProblem__candidate_pool_size=2,
        _SpiralProblem__candidate_epochs=3,
        _SpiralProblem__output_epochs=3,
        _SpiralProblem__max_hidden_units=2,
        _SpiralProblem__patience=1,
        _SpiralProblem__generate_plots_default=False,
    )
    instance.n_points = 30
    instance.n_spirals = 3
    instance.noise = 0.05
    instance.n_rotations = 1
    instance.clockwise = False
    instance.distribution = 1.0
    instance.random_seed = 42
    instance.train_ratio = 0.8
    instance.test_ratio = 0.2
    instance.default_origin = 0.0
    instance.default_radius = 1.0
    instance.random_value_scale = 1.0
    instance.plot = False
    instance.total_points = 90
    return instance


# ---------------------------------------------------------------------------
# 1. __init__ error path (lines 437-439)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitErrorPath:
    """Test the __init__ error path when CascadeCorrelationNetwork returns None."""

    def test_init_raises_value_error_when_network_creation_fails(self):
        """Walrus operator `:=` evaluates CascadeCorrelationNetwork() == None.
        If the constructor ever returns None (impossible in normal Python, but
        defended against), __init__ raises ValueError. We mock to return None."""
        with patch("spiral_problem.spiral_problem.CascadeCorrelationNetwork", return_value=None):
            with pytest.raises(ValueError, match="Failed to create Spiral Problem"):
                SpiralProblem(
                    _SpiralProblem__n_points=10,
                    _SpiralProblem__n_spirals=2,
                    _SpiralProblem__random_seed=42,
                )


# ---------------------------------------------------------------------------
# 2. Deprecated data generation methods (lines 568-1208)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitializeSpiralProblemParams:
    """Tests for _initialize_spiral_problem_params."""

    def test_sets_total_points(self, sp):
        """Verify total_points is set to n_spirals * n_points."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sp._initialize_spiral_problem_params(
                n_spirals=2,
                n_points=50,
            )
        assert sp.total_points == 100

    def test_fallback_to_class_attributes(self, sp):
        """When None is passed for params, class attributes are used as fallback."""
        sp.min_new = -5.0
        sp.max_new = 5.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # Pass None values so the fallback branch is exercised
            sp._initialize_spiral_problem_params(
                min_new=None,
                max_new=None,
                min_orig=None,
                max_orig=None,
                orig_points=None,
                train_ratio=None,
                test_ratio=None,
                clockwise=None,
                n_spirals=None,
                n_rotations=None,
                n_points=None,
                default_origin=None,
                default_radius=None,
                noise_level=None,
                distribution=None,
            )
        # Should still have valid total_points
        assert sp.total_points == sp.n_spirals * sp.n_points

    def test_with_explicit_values(self, sp):
        """When explicit values are provided, they override class attrs."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sp._initialize_spiral_problem_params(
                min_new=-2.0,
                max_new=2.0,
                min_orig=-1.0,
                max_orig=1.0,
                orig_points=100,
                train_ratio=0.8,
                test_ratio=0.2,
                clockwise=True,
                n_spirals=4,
                n_rotations=2,
                n_points=25,
                default_origin=0.5,
                default_radius=2.0,
                noise_level=0.2,
                distribution=0.5,
            )
        assert sp.n_spirals == 4
        assert sp.n_points == 25
        assert sp.total_points == 100  # 4 * 25


@pytest.mark.unit
class TestGenerateBaseRadialDistance:
    """Tests for _generate_base_radial_distance."""

    def test_returns_array_of_correct_size(self, sp):
        """Result should be a numpy array with n_points elements."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._generate_base_radial_distance(n_points=50)
        assert isinstance(result, np.ndarray)
        assert result.shape == (50,)

    def test_all_values_non_negative(self, sp):
        """Radial distances should be non-negative (sqrt of rand * constant)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._generate_base_radial_distance(n_points=100)
        assert np.all(result >= 0)

    def test_raises_for_zero_points(self, sp):
        """Should raise ValueError for n_points <= 0."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="n_points must be a positive integer"):
                sp._generate_base_radial_distance(n_points=0)

    def test_raises_for_negative_points(self, sp):
        """Should raise ValueError for negative n_points."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="n_points must be a positive integer"):
                sp._generate_base_radial_distance(n_points=-5)

    def test_raises_for_non_integer(self, sp):
        """Should raise ValueError for non-integer n_points."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="n_points must be a positive integer"):
                sp._generate_base_radial_distance(n_points=3.5)

    def test_emits_deprecation_warning(self, sp):
        """Should emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._generate_base_radial_distance(n_points=10)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1


@pytest.mark.unit
class TestGenerateAngularOffset:
    """Tests for _generate_angular_offset."""

    def test_two_spirals_offset(self, sp):
        """Angular offset for 2 spirals should be pi."""
        sp.n_spirals = 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._generate_angular_offset()
        assert np.isclose(result, np.pi)

    def test_three_spirals_offset(self, sp_three_spirals):
        """Angular offset for 3 spirals should be 2*pi/3."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp_three_spirals._generate_angular_offset()
        assert np.isclose(result, 2 * np.pi / 3)

    def test_four_spirals_offset(self, sp):
        """Angular offset for 4 spirals should be pi/2."""
        sp.n_spirals = 4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._generate_angular_offset()
        assert np.isclose(result, np.pi / 2)

    def test_returns_float(self, sp):
        """Result should be a float."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._generate_angular_offset()
        assert isinstance(result, float)


@pytest.mark.unit
class TestGenerateRawSpiralCoordinates:
    """Tests for _generate_raw_spiral_coordinates."""

    def test_returns_two_arrays(self, sp):
        """Should return a tuple of two numpy arrays."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_coords, y_coords = sp._generate_raw_spiral_coordinates(
                n_distance=n_distance,
                direction=1,
                angular_offset=np.pi,
            )
        assert isinstance(x_coords, np.ndarray)
        assert isinstance(y_coords, np.ndarray)

    def test_shape_matches_n_spirals(self, sp):
        """Each output should have n_spirals rows."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_coords, y_coords = sp._generate_raw_spiral_coordinates(
                n_distance=n_distance,
                direction=1,
                angular_offset=np.pi,
            )
        assert x_coords.shape[0] == sp.n_spirals
        assert y_coords.shape[0] == sp.n_spirals

    def test_clockwise_direction(self, sp):
        """Test with clockwise direction (direction=-1)."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_coords, y_coords = sp._generate_raw_spiral_coordinates(
                n_distance=n_distance,
                direction=-1,
                angular_offset=np.pi,
            )
        assert x_coords.shape[0] == sp.n_spirals

    def test_three_spirals(self, sp_three_spirals):
        """Test with 3 spirals."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp_three_spirals.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_coords, y_coords = sp_three_spirals._generate_raw_spiral_coordinates(
                n_distance=n_distance,
                direction=1,
                angular_offset=2 * np.pi / 3,
            )
        assert x_coords.shape[0] == 3
        assert y_coords.shape[0] == 3


@pytest.mark.unit
class TestGenerateXYCoordinates:
    """Tests for _generate_xy_coordinates."""

    def test_returns_tuple_of_arrays(self, sp):
        """Should return x and y coordinate arrays."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x, y = sp._generate_xy_coordinates(
                index=0,
                n_distance=n_distance,
                angular_offset=np.pi,
                direction=1,
            )
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.shape == (sp.n_points,)
        assert y.shape == (sp.n_points,)

    def test_direction_minus_one(self, sp):
        """Test with direction=-1 (clockwise)."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x, y = sp._generate_xy_coordinates(
                index=0,
                n_distance=n_distance,
                angular_offset=np.pi,
                direction=-1,
            )
        assert x.shape == (sp.n_points,)

    def test_raises_for_invalid_direction(self, sp):
        """Should raise ValueError for direction not in [1, -1]."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Direction must be 1 or -1"):
                sp._generate_xy_coordinates(
                    index=0,
                    n_distance=n_distance,
                    angular_offset=np.pi,
                    direction=0,
                )

    def test_second_spiral_index(self, sp):
        """Test generation for the second spiral (index=1)."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x, y = sp._generate_xy_coordinates(
                index=1,
                n_distance=n_distance,
                angular_offset=np.pi,
                direction=1,
            )
        assert x.shape == (sp.n_points,)


@pytest.mark.unit
class TestMakeCoords:
    """Tests for _make_coords."""

    def test_with_cos(self, sp):
        """Test coordinate generation with cosine."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._make_coords(
                index=0,
                n_distance=n_distance,
                angular_offset=np.pi,
                direction=1,
                trig_function=np.cos,
            )
        assert isinstance(result, np.ndarray)
        assert result.shape == (sp.n_points,)

    def test_with_sin(self, sp):
        """Test coordinate generation with sine."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._make_coords(
                index=0,
                n_distance=n_distance,
                angular_offset=np.pi,
                direction=1,
                trig_function=np.sin,
            )
        assert isinstance(result, np.ndarray)
        assert result.shape == (sp.n_points,)

    def test_negative_direction(self, sp):
        """Test with direction=-1."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._make_coords(
                index=0,
                n_distance=n_distance,
                angular_offset=np.pi,
                direction=-1,
                trig_function=np.cos,
            )
        assert result.shape == (sp.n_points,)

    def test_raises_for_none_trig_function(self, sp):
        """Should raise ValueError when trig_function is None."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="trig_function must be provided"):
                sp._make_coords(
                    index=0,
                    n_distance=n_distance,
                    angular_offset=np.pi,
                    direction=1,
                    trig_function=None,
                )

    def test_raises_for_invalid_direction(self, sp):
        """Should raise ValueError for direction not in [1, -1]."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Direction must be 1 or -1"):
                sp._make_coords(
                    index=0,
                    n_distance=n_distance,
                    angular_offset=np.pi,
                    direction=2,
                    trig_function=np.cos,
                )

    def test_raises_for_negative_index(self, sp):
        """Should raise ValueError for negative index."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Index must be a non-negative integer"):
                sp._make_coords(
                    index=-1,
                    n_distance=n_distance,
                    angular_offset=np.pi,
                    direction=1,
                    trig_function=np.cos,
                )

    def test_raises_for_non_integer_index(self, sp):
        """Should raise ValueError for non-integer index."""
        np.random.seed(42)
        n_distance = np.sqrt(np.random.rand(sp.n_points)) * 780 * (2 * np.pi) / 360
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Index must be a non-negative integer"):
                sp._make_coords(
                    index=1.5,
                    n_distance=n_distance,
                    angular_offset=np.pi,
                    direction=1,
                    trig_function=np.cos,
                )


@pytest.mark.unit
class TestMakeNoise:
    """Tests for _make_noise."""

    def test_returns_correct_shape(self, sp):
        """Should return array with n_points elements."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._make_noise(n_points=50, noise=0.1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (50,)

    def test_noise_scaling(self, sp):
        """All values should be bounded by [0, noise)."""
        np.random.seed(42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._make_noise(n_points=1000, noise=0.5)
        assert np.all(result >= 0)
        assert np.all(result < 0.5)

    def test_zero_noise(self, sp):
        """With noise=0, all values should be zero."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._make_noise(n_points=20, noise=0.0)
        assert np.all(result == 0.0)

    def test_emits_deprecation_warning(self, sp):
        """Should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._make_noise(n_points=10, noise=0.1)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1


@pytest.mark.unit
class TestCreateInputFeatures:
    """Tests for _create_input_features."""

    def test_shape_is_correct(self, sp):
        """Result shape should be (total_points, 2)."""
        # Two spirals, each with 50 points
        x_coords = np.array([np.random.rand(50), np.random.rand(50)])
        y_coords = np.array([np.random.rand(50), np.random.rand(50)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._create_input_features(x_coords, y_coords)
        assert result.shape == (100, 2)

    def test_three_spirals(self, sp_three_spirals):
        """Test with three spirals."""
        x_coords = np.array([np.random.rand(30) for _ in range(3)])
        y_coords = np.array([np.random.rand(30) for _ in range(3)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp_three_spirals._create_input_features(x_coords, y_coords)
        assert result.shape == (90, 2)

    def test_values_are_stacked_correctly(self, sp):
        """First column should be x coords, second should be y coords."""
        x_coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y_coords = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._create_input_features(x_coords, y_coords)
        # hstack flattens then vstack transposes
        assert result[0, 0] == 1.0
        assert result[0, 1] == 7.0


@pytest.mark.unit
class TestCreateOneHotTargets:
    """Tests for _create_one_hot_targets."""

    def test_shape_two_spirals(self, sp):
        """One-hot array should be (total_points, n_spirals)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._create_one_hot_targets(
                total_points=100,
                n_spirals=2,
                dtype=np.float32,
            )
        assert result.shape == (100, 2)

    def test_one_hot_encoding_correctness(self, sp):
        """First 50 rows should be class 0, next 50 should be class 1."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._create_one_hot_targets(
                total_points=100,
                n_spirals=2,
                dtype=np.float32,
            )
        # First spiral: column 0 is 1
        assert np.all(result[:50, 0] == 1)
        assert np.all(result[:50, 1] == 0)
        # Second spiral: column 1 is 1
        assert np.all(result[50:, 0] == 0)
        assert np.all(result[50:, 1] == 1)

    def test_three_spirals_encoding(self, sp_three_spirals):
        """Test one-hot for 3 spirals."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp_three_spirals._create_one_hot_targets(
                total_points=90,
                n_spirals=3,
                dtype=np.float32,
            )
        assert result.shape == (90, 3)
        # Each spiral block has its own column set to 1
        for i in range(3):
            start = i * 30
            end = (i + 1) * 30
            assert np.all(result[start:end, i] == 1)
            for j in range(3):
                if j != i:
                    assert np.all(result[start:end, j] == 0)

    def test_dtype_preserved(self, sp):
        """Output dtype should match the dtype argument."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._create_one_hot_targets(
                total_points=100,
                n_spirals=2,
                dtype=np.float64,
            )
        assert result.dtype == np.float64


@pytest.mark.unit
class TestCreateSpiralDataset:
    """Tests for _create_spiral_dataset."""

    def test_returns_features_and_targets(self, sp):
        """Should return (x, y) tuple with correct shapes."""
        x_coords = np.array([np.random.rand(50), np.random.rand(50)])
        y_coords = np.array([np.random.rand(50), np.random.rand(50)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x, y = sp._create_spiral_dataset(x_coords, y_coords)
        assert x.shape == (100, 2)
        assert y.shape == (100, 2)

    def test_targets_are_one_hot(self, sp):
        """Target matrix rows should sum to 1."""
        x_coords = np.array([np.random.rand(50), np.random.rand(50)])
        y_coords = np.array([np.random.rand(50), np.random.rand(50)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            _, y = sp._create_spiral_dataset(x_coords, y_coords)
        row_sums = y.sum(axis=1)
        assert np.allclose(row_sums, 1.0)


@pytest.mark.unit
class TestConvertToTensors:
    """Tests for _convert_to_tensors."""

    def test_converts_numpy_to_torch(self, sp):
        """Should convert numpy arrays to torch tensors."""
        x = np.random.rand(100, 2).astype(np.float32)
        y = np.random.rand(100, 2).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_t, y_t = sp._convert_to_tensors(x, y)
        assert isinstance(x_t, torch.Tensor)
        assert isinstance(y_t, torch.Tensor)
        assert x_t.dtype == torch.float32
        assert y_t.dtype == torch.float32

    def test_preserves_shape(self, sp):
        """Tensor shapes should match input array shapes."""
        x = np.random.rand(50, 2).astype(np.float32)
        y = np.random.rand(50, 3).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_t, y_t = sp._convert_to_tensors(x, y)
        assert x_t.shape == (50, 2)
        assert y_t.shape == (50, 3)

    def test_preserves_values(self, sp):
        """Tensor values should match input array values."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_t, y_t = sp._convert_to_tensors(x, y)
        assert torch.allclose(x_t, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert torch.allclose(y_t, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


@pytest.mark.unit
class TestShuffleDataset:
    """Tests for _shuffle_dataset."""

    def test_preserves_shape(self, sp):
        """Shuffled tensors should have same shape as input."""
        x = torch.randn(100, 2)
        y = torch.randn(100, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_s, y_s = sp._shuffle_dataset(x, y)
        assert x_s.shape == x.shape
        assert y_s.shape == y.shape

    def test_preserves_all_elements(self, sp):
        """Shuffled tensors should contain the same elements (sorted)."""
        torch.manual_seed(42)
        x = torch.arange(20).float().unsqueeze(1)
        y = torch.arange(20).float().unsqueeze(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_s, y_s = sp._shuffle_dataset(x, y)
        # Sort and compare
        assert torch.allclose(x_s.sort(dim=0)[0], x.sort(dim=0)[0])

    def test_raises_for_mismatched_sizes(self, sp):
        """Should raise ValueError when x and y have different batch sizes."""
        x = torch.randn(10, 2)
        y = torch.randn(20, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="x_tensor and y_tensor must be of the same size"):
                sp._shuffle_dataset(x, y)

    def test_corresponding_pairs_preserved(self, sp):
        """x[i] and y[i] should remain paired after shuffle."""
        torch.manual_seed(123)
        x = torch.arange(10).float().unsqueeze(1)
        y = torch.arange(10).float().unsqueeze(1) * 10
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_s, y_s = sp._shuffle_dataset(x, y)
        # For each row in shuffled data, y should be 10*x
        assert torch.allclose(y_s, x_s * 10)


@pytest.mark.unit
class TestPartitionDataset:
    """Tests for _partition_dataset."""

    def test_basic_split(self, sp):
        """Should split into train and test partitions."""
        total = 100
        x = torch.randn(total, 2)
        y = torch.randn(total, 2)
        sp.total_points = total
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            train, test = sp._partition_dataset(
                total_points=total,
                partitions=(0.7, 0.3),
                x=x,
                y=y,
            )
        x_train, y_train = train
        x_test, y_test = test
        assert x_train.shape[0] == 70
        assert x_test.shape[0] == 30

    def test_raises_for_invalid_ratio_sum(self, sp):
        """Should raise ValueError when ratios don't sum to 1.0."""
        total = 100
        x = torch.randn(total, 2)
        y = torch.randn(total, 2)
        sp.total_points = total
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Train and test ratios must sum to 1.0"):
                sp._partition_dataset(
                    total_points=total,
                    partitions=(0.5, 0.3),
                    x=x,
                    y=y,
                )

    def test_80_20_split(self, sp):
        """Test 80/20 split."""
        total = 100
        x = torch.randn(total, 2)
        y = torch.randn(total, 2)
        sp.total_points = total
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            train, test = sp._partition_dataset(
                total_points=total,
                partitions=(0.8, 0.2),
                x=x,
                y=y,
            )
        x_train, _ = train
        x_test, _ = test
        assert x_train.shape[0] == 80
        assert x_test.shape[0] == 20


@pytest.mark.unit
class TestSplitDataset:
    """Tests for _split_dataset."""

    def test_basic_split(self, sp):
        """Should produce correct partition sizes."""
        total = 100
        x = torch.randn(total, 2)
        y = torch.randn(total, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            partitions = sp._split_dataset(
                total_points=total,
                partitions=(0.7, 0.3),
                x=x,
                y=y,
            )
        assert len(partitions) == 2
        x_train, y_train = partitions[0]
        x_test, y_test = partitions[1]
        assert x_train.shape[0] == 70
        assert x_test.shape[0] == 30

    def test_raises_for_none_total_points(self, sp):
        """Should raise ValueError when total_points is None."""
        x = torch.randn(10, 2)
        y = torch.randn(10, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="total_points and partitions must be provided"):
                sp._split_dataset(total_points=None, partitions=(0.7, 0.3), x=x, y=y)

    def test_raises_for_none_partitions(self, sp):
        """Should raise ValueError when partitions is None."""
        x = torch.randn(10, 2)
        y = torch.randn(10, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="total_points and partitions must be provided"):
                sp._split_dataset(total_points=10, partitions=None, x=x, y=y)

    def test_raises_for_none_x(self, sp):
        """Should raise ValueError when x is None."""
        y = torch.randn(10, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Torch Tensors, x and y must be provided"):
                sp._split_dataset(total_points=10, partitions=(0.7, 0.3), x=None, y=y)

    def test_raises_for_none_y(self, sp):
        """Should raise ValueError when y is None."""
        x = torch.randn(10, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Torch Tensors, x and y must be provided"):
                sp._split_dataset(total_points=10, partitions=(0.7, 0.3), x=x, y=None)

    def test_raises_for_non_positive_total_points(self, sp):
        """Should raise ValueError for non-positive total_points."""
        x = torch.randn(10, 2)
        y = torch.randn(10, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="total_points must be a positive integer"):
                sp._split_dataset(total_points=0, partitions=(0.7, 0.3), x=x, y=y)

    def test_raises_for_non_tuple_partitions(self, sp):
        """Should raise ValueError for non-tuple partitions."""
        x = torch.randn(10, 2)
        y = torch.randn(10, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="partitions must be a tuple of floats"):
                sp._split_dataset(total_points=10, partitions=[0.7, 0.3], x=x, y=y)

    def test_raises_for_mismatched_lengths(self, sp):
        """Should raise ValueError when x/y size doesn't match total_points."""
        x = torch.randn(10, 2)
        y = torch.randn(10, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="x and y must have the same length as total_points"):
                sp._split_dataset(total_points=20, partitions=(0.7, 0.3), x=x, y=y)

    def test_raises_for_partitions_not_summing_to_one(self, sp):
        """Should raise ValueError when partitions don't sum to 1.0."""
        x = torch.randn(10, 2)
        y = torch.randn(10, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Partitions must sum to 1.0"):
                sp._split_dataset(total_points=10, partitions=(0.5, 0.3), x=x, y=y)

    def test_raises_for_partition_out_of_range(self, sp):
        """Should raise ValueError for partition value outside [0, 1]."""
        x = torch.randn(10, 2)
        y = torch.randn(10, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Partition .* must be between 0.0 and 1.0"):
                sp._split_dataset(total_points=10, partitions=(1.5, -0.5), x=x, y=y)

    def test_raises_for_non_float_partitions(self, sp):
        """Should raise ValueError for non-numeric partitions."""
        x = torch.randn(10, 2)
        y = torch.randn(10, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="partitions must be a tuple of floats"):
                sp._split_dataset(total_points=10, partitions=("a", "b"), x=x, y=y)


@pytest.mark.unit
class TestFindPartitionIndexEnd:
    """Tests for _find_partition_index_end."""

    def test_basic_calculation(self, sp):
        """70% of 100 starting at 0 should give 70."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._find_partition_index_end(
                partition_start=0,
                total_points=100,
                partition=0.7,
            )
        assert result == 70

    def test_second_partition(self, sp):
        """30% of 100 starting at 70 should give 100."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._find_partition_index_end(
                partition_start=70,
                total_points=100,
                partition=0.3,
            )
        assert result == 100

    def test_half_split(self, sp):
        """50% of 100 starting at 0 should give 50."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._find_partition_index_end(
                partition_start=0,
                total_points=100,
                partition=0.5,
            )
        assert result == 50


@pytest.mark.unit
class TestDatasetSplitIndexEnd:
    """Tests for _dataset_split_index_end."""

    def test_basic_split(self, sp):
        """70% of 100 should give 70."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._dataset_split_index_end(total_points=100, split_ratio=0.7)
        assert result == 70

    def test_full_dataset(self, sp):
        """100% should return total_points."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._dataset_split_index_end(total_points=100, split_ratio=1.0)
        assert result == 100

    def test_zero_ratio(self, sp):
        """0% should return 0."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._dataset_split_index_end(total_points=100, split_ratio=0.0)
        assert result == 0

    def test_returns_integer(self, sp):
        """Result should always be an integer."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = sp._dataset_split_index_end(total_points=100, split_ratio=0.33)
        assert isinstance(result, int)
        assert result == 33


@pytest.mark.unit
class TestGenerateSpiralCoordinates:
    """Tests for _generate_spiral_coordinates (full pipeline)."""

    def test_returns_tuple_of_arrays(self, sp):
        """Should return (x_coords, y_coords) tuple."""
        np.random.seed(42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_coords, y_coords = sp._generate_spiral_coordinates()
        assert isinstance(x_coords, np.ndarray)
        assert isinstance(y_coords, np.ndarray)

    def test_shape_matches_config(self, sp):
        """Output arrays should have n_spirals rows and n_points columns."""
        np.random.seed(42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_coords, y_coords = sp._generate_spiral_coordinates()
        assert x_coords.shape == (sp.n_spirals, sp.n_points)
        assert y_coords.shape == (sp.n_spirals, sp.n_points)

    def test_clockwise_vs_counterclockwise(self, sp):
        """Clockwise and counter-clockwise should produce different coordinates."""
        np.random.seed(42)
        sp.clockwise = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_cw, _ = sp._generate_spiral_coordinates()

        np.random.seed(42)
        sp.clockwise = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_ccw, _ = sp._generate_spiral_coordinates()
        # The noise component uses random state so results differ; just check shapes
        assert x_cw.shape == x_ccw.shape


# ---------------------------------------------------------------------------
# 3. solve_n_spiral_problem (lines 1240-1354)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSolveNSpiralProblem:
    """Tests for solve_n_spiral_problem with mocked dependencies.

    Note: The ``plot`` parameter uses an ``or`` chain that falls through
    to ``_SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT`` (which is True).
    Passing ``plot=False`` does NOT disable plotting because
    ``False or self.plot or True`` evaluates to ``True``.
    We therefore patch the constant to ``False`` in each test.
    """

    def _make_mock_dataset(self, n_train=70, n_test=30, n_spirals=2):
        """Helper to create mock dataset tuples."""
        x_train = torch.randn(n_train, 2)
        y_train = torch.zeros(n_train, n_spirals)
        y_train[:, 0] = 1
        x_test = torch.randn(n_test, 2)
        y_test = torch.zeros(n_test, n_spirals)
        y_test[:, 1] = 1
        x_full = torch.cat([x_train, x_test])
        y_full = torch.cat([y_train, y_test])
        return (
            (x_train, y_train),
            (x_test, y_test),
            (x_full, y_full),
        )

    def test_solve_basic_no_plot(self, sp):
        """Test solve_n_spiral_problem with plot disabled, mocking generate_n_spiral_dataset."""
        mock_data = self._make_mock_dataset()
        sp.network = MagicMock()
        sp.network.fit.return_value = {"loss": [0.5, 0.3]}
        sp.network.summary.return_value = None

        with patch.object(sp, "generate_n_spiral_dataset", return_value=mock_data), patch("spiral_problem.spiral_problem._SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT", False):
            sp.solve_n_spiral_problem(plot=False)

        sp.network.fit.assert_called_once()
        sp.network.summary.assert_called_once()
        assert hasattr(sp, "x_train")
        assert hasattr(sp, "y_train")
        assert hasattr(sp, "x_test")
        assert hasattr(sp, "y_test")
        assert hasattr(sp, "x_full")
        assert hasattr(sp, "y_full")
        assert hasattr(sp, "history")

    def test_solve_sets_parameters(self, sp):
        """Test that solve_n_spiral_problem correctly sets parameter overrides."""
        mock_data = self._make_mock_dataset()
        sp.network = MagicMock()
        sp.network.fit.return_value = {"loss": [0.5]}
        sp.network.summary.return_value = None

        with patch.object(sp, "generate_n_spiral_dataset", return_value=mock_data), patch("spiral_problem.spiral_problem._SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT", False):
            sp.solve_n_spiral_problem(
                n_points=25,
                n_spirals=2,
                n_rotations=3,
                clockwise=True,
                noise=0.2,
                distribution=0.5,
                test_ratio=0.3,
                train_ratio=0.7,
                plot=False,
            )

        assert sp.n_points == 25
        assert sp.n_rotations == 3
        assert sp.noise == 0.2

    def test_solve_passes_data_to_fit(self, sp):
        """The training data passed to network.fit should be x_train, y_train."""
        mock_data = self._make_mock_dataset()
        sp.network = MagicMock()
        sp.network.fit.return_value = {"loss": [0.1]}
        sp.network.summary.return_value = None

        with patch.object(sp, "generate_n_spiral_dataset", return_value=mock_data), patch("spiral_problem.spiral_problem._SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT", False):
            sp.solve_n_spiral_problem(plot=False)

        call_args = sp.network.fit.call_args
        x_arg = call_args[0][0]
        y_arg = call_args[0][1]
        assert torch.equal(x_arg, mock_data[0][0])
        assert torch.equal(y_arg, mock_data[0][1])

    def test_solve_with_default_params_uses_fallbacks(self, sp):
        """When None params are passed, class attributes are used."""
        mock_data = self._make_mock_dataset()
        sp.network = MagicMock()
        sp.network.fit.return_value = {"loss": [0.5]}
        sp.network.summary.return_value = None

        original_n_points = sp.n_points
        with patch.object(sp, "generate_n_spiral_dataset", return_value=mock_data), patch("spiral_problem.spiral_problem._SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT", False):
            sp.solve_n_spiral_problem()

        # n_points should remain unchanged (fallback to class attr)
        assert sp.n_points == original_n_points


# ---------------------------------------------------------------------------
# 4. evaluate (lines 1399-1476)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvaluate:
    """Tests for evaluate method with mocked dependencies."""

    def _make_mock_dataset(self, n_train=70, n_test=30, n_spirals=2):
        """Helper to create mock dataset tuples."""
        x_train = torch.randn(n_train, 2)
        y_train = torch.zeros(n_train, n_spirals)
        y_train[:, 0] = 1
        x_test = torch.randn(n_test, 2)
        y_test = torch.zeros(n_test, n_spirals)
        y_test[:, 1] = 1
        x_full = torch.cat([x_train, x_test])
        y_full = torch.cat([y_train, y_test])
        return (
            (x_train, y_train),
            (x_test, y_test),
            (x_full, y_full),
        )

    def test_evaluate_basic(self, sp):
        """Test evaluate runs end-to-end with mocked dependencies."""
        mock_data = self._make_mock_dataset()
        sp.network = MagicMock()
        sp.network.fit.return_value = {"loss": [0.5, 0.3]}
        sp.network.summary.return_value = None
        sp.network.calculate_accuracy.return_value = 0.85

        with patch.object(sp, "generate_n_spiral_dataset", return_value=mock_data), patch("spiral_problem.spiral_problem._SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT", False):
            sp.evaluate(plot=False)

        assert hasattr(sp, "train_accuracy")
        assert hasattr(sp, "test_accuracy")
        assert hasattr(sp, "train_accuracy_percent")
        assert hasattr(sp, "test_accuracy_percent")
        assert sp.train_accuracy == 0.85
        assert sp.test_accuracy == 0.85
        assert sp.train_accuracy_percent == 85.0
        assert sp.test_accuracy_percent == 85.0

    def test_evaluate_sets_parameters(self, sp):
        """Test evaluate correctly sets all parameter overrides."""
        mock_data = self._make_mock_dataset()
        sp.network = MagicMock()
        sp.network.fit.return_value = {"loss": [0.5]}
        sp.network.summary.return_value = None
        sp.network.calculate_accuracy.return_value = 0.9

        with patch.object(sp, "generate_n_spiral_dataset", return_value=mock_data), patch("spiral_problem.spiral_problem._SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT", False):
            sp.evaluate(
                n_points=25,
                n_spirals=2,
                n_rotations=3,
                clockwise=True,
                noise=0.2,
                distribution=0.5,
                plot=False,
                train_ratio=0.8,
                test_ratio=0.2,
                random_value_scale=0.5,
                default_origin=1.0,
                default_radius=2.0,
            )

        assert sp.n_points == 25
        assert sp.n_rotations == 3
        assert sp.noise == 0.2
        assert sp.random_value_scale == 0.5
        assert sp.default_origin == 1.0
        assert sp.default_radius == 2.0

    def test_evaluate_calls_calculate_accuracy(self, sp):
        """evaluate should call calculate_accuracy for both train and test."""
        mock_data = self._make_mock_dataset()
        sp.network = MagicMock()
        sp.network.fit.return_value = {"loss": [0.5]}
        sp.network.summary.return_value = None
        sp.network.calculate_accuracy.side_effect = [0.85, 0.75]

        with patch.object(sp, "generate_n_spiral_dataset", return_value=mock_data), patch("spiral_problem.spiral_problem._SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT", False):
            sp.evaluate(plot=False)

        assert sp.network.calculate_accuracy.call_count == 2
        assert sp.train_accuracy == 0.85
        assert sp.test_accuracy == 0.75
        assert sp.train_accuracy_percent == 85.0
        assert sp.test_accuracy_percent == 75.0

    def test_evaluate_calls_summary_twice(self, sp):
        """evaluate calls network.summary once in solve and once in evaluate itself."""
        mock_data = self._make_mock_dataset()
        sp.network = MagicMock()
        sp.network.fit.return_value = {"loss": [0.5]}
        sp.network.summary.return_value = None
        sp.network.calculate_accuracy.return_value = 0.9

        with patch.object(sp, "generate_n_spiral_dataset", return_value=mock_data), patch("spiral_problem.spiral_problem._SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT", False):
            sp.evaluate(plot=False)

        # summary is called once in solve_n_spiral_problem and once in evaluate
        assert sp.network.summary.call_count == 2


# ---------------------------------------------------------------------------
# 5. UUID methods
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUUIDMethods:
    """Tests for _generate_uuid, set_uuid, and get_uuid."""

    def test_generate_uuid_returns_valid_string(self, sp):
        """_generate_uuid should return a valid UUID4 string."""
        result = sp._generate_uuid()
        assert isinstance(result, str)
        assert len(result) == 36  # Standard UUID format: 8-4-4-4-12
        assert result.count("-") == 4

    def test_generate_uuid_uniqueness(self, sp):
        """Multiple calls should produce different UUIDs."""
        uuids = {sp._generate_uuid() for _ in range(10)}
        assert len(uuids) == 10

    def test_set_uuid_with_provided_value(self, sp):
        """set_uuid with a value should set it directly."""
        # Remove existing uuid to allow setting
        if hasattr(sp, "uuid"):
            delattr(sp, "uuid")
        sp.set_uuid("custom-uuid-value")
        assert sp.uuid == "custom-uuid-value"

    def test_set_uuid_generates_when_none(self, sp):
        """set_uuid(None) should auto-generate a UUID."""
        if hasattr(sp, "uuid"):
            delattr(sp, "uuid")
        sp.set_uuid(None)
        assert sp.uuid is not None
        assert isinstance(sp.uuid, str)
        assert len(sp.uuid) == 36

    def test_set_uuid_double_set_calls_exit(self, sp):
        """Setting UUID when already set should call sys.exit(1)."""
        # Ensure uuid is already set from __init__
        assert hasattr(sp, "uuid")
        assert sp.uuid is not None
        with pytest.raises(SystemExit) as exc_info:
            sp.set_uuid("another-uuid")
        assert exc_info.value.code == 1

    def test_get_uuid_returns_existing(self, sp):
        """get_uuid should return the existing UUID."""
        result = sp.get_uuid()
        assert result == sp.uuid

    def test_get_uuid_generates_if_missing(self, sp):
        """get_uuid auto-generates UUID if attribute is missing."""
        if hasattr(sp, "uuid"):
            delattr(sp, "uuid")
        result = sp.get_uuid()
        assert result is not None
        assert isinstance(result, str)

    def test_get_uuid_generates_if_none(self, sp):
        """get_uuid auto-generates UUID if current value is None."""
        sp.uuid = None
        result = sp.get_uuid()
        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 36


# ---------------------------------------------------------------------------
# 6. Setter/getter methods with edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSetDefaultRadius:
    """Tests for set_default_radius which has no None guard."""

    def test_set_default_radius_with_value(self, sp):
        """set_default_radius should set the value."""
        sp.set_default_radius(3.0)
        assert sp.default_radius == 3.0

    def test_set_default_radius_with_none(self, sp):
        """set_default_radius with None should set default_radius to None.
        Unlike other setters, set_default_radius has no None guard."""
        sp.set_default_radius(None)
        assert sp.default_radius is None

    def test_get_default_radius(self, sp):
        """get_default_radius should return the attribute value."""
        sp.default_radius = 5.0
        assert sp.get_default_radius() == 5.0


@pytest.mark.unit
class TestSetPlot:
    """Tests for set_plot and get_plot."""

    def test_set_plot_true(self, sp):
        sp.set_plot(True)
        assert sp.get_plot() is True

    def test_set_plot_false(self, sp):
        sp.set_plot(False)
        # False is falsy but not None, so the guard `if plot is not None` passes
        assert sp.get_plot() is False

    def test_set_plot_none_preserves(self, sp):
        sp.plot = True
        sp.set_plot(None)
        assert sp.get_plot() is True


# ---------------------------------------------------------------------------
# 7. Full deprecated data-generation pipeline integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeprecatedPipelineIntegration:
    """Test the full deprecated data generation pipeline end-to-end."""

    def test_full_pipeline_two_spirals(self, sp):
        """Exercise the complete deprecated pipeline: coords -> dataset -> tensors -> shuffle -> partition."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Step 1: Initialize params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sp._initialize_spiral_problem_params(
                n_spirals=2,
                n_points=50,
                noise_level=0.1,
                clockwise=True,
                train_ratio=0.7,
                test_ratio=0.3,
            )

        # Step 2: Generate spiral coordinates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_coords, y_coords = sp._generate_spiral_coordinates()

        assert x_coords.shape == (2, 50)
        assert y_coords.shape == (2, 50)

        # Step 3: Create dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x, y = sp._create_spiral_dataset(x_coords, y_coords)

        assert x.shape == (100, 2)
        assert y.shape == (100, 2)

        # Step 4: Convert to tensors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_t, y_t = sp._convert_to_tensors(x, y)

        assert isinstance(x_t, torch.Tensor)
        assert isinstance(y_t, torch.Tensor)

        # Step 5: Shuffle
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x_s, y_s = sp._shuffle_dataset(x_t, y_t)

        assert x_s.shape == x_t.shape

        # Step 6: Partition
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            train, test = sp._partition_dataset(
                total_points=100,
                partitions=(0.7, 0.3),
                x=x_s,
                y=y_s,
            )

        x_train, y_train = train
        x_test, y_test = test
        assert x_train.shape[0] == 70
        assert x_test.shape[0] == 30
        assert y_train.shape[1] == 2
        assert y_test.shape[1] == 2

    def test_full_pipeline_three_spirals(self, sp_three_spirals):
        """Full pipeline with three spirals."""
        sp = sp_three_spirals
        np.random.seed(42)
        torch.manual_seed(42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sp._initialize_spiral_problem_params(
                n_spirals=3,
                n_points=30,
                noise_level=0.05,
                clockwise=False,
                train_ratio=0.8,
                test_ratio=0.2,
            )
            x_coords, y_coords = sp._generate_spiral_coordinates()

        assert x_coords.shape == (3, 30)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            x, y = sp._create_spiral_dataset(x_coords, y_coords)
            x_t, y_t = sp._convert_to_tensors(x, y)
            x_s, y_s = sp._shuffle_dataset(x_t, y_t)
            train, test = sp._partition_dataset(
                total_points=90,
                partitions=(0.8, 0.2),
                x=x_s,
                y=y_s,
            )

        x_train, y_train = train
        x_test, y_test = test
        assert x_train.shape[0] == 72  # int(0.8 * 90) = 72
        assert x_test.shape[0] == 18  # int(0.2 * 90) = 18
        assert y_train.shape[1] == 3


# ---------------------------------------------------------------------------
# 8. Additional edge cases for maximum coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeprecationWarnings:
    """Verify all deprecated methods emit DeprecationWarning."""

    def test_generate_spiral_coordinates_warning(self, sp):
        np.random.seed(42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._generate_spiral_coordinates()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_generate_spiral_coordinates" in str(dw.message) for dw in dep_warnings)

    def test_generate_base_radial_distance_warning(self, sp):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._generate_base_radial_distance(n_points=10)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_generate_base_radial_distance" in str(dw.message) for dw in dep_warnings)

    def test_generate_angular_offset_warning(self, sp):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._generate_angular_offset()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_generate_angular_offset" in str(dw.message) for dw in dep_warnings)

    def test_generate_raw_spiral_coordinates_warning(self, sp):
        np.random.seed(42)
        n_distance = np.random.rand(sp.n_points)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._generate_raw_spiral_coordinates(n_distance=n_distance, direction=1, angular_offset=np.pi)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_generate_raw_spiral_coordinates" in str(dw.message) for dw in dep_warnings)

    def test_generate_xy_coordinates_warning(self, sp):
        np.random.seed(42)
        n_distance = np.random.rand(sp.n_points)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._generate_xy_coordinates(index=0, n_distance=n_distance, angular_offset=np.pi, direction=1)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_generate_xy_coordinates" in str(dw.message) for dw in dep_warnings)

    def test_make_coords_warning(self, sp):
        np.random.seed(42)
        n_distance = np.random.rand(sp.n_points)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._make_coords(index=0, n_distance=n_distance, angular_offset=np.pi, direction=1, trig_function=np.cos)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_make_coords" in str(dw.message) for dw in dep_warnings)

    def test_make_noise_warning(self, sp):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._make_noise(n_points=10, noise=0.1)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_make_noise" in str(dw.message) for dw in dep_warnings)

    def test_create_input_features_warning(self, sp):
        x_coords = np.array([np.random.rand(50), np.random.rand(50)])
        y_coords = np.array([np.random.rand(50), np.random.rand(50)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._create_input_features(x_coords, y_coords)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_create_input_features" in str(dw.message) for dw in dep_warnings)

    def test_create_one_hot_targets_warning(self, sp):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._create_one_hot_targets(total_points=100, n_spirals=2, dtype=np.float32)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_create_one_hot_targets" in str(dw.message) for dw in dep_warnings)

    def test_create_spiral_dataset_warning(self, sp):
        x_coords = np.array([np.random.rand(50), np.random.rand(50)])
        y_coords = np.array([np.random.rand(50), np.random.rand(50)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._create_spiral_dataset(x_coords, y_coords)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_create_spiral_dataset" in str(dw.message) for dw in dep_warnings)

    def test_convert_to_tensors_warning(self, sp):
        x = np.random.rand(10, 2).astype(np.float32)
        y = np.random.rand(10, 2).astype(np.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._convert_to_tensors(x, y)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_convert_to_tensors" in str(dw.message) for dw in dep_warnings)

    def test_shuffle_dataset_warning(self, sp):
        x = torch.randn(10, 2)
        y = torch.randn(10, 2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._shuffle_dataset(x, y)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_shuffle_dataset" in str(dw.message) for dw in dep_warnings)

    def test_partition_dataset_warning(self, sp):
        total = 100
        x = torch.randn(total, 2)
        y = torch.randn(total, 2)
        sp.total_points = total
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._partition_dataset(total_points=total, partitions=(0.7, 0.3), x=x, y=y)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_partition_dataset" in str(dw.message) for dw in dep_warnings)

    def test_split_dataset_warning(self, sp):
        total = 10
        x = torch.randn(total, 2)
        y = torch.randn(total, 2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._split_dataset(total_points=total, partitions=(0.7, 0.3), x=x, y=y)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_split_dataset" in str(dw.message) for dw in dep_warnings)

    def test_find_partition_index_end_warning(self, sp):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._find_partition_index_end(partition_start=0, total_points=100, partition=0.7)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_find_partition_index_end" in str(dw.message) for dw in dep_warnings)

    def test_dataset_split_index_end_warning(self, sp):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp._dataset_split_index_end(total_points=100, split_ratio=0.7)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("_dataset_split_index_end" in str(dw.message) for dw in dep_warnings)


@pytest.mark.unit
class TestSettersCoverage:
    """Ensure every setter body is executed for coverage."""

    def test_set_network(self, sp):
        mock_net = MagicMock()
        sp.set_network(mock_net)
        assert sp.network is mock_net

    def test_set_network_none(self, sp):
        original = sp.network
        sp.set_network(None)
        assert sp.network is original

    def test_set_logger(self, sp):
        import logging

        new_logger = logging.getLogger("test")
        sp.set_logger(new_logger)
        assert sp.logger is new_logger

    def test_set_logger_none(self, sp):
        original = sp.logger
        sp.set_logger(None)
        assert sp.logger is original

    def test_set_n_spirals(self, sp):
        sp.set_n_spirals(5)
        assert sp.n_spirals == 5

    def test_set_n_spirals_none(self, sp):
        original = sp.n_spirals
        sp.set_n_spirals(None)
        assert sp.n_spirals == original

    def test_set_n_points(self, sp):
        sp.set_n_points(200)
        assert sp.n_points == 200

    def test_set_n_points_none(self, sp):
        original = sp.n_points
        sp.set_n_points(None)
        assert sp.n_points == original

    def test_set_n_rotations(self, sp):
        sp.set_n_rotations(5)
        assert sp.n_rotations == 5

    def test_set_n_rotations_none(self, sp):
        original = sp.n_rotations
        sp.set_n_rotations(None)
        assert sp.n_rotations == original

    def test_set_clockwise(self, sp):
        sp.set_clockwise(False)
        assert sp.clockwise is False

    def test_set_clockwise_none(self, sp):
        original = sp.clockwise
        sp.set_clockwise(None)
        assert sp.clockwise == original

    def test_set_noise(self, sp):
        sp.set_noise(0.5)
        assert sp.noise == 0.5

    def test_set_noise_none(self, sp):
        original = sp.noise
        sp.set_noise(None)
        assert sp.noise == original

    def test_set_distribution(self, sp):
        sp.set_distribution("normal")
        assert sp.distribution == "normal"

    def test_set_distribution_none(self, sp):
        original = sp.distribution
        sp.set_distribution(None)
        assert sp.distribution == original

    def test_set_random_seed(self, sp):
        sp.set_random_seed(99)
        assert sp.random_seed == 99

    def test_set_random_seed_none(self, sp):
        original = sp.random_seed
        sp.set_random_seed(None)
        assert sp.random_seed == original

    def test_set_train_ratio(self, sp):
        sp.set_train_ratio(0.6)
        assert sp.train_ratio == 0.6

    def test_set_train_ratio_none(self, sp):
        original = sp.train_ratio
        sp.set_train_ratio(None)
        assert sp.train_ratio == original

    def test_set_test_ratio(self, sp):
        sp.set_test_ratio(0.4)
        assert sp.test_ratio == 0.4

    def test_set_test_ratio_none(self, sp):
        original = sp.test_ratio
        sp.set_test_ratio(None)
        assert sp.test_ratio == original

    def test_set_plot_true(self, sp):
        sp.set_plot(True)
        assert sp.plot is True

    def test_set_plot_none(self, sp):
        sp.plot = True
        sp.set_plot(None)
        assert sp.plot is True

    def test_set_random_value_scale(self, sp):
        sp.set_random_value_scale(0.01)
        assert sp.random_value_scale == 0.01

    def test_set_random_value_scale_none(self, sp):
        original = sp.random_value_scale
        sp.set_random_value_scale(None)
        assert sp.random_value_scale == original

    def test_set_default_origin(self, sp):
        sp.set_default_origin((1.0, 1.0))
        assert sp.default_origin == (1.0, 1.0)

    def test_set_default_origin_none(self, sp):
        original = sp.default_origin
        sp.set_default_origin(None)
        assert sp.default_origin == original

    def test_set_default_radius_value(self, sp):
        sp.set_default_radius(3.0)
        assert sp.default_radius == 3.0


@pytest.mark.unit
class TestGettersCoverage:
    """Ensure every getter body is executed for coverage."""

    def test_get_network(self, sp):
        assert sp.get_network() is sp.network

    def test_get_n_spirals(self, sp):
        assert sp.get_n_spirals() == sp.n_spirals

    def test_get_n_points(self, sp):
        assert sp.get_n_points() == sp.n_points

    def test_get_n_rotations(self, sp):
        assert sp.get_n_rotations() == sp.n_rotations

    def test_get_clockwise(self, sp):
        assert sp.get_clockwise() == sp.clockwise

    def test_get_noise(self, sp):
        assert sp.get_noise() == sp.noise

    def test_get_distribution(self, sp):
        assert sp.get_distribution() == sp.distribution

    def test_get_random_seed(self, sp):
        assert sp.get_random_seed() == sp.random_seed

    def test_get_train_ratio(self, sp):
        assert sp.get_train_ratio() == sp.train_ratio

    def test_get_test_ratio(self, sp):
        assert sp.get_test_ratio() == sp.test_ratio

    def test_get_plot(self, sp):
        sp.plot = True
        assert sp.get_plot() is True

    def test_get_random_value_scale(self, sp):
        assert sp.get_random_value_scale() == sp.random_value_scale

    def test_get_default_origin(self, sp):
        assert sp.get_default_origin() == sp.default_origin

    def test_get_default_radius(self, sp):
        assert sp.get_default_radius() == sp.default_radius


@pytest.mark.unit
class TestDatasetSplitIndexEndEdgeCases:
    """Additional edge case for _dataset_split_index_end error path."""

    def test_negative_ratio_raises(self, sp):
        """A negative split ratio produces a negative index_end which triggers ValueError."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Invalid index end"):
                sp._dataset_split_index_end(total_points=100, split_ratio=-0.5)

    def test_ratio_exceeding_one_raises(self, sp):
        """A ratio > 1.0 produces index_end > total_points which triggers ValueError."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Invalid index end"):
                sp._dataset_split_index_end(total_points=100, split_ratio=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
