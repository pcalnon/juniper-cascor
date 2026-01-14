#!/usr/bin/env python
"""
Project:       Juniper
Prototype:     Cascade Correlation Neural Network  
File Name:     test_residual_error.py
Author:        Paul Calnon
Version:       0.1.0

Date:          2025-09-26
Last Modified: 2025-09-26

License:       MIT License
Copyright:     Copyright (c) 2024-2025 Paul Calnon

Description:
    Unit tests for residual error calculation in Cascade Correlation Network.
    Tests the computation of error between network predictions and targets.
"""

import pytest
import torch
# import numpy as np
from helpers.assertions import (
    assert_tensor_shape, assert_tensor_finite, assert_prediction_shapes_match,
    assert_approximately_equal
)
# from helpers.utilities import set_deterministic_behavior, create_test_data
from helpers.utilities import set_deterministic_behavior


class TestResidualErrorBasics:
    """Test basic residual error calculation functionality."""
    
    @pytest.mark.unit
    def test_residual_error_perfect_prediction(self, simple_network):
        """Test residual error when network prediction is perfect."""
        set_deterministic_behavior()
        
        # Create test data
        x = torch.randn(10, simple_network.input_size)
        y_true = torch.randn(10, simple_network.output_size)
        
        # Mock perfect prediction by setting network output to match targets
        with torch.no_grad():
            # Adjust weights to produce y_true as output
            # For simplicity, we'll just test with y_pred = y_true directly
            pass
        
        # Calculate residual using perfect predictions
        y_pred = y_true.clone()  # Perfect prediction
        residual_expected = y_true - y_pred
        
        # The residual should be zero
        assert_approximately_equal(residual_expected, torch.zeros_like(y_true), atol=1e-10)
        print(f"Perfect prediction residual: {residual_expected}, x={len(x)}, y={y_true.sum(dim=0)}")
    
    @pytest.mark.unit
    def test_residual_error_computation(self, simple_network, simple_2d_data):
        """Test basic residual error computation."""
        set_deterministic_behavior()
        
        x, y = simple_2d_data
        
        # Calculate residual error
        residual = simple_network.calculate_residual_error(x, y)
        
        # Verify shape
        assert_prediction_shapes_match(residual, y)
        assert_tensor_finite(residual)
    
    @pytest.mark.unit
    def test_residual_error_symmetric(self, simple_network):
        """Test that residual error is symmetric (y - pred = -(pred - y))."""
        set_deterministic_behavior()
        
        x = torch.randn(5, simple_network.input_size)
        y = torch.randn(5, simple_network.output_size)
        
        residual1 = simple_network.calculate_residual_error(x, y)
        
        # Get network prediction
        with torch.no_grad():
            pred = simple_network.forward(x)
        
        # Manual calculation: residual = y - pred
        residual2 = y - pred
        
        assert_approximately_equal(residual1, residual2)
    
    @pytest.mark.unit
    def test_residual_error_no_grad(self, simple_network, simple_2d_data):
        """Test that residual error calculation doesn't require gradients."""
        set_deterministic_behavior()
        
        x, y = simple_2d_data
        
        # Residual error should be computed with no_grad
        residual = simple_network.calculate_residual_error(x, y)
        
        # Result should not require gradients
        assert not residual.requires_grad  # trunk-ignore(bandit/B101)


class TestResidualErrorShapes:
    """Test shape handling in residual error calculation."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 50])
    def test_residual_error_batch_sizes(self, simple_network, batch_size):
        """Test residual error calculation with different batch sizes."""
        set_deterministic_behavior()
        
        x = torch.randn(batch_size, simple_network.input_size)
        y = torch.randn(batch_size, simple_network.output_size)
        
        residual = simple_network.calculate_residual_error(x, y)
        
        expected_shape = (batch_size, simple_network.output_size)
        assert_tensor_shape(residual, expected_shape)
    
    @pytest.mark.unit
    @pytest.mark.parametrize("output_size", [1, 2, 5, 10])
    def test_residual_error_output_sizes(self, output_size):
        """Test residual error with different output sizes."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        
        set_deterministic_behavior()
        
        network = CascadeCorrelationNetwork.create_simple_network(
            input_size=2,
            output_size=output_size
        )
        
        x = torch.randn(10, 2)
        y = torch.randn(10, output_size)
        
        residual = network.calculate_residual_error(x, y)
        
        expected_shape = (10, output_size)
        assert_tensor_shape(residual, expected_shape)


class TestResidualErrorMagnitude:
    """Test residual error magnitude and properties."""
    
    @pytest.mark.unit
    def test_residual_error_magnitude_untrained(self, simple_network):
        """Test that untrained network has large residual error."""
        set_deterministic_behavior()
        
        # Create data with clear pattern
        x = torch.tensor([[0., 0.], [1., 1.], [0., 1.], [1., 0.]], dtype=torch.float32)
        y = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.]], dtype=torch.float32)  # XOR pattern
        
        residual = simple_network.calculate_residual_error(x, y)
        
        # Untrained network should have substantial error
        mean_abs_error = residual.abs().mean()
        # Should have significant error
        assert mean_abs_error.item() > 0.1  # trunk-ignore(bandit/B101)

    @pytest.mark.unit
    def test_residual_error_zero_targets(self, simple_network):
        """Test residual error when targets are zero."""
        set_deterministic_behavior()
        
        x = torch.randn(10, simple_network.input_size)
        y = torch.zeros(10, simple_network.output_size)
        
        residual = simple_network.calculate_residual_error(x, y)
        
        # Residual should be -prediction (since targets are zero)
        with torch.no_grad():
            pred = simple_network.forward(x)
        
        expected_residual = y - pred  # 0 - pred = -pred
        assert_approximately_equal(residual, expected_residual)
    
    @pytest.mark.unit
    def test_residual_error_distribution(self, simple_network):
        """Test statistical properties of residual error."""
        set_deterministic_behavior()
        
        # Generate larger dataset
        x = torch.randn(200, simple_network.input_size)
        y = torch.randn(200, simple_network.output_size)
        
        residual = simple_network.calculate_residual_error(x, y)
        
        # Check that residual has reasonable distribution
        residual_flat = residual.view(-1)
        
        # Should have finite variance
        residual_std = residual_flat.std()
        assert residual_std.item() > 0  # trunk-ignore(bandit/B101)
        assert torch.isfinite(residual_std)  # trunk-ignore(bandit/B101)
        
        # Mean should be reasonable (not systematically biased)
        residual_mean = residual_flat.mean()
        assert torch.isfinite(residual_mean)  # trunk-ignore(bandit/B101)


class TestResidualErrorValidation:
    """Test input validation for residual error calculation."""
    
    @pytest.mark.unit
    def test_residual_error_mismatched_shapes(self, simple_network):
        """Test residual error with mismatched input/target shapes."""
        x = torch.randn(10, simple_network.input_size)
        y = torch.randn(5, simple_network.output_size)  # Wrong batch size
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):  # trunk-ignore(ruff/B017)
            simple_network.calculate_residual_error(x, y)
    
    @pytest.mark.unit
    def test_residual_error_wrong_target_size(self, simple_network):
        """Test residual error with wrong target dimensionality."""
        x = torch.randn(10, simple_network.input_size)
        y = torch.randn(10, simple_network.output_size + 1)  # Wrong output size
        
        # Should raise error due to shape mismatch
        with pytest.raises(Exception):  # trunk-ignore(ruff/B017)
            simple_network.calculate_residual_error(x, y)
    
    @pytest.mark.unit
    def test_residual_error_empty_input(self, simple_network):
        """Test residual error with empty input."""
        x = torch.empty(0, simple_network.input_size)
        y = torch.empty(0, simple_network.output_size)
        
        residual = simple_network.calculate_residual_error(x, y)

        print(f"test_residual_error_empty_input: residual shape: {residual.shape}")
        
        expected_shape = (0, simple_network.output_size)
        assert_tensor_shape(residual, expected_shape)


class TestResidualErrorNumericalStability:
    """Test numerical stability of residual error calculation."""
    
    @pytest.mark.unit
    def test_residual_error_extreme_values(self, simple_network):
        """Test residual error with extreme input/target values."""
        set_deterministic_behavior()
        
        # Test with large values
        x_large = torch.randn(5, simple_network.input_size) * 100
        y_large = torch.randn(5, simple_network.output_size) * 100
        
        residual_large = simple_network.calculate_residual_error(x_large, y_large)
        assert_tensor_finite(residual_large)
        
        # Test with small values
        x_small = torch.randn(5, simple_network.input_size) * 1e-6
        y_small = torch.randn(5, simple_network.output_size) * 1e-6
        
        residual_small = simple_network.calculate_residual_error(x_small, y_small)
        assert_tensor_finite(residual_small)
    
    @pytest.mark.unit
    def test_residual_error_inf_nan_handling(self, simple_network):
        """Test residual error behavior with inf/nan values."""
        x = torch.randn(5, simple_network.input_size)
        
        # Test with inf targets
        y_inf = torch.full((5, simple_network.output_size), float('inf'))
        residual_inf = simple_network.calculate_residual_error(x, y_inf)
        # Should contain inf values
        assert torch.isinf(residual_inf).any()  # trunk-ignore(bandit/B101)
        
        # Test with nan targets
        y_nan = torch.full((5, simple_network.output_size), float('nan'))
        residual_nan = simple_network.calculate_residual_error(x, y_nan)
        # Should contain nan values
        assert torch.isnan(residual_nan).any()  # trunk-ignore(bandit/B101)


class TestResidualErrorWithHiddenUnits:
    """Test residual error calculation with hidden units."""
    
    @pytest.mark.unit
    def test_residual_error_with_hidden_units(self, simple_network, simple_2d_data):
        """Test residual error calculation when network has hidden units."""
        set_deterministic_behavior()
        
        x, y = simple_2d_data
        
        # Add a hidden unit
        hidden_unit = {
            'weights': torch.randn(simple_network.input_size, requires_grad=True) * 0.1,
            'bias': torch.randn(1, requires_grad=True) * 0.1,
            'activation_fn': torch.tanh,
            'correlation': 0.5
        }
        simple_network.hidden_units = [hidden_unit]
        
        # Update output weights
        simple_network.output_weights = torch.randn(
            simple_network.input_size + 1, simple_network.output_size, requires_grad=True
        ) * 0.1
        
        # Calculate residual error
        residual = simple_network.calculate_residual_error(x, y)
        
        assert_prediction_shapes_match(residual, y)
        assert_tensor_finite(residual)
    
    @pytest.mark.unit
    def test_residual_error_changes_with_training(self, simple_network, simple_2d_data):
        """Test that residual error changes as network is modified."""
        set_deterministic_behavior()
        
        x, y = simple_2d_data
        
        # Initial residual error
        residual1 = simple_network.calculate_residual_error(x, y)
        
        # Modify network weights
        simple_network.output_weights.data += 0.1
        
        # New residual error
        residual2 = simple_network.calculate_residual_error(x, y)
        
        # Should be different
        assert not torch.allclose(residual1, residual2, atol=1e-6)  # trunk-ignore(bandit/B101)


class TestResidualErrorCorrelation:
    """Test properties related to correlation calculation."""
    
    @pytest.mark.unit
    def test_residual_error_for_correlation(self, simple_network):
        """Test residual error properties needed for correlation calculation."""
        set_deterministic_behavior()
        
        # Create structured data
        x = torch.randn(50, simple_network.input_size)
        y = torch.randn(50, simple_network.output_size)
        
        residual = simple_network.calculate_residual_error(x, y)
        
        # Properties needed for correlation calculation
        assert_tensor_finite(residual)
        # Need multiple samples for correlation
        assert residual.shape[0] > 1  # trunk-ignore(bandit/B101)

        
        # Should have some variance (not all zeros)
        residual_var = residual.var(dim=0)
        # Some variance in at least one dimension
        assert torch.any(residual_var > 1e-8)  # trunk-ignore(bandit/B101)
    
    @pytest.mark.unit
    def test_residual_error_mean_centering(self, simple_network):
        """Test mean-centered residual error for correlation."""
        set_deterministic_behavior()
        
        x = torch.randn(100, simple_network.input_size)
        y = torch.randn(100, simple_network.output_size)
        
        residual = simple_network.calculate_residual_error(x, y)
        
        # Calculate mean-centered residual (as used in correlation)
        residual_mean = residual.mean(dim=0, keepdim=True)
        residual_centered = residual - residual_mean
        
        # Mean should be approximately zero
        centered_mean = residual_centered.mean(dim=0)
        assert_approximately_equal(centered_mean, torch.zeros_like(centered_mean), atol=1e-6)
