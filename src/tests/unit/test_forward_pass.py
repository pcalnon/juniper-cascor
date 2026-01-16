#!/usr/bin/env python
"""
Project:       Juniper
Prototype:     Cascade Correlation Neural Network  
File Name:     test_forward_pass.py
Author:        Paul Calnon
Version:       0.1.0

Date:          2025-09-26
Last Modified: 2025-09-26

License:       MIT License
Copyright:     Copyright (c) 2024-2025 Paul Calnon

Description:
    Unit tests for forward pass algorithm in Cascade Correlation Network.
    Tests the core network prediction functionality with varying hidden unit counts.
"""

import pytest
import torch
# import numpy as np
from helpers.assertions import (
    assert_tensor_shape,
    assert_tensor_finite,
    # assert_network_structure_valid,
    # assert_prediction_shapes_match,
    assert_approximately_equal,
)
from helpers.utilities import (
    set_deterministic_behavior,
    # reset_network_state,
    verify_network_forward_pass
)


class TestForwardPassBasics:
    """Test basic forward pass functionality."""
    
    @pytest.mark.unit
    def test_forward_pass_no_hidden_units(self, simple_network, valid_tensor_2d):
        """Test forward pass with no hidden units (direct input to output)."""
        set_deterministic_behavior()
        
        # Ensure no hidden units
        simple_network.hidden_units = []
        
        # Test forward pass
        output = simple_network.forward(valid_tensor_2d)
        
        # Verify output shape
        expected_shape = (valid_tensor_2d.shape[0], simple_network.output_size)
        assert_tensor_shape(output, expected_shape)
        assert_tensor_finite(output)
    
    @pytest.mark.unit
    def test_forward_pass_single_hidden_unit(self, simple_network, valid_tensor_2d):
        """Test forward pass with one hidden unit."""
        set_deterministic_behavior()
        
        # Add a single hidden unit manually
        hidden_unit = {
            'weights': torch.randn(simple_network.input_size, requires_grad=True) * 0.1,
            'bias': torch.randn(1, requires_grad=True) * 0.1,
            'activation_fn': torch.tanh,
            'correlation': 0.5
        }
        simple_network.hidden_units = [hidden_unit]
        
        # Update output weights to match new input size
        new_input_size = simple_network.input_size + 1  # +1 for hidden unit
        simple_network.output_weights = torch.randn(
            new_input_size, simple_network.output_size, requires_grad=True
        ) * 0.1
        
        # Test forward pass
        output = simple_network.forward(valid_tensor_2d)
        
        # Verify output shape
        expected_shape = (valid_tensor_2d.shape[0], simple_network.output_size)
        assert_tensor_shape(output, expected_shape)
        assert_tensor_finite(output)
    
    @pytest.mark.unit
    def test_forward_pass_multiple_hidden_units(self, simple_network, valid_tensor_2d):
        """Test forward pass with multiple hidden units."""
        set_deterministic_behavior()
        n_hidden = 3
        
        # Add multiple hidden units
        for i in range(n_hidden):  # sourcery skip: no-loop-in-tests
            input_size = simple_network.input_size + i  # Cascading input size
            hidden_unit = {
                'weights': torch.randn(input_size, requires_grad=True) * 0.1,
                'bias': torch.randn(1, requires_grad=True) * 0.1,
                'activation_fn': torch.tanh,
                'correlation': 0.5 + i * 0.1
            }
            simple_network.hidden_units.append(hidden_unit)
        
        # Update output weights
        final_input_size = simple_network.input_size + n_hidden
        simple_network.output_weights = torch.randn(
            final_input_size, simple_network.output_size, requires_grad=True
        ) * 0.1
        
        # Test forward pass
        output = simple_network.forward(valid_tensor_2d)
        
        # Verify output shape
        expected_shape = (valid_tensor_2d.shape[0], simple_network.output_size)
        assert_tensor_shape(output, expected_shape)
        assert_tensor_finite(output)
    
    @pytest.mark.unit
    def test_forward_pass_deterministic(self, simple_network, valid_tensor_2d):
        """Test that forward pass is deterministic."""
        set_deterministic_behavior(42)
        
        # First forward pass
        output1 = simple_network.forward(valid_tensor_2d)
        
        # Second forward pass with same input
        output2 = simple_network.forward(valid_tensor_2d)
        
        # Should be identical
        assert_approximately_equal(output1, output2, rtol=1e-8, atol=1e-10)


class TestForwardPassShapes:
    """Test shape handling in forward pass."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 50])
    def test_forward_pass_batch_sizes(self, simple_network, batch_size):
        """Test forward pass with different batch sizes."""
        set_deterministic_behavior()
        
        x = torch.randn(batch_size, simple_network.input_size)
        output = simple_network.forward(x)
        
        expected_shape = (batch_size, simple_network.output_size)
        assert_tensor_shape(output, expected_shape)
    
    @pytest.mark.unit
    @pytest.mark.parametrize("input_size", [1, 2, 5, 10])
    def test_forward_pass_input_sizes(self, input_size):
        """Test forward pass with different input sizes."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        
        set_deterministic_behavior()
        
        network = CascadeCorrelationNetwork.create_simple_network(
            input_size=input_size,
            output_size=2
        )
        
        x = torch.randn(10, input_size)
        output = network.forward(x)
        
        expected_shape = (10, 2)
        assert_tensor_shape(output, expected_shape)
    
    @pytest.mark.unit
    @pytest.mark.parametrize("output_size", [1, 2, 5, 10])
    def test_forward_pass_output_sizes(self, output_size):
        """Test forward pass with different output sizes."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        
        set_deterministic_behavior()
        
        network = CascadeCorrelationNetwork.create_simple_network(
            input_size=2,
            output_size=output_size
        )
        
        x = torch.randn(10, 2)
        output = network.forward(x)
        
        expected_shape = (10, output_size)
        assert_tensor_shape(output, expected_shape)


class TestForwardPassValidation:
    """Test input validation in forward pass."""
    
    @pytest.mark.unit
    def test_forward_pass_invalid_input_shape(self, simple_network):
        """Test forward pass with wrong input shape."""
        # Wrong number of features
        wrong_input = torch.randn(10, simple_network.input_size + 1)
        
        # Should raise validation error
        with pytest.raises(Exception):  # trunk-ignore(ruff/B017)
            simple_network.forward(wrong_input)
    
    @pytest.mark.unit
    def test_forward_pass_1d_input(self, simple_network):
        """Test forward pass with 1D input (should fail)."""
        # 1D input (should be 2D)
        wrong_input = torch.randn(simple_network.input_size)

        with pytest.raises(Exception):  # trunk-ignore(ruff/B017)
            simple_network.forward(wrong_input)
    
    @pytest.mark.unit
    def test_forward_pass_empty_input(self, simple_network):
        """Test forward pass with empty input."""
        empty_input = torch.empty(0, simple_network.input_size)
        
        output = simple_network.forward(empty_input)
        
        # Should handle gracefully
        expected_shape = (0, simple_network.output_size)
        assert_tensor_shape(output, expected_shape)
    
    @pytest.mark.unit
    def test_forward_pass_nan_input(self, simple_network):
        """Test forward pass with NaN input raises ValidationError.
        
        The network validates inputs and rejects NaN values to prevent
        silent propagation of invalid data through the network.
        """
        from cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        nan_input = torch.full((5, simple_network.input_size), float('nan'))
        
        # Network should reject NaN inputs with a ValidationError
        with pytest.raises(ValidationError, match="contains NaN values"):
            simple_network.forward(nan_input)


class TestForwardPassCascading:
    """Test cascading behavior specific to cascade correlation."""
    
    @pytest.mark.unit
    def test_cascading_connections(self, simple_network, valid_tensor_2d):
        """Test that hidden units receive cascading inputs."""
        set_deterministic_behavior()
        
        # Add two hidden units with specific input sizes
        unit1 = {
            'weights': torch.randn(simple_network.input_size, requires_grad=True) * 0.1,
            'bias': torch.randn(1, requires_grad=True) * 0.1,
            'activation_fn': torch.tanh,
            'correlation': 0.6
        }
        unit2 = {
            'weights': torch.randn(simple_network.input_size + 1, requires_grad=True) * 0.1,  # +1 for unit1 output
            'bias': torch.randn(1, requires_grad=True) * 0.1,
            'activation_fn': torch.tanh,
            'correlation': 0.7
        }
        
        simple_network.hidden_units = [unit1, unit2]
        simple_network.output_weights = torch.randn(
            simple_network.input_size + 2, simple_network.output_size, requires_grad=True
        ) * 0.1
        
        # Get forward pass diagnostic info
        info = verify_network_forward_pass(simple_network, valid_tensor_2d)
        
        assert info['success'], f"Forward pass failed: {info['error']}"  # trunk-ignore(bandit/B101)

        # Verify cascading input sizes
        assert info['hidden_outputs'][0]['input_shape'][1] == simple_network.input_size  # trunk-ignore(bandit/B101)
        assert info['hidden_outputs'][1]['input_shape'][1] == simple_network.input_size + 1  # trunk-ignore(bandit/B101)

    @pytest.mark.unit
    def test_hidden_unit_output_shapes(self, simple_network, valid_tensor_2d):
        """Test that hidden unit outputs have correct shapes."""
        set_deterministic_behavior()
        
        # Add hidden unit
        hidden_unit = {
            'weights': torch.randn(simple_network.input_size, requires_grad=True) * 0.1,
            'bias': torch.randn(1, requires_grad=True) * 0.1,
            'activation_fn': torch.tanh,
            'correlation': 0.5
        }
        simple_network.hidden_units = [hidden_unit]
        simple_network.output_weights = torch.randn(
            simple_network.input_size + 1, simple_network.output_size, requires_grad=True
        ) * 0.1
        
        info = verify_network_forward_pass(simple_network, valid_tensor_2d)

        assert info['success'], f"Forward pass failed: {info['error']}"  # trunk-ignore(bandit/B101)

        # Hidden unit output should be (batch_size, 1)
        expected_hidden_shape = (valid_tensor_2d.shape[0], 1)
        assert info['hidden_outputs'][0]['output_shape'] == expected_hidden_shape  # trunk-ignore(bandit/B101)


class TestForwardPassGradients:
    """Test gradient computation in forward pass."""
    
    @pytest.mark.unit
    def test_forward_pass_gradients_enabled(self, simple_network, valid_tensor_2d):
        """Test that forward pass preserves gradient computation."""
        set_deterministic_behavior()
        
        # Enable gradients for input
        valid_tensor_2d.requires_grad_(True)
        
        output = simple_network.forward(valid_tensor_2d)
        
        # Should be able to compute gradients
        loss = output.sum()
        loss.backward()

        assert valid_tensor_2d.grad is not None  # trunk-ignore(bandit/B101)
        assert_tensor_finite(valid_tensor_2d.grad)
    
    @pytest.mark.unit 
    def test_forward_pass_no_grad(self, simple_network, valid_tensor_2d):
        """Test forward pass with gradients disabled."""
        set_deterministic_behavior()
        
        with torch.no_grad():
            output = simple_network.forward(valid_tensor_2d)
        
        # Output should not require gradients
        assert not output.requires_grad  # trunk-ignore(bandit/B101)


class TestForwardPassActivationFunctions:
    """Test different activation functions in forward pass."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("activation_fn", [torch.tanh, torch.sigmoid, torch.relu])
    def test_forward_pass_different_activations(self, simple_network, valid_tensor_2d, activation_fn):
        """Test forward pass with different activation functions."""
        set_deterministic_behavior()
        
        # Add hidden unit with specific activation
        hidden_unit = {
            'weights': torch.randn(simple_network.input_size, requires_grad=True) * 0.1,
            'bias': torch.randn(1, requires_grad=True) * 0.1,
            'activation_fn': activation_fn,
            'correlation': 0.5
        }
        simple_network.hidden_units = [hidden_unit]
        simple_network.output_weights = torch.randn(
            simple_network.input_size + 1, simple_network.output_size, requires_grad=True
        ) * 0.1
        
        output = simple_network.forward(valid_tensor_2d)
        
        # Should produce valid output regardless of activation
        assert_tensor_finite(output)
        expected_shape = (valid_tensor_2d.shape[0], simple_network.output_size)
        assert_tensor_shape(output, expected_shape)


class TestForwardPassEdgeCases:
    """Test edge cases in forward pass."""
    
    @pytest.mark.unit
    def test_forward_pass_single_sample(self, simple_network):
        """Test forward pass with single sample."""
        set_deterministic_behavior()
        
        x = torch.randn(1, simple_network.input_size)
        output = simple_network.forward(x)
        
        expected_shape = (1, simple_network.output_size)
        assert_tensor_shape(output, expected_shape)
    
    @pytest.mark.unit
    def test_forward_pass_large_batch(self, simple_network):
        """Test forward pass with large batch."""
        set_deterministic_behavior()
        
        x = torch.randn(1000, simple_network.input_size)
        output = simple_network.forward(x)
        
        expected_shape = (1000, simple_network.output_size)
        assert_tensor_shape(output, expected_shape)
        assert_tensor_finite(output)
    
    @pytest.mark.unit
    def test_forward_pass_extreme_weights(self, simple_network, valid_tensor_2d):
        """Test forward pass with extreme weight values."""
        set_deterministic_behavior()
        
        # Set very large output weights
        simple_network.output_weights.data.fill_(10.0)
        simple_network.output_bias.data.fill_(5.0)
        
        output = simple_network.forward(valid_tensor_2d)
        
        # Should still produce valid output (might be saturated)
        assert_tensor_finite(output)
        expected_shape = (valid_tensor_2d.shape[0], simple_network.output_size)
        assert_tensor_shape(output, expected_shape)
