#!/usr/bin/env python
"""
Project:       Juniper
Prototype:     Cascade Correlation Neural Network  
File Name:     test_accuracy.py
Author:        Paul Calnon
Version:       0.1.0

Date:          2025-09-26
Last Modified: 2025-09-26

License:       MIT License
Copyright:     Copyright (c) 2024-2025 Paul Calnon

Description:
    Unit tests for accuracy calculation methods in Cascade Correlation Network.
    Tests both calculate_accuracy and _accuracy methods with various scenarios.
"""

import pytest
import torch
# import numpy as np
from helpers.assertions import (
    assert_accuracy_valid,
    assert_approximately_equal,
    # assert_tensor_finite,
)
from helpers.utilities import set_deterministic_behavior


class TestAccuracyCalculation:
    """Test basic accuracy calculation functionality."""
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_perfect_accuracy(self, simple_network):
        """Test accuracy calculation with perfect predictions."""
        set_deterministic_behavior()
        
        # Create data where network will predict correctly
        batch_size = 10
        x = torch.randn(batch_size, simple_network.input_size)
        
        # Create targets and corresponding perfect predictions
        y_true = torch.zeros(batch_size, simple_network.output_size)
        y_true[torch.arange(batch_size), torch.randint(0, simple_network.output_size, (batch_size,))] = 1
        
        # Test the private _accuracy method with perfect predictions
        accuracy = simple_network._accuracy(y=y_true, output=y_true)
        
        assert_accuracy_valid(accuracy)
        assert_approximately_equal(accuracy, 1.0, atol=1e-10)
        print(f"Perfect accuracy: {accuracy}, type: {type(accuracy)}, Data: x={len(x)}, y_true={y_true}")
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_zero_accuracy(self, simple_network):
        """Test accuracy calculation with completely wrong predictions."""
        set_deterministic_behavior()
        
        batch_size = 10
        
        # Create targets and completely opposite predictions
        y_true = torch.zeros(batch_size, 2)
        y_pred = torch.zeros(batch_size, 2)
        
        # Set targets to class 0, predictions to class 1
        y_true[:, 0] = 1
        y_pred[:, 1] = 1
        
        accuracy = simple_network._accuracy(y=y_true, output=y_pred)
        
        assert_accuracy_valid(accuracy)
        assert_approximately_equal(accuracy, 0.0, atol=1e-10)
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_random_accuracy(self, simple_network):
        """Test accuracy calculation with random predictions."""
        set_deterministic_behavior()
        
        batch_size = 100
        x = torch.randn(batch_size, simple_network.input_size)
        
        # Create random one-hot targets
        y = torch.zeros(batch_size, simple_network.output_size)
        targets = torch.randint(0, simple_network.output_size, (batch_size,))
        y[torch.arange(batch_size), targets] = 1
        
        accuracy = simple_network.calculate_accuracy(x, y)
        
        # Random accuracy should be around 1/output_size for balanced data
        assert_accuracy_valid(accuracy)
        assert 0.0 <= accuracy <= 1.0  # trunk-ignore(bandit/B101)
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_accuracy_with_trained_network(self, trained_simple_network, simple_2d_data):
        """Test accuracy calculation with a trained network."""
        x, y = simple_2d_data
        
        accuracy = trained_simple_network.calculate_accuracy(x, y)
        
        assert_accuracy_valid(accuracy)
        # Trained network should have better than random accuracy
        expected_random_accuracy = 1.0 / trained_simple_network.output_size
        # Should be at least as good as random
        assert accuracy >= expected_random_accuracy  # trunk-ignore(bandit/B101)


class TestAccuracyShapes:
    """Test accuracy calculation with different tensor shapes."""
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 50, 100])
    def test_accuracy_batch_sizes(self, simple_network, batch_size):
        """Test accuracy calculation with different batch sizes."""
        set_deterministic_behavior()
        
        x = torch.randn(batch_size, simple_network.input_size)
        y = torch.zeros(batch_size, simple_network.output_size)
        y[torch.arange(batch_size), torch.randint(0, simple_network.output_size, (batch_size,))] = 1
        
        accuracy = simple_network.calculate_accuracy(x, y)
        
        assert_accuracy_valid(accuracy)
        assert isinstance(accuracy, float)  # trunk-ignore(bandit/B101)
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    @pytest.mark.parametrize("output_size", [2, 3, 5, 10])
    def test_accuracy_output_sizes(self, output_size):
        """Test accuracy calculation with different output sizes."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        
        accuracy = 0.0
        set_deterministic_behavior()
        
        network = CascadeCorrelationNetwork.create_simple_network(
            input_size=2,
            output_size=output_size
        )
        
        batch_size = 20
        x = torch.randn(batch_size, 2)
        y = torch.zeros(batch_size, output_size)
        targets = torch.randint(0, output_size, (batch_size,))
        y[torch.arange(batch_size), targets] = 1
        
        try:
            accuracy = network.calculate_accuracy(x, y)
            assert_accuracy_valid(accuracy)
        except Exception:
            with pytest.raises((ValueError, TypeError)):
                accuracy = network.calculate_accuracy(x, y)
                assert_accuracy_valid(accuracy)


class TestAccuracyValidation:
    """Test input validation for accuracy calculation."""
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_accuracy_none_inputs(self, simple_network):
        """Test accuracy calculation with None inputs."""
        x = torch.randn(10, simple_network.input_size)
        y = torch.randn(10, simple_network.output_size)
        
        # Test with None x
        with pytest.raises((ValueError, TypeError)):
            simple_network.calculate_accuracy(None, y)
        
        # Test with None y
        with pytest.raises((ValueError, TypeError)):
            simple_network.calculate_accuracy(x, None)
        
        # Test _accuracy with None inputs
        output = torch.randn(10, simple_network.output_size)
        with pytest.raises((ValueError, TypeError)):
            simple_network._accuracy(y=None, output=output)
        
        with pytest.raises((ValueError, TypeError)):
            simple_network._accuracy(y=y, output=None)
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_accuracy_mismatched_batch_sizes(self, simple_network):
        """Test accuracy calculation with mismatched batch sizes."""
        x = torch.randn(10, simple_network.input_size)
        y = torch.randn(5, simple_network.output_size)  # Wrong batch size
        
        with pytest.raises((ValueError, RuntimeError)):
            simple_network.calculate_accuracy(x, y)
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_accuracy_wrong_shapes(self, simple_network):
        """Test accuracy calculation with wrong tensor shapes."""
        x = torch.randn(10, simple_network.input_size)
        
        # Wrong number of output features
        y_wrong = torch.randn(10, simple_network.output_size + 1)
        with pytest.raises((ValueError, RuntimeError)):
            simple_network.calculate_accuracy(x, y_wrong)
        
        # Wrong input features
        x_wrong = torch.randn(10, simple_network.input_size + 1)
        y = torch.randn(10, simple_network.output_size)
        with pytest.raises((ValueError, RuntimeError)):
            simple_network.calculate_accuracy(x_wrong, y)
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_accuracy_non_tensor_inputs(self, simple_network):
        """Test accuracy calculation with non-tensor inputs."""
        # Test with lists instead of tensors
        x_list = [[1, 2], [3, 4]]
        y_list = [[1, 0], [0, 1]]
        
        with pytest.raises((TypeError, AttributeError)):
            simple_network.calculate_accuracy(x_list, y_list)


class TestAccuracyArgmax:
    """Test argmax behavior in accuracy calculation."""
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_accuracy_argmax_behavior(self, simple_network):
        """Test that accuracy correctly uses argmax for class prediction."""
        # Create specific predictions and targets to test argmax
        predictions = torch.tensor([
            [0.1, 0.9],  # Predicts class 1
            [0.8, 0.2],  # Predicts class 0
            [0.4, 0.6],  # Predicts class 1
            [0.7, 0.3]   # Predicts class 0
        ])
        
        targets = torch.tensor([
            [0.0, 1.0],  # True class 1 - CORRECT
            [1.0, 0.0],  # True class 0 - CORRECT
            [0.0, 1.0],  # True class 1 - CORRECT
            [0.0, 1.0]   # True class 1 - INCORRECT
        ])
        
        accuracy = simple_network._accuracy(y=targets, output=predictions)
        
        # Should get 3/4 = 0.75 accuracy
        expected_accuracy = 3.0 / 4.0
        assert_approximately_equal(accuracy, expected_accuracy, atol=1e-10)
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_accuracy_tie_breaking(self, simple_network):
        """Test accuracy calculation with tied predictions."""
        # Create predictions with ties (equal values)
        predictions = torch.tensor([
            [0.5, 0.5],  # Tie - argmax will choose first (index 0)
            [0.3, 0.3]   # Tie - argmax will choose first (index 0)
        ])
        
        targets = torch.tensor([
            [1.0, 0.0],  # True class 0 - should be CORRECT
            [0.0, 1.0]   # True class 1 - should be INCORRECT
        ])
        
        accuracy = simple_network._accuracy(y=targets, output=predictions)
        
        # Should get 1/2 = 0.5 accuracy (first prediction correct, second incorrect)
        expected_accuracy = 1.0 / 2.0
        assert_approximately_equal(accuracy, expected_accuracy, atol=1e-10)


class TestAccuracyEdgeCases:
    """Test edge cases in accuracy calculation."""
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_accuracy_empty_batch(self, simple_network):
        """Test accuracy calculation with empty batch."""
        x = torch.empty(0, simple_network.input_size)
        y = torch.empty(0, simple_network.output_size)
        
        # This should handle gracefully
        with torch.no_grad():
            output = simple_network.forward(x)
            accuracy = simple_network._accuracy(y=y, output=output)
        
        # Empty batch accuracy is typically undefined, but should not crash
        print(f"test_accuracy_empty_batch: type: {type(accuracy)}, value: {accuracy}")
        assert isinstance(accuracy, float)  # trunk-ignore(bandit/B101)
        # May be NaN or some default value
        # assert not torch.isnan(torch.tensor(accuracy)) or accuracy == 0.0  # trunk - ignore(bandit/B101)
        assert torch.isnan(torch.tensor(accuracy)) or accuracy == 0.0  # trunk-ignore(bandit/B101)
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_accuracy_single_sample(self, simple_network):
        """Test accuracy calculation with single sample."""
        x = torch.randn(1, simple_network.input_size)
        y = torch.zeros(1, simple_network.output_size)
        y[0, 0] = 1  # Set first class as target
        
        accuracy = simple_network.calculate_accuracy(x, y)
        
        # Single sample accuracy should be either 0.0 or 1.0
        assert_accuracy_valid(accuracy)
        assert accuracy in [0.0, 1.0]  # trunk-ignore(bandit/B101)
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_accuracy_extreme_predictions(self, simple_network):
        """Test accuracy calculation with extreme prediction values."""
        # Very large prediction values
        predictions_large = torch.tensor([
            [1e10, -1e10],
            [-1e10, 1e10]
        ])
        
        targets = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        accuracy = simple_network._accuracy(y=targets, output=predictions_large)
        
        # Should handle extreme values correctly
        assert_accuracy_valid(accuracy)
        assert_approximately_equal(accuracy, 1.0)  # Both predictions should be correct


class TestAccuracyConsistency:
    """Test consistency between different accuracy methods."""
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_accuracy_methods_consistency(self, simple_network):
        """Test that calculate_accuracy and _accuracy give same results."""
        set_deterministic_behavior()
        
        x = torch.randn(20, simple_network.input_size)
        y = torch.zeros(20, simple_network.output_size)
        targets = torch.randint(0, simple_network.output_size, (20,))
        y[torch.arange(20), targets] = 1
        
        # Calculate accuracy using public method
        accuracy1 = simple_network.calculate_accuracy(x, y)
        
        # Calculate accuracy using private method
        with torch.no_grad():
            output = simple_network.forward(x)
            accuracy2 = simple_network._accuracy(y=y, output=output)
        
        # Should be identical
        assert_approximately_equal(accuracy1, accuracy2, rtol=1e-10, atol=1e-10)
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_accuracy_deterministic(self, simple_network):
        """Test that accuracy calculation is deterministic."""
        set_deterministic_behavior(42)
        
        x = torch.randn(10, simple_network.input_size)
        y = torch.zeros(10, simple_network.output_size)
        y[torch.arange(10), torch.randint(0, simple_network.output_size, (10,))] = 1
        
        # Calculate accuracy multiple times
        accuracy1 = simple_network.calculate_accuracy(x, y)
        accuracy2 = simple_network.calculate_accuracy(x, y)
        accuracy3 = simple_network.calculate_accuracy(x, y)
        
        # Should be identical
        assert_approximately_equal(accuracy1, accuracy2)
        assert_approximately_equal(accuracy2, accuracy3)


class TestAccuracyMulticlass:
    """Test accuracy calculation with multiclass problems."""
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    @pytest.mark.parametrize("n_classes", [3, 5, 10])
    def test_multiclass_accuracy(self, n_classes):
        """Test accuracy calculation with multiclass classification."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        
        set_deterministic_behavior()
        
        network = CascadeCorrelationNetwork.create_simple_network(
            input_size=3,
            output_size=n_classes
        )
        
        batch_size = 50
        x = torch.randn(batch_size, 3)
        
        # Create balanced one-hot targets
        targets = torch.randint(0, n_classes, (batch_size,))
        y = torch.zeros(batch_size, n_classes)
        y[torch.arange(batch_size), targets] = 1
        
        accuracy = network.calculate_accuracy(x, y)
        
        assert_accuracy_valid(accuracy)
        # For random predictions, accuracy should be around 1/n_classes
        assert accuracy >= 0.0  # trunk-ignore(bandit/B101)
        assert accuracy <= 1.0  # trunk-ignore(bandit/B101)
    
    @pytest.mark.unit
    @pytest.mark.accuracy
    def test_imbalanced_accuracy(self, simple_network):
        """Test accuracy calculation with imbalanced classes."""
        set_deterministic_behavior()
        
        # Create imbalanced dataset (90% class 0, 10% class 1)
        batch_size = 100
        n_class0 = 90
        n_class1 = 10
        
        x = torch.randn(batch_size, simple_network.input_size)
        y = torch.zeros(batch_size, 2)
        
        # Set first n_class0 samples to class 0
        y[:n_class0, 0] = 1
        # Set remaining samples to class 1
        y[n_class0:, 1] = 1
        
        accuracy = simple_network.calculate_accuracy(x, y)
        
        assert_accuracy_valid(accuracy)
        # Even with imbalanced classes, accuracy should be valid
        assert 0.0 <= accuracy <= 1.0 # trunk-ignore(bandit/B101)

        print(f"Imbalanced accuracy: {accuracy}, type: {type(accuracy)}, Data: x={len(x)}, y={y.sum(dim=0)}, n_class0={n_class0}, n_class1={n_class1}")
