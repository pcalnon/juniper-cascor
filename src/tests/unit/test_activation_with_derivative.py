#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     test_activation_with_derivative.py
# Author:        Paul Calnon
# Version:       0.3.12
#
# Date Created:  2026-01-21
# Last Modified: 2026-01-21
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Unit tests for ActivationWithDerivative class (CASCOR-P1-003 fix).
#    Verifies that activation wrappers are picklable for multiprocessing.
#
#####################################################################################################################################################################################################
# Notes:
#   - This test suite validates the CASCOR-P1-003 fix for multiprocessing pickling errors.
#   - Tests cover pickling/unpickling, activation functions, and derivative calculations.
#
#####################################################################################################################################################################################################
import pickle
import pytest
import torch

# Import from both modules that define ActivationWithDerivative
from candidate_unit.candidate_unit import (
    ActivationWithDerivative as CandidateActivationWithDerivative,
    CandidateUnit,
)
from cascade_correlation.cascade_correlation import (
    ActivationWithDerivative as CascorActivationWithDerivative,
)


class TestActivationWithDerivativePickling:
    """Tests for ActivationWithDerivative pickling support (CASCOR-P1-003)."""

    @pytest.mark.unit
    def test_pickle_tanh_function(self):
        """Test that tanh activation wrapper can be pickled and unpickled."""
        wrapper = CandidateActivationWithDerivative(torch.tanh)
        pickled = pickle.dumps(wrapper)
        restored = pickle.loads(pickled)
        
        assert restored._activation_name == 'tanh'
        
        # Verify functionality after unpickling
        x = torch.tensor([0.5, 1.0, -0.5])
        output = restored(x)
        assert output.shape == x.shape
    
    @pytest.mark.unit
    def test_pickle_tanh_module(self):
        """Test that Tanh module wrapper can be pickled and unpickled."""
        wrapper = CandidateActivationWithDerivative(torch.nn.Tanh())
        pickled = pickle.dumps(wrapper)
        restored = pickle.loads(pickled)
        
        assert restored._activation_name == 'Tanh'
        
        # Verify functionality after unpickling
        x = torch.tensor([0.5, 1.0, -0.5])
        output = restored(x)
        assert output.shape == x.shape
    
    @pytest.mark.unit
    def test_pickle_sigmoid(self):
        """Test that sigmoid activation wrapper can be pickled and unpickled."""
        wrapper = CandidateActivationWithDerivative(torch.sigmoid)
        pickled = pickle.dumps(wrapper)
        restored = pickle.loads(pickled)
        
        assert restored._activation_name == 'sigmoid'
        
        # Verify functionality
        x = torch.tensor([0.0, 1.0, -1.0])
        output = restored(x)
        expected = torch.sigmoid(x)
        assert torch.allclose(output, expected)
    
    @pytest.mark.unit
    def test_pickle_relu(self):
        """Test that relu activation wrapper can be pickled and unpickled."""
        wrapper = CandidateActivationWithDerivative(torch.relu)
        pickled = pickle.dumps(wrapper)
        restored = pickle.loads(pickled)
        
        assert restored._activation_name == 'relu'
        
        # Verify functionality
        x = torch.tensor([0.5, -0.5, 1.0])
        output = restored(x)
        expected = torch.relu(x)
        assert torch.allclose(output, expected)
    
    @pytest.mark.unit
    def test_pickle_relu_module(self):
        """Test that ReLU module wrapper can be pickled and unpickled."""
        wrapper = CandidateActivationWithDerivative(torch.nn.ReLU())
        pickled = pickle.dumps(wrapper)
        restored = pickle.loads(pickled)
        
        assert restored._activation_name == 'ReLU'


class TestActivationWithDerivativeDerivatives:
    """Tests for derivative calculations in ActivationWithDerivative."""

    @pytest.mark.unit
    def test_tanh_derivative(self):
        """Test tanh derivative: d/dx(tanh(x)) = 1 - tanh^2(x)."""
        wrapper = CandidateActivationWithDerivative(torch.tanh)
        x = torch.tensor([0.0, 0.5, -0.5, 1.0])
        
        derivative = wrapper(x, derivative=True)
        expected = 1.0 - torch.tanh(x) ** 2
        
        assert torch.allclose(derivative, expected)
    
    @pytest.mark.unit
    def test_sigmoid_derivative(self):
        """Test sigmoid derivative: d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))."""
        wrapper = CandidateActivationWithDerivative(torch.sigmoid)
        x = torch.tensor([0.0, 0.5, -0.5, 1.0])
        
        derivative = wrapper(x, derivative=True)
        y = torch.sigmoid(x)
        expected = y * (1.0 - y)
        
        assert torch.allclose(derivative, expected)
    
    @pytest.mark.unit
    def test_relu_derivative(self):
        """Test ReLU derivative: d/dx(relu(x)) = 1 if x > 0 else 0."""
        wrapper = CandidateActivationWithDerivative(torch.relu)
        x = torch.tensor([0.5, -0.5, 1.0, 0.0])
        
        derivative = wrapper(x, derivative=True)
        expected = (x > 0).float()
        
        assert torch.allclose(derivative, expected)
    
    @pytest.mark.unit
    def test_derivative_after_unpickling(self):
        """Test that derivative calculations work correctly after unpickling."""
        wrapper = CandidateActivationWithDerivative(torch.tanh)
        pickled = pickle.dumps(wrapper)
        restored = pickle.loads(pickled)
        
        x = torch.tensor([0.5, 1.0, -0.5])
        
        derivative = restored(x, derivative=True)
        expected = 1.0 - torch.tanh(x) ** 2
        
        assert torch.allclose(derivative, expected)


class TestCandidateUnitPickling:
    """Tests for CandidateUnit pickling with ActivationWithDerivative."""

    @pytest.mark.unit
    def test_candidate_unit_pickling(self):
        """Test that CandidateUnit can be pickled with new activation wrapper."""
        candidate = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__output_size=2
        )
        
        # Verify activation function is ActivationWithDerivative instance
        assert isinstance(candidate.activation_fn, CandidateActivationWithDerivative)
        
        # Pickle and restore
        pickled = pickle.dumps(candidate)
        restored = pickle.loads(pickled)
        
        # Verify restored object works
        assert isinstance(restored.activation_fn, CandidateActivationWithDerivative)
        assert restored.input_size == 2
        assert restored.output_size == 2
    
    @pytest.mark.unit
    def test_candidate_unit_forward_after_unpickling(self):
        """Test that CandidateUnit forward pass works after unpickling."""
        candidate = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__output_size=2
        )
        
        x = torch.randn(10, 2)
        original_output = candidate.forward(x)
        
        # Pickle and restore
        pickled = pickle.dumps(candidate)
        restored = pickle.loads(pickled)
        
        restored_output = restored.forward(x)
        
        # Outputs should match
        assert torch.allclose(original_output, restored_output)


class TestCascorActivationWithDerivative:
    """Tests for ActivationWithDerivative in cascade_correlation module."""

    @pytest.mark.unit
    def test_cascor_activation_pickling(self):
        """Test that CascadeCorrelation's ActivationWithDerivative is picklable."""
        wrapper = CascorActivationWithDerivative(torch.tanh)
        pickled = pickle.dumps(wrapper)
        restored = pickle.loads(pickled)
        
        assert restored._activation_name == 'tanh'
        
        x = torch.tensor([0.5, 1.0, -0.5])
        output = restored(x)
        assert output.shape == x.shape
    
    @pytest.mark.unit
    def test_both_implementations_compatible(self):
        """Test that both implementations produce same results."""
        x = torch.tensor([0.5, 1.0, -0.5])
        
        wrapper_candidate = CandidateActivationWithDerivative(torch.tanh)
        wrapper_cascor = CascorActivationWithDerivative(torch.tanh)
        
        # Both should produce same output
        assert torch.allclose(wrapper_candidate(x), wrapper_cascor(x))
        assert torch.allclose(
            wrapper_candidate(x, derivative=True),
            wrapper_cascor(x, derivative=True)
        )


class TestActivationMapCoverage:
    """Tests for ACTIVATION_MAP coverage."""

    @pytest.mark.unit
    @pytest.mark.parametrize("activation_name,expected_class", [
        ('tanh', 'tanh'),
        ('sigmoid', 'sigmoid'),
        ('relu', 'relu'),
        ('Tanh', 'Tanh'),
        ('Sigmoid', 'Sigmoid'),
        ('ReLU', 'ReLU'),
        ('GELU', 'GELU'),
        ('SELU', 'SELU'),
        ('LeakyReLU', 'LeakyReLU'),
    ])
    def test_activation_map_entries(self, activation_name, expected_class):
        """Test that common activations are in the ACTIVATION_MAP."""
        assert activation_name in CandidateActivationWithDerivative.ACTIVATION_MAP
    
    @pytest.mark.unit
    def test_unknown_activation_fallback(self):
        """Test that unknown activation falls back to ReLU."""
        class CustomActivation:
            def __call__(self, x):
                return x * 2
            
        wrapper = CandidateActivationWithDerivative(CustomActivation())
        pickled = pickle.dumps(wrapper)
        restored = pickle.loads(pickled)
        
        # Should fall back to ReLU after unpickling
        x = torch.tensor([0.5, -0.5])
        output = restored(x)
        # ReLU: max(0, x)
        expected = torch.relu(x)
        assert torch.allclose(output, expected)
