#!/usr/bin/env python
"""
Project:       Juniper
Prototype:     Cascade Correlation Neural Network  
File Name:     assertions.py
Author:        Paul Calnon
Version:       0.1.0

Date:          2025-09-26
Last Modified: 2025-09-26

License:       MIT License
Copyright:     Copyright (c) 2024-2025 Paul Calnon

Description:
    Custom assertion utilities for Cascade Correlation Network tests.
    Provides domain-specific assertions for network behavior validation.
"""

import torch
# import numpy as np
from typing import Union, Optional, List, Dict, Any
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from candidate_unit.candidate_unit import CandidateUnit


def assert_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: Union[tuple, torch.Size],
    msg: Optional[str] = None
) -> None:
    """
    Assert that tensor has expected shape.
    
    Args:
        tensor: Tensor to check
        expected_shape: Expected shape
        msg: Optional error message
    """
    if msg is None:
        msg = f"Expected shape {expected_shape}, got {tensor.shape}"
    assert tensor.shape == expected_shape, msg  # trunk-ignore(bandit/B101)


def assert_tensor_finite(
    tensor: torch.Tensor,
    msg: Optional[str] = None
) -> None:
    """
    Assert that tensor contains only finite values (no NaN or Inf).
    
    Args:
        tensor: Tensor to check
        msg: Optional error message
    """
    if msg is None:
        msg = f"Tensor contains non-finite values: {tensor}"
    assert torch.isfinite(tensor).all(), msg  # trunk-ignore(bandit/B101)



def assert_tensor_not_empty(
    tensor: torch.Tensor,
    msg: Optional[str] = None
) -> None:
    """
    Assert that tensor is not empty.
    
    Args:
        tensor: Tensor to check
        msg: Optional error message
    """
    if msg is None:
        msg = f"Tensor is empty: {tensor.shape}"
    assert tensor.numel() > 0, msg # trunk-ignore(bandit/B101)


def assert_tensor_in_range(
    tensor: torch.Tensor,
    min_val: float = -float('inf'),
    max_val: float = float('inf'),
    msg: Optional[str] = None
) -> None:
    """
    Assert that tensor values are within specified range.
    
    Args:
        tensor: Tensor to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        msg: Optional error message
    """
    if msg is None:
        msg = f"Tensor values outside range [{min_val}, {max_val}]: min={tensor.min().item()}, max={tensor.max().item()}"
    assert tensor.min().item() >= min_val and tensor.max().item() <= max_val, msg # trunk-ignore(bandit/B101)


def assert_correlation_valid(
    correlation: float,
    tolerance: float = 1e-6,
    msg: Optional[str] = None
) -> None:
    """
    Assert that correlation coefficient is valid (between -1 and 1).
    
    Args:
        correlation: Correlation coefficient
        tolerance: Tolerance for bounds checking
        msg: Optional error message
    """
    if msg is None:
        msg = f"Invalid correlation coefficient: {correlation} (must be in [-1, 1])"
    assert -1 - tolerance <= correlation <= 1 + tolerance, msg # trunk-ignore(bandit/B101)


def assert_accuracy_valid(
    accuracy: float,
    tolerance: float = 1e-6,
    msg: Optional[str] = None
) -> None:
    """
    Assert that accuracy is valid (between 0 and 1).
    
    Args:
        accuracy: Accuracy value
        tolerance: Tolerance for bounds checking
        msg: Optional error message
    """
    if msg is None:
        msg = f"Invalid accuracy: {accuracy} (must be in [0, 1])"
    assert -tolerance <= accuracy <= 1 + tolerance, msg # trunk-ignore(bandit/B101)


def assert_network_structure_valid(
    network: CascadeCorrelationNetwork,
    msg: Optional[str] = None
) -> None:
    """
    Assert that network structure is valid.
    
    Args:
        network: Network to check
        msg: Optional error message
    """
    if msg is None:
        msg = "Network structure is invalid"
    
    # Check basic attributes exist
    assert hasattr(network, 'input_size'), f"{msg}: missing input_size" # trunk-ignore(bandit/B101)
    assert hasattr(network, 'output_size'), f"{msg}: missing output_size" # trunk-ignore(bandit/B101)
    assert hasattr(network, 'hidden_units'), f"{msg}: missing hidden_units" # trunk-ignore(bandit/B101)
    assert hasattr(network, 'output_weights'), f"{msg}: missing output_weights" # trunk-ignore(bandit/B101)
    assert hasattr(network, 'output_bias'), f"{msg}: missing output_bias" # trunk-ignore(bandit/B101)
    
    # Check types
    assert isinstance(network.input_size, int), f"{msg}: input_size not int" # trunk-ignore(bandit/B101)
    assert isinstance(network.output_size, int), f"{msg}: output_size not int" # trunk-ignore(bandit/B101)
    assert isinstance(network.hidden_units, list), f"{msg}: hidden_units not list" # trunk-ignore(bandit/B101)
    assert isinstance(network.output_weights, torch.Tensor), f"{msg}: output_weights not tensor" # trunk-ignore(bandit/B101)
    assert isinstance(network.output_bias, torch.Tensor), f"{msg}: output_bias not tensor" # trunk-ignore(bandit/B101)
    
    # Check values
    assert network.input_size > 0, f"{msg}: input_size must be positive" # trunk-ignore(bandit/B101)
    assert network.output_size > 0, f"{msg}: output_size must be positive" # trunk-ignore(bandit/B101)
    
    # Check output weights/bias shapes
    expected_input_size = network.input_size + len(network.hidden_units)
    assert network.output_weights.shape == (expected_input_size, network.output_size), f"{msg}: output_weights shape mismatch: expected {(expected_input_size, network.output_size)}, got {network.output_weights.shape}" # trunk-ignore(bandit/B101)
    assert network.output_bias.shape == (network.output_size,), f"{msg}: output_bias shape mismatch: expected {(network.output_size,)}, got {network.output_bias.shape}" # trunk-ignore(bandit/B101)




def assert_hidden_unit_valid(
    unit: Dict[str, Any],
    expected_input_size: int,
    msg: Optional[str] = None
) -> None:
    """
    Assert that hidden unit structure is valid.
    
    Args:
        unit: Hidden unit dictionary
        expected_input_size: Expected input size for the unit
        msg: Optional error message
    """
    if msg is None:
        msg = "Hidden unit structure is invalid"
    
    # Check required keys
    required_keys = {'weights', 'bias', 'activation_fn', 'correlation'}
    assert all(key in unit for key in required_keys), f"{msg}: missing required keys. Expected {required_keys}, got {set(unit.keys())}"  # trunk-ignore(bandit/B101)

    # Check types and shapes
    assert isinstance(unit['weights'], torch.Tensor), f"{msg}: weights not tensor" # trunk-ignore(bandit/B101)
    assert isinstance(unit['bias'], torch.Tensor), f"{msg}: bias not tensor" # trunk-ignore(bandit/B101)
    assert callable(unit['activation_fn']), f"{msg}: activation_fn not callable" # trunk-ignore(bandit/B101)
    assert isinstance(unit['correlation'], (int, float)), f"{msg}: correlation not numeric" # trunk-ignore(bandit/B101)

    # Check shapes
    assert unit['weights'].shape == (expected_input_size,), f"{msg}: weights shape mismatch: expected {(expected_input_size,)}, got {unit['weights'].shape}" # trunk-ignore(bandit/B101)
    assert unit['bias'].shape == (), f"{msg}: bias shape mismatch: expected (), got {unit['bias'].shape}" # trunk-ignore(bandit/B101)

    # Check correlation validity
    assert_correlation_valid(unit['correlation'], msg=f"{msg}: invalid correlation")


def assert_candidate_valid(
    candidate: CandidateUnit,
    msg: Optional[str] = None
) -> None:
    """
    Assert that candidate unit is valid.
    
    Args:
        candidate: Candidate unit to check
        msg: Optional error message
    """
    if msg is None:
        msg = "Candidate unit is invalid"
    
    # Check that candidate has required attributes
    required_attrs = ['weights', 'bias', 'correlation']
    for attr in required_attrs:
        assert hasattr(candidate, attr), f"{msg}: missing attribute {attr}" # trunk-ignore(bandit/B101)
    
    # Check that weights and bias are tensors
    assert isinstance(candidate.weights, torch.Tensor), f"{msg}: weights not tensor" # trunk-ignore(bandit/B101)
    assert isinstance(candidate.bias, torch.Tensor), f"{msg}: bias not tensor" # trunk-ignore(bandit/B101)
    
    # Check that weights and bias have finite values
    assert_tensor_finite(candidate.weights, msg=f"{msg}: weights not finite")
    assert_tensor_finite(candidate.bias, msg=f"{msg}: bias not finite")


def assert_training_history_valid(
    history: Dict[str, List],
    msg: Optional[str] = None
) -> None:
    """
    Assert that training history is valid.
    
    Args:
        history: Training history dictionary
        msg: Optional error message
    """
    if msg is None:
        msg = "Training history is invalid"
    
    # Check required keys
    required_keys = {'train_loss', 'train_accuracy', 'hidden_units_added'}
    assert all(key in history for key in required_keys), f"{msg}: missing required keys. Expected {required_keys}, got {set(history.keys())}" # trunk-ignore(bandit/B101)
    
    # Check that all values are lists
    for key, value in history.items():
        assert isinstance(value, list), f"{msg}: {key} is not a list" # trunk-ignore(bandit/B101)
    
    # Check that losses and accuracies are valid numbers
    for loss in history['train_loss']:
        assert isinstance(loss, (int, float)) and loss >= 0, f"{msg}: invalid train_loss value: {loss}" # trunk-ignore(bandit/B101)
    
    for acc in history['train_accuracy']:
        assert_accuracy_valid(acc, msg=f"{msg}: invalid train_accuracy value: {acc}")


def assert_prediction_shapes_match(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    msg: Optional[str] = None
) -> None:
    """
    Assert that predictions and targets have compatible shapes.
    
    Args:
        predictions: Network predictions
        targets: Target values
        msg: Optional error message
    """
    if msg is None:
        msg = f"Prediction shapes don't match: predictions {predictions.shape}, targets {targets.shape}"
    
    assert predictions.shape == targets.shape, msg # trunk-ignore(bandit/B101)


def assert_gradient_exists(
    tensor: torch.Tensor,
    msg: Optional[str] = None
) -> None:
    """
    Assert that tensor has gradients.
    
    Args:
        tensor: Tensor to check
        msg: Optional error message
    """
    if msg is None:
        msg = f"Tensor has no gradient: {tensor}"
    
    assert tensor.requires_grad, f"{msg}: requires_grad is False" # trunk-ignore(bandit/B101)
    assert tensor.grad is not None, f"{msg}: grad is None" # trunk-ignore(bandit/B101)


def assert_gradient_finite(
    tensor: torch.Tensor,
    msg: Optional[str] = None
) -> None:
    """
    Assert that tensor gradients are finite.
    
    Args:
        tensor: Tensor to check
        msg: Optional error message
    """
    if msg is None:
        msg = f"Tensor gradient is not finite: {tensor.grad}"
    
    assert tensor.grad is not None, f"{msg}: grad is None" # trunk-ignore(bandit/B101)
    assert torch.isfinite(tensor.grad).all(), f"{msg}: grad contains non-finite values" # trunk-ignore(bandit/B101)


def assert_network_learns(
    initial_loss: float,
    final_loss: float,
    min_improvement: float = 0.01,
    msg: Optional[str] = None
) -> None:
    """
    Assert that network shows learning (loss improvement).
    
    Args:
        initial_loss: Loss at start of training
        final_loss: Loss at end of training
        min_improvement: Minimum relative improvement required
        msg: Optional error message
    """
    if msg is None:
        msg = f"Network did not learn: initial_loss={initial_loss:.6f}, final_loss={final_loss:.6f}"
    
    improvement = (initial_loss - final_loss) / initial_loss
    assert improvement >= min_improvement, f"{msg}: improvement={improvement:.6f} < min_improvement={min_improvement}" # trunk-ignore(bandit/B101)


def assert_correlation_improved(
    old_correlation: float,
    new_correlation: float,
    min_improvement: float = 1e-6,
    msg: Optional[str] = None
) -> None:
    """
    Assert that correlation improved during training.
    
    Args:
        old_correlation: Previous correlation value
        new_correlation: New correlation value
        min_improvement: Minimum improvement required
        msg: Optional error message
    """
    if msg is None:
        msg = f"Correlation did not improve: old={old_correlation:.6f}, new={new_correlation:.6f}"
    
    improvement = abs(new_correlation) - abs(old_correlation)
    assert improvement >= min_improvement, f"{msg}: improvement={improvement:.6f} < min_improvement={min_improvement}" # trunk-ignore(bandit/B101)


def assert_approximately_equal(
    actual: Union[float, torch.Tensor],
    expected: Union[float, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: Optional[str] = None
) -> None:
    """
    Assert that values are approximately equal.
    
    Args:
        actual: Actual value
        expected: Expected value
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Optional error message
    """
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        close = torch.allclose(actual, expected, rtol=rtol, atol=atol)
    elif isinstance(actual, torch.Tensor):
        close = torch.allclose(actual, torch.tensor(expected), rtol=rtol, atol=atol)
    elif isinstance(expected, torch.Tensor):
        close = torch.allclose(torch.tensor(actual), expected, rtol=rtol, atol=atol)
    else:
        close = abs(actual - expected) <= atol + rtol * abs(expected)
    
    if msg is None:
        msg = f"Values not approximately equal: actual={actual}, expected={expected}, rtol={rtol}, atol={atol}"
    
    assert close, msg  # trunk-ignore(bandit/B101)
