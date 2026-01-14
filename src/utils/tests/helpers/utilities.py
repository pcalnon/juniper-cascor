#!/usr/bin/env python
"""
Project:       Juniper
Prototype:     Cascade Correlation Neural Network  
File Name:     utilities.py
Author:        Paul Calnon
Version:       0.1.0

Date:          2025-09-26
Last Modified: 2025-09-26

License:       MIT License
Copyright:     Copyright (c) 2024-2025 Paul Calnon

Description:
    Test utility functions for Cascade Correlation Network tests.
    Provides common test setup, teardown, and helper functions.
"""

import torch
import numpy as np
import time
import psutil
import os
# from typing import Dict, List, Any, Callable, Optional, Tuple, Union
from typing import Dict, List, Any, Tuple
from contextlib import contextmanager

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from log_config.logger.logger import Logger


def set_deterministic_behavior(seed: int = 42) -> None:
    """
    Set deterministic behavior for reproducible tests.
    
    Args:
        seed: Random seed to use
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_network_state(network: CascadeCorrelationNetwork) -> None:
    """
    Reset network to initial untrained state.
    
    Args:
        network: Network to reset
    """
    # Reset hidden units
    network.hidden_units = []
    
    # Reset output weights and bias
    network.output_weights = torch.randn(
        network.input_size, network.output_size, requires_grad=True
    ) * network.random_value_scale
    network.output_bias = torch.randn(
        network.output_size, requires_grad=True
    ) * network.random_value_scale
    
    # Reset history
    network.history = {
        'train_loss': [],
        'value_loss': [],
        'train_accuracy': [],
        'value_accuracy': [],
        'hidden_units_added': []
    }


def create_minimal_network() -> CascadeCorrelationNetwork:
    """
    Create a minimal network for testing basic functionality.
    
    Returns:
        Minimal network instance
    """
    return CascadeCorrelationNetwork.create_simple_network(
        input_size=2,
        output_size=2,
        learning_rate=0.1,
        max_hidden_units=2
    )


def create_test_data(
    n_samples: int = 50,
    input_size: int = 2,
    output_size: int = 2,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create simple test data for basic functionality testing.
    
    Args:
        n_samples: Number of samples
        input_size: Input dimensionality
        output_size: Output dimensionality
        seed: Random seed
        
    Returns:
        (x, y): Input and target tensors
    """
    torch.manual_seed(seed)
    
    x = torch.randn(n_samples, input_size)
    
    # Create one-hot encoded targets
    y = torch.zeros(n_samples, output_size)
    targets = torch.randint(0, output_size, (n_samples,))
    y[torch.arange(n_samples), targets] = 1
    
    return x, y


def calculate_correlation_manually(
    x: torch.Tensor,
    y: torch.Tensor
) -> float:
    """
    Manually calculate Pearson correlation coefficient for verification.
    
    Args:
        x: First variable
        y: Second variable
        
    Returns:
        Correlation coefficient
    """
    x_flat = x.view(-1)
    y_flat = y.view(-1)
    
    x_mean = x_flat.mean()
    y_mean = y_flat.mean()
    
    numerator = ((x_flat - x_mean) * (y_flat - y_mean)).sum()
    x_std = torch.sqrt(((x_flat - x_mean) ** 2).sum())
    y_std = torch.sqrt(((y_flat - y_mean) ** 2).sum())
    
    if x_std == 0 or y_std == 0:
        return 0.0
    
    return (numerator / (x_std * y_std)).item()


def count_network_parameters(network: CascadeCorrelationNetwork) -> Dict[str, int]:
    """
    Count parameters in network components.
    
    Args:
        network: Network to analyze
        
    Returns:
        Parameter counts by component
    """
    counts = {
        'output_weights': network.output_weights.numel(),
        'output_bias': network.output_bias.numel(),
        'hidden_units': 0,
        'total': 0
    }
    
    # Count hidden unit parameters
    for unit in network.hidden_units:
        counts['hidden_units'] += unit['weights'].numel() + unit['bias'].numel()
    
    counts['total'] = counts['output_weights'] + counts['output_bias'] + counts['hidden_units']
    
    return counts


def measure_training_time(
    network: CascadeCorrelationNetwork,
    x: torch.Tensor,
    y: torch.Tensor,
    **fit_kwargs
) -> Tuple[Dict[str, List], float]:
    """
    Measure training time and return history.
    
    Args:
        network: Network to train
        x: Training input
        y: Training targets
        **fit_kwargs: Additional arguments for fit method
        
    Returns:
        (history, elapsed_time): Training history and elapsed time in seconds
    """
    start_time = time.time()
    history = network.fit(x, y, **fit_kwargs)
    elapsed_time = time.time() - start_time
    
    return history, elapsed_time


@contextmanager
def monitor_memory():
    """
    Context manager to monitor memory usage during tests.
    
    Yields:
        Dict with memory statistics
    """
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    stats = {'initial_memory_mb': initial_memory}
    
    try:
        yield stats
    finally:
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        stats['final_memory_mb'] = final_memory
        stats['memory_increase_mb'] = final_memory - initial_memory


def check_gradient_flow(network: CascadeCorrelationNetwork) -> Dict[str, bool]:
    """
    Check if gradients are flowing properly through network.
    
    Args:
        network: Network to check
        
    Returns:
        Dictionary indicating gradient status for each component
    """
    status = {
        'output_weights': network.output_weights.requires_grad
        and network.output_weights.grad is not None
        and torch.isfinite(network.output_weights.grad).all()
    }

    # Check output bias
    status['output_bias'] = (
        network.output_bias.requires_grad and
        network.output_bias.grad is not None and
        torch.isfinite(network.output_bias.grad).all()
    )

    return status


def create_mock_residual_error(
    batch_size: int = 10,
    output_size: int = 2,
    error_magnitude: float = 1.0,
    seed: int = 42
) -> torch.Tensor:
    """
    Create mock residual error for testing candidate training.
    
    Args:
        batch_size: Number of samples
        output_size: Output dimensionality
        error_magnitude: Magnitude of errors
        seed: Random seed
        
    Returns:
        Residual error tensor
    """
    torch.manual_seed(seed)
    return error_magnitude * torch.randn(batch_size, output_size)


def verify_network_forward_pass(
    network: CascadeCorrelationNetwork,
    x: torch.Tensor
) -> Dict[str, Any]:  # sourcery skip: assign-if-exp, extract-method
    """
    Verify network forward pass and return diagnostic information.
    
    Args:
        network: Network to test
        x: Input tensor
        
    Returns:
        Diagnostic information about forward pass
    """
    info = {
        'input_shape': x.shape,
        'output_shape': None,
        'hidden_outputs': [],
        'final_output': None,
        'success': True,
        'error': None
    }
    
    try:
        # Track intermediate outputs
        hidden_outputs = []
        # current_input = x
        
        # Process each hidden unit
        for i, unit in enumerate(network.hidden_units):
            unit_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
            unit_output = unit['activation_fn'](
                torch.sum(unit_input * unit['weights'], dim=1) + unit['bias']
            ).unsqueeze(1)
            hidden_outputs.append(unit_output)
            info['hidden_outputs'].append({
                'unit_index': i,
                'input_shape': unit_input.shape,
                'output_shape': unit_output.shape,
                'weights_shape': unit['weights'].shape,
                'bias_shape': unit['bias'].shape
            })
        
        # Final network input
        if hidden_outputs:
            network_input = torch.cat([x] + hidden_outputs, dim=1)
        else:
            network_input = x
        
        # Output layer
        output = torch.mm(network_input, network.output_weights) + network.output_bias
        
        info['output_shape'] = output.shape
        info['final_output'] = output
        info['network_input_shape'] = network_input.shape
        
    except Exception as e:
        info['success'] = False
        info['error'] = str(e)
    
    return info


def compare_networks(
    network1: CascadeCorrelationNetwork,
    network2: CascadeCorrelationNetwork,
    tolerance: float = 1e-6
) -> Dict[str, bool]:
    """
    Compare two networks for structural and parameter equality.
    
    Args:
        network1: First network
        network2: Second network
        tolerance: Tolerance for parameter comparison
        
    Returns:
        Comparison results
    """
    comparison = {
        'same_structure': True,
        'same_parameters': True,
        'same_hidden_units': True,
        'details': {}
    }
    
    # Check basic structure
    if (network1.input_size != network2.input_size or 
        network1.output_size != network2.output_size or
        len(network1.hidden_units) != len(network2.hidden_units)):
        comparison['same_structure'] = False
        comparison['details']['structure_mismatch'] = True
    
    # Check output parameters
    if not torch.allclose(network1.output_weights, network2.output_weights, atol=tolerance):
        comparison['same_parameters'] = False
        comparison['details']['output_weights_differ'] = True
    
    if not torch.allclose(network1.output_bias, network2.output_bias, atol=tolerance):
        comparison['same_parameters'] = False
        comparison['details']['output_bias_differs'] = True
    
    # Check hidden units
    for i, (unit1, unit2) in enumerate(zip(network1.hidden_units, network2.hidden_units, strict=False)):
        if not torch.allclose(unit1['weights'], unit2['weights'], atol=tolerance):
            comparison['same_hidden_units'] = False
            comparison['details'][f'hidden_unit_{i}_weights_differ'] = True
        
        if not torch.allclose(unit1['bias'], unit2['bias'], atol=tolerance):
            comparison['same_hidden_units'] = False
            comparison['details'][f'hidden_unit_{i}_bias_differs'] = True
    
    return comparison


def generate_training_scenarios() -> List[Dict[str, Any]]:
    """
    Generate various training scenarios for comprehensive testing.
    
    Returns:
        List of training scenario configurations
    """
    scenarios = [
        {
            'name': 'basic_training',
            'max_epochs': 5,
            'early_stopping': False,
            'expected_units': 0
        },
        {
            'name': 'early_stopping_enabled',
            'max_epochs': 20,
            'early_stopping': True,
            'expected_units': None  # Variable
        },
        {
            'name': 'max_epochs_reached',
            'max_epochs': 3,
            'early_stopping': False,
            'expected_units': None
        },
        {
            'name': 'single_epoch',
            'max_epochs': 1,
            'early_stopping': False,
            'expected_units': 0
        }
    ]
    Logger.debug(f"Generated {len(scenarios)} training scenarios for testing")
    return scenarios


def validate_test_environment() -> Dict[str, Any]:
    """
    Validate that test environment is properly configured.
    
    Returns:
        Environment validation results
    """
    validation = {
        'pytorch_available': True,
        'cuda_available': torch.cuda.is_available(),
        'deterministic_mode': torch.backends.cudnn.deterministic,
        'benchmark_mode': torch.backends.cudnn.benchmark,
        'random_seed_set': True,  # We assume it's set by conftest.py
        'issues': []
    }
    
    try:
        # Test basic tensor operations
        x = torch.randn(2, 2)
        y = torch.mm(x, x.T)
        assert y.shape == (2, 2) # trunk-ignore(bandit/B101)
    except Exception as e:
        validation['pytorch_available'] = False
        validation['issues'].append(f"PyTorch operations failed: {e}")
    
    if not validation['deterministic_mode']:
        validation['issues'].append("Deterministic mode not enabled")
    
    if validation['benchmark_mode']:
        validation['issues'].append("Benchmark mode should be disabled for deterministic tests")
    
    return validation
