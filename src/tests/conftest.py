#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network  
# Author:        Paul Calnon
# 
# Last Modified: 2026-01-12
# 
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
# 
# Description:
#####################################################################################################################################################################################################
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# Purpose:       Juniper Project Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.1.0
# File Name:     conftest.py
# File Path:     <Project>/<Sub-Project>/<Application>/src/tests/
#
# Date Created:  2025-09-26
# Last Modified: 2026-01-17
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#     Pytest configuration and shared fixtures for Cascade Correlation Network tests.
#     Provides common test setup, data generation, and network instances.
#
#####################################################################################################################################################################################################
# Notes:
#
########################################################################################################)#############################################################################################
# References:
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################
import os
import sys

# ===================================================================
# CRITICAL: Set CASCOR_LOG_LEVEL BEFORE any cascor imports
# ===================================================================
# The logging level must be set before importing cascor modules because
# constants.py reads CASCOR_LOG_LEVEL at import time. Setting it in
# pytest_configure() is too late since test collection imports modules.
#
# This dramatically improves test performance by reducing logging overhead.
# Even simple logging operations add significant time when called thousands
# of times during training loops.
if "CASCOR_LOG_LEVEL" not in os.environ:
    os.environ["CASCOR_LOG_LEVEL"] = "WARNING"

import pytest
import torch
import numpy as np
# from typing import Tuple, Dict, Any, Optional
from typing import Tuple, Dict
from unittest.mock import MagicMock

# Add parent directories to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from candidate_unit.candidate_unit import CandidateUnit


# ===================================================================
# PYTEST CONFIGURATION
# ===================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Disable GPU by default in tests
    if not config.getoption("--gpu", default=False):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # PERFORMANCE FIX: Set log level to WARNING to reduce logging overhead in tests
    # The extensive logging (TRACE, DEBUG, VERBOSE, INFO) adds significant overhead
    # even when log_level_name is set to 'ERROR' in individual components
    if config.getoption("--fast-slow", default=False) or os.environ.get("JUNIPER_FAST_SLOW") == "1":
        os.environ.setdefault("CASCOR_LOG_LEVEL", "WARNING")
    else:
        # Even in normal mode, reduce logging overhead for slow tests
        os.environ.setdefault("CASCOR_LOG_LEVEL", "WARNING")
    
    # Limit thread count to prevent CPU oversubscription when running with pytest-xdist
    # This is critical for parallel test execution performance
    if os.environ.get("PYTEST_XDIST_WORKER"):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        torch.set_num_threads(1)
    
    # Set deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configure torch for consistent behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--gpu", action="store_true", default=False, help="Run GPU tests"
    )
    parser.addoption(
        "--slow", action="store_true", default=False, help="Run slow tests"
    )
    parser.addoption(
        "--integration", action="store_true", default=False, help="Run integration tests"
    )
    parser.addoption(
        "--fast-slow", action="store_true", default=False, 
        help="Run slow tests with reduced training parameters for faster execution"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on command line options."""
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="need --slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(reason="need --integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


# ===================================================================
# FAST-SLOW MODE CONFIGURATION
# ===================================================================

@pytest.fixture(scope="session")
def fast_slow_mode(request):
    """Check if fast-slow mode is enabled via --fast-slow flag or JUNIPER_FAST_SLOW env var."""
    return request.config.getoption("--fast-slow") or os.environ.get("JUNIPER_FAST_SLOW", "0") == "1"


@pytest.fixture(scope="session")
def training_scale(fast_slow_mode):
    """Scale factor for training parameters in fast-slow mode (0.1 = 10% of normal)."""
    return 0.1 if fast_slow_mode else 1.0


@pytest.fixture(scope="session")
def fast_training_params(fast_slow_mode):
    """Return optimized training parameters for fast-slow mode.
    
    These parameters dramatically reduce training time while maintaining test validity.
    Tests should validate learning signal (improvement from baseline) rather than 
    absolute performance thresholds.
    """
    if fast_slow_mode:
        return {
            'candidate_epochs': 3,
            'output_epochs': 3,
            'candidate_pool_size': 2,
            'max_hidden_units': 2,
            'epochs_max': 5,
            'patience': 2,
            'n_per_spiral': 20,
            'n_samples': 32,
        }
    else:
        return {
            'candidate_epochs': 50,
            'output_epochs': 25,
            'candidate_pool_size': 16,
            'max_hidden_units': 10,
            'epochs_max': 100,
            'patience': 5,
            'n_per_spiral': 100,
            'n_samples': 100,
        }


# ===================================================================
# DATA GENERATION FIXTURES
# ===================================================================

@pytest.fixture
def simple_2d_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate simple 2D classification data."""
    torch.manual_seed(42)
    n_samples = 100
    
    # Create two classes in 2D space
    class_0 = torch.randn(n_samples // 2, 2) + torch.tensor([-1.0, -1.0])
    class_1 = torch.randn(n_samples // 2, 2) + torch.tensor([1.0, 1.0])
    
    x = torch.cat([class_0, class_1], dim=0)
    y = torch.cat([
        torch.tensor([[1, 0]] * (n_samples // 2)),
        torch.tensor([[0, 1]] * (n_samples // 2))
    ], dim=0).float()
    
    return x, y


@pytest.fixture
def spiral_2d_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate 2-spiral problem data."""
    torch.manual_seed(42)
    n_per_spiral = 100
    
    # Generate spiral data
    t = torch.linspace(0, 4*np.pi, n_per_spiral)
    
    # Spiral 1
    x1 = t * torch.cos(t) / (4*np.pi)
    y1 = t * torch.sin(t) / (4*np.pi)
    
    # Spiral 2 (rotated)
    x2 = -t * torch.cos(t) / (4*np.pi)
    y2 = -t * torch.sin(t) / (4*np.pi)
    
    x = torch.stack([
        torch.cat([x1, x2]),
        torch.cat([y1, y2])
    ], dim=1)
    
    y = torch.cat([
        torch.tensor([[1, 0]] * n_per_spiral),
        torch.tensor([[0, 1]] * n_per_spiral)
    ], dim=0).float()
    
    return x, y


@pytest.fixture
def n_spiral_data() -> callable:
    """Generate N-spiral problem data (parameterized)."""
    def _generate_n_spiral(n_spirals: int = 3, n_per_spiral: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        
        x_data = []
        y_data = []
        
        for i in range(n_spirals):
            t = torch.linspace(0, 4*np.pi, n_per_spiral)
            angle_offset = 2 * np.pi * i / n_spirals
            
            x_spiral = t * torch.cos(t + angle_offset) / (4*np.pi)
            y_spiral = t * torch.sin(t + angle_offset) / (4*np.pi)
            
            x_data.append(torch.stack([x_spiral, y_spiral], dim=1))
            
            # One-hot encoding for class i
            y_spiral = torch.zeros(n_per_spiral, n_spirals)
            y_spiral[:, i] = 1
            y_data.append(y_spiral)
        
        x = torch.cat(x_data, dim=0)
        y = torch.cat(y_data, dim=0)
        
        return x, y
    
    return _generate_n_spiral


@pytest.fixture
def regression_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate regression data for testing."""
    torch.manual_seed(42)
    n_samples = 200
    
    x = torch.randn(n_samples, 2)
    # Non-linear target function
    y = (x[:, 0]**2 + x[:, 1]**2).unsqueeze(1)
    
    return x, y


# ===================================================================
# NETWORK CONFIGURATION FIXTURES
# ===================================================================

@pytest.fixture
def simple_config(fast_training_params) -> CascadeCorrelationConfig:
    """Create a simple configuration for testing.
    
    Uses fast_training_params when --fast-slow mode is enabled for faster test execution.
    """
    return CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=2,
        learning_rate=0.1,
        candidate_learning_rate=0.01,
        max_hidden_units=min(5, fast_training_params['max_hidden_units']),
        candidate_pool_size=min(8, fast_training_params['candidate_pool_size']),
        correlation_threshold=0.1,
        patience=min(3, fast_training_params['patience']),
        candidate_epochs=min(10, fast_training_params['candidate_epochs']),
        output_epochs=min(10, fast_training_params['output_epochs']),
        epochs_max=min(20, fast_training_params['epochs_max'])
    )


@pytest.fixture
def spiral_config(fast_training_params) -> CascadeCorrelationConfig:
    """Create configuration optimized for spiral problems.
    
    Uses fast_training_params when --fast-slow mode is enabled for faster test execution.
    """
    return CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=2,
        learning_rate=0.05,
        candidate_learning_rate=0.01,
        max_hidden_units=fast_training_params['max_hidden_units'],
        candidate_pool_size=fast_training_params['candidate_pool_size'],
        correlation_threshold=0.2,
        patience=fast_training_params['patience'],
        candidate_epochs=fast_training_params['candidate_epochs'],
        output_epochs=fast_training_params['output_epochs'],
        epochs_max=fast_training_params['epochs_max']
    )


@pytest.fixture
def regression_config(fast_training_params) -> CascadeCorrelationConfig:
    """Create configuration for regression problems.
    
    Uses fast_training_params when --fast-slow mode is enabled for faster test execution.
    """
    return CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=1,
        learning_rate=0.01,
        candidate_learning_rate=0.005,
        max_hidden_units=min(8, fast_training_params['max_hidden_units']),
        candidate_pool_size=min(12, fast_training_params['candidate_pool_size']),
        correlation_threshold=0.15,
        patience=min(5, fast_training_params['patience']),
        candidate_epochs=min(30, fast_training_params['candidate_epochs']),
        output_epochs=min(15, fast_training_params['output_epochs']),
        epochs_max=min(50, fast_training_params['epochs_max'])
    )


# ===================================================================
# NETWORK INSTANCE FIXTURES
# ===================================================================

@pytest.fixture
def simple_network(simple_config) -> CascadeCorrelationNetwork:
    """Create a simple cascade correlation network."""
    return CascadeCorrelationNetwork(config=simple_config)


@pytest.fixture
def spiral_network(spiral_config) -> CascadeCorrelationNetwork:
    """Create a network configured for spiral problems."""
    return CascadeCorrelationNetwork(config=spiral_config)


@pytest.fixture
def regression_network(regression_config) -> CascadeCorrelationNetwork:
    """Create a network configured for regression."""
    return CascadeCorrelationNetwork(config=regression_config)


@pytest.fixture
def trained_simple_network(simple_network, simple_2d_data) -> CascadeCorrelationNetwork:
    """Create a pre-trained simple network."""
    x, y = simple_2d_data
    simple_network.fit(x, y, max_epochs=5)
    return simple_network


# ===================================================================
# CANDIDATE UNIT FIXTURES
# ===================================================================

@pytest.fixture
def simple_candidate() -> CandidateUnit:
    """Create a simple candidate unit."""
    return CandidateUnit(
        _CandidateUnit__input_size=2,
        _CandidateUnit__learning_rate=0.01,
        _CandidateUnit__epochs=10,
        _CandidateUnit__log_level_name="ERROR"
    )


# ===================================================================
# MOCK FIXTURES
# ===================================================================

@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    mock = MagicMock()
    mock.trace.return_value = None
    mock.debug.return_value = None
    mock.info.return_value = None
    mock.warning.return_value = None
    mock.error.return_value = None
    mock.verbose.return_value = None
    return mock


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    mock = MagicMock()
    mock.input_size = 2
    mock.output_size = 2
    mock.learning_rate = 0.1
    mock.candidate_learning_rate = 0.01
    mock.max_hidden_units = 5
    mock.candidate_pool_size = 8
    mock.correlation_threshold = 0.1
    mock.patience = 3
    mock.candidate_epochs = 10
    mock.output_epochs = 10
    mock.epochs_max = 20
    mock.random_seed = 42
    mock.random_max_value = 1.0
    mock.sequence_max_value = 100
    mock.random_value_scale = 0.1
    return mock


# ===================================================================
# VALIDATION FIXTURES
# ===================================================================

@pytest.fixture
def valid_tensor_2d() -> torch.Tensor:
    """Valid 2D tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(10, 2)


@pytest.fixture
def valid_target_2d() -> torch.Tensor:
    """Valid 2D target tensor (one-hot)."""
    torch.manual_seed(42)
    targets = torch.zeros(10, 2)
    targets[torch.arange(10), torch.randint(0, 2, (10,))] = 1
    return targets


@pytest.fixture
def invalid_tensors() -> Dict[str, torch.Tensor]:
    """Collection of invalid tensors for testing."""
    return {
        'empty': torch.tensor([]),
        'nan_values': torch.tensor([[1.0, float('nan')]]),
        'inf_values': torch.tensor([[1.0, float('inf')]]),
        'wrong_shape_1d': torch.tensor([1, 2, 3]),
        'wrong_shape_3d': torch.randn(5, 2, 3),
        'mismatched_batch': torch.randn(5, 2)  # when expecting batch size 10
    }


# ===================================================================
# UTILITY FIXTURES
# ===================================================================

@pytest.fixture
def tolerance() -> Dict[str, float]:
    """Standard tolerances for floating point comparisons."""
    return {
        'rtol': 1e-5,
        'atol': 1e-8,
        'correlation_tol': 1e-4,
        'accuracy_tol': 1e-3,
        'loss_tol': 1e-6
    }


@pytest.fixture
def device() -> str:
    """Get appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ===================================================================
# CLEANUP FIXTURES  
# ===================================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Clean up temporary files created during tests."""
    yield
    # Cleanup logic can be added here if needed
    # pass


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test."""
    torch.manual_seed(42)
    np.random.seed(42)
