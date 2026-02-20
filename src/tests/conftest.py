#!/usr/bin/env python
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
# Last Modified: 2026-02-19
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
#####################################################################################################################################################################################################
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

# from typing import Tuple, Dict, Any, Optional
from typing import Dict, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Add parent directories to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from candidate_unit.candidate_unit import CandidateUnit
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

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
    # if config.getoption("--fast-slow", default=False) or os.environ.get("JUNIPER_FAST_SLOW") == "1":
    if config.getoption("--fast-slow", default=False) or os.environ.get("JUNIPER_FAST_SLOW") == "0":
        os.environ.setdefault("CASCOR_LOG_LEVEL", "WARNING")
    else:
        # Even in normal mode, reduce logging overhead for slow tests
        os.environ.setdefault("CASCOR_LOG_LEVEL", "WARNING")

    # Limit thread count to prevent CPU oversubscription when running with pytest-xdist
    # This is critical for parallel test execution performance
    # TODO: Consider using a more sophisticated approach to limit thread count
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
    parser.addoption("--gpu", action="store_true", default=False, help="Run GPU tests")
    parser.addoption("--slow", action="store_true", default=False, help="Run slow tests")
    parser.addoption("--integration", action="store_true", default=False, help="Run integration tests")
    parser.addoption("--fast-slow", action="store_true", default=False, help="Run slow tests with reduced training parameters for faster execution")
    parser.addoption("--run-long", action="store_true", default=False, help="Run long-running correctness tests (e.g., deterministic training resume)")


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

    # CRIT-003: Add --run-long option for long-running correctness tests
    if not config.getoption("--run-long"):
        skip_long = pytest.mark.skip(reason="need --run-long option to run long-running correctness tests")
        for item in items:
            if "long" in item.keywords:
                item.add_marker(skip_long)


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
            "learning_rate": 0.1,
            "candidate_learning_rate": 0.1,
            "candidate_epochs": 3,
            "output_epochs": 3,
            "candidate_pool_size": 2,
            "correlation_threshold": 0.02,
            "max_hidden_units": 2,
            "epochs_max": 5,
            "patience": 2,
            "n_per_spiral": 20,
            "n_samples": 32,
        }
    else:
        return {
            "learning_rate": 0.01,
            # 'learning_rate': 0.02,
            # 'learning_rate': 0.05,
            "candidate_learning_rate": 0.005,
            # 'candidate_learning_rate': 0.01,
            "candidate_epochs": 50,
            "output_epochs": 25,
            "candidate_pool_size": 16,
            "correlation_threshold": 0.1,
            "max_hidden_units": 10,
            "epochs_max": 100,
            "patience": 5,
            "n_per_spiral": 100,
            "n_samples": 32,
        }


# ===================================================================
# DATA GENERATION FIXTURES
# ===================================================================


@pytest.fixture
def simple_2d_data(fast_training_params) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate simple 2D classification data."""
    torch.manual_seed(42)
    n_samples = fast_training_params["n_samples"]
    # Create two classes in 2D space

    class_0 = torch.randn(n_samples // 2, 2) + torch.tensor([-1.0, -1.0])
    class_1 = torch.randn(n_samples // 2, 2) + torch.tensor([1.0, 1.0])

    x = torch.cat([class_0, class_1], dim=0)
    y = torch.cat([torch.tensor([[1, 0]] * (n_samples // 2)), torch.tensor([[0, 1]] * (n_samples // 2))], dim=0).float()

    return x, y


@pytest.fixture
def spiral_2d_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate 2-spiral problem data."""
    torch.manual_seed(42)
    n_per_spiral = 100

    # Generate spiral data
    t = torch.linspace(0, 4 * np.pi, n_per_spiral)

    x1 = t * torch.cos(t) / (4 * np.pi)
    y1 = t * torch.sin(t) / (4 * np.pi)

    # Spiral 2 (rotated)
    x2 = -t * torch.cos(t) / (4 * np.pi)
    y2 = -t * torch.sin(t) / (4 * np.pi)

    x = torch.stack([torch.cat([x1, x2]), torch.cat([y1, y2])], dim=1)

    y = torch.cat([torch.tensor([[1, 0]] * n_per_spiral), torch.tensor([[0, 1]] * n_per_spiral)], dim=0).float()

    return x, y


@pytest.fixture
def n_spiral_data() -> callable:
    """Generate N-spiral problem data (parameterized)."""

    def _generate_n_spiral(n_spirals: int = 3, n_per_spiral: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)

        x_data = []
        y_data = []

        for i in range(n_spirals):
            t = torch.linspace(0, 4 * np.pi, n_per_spiral)
            angle_offset = 2 * np.pi * i / n_spirals

            x_spiral = t * torch.cos(t + angle_offset) / (4 * np.pi)
            y_spiral = t * torch.sin(t + angle_offset) / (4 * np.pi)

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
    y = (x[:, 0] ** 2 + x[:, 1] ** 2).unsqueeze(1)

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
        learning_rate=min(0.1, fast_training_params["learning_rate"]),
        candidate_learning_rate=min(0.1, fast_training_params["candidate_learning_rate"]),
        max_hidden_units=min(2, fast_training_params["max_hidden_units"]),
        candidate_pool_size=min(2, fast_training_params["candidate_pool_size"]),
        correlation_threshold=min(0.01, fast_training_params["correlation_threshold"]),
        patience=min(1, fast_training_params["patience"]),
        candidate_epochs=min(3, fast_training_params["candidate_epochs"]),
        output_epochs=min(3, fast_training_params["output_epochs"]),
        epochs_max=min(5, fast_training_params["epochs_max"]),
    )


@pytest.fixture
def spiral_config(fast_training_params, fast_slow_mode) -> CascadeCorrelationConfig:
    """Create configuration optimized for spiral problems.

    Uses fast_training_params when --fast-slow mode is enabled for faster test execution.
    Correlation threshold is lowered in fast mode to allow candidates to be added.
    """
    # Use lower correlation threshold in fast mode since candidates train fewer epochs
    return CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=2,
        learning_rate=min(0.1, fast_training_params["learning_rate"]),
        candidate_learning_rate=min(0.1, fast_training_params["candidate_learning_rate"]),
        max_hidden_units=min(2, fast_training_params["max_hidden_units"]),
        candidate_pool_size=min(2, fast_training_params["candidate_pool_size"]),
        correlation_threshold=min(0.01, fast_training_params["correlation_threshold"]),
        patience=min(1, fast_training_params["patience"]),
        candidate_epochs=min(3, fast_training_params["candidate_epochs"]),
        output_epochs=min(3, fast_training_params["output_epochs"]),
        epochs_max=min(5, fast_training_params["epochs_max"]),
    )


@pytest.fixture
def regression_config(fast_training_params) -> CascadeCorrelationConfig:
    """Create configuration for regression problems.

    Uses fast_training_params when --fast-slow mode is enabled for faster test execution.
    """
    return CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=1,
        learning_rate=min(0.1, fast_training_params["learning_rate"]),
        candidate_learning_rate=min(0.1, fast_training_params["candidate_learning_rate"]),
        max_hidden_units=min(2, fast_training_params["max_hidden_units"]),
        candidate_pool_size=min(2, fast_training_params["candidate_pool_size"]),
        correlation_threshold=0.01,
        patience=min(1, fast_training_params["patience"]),
        candidate_epochs=min(3, fast_training_params["candidate_epochs"]),
        output_epochs=min(3, fast_training_params["output_epochs"]),
        epochs_max=min(5, fast_training_params["epochs_max"]),
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
    return CandidateUnit(_CandidateUnit__input_size=2, _CandidateUnit__learning_rate=0.01, _CandidateUnit__epochs=10, _CandidateUnit__log_level_name="ERROR")


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
    return {"empty": torch.tensor([]), "nan_values": torch.tensor([[1.0, float("nan")]]), "inf_values": torch.tensor([[1.0, float("inf")]]), "wrong_shape_1d": torch.tensor([1, 2, 3]), "wrong_shape_3d": torch.randn(5, 2, 3), "mismatched_batch": torch.randn(5, 2)}  # when expecting batch size 10


# ===================================================================
# UTILITY FIXTURES
# ===================================================================


@pytest.fixture
def tolerance() -> Dict[str, float]:
    """Standard tolerances for floating point comparisons."""
    return {"rtol": 1e-5, "atol": 1e-8, "correlation_tol": 1e-4, "accuracy_tol": 1e-3, "loss_tol": 1e-6}


@pytest.fixture
def device() -> str:
    """Get appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ===================================================================
# PERFORMANCE FIXTURES
# ===================================================================


@pytest.fixture(autouse=True)
def force_sequential_training(monkeypatch):
    """Force sequential candidate training in tests to prevent multiprocessing deadlocks.

    The parallel training path spawns multiprocessing.Process workers that fail with
    BrokenPipeError in test environments. The _stop_workers() method then blocks for
    15 seconds per worker during shutdown (15s × N workers = 100+ seconds stall).
    By forcing process_count=1, all training uses the sequential path, which is
    functionally identical but avoids multiprocessing overhead and deadlock risk.

    Tests that specifically need to test multiprocessing behavior should mock
    the multiprocessing components directly rather than spawning real processes.
    """
    monkeypatch.setattr(
        CascadeCorrelationNetwork,
        "_calculate_optimal_process_count",
        lambda self: 1,
    )


# Cached logger for test performance - avoids two major costs:
# 1. inspect.getouterframes() during Logger/LogConfig initialization (~55ms per network)
# 2. f-string evaluation in filtered log calls (e.g., self.logger.debug(f"...{tensor}...")
#    evaluates tensor.__repr__() even when log level filters the message)
_cached_log_config = None
_cached_logger = None


class _NoOpLogger:
    """Ultra-lightweight logger replacement for tests.

    Eliminates two performance drains:
    - Logger initialization (inspect.getouterframes): ~55ms per instance
    - f-string argument evaluation in filtered log calls: ~0.9s per fit() call
      from 1000+ tensor.__repr__() evaluations in debug/trace/verbose messages

    WARNING and above still log to stderr for test debugging.
    """

    level = 30  # WARNING

    def trace(self, *a, **kw):
        pass

    def verbose(self, *a, **kw):
        pass

    def debug(self, *a, **kw):
        pass

    def info(self, *a, **kw):
        pass

    def warning(self, msg, *a, **kw):
        print(f"[WARNING] {msg}")

    def error(self, msg, *a, **kw):
        print(f"[ERROR] {msg}")

    def critical(self, msg, *a, **kw):
        print(f"[CRITICAL] {msg}")

    def fatal(self, msg, *a, **kw):
        print(f"[FATAL] {msg}")

    def isEnabledFor(self, level):
        return level >= 30


_noop_logger = _NoOpLogger()


def _fast_init_logging_system(self):
    """Lightweight replacement for _init_logging_system in tests.

    Uses a no-op logger that avoids:
    - LogConfig/Logger creation with inspect.getouterframes() (~55ms per call)
    - f-string evaluation overhead from debug/trace log messages containing tensors
    """
    global _cached_log_config
    import logging

    from log_config.log_config import LogConfig

    self.log_file_name = self.config.log_file_name or "cascade_correlation"
    self.log_file_path = self.config.log_file_path or str(os.path.join(os.getcwd(), "logs"))
    self.log_level_name = self.config.log_level_name or "WARNING"

    if _cached_log_config is None:
        _cached_log_config = LogConfig(
            _LogConfig__log_config=logging.config,
            _LogConfig__log_file_name=self.log_file_name,
            _LogConfig__log_file_path=self.log_file_path,
            _LogConfig__log_level_name=self.log_level_name,
        )

    self.log_config = _cached_log_config
    self.logger = _noop_logger


@pytest.fixture(autouse=True, scope="session")
def _warmup_torch():
    """Trigger lazy initialization of torch internals during collection.

    The first call to torch.nn.Linear / torch.optim.Adam triggers expensive lazy
    imports (sympy, torch._dynamo, etc.) costing ~2s. By warming up here, we move
    that one-time cost to session startup so individual tests don't pay it.
    """
    layer = torch.nn.Linear(2, 2)
    optim = torch.optim.Adam(layer.parameters(), lr=0.01)
    loss = torch.nn.functional.mse_loss(layer(torch.randn(4, 2)), torch.randn(4, 2))
    loss.backward()
    optim.step()


@pytest.fixture(autouse=True, scope="session")
def _cache_logging_system():
    """Cache the logging system to avoid expensive inspect.getouterframes() on every network creation.

    Patches three performance-critical paths:
    1. CascadeCorrelationNetwork._init_logging_system → skip Logger/LogConfig creation
    2. CandidateUnit.__init__ → replace logger with no-op after init
    3. Logger._log_at_level → no-op to eliminate inspect.getouterframes() calls
       from Logger class-level methods (trace/debug/info/verbose/warning/error)
       used by CandidateUnit.__init__, SpiralProblem.__init__, and others.
       This is the dominant cost: ~4.3s per 20 CandidateUnit creations from
       inspect.getmodule() scanning all loaded modules via hasattr().

    Tests that specifically test Logger behavior (e.g., test_logger_coverage.py)
    use @patch.object(Logger, "_log_at_level") which overrides this global patch
    for the duration of those tests.
    """
    from log_config.logger.logger import Logger

    original_init = CascadeCorrelationNetwork._init_logging_system
    CascadeCorrelationNetwork._init_logging_system = _fast_init_logging_system

    # Patch CandidateUnit to use no-op logger (it normally sets self.logger = Logger,
    # which still evaluates f-string arguments like tensor.__repr__() in verbose calls)
    original_cu_init = CandidateUnit.__init__

    def _patched_cu_init(self, *args, **kwargs):
        original_cu_init(self, *args, **kwargs)
        self.logger = _noop_logger

    CandidateUnit.__init__ = _patched_cu_init

    # Patch Logger._log_at_level to eliminate inspect.getouterframes() overhead.
    # Every Logger class method (trace, debug, info, verbose, warning, error, etc.)
    # calls _log_at_level which, even for WARNING-filtered messages, incurs overhead.
    # For messages that DO pass the filter (e.g., WARNING from _seed_random_generator),
    # getouterframes() triggers inspect.getmodule() scanning ~800k hasattr() calls.
    original_log_at_level = Logger._log_at_level

    @classmethod
    def _noop_log_at_level(cls, **kwargs):
        pass

    Logger._log_at_level = _noop_log_at_level

    yield

    CascadeCorrelationNetwork._init_logging_system = original_init
    CandidateUnit.__init__ = original_cu_init
    Logger._log_at_level = original_log_at_level


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
