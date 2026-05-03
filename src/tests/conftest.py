#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# Purpose:       Juniper Project Cascade Correlation Neural Network
#
# Author:        Paul Calnon
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
import sysconfig

# ===================================================================
# Free-threading interpreter guard (PEP 703 / 3.14t)
# ===================================================================
# Cascor pulls in psutil, torch, numpy, and other native extensions whose
# wheels are built for the regular CPython 3.14 ABI. Running pytest under
# a free-threading interpreter (``python3.14t``) makes those extensions
# load with the wrong PyObject layout and segfault during module init —
# typically inside ``psutil/_psutil_linux.abi3.so``'s ``PyInit__psutil_linux``.
# The crash happens during collection before any test runs, with a very
# unhelpful diagnostic.
#
# Bail out early with an actionable error instead of segfaulting. Override
# with ``CASCOR_ALLOW_FREE_THREADING=1`` for users who have rebuilt their
# native deps for the ``t`` ABI and want to try anyway.
if sysconfig.get_config_var("Py_GIL_DISABLED") and not os.environ.get("CASCOR_ALLOW_FREE_THREADING"):
    sys.stderr.write(
        "\n"
        "ERROR: pytest is running under a free-threading CPython build (Py_GIL_DISABLED=1).\n"
        "       The conda env's native dependencies (psutil, torch, numpy, ...) are\n"
        "       built for the regular CPython 3.14 ABI and segfault when loaded\n"
        "       under the 3.14t interpreter due to PEP 703 PyObject layout changes.\n"
        "\n"
        "       Recreate the JuniperCascor conda env with a regular (GIL) Python:\n"
        "         conda create -n JuniperCascor python=3.13 -c conda-forge -y\n"
        "         conda activate JuniperCascor\n"
        "         pip install -e .\n"
        "\n"
        "       To override this guard at your own risk, set CASCOR_ALLOW_FREE_THREADING=1.\n"
        "\n"
    )
    raise SystemExit(2)

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
# Use insert(0, ...) to ensure local src/ takes precedence over any editable
# installs of legacy JuniperCascor that may shadow the api package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from candidate_unit.candidate_unit import CandidateUnit
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

# ===================================================================
# PYTEST CONFIGURATION
# ===================================================================


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set matplotlib backend to non-interactive Agg before any pyplot imports
    import matplotlib

    matplotlib.use("Agg")

    # Disable GPU by default in tests
    if not config.getoption("--gpu", default=False):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Propagate --fast-slow flag to environment so tests that use
    # os.environ.get("JUNIPER_FAST_SLOW") (e.g., test_spiral_problem.py)
    # detect fast-slow mode correctly. Without this, only conftest fixtures
    # see the flag, and integration tests run with full training parameters.
    if config.getoption("--fast-slow", default=False):
        os.environ["JUNIPER_FAST_SLOW"] = "1"

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
    parser.addoption("--run-performance", action="store_true", default=False, help="Run performance benchmark tests (requires CASCOR_BENCHMARK_MODE=1 or this flag)")


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

    # Performance benchmarks: require --run-performance or CASCOR_BENCHMARK_MODE=1
    run_perf = config.getoption("--run-performance") or os.environ.get("CASCOR_BENCHMARK_MODE") == "1"
    if not run_perf:
        skip_perf = pytest.mark.skip(reason="need --run-performance flag or CASCOR_BENCHMARK_MODE=1 to run performance benchmarks")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_perf)


# ===================================================================
# WEBSOCKET ORIGIN HEADER INJECTION
# ===================================================================
# Phase B-pre-b: Control-path origin validation requires all WS test
# connections to include an allowed Origin header. Monkeypatch globally.
_WS_TEST_ORIGIN = "http://localhost:8050"
_original_ws_connect = None


@pytest.fixture(scope="session", autouse=True)
def _inject_ws_origin_header():
    """Inject Origin header into all TestClient.websocket_connect calls."""
    from starlette.testclient import TestClient

    global _original_ws_connect
    _original_ws_connect = TestClient.websocket_connect

    def _patched_ws_connect(self, url, subprotocols=None, **kwargs):
        headers = kwargs.get("headers", {})
        headers.setdefault("origin", _WS_TEST_ORIGIN)
        kwargs["headers"] = headers
        return _original_ws_connect(self, url, subprotocols=subprotocols, **kwargs)

    TestClient.websocket_connect = _patched_ws_connect
    yield
    TestClient.websocket_connect = _original_ws_connect


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
    # Seed is already set by the autouse reset_random_seeds fixture (CR-076)
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
    # Seed is already set by the autouse reset_random_seeds fixture (CR-076)
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
        # Seed is already set by the autouse reset_random_seeds fixture (CR-076)

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
    # Seed is already set by the autouse reset_random_seeds fixture (CR-076)
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
    # Seed is already set by the autouse reset_random_seeds fixture (CR-076)
    return torch.randn(10, 2)


@pytest.fixture
def valid_target_2d() -> torch.Tensor:
    """Valid 2D target tensor (one-hot)."""
    # Seed is already set by the autouse reset_random_seeds fixture (CR-076)
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


@pytest.fixture(autouse=True)
def skip_training_snapshots(request, monkeypatch):
    """Skip HDF5 snapshot creation during training to avoid h5py hangs.

    The create_snapshot() call inside train_output_layer() is non-fatal
    (already wrapped in try/except with warning) and is not part of the
    training logic under test. h5py 3.16.0 on Python 3.14 can hang
    during ObjectID creation and weakref operations, causing indefinite
    blocks that defeat pytest-timeout.

    Tests that specifically test snapshot/serialization behavior are
    excluded from this patch (detected by "snapshot" or "serializ" in
    the test node ID).
    """
    node_id = request.node.nodeid.lower()
    if "snapshot" in node_id or "serializ" in node_id:
        return  # Let snapshot/serialization tests use real create_snapshot
    monkeypatch.setattr(
        CascadeCorrelationNetwork,
        "create_snapshot",
        lambda self, snapshot_dir=None: None,
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

    # Patch CandidateUnit.__setstate__ to replace the Logger class reference
    # with _noop_logger after deserialization (e.g., pickle/unpickle in
    # multiprocessing or snapshot restore). Without this, __setstate__ resets
    # self.logger = Logger, bypassing the __init__ patch above.
    original_cu_setstate = CandidateUnit.__setstate__

    def _patched_cu_setstate(self, state):
        original_cu_setstate(self, state)
        self.logger = _noop_logger

    CandidateUnit.__setstate__ = _patched_cu_setstate

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
    CandidateUnit.__setstate__ = original_cu_setstate
    Logger._log_at_level = original_log_at_level


# ===================================================================
# CLEANUP FIXTURES
# ===================================================================


# ===================================================================
# MULTIPROCESSING CLEANUP HELPERS (Issue 3)
# ===================================================================
# A test that spins up a CascadeCorrelationNetwork (or directly uses
# multiprocessing) can leave forkserver / worker children behind if the
# test crashes, the network is garbage-collected without an explicit
# ``_shutdown_worker_pool`` call, or the session terminates via
# ``os._exit(0)`` (which bypasses every atexit / Finalize handler that
# would otherwise reap them). The forkserver's parent-death detection
# uses a pipe heartbeat that can take many minutes to fire, so the
# children survive across pytest sessions and accumulate to multi-GB
# RSS — observed at 12 GB across three worktrees on 2026-05-03,
# eventually saturating swap and causing OOM kills of new pytest runs.
#
# These helpers + the per-test and session-end hooks below ensure no
# multiprocessing children outlive the pytest process they were
# spawned by.


def _reap_multiprocessing_children(timeout: float = 2.0) -> None:
    """Terminate every live multiprocessing child + the forkserver.

    Walks ``multiprocessing.active_children()`` first (workers, manager
    server processes, ``mp.Process`` instances created via any context),
    then explicitly stops ``multiprocessing.forkserver._forkserver`` so
    the long-lived forkserver process itself doesn't survive as an
    orphan. Each step is wrapped in a broad ``try/except`` because this
    runs at process tear-down — re-raising would defeat the whole point.
    """
    import multiprocessing
    import os
    import signal

    # Phase 1: ask each child to terminate.
    for child in list(multiprocessing.active_children()):
        try:
            child.terminate()
        except Exception:  # nosec B110 — cleanup must not propagate
            pass

    # Phase 2: join with a deadline; SIGKILL stragglers.
    for child in list(multiprocessing.active_children()):
        try:
            child.join(timeout=timeout)
            if child.is_alive() and child.pid is not None:
                os.kill(child.pid, signal.SIGKILL)
                child.join(timeout=0.5)
        except Exception:  # nosec B110
            pass

    # Phase 3: stop the forkserver itself (the long-lived process that
    # spawns workers on demand). It auto-exits when its socket closes
    # but only after the parent-death heartbeat fires — which is what
    # we're trying to avoid.
    try:
        from multiprocessing.forkserver import _forkserver

        pid = getattr(_forkserver, "_forkserver_pid", None)
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                os.waitpid(pid, 0)
            except (ChildProcessError, OSError):
                pass
            _forkserver._forkserver_pid = None
            _forkserver._forkserver_address = None
    except Exception:  # nosec B110
        pass


@pytest.fixture
def pdeathsig_workers(monkeypatch):
    """Opt-in Linux-only defense: set ``PR_SET_PDEATHSIG = SIGKILL`` on
    every multiprocessing child this test spawns, so the kernel kills
    them immediately when the pytest parent dies (no waiting for the
    forkserver's heartbeat-based parent-death detection).

    Issue 3 step 4. Use this on tests that legitimately spawn worker
    pools and want OS-level orphan protection as a defence-in-depth on
    top of the autouse ``_reap_test_spawned_children`` fixture (which
    only fires if pytest reaches normal teardown — a SIGKILL of the
    pytest parent itself bypasses it). Example:

        def test_my_thing(pdeathsig_workers, ...):
            net = CascadeCorrelationNetwork(...)
            net._ensure_worker_pool(2)
            ...

    No-op on non-Linux platforms.
    """
    import sys

    if sys.platform != "linux":
        return

    import ctypes
    import multiprocessing.process as _mp_process

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _PR_SET_PDEATHSIG = 1
    _SIGKILL = 9

    original_bootstrap = _mp_process.BaseProcess._bootstrap

    def _bootstrap_with_pdeathsig(self, *args, **kwargs):
        # prctl runs in the child immediately after fork — failures here
        # are non-fatal (worst case is parent-death detection falls back
        # to the heartbeat).
        try:
            libc.prctl(_PR_SET_PDEATHSIG, _SIGKILL, 0, 0, 0)
        except Exception:  # nosec B110
            pass
        return original_bootstrap(self, *args, **kwargs)

    monkeypatch.setattr(_mp_process.BaseProcess, "_bootstrap", _bootstrap_with_pdeathsig)


@pytest.fixture(autouse=True)
def _reap_test_spawned_children():
    """Per-test safety net: terminate any multiprocessing children
    spawned during this test that the test itself didn't clean up.

    Issue 3 step 2. Tests that opt in to real multiprocessing (network
    construction with the autouse ``force_sequential_training`` fixture
    still produces a 1-worker pool whose forkserver lives in the
    background) historically did not call ``_shutdown_worker_pool`` on
    teardown, so each test would leak its forkserver + worker until the
    session ended — at which point ``pytest_unconfigure``'s
    ``os._exit(0)`` orphaned them entirely.

    This fixture records the children that already existed at test
    start and, after the test completes, terminates only those that
    appeared during the test. Pre-existing children (e.g. session-scoped
    workers, the forkserver if shared across tests) are left alone so
    we don't break performance tests that intentionally reuse a pool.
    """
    import multiprocessing
    import os
    import signal

    pre = {c.pid for c in multiprocessing.active_children() if c.pid is not None}
    try:
        yield
    finally:
        new_children = [c for c in multiprocessing.active_children() if c.pid is not None and c.pid not in pre]
        for child in new_children:
            try:
                child.terminate()
            except Exception:  # nosec B110
                pass
        for child in new_children:
            try:
                child.join(timeout=2.0)
                if child.is_alive() and child.pid is not None:
                    os.kill(child.pid, signal.SIGKILL)
                    child.join(timeout=0.5)
            except Exception:  # nosec B110
                pass


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Force process exit after pytest finalize to prevent hangs from orphaned threads.

    Two sources of hangs at session end:
    1. concurrent.futures.thread registers an atexit handler that calls
       shutdown(wait=True) on every live ThreadPoolExecutor.
    2. Non-daemon training threads (e.g. from API integration tests)
       prevent the main thread from exiting even after atexit handlers run.

    Earlier (pre-P-6) this lived in a ``scope="session"`` autouse fixture
    that ran during fixture teardown — *before* pytest's
    ``pytest_sessionfinish`` (writes JUnit XML, lets pytest-cov save the
    .coverage data file) and ``pytest_terminal_summary`` (prints the
    summary line and the coverage report). ``os._exit(0)`` bypasses all
    Python finalization, so the JUnit XML, coverage data, terminal
    summary, and HTML coverage report were all silently dropped on every
    CI run since the fixture was introduced — masked for months by the
    upstream test failures keeping the gate skipped.

    Moving the logic to ``pytest_unconfigure(trylast=True)`` runs it
    *after* pytest-cov's own ``pytest_unconfigure`` (and after
    ``pytest_sessionfinish`` / ``pytest_terminal_summary`` have already
    fired), so all reports land on disk before we force-exit.

    Issue 3 step 3: ``os._exit(0)`` skips ``atexit`` handlers and
    multiprocessing's ``Finalize`` callbacks, so any surviving
    forkserver / worker children would be orphaned and live for many
    minutes after the parent exits (the forkserver's parent-death
    detection is heartbeat-based). Reap them explicitly here, *before*
    forcing the exit.
    """
    import atexit
    import concurrent.futures.thread as _thread_mod
    import threading

    atexit.unregister(_thread_mod._python_exit)

    # Always reap multiprocessing children before we lose the chance.
    # Cheap when there are none; essential when there are.
    _reap_multiprocessing_children()

    # If non-daemon threads are still alive (e.g. training threads from
    # API integration tests), force immediate process exit.
    alive = [t for t in threading.enumerate() if not t.daemon and t is not threading.main_thread()]
    if alive:
        import os
        import sys

        # Issue 2: ``os._exit`` skips ``atexit``, finalizers, AND the
        # implicit stdout/stderr flush that normal Python exit performs.
        # When pytest's stdout is redirected to a file (CI, ``> log``,
        # tee, etc.) the streams are block-buffered, so the freshly-
        # written ``X passed in Ys`` summary line and any post-summary
        # diagnostic output sits in the buffer and is silently
        # discarded the moment ``os._exit`` fires. Flush explicitly so
        # the user sees the same final output whether or not the
        # non-daemon-thread escape hatch was needed.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # nosec B110 — never let cleanup raise here
            pass

        os._exit(0)


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
