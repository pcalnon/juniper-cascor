"""
Project:       Juniper
Sub-Project:   JuniperCascor
File Name:     conftest.py
File Path:     src/tests/performance/

Author:        Paul Calnon
Version:       0.1.0

Date Created:  2026-03-31
Last Modified: 2026-03-31

License:       MIT License
Copyright:     Copyright (c) 2024-2026 Paul Calnon

Description:
    Pytest fixtures for performance benchmarks.
    Provides deterministic benchmark environments, standardized datasets,
    and network configurations at various scales for profiling hot paths.

    Performance tests are gated behind CASCOR_BENCHMARK_MODE=1 or --run-performance.
    They are never collected in standard test runs.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import torch

from candidate_unit.candidate_unit import CandidateUnit
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

# ===================================================================
# BASELINES DIRECTORY
# ===================================================================

BASELINES_DIR = Path(__file__).parent / "baselines"


# ===================================================================
# DETERMINISTIC ENVIRONMENT
# ===================================================================


@pytest.fixture(autouse=True)
def deterministic_benchmark_env():
    """Pin all sources of non-determinism for reproducible benchmarks.

    Sets fixed seeds, deterministic torch mode, and single-thread BLAS
    to isolate computational cost from scheduling noise.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    yield

    torch.set_num_threads(original_threads)


# ===================================================================
# STANDARDIZED DATASETS
# ===================================================================


@pytest.fixture(scope="session")
def small_spiral_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Small 2-spiral dataset (50 samples per class, 100 total)."""
    torch.manual_seed(42)
    n_per_spiral = 50
    t = torch.linspace(0, 4 * np.pi, n_per_spiral)

    x1 = t * torch.cos(t) / (4 * np.pi)
    y1 = t * torch.sin(t) / (4 * np.pi)
    x2 = -t * torch.cos(t) / (4 * np.pi)
    y2 = -t * torch.sin(t) / (4 * np.pi)

    x = torch.stack([torch.cat([x1, x2]), torch.cat([y1, y2])], dim=1)
    y = torch.cat(
        [
            torch.tensor([[1, 0]] * n_per_spiral),
            torch.tensor([[0, 1]] * n_per_spiral),
        ],
        dim=0,
    ).float()
    return x, y


@pytest.fixture(scope="session")
def medium_spiral_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Medium 2-spiral dataset (200 samples per class, 400 total)."""
    torch.manual_seed(42)
    n_per_spiral = 200
    t = torch.linspace(0, 4 * np.pi, n_per_spiral)

    x1 = t * torch.cos(t) / (4 * np.pi)
    y1 = t * torch.sin(t) / (4 * np.pi)
    x2 = -t * torch.cos(t) / (4 * np.pi)
    y2 = -t * torch.sin(t) / (4 * np.pi)

    x = torch.stack([torch.cat([x1, x2]), torch.cat([y1, y2])], dim=1)
    y = torch.cat(
        [
            torch.tensor([[1, 0]] * n_per_spiral),
            torch.tensor([[0, 1]] * n_per_spiral),
        ],
        dim=0,
    ).float()
    return x, y


@pytest.fixture(scope="session")
def large_spiral_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Large 2-spiral dataset (1000 samples per class, 2000 total)."""
    torch.manual_seed(42)
    n_per_spiral = 1000
    t = torch.linspace(0, 4 * np.pi, n_per_spiral)

    x1 = t * torch.cos(t) / (4 * np.pi)
    y1 = t * torch.sin(t) / (4 * np.pi)
    x2 = -t * torch.cos(t) / (4 * np.pi)
    y2 = -t * torch.sin(t) / (4 * np.pi)

    x = torch.stack([torch.cat([x1, x2]), torch.cat([y1, y2])], dim=1)
    y = torch.cat(
        [
            torch.tensor([[1, 0]] * n_per_spiral),
            torch.tensor([[0, 1]] * n_per_spiral),
        ],
        dim=0,
    ).float()
    return x, y


@pytest.fixture(scope="session")
def high_dim_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """High-dimensional classification data (20 features, 200 samples)."""
    torch.manual_seed(42)
    n_samples = 200
    input_size = 20

    class_0 = torch.randn(n_samples // 2, input_size) - 0.5
    class_1 = torch.randn(n_samples // 2, input_size) + 0.5

    x = torch.cat([class_0, class_1], dim=0)
    y = torch.cat(
        [
            torch.tensor([[1, 0]] * (n_samples // 2)),
            torch.tensor([[0, 1]] * (n_samples // 2)),
        ],
        dim=0,
    ).float()
    return x, y


# ===================================================================
# BENCHMARK NETWORK CONFIGURATIONS
# ===================================================================


def _make_benchmark_config(input_size=2, output_size=2, **overrides):
    """Create a CascadeCorrelationConfig for benchmarking with sensible defaults.

    Uses WARNING log level to suppress hot-path logging overhead (Plan guardrail:
    15+ logger calls per correlation computation; at TRACE/DEBUG these involve
    string formatting with tensor values that add 5-20% overhead).
    """
    defaults = {
        "input_size": input_size,
        "output_size": output_size,
        "learning_rate": 0.01,
        "candidate_learning_rate": 0.005,
        "candidate_epochs": 100,
        "output_epochs": 25,
        "candidate_pool_size": 16,
        "correlation_threshold": 0.001,
        "max_hidden_units": 50,
        "epochs_max": 100,
        "patience": 5,
        "log_level_name": "WARNING",
    }
    defaults.update(overrides)
    return CascadeCorrelationConfig.create_simple_config(**defaults)


@pytest.fixture
def bench_config_small():
    """Small benchmark config: pool=8, epochs=50, max_hidden=5."""
    return _make_benchmark_config(candidate_pool_size=8, candidate_epochs=50, max_hidden_units=5)


@pytest.fixture
def bench_config_standard():
    """Standard benchmark config: pool=16, epochs=100, max_hidden=15."""
    return _make_benchmark_config(candidate_pool_size=16, candidate_epochs=100, max_hidden_units=15)


@pytest.fixture
def bench_config_stress():
    """Stress benchmark config: pool=32, epochs=200, max_hidden=30."""
    return _make_benchmark_config(candidate_pool_size=32, candidate_epochs=200, max_hidden_units=30)


@pytest.fixture
def bench_config_high_dim():
    """High-dimensional benchmark config: input_size=20."""
    return _make_benchmark_config(input_size=20, candidate_pool_size=8, candidate_epochs=50)


# ===================================================================
# NETWORK FIXTURES AT VARIOUS GROWTH STAGES
# ===================================================================


@pytest.fixture
def untrained_network(bench_config_small):
    """Fresh network with 0 hidden units."""
    return CascadeCorrelationNetwork(config=bench_config_small)


def _build_network_with_hidden_units(n_hidden, input_size=2, output_size=2):
    """Construct a network with N synthetic hidden units for benchmarking.

    Instead of training (which is slow), we directly inject hidden unit
    structures with random weights. This gives us a network at the right
    architectural depth for forward pass / output training / serialization
    benchmarks without paying the training cost.
    """
    config = _make_benchmark_config(
        input_size=input_size,
        output_size=output_size,
        max_hidden_units=n_hidden,
    )
    net = CascadeCorrelationNetwork(config=config)

    torch.manual_seed(42)
    for i in range(n_hidden):
        # Each hidden unit receives input + all previous hidden outputs
        unit_input_size = input_size + i
        unit = {
            "weights": torch.randn(unit_input_size) * 0.1,
            "bias": torch.tensor(0.01 * (i + 1)),
            "activation_fn": torch.nn.Tanh(),
            "correlation": 0.5 - i * 0.02,
        }
        net.hidden_units.append(unit)

    # Resize output weights to match expanded input (input_size + n_hidden)
    total_input = input_size + n_hidden
    net.output_weights = torch.randn(total_input, output_size) * 0.1
    net.output_bias = torch.randn(output_size) * 0.1

    return net


@pytest.fixture(scope="session")
def network_5_hidden():
    """Network with 5 synthetic hidden units for benchmarking."""
    return _build_network_with_hidden_units(5)


@pytest.fixture(scope="session")
def network_10_hidden():
    """Network with 10 synthetic hidden units for benchmarking."""
    return _build_network_with_hidden_units(10)


# ===================================================================
# CANDIDATE UNIT FIXTURES FOR MICRO-BENCHMARKS
# ===================================================================


@pytest.fixture
def candidate_unit_2d():
    """CandidateUnit with input_size=2 (matches spiral data)."""
    torch.manual_seed(42)
    return CandidateUnit(CandidateUnit__input_size=2, CandidateUnit__activation_function=torch.nn.Tanh())


@pytest.fixture
def candidate_unit_10d():
    """CandidateUnit with input_size=10."""
    torch.manual_seed(42)
    return CandidateUnit(CandidateUnit__input_size=10, CandidateUnit__activation_function=torch.nn.Tanh())


@pytest.fixture
def candidate_unit_50d():
    """CandidateUnit with input_size=50."""
    torch.manual_seed(42)
    return CandidateUnit(CandidateUnit__input_size=50, CandidateUnit__activation_function=torch.nn.Tanh())


# ===================================================================
# BASELINE SAVE/LOAD UTILITIES
# ===================================================================


def save_baseline(test_name: str, results: Dict, environment: Dict = None):
    """Save benchmark results to JSON baseline file.

    Args:
        test_name: Identifier for this benchmark
        results: Dict with keys like mean_ms, stddev_ms, min_ms, max_ms, iterations
        environment: Optional environment metadata
    """
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "test_name": test_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "environment": environment or _collect_environment(),
    }

    filename = f"baseline_{datetime.now().strftime('%Y%m%d')}.json"
    filepath = BASELINES_DIR / filename

    existing = []
    if filepath.exists():
        with open(filepath) as f:
            existing = json.load(f)

    existing.append(entry)

    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)


def _collect_environment() -> Dict:
    """Collect environment metadata for baseline reproducibility."""
    import platform
    import sys

    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
    }


# ===================================================================
# TIMING CONTEXT MANAGER
# ===================================================================


class BenchmarkTimer:
    """Lightweight timer for manual benchmarking outside pytest-benchmark.

    Usage:
        timer = BenchmarkTimer()
        for _ in range(iterations):
            with timer:
                expensive_operation()
        print(timer.summary())
    """

    def __init__(self):
        self.times_ns = []
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter_ns() - self._start
        self.times_ns.append(elapsed)

    @property
    def times_ms(self):
        return [t / 1_000_000 for t in self.times_ns]

    def summary(self) -> Dict:
        """Return statistical summary of collected timings."""
        if not self.times_ns:
            return {"error": "no timings collected"}

        ms = self.times_ms
        return {
            "mean_ms": float(np.mean(ms)),
            "stddev_ms": float(np.std(ms)),
            "min_ms": float(np.min(ms)),
            "max_ms": float(np.max(ms)),
            "median_ms": float(np.median(ms)),
            "iterations": len(ms),
        }
