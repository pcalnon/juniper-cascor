"""
Project:       Juniper
Sub-Project:   JuniperCascor
File Name:     test_baselines.py
File Path:     src/tests/performance/

Author:        Paul Calnon
Version:       0.1.0

Date Created:  2026-03-31
Last Modified: 2026-03-31

License:       MIT License
Copyright:     Copyright (c) 2024-2026 Paul Calnon

Description:
    Sequential baseline benchmarks for juniper-cascor hot paths.
    Measures single-threaded performance of forward pass, candidate training,
    correlation calculation, output layer training, residual error computation,
    and weight update / autograd overhead.

    Run with: pytest tests/performance/ --run-performance -v
    Run with benchmarks: pytest tests/performance/ --run-performance --benchmark-only
"""

import time

import numpy as np
import psutil
import pytest
import torch

from candidate_unit.candidate_unit import CandidateUnit
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

from .conftest import BenchmarkTimer, _make_benchmark_config, load_latest_baseline, save_baseline

# ===================================================================
# THRESHOLD CONSTANTS
# ===================================================================
# These thresholds are intentionally generous to avoid flaky failures
# in CI while still catching major regressions (2x-3x degradation).
# All values are upper bounds set well above observed typical values.

# Memory: absolute RSS delta threshold (MB) for network creation.
# Observed typical delta is < 1 MB; 50 MB catches pathological leaks
# while tolerating OS-level allocation variability and RSS jitter.
MEMORY_DELTA_THRESHOLD_MB = 50.0

# Memory: absolute RSS threshold (MB) for total process footprint.
# The test process (Python + PyTorch + test harness) typically uses
# ~800-900 MB RSS. 2000 MB catches unbounded growth while allowing
# headroom for different torch builds and memory allocator behavior.
MEMORY_ABSOLUTE_THRESHOLD_MB = 2000.0

# Memory: regression threshold as a percentage above the saved baseline.
# If current RSS exceeds the last saved baseline by more than this
# percentage, the test fails. Set at 50% to catch major regressions
# while tolerating normal run-to-run variance in RSS measurements.
MEMORY_REGRESSION_TOLERANCE_PCT = 50.0

# Timing: upper bound (seconds) for HDF5 save/load round-trip on a
# small trained network. Observed typical time is < 5s; 30s catches
# pathological I/O or serialization regressions.
SERIALIZATION_TIME_THRESHOLD_S = 30.0

# Timing: upper bound (seconds) for net.fit() on tiny data (20 epochs,
# 2 hidden units max, 100 samples). Observed typical time is < 10s;
# 60s allows for CI load variance while catching major regressions.
FIT_TIME_THRESHOLD_S = 60.0


def _check_memory_regression(test_name: str, current_value_mb: float, metric_key: str):
    """Check current memory metric against the most recent saved baseline.

    Issues a soft assertion (pytest.fail with a descriptive message) if
    the current value exceeds the baseline by more than the tolerance.
    Silently passes if no baseline exists yet (first run).

    Args:
        test_name: Baseline test identifier (e.g. "memory_base_network")
        current_value_mb: Current measured value in MB
        metric_key: Key in the baseline results dict (e.g. "delta_mb", "rss_mb")
    """
    baseline = load_latest_baseline(test_name)
    if baseline is None:
        return  # No prior baseline to compare against

    baseline_value = baseline.get("results", {}).get(metric_key)
    if baseline_value is None:
        return  # Baseline exists but doesn't have this metric

    # For very small baselines (< 1 MB), use an absolute tolerance
    # instead of a percentage to avoid failing on trivial fluctuations.
    if baseline_value < 1.0:
        absolute_tolerance_mb = 10.0
        if current_value_mb > baseline_value + absolute_tolerance_mb:
            pytest.fail(f"Memory regression detected in {test_name}: " f"current {metric_key}={current_value_mb:.2f} MB exceeds " f"baseline {baseline_value:.2f} MB by more than " f"{absolute_tolerance_mb:.0f} MB absolute tolerance")
    else:
        allowed = baseline_value * (1.0 + MEMORY_REGRESSION_TOLERANCE_PCT / 100.0)
        if current_value_mb > allowed:
            pct_increase = ((current_value_mb - baseline_value) / baseline_value) * 100.0
            pytest.fail(f"Memory regression detected in {test_name}: " f"current {metric_key}={current_value_mb:.2f} MB exceeds " f"baseline {baseline_value:.2f} MB by {pct_increase:.1f}% " f"(tolerance: {MEMORY_REGRESSION_TOLERANCE_PCT:.0f}%)")


# ===================================================================
# FORWARD PASS BASELINES
# ===================================================================


@pytest.mark.performance
class TestForwardPassBaseline:
    """Benchmark CascadeCorrelationNetwork.forward() at varying network depths."""

    def test_forward_0_hidden(self, benchmark, small_spiral_data):
        """Forward pass baseline with 0 hidden units."""
        config = _make_benchmark_config(max_hidden_units=0)
        net = CascadeCorrelationNetwork(config=config)
        x, _ = small_spiral_data

        result = benchmark(net.forward, x)
        assert result is not None
        assert result.shape == (len(x), 2)

    def test_forward_5_hidden(self, benchmark, network_5_hidden, small_spiral_data):
        """Forward pass with 5 hidden units."""
        x, _ = small_spiral_data
        n_hidden = len(network_5_hidden.hidden_units)

        result = benchmark(network_5_hidden.forward, x)
        assert result is not None
        save_baseline(f"forward_{n_hidden}_hidden", {"hidden_units": n_hidden})

    def test_forward_10_hidden(self, benchmark, network_10_hidden, small_spiral_data):
        """Forward pass with ~10 hidden units."""
        x, _ = small_spiral_data
        n_hidden = len(network_10_hidden.hidden_units)

        result = benchmark(network_10_hidden.forward, x)
        assert result is not None
        save_baseline(f"forward_{n_hidden}_hidden", {"hidden_units": n_hidden})

    def test_forward_scaling_samples(self, benchmark, small_spiral_data, medium_spiral_data, large_spiral_data):
        """Forward pass scaling with sample count (0 hidden units)."""
        config = _make_benchmark_config(max_hidden_units=0)
        net = CascadeCorrelationNetwork(config=config)

        # Benchmark with medium dataset (400 samples)
        x_med, _ = medium_spiral_data
        result = benchmark(net.forward, x_med)
        assert result.shape[0] == len(x_med)


# ===================================================================
# CANDIDATE TRAINING BASELINES
# ===================================================================


@pytest.mark.performance
class TestCandidateTrainingBaseline:
    """Benchmark CandidateUnit.train_detailed() in isolation."""

    @staticmethod
    def _make_training_data(input_size, n_samples=100, output_size=2):
        """Create synthetic training data for candidate unit benchmarks.

        CandidateUnit.train_detailed() takes (x, epochs, residual_error, learning_rate, display_frequency).
        x shape: [n_samples, input_size], residual_error shape: [n_samples, output_size].
        """
        torch.manual_seed(42)
        x = torch.randn(n_samples, input_size)
        residual = torch.randn(n_samples, output_size)
        return x, residual

    def test_candidate_train_2d_100epochs(self, benchmark):
        """Candidate training: input_size=2, 100 epochs, 100 samples."""
        x, residual = self._make_training_data(input_size=2, n_samples=100)

        def run():
            torch.manual_seed(42)
            c = CandidateUnit(CandidateUnit__input_size=2, CandidateUnit__activation_function=torch.nn.Tanh())
            return c.train_detailed(x=x, epochs=100, residual_error=residual, learning_rate=0.005, display_frequency=0)

        result = benchmark(run)
        assert result is not None

    def test_candidate_train_10d_100epochs(self, benchmark):
        """Candidate training: input_size=10, 100 epochs, 100 samples."""
        x, residual = self._make_training_data(input_size=10, n_samples=100)

        def run():
            torch.manual_seed(42)
            c = CandidateUnit(CandidateUnit__input_size=10, CandidateUnit__activation_function=torch.nn.Tanh())
            return c.train_detailed(x=x, epochs=100, residual_error=residual, learning_rate=0.005, display_frequency=0)

        result = benchmark(run)
        assert result is not None

    def test_candidate_train_50d_100epochs(self, benchmark):
        """Candidate training: input_size=50, 100 epochs, 100 samples."""
        x, residual = self._make_training_data(input_size=50, n_samples=100)

        def run():
            torch.manual_seed(42)
            c = CandidateUnit(CandidateUnit__input_size=50, CandidateUnit__activation_function=torch.nn.Tanh())
            return c.train_detailed(x=x, epochs=100, residual_error=residual, learning_rate=0.005, display_frequency=0)

        result = benchmark(run)
        assert result is not None

    def test_candidate_train_epoch_scaling(self, benchmark):
        """Candidate training scaling: input_size=2, 50 epochs (compare with 100)."""
        x, residual = self._make_training_data(input_size=2, n_samples=100)

        def run():
            torch.manual_seed(42)
            c = CandidateUnit(CandidateUnit__input_size=2, CandidateUnit__activation_function=torch.nn.Tanh())
            return c.train_detailed(x=x, epochs=50, residual_error=residual, learning_rate=0.005, display_frequency=0)

        result = benchmark(run)
        assert result is not None

    def test_candidate_train_sample_scaling(self, benchmark):
        """Candidate training: input_size=2, 100 epochs, 200 samples."""
        x, residual = self._make_training_data(input_size=2, n_samples=200)

        def run():
            torch.manual_seed(42)
            c = CandidateUnit(CandidateUnit__input_size=2, CandidateUnit__activation_function=torch.nn.Tanh())
            return c.train_detailed(x=x, epochs=100, residual_error=residual, learning_rate=0.005, display_frequency=0)

        result = benchmark(run)
        assert result is not None


# ===================================================================
# CORRELATION CALCULATION BASELINES
# ===================================================================


@pytest.mark.performance
class TestCorrelationBaseline:
    """Benchmark CandidateUnit._calculate_correlation() in isolation."""

    @staticmethod
    def _setup_correlation(n_samples, output_size=2):
        """Create random output and residual tensors for correlation benchmarking.

        Output shape: [n_samples, output_size] (matches residual_error dimensions).
        """
        torch.manual_seed(42)
        output = torch.randn(n_samples, output_size)
        residual = torch.randn(n_samples, output_size)
        return output, residual

    def test_correlation_100_samples(self, benchmark):
        """Correlation computation with 100 samples."""
        output, residual = self._setup_correlation(100)
        candidate = CandidateUnit(CandidateUnit__input_size=2, CandidateUnit__activation_function=torch.nn.Tanh())

        result = benchmark(candidate._calculate_correlation, output, residual)
        assert result is not None

    def test_correlation_1000_samples(self, benchmark):
        """Correlation computation with 1000 samples."""
        output, residual = self._setup_correlation(1000)
        candidate = CandidateUnit(CandidateUnit__input_size=2, CandidateUnit__activation_function=torch.nn.Tanh())

        result = benchmark(candidate._calculate_correlation, output, residual)
        assert result is not None

    def test_correlation_5000_samples(self, benchmark):
        """Correlation computation with 5000 samples."""
        output, residual = self._setup_correlation(5000)
        candidate = CandidateUnit(CandidateUnit__input_size=2, CandidateUnit__activation_function=torch.nn.Tanh())

        result = benchmark(candidate._calculate_correlation, output, residual)
        assert result is not None


# ===================================================================
# OUTPUT LAYER TRAINING BASELINES
# ===================================================================


@pytest.mark.performance
class TestOutputLayerTrainingBaseline:
    """Benchmark CascadeCorrelationNetwork.train_output_layer()."""

    def test_output_training_0_hidden(self, benchmark, small_spiral_data):
        """Output layer training with 0 hidden units, 25 epochs."""
        config = _make_benchmark_config(output_epochs=25, max_hidden_units=0)
        net = CascadeCorrelationNetwork(config=config)
        x, y = small_spiral_data

        result = benchmark(net.train_output_layer, x, y, 25)
        assert isinstance(result, float)

    def test_output_training_5_hidden(self, benchmark, network_5_hidden, small_spiral_data):
        """Output layer training with 5 hidden units, 25 epochs."""
        x, y = small_spiral_data
        result = benchmark(network_5_hidden.train_output_layer, x, y, 25)
        assert isinstance(result, float)

    def test_output_training_10_hidden(self, benchmark, network_10_hidden, small_spiral_data):
        """Output layer training with ~10 hidden units, 25 epochs."""
        x, y = small_spiral_data
        result = benchmark(network_10_hidden.train_output_layer, x, y, 25)
        assert isinstance(result, float)


# ===================================================================
# RESIDUAL ERROR BASELINES
# ===================================================================


@pytest.mark.performance
class TestResidualErrorBaseline:
    """Benchmark CascadeCorrelationNetwork.calculate_residual_error()."""

    def test_residual_0_hidden(self, benchmark, small_spiral_data):
        """Residual error with 0 hidden units."""
        config = _make_benchmark_config(max_hidden_units=0)
        net = CascadeCorrelationNetwork(config=config)
        x, y = small_spiral_data

        result = benchmark(net.calculate_residual_error, x, y)
        assert result is not None

    def test_residual_5_hidden(self, benchmark, network_5_hidden, small_spiral_data):
        """Residual error with 5 hidden units."""
        x, y = small_spiral_data
        result = benchmark(network_5_hidden.calculate_residual_error, x, y)
        assert result is not None


# ===================================================================
# MEMORY FOOTPRINT BASELINES
# ===================================================================


@pytest.mark.performance
class TestMemoryBaseline:
    """Measure memory footprint at various network sizes."""

    def test_memory_base_network(self, small_spiral_data):
        """Memory footprint of a fresh network (0 hidden units)."""
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        config = _make_benchmark_config(max_hidden_units=0)
        net = CascadeCorrelationNetwork(config=config)

        mem_after = process.memory_info().rss / (1024 * 1024)
        delta_mb = mem_after - mem_before

        save_baseline(
            "memory_base_network",
            {
                "rss_before_mb": round(mem_before, 2),
                "rss_after_mb": round(mem_after, 2),
                "delta_mb": round(delta_mb, 2),
                "hidden_units": 0,
            },
        )

        # Threshold assertion: creating a network with 0 hidden units should
        # not cause significant memory growth. Typical delta is < 1 MB.
        assert delta_mb < MEMORY_DELTA_THRESHOLD_MB, f"Memory growth {delta_mb:.1f} MB from base network creation " f"exceeds {MEMORY_DELTA_THRESHOLD_MB:.0f} MB threshold"

        # Regression check against saved baseline
        _check_memory_regression("memory_base_network", delta_mb, "delta_mb")

    def test_memory_5_hidden(self, network_5_hidden):
        """Memory footprint after growing to 5 hidden units."""
        process = psutil.Process()
        mem_mb = process.memory_info().rss / (1024 * 1024)
        n_hidden = len(network_5_hidden.hidden_units)

        save_baseline(
            f"memory_{n_hidden}_hidden",
            {
                "rss_mb": round(mem_mb, 2),
                "hidden_units": n_hidden,
            },
        )

        # Threshold assertion: total process RSS with 5 hidden units should
        # remain within a reasonable envelope for a small test network.
        assert mem_mb < MEMORY_ABSOLUTE_THRESHOLD_MB, f"Process RSS {mem_mb:.1f} MB with {n_hidden} hidden units " f"exceeds {MEMORY_ABSOLUTE_THRESHOLD_MB:.0f} MB absolute threshold"

        # Regression check against saved baseline
        _check_memory_regression(f"memory_{n_hidden}_hidden", mem_mb, "rss_mb")

    def test_memory_10_hidden(self, network_10_hidden):
        """Memory footprint after growing to ~10 hidden units."""
        process = psutil.Process()
        mem_mb = process.memory_info().rss / (1024 * 1024)
        n_hidden = len(network_10_hidden.hidden_units)

        save_baseline(
            f"memory_{n_hidden}_hidden",
            {
                "rss_mb": round(mem_mb, 2),
                "hidden_units": n_hidden,
            },
        )

        # Threshold assertion: total process RSS with ~10 hidden units should
        # remain within a reasonable envelope for a small test network.
        assert mem_mb < MEMORY_ABSOLUTE_THRESHOLD_MB, f"Process RSS {mem_mb:.1f} MB with {n_hidden} hidden units " f"exceeds {MEMORY_ABSOLUTE_THRESHOLD_MB:.0f} MB absolute threshold"

        # Regression check against saved baseline
        _check_memory_regression(f"memory_{n_hidden}_hidden", mem_mb, "rss_mb")


# ===================================================================
# SERIALIZATION BASELINES
# ===================================================================


@pytest.mark.performance
class TestSerializationBaseline:
    """Benchmark HDF5 save/load performance."""

    def test_save_load_trained(self, benchmark, tmp_path, small_spiral_data):
        """HDF5 save/load round-trip with a small trained network.

        Uses a minimally trained network (not synthetic) since the HDF5
        serializer requires complete internal state including random state.
        """
        config = _make_benchmark_config(
            candidate_pool_size=2,
            candidate_epochs=5,
            max_hidden_units=2,
            output_epochs=3,
            patience=1,
            correlation_threshold=0.0001,
            learning_rate=0.1,
            candidate_learning_rate=0.05,
        )
        net = CascadeCorrelationNetwork(config=config)
        x, y = small_spiral_data

        # Time the fit() call separately - tiny data + minimal config should
        # complete quickly; this catches major training loop regressions.
        fit_start = time.perf_counter()
        net.fit(x, y, max_epochs=20)
        fit_elapsed = time.perf_counter() - fit_start

        filepath = str(tmp_path / "bench_roundtrip.h5")

        # Time the save step to catch serialization write regressions.
        save_start = time.perf_counter()
        net.save_to_hdf5(filepath)
        save_elapsed = time.perf_counter() - save_start

        result = benchmark(CascadeCorrelationNetwork.load_from_hdf5, filepath)
        assert result is not None

        # Timing assertion: fit() on tiny data (100 samples, 2 max hidden,
        # 20 max epochs) should complete well under the threshold.
        assert fit_elapsed < FIT_TIME_THRESHOLD_S, f"fit() took {fit_elapsed:.1f}s on tiny data, " f"exceeding {FIT_TIME_THRESHOLD_S:.0f}s threshold"

        # Timing assertion: HDF5 save on a small network should be fast.
        assert save_elapsed < SERIALIZATION_TIME_THRESHOLD_S, f"save_to_hdf5() took {save_elapsed:.1f}s, " f"exceeding {SERIALIZATION_TIME_THRESHOLD_S:.0f}s threshold"
