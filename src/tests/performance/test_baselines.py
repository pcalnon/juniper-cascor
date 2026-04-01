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

import numpy as np
import psutil
import pytest
import torch

from candidate_unit.candidate_unit import CandidateUnit
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

from .conftest import BenchmarkTimer, _make_benchmark_config, save_baseline

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
        net.fit(x, y, max_epochs=20)

        filepath = str(tmp_path / "bench_roundtrip.h5")
        net.save_to_hdf5(filepath)

        result = benchmark(CascadeCorrelationNetwork.load_from_hdf5, filepath)
        assert result is not None
