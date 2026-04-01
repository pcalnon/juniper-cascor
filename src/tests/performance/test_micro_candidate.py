"""
Phase 2, Step 2.1: Candidate Training Micro-benchmarks.

Confirms expected O(samples x input_size x epochs) scaling.
Also measures activation function overhead differences.

Run: pytest tests/performance/test_micro_candidate.py --run-performance -v
"""

import pytest
import torch

from candidate_unit.candidate_unit import CandidateUnit


def _make_candidate(input_size, activation_fn):
    """Create a CandidateUnit with correct constructor parameters."""
    return CandidateUnit(
        CandidateUnit__input_size=input_size,
        CandidateUnit__activation_function=activation_fn,
    )


def _make_data(input_size, n_samples, output_size=2):
    """Create synthetic x and residual_error tensors."""
    torch.manual_seed(42)
    x = torch.randn(n_samples, input_size)
    residual = torch.randn(n_samples, output_size)
    return x, residual


# ===================================================================
# EPOCH SCALING
# ===================================================================


@pytest.mark.performance
class TestCandidateEpochScaling:
    """Candidate training time vs epoch count (input_size=2, 100 samples)."""

    @pytest.mark.parametrize("epochs", [10, 50, 100, 200])
    def test_epoch_scaling(self, benchmark, epochs):
        x, residual = _make_data(input_size=2, n_samples=100)

        def run():
            torch.manual_seed(42)
            c = _make_candidate(2, torch.nn.Tanh())
            return c.train_detailed(x=x, epochs=epochs, residual_error=residual, learning_rate=0.005, display_frequency=0)

        result = benchmark.pedantic(run, rounds=3, warmup_rounds=1)
        assert result is not None
        assert result.epochs_completed == epochs


# ===================================================================
# INPUT SIZE SCALING
# ===================================================================


@pytest.mark.performance
class TestCandidateInputSizeScaling:
    """Candidate training time vs input feature dimensionality (50 epochs, 100 samples)."""

    @pytest.mark.parametrize("input_size", [2, 10, 50])
    def test_input_size_scaling(self, benchmark, input_size):
        x, residual = _make_data(input_size=input_size, n_samples=100)

        def run():
            torch.manual_seed(42)
            c = _make_candidate(input_size, torch.nn.Tanh())
            return c.train_detailed(x=x, epochs=50, residual_error=residual, learning_rate=0.005, display_frequency=0)

        result = benchmark.pedantic(run, rounds=3, warmup_rounds=1)
        assert result is not None


# ===================================================================
# SAMPLE COUNT SCALING
# ===================================================================


@pytest.mark.performance
class TestCandidateSampleScaling:
    """Candidate training time vs sample count (input_size=2, 50 epochs)."""

    @pytest.mark.parametrize("n_samples", [50, 200, 1000])
    def test_sample_scaling(self, benchmark, n_samples):
        x, residual = _make_data(input_size=2, n_samples=n_samples)

        def run():
            torch.manual_seed(42)
            c = _make_candidate(2, torch.nn.Tanh())
            return c.train_detailed(x=x, epochs=50, residual_error=residual, learning_rate=0.005, display_frequency=0)

        result = benchmark.pedantic(run, rounds=3, warmup_rounds=1)
        assert result is not None


# ===================================================================
# ACTIVATION FUNCTION COMPARISON
# ===================================================================


@pytest.mark.performance
class TestCandidateActivationComparison:
    """Candidate training time across activation functions (input_size=2, 50 epochs, 100 samples)."""

    @pytest.mark.parametrize(
        "activation,name",
        [
            (torch.nn.Tanh(), "tanh"),
            (torch.nn.Sigmoid(), "sigmoid"),
            (torch.nn.ReLU(), "relu"),
        ],
        ids=["tanh", "sigmoid", "relu"],
    )
    def test_activation_comparison(self, benchmark, activation, name):
        x, residual = _make_data(input_size=2, n_samples=100)

        def run():
            torch.manual_seed(42)
            c = _make_candidate(2, activation)
            return c.train_detailed(x=x, epochs=50, residual_error=residual, learning_rate=0.005, display_frequency=0)

        result = benchmark.pedantic(run, rounds=3, warmup_rounds=1)
        assert result is not None
