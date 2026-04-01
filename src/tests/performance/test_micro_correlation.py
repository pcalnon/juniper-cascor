"""
Phase 2, Step 2.3: Correlation Calculation Benchmarks.

Measures _calculate_correlation() scaling with sample count and output dimensionality.

Run: pytest tests/performance/test_micro_correlation.py --run-performance -v
"""

import pytest
import torch

from candidate_unit.candidate_unit import CandidateUnit


def _make_candidate(input_size=2):
    return CandidateUnit(
        CandidateUnit__input_size=input_size,
        CandidateUnit__activation_function=torch.nn.Tanh(),
    )


# ===================================================================
# SAMPLE COUNT SCALING
# ===================================================================


@pytest.mark.performance
class TestCorrelationSampleScaling:
    """Correlation computation time vs sample count (output_size=2)."""

    @pytest.mark.parametrize("n_samples", [50, 200, 1000, 5000])
    def test_sample_scaling(self, benchmark, n_samples):
        torch.manual_seed(42)
        output = torch.randn(n_samples, 2)
        residual = torch.randn(n_samples, 2)
        candidate = _make_candidate()

        result = benchmark(candidate._calculate_correlation, output, residual)
        assert result is not None


# ===================================================================
# OUTPUT DIMENSIONALITY SCALING
# ===================================================================


@pytest.mark.performance
class TestCorrelationOutputScaling:
    """Correlation computation time vs output dimensionality (1000 samples)."""

    @pytest.mark.parametrize("output_size", [1, 2, 5, 10])
    def test_output_size_scaling(self, benchmark, output_size):
        torch.manual_seed(42)
        output = torch.randn(1000, output_size)
        residual = torch.randn(1000, output_size)
        candidate = _make_candidate()

        result = benchmark(candidate._calculate_correlation, output, residual)
        assert result is not None
