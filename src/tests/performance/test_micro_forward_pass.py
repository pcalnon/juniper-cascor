"""
Phase 2, Step 2.2: Forward Pass Scaling Benchmarks.

Answers key question: Does forward pass scale linearly or quadratically
with hidden unit count due to the cascading concatenation pattern?

Run: pytest tests/performance/test_micro_forward_pass.py --run-performance -v
"""

import pytest
import torch

from .conftest import _build_network_with_hidden_units


@pytest.mark.performance
class TestForwardPassScaling:
    """Parametrized forward pass benchmarks across hidden_units x samples x input_size."""

    @pytest.mark.parametrize("n_hidden", [0, 5, 10, 20, 50])
    def test_hidden_unit_scaling(self, benchmark, n_hidden):
        """Forward pass time vs hidden unit depth (100 samples, input_size=2)."""
        net = _build_network_with_hidden_units(n_hidden, input_size=2, output_size=2)
        x = torch.randn(100, 2)

        result = benchmark(net.forward, x)
        assert result.shape == (100, 2)

    @pytest.mark.parametrize("n_samples", [50, 200, 1000])
    def test_sample_count_scaling(self, benchmark, n_samples):
        """Forward pass time vs sample count (10 hidden units, input_size=2)."""
        net = _build_network_with_hidden_units(10, input_size=2, output_size=2)
        x = torch.randn(n_samples, 2)

        result = benchmark(net.forward, x)
        assert result.shape == (n_samples, 2)

    @pytest.mark.parametrize("input_size", [2, 10, 50])
    def test_input_size_scaling(self, benchmark, input_size):
        """Forward pass time vs input feature dimensionality (10 hidden, 100 samples)."""
        net = _build_network_with_hidden_units(10, input_size=input_size, output_size=2)
        x = torch.randn(100, input_size)

        result = benchmark(net.forward, x)
        assert result.shape == (100, 2)


@pytest.mark.performance
class TestForwardPassMemory:
    """Measure memory allocation behavior of the forward pass."""

    @pytest.mark.parametrize("n_hidden", [0, 10, 20, 50])
    def test_forward_no_grad_allocation(self, benchmark, n_hidden):
        """Forward pass under torch.no_grad() (inference mode, no graph)."""
        net = _build_network_with_hidden_units(n_hidden, input_size=2, output_size=2)
        x = torch.randn(100, 2)

        def run():
            with torch.no_grad():
                return net.forward(x)

        result = benchmark(run)
        assert result.shape == (100, 2)
