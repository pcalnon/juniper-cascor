"""
Phase 2, Step 2.5: Weight Update and Autograd Overhead Benchmarks.

Isolates the cost of _update_weights_and_bias() autograd pattern:
clone().detach().requires_grad_(True) -> forward -> backward -> gradient apply.

This is called candidate_pool_size * candidate_epochs times per growth cycle
(e.g., 16 x 100 = 1,600 times). Understanding this overhead is critical.

Run: pytest tests/performance/test_micro_autograd.py --run-performance -v
"""

import psutil
import pytest
import torch

from candidate_unit.candidate_unit import CandidateUnit


def _make_candidate(input_size):
    return CandidateUnit(
        CandidateUnit__input_size=input_size,
        CandidateUnit__activation_function=torch.nn.Tanh(),
    )


# ===================================================================
# AUTOGRAD OVERHEAD PER ITERATION
# ===================================================================


@pytest.mark.performance
class TestAutogradOverhead:
    """Measure per-iteration autograd graph construction and teardown cost."""

    @pytest.mark.parametrize("input_size", [2, 10, 50])
    def test_single_autograd_cycle(self, benchmark, input_size):
        """Cost of one clone->detach->requires_grad->forward->backward->apply cycle."""
        torch.manual_seed(42)
        weights = torch.randn(input_size) * 0.1
        bias = torch.tensor(0.01)
        x = torch.randn(100, input_size)
        activation = torch.nn.Tanh()

        def run():
            w = weights.clone().detach().requires_grad_(True)
            b = bias.clone().detach().requires_grad_(True)
            out = activation(torch.sum(x * w, dim=1) + b)
            loss = out.sum()
            loss.backward()
            with torch.no_grad():
                weights.sub_(0.005 * w.grad)
                bias.sub_(0.005 * b.grad)

        benchmark(run)

    def test_forward_only_no_grad(self, benchmark):
        """Forward pass without autograd (baseline for comparison)."""
        torch.manual_seed(42)
        weights = torch.randn(2) * 0.1
        bias = torch.tensor(0.01)
        x = torch.randn(100, 2)
        activation = torch.nn.Tanh()

        def run():
            with torch.no_grad():
                return activation(torch.sum(x * weights, dim=1) + bias)

        result = benchmark(run)
        assert result is not None


# ===================================================================
# AUTOGRAD MEMORY GROWTH
# ===================================================================


@pytest.mark.performance
class TestAutogradMemoryGrowth:
    """Check for memory leaks from repeated autograd graph construction."""

    @pytest.mark.parametrize("iterations", [100, 500, 1000])
    def test_memory_over_iterations(self, iterations):
        """Run N autograd cycles and check for memory growth."""
        torch.manual_seed(42)
        weights = torch.randn(10) * 0.1
        bias = torch.tensor(0.01)
        x = torch.randn(100, 10)
        activation = torch.nn.Tanh()

        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        for _ in range(iterations):
            w = weights.clone().detach().requires_grad_(True)
            b = bias.clone().detach().requires_grad_(True)
            out = activation(torch.sum(x * w, dim=1) + b)
            loss = out.sum()
            loss.backward()
            with torch.no_grad():
                weights.sub_(0.005 * w.grad)
                bias.sub_(0.005 * b.grad)

        mem_after = process.memory_info().rss / (1024 * 1024)
        growth_mb = mem_after - mem_before

        # Memory growth should be minimal (< 10MB for any iteration count)
        # since each autograd graph is freed after backward()
        assert growth_mb < 10.0, f"Memory grew {growth_mb:.1f}MB over {iterations} iterations -- possible graph leak"


# ===================================================================
# COMPARISON: AUTOGRAD vs NO-GRAD
# ===================================================================


@pytest.mark.performance
class TestAutogradVsNoGrad:
    """Quantify the overhead of autograd vs pure forward computation."""

    def test_with_autograd(self, benchmark):
        """Full autograd cycle: clone, detach, requires_grad, forward, backward."""
        torch.manual_seed(42)
        weights = torch.randn(10) * 0.1
        bias = torch.tensor(0.01)
        x = torch.randn(100, 10)
        activation = torch.nn.Tanh()

        def run():
            w = weights.clone().detach().requires_grad_(True)
            b = bias.clone().detach().requires_grad_(True)
            out = activation(torch.sum(x * w, dim=1) + b)
            loss = out.sum()
            loss.backward()

        benchmark(run)

    def test_without_autograd(self, benchmark):
        """Same computation under torch.no_grad() (no graph, no backward)."""
        torch.manual_seed(42)
        weights = torch.randn(10) * 0.1
        bias = torch.tensor(0.01)
        x = torch.randn(100, 10)
        activation = torch.nn.Tanh()

        def run():
            with torch.no_grad():
                activation(torch.sum(x * weights, dim=1) + bias)

        benchmark(run)
