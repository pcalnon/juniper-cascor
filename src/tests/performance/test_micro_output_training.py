"""
Phase 2, Step 2.4: Output Layer Training Benchmarks.

Answers key question: How does output training time scale as the network grows?
The output layer input width grows linearly with each hidden unit addition.

Run: pytest tests/performance/test_micro_output_training.py --run-performance -v
"""

import pytest
import torch

from .conftest import _build_network_with_hidden_units

# ===================================================================
# HIDDEN UNIT SCALING
# ===================================================================


@pytest.mark.performance
class TestOutputTrainingHiddenScaling:
    """Output layer training time vs network depth (25 epochs, 100 samples)."""

    @pytest.mark.parametrize("n_hidden", [0, 5, 10, 20, 50])
    def test_hidden_scaling(self, benchmark, n_hidden):
        net = _build_network_with_hidden_units(n_hidden, input_size=2, output_size=2)
        torch.manual_seed(42)
        x = torch.randn(100, 2)
        y = torch.cat(
            [
                torch.tensor([[1, 0]] * 50),
                torch.tensor([[0, 1]] * 50),
            ],
            dim=0,
        ).float()

        result = benchmark.pedantic(net.train_output_layer, args=(x, y, 25), rounds=3, warmup_rounds=1)
        assert isinstance(result, float)


# ===================================================================
# EPOCH COUNT SCALING
# ===================================================================


@pytest.mark.performance
class TestOutputTrainingEpochScaling:
    """Output layer training time vs epoch count (10 hidden, 100 samples)."""

    @pytest.mark.parametrize("epochs", [10, 50, 100])
    def test_epoch_scaling(self, benchmark, epochs):
        net = _build_network_with_hidden_units(10, input_size=2, output_size=2)
        torch.manual_seed(42)
        x = torch.randn(100, 2)
        y = torch.cat(
            [
                torch.tensor([[1, 0]] * 50),
                torch.tensor([[0, 1]] * 50),
            ],
            dim=0,
        ).float()

        result = benchmark.pedantic(net.train_output_layer, args=(x, y, epochs), rounds=3, warmup_rounds=1)
        assert isinstance(result, float)


# ===================================================================
# SAMPLE COUNT SCALING
# ===================================================================


@pytest.mark.performance
class TestOutputTrainingSampleScaling:
    """Output layer training time vs sample count (10 hidden, 25 epochs)."""

    @pytest.mark.parametrize("n_samples", [50, 200, 1000])
    def test_sample_scaling(self, benchmark, n_samples):
        net = _build_network_with_hidden_units(10, input_size=2, output_size=2)
        torch.manual_seed(42)
        x = torch.randn(n_samples, 2)
        y = torch.cat(
            [
                torch.tensor([[1, 0]] * (n_samples // 2)),
                torch.tensor([[0, 1]] * (n_samples // 2)),
            ],
            dim=0,
        ).float()

        result = benchmark.pedantic(net.train_output_layer, args=(x, y, 25), rounds=3, warmup_rounds=1)
        assert isinstance(result, float)
