#!/usr/bin/env python
"""Regression tests for init_output_weights crash paths."""

from types import SimpleNamespace

import pytest
import torch

from candidate_unit.candidate_unit import CandidateTrainingResult
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig


def _make_network(init_output_weights: str = "zero") -> CascadeCorrelationNetwork:
    """Create a small network for deterministic regression checks."""
    config = CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=2,
        candidate_pool_size=2,
        candidate_epochs=2,
        output_epochs=2,
        max_hidden_units=2,
        init_output_weights=init_output_weights,
    )
    return CascadeCorrelationNetwork(config=config)


def _make_candidate(input_size: int, correlation: float = 0.5) -> SimpleNamespace:
    """Build a lightweight candidate object with required attributes."""
    return SimpleNamespace(
        weights=torch.randn(input_size),
        bias=torch.tensor(0.1),
        correlation=correlation,
    )


@pytest.mark.unit
def test_add_unit_zero_init_preserves_existing_weights_without_runtime_error():
    """`add_unit()` should not crash when zero-init mode is enabled."""
    network = _make_network(init_output_weights="zero")
    x = torch.randn(8, 2)
    candidate = _make_candidate(input_size=2)
    old_output_weights = network.output_weights.clone().detach()

    network.add_unit(candidate, x)

    assert network.output_weights.shape == (3, 2)
    assert torch.allclose(network.output_weights[:2, :].detach(), old_output_weights)


@pytest.mark.unit
def test_add_units_as_layer_zero_init_preserves_existing_weights_without_runtime_error():
    """`add_units_as_layer()` should not crash when zero-init mode is enabled."""
    network = _make_network(init_output_weights="zero")
    x = torch.randn(8, 2)
    candidate = _make_candidate(input_size=2)
    old_output_weights = network.output_weights.clone().detach()
    candidate_result = CandidateTrainingResult(
        candidate_id=0,
        candidate_uuid="regression-candidate",
        correlation=candidate.correlation,
        candidate=candidate,
        success=True,
    )

    network.add_units_as_layer([candidate_result], x)

    assert len(network.hidden_units) == 1
    assert network.output_weights.shape == (3, 2)
    assert torch.allclose(network.output_weights[:2, :].detach(), old_output_weights)
