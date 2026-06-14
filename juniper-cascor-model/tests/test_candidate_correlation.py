"""Regression tests for candidate correlation selection metrics."""

import pytest

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch required for candidate-core")


@pytest.fixture
def candidate_unit():
    from candidate_unit.candidate_unit import CandidateUnit

    return CandidateUnit(
        CandidateUnit__activation_function=torch.nn.Tanh(),
        CandidateUnit__input_size=1,
        CandidateUnit__output_size=1,
        CandidateUnit__display_frequency=0,
        CandidateUnit__status_frequency=0,
        CandidateUnit__random_seed=11,
        CandidateUnit__sequence_max_value=2,
        CandidateUnit__random_value_scale=0.01,
        CandidateUnit__log_level_name="CRITICAL",
    )


@requires_torch
def test_correlation_validation_rejects_mismatched_batch_sizes(candidate_unit):
    output = torch.zeros(4, 1)
    residual_error = torch.zeros(3, 1)

    with pytest.raises(ValueError, match="same batch size"):
        candidate_unit._validate_correlation_params(output=output, residual_error=residual_error)


@requires_torch
def test_correlation_validation_rejects_non_tensor_inputs(candidate_unit):
    with pytest.raises(TypeError, match="torch.Tensor"):
        candidate_unit._validate_correlation_params(output=[0.0, 1.0], residual_error=torch.zeros(2))


@requires_torch
def test_calculate_correlation_returns_one_for_perfect_correlation(candidate_unit):
    output = torch.tensor([1.0, 2.0, 3.0, 4.0])
    residual_error = output.clone()

    correlation, norm_output, norm_error, numerator, denominator = candidate_unit._calculate_correlation(
        output=output,
        residual_error=residual_error,
    )

    assert correlation == pytest.approx(1.0)
    assert torch.allclose(norm_output, norm_error)
    assert numerator > 0.0
    assert denominator > 0.0


@requires_torch
def test_calculate_correlation_returns_zero_for_zero_variance(candidate_unit):
    output = torch.ones(4)
    residual_error = torch.ones(4)

    correlation, norm_output, norm_error, numerator, denominator = candidate_unit._calculate_correlation(
        output=output,
        residual_error=residual_error,
    )

    assert correlation == 0.0
    assert torch.equal(norm_output, torch.zeros_like(output))
    assert torch.equal(norm_error, torch.zeros_like(residual_error))
    assert numerator == 0.0
    assert denominator == pytest.approx(1e-8)
