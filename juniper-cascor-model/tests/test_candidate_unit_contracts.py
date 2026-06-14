"""Regression tests for CandidateUnit worker/runtime contracts."""

from __future__ import annotations

import pickle

import pytest

torch = pytest.importorskip("torch")

from candidate_unit.candidate_unit import CandidateUnit  # noqa: E402
from log_config.logger.logger import Logger  # noqa: E402


@pytest.fixture
def tiny_training_batch():
    x = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    residual_error = torch.tensor([[-0.5], [0.25], [0.25], [0.5]], dtype=torch.float32)
    return x, residual_error


def make_candidate(**overrides):
    params = {
        "CandidateUnit__input_size": 2,
        "CandidateUnit__output_size": 1,
        "CandidateUnit__random_seed": 17,
        "CandidateUnit__candidate_index": 4,
        "CandidateUnit__epochs_max": 3,
        "CandidateUnit__early_stopping": False,
        "CandidateUnit__display_frequency": 10_000,
        "CandidateUnit__status_frequency": 10_000,
        "CandidateUnit__log_level_name": "FATAL",
    }
    params.update(overrides)
    return CandidateUnit(**params)


def test_train_preserves_float_contract_and_records_detailed_result(tiny_training_batch):
    x, residual_error = tiny_training_batch
    candidate = make_candidate(CandidateUnit__candidate_index=8)
    starting_weights = candidate.weights.clone()
    starting_bias = candidate.bias.clone()

    correlation = candidate.train(x=x, residual_error=residual_error, epochs=3, learning_rate=0.05)

    assert isinstance(correlation, float)
    assert 0.0 <= correlation <= 1.0
    assert candidate.correlation == pytest.approx(correlation)
    assert hasattr(candidate, "last_training_result")
    result = candidate.last_training_result
    assert result.candidate_id == 8
    assert result.candidate_uuid == candidate.uuid
    assert result.success is True
    assert result.epochs_completed == 3
    assert len(result.all_correlations) == 1
    assert not torch.allclose(candidate.weights, starting_weights)
    assert not torch.allclose(candidate.bias, starting_bias)


def test_pickle_round_trip_preserves_forward_output_and_recreates_runtime_state(tiny_training_batch):
    x, residual_error = tiny_training_batch
    candidate = make_candidate(
        CandidateUnit__activation_function=torch.nn.Tanh(),
        CandidateUnit__candidate_index=11,
        CandidateUnit__random_seed=23,
    )
    expected_output = candidate.forward(x)

    restored = pickle.loads(pickle.dumps(candidate))

    assert restored.logger is Logger
    assert restored.candidate_index == 11
    assert torch.allclose(restored.forward(x), expected_output)
    correlation = restored.train(x=x, residual_error=residual_error, epochs=2, learning_rate=0.05)
    assert isinstance(correlation, float)
    assert restored.last_training_result.epochs_completed == 2


def test_multi_output_correlation_selects_strongest_residual_column():
    candidate = make_candidate()
    output = torch.tensor([1.0, 2.0, 3.0, 4.0])
    residual_error = torch.tensor(
        [
            [1.0, 2.0],
            [-1.0, 4.0],
            [1.0, 6.0],
            [-1.0, 8.0],
        ],
        dtype=torch.float32,
    )

    result = candidate._get_correlations(output=output, residual_error=residual_error)

    assert result.best_corr_idx == 1
    assert result.correlation == pytest.approx(1.0)
    assert len(result.all_correlations) == 2


@pytest.mark.parametrize(
    ("output", "residual_error", "expected_error", "match"),
    [
        (torch.ones(3), torch.ones(2), ValueError, "same batch size"),
        (None, torch.ones(2), ValueError, "must not be None"),
        (torch.ones(2, 1, 1), torch.ones(2, 1, 1), ValueError, "at most two dimensions"),
        (torch.ones(2, 1), torch.ones(2, 2), ValueError, "same number of features"),
    ],
)
def test_validate_correlation_params_rejects_invalid_inputs(output, residual_error, expected_error, match):
    candidate = make_candidate()

    with pytest.raises(expected_error, match=match):
        candidate._validate_correlation_params(
            output=output,
            residual_error=residual_error,
        )
