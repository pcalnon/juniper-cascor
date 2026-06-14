"""Behavioral coverage for candidate-core paths used by distributed workers."""

import pickle

import pytest

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch required for candidate-core")


@requires_torch
def test_activation_wrapper_round_trips_titlecase_softmax_through_pickle():
    """TitleCase module activations must stay usable after multiprocessing serialization."""
    from utils.activation import ActivationWithDerivative

    activation = ActivationWithDerivative(torch.nn.Softmax(dim=1))
    restored = pickle.loads(pickle.dumps(activation))

    x = torch.tensor([1.0, 2.0, 3.0])
    output = restored(x)

    assert torch.allclose(output, torch.softmax(x, dim=-1))
    assert torch.isclose(output.sum(), torch.tensor(1.0))


@requires_torch
def test_candidate_unit_accepts_worker_activation_tuple_contract():
    """juniper-cascor-worker passes (activation, derivative) tuples into CandidateUnit."""
    from candidate_unit.candidate_unit import CandidateUnit

    candidate = CandidateUnit(
        CandidateUnit__activation_function=(torch.sigmoid, lambda x: torch.sigmoid(x) * (1.0 - torch.sigmoid(x))),
        CandidateUnit__input_size=2,
        CandidateUnit__output_size=1,
        CandidateUnit__display_frequency=0,
        CandidateUnit__status_frequency=0,
        CandidateUnit__random_seed=7,
        CandidateUnit__sequence_max_value=2,
        CandidateUnit__random_value_scale=0.01,
        CandidateUnit__log_level_name="CRITICAL",
    )
    candidate.weights = torch.tensor([0.5, -0.25])
    candidate.bias = torch.tensor([0.1])
    x = torch.tensor([[2.0, 4.0], [1.0, -1.0]], dtype=torch.float32)
    expected = torch.sigmoid(torch.sum(x * candidate.weights, dim=1) + candidate.bias)

    restored = pickle.loads(pickle.dumps(candidate))

    assert torch.allclose(candidate.forward(x), expected)
    assert torch.allclose(restored.forward(x), expected)


@requires_torch
def test_dataset_save_load_round_trips_tensors(tmp_path):
    """The torch.save/torch.load helper pair should round-trip real tensor payloads."""
    from utils.utils import load_dataset, save_dataset

    x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    dataset_path = tmp_path / "candidate_dataset.pt"

    save_dataset(x, y, dataset_path)
    loaded_x, loaded_y = load_dataset(dataset_path)

    assert torch.equal(loaded_x, x)
    assert torch.equal(loaded_y, y)


@requires_torch
def test_candidate_unit_round_trips_through_pickle_and_preserves_forward_output():
    """CandidateUnit must remain executable after the worker serializes it."""
    from candidate_unit.candidate_unit import CandidateUnit

    candidate = CandidateUnit(
        CandidateUnit__activation_function=torch.nn.Tanh(),
        CandidateUnit__input_size=2,
        CandidateUnit__output_size=1,
        CandidateUnit__display_frequency=0,
        CandidateUnit__status_frequency=0,
        CandidateUnit__random_seed=7,
        CandidateUnit__sequence_max_value=2,
        CandidateUnit__random_value_scale=0.01,
        CandidateUnit__log_level_name="CRITICAL",
        CandidateUnit__candidate_index=3,
    )
    candidate.weights = torch.tensor([0.5, -0.25])
    candidate.bias = torch.tensor([0.1])
    x = torch.tensor([[2.0, 4.0], [1.0, -1.0]], dtype=torch.float32)

    expected = candidate.forward(x)
    restored = pickle.loads(pickle.dumps(candidate))

    assert restored.logger is not None
    assert torch.allclose(restored.forward(x), expected)
