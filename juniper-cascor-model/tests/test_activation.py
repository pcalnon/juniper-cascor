"""Regression tests for worker-facing activation wrappers."""

from __future__ import annotations

import pickle

import pytest

torch = pytest.importorskip("torch")


def test_titlecase_activation_wrapper_survives_pickle_roundtrip():
    from utils.activation import ActivationWithDerivative

    wrapper = ActivationWithDerivative(torch.nn.Tanh())
    values = torch.tensor([-1.0, 0.0, 1.0])

    restored = pickle.loads(pickle.dumps(wrapper))

    assert repr(restored) == "ActivationWithDerivative(Tanh)"
    assert torch.allclose(restored(values), torch.tanh(values))
    assert torch.allclose(restored(values, derivative=True), 1.0 - torch.tanh(values) ** 2)


def test_softmax_wrapper_handles_one_dimensional_candidate_outputs():
    from utils.activation import ActivationWithDerivative

    wrapper = ActivationWithDerivative(torch.nn.Softmax(dim=1))
    values = torch.tensor([1.0, 2.0, 3.0])

    result = wrapper(values)

    assert result.shape == values.shape
    assert torch.allclose(result.sum(), torch.tensor(1.0))
    assert result.argmax().item() == 2


def test_unknown_activation_deserialization_reports_activation_name():
    from utils.activation import ActivationWithDerivative

    wrapper = object.__new__(ActivationWithDerivative)

    with pytest.raises(ValueError, match="DefinitelyNotAnActivation"):
        wrapper.__setstate__({"_activation_name": "DefinitelyNotAnActivation"})
