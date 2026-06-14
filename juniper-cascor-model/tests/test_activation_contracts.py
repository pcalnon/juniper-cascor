"""Regression tests for picklable activation contracts used by workers."""

from __future__ import annotations

import pickle

import pytest

torch = pytest.importorskip("torch")

from utils.activation import ActivationWithDerivative  # noqa: E402


def test_titlecase_tanh_pickle_round_trip_preserves_outputs():
    activation = ActivationWithDerivative(torch.nn.Tanh())
    restored = pickle.loads(pickle.dumps(activation))
    x = torch.linspace(-1.0, 1.0, steps=5)

    assert restored._activation_name == "Tanh"
    assert torch.allclose(restored(x), activation(x))
    assert torch.allclose(restored(x, derivative=True), 1.0 - torch.tanh(x) ** 2)


def test_legacy_worker_activation_tuple_keeps_callable_activation():
    activation = ActivationWithDerivative((torch.tanh, lambda x: 1.0 - torch.tanh(x) ** 2))
    x = torch.tensor([0.5])

    assert activation._activation_name == "tanh"
    assert torch.allclose(activation(x), torch.tanh(x))


def test_softmax_module_accepts_one_dimensional_pre_activation_vector():
    activation = ActivationWithDerivative(torch.nn.Softmax(dim=1))
    x = torch.tensor([1.0, 2.0, 3.0])

    y = activation(x)

    assert torch.isfinite(y).all()
    assert pytest.approx(y.sum().item(), rel=1e-6) == 1.0


def test_unknown_activation_state_fails_closed_on_deserialization():
    activation = ActivationWithDerivative(torch.tanh)
    state = activation.__getstate__()
    state["_activation_name"] = "NotARealActivation"
    restored = ActivationWithDerivative.__new__(ActivationWithDerivative)

    with pytest.raises(ValueError, match="Unrecognized activation function name"):
        restored.__setstate__(state)


def test_invalid_legacy_activation_tuple_rejected():
    with pytest.raises(TypeError, match="activation tuple must start with a callable"):
        ActivationWithDerivative((None, None))
