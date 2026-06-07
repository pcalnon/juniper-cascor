#!/usr/bin/env python
"""Worker-facing contracts for the candidate core (CandidateUnit + activation).

These behaviors exist because ``CandidateUnit`` / ``ActivationWithDerivative`` are
extracted verbatim into the ``juniper-cascor-core`` PyPI package and consumed by the
distributed ``juniper-cascor-worker`` over a JSON wire. The worker reconstructs
candidates from JSON-decoded payloads, where:

* integer-valued bounds (``random_max_value`` / ``sequence_max_value`` /
  ``random_seed`` / ``candidate_index``) arrive as ``float`` (JSON has no int type),
  yet they feed ``random.randint()`` / ``range()`` which reject floats; and
* the worker's activation resolver historically returned a legacy
  ``(activation, derivative)`` tuple.

cascor's own *local* path never hits these shapes (it passes ints and callables), so
cascor's other tests only exercise the passthrough direction. This file is the
source-side guard for the worker-facing normalization that the package depends on —
mirroring the package's ``tests/test_smoke.py`` contracts so a cascor-side regression
is caught here, not only after a manual re-extraction. Do not "simplify" the
normalization away: it is the cross-service contract, not dead defensiveness.
"""

import pickle

import pytest
import torch

from candidate_unit.candidate_unit import CandidateUnit
from utils.activation import ActivationWithDerivative


# ---------------------------------------------------------------------------
# ActivationWithDerivative — accept the worker's legacy (activation, derivative) tuple
# ---------------------------------------------------------------------------
class TestActivationWorkerTupleContract:
    @pytest.mark.unit
    def test_normalizes_worker_activation_tuple_to_callable(self):
        awd = ActivationWithDerivative((torch.tanh, lambda x: 1.0 - torch.tanh(x) ** 2))
        assert awd.activation_fn is torch.tanh, "tuple must normalize to its first (callable) element"

    @pytest.mark.unit
    def test_passes_through_plain_callable(self):
        awd = ActivationWithDerivative(torch.tanh)
        assert awd.activation_fn is torch.tanh

    @pytest.mark.unit
    def test_rejects_tuple_without_leading_callable(self):
        with pytest.raises(TypeError):
            ActivationWithDerivative((0.1, 0.2))


# ---------------------------------------------------------------------------
# CandidateUnit — normalize JSON-decoded integer-valued bounds (float -> int)
# ---------------------------------------------------------------------------
class TestCandidateUnitBoundCoercion:
    @pytest.mark.unit
    def test_float_valued_integer_bounds_are_coerced(self):
        """The remote worker's JSON payload decodes these as floats; CandidateUnit
        must store them as int before they reach random.randint()/range()."""
        candidate = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__output_size=1,
            CandidateUnit__candidate_index=2.0,
            CandidateUnit__random_seed=1.0,
            CandidateUnit__random_max_value=1.0,
            CandidateUnit__sequence_max_value=100.0,
        )
        assert candidate.candidate_index == 2 and type(candidate.candidate_index) is int
        assert candidate.random_seed == 1 and type(candidate.random_seed) is int
        assert candidate.random_max_value == 1 and type(candidate.random_max_value) is int
        assert candidate.sequence_max_value == 100 and type(candidate.sequence_max_value) is int

    @pytest.mark.unit
    def test_bool_bounds_rejected(self):
        """bool is an int subclass but a programming error here — reject it loudly."""
        with pytest.raises(TypeError):
            CandidateUnit(
                CandidateUnit__input_size=2,
                CandidateUnit__output_size=1,
                CandidateUnit__random_max_value=True,
            )

    @pytest.mark.unit
    def test_non_integer_float_bounds_rejected(self):
        with pytest.raises(ValueError):
            CandidateUnit(
                CandidateUnit__input_size=2,
                CandidateUnit__output_size=1,
                CandidateUnit__sequence_max_value=1.5,
            )


# ---------------------------------------------------------------------------
# CandidateUnit — end-to-end with a worker activation tuple (forward + pickle)
# ---------------------------------------------------------------------------
class TestCandidateUnitWorkerActivation:
    @pytest.mark.unit
    def test_accepts_worker_activation_tuple_and_stays_picklable(self):
        candidate = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__output_size=1,
            CandidateUnit__activation_function=(torch.tanh, lambda x: 1.0 - torch.tanh(x) ** 2),
        )
        output = candidate.forward(torch.ones(2))
        pickle.dumps(candidate)  # local pool ships candidates by pickle — must not carry a lambda

        assert output.shape == (1,)
        assert torch.isfinite(output).all()
