"""Smoke tests for juniper-cascor-model.

These exercise the exact surface the distributed worker depends on: importing
``CandidateUnit`` the way ``task_executor`` does, the activation registry that fixes the
worker's ``Unknown activation 'Tanh'`` fallback (CW-05 gap #4), and the resilient logging
that fixes the ``/logs`` ENOENT training crash (CW-05 gap #3).
"""

import pytest
import pickle

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch required for candidate-core")


def test_version_is_importable_without_torch():
    # `import juniper_cascor_model` must stay lightweight (no torch import) for version checks.
    import juniper_cascor_model

    assert juniper_cascor_model.__version__
    assert isinstance(juniper_cascor_model.__version__, str)


@requires_torch
def test_candidate_unit_imports_self_contained():
    # The exact import juniper-cascor-worker's task_executor performs.
    from candidate_unit.candidate_unit import CandidateUnit  # noqa: F401


@requires_torch
def test_activation_registry_includes_titlecase_names():
    # CW-05 gap #4: the worker logged `Unknown activation 'Tanh'`; the core map has it
    # (both TitleCase nn-module and lowercase functional variants).
    from utils.activation import ActivationWithDerivative

    amap = ActivationWithDerivative.ACTIVATION_MAP
    for name in ("Tanh", "Sigmoid", "ReLU", "tanh", "sigmoid", "relu"):
        assert name in amap, f"activation map missing {name!r}"


@requires_torch
def test_candidate_unit_accepts_worker_activation_tuple():
    # The current juniper-cascor-worker resolver returns (activation, derivative) tuples.
    # The core wrapper must normalize that legacy shape before CandidateUnit.forward().
    import torch

    from candidate_unit.candidate_unit import CandidateUnit

    candidate = CandidateUnit(
        CandidateUnit__input_size=2,
        CandidateUnit__output_size=1,
        CandidateUnit__activation_function=(torch.tanh, lambda x: 1.0 - torch.tanh(x) ** 2),
    )

    output = candidate.forward(torch.ones(2))
    pickle.dumps(candidate)

    assert output.shape == (1,)
    assert torch.isfinite(output).all()


@requires_torch
def test_candidate_unit_accepts_remote_integer_bounds_as_floats():
    # Current remote task payloads JSON-decode these integer-valued bounds as floats.
    # CandidateUnit must normalize them before passing values to random/range APIs.
    from candidate_unit.candidate_unit import CandidateUnit

    candidate = CandidateUnit(
        CandidateUnit__input_size=2,
        CandidateUnit__output_size=1,
        CandidateUnit__candidate_index=2.0,
        CandidateUnit__random_seed=1.0,
        CandidateUnit__random_max_value=1.0,
        CandidateUnit__sequence_max_value=100.0,
    )

    assert candidate.candidate_index == 2
    assert candidate.random_seed == 1
    assert candidate.random_max_value == 1
    assert candidate.sequence_max_value == 100


@requires_torch
def test_remote_collection_timeout_constants_are_exported():
    # Current juniper-cascor imports these from cascor_constants.constants for dual-path
    # remote result collection. The extracted package must not drop them during adoption.
    from cascor_constants.constants import (
        _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MAX_TIMEOUT,
        _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MIN_TIMEOUT,
        _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_SECONDS_PER_EPOCH,
    )

    assert _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_SECONDS_PER_EPOCH == 1.0
    assert _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MIN_TIMEOUT == 120.0
    assert _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MAX_TIMEOUT == 900.0


@requires_torch
def test_logging_is_best_effort_when_log_path_unwritable(monkeypatch):
    # CW-05 gap #3: a missing/unwritable log directory must degrade to console-only,
    # never raise — otherwise it fails the candidate-training task (the original symptom).
    from log_config.logger.logger import Logger

    # Force the class-level log file onto an uncreatable path (/proc is read-only).
    monkeypatch.setattr(Logger, "_logging_file", "/proc/juniper-cascor-model-nonexistent/x.log", raising=False)
    # critical() is always above any level threshold, so it reaches the file-write path.
    Logger.critical("juniper-cascor-model smoke: resilient logging must not raise")
