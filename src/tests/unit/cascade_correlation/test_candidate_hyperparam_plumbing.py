#!/usr/bin/env python
"""Candidate early-stopping hyperparameter plumbing (CASCOR-505).

``candidate_patience`` and ``candidate_convergence_threshold`` are accepted at
the API boundary, whitelisted in ``_apply_params_unlocked``, and set as network
attributes — but the candidate pool constructs its own ``CandidateUnit`` in a
worker process that cannot see ``self``. Before this fix the two values were
never placed in the task payload, so every candidate silently ran the
``CandidateUnit`` module defaults while ``GET /v1/training/params`` echoed the
configured value back.

These tests pin the whole chain, hermetically and without worker processes:

    network attrs
      -> _generate_candidate_tasks   (shm-metadata dict AND legacy tuple)
      -> _build_candidate_inputs     (both payload shapes, plus back-compat)
      -> CandidateUnit(...)          (the constructed unit's own attributes)

The last hop is the one that matters most: this is the CASCOR-P0-005
key-name-mismatch class, where a payload key exists but the constructor reads a
different name and silently receives ``None``/the default. Asserting on the
*constructed unit* rather than on the payload dict is what makes that
undetectable-by-inspection failure visible.

Back-compat is pinned deliberately: a 6-element ``training_inputs`` tuple (the
shape every pre-fix caller and several existing fixtures build) must still
unpack and must still yield the module defaults.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

import cascade_correlation.cascade_correlation as cc_mod
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from cascor_constants.constants import _CANDIDATE_UNIT_CONVERGENCE_THRESHOLD, _CANDIDATE_UNIT_PATIENCE

pytestmark = pytest.mark.unit


# Deliberately unlike any default in the tree, so a passing assertion cannot be
# a default coincidentally matching the configured value.
_PATIENCE = 7
_CONVERGENCE = 0.25


def _candidate_data(input_size=2):
    """The candidate-metadata tuple shape unpacked by ``_build_candidate_inputs``."""
    return (
        0,  # candidate_index (element [0], skipped by the [1:] unpack)
        input_size,
        "Tanh",
        0.1,  # random_value_scale
        "test-uuid",
        42,  # candidate_seed
        1.0,  # random_max_value
        100,  # sequence_max_value
    )


def _tensors(rows=10, cols=2):
    return torch.randn(rows, cols), torch.randn(rows, cols), torch.randn(rows, cols)


def _legacy_tuple(x, y, residual, *, with_controls):
    """Build a training_inputs tuple in either the 8- or the pre-fix 6-element shape."""
    base = (x, 3, y, residual, 0.01, 10)
    return base + (_PATIENCE, _CONVERGENCE) if with_controls else base


class TestGenerateCandidateTasksCarriesControls:
    """``_generate_candidate_tasks`` must put both controls in the task payload."""

    def _network(self):
        config = CascadeCorrelationConfig.create_simple_config(
            input_size=3,
            output_size=2,
            max_hidden_units=3,
            candidate_patience=_PATIENCE,
            candidate_convergence_threshold=_CONVERGENCE,
        )
        return CascadeCorrelationNetwork(config=config)

    def test_shm_metadata_payload_carries_both_controls(self):
        """OPT-5 SharedMemory path: both controls ride in the metadata dict."""
        net = self._network()
        block = MagicMock()
        block.get_metadata.return_value = {}
        block.name = "fake-shm"

        with patch.object(cc_mod, "SharedTrainingMemory", return_value=block):
            tasks = net._generate_candidate_tasks(torch.zeros(4, 3), torch.zeros(4, 2), torch.zeros(4, 2))

        training_inputs = tasks[0][2]
        assert isinstance(training_inputs, dict)
        assert training_inputs["candidate_patience"] == _PATIENCE
        assert training_inputs["candidate_convergence_threshold"] == _CONVERGENCE

    def test_legacy_tuple_fallback_carries_both_controls(self):
        """SharedMemory-creation failure falls back to a tuple that still carries them."""
        net = self._network()
        failing_block = MagicMock()
        failing_block.get_metadata.side_effect = RuntimeError("shm metadata boom")

        with patch.object(cc_mod, "SharedTrainingMemory", return_value=failing_block):
            tasks = net._generate_candidate_tasks(torch.zeros(4, 3), torch.zeros(4, 2), torch.zeros(4, 2))

        training_inputs = tasks[0][2]
        assert isinstance(training_inputs, tuple)
        # Appended, not inserted — the first six positions keep their meaning.
        assert len(training_inputs) == 8
        assert training_inputs[1] == net.candidate_epochs
        assert training_inputs[4] == net.candidate_learning_rate
        assert training_inputs[6] == _PATIENCE
        assert training_inputs[7] == _CONVERGENCE


class TestBuildCandidateInputsSurfacesControls:
    """``_build_candidate_inputs`` must surface both controls from either shape."""

    def test_legacy_eight_tuple_surfaces_controls(self):
        x, y, residual = _tensors()
        task = (0, _candidate_data(), _legacy_tuple(x, y, residual, with_controls=True))

        result = CascadeCorrelationNetwork._build_candidate_inputs(task_data_input=task, worker_uuid="w", worker_id=1)

        assert result["candidate_patience"] == _PATIENCE
        assert result["candidate_convergence_threshold"] == _CONVERGENCE
        # The pre-existing positional fields are untouched by the extension.
        assert result["candidate_epochs"] == 3
        assert result["candidate_learning_rate"] == 0.01
        assert result["candidate_display_frequency"] == 10
        assert torch.equal(result["candidate_input"], x)

    def test_legacy_six_tuple_falls_back_to_module_defaults(self):
        """Back-compat: a pre-fix 6-tuple still unpacks, with the defaults it always had."""
        x, y, residual = _tensors()
        task = (0, _candidate_data(), _legacy_tuple(x, y, residual, with_controls=False))

        result = CascadeCorrelationNetwork._build_candidate_inputs(task_data_input=task, worker_uuid="w", worker_id=1)

        assert result["candidate_patience"] == _CANDIDATE_UNIT_PATIENCE
        assert result["candidate_convergence_threshold"] == _CANDIDATE_UNIT_CONVERGENCE_THRESHOLD

    def _dict_payload_result(self, payload):
        x, y, residual = _tensors()
        task = (0, _candidate_data(), payload)
        with patch.object(cc_mod.SharedTrainingMemory, "reconstruct_tensors", return_value=([x, y, residual], None)):
            return CascadeCorrelationNetwork._build_candidate_inputs(task_data_input=task, worker_uuid="w", worker_id=1)

    def test_shm_dict_payload_surfaces_controls(self):
        result = self._dict_payload_result(
            {
                "shm_name": "fake",
                "candidate_epochs": 3,
                "candidate_learning_rate": 0.01,
                "candidate_display_frequency": 10,
                "candidate_patience": _PATIENCE,
                "candidate_convergence_threshold": _CONVERGENCE,
            }
        )
        assert result["candidate_patience"] == _PATIENCE
        assert result["candidate_convergence_threshold"] == _CONVERGENCE

    def test_shm_dict_payload_without_controls_falls_back(self):
        """Back-compat: a metadata dict built before this fix keeps the module defaults."""
        result = self._dict_payload_result(
            {
                "shm_name": "fake",
                "candidate_epochs": 3,
                "candidate_learning_rate": 0.01,
                "candidate_display_frequency": 10,
            }
        )
        assert result["candidate_patience"] == _CANDIDATE_UNIT_PATIENCE
        assert result["candidate_convergence_threshold"] == _CANDIDATE_UNIT_CONVERGENCE_THRESHOLD


class TestWorkerConstructsUnitWithControls:
    """The CASCOR-P0-005 hop: assert on the *constructed* unit, not the payload."""

    def test_worker_builds_candidate_with_configured_controls(self):
        x, y, residual = _tensors()
        task = (0, _candidate_data(), _legacy_tuple(x, y, residual, with_controls=True))

        result = CascadeCorrelationNetwork.train_candidate_worker(task_data_input=task, parallel=False)

        assert result.candidate is not None, "worker returned no candidate; cannot assert construction"
        assert result.candidate.patience == _PATIENCE
        assert result.candidate.convergence_threshold == _CONVERGENCE

    def test_worker_falls_back_to_defaults_for_legacy_payload(self):
        """Negative control — the pre-fix payload must still yield the old behaviour."""
        x, y, residual = _tensors()
        task = (0, _candidate_data(), _legacy_tuple(x, y, residual, with_controls=False))

        result = CascadeCorrelationNetwork.train_candidate_worker(task_data_input=task, parallel=False)

        assert result.candidate is not None
        assert result.candidate.patience == _CANDIDATE_UNIT_PATIENCE
        assert result.candidate.convergence_threshold == _CANDIDATE_UNIT_CONVERGENCE_THRESHOLD


class TestNetworkToConstructedUnitRoundTrip:
    """End-to-end: a network attribute reaches the constructed candidate unit.

    This is the assertion the defect would have failed. It deliberately goes
    through the real ``_generate_candidate_tasks`` -> ``_build_candidate_inputs``
    -> ``CandidateUnit`` chain rather than hand-building a payload, so a future
    change that drops a key anywhere along it fails here.
    """

    @pytest.mark.parametrize("shm_available", [True, False], ids=["shm-dict", "legacy-tuple"])
    def test_configured_controls_reach_the_candidate_unit(self, shm_available):
        config = CascadeCorrelationConfig.create_simple_config(
            input_size=2,
            output_size=2,
            max_hidden_units=3,
            candidate_patience=_PATIENCE,
            candidate_convergence_threshold=_CONVERGENCE,
        )
        net = CascadeCorrelationNetwork(config=config)
        assert net.candidate_patience == _PATIENCE, "fixture precondition: config value reached the network"

        x, y, residual = _tensors()

        if shm_available:
            block = MagicMock()
            block.get_metadata.return_value = {"shm_name": "fake"}
            with patch.object(cc_mod, "SharedTrainingMemory", return_value=block):
                tasks = net._generate_candidate_tasks(x, y, residual)
            with patch.object(cc_mod.SharedTrainingMemory, "reconstruct_tensors", return_value=([x, y, residual], None)):
                result = CascadeCorrelationNetwork.train_candidate_worker(task_data_input=tasks[0], parallel=False)
        else:
            failing_block = MagicMock()
            failing_block.get_metadata.side_effect = RuntimeError("shm metadata boom")
            with patch.object(cc_mod, "SharedTrainingMemory", return_value=failing_block):
                tasks = net._generate_candidate_tasks(x, y, residual)
            result = CascadeCorrelationNetwork.train_candidate_worker(task_data_input=tasks[0], parallel=False)

        assert result.candidate is not None
        assert result.candidate.patience == _PATIENCE
        assert result.candidate.convergence_threshold == _CONVERGENCE
