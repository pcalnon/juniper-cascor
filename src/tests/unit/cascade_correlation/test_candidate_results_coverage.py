#!/usr/bin/env python
"""Coverage for candidate result collection, validation, and task generation
(per-file coverage lift 5, C-5).

Drives the previously-uncovered arms of the candidate-training result pipeline
on ``CascadeCorrelationNetwork``:

* ``_validate_training_result`` — the security / bounds rejections
  (non-numeric correlation, NaN candidate weights, elevated + over-limit
  weight magnitudes).
* ``_collect_training_results`` — the invalid-result discard, the stale-round
  discard, and the stale-summary log.
* ``_execute_sequential_training`` — the per-task exception fallback.
* ``_process_training_results`` — the success-vs-threshold mismatch log.
* ``train_candidates`` — the ``CandidateTrainingError`` re-raise and the
  generic-error ``TrainingError`` wrap.
* ``_generate_candidate_tasks`` — the SharedMemory-creation-failed fallback
  to full (non-shared) task tuples.

All fast unit tests with fakes / ``unittest.mock`` seams — no worker processes.
"""

import datetime
from queue import Empty
from unittest.mock import MagicMock, patch

import pytest
import torch

import cascade_correlation.cascade_correlation as cc_mod
from candidate_unit.candidate_unit import CandidateTrainingResult, CandidateUnit
from cascade_correlation.cascade_correlation import CandidateTrainingError, TrainingError

pytestmark = pytest.mark.unit


def _candidate_with_weights(weights, bias=0.0):
    cu = CandidateUnit(CandidateUnit__input_size=len(weights))
    cu.weights = torch.tensor(weights, dtype=torch.float32)
    cu.bias = torch.tensor([bias], dtype=torch.float32)
    return cu


class _ScriptedQueue:
    """Result-queue stand-in yielding a scripted sequence, then ``Empty``."""

    def __init__(self, items):
        self._items = list(items)

    def get(self, timeout=None):
        if self._items:
            return self._items.pop(0)
        raise Empty()

    def qsize(self):
        return len(self._items)

    def empty(self):
        return not self._items


class _DeadWorker:
    def is_alive(self):
        return False


class TestValidateTrainingResult:
    """``_validate_training_result`` security / bounds rejections."""

    def test_non_numeric_correlation_rejected(self, simple_network):
        result = CandidateTrainingResult()
        result.correlation = "not-a-number"
        assert simple_network._validate_training_result(result) is False

    def test_candidate_with_nan_weights_rejected(self, simple_network):
        result = CandidateTrainingResult(correlation=0.5)
        result.candidate = _candidate_with_weights([float("nan")])
        assert simple_network._validate_training_result(result) is False

    def test_elevated_weight_magnitude_warns_but_passes(self, simple_network):
        # 100 < |w| <= 1000 -> elevated warning, still valid.
        result = CandidateTrainingResult(correlation=0.5)
        result.candidate = _candidate_with_weights([200.0])
        assert simple_network._validate_training_result(result) is True

    def test_over_limit_weight_magnitude_rejected(self, simple_network):
        # |w| > _RESULT_MAX_WEIGHT_MAGNITUDE (1000) -> rejected.
        result = CandidateTrainingResult(correlation=0.5)
        result.candidate = _candidate_with_weights([2000.0])
        assert simple_network._validate_training_result(result) is False


class TestCollectTrainingResults:
    """``_collect_training_results`` invalid / stale discard paths."""

    def test_invalid_and_stale_results_discarded(self, simple_network):
        valid = CandidateTrainingResult(candidate_id=7, correlation=0.5, round_id="R1")
        stale = CandidateTrainingResult(candidate_id=8, correlation=0.5, round_id="OTHER")
        queue = _ScriptedQueue([object(), stale, valid])  # invalid, stale, then good
        results = simple_network._collect_training_results(
            queue,
            num_tasks=1,
            round_id="R1",
            workers=[_DeadWorker()],
        )
        assert len(results) == 1
        assert results[0].candidate_id == 7


class TestExecuteSequentialTraining:
    """``_execute_sequential_training`` per-task exception fallback."""

    def test_task_exception_appends_placeholder(self, simple_network):
        task = (0, (0, 1, 2, 3, "cand-uuid", 5), None)
        with patch.object(simple_network, "train_candidate_worker", side_effect=RuntimeError("task boom")):
            results = simple_network._execute_sequential_training([task])
        assert len(results) == 1
        # Placeholder tuple: (candidate_index, candidate_uuid, 0.0, None)
        assert results[0] == (0, "cand-uuid", 0.0, None)


class TestProcessTrainingResults:
    """``_process_training_results`` success-vs-threshold mismatch log."""

    def test_success_count_differs_from_threshold_count(self, simple_network):
        net = simple_network
        net.correlation_threshold = 0.5
        above = CandidateTrainingResult(candidate_id=0, candidate_uuid="u0", correlation=0.9, success=True)
        above.candidate = _candidate_with_weights([0.1, 0.2])
        below = CandidateTrainingResult(candidate_id=1, candidate_uuid="u1", correlation=0.1, success=True)
        below.candidate = _candidate_with_weights([0.1, 0.2])
        tasks = [(0, None, None), (1, None, None)]
        stats = net._process_training_results([above, below], tasks, datetime.datetime.now())
        # Both succeeded, only one cleared the 0.5 threshold.
        assert stats.success_count == 2
        assert stats.successful_candidates == 1


class TestTrainCandidatesErrorArms:
    """``train_candidates`` error propagation arms."""

    def test_candidate_training_error_reraised(self, simple_network):
        net = simple_network
        x = torch.zeros(4, 2)
        y = torch.zeros(4, 2)
        with patch.object(net, "_generate_candidate_tasks", return_value=[]), patch.object(net, "_execute_candidate_training", side_effect=CandidateTrainingError("irrecoverable")):
            with pytest.raises(CandidateTrainingError):
                net.train_candidates(x, y, torch.zeros(4, 2))

    def test_generic_error_wrapped_as_training_error(self, simple_network):
        net = simple_network
        x = torch.zeros(4, 2)
        y = torch.zeros(4, 2)
        with patch.object(net, "_generate_candidate_tasks", return_value=[]), patch.object(net, "_execute_candidate_training", side_effect=ValueError("boom")):
            with pytest.raises(TrainingError):
                net.train_candidates(x, y, torch.zeros(4, 2))


class TestGenerateCandidateTasksFallback:
    """``_generate_candidate_tasks`` SharedMemory-failure fallback."""

    def test_shared_memory_failure_falls_back_to_full_tasks(self, simple_network):
        net = simple_network
        candidate_input = torch.zeros(4, 3)
        y = torch.zeros(4, 2)
        residual = torch.zeros(4, 2)

        failing_block = MagicMock()
        failing_block.get_metadata.side_effect = RuntimeError("shm metadata boom")
        failing_block.close_and_unlink.side_effect = RuntimeError("cleanup boom")

        with patch.object(cc_mod, "SharedTrainingMemory", return_value=failing_block):
            tasks = net._generate_candidate_tasks(candidate_input, y, residual)

        assert len(tasks) == net.candidate_pool_size
        # Fallback path: training_inputs is the full tuple, not a shm-metadata dict.
        training_inputs = tasks[0][2]
        assert isinstance(training_inputs, tuple)
        # The failed block was removed from the active set during cleanup.
        assert failing_block not in net._active_shm_blocks
