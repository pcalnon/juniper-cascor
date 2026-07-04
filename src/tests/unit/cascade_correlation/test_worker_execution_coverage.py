#!/usr/bin/env python
"""Coverage for the multiprocessing worker execution path
(per-file coverage lift 5, C-5).

Drives the previously-uncovered arms of the static worker helpers on
``CascadeCorrelationNetwork`` without spawning real processes:

* ``_worker_loop`` — the RC-4 instrumentation-queue branches (worker.started /
  task_received / task_done emits) and the shared-training-inputs log, driven
  by a scripted in-process task queue that feeds one task then a sentinel.
* ``_process_worker_task`` — the RC-3 lightweight-task reconstruction branch.
* ``_publish_failure_result`` — the success-put log and the generic put-error
  swallow.
* ``train_candidate_worker`` — the worker-id retrieval failure fallback.
* ``_build_candidate_inputs`` — the OPT-5 SharedMemory ``OSError`` re-raise
  (triggers the sequential fallback in the caller).
* ``_train_candidate_unit`` — the training-exception failure result.

All fast unit tests with fakes / ``unittest.mock`` seams.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

import cascade_correlation.cascade_correlation as cc_mod
from candidate_unit.candidate_unit import CandidateTrainingResult
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from log_config.logger.logger import Logger

pytestmark = pytest.mark.unit


class _SeqTaskQueue:
    """Task-queue stand-in yielding a preset sequence via ``get``."""

    def __init__(self, items):
        self._items = list(items)

    def get(self, timeout=None):
        from queue import Empty

        if self._items:
            return self._items.pop(0)
        raise Empty()


class _PutQueue:
    """Result-queue stand-in recording (or failing) ``put`` calls."""

    def __init__(self, *, error=None):
        self.puts = []
        self._error = error

    def put(self, item, timeout=None):
        if self._error is not None:
            raise self._error
        self.puts.append(item)


class TestWorkerLoop:
    """``_worker_loop`` instrumentation-queue + shared-inputs branches."""

    def test_instrumented_loop_processes_one_task_then_sentinel(self):
        task = (0, ("cdata",), "training-inputs", "round-1")
        task_queue = _SeqTaskQueue([task, None])  # one task, then shutdown sentinel
        result_queue = MagicMock()
        instrumentation_queue = MagicMock()

        with patch("parallelism.rc4_ring_buffer.set_worker_queue"), patch("parallelism.rc4_ring_buffer.emit"), patch.object(CascadeCorrelationNetwork, "_process_worker_task") as mock_process:
            CascadeCorrelationNetwork._worker_loop(
                task_queue,
                result_queue,
                parallel=True,
                task_queue_timeout=0.01,
                worker_thread_count=1,
                shared_training_inputs=("shared",),
                progress_queue=None,
                instrumentation_queue=instrumentation_queue,
            )

        mock_process.assert_called_once()


class TestProcessWorkerTask:
    """``_process_worker_task`` RC-3 lightweight-task reconstruction."""

    def test_lightweight_task_reconstructed_with_shared_inputs(self):
        result_queue = _PutQueue()
        dummy_result = MagicMock(correlation=0.5, success=True)
        with patch("parallelism.rc4_ring_buffer.emit"), patch.object(CascadeCorrelationNetwork, "train_candidate_worker", return_value=dummy_result) as mock_worker:
            CascadeCorrelationNetwork._process_worker_task(
                task=(3, ("candidate-data",)),  # 2-element lightweight task
                shared_training_inputs=("shared-training",),
                progress_queue=None,
                result_queue=result_queue,
                parallel=False,
                logger=Logger,
            )
        # The lightweight task was expanded into a full 3-tuple.
        full_task = mock_worker.call_args.kwargs["task_data_input"]
        assert full_task == (3, ("candidate-data",), ("shared-training",))
        assert result_queue.puts == [dummy_result]


class TestPublishFailureResult:
    """``_publish_failure_result`` put success + generic error swallow."""

    def test_publish_success(self):
        queue = _PutQueue()
        task = (2, (0, 1, 2, 3, "cand-uuid", 5), None)
        CascadeCorrelationNetwork._publish_failure_result(task, RuntimeError("orig"), queue, Logger)
        assert len(queue.puts) == 1
        published = queue.puts[0]
        assert published.candidate_id == 2
        assert published.candidate_uuid == "cand-uuid"

    def test_publish_generic_error_swallowed(self):
        queue = _PutQueue(error=RuntimeError("queue exploded"))
        task = (2, (0, 1, 2, 3, "cand-uuid", 5), None)
        # Must not propagate.
        CascadeCorrelationNetwork._publish_failure_result(task, RuntimeError("orig"), queue, Logger)


class TestTrainCandidateWorker:
    """``train_candidate_worker`` worker-id retrieval fallback."""

    def test_worker_id_retrieval_failure_uses_defaults(self):
        with patch.object(cc_mod.mp, "current_process", side_effect=RuntimeError("no proc")):
            # parallel=True forces the mp.current_process() path; with no task
            # data the method short-circuits after the id fallback.
            result = CascadeCorrelationNetwork.train_candidate_worker(task_data_input=None, parallel=True)
        assert result == (None, None, 0.0, None)


class TestBuildCandidateInputs:
    """``_build_candidate_inputs`` OPT-5 SharedMemory OSError re-raise."""

    def test_missing_shared_memory_reraises_oserror(self):
        candidate_data = (0, 3, "Tanh", 1.0, "cand-uuid", 42, 1_000_000, 1_000_000)
        training_inputs = {
            "shm_name": "juniper_train_does_not_exist_lift5",
            "candidate_epochs": 1,
            "candidate_learning_rate": 0.1,
            "candidate_display_frequency": 1,
        }
        task = (0, candidate_data, training_inputs)
        with pytest.raises(OSError):
            CascadeCorrelationNetwork._build_candidate_inputs(task_data_input=task, worker_id=0, worker_uuid="w")


class TestTrainCandidateUnit:
    """``_train_candidate_unit`` training-exception failure result."""

    def test_training_exception_returns_failure_result(self):
        candidate = MagicMock()
        candidate.get_uuid.return_value = "cand-uuid"
        candidate.train_detailed.side_effect = RuntimeError("train boom")
        result = CascadeCorrelationNetwork._train_candidate_unit(
            candidate=candidate,
            candidate_uuid="cand-uuid",
            candidate_index=3,
            candidate_input=torch.zeros(4, 2),
            candidate_epochs=1,
            residual_error=torch.zeros(4, 2),
            candidate_learning_rate=0.1,
            candidate_display_frequency=1,
        )
        assert isinstance(result, CandidateTrainingResult)
        assert result.success is False
        assert result.candidate_id == 3
