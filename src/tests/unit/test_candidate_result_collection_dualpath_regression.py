#!/usr/bin/env python
"""Regression tests for the dual-path candidate result-collection bug (2026-05-30).

Symptom (observed on the deployed stack): a training run trains all candidate
units but grows ZERO hidden units and prematurely reports "Completed", with the
cascor log showing ``_process_training_results: Mismatch in results count:
expected 40, got 2``.

Two compounding root causes were confirmed and fixed:

* **Defect 1 — collection timeout shorter than candidate training time.**
  ``_collect_training_results`` used a fixed 60s *total* deadline. A 40-candidate
  pool can take >2 min, so the collector abandoned the round with ~0 results
  while the persistent workers were still training. The deadline is now an
  *inactivity* deadline (reset on each result) plus a worker-liveness early-exit.

* **Defect 2 — dual-path remote-fallback recreated the local pool mid-round.**
  ``_ensure_worker_pool`` required an exact size match, so a small remote-fallback
  retry batch (e.g. 2 tasks) tore down the live 15-worker pool — SIGKILLing
  in-flight workers and orphaning their result queue. A healthy pool with at
  least as many live workers as requested is now reused.

These tests exercise the real methods (no real worker processes spawned) so a
regression in either fix is caught.
"""

import queue
import threading
import time
from unittest import mock

import pytest

from candidate_unit.candidate_unit import CandidateTrainingResult
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig


def _make_network(**overrides):
    defaults = {
        "input_size": 2,
        "output_size": 2,
        "random_seed": 42,
        "candidate_pool_size": 2,
        "candidate_epochs": 3,
        "output_epochs": 3,
        "max_hidden_units": 2,
        "patience": 1,
    }
    defaults.update(overrides)
    return CascadeCorrelationNetwork(config=CascadeCorrelationConfig(**defaults))


class _FakeWorker:
    """Stand-in for a multiprocessing worker process (no real process spawned)."""

    def __init__(self, alive=True, pid=4242, name="fake-worker"):
        self._alive = alive
        self.pid = pid
        self.name = name

    def is_alive(self):
        return self._alive

    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def terminate(self):
        self._alive = False


# ---------------------------------------------------------------------------
# Defect 1 — _collect_training_results inactivity deadline + liveness early-exit
# ---------------------------------------------------------------------------
class TestCollectionInactivityDeadline:
    @pytest.mark.unit
    def test_inactivity_deadline_resets_on_each_result(self):
        """A round whose TOTAL duration exceeds queue_timeout must still collect
        every result, as long as no single GAP between results exceeds it.

        Regression guard: the old fixed-total-deadline behaviour would abandon
        this round partway through (returning < num_tasks)."""
        net = _make_network()
        q = queue.Queue()
        num_tasks = 5
        gap = 0.15
        inactivity = 0.6  # >> gap, << num_tasks * gap (=0.75) total

        def _producer():
            for cid in range(num_tasks):
                time.sleep(gap)
                q.put(CandidateTrainingResult(candidate_id=cid, correlation=0.1 * cid))

        t = threading.Thread(target=_producer, daemon=True)
        t.start()
        results = net._collect_training_results(q, num_tasks=num_tasks, queue_timeout=inactivity, request_timeout=0.05)
        t.join(timeout=2.0)

        assert len(results) == num_tasks, "inactivity deadline must not abandon a slow-but-progressing round"

    @pytest.mark.unit
    def test_liveness_early_exit_when_all_workers_dead(self):
        """If every worker has exited and the queue is drained, stop immediately
        instead of waiting out the (now large) inactivity timeout."""
        net = _make_network()
        q = queue.Queue()  # empty
        dead = _FakeWorker(alive=False)

        start = time.monotonic()
        results = net._collect_training_results(q, num_tasks=3, queue_timeout=10.0, request_timeout=0.1, workers=[dead])
        elapsed = time.monotonic() - start

        assert results == []
        assert elapsed < 2.0, "dead-worker + empty queue must early-exit, not wait the full inactivity timeout"

    @pytest.mark.unit
    def test_alive_worker_does_not_early_exit(self):
        """A live worker (which could still deliver) must NOT trigger the
        dead-worker early-exit; collection waits out the inactivity window."""
        net = _make_network()
        q = queue.Queue()  # empty
        alive = _FakeWorker(alive=True)

        start = time.monotonic()
        results = net._collect_training_results(q, num_tasks=2, queue_timeout=0.4, request_timeout=0.1, workers=[alive])
        elapsed = time.monotonic() - start

        assert results == []
        assert elapsed >= 0.3, "a live worker must keep the collector waiting until the inactivity timeout"


# ---------------------------------------------------------------------------
# Defect 2 — _ensure_worker_pool reentrancy (reuse a larger live pool)
# ---------------------------------------------------------------------------
class TestEnsureWorkerPoolReentrancy:
    @pytest.mark.unit
    def test_smaller_request_reuses_pool_without_teardown(self):
        """A request for FEWER workers than the live pool must reuse it — never
        tear it down (that orphaned in-flight results: the root cause)."""
        net = _make_network()
        tq, rq = object(), object()
        net._persistent_workers = [_FakeWorker(), _FakeWorker(), _FakeWorker()]
        net._persistent_pool_size = 3
        net._persistent_task_queue = tq
        net._persistent_result_queue = rq

        with mock.patch.object(net, "_shutdown_worker_pool") as shut:
            out_tq, out_rq = net._ensure_worker_pool(1)

        assert out_tq is tq and out_rq is rq, "must return the SAME queues (pool reused)"
        shut.assert_not_called()
        assert net._persistent_pool_size == 3

    @pytest.mark.unit
    def test_equal_request_reuses_pool(self):
        net = _make_network()
        tq, rq = object(), object()
        net._persistent_workers = [_FakeWorker(), _FakeWorker()]
        net._persistent_pool_size = 2
        net._persistent_task_queue = tq
        net._persistent_result_queue = rq

        with mock.patch.object(net, "_shutdown_worker_pool") as shut:
            out_tq, out_rq = net._ensure_worker_pool(2)

        assert out_tq is tq and out_rq is rq
        shut.assert_not_called()

    @pytest.mark.unit
    def test_larger_request_recreates_pool(self):
        """A request for MORE workers than the live pool legitimately recreates."""
        net = _make_network()
        net._persistent_workers = [_FakeWorker(), _FakeWorker()]
        net._persistent_pool_size = 2
        net._persistent_task_queue = object()
        net._persistent_result_queue = object()

        with mock.patch.object(net, "_shutdown_worker_pool") as shut, mock.patch.object(net._mp_ctx, "Queue", return_value=mock.MagicMock()), mock.patch.object(net._mp_ctx, "Process", return_value=_FakeWorker()):
            net._ensure_worker_pool(5)

        shut.assert_called_once()

    @pytest.mark.unit
    def test_degraded_pool_recreates(self):
        """If some workers died (alive < pool size), the pool is not healthy and
        must be recreated even for the same requested size."""
        net = _make_network()
        net._persistent_workers = [_FakeWorker(alive=True), _FakeWorker(alive=False), _FakeWorker(alive=True)]
        net._persistent_pool_size = 3
        net._persistent_task_queue = object()
        net._persistent_result_queue = object()

        with mock.patch.object(net, "_shutdown_worker_pool") as shut, mock.patch.object(net._mp_ctx, "Queue", return_value=mock.MagicMock()), mock.patch.object(net._mp_ctx, "Process", return_value=_FakeWorker()):
            net._ensure_worker_pool(3)

        shut.assert_called_once()
