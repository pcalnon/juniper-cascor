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
import uuid
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


# ---------------------------------------------------------------------------
# ISSUE-319 — remote-dispatch result conversion maps by GLOBAL candidate_id
# ---------------------------------------------------------------------------
class TestRemoteDispatchCandidateIdMapping:
    """``_dispatch_to_remote_workers`` must bind each remote result to its
    originating task by the *global* ``candidate_id`` (the pool index), NOT by
    list position into the remote subset.

    Regression (cascor#319): the old ``task_specs[tr.candidate_id]`` indexed the
    local 2-task subset by a global id (e.g. 15) → IndexError on every dual-path
    round. That exception was swallowed by ``TaskDistributor`` into an infinite
    local-retry fallback that starved, pinning the network at one hidden unit.
    These tests drive the REAL method with a mocked coordinator (no workers).
    """

    @staticmethod
    def _make_tasks(global_ids, input_size=2):
        """Build internal task tuples: (task_idx, candidate_data_tuple, training_inputs)."""
        tasks = []
        for gid in global_ids:
            candidate_data_tuple = (
                gid,  # candidate_index — the GLOBAL pool index
                input_size,
                "Tanh",  # activation_name
                1.0,  # random_value_scale
                f"uuid-{gid}",  # candidate_uuid
                gid,  # candidate_seed
                1.0,  # random_max_value
                1.0,  # sequence_max_value
            )
            # ISSUE-319: production passes the OPT-5 SharedMemory *dict* here (string
            # keys), NOT a positional tuple. The pre-fix ``training_inputs[1]`` indexed
            # this dict and raised ``KeyError(1)`` (masked as "(1)") on every round. The
            # fix sources params from ``self.candidate_*`` so the dict is never indexed;
            # using the real (dict) shape is what makes this a regression guard.
            training_inputs = {
                "candidate_epochs": 3,
                "candidate_learning_rate": 0.01,
                "candidate_display_frequency": 10,
                "shm_name": f"shm-{gid}",
            }
            tasks.append((gid, candidate_data_tuple, training_inputs))
        return tasks

    @staticmethod
    def _make_result(candidate_id, round_id=None):
        from api.workers.coordinator import TaskResult

        return TaskResult(
            task_id=f"task-{candidate_id}",
            candidate_id=candidate_id,  # worker echoes the GLOBAL index
            candidate_uuid=f"uuid-{candidate_id}",
            correlation=0.5,
            success=True,
            epochs_completed=3,
            activation_name="Tanh",
            all_correlations=[0.5],
            numerator=1.0,
            denominator=2.0,
            best_corr_idx=0,
            tensors={},  # no weights/bias/norm_* → conversion uses None for those
            error_message=None,
            round_id=round_id,
        )

    @pytest.mark.unit
    def test_global_candidate_ids_do_not_crash_conversion(self):
        """Remote results whose global id >= len(task_specs) must convert cleanly.

        This is the core #319 regression: ids 15 & 16 are the 2-remote overflow
        slice of a 40-candidate pool (38 local / 2 remote). The pre-fix code did
        ``task_specs[15]`` on a 2-element list → IndexError."""
        import torch

        net = _make_network()
        tasks = self._make_tasks([15, 16])
        coord = mock.MagicMock()
        coord.collect_results.return_value = [self._make_result(15), self._make_result(16)]
        net._worker_coordinator = coord

        ci, y, err = torch.zeros((4, 2)), torch.zeros((4, 2)), torch.zeros((4, 2))
        # Pre-fix this raised KeyError(1) when building task_specs (dict-indexed
        # training_inputs) AND would IndexError in the conversion (global candidate_id).
        results = net._dispatch_to_remote_workers(tasks, ci, y, err)

        assert len(results) == 2, "both remote results must convert (was IndexError pre-#319)"
        assert {r.candidate_id for r in results} == {15, 16}
        # The dispatched specs must carry the network's candidate config, sourced from
        # self.candidate_* — NOT positional-indexed from the OPT-5 dict (the KeyError(1)).
        _round_id, sent_specs, _tensors = coord.submit_tasks.call_args.args
        assert sent_specs, "tasks must have been submitted to the coordinator"
        assert all(s["training_params"]["epochs"] == net.candidate_epochs for s in sent_specs), "training params must come from self.candidate_* config, not the training_inputs dict"
        # ISSUE-319 dispatch coercion: int-valued candidate bounds must stay int on
        # the JSON wire. _make_tasks feeds them as float (1.0); a prior float() here
        # made the remote worker raise "'float' object cannot be interpreted as an
        # integer" inside random.randint()/range(). Guard the int() fix.
        for s in sent_specs:
            cd = s["candidate_data"]
            assert type(cd["random_max_value"]) is int, f"random_max_value must be int on the wire, got {type(cd['random_max_value']).__name__}"
            assert type(cd["sequence_max_value"]) is int, f"sequence_max_value must be int on the wire, got {type(cd['sequence_max_value']).__name__}"

    @pytest.mark.unit
    def test_unknown_candidate_id_is_skipped_not_crash(self):
        """A leaked/stale result whose candidate_id isn't in this dispatch is
        dropped — remote ``TaskResult`` has no round_id, so this guards the
        cross-round contamination gap noted in #319."""
        import torch

        net = _make_network()
        tasks = self._make_tasks([15, 16])
        coord = mock.MagicMock()
        coord.collect_results.return_value = [self._make_result(16), self._make_result(99)]
        net._worker_coordinator = coord

        ci, y, err = torch.zeros((4, 2)), torch.zeros((4, 2)), torch.zeros((4, 2))
        results = net._dispatch_to_remote_workers(tasks, ci, y, err)

        assert [r.candidate_id for r in results] == [16], "unknown candidate_id must be skipped, not crash or mis-bind"

    @pytest.mark.unit
    def test_remote_collect_timeout_scales_to_training_not_shutdown(self):
        """ISSUE-319 (defect #3): the remote-collection budget must track candidate
        training, not the ~10s candidate_training_shutdown_timeout that always expired
        before a remote round could finish — the original stall trigger."""
        from cascor_constants.constants import (
            _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MAX_TIMEOUT,
            _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MIN_TIMEOUT,
        )

        # Small epoch counts floor at MIN — far above the ~10s shutdown budget.
        net = _make_network(candidate_epochs=3)
        small = net._remote_result_collection_timeout()
        assert small == _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MIN_TIMEOUT
        assert small > net.candidate_training_shutdown_timeout, "budget must exceed the shutdown timeout that caused the stall"

        # Large epoch counts scale up but never exceed the hard ceiling (the hang bound).
        big = _make_network(candidate_epochs=100_000)._remote_result_collection_timeout()
        assert big == _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MAX_TIMEOUT

        # A mid-range count lands strictly between the bounds.
        mid_epochs = int((_CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MIN_TIMEOUT + _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MAX_TIMEOUT) / 2)
        mid = _make_network(candidate_epochs=mid_epochs)._remote_result_collection_timeout()
        assert _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MIN_TIMEOUT < mid < _CASCADE_CORRELATION_NETWORK_REMOTE_COLLECT_MAX_TIMEOUT

    @pytest.mark.unit
    def test_stale_round_remote_result_is_discarded(self):
        """ISSUE-319 (defect #4): a result tagged with a different round_id is dropped
        even when its candidate_id matches a dispatched task — round isolation for the
        remote leg, mirroring RC-5 on the local path."""
        import torch

        net = _make_network()
        tasks = self._make_tasks([15, 16])
        coord = mock.MagicMock()
        fixed = uuid.UUID(int=0x319)
        current_round = str(fixed)
        coord.collect_results.return_value = [
            self._make_result(15, round_id=current_round),  # this round → kept
            self._make_result(16, round_id="stale-round-deadbeef"),  # other round → dropped
        ]
        net._worker_coordinator = coord

        ci, y, err = torch.zeros((4, 2)), torch.zeros((4, 2)), torch.zeros((4, 2))
        with mock.patch("cascade_correlation.cascade_correlation.uuid.uuid4", return_value=fixed):
            results = net._dispatch_to_remote_workers(tasks, ci, y, err)

        assert [r.candidate_id for r in results] == [15], "stale-round result (16) must be discarded despite a valid candidate_id"
