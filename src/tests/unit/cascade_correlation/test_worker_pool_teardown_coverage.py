#!/usr/bin/env python
"""Coverage for the persistent worker-pool teardown + SharedMemory lifecycle
(per-file coverage lift 5, C-5).

Exercises the previously-uncovered teardown / cleanup arms of
``CascadeCorrelationNetwork`` that only run during pool shutdown or process
exit — worker join / terminate / SIGKILL escalation
(``_terminate_workers``), active + pending SharedMemory block cleanup
(``_cleanup_shared_memory_blocks`` / ``_cleanup_shared_memory`` /
``_cleanup_pending_shared_memory``), shutdown-sentinel dispatch
(``_send_shutdown_sentinels``), and the top-level ``_shutdown_worker_pool``
orchestrator — plus the ``SharedTrainingMemory`` close / unlink / reconstruct
edges and ``CandidateTrainingManager.start`` method validation.

All fast unit tests driven by fakes / ``unittest.mock`` seams — no real worker
processes, no /dev/shm churn beyond a single deterministic round-trip.
"""

import signal
import struct
from multiprocessing.managers import BaseManager
from unittest.mock import MagicMock, patch

import pytest
import torch

import cascade_correlation.cascade_correlation as cc_mod
from cascade_correlation.cascade_correlation import CandidateTrainingManager, SharedTrainingMemory

pytestmark = pytest.mark.unit


class _FakeWorker:
    """Minimal ``multiprocessing.Process`` stand-in for teardown tests.

    ``is_alive`` pops from a preset sequence so each test can script the
    join / terminate / SIGKILL escalation deterministically.
    """

    def __init__(self, name, alive_sequence, pid=999_999):
        self.name = name
        self.pid = pid
        self._alive = list(alive_sequence)
        self.terminated = False
        self.join_timeouts = []

    def is_alive(self):
        return self._alive.pop(0) if self._alive else False

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)
        return None

    def terminate(self):
        self.terminated = True


class _FakeShm:
    """SharedMemory-block stand-in recording close / unlink calls."""

    def __init__(self, *, raise_on=()):
        self.closed = False
        self.unlinked = False
        self._raise_on = set(raise_on)

    def close_and_unlink(self):
        if "close_and_unlink" in self._raise_on:
            raise RuntimeError("close_and_unlink boom")
        self.closed = True
        self.unlinked = True

    def close(self):
        if "close" in self._raise_on:
            raise RuntimeError("close boom")
        self.closed = True

    def unlink(self):
        if "unlink" in self._raise_on:
            raise RuntimeError("unlink boom")
        self.unlinked = True


class _FakeQueue:
    """Task-queue stand-in: records ``put`` calls, optionally raising."""

    def __init__(self, *, raise_on_put=False):
        self.puts = []
        self._raise_on_put = raise_on_put

    def put(self, item, timeout=None):
        if self._raise_on_put:
            raise RuntimeError("queue put boom")
        self.puts.append(item)


class TestShutdownWorkerPool:
    """``_shutdown_worker_pool`` orchestration + its callees."""

    def test_full_teardown_with_escalation_and_shm(self, simple_network):
        net = simple_network
        graceful = _FakeWorker("graceful", [False])
        escalate = _FakeWorker("escalate", [True, True])  # join fails, terminate fails -> SIGKILL
        net._persistent_workers = [graceful, escalate]
        net._persistent_task_queue = _FakeQueue()
        net._persistent_pool_size = 2
        active = _FakeShm()
        pending = _FakeShm()
        net._active_shm_blocks = [active]
        net._pending_shm_unlinks = [pending]

        with patch.object(cc_mod.os, "kill") as mock_kill:
            net._shutdown_worker_pool()

        # Escalation reached SIGKILL for the non-graceful worker only.
        mock_kill.assert_called_once_with(escalate.pid, signal.SIGKILL)
        assert escalate.terminated is True
        assert graceful.terminated is False
        # Sentinels were dispatched (one per worker).
        assert net._persistent_task_queue is None  # reset after shutdown
        assert net._persistent_workers == []
        assert net._persistent_pool_size == 0
        # SharedMemory blocks cleaned.
        assert active.closed and active.unlinked
        assert pending.unlinked

    def test_early_return_when_no_workers(self, simple_network):
        net = simple_network
        net._persistent_workers = []
        # Must return promptly without touching queues.
        net._shutdown_worker_pool()
        assert net._persistent_workers == []


class TestTerminateWorkers:
    """``_terminate_workers`` SIGKILL-raises defensive arm."""

    def test_sigkill_failure_swallowed(self, simple_network):
        net = simple_network
        stubborn = _FakeWorker("stubborn", [True, True])
        net._persistent_workers = [stubborn]
        with patch.object(cc_mod.os, "kill", side_effect=OSError("no such process")):
            # Must not propagate — cleanup is best-effort.
            net._terminate_workers()
        assert stubborn.terminated is True


class TestSendShutdownSentinels:
    """``_send_shutdown_sentinels`` put-failure defensive arm."""

    def test_put_failure_swallowed(self, simple_network):
        net = simple_network
        net._persistent_workers = [_FakeWorker("w0", [False]), _FakeWorker("w1", [False])]
        net._persistent_task_queue = _FakeQueue(raise_on_put=True)
        # Every put raises; the method swallows each and completes.
        net._send_shutdown_sentinels()

    def test_noop_when_no_task_queue(self, simple_network):
        net = simple_network
        net._persistent_task_queue = None
        net._send_shutdown_sentinels()  # nothing to do


class TestSharedMemoryCleanup:
    """``_cleanup_shared_memory_blocks`` / ``_cleanup_shared_memory`` /
    ``_cleanup_pending_shared_memory`` — happy + swallowed-exception arms."""

    def test_cleanup_blocks_swallows_exceptions(self, simple_network):
        net = simple_network
        good_active = _FakeShm()
        bad_active = _FakeShm(raise_on={"close_and_unlink"})
        good_pending = _FakeShm()
        bad_pending = _FakeShm(raise_on={"unlink"})
        net._active_shm_blocks = [good_active, bad_active]
        net._pending_shm_unlinks = [good_pending, bad_pending]

        net._cleanup_shared_memory_blocks()

        assert good_active.closed and good_active.unlinked
        assert good_pending.unlinked
        assert net._active_shm_blocks == []
        assert net._pending_shm_unlinks == []

    def test_cleanup_shared_memory_emergency_swallows_exceptions(self, simple_network):
        net = simple_network
        good = _FakeShm()
        bad = _FakeShm(raise_on={"close_and_unlink"})
        pend_good = _FakeShm()
        pend_bad = _FakeShm(raise_on={"unlink"})
        net._active_shm_blocks = [good, bad]
        net._pending_shm_unlinks = [pend_good, pend_bad]

        net._cleanup_shared_memory()

        assert good.closed and good.unlinked
        assert pend_good.unlinked
        assert net._active_shm_blocks == []
        assert net._pending_shm_unlinks == []

    def test_cleanup_pending_swallows_exceptions(self, simple_network):
        net = simple_network
        good = _FakeShm()
        bad = _FakeShm(raise_on={"unlink"})
        net._pending_shm_unlinks = [good, bad]

        net._cleanup_pending_shared_memory()

        assert good.unlinked
        assert net._pending_shm_unlinks == []


class TestSharedTrainingMemoryReconstruct:
    """``SharedTrainingMemory.reconstruct_tensors`` edges (real /dev/shm)."""

    def test_roundtrip_including_zero_dim_tensor(self):
        stm = SharedTrainingMemory(
            tensors=[torch.zeros(3, 2), torch.ones(3), torch.tensor(5.0)],
            name_suffix="lift5rt",
        )
        try:
            tensors, handle = SharedTrainingMemory.reconstruct_tensors(stm.get_metadata())
            try:
                assert [tuple(t.shape) for t in tensors] == [(3, 2), (3,), ()]
            finally:
                handle.close()
        finally:
            stm.close_and_unlink()

    def test_invalid_magic_closes_and_raises(self):
        # Raw block with a deliberately wrong header magic.
        from multiprocessing.shared_memory import SharedMemory as _SM

        raw = _SM(name="juniper_train_lift5bad", create=True, size=SharedTrainingMemory.HEADER_SIZE + 8)
        try:
            struct.pack_into("<4sBB58x", raw.buf, 0, b"XXXX", 1, 0)
            with pytest.raises(ValueError, match="Invalid SharedMemory block header"):
                SharedTrainingMemory.reconstruct_tensors({"shm_name": "juniper_train_lift5bad"})
        finally:
            raw.close()
            raw.unlink()

    def test_legacy_python312_attach_branch(self):
        # Force the < 3.13 attach path (no ``track=`` kwarg) via a patched
        # version tuple; the real block still round-trips on the live runtime.
        stm = SharedTrainingMemory(tensors=[torch.zeros(2, 2)], name_suffix="lift5py312")
        try:
            with patch.object(cc_mod.sys, "version_info", (3, 12, 0)):
                tensors, handle = SharedTrainingMemory.reconstruct_tensors(stm.get_metadata())
            handle.close()
            assert tuple(tensors[0].shape) == (2, 2)
        finally:
            stm.close_and_unlink()


class TestSharedTrainingMemoryCloseUnlink:
    """``SharedTrainingMemory.close`` / ``unlink`` swallow underlying errors."""

    def _detached_stm(self):
        # Build without allocating /dev/shm so we can inject a raising handle.
        stm = object.__new__(SharedTrainingMemory)
        stm._closed = False
        stm._unlinked = False
        return stm

    def test_close_swallows_exception(self):
        stm = self._detached_stm()
        stm._shm = MagicMock()
        stm._shm.close.side_effect = RuntimeError("close failed")
        stm.close()
        assert stm._closed is True

    def test_unlink_swallows_exception(self):
        stm = self._detached_stm()
        stm._shm = MagicMock()
        stm._shm.unlink.side_effect = RuntimeError("unlink failed")
        stm.unlink()
        assert stm._unlinked is True


class TestCandidateTrainingManagerStart:
    """``CandidateTrainingManager.start`` method validation + delegation."""

    def test_start_with_no_method_delegates_to_super(self):
        mgr = CandidateTrainingManager()
        with patch.object(BaseManager, "start", return_value="started") as super_start:
            result = mgr.start()
        assert result == "started"
        super_start.assert_called_once()

    def test_start_with_valid_method_but_unavailable_context_raises(self):
        mgr = CandidateTrainingManager()
        with patch.object(cc_mod.mp, "get_context", side_effect=RuntimeError("no ctx")):
            with pytest.raises(NotImplementedError, match="not implemented on this platform"):
                mgr.start(method="forkserver")

    def test_start_with_invalid_method_raises_value_error(self):
        mgr = CandidateTrainingManager()
        with pytest.raises(ValueError, match="Invalid start method"):
            mgr.start(method="bogus")


class _RecordingAdvisoryQueue:
    """Advisory-queue stand-in recording ``cancel_join_thread`` calls (Issue #586)."""

    def __init__(self, *, raise_on_cancel=False):
        self.cancelled = 0
        self._raise = raise_on_cancel

    def cancel_join_thread(self):
        if self._raise:
            raise RuntimeError("cancel boom")
        self.cancelled += 1


class _DrainableQueue:
    """Progress-queue stand-in: ``get_nowait`` pops preset items, then raises (Issue #586)."""

    def __init__(self, items, event_log):
        self._items = list(items)
        self._event_log = event_log

    def get_nowait(self):
        if not self._items:
            raise RuntimeError("empty")
        self._event_log.append("drain")
        return self._items.pop(0)


class _SentinelLoggingQueue(_FakeQueue):
    """Task-queue stand-in that also appends to a shared event log (Issue #586)."""

    def __init__(self, event_log):
        super().__init__()
        self._event_log = event_log

    def put(self, item, timeout=None):
        self._event_log.append("sentinel")
        super().put(item, timeout=timeout)


def _stuff_queue_and_exit(q, n, release):
    """Child body for the Issue #586 exit-hang repro (fork ctx: no pickling needed).

    Fills an undrained mp.Queue past pipe capacity, optionally releases the advisory
    queue the way ``_worker_loop`` now does, then returns -- interpreter exit either
    hangs on the feeder flush (release=False: the pre-fix behaviour) or completes
    (release=True).
    """
    for i in range(n):
        q.put(("progress", i, "x" * 64))
    if release:
        cc_mod.CascadeCorrelationNetwork._release_advisory_queues(q, None, cc_mod.Logger)


class TestIssue586ShutdownHang:
    """Issue #586: cap-16 CLI pool teardown burned ~35 s -- 7/7 ungraceful stops.

    GUARDS (fail on the pre-fix code): the mp-level exit-hang repro's fixed arm, the
    shared-deadline budget test, and the drain-before-sentinels ordering test.
    NOT guards (pass either way): the raise-swallowing and missing-attribute arms --
    property assertions on the new helper, listed so they are not mistaken for
    regression coverage.
    """

    def test_release_advisory_queues_cancels_both_and_tolerates_gaps(self):
        prog = _RecordingAdvisoryQueue()
        instr = _RecordingAdvisoryQueue()
        cc_mod.CascadeCorrelationNetwork._release_advisory_queues(prog, instr, cc_mod.Logger)
        assert prog.cancelled == 1 and instr.cancelled == 1
        # None slots and stdlib queues (no cancel_join_thread) are tolerated.
        cc_mod.CascadeCorrelationNetwork._release_advisory_queues(None, None, cc_mod.Logger)
        cc_mod.CascadeCorrelationNetwork._release_advisory_queues(object(), None, cc_mod.Logger)
        # A raising cancel is swallowed (exit-path cleanup must never raise).
        cc_mod.CascadeCorrelationNetwork._release_advisory_queues(_RecordingAdvisoryQueue(raise_on_cancel=True), None, cc_mod.Logger)

    def test_worker_loop_releases_advisory_queues_on_sentinel(self):
        import queue as _stdlib_queue

        task_q = _stdlib_queue.Queue()
        task_q.put(None)  # immediate sentinel
        prog = _RecordingAdvisoryQueue()
        cc_mod.CascadeCorrelationNetwork._worker_loop(
            task_q,
            _stdlib_queue.Queue(),
            parallel=False,
            task_queue_timeout=0.5,
            worker_thread_count=1,
            shared_training_inputs=None,
            progress_queue=prog,
            instrumentation_queue=None,
        )
        assert prog.cancelled == 1

    def test_terminate_workers_join_budget_is_shared_not_serial(self, simple_network):
        net = simple_network
        workers = [_FakeWorker(f"w{i}", [True, True]) for i in range(7)]
        net._persistent_workers = workers

        clock = [0.0]

        def _fake_monotonic():
            return clock[0]

        real_join = _FakeWorker.join

        def _consuming_join(self, timeout=None):
            # Simulate the worst case: a stuck worker consumes its whole join timeout.
            clock[0] += timeout or 0.0
            return real_join(self, timeout=timeout)

        with patch.object(cc_mod.time, "monotonic", _fake_monotonic), patch.object(_FakeWorker, "join", _consuming_join), patch.object(cc_mod.os, "kill"):
            net._terminate_workers()

        phase1 = [w.join_timeouts[0] for w in workers]
        # First worker may consume the whole grace budget; the rest share the remainder.
        assert phase1[0] == pytest.approx(cc_mod._WORKER_SHUTDOWN_GRACE_SECONDS)
        assert sum(phase1) == pytest.approx(cc_mod._WORKER_SHUTDOWN_GRACE_SECONDS, abs=0.01)
        # Pre-fix behaviour was 5 s PER WORKER (35 s total) -- this is the regression guard.
        assert sum(phase1) < 2 * cc_mod._WORKER_SHUTDOWN_GRACE_SECONDS
        # Escalation still reached every stuck worker.
        assert all(w.terminated for w in workers)

    def test_terminate_workers_graceful_pool_pays_nothing_extra(self, simple_network):
        net = simple_network
        workers = [_FakeWorker(f"w{i}", [False]) for i in range(3)]
        net._persistent_workers = workers
        net._terminate_workers()
        assert not any(w.terminated for w in workers)
        assert all(len(w.join_timeouts) == 1 for w in workers)

    def test_shutdown_drains_progress_before_sentinels(self, simple_network):
        net = simple_network
        event_log = []
        net._persistent_workers = [_FakeWorker("w0", [False])]
        net._persistent_task_queue = _SentinelLoggingQueue(event_log)
        net._persistent_progress_queue = _DrainableQueue(["p1", "p2", "p3"], event_log)
        net._shutdown_worker_pool()
        assert "sentinel" in event_log and "drain" in event_log
        assert event_log.index("sentinel") > len(event_log) - 1 - event_log[::-1].index("drain") - 1
        assert event_log[:3] == ["drain", "drain", "drain"]

    @pytest.mark.timeout(30)
    def test_exit_hang_repro_and_fix(self):
        """The mechanism guard, both arms (Issue #586).

        UNFIXED arm: a child that stuffs an undrained mp.Queue past pipe capacity and
        exits WITHOUT releasing hangs at interpreter exit (feeder flush) -- this arm
        proves the repro reproduces, so the fixed arm below cannot pass vacuously.
        FIXED arm: the same child calling ``_release_advisory_queues`` exits promptly.
        """
        import multiprocessing as _mp

        ctx = _mp.get_context("fork")
        n_items = 20_000  # far past the ~64 KiB pipe capacity

        hang_q = ctx.Queue()
        hang_child = ctx.Process(target=_stuff_queue_and_exit, args=(hang_q, n_items, False), daemon=True)
        hang_child.start()
        hang_child.join(timeout=2.5)
        try:
            assert hang_child.is_alive(), "repro no longer reproduces -- the fixed arm below would prove nothing"
        finally:
            hang_child.terminate()
            hang_child.join(timeout=2.0)
            hang_q.cancel_join_thread()

        fixed_q = ctx.Queue()
        fixed_child = ctx.Process(target=_stuff_queue_and_exit, args=(fixed_q, n_items, True), daemon=True)
        fixed_child.start()
        fixed_child.join(timeout=2.5)
        try:
            assert not fixed_child.is_alive(), "worker still hangs at exit despite _release_advisory_queues"
        finally:
            if fixed_child.is_alive():
                fixed_child.terminate()
                fixed_child.join(timeout=2.0)
            fixed_q.cancel_join_thread()
