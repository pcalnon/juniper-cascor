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

    def is_alive(self):
        return self._alive.pop(0) if self._alive else False

    def join(self, timeout=None):
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
