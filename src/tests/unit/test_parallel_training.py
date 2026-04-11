"""Tests for parallel training infrastructure (CR-072).

Covers the parallel training path that is normally bypassed by the
force_sequential_training autouse fixture in conftest.py. Uses mocked
multiprocessing primitives for fast, reliable CI execution without
spawning real worker processes.

Tested components:
- _calculate_optimal_process_count (env override and CPU-based default)
- _drain_stale_results (queue draining for persistent pool hygiene)
- _collect_worker_results (bounded-timeout result collection)
- _ensure_worker_pool (pool reuse vs. recreation logic)
- SharedTrainingMemory (shared memory round-trip and lifecycle)
"""

import multiprocessing
import os
import queue
import uuid
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork, SharedTrainingMemory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_network(**kwargs):
    """Create a minimal CascadeCorrelationNetwork for testing."""
    defaults = dict(input_size=2, output_size=2)
    defaults.update(kwargs)
    return CascadeCorrelationNetwork(**defaults)


@dataclass
class FakeResult:
    """Lightweight stand-in for CandidateTrainingResult."""

    candidate_id: int
    correlation: float = 0.42
    round_id: str = None


# ---------------------------------------------------------------------------
# Fixture: restore the real _calculate_optimal_process_count
# ---------------------------------------------------------------------------

# The conftest autouse fixture force_sequential_training patches
# _calculate_optimal_process_count to return 1. Tests in this module that
# exercise the real method must undo that patch.

_original_calculate = CascadeCorrelationNetwork._calculate_optimal_process_count


@pytest.fixture()
def restore_process_count(monkeypatch):
    """Undo force_sequential_training so the real method runs."""
    monkeypatch.setattr(
        CascadeCorrelationNetwork,
        "_calculate_optimal_process_count",
        _original_calculate,
    )


# ===================================================================
# _calculate_optimal_process_count
# ===================================================================


class TestCalculateOptimalProcessCount:
    """Tests for CPU-core-based and env-override process count logic."""

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_defaults_returns_at_least_one(self, restore_process_count):
        """Without env override, the count should be >= 1."""
        net = _make_network()
        count = net._calculate_optimal_process_count()
        assert isinstance(count, int)
        assert count >= 1

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_env_override(self, restore_process_count, monkeypatch):
        """CASCOR_NUM_PROCESSES env var forces a specific count."""
        monkeypatch.setenv("CASCOR_NUM_PROCESSES", "7")
        net = _make_network()
        assert net._calculate_optimal_process_count() == 7

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_env_override_clamps_to_one(self, restore_process_count, monkeypatch):
        """Env override of 0 or negative is clamped to 1."""
        monkeypatch.setenv("CASCOR_NUM_PROCESSES", "0")
        net = _make_network()
        assert net._calculate_optimal_process_count() == 1

        monkeypatch.setenv("CASCOR_NUM_PROCESSES", "-3")
        assert net._calculate_optimal_process_count() == 1


# ===================================================================
# _drain_stale_results
# ===================================================================


class TestDrainStaleResults:
    """Tests for persistent-pool queue hygiene (RC-5)."""

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_clears_queue(self):
        """Draining a queue with items empties it and returns the count."""
        net = _make_network()
        q = queue.Queue()
        q.put(FakeResult(candidate_id=0))
        q.put(FakeResult(candidate_id=1))
        q.put(FakeResult(candidate_id=2))

        drained = net._drain_stale_results(q)

        assert drained == 3
        assert q.empty()

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_empty_queue_returns_zero(self):
        """Draining an already-empty queue returns 0."""
        net = _make_network()
        q = queue.Queue()
        assert net._drain_stale_results(q) == 0


# ===================================================================
# _collect_worker_results
# ===================================================================


class TestCollectWorkerResults:
    """Tests for bounded-timeout result collection from persistent workers."""

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_basic_collection(self):
        """Results are collected when workers are dead and queue has items."""
        net = _make_network()
        # Patch _collect_training_results to just drain the queue directly,
        # since the real implementation does validation we don't need here.
        results_out = [FakeResult(candidate_id=0), FakeResult(candidate_id=1)]

        def fake_collect(rq, num_tasks, round_id=None):
            return results_out

        net._collect_training_results = fake_collect

        # Mock workers that have already finished
        dead_worker = MagicMock()
        dead_worker.is_alive.return_value = False
        workers = [dead_worker]

        result_queue = queue.Queue()
        tasks = [("task0",), ("task1",)]
        round_id = str(uuid.uuid4())

        results = net._collect_worker_results(
            workers,
            result_queue,
            tasks,
            sleepytime=0.01,
            round_id=round_id,
        )

        assert len(results) == 2
        assert results[0].candidate_id == 0

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_exits_when_all_results_received(self):
        """Exits the wait loop early when qsize >= len(tasks)."""
        net = _make_network()
        net.task_queue_timeout = 5.0

        result_queue = queue.Queue()
        # Pre-fill the queue so qsize >= len(tasks) on first check
        result_queue.put(FakeResult(candidate_id=0))
        result_queue.put(FakeResult(candidate_id=1))

        results_out = [FakeResult(candidate_id=0), FakeResult(candidate_id=1)]

        def fake_collect(rq, num_tasks, round_id=None):
            return results_out

        net._collect_training_results = fake_collect

        alive_worker = MagicMock()
        alive_worker.is_alive.return_value = True
        workers = [alive_worker]
        tasks = [("task0",), ("task1",)]

        results = net._collect_worker_results(
            workers,
            result_queue,
            tasks,
            sleepytime=0.01,
            round_id="test-round",
        )

        assert len(results) == 2


# ===================================================================
# _ensure_worker_pool
# ===================================================================


class TestEnsureWorkerPool:
    """Tests for persistent pool reuse vs. recreation (RC-4)."""

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_reuses_valid_pool(self):
        """When all workers are alive and count matches, reuse the pool."""
        net = _make_network()

        # Simulate an existing valid pool
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        net._persistent_workers = [mock_worker, mock_worker]
        mock_task_q = MagicMock()
        mock_result_q = MagicMock()
        net._persistent_task_queue = mock_task_q
        net._persistent_result_queue = mock_result_q
        net._persistent_pool_size = 2

        task_q, result_q = net._ensure_worker_pool(num_workers=2)

        assert task_q is mock_task_q
        assert result_q is mock_result_q

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_recreates_when_size_changes(self):
        """Pool is recreated when requested size differs from current."""
        net = _make_network()

        # Existing pool of size 2
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        net._persistent_workers = [mock_worker, mock_worker]
        net._persistent_task_queue = MagicMock()
        net._persistent_result_queue = MagicMock()
        net._persistent_pool_size = 2

        # Mock _shutdown_worker_pool so it doesn't actually do process cleanup
        net._shutdown_worker_pool = MagicMock()
        # Mock _mp_ctx.Process and Queue to avoid real process spawning
        mock_process = MagicMock()
        mock_process.start = MagicMock()
        mock_process.pid = 12345
        net._mp_ctx = MagicMock()
        net._mp_ctx.Process.return_value = mock_process
        net._mp_ctx.Queue.return_value = MagicMock()

        task_q, result_q = net._ensure_worker_pool(num_workers=3)

        net._shutdown_worker_pool.assert_called_once()
        # New pool was created with 3 workers
        assert net._mp_ctx.Process.call_count == 3

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_recreates_when_worker_dies(self):
        """Pool is recreated when a worker has died."""
        net = _make_network()

        alive_worker = MagicMock()
        alive_worker.is_alive.return_value = True
        dead_worker = MagicMock()
        dead_worker.is_alive.return_value = False
        net._persistent_workers = [alive_worker, dead_worker]
        net._persistent_task_queue = MagicMock()
        net._persistent_result_queue = MagicMock()
        net._persistent_pool_size = 2

        net._shutdown_worker_pool = MagicMock()
        mock_process = MagicMock()
        mock_process.start = MagicMock()
        mock_process.pid = 99999
        net._mp_ctx = MagicMock()
        net._mp_ctx.Process.return_value = mock_process
        net._mp_ctx.Queue.return_value = MagicMock()

        task_q, result_q = net._ensure_worker_pool(num_workers=2)

        net._shutdown_worker_pool.assert_called_once()
        assert net._mp_ctx.Process.call_count == 2


# ===================================================================
# SharedTrainingMemory
# ===================================================================


class TestSharedTrainingMemory:
    """Tests for OPT-5 shared memory tensor sharing."""

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_round_trip_float32(self):
        """Create SharedTrainingMemory, reconstruct tensors, verify data matches."""
        t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        t2 = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32)

        suffix = f"test_{uuid.uuid4().hex[:8]}"
        shm = SharedTrainingMemory([t1, t2], name_suffix=suffix)
        try:
            metadata = shm.get_metadata()
            assert "shm_name" in metadata

            tensors, shm_handle = SharedTrainingMemory.reconstruct_tensors(metadata)
            try:
                assert len(tensors) == 2
                assert torch.allclose(tensors[0], t1)
                assert torch.allclose(tensors[1], t2)
                assert tensors[0].shape == (2, 2)
                assert tensors[1].shape == (3,)
            finally:
                shm_handle.close()
        finally:
            shm.close_and_unlink()

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_round_trip_float64(self):
        """Verify float64 tensors survive the round-trip."""
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        suffix = f"test_{uuid.uuid4().hex[:8]}"
        shm = SharedTrainingMemory([t], name_suffix=suffix)
        try:
            tensors, handle = SharedTrainingMemory.reconstruct_tensors(shm.get_metadata())
            try:
                assert torch.allclose(tensors[0], t)
                assert tensors[0].dtype == torch.float64
            finally:
                handle.close()
        finally:
            shm.close_and_unlink()

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_round_trip_int32(self):
        """Verify int32 tensors survive the round-trip."""
        t = torch.tensor([10, 20, 30], dtype=torch.int32)
        suffix = f"test_{uuid.uuid4().hex[:8]}"
        shm = SharedTrainingMemory([t], name_suffix=suffix)
        try:
            tensors, handle = SharedTrainingMemory.reconstruct_tensors(shm.get_metadata())
            try:
                assert torch.equal(tensors[0], t)
            finally:
                handle.close()
        finally:
            shm.close_and_unlink()

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_cleanup_lifecycle(self):
        """Test close() and unlink() lifecycle without errors."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        suffix = f"test_{uuid.uuid4().hex[:8]}"
        shm = SharedTrainingMemory([t], name_suffix=suffix)

        # close() should be idempotent
        shm.close()
        shm.close()
        assert shm._closed is True

        # unlink() should be idempotent
        shm.unlink()
        shm.unlink()
        assert shm._unlinked is True

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_close_and_unlink_convenience(self):
        """close_and_unlink() calls both in one shot."""
        t = torch.tensor([1.0], dtype=torch.float32)
        suffix = f"test_{uuid.uuid4().hex[:8]}"
        shm = SharedTrainingMemory([t], name_suffix=suffix)
        shm.close_and_unlink()
        assert shm._closed is True
        assert shm._unlinked is True

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_unsupported_dtype_raises(self):
        """Unsupported tensor dtypes raise ValueError."""
        t = torch.tensor([True, False], dtype=torch.bool)
        suffix = f"test_{uuid.uuid4().hex[:8]}"
        with pytest.raises(ValueError, match="Unsupported tensor dtype"):
            SharedTrainingMemory([t], name_suffix=suffix)

    @pytest.mark.unit
    @pytest.mark.multiprocessing
    def test_name_property(self):
        """The name property returns the expected shm block name."""
        t = torch.tensor([1.0], dtype=torch.float32)
        suffix = f"test_{uuid.uuid4().hex[:8]}"
        shm = SharedTrainingMemory([t], name_suffix=suffix)
        try:
            assert shm.name == f"juniper_train_{suffix}"
        finally:
            shm.close_and_unlink()
