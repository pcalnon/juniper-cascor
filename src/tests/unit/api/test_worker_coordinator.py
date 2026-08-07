"""Tests for WorkerCoordinator — task distribution, result aggregation, and health monitoring."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.workers.coordinator import PendingTask, TaskResult, WorkerCoordinator
from api.workers.registry import WorkerRegistry


@pytest.fixture
def registry():
    """Create a WorkerRegistry with short heartbeat timeout for testing."""
    return WorkerRegistry(heartbeat_timeout=30.0)


@pytest.fixture
def coordinator(registry):
    """Create a WorkerCoordinator with test-friendly timeouts."""
    coord = WorkerCoordinator(
        registry=registry,
        task_reassignment_timeout=5.0,
        health_check_interval=0.5,
    )
    yield coord
    coord.shutdown()


def _make_tensors():
    """Create test tensors."""
    return {
        "candidate_input": np.random.randn(100, 4).astype(np.float32),
        "y": np.random.randn(100, 1).astype(np.float32),
        "residual_error": np.random.randn(100, 1).astype(np.float32),
    }


def _make_task_specs(n=2):
    """Create n task specifications."""
    return [
        {
            "candidate_index": i,
            "candidate_data": {"input_size": 4, "activation_name": "sigmoid"},
            "training_params": {"epochs": 10, "learning_rate": 0.01},
        }
        for i in range(n)
    ]


def _make_result_msg(task_id, candidate_id=0, correlation=0.85):
    """Create a valid task result message."""
    return {
        "type": "task_result",
        "task_id": task_id,
        "candidate_id": candidate_id,
        "candidate_uuid": "test-uuid",
        "correlation": correlation,
        "success": True,
        "epochs_completed": 10,
        "activation_name": "sigmoid",
        "all_correlations": [0.1, 0.5, correlation],
        "numerator": 1.0,
        "denominator": 2.0,
        "best_corr_idx": 9,
        "error_message": None,
        "tensor_manifest": {
            "weights": {"shape": [4], "dtype": "float32"},
            "bias": {"shape": [1], "dtype": "float32"},
        },
    }


def _make_result_tensors():
    """Create valid result tensors matching the manifest."""
    return {
        "weights": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "bias": np.array([0.5], dtype=np.float32),
    }


@pytest.mark.unit
class TestSubmitTasks:
    """Test task submission."""

    def test_submit_tasks_returns_ids(self, coordinator):
        """submit_tasks returns task IDs for all tasks."""
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(3), _make_tensors())
        assert len(task_ids) == 3
        assert all(isinstance(tid, str) for tid in task_ids)

    def test_submit_tasks_sets_round(self, coordinator):
        """submit_tasks sets the current round."""
        coordinator.submit_tasks("round-1", _make_task_specs(), _make_tensors())
        assert coordinator._current_round_id == "round-1"
        assert coordinator._current_round_task_count == 2

    def test_has_pending_tasks(self, coordinator):
        """has_pending_tasks reflects unassigned task state."""
        assert coordinator.has_pending_tasks() is False
        coordinator.submit_tasks("round-1", _make_task_specs(), _make_tensors())
        assert coordinator.has_pending_tasks() is True


@pytest.mark.unit
class TestGetNextAssignment:
    """Test task dispatch to workers."""

    def test_get_next_assignment(self, coordinator, registry):
        """Assignment returns message and binary frames."""
        registry.register("w1", {})
        coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())

        result = coordinator.get_next_assignment("w1")
        assert result is not None
        msg, frames = result
        assert msg["type"] == "task_assign"
        assert msg["candidate_index"] == 0
        assert len(frames) == 3  # candidate_input, y, residual_error

    def test_get_next_assignment_none_when_empty(self, coordinator, registry):
        """Returns None when no tasks pending."""
        registry.register("w1", {})
        assert coordinator.get_next_assignment("w1") is None

    def test_assignment_marks_worker_busy(self, coordinator, registry):
        """After assignment, worker is no longer idle."""
        registry.register("w1", {})
        coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        coordinator.get_next_assignment("w1")
        assert registry.get("w1").idle is False

    def test_get_next_assignment_skips_orphaned_unassigned_id(self, coordinator, registry):
        """Stale queue ids without a pending entry must not block later tasks."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        # Plant an orphan ahead of the real task id.
        coordinator._unassigned_tasks.insert(0, "orphaned-task-id")
        assert coordinator._pending_tasks.get("orphaned-task-id") is None

        result = coordinator.get_next_assignment("w1")
        assert result is not None
        msg, _frames = result
        assert msg["task_id"] == task_ids[0]
        assert "orphaned-task-id" not in coordinator._unassigned_tasks


@pytest.mark.unit
class TestAbortInFlightResult:
    """Soft result-frame abort: free worker + immediate requeue (conn stays open)."""

    def test_soft_abort_frees_worker_and_requeues(self, coordinator, registry):
        """abort_in_flight_result clears busy state and puts the task back on the queue."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        assert coordinator.get_next_assignment("w1") is not None
        assert registry.get("w1").idle is False

        coordinator.abort_in_flight_result("w1", task_ids[0])

        assert registry.get("w1").idle is True
        pending = coordinator._pending_tasks[task_ids[0]]
        assert pending.assigned_worker_id is None
        assert task_ids[0] in coordinator._unassigned_tasks

    def test_soft_abort_allows_immediate_peer_reassignment(self, coordinator, registry):
        """After soft abort, a second idle worker can pick up the same task without waiting."""
        registry.register("w1", {})
        registry.register("w2", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        assert coordinator.get_next_assignment("w1") is not None

        coordinator.abort_in_flight_result("w1", task_ids[0])

        peer = coordinator.get_next_assignment("w2")
        assert peer is not None
        msg, _frames = peer
        assert msg["task_id"] == task_ids[0]
        assert registry.get("w2").idle is False
        assert registry.get("w1").idle is True

    def test_soft_abort_unknown_task_still_frees_worker(self, coordinator, registry):
        """Unknown task_id still clears the worker's active assignment (fail-soft)."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        assert coordinator.get_next_assignment("w1") is not None

        coordinator.abort_in_flight_result("w1", "not-a-real-task")

        assert registry.get("w1").idle is True
        # Original task remains assigned at coordinator level until timeout /
        # disconnect path; soft abort with wrong id must not invent a requeue.
        assert task_ids[0] not in coordinator._unassigned_tasks

    def test_soft_abort_uses_registry_active_task_when_id_omitted(self, coordinator, registry):
        """Omitting task_id falls back to the registry's active_task_id."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        assert coordinator.get_next_assignment("w1") is not None

        coordinator.abort_in_flight_result("w1")

        assert registry.get("w1").idle is True
        assert task_ids[0] in coordinator._unassigned_tasks


@pytest.mark.unit
class TestSubmitResult:
    """Test result submission and validation."""

    def test_accept_valid_result(self, coordinator, registry):
        """Valid result is accepted."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        coordinator.get_next_assignment("w1")

        msg = _make_result_msg(task_ids[0])
        accepted = coordinator.submit_result("w1", msg, _make_result_tensors())
        assert accepted is True

    def test_reject_duplicate_result(self, coordinator, registry):
        """Duplicate result for same task_id is rejected."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        coordinator.get_next_assignment("w1")

        msg = _make_result_msg(task_ids[0])
        coordinator.submit_result("w1", msg, _make_result_tensors())

        # Re-register worker (simulate reconnect) and try duplicate
        registry.register("w1", {})
        accepted = coordinator.submit_result("w1", msg, _make_result_tensors())
        assert accepted is False

    def test_reject_unknown_task(self, coordinator, registry):
        """Result for unknown task_id is rejected."""
        registry.register("w1", {})
        msg = _make_result_msg("nonexistent-task")
        accepted = coordinator.submit_result("w1", msg, _make_result_tensors())
        assert accepted is False

    def test_reject_invalid_schema(self, coordinator, registry):
        """Result with invalid schema is rejected."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        coordinator.get_next_assignment("w1")

        msg = _make_result_msg(task_ids[0])
        msg["correlation"] = 5.0  # Out of bounds
        accepted = coordinator.submit_result("w1", msg, _make_result_tensors())
        assert accepted is False

    def test_reject_invalid_tensors(self, coordinator, registry):
        """Result with NaN tensors is rejected."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        coordinator.get_next_assignment("w1")

        msg = _make_result_msg(task_ids[0])
        bad_tensors = _make_result_tensors()
        bad_tensors["weights"] = np.array([float("nan")] * 4, dtype=np.float32)
        accepted = coordinator.submit_result("w1", msg, bad_tensors)
        assert accepted is False

    def test_reject_success_true_with_missing_weights(self, coordinator, registry):
        """success=True with no weights tensor must not be accepted (poison-candidate guard)."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        coordinator.get_next_assignment("w1")

        msg = _make_result_msg(task_ids[0])
        msg["tensor_manifest"] = {}
        accepted = coordinator.submit_result("w1", msg, {})
        assert accepted is False
        assert registry.get("w1").idle is True
        assert task_ids[0] not in coordinator._results

    def test_reject_success_true_with_empty_weights(self, coordinator, registry):
        """success=True with an empty weights array is rejected."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        coordinator.get_next_assignment("w1")

        msg = _make_result_msg(task_ids[0])
        msg["tensor_manifest"] = {"weights": {"shape": [0], "dtype": "float32"}}
        empty = {"weights": np.array([], dtype=np.float32)}
        accepted = coordinator.submit_result("w1", msg, empty)
        assert accepted is False
        assert registry.get("w1").idle is True

    def test_accept_success_false_without_weights(self, coordinator, registry):
        """Failed results may omit weights — rejection is only for success=True."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        coordinator.get_next_assignment("w1")

        msg = _make_result_msg(task_ids[0])
        msg["success"] = False
        msg["tensor_manifest"] = {}
        accepted = coordinator.submit_result("w1", msg, {})
        assert accepted is True
        assert task_ids[0] in coordinator._results

    def test_rejected_schema_result_requeues_assigned_task(self, coordinator, registry):
        """Schema rejection frees the worker and requeues immediately (no timeout wait)."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        task_id = task_ids[0]
        coordinator.get_next_assignment("w1")

        msg = _make_result_msg(task_id)
        msg["correlation"] = 5.0  # Out of bounds → schema reject
        assert coordinator.submit_result("w1", msg, _make_result_tensors()) is False

        pending = coordinator._pending_tasks[task_id]
        assert pending.completed is False
        assert pending.assigned_worker_id is None
        assert task_id in coordinator._unassigned_tasks
        assert registry.get("w1").active_task_id is None
        assert registry.get("w1").idle is True

    def test_rejected_schema_result_can_be_reassigned_immediately(self, coordinator, registry):
        """After a rejected result, a second worker can pick up the same task without waiting."""
        registry.register("w1", {})
        registry.register("w2", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        task_id = task_ids[0]
        coordinator.get_next_assignment("w1")

        msg = _make_result_msg(task_id)
        msg["correlation"] = 5.0
        assert coordinator.submit_result("w1", msg, _make_result_tensors()) is False

        reassigned = coordinator.get_next_assignment("w2")
        assert reassigned is not None
        reassigned_msg, _frames = reassigned
        assert reassigned_msg["task_id"] == task_id
        assert coordinator._pending_tasks[task_id].assigned_worker_id == "w2"

    def test_rejected_tensor_result_requeues_assigned_task(self, coordinator, registry):
        """Tensor validation failure uses the same immediate-requeue contract."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        task_id = task_ids[0]
        coordinator.get_next_assignment("w1")

        msg = _make_result_msg(task_id)
        bad_tensors = _make_result_tensors()
        bad_tensors["weights"] = np.array([float("nan")] * 4, dtype=np.float32)
        assert coordinator.submit_result("w1", msg, bad_tensors) is False

        assert coordinator._pending_tasks[task_id].assigned_worker_id is None
        assert task_id in coordinator._unassigned_tasks
        assert registry.get("w1").idle is True

    def test_reject_unassigned_task_result(self, coordinator, registry):
        """Results for pending-but-unassigned tasks (requeue window) are rejected."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        # Do not call get_next_assignment — task stays unassigned.
        assert coordinator._pending_tasks[task_ids[0]].assigned_worker_id is None

        accepted = coordinator.submit_result("w1", _make_result_msg(task_ids[0]), _make_result_tensors())
        assert accepted is False
        assert task_ids[0] not in coordinator._completed_task_ids

    def test_reject_wrong_worker_ownership(self, coordinator, registry):
        """A peer worker cannot submit a result for a task assigned to another worker."""
        registry.register("w1", {})
        registry.register("w2", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        coordinator.get_next_assignment("w1")

        msg = _make_result_msg(task_ids[0])
        accepted = coordinator.submit_result("w2", msg, _make_result_tensors())
        assert accepted is False

        # Task remains pending under the original assignee.
        pending = coordinator._pending_tasks[task_ids[0]]
        assert pending.completed is False
        assert pending.assigned_worker_id == "w1"
        assert task_ids[0] not in coordinator._completed_task_ids

        # Legitimate owner can still complete the task afterward.
        accepted = coordinator.submit_result("w1", msg, _make_result_tensors())
        assert accepted is True


@pytest.mark.unit
class TestRequeueAfterDispatchFailure:
    """Send-side dispatch failure must free the worker and requeue immediately."""

    def test_requeue_frees_worker_and_returns_task(self, coordinator, registry):
        """requeue_after_dispatch_failure clears busy state and restores the queue."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        assert coordinator.get_next_assignment("w1") is not None
        assert registry.get("w1").idle is False

        coordinator.requeue_after_dispatch_failure("w1", task_ids[0])

        assert registry.get("w1").idle is True
        assert task_ids[0] in coordinator._unassigned_tasks
        assert coordinator._pending_tasks[task_ids[0]].assigned_worker_id is None

    def test_requeue_allows_peer_to_pick_up_immediately(self, coordinator, registry):
        """After send-failure requeue, another idle worker can take the task without waiting."""
        registry.register("w1", {})
        registry.register("w2", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        assert coordinator.get_next_assignment("w1") is not None

        coordinator.requeue_after_dispatch_failure("w1", task_ids[0])

        peer = coordinator.get_next_assignment("w2")
        assert peer is not None
        assert peer[0]["task_id"] == task_ids[0]
        assert registry.get("w2").idle is False
        assert registry.get("w1").idle is True

    def test_requeue_unknown_task_still_frees_worker(self, coordinator, registry):
        """Unknown task_id still clears the worker's active assignment (fail-soft)."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        assert coordinator.get_next_assignment("w1") is not None

        coordinator.requeue_after_dispatch_failure("w1", "not-a-real-task")

        assert registry.get("w1").idle is True
        assert task_ids[0] not in coordinator._unassigned_tasks


@pytest.mark.unit
class TestCollectResults:
    """Test blocking result collection."""

    def test_collect_all_results(self, coordinator, registry):
        """collect_results returns all submitted results."""
        registry.register("w1", {})
        registry.register("w2", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(2), _make_tensors())

        coordinator.get_next_assignment("w1")
        coordinator.get_next_assignment("w2")

        coordinator.submit_result("w1", _make_result_msg(task_ids[0], candidate_id=0), _make_result_tensors())
        coordinator.submit_result("w2", _make_result_msg(task_ids[1], candidate_id=1), _make_result_tensors())

        results = coordinator.collect_results(timeout=5.0)
        assert len(results) == 2

    def test_collect_partial_on_timeout(self, coordinator, registry):
        """collect_results returns partial results when timeout expires."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(2), _make_tensors())

        coordinator.get_next_assignment("w1")
        coordinator.submit_result("w1", _make_result_msg(task_ids[0]), _make_result_tensors())

        # Only 1 of 2 results submitted — timeout will expire
        results = coordinator.collect_results(timeout=0.1)
        assert len(results) == 1

    def test_collect_results_attaches_round_id(self, coordinator, registry):
        """ISSUE-319 (defect #4): the accepted TaskResult carries the round_id of the
        dispatching round so the dual-path consumer can round-isolate remote results."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-xyz", _make_task_specs(1), _make_tensors())
        coordinator.get_next_assignment("w1")
        coordinator.submit_result("w1", _make_result_msg(task_ids[0], candidate_id=0), _make_result_tensors())

        results = coordinator.collect_results(timeout=5.0)
        assert len(results) == 1
        assert results[0].round_id == "round-xyz"

    def test_collect_results_early_exit_when_all_workers_gone(self, coordinator, registry):
        """ISSUE-319 (defect #3 safety): collect_results stops waiting promptly when no
        workers remain connected, instead of blocking for the full (training-scaled)
        budget. Without the liveness early-exit this would wait out the whole timeout."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(2), _make_tensors())
        coordinator.get_next_assignment("w1")
        coordinator.submit_result("w1", _make_result_msg(task_ids[0], candidate_id=0), _make_result_tensors())

        # Worker disconnects with 1 of 2 results outstanding.
        registry.deregister("w1")
        start = time.monotonic()
        results = coordinator.collect_results(timeout=30.0)
        elapsed = time.monotonic() - start

        assert len(results) == 1, "returns the partial result collected before the worker left"
        assert elapsed < 5.0, f"must early-exit on loss of all workers, not wait out the 30s budget (took {elapsed:.1f}s)"


@pytest.mark.unit
class TestCancelRound:
    """Test round cancellation."""

    def test_cancel_clears_state(self, coordinator, registry):
        """cancel_round clears all pending tasks and results."""
        registry.register("w1", {})
        coordinator.submit_tasks("round-1", _make_task_specs(), _make_tensors())
        coordinator.cancel_round()
        assert coordinator.has_pending_tasks() is False
        assert coordinator._current_round_id is None

    def test_cancel_frees_registry_active_task(self, coordinator, registry):
        """cancel_round must clear registry active_task_id so the worker is idle again.

        Regression: clearing only coordinator pending left ``active_task_id`` set,
        so ``get_next_assignment`` forever refused (assign_task → False) and
        ``_check_task_timeouts`` could not reclaim capacity (pending already gone).
        """
        registry.register("w1", {})
        coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        assert coordinator.get_next_assignment("w1") is not None
        assert registry.get("w1").active_task_id is not None
        assert registry.get("w1").idle is False

        coordinator.cancel_round()

        assert registry.get("w1").active_task_id is None
        assert registry.get("w1").idle is True
        # Worker can accept a fresh round without waiting for reconnect.
        coordinator.submit_tasks("round-2", _make_task_specs(1), _make_tensors())
        assert coordinator.get_next_assignment("w1") is not None


@pytest.mark.unit
class TestHealthMonitor:
    """Test background health monitoring."""

    def test_start_stop_monitor(self, coordinator):
        """Health monitor thread starts and stops cleanly."""
        coordinator.start_monitor()
        assert coordinator._monitor_thread is not None
        assert coordinator._monitor_thread.is_alive()
        coordinator.stop_monitor()
        assert coordinator._monitor_thread is None

    def test_stale_worker_cleanup(self, coordinator):
        """Stale workers are cleaned up by health monitor."""
        registry = coordinator._registry
        registry._heartbeat_timeout = 0.01  # Very short timeout
        registry.register("w1", {})
        time.sleep(0.02)

        coordinator._check_stale_workers()
        assert registry.worker_count == 0

    def test_task_reassignment_on_timeout(self, coordinator, registry):
        """Timed-out tasks are returned to the unassigned queue."""
        coordinator._task_reassignment_timeout = 0.01

        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        coordinator.get_next_assignment("w1")

        assert coordinator.has_pending_tasks() is False
        time.sleep(0.02)
        coordinator._check_task_timeouts()
        assert coordinator.has_pending_tasks() is True


@pytest.mark.unit
class TestSendCallbacks:
    """Test send callback registration."""

    def test_register_and_unregister(self, coordinator):
        """Send callbacks can be registered and unregistered."""
        mock_cb = MagicMock()
        coordinator.register_send_callback("w1", mock_cb)
        assert "w1" in coordinator._send_callbacks
        coordinator.unregister_send_callback("w1")
        assert "w1" not in coordinator._send_callbacks


@pytest.mark.unit
class TestHandleWorkerDisconnect:
    """Clean disconnect must requeue in-flight work immediately (not wait timeout)."""

    def test_disconnect_with_active_task_requeues_immediately(self, coordinator, registry):
        """Active task returns to the unassigned queue on clean disconnect."""
        registry.register("w1", {})
        registry.register("w2", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        coordinator.register_send_callback("w1", MagicMock())
        coordinator.get_next_assignment("w1")

        assert coordinator.has_pending_tasks() is False
        assert registry.get("w1").active_task_id == task_ids[0]

        coordinator.handle_worker_disconnect("w1")

        assert registry.get("w1") is None
        assert "w1" not in coordinator._send_callbacks
        assert coordinator.has_pending_tasks() is True
        assert task_ids[0] in coordinator._unassigned_tasks
        assert coordinator._pending_tasks[task_ids[0]].assigned_worker_id is None

        # Peer can pick the requeued task without waiting for reassignment timeout.
        assignment = coordinator.get_next_assignment("w2")
        assert assignment is not None
        assert assignment[0]["task_id"] == task_ids[0]

    def test_disconnect_without_active_task_only_deregisters(self, coordinator, registry):
        """Idle disconnect still deregisters and drops the send callback."""
        registry.register("w1", {})
        coordinator.register_send_callback("w1", MagicMock())
        coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())

        coordinator.handle_worker_disconnect("w1")

        assert registry.get("w1") is None
        assert "w1" not in coordinator._send_callbacks
        # Unassigned task remains available for other workers.
        assert coordinator.has_pending_tasks() is True

    def test_disconnect_unknown_worker_is_noop(self, coordinator, registry):
        """Unknown worker_id does not raise and does not mutate the queue."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        coordinator.get_next_assignment("w1")
        before = list(coordinator._unassigned_tasks)

        coordinator.handle_worker_disconnect("missing")

        assert list(coordinator._unassigned_tasks) == before
        assert registry.get("w1") is not None
        assert coordinator._pending_tasks[task_ids[0]].assigned_worker_id == "w1"

    def test_disconnect_idempotent_when_task_already_unassigned(self, coordinator, registry):
        """Second disconnect after requeue does not duplicate the unassigned entry."""
        registry.register("w1", {})
        task_ids = coordinator.submit_tasks("round-1", _make_task_specs(1), _make_tensors())
        coordinator.get_next_assignment("w1")

        coordinator.handle_worker_disconnect("w1")
        assert coordinator._unassigned_tasks.count(task_ids[0]) == 1

        # Worker already gone — second call is a no-op.
        coordinator.handle_worker_disconnect("w1")
        assert coordinator._unassigned_tasks.count(task_ids[0]) == 1
