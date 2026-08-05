"""Task distribution and result aggregation for remote WebSocket workers.

The WorkerCoordinator distributes candidate training tasks to remote workers
via WebSocket, collects results, and handles task timeouts/reassignment.

Thread-safety: The coordinator is called from both the async WebSocket handler
(for result submissions) and the synchronous training thread (for task dispatch).
All shared state is protected by locks.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any

import numpy as np

from api.workers.protocol import BinaryFrame, WorkerProtocol
from api.workers.registry import WorkerRegistry

logger = logging.getLogger("juniper_cascor.api.workers.coordinator")


# ISSUE-319 (defect #3): poll cadence for collect_results. The wait is sliced into
# intervals of this length so the worker-liveness early-exit can fire promptly when all
# workers disconnect mid-round, rather than blocking for the full (training-scaled)
# collection budget. Small enough to be responsive, large enough to be cheap.
_RESULT_COLLECTION_POLL_INTERVAL = 1.0


@dataclass
class PendingTask:
    """A task that has been dispatched but not yet completed."""

    task_id: str
    round_id: str
    candidate_index: int
    candidate_data: dict[str, Any]
    training_params: dict[str, Any]
    tensors: dict[str, np.ndarray]
    assigned_worker_id: str | None = None
    dispatched_at: float = field(default_factory=time.time)
    completed: bool = False


@dataclass
class TaskResult:
    """A validated result from a remote worker."""

    task_id: str
    candidate_id: int
    candidate_uuid: str
    correlation: float
    success: bool
    epochs_completed: int
    activation_name: str
    all_correlations: list[float]
    numerator: float
    denominator: float
    best_corr_idx: int
    tensors: dict[str, np.ndarray]
    error_message: str | None = None
    # ISSUE-319 (defect #4): authoritative training-round id, attached server-side in
    # submit_result from the dispatching PendingTask. Lets the dual-path consumer
    # (_dispatch_to_remote_workers) round-isolate remote results the way RC-5 already
    # does for the local pool, so a late result for a stale pending task cannot leak into
    # a later round's collection. Optional/defaulted for back-compat with any TaskResult
    # constructed without it.
    round_id: str | None = None


class WorkerCoordinator:
    """Coordinates task distribution and result collection for remote workers.

    The coordinator manages the lifecycle of training tasks:
    1. Tasks are submitted by the training thread via submit_tasks()
    2. Tasks are dispatched to idle workers via dispatch_pending()
    3. Results are collected from workers via submit_result()
    4. Completed results are retrieved by the training thread via collect_results()

    A background health monitor thread handles stale workers and task reassignment.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        task_reassignment_timeout: float = 120.0,
        health_check_interval: float = 10.0,
    ) -> None:
        self._registry = registry
        self._task_reassignment_timeout = task_reassignment_timeout
        self._health_check_interval = health_check_interval
        self._anomaly_detector: Any | None = None

        # Task tracking
        self._pending_tasks: dict[str, PendingTask] = {}  # task_id -> PendingTask
        self._unassigned_tasks: list[str] = []  # task_ids waiting for workers
        self._results: dict[str, TaskResult] = {}  # task_id -> TaskResult
        self._completed_task_ids: set[str] = set()  # for duplicate detection
        self._lock = Lock()

        # Current round
        self._current_round_id: str | None = None
        self._current_round_task_count: int = 0
        self._results_ready = Event()

        # Health monitor thread
        self._monitor_stop = Event()
        self._monitor_thread: Thread | None = None

        # WebSocket send callback (set by worker_stream)
        self._send_callbacks: dict[str, Any] = {}  # worker_id -> async callback

        logger.info(
            "WorkerCoordinator initialized (reassignment_timeout=%.1fs, health_check=%.1fs)",
            task_reassignment_timeout,
            health_check_interval,
        )

    def start_monitor(self) -> None:
        """Start the background health monitoring thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = Thread(
            target=self._health_monitor_loop,
            name="worker-health-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Health monitor thread started")

    def stop_monitor(self) -> None:
        """Stop the background health monitoring thread."""
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("Health monitor thread stopped")

    def register_send_callback(self, worker_id: str, callback: Any) -> None:
        """Register an async send callback for a worker connection."""
        with self._lock:
            self._send_callbacks[worker_id] = callback

    def unregister_send_callback(self, worker_id: str) -> None:
        """Remove the send callback for a disconnected worker."""
        with self._lock:
            self._send_callbacks.pop(worker_id, None)

    def submit_tasks(
        self,
        round_id: str,
        tasks: list[dict[str, Any]],
        tensors: dict[str, np.ndarray],
    ) -> list[str]:
        """Submit a batch of training tasks for the current round.

        Called by the training thread at the start of a candidate training round.

        Args:
            round_id: Unique identifier for this training round.
            tasks: List of task dicts with candidate_data and training_params.
            tensors: Shared training tensors (candidate_input, y, residual_error).

        Returns:
            List of task_ids for the submitted tasks.
        """
        with self._lock:
            # Round boundary: drop leftover pending/unassigned entries from any
            # prior round. ``_results`` / ``_completed_task_ids`` were already
            # cleared here, but stale ``_pending_tasks`` remained — a late
            # ``submit_result`` for a previous-round task_id was re-accepted and
            # could satisfy ``len(_results) >= _current_round_task_count``,
            # early-unblocking ``collect_results`` before the new round's real
            # work finished (ISSUE-319 class; cascade filters by round_id after
            # collection, but the coordinator wait must not unblock early).
            self._pending_tasks.clear()
            self._unassigned_tasks.clear()
            self._current_round_id = round_id
            self._current_round_task_count = len(tasks)
            self._results_ready.clear()
            self._results.clear()
            self._completed_task_ids.clear()
            task_ids = []

            for task_spec in tasks:
                task_id = str(uuid.uuid4())
                pending = PendingTask(
                    task_id=task_id,
                    round_id=round_id,
                    candidate_index=task_spec["candidate_index"],
                    candidate_data=task_spec["candidate_data"],
                    training_params=task_spec["training_params"],
                    tensors=tensors,
                )
                self._pending_tasks[task_id] = pending
                self._unassigned_tasks.append(task_id)
                task_ids.append(task_id)

            logger.info("Submitted %d tasks for round %s", len(tasks), round_id)
            return task_ids

    def get_next_assignment(self, worker_id: str) -> tuple[dict[str, Any], list[bytes]] | None:
        """Get the next task assignment for a worker.

        Called by the WebSocket handler when a worker is ready for work.

        Returns:
            Tuple of (JSON message, list of binary frames) or None if no tasks available.
        """
        with self._lock:
            # CONC-10 (Phase 3D): closing this race requires a registry
            # check inside the same critical section that pops the task.
            # `_check_stale_workers` deregisters under `self._lock`, so if
            # the worker is already gone we MUST NOT pop a task on its
            # behalf — otherwise the task ends up assigned to a worker
            # that no longer exists and waits the full
            # `_task_reassignment_timeout` before the next reaper sweep
            # picks it up.
            if self._registry.get(worker_id) is None:
                return None
            if not self._unassigned_tasks:
                return None

            # Find the next unassigned task
            task_id = self._unassigned_tasks.pop(0)
            task = self._pending_tasks.get(task_id)
            if task is None:
                return None

            # Assign to worker — confirm with the registry FIRST. The registry
            # enforces one active task per worker (assign_task returns False for
            # a busy or unknown worker); the pre-fix flow ignored that refusal,
            # marked the PendingTask assigned, and sent it anyway, so a busy
            # worker accumulated coordinator-level assignments the registry never
            # tracked. `_check_stale_workers` requeues only the registry's
            # `active_task_id` on deregistration, orphaning the extras until
            # `_task_reassignment_timeout` (the worker_stream heartbeat path
            # already guarded this with its own `reg.idle` check — ISSUE-319).
            # Exposed by fix C4: re-enabled coordinator/registry logging widened
            # the assign/dereg race window the CONC-10 suite hammers.
            if not self._registry.assign_task(worker_id, task_id):
                self._unassigned_tasks.insert(0, task_id)
                logger.debug("Refusing assignment of task %s to worker %s (busy or unregistered)", task_id, worker_id)
                return None
            task.assigned_worker_id = worker_id
            task.dispatched_at = time.time()

            # Build assignment message
            tensor_manifest = {}
            frames = []
            for tensor_name, arr in task.tensors.items():
                tensor_manifest[tensor_name] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }
                frames.append(BinaryFrame.encode(arr))

            msg = WorkerProtocol.build_task_assign(
                task_id=task_id,
                round_id=task.round_id,
                candidate_index=task.candidate_index,
                candidate_data=task.candidate_data,
                training_params=task.training_params,
                tensor_manifest=tensor_manifest,
            )

            logger.debug("Assigned task %s to worker %s (candidate %d)", task_id, worker_id, task.candidate_index)
            return msg, frames

    def submit_result(
        self,
        worker_id: str,
        msg: dict[str, Any],
        tensors: dict[str, np.ndarray],
    ) -> bool:
        """Submit a task result from a worker.

        Called by the WebSocket handler when a worker sends a task_result message.

        Args:
            worker_id: ID of the worker submitting the result.
            msg: The validated task_result JSON message.
            tensors: Decoded tensor data from binary frames.

        Returns:
            True if the result was accepted, False if rejected.
        """
        task_id = msg.get("task_id")

        with self._lock:
            # Duplicate detection (Section 12.7 rule 8)
            if task_id in self._completed_task_ids:
                logger.warning("Duplicate result for task %s from worker %s — rejected", task_id, worker_id)
                return False

            # Task tracking (Section 12.7 rule 2)
            task = self._pending_tasks.get(task_id)
            if task is None:
                logger.warning("Result for unknown task %s from worker %s — rejected", task_id, worker_id)
                return False

            # Stale-round defense: reject results whose PendingTask belongs to a
            # prior round even if the entry somehow lingered in ``_pending_tasks``
            # (e.g. races with a concurrent ``submit_tasks``). Complements the
            # pending clear at round start above.
            if self._current_round_id is not None and task.round_id != self._current_round_id:
                logger.warning(
                    "Result for stale-round task %s (task round %s, current %s) from worker %s — rejected",
                    task_id,
                    task.round_id,
                    self._current_round_id,
                    worker_id,
                )
                return False

            # Validate against schema (Section 12.7 rules 1, 3)
            schema_errors = WorkerProtocol.validate_task_result(msg)
            if schema_errors:
                logger.warning("Result validation failed for task %s: %s", task_id, schema_errors)
                self._registry.complete_task(worker_id, success=False)
                return False

            # Validate tensors (Section 12.7 rules 4, 5, 6, 7)
            manifest = msg.get("tensor_manifest", {})
            if manifest:
                tensor_errors = WorkerProtocol.validate_tensors(tensors, manifest)
                if tensor_errors:
                    logger.warning("Tensor validation failed for task %s: %s", task_id, tensor_errors)
                    self._registry.complete_task(worker_id, success=False)
                    return False

            # Anomaly detection (Phase 4) — log warnings but do not reject
            if self._anomaly_detector is not None:
                anomalies = self._anomaly_detector.check_result(
                    worker_id=worker_id,
                    correlation=msg.get("correlation", 0.0),
                    training_duration=msg.get("training_duration", 0.0),
                    task_id=task_id,
                )
                if anomalies:
                    logger.warning("Anomalies detected for worker %s on task %s: %s", worker_id, task_id, anomalies)

            # Accept result
            result = TaskResult(
                task_id=task_id,
                candidate_id=msg["candidate_id"],
                candidate_uuid=msg.get("candidate_uuid", ""),
                correlation=msg["correlation"],
                success=msg["success"],
                epochs_completed=msg["epochs_completed"],
                activation_name=msg.get("activation_name", ""),
                all_correlations=msg.get("all_correlations", []),
                numerator=msg.get("numerator", 0.0),
                denominator=msg.get("denominator", 1.0),
                best_corr_idx=msg.get("best_corr_idx", -1),
                tensors=tensors,
                error_message=msg.get("error_message"),
                round_id=task.round_id,
            )

            self._results[task_id] = result
            self._completed_task_ids.add(task_id)
            task.completed = True
            self._registry.complete_task(worker_id, success=msg["success"])

            logger.info(
                "Accepted result for task %s from worker %s (corr=%.4f, %d/%d complete)",
                task_id,
                worker_id,
                msg["correlation"],
                len(self._results),
                self._current_round_task_count,
            )

            # Signal if all results are in
            if len(self._results) >= self._current_round_task_count:
                self._results_ready.set()

            return True

    def collect_results(self, timeout: float = 120.0) -> list[TaskResult]:
        """Wait for all results from the current round.

        Called by the training thread. Blocks until all results are received,
        the timeout expires, or — ISSUE-319 (defect #3 safety) — no workers remain
        connected to finish the in-flight tasks.

        The last condition bounds the wait now that the collection budget is scaled up
        to the candidate-training workload (see
        ``CascadeCorrelationNetwork._remote_result_collection_timeout``): if every remote
        worker disconnects mid-round the round can never complete, so we stop waiting
        promptly and let the caller fall back to local retry instead of blocking for the
        full (now much larger) budget. The wait still returns the instant all results
        arrive, so a healthy round is unaffected.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            List of TaskResults received (may be fewer than submitted if the wait ended
            early on timeout or loss of all workers).
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                complete = len(self._results) >= self._current_round_task_count
            if complete:
                break  # all results in (or the round was cancelled to count 0)
            # Worker-liveness early-exit: with no registered workers the remaining remote
            # tasks can never be completed or reassigned, so there is nothing left to wait
            # for — return what we have and let the caller fall back to local retry.
            if self._registry.worker_count == 0:
                logger.warning(
                    "collect_results: no remote workers connected — abandoning wait for round %s after %d/%d results",
                    self._current_round_id,
                    len(self._results),
                    self._current_round_task_count,
                )
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Wait in a slice so the loop re-evaluates liveness/completion periodically.
            self._results_ready.wait(timeout=min(_RESULT_COLLECTION_POLL_INTERVAL, remaining))

        with self._lock:
            results = list(self._results.values())
            logger.info(
                "Collected %d/%d results for round %s",
                len(results),
                self._current_round_task_count,
                self._current_round_id,
            )
            return results

    def has_pending_tasks(self) -> bool:
        """Check if there are unassigned tasks waiting for workers."""
        with self._lock:
            return len(self._unassigned_tasks) > 0

    def pending_tasks_count(self) -> int:
        """Return the current number of pending (in-flight) tasks.

        Snapshot read under :attr:`_lock`. Used by the
        :class:`api.workers.metrics.WorkerRegistryCollector` to emit the
        ``juniper_cascor_pending_tasks`` gauge on every Prometheus
        scrape (audit-doc §4.2). Counts every task in
        :attr:`_pending_tasks` regardless of round-id, status, or
        assignment — the dict tracks all tasks the coordinator considers
        in-flight (unassigned + dispatched-but-not-yet-completed).
        """
        with self._lock:
            return len(self._pending_tasks)

    def cancel_round(self) -> None:
        """Cancel the current round and clear all pending tasks."""
        with self._lock:
            self._pending_tasks.clear()
            self._unassigned_tasks.clear()
            self._results.clear()
            self._completed_task_ids.clear()
            self._current_round_id = None
            self._current_round_task_count = 0
            self._results_ready.set()  # Unblock any waiting thread
            logger.info("Current round cancelled")

    def shutdown(self) -> None:
        """Shutdown the coordinator: stop monitor, cancel tasks."""
        self.stop_monitor()
        self.cancel_round()
        with self._lock:
            self._send_callbacks.clear()
        logger.info("WorkerCoordinator shut down")

    def _health_monitor_loop(self) -> None:
        """Background thread that monitors worker health and handles task reassignment."""
        logger.debug("Health monitor loop started")
        while not self._monitor_stop.wait(timeout=self._health_check_interval):
            self._check_stale_workers()
            self._check_task_timeouts()

    def _check_stale_workers(self) -> None:
        """Deregister workers whose heartbeat has timed out.

        To close the TOCTOU gap between get_stale_workers() and deregister(),
        each worker's staleness is re-checked via the registry before removal.
        A worker that sent a heartbeat between the snapshot and the re-check
        will be skipped.

        CONC-10 (Phase 3D): also closes the race window between this monitor
        thread's deregister and `get_next_assignment()` assigning a task to
        the same worker. The pre-fix flow re-checked liveness, popped the
        active task back onto the queue under `self._lock`, then called
        `self._registry.deregister(...)` *outside* the lock — leaving a
        window in which `get_next_assignment()` (which holds `self._lock`
        for its entire critical section) could pick a task and call
        `self._registry.assign_task(worker_id, ...)` after the active-task
        handling but before the deregister. That assignment would land on a
        worker about to disappear, and the task would wait for the next
        120-second `_task_reassignment_timeout`. Holding `self._lock` across
        the re-check, the active-task reassignment, and the deregister makes
        the deregistration atomic with respect to assignment so any
        in-flight `get_next_assignment()` either runs before the worker is
        considered dead (and assigns normally) or after it is removed (and
        sees no eligible task / no registered worker).
        """
        stale = self._registry.get_stale_workers()
        for worker in stale:
            with self._lock:
                # Re-check: the worker may have sent a heartbeat since the
                # snapshot or already been deregistered by another path.
                current = self._registry.get(worker.worker_id)
                if current is None:
                    continue
                if current.is_alive(self._registry._heartbeat_timeout):
                    logger.debug("Worker %s recovered (heartbeat received since stale snapshot) — skipping deregister", worker.worker_id)
                    continue

                logger.warning("Worker %s heartbeat timeout — deregistering", worker.worker_id)
                # If worker had an active task, put it back in the unassigned
                # queue. Read the snapshot's `active_task_id` *and* the
                # current registry entry's so we don't lose a task that was
                # just dispatched (between snapshot and re-check) by an
                # `assign_task()` call serialized under the same lock.
                active_task_id = current.active_task_id or worker.active_task_id
                if active_task_id is not None:
                    task = self._pending_tasks.get(active_task_id)
                    if task is not None and not task.completed:
                        task.assigned_worker_id = None
                        self._unassigned_tasks.append(active_task_id)
                        logger.info("Task %s reassigned to queue (worker %s died)", active_task_id, worker.worker_id)
                # Deregister inside the same critical section so a concurrent
                # `get_next_assignment()` cannot land a task on this worker
                # between the active-task reassignment and the deregister.
                self._registry.deregister(worker.worker_id)
            # Send-callback bookkeeping is independent of the registry lock
            # and may itself take a per-callback lock; keep it outside.
            self.unregister_send_callback(worker.worker_id)

    def _check_task_timeouts(self) -> None:
        """Reassign tasks that have been pending too long."""
        now = time.time()
        with self._lock:
            for task in self._pending_tasks.values():
                if task.assigned_worker_id is not None and not task.completed and (now - task.dispatched_at) > self._task_reassignment_timeout:
                    logger.warning(
                        "Task %s timed out on worker %s (%.1fs) — reassigning",
                        task.task_id,
                        task.assigned_worker_id,
                        now - task.dispatched_at,
                    )
                    self._registry.complete_task(task.assigned_worker_id, success=False)
                    task.assigned_worker_id = None
                    task.dispatched_at = now
                    self._unassigned_tasks.append(task.task_id)
