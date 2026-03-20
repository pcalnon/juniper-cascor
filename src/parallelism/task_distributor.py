"""Unified task distribution across local multiprocessing and remote WebSocket workers.

The TaskDistributor implements a local-first scheduling policy: local workers
always fill first, remote workers handle overflow. When remote tasks fail or
timeout, they are retried on the local pool.

This module is the Phase 3 component of the CasCor Concurrency Architecture.
"""

import logging
import time
from typing import Any, Callable

logger = logging.getLogger("juniper_cascor.parallelism.task_distributor")


class TaskDistributor:
    """Splits and executes candidate training tasks across local and remote workers.

    Scheduling policy (local-first):
        1. Tasks up to local_capacity go to the local MP pool
        2. Overflow tasks go to remote WS workers (if available)
        3. If no remote workers are available, all tasks go to local
        4. Failed remote tasks are retried on the local pool

    Usage:
        distributor = TaskDistributor(logger=network_logger)
        distributor.set_coordinator(coordinator)
        results = distributor.distribute_and_collect(
            tasks=tasks,
            local_capacity=process_count,
            local_fn=execute_parallel_training,
            remote_fn=dispatch_to_remote_workers,
        )
    """

    def __init__(self, dist_logger: logging.Logger | None = None) -> None:
        self._logger = dist_logger or logger
        self._coordinator: Any = None

    def set_coordinator(self, coordinator: Any) -> None:
        """Set the remote worker coordinator for dual-path dispatch."""
        self._coordinator = coordinator
        self._logger.info("TaskDistributor: Remote worker coordinator set")

    @property
    def remote_worker_count(self) -> int:
        """Number of idle remote workers currently available."""
        if self._coordinator is not None:
            return self._coordinator._registry.available_worker_count
        return 0

    def distribute_and_collect(
        self,
        tasks: list,
        local_capacity: int,
        local_fn: Callable[[list, int], list],
        remote_fn: Callable[[list], list],
        remote_retry_fn: Callable[[list, int], list] | None = None,
        timeout: float = 120.0,
    ) -> list:
        """Distribute tasks across tiers and collect unified results.

        Args:
            tasks: Training task tuples from _generate_candidate_tasks.
            local_capacity: Number of local MP workers available.
            local_fn: Callable(tasks, process_count) -> list of results.
                Executes tasks on the local multiprocessing pool.
            remote_fn: Callable(tasks) -> list of results.
                Dispatches tasks to remote WebSocket workers.
            remote_retry_fn: Optional callable(tasks, process_count) -> list of results.
                Used to retry failed remote tasks locally. Defaults to local_fn.
            timeout: Maximum time to wait for remote results.

        Returns:
            Unified list of results from both tiers.
        """
        if remote_retry_fn is None:
            remote_retry_fn = local_fn

        remote_count = self.remote_worker_count
        local_tasks, remote_tasks = self._split_tasks(tasks, local_capacity, remote_count)

        self._logger.info(
            "TaskDistributor: Distributing %d tasks — %d local, %d remote (local_capacity=%d, remote_workers=%d)",
            len(tasks),
            len(local_tasks),
            len(remote_tasks),
            local_capacity,
            remote_count,
        )

        # Execute based on split
        if not remote_tasks:
            # All local (or sequential if local_capacity <= 1)
            return local_fn(local_tasks, local_capacity)

        if not local_tasks:
            # All remote
            remote_results = self._execute_remote_with_fallback(
                remote_tasks, remote_fn, remote_retry_fn, local_capacity, timeout
            )
            return remote_results

        # Dual-path: execute both tiers
        local_results = local_fn(local_tasks, local_capacity)
        remote_results = self._execute_remote_with_fallback(
            remote_tasks, remote_fn, remote_retry_fn, local_capacity, timeout
        )
        return local_results + remote_results

    def _split_tasks(self, tasks: list, local_capacity: int, remote_count: int) -> tuple[list, list]:
        """Split tasks with local-first priority.

        Local workers always fill first. Only overflow goes to remote.
        If no remote workers, everything is local.

        Args:
            tasks: All tasks to distribute.
            local_capacity: Number of local workers.
            remote_count: Number of available remote workers.

        Returns:
            (local_tasks, remote_tasks) tuple.
        """
        if remote_count <= 0 or local_capacity <= 0:
            return tasks, []

        # Local-first: local gets up to local_capacity tasks
        # Remote gets the remainder (if any remote workers available)
        if len(tasks) <= local_capacity:
            return tasks, []

        local_tasks = tasks[:local_capacity]
        remote_tasks = tasks[local_capacity:]

        # Don't send more tasks to remote than they can handle
        if len(remote_tasks) > remote_count:
            # Move excess back to local
            excess = remote_tasks[remote_count:]
            remote_tasks = remote_tasks[:remote_count]
            local_tasks = local_tasks + excess

        return local_tasks, remote_tasks

    def _execute_remote_with_fallback(
        self,
        tasks: list,
        remote_fn: Callable[[list], list],
        retry_fn: Callable[[list, int], list],
        local_capacity: int,
        timeout: float,
    ) -> list:
        """Execute tasks on remote workers with local fallback on failure.

        If remote execution fails or returns incomplete results, the missing
        tasks are retried on the local pool.

        Args:
            tasks: Tasks to execute remotely.
            remote_fn: Remote execution callable.
            retry_fn: Local retry callable for failed tasks.
            local_capacity: Local worker count for retries.
            timeout: Timeout for remote execution.

        Returns:
            List of results (may include results from both remote and local retry).
        """
        start = time.time()
        try:
            remote_results = remote_fn(tasks)
        except Exception as e:
            self._logger.warning(
                "TaskDistributor: Remote execution failed (%s) — retrying %d tasks locally",
                e,
                len(tasks),
            )
            return retry_fn(tasks, local_capacity)

        elapsed = time.time() - start

        # Check for incomplete results
        if len(remote_results) < len(tasks):
            completed_indices = set()
            for r in remote_results:
                cid = getattr(r, "candidate_id", None)
                if cid is not None:
                    completed_indices.add(cid)

            failed_tasks = [t for t in tasks if t[0] not in completed_indices]

            if failed_tasks:
                self._logger.warning(
                    "TaskDistributor: Remote returned %d/%d results (%.1fs) — retrying %d failed tasks locally",
                    len(remote_results),
                    len(tasks),
                    elapsed,
                    len(failed_tasks),
                )
                retry_results = retry_fn(failed_tasks, local_capacity)
                return remote_results + retry_results

        # Check for failed results and retry those
        successful = []
        failed_tasks = []
        for r in remote_results:
            success = getattr(r, "success", True)
            if success:
                successful.append(r)
            else:
                cid = getattr(r, "candidate_id", -1)
                matching = [t for t in tasks if t[0] == cid]
                if matching:
                    failed_tasks.extend(matching)
                else:
                    successful.append(r)

        if failed_tasks:
            self._logger.warning(
                "TaskDistributor: %d remote tasks returned success=False — retrying locally",
                len(failed_tasks),
            )
            retry_results = retry_fn(failed_tasks, local_capacity)
            return successful + retry_results

        self._logger.info(
            "TaskDistributor: All %d remote tasks completed successfully (%.1fs)",
            len(remote_results),
            elapsed,
        )
        return remote_results
