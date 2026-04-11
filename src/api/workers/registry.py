"""Thread-safe registry for connected WebSocket workers.

Tracks worker connections, capabilities, heartbeats, and health scores.
Enforces one active connection per worker_id (new connections replace old).
"""

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

logger = logging.getLogger("juniper_cascor.api.workers.registry")


@dataclass
class WorkerRegistration:
    """Tracks a single connected worker.

    The ``worker_id`` field is the server-assigned authoritative identity
    (see ``WorkerRegistry.register``). The optional ``client_name`` captures
    the worker's self-proposed display name for audit logging and operator
    debugging — it is never used as an identity and two workers may report
    the same ``client_name`` without collision because their ``worker_id``
    values are independently generated.
    """

    worker_id: str
    capabilities: dict[str, Any] = field(default_factory=dict)
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    tasks_completed: int = 0
    tasks_failed: int = 0
    active_task_id: str | None = None
    client_name: str | None = None

    @property
    def health_score(self) -> float:
        """Compute health score [0.0, 1.0] based on task success rate.

        Workers with no completed tasks get a neutral score of 1.0.
        """
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 1.0
        return self.tasks_completed / total

    @property
    def idle(self) -> bool:
        """Whether the worker is idle (not assigned a task)."""
        return self.active_task_id is None

    def record_heartbeat(self) -> None:
        """Update last heartbeat timestamp."""
        self.last_heartbeat = time.time()

    def is_alive(self, timeout: float) -> bool:
        """Check if the worker is alive based on heartbeat timeout."""
        return (time.time() - self.last_heartbeat) < timeout


class WorkerRegistry:
    """Thread-safe registry of connected workers.

    Provides methods for registration, deregistration, heartbeat tracking,
    and querying available workers. All public methods are thread-safe.
    """

    def __init__(self, heartbeat_timeout: float = 30.0) -> None:
        self._workers: dict[str, WorkerRegistration] = {}
        self._lock = Lock()
        self._heartbeat_timeout = heartbeat_timeout
        logger.info("WorkerRegistry initialized (heartbeat_timeout=%.1fs)", heartbeat_timeout)

    @property
    def worker_count(self) -> int:
        """Number of currently registered workers."""
        with self._lock:
            return len(self._workers)

    @property
    def available_worker_count(self) -> int:
        """Number of idle, alive workers available for task assignment."""
        with self._lock:
            return sum(1 for w in self._workers.values() if w.idle and w.is_alive(self._heartbeat_timeout))

    def register(
        self,
        worker_id: str,
        capabilities: dict[str, Any],
        client_name: str | None = None,
    ) -> WorkerRegistration:
        """Register a worker. Replaces any existing registration for the same ID.

        The ``worker_id`` must be a server-assigned authoritative identity
        (see ``api.websocket.worker_stream._handle_registration``), not a
        client-supplied value. Callers are responsible for generating the
        server-side identity before calling this method.

        Args:
            worker_id: Server-assigned unique worker identifier (e.g. a UUID).
            capabilities: Worker capability metadata.
            client_name: Optional client-proposed display name for audit
                logging. Never used as identity.

        Returns:
            The new WorkerRegistration.
        """
        with self._lock:
            if worker_id in self._workers:
                logger.warning("Worker %s re-registering (replacing existing connection)", worker_id)
            reg = WorkerRegistration(worker_id=worker_id, capabilities=capabilities, client_name=client_name)
            self._workers[worker_id] = reg
            logger.info(
                "Worker registered: %s (client_name=%s, total=%d)",
                worker_id,
                client_name or "<none>",
                len(self._workers),
            )
            return reg

    def deregister(self, worker_id: str) -> WorkerRegistration | None:
        """Remove a worker from the registry.

        Args:
            worker_id: Worker to remove.

        Returns:
            The removed registration, or None if not found.
        """
        with self._lock:
            reg = self._workers.pop(worker_id, None)
            if reg:
                logger.info("Worker deregistered: %s (total: %d)", worker_id, len(self._workers))
            return reg

    def get(self, worker_id: str) -> WorkerRegistration | None:
        """Get a worker registration by ID."""
        with self._lock:
            return self._workers.get(worker_id)

    def heartbeat(self, worker_id: str) -> bool:
        """Record a heartbeat for a worker.

        Returns:
            True if the worker exists, False otherwise.
        """
        with self._lock:
            reg = self._workers.get(worker_id)
            if reg is None:
                return False
            reg.record_heartbeat()
            return True

    def assign_task(self, worker_id: str, task_id: str) -> bool:
        """Assign a task to a worker.

        Returns:
            True if successful, False if worker not found or already busy.
        """
        with self._lock:
            reg = self._workers.get(worker_id)
            if reg is None or not reg.idle:
                return False
            reg.active_task_id = task_id
            return True

    def complete_task(self, worker_id: str, success: bool) -> bool:
        """Mark a worker's active task as complete.

        Returns:
            True if successful, False if worker not found or had no active task.
        """
        with self._lock:
            reg = self._workers.get(worker_id)
            if reg is None or reg.active_task_id is None:
                return False
            reg.active_task_id = None
            if success:
                reg.tasks_completed += 1
            else:
                reg.tasks_failed += 1
            return True

    def get_idle_workers(self) -> list[WorkerRegistration]:
        """Get all idle, alive workers sorted by health score (best first)."""
        with self._lock:
            idle = [w for w in self._workers.values() if w.idle and w.is_alive(self._heartbeat_timeout)]
            return sorted(idle, key=lambda w: w.health_score, reverse=True)

    def get_stale_workers(self) -> list[WorkerRegistration]:
        """Get workers whose heartbeat has timed out."""
        with self._lock:
            return [w for w in self._workers.values() if not w.is_alive(self._heartbeat_timeout)]

    def get_all_workers(self) -> list[WorkerRegistration]:
        """Get a snapshot of all registered workers."""
        with self._lock:
            return list(self._workers.values())

    def clear(self) -> int:
        """Remove all workers. Returns the number removed."""
        with self._lock:
            count = len(self._workers)
            self._workers.clear()
            logger.info("Registry cleared (%d workers removed)", count)
            return count
