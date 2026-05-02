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

    METRICS-MON R1.3 / seed-04: ``in_flight_tasks``, ``last_task_completed_at``,
    and ``rss_mb`` are populated by enriched heartbeat payloads from
    workers that report them. Older workers send only ``worker_id`` /
    ``timestamp`` and these fields stay at their defaults (0, None,
    None) until the worker upgrades.
    """

    worker_id: str
    capabilities: dict[str, Any] = field(default_factory=dict)
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    tasks_completed: int = 0
    tasks_failed: int = 0
    active_task_id: str | None = None
    client_name: str | None = None
    # METRICS-MON R1.3 / seed-04: enriched heartbeat fields. Populated by
    # workers that send the R1.3 heartbeat shape; left at defaults for
    # workers running older images.
    in_flight_tasks: int = 0
    last_task_completed_at: float | None = None
    rss_mb: float | None = None
    # METRICS-MON R4.4: training-loop instrumentation fields populated by
    # workers running images >= R4.4. Defaults match the R1.3 ``None`` /
    # ``[]`` pattern so older workers leave the registration's prior
    # values untouched. Field semantics:
    # * ``last_task_duration_seconds`` — wall-clock duration of the
    #   most recently completed task (any outcome).
    # * ``recent_task_durations_seconds`` — sliding window (oldest →
    #   newest) of the last N task durations. Server-side percentile
    #   estimators (p50, p99) compute over the union of all workers'
    #   windows on each scrape.
    # * ``gpu_utilization_pct`` — best-effort 0–100 reading; ``None``
    #   when no CUDA / NVML / torch.
    last_task_duration_seconds: float | None = None
    recent_task_durations_seconds: list[float] = field(default_factory=list)
    gpu_utilization_pct: float | None = None

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

    def record_heartbeat(
        self,
        *,
        in_flight_tasks: int | None = None,
        last_task_completed_at: float | None = None,
        rss_mb: float | None = None,
        tasks_completed: int | None = None,
        tasks_failed: int | None = None,
        last_task_duration_seconds: float | None = None,
        recent_task_durations_seconds: list[float] | None = None,
        gpu_utilization_pct: float | None = None,
    ) -> None:
        """Update last heartbeat timestamp and (optionally) enriched fields.

        METRICS-MON R1.3 / seed-04: keyword-only enriched fields are
        accepted from R1.3-aware workers. Older workers omit them; the
        registration's prior values are preserved.

        METRICS-MON R4.4: three additional optional fields accepted from
        R4.4-aware workers (``last_task_duration_seconds``,
        ``recent_task_durations_seconds``, ``gpu_utilization_pct``).
        Same ``None``-default-preserves-prior-value pattern as R1.3.
        """
        self.last_heartbeat = time.time()
        if in_flight_tasks is not None:
            self.in_flight_tasks = in_flight_tasks
        if last_task_completed_at is not None:
            self.last_task_completed_at = last_task_completed_at
        if rss_mb is not None:
            self.rss_mb = rss_mb
        if tasks_completed is not None:
            self.tasks_completed = tasks_completed
        if tasks_failed is not None:
            self.tasks_failed = tasks_failed
        if last_task_duration_seconds is not None:
            self.last_task_duration_seconds = last_task_duration_seconds
        if recent_task_durations_seconds is not None:
            # Defensive copy: callers shouldn't be able to mutate the
            # registration's window through a shared reference.
            self.recent_task_durations_seconds = list(recent_task_durations_seconds)
        if gpu_utilization_pct is not None:
            self.gpu_utilization_pct = gpu_utilization_pct

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

    def heartbeat(
        self,
        worker_id: str,
        *,
        in_flight_tasks: int | None = None,
        last_task_completed_at: float | None = None,
        rss_mb: float | None = None,
        tasks_completed: int | None = None,
        tasks_failed: int | None = None,
        last_task_duration_seconds: float | None = None,
        recent_task_durations_seconds: list[float] | None = None,
        gpu_utilization_pct: float | None = None,
    ) -> bool:
        """Record a heartbeat for a worker.

        METRICS-MON R1.3 / seed-04: keyword-only enriched fields are
        forwarded to ``WorkerRegistration.record_heartbeat``. Pass
        ``None`` (the default) for any field not reported by the worker.

        METRICS-MON R4.4: three additional R4.4-only fields forwarded to
        ``record_heartbeat`` (training-loop instrumentation). Same
        ``None``-default semantics: workers running pre-R4.4 images
        don't send them; prior values are preserved.

        Returns:
            True if the worker exists, False otherwise.
        """
        with self._lock:
            reg = self._workers.get(worker_id)
            if reg is None:
                return False
            reg.record_heartbeat(
                in_flight_tasks=in_flight_tasks,
                last_task_completed_at=last_task_completed_at,
                rss_mb=rss_mb,
                tasks_completed=tasks_completed,
                tasks_failed=tasks_failed,
                last_task_duration_seconds=last_task_duration_seconds,
                recent_task_durations_seconds=recent_task_durations_seconds,
                gpu_utilization_pct=gpu_utilization_pct,
            )
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
