"""Security audit logging for the WebSocket worker subsystem.

Records security-relevant events with structured fields for:
- Worker registration/deregistration
- Authentication attempts (success/failure)
- Task assignment and result submission
- Rate limiting events
- Anomaly detections
- TLS handshake events

Events are logged through Python's logging framework at a dedicated
AUDIT level (between INFO and WARNING) for easy filtering.
"""

import json
import logging
import time
from enum import StrEnum
from threading import Lock
from typing import Any

logger = logging.getLogger("juniper_cascor.api.workers.audit")

# Custom log level for audit events (between INFO=20 and WARNING=30)
AUDIT_LEVEL = 25
logging.addLevelName(AUDIT_LEVEL, "AUDIT")


class AuditEventType(StrEnum):
    """Categories of security audit events."""

    WORKER_REGISTER = "worker_register"
    WORKER_DEREGISTER = "worker_deregister"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    TASK_ASSIGNED = "task_assigned"
    RESULT_ACCEPTED = "result_accepted"
    RESULT_REJECTED = "result_rejected"
    RATE_LIMITED = "rate_limited"
    ANOMALY_DETECTED = "anomaly_detected"
    TLS_HANDSHAKE = "tls_handshake"
    CONNECTION_CLOSED = "connection_closed"


class AuditLogger:
    """Structured security event logger for the worker subsystem.

    Usage:
        audit = AuditLogger()
        audit.log(AuditEventType.AUTH_SUCCESS, worker_id="w-123", source_ip="10.0.0.5")
        audit.log(AuditEventType.ANOMALY_DETECTED, worker_id="w-456", details={"type": "perfect_corr"})
    """

    def __init__(self, logger_name: str = "juniper_cascor.api.workers.audit") -> None:
        self._logger = logging.getLogger(logger_name)
        self._counter: dict[str, int] = {}
        self._lock = Lock()

    def log(self, event_type: AuditEventType, **fields: Any) -> None:
        """Log a security audit event.

        Args:
            event_type: Category of the event.
            **fields: Key-value pairs to include in the structured log.
                Common fields: worker_id, source_ip, task_id, details.
        """
        with self._lock:
            self._counter[event_type] = self._counter.get(event_type, 0) + 1
            seq = self._counter[event_type]

        record = {
            "event": event_type.value,
            "timestamp": time.time(),
            "seq": seq,
            **fields,
        }

        # Log at AUDIT level (25)
        self._logger.log(AUDIT_LEVEL, "AUDIT: %s", json.dumps(record, default=str))

    def get_counts(self) -> dict[str, int]:
        """Return event counts by type."""
        with self._lock:
            return dict(self._counter)

    def reset_counts(self) -> None:
        """Reset event counters."""
        with self._lock:
            self._counter.clear()


class WorkerMetrics:
    """Per-worker performance and security metrics.

    Tracks:
    - Task completion rate (success vs failure)
    - Average training duration
    - Connection uptime
    - Anomaly count
    """

    def __init__(self) -> None:
        self._workers: dict[str, _WorkerMetricData] = {}
        self._lock = Lock()

    def on_register(self, worker_id: str, source_ip: str = "") -> None:
        """Record a worker registration."""
        with self._lock:
            self._workers[worker_id] = _WorkerMetricData(
                worker_id=worker_id,
                source_ip=source_ip,
                registered_at=time.time(),
            )

    def on_deregister(self, worker_id: str) -> None:
        """Record a worker deregistration."""
        with self._lock:
            data = self._workers.get(worker_id)
            if data:
                data.deregistered_at = time.time()

    def on_task_complete(self, worker_id: str, success: bool, duration: float) -> None:
        """Record a task completion."""
        with self._lock:
            data = self._workers.get(worker_id)
            if data is None:
                return
            data.tasks_completed += 1
            if success:
                data.tasks_succeeded += 1
            else:
                data.tasks_failed += 1
            data.total_duration += duration

    def on_anomaly(self, worker_id: str, anomaly_type: str) -> None:
        """Record an anomaly detection."""
        with self._lock:
            data = self._workers.get(worker_id)
            if data is None:
                return
            data.anomaly_count += 1
            data.anomaly_types.append(anomaly_type)

    def get_worker_metrics(self, worker_id: str) -> dict[str, Any] | None:
        """Get metrics for a specific worker."""
        with self._lock:
            data = self._workers.get(worker_id)
            if data is None:
                return None
            return {
                "worker_id": data.worker_id,
                "source_ip": data.source_ip,
                "registered_at": data.registered_at,
                "deregistered_at": data.deregistered_at,
                "tasks_completed": data.tasks_completed,
                "tasks_succeeded": data.tasks_succeeded,
                "tasks_failed": data.tasks_failed,
                "avg_duration": data.total_duration / max(1, data.tasks_completed),
                "anomaly_count": data.anomaly_count,
                "success_rate": data.tasks_succeeded / max(1, data.tasks_completed),
            }

    def get_all_metrics(self) -> list[dict[str, Any]]:
        """Get metrics for all workers."""
        with self._lock:
            worker_ids = list(self._workers.keys())
        return [m for wid in worker_ids if (m := self.get_worker_metrics(wid)) is not None]


class _WorkerMetricData:
    """Internal mutable state for per-worker metrics."""

    __slots__ = (
        "worker_id",
        "source_ip",
        "registered_at",
        "deregistered_at",
        "tasks_completed",
        "tasks_succeeded",
        "tasks_failed",
        "total_duration",
        "anomaly_count",
        "anomaly_types",
    )

    def __init__(self, worker_id: str, source_ip: str, registered_at: float) -> None:
        self.worker_id = worker_id
        self.source_ip = source_ip
        self.registered_at = registered_at
        self.deregistered_at: float | None = None
        self.tasks_completed = 0
        self.tasks_succeeded = 0
        self.tasks_failed = 0
        self.total_duration = 0.0
        self.anomaly_count = 0
        self.anomaly_types: list[str] = []
