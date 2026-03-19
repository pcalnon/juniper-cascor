"""Worker infrastructure for WebSocket-based distributed candidate training."""

from api.workers.coordinator import WorkerCoordinator
from api.workers.protocol import BinaryFrame, MessageType, WorkerProtocol
from api.workers.registry import WorkerRegistration, WorkerRegistry

__all__ = [
    "BinaryFrame",
    "MessageType",
    "WorkerCoordinator",
    "WorkerProtocol",
    "WorkerRegistration",
    "WorkerRegistry",
]
