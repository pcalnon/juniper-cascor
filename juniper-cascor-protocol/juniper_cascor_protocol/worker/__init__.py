"""Worker-protocol surface for ``/ws/v1/workers``.

Importing this subpackage **does not load Pydantic** — the worker keeps
its existing imperative validators and only needs the StrEnum + binary
codec from here. See METRICS-MON R2.2 design Q3 for the rationale.
"""

from juniper_cascor_protocol.worker.binary_frame import BinaryFrame
from juniper_cascor_protocol.worker.messages import WorkerMessageType

__all__ = ["WorkerMessageType", "BinaryFrame"]
