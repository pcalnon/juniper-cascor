"""WorkerMessageType enum — single source of truth for /ws/v1/workers.

Importing this module does **not** load Pydantic. Worker consumers can
``from juniper_cascor_protocol.worker import WorkerMessageType`` and
keep their existing imperative validators while still single-sourcing
the wire-protocol type strings.
"""

from enum import StrEnum


class WorkerMessageType(StrEnum):
    """All valid wire protocol message types on ``/ws/v1/workers``.

    Mirrors :class:`juniper-cascor/src/api/workers/protocol.MessageType`
    plus the server-emitted handshake / acknowledgement strings the
    cascor server returns. Worker code (and the server itself) should
    import from here rather than redefining the literals.
    """

    # Worker → server
    REGISTER = "register"
    HEARTBEAT = "heartbeat"
    TASK_RESULT = "task_result"

    # Server → worker
    REGISTRATION_ACK = "registration_ack"
    RESULT_ACK = "result_ack"
    TASK_ASSIGN = "task_assign"
    TOKEN_REFRESH = "token_refresh"  # nosec B105 — protocol message type, not a password
    CONNECTION_ESTABLISHED = "connection_established"

    # Either direction
    ERROR = "error"
