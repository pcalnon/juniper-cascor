"""WebSocket handler for /ws/v1/workers — remote worker communication channel.

Handles bidirectional communication with remote candidate training workers:
- Registration and heartbeat keepalive
- Task assignment (JSON envelope + binary tensor frames)
- Result collection (JSON envelope + binary tensor frames)
- Origin header rejection (machine-to-machine only)

Authentication follows the same pattern as control_stream.py: API key
validation via X-API-Key header on the WebSocket upgrade request.

Wire protocol details: see api.workers.protocol module.
"""

import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from api.workers.coordinator import WorkerCoordinator
from api.workers.protocol import BinaryFrame, MessageType, WorkerProtocol
from api.workers.registry import WorkerRegistry

logger = logging.getLogger("juniper_cascor.api.websocket.worker_stream")

_MAX_JSON_SIZE = 65536  # 64KB for JSON messages
_MAX_BINARY_SIZE = 100 * 1024 * 1024  # 100MB for tensor frames


async def worker_stream_handler(websocket: WebSocket) -> None:
    """Handle /ws/v1/workers WebSocket connections.

    Protocol flow:
    1. Authenticate via X-API-Key header
    2. Reject connections with Origin header (browser protection)
    3. Accept connection
    4. Receive registration message from worker
    5. Enter message loop: heartbeats, task dispatch, result collection
    """
    # Reject connections with Origin header (Section 12.3 — workers are machine-to-machine)
    if websocket.headers.get("origin"):
        await websocket.close(code=4003, reason="Origin header not allowed on worker endpoint")
        return

    # Authenticate (same pattern as control_stream.py)
    auth = getattr(websocket.app.state, "api_key_auth", None)
    if auth is not None and auth.enabled:
        api_key = websocket.headers.get("X-API-Key")
        if not auth.validate(api_key):
            await websocket.close(code=4001, reason="Authentication required")
            return

    # Get coordinator and registry from app state
    coordinator: WorkerCoordinator | None = getattr(websocket.app.state, "worker_coordinator", None)
    registry: WorkerRegistry | None = getattr(websocket.app.state, "worker_registry", None)

    if coordinator is None or registry is None:
        await websocket.close(code=4004, reason="Worker system not initialized")
        return

    await websocket.accept()
    await websocket.send_json(
        {
            "type": "connection_established",
            "data": {"channel": "workers"},
        }
    )

    worker_id: str | None = None

    try:
        # Step 1: Wait for registration message
        worker_id = await _handle_registration(websocket, registry)
        if worker_id is None:
            return

        # Register send callback for this connection
        coordinator.register_send_callback(worker_id, _make_send_callback(websocket))

        logger.info("Worker %s connected and registered", worker_id)

        # Step 2: Enter message loop
        await _message_loop(websocket, worker_id, registry, coordinator)

    except WebSocketDisconnect:
        logger.info("Worker %s disconnected", worker_id or "unknown")
    except Exception:
        logger.exception("Unexpected error in worker stream for %s", worker_id or "unknown")
    finally:
        # Cleanup on disconnect
        if worker_id is not None:
            coordinator.unregister_send_callback(worker_id)
            registry.deregister(worker_id)
            logger.info("Worker %s cleaned up", worker_id)


async def _handle_registration(websocket: WebSocket, registry: WorkerRegistry) -> str | None:
    """Wait for and process the worker registration message.

    Returns:
        The worker_id if registration succeeded, None otherwise.
    """
    raw = await websocket.receive_text()

    if len(raw) > _MAX_JSON_SIZE:
        await websocket.send_json(WorkerProtocol.build_error("Registration message too large"))
        await websocket.close(code=4005, reason="Message too large")
        return None

    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.send_json(WorkerProtocol.build_error("Invalid JSON"))
        await websocket.close(code=4006, reason="Invalid JSON")
        return None

    if msg.get("type") != MessageType.REGISTER:
        await websocket.send_json(WorkerProtocol.build_error("First message must be registration"))
        await websocket.close(code=4007, reason="Expected registration")
        return None

    errors = WorkerProtocol.validate_register(msg)
    if errors:
        await websocket.send_json(WorkerProtocol.build_error("Invalid registration", details="; ".join(errors)))
        await websocket.close(code=4008, reason="Invalid registration")
        return None

    worker_id = msg["worker_id"]
    capabilities = msg["capabilities"]

    registry.register(worker_id, capabilities)

    await websocket.send_json(
        {
            "type": "registration_ack",
            "worker_id": worker_id,
            "data": {"status": "registered"},
        }
    )

    return worker_id


async def _message_loop(
    websocket: WebSocket,
    worker_id: str,
    registry: WorkerRegistry,
    coordinator: WorkerCoordinator,
) -> None:
    """Main message processing loop for a connected worker.

    Handles heartbeats, task results, and proactively dispatches tasks.
    """
    # Check if there are tasks to dispatch immediately
    await _try_dispatch_task(websocket, worker_id, coordinator)

    while True:
        # Receive next message (text or binary)
        message = await websocket.receive()

        if "text" in message:
            raw = message["text"]
            if len(raw) > _MAX_JSON_SIZE:
                await websocket.send_json(WorkerProtocol.build_error("Message too large"))
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(WorkerProtocol.build_error("Invalid JSON"))
                continue

            msg_type = msg.get("type")

            if msg_type == MessageType.HEARTBEAT:
                registry.heartbeat(worker_id)
                await websocket.send_json(WorkerProtocol.build_heartbeat(worker_id))

            elif msg_type == MessageType.TASK_RESULT:
                await _handle_task_result(websocket, worker_id, msg, coordinator)
                # After completing a task, try to dispatch the next one
                await _try_dispatch_task(websocket, worker_id, coordinator)

            else:
                await websocket.send_json(WorkerProtocol.build_error(f"Unknown message type: {msg_type}"))

        elif "bytes" in message:
            # Binary frames are only expected as part of a task_result sequence.
            # They are collected by _handle_task_result, not here.
            # Stray binary frames are ignored with a warning.
            logger.warning("Unexpected binary frame from worker %s (outside result sequence)", worker_id)


async def _handle_task_result(
    websocket: WebSocket,
    worker_id: str,
    msg: dict,
    coordinator: WorkerCoordinator,
) -> None:
    """Handle a task_result message and its associated binary tensor frames."""
    manifest = msg.get("tensor_manifest", {})
    tensors: dict = {}

    # Receive binary frames for each tensor in manifest order
    for tensor_name in manifest:
        frame_msg = await websocket.receive()
        if "bytes" not in frame_msg:
            logger.error("Expected binary frame for tensor %s, got text from worker %s", tensor_name, worker_id)
            await websocket.send_json(WorkerProtocol.build_error(f"Expected binary frame for tensor: {tensor_name}"))
            return

        raw_bytes = frame_msg["bytes"]
        if len(raw_bytes) > _MAX_BINARY_SIZE:
            logger.error("Binary frame for %s exceeds size limit from worker %s", tensor_name, worker_id)
            await websocket.send_json(WorkerProtocol.build_error("Binary frame too large"))
            return

        try:
            tensors[tensor_name] = BinaryFrame.decode(raw_bytes)
        except ValueError as e:
            logger.error("Failed to decode binary frame for %s from worker %s: %s", tensor_name, worker_id, e)
            await websocket.send_json(WorkerProtocol.build_error(f"Invalid binary frame for {tensor_name}: {e}"))
            return

    # Submit the result to the coordinator
    accepted = coordinator.submit_result(worker_id, msg, tensors)

    if accepted:
        await websocket.send_json(
            {
                "type": "result_ack",
                "task_id": msg.get("task_id"),
                "status": "accepted",
            }
        )
    else:
        await websocket.send_json(
            {
                "type": "result_ack",
                "task_id": msg.get("task_id"),
                "status": "rejected",
            }
        )


async def _try_dispatch_task(
    websocket: WebSocket,
    worker_id: str,
    coordinator: WorkerCoordinator,
) -> None:
    """Try to dispatch a pending task to this worker."""
    assignment = coordinator.get_next_assignment(worker_id)
    if assignment is None:
        return

    msg, frames = assignment

    # Send JSON envelope
    await websocket.send_json(msg)

    # Send binary tensor frames
    for frame in frames:
        await websocket.send_bytes(frame)

    logger.debug("Dispatched task %s to worker %s", msg.get("task_id"), worker_id)


def _make_send_callback(websocket: WebSocket):
    """Create an async send callback for dispatching tasks to a specific worker."""

    async def callback(msg: dict, frames: list[bytes] | None = None) -> bool:
        try:
            await websocket.send_json(msg)
            if frames:
                for frame in frames:
                    await websocket.send_bytes(frame)
            return True
        except Exception:
            return False

    return callback
