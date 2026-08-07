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
import uuid

from fastapi import WebSocket, WebSocketDisconnect

from api.websocket.manager import ws_authenticate
from api.workers.coordinator import WorkerCoordinator
from api.workers.protocol import BinaryFrame, MessageType, WorkerProtocol
from api.workers.registry import WorkerRegistry, WorkerRegistryFullError

logger = logging.getLogger("juniper_cascor.api.websocket.worker_stream")

_MAX_JSON_SIZE = 65536  # 64KB for JSON messages
_MAX_BINARY_SIZE = 100 * 1024 * 1024  # 100MB for tensor frames
# Mirror WorkerProtocol._MAX_TENSOR_MANIFEST_ENTRIES — keep the receive-site
# guard local so a hostile worker cannot stall on binary frames before the
# coordinator schema check runs.
_MAX_TENSOR_MANIFEST_ENTRIES = 32


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

    if not await ws_authenticate(websocket):
        return

    # Rate limiting check (before accepting the connection)
    rate_limiter = getattr(websocket.app.state, "worker_rate_limiter", None)
    if rate_limiter:
        source_ip = websocket.client[0] if websocket.client else "unknown"
        if not rate_limiter.allow(source_ip):
            await websocket.close(code=4029, reason="Rate limited")
            return

    # Get coordinator and registry from app state
    coordinator: WorkerCoordinator | None = getattr(websocket.app.state, "worker_coordinator", None)
    registry: WorkerRegistry | None = getattr(websocket.app.state, "worker_registry", None)

    if coordinator is None or registry is None:
        await websocket.close(code=4004, reason="Worker system not initialized")
        return

    # SEC-F19 D4: reserve a stack-absolute GLOBAL admission slot before
    # accepting. Per-identity keying is intentionally GLOBAL-only here
    # (identity=None): a worker fleet shares one token, so keying on it would
    # cap horizontal scaling, and the unique server-assigned worker_id is only
    # known post-registration — meaningful worker per-identity keying is not
    # cleanly available, so the global cap is the worker minimum (design §8
    # OQ-2; a documented follow-up). On over-cap try_admit closes with 1013.
    ws_manager = getattr(websocket.app.state, "ws_manager", None)
    if ws_manager is not None and not await ws_manager.try_admit(websocket, endpoint="workers", identity=None):
        return

    try:
        await _run_worker_session(websocket, coordinator, registry, ws_manager)
    finally:
        # SEC-F19 D4: release the admission slot on every disconnect path
        # (including an exception in accept/session), exactly once per admit.
        if ws_manager is not None:
            await ws_manager.release_admission(identity=None)


async def _run_worker_session(
    websocket: WebSocket,
    coordinator: WorkerCoordinator,
    registry: WorkerRegistry,
    ws_manager,
) -> None:
    """Accept an already-admitted /ws/v1/workers connection and run its session.

    Split out of :func:`worker_stream_handler` so the SEC-F19 D4 admission
    reserve/release wraps the whole session (including ``accept``) in a single
    outer try/finally, while keeping each function within the flake8 C901
    complexity budget.
    """
    await websocket.accept()
    await websocket.send_json(
        {
            "type": "connection_established",
            "data": {"channel": "workers"},
        }
    )

    # OBS-WIRE-02 (Q3): per-endpoint ``connections_active{endpoint="workers"}``
    # bookkeeping. Register *after* successful accept so the gauge
    # reflects broadcast-eligible connections; the matching unregister
    # in ``finally`` re-emits on every disconnect path including
    # exceptions.
    if ws_manager is not None:
        ws_manager.register_endpoint_connection(websocket, "workers")

    worker_id: str | None = None
    audit_logger = getattr(websocket.app.state, "audit_logger", None)
    worker_metrics = getattr(websocket.app.state, "worker_metrics", None)

    try:
        # Step 1: Wait for registration message
        worker_id = await _handle_registration(websocket, registry)
        if worker_id is None:
            return

        # Register send callback for this connection
        coordinator.register_send_callback(worker_id, _make_send_callback(websocket))

        # Audit and metrics on successful registration
        if audit_logger:
            from api.workers.audit import AuditEventType

            audit_logger.log(AuditEventType.WORKER_REGISTER, worker_id=worker_id)
        if worker_metrics:
            source_ip = websocket.client[0] if websocket.client else ""
            worker_metrics.on_register(worker_id, source_ip)

        logger.info("Worker %s connected and registered", worker_id)

        # Step 2: Enter message loop
        await _message_loop(websocket, worker_id, registry, coordinator)

    except WebSocketDisconnect:
        logger.info("Worker %s disconnected", worker_id or "unknown")
    except Exception:
        logger.exception("Unexpected error in worker stream for %s", worker_id or "unknown")
    finally:
        # OBS-WIRE-02 (Q3): always re-emit the gauge on disconnect.
        if ws_manager is not None:
            ws_manager.unregister_endpoint_connection(websocket)
        # Cleanup on disconnect
        if worker_id is not None:
            coordinator.unregister_send_callback(worker_id)
            registry.deregister(worker_id)
            if audit_logger:
                from api.workers.audit import AuditEventType

                audit_logger.log(AuditEventType.WORKER_DEREGISTER, worker_id=worker_id)
            if worker_metrics:
                worker_metrics.on_deregister(worker_id)
            # Drop anomaly history for this worker so (a) the per-worker
            # history dict cannot grow without bound across churn and
            # (b) a recycled worker_id cannot inherit stale
            # duplicate_correlations / perfect_correlation signals from a
            # prior occupant. clear_worker is idempotent.
            anomaly_detector = getattr(websocket.app.state, "anomaly_detector", None)
            if anomaly_detector is not None:
                anomaly_detector.clear_worker(worker_id)
            logger.info("Worker %s cleaned up", worker_id)


async def _handle_registration(websocket: WebSocket, registry: WorkerRegistry) -> str | None:
    """Wait for and process the worker registration message.

    The worker proposes a ``worker_id`` in its REGISTER payload, but the
    server does NOT trust it (CR-026). That field is treated as an
    untrusted client-supplied display name and captured as ``client_name``
    on the registration for audit logging only. The server generates a
    fresh UUID as the authoritative ``worker_id`` and returns it to the
    worker in the ``registration_ack`` payload. Workers must use the
    server-assigned ID for any out-of-band references (audit logs,
    metrics, dashboards).

    Returns:
        The server-assigned worker_id if registration succeeded, None otherwise.
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

    # JSON null / arrays / scalars parse successfully but are not objects;
    # calling ``.get`` would raise AttributeError and tear down the session.
    if not isinstance(msg, dict):
        await websocket.send_json(WorkerProtocol.build_error("Registration message must be a JSON object"))
        await websocket.close(code=4008, reason="Invalid registration")
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

    # CR-026: Client-supplied worker_id is treated as an untrusted display name
    # (client_name) and discarded as an identity. Server generates an
    # authoritative UUID.
    client_name = msg["worker_id"]
    capabilities = msg["capabilities"]
    worker_id = f"worker-{uuid.uuid4().hex[:12]}"

    try:
        registry.register(worker_id, capabilities, client_name=client_name)
    except WorkerRegistryFullError as exc:
        # Audit-doc E.6: the registry has hit its capacity cap. Reject
        # the handshake cleanly with a structured error frame and a
        # dedicated close code so operators (and future tests) can
        # distinguish "saturation" from generic 4008 "invalid
        # registration" failures.
        logger.warning(
            "Worker handshake rejected — registry at capacity (client_name=%s): %s",
            client_name,
            exc,
        )
        await websocket.send_json(
            WorkerProtocol.build_error(
                "Worker registry at capacity",
                details=str(exc),
            )
        )
        await websocket.close(code=4013, reason="Worker registry at capacity")
        return None

    logger.info("Worker registered with server ID %s (client_name=%s)", worker_id, client_name)

    await websocket.send_json(
        {
            "type": "registration_ack",
            "worker_id": worker_id,
            "data": {"status": "registered", "client_name": client_name},
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

            # Non-object JSON (null/array/scalar) must not reach ``msg.get`` —
            # that AttributeError aborts the whole worker session.
            if not isinstance(msg, dict):
                await websocket.send_json(WorkerProtocol.build_error("Message must be a JSON object"))
                continue

            msg_type = msg.get("type")

            if msg_type == MessageType.HEARTBEAT:
                # METRICS-MON R1.3 / seed-04: forward enriched heartbeat
                # fields when present. Workers running older images send
                # only ``worker_id`` + ``timestamp``; the missing kwargs
                # stay at None and prior values are preserved.
                # METRICS-MON R4.4: three additional optional fields from
                # R4.4-aware workers (training-loop instrumentation).
                # ``msg.get(...)`` returns None for absent keys — same
                # additive-compatibility shape as R1.3.
                registry.heartbeat(
                    worker_id,
                    in_flight_tasks=msg.get("in_flight_tasks"),
                    last_task_completed_at=msg.get("last_task_completed_at"),
                    rss_mb=msg.get("rss_mb"),
                    tasks_completed=msg.get("tasks_completed"),
                    tasks_failed=msg.get("tasks_failed"),
                    last_task_duration_seconds=msg.get("last_task_duration_seconds"),
                    recent_task_durations_seconds=msg.get("recent_task_durations_seconds"),
                    gpu_utilization_pct=msg.get("gpu_utilization_pct"),
                )
                await websocket.send_json(WorkerProtocol.build_heartbeat(worker_id))

                # ISSUE-319 (defect #5 — the dual-path unlock): deliver tasks that were
                # submitted AFTER this worker connected. _try_dispatch_task otherwise runs
                # only at connect (above) and after a task result (below), so an idle,
                # already-connected worker has no trigger to pick up candidate tasks
                # submitted mid-session. Those tasks then sit unassigned until the remote
                # collection budget expires and the round falls back to local retry — so the
                # remote tier never contributes and the cascade stalls. A heartbeat arrives
                # well within the registry heartbeat timeout, bounding dispatch latency to
                # roughly one heartbeat interval. Guard on idle state: a heartbeat can arrive
                # mid-task, and get_next_assignment does not itself refuse a busy worker
                # (registry.assign_task would silently reject, but the task would already be
                # popped and sent), so only pull work for a genuinely idle worker.
                reg = registry.get(worker_id)
                if reg is not None and reg.idle:
                    await _try_dispatch_task(websocket, worker_id, coordinator)

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
    # ``.get(..., {})`` only substitutes when the key is absent — a present
    # null/string/list would otherwise be iterated (TypeError / char-by-char
    # receive loop) before any schema validation runs.
    if "tensor_manifest" in msg and not isinstance(msg["tensor_manifest"], dict):
        logger.error(
            "Invalid tensor_manifest type from worker %s: %s",
            worker_id,
            type(msg["tensor_manifest"]).__name__,
        )
        await websocket.send_json(WorkerProtocol.build_error(f"tensor_manifest must be a JSON object, got {type(msg['tensor_manifest']).__name__}"))
        return

    manifest = msg.get("tensor_manifest", {})
    if len(manifest) > _MAX_TENSOR_MANIFEST_ENTRIES:
        logger.error(
            "tensor_manifest too large from worker %s: %d entries",
            worker_id,
            len(manifest),
        )
        await websocket.send_json(WorkerProtocol.build_error(f"tensor_manifest has too many entries: {len(manifest)} > {_MAX_TENSOR_MANIFEST_ENTRIES}"))
        return

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
