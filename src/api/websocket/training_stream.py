"""WebSocket handler for /ws/training — real-time training metrics stream.

Server-to-client streaming endpoint with optional resume support. On connect:
1. connection_established message (from manager.connect_pending)
2. Optional resume handshake: client sends ``resume`` frame within timeout
3. If resume succeeds: replayed events + promote to active
4. If fresh connect: initial_status + state + promote to active
5. Ongoing metrics/state/topology broadcasts during training

After promotion, the server sends application-level ``{"type":"ping","ts":<float>}``
heartbeats every ``ws_heartbeat_interval_sec`` (default 30s). Client must reply
with ``{"type":"pong"}`` within ``ws_heartbeat_pong_timeout_sec`` (default 10s)
or the connection is closed with 1006 (heartbeat timeout).
"""

import asyncio
import json
import logging
import time

from fastapi import WebSocket, WebSocketDisconnect

from api.websocket.manager import ReplayOutOfRange, ws_authenticate
from api.websocket.messages import create_initial_metrics_message, create_state_message

logger = logging.getLogger("juniper_cascor.api.websocket.training")


async def _await_resume_frame(websocket: WebSocket, ws_manager, resume_timeout: float) -> tuple[bool, bool]:
    """Wait for an optional resume frame.

    Returns (resumed, disconnected). If disconnected is True, the caller should
    return without further work.
    """
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=resume_timeout)
        msg = json.loads(raw)
        if msg.get("type") == "resume":
            return await _handle_resume(websocket, ws_manager, msg), False
    except asyncio.TimeoutError:
        pass  # No resume frame — fresh connect
    except json.JSONDecodeError:
        logger.debug("Non-JSON frame during resume handshake, treating as fresh connect")
    except WebSocketDisconnect:
        return False, True
    except Exception:
        logger.debug("Error during resume handshake, treating as fresh connect")
    return False, False


async def _send_initial_state(
    websocket: WebSocket,
    ws_manager,
    lifecycle,
    initial_metrics_count: int = 100,
) -> None:
    """Send initial status + state + recent metrics to a freshly-connected client.

    GAP-WS-16: ``initial_metrics`` is sent after ``state`` so a client can
    paint its time-series chart without a parallel REST poll. Set
    ``initial_metrics_count=0`` to disable the burst (clients then must
    request via ``subscribe_metrics`` or fall back to REST).
    """
    if lifecycle is None:
        return
    status = lifecycle.get_status()
    await ws_manager.send_personal_message(
        websocket,
        {"type": "initial_status", "data": status},
    )
    state_data = lifecycle.training_state.get_state()
    await ws_manager.send_personal_message(
        websocket,
        create_state_message(state_data),
    )
    if initial_metrics_count > 0:
        await _send_metrics_burst(websocket, ws_manager, lifecycle, initial_metrics_count)


async def _send_metrics_burst(websocket: WebSocket, ws_manager, lifecycle, count: int) -> None:
    """GAP-WS-16: send an ``initial_metrics`` burst with the most recent metrics."""
    try:
        metrics = lifecycle.get_metrics_history(count=count)
    except Exception:
        logger.debug("Training WS: get_metrics_history failed; sending empty burst", exc_info=True)
        metrics = []
    if metrics is None:
        metrics = []
    await ws_manager.send_personal_message(
        websocket,
        create_initial_metrics_message(metrics, current_seq=ws_manager.current_seq),
    )


async def _heartbeat_ping_loop(websocket: WebSocket, hb_interval: float, hb_timeout: float, pong_received: asyncio.Event) -> None:
    """Application-level ping/pong loop closing the connection on pong timeout."""
    while True:
        await asyncio.sleep(hb_interval)
        pong_received.clear()
        try:
            await websocket.send_json({"type": "ping", "ts": time.time()})
        except Exception:
            return  # Connection already closed
        try:
            await asyncio.wait_for(pong_received.wait(), timeout=hb_timeout)
        except asyncio.TimeoutError:
            logger.info("Training WS: heartbeat timeout, closing connection")
            try:
                await websocket.close(code=1006, reason="Heartbeat timeout")
            except Exception:
                logger.debug("Training WS: close after heartbeat timeout failed", exc_info=True)
            return


async def _recv_pong_loop(
    websocket: WebSocket,
    pong_received: asyncio.Event,
    ws_manager=None,
    lifecycle=None,
    subscribe_metrics_max_count: int = 100,
) -> None:
    """Receive loop dispatcher.

    Forwards ``pong`` frames to the heartbeat event and handles
    GAP-WS-16 ``subscribe_metrics`` requests by replying with an
    ``initial_metrics`` envelope. All other frame types are ignored
    (clients may send keep-alives the server does not need to act on).
    """
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, AttributeError):
                continue  # Non-JSON keep-alive frames are fine
            mtype = msg.get("type") if isinstance(msg, dict) else None
            if mtype == "pong":
                pong_received.set()
            elif mtype == "subscribe_metrics" and ws_manager is not None and lifecycle is not None:
                await _handle_subscribe_metrics(websocket, ws_manager, lifecycle, msg, subscribe_metrics_max_count)
    except WebSocketDisconnect:
        pass


async def _handle_subscribe_metrics(
    websocket: WebSocket,
    ws_manager,
    lifecycle,
    msg: dict,
    default_count: int,
) -> None:
    """GAP-WS-16: respond to a client ``subscribe_metrics`` with a metrics burst.

    Used when a client reconnects via the resume path (which only replays
    seq>last_seq from the broadcast buffer) but also wants the rolling
    metrics window — useful for tab reactivation or after a long gap.
    """
    data = msg.get("data") or {}
    requested = data.get("max_count")
    try:
        count = int(requested) if requested is not None else default_count
    except (TypeError, ValueError):
        count = default_count
    count = max(1, min(count, default_count))
    await _send_metrics_burst(websocket, ws_manager, lifecycle, count)


async def training_stream_handler(websocket: WebSocket) -> None:
    """Handle /ws/training WebSocket connections with optional resume.

    Protocol flow:
    1. Authenticate via API key header
    2. Accept as pending (connect_pending — not broadcast-eligible)
    3. Wait for optional 'resume' frame within handshake timeout
    4. On resume: validate server_instance_id, replay buffered events
    5. On fresh connect / failed resume: send initial_status + state
    6. Promote to active (now receives broadcasts)
    7. Keep-alive recv loop
    """
    if not await ws_authenticate(websocket):
        return

    ws_manager = getattr(websocket.app.state, "ws_manager", None)
    lifecycle = getattr(websocket.app.state, "lifecycle", None)
    settings = getattr(websocket.app.state, "settings", None)

    if ws_manager is None:
        await websocket.close(code=1011, reason="WebSocket manager not available")
        return

    connected = await ws_manager.connect_pending(websocket)
    if not connected:
        return

    resume_timeout = getattr(settings, "ws_resume_handshake_timeout_s", 5.0) if settings else 5.0
    initial_metrics_count = getattr(settings, "ws_initial_metrics_count", 100) if settings else 100

    try:
        resumed, disconnected = await _await_resume_frame(websocket, ws_manager, resume_timeout)
        if disconnected:
            return

        # Promote to active (now eligible for broadcasts)
        await ws_manager.promote_to_active(websocket)

        if not resumed:
            await _send_initial_state(websocket, ws_manager, lifecycle, initial_metrics_count)

        # Heartbeat + keep-alive: broadcasts come from training thread
        # via ws_manager.broadcast_from_thread(). The recv loop also
        # handles pong responses for dead-connection detection and
        # GAP-WS-16 subscribe_metrics requests.
        hb_interval = getattr(settings, "ws_heartbeat_interval_sec", 30) if settings else 30
        hb_timeout = getattr(settings, "ws_heartbeat_pong_timeout_sec", 10) if settings else 10
        pong_received = asyncio.Event()
        pong_received.set()  # No outstanding ping at start

        ping_task = asyncio.create_task(_heartbeat_ping_loop(websocket, hb_interval, hb_timeout, pong_received))
        try:
            await _recv_pong_loop(
                websocket,
                pong_received,
                ws_manager=ws_manager,
                lifecycle=lifecycle,
                subscribe_metrics_max_count=initial_metrics_count or 100,
            )
        finally:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass
    finally:
        await ws_manager.disconnect(websocket)


async def _handle_resume(
    websocket: WebSocket,
    ws_manager,
    msg: dict,
) -> bool:
    """Handle a resume request. Returns True if resume succeeded."""
    data = msg.get("data", {})
    last_seq = data.get("last_seq")
    client_server_id = data.get("server_instance_id")

    if last_seq is None or client_server_id is None:
        await ws_manager.send_personal_message(
            websocket,
            {
                "type": "resume_failed",
                "timestamp": time.time(),
                "data": {"reason": "malformed_resume"},
            },
        )
        logger.debug("Resume failed: malformed resume frame (missing last_seq or server_instance_id)")
        return False

    # D-15: server restart detection via UUID mismatch
    if client_server_id != ws_manager.server_instance_id:
        await ws_manager.send_personal_message(
            websocket,
            {
                "type": "resume_failed",
                "timestamp": time.time(),
                "data": {"reason": "server_restarted"},
            },
        )
        logger.info("Resume failed: server_instance_id mismatch (server restarted)")
        return False

    try:
        events = ws_manager.replay_since(last_seq)
    except ReplayOutOfRange as e:
        await ws_manager.send_personal_message(
            websocket,
            {
                "type": "resume_failed",
                "timestamp": time.time(),
                "data": {"reason": "out_of_range"},
            },
        )
        logger.info("Resume failed: %s", e)
        return False

    # Resume succeeded
    await ws_manager.send_personal_message(
        websocket,
        {
            "type": "resume_ok",
            "timestamp": time.time(),
            "data": {"replayed_count": len(events)},
        },
    )

    # Replay buffered events
    for event in events:
        await ws_manager.send_personal_message(websocket, event)

    logger.info("Resume succeeded: replayed %d events from seq %d", len(events), last_seq)
    return True
