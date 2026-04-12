"""WebSocket handler for /ws/training — real-time training metrics stream.

Server-to-client streaming endpoint with optional resume support. On connect:
1. connection_established message (from manager.connect_pending)
2. Optional resume handshake: client sends ``resume`` frame within timeout
3. If resume succeeds: replayed events + promote to active
4. If fresh connect: initial_status + state + promote to active
5. Ongoing metrics/state/topology broadcasts during training

The client sends only the optional resume frame during the handshake window.
After promotion, the recv loop detects disconnection only.
"""

import asyncio
import json
import logging
import time

from fastapi import WebSocket, WebSocketDisconnect

from api.websocket.manager import ReplayOutOfRange, ws_authenticate
from api.websocket.messages import create_state_message

logger = logging.getLogger("juniper_cascor.api.websocket.training")


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

    try:
        # Wait for optional resume frame
        resumed = False
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=resume_timeout)
            msg = json.loads(raw)
            if msg.get("type") == "resume":
                resumed = await _handle_resume(websocket, ws_manager, msg)
        except asyncio.TimeoutError:
            pass  # No resume frame — fresh connect
        except json.JSONDecodeError:
            logger.debug("Non-JSON frame during resume handshake, treating as fresh connect")
        except WebSocketDisconnect:
            return
        except Exception:
            logger.debug("Error during resume handshake, treating as fresh connect")

        # Promote to active (now eligible for broadcasts)
        await ws_manager.promote_to_active(websocket)

        if not resumed:
            # Fresh connect: send initial status + current state
            if lifecycle is not None:
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

        # Keep connection alive — broadcasts come from training thread
        # via ws_manager.broadcast_from_thread()
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
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
