"""WebSocket handler for /ws/training — real-time training metrics stream.

Server-to-client streaming endpoint. On connect, clients receive:
1. connection_established message (from manager.connect)
2. initial_status message with current training state
3. Ongoing metrics/state/topology broadcasts during training

The client does not send messages on this channel — it is read-only.
"""

import logging

from fastapi import WebSocket, WebSocketDisconnect

from api.websocket.messages import create_state_message

logger = logging.getLogger("juniper_cascor.api.websocket.training")


async def training_stream_handler(websocket: WebSocket) -> None:
    """Handle /ws/training WebSocket connections."""
    ws_manager = getattr(websocket.app.state, "ws_manager", None)
    lifecycle = getattr(websocket.app.state, "lifecycle", None)

    if ws_manager is None:
        await websocket.close(code=1011, reason="WebSocket manager not available")
        return

    connected = await ws_manager.connect(websocket)
    if not connected:
        return

    try:
        # Send initial training status
        if lifecycle is not None:
            status = lifecycle.get_status()
            await ws_manager.send_personal_message(
                websocket,
                {"type": "initial_status", "data": status},
            )

            # Send current training state
            state_data = lifecycle.training_state.get_state()
            await ws_manager.send_personal_message(
                websocket,
                create_state_message(state_data),
            )

        # Keep connection alive — broadcasts come from training thread
        # via ws_manager.broadcast_from_thread()
        while True:
            # Wait for client messages (WebSocket requires recv loop to detect disconnect)
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(websocket)
