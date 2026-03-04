"""WebSocket handler for /ws/control — training command channel.

Client-to-server command endpoint. Accepts JSON commands:
{
    "command": "start" | "stop" | "pause" | "resume" | "reset" | "set_params",
    "params": { ... }  // optional, for start/set_params
}

Responds with command_response acknowledgments.
"""

import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from api.websocket.messages import create_control_ack_message

logger = logging.getLogger("juniper_cascor.api.websocket.control")

_VALID_COMMANDS = {"start", "stop", "pause", "resume", "reset"}
_MAX_MESSAGE_SIZE = 65536  # 64KB


async def control_stream_handler(websocket: WebSocket) -> None:
    """Handle /ws/control WebSocket connections."""
    # Authenticate WebSocket connection (BaseHTTPMiddleware does not intercept WS)
    auth = getattr(websocket.app.state, "api_key_auth", None)
    if auth is not None and auth.enabled:
        api_key = websocket.headers.get("X-API-Key")
        if not auth.validate(api_key):
            await websocket.close(code=4001, reason="Authentication required")
            return

    lifecycle = getattr(websocket.app.state, "lifecycle", None)

    await websocket.accept()
    await websocket.send_json(
        {
            "type": "connection_established",
            "data": {"channel": "control"},
        }
    )

    try:
        while True:
            raw = await websocket.receive_text()

            if len(raw) > _MAX_MESSAGE_SIZE:
                await websocket.send_json(create_control_ack_message("unknown", "error", error="Message too large"))
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(create_control_ack_message("unknown", "error", error="Invalid JSON"))
                continue

            command = msg.get("command", "")

            if command not in _VALID_COMMANDS:
                await websocket.send_json(create_control_ack_message(command, "error", error=f"Unknown command: {command}"))
                continue

            if lifecycle is None:
                await websocket.send_json(create_control_ack_message(command, "error", error="Lifecycle manager not available"))
                continue

            try:
                result = _execute_command(lifecycle, command, msg.get("params"))
                await websocket.send_json(create_control_ack_message(command, "success", data=result))
            except Exception as e:
                logger.error("Command '%s' failed: %s", command, e)
                await websocket.send_json(create_control_ack_message(command, "error", error="Command execution failed"))

    except WebSocketDisconnect:
        pass


def _execute_command(lifecycle, command: str, params: dict = None) -> dict:
    """Execute a training control command.

    Args:
        lifecycle: TrainingLifecycleManager instance
        command: Command name
        params: Optional parameters

    Returns:
        Command result dictionary
    """
    if command == "start":
        return lifecycle.start_training()
    elif command == "stop":
        return lifecycle.stop_training()
    elif command == "pause":
        return lifecycle.pause_training()
    elif command == "resume":
        return lifecycle.resume_training()
    elif command == "reset":
        return lifecycle.reset()
    else:
        raise ValueError(f"Unhandled command: {command}")
