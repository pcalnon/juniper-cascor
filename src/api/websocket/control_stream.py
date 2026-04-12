"""WebSocket handler for /ws/control — training command channel.

Client-to-server command endpoint. Accepts JSON commands:
{
    "command": "start" | "stop" | "pause" | "resume" | "reset" | "set_params",
    "command_id": "<optional-uuid>",  // echoed in response for correlation
    "params": { ... }  // optional, for start/set_params
}

Responds with command_response acknowledgments. Note: command_response
messages have NO ``seq`` field (D-03 canonical). The /ws/control channel
has no replay buffer.
"""

import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from api.websocket.manager import ws_authenticate
from api.websocket.messages import create_control_ack_message

logger = logging.getLogger("juniper_cascor.api.websocket.control")

_VALID_COMMANDS = {"start", "stop", "pause", "resume", "reset", "set_params"}
_MAX_MESSAGE_SIZE = 65536  # 64KB


async def control_stream_handler(websocket: WebSocket) -> None:
    """Handle /ws/control WebSocket connections."""
    if not await ws_authenticate(websocket):
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
                await websocket.close(code=1003, reason="Malformed JSON")
                return

            command = msg.get("command", "")
            command_id = msg.get("command_id")

            if command not in _VALID_COMMANDS:
                await websocket.send_json(create_control_ack_message(command, "error", error=f"Unknown command: {command}", command_id=command_id))
                continue

            if lifecycle is None:
                await websocket.send_json(create_control_ack_message(command, "error", error="Lifecycle manager not available", command_id=command_id))
                continue

            try:
                result = _execute_command(lifecycle, command, msg.get("params"))
                await websocket.send_json(create_control_ack_message(command, "success", data=result, command_id=command_id))
            except Exception as e:
                logger.error("Command '%s' failed: %s", command, e)
                await websocket.send_json(create_control_ack_message(command, "error", error="Command execution failed", command_id=command_id))

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
    elif command == "set_params":
        if not params:
            raise ValueError("set_params requires a 'params' dict")
        return lifecycle.update_params(params)
    else:
        raise ValueError(f"Unhandled command: {command}")
