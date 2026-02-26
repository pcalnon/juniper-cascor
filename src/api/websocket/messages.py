"""Standardized WebSocket message builders.

All WebSocket messages follow the format:
{
    "type": "<message_type>",
    "timestamp": <unix_timestamp>,
    "data": { ... }
}

Compatible with juniper-canopy's WebSocket message protocol.
"""

import time
from typing import Any, Dict, Optional


def create_metrics_message(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a metrics update message."""
    return {
        "type": "metrics",
        "timestamp": time.time(),
        "data": data,
    }


def create_state_message(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a training state update message."""
    return {
        "type": "state",
        "timestamp": time.time(),
        "data": data,
    }


def create_topology_message(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a network topology message."""
    return {
        "type": "topology",
        "timestamp": time.time(),
        "data": data,
    }


def create_event_message(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a training event message."""
    return {
        "type": "event",
        "timestamp": time.time(),
        "data": data,
    }


def create_cascade_add_message(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a cascade unit addition message."""
    return {
        "type": "cascade_add",
        "timestamp": time.time(),
        "data": data,
    }


def create_control_ack_message(
    command: str,
    status: str,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a control command acknowledgment message."""
    msg: Dict[str, Any] = {
        "type": "command_response",
        "timestamp": time.time(),
        "data": {
            "command": command,
            "status": status,
        },
    }
    if data:
        msg["data"]["result"] = data
    if error:
        msg["data"]["error"] = error
    return msg
