"""Origin validation for WebSocket connections (M-SEC-01b).

Validates the Origin header against a configured allowlist before
accepting WebSocket upgrades. Empty allowlist means reject all
(fail-closed). Wildcard '*' is refused at the settings level (C-30).
"""

import logging
from typing import List

from fastapi import WebSocket

logger = logging.getLogger("juniper_cascor.api.websocket.security")


def validate_origin(websocket: WebSocket, allowlist: List[str]) -> bool:
    """Check if the WebSocket connection's Origin header is allowed.

    Args:
        websocket: The incoming WebSocket connection.
        allowlist: List of allowed origin strings. Empty = reject all.

    Returns:
        True if the origin is in the allowlist, False otherwise.
        Missing Origin header is also rejected.
    """
    origin = websocket.headers.get("origin")

    # No Origin header — reject (browser connections always send Origin)
    if not origin:
        logger.info("WebSocket origin rejected: no Origin header present")
        return False

    # Case-insensitive, port-significant comparison
    origin_lower = origin.lower().rstrip("/")
    for allowed in allowlist:
        if origin_lower == allowed.lower().rstrip("/"):
            return True

    logger.info("WebSocket origin rejected: %s not in allowlist", origin)
    return False
