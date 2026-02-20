"""WebSocket connection manager for real-time streaming.

Thread-safe manager that handles:
- Connection lifecycle (connect/disconnect)
- Broadcasting to all connected clients
- Thread-safe bridge for broadcasting from training threads
- Bounded connection limit
"""

import asyncio
import contextlib
import logging
import time
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger("juniper_cascor.api.websocket")


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting.

    Provides both async and thread-safe broadcasting to support the
    training thread -> async WebSocket bridge pattern.
    """

    def __init__(self, max_connections: int = 50):
        self._active_connections: Set[WebSocket] = set()
        self._max_connections = max_connections
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._connection_meta: Dict[WebSocket, Dict[str, Any]] = {}
        logger.info(f"WebSocketManager initialized (max_connections={max_connections})")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store the event loop reference for thread-safe broadcasting."""
        self._event_loop = loop

    @property
    def connection_count(self) -> int:
        return len(self._active_connections)

    async def connect(self, websocket: WebSocket) -> bool:
        """Accept and register a WebSocket connection.

        Returns:
            True if connected, False if connection limit reached.
        """
        if len(self._active_connections) >= self._max_connections:
            await websocket.close(code=1013, reason="Maximum connections reached")
            logger.warning(f"Connection rejected: limit of {self._max_connections} reached")
            return False

        await websocket.accept()
        self._active_connections.add(websocket)
        self._connection_meta[websocket] = {
            "connected_at": time.time(),
        }
        logger.info(f"WebSocket connected ({self.connection_count} active)")

        # Send connection established message
        await self._send_json(
            websocket,
            {
                "type": "connection_established",
                "timestamp": time.time(),
                "data": {"connections": self.connection_count},
            },
        )
        return True

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        self._active_connections.discard(websocket)
        self._connection_meta.pop(websocket, None)
        logger.info(f"WebSocket disconnected ({self.connection_count} active)")

    async def broadcast(self, message: dict) -> None:
        """Send a message to all connected clients."""
        if not self._active_connections:
            return
        disconnected = []
        for ws in self._active_connections.copy():
            if not await self._send_json(ws, message):
                disconnected.append(ws)
        for ws in disconnected:
            await self.disconnect(ws)

    def broadcast_from_thread(self, message: dict) -> None:
        """Thread-safe broadcast using asyncio.run_coroutine_threadsafe.

        Called from the training thread to push messages to all WebSocket clients.
        """
        if self._event_loop is None or self._event_loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self._event_loop)
        except RuntimeError:
            logger.debug("Event loop closed, cannot broadcast")

    async def send_personal_message(self, websocket: WebSocket, message: dict) -> bool:
        """Send a message to a specific client."""
        return await self._send_json(websocket, message)

    async def _send_json(self, websocket: WebSocket, message: dict) -> bool:
        """Send JSON message to a single WebSocket. Returns False on failure."""
        try:
            await websocket.send_json(message)
            return True
        except Exception:
            return False

    async def close_all(self) -> None:
        """Close all active connections (used during shutdown)."""
        for ws in self._active_connections.copy():
            with contextlib.suppress(Exception):
                await ws.close(code=1001, reason="Server shutting down")
        self._active_connections.clear()
        self._connection_meta.clear()
        logger.info("All WebSocket connections closed")
