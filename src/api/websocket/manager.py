"""WebSocket connection manager for real-time streaming.

Thread-safe manager that handles:
- Connection lifecycle (connect/disconnect/pending)
- Monotonically increasing sequence numbers on broadcasts
- Replay buffer for client reconnection
- Thread-safe bridge for broadcasting from training threads
- Configurable send timeout to prevent slow-client fan-out blocking
- Bounded connection limit
"""

import asyncio
import bisect
import contextlib
import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import CancelledError
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger("juniper_cascor.api.websocket")


class ReplayOutOfRange(Exception):
    """Raised when the requested seq is no longer in the replay buffer."""


async def ws_authenticate(websocket: WebSocket) -> bool:
    """Authenticate a WebSocket connection via X-API-Key header.

    Shared utility replacing inline auth boilerplate in each stream handler.
    BaseHTTPMiddleware cannot intercept WebSocket upgrades, so each WS
    endpoint must authenticate independently.

    Returns:
        True if authenticated (or auth disabled). False if auth failed
        (connection is closed with 4001).
    """
    auth = getattr(websocket.app.state, "api_key_auth", None)
    if auth is not None and auth.enabled:
        api_key = websocket.headers.get("X-API-Key")
        if not auth.validate(api_key):
            await websocket.close(code=4001, reason="Authentication required")
            return False
    return True


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting.

    Provides both async and thread-safe broadcasting to support the
    training thread -> async WebSocket bridge pattern. Assigns monotonically
    increasing sequence numbers to all broadcast messages and maintains a
    bounded replay buffer for client reconnection.
    """

    def __init__(
        self,
        max_connections: int = 50,
        max_replay_buffer_size: int = 1024,
        send_timeout_seconds: float = 0.5,
        max_connections_per_ip: int = 5,
    ):
        # Connection tracking
        self._active_connections: Set[WebSocket] = set()
        self._pending_connections: Set[WebSocket] = set()
        self._max_connections = max_connections
        # SEC-03: per-IP cap. Stored as ``ip -> count`` and updated under
        # ``self._lock`` alongside the connection sets. Unknown clients
        # (no ``websocket.client``) all share the sentinel key "unknown",
        # which is intentional — if the reverse proxy strips the client
        # address we cannot distinguish attackers anyway.
        self._max_connections_per_ip = max_connections_per_ip
        self._per_ip_counts: Dict[str, int] = {}
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._connection_meta: Dict[WebSocket, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

        # Server identity (D-15: programmatic restart detection)
        self._server_instance_id: str = str(uuid.uuid4())
        self._server_start_time: float = time.monotonic()

        # Sequencing and replay (threading.Lock: used from both async and sync contexts)
        self._next_seq: int = 1
        self._seq_lock = threading.Lock()
        self._replay_buffer: deque = deque(maxlen=max_replay_buffer_size if max_replay_buffer_size > 0 else None)
        self._replay_buffer_max_size = max_replay_buffer_size

        # Send timeout (GAP-WS-07 quick-fix)
        self._send_timeout_seconds = send_timeout_seconds

        logger.info(
            "WebSocketManager initialized (max_connections=%d, replay_buffer=%d, send_timeout=%.1fs)",
            max_connections,
            max_replay_buffer_size,
            send_timeout_seconds,
        )

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store the event loop reference for thread-safe broadcasting."""
        self._event_loop = loop

    @property
    def server_instance_id(self) -> str:
        """UUID4 identifying this server process (D-15)."""
        return self._server_instance_id

    @property
    def current_seq(self) -> int:
        """The last assigned sequence number (0 if none assigned yet)."""
        with self._seq_lock:
            return self._next_seq - 1

    @property
    def connection_count(self) -> int:
        return len(self._active_connections)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _build_connection_established(self) -> dict:
        """Build the connection_established handshake message."""
        return {
            "type": "connection_established",
            "timestamp": time.time(),
            "data": {
                "connections": self.connection_count + len(self._pending_connections),
                "server_instance_id": self._server_instance_id,
                "server_start_time": self._server_start_time,
                "replay_buffer_capacity": self._replay_buffer_max_size,
            },
        }

    def _source_ip(self, websocket: WebSocket) -> str:
        """Best-effort source-IP for per-IP accounting (SEC-03)."""
        client = getattr(websocket, "client", None)
        if client is not None:
            try:
                return client[0] or "unknown"
            except (IndexError, TypeError):
                return "unknown"
        return "unknown"

    def _check_and_reserve_per_ip_slot(self, source_ip: str) -> bool:
        """Reserve a per-IP slot; returns False if the cap would be exceeded.

        Must be called with ``self._lock`` held. On success the caller is
        responsible for invoking ``_release_per_ip_slot`` on disconnect.
        """
        current = self._per_ip_counts.get(source_ip, 0)
        if current >= self._max_connections_per_ip:
            return False
        self._per_ip_counts[source_ip] = current + 1
        return True

    def _release_per_ip_slot(self, source_ip: Optional[str]) -> None:
        """Release a per-IP slot previously reserved by _check_and_reserve_per_ip_slot."""
        if not source_ip:
            return
        current = self._per_ip_counts.get(source_ip, 0)
        if current <= 1:
            self._per_ip_counts.pop(source_ip, None)
        else:
            self._per_ip_counts[source_ip] = current - 1

    async def connect(self, websocket: WebSocket) -> bool:
        """Accept and register a WebSocket connection as immediately active.

        Returns:
            True if connected, False if connection limit reached.
        """
        async with self._lock:
            total = len(self._active_connections) + len(self._pending_connections)
            if total >= self._max_connections:
                await websocket.close(code=1013, reason="Maximum connections reached")
                logger.warning("Connection rejected: limit of %d reached", self._max_connections)
                return False

            source_ip = self._source_ip(websocket)
            if not self._check_and_reserve_per_ip_slot(source_ip):
                # SEC-03: per-IP cap exceeded. Close with the same 1013
                # used for the global cap so clients see a single "too
                # many connections" signal.
                await websocket.close(code=1013, reason="Per-IP connection limit reached")
                logger.warning(
                    "Connection rejected: per-IP limit of %d reached for %s",
                    self._max_connections_per_ip,
                    source_ip,
                )
                return False

            await websocket.accept()
            self._active_connections.add(websocket)
            self._connection_meta[websocket] = {"connected_at": time.time(), "source_ip": source_ip}
            logger.info("WebSocket connected (%d active)", self.connection_count)

        await self._send_json(websocket, self._build_connection_established())
        return True

    async def connect_pending(self, websocket: WebSocket) -> bool:
        """Accept a WebSocket in pending state (not broadcast-eligible).

        Used during the resume handshake window (D-14). The connection
        receives connection_established but does NOT receive broadcasts
        until promote_to_active() is called.

        Returns:
            True if accepted, False if connection limit reached.
        """
        async with self._lock:
            total = len(self._active_connections) + len(self._pending_connections)
            if total >= self._max_connections:
                await websocket.close(code=1013, reason="Maximum connections reached")
                logger.warning("Connection rejected: limit of %d reached", self._max_connections)
                return False

            source_ip = self._source_ip(websocket)
            if not self._check_and_reserve_per_ip_slot(source_ip):
                await websocket.close(code=1013, reason="Per-IP connection limit reached")
                logger.warning(
                    "Pending connection rejected: per-IP limit of %d reached for %s",
                    self._max_connections_per_ip,
                    source_ip,
                )
                return False

            await websocket.accept()
            self._pending_connections.add(websocket)
            self._connection_meta[websocket] = {"connected_at": time.time(), "pending": True, "source_ip": source_ip}
            logger.info("WebSocket connected as pending (%d pending)", len(self._pending_connections))

        await self._send_json(websocket, self._build_connection_established())
        return True

    async def promote_to_active(self, websocket: WebSocket) -> None:
        """Move a pending connection to active (broadcast-eligible).

        Must be called after the resume handshake completes (D-14).
        """
        async with self._lock:
            self._pending_connections.discard(websocket)
            self._active_connections.add(websocket)
            meta = self._connection_meta.get(websocket, {})
            meta.pop("pending", None)
            logger.info(
                "WebSocket promoted to active (%d active, %d pending)",
                self.connection_count,
                len(self._pending_connections),
            )

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection (from active or pending)."""
        async with self._lock:
            self._active_connections.discard(websocket)
            self._pending_connections.discard(websocket)
            meta = self._connection_meta.pop(websocket, None)
            # SEC-03: release the per-IP slot reserved at connect time so
            # a client who reconnects cleanly does not accrue phantom
            # counts against the cap.
            if meta is not None:
                self._release_per_ip_slot(meta.get("source_ip"))
            logger.info(
                "WebSocket disconnected (%d active, %d pending)",
                self.connection_count,
                len(self._pending_connections),
            )

    # ------------------------------------------------------------------
    # Sequencing and replay
    # ------------------------------------------------------------------

    def _assign_seq_and_buffer(self, message: dict) -> dict:
        """Assign monotonically increasing seq and buffer for replay.

        Called on the event loop thread before broadcast fan-out.
        Uses threading.Lock (not asyncio.Lock) because it may be invoked
        from both sync and async contexts. The lock is held only for O(1)
        operations (integer increment + deque append).
        """
        with self._seq_lock:
            seq = self._next_seq
            self._next_seq += 1
            enriched = {**message, "seq": seq, "emitted_at_monotonic": time.monotonic()}
            if self._replay_buffer_max_size > 0:
                self._replay_buffer.append(enriched)
            return enriched

    def replay_since(self, last_seq: int) -> List[dict]:
        """Return buffered messages with seq > last_seq.

        Args:
            last_seq: The last sequence number the client received.

        Returns:
            List of message dicts with seq > last_seq, in order.

        Raises:
            ReplayOutOfRange: If last_seq is older than the oldest buffered
                message, or if the replay buffer is disabled (size 0).
        """
        with self._seq_lock:
            if self._replay_buffer_max_size <= 0:
                raise ReplayOutOfRange("Replay buffer disabled")
            if not self._replay_buffer:
                if last_seq > 0:
                    raise ReplayOutOfRange("Buffer empty, cannot verify continuity")
                return []
            oldest_seq = self._replay_buffer[0].get("seq", 0)
            if last_seq < oldest_seq - 1:
                raise ReplayOutOfRange(f"Requested seq {last_seq} older than oldest buffered seq {oldest_seq}")
            # PERF-CC-02: replay buffer seqs are monotonically increasing
            # (assigned by _assign_seq_and_buffer under _seq_lock), so a
            # bisect_right gets us O(log n) instead of an O(n) linear scan.
            buffered = list(self._replay_buffer)
            seqs = [msg.get("seq", 0) for msg in buffered]
            idx = bisect.bisect_right(seqs, last_seq)
            return buffered[idx:]

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast(self, message: dict) -> None:
        """Assign seq and send a message to all active (non-pending) clients."""
        if not self._active_connections:
            return
        message = self._assign_seq_and_buffer(message)
        disconnected = []
        for ws in self._active_connections.copy():
            if not await self._send_json(ws, message):
                disconnected.append(ws)
        for ws in disconnected:
            await self.disconnect(ws)

    def broadcast_from_thread(self, message: dict) -> None:
        """Thread-safe broadcast using asyncio.run_coroutine_threadsafe.

        Called from the training thread to push messages to all WebSocket
        clients. Adds a done callback to log exceptions from the coroutine
        (GAP-WS-29).
        """
        if self._event_loop is None or self._event_loop.is_closed():
            return
        coro = self.broadcast(message)
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
            future.add_done_callback(self._log_broadcast_exception)
        except Exception:
            coro.close()
            logger.debug("Cannot broadcast: event loop unavailable or closed")

    @staticmethod
    def _log_broadcast_exception(future) -> None:
        """Done callback for broadcast futures — logs exceptions (GAP-WS-29)."""
        try:
            exc = future.exception()
        except CancelledError:
            return
        if exc is not None:
            logger.error("Broadcast from thread failed: %s", exc, exc_info=exc)

    async def send_personal_message(self, websocket: WebSocket, message: dict) -> bool:
        """Send a message to a specific client (no seq assignment)."""
        return await self._send_json(websocket, message)

    async def _send_json(self, websocket: WebSocket, message: dict) -> bool:
        """Send JSON message to a single WebSocket with timeout.

        Applies a configurable send timeout (default 0.5s) to prevent slow
        clients from blocking broadcast fan-out (GAP-WS-07 quick-fix).
        Returns False on failure or timeout.
        """
        try:
            await asyncio.wait_for(
                websocket.send_json(message),
                timeout=self._send_timeout_seconds,
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("WebSocket send timed out after %.1fs", self._send_timeout_seconds)
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def close_all(self) -> None:
        """Close all active and pending connections (used during shutdown).

        Holds ``self._lock`` around the set mutations so that a concurrent
        ``connect()`` or ``disconnect()`` cannot race with shutdown and
        corrupt the connection set (CR-025). The actual ``ws.close()`` calls
        are issued against a snapshot taken under the lock, avoiding
        deadlock risk from re-entering the lock via ``disconnect()`` on
        exception paths.
        """
        async with self._lock:
            snapshot = list(self._active_connections) + list(self._pending_connections)
            self._active_connections.clear()
            self._pending_connections.clear()
            self._connection_meta.clear()

        for ws in snapshot:
            with contextlib.suppress(Exception):
                await ws.close(code=1001, reason="Server shutting down")
        logger.info("All WebSocket connections closed")
