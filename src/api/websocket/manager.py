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
import json
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
        max_message_size_bytes: int = 60_000,
        chunk_payload_size_bytes: int = 32_000,
    ):
        # Connection tracking
        self._active_connections: Set[WebSocket] = set()
        self._pending_connections: Set[WebSocket] = set()
        # OBS-WIRE-02 (Q3): per-endpoint active-connection sets, used to
        # populate the ``cascor_ws_connections_active{endpoint}`` gauge
        # without inferring endpoint from request paths after the fact.
        # Keys are the closed-set endpoint values registered in
        # ``api.observability._WS_ENDPOINTS``. Sets are kept disjoint
        # from one another and disjoint from ``_active_connections``
        # only at construction; in steady state every WS in
        # ``_active_connections`` lives in exactly one endpoint set.
        # Pending (resume-handshake) connections are NOT counted here —
        # the gauge tracks broadcast-eligible connections.
        self._endpoint_connections: Dict[str, Set[WebSocket]] = {
            "training": set(),
            "control": set(),
            "workers": set(),
        }
        # Reverse lookup: ws -> endpoint, populated at connect time
        # and consulted at disconnect to know which endpoint set to
        # mutate without scanning all of them.
        self._connection_endpoint: Dict[WebSocket, str] = {}
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

        # GAP-WS-16: bandwidth counters. Updated under _seq_lock because the
        # send path is invoked from both the asyncio event loop and the
        # broadcast-from-thread shim. Counters are cumulative since process
        # start; surfaced via /v1/metrics/transport for before/after validation.
        self._bytes_sent_total: int = 0
        self._messages_sent_total: int = 0
        self._messages_sent_by_type: Dict[str, int] = {}
        self._bytes_sent_by_type: Dict[str, int] = {}
        self._send_failures: int = 0

        # GAP-WS-18: message-size guard + chunking. Broadcasts whose serialized
        # JSON exceeds ``max_message_size_bytes`` are split into a sequence of
        # ``chunked_message`` envelopes (each with payload ≤ ``chunk_payload_size_bytes``)
        # so we never push a single frame over the typical 64 KB WebSocket
        # intermediary limit. Each chunk is its own broadcast (own seq, own
        # replay-buffer slot), so resume on reconnect reorders them naturally.
        # ``messages_chunked_total`` counts how often a logical message was
        # chunked (not how many chunks were emitted) — surfaced via
        # transport_stats() for observability.
        self._max_message_size_bytes = max_message_size_bytes
        self._chunk_payload_size_bytes = chunk_payload_size_bytes
        self._messages_chunked_total: int = 0
        self._chunks_emitted_total: int = 0

        logger.info(
            "WebSocketManager initialized (max_connections=%d, replay_buffer=%d, send_timeout=%.1fs)",
            max_connections,
            max_replay_buffer_size,
            send_timeout_seconds,
        )

        # OBS-WIRE-02 (3.3): emit the configured replay-buffer capacity
        # exactly once at construction. Defensive try/except mirrors
        # the OBS-WIRE-01 pattern — prometheus_client may be unavailable
        # in certain test environments.
        try:
            from api.observability import ws_set_replay_buffer_capacity

            ws_set_replay_buffer_capacity(max_replay_buffer_size)
        except Exception:
            logger.debug("ws_set_replay_buffer_capacity emission failed", exc_info=True)

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
    # Per-endpoint connection bookkeeping (OBS-WIRE-02 / Q3)
    # ------------------------------------------------------------------

    def _emit_endpoint_gauge(self, endpoint: str) -> None:
        """Push the current per-endpoint count to the Prometheus gauge.

        OBS-WIRE-02 (Q3): pure best-effort emission — defensive so a
        prometheus_client absence (test envs) does not block the
        connect/disconnect path. ``endpoint`` MUST be in the closed
        set validated by ``ws_set_connections_active``.
        """
        bucket = self._endpoint_connections.get(endpoint)
        if bucket is None:
            return
        try:
            from api.observability import ws_set_connections_active

            ws_set_connections_active(endpoint, len(bucket))
        except Exception:
            logger.debug("ws_set_connections_active emission failed for endpoint=%s", endpoint, exc_info=True)

    def register_endpoint_connection(self, websocket: WebSocket, endpoint: str) -> None:
        """Add ``websocket`` to the ``endpoint`` bucket and emit the gauge.

        OBS-WIRE-02 (Q3): callers (each WS endpoint handler) invoke this
        after ``websocket.accept()`` so the gauge reflects the
        broadcast-eligible connection set, not the in-handshake set.
        Wrap in a try/finally with :meth:`unregister_endpoint_connection`
        so disconnects always re-emit, even on exception paths.
        """
        if endpoint not in self._endpoint_connections:
            # Unknown endpoint — log once and treat as no-op rather than
            # silently miscount. The closed set is enforced again by
            # ``ws_set_connections_active`` so this is just defense in
            # depth.
            logger.warning("register_endpoint_connection: unknown endpoint %r (no-op)", endpoint)
            return
        self._endpoint_connections[endpoint].add(websocket)
        self._connection_endpoint[websocket] = endpoint
        self._emit_endpoint_gauge(endpoint)

    def unregister_endpoint_connection(self, websocket: WebSocket) -> None:
        """Remove ``websocket`` from its endpoint bucket and re-emit the gauge.

        OBS-WIRE-02 (Q3): idempotent — repeat calls are no-ops. Looks up
        the endpoint via ``_connection_endpoint`` so the caller does not
        need to re-pass it (matches the existing
        :meth:`disconnect` pattern that has no endpoint argument).
        """
        endpoint = self._connection_endpoint.pop(websocket, None)
        if endpoint is None:
            return
        bucket = self._endpoint_connections.get(endpoint)
        if bucket is not None:
            bucket.discard(websocket)
        self._emit_endpoint_gauge(endpoint)

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

        OBS-WIRE-02 (3.1, 3.2): emits ``cascor_ws_seq_current`` (current
        sequence number) and ``cascor_ws_replay_buffer_occupancy`` (deque
        size) on every assignment. Both are computed under the lock — the
        deque ``len()`` is O(1) — and emission is fired *after* releasing
        the lock so a Prometheus exception cannot cascade into a held
        lock. Defensive try/except mirrors the OBS-WIRE-01 pattern.
        """
        with self._seq_lock:
            seq = self._next_seq
            self._next_seq += 1
            enriched = {**message, "seq": seq, "emitted_at_monotonic": time.monotonic()}
            if self._replay_buffer_max_size > 0:
                self._replay_buffer.append(enriched)
            occupancy_snapshot = len(self._replay_buffer) if self._replay_buffer_max_size > 0 else 0
        try:
            from api.observability import ws_set_replay_buffer_occupancy, ws_set_seq_current

            ws_set_seq_current(seq)
            ws_set_replay_buffer_occupancy(occupancy_snapshot)
        except Exception:
            logger.debug("ws_set_seq_current / ws_set_replay_buffer_occupancy emission failed", exc_info=True)
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
    # Chunking (GAP-WS-18)
    # ------------------------------------------------------------------

    def _maybe_chunk_message(self, message: dict) -> List[dict]:
        """Return [message] if under threshold, else a list of chunked envelopes.

        GAP-WS-18: oversized broadcasts (~64 KB) silently tear down WebSocket
        connections at intermediaries. We split here, BEFORE seq assignment,
        so each chunk gets its own seq + replay slot and resume-on-reconnect
        reorders chunks naturally.

        Chunking is skipped (returns ``[message]``) when:
        - ``ws_max_message_size_bytes`` is 0 (kill-switch for tests)
        - The serialized JSON length is ≤ the threshold
        - The message is already a ``chunked_message`` envelope (no recursion)
        """
        if self._max_message_size_bytes <= 0:
            return [message]
        if message.get("type") == "chunked_message":
            return [message]
        try:
            serialized = json.dumps(message, default=str)
        except (TypeError, ValueError):
            # If we can't serialize for sizing, let _send_json handle the error.
            return [message]
        if len(serialized) <= self._max_message_size_bytes:
            return [message]

        chunk_size = self._chunk_payload_size_bytes
        chunk_id = str(uuid.uuid4())
        original_type = str(message.get("type") or "unknown")
        payloads = [serialized[i : i + chunk_size] for i in range(0, len(serialized), chunk_size)]
        total = len(payloads)
        chunks = [
            {
                "type": "chunked_message",
                "timestamp": time.time(),
                "data": {
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "total_chunks": total,
                    "original_type": original_type,
                    "payload": payload,
                },
            }
            for idx, payload in enumerate(payloads)
        ]
        with self._seq_lock:
            self._messages_chunked_total += 1
            self._chunks_emitted_total += total
        logger.info(
            "WebSocket: chunked %s message (%d bytes) into %d chunks (chunk_id=%s)",
            original_type,
            len(serialized),
            total,
            chunk_id,
        )
        return chunks

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast(self, message: dict) -> None:
        """Assign seq and send a message to all active (non-pending) clients.

        GAP-WS-18: oversized messages are split into chunked_message envelopes
        before seq assignment so each chunk gets its own seq and replay slot.
        """
        if not self._active_connections:
            return
        for sub_message in self._maybe_chunk_message(message):
            enriched = self._assign_seq_and_buffer(sub_message)
            disconnected = []
            for ws in self._active_connections.copy():
                if not await self._send_json(ws, enriched):
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
        """Done callback for broadcast futures — logs exceptions (GAP-WS-29).

        OBS-WIRE-02 (3.8): also increments
        ``cascor_ws_broadcast_from_thread_errors_total`` so the
        previously log-only error path is observable as a counter
        (paired with GAP-WS-29).
        """
        try:
            exc = future.exception()
        except CancelledError:
            return
        if exc is not None:
            logger.error("Broadcast from thread failed: %s", exc, exc_info=exc)
            try:
                from api.observability import ws_inc_broadcast_from_thread_errors

                ws_inc_broadcast_from_thread_errors()
            except Exception:
                logger.debug("ws_inc_broadcast_from_thread_errors emission failed", exc_info=True)

    async def send_personal_message(self, websocket: WebSocket, message: dict) -> bool:
        """Send a message to a specific client (no seq assignment).

        GAP-WS-18: oversized personal messages are split into chunked_message
        envelopes the same way broadcasts are. Personal messages don't carry
        seq, so on reconnect a partially-delivered chunk group is dropped by
        the client (no resume), but no socket teardown.

        Returns True only if every chunk was delivered successfully.
        """
        chunks = self._maybe_chunk_message(message)
        for sub_message in chunks:
            if not await self._send_json(websocket, sub_message):
                return False
        return True

    async def _send_json(self, websocket: WebSocket, message: dict) -> bool:
        """Send JSON message to a single WebSocket with timeout.

        Applies a configurable send timeout (default 0.5s) to prevent slow
        clients from blocking broadcast fan-out (GAP-WS-07 quick-fix).
        Returns False on failure or timeout.

        OBS-WIRE-01 (A.3): the per-connection send is timed with
        ``time.perf_counter`` and observed into
        ``cascor_ws_broadcast_send_duration_seconds`` (catalog SLI 4.3 —
        broadcast fan-out p95 < 1 ms). Only the wait_for() span is
        timed; serialization-for-bandwidth-accounting (``_account_send``)
        is excluded so the histogram reflects pure socket-write latency.
        """
        # OBS-WIRE-01 (A.3): start the broadcast-send timer. The metric
        # ``type`` label is closed-set-by-convention — fall back to
        # ``"unknown"`` if the message has no ``type`` field, matching
        # the pattern in ``_account_send``.
        msg_type = str(message.get("type") or "unknown")
        send_start = time.perf_counter()
        try:
            await asyncio.wait_for(
                websocket.send_json(message),
                timeout=self._send_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("WebSocket send timed out after %.1fs", self._send_timeout_seconds)
            with self._seq_lock:
                self._send_failures += 1
            # OBS-WIRE-02 (3.6): increment broadcast-timeout counter
            # by message ``type`` (closed-by-convention). Paired with
            # SLI 4.3 fan-out p95.
            try:
                from api.observability import ws_inc_broadcast_timeout

                ws_inc_broadcast_timeout(msg_type)
            except Exception:
                logger.debug("ws_inc_broadcast_timeout emission failed", exc_info=True)
            return False
        except Exception:
            with self._seq_lock:
                self._send_failures += 1
            return False
        finally:
            # OBS-WIRE-01 (A.3): observe the broadcast-send histogram
            # in ``finally`` so timeouts and exceptions still produce a
            # sample (the slow / failed sends are exactly what we care
            # about for SLI 4.3). Defensive try/except: prometheus_client
            # may not be importable in some test environments.
            try:
                from api.observability import _ensure_ws_metrics

                _ensure_ws_metrics()["broadcast_send_duration_seconds"].labels(type=msg_type).observe(time.perf_counter() - send_start)
            except Exception:
                logger.debug("broadcast_send_duration emission failed", exc_info=True)
        # GAP-WS-16: account bytes after a successful send. We re-serialize
        # to size the payload because Starlette's send_json hides the wire
        # bytes from us. The double-encode is cheap; if size estimation
        # itself fails we record the message but with byte_size=0 so the
        # message-count counter stays consistent.
        try:
            byte_size = len(json.dumps(message, default=str))
        except (TypeError, ValueError):
            byte_size = 0
        self._account_send(message, byte_size)
        return True

    def _account_send(self, message: dict, byte_size: int) -> None:
        """GAP-WS-16: record a successful WS send for bandwidth telemetry."""
        msg_type = str(message.get("type") or "unknown")
        with self._seq_lock:
            self._bytes_sent_total += byte_size
            self._messages_sent_total += 1
            self._messages_sent_by_type[msg_type] = self._messages_sent_by_type.get(msg_type, 0) + 1
            self._bytes_sent_by_type[msg_type] = self._bytes_sent_by_type.get(msg_type, 0) + byte_size

    def transport_stats(self) -> Dict[str, Any]:
        """GAP-WS-16: snapshot of cumulative WS transport counters.

        Surfaced via ``GET /v1/metrics/transport`` to validate the bandwidth
        delta from REST polling (P0 motivator) once GAP-WS-16 lands. All
        counters are cumulative since process start.

        GAP-WS-18: also exposes ``messages_chunked_total`` (number of logical
        messages that exceeded the size threshold and were split) and
        ``chunks_emitted_total`` (total chunk envelopes emitted), so we can
        see how often the chunker is firing in production.
        """
        with self._seq_lock:
            return {
                "bytes_sent_total": self._bytes_sent_total,
                "messages_sent_total": self._messages_sent_total,
                "send_failures": self._send_failures,
                "messages_sent_by_type": dict(self._messages_sent_by_type),
                "bytes_sent_by_type": dict(self._bytes_sent_by_type),
                "uptime_seconds": time.monotonic() - self._server_start_time,
                "active_connections": len(self._active_connections),
                "pending_connections": len(self._pending_connections),
                "current_seq": self._next_seq - 1,
                "replay_buffer_size": len(self._replay_buffer),
                "replay_buffer_capacity": self._replay_buffer_max_size,
                "messages_chunked_total": self._messages_chunked_total,
                "chunks_emitted_total": self._chunks_emitted_total,
                "max_message_size_bytes": self._max_message_size_bytes,
                "chunk_payload_size_bytes": self._chunk_payload_size_bytes,
            }

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
            self._per_ip_counts.clear()
            self._connection_endpoint.clear()
            for bucket in self._endpoint_connections.values():
                bucket.clear()

        for endpoint in self._endpoint_connections:
            self._emit_endpoint_gauge(endpoint)

        for ws in snapshot:
            with contextlib.suppress(Exception):
                await ws.close(code=1001, reason="Server shutting down")
        logger.info("All WebSocket connections closed")
