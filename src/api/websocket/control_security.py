"""Phase B-pre-b: Control-path security primitives.

Provides:
- Origin validation for /ws/control (reuses allowlist pattern from worker_stream)
- Per-connection leaky bucket rate limiter (10 tokens, 10/s refill)
- Per-origin handshake cooldown (10 rejections in 60s → 5-min IP block)
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Optional

from fastapi import WebSocket

logger = logging.getLogger("juniper_cascor.api.websocket.control_security")


def validate_control_origin(websocket: WebSocket, allowed_origins: list[str]) -> bool:
    """Validate Origin header against allowlist for /ws/control.

    Unlike worker endpoints (which reject ALL Origins), the control endpoint
    accepts connections from allowed browser origins.

    Returns True if origin is allowed, False otherwise.
    """
    origin = websocket.headers.get("origin", "").rstrip("/").lower()
    if not origin:
        logger.info("Control WS: no Origin header — rejecting (fail-closed)")
        return False
    normalized_allowed = [o.rstrip("/").lower() for o in allowed_origins]
    if origin in normalized_allowed:
        return True
    logger.info("Control WS: origin %r not in allowlist — rejecting", origin)
    return False


class LeakyBucket:
    """Per-connection leaky bucket rate limiter.

    Allows `capacity` commands, refills at `refill_rate` per second.
    Thread-safe for use from async context.
    """

    def __init__(self, capacity: int = 10, refill_rate: float = 10.0):
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Try to consume one token. Returns True if allowed, False if rate-limited."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    @property
    def retry_after(self) -> float:
        """Estimated seconds until a token is available.

        When ``refill_rate`` is zero (or negative), tokens never refill, so
        there is no finite wait — return ``0.0`` rather than dividing by zero.
        Callers still see ``try_acquire() is False`` and can surface a
        rate-limit ack; a zero ``retry_after`` signals "no estimated wait".
        """
        with self._lock:
            if self._tokens >= 1.0:
                return 0.0
            if self._refill_rate <= 0:
                return 0.0
            deficit = 1.0 - self._tokens
            return round(deficit / self._refill_rate, 1)


class HandshakeCooldown:
    """Per-origin handshake cooldown tracker.

    Tracks rejected handshakes per client IP. When `max_rejections` are
    exceeded within `window_sec`, the IP is blocked for `block_sec`.
    Cleared on server restart (NAT-hostile escape hatch).
    """

    # BUG-CC-14: prune stale rejection histories every N record_rejection calls.
    _CLEANUP_EVERY_N = 50

    def __init__(self, max_rejections: int = 10, window_sec: int = 60, block_sec: int = 300):
        self._max_rejections = max_rejections
        self._window_sec = window_sec
        self._block_sec = block_sec
        self._rejections: dict[str, list[float]] = defaultdict(list)
        self._blocked_until: dict[str, float] = {}
        self._lock = threading.Lock()
        self._total_rejections_since_cleanup = 0  # BUG-CC-14: cleanup tick counter.

    def _maybe_full_cleanup(self) -> None:
        """BUG-CC-14: prune ALL stale rejection entries across all IPs. Caller must hold ``_lock``."""
        now = time.monotonic()
        cutoff = now - (2 * self._window_sec)
        stale_ips = [ip for ip, timestamps in self._rejections.items() if not timestamps or all(t < cutoff for t in timestamps)]
        for ip in stale_ips:
            del self._rejections[ip]
        if stale_ips:
            logger.debug("HandshakeCooldown: pruned rejection history for %d stale IPs", len(stale_ips))

    def is_blocked(self, client_ip: str) -> bool:
        """Check if an IP is currently blocked."""
        with self._lock:
            blocked_until = self._blocked_until.get(client_ip)
            if blocked_until is not None:
                if time.monotonic() < blocked_until:
                    return True
                del self._blocked_until[client_ip]
            return False

    def record_rejection(self, client_ip: str) -> bool:
        """Record a handshake rejection. Returns True if IP is now blocked."""
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._window_sec
            timestamps = self._rejections[client_ip]
            # Prune old entries
            self._rejections[client_ip] = [t for t in timestamps if t > cutoff]
            self._rejections[client_ip].append(now)

            # BUG-CC-14: periodically prune stale histories for non-blocked IPs.
            self._total_rejections_since_cleanup += 1
            if self._total_rejections_since_cleanup >= self._CLEANUP_EVERY_N:
                self._maybe_full_cleanup()
                self._total_rejections_since_cleanup = 0

            if len(self._rejections[client_ip]) >= self._max_rejections:
                self._blocked_until[client_ip] = now + self._block_sec
                self._rejections[client_ip] = []
                logger.warning("Control WS: IP %s blocked for %ds (cooldown triggered)", client_ip, self._block_sec)
                return True
            return False

    def get_block_remaining(self, client_ip: str) -> Optional[float]:
        """Get remaining block time in seconds, or None if not blocked."""
        with self._lock:
            blocked_until = self._blocked_until.get(client_ip)
            if blocked_until is not None:
                remaining = blocked_until - time.monotonic()
                if remaining > 0:
                    return round(remaining, 1)
                del self._blocked_until[client_ip]
            return None
