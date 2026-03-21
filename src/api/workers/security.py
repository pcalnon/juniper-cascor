"""Security hardening for the WebSocket worker subsystem.

Phase 4 components:
- mTLS enforcement for worker WebSocket connections
- JWT token lifecycle with rotation support
- Connection rate limiting per IP/worker
- Anomaly detection for suspicious training results
"""

import hashlib
import logging
import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger("juniper_cascor.api.workers.security")


# ---------------------------------------------------------------------------
# mTLS Configuration
# ---------------------------------------------------------------------------


@dataclass
class TLSConfig:
    """TLS/mTLS configuration for the worker WebSocket endpoint."""

    enabled: bool = False
    cert_file: str | None = None
    key_file: str | None = None
    ca_file: str | None = None
    require_client_cert: bool = False
    min_tls_version: str = "TLSv1.3"

    def build_ssl_context(self) -> ssl.SSLContext | None:
        """Build an SSL context for the server side.

        Returns:
            SSLContext configured for TLS/mTLS, or None if TLS is disabled.

        Raises:
            FileNotFoundError: If cert/key/CA files don't exist.
            ssl.SSLError: If cert/key are invalid.
        """
        if not self.enabled:
            return None

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        # Minimum TLS version
        if self.min_tls_version == "TLSv1.3":
            ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        else:
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2

        # CA for client cert verification (mTLS) — check before loading server cert
        if self.require_client_cert and self.ca_file:
            ca_path = Path(self.ca_file)
            if not ca_path.exists():
                raise FileNotFoundError(f"CA cert not found: {self.ca_file}")
            ctx.load_verify_locations(cafile=str(ca_path))
            ctx.verify_mode = ssl.CERT_REQUIRED
            logger.info("mTLS enabled: requiring client certificates (CA: %s)", self.ca_file)
        elif self.require_client_cert:
            ctx.verify_mode = ssl.CERT_REQUIRED
            logger.warning("mTLS: require_client_cert=True but no CA file — using system trust store")

        # Server certificate
        if self.cert_file and self.key_file:
            cert_path = Path(self.cert_file)
            key_path = Path(self.key_file)
            if not cert_path.exists():
                raise FileNotFoundError(f"TLS cert not found: {self.cert_file}")
            if not key_path.exists():
                raise FileNotFoundError(f"TLS key not found: {self.key_file}")
            ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
            logger.info("Loaded server TLS certificate: %s", self.cert_file)

        return ctx


# ---------------------------------------------------------------------------
# Connection Rate Limiter
# ---------------------------------------------------------------------------


class ConnectionRateLimiter:
    """Token-bucket rate limiter for WebSocket connection attempts.

    Tracks connection attempts per source identifier (IP address or worker_id).
    Rejects attempts that exceed the configured rate.
    """

    def __init__(
        self,
        max_connections_per_minute: int = 10,
        burst_size: int = 3,
        cleanup_interval: float = 300.0,
    ) -> None:
        self._max_rate = max_connections_per_minute
        self._burst = burst_size
        self._cleanup_interval = cleanup_interval
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = Lock()
        self._last_cleanup = time.time()

        logger.info(
            "Rate limiter initialized: %d/min, burst=%d",
            max_connections_per_minute,
            burst_size,
        )

    def allow(self, source_id: str) -> bool:
        """Check if a connection attempt should be allowed.

        Args:
            source_id: Identifier for the connection source (e.g., IP address).

        Returns:
            True if allowed, False if rate-limited.
        """
        now = time.time()
        with self._lock:
            self._maybe_cleanup(now)

            if source_id not in self._buckets:
                self._buckets[source_id] = _TokenBucket(
                    rate=self._max_rate / 60.0,
                    burst=self._burst,
                )

            bucket = self._buckets[source_id]
            allowed = bucket.consume(now)

            if not allowed:
                logger.warning("Rate limited connection from %s", source_id)
            return allowed

    def _maybe_cleanup(self, now: float) -> None:
        """Remove stale buckets to prevent memory growth."""
        if now - self._last_cleanup < self._cleanup_interval:
            return
        stale_keys = [k for k, b in self._buckets.items() if now - b.last_access > self._cleanup_interval]
        for k in stale_keys:
            del self._buckets[k]
        if stale_keys:
            logger.debug("Rate limiter cleanup: removed %d stale entries", len(stale_keys))
        self._last_cleanup = now


@dataclass
class _TokenBucket:
    """Simple token bucket for rate limiting."""

    rate: float  # tokens per second
    burst: int
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.tokens = float(self.burst)

    def consume(self, now: float) -> bool:
        """Try to consume one token. Returns True if successful."""
        elapsed = now - self.last_refill
        self.tokens = min(float(self.burst), self.tokens + elapsed * self.rate)
        self.last_refill = now
        self.last_access = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """Detects suspicious patterns in training results from remote workers.

    Monitors:
    - Correlation values that are consistently too perfect (>0.999)
    - Correlation values that never improve (always near 0.0)
    - Results that arrive suspiciously fast
    - Results with identical correlations across different candidates
    """

    def __init__(
        self,
        min_training_time: float = 0.1,
        perfect_corr_threshold: float = 0.999,
        stale_corr_threshold: float = 0.001,
        duplicate_corr_window: int = 10,
    ) -> None:
        self._min_training_time = min_training_time
        self._perfect_threshold = perfect_corr_threshold
        self._stale_threshold = stale_corr_threshold
        self._duplicate_window = duplicate_corr_window
        self._worker_history: dict[str, list[_ResultRecord]] = {}
        self._lock = Lock()

    def check_result(
        self,
        worker_id: str,
        correlation: float,
        training_duration: float,
        task_id: str,
    ) -> list[str]:
        """Check a training result for anomalies.

        Args:
            worker_id: ID of the worker that produced the result.
            correlation: The correlation value from training.
            training_duration: How long the worker took (seconds).
            task_id: The task identifier.

        Returns:
            List of anomaly descriptions (empty if clean).
        """
        anomalies = []

        # Check suspiciously fast training
        if training_duration < self._min_training_time:
            anomalies.append(f"suspiciously_fast: {training_duration:.3f}s (min={self._min_training_time}s)")

        # Check perfect correlation
        if correlation > self._perfect_threshold:
            anomalies.append(f"perfect_correlation: {correlation:.6f}")

        # Check stale (zero) correlation
        if abs(correlation) < self._stale_threshold:
            anomalies.append(f"stale_correlation: {correlation:.6f}")

        with self._lock:
            history = self._worker_history.setdefault(worker_id, [])
            history.append(
                _ResultRecord(
                    task_id=task_id,
                    correlation=correlation,
                    duration=training_duration,
                    timestamp=time.time(),
                )
            )

            # Trim history
            if len(history) > self._duplicate_window:
                history[:] = history[-self._duplicate_window :]

            # Check for duplicate correlations (possible replay attack)
            if len(history) >= 3:
                recent_corrs = [r.correlation for r in history[-self._duplicate_window :]]
                corr_hash = hashlib.sha256(str(sorted(recent_corrs)).encode()).hexdigest()[:8]
                unique = len({f"{c:.6f}" for c in recent_corrs})
                if unique == 1 and len(recent_corrs) >= 3:
                    anomalies.append(f"duplicate_correlations: {unique}/{len(recent_corrs)} unique (hash={corr_hash})")

        if anomalies:
            logger.warning(
                "ANOMALY: worker=%s task=%s anomalies=%s",
                worker_id,
                task_id,
                anomalies,
            )

        return anomalies

    def get_worker_stats(self, worker_id: str) -> dict[str, Any]:
        """Get anomaly statistics for a worker."""
        with self._lock:
            history = self._worker_history.get(worker_id, [])
            if not history:
                return {"total_results": 0}
            correlations = [r.correlation for r in history]
            durations = [r.duration for r in history]
            return {
                "total_results": len(history),
                "avg_correlation": sum(correlations) / len(correlations),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_correlation": max(correlations),
            }

    def clear_worker(self, worker_id: str) -> None:
        """Clear history for a deregistered worker."""
        with self._lock:
            self._worker_history.pop(worker_id, None)


@dataclass
class _ResultRecord:
    """A single training result record for anomaly tracking."""

    task_id: str
    correlation: float
    duration: float
    timestamp: float
