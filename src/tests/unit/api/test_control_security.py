"""Tests for Phase B-pre-b control-path security: origin validation, rate limiting, cooldown."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.websocket.control_security import HandshakeCooldown, LeakyBucket, validate_control_origin


@pytest.mark.unit
class TestControlOriginValidation:
    """Test origin validation for /ws/control (§S8)."""

    def _make_ws(self, origin=None):
        ws = AsyncMock()
        headers = {}
        if origin is not None:
            headers["origin"] = origin
        ws.headers = headers
        return ws

    def test_allowed_origin_accepted(self):
        ws = self._make_ws("http://localhost:8050")
        assert validate_control_origin(ws, ["http://localhost:8050"]) is True

    def test_disallowed_origin_rejected(self):
        ws = self._make_ws("http://evil.example.com")
        assert validate_control_origin(ws, ["http://localhost:8050"]) is False

    def test_missing_origin_rejected_fail_closed(self):
        ws = self._make_ws(origin=None)
        assert validate_control_origin(ws, ["http://localhost:8050"]) is False

    def test_empty_allowlist_rejects_all(self):
        ws = self._make_ws("http://localhost:8050")
        assert validate_control_origin(ws, []) is False

    def test_case_insensitive(self):
        ws = self._make_ws("HTTP://LOCALHOST:8050")
        assert validate_control_origin(ws, ["http://localhost:8050"]) is True

    def test_trailing_slash_ignored(self):
        ws = self._make_ws("http://localhost:8050/")
        assert validate_control_origin(ws, ["http://localhost:8050"]) is True

    def test_port_significant(self):
        ws = self._make_ws("http://localhost:9999")
        assert validate_control_origin(ws, ["http://localhost:8050"]) is False


@pytest.mark.unit
class TestLeakyBucket:
    """Test per-connection leaky bucket rate limiter."""

    def test_allows_up_to_capacity(self):
        bucket = LeakyBucket(capacity=3, refill_rate=3.0)
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is True

    def test_rejects_after_capacity_exhausted(self):
        bucket = LeakyBucket(capacity=2, refill_rate=2.0)
        bucket.try_acquire()
        bucket.try_acquire()
        assert bucket.try_acquire() is False

    def test_refills_over_time(self):
        bucket = LeakyBucket(capacity=1, refill_rate=100.0)
        bucket.try_acquire()  # exhaust
        assert bucket.try_acquire() is False
        time.sleep(0.02)  # 20ms → should refill at 100/s
        assert bucket.try_acquire() is True

    def test_retry_after_positive_when_empty(self):
        bucket = LeakyBucket(capacity=1, refill_rate=10.0)
        bucket.try_acquire()
        retry = bucket.retry_after
        assert retry > 0

    def test_retry_after_zero_when_tokens_available(self):
        bucket = LeakyBucket(capacity=5, refill_rate=5.0)
        assert bucket.retry_after == 0.0

    def test_response_not_close_connection(self):
        """Rate limit sends response but does NOT close the connection (§S8)."""
        bucket = LeakyBucket(capacity=1, refill_rate=1.0)
        bucket.try_acquire()
        # After exhaustion, try_acquire returns False but bucket still exists
        assert bucket.try_acquire() is False
        # Connection stays up — we just need to verify the bucket doesn't raise
        time.sleep(1.1)
        assert bucket.try_acquire() is True


@pytest.mark.unit
class TestHandshakeCooldown:
    """Test per-origin handshake cooldown."""

    def test_not_blocked_initially(self):
        cd = HandshakeCooldown(max_rejections=3, window_sec=60, block_sec=10)
        assert cd.is_blocked("1.2.3.4") is False

    def test_not_blocked_below_threshold(self):
        cd = HandshakeCooldown(max_rejections=3, window_sec=60, block_sec=10)
        cd.record_rejection("1.2.3.4")
        cd.record_rejection("1.2.3.4")
        assert cd.is_blocked("1.2.3.4") is False

    def test_blocked_at_threshold(self):
        cd = HandshakeCooldown(max_rejections=3, window_sec=60, block_sec=10)
        cd.record_rejection("1.2.3.4")
        cd.record_rejection("1.2.3.4")
        blocked = cd.record_rejection("1.2.3.4")  # 3rd rejection
        assert blocked is True
        assert cd.is_blocked("1.2.3.4") is True

    def test_block_expires(self):
        cd = HandshakeCooldown(max_rejections=1, window_sec=60, block_sec=0.05)
        cd.record_rejection("1.2.3.4")
        assert cd.is_blocked("1.2.3.4") is True
        time.sleep(0.06)
        assert cd.is_blocked("1.2.3.4") is False

    def test_different_ips_independent(self):
        cd = HandshakeCooldown(max_rejections=2, window_sec=60, block_sec=10)
        cd.record_rejection("1.1.1.1")
        cd.record_rejection("1.1.1.1")  # blocked
        assert cd.is_blocked("1.1.1.1") is True
        assert cd.is_blocked("2.2.2.2") is False

    def test_get_block_remaining_when_blocked(self):
        cd = HandshakeCooldown(max_rejections=1, window_sec=60, block_sec=300)
        cd.record_rejection("1.2.3.4")
        remaining = cd.get_block_remaining("1.2.3.4")
        assert remaining is not None
        assert remaining > 290  # ~300 seconds remaining

    def test_get_block_remaining_when_not_blocked(self):
        cd = HandshakeCooldown(max_rejections=10, window_sec=60, block_sec=300)
        assert cd.get_block_remaining("1.2.3.4") is None

    def test_old_rejections_pruned(self):
        cd = HandshakeCooldown(max_rejections=3, window_sec=0.05, block_sec=10)
        cd.record_rejection("1.2.3.4")
        cd.record_rejection("1.2.3.4")
        time.sleep(0.06)  # window expires
        # Third rejection is now the first in a fresh window
        blocked = cd.record_rejection("1.2.3.4")
        assert blocked is False


@pytest.mark.unit
class TestHandshakeCooldownBugCC14Cleanup:
    """BUG-CC-14: HandshakeCooldown._rejections must be pruned for non-blocked IPs."""

    def test_stale_non_blocked_ips_pruned_after_threshold(self):
        """After CLEANUP_EVERY_N record_rejection calls, stale non-blocked IPs are dropped."""
        cd = HandshakeCooldown(max_rejections=1000, window_sec=60, block_sec=300)
        cd._CLEANUP_EVERY_N = 10
        # Seed 50 non-blocked IPs each with a single rejection.
        for i in range(50):
            cd.record_rejection(f"10.0.0.{i}")
        # All present so far (cleanup not yet triggered to prune them since they are fresh).
        assert len(cd._rejections) == 50

        # Push their timestamps far into the past (beyond 2 * window_sec).
        with cd._lock:
            for ip in list(cd._rejections):
                cd._rejections[ip] = [time.monotonic() - (2 * cd._window_sec) - 5]

        # Drive cleanup using a single different IP so its history stays fresh.
        for _ in range(cd._CLEANUP_EVERY_N):
            cd.record_rejection("9.9.9.9")

        # The 50 stale non-blocked IPs should have been pruned.
        for i in range(50):
            assert f"10.0.0.{i}" not in cd._rejections
        assert "9.9.9.9" in cd._rejections

    def test_blocked_ips_not_in_rejections_after_block(self):
        """Sanity: a blocked IP's rejection list is cleared by record_rejection itself."""
        cd = HandshakeCooldown(max_rejections=2, window_sec=60, block_sec=300)
        cd.record_rejection("1.1.1.1")
        cd.record_rejection("1.1.1.1")  # triggers block, clears list.
        assert cd._rejections.get("1.1.1.1", []) == []
        assert cd.is_blocked("1.1.1.1") is True

    def test_fresh_non_blocked_ips_preserved(self):
        """Cleanup must not remove IPs whose timestamps are within the active window."""
        cd = HandshakeCooldown(max_rejections=100, window_sec=60, block_sec=300)
        cd._CLEANUP_EVERY_N = 5
        for i in range(20):
            cd.record_rejection(f"172.16.0.{i}")
        # All entries are fresh, so even after cleanup runs they remain.
        for i in range(20):
            assert f"172.16.0.{i}" in cd._rejections


@pytest.mark.unit
class TestControlSecuritySettings:
    """Test Phase B-pre-b settings."""

    def test_control_settings_defaults(self):
        from api.settings import Settings

        s = Settings(auto_start=False)
        assert s.ws_control_rate_limit_per_sec == 10
        assert s.ws_control_idle_timeout_sec == 120
        assert s.ws_control_cooldown_rejections == 10
        assert s.ws_control_cooldown_window_sec == 60
        assert s.ws_control_cooldown_block_sec == 300
        assert s.disable_ws_control_endpoint is False

    def test_kill_switch_default_off(self):
        from api.settings import Settings

        s = Settings(auto_start=False)
        assert s.disable_ws_control_endpoint is False

    def test_control_allowed_origins_has_localhost(self):
        from api.settings import Settings

        s = Settings(auto_start=False)
        assert "http://localhost:8050" in s.ws_control_allowed_origins
