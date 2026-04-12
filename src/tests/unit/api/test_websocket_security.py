"""Tests for Phase B-pre-a WebSocket security: origin validation, per-IP caps, idle timeout."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from api.websocket.manager import WebSocketManager
from api.websocket.origin import validate_origin


@pytest.mark.unit
class TestOriginValidation:
    """Test origin allowlist validation (M-SEC-01b)."""

    def _make_ws(self, origin=None):
        """Create a mock WebSocket with configurable Origin header."""
        ws = AsyncMock()
        headers = {}
        if origin is not None:
            headers["origin"] = origin
        ws.headers = headers
        return ws

    def test_origin_allowlist_accepts_configured_origin(self):
        """Configured origin is accepted."""
        ws = self._make_ws("http://localhost:8050")
        assert validate_origin(ws, ["http://localhost:8050"]) is True

    def test_origin_allowlist_rejects_third_party(self):
        """Unknown origin is rejected."""
        ws = self._make_ws("http://evil.example.com")
        assert validate_origin(ws, ["http://localhost:8050"]) is False

    def test_origin_allowlist_rejects_missing_origin(self):
        """Missing Origin header is rejected."""
        ws = self._make_ws(origin=None)
        assert validate_origin(ws, ["http://localhost:8050"]) is False

    def test_empty_allowlist_rejects_all_fail_closed(self):
        """Empty allowlist rejects everything (fail-closed)."""
        ws = self._make_ws("http://localhost:8050")
        assert validate_origin(ws, []) is False

    def test_origin_case_insensitive(self):
        """Origin comparison is case-insensitive."""
        ws = self._make_ws("HTTP://LOCALHOST:8050")
        assert validate_origin(ws, ["http://localhost:8050"]) is True

    def test_origin_trailing_slash_ignored(self):
        """Trailing slashes don't affect comparison."""
        ws = self._make_ws("http://localhost:8050/")
        assert validate_origin(ws, ["http://localhost:8050"]) is True

    def test_origin_port_significant(self):
        """Different ports are different origins."""
        ws = self._make_ws("http://localhost:9999")
        assert validate_origin(ws, ["http://localhost:8050"]) is False

    def test_allowed_origins_wildcard_refused(self):
        """Settings validator refuses '*' in allowed_origins (C-30)."""
        from pydantic import ValidationError

        from api.settings import Settings

        with pytest.raises(ValidationError, match="Wildcard"):
            Settings(auto_start=False, ws_allowed_origins=["*"])


@pytest.mark.unit
class TestPerIpLimit:
    """Test per-IP connection caps (M-SEC-04)."""

    def _make_ws_with_ip(self, ip="127.0.0.1"):
        ws = AsyncMock()
        ws.client = (ip, 12345)
        return ws

    def test_per_ip_cap_allows_under_limit(self):
        """Connections under the per-IP limit are allowed."""
        mgr = WebSocketManager()
        ws = self._make_ws_with_ip("192.168.1.1")
        assert mgr.check_per_ip_limit(ws, max_per_ip=5) is True
        assert mgr._per_ip_counts["192.168.1.1"] == 1

    def test_per_ip_cap_enforced_6th_rejected(self):
        """6th connection from same IP is rejected (default limit=5)."""
        mgr = WebSocketManager()
        for i in range(5):
            ws = self._make_ws_with_ip("10.0.0.1")
            assert mgr.check_per_ip_limit(ws, max_per_ip=5) is True
        ws6 = self._make_ws_with_ip("10.0.0.1")
        assert mgr.check_per_ip_limit(ws6, max_per_ip=5) is False

    @pytest.mark.asyncio
    async def test_per_ip_counter_decrements_on_disconnect(self):
        """Disconnect decrements the per-IP counter."""
        mgr = WebSocketManager()
        ws = self._make_ws_with_ip("10.0.0.1")
        mgr.check_per_ip_limit(ws, max_per_ip=5)
        assert mgr._per_ip_counts.get("10.0.0.1") == 1

        # Simulate the ws being in active set
        mgr._active_connections.add(ws)
        mgr._connection_meta[ws] = {}
        await mgr.disconnect(ws)
        assert mgr._per_ip_counts.get("10.0.0.1", 0) == 0

    def test_per_ip_different_ips_independent(self):
        """Different IPs have independent counters."""
        mgr = WebSocketManager()
        ws1 = self._make_ws_with_ip("10.0.0.1")
        ws2 = self._make_ws_with_ip("10.0.0.2")
        mgr.check_per_ip_limit(ws1, max_per_ip=5)
        mgr.check_per_ip_limit(ws2, max_per_ip=5)
        assert mgr._per_ip_counts["10.0.0.1"] == 1
        assert mgr._per_ip_counts["10.0.0.2"] == 1

    @pytest.mark.asyncio
    async def test_per_ip_map_shrinks_to_zero(self):
        """IP entry is removed entirely when count reaches zero."""
        mgr = WebSocketManager()
        ws = self._make_ws_with_ip("10.0.0.1")
        mgr.check_per_ip_limit(ws, max_per_ip=5)
        mgr._active_connections.add(ws)
        mgr._connection_meta[ws] = {}
        await mgr.disconnect(ws)
        assert "10.0.0.1" not in mgr._per_ip_counts

    @pytest.mark.asyncio
    async def test_close_all_clears_per_ip(self):
        """close_all resets per-IP tracking."""
        mgr = WebSocketManager()
        ws = self._make_ws_with_ip("10.0.0.1")
        mgr.check_per_ip_limit(ws, max_per_ip=5)
        mgr._active_connections.add(ws)
        await mgr.close_all()
        assert len(mgr._per_ip_counts) == 0
