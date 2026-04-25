"""Phase 1C Track 1 security remediation tests (SEC-03/11/15/17)."""

import pickle  # nosec B403 — needed to hand-craft a hostile payload for SEC-11
from unittest.mock import AsyncMock

import pytest

# =============================================================================
# SEC-03: per-IP WebSocket connection cap
# =============================================================================


@pytest.mark.unit
class TestSEC03PerIPLimit:
    """WebSocketManager must enforce a per-IP cap in addition to the global one."""

    @pytest.mark.asyncio
    async def test_per_ip_cap_rejects_overflow(self) -> None:
        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager(max_connections=50, max_connections_per_ip=2)
        connections = []
        for _ in range(2):
            ws = AsyncMock()
            ws.client = ("10.0.0.1", 12345)
            ok = await mgr.connect(ws)
            assert ok is True
            connections.append(ws)

        overflow = AsyncMock()
        overflow.client = ("10.0.0.1", 23456)
        result = await mgr.connect(overflow)
        assert result is False
        overflow.close.assert_awaited_once()
        args, kwargs = overflow.close.call_args
        assert kwargs.get("code") == 1013
        assert "per-ip" in kwargs.get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_per_ip_cap_isolated_per_source(self) -> None:
        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager(max_connections=50, max_connections_per_ip=1)

        ws_a = AsyncMock()
        ws_a.client = ("10.0.0.1", 1)
        ws_b = AsyncMock()
        ws_b.client = ("10.0.0.2", 1)

        assert await mgr.connect(ws_a) is True
        assert await mgr.connect(ws_b) is True

    @pytest.mark.asyncio
    async def test_disconnect_releases_per_ip_slot(self) -> None:
        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager(max_connections=50, max_connections_per_ip=1)

        ws1 = AsyncMock()
        ws1.client = ("10.0.0.5", 1)
        assert await mgr.connect(ws1) is True

        await mgr.disconnect(ws1)

        ws2 = AsyncMock()
        ws2.client = ("10.0.0.5", 2)
        assert await mgr.connect(ws2) is True


# =============================================================================
# SEC-11: snapshot RestrictedUnpickler
# =============================================================================


class TestSEC11SnapshotRestrictedUnpickler:
    """Snapshot deserialization must reject classes outside the allowlist."""

    def test_accepts_python_random_state(self) -> None:
        import random as _random

        from snapshots.snapshot_serializer import _snapshot_restricted_loads

        ref = _random.Random()
        ref.seed(1234)
        state_bytes = pickle.dumps(ref.getstate())
        restored = _snapshot_restricted_loads(state_bytes)
        assert isinstance(restored, tuple)
        rng = _random.Random()
        rng.setstate(restored)
        assert rng.random() == ref.random()

    def test_rejects_os_system_gadget(self) -> None:
        from snapshots.snapshot_serializer import SnapshotUnpicklingError, _snapshot_restricted_loads

        class _Gadget:
            def __reduce__(self):
                import os  # noqa: PLC0415

                return (os.system, ("true",))

        payload = pickle.dumps(_Gadget())
        # Python serializes os.system under the platform-specific module name
        # (posix on Linux, nt on Windows). Match the common ``.system`` suffix.
        with pytest.raises(SnapshotUnpicklingError, match=r"\.system"):
            _snapshot_restricted_loads(payload)

    def test_rejects_arbitrary_module(self) -> None:
        from snapshots.snapshot_serializer import SnapshotUnpicklingError, _snapshot_restricted_loads

        class _Unknown:
            def __reduce__(self):
                import subprocess  # noqa: PLC0415

                return (subprocess.run, (["true"],))

        payload = pickle.dumps(_Unknown())
        with pytest.raises(SnapshotUnpicklingError, match="subprocess"):
            _snapshot_restricted_loads(payload)


# =============================================================================
# SEC-15: Sentry send_default_pii=False + before_send filter
# =============================================================================


class TestSEC15SentryPII:
    def test_before_send_filter_redacts_sensitive_headers(self) -> None:
        from api.observability import _strip_sensitive_headers

        event = {
            "request": {
                "headers": {
                    "X-API-Key": "super-secret-key",
                    "authorization": "Bearer super-secret-token",
                    "cookie": "session=abc",
                    "User-Agent": "juniper-canopy/0.1",
                }
            }
        }
        filtered = _strip_sensitive_headers(event, hint={})
        assert filtered is event
        assert filtered["request"]["headers"]["X-API-Key"] == "[Filtered]"
        assert filtered["request"]["headers"]["authorization"] == "[Filtered]"
        assert filtered["request"]["headers"]["cookie"] == "[Filtered]"
        assert filtered["request"]["headers"]["User-Agent"] == "juniper-canopy/0.1"

    def test_before_send_filter_handles_missing_request(self) -> None:
        from api.observability import _strip_sensitive_headers

        filtered = _strip_sensitive_headers({}, hint={})
        assert filtered == {}

    def test_configure_sentry_sets_pii_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        class _FakeSentry:
            @staticmethod
            def init(**kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("api.observability.sentry_sdk", _FakeSentry, raising=False)
        # Inject the fake into the function-local import by patching sys.modules
        import sys

        monkeypatch.setitem(sys.modules, "sentry_sdk", _FakeSentry)

        from api.observability import configure_sentry

        configure_sentry("https://public@example.com/1", "juniper-cascor", "0.0.0")
        assert captured.get("send_default_pii") is False
        assert callable(captured.get("before_send"))


# =============================================================================
# SEC-17: snapshot_id traversal validation
# =============================================================================


class TestSEC17SnapshotIDValidation:
    @pytest.mark.parametrize(
        "snapshot_id",
        [
            "snapshot_20260101_120000",
            "my-snapshot-v2",
            "abc",
            "A" + "b" * 127,
        ],
    )
    def test_valid_snapshot_id_accepted(self, snapshot_id: str) -> None:
        from api.routes.snapshots import _validate_snapshot_id

        _validate_snapshot_id(snapshot_id)

    @pytest.mark.parametrize(
        "snapshot_id",
        [
            "",
            "..",
            "../etc/passwd",
            "snap/sub",
            "snap\x00",
            "snap.h5",
            "snap with space",
            "a" * 129,
        ],
    )
    def test_invalid_snapshot_id_rejected(self, snapshot_id: str) -> None:
        from fastapi import HTTPException

        from api.routes.snapshots import _validate_snapshot_id

        with pytest.raises(HTTPException) as exc_info:
            _validate_snapshot_id(snapshot_id)
        assert exc_info.value.status_code == 400
