"""Tests for the SEC-F19 / D4 WebSocket connection caps.

Covers the stack-absolute GLOBAL cap (D4a — spanning training/control/worker
admissions) and the per-identity cap (D4b — /ws/control, keyed on the API-key
token hash), plus the ``ws_identity_key`` helper. These are the DoS-dampening
controls that survive Docker NAT, where the per-IP cap collapses to one shared
bridge-gateway bucket (HO-3).

Design of record: juniper-ml
``notes/JUNIPER_CANOPY_CONTROL_SURFACE_AUTH_AND_NAT_DESIGN_2026-07-03.md``
§5 Option B / §8 D4.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect

from api.settings import Settings
from api.websocket.control_stream import control_stream_handler
from api.websocket.manager import WebSocketManager, ws_identity_key
from api.websocket.worker_stream import worker_stream_handler

pytestmark = pytest.mark.unit


def _close_code(ws: AsyncMock) -> int:
    """Return the ``code`` kwarg from the (single) ``ws.close`` call."""
    return ws.close.call_args.kwargs["code"]


def _make_handler_ws(*, api_key: str | None = "principal-key") -> AsyncMock:
    """Build a WebSocket double with a minimal app.state for handler admission tests."""
    ws = AsyncMock()
    ws.headers = {"X-API-Key": api_key} if api_key is not None else {}
    ws.client = ("127.0.0.1", 12345)
    app = MagicMock()
    app.state.api_key_auth = None
    app.state.lifecycle = MagicMock()
    app.state.worker_coordinator = MagicMock()
    app.state.worker_registry = MagicMock()
    app.state.ws_manager.try_admit = AsyncMock(return_value=True)
    app.state.ws_manager.release_admission = AsyncMock()
    ws.app = app
    return ws


class TestWsCapSettings:
    """Settings expose the new caps and bind-guard attestation via env vars."""

    def test_defaults_are_fail_closed_and_bounded(self):
        settings = Settings()

        assert settings.fronting_auth_attested is False
        assert settings.ws_max_connections_global == 200
        assert settings.ws_max_connections_per_identity == 5

    def test_env_overrides_new_security_controls(self, monkeypatch):
        monkeypatch.setenv("JUNIPER_CASCOR_FRONTING_AUTH_ATTESTED", "true")
        monkeypatch.setenv("JUNIPER_CASCOR_WS_MAX_CONNECTIONS_GLOBAL", "17")
        monkeypatch.setenv("JUNIPER_CASCOR_WS_MAX_CONNECTIONS_PER_IDENTITY", "3")

        settings = Settings()

        assert settings.fronting_auth_attested is True
        assert settings.ws_max_connections_global == 17
        assert settings.ws_max_connections_per_identity == 3


@pytest.mark.unit
class TestWsIdentityKey:
    """The per-identity key derives a non-reversible digest of the API key."""

    def test_hashes_api_key(self):
        ws = MagicMock()
        ws.headers = {"X-API-Key": "super-secret-key"}
        key = ws_identity_key(ws)
        assert key is not None
        assert key != "super-secret-key"
        assert len(key) == 16  # truncated sha256 hexdigest

    def test_stable_for_same_key(self):
        ws1 = MagicMock()
        ws1.headers = {"X-API-Key": "k"}
        ws2 = MagicMock()
        ws2.headers = {"X-API-Key": "k"}
        assert ws_identity_key(ws1) == ws_identity_key(ws2)

    def test_distinct_for_distinct_keys(self):
        ws1 = MagicMock()
        ws1.headers = {"X-API-Key": "key-a"}
        ws2 = MagicMock()
        ws2.headers = {"X-API-Key": "key-b"}
        assert ws_identity_key(ws1) != ws_identity_key(ws2)

    def test_none_when_absent(self):
        ws = MagicMock()
        ws.headers = {}
        assert ws_identity_key(ws) is None


@pytest.mark.unit
class TestGlobalCap:
    """SEC-F19 D4a: the stack-absolute global cap across all WS endpoints."""

    @pytest.mark.asyncio
    async def test_global_cap_saturates_via_try_admit(self):
        mgr = WebSocketManager(max_connections_global=2)
        ws1, ws2, ws3 = AsyncMock(), AsyncMock(), AsyncMock()

        assert await mgr.try_admit(ws1, endpoint="control", identity=None) is True
        assert await mgr.try_admit(ws2, endpoint="workers", identity=None) is True
        # N+1 (the 3rd) is rejected and closed with the cap close-code.
        assert await mgr.try_admit(ws3, endpoint="control", identity=None) is False
        ws3.close.assert_awaited_once()
        assert _close_code(ws3) == 1013

    @pytest.mark.asyncio
    async def test_global_cap_spans_training_and_admission(self):
        # The global cap is stack-absolute: a /ws/training connect() and a
        # /ws/control try_admit() draw from the SAME budget.
        mgr = WebSocketManager(max_connections=100, max_connections_global=2)
        ws_train, ws_ctrl, ws_extra = AsyncMock(), AsyncMock(), AsyncMock()

        assert await mgr.connect(ws_train) is True  # training path
        assert await mgr.try_admit(ws_ctrl, endpoint="control", identity=None) is True
        # Global budget (2) is now exhausted; a further training connect is
        # rejected even though the training cap (100) has ample room.
        assert await mgr.connect(ws_extra) is False
        ws_extra.close.assert_awaited()
        assert mgr.connection_count == 1  # only ws_train is broadcast-active

    @pytest.mark.asyncio
    async def test_release_admission_frees_global_slot(self):
        mgr = WebSocketManager(max_connections_global=1)
        ws1, ws2 = AsyncMock(), AsyncMock()
        assert await mgr.try_admit(ws1, endpoint="control", identity=None) is True
        assert await mgr.try_admit(ws2, endpoint="control", identity=None) is False
        # Release ws1's slot; a subsequent admit now succeeds.
        await mgr.release_admission(identity=None)
        ws3 = AsyncMock()
        assert await mgr.try_admit(ws3, endpoint="control", identity=None) is True

    @pytest.mark.asyncio
    async def test_training_disconnect_frees_global_slot(self):
        # A training connect() reserves a global slot; disconnect() releases it,
        # proving connect/try_admit share one counter released on both paths.
        mgr = WebSocketManager(max_connections=100, max_connections_global=1)
        ws_train = AsyncMock()
        assert await mgr.connect(ws_train) is True
        ws_ctrl = AsyncMock()
        assert await mgr.try_admit(ws_ctrl, endpoint="control", identity=None) is False

        await mgr.disconnect(ws_train)
        ws_ctrl2 = AsyncMock()
        assert await mgr.try_admit(ws_ctrl2, endpoint="control", identity=None) is True

    @pytest.mark.asyncio
    async def test_training_connect_accept_failure_rolls_back_slots(self):
        # If accept() fails after reservation but before connection metadata is
        # recorded, the cap slots must still be released. Otherwise repeated
        # cancelled handshakes can exhaust the global/per-IP caps until restart.
        mgr = WebSocketManager(max_connections=100, max_connections_global=1, max_connections_per_ip=1)
        failed_ws = AsyncMock()
        failed_ws.client = ("203.0.113.10", 12345)
        failed_ws.accept.side_effect = RuntimeError("accept failed")

        with pytest.raises(RuntimeError, match="accept failed"):
            await mgr.connect(failed_ws)

        next_ws = AsyncMock()
        next_ws.client = ("203.0.113.10", 12346)
        assert await mgr.connect(next_ws) is True

    @pytest.mark.asyncio
    async def test_training_pending_accept_failure_rolls_back_slots(self):
        # Same rollback requirement for the resume/pending path used by
        # /ws/training before promotion to broadcast-active.
        mgr = WebSocketManager(max_connections=100, max_connections_global=1, max_connections_per_ip=1)
        failed_ws = AsyncMock()
        failed_ws.client = ("203.0.113.20", 12345)
        failed_ws.accept.side_effect = RuntimeError("accept failed")

        with pytest.raises(RuntimeError, match="accept failed"):
            await mgr.connect_pending(failed_ws)

        next_ws = AsyncMock()
        next_ws.client = ("203.0.113.20", 12346)
        assert await mgr.connect_pending(next_ws) is True

    @pytest.mark.asyncio
    async def test_release_never_underflows(self):
        # Over-release must not drive the counter negative (it would let the cap
        # be exceeded later). Release with nothing reserved, then admit fully.
        mgr = WebSocketManager(max_connections_global=1)
        await mgr.release_admission(identity=None)
        await mgr.release_admission(identity=None)
        ws1, ws2 = AsyncMock(), AsyncMock()
        assert await mgr.try_admit(ws1, endpoint="control", identity=None) is True
        assert await mgr.try_admit(ws2, endpoint="control", identity=None) is False


@pytest.mark.unit
class TestPerIdentityCap:
    """SEC-F19 D4b: the per-identity cap (control), independent of source IP."""

    @pytest.mark.asyncio
    async def test_per_identity_cap_rejects_over_limit(self):
        mgr = WebSocketManager(max_connections_global=100, max_connections_per_identity=2)
        ws1, ws2, ws3 = AsyncMock(), AsyncMock(), AsyncMock()
        assert await mgr.try_admit(ws1, endpoint="control", identity="principal-A") is True
        assert await mgr.try_admit(ws2, endpoint="control", identity="principal-A") is True
        assert await mgr.try_admit(ws3, endpoint="control", identity="principal-A") is False
        ws3.close.assert_awaited_once()
        assert _close_code(ws3) == 1013

    @pytest.mark.asyncio
    async def test_per_identity_fairness_across_principals(self):
        # Two principals sharing a peer IP (the HO-3 NAT scenario) each keep
        # their own allocation — the cap is keyed on identity, not on the peer.
        mgr = WebSocketManager(max_connections_global=100, max_connections_per_identity=1)
        a1, b1 = AsyncMock(), AsyncMock()
        assert await mgr.try_admit(a1, endpoint="control", identity="A") is True
        # A is at its cap...
        a2 = AsyncMock()
        assert await mgr.try_admit(a2, endpoint="control", identity="A") is False
        # ...but B, from the same shared IP, is unaffected.
        assert await mgr.try_admit(b1, endpoint="control", identity="B") is True

    @pytest.mark.asyncio
    async def test_anonymous_identity_exempt_from_per_identity(self):
        # identity=None (auth disabled / anonymous) is exempt from the
        # per-identity cap and relies on the global cap only.
        mgr = WebSocketManager(max_connections_global=100, max_connections_per_identity=1)
        for _ in range(5):
            ws = AsyncMock()
            assert await mgr.try_admit(ws, endpoint="workers", identity=None) is True

    @pytest.mark.asyncio
    async def test_release_frees_per_identity_slot(self):
        mgr = WebSocketManager(max_connections_global=100, max_connections_per_identity=1)
        ws1, ws2 = AsyncMock(), AsyncMock()
        assert await mgr.try_admit(ws1, endpoint="control", identity="A") is True
        assert await mgr.try_admit(ws2, endpoint="control", identity="A") is False
        await mgr.release_admission(identity="A")
        ws3 = AsyncMock()
        assert await mgr.try_admit(ws3, endpoint="control", identity="A") is True

    @pytest.mark.asyncio
    async def test_per_identity_rejection_releases_global_slot(self):
        # A per-identity rejection must NOT leak the global slot it briefly
        # reserved: an anonymous admit afterwards still has full global room.
        mgr = WebSocketManager(max_connections_global=2, max_connections_per_identity=1)
        a1, a2 = AsyncMock(), AsyncMock()
        assert await mgr.try_admit(a1, endpoint="control", identity="A") is True  # global 1/2
        assert await mgr.try_admit(a2, endpoint="control", identity="A") is False  # per-identity reject; global must roll back to 1/2
        # Two more distinct-identity admits should fit within global=2.
        b1 = AsyncMock()
        assert await mgr.try_admit(b1, endpoint="control", identity="B") is True  # global 2/2
        c1 = AsyncMock()
        assert await mgr.try_admit(c1, endpoint="control", identity="C") is False  # global full


@pytest.mark.unit
class TestHandlerAdmissionCaps:
    """Handlers must reserve/release caps exactly around accepted sessions."""

    @pytest.mark.asyncio
    async def test_control_handler_releases_same_identity_on_session_exception(self):
        ws = _make_handler_ws(api_key="principal-key")
        expected_identity = ws_identity_key(ws)

        with patch("api.websocket.control_stream._check_handshake_gates", new=AsyncMock(return_value=True)):
            with patch("api.websocket.control_stream._run_control_session", new=AsyncMock(side_effect=WebSocketDisconnect())):
                with pytest.raises(WebSocketDisconnect):
                    await control_stream_handler(ws)

        ws.app.state.ws_manager.try_admit.assert_awaited_once_with(ws, endpoint="control", identity=expected_identity)
        ws.app.state.ws_manager.release_admission.assert_awaited_once_with(identity=expected_identity)

    @pytest.mark.asyncio
    async def test_control_handler_does_not_release_when_admission_rejected(self):
        ws = _make_handler_ws(api_key="principal-key")
        ws.app.state.ws_manager.try_admit = AsyncMock(return_value=False)

        with patch("api.websocket.control_stream._check_handshake_gates", new=AsyncMock(return_value=True)):
            with patch("api.websocket.control_stream._run_control_session", new=AsyncMock()) as run_session:
                await control_stream_handler(ws)

        run_session.assert_not_awaited()
        ws.app.state.ws_manager.release_admission.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_worker_handler_global_cap_rejection_does_not_accept_or_release(self):
        ws = _make_handler_ws(api_key="worker-key")
        ws.headers = {}
        ws.app.state.ws_manager.try_admit = AsyncMock(return_value=False)

        await worker_stream_handler(ws)

        ws.app.state.ws_manager.try_admit.assert_awaited_once_with(ws, endpoint="workers", identity=None)
        ws.accept.assert_not_awaited()
        ws.app.state.ws_manager.release_admission.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_worker_handler_releases_global_admission_after_session(self):
        ws = _make_handler_ws(api_key="worker-key")
        ws.headers = {}

        with patch("api.websocket.worker_stream._run_worker_session", new=AsyncMock(return_value=None)) as run_session:
            await worker_stream_handler(ws)

        ws.app.state.ws_manager.try_admit.assert_awaited_once_with(ws, endpoint="workers", identity=None)
        run_session.assert_awaited_once_with(
            ws,
            ws.app.state.worker_coordinator,
            ws.app.state.worker_registry,
            ws.app.state.ws_manager,
        )
        ws.app.state.ws_manager.release_admission.assert_awaited_once_with(identity=None)
