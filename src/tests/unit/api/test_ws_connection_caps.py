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

from unittest.mock import AsyncMock, MagicMock

import pytest

from api.websocket.manager import WebSocketManager, ws_identity_key

pytestmark = pytest.mark.unit


def _close_code(ws: AsyncMock) -> int:
    """Return the ``code`` kwarg from the (single) ``ws.close`` call."""
    return ws.close.call_args.kwargs["code"]


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
