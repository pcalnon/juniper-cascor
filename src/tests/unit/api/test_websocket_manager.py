"""Tests for WebSocket connection manager."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.websocket.manager import WebSocketManager


@pytest.mark.unit
class TestWebSocketManager:
    """Test WebSocket connection manager."""

    def test_init_defaults(self):
        """Manager initializes with defaults."""
        mgr = WebSocketManager()
        assert mgr.connection_count == 0
        assert mgr._max_connections == 50

    def test_init_custom_max(self):
        """Manager respects custom max_connections."""
        mgr = WebSocketManager(max_connections=5)
        assert mgr._max_connections == 5

    @pytest.mark.asyncio
    async def test_connect(self):
        """Connect accepts and registers a WebSocket."""
        mgr = WebSocketManager()
        ws = AsyncMock()

        result = await mgr.connect(ws)

        assert result is True
        assert mgr.connection_count == 1
        ws.accept.assert_awaited_once()
        ws.send_json.assert_awaited_once()
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "connection_established"

    @pytest.mark.asyncio
    async def test_connect_max_reached(self):
        """Connect rejects when max connections reached."""
        mgr = WebSocketManager(max_connections=1)
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        await mgr.connect(ws1)
        result = await mgr.connect(ws2)

        assert result is False
        assert mgr.connection_count == 1
        ws2.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Disconnect removes a connection."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)
        assert mgr.connection_count == 1

        await mgr.disconnect(ws)
        assert mgr.connection_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_unknown(self):
        """Disconnect of unknown connection doesn't error."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.disconnect(ws)
        assert mgr.connection_count == 0

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Broadcast sends to all connected clients."""
        mgr = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await mgr.connect(ws1)
        await mgr.connect(ws2)

        message = {"type": "test", "data": {}}
        await mgr.broadcast(message)

        # Each WS received connection_established + broadcast
        assert ws1.send_json.await_count == 2
        assert ws2.send_json.await_count == 2

    @pytest.mark.asyncio
    async def test_broadcast_removes_failed(self):
        """Broadcast removes connections that fail to send."""
        mgr = WebSocketManager()
        ws_good = AsyncMock()
        ws_bad = AsyncMock()
        ws_bad.send_json.side_effect = Exception("Connection closed")

        await mgr.connect(ws_good)
        # Manually add ws_bad since connect sends a message which would fail
        mgr._active_connections.add(ws_bad)

        await mgr.broadcast({"type": "test"})

        assert ws_bad not in mgr._active_connections
        assert mgr.connection_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_empty(self):
        """Broadcast with no connections does nothing."""
        mgr = WebSocketManager()
        await mgr.broadcast({"type": "test"})  # Should not raise

    def test_broadcast_from_thread_no_loop(self):
        """broadcast_from_thread without event loop doesn't error."""
        mgr = WebSocketManager()
        mgr.broadcast_from_thread({"type": "test"})  # Should not raise

    def test_broadcast_from_thread_with_loop(self):
        """broadcast_from_thread submits coroutine to event loop."""
        mgr = WebSocketManager()
        loop = MagicMock()
        loop.is_closed.return_value = False
        mgr.set_event_loop(loop)

        with patch("api.websocket.manager.asyncio.run_coroutine_threadsafe") as mock_submit:
            mgr.broadcast_from_thread({"type": "test"})
            mock_submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_personal_message(self):
        """send_personal_message sends to specific client."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)

        result = await mgr.send_personal_message(ws, {"type": "personal"})
        assert result is True

    @pytest.mark.asyncio
    async def test_send_personal_message_failure(self):
        """send_personal_message returns False on failure."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        ws.send_json.side_effect = Exception("fail")

        result = await mgr.send_personal_message(ws, {"type": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_close_all(self):
        """close_all closes all connections."""
        mgr = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await mgr.connect(ws1)
        await mgr.connect(ws2)

        await mgr.close_all()

        assert mgr.connection_count == 0
        ws1.close.assert_awaited_once()
        ws2.close.assert_awaited_once()

    def test_set_event_loop(self):
        """set_event_loop stores loop reference."""
        mgr = WebSocketManager()
        loop = MagicMock()
        mgr.set_event_loop(loop)
        assert mgr._event_loop is loop
