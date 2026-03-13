#!/usr/bin/env python
"""
Coverage tests for api/websocket/control_stream.py — targets uncovered lines
to bring coverage from ~89% to ≥90%.

Covers:
- Authentication check (lines 30-33): invalid API key closes WebSocket
- Message too large (lines 50-51): oversized message triggers error
- Lifecycle unavailable (lines 66-67): None lifecycle triggers error
- Unhandled command (line 102): raises ValueError
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.websocket.control_stream import _execute_command, control_stream_handler


class TestControlStreamAuth:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_auth_enabled_invalid_key_closes_connection(self):
        """WebSocket closed with 4001 when auth enabled and key invalid."""
        ws = AsyncMock()
        ws.headers = {"X-API-Key": "bad-key"}

        auth = MagicMock()
        auth.enabled = True
        auth.validate.return_value = False

        app_state = MagicMock()
        app_state.api_key_auth = auth

        ws.app.state = app_state

        await control_stream_handler(ws)

        ws.close.assert_called_once_with(code=4001, reason="Authentication required")
        ws.accept.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_auth_enabled_missing_key_closes_connection(self):
        """WebSocket closed when auth enabled and no key provided."""
        ws = AsyncMock()
        ws.headers = {}

        auth = MagicMock()
        auth.enabled = True
        auth.validate.return_value = False

        app_state = MagicMock()
        app_state.api_key_auth = auth

        ws.app.state = app_state

        await control_stream_handler(ws)

        ws.close.assert_called_once_with(code=4001, reason="Authentication required")


class TestControlStreamMessageSize:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_message_too_large(self):
        """Oversized message triggers error response."""
        from fastapi import WebSocketDisconnect

        ws = AsyncMock()
        app_state = MagicMock()
        app_state.api_key_auth = None
        app_state.lifecycle = MagicMock()
        ws.app.state = app_state

        # First call returns oversized message, second raises disconnect
        large_msg = "x" * 70000
        ws.receive_text.side_effect = [large_msg, WebSocketDisconnect(code=1000)]

        await control_stream_handler(ws)

        # Verify error was sent
        ws.send_json.assert_any_call({
            "type": "connection_established",
            "data": {"channel": "control"},
        })
        # Check that an error about size was sent
        calls = ws.send_json.call_args_list
        assert any("too large" in str(c).lower() or "Message too large" in str(c) for c in calls)


class TestControlStreamLifecycleUnavailable:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_lifecycle_none_returns_error(self):
        """Valid command with None lifecycle returns error."""
        from fastapi import WebSocketDisconnect

        ws = AsyncMock()
        app_state = MagicMock()
        app_state.api_key_auth = None
        app_state.lifecycle = None
        ws.app.state = app_state

        ws.receive_text.side_effect = [
            json.dumps({"command": "start"}),
            WebSocketDisconnect(code=1000),
        ]

        await control_stream_handler(ws)

        calls = ws.send_json.call_args_list
        assert any("not available" in str(c) or "Lifecycle manager not available" in str(c) for c in calls)


class TestExecuteCommandEdge:

    @pytest.mark.unit
    def test_unhandled_command_raises_value_error(self):
        """_execute_command raises ValueError for unhandled command."""
        lifecycle = MagicMock()
        with pytest.raises(ValueError, match="Unhandled command"):
            _execute_command(lifecycle, "nonexistent_command")
