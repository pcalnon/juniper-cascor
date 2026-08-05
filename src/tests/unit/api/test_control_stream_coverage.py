#!/usr/bin/env python
"""
Coverage tests for api/websocket/control_stream.py — targets uncovered lines
to bring coverage from ~89% to ≥90%.

Covers:
- Authentication check (lines 30-33): invalid API key closes WebSocket
- Message too large (lines 50-51): oversized message triggers error
- Lifecycle unavailable (lines 66-67): None lifecycle triggers error
- Unhandled command (line 102): raises ValueError

Phase C-5 (per-file coverage rollout, PR-2 — websocket layer) extends this
file with the remaining uncovered branches: the handshake gates
(kill-switch / cooldown / origin), the leaky-bucket rate-limited arm, the
``ValidationError`` (invalid-params) arm, the control ping loop failure
paths, the idle-timeout recv path, and the defensive metric-emission guards.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect

import api.websocket.control_stream as cs
from api.websocket.control_security import LeakyBucket
from api.websocket.control_stream import (
    _check_handshake_gates,
    _control_ping_loop,
    _control_recv_loop,
    _execute_command,
    _get_client_ip,
    _get_command_counter,
    _handle_command_message,
    control_stream_handler,
)


class TestControlStreamAuth:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_auth_enabled_invalid_key_closes_connection(self):
        """WebSocket closed with 4001 when auth enabled and key invalid."""
        ws = AsyncMock()
        ws.headers = {"X-API-Key": "bad-key", "origin": "http://localhost:8050"}
        ws.client = ("127.0.0.1", 12345)

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
        ws.headers = {"origin": "http://localhost:8050"}
        ws.client = ("127.0.0.1", 12345)

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
        ws.headers = {"origin": "http://localhost:8050"}
        ws.client = ("127.0.0.1", 12345)
        app_state = MagicMock()
        app_state.api_key_auth = None
        app_state.lifecycle = MagicMock()
        # SEC-F19 D4: handler reserves an admission slot via ws_manager before
        # accepting; provide awaitable doubles so the mock-based test passes
        # through the admission path.
        app_state.ws_manager.try_admit = AsyncMock(return_value=True)
        app_state.ws_manager.release_admission = AsyncMock()
        ws.app.state = app_state

        # First call returns oversized message, second raises disconnect
        large_msg = "x" * 70000
        ws.receive_text.side_effect = [large_msg, WebSocketDisconnect(code=1000)]

        await control_stream_handler(ws)

        # Verify error was sent
        ws.send_json.assert_any_call(
            {
                "type": "connection_established",
                "data": {"channel": "control"},
            }
        )
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
        ws.headers = {"origin": "http://localhost:8050"}
        ws.client = ("127.0.0.1", 12345)
        app_state = MagicMock()
        app_state.api_key_auth = None
        app_state.lifecycle = None
        # SEC-F19 D4: handler reserves an admission slot via ws_manager before
        # accepting; provide awaitable doubles so the mock-based test passes
        # through the admission path.
        app_state.ws_manager.try_admit = AsyncMock(return_value=True)
        app_state.ws_manager.release_admission = AsyncMock()
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


@pytest.mark.unit
class TestGetCommandCounter:
    """``_get_command_counter`` lazy init + import-unavailable sentinel."""

    def test_import_unavailable_sets_false_sentinel(self):
        """When juniper_observability is unavailable, the counter caches ``False`` (lines 82-83)."""
        original = cs._command_received_counter
        try:
            cs._command_received_counter = None  # reset lazy cache
            with patch.dict("sys.modules", {"juniper_observability": None}):
                result = _get_command_counter()
            assert result is False
        finally:
            cs._command_received_counter = original


@pytest.mark.unit
class TestGetClientIp:
    """``_get_client_ip`` fallback."""

    def test_missing_client_returns_unknown(self):
        """A WebSocket with no ``client`` yields the ``unknown`` sentinel (line 108)."""
        ws = MagicMock()
        ws.client = None
        assert _get_client_ip(ws) == "unknown"


@pytest.mark.unit
class TestHandshakeGates:
    """``_check_handshake_gates`` pre-accept rejection paths."""

    @pytest.mark.asyncio
    async def test_kill_switch_disabled_endpoint(self):
        """The kill-switch closes with 1013 (lines 114-115)."""
        ws = AsyncMock()
        settings = MagicMock()
        settings.disable_ws_control_endpoint = True

        allowed = await _check_handshake_gates(ws, settings, "127.0.0.1")

        assert allowed is False
        ws.close.assert_awaited_once_with(code=1013, reason="Control endpoint disabled")

    @pytest.mark.asyncio
    async def test_cooldown_blocked_ip(self):
        """A cooldown-blocked IP closes with 4029 (lines 119-122)."""
        ws = AsyncMock()
        settings = MagicMock()
        settings.disable_ws_control_endpoint = False

        cooldown = MagicMock()
        cooldown.is_blocked.return_value = True
        cooldown.get_block_remaining.return_value = 42.0

        with patch("api.websocket.control_stream._get_cooldown", return_value=cooldown):
            allowed = await _check_handshake_gates(ws, settings, "10.0.0.9")

        assert allowed is False
        ws.close.assert_awaited_once_with(code=4029, reason="Too many rejected handshakes")

    @pytest.mark.asyncio
    async def test_origin_not_allowed_records_rejection(self):
        """A disallowed Origin closes with 4003 and records a rejection (lines 130-132)."""
        ws = AsyncMock()
        ws.app.state.api_key_auth = None  # auth disabled → ws_authenticate True
        settings = MagicMock()
        settings.disable_ws_control_endpoint = False
        settings.ws_control_allowed_origins = ["http://localhost:8050"]

        cooldown = MagicMock()
        cooldown.is_blocked.return_value = False

        with patch("api.websocket.control_stream._get_cooldown", return_value=cooldown), patch("api.websocket.control_stream.validate_control_origin", return_value=False):
            allowed = await _check_handshake_gates(ws, settings, "10.0.0.9")

        assert allowed is False
        cooldown.record_rejection.assert_called_once_with("10.0.0.9")
        ws.close.assert_awaited_once_with(code=4003, reason="Origin not allowed")

    @pytest.mark.asyncio
    async def test_empty_origin_allowlist_skips_origin_gate(self):
        """Empty ``ws_control_allowed_origins`` opts out of the CSWSH Origin check.

        Settings document ``[]`` as an intentional opt-out. The handshake gate
        skips ``validate_control_origin`` when the allowlist is empty — pin that
        contract so a future "fail-closed by default" refactor cannot silently
        change operator-facing behavior without a failing test. Note the helper
        ``validate_control_origin([], …)`` itself is fail-closed; only the gate
        skip path (not the helper) implements the opt-out.
        """
        ws = AsyncMock()
        ws.app.state.api_key_auth = None  # auth disabled → ws_authenticate True
        settings = MagicMock()
        settings.disable_ws_control_endpoint = False
        settings.ws_control_allowed_origins = []

        cooldown = MagicMock()
        cooldown.is_blocked.return_value = False

        with (
            patch("api.websocket.control_stream._get_cooldown", return_value=cooldown),
            patch("api.websocket.control_stream.validate_control_origin") as validate_origin,
        ):
            allowed = await _check_handshake_gates(ws, settings, "10.0.0.9")

        assert allowed is True
        validate_origin.assert_not_called()
        cooldown.record_rejection.assert_not_called()
        ws.close.assert_not_awaited()


@pytest.mark.unit
class TestHandleCommandMessageBranches:
    """``_handle_command_message`` rate-limit / validation / emission arms."""

    @pytest.mark.asyncio
    async def test_rate_limited_command(self):
        """An exhausted bucket returns a rate_limited response (lines 156-167)."""
        ws = AsyncMock()
        bucket = MagicMock()
        bucket.try_acquire.return_value = False
        bucket.retry_after = 0.5

        await _handle_command_message(ws, MagicMock(), {"command": "stop", "command_id": "rl-1"}, bucket)

        sent = ws.send_json.await_args[0][0]
        assert sent["status"] == "rate_limited"
        assert sent["retry_after"] == 0.5
        assert sent["command_id"] == "rl-1"

    @pytest.mark.asyncio
    async def test_invalid_params_returns_invalid_params_ack(self):
        """A set_params payload violating the shared bounds returns invalid_params (lines 218-221)."""
        ws = AsyncMock()
        bucket = LeakyBucket(capacity=10, refill_rate=10.0)
        lifecycle = MagicMock()

        # ``learning_rate`` has ``gt=0`` — a negative value fails validation
        # inside ``_execute_command`` before ``update_params`` is reached.
        await _handle_command_message(ws, lifecycle, {"command": "set_params", "params": {"learning_rate": -1.0}, "command_id": "bad-1"}, bucket)

        sent = ws.send_json.await_args[0][0]
        assert sent["type"] == "command_response"
        assert sent["data"]["status"] == "error"
        assert sent["data"]["code"] == "invalid_params"
        lifecycle.update_params.assert_not_called()

    @pytest.mark.asyncio
    async def test_emit_response_guard_swallows_exception(self):
        """A failing ``ws_inc_command_responses`` is swallowed (lines 152-153)."""
        ws = AsyncMock()
        bucket = LeakyBucket(capacity=10, refill_rate=10.0)

        with patch("api.observability.ws_inc_command_responses", side_effect=RuntimeError("emit down")):
            # An unknown command triggers a single response-emission attempt.
            await _handle_command_message(ws, MagicMock(), {"command": "not-a-command"}, bucket)

        sent = ws.send_json.await_args[0][0]
        assert sent["data"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_command_handler_observe_guard_swallows_exception(self):
        """A failing ``ws_observe_command_handler`` is swallowed (lines 235-236)."""
        ws = AsyncMock()
        bucket = LeakyBucket(capacity=10, refill_rate=10.0)
        lifecycle = MagicMock()
        lifecycle.stop_training.return_value = {"state": "idle"}

        with patch("api.observability.ws_observe_command_handler", side_effect=RuntimeError("emit down")):
            await _handle_command_message(ws, lifecycle, {"command": "stop"}, bucket)

        sent = ws.send_json.await_args[0][0]
        assert sent["data"]["status"] == "success"


@pytest.mark.unit
class TestControlPingLoop:
    """``_control_ping_loop`` failure paths."""

    @pytest.mark.asyncio
    async def test_ping_send_failure_returns(self):
        """A failed ping send ends the loop (lines 246-247)."""
        ws = AsyncMock()
        ws.send_json = AsyncMock(side_effect=RuntimeError("closed"))
        pong_received = asyncio.Event()

        await asyncio.wait_for(
            _control_ping_loop(ws, "127.0.0.1", hb_interval=0, hb_timeout=0.01, pong_received=pong_received),
            timeout=2.0,
        )
        ws.send_json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_failure_after_pong_timeout_is_swallowed(self):
        """A close() failure after pong timeout is swallowed (lines 254-255)."""
        ws = AsyncMock()
        ws.send_json = AsyncMock(return_value=None)
        ws.close = AsyncMock(side_effect=RuntimeError("already closed"))
        pong_received = asyncio.Event()  # never set → pong wait times out

        await asyncio.wait_for(
            _control_ping_loop(ws, "127.0.0.1", hb_interval=0, hb_timeout=0.01, pong_received=pong_received),
            timeout=2.0,
        )
        ws.close.assert_awaited_once()
        # C3: 1011, not 1006 — RFC 6455 §7.4.1 forbids sending 1006 on the
        # wire (the websockets server impl rejects it, so the pre-C3 close
        # frame never reached a real peer).
        assert ws.close.await_args.kwargs["code"] == 1011


@pytest.mark.unit
class TestControlRecvLoop:
    """``_control_recv_loop`` idle-timeout and no-timeout receive paths."""

    @pytest.mark.asyncio
    async def test_idle_timeout_closes_connection(self):
        """A receive that times out closes with 1000 (lines 275-277)."""
        ws = AsyncMock()
        ws.receive_text = AsyncMock(side_effect=asyncio.TimeoutError())
        bucket = LeakyBucket()
        pong_received = asyncio.Event()

        await asyncio.wait_for(
            _control_recv_loop(ws, MagicMock(), bucket, pong_received, idle_timeout=5, client_ip="127.0.0.1"),
            timeout=2.0,
        )
        ws.close.assert_awaited_once_with(code=1000, reason="Idle timeout")

    @pytest.mark.asyncio
    async def test_no_idle_timeout_uses_plain_receive(self):
        """With idle_timeout falsy the loop uses a plain receive (line 273)."""
        ws = AsyncMock()
        # A malformed frame drives the loop through the plain-receive branch
        # and out via the JSONDecodeError return path.
        ws.receive_text = AsyncMock(side_effect=["not json"])
        bucket = LeakyBucket()
        pong_received = asyncio.Event()

        await asyncio.wait_for(
            _control_recv_loop(ws, MagicMock(), bucket, pong_received, idle_timeout=0, client_ip="127.0.0.1"),
            timeout=2.0,
        )
        ws.close.assert_awaited_once_with(code=1003, reason="Malformed JSON")
