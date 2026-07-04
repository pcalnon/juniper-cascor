"""Phase D: Per-command timeout tests for /ws/control (§S10).

Tests that _execute_command is bounded by per-command timeouts:
start=10s, stop/pause/resume=2s, set_params=1s, reset=2s.
Timeout → command_response{status:"error", error:"...timed out..."}.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect

from api.websocket.control_stream import _COMMAND_TIMEOUTS, _execute_command, control_stream_handler


def _make_ws(lifecycle=None):
    """Create a mock WebSocket with standard test setup."""
    ws = AsyncMock()
    ws.headers = {"origin": "http://localhost:8050"}
    ws.client = ("127.0.0.1", 12345)
    app_state = MagicMock()
    app_state.api_key_auth = None
    app_state.lifecycle = lifecycle if lifecycle is not None else MagicMock()
    app_state.settings = None
    ws.app.state = app_state
    return ws


def _blocking_lifecycle(hang_seconds: float = 0.5):
    """Create a lifecycle whose methods block for ``hang_seconds`` seconds.

    The sleep must be longer than the patched ``_COMMAND_TIMEOUTS`` (0.1s) so
    that ``asyncio.wait_for`` trips, but short enough to keep the test fast:
    ``asyncio.to_thread`` futures can't be forcibly cancelled, so the wait_for
    waits for the thread to finish before raising TimeoutError.
    """
    lifecycle = MagicMock()

    def hang(*args, **kwargs):
        time.sleep(hang_seconds)

    lifecycle.start_training.side_effect = hang
    lifecycle.stop_training.side_effect = hang
    lifecycle.pause_training.side_effect = hang
    lifecycle.resume_training.side_effect = hang
    lifecycle.reset.side_effect = hang
    lifecycle.update_params.side_effect = hang
    return lifecycle


# ===================================================================
# §S10 timeout value tests
# ===================================================================


@pytest.mark.unit
class TestCommandTimeoutValues:
    """Verify the per-command timeout constants match the §S10 spec."""

    def test_start_timeout_10s(self):
        assert _COMMAND_TIMEOUTS["start"] == 10.0

    def test_stop_timeout_2s(self):
        assert _COMMAND_TIMEOUTS["stop"] == 2.0

    def test_pause_timeout_2s(self):
        assert _COMMAND_TIMEOUTS["pause"] == 2.0

    def test_resume_timeout_2s(self):
        assert _COMMAND_TIMEOUTS["resume"] == 2.0

    def test_reset_timeout_2s(self):
        assert _COMMAND_TIMEOUTS["reset"] == 2.0

    def test_set_params_timeout_1s(self):
        assert _COMMAND_TIMEOUTS["set_params"] == 1.0

    def test_all_valid_commands_have_timeouts(self):
        from api.websocket.control_stream import _VALID_COMMANDS

        for cmd in _VALID_COMMANDS:
            assert cmd in _COMMAND_TIMEOUTS, f"Missing timeout for '{cmd}'"


# ===================================================================
# §S10 timeout behavior tests
# ===================================================================


@pytest.mark.unit
class TestCommandTimeoutBehavior:
    """Verify that timed-out commands produce the correct error response."""

    @pytest.mark.asyncio
    async def test_stop_command_timeout_returns_error(self):
        """A hanging stop command produces a timeout error response."""
        lifecycle = _blocking_lifecycle()
        ws = _make_ws(lifecycle)

        ws.receive_text.side_effect = [
            json.dumps({"command": "stop", "command_id": "timeout-test-1"}),
            WebSocketDisconnect(code=1000),
        ]

        with patch("api.websocket.control_stream._COMMAND_TIMEOUTS", {"stop": 0.1, "start": 0.1, "pause": 0.1, "resume": 0.1, "reset": 0.1, "set_params": 0.1}):
            await control_stream_handler(ws)

        # Find the timeout error response (skip connection_established)
        responses = [call.args[0] for call in ws.send_json.call_args_list if isinstance(call.args[0], dict) and call.args[0].get("type") == "command_response"]
        assert len(responses) == 1
        resp = responses[0]
        assert resp["data"]["command"] == "stop"
        assert resp["data"]["status"] == "error"
        assert "timed out" in resp["data"]["error"].lower()
        assert resp["data"]["command_id"] == "timeout-test-1"

    @pytest.mark.asyncio
    async def test_start_command_timeout_returns_error(self):
        """A hanging start command produces a timeout error response."""
        lifecycle = _blocking_lifecycle()
        ws = _make_ws(lifecycle)

        ws.receive_text.side_effect = [
            json.dumps({"command": "start", "command_id": "timeout-start"}),
            WebSocketDisconnect(code=1000),
        ]

        with patch("api.websocket.control_stream._COMMAND_TIMEOUTS", dict.fromkeys(_COMMAND_TIMEOUTS, 0.1)):
            await control_stream_handler(ws)

        responses = [call.args[0] for call in ws.send_json.call_args_list if isinstance(call.args[0], dict) and call.args[0].get("type") == "command_response"]
        assert len(responses) == 1
        assert responses[0]["data"]["status"] == "error"
        assert "timed out" in responses[0]["data"]["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_command_within_timeout(self):
        """A fast command completes normally within its timeout window."""
        lifecycle = MagicMock()
        lifecycle.stop_training.return_value = {"state": "idle"}
        ws = _make_ws(lifecycle)

        ws.receive_text.side_effect = [
            json.dumps({"command": "stop", "command_id": "fast-1"}),
            WebSocketDisconnect(code=1000),
        ]

        await control_stream_handler(ws)

        responses = [call.args[0] for call in ws.send_json.call_args_list if isinstance(call.args[0], dict) and call.args[0].get("type") == "command_response"]
        assert len(responses) == 1
        assert responses[0]["data"]["status"] == "success"
        assert responses[0]["data"]["command_id"] == "fast-1"

    @pytest.mark.asyncio
    async def test_timeout_error_does_not_close_connection(self):
        """After a timeout error, the connection stays open for more commands."""
        lifecycle = _blocking_lifecycle()
        # Make reset block but stop succeed
        lifecycle.stop_training.side_effect = None
        lifecycle.stop_training.return_value = {"state": "idle"}

        ws = _make_ws(lifecycle)

        ws.receive_text.side_effect = [
            json.dumps({"command": "reset", "command_id": "hang-1"}),
            json.dumps({"command": "stop", "command_id": "ok-2"}),
            WebSocketDisconnect(code=1000),
        ]

        with patch("api.websocket.control_stream._COMMAND_TIMEOUTS", dict.fromkeys(_COMMAND_TIMEOUTS, 0.1)):
            await control_stream_handler(ws)

        responses = [call.args[0] for call in ws.send_json.call_args_list if isinstance(call.args[0], dict) and call.args[0].get("type") == "command_response"]
        assert len(responses) == 2
        assert responses[0]["data"]["status"] == "error"  # reset timed out
        assert responses[1]["data"]["status"] == "success"  # stop succeeded


# ===================================================================
# _execute_command dispatch tests
# ===================================================================


@pytest.mark.unit
class TestExecuteCommandDispatch:
    """Verify _execute_command routes to correct lifecycle methods."""

    def test_start_calls_start_training(self):
        lifecycle = MagicMock()
        lifecycle.start_training.return_value = {"started": True}
        result = _execute_command(lifecycle, "start", {"epochs": 100})
        lifecycle.start_training.assert_called_once()
        assert result == {"started": True}

    def test_set_params_passes_params(self):
        lifecycle = MagicMock()
        lifecycle.update_params.return_value = {"updated": True}
        result = _execute_command(lifecycle, "set_params", {"learning_rate": 0.01})
        lifecycle.update_params.assert_called_once_with({"learning_rate": 0.01})
        assert result == {"updated": True}

    def test_set_params_no_params_raises(self):
        lifecycle = MagicMock()
        with pytest.raises(ValueError, match="set_params requires"):
            _execute_command(lifecycle, "set_params", None)


# ===================================================================
# §S10.3 unknown command rejection (test_unknown_command_rejected)
# ===================================================================


@pytest.mark.unit
class TestUnknownCommandRejection:
    """§S10.3: unknown command → command_response{status:"error", code:"unknown_command"}."""

    @pytest.mark.asyncio
    async def test_unknown_command_rejected(self):
        """Unknown command yields error envelope with code=unknown_command."""
        ws = _make_ws()

        ws.receive_text.side_effect = [
            json.dumps({"command": "explode", "command_id": "reject-1"}),
            WebSocketDisconnect(code=1000),
        ]

        await control_stream_handler(ws)

        responses = [call.args[0] for call in ws.send_json.call_args_list if isinstance(call.args[0], dict) and call.args[0].get("type") == "command_response"]
        assert len(responses) == 1
        resp = responses[0]
        assert resp["data"]["command"] == "explode"
        assert resp["data"]["status"] == "error"
        assert resp["data"]["code"] == "unknown_command"
        assert resp["data"]["command_id"] == "reject-1"
