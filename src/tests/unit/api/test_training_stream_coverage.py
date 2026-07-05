"""Coverage tests for ``api/websocket/training_stream.py``.

Phase C-5 (per-file coverage rollout, PR-2 — websocket layer). Drives the
uncovered branches of the /ws/training handler and its helpers so the file
reaches ≥90% statement coverage. All tests are pure ``unit`` (fast,
selectable by ``-m "unit and not slow"``) and exercise the handler helpers
directly with ``AsyncMock`` seams rather than a live socket.

Targets the previously-uncovered paths:

- ``_await_resume_frame``: resume-frame dispatch + non-``dict`` payload.
- ``_send_metrics_burst``: ``get_metrics_history`` raising, and returning ``None``.
- ``_heartbeat_ping_loop``: ping-send failure, and close-after-pong-timeout failure.
- ``_recv_pong_loop``: non-JSON keep-alive frame (continue).
- ``_handle_subscribe_metrics``: non-numeric ``max_count`` (fallback to default).
- ``training_stream_handler``: ``connect_pending`` rejection.
- ``_handle_resume``: server-restart, out-of-range, and success arms, plus the
  defensive metric-emission ``except`` guards.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect

from api.websocket.manager import ReplayOutOfRange
from api.websocket.training_stream import (
    _await_resume_frame,
    _handle_resume,
    _handle_subscribe_metrics,
    _heartbeat_ping_loop,
    _recv_pong_loop,
    _send_metrics_burst,
    training_stream_handler,
)


def _resume_manager(server_instance_id: str = "sid-1") -> MagicMock:
    """Build a stub ws_manager for _handle_resume with an async send seam."""
    mgr = MagicMock()
    mgr.server_instance_id = server_instance_id
    mgr.current_seq = 0
    mgr.send_personal_message = AsyncMock(return_value=True)
    return mgr


@pytest.mark.unit
class TestAwaitResumeFrame:
    """``_await_resume_frame`` dispatch + defensive parsing."""

    @pytest.mark.asyncio
    async def test_resume_frame_dispatches_to_handle_resume(self):
        """A ``resume`` frame is routed to ``_handle_resume`` (lines 38-39)."""
        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(return_value=json.dumps({"type": "resume", "data": {"last_seq": 3, "server_instance_id": "sid-1"}}))
        ws_manager = _resume_manager()

        with patch("api.websocket.training_stream._handle_resume", new=AsyncMock(return_value=True)) as mock_resume:
            resumed, disconnected = await _await_resume_frame(websocket, ws_manager, resume_timeout=1.0)

        assert resumed is True
        assert disconnected is False
        mock_resume.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_dict_payload_treated_as_fresh_connect(self):
        """A valid-JSON but non-dict frame hits the generic guard (lines 46-47)."""
        websocket = AsyncMock()
        # ``json.loads("123")`` yields an int, so ``msg.get`` raises
        # AttributeError → caught by the broad ``except Exception``.
        websocket.receive_text = AsyncMock(return_value="123")
        ws_manager = _resume_manager()

        resumed, disconnected = await _await_resume_frame(websocket, ws_manager, resume_timeout=1.0)

        assert resumed is False
        assert disconnected is False

    @pytest.mark.asyncio
    async def test_disconnect_during_handshake_signals_disconnected(self):
        """A ``WebSocketDisconnect`` during the handshake returns disconnected=True."""
        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=WebSocketDisconnect(code=1000))
        ws_manager = _resume_manager()

        resumed, disconnected = await _await_resume_frame(websocket, ws_manager, resume_timeout=1.0)

        assert resumed is False
        assert disconnected is True


@pytest.mark.unit
class TestSendMetricsBurst:
    """``_send_metrics_burst`` degraded-source handling."""

    @pytest.mark.asyncio
    async def test_get_metrics_history_raises_sends_empty_burst(self):
        """A raising ``get_metrics_history`` falls back to an empty burst (lines 84-86)."""
        websocket = AsyncMock()
        ws_manager = _resume_manager()
        lifecycle = MagicMock()
        lifecycle.get_metrics_history.side_effect = RuntimeError("boom")

        await _send_metrics_burst(websocket, ws_manager, lifecycle, count=10)

        ws_manager.send_personal_message.assert_awaited_once()
        sent = ws_manager.send_personal_message.await_args[0][1]
        assert sent["type"] == "initial_metrics"
        assert sent["data"]["count"] == 0

    @pytest.mark.asyncio
    async def test_get_metrics_history_none_normalized_to_empty(self):
        """A ``None`` metrics history is normalized to an empty list (line 88)."""
        websocket = AsyncMock()
        ws_manager = _resume_manager()
        lifecycle = MagicMock()
        lifecycle.get_metrics_history.return_value = None

        await _send_metrics_burst(websocket, ws_manager, lifecycle, count=10)

        sent = ws_manager.send_personal_message.await_args[0][1]
        assert sent["data"]["metrics"] == []


@pytest.mark.unit
class TestHeartbeatPingLoop:
    """``_heartbeat_ping_loop`` failure paths."""

    @pytest.mark.asyncio
    async def test_ping_send_failure_returns(self):
        """A failed ping send ends the loop cleanly (lines 102-103)."""
        websocket = AsyncMock()
        websocket.send_json = AsyncMock(side_effect=RuntimeError("closed"))
        pong_received = asyncio.Event()

        # Must return promptly (no live socket); guard against a regression hang.
        await asyncio.wait_for(
            _heartbeat_ping_loop(websocket, hb_interval=0, hb_timeout=0.01, pong_received=pong_received),
            timeout=2.0,
        )
        websocket.send_json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_failure_after_pong_timeout_is_swallowed(self):
        """When the pong times out and close() raises, the loop still returns (lines 110-111)."""
        websocket = AsyncMock()
        websocket.send_json = AsyncMock(return_value=None)
        websocket.close = AsyncMock(side_effect=RuntimeError("already closed"))
        pong_received = asyncio.Event()  # never set → pong wait times out

        await asyncio.wait_for(
            _heartbeat_ping_loop(websocket, hb_interval=0, hb_timeout=0.01, pong_received=pong_received),
            timeout=2.0,
        )
        websocket.close.assert_awaited_once()
        assert websocket.close.await_args.kwargs["code"] == 1006


@pytest.mark.unit
class TestRecvPongLoop:
    """``_recv_pong_loop`` non-JSON keep-alive handling."""

    @pytest.mark.asyncio
    async def test_non_json_frame_is_skipped(self):
        """A non-JSON keep-alive frame is ignored, not fatal (lines 134-135)."""
        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=["not-json", WebSocketDisconnect(code=1000)])
        pong_received = asyncio.Event()

        # Terminates on the WebSocketDisconnect after skipping the bad frame.
        await asyncio.wait_for(_recv_pong_loop(websocket, pong_received), timeout=2.0)
        assert websocket.receive_text.await_count == 2


@pytest.mark.unit
class TestHandleSubscribeMetrics:
    """``_handle_subscribe_metrics`` bad ``max_count`` handling."""

    @pytest.mark.asyncio
    async def test_non_numeric_max_count_falls_back_to_default(self):
        """A non-numeric ``max_count`` falls back to the default (lines 162-163)."""
        websocket = AsyncMock()
        ws_manager = _resume_manager()
        lifecycle = MagicMock()
        lifecycle.get_metrics_history.return_value = []

        await _handle_subscribe_metrics(websocket, ws_manager, lifecycle, {"data": {"max_count": "not-a-number"}}, default_count=100)

        ws_manager.send_personal_message.assert_awaited_once()
        sent = ws_manager.send_personal_message.await_args[0][1]
        assert sent["type"] == "initial_metrics"


@pytest.mark.unit
class TestTrainingStreamHandlerConnectPending:
    """``training_stream_handler`` connection-limit rejection."""

    @pytest.mark.asyncio
    async def test_connect_pending_rejected_returns_early(self):
        """When ``connect_pending`` returns False the handler returns (line 193)."""
        websocket = AsyncMock()
        app_state = MagicMock()
        app_state.api_key_auth = None  # auth disabled → ws_authenticate True
        ws_manager = MagicMock()
        ws_manager.connect_pending = AsyncMock(return_value=False)
        ws_manager.promote_to_active = AsyncMock()
        app_state.ws_manager = ws_manager
        app_state.lifecycle = None
        app_state.settings = None
        websocket.app.state = app_state

        await training_stream_handler(websocket)

        ws_manager.connect_pending.assert_awaited_once()
        ws_manager.promote_to_active.assert_not_called()


@pytest.mark.unit
class TestHandleResumeArms:
    """``_handle_resume`` outcome arms beyond the malformed case."""

    @pytest.mark.asyncio
    async def test_server_restarted_arm(self):
        """A mismatched ``server_instance_id`` yields the server_restarted failure (lines 285-296)."""
        websocket = AsyncMock()
        ws_manager = _resume_manager(server_instance_id="sid-current")
        msg = {"data": {"last_seq": 5, "server_instance_id": "sid-old"}}

        result = await _handle_resume(websocket, ws_manager, msg)

        assert result is False
        sent = ws_manager.send_personal_message.await_args[0][1]
        assert sent["type"] == "resume_failed"
        assert sent["data"]["reason"] == "server_restarted"

    @pytest.mark.asyncio
    async def test_out_of_range_arm(self):
        """A ``ReplayOutOfRange`` from replay_since yields the out_of_range failure (lines 298-311)."""
        websocket = AsyncMock()
        ws_manager = _resume_manager()
        ws_manager.replay_since = MagicMock(side_effect=ReplayOutOfRange("too old"))
        msg = {"data": {"last_seq": 5, "server_instance_id": "sid-1"}}

        result = await _handle_resume(websocket, ws_manager, msg)

        assert result is False
        sent = ws_manager.send_personal_message.await_args[0][1]
        assert sent["type"] == "resume_failed"
        assert sent["data"]["reason"] == "out_of_range"

    @pytest.mark.asyncio
    async def test_success_arm_replays_events(self):
        """A successful resume acks and replays each buffered event (lines 314-335)."""
        websocket = AsyncMock()
        ws_manager = _resume_manager()
        events = [{"seq": 6, "type": "metrics"}, {"seq": 7, "type": "state"}]
        ws_manager.replay_since = MagicMock(return_value=events)
        msg = {"data": {"last_seq": 5, "server_instance_id": "sid-1"}}

        result = await _handle_resume(websocket, ws_manager, msg)

        assert result is True
        # 1 resume_ok ack + one send per replayed event.
        assert ws_manager.send_personal_message.await_count == 1 + len(events)
        first = ws_manager.send_personal_message.await_args_list[0][0][1]
        assert first["type"] == "resume_ok"
        assert first["data"]["replayed_count"] == len(events)

    @pytest.mark.asyncio
    async def test_success_arm_survives_metric_emission_failure(self):
        """Defensive metric-emission guards swallow exceptions (lines 268-269, 333-334)."""
        websocket = AsyncMock()
        ws_manager = _resume_manager()
        ws_manager.replay_since = MagicMock(return_value=[{"seq": 6, "type": "metrics"}])
        msg = {"data": {"last_seq": 5, "server_instance_id": "sid-1"}}

        with patch("api.observability.ws_inc_resume_requests", side_effect=RuntimeError("emit down")), patch("api.observability.ws_observe_resume_replayed", side_effect=RuntimeError("emit down")):
            result = await _handle_resume(websocket, ws_manager, msg)

        # The resume still succeeds despite both emitters raising.
        assert result is True
