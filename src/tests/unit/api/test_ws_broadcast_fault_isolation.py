"""F-CASCOR-004: a WebSocket send failure is attributed to whatever caused it.

juniper-ml ``notes/JUNIPER_2026-08-09_JUNIPER-CANOPY_E2E-VALIDATION-EVIDENCE.md``,
entry F-CASCOR-004: on 2026-09-08 a ``state`` frame carrying NumPy-typed
tunables (F-CASCOR-003) could not be serialized by Starlette's ``send_json``.
The ``TypeError`` surfaced inside every client's send, ``_send_json`` swallowed
it with no log, and ``broadcast`` dropped each subscriber in turn -- forgetting
it without closing its socket, so both canopy relays sat on half-open streams
that the heartbeat pings kept looking healthy.

What is pinned here:

1. An unserializable message is the MESSAGE's fault. It is refused once, before
   seq assignment and fan-out -- logged at ERROR, counted, skipped -- and no
   subscriber is dropped or closed. ``send_personal_message`` still returns
   ``False`` for it, as it did before, but now through the same refusal
   instead of a silent send failure.
2. ``_send_json``'s generic failure branch logs a WARNING naming the exception.
3. A PER-CONNECTION failure (send error or timeout) drops that subscriber AND
   closes its socket -- errors suppressed, the broadcast's wait for the close
   bounded, the close itself left to finish rather than cancelled, and a close
   that fails after the wait not reported as an asyncio error -- while the
   endpoint's receive loop drains until the server reports the disconnect and
   its ``finally`` -> ``disconnect()`` stays idempotent.

The clients are REAL ``starlette.websockets.WebSocket`` objects over an
in-memory ASGI transport (:class:`_AsgiPeer`) that, like the server, encodes
each text frame as UTF-8. An ``AsyncMock`` cannot reproduce the defect: its
``send_json`` serializes nothing, whereas Starlette's runs stdlib
``json.dumps`` with no ``default=`` and raises ``TypeError`` on ``np.int64``.
What this does NOT exercise is a real server: uvicorn's close handshake,
its timeouts and the wire are outside these tests (see
``util/ad-hoc/2026-09-22_f_cascor_004_live_ws_probe.py`` for those).
"""

import asyncio
import contextlib
import json
import logging
import time
from types import SimpleNamespace
from typing import Callable, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket

import api.observability as obs
from api.app import create_app
from api.settings import Settings
from api.websocket.manager import BROADCAST_DROP_CLOSE_CODE, BROADCAST_DROP_CLOSE_REASON, WebSocketManager
from api.websocket.training_stream import training_stream_handler


def _unserializable_state() -> dict:
    """The F-CASCOR-003 shape: runtime tunables restored as NumPy scalars."""
    return {"type": "state", "data": {"status": "Started", "patience": np.int64(3), "multiprocessing": np.bool_(True)}}


class _AsgiPeer:
    """One subscriber as the server sees it: a real Starlette ``WebSocket`` over an in-memory ASGI transport.

    ``sent`` records every ASGI message the server emitted. The knobs make the
    transport misbehave the ways a real one can.
    """

    def __init__(self, host: str, *, app: Optional[SimpleNamespace] = None) -> None:
        self.sent: List[dict] = []
        self.close_attempts: List[dict] = []
        self.fail_sends_with: Optional[BaseException] = None  # every data frame raises this
        self.stall_sends = False  # a data frame blocks until release(): a peer that stopped reading
        self.stall_closes = False  # ...and so does the close frame
        self.fail_close_with: Optional[BaseException] = None  # the close raises this (after any stall): the peer is gone
        self.frames_after_close: List[str] = []  # frames the peer had in flight when the close went out
        self._drained = asyncio.Event()
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._inbox.put_nowait({"type": "websocket.connect"})
        scope = {"type": "websocket", "path": "/ws/training", "headers": [], "query_string": b"", "client": (host, 50000), "app": app}
        self.websocket = WebSocket(scope, self._receive, self._send)

    async def _receive(self) -> dict:
        return await self._inbox.get()

    async def _send(self, message: dict) -> None:
        if message["type"] == "websocket.send":
            if self.stall_sends:
                await self._drained.wait()
            if self.fail_sends_with is not None:
                raise self.fail_sends_with
            text = message.get("text")
            if text is not None:
                text.encode("utf-8")  # what the ASGI server does to a text frame on its way to the wire
        elif message["type"] == "websocket.close":
            self.close_attempts.append(message)
            if self.stall_closes:
                await self._drained.wait()
            if self.fail_close_with is not None:
                # The connection is lost: the server tells the app, and the close raises.
                self._inbox.put_nowait({"type": "websocket.disconnect", "code": 1006})
                raise self.fail_close_with
            for text in self.frames_after_close:
                self._inbox.put_nowait({"type": "websocket.receive", "text": text})
            # What the ASGI server hands the app once the close frame is written.
            self._inbox.put_nowait({"type": "websocket.disconnect", "code": message["code"]})
        self.sent.append(message)

    def hang_up(self) -> None:
        """The peer disconnects on its own."""
        self._inbox.put_nowait({"type": "websocket.disconnect", "code": 1000})

    def peer_sends(self, text: str) -> None:
        """The peer sends a text frame, whatever the server is doing."""
        self._inbox.put_nowait({"type": "websocket.receive", "text": text})

    @property
    def inbox_empty(self) -> bool:
        """Every frame the peer sent has been read by the server."""
        return self._inbox.empty()

    def release(self) -> None:
        """The stalled peer starts reading again: every blocked frame completes."""
        self._drained.set()

    @property
    def frames(self) -> List[dict]:
        return [json.loads(m["text"]) for m in self.sent if m["type"] == "websocket.send"]

    @property
    def frame_types(self) -> List[str]:
        return [frame["type"] for frame in self.frames]

    @property
    def close_codes(self) -> List[int]:
        return [m["code"] for m in self.sent if m["type"] == "websocket.close"]


async def _connected_pair(mgr: WebSocketManager, first: str = "10.0.0.1", second: str = "10.0.0.2"):
    """Two subscribers admitted through ``connect()`` (each holds a global + per-IP slot)."""
    a, b = _AsgiPeer(first), _AsgiPeer(second)
    assert await mgr.connect(a.websocket) is True
    assert await mgr.connect(b.websocket) is True
    return a, b


def _render(call) -> str:
    """The line a mocked ``logger.<level>(fmt, *args)`` call would have logged."""
    fmt, *args = call.args
    return fmt % tuple(args)


def _handler_app(mgr: WebSocketManager) -> SimpleNamespace:
    """The slice of ``app.state`` that ``training_stream_handler`` reads: auth off, no lifecycle, no heartbeat."""
    settings = SimpleNamespace(
        ws_resume_handshake_timeout_s=0.01,  # no resume frame arrives: a fresh connect after 10 ms
        ws_initial_metrics_count=0,
        ws_heartbeat_interval_sec=0,  # <= 0 disables the heartbeat
        ws_heartbeat_pong_timeout_sec=10,
    )
    return SimpleNamespace(state=SimpleNamespace(ws_manager=mgr, lifecycle=None, settings=settings, api_key_auth=None))


async def _until(predicate: Callable[[], bool], *, attempts: int = 500) -> None:
    """Yield to the event loop until ``predicate()`` holds (bounded at ~5 s)."""
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def _reset_ws_metrics() -> None:
    """Force a lazy re-init of the ws metrics (the ``test_metrics_obs_wire_02`` pattern)."""
    from prometheus_client import REGISTRY

    if obs._ws_metrics is not None:
        for metric in list(obs._ws_metrics.values()):
            with contextlib.suppress(Exception):
                REGISTRY.unregister(metric)
        obs._ws_metrics = None


# ---------------------------------------------------------------------------
# 1. An unserializable message is the message's fault
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnserializableMessageIsTheMessagesFault:
    """Refused once, before seq assignment and fan-out: logged, counted, skipped; nobody dropped."""

    @pytest.mark.asyncio
    async def test_numpy_broadcast_drops_and_closes_nobody(self):
        mgr = WebSocketManager()
        a, b = await _connected_pair(mgr)

        with patch("api.websocket.manager.logger") as mock_logger:
            await mgr.broadcast(_unserializable_state())

        # Neither subscriber is disconnected, and neither socket is closed.
        assert mgr._active_connections == {a.websocket, b.websocket}
        assert a.close_codes == [] and b.close_codes == []
        # Nothing reached either wire.
        assert a.frame_types == ["connection_established"]
        assert b.frame_types == ["connection_established"]
        # Logged once, at ERROR, naming the message type and the exception -- and no per-connection WARNING.
        assert mock_logger.error.call_count == 1
        line = _render(mock_logger.error.call_args)
        assert "state message" in line
        assert "TypeError" in line and "int64" in line
        mock_logger.warning.assert_not_called()
        # Counted as a message fault, not as a delivery failure.
        stats = mgr.transport_stats()
        assert stats["unserializable_messages_total"] == 1
        assert stats["send_failures"] == 0
        # No seq consumed, nothing buffered for replay.
        assert mgr.current_seq == 0
        assert mgr.replay_since(0) == []

    @pytest.mark.asyncio
    async def test_next_serializable_broadcast_reaches_both_as_seq_1(self):
        mgr = WebSocketManager()
        a, b = await _connected_pair(mgr)

        await mgr.broadcast(_unserializable_state())
        await mgr.broadcast({"type": "metrics", "data": {"epoch": 1}})

        for peer in (a, b):
            assert peer.frame_types == ["connection_established", "metrics"]
            assert peer.frames[-1]["seq"] == 1
            assert peer.frames[-1]["data"] == {"epoch": 1}
        assert [event["seq"] for event in mgr.replay_since(0)] == [1]

    @pytest.mark.asyncio
    async def test_undecodable_text_is_refused_the_same_way(self):
        """A lone surrogate passes ``json.dumps`` but not the transport's UTF-8 encode: still the message's fault."""
        mgr = WebSocketManager()
        a, b = await _connected_pair(mgr)

        with patch("api.websocket.manager.logger") as mock_logger:
            await mgr.broadcast({"type": "event", "data": {"snapshot_path": "run_\udc80.h5"}})

        assert mgr._active_connections == {a.websocket, b.websocket}
        assert a.close_codes == [] and b.close_codes == []
        assert "UnicodeEncodeError" in _render(mock_logger.error.call_args)
        assert mgr.transport_stats()["unserializable_messages_total"] == 1
        assert mgr.current_seq == 0

    @pytest.mark.asyncio
    async def test_what_starlette_accepts_is_not_refused(self):
        """The guard mirrors the transport, no stricter: ``np.float64`` subclasses ``float`` and serializes."""
        mgr = WebSocketManager()
        a, _ = await _connected_pair(mgr)

        await mgr.broadcast({"type": "metrics", "data": {"loss": np.float64(0.25)}})

        assert a.frames[-1]["data"] == {"loss": 0.25}
        assert mgr.transport_stats()["unserializable_messages_total"] == 0

    @pytest.mark.asyncio
    async def test_send_personal_message_returns_false_and_leaves_the_connection_alone(self):
        mgr = WebSocketManager()
        a, _ = await _connected_pair(mgr)

        with patch("api.websocket.manager.logger") as mock_logger:
            delivered = await mgr.send_personal_message(a.websocket, {"type": "initial_status", "data": {"epoch": np.int64(7)}})

        assert delivered is False
        assert a.frame_types == ["connection_established"]
        assert a.websocket in mgr._active_connections
        assert a.close_codes == []
        assert "initial_status message" in _render(mock_logger.error.call_args)
        stats = mgr.transport_stats()
        assert stats["unserializable_messages_total"] == 1
        assert stats["send_failures"] == 0


@pytest.mark.unit
class TestUnserializableMessagesMetric:
    """``cascor_ws_unserializable_messages_total{type}``: once per message, not per subscriber."""

    def setup_method(self):
        _reset_ws_metrics()

    def teardown_method(self):
        _reset_ws_metrics()

    @pytest.mark.asyncio
    async def test_counter_increments_once_per_message_by_type(self):
        mgr = WebSocketManager()
        a, _ = await _connected_pair(mgr)
        counter = obs._ensure_ws_metrics()["unserializable_messages_total"]

        await mgr.broadcast(_unserializable_state())  # two subscribers, one message
        await mgr.send_personal_message(a.websocket, {"type": "initial_status", "data": {"epoch": np.int64(7)}})

        assert counter.labels(type="state")._value.get() == 1
        assert counter.labels(type="initial_status")._value.get() == 1

    @pytest.mark.asyncio
    async def test_emission_failure_is_swallowed(self):
        """A failing counter emission never turns the refusal into an exception."""
        mgr = WebSocketManager()
        a, b = await _connected_pair(mgr)

        with patch("api.observability.ws_inc_unserializable_messages", side_effect=RuntimeError("emit down")):
            await mgr.broadcast(_unserializable_state())

        assert mgr._active_connections == {a.websocket, b.websocket}
        assert mgr.transport_stats()["unserializable_messages_total"] == 1


@pytest.mark.unit
class TestTransportEndpointServesTheCounter:
    """End to end: real app, real ``/ws/training`` subscriber, the counter on ``GET /v1/metrics/transport``."""

    @pytest.fixture
    def fast_client(self):
        app = create_app(Settings(auto_start=False, ws_resume_handshake_timeout_s=0.1, ws_initial_metrics_count=0))
        with TestClient(app) as client:
            yield client

    @staticmethod
    def _transport_when(client, predicate, *, attempts: int = 500) -> dict:
        data = {}
        for _ in range(attempts):
            data = client.get("/v1/metrics/transport").json()["data"]
            if predicate(data):
                return data
            time.sleep(0.01)
        raise AssertionError(f"transport stats never satisfied the predicate: {data}")

    def test_unserializable_broadcast_is_counted_and_the_subscriber_keeps_receiving(self, fast_client):
        ws_manager = fast_client.app.state.ws_manager
        before = fast_client.get("/v1/metrics/transport").json()["data"]
        assert before["unserializable_messages_total"] == 0

        with fast_client.websocket_connect("/ws/training") as ws:
            for expected in ("connection_established", "initial_status", "state"):
                assert ws.receive_json()["type"] == expected
            seq_before = fast_client.get("/v1/metrics/transport").json()["data"]["current_seq"]

            ws_manager.broadcast_from_thread(_unserializable_state())
            ws_manager.broadcast_from_thread({"type": "metrics", "data": {"epoch": 1}})
            data = self._transport_when(fast_client, lambda d: d["messages_sent_by_type"].get("metrics", 0) >= 1)

            assert data["unserializable_messages_total"] == 1
            assert data["send_failures"] == 0
            assert data["active_connections"] == 1
            frame = ws.receive_json()
            assert frame["type"] == "metrics"
            assert frame["seq"] == seq_before + 1  # the refused message consumed no seq


# ---------------------------------------------------------------------------
# 2. _send_json's generic failure branch logs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSendJsonGenericFailureLogs:
    """The branch that used to count and return with no log now names the exception."""

    @pytest.mark.asyncio
    async def test_generic_send_failure_logs_a_warning(self):
        mgr = WebSocketManager()
        ws = AsyncMock()
        ws.send_json.side_effect = ConnectionResetError("peer reset the stream")

        with patch("api.websocket.manager.logger") as mock_logger:
            result = await mgr._send_json(ws, {"type": "metrics", "data": {}})

        assert result is False
        mock_logger.warning.assert_called_once()
        line = _render(mock_logger.warning.call_args)
        assert "metrics message" in line
        assert "ConnectionResetError" in line and "peer reset the stream" in line
        assert mgr.transport_stats()["send_failures"] == 1


# ---------------------------------------------------------------------------
# 3. A per-connection failure drops AND closes that subscriber only
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerConnectionFailureClosesTheSocket:
    """Forgotten and closed, bounded and with errors suppressed; the other subscriber is untouched."""

    @pytest.mark.asyncio
    async def test_send_error_drops_and_closes_only_that_subscriber(self):
        mgr = WebSocketManager()
        bad, good = await _connected_pair(mgr)
        bad.fail_sends_with = RuntimeError("transport write failed")

        with patch("api.websocket.manager.logger") as mock_logger:
            await mgr.broadcast({"type": "metrics", "data": {"epoch": 1}})

        assert mgr._active_connections == {good.websocket}
        assert bad.close_codes == [BROADCAST_DROP_CLOSE_CODE]
        assert bad.sent[-1]["reason"] == BROADCAST_DROP_CLOSE_REASON
        assert good.frame_types == ["connection_established", "metrics"]
        assert good.close_codes == []
        warnings = [_render(call) for call in mock_logger.warning.call_args_list]
        assert any("RuntimeError" in w and "transport write failed" in w for w in warnings)
        assert mgr.transport_stats()["send_failures"] == 1

        await mgr.broadcast({"type": "metrics", "data": {"epoch": 2}})
        assert good.frames[-1]["data"] == {"epoch": 2}

    @pytest.mark.asyncio
    async def test_send_timeout_drops_and_closes_that_subscriber(self):
        mgr = WebSocketManager(send_timeout_seconds=0.05)
        slow, good = await _connected_pair(mgr)
        slow.stall_sends = True

        await asyncio.wait_for(mgr.broadcast({"type": "metrics", "data": {"epoch": 1}}), timeout=5.0)

        assert mgr._active_connections == {good.websocket}
        assert slow.close_codes == [BROADCAST_DROP_CLOSE_CODE]
        assert good.frames[-1]["type"] == "metrics"

    @pytest.mark.asyncio
    async def test_close_wait_is_bounded_and_the_close_finishes_in_the_background(self):
        """A peer that takes neither the frame nor the close cannot wedge the broadcast -- nor lose its close.

        The outer ``wait_for`` is the first assertion: both stalls last until
        ``release()``, so an unbounded wait on the close would never return.
        The second is that the close is NOT cancelled at the bound: once the
        peer drains, it completes. (Cancelling it would abandon the closing
        handshake part-way; ``websockets`` says not to.)
        """
        mgr = WebSocketManager(send_timeout_seconds=0.05)
        wedged, good = await _connected_pair(mgr)
        wedged.stall_sends = True
        wedged.stall_closes = True

        await asyncio.wait_for(mgr.broadcast({"type": "metrics", "data": {"epoch": 1}}), timeout=5.0)

        assert mgr._active_connections == {good.websocket}
        assert good.frames[-1]["type"] == "metrics"
        assert len(wedged.close_attempts) == 1  # the close was started...
        assert wedged.close_codes == []  # ...and was still in flight when the broadcast moved on

        wedged.release()
        await _until(lambda: wedged.close_codes == [BROADCAST_DROP_CLOSE_CODE])
        await _until(lambda: not mgr._pending_closes)

    @pytest.mark.asyncio
    async def test_a_close_that_fails_after_the_wait_is_not_reported_as_an_asyncio_error(self, caplog):
        """A slow subscriber is dropped, then its peer resets before the backlog drains: that is no ERROR.

        Written as ``wait_for(shield(close))``, the timed-out wait made Python
        >= 3.14 attach a logger to the shielded close, so its later failure
        was reported through the loop's exception handler -- "WebSocketDisconnect
        exception in shielded future", at ERROR, with a traceback -- although
        ``_close_finished`` retrieves the exception. 3.12 and 3.13 have no such
        logger, so there this passes either way; the 3.14 CI leg is the guard.

        The loop's exception handler is the primary instrument: it sees the
        report even when logging is configured to drop it, which would make a
        caplog-only assertion pass vacuously.
        """
        loop = asyncio.get_running_loop()
        reported: List[dict] = []
        previous_handler = loop.get_exception_handler()

        def _record(event_loop, context):
            reported.append(context)
            event_loop.default_exception_handler(context)

        loop.set_exception_handler(_record)
        try:
            with caplog.at_level(logging.DEBUG, logger="asyncio"):
                mgr = WebSocketManager(send_timeout_seconds=0.05)
                wedged, good = await _connected_pair(mgr)
                wedged.stall_sends = True
                wedged.stall_closes = True
                wedged.fail_close_with = ConnectionResetError("peer reset before the backlog drained")

                await asyncio.wait_for(mgr.broadcast({"type": "metrics", "data": {"epoch": 1}}), timeout=5.0)
                assert len(wedged.close_attempts) == 1 and wedged.close_codes == []  # still in flight at the bound

                wedged.release()  # ...and now it fails
                await _until(lambda: not mgr._pending_closes)
                for _ in range(3):
                    await asyncio.sleep(0)  # let every done-callback of the failed close run
        finally:
            loop.set_exception_handler(previous_handler)

        assert reported == []
        assert [record.getMessage() for record in caplog.records if record.name == "asyncio" and record.levelno >= logging.ERROR] == []
        assert good.frames[-1]["type"] == "metrics"

    @pytest.mark.asyncio
    async def test_a_close_cancelled_at_shutdown_is_released_quietly(self):
        """At shutdown the loop cancels a close still in flight; its done callback must not raise.

        ``exception()`` on a cancelled future raises ``CancelledError``, which a
        done callback would surface as an "Exception in callback" error.
        """
        mgr = WebSocketManager()
        closing = asyncio.get_running_loop().create_future()
        mgr._pending_closes.add(closing)
        closing.cancel()

        mgr._close_finished(closing)

        assert mgr._pending_closes == set()

    @pytest.mark.asyncio
    async def test_every_failed_subscriber_is_forgotten_before_any_close(self):
        """While the first close is in flight, no failed subscriber is still a broadcast target."""
        mgr = WebSocketManager(send_timeout_seconds=2.0)
        good, bad1, bad2 = _AsgiPeer("10.0.0.1"), _AsgiPeer("10.0.0.2"), _AsgiPeer("10.0.0.3")
        for peer in (good, bad1, bad2):
            assert await mgr.connect(peer.websocket) is True
        for bad in (bad1, bad2):
            bad.fail_sends_with = RuntimeError("transport write failed")
            bad.stall_closes = True

        task = asyncio.create_task(mgr.broadcast({"type": "metrics", "data": {"epoch": 1}}))
        await _until(lambda: len(bad1.close_attempts) + len(bad2.close_attempts) == 1)

        assert mgr._active_connections == {good.websocket}

        bad1.release()
        bad2.release()
        await asyncio.wait_for(task, timeout=5.0)
        assert bad1.close_codes == [BROADCAST_DROP_CLOSE_CODE]
        assert bad2.close_codes == [BROADCAST_DROP_CLOSE_CODE]
        assert good.frames[-1]["type"] == "metrics"

    @pytest.mark.asyncio
    async def test_close_errors_are_suppressed(self):
        """A socket whose close raises -- or cannot even start -- is still dropped, and the broadcast completes."""
        mgr = WebSocketManager()
        good = _AsgiPeer("10.0.0.2")
        assert await mgr.connect(good.websocket) is True
        gone = AsyncMock()  # the transport is already gone, so the close raises
        gone.send_json.side_effect = OSError("broken pipe")
        gone.close.side_effect = RuntimeError('Cannot call "send" once a close message has been sent.')
        unclosable = MagicMock()  # a synchronous close() hands back nothing awaitable
        unclosable.send_json = AsyncMock(side_effect=OSError("broken pipe"))
        mgr._active_connections.update({gone, unclosable})

        await mgr.broadcast({"type": "metrics", "data": {"epoch": 1}})

        gone.close.assert_awaited_once_with(code=BROADCAST_DROP_CLOSE_CODE, reason=BROADCAST_DROP_CLOSE_REASON)
        unclosable.close.assert_called_once_with(code=BROADCAST_DROP_CLOSE_CODE, reason=BROADCAST_DROP_CLOSE_REASON)
        assert mgr._active_connections == {good.websocket}
        assert good.frames[-1]["type"] == "metrics"
        await _until(lambda: not mgr._pending_closes)

    @pytest.mark.asyncio
    async def test_a_second_disconnect_releases_no_slot_twice(self):
        """The endpoint's ``finally`` runs ``disconnect()`` again after a drop: the survivor's slots are intact."""
        mgr = WebSocketManager()
        bad, good = await _connected_pair(mgr, "10.0.0.9", "10.0.0.9")  # one shared per-IP bucket
        assert mgr._per_ip_counts == {"10.0.0.9": 2} and mgr._global_ws_count == 2
        bad.fail_sends_with = RuntimeError("transport write failed")

        await mgr.broadcast({"type": "metrics", "data": {"epoch": 1}})
        assert mgr._per_ip_counts == {"10.0.0.9": 1} and mgr._global_ws_count == 1

        mgr.unregister_endpoint_connection(bad.websocket)
        await mgr.disconnect(bad.websocket)

        assert mgr._per_ip_counts == {"10.0.0.9": 1} and mgr._global_ws_count == 1
        assert mgr._active_connections == {good.websocket}


@pytest.mark.unit
class TestTrainingStreamHandlerAfterADrop:
    """The real ``/ws/training`` handler: the close ends its receive loop cleanly and its ``finally`` is idempotent."""

    @pytest.mark.asyncio
    async def test_dropped_subscriber_handler_returns_through_its_finally(self):
        mgr = WebSocketManager()
        app = _handler_app(mgr)
        bad, good = _AsgiPeer("10.0.0.1", app=app), _AsgiPeer("10.0.0.2", app=app)
        tasks = [asyncio.create_task(training_stream_handler(peer.websocket)) for peer in (bad, good)]
        await _until(lambda: mgr.connection_count == 2)
        bad.fail_sends_with = RuntimeError("transport write failed")

        await mgr.broadcast({"type": "metrics", "data": {"epoch": 1}})

        # Before the fix the socket was never closed, so this handler sat in its receive loop for good.
        await asyncio.wait_for(tasks[0], timeout=5.0)
        assert bad.close_codes == [BROADCAST_DROP_CLOSE_CODE]
        assert bad.websocket not in mgr._endpoint_connections["training"]
        # The survivor: still running, still active, still registered, still receiving.
        assert not tasks[1].done()
        assert mgr._active_connections == {good.websocket}
        assert mgr._endpoint_connections["training"] == {good.websocket}
        assert good.frames[-1]["type"] == "metrics"
        # Exactly one connection's worth of slots is held: the drop and the handler's finally released one, once.
        assert mgr._global_ws_count == 1 and mgr._per_ip_counts == {"10.0.0.2": 1}

        good.hang_up()
        await asyncio.wait_for(tasks[1], timeout=5.0)
        assert mgr.connection_count == 0 and mgr._global_ws_count == 0 and mgr._per_ip_counts == {}

    @pytest.mark.asyncio
    async def test_handler_drains_until_the_server_reports_the_disconnect(self):
        """While the server's close is still waiting to go out, the handler must not return.

        uvicorn's sans-I/O protocol (its ``ws="auto"`` choice at the pinned
        uvicorn 0.53.0 / websockets 17.1) holds a close frame while its write
        buffer is full, and closes the transport as soon as the app returns if
        no close frame has been written. A handler that returned on the peer's
        next frame -- here a pong that was in flight -- would cost the peer its
        1011 close frame: it would see an abnormal 1006 instead. The fake plays
        that server: the close stays pending until ``release()``, and only then
        is the disconnect reported.
        """
        mgr = WebSocketManager(send_timeout_seconds=0.05)
        app = _handler_app(mgr)
        slow, good = _AsgiPeer("10.0.0.1", app=app), _AsgiPeer("10.0.0.2", app=app)
        tasks = [asyncio.create_task(training_stream_handler(peer.websocket)) for peer in (slow, good)]
        await _until(lambda: mgr.connection_count == 2)
        slow.stall_sends = True
        slow.stall_closes = True

        await asyncio.wait_for(mgr.broadcast({"type": "metrics", "data": {"epoch": 1}}), timeout=5.0)
        assert len(slow.close_attempts) == 1 and slow.close_codes == []  # the close is waiting to go out

        slow.peer_sends('{"type":"pong"}')
        await _until(lambda: slow.inbox_empty)  # the handler has read the pong...
        await asyncio.sleep(0.05)  # ...and a handler about to return has had ample time to
        assert not tasks[0].done()

        slow.peer_sends('{"type":"pong"}')  # a frame that arrives mid-drain is consumed, and does not end it either
        await _until(lambda: slow.inbox_empty)
        await asyncio.sleep(0.05)
        assert not tasks[0].done()

        slow.release()  # the buffer drains: the close frame is written and the disconnect reported
        await asyncio.wait_for(tasks[0], timeout=5.0)
        assert slow.close_codes == [BROADCAST_DROP_CLOSE_CODE]
        await _until(lambda: not mgr._pending_closes)
        assert mgr._global_ws_count == 1 and mgr._per_ip_counts == {"10.0.0.2": 1}

        good.hang_up()
        await asyncio.wait_for(tasks[1], timeout=5.0)
        assert mgr._global_ws_count == 0 and mgr._per_ip_counts == {}

    @pytest.mark.asyncio
    async def test_frame_in_flight_when_the_close_went_out_does_not_crash_the_handler(self):
        """After the server's close, Starlette refuses ``receive_text()`` with RuntimeError; the loop must stop first."""
        mgr = WebSocketManager()
        app = _handler_app(mgr)
        bad, good = _AsgiPeer("10.0.0.1", app=app), _AsgiPeer("10.0.0.2", app=app)
        bad.frames_after_close = ['{"type":"pong"}']
        tasks = [asyncio.create_task(training_stream_handler(peer.websocket)) for peer in (bad, good)]
        await _until(lambda: mgr.connection_count == 2)
        bad.fail_sends_with = RuntimeError("transport write failed")

        await mgr.broadcast({"type": "metrics", "data": {"epoch": 1}})

        await asyncio.wait_for(tasks[0], timeout=5.0)  # re-raises anything the handler raised
        assert bad.close_codes == [BROADCAST_DROP_CLOSE_CODE]
        assert mgr._global_ws_count == 1 and mgr._per_ip_counts == {"10.0.0.2": 1}

        good.hang_up()
        await asyncio.wait_for(tasks[1], timeout=5.0)
        assert mgr._global_ws_count == 0 and mgr._per_ip_counts == {}
