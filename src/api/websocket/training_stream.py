"""WebSocket handler for /ws/training — real-time training metrics stream.

Server-to-client streaming endpoint with optional resume support. On connect:
1. connection_established message (from manager.connect_pending)
2. Optional resume handshake: client sends ``resume`` frame within timeout
3. If resume succeeds: ``resume_ok``, the replayed events, then whatever was
   broadcast while they were being sent, then promote to active -- no seq
   after the client's ``last_seq`` is skipped. A resume that cannot be kept
   gap-free from ``resume_ok`` on (a send fails, ``resume_ok``'s own
   included, or the replay buffer evicts a message the client still needs)
   is closed with 1011 ``"Resume replay incomplete"`` instead; the client
   should reconnect and resume again.
4. If fresh connect (or a failed resume): promote to active, then
   initial_status + state + initial_metrics
5. Ongoing metrics/state/topology broadcasts during training

Phase F / C3: Application-level heartbeat — the explicit contract:

* After promotion, the server sends ``{"type": "ping", "ts": <float>}`` every
  ``ws_heartbeat_interval_sec`` seconds (default 30; a value <= 0 disables the
  heartbeat entirely, the operator escape hatch for legacy clients).
* The client SHOULD reply ``{"type": "pong"}``. As a C3 tolerance, ANY
  well-formed inbound frame received within ``ws_heartbeat_pong_timeout_sec``
  seconds (default 10) of a ping also counts as proof of liveness (dead-peer
  detection, not frame-type compliance).
* A silent client is closed with close code 1011 and reason
  ``"Heartbeat timeout: no pong or traffic within <N>s"``. Pre-C3 this used
  1006, which RFC 6455 §7.4.1 forbids on the wire — the ``websockets`` server
  implementation raises ``ProtocolError`` for it, so the close frame never
  reached the peer (clients saw a half-open socket, not a close).
* juniper-cascor-client >= 0.7.0 answers pings automatically (CL1); older
  consumers must reply themselves (juniper-canopy's relay does).

T5 instrumentation: every heartbeat ping is recorded in the WS manager's
transport counters (``record_out_of_band_send``), and the manager emits a
periodic INFO emission summary (frames emitted by type since the last
summary) so "relay connected but nothing flowing" is diagnosable from the
server log alone.
"""

import asyncio
import json
import logging
import time
from typing import NoReturn

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from api.websocket.manager import ReplayOutOfRange, ws_authenticate
from api.websocket.messages import create_initial_metrics_message, create_state_message

logger = logging.getLogger("juniper_cascor.api.websocket.training")

# The close for a resume whose continuity could not be kept once ``resume_ok``
# was sent or attempted: a send failed (``resume_ok``'s own included), or the
# replay buffer evicted a message the client still needed before it could be
# sent. 1011, as for this endpoint's heartbeat-timeout and dropped-subscriber
# closes; only the reason tells them apart. The client should reconnect and
# resume again; if the gap has left the buffer by then, it is answered
# ``resume_failed`` ``out_of_range`` and should start from a fresh snapshot.
RESUME_INCOMPLETE_CLOSE_CODE = 1011
RESUME_INCOMPLETE_CLOSE_REASON = "Resume replay incomplete"


def _numeric_setting(obj, name: str, fallback):
    """Read a numeric setting attribute defensively.

    Returns ``getattr(obj, name)`` when it is a real number, else
    ``fallback`` — so a missing ``app.state.settings`` or a non-Settings
    double (whose attribute lookups return stub objects) can never leak a
    non-numeric value into ``asyncio.sleep`` / ``asyncio.wait_for``.
    """
    value = getattr(obj, name, None) if obj is not None else None
    return value if isinstance(value, (int, float)) else fallback


async def _await_resume_frame(websocket: WebSocket, ws_manager, resume_timeout: float) -> tuple[bool, bool]:
    """Wait for an optional resume frame.

    Returns (resumed, disconnected). If disconnected is True, the caller should
    return without further work.
    """
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=resume_timeout)
        msg = json.loads(raw)
        if msg.get("type") == "resume":
            return await _handle_resume(websocket, ws_manager, msg), False
    except asyncio.TimeoutError:
        pass  # No resume frame — fresh connect
    except json.JSONDecodeError:
        logger.debug("Non-JSON frame during resume handshake, treating as fresh connect")
    except WebSocketDisconnect:
        return False, True
    except Exception:
        logger.debug("Error during resume handshake, treating as fresh connect")
    return False, False


async def _send_initial_state(
    websocket: WebSocket,
    ws_manager,
    lifecycle,
    initial_metrics_count: int = 100,
) -> None:
    """Send initial status + state + recent metrics to a freshly-connected client.

    GAP-WS-16: ``initial_metrics`` is sent after ``state`` so a client can
    paint its time-series chart without a parallel REST poll. Set
    ``initial_metrics_count=0`` to disable the burst (clients then must
    request via ``subscribe_metrics`` or fall back to REST).
    """
    if lifecycle is None:
        return
    status = lifecycle.get_status()
    await ws_manager.send_personal_message(
        websocket,
        {"type": "initial_status", "data": status},
    )
    state_data = lifecycle.training_state.get_state()
    await ws_manager.send_personal_message(
        websocket,
        create_state_message(state_data),
    )
    if initial_metrics_count > 0:
        await _send_metrics_burst(websocket, ws_manager, lifecycle, initial_metrics_count)


async def _send_metrics_burst(websocket: WebSocket, ws_manager, lifecycle, count: int) -> None:
    """GAP-WS-16: send an ``initial_metrics`` burst with the most recent metrics."""
    try:
        metrics = lifecycle.get_metrics_history(count=count)
    except Exception:
        logger.debug("Training WS: get_metrics_history failed; sending empty burst", exc_info=True)
        metrics = []
    if metrics is None:
        metrics = []
    await ws_manager.send_personal_message(
        websocket,
        create_initial_metrics_message(metrics, current_seq=ws_manager.current_seq),
    )


async def _heartbeat_ping_loop(websocket: WebSocket, hb_interval: float, hb_timeout: float, pong_received: asyncio.Event, ws_manager=None) -> None:
    """Application-level ping/pong loop closing the connection on liveness timeout.

    C3: the wait is satisfied by a ``pong`` OR any other well-formed inbound
    frame (see :func:`_recv_pong_loop`) — dead-peer detection, not frame-type
    policing. The close uses code 1011: RFC 6455 §7.4.1 forbids sending 1006
    on the wire, and the ``websockets`` server implementation raises
    ``ProtocolError`` for it, so the pre-C3 ``close(code=1006)`` never actually
    delivered a close frame. Each ping sent is recorded in the WS manager's
    transport counters (T5) when available.
    """
    while True:
        await asyncio.sleep(hb_interval)
        pong_received.clear()
        try:
            await websocket.send_json({"type": "ping", "ts": time.time()})
        except Exception:
            return  # Connection already closed
        if ws_manager is not None:
            ws_manager.record_out_of_band_send({"type": "ping"})
        try:
            await asyncio.wait_for(pong_received.wait(), timeout=hb_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Training WS: heartbeat timeout, closing connection — no pong or traffic within %.0fs of ping (interval=%.0fs); clients must answer {'type':'ping'} with {'type':'pong'} (juniper-cascor-client >= 0.7.0 does this automatically)",
                hb_timeout,
                hb_interval,
            )
            try:
                await websocket.close(code=1011, reason=f"Heartbeat timeout: no pong or traffic within {hb_timeout:.0f}s")
            except Exception:
                logger.debug("Training WS: close after heartbeat timeout failed", exc_info=True)
            return


async def _recv_pong_loop(
    websocket: WebSocket,
    pong_received: asyncio.Event,
    ws_manager=None,
    lifecycle=None,
    subscribe_metrics_max_count: int = 100,
) -> None:
    """Receive loop dispatcher.

    Forwards ``pong`` frames to the heartbeat event and handles
    GAP-WS-16 ``subscribe_metrics`` requests by replying with an
    ``initial_metrics`` envelope. All other frame types are ignored
    (clients may send keep-alives the server does not need to act on).

    C3 liveness tolerance: EVERY received frame sets ``pong_received`` (before
    parsing) — inbound traffic of any shape proves the peer is alive, so the
    heartbeat loop only reaps genuinely silent peers.

    F-CASCOR-004: once the SERVER has closed the socket -- the WS manager
    closes a subscriber whose broadcast send failed or timed out, and the
    heartbeat closes a silent one -- the loop stops dispatching and drains
    (:func:`_drain_until_disconnect`). It cannot keep calling
    ``receive_text()``: after the server's own close Starlette refuses that
    with ``RuntimeError``, not ``WebSocketDisconnect``, so a frame the peer
    had in flight would end the handler with an unhandled exception.
    """
    try:
        while True:
            if getattr(websocket, "application_state", None) == WebSocketState.DISCONNECTED:
                await _drain_until_disconnect(websocket)
                return
            raw = await websocket.receive_text()
            # C3: any inbound frame is proof of liveness for the heartbeat loop.
            pong_received.set()
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, AttributeError):
                continue  # Non-JSON keep-alive frames are fine
            mtype = msg.get("type") if isinstance(msg, dict) else None
            if mtype == "pong":
                pass  # Already counted as liveness above; nothing else to do.
            elif mtype == "subscribe_metrics" and ws_manager is not None and lifecycle is not None:
                await _handle_subscribe_metrics(websocket, ws_manager, lifecycle, msg, subscribe_metrics_max_count)
    except WebSocketDisconnect:
        pass


async def _drain_until_disconnect(websocket: WebSocket) -> None:
    """After the server has closed the socket, consume frames until the ASGI server reports the disconnect.

    F-CASCOR-004: the handler must not return while the close frame may still
    be waiting to go out. uvicorn's sans-I/O protocol (its ``ws="auto"``
    choice at the pinned uvicorn 0.53.0 / websockets 17.1) holds the frame
    while the write buffer is full, and closes the transport the moment the
    app returns if no close frame has been written yet -- so a handler that
    returned on the peer's next frame (an auto-pong, say) would turn a clean
    ``1011`` into an abnormal ``1006`` on the peer's side. uvicorn reports
    the disconnect once the frame is written, or once the connection is
    lost, so this returns exactly when leaving is safe.

    ``receive()`` checks only the peer's side of the connection, so unlike
    ``receive_text()`` it still works after the server's close. What arrives
    meanwhile is discarded, not dispatched: the stream is over.
    """
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return


async def _handle_subscribe_metrics(
    websocket: WebSocket,
    ws_manager,
    lifecycle,
    msg: dict,
    default_count: int,
) -> None:
    """GAP-WS-16: respond to a client ``subscribe_metrics`` with a metrics burst.

    Used when a client reconnects via the resume path (which only replays
    seq>last_seq from the broadcast buffer) but also wants the rolling
    metrics window — useful for tab reactivation or after a long gap.
    """
    data = msg.get("data") or {}
    requested = data.get("max_count")
    try:
        count = int(requested) if requested is not None else default_count
    except (TypeError, ValueError):
        count = default_count
    count = max(1, min(count, default_count))
    await _send_metrics_burst(websocket, ws_manager, lifecycle, count)


async def training_stream_handler(websocket: WebSocket) -> None:
    """Handle /ws/training WebSocket connections with optional resume.

    Protocol flow:
    1. Authenticate via API key header
    2. Accept as pending (connect_pending — not broadcast-eligible)
    3. Wait for optional 'resume' frame within handshake timeout
    4. On resume: validate server_instance_id, replay buffered events, send
       whatever was broadcast meanwhile, and promote to active
       (``_handle_resume``) -- or close 1011 and return when that cannot be
       kept gap-free
    5. On fresh connect / failed resume: promote to active (now receives
       broadcasts), then send initial_status + state
    6. Keep-alive recv loop
    """
    if not await ws_authenticate(websocket):
        return

    ws_manager = getattr(websocket.app.state, "ws_manager", None)
    lifecycle = getattr(websocket.app.state, "lifecycle", None)
    settings = getattr(websocket.app.state, "settings", None)

    if ws_manager is None:
        await websocket.close(code=1011, reason="WebSocket manager not available")
        return

    connected = await ws_manager.connect_pending(websocket)
    if not connected:
        return

    resume_timeout = getattr(settings, "ws_resume_handshake_timeout_s", 5.0) if settings else 5.0
    initial_metrics_count = getattr(settings, "ws_initial_metrics_count", 100) if settings else 100

    try:
        resumed, disconnected = await _await_resume_frame(websocket, ws_manager, resume_timeout)
        if disconnected:
            return

        if not resumed:
            # Promote to active (now eligible for broadcasts). A resumed
            # socket is active already: _handle_resume promoted it once its
            # replay, and anything broadcast meanwhile, had been sent.
            await ws_manager.promote_to_active(websocket)
        # OBS-WIRE-02 (Q3): record this connection in the per-endpoint
        # bookkeeping bucket and re-emit
        # ``cascor_ws_connections_active{endpoint="training"}``. The
        # matching ``unregister_endpoint_connection`` lives in the
        # outer ``finally`` so the gauge always re-emits on disconnect,
        # including exception paths.
        ws_manager.register_endpoint_connection(websocket, "training")

        if not resumed:
            await _send_initial_state(websocket, ws_manager, lifecycle, initial_metrics_count)

        # Heartbeat + keep-alive: broadcasts come from training thread
        # via ws_manager.broadcast_from_thread(). The recv loop also
        # handles pong responses for dead-connection detection and
        # GAP-WS-16 subscribe_metrics requests.
        hb_interval = _numeric_setting(settings, "ws_heartbeat_interval_sec", 30)
        hb_timeout = _numeric_setting(settings, "ws_heartbeat_pong_timeout_sec", 10)
        pong_received = asyncio.Event()
        pong_received.set()  # No outstanding ping at start

        # C3: an interval <= 0 disables the heartbeat entirely (operator
        # escape hatch for legacy clients that cannot answer pings).
        ping_task = None
        if hb_interval and hb_interval > 0:
            ping_task = asyncio.create_task(_heartbeat_ping_loop(websocket, hb_interval, hb_timeout, pong_received, ws_manager=ws_manager))
        try:
            await _recv_pong_loop(
                websocket,
                pong_received,
                ws_manager=ws_manager,
                lifecycle=lifecycle,
                subscribe_metrics_max_count=initial_metrics_count or 100,
            )
        finally:
            if ping_task is not None:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass
    finally:
        # OBS-WIRE-02 (Q3): re-emit the per-endpoint gauge before the
        # outer disconnect (which only mutates the unlabeled active /
        # pending sets).
        ws_manager.unregister_endpoint_connection(websocket)
        await ws_manager.disconnect(websocket)


def _is_seq_cursor(value: object) -> bool:
    """True for a plain ``int`` -- the only type a ``seq`` this server sends has (``bool`` excluded)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _resume_cursor(msg: dict) -> tuple:
    """``(last_seq, server_instance_id)`` as a resume frame gives them; ``(None, None)`` when it gives neither.

    A ``data`` that is missing, null, or a list, string or number holds no
    cursor. Read as it came, a ``data`` that is not an object raises at
    ``data.get``, and the handshake's broad ``except`` reads that as "no
    resume frame": a fresh connect, no ``resume_failed``, nothing counted.
    """
    data = msg.get("data")
    if not isinstance(data, dict):
        return None, None
    return data.get("last_seq"), data.get("server_instance_id")


async def _abort_resume(websocket: WebSocket, why: str) -> NoReturn:
    """End a resume whose continuity can no longer be kept: close the socket, report a disconnect.

    ``resume_ok`` has already been sent (or its send failed), so a
    ``resume_failed`` would contradict it; the close is the signal left. The
    close is awaited before the handler returns, rather than left running, so
    the handler does not return while the close frame may still be waiting to
    go out. Raising ``WebSocketDisconnect`` takes the handler down its
    disconnected path (``_await_resume_frame`` returns ``disconnected``), so
    the socket is never promoted and the handler's ``finally`` releases its
    connection slots.
    """
    logger.warning(
        "Training WS: resume aborted at or after resume_ok (%s); closing with %d so the client can reconnect",
        why,
        RESUME_INCOMPLETE_CLOSE_CODE,
    )
    try:
        await websocket.close(code=RESUME_INCOMPLETE_CLOSE_CODE, reason=RESUME_INCOMPLETE_CLOSE_REASON)
    except Exception:
        logger.debug("Training WS: close after an incomplete resume failed", exc_info=True)
    raise WebSocketDisconnect(code=RESUME_INCOMPLETE_CLOSE_CODE, reason=RESUME_INCOMPLETE_CLOSE_REASON)


async def _handle_resume(
    websocket: WebSocket,
    ws_manager,
    msg: dict,
) -> bool:
    """Handle a resume request. Returns True if resume succeeded.

    On success the socket is already active. Every buffered seq after the
    client's ``last_seq`` has been sent to it once, in seq order: first the
    snapshot that ``resume_ok``'s ``replayed_count`` counts, then whatever was
    broadcast while that was being sent (``promote_to_active``'s catch-up).
    Later broadcasts reach it through the normal fan-out. If that cannot be
    kept from ``resume_ok`` on -- a send fails, ``resume_ok``'s own included,
    or the replay buffer evicts a message the client still needs -- the
    socket is closed with :data:`RESUME_INCOMPLETE_CLOSE_CODE` and
    ``WebSocketDisconnect`` is raised instead (:func:`_abort_resume`).

    A ``last_seq`` that is not a plain integer (``NaN``, ``Infinity``, a
    fraction, a boolean, a string) is answered ``resume_failed``
    ``malformed_resume``, and one above the last seq numbered
    ``out_of_range`` (:meth:`WebSocketManager.replay_since`): no seq this
    server sent can be either, so a ``resume_ok`` would vouch for nothing.
    A frame whose ``data`` is not an object carries no cursor at all and is
    ``malformed_resume`` too. A negative ``last_seq`` is read as ``0``: seqs
    start at 1, so it holds no more than ``0`` does, and the answer then
    depends on the value, not on what the buffer happens to hold.
    """
    last_seq, client_server_id = _resume_cursor(msg)

    # OBS-WIRE-02 (3.4, 3.5): the four return arms of this function
    # populate ``cascor_ws_resume_requests_total{outcome=...}``, and a
    # resume aborted after ``resume_ok`` counts as ``replay_incomplete``;
    # the success arm additionally observes
    # ``cascor_ws_resume_replayed_events`` with the count replayed.
    # All emissions are wrapped in defensive try/except per OBS-WIRE-01.
    def _emit_outcome(outcome: str) -> None:
        try:
            from api.observability import ws_inc_resume_requests

            ws_inc_resume_requests(outcome)
        except Exception:
            logger.debug("ws_inc_resume_requests emission failed for outcome=%s", outcome, exc_info=True)

    async def _abort(why: str) -> NoReturn:
        _emit_outcome("replay_incomplete")
        await _abort_resume(websocket, why)

    # ``json.loads`` accepts ``NaN`` and ``Infinity``, and Python counts a bool
    # as an int: a cursor that is not a plain integer is malformed, as a
    # missing one is.
    if client_server_id is None or not _is_seq_cursor(last_seq):
        await ws_manager.send_personal_message(
            websocket,
            {
                "type": "resume_failed",
                "timestamp": time.time(),
                "data": {"reason": "malformed_resume"},
            },
        )
        logger.debug("Resume failed: malformed resume frame (data is not an object, server_instance_id is missing, or last_seq is missing or not an integer)")
        _emit_outcome("malformed_resume")
        return False

    # D-15: server restart detection via UUID mismatch
    if client_server_id != ws_manager.server_instance_id:
        await ws_manager.send_personal_message(
            websocket,
            {
                "type": "resume_failed",
                "timestamp": time.time(),
                "data": {"reason": "server_restarted"},
            },
        )
        logger.info("Resume failed: server_instance_id mismatch (server restarted)")
        _emit_outcome("server_restarted")
        return False

    try:
        # A cursor below 0 holds no more than 0 does (seqs start at 1), so it
        # is read as 0, as promote_to_active's catch-up reads it. Unclamped,
        # replay_since answers it by buffer state: [] on an empty buffer, out
        # of range once anything is buffered, seq 1 included.
        events = ws_manager.replay_since(max(last_seq, 0))
    except ReplayOutOfRange as e:
        await ws_manager.send_personal_message(
            websocket,
            {
                "type": "resume_failed",
                "timestamp": time.time(),
                "data": {"reason": "out_of_range"},
            },
        )
        logger.info("Resume failed: %s", e)
        _emit_outcome("out_of_range")
        return False

    # Resume succeeded
    if not await ws_manager.send_personal_message(
        websocket,
        {
            "type": "resume_ok",
            "timestamp": time.time(),
            "data": {"replayed_count": len(events)},
        },
    ):
        await _abort("resume_ok was not delivered")

    # Replay buffered events. A frame that fails to send would leave a hole
    # the client cannot see, so the resume ends there instead of skipping it.
    # ``replayed_through`` is the last seq the client now holds.
    replayed_through = last_seq
    for event in events:
        if not await ws_manager.send_personal_message(websocket, event):
            await _abort(f"replayed seq {event.get('seq')} was not delivered")
        replayed_through = event["seq"]

    # A broadcast made while the replay was being sent is numbered and
    # buffered, but its fan-out reached only the sockets already active. The
    # manager sends those, then activates the socket with no gap.
    if not await ws_manager.promote_to_active(websocket, replayed_through=replayed_through):
        await _abort("the catch-up after the replay did not complete")

    logger.info("Resume succeeded: replayed %d events from seq %d", len(events), last_seq)
    _emit_outcome("success")
    try:
        from api.observability import ws_observe_resume_replayed

        ws_observe_resume_replayed(len(events))
    except Exception:
        logger.debug("ws_observe_resume_replayed emission failed", exc_info=True)
    return True
