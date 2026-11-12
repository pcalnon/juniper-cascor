"""Resume continuity on ``/ws/training``: a ``resume_ok`` means the client missed nothing.

There were two ways a resuming client could miss broadcasts while the server answered
``resume_ok``, in cascor's own WebSocket code. juniper-ml's
``notes/JUNIPER_2026-11-06_JUNIPER-ECOSYSTEM_CURSOR-FLOOD-4-DISPOSITION.md``, §8, ruling 11,
records them. juniper-ml's shared ``juniper-service-core`` had the same two: juniper-ml#2534
fixed them on juniper-ml's ``main`` on 2026-11-09, and juniper-service-core 0.7.2, the latest
on PyPI on 2026-11-12, was published before that fix and still has both. The two:

* ``WebSocketManager.broadcast`` returned before numbering a message when no socket was active
  (the ``if not self._active_connections: return`` guard in ``api/websocket/manager.py``). A
  broadcast made while nobody was connected -- including while the only client was still
  pending its resume handshake -- got no seq and no replay slot, so a client resuming from its
  last seq was answered ``resume_ok`` over a gap it could not see.
* A broadcast made while ``_handle_resume`` was sending the replay was numbered and buffered,
  but fanned out only to the sockets already active. The resuming socket was promoted only after
  the replay, so that broadcast reached it by neither path.

Every test drives the real ``WebSocketManager`` and, where a resume is involved, the real
``training_stream_handler``: with a fake socket (:class:`FakeWS`), or, in the last three tests, a
real Starlette ``WebSocket`` over an in-memory ASGI transport (:class:`AsgiResumePeer`). A send
can suspend (as a send waiting on transport back-pressure does) or run a hook before its frame
lands, which is how a broadcast is made to land *during* a replay send deterministically. The
resume tests assert the exact seq list the client received: every seq after its ``last_seq``,
once each, in order -- or, where that cannot be kept, a gap-free prefix and a ``1011`` close,
never a hole. Broadcasts in these tests are made one at a time, so the order asserted is the
order the resume path produced.

The send timeout and the resume-handshake timeout are set far above anything a test waits for
(:data:`_SEND_TIMEOUT`, :data:`_HANDSHAKE_TIMEOUT`), so no outcome here depends on how fast the
host runs. The two tests whose subject includes the send timeout block a send forever, so that
timeout always fires, however late.
"""

import asyncio
import json
import random
import types
from typing import Any, Awaitable, Callable, List, Optional, Tuple, Union

import numpy as np
import pytest
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocket

import api.observability as obs
from api.websocket.manager import ReplayOutOfRange, WebSocketManager
from api.websocket.training_stream import _handle_resume, training_stream_handler

pytestmark = pytest.mark.unit

#: The close an aborted resume sends: the wire values a client sees, pinned literally.
RESUME_INCOMPLETE = (1011, "Resume replay incomplete")

#: A send timeout no healthy fake send comes near, however loaded the host: a send that
#: suspends is resumed within a few event-loop iterations.
_SEND_TIMEOUT = 60.0

#: Likewise for the window in which the handler waits for a ``resume`` frame.
_HANDSHAKE_TIMEOUT = 60.0

#: Upper bound on waiting for a handler to finish after the peer hangs up (a guard, not a
#: timing assumption: a correct handler finishes within a few loop iterations).
_FINISH_TIMEOUT = 30.0

#: A hook :class:`FakeWS` runs inside a send, before the frame lands.
SendHook = Callable[["FakeWS", dict], Awaitable[None]]


class FakeWS:
    """A ``/ws/training`` fake socket.

    ``receive_text`` returns the ``incoming`` frames in order (the first one only once
    ``ready_to_resume`` is set, when one is given). After them it sets ``listening`` -- the
    handler is serving its receive loop -- and blocks until :meth:`hang_up`, then raises
    ``WebSocketDisconnect``.

    ``yields`` is how many times a send suspends before its frame lands: an int, or a callable
    drawn once per send. ``on_send`` runs inside a send, before the frame lands.
    ``fail_on_seq`` / ``fail_on_type`` make the first send of that seq / frame type raise, as a
    send that meets a dead transport does. ``block_on_seq`` makes a send of that seq block
    forever, as a send to a peer that stopped reading does; only the send timeout ends it.
    """

    def __init__(
        self,
        *,
        client: Tuple[str, int] = ("1.2.3.4", 5000),
        app: Optional[types.SimpleNamespace] = None,
        incoming: Optional[List[str]] = None,
        ready_to_resume: Optional[asyncio.Event] = None,
        yields: Union[int, Callable[[], int]] = 0,
        on_send: Optional[SendHook] = None,
        fail_on_seq: Optional[int] = None,
        fail_on_type: Optional[str] = None,
        block_on_seq: Optional[int] = None,
    ) -> None:
        self.client = client
        self.headers: dict = {}
        self.app = app
        self.sent: List[dict] = []
        self.closed: Optional[Tuple[int, str]] = None
        self.listening = asyncio.Event()
        self._incoming = list(incoming or [])
        self._ready_to_resume = ready_to_resume
        self._yields = yields
        self._on_send = on_send
        self._fail_on_seq = fail_on_seq
        self._fail_on_type = fail_on_type
        self._block_on_seq = block_on_seq
        self._hung_up = asyncio.Event()

    async def accept(self) -> None:
        return None

    async def send_json(self, message: dict) -> None:
        if self._on_send is not None:
            await self._on_send(self, message)
        for _ in range(self._yields() if callable(self._yields) else self._yields):
            await asyncio.sleep(0)
        if self._block_on_seq is not None and message.get("seq") == self._block_on_seq:
            await asyncio.Event().wait()  # never set: only the manager's send timeout ends this
        if self._fail_on_seq is not None and message.get("seq") == self._fail_on_seq:
            self._fail_on_seq = None
            raise RuntimeError("send failed")
        if self._fail_on_type is not None and message.get("type") == self._fail_on_type:
            self._fail_on_type = None
            raise RuntimeError("send failed")
        self.sent.append(message)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)

    async def receive_text(self) -> str:
        if self._ready_to_resume is not None:
            await self._ready_to_resume.wait()
            self._ready_to_resume = None
        if self._incoming:
            return self._incoming.pop(0)
        self.listening.set()
        await self._hung_up.wait()
        raise WebSocketDisconnect(code=1000)

    def hang_up(self) -> None:
        self._hung_up.set()

    def seqs(self) -> List[int]:
        return [m["seq"] for m in self.sent if "seq" in m]

    def types(self) -> List[Optional[str]]:
        return [m.get("type") for m in self.sent]

    def frame(self, frame_type: str) -> dict:
        return next(m for m in self.sent if m.get("type") == frame_type)


class AsgiResumePeer:
    """A ``/ws/training`` client as the server sees it: a real Starlette ``WebSocket`` over an in-memory ASGI transport.

    Unlike :class:`FakeWS`, every frame goes through Starlette's own ``send_json`` (stdlib
    ``json.dumps``) and ``close`` (its state machine), and what the server emits is recorded as
    the ASGI messages the server would hand to the wire. ``on_frame`` runs inside the transport
    write, before the frame lands. ``stall_seq`` makes the write of that seq block forever, as
    for a peer that stopped reading; only the manager's send timeout ends it. ``dead_on_seq``
    makes the write of that seq raise ``OSError``, as a reset connection does; Starlette then
    marks the socket disconnected, so nothing more, a close included, can be sent on it.
    """

    def __init__(
        self,
        app: types.SimpleNamespace,
        *,
        host: str = "3.3.3.3",
        on_frame: Optional[Callable[[dict], Awaitable[None]]] = None,
        stall_seq: Optional[int] = None,
        dead_on_seq: Optional[int] = None,
    ) -> None:
        self.frames: List[dict] = []
        self.closes: List[dict] = []
        self.receiving = asyncio.Event()
        self._on_frame = on_frame
        self._stall_seq = stall_seq
        self._dead_on_seq = dead_on_seq
        self._inbox: "asyncio.Queue[dict]" = asyncio.Queue()
        self._inbox.put_nowait({"type": "websocket.connect"})
        scope = {"type": "websocket", "path": "/ws/training", "headers": [], "query_string": b"", "client": (host, 50000), "app": app}
        self.websocket = WebSocket(scope, self._receive, self._send)

    async def _receive(self) -> dict:
        if self._inbox.empty():
            self.receiving.set()  # the server is waiting on the peer: its receive loop
        return await self._inbox.get()

    async def _send(self, message: dict) -> None:
        if message["type"] == "websocket.send":
            frame = json.loads(message["text"])
            if self._on_frame is not None:
                await self._on_frame(frame)
            if self._stall_seq is not None and frame.get("seq") == self._stall_seq:
                await asyncio.Event().wait()  # never set
            if self._dead_on_seq is not None and frame.get("seq") == self._dead_on_seq:
                raise OSError("connection reset by peer")
            self.frames.append(frame)
        elif message["type"] == "websocket.close":
            self.closes.append(message)
            self._inbox.put_nowait({"type": "websocket.disconnect", "code": message["code"]})

    def peer_sends(self, text: str) -> None:
        self._inbox.put_nowait({"type": "websocket.receive", "text": text})

    def hang_up(self) -> None:
        self._inbox.put_nowait({"type": "websocket.disconnect", "code": 1000})

    def seqs(self) -> List[int]:
        return [f["seq"] for f in self.frames if "seq" in f]


def _manager(**kwargs: Any) -> WebSocketManager:
    """A manager whose send timeout no test reaches unless the test is about that timeout."""
    kwargs.setdefault("send_timeout_seconds", _SEND_TIMEOUT)
    return WebSocketManager(**kwargs)


def _metrics(tag: object) -> dict:
    return {"type": "metrics", "data": {"tag": tag}}


def _app(mgr: WebSocketManager) -> types.SimpleNamespace:
    """``app.state`` for the handler: this manager, no lifecycle, no auth, no heartbeat."""
    settings = types.SimpleNamespace(
        ws_resume_handshake_timeout_s=_HANDSHAKE_TIMEOUT,
        ws_initial_metrics_count=0,
        ws_heartbeat_interval_sec=0,  # <= 0 disables the heartbeat
        ws_heartbeat_pong_timeout_sec=_HANDSHAKE_TIMEOUT,
    )
    return types.SimpleNamespace(state=types.SimpleNamespace(ws_manager=mgr, settings=settings, lifecycle=None))


def _resume_frame(mgr: WebSocketManager, last_seq: int) -> str:
    return json.dumps({"type": "resume", "data": {"last_seq": last_seq, "server_instance_id": mgr.server_instance_id}})


async def _observer(mgr: WebSocketManager) -> FakeWS:
    """An already-active socket, so broadcasts have somewhere to go on the old code too."""
    other = FakeWS(client=("9.9.9.9", 1))
    assert await mgr.connect(other)
    return other


def _start_resume(mgr: WebSocketManager, last_seq: int, **ws_kwargs: Any) -> Tuple[FakeWS, "asyncio.Task[None]"]:
    ws = FakeWS(app=_app(mgr), incoming=[_resume_frame(mgr, last_seq)], **ws_kwargs)
    return ws, asyncio.create_task(training_stream_handler(ws))


async def _settle(task: "asyncio.Task[None]", ws: FakeWS) -> None:
    """Run the loop until the handler is serving its receive loop, or has returned."""
    for _ in range(10_000):
        if task.done() or ws.listening.is_set():
            return
        await asyncio.sleep(0)
    raise AssertionError("the handler neither returned nor reached its receive loop")


async def _finish(task: "asyncio.Task[None]", ws: Union[FakeWS, "AsgiResumePeer"]) -> None:
    """Hang up if the handler is still serving, then await it (bounded)."""
    if not task.done():
        ws.hang_up()
    await asyncio.wait_for(task, timeout=_FINISH_TIMEOUT)


async def _until_serving_or_done(task: "asyncio.Task[None]", serving: asyncio.Event) -> None:
    """:func:`_settle` for a test that waits on a timer (bounded by wall time, not loop iterations).

    Returns once the handler has returned or is serving its receive loop, whichever comes first,
    so a handler that wrongly carries on fails the test at once rather than at the bound.
    """
    waiter = asyncio.ensure_future(serving.wait())
    try:
        await asyncio.wait({task, waiter}, timeout=_FINISH_TIMEOUT, return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiter.cancel()
    assert task.done() or serving.is_set(), "the handler neither returned nor reached its receive loop"


def _outcome_count(outcome: str) -> float:
    return obs._ensure_ws_metrics()["resume_requests_total"].labels(outcome=outcome)._value.get()


# ======================================================================================
# Defect 1 -- a broadcast with no socket active is numbered and buffered
# ======================================================================================


@pytest.mark.asyncio
async def test_a_broadcast_with_no_socket_connected_is_sequenced_and_buffered() -> None:
    mgr = _manager()
    await mgr.broadcast(_metrics(0))
    await mgr.broadcast({"type": "state", "data": {"status": "running"}})

    assert mgr.current_seq == 2
    assert [m["seq"] for m in mgr.replay_since(0)] == [1, 2]
    stats = mgr.transport_stats()
    assert stats["replay_buffer_size"] == 2
    # Nobody to send to: numbered and buffered, never sent.
    assert stats["messages_sent_total"] == 0
    assert stats["send_failures"] == 0


@pytest.mark.asyncio
async def test_resume_replays_what_was_broadcast_while_nobody_was_connected() -> None:
    mgr = _manager()
    first = FakeWS(client=("9.9.9.9", 1))
    assert await mgr.connect(first)
    await mgr.broadcast(_metrics(0))  # seq 1: the client holds it
    await mgr.disconnect(first)
    await mgr.broadcast(_metrics(1))  # nobody connected
    await mgr.broadcast(_metrics(2))

    ws, task = _start_resume(mgr, last_seq=1)
    await _settle(task, ws)
    await _finish(task, ws)

    assert ws.types()[:2] == ["connection_established", "resume_ok"]
    assert ws.frame("resume_ok")["data"]["replayed_count"] == 2
    assert ws.seqs() == [2, 3]


@pytest.mark.asyncio
async def test_a_broadcast_during_the_only_clients_pending_handshake_is_replayed() -> None:
    """The socket waiting to send ``resume`` is pending, so this was the case "no socket active"."""
    mgr = _manager()
    first = FakeWS(client=("9.9.9.9", 1))
    assert await mgr.connect(first)
    await mgr.broadcast(_metrics(0))  # seq 1
    await mgr.disconnect(first)

    ready = asyncio.Event()
    ws, task = _start_resume(mgr, last_seq=1, ready_to_resume=ready)
    for _ in range(10):  # let the handler accept the socket as pending and wait for the frame
        await asyncio.sleep(0)
    assert ws in mgr._pending_connections
    await mgr.broadcast(_metrics("while-pending"))  # seq 2: the only socket is pending
    ready.set()
    await _settle(task, ws)
    await _finish(task, ws)

    assert ws.frame("resume_ok")["data"]["replayed_count"] == 1
    assert ws.seqs() == [2]


@pytest.mark.asyncio
async def test_a_disabled_replay_buffer_still_sequences_but_buffers_nothing() -> None:
    mgr = _manager(max_replay_buffer_size=0)
    await mgr.broadcast(_metrics(0))

    assert mgr.current_seq == 1  # the same numbering whether or not a client is connected
    assert mgr.transport_stats()["replay_buffer_size"] == 0
    with pytest.raises(ReplayOutOfRange, match="disabled"):
        mgr.replay_since(0)


@pytest.mark.asyncio
async def test_with_nobody_connected_the_buffer_keeps_its_maxlen_and_an_evicted_resume_is_refused() -> None:
    mgr = _manager(max_replay_buffer_size=3)
    for i in range(5):  # nobody connected: seqs 1-5, the buffer keeps 3-5
        await mgr.broadcast(_metrics(i))

    assert [m["seq"] for m in mgr.replay_since(2)] == [3, 4, 5]
    assert mgr.transport_stats()["replay_buffer_size"] == 3

    # A client that last saw seq 1 missed seq 2, which has left the buffer: refused, not
    # resumed over the gap.
    ws, task = _start_resume(mgr, last_seq=1)
    await _settle(task, ws)
    await _finish(task, ws)
    assert ws.frame("resume_failed")["data"]["reason"] == "out_of_range"
    assert "resume_ok" not in ws.types()


@pytest.mark.asyncio
async def test_a_chunked_broadcast_made_while_nobody_was_connected_is_replayed_whole() -> None:
    mgr = _manager(max_message_size_bytes=200, chunk_payload_size_bytes=64)
    big = {"type": "topology", "data": {"blob": "x" * 1000}}
    await mgr.broadcast(big)
    n_chunks = mgr.current_seq
    assert n_chunks > 1  # one seq, and one replay slot, per chunk

    ws, task = _start_resume(mgr, last_seq=0)
    await _settle(task, ws)
    await _finish(task, ws)

    chunks = [m for m in ws.sent if m.get("type") == "chunked_message"]
    assert [c["seq"] for c in chunks] == list(range(1, n_chunks + 1))
    assert [c["data"]["chunk_index"] for c in chunks] == list(range(n_chunks))
    assert json.loads("".join(c["data"]["payload"] for c in chunks)) == big


@pytest.mark.asyncio
async def test_an_unserializable_broadcast_with_nobody_connected_is_refused_and_takes_no_seq() -> None:
    """F-CASCOR-004's refusal now runs whether or not anyone is connected: counted, no seq, no slot."""
    mgr = _manager()
    await mgr.broadcast({"type": "state", "data": {"patience": np.int64(3)}})
    await mgr.broadcast(_metrics(0))

    assert mgr.current_seq == 1
    assert [m["seq"] for m in mgr.replay_since(0)] == [1]
    assert mgr.transport_stats()["unserializable_messages_total"] == 1


# ======================================================================================
# Defect 2 -- a broadcast landing between the replay snapshot and activation is delivered
# ======================================================================================


@pytest.mark.asyncio
async def test_a_broadcast_that_lands_while_the_replay_is_being_sent_reaches_the_resuming_client() -> None:
    mgr = _manager()
    other = await _observer(mgr)
    for i in range(5):  # seqs 1-5
        await mgr.broadcast(_metrics(i))

    async def broadcast_mid_replay(_ws: FakeWS, message: dict) -> None:
        if message.get("seq") == 3:  # the first replayed frame is in flight
            await mgr.broadcast(_metrics("during-replay"))  # seq 6

    ws, task = _start_resume(mgr, last_seq=2, on_send=broadcast_mid_replay)
    await _settle(task, ws)
    assert ws in mgr._active_connections
    await mgr.broadcast(_metrics("live"))  # seq 7
    await _finish(task, ws)

    assert ws.types()[:2] == ["connection_established", "resume_ok"]
    # replayed_count is the snapshot; seq 6 follows it, before the live stream.
    assert ws.frame("resume_ok")["data"]["replayed_count"] == 3
    assert ws.seqs() == [3, 4, 5, 6, 7]
    assert other.seqs() == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.asyncio
async def test_the_flood4_validator_race_a_broadcast_scheduled_during_a_yielding_replay_send() -> None:
    """The validator's own driver: ``_handle_resume`` called directly, then ``promote_to_active``.

    That is the order the handler used before the fix; after it, ``_handle_resume`` has already
    promoted the socket gap-free and the second promotion changes nothing.
    """
    mgr = _manager()
    await _observer(mgr)
    for i in range(5):
        await mgr.broadcast(_metrics(i))
    resuming = FakeWS(client=("2.2.2.2", 2), yields=1)
    assert await mgr.connect_pending(resuming)

    late = asyncio.ensure_future(mgr.broadcast(_metrics("during-replay")))
    msg = {"type": "resume", "data": {"last_seq": 2, "server_instance_id": mgr.server_instance_id}}
    assert await _handle_resume(resuming, mgr, msg) is True
    await asyncio.gather(late)
    await mgr.promote_to_active(resuming)
    await mgr.broadcast(_metrics("after-promote"))

    assert [m["seq"] for m in mgr.replay_since(0)] == [1, 2, 3, 4, 5, 6, 7]
    assert resuming.seqs() == [3, 4, 5, 6, 7]


@pytest.mark.asyncio
async def test_a_broadcast_while_the_resuming_socket_waits_for_the_manager_lock_is_not_lost() -> None:
    """The last check and the activation must both come after the lock is acquired.

    The socket's last replayed frame takes ``mgr._lock`` (as a concurrent ``connect`` that is
    slow to ``accept()`` holds it), so the activation has to wait; a broadcast lands during that
    wait and the lock is then released.
    """
    mgr = _manager()
    await _observer(mgr)
    for i in range(3):  # seqs 1-3
        await mgr.broadcast(_metrics(i))

    async def broadcast_then_release() -> None:
        await mgr.broadcast(_metrics("while-waiting-for-the-lock"))  # seq 4
        mgr._lock.release()

    background: List["asyncio.Task[None]"] = []

    async def take_the_lock_on_the_last_replayed_frame(_ws: FakeWS, message: dict) -> None:
        if message.get("seq") == 3:
            await mgr._lock.acquire()
            background.append(asyncio.create_task(broadcast_then_release()))

    ws, task = _start_resume(mgr, last_seq=0, on_send=take_the_lock_on_the_last_replayed_frame)
    await _settle(task, ws)
    await asyncio.gather(*background)
    await mgr.broadcast(_metrics("live"))  # seq 5
    await _finish(task, ws)

    assert ws.seqs() == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_a_chunked_broadcast_that_lands_during_the_replay_arrives_whole_and_in_order() -> None:
    mgr = _manager(max_message_size_bytes=200, chunk_payload_size_bytes=64)
    await _observer(mgr)
    for i in range(2):  # seqs 1-2, each one frame
        await mgr.broadcast(_metrics(i))
    big = {"type": "topology", "data": {"blob": "y" * 1000}}

    async def chunked_broadcast_mid_replay(_ws: FakeWS, message: dict) -> None:
        if message.get("seq") == 1:
            await mgr.broadcast(big)

    ws, task = _start_resume(mgr, last_seq=0, on_send=chunked_broadcast_mid_replay)
    await _settle(task, ws)
    await _finish(task, ws)

    assert ws.seqs() == list(range(1, mgr.current_seq + 1))
    chunks = [m for m in ws.sent if m.get("type") == "chunked_message"]
    assert len(chunks) == mgr.current_seq - 2
    assert json.loads("".join(c["data"]["payload"] for c in chunks)) == big


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(40))
async def test_resume_interleaved_with_broadcasts_delivers_every_seq_once_in_order(seed: int) -> None:
    """Randomised (seeded) yields on both sides; broadcasts are sequential, the resume concurrent.

    Sequential broadcasts cannot overtake one another, so any hole, duplicate or reordering in
    what the resuming socket receives comes from the resume path.
    """
    rng = random.Random(seed)
    mgr = _manager()
    other = await _observer(mgr)
    for i in range(rng.randint(0, 6)):
        await mgr.broadcast(_metrics(f"before-{i}"))
    last_seq = rng.randint(0, mgr.current_seq)
    ws, task = _start_resume(mgr, last_seq, yields=lambda: rng.randint(0, 3))

    for i in range(rng.randint(1, 12)):
        for _ in range(rng.randint(0, 3)):
            await asyncio.sleep(0)
        await mgr.broadcast(_metrics(f"during-{i}"))
    await _settle(task, ws)
    await mgr.broadcast(_metrics("live"))
    await _finish(task, ws)

    assert ws.seqs() == list(range(last_seq + 1, mgr.current_seq + 1))
    assert other.seqs() == list(range(1, mgr.current_seq + 1))


@pytest.mark.asyncio
async def test_a_broadcast_during_resume_ok_with_nothing_to_replay_still_arrives() -> None:
    """A resume with nothing to replay still runs the catch-up.

    The client holds every seq, so the replay is empty, but a broadcast lands while ``resume_ok``
    is being sent and the socket is still pending: only the catch-up can deliver it.
    """
    mgr = _manager()
    await _observer(mgr)
    for i in range(3):  # seqs 1-3: the client holds all of them
        await mgr.broadcast(_metrics(i))

    async def broadcast_during_resume_ok(_ws: FakeWS, message: dict) -> None:
        if message.get("type") == "resume_ok":
            await mgr.broadcast(_metrics("during-resume-ok"))  # seq 4

    ws, task = _start_resume(mgr, last_seq=3, on_send=broadcast_during_resume_ok)
    await _settle(task, ws)
    await mgr.broadcast(_metrics("live"))  # seq 5
    await _finish(task, ws)

    assert ws.frame("resume_ok")["data"]["replayed_count"] == 0
    assert ws.seqs() == [4, 5]


@pytest.mark.asyncio
async def test_a_resumed_socket_is_promoted_once_through_the_catch_up_only() -> None:
    """The handler must not promote a resumed socket a second time, without a cursor.

    ``_handle_resume`` promotes it through the catch-up. A second, unconditional
    ``promote_to_active`` after it is a no-op while the socket is still active, but whenever
    that call has to wait for the manager lock, a socket that a failed broadcast send dropped
    meanwhile is put back into the broadcast set: the resurrection the catch-up refuses.
    """
    mgr = _manager()
    await _observer(mgr)
    for i in range(3):  # seqs 1-3
        await mgr.broadcast(_metrics(i))
    calls: List[dict] = []
    real = mgr.promote_to_active

    async def spy(websocket: FakeWS, **kwargs: Any) -> bool:
        calls.append(kwargs)
        return await real(websocket, **kwargs)

    mgr.promote_to_active = spy
    ws, task = _start_resume(mgr, last_seq=1)
    await _settle(task, ws)
    await _finish(task, ws)

    assert calls == [{"replayed_through": 3}]


# ======================================================================================
# The window between the final check and the activation holds no await
# ======================================================================================


@pytest.mark.asyncio
async def test_a_broadcast_ready_to_run_at_the_final_check_still_reaches_the_promoted_socket() -> None:
    """Pins "no ``await`` between the catch-up's final check and the activation".

    A broadcast is scheduled, not awaited, just before ``promote_to_active``, so it is the next
    thing the loop runs whenever the promotion yields. If the promotion yields before its final
    check, the broadcast is numbered first and the check finds it (the catch-up sends it). If it
    yields after the check but before the socket joins the active set, the broadcast is numbered
    after the check and fanned out without the socket: the socket never receives it. With no
    ``await`` in that window, the activation lands first and the fan-out includes the socket.
    """
    mgr = _manager()
    observer = await _observer(mgr)
    ws = FakeWS(client=("2.2.2.2", 2))
    assert await mgr.connect_pending(ws)
    for i in range(3):  # seqs 1-3, while the socket is pending
        await mgr.broadcast(_metrics(i))
    for event in mgr.replay_since(0):  # the replay
        assert await mgr.send_personal_message(ws, event)

    late = asyncio.ensure_future(mgr.broadcast(_metrics("late")))  # seq 4: ready to run, not yet run
    assert await mgr.promote_to_active(ws, replayed_through=3) is True
    await asyncio.gather(late)

    assert observer.seqs() == [1, 2, 3, 4]
    assert ws.seqs() == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_a_broadcast_numbered_before_the_catch_up_reaches_the_socket_once() -> None:
    """Pins the broadcast side of the same rule: no ``await`` between numbering and copying the active set.

    The broadcast starts first, so its seq is in the buffer when the catch-up checks and the
    catch-up sends it. Its fan-out was fixed when it was numbered, before the socket was
    active, so it does not send it again. If ``broadcast`` yielded between numbering and copying
    the active set, the catch-up would send the seq and activate the socket meanwhile, and the
    fan-out would then send the same seq a second time.
    """
    mgr = _manager()
    observer = await _observer(mgr)
    ws = FakeWS(client=("2.2.2.2", 2))
    assert await mgr.connect_pending(ws)
    for i in range(2):  # seqs 1-2, while the socket is pending
        await mgr.broadcast(_metrics(i))
    for event in mgr.replay_since(0):  # the replay
        assert await mgr.send_personal_message(ws, event)

    late = asyncio.ensure_future(mgr.broadcast(_metrics("late")))  # seq 3
    await asyncio.sleep(0)  # let it start: numbered and buffered before the catch-up looks
    assert await mgr.promote_to_active(ws, replayed_through=2) is True
    await asyncio.gather(late)

    assert observer.seqs() == [1, 2, 3]
    assert ws.seqs() == [1, 2, 3]


@pytest.mark.asyncio
async def test_a_task_queued_on_the_manager_lock_cannot_come_between_the_final_check_and_the_activation() -> None:
    """Pins "the final check and the activation share ONE ``self._lock`` section".

    A second task (a ``connect_pending``, as a concurrent connect that is slow to ``accept()``)
    is queued on ``mgr._lock`` behind the promotion when the lock is released, and a broadcast
    is scheduled right after. An activation in a second ``async with self._lock``, after the
    check, would hand the lock to the queued task in between, so re-acquiring it would wait; the
    broadcast would be numbered and fanned out while the socket was still pending, and the
    socket would never receive that seq. No ``await`` in the source spells that window, so the
    explicit-``await`` test above cannot see it.
    """
    mgr = _manager()
    observer = await _observer(mgr)
    ws = FakeWS(client=("2.2.2.2", 2))
    assert await mgr.connect_pending(ws)
    for i in range(3):  # seqs 1-3, while the socket is pending
        await mgr.broadcast(_metrics(i))
    for event in mgr.replay_since(0):  # the replay
        assert await mgr.send_personal_message(ws, event)

    await mgr._lock.acquire()
    promote = asyncio.create_task(mgr.promote_to_active(ws, replayed_through=3))
    await asyncio.sleep(0)  # the promotion now waits on the lock
    contender = asyncio.create_task(mgr.connect_pending(FakeWS(client=("5.5.5.5", 5))))
    await asyncio.sleep(0)  # and the contender is queued behind it
    mgr._lock.release()
    late = asyncio.create_task(mgr.broadcast(_metrics("late")))  # seq 4
    assert await promote is True
    await asyncio.gather(late)
    assert await contender is True

    assert observer.seqs() == [1, 2, 3, 4]
    assert ws.seqs() == [1, 2, 3, 4]


# ======================================================================================
# Continuity that cannot be kept ends the connection rather than leaving a hole
# ======================================================================================


@pytest.mark.asyncio
async def test_a_resume_whose_catch_up_was_evicted_is_closed_not_promoted_over_the_gap() -> None:
    mgr = _manager(max_replay_buffer_size=4)
    await _observer(mgr)
    for i in range(4):  # seqs 1-4
        await mgr.broadcast(_metrics(i))

    async def burst_mid_replay(_ws: FakeWS, message: dict) -> None:
        if message.get("seq") == 2:  # first replayed frame in flight
            for i in range(6):  # seqs 5-10; the 4-slot buffer now holds 7-10
                await mgr.broadcast(_metrics(f"burst-{i}"))

    ws, task = _start_resume(mgr, last_seq=1, on_send=burst_mid_replay)
    await _settle(task, ws)
    if ws.listening.is_set():  # promoted: show what it receives next
        await mgr.broadcast(_metrics("live"))
    await _finish(task, ws)

    assert ws.seqs() == [2, 3, 4]  # the stream up to the gap, and nothing after it
    assert ws.closed == RESUME_INCOMPLETE
    assert ws not in mgr._active_connections
    assert ws not in mgr._pending_connections


@pytest.mark.asyncio
@pytest.mark.parametrize("evicting_pass", [1, 2, 3])
async def test_an_eviction_during_any_catch_up_pass_closes_the_resume_without_a_hole(evicting_pass: int) -> None:
    """Every catch-up pass runs ``replay_since``'s range check, not only the first.

    The test above evicts during the replay, so the first catch-up pass sees it. Here each frame
    the catch-up sends triggers one more broadcast, so each pass finds exactly one new seq and
    another pass follows; while pass ``evicting_pass`` is sending, a burst pushes the next seqs
    the socket needs out of the buffer. The pass after it must see the eviction and end the
    resume. A pass that read the buffer without the range check would skip the evicted seqs and
    carry on under ``resume_ok``.
    """
    mgr = _manager(max_replay_buffer_size=4)
    await _observer(mgr)
    for i in range(2):  # seqs 1-2
        await mgr.broadcast(_metrics(i))
    catch_up_frames = {"sent": 0}

    async def one_more_broadcast_per_frame_then_a_burst(_ws: FakeWS, message: dict) -> None:
        seq = message.get("seq")
        if seq is None:  # connection_established, resume_ok
            return
        if seq == 2:  # the replay's only frame: makes catch-up pass 1 necessary
            await mgr.broadcast(_metrics("for-pass-1"))  # seq 3
            return
        catch_up_frames["sent"] += 1  # one frame per pass here
        if catch_up_frames["sent"] < evicting_pass:
            await mgr.broadcast(_metrics(f"for-pass-{catch_up_frames['sent'] + 1}"))
        elif catch_up_frames["sent"] == evicting_pass:
            for i in range(6):  # six seqs into the 4-slot buffer: the next two the socket needs go
                await mgr.broadcast(_metrics(f"burst-{i}"))

    ws, task = _start_resume(mgr, last_seq=1, on_send=one_more_broadcast_per_frame_then_a_burst)
    await _settle(task, ws)
    if ws.listening.is_set():  # promoted: show what it receives next
        await mgr.broadcast(_metrics("live"))
    await _finish(task, ws)

    assert ws.seqs() == list(range(2, 3 + evicting_pass))  # the stream up to the gap, nothing after it
    assert ws.closed == RESUME_INCOMPLETE
    assert ws not in mgr._active_connections
    assert ws not in mgr._pending_connections


@pytest.mark.asyncio
async def test_a_replay_frame_that_fails_to_send_ends_the_resume_instead_of_leaving_a_hole() -> None:
    mgr = _manager()
    await _observer(mgr)
    for i in range(4):  # seqs 1-4
        await mgr.broadcast(_metrics(i))

    ws, task = _start_resume(mgr, last_seq=1, fail_on_seq=3)
    await _settle(task, ws)
    if ws.listening.is_set():
        await mgr.broadcast(_metrics("live"))
    await _finish(task, ws)

    assert ws.seqs() == [2]
    assert ws.closed == RESUME_INCOMPLETE
    assert ws not in mgr._active_connections


@pytest.mark.asyncio
async def test_a_catch_up_frame_that_fails_to_send_ends_the_resume_instead_of_leaving_a_hole() -> None:
    mgr = _manager()
    await _observer(mgr)
    for i in range(4):  # seqs 1-4
        await mgr.broadcast(_metrics(i))

    async def broadcast_mid_replay(_ws: FakeWS, message: dict) -> None:
        if message.get("seq") == 2:
            await mgr.broadcast(_metrics("during-replay"))  # seq 5, delivered by the catch-up

    ws, task = _start_resume(mgr, last_seq=1, on_send=broadcast_mid_replay, fail_on_seq=5)
    await _settle(task, ws)
    if ws.listening.is_set():
        await mgr.broadcast(_metrics("live"))
    await _finish(task, ws)

    assert ws.seqs() == [2, 3, 4]
    assert ws.closed == RESUME_INCOMPLETE
    assert ws not in mgr._active_connections


@pytest.mark.asyncio
async def test_a_resume_ok_that_fails_to_send_ends_the_resume() -> None:
    mgr = _manager()
    await _observer(mgr)
    for i in range(3):  # seqs 1-3
        await mgr.broadcast(_metrics(i))

    ws, task = _start_resume(mgr, last_seq=1, fail_on_type="resume_ok")
    await _settle(task, ws)
    if ws.listening.is_set():
        await mgr.broadcast(_metrics("live"))
    await _finish(task, ws)

    assert ws.seqs() == []  # nothing replayed behind an acknowledgement the client never saw
    assert ws.closed == RESUME_INCOMPLETE
    assert ws not in mgr._active_connections


@pytest.mark.asyncio
async def test_a_replay_send_that_times_out_ends_the_resume() -> None:
    """The subject here IS the send timeout, so it is short; the blocked send never returns on its own."""
    mgr = _manager(send_timeout_seconds=0.05)
    await _observer(mgr)
    for i in range(4):  # seqs 1-4
        await mgr.broadcast(_metrics(i))

    ws, task = _start_resume(mgr, last_seq=1, block_on_seq=3)
    await _until_serving_or_done(task, ws.listening)
    assert task.done()  # the abort ended the handler by itself
    await _finish(task, ws)

    assert ws.seqs() == [2]
    assert ws.closed == RESUME_INCOMPLETE
    assert ws not in mgr._active_connections
    assert ws not in mgr._pending_connections


@pytest.mark.asyncio
async def test_an_aborted_resume_releases_its_connection_slots() -> None:
    mgr = _manager()
    await _observer(mgr)  # holds one global slot and one 9.9.9.9 slot
    for i in range(3):
        await mgr.broadcast(_metrics(i))
    global_before = mgr._global_ws_count
    per_ip_before = dict(mgr._per_ip_counts)

    ws, task = _start_resume(mgr, last_seq=1, fail_on_seq=2)
    await _settle(task, ws)
    await _finish(task, ws)

    assert ws.closed == RESUME_INCOMPLETE
    assert mgr._global_ws_count == global_before
    assert mgr._per_ip_counts == per_ip_before
    assert ws not in mgr._connection_meta


@pytest.mark.asyncio
async def test_an_aborted_resume_is_counted_as_replay_incomplete_not_success() -> None:
    mgr = _manager()
    await _observer(mgr)
    for i in range(3):
        await mgr.broadcast(_metrics(i))
    success_before = _outcome_count("success")
    incomplete_before = _outcome_count("replay_incomplete")

    ws, task = _start_resume(mgr, last_seq=1, fail_on_seq=3)
    await _settle(task, ws)
    await _finish(task, ws)

    assert ws.closed == RESUME_INCOMPLETE
    assert _outcome_count("replay_incomplete") - incomplete_before == 1
    assert _outcome_count("success") - success_before == 0


@pytest.mark.asyncio
async def test_a_completed_resume_is_counted_as_success_once() -> None:
    mgr = _manager()
    await _observer(mgr)
    for i in range(3):
        await mgr.broadcast(_metrics(i))
    success_before = _outcome_count("success")
    incomplete_before = _outcome_count("replay_incomplete")

    ws, task = _start_resume(mgr, last_seq=1)
    await _settle(task, ws)
    await _finish(task, ws)

    assert ws.closed is None
    assert ws.seqs() == [2, 3]
    assert _outcome_count("success") - success_before == 1
    assert _outcome_count("replay_incomplete") - incomplete_before == 0


# ======================================================================================
# promote_to_active(replayed_through=...) -- the manager-side contract
# ======================================================================================


@pytest.mark.asyncio
async def test_promote_without_a_cursor_is_the_unconditional_promotion_it_was() -> None:
    mgr = _manager()
    ws = FakeWS()
    assert await mgr.connect_pending(ws)
    await mgr.broadcast(_metrics(0))  # seq 1, while pending: not delivered by this call

    assert await mgr.promote_to_active(ws) is True
    assert ws in mgr._active_connections
    assert ws not in mgr._pending_connections
    assert ws.seqs() == []


@pytest.mark.asyncio
async def test_promote_with_a_cursor_leaves_an_already_active_socket_alone() -> None:
    mgr = _manager()
    ws = FakeWS()
    assert await mgr.connect(ws)
    await mgr.broadcast(_metrics(0))  # seq 1, delivered live
    sent_before = list(ws.sent)

    assert await mgr.promote_to_active(ws, replayed_through=0) is True
    assert ws.sent == sent_before  # no catch-up copy of seq 1


@pytest.mark.asyncio
async def test_promote_with_a_cursor_does_not_resurrect_a_socket_disconnected_during_the_catch_up() -> None:
    mgr = _manager()

    async def disconnect_mid_catch_up(sock: FakeWS, message: dict) -> None:
        if message.get("seq") == 1:
            await mgr.disconnect(sock)

    ws = FakeWS(on_send=disconnect_mid_catch_up)
    assert await mgr.connect_pending(ws)
    await mgr.broadcast(_metrics(0))  # seq 1
    await mgr.broadcast(_metrics(1))  # seq 2

    assert await mgr.promote_to_active(ws, replayed_through=0) is False
    assert ws not in mgr._active_connections
    assert ws not in mgr._pending_connections


@pytest.mark.asyncio
async def test_promote_with_a_negative_cursor_delivers_from_the_first_seq() -> None:
    """Seqs start at 1, so a cursor below 0 holds no more than 0 does -- it is not a gap."""
    mgr = _manager()
    ws = FakeWS()
    assert await mgr.connect_pending(ws)
    await mgr.broadcast(_metrics(0))  # seq 1, while the only socket is pending

    assert await mgr.promote_to_active(ws, replayed_through=-5) is True
    assert ws in mgr._active_connections
    assert ws.seqs() == [1]


@pytest.mark.asyncio
async def test_promote_with_a_cursor_on_an_empty_buffer_activates() -> None:
    mgr = _manager()
    ws = FakeWS()
    assert await mgr.connect_pending(ws)

    assert await mgr.promote_to_active(ws, replayed_through=0) is True
    assert ws in mgr._active_connections


@pytest.mark.asyncio
async def test_promote_with_a_cursor_and_a_disabled_buffer_is_refused() -> None:
    """With replay disabled nothing can be caught up, so continuity cannot be vouched for."""
    mgr = _manager(max_replay_buffer_size=0)
    ws = FakeWS()
    assert await mgr.connect_pending(ws)

    assert await mgr.promote_to_active(ws, replayed_through=0) is False
    assert ws not in mgr._active_connections
    assert ws in mgr._pending_connections  # left for the caller to close


# ======================================================================================
# A cursor that matches no seq this server sent is refused, never answered resume_ok
# ======================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_last_seq", ["NaN", "Infinity", "-Infinity", "2.5", "2.0", "true", '"2"', "[2]"])
async def test_a_last_seq_that_is_not_a_plain_integer_is_refused_as_malformed(raw_last_seq: str) -> None:
    """``json.loads`` accepts ``NaN`` and ``Infinity``, and a bool is an int to Python; no seq is any of them.

    Before this was checked, ``NaN``, ``Infinity``, ``2.5``, ``2.0`` and ``true`` were answered
    ``resume_ok``: nothing compares above ``NaN`` or ``Infinity``, so for those two nothing was
    replayed, yet the client was told its resume had succeeded.
    """
    mgr = _manager()
    await _observer(mgr)
    for i in range(3):  # seqs 1-3
        await mgr.broadcast(_metrics(i))
    frame = json.dumps({"type": "resume", "data": {"last_seq": "@LAST_SEQ@", "server_instance_id": mgr.server_instance_id}})
    malformed_before = _outcome_count("malformed_resume")

    ws = FakeWS(app=_app(mgr), incoming=[frame.replace('"@LAST_SEQ@"', raw_last_seq)])
    task = asyncio.create_task(training_stream_handler(ws))
    await _settle(task, ws)
    await _finish(task, ws)

    assert "resume_ok" not in ws.types()
    assert ws.frame("resume_failed")["data"] == {"reason": "malformed_resume"}
    assert _outcome_count("malformed_resume") - malformed_before == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ahead_by", [1, 100])
async def test_a_last_seq_above_the_last_seq_numbered_is_refused_out_of_range(ahead_by: int) -> None:
    """No client can hold a seq the server has not numbered yet, so ``resume_ok`` would vouch for nothing.

    Before this was checked, such a resume was answered ``resume_ok`` with nothing replayed, and
    the next ``ahead_by`` broadcasts then carried seqs at or below the client's own cursor.
    """
    mgr = _manager()
    await _observer(mgr)
    for i in range(3):  # seqs 1-3
        await mgr.broadcast(_metrics(i))
    out_of_range_before = _outcome_count("out_of_range")

    ws, task = _start_resume(mgr, last_seq=mgr.current_seq + ahead_by)
    await _settle(task, ws)
    await _finish(task, ws)

    assert "resume_ok" not in ws.types()
    assert ws.frame("resume_failed")["data"] == {"reason": "out_of_range"}
    assert _outcome_count("out_of_range") - out_of_range_before == 1


@pytest.mark.asyncio
async def test_replay_since_and_the_catch_up_refuse_a_cursor_above_the_last_seq_numbered() -> None:
    mgr = _manager()
    for i in range(3):  # seqs 1-3, nobody connected
        await mgr.broadcast(_metrics(i))

    assert mgr.replay_since(3) == []  # the newest seq itself: nothing missed
    with pytest.raises(ReplayOutOfRange):
        mgr.replay_since(4)

    ws = FakeWS()
    assert await mgr.connect_pending(ws)
    assert await mgr.promote_to_active(ws, replayed_through=4) is False
    assert ws in mgr._pending_connections  # left for the caller to close
    assert await mgr.promote_to_active(ws, replayed_through=3) is True
    assert ws in mgr._active_connections


@pytest.mark.asyncio
@pytest.mark.parametrize("negative", [-1, -5, -(10**29)])
async def test_a_negative_last_seq_is_read_as_zero_whatever_the_buffer_holds(negative: int) -> None:
    """Seqs start at 1, so a cursor below 0 holds no more than 0 does: it is answered as 0 is.

    Unclamped, the answer depended on the buffer and not on the value: ``resume_ok`` with nothing
    buffered, ``out_of_range`` once anything was, even with seq 1 still buffered, where
    ``last_seq`` 0 got a full replay.
    """
    mgr = _manager()
    await _observer(mgr)
    for i in range(3):  # seqs 1-3, all still buffered
        await mgr.broadcast(_metrics(i))

    ws, task = _start_resume(mgr, last_seq=negative)
    await _settle(task, ws)
    await mgr.broadcast(_metrics("live"))  # seq 4
    await _finish(task, ws)

    assert ws.frame("resume_ok")["data"]["replayed_count"] == 3
    assert ws.seqs() == [1, 2, 3, 4]


@pytest.mark.asyncio
@pytest.mark.parametrize("last_seq", [-1, 0])
async def test_a_negative_last_seq_is_out_of_range_once_seq_1_has_left_the_buffer_as_zero_is(last_seq: int) -> None:
    """A control for the test above: reading a negative cursor as 0 vouches for no more than 0 does."""
    mgr = _manager(max_replay_buffer_size=3)
    await _observer(mgr)
    for i in range(5):  # seqs 1-5; the buffer keeps 3-5
        await mgr.broadcast(_metrics(i))

    ws, task = _start_resume(mgr, last_seq=last_seq)
    await _settle(task, ws)
    await _finish(task, ws)

    assert "resume_ok" not in ws.types()
    assert ws.frame("resume_failed")["data"] == {"reason": "out_of_range"}


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_data", ["null", "[]", '"last_seq"', "7"])
async def test_a_resume_frame_whose_data_is_not_an_object_is_refused_as_malformed(raw_data: str) -> None:
    """``data`` that is null, a list, a string or a number carries no cursor: ``malformed_resume``, and counted.

    Before this was checked, ``data.get`` raised ``AttributeError``, which the handshake's broad
    ``except`` read as "no resume frame": a fresh connect, with no ``resume_failed`` and no
    outcome counted.
    """
    mgr = _manager()
    await _observer(mgr)
    for i in range(3):  # seqs 1-3
        await mgr.broadcast(_metrics(i))
    malformed_before = _outcome_count("malformed_resume")

    ws = FakeWS(app=_app(mgr), incoming=['{"type": "resume", "data": ' + raw_data + "}"])
    task = asyncio.create_task(training_stream_handler(ws))
    await _settle(task, ws)
    await _finish(task, ws)

    assert "resume_ok" not in ws.types()
    assert ws.frame("resume_failed")["data"] == {"reason": "malformed_resume"}
    assert _outcome_count("malformed_resume") - malformed_before == 1


# ======================================================================================
# Through a real Starlette WebSocket: the frames and the close the ASGI server is handed
# ======================================================================================


@pytest.mark.asyncio
async def test_through_a_real_socket_a_broadcast_mid_replay_reaches_the_resuming_client() -> None:
    mgr = _manager()
    await _observer(mgr)
    for i in range(4):  # seqs 1-4
        await mgr.broadcast(_metrics(i))

    async def broadcast_mid_replay(frame: dict) -> None:
        if frame.get("seq") == 2:  # the first replayed frame is being written
            await mgr.broadcast(_metrics("during-replay"))  # seq 5

    peer = AsgiResumePeer(_app(mgr), on_frame=broadcast_mid_replay)
    peer.peer_sends(_resume_frame(mgr, last_seq=1))
    task = asyncio.create_task(training_stream_handler(peer.websocket))
    await asyncio.wait_for(peer.receiving.wait(), timeout=_FINISH_TIMEOUT)
    await mgr.broadcast(_metrics("live"))  # seq 6
    peer.hang_up()
    await asyncio.wait_for(task, timeout=_FINISH_TIMEOUT)

    assert [f["type"] for f in peer.frames[:2]] == ["connection_established", "resume_ok"]
    assert peer.frames[1]["data"]["replayed_count"] == 3
    assert peer.seqs() == [2, 3, 4, 5, 6]
    assert peer.closes == []


@pytest.mark.asyncio
async def test_through_a_real_socket_an_aborted_resume_hands_the_server_a_1011_close() -> None:
    """The subject includes the send timeout, so it is short; the stalled write never returns on its own."""
    mgr = _manager(send_timeout_seconds=0.05)
    await _observer(mgr)
    for i in range(4):  # seqs 1-4
        await mgr.broadcast(_metrics(i))

    peer = AsgiResumePeer(_app(mgr), stall_seq=3)
    peer.peer_sends(_resume_frame(mgr, last_seq=1))
    task = asyncio.create_task(training_stream_handler(peer.websocket))
    await _until_serving_or_done(task, peer.receiving)
    assert task.done()  # the abort ended the handler by itself
    await _finish(task, peer)

    assert peer.seqs() == [2]
    assert peer.closes == [{"type": "websocket.close", "code": RESUME_INCOMPLETE[0], "reason": RESUME_INCOMPLETE[1]}]
    assert peer.websocket not in mgr._active_connections
    assert peer.websocket not in mgr._pending_connections


@pytest.mark.asyncio
async def test_through_a_real_socket_an_abort_on_a_dead_transport_still_ends_the_resume() -> None:
    """The connection resets mid-replay: Starlette marks the socket disconnected, so the abort's close raises.

    The abort must still end the resume: the handler returns, the socket is never promoted and its
    slots are released. Had that error escaped, ``_await_resume_frame`` would have read it as a
    failed handshake and promoted the dead socket as a fresh connect.
    """
    mgr = _manager()
    await _observer(mgr)
    for i in range(4):  # seqs 1-4
        await mgr.broadcast(_metrics(i))
    global_before = mgr._global_ws_count

    peer = AsgiResumePeer(_app(mgr), dead_on_seq=3)
    peer.peer_sends(_resume_frame(mgr, last_seq=1))
    task = asyncio.create_task(training_stream_handler(peer.websocket))
    await _until_serving_or_done(task, peer.receiving)
    assert task.done()  # ended by the abort, not left serving a dead socket
    await _finish(task, peer)

    assert peer.seqs() == [2]
    assert peer.closes == []  # nothing, a close included, can be written once the transport is gone
    assert peer.websocket not in mgr._active_connections
    assert peer.websocket not in mgr._pending_connections
    assert mgr._global_ws_count == global_before
