"""The shared resume answer table: ``/ws/training`` answers every row as the table says, read off a real WebSocket.

``src/tests/fixtures/ws_resume_answers.json`` is one of two byte-identical copies of a table of resume frames and
the answers a client reads off the wire, per server state. juniper-ml holds the other copy
(``juniper-service-core/tests/fixtures/ws_resume_answers.json``), and juniper-service-core's own test drives its
handler against the same rows. juniper-ml's ``tests/test_ws_resume_answer_table_drift.py`` fails when the two
copies differ. So a change to how either service answers a resume frame fails that service's own suite, and a
change to one copy alone fails the drift gate.

This module serves cascor's own app (``create_app`` with its lifespan) with uvicorn on an ephemeral ``127.0.0.1``
port, and drives each (frame, state) pair with the ``websockets`` sync client, as the table's ``flow`` says. Per
case: a fresh ``WebSocketManager`` with the state's capacity on ``app.state.ws_manager``, the state's broadcasts,
the frame, then a ``subscribe_metrics`` sentinel, which the server answers only from its receive loop, after the
handshake and any snapshot. So no timing decides what is compared: the handshake timeout is 30 s when a frame is
sent (never reached) and 0.25 s for ``no-frame``, where the client sends nothing until the snapshot has arrived.

Compared: each reply frame's label (its ``type``, and the one field a client acts on), in wire order, during the
handshake and then live. Not compared, as the table's ``not_compared`` list says: timestamps, the instance id and
the other per-process fields of ``connection_established``, the contents of ``initial_status`` and ``state``, the
metrics history in ``initial_metrics``, and a broadcast's payload beyond its type and seq.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from api.app import create_app
from api.settings import Settings
from api.websocket.manager import WebSocketManager

pytestmark = pytest.mark.unit

TABLE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "ws_resume_answers.json"
TABLE = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
ROWS = {row["id"]: row for row in TABLE["frames"]}
CASES = [(row["id"], state) for row in TABLE["frames"] for state in TABLE["states"]]

RECV_TIMEOUT_S = 15.0
FRAME_HANDSHAKE_TIMEOUT_S = 30.0
NO_FRAME_HANDSHAKE_TIMEOUT_S = 0.25
ORIGIN = "http://localhost:8050"
SENTINEL = '{"type":"subscribe_metrics","data":{"max_count":1}}'

#: Every label the table may hold (its ``labels`` grammar).
LABEL = re.compile(r"resume_ok\(replayed_count=\d+\)|resume_failed\([a-z_]+\)|initial_status|state|initial_metrics\(current_seq=\d+\)|[a-z_]+#\d+|close\(\d+\)")


def _label(frame: dict) -> str:
    """One received frame as a table label."""
    kind = frame.get("type")
    data = frame.get("data") if isinstance(frame.get("data"), dict) else {}
    if kind == "resume_ok":
        return f"resume_ok(replayed_count={data.get('replayed_count')})"
    if kind == "resume_failed":
        return f"resume_failed({data.get('reason')})"
    if kind == "initial_metrics":
        return f"initial_metrics(current_seq={data.get('current_seq')})"
    if "seq" in frame:
        return f"{kind}#{frame['seq']}"
    return str(kind)


class _LoopbackServer:
    """uvicorn serving ``app`` on a bound 127.0.0.1 listener, on its own thread and event loop."""

    def __init__(self, app, listener: socket.socket) -> None:
        self._listener = listener
        self.port = listener.getsockname()[1]
        self.loop: asyncio.AbstractEventLoop | None = None
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_config=None, log_level="warning", access_log=False, lifespan="on"))
        self._thread = threading.Thread(target=asyncio.run, args=(self._serve(),), name="ws-resume-table-uvicorn", daemon=True)

    async def _serve(self) -> None:
        self.loop = asyncio.get_running_loop()
        await self._server.serve(sockets=[self._listener])

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 60.0
        while not self._server.started:
            assert self._thread.is_alive(), "the server thread ended during startup"
            assert time.monotonic() < deadline, "the server did not start within 60 s"
            time.sleep(0.01)

    def call(self, coroutine):
        """Run ``coroutine`` on the server's loop and wait for it."""
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout=RECV_TIMEOUT_S)

    def stop(self) -> bool:
        self._server.should_exit = True
        self._thread.join(timeout=20.0)
        return not self._thread.is_alive()


def _open(port: int):
    """The ``websockets`` sync client's connection to ``/ws/training``, to be entered with ``with``.

    websockets 17.1 deprecated using ``connect()``'s return value without a ``with``.
    """
    # websockets 15+ would route through a proxy named in the environment; this is loopback.
    options = {"proxy": None} if "proxy" in inspect.signature(connect).parameters else {}
    return connect(f"ws://127.0.0.1:{port}/ws/training", origin=ORIGIN, open_timeout=RECV_TIMEOUT_S, close_timeout=2.0, **options)


class _Client:
    """Labels every frame an open connection receives, and the close if the server closes."""

    def __init__(self, connection) -> None:
        self._socket = connection
        self.close: str | None = None

    def send(self, text: str) -> bool:
        try:
            self._socket.send(text)
        except ConnectionClosed as closed:
            self._record_close(closed)
            return False
        return True

    def receive(self) -> dict | None:
        """The next frame, or ``None`` once the server has closed; a frame that never comes fails the case."""
        try:
            raw = self._socket.recv(timeout=RECV_TIMEOUT_S)
        except TimeoutError as timed_out:
            raise AssertionError(f"nothing received within {RECV_TIMEOUT_S} s") from timed_out
        except ConnectionClosed as closed:
            self._record_close(closed)
            return None
        return json.loads(raw)

    def _record_close(self, closed) -> None:
        received = closed.rcvd
        self.close = f"close({received.code if received is not None else 1006})"

    def until_sentinel_answer(self) -> list[str]:
        """Labels of the frames before the ``initial_metrics`` that answers the sentinel.

        A snapshot is ``initial_status``, ``state``, ``initial_metrics``: an ``initial_metrics`` after an
        ``initial_status`` belongs to the snapshot, and the next one is the answer.
        """
        labels: list[str] = []
        snapshot_open = False
        while True:
            frame = self.receive()
            if frame is None:
                return [*labels, self.close]
            kind = frame.get("type")
            if kind == "initial_status":
                snapshot_open = True
            elif kind == "initial_metrics":
                if not snapshot_open:
                    return labels
                snapshot_open = False
            labels.append(_label(frame))


class _CascorApp:
    """cascor's own app, served once for the module, with a fresh manager per case."""

    def __init__(self, port: int) -> None:
        settings = Settings(_env_file=None, host="127.0.0.1", port=port, auto_start=False, auto_start_data_service=False, auto_start_canopy=False, ws_resume_handshake_timeout_s=FRAME_HANDSHAKE_TIMEOUT_S)
        self.app = create_app(settings)
        self.loop: asyncio.AbstractEventLoop | None = None

    def new_manager(self, capacity: int) -> WebSocketManager:
        # The lifespan made the manager this replaces, and set its loop the same way.
        manager = WebSocketManager(max_replay_buffer_size=capacity)
        manager.set_event_loop(self.loop)
        self.app.state.ws_manager = manager
        return manager

    def set_handshake_timeout(self, seconds: float) -> None:
        self.app.state.settings.ws_resume_handshake_timeout_s = seconds


@pytest.fixture(scope="module")
def served():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    target = _CascorApp(listener.getsockname()[1])
    server = _LoopbackServer(target.app, listener)
    server.start()
    target.loop = server.loop
    yield target, server
    assert server.stop(), "the server thread did not stop within 20 s"


def _drive(target: _CascorApp, server: _LoopbackServer, state: dict, sent: str | None) -> dict:
    """One case, as the table's ``flow`` says: the labels received during the handshake, then live."""
    manager = target.new_manager(state["replay_buffer_capacity"])
    target.set_handshake_timeout(NO_FRAME_HANDSHAKE_TIMEOUT_S if sent is None else FRAME_HANDSHAKE_TIMEOUT_S)
    for index in range(state["broadcasts_before"]):
        server.call(manager.broadcast({"type": "metrics", "timestamp": time.time(), "data": {"before": index + 1}}))
    assert manager.current_seq == state["broadcasts_before"]
    assert [message["seq"] for message in list(manager._replay_buffer)] == state["buffered_seqs"]

    with _open(server.port) as connection:
        client = _Client(connection)
        established = client.receive()
        assert established is not None and established["type"] == "connection_established"
        assert established["data"]["replay_buffer_capacity"] == state["replay_buffer_capacity"]
        if sent is None:
            handshake = [_label(frame) for frame in (client.receive() for _ in range(3)) if frame is not None]
            client.send(SENTINEL)
            handshake.extend(client.until_sentinel_answer())
        else:
            client.send(sent.replace("@ID@", established["data"]["server_instance_id"]))
            client.send(SENTINEL)
            handshake = client.until_sentinel_answer()
        live: list[str] = []
        if client.close is None:
            server.call(manager.broadcast({"type": "metrics", "timestamp": time.time(), "data": {"live": True}}))
            if client.send(SENTINEL):
                live = client.until_sentinel_answer()
        return {"handshake": handshake, "live": live}


@pytest.mark.parametrize(("frame_id", "state"), CASES, ids=[f"{state}-{frame_id}" for frame_id, state in CASES])
def test_the_handler_answers_each_row_as_the_shared_table_says(served, frame_id: str, state: str) -> None:
    target, server = served
    row = ROWS[frame_id]
    assert _drive(target, server, TABLE["states"][state], row["sent"]) == row["answers"][state]


def test_the_table_answers_sixty_frames_in_four_states() -> None:
    assert list(TABLE["states"]) == ["fresh", "buffered5", "evicted", "disabled"]
    assert len(TABLE["frames"]) == 60
    assert len(ROWS) == 60, "frame ids must be unique"
    for row in TABLE["frames"]:
        assert list(row["answers"]) == list(TABLE["states"]), row["id"]
        assert row["sent"] is None or isinstance(row["sent"], str), row["id"]


def test_every_answer_label_is_in_the_tables_grammar() -> None:
    for row in TABLE["frames"]:
        for state, answer in row["answers"].items():
            assert list(answer) == ["handshake", "live"], (row["id"], state)
            for item in [*answer["handshake"], *answer["live"]]:
                assert LABEL.fullmatch(item), (row["id"], state, item)
