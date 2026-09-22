"""Wire-level check of the F-CASCOR-004 fix: a real uvicorn server, real WebSocket clients, TCP loopback.

Project: juniper-cascor
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-22
Status: ad-hoc — investigation (verifies the F-CASCOR-004 fix where the unit tier cannot reach)
Retire when: F-CASCOR-004 is closed in the juniper-ml evidence ledger and the wire check is no longer wanted
Related: finding F-CASCOR-004 in juniper-ml
         notes/JUNIPER_2026-08-09_JUNIPER-CANOPY_E2E-VALIDATION-EVIDENCE.md;
         src/tests/unit/api/test_ws_broadcast_fault_isolation.py (the unit tier, which
         stops at an in-memory ASGI transport)

The unit tests drive a real Starlette ``WebSocket`` over an in-memory transport, so
they cannot show what a peer sees on the wire. This probe serves the real app with
uvicorn (its default ``websockets`` protocol) and connects with the ``websockets``
client library:

A. an unserializable broadcast (NumPy scalars in a ``state`` frame, the F-CASCOR-003
   shape) drops nobody: both clients stay open and receive the next broadcast, and
   the transport counters book it as a message fault, not a send failure;
B. a per-connection send failure -- injected on ONE server-side socket, since a
   genuine one cannot be provoked on demand -- reaches that client as a close frame
   with code 1011, the other client keeps receiving, the dropped client's server-side
   handler returns, and uvicorn logs no "Exception in ASGI application";
C. the dropped client can reconnect: its admission slots were released exactly once.

Usage, from the repo root in the JuniperCascor1 env:

    python util/ad-hoc/2026-09-22_f_cascor_004_live_ws_probe.py

Exit 0 = every check passed; 1 = at least one failed. On a tree without the fix,
A fails (both clients are forgotten without a close) and B / C cannot run.
"""

import asyncio
import json
import logging
import socket
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from websockets.asyncio.client import ClientConnection, connect  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402

from api.app import create_app  # noqa: E402
from api.settings import Settings  # noqa: E402

RECV_TIMEOUT_S = 3.0
POLL_ATTEMPTS = 100
POLL_INTERVAL_S = 0.05


class _LogCapture(logging.Handler):
    """Collects every message logged to the logger it is attached to."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class _Client:
    """One ``/ws/training`` subscriber, reading frames by type and skipping heartbeats."""

    def __init__(self, conn: ClientConnection) -> None:
        self.conn = conn

    async def next_of_type(self, wanted: str) -> Optional[dict]:
        """The next frame of type ``wanted``; ``None`` on timeout or close."""
        for _ in range(100):
            try:
                raw = await asyncio.wait_for(self.conn.recv(), timeout=RECV_TIMEOUT_S)
            except (asyncio.TimeoutError, ConnectionClosed):
                return None
            frame = json.loads(raw)
            if frame.get("type") == wanted:
                return frame
        return None


class _Report:
    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
        return bool(ok)

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.results)


async def _poll(predicate) -> bool:
    for _ in range(POLL_ATTEMPTS):
        if predicate():
            return True
        await asyncio.sleep(POLL_INTERVAL_S)
    return predicate()


async def _subscribe(url: str) -> _Client:
    client = _Client(await connect(url))
    for frame_type in ("connection_established", "initial_status", "state"):
        if await client.next_of_type(frame_type) is None:
            raise RuntimeError(f"handshake frame {frame_type!r} never arrived")
    return client


async def main() -> int:
    report = _Report()
    app = create_app(Settings(auto_start=False, ws_resume_handshake_timeout_s=0.1, ws_initial_metrics_count=0))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    serve_task = asyncio.create_task(server.serve(sockets=[sock]))
    if not await _poll(lambda: server.started):
        print("uvicorn did not start", flush=True)
        return 1
    asgi_errors = _LogCapture()
    logging.getLogger("uvicorn.error").addHandler(asgi_errors)
    mgr = app.state.ws_manager
    url = f"ws://127.0.0.1:{port}/ws/training"
    print(f"serving on 127.0.0.1:{port}", flush=True)

    try:
        a, b = await _subscribe(url), await _subscribe(url)
        report.check("setup: two active subscribers", await _poll(lambda: mgr.connection_count == 2), f"active={mgr.connection_count}")

        # A -- an unserializable broadcast is the message's fault.
        await mgr.broadcast({"type": "state", "data": {"status": "Started", "patience": np.int64(3), "multiprocessing": np.bool_(True)}})
        await mgr.broadcast({"type": "metrics", "data": {"epoch": 1}})
        got_a, got_b = await a.next_of_type("metrics"), await b.next_of_type("metrics")
        report.check("A: both clients received the next broadcast", got_a is not None and got_b is not None, f"a={got_a} b={got_b}")
        stats = mgr.transport_stats()
        report.check("A: counted as unserializable", stats.get("unserializable_messages_total") == 1, f"unserializable_messages_total={stats.get('unserializable_messages_total')}")
        report.check("A: not counted as a send failure", stats["send_failures"] == 0, f"send_failures={stats['send_failures']}")
        report.check("A: both still active", stats["active_connections"] == 2, f"active_connections={stats['active_connections']}")

        # B -- a per-connection failure closes that client, on the wire.
        a_port = a.conn.local_address[1]
        server_side_a = next((ws for ws in mgr._active_connections if ws.client.port == a_port), None)
        if not report.check("B: found a's server-side socket", server_side_a is not None):
            print("SOME CHECKS FAILED (B and C need the subscriber A should have kept)", flush=True)
            return 1

        async def _failing_send_json(data, mode="text"):
            raise RuntimeError("injected per-connection transport failure")

        server_side_a.send_json = _failing_send_json
        await mgr.broadcast({"type": "metrics", "data": {"epoch": 2}})
        try:
            await asyncio.wait_for(a.conn.wait_closed(), timeout=RECV_TIMEOUT_S)
        except asyncio.TimeoutError:
            pass
        report.check("B: dropped client got close 1011", a.conn.close_code == 1011, f"close_code={a.conn.close_code} reason={a.conn.close_reason!r}")
        got_b2 = await b.next_of_type("metrics")
        report.check("B: the other client got the broadcast", got_b2 is not None and got_b2["data"] == {"epoch": 2}, f"b={got_b2}")
        report.check("B: dropped client's handler returned", await _poll(lambda: len(mgr._endpoint_connections["training"]) == 1), f"training bucket={len(mgr._endpoint_connections['training'])}")
        report.check("B: exactly one admission slot held", mgr._global_ws_count == 1, f"global={mgr._global_ws_count} per_ip={mgr._per_ip_counts}")
        report.check("B: no unhandled handler exception", not any("Exception in ASGI application" in m for m in asgi_errors.messages), f"uvicorn.error={asgi_errors.messages}")
        stats = mgr.transport_stats()
        report.check("B: one send failure counted", stats["send_failures"] == 1, f"send_failures={stats['send_failures']}")

        # C -- the dropped client can come back.
        a2 = await _subscribe(url)
        report.check("C: dropped client reconnected", await _poll(lambda: mgr.connection_count == 2), f"active={mgr.connection_count}")
        await mgr.broadcast({"type": "metrics", "data": {"epoch": 3}})
        got_a2 = await a2.next_of_type("metrics")
        report.check("C: reconnected client receives broadcasts", got_a2 is not None and got_a2["data"] == {"epoch": 3}, f"a2={got_a2}")

        await a2.conn.close()
        await b.conn.close()
        report.check("teardown: every slot released", await _poll(lambda: mgr._global_ws_count == 0 and not mgr._per_ip_counts), f"global={mgr._global_ws_count} per_ip={mgr._per_ip_counts}")
    finally:
        server.should_exit = True
        await serve_task

    print("ALL CHECKS PASSED" if report.passed else "SOME CHECKS FAILED", flush=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
