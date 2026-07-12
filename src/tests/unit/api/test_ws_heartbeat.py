"""Phase F / C3: Unit tests for WebSocket heartbeat ping/pong.

§S12 — Tests for application-level heartbeat on /ws/training and /ws/control.
Validates ping cadence, pong cancellation, dead-connection detection, and
the full heartbeat-timeout close cycle.

C3 contract updates covered here:

* The heartbeat-timeout close now uses code 1011 (was 1006 — RFC 6455 §7.4.1
  forbids sending 1006 on the wire; the ``websockets`` server implementation
  raises ``ProtocolError`` for it, so the pre-C3 close frame never reached
  the peer in production; Starlette's TestClient bypasses wire serialization,
  which is why the old assertion passed in tests while production silently
  half-opened).
* Any well-formed inbound frame counts as liveness (tolerance for clients
  that send traffic but do not implement the pong reply).
* ``ws_heartbeat_interval_sec <= 0`` disables the heartbeat entirely.
"""

import json
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.app import create_app
from api.settings import Settings

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def client_fast_heartbeat():
    """Test client with short heartbeat interval for fast tests.

    Uses 0.3s interval + 0.5s pong timeout to avoid slow tests while
    exercising the full heartbeat cycle.
    """
    settings = Settings(
        auto_start=False,
        ws_heartbeat_interval_sec=1,
        ws_heartbeat_pong_timeout_sec=1,
    )
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_fast_heartbeat_with_network(client_fast_heartbeat):
    """Fast-heartbeat client with network created."""
    client_fast_heartbeat.post("/v1/network", json={"input_size": 2, "output_size": 2})
    return client_fast_heartbeat


# ===================================================================
# Tests — Phase F (§S12)
# ===================================================================


@pytest.mark.unit
class TestWsHeartbeat:
    """Unit tests for WebSocket heartbeat ping/pong."""

    # ------------------------------------------------------------------
    # 1. Ping sent at configured interval
    # ------------------------------------------------------------------

    def test_ping_sent_every_30_seconds(self, client_fast_heartbeat):
        """Server sends ping at configured heartbeat interval.

        Uses short interval (1s) for test speed. Verifies the ping
        message format: {"type": "ping", "ts": <float>}.
        """
        with client_fast_heartbeat.websocket_connect("/ws/training") as ws:
            # Drain non-ping connect frames (connection_established,
            # initial_status, state, plus the GAP-WS-16 initial_metrics
            # frame added on /ws/training). The exact set may grow, so
            # drain until we see the first ping rather than counting.
            ping = None
            for _ in range(10):
                frame = ws.receive_json()
                if frame.get("type") == "ping":
                    ping = frame
                    break
            assert ping is not None, "no ping received within drain budget"
            assert ping["type"] == "ping"
            assert isinstance(ping["ts"], float)
            assert ping["ts"] > 0

            # Reply with pong to keep connection alive
            ws.send_text(json.dumps({"type": "pong"}))

        # Also verify on /ws/control — drain until ping (control stream
        # may send connection_established and possibly other initial frames).
        with client_fast_heartbeat.websocket_connect("/ws/control") as ws:
            ping = None
            for _ in range(10):
                frame = ws.receive_json()
                if frame.get("type") == "ping":
                    ping = frame
                    break
            assert ping is not None, "no ping received within drain budget on /ws/control"
            assert ping["type"] == "ping"
            assert isinstance(ping["ts"], float)

            ws.send_text(json.dumps({"type": "pong"}))

    # ------------------------------------------------------------------
    # 2. Pong received cancels close
    # ------------------------------------------------------------------

    def test_pong_received_cancels_close(self, client_fast_heartbeat):
        """Replying with pong prevents heartbeat timeout close.

        Survives two full ping/pong cycles — connection stays open.
        """
        with client_fast_heartbeat.websocket_connect("/ws/training") as ws:
            # Drain non-ping connect frames; see V37b note in
            # test_ping_sent_every_30_seconds for why we look up
            # ``ping`` rather than counting.
            for _ in range(10):
                frame = ws.receive_json()
                if frame.get("type") == "ping":
                    ws.send_text(json.dumps({"type": "pong"}))
                    break

            # Survive one more ping/pong cycle
            for _ in range(10):
                frame = ws.receive_json()
                if frame.get("type") == "ping":
                    ws.send_text(json.dumps({"type": "pong"}))
                    break
            else:
                raise AssertionError("expected a second ping within drain budget")

            # Connection still alive — no disconnect

    # ------------------------------------------------------------------
    # 3. Dead connection detected via missing pong
    # ------------------------------------------------------------------

    def test_dead_connection_detected_via_missing_pong(self, client_fast_heartbeat):
        """Not replying with pong triggers a 1011 close after timeout.

        With 1s interval + 1s timeout, the connection should close
        within ~2s of the ping being sent (1s interval + 1s timeout).

        C3: the close code is 1011 (was 1006 pre-C3 — RFC 6455 §7.4.1 forbids
        sending 1006 on the wire, and the production ``websockets`` server
        implementation raises ``ProtocolError`` for it, so the old close
        frame never actually reached a real peer).
        """
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client_fast_heartbeat.websocket_connect("/ws/training") as ws:
                # Drain non-ping connect frames until we see the ping;
                # do NOT send pong (so the server times out and closes).
                ping = None
                for _ in range(10):
                    frame = ws.receive_json()
                    if frame.get("type") == "ping":
                        ping = frame
                        break
                assert ping is not None, "no ping received within drain budget"

                # Wait for server to close — next receive triggers disconnect
                ws.receive_json()

        assert exc_info.value.code == 1011
        assert "Heartbeat timeout" in (exc_info.value.reason or "")

    # ------------------------------------------------------------------
    # 4. Full cycle: 1006 → heartbeat detection → reconnect trigger
    # ------------------------------------------------------------------

    def test_broken_connection_close_triggers_heartbeat_detection_and_reconnect(self, client_fast_heartbeat):
        """Full heartbeat failure cycle on /ws/control.

        1. Connect to /ws/control
        2. Receive ping
        3. Do NOT reply with pong (and send nothing else — C3 counts any
           inbound frame as liveness)
        4. Server closes with 1011 (C3; was 1006 pre-C3, which RFC 6455
           §7.4.1 forbids on the wire)
        5. Client reconnects successfully (new connection accepted)
        """
        # Phase 1: Connect and let heartbeat timeout
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client_fast_heartbeat.websocket_connect("/ws/control") as ws:
                ws.receive_json()  # connection_established

                # Receive ping, don't reply
                ping = ws.receive_json()
                assert ping["type"] == "ping"

                # Wait for close
                ws.receive_json()

        assert exc_info.value.code == 1011
        assert "Heartbeat timeout" in (exc_info.value.reason or "")

        # Phase 2: Reconnect — fresh connection works
        with client_fast_heartbeat.websocket_connect("/ws/control") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connection_established"
            assert msg["data"]["channel"] == "control"

            # Respond to ping to keep alive
            ping = ws.receive_json()
            assert ping["type"] == "ping"
            ws.send_text(json.dumps({"type": "pong"}))

    # ------------------------------------------------------------------
    # 5. Pong on /ws/control doesn't count as a command (no rate limit)
    # ------------------------------------------------------------------

    def test_pong_not_counted_as_command(self, client_fast_heartbeat):
        """Pong messages bypass the rate limiter (not a command)."""
        with client_fast_heartbeat.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established

            # Receive ping and reply
            ping = ws.receive_json()
            assert ping["type"] == "ping"
            ws.send_text(json.dumps({"type": "pong"}))

            # Send a real command — should still succeed (pong didn't consume a token)
            ws.send_text(json.dumps({"command": "stop"}))
            response = ws.receive_json()
            assert response["data"]["status"] == "success"

    # ------------------------------------------------------------------
    # 6. Pong resets idle timeout on /ws/control
    # ------------------------------------------------------------------

    def test_heartbeat_resets_idle_timeout(self):
        """Pong responses reset the idle timeout timer.

        Uses a 2s idle timeout with 1s heartbeat interval. Without
        heartbeat resetting idle, the connection would close at 2s.
        With heartbeat, it survives past 2s.
        """
        settings = Settings(
            auto_start=False,
            ws_heartbeat_interval_sec=1,
            ws_heartbeat_pong_timeout_sec=1,
            ws_control_idle_timeout_sec=2,
        )
        app = create_app(settings)
        with TestClient(app) as tc:
            with tc.websocket_connect("/ws/control") as ws:
                ws.receive_json()  # connection_established

                # First ping at ~1s
                ping1 = ws.receive_json()
                assert ping1["type"] == "ping"
                ws.send_text(json.dumps({"type": "pong"}))

                # Second ping at ~2s — if idle timeout weren't reset,
                # connection would close here
                ping2 = ws.receive_json()
                assert ping2["type"] == "ping"
                ws.send_text(json.dumps({"type": "pong"}))

                # Still alive past the 2s idle timeout


@pytest.mark.unit
class TestWsHeartbeatToleranceC3:
    """C3: any inbound frame counts as liveness; interval <= 0 disables."""

    def test_any_inbound_frame_counts_as_liveness_on_control(self, client_fast_heartbeat):
        """A pong-less client that keeps sending commands is never reaped.

        C3 tolerance: after the ping, the client sends a ``stop`` command
        (never a pong). Pre-C3 the server would close 1s later; post-C3 the
        command itself proves liveness, so the connection survives to the
        NEXT ping.
        """
        with client_fast_heartbeat.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established

            ping1 = ws.receive_json()
            assert ping1["type"] == "ping"

            # No pong — send a real command instead.
            ws.send_text(json.dumps({"command": "stop"}))

            # Drain until the SECOND ping: its arrival proves the server did
            # not close the connection when the pong window lapsed.
            saw_response = False
            second_ping = None
            for _ in range(10):
                frame = ws.receive_json()
                if frame.get("type") == "command_response":
                    saw_response = True
                elif frame.get("type") == "ping":
                    second_ping = frame
                    break
            assert saw_response, "command_response not received"
            assert second_ping is not None, "connection did not survive to the second ping"
            ws.send_text(json.dumps({"type": "pong"}))

    def test_any_inbound_frame_counts_as_liveness_on_training(self, client_fast_heartbeat):
        """A pong-less /ws/training client that sends any frame survives the window."""
        with client_fast_heartbeat.websocket_connect("/ws/training") as ws:
            # Drain connect frames until the first ping.
            ping1 = None
            for _ in range(10):
                frame = ws.receive_json()
                if frame.get("type") == "ping":
                    ping1 = frame
                    break
            assert ping1 is not None, "no ping received within drain budget"

            # No pong — send an arbitrary well-formed frame instead.
            ws.send_text(json.dumps({"type": "noop"}))

            # Surviving to the second ping proves the tolerance.
            second_ping = None
            for _ in range(10):
                frame = ws.receive_json()
                if frame.get("type") == "ping":
                    second_ping = frame
                    break
            assert second_ping is not None, "connection did not survive to the second ping"
            ws.send_text(json.dumps({"type": "pong"}))

    def test_heartbeat_interval_zero_disables_pings_on_control(self):
        """``ws_heartbeat_interval_sec=0`` disables the heartbeat entirely.

        With the heartbeat off and a 2s idle timeout, the connection is
        closed by the idle timeout (code 1000) and no ping frame is ever
        sent — the operator escape hatch for legacy clients.
        """
        settings = Settings(
            auto_start=False,
            ws_heartbeat_interval_sec=0,
            ws_heartbeat_pong_timeout_sec=1,
            ws_control_idle_timeout_sec=2,
        )
        app = create_app(settings)
        frames = []
        with TestClient(app) as tc:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with tc.websocket_connect("/ws/control") as ws:
                    ws.receive_json()  # connection_established
                    # Nothing else should arrive until the idle-timeout close.
                    while True:
                        frames.append(ws.receive_json())
        assert exc_info.value.code == 1000
        assert all(f.get("type") != "ping" for f in frames), f"unexpected ping despite disabled heartbeat: {frames}"

    def test_heartbeat_interval_zero_disables_pings_on_training(self):
        """/ws/training with the heartbeat disabled sends no pings and stays open.

        The connection is exercised past two would-be ping intervals via
        ``subscribe_metrics`` round-trips; none of the received frames may
        be a ping.
        """
        settings = Settings(
            auto_start=False,
            ws_heartbeat_interval_sec=0,
            ws_heartbeat_pong_timeout_sec=1,
        )
        app = create_app(settings)
        with TestClient(app) as tc:
            with tc.websocket_connect("/ws/training") as ws:
                # Drain the connect burst (connection_established,
                # initial_status, state, initial_metrics).
                frames = []
                for _ in range(4):
                    frames.append(ws.receive_json())

                # Wait past two would-be heartbeat intervals, then do a
                # subscribe_metrics round-trip to prove the socket is alive.
                time.sleep(2.5)
                ws.send_text(json.dumps({"type": "subscribe_metrics", "data": {"max_count": 1}}))
                for _ in range(5):
                    frame = ws.receive_json()
                    frames.append(frame)
                    if frame.get("type") == "initial_metrics":
                        break
                assert any(f.get("type") == "initial_metrics" for f in frames), "subscribe_metrics round-trip failed"
                assert all(f.get("type") != "ping" for f in frames), f"unexpected ping despite disabled heartbeat: {frames}"
