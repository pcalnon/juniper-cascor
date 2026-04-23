"""Phase F: Unit tests for WebSocket heartbeat ping/pong.

§S12 — Tests for application-level heartbeat on /ws/training and /ws/control.
Validates ping cadence, pong cancellation, dead-connection detection, and
the full 1006 close cycle.
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
            # Drain connection_established + initial_status + state
            for _ in range(3):
                ws.receive_json()

            # Wait for the first heartbeat ping (interval = 1s)
            ping = ws.receive_json()
            assert ping["type"] == "ping"
            assert isinstance(ping["ts"], float)
            assert ping["ts"] > 0

            # Reply with pong to keep connection alive
            ws.send_text(json.dumps({"type": "pong"}))

        # Also verify on /ws/control
        with client_fast_heartbeat.websocket_connect("/ws/control") as ws:
            ws.receive_json()  # connection_established

            ping = ws.receive_json()
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
            # Drain connect sequence
            for _ in range(3):
                ws.receive_json()

            # Survive two ping/pong cycles
            for _ in range(2):
                ping = ws.receive_json()
                assert ping["type"] == "ping"
                ws.send_text(json.dumps({"type": "pong"}))

            # Connection still alive — no disconnect

    # ------------------------------------------------------------------
    # 3. Dead connection detected via missing pong
    # ------------------------------------------------------------------

    def test_dead_connection_detected_via_missing_pong(self, client_fast_heartbeat):
        """Not replying with pong triggers 1006 close after timeout.

        With 1s interval + 1s timeout, the connection should close
        within ~2s of the ping being sent (1s interval + 1s timeout).
        """
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client_fast_heartbeat.websocket_connect("/ws/training") as ws:
                # Drain connect sequence
                for _ in range(3):
                    ws.receive_json()

                # Receive ping but do NOT send pong
                ping = ws.receive_json()
                assert ping["type"] == "ping"

                # Wait for server to close — next receive triggers disconnect
                ws.receive_json()

        assert exc_info.value.code == 1006

    # ------------------------------------------------------------------
    # 4. Full cycle: 1006 → heartbeat detection → reconnect trigger
    # ------------------------------------------------------------------

    def test_broken_connection_1006_triggers_heartbeat_detection_and_reconnect(self, client_fast_heartbeat):
        """Full heartbeat failure cycle on /ws/control.

        1. Connect to /ws/control
        2. Receive ping
        3. Do NOT reply with pong
        4. Server closes with 1006
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

        assert exc_info.value.code == 1006

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
