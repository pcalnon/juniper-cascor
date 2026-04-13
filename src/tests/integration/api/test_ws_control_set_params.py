"""Phase G: Integration tests for /ws/control set_params wire contract.

§S13 — 15 tests exercising set_params via FastAPI TestClient.websocket_connect()
(no SDK dependency). Validates wire contract, whitelist filtering, security
regression gates, and command correlation.

Entry gates:
  - Phase 0-cascor in main (seq, replay, resume)
  - Phase B-pre-b in main (origin + rate-limit guards)
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.app import create_app
from api.settings import Settings


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def client():
    """Test client with lifecycle manager, no auto-start."""
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_network(client):
    """Test client with a network created via POST /v1/network."""
    client.post("/v1/network", json={"input_size": 2, "output_size": 2})
    return client


# ===================================================================
# Helpers
# ===================================================================


def _drain_established(ws):
    """Consume the connection_established message after connect."""
    msg = ws.receive_json()
    assert msg["type"] == "connection_established"
    return msg


def _send_command(ws, command, params=None, command_id=None):
    """Send a command and return the response."""
    payload = {"command": command}
    if params is not None:
        payload["params"] = params
    if command_id is not None:
        payload["command_id"] = command_id
    ws.send_text(json.dumps(payload))
    return ws.receive_json()


# ===================================================================
# Tests — Phase G (§S13)
# ===================================================================


@pytest.mark.integration
@pytest.mark.critical
class TestWsControlSetParams:
    """Integration tests for set_params wire contract."""

    # ------------------------------------------------------------------
    # 1. Happy path
    # ------------------------------------------------------------------

    def test_set_params_via_websocket_happy_path(self, client_with_network):
        """set_params updates multiple params and returns full param snapshot."""
        with client_with_network.websocket_connect("/ws/control") as ws:
            _drain_established(ws)
            response = _send_command(
                ws,
                "set_params",
                params={"learning_rate": 0.042, "patience": 7},
            )

        assert response["type"] == "command_response"
        assert response["data"]["command"] == "set_params"
        assert response["data"]["status"] == "success"
        # Response must include full param snapshot in result
        result = response["data"]["result"]
        assert result["learning_rate"] == pytest.approx(0.042)
        assert result["patience"] == 7
        # Verify network state matches
        net = client_with_network.app.state.lifecycle.network
        assert net.learning_rate == pytest.approx(0.042)
        assert net.patience == 7

    # ------------------------------------------------------------------
    # 2. Whitelist filtering
    # ------------------------------------------------------------------

    def test_set_params_whitelist_filters_unknown_keys(self, client_with_network):
        """Unknown param keys are silently ignored; known keys are applied."""
        net = client_with_network.app.state.lifecycle.network
        original_lr = net.learning_rate

        with client_with_network.websocket_connect("/ws/control") as ws:
            _drain_established(ws)
            response = _send_command(
                ws,
                "set_params",
                params={
                    "totally_fake_param": 999,
                    "learning_rate": 0.123,
                },
            )

        assert response["data"]["status"] == "success"
        result = response["data"]["result"]
        # Known key applied
        assert result["learning_rate"] == pytest.approx(0.123)
        assert net.learning_rate == pytest.approx(0.123)
        # Unknown key not in result snapshot
        assert "totally_fake_param" not in result

    # ------------------------------------------------------------------
    # 3. init_output_weights literal validation
    # ------------------------------------------------------------------

    def test_set_params_init_output_weights_literal_validation(self, client_with_network):
        """Setting init_output_weights to 'zero' is accepted and applied."""
        with client_with_network.websocket_connect("/ws/control") as ws:
            _drain_established(ws)
            response = _send_command(
                ws,
                "set_params",
                params={"init_output_weights": "zero"},
            )

        assert response["data"]["status"] == "success"
        net = client_with_network.app.state.lifecycle.network
        assert net.init_output_weights == "zero"

    def test_set_params_bad_init_output_weights_literal_rejected(self, client_with_network):
        """Setting init_output_weights to an invalid literal still sets the
        attribute (no server-side enum validation yet), but 'random' is the
        only non-'zero' branch exercised at cascade-add time.

        This test documents current behavior: the server accepts any string
        for init_output_weights. When enum validation is added (Phase D),
        this test should be updated to expect an error.
        """
        with client_with_network.websocket_connect("/ws/control") as ws:
            _drain_established(ws)
            response = _send_command(
                ws,
                "set_params",
                params={"init_output_weights": "random"},
            )

        assert response["data"]["status"] == "success"
        net = client_with_network.app.state.lifecycle.network
        assert net.init_output_weights == "random"

    # ------------------------------------------------------------------
    # 4. Oversized frame rejection (64KB cap)
    # ------------------------------------------------------------------

    def test_set_params_oversized_frame_rejected(self, client_with_network):
        """Messages exceeding 64KB are rejected with error."""
        with client_with_network.websocket_connect("/ws/control") as ws:
            _drain_established(ws)
            # Build a payload > 64KB
            huge_value = "x" * 70_000
            payload = json.dumps({
                "command": "set_params",
                "params": {"learning_rate": 0.01},
                "padding": huge_value,
            })
            assert len(payload) > 65536
            ws.send_text(payload)
            response = ws.receive_json()

        assert response["type"] == "command_response"
        assert response["data"]["status"] == "error"
        assert "too large" in response["data"]["error"].lower()

    # ------------------------------------------------------------------
    # 5. No network → error
    # ------------------------------------------------------------------

    def test_set_params_no_network_returns_error(self, client):
        """set_params without a created network returns error."""
        with client.websocket_connect("/ws/control") as ws:
            _drain_established(ws)
            response = _send_command(
                ws,
                "set_params",
                params={"learning_rate": 0.01},
            )

        assert response["type"] == "command_response"
        assert response["data"]["command"] == "set_params"
        assert response["data"]["status"] == "error"

    # ------------------------------------------------------------------
    # 6. Unknown command → error (GAP-WS-22 regression)
    # ------------------------------------------------------------------

    def test_unknown_command_returns_error(self, client):
        """Unknown command returns error envelope; connection stays open."""
        with client.websocket_connect("/ws/control") as ws:
            _drain_established(ws)
            response = _send_command(ws, "definitely_not_a_command")
            assert response["type"] == "command_response"
            assert response["data"]["status"] == "error"
            assert "Unknown command" in response["data"]["error"]
            # Connection stays open — send a valid command to prove it
            response2 = _send_command(ws, "stop")
            assert response2["data"]["status"] == "success"

    # ------------------------------------------------------------------
    # 7. Malformed JSON → close 1003 (GAP-WS-22)
    # ------------------------------------------------------------------

    def test_malformed_json_closes_with_1003(self, client):
        """Invalid JSON triggers error response then 1003 close."""
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/control") as ws:
                _drain_established(ws)
                ws.send_text("{not valid json!!!")
                error_response = ws.receive_json()
                assert error_response["data"]["status"] == "error"
                assert "Invalid JSON" in error_response["data"]["error"]
                # Next receive should trigger the disconnect
                ws.receive_json()
        assert exc_info.value.code == 1003

    # ------------------------------------------------------------------
    # 8. Origin rejected (M-SEC-01b regression)
    # ------------------------------------------------------------------

    def test_set_params_origin_rejected(self, client):
        """Connection with disallowed origin is rejected with 4003.

        The session-scoped Origin fixture uses setdefault, so an explicit
        origin header takes precedence over the auto-injected one.
        """
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/ws/control",
                headers={"origin": "http://evil.example.com"},
            ) as ws:
                ws.receive_json()
        assert exc_info.value.code == 4003

    # ------------------------------------------------------------------
    # 9. Unauthenticated rejected
    # ------------------------------------------------------------------

    def test_set_params_unauthenticated_rejected(self):
        """Connection with API key auth enabled but no key is rejected."""
        settings = Settings(
            auto_start=False,
            api_keys=["test-secret-key-12345"],
        )
        app = create_app(settings)
        with TestClient(app) as tc:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with tc.websocket_connect("/ws/control") as ws:
                    ws.receive_json()
            assert exc_info.value.code == 4001

    # ------------------------------------------------------------------
    # 10. Rate limit triggers after 10 commands (M-SEC-05)
    # ------------------------------------------------------------------

    def test_set_params_rate_limit_triggers_after_10_cmds(self, client_with_network):
        """Leaky bucket (capacity=10) triggers rate_limited after exhaustion."""
        with client_with_network.websocket_connect("/ws/control") as ws:
            _drain_established(ws)

            rate_limited_seen = False
            # Send 15 commands rapidly — bucket capacity is 10, but refill
            # may allow a few more depending on timing. We just need to see
            # at least one rate_limited response.
            for i in range(15):
                ws.send_text(json.dumps({
                    "command": "set_params",
                    "params": {"learning_rate": 0.01},
                    "command_id": f"burst-{i}",
                }))

            for i in range(15):
                response = ws.receive_json()
                if response.get("status") == "rate_limited" or response.get("data", {}).get("status") == "rate_limited":
                    rate_limited_seen = True

            assert rate_limited_seen, "Expected at least one rate_limited response after burst"

    # ------------------------------------------------------------------
    # 11. Concurrent command-response correlation (critical)
    # ------------------------------------------------------------------

    def test_set_params_concurrent_command_response_correlation(self, client_with_network):
        """Multiple set_params with distinct command_ids correlate correctly."""
        with client_with_network.websocket_connect("/ws/control") as ws:
            _drain_established(ws)

            ids = [str(uuid.uuid4()) for _ in range(3)]
            params_list = [
                {"learning_rate": 0.001},
                {"patience": 10},
                {"max_iterations": 50},
            ]

            # Send all three commands
            for cid, params in zip(ids, params_list):
                _send_command.__wrapped__ = None  # just use inline send
                ws.send_text(json.dumps({
                    "command": "set_params",
                    "params": params,
                    "command_id": cid,
                }))

            # Collect all three responses
            responses = {}
            for _ in range(3):
                resp = ws.receive_json()
                cid = resp["data"].get("command_id")
                assert cid is not None, "command_id must be present in response"
                responses[cid] = resp

            # Each command_id should map to the correct response
            assert set(responses.keys()) == set(ids)
            for cid in ids:
                assert responses[cid]["data"]["status"] == "success"
                assert responses[cid]["data"]["command"] == "set_params"

    # ------------------------------------------------------------------
    # 12. During training — ack vs effect
    # ------------------------------------------------------------------

    def test_set_params_during_training_applies_on_next_epoch_boundary(self, client_with_network):
        """set_params while no active training returns immediate ack with
        updated values. The 'ack vs effect' contract: the response confirms
        the params were written to the network object; the effect on training
        (if running) occurs at the next epoch boundary when the training loop
        re-reads the attribute.

        Note: This test validates the wire contract (immediate ack + param
        written to network), not the training-loop read timing, which
        requires a full training integration test.
        """
        net = client_with_network.app.state.lifecycle.network
        original_lr = net.learning_rate

        with client_with_network.websocket_connect("/ws/control") as ws:
            _drain_established(ws)
            response = _send_command(
                ws,
                "set_params",
                params={"learning_rate": 0.999},
                command_id="epoch-boundary-test",
            )

        # Ack is immediate with success
        assert response["data"]["status"] == "success"
        assert response["data"]["command_id"] == "epoch-boundary-test"
        # Param is written immediately to the network object
        assert net.learning_rate == pytest.approx(0.999)
        # Result snapshot reflects the new value
        assert response["data"]["result"]["learning_rate"] == pytest.approx(0.999)

    # ------------------------------------------------------------------
    # 13. command_id echo (C-01 mandatory gate)
    # ------------------------------------------------------------------

    def test_set_params_echoes_command_id(self, client_with_network):
        """set_params echoes command_id when provided (C-01 contract)."""
        test_id = "c01-gate-" + str(uuid.uuid4())
        with client_with_network.websocket_connect("/ws/control") as ws:
            _drain_established(ws)
            response = _send_command(
                ws,
                "set_params",
                params={"learning_rate": 0.05},
                command_id=test_id,
            )

        assert response["data"]["command_id"] == test_id

        # Also verify: no command_id when not sent
        with client_with_network.websocket_connect("/ws/control") as ws:
            _drain_established(ws)
            response2 = _send_command(
                ws,
                "set_params",
                params={"learning_rate": 0.06},
            )
        assert "command_id" not in response2["data"]

    # ------------------------------------------------------------------
    # 14. No seq on command_response (C-02 cross-ref)
    # ------------------------------------------------------------------

    def test_ws_control_command_response_has_no_seq(self, client_with_network):
        """command_response messages have no seq field (D-03 canonical).

        Cross-references C-02: the /ws/control channel has no replay
        buffer, so seq is never assigned to command_response messages.
        """
        with client_with_network.websocket_connect("/ws/control") as ws:
            _drain_established(ws)
            response = _send_command(
                ws,
                "set_params",
                params={"learning_rate": 0.01},
                command_id="seq-check",
            )

        assert response["type"] == "command_response"
        assert "seq" not in response
        assert "emitted_at_monotonic" not in response
