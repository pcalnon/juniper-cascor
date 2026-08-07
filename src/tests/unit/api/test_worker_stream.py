"""Tests for /ws/v1/workers WebSocket endpoint handler."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import WebSocketDisconnect

from api.websocket.worker_stream import _handle_registration, _handle_task_result, _make_send_callback, _message_loop, _try_dispatch_task, worker_stream_handler
from api.workers.coordinator import WorkerCoordinator
from api.workers.protocol import BinaryFrame, MessageType, WorkerProtocol
from api.workers.registry import WorkerRegistry


def _make_websocket(headers=None, app_state=None):
    """Create a mock WebSocket with configurable headers and app state."""
    ws = AsyncMock()
    ws.headers = headers or {}

    state = MagicMock()
    state.api_key_auth = None
    state.worker_coordinator = None
    state.worker_registry = None
    # SEC-F19 D4: the handler reserves a stack-global admission slot via the
    # ws_manager (try_admit) before accepting and releases it on teardown
    # (release_admission). Provide awaitable doubles so the mock-based unit
    # tests exercise (and pass through) that admission path; the sync gauge
    # methods stay MagicMock no-ops.
    state.ws_manager.try_admit = AsyncMock(return_value=True)
    state.ws_manager.release_admission = AsyncMock()
    if app_state:
        for key, val in app_state.items():
            setattr(state, key, val)

    app = MagicMock()
    app.state = state
    ws.app = app
    return ws


@pytest.fixture
def registry():
    return WorkerRegistry(heartbeat_timeout=30.0)


@pytest.fixture
def coordinator(registry):
    coord = WorkerCoordinator(registry=registry, task_reassignment_timeout=5.0)
    yield coord
    coord.shutdown()


@pytest.mark.unit
class TestWorkerStreamAuth:
    """Test authentication on worker WebSocket endpoint."""

    @pytest.mark.asyncio
    async def test_reject_origin_header(self):
        """Connections with Origin header are rejected (machine-to-machine only)."""
        ws = _make_websocket(headers={"origin": "http://evil.com"})
        await worker_stream_handler(ws)
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4003

    @pytest.mark.asyncio
    async def test_reject_without_api_key(self):
        """Connections without API key are rejected when auth is enabled."""
        auth = MagicMock()
        auth.enabled = True
        auth.validate = MagicMock(return_value=False)

        ws = _make_websocket(app_state={"api_key_auth": auth})
        await worker_stream_handler(ws)
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4001

    @pytest.mark.asyncio
    async def test_reject_no_coordinator(self):
        """Connections are rejected when worker system is not initialized."""
        ws = _make_websocket()
        await worker_stream_handler(ws)
        # Should close BEFORE accepting because coordinator is None
        ws.accept.assert_not_awaited()
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4004


@pytest.mark.unit
class TestHandleRegistration:
    """Test worker registration message handling."""

    @pytest.mark.asyncio
    async def test_valid_registration(self, registry):
        """Valid registration message registers the worker with a
        server-assigned ID, not the client-proposed one (CR-026)."""
        ws = AsyncMock()
        ws.receive_text = AsyncMock(return_value=json.dumps(WorkerProtocol.build_register("w1", {"cpu_cores": 4})))

        worker_id = await _handle_registration(ws, registry)
        # Server assigns a UUID-based worker_id; client's "w1" proposal is ignored as identity
        assert worker_id != "w1"
        assert worker_id.startswith("worker-")
        assert registry.worker_count == 1
        # Client-proposed name is retained as a display label only
        reg = registry.get(worker_id)
        assert reg is not None
        assert reg.client_name == "w1"

    @pytest.mark.asyncio
    async def test_registration_ack_returns_server_assigned_id(self, registry):
        """The registration_ack payload returns the server-assigned worker_id,
        not the client's proposal (CR-026)."""
        ws = AsyncMock()
        ws.receive_text = AsyncMock(return_value=json.dumps(WorkerProtocol.build_register("my-cool-name", {"cpu_cores": 4})))

        worker_id = await _handle_registration(ws, registry)
        assert worker_id is not None
        # The registration_ack send_json call receives the SERVER-ASSIGNED id
        ack_call_args = None
        for call in ws.send_json.call_args_list:
            payload = call.args[0] if call.args else call.kwargs.get("payload")
            if isinstance(payload, dict) and payload.get("type") == "registration_ack":
                ack_call_args = payload
                break
        assert ack_call_args is not None, "No registration_ack sent"
        assert ack_call_args["worker_id"] == worker_id
        assert ack_call_args["worker_id"] != "my-cool-name"
        assert ack_call_args["data"]["client_name"] == "my-cool-name"

    @pytest.mark.asyncio
    async def test_two_workers_with_same_client_name_get_distinct_ids(self, registry):
        """Two workers proposing the same client-side name are assigned
        distinct server IDs — impersonation is impossible (CR-026)."""
        ws1 = AsyncMock()
        ws1.receive_text = AsyncMock(return_value=json.dumps(WorkerProtocol.build_register("victim-worker", {"cpu_cores": 4})))
        ws2 = AsyncMock()
        ws2.receive_text = AsyncMock(return_value=json.dumps(WorkerProtocol.build_register("victim-worker", {"cpu_cores": 8})))

        wid1 = await _handle_registration(ws1, registry)
        wid2 = await _handle_registration(ws2, registry)

        assert wid1 is not None and wid2 is not None
        assert wid1 != wid2
        assert registry.worker_count == 2
        # Both workers retain their (identical) client_name for auditing,
        # but the registry keys them by distinct server-assigned IDs.
        assert registry.get(wid1).client_name == "victim-worker"
        assert registry.get(wid2).client_name == "victim-worker"

    @pytest.mark.asyncio
    async def test_invalid_json(self, registry):
        """Invalid JSON closes the connection."""
        ws = AsyncMock()
        ws.receive_text = AsyncMock(return_value="not json{{{")

        worker_id = await _handle_registration(ws, registry)
        assert worker_id is None
        ws.close.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", [None, [], 42, "register", True])
    async def test_non_object_json_rejected(self, registry, payload):
        """Non-object JSON must close cleanly — not AttributeError on ``.get``."""
        ws = AsyncMock()
        ws.receive_text = AsyncMock(return_value=json.dumps(payload))

        worker_id = await _handle_registration(ws, registry)
        assert worker_id is None
        ws.send_json.assert_awaited()
        err = ws.send_json.call_args[0][0]
        assert err["type"] == "error"
        assert "JSON object" in err["error"]
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4008
        assert registry.worker_count == 0

    @pytest.mark.asyncio
    async def test_wrong_message_type(self, registry):
        """Non-registration first message closes with structured error + code 4007."""
        ws = AsyncMock()
        ws.receive_text = AsyncMock(return_value=json.dumps({"type": "heartbeat", "worker_id": "w1"}))

        worker_id = await _handle_registration(ws, registry)
        assert worker_id is None
        # Ops/workers distinguish "expected REGISTER" (4007) from 4005/4006/4008/4013.
        ws.send_json.assert_awaited_once()
        error_msg = ws.send_json.call_args[0][0]
        assert error_msg["type"] == "error"
        assert "registration" in error_msg["error"].lower()
        ws.close.assert_awaited_once()
        assert ws.close.call_args.kwargs["code"] == 4007
        assert ws.close.call_args.kwargs["reason"] == "Expected registration"

    @pytest.mark.asyncio
    async def test_missing_fields(self, registry):
        """Registration with missing fields closes with structured error + code 4008."""
        ws = AsyncMock()
        ws.receive_text = AsyncMock(return_value=json.dumps({"type": "register"}))  # Missing worker_id and capabilities

        worker_id = await _handle_registration(ws, registry)
        assert worker_id is None
        # Invalid-registration (4008) must stay distinct from registry-full (4013).
        ws.send_json.assert_awaited_once()
        error_msg = ws.send_json.call_args[0][0]
        assert error_msg["type"] == "error"
        assert "invalid registration" in error_msg["error"].lower()
        ws.close.assert_awaited_once()
        assert ws.close.call_args.kwargs["code"] == 4008
        assert ws.close.call_args.kwargs["reason"] == "Invalid registration"


@pytest.mark.unit
class TestHandleTaskResult:
    """Test task result handling in the message loop."""

    @pytest.mark.asyncio
    async def test_accept_result_with_tensors(self, coordinator, registry):
        """Task result with matching binary frames is accepted."""
        registry.register("w1", {})
        tensors = {
            "candidate_input": np.zeros((10, 4), dtype=np.float32),
            "y": np.zeros((10, 1), dtype=np.float32),
            "residual_error": np.zeros((10, 1), dtype=np.float32),
        }
        task_ids = coordinator.submit_tasks(
            "r1",
            [{"candidate_index": 0, "candidate_data": {}, "training_params": {}}],
            tensors,
        )
        coordinator.get_next_assignment("w1")

        msg = {
            "type": "task_result",
            "task_id": task_ids[0],
            "candidate_id": 0,
            "candidate_uuid": "uuid",
            "correlation": 0.85,
            "success": True,
            "epochs_completed": 10,
            "activation_name": "sigmoid",
            "all_correlations": [0.85],
            "numerator": 1.0,
            "denominator": 2.0,
            "best_corr_idx": 9,
            "error_message": None,
            "tensor_manifest": {
                "weights": {"shape": [4], "dtype": "float32"},
                "bias": {"shape": [1], "dtype": "float32"},
            },
        }

        # Mock WebSocket to return binary frames in sequence
        weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        bias = np.array([0.5], dtype=np.float32)
        ws = AsyncMock()
        ws.receive = AsyncMock(
            side_effect=[
                {"bytes": BinaryFrame.encode(weights)},
                {"bytes": BinaryFrame.encode(bias)},
            ]
        )

        await _handle_task_result(ws, "w1", msg, coordinator)

        # Check that result was accepted
        ws.send_json.assert_awaited()
        ack = ws.send_json.call_args[0][0]
        assert ack["status"] == "accepted"


@pytest.mark.unit
class TestTryDispatchTask:
    """Test proactive task dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_when_tasks_available(self, coordinator, registry):
        """Tasks are dispatched when available."""
        registry.register("w1", {})
        tensors = {
            "candidate_input": np.zeros((10, 4), dtype=np.float32),
            "y": np.zeros((10, 1), dtype=np.float32),
            "residual_error": np.zeros((10, 1), dtype=np.float32),
        }
        coordinator.submit_tasks(
            "r1",
            [{"candidate_index": 0, "candidate_data": {}, "training_params": {}}],
            tensors,
        )

        ws = AsyncMock()
        await _try_dispatch_task(ws, "w1", coordinator)

        # Should have sent JSON + binary frames
        ws.send_json.assert_awaited_once()
        assert ws.send_bytes.await_count == 3  # 3 tensors

    @pytest.mark.asyncio
    async def test_no_dispatch_when_empty(self, coordinator, registry):
        """No dispatch when no tasks pending."""
        registry.register("w1", {})
        ws = AsyncMock()
        await _try_dispatch_task(ws, "w1", coordinator)
        ws.send_json.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_json_failure_requeues_assigned_task(self, coordinator, registry):
        """send_json failure after assignment frees the worker and requeues immediately."""
        registry.register("w1", {})
        registry.register("w2", {})
        tensors = {
            "candidate_input": np.zeros((10, 4), dtype=np.float32),
            "y": np.zeros((10, 1), dtype=np.float32),
            "residual_error": np.zeros((10, 1), dtype=np.float32),
        }
        task_ids = coordinator.submit_tasks(
            "r1",
            [{"candidate_index": 0, "candidate_data": {}, "training_params": {}}],
            tensors,
        )

        ws = AsyncMock()
        ws.send_json = AsyncMock(side_effect=RuntimeError("socket closed"))

        await _try_dispatch_task(ws, "w1", coordinator)

        assert registry.get("w1").idle is True
        assert task_ids[0] in coordinator._unassigned_tasks
        assert coordinator._pending_tasks[task_ids[0]].assigned_worker_id is None

        peer = coordinator.get_next_assignment("w2")
        assert peer is not None
        assert peer[0]["task_id"] == task_ids[0]

    @pytest.mark.asyncio
    async def test_send_bytes_failure_requeues_after_partial_send(self, coordinator, registry):
        """Failure mid binary-frame send also rolls back the assignment."""
        registry.register("w1", {})
        tensors = {
            "candidate_input": np.zeros((10, 4), dtype=np.float32),
            "y": np.zeros((10, 1), dtype=np.float32),
            "residual_error": np.zeros((10, 1), dtype=np.float32),
        }
        task_ids = coordinator.submit_tasks(
            "r1",
            [{"candidate_index": 0, "candidate_data": {}, "training_params": {}}],
            tensors,
        )

        ws = AsyncMock()
        ws.send_json = AsyncMock()
        ws.send_bytes = AsyncMock(side_effect=RuntimeError("frame write failed"))

        await _try_dispatch_task(ws, "w1", coordinator)

        assert registry.get("w1").idle is True
        assert task_ids[0] in coordinator._unassigned_tasks
        ws.send_json.assert_awaited_once()


@pytest.mark.unit
class TestHeartbeatDispatch:
    """ISSUE-319 (defect #5 — the dual-path unlock): an idle worker must pick up tasks
    submitted AFTER it connected. _try_dispatch_task otherwise runs only at connect and
    after a task result, so candidate tasks submitted mid-session sit unassigned until the
    remote collection budget expires and the round falls back to local retry — the stall."""

    @pytest.mark.asyncio
    async def test_heartbeat_delivers_task_submitted_after_connect(self, registry, coordinator):
        """The exact #319 scenario: worker connects idle with no work; a candidate task is
        submitted afterwards; the worker's next heartbeat must deliver it."""
        registry.register("w1", {})
        hb = json.dumps({"type": "heartbeat", "worker_id": "w1"})
        tensors = {
            "candidate_input": np.zeros((10, 4), dtype=np.float32),
            "y": np.zeros((10, 1), dtype=np.float32),
            "residual_error": np.zeros((10, 1), dtype=np.float32),
        }
        calls = {"n": 0}

        def receive_seq(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                # Submit only NOW — after the connect-time dispatch has already run against
                # an empty queue — so ONLY the heartbeat path can deliver this task.
                coordinator.submit_tasks(
                    "r1",
                    [{"candidate_index": 7, "candidate_data": {}, "training_params": {}}],
                    tensors,
                )
                return {"text": hb}
            raise WebSocketDisconnect()

        ws = AsyncMock()
        ws.receive = AsyncMock(side_effect=receive_seq)

        with pytest.raises(WebSocketDisconnect):
            await _message_loop(ws, "w1", registry, coordinator)

        # The heartbeat pulled the task off the unassigned queue and sent it to the worker.
        assert coordinator.has_pending_tasks() is False
        assert not registry.get("w1").idle, "worker should be marked busy after assignment"

    @pytest.mark.asyncio
    async def test_heartbeat_skips_dispatch_for_busy_worker(self, registry, coordinator):
        """Guard: a heartbeat arriving mid-task must not pull a second task onto a busy
        worker (get_next_assignment does not itself refuse a busy worker)."""
        registry.register("w1", {})
        assert registry.assign_task("w1", "t-active")  # mark the worker busy
        assert not registry.get("w1").idle
        hb = json.dumps({"type": "heartbeat", "worker_id": "w1"})

        ws = AsyncMock()
        ws.receive = AsyncMock(side_effect=[{"text": hb}, WebSocketDisconnect()])

        with patch("api.websocket.worker_stream._try_dispatch_task", new=AsyncMock()) as dispatch:
            with pytest.raises(WebSocketDisconnect):
                await _message_loop(ws, "w1", registry, coordinator)
            # connect-time dispatch fires once; the heartbeat must skip dispatch (busy worker).
            assert dispatch.await_count == 1


@pytest.mark.unit
class TestWorkerStreamHandlerFullFlow:
    """Test the full worker_stream_handler flow covering try/except/finally."""

    @pytest.mark.asyncio
    async def test_successful_connect_then_disconnect(self, registry, coordinator):
        """Full handler: auth passes, registration succeeds, WebSocketDisconnect during message loop."""
        reg_msg = json.dumps(WorkerProtocol.build_register("w1", {"cpu_cores": 4}))

        ws = _make_websocket(
            app_state={
                "worker_coordinator": coordinator,
                "worker_registry": registry,
            }
        )
        ws.receive_text = AsyncMock(return_value=reg_msg)
        # _message_loop will call receive(); raise disconnect on the first receive
        ws.receive = AsyncMock(side_effect=WebSocketDisconnect())

        # get_next_assignment returns None so _try_dispatch_task is a no-op
        coordinator.get_next_assignment = MagicMock(return_value=None)

        await worker_stream_handler(ws)

        ws.accept.assert_awaited_once()
        ws.send_json.assert_awaited()  # connection_established + registration_ack

    @pytest.mark.asyncio
    async def test_unexpected_exception_in_message_loop(self, registry, coordinator):
        """Full handler: unexpected exception is caught and worker is cleaned up."""
        reg_msg = json.dumps(WorkerProtocol.build_register("w1", {"cpu_cores": 4}))

        ws = _make_websocket(
            app_state={
                "worker_coordinator": coordinator,
                "worker_registry": registry,
            }
        )
        ws.receive_text = AsyncMock(return_value=reg_msg)
        ws.receive = AsyncMock(side_effect=RuntimeError("boom"))
        coordinator.get_next_assignment = MagicMock(return_value=None)

        await worker_stream_handler(ws)

        # Worker should have been cleaned up in finally block
        ws.accept.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_registration_fails_returns_none(self, registry, coordinator):
        """Full handler: if registration fails, handler exits without cleanup."""
        ws = _make_websocket(
            app_state={
                "worker_coordinator": coordinator,
                "worker_registry": registry,
            }
        )
        # Send invalid JSON for registration
        ws.receive_text = AsyncMock(return_value="not valid json{{{")

        await worker_stream_handler(ws)

        ws.accept.assert_awaited_once()
        ws.close.assert_awaited()  # closed due to invalid registration

    @pytest.mark.asyncio
    async def test_auth_valid_key_passes(self, registry, coordinator):
        """Auth enabled with a valid key proceeds past authentication."""
        auth = MagicMock()
        auth.enabled = True
        auth.validate = MagicMock(return_value=True)

        reg_msg = json.dumps(WorkerProtocol.build_register("w1", {"cpu_cores": 2}))

        ws = _make_websocket(
            headers={"X-API-Key": "valid-key"},
            app_state={
                "api_key_auth": auth,
                "worker_coordinator": coordinator,
                "worker_registry": registry,
            },
        )
        ws.receive_text = AsyncMock(return_value=reg_msg)
        ws.receive = AsyncMock(side_effect=WebSocketDisconnect())
        coordinator.get_next_assignment = MagicMock(return_value=None)

        await worker_stream_handler(ws)

        auth.validate.assert_called_once_with("valid-key")
        ws.accept.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_websocket_disconnect_with_unknown_worker_id(self, registry, coordinator):
        """WebSocketDisconnect before registration completes logs 'unknown'."""
        ws = _make_websocket(
            app_state={
                "worker_coordinator": coordinator,
                "worker_registry": registry,
            }
        )
        ws.receive_text = AsyncMock(side_effect=WebSocketDisconnect())

        await worker_stream_handler(ws)

        ws.accept.assert_awaited_once()


@pytest.mark.unit
class TestHandleRegistrationExtraEdgeCases:
    """Test additional edge cases in _handle_registration."""

    @pytest.mark.asyncio
    async def test_registration_message_too_large(self, registry):
        """Registration messages exceeding 64KB are rejected."""
        ws = AsyncMock()
        # Create a message larger than 64KB
        large_msg = "x" * 65537
        ws.receive_text = AsyncMock(return_value=large_msg)

        worker_id = await _handle_registration(ws, registry)

        assert worker_id is None
        ws.send_json.assert_awaited_once()
        error_msg = ws.send_json.call_args[0][0]
        assert error_msg["type"] == "error"
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4005

    @pytest.mark.asyncio
    async def test_registry_at_capacity_sends_error_and_closes_4013(self):
        """Registry saturation rejects with a structured error frame and close 4013.

        Distinct from 4008 (invalid registration) so operators can tell
        capacity pressure from schema failures — important for websockets
        major bumps where close-frame handling can regress.
        """
        registry = WorkerRegistry(heartbeat_timeout=30.0, max_workers=1)
        registry.register("worker-already-here", {"cpu_cores": 2})

        ws = AsyncMock()
        ws.receive_text = AsyncMock(return_value=json.dumps(WorkerProtocol.build_register("new-worker", {"cpu_cores": 4})))

        worker_id = await _handle_registration(ws, registry)

        assert worker_id is None
        assert registry.worker_count == 1
        ws.send_json.assert_awaited_once()
        error_msg = ws.send_json.call_args[0][0]
        assert error_msg["type"] == "error"
        assert "capacity" in error_msg["error"].lower()
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4013
        assert "capacity" in ws.close.call_args[1]["reason"].lower()


@pytest.mark.unit
class TestWorkerStreamAdmissionReject:
    """SEC-F19 D4: try_admit=False must fail closed before accept/release."""

    @pytest.mark.asyncio
    async def test_try_admit_false_rejects_without_accept(self, registry, coordinator):
        """Over-cap admission returns early: no accept, no release_admission."""

        async def _reject_and_close(websocket, *, endpoint, identity=None):
            await websocket.close(code=1013, reason="Maximum connections reached")
            return False

        ws = _make_websocket(
            app_state={
                "worker_coordinator": coordinator,
                "worker_registry": registry,
            }
        )
        ws.app.state.ws_manager.try_admit = AsyncMock(side_effect=_reject_and_close)
        ws.app.state.ws_manager.release_admission = AsyncMock()

        await worker_stream_handler(ws)

        ws.app.state.ws_manager.try_admit.assert_awaited_once()
        assert ws.app.state.ws_manager.try_admit.await_args.kwargs["endpoint"] == "workers"
        ws.accept.assert_not_awaited()
        ws.app.state.ws_manager.release_admission.assert_not_awaited()
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 1013


@pytest.mark.unit
class TestMessageLoop:
    """Test the _message_loop function covering all message type branches."""

    @pytest.mark.asyncio
    async def test_heartbeat_message(self, registry, coordinator):
        """Heartbeat messages trigger registry.heartbeat and send a response."""
        registry.register("w1", {})
        coordinator.get_next_assignment = MagicMock(return_value=None)

        hb_msg = json.dumps({"type": MessageType.HEARTBEAT, "worker_id": "w1"})
        ws = AsyncMock()
        ws.receive = AsyncMock(
            side_effect=[
                {"text": hb_msg},
                WebSocketDisconnect(),
            ]
        )

        with pytest.raises(WebSocketDisconnect):
            await _message_loop(ws, "w1", registry, coordinator)

        # Should have sent a heartbeat response
        calls = ws.send_json.call_args_list
        # Find the heartbeat ack among all send_json calls
        hb_responses = [c for c in calls if c[0][0].get("type") == MessageType.HEARTBEAT]
        assert len(hb_responses) == 1

    @pytest.mark.asyncio
    async def test_heartbeat_forwards_enriched_fields(self, registry, coordinator):
        """METRICS-MON R1.3 / seed-04: enriched fields in the heartbeat payload reach the registry."""
        registry.register("w1", {})
        coordinator.get_next_assignment = MagicMock(return_value=None)

        hb_msg = json.dumps(
            {
                "type": MessageType.HEARTBEAT,
                "worker_id": "w1",
                "in_flight_tasks": 3,
                "last_task_completed_at": 1745816400.0,
                "rss_mb": 256.5,
                "tasks_completed": 17,
                "tasks_failed": 1,
            }
        )
        ws = AsyncMock()
        ws.receive = AsyncMock(side_effect=[{"text": hb_msg}, WebSocketDisconnect()])

        with pytest.raises(WebSocketDisconnect):
            await _message_loop(ws, "w1", registry, coordinator)

        reg = registry.get("w1")
        assert reg is not None
        assert reg.in_flight_tasks == 3
        assert reg.last_task_completed_at == 1745816400.0
        assert reg.rss_mb == 256.5
        assert reg.tasks_completed == 17
        assert reg.tasks_failed == 1

    @pytest.mark.asyncio
    async def test_unknown_message_type(self, registry, coordinator):
        """Unknown message types trigger an error response."""
        registry.register("w1", {})
        coordinator.get_next_assignment = MagicMock(return_value=None)

        unknown_msg = json.dumps({"type": "some_unknown_type"})
        ws = AsyncMock()
        ws.receive = AsyncMock(
            side_effect=[
                {"text": unknown_msg},
                WebSocketDisconnect(),
            ]
        )

        with pytest.raises(WebSocketDisconnect):
            await _message_loop(ws, "w1", registry, coordinator)

        calls = ws.send_json.call_args_list
        error_responses = [c for c in calls if c[0][0].get("type") == "error"]
        assert len(error_responses) == 1
        assert "Unknown message type" in error_responses[0][0][0]["error"]

    @pytest.mark.asyncio
    async def test_message_too_large_in_loop(self, registry, coordinator):
        """Messages exceeding 64KB in the loop trigger an error and continue."""
        registry.register("w1", {})
        coordinator.get_next_assignment = MagicMock(return_value=None)

        large_text = "x" * 65537
        ws = AsyncMock()
        ws.receive = AsyncMock(
            side_effect=[
                {"text": large_text},
                WebSocketDisconnect(),
            ]
        )

        with pytest.raises(WebSocketDisconnect):
            await _message_loop(ws, "w1", registry, coordinator)

        calls = ws.send_json.call_args_list
        error_responses = [c for c in calls if c[0][0].get("type") == "error"]
        assert len(error_responses) == 1
        assert "too large" in error_responses[0][0][0]["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_json_in_loop(self, registry, coordinator):
        """Invalid JSON in the loop triggers an error and continues."""
        registry.register("w1", {})
        coordinator.get_next_assignment = MagicMock(return_value=None)

        ws = AsyncMock()
        ws.receive = AsyncMock(
            side_effect=[
                {"text": "not valid json{{{"},
                WebSocketDisconnect(),
            ]
        )

        with pytest.raises(WebSocketDisconnect):
            await _message_loop(ws, "w1", registry, coordinator)

        calls = ws.send_json.call_args_list
        error_responses = [c for c in calls if c[0][0].get("type") == "error"]
        assert len(error_responses) == 1
        assert "Invalid JSON" in error_responses[0][0][0]["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", [None, [], 7, "heartbeat"])
    async def test_non_object_json_in_loop(self, registry, coordinator, payload):
        """Non-object JSON in the loop errors and continues (session stays up)."""
        registry.register("w1", {})
        coordinator.get_next_assignment = MagicMock(return_value=None)

        ws = AsyncMock()
        ws.receive = AsyncMock(
            side_effect=[
                {"text": json.dumps(payload)},
                WebSocketDisconnect(),
            ]
        )

        with pytest.raises(WebSocketDisconnect):
            await _message_loop(ws, "w1", registry, coordinator)

        error_responses = [c for c in ws.send_json.call_args_list if c[0][0].get("type") == "error"]
        assert len(error_responses) == 1
        assert "JSON object" in error_responses[0][0][0]["error"]
        # Worker remains registered — loop continued rather than crashing.
        assert registry.get("w1") is not None

    @pytest.mark.asyncio
    async def test_stray_binary_frame(self, registry, coordinator):
        """Binary frames outside a task_result sequence are warned about and ignored."""
        registry.register("w1", {})
        coordinator.get_next_assignment = MagicMock(return_value=None)

        ws = AsyncMock()
        ws.receive = AsyncMock(
            side_effect=[
                {"bytes": b"\x00\x01\x02\x03"},
                WebSocketDisconnect(),
            ]
        )

        with pytest.raises(WebSocketDisconnect):
            await _message_loop(ws, "w1", registry, coordinator)

        # No send_json error sent for stray binary frames — just a log warning
        # The loop should continue after the binary frame
        assert ws.receive.await_count == 2

    @pytest.mark.asyncio
    async def test_task_result_triggers_dispatch(self, registry, coordinator):
        """After processing a task result, the loop tries to dispatch the next task."""
        registry.register("w1", {})

        # Submit a task so we can get a task_id
        tensors = {
            "candidate_input": np.zeros((10, 4), dtype=np.float32),
            "y": np.zeros((10, 1), dtype=np.float32),
            "residual_error": np.zeros((10, 1), dtype=np.float32),
        }
        task_ids = coordinator.submit_tasks(
            "r1",
            [{"candidate_index": 0, "candidate_data": {}, "training_params": {}}],
            tensors,
        )
        coordinator.get_next_assignment("w1")  # Assign the task

        result_msg = json.dumps(
            {
                "type": "task_result",
                "task_id": task_ids[0],
                "candidate_id": 0,
                "candidate_uuid": "uuid",
                "correlation": 0.85,
                "success": True,
                "epochs_completed": 10,
                "activation_name": "sigmoid",
                "all_correlations": [0.85],
                "numerator": 1.0,
                "denominator": 2.0,
                "best_corr_idx": 9,
                "error_message": None,
                "tensor_manifest": {
                    "weights": {"shape": [4], "dtype": "float32"},
                },
            }
        )

        weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        ws = AsyncMock()
        # First call from _message_loop, then binary frame from _handle_task_result
        ws.receive = AsyncMock(
            side_effect=[
                {"text": result_msg},
                {"bytes": BinaryFrame.encode(weights)},
                WebSocketDisconnect(),
            ]
        )
        # After handling result, get_next_assignment returns None
        original_get = coordinator.get_next_assignment
        call_count = [0]

        def mock_get(wid):
            call_count[0] += 1
            if call_count[0] <= 1:
                return original_get(wid)
            return None

        coordinator.get_next_assignment = mock_get

        with pytest.raises(WebSocketDisconnect):
            await _message_loop(ws, "w1", registry, coordinator)


@pytest.mark.unit
class TestHandleTaskResultEdgeCases:
    """Test edge cases in _handle_task_result."""

    @pytest.mark.asyncio
    async def test_missing_binary_frame_got_text(self, coordinator, registry):
        """When a binary frame is expected but text is received, an error is sent."""
        registry.register("w1", {})
        msg = {
            "type": "task_result",
            "task_id": "t1",
            "tensor_manifest": {
                "weights": {"shape": [4], "dtype": "float32"},
            },
        }

        ws = AsyncMock()
        # Return a text frame instead of binary
        ws.receive = AsyncMock(return_value={"text": "unexpected text"})

        await _handle_task_result(ws, "w1", msg, coordinator)

        ws.send_json.assert_awaited_once()
        error_msg = ws.send_json.call_args[0][0]
        assert error_msg["type"] == "error"
        assert "Expected binary frame" in error_msg["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_manifest", ["weights", None, ["weights"], 3])
    async def test_non_dict_tensor_manifest_rejected_before_receive(self, coordinator, registry, bad_manifest):
        """Non-dict tensor_manifest must error immediately — never call receive()."""
        registry.register("w1", {})
        msg = {
            "type": "task_result",
            "task_id": "t1",
            "tensor_manifest": bad_manifest,
        }

        ws = AsyncMock()
        coordinator.submit_result = MagicMock(return_value=True)

        await _handle_task_result(ws, "w1", msg, coordinator)

        ws.receive.assert_not_awaited()
        coordinator.submit_result.assert_not_called()
        ws.send_json.assert_awaited_once()
        error_msg = ws.send_json.call_args[0][0]
        assert error_msg["type"] == "error"
        assert "tensor_manifest must be a JSON object" in error_msg["error"]

    @pytest.mark.asyncio
    async def test_oversized_tensor_manifest_rejected_before_receive(self, coordinator, registry):
        """Huge tensor_manifest is rejected before waiting for N binary frames."""
        registry.register("w1", {})
        msg = {
            "type": "task_result",
            "task_id": "t1",
            "tensor_manifest": {f"t{i}": {"shape": [1], "dtype": "float32"} for i in range(33)},
        }

        ws = AsyncMock()
        coordinator.submit_result = MagicMock(return_value=True)

        await _handle_task_result(ws, "w1", msg, coordinator)

        ws.receive.assert_not_awaited()
        coordinator.submit_result.assert_not_called()
        error_msg = ws.send_json.call_args[0][0]
        assert error_msg["type"] == "error"
        assert "too many entries" in error_msg["error"]

    @pytest.mark.asyncio
    async def test_non_utf8_dtype_binary_frame_returns_error(self, coordinator, registry):
        """Non-UTF-8 dtype bytes surface as Invalid binary frame (ValueError path)."""
        import struct

        registry.register("w1", {})
        msg = {
            "type": "task_result",
            "task_id": "t1",
            "tensor_manifest": {
                "weights": {"shape": [1], "dtype": "float32"},
            },
        }
        # Valid header shape but invalid UTF-8 dtype bytes
        bad_frame = struct.pack("<I", 1) + struct.pack("<I", 1) + struct.pack("<I", 2) + b"\xff\xfe" + b"\x00\x00\x00\x00"

        ws = AsyncMock()
        ws.receive = AsyncMock(return_value={"bytes": bad_frame})

        await _handle_task_result(ws, "w1", msg, coordinator)

        ws.send_json.assert_awaited_once()
        error_msg = ws.send_json.call_args[0][0]
        assert error_msg["type"] == "error"
        assert "Invalid binary frame" in error_msg["error"]
        assert "UTF-8" in error_msg["error"]

    @pytest.mark.asyncio
    async def test_binary_frame_too_large(self, coordinator, registry):
        """Binary frames exceeding 100MB are rejected."""
        registry.register("w1", {})
        msg = {
            "type": "task_result",
            "task_id": "t1",
            "tensor_manifest": {
                "weights": {"shape": [4], "dtype": "float32"},
            },
        }

        ws = AsyncMock()
        # Return a frame larger than 100MB
        huge_bytes = b"\x00" * (100 * 1024 * 1024 + 1)
        ws.receive = AsyncMock(return_value={"bytes": huge_bytes})

        await _handle_task_result(ws, "w1", msg, coordinator)

        ws.send_json.assert_awaited_once()
        error_msg = ws.send_json.call_args[0][0]
        assert error_msg["type"] == "error"
        assert "too large" in error_msg["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_binary_frame_encoding(self, coordinator, registry):
        """Malformed binary frames that fail decoding return an error."""
        registry.register("w1", {})
        msg = {
            "type": "task_result",
            "task_id": "t1",
            "tensor_manifest": {
                "weights": {"shape": [4], "dtype": "float32"},
            },
        }

        ws = AsyncMock()
        # Return invalid binary data (too short to be a valid frame)
        ws.receive = AsyncMock(return_value={"bytes": b"\x01"})

        await _handle_task_result(ws, "w1", msg, coordinator)

        ws.send_json.assert_awaited_once()
        error_msg = ws.send_json.call_args[0][0]
        assert error_msg["type"] == "error"
        assert "Invalid binary frame" in error_msg["error"]

    @pytest.mark.asyncio
    async def test_result_rejected(self, coordinator, registry):
        """When coordinator rejects a result, a rejected ack is sent."""
        registry.register("w1", {})
        msg = {
            "type": "task_result",
            "task_id": "unknown_task",
            "tensor_manifest": {},
        }

        ws = AsyncMock()

        # Mock coordinator.submit_result to return False
        coordinator.submit_result = MagicMock(return_value=False)

        await _handle_task_result(ws, "w1", msg, coordinator)

        ws.send_json.assert_awaited_once()
        ack = ws.send_json.call_args[0][0]
        assert ack["type"] == "result_ack"
        assert ack["status"] == "rejected"

    @pytest.mark.asyncio
    async def test_result_with_no_tensors(self, coordinator, registry):
        """Task result with empty manifest (no tensors) is processed correctly."""
        registry.register("w1", {})
        msg = {
            "type": "task_result",
            "task_id": "t1",
            "tensor_manifest": {},
        }

        ws = AsyncMock()
        coordinator.submit_result = MagicMock(return_value=True)

        await _handle_task_result(ws, "w1", msg, coordinator)

        ws.send_json.assert_awaited_once()
        ack = ws.send_json.call_args[0][0]
        assert ack["type"] == "result_ack"
        assert ack["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_multiple_tensors_second_frame_invalid(self, coordinator, registry):
        """When the second binary frame in a multi-tensor result is invalid, error is sent."""
        registry.register("w1", {})
        msg = {
            "type": "task_result",
            "task_id": "t1",
            "tensor_manifest": {
                "weights": {"shape": [4], "dtype": "float32"},
                "bias": {"shape": [1], "dtype": "float32"},
            },
        }

        weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        ws = AsyncMock()
        ws.receive = AsyncMock(
            side_effect=[
                {"bytes": BinaryFrame.encode(weights)},
                {"bytes": b"\x02"},  # invalid second frame
            ]
        )

        await _handle_task_result(ws, "w1", msg, coordinator)

        # Last send_json should be an error about the bias tensor
        last_call = ws.send_json.call_args[0][0]
        assert last_call["type"] == "error"
        assert "bias" in last_call["error"]


@pytest.mark.unit
class TestMakeSendCallback:
    """Test _make_send_callback success and failure paths."""

    @pytest.mark.asyncio
    async def test_callback_success_no_frames(self):
        """Callback sends JSON msg and returns True when no frames provided."""
        ws = AsyncMock()
        callback = _make_send_callback(ws)

        result = await callback({"type": "task_assign", "task_id": "t1"})

        assert result is True
        ws.send_json.assert_awaited_once_with({"type": "task_assign", "task_id": "t1"})
        ws.send_bytes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_callback_success_with_frames(self):
        """Callback sends JSON msg and binary frames, returns True."""
        ws = AsyncMock()
        callback = _make_send_callback(ws)

        frames = [b"\x01\x02\x03", b"\x04\x05\x06"]
        result = await callback({"type": "task_assign", "task_id": "t1"}, frames=frames)

        assert result is True
        ws.send_json.assert_awaited_once()
        assert ws.send_bytes.await_count == 2

    @pytest.mark.asyncio
    async def test_callback_failure_returns_false(self):
        """Callback returns False when WebSocket send raises an exception."""
        ws = AsyncMock()
        ws.send_json = AsyncMock(side_effect=RuntimeError("connection lost"))
        callback = _make_send_callback(ws)

        result = await callback({"type": "task_assign", "task_id": "t1"})

        assert result is False

    @pytest.mark.asyncio
    async def test_callback_failure_on_frame_send(self):
        """Callback returns False when binary frame send raises an exception."""
        ws = AsyncMock()
        ws.send_bytes = AsyncMock(side_effect=RuntimeError("connection lost"))
        callback = _make_send_callback(ws)

        result = await callback({"type": "task_assign"}, frames=[b"\x01"])

        assert result is False
