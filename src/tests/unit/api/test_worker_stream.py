"""Tests for /ws/v1/workers WebSocket endpoint handler."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from api.websocket.worker_stream import _handle_registration, _handle_task_result, _try_dispatch_task, worker_stream_handler
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
        """Valid registration message registers the worker."""
        ws = AsyncMock()
        ws.receive_text = AsyncMock(return_value=json.dumps(WorkerProtocol.build_register("w1", {"cpu_cores": 4})))

        worker_id = await _handle_registration(ws, registry)
        assert worker_id == "w1"
        assert registry.worker_count == 1

    @pytest.mark.asyncio
    async def test_invalid_json(self, registry):
        """Invalid JSON closes the connection."""
        ws = AsyncMock()
        ws.receive_text = AsyncMock(return_value="not json{{{")

        worker_id = await _handle_registration(ws, registry)
        assert worker_id is None
        ws.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wrong_message_type(self, registry):
        """Non-registration first message closes the connection."""
        ws = AsyncMock()
        ws.receive_text = AsyncMock(return_value=json.dumps({"type": "heartbeat", "worker_id": "w1"}))

        worker_id = await _handle_registration(ws, registry)
        assert worker_id is None

    @pytest.mark.asyncio
    async def test_missing_fields(self, registry):
        """Registration with missing fields closes the connection."""
        ws = AsyncMock()
        ws.receive_text = AsyncMock(return_value=json.dumps({"type": "register"}))  # Missing worker_id and capabilities

        worker_id = await _handle_registration(ws, registry)
        assert worker_id is None


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
