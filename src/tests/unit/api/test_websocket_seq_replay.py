"""Tests for Phase 0-cascor: sequence numbers, replay buffer, pending connections, send timeout, broadcast fix."""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.websocket.manager import ReplayOutOfRange, WebSocketManager


@pytest.mark.unit
class TestSequenceNumbers:
    """Test monotonically increasing sequence number assignment."""

    @pytest.mark.asyncio
    async def test_seq_monotonically_increases_across_broadcasts(self):
        """seq(n+1) > seq(n) for consecutive broadcasts."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)

        await mgr.broadcast({"type": "metrics", "data": {}})
        await mgr.broadcast({"type": "metrics", "data": {}})
        await mgr.broadcast({"type": "metrics", "data": {}})

        # call_args_list: [0] = connection_established, [1..3] = broadcasts
        calls = ws.send_json.call_args_list
        seqs = [c[0][0]["seq"] for c in calls[1:]]
        assert seqs == [1, 2, 3]
        assert all(seqs[i] < seqs[i + 1] for i in range(len(seqs) - 1))

    @pytest.mark.asyncio
    async def test_seq_is_assigned_on_loop_thread(self):
        """Seq assignment happens inside broadcast(), not in the message builder."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)

        msg = {"type": "metrics", "data": {"loss": 0.5}}
        assert "seq" not in msg  # Not assigned yet

        await mgr.broadcast(msg)
        sent = ws.send_json.call_args_list[-1][0][0]
        assert "seq" in sent
        assert sent["seq"] == 1

    @pytest.mark.asyncio
    async def test_emitted_at_monotonic_present_on_every_broadcast(self):
        """Every broadcast message includes emitted_at_monotonic."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)

        await mgr.broadcast({"type": "state", "data": {}})
        sent = ws.send_json.call_args_list[-1][0][0]
        assert "emitted_at_monotonic" in sent
        assert isinstance(sent["emitted_at_monotonic"], float)

    @pytest.mark.asyncio
    async def test_seq_lock_does_not_block_broadcast_iteration(self):
        """Broadcast with 10 clients completes in reasonable time."""
        mgr = WebSocketManager()
        clients = [AsyncMock() for _ in range(10)]
        for ws in clients:
            await mgr.connect(ws)

        await asyncio.wait_for(
            mgr.broadcast({"type": "metrics", "data": {}}),
            timeout=1.0,
        )
        for ws in clients:
            assert ws.send_json.await_count == 2  # established + broadcast

    def test_current_seq_property(self):
        """current_seq returns last assigned seq (0 if none)."""
        mgr = WebSocketManager()
        assert mgr.current_seq == 0

    @pytest.mark.asyncio
    async def test_current_seq_after_broadcast(self):
        """current_seq updates after each broadcast."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)

        await mgr.broadcast({"type": "test", "data": {}})
        assert mgr.current_seq == 1

        await mgr.broadcast({"type": "test", "data": {}})
        assert mgr.current_seq == 2


@pytest.mark.unit
class TestReplayBuffer:
    """Test replay buffer storage and query."""

    @pytest.mark.asyncio
    async def test_replay_buffer_bounded_to_configured_capacity(self):
        """Buffer does not exceed configured maxlen."""
        mgr = WebSocketManager(max_replay_buffer_size=10)
        ws = AsyncMock()
        await mgr.connect(ws)

        for i in range(20):
            await mgr.broadcast({"type": "metrics", "data": {"i": i}})

        assert len(mgr._replay_buffer) == 10
        # Oldest should be seq 11 (first 10 evicted)
        assert mgr._replay_buffer[0]["seq"] == 11

    def test_replay_buffer_capacity_configurable(self):
        """max_replay_buffer_size parameter controls buffer capacity."""
        mgr = WebSocketManager(max_replay_buffer_size=256)
        assert mgr._replay_buffer_max_size == 256

    def test_replay_buffer_size_zero_disables_replay(self):
        """Size 0 disables replay; replay_since raises ReplayOutOfRange."""
        mgr = WebSocketManager(max_replay_buffer_size=0)
        with pytest.raises(ReplayOutOfRange, match="disabled"):
            mgr.replay_since(0)

    @pytest.mark.asyncio
    async def test_replay_since_returns_correct_subset(self):
        """replay_since(N) returns messages with seq > N."""
        mgr = WebSocketManager(max_replay_buffer_size=100)
        ws = AsyncMock()
        await mgr.connect(ws)

        for _ in range(10):
            await mgr.broadcast({"type": "metrics", "data": {}})

        result = mgr.replay_since(5)
        assert len(result) == 5
        assert [m["seq"] for m in result] == [6, 7, 8, 9, 10]

    @pytest.mark.asyncio
    async def test_replay_since_out_of_range(self):
        """replay_since with too-old seq raises ReplayOutOfRange."""
        mgr = WebSocketManager(max_replay_buffer_size=5)
        ws = AsyncMock()
        await mgr.connect(ws)

        for _ in range(20):
            await mgr.broadcast({"type": "metrics", "data": {}})

        # Buffer has seq 16-20; requesting from seq 5 is out of range
        with pytest.raises(ReplayOutOfRange):
            mgr.replay_since(5)

    @pytest.mark.asyncio
    async def test_replay_since_zero_with_empty_buffer(self):
        """replay_since(0) on empty buffer returns empty list."""
        mgr = WebSocketManager(max_replay_buffer_size=100)
        result = mgr.replay_since(0)
        assert result == []

    @pytest.mark.asyncio
    async def test_replay_since_nonzero_with_empty_buffer(self):
        """replay_since(N>0) on empty buffer raises (cannot verify continuity)."""
        mgr = WebSocketManager(max_replay_buffer_size=100)
        with pytest.raises(ReplayOutOfRange, match="empty"):
            mgr.replay_since(5)

    @pytest.mark.asyncio
    async def test_replay_since_uses_bisect_for_log_n_lookup(self):
        """PERF-CC-02: replay_since must use bisect (O(log n)) not a linear scan.

        We verify behavior at the buffer boundary — a request for an
        in-range seq must return exactly the entries with seq > last_seq,
        in order, with no off-by-one. The original linear scan and the
        bisect_right-based path are functionally equivalent; this test
        guards against regressions when refactoring the lookup.
        """
        mgr = WebSocketManager(max_replay_buffer_size=200)
        ws = AsyncMock()
        await mgr.connect(ws)

        for _ in range(150):
            await mgr.broadcast({"type": "metrics", "data": {}})

        # Boundary test 1: exact match — last_seq=100 returns 101..150 (50 entries)
        boundary = mgr.replay_since(100)
        assert [m["seq"] for m in boundary] == list(range(101, 151))

        # Boundary test 2: one before — last_seq=99 returns 100..150 (51 entries)
        before = mgr.replay_since(99)
        assert [m["seq"] for m in before] == list(range(100, 151))

        # Boundary test 3: latest seq — last_seq=150 returns nothing
        empty = mgr.replay_since(150)
        assert empty == []

        # Boundary test 4: oldest seq still in buffer — last_seq=0 returns all
        all_msgs = mgr.replay_since(0)
        assert [m["seq"] for m in all_msgs] == list(range(1, 151))

    @pytest.mark.asyncio
    async def test_replay_buffer_uses_bisect_module(self):
        """PERF-CC-02: import side — ensure bisect is wired into manager."""
        # The import must succeed and bisect must be available at module level
        # (locked in to prevent accidental removal during cleanup passes).
        import bisect

        from api.websocket import manager as mgr_module

        assert getattr(mgr_module, "bisect", None) is bisect


@pytest.mark.unit
class TestConnectionEstablished:
    """Test connection_established message content."""

    @pytest.mark.asyncio
    async def test_connection_established_advertises_instance_id_and_capacity(self):
        """connection_established includes server_instance_id, server_start_time, replay_buffer_capacity."""
        mgr = WebSocketManager(max_replay_buffer_size=512)
        ws = AsyncMock()
        await mgr.connect(ws)

        msg = ws.send_json.call_args_list[-1][0][0]
        assert msg["type"] == "connection_established"
        data = msg["data"]

        # server_instance_id is a UUID
        uuid.UUID(data["server_instance_id"])  # validates format, raises on bad UUID
        assert data["server_instance_id"] == mgr.server_instance_id

        assert isinstance(data["server_start_time"], float)
        assert data["replay_buffer_capacity"] == 512

    @pytest.mark.asyncio
    async def test_server_instance_id_is_stable_across_connections(self):
        """server_instance_id is the same for all connections on the same manager."""
        mgr = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await mgr.connect(ws1)
        await mgr.connect(ws2)

        msg1 = ws1.send_json.call_args_list[-1][0][0]
        msg2 = ws2.send_json.call_args_list[-1][0][0]
        assert msg1["data"]["server_instance_id"] == msg2["data"]["server_instance_id"]


@pytest.mark.unit
class TestPendingConnections:
    """Test the pending connection lifecycle (D-14)."""

    @pytest.mark.asyncio
    async def test_pending_connections_not_eligible_for_broadcast(self):
        """Pending connections do not receive broadcast messages."""
        mgr = WebSocketManager()
        ws_pending = AsyncMock()
        ws_active = AsyncMock()

        await mgr.connect(ws_active)
        await mgr.connect_pending(ws_pending)

        # Reset send counts after connection_established
        ws_active.send_json.reset_mock()
        ws_pending.send_json.reset_mock()

        await mgr.broadcast({"type": "metrics", "data": {}})

        assert ws_active.send_json.await_count == 1
        assert ws_pending.send_json.await_count == 0

    @pytest.mark.asyncio
    async def test_promote_to_active_receives_broadcasts(self):
        """After promotion, connection receives broadcasts."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect_pending(ws)
        await mgr.promote_to_active(ws)

        ws.send_json.reset_mock()
        await mgr.broadcast({"type": "metrics", "data": {}})

        assert ws.send_json.await_count == 1

    @pytest.mark.asyncio
    async def test_pending_counts_toward_max_connections(self):
        """Pending connections count against the max_connections limit."""
        mgr = WebSocketManager(max_connections=2)
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()

        await mgr.connect(ws1)
        await mgr.connect_pending(ws2)
        result = await mgr.connect(ws3)

        assert result is False
        ws3.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_pending_sends_connection_established(self):
        """connect_pending sends connection_established with server identity."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect_pending(ws)

        msg = ws.send_json.call_args_list[-1][0][0]
        assert msg["type"] == "connection_established"
        assert "server_instance_id" in msg["data"]

    @pytest.mark.asyncio
    async def test_disconnect_removes_pending(self):
        """disconnect removes a pending connection."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect_pending(ws)
        assert len(mgr._pending_connections) == 1

        await mgr.disconnect(ws)
        assert len(mgr._pending_connections) == 0

    @pytest.mark.asyncio
    async def test_close_all_clears_pending(self):
        """close_all clears both active and pending connections."""
        mgr = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await mgr.connect(ws1)
        await mgr.connect_pending(ws2)

        await mgr.close_all()
        assert mgr.connection_count == 0
        assert len(mgr._pending_connections) == 0


@pytest.mark.unit
class TestSendTimeout:
    """Test send timeout behavior (GAP-WS-07)."""

    @pytest.mark.asyncio
    async def test_slow_client_send_timeout_does_not_block_fanout(self):
        """Slow client times out and is removed; fast client unaffected."""
        mgr = WebSocketManager(send_timeout_seconds=0.1)
        ws_fast = AsyncMock()
        ws_slow = AsyncMock()

        async def slow_send(msg):
            await asyncio.sleep(5)

        ws_slow.send_json.side_effect = slow_send

        await mgr.connect(ws_fast)
        # Manually add slow client to avoid connection_established timeout
        mgr._active_connections.add(ws_slow)

        await asyncio.wait_for(
            mgr.broadcast({"type": "test", "data": {}}),
            timeout=2.0,
        )

        # Slow client should be disconnected
        assert ws_slow not in mgr._active_connections

    def test_send_timeout_configurable(self):
        """send_timeout_seconds parameter is stored."""
        mgr = WebSocketManager(send_timeout_seconds=2.0)
        assert mgr._send_timeout_seconds == 2.0


@pytest.mark.unit
class TestBroadcastFromThreadFix:
    """Test GAP-WS-29: broadcast_from_thread exception logging."""

    def test_broadcast_from_thread_exception_logged(self):
        """Broadcast coroutine exceptions are logged, not swallowed (GAP-WS-29).

        Uses mock on logger.error directly because the conftest no-op logger
        fixture interferes with caplog when TestClient tests run first.
        """
        mgr = WebSocketManager()
        loop = MagicMock()
        loop.is_closed.return_value = False
        mgr.set_event_loop(loop)

        mock_future = MagicMock()
        mock_future.exception.return_value = RuntimeError("test error")

        with patch("api.websocket.manager.asyncio.run_coroutine_threadsafe") as mock_submit:
            mock_submit.return_value = mock_future
            mgr.broadcast_from_thread({"type": "test"})

            # Verify done callback was registered
            mock_future.add_done_callback.assert_called_once()
            callback = mock_future.add_done_callback.call_args[0][0]

            # Invoke the callback and verify it logs the error
            with patch("api.websocket.manager.logger") as mock_logger:
                callback(mock_future)
                mock_logger.error.assert_called_once()
                assert "Broadcast from thread failed" in mock_logger.error.call_args[0][0]

            # Close the coroutine to prevent warning
            coro = mock_submit.call_args[0][0]
            coro.close()

    def test_broadcast_from_thread_no_exception_no_log(self):
        """Done callback does nothing when future succeeds."""
        mock_future = MagicMock()
        mock_future.exception.return_value = None

        with patch("api.websocket.manager.logger") as mock_logger:
            WebSocketManager._log_broadcast_exception(mock_future)
            mock_logger.error.assert_not_called()


@pytest.mark.unit
class TestLegacyCompatibility:
    """Test backward compatibility with legacy clients."""

    @pytest.mark.asyncio
    async def test_legacy_client_ignores_seq_field(self):
        """Message with seq can still be consumed by code reading only type/timestamp/data."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)

        await mgr.broadcast({"type": "metrics", "timestamp": 123.0, "data": {"loss": 0.5}})
        sent = ws.send_json.call_args_list[-1][0][0]

        # Legacy client reads only these fields
        assert sent["type"] == "metrics"
        assert "data" in sent
        assert sent["data"]["loss"] == 0.5
        # New fields present but don't break old readers
        assert "seq" in sent
        assert "emitted_at_monotonic" in sent
