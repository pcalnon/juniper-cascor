"""Tests for WebSocket connection manager."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.websocket.manager import WebSocketManager


@pytest.mark.unit
class TestWebSocketManager:
    """Test WebSocket connection manager."""

    def test_init_defaults(self):
        """Manager initializes with defaults."""
        mgr = WebSocketManager()
        assert mgr.connection_count == 0
        assert mgr._max_connections == 50

    def test_init_custom_max(self):
        """Manager respects custom max_connections."""
        mgr = WebSocketManager(max_connections=5)
        assert mgr._max_connections == 5

    @pytest.mark.asyncio
    async def test_connect(self):
        """Connect accepts and registers a WebSocket."""
        mgr = WebSocketManager()
        ws = AsyncMock()

        result = await mgr.connect(ws)

        assert result is True
        assert mgr.connection_count == 1
        ws.accept.assert_awaited_once()
        ws.send_json.assert_awaited_once()
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "connection_established"

    @pytest.mark.asyncio
    async def test_connect_max_reached(self):
        """Connect rejects when max connections reached."""
        mgr = WebSocketManager(max_connections=1)
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        await mgr.connect(ws1)
        result = await mgr.connect(ws2)

        assert result is False
        assert mgr.connection_count == 1
        ws2.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Disconnect removes a connection."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)
        assert mgr.connection_count == 1

        await mgr.disconnect(ws)
        assert mgr.connection_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_unknown(self):
        """Disconnect of unknown connection doesn't error."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.disconnect(ws)
        assert mgr.connection_count == 0

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Broadcast sends to all connected clients."""
        mgr = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await mgr.connect(ws1)
        await mgr.connect(ws2)

        message = {"type": "test", "data": {}}
        await mgr.broadcast(message)

        # Each WS received connection_established + broadcast
        assert ws1.send_json.await_count == 2
        assert ws2.send_json.await_count == 2

    @pytest.mark.asyncio
    async def test_broadcast_removes_failed(self):
        """Broadcast removes connections that fail to send."""
        mgr = WebSocketManager()
        ws_good = AsyncMock()
        ws_bad = AsyncMock()
        ws_bad.send_json.side_effect = Exception("Connection closed")

        await mgr.connect(ws_good)
        # Manually add ws_bad since connect sends a message which would fail
        mgr._active_connections.add(ws_bad)

        await mgr.broadcast({"type": "test"})

        assert ws_bad not in mgr._active_connections
        assert mgr.connection_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_empty(self):
        """Broadcast with no connections does nothing."""
        mgr = WebSocketManager()
        await mgr.broadcast({"type": "test"})  # Should not raise

    def test_broadcast_from_thread_no_loop(self):
        """broadcast_from_thread without event loop doesn't error."""
        mgr = WebSocketManager()
        mgr.broadcast_from_thread({"type": "test"})  # Should not raise

    def test_broadcast_from_thread_with_loop(self):
        """broadcast_from_thread submits coroutine to event loop."""
        mgr = WebSocketManager()
        loop = MagicMock()
        loop.is_closed.return_value = False
        mgr.set_event_loop(loop)

        with patch("api.websocket.manager.asyncio.run_coroutine_threadsafe") as mock_submit:
            mgr.broadcast_from_thread({"type": "test"})
            mock_submit.assert_called_once()
            # Close the coroutine to prevent "coroutine was never awaited" RuntimeWarning
            coro = mock_submit.call_args[0][0]
            coro.close()

    @pytest.mark.asyncio
    async def test_send_personal_message(self):
        """send_personal_message sends to specific client."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)

        result = await mgr.send_personal_message(ws, {"type": "personal"})
        assert result is True

    @pytest.mark.asyncio
    async def test_send_personal_message_failure(self):
        """send_personal_message returns False on failure."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        ws.send_json.side_effect = Exception("fail")

        result = await mgr.send_personal_message(ws, {"type": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_close_all(self):
        """close_all closes all connections."""
        mgr = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await mgr.connect(ws1)
        await mgr.connect(ws2)

        await mgr.close_all()

        assert mgr.connection_count == 0
        ws1.close.assert_awaited_once()
        ws2.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_all_holds_lock_during_snapshot(self):
        """close_all acquires the manager lock before snapshotting and
        clearing the connection set (CR-025). This prevents a connect/
        shutdown race where a new connection could be added to a partially
        cleared set."""
        import asyncio

        mgr = WebSocketManager()
        ws1 = AsyncMock()
        await mgr.connect(ws1)

        # Manually acquire the lock and verify close_all is blocked until
        # released. If close_all bypassed the lock, it would proceed
        # immediately and the wait_for below would NOT time out.
        await mgr._lock.acquire()
        try:
            task = asyncio.create_task(mgr.close_all())
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
        finally:
            mgr._lock.release()

        # Once we release the lock, close_all completes.
        await task
        assert mgr.connection_count == 0
        ws1.close.assert_awaited_once()

    def test_set_event_loop(self):
        """set_event_loop stores loop reference."""
        mgr = WebSocketManager()
        loop = MagicMock()
        mgr.set_event_loop(loop)
        assert mgr._event_loop is loop


@pytest.mark.unit
class TestTransportStats:
    """GAP-WS-16: bandwidth instrumentation surfaced via transport_stats()."""

    def test_initial_stats_zeroed(self):
        mgr = WebSocketManager()
        stats = mgr.transport_stats()
        assert stats["bytes_sent_total"] == 0
        assert stats["messages_sent_total"] == 0
        assert stats["send_failures"] == 0
        assert stats["messages_sent_by_type"] == {}
        assert stats["bytes_sent_by_type"] == {}
        assert stats["active_connections"] == 0
        assert stats["pending_connections"] == 0

    @pytest.mark.asyncio
    async def test_successful_send_increments_counters(self):
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)
        await mgr.broadcast({"type": "metrics", "data": {"epoch": 1}})

        stats = mgr.transport_stats()
        # connection_established + broadcast = 2 messages
        assert stats["messages_sent_total"] == 2
        assert stats["bytes_sent_total"] > 0
        assert stats["messages_sent_by_type"]["connection_established"] == 1
        assert stats["messages_sent_by_type"]["metrics"] == 1
        assert stats["bytes_sent_by_type"]["metrics"] > 0
        assert stats["send_failures"] == 0

    @pytest.mark.asyncio
    async def test_send_timeout_counts_as_failure(self):
        mgr = WebSocketManager(send_timeout_seconds=0.01)
        ws = AsyncMock()
        await mgr.connect(ws)

        async def slow_send(_):
            await asyncio.sleep(0.5)

        ws.send_json.side_effect = slow_send
        result = await mgr._send_json(ws, {"type": "metrics", "data": {}})
        assert result is False
        stats = mgr.transport_stats()
        assert stats["send_failures"] >= 1

    @pytest.mark.asyncio
    async def test_per_type_accounting_distinguishes_msg_types(self):
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)
        await mgr.broadcast({"type": "metrics", "data": {"epoch": 1}})
        await mgr.broadcast({"type": "topology", "data": {"hidden": []}})
        await mgr.broadcast({"type": "metrics", "data": {"epoch": 2}})

        stats = mgr.transport_stats()
        assert stats["messages_sent_by_type"]["metrics"] == 2
        assert stats["messages_sent_by_type"]["topology"] == 1

    def test_transport_stats_includes_replay_buffer_state(self):
        mgr = WebSocketManager(max_replay_buffer_size=512)
        stats = mgr.transport_stats()
        assert stats["replay_buffer_capacity"] == 512
        assert stats["replay_buffer_size"] == 0
        assert stats["current_seq"] == 0


@pytest.mark.unit
class TestSizeGuardAndChunking:
    """GAP-WS-18: oversized broadcasts split into chunked_message envelopes
    so we never push a single frame past the 64 KB intermediary limit."""

    def test_small_message_is_not_chunked(self):
        mgr = WebSocketManager(max_message_size_bytes=60_000, chunk_payload_size_bytes=32_000)
        chunks = mgr._maybe_chunk_message({"type": "topology", "data": {"x": 1}})
        assert len(chunks) == 1
        assert chunks[0]["type"] == "topology"

    def test_oversized_message_is_chunked(self):
        # Build a message whose serialized JSON exceeds the threshold.
        big_data = {"hidden_layers": [{"name": f"layer_{i}", "weights": list(range(100))} for i in range(200)]}
        mgr = WebSocketManager(max_message_size_bytes=10_000, chunk_payload_size_bytes=4_000)
        chunks = mgr._maybe_chunk_message({"type": "topology", "data": big_data})
        assert len(chunks) >= 2
        # All chunks share a chunk_id.
        chunk_ids = {c["data"]["chunk_id"] for c in chunks}
        assert len(chunk_ids) == 1
        # chunk_index runs 0..N-1, total_chunks is consistent.
        total = chunks[0]["data"]["total_chunks"]
        assert total == len(chunks)
        for idx, chunk in enumerate(chunks):
            assert chunk["type"] == "chunked_message"
            assert chunk["data"]["chunk_index"] == idx
            assert chunk["data"]["total_chunks"] == total
            assert chunk["data"]["original_type"] == "topology"

    def test_chunks_reassemble_to_original_message(self):
        """Concatenating payloads in chunk_index order must yield the original JSON."""
        original = {"type": "topology", "data": {"big": "x" * 50_000}}
        mgr = WebSocketManager(max_message_size_bytes=10_000, chunk_payload_size_bytes=4_000)
        chunks = mgr._maybe_chunk_message(original)
        assert len(chunks) > 1
        reassembled_text = "".join(c["data"]["payload"] for c in chunks)
        import json as _json
        reassembled = _json.loads(reassembled_text)
        assert reassembled["type"] == original["type"]
        assert reassembled["data"]["big"] == original["data"]["big"]

    def test_chunking_disabled_when_threshold_zero(self):
        """ws_max_message_size_bytes=0 is a kill-switch that disables chunking."""
        mgr = WebSocketManager(max_message_size_bytes=0, chunk_payload_size_bytes=4_000)
        big_data = {"hidden_layers": list(range(10_000))}
        chunks = mgr._maybe_chunk_message({"type": "topology", "data": big_data})
        assert len(chunks) == 1
        assert chunks[0]["type"] == "topology"

    def test_chunked_message_is_not_re_chunked(self):
        """Already-chunked envelopes must never recurse through the chunker."""
        mgr = WebSocketManager(max_message_size_bytes=100, chunk_payload_size_bytes=50)
        chunked = {
            "type": "chunked_message",
            "data": {"chunk_id": "x", "chunk_index": 0, "total_chunks": 1, "original_type": "topology", "payload": "y" * 1000},
        }
        chunks = mgr._maybe_chunk_message(chunked)
        assert len(chunks) == 1
        assert chunks[0] is chunked

    @pytest.mark.asyncio
    async def test_broadcast_assigns_consecutive_seqs_to_chunks(self):
        """Each chunk gets its own seq so the replay buffer reorders them on resume."""
        mgr = WebSocketManager(max_message_size_bytes=10_000, chunk_payload_size_bytes=4_000)
        ws = AsyncMock()
        await mgr.connect(ws)
        seq_before = mgr.current_seq
        await mgr.broadcast({"type": "topology", "data": {"big": "x" * 50_000}})
        seq_after = mgr.current_seq
        # current_seq advanced by N (number of chunks), not 1.
        assert seq_after - seq_before >= 2

    @pytest.mark.asyncio
    async def test_broadcast_chunks_each_carry_seq_and_replay(self):
        mgr = WebSocketManager(max_message_size_bytes=10_000, chunk_payload_size_bytes=4_000, max_replay_buffer_size=100)
        ws = AsyncMock()
        await mgr.connect(ws)
        await mgr.broadcast({"type": "topology", "data": {"big": "x" * 30_000}})
        # Replay buffer holds connection_established? No — that's a personal
        # message. Only broadcast chunks land in the replay buffer.
        replayed = mgr.replay_since(0)
        assert len(replayed) >= 2
        # All replayed entries are chunked_message envelopes carrying a seq.
        for msg in replayed:
            assert msg["type"] == "chunked_message"
            assert "seq" in msg

    def test_transport_stats_exposes_chunk_counters(self):
        mgr = WebSocketManager(max_message_size_bytes=60_000, chunk_payload_size_bytes=32_000)
        stats = mgr.transport_stats()
        assert "messages_chunked_total" in stats
        assert "chunks_emitted_total" in stats
        assert "max_message_size_bytes" in stats
        assert "chunk_payload_size_bytes" in stats
        assert stats["messages_chunked_total"] == 0
        assert stats["chunks_emitted_total"] == 0
        assert stats["max_message_size_bytes"] == 60_000
        assert stats["chunk_payload_size_bytes"] == 32_000

    def test_chunk_counters_increment_on_split(self):
        mgr = WebSocketManager(max_message_size_bytes=10_000, chunk_payload_size_bytes=4_000)
        chunks = mgr._maybe_chunk_message({"type": "topology", "data": {"big": "x" * 50_000}})
        stats = mgr.transport_stats()
        assert stats["messages_chunked_total"] == 1
        assert stats["chunks_emitted_total"] == len(chunks)

    @pytest.mark.asyncio
    async def test_send_personal_message_chunks_oversized(self):
        mgr = WebSocketManager(max_message_size_bytes=10_000, chunk_payload_size_bytes=4_000)
        ws = AsyncMock()
        await mgr.connect(ws)
        # Reset send_json call count after the connection_established message
        ws.send_json.reset_mock()
        result = await mgr.send_personal_message(ws, {"type": "topology", "data": {"big": "x" * 30_000}})
        assert result is True
        # Multiple sends → multiple chunks
        assert ws.send_json.await_count >= 2
