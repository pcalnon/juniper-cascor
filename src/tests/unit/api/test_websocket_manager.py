"""Tests for WebSocket connection manager."""

import asyncio
from concurrent.futures import CancelledError
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
    async def test_close_all_releases_per_ip_slots(self):
        """close_all releases per-IP reservations so reconnects are not rejected."""
        mgr = WebSocketManager(max_connections_per_ip=1)
        ws1 = AsyncMock()
        ws1.client = ("10.0.0.5", 1111)
        assert await mgr.connect_pending(ws1) is True

        await mgr.close_all()

        ws2 = AsyncMock()
        ws2.client = ("10.0.0.5", 2222)
        assert await mgr.connect_pending(ws2) is True
        ws2.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_close_all_clears_endpoint_bookkeeping(self):
        """close_all removes endpoint-bucket entries for closed sockets."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)
        mgr.register_endpoint_connection(ws, "training")
        assert ws in mgr._endpoint_connections["training"]

        await mgr.close_all()

        assert mgr._endpoint_connections["training"] == set()
        assert mgr._connection_endpoint == {}

    @pytest.mark.asyncio
    async def test_close_all_closes_endpoint_only_connections(self):
        """close_all closes sockets that are tracked only by endpoint buckets."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        mgr.register_endpoint_connection(ws, "control")

        await mgr.close_all()

        ws.close.assert_awaited_once_with(code=1001, reason="Server shutting down")
        assert mgr._endpoint_connections["control"] == set()
        assert mgr._connection_endpoint == {}

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
class TestEmissionSummary:
    """C3 / T5: per-frame-type emission counters + periodic INFO summary.

    The summary is the server-side diagnostic for "relay connected but
    nothing flowing" — a line reading ``ping=N`` with no ``metrics``
    entries proves the socket was writable while no training frames were
    emitted.
    """

    def test_record_out_of_band_send_accounts_frame(self):
        """Heartbeat pings sent outside the manager's send paths are counted."""
        mgr = WebSocketManager()
        mgr.record_out_of_band_send({"type": "ping"})
        mgr.record_out_of_band_send({"type": "ping"})

        stats = mgr.transport_stats()
        assert stats["messages_sent_by_type"]["ping"] == 2
        assert stats["bytes_sent_by_type"]["ping"] > 0
        assert stats["messages_sent_total"] == 2

    def test_summary_rate_limited_by_interval(self, caplog):
        """With the default 60s interval, an immediate send logs no summary."""
        import logging

        mgr = WebSocketManager()
        with caplog.at_level(logging.INFO, logger="juniper_cascor.api.websocket"):
            mgr.record_out_of_band_send({"type": "ping"})
        assert not any("WS emission summary" in rec.message for rec in caplog.records)

    def test_forced_summary_reports_deltas_and_resets_baseline(self, caplog):
        """force=True emits per-type deltas since the last summary, then resets."""
        import logging

        mgr = WebSocketManager()
        mgr.record_out_of_band_send({"type": "ping"})
        mgr.record_out_of_band_send({"type": "ping"})
        mgr._account_send({"type": "metrics", "data": {}}, 128)

        with caplog.at_level(logging.INFO, logger="juniper_cascor.api.websocket"):
            deltas = mgr.maybe_log_emission_summary(force=True)
        assert deltas == {"ping": 2, "metrics": 1}
        summary_lines = [rec.message for rec in caplog.records if "WS emission summary" in rec.message]
        assert len(summary_lines) == 1
        assert "metrics=1" in summary_lines[0]
        assert "ping=2" in summary_lines[0]

        # Baseline reset: a second forced summary reports nothing new.
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="juniper_cascor.api.websocket"):
            deltas2 = mgr.maybe_log_emission_summary(force=True)
        assert deltas2 == {}
        assert any("no frames emitted" in rec.message for rec in caplog.records)

    def test_summary_emits_automatically_after_interval(self, caplog):
        """Once the interval elapses, the next accounted send logs the summary."""
        import logging
        import time as _time

        mgr = WebSocketManager(emission_summary_interval_sec=0.05)
        mgr.record_out_of_band_send({"type": "ping"})  # within interval: no summary
        _time.sleep(0.06)
        with caplog.at_level(logging.INFO, logger="juniper_cascor.api.websocket"):
            mgr.record_out_of_band_send({"type": "ping"})
        summary_lines = [rec.message for rec in caplog.records if "WS emission summary" in rec.message]
        assert len(summary_lines) == 1
        assert "ping=2" in summary_lines[0]

    def test_summary_interval_zero_disables_unless_forced(self, caplog):
        """<= 0 disables the periodic summary; force=True still works."""
        import logging
        import time as _time

        mgr = WebSocketManager(emission_summary_interval_sec=0)
        mgr.record_out_of_band_send({"type": "ping"})
        _time.sleep(0.01)
        with caplog.at_level(logging.INFO, logger="juniper_cascor.api.websocket"):
            assert mgr.maybe_log_emission_summary() is None
            mgr.record_out_of_band_send({"type": "ping"})
        assert not any("WS emission summary" in rec.message for rec in caplog.records)

        with caplog.at_level(logging.INFO, logger="juniper_cascor.api.websocket"):
            deltas = mgr.maybe_log_emission_summary(force=True)
        assert deltas == {"ping": 2}


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


# ═══════════════════════════════════════════════════════════════════════════
# Phase C-5 (per-file coverage rollout, PR-2 — websocket layer): remaining
# uncovered manager branches — per-endpoint bookkeeping, per-IP accounting,
# pending-connection rejection, and the defensive metric-emission guards.
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestManagerConstructionGuards:
    """Defensive metric-emission guard in ``__init__``."""

    def test_replay_capacity_emit_failure_swallowed(self):
        """A failing ``ws_set_replay_buffer_capacity`` never blocks construction (lines 157-158)."""
        with patch("api.observability.ws_set_replay_buffer_capacity", side_effect=RuntimeError("emit down")):
            mgr = WebSocketManager(max_replay_buffer_size=256)
        assert mgr._replay_buffer_max_size == 256


@pytest.mark.unit
class TestEndpointGauge:
    """``_emit_endpoint_gauge`` unknown-endpoint short-circuit + emission guard."""

    def test_unknown_endpoint_is_noop(self):
        """An unregistered endpoint returns before emitting (line 193)."""
        mgr = WebSocketManager()
        # No bucket for this endpoint → early return, no exception.
        mgr._emit_endpoint_gauge("does-not-exist")

    def test_emit_failure_swallowed(self):
        """A failing ``ws_set_connections_active`` is swallowed (lines 198-199)."""
        mgr = WebSocketManager()
        with patch("api.observability.ws_set_connections_active", side_effect=RuntimeError("emit down")):
            mgr._emit_endpoint_gauge("training")


@pytest.mark.unit
class TestSourceIpAndPerIpAccounting:
    """``_source_ip`` fallbacks and ``_release_per_ip_slot`` branches."""

    def test_source_ip_subscript_error_returns_unknown(self):
        """A non-subscriptable ``client`` yields the unknown sentinel (lines 260-261)."""
        mgr = WebSocketManager()
        ws = MagicMock()
        ws.client = 12345  # int → client[0] raises TypeError
        assert mgr._source_ip(ws) == "unknown"

    def test_source_ip_none_client_returns_unknown(self):
        """A ``None`` client yields the unknown sentinel (line 262)."""
        mgr = WebSocketManager()
        ws = MagicMock()
        ws.client = None
        assert mgr._source_ip(ws) == "unknown"

    def test_release_per_ip_slot_empty_is_noop(self):
        """Releasing a falsy source-IP is a no-op (line 279)."""
        mgr = WebSocketManager()
        mgr._release_per_ip_slot(None)
        mgr._release_per_ip_slot("")

    def test_release_per_ip_slot_decrements_when_multiple(self):
        """Releasing an IP with count > 1 decrements rather than popping (line 284)."""
        mgr = WebSocketManager()
        mgr._per_ip_counts["10.0.0.5"] = 3
        mgr._release_per_ip_slot("10.0.0.5")
        assert mgr._per_ip_counts["10.0.0.5"] == 2


@pytest.mark.unit
class TestConnectPendingRejection:
    """``connect_pending`` connection-limit rejection paths."""

    @pytest.mark.asyncio
    async def test_rejected_when_max_connections_reached(self):
        """A pending connect past the global cap is rejected (lines 333-335)."""
        mgr = WebSocketManager(max_connections=1)
        ws1 = AsyncMock()
        await mgr.connect(ws1)  # fills the single active slot

        ws2 = AsyncMock()
        result = await mgr.connect_pending(ws2)

        assert result is False
        ws2.close.assert_awaited_once_with(code=1013, reason="Maximum connections reached")

    @pytest.mark.asyncio
    async def test_rejected_when_per_ip_limit_reached(self):
        """A pending connect past the per-IP cap is rejected (lines 339-340, 345)."""
        mgr = WebSocketManager(max_connections=50, max_connections_per_ip=1)
        ws1 = AsyncMock()
        ws1.client = ("10.0.0.5", 1111)
        assert await mgr.connect_pending(ws1) is True  # reserves the only per-IP slot

        ws2 = AsyncMock()
        ws2.client = ("10.0.0.5", 2222)  # same IP
        result = await mgr.connect_pending(ws2)

        assert result is False
        ws2.close.assert_awaited_once_with(code=1013, reason="Per-IP connection limit reached")


@pytest.mark.unit
class TestAssignSeqEmissionGuard:
    """``_assign_seq_and_buffer`` defensive emission guard."""

    def test_emit_failure_swallowed(self):
        """A failing seq/occupancy emission never blocks assignment (lines 419-420)."""
        mgr = WebSocketManager()
        with patch("api.observability.ws_set_seq_current", side_effect=RuntimeError("emit down")):
            enriched = mgr._assign_seq_and_buffer({"type": "metrics", "data": {}})
        assert enriched["seq"] == 1
        assert "emitted_at_monotonic" in enriched


@pytest.mark.unit
class TestChunkSerializationGuard:
    """``_maybe_chunk_message`` unserializable-message fallback."""

    def test_unserializable_message_returns_original(self):
        """A message that cannot be serialized is returned unchunked (lines 477, 479)."""
        mgr = WebSocketManager(max_message_size_bytes=60_000)
        msg = {"type": "topology"}
        msg["self"] = msg  # circular reference → json.dumps raises ValueError

        result = mgr._maybe_chunk_message(msg)

        assert len(result) == 1
        assert result[0] is msg


@pytest.mark.unit
class TestBroadcastFromThreadGuards:
    """``broadcast_from_thread`` submit-failure + ``_log_broadcast_exception`` arms."""

    def test_submit_failure_closes_coroutine(self):
        """A failing run_coroutine_threadsafe closes the coroutine and logs (lines 548-550)."""
        mgr = WebSocketManager()
        loop = MagicMock()
        loop.is_closed.return_value = False
        mgr.set_event_loop(loop)

        with patch("api.websocket.manager.asyncio.run_coroutine_threadsafe", side_effect=RuntimeError("submit failed")):
            # Must not raise; the orphaned coroutine is closed to avoid a warning.
            mgr.broadcast_from_thread({"type": "test"})

    def test_log_broadcast_exception_cancelled_is_ignored(self):
        """A cancelled broadcast future is ignored (lines 563-564)."""
        future = MagicMock()
        future.exception.side_effect = CancelledError()
        # Static method — no manager instance needed.
        WebSocketManager._log_broadcast_exception(future)

    def test_log_broadcast_exception_emit_failure_swallowed(self):
        """A failing error-counter emission is swallowed (lines 571-572)."""
        future = MagicMock()
        future.exception.return_value = RuntimeError("broadcast failed")
        with patch("api.observability.ws_inc_broadcast_from_thread_errors", side_effect=RuntimeError("emit down")):
            WebSocketManager._log_broadcast_exception(future)


@pytest.mark.unit
class TestSendJsonEmissionGuards:
    """``_send_json`` defensive emission + accounting guards."""

    @pytest.mark.asyncio
    async def test_timeout_broadcast_timeout_emit_failure_swallowed(self):
        """A failing broadcast-timeout counter emission is swallowed on timeout (lines 626-627)."""
        mgr = WebSocketManager(send_timeout_seconds=0.5)
        ws = AsyncMock()
        ws.send_json = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("api.observability.ws_inc_broadcast_timeout", side_effect=RuntimeError("emit down")):
            result = await mgr._send_json(ws, {"type": "metrics", "data": {}})

        assert result is False
        assert mgr.transport_stats()["send_failures"] >= 1

    @pytest.mark.asyncio
    async def test_broadcast_duration_emit_failure_swallowed(self):
        """A failing broadcast-send-duration emission is swallowed (lines 643-644)."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        ws.send_json = AsyncMock(return_value=None)

        with patch("api.observability._ensure_ws_metrics", side_effect=RuntimeError("emit down")):
            result = await mgr._send_json(ws, {"type": "metrics", "data": {}})

        assert result is True

    @pytest.mark.asyncio
    async def test_accounting_serialize_failure_uses_zero_bytes(self):
        """An unserializable message accounts as zero bytes, not a failure (lines 652-653)."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        ws.send_json = AsyncMock(return_value=None)
        msg = {"type": "metrics"}
        msg["self"] = msg  # circular reference → accounting json.dumps raises

        result = await mgr._send_json(ws, msg)

        assert result is True
        # The message still counts even though its byte size could not be computed.
        assert mgr.transport_stats()["messages_sent_total"] == 1
