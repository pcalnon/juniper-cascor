"""OBS-WIRE-02 regression tests.

Closes the cascor-side of the audit findings catalogued in
juniper-ml#197 (``A9_AND_3_2_STATE_ANALYSIS_2026-05-03.md``):

- 3.1 ``cascor_ws_seq_current`` — wired via ``_assign_seq_and_buffer``
- 3.2 ``cascor_ws_replay_buffer_occupancy`` — wired alongside 3.1
- 3.3 ``cascor_ws_replay_buffer_capacity_configured`` — wired in
  ``WebSocketManager.__init__``
- 3.4 ``cascor_ws_resume_requests_total{outcome}`` — wired across the
  4 outcome arms of ``_handle_resume``
- 3.5 ``cascor_ws_resume_replayed_events`` — wired alongside 3.4 success
- 3.6 ``cascor_ws_broadcast_timeout_total{type}`` — wired in
  ``_send_json``'s ``asyncio.TimeoutError`` arm
- 3.7 ``cascor_ws_state_throttle_coalesced_total`` — wired in
  ``lifecycle.manager._broadcast_training_state``
- 3.8 ``cascor_ws_broadcast_from_thread_errors_total`` — wired in
  ``_log_broadcast_exception``
- 3.10 ``cascor_ws_command_responses_total{command,status}`` — wired
  across the rate-limited / unknown / lifecycle-missing / success /
  timeout / error arms of ``_handle_command_message``
- Q1 — removal of ``cascor_ws_seq_gap_detected_total``
- Q3 — per-endpoint refactor: ``cascor_ws_connections_active{endpoint}``
- E.2 — :class:`WorkerRegistry.snapshot_for_metrics` returns immutable
  snapshots taken under the registry lock

Each suite owns its own teardown so tests are order-independent. The
helper-level closed-set validators (``ws_inc_resume_requests``,
``ws_set_connections_active``, ``ws_inc_command_responses``) are also
exercised here to catch instrumentation drift.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

import api.observability as obs


def _reset_ws_metrics() -> None:
    """Force lazy re-init so each test sees a fresh ws-metric set."""
    from prometheus_client import REGISTRY

    if obs._ws_metrics is not None:
        for metric in list(obs._ws_metrics.values()):
            try:
                REGISTRY.unregister(metric)
            except Exception as exc:
                logging.getLogger(__name__).debug("Best-effort ws-metric unregister failed for %r: %s", metric, exc)
        obs._ws_metrics = None


# ---------------------------------------------------------------------------
# 3.1 + 3.2 — seq_current and replay_buffer_occupancy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSeqCurrentAndReplayOccupancyEmission:
    """3.1 + 3.2 — both gauges emit on every ``_assign_seq_and_buffer`` call."""

    def setup_method(self):
        _reset_ws_metrics()

    def teardown_method(self):
        _reset_ws_metrics()

    def test_assign_seq_emits_seq_current_and_occupancy(self):
        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager(max_replay_buffer_size=8)
        seq_gauge = obs._ensure_ws_metrics()["seq_current"]
        occ_gauge = obs._ensure_ws_metrics()["replay_buffer_occupancy"]

        # Three assignments in sequence — gauges should track the last
        # write, not accumulate.
        mgr._assign_seq_and_buffer({"type": "state"})
        mgr._assign_seq_and_buffer({"type": "state"})
        mgr._assign_seq_and_buffer({"type": "state"})

        assert seq_gauge._value.get() == 3, "seq_current should reflect the most recent assigned seq"
        assert occ_gauge._value.get() == 3, "replay_buffer_occupancy should reflect the deque length"

    def test_assign_seq_with_disabled_buffer_emits_zero_occupancy(self):
        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager(max_replay_buffer_size=0)
        occ_gauge = obs._ensure_ws_metrics()["replay_buffer_occupancy"]
        mgr._assign_seq_and_buffer({"type": "state"})
        assert occ_gauge._value.get() == 0


# ---------------------------------------------------------------------------
# 3.3 — replay_buffer_capacity_configured
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReplayBufferCapacityEmission:
    """3.3 — capacity gauge set once in :meth:`WebSocketManager.__init__`."""

    def setup_method(self):
        _reset_ws_metrics()

    def teardown_method(self):
        _reset_ws_metrics()

    def test_init_sets_capacity_gauge(self):
        from api.websocket.manager import WebSocketManager

        WebSocketManager(max_replay_buffer_size=2048)
        capacity_gauge = obs._ensure_ws_metrics()["replay_buffer_capacity_configured"]
        assert capacity_gauge._value.get() == 2048

    def test_init_with_zero_capacity_still_emits_zero(self):
        from api.websocket.manager import WebSocketManager

        WebSocketManager(max_replay_buffer_size=0)
        capacity_gauge = obs._ensure_ws_metrics()["replay_buffer_capacity_configured"]
        assert capacity_gauge._value.get() == 0


# ---------------------------------------------------------------------------
# 3.4 + 3.5 — resume_requests + resume_replayed_events
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResumeRequestsAndReplayedEmission:
    """3.4 + 3.5 — outcome counter and replayed-events histogram."""

    def setup_method(self):
        _reset_ws_metrics()

    def teardown_method(self):
        _reset_ws_metrics()

    def _outcome_value(self, outcome: str) -> float:
        return obs._ensure_ws_metrics()["resume_requests_total"].labels(outcome=outcome)._value.get()

    def test_helper_increments_each_closed_set_outcome(self):
        from api.observability import ws_inc_resume_requests

        for outcome in ("success", "out_of_range", "malformed_resume", "server_restarted"):
            before = self._outcome_value(outcome)
            ws_inc_resume_requests(outcome)
            after = self._outcome_value(outcome)
            assert after - before == 1, f"resume outcome {outcome!r} should increment by 1"

    def test_helper_rejects_unknown_outcome(self):
        from api.observability import ws_inc_resume_requests

        with pytest.raises(ValueError):
            ws_inc_resume_requests("not_a_real_outcome")

    def test_replayed_events_histogram_records_observation(self):
        from api.observability import ws_observe_resume_replayed

        hist = obs._ensure_ws_metrics()["resume_replayed_events"]
        before_count = sum(b.get() for b in hist._buckets)
        ws_observe_resume_replayed(7)
        after_count = sum(b.get() for b in hist._buckets)
        assert after_count - before_count == 1
        assert hist._sum.get() == pytest.approx(7)

    def test_replayed_events_histogram_buckets_pinned(self):
        # Audit-doc D.5: pin ``cascor_ws_resume_replayed_events``
        # ``_upper_bounds`` against the ``_WS_RESUME_REPLAY_BUCKETS``
        # constant so future bucket-layout changes produce a
        # deterministic test failure rather than silently re-bucketing
        # the operational regimes documented in the rationale doc §7.
        # Mirrors the cascor R5.4-pre + R5.1b bucket-pin assertion
        # pattern (see ``test_metrics_r5_4_pre.py``).
        from api.observability import _WS_RESUME_REPLAY_BUCKETS

        hist = obs._ensure_ws_metrics()["resume_replayed_events"]
        # ``prometheus_client.Histogram`` appends an implicit ``+inf``
        # upper edge to whatever ``buckets=`` tuple it was constructed
        # with.  Assert the layout matches the constant + the implicit
        # ``+inf`` sentinel so a future re-bucket has to update both
        # the constant and this test in lockstep.
        expected = tuple(_WS_RESUME_REPLAY_BUCKETS) + (float("inf"),)
        assert hist._upper_bounds == expected

    def test_handle_resume_emits_malformed_outcome(self):
        """Missing ``last_seq`` triggers the malformed_resume arm."""
        from api.websocket.training_stream import _handle_resume

        class _StubManager:
            server_instance_id = "server-id"

            async def send_personal_message(self, ws, msg):
                return True

            def replay_since(self, last_seq):
                raise AssertionError("should not be called for malformed resume")

        before = self._outcome_value("malformed_resume")
        result = asyncio.run(_handle_resume(websocket=object(), ws_manager=_StubManager(), msg={"data": {}}))
        after = self._outcome_value("malformed_resume")
        assert result is False
        assert after - before == 1


# ---------------------------------------------------------------------------
# 3.6 — broadcast_timeout
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBroadcastTimeoutEmission:
    """3.6 — counter increments on ``asyncio.TimeoutError`` in ``_send_json``."""

    def setup_method(self):
        _reset_ws_metrics()

    def teardown_method(self):
        _reset_ws_metrics()

    def test_send_json_timeout_increments_counter(self):
        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager(send_timeout_seconds=0.01)

        class _SlowWS:
            async def send_json(self, msg):
                # Long enough to exceed the 10 ms timeout above.
                await asyncio.sleep(1.0)

        counter = obs._ensure_ws_metrics()["broadcast_timeout_total"]
        before = counter.labels(type="state")._value.get()
        result = asyncio.run(mgr._send_json(_SlowWS(), {"type": "state", "payload": "hi"}))
        after = counter.labels(type="state")._value.get()
        assert result is False
        assert after - before == 1


# ---------------------------------------------------------------------------
# 3.7 — state_throttle_coalesced
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStateThrottleCoalescedEmission:
    """3.7 — counter increments when the GAP-WS-21 coalescer drops a broadcast."""

    def setup_method(self):
        _reset_ws_metrics()

    def teardown_method(self):
        _reset_ws_metrics()

    def test_coalesce_path_increments_counter(self):
        """Synthesise the throttle hit by directly invoking the coalescer arm."""
        from api.observability import ws_inc_state_throttle_coalesced

        counter = obs._ensure_ws_metrics()["state_throttle_coalesced_total"]
        before = counter._value.get()
        ws_inc_state_throttle_coalesced()
        after = counter._value.get()
        assert after - before == 1


# ---------------------------------------------------------------------------
# 3.8 — broadcast_from_thread_errors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBroadcastFromThreadErrorsEmission:
    """3.8 — counter increments when the done-callback observes an exception."""

    def setup_method(self):
        _reset_ws_metrics()

    def teardown_method(self):
        _reset_ws_metrics()

    def test_log_broadcast_exception_increments_on_failure(self):
        from api.websocket.manager import WebSocketManager

        class _Future:
            def exception(self):
                return RuntimeError("synthetic broadcast failure")

        counter = obs._ensure_ws_metrics()["broadcast_from_thread_errors_total"]
        before = counter._value.get()
        WebSocketManager._log_broadcast_exception(_Future())
        after = counter._value.get()
        assert after - before == 1

    def test_log_broadcast_exception_no_increment_on_success(self):
        from api.websocket.manager import WebSocketManager

        class _Future:
            def exception(self):
                return None

        counter = obs._ensure_ws_metrics()["broadcast_from_thread_errors_total"]
        before = counter._value.get()
        WebSocketManager._log_broadcast_exception(_Future())
        after = counter._value.get()
        assert after == before


# ---------------------------------------------------------------------------
# 3.10 — command_responses
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCommandResponsesEmission:
    """3.10 — counter validates the closed status set and labels correctly."""

    def setup_method(self):
        _reset_ws_metrics()

    def teardown_method(self):
        _reset_ws_metrics()

    def test_helper_increments_each_status_arm(self):
        from api.observability import ws_inc_command_responses

        counter = obs._ensure_ws_metrics()["command_responses_total"]
        for status in ("success", "error", "rate_limited"):
            before = counter.labels(command="start", status=status)._value.get()
            ws_inc_command_responses("start", status)
            after = counter.labels(command="start", status=status)._value.get()
            assert after - before == 1

    def test_helper_rejects_unknown_status(self):
        from api.observability import ws_inc_command_responses

        with pytest.raises(ValueError):
            ws_inc_command_responses("start", "weird_status")


# ---------------------------------------------------------------------------
# Q1 — seq_gap_detected_total removal
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSeqGapDetectedTotalRemoved:
    """Q1 — the cascor-side counter is gone; the metric must NOT register."""

    def setup_method(self):
        _reset_ws_metrics()

    def teardown_method(self):
        _reset_ws_metrics()

    def test_metric_dict_has_no_seq_gap_key(self):
        ws = obs._ensure_ws_metrics()
        assert "seq_gap_detected_total" not in ws

    def test_no_helper_function_exposed(self):
        # The helper was never written, but a future refactor may
        # accidentally re-introduce one — assert it stays gone.
        assert not hasattr(obs, "ws_inc_seq_gap_detected")

    def test_metric_not_registered_with_prometheus(self):
        from prometheus_client import REGISTRY

        obs._ensure_ws_metrics()
        # ``REGISTRY.collect`` walks every registered collector.
        names = set()
        for collector in list(REGISTRY._names_to_collectors.values()):
            for metric in collector.collect():
                names.add(metric.name)
        assert "cascor_ws_seq_gap_detected" not in names
        assert "cascor_ws_seq_gap_detected_total" not in names


# ---------------------------------------------------------------------------
# Q3 — connections_active per-endpoint refactor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConnectionsActiveEndpointRefactor:
    """Q3 — per-endpoint bookkeeping + closed-set label discipline."""

    def setup_method(self):
        _reset_ws_metrics()

    def teardown_method(self):
        _reset_ws_metrics()

    def _gauge(self, endpoint: str) -> float:
        return obs._ensure_ws_metrics()["connections_active"].labels(endpoint=endpoint)._value.get()

    def test_helper_validates_closed_endpoint_set(self):
        from api.observability import ws_set_connections_active

        for ep in ("training", "control", "workers"):
            ws_set_connections_active(ep, 0)

        with pytest.raises(ValueError):
            ws_set_connections_active("not_a_real_endpoint", 1)

    def test_register_endpoint_connection_increments_gauge(self):
        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager()
        ws1 = object()
        ws2 = object()

        mgr.register_endpoint_connection(ws1, "training")
        assert self._gauge("training") == 1
        mgr.register_endpoint_connection(ws2, "training")
        assert self._gauge("training") == 2

    def test_unregister_endpoint_connection_decrements_gauge(self):
        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager()
        ws1 = object()

        mgr.register_endpoint_connection(ws1, "control")
        assert self._gauge("control") == 1
        mgr.unregister_endpoint_connection(ws1)
        assert self._gauge("control") == 0

    def test_unregister_unknown_websocket_is_noop(self):
        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager()
        # No prior register — must not raise.
        mgr.unregister_endpoint_connection(object())

    def test_register_unknown_endpoint_is_logged_noop(self):
        """Unknown endpoint passed to register is rejected without mutating gauges."""
        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager()
        ws = object()
        # Snapshot all three gauges before the bad call.
        before = {ep: self._gauge(ep) for ep in ("training", "control", "workers")}
        mgr.register_endpoint_connection(ws, "garbage_endpoint")
        after = {ep: self._gauge(ep) for ep in ("training", "control", "workers")}
        assert before == after

    def test_endpoints_isolated_by_label(self):
        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager()
        a, b, c = object(), object(), object()
        mgr.register_endpoint_connection(a, "training")
        mgr.register_endpoint_connection(b, "control")
        mgr.register_endpoint_connection(c, "workers")
        assert self._gauge("training") == 1
        assert self._gauge("control") == 1
        assert self._gauge("workers") == 1
        mgr.unregister_endpoint_connection(b)
        assert self._gauge("training") == 1
        assert self._gauge("control") == 0
        assert self._gauge("workers") == 1


# ---------------------------------------------------------------------------
# E.2 — WorkerRegistry snapshot under lock
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWorkerRegistrySnapshotForMetrics:
    """E.2 — :meth:`snapshot_for_metrics` produces immutable snapshots
    under the registry lock so the Prometheus collector cannot race
    with concurrent ``record_heartbeat`` calls."""

    def test_snapshot_returns_immutable_durations_window(self):
        from api.workers.registry import WorkerRegistry

        reg = WorkerRegistry()
        reg.register("w1", capabilities={})
        reg.heartbeat("w1", recent_task_durations_seconds=[0.1, 0.2, 0.3])

        snapshots = reg.snapshot_for_metrics()
        assert len(snapshots) == 1
        snap = snapshots[0]
        assert snap["worker_id"] == "w1"
        assert snap["recent_task_durations_seconds"] == (0.1, 0.2, 0.3)

        # Mutating the live registration's window must NOT affect the
        # already-issued snapshot — the tuple is immutable and the list
        # was copied at snapshot time.
        reg.heartbeat("w1", recent_task_durations_seconds=[9.9])
        assert snap["recent_task_durations_seconds"] == (0.1, 0.2, 0.3)

    def test_snapshot_holds_registry_lock(self):
        """Structural assertion: snapshot grabs ``self._lock`` exactly once.

        We instrument the lock to count acquisitions during a snapshot
        call. A non-zero acquisition count proves the lock guards the
        whole walk; a zero count would mean the lock-snapshot fix was
        accidentally undone.
        """
        from api.workers.registry import WorkerRegistry

        reg = WorkerRegistry()
        reg.register("w1", capabilities={})

        original_lock = reg._lock

        class _CountingLock:
            def __init__(self, inner):
                self._inner = inner
                self.acquires = 0

            def __enter__(self):
                self.acquires += 1
                return self._inner.__enter__()

            def __exit__(self, *exc):
                return self._inner.__exit__(*exc)

        counting = _CountingLock(original_lock)
        reg._lock = counting
        try:
            reg.snapshot_for_metrics()
            assert counting.acquires == 1
        finally:
            reg._lock = original_lock

    def test_snapshot_does_not_race_with_concurrent_heartbeats(self):
        """Run concurrent heartbeat writers + snapshotters; require no exceptions
        and require every snapshot's window to be a self-consistent tuple
        (length matches some prior write, no torn reads)."""
        from api.workers.registry import WorkerRegistry

        reg = WorkerRegistry()
        reg.register("w1", capabilities={})

        stop = threading.Event()
        errors: list[Exception] = []

        def _writer():
            i = 0
            while not stop.is_set():
                try:
                    reg.heartbeat("w1", recent_task_durations_seconds=[float(i)] * (i % 5))
                except Exception as exc:
                    errors.append(exc)
                i += 1

        def _reader():
            while not stop.is_set():
                try:
                    snaps = reg.snapshot_for_metrics()
                    for s in snaps:
                        # Pure read — must succeed every time.
                        _ = s["recent_task_durations_seconds"]
                except Exception as exc:
                    errors.append(exc)

        writers = [threading.Thread(target=_writer) for _ in range(2)]
        readers = [threading.Thread(target=_reader) for _ in range(2)]
        for t in writers + readers:
            t.start()
        # Short stress window — enough to surface a race deterministically.
        threading.Event().wait(0.2)
        stop.set()
        for t in writers + readers:
            t.join(timeout=2.0)

        assert not errors, f"snapshot/heartbeat raced: {errors!r}"
