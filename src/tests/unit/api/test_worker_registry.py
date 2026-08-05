"""Tests for WorkerRegistry — thread-safe worker connection tracking."""

import time
from unittest.mock import patch

import pytest

from api.workers.registry import WorkerRegistration, WorkerRegistry


@pytest.mark.unit
class TestWorkerRegistration:
    """Test WorkerRegistration dataclass."""

    def test_default_values(self):
        """New registration has sensible defaults."""
        reg = WorkerRegistration(worker_id="w1")
        assert reg.worker_id == "w1"
        assert reg.tasks_completed == 0
        assert reg.tasks_failed == 0
        assert reg.active_task_id is None
        assert reg.idle is True
        assert reg.health_score == 1.0

    def test_health_score_calculation(self):
        """Health score reflects success/failure ratio."""
        reg = WorkerRegistration(worker_id="w1")
        reg.tasks_completed = 8
        reg.tasks_failed = 2
        assert reg.health_score == pytest.approx(0.8)

    def test_health_score_all_failures(self):
        """Health score is 0 when all tasks fail."""
        reg = WorkerRegistration(worker_id="w1")
        reg.tasks_failed = 5
        assert reg.health_score == 0.0

    def test_idle_property(self):
        """idle is False when a task is assigned."""
        reg = WorkerRegistration(worker_id="w1")
        reg.active_task_id = "task-1"
        assert reg.idle is False

    def test_heartbeat_updates_timestamp(self):
        """record_heartbeat updates last_heartbeat."""
        reg = WorkerRegistration(worker_id="w1")
        old_hb = reg.last_heartbeat
        time.sleep(0.01)
        reg.record_heartbeat()
        assert reg.last_heartbeat > old_hb

    def test_is_alive(self):
        """Worker is alive if heartbeat is recent."""
        reg = WorkerRegistration(worker_id="w1")
        assert reg.is_alive(timeout=30.0) is True

    def test_is_not_alive(self):
        """Worker is not alive if heartbeat is stale."""
        reg = WorkerRegistration(worker_id="w1")
        reg.last_heartbeat = time.time() - 60
        assert reg.is_alive(timeout=30.0) is False


@pytest.mark.unit
class TestWorkerRegistrationR1_3:
    """METRICS-MON R1.3 / seed-04: enriched heartbeat fields."""

    def test_enriched_field_defaults(self):
        """New registration defaults enriched fields to 0/None."""
        reg = WorkerRegistration(worker_id="w1")
        assert reg.in_flight_tasks == 0
        assert reg.last_task_completed_at is None
        assert reg.rss_mb is None

    def test_record_heartbeat_minimal_preserves_enriched(self):
        """Backward compat: minimal heartbeat (no kwargs) preserves prior enriched values."""
        reg = WorkerRegistration(worker_id="w1")
        reg.in_flight_tasks = 3
        reg.last_task_completed_at = 1745816350.0
        reg.rss_mb = 412.7
        reg.record_heartbeat()
        # Prior values preserved
        assert reg.in_flight_tasks == 3
        assert reg.last_task_completed_at == 1745816350.0
        assert reg.rss_mb == 412.7

    def test_record_heartbeat_with_enriched_fields(self):
        """R1.3-aware worker reports enriched fields → applied atomically."""
        reg = WorkerRegistration(worker_id="w1")
        reg.record_heartbeat(in_flight_tasks=2, last_task_completed_at=1745816400.0, rss_mb=512.0)
        assert reg.in_flight_tasks == 2
        assert reg.last_task_completed_at == 1745816400.0
        assert reg.rss_mb == 512.0

    def test_record_heartbeat_partial_kwargs(self):
        """Mixed minimal/enriched: only the provided fields update."""
        reg = WorkerRegistration(worker_id="w1")
        reg.in_flight_tasks = 5
        reg.rss_mb = 100.0
        reg.record_heartbeat(in_flight_tasks=1)  # only in_flight_tasks updated
        assert reg.in_flight_tasks == 1
        assert reg.rss_mb == 100.0  # preserved
        assert reg.last_task_completed_at is None  # preserved

    def test_record_heartbeat_updates_task_counters(self):
        """tasks_completed/failed kwargs update the counters."""
        reg = WorkerRegistration(worker_id="w1")
        reg.record_heartbeat(tasks_completed=10, tasks_failed=2)
        assert reg.tasks_completed == 10
        assert reg.tasks_failed == 2


@pytest.mark.unit
class TestWorkerRegistrationR4_4:
    """METRICS-MON R4.4: training-loop instrumentation fields.

    Mirrors R1.3 test patterns above — defaults, minimal-heartbeat
    preserves prior values, kwargs update only what's provided.
    """

    def test_r4_4_field_defaults(self):
        """New registration defaults R4.4 fields to None / [] / None."""
        reg = WorkerRegistration(worker_id="w1")
        assert reg.last_task_duration_seconds is None
        assert reg.recent_task_durations_seconds == []
        assert reg.gpu_utilization_pct is None

    def test_record_heartbeat_minimal_preserves_r4_4_fields(self):
        """Pre-R4.4 worker (minimal heartbeat) preserves prior R4.4 values.

        Backward-compat regression guard: if a worker is upgraded to R4.4,
        downgraded to pre-R4.4, then sends a minimal heartbeat, the prior
        R4.4 values must persist (matches the R1.3 pattern). Operators
        watching the dashboard for the percentile window during a
        rolling restart should not see the values reset to defaults.
        """
        reg = WorkerRegistration(worker_id="w1")
        reg.last_task_duration_seconds = 0.42
        reg.recent_task_durations_seconds = [0.1, 0.2, 0.42]
        reg.gpu_utilization_pct = 67.5
        reg.record_heartbeat()
        assert reg.last_task_duration_seconds == 0.42
        assert reg.recent_task_durations_seconds == [0.1, 0.2, 0.42]
        assert reg.gpu_utilization_pct == 67.5

    def test_record_heartbeat_with_r4_4_fields(self):
        """R4.4-aware worker reports all 3 fields → applied atomically."""
        reg = WorkerRegistration(worker_id="w1")
        reg.record_heartbeat(
            last_task_duration_seconds=1.25,
            recent_task_durations_seconds=[0.5, 0.8, 1.25],
            gpu_utilization_pct=88.0,
        )
        assert reg.last_task_duration_seconds == 1.25
        assert reg.recent_task_durations_seconds == [0.5, 0.8, 1.25]
        assert reg.gpu_utilization_pct == 88.0

    def test_record_heartbeat_recent_durations_defensive_copy(self):
        """Caller's list mutation must not leak through to the registration.

        Pinning this matters because the worker's deque is converted to a
        list right before send; if cascor stored a shared reference, a
        future refactor that reuses the list buffer (e.g. via msgpack
        zero-copy) would silently mutate registered worker state.
        """
        reg = WorkerRegistration(worker_id="w1")
        caller_list = [0.1, 0.2, 0.3]
        reg.record_heartbeat(recent_task_durations_seconds=caller_list)
        caller_list.append(999.0)
        assert reg.recent_task_durations_seconds == [0.1, 0.2, 0.3]

    def test_record_heartbeat_partial_r4_4_kwargs(self):
        """Mixed: only the provided R4.4 fields update; others preserved."""
        reg = WorkerRegistration(worker_id="w1")
        reg.last_task_duration_seconds = 0.5
        reg.gpu_utilization_pct = 50.0
        reg.record_heartbeat(last_task_duration_seconds=0.8)
        assert reg.last_task_duration_seconds == 0.8
        assert reg.gpu_utilization_pct == 50.0  # preserved
        assert reg.recent_task_durations_seconds == []  # never set

    def test_registry_heartbeat_forwards_r4_4_fields(self):
        """``WorkerRegistry.heartbeat()`` forwards R4.4 kwargs to ``record_heartbeat``."""
        registry = WorkerRegistry()
        registry.register("w1", {"cpu_cores": 8})
        ok = registry.heartbeat(
            "w1",
            last_task_duration_seconds=2.5,
            recent_task_durations_seconds=[1.0, 2.5],
            gpu_utilization_pct=90.0,
        )
        assert ok is True
        reg = registry.get("w1")
        assert reg.last_task_duration_seconds == 2.5
        assert reg.recent_task_durations_seconds == [1.0, 2.5]
        assert reg.gpu_utilization_pct == 90.0


@pytest.mark.unit
class TestWorkerRegistry:
    """Test WorkerRegistry thread-safe operations."""

    def test_register(self):
        """Register adds a worker."""
        registry = WorkerRegistry()
        reg = registry.register("w1", {"cpu_cores": 4})
        assert reg.worker_id == "w1"
        assert registry.worker_count == 1

    def test_deregister(self):
        """Deregister removes a worker."""
        registry = WorkerRegistry()
        registry.register("w1", {})
        removed = registry.deregister("w1")
        assert removed is not None
        assert registry.worker_count == 0

    def test_deregister_nonexistent(self):
        """Deregistering a non-existent worker returns None."""
        registry = WorkerRegistry()
        assert registry.deregister("nope") is None

    def test_register_replaces_existing(self):
        """Re-registering the same worker_id replaces the old registration."""
        registry = WorkerRegistry()
        registry.register("w1", {"gpu": False})
        registry.register("w1", {"gpu": True})
        assert registry.worker_count == 1
        reg = registry.get("w1")
        assert reg.capabilities["gpu"] is True

    def test_register_replacement_while_busy_resets_to_idle(self):
        """Re-registering a busy worker replaces with a fresh idle registration.

        Pins the double-dispatch footgun: ``register`` builds a new
        ``WorkerRegistration`` (``active_task_id=None``) even when the prior
        entry held an in-flight task. Coordinators that still track the old
        assignment can then ``get_next_assignment`` / ``assign_task`` again
        because the registry reports the worker idle.
        """
        registry = WorkerRegistry()
        registry.register("w1", {"cpu_cores": 4})
        assert registry.assign_task("w1", "task-busy-1") is True
        assert registry.get("w1").active_task_id == "task-busy-1"
        assert registry.get("w1").idle is False

        replaced = registry.register("w1", {"cpu_cores": 8})
        assert replaced.worker_id == "w1"
        assert replaced.capabilities["cpu_cores"] == 8
        assert replaced.active_task_id is None
        assert replaced.idle is True
        assert registry.get("w1") is replaced
        # Fresh idle entry accepts a second assignment under the same worker_id.
        assert registry.assign_task("w1", "task-busy-2") is True
        assert registry.get("w1").active_task_id == "task-busy-2"

    def test_get(self):
        """get returns the registration or None."""
        registry = WorkerRegistry()
        assert registry.get("w1") is None
        registry.register("w1", {})
        assert registry.get("w1") is not None

    def test_heartbeat(self):
        """heartbeat updates timestamp and returns True."""
        registry = WorkerRegistry()
        registry.register("w1", {})
        assert registry.heartbeat("w1") is True

    def test_heartbeat_unknown_worker(self):
        """heartbeat returns False for unknown worker."""
        registry = WorkerRegistry()
        assert registry.heartbeat("nope") is False

    def test_heartbeat_forwards_enriched_kwargs(self):
        """METRICS-MON R1.3 / seed-04: enriched fields on the registry call propagate to the registration."""
        registry = WorkerRegistry()
        registry.register("w1", {})
        assert (
            registry.heartbeat(
                "w1",
                in_flight_tasks=4,
                last_task_completed_at=1745816400.0,
                rss_mb=256.0,
                tasks_completed=42,
                tasks_failed=1,
            )
            is True
        )
        reg = registry.get("w1")
        assert reg is not None
        assert reg.in_flight_tasks == 4
        assert reg.last_task_completed_at == 1745816400.0
        assert reg.rss_mb == 256.0
        assert reg.tasks_completed == 42
        assert reg.tasks_failed == 1

    def test_heartbeat_minimal_preserves_prior_enriched(self):
        """METRICS-MON R1.3: backward compat — old worker's minimal heartbeat preserves prior enriched values."""
        registry = WorkerRegistry()
        registry.register("w1", {})
        registry.heartbeat("w1", in_flight_tasks=7, rss_mb=128.0)
        # Older worker upgrade rolls back, sends minimal heartbeat — prior enriched
        # values must NOT be reset to defaults.
        registry.heartbeat("w1")
        reg = registry.get("w1")
        assert reg is not None
        assert reg.in_flight_tasks == 7
        assert reg.rss_mb == 128.0

    def test_assign_task(self):
        """assign_task marks worker as busy."""
        registry = WorkerRegistry()
        registry.register("w1", {})
        assert registry.assign_task("w1", "task-1") is True
        assert registry.get("w1").idle is False

    def test_assign_task_busy_worker(self):
        """assign_task fails for already-busy worker."""
        registry = WorkerRegistry()
        registry.register("w1", {})
        registry.assign_task("w1", "task-1")
        assert registry.assign_task("w1", "task-2") is False

    def test_complete_task_success(self):
        """complete_task increments completed count."""
        registry = WorkerRegistry()
        registry.register("w1", {})
        registry.assign_task("w1", "task-1")
        assert registry.complete_task("w1", success=True) is True
        reg = registry.get("w1")
        assert reg.tasks_completed == 1
        assert reg.idle is True

    def test_complete_task_failure(self):
        """complete_task increments failed count on failure."""
        registry = WorkerRegistry()
        registry.register("w1", {})
        registry.assign_task("w1", "task-1")
        registry.complete_task("w1", success=False)
        assert registry.get("w1").tasks_failed == 1

    def test_available_worker_count(self):
        """available_worker_count counts idle, alive workers."""
        registry = WorkerRegistry()
        registry.register("w1", {})
        registry.register("w2", {})
        assert registry.available_worker_count == 2
        registry.assign_task("w1", "task-1")
        assert registry.available_worker_count == 1

    def test_get_idle_workers_sorted_by_health(self):
        """get_idle_workers returns workers sorted by health score."""
        registry = WorkerRegistry()
        registry.register("w1", {})
        registry.register("w2", {})
        # Make w1 have worse health
        reg1 = registry.get("w1")
        reg1.tasks_completed = 1
        reg1.tasks_failed = 9

        idle = registry.get_idle_workers()
        assert len(idle) == 2
        assert idle[0].worker_id == "w2"  # Better health first

    def test_get_stale_workers(self):
        """get_stale_workers returns workers with timed-out heartbeats."""
        registry = WorkerRegistry(heartbeat_timeout=0.01)
        registry.register("w1", {})
        time.sleep(0.02)
        stale = registry.get_stale_workers()
        assert len(stale) == 1
        assert stale[0].worker_id == "w1"

    def test_clear(self):
        """clear removes all workers."""
        registry = WorkerRegistry()
        registry.register("w1", {})
        registry.register("w2", {})
        count = registry.clear()
        assert count == 2
        assert registry.worker_count == 0


@pytest.mark.unit
class TestWorkerRegistrySizeCapAuditE6:
    """Audit-doc juniper-ml#195 finding E.6 — registry size cap.

    The registry was previously unbounded; a misbehaving worker pool
    or a malicious / runaway client storm could grow ``_workers``
    without limit. This test class pins the cap behavior end-to-end:

      * Default cap is 250.
      * Reaching the cap raises :class:`WorkerRegistryFullError` for a
        new ``worker_id``.
      * Re-registering an existing ``worker_id`` at the cap is allowed
        (the dict size stays unchanged).
      * Custom ``max_workers`` constructor kwarg is honored.
      * Non-positive caps are rejected at construction time.
      * The exception class is distinct so the WS handshake handler
        can catch it specifically.
    """

    def test_default_cap_is_250(self):
        from api.workers.registry import _DEFAULT_MAX_WORKERS

        assert _DEFAULT_MAX_WORKERS == 250
        registry = WorkerRegistry()
        assert registry.max_workers == 250

    def test_constructor_validates_max_workers(self):
        with pytest.raises(ValueError, match="max_workers must be positive"):
            WorkerRegistry(max_workers=0)
        with pytest.raises(ValueError, match="max_workers must be positive"):
            WorkerRegistry(max_workers=-1)

    def test_register_rejects_new_worker_at_cap(self):
        from api.workers.registry import WorkerRegistryFullError

        registry = WorkerRegistry(max_workers=3)
        registry.register("w1", {})
        registry.register("w2", {})
        registry.register("w3", {})
        assert registry.worker_count == 3

        with pytest.raises(WorkerRegistryFullError, match=r"at capacity \(3\)"):
            registry.register("w4", {})

        # The 4th attempt did NOT pollute the dict.
        assert registry.worker_count == 3
        assert "w4" not in {w.worker_id for w in registry.get_all_workers()}

    def test_re_register_at_cap_is_allowed(self):
        """Re-registering an existing worker_id keeps dict size unchanged → no raise."""
        registry = WorkerRegistry(max_workers=2)
        registry.register("w1", {"gpu": "rtx-4090"})
        registry.register("w2", {"gpu": "a100"})
        # At cap. Re-register w1 with new capabilities — replacement,
        # not new entry.
        replaced = registry.register("w1", {"gpu": "rtx-5090"})
        assert replaced.capabilities == {"gpu": "rtx-5090"}
        assert registry.worker_count == 2

    def test_deregister_below_cap_re_enables_new_registration(self):
        """After deregistering, the freed slot accepts a new worker."""
        from api.workers.registry import WorkerRegistryFullError

        registry = WorkerRegistry(max_workers=2)
        registry.register("w1", {})
        registry.register("w2", {})
        with pytest.raises(WorkerRegistryFullError):
            registry.register("w3", {})

        registry.deregister("w1")
        assert registry.worker_count == 1

        # Now w3 fits.
        registry.register("w3", {})
        assert registry.worker_count == 2

    def test_custom_cap_independent_of_default(self):
        """A registry with max_workers=500 accepts up to 500 (smoke test, capped at 5)."""
        registry = WorkerRegistry(max_workers=500)
        assert registry.max_workers == 500
        # Smoke: the first 5 registrations fit; default-cap=250 logic
        # would not affect us.
        for i in range(5):
            registry.register(f"w{i}", {})
        assert registry.worker_count == 5

    def test_full_error_is_runtime_error_subclass(self):
        """Catchable as RuntimeError for callers that don't import the specific class."""
        from api.workers.registry import WorkerRegistryFullError

        assert issubclass(WorkerRegistryFullError, RuntimeError)

    def test_clear_resets_capacity_state(self):
        """After clear(), the registry can fully repopulate to the cap."""
        from api.workers.registry import WorkerRegistryFullError

        registry = WorkerRegistry(max_workers=2)
        registry.register("w1", {})
        registry.register("w2", {})
        with pytest.raises(WorkerRegistryFullError):
            registry.register("w3", {})

        cleared = registry.clear()
        assert cleared == 2

        # All slots free again.
        registry.register("w1", {})
        registry.register("w2", {})
        assert registry.worker_count == 2
