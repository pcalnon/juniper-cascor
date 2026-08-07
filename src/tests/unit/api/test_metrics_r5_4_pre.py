"""METRICS-MON R5.4-pre regression tests.

Covers the three instrumentation gaps closed by R5.4-pre:

1. ``juniper_cascor_training_sessions_completed_total`` — closed-set
   ``status`` counter bumped at every terminal lifecycle transition.
2. ``juniper_cascor_training_step_duration_seconds`` — train-step
   duration histogram emitted from the api-lifecycle output-phase
   epoch callback.
3. :class:`api.workers.metrics.WorkerRegistryCollector` — bridge that
   snapshots :class:`WorkerRegistry` on each scrape and emits per-worker
   gauges, omitting unset fields rather than zero-emitting.
"""

import logging
import time

import pytest

from api.observability import TRAINING_SESSION_STATUS_CANCELLED, TRAINING_SESSION_STATUS_FAILURE, TRAINING_SESSION_STATUS_SUCCESS, _ensure_training_metrics, inc_training_session_completed, observe_training_step_duration


def _reset_training_metrics() -> None:
    """Force lazy re-init so each test sees a fresh metric set.

    Mirrors the pattern used by the existing R5.1b shim tests in
    ``test_api_observability.py``: drop the cached dict, unregister
    each metric from the global REGISTRY, and let the next access
    recreate the family. Keeps tests order-independent and avoids the
    "Duplicated timeseries" guard that prometheus_client raises on
    re-registration.
    """
    from prometheus_client import REGISTRY

    import api.observability as obs

    if obs._training_metrics is not None:
        for metric in list(obs._training_metrics.values()):
            try:
                REGISTRY.unregister(metric)
            except Exception as exc:
                logging.getLogger(__name__).debug("Best-effort training-metric unregister failed for %r: %s", metric, exc)
        obs._training_metrics = None


@pytest.mark.unit
class TestTrainingSessionsCompletedCounter:
    """Gap 1 — terminal-transition counter for SLO 3.3."""

    def setup_method(self):
        _reset_training_metrics()

    def teardown_method(self):
        _reset_training_metrics()

    def _value(self, status: str) -> float:
        m = _ensure_training_metrics()["sessions_completed_total"]
        return m.labels(status=status)._value.get()

    def test_each_status_increments_counter(self):
        """Every closed-set status increments by exactly 1 per call."""
        for status in (TRAINING_SESSION_STATUS_SUCCESS, TRAINING_SESSION_STATUS_FAILURE, TRAINING_SESSION_STATUS_CANCELLED):
            before = self._value(status)
            inc_training_session_completed(status)
            after = self._value(status)
            assert after - before == pytest.approx(1.0), f"counter for status={status!r} did not increment by 1"

    def test_repeated_increments_accumulate(self):
        """Multiple terminal transitions accumulate correctly."""
        before = self._value(TRAINING_SESSION_STATUS_SUCCESS)
        for _ in range(5):
            inc_training_session_completed(TRAINING_SESSION_STATUS_SUCCESS)
        after = self._value(TRAINING_SESSION_STATUS_SUCCESS)
        assert after - before == pytest.approx(5.0)

    def test_invalid_status_raises_value_error(self):
        """R1.1 cardinality discipline: open-set values are rejected."""
        with pytest.raises(ValueError, match="invalid training-session status"):
            inc_training_session_completed("rolled_back")  # not in the closed set

    def test_label_set_is_exactly_three_values(self):
        """The closed set is exactly {success, failure, cancelled}."""
        from api.observability import _TRAINING_SESSION_STATUSES

        assert _TRAINING_SESSION_STATUSES == frozenset({"success", "failure", "cancelled"})


@pytest.mark.unit
class TestTrainingStepDurationHistogram:
    """Gap 2 — train-step duration histogram for SLO 3.4."""

    EXPECTED_UPPER_BOUNDS = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float("inf")]

    def setup_method(self):
        _reset_training_metrics()

    def teardown_method(self):
        _reset_training_metrics()

    def test_buckets_match_constant(self):
        """Histogram is wired to the R5.4-pre bucket layout."""
        from api.observability import _TRAINING_STEP_DURATION_BUCKETS

        assert list(_TRAINING_STEP_DURATION_BUCKETS) == self.EXPECTED_UPPER_BOUNDS
        hist = _ensure_training_metrics()["step_duration_seconds"]
        # ``_upper_bounds`` includes the implicit ``+inf`` upper edge.
        assert hist._upper_bounds == self.EXPECTED_UPPER_BOUNDS

    def test_help_string_carries_slo_marker(self):
        """The HELP line points at SLO 3.4 and does NOT carry the R4.1 tentative suffix."""
        hist = _ensure_training_metrics()["step_duration_seconds"]
        assert "SLO 3.4" in hist._documentation
        assert "tentative pending R5.1" not in hist._documentation
        assert "(R4.1 buckets tentative pending R5.1)" not in hist._documentation

    def test_observation_lands_in_expected_bucket(self):
        """A 0.7 s sample increments the (0.5, 1.0] bucket exactly.

        ``prometheus_client``'s internal ``_buckets`` array stores a
        non-cumulative count per bucket (the WIRE-format cumulative
        view is computed at scrape time). A 0.7 s observation sits in
        the half-open interval (0.5, 1.0] and so increments the bucket
        whose upper bound is 1.0 — and ONLY that bucket.

        OBS-WIRE-01 (A.6): the histogram no longer carries a ``phase``
        label — see the metric-definition comment in
        ``api.observability``.
        """
        observe_training_step_duration(0.7)

        hist = _ensure_training_metrics()["step_duration_seconds"]
        bucket_counts = {ub: bucket.get() for ub, bucket in zip(hist._upper_bounds, hist._buckets)}

        assert bucket_counts[1.0] == 1, f"sample of 0.7 must land in the le=1.0 bucket: counts={bucket_counts}"
        for ub, count in bucket_counts.items():
            if ub == 1.0:
                continue
            assert count == 0, f"unexpected count {count} in bucket le={ub} — only le=1.0 should fire for a 0.7 s sample"

        # The cumulative contract IS still observable via _sum and the
        # final +inf observation count: every observation contributes
        # to _sum exactly once.
        assert hist._sum.get() == pytest.approx(0.7)

    def test_phase_label_dropped(self):
        """OBS-WIRE-01 (A.6): the histogram is unlabelled — no labelnames."""
        hist = _ensure_training_metrics()["step_duration_seconds"]
        # ``Histogram._labelnames`` is the public-ish accessor for the
        # declared label set; an empty tuple means the metric has no
        # labels (post-A.6 contract).
        assert hist._labelnames == (), f"expected no labels post-A.6, got {hist._labelnames!r}"
        observe_training_step_duration(0.123)
        assert hist._sum.get() == pytest.approx(0.123)


@pytest.mark.unit
class TestWorkerRegistryCollector:
    """Gap 3 — worker -> Prometheus bridge collector."""

    def _build_registry_with_two_workers(self):
        """Populate a fake registry: w1 fully-instrumented, w2 heartbeat-only."""
        from api.workers.registry import WorkerRegistry

        reg = WorkerRegistry(heartbeat_timeout=30.0)
        # w1 — fully-instrumented worker (R4.4 fields populated).
        reg.register("w1", capabilities={"gpu": "rtx-4090"})
        reg.heartbeat(
            "w1",
            in_flight_tasks=0,
            tasks_completed=42,
            last_task_duration_seconds=0.85,
            recent_task_durations_seconds=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            gpu_utilization_pct=72.5,
        )
        # w2 — minimal worker (only the registration constructor's
        # default last_heartbeat is set; R4.4 fields stay None / []).
        reg.register("w2", capabilities={"gpu": None})
        return reg

    def _samples_by_metric(self, collector):
        """Run collect() and bucket the samples by their metric name."""
        out: dict[str, list] = {}
        for fam in collector.collect():
            out.setdefault(fam.name, []).extend(fam.samples)
        return out

    def test_collect_emits_expected_metric_names(self):
        """All five families are present in a single scrape."""
        from api.workers.metrics import WorkerRegistryCollector

        reg = self._build_registry_with_two_workers()
        collector = WorkerRegistryCollector(reg)
        samples = self._samples_by_metric(collector)

        assert "juniper_cascor_worker_heartbeat_age_seconds" in samples
        assert "juniper_cascor_worker_last_task_duration_seconds" in samples
        assert "juniper_cascor_worker_gpu_utilization_pct" in samples
        assert "juniper_cascor_worker_recent_task_duration_seconds_p50" in samples
        assert "juniper_cascor_worker_recent_task_duration_seconds_p95" in samples

    def test_heartbeat_age_emitted_for_every_worker(self):
        """Every worker has a populated last_heartbeat — both must appear."""
        from api.workers.metrics import WorkerRegistryCollector

        reg = self._build_registry_with_two_workers()
        # Inject a fixed time source 5 s after registration so the age
        # is non-zero and deterministic.
        registration_now = max(w.last_heartbeat for w in reg.get_all_workers())
        collector = WorkerRegistryCollector(reg, time_source=lambda: registration_now + 5.0)

        samples = self._samples_by_metric(collector)
        ages = {s.labels["worker_id"]: s.value for s in samples["juniper_cascor_worker_heartbeat_age_seconds"]}
        assert set(ages.keys()) == {"w1", "w2"}
        # Both should be approximately 5 s old (clock-injected).
        for wid, age in ages.items():
            assert age == pytest.approx(5.0, abs=1.0), f"heartbeat age for {wid} unexpected: {age}"

    def test_unset_fields_are_omitted_not_zero_emitted(self):
        """w2 has no R4.4 fields populated — its series must NOT appear."""
        from api.workers.metrics import WorkerRegistryCollector

        reg = self._build_registry_with_two_workers()
        collector = WorkerRegistryCollector(reg)
        samples = self._samples_by_metric(collector)

        for metric_name in (
            "juniper_cascor_worker_last_task_duration_seconds",
            "juniper_cascor_worker_gpu_utilization_pct",
            "juniper_cascor_worker_recent_task_duration_seconds_p50",
            "juniper_cascor_worker_recent_task_duration_seconds_p95",
        ):
            worker_ids = {s.labels["worker_id"] for s in samples[metric_name]}
            assert "w2" not in worker_ids, f"{metric_name}: w2 unset field was zero-emitted (forbidden)"
            assert "w1" in worker_ids, f"{metric_name}: w1 (fully instrumented) is missing"

    def test_w1_last_task_and_gpu_values_match(self):
        """Fully-instrumented worker emits the actual reported values."""
        from api.workers.metrics import WorkerRegistryCollector

        reg = self._build_registry_with_two_workers()
        collector = WorkerRegistryCollector(reg)
        samples = self._samples_by_metric(collector)

        last = {s.labels["worker_id"]: s.value for s in samples["juniper_cascor_worker_last_task_duration_seconds"]}
        gpu = {s.labels["worker_id"]: s.value for s in samples["juniper_cascor_worker_gpu_utilization_pct"]}

        assert last["w1"] == pytest.approx(0.85)
        assert gpu["w1"] == pytest.approx(72.5)

    def test_recent_p50_p95_within_window(self):
        """Percentiles fall inside the [min, max] of the recent durations window."""
        from api.workers.metrics import WorkerRegistryCollector

        reg = self._build_registry_with_two_workers()
        collector = WorkerRegistryCollector(reg)
        samples = self._samples_by_metric(collector)

        p50 = {s.labels["worker_id"]: s.value for s in samples["juniper_cascor_worker_recent_task_duration_seconds_p50"]}
        p95 = {s.labels["worker_id"]: s.value for s in samples["juniper_cascor_worker_recent_task_duration_seconds_p95"]}

        # Window for w1 is [0.5..1.4] in 0.1 increments. Median is
        # ~0.95 (statistics.quantiles inclusive of 10 samples). p95 is
        # near the upper end (~1.36).
        assert 0.5 <= p50["w1"] <= 1.4
        assert 0.5 <= p95["w1"] <= 1.4
        assert p95["w1"] >= p50["w1"]

    def test_single_sample_window_omits_percentiles(self):
        """Window with <2 samples skips p50/p95 emission for that worker."""
        from api.workers.metrics import WorkerRegistryCollector
        from api.workers.registry import WorkerRegistry

        reg = WorkerRegistry()
        reg.register("solo", capabilities={})
        reg.heartbeat("solo", recent_task_durations_seconds=[0.5])  # only one sample

        collector = WorkerRegistryCollector(reg)
        samples = self._samples_by_metric(collector)

        worker_ids = {s.labels["worker_id"] for s in samples["juniper_cascor_worker_recent_task_duration_seconds_p50"]}
        assert "solo" not in worker_ids, "single-sample window must NOT emit p50 (degenerate quantile)"

    def test_collector_is_robust_to_registry_failure(self):
        """A broken registry surfaces as an empty scrape, not an exception."""
        from api.workers.metrics import WorkerRegistryCollector

        class _BrokenRegistry:
            def get_all_workers(self):
                raise RuntimeError("simulated registry corruption")

        collector = WorkerRegistryCollector(_BrokenRegistry())
        samples = self._samples_by_metric(collector)
        # All families exist; all are empty — no crash, no zero-emit.
        for fam_samples in samples.values():
            assert fam_samples == [], "broken registry leaked samples; expected empty scrape"


@pytest.mark.unit
class TestLifecycleManagerTerminalCounterIntegration:
    """End-to-end: ``_run_training`` bumps the terminal counter at each terminal transition.

    WS-6 PR-B3.3: the counter increments live in ``_run_training`` around
    ``self.model.fit`` (not in a network monkey-patch). These tests drive synthetic
    terminal transitions through a ``model.fit`` stub rather than running real cascor
    training (which would require a torch-capable test environment — see the prior R5.1b
    PR's note about the JuniperCascor conda env's torch ImportError under Py3.14).
    """

    def setup_method(self):
        _reset_training_metrics()

    def teardown_method(self):
        _reset_training_metrics()

    def _counter_value(self, status: str) -> float:
        m = _ensure_training_metrics()["sessions_completed_total"]
        return m.labels(status=status)._value.get()

    def _build_manager_with_fake_model(self, fit_outcome: str):
        """Return a manager whose ``model.fit`` synthesizes a terminal outcome.

        ``fit_outcome`` is one of:
          - ``"success"`` — fit returns normally and stop_event stays clear.
          - ``"cancelled"`` — fit returns normally but stop_event is set before
            ``_run_training`` checks (simulating a stop_training() arriving mid-flight).
          - ``"failure"`` — fit raises ``RuntimeError``.

        WS-6 PR-B3.3: ``_run_training`` drives ``self.model.fit`` and owns the terminal
        counter increments, so the synthetic outcome is produced by stubbing ``model.fit``.
        """
        from api.lifecycle.manager import TrainingLifecycleManager

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        def _fake_fit(x, y, *, X_val=None, y_val=None, on_event=None, **kwargs):
            if fit_outcome == "failure":
                raise RuntimeError("synthetic training failure")
            if fit_outcome == "cancelled":
                # Simulate stop_training() landing mid-fit.
                mgr._stop_event.set()
            return None

        mgr.model.fit = _fake_fit
        return mgr

    def test_success_bumps_success_counter(self):
        import torch

        mgr = self._build_manager_with_fake_model("success")
        before = self._counter_value(TRAINING_SESSION_STATUS_SUCCESS)
        mgr._run_training(torch.zeros(2, 2), torch.zeros(2, 2), None, None)
        after = self._counter_value(TRAINING_SESSION_STATUS_SUCCESS)
        assert after - before == pytest.approx(1.0)

    def test_failure_bumps_failure_counter(self):
        import torch

        mgr = self._build_manager_with_fake_model("failure")
        before = self._counter_value(TRAINING_SESSION_STATUS_FAILURE)
        with pytest.raises(RuntimeError, match="synthetic training failure"):
            mgr._run_training(torch.zeros(2, 2), torch.zeros(2, 2), None, None)
        after = self._counter_value(TRAINING_SESSION_STATUS_FAILURE)
        assert after - before == pytest.approx(1.0)

    def test_cancelled_bumps_cancelled_counter(self):
        import torch

        mgr = self._build_manager_with_fake_model("cancelled")
        before = self._counter_value(TRAINING_SESSION_STATUS_CANCELLED)
        mgr._run_training(torch.zeros(2, 2), torch.zeros(2, 2), None, None)
        after = self._counter_value(TRAINING_SESSION_STATUS_CANCELLED)
        assert after - before == pytest.approx(1.0)


@pytest.mark.unit
class TestLifecycleStepDurationCallback:
    """``_handle_event`` emits a step-duration histogram sample on the second epoch_end."""

    def setup_method(self):
        _reset_training_metrics()

    def teardown_method(self):
        _reset_training_metrics()

    def test_two_back_to_back_epoch_events_observe_one_sample(self):
        """First epoch_end seeds the per-run timer; the second emits the delta as one sample."""
        from juniper_model_core.events import TrainingEvent

        from api.lifecycle.manager import TrainingLifecycleManager

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr._step_timer_prev = None

        # WS-6 PR-B3.3: the step-duration histogram lives in _handle_event's epoch_end
        # branch (was _output_training_callback). Drive two epoch_end events directly.
        mgr._handle_event(TrainingEvent("epoch_end", {"epoch": 1, "metrics": {"loss": 0.5}}, 0))
        time.sleep(0.06)  # land in the (0.05, 0.1] bucket
        mgr._handle_event(TrainingEvent("epoch_end", {"epoch": 2, "metrics": {"loss": 0.4}}, 1))

        hist = _ensure_training_metrics()["step_duration_seconds"]
        # OBS-WIRE-01 (A.6): the histogram is unlabelled — observations
        # land directly on ``hist._sum`` / ``hist._buckets`` rather
        # than the per-label child.
        observed_sum = hist._sum.get()
        assert observed_sum > 0.0, "no train-step duration sample emitted"
        # ``_buckets`` is non-cumulative in modern prometheus_client.
        # Across all buckets the observation count must total exactly 1.
        total_count = sum(bucket.get() for bucket in hist._buckets)
        assert total_count == 1, f"expected exactly 1 sample emitted across all buckets, got {total_count}"


@pytest.mark.unit
class TestPendingTasksGaugeAudit_4_2:
    """Audit-doc §4.2 — ``juniper_cascor_pending_tasks`` bridge gauge.

    The pre-existing :class:`api.workers.coordinator.WorkerCoordinator`
    tracks in-flight tasks in an internal ``_pending_tasks`` dict that
    was JSON-only (no Prometheus surface) until this audit follow-up.
    The corresponding ``CascorPendingTasksSaturated`` alert
    (juniper-deploy/prometheus/alert_rules.yml) carried an
    ``absent_over_time(...) == 0`` inertness guard while the bridge
    was missing.

    These tests exercise the wire-up:
      * ``pending_tasks_count()`` accessor on the coordinator.
      * The optional ``coordinator=`` kwarg on
        :class:`api.workers.metrics.WorkerRegistryCollector`.
      * The ``juniper_cascor_pending_tasks`` gauge emission when wired.
      * Backward-compat: missing-coordinator path stays silent
        (preserves the alert's inertness guard for test fixtures).
    """

    def _samples_by_metric(self, collector):
        out: dict[str, list] = {}
        for fam in collector.collect():
            out.setdefault(fam.name, []).extend(fam.samples)
        return out

    def _build_coordinator_with_n_pending(self, n: int):
        """Inject ``n`` pending tasks into a freshly-constructed coordinator.

        The coordinator's full ``submit_tasks`` flow requires a round-id
        and websocket dispatch; for a unit test we inject directly into
        ``_pending_tasks`` under the lock to avoid the integration
        surface. ``pending_tasks_count()`` only reads the dict size, but
        ``cancel_round()`` walks the entries to release registry busy-state,
        so the stand-in must expose the real task surface rather than a bare
        ``object()``.
        """
        from api.workers.coordinator import WorkerCoordinator
        from api.workers.registry import WorkerRegistry

        class _PendingTaskStub:
            """Minimal stand-in for ``coordinator.PendingTask``.

            Models the attributes the coordinator reads off a pending entry,
            with the same defaults as the real dataclass — a freshly submitted
            task is unassigned and not yet complete.
            """

            def __init__(self) -> None:
                self.assigned_worker_id = None
                self.completed = False

        reg = WorkerRegistry(heartbeat_timeout=30.0)
        coord = WorkerCoordinator(registry=reg)
        with coord._lock:  # noqa: SLF001 — test poke under the same lock
            for i in range(n):
                coord._pending_tasks[f"task-{i}"] = _PendingTaskStub()  # type: ignore[assignment]
        return coord, reg

    def test_pending_tasks_count_returns_dict_size(self):
        """Direct accessor: returns ``len(_pending_tasks)`` under the lock."""
        coord, _reg = self._build_coordinator_with_n_pending(7)
        assert coord.pending_tasks_count() == 7

    def test_pending_tasks_count_zero_when_empty(self):
        """No tasks pending → 0 (not None / not raises)."""
        coord, _reg = self._build_coordinator_with_n_pending(0)
        assert coord.pending_tasks_count() == 0

    def test_collector_emits_pending_tasks_when_coordinator_wired(self):
        """When ``coordinator=`` is set, the gauge appears in collect() output."""
        from api.workers.metrics import WorkerRegistryCollector

        coord, reg = self._build_coordinator_with_n_pending(3)
        collector = WorkerRegistryCollector(reg, coordinator=coord)
        samples = self._samples_by_metric(collector)

        assert "juniper_cascor_pending_tasks" in samples, "pending_tasks gauge must appear when coordinator is wired"
        gauge_samples = samples["juniper_cascor_pending_tasks"]
        # Single unlabelled sample.
        assert len(gauge_samples) == 1
        assert gauge_samples[0].labels == {}
        assert gauge_samples[0].value == 3.0

    def test_collector_omits_pending_tasks_when_no_coordinator(self):
        """Without ``coordinator=``, the gauge is silently skipped (back-compat)."""
        from api.workers.metrics import WorkerRegistryCollector
        from api.workers.registry import WorkerRegistry

        reg = WorkerRegistry(heartbeat_timeout=30.0)
        collector = WorkerRegistryCollector(reg)  # NO coordinator
        samples = self._samples_by_metric(collector)

        assert "juniper_cascor_pending_tasks" not in samples, "pending_tasks gauge must be omitted when no coordinator wired — " "preserves the alert rule's absent_over_time(...) == 0 inertness " "guard for test fixtures + lightweight harnesses"

    def test_collector_skips_gauge_on_coordinator_exception(self):
        """If the coordinator raises during count read, scrape continues."""
        from api.workers.metrics import WorkerRegistryCollector
        from api.workers.registry import WorkerRegistry

        class _BrokenCoordinator:
            def pending_tasks_count(self) -> int:
                raise RuntimeError("simulated coordinator failure")

        reg = WorkerRegistry(heartbeat_timeout=30.0)
        collector = WorkerRegistryCollector(reg, coordinator=_BrokenCoordinator())
        samples = self._samples_by_metric(collector)

        # Other gauges still emit; the pending_tasks gauge is skipped
        # (logged but not raised — matches the per-snapshot try/except
        # pattern in the rest of collect()).
        assert "juniper_cascor_pending_tasks" not in samples

    def test_pending_tasks_count_drops_to_zero_after_cancel_round(self):
        """Wire-up regression: cancel_round() clears _pending_tasks → gauge drops."""
        coord, _reg = self._build_coordinator_with_n_pending(5)
        assert coord.pending_tasks_count() == 5
        coord.cancel_round()
        assert coord.pending_tasks_count() == 0
