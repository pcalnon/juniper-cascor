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
    """End-to-end: monitored_fit() bumps the counter at each terminal transition.

    These tests drive synthetic terminal transitions through the
    monkey-patched ``network.fit`` wrapper rather than running real
    cascor training (which would require a torch-capable test
    environment — see the prior R5.1b PR's note about the JuniperCascor
    conda env's torch ImportError under Py3.14 free-threading).
    """

    def setup_method(self):
        _reset_training_metrics()

    def teardown_method(self):
        _reset_training_metrics()

    def _counter_value(self, status: str) -> float:
        m = _ensure_training_metrics()["sessions_completed_total"]
        return m.labels(status=status)._value.get()

    def _build_manager_with_fake_network(self, fit_outcome: str):
        """Return a manager whose ``network.fit`` synthesizes a terminal outcome.

        ``fit_outcome`` is one of:
          - ``"success"`` — fit returns normally and stop_event stays clear.
          - ``"cancelled"`` — fit returns normally but stop_event is set
            before the wrapper checks (simulating a stop_training()
            arriving mid-flight).
          - ``"failure"`` — fit raises ``RuntimeError``.
        """
        from api.lifecycle.manager import TrainingLifecycleManager

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        # _install_monitoring_hooks ran inside create_network; the
        # monkey-patched network.fit is now ``monitored_fit``. Replace
        # the captured ``original_fit`` so it produces our synthetic
        # outcome WITHOUT running the real cascor training path.
        original_fit_marker = mgr._original_methods["fit"]

        def _fake_fit(x, y, x_val=None, y_val=None, **kwargs):
            if fit_outcome == "failure":
                raise RuntimeError("synthetic training failure")
            if fit_outcome == "cancelled":
                # Simulate stop_training() landing mid-fit.
                mgr._stop_requested.set()
            return None

        # Splice the fake into the slot where monitored_fit() captured
        # the original. monitored_fit holds ``original_fit`` via
        # closure, so we have to rebuild the closure: easiest is to
        # reinstall hooks against a network whose .fit IS the fake.
        mgr._restore_original_methods()
        mgr._monitoring_active = False

        # Replace network.fit with the fake, then re-install hooks so
        # monitored_fit wraps the fake as the new "original_fit".
        mgr.network.fit = _fake_fit
        mgr._install_monitoring_hooks()

        # Mark unused for type checkers — the marker is incidental.
        del original_fit_marker
        return mgr

    def test_success_bumps_success_counter(self):
        import torch

        mgr = self._build_manager_with_fake_network("success")
        before = self._counter_value(TRAINING_SESSION_STATUS_SUCCESS)
        x = torch.zeros(2, 2)
        y = torch.zeros(2, 2)
        mgr.network.fit(x, y)
        after = self._counter_value(TRAINING_SESSION_STATUS_SUCCESS)
        assert after - before == pytest.approx(1.0)

    def test_failure_bumps_failure_counter(self):
        import torch

        mgr = self._build_manager_with_fake_network("failure")
        before = self._counter_value(TRAINING_SESSION_STATUS_FAILURE)
        x = torch.zeros(2, 2)
        y = torch.zeros(2, 2)
        with pytest.raises(RuntimeError, match="synthetic training failure"):
            mgr.network.fit(x, y)
        after = self._counter_value(TRAINING_SESSION_STATUS_FAILURE)
        assert after - before == pytest.approx(1.0)

    def test_cancelled_bumps_cancelled_counter(self):
        import torch

        mgr = self._build_manager_with_fake_network("cancelled")
        before = self._counter_value(TRAINING_SESSION_STATUS_CANCELLED)
        x = torch.zeros(2, 2)
        y = torch.zeros(2, 2)
        mgr.network.fit(x, y)
        after = self._counter_value(TRAINING_SESSION_STATUS_CANCELLED)
        assert after - before == pytest.approx(1.0)


@pytest.mark.unit
class TestLifecycleStepDurationCallback:
    """The output-phase epoch callback emits a histogram sample on the second invocation."""

    def setup_method(self):
        _reset_training_metrics()

    def teardown_method(self):
        _reset_training_metrics()

    def test_two_back_to_back_callbacks_observe_one_sample(self):
        """First callback seeds the timer; second emits the delta as a sample."""
        from api.lifecycle.manager import TrainingLifecycleManager

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        # Pull the injected callback out of the network attribute set
        # by monitored_fit. monitored_fit hasn't been called yet — but
        # the callback only flows through fit's setup. Instead, drive
        # the fit() wrapper with a fake that runs the callback twice
        # then returns successfully.
        def _fake_fit(x, y, x_val=None, y_val=None, **kwargs):
            cb = mgr.network._output_epoch_callback
            cb(epoch=1, epochs=2, loss=0.5)
            time.sleep(0.06)  # land in the (0.05, 0.1] bucket
            cb(epoch=2, epochs=2, loss=0.4)
            return None

        # Re-install hooks around our fake fit (matches the
        # terminal-counter integration tests above).
        mgr._restore_original_methods()
        mgr._monitoring_active = False
        mgr.network.fit = _fake_fit
        mgr._install_monitoring_hooks()

        import torch

        mgr.network.fit(torch.zeros(2, 2), torch.zeros(2, 2))

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
