"""OBS-WIRE-01 regression tests.

Closes 6 of 27 audit findings from
``juniper-ml notes/code-review/OBSERVABILITY_AUDIT_AND_OUTSTANDING_ISSUES_2026-05-03.md``:

- A.1: ``juniper_cascor_training_sessions_active`` gauge inc/dec
- A.2: training_loss / epochs_total / candidate_correlation /
  accuracy_ratio / hidden_units_total emission
- A.3: ``cascor_ws_broadcast_send_duration_seconds`` and
  ``cascor_ws_command_handler_seconds`` observation
- A.5: removal of dead ``juniper_cascor_inference_*`` family
- A.6: drop ``phase`` label from
  ``juniper_cascor_training_step_duration_seconds``
- E.1: lazy-init race in ``_ensure_training_metrics`` /
  ``_ensure_ws_metrics``

Each suite owns its own teardown so tests are order-independent.
"""

import logging
import threading

import pytest

import api.observability as obs


def _reset_training_metrics() -> None:
    """Force lazy re-init so each test sees a fresh metric set."""
    from prometheus_client import REGISTRY

    if obs._training_metrics is not None:
        for metric in list(obs._training_metrics.values()):
            try:
                REGISTRY.unregister(metric)
            except Exception as exc:
                logging.getLogger(__name__).debug("Best-effort training-metric unregister failed for %r: %s", metric, exc)
        obs._training_metrics = None


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


@pytest.mark.unit
class TestActiveSessionsGaugeInLifecycleManager:
    """A.1 — inc at session start, dec on every terminal path (incl. exception)."""

    def setup_method(self):
        _reset_training_metrics()

    def teardown_method(self):
        _reset_training_metrics()

    def _gauge_value(self) -> float:
        return obs._ensure_training_metrics()["sessions_active"]._value.get()

    def _build_manager_with_fake_model(self, fit_outcome: str):
        """A manager whose ``model.fit`` is a synthetic stub (WS-6 PR-B3.3: the gauge
        inc/dec pair lives in ``_run_training`` around ``self.model.fit``, not in a
        network monkey-patch). Replica of the helper in ``test_metrics_r5_4_pre.py``."""
        from api.lifecycle.manager import TrainingLifecycleManager

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)

        def _fake_fit(x, y, *, X_val=None, y_val=None, on_event=None, **kwargs):
            if fit_outcome == "failure":
                raise RuntimeError("synthetic training failure")
            if fit_outcome == "cancelled":
                mgr._stop_event.set()
            return None

        mgr.model.fit = _fake_fit
        return mgr

    def test_success_path_increments_then_decrements(self):
        import torch

        mgr = self._build_manager_with_fake_model("success")
        before = self._gauge_value()
        mgr._run_training(torch.zeros(2, 2), torch.zeros(2, 2), None, None)
        after = self._gauge_value()
        # Inc(+1) then Dec(-1) → net zero.
        assert after == pytest.approx(before)

    def test_cancelled_path_decrements(self):
        import torch

        mgr = self._build_manager_with_fake_model("cancelled")
        before = self._gauge_value()
        mgr._run_training(torch.zeros(2, 2), torch.zeros(2, 2), None, None)
        after = self._gauge_value()
        assert after == pytest.approx(before)

    def test_exception_path_still_decrements(self):
        """The try/finally invariant — failure must not leak the gauge."""
        import torch

        mgr = self._build_manager_with_fake_model("failure")
        before = self._gauge_value()
        with pytest.raises(RuntimeError, match="synthetic training failure"):
            mgr._run_training(torch.zeros(2, 2), torch.zeros(2, 2), None, None)
        after = self._gauge_value()
        # Most important assertion of this test: gauge balanced even on
        # the exception path.
        assert after == pytest.approx(before)


@pytest.mark.unit
class TestTrainingMetricsExtractAndRecord:
    """A.2 — _extract_and_record_metrics emits epochs / loss / accuracy / hidden_units."""

    def setup_method(self):
        _reset_training_metrics()

    def teardown_method(self):
        _reset_training_metrics()

    def _build_manager_with_synth_history(self, train_loss, train_acc, val_loss, val_acc, n_hidden):
        """Build a manager whose network.history mimics a few completed epochs.

        The manager is constructed via the normal ``create_network``
        path (which installs monitoring hooks), then we synthesize the
        ``history`` dict and ``hidden_units`` list directly so
        ``_extract_and_record_metrics`` has data to drain.
        """
        from api.lifecycle.manager import TrainingLifecycleManager

        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.network.history = {
            "train_loss": list(train_loss),
            "train_accuracy": list(train_acc),
            "value_loss": list(val_loss),
            "value_accuracy": list(val_acc),
            "hidden_units_added": [],
        }
        mgr.network.hidden_units = [object()] * n_hidden
        mgr._last_emitted_history_len = 0
        return mgr

    def test_drain_emits_loss_accuracy_hidden_units_and_counter(self):
        mgr = self._build_manager_with_synth_history(
            train_loss=[1.0, 0.5, 0.25],
            train_acc=[0.6, 0.8, 0.95],
            val_loss=[1.1, 0.6],  # only 2 validation rows
            val_acc=[0.5, 0.75],
            n_hidden=3,
        )
        epochs_total = obs._ensure_training_metrics()["epochs_total"]
        loss_gauge = obs._ensure_training_metrics()["loss"]
        accuracy_gauge = obs._ensure_training_metrics()["accuracy_ratio"]
        hidden_gauge = obs._ensure_training_metrics()["hidden_units_total"]

        epochs_before = epochs_total.labels(phase="output")._value.get()

        mgr._extract_and_record_metrics()

        # Counter advanced by 3 — one per history row drained.
        epochs_after = epochs_total.labels(phase="output")._value.get()
        assert epochs_after - epochs_before == pytest.approx(3.0)

        # Loss gauge holds the LAST value (gauge contract).
        assert loss_gauge.labels(phase="output", loss_type="train")._value.get() == pytest.approx(0.25)
        # Validation loss gauge — last validation row was 0.6.
        assert loss_gauge.labels(phase="output", loss_type="validation")._value.get() == pytest.approx(0.6)

        # Accuracy gauge — last train_accuracy is 0.95.
        assert accuracy_gauge.labels(phase="output")._value.get() == pytest.approx(0.95)
        # Validation accuracy at phase="validation".
        assert accuracy_gauge.labels(phase="validation")._value.get() == pytest.approx(0.75)

        # Hidden units gauge reflects the network's current count.
        assert hidden_gauge._value.get() == pytest.approx(3.0)

    def test_drain_is_idempotent_without_new_data(self):
        """Re-call without new history rows must not double-bump the counter."""
        mgr = self._build_manager_with_synth_history(
            train_loss=[1.0],
            train_acc=[0.5],
            val_loss=[],
            val_acc=[],
            n_hidden=1,
        )
        epochs_total = obs._ensure_training_metrics()["epochs_total"]
        mgr._extract_and_record_metrics()
        first = epochs_total.labels(phase="output")._value.get()
        mgr._extract_and_record_metrics()
        second = epochs_total.labels(phase="output")._value.get()
        assert second == pytest.approx(first), "second drain emitted phantom epochs"


@pytest.mark.unit
class TestCandidateCorrelationGaugeWiring:
    """A.2 — candidate_correlation gauge reachable via the helper.

    The actual emission site lives inside ``cascade_correlation.py``'s
    ``grow_network`` loop. Driving real cascade-correlation training
    here would require a working torch environment (see project memory
    re: JuniperCascor torch ImportError under Py3.14 free-threading).
    Instead, exercise the import-and-emit path the production code uses
    so a refactor that breaks the helper is caught at unit-test time.
    """

    def setup_method(self):
        _reset_training_metrics()

    def teardown_method(self):
        _reset_training_metrics()

    def test_helper_sets_gauge_value(self):
        """The set_candidate_correlation helper writes to the right gauge."""
        from api.observability import set_candidate_correlation

        gauge = obs._ensure_training_metrics()["candidate_correlation"]
        set_candidate_correlation(0.42)
        assert gauge._value.get() == pytest.approx(0.42)
        # Gauge contract: a second set() OVERWRITES, doesn't accumulate.
        set_candidate_correlation(0.99)
        assert gauge._value.get() == pytest.approx(0.99)


@pytest.mark.unit
class TestStepDurationPhaseLabelDropped:
    """A.6 — phase label gone from ``juniper_cascor_training_step_duration_seconds``."""

    def setup_method(self):
        _reset_training_metrics()

    def teardown_method(self):
        _reset_training_metrics()

    def test_metric_has_no_label_names(self):
        hist = obs._ensure_training_metrics()["step_duration_seconds"]
        assert hist._labelnames == ()

    def test_helper_signature_takes_only_duration(self):
        """A.6: ``observe_training_step_duration`` no longer accepts ``phase``."""
        from api.observability import observe_training_step_duration

        # Positional-only call with one arg should succeed.
        observe_training_step_duration(0.05)
        # And calling with the legacy two-arg shape should fail because
        # the helper was simplified.
        with pytest.raises(TypeError):
            observe_training_step_duration("output", 0.05)  # type: ignore[call-arg]


@pytest.mark.unit
class TestInferenceMetricsRemoved:
    """A.5 — the dead ``juniper_cascor_inference_*`` family is gone."""

    def setup_method(self):
        _reset_training_metrics()

    def teardown_method(self):
        _reset_training_metrics()

    def test_inference_metrics_absent_from_dict(self):
        m = obs._ensure_training_metrics()
        assert "inference_requests_total" not in m
        assert "inference_duration_seconds" not in m

    def test_record_inference_helper_removed(self):
        assert not hasattr(obs, "record_inference"), "record_inference should have been removed (A.5)"


@pytest.mark.unit
class TestBroadcastSendDurationWiring:
    """A.3 — ``cascor_ws_broadcast_send_duration_seconds`` observed per send."""

    def setup_method(self):
        _reset_ws_metrics()

    def teardown_method(self):
        _reset_ws_metrics()

    def test_send_json_observes_histogram(self):
        """A successful _send_json call increments the histogram by exactly one observation."""
        import asyncio

        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager()

        # Fake WebSocket whose ``send_json`` is a no-op coroutine.
        class _FakeWS:
            async def send_json(self, msg):
                return None

        ws = _FakeWS()
        hist = obs._ensure_ws_metrics()["broadcast_send_duration_seconds"]

        before_count = sum(b.get() for b in hist.labels(type="state")._buckets)
        result = asyncio.run(mgr._send_json(ws, {"type": "state", "payload": "hi"}))
        assert result is True
        after_count = sum(b.get() for b in hist.labels(type="state")._buckets)
        assert after_count - before_count == 1

    def test_send_json_observes_histogram_on_failure(self):
        """Even when send fails, we still record the latency sample (slow/failed sends matter for SLI 4.3)."""
        import asyncio

        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager()

        class _BrokenWS:
            async def send_json(self, msg):
                raise RuntimeError("synthetic send failure")

        ws = _BrokenWS()
        hist = obs._ensure_ws_metrics()["broadcast_send_duration_seconds"]
        before_count = sum(b.get() for b in hist.labels(type="state")._buckets)

        result = asyncio.run(mgr._send_json(ws, {"type": "state", "payload": "hi"}))
        assert result is False
        after_count = sum(b.get() for b in hist.labels(type="state")._buckets)
        assert after_count - before_count == 1, "failed sends must still produce a latency sample"

    def test_unknown_message_type_falls_back(self):
        """Messages without a ``type`` field land in the ``unknown`` bucket family (consistent with _account_send)."""
        import asyncio

        from api.websocket.manager import WebSocketManager

        mgr = WebSocketManager()

        class _FakeWS:
            async def send_json(self, msg):
                return None

        ws = _FakeWS()
        hist = obs._ensure_ws_metrics()["broadcast_send_duration_seconds"]
        before_count = sum(b.get() for b in hist.labels(type="unknown")._buckets)
        asyncio.run(mgr._send_json(ws, {"payload": "no-type-field"}))
        after_count = sum(b.get() for b in hist.labels(type="unknown")._buckets)
        assert after_count - before_count == 1


@pytest.mark.unit
class TestCommandHandlerHistogramWiring:
    """A.3 — ``cascor_ws_command_handler_seconds`` observed per dispatch."""

    def setup_method(self):
        _reset_ws_metrics()

    def teardown_method(self):
        _reset_ws_metrics()

    def test_helper_observes_histogram(self):
        """``ws_observe_command_handler`` writes to the right histogram with the right label."""
        from api.observability import ws_observe_command_handler

        hist = obs._ensure_ws_metrics()["command_handler_seconds"]
        before = sum(b.get() for b in hist.labels(command="start")._buckets)
        ws_observe_command_handler("start", 0.012)
        after = sum(b.get() for b in hist.labels(command="start")._buckets)
        assert after - before == 1
        # And the sum got the actual duration.
        assert hist.labels(command="start")._sum.get() == pytest.approx(0.012)


@pytest.mark.unit
class TestLazyInitRaceFix:
    """E.1 — concurrent first-callers don't orphan a collector.

    Pre-OBS-WIRE-01: two threads could both enter the
    ``if _xxx_metrics is None:`` branch; the second hit
    ``Duplicated timeseries``, the recovery path unregistered the
    live collector, and the first thread's reference was orphaned.

    Post-OBS-WIRE-01: ``threading.Lock`` makes the check-and-init
    atomic so exactly one thread registers and every caller sees the
    same dict.
    """

    def setup_method(self):
        _reset_training_metrics()
        _reset_ws_metrics()

    def teardown_method(self):
        _reset_training_metrics()
        _reset_ws_metrics()

    def _race_ensure(self, ensure_fn, n_threads: int = 16):
        """Drive ``n_threads`` concurrent first-callers and return their results."""
        results = [None] * n_threads
        barrier = threading.Barrier(n_threads)

        def worker(idx: int) -> None:
            barrier.wait()  # synchronize the entry instant
            results[idx] = ensure_fn()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def test_concurrent_callers_see_same_training_dict(self):
        """All threads receive the same dict instance — no orphaned collectors."""
        results = self._race_ensure(obs._ensure_training_metrics)
        first = results[0]
        assert first is not None
        for other in results[1:]:
            assert other is first, "concurrent first-callers received different dict instances (race not fixed)"

    def test_concurrent_callers_see_same_ws_dict(self):
        results = self._race_ensure(obs._ensure_ws_metrics)
        first = results[0]
        assert first is not None
        for other in results[1:]:
            assert other is first, "concurrent first-callers received different ws-dict instances (race not fixed)"

    def test_collector_registered_in_global_registry_after_race(self):
        """The collector is reachable in REGISTRY (no half-registered orphans)."""
        from prometheus_client import REGISTRY

        self._race_ensure(obs._ensure_training_metrics)
        # The active-sessions Gauge name must be findable in the registry.
        names: set[str] = set()
        for _collector, collector_names in REGISTRY._collector_to_names.items():
            names.update(collector_names)
        assert "juniper_cascor_training_sessions_active" in names

    def test_lock_present(self):
        """E.1 contract: the locks are module-level threading.Lock instances."""
        assert isinstance(obs._training_metrics_lock, type(threading.Lock()))
        assert isinstance(obs._ws_metrics_lock, type(threading.Lock()))
