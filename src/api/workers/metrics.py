"""Worker -> Prometheus bridge collector (METRICS-MON R5.4-pre).

The R4.4 PR added three training-loop instrumentation fields to
``WorkerRegistration`` (``last_task_duration_seconds``,
``recent_task_durations_seconds``, ``gpu_utilization_pct``) plus
``last_heartbeat`` from the original R1.3 work, but never exposed any
of them on the cascor Prometheus surface — they were JSON-only via
``/v1/workers``. The R5.1 SLO catalog (juniper-deploy#48) and the R5.3
operator dashboard (juniper-deploy#46) both flagged this as the
missing surface for two internal-supporting SLIs ("worker heartbeat
freshness" and "worker recent task duration p95").

This module defines :class:`WorkerRegistryCollector`, a
``prometheus_client``-compatible custom collector that holds a
reference to the in-process :class:`WorkerRegistry` and snapshots it
on every ``collect()`` call. The collector emits, **per worker**:

- ``juniper_cascor_worker_last_task_duration_seconds{worker_id}``
  (Gauge) — wall-clock duration of the most recently completed task.
- ``juniper_cascor_worker_gpu_utilization_pct{worker_id}`` (Gauge) —
  best-effort 0-100 reading.
- ``juniper_cascor_worker_recent_task_duration_seconds_p50{worker_id}``
  (Gauge) — p50 over the registration's sliding window of recent
  durations.
- ``juniper_cascor_worker_recent_task_duration_seconds_p95{worker_id}``
  (Gauge) — p95 over the same window.
- ``juniper_cascor_worker_heartbeat_age_seconds{worker_id}`` (Gauge)
  — ``time.time() - last_heartbeat``.

**Gauge-omission strategy.** Workers that haven't reported a given
field (``None`` on the registration) are NOT emitted with a zero
value — Prometheus interprets a missing series as "no data" which is
the correct semantic for "no observation yet". Specifically:

- ``last_task_duration_seconds`` / ``gpu_utilization_pct``: skipped
  when ``None``.
- ``recent_task_duration_seconds_p50/p95``: skipped when the window
  has fewer than 2 samples (``statistics.quantiles`` requires >=2).
- ``heartbeat_age_seconds``: always emitted (every registered worker
  has a ``last_heartbeat`` populated by the registration constructor).

Per R1.4 single-registration discipline the collector instance must
be registered with the cascor ``prometheus_client.REGISTRY`` exactly
once at startup; the lifespan handler in :mod:`api.app` performs that
registration.

References:

- ``notes/code-review/SLO_CATALOG_2026-05-03.md`` (juniper-deploy#48,
  §3.3 / §3.4 / supporting-SLI catalog).
- ``notes/observability/HISTOGRAM_BUCKETS_RATIONALE_2026-05-02.md`` —
  the rationale for buckets, not directly relevant here (these are
  gauges) but referenced because future histogramization of
  ``recent_task_durations_seconds`` is on the R5.4 follow-up queue.
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:
    from prometheus_client.core import Metric

    from api.workers.registry import WorkerRegistry

logger = logging.getLogger("juniper_cascor.api.workers.metrics")


# METRICS-MON R5.4-pre: emitted metric names. Kept module-level so the
# regression test can assert exact wire shape without importing
# private symbols.
_METRIC_LAST_TASK_DURATION: str = "juniper_cascor_worker_last_task_duration_seconds"
_METRIC_GPU_UTILIZATION_PCT: str = "juniper_cascor_worker_gpu_utilization_pct"
_METRIC_RECENT_DURATION_P50: str = "juniper_cascor_worker_recent_task_duration_seconds_p50"
_METRIC_RECENT_DURATION_P95: str = "juniper_cascor_worker_recent_task_duration_seconds_p95"
_METRIC_HEARTBEAT_AGE: str = "juniper_cascor_worker_heartbeat_age_seconds"


class WorkerRegistryCollector:
    """Bridge a :class:`WorkerRegistry` snapshot to Prometheus on each scrape.

    Intentionally NOT a subclass of any prometheus_client type — the
    library uses duck-typing on ``collect()`` so a plain class with
    that one method works everywhere a ``Collector`` does. This keeps
    the unit test free of prometheus_client registry side-effects (the
    test instantiates the class directly and calls ``collect()``).

    Args:
        registry: The cascor :class:`WorkerRegistry` to snapshot. The
            collector keeps a reference and re-reads it on every
            scrape; it does NOT copy on construction.
        time_source: Optional callable returning the current wall-clock
            time as a float (defaults to :func:`time.time`). Injected
            for deterministic unit tests.
    """

    def __init__(
        self,
        registry: "WorkerRegistry",
        *,
        time_source: Callable[[], float] = time.time,
    ) -> None:
        self._registry = registry
        self._now = time_source

    def collect(self) -> Iterable["Metric"]:  # noqa: D401 — prometheus_client interface
        """Snapshot the registry and emit per-worker gauge samples.

        Called by ``prometheus_client.REGISTRY`` on every ``/metrics``
        scrape. All sample emission is best-effort — a malformed
        registration is logged and skipped rather than failing the
        scrape.
        """
        # Local imports keep prometheus_client an optional runtime
        # dependency at module import time (the cascor pattern in
        # ``observability.py``).
        from prometheus_client.core import GaugeMetricFamily

        last_task_duration = GaugeMetricFamily(
            _METRIC_LAST_TASK_DURATION,
            "Wall-clock duration of the most recently completed task on this worker.",
            labels=["worker_id"],
        )
        gpu_utilization = GaugeMetricFamily(
            _METRIC_GPU_UTILIZATION_PCT,
            "Best-effort 0-100 GPU utilization reading on this worker (CUDA / NVML / torch).",
            labels=["worker_id"],
        )
        recent_p50 = GaugeMetricFamily(
            _METRIC_RECENT_DURATION_P50,
            "p50 of the worker's sliding window of recent task durations (seconds).",
            labels=["worker_id"],
        )
        recent_p95 = GaugeMetricFamily(
            _METRIC_RECENT_DURATION_P95,
            "p95 of the worker's sliding window of recent task durations (seconds).",
            labels=["worker_id"],
        )
        heartbeat_age = GaugeMetricFamily(
            _METRIC_HEARTBEAT_AGE,
            "Seconds since the worker's last heartbeat. Critical for SLI 'heartbeat freshness'.",
            labels=["worker_id"],
        )

        now = self._now()
        # OBS-WIRE-02 (E.2): take a frozen snapshot of every metric-
        # relevant field under ``self._registry._lock`` via the
        # registry's ``snapshot_for_metrics`` accessor. Reading
        # ``recent_task_durations_seconds`` directly off the live
        # registration object outside the lock raced with concurrent
        # ``record_heartbeat`` calls; the snapshot returns immutable
        # tuples so percentile computation cannot observe a partial
        # write.
        try:
            snapshots = self._registry.snapshot_for_metrics()
        except Exception:
            logger.exception("WorkerRegistryCollector failed to snapshot registry — emitting empty scrape")
            snapshots = []

        for snap in snapshots:
            try:
                worker_id = snap["worker_id"]

                # Always emit heartbeat age — every registration has a
                # ``last_heartbeat`` populated by the constructor.
                heartbeat_age.add_metric([worker_id], max(0.0, now - snap["last_heartbeat"]))

                # Optional fields: skip on None / empty (do NOT zero-emit).
                last_dur = snap["last_task_duration_seconds"]
                if last_dur is not None:
                    last_task_duration.add_metric([worker_id], float(last_dur))

                gpu_util = snap["gpu_utilization_pct"]
                if gpu_util is not None:
                    gpu_utilization.add_metric([worker_id], float(gpu_util))

                # ``statistics.quantiles`` requires at least 2 samples;
                # below that threshold any percentile is degenerate so
                # we omit rather than emit a misleading value.
                window = list(snap["recent_task_durations_seconds"] or ())
                if len(window) >= 2:
                    p50, p95 = _percentiles(window)
                    recent_p50.add_metric([worker_id], p50)
                    recent_p95.add_metric([worker_id], p95)
            except Exception:
                logger.exception(
                    "WorkerRegistryCollector skipping malformed snapshot (worker_id=%r)",
                    snap.get("worker_id", "<unknown>") if isinstance(snap, dict) else "<unknown>",
                )

        yield heartbeat_age
        yield last_task_duration
        yield gpu_utilization
        yield recent_p50
        yield recent_p95


def _percentiles(samples: list[float]) -> tuple[float, float]:
    """Compute (p50, p95) over a list of floats.

    Uses :func:`statistics.quantiles` with ``n=20`` so the boundary at
    index 9 is the 50th percentile (10/20) and the boundary at index
    18 is the 95th percentile (19/20). Caller MUST guarantee
    ``len(samples) >= 2`` — :func:`statistics.quantiles` raises on
    fewer.
    """
    qs = statistics.quantiles(samples, n=20, method="inclusive")
    # qs has 19 cut points: qs[i] is the (i+1)/20 percentile.
    p50 = qs[9]
    p95 = qs[18]
    return float(p50), float(p95)
