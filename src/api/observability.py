"""Observability surface for juniper-cascor.

METRICS-MON R2.1.4 / seed-06: the cross-cutting machinery
(:class:`JuniperJsonFormatter`, :class:`RequestIdMiddleware`,
:class:`PrometheusMiddleware`, :data:`UNMATCHED_ENDPOINT_LABEL`,
:data:`request_id_var`, :func:`get_prometheus_app`,
:func:`set_build_info`) lives in the shared
:mod:`juniper_observability` package and is re-exported here for
backwards compatibility with existing imports across
``api.app``, route handlers, and tests.

What stays in this module:

- :func:`configure_logging` — wraps the shared formatter with cascor's
  :class:`RotatingFileHandler` for on-disk persistence.
- :func:`configure_sentry` — thin wrapper that delegates to the shared
  implementation while pinning cascor's ``traces_sample_rate``.
- The service-specific training and WebSocket Prometheus metrics
  (:func:`record_training_epoch`, :func:`set_training_loss`, the
  ``ws_*`` helpers, and the lazy-init dicts that back them).

New code should prefer ``from juniper_observability import …`` for the
re-exported symbols to make the dependency on the shared lib explicit.

See: notes/code-review/METRICS_MONITORING_R2.1_SHARED_OBSERVABILITY_DESIGN_2026-04-28.md
in juniper-ml.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Cross-service primitives — re-exported from juniper-observability.
from juniper_observability import DEFAULT_LOG_FORMAT_PLAIN, DEFAULT_SENTRY_TRACES_SAMPLE_RATE, LOG_FORMAT_JSON, UNMATCHED_ENDPOINT_LABEL, JuniperJsonFormatter, PrometheusMiddleware, RequestIdMiddleware  # noqa: F401 — re-exported for backwards compat
from juniper_observability import configure_sentry as _shared_configure_sentry
from juniper_observability import get_prometheus_app, request_id_var, set_build_info  # noqa: F401 — re-exported for backwards compat

# Re-export the SEC-15 hook so ``main.py``'s direct sentry_sdk.init still
# resolves it through the historical import path.
from juniper_observability.sentry import _strip_sensitive_headers  # noqa: F401 — re-exported for backwards compat

from cascor_constants.constants_logging.constants_logging import _LOGGER_LOG_FILE_BACKUP_COUNT, _LOGGER_LOG_FILE_MAX_BYTES, _LOGGER_PROMETHEUS_LATENCY_BUCKETS, _LOGGER_SENTRY_TRACES_SAMPLE_RATE

_SERVICE_NAME_DEFAULT: str = "juniper-cascor"
_NAMESPACE_DEFAULT: str = "juniper_cascor"


def _resolve_log_dir() -> Path:
    """Resolve the absolute log directory path.

    Uses the project constants if available, otherwise falls back to
    a ``logs/`` directory relative to the project root (two levels up
    from ``src/api/``).
    """
    try:
        from cascor_constants.constants import _PROJECT_LOG_DIR_DEFAULT

        return Path(_PROJECT_LOG_DIR_DEFAULT)
    except ImportError:
        return Path(__file__).resolve().parent.parent.parent / "logs"


def configure_logging(log_level: str, log_format: str, service_name: str = _SERVICE_NAME_DEFAULT) -> None:
    """Configure logging — JSON when log_format='json', plain text otherwise.

    Wraps the shared :class:`JuniperJsonFormatter` with juniper-cascor's
    :class:`RotatingFileHandler` so structured logs continue to land in
    the canonical ``logs/juniper_cascor.log`` (fix H2). The shared lib's
    :func:`juniper_observability.configure_logging` is intentionally
    *not* called here — its console handler would race the file handler
    set up below and the lib has no notion of cascor's log directory
    layout.

    Args:
        log_level: Logging level string (e.g. "INFO", "DEBUG").
        log_format: Format mode — "json" for structured JSON, anything else for plain text.
        service_name: Service name included in JSON log entries.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    if log_format == LOG_FORMAT_JSON:
        console_handler.setFormatter(JuniperJsonFormatter(service=service_name))
    else:
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    root.addHandler(console_handler)

    log_dir = _resolve_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / "juniper_cascor.log"
    file_handler = RotatingFileHandler(
        str(log_file),
        maxBytes=_LOGGER_LOG_FILE_MAX_BYTES,
        backupCount=_LOGGER_LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("[%(filename)s: %(funcName)s:%(lineno)d] (%(asctime)s) [%(levelname)s] %(message)s"))
    root.addHandler(file_handler)


def configure_sentry(dsn: str | None, service_name: str, version: str) -> None:
    """Initialize Sentry via the shared :func:`juniper_observability.configure_sentry`.

    Cascor pins ``traces_sample_rate`` to ``_LOGGER_SENTRY_TRACES_SAMPLE_RATE``
    (1.0 — full trace sampling for the research workload) rather than the
    shared lib's default (0.1). ``send_pii`` stays at the secure default
    (False); the shared ``before_send`` hook still scrubs sensitive headers.

    Args:
        dsn: Sentry DSN URL. Pass None or empty string to skip initialization.
        service_name: Service name for Sentry environment tag.
        version: Application version string.
    """
    _shared_configure_sentry(
        dsn,
        service_name,
        version,
        traces_sample_rate=_LOGGER_SENTRY_TRACES_SAMPLE_RATE,
    )


# ---------------------------------------------------------------------------
# Custom application metrics — lazily initialized to avoid requiring
# prometheus_client at import time (it is an optional dependency).
# ---------------------------------------------------------------------------

_training_metrics: dict | None = None

# METRICS-MON R5.4-pre: bucket layout for the train-step duration
# histogram. Targets SLO 3.4 ("p95 train-step duration < 5 s"; see
# juniper-deploy notes/SLO_CATALOG_2026-05-03.md §3.4) and bracket every
# legitimate operational regime: sub-100 ms (small models, hot cache),
# ~1 s (typical mid-sized network output-phase epoch), 5 s (SLO p95
# target), 30 s (saturated GPU / very large network). Boundaries are
# logarithmic-ish to keep bucket count low (9 incl. +inf) while still
# giving quantile-precision around the SLO target. Per-boundary
# rationale in
# ``notes/observability/HISTOGRAM_BUCKETS_RATIONALE_2026-05-02.md`` §6.
_TRAINING_STEP_DURATION_BUCKETS: tuple = (
    0.05,  # 50 ms
    0.1,  # 100 ms
    0.5,  # 500 ms
    1.0,  # 1 s
    2.0,  # 2 s
    5.0,  # 5 s — SLO 3.4 p95 target
    10.0,  # 10 s
    30.0,  # 30 s
    float("inf"),
)


def _register_or_reuse(cls, name: str, *args, **kwargs):
    """Construct a Prometheus collector idempotently against the default registry.

    The lazy ``_ensure_*_metrics`` singletons cache module-globals as
    Python references. Tests sometimes clear those globals (to force a
    fresh ``_ensure`` path) without unregistering the underlying
    collectors from ``prometheus_client.REGISTRY``. The next ``cls(name,
    ...)`` then raises ``ValueError: Duplicated timeseries`` because the
    metric name is still bound in the registry even though our cache
    forgot the Python reference.

    Catch that specific case, look the existing collector up by name,
    unregister it, and retry — yielding a fresh collector that reflects
    the current call's labels/buckets without depending on stale state.
    Other ``ValueError`` causes (genuinely invalid name, etc.) are
    re-raised unchanged.
    """
    from prometheus_client import REGISTRY

    try:
        return cls(name, *args, **kwargs)
    except ValueError as exc:
        if "Duplicated timeseries" not in str(exc):
            raise
        # Find the orphaned collector by name and unregister it. The
        # ``_collector_to_names`` mapping is the same lookup ``register``
        # itself uses for the duplicate check.
        for collector, names in list(REGISTRY._collector_to_names.items()):
            if name in names:
                REGISTRY.unregister(collector)
                break
        return cls(name, *args, **kwargs)


def _ensure_training_metrics() -> dict:
    """Create training-related Prometheus metrics on first access."""
    global _training_metrics
    if _training_metrics is None:
        from prometheus_client import Counter, Gauge, Histogram

        _training_metrics = {
            "sessions_active": _register_or_reuse(
                Gauge,
                "juniper_cascor_training_sessions_active",
                "Number of currently active training sessions",
            ),
            # METRICS-MON R5.4-pre: terminal-transition counter that
            # binds SLO 3.3 (training-session success ratio). Closed-set
            # ``status`` label values — ``success`` / ``failure`` /
            # ``cancelled`` — per R1.1 cardinality discipline. Bumped
            # exactly once per terminal FSM transition by the lifecycle
            # manager. SLO PromQL:
            #   sum(rate(juniper_cascor_training_sessions_completed_total{status="success"}[5m]))
            #   / sum(rate(juniper_cascor_training_sessions_completed_total[5m]))
            # See juniper-deploy notes/SLO_CATALOG_2026-05-03.md §3.3.
            "sessions_completed_total": _register_or_reuse(
                Counter,
                "juniper_cascor_training_sessions_completed_total",
                "Total terminal training-session transitions by outcome (closed-set status)",
                ["status"],
            ),
            "epochs_total": _register_or_reuse(
                Counter,
                "juniper_cascor_training_epochs_total",
                "Total training epochs completed across all sessions",
                ["phase"],
            ),
            "loss": _register_or_reuse(
                Gauge,
                "juniper_cascor_training_loss",
                "Current training loss value",
                ["phase", "loss_type"],
            ),
            "accuracy_ratio": _register_or_reuse(
                Gauge,
                "juniper_cascor_training_accuracy_ratio",
                "Current training accuracy (0-1 ratio)",
                ["phase"],
            ),
            "hidden_units_total": _register_or_reuse(
                Gauge,
                "juniper_cascor_hidden_units_total",
                "Current number of hidden units in the cascade network",
            ),
            "candidate_correlation": _register_or_reuse(
                Gauge,
                "juniper_cascor_candidate_correlation",
                "Best candidate unit correlation with residual error",
            ),
            "inference_requests_total": _register_or_reuse(
                Counter,
                "juniper_cascor_inference_requests_total",
                "Total inference requests processed",
            ),
            "inference_duration_seconds": _register_or_reuse(
                Histogram,
                "juniper_cascor_inference_duration_seconds",
                # METRICS-MON R4.1: bucket layout is **tentative pending
                # R5.1**. Per-boundary SLO rationale in
                # ``notes/observability/HISTOGRAM_BUCKETS_RATIONALE_2026-05-02.md``.
                "Inference latency in seconds (R4.1 buckets tentative pending R5.1)",
                buckets=_LOGGER_PROMETHEUS_LATENCY_BUCKETS,
            ),
            # METRICS-MON R5.4-pre: train-step duration histogram born
            # SLO-aligned (no "tentative pending R5.1" suffix). Buckets
            # target SLO 3.4 (p95 < 5s); see SLO_CATALOG §3.4 in
            # juniper-deploy and §6 of
            # ``notes/observability/HISTOGRAM_BUCKETS_RATIONALE_2026-05-02.md``
            # for per-boundary rationale and the train-step boundary
            # choice. The metric measures wall-clock around one
            # forward+backward+update cycle (an "epoch" at the
            # api-lifecycle level — see §1 / §6 of the rationale doc for
            # the boundary discussion).
            "step_duration_seconds": _register_or_reuse(
                Histogram,
                "juniper_cascor_training_step_duration_seconds",
                "Buckets target SLO 3.4 (p95 < 5s); see SLO_CATALOG §3.4 in juniper-deploy.",
                ["phase"],
                buckets=_TRAINING_STEP_DURATION_BUCKETS,
            ),
        }
    return _training_metrics


def record_training_epoch(phase: str) -> None:
    """Increment the training epoch counter.

    Args:
        phase: Training phase — "input", "candidate", or "output".
    """
    _ensure_training_metrics()["epochs_total"].labels(phase=phase).inc()


def set_training_loss(phase: str, loss_type: str, value: float) -> None:
    """Update the current training loss gauge.

    Args:
        phase: Training phase — "input", "candidate", or "output".
        loss_type: Loss type — "train" or "validation".
        value: Loss value.
    """
    _ensure_training_metrics()["loss"].labels(phase=phase, loss_type=loss_type).set(value)


def set_training_accuracy(phase: str, value: float) -> None:
    """Update the current training accuracy gauge.

    Args:
        phase: Training phase — "input", "candidate", or "output".
        value: Accuracy as a 0-1 ratio.
    """
    _ensure_training_metrics()["accuracy_ratio"].labels(phase=phase).set(value)


def set_hidden_units(count: int) -> None:
    """Update the hidden units gauge.

    Args:
        count: Current number of hidden units in the cascade network.
    """
    _ensure_training_metrics()["hidden_units_total"].set(count)


def set_candidate_correlation(value: float) -> None:
    """Update the best candidate correlation gauge.

    Args:
        value: Best candidate correlation with residual error.
    """
    _ensure_training_metrics()["candidate_correlation"].set(value)


def inc_training_sessions() -> None:
    """Increment the active training sessions gauge."""
    _ensure_training_metrics()["sessions_active"].inc()


def dec_training_sessions() -> None:
    """Decrement the active training sessions gauge."""
    _ensure_training_metrics()["sessions_active"].dec()


def record_inference(duration: float) -> None:
    """Record an inference request.

    Args:
        duration: Inference duration in seconds.
    """
    m = _ensure_training_metrics()
    m["inference_requests_total"].inc()
    m["inference_duration_seconds"].observe(duration)


# METRICS-MON R5.4-pre: closed-set status values for the
# ``training_sessions_completed_total`` counter (R1.1 cardinality
# discipline — every increment site MUST pass one of these strings).
TRAINING_SESSION_STATUS_SUCCESS: str = "success"
TRAINING_SESSION_STATUS_FAILURE: str = "failure"
TRAINING_SESSION_STATUS_CANCELLED: str = "cancelled"
_TRAINING_SESSION_STATUSES: frozenset[str] = frozenset(
    {
        TRAINING_SESSION_STATUS_SUCCESS,
        TRAINING_SESSION_STATUS_FAILURE,
        TRAINING_SESSION_STATUS_CANCELLED,
    }
)


def inc_training_session_completed(status: str) -> None:
    """Increment the terminal-transition counter for a training session.

    METRICS-MON R5.4-pre: bumped exactly once per terminal FSM transition
    by the lifecycle manager. Binds SLO 3.3 (training-session success
    ratio) — see juniper-deploy notes/SLO_CATALOG_2026-05-03.md §3.3.

    Args:
        status: One of ``"success"``, ``"failure"``, or ``"cancelled"``
            (closed set, R1.1 cardinality discipline).

    Raises:
        ValueError: If ``status`` is not in the closed set. Catches
            instrumentation drift early rather than silently emitting
            high-cardinality labels.
    """
    if status not in _TRAINING_SESSION_STATUSES:
        raise ValueError(f"invalid training-session status {status!r}; expected one of {sorted(_TRAINING_SESSION_STATUSES)!r}")
    _ensure_training_metrics()["sessions_completed_total"].labels(status=status).inc()


def observe_training_step_duration(phase: str, duration: float) -> None:
    """Record a single train-step duration observation.

    METRICS-MON R5.4-pre: measures wall-clock around one
    forward+backward+update cycle. Binds SLO 3.4 (p95 train-step
    duration < 5 s) — see juniper-deploy notes/SLO_CATALOG_2026-05-03.md
    §3.4 and §6 of the cascor histogram-buckets rationale doc.

    Args:
        phase: Training phase — "output", "candidate", or "input".
        duration: Step duration in seconds (typically a
            ``time.perf_counter`` delta).
    """
    _ensure_training_metrics()["step_duration_seconds"].labels(phase=phase).observe(duration)


# ---------------------------------------------------------------------------
# WebSocket metrics — Phase 0-cascor (15 metrics)
# ---------------------------------------------------------------------------

_ws_metrics: dict | None = None

_WS_RESUME_REPLAY_BUCKETS = (0, 1, 5, 25, 100, 500, 1024)

# METRICS-MON R5.1b: sub-millisecond bucket layout for the two
# WebSocket-side latency histograms whose actual distributions sit
# below the 5 ms floor of the Prometheus default layout. Boundaries
# (in seconds) — 100 µs, 500 µs, 1 ms, 5 ms, 10 ms, 50 ms, 100 ms,
# +inf — give sub-millisecond resolution for the healthy regime
# (socket writes; ``pause``/``resume`` flips) while still bracketing
# pathological slow-paths at 50 ms / 100 ms. Rationale and per-boundary
# justification live in
# ``notes/observability/HISTOGRAM_BUCKETS_RATIONALE_2026-05-02.md`` §4.
_WS_SUB_MS_LATENCY_BUCKETS: tuple = (
    0.0001,  # 100 µs
    0.0005,  # 500 µs
    0.001,  # 1 ms
    0.005,  # 5 ms
    0.01,  # 10 ms
    0.05,  # 50 ms
    0.1,  # 100 ms
    float("inf"),
)


def _ensure_ws_metrics() -> dict:
    """Create WebSocket-related Prometheus metrics on first access."""
    global _ws_metrics
    if _ws_metrics is None:
        from prometheus_client import Counter, Gauge, Histogram

        _ws_metrics = {
            "seq_current": _register_or_reuse(
                Gauge,
                "cascor_ws_seq_current",
                "Current sequence number for WebSocket broadcasts",
            ),
            "replay_buffer_occupancy": _register_or_reuse(
                Gauge,
                "cascor_ws_replay_buffer_occupancy",
                "Current number of messages in the replay buffer",
            ),
            "replay_buffer_bytes": _register_or_reuse(
                Gauge,
                "cascor_ws_replay_buffer_bytes",
                "Approximate memory usage of the replay buffer in bytes",
            ),
            "replay_buffer_capacity_configured": _register_or_reuse(
                Gauge,
                "cascor_ws_replay_buffer_capacity_configured",
                "Configured maximum replay buffer size",
            ),
            "resume_requests_total": _register_or_reuse(
                Counter,
                "cascor_ws_resume_requests_total",
                "Total resume requests by outcome",
                ["outcome"],
            ),
            "resume_replayed_events": _register_or_reuse(
                Histogram,
                "cascor_ws_resume_replayed_events",
                # METRICS-MON R4.1: bucket layout is **tentative pending
                # R5.1**. Discrete-count metric, not duration; boundaries
                # map to operational regimes of the replay buffer.
                # Rationale in
                # ``notes/observability/HISTOGRAM_BUCKETS_RATIONALE_2026-05-02.md``.
                "Number of events replayed per successful resume (R4.1 buckets tentative pending R5.1)",
                buckets=_WS_RESUME_REPLAY_BUCKETS,
            ),
            "broadcast_timeout_total": _register_or_reuse(
                Counter,
                "cascor_ws_broadcast_timeout_total",
                "Total broadcast send timeouts",
                ["type"],
            ),
            "broadcast_send_duration_seconds": _register_or_reuse(
                Histogram,
                "cascor_ws_broadcast_send_duration_seconds",
                # METRICS-MON R5.1b: re-bucketed from the Prometheus
                # default layout (5 ms floor) to a sub-millisecond
                # layout matching the actual distribution of WS socket
                # writes on a healthy connection. Per-boundary rationale
                # in
                # ``notes/observability/HISTOGRAM_BUCKETS_RATIONALE_2026-05-02.md``
                # §4.
                "Duration of individual WebSocket send operations",
                ["type"],
                buckets=_WS_SUB_MS_LATENCY_BUCKETS,
            ),
            "pending_connections": _register_or_reuse(
                Gauge,
                "cascor_ws_pending_connections",
                "Number of WebSocket connections in pending (resume handshake) state",
            ),
            "state_throttle_coalesced_total": _register_or_reuse(
                Counter,
                "cascor_ws_state_throttle_coalesced_total",
                "Total state broadcasts coalesced by throttle",
            ),
            "broadcast_from_thread_errors_total": _register_or_reuse(
                Counter,
                "cascor_ws_broadcast_from_thread_errors_total",
                "Total errors from broadcast_from_thread coroutine execution",
            ),
            "seq_gap_detected_total": _register_or_reuse(
                Counter,
                "cascor_ws_seq_gap_detected_total",
                "Total sequence gaps detected (should be zero in healthy operation)",
            ),
            "connections_active": _register_or_reuse(
                Gauge,
                "cascor_ws_connections_active",
                "Number of active WebSocket connections by endpoint",
                ["endpoint"],
            ),
            "command_responses_total": _register_or_reuse(
                Counter,
                "cascor_ws_command_responses_total",
                "Total command responses sent",
                ["command", "status"],
            ),
            "command_handler_seconds": _register_or_reuse(
                Histogram,
                "cascor_ws_command_handler_seconds",
                # METRICS-MON R5.1b: re-bucketed from the Prometheus
                # default layout to the sub-millisecond layout shared
                # with ``broadcast_send_duration_seconds``. The
                # ``command`` label spans a wide duration range — sub-ms
                # ``pause``/``resume`` flips through ~50 ms
                # ``update_params`` lifecycle paths — and the chosen
                # boundaries (100 µs → 100 ms, +inf) bracket every
                # command class without splitting the metric. Per-boundary
                # rationale in
                # ``notes/observability/HISTOGRAM_BUCKETS_RATIONALE_2026-05-02.md``
                # §4–§5.
                "Duration of command handler execution",
                ["command"],
                buckets=_WS_SUB_MS_LATENCY_BUCKETS,
            ),
        }
    return _ws_metrics


def ws_set_seq_current(value: int) -> None:
    """Update the current sequence number gauge."""
    _ensure_ws_metrics()["seq_current"].set(value)


def ws_set_replay_buffer_occupancy(value: int) -> None:
    """Update the replay buffer occupancy gauge."""
    _ensure_ws_metrics()["replay_buffer_occupancy"].set(value)


def ws_set_replay_buffer_capacity(value: int) -> None:
    """Set the configured replay buffer capacity gauge."""
    _ensure_ws_metrics()["replay_buffer_capacity_configured"].set(value)


def ws_inc_resume_requests(outcome: str) -> None:
    """Increment the resume requests counter by outcome."""
    _ensure_ws_metrics()["resume_requests_total"].labels(outcome=outcome).inc()


def ws_observe_resume_replayed(count: int) -> None:
    """Record the number of events replayed in a successful resume."""
    _ensure_ws_metrics()["resume_replayed_events"].observe(count)


def ws_inc_broadcast_timeout(msg_type: str) -> None:
    """Increment the broadcast timeout counter."""
    _ensure_ws_metrics()["broadcast_timeout_total"].labels(type=msg_type).inc()


def ws_inc_state_throttle_coalesced() -> None:
    """Increment the state throttle coalesced counter."""
    _ensure_ws_metrics()["state_throttle_coalesced_total"].inc()


def ws_inc_broadcast_from_thread_errors() -> None:
    """Increment the broadcast-from-thread errors counter."""
    _ensure_ws_metrics()["broadcast_from_thread_errors_total"].inc()


def ws_set_connections_active(endpoint: str, value: int) -> None:
    """Set the active connections gauge for a given endpoint."""
    _ensure_ws_metrics()["connections_active"].labels(endpoint=endpoint).set(value)


def ws_inc_command_responses(command: str, status: str) -> None:
    """Increment the command responses counter."""
    _ensure_ws_metrics()["command_responses_total"].labels(command=command, status=status).inc()


def ws_observe_command_handler(command: str, duration: float) -> None:
    """Record command handler execution duration."""
    _ensure_ws_metrics()["command_handler_seconds"].labels(command=command).observe(duration)
