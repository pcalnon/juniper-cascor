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

import ipaddress
import logging
import os
import threading
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union

# Cross-service primitives — re-exported from juniper-observability.
from juniper_observability import DEFAULT_LOG_FORMAT_PLAIN, DEFAULT_SENTRY_TRACES_SAMPLE_RATE, LOG_FORMAT_JSON, UNMATCHED_ENDPOINT_LABEL, JuniperJsonFormatter, PrometheusMiddleware, RequestIdMiddleware  # noqa: F401 — re-exported for backwards compat
from juniper_observability import configure_sentry as _shared_configure_sentry
from juniper_observability import get_prometheus_app, request_id_var, set_build_info  # noqa: F401 — re-exported for backwards compat

# Re-export the SEC-15 hook so ``main.py``'s direct sentry_sdk.init still
# resolves it through the historical import path.
from juniper_observability.sentry import _strip_sensitive_headers  # noqa: F401 — re-exported for backwards compat

from cascor_constants.constants_logging.constants_logging import _LOGGER_LOG_FILE_BACKUP_COUNT, _LOGGER_LOG_FILE_MAX_BYTES, _LOGGER_SENTRY_TRACES_SAMPLE_RATE

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

# OBS-WIRE-01 (E.1): module-level locks guarding the lazy-init dicts.
# Prior to OBS-WIRE-01 the ``if _xxx_metrics is None:`` branches in
# ``_ensure_training_metrics`` / ``_ensure_ws_metrics`` were not lock-
# protected. Two concurrent first-callers could both enter the
# ``is None`` branch; the second would hit ``Duplicated timeseries``,
# ``_register_or_reuse`` would unregister the live collector, and the
# first caller's reference would be orphaned. The locks make the
# check-and-init pair atomic so exactly one caller registers.
_training_metrics_lock = threading.Lock()
_ws_metrics_lock = threading.Lock()

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


# Idempotent collector registration helper. The original local
# implementation (cascor PR ?? from before juniper-observability 0.2.0)
# used drop-and-recreate semantics: on a duplicate, unregister the
# existing collector and retry the construction so the latest call's
# labels/buckets win at the cost of discarding any accumulated samples.
#
# Phase 2c migration (juniper-ml#216 / juniper-observability 0.2.0):
# the call sites below were already calling ``_register_or_reuse(cls,
# name, *args, **kwargs)`` with identical args every time (lazy-init
# singleton dict pattern), so the drop-and-recreate semantics never
# meaningfully differed from adopt-existing for production. The
# canonical helper is ``juniper_observability.register_or_reuse`` which
# adopts the existing collector — preserving accumulated samples across
# in-process re-init (test fixtures rebuilding the app, dev autoreload).
# Aliased here as ``_register_or_reuse`` so the 22+ call sites in this
# module remain stable.
#
# Use ``juniper_observability.register_fresh`` instead at any future
# call site that genuinely wants the drop-and-recreate behaviour
# (test fixture exercising different bucket boundaries, etc.).
from juniper_observability import register_or_reuse as _register_or_reuse


def _ensure_training_metrics() -> dict:
    """Create training-related Prometheus metrics on first access.

    OBS-WIRE-01 (E.1): the check-and-init is wrapped in
    ``_training_metrics_lock`` so concurrent first-callers cannot both
    enter the ``is None`` branch and orphan a collector via the
    ``_register_or_reuse`` recovery path.
    """
    global _training_metrics
    # Fast path — the lock is only needed during the one-time init.
    if _training_metrics is not None:
        return _training_metrics
    with _training_metrics_lock:
        # Double-checked: another thread may have completed init while
        # we waited on the lock.
        if _training_metrics is not None:
            return _training_metrics
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
            # OBS-WIRE-01 (A.5): the ``juniper_cascor_inference_*``
            # family was removed because cascor exposes no inference
            # endpoint — predictions are produced by downstream
            # consumers (canopy / data services), not by this service.
            # The previously-defined Counter and Histogram had zero
            # production callers. See PR body for the wire-vs-remove
            # rationale.
            #
            # OBS-WIRE-01 (A.6): the ``phase`` label was dropped from
            # ``juniper_cascor_training_step_duration_seconds``. Pre-
            # OBS-WIRE-01 the only emission site (lifecycle/manager.py
            # output-phase callback) hard-coded ``phase="output"`` so
            # the regex ``phase=~"input|candidate|output"`` in the SLI
            # PromQL only ever matched a single value. The label
            # carried no signal and was misleading; keeping it would
            # have required adding analogous emission sites at the
            # candidate / input phases (option (b) of the audit), which
            # is not warranted while cascade-correlation has only an
            # output-phase epoch loop. See PR body for the
            # juniper-deploy SLI / alert follow-up.
            "step_duration_seconds": _register_or_reuse(
                Histogram,
                "juniper_cascor_training_step_duration_seconds",
                "Buckets target SLO 3.4 (p95 < 5s); see SLO_CATALOG §3.4 in juniper-deploy.",
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


# OBS-WIRE-01 (A.5): ``record_inference`` removed alongside the
# ``juniper_cascor_inference_*`` metric family. Cascor has no inference
# endpoint, so the helper was dead. Restore it (and the metric
# definitions above) if a future PR adds ``POST /v1/predict`` or
# similar.


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


def observe_training_step_duration(duration: float) -> None:
    """Record a single train-step duration observation.

    METRICS-MON R5.4-pre: measures wall-clock around one
    forward+backward+update cycle. Binds SLO 3.4 (p95 train-step
    duration < 5 s) — see juniper-deploy notes/SLO_CATALOG_2026-05-03.md
    §3.4 and §6 of the cascor histogram-buckets rationale doc.

    OBS-WIRE-01 (A.6): the ``phase`` label was dropped — see the
    metric-definition comment in ``_ensure_training_metrics`` above.

    Args:
        duration: Step duration in seconds (typically a
            ``time.perf_counter`` delta).
    """
    _ensure_training_metrics()["step_duration_seconds"].observe(duration)


# ---------------------------------------------------------------------------
# WebSocket metrics — Phase 0-cascor (15 metrics)
# ---------------------------------------------------------------------------

_ws_metrics: dict | None = None

_WS_RESUME_REPLAY_BUCKETS = (0, 1, 5, 25, 100, 500, 1024)

# OBS-WIRE-02 (Q3): closed-set endpoint values for the
# ``cascor_ws_connections_active{endpoint}`` gauge. The cascor app
# exposes exactly these three websocket endpoints (see
# ``api.app:457-459`` — ``app.websocket("/ws/training")`` /
# ``app.websocket("/ws/control")`` / ``app.websocket("/ws/v1/workers")``).
# The label value is the trailing path segment chosen for human
# readability ("training" / "control" / "workers"). Mirrors the R5.4-pre
# ``_TRAINING_SESSION_STATUSES`` closed-set discipline.
WS_ENDPOINT_TRAINING: str = "training"
WS_ENDPOINT_CONTROL: str = "control"
WS_ENDPOINT_WORKERS: str = "workers"
_WS_ENDPOINTS: frozenset[str] = frozenset(
    {
        WS_ENDPOINT_TRAINING,
        WS_ENDPOINT_CONTROL,
        WS_ENDPOINT_WORKERS,
    }
)

# OBS-WIRE-02 (3.4): closed-set outcome values for the
# ``cascor_ws_resume_requests_total{outcome}`` counter. Mirrors the four
# return arms of ``training_stream._handle_resume`` — see
# state-analysis doc ``A9_AND_3_2_STATE_ANALYSIS_2026-05-03.md`` §3.4.
WS_RESUME_OUTCOME_SUCCESS: str = "success"
WS_RESUME_OUTCOME_OUT_OF_RANGE: str = "out_of_range"
WS_RESUME_OUTCOME_MALFORMED: str = "malformed_resume"
WS_RESUME_OUTCOME_SERVER_RESTARTED: str = "server_restarted"
_WS_RESUME_OUTCOMES: frozenset[str] = frozenset(
    {
        WS_RESUME_OUTCOME_SUCCESS,
        WS_RESUME_OUTCOME_OUT_OF_RANGE,
        WS_RESUME_OUTCOME_MALFORMED,
        WS_RESUME_OUTCOME_SERVER_RESTARTED,
    }
)

# OBS-WIRE-02 (3.10): closed-set status values for the
# ``cascor_ws_command_responses_total{command,status}`` counter. The
# ``command`` label is open-set-by-convention (validated upstream via
# ``_VALID_COMMANDS`` in ``control_stream``); the ``status`` label is
# the closed three-way split used by the dispatch in
# ``control_stream._handle_command_message``.
WS_COMMAND_STATUS_SUCCESS: str = "success"
WS_COMMAND_STATUS_ERROR: str = "error"
WS_COMMAND_STATUS_RATE_LIMITED: str = "rate_limited"
_WS_COMMAND_STATUSES: frozenset[str] = frozenset(
    {
        WS_COMMAND_STATUS_SUCCESS,
        WS_COMMAND_STATUS_ERROR,
        WS_COMMAND_STATUS_RATE_LIMITED,
    }
)

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
    """Create WebSocket-related Prometheus metrics on first access.

    OBS-WIRE-01 (E.1): same lock-guarded check-and-init pattern used in
    ``_ensure_training_metrics`` — see that function's docstring for
    the rationale.
    """
    global _ws_metrics
    if _ws_metrics is not None:
        return _ws_metrics
    with _ws_metrics_lock:
        if _ws_metrics is not None:
            return _ws_metrics
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
            # OBS-WIRE-02 (Q1, option (a)): the formerly-defined
            # ``cascor_ws_seq_gap_detected_total`` Counter has been
            # removed. Sequence-gap detection is inherently client-side
            # truth — the server emits monotonic seq numbers and has no
            # read-back signal that would let it observe its own gaps.
            # The replacement metric ``juniper_canopy_ws_seq_gap_detected_total``
            # (with cross-service correlation labels) lives on the
            # canopy side per the OBS-WIRE-02 sister PR.
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
    """Increment the resume requests counter by outcome.

    OBS-WIRE-02 (3.4): ``outcome`` is closed-set — one of
    ``success`` / ``out_of_range`` / ``malformed_resume`` /
    ``server_restarted``. Drift is rejected at the helper boundary so
    instrumentation regressions surface as test failures rather than
    high-cardinality silent drift (R1.1).
    """
    if outcome not in _WS_RESUME_OUTCOMES:
        raise ValueError(f"invalid resume outcome {outcome!r}; expected one of {sorted(_WS_RESUME_OUTCOMES)!r}")
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
    """Set the active connections gauge for a given endpoint.

    OBS-WIRE-02 (Q3, option (b)): ``endpoint`` is closed-set and must
    match one of the three websocket routes registered in
    :mod:`api.app` — ``training`` / ``control`` / ``workers``. The
    value is wired by :class:`WebSocketManager` per-endpoint
    bookkeeping. Drift is rejected at the helper boundary to keep
    cardinality bounded (R1.1).
    """
    if endpoint not in _WS_ENDPOINTS:
        raise ValueError(f"invalid ws endpoint {endpoint!r}; expected one of {sorted(_WS_ENDPOINTS)!r}")
    _ensure_ws_metrics()["connections_active"].labels(endpoint=endpoint).set(value)


def ws_inc_command_responses(command: str, status: str) -> None:
    """Increment the command responses counter.

    OBS-WIRE-02 (3.10): the ``status`` label is closed-set — one of
    ``success`` / ``error`` / ``rate_limited``. ``command`` is
    open-set-by-convention (validated upstream via ``_VALID_COMMANDS``
    in ``control_stream``).
    """
    if status not in _WS_COMMAND_STATUSES:
        raise ValueError(f"invalid command status {status!r}; expected one of {sorted(_WS_COMMAND_STATUSES)!r}")
    _ensure_ws_metrics()["command_responses_total"].labels(command=command, status=status).inc()


def ws_observe_command_handler(command: str, duration: float) -> None:
    """Record command handler execution duration."""
    _ensure_ws_metrics()["command_handler_seconds"].labels(command=command).observe(duration)


# ---------------------------------------------------------------------------
# SEC-16 / POC remediation §3.1: MetricsAuthMiddleware — cascor-side IP
# allowlist for the ``/metrics`` mount. Mirrors
# ``juniper_data.api.observability.MetricsAuthMiddleware`` verbatim
# (validator-A recommendation in POC_REMEDIATION_PLAN_2026-05-27 §3.1 was
# "promote from juniper-data or duplicate inline"; promotion to
# ``juniper-observability`` is a roadmap §R5 gating issue, so duplicate-
# inline keeps the rollout independent).
#
# Why this is needed even with ``/metrics`` exempt from
# ``SecurityMiddleware`` (POC §3.1 "exempt" half): a misconfigured
# deployment (port 8200 published directly, or running outside compose /
# K8s) would expose ``/metrics`` with zero auth. The IP allowlist closes
# that exposure surface.
# ---------------------------------------------------------------------------

METRICS_DEFAULT_TRUSTED_IPS: tuple[str, ...] = ("127.0.0.1", "::1")

_NetworkT = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


def _parse_trusted_networks(raw: Iterable[str]) -> tuple[_NetworkT, ...]:
    """Compile a list of CIDR strings / bare IPs into ``ip_network`` objects.

    Bare-IP entries are widened to host networks (``/32`` or ``/128``) by
    ``ip_network(..., strict=False)``. Unparseable entries fail loud at
    init time — operator typos must not silently 403 every scrape. Mirror
    of ``juniper_data.api.observability._parse_trusted_networks``.
    """
    nets: list[_NetworkT] = []
    for entry in raw:
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
        except ValueError as exc:
            raise ValueError(f"JUNIPER_CASCOR_METRICS_TRUSTED_IPS entry {entry!r} is not a valid IP or CIDR: {exc}") from exc
    return tuple(nets)


def _normalize_client_ip(client_ip: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Strip IPv6 zone-id and unwrap IPv4-mapped IPv6 to its IPv4 form.

    - ``fe80::1%eth0`` → ``fe80::1`` (zone-id rejected by ``ip_address``).
    - ``::ffff:172.18.0.5`` → ``172.18.0.5`` (membership in an IPv4 CIDR
      otherwise returns ``False`` despite the underlying address being IPv4).
    """
    if "%" in client_ip:
        client_ip = client_ip.split("%", 1)[0]
    addr = ipaddress.ip_address(client_ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr


class MetricsAuthMiddleware:
    """ASGI wrapper that restricts ``/metrics`` to a trusted IP allowlist.

    Accepts bare IP literals, CIDR ranges, and a mix of both. IPv6 zone
    identifiers are stripped from the client address; IPv4-mapped IPv6
    addresses are unwrapped before membership check so a Docker container
    appearing as ``::ffff:172.18.0.5`` matches an IPv4 ``172.18.0.0/16``
    range. Unparseable allowlist entries raise at init time (fail-loud).
    """

    def __init__(
        self,
        app,
        trusted_ips: Iterable[str] | None = None,
    ) -> None:
        self.app = app
        raw = trusted_ips if trusted_ips is not None else METRICS_DEFAULT_TRUSTED_IPS
        self.networks: tuple[_NetworkT, ...] = _parse_trusted_networks(raw)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            allowed = False
            client = scope.get("client")
            client_ip = client[0] if client else None
            if client_ip:
                try:
                    addr = _normalize_client_ip(client_ip)
                    allowed = any(addr in net for net in self.networks)
                except ValueError:
                    logging.getLogger(__name__).warning(
                        "Denying /metrics request due to unparseable client IP: %r",
                        client_ip,
                    )
            if not allowed:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [(b"content-type", b"text/plain; charset=utf-8")],
                    }
                )
                await send({"type": "http.response.body", "body": b"Forbidden"})
                return
        await self.app(scope, receive, send)
