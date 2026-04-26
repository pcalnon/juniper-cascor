"""Observability module for structured logging, Prometheus metrics, and Sentry integration."""

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from cascor_constants.constants_logging.constants_logging import _LOGGER_LOG_FILE_BACKUP_COUNT, _LOGGER_LOG_FILE_MAX_BYTES, _LOGGER_PROMETHEUS_LATENCY_BUCKETS, _LOGGER_SENTRY_TRACES_SAMPLE_RATE

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

_SERVICE_NAME_DEFAULT: str = "juniper-cascor"
_NAMESPACE_DEFAULT: str = "juniper_cascor"


class JuniperJsonFormatter(logging.Formatter):
    """JSON log formatter with request_id propagation."""

    def __init__(self, service: str = _SERVICE_NAME_DEFAULT) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self._service,
            "request_id": request_id_var.get(""),
        }
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Injects X-Request-ID into ContextVar and response header."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id_var.reset(token)


# METRICS-MON seed-01 / R1.1: bound cardinality. Restrict the ``endpoint``
# label to the resolved Starlette route template; collapse unmatched
# requests into ``UNMATCHED_ENDPOINT_LABEL`` and increment a separate
# counter so unmatched volume stays observable without polluting the
# histogram. Aligned with the same fix in juniper-data and juniper-canopy.
UNMATCHED_ENDPOINT_LABEL = "_unmatched"


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Tracks http_requests_total and http_request_duration_seconds with namespace prefix."""

    def __init__(self, app: object, service_name: str = _SERVICE_NAME_DEFAULT, namespace: str = _NAMESPACE_DEFAULT) -> None:
        super().__init__(app)
        from prometheus_client import Counter, Histogram

        prefix = f"{namespace}_" if namespace else ""
        self._request_count = Counter(
            f"{prefix}http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
        )
        self._request_duration = Histogram(
            f"{prefix}http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
        )
        self._unmatched_count = Counter(
            f"{prefix}http_unmatched_requests_total",
            "HTTP requests not matching any registered route template",
            ["method"],
        )

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        route = request.scope.get("route")
        template = getattr(route, "path", None) if route is not None else None
        method = request.method
        if template:
            endpoint = template
        else:
            endpoint = UNMATCHED_ENDPOINT_LABEL
            self._unmatched_count.labels(method=method).inc()

        status = str(response.status_code)

        self._request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self._request_duration.labels(method=method, endpoint=endpoint).observe(duration)

        return response


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
        # Fallback: derive from this file's location (src/api/observability.py -> project root)
        return Path(__file__).resolve().parent.parent.parent / "logs"


def configure_logging(log_level: str, log_format: str, service_name: str = _SERVICE_NAME_DEFAULT) -> None:
    """Configure logging — JSON when log_format='json', plain text otherwise.

    Args:
        log_level: Logging level string (e.g. "INFO", "DEBUG").
        log_format: Format mode — "json" for structured JSON, anything else for plain text.
        service_name: Service name included in JSON log entries.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicate output
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Console handler (StreamHandler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    if log_format == "json":
        console_handler.setFormatter(JuniperJsonFormatter(service=service_name))
    else:
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    root.addHandler(console_handler)

    # File handler — persist API logs to the canonical logs/ directory (fix H2)
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


# SEC-15: Header names that may carry secrets. We always strip them from
# Sentry events regardless of ``send_default_pii``; the default is now
# ``False`` so request headers are not uploaded at all, but the filter
# remains as a second line of defense if any future integration re-enables
# per-event header capture (e.g. a custom logging integration).
_SENTRY_SENSITIVE_HEADERS = frozenset({"x-api-key", "authorization", "cookie"})


def _strip_sensitive_headers(event, hint):  # noqa: ARG001 — Sentry hook signature
    """Replace any sensitive request headers in a Sentry event with ``[Filtered]``.

    Sentry calls this via ``before_send`` for every outbound event. The
    filter walks the request headers dict and only rewrites keys that
    match the sensitive set, so non-sensitive diagnostic headers still
    reach Sentry unchanged.
    """
    request_data = event.get("request", {}) if isinstance(event, dict) else {}
    headers = request_data.get("headers", {}) if isinstance(request_data, dict) else {}
    if isinstance(headers, dict):
        for key in list(headers.keys()):
            if key.lower() in _SENTRY_SENSITIVE_HEADERS:
                headers[key] = "[Filtered]"
    return event


def configure_sentry(dsn: str | None, service_name: str, version: str) -> None:
    """Initialize Sentry with FastAPI integration. No-op when dsn is None or empty.

    Args:
        dsn: Sentry DSN URL. Pass None or empty string to skip initialization.
        service_name: Service name for Sentry environment tag.
        version: Application version string.
    """
    if not dsn:
        return

    import sentry_sdk

    sentry_sdk.init(
        dsn=dsn,
        # SEC-15: never upload default PII (IP, cookies, user identifiers,
        # request headers). The before_send filter scrubs any headers that
        # slip through via other integrations.
        send_default_pii=False,
        enable_logs=True,
        traces_sample_rate=_LOGGER_SENTRY_TRACES_SAMPLE_RATE,
        release=f"{service_name}@{version}",
        before_send=_strip_sensitive_headers,
    )


def get_prometheus_app():
    """Return ASGI app for /metrics endpoint via prometheus_client.make_asgi_app().

    Returns:
        ASGI application serving Prometheus metrics.
    """
    from prometheus_client import make_asgi_app

    return make_asgi_app()


def set_build_info(namespace: str, version: str) -> None:
    """Set build information as a Prometheus Info metric.

    Args:
        namespace: Metric namespace prefix (e.g. "juniper_cascor").
        version: Application version string.
    """
    from prometheus_client import Info

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    info = Info(f"{namespace}_build", f"Build information for {namespace.replace('_', '-')} service")
    info.info({"version": version, "python_version": python_version})


# ---------------------------------------------------------------------------
# Custom application metrics — lazily initialized to avoid requiring
# prometheus_client at import time (it is an optional dependency).
# ---------------------------------------------------------------------------

_training_metrics: dict | None = None


def _ensure_training_metrics() -> dict:
    """Create training-related Prometheus metrics on first access."""
    global _training_metrics
    if _training_metrics is None:
        from prometheus_client import Counter, Gauge, Histogram

        _training_metrics = {
            "sessions_active": Gauge(
                "juniper_cascor_training_sessions_active",
                "Number of currently active training sessions",
            ),
            "epochs_total": Counter(
                "juniper_cascor_training_epochs_total",
                "Total training epochs completed across all sessions",
                ["phase"],
            ),
            "loss": Gauge(
                "juniper_cascor_training_loss",
                "Current training loss value",
                ["phase", "loss_type"],
            ),
            "accuracy_ratio": Gauge(
                "juniper_cascor_training_accuracy_ratio",
                "Current training accuracy (0-1 ratio)",
                ["phase"],
            ),
            "hidden_units_total": Gauge(
                "juniper_cascor_hidden_units_total",
                "Current number of hidden units in the cascade network",
            ),
            "candidate_correlation": Gauge(
                "juniper_cascor_candidate_correlation",
                "Best candidate unit correlation with residual error",
            ),
            "inference_requests_total": Counter(
                "juniper_cascor_inference_requests_total",
                "Total inference requests processed",
            ),
            "inference_duration_seconds": Histogram(
                "juniper_cascor_inference_duration_seconds",
                "Inference latency in seconds",
                buckets=_LOGGER_PROMETHEUS_LATENCY_BUCKETS,
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


# ---------------------------------------------------------------------------
# WebSocket metrics — Phase 0-cascor (15 metrics)
# ---------------------------------------------------------------------------

_ws_metrics: dict | None = None

_WS_RESUME_REPLAY_BUCKETS = (0, 1, 5, 25, 100, 500, 1024)


def _ensure_ws_metrics() -> dict:
    """Create WebSocket-related Prometheus metrics on first access."""
    global _ws_metrics
    if _ws_metrics is None:
        from prometheus_client import Counter, Gauge, Histogram

        _ws_metrics = {
            "seq_current": Gauge(
                "cascor_ws_seq_current",
                "Current sequence number for WebSocket broadcasts",
            ),
            "replay_buffer_occupancy": Gauge(
                "cascor_ws_replay_buffer_occupancy",
                "Current number of messages in the replay buffer",
            ),
            "replay_buffer_bytes": Gauge(
                "cascor_ws_replay_buffer_bytes",
                "Approximate memory usage of the replay buffer in bytes",
            ),
            "replay_buffer_capacity_configured": Gauge(
                "cascor_ws_replay_buffer_capacity_configured",
                "Configured maximum replay buffer size",
            ),
            "resume_requests_total": Counter(
                "cascor_ws_resume_requests_total",
                "Total resume requests by outcome",
                ["outcome"],
            ),
            "resume_replayed_events": Histogram(
                "cascor_ws_resume_replayed_events",
                "Number of events replayed per successful resume",
                buckets=_WS_RESUME_REPLAY_BUCKETS,
            ),
            "broadcast_timeout_total": Counter(
                "cascor_ws_broadcast_timeout_total",
                "Total broadcast send timeouts",
                ["type"],
            ),
            "broadcast_send_duration_seconds": Histogram(
                "cascor_ws_broadcast_send_duration_seconds",
                "Duration of individual WebSocket send operations",
                ["type"],
            ),
            "pending_connections": Gauge(
                "cascor_ws_pending_connections",
                "Number of WebSocket connections in pending (resume handshake) state",
            ),
            "state_throttle_coalesced_total": Counter(
                "cascor_ws_state_throttle_coalesced_total",
                "Total state broadcasts coalesced by throttle",
            ),
            "broadcast_from_thread_errors_total": Counter(
                "cascor_ws_broadcast_from_thread_errors_total",
                "Total errors from broadcast_from_thread coroutine execution",
            ),
            "seq_gap_detected_total": Counter(
                "cascor_ws_seq_gap_detected_total",
                "Total sequence gaps detected (should be zero in healthy operation)",
            ),
            "connections_active": Gauge(
                "cascor_ws_connections_active",
                "Number of active WebSocket connections by endpoint",
                ["endpoint"],
            ),
            "command_responses_total": Counter(
                "cascor_ws_command_responses_total",
                "Total command responses sent",
                ["command", "status"],
            ),
            "command_handler_seconds": Histogram(
                "cascor_ws_command_handler_seconds",
                "Duration of command handler execution",
                ["command"],
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
