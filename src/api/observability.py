"""Observability module for structured logging, Prometheus metrics, and Sentry integration."""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

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

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        endpoint = request.url.path
        method = request.method
        status = str(response.status_code)

        self._request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self._request_duration.labels(method=method, endpoint=endpoint).observe(duration)

        return response


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

    handler = logging.StreamHandler()
    handler.setLevel(level)

    if log_format == "json":
        handler.setFormatter(JuniperJsonFormatter(service=service_name))
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    root.addHandler(handler)


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
        send_default_pii=True,
        enable_logs=True,
        traces_sample_rate=1.0,
        release=f"{service_name}@{version}",
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
                buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, float("inf")),
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
