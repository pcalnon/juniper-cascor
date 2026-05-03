"""Unit tests for the API observability module."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from api.observability import UNMATCHED_ENDPOINT_LABEL, JuniperJsonFormatter, PrometheusMiddleware, RequestIdMiddleware, configure_logging, configure_sentry, get_prometheus_app, record_training_epoch, request_id_var, set_build_info, set_training_loss


@pytest.mark.unit
class TestJuniperJsonFormatter:
    """Tests for JuniperJsonFormatter."""

    def test_format_produces_valid_json(self):
        formatter = JuniperJsonFormatter(service="test-service")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert parsed["service"] == "test-service"
        assert "timestamp" in parsed
        assert "request_id" in parsed

    def test_format_includes_request_id_from_contextvar(self):
        formatter = JuniperJsonFormatter(service="test-service")
        token = request_id_var.set("abc-123")
        try:
            record = logging.LogRecord(name="test", level=logging.INFO, pathname="", lineno=0, msg="hi", args=None, exc_info=None)
            output = formatter.format(record)
            parsed = json.loads(output)
            assert parsed["request_id"] == "abc-123"
        finally:
            request_id_var.reset(token)

    def test_format_includes_exception_info(self):
        formatter = JuniperJsonFormatter(service="test-service")
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
            record = logging.LogRecord(name="test", level=logging.ERROR, pathname="", lineno=0, msg="error", args=None, exc_info=exc_info)
            output = formatter.format(record)
            parsed = json.loads(output)
            assert "exception" in parsed
            assert "ValueError" in parsed["exception"]

    def test_format_default_service_name_is_shared_lib_default(self):
        """METRICS-MON R2.1.4: cascor consumes the shared formatter.

        Cascor used to default to ``"juniper-cascor"``; after migrating
        to ``juniper_observability.JuniperJsonFormatter``, the unset
        default is the shared lib's ``"juniper-service"``. All cascor
        call sites pass the service name explicitly (see
        ``configure_logging`` and ``api.app.lifespan``).
        """
        formatter = JuniperJsonFormatter()
        record = logging.LogRecord(name="test", level=logging.INFO, pathname="", lineno=0, msg="hi", args=None, exc_info=None)
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["service"] == "juniper-service"


@pytest.mark.unit
class TestConfigureLogging:
    """Tests for configure_logging function."""

    def setup_method(self):
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    def test_text_mode_uses_standard_formatter(self):
        configure_logging("INFO", "text", "test-service")
        root = logging.getLogger()
        assert len(root.handlers) == 2  # StreamHandler + RotatingFileHandler
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and type(h) is logging.StreamHandler]
        assert len(stream_handlers) == 1
        assert not isinstance(stream_handlers[0].formatter, JuniperJsonFormatter)

    def test_json_mode_uses_json_formatter(self):
        configure_logging("INFO", "json", "test-service")
        root = logging.getLogger()
        assert len(root.handlers) == 2  # StreamHandler + RotatingFileHandler
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and type(h) is logging.StreamHandler]
        assert len(stream_handlers) == 1
        assert isinstance(stream_handlers[0].formatter, JuniperJsonFormatter)

    def test_sets_log_level(self):
        configure_logging("DEBUG", "text", "test-service")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_removes_existing_handlers(self):
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        stream_handlers_before = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and type(h) is logging.StreamHandler]
        assert len(stream_handlers_before) == 2
        configure_logging("INFO", "text", "test-service")
        # configure_logging removes all handlers and adds StreamHandler + RotatingFileHandler
        assert len(root.handlers) == 2


@pytest.mark.unit
class TestConfigureSentry:
    """Tests for configure_sentry function."""

    def test_noop_when_dsn_is_none(self):
        configure_sentry(None, "test-service", "1.0.0")

    def test_noop_when_dsn_is_empty(self):
        configure_sentry("", "test-service", "1.0.0")

    def test_initializes_when_dsn_provided(self):
        with patch("sentry_sdk.init") as mock_init:
            configure_sentry("https://examplePublicKey@o0.ingest.sentry.io/0", "test-service", "1.0.0")
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["dsn"] == "https://examplePublicKey@o0.ingest.sentry.io/0"
            assert call_kwargs["release"] == "test-service@1.0.0"


@pytest.mark.unit
class TestRequestIdMiddleware:
    """Tests for RequestIdMiddleware."""

    @pytest.mark.asyncio
    async def test_generates_request_id_when_not_provided(self):
        middleware = RequestIdMiddleware(app=MagicMock())
        captured_rid = None

        async def mock_call_next(request):
            nonlocal captured_rid
            captured_rid = request_id_var.get("")
            response = MagicMock()
            response.headers = {}
            return response

        request = MagicMock()
        request.headers = {}

        response = await middleware.dispatch(request, mock_call_next)
        assert captured_rid != ""
        assert "X-Request-ID" in response.headers

    @pytest.mark.asyncio
    async def test_uses_provided_request_id(self):
        middleware = RequestIdMiddleware(app=MagicMock())
        captured_rid = None

        async def mock_call_next(request):
            nonlocal captured_rid
            captured_rid = request_id_var.get("")
            response = MagicMock()
            response.headers = {}
            return response

        request = MagicMock()
        request.headers = {"X-Request-ID": "custom-id-123"}

        response = await middleware.dispatch(request, mock_call_next)
        assert captured_rid == "custom-id-123"
        assert response.headers["X-Request-ID"] == "custom-id-123"


@pytest.mark.unit
class TestPrometheusMiddleware:
    """Tests for PrometheusMiddleware."""

    @staticmethod
    def _build_request(*, method: str, url_path: str, route_template: str | None) -> MagicMock:
        request = MagicMock()
        request.url.path = url_path
        request.method = method
        if route_template is None:
            request.scope = {}
        else:
            route = MagicMock()
            route.path = route_template
            request.scope = {"route": route}
        return request

    @pytest.mark.asyncio
    async def test_matched_route_uses_template_for_endpoint_label(self):
        """When Starlette resolves a route, the endpoint label is the template, not the raw URL."""
        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Histogram") as MockHistogram:
            request_count = MagicMock()
            unmatched_count = MagicMock()
            MockCounter.side_effect = [request_count, unmatched_count]
            mock_histogram = MagicMock()
            MockHistogram.return_value = mock_histogram

            middleware = PrometheusMiddleware(app=MagicMock(), service_name="test", namespace="juniper_cascor")

            response = MagicMock()
            response.status_code = 200

            async def mock_call_next(request):
                return response

            request = self._build_request(method="GET", url_path="/v1/training/abc-123", route_template="/v1/training/{run_id}")
            result = await middleware.dispatch(request, mock_call_next)

            request_count.labels.assert_called_once_with(method="GET", endpoint="/v1/training/{run_id}", status="200")
            request_count.labels().inc.assert_called_once()
            mock_histogram.labels.assert_called_once_with(method="GET", endpoint="/v1/training/{run_id}")
            mock_histogram.labels().observe.assert_called_once()
            unmatched_count.labels.assert_not_called()
            assert result == response

    @pytest.mark.asyncio
    async def test_unmatched_route_collapses_to_single_label(self):
        """No resolved route → endpoint label collapses to UNMATCHED_ENDPOINT_LABEL and unmatched counter increments."""
        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Histogram") as MockHistogram:
            request_count = MagicMock()
            unmatched_count = MagicMock()
            MockCounter.side_effect = [request_count, unmatched_count]
            MockHistogram.return_value = MagicMock()

            middleware = PrometheusMiddleware(app=MagicMock(), service_name="test", namespace="juniper_cascor")

            response = MagicMock()
            response.status_code = 404

            async def mock_call_next(request):
                return response

            request = self._build_request(method="GET", url_path="/totally/unknown/path", route_template=None)
            await middleware.dispatch(request, mock_call_next)

            request_count.labels.assert_called_once_with(method="GET", endpoint=UNMATCHED_ENDPOINT_LABEL, status="404")
            unmatched_count.labels.assert_called_once_with(method="GET")
            unmatched_count.labels().inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_cardinality_bounded_under_high_entropy_paths(self):
        """Sending N distinct unmatched URLs must still produce only one endpoint label value."""
        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Histogram") as MockHistogram:
            request_count = MagicMock()
            unmatched_count = MagicMock()
            MockCounter.side_effect = [request_count, unmatched_count]
            MockHistogram.return_value = MagicMock()

            middleware = PrometheusMiddleware(app=MagicMock(), service_name="test", namespace="juniper_cascor")

            response = MagicMock()
            response.status_code = 404

            async def mock_call_next(request):
                return response

            for i in range(50):
                request = self._build_request(method="GET", url_path=f"/attacker/{i}/abc", route_template=None)
                await middleware.dispatch(request, mock_call_next)

            distinct_endpoints = {call.kwargs["endpoint"] for call in request_count.labels.call_args_list}
            assert distinct_endpoints == {UNMATCHED_ENDPOINT_LABEL}, f"endpoint label cardinality leaked: {distinct_endpoints}"
            assert unmatched_count.labels.call_count == 50

    @pytest.mark.asyncio
    async def test_namespace_prefix_applied_to_metric_names(self):
        """Verify that the namespace parameter prefixes all three metric names."""
        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Histogram") as MockHistogram:
            MockCounter.return_value = MagicMock()
            MockHistogram.return_value = MagicMock()

            PrometheusMiddleware(app=MagicMock(), service_name="test", namespace="juniper_cascor")

            counter_names = [call.args[0] for call in MockCounter.call_args_list]
            assert "juniper_cascor_http_requests_total" in counter_names
            assert "juniper_cascor_http_unmatched_requests_total" in counter_names
            MockHistogram.assert_called_once_with(
                "juniper_cascor_http_request_duration_seconds",
                "HTTP request duration in seconds",
                ["method", "endpoint"],
            )


@pytest.mark.unit
class TestGetPrometheusApp:
    """Tests for get_prometheus_app function."""

    def test_returns_asgi_app(self):
        app = get_prometheus_app()
        assert callable(app)


@pytest.mark.unit
class TestSetBuildInfo:
    """Tests for set_build_info function."""

    def test_creates_info_metric(self):
        with patch("prometheus_client.Info") as MockInfo:
            mock_info = MagicMock()
            MockInfo.return_value = mock_info
            set_build_info("juniper_cascor", "0.4.0")
            MockInfo.assert_called_once_with("juniper_cascor_build", "Build information for juniper-cascor service")
            mock_info.info.assert_called_once()
            call_args = mock_info.info.call_args[0][0]
            assert call_args["version"] == "0.4.0"
            assert "python_version" in call_args


@pytest.mark.unit
class TestTrainingMetrics:
    """Tests for custom training metrics helpers."""

    def test_record_training_epoch(self):
        import api.observability as obs

        obs._training_metrics = None
        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Gauge"), patch("prometheus_client.Histogram"):
            mock_counter = MagicMock()
            MockCounter.return_value = mock_counter

            record_training_epoch("output")
            mock_counter.labels.assert_called_with(phase="output")
            mock_counter.labels().inc.assert_called_once()

        obs._training_metrics = None

    def test_set_training_loss(self):
        import api.observability as obs

        obs._training_metrics = None
        with patch("prometheus_client.Counter"), patch("prometheus_client.Gauge") as MockGauge, patch("prometheus_client.Histogram"):
            mock_gauge = MagicMock()
            MockGauge.return_value = mock_gauge

            set_training_loss("output", "train", 0.25)
            mock_gauge.labels.assert_called_with(phase="output", loss_type="train")
            mock_gauge.labels().set.assert_called_with(0.25)

        obs._training_metrics = None


@pytest.mark.unit
class TestObservabilityShim:
    """METRICS-MON R2.1.4: ``api.observability`` re-exports from the shared lib.

    These tests pin the migration: every cross-cutting symbol that
    historically lived inline must now resolve to the same object the
    shared :mod:`juniper_observability` package exposes. If a future
    change accidentally re-introduces a local copy, these assertions
    fail loudly.
    """

    def test_json_formatter_is_shared(self):
        import juniper_observability

        import api.observability as cascor_obs

        assert cascor_obs.JuniperJsonFormatter is juniper_observability.JuniperJsonFormatter

    def test_request_id_middleware_is_shared(self):
        import juniper_observability

        import api.observability as cascor_obs

        assert cascor_obs.RequestIdMiddleware is juniper_observability.RequestIdMiddleware
        assert cascor_obs.request_id_var is juniper_observability.request_id_var

    def test_prometheus_middleware_is_shared(self):
        import juniper_observability

        import api.observability as cascor_obs

        assert cascor_obs.PrometheusMiddleware is juniper_observability.PrometheusMiddleware
        assert cascor_obs.UNMATCHED_ENDPOINT_LABEL == juniper_observability.UNMATCHED_ENDPOINT_LABEL

    def test_prometheus_app_helpers_are_shared(self):
        import juniper_observability

        import api.observability as cascor_obs

        assert cascor_obs.get_prometheus_app is juniper_observability.get_prometheus_app
        assert cascor_obs.set_build_info is juniper_observability.set_build_info

    def test_strip_sensitive_headers_is_shared(self):
        from juniper_observability.sentry import _strip_sensitive_headers as shared

        import api.observability as cascor_obs

        assert cascor_obs._strip_sensitive_headers is shared

    def test_health_models_are_shared(self):
        import juniper_observability

        from api.models.health import DependencyStatus, ReadinessResponse, probe_dependency

        assert DependencyStatus is juniper_observability.DependencyStatus
        assert ReadinessResponse is juniper_observability.ReadinessResponse
        assert probe_dependency is juniper_observability.probe_dependency

    def test_route_constants_are_shared(self):
        import juniper_observability

        import api.routes.health as health_routes

        assert health_routes.LIVENESS_TICK_BUDGET_MS == juniper_observability.LIVENESS_TICK_BUDGET_MS
        assert health_routes.LIVENESS_STALENESS_SECONDS == juniper_observability.LIVENESS_STALENESS_SECONDS
        assert health_routes.READINESS_HEADER == juniper_observability.READINESS_HEADER


@pytest.mark.unit
class TestWebSocketHistogramBuckets:
    """METRICS-MON R5.1b: sub-millisecond bucket layout for WS latency histograms.

    Pins the bucket boundaries on
    ``cascor_ws_broadcast_send_duration_seconds`` and
    ``cascor_ws_command_handler_seconds`` against the
    ``_WS_SUB_MS_LATENCY_BUCKETS`` constant. If a future change
    accidentally reverts to the Prometheus default layout (5 ms floor)
    or alters the boundaries without updating the rationale doc, these
    assertions fail loudly. Rationale lives in
    ``notes/observability/HISTOGRAM_BUCKETS_RATIONALE_2026-05-02.md``
    §4 (broadcast_send) and §5 (command_handler).
    """

    EXPECTED_UPPER_BOUNDS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, float("inf")]

    def setup_method(self):
        # Force lazy re-init so each test sees a freshly-registered set
        # of metrics. The shim teardown also clears the registry below.
        import api.observability as obs

        obs._ws_metrics = None

    def teardown_method(self):
        from prometheus_client import REGISTRY

        import api.observability as obs

        if obs._ws_metrics is not None:
            for metric in list(obs._ws_metrics.values()):
                try:
                    REGISTRY.unregister(metric)
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "Ignoring metric unregister failure during teardown for %r: %s",
                        metric,
                        exc,
                    )
            obs._ws_metrics = None

    def test_sub_ms_bucket_constant_matches_rationale_doc(self):
        """The shared constant is the authoritative source; verify its shape."""
        from api.observability import _WS_SUB_MS_LATENCY_BUCKETS

        assert list(_WS_SUB_MS_LATENCY_BUCKETS) == self.EXPECTED_UPPER_BOUNDS

    def test_broadcast_send_duration_uses_sub_ms_buckets(self):
        from api.observability import _ensure_ws_metrics

        metrics = _ensure_ws_metrics()
        hist = metrics["broadcast_send_duration_seconds"]
        # ``_upper_bounds`` includes the implicit ``+inf`` upper edge.
        assert hist._upper_bounds == self.EXPECTED_UPPER_BOUNDS

    def test_command_handler_seconds_uses_sub_ms_buckets(self):
        from api.observability import _ensure_ws_metrics

        metrics = _ensure_ws_metrics()
        hist = metrics["command_handler_seconds"]
        assert hist._upper_bounds == self.EXPECTED_UPPER_BOUNDS

    def test_broadcast_send_help_string_no_longer_carries_r4_1_marker(self):
        """R5.1b removed the ``(R4.1 buckets tentative pending R5.1)`` suffix."""
        from api.observability import _ensure_ws_metrics

        metrics = _ensure_ws_metrics()
        hist = metrics["broadcast_send_duration_seconds"]
        assert "tentative pending R5.1" not in hist._documentation
        assert "R4.1" not in hist._documentation

    def test_command_handler_help_string_no_longer_carries_r4_1_marker(self):
        from api.observability import _ensure_ws_metrics

        metrics = _ensure_ws_metrics()
        hist = metrics["command_handler_seconds"]
        assert "tentative pending R5.1" not in hist._documentation
        assert "R4.1" not in hist._documentation


@pytest.mark.unit
class TestObservabilityShim_R5_1b:
    """Sanity-check that the **other** four histograms still carry the
    R4.1 ``(buckets tentative pending R5.1)`` suffix — those still await
    R5.1 SLO ratification, not re-bucketing. Guards against accidental
    HELP-string drift in future PRs.
    """

    def setup_method(self):
        import api.observability as obs

        obs._ws_metrics = None
        obs._training_metrics = None

    def teardown_method(self):
        from prometheus_client import REGISTRY

        import api.observability as obs

        for cache_attr in ("_ws_metrics", "_training_metrics"):
            cache = getattr(obs, cache_attr, None)
            if cache is not None:
                for metric in list(cache.values()):
                    try:
                        REGISTRY.unregister(metric)
                    except Exception as exc:
                        logging.debug("Best-effort metric unregister failed in teardown for %r: %s", metric, exc)
                setattr(obs, cache_attr, None)

    def test_inference_duration_seconds_help_keeps_r4_1_marker(self):
        from api.observability import _ensure_training_metrics

        metrics = _ensure_training_metrics()
        hist = metrics["inference_duration_seconds"]
        assert "(R4.1 buckets tentative pending R5.1)" in hist._documentation

    def test_resume_replayed_events_help_keeps_r4_1_marker(self):
        from api.observability import _ensure_ws_metrics

        metrics = _ensure_ws_metrics()
        hist = metrics["resume_replayed_events"]
        assert "(R4.1 buckets tentative pending R5.1)" in hist._documentation


@pytest.mark.unit
class TestObservabilityReadinessTimestamp:
    def test_readiness_timestamp_is_tz_aware_utc(self):
        """METRICS-MON R2.1.4: closes BUG-JD-06-equivalent naive-tz drift.

        Cascor's former ``ReadinessResponse.timestamp`` defaulted to
        ``datetime.now().timestamp()`` (locale-dependent). The shared
        model uses ``datetime.now(UTC).timestamp()`` so all services emit
        the same epoch-seconds value regardless of the host timezone.
        """
        import time

        from juniper_observability import ReadinessResponse

        rr = ReadinessResponse(status="ready", version="0.4.0", service="juniper-cascor")
        # tz-aware UTC unix timestamp must be within 60s of "now".
        assert abs(time.time() - rr.timestamp) < 60.0
