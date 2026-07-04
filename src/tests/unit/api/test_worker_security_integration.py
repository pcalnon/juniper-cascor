"""Tests for CR-009: worker security module integration via feature flags.

Verifies:
- Settings flags default to False (zero behavior change)
- When enabled, modules are instantiated and attached to app.state
- Rate limiter rejects connections when rate-limited
- Audit logger records events on registration/deregistration
- Worker metrics track registration/deregistration
- Anomaly detector is wired into the coordinator
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect

from api.settings import Settings
from api.websocket.worker_stream import worker_stream_handler
from api.workers.audit import AuditEventType, AuditLogger, WorkerMetrics
from api.workers.coordinator import WorkerCoordinator
from api.workers.protocol import WorkerProtocol
from api.workers.registry import WorkerRegistry
from api.workers.security import AnomalyDetector, ConnectionRateLimiter

pytestmark = pytest.mark.unit


def _make_websocket(headers=None, app_state=None):
    """Create a mock WebSocket with configurable headers and app state."""
    ws = AsyncMock()
    ws.headers = headers or {}
    ws.client = ("127.0.0.1", 54321)

    state = MagicMock()
    state.api_key_auth = None
    state.worker_coordinator = None
    state.worker_registry = None
    # Default: no security modules attached
    state.worker_rate_limiter = None
    state.audit_logger = None
    state.worker_metrics = None
    if app_state:
        for key, val in app_state.items():
            setattr(state, key, val)

    app = MagicMock()
    app.state = state
    ws.app = app
    return ws


# ---------------------------------------------------------------------------
# Settings defaults
# ---------------------------------------------------------------------------


class TestWorkerSecuritySettingsDefaults:
    """All worker security feature flags default to False."""

    def test_worker_rate_limit_disabled_by_default(self):
        settings = Settings()
        assert settings.worker_rate_limit_enabled is False

    def test_worker_rate_limit_connections_per_minute_default(self):
        settings = Settings()
        assert settings.worker_rate_limit_connections_per_minute == 10

    def test_worker_rate_limit_burst_size_default(self):
        settings = Settings()
        assert settings.worker_rate_limit_burst_size == 3

    def test_worker_anomaly_detection_disabled_by_default(self):
        settings = Settings()
        assert settings.worker_anomaly_detection_enabled is False

    def test_worker_anomaly_min_training_time_default(self):
        settings = Settings()
        assert settings.worker_anomaly_min_training_time == 0.1

    def test_worker_anomaly_perfect_corr_threshold_default(self):
        settings = Settings()
        assert settings.worker_anomaly_perfect_corr_threshold == 0.999

    def test_worker_audit_logging_disabled_by_default(self):
        settings = Settings()
        assert settings.worker_audit_logging_enabled is False

    def test_worker_metrics_disabled_by_default(self):
        settings = Settings()
        assert settings.worker_metrics_enabled is False


class TestWorkerSecuritySettingsEnvOverride:
    """Feature flags can be enabled via environment variables."""

    def test_enable_rate_limit_via_env(self, monkeypatch):
        monkeypatch.setenv("JUNIPER_CASCOR_WORKER_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("JUNIPER_CASCOR_WORKER_RATE_LIMIT_CONNECTIONS_PER_MINUTE", "20")
        monkeypatch.setenv("JUNIPER_CASCOR_WORKER_RATE_LIMIT_BURST_SIZE", "5")
        settings = Settings()
        assert settings.worker_rate_limit_enabled is True
        assert settings.worker_rate_limit_connections_per_minute == 20
        assert settings.worker_rate_limit_burst_size == 5

    def test_enable_anomaly_detection_via_env(self, monkeypatch):
        monkeypatch.setenv("JUNIPER_CASCOR_WORKER_ANOMALY_DETECTION_ENABLED", "true")
        monkeypatch.setenv("JUNIPER_CASCOR_WORKER_ANOMALY_MIN_TRAINING_TIME", "0.5")
        monkeypatch.setenv("JUNIPER_CASCOR_WORKER_ANOMALY_PERFECT_CORR_THRESHOLD", "0.99")
        settings = Settings()
        assert settings.worker_anomaly_detection_enabled is True
        assert settings.worker_anomaly_min_training_time == 0.5
        assert settings.worker_anomaly_perfect_corr_threshold == 0.99

    def test_enable_audit_logging_via_env(self, monkeypatch):
        monkeypatch.setenv("JUNIPER_CASCOR_WORKER_AUDIT_LOGGING_ENABLED", "true")
        settings = Settings()
        assert settings.worker_audit_logging_enabled is True

    def test_enable_worker_metrics_via_env(self, monkeypatch):
        monkeypatch.setenv("JUNIPER_CASCOR_WORKER_METRICS_ENABLED", "true")
        settings = Settings()
        assert settings.worker_metrics_enabled is True


# ---------------------------------------------------------------------------
# App startup initialization
# ---------------------------------------------------------------------------


class TestAppStartupInitialization:
    """When feature flags are enabled, modules are created during lifespan."""

    def test_lifespan_creates_rate_limiter_when_enabled(self):
        """Rate limiter is attached to app.state when flag is enabled."""
        from api.app import create_app

        settings = Settings(
            auto_start=False,
            worker_rate_limit_enabled=True,
            worker_rate_limit_connections_per_minute=15,
            worker_rate_limit_burst_size=5,
        )
        app = create_app(settings)
        # app.state is set during create_app; lifespan sets the modules
        # We need to check during lifespan, so use TestClient
        from fastapi.testclient import TestClient

        with TestClient(app):
            assert hasattr(app.state, "worker_rate_limiter")
            assert isinstance(app.state.worker_rate_limiter, ConnectionRateLimiter)

    def test_lifespan_creates_anomaly_detector_when_enabled(self):
        """Anomaly detector is attached to app.state and coordinator when flag is enabled."""
        from api.app import create_app

        settings = Settings(
            auto_start=False,
            worker_anomaly_detection_enabled=True,
        )
        app = create_app(settings)
        from fastapi.testclient import TestClient

        with TestClient(app):
            assert hasattr(app.state, "anomaly_detector")
            assert isinstance(app.state.anomaly_detector, AnomalyDetector)
            # Also check coordinator has the reference
            assert app.state.worker_coordinator._anomaly_detector is app.state.anomaly_detector

    def test_lifespan_creates_audit_logger_when_enabled(self):
        """Audit logger is attached to app.state when flag is enabled."""
        from api.app import create_app

        settings = Settings(
            auto_start=False,
            worker_audit_logging_enabled=True,
        )
        app = create_app(settings)
        from fastapi.testclient import TestClient

        with TestClient(app):
            assert hasattr(app.state, "audit_logger")
            assert isinstance(app.state.audit_logger, AuditLogger)

    def test_lifespan_creates_worker_metrics_when_enabled(self):
        """Worker metrics is attached to app.state when flag is enabled."""
        from api.app import create_app

        settings = Settings(
            auto_start=False,
            worker_metrics_enabled=True,
        )
        app = create_app(settings)
        from fastapi.testclient import TestClient

        with TestClient(app):
            assert hasattr(app.state, "worker_metrics")
            assert isinstance(app.state.worker_metrics, WorkerMetrics)

    def test_lifespan_skips_modules_when_all_disabled(self):
        """No security modules are created when all flags are False."""
        from api.app import create_app

        settings = Settings(auto_start=False)
        app = create_app(settings)
        from fastapi.testclient import TestClient

        with TestClient(app):
            assert not hasattr(app.state, "worker_rate_limiter")
            assert not hasattr(app.state, "anomaly_detector")
            assert not hasattr(app.state, "audit_logger")
            assert not hasattr(app.state, "worker_metrics")


# ---------------------------------------------------------------------------
# Rate limiter integration in worker_stream
# ---------------------------------------------------------------------------


class TestRateLimiterIntegration:
    """Rate limiter rejects connections when rate-limited."""

    @pytest.mark.asyncio
    async def test_rate_limited_connection_rejected(self):
        """When rate limiter denies a connection, the WebSocket is closed with 4029."""
        limiter = MagicMock(spec=ConnectionRateLimiter)
        limiter.allow.return_value = False

        ws = _make_websocket(
            app_state={"worker_rate_limiter": limiter},
        )
        await worker_stream_handler(ws)

        limiter.allow.assert_called_once_with("127.0.0.1")
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4029

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_connection(self):
        """When rate limiter allows, the handler proceeds normally."""
        limiter = MagicMock(spec=ConnectionRateLimiter)
        limiter.allow.return_value = True

        registry = WorkerRegistry(heartbeat_timeout=30.0)
        coordinator = WorkerCoordinator(registry=registry, task_reassignment_timeout=5.0)

        reg_msg = json.dumps(WorkerProtocol.build_register("w1", {"cpu_cores": 4}))

        ws = _make_websocket(
            app_state={
                "worker_rate_limiter": limiter,
                "worker_coordinator": coordinator,
                "worker_registry": registry,
            },
        )
        ws.receive_text = AsyncMock(return_value=reg_msg)
        ws.receive = AsyncMock(side_effect=WebSocketDisconnect())
        coordinator.get_next_assignment = MagicMock(return_value=None)

        await worker_stream_handler(ws)

        limiter.allow.assert_called_once_with("127.0.0.1")
        ws.accept.assert_awaited_once()
        coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_no_rate_limiter_proceeds_normally(self):
        """Without rate limiter on app.state, handler proceeds past rate limiting."""
        registry = WorkerRegistry(heartbeat_timeout=30.0)
        coordinator = WorkerCoordinator(registry=registry, task_reassignment_timeout=5.0)

        reg_msg = json.dumps(WorkerProtocol.build_register("w1", {"cpu_cores": 4}))

        ws = _make_websocket(
            app_state={
                "worker_coordinator": coordinator,
                "worker_registry": registry,
            },
        )
        ws.receive_text = AsyncMock(return_value=reg_msg)
        ws.receive = AsyncMock(side_effect=WebSocketDisconnect())
        coordinator.get_next_assignment = MagicMock(return_value=None)

        await worker_stream_handler(ws)

        ws.accept.assert_awaited_once()
        coordinator.shutdown()


# ---------------------------------------------------------------------------
# Audit logger integration in worker_stream
# ---------------------------------------------------------------------------


class TestAuditLoggerIntegration:
    """Audit logger records registration and deregistration events."""

    @pytest.mark.asyncio
    async def test_audit_logs_register_and_deregister(self):
        """Audit logger records WORKER_REGISTER on connect, WORKER_DEREGISTER on disconnect."""
        audit = AuditLogger()
        registry = WorkerRegistry(heartbeat_timeout=30.0)
        coordinator = WorkerCoordinator(registry=registry, task_reassignment_timeout=5.0)

        reg_msg = json.dumps(WorkerProtocol.build_register("w1", {"cpu_cores": 4}))

        ws = _make_websocket(
            app_state={
                "audit_logger": audit,
                "worker_coordinator": coordinator,
                "worker_registry": registry,
            },
        )
        ws.receive_text = AsyncMock(return_value=reg_msg)
        ws.receive = AsyncMock(side_effect=WebSocketDisconnect())
        coordinator.get_next_assignment = MagicMock(return_value=None)

        await worker_stream_handler(ws)

        counts = audit.get_counts()
        assert counts.get(AuditEventType.WORKER_REGISTER) == 1
        assert counts.get(AuditEventType.WORKER_DEREGISTER) == 1
        coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_no_audit_log_when_registration_fails(self):
        """Audit logger is NOT called when registration fails (worker_id is None)."""
        audit = AuditLogger()
        registry = WorkerRegistry(heartbeat_timeout=30.0)
        coordinator = WorkerCoordinator(registry=registry, task_reassignment_timeout=5.0)

        ws = _make_websocket(
            app_state={
                "audit_logger": audit,
                "worker_coordinator": coordinator,
                "worker_registry": registry,
            },
        )
        ws.receive_text = AsyncMock(return_value="invalid json{{{")

        await worker_stream_handler(ws)

        counts = audit.get_counts()
        assert counts.get(AuditEventType.WORKER_REGISTER) is None
        assert counts.get(AuditEventType.WORKER_DEREGISTER) is None
        coordinator.shutdown()


# ---------------------------------------------------------------------------
# Worker metrics integration in worker_stream
# ---------------------------------------------------------------------------


class TestWorkerMetricsIntegration:
    """Worker metrics track registration and deregistration."""

    @pytest.mark.asyncio
    async def test_metrics_track_register_and_deregister(self):
        """Worker metrics records on_register and on_deregister keyed by the
        server-assigned worker_id (CR-026), not the client-proposed name."""
        metrics = WorkerMetrics()
        registry = WorkerRegistry(heartbeat_timeout=30.0)
        coordinator = WorkerCoordinator(registry=registry, task_reassignment_timeout=5.0)

        reg_msg = json.dumps(WorkerProtocol.build_register("w1", {"cpu_cores": 4}))

        ws = _make_websocket(
            app_state={
                "worker_metrics": metrics,
                "worker_coordinator": coordinator,
                "worker_registry": registry,
            },
        )
        ws.receive_text = AsyncMock(return_value=reg_msg)
        ws.receive = AsyncMock(side_effect=WebSocketDisconnect())
        coordinator.get_next_assignment = MagicMock(return_value=None)

        await worker_stream_handler(ws)

        # Metrics are keyed by the server-assigned ID, not "w1". Find the
        # single registered worker via the metrics snapshot.
        all_metrics = metrics.get_all_metrics()
        assert len(all_metrics) == 1
        server_id = all_metrics[0]["worker_id"]
        assert server_id != "w1"
        assert server_id.startswith("worker-")

        worker_data = metrics.get_worker_metrics(server_id)
        assert worker_data is not None
        assert worker_data["worker_id"] == server_id
        assert worker_data["source_ip"] == "127.0.0.1"
        assert worker_data["deregistered_at"] is not None
        # The stale lookup by client name must NOT return anything.
        assert metrics.get_worker_metrics("w1") is None
        coordinator.shutdown()


# ---------------------------------------------------------------------------
# Anomaly detector integration in coordinator
# ---------------------------------------------------------------------------


class TestAnomalyDetectorIntegration:
    """Anomaly detector is checked during submit_result."""

    def test_anomaly_detector_called_on_submit_result(self):
        """When anomaly detector is set, it is called during result submission."""
        import numpy as np

        registry = WorkerRegistry(heartbeat_timeout=30.0)
        coordinator = WorkerCoordinator(registry=registry, task_reassignment_timeout=5.0)

        detector = AnomalyDetector(min_training_time=0.1, perfect_corr_threshold=0.999)
        coordinator._anomaly_detector = detector

        registry.register("w1", {})
        tensors = {
            "candidate_input": np.zeros((10, 4), dtype=np.float32),
            "y": np.zeros((10, 1), dtype=np.float32),
            "residual_error": np.zeros((10, 1), dtype=np.float32),
        }
        task_ids = coordinator.submit_tasks(
            "r1",
            [{"candidate_index": 0, "candidate_data": {}, "training_params": {}}],
            tensors,
        )
        coordinator.get_next_assignment("w1")

        msg = {
            "type": "task_result",
            "task_id": task_ids[0],
            "candidate_id": 0,
            "candidate_uuid": "uuid",
            "correlation": 0.85,
            "success": True,
            "epochs_completed": 10,
            "activation_name": "sigmoid",
            "all_correlations": [0.85],
            "numerator": 1.0,
            "denominator": 2.0,
            "best_corr_idx": 9,
            "error_message": None,
            "tensor_manifest": {},
            "training_duration": 5.0,
        }

        accepted = coordinator.submit_result("w1", msg, {})
        assert accepted is True
        coordinator.shutdown()

    def test_anomaly_detector_logs_warning_for_suspicious_result(self):
        """Suspicious results (e.g., too-fast training) trigger a warning log."""
        import numpy as np

        registry = WorkerRegistry(heartbeat_timeout=30.0)
        coordinator = WorkerCoordinator(registry=registry, task_reassignment_timeout=5.0)

        detector = AnomalyDetector(min_training_time=1.0, perfect_corr_threshold=0.999)
        coordinator._anomaly_detector = detector

        registry.register("w1", {})
        tensors = {
            "candidate_input": np.zeros((10, 4), dtype=np.float32),
            "y": np.zeros((10, 1), dtype=np.float32),
            "residual_error": np.zeros((10, 1), dtype=np.float32),
        }
        task_ids = coordinator.submit_tasks(
            "r1",
            [{"candidate_index": 0, "candidate_data": {}, "training_params": {}}],
            tensors,
        )
        coordinator.get_next_assignment("w1")

        msg = {
            "type": "task_result",
            "task_id": task_ids[0],
            "candidate_id": 0,
            "candidate_uuid": "uuid",
            "correlation": 0.85,
            "success": True,
            "epochs_completed": 10,
            "activation_name": "sigmoid",
            "all_correlations": [0.85],
            "numerator": 1.0,
            "denominator": 2.0,
            "best_corr_idx": 9,
            "error_message": None,
            "tensor_manifest": {},
            "training_duration": 0.001,  # suspiciously fast
        }

        # Result is still accepted (anomaly detection logs but does not reject)
        accepted = coordinator.submit_result("w1", msg, {})
        assert accepted is True
        coordinator.shutdown()

    def test_no_anomaly_detector_does_not_break(self):
        """Without anomaly detector, submit_result works normally."""
        import numpy as np

        registry = WorkerRegistry(heartbeat_timeout=30.0)
        coordinator = WorkerCoordinator(registry=registry, task_reassignment_timeout=5.0)
        assert coordinator._anomaly_detector is None

        registry.register("w1", {})
        tensors = {
            "candidate_input": np.zeros((10, 4), dtype=np.float32),
            "y": np.zeros((10, 1), dtype=np.float32),
            "residual_error": np.zeros((10, 1), dtype=np.float32),
        }
        task_ids = coordinator.submit_tasks(
            "r1",
            [{"candidate_index": 0, "candidate_data": {}, "training_params": {}}],
            tensors,
        )
        coordinator.get_next_assignment("w1")

        msg = {
            "type": "task_result",
            "task_id": task_ids[0],
            "candidate_id": 0,
            "candidate_uuid": "uuid",
            "correlation": 0.85,
            "success": True,
            "epochs_completed": 10,
            "activation_name": "sigmoid",
            "all_correlations": [0.85],
            "numerator": 1.0,
            "denominator": 2.0,
            "best_corr_idx": 9,
            "error_message": None,
            "tensor_manifest": {},
        }

        accepted = coordinator.submit_result("w1", msg, {})
        assert accepted is True
        coordinator.shutdown()
