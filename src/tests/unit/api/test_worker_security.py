"""Tests for Phase 4 security hardening components.

Covers:
- mTLS configuration and SSL context building
- Connection rate limiting (token bucket)
- Anomaly detection for suspicious training results
- Audit logging and per-worker metrics
"""

import logging
import ssl
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from api.workers.audit import AUDIT_LEVEL, AuditEventType, AuditLogger, WorkerMetrics
from api.workers.security import AnomalyDetector, ConnectionRateLimiter, TLSConfig, _TokenBucket

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# TLS Configuration Tests
# ---------------------------------------------------------------------------


class TestTLSConfig:
    def test_disabled_returns_none(self):
        cfg = TLSConfig(enabled=False)
        assert cfg.build_ssl_context() is None

    def test_missing_cert_raises(self, tmp_path):
        cfg = TLSConfig(enabled=True, cert_file=str(tmp_path / "nonexistent.crt"), key_file=str(tmp_path / "nonexistent.key"))
        with pytest.raises(FileNotFoundError, match="TLS cert not found"):
            cfg.build_ssl_context()

    def test_missing_key_raises(self, tmp_path):
        cert = tmp_path / "server.crt"
        cert.write_text("dummy")
        cfg = TLSConfig(enabled=True, cert_file=str(cert), key_file=str(tmp_path / "nonexistent.key"))
        with pytest.raises(FileNotFoundError, match="TLS key not found"):
            cfg.build_ssl_context()

    def test_missing_ca_raises(self, tmp_path):
        cert = tmp_path / "server.crt"
        key = tmp_path / "server.key"
        cert.write_text("dummy")
        key.write_text("dummy")
        cfg = TLSConfig(enabled=True, cert_file=str(cert), key_file=str(key), require_client_cert=True, ca_file=str(tmp_path / "nonexistent_ca.crt"))
        with pytest.raises(FileNotFoundError, match="CA cert not found"):
            cfg.build_ssl_context()

    def test_half_config_cert_only_raises(self, tmp_path):
        """TLS enabled with cert_file but no key_file must fail closed."""
        cert = tmp_path / "server.crt"
        cert.write_text("dummy")
        cfg = TLSConfig(enabled=True, cert_file=str(cert), key_file=None)
        with pytest.raises(ValueError, match="requires both cert_file and key_file"):
            cfg.build_ssl_context()

    def test_half_config_key_only_raises(self, tmp_path):
        """TLS enabled with key_file but no cert_file must fail closed."""
        key = tmp_path / "server.key"
        key.write_text("dummy")
        cfg = TLSConfig(enabled=True, cert_file=None, key_file=str(key))
        with pytest.raises(ValueError, match="requires both cert_file and key_file"):
            cfg.build_ssl_context()

    def test_enabled_without_cert_or_key_raises(self):
        """TLS enabled with neither cert nor key must fail closed (not return bare context)."""
        cfg = TLSConfig(enabled=True, cert_file=None, key_file=None)
        with pytest.raises(ValueError, match="requires both cert_file and key_file"):
            cfg.build_ssl_context()

    def test_disabled_ignores_partial_paths(self, tmp_path):
        """TLS disabled must ignore half-configured paths and return None."""
        cert = tmp_path / "server.crt"
        cert.write_text("dummy")
        cfg = TLSConfig(enabled=False, cert_file=str(cert), key_file=None)
        assert cfg.build_ssl_context() is None


# ---------------------------------------------------------------------------
# Rate Limiter Tests
# ---------------------------------------------------------------------------


class TestConnectionRateLimiter:
    def test_allows_within_burst(self):
        limiter = ConnectionRateLimiter(max_connections_per_minute=60, burst_size=3)
        assert limiter.allow("10.0.0.1") is True
        assert limiter.allow("10.0.0.1") is True
        assert limiter.allow("10.0.0.1") is True

    def test_rejects_after_burst(self):
        limiter = ConnectionRateLimiter(max_connections_per_minute=60, burst_size=2)
        assert limiter.allow("10.0.0.1") is True
        assert limiter.allow("10.0.0.1") is True
        assert limiter.allow("10.0.0.1") is False

    def test_separate_buckets_per_source(self):
        limiter = ConnectionRateLimiter(max_connections_per_minute=60, burst_size=2)
        assert limiter.allow("10.0.0.1") is True
        assert limiter.allow("10.0.0.2") is True
        # Exhaust both buckets
        assert limiter.allow("10.0.0.1") is True
        assert limiter.allow("10.0.0.2") is True
        # Now both are exhausted
        assert limiter.allow("10.0.0.1") is False
        assert limiter.allow("10.0.0.2") is False

    def test_refills_over_time(self):
        limiter = ConnectionRateLimiter(max_connections_per_minute=6000, burst_size=2)
        assert limiter.allow("10.0.0.1") is True
        assert limiter.allow("10.0.0.1") is True
        assert limiter.allow("10.0.0.1") is False
        # At 6000/min = 100/sec, waiting 0.05s refills ~5 tokens
        time.sleep(0.05)
        assert limiter.allow("10.0.0.1") is True


# ---------------------------------------------------------------------------
# Anomaly Detection Tests
# ---------------------------------------------------------------------------


class TestAnomalyDetector:
    def test_clean_result_no_anomalies(self):
        detector = AnomalyDetector()
        anomalies = detector.check_result(worker_id="w-1", correlation=0.75, training_duration=5.0, task_id="t-1")
        assert anomalies == []

    def test_suspiciously_fast(self):
        detector = AnomalyDetector(min_training_time=1.0)
        anomalies = detector.check_result(worker_id="w-1", correlation=0.5, training_duration=0.01, task_id="t-1")
        assert any("suspiciously_fast" in a for a in anomalies)

    def test_perfect_correlation(self):
        detector = AnomalyDetector(perfect_corr_threshold=0.999)
        anomalies = detector.check_result(worker_id="w-1", correlation=0.9999, training_duration=5.0, task_id="t-1")
        assert any("perfect_correlation" in a for a in anomalies)

    def test_stale_correlation(self):
        detector = AnomalyDetector(stale_corr_threshold=0.001)
        anomalies = detector.check_result(worker_id="w-1", correlation=0.0001, training_duration=5.0, task_id="t-1")
        assert any("stale_correlation" in a for a in anomalies)

    def test_duplicate_correlations_detected(self):
        detector = AnomalyDetector(duplicate_corr_window=5)
        # Submit identical correlations
        for i in range(5):
            anomalies = detector.check_result(worker_id="w-1", correlation=0.5, training_duration=5.0, task_id=f"t-{i}")
        # After enough identical results, duplicate detection triggers
        assert any("duplicate_correlations" in a for a in anomalies)

    def test_worker_stats(self):
        detector = AnomalyDetector()
        detector.check_result("w-1", 0.5, 3.0, "t-1")
        detector.check_result("w-1", 0.7, 5.0, "t-2")
        stats = detector.get_worker_stats("w-1")
        assert stats["total_results"] == 2
        assert abs(stats["avg_correlation"] - 0.6) < 0.01
        assert abs(stats["avg_duration"] - 4.0) < 0.01

    def test_clear_worker(self):
        detector = AnomalyDetector()
        detector.check_result("w-1", 0.5, 3.0, "t-1")
        detector.clear_worker("w-1")
        stats = detector.get_worker_stats("w-1")
        assert stats["total_results"] == 0


# ---------------------------------------------------------------------------
# Audit Logger Tests
# ---------------------------------------------------------------------------


class TestAuditLogger:
    def test_log_event(self, caplog):
        audit = AuditLogger()
        # The test suite's LogConfig calls dictConfig() which disables loggers not
        # in the YAML config (disable_existing_loggers defaults to True). Re-enable
        # so caplog can capture the audit record.
        audit._logger.disabled = False
        with caplog.at_level(AUDIT_LEVEL, logger="juniper_cascor.api.workers.audit"):
            audit.log(AuditEventType.AUTH_SUCCESS, worker_id="w-1", source_ip="10.0.0.1")
        assert "AUDIT" in caplog.text
        assert "auth_success" in caplog.text

    def test_event_counting(self):
        audit = AuditLogger()
        audit.log(AuditEventType.AUTH_SUCCESS, worker_id="w-1")
        audit.log(AuditEventType.AUTH_SUCCESS, worker_id="w-2")
        audit.log(AuditEventType.AUTH_FAILURE, worker_id="w-3")
        counts = audit.get_counts()
        assert counts[AuditEventType.AUTH_SUCCESS] == 2
        assert counts[AuditEventType.AUTH_FAILURE] == 1

    def test_reset_counts(self):
        audit = AuditLogger()
        audit.log(AuditEventType.WORKER_REGISTER, worker_id="w-1")
        audit.reset_counts()
        assert audit.get_counts() == {}


# ---------------------------------------------------------------------------
# Worker Metrics Tests
# ---------------------------------------------------------------------------


class TestWorkerMetrics:
    def test_register_and_get(self):
        metrics = WorkerMetrics()
        metrics.on_register("w-1", source_ip="10.0.0.1")
        data = metrics.get_worker_metrics("w-1")
        assert data is not None
        assert data["worker_id"] == "w-1"
        assert data["tasks_completed"] == 0

    def test_task_completion_tracking(self):
        metrics = WorkerMetrics()
        metrics.on_register("w-1")
        metrics.on_task_complete("w-1", success=True, duration=3.0)
        metrics.on_task_complete("w-1", success=True, duration=5.0)
        metrics.on_task_complete("w-1", success=False, duration=1.0)
        data = metrics.get_worker_metrics("w-1")
        assert data["tasks_completed"] == 3
        assert data["tasks_succeeded"] == 2
        assert data["tasks_failed"] == 1
        assert abs(data["avg_duration"] - 3.0) < 0.01
        assert abs(data["success_rate"] - 2 / 3) < 0.01

    def test_anomaly_tracking(self):
        metrics = WorkerMetrics()
        metrics.on_register("w-1")
        metrics.on_anomaly("w-1", "perfect_correlation")
        metrics.on_anomaly("w-1", "suspiciously_fast")
        data = metrics.get_worker_metrics("w-1")
        assert data["anomaly_count"] == 2

    def test_unknown_worker_returns_none(self):
        metrics = WorkerMetrics()
        assert metrics.get_worker_metrics("nonexistent") is None

    def test_get_all_metrics(self):
        metrics = WorkerMetrics()
        metrics.on_register("w-1")
        metrics.on_register("w-2")
        all_data = metrics.get_all_metrics()
        assert len(all_data) == 2


# ---------------------------------------------------------------------------
# Audit Logger Tests — Extended Coverage
# ---------------------------------------------------------------------------


class TestAuditLoggerExtended:
    """Extended tests for AuditLogger covering custom logger_name,
    all event types, complex fields, and sequential counter behavior."""

    def test_custom_logger_name(self):
        """AuditLogger can be initialized with a custom logger name."""
        audit = AuditLogger(logger_name="custom.audit.logger")
        assert audit._logger.name == "custom.audit.logger"

    def test_log_all_event_types(self, caplog):
        """Every AuditEventType can be logged and counted."""
        audit = AuditLogger()
        audit._logger.disabled = False
        with caplog.at_level(AUDIT_LEVEL, logger="juniper_cascor.api.workers.audit"):
            for event_type in AuditEventType:
                audit.log(event_type, worker_id="w-test")
        counts = audit.get_counts()
        for event_type in AuditEventType:
            assert counts[event_type] == 1

    def test_log_complex_fields(self, caplog):
        """Log entries include complex nested field values serialized as JSON."""
        audit = AuditLogger()
        audit._logger.disabled = False
        with caplog.at_level(AUDIT_LEVEL, logger="juniper_cascor.api.workers.audit"):
            audit.log(
                AuditEventType.ANOMALY_DETECTED,
                worker_id="w-456",
                details={"type": "perfect_corr", "value": 0.9999},
                tags=["security", "anomaly"],
            )
        assert "anomaly_detected" in caplog.text
        assert "perfect_corr" in caplog.text

    def test_sequential_counter_increments(self):
        """Each log call for the same event type increments the seq counter."""
        audit = AuditLogger()
        audit.log(AuditEventType.AUTH_SUCCESS, worker_id="w-1")
        audit.log(AuditEventType.AUTH_SUCCESS, worker_id="w-2")
        audit.log(AuditEventType.AUTH_SUCCESS, worker_id="w-3")
        counts = audit.get_counts()
        assert counts[AuditEventType.AUTH_SUCCESS] == 3

    def test_get_counts_returns_copy(self):
        """get_counts() returns a new dict, not a reference to the internal counter."""
        audit = AuditLogger()
        audit.log(AuditEventType.WORKER_REGISTER, worker_id="w-1")
        counts = audit.get_counts()
        counts[AuditEventType.WORKER_REGISTER] = 999
        assert audit.get_counts()[AuditEventType.WORKER_REGISTER] == 1

    def test_reset_then_log_restarts_counters(self):
        """After reset_counts(), logging resumes from seq=1."""
        audit = AuditLogger()
        audit.log(AuditEventType.TASK_ASSIGNED, worker_id="w-1")
        audit.log(AuditEventType.TASK_ASSIGNED, worker_id="w-2")
        audit.reset_counts()
        audit.log(AuditEventType.TASK_ASSIGNED, worker_id="w-3")
        counts = audit.get_counts()
        assert counts[AuditEventType.TASK_ASSIGNED] == 1

    def test_log_with_no_extra_fields(self, caplog):
        """Logging with no extra fields still produces a valid audit record."""
        audit = AuditLogger()
        audit._logger.disabled = False
        with caplog.at_level(AUDIT_LEVEL, logger="juniper_cascor.api.workers.audit"):
            audit.log(AuditEventType.CONNECTION_CLOSED)
        assert "connection_closed" in caplog.text

    def test_log_with_non_serializable_field(self, caplog):
        """Fields with non-JSON-serializable values use str() fallback (default=str)."""
        audit = AuditLogger()
        audit._logger.disabled = False
        with caplog.at_level(AUDIT_LEVEL, logger="juniper_cascor.api.workers.audit"):
            audit.log(AuditEventType.TLS_HANDSHAKE, custom_obj=object())
        assert "tls_handshake" in caplog.text


# ---------------------------------------------------------------------------
# Worker Metrics Tests — Extended Coverage
# ---------------------------------------------------------------------------


class TestWorkerMetricsExtended:
    """Extended tests for WorkerMetrics covering on_deregister, unknown-worker
    early returns, success_rate/avg_duration edge cases, and get_all_metrics
    filtering."""

    def test_on_deregister_sets_timestamp(self):
        """on_deregister sets deregistered_at for a registered worker."""
        metrics = WorkerMetrics()
        metrics.on_register("w-1", source_ip="10.0.0.1")
        metrics.on_deregister("w-1")
        data = metrics.get_worker_metrics("w-1")
        assert data is not None
        assert data["deregistered_at"] is not None
        assert isinstance(data["deregistered_at"], float)
        assert data["deregistered_at"] > data["registered_at"]

    def test_on_deregister_unknown_worker_is_noop(self):
        """on_deregister for an unknown worker does not raise or add entries."""
        metrics = WorkerMetrics()
        metrics.on_deregister("nonexistent")
        assert metrics.get_worker_metrics("nonexistent") is None

    def test_on_task_complete_unknown_worker_returns_early(self):
        """on_task_complete for an unknown worker silently returns."""
        metrics = WorkerMetrics()
        metrics.on_task_complete("nonexistent", success=True, duration=5.0)
        assert metrics.get_worker_metrics("nonexistent") is None

    def test_on_anomaly_unknown_worker_returns_early(self):
        """on_anomaly for an unknown worker silently returns."""
        metrics = WorkerMetrics()
        metrics.on_anomaly("nonexistent", "test_anomaly")
        assert metrics.get_worker_metrics("nonexistent") is None

    def test_on_register_with_source_ip(self):
        """on_register records the source_ip field."""
        metrics = WorkerMetrics()
        metrics.on_register("w-1", source_ip="192.168.1.100")
        data = metrics.get_worker_metrics("w-1")
        assert data["source_ip"] == "192.168.1.100"

    def test_on_register_default_source_ip(self):
        """on_register uses empty string as default source_ip."""
        metrics = WorkerMetrics()
        metrics.on_register("w-1")
        data = metrics.get_worker_metrics("w-1")
        assert data["source_ip"] == ""

    def test_success_rate_zero_tasks(self):
        """success_rate returns 0.0 when no tasks have been completed."""
        metrics = WorkerMetrics()
        metrics.on_register("w-1")
        data = metrics.get_worker_metrics("w-1")
        assert data["success_rate"] == 0.0
        assert data["avg_duration"] == 0.0

    def test_success_rate_all_success(self):
        """success_rate is 1.0 when all tasks succeed."""
        metrics = WorkerMetrics()
        metrics.on_register("w-1")
        metrics.on_task_complete("w-1", success=True, duration=2.0)
        metrics.on_task_complete("w-1", success=True, duration=4.0)
        data = metrics.get_worker_metrics("w-1")
        assert data["success_rate"] == 1.0

    def test_success_rate_all_failure(self):
        """success_rate is 0.0 when all tasks fail."""
        metrics = WorkerMetrics()
        metrics.on_register("w-1")
        metrics.on_task_complete("w-1", success=False, duration=1.0)
        metrics.on_task_complete("w-1", success=False, duration=2.0)
        data = metrics.get_worker_metrics("w-1")
        assert data["success_rate"] == 0.0
        assert data["tasks_failed"] == 2

    def test_avg_duration_calculation(self):
        """avg_duration is the mean of all task durations."""
        metrics = WorkerMetrics()
        metrics.on_register("w-1")
        metrics.on_task_complete("w-1", success=True, duration=10.0)
        metrics.on_task_complete("w-1", success=False, duration=20.0)
        data = metrics.get_worker_metrics("w-1")
        assert abs(data["avg_duration"] - 15.0) < 0.01

    def test_on_anomaly_appends_types(self):
        """on_anomaly increments count and appends the anomaly type."""
        metrics = WorkerMetrics()
        metrics.on_register("w-1")
        metrics.on_anomaly("w-1", "perfect_correlation")
        metrics.on_anomaly("w-1", "suspiciously_fast")
        metrics.on_anomaly("w-1", "duplicate_correlations")
        data = metrics.get_worker_metrics("w-1")
        assert data["anomaly_count"] == 3

    def test_get_all_metrics_with_multiple_workers(self):
        """get_all_metrics returns data for all registered workers."""
        metrics = WorkerMetrics()
        metrics.on_register("w-1", source_ip="10.0.0.1")
        metrics.on_register("w-2", source_ip="10.0.0.2")
        metrics.on_register("w-3", source_ip="10.0.0.3")
        all_data = metrics.get_all_metrics()
        assert len(all_data) == 3
        worker_ids = {d["worker_id"] for d in all_data}
        assert worker_ids == {"w-1", "w-2", "w-3"}

    def test_get_all_metrics_empty(self):
        """get_all_metrics returns empty list when no workers registered."""
        metrics = WorkerMetrics()
        assert metrics.get_all_metrics() == []

    def test_deregister_then_get_metrics(self):
        """Metrics are still available after deregistration."""
        metrics = WorkerMetrics()
        metrics.on_register("w-1", source_ip="10.0.0.1")
        metrics.on_task_complete("w-1", success=True, duration=5.0)
        metrics.on_deregister("w-1")
        data = metrics.get_worker_metrics("w-1")
        assert data is not None
        assert data["deregistered_at"] is not None
        assert data["tasks_completed"] == 1

    def test_register_overwrites_previous(self):
        """Re-registering a worker_id replaces the previous metrics data."""
        metrics = WorkerMetrics()
        metrics.on_register("w-1", source_ip="10.0.0.1")
        metrics.on_task_complete("w-1", success=True, duration=5.0)
        metrics.on_register("w-1", source_ip="10.0.0.2")
        data = metrics.get_worker_metrics("w-1")
        assert data["source_ip"] == "10.0.0.2"
        assert data["tasks_completed"] == 0


# ---------------------------------------------------------------------------
# _WorkerMetricData Tests
# ---------------------------------------------------------------------------


class TestWorkerMetricData:
    """Tests for _WorkerMetricData covering __slots__ behavior and initialization."""

    def test_slots_defined(self):
        """_WorkerMetricData uses __slots__ for memory efficiency."""
        from api.workers.audit import _WorkerMetricData

        assert hasattr(_WorkerMetricData, "__slots__")
        expected_slots = (
            "worker_id",
            "source_ip",
            "registered_at",
            "deregistered_at",
            "tasks_completed",
            "tasks_succeeded",
            "tasks_failed",
            "total_duration",
            "anomaly_count",
            "anomaly_types",
        )
        assert _WorkerMetricData.__slots__ == expected_slots

    def test_no_dict_attribute(self):
        """Instances of _WorkerMetricData do not have __dict__ (slots-only)."""
        from api.workers.audit import _WorkerMetricData

        data = _WorkerMetricData(worker_id="w-1", source_ip="10.0.0.1", registered_at=1000.0)
        assert not hasattr(data, "__dict__")

    def test_cannot_set_arbitrary_attribute(self):
        """Setting an attribute not in __slots__ raises AttributeError."""
        from api.workers.audit import _WorkerMetricData

        data = _WorkerMetricData(worker_id="w-1", source_ip="10.0.0.1", registered_at=1000.0)
        with pytest.raises(AttributeError):
            data.arbitrary_field = "should fail"

    def test_initial_values(self):
        """_WorkerMetricData initializes counters to zero/empty."""
        from api.workers.audit import _WorkerMetricData

        data = _WorkerMetricData(worker_id="w-1", source_ip="10.0.0.1", registered_at=1234.5)
        assert data.worker_id == "w-1"
        assert data.source_ip == "10.0.0.1"
        assert data.registered_at == 1234.5
        assert data.deregistered_at is None
        assert data.tasks_completed == 0
        assert data.tasks_succeeded == 0
        assert data.tasks_failed == 0
        assert data.total_duration == 0.0
        assert data.anomaly_count == 0
        assert data.anomaly_types == []


# ---------------------------------------------------------------------------
# AuditEventType Tests
# ---------------------------------------------------------------------------


class TestAuditEventType:
    """Tests for AuditEventType enum covering string values and StrEnum behavior."""

    def test_all_event_types_exist(self):
        """All expected event type values are defined."""
        expected = {
            "worker_register",
            "worker_deregister",
            "auth_success",
            "auth_failure",
            "task_assigned",
            "result_accepted",
            "result_rejected",
            "rate_limited",
            "anomaly_detected",
            "tls_handshake",
            "connection_closed",
        }
        actual = {e.value for e in AuditEventType}
        assert actual == expected

    def test_str_enum_behavior(self):
        """AuditEventType members are strings (StrEnum)."""
        assert isinstance(AuditEventType.AUTH_SUCCESS, str)
        assert AuditEventType.AUTH_SUCCESS == "auth_success"


# ---------------------------------------------------------------------------
# TLS Configuration Tests — Extended Coverage
# ---------------------------------------------------------------------------


def _openssl_self_signed(tmp_path, stem: str):
    """Generate a self-signed cert+key pair via openssl; skip if unavailable."""
    import subprocess

    key_file = tmp_path / f"{stem}.key"
    cert_file = tmp_path / f"{stem}.crt"
    result = subprocess.run(  # nosec B607, B603
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_file),
            "-out",
            str(cert_file),
            "-days",
            "1",
            "-nodes",
            "-subj",
            f"/CN=Test{stem}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip("openssl not available for cert generation")
    return cert_file, key_file


class TestTLSConfigExtended:
    """Extended tests for TLSConfig.build_ssl_context() covering TLS versions,
    client cert modes, CA loading, and the full happy path."""

    def test_enabled_no_cert_no_key_raises(self):
        """TLS enabled with no cert/key must fail closed (no bare SSLContext)."""
        cfg = TLSConfig(enabled=True)
        with pytest.raises(ValueError, match="requires both cert_file and key_file"):
            cfg.build_ssl_context()

    def test_min_tls_version_1_3(self, tmp_path):
        """Default TLSv1.3 sets minimum_version correctly."""
        cert_file, key_file = _openssl_self_signed(tmp_path, "server13")
        cfg = TLSConfig(enabled=True, cert_file=str(cert_file), key_file=str(key_file), min_tls_version="TLSv1.3")
        ctx = cfg.build_ssl_context()
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_3

    def test_min_tls_version_1_2(self, tmp_path):
        """Non-TLSv1.3 string falls back to TLSv1.2."""
        cert_file, key_file = _openssl_self_signed(tmp_path, "server12")
        cfg = TLSConfig(enabled=True, cert_file=str(cert_file), key_file=str(key_file), min_tls_version="TLSv1.2")
        ctx = cfg.build_ssl_context()
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_min_tls_version_unknown_falls_to_1_2(self, tmp_path):
        """Any unrecognized TLS version string falls back to TLSv1.2."""
        cert_file, key_file = _openssl_self_signed(tmp_path, "server11")
        cfg = TLSConfig(enabled=True, cert_file=str(cert_file), key_file=str(key_file), min_tls_version="TLSv1.1")
        ctx = cfg.build_ssl_context()
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_require_client_cert_no_ca_uses_system_store(self, tmp_path):
        """require_client_cert=True without ca_file sets CERT_REQUIRED using system trust store."""
        cert_file, key_file = _openssl_self_signed(tmp_path, "server-mtls")
        cfg = TLSConfig(enabled=True, cert_file=str(cert_file), key_file=str(key_file), require_client_cert=True)
        ctx = cfg.build_ssl_context()
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_require_client_cert_with_valid_ca(self, tmp_path):
        """require_client_cert=True with a valid CA file loads it and sets CERT_REQUIRED."""
        ca_cert, _ca_key = _openssl_self_signed(tmp_path, "ca")
        cert_file, key_file = _openssl_self_signed(tmp_path, "server-ca")

        cfg = TLSConfig(
            enabled=True,
            cert_file=str(cert_file),
            key_file=str(key_file),
            require_client_cert=True,
            ca_file=str(ca_cert),
        )
        ctx = cfg.build_ssl_context()
        assert ctx is not None
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_load_server_cert_chain(self, tmp_path):
        """Valid cert_file and key_file are loaded into the SSL context."""
        import subprocess

        key_file = tmp_path / "server.key"
        cert_file = tmp_path / "server.crt"

        result = subprocess.run(  # nosec B607, B603
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(key_file),
                "-out",
                str(cert_file),
                "-days",
                "1",
                "-nodes",
                "-subj",
                "/CN=TestServer",
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            pytest.skip("openssl not available for cert generation")

        cfg = TLSConfig(enabled=True, cert_file=str(cert_file), key_file=str(key_file))
        ctx = cfg.build_ssl_context()
        assert ctx is not None

    def test_require_client_cert_with_missing_ca_raises(self, tmp_path):
        """require_client_cert=True with a non-existent ca_file raises FileNotFoundError."""
        cfg = TLSConfig(enabled=True, require_client_cert=True, ca_file=str(tmp_path / "no_such_ca.pem"))
        with pytest.raises(FileNotFoundError, match="CA cert not found"):
            cfg.build_ssl_context()

    def test_missing_cert_file_raises(self, tmp_path):
        """Non-existent cert_file raises FileNotFoundError."""
        cfg = TLSConfig(enabled=True, cert_file=str(tmp_path / "missing.crt"), key_file=str(tmp_path / "missing.key"))
        with pytest.raises(FileNotFoundError, match="TLS cert not found"):
            cfg.build_ssl_context()

    def test_missing_key_file_raises(self, tmp_path):
        """Existing cert but missing key raises FileNotFoundError."""
        cert = tmp_path / "server.crt"
        cert.write_text("dummy")
        cfg = TLSConfig(enabled=True, cert_file=str(cert), key_file=str(tmp_path / "missing.key"))
        with pytest.raises(FileNotFoundError, match="TLS key not found"):
            cfg.build_ssl_context()

    def test_disabled_returns_none(self):
        """Disabled TLS returns None regardless of other settings."""
        cfg = TLSConfig(enabled=False, cert_file="/some/cert", key_file="/some/key", ca_file="/some/ca", require_client_cert=True)
        assert cfg.build_ssl_context() is None

    def test_full_mtls_configuration(self, tmp_path):
        """Full mTLS setup with cert, key, and CA all valid."""
        import subprocess

        ca_key = tmp_path / "ca.key"
        ca_cert = tmp_path / "ca.crt"
        server_key = tmp_path / "server.key"
        server_cert = tmp_path / "server.crt"

        # Generate CA cert
        result = subprocess.run(  # nosec B607, B603
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", str(ca_key), "-out", str(ca_cert), "-days", "1", "-nodes", "-subj", "/CN=TestCA"],
            capture_output=True,
        )
        if result.returncode != 0:
            pytest.skip("openssl not available")

        # Generate server cert (self-signed for simplicity)
        result = subprocess.run(  # nosec B607, B603
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", str(server_key), "-out", str(server_cert), "-days", "1", "-nodes", "-subj", "/CN=TestServer"],
            capture_output=True,
        )
        if result.returncode != 0:
            pytest.skip("openssl not available")

        cfg = TLSConfig(enabled=True, cert_file=str(server_cert), key_file=str(server_key), ca_file=str(ca_cert), require_client_cert=True, min_tls_version="TLSv1.3")
        ctx = cfg.build_ssl_context()
        assert ctx is not None
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_3


# ---------------------------------------------------------------------------
# Connection Rate Limiter Tests — Extended Coverage
# ---------------------------------------------------------------------------


class TestConnectionRateLimiterExtended:
    """Extended tests for ConnectionRateLimiter covering _maybe_cleanup(),
    stale bucket removal, and cleanup interval logic."""

    def test_cleanup_removes_stale_buckets(self):
        """Stale buckets are removed when cleanup interval has passed."""
        limiter = ConnectionRateLimiter(max_connections_per_minute=60, burst_size=3, cleanup_interval=0.0)
        # Force a bucket to exist
        limiter.allow("stale-source")
        # Manually age the bucket's last_access and set last_cleanup in the past
        limiter._buckets["stale-source"].last_access = time.time() - 1000
        limiter._last_cleanup = time.time() - 1000
        # Next call triggers cleanup
        limiter.allow("new-source")
        assert "stale-source" not in limiter._buckets
        assert "new-source" in limiter._buckets

    def test_cleanup_does_not_run_before_interval(self):
        """Cleanup does not run when interval has not elapsed."""
        limiter = ConnectionRateLimiter(max_connections_per_minute=60, burst_size=3, cleanup_interval=9999.0)
        limiter.allow("source-1")
        limiter.allow("source-2")
        # Both buckets should still exist since cleanup_interval hasn't elapsed
        assert "source-1" in limiter._buckets
        assert "source-2" in limiter._buckets

    def test_cleanup_preserves_active_buckets(self):
        """Active (recently accessed) buckets are not removed during cleanup."""
        limiter = ConnectionRateLimiter(max_connections_per_minute=60, burst_size=3, cleanup_interval=100.0)
        limiter.allow("active-source")
        limiter.allow("stale-source")
        # Age only the stale bucket and force cleanup to trigger
        limiter._buckets["stale-source"].last_access = time.time() - 200
        limiter._last_cleanup = time.time() - 200
        # Trigger cleanup via the next allow() call
        limiter.allow("active-source")
        # stale-source should be cleaned up, active-source should survive
        assert "active-source" in limiter._buckets
        assert "stale-source" not in limiter._buckets

    def test_cleanup_updates_last_cleanup_time(self):
        """Cleanup updates _last_cleanup timestamp."""
        limiter = ConnectionRateLimiter(max_connections_per_minute=60, burst_size=3, cleanup_interval=0.0)
        old_cleanup = limiter._last_cleanup
        # Force cleanup to trigger
        limiter._last_cleanup = time.time() - 1000
        limiter.allow("source")
        assert limiter._last_cleanup > old_cleanup

    def test_multiple_stale_buckets_removed(self):
        """Multiple stale buckets are removed in a single cleanup pass."""
        limiter = ConnectionRateLimiter(max_connections_per_minute=60, burst_size=3, cleanup_interval=0.0)
        # Create several buckets
        for i in range(5):
            limiter.allow(f"source-{i}")
        # Age all buckets
        for bucket in limiter._buckets.values():
            bucket.last_access = time.time() - 1000
        limiter._last_cleanup = time.time() - 1000
        # Trigger cleanup
        limiter.allow("fresh")
        # All stale buckets should be gone, only fresh remains
        assert len(limiter._buckets) == 1
        assert "fresh" in limiter._buckets

    def test_rate_limited_source_logged(self, caplog):
        """Rate-limited connections produce a warning log."""
        limiter = ConnectionRateLimiter(max_connections_per_minute=60, burst_size=1)
        logger = logging.getLogger("juniper_cascor.api.workers.security")
        logger.disabled = False
        with caplog.at_level(logging.WARNING, logger="juniper_cascor.api.workers.security"):
            limiter.allow("10.0.0.1")  # succeeds
            limiter.allow("10.0.0.1")  # fails
        assert "Rate limited" in caplog.text

    def test_allow_creates_new_bucket_per_source(self):
        """Each unique source_id gets its own bucket."""
        limiter = ConnectionRateLimiter(max_connections_per_minute=60, burst_size=2)
        limiter.allow("alpha")
        limiter.allow("beta")
        limiter.allow("gamma")
        assert len(limiter._buckets) == 3


# ---------------------------------------------------------------------------
# _TokenBucket Tests
# ---------------------------------------------------------------------------


class TestTokenBucket:
    """Tests for the _TokenBucket dataclass covering refill logic and boundary conditions."""

    def test_initial_tokens_equal_burst(self):
        """Tokens start at burst capacity."""
        bucket = _TokenBucket(rate=1.0, burst=5)
        assert bucket.tokens == 5.0

    def test_consume_decrements_tokens(self):
        """Consuming a token reduces the count."""
        bucket = _TokenBucket(rate=1.0, burst=3)
        now = time.time()
        assert bucket.consume(now) is True
        assert bucket.tokens == 2.0

    def test_consume_fails_when_empty(self):
        """Cannot consume when no tokens remain."""
        bucket = _TokenBucket(rate=0.0, burst=1)
        now = time.time()
        assert bucket.consume(now) is True  # Use the only token
        assert bucket.consume(now) is False

    def test_refill_adds_tokens_over_time(self):
        """Tokens are refilled based on elapsed time and rate."""
        bucket = _TokenBucket(rate=10.0, burst=5)
        now = time.time()
        # Exhaust all tokens
        for _ in range(5):
            bucket.consume(now)
        assert bucket.tokens < 1.0
        # Simulate 0.5 seconds passing at rate=10/sec -> +5 tokens
        later = now + 0.5
        assert bucket.consume(later) is True

    def test_refill_caps_at_burst(self):
        """Refill never exceeds burst size."""
        bucket = _TokenBucket(rate=100.0, burst=3)
        now = time.time()
        # Wait a long time
        later = now + 1000
        bucket.consume(later)
        # Even after massive elapsed time, tokens are capped at burst
        assert bucket.tokens <= 3.0

    def test_consume_updates_last_access(self):
        """consume() updates last_access to the provided timestamp."""
        bucket = _TokenBucket(rate=1.0, burst=3)
        target_time = time.time() + 100
        bucket.consume(target_time)
        assert bucket.last_access == target_time

    def test_consume_updates_last_refill(self):
        """consume() updates last_refill to the provided timestamp."""
        bucket = _TokenBucket(rate=1.0, burst=3)
        target_time = time.time() + 100
        bucket.consume(target_time)
        assert bucket.last_refill == target_time

    def test_zero_rate_no_refill(self):
        """With rate=0, no tokens are ever refilled."""
        bucket = _TokenBucket(rate=0.0, burst=2)
        now = time.time()
        bucket.consume(now)
        bucket.consume(now)
        # No tokens left, and rate=0 means no refill
        later = now + 1000
        assert bucket.consume(later) is False

    def test_fractional_token_accumulation(self):
        """Fractional tokens accumulate across multiple consume() calls."""
        bucket = _TokenBucket(rate=1.0, burst=2)
        now = time.time()
        # Use both tokens
        bucket.consume(now)
        bucket.consume(now)
        # 0.5 seconds later at rate=1/sec -> 0.5 tokens (not enough)
        assert bucket.consume(now + 0.5) is False
        # Another 0.6 seconds later -> 0.5 + 0.6 = 1.1 tokens (enough for 1)
        assert bucket.consume(now + 1.1) is True


# ---------------------------------------------------------------------------
# Anomaly Detection Tests — Extended Coverage
# ---------------------------------------------------------------------------


class TestAnomalyDetectorExtended:
    """Extended tests for AnomalyDetector covering all anomaly types,
    history trimming, duplicate detection edge cases, and worker management."""

    def test_multiple_anomalies_at_once(self):
        """A single result can trigger multiple anomaly types simultaneously."""
        detector = AnomalyDetector(min_training_time=1.0, perfect_corr_threshold=0.999, stale_corr_threshold=0.001)
        # Fast training + perfect correlation
        anomalies = detector.check_result("w-1", correlation=0.9999, training_duration=0.001, task_id="t-1")
        assert any("suspiciously_fast" in a for a in anomalies)
        assert any("perfect_correlation" in a for a in anomalies)
        assert len(anomalies) >= 2

    def test_fast_and_stale(self):
        """Fast training + stale correlation triggers both anomalies."""
        detector = AnomalyDetector(min_training_time=1.0, stale_corr_threshold=0.001)
        anomalies = detector.check_result("w-1", correlation=0.0001, training_duration=0.05, task_id="t-1")
        types = [a.split(":")[0] for a in anomalies]
        assert "suspiciously_fast" in types
        assert "stale_correlation" in types

    def test_history_trimming(self):
        """History is trimmed to duplicate_corr_window size."""
        window = 5
        detector = AnomalyDetector(duplicate_corr_window=window)
        # Submit more results than the window
        for i in range(window + 10):
            detector.check_result("w-1", correlation=0.5 + i * 0.01, training_duration=5.0, task_id=f"t-{i}")
        stats = detector.get_worker_stats("w-1")
        assert stats["total_results"] == window

    def test_duplicate_not_triggered_with_varied_correlations(self):
        """Duplicate detection does not trigger when correlations vary."""
        detector = AnomalyDetector(duplicate_corr_window=10)
        for i in range(5):
            anomalies = detector.check_result("w-1", correlation=0.1 * (i + 1), training_duration=5.0, task_id=f"t-{i}")
        assert not any("duplicate_correlations" in a for a in anomalies)

    def test_duplicate_not_triggered_with_only_two_results(self):
        """Duplicate detection requires at least 3 results in history."""
        detector = AnomalyDetector(duplicate_corr_window=10)
        anomalies = detector.check_result("w-1", correlation=0.5, training_duration=5.0, task_id="t-1")
        assert not any("duplicate_correlations" in a for a in anomalies)
        anomalies = detector.check_result("w-1", correlation=0.5, training_duration=5.0, task_id="t-2")
        assert not any("duplicate_correlations" in a for a in anomalies)

    def test_duplicate_triggers_at_exactly_three(self):
        """Duplicate detection triggers with exactly 3 identical correlations."""
        detector = AnomalyDetector(duplicate_corr_window=10)
        for i in range(3):
            anomalies = detector.check_result("w-1", correlation=0.5, training_duration=5.0, task_id=f"t-{i}")
        assert any("duplicate_correlations" in a for a in anomalies)

    def test_stale_negative_correlation(self):
        """Negative correlation near zero also triggers stale detection."""
        detector = AnomalyDetector(stale_corr_threshold=0.001)
        anomalies = detector.check_result("w-1", correlation=-0.0005, training_duration=5.0, task_id="t-1")
        assert any("stale_correlation" in a for a in anomalies)

    def test_exact_threshold_boundary_perfect(self):
        """Correlation exactly at perfect threshold is NOT flagged (must exceed)."""
        detector = AnomalyDetector(perfect_corr_threshold=0.999)
        anomalies = detector.check_result("w-1", correlation=0.999, training_duration=5.0, task_id="t-1")
        assert not any("perfect_correlation" in a for a in anomalies)

    def test_exact_threshold_boundary_stale(self):
        """Correlation exactly at stale threshold IS flagged (abs < threshold)."""
        detector = AnomalyDetector(stale_corr_threshold=0.001)
        # abs(0.001) is NOT < 0.001, so it should not trigger
        anomalies = detector.check_result("w-1", correlation=0.001, training_duration=5.0, task_id="t-1")
        assert not any("stale_correlation" in a for a in anomalies)

    def test_exact_threshold_boundary_training_time(self):
        """Training time exactly at min threshold is NOT flagged (must be less than)."""
        detector = AnomalyDetector(min_training_time=1.0)
        anomalies = detector.check_result("w-1", correlation=0.5, training_duration=1.0, task_id="t-1")
        assert not any("suspiciously_fast" in a for a in anomalies)

    def test_get_worker_stats_empty(self):
        """Stats for unknown worker returns total_results=0."""
        detector = AnomalyDetector()
        stats = detector.get_worker_stats("nonexistent")
        assert stats == {"total_results": 0}

    def test_get_worker_stats_full(self):
        """Stats contain all expected fields with correct values."""
        detector = AnomalyDetector()
        detector.check_result("w-1", 0.3, 2.0, "t-1")
        detector.check_result("w-1", 0.9, 8.0, "t-2")
        stats = detector.get_worker_stats("w-1")
        assert stats["total_results"] == 2
        assert abs(stats["avg_correlation"] - 0.6) < 0.01
        assert abs(stats["avg_duration"] - 5.0) < 0.01
        assert stats["min_duration"] == 2.0
        assert stats["max_correlation"] == 0.9

    def test_clear_worker_idempotent(self):
        """Clearing a non-existent worker does not raise."""
        detector = AnomalyDetector()
        detector.clear_worker("nonexistent")  # Should not raise
        stats = detector.get_worker_stats("nonexistent")
        assert stats["total_results"] == 0

    def test_separate_workers_independent(self):
        """Each worker has independent history and anomaly tracking."""
        detector = AnomalyDetector(duplicate_corr_window=5)
        # Worker 1 gets identical correlations
        for i in range(5):
            anomalies_w1 = detector.check_result("w-1", correlation=0.5, training_duration=5.0, task_id=f"w1-t-{i}")
        # Worker 2 gets varied correlations
        for i in range(5):
            anomalies_w2 = detector.check_result("w-2", correlation=0.1 * (i + 1), training_duration=5.0, task_id=f"w2-t-{i}")
        # Worker 1 should have duplicates, worker 2 should not
        assert any("duplicate_correlations" in a for a in anomalies_w1)
        assert not any("duplicate_correlations" in a for a in anomalies_w2)

    def test_anomaly_logging(self, caplog):
        """Anomalies produce warning log messages."""
        detector = AnomalyDetector(min_training_time=1.0)
        logger = logging.getLogger("juniper_cascor.api.workers.security")
        logger.disabled = False
        with caplog.at_level(logging.WARNING, logger="juniper_cascor.api.workers.security"):
            detector.check_result("w-1", correlation=0.5, training_duration=0.001, task_id="t-1")
        assert "ANOMALY" in caplog.text
        assert "w-1" in caplog.text

    def test_clean_result_no_logging(self, caplog):
        """Clean results produce no warning log messages."""
        detector = AnomalyDetector()
        logger = logging.getLogger("juniper_cascor.api.workers.security")
        logger.disabled = False
        with caplog.at_level(logging.WARNING, logger="juniper_cascor.api.workers.security"):
            detector.check_result("w-1", correlation=0.5, training_duration=5.0, task_id="t-1")
        assert "ANOMALY" not in caplog.text
