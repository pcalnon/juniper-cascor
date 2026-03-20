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
from api.workers.security import AnomalyDetector, ConnectionRateLimiter, TLSConfig

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
