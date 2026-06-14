"""Wire-compat snapshot tests for the R2.1.4 juniper-observability migration.

METRICS-MON R2.1.4 / seed-06: per the R2.1 design §7, every consumer
migration ships a snapshot test that pins the externally-observable
wire format of ``/v1/health/ready`` and the Prometheus contract so the
shared-lib swap cannot silently drift the contract.

The snapshot below was captured from juniper-cascor ``main`` at
HEAD = ``f14cd906`` (commit immediately before the R2.1.4 migration
landed). Any future bump of the shared lib that changes these keys,
status codes, or label sets will fail this test first.

Two keys deliberately differ from the pre-migration snapshot:

- ``timestamp`` is now tz-aware UTC (closes BUG-JD-06-equivalent
  naive-tz drift). The value remains a unix epoch float; only its
  derivation changes.
- ``service`` and ``status`` strings are unchanged.
"""

import os

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings


@pytest.fixture
def healthy_client(monkeypatch):
    """A TestClient with a JUNIPER_DATA_URL unset so dep probing collapses."""
    monkeypatch.delenv("JUNIPER_DATA_URL", raising=False)
    settings = Settings(auto_start=False)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        yield c


# Snapshot captured pre-R2.1.4 (cascor main @ f14cd906). The shared lib
# migration must preserve every entry below. ``git_sha`` / ``build_date`` were
# added additively by the build-provenance effort (obs 0.4.0 / juniper-ml
# notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md) — optional, default ``None``, so
# the extension stays wire-compatible with pre-0.4.0 consumers.
EXPECTED_TOP_LEVEL_KEYS = {"build_date", "dependencies", "details", "git_sha", "service", "status", "timestamp", "version"}
EXPECTED_DEP_KEYS = {"lifecycle"}
EXPECTED_DETAILS_KEYS = {"network_loaded", "training_state"}


class TestReadinessWireCompat:
    """METRICS-MON R2.1.4: /v1/health/ready JSON shape pinned across the migration."""

    def test_status_code_unchanged_when_lifecycle_bound(self, healthy_client):
        response = healthy_client.get("/v1/health/ready")
        assert response.status_code == 200

    def test_x_juniper_readiness_header_unchanged(self, healthy_client):
        """R1.2 contract: header mirrors body status."""
        response = healthy_client.get("/v1/health/ready")
        assert response.headers.get("X-Juniper-Readiness") == "ready"

    def test_top_level_keys_unchanged(self, healthy_client):
        """The standard ReadinessResponse shape, plus the additive
        build-provenance ``git_sha`` / ``build_date`` keys (obs 0.4.0)."""
        response = healthy_client.get("/v1/health/ready")
        body = response.json()
        assert set(body.keys()) == EXPECTED_TOP_LEVEL_KEYS

    def test_status_value_unchanged(self, healthy_client):
        response = healthy_client.get("/v1/health/ready")
        assert response.json()["status"] == "ready"

    def test_service_identity_unchanged(self, healthy_client):
        response = healthy_client.get("/v1/health/ready")
        assert response.json()["service"] == "juniper-cascor"

    def test_dependency_set_unchanged_when_data_unconfigured(self, healthy_client):
        """When ``JUNIPER_DATA_URL`` is unset, only the lifecycle dep appears."""
        response = healthy_client.get("/v1/health/ready")
        assert set(response.json()["dependencies"].keys()) == EXPECTED_DEP_KEYS

    def test_details_keys_unchanged(self, healthy_client):
        """``details`` always carries cascor's training/network state."""
        response = healthy_client.get("/v1/health/ready")
        assert set(response.json()["details"].keys()) == EXPECTED_DETAILS_KEYS

    def test_timestamp_is_unix_epoch_float(self, healthy_client):
        """The shared lib reconciliation kept ``timestamp`` as a unix-epoch float."""
        import time

        response = healthy_client.get("/v1/health/ready")
        ts = response.json()["timestamp"]
        assert isinstance(ts, float)
        # R2.1.4 fix: now tz-aware UTC so this should be within seconds of
        # ``time.time()`` (which is also UTC unix epoch) on every host.
        assert abs(time.time() - ts) < 60.0


class TestReadiness503OnLifecycleMissing:
    """R1.2 contract: lifecycle-missing → 503 with status=not_ready."""

    def test_503_and_not_ready_when_lifecycle_unbound(self, healthy_client):
        original = healthy_client.app.state.lifecycle
        healthy_client.app.state.lifecycle = None
        try:
            response = healthy_client.get("/v1/health/ready")
            assert response.status_code == 503
            assert response.json()["status"] == "not_ready"
            assert response.headers.get("X-Juniper-Readiness") == "not_ready"
        finally:
            healthy_client.app.state.lifecycle = original


class TestPrometheusContract:
    """METRICS-MON R2.1.4: HTTP metric names + label sets pinned."""

    def test_unmatched_endpoint_label_value(self):
        """The R1.1 cardinality bound must remain the same string post-migration."""
        from api.observability import UNMATCHED_ENDPOINT_LABEL

        assert UNMATCHED_ENDPOINT_LABEL == "_unmatched"

    def test_namespace_prefix_preserved(self):
        """``juniper_cascor_*`` prefix is the R1.1 contract for metric names."""
        from unittest.mock import MagicMock, patch

        from api.observability import PrometheusMiddleware

        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Histogram") as MockHistogram:
            MockCounter.return_value = MagicMock()
            MockHistogram.return_value = MagicMock()

            PrometheusMiddleware(app=MagicMock(), service_name="juniper-cascor", namespace="juniper_cascor")

            counter_names = {call.args[0] for call in MockCounter.call_args_list}
            histogram_names = {call.args[0] for call in MockHistogram.call_args_list}
            assert "juniper_cascor_http_requests_total" in counter_names
            assert "juniper_cascor_http_unmatched_requests_total" in counter_names
            assert "juniper_cascor_http_request_duration_seconds" in histogram_names
