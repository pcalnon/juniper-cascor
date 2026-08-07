"""Route-level tests for Phase 2 P2-1a endpoints.

Covers:
  - ``GET /v1/admin/experimental_functions`` — initial state
  - ``POST /v1/admin/experimental_functions`` — toggle
  - ``POST /v1/training/dataset/live`` — exception translation to HTTP status

Lifecycle-method behavior is covered in detail by
``tests/integration/api/test_swap_dataset_live.py``; this file pins ONLY the
route-layer contract (which exception → which HTTP code, response shape).

See ``ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md`` §3.3.
"""

import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.app import create_app
from api.lifecycle.manager import (
    NoSwapInProgressError,
    SwapCancelledError,
    SwapInProgressError,
)
from api.settings import Settings

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Admin gate route
# ---------------------------------------------------------------------------


class TestAdminExperimentalFunctionsRoute:
    def test_get_initial_state_false(self, client):
        """Gate is closed by default — the F2.10 safe default."""
        resp = client.get("/v1/admin/experimental_functions")
        assert resp.status_code == 200
        body = resp.json()
        # success_response envelope: {"status": "success", "data": {...}}
        assert body["data"]["enabled"] is False

    def test_post_opens_gate(self, client):
        resp = client.post("/v1/admin/experimental_functions", json={"enabled": True})
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["experimental_functions_enabled"] is True

        # Subsequent GET reflects the new state.
        resp2 = client.get("/v1/admin/experimental_functions")
        assert resp2.json()["data"]["enabled"] is True

    def test_post_closes_gate(self, client):
        client.post("/v1/admin/experimental_functions", json={"enabled": True})
        resp = client.post("/v1/admin/experimental_functions", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json()["data"]["experimental_functions_enabled"] is False

    def test_post_missing_enabled_field_422(self, client):
        """Pydantic validation: ``enabled`` is required."""
        resp = client.post("/v1/admin/experimental_functions", json={})
        assert resp.status_code == 422

    def test_get_503_when_lifecycle_unbound(self, client):
        """Admin gate must fail closed with 503 when lifespan binding is missing."""
        original = client.app.state.lifecycle
        client.app.state.lifecycle = None
        try:
            resp = client.get("/v1/admin/experimental_functions")
            assert resp.status_code == 503
            assert "Lifecycle manager not initialized" in resp.json()["error"]["message"]
        finally:
            client.app.state.lifecycle = original

    def test_post_503_when_lifecycle_unbound(self, client):
        """Mutating the experimental gate must also 503 when lifecycle is unbound."""
        original = client.app.state.lifecycle
        client.app.state.lifecycle = None
        try:
            resp = client.post("/v1/admin/experimental_functions", json={"enabled": True})
            assert resp.status_code == 503
            assert "Lifecycle manager not initialized" in resp.json()["error"]["message"]
        finally:
            client.app.state.lifecycle = original


# ---------------------------------------------------------------------------
# Live-swap route — exception translation table
# ---------------------------------------------------------------------------


class TestSwapDatasetLiveRoute:
    """Each test pins the contract: a specific lifecycle exception maps to a
    specific HTTP status. Lifecycle-method internals (rollback, snapshot
    capture, etc.) are covered by the integration tests."""

    def test_403_when_gate_closed(self, client):
        """PermissionError → 403. The default-closed gate makes this the
        first-line check; tests it without needing a running training session."""
        resp = client.post("/v1/training/dataset/live", json={"dataset_type": "spirals"})
        assert resp.status_code == 403
        assert "experimental_functions_disabled" in resp.json()["error"]["message"]

    def test_422_when_training_not_running(self, client):
        """Gate open but no active training → ValueError → 422."""
        client.post("/v1/admin/experimental_functions", json={"enabled": True})
        resp = client.post("/v1/training/dataset/live", json={"dataset_type": "spirals"})
        assert resp.status_code == 422
        assert "training_not_running" in resp.json()["error"]["message"]

    def test_409_when_swap_in_progress(self, client):
        """SwapInProgressError → 409. Patches the lifecycle method to raise
        directly so we don't need a real concurrent swap to trigger it."""
        client.post("/v1/admin/experimental_functions", json={"enabled": True})
        with patch.object(
            client.app.state.lifecycle,
            "swap_dataset_live",
            side_effect=SwapInProgressError("swap_already_in_progress"),
        ):
            resp = client.post("/v1/training/dataset/live", json={"dataset_type": "spirals"})
        assert resp.status_code == 409
        assert "swap_already_in_progress" in resp.json()["error"]["message"]

    def test_502_on_juniper_data_fetch_failure(self, client):
        """RuntimeError (juniper-data unreachable etc.) → 502. Distinguished
        from 5xx-failure-class generic errors by the specific exception type."""
        client.post("/v1/admin/experimental_functions", json={"enabled": True})
        with patch.object(
            client.app.state.lifecycle,
            "swap_dataset_live",
            side_effect=RuntimeError("juniper-data fetch failed: connection refused"),
        ):
            resp = client.post("/v1/training/dataset/live", json={"dataset_type": "spirals"})
        assert resp.status_code == 502
        assert "juniper-data" in resp.json()["error"]["message"]

    def test_504_on_pause_timeout(self, client):
        """TimeoutError (from future.result(timeout=10)) → 504 per §3.7 #2."""
        client.post("/v1/admin/experimental_functions", json={"enabled": True})
        with patch.object(
            client.app.state.lifecycle,
            "swap_dataset_live",
            side_effect=TimeoutError("training thread did not pause within 10s"),
        ):
            resp = client.post("/v1/training/dataset/live", json={"dataset_type": "spirals"})
        assert resp.status_code == 504
        assert "pause_timeout" in resp.json()["error"]["message"]

    def test_success_response_envelope(self, client):
        """Happy-path response is wrapped in the success envelope. Patches the
        lifecycle method to return the §3.3 response without driving a real
        swap (covered by the integration tests)."""
        client.post("/v1/admin/experimental_functions", json={"enabled": True})
        canned = {
            "status": "swapped",
            "before_cfg": {"dataset_type": "spirals"},
            "after_cfg": {"dataset_type": "moons"},
            "arch_changes": {
                "input_delta": 0,
                "output_delta": 0,
                "hidden_preserved": 0,
                "abandoned_candidate_pool_size": 0,
                "appended_nodes": {"input": 0, "output": 0},
                "prepended_layers": [],
            },
            "mode": "output_training_first",
        }
        with patch.object(client.app.state.lifecycle, "swap_dataset_live", return_value=canned):
            resp = client.post("/v1/training/dataset/live", json={"dataset_type": "moons"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == canned

    def test_200_cancelled_when_swap_cancelled_mid_flight(self, client):
        """P2-1b: ``SwapCancelledError`` → 200 with ``status="cancelled"`` in the
        response data. Distinct from a 5xx failure so the canopy "Live Switch
        failed" toast does NOT fire on a user-initiated cancel."""
        client.post("/v1/admin/experimental_functions", json={"enabled": True})
        with patch.object(
            client.app.state.lifecycle,
            "swap_dataset_live",
            side_effect=SwapCancelledError("swap_cancelled_by_client"),
        ):
            resp = client.post("/v1/training/dataset/live", json={"dataset_type": "spirals"})
        assert resp.status_code == 200
        assert resp.json()["data"] == {"status": "cancelled"}


# ---------------------------------------------------------------------------
# P2-1b: DELETE /v1/training/dataset/live — cancel route
# ---------------------------------------------------------------------------


class TestCancelSwapDatasetLiveRoute:
    """Pin the DELETE-route contract per §7 P2-1b row of the spec."""

    def test_403_when_gate_closed(self, client):
        """Closed gate hides the cancel surface too — avoids a probe vector
        that would distinguish "swap enabled but idle" from "swap disabled"."""
        resp = client.delete("/v1/training/dataset/live")
        assert resp.status_code == 403
        assert "experimental_functions_disabled" in resp.json()["error"]["message"]

    def test_404_when_no_swap_in_progress(self, client):
        """``NoSwapInProgressError`` → 404. The "swap finished racing the
        click" signal for the canopy Cancel button."""
        client.post("/v1/admin/experimental_functions", json={"enabled": True})
        with patch.object(
            client.app.state.lifecycle,
            "request_swap_cancel",
            side_effect=NoSwapInProgressError("no_swap_in_progress"),
        ):
            resp = client.delete("/v1/training/dataset/live")
        assert resp.status_code == 404
        assert "no_swap_in_progress" in resp.json()["error"]["message"]

    def test_200_when_swap_in_progress(self, client):
        """Happy path: returns the lifecycle's descriptor dict (cancel signal
        accepted) wrapped in the success envelope. The swap rollback itself
        completes asynchronously — see integration tests."""
        client.post("/v1/admin/experimental_functions", json={"enabled": True})
        with patch.object(
            client.app.state.lifecycle,
            "request_swap_cancel",
            return_value={"status": "cancel_requested"},
        ):
            resp = client.delete("/v1/training/dataset/live")
        assert resp.status_code == 200
        assert resp.json()["data"] == {"status": "cancel_requested"}
