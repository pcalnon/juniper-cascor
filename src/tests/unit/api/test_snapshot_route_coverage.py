"""
Unit tests for api/routes/snapshots.py to improve code coverage.

Covers:
- _get_lifecycle: success path and HTTPException when lifecycle not initialized
- save_snapshot: success, no network (404), save returns None
- list_snapshots: empty list, populated list
- get_snapshot: found, not found (404)
- restore_snapshot: success, not found (404), load fails
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.app import create_app
from api.settings import Settings

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    """Create a test client with lifecycle manager (lifespan runs)."""
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# _get_lifecycle helper
# ---------------------------------------------------------------------------


class TestSnapshotRouteLifecycle:
    """Tests for _get_lifecycle in snapshot routes."""

    def test_lifecycle_not_initialized_returns_503(self):
        """Snapshot routes should return 503 when lifecycle is not initialized."""
        settings = Settings(auto_start=False)
        app = create_app(settings)

        with TestClient(app) as c:
            lifecycle = c.app.state.lifecycle
            del c.app.state.lifecycle

            response = c.get("/v1/snapshots")
            assert response.status_code == 503
            assert "Lifecycle manager not initialized" in response.json()["detail"]

            # Restore to allow clean teardown
            c.app.state.lifecycle = lifecycle

    def test_lifecycle_initialized_does_not_raise(self, client):
        """Snapshot routes should proceed when lifecycle is initialized."""
        with patch.object(client.app.state.lifecycle, "list_snapshots", return_value=[]):
            response = client.get("/v1/snapshots")
            assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /v1/snapshots  (save_snapshot)
# ---------------------------------------------------------------------------


class TestSaveSnapshot:
    """Tests for save_snapshot endpoint."""

    def test_save_snapshot_success(self, client):
        """save_snapshot should return success with snapshot data."""
        snapshot_data = {"snapshot_id": "snap-001", "description": "test snapshot", "created_at": "2026-04-01T00:00:00Z"}
        with patch.object(client.app.state.lifecycle, "has_network", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", return_value=snapshot_data):
            response = client.post("/v1/snapshots", json={"description": "test snapshot"})
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "success"
            assert body["data"]["snapshot_id"] == "snap-001"
            assert body["data"]["description"] == "test snapshot"

    def test_save_snapshot_success_no_body(self, client):
        """save_snapshot should work with no request body (default description)."""
        snapshot_data = {"snapshot_id": "snap-002", "description": "", "created_at": "2026-04-01T00:00:00Z"}
        with patch.object(client.app.state.lifecycle, "has_network", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", return_value=snapshot_data):
            response = client.post("/v1/snapshots")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "success"

    def test_save_snapshot_no_network_returns_404(self, client):
        """save_snapshot should return 404 when no network exists."""
        with patch.object(client.app.state.lifecycle, "has_network", return_value=False):
            response = client.post("/v1/snapshots", json={"description": "no net"})
            assert response.status_code == 404
            assert "No network created" in response.json()["detail"]

    def test_save_snapshot_returns_none_gives_404(self, client):
        """save_snapshot should return 404 when save_snapshot returns None."""
        with patch.object(client.app.state.lifecycle, "has_network", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", return_value=None):
            response = client.post("/v1/snapshots", json={"description": "will fail"})
            assert response.status_code == 404
            assert "No network available to snapshot" in response.json()["detail"]

    def test_save_snapshot_runs_in_thread(self, client):
        """PERF-CC-01: save_snapshot must be invoked via asyncio.to_thread.

        Confirms the route handler offloads the blocking serializer.save_network
        call rather than calling lifecycle.save_snapshot synchronously on the
        event loop.
        """
        snapshot_data = {"snapshot_id": "snap-thread", "description": "", "created_at": "2026-04-01T00:00:00Z"}
        captured = {}

        def fake_save_snapshot(description: str = ""):
            # Record the thread name so we can confirm we're not on the
            # event-loop thread (FastAPI runs uvicorn's worker on
            # MainThread; asyncio.to_thread spawns a worker named
            # asyncio_*).
            import threading

            captured["thread"] = threading.current_thread().name
            return snapshot_data

        with patch.object(client.app.state.lifecycle, "has_network", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", side_effect=fake_save_snapshot):
            response = client.post("/v1/snapshots", json={"description": "thread check"})
            assert response.status_code == 200
            assert "thread" in captured, "save_snapshot was not invoked"
            assert captured["thread"] != "MainThread", f"save_snapshot ran on MainThread ({captured['thread']!r}); expected an asyncio worker"


# ---------------------------------------------------------------------------
# GET /v1/snapshots  (list_snapshots)
# ---------------------------------------------------------------------------


class TestListSnapshots:
    """Tests for list_snapshots endpoint."""

    def test_list_snapshots_empty(self, client):
        """list_snapshots should return an empty list when no snapshots exist."""
        with patch.object(client.app.state.lifecycle, "list_snapshots", return_value=[]):
            response = client.get("/v1/snapshots")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "success"
            assert body["data"] == []

    def test_list_snapshots_populated(self, client):
        """list_snapshots should return all available snapshots."""
        snapshots = [
            {"snapshot_id": "snap-001", "description": "first", "created_at": "2026-04-01T00:00:00Z"},
            {"snapshot_id": "snap-002", "description": "second", "created_at": "2026-04-01T01:00:00Z"},
        ]
        with patch.object(client.app.state.lifecycle, "list_snapshots", return_value=snapshots):
            response = client.get("/v1/snapshots")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "success"
            assert len(body["data"]) == 2
            assert body["data"][0]["snapshot_id"] == "snap-001"
            assert body["data"][1]["snapshot_id"] == "snap-002"


# ---------------------------------------------------------------------------
# GET /v1/snapshots/{snapshot_id}  (get_snapshot)
# ---------------------------------------------------------------------------


class TestGetSnapshot:
    """Tests for get_snapshot endpoint."""

    def test_get_snapshot_found(self, client):
        """get_snapshot should return snapshot metadata when found."""
        snapshot_data = {"snapshot_id": "snap-001", "description": "test", "created_at": "2026-04-01T00:00:00Z"}
        with patch.object(client.app.state.lifecycle, "get_snapshot", return_value=snapshot_data):
            response = client.get("/v1/snapshots/snap-001")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "success"
            assert body["data"]["snapshot_id"] == "snap-001"

    def test_get_snapshot_not_found_returns_404(self, client):
        """get_snapshot should return 404 when snapshot does not exist."""
        with patch.object(client.app.state.lifecycle, "get_snapshot", return_value=None):
            response = client.get("/v1/snapshots/nonexistent")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /v1/snapshots/{snapshot_id}/restore  (restore_snapshot)
# ---------------------------------------------------------------------------


class TestRestoreSnapshot:
    """Tests for restore_snapshot endpoint."""

    def test_restore_snapshot_success(self, client):
        """restore_snapshot should return success with restored status."""
        with patch.object(client.app.state.lifecycle, "load_snapshot", return_value=True):
            response = client.post("/v1/snapshots/snap-001/restore")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "success"
            assert body["data"]["snapshot_id"] == "snap-001"
            assert body["data"]["status"] == "restored"

    def test_restore_snapshot_surfaces_post_restore_training_params(self, client):
        """CAN-014 (Phase 6E Sprint A-5): restore response includes the
        post-restore ``training_params`` so a tuning UI can reconcile
        local state without an extra ``GET /v1/training/params`` call.

        Mocks ``get_training_params`` to a known shape (the test client
        starts without a network, so the real call would raise) — the
        contract under test is "the route surfaces whatever
        get_training_params returns under the ``training_params`` key,"
        not the precise field set, which is owned by the lifecycle."""
        fake_params = {
            "learning_rate": 0.005,
            "max_hidden_units": 25,
            "epochs_max": 1234,
            "max_iterations": 78,
            "output_epochs": 66,
            "init_output_weights": "random",
            "optimizer_type": "AdamW",
            "activation_function_name": "ReLU",
        }
        with patch.object(client.app.state.lifecycle, "load_snapshot", return_value=True), patch.object(client.app.state.lifecycle, "get_training_params", return_value=fake_params):
            response = client.post("/v1/snapshots/snap-with-params/restore")
            assert response.status_code == 200
            body = response.json()
            assert "training_params" in body["data"], "restore response is missing training_params (CAN-014 contract)"
            assert body["data"]["training_params"] == fake_params

    def test_restore_snapshot_falls_back_when_get_training_params_fails(self, client):
        """If ``get_training_params`` raises after a successful restore
        the route still returns 200 with the minimal payload — surfacing
        params is best-effort and must not undo a successful load."""
        with patch.object(client.app.state.lifecycle, "load_snapshot", return_value=True), patch.object(client.app.state.lifecycle, "get_training_params", side_effect=RuntimeError("boom")):
            response = client.post("/v1/snapshots/snap-fallback/restore")
            assert response.status_code == 200
            body = response.json()
            assert body["data"]["snapshot_id"] == "snap-fallback"
            assert body["data"]["status"] == "restored"
            # Defensive fallback: training_params is omitted rather than
            # the whole restore appearing failed.
            assert "training_params" not in body["data"]

    def test_restore_snapshot_not_found_returns_404(self, client):
        """restore_snapshot should return 404 when snapshot is not found."""
        with patch.object(client.app.state.lifecycle, "load_snapshot", return_value=False):
            response = client.post("/v1/snapshots/nonexistent/restore")
            assert response.status_code == 404
            assert "not found or failed to load" in response.json()["detail"]

    def test_restore_snapshot_load_fails_returns_404(self, client):
        """restore_snapshot should return 404 when load_snapshot fails (returns False)."""
        with patch.object(client.app.state.lifecycle, "load_snapshot", return_value=False):
            response = client.post("/v1/snapshots/snap-bad/restore")
            assert response.status_code == 404
            assert "not found or failed to load" in response.json()["detail"]

    def test_restore_snapshot_runs_in_thread(self, client):
        """PERF-CC-01: load_snapshot must be invoked via asyncio.to_thread."""
        captured = {}

        def fake_load_snapshot(snapshot_id: str):
            import threading

            captured["thread"] = threading.current_thread().name
            return True

        with patch.object(client.app.state.lifecycle, "load_snapshot", side_effect=fake_load_snapshot):
            response = client.post("/v1/snapshots/snap-thread/restore")
            assert response.status_code == 200
            assert captured["thread"] != "MainThread", f"load_snapshot ran on MainThread ({captured['thread']!r}); expected an asyncio worker"


class TestPerfCC01InvariantSource:
    """PERF-CC-01: lock asyncio.to_thread usage in route source."""

    def test_save_route_uses_to_thread(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[3] / "api" / "routes" / "snapshots.py"
        source = path.read_text(encoding="utf-8")
        # Both save and restore route handlers must offload via asyncio.to_thread.
        assert "asyncio.to_thread(lifecycle.save_snapshot" in source
        assert "asyncio.to_thread(lifecycle.load_snapshot" in source

    def test_imports_asyncio(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[3] / "api" / "routes" / "snapshots.py"
        source = path.read_text(encoding="utf-8")
        assert "import asyncio" in source
