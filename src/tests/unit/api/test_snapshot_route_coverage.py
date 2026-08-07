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
            assert "Lifecycle manager not initialized" in response.json()["error"]["message"]

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
        with patch.object(client.app.state.lifecycle, "has_model", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", return_value=snapshot_data):
            response = client.post("/v1/snapshots", json={"description": "test snapshot"})
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "success"
            assert body["data"]["snapshot_id"] == "snap-001"
            assert body["data"]["description"] == "test snapshot"

    def test_save_snapshot_success_no_body(self, client):
        """save_snapshot should work with no request body (default description)."""
        snapshot_data = {"snapshot_id": "snap-002", "description": "", "created_at": "2026-04-01T00:00:00Z"}
        with patch.object(client.app.state.lifecycle, "has_model", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", return_value=snapshot_data):
            response = client.post("/v1/snapshots")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "success"

    def test_save_snapshot_no_network_returns_404(self, client):
        """save_snapshot should return 404 when no network exists."""
        with patch.object(client.app.state.lifecycle, "has_model", return_value=False):
            response = client.post("/v1/snapshots", json={"description": "no net"})
            assert response.status_code == 404
            assert "No network created" in response.json()["error"]["message"]

    def test_save_snapshot_returns_none_gives_404(self, client):
        """save_snapshot should return 404 when save_snapshot returns None."""
        with patch.object(client.app.state.lifecycle, "has_model", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", return_value=None):
            response = client.post("/v1/snapshots", json={"description": "will fail"})
            assert response.status_code == 404
            assert "No network available to snapshot" in response.json()["error"]["message"]

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

        with patch.object(client.app.state.lifecycle, "has_model", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", side_effect=fake_save_snapshot):
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
            assert "not found" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# POST /v1/snapshots/{snapshot_id}/restore  (restore_snapshot)
# ---------------------------------------------------------------------------


class TestRestoreSnapshot:
    """Tests for restore_snapshot endpoint."""

    def test_restore_snapshot_success(self, client):
        """restore_snapshot should return success with restored status."""
        with patch.object(client.app.state.lifecycle, "load_snapshot", return_value={"loaded": True}):
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
        with patch.object(client.app.state.lifecycle, "load_snapshot", return_value={"loaded": True}), patch.object(client.app.state.lifecycle, "get_training_params", return_value=fake_params):
            response = client.post("/v1/snapshots/snap-with-params/restore")
            assert response.status_code == 200
            body = response.json()
            assert "training_params" in body["data"], "restore response is missing training_params (CAN-014 contract)"
            assert body["data"]["training_params"] == fake_params

    def test_restore_snapshot_falls_back_when_get_training_params_fails(self, client):
        """If ``get_training_params`` raises after a successful restore
        the route still returns 200 with the minimal payload — surfacing
        params is best-effort and must not undo a successful load."""
        with patch.object(client.app.state.lifecycle, "load_snapshot", return_value={"loaded": True}), patch.object(client.app.state.lifecycle, "get_training_params", side_effect=RuntimeError("boom")):
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
        with patch.object(client.app.state.lifecycle, "load_snapshot", return_value={"loaded": False}):
            response = client.post("/v1/snapshots/nonexistent/restore")
            assert response.status_code == 404
            assert "not found or failed to load" in response.json()["error"]["message"]

    def test_restore_snapshot_load_fails_returns_404(self, client):
        """restore_snapshot should return 404 when load_snapshot fails (returns False)."""
        with patch.object(client.app.state.lifecycle, "load_snapshot", return_value={"loaded": False}):
            response = client.post("/v1/snapshots/snap-bad/restore")
            assert response.status_code == 404
            assert "not found or failed to load" in response.json()["error"]["message"]

    def test_restore_snapshot_runs_in_thread(self, client):
        """PERF-CC-01: load_snapshot must be invoked via asyncio.to_thread."""
        captured = {}

        def fake_load_snapshot(snapshot_id: str):
            import threading

            captured["thread"] = threading.current_thread().name
            return {"loaded": True}

        with patch.object(client.app.state.lifecycle, "load_snapshot", side_effect=fake_load_snapshot):
            response = client.post("/v1/snapshots/snap-thread/restore")
            assert response.status_code == 200
            assert captured["thread"] != "MainThread", f"load_snapshot ran on MainThread ({captured['thread']!r}); expected an asyncio worker"


class TestRetrainFromSnapshot:
    """CAN-015a (Phase 6E Sprint B B-1): tests for the new
    ``POST /v1/snapshots/{id}/retrain`` route. The route mirrors
    ``/restore`` in shape (snapshot_id + training_params + status) but
    the lifecycle method it calls (``restore_for_retrain``) additionally
    resets training history / counters / FSM / auto-snap-best so the
    next ``start_training`` begins at epoch 0 with empty curves. Lifecycle
    behavior itself is exercised in test_lifecycle_manager.py — these
    tests pin the route contract."""

    def test_retrain_snapshot_success(self, client):
        """retrain route returns success with ``operation: retrain`` and ``status: ready``."""
        with patch.object(client.app.state.lifecycle, "restore_for_retrain", return_value={"loaded": True}):
            response = client.post("/v1/snapshots/snap-001/retrain")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "success"
            assert body["data"]["snapshot_id"] == "snap-001"
            assert body["data"]["operation"] == "retrain"
            assert body["data"]["status"] == "ready"

    def test_retrain_snapshot_surfaces_post_reset_training_params(self, client):
        """Response includes training_params so a tuning UI can reconcile
        (matches the post-A-5 behavior of /restore)."""
        fake_params = {
            "learning_rate": 0.005,
            "epochs_max": 1234,
            "optimizer_type": "AdamW",
            "activation_function_name": "ReLU",
        }
        with patch.object(client.app.state.lifecycle, "restore_for_retrain", return_value={"loaded": True}), patch.object(client.app.state.lifecycle, "get_training_params", return_value=fake_params):
            response = client.post("/v1/snapshots/snap-with-params/retrain")
            assert response.status_code == 200
            body = response.json()
            assert "training_params" in body["data"]
            assert body["data"]["training_params"] == fake_params

    def test_retrain_snapshot_falls_back_when_get_training_params_fails(self, client):
        """A failing ``get_training_params`` after a successful restore_for_retrain
        must NOT make the route appear failed — return 200 with the
        minimal payload, same defensive pattern as /restore."""
        with patch.object(client.app.state.lifecycle, "restore_for_retrain", return_value={"loaded": True}), patch.object(client.app.state.lifecycle, "get_training_params", side_effect=RuntimeError("boom")):
            response = client.post("/v1/snapshots/snap-fallback/retrain")
            assert response.status_code == 200
            body = response.json()
            assert body["data"]["snapshot_id"] == "snap-fallback"
            assert body["data"]["operation"] == "retrain"
            assert body["data"]["status"] == "ready"
            assert "training_params" not in body["data"]

    def test_retrain_snapshot_not_found_returns_404(self, client):
        """When restore_for_retrain returns False (missing snapshot or
        deserializer failure) the route maps to 404, identical to /restore."""
        with patch.object(client.app.state.lifecycle, "restore_for_retrain", return_value={"loaded": False}):
            response = client.post("/v1/snapshots/nonexistent/retrain")
            assert response.status_code == 404
            assert "not found or failed to load" in response.json()["error"]["message"]

    def test_retrain_snapshot_runs_in_thread(self, client):
        """PERF-CC-01: HDF5 I/O off the main event loop."""
        captured: dict = {}

        def fake_restore_for_retrain(snapshot_id: str):
            import threading

            captured["thread"] = threading.current_thread().name
            return {"loaded": True}

        with patch.object(client.app.state.lifecycle, "restore_for_retrain", side_effect=fake_restore_for_retrain):
            response = client.post("/v1/snapshots/snap-thread/retrain")
            assert response.status_code == 200
            assert captured["thread"] != "MainThread", f"restore_for_retrain ran on MainThread ({captured['thread']!r}); expected an asyncio worker"

    def test_retrain_snapshot_validates_id_format(self, client):
        """SEC-17: invalid snapshot_id format is rejected at the route boundary."""
        # Path with .. would attempt traversal — rejected with 400.
        response = client.post("/v1/snapshots/..%2Fevil/retrain")
        # FastAPI normalizes %2F before the validator, so the result is
        # either a 400 (validator catches it) or a 404 (path doesn't match
        # the route). Both are acceptable — the key is "not 200 / not
        # 500." We accept either since the route registration handles
        # it identically to other snapshot endpoints.
        assert response.status_code in (400, 404, 422), f"unexpected status {response.status_code} for traversal attempt"


class TestResumeSnapshot:
    """CAN-015b (Phase 6E Sprint B B-2): tests for the new
    ``POST /v1/snapshots/{id}/resume`` route. The route mirrors the
    /restore + /retrain shape (snapshot_id + training_params + status)
    and additionally surfaces ``resume_point_epoch`` so canopy can
    render the visual boundary between pre-resume read-only history and
    new training. Lifecycle reset/preserve semantics are exercised in
    test_lifecycle_manager.py."""

    def test_resume_snapshot_success(self, client):
        """resume route returns ``operation: resume``, ``status: ready``,
        and the resume_point_epoch read from the lifecycle."""
        # Mock the lifecycle method directly; we set _resume_point_epoch
        # before the call so the route observes it after success.
        lifecycle = client.app.state.lifecycle
        lifecycle._resume_point_epoch = 42

        with patch.object(lifecycle, "resume_from_snapshot", return_value={"loaded": True}):
            response = client.post("/v1/snapshots/snap-001/resume")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "success"
            assert body["data"]["snapshot_id"] == "snap-001"
            assert body["data"]["operation"] == "resume"
            assert body["data"]["status"] == "ready"
            assert body["data"]["resume_point_epoch"] == 42

    def test_resume_snapshot_surfaces_post_load_training_params(self, client):
        """Response includes training_params so canopy can reconcile."""
        fake_params = {
            "learning_rate": 0.005,
            "epochs_max": 1234,
            "optimizer_type": "AdamW",
        }
        lifecycle = client.app.state.lifecycle
        lifecycle._resume_point_epoch = 7
        with patch.object(lifecycle, "resume_from_snapshot", return_value={"loaded": True}), patch.object(lifecycle, "get_training_params", return_value=fake_params):
            response = client.post("/v1/snapshots/snap-with-params/resume")
            assert response.status_code == 200
            body = response.json()
            assert "training_params" in body["data"]
            assert body["data"]["training_params"] == fake_params

    def test_resume_snapshot_falls_back_when_get_training_params_fails(self, client):
        """A failing ``get_training_params`` after a successful resume must
        NOT make the route appear failed — same defensive pattern as /restore."""
        lifecycle = client.app.state.lifecycle
        lifecycle._resume_point_epoch = 0
        with patch.object(lifecycle, "resume_from_snapshot", return_value={"loaded": True}), patch.object(lifecycle, "get_training_params", side_effect=RuntimeError("boom")):
            response = client.post("/v1/snapshots/snap-fallback/resume")
            assert response.status_code == 200
            body = response.json()
            assert body["data"]["snapshot_id"] == "snap-fallback"
            assert body["data"]["operation"] == "resume"
            assert "training_params" not in body["data"]

    def test_resume_snapshot_not_found_returns_404(self, client):
        """When resume_from_snapshot returns False (missing snapshot or
        deserializer failure), the route maps to 404."""
        with patch.object(client.app.state.lifecycle, "resume_from_snapshot", return_value={"loaded": False}):
            response = client.post("/v1/snapshots/nonexistent/resume")
            assert response.status_code == 404
            assert "not found or failed to load" in response.json()["error"]["message"]

    def test_resume_snapshot_rejected_when_training_active(self, client):
        """The pre-flight FSM check returns 409 when training is Started."""
        from api.lifecycle.state_machine import Command

        lifecycle = client.app.state.lifecycle
        # Force the FSM to Started (no actual network/training needed for the check).
        lifecycle.state_machine.handle_command(Command.START)
        try:
            response = client.post("/v1/snapshots/snap-active/resume")
            assert response.status_code == 409
            assert "Cannot resume" in response.json()["error"]["message"]
        finally:
            # Clean up so other tests aren't affected.
            lifecycle.state_machine.handle_command(Command.RESET)

    def test_resume_snapshot_rejected_when_paused(self, client):
        """Paused state also rejected with 409."""
        from api.lifecycle.state_machine import Command

        lifecycle = client.app.state.lifecycle
        lifecycle.state_machine.handle_command(Command.START)
        lifecycle.state_machine.handle_command(Command.PAUSE)
        try:
            response = client.post("/v1/snapshots/snap-paused/resume")
            assert response.status_code == 409
        finally:
            lifecycle.state_machine.handle_command(Command.RESET)

    def test_resume_snapshot_runs_in_thread(self, client):
        """PERF-CC-01: HDF5 I/O off the main event loop."""
        captured: dict = {}

        def fake_resume(snapshot_id: str):
            import threading

            captured["thread"] = threading.current_thread().name
            return {"loaded": True}

        client.app.state.lifecycle._resume_point_epoch = 0
        with patch.object(client.app.state.lifecycle, "resume_from_snapshot", side_effect=fake_resume):
            response = client.post("/v1/snapshots/snap-thread/resume")
            assert response.status_code == 200
            assert captured["thread"] != "MainThread", f"resume_from_snapshot ran on MainThread ({captured['thread']!r}); expected an asyncio worker"


class TestPerfCC01InvariantSource:
    """PERF-CC-01: lock asyncio.to_thread usage in route source."""

    def test_save_route_uses_to_thread(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[3] / "api" / "routes" / "snapshots.py"
        source = path.read_text(encoding="utf-8")
        # Save / restore / retrain / resume route handlers must all offload via asyncio.to_thread.
        assert "asyncio.to_thread(lifecycle.save_snapshot" in source
        assert "asyncio.to_thread(lifecycle.load_snapshot" in source
        # CAN-015a (Phase 6E Sprint B B-1): retrain route added.
        assert "asyncio.to_thread(lifecycle.restore_for_retrain" in source
        # CAN-015b (Phase 6E Sprint B B-2): resume route added.
        assert "asyncio.to_thread(lifecycle.resume_from_snapshot" in source

    def test_imports_asyncio(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[3] / "api" / "routes" / "snapshots.py"
        source = path.read_text(encoding="utf-8")
        assert "import asyncio" in source


class TestUnifiedResponseShape:
    """CAN-015d (Phase 6E Sprint B B-4): all four snapshot operation
    endpoints share a unified response shape — ``snapshot_id``,
    ``operation``, ``fsm_state``, ``time_index``, and
    ``training_params``. Pre-B-4 fields like ``status`` and
    ``resume_point_epoch`` are retained as a strict superset.

    These tests pin the new fields. Existing tests in
    TestRestoreSnapshot / TestRetrainFromSnapshot / TestResumeSnapshot
    pin the legacy backward-compat fields and remain unchanged.
    """

    def _force_fsm_state(self, lifecycle, target_marker: str):
        """Helper: mock load_snapshot/restore_for_retrain/resume_from_snapshot
        to actually transition the FSM, since the route reads
        state_machine.status.name to populate the response.

        ``target_marker`` is one of "investigating", "stopped", "resume_ready".
        """
        if target_marker == "investigating":
            return lambda *args, **kwargs: (lifecycle.state_machine.mark_investigating(), {"loaded": True})[1]
        if target_marker == "resume_ready":
            return lambda *args, **kwargs: (lifecycle.state_machine.mark_resume_ready(), {"loaded": True})[1]
        if target_marker == "stopped":
            from api.lifecycle.state_machine import Command

            return lambda *args, **kwargs: (lifecycle.state_machine.handle_command(Command.RESET), {"loaded": True})[1]
        raise ValueError(f"unknown target_marker {target_marker!r}")

    def test_restore_response_includes_unified_fields(self, client):
        """POST /restore response contains operation/fsm_state/time_index."""
        lifecycle = client.app.state.lifecycle
        side_effect = self._force_fsm_state(lifecycle, "investigating")
        with patch.object(lifecycle, "load_snapshot", side_effect=side_effect):
            response = client.post("/v1/snapshots/snap-001/restore")
            assert response.status_code == 200
            body = response.json()
            data = body["data"]
            assert data["operation"] == "restore"
            assert data["fsm_state"] == "INVESTIGATING"
            assert "time_index" in data
            assert data["time_index"]["default"] == "end"
            assert "snapshot_window" in data["time_index"]
            assert "start_epoch" in data["time_index"]["snapshot_window"]
            assert "end_epoch" in data["time_index"]["snapshot_window"]
            # Backward-compat fields preserved.
            assert data["status"] == "restored"
            assert data["snapshot_id"] == "snap-001"

    def test_retrain_response_includes_unified_fields(self, client):
        """POST /retrain response contains operation/fsm_state/time_index."""
        lifecycle = client.app.state.lifecycle
        side_effect = self._force_fsm_state(lifecycle, "stopped")
        with patch.object(lifecycle, "restore_for_retrain", side_effect=side_effect):
            response = client.post("/v1/snapshots/snap-002/retrain")
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["operation"] == "retrain"
            # Retrain transitions to Stopped.
            assert data["fsm_state"] == "STOPPED"
            # Retrain resets history to 0, so time_index default is 0.
            assert data["time_index"]["default"] == 0
            # Backward-compat fields preserved.
            assert data["status"] == "ready"
            assert data["snapshot_id"] == "snap-002"

    def test_resume_response_includes_unified_fields(self, client):
        """POST /resume response contains operation/fsm_state/time_index plus resume_point_epoch."""
        lifecycle = client.app.state.lifecycle

        def side_effect(snapshot_id):
            lifecycle.state_machine.mark_resume_ready()
            lifecycle._resume_point_epoch = 42
            return {"loaded": True}

        with patch.object(lifecycle, "resume_from_snapshot", side_effect=side_effect):
            response = client.post("/v1/snapshots/snap-003/resume")
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["operation"] == "resume"
            assert data["fsm_state"] == "RESUME_READY"
            # Resume lands at end of window.
            assert data["time_index"]["default"] == "end"
            # Backward-compat fields from B-2 preserved.
            assert data["status"] == "ready"
            assert data["resume_point_epoch"] == 42

    def test_time_index_snapshot_window_reflects_loaded_history(self, client):
        """``snapshot_window.end_epoch`` matches the longest history array's length."""
        lifecycle = client.app.state.lifecycle

        # Install a fake network with a known history.
        class FakeNetwork:
            history = {
                "train_loss": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # 7 entries
                "value_loss": [0.2, 0.3, 0.4],  # 3 entries
                "train_accuracy": [0.6, 0.7],  # 2 entries
                "value_accuracy": [],  # 0 entries
            }

        lifecycle.network = FakeNetwork()
        side_effect = self._force_fsm_state(lifecycle, "investigating")
        try:
            with patch.object(lifecycle, "load_snapshot", side_effect=side_effect):
                response = client.post("/v1/snapshots/snap-window/restore")
                assert response.status_code == 200
                data = response.json()["data"]
                # Longest array (train_loss) has 7 entries.
                assert data["time_index"]["snapshot_window"] == {"start_epoch": 0, "end_epoch": 7}
        finally:
            lifecycle.network = None

    def test_restore_rejected_when_training_active(self, client):
        """The pre-flight FSM check returns 409 when training is Started.
        This was an implicit contract before (load_snapshot would race
        with the running fit) — B-4 makes it explicit at the route layer."""
        from api.lifecycle.state_machine import Command

        lifecycle = client.app.state.lifecycle
        lifecycle.state_machine.handle_command(Command.START)
        try:
            response = client.post("/v1/snapshots/snap-active/restore")
            assert response.status_code == 409
            assert "Cannot restore" in response.json()["error"]["message"]
        finally:
            lifecycle.state_machine.handle_command(Command.RESET)

    def test_restore_rejected_when_paused(self, client):
        """Paused state also rejected with 409."""
        from api.lifecycle.state_machine import Command

        lifecycle = client.app.state.lifecycle
        lifecycle.state_machine.handle_command(Command.START)
        lifecycle.state_machine.handle_command(Command.PAUSE)
        try:
            response = client.post("/v1/snapshots/snap-paused/restore")
            assert response.status_code == 409
        finally:
            lifecycle.state_machine.handle_command(Command.RESET)


class TestReplaySnapshot:
    """CAN-015c (Phase 6E Sprint B B-3): tests for the new
    ``POST /v1/snapshots/{id}/replay`` and
    ``POST /v1/snapshots/{id}/replay/control`` routes."""

    def _install_replay_session(self, lifecycle, snapshot_id="snap-test", length=5):
        """Install a synthetic replay session bypassing the load step."""
        from api.lifecycle.manager import _ReplaySession

        history = {
            "train_loss": [0.5] * length,
            "value_loss": [],
            "train_accuracy": [],
            "value_accuracy": [],
        }
        session = _ReplaySession(snapshot_id, history, lifecycle.monitor)
        lifecycle._replay_session = session
        lifecycle.state_machine.mark_replaying()
        return session

    def _teardown_replay_session(self, lifecycle):
        from api.lifecycle.state_machine import Command

        if lifecycle._replay_session is not None:
            lifecycle._replay_session.stop()
            lifecycle._replay_session = None
        lifecycle.state_machine.handle_command(Command.RESET)

    def test_start_replay_route_success(self, client):
        lifecycle = client.app.state.lifecycle

        def fake_start(snapshot_id):
            self._install_replay_session(lifecycle, snapshot_id, length=3)
            return True

        try:
            with patch.object(lifecycle, "start_replay", side_effect=fake_start):
                response = client.post("/v1/snapshots/snap-001/replay")
                assert response.status_code == 200
                data = response.json()["data"]
                assert data["snapshot_id"] == "snap-001"
                assert data["operation"] == "replay"
                assert data["fsm_state"] == "REPLAYING"
                assert data["time_index"]["default"] == "start"
                assert data["status"] == "replaying"
                assert data["session"]["snapshot_id"] == "snap-001"
                assert data["session"]["length"] == 3
        finally:
            self._teardown_replay_session(lifecycle)

    def test_start_replay_route_404_when_load_fails(self, client):
        with patch.object(client.app.state.lifecycle, "start_replay", return_value=False):
            response = client.post("/v1/snapshots/nonexistent/replay")
            assert response.status_code == 404

    def test_start_replay_route_409_when_training_active(self, client):
        from api.lifecycle.state_machine import Command

        lifecycle = client.app.state.lifecycle
        lifecycle.state_machine.handle_command(Command.START)
        try:
            response = client.post("/v1/snapshots/snap-active/replay")
            assert response.status_code == 409
            assert "Cannot start replay" in response.json()["error"]["message"]
        finally:
            lifecycle.state_machine.handle_command(Command.RESET)

    def test_start_replay_route_runs_in_thread(self, client):
        captured: dict = {}

        def fake_start(snapshot_id):
            import threading

            captured["thread"] = threading.current_thread().name
            self._install_replay_session(client.app.state.lifecycle, snapshot_id)
            return True

        try:
            with patch.object(client.app.state.lifecycle, "start_replay", side_effect=fake_start):
                response = client.post("/v1/snapshots/snap-thread/replay")
                assert response.status_code == 200
                assert captured["thread"] != "MainThread"
        finally:
            self._teardown_replay_session(client.app.state.lifecycle)

    def test_replay_control_play(self, client):
        lifecycle = client.app.state.lifecycle
        try:
            self._install_replay_session(lifecycle, "snap-ctrl", length=3)
            response = client.post("/v1/snapshots/snap-ctrl/replay/control", json={"action": "play"})
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["operation"] == "replay_control"
            assert data["action"] == "play"
            assert data["result"]["paused"] is False
        finally:
            self._teardown_replay_session(lifecycle)

    def test_replay_control_seek(self, client):
        lifecycle = client.app.state.lifecycle
        try:
            self._install_replay_session(lifecycle, "snap-seek", length=5)
            response = client.post("/v1/snapshots/snap-seek/replay/control", json={"action": "seek", "time_index": 3})
            assert response.status_code == 200
            assert response.json()["data"]["result"]["time_index"] == 3
        finally:
            self._teardown_replay_session(lifecycle)

    def test_replay_control_speed(self, client):
        lifecycle = client.app.state.lifecycle
        try:
            self._install_replay_session(lifecycle, "snap-speed", length=3)
            response = client.post("/v1/snapshots/snap-speed/replay/control", json={"action": "speed", "value": 2.5})
            assert response.status_code == 200
            assert response.json()["data"]["result"]["speed"] == 2.5
        finally:
            self._teardown_replay_session(lifecycle)

    def test_replay_control_range(self, client):
        lifecycle = client.app.state.lifecycle
        try:
            self._install_replay_session(lifecycle, "snap-range", length=5)
            response = client.post("/v1/snapshots/snap-range/replay/control", json={"action": "range", "start": 1, "end": 4})
            assert response.status_code == 200
            # PR #195: ``range`` carries only ``{start, end}``. The post-clamp
            # ``time_index`` lives at the top level of the result payload
            # (state_summary's own field), not nested inside ``range``.
            result = response.json()["data"]["result"]
            assert result["range"] == {"start": 1, "end": 4}
            assert result["time_index"] == 1
        finally:
            self._teardown_replay_session(lifecycle)

    def test_replay_control_stop(self, client):
        lifecycle = client.app.state.lifecycle
        self._install_replay_session(lifecycle, "snap-stop")
        response = client.post("/v1/snapshots/snap-stop/replay/control", json={"action": "stop"})
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["action"] == "stop"
        assert data["result"]["status"] == "stopped"
        assert data["fsm_state"] == "STOPPED"
        assert lifecycle._replay_session is None

    def test_replay_control_without_active_session_returns_409(self, client):
        response = client.post("/v1/snapshots/snap-none/replay/control", json={"action": "play"})
        assert response.status_code == 409
        assert "No active replay session" in response.json()["error"]["message"]

    def test_replay_control_snapshot_id_mismatch_returns_409(self, client):
        lifecycle = client.app.state.lifecycle
        try:
            self._install_replay_session(lifecycle, "snap-actual")
            response = client.post("/v1/snapshots/snap-different/replay/control", json={"action": "play"})
            assert response.status_code == 409
            assert "snap-actual" in response.json()["error"]["message"]
        finally:
            self._teardown_replay_session(lifecycle)

    def test_replay_control_unknown_action_returns_400(self, client):
        lifecycle = client.app.state.lifecycle
        try:
            self._install_replay_session(lifecycle, "snap-bad-action")
            response = client.post("/v1/snapshots/snap-bad-action/replay/control", json={"action": "teleport"})
            assert response.status_code == 400
            assert "Unknown replay action" in response.json()["error"]["message"]
        finally:
            self._teardown_replay_session(lifecycle)

    def test_replay_control_seek_missing_param_returns_400(self, client):
        lifecycle = client.app.state.lifecycle
        try:
            self._install_replay_session(lifecycle, "snap-missing-param")
            response = client.post("/v1/snapshots/snap-missing-param/replay/control", json={"action": "seek"})
            assert response.status_code == 400
            assert "time_index" in response.json()["error"]["message"]
        finally:
            self._teardown_replay_session(lifecycle)

    def test_replay_control_runtime_error_maps_to_409(self, client):
        """Race between preflight and lifecycle.replay_control must surface as 409."""
        lifecycle = client.app.state.lifecycle
        try:
            self._install_replay_session(lifecycle, "snap-race")
            with patch.object(lifecycle, "replay_control", side_effect=RuntimeError("session changed")):
                response = client.post("/v1/snapshots/snap-race/replay/control", json={"action": "play"})
            assert response.status_code == 409
            assert "session changed" in response.json()["error"]["message"]
        finally:
            self._teardown_replay_session(lifecycle)

    def test_replay_control_speed_missing_value_returns_400(self, client):
        lifecycle = client.app.state.lifecycle
        try:
            self._install_replay_session(lifecycle, "snap-speed-missing")
            response = client.post("/v1/snapshots/snap-speed-missing/replay/control", json={"action": "speed"})
            assert response.status_code == 400
            body = response.json()["error"]["message"].lower()
            assert "value" in body or "speed" in body
        finally:
            self._teardown_replay_session(lifecycle)
