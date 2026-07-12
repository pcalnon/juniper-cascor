"""C1 (I-3 upstream half) — snapshot-create correctness + error propagation.

Plan of record: juniper-ml
``notes/JUNIPER_2026-07-11_JUNIPER-CANOPY_TRAINING-RUNTIME-DEFECTS-PLAN.md``
§4 I-3 / §7 C1.

Pins the corrected route/lifecycle contracts:

* ``SnapshotCreateRequest.description`` tolerates an explicit JSON ``null``
  (live incident 2026-07-11: canopy's route seam posts ``{"description": null}``
  for a blank description and cascor 422'd it with a ``string_type``
  rejection). Omission, ``null``, and a plain string are all accepted;
  ``null`` normalizes to ``""``.
* A FAILED SAVE surfaces as 500 with the serializer's reason in the detail —
  no longer collapsed into the 404 "No network available to snapshot" that
  made a disk/HDF5 failure masquerade as a missing network.
* The no-network cases remain 404 (both the ``has_model`` pre-check and a
  ``None`` result from the lifecycle).
* Lifecycle level: ``save_snapshot`` raises ``SnapshotSaveError`` (reason
  attached) on write failure and returns ``None`` only when there is no
  network.

All snapshot writes in this module go to pytest ``tmp_path`` —
``src/snapshots/`` holds committed .h5 artifacts and must never be written
by tests.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.app import create_app
from api.lifecycle.manager import TrainingLifecycleManager
from api.settings import Settings
from snapshots.snapshot_errors import SnapshotSaveError

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    """Create a test client with lifecycle manager (lifespan runs)."""
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


SNAPSHOT_OK = {"id": "snap-c1", "path": "snap-c1.h5", "timestamp": "20260711T000000Z", "description": ""}


# ---------------------------------------------------------------------------
# POST /v1/snapshots — description shapes (explicit null / omission / string)
# ---------------------------------------------------------------------------


class TestCreateDescriptionShapes:
    """Explicit ``null``, omission, and a plain string must all be accepted."""

    def test_explicit_null_description_is_accepted(self, client):
        """Regression for the live 2026-07-11 incident: ``{"description": null}``
        must not 422 — it normalizes to the empty description."""
        captured = {}

        def fake_save(description: str = ""):
            captured["description"] = description
            return dict(SNAPSHOT_OK)

        with patch.object(client.app.state.lifecycle, "has_model", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", side_effect=fake_save):
            response = client.post("/v1/snapshots", json={"description": None})
            assert response.status_code == 200, f"explicit-null description rejected: {response.text}"
            assert captured["description"] == "", "explicit null must normalize to the empty description"

    def test_omitted_description_still_works(self, client):
        captured = {}

        def fake_save(description: str = ""):
            captured["description"] = description
            return dict(SNAPSHOT_OK)

        with patch.object(client.app.state.lifecycle, "has_model", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", side_effect=fake_save):
            response = client.post("/v1/snapshots", json={})
            assert response.status_code == 200, response.text
            assert captured["description"] == ""

    def test_string_description_still_works(self, client):
        captured = {}

        def fake_save(description: str = ""):
            captured["description"] = description
            return dict(SNAPSHOT_OK)

        with patch.object(client.app.state.lifecycle, "has_model", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", side_effect=fake_save):
            response = client.post("/v1/snapshots", json={"description": "before the big change"})
            assert response.status_code == 200, response.text
            assert captured["description"] == "before the big change"

    def test_no_body_still_works(self, client):
        with patch.object(client.app.state.lifecycle, "has_model", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", return_value=dict(SNAPSHOT_OK)):
            response = client.post("/v1/snapshots")
            assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# POST /v1/snapshots — failure-status separation (500 with reason vs 404)
# ---------------------------------------------------------------------------


class TestCreateFailureStatusSeparation:
    """A failed WRITE is a 500 carrying the reason; no-network stays 404."""

    def test_failed_save_returns_500_with_reason(self, client):
        with patch.object(client.app.state.lifecycle, "has_model", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", side_effect=SnapshotSaveError("OSError: [Errno 28] No space left on device")):
            response = client.post("/v1/snapshots", json={"description": "will fail"})
            assert response.status_code == 500, f"failed save must be 500, got {response.status_code}: {response.text}"
            message = response.json()["error"]["message"]
            assert "Snapshot save failed" in message
            assert "No space left on device" in message, f"detail must carry the underlying reason, got: {message!r}"

    def test_failed_save_is_not_the_no_network_404(self, client):
        """The pre-C1 defect: a failed save masqueraded as a missing network."""
        with patch.object(client.app.state.lifecycle, "has_model", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", side_effect=SnapshotSaveError("boom")):
            response = client.post("/v1/snapshots", json={"description": "x"})
            assert response.status_code != 404
            assert "No network available to snapshot" not in response.text

    def test_no_network_precheck_stays_404(self, client):
        with patch.object(client.app.state.lifecycle, "has_model", return_value=False):
            response = client.post("/v1/snapshots", json={"description": "no net"})
            assert response.status_code == 404
            assert "No network created" in response.json()["error"]["message"]

    def test_lifecycle_none_result_stays_404(self, client):
        """``None`` now strictly means "no network at save time" — still 404."""
        with patch.object(client.app.state.lifecycle, "has_model", return_value=True), patch.object(client.app.state.lifecycle, "save_snapshot", return_value=None):
            response = client.post("/v1/snapshots", json={"description": "raced away"})
            assert response.status_code == 404
            assert "No network available to snapshot" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# End-to-end through the real lifecycle + serializer (writes to tmp_path)
# ---------------------------------------------------------------------------


class TestCreateEndToEnd:
    """Full route → lifecycle → serializer path with a real tiny network."""

    def test_null_description_creates_real_snapshot_in_tmp_path(self, client, tmp_path):
        lifecycle = client.app.state.lifecycle
        lifecycle.create_network(input_size=2, output_size=2)
        original_dir = lifecycle._get_snapshots_dir
        lifecycle._get_snapshots_dir = lambda: tmp_path  # type: ignore[method-assign]
        try:
            response = client.post("/v1/snapshots", json={"description": None})
            assert response.status_code == 200, response.text
            data = response.json()["data"]
            assert data["description"] == ""
            written = Path(data["path"])
            assert written.parent == tmp_path, f"snapshot must be written to tmp_path, not {written.parent}"
            assert written.exists()
        finally:
            lifecycle._get_snapshots_dir = original_dir  # type: ignore[method-assign]

    def test_real_write_failure_surfaces_500_with_reason(self, client, tmp_path):
        """An uncreatable snapshots dir (parent is a file) drives the real
        serializer failure path end-to-end: 500 + reason, not 404."""
        lifecycle = client.app.state.lifecycle
        lifecycle.create_network(input_size=2, output_size=2)
        blocker = tmp_path / "blocker"
        blocker.write_text("a file where a directory must go")
        original_dir = lifecycle._get_snapshots_dir
        lifecycle._get_snapshots_dir = lambda: blocker / "nested"  # type: ignore[method-assign]
        try:
            response = client.post("/v1/snapshots", json={"description": "doomed"})
            assert response.status_code == 500, f"expected 500 for a real write failure, got {response.status_code}: {response.text}"
            message = response.json()["error"]["message"]
            assert "Snapshot save failed" in message
        finally:
            lifecycle._get_snapshots_dir = original_dir  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Lifecycle-level contract (save_snapshot)
# ---------------------------------------------------------------------------


class TestLifecycleSaveSnapshotErrorContract:
    """``save_snapshot``: ``None`` only for no-network; failures raise with reason."""

    @pytest.fixture
    def mgr(self, tmp_path):
        m = TrainingLifecycleManager()
        m.create_network(input_size=2, output_size=2)
        m._get_snapshots_dir = lambda: tmp_path  # type: ignore[method-assign]
        try:
            yield m
        finally:
            m.shutdown()

    def test_no_network_returns_none(self):
        m = TrainingLifecycleManager()
        try:
            assert m.network is None
            assert m.save_snapshot(description="x") is None
        finally:
            m.shutdown()

    def test_serializer_exception_propagates_with_reason(self, mgr):
        with patch("snapshots.snapshot_serializer.CascadeHDF5Serializer.save_network", side_effect=SnapshotSaveError("ValueError: boom")):
            with pytest.raises(SnapshotSaveError, match="boom"):
                mgr.save_snapshot(description="x")

    def test_real_write_failure_raises_with_underlying_cause(self, mgr, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        mgr._get_snapshots_dir = lambda: blocker / "nested"  # type: ignore[method-assign]
        with pytest.raises(SnapshotSaveError) as excinfo:
            mgr.save_snapshot(description="x")
        assert excinfo.value.__cause__ is not None, "the underlying OS error must be chained"

    def test_success_path_unchanged(self, mgr, tmp_path):
        result = mgr.save_snapshot(description="all good")
        assert result is not None
        assert result["description"] == "all good"
        assert Path(result["path"]).parent == tmp_path
        assert Path(result["path"]).exists()
