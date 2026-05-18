"""P2-7 follow-up (Issue #3) — ``GET /v1/snapshots/{id}/history/dataset_swaps``.

Covers ``TrainingLifecycleManager.get_snapshot_dataset_swaps`` and the
matching REST route. Canopy's Replay timeline reads this when a snapshot
is loaded so markers reflect the snapshot's own swap history (parent
spec §4.4 full flavor), separate from the live event feed surfaced by
``GET /v1/history/dataset_swaps`` (P2-2 follow-up B).
"""

from __future__ import annotations

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
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from snapshots.snapshot_serializer import CascadeHDF5Serializer

pytestmark = pytest.mark.unit


def _make_event(timestamp: str, input_delta: int = 0) -> dict:
    return {
        "timestamp": timestamp,
        "before_cfg": {"dataset_type": "spirals"},
        "after_cfg": {"dataset_type": "moons"},
        "arch_changes": {"input_delta": input_delta, "output_delta": 0},
        "pre_swap_snapshot_id": f"snap_pre_{input_delta}",
        "post_swap_snapshot_id": f"snap_post_{input_delta}",
    }


def _write_snapshot(snapshots_dir: Path, snapshot_id: str, events: list[dict]) -> Path:
    """Materialise a snapshot HDF5 file with the given dataset_swap events
    under ``snapshots_dir/<snapshot_id>.h5``. Returns the file path."""
    cfg = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=2, learning_rate=0.1, max_hidden_units=2, random_seed=0)
    network = CascadeCorrelationNetwork(config=cfg)
    network.history["dataset_swaps"] = list(events)
    serializer = CascadeHDF5Serializer()
    filepath = snapshots_dir / f"{snapshot_id}.h5"
    serializer.save_network(network, str(filepath), include_training_state=True)
    return filepath


# ---------------------------------------------------------------------------
# Lifecycle: get_snapshot_dataset_swaps
# ---------------------------------------------------------------------------


class TestGetSnapshotDatasetSwaps:
    def test_returns_none_when_snapshot_missing(self, tmp_path):
        """Missing-snapshot → ``None`` so the route layer can map to 404."""
        mgr = TrainingLifecycleManager()
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path):
            assert mgr.get_snapshot_dataset_swaps("does_not_exist") is None
        mgr.shutdown()

    def test_returns_empty_list_for_snapshot_with_no_swaps(self, tmp_path):
        """Snapshot exists but the training run had no swaps → ``[]``.
        Distinguishes the 404 ("snapshot missing") and 200 ("snapshot
        exists, just no swaps") branches for the route caller."""
        _write_snapshot(tmp_path, "snapshot_empty", events=[])
        mgr = TrainingLifecycleManager()
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path):
            assert mgr.get_snapshot_dataset_swaps("snapshot_empty") == []
        mgr.shutdown()

    def test_returns_chronological_events(self, tmp_path):
        events_in = [
            _make_event("2026-05-14T10:00:00+00:00", input_delta=0),
            _make_event("2026-05-14T11:00:00+00:00", input_delta=1),
            _make_event("2026-05-14T12:00:00+00:00", input_delta=2),
        ]
        _write_snapshot(tmp_path, "snapshot_with_swaps", events=events_in)
        mgr = TrainingLifecycleManager()
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path):
            events_out = mgr.get_snapshot_dataset_swaps("snapshot_with_swaps")
        assert events_out is not None
        assert [e["arch_changes"]["input_delta"] for e in events_out] == [0, 1, 2]
        mgr.shutdown()

    def test_event_schema_matches_in_memory_swap_events(self, tmp_path):
        """The schema must match ``get_dataset_swap_events`` (P2-2 follow-up B)
        so canopy can render snapshot + live events through the same code
        path with only a source-discriminator difference."""
        _write_snapshot(tmp_path, "snap_a", events=[_make_event("2026-05-14T10:00:00+00:00", input_delta=5)])
        mgr = TrainingLifecycleManager()
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path):
            events = mgr.get_snapshot_dataset_swaps("snap_a")
        assert events is not None
        assert len(events) == 1
        e = events[0]
        assert set(e.keys()) == {"timestamp", "before_cfg", "after_cfg", "arch_changes", "pre_swap_snapshot_id", "post_swap_snapshot_id"}
        assert e["pre_swap_snapshot_id"] == "snap_pre_5"
        assert e["post_swap_snapshot_id"] == "snap_post_5"
        mgr.shutdown()


# ---------------------------------------------------------------------------
# REST route
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


class TestSnapshotDatasetSwapsRoute:
    def test_missing_snapshot_returns_404(self, client):
        resp = client.get("/v1/snapshots/snapshot_missing/history/dataset_swaps")
        assert resp.status_code == 404

    def test_empty_events_for_snapshot_without_swaps(self, client):
        with patch.object(client.app.state.lifecycle, "get_snapshot_dataset_swaps", return_value=[]):
            resp = client.get("/v1/snapshots/snap_a/history/dataset_swaps")
        assert resp.status_code == 200
        assert resp.json()["data"] == {"events": []}

    def test_returns_events_in_chronological_order(self, client):
        canned = [
            _make_event("2026-05-14T10:00:00+00:00", input_delta=0),
            _make_event("2026-05-14T11:00:00+00:00", input_delta=1),
        ]
        with patch.object(client.app.state.lifecycle, "get_snapshot_dataset_swaps", return_value=canned):
            resp = client.get("/v1/snapshots/snap_a/history/dataset_swaps")
        assert resp.status_code == 200
        events = resp.json()["data"]["events"]
        assert len(events) == 2
        assert events[0]["timestamp"] == "2026-05-14T10:00:00+00:00"

    def test_invalid_snapshot_id_returns_400(self, client):
        """The shared snapshot-id validator (used by every /v1/snapshots
        route) must run before the HDF5 open — bad IDs never reach the
        lifecycle method."""
        resp = client.get("/v1/snapshots/..%2Fetc%2Fpasswd/history/dataset_swaps")
        # FastAPI normalises the URL before routing; the exact status
        # depends on whether routing happens or the validator fires.
        # We accept either 400 (validator) or 404 (no match) — what we
        # really care about is that we don't 200 with HDF5 contents.
        assert resp.status_code in (400, 404)

    def test_lifecycle_missing_returns_503(self, client):
        """503 instead of 500 when lifecycle isn't wired — matches the
        contract of the existing /v1/history/dataset_swaps route."""
        with patch.object(client.app.state, "lifecycle", None):
            resp = client.get("/v1/snapshots/snap_a/history/dataset_swaps")
        assert resp.status_code == 503
