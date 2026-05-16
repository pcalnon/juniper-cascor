"""P2-2 Follow-up B (Issue #3) — ``GET /v1/history/dataset_swaps`` route + lifecycle.

Covers the lifecycle method ``get_dataset_swap_events(since=None)`` and the
new ``/v1/history/dataset_swaps`` REST surface. Canopy P2-7's timeline UI
reads dataset_swap events via this route without fetching a full HDF5
snapshot.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.app import create_app
from api.lifecycle.manager import TrainingLifecycleManager
from api.settings import Settings

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Lifecycle: get_dataset_swap_events
# ---------------------------------------------------------------------------


def _make_event(timestamp: str, input_delta: int = 0) -> dict:
    return {
        "timestamp": timestamp,
        "before_cfg": {"dataset_type": "spirals"},
        "after_cfg": {"dataset_type": "moons"},
        "arch_changes": {"input_delta": input_delta, "output_delta": 0},
        "pre_swap_snapshot_id": f"snap_pre_{input_delta}",
        "post_swap_snapshot_id": f"snap_post_{input_delta}",
    }


class TestGetDatasetSwapEvents:
    def test_returns_empty_when_no_network(self):
        """No network installed → empty list (not None, not raising)."""
        mgr = TrainingLifecycleManager()
        assert mgr.network is None
        assert mgr.get_dataset_swap_events() == []
        mgr.shutdown()

    def test_returns_empty_when_history_has_no_swaps(self):
        """Network with empty dataset_swaps list → empty list."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        assert mgr.get_dataset_swap_events() == []
        mgr.shutdown()

    def test_returns_all_events_when_no_since_filter(self):
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.network.history["dataset_swaps"] = [
            _make_event("2026-05-14T10:00:00+00:00", input_delta=0),
            _make_event("2026-05-14T11:00:00+00:00", input_delta=1),
            _make_event("2026-05-14T12:00:00+00:00", input_delta=2),
        ]
        events = mgr.get_dataset_swap_events()
        assert len(events) == 3
        assert [e["arch_changes"]["input_delta"] for e in events] == [0, 1, 2]
        mgr.shutdown()

    def test_returns_only_events_strictly_after_since(self):
        """``since`` is exclusive — events with timestamp equal to ``since``
        are NOT returned (lets a poller pass the last-seen timestamp and
        get only strictly-newer events)."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.network.history["dataset_swaps"] = [
            _make_event("2026-05-14T10:00:00+00:00", input_delta=0),
            _make_event("2026-05-14T11:00:00+00:00", input_delta=1),
            _make_event("2026-05-14T12:00:00+00:00", input_delta=2),
        ]
        events = mgr.get_dataset_swap_events(since="2026-05-14T11:00:00+00:00")
        assert [e["arch_changes"]["input_delta"] for e in events] == [2]
        mgr.shutdown()

    def test_since_filter_skips_events_without_timestamp(self):
        """An event with no string timestamp (e.g. loaded from a malformed
        snapshot) is excluded when ``since`` is set. Including such events
        could cause a poller to repeatedly re-fetch them."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.network.history["dataset_swaps"] = [
            {**_make_event("2026-05-14T10:00:00+00:00"), "timestamp": None},
            _make_event("2026-05-14T11:00:00+00:00", input_delta=5),
        ]
        events = mgr.get_dataset_swap_events(since="2026-05-14T09:00:00+00:00")
        assert len(events) == 1
        assert events[0]["arch_changes"]["input_delta"] == 5
        mgr.shutdown()

    def test_returned_list_is_a_copy(self):
        """Mutating the returned events MUST NOT affect persisted history.
        Without the copy, a caller iterating the list could accidentally
        corrupt the canonical record."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr.network.history["dataset_swaps"] = [_make_event("2026-05-14T10:00:00+00:00")]
        events = mgr.get_dataset_swap_events()
        events[0]["arch_changes"]["input_delta"] = 999
        # Persisted event unchanged.
        assert mgr.network.history["dataset_swaps"][0]["arch_changes"]["input_delta"] == 0
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


class TestDatasetSwapsRoute:
    def test_empty_history_returns_empty_list(self, client):
        resp = client.get("/v1/history/dataset_swaps")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == {"events": []}

    def test_returns_events_in_chronological_order(self, client):
        canned = [
            _make_event("2026-05-14T10:00:00+00:00", input_delta=0),
            _make_event("2026-05-14T11:00:00+00:00", input_delta=1),
        ]
        with patch.object(client.app.state.lifecycle, "get_dataset_swap_events", return_value=canned):
            resp = client.get("/v1/history/dataset_swaps")
        assert resp.status_code == 200
        events = resp.json()["data"]["events"]
        assert len(events) == 2
        assert events[0]["timestamp"] == "2026-05-14T10:00:00+00:00"

    def test_since_query_param_passed_to_lifecycle(self, client):
        """The route is a thin shim — it passes ``since`` through verbatim
        to the lifecycle method, which owns the filter semantics."""
        mock_get = MagicMock(return_value=[])
        with patch.object(client.app.state.lifecycle, "get_dataset_swap_events", mock_get):
            resp = client.get("/v1/history/dataset_swaps?since=2026-05-14T11:00:00%2B00:00")
        assert resp.status_code == 200
        mock_get.assert_called_once_with(since="2026-05-14T11:00:00+00:00")

    def test_lifecycle_missing_returns_503(self, client):
        """When the lifecycle is somehow None (e.g. startup hasn't finished
        wiring it), the route returns 503 like other history-adjacent
        routes rather than crashing with a 500."""
        with patch.object(client.app.state, "lifecycle", None):
            resp = client.get("/v1/history/dataset_swaps")
        assert resp.status_code == 503
