"""Golden API snapshot regression (OUT-12 / WS-6 pre-refactor baseline).

Snapshots scrubbed JSON response bodies for the read routes, in two states:

  * **fresh, no network** (deterministic, no 404s): /v1/health,
    /v1/training/status, /v1/metrics/history, /v1/metrics/transport,
    /v1/history/dataset_swaps.
  * **post-train**: create a network + run a tiny fixed-seed train to
    completion, then snapshot /v1/network, /v1/network/topology,
    /v1/network/stats, /v1/metrics.

Volatile fields (timestamps, git_sha/build_date/version, uptime, *_total
counters, uuid/server_instance_id/snapshot_seq, ...) are stripped by the
recursive scrubber before comparison; numeric weight arrays are tolerance-
compared. Route status codes are asserted exactly (200).

The app is built with ``create_app(Settings(auto_start=False))`` under a
TestClient using the daemon-thread ``__exit__`` shape from
test_api_full_lifecycle.py (dodges the anyio shutdown hang).

Run / re-capture: see test_golden_trajectory.py header (same flags;
``GOLDEN_CAPTURE=1`` to re-capture).
"""

import threading
import time

import golden_support as gs
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings


def _make_client():
    """Build a TestClient on a fresh no-auto-start app (daemon-thread exit)."""
    settings = Settings(auto_start=False)
    app = create_app(settings)
    tc = TestClient(app)
    tc.__enter__()
    return app, tc


def _shutdown_client(app, tc):
    """Tear down the TestClient without hanging on the anyio portal join."""
    lifecycle = getattr(app.state, "lifecycle", None)
    if lifecycle:
        stop = getattr(lifecycle, "_stop_event", None)
        if stop is not None:
            stop.set()
        executor = getattr(lifecycle, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
    coord = getattr(app.state, "worker_coordinator", None)
    if coord:
        coord.shutdown()
    exit_thread = threading.Thread(target=lambda: tc.__exit__(None, None, None), daemon=True)
    exit_thread.start()
    exit_thread.join(timeout=5)


@pytest.fixture
def golden_client():
    """Fresh, deterministic, no-auto-start API client per test."""
    gs.harden_determinism()
    app, tc = _make_client()
    try:
        yield tc
    finally:
        _shutdown_client(app, tc)


def _wait_for_state(client, expected_states, *, timeout=30.0, poll_interval=0.1):
    """Poll /v1/training/status until the state machine reaches expected_states."""
    deadline = time.monotonic() + timeout
    sm_status = None
    while time.monotonic() < deadline:
        data = client.get("/v1/training/status").json()
        sm_status = data["data"]["state_machine"]["status"].upper()
        if sm_status in {s.upper() for s in expected_states}:
            return data
        time.sleep(poll_interval)
    raise TimeoutError(f"Training did not reach {expected_states} within {timeout}s (last={sm_status})")


def _snapshot(client, path, name, *, drop_array_values_under=()):
    """GET ``path``, assert 200, and capture/assert the scrubbed JSON body."""
    resp = client.get(path)
    assert resp.status_code == 200, f"{path} -> {resp.status_code}: {resp.text[:200]}"
    gs.assert_or_capture(
        f"api_snapshots/{name}.json",
        resp.json(),
        rtol=gs.RTOL,
        atol=gs.ATOL,
        drop_array_values_under=drop_array_values_under,
    )


# Tiny network/train config for the API post-train state. Kept minimal so
# training reaches COMPLETED in a couple of seconds.
_API_NET_BODY = {
    "input_size": 2,
    "output_size": 2,
    "epochs_max": 3,
    "output_epochs": 2,
    "candidate_epochs": 2,
    "max_hidden_units": 2,
    "patience": 1,
}


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.golden
class TestGoldenApiSnapshots:
    """Golden snapshots of the read routes, fresh and post-train."""

    def test_fresh_no_network_snapshots(self, golden_client):
        """Routes in the fresh, no-network state match their goldens."""
        _snapshot(golden_client, "/v1/health", "health")
        _snapshot(golden_client, "/v1/training/status", "training_status_fresh")
        _snapshot(golden_client, "/v1/metrics/history", "metrics_history_fresh")
        _snapshot(golden_client, "/v1/metrics/transport", "metrics_transport_fresh")
        _snapshot(golden_client, "/v1/history/dataset_swaps", "history_dataset_swaps_fresh")

    def test_post_train_snapshots(self, golden_client):
        """Routes after a tiny fixed-seed train match their goldens."""
        train_x, train_y = gs.two_spiral_inline()

        resp = golden_client.post("/v1/network", json=_API_NET_BODY)
        assert resp.status_code == 200, resp.text[:300]

        resp = golden_client.post(
            "/v1/training/start",
            json={"inline_data": {"train_x": train_x, "train_y": train_y}},
        )
        assert resp.status_code == 200, resp.text[:300]

        _wait_for_state(golden_client, ("COMPLETED", "FAILED", "STOPPED"), timeout=30.0)

        _snapshot(golden_client, "/v1/network", "network_post_train")
        # Weight arrays in topology/stats are tolerance-compared; if a future
        # build proves them cross-machine unstable beyond tolerance, add the
        # offending key to drop_array_values_under here.
        _snapshot(golden_client, "/v1/network/topology", "network_topology_post_train")
        _snapshot(golden_client, "/v1/network/stats", "network_stats_post_train")
        _snapshot(golden_client, "/v1/metrics", "metrics_post_train")
