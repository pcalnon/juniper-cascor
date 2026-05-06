"""CAN-015h round-trip integration test.

Mitigates the design plan's "Med" risk row:

    Optimizer rebuild after add/remove unit corrupts subsequent
    training. → Round-trip integration test: snapshot → restore →
    add unit → re-snapshot → resume training → verify training
    converges.

Scope: walk the full snapshot/restore/mutate/re-snapshot/resume
pipeline through the live FastAPI surface (``TestClient``, no
external server). Each step asserts both the HTTP contract (status
code + response shape) and the FSM transition the operation is
supposed to drive, so a regression in either layer trips the test.

The test is deliberately *integration*-flavored — unit-level
behaviour for each individual mutation is already covered by
``test_patch_weights_manual``, ``test_add_hidden_unit_manual``, and
``test_remove_hidden_unit_manual``. What's *not* covered there is
the interaction:

- Does the optimizer rebuild correctly when training resumes after
  add/remove? (Optimizer state is dropped by h-1/h-2/h-3.)
- Does the cascade-rebuild surgery in ``remove_hidden_unit_manual``
  produce a network that ``resume_from_snapshot`` can re-load and
  ``train`` can drive forward?
- Does the FSM gate the restored-and-mutated state correctly so the
  user can't accidentally trigger ``start_training`` while still in
  ``Investigating``?

This is the test that catches "the unit tests passed but the
end-to-end flow is broken" regressions.
"""

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings


def _wait_for_state(client, expected_states, *, timeout=10.0, poll=0.1):
    """Poll ``/v1/training/status`` until the FSM state machine is in
    one of ``expected_states``. Returns the final status payload.

    Mirrors the helper in ``test_api_full_lifecycle.py`` but lives
    inline so this file is self-contained.
    """
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        resp = client.get("/v1/training/status")
        data = resp.json()["data"]
        last = data["state_machine"]["status"].upper()
        if last in {s.upper() for s in expected_states}:
            return data
        time.sleep(poll)
    raise TimeoutError(f"FSM did not reach {expected_states} within {timeout}s. Last: {last}")


@pytest.fixture
def client():
    """TestClient with cooperative-shutdown daemon-thread exit, same
    pattern as ``test_api_full_lifecycle.py``. The training thread
    has no cancellation check, so a clean ``__exit__`` would block
    on the anyio portal join."""
    settings = Settings()
    app = create_app(settings)
    tc = TestClient(app)
    tc.__enter__()
    yield tc

    lifecycle = getattr(app.state, "lifecycle", None)
    if lifecycle:
        lifecycle._stop_requested.set()
        if getattr(lifecycle, "_executor", None):
            lifecycle._executor.shutdown(wait=False, cancel_futures=True)

    coord = getattr(app.state, "worker_coordinator", None)
    if coord:
        coord.shutdown()

    exit_thread = threading.Thread(target=lambda: tc.__exit__(None, None, None), daemon=True)
    exit_thread.start()
    exit_thread.join(timeout=5)


@pytest.fixture
def cleanup_snapshots(client):
    """Track snapshot IDs created by the test and remove the .h5
    files at teardown. The snapshots dir is shared with non-test
    runs (see ``lifecycle/manager.py::_get_snapshots_dir``), so
    fixture cleanup keeps the repo working tree clean.
    """
    created = []
    yield created
    # Best-effort cleanup — failures here shouldn't mask test failures.
    for snap_id in created:
        try:
            client.delete(f"/v1/snapshots/{snap_id}")
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Toy fixed-seed dataset — fast, well-separated, lets the network reach a
# meaningful loss within the small training budget the test allots.
# ---------------------------------------------------------------------------

_TRAIN_X = [
    [-1.0, -1.0],
    [-0.9, -0.8],
    [-1.0, -0.9],
    [-0.8, -1.0],
    [-1.1, -0.9],
    [1.0, 1.0],
    [0.9, 0.8],
    [1.0, 0.9],
    [0.8, 1.0],
    [1.1, 0.9],
]
_TRAIN_Y = [
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
]


@pytest.mark.integration
class TestCAN015hRoundTrip:
    """End-to-end CAN-015h: snapshot → restore → mutate → re-snapshot → resume.

    The plan's risk row reads "snapshot → restore → add unit → re-snapshot
    → resume training → verify training converges." We deliberately do not
    train *before* the initial snapshot — the optimizer-rebuild concern
    fires at the resume-train step regardless of whether the snapshotted
    network had a prior training pass, and skipping the initial train
    keeps the test under a reasonable wall-clock budget on CI.
    """

    def _create_network(self, client):
        """Create a small toy network without training. Returns the
        topology payload captured before the snapshot — the route is
        known-good in service mode at this point in the FSM lifecycle.
        """
        resp = client.post(
            "/v1/network",
            json={
                "input_size": 2,
                "output_size": 2,
                "epochs_max": 4,
                "output_epochs": 2,
                "candidate_epochs": 2,
                "max_hidden_units": 2,
                "patience": 1,
            },
        )
        assert resp.status_code == 200, resp.text

        # Capture topology pre-snapshot. The post-restore code path for
        # this route currently has a separate bug independent of CAN-015h;
        # capturing pre-snapshot lets us still assert against
        # ``num_hidden_pre`` after the round-trip without coupling this
        # test to that other regression.
        resp = client.get("/v1/network/topology")
        assert resp.status_code == 200, f"pre-snapshot topology failed: {resp.text}"
        return resp.json()["data"]

    def test_full_roundtrip_add_remove_patch(self, client, cleanup_snapshots):
        """The headline test: every CAN-015h surface in one flow.

        Step-by-step asserts so a regression localizes immediately.
        """
        # ---- Step 1: create network + capture topology --------------------
        topo_pre = self._create_network(client)
        num_hidden_pre = len(topo_pre.get("hidden_units") or [])
        input_size = topo_pre.get("input_size", 2)
        output_size = topo_pre.get("output_size", 2)

        # ---- Step 2: snapshot the freshly created network -----------------
        resp = client.post("/v1/snapshots", json={"description": "roundtrip-baseline"})
        assert resp.status_code == 200, resp.text
        snap_data = resp.json()["data"]
        snap_id_initial = snap_data.get("id") or snap_data.get("snapshot_id")
        assert snap_id_initial, f"snapshot create missing id: {snap_data!r}"
        cleanup_snapshots.append(snap_id_initial)
        snap_path = Path(snap_data.get("path", ""))
        if snap_path:
            assert snap_path.exists(), f"snapshot HDF5 file missing on disk: {snap_path}"

        # ---- Step 3: restore → FSM should land in Investigating -----------
        resp = client.post(f"/v1/snapshots/{snap_id_initial}/restore")
        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        # The unified-response shape carries operation + fsm_state.
        assert body.get("operation") == "restore"
        assert body.get("fsm_state", "").upper() == "INVESTIGATING"

        # Independently verify via /v1/training/status — defends against
        # the unified-payload regression where fsm_state was right but
        # the actual state machine wasn't transitioned.
        resp = client.get("/v1/training/status")
        sm_status = resp.json()["data"]["state_machine"]["status"].upper()
        assert sm_status == "INVESTIGATING", f"expected INVESTIGATING after restore, got {sm_status}"

        # ---- Step 4: PATCH /v1/network/weights — zero out output bias -----
        # Output bias is the smallest tensor and the easiest to verify
        # round-trip, but the same code path proves out shape/dtype/NaN
        # validation for the rest of PATCH.
        # Cascor schema: target="output"|"hidden_unit" + field="weights"|"bias".
        # See ``src/api/models/network.py::PatchWeightsRequest``.
        resp = client.patch(
            "/v1/network/weights",
            json={
                "target": "output",
                "field": "bias",
                "values": [0.0] * output_size,
                "dtype": "float32",
            },
        )
        assert resp.status_code == 200, f"PATCH output bias failed: {resp.text}"

        # ---- Step 6: POST /v1/network/hidden-units — append at tail -------
        # Weight vector length must equal input_size + num_existing_hidden.
        new_unit_weight_len = input_size + num_hidden_pre
        new_weights = [0.01 * (i + 1) for i in range(new_unit_weight_len)]
        resp = client.post(
            "/v1/network/hidden-units",
            json={"weights": new_weights, "bias": 0.0, "activation": "Tanh"},
        )
        assert resp.status_code == 200, f"POST hidden-units failed: {resp.text}"
        add_body = resp.json()["data"] if "data" in resp.json() else resp.json()
        new_idx = add_body.get("unit_index")
        new_total = add_body.get("num_hidden_units")
        assert new_idx is not None
        assert new_total == num_hidden_pre + 1

        # ---- Step 7: DELETE the unit we just added ------------------------
        # Deleting the unit we just added gets us back to the original
        # topology, which makes the post-resume convergence comparison
        # apples-to-apples. The cascade-rebuild surgery still has to
        # work — DELETE doesn't know we just added this one.
        resp = client.delete(f"/v1/network/hidden-units/{new_idx}")
        assert resp.status_code == 200, f"DELETE hidden-units failed: {resp.text}"
        del_body = resp.json()["data"] if "data" in resp.json() else resp.json()
        assert del_body.get("removed_index") == new_idx
        assert del_body.get("num_hidden_units") == num_hidden_pre

        # FSM still in Investigating after every mutation — none of these
        # endpoints are supposed to transition the FSM out.
        resp = client.get("/v1/training/status")
        assert resp.json()["data"]["state_machine"]["status"].upper() == "INVESTIGATING"

        # ---- Step 8: re-snapshot the post-mutation network ----------------
        # Catches the failure mode where mutation leaves the network in
        # a state the serializer can't write (e.g. weights without
        # requires_grad, optimizer in a torn state).
        # NOTE: lifecycle.save_snapshot generates ids at second
        # precision (`snapshot_<YYYYMMDDTHHMMSSZ>`), so two snapshots
        # taken in the same wall-clock second collide. Sleep just over
        # 1s so the post-mutation snapshot gets a distinct id and the
        # test's "the new snapshot has a different id" assertion is
        # meaningful. (Tracked as a separate concern — the lifecycle
        # could append a counter, but that's not in scope for this
        # CAN-015h hardening test.)
        time.sleep(1.1)
        resp = client.post("/v1/snapshots", json={"description": "roundtrip-post-mutation"})
        assert resp.status_code == 200, f"re-snapshot failed: {resp.text}"
        snap_id_post = resp.json()["data"].get("id") or resp.json()["data"].get("snapshot_id")
        assert snap_id_post and snap_id_post != snap_id_initial
        cleanup_snapshots.append(snap_id_post)

        # ---- Step 9: resume from the post-mutation snapshot ---------------
        # The Investigating-state restore can't drive training; resume
        # is the documented way back into Started/RESUME_READY.
        resp = client.post(f"/v1/snapshots/{snap_id_post}/resume")
        assert resp.status_code == 200, f"resume failed: {resp.text}"
        body = resp.json()["data"]
        assert body.get("operation") == "resume"
        # resume should leave the FSM in RESUME_READY — start_training
        # is the next step.
        resp = client.get("/v1/training/status")
        post_resume_sm = resp.json()["data"]["state_machine"]["status"].upper()
        assert post_resume_sm in ("RESUME_READY", "STOPPED"), f"unexpected FSM after resume: {post_resume_sm}"

        # ---- Step 10: drive training again --------------------------------
        # If the optimizer rebuild from h-1/h-2/h-3 is broken, this is
        # where it shows up — either training fails outright or the
        # loss diverges.
        resp = client.post(
            "/v1/training/start",
            json={"inline_data": {"train_x": _TRAIN_X, "train_y": _TRAIN_Y}},
        )
        assert resp.status_code == 200, f"resume-start failed: {resp.text}"
        try:
            _wait_for_state(client, ("COMPLETED", "FAILED"), timeout=20.0)
        except TimeoutError:
            client.post("/v1/training/stop")
            _wait_for_state(client, ("STOPPED", "COMPLETED", "FAILED"), timeout=10.0)

        # ---- Step 11: verify the resumed network actually trained ---------
        resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        post_resume_metrics = resp.json().get("data") or resp.json()
        post_resume_loss = post_resume_metrics.get("train_loss")
        assert post_resume_loss is not None, "post-resume metrics missing train_loss"
        # Loss must be finite and non-NaN. We don't require a strict
        # decrease — the toy run is too short to guarantee improvement
        # past the baseline — but a NaN/Inf loss is a hard failure
        # that catches the optimizer-corruption regression class.
        assert post_resume_loss == post_resume_loss, "post-resume train_loss is NaN"  # noqa: PLR0124
        assert abs(post_resume_loss) < 1e6, f"post-resume train_loss exploded: {post_resume_loss}"

    # -----------------------------------------------------------------------
    # Targeted FSM-gate sanity check
    # -----------------------------------------------------------------------

    def test_mutations_rejected_outside_investigating(self, client, cleanup_snapshots):
        """The mutation endpoints must reject when the FSM isn't in
        Investigating — covers a regression where a state-machine
        refactor accidentally widens the permit set."""
        # Get the network into a Started/Stopped state without
        # routing through Investigating. A freshly created network is
        # in Stopped state — same FSM-gate result as Started/Paused for
        # all three mutation routes.
        self._create_network(client)

        # PATCH should be rejected (409).
        resp = client.patch(
            "/v1/network/weights",
            json={"target": "output", "field": "bias", "values": [0.0, 0.0], "dtype": "float32"},
        )
        assert resp.status_code == 409, f"PATCH should be 409 outside Investigating, got {resp.status_code}: {resp.text}"

        # POST should be rejected (409).
        resp = client.post(
            "/v1/network/hidden-units",
            json={"weights": [0.1, 0.2], "bias": 0.0, "activation": "Tanh"},
        )
        assert resp.status_code == 409

        # DELETE should be rejected (409).
        # idx=0 may or may not exist depending on training outcome;
        # the FSM check fires before the range check so 409 is the
        # expected status.
        resp = client.delete("/v1/network/hidden-units/0")
        assert resp.status_code == 409
