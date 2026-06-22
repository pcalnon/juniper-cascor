"""WS-6 gate: live ``/ws/training`` per-grow-iteration granularity (PR-B3 kill-criterion).

Behavioral guard (NOT a snapshot) that a real, manager-driven ``fit`` + cascade ``grow``
still streams **per-grow-iteration** progress to the WebSocket — the synchronous
``grow_network`` -> ``_grow_iteration_callback`` -> ``state.update_state(...)`` ->
``_broadcast_training_state()`` -> ``create_state_message`` chain, carrying live
candidate-pool detail (``grow_iteration`` / ``candidates_trained`` / ``best_correlation``).

This is the exit test for **WS-6 PR-B3** (the ``on_event`` monitoring migration). PR-B3
moves this coarse per-iteration projection off the manager-bound native callback and onto
``CascorModel.fit(on_event=...)`` -> ``manager._handle_event(...)``; this test pins that the
WS still sees per-iteration grow progress after the cutover. It is path-agnostic (passes on
the current monkey-patch path *and* must keep passing after B3.3), serial-deterministic, and
asserts a lower-bound invariant rather than a fixture comparison, so it is robust to timing.

Scope note (granularities): cascor exposes two live-progress streams.
  1. **Per-grow-iteration** (this test): a *synchronous* CCN native hook in the main
     process; flows in both serial and multiprocessing modes; it is the stream B3.3 actually
     rewires (native-callback -> on_event), so it is the high-value gate guard.
  2. **Per-candidate 50 Hz** (``candidate_progress``): an *async* worker-process queue
     (``_persistent_progress_queue`` -> drain thread). It exists **only** under real
     multiprocessing (``CASCOR_NUM_PROCESSES > 1``); the serial gate lane uses the
     sequential candidate fallback that never creates the queue, so it cannot be asserted
     here. B3.3 *relocates* that drain wiring verbatim (monitored_grow -> run); its
     regression guard is the faked-queue unit test in ``tests/unit/api/test_monitoring_hooks``.

Marked ``golden`` so it runs in the WS-6 serial gate lane (``golden-regression.yml``,
``-m golden --golden --slow --integration src/tests/integration``) as a required check.

Run (serial, GIL env):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    CASCOR_NUM_PROCESSES=1 \
    python -m pytest -m golden --golden --slow --integration \
        src/tests/integration/test_ws_training_granularity.py
"""

from unittest.mock import MagicMock

import golden_support as gs
import pytest

from api.lifecycle.manager import TrainingLifecycleManager


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.golden
def test_live_grow_iteration_progress_streamed_to_ws_during_real_training():
    """A real fit + cascade grow streams >=1 per-grow-iteration ``state`` WS frame."""
    # Serial, single-process candidate path + seed 42 (parity with the golden lane). The
    # per-iteration callback is synchronous and fires on this path regardless.
    gs.harden_determinism()

    x, y = gs.load_two_spiral()

    mgr = TrainingLifecycleManager()
    # Canonical golden net config: small but grows real hidden units, so grow_network runs
    # multiple cascade iterations and fires the per-iteration callback.
    mgr.create_network(input_size=int(x.shape[1]), output_size=int(y.shape[1]), **gs.GOLDEN_NET_CONFIG)

    # Recording stand-in for the WebSocket manager. throttle=0.0 disables the GAP-WS-21
    # coalescer so every per-iteration state broadcast is observable (no flush dependence).
    ws_mgr = MagicMock()
    mgr.set_ws_manager(ws_mgr, state_throttle_interval=0.0)

    try:
        # early_stopping=False forces the full max_iterations of cascade growth.
        mgr.start_training(X=x, y=y, X_val=x, y_val=y, early_stopping=False)
        assert mgr._training_future is not None, "start_training did not submit a training future"
        # Block until the background training thread finishes; re-raises any error it hit.
        mgr._training_future.result(timeout=120)
    finally:
        mgr.shutdown()

    # Every training-"state" frame broadcast through the WS during the run.
    state_frames = [call.args[0] for call in ws_mgr.broadcast_from_thread.call_args_list if call.args and isinstance(call.args[0], dict) and call.args[0].get("type") == "state"]
    # Frames that carry per-grow-iteration candidate-pool progress (the granularity signal).
    grow_frames = [f for f in state_frames if isinstance(f.get("data"), dict) and f["data"].get("candidates_trained", 0) >= 1]

    max_candidates = max((f["data"].get("candidates_trained", 0) for f in state_frames if isinstance(f.get("data"), dict)), default=0)
    max_iteration = max((f["data"].get("grow_iteration", -1) for f in state_frames if isinstance(f.get("data"), dict)), default=-1)

    assert grow_frames, "expected >=1 per-grow-iteration 'state' WS frame carrying candidate-pool progress " f"(candidates_trained>=1) from a real fit+grow; saw {len(state_frames)} state frame(s), " f"max candidates_trained={max_candidates}, max grow_iteration={max_iteration}. The live " "/ws/training per-iteration stream did not flow. This is the WS-6 PR-B3 kill-criterion."

    # Granularity sanity: a grow frame exposes the iteration field the live view renders.
    assert "grow_iteration" in grow_frames[0]["data"], f"grow 'state' frame missing grow_iteration field: {grow_frames[0]['data']!r}"
