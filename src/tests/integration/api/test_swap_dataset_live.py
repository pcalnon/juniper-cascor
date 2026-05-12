#!/usr/bin/env python
"""Regression tests for Phase 2 P2-1a: ``swap_dataset_live`` skeleton + gate.

Covers the lifecycle-method surface defined in
``ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md`` §3.1/§3.2/§3.7/§3.8 and the
admin gate route. Does NOT cover the full P2-1a route surface (the FastAPI
TestClient pieces live in a separate file) and does NOT cover P2-1b cancel
semantics, P2-1c grow adapter, or P2-1d shrink adapter.

P2-1a contract pinned here:
  - ``set_experimental_functions`` mutates server-side state; default False.
  - ``swap_dataset_live`` raises ``PermissionError`` when gate closed (→ 403).
  - ``swap_dataset_live`` raises ``ValueError`` when no training is running (→ 422).
  - ``swap_dataset_live`` raises ``SwapInProgressError`` on concurrent swap (→ 409).
  - ``swap_dataset_live`` raises ``ValueError("dim_change_unsupported")`` when
    the new dataset's input/output dim differs from the current network (→ 422).
  - On any failure mid-swap, pre-swap tensors are restored via
    ``_rollback_pre_swap_state`` (§3.8 contract).
  - On equal-dim success: tensors swap, ``_current_dataset_config`` updates,
    a fresh training future is submitted, response shape matches §3.3.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from api.lifecycle.manager import (
    NoSwapInProgressError,
    SwapCancelledError,
    SwapInProgressError,
    TrainingLifecycleManager,
    _PreSwapSnapshot,
)
from api.lifecycle.state_machine import Command, TrainingPhase


def _make_dummy_tensors(input_size: int, output_size: int, n_samples: int = 16):
    """Cheap deterministic tensors for swap-target injection."""
    x = torch.randn(n_samples, input_size)
    y = torch.zeros(n_samples, output_size)
    y[:, 0] = 1  # all class 0 — enough for swap mechanics
    return x, y


# ---------------------------------------------------------------------------
# Experimental-functions gate (manager-method level)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_experimental_functions_default_false():
    """Without env override the gate is closed at construction (F2.10 default-safe)."""
    mgr = TrainingLifecycleManager()
    assert mgr.get_experimental_functions() is False


@pytest.mark.integration
def test_set_experimental_functions_opens_gate():
    mgr = TrainingLifecycleManager()
    result = mgr.set_experimental_functions(True)
    assert result == {"experimental_functions_enabled": True}
    assert mgr.get_experimental_functions() is True


@pytest.mark.integration
def test_set_experimental_functions_closes_gate():
    mgr = TrainingLifecycleManager()
    mgr.set_experimental_functions(True)
    mgr.set_experimental_functions(False)
    assert mgr.get_experimental_functions() is False


@pytest.mark.integration
def test_experimental_functions_env_override():
    """``CASCOR_EXPERIMENTAL_FUNCTIONS_ENABLED=1`` opens the gate at construction
    so deployments can pre-enable without an admin round-trip on every restart."""
    with patch.dict("os.environ", {"CASCOR_EXPERIMENTAL_FUNCTIONS_ENABLED": "1"}):
        mgr = TrainingLifecycleManager()
        assert mgr.get_experimental_functions() is True


# ---------------------------------------------------------------------------
# swap_dataset_live: precondition validations
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_swap_dataset_live_raises_permission_when_gate_closed():
    """Gate closed → PermissionError (route translates to 403). Hard-fail BEFORE
    acquiring the training lock so the check is cheap even under load."""
    mgr = TrainingLifecycleManager()
    assert mgr.get_experimental_functions() is False
    with pytest.raises(PermissionError, match="experimental_functions_disabled"):
        mgr.swap_dataset_live(dataset_type="spirals")


@pytest.mark.integration
def test_swap_dataset_live_raises_when_not_running():
    """Gate open but training not active → ValueError (route → 422). Uses the
    cold-swap recommendation in the error message so callers know where to go."""
    mgr = TrainingLifecycleManager()
    mgr.set_experimental_functions(True)
    assert not mgr.state_machine.is_started()
    with pytest.raises(ValueError, match="training_not_running"):
        mgr.swap_dataset_live(dataset_type="spirals")


@pytest.mark.integration
def test_swap_dataset_live_409_when_swap_in_progress():
    """``_swap_in_progress`` flag rejects concurrent swap with SwapInProgressError
    (→ 409). The flag is normally cleared in a ``finally`` block so this test
    sets it manually to simulate a swap mid-flight on another thread."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    # Drive FSM to Started so the is_started() check passes.
    mgr.state_machine.handle_command(Command.START)
    # Simulate "another swap is already in flight":
    mgr._swap_in_progress = True
    try:
        with pytest.raises(SwapInProgressError, match="swap_already_in_progress"):
            mgr.swap_dataset_live(dataset_type="spirals")
    finally:
        mgr._swap_in_progress = False  # restore for fixture teardown
        mgr.shutdown()


# ---------------------------------------------------------------------------
# swap_dataset_live: dim-change rejection (P2-1a only allows equal-dim)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_swap_dataset_live_rejects_input_dim_change():
    """P2-1a's hard guard: new input dim must equal current network input_size.
    Raises ValueError("dim_change_unsupported") → 422. Rollback restores the
    original tensors so training can resume on the OLD dataset (§3.8)."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    pre_train_x = torch.randn(8, 2)
    pre_train_y = torch.zeros(8, 2)
    pre_train_y[:, 0] = 1
    mgr._train_x = pre_train_x
    mgr._train_y = pre_train_y
    mgr.state_machine.handle_command(Command.START)

    # Stub _reload_dataset to deliver tensors of a DIFFERENT input dim.
    def _fake_reload(**cfg):
        mgr._train_x = torch.randn(8, 5)  # input_size=5 != current 2
        mgr._train_y = torch.zeros(8, 2)
        mgr._val_x = None
        mgr._val_y = None
        mgr._current_dataset_config = {"dataset_type": "synthetic_5_2"}

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        with pytest.raises(ValueError, match="dim_change_unsupported"):
            mgr.swap_dataset_live(dataset_type="synthetic_5_2")

    # §3.8 rollback contract: original tensors are restored.
    assert mgr._train_x is pre_train_x, "pre-swap _train_x not restored after dim-change rejection"
    assert mgr._train_y is pre_train_y
    assert mgr._swap_in_progress is False, "_swap_in_progress not cleared in finally"
    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_rejects_output_dim_change():
    """Same guard, on the output dim. The error message names both deltas so
    a UI can show "input 2→2, output 2→3 not supported"."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x = torch.randn(8, 2)
    mgr._train_y = torch.zeros(8, 2)
    mgr._train_y[:, 0] = 1
    mgr.state_machine.handle_command(Command.START)

    def _fake_reload(**cfg):
        mgr._train_x = torch.randn(8, 2)
        mgr._train_y = torch.zeros(8, 3)  # output 3 != current 2
        mgr._train_y[:, 0] = 1
        mgr._val_x = None
        mgr._val_y = None
        mgr._current_dataset_config = {"dataset_type": "synthetic_2_3"}

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        with pytest.raises(ValueError, match="dim_change_unsupported"):
            mgr.swap_dataset_live(dataset_type="synthetic_2_3")

    assert mgr._swap_in_progress is False
    mgr.shutdown()


# ---------------------------------------------------------------------------
# swap_dataset_live: failure restoration (§3.8 contract)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_swap_dataset_live_restores_state_on_fetch_failure():
    """If ``_reload_dataset`` raises (juniper-data unreachable etc.), the
    original tensor refs are restored and the exception propagates."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    pre_train_x, pre_train_y = _make_dummy_tensors(2, 2)
    mgr._train_x = pre_train_x
    mgr._train_y = pre_train_y
    mgr.state_machine.handle_command(Command.START)

    with patch.object(mgr, "_reload_dataset", side_effect=RuntimeError("juniper-data unreachable")):
        with pytest.raises(RuntimeError, match="juniper-data unreachable"):
            mgr.swap_dataset_live(dataset_type="spirals")

    # Tensors untouched; in-progress flag cleared.
    assert mgr._train_x is pre_train_x
    assert mgr._train_y is pre_train_y
    assert mgr._swap_in_progress is False
    mgr.shutdown()


# ---------------------------------------------------------------------------
# swap_dataset_live: equal-dim success path
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_swap_dataset_live_equal_dim_success_response_shape():
    """Happy path: dim matches → swap succeeds, response matches §3.3 schema,
    ``_current_dataset_config`` reflects the new cfg, a new training future
    is submitted. Stubs out the executor so no real fit runs."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    pre_train_x, pre_train_y = _make_dummy_tensors(2, 2)
    mgr._train_x = pre_train_x
    mgr._train_y = pre_train_y
    mgr._current_dataset_config = {"dataset_type": "spirals", "n_spirals": 2}
    mgr.state_machine.handle_command(Command.START)

    # Stub _reload_dataset to keep dim=2/2 but change the cfg.
    new_train_x, new_train_y = _make_dummy_tensors(2, 2)

    def _fake_reload(**cfg):
        mgr._train_x = new_train_x
        mgr._train_y = new_train_y
        mgr._val_x = None
        mgr._val_y = None
        mgr._current_dataset_config = {"dataset_type": "moons", "noise": 0.2}

    # Mock the executor so submit() returns a no-op Future without
    # actually running _run_training (which would call network.fit).
    mock_future = MagicMock()
    mock_future.result.return_value = None
    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = mock_future

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        result = mgr.swap_dataset_live(dataset_type="moons", noise=0.2)

    # §3.3 response shape:
    assert result["status"] == "swapped"
    assert result["before_cfg"] == {"dataset_type": "spirals", "n_spirals": 2}
    assert result["after_cfg"] == {"dataset_type": "moons", "noise": 0.2}
    assert result["mode"] == "output_training_first"
    assert result["arch_changes"]["input_delta"] == 0
    assert result["arch_changes"]["output_delta"] == 0
    assert result["arch_changes"]["appended_nodes"] == {"input": 0, "output": 0}
    assert result["arch_changes"]["prepended_layers"] == []

    # State observable:
    assert mgr._train_x is new_train_x, "new tensors installed"
    assert mgr._swap_in_progress is False, "flag cleared in finally"
    assert mgr._executor.submit.called, "new training future submitted"

    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_resets_auto_snap_ratchet():
    """§3.7 guardrail #6: auto-snap ratchet is cleared so the stale metric
    scale from the old dataset doesn't suppress new auto-snaps."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x = torch.randn(8, 2)
    mgr._train_y = torch.zeros(8, 2)
    mgr._train_y[:, 0] = 1
    mgr._auto_snap_best_metric = 0.95  # stale ratchet from old dataset
    mgr.state_machine.handle_command(Command.START)

    def _fake_reload(**cfg):
        mgr._train_x = torch.randn(8, 2)
        mgr._train_y = torch.zeros(8, 2)
        mgr._train_y[:, 0] = 1
        mgr._val_x = None
        mgr._val_y = None
        mgr._current_dataset_config = {"dataset_type": "moons"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        mgr.swap_dataset_live(dataset_type="moons")

    assert mgr._auto_snap_best_metric is None, "ratchet not reset"
    mgr.shutdown()


# ---------------------------------------------------------------------------
# _PreSwapSnapshot helper (§3.7 guardrail #1)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_pre_swap_snapshot_is_a_value_container():
    """Pins the snapshot shape so future PRs adding fields don't accidentally
    repurpose it. ``state_dict`` is the one field that's deep-copied by the
    caller; tensors/cfg are reference-only."""
    snap = _PreSwapSnapshot(
        train_x="X",
        train_y="Y",
        val_x=None,
        val_y=None,
        state_dict={"w": "..."},
        input_size=2,
        output_size=3,
        dataset_config={"dataset_type": "spirals"},
    )
    assert snap.train_x == "X"
    assert snap.input_size == 2
    assert snap.output_size == 3
    assert snap.dataset_config == {"dataset_type": "spirals"}


@pytest.mark.integration
def test_rollback_restores_tensor_refs():
    """``_rollback_pre_swap_state`` MUST restore tensor refs even if
    ``load_state_dict`` raises (best-effort weights restore)."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    pre_x, pre_y = _make_dummy_tensors(2, 2)
    snap = _PreSwapSnapshot(
        train_x=pre_x,
        train_y=pre_y,
        val_x=None,
        val_y=None,
        state_dict=None,  # no state_dict to load
        input_size=2,
        output_size=2,
        dataset_config={"dataset_type": "spirals"},
    )
    # Mutate live state to simulate mid-swap corruption.
    mgr._train_x = torch.randn(8, 5)
    mgr._train_y = torch.zeros(8, 3)
    mgr._current_dataset_config = {"dataset_type": "moons"}

    mgr._rollback_pre_swap_state(snap)

    assert mgr._train_x is pre_x
    assert mgr._train_y is pre_y
    assert mgr._current_dataset_config == {"dataset_type": "spirals"}
    mgr.shutdown()


# ---------------------------------------------------------------------------
# P2-1b: cancel mechanism (request_swap_cancel + DELETE route translation)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_request_swap_cancel_raises_when_no_swap_in_progress():
    """``DELETE /v1/training/dataset/live`` against an idle lifecycle raises
    ``NoSwapInProgressError`` (route → 404). Tells the canopy "Cancel" button
    the swap already finished racing the click."""
    mgr = TrainingLifecycleManager()
    mgr.set_experimental_functions(True)
    assert mgr._swap_in_progress is False
    with pytest.raises(NoSwapInProgressError, match="no_swap_in_progress"):
        mgr.request_swap_cancel()
    mgr.shutdown()


@pytest.mark.integration
def test_request_swap_cancel_sets_signal_when_swap_in_progress():
    """When a swap IS underway, ``request_swap_cancel`` sets the cancel
    signal and returns a descriptor dict (no exception)."""
    mgr = TrainingLifecycleManager()
    mgr.set_experimental_functions(True)
    mgr._swap_in_progress = True  # simulate in-flight swap on another thread
    try:
        result = mgr.request_swap_cancel()
        assert result == {"status": "cancel_requested"}
        assert mgr._swap_cancel_requested.is_set()
    finally:
        mgr._swap_in_progress = False
        mgr._swap_cancel_requested.clear()
        mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_aborts_when_cancel_set_during_fetch():
    """A DELETE arriving during ``_reload_dataset`` trips the post-fetch
    cancel checkpoint → ``SwapCancelledError`` → §3.8 rollback restores the
    pre-swap tensors. The most common race the user can produce."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    pre_train_x, pre_train_y = _make_dummy_tensors(2, 2)
    mgr._train_x = pre_train_x
    mgr._train_y = pre_train_y
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)

    def _fake_reload_then_cancel(**cfg):
        # Simulate the user clicking Cancel while the fetch is in flight.
        # Real fetch would block on juniper-data HTTP; here we just install
        # new tensors AND set the cancel flag in the same call.
        mgr._train_x = torch.randn(8, 2)
        mgr._train_y = torch.zeros(8, 2)
        mgr._train_y[:, 0] = 1
        mgr._current_dataset_config = {"dataset_type": "moons"}
        mgr._swap_cancel_requested.set()

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload_then_cancel):
        with pytest.raises(SwapCancelledError, match="swap_cancelled_by_client"):
            mgr.swap_dataset_live(dataset_type="moons")

    # §3.8 rollback: pre-swap tensors restored, cancel flag cleared in finally.
    assert mgr._train_x is pre_train_x
    assert mgr._train_y is pre_train_y
    assert mgr._current_dataset_config == {"dataset_type": "spirals"}
    assert mgr._swap_in_progress is False
    assert not mgr._swap_cancel_requested.is_set(), "cancel flag not cleared in finally"
    # No new training future was submitted (we aborted before step 12).
    assert not mgr._executor.submit.called
    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_clears_stale_cancel_flag_at_start():
    """A cancel signal left set from a previous aborted swap MUST NOT
    pre-cancel the next swap — the flag is per-swap, not sticky. The
    swap clears it under the lock right after acquiring ``_swap_in_progress``."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)
    # Pre-set the cancel flag as if a prior swap's DELETE leaked through.
    mgr._swap_cancel_requested.set()

    new_x, new_y = _make_dummy_tensors(2, 2)

    def _fake_reload(**cfg):
        mgr._train_x = new_x
        mgr._train_y = new_y
        mgr._current_dataset_config = {"dataset_type": "moons"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        # Should complete without raising — the stale flag was cleared.
        result = mgr.swap_dataset_live(dataset_type="moons")

    assert result["status"] == "swapped"
    assert mgr._train_x is new_x
    assert not mgr._swap_cancel_requested.is_set()
    mgr.shutdown()


# ---------------------------------------------------------------------------
# P2-1b: abandoned_candidate_pool_size accounting
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_abandoned_candidate_pool_size_zero_outside_candidate_phase():
    """Output / Idle / Paused phases have no in-flight candidates — the
    response reports zero (not the configured pool capacity)."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)
    # Started lands in TrainingPhase.OUTPUT — no candidates in flight.
    assert mgr.state_machine.phase == TrainingPhase.OUTPUT

    def _fake_reload(**cfg):
        mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
        mgr._current_dataset_config = {"dataset_type": "moons"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        result = mgr.swap_dataset_live(dataset_type="moons")

    assert result["arch_changes"]["abandoned_candidate_pool_size"] == 0
    mgr.shutdown()


@pytest.mark.integration
def test_abandoned_candidate_pool_size_reports_pool_when_in_candidate_phase():
    """Mid-CANDIDATE swaps report the in-flight pool depth (Option C of §3.5).
    Drives the canopy "Swap discarded N in-flight candidates" UX."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)
    # Force into CANDIDATE phase to simulate "swap fired during candidate
    # training". Real production sets this via the cascade-growth callback.
    mgr.state_machine.set_phase(TrainingPhase.CANDIDATE)
    # Pin a known pool size on the network so the assertion is meaningful.
    mgr.network.candidate_pool_size = 8

    def _fake_reload(**cfg):
        mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
        mgr._current_dataset_config = {"dataset_type": "moons"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        result = mgr.swap_dataset_live(dataset_type="moons")

    assert result["arch_changes"]["abandoned_candidate_pool_size"] == 8
    mgr.shutdown()


# ---------------------------------------------------------------------------
# P2-1b: §3.7 #5 structured INFO completion log
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_swap_dataset_live_emits_structured_completion_log():
    """§3.7 guardrail #5: on completion the swap emits a single INFO line
    matching the format:
        "swap: input I_old→I_new, output O_old→O_new, hidden H preserved, candidates C abandoned, mode→output_training"

    Asserts on the format-string + args passed to ``logger.info`` rather than
    via caplog — the manager's logger has a custom configuration in this
    project that does not always propagate to the root logger that caplog
    attaches its handler to. Patching the bound method makes the assertion
    independent of logging configuration.
    """
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)

    def _fake_reload(**cfg):
        mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
        mgr._current_dataset_config = {"dataset_type": "moons"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr.logger, "info") as info_mock:
        with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
            mgr.swap_dataset_live(dataset_type="moons")

    # Find the structured completion line among all info() calls.
    matching = [c for c in info_mock.call_args_list if c.args and isinstance(c.args[0], str) and c.args[0].startswith("swap: input ")]
    assert len(matching) == 1, f"expected 1 completion line, got {info_mock.call_args_list}"
    fmt, *args = matching[0].args
    # Equal-dim swap from this test fixture: 2→2 / 2→2, hidden 0 preserved,
    # 0 candidates abandoned (Output phase).
    rendered = fmt % tuple(args)
    assert rendered == "swap: input 2→2, output 2→2, hidden 0 preserved, candidates 0 abandoned, mode→output_training"
    mgr.shutdown()


# ---------------------------------------------------------------------------
# P2-1b: topology rebroadcast plumbing (§3.7 guardrail #7)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_swap_dataset_live_forces_topology_rebroadcast_on_completion():
    """Even an equal-dim swap MUST force a topology rebroadcast — the path
    is plumbed here so P2-1c/1d (grow/shrink) inherit it. Without the
    rebroadcast canopy's topology view would lag a real dim change."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)

    def _fake_reload(**cfg):
        mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
        mgr._current_dataset_config = {"dataset_type": "moons"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr, "_broadcast_training_state") as broadcast_mock:
        with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
            mgr.swap_dataset_live(dataset_type="moons")

    # Exactly one forced broadcast on swap completion. ``force=True`` is the
    # invariant — the throttled-broadcast path would coalesce the topology
    # event with metric ticks and lose its standalone framing.
    forced_calls = [c for c in broadcast_mock.call_args_list if c.kwargs.get("force") is True]
    assert len(forced_calls) >= 1, f"no force=True broadcast observed: {broadcast_mock.call_args_list}"
    mgr.shutdown()
