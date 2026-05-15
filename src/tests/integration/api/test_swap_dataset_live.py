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
# swap_dataset_live: shrink-via-padding (P2-1d — network monotonically grows;
# shrink is handled by zero-padding the dataset up to network capacity)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_swap_dataset_live_input_shrink_pads_dataset_tensors():
    """Per the P2-1d contract, a swap that brings a smaller-input dataset
    succeeds: the dataset tensors are zero-padded up to ``network.input_size``
    (which never shrinks). The pre-pad active dim is captured on the
    response but the live tensors stored on the manager reflect the padded
    shape so subsequent ``forward()`` calls see consistent dims."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=5, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x = torch.randn(8, 5)
    mgr._train_y = torch.zeros(8, 2)
    mgr._train_y[:, 0] = 1
    mgr._current_dataset_config = {"dataset_type": "synthetic_5_2"}
    mgr.state_machine.handle_command(Command.START)

    def _fake_reload(**cfg):
        mgr._train_x = torch.randn(8, 2)  # input shrunk 5→2
        mgr._train_y = torch.zeros(8, 2)
        mgr._train_y[:, 0] = 1
        mgr._val_x = None
        mgr._val_y = None
        mgr._current_dataset_config = {"dataset_type": "synthetic_2_2"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        result = mgr.swap_dataset_live(dataset_type="synthetic_2_2")

    # Network monotonically non-decreasing.
    assert mgr.network.input_size == 5, "network.input_size must NOT shrink"
    # Dataset tensors padded up to network dim.
    assert mgr._train_x.shape == (8, 5), "input tensor padded to network input_size"
    # The last 3 columns should be zeros (the pad).
    assert torch.equal(mgr._train_x[:, 2:], torch.zeros(8, 3))
    # Response reports the dataset-side delta (negative on shrink) but no
    # network-side append (network didn't grow).
    assert result["arch_changes"]["input_delta"] == -3, "input_delta is dataset-vs-pre-swap diff"
    assert result["arch_changes"]["appended_nodes"] == {"input": 0, "output": 0}
    assert result["arch_changes"]["active_output_dim"] == 2  # output dim unchanged
    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_output_shrink_pads_targets_and_sets_active_output_dim():
    """An output-shrink swap zero-pads ``_train_y`` and sets
    ``network.active_output_dim`` so the next training run masks loss to
    the real output slots (avoiding zero-target gradient drift)."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=3)
    mgr.set_experimental_functions(True)
    mgr._train_x = torch.randn(8, 2)
    mgr._train_y = torch.zeros(8, 3)
    mgr._train_y[:, 0] = 1
    mgr._current_dataset_config = {"dataset_type": "synthetic_2_3"}
    mgr.state_machine.handle_command(Command.START)

    def _fake_reload(**cfg):
        mgr._train_x = torch.randn(8, 2)
        mgr._train_y = torch.zeros(8, 2)
        mgr._train_y[:, 0] = 1
        mgr._val_x = None
        mgr._val_y = None
        mgr._current_dataset_config = {"dataset_type": "synthetic_2_2"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        result = mgr.swap_dataset_live(dataset_type="synthetic_2_2")

    assert mgr.network.output_size == 3, "network output dim never shrinks"
    assert mgr._train_y.shape == (8, 3), "targets padded up to network output_size"
    assert torch.equal(mgr._train_y[:, 2:], torch.zeros(8, 1))  # one zero column
    assert mgr.network.active_output_dim == 2, "loss-mask depth = pre-pad dataset output dim"
    assert result["arch_changes"]["output_delta"] == -1
    assert result["arch_changes"]["active_output_dim"] == 2
    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_mixed_input_grow_output_shrink():
    """Composes grow on one side with pad on the other: input dim grows on
    the network; output dim is dataset-padded up to network."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=3)
    mgr.set_experimental_functions(True)
    mgr._train_x = torch.randn(8, 2)
    mgr._train_y = torch.zeros(8, 3)
    mgr._train_y[:, 0] = 1
    mgr._current_dataset_config = {"dataset_type": "synthetic_2_3"}
    mgr.state_machine.handle_command(Command.START)

    def _fake_reload(**cfg):
        # Input grows 2→5, output shrinks 3→1.
        mgr._train_x = torch.randn(8, 5)
        mgr._train_y = torch.zeros(8, 1)
        mgr._train_y[:, 0] = 1
        mgr._val_x = None
        mgr._val_y = None
        mgr._current_dataset_config = {"dataset_type": "synthetic_5_1"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        result = mgr.swap_dataset_live(dataset_type="synthetic_5_1")

    # Network grew to (5, 3); dataset output padded back up to 3.
    assert mgr.network.input_size == 5
    assert mgr.network.output_size == 3
    assert mgr._train_x.shape == (8, 5)
    assert mgr._train_y.shape == (8, 3)
    assert mgr.network.active_output_dim == 1
    assert result["arch_changes"]["input_delta"] == 3  # 2 → 5
    assert result["arch_changes"]["output_delta"] == -2  # 3 → 1
    assert result["arch_changes"]["appended_nodes"] == {"input": 3, "output": 0}
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


# ---------------------------------------------------------------------------
# P2-1c: grow-input / grow-output happy paths through swap_dataset_live
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_swap_dataset_live_grow_input_success():
    """Live swap from input_size=2 to input_size=4 succeeds. Response surfaces
    ``input_delta=2`` + ``appended_nodes.input=2`` per §3.3. The architecture
    adapter handles the weight-tensor expansion under the hood; this test
    pins the response contract callers (canopy) rely on."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)

    new_x, new_y = _make_dummy_tensors(4, 2)

    def _fake_reload(**cfg):
        mgr._train_x = new_x
        mgr._train_y = new_y
        mgr._current_dataset_config = {"dataset_type": "synthetic_4_2"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        result = mgr.swap_dataset_live(dataset_type="synthetic_4_2")

    assert result["status"] == "swapped"
    assert result["arch_changes"]["input_delta"] == 2
    assert result["arch_changes"]["output_delta"] == 0
    assert result["arch_changes"]["appended_nodes"] == {"input": 2, "output": 0}
    assert result["arch_changes"]["prepended_layers"] == []
    # Network input_size was actually updated in place.
    assert mgr.network.input_size == 4
    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_grow_output_success():
    """Live swap from output_size=2 to output_size=4 succeeds with
    ``output_delta=2``. Bias + output_weights gain zero-init columns/elements."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)

    new_x, new_y = _make_dummy_tensors(2, 4)

    def _fake_reload(**cfg):
        mgr._train_x = new_x
        mgr._train_y = new_y
        mgr._current_dataset_config = {"dataset_type": "synthetic_2_4"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        result = mgr.swap_dataset_live(dataset_type="synthetic_2_4")

    assert result["arch_changes"]["input_delta"] == 0
    assert result["arch_changes"]["output_delta"] == 2
    assert result["arch_changes"]["appended_nodes"] == {"input": 0, "output": 2}
    assert mgr.network.output_size == 4
    assert mgr.network.output_weights.shape[1] == 4
    assert mgr.network.output_bias.shape == torch.Size([4])
    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_grow_both_input_and_output():
    """Mixed grow (input AND output expand) propagates both deltas in the
    response. Pins that grow-input + grow-output compose without one
    clobbering the other's tensor mutations."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)

    new_x, new_y = _make_dummy_tensors(5, 4)

    def _fake_reload(**cfg):
        mgr._train_x = new_x
        mgr._train_y = new_y
        mgr._current_dataset_config = {"dataset_type": "synthetic_5_4"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        result = mgr.swap_dataset_live(dataset_type="synthetic_5_4")

    assert result["arch_changes"]["input_delta"] == 3
    assert result["arch_changes"]["output_delta"] == 2
    assert result["arch_changes"]["appended_nodes"] == {"input": 3, "output": 2}
    assert mgr.network.input_size == 5
    assert mgr.network.output_size == 4
    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_grow_input_log_line_reports_actual_deltas():
    """§3.7 #5 log line reflects the real dim deltas (pre-P2-1c it always
    showed equal-dim because the adapter wasn't wired). Canopy log scraping
    + human readability both depend on this format being accurate."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)

    new_x, new_y = _make_dummy_tensors(5, 3)

    def _fake_reload(**cfg):
        mgr._train_x = new_x
        mgr._train_y = new_y
        mgr._current_dataset_config = {"dataset_type": "synthetic_5_3"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr.logger, "info") as info_mock:
        with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
            mgr.swap_dataset_live(dataset_type="synthetic_5_3")

    matching = [c for c in info_mock.call_args_list if c.args and isinstance(c.args[0], str) and c.args[0].startswith("swap: input ")]
    assert len(matching) == 1, f"expected 1 completion line, got {info_mock.call_args_list}"
    fmt, *args = matching[0].args
    rendered = fmt % tuple(args)
    assert rendered == "swap: input 2→5, output 2→3, hidden 0 preserved, candidates 0 abandoned, mode→output_training"
    mgr.shutdown()


# ---------------------------------------------------------------------------
# P2-2 (Issue #3): dataset_swap history-event recording from the lifecycle.
# Pins the success-only contract — cancelled / failed swaps must NOT append
# an event, so the history reflects only what canopy should render.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_swap_dataset_live_records_history_event_on_success():
    """Happy path: a successful swap appends exactly one event to
    ``network.history["dataset_swaps"]`` whose payload mirrors the §3.3
    response shape. This is the canopy P2-7 timeline-marker entrypoint."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)

    new_x, new_y = _make_dummy_tensors(4, 2)

    def _fake_reload(**cfg):
        mgr._train_x = new_x
        mgr._train_y = new_y
        mgr._current_dataset_config = {"dataset_type": "synthetic_4_2"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    pre_count = len(mgr.network.history["dataset_swaps"])
    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        result = mgr.swap_dataset_live(dataset_type="synthetic_4_2")

    swaps = mgr.network.history["dataset_swaps"]
    assert len(swaps) == pre_count + 1, "exactly one history event appended per successful swap"
    event = swaps[-1]
    # Schema present.
    for key in ("timestamp", "before_cfg", "after_cfg", "arch_changes", "pre_swap_snapshot_id", "post_swap_snapshot_id"):
        assert key in event, f"missing schema field {key!r}"
    # Payload aligns with the §3.3 response (P2-2 records the SAME
    # arch_changes dict that the route returns to canopy).
    assert event["arch_changes"] == result["arch_changes"]
    assert event["before_cfg"] == {"dataset_type": "spirals"}
    assert event["after_cfg"] == {"dataset_type": "synthetic_4_2"}
    # P2-3 snapshot IDs are populated by the pre/post-swap auto-snap.
    # Both should be strings matching the ``snapshot_<TIMESTAMP>[_<n>]``
    # format. The dedicated P2-3 tests below assert ordering and contents
    # in detail; here we just confirm the threading is wired.
    assert isinstance(event["pre_swap_snapshot_id"], str)
    assert event["pre_swap_snapshot_id"].startswith("snapshot_")
    assert isinstance(event["post_swap_snapshot_id"], str)
    assert event["post_swap_snapshot_id"].startswith("snapshot_")
    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_does_NOT_record_on_cancellation():
    """A cancelled swap rolls back state and must NOT leave a stale
    history event — canopy would render a swap that didn't happen."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)

    def _fake_reload_then_cancel(**cfg):
        mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
        mgr._current_dataset_config = {"dataset_type": "moons"}
        mgr._swap_cancel_requested.set()

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    pre_count = len(mgr.network.history["dataset_swaps"])
    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload_then_cancel):
        with pytest.raises(SwapCancelledError):
            mgr.swap_dataset_live(dataset_type="moons")

    assert len(mgr.network.history["dataset_swaps"]) == pre_count, "cancelled swap must not append an event"
    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_does_NOT_record_on_fetch_failure():
    """``_reload_dataset`` raising before the network mutation → no event.
    The rollback path returns the lifecycle to pre-swap state; the
    history must reflect that nothing was committed."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)

    pre_count = len(mgr.network.history["dataset_swaps"])
    with patch.object(mgr, "_reload_dataset", side_effect=RuntimeError("juniper-data unreachable")):
        with pytest.raises(RuntimeError, match="juniper-data unreachable"):
            mgr.swap_dataset_live(dataset_type="spirals")

    assert len(mgr.network.history["dataset_swaps"]) == pre_count
    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_records_equal_dim_swap():
    """Equal-dim swaps are still real user-initiated events (dataset
    metadata changed; candidates may have been abandoned). The history
    records them so canopy's timeline doesn't silently drop them."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals", "n_spirals": 2}
    mgr.state_machine.handle_command(Command.START)

    new_x, new_y = _make_dummy_tensors(2, 2)

    def _fake_reload(**cfg):
        mgr._train_x = new_x
        mgr._train_y = new_y
        mgr._current_dataset_config = {"dataset_type": "moons", "noise": 0.2}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
        mgr.swap_dataset_live(dataset_type="moons", noise=0.2)

    swaps = mgr.network.history["dataset_swaps"]
    assert len(swaps) == 1
    assert swaps[0]["before_cfg"] == {"dataset_type": "spirals", "n_spirals": 2}
    assert swaps[0]["after_cfg"] == {"dataset_type": "moons", "noise": 0.2}
    assert swaps[0]["arch_changes"]["input_delta"] == 0
    assert swaps[0]["arch_changes"]["output_delta"] == 0
    mgr.shutdown()


# ---------------------------------------------------------------------------
# P2-3 (Issue #3): pre + post-swap auto-snapshot + ID threading into the
# dataset_swap event. The replay-engine rework is deferred per
# ``notes/PHASE_2_P2_3_FOLLOWUP_REPLAY_REWORK_2026-05-14.md`` — these tests
# pin only the snapshot infrastructure that P2-3 actually ships.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_swap_dataset_live_takes_pre_and_post_snapshots():
    """Successful swap triggers exactly two ``save_snapshot`` calls — one
    pre-swap (before resize), one post-swap (after the new training
    future submits). Both IDs are threaded into the history event so
    canopy P2-7 can drive a snapshot-orchestrated transition."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals", "n_spirals": 2}
    mgr.state_machine.handle_command(Command.START)

    new_x, new_y = _make_dummy_tensors(4, 2)

    def _fake_reload(**cfg):
        mgr._train_x = new_x
        mgr._train_y = new_y
        mgr._current_dataset_config = {"dataset_type": "moons", "noise": 0.2}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    # Patch save_snapshot to return stable IDs without writing HDF5 to
    # disk — the file-IO behaviour is covered by the unit tests.
    saved_descriptions = []

    def _fake_save(description=""):
        saved_descriptions.append(description)
        return {
            "id": f"snapshot_test_{len(saved_descriptions):02d}",
            "path": f"/tmp/{len(saved_descriptions):02d}.h5",
            "timestamp": "20260514T000000Z",
            "description": description,
        }

    with patch.object(mgr, "save_snapshot", side_effect=_fake_save):
        with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
            result = mgr.swap_dataset_live(dataset_type="moons", noise=0.2)

    # Two snapshots fired, in order.
    assert len(saved_descriptions) == 2
    assert "pre-swap" in saved_descriptions[0]
    assert "post-swap" in saved_descriptions[1]
    # Descriptions encode swap metadata (deliverable Q2 of the design).
    assert "spirals" in saved_descriptions[0], f"pre-swap description should name the before dataset: {saved_descriptions[0]!r}"
    assert "moons" in saved_descriptions[1], f"post-swap description should name the after dataset: {saved_descriptions[1]!r}"
    # Event has both IDs threaded.
    swaps = mgr.network.history["dataset_swaps"]
    assert len(swaps) == 1
    assert swaps[0]["pre_swap_snapshot_id"] == "snapshot_test_01"
    assert swaps[0]["post_swap_snapshot_id"] == "snapshot_test_02"
    # Swap itself succeeded (not just the snapshots).
    assert result["status"] == "swapped"
    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_pre_swap_snapshot_taken_before_resize():
    """Ordering invariant: the pre-swap snapshot must capture the network
    BEFORE ``_resize_network_for_dataset`` mutates it. If the snapshot
    fired after resize, ``pre_swap_snapshot_id`` would point at a
    network with the post-swap dims, defeating the whole "Restore from
    pre-swap" P2-7 affordance."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)

    new_x, new_y = _make_dummy_tensors(5, 3)

    def _fake_reload(**cfg):
        mgr._train_x = new_x
        mgr._train_y = new_y
        mgr._current_dataset_config = {"dataset_type": "moons"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    observed_dims = []

    def _fake_save(description=""):
        # Capture the network's dims at the moment of each snapshot call.
        observed_dims.append((description, mgr.network.input_size, mgr.network.output_size))
        return {"id": f"snap_{len(observed_dims)}", "path": "/tmp/x", "timestamp": "T", "description": description}

    with patch.object(mgr, "save_snapshot", side_effect=_fake_save):
        with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
            mgr.swap_dataset_live(dataset_type="moons")

    assert len(observed_dims) == 2
    pre_desc, pre_in, pre_out = observed_dims[0]
    post_desc, post_in, post_out = observed_dims[1]
    assert "pre-swap" in pre_desc
    # Pre-swap snapshot saw the OLD dims.
    assert pre_in == 2 and pre_out == 2
    assert "post-swap" in post_desc
    # Post-swap snapshot saw the NEW dims (after grow).
    assert post_in == 5 and post_out == 3
    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_pre_swap_snapshot_kept_on_cancel():
    """Per the P2-3 design Q1 decision: a pre-swap snapshot taken before
    the user cancels stays on disk as a valid checkpoint of the moment
    the swap was attempted. The dataset_swap event itself does NOT
    record (the cancel raises before record), so the snapshot exists
    without a corresponding history event — that's intentional."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)

    def _fake_reload_then_cancel(**cfg):
        mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
        mgr._current_dataset_config = {"dataset_type": "moons"}
        mgr._swap_cancel_requested.set()

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    save_calls = []

    def _fake_save(description=""):
        save_calls.append(description)
        return {"id": f"snap_{len(save_calls)}", "path": "/tmp/x", "timestamp": "T", "description": description}

    with patch.object(mgr, "save_snapshot", side_effect=_fake_save):
        with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload_then_cancel):
            with pytest.raises(SwapCancelledError):
                mgr.swap_dataset_live(dataset_type="moons")

    # Pre-swap snap fired; post-swap did NOT (the cancel checkpoint trips
    # before we reach the post-swap snap call site).
    assert len(save_calls) == 1
    assert "pre-swap" in save_calls[0]
    # Event NOT recorded (already covered by P2-2 tests but pinned here
    # to make the snapshot-vs-event asymmetry explicit).
    assert mgr.network.history["dataset_swaps"] == []
    mgr.shutdown()


@pytest.mark.integration
def test_swap_dataset_live_pre_swap_snapshot_failure_does_not_abort_swap():
    """Snapshot writes are observability, not core functionality. A
    failed pre-swap snapshot logs and leaves ``pre_swap_snapshot_id`` as
    None in the recorded event — the swap itself still succeeds."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.set_experimental_functions(True)
    mgr._train_x, mgr._train_y = _make_dummy_tensors(2, 2)
    mgr._current_dataset_config = {"dataset_type": "spirals"}
    mgr.state_machine.handle_command(Command.START)

    new_x, new_y = _make_dummy_tensors(2, 2)

    def _fake_reload(**cfg):
        mgr._train_x = new_x
        mgr._train_y = new_y
        mgr._current_dataset_config = {"dataset_type": "moons"}

    mgr._executor = MagicMock()
    mgr._executor.submit.return_value = MagicMock()

    call_count = {"n": 0}

    def _fake_save_pre_fails(description=""):
        call_count["n"] += 1
        # First call (pre-swap) raises; second (post-swap) succeeds.
        if call_count["n"] == 1:
            raise RuntimeError("disk full simulation")
        return {"id": "snap_post_ok", "path": "/tmp/x", "timestamp": "T", "description": description}

    with patch.object(mgr, "save_snapshot", side_effect=_fake_save_pre_fails):
        with patch.object(mgr, "_reload_dataset", side_effect=_fake_reload):
            result = mgr.swap_dataset_live(dataset_type="moons")

    assert result["status"] == "swapped"
    swaps = mgr.network.history["dataset_swaps"]
    assert len(swaps) == 1
    # Pre-swap failed → None; post-swap succeeded → populated.
    assert swaps[0]["pre_swap_snapshot_id"] is None
    assert swaps[0]["post_swap_snapshot_id"] == "snap_post_ok"
    mgr.shutdown()
