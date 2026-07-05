#!/usr/bin/env python
"""Unit coverage for the live dataset-swap support surface of
``TrainingLifecycleManager`` (per-file coverage lift 4, C-5).

Targets the previously-uncovered swap helpers: ``_snapshot_abandoned_
candidate_pool_size``, ``_rollback_pre_swap_state`` (incl. its best-effort
exception arms), ``_reload_dataset`` (via a stubbed ``juniper_data_client``
module — no network I/O), the ``swap_dataset_live`` concurrent-swap guard,
``request_swap_cancel`` / ``_check_swap_cancel``, and the pending-dataset
staging methods. All fast unit tests; the live network is faked so no
training runs.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from api.lifecycle.manager import NoSwapInProgressError, SwapCancelledError, SwapInProgressError, TrainingLifecycleManager, _PreSwapSnapshot
from api.lifecycle.state_machine import TrainingPhase

pytestmark = pytest.mark.unit


@pytest.fixture
def mgr():
    m = TrainingLifecycleManager()
    try:
        yield m
    finally:
        m.shutdown()


def _fake_network(**overrides) -> types.SimpleNamespace:
    net = types.SimpleNamespace(
        input_size=2,
        output_size=2,
        active_output_dim=2,
        output_weights=torch.zeros(2, 2),
        output_bias=torch.zeros(2),
        hidden_units=[{"weights": torch.zeros(3)}],
        candidate_pool_size=8,
    )
    for k, v in overrides.items():
        setattr(net, k, v)
    return net


def _attach_network(m, net) -> None:
    """Install a fake network without going through CascorModel wrapping."""
    m.model = types.SimpleNamespace(network=net)


class TestSnapshotAbandonedCandidatePoolSize:
    """``_snapshot_abandoned_candidate_pool_size`` phase-gated read."""

    def test_non_candidate_phase_returns_zero(self, mgr):
        _attach_network(mgr, _fake_network())
        # Fresh FSM is Idle, not CANDIDATE → 0 abandoned.
        assert mgr._snapshot_abandoned_candidate_pool_size() == 0

    def test_candidate_phase_returns_pool_size(self, mgr):
        _attach_network(mgr, _fake_network(candidate_pool_size=16))
        # The FSM only permits the CANDIDATE phase while STARTED; stub the
        # phase read directly rather than driving a full training run.
        fake_sm = MagicMock()
        fake_sm.phase = TrainingPhase.CANDIDATE
        mgr.state_machine = fake_sm
        assert mgr._snapshot_abandoned_candidate_pool_size() == 16

    def test_phase_read_exception_returns_zero(self, mgr):
        _attach_network(mgr, _fake_network())

        class _Boom:
            @property
            def phase(self):
                raise RuntimeError("fsm unavailable")

        mgr.state_machine = _Boom()
        assert mgr._snapshot_abandoned_candidate_pool_size() == 0


class TestRollbackPreSwapState:
    """``_rollback_pre_swap_state`` restores tensors + sizes; arms are best-effort."""

    def _make_pre(self, **overrides) -> _PreSwapSnapshot:
        defaults = {
            "train_x": torch.ones(4, 2),
            "train_y": torch.zeros(4, 2),
            "val_x": None,
            "val_y": None,
            "state_dict": None,
            "input_size": 2,
            "output_size": 2,
            "dataset_config": {"dataset_type": "spiral"},
            "active_output_dim": 2,
            "output_weights": torch.full((2, 2), 3.0),
            "output_bias": torch.full((2,), 1.0),
            "hidden_unit_weights": [torch.full((3,), 5.0)],
        }
        defaults.update(overrides)
        return _PreSwapSnapshot(**defaults)

    def test_restores_tensors_sizes_and_config(self, mgr):
        net = _fake_network(input_size=9, output_size=9, active_output_dim=1)
        _attach_network(mgr, net)
        pre = self._make_pre()
        mgr._rollback_pre_swap_state(pre)
        assert mgr._train_x is pre.train_x
        assert net.input_size == 2
        assert net.output_size == 2
        assert net.active_output_dim == 2
        assert torch.equal(net.output_weights.detach(), torch.full((2, 2), 3.0))
        assert net.output_weights.requires_grad is True
        assert torch.equal(net.hidden_units[0]["weights"], torch.full((3,), 5.0))
        assert mgr._current_dataset_config == {"dataset_type": "spiral"}

    def test_output_weights_clone_failure_is_logged_not_raised(self, mgr):
        _attach_network(mgr, _fake_network())
        bad = MagicMock()
        bad.clone.side_effect = RuntimeError("clone failed")
        pre = self._make_pre(output_weights=bad, output_bias=None, hidden_unit_weights=None)
        mgr._rollback_pre_swap_state(pre)  # exception swallowed

    def test_output_bias_clone_failure_is_logged_not_raised(self, mgr):
        _attach_network(mgr, _fake_network())
        bad = MagicMock()
        bad.clone.side_effect = RuntimeError("clone failed")
        pre = self._make_pre(output_weights=None, output_bias=bad, hidden_unit_weights=None)
        mgr._rollback_pre_swap_state(pre)

    def test_hidden_unit_restore_failure_is_logged_not_raised(self, mgr):
        _attach_network(mgr, _fake_network())
        bad = MagicMock()
        bad.clone.side_effect = RuntimeError("clone failed")
        pre = self._make_pre(output_weights=None, output_bias=None, hidden_unit_weights=[bad])
        mgr._rollback_pre_swap_state(pre)

    def test_state_dict_load_path_and_failure(self, mgr):
        # A network exposing load_state_dict exercises the legacy nn.Module arm;
        # make it raise so the best-effort except is covered too.
        net = _fake_network(load_state_dict=MagicMock(side_effect=RuntimeError("bad state")))
        _attach_network(mgr, net)
        pre = self._make_pre(state_dict={"weights": 1}, output_weights=None, output_bias=None, hidden_unit_weights=None)
        mgr._rollback_pre_swap_state(pre)
        net.load_state_dict.assert_called_once()


def _fake_data_client_module(*, arrays=None, create_raises=False, missing_key=False):
    mod = types.ModuleType("juniper_data_client")
    client = MagicMock()
    if create_raises:
        client.create_dataset.side_effect = RuntimeError("connection refused")
    else:
        client.create_dataset.return_value = {"dataset_id": "ds-42"}
    if arrays is None:
        arrays = {
            "X_train": np.zeros((6, 2), dtype=np.float32),
            "y_train": np.zeros((6, 2), dtype=np.float32),
            "X_test": np.zeros((2, 2), dtype=np.float32),
            "y_test": np.zeros((2, 2), dtype=np.float32),
        }
    if missing_key:
        arrays = {k: v for k, v in arrays.items() if k != "X_train"}
    client.download_artifact_npz.return_value = arrays
    mod.JuniperDataClient = MagicMock(return_value=client)
    return mod, client


class TestReloadDataset:
    """``_reload_dataset`` — juniper-data fetch path, fully stubbed."""

    def test_import_error_raises_runtime_error(self, mgr):
        with patch.dict(sys.modules, {"juniper_data_client": None}):
            with pytest.raises(RuntimeError, match="juniper-data-client is not installed"):
                mgr._reload_dataset(dataset_type="spiral")

    def test_missing_dataset_type_raises(self, mgr):
        mod, _ = _fake_data_client_module()
        with patch.dict(sys.modules, {"juniper_data_client": mod}):
            with pytest.raises(RuntimeError, match="missing required 'dataset_type'"):
                mgr._reload_dataset(n_samples=100)  # no dataset_type

    def test_happy_path_with_validation_arrays(self, mgr):
        mod, client = _fake_data_client_module()
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            mgr._reload_dataset(dataset_type="spiral", n_samples=6)
        assert mgr._train_x is not None and mgr._train_x.shape[0] == 6
        assert mgr._val_x is not None and mgr._val_x.shape[0] == 2
        assert mgr._current_dataset_config["dataset_type"] == "spiral"

    def test_happy_path_without_validation_arrays(self, mgr):
        arrays = {
            "X_train": np.zeros((5, 2), dtype=np.float32),
            "y_train": np.zeros((5, 2), dtype=np.float32),
        }
        mod, client = _fake_data_client_module(arrays=arrays)
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            mgr._reload_dataset(dataset_type="xor")
        assert mgr._val_x is None and mgr._val_y is None

    def test_generic_params_are_merged(self, mgr):
        mod, client = _fake_data_client_module()
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            mgr._reload_dataset(dataset_type="equities", params={"ticker": "AAPL"})
        # The generic params dict was merged into the create_dataset params.
        _, kwargs = client.create_dataset.call_args
        assert kwargs["params"].get("ticker") == "AAPL"

    def test_fetch_failure_raises_runtime_error(self, mgr):
        mod, _ = _fake_data_client_module(create_raises=True)
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            with pytest.raises(RuntimeError, match="juniper-data fetch failed"):
                mgr._reload_dataset(dataset_type="spiral")

    def test_missing_artifact_key_raises_runtime_error(self, mgr):
        mod, _ = _fake_data_client_module(missing_key=True)
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            with pytest.raises(RuntimeError, match="artifact missing required key"):
                mgr._reload_dataset(dataset_type="spiral")


class TestSwapConcurrencyAndCancel:
    """``swap_dataset_live`` concurrent guard + cancel signalling."""

    def test_concurrent_swap_raises_in_progress(self, mgr):
        mgr._experimental_functions_enabled = True
        mgr._swap_in_progress = True
        with patch.object(mgr.state_machine, "is_started", return_value=True):
            with pytest.raises(SwapInProgressError):
                mgr.swap_dataset_live(dataset_type="spiral")

    def test_request_swap_cancel_without_swap_raises(self, mgr):
        assert mgr._swap_in_progress is False
        with pytest.raises(NoSwapInProgressError):
            mgr.request_swap_cancel()

    def test_request_swap_cancel_sets_flag(self, mgr):
        mgr._swap_in_progress = True
        result = mgr.request_swap_cancel()
        assert result["status"] == "cancel_requested"
        assert mgr._swap_cancel_requested.is_set()

    def test_check_swap_cancel_raises_when_signalled(self, mgr):
        mgr._swap_cancel_requested.set()
        with pytest.raises(SwapCancelledError):
            mgr._check_swap_cancel()

    def test_check_swap_cancel_noop_when_clear(self, mgr):
        mgr._swap_cancel_requested.clear()
        mgr._check_swap_cancel()  # no raise


def _swap_network() -> types.SimpleNamespace:
    """A live-swap-capable fake network (equal-dim, no real resize/fit)."""
    net = types.SimpleNamespace(
        input_size=2,
        output_size=2,
        active_output_dim=2,
        output_weights=torch.zeros(2, 2),
        output_bias=torch.zeros(2),
        hidden_units=[{"weights": torch.zeros(3)}],
        candidate_pool_size=8,
    )
    net._resize_network_for_dataset = MagicMock(return_value={"hidden_preserved": 1, "input_delta": 0, "output_delta": 0})
    net.record_dataset_swap_event = MagicMock(return_value={"event": "dataset_swap", "id": 1})
    return net


def _reload_sets_equal_dim(mgr):
    def _reload(**cfg):
        mgr._train_x = torch.zeros(6, 2)
        mgr._train_y = torch.zeros(6, 2)
        mgr._current_dataset_config = {"dataset_type": cfg.get("dataset_type")}

    return _reload


class TestSwapDatasetLiveHappyPath:
    """``swap_dataset_live`` end-to-end success path (network fully faked)."""

    def test_swap_succeeds_and_broadcasts(self, mgr):
        net = _swap_network()
        _attach_network(mgr, net)
        mgr._experimental_functions_enabled = True
        mgr._ws_manager = MagicMock()
        # A prior training future that raises on join exercises the
        # future-drain except arm.
        prior_future = MagicMock()
        prior_future.result.side_effect = RuntimeError("worker already gone")
        mgr._training_future = prior_future

        with patch.object(mgr.state_machine, "is_started", return_value=True), patch.object(mgr, "save_snapshot", return_value={"id": "snap"}), patch.object(mgr, "_reload_dataset", side_effect=_reload_sets_equal_dim(mgr)), patch.object(mgr, "_run_training"):
            result = mgr.swap_dataset_live(dataset_type="xor")

        assert result["status"] == "swapped"
        assert result["mode"] == "output_training_first"
        net._resize_network_for_dataset.assert_called_once()
        net.record_dataset_swap_event.assert_called_once()
        mgr._ws_manager.broadcast_from_thread.assert_called()
        # In-progress flag cleared in finally.
        assert mgr._swap_in_progress is False

    def test_swap_tolerates_record_event_failure(self, mgr):
        net = _swap_network()
        net.record_dataset_swap_event.side_effect = RuntimeError("history write failed")
        _attach_network(mgr, net)
        mgr._experimental_functions_enabled = True
        mgr._training_future = None

        with patch.object(mgr.state_machine, "is_started", return_value=True), patch.object(mgr, "save_snapshot", return_value={"id": "snap"}), patch.object(mgr, "_reload_dataset", side_effect=_reload_sets_equal_dim(mgr)), patch.object(mgr, "_run_training"):
            result = mgr.swap_dataset_live(dataset_type="xor")
        # The swap itself still succeeds even though the event record failed.
        assert result["status"] == "swapped"


class TestSwapDatasetLiveRollback:
    """``swap_dataset_live`` cancel / validation / generic failure arms."""

    def _base_patches(self, mgr):
        return [
            patch.object(mgr.state_machine, "is_started", return_value=True),
            patch.object(mgr, "save_snapshot", return_value={"id": "snap"}),
            patch.object(mgr, "_run_training"),
        ]

    def test_cancelled_swap_rolls_back(self, mgr):
        net = _swap_network()
        _attach_network(mgr, net)
        mgr._experimental_functions_enabled = True
        mgr._train_x = torch.ones(4, 2)
        mgr._train_y = torch.zeros(4, 2)

        def _reload(**cfg):
            mgr._train_x = torch.zeros(6, 2)
            mgr._train_y = torch.zeros(6, 2)
            mgr._swap_cancel_requested.set()  # trip the post-fetch checkpoint

        with patch.object(mgr.state_machine, "is_started", return_value=True), patch.object(mgr, "save_snapshot", return_value={"id": "snap"}), patch.object(mgr, "_run_training"), patch.object(mgr, "_reload_dataset", side_effect=_reload), patch.object(mgr, "_rollback_pre_swap_state") as rollback:
            with pytest.raises(SwapCancelledError):
                mgr.swap_dataset_live(dataset_type="xor")
        rollback.assert_called_once()
        assert mgr._swap_in_progress is False

    def test_value_error_rolls_back(self, mgr):
        net = _swap_network()
        net._resize_network_for_dataset.side_effect = ValueError("shrink_unsupported")
        _attach_network(mgr, net)
        mgr._experimental_functions_enabled = True

        with patch.object(mgr.state_machine, "is_started", return_value=True), patch.object(mgr, "save_snapshot", return_value={"id": "snap"}), patch.object(mgr, "_run_training"), patch.object(mgr, "_reload_dataset", side_effect=_reload_sets_equal_dim(mgr)), patch.object(mgr, "_rollback_pre_swap_state") as rollback:
            with pytest.raises(ValueError):
                mgr.swap_dataset_live(dataset_type="xor")
        rollback.assert_called_once()

    def test_generic_exception_rolls_back(self, mgr):
        net = _swap_network()
        _attach_network(mgr, net)
        mgr._experimental_functions_enabled = True

        def _reload(**cfg):
            raise RuntimeError("juniper-data unreachable")

        with patch.object(mgr.state_machine, "is_started", return_value=True), patch.object(mgr, "save_snapshot", return_value={"id": "snap"}), patch.object(mgr, "_run_training"), patch.object(mgr, "_reload_dataset", side_effect=_reload), patch.object(mgr, "_rollback_pre_swap_state") as rollback:
            with pytest.raises(RuntimeError):
                mgr.swap_dataset_live(dataset_type="xor")
        rollback.assert_called_once()


class TestPendingDatasetConfig:
    """``stage_dataset_config`` / ``clear_pending_dataset_config``."""

    def test_stage_records_config(self, mgr):
        result = mgr.stage_dataset_config(dataset_type="spiral", n_samples=200)
        assert result["status"] == "staged"
        assert mgr.get_pending_dataset_config() == {"dataset_type": "spiral", "n_samples": 200}

    def test_stage_empty_clears(self, mgr):
        mgr.stage_dataset_config(dataset_type="spiral")
        result = mgr.stage_dataset_config()  # empty cfg → cleared
        assert result["status"] == "cleared"
        assert mgr.get_pending_dataset_config() is None

    def test_clear_pending_returns_discarded(self, mgr):
        mgr.stage_dataset_config(dataset_type="xor")
        result = mgr.clear_pending_dataset_config()
        assert result["status"] == "cleared"
        assert result["discarded"] == {"dataset_type": "xor"}
        assert mgr.get_pending_dataset_config() is None
