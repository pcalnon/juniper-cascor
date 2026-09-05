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

    def test_train_only_artifact_is_refused(self, mgr):
        """§6.1 rule 3: no val AND no test means nothing held out to report from.

        This previously asserted the opposite -- that a train-only artifact loads
        happily with ``_val_x`` left None. It did, and then scored the training
        rows under an evaluation label, which is the defect the three-way
        partition removes. No switch re-enables it.
        """
        arrays = {
            "X_train": np.zeros((5, 2), dtype=np.float32),
            "y_train": np.zeros((5, 2), dtype=np.float32),
        }
        mod, client = _fake_data_client_module(arrays=arrays)
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            with pytest.raises(RuntimeError, match="NEITHER a validation split"):
                mgr._reload_dataset(dataset_type="xor")

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

    def test_train_sample_count_mismatch_raises(self, mgr):
        """X_train/y_train row counts must agree — otherwise fit fails mid-run."""
        arrays = {
            "X_train": np.zeros((4, 2), dtype=np.float32),
            "y_train": np.zeros((3, 2), dtype=np.float32),
        }
        mod, _ = _fake_data_client_module(arrays=arrays)
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            with pytest.raises(RuntimeError, match="train sample count mismatch"):
                mgr._reload_dataset(dataset_type="spiral")
        assert mgr._train_x is None and mgr._train_y is None

    def test_partial_validation_split_raises(self, mgr):
        """One of X_val/y_val without the other is a malformed artifact.

        Repointed from X_test to X_val: this test's NAME was already the right
        one, but before the third partition existed the split cascor validated on
        was the artifact's ``X_test``. Now it is ``X_val``, and the test-split
        sibling below covers the other half.
        """
        arrays = {
            "X_train": np.zeros((4, 2), dtype=np.float32),
            "y_train": np.zeros((4, 2), dtype=np.float32),
            "X_val": np.zeros((2, 2), dtype=np.float32),
        }
        mod, _ = _fake_data_client_module(arrays=arrays)
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            with pytest.raises(RuntimeError, match="partial validation split"):
                mgr._reload_dataset(dataset_type="spiral")
        assert mgr._train_x is None and mgr._val_x is None

    def test_validation_sample_count_mismatch_raises(self, mgr):
        arrays = {
            "X_train": np.zeros((4, 2), dtype=np.float32),
            "y_train": np.zeros((4, 2), dtype=np.float32),
            "X_val": np.zeros((2, 2), dtype=np.float32),
            "y_val": np.zeros((1, 2), dtype=np.float32),
        }
        mod, _ = _fake_data_client_module(arrays=arrays)
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            with pytest.raises(RuntimeError, match="validation sample count mismatch"):
                mgr._reload_dataset(dataset_type="spiral")
        assert mgr._train_x is None and mgr._val_x is None

    def test_partial_test_split_raises(self, mgr):
        """The reported partition gets the same guard as the in-loop one.

        Both run through one helper precisely so they cannot drift; this is the
        half that would silently stop being checked if they ever did.
        """
        arrays = {
            "X_train": np.zeros((4, 2), dtype=np.float32),
            "y_train": np.zeros((4, 2), dtype=np.float32),
            "X_val": np.zeros((2, 2), dtype=np.float32),
            "y_val": np.zeros((2, 2), dtype=np.float32),
            "X_test": np.zeros((2, 2), dtype=np.float32),
        }
        mod, _ = _fake_data_client_module(arrays=arrays)
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            with pytest.raises(RuntimeError, match="partial test split"):
                mgr._reload_dataset(dataset_type="spiral")

    def test_validation_feature_count_mismatch_raises(self, mgr):
        """§6a: a val split whose feature count differs from train is malformed.

        A forward pass on it would fail mid-run, or worse, succeed on a
        coincidentally-broadcastable shape.
        """
        arrays = {
            "X_train": np.zeros((4, 2), dtype=np.float32),
            "y_train": np.zeros((4, 2), dtype=np.float32),
            "X_val": np.zeros((2, 5), dtype=np.float32),
            "y_val": np.zeros((2, 2), dtype=np.float32),
        }
        mod, _ = _fake_data_client_module(arrays=arrays)
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            with pytest.raises(RuntimeError, match="validation feature count mismatch"):
                mgr._reload_dataset(dataset_type="spiral")

    def test_non_2d_train_arrays_raise(self, mgr):
        arrays = {
            "X_train": np.zeros((4,), dtype=np.float32),  # 1-D — not a feature matrix
            "y_train": np.zeros((4, 2), dtype=np.float32),
        }
        mod, _ = _fake_data_client_module(arrays=arrays)
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            with pytest.raises(RuntimeError, match="train arrays must be 2-D.*juniper-recurrence tier"):
                mgr._reload_dataset(dataset_type="spiral")

    def test_malformed_train_payload_raises(self, mgr):
        """Non-numeric train payloads surface as a stable RuntimeError."""
        arrays = {
            "X_train": "not-an-array",
            "y_train": np.zeros((2, 2), dtype=np.float32),
        }
        mod, _ = _fake_data_client_module(arrays=arrays)
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            with pytest.raises(RuntimeError, match="train arrays are malformed"):
                mgr._reload_dataset(dataset_type="spiral")

    def test_canopy_staged_spirals_config_is_translated(self, mgr):
        """The canopy-facing staged dialect (``dataset_type='spirals'``, total
        ``n_samples``, ``rotations``) is translated to juniper-data's registry
        key + spiral params at the fetch boundary; the stored config keeps the
        canopy dialect (PR-B, training-start diagnosis 2026-07-09 — previously
        this reload died at juniper-data with "Unknown generator 'spirals'")."""
        mod, client = _fake_data_client_module()
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            mgr._reload_dataset(dataset_type="spirals", n_samples=1000, noise=0.1, rotations=1.5, n_spirals=2)
        _, kwargs = client.create_dataset.call_args
        assert kwargs["generator"] == "spiral"
        assert kwargs["params"] == {"n_points_per_spiral": 500, "n_rotations": 1.5, "noise": 0.1, "n_spirals": 2}
        assert mgr._current_dataset_config["dataset_type"] == "spirals"
        assert mgr._current_dataset_config["n_samples"] == 1000

    def test_canopy_staged_xor_config_is_translated(self, mgr):
        """XOR: total ``n_samples`` becomes per-quadrant; spiral-only typed
        fields are dropped rather than leaking to a generator that never
        declared them."""
        mod, client = _fake_data_client_module()
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            mgr._reload_dataset(dataset_type="xor", n_samples=1000, noise=0.2, rotations=1.5, n_spirals=2)
        _, kwargs = client.create_dataset.call_args
        assert kwargs["generator"] == "xor"
        assert kwargs["params"] == {"n_points_per_quadrant": 250, "noise": 0.2}

    def test_generic_params_win_over_translated_typed_fields(self, mgr):
        """Caller-supplied generic ``params`` keep winning over the translated
        typed fields (the pre-existing merge contract, preserved via setdefault)."""
        mod, client = _fake_data_client_module()
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"):
            mgr._reload_dataset(dataset_type="spirals", n_samples=1000, params={"n_points_per_spiral": 7})
        _, kwargs = client.create_dataset.call_args
        assert kwargs["params"]["n_points_per_spiral"] == 7

    def test_start_training_consumes_pending_and_creates_network(self, mgr):
        """End-to-end through ``start_training``: a pending canopy-staged dataset
        is consumed and the network is created from the fetched array dims
        (PR-B create-on-start; the fresh-boot UI flow)."""
        mod, client = _fake_data_client_module()
        mgr._pending_dataset_config = {"dataset_type": "spirals", "n_samples": 6, "n_spirals": 2}
        with patch.dict(sys.modules, {"juniper_data_client": mod}), patch("api.secrets.get_secret", return_value=None), patch("api.settings.Settings"), patch.object(mgr, "_run_training"):
            result = mgr.start_training()
            assert result["status"] == "training_started"
            if mgr._training_future is not None:
                mgr._training_future.result(timeout=10)
        assert mgr._pending_dataset_config is None
        assert mgr.network is not None
        assert mgr.network.input_size == 2
        assert mgr.network.output_size == 2
        _, kwargs = client.create_dataset.call_args
        assert kwargs["generator"] == "spiral"
        assert kwargs["params"]["n_points_per_spiral"] == 3


class TestTranslateStagedConfig:
    """Direct unit coverage for canopy→juniper-data dialect translation.

    Happy-path spirals/xor already exercise ``_reload_dataset``; these cases
    pin moons aliasing, zero-clamp arithmetic, and non-spiral field stripping
    without needing a juniper-data stub.
    """

    def test_moons_alias(self):
        generator, params = TrainingLifecycleManager._translate_staged_config("moons", {"n_samples": 50, "noise": 0.1})
        assert generator == "moon"
        assert params == {"n_samples": 50, "noise": 0.1}

    def test_spiral_clamps_zero_n_samples(self):
        generator, params = TrainingLifecycleManager._translate_staged_config("spirals", {"n_samples": 0, "n_spirals": 2})
        assert generator == "spiral"
        assert params["n_points_per_spiral"] == 1
        assert "n_samples" not in params

    def test_spiral_clamps_zero_n_spirals_divisor(self):
        """``n_spirals=0`` must not ZeroDivisionError; divisor clamps via max(1, …)."""
        generator, params = TrainingLifecycleManager._translate_staged_config("spirals", {"n_samples": 10, "n_spirals": 0})
        assert generator == "spiral"
        assert params["n_points_per_spiral"] >= 1
        assert params["n_spirals"] == 0

    def test_gaussian_n_samples_translates_to_per_class(self):
        """W-3: gaussian has no n_samples — a staged total divides by n_classes."""
        generator, params = TrainingLifecycleManager._translate_staged_config("gaussian", {"n_samples": 600, "n_classes": 3, "noise": 0.05})
        assert generator == "gaussian"
        assert params["n_samples_per_class"] == 200
        assert "n_samples" not in params

    def test_gaussian_default_class_divisor(self):
        """Absent n_classes uses juniper-data's default (2)."""
        generator, params = TrainingLifecycleManager._translate_staged_config("gaussian", {"n_samples": 100})
        assert params["n_samples_per_class"] == 50

    def test_gaussian_explicit_per_class_wins(self):
        generator, params = TrainingLifecycleManager._translate_staged_config("gaussian", {"n_samples": 600, "n_samples_per_class": 42})
        assert params["n_samples_per_class"] == 42

    def test_checkerboard_keeps_n_samples_and_n_squares(self):
        """W-3: checkerboard takes both directly; spiral-only fields still strip."""
        generator, params = TrainingLifecycleManager._translate_staged_config("checkerboard", {"n_samples": 2000, "n_squares": 4, "rotations": 1.0})
        assert generator == "checkerboard"
        assert params["n_samples"] == 2000
        assert params["n_squares"] == 4
        assert "rotations" not in params

    def test_n_squares_stripped_from_non_checkerboard(self):
        for dataset_type in ("spirals", "xor", "circles", "gaussian"):
            _, params = TrainingLifecycleManager._translate_staged_config(dataset_type, {"n_samples": 40, "n_squares": 4})
            assert "n_squares" not in params, dataset_type

    def test_non_spiral_strips_spiral_only_fields(self):
        generator, params = TrainingLifecycleManager._translate_staged_config(
            "circles",
            {"n_samples": 50, "rotations": 1.5, "n_spirals": 2, "noise": 0.05},
        )
        assert generator == "circles"
        assert params == {"n_samples": 50, "noise": 0.05}
        assert "rotations" not in params
        assert "n_spirals" not in params

    def test_unknown_generator_passthrough(self):
        generator, params = TrainingLifecycleManager._translate_staged_config("equities", {"max_symbols": 5, "rotations": 1.0})
        assert generator == "equities"
        assert params == {"max_symbols": 5}


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
