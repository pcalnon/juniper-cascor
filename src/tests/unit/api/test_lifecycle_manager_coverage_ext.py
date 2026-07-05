#!/usr/bin/env python
"""Extended branch coverage for ``TrainingLifecycleManager`` methods
(per-file coverage lift 4, C-5).

Covers the scattered previously-uncovered arms across the manager surface:
network-guard rejections, the broadcast-throttle emission guard, the
interrupt-during-pause path, live-monitoring event / metric-emission guards,
the candidate-progress drain break arms, weight-recorder (re)attach,
session-gauge emission guards, pause/resume happy paths, dataset-data with
validation, param-update rollback, optimizer-state zeroing, manual hidden-unit
validation, snapshot save/load/list, restore/retrain/resume edges, replay
control validation, and shutdown teardown. All fast unit tests against real or
lightly-faked networks — no training, no I/O.
"""

import threading
import time
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from api.lifecycle.manager import InvalidCandidatePoolError, TrainingInterrupted, TrainingLifecycleManager
from api.lifecycle.state_machine import Command

pytestmark = pytest.mark.unit


@pytest.fixture
def mgr():
    m = TrainingLifecycleManager()
    try:
        yield m
    finally:
        m.shutdown()


def _attach_network(m, net) -> None:
    m.model = types.SimpleNamespace(network=net)


class TestNetworkGuards:
    """create/delete network reject while training is active."""

    def test_create_network_rejected_while_started(self, mgr):
        with patch.object(mgr.state_machine, "is_started", return_value=True):
            with pytest.raises(RuntimeError, match="while training is active"):
                mgr.create_network(input_size=2, output_size=2)

    def test_delete_network_rejected_while_started(self, mgr):
        with patch.object(mgr.state_machine, "is_started", return_value=True):
            with pytest.raises(RuntimeError, match="while training is active"):
                mgr.delete_network()


class TestBroadcastThrottleGuard:
    """The GAP-WS-21 coalescer's metric-emission guard is defensive."""

    def test_coalesced_emission_failure_swallowed(self, mgr):
        mgr._ws_manager = MagicMock()
        mgr.training_state.update_state(status="Started")  # non-terminal
        mgr._state_throttle_interval = 1.0
        mgr._last_state_broadcast_time = time.monotonic()  # within throttle window
        with patch("api.observability.ws_inc_state_throttle_coalesced", side_effect=RuntimeError("emit down")):
            mgr._broadcast_training_state()  # throttled → emit attempt → swallowed
        # Throttled: the state message itself was not broadcast.
        mgr._ws_manager.broadcast_from_thread.assert_not_called()


class TestAutoSnapCallback:
    """``_maybe_auto_snap_callback`` no-metric short-circuit."""

    def test_no_usable_metric_returns(self, mgr):
        mgr._auto_snap_best = True
        mgr._auto_snap_min_epochs = 0
        with patch.object(mgr, "save_snapshot") as save:
            mgr._maybe_auto_snap_callback(metrics={}, epoch=5, accuracy=None)
        save.assert_not_called()


class TestCheckForInterrupt:
    """``_check_for_interrupt`` raises on stop signalled during pause."""

    def test_stop_during_pause_raises(self, mgr):
        mgr._stop_event.clear()
        mgr._pause_event.clear()  # paused → enters the wait loop
        timer = threading.Timer(0.05, mgr._stop_event.set)
        timer.start()
        try:
            with pytest.raises(TrainingInterrupted):
                mgr._check_for_interrupt()
        finally:
            timer.cancel()


class TestHandleEventGuards:
    """``_handle_event`` epoch_end step-duration emission guard."""

    def test_step_duration_emission_failure_swallowed(self, mgr):
        mgr.create_network(input_size=2, output_size=2)
        mgr._step_timer_prev = 1.0  # prior timestamp → duration emitted
        event = types.SimpleNamespace(type="epoch_end", payload={"metrics": {"loss": 0.5}, "epoch": 3})
        with patch("api.lifecycle.manager.observe_training_step_duration", side_effect=RuntimeError("hist down")):
            mgr._handle_event(event)  # emission failure swallowed


class TestExtractMetricsGuards:
    """``_extract_and_record_metrics`` counter / gauge emission guards."""

    def _prime_history(self, mgr):
        mgr.create_network(input_size=2, output_size=2)
        mgr.network.history = {
            "train_loss": [0.5],
            "train_accuracy": [0.6],
            "value_loss": [0.55],
            "value_accuracy": [0.55],
        }
        mgr._last_emitted_history_len = 0

    def test_counter_emission_failure_swallowed(self, mgr):
        self._prime_history(mgr)
        with patch("api.lifecycle.manager.record_training_epoch", side_effect=RuntimeError("counter down")):
            mgr._extract_and_record_metrics()

    def test_gauge_emission_failure_swallowed(self, mgr):
        self._prime_history(mgr)
        with patch("api.lifecycle.manager.set_training_loss", side_effect=RuntimeError("gauge down")):
            mgr._extract_and_record_metrics()


class TestDrainProgressQueue:
    """``_drain_progress_queue`` break arms (best-effort)."""

    def test_wait_exception_breaks(self, mgr):
        stop_event = MagicMock()
        stop_event.is_set.return_value = False
        stop_event.wait.side_effect = RuntimeError("wait blew up")
        network_ref = types.SimpleNamespace()  # no _persistent_progress_queue → _pq stays None
        # Must return promptly (break) rather than loop forever.
        TrainingLifecycleManager._drain_progress_queue(network_ref, stop_event, MagicMock(), MagicMock(), mgr)

    def test_queue_get_exception_breaks(self, mgr):
        stop_event = threading.Event()  # clear
        bad_queue = MagicMock()
        bad_queue.get.side_effect = RuntimeError("queue exploded")
        network_ref = types.SimpleNamespace(_persistent_progress_queue=bad_queue)
        TrainingLifecycleManager._drain_progress_queue(network_ref, stop_event, MagicMock(), MagicMock(), mgr)


class TestAttachWeightRecorder:
    """``_attach_weight_history_recorder`` no-network + re-init arms."""

    def test_no_network_returns(self, mgr):
        assert mgr.network is None
        mgr._attach_weight_history_recorder()  # early return, no recorder created
        assert mgr._weight_history_recorder is None

    def test_reinit_existing_recorder(self, mgr):
        mgr.create_network(input_size=2, output_size=2)
        mgr._attach_weight_history_recorder()
        first = mgr._weight_history_recorder
        assert first is not None
        # Second attach against the same network re-inits the existing recorder.
        mgr._attach_weight_history_recorder()
        assert mgr._weight_history_recorder is first


class TestStartTrainingPendingReload:
    """``start_training`` consumes a staged dataset config before running."""

    def test_pending_config_is_reloaded(self, mgr):
        mgr.create_network(input_size=2, output_size=2)
        mgr._pending_dataset_config = {"dataset_type": "spiral"}
        with patch.object(mgr, "_reload_dataset") as reload:
            # No data supplied and the mocked reload doesn't set any → the
            # subsequent "no training data" guard raises, but the pending
            # config was consumed first.
            with pytest.raises(ValueError, match="Training data not provided"):
                mgr.start_training()
        reload.assert_called_once()
        assert mgr._pending_dataset_config is None


class TestRunTrainingSessionGaugeGuards:
    """``_run_training`` session-active gauge inc/dec guards are defensive."""

    def test_inc_and_dec_emission_failures_swallowed(self, mgr):
        mgr.create_network(input_size=2, output_size=2)
        x = torch.zeros(4, 2)
        y = torch.zeros(4, 2)
        with patch.object(mgr.model, "fit"), patch("api.lifecycle.manager.inc_training_sessions", side_effect=RuntimeError("inc down")), patch("api.lifecycle.manager.dec_training_sessions", side_effect=RuntimeError("dec down")):
            mgr._run_training(x, y, None, None)  # both guards exercised; completes cleanly
        assert mgr.state_machine.status.name in {"COMPLETED", "STOPPED"}


class TestPauseResumeHappyPath:
    """pause/resume success transitions (FSM stubbed to the active states)."""

    def test_pause_training_when_started(self, mgr):
        mgr.state_machine = MagicMock()
        mgr.state_machine.is_started.return_value = True
        result = mgr.pause_training()
        assert result["status"] == "paused"
        assert not mgr._pause_event.is_set()

    def test_resume_training_when_paused(self, mgr):
        mgr.state_machine = MagicMock()
        mgr.state_machine.is_paused.return_value = True
        result = mgr.resume_training()
        assert result["status"] == "resumed"
        assert mgr._pause_event.is_set()


class TestGetMetricsAndDataset:
    """``get_metrics`` error guard + ``get_dataset_data`` validation branch."""

    def test_get_metrics_history_error_returns_empty(self, mgr):
        mgr.create_network(input_size=2, output_size=2)
        bad_history = MagicMock()
        bad_history.get.side_effect = RuntimeError("concurrent mutation")
        mgr.network.history = bad_history
        assert mgr.get_metrics() == {}

    def test_get_dataset_data_includes_validation(self, mgr):
        mgr._train_x = torch.randn(10, 2)
        mgr._train_y = torch.randn(10, 2)
        mgr._val_x = torch.randn(3, 2)
        mgr._val_y = torch.randn(3, 2)
        result = mgr.get_dataset_data()
        assert "val_x" in result and "val_y" in result
        assert len(result["val_x"]) == 3


class TestUpdateParamsRollback:
    """``_apply_params_unlocked`` triple validation + atomic rollback."""

    def test_invalid_candidate_pool_triple_raises(self, mgr):
        mgr.create_network(input_size=2, output_size=2)
        with pytest.raises(InvalidCandidatePoolError):
            mgr.update_params({"selected_candidates": 5, "candidate_pool_size": 2})

    def test_rollback_reverts_optimizer_on_activation_failure(self, mgr):
        net = types.SimpleNamespace(
            config=types.SimpleNamespace(
                optimizer_config=types.SimpleNamespace(optimizer_type="Adam"),
                activation_function_name="Tanh",
            ),
            activation_function_name="Tanh",
        )
        # Re-init raises → the activation write fails after the optimizer write
        # succeeded, so the rollback must revert optimizer_type back to Adam.
        net._init_activation_function = MagicMock(side_effect=RuntimeError("reinit boom"))
        _attach_network(mgr, net)
        with pytest.raises(RuntimeError, match="reinit boom"):
            mgr.update_params({"optimizer_type": "SGD", "activation_function_name": "ReLU"})
        assert net.config.optimizer_config.optimizer_type == "Adam"


class TestZeroOptimizerState:
    """``_zero_optimizer_state_for`` zeroes running buffers, skips the rest."""

    def test_zeroes_running_buffers_only(self, mgr):
        param = object()
        exp_avg = torch.ones(2, 2)
        step = torch.tensor(5.0)  # 0-dim → skipped
        optimizer = types.SimpleNamespace(state={param: {"exp_avg": exp_avg, "step": step, "note": "not-a-tensor"}})
        net = types.SimpleNamespace(output_optimizer=optimizer)
        _attach_network(mgr, net)
        mgr._zero_optimizer_state_for(param)
        assert torch.count_nonzero(optimizer.state[param]["exp_avg"]) == 0
        assert optimizer.state[param]["step"].item() == 5.0  # untouched
        assert optimizer.state[param]["note"] == "not-a-tensor"


class TestAddHiddenUnitValidation:
    """``add_hidden_unit_manual`` invalid-weights arm."""

    def test_invalid_weights_returns_nan_inf(self, mgr):
        mgr.create_network(input_size=2, output_size=2)
        mgr.network.activation_functions_dict = {"Tanh": lambda x: x}
        with patch.object(mgr.state_machine, "is_investigating", return_value=True):
            result = mgr.add_hidden_unit_manual(weights=[[1.0, 2.0], [3.0]], activation="Tanh")
        assert result["status"] == mgr._ADD_NAN_INF


class TestSnapshotSaveLoad:
    """save/load snapshot edges."""

    def test_save_snapshot_no_network_returns_none(self, mgr):
        assert mgr.save_snapshot() is None

    def test_save_snapshot_serializer_failure_returns_none(self, mgr, tmp_path):
        mgr.create_network(input_size=2, output_size=2)
        fake_serializer = MagicMock()
        fake_serializer.save_network.return_value = False
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer", return_value=fake_serializer):
            assert mgr.save_snapshot(description="x") is None

    def test_load_snapshot_injects_worker_coordinator(self, mgr, tmp_path):
        (tmp_path / "snap_wc.h5").write_bytes(b"stub")
        loaded_net = types.SimpleNamespace(set_worker_coordinator=MagicMock())
        fake_serializer = MagicMock()
        fake_serializer.load_network.return_value = loaded_net
        mgr._worker_coordinator = MagicMock()
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path), patch("snapshots.snapshot_serializer.CascadeHDF5Serializer", return_value=fake_serializer), patch("api.lifecycle.manager.CascorModel", side_effect=lambda network: types.SimpleNamespace(network=network)):
            ok = mgr._load_snapshot_to_network("snap_wc")
        assert ok is True
        loaded_net.set_worker_coordinator.assert_called_once_with(mgr._worker_coordinator)

    def test_load_snapshot_not_found(self, mgr, tmp_path):
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path):
            result = mgr.load_snapshot("does-not-exist")
        assert result["loaded"] is False


class TestRestoreRetrainResume:
    """restore/retrain/resume rejection + tolerant-history edges."""

    def test_restore_for_retrain_rejected_while_active(self, mgr):
        with patch.object(mgr.state_machine, "is_started", return_value=True):
            result = mgr.restore_for_retrain("snap")
        assert result["loaded"] is False

    def test_restore_for_retrain_history_reset_fallback(self, mgr):
        class _NoDefaultList(list):
            def __init__(self, *args):
                if not args:
                    raise TypeError("requires an argument")
                super().__init__(args[0])

        net = types.SimpleNamespace(history={"train_loss": _NoDefaultList([0.1, 0.2])})
        _attach_network(mgr, net)
        with patch.object(mgr, "_load_snapshot_to_network", return_value=True):
            result = mgr.restore_for_retrain("snap")
        assert result["loaded"] is True
        # The un-reconstructable container fell back to a plain empty list.
        assert net.history["train_loss"] == []

    def test_resume_from_snapshot_tolerates_unsized_history(self, mgr):
        net = types.SimpleNamespace(history={"train_loss": 5})  # int has no len()
        _attach_network(mgr, net)
        with patch.object(mgr, "_load_snapshot_to_network", return_value=True):
            result = mgr.resume_from_snapshot("snap")
        assert result["loaded"] is True
        assert mgr._resume_point_epoch == 0


class TestReplayControlAndStart:
    """start_replay teardown-error arm + replay_control param validation."""

    def test_start_replay_tolerates_prior_session_stop_error(self, mgr):
        prior = MagicMock()
        prior.stop.side_effect = RuntimeError("stuck thread")
        mgr._replay_session = prior
        net = types.SimpleNamespace(history={}, weight_history=None)
        _attach_network(mgr, net)
        with patch.object(mgr, "_load_snapshot_to_network", return_value=True):
            ok = mgr.start_replay("snap")
        assert ok is True
        prior.stop.assert_called_once()

    def test_replay_control_speed_requires_value(self, mgr):
        mgr._replay_session = MagicMock()
        with patch.object(mgr.state_machine, "is_replaying", return_value=True):
            with pytest.raises(ValueError, match="speed requires"):
                mgr.replay_control("speed")

    def test_replay_control_range_requires_bounds(self, mgr):
        mgr._replay_session = MagicMock()
        with patch.object(mgr.state_machine, "is_replaying", return_value=True):
            with pytest.raises(ValueError, match="range requires"):
                mgr.replay_control("range", start=0)


class TestSnapshotListing:
    """list_snapshots / get_snapshot over a stubbed snapshots dir."""

    def test_list_snapshots_returns_metadata(self, mgr, tmp_path):
        (tmp_path / "snapshot_A.h5").write_bytes(b"a")
        (tmp_path / "snapshot_B.h5").write_bytes(b"bb")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path):
            snaps = mgr.list_snapshots()
        assert {s["id"] for s in snaps} == {"snapshot_A", "snapshot_B"}
        assert all("size_bytes" in s and "modified" in s for s in snaps)

    def test_get_snapshot_found_and_missing(self, mgr, tmp_path):
        (tmp_path / "snapshot_C.h5").write_bytes(b"ccc")
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path):
            found = mgr.get_snapshot("snapshot_C")
            missing = mgr.get_snapshot("snapshot_Z")
        assert found is not None and found["id"] == "snapshot_C"
        assert missing is None


class TestShutdownTeardown:
    """``shutdown`` drains an active replay session + executor."""

    def test_shutdown_stops_replay_and_executor(self):
        m = TrainingLifecycleManager()
        replay = MagicMock()
        replay.stop.side_effect = RuntimeError("stuck")  # tolerated
        m._replay_session = replay
        executor = MagicMock()
        m._executor = executor
        m.shutdown()  # single explicit shutdown (no fixture double-call)
        replay.stop.assert_called_once()
        # shutdown() nulls _executor after shutting it down; assert on the ref.
        executor.shutdown.assert_called_once()
        assert m._replay_session is None
        assert m._executor is None
