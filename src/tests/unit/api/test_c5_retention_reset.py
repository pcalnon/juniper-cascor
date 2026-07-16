#!/usr/bin/env python
"""C5 (Q4/U-1) — metrics/history retention & reset semantics.

Regression coverage for the ratified Q4 posture (juniper-ml
``notes/JUNIPER_2026-07-11_JUNIPER-CANOPY_TRAINING-RUNTIME-DEFECTS-PLAN.md``
§7 row C5 / §12 Q4):

  * **Retain by default across runs** — a new training run no longer empties the
    metrics/history buffer; it is retained for cross-dataset continuity, and a
    retaining run appends only its NEW history rows (no re-emit/duplication of a
    prior run's tail).
  * **Explicit clear + undo** — ``clear_metrics_with_undo`` empties the buffer
    while stashing an undo snapshot; ``undo_clear_metrics`` reverses it, valid
    only until the next run starts (starting a run drops the snapshot → 409).
  * **Start-fresh** — ``start_training(start_fresh=True)`` performs a clean-launch
    reset (discard model + retained metrics/history, rebuild a vanilla network
    from the dataset dims) that PRESERVES on-disk snapshot artifacts.

All manager-level tests are fast: real or lightly-faked networks, ``fit`` mocked
or ``_run_training`` patched — no real training, no waiting on the executor.
Route-level tests drive a ``TestClient`` against the real app.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from api.app import create_app
from api.lifecycle.manager import NoMetricsUndoError, TrainingLifecycleManager
from api.settings import Settings

pytestmark = pytest.mark.unit


@pytest.fixture
def mgr():
    m = TrainingLifecycleManager()
    try:
        yield m
    finally:
        m.shutdown()


@pytest.fixture
def client():
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def _fill_buffer(m, n):
    """Append ``n`` metric rows to the monitor buffer (simulating a prior run)."""
    for i in range(n):
        m.monitor.on_epoch_end(epoch=i + 1, loss=0.5, accuracy=0.5, learning_rate=0.01, hidden_units=0)


def _total(m):
    return m.monitor.get_current_state()["total_metrics"]


# ---------------------------------------------------------------------------
# Explicit clear + undo lifecycle
# ---------------------------------------------------------------------------


class TestC5ClearUndo:
    def test_clear_then_undo_roundtrip(self, mgr):
        _fill_buffer(mgr, 3)
        assert _total(mgr) == 3

        cleared = mgr.clear_metrics_with_undo()
        assert cleared == {"status": "cleared", "cleared_count": 3, "undo_available": True}
        assert _total(mgr) == 0
        assert mgr._metrics_undo_available() is True

        restored = mgr.undo_clear_metrics()
        assert restored == {"status": "restored", "restored_count": 3, "undo_available": False}
        assert _total(mgr) == 3
        assert mgr._metrics_undo_available() is False

    def test_undo_restores_exact_rows(self, mgr):
        _fill_buffer(mgr, 4)
        snapshot = mgr.monitor.get_all_metrics()
        mgr.clear_metrics_with_undo()
        mgr.undo_clear_metrics()
        assert mgr.monitor.get_all_metrics() == snapshot

    def test_undo_without_clear_raises(self, mgr):
        with pytest.raises(NoMetricsUndoError):
            mgr.undo_clear_metrics()

    def test_undo_twice_raises_second_time(self, mgr):
        _fill_buffer(mgr, 2)
        mgr.clear_metrics_with_undo()
        mgr.undo_clear_metrics()
        # The undo snapshot is consumed — a second undo has nothing to restore.
        with pytest.raises(NoMetricsUndoError):
            mgr.undo_clear_metrics()

    def test_undo_dropped_when_run_starts(self, mgr):
        """Starting a run finalizes the clear: the undo is no longer available."""
        _fill_buffer(mgr, 3)
        mgr.clear_metrics_with_undo()
        assert mgr._metrics_undo_available() is True

        # Patch _run_training so start_training only does its synchronous work
        # (consume dataset, create network, drop undo, submit) without training.
        with patch.object(mgr, "_run_training"):
            mgr.start_training(X=torch.zeros(4, 2), y=torch.zeros(4, 2))

        assert mgr._metrics_undo_available() is False
        with pytest.raises(NoMetricsUndoError):
            mgr.undo_clear_metrics()


# ---------------------------------------------------------------------------
# Retention across run boundaries
# ---------------------------------------------------------------------------


class TestC5Retention:
    def test_run_retains_buffer_by_default(self, mgr):
        """A plain run retains the metrics buffer across the run boundary."""
        mgr.create_network(input_size=2, output_size=2)
        _fill_buffer(mgr, 3)
        assert _total(mgr) == 3

        # Default posture is retain (set in __init__ and by start_training).
        assert mgr._retain_metrics_next_run is True
        with patch.object(mgr.model, "fit"):
            mgr._run_training(torch.zeros(4, 2), torch.zeros(4, 2), None, None)

        assert _total(mgr) == 3  # retained, not wiped

    def test_start_fresh_posture_clears_buffer(self, mgr):
        """The clean-launch posture (retain=False) empties the buffer at start."""
        mgr.create_network(input_size=2, output_size=2)
        _fill_buffer(mgr, 3)
        mgr._retain_metrics_next_run = False
        with patch.object(mgr.model, "fit"):
            mgr._run_training(torch.zeros(4, 2), torch.zeros(4, 2), None, None)

        assert _total(mgr) == 0

    def test_retained_run_appends_only_new_history_rows(self, mgr):
        """The retention high-water-mark prevents re-emitting a prior run's tail.

        Run 1 leaves 3 history rows mirrored by 3 buffer rows. A retaining run 2
        that appends 2 new history rows must yield 5 buffer rows (3 retained + 2
        new) — NOT 8 (3 retained + 3+2 re-emitted from a zero baseline, the bug
        the ``_current_history_len`` baseline guards against).
        """
        mgr.create_network(input_size=2, output_size=2)
        net = mgr.network
        # Simulate run 1: 3 history rows, drained into the buffer.
        for _ in range(3):
            net.history["train_loss"].append(0.5)
            net.history["train_accuracy"].append(0.5)
        mgr._last_emitted_history_len = 0
        mgr._extract_and_record_metrics()
        assert _total(mgr) == 3
        assert mgr._last_emitted_history_len == 3

        # Run 2: retaining, fit appends 2 more history rows.
        def fake_fit(*_a, **_k):
            for _ in range(2):
                net.history["train_loss"].append(0.4)
                net.history["train_accuracy"].append(0.6)

        mgr._retain_metrics_next_run = True
        with patch.object(mgr.model, "fit", side_effect=fake_fit):
            mgr._run_training(torch.zeros(4, 2), torch.zeros(4, 2), None, None)

        assert _total(mgr) == 5  # 3 retained + 2 new (no duplication)

    def test_default_retain_flag_initialized(self, mgr):
        assert mgr._retain_metrics_next_run is True
        assert mgr._metrics_undo_available() is False


# ---------------------------------------------------------------------------
# Start-fresh (clean-launch) reset
# ---------------------------------------------------------------------------


class TestC5StartFreshReset:
    def test_reset_discards_model_and_clears_metrics(self, mgr):
        mgr.create_network(input_size=2, output_size=2)
        _fill_buffer(mgr, 3)
        mgr.clear_metrics_with_undo()  # leaves an undo pending
        mgr.undo_clear_metrics()  # buffer back to 3, no undo
        _fill_buffer(mgr, 0)
        assert mgr.has_model() is True

        with mgr._lock:
            mgr._start_fresh_reset_locked()

        assert mgr.has_model() is False
        assert _total(mgr) == 0
        assert mgr._metrics_undo_available() is False
        assert mgr._last_emitted_history_len == 0
        assert mgr.state_machine.is_stopped() is True

    def test_start_fresh_preserves_snapshots(self, mgr, tmp_path):
        """The critical safety rail: a start-fresh reset never deletes a snapshot."""
        mgr.create_network(input_size=2, output_size=2)
        with patch.object(mgr, "_get_snapshots_dir", return_value=tmp_path):
            saved = mgr.save_snapshot(description="pre-fresh")
            snap_path = Path(saved["path"])
            assert snap_path.exists()

            with mgr._lock:
                mgr._start_fresh_reset_locked()

            # Model discarded, but the on-disk snapshot artifact is untouched.
            assert mgr.has_model() is False
            assert snap_path.exists(), "start_fresh must NOT delete snapshot artifacts"
            # The snapshot is still loadable/listable after the fresh reset.
            listed = mgr.list_snapshots()
            assert any(s.get("id") == saved["id"] for s in listed)

    def test_start_training_start_fresh_rebuilds_network_from_new_dims(self, mgr):
        """start_fresh discards the old model and rebuilds from the new dataset dims.

        A 3-feature dataset would fail against a retained 2-input network (the
        pad path rejects a grow), so a successful rebuild to input_size=3 proves
        the old model was discarded and a vanilla network created from the new
        dims. The network is created synchronously before the future is
        submitted, so patching ``_run_training`` keeps the test fast.
        """
        mgr.create_network(input_size=2, output_size=2)
        assert mgr.get_network_info()["input_size"] == 2

        x = torch.zeros(4, 3)  # 3 features — different from the 2-input network
        y = torch.zeros(4, 2)
        with patch.object(mgr, "_run_training"):
            mgr.start_training(X=x, y=y, start_fresh=True)

        assert mgr.get_network_info()["input_size"] == 3
        assert mgr.get_network_info()["hidden_units"] == 0  # vanilla, untrained


# ---------------------------------------------------------------------------
# Status surface
# ---------------------------------------------------------------------------


class TestC5Status:
    def test_status_exposes_metrics_clear_undo_available(self, mgr):
        status = mgr.get_status()
        assert status["metrics_clear_undo_available"] is False

        _fill_buffer(mgr, 2)
        mgr.clear_metrics_with_undo()
        assert mgr.get_status()["metrics_clear_undo_available"] is True

        mgr.undo_clear_metrics()
        assert mgr.get_status()["metrics_clear_undo_available"] is False


# ---------------------------------------------------------------------------
# Route surface (end-to-end via TestClient)
# ---------------------------------------------------------------------------


class TestC5Routes:
    def test_clear_and_undo_endpoints(self, client):
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        lifecycle = client.app.state.lifecycle
        _fill_buffer(lifecycle, 3)

        resp = client.post("/v1/training/metrics/clear")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data == {"status": "cleared", "cleared_count": 3, "undo_available": True}
        # History is now empty.
        hist = client.get("/v1/metrics/history").json()["data"]
        assert hist == []

        resp = client.post("/v1/training/metrics/clear/undo")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data == {"status": "restored", "restored_count": 3, "undo_available": False}
        hist = client.get("/v1/metrics/history").json()["data"]
        assert len(hist) == 3

    def test_undo_without_clear_returns_409(self, client):
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        resp = client.post("/v1/training/metrics/clear/undo")
        assert resp.status_code == 409

    def test_status_exposes_metrics_clear_undo_available(self, client):
        resp = client.get("/v1/training/status")
        assert resp.status_code == 200
        assert resp.json()["data"]["metrics_clear_undo_available"] is False

    def test_start_accepts_start_fresh_flag(self, client):
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        lifecycle = client.app.state.lifecycle
        with patch.object(lifecycle, "_run_training"):
            resp = client.post(
                "/v1/training/start",
                json={
                    "start_fresh": True,
                    "inline_data": {
                        "train_x": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
                        "train_y": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                    },
                },
            )
        assert resp.status_code == 200
        # start_fresh discarded the 2-input network and rebuilt from the 3-feature dims.
        assert lifecycle.get_network_info()["input_size"] == 3

    def test_start_without_start_fresh_defaults_false(self, client):
        """Backward-compat: a start body omitting start_fresh continues the model."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        lifecycle = client.app.state.lifecycle
        with patch.object(lifecycle, "_run_training"):
            resp = client.post(
                "/v1/training/start",
                json={"inline_data": {"train_x": [[0.0, 0.0], [1.0, 1.0]], "train_y": [[1.0, 0.0], [0.0, 1.0]]}},
            )
        assert resp.status_code == 200
        # Same 2-input network retained (not rebuilt); still present.
        assert lifecycle.get_network_info()["input_size"] == 2
