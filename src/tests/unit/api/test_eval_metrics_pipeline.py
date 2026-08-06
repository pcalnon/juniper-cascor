"""C7 (U-4) phase 1 — pipeline / surface tests for the scalar evaluation metrics.

Exercises the REAL drain (``TrainingLifecycleManager._extract_and_record_metrics``)
against a real network and asserts the additive surfaces:

- metrics-history buffer rows gain the nullable ``f1``/``precision``/``recall``/
  ``roc_auc`` fields (attached to the terminal drained row only);
- the ``/v1/metrics`` snapshot (``get_metrics``) gains the flat fields plus the
  self-describing ``eval_metrics`` metadata block;
- the WS ``metrics`` frame (built from a buffer row) carries the fields;
- the ``JUNIPER_CASCOR_EVAL_METRICS_ENABLED`` toggle disables computation;
- and the existing loss / accuracy pipeline is unchanged (no-regression pins).
"""

from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from api.app import create_app
from api.lifecycle.manager import TrainingLifecycleManager, _env_flag
from api.lifecycle.monitor import TrainingMonitor
from api.settings import Settings
from api.websocket.messages import create_metrics_message

_SCALAR_FIELDS = ("f1", "precision", "recall", "roc_auc")


def _two_class_eval():
    """Small balanced 2-feature / 2-class one-hot eval split."""
    x = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.1, 0.7], [0.8, 0.2]], dtype=torch.float32)
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    return x, y


def _manager_with_network_and_eval():
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    x, y = _two_class_eval()
    mgr._train_x, mgr._train_y = x, y
    return mgr


def _append_step(mgr, loss=0.25, accuracy=0.75):
    """Simulate one completed training-step history row."""
    mgr.network.history["train_loss"].append(loss)
    mgr.network.history["train_accuracy"].append(accuracy)


@pytest.mark.unit
class TestMonitorAdditiveFields:
    """``on_epoch_end`` always emits the four nullable scalar keys."""

    def test_fields_present_and_null_by_default(self):
        monitor = TrainingMonitor()
        monitor.on_epoch_end(epoch=1, loss=0.1, accuracy=0.9, learning_rate=0.01)
        row = monitor.get_all_metrics()[-1]
        for field in _SCALAR_FIELDS:
            assert field in row
            assert row[field] is None

    def test_scalar_metrics_merged_when_provided(self):
        monitor = TrainingMonitor()
        scalars = {"f1": 0.5, "precision": 0.6, "recall": 0.7, "roc_auc": 0.8}
        monitor.on_epoch_end(epoch=1, loss=0.1, accuracy=0.9, learning_rate=0.01, scalar_metrics=scalars)
        row = monitor.get_all_metrics()[-1]
        assert row["f1"] == 0.5
        assert row["precision"] == 0.6
        assert row["recall"] == 0.7
        assert row["roc_auc"] == 0.8

    def test_loss_and_accuracy_unchanged_no_regression(self):
        """Existing loss/accuracy carriage is untouched by the additive fields."""
        monitor = TrainingMonitor()
        monitor.on_epoch_end(epoch=3, loss=0.123, accuracy=0.987, learning_rate=0.02, kind="training_step")
        row = monitor.get_all_metrics()[-1]
        assert row["loss"] == 0.123
        assert row["accuracy"] == 0.987
        assert row["epoch"] == 3
        assert row["kind"] == "training_step"


@pytest.mark.unit
class TestDrainScalarSurface:
    """The manager drain computes scalars over the eval split and attaches them."""

    def test_terminal_row_carries_scalars(self):
        mgr = _manager_with_network_and_eval()
        _append_step(mgr)
        mgr._extract_and_record_metrics()
        row = mgr.monitor.get_all_metrics()[-1]
        for field in _SCALAR_FIELDS:
            assert row[field] is not None, field
            assert 0.0 <= row[field] <= 1.0

    def test_only_terminal_row_of_a_multi_row_drain_carries_scalars(self):
        """A single forward pass reflects current state -> only the last row."""
        mgr = _manager_with_network_and_eval()
        _append_step(mgr, loss=0.5, accuracy=0.5)
        _append_step(mgr, loss=0.2, accuracy=0.8)
        mgr._extract_and_record_metrics()
        rows = mgr.monitor.get_all_metrics()
        assert len(rows) == 2
        # First (backfilled) row: nullable scalars stay None.
        assert all(rows[0][field] is None for field in _SCALAR_FIELDS)
        # Terminal row: scalars populated.
        assert all(rows[1][field] is not None for field in _SCALAR_FIELDS)
        # No-regression: loss/accuracy still carried on both rows.
        assert rows[0]["loss"] == 0.5 and rows[1]["loss"] == 0.2

    def test_no_forward_pass_when_no_new_rows(self):
        """A drain with no new history row emits nothing and leaves state clean."""
        mgr = _manager_with_network_and_eval()
        mgr._extract_and_record_metrics()  # history empty -> no-op
        assert mgr.monitor.get_all_metrics() == []
        assert mgr._latest_scalar_metrics is None

    def test_prefers_validation_split(self):
        """When a validation/test split is present it is used over training."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        x, y = _two_class_eval()
        mgr._train_x, mgr._train_y = x, y
        mgr._val_x, mgr._val_y = x, y
        _append_step(mgr)
        mgr._extract_and_record_metrics()
        assert mgr.get_metrics()["eval_metrics"]["split"] == "validation"

    def test_forward_failure_degrades_without_crashing_drain(self):
        """C7 contract: a failed eval forward must never abort the metrics drain.

        Loss/accuracy still record; scalar fields stay None on the terminal row.
        """
        mgr = _manager_with_network_and_eval()
        mgr.network.forward = MagicMock(side_effect=RuntimeError("boom"))
        _append_step(mgr)
        mgr._extract_and_record_metrics()  # must NOT raise
        row = mgr.monitor.get_all_metrics()[-1]
        assert row["loss"] == 0.25
        assert row["accuracy"] == 0.75
        assert all(row[field] is None for field in _SCALAR_FIELDS)
        assert mgr._latest_scalar_metrics is None

    def test_single_class_eval_surfaces_undefined_on_snapshot(self):
        """Single-class eval y degrades to null scalars + single_class reasons."""
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        mgr._train_x = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.1, 0.7], [0.8, 0.2]], dtype=torch.float32)
        mgr._train_y = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        _append_step(mgr)
        mgr._extract_and_record_metrics()
        row = mgr.monitor.get_all_metrics()[-1]
        assert all(row[field] is None for field in _SCALAR_FIELDS)
        snap = mgr.get_metrics()
        for field in _SCALAR_FIELDS:
            assert snap[field] is None
            assert snap["eval_metrics"]["undefined"][field] == "single_class"

    def test_run_start_clears_stale_latest_scalar_metrics(self):
        """Prior-run cached scalars must not linger on /v1/metrics across runs."""
        mgr = _manager_with_network_and_eval()
        mgr._latest_scalar_metrics = {
            "f1": 0.9,
            "precision": 0.9,
            "recall": 0.9,
            "roc_auc": 0.9,
            "average": "macro",
            "n_samples": 4,
            "n_classes": 2,
            "undefined": {},
        }
        cleared = {}

        def fake_fit(x, y, *, X_val=None, y_val=None, on_event=None, **kw):
            cleared["latest"] = mgr._latest_scalar_metrics

        mgr.model.fit = fake_fit
        x, y = _two_class_eval()
        mgr._run_training(x, y, x, y)

        assert cleared["latest"] is None
        snap = mgr.get_metrics()
        assert all(snap[field] is None for field in _SCALAR_FIELDS)


@pytest.mark.unit
class TestEnvFlag:
    """``_env_flag`` truthy/falsy matrix for JUNIPER_CASCOR_EVAL_METRICS_ENABLED."""

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " Off "])
    def test_falsy_values_disable(self, monkeypatch, value):
        monkeypatch.setenv("JUNIPER_CASCOR_EVAL_METRICS_ENABLED", value)
        assert _env_flag("JUNIPER_CASCOR_EVAL_METRICS_ENABLED", default=True) is False

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on", " True "])
    def test_truthy_values_enable(self, monkeypatch, value):
        monkeypatch.setenv("JUNIPER_CASCOR_EVAL_METRICS_ENABLED", value)
        assert _env_flag("JUNIPER_CASCOR_EVAL_METRICS_ENABLED", default=False) is True

    def test_blank_uses_default(self, monkeypatch):
        monkeypatch.setenv("JUNIPER_CASCOR_EVAL_METRICS_ENABLED", "   ")
        assert _env_flag("JUNIPER_CASCOR_EVAL_METRICS_ENABLED", default=True) is True
        assert _env_flag("JUNIPER_CASCOR_EVAL_METRICS_ENABLED", default=False) is False


@pytest.mark.unit
class TestSnapshotSurface:
    """``get_metrics`` (the /v1/metrics snapshot) gains flat fields + metadata."""

    def test_snapshot_has_flat_fields_and_metadata(self):
        mgr = _manager_with_network_and_eval()
        _append_step(mgr)
        mgr._extract_and_record_metrics()
        snap = mgr.get_metrics()
        for field in _SCALAR_FIELDS:
            assert field in snap
            assert snap[field] is not None
        meta = snap["eval_metrics"]
        assert meta["enabled"] is True
        assert meta["average"] == "macro"
        assert meta["split"] == "training"
        assert meta["n_samples"] == 4
        assert meta["n_classes"] == 2
        assert meta["undefined"] == {}

    def test_snapshot_scalars_match_latest_row(self):
        mgr = _manager_with_network_and_eval()
        _append_step(mgr)
        mgr._extract_and_record_metrics()
        snap = mgr.get_metrics()
        row = mgr.monitor.get_all_metrics()[-1]
        for field in _SCALAR_FIELDS:
            assert snap[field] == row[field]

    def test_snapshot_no_regression_loss_accuracy(self):
        mgr = _manager_with_network_and_eval()
        _append_step(mgr, loss=0.33, accuracy=0.66)
        mgr._extract_and_record_metrics()
        snap = mgr.get_metrics()
        assert snap["train_loss"] == 0.33
        assert snap["train_accuracy"] == 0.66
        assert snap["hidden_units"] == 0
        assert snap["epoch"] == 1


@pytest.mark.unit
class TestDisableToggle:
    """``JUNIPER_CASCOR_EVAL_METRICS_ENABLED`` gates the computation."""

    def test_env_flag_disables_at_construction(self, monkeypatch):
        monkeypatch.setenv("JUNIPER_CASCOR_EVAL_METRICS_ENABLED", "0")
        mgr = TrainingLifecycleManager()
        assert mgr._eval_metrics_enabled is False

    def test_env_flag_default_enabled(self, monkeypatch):
        monkeypatch.delenv("JUNIPER_CASCOR_EVAL_METRICS_ENABLED", raising=False)
        mgr = TrainingLifecycleManager()
        assert mgr._eval_metrics_enabled is True

    def test_disabled_leaves_scalars_null(self):
        mgr = _manager_with_network_and_eval()
        mgr._eval_metrics_enabled = False
        _append_step(mgr)
        mgr._extract_and_record_metrics()
        row = mgr.monitor.get_all_metrics()[-1]
        assert all(row[field] is None for field in _SCALAR_FIELDS)
        # No-regression: the row still carries loss/accuracy while disabled.
        assert row["loss"] == 0.25 and row["accuracy"] == 0.75
        snap = mgr.get_metrics()
        assert snap["eval_metrics"]["enabled"] is False
        assert all(snap[field] is None for field in _SCALAR_FIELDS)


@pytest.mark.unit
class TestWebSocketFrameAdditive:
    """The WS ``metrics`` frame (built from a buffer row) carries the fields."""

    def test_metrics_frame_preserves_scalar_fields(self):
        monitor = TrainingMonitor()
        scalars = {"f1": 0.5, "precision": None, "recall": 0.7, "roc_auc": None}
        monitor.on_epoch_end(epoch=1, loss=0.1, accuracy=0.9, learning_rate=0.01, scalar_metrics=scalars)
        row = monitor.get_all_metrics()[-1]
        frame = create_metrics_message(row)
        assert frame["type"] == "metrics"
        data = frame["data"]
        # Present including the None-valued ones (stable schema; exclude_none
        # applies only to top-level envelope fields, not inside ``data``).
        for field in _SCALAR_FIELDS:
            assert field in data
        assert data["f1"] == 0.5
        assert data["recall"] == 0.7
        assert data["precision"] is None
        assert data["roc_auc"] is None


@pytest.mark.unit
class TestRestSurface:
    """The REST metrics surfaces expose the additive fields."""

    @pytest.fixture
    def client(self):
        app = create_app(Settings(auto_start=False))
        with TestClient(app) as test_client:
            yield test_client

    def test_get_metrics_snapshot_exposes_fields(self, client):
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        response = client.get("/v1/metrics")
        assert response.status_code == 200
        data = response.json()["data"]
        for field in _SCALAR_FIELDS:
            assert field in data  # nullable before the first drain
        assert "eval_metrics" in data
        assert data["eval_metrics"]["average"] == "macro"

    def test_history_rows_include_fields_after_drain(self, client):
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        lifecycle = client.app.state.lifecycle
        x, y = _two_class_eval()
        lifecycle._train_x, lifecycle._train_y = x, y
        _append_step(lifecycle)
        lifecycle._extract_and_record_metrics()
        response = client.get("/v1/metrics/history")
        assert response.status_code == 200
        rows = response.json()["data"]
        assert len(rows) >= 1
        terminal = rows[-1]
        for field in _SCALAR_FIELDS:
            assert field in terminal
            assert terminal[field] is not None
