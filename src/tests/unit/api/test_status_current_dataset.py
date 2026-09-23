#!/usr/bin/env python
"""``current_dataset`` on ``GET /v1/training/status`` — WHICH dataset is loaded.

``pending_dataset`` answered "what will change at the next start"; nothing answered
"what is there now". ``_current_dataset_config`` has been tracked since the live-swap
work but reached the API only as a swap's ``before_cfg``. juniper-canopy needs the
answer to hydrate its dataset selector after a page reload (juniper-ml
``notes/JUNIPER_2026-09-02_JUNIPER-CANOPY_SELECTION-REACHABILITY-DESIGN.md`` §4.10,
guardrail G7): without it the selector shows its layout default over whatever this
service is actually training on, and a benchmark result is filed against the wrong
dataset.

Three readings a consumer must be able to tell apart:

* ``None`` — nothing is loaded;
* ``{"dataset_type": None}`` — something is loaded and its identity is unknown
  (raw inline tensors);
* ``{"dataset_type": <name>, **params}`` — the config it was loaded from.

The reload path is exercised for real: ``juniper_data_client`` is replaced by a fake
module at the FAR seam (the HTTP client), so ``_reload_dataset`` itself runs and is
what sets the record. A test that stubbed ``_reload_dataset`` would be asserting the
stub.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from api.app import create_app
from api.lifecycle.manager import TrainingLifecycleManager
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
def fake_juniper_data(monkeypatch):
    """A fake ``juniper_data_client`` module so the REAL ``_reload_dataset`` runs offline."""
    module = types.ModuleType("juniper_data_client")
    client_cls = MagicMock()
    instance = MagicMock()
    instance.create_dataset.return_value = {"dataset_id": "fake-id-001", "meta": {}}
    instance.download_artifact_npz.return_value = {
        "X_train": np.zeros((4, 2), dtype=np.float32),
        "y_train": np.zeros((4, 2), dtype=np.float32),
        "X_val": np.zeros((2, 2), dtype=np.float32),
        "y_val": np.zeros((2, 2), dtype=np.float32),
        "X_test": np.zeros((1, 2), dtype=np.float32),
        "y_test": np.zeros((1, 2), dtype=np.float32),
    }
    client_cls.return_value = instance
    module.JuniperDataClient = client_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "juniper_data_client", module)
    return instance


def _start(mgr, **kwargs):
    """Run ``start_training``'s synchronous half only; no training is submitted."""
    with patch.object(mgr, "_run_training"):
        return mgr.start_training(**kwargs)


class TestCurrentDatasetReadings:
    def test_nothing_loaded_reads_none(self, mgr):
        assert mgr.get_status()["current_dataset"] is None

    def test_inline_tensors_without_identity_read_loaded_but_unknown(self, mgr):
        _start(mgr, X=torch.zeros(4, 2), y=torch.zeros(4, 2))
        # NOT None: the service holds data and trains on it. Folding "unknown" into
        # "nothing" would tell a client the opposite of the truth.
        assert mgr.get_status()["current_dataset"] == {"dataset_type": None}

    def test_inline_tensors_with_identity_are_named(self, mgr):
        _start(mgr, X=torch.zeros(4, 2), y=torch.zeros(4, 2), dataset_config={"dataset_type": "spiral", "n_points_per_spiral": 2})
        assert mgr.get_status()["current_dataset"] == {"dataset_type": "spiral", "n_points_per_spiral": 2}

    def test_new_inline_tensors_do_not_inherit_the_previous_identity(self, mgr):
        # The failure this guards: the record kept naming the LAST STAGED dataset
        # after inline data replaced it, so the status route named a dataset the
        # run was not training on.
        mgr._current_dataset_config = {"dataset_type": "circles", "n_samples": 100}
        _start(mgr, X=torch.ones(4, 2), y=torch.ones(4, 2))
        assert mgr.get_status()["current_dataset"] == {"dataset_type": None}

    def test_the_view_is_a_copy(self, mgr):
        _start(mgr, X=torch.zeros(4, 2), y=torch.zeros(4, 2), dataset_config={"dataset_type": "xor"})
        mgr.get_status()["current_dataset"]["dataset_type"] = "mutated"
        assert mgr.get_status()["current_dataset"] == {"dataset_type": "xor"}


class TestReloadRecordsTheStagedDataset:
    def test_a_consumed_staged_config_becomes_the_current_dataset(self, mgr, fake_juniper_data):
        mgr.stage_dataset_config(dataset_type="xor", n_samples=4)
        _start(mgr)
        status = mgr.get_status()
        assert status["pending_dataset"] is None
        assert status["current_dataset"] == {"dataset_type": "xor", "n_samples": 4}
        fake_juniper_data.create_dataset.assert_called_once()

    def test_a_pending_dataset_still_wins_over_inline_tensors(self, mgr, fake_juniper_data):
        # The staged config is applied AFTER inline tensors are bound, exactly as it
        # replaces the tensors themselves, so the record must follow it too.
        mgr.stage_dataset_config(dataset_type="xor", n_samples=4)
        _start(mgr, X=torch.ones(3, 2), y=torch.ones(3, 2), dataset_config={"dataset_type": "spiral"})
        assert mgr.get_status()["current_dataset"] == {"dataset_type": "xor", "n_samples": 4}
        assert mgr._train_x.shape[0] == 4


class TestStatusRoute:
    @pytest.fixture
    def client(self):
        settings = Settings(auto_start=False)
        app = create_app(settings)
        with TestClient(app) as c:
            yield c

    def test_route_carries_the_field_from_a_fresh_service(self, client):
        data = client.get("/v1/training/status").json()["data"]
        assert "current_dataset" in data
        assert data["current_dataset"] is None

    def test_an_in_process_spiral_start_is_named_on_the_route(self, client):
        lifecycle = client.app.state.lifecycle
        with patch.object(lifecycle, "_run_training"):
            resp = client.post("/v1/training/start", json={"dataset": {"generator": "spiral", "params": {"n_points_per_spiral": 10}}})
        assert resp.status_code == 200, resp.text
        assert client.get("/v1/training/status").json()["data"]["current_dataset"] == {"dataset_type": "spiral", "n_points_per_spiral": 10}

    def test_a_raw_inline_start_is_loaded_but_unknown_on_the_route(self, client):
        lifecycle = client.app.state.lifecycle
        with patch.object(lifecycle, "_run_training"):
            resp = client.post("/v1/training/start", json={"inline_data": {"train_x": [[0.0, 0.0], [1.0, 1.0]], "train_y": [[1.0, 0.0], [0.0, 1.0]]}})
        assert resp.status_code == 200, resp.text
        assert client.get("/v1/training/status").json()["data"]["current_dataset"] == {"dataset_type": None}
