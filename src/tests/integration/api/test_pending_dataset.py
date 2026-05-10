"""End-to-end PATCH /v1/training/dataset tests for Issue #3 Phase 1.

Pins the spec from FRONTEND_ISSUES_PLAN_2026-05-09 §3.5.1 + §3.5.2 P1:

  - POST /v1/training/dataset stages a config; GET /pending echoes it.
  - DELETE /v1/training/dataset clears it; GET /pending returns null.
  - Status payload (`pending_dataset` field) reflects staging state.
  - start_training picks up + clears the staged config (verified via a
    JuniperDataClient mock — no live juniper-data instance needed).
  - Reload failure leaves the staged config in place so the user can
    fix the upstream issue and retry.
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings


@pytest.fixture
def client():
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
    exit_thread = threading.Thread(target=lambda: tc.__exit__(None, None, None), daemon=True)
    exit_thread.start()
    exit_thread.join(timeout=5)


def _create_network(client) -> None:
    resp = client.post(
        "/v1/network",
        json={"input_size": 2, "output_size": 2, "epochs_max": 5, "candidate_epochs": 2, "output_epochs": 2, "patience": 1},
    )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Staging round-trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestStagingRoundTrip:
    def test_post_then_get_pending_echoes_config(self, client):
        _create_network(client)
        cfg = {"dataset_type": "spirals", "n_samples": 200, "noise": 0.05}
        resp = client.post("/v1/training/dataset", json=cfg)
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["status"] == "staged"
        assert resp.json()["data"]["config"] == cfg

        resp = client.get("/v1/training/dataset/pending")
        assert resp.status_code == 200
        assert resp.json()["data"]["pending"] == cfg

    def test_delete_clears_pending(self, client):
        _create_network(client)
        client.post("/v1/training/dataset", json={"dataset_type": "spirals", "n_samples": 200})

        resp = client.delete("/v1/training/dataset")
        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["status"] == "cleared"
        assert body["discarded"] == {"dataset_type": "spirals", "n_samples": 200}

        resp = client.get("/v1/training/dataset/pending")
        assert resp.json()["data"]["pending"] is None

    def test_empty_post_clears_pending(self, client):
        """Empty body should be idempotent with DELETE for the cleared case."""
        _create_network(client)
        client.post("/v1/training/dataset", json={"dataset_type": "spirals", "n_samples": 200})
        resp = client.post("/v1/training/dataset", json={})
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "cleared"
        resp = client.get("/v1/training/dataset/pending")
        assert resp.json()["data"]["pending"] is None

    def test_unknown_dataset_type_rejected_422(self, client):
        _create_network(client)
        resp = client.post("/v1/training/dataset", json={"dataset_type": "lottery"})
        assert resp.status_code == 422, resp.text  # pydantic Literal rejection


# ---------------------------------------------------------------------------
# Status payload
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestStatusPayload:
    def test_status_includes_pending_dataset_field(self, client):
        _create_network(client)
        resp = client.get("/v1/training/status")
        assert resp.status_code == 200
        assert "pending_dataset" in resp.json()["data"]
        assert resp.json()["data"]["pending_dataset"] is None

    def test_status_reflects_staged_config(self, client):
        _create_network(client)
        cfg = {"dataset_type": "xor", "n_samples": 100}
        client.post("/v1/training/dataset", json=cfg)
        resp = client.get("/v1/training/status")
        assert resp.json()["data"]["pending_dataset"] == cfg


# ---------------------------------------------------------------------------
# Reload-on-start (mocked juniper-data-client)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_juniper_data_client(monkeypatch):
    """Install a fake ``juniper_data_client`` module so ``_reload_dataset``
    works without a live juniper-data instance.

    Returns the mock ``JuniperDataClient`` class so tests can configure
    ``create_dataset`` / ``download_artifact_npz`` returns and assert on
    call args.
    """
    module = types.ModuleType("juniper_data_client")
    mock_client_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.create_dataset.return_value = {"dataset_id": "fake-id-001"}
    mock_instance.download_artifact_npz.return_value = {
        "X_train": np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8]], dtype=np.float32),
        "y_train": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    }
    mock_client_cls.return_value = mock_instance
    module.JuniperDataClient = mock_client_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "juniper_data_client", module)
    return mock_client_cls, mock_instance


@pytest.mark.integration
def test_start_training_consumes_and_clears_staged_config(client, fake_juniper_data_client):
    _, mock_instance = fake_juniper_data_client
    _create_network(client)
    cfg = {"dataset_type": "spirals", "n_samples": 4}
    client.post("/v1/training/dataset", json=cfg)

    # Provide inline data so start_training has *something* before reload runs.
    client.post(
        "/v1/training/start",
        json={"inline_data": {"train_x": [[0.0, 0.0]], "train_y": [[1.0, 0.0]]}},
    )

    # Reload must have called the JuniperDataClient with the staged params.
    mock_instance.create_dataset.assert_called_once()
    call_kwargs = mock_instance.create_dataset.call_args.kwargs
    assert call_kwargs["generator"] == "spirals"
    assert call_kwargs["params"] == {"n_samples": 4}

    # And the pending config must have been cleared.
    resp = client.get("/v1/training/dataset/pending")
    assert resp.json()["data"]["pending"] is None


@pytest.mark.integration
def test_reload_failure_keeps_pending_for_retry(client, fake_juniper_data_client):
    """If juniper-data fails the fetch, the staged config must survive so the
    user can fix the upstream issue and Restart-and-retry without re-clicking
    Apply Dataset."""
    _, mock_instance = fake_juniper_data_client
    mock_instance.create_dataset.side_effect = RuntimeError("juniper-data down")

    _create_network(client)
    cfg = {"dataset_type": "spirals", "n_samples": 4}
    client.post("/v1/training/dataset", json=cfg)

    resp = client.post(
        "/v1/training/start",
        json={"inline_data": {"train_x": [[0.0, 0.0]], "train_y": [[1.0, 0.0]]}},
    )
    # start_training surfaces the RuntimeError as 4xx/5xx; the exact code
    # depends on the route's exception handler, just assert it's not 200.
    assert resp.status_code >= 400, resp.text

    pending = client.get("/v1/training/dataset/pending").json()["data"]["pending"]
    assert pending == cfg, "staged config should survive a reload failure"


@pytest.mark.integration
def test_start_training_without_pending_does_not_invoke_juniper_data(client, fake_juniper_data_client):
    mock_cls, mock_instance = fake_juniper_data_client
    _create_network(client)

    client.post(
        "/v1/training/start",
        json={"inline_data": {"train_x": [[0.0, 0.0]], "train_y": [[1.0, 0.0]]}},
    )
    mock_cls.assert_not_called()
    mock_instance.create_dataset.assert_not_called()
