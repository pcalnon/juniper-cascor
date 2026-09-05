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
        lifecycle._stop_event.set()
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
        # §6.1: a train-only artifact is REFUSED, and a refused reload deliberately
        # leaves the staged config in place for retry -- which is what this fixture
        # was tripping over. Distinct row counts so a mis-bound partition is visible.
        "X_val": np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32),
        "y_val": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "X_test": np.array([[0.3, 0.7]], dtype=np.float32),
        "y_test": np.array([[1.0, 0.0]], dtype=np.float32),
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

    # Reload must have called the JuniperDataClient with the staged params,
    # TRANSLATED to juniper-data's dialect (#396 `_translate_staged_config`):
    # registry key "spiral" (canopy's dialect says "spirals") and per-arm
    # counts — total n_samples=4 over the default 2 spirals →
    # n_points_per_spiral=2. The pre-#396 assertions here pinned the
    # untranslated forwarding that died at juniper-data with
    # "Unknown generator 'spirals'" (training-start diagnosis 2026-07-09).
    mock_instance.create_dataset.assert_called_once()
    call_kwargs = mock_instance.create_dataset.call_args.kwargs
    assert call_kwargs["generator"] == "spiral"
    assert call_kwargs["params"] == {"n_points_per_spiral": 2}

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
def test_reload_dataset_resolves_juniper_data_api_key_from_secret_file(client, fake_juniper_data_client, monkeypatch, tmp_path):
    """Regression: ``_reload_dataset`` must send the juniper-data API key.

    The dataset-reload path resolves the outbound key via ``api.secrets.get_secret``
    (which honors the ``JUNIPER_DATA_API_KEY_FILE`` Docker-secret indirection) and pass
    it to ``JuniperDataClient(api_key=...)``. A prior bug imported a nonexistent
    ``secrets_util`` module whose ``except ImportError`` branch silently substituted a
    ``None``-returning lambda, so every live dataset swap went out with no ``X-API-Key``
    and juniper-data answered 401 -> cascor 502. This asserts the resolved key (from the
    ``_FILE`` secret) actually reaches the client constructor, not ``None``.
    """
    mock_client_cls, _ = fake_juniper_data_client
    secret_file = tmp_path / "juniper_data_api_key"
    secret_file.write_text("super-secret-key-123\n")
    monkeypatch.setenv("JUNIPER_DATA_API_KEY_FILE", str(secret_file))
    monkeypatch.delenv("JUNIPER_DATA_API_KEY", raising=False)

    _create_network(client)
    client.post("/v1/training/dataset", json={"dataset_type": "spirals", "n_samples": 4})
    client.post(
        "/v1/training/start",
        json={"inline_data": {"train_x": [[0.0, 0.0]], "train_y": [[1.0, 0.0]]}},
    )

    mock_client_cls.assert_called_once()
    ctor_kwargs = mock_client_cls.call_args.kwargs
    assert ctor_kwargs.get("api_key") == "super-secret-key-123", f"expected resolved JUNIPER_DATA_API_KEY_FILE value, got {ctor_kwargs.get('api_key')!r} " "(regression: secrets_util mis-import -> None key -> juniper-data 401)"


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


# ---------------------------------------------------------------------------
# Equities (generic-params) staging — native support for juniper-data
# generators beyond the legacy spiral-shaped typed fields.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_equities_staging_accepts_and_echoes_generic_params(client):
    """``equities`` is a valid dataset_type and its inputs ride in the generic
    ``params`` dict (not covered by the spiral-shaped typed fields)."""
    _create_network(client)
    cfg = {"dataset_type": "equities", "params": {"max_symbols": 5, "start_date": "2018-01-01", "normalize_features": True}}
    resp = client.post("/v1/training/dataset", json=cfg)
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["status"] == "staged"
    assert client.get("/v1/training/dataset/pending").json()["data"]["pending"] == cfg


@pytest.mark.integration
def test_equities_reload_forwards_generic_params_flattened(client, fake_juniper_data_client):
    """Staging equities forwards the generic ``params`` to ``create_dataset``
    flattened (no nested ``params`` key, no ``dataset_type``) and merged with
    any typed fields — the native path for non-spiral juniper-data generators."""
    _, mock_instance = fake_juniper_data_client
    _create_network(client)
    cfg = {"dataset_type": "equities", "params": {"max_symbols": 5, "start_date": "2018-01-01", "normalize_features": True}}
    client.post("/v1/training/dataset", json=cfg)

    client.post("/v1/training/start", json={"inline_data": {"train_x": [[0.0, 0.0]], "train_y": [[1.0, 0.0]]}})

    mock_instance.create_dataset.assert_called_once()
    call_kwargs = mock_instance.create_dataset.call_args.kwargs
    assert call_kwargs["generator"] == "equities"
    assert call_kwargs["params"] == {"max_symbols": 5, "start_date": "2018-01-01", "normalize_features": True}


@pytest.mark.integration
def test_spirals_typed_fields_translate_to_juniper_data_schema(client, fake_juniper_data_client):
    """Regression guard: the spiral path (typed fields, no ``params`` key)
    reaches juniper-data in ITS dialect (#396 `_translate_staged_config`):
    registry key "spiral", total ``n_samples`` converted to per-arm
    ``n_points_per_spiral``, pass-through ``noise`` untouched. (Formerly
    ``test_spirals_typed_fields_still_forward_unchanged``, which pinned the
    pre-#396 untranslated forwarding — the masked seam that died at real
    juniper-data with "Unknown generator 'spirals'".)"""
    _, mock_instance = fake_juniper_data_client
    _create_network(client)
    client.post("/v1/training/dataset", json={"dataset_type": "spirals", "n_samples": 4, "noise": 0.05})

    client.post("/v1/training/start", json={"inline_data": {"train_x": [[0.0, 0.0]], "train_y": [[1.0, 0.0]]}})

    call_kwargs = mock_instance.create_dataset.call_args.kwargs
    assert call_kwargs["generator"] == "spiral"
    assert call_kwargs["params"] == {"noise": 0.05, "n_points_per_spiral": 2}
