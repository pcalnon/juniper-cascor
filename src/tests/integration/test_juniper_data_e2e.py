#!/usr/bin/env python
"""
Project:       Juniper
Prototype:     Cascade Correlation Neural Network
File Name:     test_juniper_data_e2e.py
Author:        Paul Calnon
Version:       0.1.0

Date:          2026-02-07
Last Modified: 2026-02-07

License:       MIT License
Copyright:     Copyright (c) 2024-2026 Paul Calnon

Description:
    End-to-end integration tests exercising the full data flow:
    JuniperDataClient → JuniperData API → NPZ artifact → SpiralDataProvider → PyTorch tensors.
    Uses FastAPI TestClient to run the JuniperData API in-process.
"""
import tempfile

import numpy as np
import pytest
import requests
import torch

from juniper_data_client.client import JuniperDataClient
from spiral_problem.data_provider import SpiralDataProvider

_SPIRAL_PARAMS = {
    "n_spirals": 2,
    "n_points_per_spiral": 50,
    "n_rotations": 1.5,
    "noise": 0.1,
    "clockwise": True,
    "train_ratio": 0.7,
    "test_ratio": 0.3,
    "seed": 42,
}


class _RequestsSessionAdapter:
    """Wraps an httpx-based FastAPI TestClient so it behaves like requests.Session.

    The JuniperDataClient uses ``self.session.request()`` and inspects
    ``response.ok``, ``response.content``, ``response.json()`` — all attributes
    of ``requests.Response``.  Starlette's ``TestClient`` inherits from
    ``httpx.Client`` whose responses lack ``.ok``.  This adapter converts each
    httpx response into a ``requests.Response`` before returning it.
    """

    def __init__(self, test_client):
        self._tc = test_client
        self.headers = {}

    def _convert(self, httpx_resp) -> requests.Response:
        r = requests.Response()
        r.status_code = httpx_resp.status_code
        r._content = httpx_resp.content
        r.headers.update(httpx_resp.headers)
        r.encoding = httpx_resp.encoding or "utf-8"
        return r

    def request(self, method, url, **kwargs):
        return self._convert(self._tc.request(method, url, **kwargs))

    def get(self, url, **kwargs):
        return self._convert(self._tc.get(url, **kwargs))

    def post(self, url, **kwargs):
        return self._convert(self._tc.post(url, **kwargs))


@pytest.fixture(scope="module")
def juniper_data_test_client():
    """Create an in-process FastAPI TestClient for the JuniperData API."""
    try:
        from juniper_data.api.app import create_app
        from juniper_data.api.settings import Settings
    except ImportError:
        pytest.skip("JuniperData package not installed")

    from fastapi.testclient import TestClient

    settings = Settings(storage_path=tempfile.mkdtemp(), api_keys=None, rate_limit_enabled=False)
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


def _make_patched_jd_client(test_client):
    """Create a JuniperDataClient with its session replaced by the FastAPI TestClient."""
    client = JuniperDataClient(base_url="http://testserver")
    client.session = _RequestsSessionAdapter(test_client)
    return client


@pytest.fixture(scope="module")
def patched_jd_client(juniper_data_test_client):
    """Module-scoped JuniperDataClient patched to use the in-process TestClient."""
    return _make_patched_jd_client(juniper_data_test_client)


@pytest.fixture
def patched_spiral_data_provider(juniper_data_test_client):
    """SpiralDataProvider patched to route requests through the in-process TestClient."""
    provider = SpiralDataProvider(juniper_data_url="http://testserver")
    provider._get_client()
    provider._client.session = _RequestsSessionAdapter(juniper_data_test_client)
    return provider


@pytest.mark.integration
@pytest.mark.requires_juniper_data
class TestJuniperDataE2EHealth:
    """Verify connectivity to the JuniperData API via health endpoints."""

    def test_health_endpoint_reachable(self, juniper_data_test_client):
        """GET /v1/health returns 200 with status ok."""
        resp = juniper_data_test_client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_check_method(self, patched_jd_client):
        """JuniperDataClient.health_check() returns True against the test server."""
        assert patched_jd_client.health_check() is True


@pytest.mark.integration
@pytest.mark.requires_juniper_data
class TestJuniperDataE2EDatasetCreation:
    """Verify dataset creation, metadata, and NPZ artifact download via the API."""

    def test_create_spiral_dataset_returns_metadata(self, patched_jd_client):
        """create_dataset returns response with dataset_id and meta keys."""
        response = patched_jd_client.create_dataset(generator="spiral", params=_SPIRAL_PARAMS)
        assert "dataset_id" in response
        assert "meta" in response

    def test_download_npz_artifact(self, patched_jd_client):
        """download_artifact_npz returns dict with all 6 required array keys as ndarrays."""
        response = patched_jd_client.create_dataset(generator="spiral", params=_SPIRAL_PARAMS)
        dataset_id = response["dataset_id"]

        arrays = patched_jd_client.download_artifact_npz(dataset_id)

        required_keys = {"X_train", "y_train", "X_test", "y_test", "X_full", "y_full"}
        assert required_keys <= set(arrays.keys())

        for key in required_keys:
            assert isinstance(arrays[key], np.ndarray), f"{key} should be ndarray"

        for key in ["X_train", "X_test", "X_full"]:
            assert arrays[key].shape[1] == 2, f"{key} should have 2 feature columns"

    def test_npz_arrays_have_correct_shapes(self, patched_jd_client):
        """Array shapes are consistent: train + test == full, total points == n_spirals * n_points."""
        response = patched_jd_client.create_dataset(generator="spiral", params=_SPIRAL_PARAMS)
        arrays = patched_jd_client.download_artifact_npz(response["dataset_id"])

        assert arrays["X_train"].shape[0] + arrays["X_test"].shape[0] == arrays["X_full"].shape[0]
        assert arrays["X_full"].shape[0] == 100  # 2 spirals * 50 points
        assert arrays["y_train"].shape[1] == 2  # number of classes == number of spirals

    def test_idempotent_dataset_creation(self, patched_jd_client):
        """Creating a dataset with identical params twice returns the same dataset_id."""
        r1 = patched_jd_client.create_dataset(generator="spiral", params=_SPIRAL_PARAMS)
        r2 = patched_jd_client.create_dataset(generator="spiral", params=_SPIRAL_PARAMS)
        assert r1["dataset_id"] == r2["dataset_id"]


@pytest.mark.integration
@pytest.mark.requires_juniper_data
class TestJuniperDataE2EFullFlow:
    """End-to-end: SpiralDataProvider → JuniperData API → NPZ → PyTorch tensors → CasCor network."""

    def test_spiral_data_provider_full_roundtrip(self, patched_spiral_data_provider):
        """Provider returns 3 pairs of float32 tensors with correct shapes."""
        result = patched_spiral_data_provider.get_spiral_dataset(
            n_spirals=2,
            n_points=50,
            n_rotations=1.5,
            noise_level=0.1,
            clockwise=True,
            train_ratio=0.7,
            test_ratio=0.3,
            seed=42,
        )

        assert isinstance(result, tuple) and len(result) == 3

        (x_train, y_train), (x_test, y_test), (x_full, y_full) = result

        for t in (x_train, y_train, x_test, y_test, x_full, y_full):
            assert isinstance(t, torch.Tensor)
            assert t.dtype == torch.float32

        assert x_train.shape[1] == 2
        assert y_train.shape[1] == 2
        assert x_full.shape[0] == 100

    def test_provider_with_3_spirals(self, patched_spiral_data_provider):
        """3-spiral dataset produces 3-class labels and correct total points."""
        (x_train, y_train), (x_test, y_test), (x_full, y_full) = patched_spiral_data_provider.get_spiral_dataset(
            n_spirals=3,
            n_points=30,
            n_rotations=1.5,
            noise_level=0.1,
            clockwise=True,
            train_ratio=0.7,
            test_ratio=0.3,
            seed=42,
        )

        assert y_train.shape[1] == 3
        assert x_full.shape[0] == 90

    def test_provider_with_legacy_algorithm(self, patched_spiral_data_provider):
        """Legacy algorithm produces the same valid tensor structure."""
        result = patched_spiral_data_provider.get_spiral_dataset(
            n_spirals=2,
            n_points=50,
            n_rotations=1.5,
            noise_level=0.1,
            clockwise=True,
            train_ratio=0.7,
            test_ratio=0.3,
            seed=42,
            algorithm="legacy_cascor",
        )

        assert isinstance(result, tuple) and len(result) == 3
        (x_train, y_train), _, _ = result
        assert isinstance(x_train, torch.Tensor)
        assert x_train.shape[1] == 2

    def test_tensors_usable_for_cascor_network(self, patched_spiral_data_provider):
        """Tensors produced by the provider can be fed directly into a CascadeCorrelationNetwork."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

        n_spirals = 2
        (x_train, y_train), _, _ = patched_spiral_data_provider.get_spiral_dataset(
            n_spirals=n_spirals,
            n_points=50,
            n_rotations=1.5,
            noise_level=0.1,
            clockwise=True,
            train_ratio=0.7,
            test_ratio=0.3,
            seed=42,
        )

        config = CascadeCorrelationConfig.create_simple_config(
            input_size=2,
            output_size=n_spirals,
            learning_rate=0.1,
            max_hidden_units=3,
        )
        network = CascadeCorrelationNetwork(config=config)

        output = network.forward(x_train)
        assert output.shape == (x_train.shape[0], n_spirals)
