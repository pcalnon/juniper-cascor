#!/usr/bin/env python
"""
Unit tests for api/routes/network.py to improve code coverage.

Covers:
- _get_lifecycle: HTTPException when lifecycle not initialized (line 14)
- create_network: RuntimeError path (lines 25-26)
- delete_network: RuntimeError path (lines 45-46)
- get_topology: None topology path (line 57)
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.app import create_app
from api.settings import Settings

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    """Create a test client with lifecycle manager (lifespan runs)."""
    settings = Settings()
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


class TestNetworkRouteLifecycleErrors:
    """Tests for _get_lifecycle error paths."""

    def test_network_lifecycle_not_initialized_returns_503(self):
        """Network routes should return 503 when lifecycle is not initialized."""
        settings = Settings()
        app = create_app(settings)

        with TestClient(app) as c:
            lifecycle = c.app.state.lifecycle
            del c.app.state.lifecycle

            response = c.get("/v1/network")
            assert response.status_code == 503
            assert "Lifecycle manager not initialized" in response.json()["detail"]

            c.app.state.lifecycle = lifecycle


class TestCreateNetworkErrors:
    """Tests for create_network error paths."""

    def test_create_network_runtime_error(self, client):
        """create_network should return 409 on RuntimeError."""
        with patch.object(client.app.state.lifecycle, "create_network", side_effect=RuntimeError("Network already exists")):
            response = client.post("/v1/network", json={"input_size": 2, "output_size": 2})
            assert response.status_code == 409
            assert "Network already exists" in response.json()["detail"]


class TestDeleteNetworkErrors:
    """Tests for delete_network error paths."""

    def test_delete_network_runtime_error(self, client):
        """delete_network should return 409 on RuntimeError."""
        with patch.object(client.app.state.lifecycle, "delete_network", side_effect=RuntimeError("Cannot delete during training")):
            response = client.delete("/v1/network")
            assert response.status_code == 409
            assert "Cannot delete during training" in response.json()["detail"]


class TestGetTopologyErrors:
    """Tests for get_topology error paths."""

    def test_get_topology_returns_500_when_extraction_fails(self, client):
        """get_topology should return 500 when topology extraction returns None."""
        # Create a network first
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})

        with patch.object(client.app.state.lifecycle, "get_topology", return_value=None):
            response = client.get("/v1/network/topology")
            assert response.status_code == 500
            assert "Failed to extract topology" in response.json()["detail"]
