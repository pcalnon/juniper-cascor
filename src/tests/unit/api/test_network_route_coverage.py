#!/usr/bin/env python
"""
Unit tests for api/routes/network.py to improve code coverage.

Covers:
- _get_lifecycle: HTTPException when lifecycle not initialized (line 14)
- create_network: RuntimeError path (lines 25-26)
- delete_network: RuntimeError path (lines 45-46)
- get_topology: None topology path (line 57)
- patch_weights (CAN-015h-1): every lifecycle-status -> HTTP-code branch,
  including the defensive unmapped-sentinel 500 path
- add_hidden_unit (CAN-015h-2): every status branch + unmapped 500
- delete_hidden_unit (CAN-015h-3): every status branch + unmapped 500

The CAN-015h status-dispatch branches are the bulk of the file's coverage
gap; each is driven by mocking the lifecycle method to return a crafted
status dict (sentinels resolved off the real lifecycle instance so the test
never drifts from the manager's constants). Part of the per-file coverage
rollout (Phase C-5); see juniper-ml
``notes/JUNIPER_ECOSYSTEM_PER_FILE_COVERAGE_ROLLOUT_SCOPING_2026-06-30.md``.
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
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


class TestNetworkRouteLifecycleErrors:
    """Tests for _get_lifecycle error paths."""

    def test_network_lifecycle_not_initialized_returns_503(self):
        """Network routes should return 503 when lifecycle is not initialized."""
        settings = Settings(auto_start=False)
        app = create_app(settings)

        with TestClient(app) as c:
            lifecycle = c.app.state.lifecycle
            del c.app.state.lifecycle

            response = c.get("/v1/network")
            assert response.status_code == 503
            assert "Lifecycle manager not initialized" in response.json()["error"]["message"]

            c.app.state.lifecycle = lifecycle


class TestCreateNetworkErrors:
    """Tests for create_network error paths."""

    def test_create_network_runtime_error(self, client):
        """create_network should return 409 on RuntimeError."""
        with patch.object(client.app.state.lifecycle, "create_network", side_effect=RuntimeError("Network already exists")):
            response = client.post("/v1/network", json={"input_size": 2, "output_size": 2})
            assert response.status_code == 409
            assert "cannot be created" in response.json()["error"]["message"]

    def test_create_network_while_resume_ready_returns_409(self, client):
        """Create while RESUME_READY must 409 and leave the snapshotted model in place."""
        created = client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        assert created.status_code == 200
        original_uuid = created.json()["data"]["uuid"]
        lifecycle = client.app.state.lifecycle
        lifecycle._resume_point_epoch = 7
        assert lifecycle.state_machine.mark_resume_ready()
        response = client.post("/v1/network", json={"input_size": 3, "output_size": 2})
        assert response.status_code == 409
        assert "cannot be created" in response.json()["error"]["message"]
        assert lifecycle.state_machine.is_resume_ready()
        assert lifecycle.get_network_info()["uuid"] == original_uuid
        assert lifecycle.get_network_info()["input_size"] == 2
        assert lifecycle._resume_point_epoch == 7


class TestDeleteNetworkErrors:
    """Tests for delete_network error paths."""

    def test_delete_network_runtime_error(self, client):
        """delete_network should return 409 on RuntimeError."""
        with patch.object(client.app.state.lifecycle, "delete_network", side_effect=RuntimeError("Cannot delete during training")):
            response = client.delete("/v1/network")
            assert response.status_code == 409
            assert "cannot be deleted" in response.json()["error"]["message"]


class TestGetTopologyErrors:
    """Tests for get_topology error paths."""

    def test_get_topology_returns_500_when_extraction_fails(self, client):
        """get_topology should return 500 when topology extraction returns None."""
        # Create a network first
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})

        with patch.object(client.app.state.lifecycle, "get_topology", return_value=None):
            response = client.get("/v1/network/topology")
            assert response.status_code == 500
            assert "Failed to extract topology" in response.json()["error"]["message"]


# A body that satisfies PatchWeightsRequest / AddHiddenUnitRequest validation.
# Contents are irrelevant because the lifecycle call is always mocked; the
# handler is reached only after FastAPI validates the request body.
_PATCH_BODY = {"target": "output", "field": "weights", "values": [[1.0]]}
_ADD_BODY = {"weights": [0.1, 0.2, 0.3]}


class TestPatchWeightsRoute:
    """Tests for patch_weights (CAN-015h-1) status -> HTTP-code dispatch."""

    def test_patch_weights_ok_returns_200(self, client):
        lc = client.app.state.lifecycle
        with patch.object(lc, "patch_weights", return_value={"status": lc._PATCH_OK}), patch.object(lc, "get_network_info", return_value={"input_size": 2}):
            response = client.patch("/v1/network/weights", json=_PATCH_BODY)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["data"]["operation"] == "patch_weights"
        assert "fsm_state" in body["data"]

    @pytest.mark.parametrize(
        "status_attr,expected_code",
        [
            ("_PATCH_NO_NETWORK", 404),
            ("_PATCH_FSM_REJECTED", 409),
            ("_PATCH_HIDDEN_UNIT_OUT_OF_RANGE", 404),
            ("_PATCH_NAN_INF", 422),
            ("_PATCH_BAD_TARGET", 400),
            ("_PATCH_SHAPE_MISMATCH", 400),
        ],
    )
    def test_patch_weights_error_status_maps_to_http(self, client, status_attr, expected_code):
        lc = client.app.state.lifecycle
        status = getattr(lc, status_attr)
        with patch.object(lc, "patch_weights", return_value={"status": status, "detail": "patch boom"}):
            response = client.patch("/v1/network/weights", json=_PATCH_BODY)
        assert response.status_code == expected_code
        assert response.json()["error"]["message"] == "patch boom"

    def test_patch_weights_unmapped_status_returns_500(self, client):
        lc = client.app.state.lifecycle
        with patch.object(lc, "patch_weights", return_value={"status": "totally_unexpected"}):
            response = client.patch("/v1/network/weights", json=_PATCH_BODY)
        assert response.status_code == 500
        assert "unexpected status" in response.json()["error"]["message"]


class TestAddHiddenUnitRoute:
    """Tests for add_hidden_unit (CAN-015h-2) status -> HTTP-code dispatch."""

    def test_add_hidden_unit_ok_returns_200(self, client):
        lc = client.app.state.lifecycle
        result = {"status": lc._ADD_OK, "unit_index": 1, "num_hidden_units": 2}
        with patch.object(lc, "add_hidden_unit_manual", return_value=result), patch.object(lc, "get_network_info", return_value={"input_size": 2}):
            response = client.post("/v1/network/hidden-units", json=_ADD_BODY)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["operation"] == "add_hidden_unit"
        assert data["unit_index"] == 1
        assert data["num_hidden_units"] == 2

    @pytest.mark.parametrize(
        "status_attr,expected_code",
        [
            ("_ADD_NO_NETWORK", 404),
            ("_ADD_FSM_REJECTED", 409),
            ("_ADD_AT_CAP", 409),
            ("_ADD_NAN_INF", 422),
            ("_ADD_BAD_ACTIVATION", 422),
            ("_ADD_BAD_SHAPE", 400),
        ],
    )
    def test_add_hidden_unit_error_status_maps_to_http(self, client, status_attr, expected_code):
        lc = client.app.state.lifecycle
        status = getattr(lc, status_attr)
        with patch.object(lc, "add_hidden_unit_manual", return_value={"status": status, "detail": "add boom"}):
            response = client.post("/v1/network/hidden-units", json=_ADD_BODY)
        assert response.status_code == expected_code
        assert response.json()["error"]["message"] == "add boom"

    def test_add_hidden_unit_unmapped_status_returns_500(self, client):
        lc = client.app.state.lifecycle
        with patch.object(lc, "add_hidden_unit_manual", return_value={"status": "totally_unexpected"}):
            response = client.post("/v1/network/hidden-units", json=_ADD_BODY)
        assert response.status_code == 500
        assert "unexpected status" in response.json()["error"]["message"]


class TestDeleteHiddenUnitRoute:
    """Tests for delete_hidden_unit (CAN-015h-3) status -> HTTP-code dispatch."""

    def test_delete_hidden_unit_ok_returns_200(self, client):
        lc = client.app.state.lifecycle
        result = {"status": lc._REMOVE_OK, "removed_index": 0, "num_hidden_units": 1}
        with patch.object(lc, "remove_hidden_unit_manual", return_value=result), patch.object(lc, "get_network_info", return_value={"input_size": 2}):
            response = client.delete("/v1/network/hidden-units/0")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["operation"] == "remove_hidden_unit"
        assert data["removed_index"] == 0
        assert data["num_hidden_units"] == 1

    @pytest.mark.parametrize(
        "status_attr,expected_code",
        [
            ("_REMOVE_NO_NETWORK", 404),
            ("_REMOVE_OUT_OF_RANGE", 404),
            ("_REMOVE_FSM_REJECTED", 409),
        ],
    )
    def test_delete_hidden_unit_error_status_maps_to_http(self, client, status_attr, expected_code):
        lc = client.app.state.lifecycle
        status = getattr(lc, status_attr)
        with patch.object(lc, "remove_hidden_unit_manual", return_value={"status": status, "detail": "remove boom"}):
            response = client.delete("/v1/network/hidden-units/3")
        assert response.status_code == expected_code
        assert response.json()["error"]["message"] == "remove boom"

    def test_delete_hidden_unit_unmapped_status_returns_500(self, client):
        lc = client.app.state.lifecycle
        with patch.object(lc, "remove_hidden_unit_manual", return_value={"status": "totally_unexpected"}):
            response = client.delete("/v1/network/hidden-units/0")
        assert response.status_code == 500
        assert "unexpected status" in response.json()["error"]["message"]
