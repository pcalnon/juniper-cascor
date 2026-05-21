#!/usr/bin/env python
"""
Wire-format snapshot for API-09 PR 1: ``HTTPException`` envelope shape.

This is the byte-for-byte contract that downstream consumers
(``juniper-cascor-client``, browser dashboards, hand-rolled HTTP
scripts) depend on during the API-09 deprecation window. It pins both
halves of the dual-shape response:

  * the new ``ResponseEnvelope`` shape
    (``{"status":"error","error":{"code","message","detail"},"meta":{}}``)
  * the legacy top-level ``"detail"`` deprecation alias

PR 3 of the API-09 migration will:

  * delete the ``test_legacy_detail_alias_*`` cases here, and
  * add ``test_legacy_detail_alias_is_absent`` to assert the alias is
    gone.

Until then, **both** cases must pass.

The fluctuating ``meta.timestamp`` (epoch seconds) and
``meta.version`` fields are not pinned — only their presence and types
are.
"""

import os
import sys

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.app import create_app  # noqa: E402

pytestmark = pytest.mark.unit


# Top-level keys in the migrated response body — pinned exactly.
EXPECTED_TOP_LEVEL_KEYS = {"status", "error", "meta", "detail"}

# Keys inside the nested ``error`` object — pinned exactly.
EXPECTED_ERROR_KEYS = {"code", "message", "detail"}

# Keys inside ``meta`` — pinned exactly.
EXPECTED_META_KEYS = {"timestamp", "version"}


@pytest.fixture(scope="module")
def client() -> TestClient:
    app: FastAPI = create_app()

    @app.get("/_wire/raise-404")
    async def _raise_404() -> dict:
        raise HTTPException(status_code=404, detail="No network loaded")

    @app.get("/_wire/raise-503")
    async def _raise_503() -> dict:
        raise HTTPException(status_code=503, detail="Lifecycle manager not initialized")

    return TestClient(app)


class TestEnvelopeKeysExact:
    """Top-level and nested key sets are pinned byte-for-byte."""

    def test_top_level_keys_exact(self, client: TestClient):
        response = client.get("/_wire/raise-404")
        assert response.status_code == 404
        body = response.json()
        assert set(body.keys()) == EXPECTED_TOP_LEVEL_KEYS, f"Top-level keys drifted: got {set(body.keys())!r}"

    def test_error_keys_exact(self, client: TestClient):
        response = client.get("/_wire/raise-404")
        body = response.json()
        assert set(body["error"].keys()) == EXPECTED_ERROR_KEYS, f"error.* keys drifted: got {set(body['error'].keys())!r}"

    def test_meta_keys_exact(self, client: TestClient):
        response = client.get("/_wire/raise-404")
        body = response.json()
        assert set(body["meta"].keys()) == EXPECTED_META_KEYS, f"meta.* keys drifted: got {set(body['meta'].keys())!r}"


class TestEnvelopeValuesPinned:
    """Pin the exact values for the snapshot 404 / 503 cases."""

    def test_404_envelope_snapshot(self, client: TestClient):
        response = client.get("/_wire/raise-404")
        assert response.status_code == 404
        body = response.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "HTTP_404"
        assert body["error"]["message"] == "No network loaded"
        assert body["error"]["detail"] is None
        assert isinstance(body["meta"]["timestamp"], (int, float))
        assert isinstance(body["meta"]["version"], str)
        # Deprecation-alias half of the dual shape
        assert body["detail"] == "No network loaded"

    def test_503_envelope_snapshot(self, client: TestClient):
        response = client.get("/_wire/raise-503")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "HTTP_503"
        assert body["error"]["message"] == "Lifecycle manager not initialized"
        assert body["error"]["detail"] is None
        # Deprecation-alias half of the dual shape
        assert body["detail"] == "Lifecycle manager not initialized"


class TestLegacyDetailAliasPresent:
    """The top-level ``"detail"`` deprecation alias is REQUIRED during
    the migration window (PR 1 through PR 3 of the API-09 plan).

    PR 3 deletes this entire class and replaces it with
    ``TestLegacyDetailAliasAbsent`` asserting the alias is gone. If
    these tests start failing for the wrong reason during the
    deprecation window, treat it as a regression — clients on the
    pre-PR-2 cascor-client release still depend on the alias.
    """

    def test_404_carries_top_level_detail(self, client: TestClient):
        body = client.get("/_wire/raise-404").json()
        assert "detail" in body
        assert body["detail"] == body["error"]["message"]

    def test_503_carries_top_level_detail(self, client: TestClient):
        body = client.get("/_wire/raise-503").json()
        assert "detail" in body
        assert body["detail"] == body["error"]["message"]
