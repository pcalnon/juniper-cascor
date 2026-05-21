#!/usr/bin/env python
"""
Wire-format snapshot for API-09 (migration complete after PR 3):
``HTTPException`` envelope shape.

This is the byte-for-byte contract that downstream consumers
(``juniper-cascor-client``, browser dashboards, hand-rolled HTTP
scripts) depend on after the API-09 migration completes. It pins the
final single-shape response:

  * the ``ResponseEnvelope`` shape
    (``{"status":"error","error":{"code","message","detail"},"meta":{}}``)

PR 1 of the migration emitted both this envelope **and** a top-level
``"detail"`` deprecation alias of ``error.message``. PR 3 (this PR)
dropped the alias after juniper-cascor-client #59 pinned the
envelope-aware parser; the previous ``TestLegacyDetailAliasPresent``
class has been **replaced** with ``TestLegacyDetailAliasAbsent`` which
asserts the alias is now gone.

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
# PR 3: dropped "detail" — the top-level alias is gone.
EXPECTED_TOP_LEVEL_KEYS = {"status", "error", "meta"}

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

    def test_503_envelope_snapshot(self, client: TestClient):
        response = client.get("/_wire/raise-503")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "HTTP_503"
        assert body["error"]["message"] == "Lifecycle manager not initialized"
        assert body["error"]["detail"] is None


class TestLegacyDetailAliasAbsent:
    """The top-level ``"detail"`` deprecation alias is GONE after PR 3.

    PR 1 added the alias as a transitional measure so pre-migration
    consumers (notably ``juniper-cascor-client`` before commit
    b0a636a3, 2026-02-21) kept working unchanged. PR 3 dropped it
    after juniper-cascor-client #59 pinned the envelope-aware parser
    and the soak window completed.

    This class **replaces** the PR 1's ``TestLegacyDetailAliasPresent``.
    If the alias accidentally comes back (e.g., via a botched revert),
    these tests fail loudly so the migration regression is caught
    before downstream consumers re-couple to the dead alias.
    """

    def test_404_does_not_carry_top_level_detail(self, client: TestClient):
        body = client.get("/_wire/raise-404").json()
        assert "detail" not in body, "Top-level ``detail`` alias re-introduced — PR 3 of the " "API-09 migration dropped it. See " "notes/API_09_ERROR_ENVELOPE_MIGRATION_DESIGN_2026-05-21.md " "for the migration history."

    def test_503_does_not_carry_top_level_detail(self, client: TestClient):
        body = client.get("/_wire/raise-503").json()
        assert "detail" not in body, "Top-level ``detail`` alias re-introduced — PR 3 of the " "API-09 migration dropped it. See " "notes/API_09_ERROR_ENVELOPE_MIGRATION_DESIGN_2026-05-21.md " "for the migration history."
