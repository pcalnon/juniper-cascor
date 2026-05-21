#!/usr/bin/env python
"""
Unit tests for API-09 (migration complete after PR 3):
``HTTPException`` handler in ``api/app.py``.

API-09 converges cascor's dual error response shapes (FastAPI's default
``{"detail": "..."}`` for ``HTTPException`` vs. the project's
``ResponseEnvelope`` for ``ValueError`` / ``Exception``) onto a single
envelope shape.

  * PR 1 added the handler emitting a transitional dual-shape envelope
    (envelope + top-level ``"detail"`` deprecation alias).
  * PR 2 (juniper-cascor-client #59) added explicit regression coverage
    for the dual-shape parser already shipped in cascor-client on
    2026-02-21.
  * **PR 3 (this commit)** dropped the alias after the soak window
    completed; the envelope is now the only shape emitted.

These tests pin the handler's contract across the canonical HTTP
status codes raised by cascor's routes (400, 401, 403, 404, 409, 422,
500, 503) plus headers-passthrough and edge cases:

  * envelope contains ``status="error"`` + ``error.code="HTTP_NNN"`` +
    ``error.message=<exc.detail>`` + ``meta``
  * **no** top-level ``"detail"`` alias (PR 3 dropped it; wire-compat
    snapshot at ``test_api_09_http_exception_wire_compat.py`` also
    pins the alias-absent state explicitly via
    ``TestLegacyDetailAliasAbsent``)
  * ``status_code`` in the HTTP response matches ``exc.status_code``
  * ``WWW-Authenticate`` (401), ``Retry-After`` (429), and arbitrary
    custom headers are preserved via ``exc.headers`` passthrough
  * when the route omits ``detail=``, Starlette auto-fills it with
    ``HTTPStatus(status_code).phrase`` (``"Not Found"``, ``"Service
    Unavailable"``, etc.) — the envelope must carry that verbatim

The design doc lives at
``notes/API_09_ERROR_ENVELOPE_MIGRATION_DESIGN_2026-05-21.md``.
"""

import os
import sys
from http import HTTPStatus

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.app import create_app  # noqa: E402

pytestmark = pytest.mark.unit


# Status codes the design doc commits to pinning.
CANONICAL_STATUS_CODES = (400, 401, 403, 404, 409, 422, 500, 503)


@pytest.fixture(scope="module")
def app_with_test_routes() -> FastAPI:
    """Build a fresh app with a handful of test routes that raise
    ``HTTPException`` with each canonical status code, plus header
    + detail=None edge cases.

    Using ``create_app()`` (rather than a stripped-down FastAPI
    instance) ensures the handler under test is wired exactly as
    cascor wires it in production — same registration order, same
    middleware stack, same lifespan. Each test below hits one of
    these routes and asserts the handler's contract.
    """
    app = create_app()

    @app.get("/_test/raise/{status}")
    async def _raise(status: int) -> dict:
        raise HTTPException(status_code=status, detail=f"test-detail-for-{status}")

    @app.get("/_test/raise-no-detail/{status}")
    async def _raise_no_detail(status: int) -> dict:
        # exc.detail = None edge case
        raise HTTPException(status_code=status)

    @app.get("/_test/raise-401-with-www-authenticate")
    async def _raise_401() -> dict:
        # FastAPI / starlette convention: 401 with WWW-Authenticate
        raise HTTPException(
            status_code=401,
            detail="auth required",
            headers={"WWW-Authenticate": 'Bearer realm="cascor"'},
        )

    @app.get("/_test/raise-429-with-retry-after")
    async def _raise_429() -> dict:
        raise HTTPException(
            status_code=429,
            detail="too many requests",
            headers={"Retry-After": "30"},
        )

    @app.get("/_test/raise-with-custom-header")
    async def _raise_custom() -> dict:
        raise HTTPException(
            status_code=503,
            detail="custom-header-case",
            headers={"X-Cascor-Test": "preserved"},
        )

    return app


@pytest.fixture(scope="module")
def client(app_with_test_routes: FastAPI) -> TestClient:
    return TestClient(app_with_test_routes)


class TestEnvelopeShapePerStatusCode:
    """The envelope-shape contract holds for every canonical status."""

    @pytest.mark.parametrize("status", CANONICAL_STATUS_CODES)
    def test_envelope_shape_at_each_status(self, client: TestClient, status: int):
        """Envelope keys + error.code + error.message + alias absent."""
        response = client.get(f"/_test/raise/{status}")
        assert response.status_code == status
        body = response.json()

        # Envelope-shape contract
        assert body["status"] == "error"
        assert isinstance(body["error"], dict)
        assert body["error"]["code"] == f"HTTP_{status}"
        assert body["error"]["message"] == f"test-detail-for-{status}"
        assert body["error"]["detail"] is None  # ErrorDetail.detail is optional
        assert isinstance(body["meta"], dict)
        assert "timestamp" in body["meta"]
        assert "version" in body["meta"]

        # PR 3: top-level ``"detail"`` alias dropped.
        assert "detail" not in body

    @pytest.mark.parametrize("status", CANONICAL_STATUS_CODES)
    def test_envelope_when_caller_omits_detail(self, client: TestClient, status: int):
        """When the route omits ``detail=``, Starlette's
        ``HTTPException.__init__`` auto-fills it with
        ``HTTPStatus(status_code).phrase`` (e.g. ``"Service
        Unavailable"`` for 503). The envelope must carry that
        auto-filled value verbatim — not silently swap it for
        ``"HTTP NNN"`` or strip it — so operators don't lose the
        reason-phrase context.
        """
        response = client.get(f"/_test/raise-no-detail/{status}")
        assert response.status_code == status
        body = response.json()
        expected_message = HTTPStatus(status).phrase
        assert body["status"] == "error"
        assert body["error"]["code"] == f"HTTP_{status}"
        assert body["error"]["message"] == expected_message
        # PR 3: top-level ``"detail"`` alias dropped.
        assert "detail" not in body


class TestHeaderPassthrough:
    """``headers=exc.headers`` passthrough preserves downstream HTTP semantics."""

    def test_401_preserves_www_authenticate(self, client: TestClient):
        response = client.get("/_test/raise-401-with-www-authenticate")
        assert response.status_code == 401
        assert response.headers.get("WWW-Authenticate") == 'Bearer realm="cascor"'
        # And the envelope is still wrapped correctly
        body = response.json()
        assert body["error"]["code"] == "HTTP_401"
        assert body["error"]["message"] == "auth required"
        # PR 3: top-level ``"detail"`` alias dropped.
        assert "detail" not in body

    def test_429_preserves_retry_after(self, client: TestClient):
        response = client.get("/_test/raise-429-with-retry-after")
        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "30"
        body = response.json()
        assert body["error"]["code"] == "HTTP_429"

    def test_arbitrary_custom_headers_preserved(self, client: TestClient):
        response = client.get("/_test/raise-with-custom-header")
        assert response.status_code == 503
        assert response.headers.get("X-Cascor-Test") == "preserved"


class TestExistingHandlersUntouched:
    """The new ``HTTPException`` handler must not shadow the existing
    ``ValueError`` and ``Exception`` handlers — those already emit the
    envelope shape and were registered before this PR.
    """

    def test_value_error_still_returns_400_envelope(self, app_with_test_routes: FastAPI):
        @app_with_test_routes.get("/_test/raise-valueerror")
        async def _raise_ve() -> dict:
            raise ValueError("bad input")

        client = TestClient(app_with_test_routes)
        response = client.get("/_test/raise-valueerror")
        assert response.status_code == 400
        body = response.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "VALIDATION_ERROR"
        # ValueError handler does NOT add the API-09 alias — only
        # HTTPException does, since the alias only protects against
        # the FastAPI-default-shape regression for HTTPException.
        assert "detail" not in body or body.get("detail") is None

    def test_unhandled_exception_still_returns_500_envelope(self, app_with_test_routes: FastAPI):
        @app_with_test_routes.get("/_test/raise-unhandled")
        async def _raise_unhandled() -> dict:
            raise RuntimeError("boom")

        # raise_server_exceptions=False so TestClient lets the
        # exception flow through the handler instead of re-raising.
        client = TestClient(app_with_test_routes, raise_server_exceptions=False)
        response = client.get("/_test/raise-unhandled")
        assert response.status_code == 500
        body = response.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "INTERNAL_ERROR"
