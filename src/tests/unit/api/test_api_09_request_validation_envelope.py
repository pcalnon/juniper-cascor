#!/usr/bin/env python
"""
Unit tests for the API-09 **422 completion**: the ``RequestValidationError``
handler in ``api/app.py``.

API-09 claimed completion after PR 3, but only covered ``HTTPException``,
``ValueError`` and ``Exception``. ``RequestValidationError`` is **not** a
``ValueError`` subclass (MRO: ``ValidationException -> Exception``), and FastAPI
*actively installs* its own default handler for it, so every Pydantic
field-validation 422 kept returning the raw ``{"detail": [...]}`` while every
other error returned the envelope. Clients therefore did still parse two shapes
-- ``juniper-cascor-client._handle_response`` carried the sniff as
defect-register ``APD-CCLIENT-008``.

These tests pin the completed contract:

  * the 422 body is the envelope (``status`` / ``error`` / ``meta``), with
    ``error.code == "VALIDATION_ERROR"``
  * **the per-field list survives unflattened on ``error.detail``** -- the
    decisive arm. ``juniper-data`` recorded the same constraint on
    ``APD-DATA-013``: flattening the list "would destroy the per-field structure
    the client was just built to consume"
  * ``error.message`` carries a readable summary naming the failed field, so a
    consumer reading only the prose half still learns what went wrong
  * **no** top-level ``"detail"`` alias -- PR 3 retired that alias deliberately,
    and reinstating it would not help the only known consumer anyway (the client
    tests ``body["error"]`` first)
  * the handler is **ours**, not FastAPI's default

**On the vacuous-pass trap, and why it does *not* apply here.** ``APD-DATA-013``
had to assert ``handler.__module__`` directly, because juniper-data's handler was
byte-identical to FastAPI's default and every payload assertion passed just as
happily with it unregistered. This handler changes the payload, so the payload
assertions below are genuinely decisive -- unregister it and ``body["error"]``
raises ``KeyError``. The ownership assertion is kept anyway as a cheap, explicit
statement of intent, not because the others are vacuous.

Companion: ``test_api_09_http_exception_envelope.py`` (the ``HTTPException``
half, including the 422 raised via ``raise HTTPException(422)``, which is a
different code path and is unaffected by this handler).

Design doc: ``notes/API_09_ERROR_ENVELOPE_MIGRATION_DESIGN_2026-05-21.md``.
"""

import os
import sys

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.app import _render_validation_message, create_app  # noqa: E402

pytestmark = pytest.mark.unit


class _Payload(BaseModel):
    input_size: int
    hidden_units: int


@pytest.fixture(scope="module")
def app_with_validated_route() -> FastAPI:
    """Build a fresh app via ``create_app()`` so the handler under test is wired
    exactly as cascor wires it in production -- same registration order, same
    middleware stack, same lifespan."""
    app = create_app()

    @app.post("/_test/validate")
    async def _validate(payload: _Payload) -> dict:
        return {"ok": True}

    @app.get("/_test/raise-422")
    async def _raise_422() -> dict:
        # The OTHER 422 path: an explicit HTTPException, which goes through the
        # HTTPException handler and is untouched by this change.
        raise HTTPException(status_code=422, detail="explicit 422")

    return app


@pytest.fixture(scope="module")
def client(app_with_validated_route: FastAPI) -> TestClient:
    return TestClient(app_with_validated_route)


class TestValidationEnvelope:
    """The 422 body is now cascor's envelope."""

    def test_returns_envelope_not_bare_detail(self, client: TestClient) -> None:
        resp = client.post("/_test/validate", json={})
        assert resp.status_code == 422
        body = resp.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert "meta" in body

    def test_no_top_level_detail_alias(self, client: TestClient) -> None:
        """PR 3 retired the alias; the 422 completion does not reinstate it."""
        body = client.post("/_test/validate", json={}).json()
        assert "detail" not in body


class TestPerFieldStructurePreserved:
    """The decisive arm: the list must survive, unflattened."""

    def test_detail_is_the_unflattened_error_list(self, client: TestClient) -> None:
        body = client.post("/_test/validate", json={}).json()
        detail = body["error"]["detail"]
        assert isinstance(detail, list), "flattening destroys the per-field structure"
        assert detail, "the error list must not be empty"
        assert all(isinstance(entry, dict) for entry in detail)
        # The fields that make a validation error actionable.
        first = detail[0]
        assert "loc" in first and "msg" in first

    def test_every_failing_field_is_reported(self, client: TestClient) -> None:
        """Two missing fields must yield two entries -- a summary string alone
        would collapse them into one line and lose the second."""
        body = client.post("/_test/validate", json={}).json()
        locs = {".".join(str(p) for p in entry["loc"]) for entry in body["error"]["detail"]}
        assert locs == {"body.input_size", "body.hidden_units"}

    def test_message_names_the_failed_field(self, client: TestClient) -> None:
        body = client.post("/_test/validate", json={"hidden_units": 4}).json()
        assert "input_size" in body["error"]["message"]


class TestHandlerOwnership:
    """Ownership is ours, so a FastAPI upgrade cannot silently reshape the 422."""

    def test_handler_is_not_the_fastapi_default(self, app_with_validated_route: FastAPI) -> None:
        handler = app_with_validated_route.exception_handlers[RequestValidationError]
        assert handler.__module__.startswith("api."), handler.__module__


class TestHttpExceptionPathUnaffected:
    """The other 422 path keeps its existing contract."""

    def test_explicit_httpexception_422_still_envelope(self, client: TestClient) -> None:
        body = client.get("/_test/raise-422").json()
        assert body["error"]["code"] == "HTTP_422"
        assert body["error"]["message"] == "explicit 422"
        # HTTPException carries no per-field structure, so detail stays None.
        assert body["error"]["detail"] is None


class TestRenderValidationMessage:
    """The prose half, including the malformed-entry guard: a bad entry must not
    turn a 422 into a 500 from inside the error handler."""

    def test_joins_multiple_fields(self) -> None:
        rendered = _render_validation_message(
            [
                {"loc": ("body", "a"), "msg": "Field required"},
                {"loc": ("body", "b"), "msg": "Input should be a valid integer"},
            ]
        )
        assert rendered == "body.a: Field required; body.b: Input should be a valid integer"

    def test_empty_list_falls_back_to_prose(self) -> None:
        assert _render_validation_message([]) == "Invalid request parameters"

    def test_non_dict_entries_are_skipped(self) -> None:
        assert _render_validation_message(["nonsense", None]) == "Invalid request parameters"

    def test_missing_loc_renders_message_only(self) -> None:
        assert _render_validation_message([{"msg": "boom"}]) == "boom"
