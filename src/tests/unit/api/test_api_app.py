"""Tests for API app factory."""

import asyncio
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic_core import PydanticSerializationError

from api.app import _API_VERSION, create_app
from api.lifecycle.manager import TrainingLifecycleManager
from api.settings import Settings


@pytest.mark.unit
class TestAppFactory:
    """Test create_app factory function."""

    def test_create_app_returns_fastapi_instance(self):
        """Test that create_app returns a FastAPI app."""
        from fastapi import FastAPI

        settings = Settings(auto_start=False)
        app = create_app(settings)
        assert isinstance(app, FastAPI)

    def test_create_app_with_default_settings(self):
        """Test create_app with no settings uses defaults."""
        app = create_app()
        assert hasattr(app.state, "settings")

    def test_app_title_and_version(self):
        """Test app metadata."""
        app = create_app(Settings(auto_start=False))
        assert app.title == "JuniperCascor API"
        # Assert against the BUG-CC-04 canonical runtime read, never a pinned
        # literal — a release version bump must not break this wiring test.
        assert app.version == _API_VERSION

    def test_cors_middleware_skipped_with_empty_origins(self):
        """Test that CORS middleware is not applied when origins is empty."""
        app = create_app(Settings(auto_start=False))
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" not in middleware_classes

    def test_cors_middleware_applied_with_explicit_origins(self):
        """Test that CORS middleware is applied when origins are configured."""
        settings = Settings(cors_origins=["http://localhost:3000"])
        app = create_app(settings)
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes

    def test_lifespan_wires_ws_connection_cap_settings(self):
        """WebSocket cap env/settings values must reach the live manager."""
        settings = Settings(
            auto_start=False,
            ws_max_connections=17,
            ws_max_connections_global=23,
            ws_max_connections_per_identity=3,
            ws_max_connections_per_ip=4,
        )
        app = create_app(settings)

        with TestClient(app):
            manager = app.state.ws_manager
            assert manager._max_connections == 17
            assert manager._max_connections_global == 23
            assert manager._max_connections_per_identity == 3
            assert manager._max_connections_per_ip == 4

    def test_lifespan_runs_lifecycle_shutdown_off_the_event_loop(self):
        """2026-08-25 stop-during-training fix: the lifespan must await
        ``lifecycle.shutdown()`` in a worker thread. It joins a live training thread for
        up to seconds, and the event loop has to stay free to run the terminal-state
        broadcasts that thread hands it -- so no running loop may be visible from
        inside the call."""
        seen = {}

        def record(_self):
            seen["running_loop"] = asyncio._get_running_loop()

        app = create_app(Settings(auto_start=False))
        with patch.object(TrainingLifecycleManager, "shutdown", autospec=True, side_effect=record):
            with TestClient(app):
                pass
        assert "running_loop" in seen, "lifespan did not call lifecycle.shutdown()"
        assert seen["running_loop"] is None, "lifecycle.shutdown() ran on the event-loop thread"

    def test_value_error_handler(self):
        """Test that ValueError returns 400."""
        app = create_app(Settings(auto_start=False))

        @app.get("/test-value-error")
        async def raise_value_error():
            raise ValueError("test error")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-value-error")
        assert response.status_code == 400
        body = response.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "VALIDATION_ERROR"

    def test_serialization_fault_returns_500_not_400(self):
        """APD-CASCOR-002: a serialisation fault is the server's, not the caller's.

        ``PydanticSerializationError`` subclasses ``ValueError``, so the blanket
        handler reported the app's own failure to serialise a response as a 400 --
        invisible to 5xx alerting, misattributed to the client, and stripped of its
        diagnostic by the generic "Invalid request parameters" message.
        """
        app = create_app(Settings(auto_start=False))

        @app.get("/test-serialization-error")
        async def raise_serialization_error():
            raise PydanticSerializationError("Unable to serialize unknown type: <class 'numpy.float32'>")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-serialization-error")
        assert response.status_code == 500
        body = response.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "INTERNAL_ERROR"

    def test_general_exception_handler(self):
        """Test that unhandled exceptions return 500."""
        app = create_app(Settings(auto_start=False))

        @app.get("/test-error")
        async def raise_error():
            raise RuntimeError("unexpected")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-error")
        assert response.status_code == 500
        body = response.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "INTERNAL_ERROR"


@pytest.mark.unit
class TestCorsPreflight:
    """CORS must execute OUTSIDE SecurityMiddleware.

    Regression coverage for APD-CASCOR-001b (sibling of APD-DATA-035). CORS was
    registered first, which under Starlette's prepending ``add_middleware`` made
    it the INNERMOST layer -- so ``SecurityMiddleware`` saw browser preflights
    first and answered them 401. A preflight carries no ``X-API-Key`` by
    construction, so no browser could ever reach a protected endpoint.
    """

    # A real, non-exempt route: not in ``api.middleware.EXEMPT_PATHS``.
    PROTECTED_PATH = "/v1/network/stats"

    @staticmethod
    def _app():
        """An app with BOTH a CORS origin and an API key, so auth is really active."""
        return create_app(
            Settings(
                auto_start=False,
                cors_origins=["http://localhost:3000"],
                api_keys=["preflight-test-key"],
            )
        )

    def test_cors_executes_outside_security_middleware(self):
        """Order is the contract: index 0 runs outermost, so CORS must precede Security."""
        order = [m.cls.__name__ for m in self._app().user_middleware]

        assert "CORSMiddleware" in order, order
        assert "SecurityMiddleware" in order, order
        assert order.index("CORSMiddleware") < order.index("SecurityMiddleware"), f"CORS must run outside SecurityMiddleware, got outermost-first order {order}"

    def test_preflight_to_protected_path_is_not_answered_401(self):
        """The defect itself: a genuine preflight must get CORS headers, not 401."""
        client = TestClient(self._app())

        response = client.options(
            self.PROTECTED_PATH,
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )

        assert response.status_code != 401, "preflight was rejected by auth; it carries no API key by design"
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_preflight_from_disallowed_origin_is_still_rejected(self):
        """Negative control: moving CORS outermost must not accept arbitrary origins."""
        client = TestClient(self._app())

        response = client.options(
            self.PROTECTED_PATH,
            headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
        )

        assert response.headers.get("access-control-allow-origin") is None
        assert response.status_code == 400

    def test_non_preflight_options_still_requires_auth(self):
        """The auth surface must not widen.

        This is why the fix is a reorder and not an ``OPTIONS`` bypass inside
        ``_is_exempt``: a bypass would exempt every ``OPTIONS`` request, while
        CORS short-circuits only a genuine preflight (one carrying
        ``Access-Control-Request-Method``).
        """
        client = TestClient(self._app())

        with_origin = client.options(self.PROTECTED_PATH, headers={"Origin": "http://localhost:3000"})
        bare = client.options(self.PROTECTED_PATH)

        assert with_origin.status_code == 401
        assert bare.status_code == 401

    def test_auth_failure_still_carries_cors_headers(self):
        """Outermost CORS also annotates error responses.

        Without this a browser sees an opaque CORS failure instead of the real
        401, which is why the misordering was hard to diagnose client-side.
        """
        client = TestClient(self._app())

        response = client.get(self.PROTECTED_PATH, headers={"Origin": "http://localhost:3000"})

        assert response.status_code == 401
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
