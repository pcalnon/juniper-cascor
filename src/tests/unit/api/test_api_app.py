"""Tests for API app factory."""

import pytest
from fastapi.testclient import TestClient
from pydantic_core import PydanticSerializationError

from api.app import _API_VERSION, create_app
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
