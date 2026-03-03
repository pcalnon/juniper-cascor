"""Tests for API app factory."""

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings


@pytest.mark.unit
class TestAppFactory:
    """Test create_app factory function."""

    def test_create_app_returns_fastapi_instance(self):
        """Test that create_app returns a FastAPI app."""
        from fastapi import FastAPI

        settings = Settings()
        app = create_app(settings)
        assert isinstance(app, FastAPI)

    def test_create_app_with_default_settings(self):
        """Test create_app with no settings uses defaults."""
        app = create_app()
        assert hasattr(app.state, "settings")

    def test_app_title_and_version(self):
        """Test app metadata."""
        app = create_app(Settings())
        assert app.title == "JuniperCascor API"
        assert app.version == "0.4.0"

    def test_cors_middleware_skipped_with_empty_origins(self):
        """Test that CORS middleware is not applied when origins is empty."""
        app = create_app(Settings())
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" not in middleware_classes

    def test_cors_middleware_applied_with_explicit_origins(self):
        """Test that CORS middleware is applied when origins are configured."""
        settings = Settings(cors_origins=["http://localhost:3000"])
        app = create_app(settings)
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes

    def test_value_error_handler(self):
        """Test that ValueError returns 400."""
        app = create_app(Settings())

        @app.get("/test-value-error")
        async def raise_value_error():
            raise ValueError("test error")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-value-error")
        assert response.status_code == 400
        body = response.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "VALIDATION_ERROR"

    def test_general_exception_handler(self):
        """Test that unhandled exceptions return 500."""
        app = create_app(Settings())

        @app.get("/test-error")
        async def raise_error():
            raise RuntimeError("unexpected")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-error")
        assert response.status_code == 500
        body = response.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "INTERNAL_ERROR"
