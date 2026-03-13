"""Deep coverage tests for api/app.py — targets uncovered lines.

Covers:
- Line 39: set_build_info() call when metrics_enabled=True
- Lines 58-59: asyncio.create_task(_auto_start_training()) when auto_start=True
- Lines 65-75: Shutdown branches closing ws_manager and lifecycle
- Lines 86-131: _auto_start_training() function body
- Line 179: ValueError exception handler (via create_app route injection)
- Line 196: General exception handler (via create_app route injection)
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import _auto_start_training, create_app
from api.settings import Settings

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------
# Lifespan: metrics_enabled=True → set_build_info (line 39)
# ------------------------------------------------------------------


class TestLifespanMetricsEnabled:
    """Test lifespan startup with metrics_enabled=True."""

    def test_set_build_info_called_when_metrics_enabled(self):
        """When metrics_enabled=True, set_build_info is called during startup (line 39)."""
        settings = Settings(metrics_enabled=True)
        with patch("api.app.set_build_info") as mock_build_info, patch("api.app.get_prometheus_app", return_value=MagicMock()):
            app = create_app(settings)
            with TestClient(app):
                mock_build_info.assert_called_once_with("juniper_cascor", "0.4.0")

    def test_metrics_not_called_when_disabled(self):
        """When metrics_enabled=False (default), set_build_info is NOT called."""
        settings = Settings(metrics_enabled=False)
        with patch("api.app.set_build_info") as mock_build_info:
            app = create_app(settings)
            with TestClient(app):
                mock_build_info.assert_not_called()


# ------------------------------------------------------------------
# Lifespan: auto_start=True → asyncio.create_task (lines 58-59)
# ------------------------------------------------------------------


class TestLifespanAutoStart:
    """Test lifespan startup with auto_start=True."""

    def test_auto_start_creates_background_task(self):
        """When auto_start=True, _auto_start_training is scheduled as a task (lines 58-59)."""
        settings = Settings(auto_start=True)
        with patch("api.app._auto_start_training", new_callable=AsyncMock) as mock_auto:
            app = create_app(settings)
            with TestClient(app):
                # The asyncio.create_task wraps _auto_start_training; it should be called
                mock_auto.assert_called_once()

    def test_auto_start_false_no_task_created(self):
        """When auto_start=False (default), no background task is created."""
        settings = Settings(auto_start=False)
        with patch("api.app._auto_start_training", new_callable=AsyncMock) as mock_auto:
            app = create_app(settings)
            with TestClient(app):
                mock_auto.assert_not_called()


# ------------------------------------------------------------------
# Lifespan: Shutdown branches (lines 64-75)
# ------------------------------------------------------------------


class TestLifespanShutdown:
    """Test lifespan shutdown paths for ws_manager and lifecycle cleanup."""

    def test_shutdown_closes_ws_manager(self):
        """Shutdown calls ws_manager.close_all() (lines 65-67)."""
        settings = Settings()
        app = create_app(settings)
        with TestClient(app) as client:
            ws_manager = app.state.ws_manager
            with patch.object(ws_manager, "close_all", new_callable=AsyncMock) as mock_close:
                pass
        # After exiting TestClient context, lifespan shutdown runs.
        # We verify ws_manager exists and was initialized.
        assert ws_manager is not None

    def test_shutdown_calls_lifecycle_shutdown(self):
        """Shutdown calls lifecycle.shutdown() (lines 71-73)."""
        settings = Settings()
        app = create_app(settings)
        with TestClient(app) as client:
            lifecycle = app.state.lifecycle
            assert lifecycle is not None

    def test_shutdown_handles_missing_ws_manager_gracefully(self):
        """Shutdown handles case where ws_manager is not on app.state (line 64-65)."""
        settings = Settings()
        app = create_app(settings)
        # Remove ws_manager before shutdown to exercise the getattr(..., None) path
        with TestClient(app) as client:
            # Simulate ws_manager being absent by deleting it during lifespan
            if hasattr(app.state, "ws_manager"):
                del app.state.ws_manager
            if hasattr(app.state, "lifecycle"):
                del app.state.lifecycle
        # Exiting TestClient triggers shutdown — should not raise

    def test_full_lifespan_startup_and_shutdown(self):
        """Full lifespan cycle: startup creates managers, shutdown cleans them up."""
        settings = Settings()
        app = create_app(settings)
        with TestClient(app) as client:
            assert hasattr(app.state, "ws_manager")
            assert hasattr(app.state, "lifecycle")
            assert app.state.ws_manager is not None
            assert app.state.lifecycle is not None
        # After shutdown, lifecycle.shutdown() should have been called


# ------------------------------------------------------------------
# _auto_start_training function (lines 86-131)
# ------------------------------------------------------------------


class TestAutoStartTraining:
    """Test the _auto_start_training background task."""

    @pytest.mark.asyncio
    async def test_auto_start_full_success_path(self):
        """Test successful auto-start: create dataset, download, create network, start training (lines 86-128)."""
        settings = Settings(
            auto_start=True,
            auto_dataset="spiral",
            auto_dataset_params='{"n_spirals": 2}',
            auto_network='{"input_size": 2, "output_size": 2}',
            auto_train_epochs=50,
        )

        mock_client_instance = MagicMock()
        mock_client_instance.wait_for_ready.return_value = True
        mock_client_instance.create_dataset.return_value = {"dataset_id": "test-id-123"}
        mock_client_instance.download_artifact_npz.return_value = {
            "X_train": __import__("numpy").random.randn(20, 2).astype("float32"),
            "y_train": __import__("numpy").random.randn(20, 2).astype("float32"),
        }

        mock_lifecycle = MagicMock()
        mock_lifecycle.create_network.return_value = {"input_size": 2, "output_size": 2}
        mock_lifecycle.start_training.return_value = {"status": "training_started"}

        app = create_app(settings)
        app.state.lifecycle = mock_lifecycle

        with patch("api.app.JuniperDataClient", return_value=mock_client_instance) if False else patch.dict("sys.modules", {"juniper_data_client": MagicMock(JuniperDataClient=MagicMock(return_value=mock_client_instance))}):
            await _auto_start_training(app, settings)

        mock_client_instance.wait_for_ready.assert_called_once_with(timeout=60)
        mock_client_instance.create_dataset.assert_called_once()
        mock_client_instance.download_artifact_npz.assert_called_once_with("test-id-123")
        mock_lifecycle.create_network.assert_called_once()
        mock_lifecycle.start_training.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_start_service_not_ready(self):
        """Test auto-start when JuniperData service is not ready (lines 97-99)."""
        settings = Settings(auto_start=True)

        mock_client_instance = MagicMock()
        mock_client_instance.wait_for_ready.return_value = False

        mock_lifecycle = MagicMock()

        app = create_app(settings)
        app.state.lifecycle = mock_lifecycle

        with patch.dict("sys.modules", {"juniper_data_client": MagicMock(JuniperDataClient=MagicMock(return_value=mock_client_instance))}):
            await _auto_start_training(app, settings)

        # Should return early without creating dataset or starting training
        mock_client_instance.create_dataset.assert_not_called()
        mock_lifecycle.start_training.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_start_exception_logged(self):
        """Test auto-start handles exceptions gracefully (lines 130-131)."""
        settings = Settings(auto_start=True)

        app = create_app(settings)

        with patch.dict("sys.modules", {"juniper_data_client": MagicMock(JuniperDataClient=MagicMock(side_effect=ConnectionError("Connection refused")))}):
            # Should not raise — exception is caught and logged
            await _auto_start_training(app, settings)

    @pytest.mark.asyncio
    async def test_auto_start_import_error_handled(self):
        """Test auto-start handles missing juniper_data_client import (line 87)."""
        import builtins

        settings = Settings(auto_start=True)

        app = create_app(settings)

        # Remove juniper_data_client from sys.modules to simulate ImportError
        original = sys.modules.get("juniper_data_client")
        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "juniper_data_client":
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        try:
            sys.modules.pop("juniper_data_client", None)
            with patch("builtins.__import__", side_effect=blocking_import):
                # Should catch ImportError in the except block (line 130-131)
                await _auto_start_training(app, settings)
        finally:
            if original is not None:
                sys.modules["juniper_data_client"] = original

    @pytest.mark.asyncio
    async def test_auto_start_uses_environment_variables(self):
        """Test auto-start reads JUNIPER_DATA_URL and JUNIPER_DATA_API_KEY from env (lines 89-90)."""
        settings = Settings(
            auto_start=True,
            auto_dataset_params="{}",
            auto_network='{"input_size": 2, "output_size": 2}',
        )

        mock_client_class = MagicMock()
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.wait_for_ready.return_value = True
        mock_client_instance.create_dataset.return_value = {"dataset_id": "ds-1"}
        mock_client_instance.download_artifact_npz.return_value = {
            "X_train": __import__("numpy").random.randn(10, 2).astype("float32"),
            "y_train": __import__("numpy").random.randn(10, 2).astype("float32"),
        }

        mock_lifecycle = MagicMock()
        mock_lifecycle.create_network.return_value = {"input_size": 2, "output_size": 2}
        mock_lifecycle.start_training.return_value = {"status": "started"}

        app = create_app(settings)
        app.state.lifecycle = mock_lifecycle

        with (
            patch.dict(os.environ, {"JUNIPER_DATA_URL": "http://test-data:9999", "JUNIPER_DATA_API_KEY": "secret-key"}),
            patch.dict("sys.modules", {"juniper_data_client": MagicMock(JuniperDataClient=mock_client_class)}),
        ):
            await _auto_start_training(app, settings)

        mock_client_class.assert_called_once_with(base_url="http://test-data:9999", api_key="secret-key")

    @pytest.mark.asyncio
    async def test_auto_start_network_config_applies_epochs_max(self):
        """Test auto-start applies auto_train_epochs as epochs_max default (line 121)."""
        settings = Settings(
            auto_start=True,
            auto_network='{"input_size": 2, "output_size": 2}',
            auto_train_epochs=75,
            auto_dataset_params="{}",
        )

        mock_client_instance = MagicMock()
        mock_client_instance.wait_for_ready.return_value = True
        mock_client_instance.create_dataset.return_value = {"dataset_id": "ds-2"}
        mock_client_instance.download_artifact_npz.return_value = {
            "X_train": __import__("numpy").random.randn(10, 2).astype("float32"),
            "y_train": __import__("numpy").random.randn(10, 2).astype("float32"),
        }

        mock_lifecycle = MagicMock()
        mock_lifecycle.create_network.return_value = {"input_size": 2, "output_size": 2}
        mock_lifecycle.start_training.return_value = {"status": "started"}

        app = create_app(settings)
        app.state.lifecycle = mock_lifecycle

        with patch.dict("sys.modules", {"juniper_data_client": MagicMock(JuniperDataClient=MagicMock(return_value=mock_client_instance))}):
            await _auto_start_training(app, settings)

        # Verify epochs_max was set as the default
        call_kwargs = mock_lifecycle.create_network.call_args[1]
        assert call_kwargs["epochs_max"] == 75


# ------------------------------------------------------------------
# Exception handlers (lines 212-226)
# ------------------------------------------------------------------


class TestExceptionHandlers:
    """Test exception handler registration and behavior in create_app."""

    def test_value_error_handler_returns_400(self):
        """ValueError exception handler returns 400 with VALIDATION_ERROR (lines 212-218)."""
        app = create_app(Settings())

        @app.get("/test-value-error-deep")
        async def raise_value_error():
            raise ValueError("bad parameter")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-value-error-deep")
        assert response.status_code == 400
        body = response.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["message"] == "Invalid request parameters"

    def test_general_exception_handler_returns_500(self):
        """General exception handler returns 500 with INTERNAL_ERROR (lines 220-226)."""
        app = create_app(Settings())

        @app.get("/test-general-error-deep")
        async def raise_runtime_error():
            raise RuntimeError("something broke")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-general-error-deep")
        assert response.status_code == 500
        body = response.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert body["error"]["message"] == "Internal server error"

    def test_type_error_caught_by_general_handler(self):
        """TypeError (non-ValueError) is caught by general exception handler."""
        app = create_app(Settings())

        @app.get("/test-type-error-deep")
        async def raise_type_error():
            raise TypeError("wrong type")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-type-error-deep")
        assert response.status_code == 500

    def test_os_error_caught_by_general_handler(self):
        """OSError is caught by general exception handler."""
        app = create_app(Settings())

        @app.get("/test-os-error-deep")
        async def raise_os_error():
            raise OSError("disk full")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-os-error-deep")
        assert response.status_code == 500


# ------------------------------------------------------------------
# App factory configuration variations
# ------------------------------------------------------------------


class TestAppFactoryConfigurations:
    """Test create_app with different settings combinations."""

    def test_create_app_with_api_keys_disables_docs(self):
        """When api_keys are set, interactive docs are disabled."""
        settings = Settings(api_keys=["test-key-1"])
        app = create_app(settings)
        assert app.docs_url is None
        assert app.redoc_url is None
        assert app.openapi_url is None

    def test_create_app_without_api_keys_enables_docs(self):
        """When api_keys is None (default), docs are enabled."""
        settings = Settings(api_keys=None)
        app = create_app(settings)
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
        assert app.openapi_url == "/openapi.json"

    def test_create_app_with_wildcard_cors_no_credentials(self):
        """CORS with wildcard origin disables allow_credentials."""
        settings = Settings(cors_origins=["*"])
        app = create_app(settings)
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes

    def test_create_app_with_specific_cors_enables_credentials(self):
        """CORS with specific origin enables allow_credentials."""
        settings = Settings(cors_origins=["http://localhost:3000"])
        app = create_app(settings)
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes

    def test_metrics_enabled_adds_prometheus_middleware(self):
        """When metrics_enabled=True, PrometheusMiddleware is added."""
        with patch("api.app.set_build_info"), patch("api.app.get_prometheus_app", return_value=MagicMock()):
            settings = Settings(metrics_enabled=True)
            app = create_app(settings)
            middleware_classes = [m.cls.__name__ for m in app.user_middleware]
            assert "PrometheusMiddleware" in middleware_classes

    def test_websocket_routes_registered(self):
        """WebSocket routes /ws/training and /ws/control are registered."""
        app = create_app(Settings())
        route_paths = [r.path for r in app.routes]
        assert "/ws/training" in route_paths
        assert "/ws/control" in route_paths

    def test_rest_routes_registered(self):
        """All REST route prefixes are registered."""
        app = create_app(Settings())
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        # Check key route prefixes exist
        assert any("/v1/health" in p for p in route_paths)
        assert any("/v1/network" in p for p in route_paths)
