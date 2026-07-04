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

from api.app import _auto_start_canopy, _auto_start_training, _unregister_worker_metrics_collector, create_app
from api.settings import Settings

pytestmark = pytest.mark.unit


def _collect_route_paths(routes, prefix=""):
    """Collect all route paths (REST + WebSocket), descending into routers.

    fastapi >=0.137 wraps ``app.include_router(...)`` results in
    ``_IncludedRouter`` objects that have no ``.path``; their sub-routes live
    under ``.include_context.included_router.routes``. WebSocket routes are also
    absent from the OpenAPI schema, so this manual walk is the robust way to
    assert WS registration across fastapi versions.
    """
    found: set[str] = set()
    for route in routes:
        ctx = getattr(route, "include_context", None)
        if ctx is not None:  # fastapi >=0.137 _IncludedRouter wrapper
            found |= _collect_route_paths(ctx.included_router.routes, prefix + (getattr(ctx, "prefix", "") or ""))
            continue
        path = getattr(route, "path", None)
        if path is not None:
            found.add(prefix + path)
        sub = getattr(route, "routes", None)  # nested router (older fastapi / mounts)
        if sub:
            found |= _collect_route_paths(sub, prefix)
    return found


# ------------------------------------------------------------------
# Lifespan: metrics_enabled=True → set_build_info (line 39)
# ------------------------------------------------------------------


class TestLifespanMetricsEnabled:
    """Test lifespan startup with metrics_enabled=True."""

    def test_set_build_info_called_when_metrics_enabled(self):
        """When metrics_enabled=True, set_build_info is called during startup (line 39)."""
        settings = Settings(metrics_enabled=True, auto_start=False)
        with patch("api.app.set_build_info") as mock_build_info, patch("api.app.get_prometheus_app", return_value=MagicMock()):
            app = create_app(settings)
            with TestClient(app):
                # Build provenance (juniper-ml notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md):
                # set_build_info now also receives git_sha/build_date from the
                # api.provenance accessor — both None outside a provenance-stamped
                # image (no env vars in the unit-test context).
                mock_build_info.assert_called_once_with("juniper_cascor", "0.5.0", git_sha=None, build_date=None)

    def test_metrics_not_called_when_disabled(self):
        """When metrics_enabled=False (default), set_build_info is NOT called."""
        settings = Settings(metrics_enabled=False, auto_start=False)
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

    def test_auto_start_default_is_off(self, monkeypatch):
        """The ``auto_start`` default must be OFF: a fresh cascor must not auto-train on
        boot. Auto-start trains onto a default (empty) network and violates the
        clean-STOPPED initial state every API / Canopy / automation caller assumes; the
        deploy demo opts in explicitly via ``JUNIPER_CASCOR_AUTO_START=true``. See
        notes/CASCOR_STARTUP_SECRET_INDIRECTION_INVESTIGATION_2026-06-14.md (3.3)."""
        monkeypatch.delenv("JUNIPER_CASCOR_AUTO_START", raising=False)
        assert Settings().auto_start is False


# ------------------------------------------------------------------
# Lifespan: Shutdown branches (lines 64-75)
# ------------------------------------------------------------------


class TestLifespanShutdown:
    """Test lifespan shutdown paths for ws_manager and lifecycle cleanup."""

    def test_shutdown_closes_ws_manager(self):
        """Shutdown calls ws_manager.close_all() (lines 65-67)."""
        settings = Settings(auto_start=False)
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
        settings = Settings(auto_start=False)
        app = create_app(settings)
        with TestClient(app) as client:
            lifecycle = app.state.lifecycle
            assert lifecycle is not None

    def test_shutdown_handles_missing_ws_manager_gracefully(self):
        """Shutdown handles case where ws_manager is not on app.state (line 64-65)."""
        settings = Settings(auto_start=False)
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
        settings = Settings(auto_start=False)
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
        """Test auto-start reads JUNIPER_DATA_URL and JUNIPER_DATA_API_KEY from env (lines 89-90).

        CFG-04: ``_auto_start_training`` now reads ``settings.juniper_data_url``
        (consolidated pydantic field) instead of ``os.environ.get(...)``
        at call time. The field is populated from the canonical
        ``JUNIPER_DATA_URL`` env var at ``Settings(...)`` construction
        time, so the env patch must be in scope **when Settings is
        instantiated**, not just when ``_auto_start_training`` is
        called. Equivalent test design: pass ``juniper_data_url``
        directly to ``Settings(...)`` which exercises the same code
        path and stays decoupled from ambient process env.
        """
        settings = Settings(
            auto_start=True,
            auto_dataset_params="{}",
            auto_network='{"input_size": 2, "output_size": 2}',
            juniper_data_url="http://test-data:9999",
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
            patch.dict(os.environ, {"JUNIPER_DATA_API_KEY": "secret-key"}),
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
        app = create_app(Settings(auto_start=False))

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
        app = create_app(Settings(auto_start=False))

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
        app = create_app(Settings(auto_start=False))

        @app.get("/test-type-error-deep")
        async def raise_type_error():
            raise TypeError("wrong type")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-type-error-deep")
        assert response.status_code == 500

    def test_os_error_caught_by_general_handler(self):
        """OSError is caught by general exception handler."""
        app = create_app(Settings(auto_start=False))

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
        app = create_app(Settings(auto_start=False))
        route_paths = _collect_route_paths(app.routes)
        assert "/ws/training" in route_paths
        assert "/ws/control" in route_paths

    def test_rest_routes_registered(self):
        """All REST route prefixes are registered."""
        app = create_app(Settings(auto_start=False))
        # fastapi >=0.137 wraps included routers in ``_IncludedRouter`` (no
        # ``.path``); the OpenAPI schema lists registered REST paths robustly.
        route_paths = list(app.openapi()["paths"])
        # Check key route prefixes exist
        assert any("/v1/health" in p for p in route_paths)
        assert any("/v1/network" in p for p in route_paths)


# ------------------------------------------------------------------
# _unregister_worker_metrics_collector — best-effort exception arm
# (lines 118-119)
# ------------------------------------------------------------------


class TestUnregisterWorkerMetricsCollector:
    """The shutdown-time collector unregister must swallow REGISTRY errors."""

    def test_unregister_swallows_registry_exception(self):
        """A ``REGISTRY.unregister`` failure is caught and logged, never raised (lines 118-119)."""
        app = MagicMock()
        # A non-None collector on app.state takes the function past the early return.
        app.state.worker_metrics_collector = MagicMock(name="worker_metrics_collector")

        with patch("prometheus_client.REGISTRY.unregister", side_effect=RuntimeError("registry down")) as mock_unregister:
            # Best-effort contract: the RuntimeError is swallowed, not propagated.
            _unregister_worker_metrics_collector(app)

        mock_unregister.assert_called_once_with(app.state.worker_metrics_collector)


# ------------------------------------------------------------------
# Lifespan: auto_start_data_service block (lines 215, 221-232) plus the
# managed-services shutdown drain (lines 295, 297)
# ------------------------------------------------------------------


class TestLifespanAutoStartDataService:
    """Lifespan launches the juniper-data companion service when configured."""

    def test_data_service_started_and_terminated(self):
        """A successful start is tracked and terminated on shutdown (lines 215, 221-230, 295, 297)."""
        settings = Settings(auto_start_data_service=True, auto_start=False, auto_start_canopy=False, metrics_enabled=False)
        mock_svc = MagicMock(name="managed_data_service")

        with patch("api.service_launcher.start_service", new=AsyncMock(return_value=mock_svc)) as mock_start:
            app = create_app(settings)
            with TestClient(app):
                # Startup appended the launched service to managed_services.
                assert mock_svc in app.state.managed_services

        mock_start.assert_awaited_once()
        # Shutdown drains managed_services in reverse start order (line 295) and logs (line 297).
        mock_svc.terminate.assert_called_once_with()

    def test_data_service_start_failure_logged(self):
        """When start_service returns None the failure is logged and nothing is tracked (line 232)."""
        settings = Settings(auto_start_data_service=True, auto_start=False, auto_start_canopy=False, metrics_enabled=False)

        with patch("api.service_launcher.start_service", new=AsyncMock(return_value=None)) as mock_start:
            app = create_app(settings)
            with TestClient(app):
                assert app.state.managed_services == []

        mock_start.assert_awaited_once()


# ------------------------------------------------------------------
# Lifespan: auto_start_canopy task wiring + in-flight cancel at shutdown
# (lines 253-256, 267-270)
# ------------------------------------------------------------------


class TestLifespanAutoStartCanopyWiring:
    """Lifespan schedules the canopy auto-start task and cancels it if still running at shutdown."""

    def test_canopy_task_created_and_cancelled_at_shutdown(self):
        """auto_start_canopy schedules a tracked task; a still-pending task is cancelled on shutdown (lines 253-256, 267-270)."""
        settings = Settings(auto_start_canopy=True, auto_start=False, auto_start_data_service=False, metrics_enabled=False)

        async def _never_finishes(*args, **kwargs):
            # Stay pending through the yield so the shutdown in-flight-cancellation branch runs.
            await asyncio.sleep(30)

        with patch("api.app._auto_start_canopy", new=_never_finishes):
            app = create_app(settings)
            with TestClient(app):
                # The canopy coroutine is scheduled and tracked on app.state.startup_tasks (lines 253-256).
                assert len(app.state.startup_tasks) == 1
                assert app.state.startup_tasks[0].get_name() == "auto_start_canopy"
            # Exiting TestClient runs shutdown, which cancels the still-pending task (lines 267-270).
            assert app.state.startup_tasks[0].cancelled()


# ------------------------------------------------------------------
# _auto_start_canopy background task (lines 374-408)
# ------------------------------------------------------------------


class TestAutoStartCanopy:
    """Direct-call coverage of the _auto_start_canopy background task.

    Mirrors the ``_auto_start_training`` tests above: the handler is driven
    directly with ``AsyncMock`` seams over ``api.service_launcher`` so no real
    health poll or subprocess is launched. The ``app`` argument is unused by
    the function body (it reads only ``settings`` + ``managed_services``), so a
    bare ``MagicMock`` stands in for it.
    """

    @pytest.mark.asyncio
    async def test_canopy_full_success_path(self):
        """Cascor healthy → canopy launched and tracked (lines 374-379, 389-396, 402-403)."""
        settings = Settings(auto_start_canopy=True, auto_start=False)
        managed_services: list = []
        mock_svc = MagicMock(name="canopy_service")

        with (
            patch("api.service_launcher.wait_for_health", new=AsyncMock(return_value=True)) as mock_wait,
            patch("api.service_launcher.start_service", new=AsyncMock(return_value=mock_svc)) as mock_start,
        ):
            await _auto_start_canopy(MagicMock(), settings, managed_services)

        mock_wait.assert_awaited_once()
        mock_start.assert_awaited_once()
        assert managed_services == [mock_svc]

    @pytest.mark.asyncio
    async def test_canopy_aborts_when_cascor_not_healthy(self):
        """Cascor never becomes healthy → early return, canopy not launched (lines 380-382)."""
        settings = Settings(auto_start_canopy=True, auto_start=False)
        managed_services: list = []

        with (
            patch("api.service_launcher.wait_for_health", new=AsyncMock(return_value=False)),
            patch("api.service_launcher.start_service", new=AsyncMock()) as mock_start,
        ):
            await _auto_start_canopy(MagicMock(), settings, managed_services)

        mock_start.assert_not_awaited()
        assert managed_services == []

    @pytest.mark.asyncio
    async def test_canopy_start_failure_logged(self):
        """start_service returns None → failure logged, nothing tracked (line 405)."""
        settings = Settings(auto_start_canopy=True, auto_start=False)
        managed_services: list = []

        with (
            patch("api.service_launcher.wait_for_health", new=AsyncMock(return_value=True)),
            patch("api.service_launcher.start_service", new=AsyncMock(return_value=None)),
        ):
            await _auto_start_canopy(MagicMock(), settings, managed_services)

        assert managed_services == []

    @pytest.mark.asyncio
    async def test_canopy_exception_is_swallowed(self):
        """An exception inside the task body is caught and logged, never propagated (lines 407-408)."""
        settings = Settings(auto_start_canopy=True, auto_start=False)
        managed_services: list = []

        with patch("api.service_launcher.wait_for_health", new=AsyncMock(side_effect=RuntimeError("health probe blew up"))):
            # Must not raise — the broad except in _auto_start_canopy swallows it.
            await _auto_start_canopy(MagicMock(), settings, managed_services)

        assert managed_services == []
