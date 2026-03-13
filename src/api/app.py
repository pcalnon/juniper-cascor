"""FastAPI application factory and configuration."""

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.lifecycle.manager import TrainingLifecycleManager
from api.middleware import RequestBodyLimitMiddleware, SecurityHeadersMiddleware, SecurityMiddleware
from api.models.common import error_response
from api.observability import PrometheusMiddleware, RequestIdMiddleware, configure_logging, configure_sentry, get_prometheus_app, set_build_info
from api.routes import dataset, decision_boundary, health, metrics, network, training
from api.security import APIKeyAuth, RateLimiter
from api.settings import Settings, get_settings
from api.websocket.control_stream import control_stream_handler
from api.websocket.manager import WebSocketManager
from api.websocket.training_stream import training_stream_handler

_API_VERSION: str = "0.4.0"

logger = logging.getLogger("juniper_cascor.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    settings: Settings = app.state.settings

    configure_logging(settings.log_level, settings.log_format, "juniper-cascor")
    configure_sentry(settings.sentry_dsn, "juniper-cascor", _API_VERSION)
    if settings.metrics_enabled:
        set_build_info("juniper_cascor", _API_VERSION)

    logger.info(f"JuniperCascor API v{_API_VERSION} starting")
    logger.info(f"Listening on {settings.host}:{settings.port}")

    # Create WebSocket manager
    ws_manager = WebSocketManager(max_connections=settings.ws_max_connections)
    ws_manager.set_event_loop(asyncio.get_running_loop())
    app.state.ws_manager = ws_manager
    logger.info("WebSocket manager initialized")

    # Create lifecycle manager for training coordination
    lifecycle = TrainingLifecycleManager()
    lifecycle.set_ws_manager(ws_manager)
    app.state.lifecycle = lifecycle
    logger.info("Lifecycle manager initialized")

    # Auto-start companion services (non-containerized mode)
    managed_services: list = []
    app.state.managed_services = managed_services

    if settings.auto_start_data_service:
        from api.service_launcher import start_service

        data_url = os.environ.get("JUNIPER_DATA_URL", "http://localhost:8100")
        logger.info("Auto-start juniper-data service is ENABLED")
        svc = await start_service(
            name="juniper-data",
            command=settings.auto_start_data_service_command,
            health_url=f"{data_url.rstrip('/')}/v1/health",
            env_overrides={"JUNIPER_DATA_HOST": "0.0.0.0"},
        )
        if svc:
            managed_services.append(svc)
        else:
            logger.error("Failed to auto-start juniper-data service")

    # Auto-start training if configured (runs as background task)
    if settings.auto_start:
        logger.warning("Auto-start training is ENABLED — this should only be used in demo/dev environments")
        asyncio.create_task(_auto_start_training(app, settings))

    # Auto-start canopy as background task (waits for cascor to be accepting connections)
    if settings.auto_start_canopy:
        logger.info("Auto-start juniper-canopy is ENABLED (normal mode)")
        asyncio.create_task(_auto_start_canopy(app, settings, managed_services))

    yield

    # Shutdown: close all WebSocket connections
    ws_manager = getattr(app.state, "ws_manager", None)
    if ws_manager is not None:
        await ws_manager.close_all()
        logger.info("WebSocket connections closed")

    # Shutdown: clean up lifecycle manager if present
    lifecycle = getattr(app.state, "lifecycle", None)
    if lifecycle is not None:
        lifecycle.shutdown()
        logger.info("Lifecycle manager shut down")

    # Shutdown: terminate managed companion services (reverse start order)
    managed_services = getattr(app.state, "managed_services", [])
    for svc in reversed(managed_services):
        svc.terminate()
    if managed_services:
        logger.info("Managed companion services terminated")

    logger.info("JuniperCascor API shutting down")


async def _auto_start_training(app: FastAPI, settings: Settings) -> None:
    """Auto-start training sequence: create dataset, network, and begin training.

    Runs as a background asyncio task so the server becomes healthy before
    the auto-start sequence completes. Uses JuniperDataClient to create and
    fetch training data, then uses the lifecycle manager to create a network
    and start training.
    """
    try:
        from juniper_data_client import JuniperDataClient

        data_url = os.environ.get("JUNIPER_DATA_URL", "http://localhost:8100")
        api_key = os.environ.get("JUNIPER_DATA_API_KEY")

        client = JuniperDataClient(base_url=data_url, api_key=api_key)

        # Wait for JuniperData service
        logger.info(f"Auto-start: waiting for JuniperData at {data_url}")
        ready = await asyncio.to_thread(client.wait_for_ready, timeout=60)
        if not ready:
            logger.error("Auto-start failed: JuniperData not ready after 60s")
            return

        # Create dataset via JuniperData
        dataset_params = json.loads(settings.auto_dataset_params)
        logger.info(f"Auto-start: creating '{settings.auto_dataset}' dataset with params={dataset_params}")
        result = await asyncio.to_thread(
            client.create_dataset,
            generator=settings.auto_dataset,
            params=dataset_params,
            persist=True,
        )
        dataset_id = result["dataset_id"]
        logger.info(f"Auto-start: dataset created — id={dataset_id}")

        # Download training data as numpy arrays
        arrays = await asyncio.to_thread(client.download_artifact_npz, dataset_id)
        x_train = torch.tensor(arrays["X_train"], dtype=torch.float32)
        y_train = torch.tensor(arrays["y_train"], dtype=torch.float32)
        logger.info(f"Auto-start: training data loaded — {x_train.shape[0]} samples, {x_train.shape[1]} features")

        # Create network
        network_config = json.loads(settings.auto_network)
        network_config.setdefault("epochs_max", settings.auto_train_epochs)
        lifecycle: TrainingLifecycleManager = app.state.lifecycle
        network_info = lifecycle.create_network(**network_config)
        logger.info(f"Auto-start: network created — {network_info['input_size']}x{network_info['output_size']}")

        # Start training
        train_result = lifecycle.start_training(x=x_train, y=y_train)
        logger.info(f"Auto-start: training initiated — {train_result}")

    except Exception:
        logger.exception("Auto-start training failed")


async def _auto_start_canopy(
    app: FastAPI,
    settings: Settings,
    managed_services: list,
) -> None:
    """Start juniper-canopy after juniper-cascor is accepting connections.

    Runs as a background asyncio task. Waits for the cascor API to become
    healthy before launching canopy, so canopy can connect on startup.
    Canopy is always started in normal mode (JUNIPER_CANOPY_DEMO_MODE=false).
    """
    try:
        from api.service_launcher import start_service, wait_for_health

        own_url = f"http://localhost:{settings.port}/v1/health"
        logger.info(f"Auto-start canopy: waiting for cascor at {own_url}")
        ready = await wait_for_health(own_url, timeout=30.0, interval=1.0)
        if not ready:
            logger.error("Auto-start canopy: cascor did not become healthy in 30s, aborting")
            return

        data_url = os.environ.get("JUNIPER_DATA_URL", "http://localhost:8100")
        canopy_env = {
            "JUNIPER_CANOPY_DEMO_MODE": "false",
            "JUNIPER_CANOPY_CASCOR_SERVICE_URL": f"http://localhost:{settings.port}",
            "JUNIPER_CANOPY_JUNIPER_DATA_URL": data_url,
        }

        svc = await start_service(
            name="juniper-canopy",
            command=settings.auto_start_canopy_command,
            health_url="http://localhost:8050/v1/health",
            env_overrides=canopy_env,
        )
        if svc:
            managed_services.append(svc)
        else:
            logger.error("Failed to auto-start juniper-canopy service")

    except Exception:
        logger.exception("Auto-start canopy failed")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override. If not provided,
                  settings are loaded from environment variables.

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    # Disable interactive API docs when authentication is enabled (production).
    docs_enabled = not settings.api_keys
    app = FastAPI(
        title="JuniperCascor API",
        description="Cascade Correlation Neural Network training service",
        version=_API_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    app.state.settings = settings

    # CORS: only enable when origins are explicitly configured.
    allow_credentials = bool(settings.cors_origins) and "*" not in settings.cors_origins

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Request body size limit
    app.add_middleware(RequestBodyLimitMiddleware)

    # Security headers (outermost — runs on every response)
    app.add_middleware(SecurityHeadersMiddleware)

    # Security (API key auth + rate limiting)
    api_key_auth = APIKeyAuth(settings.api_keys)
    rate_limiter = RateLimiter(
        requests_per_minute=settings.rate_limit_requests_per_minute,
        enabled=settings.rate_limit_enabled,
    )
    app.add_middleware(SecurityMiddleware, api_key_auth=api_key_auth, rate_limiter=rate_limiter)
    app.state.api_key_auth = api_key_auth

    # Observability middleware (added after SecurityMiddleware, before CORS)
    # Middleware execution is LIFO: last added runs first.
    # Order: RequestIdMiddleware → PrometheusMiddleware → SecurityMiddleware → SecurityHeaders → CORS
    if settings.metrics_enabled:
        app.add_middleware(PrometheusMiddleware, service_name="juniper-cascor", namespace="juniper_cascor")
    app.add_middleware(RequestIdMiddleware)

    # REST Routes
    app.include_router(health.router, prefix="/v1")
    app.include_router(network.router, prefix="/v1")
    app.include_router(training.router, prefix="/v1")
    app.include_router(metrics.router, prefix="/v1")
    app.include_router(dataset.router, prefix="/v1")
    app.include_router(decision_boundary.router, prefix="/v1")

    # WebSocket Routes
    app.websocket("/ws/training")(training_stream_handler)
    app.websocket("/ws/control")(control_stream_handler)

    # Mount Prometheus metrics endpoint
    if settings.metrics_enabled:
        app.mount("/metrics", get_prometheus_app())

    # Exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        logger.debug("Validation error: %s", exc)
        return JSONResponse(
            status_code=400,
            content=error_response("VALIDATION_ERROR", "Invalid request parameters"),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content=error_response("INTERNAL_ERROR", "Internal server error"),
        )

    return app
