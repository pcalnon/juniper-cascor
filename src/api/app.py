"""FastAPI application factory and configuration."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.lifecycle.manager import TrainingLifecycleManager
from api.models.common import error_response
from api.routes import dataset, decision_boundary, health, metrics, network, training
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

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
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

    logger.info("JuniperCascor API shutting down")


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

    app = FastAPI(
        title="JuniperCascor API",
        description="Cascade Correlation Neural Network training service",
        version=_API_VERSION,
        lifespan=lifespan,
    )

    app.state.settings = settings

    # CORS
    allow_credentials = bool(settings.cors_origins) and "*" not in settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    # Exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_response("VALIDATION_ERROR", str(exc)),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content=error_response("INTERNAL_ERROR", "Internal server error"),
        )

    return app
