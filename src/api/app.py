"""FastAPI application factory and configuration."""

import asyncio
import importlib.metadata
import ipaddress
import json
import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from juniper_service_core import enforce_auth_posture
from pydantic_core import PydanticSerializationError

from api import provenance
from api.lifecycle.manager import TrainingLifecycleManager
from api.middleware import RequestBodyLimitMiddleware, SecurityHeadersMiddleware, SecurityMiddleware
from api.models.common import error_response
from api.observability import MetricsAuthMiddleware, PrometheusMiddleware, RequestIdMiddleware, configure_logging, configure_sentry, get_prometheus_app, set_build_info
from api.routes import admin, dataset, decision_boundary, health, history, metrics, network, snapshots, training, workers
from api.secrets import get_secret
from api.security import APIKeyAuth, RateLimiter
from api.settings import Settings, get_settings
from api.websocket.control_stream import control_stream_handler
from api.websocket.manager import WebSocketManager
from api.websocket.training_stream import training_stream_handler
from api.websocket.worker_stream import worker_stream_handler
from api.workers.coordinator import WorkerCoordinator
from api.workers.registry import WorkerRegistry
from cascor_constants.constants_api import _PROJECT_API_CANOPY_DEMO_MODE_DISABLED, _PROJECT_API_CANOPY_HEALTH_CHECK_URL, _PROJECT_API_CANOPY_STARTUP_CHECK_INTERVAL, _PROJECT_API_CANOPY_STARTUP_WAIT_TIMEOUT, _PROJECT_API_JUNIPER_DATA_READY_TIMEOUT, _PROJECT_API_JUNIPER_DATA_URL_DEFAULT, _PROJECT_API_SELF_HEALTH_CHECK_URL_TEMPLATE

# BUG-CC-04: single source of truth for version is pyproject.toml; read at runtime.
try:
    _API_VERSION: str = importlib.metadata.version("juniper-cascor")
except importlib.metadata.PackageNotFoundError:
    _API_VERSION = "0.0.0-dev"

logger = logging.getLogger("juniper_cascor.api")


class NonLoopbackBindError(RuntimeError):
    """Raised at startup when cascor is configured to bind a non-loopback
    interface without a bind attestation (neither a loopback-only host
    publish nor a fronting authenticating proxy attested).

    SEC-F22 / D2 (juniper-ml
    ``notes/JUNIPER_CANOPY_CONTROL_SURFACE_AUTH_AND_NAT_DESIGN_2026-07-03.md``
    §4 Option A / §8 D2). Fail-closed: the process refuses to start rather
    than silently exposing the un-fronted control surface on a public
    interface.
    """


def _is_loopback_host(host: str) -> bool:
    """Return True when ``host`` names a loopback bind target.

    Treats the literal hostname ``localhost`` and every loopback IP literal
    (127.0.0.0/8, ::1, and IPv4-mapped IPv6 forms such as ``::ffff:127.0.0.1``)
    as loopback. A non-IP hostname other than ``localhost`` is treated
    conservatively as NON-loopback (fail-closed) because it may resolve to a
    routable address. The unspecified addresses ``0.0.0.0`` / ``::`` (bind-all)
    are — correctly — NOT loopback.
    """
    h = (host or "").strip().lower()
    if not h:
        # An empty host is bind-all in most servers — treat as non-loopback.
        return False
    if h in {"localhost", "localhost.localdomain"}:
        return True
    # Strip IPv6 brackets and any zone-id before parsing.
    h = h.strip("[]")
    if "%" in h:
        h = h.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        # Not an IP literal (a hostname); conservatively non-loopback.
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def _cli_option_value(argv: list[str], option: str) -> str | None:
    """Return a CLI option value from ``--name value`` or ``--name=value``."""
    prefix = f"{option}="
    for index, arg in enumerate(argv):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == option and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _settings_with_uvicorn_cli_bind(settings: Settings, argv: list[str] | None = None) -> Settings:
    """Overlay uvicorn CLI bind args onto settings for startup guard parity.

    ``uvicorn api.app:create_app --factory --host 0.0.0.0`` is a documented
    operational entry point. Uvicorn consumes ``--host`` itself and does not set
    ``JUNIPER_CASCOR_HOST``, so a guard that only checks ``Settings.host`` would
    see the default loopback host while uvicorn binds a public socket. When the
    factory is invoked from the uvicorn CLI, mirror the CLI bind host/port into a
    transient Settings copy before the lifespan guard runs.
    """
    args = list(sys.argv if argv is None else argv)
    if not any("uvicorn" in arg for arg in args[:2]) and "api.app:create_app" not in args:
        return settings

    updates: dict[str, object] = {}
    host = _cli_option_value(args, "--host")
    if host:
        updates["host"] = host

    port = _cli_option_value(args, "--port")
    if port:
        try:
            updates["port"] = int(port)
        except ValueError:
            logger.warning("Ignoring non-integer uvicorn --port value for bind guard parity: %r", port)

    if not updates:
        return settings
    return settings.model_copy(update=updates)


def enforce_bind_attestation_guard(settings: Settings) -> None:
    """Refuse to start on a non-loopback bind without a bind attestation.

    SEC-F22 / D2 — the symmetric counterpart to the canopy bind-guard. The
    only effective control protecting cascor's un-authenticated control/worker
    WebSocket surface in the containerized stack is the network boundary. This
    guard converts that load-bearing precondition into an enforced invariant
    using the two-flag attestation scheme (identical across canopy / cascor /
    juniper-deploy):

    * Loopback ``host`` (127.0.0.0/8, ::1, localhost, IPv4-mapped loopback)
      -> always start.
    * Non-loopback ``host`` (e.g. ``0.0.0.0``) with EITHER
      ``loopback_publish_attested`` (env
      ``JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED`` — the port is reachable only
      via a loopback-only host publish) OR ``auth_proxy_attested`` (env
      ``JUNIPER_CASCOR_AUTH_PROXY_ATTESTED`` — a fronting authenticating reverse
      proxy terminates access) -> start, logging a WARNING that names which
      attestation permitted the bind so it is auditable.
    * Non-loopback ``host`` with NEITHER attestation (the default) -> raise
      :class:`NonLoopbackBindError` after a CRITICAL log (fail-closed, loud).
      There is no warning-only mode: an un-attested non-loopback bind always
      hard-fails.

    Called from the application ``lifespan`` startup, before uvicorn binds the
    socket, so a mis-configured bring-up never begins accepting connections.

    Note (deploy roll-out is owner-gated): the container binds
    ``JUNIPER_CASCOR_HOST=0.0.0.0`` behind a loopback host-publish, so enabling
    this guard in the deploy requires setting
    ``JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true`` there — a Phase-1 deploy
    change the platform owner approves separately (design §7 Phase 1).
    """
    host = settings.host
    if _is_loopback_host(host):
        return
    if settings.loopback_publish_attested or settings.auth_proxy_attested:
        if settings.loopback_publish_attested and settings.auth_proxy_attested:
            permitted_by = "JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED and JUNIPER_CASCOR_AUTH_PROXY_ATTESTED"
        elif settings.loopback_publish_attested:
            permitted_by = "JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED"
        else:
            permitted_by = "JUNIPER_CASCOR_AUTH_PROXY_ATTESTED"
        logger.warning(
            "cascor is binding a NON-loopback interface (%s:%s) — permitted by %s. The control/worker WebSocket surface has no app-layer auth of its own; this bind is safe only while that attestation holds (a loopback-only host publish and/or a fronting authenticating reverse proxy actually fronts this port). If neither is true it exposes the surface to the whole reachable network (SEC-F22).",
            host,
            settings.port,
            permitted_by,
        )
        return
    logger.critical(
        "REFUSING TO START: cascor is configured to bind a NON-loopback interface (%s:%s) without a bind attestation. The control/worker WebSocket surface has no app-layer authentication of its own — its only effective control is the loopback network boundary (SEC-F22). "
        "Bind 127.0.0.1 (recommended for local/dev), or attest the control that actually protects the port: set JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true when the port is reachable only via a loopback-only host publish, or JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true when a fronting authenticating reverse proxy terminates access.",
        host,
        settings.port,
    )
    raise NonLoopbackBindError(f"Refusing to bind non-loopback host {host!r} without a bind attestation (set JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true or JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true) — SEC-F22 bind-guard.")


def _log_startup_task_exception(task: asyncio.Task) -> None:
    """Done-callback that surfaces exceptions from fire-and-forget startup tasks.

    CONC-09 (Phase 3C): without this hook the auto_start_training and
    auto_start_canopy tasks were created with `asyncio.create_task(...)` and
    no saved reference, which (a) made them eligible for garbage collection
    while still pending and (b) meant any exception they raised was logged
    only as the cryptic "Task exception was never retrieved" warning emitted
    by the loop at GC time. The lifespan handler now stores the task in
    `app.state.startup_tasks` AND attaches this callback so any non-cancel
    exception is logged with full traceback at error level the moment the
    task finishes.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Startup task %s failed: %s", task.get_name(), exc, exc_info=exc)


def _register_worker_metrics_collector(
    app: FastAPI,
    settings: Settings,
    worker_registry: WorkerRegistry,
    worker_coordinator: WorkerCoordinator,
) -> None:
    """Register the worker -> Prometheus bridge collector at startup.

    METRICS-MON R5.4-pre: closes the gap flagged by R5.1
    (juniper-deploy#48) and R5.3 (juniper-deploy#46) — heartbeat
    freshness, last-task duration, recent-task p50/p95, and GPU
    utilisation are now exposed on the cascor ``/metrics`` surface for
    two internal SLIs and the operator dashboard. Single-registration
    per R1.4: the collector instance is held on ``app.state`` so the
    shutdown path can unregister it cleanly when the lifespan exits.
    Extracted from the lifespan handler purely for cyclomatic-complexity
    discipline (flake8 C901 budget) — the initialization itself is
    one-shot and unconditional given ``metrics_enabled``.

    Audit-doc §4.2 (2026-05-04): the coordinator is now also wired so
    the collector can emit ``juniper_cascor_pending_tasks`` on each
    scrape. Closes the catalog §4.2 SLI gap and lifts the
    ``CascorPendingTasksSaturated`` alert's ``absent_over_time(...) == 0``
    inertness guard (juniper-deploy/prometheus/alert_rules.yml).
    """
    if not settings.metrics_enabled:
        return

    from prometheus_client import REGISTRY

    from api.workers.metrics import WorkerRegistryCollector

    worker_metrics_collector = WorkerRegistryCollector(
        worker_registry,
        coordinator=worker_coordinator,
    )
    REGISTRY.register(worker_metrics_collector)
    app.state.worker_metrics_collector = worker_metrics_collector
    logger.info("Worker -> Prometheus bridge collector registered")


def _unregister_worker_metrics_collector(app: FastAPI) -> None:
    """Unregister the worker bridge collector at shutdown.

    METRICS-MON R5.4-pre: re-creating the app (test harness, in-process
    restart) without unregistering would trip the prometheus_client
    "duplicated metric" guard. Best-effort: a missing or already-gone
    collector is logged at debug level rather than raised.
    """
    worker_metrics_collector = getattr(app.state, "worker_metrics_collector", None)
    if worker_metrics_collector is None:
        return
    try:
        from prometheus_client import REGISTRY

        REGISTRY.unregister(worker_metrics_collector)
        logger.info("Worker -> Prometheus bridge collector unregistered")
    except Exception as exc:
        logger.debug("Best-effort worker collector unregister failed: %s", exc)


def _init_worker_security(app: FastAPI, settings: Settings, worker_coordinator: WorkerCoordinator) -> None:
    """Initialize optional worker-security subsystems based on feature flags."""
    if settings.worker_rate_limit_enabled:
        from api.workers.security import ConnectionRateLimiter

        app.state.worker_rate_limiter = ConnectionRateLimiter(
            max_connections_per_minute=settings.worker_rate_limit_connections_per_minute,
            burst_size=settings.worker_rate_limit_burst_size,
        )
        logger.info("Worker connection rate limiter enabled")

    if settings.worker_anomaly_detection_enabled:
        from api.workers.security import AnomalyDetector

        anomaly_detector = AnomalyDetector(
            min_training_time=settings.worker_anomaly_min_training_time,
            perfect_corr_threshold=settings.worker_anomaly_perfect_corr_threshold,
        )
        app.state.anomaly_detector = anomaly_detector
        worker_coordinator._anomaly_detector = anomaly_detector
        logger.info("Worker anomaly detection enabled")

    if settings.worker_audit_logging_enabled:
        from api.workers.audit import AuditLogger

        app.state.audit_logger = AuditLogger()
        logger.info("Worker audit logging enabled")

    if settings.worker_metrics_enabled:
        from api.workers.audit import WorkerMetrics

        app.state.worker_metrics = WorkerMetrics()
        logger.info("Worker metrics collection enabled")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    settings: Settings = app.state.settings

    configure_logging(settings.log_level, settings.log_format, "juniper-cascor")

    # SEC-F22 / D2 — refuse to start on a non-loopback bind without a bind
    # attestation (loopback-only host publish OR fronting authenticating
    # proxy), before uvicorn binds the socket or any background thread is
    # spawned. Fail-closed and loud.
    enforce_bind_attestation_guard(settings)

    # SEC-F01 (HO-2): boot-time auth-posture self-check. An empty/placeholder
    # JUNIPER_CASCOR_API_KEYS secret silently disables APIKeyAuth and cascor
    # serves protected routes OPEN behind a healthy health check; make that
    # posture loud here, before serving begins. The intended posture comes
    # from JUNIPER_CASCOR_REQUIRE_AUTH (settings.require_auth; default
    # false): false = loud WARNING only (bare/dev profile), true = a
    # missing/blank key is a boot FAILURE (CRITICAL + AuthPostureError) —
    # set true wherever secrets are provisioned (the composed juniper-deploy
    # stack). Bypass with JUNIPER_SKIP_AUTH_POSTURE_CHECK=1 (logged loudly).
    enforce_auth_posture(
        settings.api_keys,
        require_auth=settings.require_auth,
        service_name="juniper-cascor",
        logger=logger,
    )

    configure_sentry(settings.sentry_dsn, "juniper-cascor", _API_VERSION)
    if settings.metrics_enabled:
        set_build_info("juniper_cascor", _API_VERSION, git_sha=provenance.git_sha(), build_date=provenance.build_date())

    logger.info(f"JuniperCascor API v{_API_VERSION} starting")
    logger.info(f"Listening on {settings.host}:{settings.port}")

    # Create WebSocket manager
    ws_manager = WebSocketManager(
        max_connections=settings.ws_max_connections,
        max_replay_buffer_size=settings.ws_replay_buffer_size,
        send_timeout_seconds=settings.ws_send_timeout_seconds,
        max_connections_per_ip=settings.ws_max_connections_per_ip,  # SEC-03
        max_connections_global=settings.ws_max_connections_global,  # SEC-F19 D4a
        max_connections_per_identity=settings.ws_max_connections_per_identity,  # SEC-F19 D4b
        max_message_size_bytes=settings.ws_max_message_size_bytes,  # GAP-WS-18
        chunk_payload_size_bytes=settings.ws_chunk_payload_size_bytes,  # GAP-WS-18
        emission_summary_interval_sec=settings.ws_emission_summary_interval_sec,  # C3/T5
    )
    ws_manager.set_event_loop(asyncio.get_running_loop())
    app.state.ws_manager = ws_manager
    app.state.settings = settings
    logger.info("WebSocket manager initialized")

    # Create lifecycle manager for training coordination
    lifecycle = TrainingLifecycleManager()
    lifecycle.set_ws_manager(
        ws_manager,
        state_throttle_interval=settings.ws_state_throttle_coalesce_ms / 1000.0,
    )
    app.state.lifecycle = lifecycle
    logger.info("Lifecycle manager initialized")

    # Create worker registry and coordinator for remote WebSocket workers
    worker_registry = WorkerRegistry(heartbeat_timeout=settings.remote_workers_heartbeat_timeout)
    worker_coordinator = WorkerCoordinator(
        registry=worker_registry,
        task_reassignment_timeout=settings.remote_workers_task_reassignment_timeout,
    )
    worker_coordinator.start_monitor()
    app.state.worker_registry = worker_registry
    app.state.worker_coordinator = worker_coordinator
    lifecycle.set_worker_coordinator(worker_coordinator)
    logger.info("Worker registry and coordinator initialized")

    _register_worker_metrics_collector(app, settings, worker_registry, worker_coordinator)

    # Worker Security (Phase 4) — conditionally initialize based on feature flags
    _init_worker_security(app, settings, worker_coordinator)

    # Auto-start companion services (non-containerized mode)
    managed_services: list = []
    app.state.managed_services = managed_services

    if settings.auto_start_data_service:
        from api.service_launcher import start_service

        # CFG-04: Settings field consolidates the JUNIPER_DATA_URL env-var
        # lookup; ``or DEFAULT`` preserves the legacy localhost:8100
        # fallback when neither the canonical nor the prefixed env var
        # is set.
        data_url = settings.juniper_data_url or _PROJECT_API_JUNIPER_DATA_URL_DEFAULT
        logger.info("Auto-start juniper-data service is ENABLED")
        svc = await start_service(
            name="juniper-data",
            command=settings.auto_start_data_service_command,
            health_url=f"{data_url.rstrip('/')}/v1/health",
            env_overrides={"JUNIPER_DATA_HOST": "0.0.0.0"},  # nosec B104 — env override for local dev, not a socket bind
        )
        if svc:
            managed_services.append(svc)
        else:
            logger.error("Failed to auto-start juniper-data service")

    # CONC-09 (Phase 3C): track every fire-and-forget startup task on
    # `app.state.startup_tasks` so the loop keeps a strong reference (the
    # asyncio docs explicitly warn that tasks created by `asyncio.create_task`
    # without a saved reference can be garbage-collected mid-flight) and
    # exceptions are surfaced via a done-callback rather than silently
    # swallowed by the loop. The task list is also drained during shutdown so
    # in-flight auto-start work is cancelled cleanly instead of leaking past
    # the lifespan boundary.
    startup_tasks: list[asyncio.Task] = []
    app.state.startup_tasks = startup_tasks

    if settings.auto_start:
        logger.warning("Auto-start training is ENABLED — this should only be used in demo/dev environments")
        task = asyncio.create_task(_auto_start_training(app, settings), name="auto_start_training")
        task.add_done_callback(_log_startup_task_exception)
        startup_tasks.append(task)

    # Auto-start canopy as background task (waits for cascor to be accepting connections)
    if settings.auto_start_canopy:
        logger.info("Auto-start juniper-canopy is ENABLED (normal mode)")
        task = asyncio.create_task(_auto_start_canopy(app, settings, managed_services), name="auto_start_canopy")
        task.add_done_callback(_log_startup_task_exception)
        startup_tasks.append(task)

    yield

    # CONC-09 (Phase 3C): cancel any startup tasks still running at
    # shutdown so they don't outlive the lifespan and access freshly torn
    # down state. Awaiting with `return_exceptions=True` lets every task
    # finish its CancelledError handling without one bad task masking the
    # others.
    in_flight_startup_tasks = [t for t in getattr(app.state, "startup_tasks", []) if not t.done()]
    if in_flight_startup_tasks:
        logger.info("Cancelling %d in-flight startup task(s) at shutdown", len(in_flight_startup_tasks))
        for task in in_flight_startup_tasks:
            task.cancel()
        await asyncio.gather(*in_flight_startup_tasks, return_exceptions=True)

    _unregister_worker_metrics_collector(app)

    # Shutdown: stop worker coordinator
    worker_coordinator = getattr(app.state, "worker_coordinator", None)
    if worker_coordinator is not None:
        worker_coordinator.shutdown()
        logger.info("Worker coordinator shut down")

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

        # CFG-04: Settings field consolidates the JUNIPER_DATA_URL env-var
        # lookup; ``or DEFAULT`` preserves the legacy localhost:8100
        # fallback.
        data_url = settings.juniper_data_url or _PROJECT_API_JUNIPER_DATA_URL_DEFAULT
        api_key = get_secret("JUNIPER_DATA_API_KEY")

        client = JuniperDataClient(base_url=data_url, api_key=api_key)

        # Wait for JuniperData service
        logger.info(f"Auto-start: waiting for JuniperData at {data_url}")
        ready = await asyncio.to_thread(client.wait_for_ready, timeout=_PROJECT_API_JUNIPER_DATA_READY_TIMEOUT)
        if not ready:
            logger.error(f"Auto-start failed: JuniperData not ready after {_PROJECT_API_JUNIPER_DATA_READY_TIMEOUT}s")
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

        # Create network — infer input/output sizes from training data.
        # C2b / Q1 outcome (c): the ``epochs_max`` seed (``settings.auto_train_epochs``)
        # is gone — the engine never read the attribute (the granular limits do the
        # gating), and ``epochs_max`` is now derived per run from those limits
        # (TrainingLifecycleManager.derive_epochs_cap). The seed's only observable
        # effect was an incoherent ``max_epochs`` on the status surface.
        # ``auto_train_epochs`` remains in Settings as a deprecated no-op for
        # env-var compatibility (see settings.py).
        network_config = json.loads(settings.auto_network)
        network_config.setdefault("input_size", x_train.shape[1])
        network_config.setdefault("output_size", y_train.shape[1] if y_train.dim() > 1 else 1)
        lifecycle: TrainingLifecycleManager = app.state.lifecycle
        network_info = lifecycle.create_network(**network_config)
        logger.info(f"Auto-start: network created — {network_info['input_size']}x{network_info['output_size']}")

        # Start training
        train_result = lifecycle.start_training(X=x_train, y=y_train)
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

        own_url = _PROJECT_API_SELF_HEALTH_CHECK_URL_TEMPLATE.format(port=settings.port)
        logger.info(f"Auto-start canopy: waiting for cascor at {own_url}")
        ready = await wait_for_health(own_url, timeout=_PROJECT_API_CANOPY_STARTUP_WAIT_TIMEOUT, interval=_PROJECT_API_CANOPY_STARTUP_CHECK_INTERVAL)
        if not ready:
            logger.error(f"Auto-start canopy: cascor did not become healthy in {_PROJECT_API_CANOPY_STARTUP_WAIT_TIMEOUT}s, aborting")
            return

        # CFG-04: Settings field consolidates the JUNIPER_DATA_URL env-var
        # lookup; the resolved value is forwarded to the canopy subprocess
        # via ``JUNIPER_CANOPY_JUNIPER_DATA_URL`` (canopy's prefixed env
        # var). ``or DEFAULT`` preserves the legacy localhost:8100
        # fallback.
        data_url = settings.juniper_data_url or _PROJECT_API_JUNIPER_DATA_URL_DEFAULT
        canopy_env = {
            "JUNIPER_CANOPY_DEMO_MODE": _PROJECT_API_CANOPY_DEMO_MODE_DISABLED,
            "JUNIPER_CANOPY_CASCOR_SERVICE_URL": f"http://localhost:{settings.port}",
            "JUNIPER_CANOPY_JUNIPER_DATA_URL": data_url,
        }

        svc = await start_service(
            name="juniper-canopy",
            command=settings.auto_start_canopy_command,
            health_url=_PROJECT_API_CANOPY_HEALTH_CHECK_URL,
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
        settings = _settings_with_uvicorn_cli_bind(get_settings())

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
    app.include_router(snapshots.router, prefix="/v1")
    app.include_router(workers.router, prefix="/v1")
    app.include_router(admin.router, prefix="/v1")  # Phase 2 P2-1a — experimental-functions gate
    app.include_router(history.router, prefix="/v1")  # Phase 2 P2-2 Follow-up B — dataset_swap event fetch

    # WebSocket Routes
    app.websocket("/ws/training")(training_stream_handler)
    app.websocket("/ws/control")(control_stream_handler)
    app.websocket("/ws/v1/workers")(worker_stream_handler)

    # Mount Prometheus metrics endpoint (SEC-16 / POC §3.1: wrap with
    # trusted-IP auth because ASGI sub-app mounts bypass SecurityMiddleware
    # — and ``/metrics`` is now in EXEMPT_PATHS specifically so it can be).
    if settings.metrics_enabled:
        app.mount(
            "/metrics",
            MetricsAuthMiddleware(get_prometheus_app(), settings.metrics_trusted_ips),
        )

    # Exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        # ``PydanticSerializationError`` subclasses ValueError, but it is a SERVER
        # fault: the app failed to serialise its own response. Reporting it as 400
        # misattributes our defect to the caller, hides it from 5xx alerting, and
        # replaces the diagnostic with "Invalid request parameters". Classify it as
        # the 500 it is, and log at exception level so the traceback survives.
        #
        # ``coerce_native_scalars`` (api/models/common.py) pre-empts the common
        # numpy-scalar case inside ``success_response``; this catches every other
        # serialisation fault, which that helper by construction cannot.
        if isinstance(exc, PydanticSerializationError):
            logger.exception("Response serialization failed")
            return JSONResponse(
                status_code=500,
                content=error_response("INTERNAL_ERROR", "Internal server error"),
            )
        logger.debug("Validation error: %s", exc)
        return JSONResponse(
            status_code=400,
            content=error_response("VALIDATION_ERROR", "Invalid request parameters"),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """API-09 (migration complete after PR 3).

        Wraps every ``raise HTTPException(...)`` in cascor's API
        routes into the project's standard ``ResponseEnvelope`` /
        ``ErrorResponse`` shape so clients no longer have to parse
        two error formats (the FastAPI default ``{"detail": "..."}``
        for ``HTTPException`` vs. the envelope shape for ``ValueError``
        and ``Exception``). See
        ``notes/API_09_ERROR_ENVELOPE_MIGRATION_DESIGN_2026-05-21.md``
        for the full migration history.

        Response shape (final, post-PR-3):

        .. code-block:: json

            {
              "status": "error",
              "error": {
                "code": "HTTP_404",
                "message": "No network loaded",
                "detail": null
              },
              "meta": {"timestamp": ..., "version": ...}
            }

        The PR 1 → PR 3 deprecation window also emitted a top-level
        ``"detail"`` alias of ``error.message`` so pre-migration
        consumers (notably ``juniper-cascor-client`` before commit
        b0a636a3, 2026-02-21) kept working unchanged. PR 3 dropped
        the alias after juniper-cascor-client #59 pinned the
        envelope-aware parser and the design-doc-mandated soak
        window completed.

        Headers attached to the ``HTTPException`` (e.g.
        ``WWW-Authenticate`` on 401, ``Retry-After`` on 429) are
        preserved via ``headers=exc.headers`` to match FastAPI's
        default-handler behavior; stripping them would be a
        regression for downstream HTTP semantics.

        ``error.code`` uses the string form ``"HTTP_NNN"`` rather
        than the bare integer to (a) match the existing
        ``ErrorDetail.code: str`` schema in
        ``api/models/common.py`` (Pydantic v2 strict mode does not
        coerce ``int -> str``), (b) match the existing in-use
        semantic codes (``VALIDATION_ERROR``, ``INTERNAL_ERROR``),
        and (c) preserve the future migration path to richer
        semantic codes (``NETWORK_NOT_FOUND``, etc. — tracked as
        API-09b in the design doc's out-of-scope section) without
        any schema change.
        """
        # Starlette's ``HTTPException.__init__`` auto-fills ``detail``
        # with ``HTTPStatus(status_code).phrase`` when the caller
        # doesn't pass one, so ``exc.detail`` is always a string by
        # the time we get here. ``str()`` defends against future
        # subclasses that might return non-str detail objects.
        message = str(exc.detail)
        envelope = error_response(
            code=f"HTTP_{exc.status_code}",
            message=message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=envelope,
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content=error_response("INTERNAL_ERROR", "Internal server error"),
        )

    return app
