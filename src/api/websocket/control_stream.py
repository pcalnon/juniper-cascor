"""WebSocket handler for /ws/control — training command channel.

Client-to-server command endpoint. Accepts JSON commands:
{
    "command": "start" | "stop" | "pause" | "resume" | "reset" | "set_params",
    "command_id": "<optional-uuid>",  // echoed in response for correlation
    "params": { ... }  // optional, for start/set_params
}

Responds with command_response acknowledgments. Note: command_response
messages have NO ``seq`` field (D-03 canonical). The /ws/control channel
has no replay buffer.

Phase B-pre-b: Origin validation, per-connection leaky bucket (10 cmd/s),
idle timeout (120s bidirectional), per-origin handshake cooldown.

Phase D: Per-command execution timeouts via ``asyncio.wait_for``. Commands
are dispatched to the thread pool (``asyncio.to_thread``) to avoid blocking
the event loop, then bounded: start=10s, stop/pause/resume=2s, set_params=1s,
reset=2s. Timeout → ``command_response{status:"error", error:"...timed out..."}``.

Phase F: Application-level heartbeat. Server sends ``{"type":"ping","ts":<float>}``
every ``ws_heartbeat_interval_sec`` (30s). Client must reply
``{"type":"pong"}`` within ``ws_heartbeat_pong_timeout_sec`` (10s) or
connection is closed with 1006. Pong receipt resets the idle timeout timer.
"""

import asyncio
import json
import logging
import time

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from api.models.training import TrainingParamUpdateRequest
from api.settings import get_settings
from api.websocket.control_security import HandshakeCooldown, LeakyBucket, validate_control_origin
from api.websocket.manager import ws_authenticate, ws_identity_key
from api.websocket.messages import create_control_ack_message

logger = logging.getLogger("juniper_cascor.api.websocket.control")

_VALID_COMMANDS = {"start", "stop", "pause", "resume", "reset", "set_params"}
_MAX_MESSAGE_SIZE = 65536  # 64KB

# Phase D: per-command execution timeouts (§S10)
_COMMAND_TIMEOUTS: dict[str, float] = {
    "start": 10.0,
    "stop": 2.0,
    "pause": 2.0,
    "resume": 2.0,
    "reset": 2.0,
    "set_params": 1.0,
}

# Observability: lazy-initialized Prometheus counter (§S10.7)
_command_received_counter = None


def _get_command_counter():
    """Lazily create the command received counter when metrics are available.

    Idempotent via :func:`juniper_observability.register_or_reuse`: if a
    test fixture (or in-process re-init) has nulled the module-level
    cache while leaving the underlying Counter registered, the helper
    adopts the existing collector instead of raising
    ``ValueError: Duplicated timeseries``.
    """
    global _command_received_counter
    if _command_received_counter is None:
        try:
            from juniper_observability import register_or_reuse
            from prometheus_client import Counter

            _command_received_counter = register_or_reuse(
                Counter,
                "cascor_ws_control_command_received_total",
                "WebSocket control commands received",
                ["command"],
            )
        except ImportError:
            _command_received_counter = False  # sentinel: prometheus not available
    return _command_received_counter


# Module-level handshake cooldown (shared across connections, cleared on restart)
_handshake_cooldown = None


def _get_cooldown() -> HandshakeCooldown:
    """Lazily initialize the handshake cooldown from settings."""
    global _handshake_cooldown
    if _handshake_cooldown is None:
        settings = get_settings()
        _handshake_cooldown = HandshakeCooldown(
            max_rejections=settings.ws_control_cooldown_rejections,
            window_sec=settings.ws_control_cooldown_window_sec,
            block_sec=settings.ws_control_cooldown_block_sec,
        )
    return _handshake_cooldown


def _get_client_ip(websocket: WebSocket) -> str:
    """Extract client IP from WebSocket connection."""
    if websocket.client:
        return websocket.client[0]
    return "unknown"


async def _check_handshake_gates(websocket: WebSocket, settings, client_ip: str) -> bool:
    """Run pre-accept handshake gates. Returns True if the connection may proceed."""
    if settings.disable_ws_control_endpoint:
        await websocket.close(code=1013, reason="Control endpoint disabled")
        return False

    cooldown = _get_cooldown()
    if cooldown.is_blocked(client_ip):
        remaining = cooldown.get_block_remaining(client_ip)
        logger.warning("Control WS: IP %s blocked (cooldown), remaining=%ss", client_ip, remaining)
        await websocket.close(code=4029, reason="Too many rejected handshakes")
        return False

    if not await ws_authenticate(websocket):
        cooldown.record_rejection(client_ip)
        return False

    if settings.ws_control_allowed_origins:
        if not validate_control_origin(websocket, settings.ws_control_allowed_origins):
            cooldown.record_rejection(client_ip)
            await websocket.close(code=4003, reason="Origin not allowed")
            return False

    return True


async def _handle_command_message(websocket: WebSocket, lifecycle, msg: dict, bucket: LeakyBucket) -> None:
    """Validate and dispatch a single command message; send the response."""
    command = msg.get("command", "")
    command_id = msg.get("command_id")

    # OBS-WIRE-02 (3.10): emit ``cascor_ws_command_responses_total`` on
    # every response arm. ``status`` is closed-set (``success`` /
    # ``error`` / ``rate_limited``); ``command`` is open-set-by-convention
    # but bounded by ``_VALID_COMMANDS`` upstream. All emissions are
    # wrapped in defensive try/except per OBS-WIRE-01.
    def _emit_response(cmd: str, status: str) -> None:
        try:
            from api.observability import ws_inc_command_responses

            ws_inc_command_responses(cmd, status)
        except Exception:
            logger.debug("ws_inc_command_responses emission failed for command=%s status=%s", cmd, status, exc_info=True)

    if not bucket.try_acquire():
        retry_after = bucket.retry_after
        await websocket.send_json(
            {
                "type": "command_response",
                "command": command,
                "status": "rate_limited",
                "retry_after": retry_after,
                **({"command_id": command_id} if command_id else {}),
            }
        )
        _emit_response(command, "rate_limited")
        return

    if command not in _VALID_COMMANDS:
        await websocket.send_json(
            create_control_ack_message(
                command,
                "error",
                error=f"Unknown command: {command}",
                command_id=command_id,
                code="unknown_command",
            )
        )
        _emit_response(command, "error")
        return

    if lifecycle is None:
        await websocket.send_json(create_control_ack_message(command, "error", error="Lifecycle manager not available", command_id=command_id))
        _emit_response(command, "error")
        return

    counter = _get_command_counter()
    if counter:
        counter.labels(command=command).inc()

    timeout = _COMMAND_TIMEOUTS.get(command, 2.0)
    # OBS-WIRE-01 (A.3): time only the dispatch — the
    # ``asyncio.wait_for(asyncio.to_thread(...))`` span — NOT the
    # surrounding parse/validate/ack-send work. This binds catalog
    # SLI 4.4 (command-handler p95 < 50 ms). The send-ack call below
    # is itself instrumented via the broadcast-send histogram from
    # ``WebSocketManager._send_json``, so we don't double-count.
    handler_start = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_execute_command, lifecycle, command, msg.get("params")),
            timeout=timeout,
        )
        handler_duration = time.perf_counter() - handler_start
        await websocket.send_json(create_control_ack_message(command, "success", data=result, command_id=command_id))
        _emit_response(command, "success")
    except asyncio.TimeoutError:
        handler_duration = time.perf_counter() - handler_start
        logger.error("Command '%s' timed out after %ss", command, timeout)
        await websocket.send_json(create_control_ack_message(command, "error", error=f"Command timed out after {timeout}s", command_id=command_id))
        _emit_response(command, "error")
    except ValidationError as exc:
        # SEC-F10 (HO-5): a ``set_params`` payload that violates the shared
        # ``TrainingParamUpdateRequest`` bounds (negative / over-ceiling) is a
        # client error, not an execution failure — surface a clean, specific
        # error ack (the WS analogue of the REST PATCH 422) and keep the
        # connection open. Logged at INFO because it is expected client input.
        handler_duration = time.perf_counter() - handler_start
        logger.info("Command '%s' rejected — invalid params: %s", command, exc)
        await websocket.send_json(create_control_ack_message(command, "error", error="Invalid parameters", command_id=command_id, code="invalid_params"))
        _emit_response(command, "error")
    except Exception as e:
        handler_duration = time.perf_counter() - handler_start
        logger.error("Command '%s' failed: %s", command, e)
        await websocket.send_json(create_control_ack_message(command, "error", error="Command execution failed", command_id=command_id))
        _emit_response(command, "error")
    # OBS-WIRE-01 (A.3): observe the command-handler histogram. Done
    # outside the try/except chain so a single observation is emitted
    # whether the handler completed, timed out, or raised — the
    # SLO PromQL aggregates across all three.
    try:
        from api.observability import ws_observe_command_handler

        ws_observe_command_handler(command, handler_duration)
    except Exception:
        logger.debug("ws_observe_command_handler emission failed", exc_info=True)


async def _control_ping_loop(websocket: WebSocket, client_ip: str, hb_interval: float, hb_timeout: float, pong_received: asyncio.Event) -> None:
    """Application-level ping/pong loop closing the connection on pong timeout."""
    while True:
        await asyncio.sleep(hb_interval)
        pong_received.clear()
        try:
            await websocket.send_json({"type": "ping", "ts": time.time()})
        except Exception:
            return
        try:
            await asyncio.wait_for(pong_received.wait(), timeout=hb_timeout)
        except asyncio.TimeoutError:
            logger.info("Control WS: heartbeat timeout, closing: %s", client_ip)
            try:
                await websocket.close(code=1006, reason="Heartbeat timeout")
            except Exception:
                logger.debug("Control WS: close after heartbeat timeout failed for %s", client_ip, exc_info=True)
            return


async def _control_recv_loop(
    websocket: WebSocket,
    lifecycle,
    bucket: LeakyBucket,
    pong_received: asyncio.Event,
    idle_timeout: float,
    client_ip: str,
) -> None:
    """Receive loop: enforce idle timeout, dispatch commands, route pong frames."""
    while True:
        try:
            if idle_timeout and idle_timeout > 0:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=idle_timeout)
            else:
                raw = await websocket.receive_text()
        except asyncio.TimeoutError:
            logger.info("Control WS: idle timeout (%ds), closing: %s", idle_timeout, client_ip)
            await websocket.close(code=1000, reason="Idle timeout")
            return

        if len(raw) > _MAX_MESSAGE_SIZE:
            await websocket.send_json(create_control_ack_message("unknown", "error", error="Message too large"))
            continue

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_json(create_control_ack_message("unknown", "error", error="Invalid JSON"))
            await websocket.close(code=1003, reason="Malformed JSON")
            return

        if msg.get("type") == "pong":
            pong_received.set()
            continue

        await _handle_command_message(websocket, lifecycle, msg, bucket)


async def control_stream_handler(websocket: WebSocket) -> None:
    """Handle /ws/control WebSocket connections.

    Security gates (Phase B-pre-b):
    1. Kill switch check
    2. Per-origin handshake cooldown (IP block)
    3. API key authentication
    4. Origin header validation
    5. SEC-F19 D4 admission: stack-global + per-identity connection caps
    6. Per-connection leaky bucket rate limiting on commands
    7. Bidirectional idle timeout
    """
    settings = get_settings()
    client_ip = _get_client_ip(websocket)

    if not await _check_handshake_gates(websocket, settings, client_ip):
        return

    ws_manager = getattr(websocket.app.state, "ws_manager", None)

    # SEC-F19 D4: reserve a stack-global + per-identity admission slot before
    # accepting. /ws/control accepts directly (it is not broadcast-eligible via
    # the manager's active set) so it does not pass through ``connect*`` —
    # ``try_admit`` is its admission gate, keyed per-identity on the API-key
    # token hash. On over-cap the socket is closed with 1013 by try_admit.
    control_identity = ws_identity_key(websocket)
    if ws_manager is not None and not await ws_manager.try_admit(websocket, endpoint="control", identity=control_identity):
        return

    try:
        await _run_control_session(websocket, settings, ws_manager, client_ip)
    finally:
        # SEC-F19 D4: release the admission slot on every disconnect path
        # (including an exception in accept/session), exactly once per admit.
        if ws_manager is not None:
            await ws_manager.release_admission(identity=control_identity)


async def _run_control_session(websocket: WebSocket, settings, ws_manager, client_ip: str) -> None:
    """Accept an already-admitted /ws/control connection and run its session.

    Split out of :func:`control_stream_handler` so the admission reserve/release
    (SEC-F19 D4) wraps the whole session in a single outer try/finally while
    keeping each function's cyclomatic complexity within budget.
    """
    lifecycle = getattr(websocket.app.state, "lifecycle", None)

    await websocket.accept()
    await websocket.send_json(
        {
            "type": "connection_established",
            "data": {"channel": "control"},
        }
    )
    # OBS-WIRE-02 (Q3): per-endpoint ``connections_active{endpoint="control"}``
    # bookkeeping. Register *after* successful accept so the gauge
    # reflects broadcast-eligible connections; the matching unregister
    # in ``finally`` re-emits on every disconnect path including
    # exceptions.
    if ws_manager is not None:
        ws_manager.register_endpoint_connection(websocket, "control")

    bucket = LeakyBucket(
        capacity=settings.ws_control_rate_limit_per_sec,
        refill_rate=float(settings.ws_control_rate_limit_per_sec),
    )

    idle_timeout = settings.ws_control_idle_timeout_sec

    # Phase F: heartbeat settings from app.state.settings (testable)
    app_settings = getattr(websocket.app.state, "settings", None)
    hb_interval = getattr(app_settings, "ws_heartbeat_interval_sec", 30) if app_settings else 30
    hb_timeout = getattr(app_settings, "ws_heartbeat_pong_timeout_sec", 10) if app_settings else 10

    pong_received = asyncio.Event()
    pong_received.set()  # No outstanding ping at start

    ping_task = asyncio.create_task(_control_ping_loop(websocket, client_ip, hb_interval, hb_timeout, pong_received))

    try:
        await _control_recv_loop(websocket, lifecycle, bucket, pong_received, idle_timeout, client_ip)
    except WebSocketDisconnect:
        pass
    finally:
        ping_task.cancel()
        try:
            await ping_task
        except asyncio.CancelledError:
            pass
        # OBS-WIRE-02 (Q3): always re-emit the gauge on disconnect.
        if ws_manager is not None:
            ws_manager.unregister_endpoint_connection(websocket)


def _execute_command(lifecycle, command: str, params: dict = None) -> dict:
    """Execute a training control command.

    Args:
        lifecycle: TrainingLifecycleManager instance
        command: Command name
        params: Optional parameters

    Returns:
        Command result dictionary
    """
    if command == "start":
        return lifecycle.start_training()
    elif command == "stop":
        return lifecycle.stop_training()
    elif command == "pause":
        return lifecycle.pause_training()
    elif command == "resume":
        return lifecycle.resume_training()
    elif command == "reset":
        return lifecycle.reset()
    elif command == "set_params":
        if not params:
            raise ValueError("set_params requires a 'params' dict")
        # SEC-F10 (HO-5): validate the raw JSON ``params`` through the same
        # Pydantic model the REST ``PATCH /v1/training/params`` route uses, so
        # the WebSocket runtime-update path enforces identical numeric bounds.
        # Without this, an out-of-range value (e.g. ``max_hidden_units=999999999``)
        # reached ``update_params`` unchecked — the downstream key-whitelist +
        # candidate-pool guard never range-checks scalar fields. A
        # ``ValidationError`` propagates to ``_handle_command_message``, which
        # returns a clean error ack (the WS analogue of the REST 422) without crashing.
        validated = TrainingParamUpdateRequest(**params)
        return lifecycle.update_params(validated.model_dump(exclude_none=True))
    else:
        raise ValueError(f"Unhandled command: {command}")
