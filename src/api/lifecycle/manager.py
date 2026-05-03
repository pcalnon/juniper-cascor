"""Training lifecycle manager — central coordinator for CasCor training.

Wraps CascadeCorrelationNetwork with:
- Thread-safe training via ThreadPoolExecutor
- State machine for deterministic control flow
- Monitoring hooks for real-time metrics
- Topology and statistics extraction
"""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from api.lifecycle.monitor import TrainingMonitor, TrainingState
from api.lifecycle.state_machine import Command, TrainingPhase, TrainingStateMachine
from cascor_constants.constants_api import _PROJECT_API_DRAIN_THREAD_JOIN_TIMEOUT, _PROJECT_API_LIFECYCLE_DEFAULT_CANDIDATE_PATIENCE, _PROJECT_API_LIFECYCLE_DEFAULT_EPOCHS_MAX, _PROJECT_API_LIFECYCLE_DEFAULT_MAX_HIDDEN_UNITS, _PROJECT_API_LIFECYCLE_DEFAULT_MAX_ITERATIONS, _PROJECT_API_NETWORK_INPUT_SIZE_DEFAULT, _PROJECT_API_NETWORK_LEARNING_RATE_DEFAULT, _PROJECT_API_NETWORK_OUTPUT_SIZE_DEFAULT, _PROJECT_API_PROGRESS_QUEUE_GET_TIMEOUT, _PROJECT_API_PROGRESS_QUEUE_WAIT_TIMEOUT


def _read_optimizer_type(network: Any) -> str:
    """CAN-010 / ENH-006 (A-2): read ``optimizer_type`` through the nested
    ``config.optimizer_config`` path. Falls back to ``"Adam"`` if the chain
    is missing — same default as ``OptimizerConfig`` itself."""
    config = getattr(network, "config", None)
    optimizer_config = getattr(config, "optimizer_config", None) if config is not None else None
    return getattr(optimizer_config, "optimizer_type", "Adam") if optimizer_config is not None else "Adam"


def _write_optimizer_type(network: Any, value: str) -> None:
    """CAN-010 / ENH-006 (A-2): set ``optimizer_type`` through the nested
    ``config.optimizer_config`` path. Used by ``update_params`` so the
    setattr-on-network pattern in ``updatable_keys`` works for this nested
    field. Raises if the chain is missing — matches the contract of the
    other setters."""
    network.config.optimizer_config.optimizer_type = value


def _read_activation_function_name(network: Any) -> str:
    """CAN-011 (A-3): read ``activation_function_name`` from the network.
    Falls back to ``"Tanh"`` (matches ``_init_activation_function``'s
    fallback in ``cascade_correlation.py``) when the attribute is missing."""
    return getattr(network, "activation_function_name", "Tanh") or "Tanh"


def _write_activation_function_name(network: Any, value: str) -> None:
    """CAN-011 (A-3): swap ``activation_function_name`` and re-run
    ``_init_activation_function`` so ``activation_fn`` / ``activation_fn_no_diff``
    pick up the new mapping from the registry. Without the re-init the
    surface attribute would change but the network would keep computing the
    old activation. Existing cascaded units retain whatever activation they
    were trained with — this only affects future cascade growth and the
    output-layer activation chain."""
    network.config.activation_function_name = value
    network._init_activation_function()


class _ReplaySession:
    """CAN-015c (Phase 6E Sprint B B-3): per-snapshot replay session.

    Holds the playback state for a single replay run — current time
    index, speed (with sign for direction), pause flag, range
    sub-window — plus the background thread that ticks while playing
    and emits synthetic ``epoch_end`` events from the loaded network's
    history arrays.

    V1 scope (per ``notes/PHASE_6E_SPRINT_B_DESIGN.md`` §2.2 / §10.1):
    metric arrays + topology evolution metadata only. Per-epoch weight
    history (decision boundary playback, per-unit weight evolution) is
    deferred to CAN-015g — would require a snapshot-format extension.

    The thread emits via ``monitor._trigger_callbacks`` directly rather
    than ``monitor.on_epoch_end`` so synthetic frames don't pollute the
    live ``metrics_buffer``. Subscribers (WS broadcasters, canopy
    metrics-curve renderer) receive the events identically — only the
    replay-buffer side-effect differs.
    """

    # Allowed speed range. 0 ≡ pause, sign carries direction, magnitude
    # caps at 10× to avoid pathological CPU usage on very long
    # snapshots. Values beyond the range are clamped at /control time.
    _MIN_SPEED: float = -10.0
    _MAX_SPEED: float = 10.0
    _MIN_NONZERO_MAG: float = 0.1
    # Cap inter-frame sleeps so /pause / /seek wake up promptly even
    # at very low speeds.
    _MAX_TICK_SLEEP: float = 0.5

    def __init__(self, snapshot_id: str, history: Dict[str, list], monitor) -> None:
        self.snapshot_id = snapshot_id
        # Pre-extract the history arrays we know about so the loop
        # doesn't re-fetch every tick. Stored as plain lists (not
        # references to network.history) so a future Restore-while-
        # somehow-still-replaying race doesn't mutate them under us.
        self._history: Dict[str, list] = {key: list(history.get(key, [])) for key in ("train_loss", "value_loss", "train_accuracy", "value_accuracy")}
        # Length is the longest known array. Time index is bounded
        # exclusively by ``self.length`` (i.e. valid indices are
        # ``[0, length-1]``). Empty histories produce length=0 and the
        # loop correctly idles.
        self.length: int = max((len(v) for v in self._history.values()), default=0)
        self._monitor = monitor
        # Playback state — guarded by the lock for cross-thread reads.
        self._lock = threading.Lock()
        self.time_index: int = 0
        self.speed: float = 1.0
        self.paused: bool = True
        self.range_start: int = 0
        self.range_end: int = self.length  # exclusive
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)

    def start_thread(self) -> None:
        """Start the playback driver thread. Idempotent (a session can
        be re-started after pause without spawning a new thread)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name=f"replay-{self.snapshot_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the playback thread to exit and wait briefly for it
        to drain. Safe to call from any thread including the lifecycle
        shutdown path."""
        self._stop_event.set()
        self._wake_event.set()  # break out of any pending wait
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def play(self) -> None:
        with self._lock:
            self.paused = False
        self._wake_event.set()

    def pause(self) -> None:
        with self._lock:
            self.paused = True
        self._wake_event.set()

    def seek(self, target: int) -> int:
        """Jump to a specific time index. Returns the actual landed
        position after clamping to the active range."""
        with self._lock:
            self.time_index = self._clamp_to_range(target)
            landed = self.time_index
        self._wake_event.set()
        # Emit the seek-target frame immediately so canopy gets visual
        # feedback even if we're paused.
        self._emit_frame(landed)
        return landed

    def set_speed(self, value: float) -> float:
        """Set playback speed. ``0`` is treated as pause. Returns the
        effective (clamped) speed."""
        # Clamp magnitude to [_MIN_NONZERO_MAG, _MAX_SPEED] preserving
        # sign; treat tiny magnitudes as 0 (pause).
        if abs(value) < self._MIN_NONZERO_MAG:
            value = 0.0
        elif value > 0:
            value = min(value, self._MAX_SPEED)
        else:
            value = max(value, self._MIN_SPEED)
        with self._lock:
            self.speed = value
            # speed=0 is functionally pause; surface the flag too so
            # /play later doesn't have to also call /speed.
            if value == 0.0:
                self.paused = True
        self._wake_event.set()
        return value

    def set_range(self, start: int, end: int) -> Dict[str, int]:
        """Restrict playback to ``[start, end)``. End may be at most
        ``self.length``. Time index is re-clamped if it's now outside
        the new range. Returns the resulting range as a dict."""
        with self._lock:
            self.range_start = max(0, min(start, self.length))
            self.range_end = max(self.range_start, min(end, self.length))
            self.time_index = self._clamp_to_range(self.time_index)
            result = {"start": self.range_start, "end": self.range_end, "time_index": self.time_index}
        self._wake_event.set()
        return result

    def state_summary(self) -> Dict[str, Any]:
        """Snapshot of the current session state for the route response."""
        with self._lock:
            return {
                "snapshot_id": self.snapshot_id,
                "length": self.length,
                "time_index": self.time_index,
                "speed": self.speed,
                "paused": self.paused,
                "range": {"start": self.range_start, "end": self.range_end},
            }

    def _clamp_to_range(self, index: int) -> int:
        if self.range_end <= self.range_start:
            return self.range_start
        return max(self.range_start, min(self.range_end - 1, index))

    def _emit_frame(self, index: int) -> None:
        """Emit a synthetic ``epoch_end`` event for the given index.

        Bypasses ``monitor.on_epoch_end`` (which would write to
        ``metrics_buffer``) and calls ``_trigger_callbacks`` directly
        so the WS broadcasters fire but live training state stays
        untouched. Per the design's read-only-history guarantee."""
        if self._monitor is None:
            return
        if index < 0 or index >= self.length:
            return

        def _series_at(key: str):
            series = self._history.get(key, [])
            return series[index] if index < len(series) else None

        metrics = {
            "epoch": index + 1,  # 1-indexed for canopy display, matches on_epoch_end
            "loss": _series_at("train_loss"),
            "accuracy": _series_at("train_accuracy"),
            "validation_loss": _series_at("value_loss"),
            "validation_accuracy": _series_at("value_accuracy"),
            "phase": "Replay",
            "replay": True,  # marker so subscribers can distinguish synthetic frames
            "snapshot_id": self.snapshot_id,
        }
        try:
            self._monitor._trigger_callbacks(
                "epoch_end",
                metrics=metrics,
                epoch=metrics["epoch"],
                loss=metrics["loss"],
                accuracy=metrics["accuracy"],
            )
        except Exception:
            # Best-effort emission — a subscriber that raises mustn't
            # crash the playback thread.
            self.logger.exception("replay session: synthetic _trigger_callbacks raised")

    def _run(self) -> None:
        """Background thread driver. Sleeps for ``1/abs(speed)`` between
        frames while playing, polls every ``_MAX_TICK_SLEEP`` while
        paused. Wake-event short-circuits any wait so /pause / /seek /
        /speed take effect immediately."""
        # Emit an initial frame on session start so subscribers see
        # the entry point (epoch 0) before any /play.
        self._emit_frame(0)
        while not self._stop_event.is_set():
            with self._lock:
                paused = self.paused
                speed = self.speed
                time_index = self.time_index
                range_start = self.range_start
                range_end = self.range_end
            if paused or abs(speed) < self._MIN_NONZERO_MAG:
                # Idle until woken up — by /play, /seek, /speed change,
                # or /stop. Bounded wait so /stop_event from a separate
                # call still terminates the thread promptly.
                if self._wake_event.wait(self._MAX_TICK_SLEEP):
                    self._wake_event.clear()
                continue
            # Compute sleep duration for this frame. Bounded above so
            # very low speeds still yield to wake events promptly.
            sleep_s = min(1.0 / abs(speed), self._MAX_TICK_SLEEP)
            if self._wake_event.wait(sleep_s):
                self._wake_event.clear()
                continue
            # Advance the time index respecting direction and range.
            with self._lock:
                step = 1 if speed > 0 else -1
                new_index = time_index + step
                if new_index < range_start or new_index >= range_end:
                    # Reached a boundary — auto-pause at the edge.
                    self.paused = True
                    self.time_index = self._clamp_to_range(new_index)
                    landed = self.time_index
                else:
                    self.time_index = new_index
                    landed = new_index
            self._emit_frame(landed)


class TrainingLifecycleManager:
    """Central coordinator for CasCor network training lifecycle.

    Manages network creation, training execution (async via ThreadPoolExecutor),
    monitoring hooks, state tracking, and metrics collection.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Core components
        self.network = None
        self.state_machine = TrainingStateMachine()
        self.training_state = TrainingState()
        self.training_monitor = TrainingMonitor()

        # Threading
        self._training_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._topology_lock = threading.Lock()
        # CONC-02 / BUG-CC-16 (Phase 3B): guard the broadcast throttle so two
        # callers cannot both pass the (now - last) < interval check and emit
        # duplicate state messages. _last_state_broadcast_time is initialized
        # here so the read in _broadcast_training_state never has to gate on
        # hasattr() (which itself was part of the original race window).
        self._broadcast_lock = threading.Lock()
        self._last_state_broadcast_time: float = 0.0
        self._executor: Optional[ThreadPoolExecutor] = None
        self._training_future: Optional[Future] = None
        self._stop_requested = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially
        self._last_emitted_history_len = 0

        # Monkey-patched originals
        self._original_methods: Dict[str, Callable] = {}
        self._monitoring_active = False

        # Training data
        self._train_x: Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None
        self._val_x: Optional[torch.Tensor] = None
        self._val_y: Optional[torch.Tensor] = None

        # Network creation params (for reset)
        self._network_params: Optional[Dict[str, Any]] = None

        # WebSocket manager (set via set_ws_manager)
        self._ws_manager = None
        self._state_throttle_interval: float = 1.0  # seconds, configurable via set_ws_manager

        # Worker coordinator (set via set_worker_coordinator)
        self._worker_coordinator = None

        # METRICS-MON R1.2 / seed-03: liveness heartbeat. A 1-second daemon
        # bumps ``_liveness_counter`` and ``_liveness_last_tick_at``; the
        # liveness probe consults ``is_alive()`` to detect a wedged process.
        # The TrainingMonitor callbacks also bump the counter so progress
        # in the training thread is an additional liveness signal.
        self._liveness_counter: int = 0
        self._liveness_last_tick_at: float = time.monotonic()
        self._liveness_lock = threading.Lock()
        self._liveness_stop_event = threading.Event()
        self._liveness_thread: Optional[threading.Thread] = None
        self._start_liveness_thread()
        self._register_liveness_monitor_callbacks()

        # CAS-006 (Phase 6E Sprint A-4): auto-snap-best.
        # Hooks ``training_monitor.epoch_end`` and saves a snapshot every
        # time (validation) accuracy beats the best-seen-so-far for the
        # current run. Defaults: feature off, 50-epoch warmup. Both are
        # exposed via TrainingParams + TrainingParamUpdateRequest and
        # included in ``updatable_keys`` so users can toggle mid-run.
        self._auto_snap_best: bool = False
        self._auto_snap_min_epochs: int = 50
        self._auto_snap_best_metric: Optional[float] = None
        self._auto_snap_lock = threading.Lock()

        # CAN-015c (Phase 6E Sprint B B-3): replay session. ``None``
        # outside of an active replay; an ``_ReplaySession`` instance
        # while ``state_machine.is_replaying()``. The route layer reads
        # the session state for /control responses and dispatches
        # play/pause/seek/speed/range/stop into it.
        self._replay_session: Optional["_ReplaySession"] = None

        # CAN-015b (Phase 6E Sprint B B-2): resume-from-snapshot marker.
        # ``resume_from_snapshot`` sets this to the snapshot's terminal
        # epoch count so canopy can render a visual boundary in the
        # metrics-curve component (a vertical line separating the
        # pre-resume read-only history from the new training that
        # appends past it). Cleared once consumed by ``start_training``
        # (so a subsequent run from the same snapshot doesn't mistakenly
        # carry over the marker).
        self._resume_point_epoch: Optional[int] = None
        self.training_monitor.register_callback("epoch_end", self._maybe_auto_snap_callback)

        self.logger.info("TrainingLifecycleManager initialized")

    def _maybe_auto_snap_callback(self, metrics=None, epoch=None, loss=None, accuracy=None, **_kwargs) -> None:
        """CAS-006 (A-4): epoch_end callback that saves a snapshot when the
        current (validation) accuracy beats the best-seen-so-far for the
        current run. No-op when the feature is disabled, when the warmup
        threshold has not been reached, or when no usable accuracy metric
        is available.

        Tracks the best metric on the lifecycle (not the network) because
        a single network instance can be used across multiple training
        runs; ``start_training`` resets the tracker so each run starts
        fresh. Prefers ``validation_accuracy`` over ``accuracy`` so the
        snapshot reflects generalization rather than training-set fit.
        """
        with self._auto_snap_lock:
            if not self._auto_snap_best:
                return
            if epoch is None or epoch < self._auto_snap_min_epochs:
                return
            current = None
            if isinstance(metrics, dict):
                current = metrics.get("validation_accuracy")
            if current is None:
                current = accuracy
            if current is None:
                return
            best = self._auto_snap_best_metric
            if best is not None and current <= best:
                return
            self._auto_snap_best_metric = current
            description = f"auto_snap_best epoch={epoch} accuracy={current:.6f}"
        # Save outside the auto_snap_lock so a slow filesystem doesn't
        # serialize the next epoch_end callback. ``save_snapshot`` has
        # its own internal failure handling.
        try:
            self.save_snapshot(description=description)
        except Exception:
            self.logger.exception("auto_snap_best: save_snapshot failed (epoch=%s, accuracy=%s)", epoch, current)

    def _register_liveness_monitor_callbacks(self) -> None:
        """Bump the heartbeat from every training-monitor event so progress
        in the training thread is an additional liveness signal.
        """
        bump = lambda **_kw: self.bump_liveness()  # noqa: E731 — concise wrapper
        for event in ("epoch_start", "epoch_end", "cascade_add", "training_start", "training_end", "topology_change", "candidate_progress", "phase_change"):
            self.training_monitor.register_callback(event, bump)

    def bump_liveness(self) -> None:
        """Record that the lifecycle is making forward progress.

        Called from the 1-second daemon thread and from TrainingMonitor
        event callbacks. The probe layer reads the resulting timestamp
        via ``is_alive()`` to decide liveness.
        """
        with self._liveness_lock:
            self._liveness_counter += 1
            self._liveness_last_tick_at = time.monotonic()

    def is_alive(self, stale_after_seconds: float = 30.0) -> bool:
        """Return True if the heartbeat has been bumped within the window.

        ``stale_after_seconds`` defaults to 30 s — well above the daemon
        thread's 1-second cadence, so transient scheduling jitter does
        not flap liveness, but well below typical Helm
        ``failureThreshold`` × ``periodSeconds`` so real wedges still
        get caught.
        """
        with self._liveness_lock:
            last = self._liveness_last_tick_at
        return (time.monotonic() - last) < stale_after_seconds

    def _start_liveness_thread(self) -> None:
        """Start the 1-second daemon that bumps the heartbeat."""

        def _loop() -> None:
            while not self._liveness_stop_event.is_set():
                self.bump_liveness()
                self._liveness_stop_event.wait(1.0)

        self._liveness_thread = threading.Thread(
            target=_loop,
            name="lifecycle-liveness",
            daemon=True,
        )
        self._liveness_thread.start()

    def stop_liveness_heartbeat(self) -> None:
        """Stop the heartbeat thread (used in shutdown / tests)."""
        self._liveness_stop_event.set()
        if self._liveness_thread is not None:
            self._liveness_thread.join(timeout=2.0)

    def set_ws_manager(self, ws_manager, state_throttle_interval: float = 1.0) -> None:
        """Set the WebSocket manager for real-time broadcasting.

        Registers monitor callbacks that broadcast metrics/events via WebSocket.

        Args:
            ws_manager: WebSocketManager instance.
            state_throttle_interval: Minimum interval in seconds between
                non-terminal state broadcasts (GAP-WS-21 coalescer).
        """
        self._ws_manager = ws_manager
        self._state_throttle_interval = state_throttle_interval
        self._register_ws_callbacks()

    def set_worker_coordinator(self, coordinator) -> None:
        """Set the worker coordinator for remote WebSocket worker dispatch.

        When a coordinator is set, newly created networks will have it injected
        so they can dispatch candidate training tasks to remote workers.
        If a network already exists, the coordinator is injected immediately.
        """
        self._worker_coordinator = coordinator
        if self.network is not None and hasattr(self.network, "set_worker_coordinator"):
            self.network.set_worker_coordinator(coordinator)
            self.logger.info("Worker coordinator injected into existing network")

    def _register_ws_callbacks(self) -> None:
        """Register WebSocket broadcast callbacks on the training monitor."""
        if self._ws_manager is None:
            return

        from api.websocket.messages import create_candidate_progress_message, create_cascade_add_message, create_event_message, create_metrics_message

        ws = self._ws_manager

        self.training_monitor.register_callback(
            "epoch_end",
            lambda metrics, **kw: ws.broadcast_from_thread(create_metrics_message(metrics)),
        )
        self.training_monitor.register_callback(
            "cascade_add",
            lambda event, **kw: ws.broadcast_from_thread(create_cascade_add_message(event)),
        )
        self.training_monitor.register_callback(
            "training_start",
            lambda **kw: self._broadcast_training_state(force=True),
        )
        self.training_monitor.register_callback(
            "training_end",
            lambda **kw: ws.broadcast_from_thread(create_event_message({"event": "training_complete"})),
        )
        self.training_monitor.register_callback(
            "candidate_progress",
            lambda progress, **kw: ws.broadcast_from_thread(create_candidate_progress_message(progress)),
        )

        self.logger.info("WebSocket broadcast callbacks registered")

    # Terminal statuses that must always bypass the broadcast throttle (GAP-WS-21)
    _TERMINAL_STATUSES = frozenset({"Completed", "Failed", "Stopped"})

    def _broadcast_training_state(self, force: bool = False) -> None:
        """Broadcast full training state via WebSocket.

        Uses a terminal-aware debounced coalescer (GAP-WS-21):
        - Terminal transitions (Completed/Failed/Stopped) always bypass throttle
        - force=True always bypasses throttle
        - Non-terminal transitions throttled to at most once per coalesce interval
        """
        if self._ws_manager is None:
            return

        state_data = self.training_state.get_state()
        status = state_data.get("status", "")
        is_terminal = status in self._TERMINAL_STATUSES

        # CONC-02 / BUG-CC-16 (Phase 3B): the throttle is a check-then-set on
        # _last_state_broadcast_time. Without a lock two threads (training
        # thread, monitor thread, control endpoint) could both observe
        # `now - last >= interval`, both pass, and both broadcast — defeating
        # the GAP-WS-21 coalescer. Hold _broadcast_lock across the read and
        # the write so only one caller wins the throttle window. Terminal
        # transitions and force=True still bypass the throttle but still
        # update the timestamp under the lock for consistency.
        with self._broadcast_lock:
            now = time.monotonic()
            if not force and not is_terminal:
                if now - self._last_state_broadcast_time < self._state_throttle_interval:
                    return
            self._last_state_broadcast_time = now

        from api.websocket.messages import create_state_message

        self._ws_manager.broadcast_from_thread(create_state_message(state_data))

    # ------------------------------------------------------------------
    # Network management
    # ------------------------------------------------------------------

    def create_network(self, **kwargs) -> Dict[str, Any]:
        """Create a new CascadeCorrelationNetwork.

        Args:
            **kwargs: Parameters passed to CascadeCorrelationConfig.create_simple_config()

        Returns:
            Network info dictionary
        """
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

        with self._training_lock:
            if self.state_machine.is_started():
                raise RuntimeError("Cannot create network while training is active")

            self._network_params = kwargs.copy()
            config = CascadeCorrelationConfig.create_simple_config(**kwargs)
            self.network = CascadeCorrelationNetwork(config=config)
            self._install_monitoring_hooks()

            # Inject worker coordinator for remote dispatch if available
            if self._worker_coordinator is not None and hasattr(self.network, "set_worker_coordinator"):
                self.network.set_worker_coordinator(self._worker_coordinator)

            self.training_state.update_state(
                status="Stopped",
                phase="Idle",
                learning_rate=kwargs.get("learning_rate", _PROJECT_API_NETWORK_LEARNING_RATE_DEFAULT),
                max_hidden_units=kwargs.get("max_hidden_units", _PROJECT_API_LIFECYCLE_DEFAULT_MAX_HIDDEN_UNITS),
                max_epochs=kwargs.get("epochs_max", _PROJECT_API_LIFECYCLE_DEFAULT_EPOCHS_MAX),
                max_iterations=kwargs.get("max_iterations", _PROJECT_API_LIFECYCLE_DEFAULT_MAX_ITERATIONS),
                network_name=f"CasCor-{kwargs.get('input_size', _PROJECT_API_NETWORK_INPUT_SIZE_DEFAULT)}x{kwargs.get('output_size', _PROJECT_API_NETWORK_OUTPUT_SIZE_DEFAULT)}",
            )

            info = self.get_network_info()
            self.logger.info(f"Network created: {info['input_size']}x{info['output_size']}")
            return info

    def delete_network(self) -> None:
        """Delete the current network."""
        with self._training_lock:
            if self.state_machine.is_started():
                raise RuntimeError("Cannot delete network while training is active")
            self._restore_original_methods()
            self.network = None
            self._network_params = None
            self.state_machine.handle_command(Command.RESET)
            self.training_state.update_state(status="Stopped", phase="Idle")
            self.logger.info("Network deleted")

    def has_network(self) -> bool:
        return self.network is not None

    def get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        if self.network is None:
            return {}
        return {
            "input_size": self.network.input_size,
            "output_size": self.network.output_size,
            "hidden_units": len(self.network.hidden_units),
            "max_hidden_units": getattr(self.network, "max_hidden_units", 0),
            "learning_rate": getattr(self.network, "learning_rate", 0.0),
            "uuid": str(getattr(self.network, "uuid", "")),
        }

    # ------------------------------------------------------------------
    # Monitoring hooks (monkey-patch approach from CascorIntegration)
    # ------------------------------------------------------------------

    def _install_monitoring_hooks(self) -> None:
        """Install monitoring hooks on the network via monkey-patching.

        Hooks:
        - fit(): Wraps the top-level training call with start/end tracking
        - validate_training(): Wraps per-iteration validation for metrics emission
        - grow_network(): Wraps cascade growth for cascade_add events and phase tracking
        """
        if self.network is None or self._monitoring_active:
            return

        original_fit = self.network.fit
        self._original_methods["fit"] = original_fit

        monitor = self.training_monitor
        state = self.training_state
        stop_event = self._stop_requested
        sm = self.state_machine
        manager_ref = self

        def _output_training_callback(epoch, epochs, loss):
            monitor.on_epoch_end(
                epoch=epoch,
                loss=loss,
                accuracy=None,
                learning_rate=getattr(manager_ref.network, "learning_rate", 0.0),
                hidden_units=len(manager_ref.network.hidden_units),
            )
            state.update_state(
                current_epoch=epoch,
                phase_detail="training_output",
            )

        def monitored_fit(x, y, x_val=None, y_val=None, **kwargs):
            manager_ref._last_emitted_history_len = 0
            monitor.on_training_start()
            # BUG-CC-07: phase is updated via state-machine wrapper, not manually.
            sm.handle_command(Command.START)
            monitor.on_phase_change(sm.phase.name.lower())
            state.update_state(status="Started", phase="Output", phase_started_at=datetime.now().isoformat())
            manager_ref._broadcast_training_state(force=True)

            # Inject output training callback (Approach B — attribute fallback)
            manager_ref.network._output_epoch_callback = _output_training_callback

            try:
                result = original_fit(x, y, x_val=x_val, y_val=y_val, **kwargs)

                # Extract any remaining metrics after fit completes
                manager_ref._extract_and_record_metrics()

                if stop_event.is_set():
                    sm.handle_command(Command.STOP)
                    state.update_state(status="Stopped", phase="Idle")
                    manager_ref._broadcast_training_state(force=True)
                else:
                    sm.mark_completed()
                    state.update_state(status="Completed", phase="Idle")
                    manager_ref._broadcast_training_state(force=True)

                return result
            except Exception as e:
                sm.mark_failed(str(e))
                state.update_state(status="Failed", phase="Idle")
                manager_ref._broadcast_training_state(force=True)
                raise
            finally:
                monitor.on_training_end()

        self.network.fit = monitored_fit

        # Hook validate_training for per-iteration metrics emission.
        # validate_training is called at the end of each grow_network iteration,
        # AFTER _retrain_output_layer appends train_loss, _calculate_train_accuracy
        # appends train_accuracy, and validate_training itself appends value_loss
        # and value_accuracy. This is the correct point to extract metrics.
        if hasattr(self.network, "validate_training"):
            original_validate = self.network.validate_training
            self._original_methods["validate_training"] = original_validate

            def monitored_validate(*args, **kwargs):
                result = original_validate(*args, **kwargs)
                manager_ref._extract_and_record_metrics()
                return result

            self.network.validate_training = monitored_validate

        # Hook grow_network for cascade_add events and phase tracking
        self._install_grow_network_hook(monitor, state, sm, manager_ref)

        # BUG-CC-07: wrap state machine set_phase to notify monitor
        self._install_phase_tracker(monitor, sm)

        self._monitoring_active = True
        self.logger.info("Monitoring hooks installed")

    def _install_phase_tracker(self, monitor, sm) -> None:
        """Wrap TrainingStateMachine.set_phase to notify the monitor (BUG-CC-07)."""
        # Avoid re-wrapping on reinstall.
        if getattr(self, "_original_set_phase", None) is not None:
            return
        original_set_phase = sm.set_phase
        self._original_set_phase = original_set_phase

        def tracked_set_phase(phase):
            original_set_phase(phase)
            phase_name = phase.name.lower() if hasattr(phase, "name") else str(phase)
            monitor.on_phase_change(phase_name)

        sm.set_phase = tracked_set_phase

    @staticmethod
    def _drain_progress_queue(network_ref, stop_event, state, monitor, manager_ref):
        """Background thread that reads candidate progress from workers.

        Uses deferred queue discovery: the persistent progress queue is created
        lazily inside grow_network() -> _ensure_worker_pool(), so it may not
        exist when this thread starts. We poll for it until it appears or the
        stop event is set.
        """
        import queue as _queue_mod

        _pq = None
        while not stop_event.is_set():
            # Deferred discovery — queue is created inside grow_network
            if _pq is None:
                _pq = getattr(network_ref, "_persistent_progress_queue", None)
                if _pq is None:
                    try:
                        stop_event.wait(timeout=_PROJECT_API_PROGRESS_QUEUE_WAIT_TIMEOUT)
                    except Exception:
                        break
                    continue
            try:
                progress = _pq.get(timeout=_PROJECT_API_PROGRESS_QUEUE_GET_TIMEOUT)
            except _queue_mod.Empty:
                continue
            except Exception:
                break
            state.update_state(
                phase_detail="training_candidates",
                candidate_epoch=progress.get("epoch", 0),
                candidate_total_epochs=progress.get("total_epochs", 0),
                best_correlation=progress.get("correlation", 0.0),
            )
            monitor.on_candidate_progress(progress)
            manager_ref._broadcast_training_state()

    def _install_grow_network_hook(self, monitor, state, sm, manager_ref) -> None:
        """Install monitoring hook on grow_network for cascade_add events and phase tracking."""
        if not hasattr(self.network, "grow_network"):
            return

        original_grow = self.network.grow_network
        self._original_methods["grow_network"] = original_grow

        def _grow_iteration_callback(iteration, max_iterations, best_correlation, candidates_trained, candidates_total, phase_detail, **kwargs):
            state.update_state(
                grow_iteration=iteration,
                grow_max=max_iterations,
                best_correlation=best_correlation,
                candidates_trained=candidates_trained,
                candidates_total=candidates_total,
                phase_detail=phase_detail,
                best_candidate_id=kwargs.get("best_candidate_id", -1),
                best_candidate_uuid=kwargs.get("best_candidate_uuid", ""),
                second_candidate_id=kwargs.get("second_candidate_id"),
                second_candidate_correlation=kwargs.get("second_candidate_correlation", 0.0),
                all_correlations=kwargs.get("all_correlations", []),
            )
            manager_ref._broadcast_training_state()

        def monitored_grow(*args, **kwargs):
            # Pre-call: capture initial output training metrics
            # (appended by fit() between train_output_layer() and grow_network())
            manager_ref._extract_and_record_metrics()

            prev_hidden = len(manager_ref.network.hidden_units)
            # BUG-CC-07: phase is updated via state-machine wrapper, not manually.
            sm.set_phase(TrainingPhase.CANDIDATE)
            state.update_state(phase="Candidate", phase_started_at=datetime.now().isoformat())
            manager_ref._broadcast_training_state(force=True)

            # Inject grow iteration callback (Approach B — attribute fallback)
            manager_ref.network._grow_iteration_callback = _grow_iteration_callback

            # Start drain thread for candidate progress from worker pool.
            # Always started unconditionally — uses deferred queue discovery
            # because _persistent_progress_queue is created lazily inside
            # grow_network() → _ensure_worker_pool().
            _drain_stop = threading.Event()
            _drain_thread = threading.Thread(
                target=TrainingLifecycleManager._drain_progress_queue,
                args=(manager_ref.network, _drain_stop, state, monitor, manager_ref),
                daemon=True,
                name="candidate-progress-drain",
            )
            _drain_thread.start()

            try:
                result = original_grow(*args, **kwargs)
            finally:
                # Stop drain thread
                _drain_stop.set()
                _drain_thread.join(timeout=_PROJECT_API_DRAIN_THREAD_JOIN_TIMEOUT)

            new_hidden = len(manager_ref.network.hidden_units)

            if new_hidden > prev_hidden:
                # BUG-CC-01: Wire create_topology_message into lifecycle events
                # BUG-CC-02: Extract actual correlation from each installed hidden unit
                from api.websocket.messages import create_topology_message

                for i in range(prev_hidden, new_hidden):
                    unit = manager_ref.network.hidden_units[i]
                    actual_correlation = float(getattr(unit, "best_correlation", 0.0) or 0.0)
                    monitor.on_cascade_add(
                        hidden_unit_index=i,
                        correlation=actual_correlation,
                    )
                if manager_ref._ws_manager is not None:
                    topology_data = {
                        "hidden_units": new_hidden,
                        "input_size": getattr(manager_ref.network, "input_size", 0),
                        "output_size": getattr(manager_ref.network, "output_size", 0),
                        "event": "cascade_add",
                    }
                    manager_ref._ws_manager.broadcast_from_thread(create_topology_message(topology_data))

            # Post-call: return to output phase after grow completes
            sm.set_phase(TrainingPhase.OUTPUT)
            state.update_state(phase="Output", phase_detail="", candidate_epoch=0, candidate_total_epochs=0)
            manager_ref._broadcast_training_state(force=True)
            # Catch-all for any remaining metrics
            manager_ref._extract_and_record_metrics()
            return result

        self.network.grow_network = monitored_grow

    def _restore_original_methods(self) -> None:
        """Restore original network methods."""
        # BUG-CC-07: restore unwrapped state-machine set_phase if installed.
        original_set_phase = getattr(self, "_original_set_phase", None)
        if original_set_phase is not None:
            self.state_machine.set_phase = original_set_phase
            self._original_set_phase = None
        if not self._original_methods or self.network is None:
            self._monitoring_active = False
            return
        for method_name, original in self._original_methods.items():
            setattr(self.network, method_name, original)
        self._original_methods.clear()
        self._monitoring_active = False
        self.logger.info("Original methods restored")

    def _extract_and_record_metrics(self) -> None:
        """Extract NEW metrics from network history and record them.

        Uses a high-water-mark (_last_emitted_history_len) to only emit
        history entries that haven't been emitted yet. Safe to call
        multiple times — idempotent when no new data exists.
        """
        if self.network is None or not hasattr(self.network, "history"):
            return

        # CONC-03 / BUG-CC-17 (Phase 3B): the previous implementation
        # released self._metrics_lock between the snapshot+high-water-mark read
        # and the high-water-mark write. Two concurrent callers could both
        # observe the same `last_emitted`, both emit the slice
        # [last_emitted:current_len), and only then race on the write — so each
        # epoch in that slice was reported to TrainingMonitor twice.
        # Hold _metrics_lock across the read, the per-entry on_epoch_end calls,
        # and the high-water-mark advance so the read-process-write cycle is
        # atomic. The training_state update is idempotent and is left outside
        # the lock to keep the critical section bounded.
        with self._metrics_lock:
            try:
                history = self.network.history
                train_loss_list = list(history.get("train_loss", []))
                train_accuracy_list = list(history.get("train_accuracy", []))
                val_loss_list = list(history.get("value_loss", []))
                val_accuracy_list = list(history.get("value_accuracy", []))
                hidden_units_count = len(self.network.hidden_units)
                last_emitted = self._last_emitted_history_len
            except (RuntimeError, KeyError):
                return

            current_len = len(train_loss_list)
            if current_len <= last_emitted:
                return  # No new data

            # Emit all new entries
            for i in range(last_emitted, current_len):
                epoch = i + 1
                self.training_monitor.on_epoch_end(
                    epoch=epoch,
                    loss=train_loss_list[i],
                    accuracy=train_accuracy_list[i] if i < len(train_accuracy_list) else None,
                    learning_rate=getattr(self.network, "learning_rate", 0.0),
                    hidden_units=hidden_units_count,
                    validation_loss=val_loss_list[i] if i < len(val_loss_list) else None,
                    validation_accuracy=val_accuracy_list[i] if i < len(val_accuracy_list) else None,
                )

            # Advance the high-water-mark before releasing the lock — this is
            # the second half of the formerly-split section.
            self._last_emitted_history_len = current_len

        self.training_state.update_state(
            current_epoch=current_len,
            current_step=current_len,
        )

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    # Network.fit()'s narrow signature — anything outside this set raises
    # TypeError if passed to fit(**kwargs). TrainingParams is intentionally
    # broader (covers every runtime-tunable param), so start_training has
    # to split the request body into "fit-shaped" and "network-attribute"
    # kwargs and route them through different paths. See
    # juniper-ml/notes/CASCOR_FIT_KWARGS_LATENT_BUG.md for the full trace
    # and rationale (Option 1 — filter at the start_training boundary).
    _FIT_KWARGS: frozenset = frozenset({"max_epochs", "epochs", "max_iterations", "early_stopping"})

    def start_training(
        self,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Start training asynchronously.

        Args:
            x: Training features tensor
            y: Training targets tensor
            x_val: Validation features
            y_val: Validation targets
            **kwargs: TrainingParams body. Fields in ``_FIT_KWARGS`` are
                forwarded to ``network.fit``; everything else is applied
                in-place via ``update_params`` so the next fit pass sees
                the new values. Unknown keys (not in fit and not in
                ``update_params``' whitelist) raise immediately so a
                typo at the API boundary fails loud rather than getting
                swallowed on the background thread.

        Returns:
            Status dictionary
        """
        if self.network is None:
            raise RuntimeError("No network created")

        with self._training_lock:
            if self.state_machine.is_started():
                raise RuntimeError("Training already in progress")
            # CAN-015d (B-4): Investigating is the inspection / modification
            # mode loaded by ``/restore``. Training commands are explicitly
            # rejected — the user must invoke ``/retrain`` or ``/resume`` to
            # transition out of Investigating before starting training.
            # Failing fast at the API boundary is much clearer than
            # letting the future submit and the FSM transition fail
            # silently inside monitored_fit.
            if self.state_machine.is_investigating():
                raise RuntimeError("Cannot start training while Investigating a snapshot — invoke /v1/snapshots/{id}/retrain or /resume to transition out of Investigating first")
            # CAN-015c (B-3): Replaying is read-only playback. Same
            # rejection contract — user must /replay/control stop first.
            if self.state_machine.is_replaying():
                raise RuntimeError("Cannot start training while replaying a snapshot — invoke /v1/snapshots/{id}/replay/control with action='stop' first")

            if x is not None:
                self._train_x = x
                self._train_y = y
            if x_val is not None:
                self._val_x = x_val
                self._val_y = y_val

            if self._train_x is None or self._train_y is None:
                raise ValueError("Training data not provided")

            self._stop_requested.clear()
            self._pause_event.set()

            # CAS-006 (A-4) + CAN-015b (B-2): each training run normally
            # starts fresh — we don't want a snapshot from a previous run's
            # accuracy ceiling to suppress auto-snaps in this run. EXCEPTION:
            # when the FSM is RESUME_READY we're continuing a snapshotted
            # run, so the loaded ratchet stays as the baseline (a re-snap
            # only fires when the resumed training truly beats the prior
            # run's best). We also clear the resume marker once consumed
            # so a stop-then-restart-without-resume doesn't carry it over.
            resuming = self.state_machine.is_resume_ready()
            if not resuming:
                with self._auto_snap_lock:
                    self._auto_snap_best_metric = None
            else:
                self._resume_point_epoch = None

            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cascor-train")

            fit_kwargs = {k: v for k, v in kwargs.items() if k in self._FIT_KWARGS}
            network_kwargs = {k: v for k, v in kwargs.items() if k not in self._FIT_KWARGS and v is not None}

            # Apply network-attribute kwargs in-place BEFORE submitting the
            # training future so the next fit pass observes the new values.
            # ``_apply_params_unlocked`` shares the same whitelist + atomic-
            # rollback path as ``update_params``; calling it here while we
            # hold ``_training_lock`` avoids re-entering the non-reentrant
            # lock and avoids the race where the background thread could
            # start fit() before update_params lands.
            if network_kwargs:
                self._apply_params_unlocked(network_kwargs)

            self._training_future = self._executor.submit(self._run_training, self._train_x, self._train_y, self._val_x, self._val_y, **fit_kwargs)

        return {"status": "training_started", "timestamp": time.time()}

    def _run_training(self, x, y, x_val, y_val, **kwargs) -> None:
        """Execute training in background thread.

        Note: Exception handling (state transitions, status updates, broadcasts)
        is performed by monitored_fit() which wraps network.fit(). This method
        intentionally does not duplicate that handling (CR-007 Option C).
        """
        self.network.fit(x, y, x_val=x_val, y_val=y_val, **kwargs)

    def stop_training(self) -> Dict[str, Any]:
        """Request training stop."""
        self._stop_requested.set()
        self.state_machine.handle_command(Command.STOP)
        self.training_state.update_state(status="Stopped", phase="Idle")
        self._broadcast_training_state(force=True)
        return {"status": "stop_requested", "timestamp": time.time()}

    def pause_training(self) -> Dict[str, Any]:
        """Pause training."""
        if not self.state_machine.is_started():
            raise RuntimeError("Training is not active")
        self._pause_event.clear()
        self.state_machine.handle_command(Command.PAUSE)
        self.training_state.update_state(status="Paused")
        self._broadcast_training_state(force=True)
        return {"status": "paused", "timestamp": time.time()}

    def resume_training(self) -> Dict[str, Any]:
        """Resume paused training."""
        if not self.state_machine.is_paused():
            raise RuntimeError("Training is not paused")
        self._pause_event.set()
        self.state_machine.handle_command(Command.RESUME)
        self.training_state.update_state(status="Started")
        self._broadcast_training_state(force=True)
        return {"status": "resumed", "timestamp": time.time()}

    def reset(self) -> Dict[str, Any]:
        """Reset training state."""
        self._stop_requested.set()
        self._last_emitted_history_len = 0
        self.state_machine.handle_command(Command.RESET)
        self.training_monitor.clear_metrics()
        self.training_state.update_state(
            status="Stopped",
            phase="Idle",
            current_epoch=0,
            current_step=0,
        )
        self._broadcast_training_state(force=True)
        return {"status": "reset", "timestamp": time.time()}

    # ------------------------------------------------------------------
    # Status & metrics
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        state_summary = self.state_machine.get_state_summary()
        monitor_state = self.training_monitor.get_current_state()
        training_state = self.training_state.get_state()

        if self.network is not None:
            training_state.setdefault("input_size", getattr(self.network, "input_size", 0))
            training_state.setdefault("output_size", getattr(self.network, "output_size", 0))

        return {
            "state_machine": state_summary,
            "monitor": monitor_state,
            "training_state": training_state,
            "network_loaded": self.network is not None,
            "training_active": self.state_machine.is_started(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        if self.network is None:
            return {}
        with self._metrics_lock:
            try:
                history = self.network.history
                train_loss = list(history.get("train_loss", []))
                train_accuracy = list(history.get("train_accuracy", []))
                val_loss = list(history.get("value_loss", []))
                val_accuracy = list(history.get("value_accuracy", []))
                hidden_units = len(self.network.hidden_units)
            except (RuntimeError, KeyError):
                return {}

        return {
            "epoch": len(train_loss),
            "train_loss": train_loss[-1] if train_loss else None,
            "train_accuracy": train_accuracy[-1] if train_accuracy else None,
            "val_loss": val_loss[-1] if val_loss else None,
            "val_accuracy": val_accuracy[-1] if val_accuracy else None,
            "hidden_units": hidden_units,
            "timestamp": datetime.now().isoformat(),
        }

    def get_metrics_history(self, count: Optional[int] = None) -> list:
        """Get metrics history."""
        if count:
            return self.training_monitor.get_recent_metrics(count)
        return self.training_monitor.get_all_metrics()

    def has_training_data(self) -> bool:
        """Check if training data is loaded."""
        return self._train_x is not None and self._train_y is not None

    def get_dataset(self) -> Dict[str, Any]:
        """Return dataset metadata."""
        if self._train_x is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "train_samples": self._train_x.shape[0],
            "test_samples": self._val_x.shape[0] if self._val_x is not None else 0,
            "input_features": self._train_x.shape[1],
            "output_features": self._train_y.shape[1],
        }

    def get_dataset_data(self) -> Optional[Dict[str, Any]]:
        """Return dataset arrays for visualization."""
        if self._train_x is None:
            return None
        result = {
            "train_x": self._train_x.detach().cpu().tolist(),
            "train_y": self._train_y.detach().cpu().tolist(),
        }
        if self._val_x is not None:
            result["val_x"] = self._val_x.detach().cpu().tolist()
            result["val_y"] = self._val_y.detach().cpu().tolist()
        return result

    def get_training_params(self) -> Dict[str, Any]:
        """Get current training parameters.

        Returns every field listed in ``update_params``' ``updatable_keys`` so that
        clients reconciling UI state after a reconnect observe the live network
        values rather than falling back to stale defaults.
        """
        if self.network is None:
            return {}
        return {
            "learning_rate": getattr(self.network, "learning_rate", 0.0),
            "candidate_learning_rate": getattr(self.network, "candidate_learning_rate", 0.0),
            "max_hidden_units": getattr(self.network, "max_hidden_units", 0),
            "epochs_max": getattr(self.network, "epochs_max", 0),
            "max_iterations": getattr(self.network, "max_iterations", 0),
            "patience": getattr(self.network, "patience", 0),
            "candidate_pool_size": getattr(self.network, "candidate_pool_size", 0),
            "correlation_threshold": getattr(self.network, "correlation_threshold", 0.0),
            "convergence_threshold": getattr(self.network, "convergence_threshold", 0.001),
            "candidate_patience": getattr(self.network, "candidate_patience", _PROJECT_API_LIFECYCLE_DEFAULT_CANDIDATE_PATIENCE),
            "candidate_convergence_threshold": getattr(self.network, "candidate_convergence_threshold", 0.001),
            "candidate_epochs": getattr(self.network, "candidate_epochs", 0),
            # CAS-002 (Phase 6E Sprint A-1): per-output-training-phase budget,
            # distinct from ``epochs_max`` (the global cap).
            "output_epochs": getattr(self.network, "output_epochs", 0),
            "init_output_weights": getattr(self.network, "init_output_weights", "zero"),
            # CAN-010 / ENH-006 (Phase 6E Sprint A-2): output-layer optimizer.
            # Reads through the nested ``config.optimizer_config`` so a runtime
            # patch via ``update_params`` is reflected here on the next GET.
            "optimizer_type": _read_optimizer_type(self.network),
            # CAN-011 (Phase 6E Sprint A-3): hidden-unit activation function.
            "activation_function_name": _read_activation_function_name(self.network),
            # CAS-006 (Phase 6E Sprint A-4): auto-snap-best lifecycle flags.
            # These live on the lifecycle (not the network) so a single
            # network instance can be re-used across runs while the auto-
            # snap counter resets each ``start_training``.
            "auto_snap_best": self._auto_snap_best,
            "auto_snap_min_epochs": self._auto_snap_min_epochs,
        }

    def update_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update runtime-modifiable training parameters (thread-safe).

        Modifies the live network's attributes directly. Parameters that are
        safe to update while training is running: learning_rate,
        candidate_learning_rate, correlation_threshold, candidate_pool_size.
        Parameters effective at next cascade/epoch: max_hidden_units, epochs_max,
        patience.

        GAP-WS-28: applies all updates atomically. If any setattr raises,
        every previously-applied key is reverted to its pre-call value
        before re-raising, so the network is never left in a half-updated
        state. The ``_training_lock`` already prevents the race itself; this
        adds the all-or-nothing semantics for the case where a property
        setter rejects a value (currently no setters do, but adding a
        defensive guard now means future validation can be wired in
        without re-introducing torn writes).

        Args:
            params: Dict of parameter names and new values (None values excluded).

        Returns:
            Updated training parameters dict.

        Raises:
            ValueError: If no network exists.
            Exception: Re-raises whatever setattr raised, after rolling back
                any partially-applied updates.
        """
        with self._training_lock:

            ########################################################################################
            # Do NOT remove this commented out code block until explicit approval has been granted
            ########################################################################################
            # if self.network is None:
            #     raise ValueError("No network exists — create a network first")
            # updatable_keys = {
            #     "learning_rate",
            #     "candidate_learning_rate",
            #     "correlation_threshold",
            #     "candidate_pool_size",
            #     "max_hidden_units",
            #     "epochs_max",
            #     "max_iterations",
            #     "patience",
            #     "convergence_threshold",
            #     "candidate_convergence_threshold",
            #     "candidate_patience",
            #     "candidate_epochs",
            #     "output_epochs",  # CAS-002 (Phase 6E Sprint A-1)
            #     "init_output_weights",
            #     "optimizer_type",  # CAN-010 / ENH-006 (Phase 6E Sprint A-2) — nested setter
            #     "activation_function_name",  # CAN-011 (Phase 6E Sprint A-3) — re-init on swap
            # }
            # # Plain setattr targets — keys that map directly to network attributes.
            # # ``optimizer_type`` and ``activation_function_name`` go through
            # # special-cased setters that touch nested config / re-init paths.
            # nested_keys = {"optimizer_type", "activation_function_name"}
            # simple_keys = updatable_keys - nested_keys
            # applicable = {k: v for k, v in params.items() if k in simple_keys and hasattr(self.network, k)}
            # old_values = {k: getattr(self.network, k) for k in applicable}

            # # CAN-010 / ENH-006: ``optimizer_type`` lives at
            # # ``self.network.config.optimizer_config.optimizer_type``, not on
            # # the network directly. Treated separately so the rollback path
            # # below still works through the same revert mechanism.
            # optimizer_pending = "optimizer_type" in params and params["optimizer_type"] is not None
            # old_optimizer_type = _read_optimizer_type(self.network) if optimizer_pending else None

            # # CAN-011 (A-3): ``activation_function_name`` requires re-running
            # # ``_init_activation_function`` so ``activation_fn`` picks up the
            # # new mapping. Same revert pattern as optimizer_type.
            # activation_pending = "activation_function_name" in params and params["activation_function_name"] is not None
            # old_activation_function_name = _read_activation_function_name(self.network) if activation_pending else None

            # applied: list[str] = []
            # try:
            #     for key, value in applicable.items():
            #         setattr(self.network, key, value)
            #         applied.append(key)
            #     if optimizer_pending:
            #         _write_optimizer_type(self.network, params["optimizer_type"])
            #         applied.append("optimizer_type")
            #     if activation_pending:
            #         _write_activation_function_name(self.network, params["activation_function_name"])
            #         applied.append("activation_function_name")
            # except Exception:
            #     # GAP-WS-28: revert any partial application before propagating.
            #     # CAN-010 / ENH-006 (A-2) + CAN-011 (A-3): nested setters
            #     # need their own revert path — mirror the apply branch.
            #     for key in reversed(applied):
            #         try:
            #             if key == "optimizer_type":
            #                 _write_optimizer_type(self.network, old_optimizer_type)
            #             elif key == "activation_function_name":
            #                 _write_activation_function_name(self.network, old_activation_function_name)
            #             else:
            #                 setattr(self.network, key, old_values[key])
            #         except Exception:
            #             # If revert itself raises, log and continue rolling
            #             # back the rest — best-effort consistency.
            #             self.logger.exception("update_params rollback: revert of %s failed", key)
            #     raise
            ########################################################################################

            return self._apply_params_unlocked(params)

    def _apply_params_unlocked(self, params: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
        """Apply runtime params assuming the caller already holds ``_training_lock``.

        Internal helper extracted from ``update_params`` so that
        ``start_training`` can route TrainingParams body fields through the
        same whitelist + atomic-rollback path without re-entering the
        non-reentrant ``_training_lock`` (see CASCOR_FIT_KWARGS_LATENT_BUG.md
        for the full rationale of the split).

        Three storage flavors are supported:

        - **simple_keys** — plain attributes on ``self.network`` set via
          ``setattr``. The bulk of ``updatable_keys``.
        - **nested_keys** — fields that live on ``network.config`` or in
          a sub-config and need a special-cased setter (``optimizer_type``
          via ``_write_optimizer_type``; ``activation_function_name`` via
          ``_write_activation_function_name``, which also re-runs
          ``_init_activation_function`` so ``activation_fn`` actually
          refreshes from the registry).
        - **lifecycle_keys** — flags that live on the lifecycle (``self``)
          rather than the network (``auto_snap_*``).

        All three flavors share the same GAP-WS-28 atomic-rollback contract:
        if any setter raises, every previously-applied key is reverted to
        its pre-call value before re-raising.
        """
        if self.network is None:
            raise ValueError("No network exists — create a network first")
        updatable_keys = {
            "learning_rate",
            "candidate_learning_rate",
            "correlation_threshold",
            "candidate_pool_size",
            "max_hidden_units",
            "epochs_max",
            "max_iterations",
            "patience",
            "convergence_threshold",
            "candidate_convergence_threshold",
            "candidate_patience",
            "candidate_epochs",
            "output_epochs",  # CAS-002 (Phase 6E Sprint A-1)
            "init_output_weights",
            "optimizer_type",  # CAN-010 / ENH-006 (Phase 6E Sprint A-2) — nested setter
            "activation_function_name",  # CAN-011 (Phase 6E Sprint A-3) — re-init on swap
            "auto_snap_best",  # CAS-006 (Phase 6E Sprint A-4) — lifecycle attribute
            "auto_snap_min_epochs",  # CAS-006 (Phase 6E Sprint A-4) — lifecycle attribute
        }
        nested_keys = {"optimizer_type", "activation_function_name"}
        lifecycle_keys = {"auto_snap_best", "auto_snap_min_epochs"}
        simple_keys = updatable_keys - nested_keys - lifecycle_keys
        applicable = {k: v for k, v in params.items() if k in simple_keys and hasattr(self.network, k)}
        old_values = {k: getattr(self.network, k) for k in applicable}

        # CAN-010 / ENH-006 (A-2): ``optimizer_type`` lives at
        # ``self.network.config.optimizer_config.optimizer_type``.
        optimizer_pending = "optimizer_type" in params and params["optimizer_type"] is not None
        old_optimizer_type = _read_optimizer_type(self.network) if optimizer_pending else None

        # CAN-011 (A-3): ``activation_function_name`` requires re-running
        # ``_init_activation_function`` so ``activation_fn`` picks up the
        # new mapping.
        activation_pending = "activation_function_name" in params and params["activation_function_name"] is not None
        old_activation_function_name = _read_activation_function_name(self.network) if activation_pending else None

        # CAS-006 (A-4): ``auto_snap_*`` live on the lifecycle. Snapshot
        # the old values (plus the best-metric tracker) so the same
        # rollback semantics extend to lifecycle storage.
        auto_snap_pending = {k: params[k] for k in lifecycle_keys if k in params and params[k] is not None}
        old_lifecycle_values: Dict[str, Any] = {}
        old_auto_snap_best_metric: Optional[float] = None
        if auto_snap_pending:
            with self._auto_snap_lock:
                old_lifecycle_values = {k: getattr(self, f"_{k}") for k in auto_snap_pending}
                old_auto_snap_best_metric = self._auto_snap_best_metric

        applied: list[str] = []
        try:
            for key, value in applicable.items():
                setattr(self.network, key, value)
                applied.append(key)
            if optimizer_pending:
                _write_optimizer_type(self.network, params["optimizer_type"])
                applied.append("optimizer_type")
            if activation_pending:
                _write_activation_function_name(self.network, params["activation_function_name"])
                applied.append("activation_function_name")
            if auto_snap_pending:
                with self._auto_snap_lock:
                    for key, value in auto_snap_pending.items():
                        setattr(self, f"_{key}", value)
                        applied.append(key)
                    # Toggling auto_snap_best off-then-on within a run would
                    # otherwise inherit the prior ceiling. Reset the tracker
                    # whenever the toggle flips on so the next epoch is
                    # treated as a fresh baseline.
                    if "auto_snap_best" in auto_snap_pending and auto_snap_pending["auto_snap_best"] and not old_lifecycle_values.get("auto_snap_best", False):
                        self._auto_snap_best_metric = None
        except Exception:
            # GAP-WS-28: revert any partial application before propagating.
            # CAN-010 / ENH-006 (A-2) + CAN-011 (A-3) + CAS-006 (A-4): each
            # flavor has its own revert path — mirror the apply branch.
            for key in reversed(applied):
                try:
                    if key == "optimizer_type":
                        _write_optimizer_type(self.network, old_optimizer_type)
                    elif key == "activation_function_name":
                        _write_activation_function_name(self.network, old_activation_function_name)
                    elif key in lifecycle_keys:
                        with self._auto_snap_lock:
                            setattr(self, f"_{key}", old_lifecycle_values[key])
                            self._auto_snap_best_metric = old_auto_snap_best_metric
                    else:
                        setattr(self.network, key, old_values[key])
                except Exception:
                    # If revert itself raises, log and continue rolling
                    # back the rest — best-effort consistency.
                    self.logger.exception("update_params rollback: revert of %s failed", key)
            raise
        return self.get_training_params()

    # ------------------------------------------------------------------
    # Topology & statistics
    # ------------------------------------------------------------------

    def get_topology(self) -> Optional[Dict[str, Any]]:
        """Extract network topology for visualization (thread-safe)."""
        if self.network is None:
            return None
        try:
            with self._topology_lock, torch.no_grad():
                topology = {
                    "input_size": self.network.input_size,
                    "output_size": self.network.output_size,
                    "hidden_units": [],
                    "output_weights": self.network.output_weights.detach().cpu().tolist(),
                    "output_bias": self.network.output_bias.detach().cpu().tolist(),
                }
                for i, unit in enumerate(self.network.hidden_units):
                    topology["hidden_units"].append(
                        {
                            "id": i,
                            "weights": unit["weights"].detach().cpu().tolist(),
                            "bias": float(unit["bias"]),
                            "activation": unit.get("activation_fn", torch.sigmoid).__name__,
                        }
                    )
            return topology
        except Exception as e:
            self.logger.error(f"Failed to extract topology: {e}", exc_info=True)
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get network weight statistics."""
        if self.network is None:
            return {}
        try:
            with self._topology_lock, torch.no_grad():
                output_weights = self.network.output_weights.detach().cpu()
                stats = {
                    "total_hidden_units": len(self.network.hidden_units),
                    "output_weight_mean": float(output_weights.mean()),
                    "output_weight_std": float(output_weights.std()),
                    "output_weight_min": float(output_weights.min()),
                    "output_weight_max": float(output_weights.max()),
                }
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}", exc_info=True)
            return {}

    # ------------------------------------------------------------------
    # Decision boundary
    # ------------------------------------------------------------------

    def get_decision_boundary(self, resolution: int = 50) -> Optional[Dict[str, Any]]:
        """Compute decision boundary grid for 2D visualization.

        Args:
            resolution: Number of grid points per axis.

        Returns:
            Dictionary with x_range, y_range, grid predictions, or None on failure.
        """
        if self.network is None or self._train_x is None:
            return None
        if self._train_x.shape[1] != 2:
            return None

        try:
            with self._topology_lock, torch.no_grad():
                x_data = self._train_x.cpu().numpy()
                x_min, x_max = float(x_data[:, 0].min()) - 0.5, float(x_data[:, 0].max()) + 0.5
                y_min, y_max = float(x_data[:, 1].min()) - 0.5, float(x_data[:, 1].max()) + 0.5

                xx = np.linspace(x_min, x_max, resolution)
                yy = np.linspace(y_min, y_max, resolution)
                grid_x, grid_y = np.meshgrid(xx, yy)
                grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

                grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
                predictions = self.network.forward(grid_tensor)
                pred_classes = predictions.argmax(dim=1).cpu().numpy()

            return {
                "x_range": [x_min, x_max],
                "y_range": [y_min, y_max],
                "resolution": resolution,
                "grid_x": grid_x.tolist(),
                "grid_y": grid_y.tolist(),
                "predictions": pred_classes.reshape(resolution, resolution).tolist(),
            }
        except Exception as e:
            self.logger.error(f"Failed to compute decision boundary: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def _get_snapshots_dir(self) -> Path:
        """Return the snapshots directory, creating it if needed."""
        snapshots_dir = Path(__file__).resolve().parent.parent.parent / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        return snapshots_dir

    def save_snapshot(self, description: str = "") -> Optional[Dict[str, Any]]:
        """Save current network state to an HDF5 snapshot."""
        if self.network is None:
            return None

        from snapshots.snapshot_serializer import CascadeHDF5Serializer

        serializer = CascadeHDF5Serializer()
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        snapshot_id = f"snapshot_{timestamp}"
        filepath = self._get_snapshots_dir() / f"{snapshot_id}.h5"

        success = serializer.save_network(self.network, filepath, include_training_state=True)
        if not success:
            self.logger.error(f"Failed to save snapshot to {filepath}")
            return None

        self.logger.info(f"Snapshot saved: {snapshot_id}")
        return {
            "id": snapshot_id,
            "path": str(filepath),
            "timestamp": timestamp,
            "description": description,
        }

    # CAN-015 (Phase 6E Sprint B): keys on ``network.history`` whose contents
    # represent per-epoch training metrics. ``restore_for_retrain`` empties
    # each one so a freshly-retrained run starts with a clean curve. Kept as
    # a class-level constant so the four B-sprint endpoints (Restore /
    # Replay / Resume / Retrain) share a single source of truth for what
    # "history" means; B-2 (Resume) and B-3 (Replay) will read from the
    # same set when locking it as read-only.
    _NETWORK_HISTORY_KEYS: tuple = ("train_loss", "value_loss", "train_accuracy", "value_accuracy")

    def _load_snapshot_to_network(self, snapshot_id: str) -> bool:
        """Locate the snapshot, deserialize, and install on the lifecycle.

        Internal helper extracted from ``load_snapshot`` so each Phase 6E
        Sprint B operation (Restore / Replay / Resume / Retrain) can share
        the load semantics while diverging on post-load state mutations
        (FSM transitions, history resets, replay-session setup, etc.).
        Returns True on success, False if the snapshot is missing or the
        deserializer fails.
        """
        snapshots_dir = self._get_snapshots_dir()
        matches = [f for f in snapshots_dir.glob("*.h5") if f.stem == snapshot_id]
        if not matches:
            self.logger.warning(f"Snapshot not found: {snapshot_id}")
            return False

        from snapshots.snapshot_serializer import CascadeHDF5Serializer

        serializer = CascadeHDF5Serializer()
        network = serializer.load_network(matches[0])
        if network is None:
            self.logger.error(f"Failed to load snapshot: {snapshot_id}")
            return False

        self._restore_original_methods()
        self.network = network
        self._install_monitoring_hooks()
        if self._worker_coordinator is not None and hasattr(self.network, "set_worker_coordinator"):
            self.network.set_worker_coordinator(self._worker_coordinator)
        return True

    def load_snapshot(self, snapshot_id: str) -> bool:
        """Load a network snapshot by ID (Restore semantics).

        Preserves the full snapshotted state — weights, topology, training
        history, all meta-parameters per A-5 (CAN-014). The FSM transitions
        to ``INVESTIGATING`` so the user can edit meta-params, replace the
        dataset, and re-snapshot, but cannot start training directly. To
        enter a training state, the user must invoke ``restore_for_retrain``
        (clean slate) or ``resume_from_snapshot`` (extend history).

        Rejected when training is currently active (Started / Paused) —
        same FSM-guard contract as Resume / Retrain. Returns False so
        the route layer can map to 409.

        See ``notes/PHASE_6E_SPRINT_B_DESIGN.md`` §2.1.
        """
        # Same pre-flight as resume_from_snapshot: investigating an
        # active training run would race with the running fit() and
        # leave the lifecycle in a confused state. CAN-015c (B-3) adds
        # the REPLAYING rejection so a Restore can't yank the network
        # out from under an active replay thread.
        if self.state_machine.is_started() or self.state_machine.is_paused() or self.state_machine.is_replaying():
            self.logger.warning(f"load_snapshot rejected: lifecycle is {self.state_machine.status.name}")
            return False

        ok = self._load_snapshot_to_network(snapshot_id)
        if not ok:
            return False

        # CAN-015d (B-4): transition to Investigating and clear any
        # state from prior snapshot operations. The user explicitly
        # invoked /restore (not /retrain or /resume) so we want the
        # inspection-only contract: training commands rejected, no
        # implicit history reset, no resume marker.
        self.state_machine.mark_investigating()
        self._resume_point_epoch = None
        self.training_state.update_state(status="Stopped", phase="Idle")
        self._broadcast_training_state(force=True)
        self.logger.info(f"Snapshot restored: {snapshot_id} (FSM=Investigating)")
        return True

    def restore_for_retrain(self, snapshot_id: str) -> bool:
        """Load a snapshot and reset training history for a fresh run (CAN-015a).

        Phase 6E Sprint B B-1. Loads the snapshot identically to
        ``load_snapshot`` (so weights, topology, and meta-params are
        preserved per A-5) then resets every history-bearing field so the
        next ``start_training`` call starts at epoch 0 with empty metric
        curves. The user benefits from the snapshot's prior training as a
        starting point but the new run is judged on its own merits.

        Reset scope per ``notes/PHASE_6E_SPRINT_B_DESIGN.md`` §9:

        - Network ``history`` arrays (train/value loss + accuracy) — cleared
        - ``training_state`` counters (current_epoch, current_step) — 0
        - ``_auto_snap_best_metric`` — None (so the new run gets fresh
          ratchet baseline; this also happens in ``start_training`` but
          we do it here too so a ``GET /v1/training/params`` between
          retrain and start_training already shows the cleared value)
        - FSM — Stopped / Idle (via ``Command.RESET``)
        - ``training_monitor.metrics_buffer`` — cleared
        - ``_last_emitted_history_len`` — 0

        CAN-015c (B-3): rejected when a replay session is active (would
        race with the replay thread reading from network.history).
        """
        if self.state_machine.is_started() or self.state_machine.is_paused() or self.state_machine.is_replaying():
            self.logger.warning(f"restore_for_retrain rejected: lifecycle is {self.state_machine.status.name}")
            return False
        ok = self._load_snapshot_to_network(snapshot_id)
        if not ok:
            return False

        # Clear history arrays on the network. ``getattr`` rather than direct
        # attribute access so a network that doesn't expose ``history`` yet
        # (older snapshots, or a partially-initialized network from a corner
        # case) doesn't crash the retrain — best-effort consistency mirrors
        # the legacy-snapshot tolerance from A-5.
        history = getattr(self.network, "history", None)
        if isinstance(history, dict):
            for key in self._NETWORK_HISTORY_KEYS:
                if key in history:
                    # Preserve the container type (list vs deque vs other) by
                    # replacing with an empty instance of the same type. Falls
                    # back to ``[]`` if the container isn't a known builtin.
                    try:
                        history[key] = type(history[key])()
                    except Exception:
                        history[key] = []

        # Reset lifecycle-level training state. Mirrors ``reset()`` (line ~840)
        # but without the ``_stop_requested.set()`` since no training is
        # currently running — Retrain is invoked from a stopped state and
        # ``start_training`` will clear the event itself.
        self._last_emitted_history_len = 0
        self.state_machine.handle_command(Command.RESET)
        self.training_monitor.clear_metrics()
        self.training_state.update_state(
            status="Stopped",
            phase="Idle",
            current_epoch=0,
            current_step=0,
        )
        with self._auto_snap_lock:
            self._auto_snap_best_metric = None
        # CAN-015b (B-2): a Retrain over a previously-loaded Resume
        # snapshot should not carry forward the resume marker — the
        # whole point of Retrain is the clean slate.
        self._resume_point_epoch = None
        self._broadcast_training_state(force=True)

        self.logger.info(f"Snapshot restored for retrain: {snapshot_id}")
        return True

    def resume_from_snapshot(self, snapshot_id: str) -> bool:
        """Load a snapshot and prepare to continue training (CAN-015b).

        Phase 6E Sprint B B-2. Loads the snapshot identically to
        ``load_snapshot`` (so weights, topology, meta-params, AND the
        training history are preserved) then transitions the FSM to
        ``RESUME_READY`` and records the snapshot's terminal-epoch count
        as ``_resume_point_epoch`` so canopy can render a visual
        boundary between the pre-resume read-only history and the new
        training that extends past it.

        In contrast to ``restore_for_retrain`` (which clears history,
        counters, and the auto-snap-best ratchet so the new run starts
        fresh), Resume PRESERVES every history-bearing field. The next
        ``start_training`` extends the existing arrays rather than
        starting at epoch 0, and the auto-snap-best ratchet keeps its
        prior accuracy ceiling so a re-snapshot only fires when the new
        training genuinely beats the previous run.

        Resume requires a non-active state (Stopped / Completed /
        Failed / RESUME_READY again). From STARTED or PAUSED the
        underlying ``mark_resume_ready`` call rejects and this method
        returns False.

        See ``notes/PHASE_6E_SPRINT_B_DESIGN.md`` §2.3 for the full
        spec.
        """
        if self.state_machine.is_started() or self.state_machine.is_paused() or self.state_machine.is_replaying():
            self.logger.warning(f"resume_from_snapshot rejected: lifecycle is {self.state_machine.status.name}")
            return False

        ok = self._load_snapshot_to_network(snapshot_id)
        if not ok:
            return False

        # Compute the resume-point epoch from the loaded network's
        # history. Use the longest array's length so a snapshot that's
        # missing some keys still produces a sensible marker. Falls back
        # to 0 if no history is present (a network freshly loaded with
        # no training-state included would land here — unusual but
        # tolerated).
        history = getattr(self.network, "history", None)
        resume_point = 0
        if isinstance(history, dict):
            for key in self._NETWORK_HISTORY_KEYS:
                series = history.get(key, ())
                try:
                    resume_point = max(resume_point, len(series))
                except TypeError:
                    # Unexpected type — skip, keep current best.
                    continue

        self._resume_point_epoch = resume_point
        self.state_machine.mark_resume_ready()
        # Surface the resume point in the broadcast so canopy clients
        # that subscribe to state updates pick it up immediately.
        # Mirrors the pattern used by reset() / restore_for_retrain.
        # NOTE: ``training_state.status`` stays "Stopped" rather than
        # "ResumeReady" — canopy reads RESUME_READY from the FSM summary
        # (state_machine.get_state_summary()), not from training_state.
        self.training_state.update_state(
            status="Stopped",
            phase="Idle",
            current_epoch=resume_point,
        )
        self._broadcast_training_state(force=True)

        self.logger.info(f"Snapshot restored for resume: {snapshot_id} (resume_point_epoch={resume_point})")
        return True

    def start_replay(self, snapshot_id: str) -> bool:
        """Load a snapshot and start a replay session (CAN-015c).

        Phase 6E Sprint B B-3. Loads the snapshot identically to
        ``load_snapshot`` then transitions the FSM to ``REPLAYING`` and
        spawns a background ``_ReplaySession`` thread that emits
        synthetic ``epoch_end`` events from the loaded network's
        history arrays at a configurable speed.

        V1 scope: metric arrays + topology evolution metadata only.
        Per-epoch weight history (decision-boundary playback) is
        deferred to CAN-015g — would require a snapshot-format
        extension.

        Rejected when training is currently active (Started / Paused).
        Replacing one replay session with another is permitted — the
        old session's thread is stopped and the new session is
        installed.

        See ``notes/PHASE_6E_SPRINT_B_DESIGN.md`` §2.2.
        """
        if self.state_machine.is_started() or self.state_machine.is_paused():
            self.logger.warning(f"start_replay rejected: training is {self.state_machine.status.name}")
            return False

        ok = self._load_snapshot_to_network(snapshot_id)
        if not ok:
            return False

        # If a previous replay session was running, tear it down first
        # so its thread doesn't keep emitting against the new history.
        prev_session = self._replay_session
        if prev_session is not None:
            try:
                prev_session.stop()
            except Exception:
                self.logger.exception("start_replay: failed to stop previous replay session")

        history = getattr(self.network, "history", None)
        history_dict = history if isinstance(history, dict) else {}
        session = _ReplaySession(snapshot_id, history_dict, self.training_monitor)
        self._replay_session = session
        # Marker fields used by Resume / Restore are not relevant here.
        self._resume_point_epoch = None
        self.state_machine.mark_replaying()
        self.training_state.update_state(status="Stopped", phase="Idle")
        self._broadcast_training_state(force=True)
        # Start the driver thread AFTER the FSM transitions so the
        # initial frame emission lands while subscribers are looking
        # at a Replaying state.
        session.start_thread()

        self.logger.info(f"Snapshot replay started: {snapshot_id} (length={session.length})")
        return True

    def replay_control(self, action: str, **params: Any) -> Dict[str, Any]:
        """Apply a control action to the active replay session (CAN-015c).

        Supported actions: ``play`` / ``pause`` / ``seek`` (param
        ``time_index``) / ``speed`` (param ``value``) / ``range``
        (params ``start`` and ``end``) / ``stop``. ``stop`` exits
        Replaying — the FSM transitions back to ``STOPPED`` and the
        session thread is joined.

        Returns the post-action session state for the route response.
        Raises ``RuntimeError`` if no session is active.
        """
        session = self._replay_session
        if session is None or not self.state_machine.is_replaying():
            raise RuntimeError("No active replay session")

        action_lower = action.lower() if isinstance(action, str) else ""
        if action_lower == "play":
            session.play()
        elif action_lower == "pause":
            session.pause()
        elif action_lower == "seek":
            target = params.get("time_index")
            if target is None:
                raise ValueError("seek requires a 'time_index' parameter")
            session.seek(int(target))
        elif action_lower == "speed":
            value = params.get("value")
            if value is None:
                raise ValueError("speed requires a 'value' parameter")
            session.set_speed(float(value))
        elif action_lower == "range":
            start = params.get("start")
            end = params.get("end")
            if start is None or end is None:
                raise ValueError("range requires both 'start' and 'end' parameters")
            session.set_range(int(start), int(end))
        elif action_lower == "stop":
            return self.stop_replay()
        else:
            raise ValueError(f"Unknown replay action: {action!r}")
        return session.state_summary()

    def stop_replay(self) -> Dict[str, Any]:
        """End the active replay session (CAN-015c).

        Joins the background thread, clears ``_replay_session``,
        transitions the FSM to STOPPED via ``Command.RESET``, and
        broadcasts the resulting state. Idempotent — calling on an
        inactive session returns a minimal "not_active" status.
        """
        session = self._replay_session
        if session is None:
            return {"status": "not_active"}
        try:
            session.stop()
        finally:
            self._replay_session = None
        # RESET is the universal "back to Stopped" transition. The FSM
        # already documents that REPLAYING accepts RESET as the escape
        # hatch alongside the explicit /control stop.
        self.state_machine.handle_command(Command.RESET)
        self.training_state.update_state(status="Stopped", phase="Idle")
        self._broadcast_training_state(force=True)
        self.logger.info(f"Snapshot replay stopped: {session.snapshot_id}")
        return {"status": "stopped", "snapshot_id": session.snapshot_id}

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available snapshots."""
        snapshots_dir = self._get_snapshots_dir()
        snapshots = []
        for filepath in sorted(snapshots_dir.glob("*.h5")):
            snapshots.append(
                {
                    "id": filepath.stem,
                    "path": str(filepath),
                    "size_bytes": filepath.stat().st_size,
                    "modified": datetime.fromtimestamp(filepath.stat().st_mtime, tz=UTC).isoformat(),
                }
            )
        return snapshots

    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific snapshot."""
        snapshots_dir = self._get_snapshots_dir()
        matches = [f for f in snapshots_dir.glob("*.h5") if f.stem == snapshot_id]
        if not matches:
            return None
        filepath = matches[0]
        return {
            "id": filepath.stem,
            "path": str(filepath),
            "size_bytes": filepath.stat().st_size,
            "modified": datetime.fromtimestamp(filepath.stat().st_mtime, tz=UTC).isoformat(),
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean up resources."""
        self._stop_requested.set()
        self.stop_liveness_heartbeat()
        self._restore_original_methods()
        # CAN-015c (B-3): drain any active replay session so the
        # background driver thread doesn't outlive the lifecycle.
        if self._replay_session is not None:
            try:
                self._replay_session.stop()
            except Exception:
                self.logger.exception("shutdown: failed to stop replay session")
            self._replay_session = None
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self.logger.info("TrainingLifecycleManager shut down")
