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

        self.logger.info("TrainingLifecycleManager initialized")

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
            **kwargs: Additional kwargs passed to network.fit()

        Returns:
            Status dictionary
        """
        if self.network is None:
            raise RuntimeError("No network created")

        with self._training_lock:
            if self.state_machine.is_started():
                raise RuntimeError("Training already in progress")

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

            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cascor-train")

            self._training_future = self._executor.submit(self._run_training, self._train_x, self._train_y, self._val_x, self._val_y, **kwargs)

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
            "init_output_weights": getattr(self.network, "init_output_weights", "zero"),
        }

    def update_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update runtime-modifiable training parameters (thread-safe).

        Modifies the live network's attributes directly. Parameters that are
        safe to update while training is running: learning_rate,
        candidate_learning_rate, correlation_threshold, candidate_pool_size.
        Parameters effective at next cascade/epoch: max_hidden_units, epochs_max,
        patience.

        Args:
            params: Dict of parameter names and new values (None values excluded).

        Returns:
            Updated training parameters dict.

        Raises:
            ValueError: If no network exists.
        """
        with self._training_lock:
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
                "init_output_weights",
            }
            for key, value in params.items():
                if key in updatable_keys and hasattr(self.network, key):
                    setattr(self.network, key, value)
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

    def load_snapshot(self, snapshot_id: str) -> bool:
        """Load a network snapshot by ID."""
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
        self.logger.info(f"Snapshot restored: {snapshot_id}")
        return True

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
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self.logger.info("TrainingLifecycleManager shut down")
