"""Training monitor for real-time metrics collection.

Ported from juniper-canopy backend/training_monitor.py.
Monitors CasCor training and collects metrics. Simplified version without
DataAdapter dependency — metrics are stored as plain dicts.
"""

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from cascor_constants.constants_api import _PROJECT_API_METRICS_BUFFER_SIZE


class TrainingState:
    """Thread-safe single source of truth for all training state.

    Provides atomic state updates and serialization for REST/WebSocket broadcasting.

    Counter semantics (C2b, I-1c / S12 — the contract canopy's header/tiles consume;
    see also ``docs/api/JUNIPER_CASCOR_API_REFERENCE.md`` §"Counter semantics"):

    - ``current_epoch`` / ``current_step`` — completed **training steps**: entries in
      the engine's per-pass ``history`` arrays, i.e. one initial output-training pass
      plus one per cascade growth iteration. NOT inner output-training epochs. Single
      writer: the manager's history drain (``_extract_and_record_metrics``); the two
      fields are currently written in lock-step and are aliases of each other.
    - ``max_epochs`` — the Q1 **derived** total-epoch cap implied by the granular
      limits (``TrainingLifecycleManager.derive_epochs_cap``): ``output_epochs +
      min(max_iterations, max_hidden_units) * (candidate_epochs + output_epochs)``.
      A reporting/display budget (the ``Epoch: X / Y`` denominator), not an enforced
      abort — enforcement stays with the granular limits themselves. Refreshed at
      network create / param apply / snapshot load. Pre-C2b this was an independently
      seeded value (default 1e11) the training loop never read.
    - ``output_epoch`` / ``output_total_epochs`` — live within-pass progress of the
      CURRENT output-training pass (inner epoch / pass budget, throttled to ~every
      25th epoch by the engine callback). The output-phase sibling of the
      ``candidate_epoch`` pair; zeroed at run start, growth-phase exit, and run end.
    - ``candidate_epoch`` / ``candidate_total_epochs`` — live within-pass progress of
      the current candidate-pool training pass (from the worker progress queue).
    - ``grow_iteration`` / ``grow_max`` — cascade growth iteration counter vs its
      ``max_iterations`` limit.
    - ``learning_rate`` / ``max_hidden_units`` / ``max_iterations`` — projections of
      the live network's effective values (synced by the manager's
      ``_sync_training_state_from_network`` at create / apply / snapshot-load), NOT
      an independent default layer.
    """

    _STATE_FIELDS = {
        "status",
        "phase",
        "learning_rate",
        "max_hidden_units",
        "max_epochs",
        "max_iterations",
        "current_epoch",
        "current_step",
        "network_name",
        "dataset_name",
        "threshold_function",
        "optimizer_name",
        "timestamp",
        "phase_detail",
        "grow_iteration",
        "grow_max",
        "best_correlation",
        "candidates_trained",
        "candidates_total",
        "phase_started_at",
        "candidate_epoch",
        "candidate_total_epochs",
        "output_epoch",
        "output_total_epochs",
        "best_candidate_id",
        "best_candidate_uuid",
        "second_candidate_id",
        "second_candidate_correlation",
        "all_correlations",
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._status: str = "Stopped"
        self._phase: str = "Idle"
        self._learning_rate: float = 0.0
        # C2b: pre-network defaults are 0 ("no network / unknown") — the old
        # literals (200 / 1000) were a third default layer that reported
        # limits no network was actually configured with. The manager's
        # _sync_training_state_from_network overwrites these at create time.
        self._max_hidden_units: int = 0
        self._max_epochs: int = 0
        self._max_iterations: int = 0
        self._current_epoch: int = 0
        self._current_step: int = 0
        self._network_name: str = ""
        self._dataset_name: str = ""
        self._threshold_function: str = ""
        self._optimizer_name: str = ""
        self._timestamp: float = time.time()
        self._phase_detail: str = ""
        self._grow_iteration: int = 0
        self._grow_max: int = 0
        self._best_correlation: float = 0.0
        self._candidates_trained: int = 0
        self._candidates_total: int = 0
        self._phase_started_at: str = ""
        self._candidate_epoch: int = 0
        self._candidate_total_epochs: int = 0
        self._output_epoch: int = 0
        self._output_total_epochs: int = 0
        self._best_candidate_id: int = -1
        self._best_candidate_uuid: str = ""
        self._second_candidate_id: Optional[int] = None
        self._second_candidate_correlation: float = 0.0
        self._all_correlations: List[float] = []

    def get_state(self) -> Dict[str, Any]:
        """Get current state as dictionary."""
        with self._lock:
            return {
                "status": self._status,
                "phase": self._phase,
                "learning_rate": self._learning_rate,
                "max_hidden_units": self._max_hidden_units,
                "max_epochs": self._max_epochs,
                "max_iterations": self._max_iterations,
                "current_epoch": self._current_epoch,
                "current_step": self._current_step,
                "network_name": self._network_name,
                "dataset_name": self._dataset_name,
                "threshold_function": self._threshold_function,
                "optimizer_name": self._optimizer_name,
                "timestamp": self._timestamp,
                "phase_detail": self._phase_detail,
                "grow_iteration": self._grow_iteration,
                "grow_max": self._grow_max,
                "best_correlation": self._best_correlation,
                "candidates_trained": self._candidates_trained,
                "candidates_total": self._candidates_total,
                "phase_started_at": self._phase_started_at,
                "candidate_epoch": self._candidate_epoch,
                "candidate_total_epochs": self._candidate_total_epochs,
                "output_epoch": self._output_epoch,
                "output_total_epochs": self._output_total_epochs,
                "best_candidate_id": self._best_candidate_id,
                "best_candidate_uuid": self._best_candidate_uuid,
                "second_candidate_id": self._second_candidate_id,
                "second_candidate_correlation": self._second_candidate_correlation,
                "all_correlations": list(self._all_correlations),
            }

    def update_state(self, **kwargs) -> None:
        """Update state fields atomically.

        Accepts keyword arguments using field names.
        Unknown fields are ignored. Passing None leaves the field unchanged.
        """
        with self._lock:
            updated = False
            for key, value in kwargs.items():
                if value is None or key not in self._STATE_FIELDS:
                    continue
                attr = f"_{key}"
                if hasattr(self, attr):
                    setattr(self, attr, value)
                    updated = True
            if updated and "timestamp" not in kwargs:
                self._timestamp = time.time()

    def to_json(self) -> str:
        """Serialize state to JSON string."""
        return json.dumps(self.get_state())


class TrainingMonitor:
    """Monitors CasCor training process and collects real-time metrics.

    Provides callbacks for training events:
    - Epoch start/end
    - Cascade unit addition
    - Training state changes
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.metrics_buffer: deque = deque(maxlen=_PROJECT_API_METRICS_BUFFER_SIZE)
        self.is_training = False
        self.current_epoch = 0
        self.current_hidden_units = 0
        self.current_phase = "output"

        self.callbacks: Dict[str, List[Callable]] = {
            "epoch_start": [],
            "epoch_end": [],
            "cascade_add": [],
            "training_start": [],
            "training_end": [],
            "topology_change": [],
            "candidate_progress": [],
            # BUG-CC-07: phase change driven by state machine
            "phase_change": [],
        }

        self._lock = threading.Lock()
        self.logger.info("TrainingMonitor initialized")

    def on_phase_change(self, phase: str) -> None:
        """Update current_phase from state machine notification (BUG-CC-07)."""
        with self._lock:
            self.current_phase = phase
        self._trigger_callbacks("phase_change", phase=phase)

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register callback for training event."""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            self.logger.warning(f"Unknown event type: {event_type}")

    def _trigger_callbacks(self, event_type: str, **kwargs) -> None:
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(**kwargs)
            except Exception as e:
                self.logger.error(f"Callback error for {event_type}: {e}")

    def on_training_start(self) -> None:
        with self._lock:
            self.is_training = True
            self.current_epoch = 0
            self.metrics_buffer.clear()
        self.logger.info("Training started")
        self._trigger_callbacks("training_start")

    def on_training_end(self, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self.is_training = False
        self.logger.info("Training ended")
        self._trigger_callbacks("training_end", final_metrics=final_metrics)

    def on_epoch_end(
        self,
        epoch: int,
        loss: float,
        accuracy: float,
        learning_rate: float,
        hidden_units: int = 0,
        validation_loss: Optional[float] = None,
        validation_accuracy: Optional[float] = None,
    ) -> None:
        metrics = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "loss": loss,
            "accuracy": accuracy,
            "learning_rate": learning_rate,
            "hidden_units": hidden_units,
            "phase": self.current_phase,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }

        with self._lock:
            self.current_epoch = epoch
            # Track the live hidden-unit count from the per-epoch metric stream.
            # The caller passes ``hidden_units=len(network.hidden_units)``
            # (manager.py output-training callback). ``current_hidden_units``
            # previously updated ONLY in ``on_cascade_add`` — which has no
            # production caller — so the status field (and canopy's status bar,
            # which reads it) sat at 0 even as the cascade grew units. Sourcing
            # it here keeps it correct without relying on the unwired callback.
            self.current_hidden_units = hidden_units
            self.metrics_buffer.append(metrics)

        self._trigger_callbacks("epoch_end", metrics=metrics, epoch=epoch, loss=loss, accuracy=accuracy)

    def on_cascade_add(self, hidden_unit_index: int, correlation: float) -> None:
        with self._lock:
            self.current_hidden_units += 1

        event = {
            "timestamp": datetime.now().isoformat(),
            "hidden_unit_index": hidden_unit_index,
            "correlation": correlation,
            "total_hidden_units": self.current_hidden_units,
        }
        self.logger.info(f"Cascade unit {hidden_unit_index} added (correlation={correlation:.4f})")
        self._trigger_callbacks("cascade_add", event=event)

    def on_candidate_progress(self, progress: Dict[str, Any]) -> None:
        """Handle candidate training progress update from worker pool."""
        self._trigger_callbacks("candidate_progress", progress=progress)

    def get_recent_metrics(self, count: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self.metrics_buffer)
            return items[-count:]

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.metrics_buffer)

    def get_current_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "is_training": self.is_training,
                "current_epoch": self.current_epoch,
                "current_hidden_units": self.current_hidden_units,
                "current_phase": self.current_phase,
                "total_metrics": len(self.metrics_buffer),
            }

    def clear_metrics(self) -> None:
        with self._lock:
            self.metrics_buffer.clear()
        self.logger.info("Metrics buffer cleared")
