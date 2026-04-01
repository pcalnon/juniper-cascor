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


class TrainingState:
    """Thread-safe single source of truth for all training state.

    Provides atomic state updates and serialization for REST/WebSocket broadcasting.
    """

    _STATE_FIELDS = {
        "status",
        "phase",
        "learning_rate",
        "max_hidden_units",
        "max_epochs",
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
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._status: str = "Stopped"
        self._phase: str = "Idle"
        self._learning_rate: float = 0.0
        self._max_hidden_units: int = 0
        self._max_epochs: int = 200
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

    def get_state(self) -> Dict[str, Any]:
        """Get current state as dictionary."""
        with self._lock:
            return {
                "status": self._status,
                "phase": self._phase,
                "learning_rate": self._learning_rate,
                "max_hidden_units": self._max_hidden_units,
                "max_epochs": self._max_epochs,
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

        self.metrics_buffer: deque = deque(maxlen=10000)
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
        }

        self._lock = threading.Lock()
        self.logger.info("TrainingMonitor initialized")

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
