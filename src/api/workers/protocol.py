"""Wire protocol for WebSocket worker communication.

METRICS-MON R2.2.2 / seed-05: the canonical wire-protocol message types
and binary-frame codec live in :mod:`juniper_cascor_protocol.worker`
(published as ``juniper-cascor-protocol`` on PyPI). This module re-
exports them for backwards compatibility with existing imports across
``api/workers/``, ``api/websocket/worker_stream.py``, and the test
suites — new code should prefer
``from juniper_cascor_protocol.worker import …``.

What stays in this module:

- :class:`WorkerProtocol` — imperative builder/validator helpers used by
  the cascor server's worker_stream handler. The builders now reference
  :class:`MessageType` re-exported from the shared lib so the wire
  values stay single-sourced.
- :class:`TaskAssignment`, :class:`TaskResultMessage` — typed dataclasses
  cascor uses internally for type-safe construction. They serialize to
  the same dicts the workers expect.

Binary frame format and the message-type enum are defined in
:mod:`juniper_cascor_protocol.worker.binary_frame` and
:mod:`juniper_cascor_protocol.worker.messages` respectively. See
``notes/code-review/METRICS_MONITORING_R2.2_WS_FRAME_SCHEMA_DESIGN_2026-04-29.md``
in juniper-ml for the cross-repo design.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from juniper_cascor_protocol.worker import BinaryFrame, WorkerMessageType

# METRICS-MON R2.2.2: re-export the shared StrEnum under its historical
# server-side alias. Worker stream code, tests, and downstream callers
# that imported ``MessageType`` continue to work unchanged.
MessageType = WorkerMessageType

__all__ = [
    "BinaryFrame",
    "MessageType",
    "WorkerProtocol",
    "TaskAssignment",
    "TaskResultMessage",
]


# Validation bounds (Section 12.7 of concurrency plan)
_MAX_CORRELATION = 1.0
_MIN_CORRELATION = 0.0
_MAX_WEIGHT_MAGNITUDE = 100.0


class WorkerProtocol:
    """Build and validate wire protocol messages."""

    # --- Message Builders ---

    @staticmethod
    def build_register(worker_id: str, capabilities: dict[str, Any]) -> dict[str, Any]:
        """Build a worker registration message."""
        return {
            "type": MessageType.REGISTER,
            "worker_id": worker_id,
            "capabilities": capabilities,
        }

    @staticmethod
    def build_heartbeat(worker_id: str) -> dict[str, Any]:
        """Build a heartbeat message."""
        return {
            "type": MessageType.HEARTBEAT,
            "worker_id": worker_id,
            "timestamp": time.time(),
        }

    @staticmethod
    def build_task_assign(
        task_id: str,
        round_id: str,
        candidate_index: int,
        candidate_data: dict[str, Any],
        training_params: dict[str, Any],
        tensor_manifest: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a task assignment message.

        Args:
            task_id: Unique task identifier.
            round_id: Training round identifier.
            candidate_index: Index of the candidate in the pool.
            candidate_data: Candidate configuration (input_size, activation_name, etc.).
            training_params: Training hyperparameters (epochs, learning_rate, etc.).
            tensor_manifest: Description of binary frames to follow.
        """
        return {
            "type": MessageType.TASK_ASSIGN,
            "task_id": task_id,
            "round_id": round_id,
            "candidate_index": candidate_index,
            "candidate_data": candidate_data,
            "training_params": training_params,
            "tensor_manifest": tensor_manifest,
        }

    @staticmethod
    def build_task_result(
        task_id: str,
        candidate_id: int,
        candidate_uuid: str,
        correlation: float,
        success: bool,
        epochs_completed: int,
        activation_name: str,
        all_correlations: list[float],
        numerator: float,
        denominator: float,
        best_corr_idx: int,
        tensor_manifest: dict[str, dict[str, Any]],
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """Build a task result message."""
        return {
            "type": MessageType.TASK_RESULT,
            "task_id": task_id,
            "candidate_id": candidate_id,
            "candidate_uuid": candidate_uuid,
            "correlation": correlation,
            "success": success,
            "epochs_completed": epochs_completed,
            "activation_name": activation_name,
            "all_correlations": all_correlations,
            "numerator": numerator,
            "denominator": denominator,
            "best_corr_idx": best_corr_idx,
            "error_message": error_message,
            "tensor_manifest": tensor_manifest,
        }

    @staticmethod
    def build_error(error: str, details: str | None = None) -> dict[str, Any]:
        """Build an error message."""
        msg: dict[str, Any] = {
            "type": MessageType.ERROR,
            "error": error,
            "timestamp": time.time(),
        }
        if details is not None:
            msg["details"] = details
        return msg

    # --- Validation ---

    @staticmethod
    def validate_task_result(msg: dict[str, Any]) -> list[str]:
        """Validate a task_result message against the schema (Section 12.7).

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        # Required fields
        required = ["task_id", "candidate_id", "correlation", "success", "epochs_completed"]
        for field_name in required:
            if field_name not in msg:
                errors.append(f"Missing required field: {field_name}")

        if errors:
            return errors

        # Type checks
        if not isinstance(msg["candidate_id"], int):
            errors.append(f"candidate_id must be int, got {type(msg['candidate_id']).__name__}")
        if not isinstance(msg["correlation"], (int, float)):
            errors.append(f"correlation must be numeric, got {type(msg['correlation']).__name__}")
        if not isinstance(msg["success"], bool):
            errors.append(f"success must be bool, got {type(msg['success']).__name__}")
        if not isinstance(msg["epochs_completed"], int):
            errors.append(f"epochs_completed must be int, got {type(msg['epochs_completed']).__name__}")

        if errors:
            return errors

        # Bounds checking
        corr = msg["correlation"]
        if not (_MIN_CORRELATION <= corr <= _MAX_CORRELATION):
            errors.append(f"correlation out of bounds: {corr} (expected [{_MIN_CORRELATION}, {_MAX_CORRELATION}])")

        return errors

    @staticmethod
    def validate_tensors(tensors: dict[str, np.ndarray], manifest: dict[str, dict[str, Any]]) -> list[str]:
        """Validate received tensors against the manifest.

        Checks shapes, dtypes, and values (NaN/Inf/magnitude).

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        for name, spec in manifest.items():
            if name not in tensors:
                errors.append(f"Missing tensor: {name}")
                continue

            if not isinstance(spec, dict):
                errors.append(f"Tensor {name}: manifest entry must be a dict with shape/dtype")
                continue

            if "shape" not in spec:
                errors.append(f"Tensor {name}: manifest missing required field: shape")
                continue
            if "dtype" not in spec:
                errors.append(f"Tensor {name}: manifest missing required field: dtype")
                continue

            arr = tensors[name]
            expected_shape = tuple(spec["shape"])
            expected_dtype = spec["dtype"]

            if arr.shape != expected_shape:
                errors.append(f"Tensor {name}: shape mismatch — expected {expected_shape}, got {arr.shape}")

            if str(arr.dtype) != expected_dtype:
                errors.append(f"Tensor {name}: dtype mismatch — expected {expected_dtype}, got {arr.dtype}")

            if np.any(np.isnan(arr)):
                errors.append(f"Tensor {name}: contains NaN values")

            if np.any(np.isinf(arr)):
                errors.append(f"Tensor {name}: contains Inf values")

        # Check for weight magnitude (configurable). Empty arrays have no
        # magnitude to evaluate — treat them as a validation error rather
        # than letting ``np.max`` raise on a zero-size reduction.
        if "weights" in tensors:
            weights = tensors["weights"]
            if weights.size == 0:
                errors.append("Tensor weights: empty array")
            else:
                max_weight = float(np.max(np.abs(weights)))
                if max_weight > _MAX_WEIGHT_MAGNITUDE:
                    errors.append(f"Weight magnitude too large: {max_weight:.2f} > {_MAX_WEIGHT_MAGNITUDE}")

        return errors

    _WORKER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

    @staticmethod
    def validate_register(msg: dict[str, Any]) -> list[str]:
        """Validate a registration message.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []
        if "worker_id" not in msg:
            errors.append("Missing required field: worker_id")
        else:
            wid = msg["worker_id"]
            if not isinstance(wid, str):
                errors.append("worker_id must be a string")
            elif not WorkerProtocol._WORKER_ID_PATTERN.match(wid):
                errors.append("worker_id must be 1-64 characters, alphanumeric/hyphens/underscores, starting with alphanumeric")
        if "capabilities" not in msg:
            errors.append("Missing required field: capabilities")
        elif not isinstance(msg["capabilities"], dict):
            errors.append("capabilities must be a dict")
        return errors


# ---------------------------------------------------------------------------
# Typed message dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TaskAssignment:
    """Typed representation of a task_assign message.

    Provides type-safe construction of task assignment payloads.
    Use ``to_dict()`` to serialize for the wire protocol.
    """

    task_id: str
    round_id: str
    candidate_index: int
    candidate_data: dict[str, Any]
    training_params: dict[str, Any]
    tensor_manifest: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a wire-protocol dict via WorkerProtocol."""
        return WorkerProtocol.build_task_assign(
            task_id=self.task_id,
            round_id=self.round_id,
            candidate_index=self.candidate_index,
            candidate_data=self.candidate_data,
            training_params=self.training_params,
            tensor_manifest=self.tensor_manifest,
        )


@dataclass
class TaskResultMessage:
    """Typed representation of a task_result message.

    Provides type-safe construction and parsing of task result payloads.
    Named ``TaskResultMessage`` to distinguish from the coordinator's
    internal ``TaskResult`` which includes decoded tensor arrays.
    """

    task_id: str
    candidate_id: int
    candidate_uuid: str = ""
    correlation: float = 0.0
    success: bool = False
    epochs_completed: int = 0
    activation_name: str = ""
    all_correlations: list[float] = field(default_factory=list)
    numerator: float = 0.0
    denominator: float = 1.0
    best_corr_idx: int = -1
    tensor_manifest: dict[str, dict[str, Any]] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a wire-protocol dict via WorkerProtocol."""
        return WorkerProtocol.build_task_result(
            task_id=self.task_id,
            candidate_id=self.candidate_id,
            candidate_uuid=self.candidate_uuid,
            correlation=self.correlation,
            success=self.success,
            epochs_completed=self.epochs_completed,
            activation_name=self.activation_name,
            all_correlations=self.all_correlations,
            numerator=self.numerator,
            denominator=self.denominator,
            best_corr_idx=self.best_corr_idx,
            tensor_manifest=self.tensor_manifest,
            error_message=self.error_message,
        )

    @classmethod
    def from_dict(cls, msg: dict[str, Any]) -> "TaskResultMessage":
        """Parse a wire-protocol dict into a typed dataclass.

        Raises:
            ValueError: If required fields are missing or validation fails.
        """
        errors = WorkerProtocol.validate_task_result(msg)
        if errors:
            raise ValueError(f"Invalid task_result message: {'; '.join(errors)}")
        return cls(
            task_id=msg["task_id"],
            candidate_id=msg["candidate_id"],
            candidate_uuid=msg.get("candidate_uuid", ""),
            correlation=msg["correlation"],
            success=msg["success"],
            epochs_completed=msg["epochs_completed"],
            activation_name=msg.get("activation_name", ""),
            all_correlations=msg.get("all_correlations", []),
            numerator=msg.get("numerator", 0.0),
            denominator=msg.get("denominator", 1.0),
            best_corr_idx=msg.get("best_corr_idx", -1),
            tensor_manifest=msg.get("tensor_manifest", {}),
            error_message=msg.get("error_message"),
        )
