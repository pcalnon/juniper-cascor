"""Wire protocol for WebSocket worker communication.

Defines message formats and binary frame encoding for the worker protocol.
All serialization uses JSON envelopes + raw numpy binary frames — no pickle.

Binary Frame Format:
    [4 bytes: shape dimension count (uint32)]
    [N * 4 bytes: shape values (uint32 each)]
    [4 bytes: dtype string length (uint32)]
    [M bytes: dtype string (utf-8)]
    [remaining bytes: raw array data]

Message Types:
    register       — Worker -> Server: first message after connect
    heartbeat      — Bidirectional: keepalive
    task_assign    — Server -> Worker: candidate training task
    task_result    — Worker -> Server: training result
    token_refresh  — Server -> Worker: new auth token before expiry
    error          — Either direction: error notification
"""

import struct
import time
from enum import StrEnum
from typing import Any

import numpy as np


class MessageType(StrEnum):
    """All valid wire protocol message types."""

    REGISTER = "register"
    HEARTBEAT = "heartbeat"
    TASK_ASSIGN = "task_assign"
    TASK_RESULT = "task_result"
    TOKEN_REFRESH = "token_refresh"
    ERROR = "error"


# Validation bounds (Section 12.7 of concurrency plan)
_MAX_CORRELATION = 1.0
_MIN_CORRELATION = 0.0
_MAX_WEIGHT_MAGNITUDE = 100.0
_MAX_FRAME_SIZE = 100 * 1024 * 1024  # 100MB


class BinaryFrame:
    """Encode/decode numpy arrays as binary WebSocket frames.

    Format: shape header + dtype header + raw data bytes.
    No pickle involved — reconstructible from numpy + struct only.
    """

    @staticmethod
    def encode(array: np.ndarray) -> bytes:
        """Encode a numpy array into a binary frame.

        Args:
            array: C-contiguous numpy array to encode.

        Returns:
            Binary frame bytes.
        """
        arr = np.ascontiguousarray(array)
        shape = arr.shape
        dtype_str = str(arr.dtype).encode("utf-8")

        header = struct.pack(f"<I{len(shape)}I", len(shape), *shape)
        header += struct.pack("<I", len(dtype_str))
        header += dtype_str

        return header + arr.tobytes()

    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        """Decode a binary frame into a numpy array.

        Args:
            data: Binary frame bytes.

        Returns:
            Reconstructed numpy array.

        Raises:
            ValueError: If frame is malformed or exceeds size limits.
        """
        if len(data) > _MAX_FRAME_SIZE:
            raise ValueError(f"Binary frame exceeds maximum size ({len(data)} > {_MAX_FRAME_SIZE})")

        offset = 0

        # Read shape dimension count
        if len(data) < 4:
            raise ValueError("Binary frame too short for shape header")
        (ndim,) = struct.unpack_from("<I", data, offset)
        offset += 4

        if ndim > 10:
            raise ValueError(f"Unreasonable number of dimensions: {ndim}")

        # Read shape values
        if len(data) < offset + ndim * 4:
            raise ValueError("Binary frame too short for shape values")
        shape = struct.unpack_from(f"<{ndim}I", data, offset)
        offset += ndim * 4

        # Read dtype string
        if len(data) < offset + 4:
            raise ValueError("Binary frame too short for dtype header")
        (dtype_len,) = struct.unpack_from("<I", data, offset)
        offset += 4

        if dtype_len > 64:
            raise ValueError(f"Unreasonable dtype string length: {dtype_len}")
        if len(data) < offset + dtype_len:
            raise ValueError("Binary frame too short for dtype string")
        dtype_str = data[offset : offset + dtype_len].decode("utf-8")
        offset += dtype_len

        # Validate dtype before use
        try:
            dtype = np.dtype(dtype_str)
        except TypeError as e:
            raise ValueError(f"Invalid dtype string: {dtype_str!r}") from e

        # Read array data
        expected_size = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
        actual_size = len(data) - offset
        if actual_size != expected_size:
            raise ValueError(f"Data size mismatch: expected {expected_size} bytes, got {actual_size}")

        array = np.frombuffer(data[offset:], dtype=dtype).reshape(shape)
        return array.copy()  # Return owned copy, not view into buffer


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
        for field in required:
            if field not in msg:
                errors.append(f"Missing required field: {field}")

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

        # Check for weight magnitude (configurable)
        if "weights" in tensors:
            max_weight = float(np.max(np.abs(tensors["weights"])))
            if max_weight > _MAX_WEIGHT_MAGNITUDE:
                errors.append(f"Weight magnitude too large: {max_weight:.2f} > {_MAX_WEIGHT_MAGNITUDE}")

        return errors

    @staticmethod
    def validate_register(msg: dict[str, Any]) -> list[str]:
        """Validate a registration message.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []
        if "worker_id" not in msg:
            errors.append("Missing required field: worker_id")
        if "capabilities" not in msg:
            errors.append("Missing required field: capabilities")
        elif not isinstance(msg["capabilities"], dict):
            errors.append("capabilities must be a dict")
        return errors
