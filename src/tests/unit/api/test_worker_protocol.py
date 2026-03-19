"""Tests for WebSocket worker wire protocol — binary frames, message builders, and validation."""

import struct

import numpy as np
import pytest

from api.workers.protocol import BinaryFrame, MessageType, WorkerProtocol


@pytest.mark.unit
class TestBinaryFrame:
    """Test binary frame encoding and decoding of numpy arrays."""

    def test_roundtrip_1d_float32(self):
        """1D float32 array roundtrips through encode/decode."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        decoded = BinaryFrame.decode(BinaryFrame.encode(arr))
        np.testing.assert_array_equal(arr, decoded)
        assert decoded.dtype == np.float32

    def test_roundtrip_2d_float32(self):
        """2D float32 array roundtrips through encode/decode."""
        arr = np.random.randn(100, 4).astype(np.float32)
        decoded = BinaryFrame.decode(BinaryFrame.encode(arr))
        np.testing.assert_array_almost_equal(arr, decoded)
        assert decoded.shape == (100, 4)

    def test_roundtrip_float64(self):
        """float64 array roundtrips correctly."""
        arr = np.array([1.0, 2.0], dtype=np.float64)
        decoded = BinaryFrame.decode(BinaryFrame.encode(arr))
        np.testing.assert_array_equal(arr, decoded)
        assert decoded.dtype == np.float64

    def test_roundtrip_int64(self):
        """int64 array roundtrips correctly."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        decoded = BinaryFrame.decode(BinaryFrame.encode(arr))
        np.testing.assert_array_equal(arr, decoded)

    def test_roundtrip_scalar(self):
        """Scalar value roundtrips (promoted to 1-d by ascontiguousarray)."""
        arr = np.array(3.14, dtype=np.float32)
        decoded = BinaryFrame.decode(BinaryFrame.encode(arr))
        # np.ascontiguousarray promotes 0-d to 1-d; value is preserved
        assert decoded.shape == (1,)
        assert float(decoded[0]) == pytest.approx(3.14, rel=1e-5)

    def test_roundtrip_large_array(self):
        """Large array roundtrips without error."""
        arr = np.random.randn(1000, 50).astype(np.float32)
        decoded = BinaryFrame.decode(BinaryFrame.encode(arr))
        np.testing.assert_array_almost_equal(arr, decoded)

    def test_roundtrip_preserves_shape(self):
        """3D array shape is preserved."""
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        decoded = BinaryFrame.decode(BinaryFrame.encode(arr))
        assert decoded.shape == (2, 3, 4)

    def test_decode_returns_owned_copy(self):
        """Decoded array is an owned copy, not a view into the buffer."""
        arr = np.array([1.0, 2.0], dtype=np.float32)
        encoded = BinaryFrame.encode(arr)
        decoded = BinaryFrame.decode(encoded)
        assert decoded.flags["OWNDATA"]

    def test_decode_too_short_raises(self):
        """Decoding a truncated frame raises ValueError."""
        with pytest.raises(ValueError, match="too short"):
            BinaryFrame.decode(b"\x00\x01")

    def test_decode_bad_shape_raises(self):
        """Unreasonable dimension count raises ValueError."""
        # ndim = 20 (exceeds limit of 10)
        data = struct.pack("<I", 20)
        with pytest.raises(ValueError, match="Unreasonable"):
            BinaryFrame.decode(data)

    def test_decode_data_size_mismatch_raises(self):
        """Data size mismatch raises ValueError."""
        arr = np.array([1.0, 2.0], dtype=np.float32)
        encoded = BinaryFrame.encode(arr)
        # Truncate the data portion
        with pytest.raises(ValueError, match="size mismatch"):
            BinaryFrame.decode(encoded[:-2])

    def test_c_contiguous_enforcement(self):
        """Non-contiguous input is made contiguous before encoding."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        transposed = arr.T  # Fortran order, not C-contiguous
        assert not transposed.flags["C_CONTIGUOUS"]
        decoded = BinaryFrame.decode(BinaryFrame.encode(transposed))
        np.testing.assert_array_equal(transposed, decoded)


@pytest.mark.unit
class TestMessageType:
    """Test message type enum."""

    def test_values(self):
        """All expected message types exist."""
        assert MessageType.REGISTER == "register"
        assert MessageType.HEARTBEAT == "heartbeat"
        assert MessageType.TASK_ASSIGN == "task_assign"
        assert MessageType.TASK_RESULT == "task_result"
        assert MessageType.ERROR == "error"


@pytest.mark.unit
class TestWorkerProtocolBuilders:
    """Test message builder functions."""

    def test_build_register(self):
        """Registration message has correct structure."""
        msg = WorkerProtocol.build_register("w1", {"cpu_cores": 4})
        assert msg["type"] == "register"
        assert msg["worker_id"] == "w1"
        assert msg["capabilities"]["cpu_cores"] == 4

    def test_build_heartbeat(self):
        """Heartbeat message has timestamp."""
        msg = WorkerProtocol.build_heartbeat("w1")
        assert msg["type"] == "heartbeat"
        assert msg["worker_id"] == "w1"
        assert "timestamp" in msg

    def test_build_task_assign(self):
        """Task assignment message has all fields."""
        msg = WorkerProtocol.build_task_assign(
            task_id="t1",
            round_id="r1",
            candidate_index=0,
            candidate_data={"input_size": 4},
            training_params={"epochs": 200},
            tensor_manifest={"weights": {"shape": [4], "dtype": "float32"}},
        )
        assert msg["type"] == "task_assign"
        assert msg["task_id"] == "t1"
        assert msg["candidate_index"] == 0
        assert msg["tensor_manifest"]["weights"]["shape"] == [4]

    def test_build_task_result(self):
        """Task result message has all required fields."""
        msg = WorkerProtocol.build_task_result(
            task_id="t1",
            candidate_id=0,
            candidate_uuid="uuid-1",
            correlation=0.85,
            success=True,
            epochs_completed=200,
            activation_name="sigmoid",
            all_correlations=[0.1, 0.5, 0.85],
            numerator=1.0,
            denominator=2.0,
            best_corr_idx=199,
            tensor_manifest={"weights": {"shape": [4], "dtype": "float32"}},
        )
        assert msg["type"] == "task_result"
        assert msg["correlation"] == 0.85
        assert msg["success"] is True

    def test_build_error(self):
        """Error message has error field."""
        msg = WorkerProtocol.build_error("something broke", details="details here")
        assert msg["type"] == "error"
        assert msg["error"] == "something broke"
        assert msg["details"] == "details here"

    def test_build_error_no_details(self):
        """Error message without details omits the field."""
        msg = WorkerProtocol.build_error("oops")
        assert "details" not in msg


@pytest.mark.unit
class TestValidateTaskResult:
    """Test task result schema validation (Section 12.7)."""

    def _valid_result(self):
        return {
            "type": "task_result",
            "task_id": "t1",
            "candidate_id": 0,
            "correlation": 0.85,
            "success": True,
            "epochs_completed": 200,
        }

    def test_valid_result_passes(self):
        """A well-formed result produces no errors."""
        errors = WorkerProtocol.validate_task_result(self._valid_result())
        assert errors == []

    def test_missing_field(self):
        """Missing required field produces error."""
        msg = self._valid_result()
        del msg["correlation"]
        errors = WorkerProtocol.validate_task_result(msg)
        assert any("correlation" in e for e in errors)

    def test_wrong_type(self):
        """Wrong type for field produces error."""
        msg = self._valid_result()
        msg["candidate_id"] = "not-an-int"
        errors = WorkerProtocol.validate_task_result(msg)
        assert any("candidate_id" in e for e in errors)

    def test_correlation_out_of_bounds_high(self):
        """Correlation > 1.0 produces error."""
        msg = self._valid_result()
        msg["correlation"] = 1.5
        errors = WorkerProtocol.validate_task_result(msg)
        assert any("out of bounds" in e for e in errors)

    def test_correlation_out_of_bounds_negative(self):
        """Negative correlation produces error."""
        msg = self._valid_result()
        msg["correlation"] = -0.1
        errors = WorkerProtocol.validate_task_result(msg)
        assert any("out of bounds" in e for e in errors)

    def test_correlation_boundary_values(self):
        """Correlation at exact boundaries passes."""
        msg = self._valid_result()
        msg["correlation"] = 0.0
        assert WorkerProtocol.validate_task_result(msg) == []
        msg["correlation"] = 1.0
        assert WorkerProtocol.validate_task_result(msg) == []


@pytest.mark.unit
class TestValidateTensors:
    """Test tensor validation against manifest (Section 12.7)."""

    def test_valid_tensors(self):
        """Matching tensors produce no errors."""
        manifest = {
            "weights": {"shape": [4], "dtype": "float32"},
            "bias": {"shape": [1], "dtype": "float32"},
        }
        tensors = {
            "weights": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "bias": np.array([0.5], dtype=np.float32),
        }
        errors = WorkerProtocol.validate_tensors(tensors, manifest)
        assert errors == []

    def test_missing_tensor(self):
        """Missing tensor produces error."""
        manifest = {"weights": {"shape": [4], "dtype": "float32"}}
        errors = WorkerProtocol.validate_tensors({}, manifest)
        assert any("Missing" in e for e in errors)

    def test_shape_mismatch(self):
        """Wrong shape produces error."""
        manifest = {"weights": {"shape": [4], "dtype": "float32"}}
        tensors = {"weights": np.zeros(3, dtype=np.float32)}
        errors = WorkerProtocol.validate_tensors(tensors, manifest)
        assert any("shape mismatch" in e for e in errors)

    def test_dtype_mismatch(self):
        """Wrong dtype produces error."""
        manifest = {"weights": {"shape": [4], "dtype": "float32"}}
        tensors = {"weights": np.zeros(4, dtype=np.float64)}
        errors = WorkerProtocol.validate_tensors(tensors, manifest)
        assert any("dtype mismatch" in e for e in errors)

    def test_nan_detection(self):
        """NaN values produce error."""
        manifest = {"weights": {"shape": [4], "dtype": "float32"}}
        arr = np.array([1.0, float("nan"), 3.0, 4.0], dtype=np.float32)
        errors = WorkerProtocol.validate_tensors({"weights": arr}, manifest)
        assert any("NaN" in e for e in errors)

    def test_inf_detection(self):
        """Inf values produce error."""
        manifest = {"weights": {"shape": [4], "dtype": "float32"}}
        arr = np.array([1.0, float("inf"), 3.0, 4.0], dtype=np.float32)
        errors = WorkerProtocol.validate_tensors({"weights": arr}, manifest)
        assert any("Inf" in e for e in errors)

    def test_weight_magnitude_check(self):
        """Excessive weight magnitude produces error."""
        manifest = {"weights": {"shape": [4], "dtype": "float32"}}
        arr = np.array([1.0, 200.0, 3.0, 4.0], dtype=np.float32)
        errors = WorkerProtocol.validate_tensors({"weights": arr}, manifest)
        assert any("magnitude" in e for e in errors)


@pytest.mark.unit
class TestValidateRegister:
    """Test registration message validation."""

    def test_valid_register(self):
        """Valid registration produces no errors."""
        msg = {"worker_id": "w1", "capabilities": {"cpu_cores": 4}}
        assert WorkerProtocol.validate_register(msg) == []

    def test_missing_worker_id(self):
        """Missing worker_id produces error."""
        msg = {"capabilities": {}}
        errors = WorkerProtocol.validate_register(msg)
        assert any("worker_id" in e for e in errors)

    def test_missing_capabilities(self):
        """Missing capabilities produces error."""
        msg = {"worker_id": "w1"}
        errors = WorkerProtocol.validate_register(msg)
        assert any("capabilities" in e for e in errors)

    def test_capabilities_wrong_type(self):
        """Non-dict capabilities produces error."""
        msg = {"worker_id": "w1", "capabilities": "not a dict"}
        errors = WorkerProtocol.validate_register(msg)
        assert any("dict" in e for e in errors)
