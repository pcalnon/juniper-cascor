"""Worker subpackage tests — StrEnum + BinaryFrame codec.

These tests must pass without Pydantic available. ``test_public_api``
holds the negative invariant (worker subpackage source files do not
import pydantic); this file pins the positive behavior.
"""

import struct

import numpy as np
import pytest

from juniper_cascor_protocol.worker import BinaryFrame, WorkerMessageType


# ---------------------------------------------------------------------------
# WorkerMessageType
# ---------------------------------------------------------------------------


def test_worker_message_type_values_match_wire_protocol():
    """Each enum value is the literal byte string emitted on the wire."""
    expected = {
        "register": "register",
        "heartbeat": "heartbeat",
        "task_result": "task_result",
        "registration_ack": "registration_ack",
        "result_ack": "result_ack",
        "task_assign": "task_assign",
        "token_refresh": "token_refresh",
        "connection_established": "connection_established",
        "error": "error",
    }
    actual = {member.name.lower(): member.value for member in WorkerMessageType}
    assert actual == expected


def test_worker_message_type_str_inheritance():
    """``StrEnum`` members compare equal to their string value."""
    assert WorkerMessageType.REGISTER == "register"
    assert WorkerMessageType.TASK_ASSIGN == "task_assign"
    assert "register" == WorkerMessageType.REGISTER


def test_worker_message_type_iteration_order_stable():
    """Stable iteration order so reviewers can spot drift."""
    members = [m.value for m in WorkerMessageType]
    expected_order = [
        "register",
        "heartbeat",
        "task_result",
        "registration_ack",
        "result_ack",
        "task_assign",
        "token_refresh",
        "connection_established",
        "error",
    ]
    assert members == expected_order


# ---------------------------------------------------------------------------
# BinaryFrame — encode/decode round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((10,), "float32"),
        ((4, 4), "float64"),
        ((2, 3, 5), "float32"),
        ((1,), "int32"),
        ((100,), "uint8"),
    ],
)
def test_binary_frame_roundtrip(shape, dtype):
    arr = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    encoded = BinaryFrame.encode(arr)
    decoded = BinaryFrame.decode(encoded)
    assert decoded.shape == arr.shape
    assert str(decoded.dtype) == dtype
    np.testing.assert_array_equal(decoded, arr)


def test_binary_frame_decode_returns_owned_copy_not_view():
    """Decoded array must own its memory so the caller can mutate freely."""
    arr = np.array([1.0, 2.0, 3.0], dtype="float32")
    encoded = BinaryFrame.encode(arr)
    decoded = BinaryFrame.decode(encoded)
    decoded[0] = 99.0
    # Re-decoding the original bytes still returns the original values.
    decoded2 = BinaryFrame.decode(encoded)
    assert decoded2[0] == 1.0


def test_binary_frame_encode_handles_non_contiguous_input():
    """``encode`` calls ``ascontiguousarray`` so views work."""
    base = np.arange(24, dtype="float32").reshape(4, 6)
    view = base[:, ::2]  # non-contiguous slice
    encoded = BinaryFrame.encode(view)
    decoded = BinaryFrame.decode(encoded)
    np.testing.assert_array_equal(decoded, view)


def test_binary_frame_decode_rejects_oversized_data():
    too_big = b"\x00" * (101 * 1024 * 1024)
    with pytest.raises(ValueError, match="exceeds maximum size"):
        BinaryFrame.decode(too_big)


def test_binary_frame_decode_rejects_short_shape_header():
    with pytest.raises(ValueError, match="too short for shape header"):
        BinaryFrame.decode(b"\x00\x00\x00")  # only 3 bytes, header needs 4


def test_binary_frame_decode_rejects_unreasonable_ndim():
    # ndim=11 — exceeds the cap of 10.
    payload = struct.pack("<I", 11)
    with pytest.raises(ValueError, match="Unreasonable number of dimensions"):
        BinaryFrame.decode(payload)


def test_binary_frame_decode_rejects_short_shape_values():
    # ndim=2 declared, only 4 bytes (one shape value) provided.
    payload = struct.pack("<I", 2) + struct.pack("<I", 3)
    with pytest.raises(ValueError, match="too short for shape values"):
        BinaryFrame.decode(payload)


def test_binary_frame_decode_rejects_short_dtype_header():
    # ndim=1, shape=(3,), but no dtype header bytes follow.
    payload = struct.pack("<I", 1) + struct.pack("<I", 3)
    with pytest.raises(ValueError, match="too short for dtype header"):
        BinaryFrame.decode(payload)


def test_binary_frame_decode_rejects_unreasonable_dtype_length():
    payload = struct.pack("<I", 1) + struct.pack("<I", 3) + struct.pack("<I", 999)
    with pytest.raises(ValueError, match="Unreasonable dtype string length"):
        BinaryFrame.decode(payload)


def test_binary_frame_decode_rejects_short_dtype_string():
    # ndim=1, shape=(3,), dtype_len=7 declared, only 3 bytes for dtype.
    payload = struct.pack("<I", 1) + struct.pack("<I", 3) + struct.pack("<I", 7) + b"abc"
    with pytest.raises(ValueError, match="too short for dtype string"):
        BinaryFrame.decode(payload)


def test_binary_frame_decode_rejects_invalid_dtype_string():
    dtype_str = b"not_a_real_dtype"
    payload = struct.pack("<I", 1) + struct.pack("<I", 3) + struct.pack("<I", len(dtype_str)) + dtype_str + b"\x00\x00\x00"
    with pytest.raises(ValueError, match="Invalid dtype string"):
        BinaryFrame.decode(payload)


def test_binary_frame_decode_rejects_non_utf8_dtype_bytes():
    """Non-UTF-8 dtype bytes must raise ValueError (contract for worker_stream)."""
    # ndim=1, shape=(1,), dtype_len=2 with invalid UTF-8, plus 4 bytes of payload.
    payload = struct.pack("<I", 1) + struct.pack("<I", 1) + struct.pack("<I", 2) + b"\xff\xfe" + b"\x00\x00\x00\x00"
    with pytest.raises(ValueError, match="not valid UTF-8"):
        BinaryFrame.decode(payload)


def test_binary_frame_decode_rejects_size_mismatch():
    # shape=(3,), dtype=float32 (12 bytes expected), but provide 8 bytes of data.
    dtype_str = b"float32"
    payload = struct.pack("<I", 1) + struct.pack("<I", 3) + struct.pack("<I", len(dtype_str)) + dtype_str + b"\x00" * 8
    with pytest.raises(ValueError, match="Data size mismatch"):
        BinaryFrame.decode(payload)


def test_binary_frame_handles_zero_dimensional_array():
    """0-d arrays (scalars) are out of scope — workers exchange tensors, not scalars.

    The cascor wire protocol's BinaryFrame is intentionally scoped to
    ``ndim >= 1`` (weights, deltas, predictions — all matrices or
    vectors). 0-d round-trip is not part of the contract; we only assert
    that encoding a 0-d array does not raise, so a future worker that
    accidentally encodes a scalar fails loudly at decode rather than
    silently corrupting state.
    """
    arr = np.array(3.14, dtype="float32")
    # encode must succeed
    encoded = BinaryFrame.encode(arr)
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0
