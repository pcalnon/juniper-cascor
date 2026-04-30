"""Binary frame codec for ``/ws/v1/workers`` numpy tensor side-channel.

Wire format:

  ``[4 bytes: shape ndim (uint32, little-endian)]``
  ``[N * 4 bytes: shape values (uint32 each)]``
  ``[4 bytes: dtype string length (uint32)]``
  ``[M bytes: dtype string (utf-8)]``
  ``[remaining bytes: raw array data]``

Reconstructible from numpy + struct only — no pickle. Mirrors the
canonical implementation in ``juniper-cascor/src/api/workers/protocol.py``;
this module exists so the worker (and the server) single-source the
codec rather than maintain two byte-identical copies.

Importing this module does **not** load Pydantic.
"""

from __future__ import annotations

import struct

import numpy as np

# Validation bounds — mirror the cascor server's protocol.py limits so
# both ends agree on what counts as a malformed frame.
_MAX_FRAME_SIZE = 100 * 1024 * 1024  # 100 MB
_MAX_NDIM = 10
_MAX_DTYPE_LEN = 64

# struct format characters for the header.
_HEADER_LENGTH_FORMAT = "<I"
_HEADER_LENGTH_BYTES = 4


class BinaryFrame:
    """Encode/decode numpy arrays as binary WebSocket frames.

    All methods are static — :class:`BinaryFrame` is a namespace, not
    a stateful coder.
    """

    @staticmethod
    def encode(array: np.ndarray) -> bytes:
        """Encode a numpy array into a binary frame.

        Args:
            array: C-contiguous numpy array to encode.

        Returns:
            Binary frame bytes with shape + dtype header followed by raw
            tensor data.
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
            Reconstructed numpy array (owned copy, not a view).

        Raises:
            ValueError: If the frame is malformed or exceeds size limits.
        """
        if len(data) > _MAX_FRAME_SIZE:
            raise ValueError(f"Binary frame exceeds maximum size ({len(data)} > {_MAX_FRAME_SIZE})")

        offset = 0

        # Read shape ndim
        if len(data) < _HEADER_LENGTH_BYTES:
            raise ValueError("Binary frame too short for shape header")
        (ndim,) = struct.unpack_from(_HEADER_LENGTH_FORMAT, data, offset)
        offset += _HEADER_LENGTH_BYTES

        if ndim > _MAX_NDIM:
            raise ValueError(f"Unreasonable number of dimensions: {ndim}")

        # Read shape values
        if len(data) < offset + ndim * _HEADER_LENGTH_BYTES:
            raise ValueError("Binary frame too short for shape values")
        shape = struct.unpack_from(f"<{ndim}I", data, offset)
        offset += ndim * _HEADER_LENGTH_BYTES

        # Read dtype string length
        if len(data) < offset + _HEADER_LENGTH_BYTES:
            raise ValueError("Binary frame too short for dtype header")
        (dtype_len,) = struct.unpack_from(_HEADER_LENGTH_FORMAT, data, offset)
        offset += _HEADER_LENGTH_BYTES

        if dtype_len > _MAX_DTYPE_LEN:
            raise ValueError(f"Unreasonable dtype string length: {dtype_len}")
        if len(data) < offset + dtype_len:
            raise ValueError("Binary frame too short for dtype string")

        dtype_str = data[offset : offset + dtype_len].decode("utf-8")
        offset += dtype_len

        try:
            dtype = np.dtype(dtype_str)
        except TypeError as exc:
            raise ValueError(f"Invalid dtype string: {dtype_str!r}") from exc

        # Read array data
        expected_size = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
        actual_size = len(data) - offset
        if actual_size != expected_size:
            raise ValueError(f"Data size mismatch: expected {expected_size} bytes, got {actual_size}")

        array = np.frombuffer(data[offset:], dtype=dtype).reshape(shape)
        return array.copy()
