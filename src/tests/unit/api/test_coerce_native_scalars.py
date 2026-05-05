"""Unit tests for _coerce_native_scalars (snapshot-restore JSON-safety fix).

Regression coverage for the bug where ``POST /v1/snapshots/{id}/restore``
returned 400 with an opaque ``VALIDATION_ERROR`` because
``lifecycle.get_training_params()`` carried numpy scalar types
(``numpy.int64`` / ``numpy.float64``) through to the response payload.
pydantic-core's JSON serializer rejects those types with
``PydanticSerializationError``, which the cascor app's
``value_error_handler`` mapped to 400 with the detail field stripped.

These tests pin the helper's behaviour so a future "simplify the
helper" refactor doesn't regress the fix.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.lifecycle.manager import _coerce_native_scalars  # noqa: E402

pytestmark = pytest.mark.unit


class TestCoerceNumpyScalars:
    def test_numpy_int64_to_python_int(self):
        result = _coerce_native_scalars(np.int64(42))
        assert result == 42
        assert type(result) is int

    def test_numpy_int32_to_python_int(self):
        result = _coerce_native_scalars(np.int32(7))
        assert result == 7
        assert type(result) is int

    def test_numpy_float64_to_python_float(self):
        result = _coerce_native_scalars(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert type(result) is float

    def test_numpy_float32_to_python_float(self):
        result = _coerce_native_scalars(np.float32(1.5))
        assert result == pytest.approx(1.5)
        assert type(result) is float

    def test_numpy_bool_to_python_bool(self):
        result = _coerce_native_scalars(np.bool_(True))
        assert result is True
        assert type(result) is bool

    def test_zero_d_numpy_array(self):
        # 0-d arrays also support ``.item()`` and are common from
        # h5py attribute reads.
        result = _coerce_native_scalars(np.array(99, dtype=np.int64))
        assert result == 99
        assert type(result) is int


class TestCoerceContainers:
    def test_dict_recursive(self):
        payload = {
            "max_hidden_units": np.int64(2),
            "learning_rate": np.float64(0.01),
            "name": "demo",
        }
        result = _coerce_native_scalars(payload)
        assert result == {"max_hidden_units": 2, "learning_rate": 0.01, "name": "demo"}
        assert type(result["max_hidden_units"]) is int
        assert type(result["learning_rate"]) is float

    def test_nested_dict(self):
        payload = {"outer": {"inner": np.int64(5)}}
        result = _coerce_native_scalars(payload)
        assert result == {"outer": {"inner": 5}}
        assert type(result["outer"]["inner"]) is int

    def test_list_recursive(self):
        result = _coerce_native_scalars([np.int64(1), np.float64(2.0), 3])
        assert result == [1, 2.0, 3]

    def test_tuple_preserves_type(self):
        result = _coerce_native_scalars((np.int64(1), np.float64(2.0)))
        assert isinstance(result, tuple)
        assert result == (1, 2.0)

    def test_mixed_nesting(self):
        payload = {"a": [np.int64(1), {"b": np.float64(2.5)}]}
        result = _coerce_native_scalars(payload)
        assert result == {"a": [1, {"b": 2.5}]}


class TestCoercePassthrough:
    def test_python_int_passes_through(self):
        # Plain ints don't have ``.item()`` — they should pass
        # through unchanged.
        result = _coerce_native_scalars(42)
        assert result == 42
        assert type(result) is int

    def test_python_float_passes_through(self):
        result = _coerce_native_scalars(3.14)
        assert result == 3.14
        assert type(result) is float

    def test_string_passes_through(self):
        result = _coerce_native_scalars("hello")
        assert result == "hello"

    def test_none_passes_through(self):
        assert _coerce_native_scalars(None) is None

    def test_bool_passes_through(self):
        assert _coerce_native_scalars(True) is True
        assert _coerce_native_scalars(False) is False

    def test_empty_dict(self):
        assert _coerce_native_scalars({}) == {}

    def test_empty_list(self):
        assert _coerce_native_scalars([]) == []


class TestJSONSerializability:
    """The whole point: output must round-trip through ``json.dumps``."""

    def test_coerced_payload_is_json_serializable(self):
        # Exact shape of the bug: ``get_training_params`` after a
        # snapshot restore. Pre-fix this raised
        # ``TypeError: Object of type int64 is not JSON serializable``.
        payload = {
            "learning_rate": np.float64(0.01),
            "max_hidden_units": np.int64(2),
            "epochs_max": np.int64(100),
            "init_output_weights": "zero",
            "auto_snap_best": False,
        }
        coerced = _coerce_native_scalars(payload)
        encoded = json.dumps(coerced)
        decoded = json.loads(encoded)
        assert decoded["learning_rate"] == pytest.approx(0.01)
        assert decoded["max_hidden_units"] == 2
        assert decoded["epochs_max"] == 100

    def test_raw_numpy_payload_would_fail_json(self):
        # Sanity check the test premise — without the helper, the
        # payload is genuinely not JSON-serializable.
        with pytest.raises(TypeError):
            json.dumps({"x": np.int64(1)})
