"""Unit tests for ``success_response`` envelope-level numpy coercion.

Pins the behaviour that ``success_response`` coerces numpy scalars in
``data`` so every cascor route returns a JSON-serializable envelope —
not just the routes that thread through ``get_training_params``.
Catches the pre-existing bug class where post-restore endpoints
(``GET /v1/network``, ``PATCH /v1/network/weights``,
``POST /v1/snapshots/{id}/restore``, etc.) returned 400
``VALIDATION_ERROR`` because pydantic-core couldn't serialize
``numpy.int64`` / ``numpy.float64`` carried through from h5py reads.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.models.common import coerce_native_scalars, error_response, success_response  # noqa: E402

pytestmark = pytest.mark.unit


class TestSuccessResponseCoercesData:
    """The envelope's ``data`` block must round-trip through JSON even
    when callers hand it raw numpy scalars (typical post-restore)."""

    def test_numpy_int64_data_is_serializable(self):
        envelope = success_response({"hidden_units": np.int64(2)})
        # ``json.dumps`` is the actual transport — if this raises, the
        # response would 500 in production with no useful detail.
        encoded = json.dumps(envelope)
        decoded = json.loads(encoded)
        assert decoded["data"]["hidden_units"] == 2
        assert type(decoded["data"]["hidden_units"]) is int

    def test_numpy_float64_data_is_serializable(self):
        envelope = success_response({"learning_rate": np.float64(0.01)})
        encoded = json.dumps(envelope)
        decoded = json.loads(encoded)
        assert decoded["data"]["learning_rate"] == pytest.approx(0.01)
        assert type(decoded["data"]["learning_rate"]) is float

    def test_nested_numpy_scalars(self):
        # Mirrors the actual unified-response payload shape after
        # /v1/snapshots/{id}/restore.
        payload = {
            "operation": "restore",
            "fsm_state": "Investigating",
            "training_params": {
                "max_hidden_units": np.int64(2),
                "learning_rate": np.float64(0.01),
            },
            "time_index": {
                "snapshot_window": {
                    "start_epoch": 0,
                    "end_epoch": np.int64(50),
                },
            },
        }
        envelope = success_response(payload)
        encoded = json.dumps(envelope)
        decoded = json.loads(encoded)
        assert decoded["data"]["training_params"]["max_hidden_units"] == 2
        assert decoded["data"]["time_index"]["snapshot_window"]["end_epoch"] == 50

    def test_list_with_numpy_scalars(self):
        envelope = success_response({"epochs": [np.int64(1), np.int64(2), 3]})
        encoded = json.dumps(envelope)
        decoded = json.loads(encoded)
        assert decoded["data"]["epochs"] == [1, 2, 3]

    def test_envelope_shape_unchanged(self):
        envelope = success_response({"x": np.int64(1)})
        # Same envelope shape as before — coercion is invisible to
        # well-behaved (Python-native) callers.
        assert envelope["status"] == "success"
        assert "data" in envelope
        assert "meta" in envelope
        assert "timestamp" in envelope["meta"]
        assert "version" in envelope["meta"]


class TestSuccessResponseUnchangedForNativeData:
    """Pre-existing callers that already pass Python natives must see
    no behaviour change."""

    def test_dict_passes_through(self):
        envelope = success_response({"x": 1, "y": "hello", "z": True})
        assert envelope["data"] == {"x": 1, "y": "hello", "z": True}

    def test_list_passes_through(self):
        envelope = success_response([1, 2, 3])
        assert envelope["data"] == [1, 2, 3]

    def test_none_passes_through(self):
        envelope = success_response(None)
        assert envelope["data"] is None

    def test_default_no_data(self):
        envelope = success_response()
        assert envelope["data"] is None


class TestErrorResponseUntouched:
    """``error_response`` doesn't carry user-supplied data so it
    intentionally doesn't auto-coerce; pin that fact so a future
    refactor doesn't quietly start coercing error details too."""

    def test_error_response_shape(self):
        envelope = error_response("VALIDATION_ERROR", "bad input", "x must be int")
        assert envelope["status"] == "error"
        assert envelope["error"]["code"] == "VALIDATION_ERROR"
        assert envelope["error"]["message"] == "bad input"
        assert envelope["error"]["detail"] == "x must be int"


class TestCoerceHelperPublic:
    """``coerce_native_scalars`` is now public in ``api.models.common``;
    the same behavioural contract applies as the previously-private
    helper in ``lifecycle/manager.py``. ``test_coerce_native_scalars.py``
    covers the deep matrix; this is just the one-line re-import sanity
    check."""

    def test_helper_is_importable_from_common(self):
        # Smoke test: the import worked at module load. Verify the
        # function still does what it says.
        assert coerce_native_scalars(np.int64(7)) == 7
        assert type(coerce_native_scalars(np.int64(7))) is int
