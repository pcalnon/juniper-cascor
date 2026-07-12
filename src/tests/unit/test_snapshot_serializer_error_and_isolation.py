"""C1 (I-3) — serializer failure typing + training-history write isolation.

Plan of record: juniper-ml
``notes/JUNIPER_2026-07-11_JUNIPER-CANOPY_TRAINING-RUNTIME-DEFECTS-PLAN.md``
§4 I-3 / §7 C1.

* ``save_network`` raises ``SnapshotSaveError`` (underlying cause chained)
  instead of swallowing every exception into ``False``.
* ``_save_training_history`` serializes from ``_snapshot_history_view``'s
  point-in-time copy, so a mid-training save cannot crash on concurrent
  history mutation by the training thread. The helper's copy contract is
  pinned deterministically; a threaded append loop exercises the real save
  path as behavioral evidence; a source pin (same style as the route
  coverage suite's PERF-CC-01 invariant) keeps the writer on the view.

All writes go to pytest ``tmp_path`` — ``src/snapshots/`` holds committed
.h5 artifacts and must never be written by tests.
"""

import inspect
import os
import sys
import threading

import h5py
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from snapshots.snapshot_errors import SnapshotSaveError
from snapshots.snapshot_serializer import CascadeHDF5Serializer

pytestmark = pytest.mark.unit


@pytest.fixture
def serializer():
    return CascadeHDF5Serializer()


@pytest.fixture
def tiny_network():
    config = CascadeCorrelationConfig(input_size=2, output_size=1)
    return CascadeCorrelationNetwork(config=config)


# ---------------------------------------------------------------------------
# Failure typing
# ---------------------------------------------------------------------------


class TestSaveNetworkFailureTyping:
    """save_network raises SnapshotSaveError with the underlying reason."""

    def test_write_failure_raises_snapshot_save_error(self, serializer, tiny_network, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("a file where a directory must go")
        target = blocker / "sub" / "snap.h5"  # parent mkdir must fail
        with pytest.raises(SnapshotSaveError) as excinfo:
            serializer.save_network(tiny_network, target, include_training_state=True)
        assert excinfo.value.__cause__ is not None, "the underlying exception must be chained for diagnosis"
        assert str(excinfo.value), "the error must carry a human-readable reason"

    def test_success_still_returns_true(self, serializer, tiny_network, tmp_path):
        target = tmp_path / "snap.h5"
        assert serializer.save_network(tiny_network, target, include_training_state=True) is True
        assert target.exists()


# ---------------------------------------------------------------------------
# History view contract (write isolation)
# ---------------------------------------------------------------------------


class TestHistoryViewContract:
    """_snapshot_history_view returns an independent, point-in-time copy."""

    def test_view_lists_are_independent_copies(self, tiny_network):
        tiny_network.history["train_loss"].extend([0.5, 0.4])
        view = CascadeHDF5Serializer._snapshot_history_view(tiny_network)
        assert view["train_loss"] == [0.5, 0.4]
        assert view["train_loss"] is not tiny_network.history["train_loss"]
        tiny_network.history["train_loss"].append(0.3)
        assert view["train_loss"] == [0.5, 0.4], "appends after the view is taken must not be visible in the view"

    def test_view_covers_every_history_key(self, tiny_network):
        view = CascadeHDF5Serializer._snapshot_history_view(tiny_network)
        assert set(view.keys()) == set(tiny_network.history.keys())
        for key, value in tiny_network.history.items():
            if isinstance(value, list):
                assert view[key] is not value, f"list under {key!r} must be copied, not aliased"

    def test_view_handles_missing_or_empty_history(self):
        class NoHistory:
            pass

        class EmptyHistory:
            history = {}

        assert CascadeHDF5Serializer._snapshot_history_view(NoHistory()) == {}
        assert CascadeHDF5Serializer._snapshot_history_view(EmptyHistory()) == {}

    def test_view_passes_non_list_values_through(self):
        class Weird:
            history = {"train_loss": [1.0], "note": "not-a-list"}

        view = CascadeHDF5Serializer._snapshot_history_view(Weird())
        assert view["note"] == "not-a-list"
        assert view["train_loss"] == [1.0]

    def test_save_training_history_reads_from_view(self):
        """Source pin (same style as the route suite's PERF-CC-01 invariant):
        the history writer must serialize from the point-in-time view, never
        index into the live history dict on the network."""
        source = inspect.getsource(CascadeHDF5Serializer._save_training_history)
        assert "_snapshot_history_view(" in source
        assert "network.history[" not in source


# ---------------------------------------------------------------------------
# Concurrent-append behavioral evidence
# ---------------------------------------------------------------------------


class TestConcurrentAppendDuringSave:
    """A training-thread-shaped writer appending per-epoch entries while
    save_network runs must not crash the save (the I-3 latent hazard:
    ``save_snapshot`` takes no lock and pre-C1 the serializer iterated the
    live lists during slow HDF5 writes)."""

    def test_history_appends_during_save_do_not_crash(self, serializer, tiny_network, tmp_path):
        stop = threading.Event()
        error: list = []

        def churn():
            i = 0
            try:
                while not stop.is_set() and i < 200_000:
                    tiny_network.history["train_loss"].append(0.1 * i)
                    tiny_network.history["value_loss"].append(0.2 * i)
                    tiny_network.history["train_accuracy"].append(0.3)
                    tiny_network.history["value_accuracy"].append(0.4)
                    if i % 500 == 0:
                        tiny_network.history["hidden_units_added"].append({"correlation": 0.0, "weight_shape": (2,), "unit_index": i})
                    i += 1
            except Exception as exc:  # pragma: no cover - diagnostic aid
                error.append(exc)

        writer = threading.Thread(target=churn, name="history-churn", daemon=True)
        writer.start()
        try:
            for n in range(5):
                target = tmp_path / f"snap_{n}.h5"
                assert serializer.save_network(tiny_network, target, include_training_state=True) is True, f"save {n} failed under concurrent history appends"
                with h5py.File(target, "r") as f:
                    assert "history" in f, f"save {n} produced no history group"
        finally:
            stop.set()
            writer.join(timeout=10)
        assert not writer.is_alive(), "churn thread failed to stop"
        assert not error, f"churn thread raised: {error}"
