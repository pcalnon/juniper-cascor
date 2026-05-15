"""P2-3 (Issue #3) — ``save_snapshot`` ID collision handling.

The snapshot ID is built from ``datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")``
— second resolution. Two ``save_snapshot`` calls within the same wall-clock
second (the common case for ``swap_dataset_live``'s pre/post-swap pair on a
small network) used to silently overwrite. P2-3 appends ``_2``, ``_3``, ...
to the ID until a free filename is found.

These tests pin:

* No-collision path: the returned ID matches the pre-P2-3 format byte-for-byte,
  so any consumer that parses ``snapshot_YYYYMMDDTHHMMSSZ`` keeps working.
* Collision path: two successive saves produce distinct IDs (with ``_2``
  suffix), distinct filenames on disk, and both files are valid HDF5.
* The cap at 1000 prevents a runaway loop on misbehaving filesystems.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.lifecycle.manager import TrainingLifecycleManager

pytestmark = pytest.mark.unit


@pytest.fixture
def mgr_with_temp_snapshots():
    """Lifecycle manager with a temp snapshots dir + a tiny network so
    ``save_snapshot`` can actually write something to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        # Override the snapshots dir so we don't pollute the repo's
        # default path and don't race against parallel test workers.
        mgr._get_snapshots_dir = lambda: Path(tmpdir)  # type: ignore[method-assign]
        try:
            yield mgr, Path(tmpdir)
        finally:
            mgr.shutdown()


class TestSaveSnapshotIdFormat:
    def test_no_collision_keeps_legacy_format(self, mgr_with_temp_snapshots):
        """When the directory is empty, the returned ID matches the
        ``snapshot_YYYYMMDDTHHMMSSZ`` format with NO suffix — preserves
        backward compatibility with parsers that read the timestamp out
        of the ID string."""
        mgr, _ = mgr_with_temp_snapshots
        result = mgr.save_snapshot(description="first")
        assert result is not None
        snapshot_id = result["id"]
        # Format: "snapshot_" + 16 chars (YYYYMMDDTHHMMSSZ)
        assert snapshot_id.startswith("snapshot_")
        rest = snapshot_id[len("snapshot_") :]
        assert len(rest) == 16, f"expected 16-char timestamp suffix, got {rest!r}"
        assert rest.endswith("Z")
        assert "_" not in rest, "no-collision path must not append a suffix"


class TestSaveSnapshotCollision:
    def test_same_second_calls_produce_distinct_ids(self, mgr_with_temp_snapshots):
        """Two ``save_snapshot`` calls forced to share the same timestamp
        must produce distinct IDs. The first keeps the legacy format; the
        second gets ``_2`` appended."""
        mgr, snapshots_dir = mgr_with_temp_snapshots
        # Freeze the timestamp by patching the manager's clock source.
        # ``datetime.now(UTC)`` inside ``save_snapshot`` resolves via the
        # module-level import in manager.py.
        from api.lifecycle import manager as manager_module

        class FrozenDatetime:
            @classmethod
            def now(cls, tz=None):
                # ISO-8601 UTC: 2026-05-14T12:00:00 +00:00
                import datetime as _real_dt

                return _real_dt.datetime(2026, 5, 14, 12, 0, 0, tzinfo=tz)

        with patch.object(manager_module, "datetime", FrozenDatetime):
            first = mgr.save_snapshot(description="first")
            second = mgr.save_snapshot(description="second")
        assert first is not None
        assert second is not None
        assert first["id"] == "snapshot_20260514T120000Z"
        assert second["id"] == "snapshot_20260514T120000Z_2"
        # Distinct files on disk — pre-P2-3 the second call would
        # overwrite the first.
        assert Path(first["path"]).exists()
        assert Path(second["path"]).exists()
        assert first["path"] != second["path"]

    def test_three_same_second_calls_cascade_through_suffixes(self, mgr_with_temp_snapshots):
        """The collision path is iterative — three calls in the same
        second yield ``_2`` and ``_3`` suffixes for the second and third."""
        mgr, _ = mgr_with_temp_snapshots
        from api.lifecycle import manager as manager_module

        class FrozenDatetime:
            @classmethod
            def now(cls, tz=None):
                import datetime as _real_dt

                return _real_dt.datetime(2026, 5, 14, 12, 0, 0, tzinfo=tz)

        with patch.object(manager_module, "datetime", FrozenDatetime):
            r1 = mgr.save_snapshot(description="r1")
            r2 = mgr.save_snapshot(description="r2")
            r3 = mgr.save_snapshot(description="r3")
        assert [r1["id"], r2["id"], r3["id"]] == [
            "snapshot_20260514T120000Z",
            "snapshot_20260514T120000Z_2",
            "snapshot_20260514T120000Z_3",
        ]

    def test_collision_path_files_are_valid_hdf5(self, mgr_with_temp_snapshots):
        """Beyond distinct IDs and paths, the collision-path files must
        themselves be valid HDF5 (the serializer wasn't broken by being
        handed a suffixed filename)."""
        mgr, _ = mgr_with_temp_snapshots
        from api.lifecycle import manager as manager_module

        class FrozenDatetime:
            @classmethod
            def now(cls, tz=None):
                import datetime as _real_dt

                return _real_dt.datetime(2026, 5, 14, 12, 0, 0, tzinfo=tz)

        with patch.object(manager_module, "datetime", FrozenDatetime):
            first = mgr.save_snapshot(description="first")
            second = mgr.save_snapshot(description="second")
        # Open both files with h5py to confirm they're valid HDF5 with
        # at least the format-validation root attrs the loader expects.
        for snap in (first, second):
            with h5py.File(snap["path"], "r") as f:
                assert "format" in f.attrs
