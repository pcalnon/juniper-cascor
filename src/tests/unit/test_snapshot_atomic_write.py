"""
A failed snapshot save must leave NO artifact at the destination.

Until 2026-08-23 ``save_network`` opened ``h5py.File(filepath, "w")`` straight at the
destination, so a save that was interrupted -- or that raised partway through -- left a
PARTIAL ``.h5`` under the final name, indistinguishable from a good snapshot without
opening it.

That is the sole root cause of every structurally-incomplete file in the live archive.
Measured 2026-08-22 (juniper-ml#1254): **273 of 27,908**, and every one a strict PREFIX of
the fixed write order (``root -> meta -> config -> arch -> params -> hidden_units ->
random -> mp``) --

    6    no groups at all      (died before the root attributes)
    2    config + meta only    (died before ``arch``)
    265  ``unit_0`` and no more (died inside ``_save_hidden_units``)

The 265 are **irrecoverable**. ``_save_hidden_units`` writes ``num_units`` before the unit
loop, so they declare 5/10/20/50 hidden units, agree with their own output-layer geometry,
and contain one. Nothing in the file can reconstruct the others.

WHY THE TESTS INJECT FAILURE MID-WRITE
    Testing "does a good save work" cannot catch this class -- the old code passed that.
    The defect only shows when the write DIES PARTWAY, so each test below fails a specific
    stage of ``_save_network_objects_helper`` and then asserts what is on disk. That is the
    archive's actual failure mode reproduced, not an approximation of it.
"""

import os
import sys

import h5py
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork  # noqa: E402
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig  # noqa: E402
from snapshots.snapshot_errors import SnapshotSaveError  # noqa: E402
from snapshots.snapshot_serializer import CascadeHDF5Serializer  # noqa: E402

pytestmark = pytest.mark.unit


def _network(hidden=0):
    torch.manual_seed(17)
    config = CascadeCorrelationConfig(input_size=2, output_size=2, random_seed=17)
    return CascadeCorrelationNetwork(config=config)


def _leftovers(directory):
    """Everything in the directory, including the dot-prefixed temporaries."""
    return sorted(p.name for p in directory.iterdir())


class TestSuccessfulSaveIsUnchanged:
    """The fix must be invisible on the happy path -- ~27.6k archive files load through it."""

    def test_the_snapshot_lands_at_the_destination(self, tmp_path):
        path = tmp_path / "good.h5"
        assert CascadeHDF5Serializer().save_network(_network(), str(path))
        assert path.is_file()

    def test_it_round_trips(self, tmp_path):
        path = tmp_path / "good.h5"
        serializer = CascadeHDF5Serializer()
        serializer.save_network(_network(), str(path))
        network = serializer.load_network(str(path), restore_multiprocessing=False)
        assert network is not None
        assert network.output_size == 2

    def test_no_temporary_survives(self, tmp_path):
        path = tmp_path / "good.h5"
        CascadeHDF5Serializer().save_network(_network(), str(path))
        assert _leftovers(tmp_path) == ["good.h5"], "the temporary must be renamed away, not left beside the snapshot"

    def test_missing_parent_directory_is_still_created(self, tmp_path):
        path = tmp_path / "nested" / "deeper" / "good.h5"
        assert CascadeHDF5Serializer().save_network(_network(), str(path))
        assert path.is_file()


class TestFailedSaveLeavesNothing:
    """The archive's actual failure mode: the write dies partway through."""

    def _fail_during(self, monkeypatch, method_name):
        """Make one stage of the write raise, exactly as an interrupted save would."""

        def _boom(*args, **kwargs):
            raise RuntimeError(f"simulated failure in {method_name}")

        monkeypatch.setattr(CascadeHDF5Serializer, method_name, _boom)

    @pytest.mark.parametrize(
        "stage",
        [
            "_save_metadata",  # the 6 empty files
            "_save_architecture",  # the 2 config+meta files
            "_save_hidden_units",  # the 265 -- the irrecoverable cohort
            "_save_random_state",
        ],
    )
    def test_no_file_appears_at_the_destination(self, tmp_path, monkeypatch, stage):
        path = tmp_path / "doomed.h5"
        self._fail_during(monkeypatch, stage)
        with pytest.raises(SnapshotSaveError):
            CascadeHDF5Serializer().save_network(_network(), str(path))
        assert not path.exists(), f"a save that died in {stage} must leave NO snapshot at the destination"

    @pytest.mark.parametrize("stage", ["_save_metadata", "_save_hidden_units"])
    def test_no_temporary_is_left_behind(self, tmp_path, monkeypatch, stage):
        path = tmp_path / "doomed.h5"
        self._fail_during(monkeypatch, stage)
        with pytest.raises(SnapshotSaveError):
            CascadeHDF5Serializer().save_network(_network(), str(path))
        assert _leftovers(tmp_path) == [], f"the temporary must be cleaned up when {stage} fails"

    def test_the_error_still_names_the_reason(self, tmp_path, monkeypatch):
        """C1/I-3: a failed save must not be indistinguishable from 'no network'."""
        path = tmp_path / "doomed.h5"
        self._fail_during(monkeypatch, "_save_architecture")
        with pytest.raises(SnapshotSaveError, match="simulated failure"):
            CascadeHDF5Serializer().save_network(_network(), str(path))


class TestAnExistingSnapshotSurvivesAFailedOverwrite:
    """The strongest property the rename buys, and the one a partial-write cannot give.

    Before this, re-saving over an existing snapshot truncated it at ``open`` -- so a
    failure midway destroyed the GOOD file that was already there and replaced it with a
    fragment. Now a failed overwrite is a no-op.
    """

    def test_the_previous_snapshot_is_intact_after_a_failed_overwrite(self, tmp_path, monkeypatch):
        path = tmp_path / "keeper.h5"
        serializer = CascadeHDF5Serializer()
        serializer.save_network(_network(), str(path))
        original_bytes = path.read_bytes()

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(CascadeHDF5Serializer, "_save_hidden_units", _boom)
        with pytest.raises(SnapshotSaveError):
            CascadeHDF5Serializer().save_network(_network(), str(path))

        assert path.read_bytes() == original_bytes, "a failed overwrite must not damage the snapshot already on disk"

    def test_and_it_still_loads(self, tmp_path, monkeypatch):
        path = tmp_path / "keeper.h5"
        serializer = CascadeHDF5Serializer()
        serializer.save_network(_network(), str(path))

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(CascadeHDF5Serializer, "_save_architecture", _boom)
        with pytest.raises(SnapshotSaveError):
            CascadeHDF5Serializer().save_network(_network(), str(path))

        assert CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False) is not None


class TestTemporaryIsNotMistakenForASnapshot:
    """A temp file left by a hard kill (SIGKILL -- no exception, no cleanup) must not be
    picked up by anything that selects snapshots.

    ``juniper-ml/util/snapshot_index.py`` scans ``suffix == ".h5"`` and ``snapshot_utils``
    globs ``cascor_snapshot_*.h5``. A temporary named ``*.h5`` would be scanned, counted and
    cleanup-eligible as though it were a snapshot -- reintroducing exactly the confusion
    this fix removes.
    """

    def test_the_temporary_does_not_end_in_h5(self, tmp_path):
        seen = {}
        real_open = h5py.File

        def _record(name, mode, *args, **kwargs):
            if mode == "w":
                seen["path"] = str(name)
            return real_open(name, mode, *args, **kwargs)

        import snapshots.snapshot_serializer as module

        original = module.h5py.File
        module.h5py.File = _record
        try:
            CascadeHDF5Serializer().save_network(_network(), str(tmp_path / "final.h5"))
        finally:
            module.h5py.File = original

        assert "path" in seen, "the save must have opened a file for writing"
        assert not seen["path"].endswith(".h5"), f"the temporary must not look like a snapshot: {seen['path']}"
        assert seen["path"] != str(tmp_path / "final.h5"), "the write must not go straight at the destination"

    def test_two_concurrent_writers_to_one_path_do_not_collide(self, tmp_path):
        """The temp name carries pid + a random suffix precisely so this is safe."""
        path = tmp_path / "shared.h5"
        names = set()
        real_open = h5py.File

        def _record(name, mode, *args, **kwargs):
            if mode == "w":
                names.add(str(name))
            return real_open(name, mode, *args, **kwargs)

        import snapshots.snapshot_serializer as module

        original = module.h5py.File
        module.h5py.File = _record
        try:
            for _ in range(4):
                CascadeHDF5Serializer().save_network(_network(), str(path))
        finally:
            module.h5py.File = original

        assert len(names) == 4, f"each write must use a distinct temporary; got {names}"
