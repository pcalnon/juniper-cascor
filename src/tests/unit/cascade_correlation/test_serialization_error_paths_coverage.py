#!/usr/bin/env python
"""Coverage for HDF5 serialization error / edge paths
(per-file coverage lift 5, C-5).

Drives the previously-uncovered arms of the ``CascadeCorrelationNetwork``
HDF5 helpers via a serializer seam that raises:

* ``_save_to_hdf5`` — the outer exception handler (serializer construction /
  save failure -> ``False``).
* ``_load_from_hdf5`` — the outer exception handler (-> ``None``).
* ``verify_hdf5_file`` — the outer exception handler (-> ``{"valid": False}``).
* ``save_object`` — the default-snapshot-directory branch + successful save.

All fast unit tests: the serializer is patched, so no real h5py I/O runs.

Note: ``list_hdf5_snapshots``'s success path used to be unreachable —
``HDF5Utils.list_hdf5_files`` was undefined, so every existing-directory call
raised ``AttributeError`` and fell through to the ``except`` -> ``return []``.
The helper now exists (``snapshots/snapshot_utils.py``) and the success path
is pinned by ``tests/unit/test_latent_defect_repairs.py``.
"""

from unittest.mock import patch

import pytest

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

pytestmark = pytest.mark.unit


class _Objectify:
    """Serializable stand-in with the ``__name__`` + ``get_uuid`` surface
    ``save_object`` reads."""

    __name__ = "LiftFiveObject"

    def get_uuid(self):
        return "obj-uuid-123"


class TestSaveToHdf5ErrorPath:
    def test_serializer_failure_returns_false(self, simple_network, tmp_path):
        target = tmp_path / "net.h5"
        with patch("snapshots.snapshot_serializer.CascadeHDF5Serializer", side_effect=RuntimeError("serializer down")):
            assert simple_network._save_to_hdf5(filepath=target) is False


class TestLoadFromHdf5ErrorPath:
    def test_serializer_failure_returns_none(self, tmp_path):
        target = tmp_path / "net.h5"
        target.write_bytes(b"stub")
        with patch("snapshots.snapshot_serializer.CascadeHDF5Serializer", side_effect=RuntimeError("serializer down")):
            assert CascadeCorrelationNetwork._load_from_hdf5(filepath=target) is None


class TestVerifyHdf5FileErrorPath:
    def test_serializer_failure_returns_invalid_dict(self, simple_network, tmp_path):
        target = tmp_path / "net.h5"
        with patch("snapshots.snapshot_serializer.CascadeHDF5Serializer", side_effect=RuntimeError("verify down")):
            result = simple_network.verify_hdf5_file(target)
        assert result["valid"] is False
        assert "error" in result


class TestSaveObjectDefaultDir:
    def test_default_snapshot_dir_and_successful_save(self, simple_network, tmp_path):
        net = simple_network
        net.cascade_correlation_network_snapshots_dir = str(tmp_path)
        with patch.object(net, "_save_to_hdf5", return_value=True):
            snapshot_path = net.save_object(objectify=_Objectify(), snapshot_dir=None)
        assert snapshot_path is not None
        assert snapshot_path.name.startswith("LiftFiveObject_snapshot_")
        assert snapshot_path.suffix == ".h5"
