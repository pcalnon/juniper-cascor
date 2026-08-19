"""W-6 (CLI experimentation plan §11 / H-4+H-5) — ``JUNIPER_CASCOR_SNAPSHOTS_DIR`` override.

Service tier: ``TrainingLifecycleManager._get_snapshots_dir`` must honour the env
override at call time (created on demand), fall back to the legacy
``<repo>/src/snapshots`` when unset, and treat a set-but-blank value as unset
(the blank-env guard class). Direct-CLI tier: ``constants_hdf5`` resolves
``_HDF5_PROJECT_SNAPSHOTS_DIR`` from the same env var at import time — pinned via
a module re-exec so the test does not depend on this process's import order.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.lifecycle.manager import TrainingLifecycleManager

pytestmark = pytest.mark.unit

_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent
_CONSTANTS_HDF5 = _SRC_DIR / "cascor_constants" / "constants_hdf5" / "constants_hdf5.py"


@pytest.fixture
def mgr():
    m = TrainingLifecycleManager()
    try:
        yield m
    finally:
        m.shutdown()


def _reexec_constants_hdf5(env_value: "str | None", monkeypatch) -> Path:
    """Re-execute constants_hdf5.py under a controlled env and return its snapshots dir.

    A fresh module object (not ``importlib.reload``) so the pin is hermetic to
    this test regardless of what the suite already imported.
    """
    if env_value is None:
        monkeypatch.delenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", raising=False)
    else:
        monkeypatch.setenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", env_value)
    spec = importlib.util.spec_from_file_location("_w6_constants_hdf5_probe", _CONSTANTS_HDF5)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return Path(module._HDF5_PROJECT_SNAPSHOTS_DIR)


class TestServiceTierOverride:
    def test_env_override_wins_and_is_created(self, mgr, tmp_path, monkeypatch):
        target = tmp_path / "run" / "snapshots"
        monkeypatch.setenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", str(target))
        resolved = mgr._get_snapshots_dir()
        assert resolved == target
        assert resolved.is_dir()

    def test_unset_falls_back_to_legacy_src_snapshots(self, mgr, monkeypatch):
        monkeypatch.delenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", raising=False)
        resolved = mgr._get_snapshots_dir()
        assert resolved == _SRC_DIR.parent / "snapshots"

    def test_blank_value_is_treated_as_unset(self, mgr, monkeypatch):
        monkeypatch.setenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", "   ")
        resolved = mgr._get_snapshots_dir()
        assert resolved == _SRC_DIR.parent / "snapshots"

    def test_call_time_read_not_cached(self, mgr, tmp_path, monkeypatch):
        """Two calls under different env values resolve differently — the
        override is read per call, never captured at import/instance time."""
        first = tmp_path / "a"
        second = tmp_path / "b"
        monkeypatch.setenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", str(first))
        assert mgr._get_snapshots_dir() == first
        monkeypatch.setenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", str(second))
        assert mgr._get_snapshots_dir() == second


class TestDirectCliTierOverride:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        target = tmp_path / "cli-snapshots"
        assert _reexec_constants_hdf5(str(target), monkeypatch) == target

    def test_unset_keeps_legacy_cascor_snapshots(self, monkeypatch):
        resolved = _reexec_constants_hdf5(None, monkeypatch)
        assert resolved == _SRC_DIR / "cascor_snapshots"

    def test_blank_value_is_treated_as_unset(self, monkeypatch):
        resolved = _reexec_constants_hdf5("", monkeypatch)
        assert resolved == _SRC_DIR / "cascor_snapshots"
