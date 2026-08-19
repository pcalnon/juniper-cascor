"""Env-var directory overrides in the extracted ``cascor_constants`` copy.

These mirror ``juniper-cascor/src/tests/unit/api/test_w6_snapshots_dir_override.py``. The
package copy needs its own coverage because the overrides resolve at **import time**, and
because this package is what ``site-packages`` resolves ``cascor_constants`` to -- so a
consumer importing it from outside the ``src/`` checkout gets these code paths, not src's.

Both overrides previously diverged here:

* **W-6** ``JUNIPER_CASCOR_SNAPSHOTS_DIR`` was absent from this copy entirely, so snapshots
  landed in the source tree no matter what the launcher exported.
* **Q-6** ``JUNIPER_CASCOR_LOG_DIR`` was honoured, but via a bare ``or`` rather than
  ``.strip()``, so a whitespace-only value was truthy and produced a directory literally
  named ``"   "``.

Constants bind at import, so each case reloads the module under a patched environment.
"""

from __future__ import annotations

import importlib
import pathlib

import pytest

import cascor_constants.constants as constants_module
import cascor_constants.constants_hdf5.constants_hdf5 as hdf5_module


def _reload(module, monkeypatch, var, value):
    """Reload ``module`` with ``var`` set to ``value`` (or removed when ``None``)."""
    if value is None:
        monkeypatch.delenv(var, raising=False)
    else:
        monkeypatch.setenv(var, value)
    return importlib.reload(module)


@pytest.fixture(autouse=True)
def _restore_modules():
    """Always reload both modules back to the ambient environment."""
    yield
    importlib.reload(hdf5_module)
    importlib.reload(constants_module)


class TestSnapshotsDirOverride:
    """W-6: ``JUNIPER_CASCOR_SNAPSHOTS_DIR``."""

    VAR = "JUNIPER_CASCOR_SNAPSHOTS_DIR"

    def test_unset_keeps_legacy_cascor_snapshots(self, monkeypatch):
        mod = _reload(hdf5_module, monkeypatch, self.VAR, None)
        assert mod._HDF5_PROJECT_SNAPSHOTS_DIR.name == "cascor_snapshots"

    def test_override_is_honoured(self, monkeypatch, tmp_path):
        target = tmp_path / "run-snapshots"
        mod = _reload(hdf5_module, monkeypatch, self.VAR, str(target))
        assert mod._HDF5_PROJECT_SNAPSHOTS_DIR == target

    def test_user_home_is_expanded(self, monkeypatch):
        mod = _reload(hdf5_module, monkeypatch, self.VAR, "~/snap-probe")
        assert mod._HDF5_PROJECT_SNAPSHOTS_DIR == pathlib.Path.home() / "snap-probe"

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_value_is_treated_as_unset(self, monkeypatch, blank):
        mod = _reload(hdf5_module, monkeypatch, self.VAR, blank)
        assert mod._HDF5_PROJECT_SNAPSHOTS_DIR.name == "cascor_snapshots"


class TestLogDirOverride:
    """Q-6: ``JUNIPER_CASCOR_LOG_DIR``."""

    VAR = "JUNIPER_CASCOR_LOG_DIR"

    def test_unset_keeps_project_logs(self, monkeypatch):
        mod = _reload(constants_module, monkeypatch, self.VAR, None)
        assert mod._PROJECT_LOG_DIR_DEFAULT.name == mod._PROJECT_LOG_DIR_NAME_DEFAULT

    def test_override_is_honoured(self, monkeypatch, tmp_path):
        target = tmp_path / "run-logs"
        mod = _reload(constants_module, monkeypatch, self.VAR, str(target))
        assert mod._PROJECT_LOG_DIR_DEFAULT == target

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_value_is_treated_as_unset(self, monkeypatch, blank):
        """A whitespace-only value must not become a directory named "   "."""
        mod = _reload(constants_module, monkeypatch, self.VAR, blank)
        assert mod._PROJECT_LOG_DIR_DEFAULT.name == mod._PROJECT_LOG_DIR_NAME_DEFAULT
