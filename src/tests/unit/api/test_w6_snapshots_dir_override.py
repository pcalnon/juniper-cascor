"""W-6 (CLI experimentation plan §11 / H-4+H-5) — ``JUNIPER_CASCOR_SNAPSHOTS_DIR`` override.

Both tiers must honour the env override, treat a set-but-blank value as unset (the
blank-env guard class), and — when unset — resolve to the ONE canonical root,
``<repo>/cascor-snapshots``. Service tier reads at CALL time
(``TrainingLifecycleManager._get_snapshots_dir``, created on demand); direct-CLI
tier reads at IMPORT time (``constants_hdf5._HDF5_PROJECT_SNAPSHOTS_DIR``), pinned
via a module re-exec so the test does not depend on this process's import order.

The shared-default assertions are the point: a container, a systemd service, and a
direct CLI run on one host must land in the SAME directory, because a snapshot
saved by one is restored and resumed by another. Two earlier roots
(``src/cascor_snapshots``, ``<repo>/snapshots``) are superseded and asserted
against below so a revert cannot pass silently.
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
_REPO_DIR = _SRC_DIR.parent
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

    def test_unset_falls_back_to_canonical_root(self, mgr, monkeypatch):
        monkeypatch.delenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", raising=False)
        resolved = mgr._get_snapshots_dir()
        assert resolved == _REPO_DIR / "cascor-snapshots"

    def test_blank_value_is_treated_as_unset(self, mgr, monkeypatch):
        monkeypatch.setenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", "   ")
        resolved = mgr._get_snapshots_dir()
        assert resolved == _REPO_DIR / "cascor-snapshots"

    def test_call_time_read_not_cached(self, mgr, tmp_path, monkeypatch):
        """Two calls under different env values resolve differently — the
        override is read per call, never captured at import/instance time."""
        first = tmp_path / "a"
        second = tmp_path / "b"
        monkeypatch.setenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", str(first))
        assert mgr._get_snapshots_dir() == first
        monkeypatch.setenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", str(second))
        assert mgr._get_snapshots_dir() == second


class TestInstalledCopyFallback:
    """From an INSTALLED copy there is no project root, and the repo-relative default would
    resolve into the interpreter's own library tree (``<python-lib>/cascor-snapshots``) --
    root-owned on a system Python, and never captured by the project's offline backup.

    ``cascor_constants`` is vendored verbatim into the published ``juniper-cascor-model``
    wheel, so this is a real deployment shape (the distributed worker), not a hypothetical.
    """

    @staticmethod
    def _fake_installed(monkeypatch, root):
        real = __import__("sysconfig").get_paths

        def fake_get_paths(*a, **kw):
            paths = dict(real(*a, **kw))
            paths["purelib"] = str(root)
            paths["platlib"] = str(root)
            return paths

        monkeypatch.setattr("sysconfig.get_paths", fake_get_paths)

    def test_installed_copy_falls_back_to_cwd_and_warns(self, tmp_path, monkeypatch):
        # Claim the module's own tree as "site-packages" so the guard fires.
        self._fake_installed(monkeypatch, _SRC_DIR)
        monkeypatch.chdir(tmp_path)
        with pytest.warns(RuntimeWarning, match="JUNIPER_CASCOR_SNAPSHOTS_DIR"):
            resolved = _reexec_constants_hdf5(None, monkeypatch)
        assert resolved == tmp_path / "cascor-snapshots"

    def test_installed_copy_never_resolves_into_the_python_tree(self, tmp_path, monkeypatch):
        """The specific regression: the default must not land under purelib/platlib."""
        self._fake_installed(monkeypatch, _SRC_DIR)
        monkeypatch.chdir(tmp_path)
        with pytest.warns(RuntimeWarning):
            resolved = _reexec_constants_hdf5(None, monkeypatch)
        assert not resolved.is_relative_to(_SRC_DIR)

    def test_explicit_override_still_wins_when_installed(self, tmp_path, monkeypatch):
        """A configured deployment must not warn -- compose, the systemd unit and the image
        all set the variable, so the warning means genuinely unconfigured."""
        self._fake_installed(monkeypatch, _SRC_DIR)
        target = tmp_path / "configured"
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error", RuntimeWarning)
            assert _reexec_constants_hdf5(str(target), monkeypatch) == target

    def test_checkout_copy_is_not_treated_as_installed(self, monkeypatch):
        monkeypatch.delenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", raising=False)
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error", RuntimeWarning)
            assert _reexec_constants_hdf5(None, monkeypatch) == _REPO_DIR / "cascor-snapshots"


class TestDirectCliTierOverride:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        target = tmp_path / "cli-snapshots"
        assert _reexec_constants_hdf5(str(target), monkeypatch) == target

    def test_unset_resolves_to_canonical_root(self, monkeypatch):
        resolved = _reexec_constants_hdf5(None, monkeypatch)
        assert resolved == _REPO_DIR / "cascor-snapshots"

    def test_blank_value_is_treated_as_unset(self, monkeypatch):
        resolved = _reexec_constants_hdf5("", monkeypatch)
        assert resolved == _REPO_DIR / "cascor-snapshots"


class TestOneSharedRootAcrossTiers:
    """The convention's actual guarantee: with no override, BOTH tiers land in the
    same directory — the thing that makes a snapshot written by a container run
    loadable by a CLI run on the same host."""

    def test_both_tiers_resolve_to_the_same_directory(self, mgr, monkeypatch):
        monkeypatch.delenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", raising=False)
        assert mgr._get_snapshots_dir() == _reexec_constants_hdf5(None, monkeypatch)

    @pytest.mark.parametrize("superseded", ["snapshots", "src/cascor_snapshots", "src/snapshots"])
    def test_superseded_roots_are_not_used(self, mgr, monkeypatch, superseded):
        monkeypatch.delenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", raising=False)
        stale = _REPO_DIR / superseded
        assert mgr._get_snapshots_dir() != stale
        assert _reexec_constants_hdf5(None, monkeypatch) != stale

    def test_canonical_root_name_is_not_a_python_identifier(self):
        """The hyphen blocks the ``import cascor_snapshots`` statement form and keeps the
        directory out of plain ``find_packages``.

        It is NOT, on its own, what keeps the archive out of the distribution -- an earlier
        version of this test claimed that. pyproject's ``[tool.setuptools.packages.find]``
        defaults to ``namespaces = true`` (PEP 420), whose finder needs no ``__init__.py``
        and rejects only names containing a dot, so ``find_namespace_packages`` returns
        ``cascor-snapshots`` and a built wheel carries it in ``top_level.txt``. The
        structural guard is ``namespaces = false`` in ``pyproject.toml``, pinned by
        ``TestPackagingExcludesArtifacts`` below. The hyphen is defence in depth.
        """
        assert not "cascor-snapshots".isidentifier()


class TestPackagingExcludesArtifacts:
    """``namespaces = false`` is what actually keeps artifacts and sibling distributions
    out of the juniper-cascor wheel (C-6: snapshots never reach PyPI)."""

    @staticmethod
    def _resolved_top_levels():
        import tomllib

        from setuptools.config import expand

        cfg = tomllib.loads((_REPO_DIR / "pyproject.toml").read_text())
        find = cfg["tool"]["setuptools"]["packages"]["find"]
        pkgs = expand.find_packages(
            where=find["where"],
            exclude=find.get("exclude", []),
            namespaces=find.get("namespaces", True),
            root_dir=str(_REPO_DIR),
        )
        return {p.split(".")[0] for p in pkgs}

    def test_namespaces_is_disabled(self):
        import tomllib

        cfg = tomllib.loads((_REPO_DIR / "pyproject.toml").read_text())
        assert cfg["tool"]["setuptools"]["packages"]["find"]["namespaces"] is False

    def test_snapshot_root_is_not_a_discovered_package(self):
        tops = self._resolved_top_levels()
        assert "cascor-snapshots" not in tops
        assert "cascor_snapshots" not in tops

    def test_sibling_distributions_are_not_bundled(self):
        """juniper-cascor-model and juniper-cascor-protocol publish their own wheels and
        must never ship inside this one; namespaces=true discovered both."""
        tops = self._resolved_top_levels()
        assert "juniper-cascor-model" not in tops
        assert "juniper-cascor-protocol" not in tops

    def test_non_code_directories_are_not_discovered(self):
        tops = self._resolved_top_levels()
        assert not tops & {"logs", "notes", "docs", "images", "conf", "data", "scripts", "util"}

    def test_real_packages_survive(self):
        """The guard must not cost coverage: every directory carrying an __init__.py stays."""
        tops = self._resolved_top_levels()
        assert {"api", "cascade_correlation", "cascor_constants", "snapshots"} <= tops
