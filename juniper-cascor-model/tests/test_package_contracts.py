"""Package and publish-contract tests for ``juniper-cascor-model``.

The distributed worker consumes this package from a built wheel, not from the
source checkout. These tests pin the lightweight metadata and package-discovery
contracts that keep the published artifact usable by worker runtime imports.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _PACKAGE_ROOT / "pyproject.toml"

_WORKER_IMPORT_PACKAGES = {
    "juniper_cascor_model*",
    "candidate_unit*",
    "utils*",
    "log_config*",
    "cascor_constants*",
}


def _pyproject_metadata() -> dict:
    return tomllib.loads(_PYPROJECT.read_text())


def test_runtime_version_matches_project_metadata():
    """Publishing metadata and import-time ``__version__`` must not drift."""
    from juniper_cascor_model import __version__

    metadata = _pyproject_metadata()

    assert metadata["project"]["version"] == __version__


def test_version_module_imports_without_site_packages_or_torch():
    """The publish workflow's no-deps verification path must stay lightweight."""
    code = (
        "import sys\n"
        "import juniper_cascor_model\n"
        "assert 'torch' not in sys.modules\n"
        "print(juniper_cascor_model.__version__)\n"
    )

    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=_PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == _pyproject_metadata()["project"]["version"]


def test_setuptools_package_discovery_includes_worker_import_surface():
    """The wheel must continue shipping the top-level modules documented for workers."""
    metadata = _pyproject_metadata()
    package_find = metadata["tool"]["setuptools"]["packages"]["find"]

    assert set(package_find["include"]) >= _WORKER_IMPORT_PACKAGES
    assert set(package_find["exclude"]) >= {"tests*", "images*"}


def test_twine_metadata_contract_disables_implicit_license_file_globs():
    """Keep the metadata fix that prevents invalid missing-license warnings on build."""
    metadata = _pyproject_metadata()

    assert metadata["tool"]["setuptools"]["license-files"] == []
