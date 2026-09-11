"""Regression: ``.dockerignore`` must exclude *nested* egg-info / dist-info dirs.

juniper-cascor runs from ``/app/src`` (``ENV PYTHONPATH=/app/src`` +
``COPY src/ ./src/``), so a stale build artifact under ``src/`` would land on
the import path ahead of site-packages and shadow the installed package's
metadata version — the class of bug fixed for juniper-canopy in #362. A
root-only ``*.egg-info/`` does not match a nested ``src/*.egg-info``; this pins
the ``**/``-prefixed forms so the hardening can't silently regress. See
juniper-ml ``notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md``.
"""

from pathlib import Path

import pytest

# CI's unit lane runs ``pytest -m "unit and not slow" src/tests/unit`` (ci.yml:312-317),
# so a test here with no ``unit`` marker is collected and then silently DESELECTED -- it
# never runs and the job still reports success. Module-level, so it covers every test in
# the file including ones added later.
pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


def _patterns() -> list[str]:
    """Return the non-comment, non-blank pattern lines from ``.dockerignore``."""
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_dockerignore_exists() -> None:
    assert DOCKERIGNORE.is_file(), f".dockerignore must exist at the repo root ({REPO_ROOT})."


def test_dockerignore_excludes_nested_egg_info() -> None:
    assert "**/*.egg-info/" in _patterns(), "`.dockerignore` must contain `**/*.egg-info/` so a nested src/*.egg-info build artifact cannot be COPYed into the image and shadow the installed package version on PYTHONPATH=/app/src; a root-only `*.egg-info/` is insufficient."


def test_dockerignore_excludes_nested_dist_info() -> None:
    assert "**/*.dist-info/" in _patterns(), "`.dockerignore` must exclude nested `**/*.dist-info/` for the same reason."
