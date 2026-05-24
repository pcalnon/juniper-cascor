#!/usr/bin/env python
"""
Unit tests for CFG-02: ``sentry-sdk`` moved to the ``[observability]`` extra.

CFG-02 (v7 roadmap §13524) moves ``sentry-sdk`` out of cascor's
``[project] dependencies`` and into the ``[observability]`` optional
extra, with a try/except ImportError guard around the bootstrap import
in ``src/main.py`` so bare ``pip install juniper-cascor`` (no extras)
no longer crashes at module-load time.

These tests are source-level guards that pin the three CFG-02
invariants so the move does not silently regress:

    1. ``sentry-sdk`` is NOT in ``[project] dependencies``.
    2. ``sentry-sdk`` IS in ``[project.optional-dependencies][observability]``.
    3. ``src/main.py`` has no top-level ``import sentry_sdk``; the lazy
       import lives inside the ``if _sentry_dsn:`` block and is wrapped
       in ``try/except ImportError``.

Mirrors the source-level guard pattern used by the other CFG-* /
SEC-* regression suites in this directory.
"""

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_MAIN_PY = _REPO_ROOT / "src" / "main.py"


def _load_pyproject() -> dict:
    with _PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def test_sentry_sdk_not_in_core_dependencies():
    """``sentry-sdk`` must not appear in ``[project] dependencies``.

    Sentry is opt-in (only used when DSN is configured), so it lives in
    the ``[observability]`` extra. Promoting it back to core would
    reverse the install-footprint reduction this PR delivers.
    """
    data = _load_pyproject()
    deps = data["project"]["dependencies"]
    leaked = [dep for dep in deps if dep.split(">")[0].split("=")[0].split("<")[0].strip() == "sentry-sdk"]
    assert not leaked, f"sentry-sdk leaked back into [project] dependencies (CFG-02 regression): {leaked}"


def test_sentry_sdk_in_observability_extra():
    """``sentry-sdk`` must be declared in the ``[observability]`` extra.

    The floor (``>=2.0.0``) matches the pre-move floor; bumping is out
    of CFG-02's scope.
    """
    data = _load_pyproject()
    observability = data["project"]["optional-dependencies"]["observability"]
    matches = [dep for dep in observability if dep.startswith("sentry-sdk")]
    assert matches, f"sentry-sdk must be declared in the [observability] extra (CFG-02). Current contents: {observability}"


def test_main_py_has_no_top_level_sentry_import():
    """``src/main.py`` must not import ``sentry_sdk`` at module top-level.

    The previous unconditional top-level ``import sentry_sdk`` made
    ``pip install juniper-cascor`` (no extras) crash on import even
    when no DSN was configured. CFG-02 moves the import inside the
    ``if _sentry_dsn:`` block.
    """
    src = _MAIN_PY.read_text()
    top_level_pattern = re.compile(r"^import sentry_sdk\b", re.MULTILINE)
    assert not top_level_pattern.search(src), "src/main.py still has a top-level ``import sentry_sdk`` (CFG-02 regression). " "It must be lazy-imported inside the ``if _sentry_dsn:`` block."


def test_main_py_lazy_imports_sentry_sdk_with_guard():
    """``src/main.py`` must still reference ``sentry_sdk`` (lazy-imported, guarded).

    The lazy import goes inside ``if _sentry_dsn:`` and is wrapped in
    ``try/except ImportError`` so a DSN-set-but-SDK-not-installed
    deployment emits a clear stderr warning rather than crashing.
    """
    src = _MAIN_PY.read_text()
    assert "import sentry_sdk" in src, "src/main.py no longer references sentry_sdk at all (CFG-02 over-correction). " "The lazy import inside the ``if _sentry_dsn:`` block must still be present."
    assert "except ImportError" in src, "src/main.py must wrap the lazy ``import sentry_sdk`` in ``try/except ImportError`` " "(CFG-02 guardrail) so ``pip install juniper-cascor`` (no [observability] extra) " "still works when no DSN is configured, and emits a clear warning when a DSN is set."
