#!/usr/bin/env python
"""Regression: every runtime version surface reads the installed distribution's metadata.

juniper-cascor#668. BUG-CC-04 moved ``api.app._API_VERSION`` and ``/v1/health`` onto
``importlib.metadata``, but three surfaces still restated the version as a literal, and all
three read ``"0.6.0"`` while ``pyproject.toml`` read 0.11.0:

- ``juniper_cascor.__version__`` -- what ``publish.yml``'s TestPyPI check prints on every release;
- ``api.models.common._API_VERSION`` -- the default ``meta.version`` of every enveloped response;
- ``api.routes.health._API_VERSION``'s source-checkout fallback.

The first two tests pin the installed path. The third pins the mechanism: no version surface
may assign a release-number literal at all, because a literal that has to be bumped by hand is
how the drift happened -- and ``propose.py`` does not bump these files.
"""

import ast
import importlib.metadata
import re
from pathlib import Path

import pytest

import juniper_cascor
from api.models.common import Meta, ResponseEnvelope
from api.routes import health

# CI's unit lane runs ``pytest -m "unit and not slow" src/tests/unit``, so an unmarked test is
# collected and then silently DESELECTED. Module-level, so it covers tests added later.
pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
VERSION_SURFACES = {
    "juniper_cascor/__init__.py": "__version__",
    "src/api/app.py": "_API_VERSION",
    "src/api/models/common.py": "_API_VERSION",
    "src/api/routes/health.py": "_API_VERSION",
}
_RELEASE_LITERAL = re.compile(r"^\d+\.\d+\.\d+$")


def _installed() -> str:
    return importlib.metadata.version("juniper-cascor")


def _release_literals_assigned_to(source: str, name: str) -> list[str]:
    """Return every release-number string literal assigned to ``name`` anywhere in ``source``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str) and _RELEASE_LITERAL.match(value.value):
            found.append(value.value)
    return found


def test_package_version_is_the_installed_distribution_version() -> None:
    assert juniper_cascor.__version__ == _installed()


def test_envelope_and_health_versions_are_the_installed_distribution_version() -> None:
    assert Meta().version == _installed()
    assert ResponseEnvelope().meta.version == _installed()
    assert health._API_VERSION == _installed()


@pytest.mark.parametrize(("relpath", "name"), sorted(VERSION_SURFACES.items()))
def test_no_version_surface_assigns_a_release_literal(relpath: str, name: str) -> None:
    source = (REPO_ROOT / relpath).read_text(encoding="utf-8")
    assert name in source, f"{relpath} no longer defines {name}; update VERSION_SURFACES"
    literals = _release_literals_assigned_to(source, name)
    assert literals == [], f"{relpath} assigns {name} the release literal(s) {literals}; read importlib.metadata instead (juniper-cascor#668)"
