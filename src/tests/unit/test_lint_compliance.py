"""Lint compliance tests to detect common code quality issues early.

These tests catch patterns that would fail pre-commit hooks, providing faster
feedback during development than waiting for the full pre-commit run.
"""

import ast
import os
import re
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent.parent
SOURCE_DIRS = [SRC_DIR / "api", SRC_DIR / "cascade_correlation", SRC_DIR / "candidate_unit", SRC_DIR / "cascor_constants", SRC_DIR / "log_config", SRC_DIR / "parallelism", SRC_DIR / "snapshots", SRC_DIR / "spiral_problem", SRC_DIR / "utils"]


def _python_files(dirs: list[Path]) -> list[Path]:
    """Collect all Python source files from the given directories."""
    files = []
    for d in dirs:
        if d.exists():
            files.extend(d.rglob("*.py"))
    return sorted(files)


def _parse_imports(filepath: Path) -> list[tuple[str, int]]:
    """Parse a Python file and return imported names with line numbers."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.names:
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports.append((name, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports.append((name, node.lineno))
    return imports


@pytest.mark.unit
class TestNoVariableShadowingImports:
    """Detect loop variables that shadow module-level imports (F402/F811)."""

    @pytest.mark.parametrize("filepath", _python_files(SOURCE_DIRS), ids=lambda p: str(p.relative_to(SRC_DIR)))
    def test_no_loop_variable_shadows_import(self, filepath: Path):
        """No for-loop variable should shadow a module-level import name."""
        try:
            tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
        except SyntaxError:
            pytest.skip(f"Syntax error in {filepath}")

        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.names:
                for alias in node.names:
                    imported_names.add(alias.asname if alias.asname else alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname if alias.asname else alias.name)

        shadowed = []
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                target = node.target
                if isinstance(target, ast.Name) and target.id in imported_names:
                    shadowed.append((target.id, target.lineno))
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name) and elt.id in imported_names:
                            shadowed.append((elt.id, elt.lineno))

        assert not shadowed, f"Loop variables shadow imports in {filepath.name}: {shadowed}"


@pytest.mark.unit
class TestNoSetFromGenerator:
    """Detect set(generator) patterns that should be set comprehensions (C401)."""

    @pytest.mark.parametrize("filepath", _python_files(SOURCE_DIRS), ids=lambda p: str(p.relative_to(SRC_DIR)))
    def test_no_set_wrapping_generator(self, filepath: Path):
        """No set() call should wrap a generator expression (use set comprehension instead)."""
        try:
            tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
        except SyntaxError:
            pytest.skip(f"Syntax error in {filepath}")

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "set" and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
                violations.append(node.lineno)

        assert not violations, f"set(generator) found at lines {violations} in {filepath.name} — use set comprehension instead"


@pytest.mark.unit
class TestNoSuppressedImportNotFound:
    """Forbid inline ``# type: ignore[import-not-found]`` — it masks an unresolvable import.

    cascor#331: ``from secrets_util import get_secret  # type: ignore[import-not-found]``
    referenced a module that never existed; the ``except ImportError`` substituted a
    ``None``-returning stub, so cascor sent no juniper-data API key (401 -> 502) on every
    live dataset swap. mypy DID flag the import — the inline suppression hid it. If a module
    cannot be imported, fix the import; a genuinely-optional third-party dependency belongs
    in mypy's ``ignore_missing_imports`` config (or behind a typed shim), not silenced
    inline on a first-party import.
    """

    _SUPPRESSION = re.compile(r"#\s*type:\s*ignore\[import-not-found\]")

    @pytest.mark.parametrize("filepath", _python_files(SOURCE_DIRS), ids=lambda p: str(p.relative_to(SRC_DIR)))
    def test_no_inline_import_not_found_suppression(self, filepath: Path):
        """No source file may silence mypy's import-not-found inline (see class docstring)."""
        offenders = [i for i, line in enumerate(filepath.read_text(encoding="utf-8").splitlines(), 1) if self._SUPPRESSION.search(line)]
        assert not offenders, f"{filepath.relative_to(SRC_DIR)} silences mypy import-not-found inline at line(s) " f"{offenders} — this masks an unresolvable import (see cascor#331). Fix the import, " "or add the optional dependency to mypy's ignore_missing_imports config."
