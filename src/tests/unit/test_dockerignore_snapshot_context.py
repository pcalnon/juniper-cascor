"""``.dockerignore`` must exclude the nested snapshot archive and the test suite.

``COPY src/ ./src/`` is a directory copy. Docker matches patterns with
``filepath.Match`` relative to the context root, so a root-anchored
``cascor_snapshots/`` does not match ``src/cascor_snapshots/``. That was
measured on 2026-09-21: 766 ``.h5`` files, 39 MB, every one carrying a
plaintext multiprocessing authkey (``authkey_hex``). A root-anchored
``tests/`` likewise leaves ``src/tests/`` in the image — 268 test files
shipped in ``juniper-cascor:0.11.0``. The ``**/`` twins are what actually
exclude those paths. See ``.dockerignore`` and juniper-cascor#661 / #660.
"""

import re
from pathlib import Path

import pytest

# CI's unit lane runs ``pytest -m "unit and not slow" src/tests/unit``, so an
# unmarked test is collected and then deselected.
pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERIGNORE = REPO_ROOT / ".dockerignore"

# Paths the root-anchored patterns do not cover, and which ``COPY src/`` ships.
_NESTED_LEAKS = (
    "src/cascor_snapshots/run.h5",
    "src/cascor-snapshots/run.h5",
    "src/snapshots/run.h5",
    "src/tests/unit/test_api_security.py",
)

_PRODUCTION_SOURCES = (
    "src/cascade_correlation/cascade_correlation.py",
    "src/api/security.py",
    "src/api/lifecycle/manager.py",
    "src/snapshots/snapshot_serializer.py",
    "pyproject.toml",
    "Dockerfile",
)


def _patterns() -> list[str]:
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _to_regex(pattern: str) -> str:
    """Docker-style glob: ``**`` crosses directories, ``*`` does not, and a leading ``**/`` also matches at the root."""
    out = ["^"]
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            out.append("(?:.*/)?")
            index += 3
            continue
        if pattern.startswith("**", index):
            out.append(".*")
            index += 2
            continue
        character = pattern[index]
        if character == "*":
            out.append("[^/]*")
        elif character == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(character))
        index += 1
    out.append("$")
    return "".join(out)


def _glob_match(pattern: str, path: str) -> bool:
    return re.fullmatch(_to_regex(pattern), path) is not None


def _matches(pattern: str, relpath: str) -> bool:
    """A trailing slash matches that directory and everything under it. Otherwise the path itself must match."""
    directory = pattern.endswith("/")
    body = pattern.rstrip("/")
    path = relpath.strip("/")
    if _glob_match(body, path):
        return True
    if not directory:
        return False
    parts = path.split("/")
    return any(_glob_match(body, "/".join(parts[:length])) for length in range(1, len(parts)))


def _excluded(patterns: list[str], relpath: str) -> bool:
    """Last matching pattern wins, and a leading ``!`` re-includes."""
    excluded = False
    for raw in patterns:
        negate = raw.startswith("!")
        pattern = raw[1:] if negate else raw
        if _matches(pattern, relpath):
            excluded = not negate
    return excluded


class TestDockerignoreMatcher:
    """The matcher encodes the #661 rule. These cases are the measured ones, not a full Docker implementation."""

    def test_root_anchored_directory_misses_a_nested_path(self) -> None:
        assert _matches("cascor_snapshots/", "cascor_snapshots/run.h5")
        assert not _matches("cascor_snapshots/", "src/cascor_snapshots/run.h5")

    def test_double_star_twin_hits_the_nested_path(self) -> None:
        assert _matches("**/cascor_snapshots/", "src/cascor_snapshots/run.h5")
        assert _matches("**/cascor_snapshots/", "cascor_snapshots/run.h5")

    def test_h5_glob_is_root_anchored_without_the_twin(self) -> None:
        assert _matches("snapshots/*.h5", "snapshots/run.h5")
        assert not _matches("snapshots/*.h5", "src/snapshots/run.h5")
        assert _matches("**/snapshots/*.h5", "src/snapshots/run.h5")

    def test_tests_directory_needs_the_twin(self) -> None:
        assert _matches("tests/", "tests/unit/test_api_security.py")
        assert not _matches("tests/", "src/tests/unit/test_api_security.py")
        assert _matches("**/tests/", "src/tests/unit/test_api_security.py")

    def test_exception_after_a_match_reincludes(self) -> None:
        patterns = [".env.*", "**/.env.*", "!.env.example"]
        assert _excluded(patterns, ".env.production")
        assert not _excluded(patterns, ".env.example")


class TestSnapshotAndSuiteExclusions:
    def test_dockerignore_exists(self) -> None:
        assert DOCKERIGNORE.is_file()

    @pytest.mark.parametrize("relpath", _NESTED_LEAKS)
    def test_nested_snapshot_and_suite_paths_are_excluded(self, relpath: str) -> None:
        assert _excluded(_patterns(), relpath), relpath

    @pytest.mark.parametrize("relpath", _NESTED_LEAKS)
    def test_root_anchored_patterns_alone_do_not_exclude_them(self, relpath: str) -> None:
        """Deleting the ``**/`` lines puts these paths back in a ``COPY src/`` image."""
        root_only = [pattern for pattern in _patterns() if not pattern.startswith("**/")]
        assert not _excluded(root_only, relpath), relpath

    @pytest.mark.parametrize("relpath", _PRODUCTION_SOURCES)
    def test_production_sources_stay_in_the_context(self, relpath: str) -> None:
        assert not _excluded(_patterns(), relpath), relpath

    def test_env_example_stays_and_a_nested_credential_does_not(self) -> None:
        patterns = _patterns()
        assert not _excluded(patterns, ".env.example")
        assert _excluded(patterns, "src/secrets/id_rsa.pem")
        assert _excluded(patterns, "src/keys/server.key")
