"""
Pin that ``src/tests/conftest.py``'s forced exit keeps the session's exit status.

``pytest_unconfigure`` ends the process with ``os._exit`` whenever a non-daemon thread outlives the
session, so that such a thread cannot hang the job. Until 2026-10-29 it called ``os._exit(0)``, which
replaced pytest's exit status: a session with a failing test and a surviving thread exited 0, and
every lane read that as a pass. juniper-cascor#735 measured it: ``1 failed``, exit status 0.

Each case runs a real pytest in a subprocess, over a one-test tree whose conftest loads only this
repo's ``pytest_sessionfinish`` and ``pytest_unconfigure`` from ``src/tests/conftest.py``. The
surviving thread waits on an event that is never set, so the child can exit promptly only through the
forced exit. If that path were lost, the child would hang until the subprocess timeout and the case
would fail loudly rather than pass.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - runs a child pytest with a fixed argv, never a shell
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT_CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"
CHILD_TIMEOUT_S = 45

CHILD_CONFTEST = textwrap.dedent(
    """\
    import importlib.util
    import sys

    _spec = importlib.util.spec_from_file_location("_cascor_root_conftest", {path!r})
    _root = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _root
    _spec.loader.exec_module(_root)

    # Only the session-exit hooks: the root conftest's options, fixtures and skips stay out.
    pytest_unconfigure = _root.pytest_unconfigure
    # Guarded so the same tree also drives the pre-fix conftest, which has no such hook.
    if hasattr(_root, "pytest_sessionfinish"):
        pytest_sessionfinish = _root.pytest_sessionfinish
    """
)

FAILING_WITH_SURVIVOR = textwrap.dedent(
    """\
    import threading


    def test_fails_while_a_non_daemon_thread_survives():
        threading.Thread(target=threading.Event().wait, daemon=False).start()
        assert False, "deliberate failure"
    """
)

PASSING_WITH_SURVIVOR = textwrap.dedent(
    """\
    import threading


    def test_passes_while_a_non_daemon_thread_survives():
        threading.Thread(target=threading.Event().wait, daemon=False).start()
    """
)

FAILING_WITHOUT_SURVIVOR = textwrap.dedent(
    """\
    def test_fails_with_no_surviving_thread():
        assert False, "deliberate failure"
    """
)


def _child_pytest(tmp_path: Path, test_body: str) -> subprocess.CompletedProcess:
    """Run a real, isolated pytest over ``test_body``: no repo config, no autoloaded plugins."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text(CHILD_CONFTEST.format(path=str(ROOT_CONFTEST)), encoding="utf-8")
    (tmp_path / "test_child.py").write_text(test_body, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # The root conftest otherwise makes a fresh snapshot directory under TMPDIR on import.
    env["JUNIPER_CASCOR_SNAPSHOTS_DIR"] = str(tmp_path / "snapshots")
    argv = [sys.executable, "-m", "pytest", "-c", str(tmp_path / "pytest.ini"), "--rootdir", str(tmp_path), "--confcutdir", str(tmp_path), "-p", "no:cacheprovider", "-q", str(tmp_path / "test_child.py")]
    return subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=CHILD_TIMEOUT_S)  # nosec B603 - fixed argv, no shell


def _output(result: subprocess.CompletedProcess) -> str:
    return f"exit status {result.returncode}\n--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}"


class TestForcedExitKeepsTheStatus:
    def test_a_failing_session_with_a_surviving_thread_exits_nonzero(self, tmp_path):
        """The defect: ``os._exit(0)`` turned this into exit status 0."""
        result = _child_pytest(tmp_path, FAILING_WITH_SURVIVOR)
        assert "1 failed" in result.stdout, _output(result)
        assert result.returncode == int(pytest.ExitCode.TESTS_FAILED), _output(result)

    def test_a_passing_session_with_a_surviving_thread_exits_zero(self, tmp_path):
        result = _child_pytest(tmp_path, PASSING_WITH_SURVIVOR)
        assert "1 passed" in result.stdout, _output(result)
        assert result.returncode == int(pytest.ExitCode.OK), _output(result)

    def test_a_failing_session_without_a_survivor_exits_nonzero(self, tmp_path):
        """Control: no thread outlives the session, so the forced exit never runs."""
        result = _child_pytest(tmp_path, FAILING_WITHOUT_SURVIVOR)
        assert "1 failed" in result.stdout, _output(result)
        assert result.returncode == int(pytest.ExitCode.TESTS_FAILED), _output(result)
