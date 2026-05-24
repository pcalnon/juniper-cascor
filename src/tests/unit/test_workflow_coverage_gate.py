"""Workflow-lint regression: forbid the cov-gate-on-partial-test-set pattern.

The bug this guards against: running ``pytest --cov=src`` together with a
marker filter that selects only a fraction of the test suite (for example
``-m "unit and slow"``, which selects ~10 tests). Coverage measured over
~10 tests will inevitably be far below the project-wide ``fail_under = 80``
threshold defined in ``pyproject.toml`` ``[tool.coverage.report]``, so
``coverage report`` exits 1 ("Required test coverage of 80.0% not reached")
and fails the CI step even when every test passes.

This pattern silently broke ``.github/workflows/scheduled-tests.yml`` for
~3 weeks (last green 2026-05-03, fix landed 2026-05-24) once juniper-cascor
PR #186 made pytest-cov actually write coverage data — the cov-gate failure
had been masked by an earlier "No data to report" failure with the same
exit code.

Catch the regression at lint time. A workflow step that runs pytest with
``--cov=`` AND a partial marker filter must either:

 - opt the run out of the gate via ``--cov-fail-under=0``, OR
 - declare an explicit lower threshold via ``--cov-fail-under=<N>``.

Steps that pass ``--no-cov`` are fine — coverage is disabled entirely.
Steps without ``--cov=`` are fine — no coverage data is generated.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# .../src/tests/unit/this_file.py → repo root is parent.parent.parent.parent
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# A marker filter selects a partial test set when it AND-narrows with
# ``slow``, ``long``, ``performance``, or other opt-in markers. The full
# unit suite uses ``unit and not slow`` (the inverse) — that one is the
# legitimate place to apply the project-wide coverage gate.
_PARTIAL_FILTER_PATTERNS = (
    re.compile(r'-m\s+["\']?[^"\']*\band\s+slow\b'),
    re.compile(r'-m\s+["\']?[^"\']*\band\s+long\b'),
    re.compile(r'-m\s+["\']?\s*slow\b'),
    re.compile(r'-m\s+["\']?\s*long\b'),
    re.compile(r'-m\s+["\']?\s*performance\b'),
)

_COV_FLAG = re.compile(r"\s--cov\b")
_NO_COV_FLAG = re.compile(r"\s--no-cov\b")
_COV_FAIL_UNDER = re.compile(r"--cov-fail-under\s*=\s*\d+|--no-cov-fail-under")


def _workflow_files() -> list[Path]:
    return sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(WORKFLOWS_DIR.glob("*.yaml"))


def _extract_pytest_invocations(text: str) -> list[tuple[int, str]]:
    """Return ``(start_lineno, joined_invocation)`` for every ``pytest`` call.

    Each entry concatenates the line that contains ``pytest`` (or
    ``python -m pytest``) with all immediately-following lines whose
    previous line ends in a backslash continuation. This is a YAML-naive
    scan — it matches both ``run:`` block bodies and inline ``run:`` one-
    liners — but the only false positive would be a YAML string that
    *happens* to look like a pytest command, which we want to flag anyway.
    """
    lines = text.splitlines()
    invocations: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # ``pytest`` or ``python -m pytest`` — the leading word matters,
        # not the in-text occurrence (e.g. ``# pytest`` comment).
        if re.search(r"(?:^|\s)(?:python\s+-m\s+)?pytest\b", line):
            start = i
            joined_parts = [line]
            # Stitch continuation lines together so multi-line invocations
            # are scanned as one command.
            while joined_parts[-1].rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                joined_parts.append(lines[i])
            invocations.append((start + 1, " ".join(part.rstrip("\\").strip() for part in joined_parts)))
        i += 1
    return invocations


@pytest.mark.unit
class TestWorkflowCoverageGate:
    """Forbid ``--cov=`` with a partial-test marker filter unless the gate is overridden."""

    def test_workflows_directory_exists(self):
        assert WORKFLOWS_DIR.is_dir(), f"Expected {WORKFLOWS_DIR} to exist"

    def test_at_least_one_pytest_workflow_invocation_present(self):
        """Sanity: confirm the scanner is actually looking at pytest calls."""
        any_seen = False
        for wf in _workflow_files():
            if _extract_pytest_invocations(wf.read_text(encoding="utf-8")):
                any_seen = True
                break
        assert any_seen, "No pytest invocations found in any .github/workflows/*.yml — scanner is mis-targeted."

    def test_no_partial_marker_with_unguarded_cov(self):
        """The actual lint.

        For every ``pytest`` call in ``.github/workflows/*.yml``:

         * if the call selects a partial test set via a ``slow`` / ``long`` /
           ``performance`` marker filter, AND
         * the call uses ``--cov=`` (any form), AND
         * the call does NOT pass ``--no-cov``, AND
         * the call does NOT pass ``--cov-fail-under=<N>`` (or ``--no-cov-fail-under``)

        then fail the test with a precise pointer to the offending workflow
        and line number.
        """
        violations: list[str] = []
        for wf in _workflow_files():
            text = wf.read_text(encoding="utf-8")
            for lineno, invocation in _extract_pytest_invocations(text):
                if not any(p.search(invocation) for p in _PARTIAL_FILTER_PATTERNS):
                    continue
                if not _COV_FLAG.search(invocation):
                    continue
                if _NO_COV_FLAG.search(invocation):
                    continue
                if _COV_FAIL_UNDER.search(invocation):
                    continue
                rel = wf.relative_to(REPO_ROOT)
                violations.append(f"  - {rel}:{lineno}\n" f"      partial marker (slow/long/performance) + --cov= without " f"--cov-fail-under=0 / --no-cov-fail-under / --no-cov\n" f"      invocation: {invocation[:200]}{'…' if len(invocation) > 200 else ''}")
        assert not violations, "Workflow lint failure: combining --cov=src with a partial-test marker filter trips " "pyproject's fail_under=80 gate because only a small subset of tests run. Either drop " "--cov=src from the partial-marker step or pass --cov-fail-under=0. See " "src/tests/unit/test_workflow_coverage_gate.py for context.\n\n" "Offending invocations:\n" + "\n".join(violations)
