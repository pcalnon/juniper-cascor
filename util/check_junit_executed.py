#!/usr/bin/env python3
"""
Project:     Juniper
Sub-Project: juniper-cascor
Application: util
Author:      Paul Calnon
Version:     0.1.0
License:     MIT License

Fail a test lane whose JUnit report shows that it executed no test, or that a test failed or
errored. Run it after the lane's pytest step, on that step's ``--junitxml`` file::

    python util/check_junit_executed.py reports/junit/junit-long.xml --lane "Long-Running Correctness Tests"

Stdlib only.

Why this exists
---------------
The nightly ``Long-Running Correctness Tests`` job in ``.github/workflows/scheduled-tests.yml``
ran zero tests and passed. It passed ``--run-long`` only, and ``src/tests/conftest.py`` gives its
one pick, ``test_deterministic_training_resume``, two more skips: ``slow`` (lifted by ``--slow``)
and ``integration`` (lifted by ``--integration``; the conftest tests ``"integration" in
item.keywords``, which holds the ``src/tests/integration`` directory's name). Job 121983548401
(2026-10-27) logged ``1 skipped, 6214 deselected`` and succeeded. pytest exits 5 only when NOTHING
is collected. A lane whose every pick is skipped exits 0, so its exit status cannot see this, and
only the report can.

The report is also the more reliable record of a failure. ``src/tests/conftest.py``'s
``pytest_unconfigure`` ends the process with ``os._exit(0)`` whenever a non-daemon thread outlives
the session, so that such a thread cannot hang the job. It runs after the report is written, but
it exits 0 whatever the session's outcome, so a lane with a failing test can still exit 0. The
report records the failure either way.

What counts as executed
-----------------------
A ``<testcase>`` executed unless pytest skipped it. pytest writes an XFAIL as
``<skipped type="pytest.xfail">``, but the test body ran and failed as expected, so an XFAIL counts
as executed. An XPASS (non-strict) is written as a pass. Every other ``<skipped>`` is a skip, and
the skip reasons are printed when nothing executed, since a missing opt-in flag (``--slow``,
``--integration``, ``--run-long``, ``--run-performance``) is the usual cause.

Exit codes: 0 the lane executed at least ``--min-executed`` tests (default 1) and none failed or
errored; 1 it did not; 2 the report is missing or is not a JUnit report, or a usage error.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET  # nosec B405 - parses the lane's own pytest report, written on the same runner
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

XFAIL_TYPE = "pytest.xfail"
REPORT_ROOTS = ("testsuites", "testsuite")
SKIP_REASONS_SHOWN = 5


@dataclass(frozen=True)
class LaneCounts:
    """Outcome counts of one JUnit report: one outcome per ``<testcase>``."""

    passed: int = 0
    xfailed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    skip_reasons: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return self.passed + self.xfailed + self.failed + self.errors + self.skipped

    @property
    def executed(self) -> int:
        """Tests whose body ran: passes (XPASS included), XFAILs and failures."""
        return self.passed + self.xfailed + self.failed


def classify(testcase: ET.Element) -> str:
    """One outcome per testcase, in this precedence: failed, errors, xfailed, skipped, passed.

    A failure in the call and an error in teardown can share one testcase; it counts once, as failed.
    """
    if testcase.find("failure") is not None:
        return "failed"
    if testcase.find("error") is not None:
        return "errors"
    skipped = testcase.find("skipped")
    if skipped is not None:
        return "xfailed" if skipped.get("type") == XFAIL_TYPE else "skipped"
    return "passed"


def count(root: ET.Element) -> LaneCounts:
    """Tally every ``<testcase>`` under ``root``, across however many ``<testsuite>`` elements it holds."""
    tally: Counter = Counter()
    reasons: Counter = Counter()
    for testcase in root.iter("testcase"):
        outcome = classify(testcase)
        tally[outcome] += 1
        if outcome == "skipped":
            skipped = testcase.find("skipped")
            reasons[(skipped.get("message") or "").strip() or "(no reason given)"] += 1
    return LaneCounts(passed=tally["passed"], xfailed=tally["xfailed"], failed=tally["failed"], errors=tally["errors"], skipped=tally["skipped"], skip_reasons=reasons)


def problems(counts: LaneCounts, min_executed: int) -> list[str]:
    """Why the lane fails the check, or an empty list when it passes."""
    found: list[str] = []
    if counts.failed or counts.errors:
        found.append(f"{counts.failed} test(s) failed and {counts.errors} errored")
    if counts.executed < min_executed:
        found.append(f"the lane executed {counts.executed} test(s), fewer than the {min_executed} required: of {counts.total} reported, {counts.skipped} were skipped")
    return found


def summary_line(lane: str, report: Path, counts: LaneCounts) -> str:
    outcomes = f"passed {counts.passed}, xfailed {counts.xfailed}, failed {counts.failed}"
    return f"{lane}: {counts.total} testcase(s) in {report}: executed {counts.executed} ({outcomes}), errors {counts.errors}, skipped {counts.skipped}"


def _escape_data(text: str) -> str:
    """Escape a GitHub Actions workflow-command message."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    """Escape a GitHub Actions workflow-command property value (``title=...``)."""
    return _escape_data(text).replace(":", "%3A").replace(",", "%2C")


def annotate(lane: str, message: str) -> str:
    return f"::error title={_escape_property(lane)}::{_escape_data(message)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail a test lane whose JUnit report shows that it executed no test, or that a test failed or errored.")
    parser.add_argument("report", type=Path, help="the lane's --junitxml report")
    parser.add_argument("--min-executed", type=int, default=1, help="fewest tests the lane must execute (default: 1)")
    parser.add_argument("--lane", default=None, help="lane name used in messages (default: the report's file name)")
    args = parser.parse_args(argv)
    if args.min_executed < 1:
        parser.error("--min-executed must be at least 1")
    lane = args.lane or args.report.name

    try:
        root = ET.parse(args.report).getroot()  # nosec B314 - the lane's own pytest report, written on the same runner
    except (OSError, ET.ParseError) as exc:
        print(annotate(lane, f"no readable JUnit report at {args.report}: {exc}"))
        return 2
    if root.tag not in REPORT_ROOTS:
        print(annotate(lane, f"{args.report} is not a JUnit report: its root element is <{root.tag}>, not <testsuites> or <testsuite>"))
        return 2

    counts = count(root)
    print(summary_line(lane, args.report, counts))
    found = problems(counts, args.min_executed)
    if counts.executed < args.min_executed and counts.skip_reasons:
        print("Skip reasons, most frequent first:")
        for reason, n in counts.skip_reasons.most_common(SKIP_REASONS_SHOWN):
            print(f"  {n:>5}  {reason}")
    for problem in found:
        print(annotate(lane, problem))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
