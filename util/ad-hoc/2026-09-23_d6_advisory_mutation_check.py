#!/usr/bin/env python3
"""Mutation-check the D6 ``epochs_completed`` advisory: every deliberate break must be caught.

Project: juniper-cascor
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-23
Status: ad-hoc — investigation (verifies src/tests/unit/test_candidate_epochs_completed_advisory.py)
Retire when: the owner rules whether the D6 check blocks, and the advisory is rebuilt or removed
Related: juniper-ml notes/JUNIPER_2026-09-11_JUNIPER-ECOSYSTEM_PERF-LANE-SIX-OWNER-DECISIONS-RULED.md §5.4
         (owner decision D6: build the check advisory first, count its firings, then decide);
         src/tests/unit/test_candidate_epochs_completed_advisory.py (the module under test)

Two detectors, because one class of break is invisible to the unit tests:

  SUITE  the module's own tests, selected as CI selects them (``-m "unit and not slow"``).
         Red = caught.
  E2E    the REAL check with the reference drifted (100 -> 67) so that it fires, run under
         pytest's real fd capture with ``GITHUB_ACTIONS=true`` and a private
         ``GITHUB_STEP_SUMMARY``. It must exit 0 (advisory), print exactly ONE line that starts
         ``::warning title=D6 advisory - epochs_completed drift::``, and append exactly one line
         to the summary. Anything else = caught. Only this detector can see the real check lose
         its capture suspension, since no unit test can observe pytest's own capture.

Hermetic: each run copies ``src/``, ``conf/`` and ``pyproject.toml`` into a TemporaryDirectory
and mutates the copy, so the working tree is never edited and no stale ``.pyc`` can answer for a
mutant (a fresh tree has no ``__pycache__``; ``PYTHONDONTWRITEBYTECODE=1`` besides). The
unmutated tree runs first as a control: a detector that fires WITHOUT a mutation proves nothing
when it fires with one. Exit codes are read from the subprocess directly, never through a pipe.

Usage:
    /opt/miniforge3/envs/JuniperCascor1/bin/python util/ad-hoc/2026-09-23_d6_advisory_mutation_check.py
Exit: 0 when the control is clean on both detectors and every mutation is caught; 1 otherwise.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404 -- fixed-argv pytest runs in a temp tree
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = "src/tests/unit/test_candidate_epochs_completed_advisory.py"
ANNOTATION_PREFIX = "::warning title=D6 advisory - epochs_completed drift::"
REFERENCE = "EPOCHS_COMPLETED_REFERENCE = {100: 68, 200: 68}\n"
DRIFTED_REFERENCE = "EPOCHS_COMPLETED_REFERENCE = {100: 67, 200: 68}\n"
COPIED = ("src", "conf")
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "logs", "reports")

# (name, old, new) -- ``old`` must occur exactly once in the target. Backslashes are doubled
# where the target's source text contains a literal ``\n`` escape inside a string.
MUTATIONS = [
    ("annotation written outside the capture suspension", "        with suspend_capture():\n            sys.__stdout__.write(", "        with contextlib.nullcontext():\n            sys.__stdout__.write("),
    ("no leading newline before the workflow command", 'sys.__stdout__.write(f"\\n::warning title=', 'sys.__stdout__.write(f"::warning title='),
    ("annotation not flushed inside the suspension", "            sys.__stdout__.flush()\n", "            pass\n"),
    ("no local 'always' filter (-W error would make it a gate)", '        warnings.simplefilter("always", EpochsCompletedDriftWarning)\n', "        pass\n"),
    ("step summary truncated instead of appended", 'open(summary_path, "a", encoding="utf-8")', 'open(summary_path, "w", encoding="utf-8")'),
    ("summary written even when GITHUB_STEP_SUMMARY is unset", "        if summary_path:\n", "        if True:\n"),
    ("no step-summary line written", '                    handle.write(f"- {message}\\n")\n', "                    pass\n"),
    ("an unwritable step summary raises", "            except OSError as exc:\n", "            except ZeroDivisionError as exc:\n"),
    ("no warning issued", "        warnings.warn(message, EpochsCompletedDriftWarning, stacklevel=3)\n", "        pass\n"),
    ("GITHUB_ACTIONS read as truthy, not == 'true'", 'os.environ.get("GITHUB_ACTIONS") == "true"', 'os.environ.get("GITHUB_ACTIONS")'),
    ("type check dropped (68.0 == 68 reads as a match)", "    assert isinstance(observed, int) and not isinstance(observed, bool), ", "    assert True, "),
    ("bool accepted as an int", "isinstance(observed, int) and not isinstance(observed, bool)", "isinstance(observed, int)"),
    ("range check dropped", "    assert 1 <= observed <= budget, ", "    assert True, "),
    ("mismatch made blocking (the advisory becomes a gate)", "    _report_drift(message, suspend_capture=suspend_capture)\n    return message\n", "    raise AssertionError(message)\n"),
    ("a match is reported anyway", "    if observed == expected:\n        return None\n", "    if False:\n        return None\n"),
    ("thread pin never applied", "    torch.set_num_threads(width)\n", "    pass\n"),
    ("thread width not restored", "    finally:\n        torch.set_num_threads(previous)\n", "    finally:\n        pass\n"),
    ("message omits the thread pin", " (torch thread pin {REFERENCE_THREAD_PIN}). ", ". "),
    ("message omits the reference SHA", "Reference: cascor {REFERENCE_CASCOR_SHA[:7]}, ", "Reference: "),
    ("message does not start with the title (the job log drops it)", 'f"{ANNOTATION_TITLE} at budget {budget}: ', 'f"at budget {budget}: '),
    ("the real check drops capture suspension", "check_epochs_completed(budget, observed, suspend_capture=capsys.disabled)", "check_epochs_completed(budget, observed, suspend_capture=contextlib.nullcontext)"),
]


def _build_tree(dest: Path, source: str) -> None:
    for name in COPIED:
        shutil.copytree(ROOT / name, dest / name, ignore=IGNORED)
    shutil.copy2(ROOT / "pyproject.toml", dest / "pyproject.toml")
    (dest / TARGET).write_text(source, encoding="utf-8")


def _env(**extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_ACTIONS", "GITHUB_STEP_SUMMARY")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env


def _pytest(tree: Path, env: dict, *selector: str) -> subprocess.CompletedProcess:
    argv = [sys.executable, "-m", "pytest", "-m", "unit and not slow", TARGET, "-p", "no:cacheprovider", "--verbose", *selector]
    return subprocess.run(argv, cwd=tree, env=env, capture_output=True, text=True, timeout=600, check=False)  # nosec B603 -- fixed argv


def suite(source: str) -> str:
    """Run the module's tests; return "" when green, else a one-line reason."""
    with tempfile.TemporaryDirectory(prefix="d6-suite-") as tmp:
        tree = Path(tmp)
        _build_tree(tree, source)
        proc = _pytest(tree, _env())
    if proc.returncode == 0:
        return ""
    tail = [line for line in proc.stdout.splitlines() if line.startswith("FAILED") or " passed" in line or " failed" in line or "error" in line.lower()]
    return f"exit {proc.returncode}: " + (tail[-1] if tail else "(no summary line)")


def e2e(source: str) -> str:
    """Force a firing on the real check; return "" when it behaves, else what went wrong."""
    drifted = source.replace(REFERENCE, DRIFTED_REFERENCE)
    if drifted == source:
        return "reference line not found -- cannot drift it"
    with tempfile.TemporaryDirectory(prefix="d6-e2e-") as tmp:
        tree = Path(tmp)
        _build_tree(tree, drifted)
        summary = tree / "step_summary.md"
        proc = _pytest(tree, _env(GITHUB_ACTIONS="true", GITHUB_STEP_SUMMARY=str(summary)), "-k", "TestEpochsCompletedAdvisory")
        annotations = [line for line in proc.stdout.splitlines() if line.startswith(ANNOTATION_PREFIX)]
        summary_lines = summary.read_text(encoding="utf-8").splitlines() if summary.exists() else []
    problems = []
    if proc.returncode != 0:
        problems.append(f"exit {proc.returncode}")
    if len(annotations) != 1:
        problems.append(f"{len(annotations)} annotation lines at column 0")
    if len(summary_lines) != 1:
        problems.append(f"{len(summary_lines)} summary lines")
    return "; ".join(problems)


def main() -> int:
    source = (ROOT / TARGET).read_text(encoding="utf-8")
    for name, old, _ in MUTATIONS:
        if source.count(old) != 1:
            print(f"ANCHOR ERROR: {name!r}: expected 1 occurrence, found {source.count(old)}")
            return 1

    control_suite, control_e2e = suite(source), e2e(source)
    print(f"CONTROL  suite: {control_suite or 'green'} | e2e: {control_e2e or 'one annotation, one summary line, exit 0'}")
    if control_suite or control_e2e:
        print("CONTROL IS NOT CLEAN -- no mutation result below would mean anything.")
        return 1

    survivors = 0
    for index, (name, old, new) in enumerate(MUTATIONS, start=1):
        mutated = source.replace(old, new)
        by_suite, by_e2e = suite(mutated), e2e(mutated)
        caught = bool(by_suite or by_e2e)
        survivors += not caught
        print(f"M{index:02d} {'CAUGHT  ' if caught else 'SURVIVED'} {name}")
        print(f"      suite: {by_suite or 'green'}")
        print(f"      e2e:   {by_e2e or 'clean'}")

    print(f"\n{len(MUTATIONS) - survivors}/{len(MUTATIONS)} mutations caught, {survivors} survived.")
    return 0 if survivors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
