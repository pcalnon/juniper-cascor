#!/usr/bin/env python3
"""Mutation-check the paused-fit settle and the lifecycle-ownership guard: every listed mutant must be caught.

Project: juniper-cascor
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-11-08
Status: ad-hoc — investigation (verifies src/tests/unit/api/test_lifecycle_paused_fit_return.py
        and the mark_completed and lock tests in src/tests/unit/api/test_lifecycle_state_machine.py)
Retire when: the PR that settles a paused completion in one FSM transition has merged and
             nobody needs to re-run the evidence
Related: juniper-ml notes/JUNIPER_2026-11-06_JUNIPER-ECOSYSTEM_CURSOR-FLOOD-4-DISPOSITION.md §8
         ruling 11 (owner ruling "Fix all four now", 2026-11-08); §4 has service-core's copy

M00 swaps in the base ref's ``manager.py`` and ``state_machine.py`` (the whole fix reverted), so
the detector must fail exactly as it did before the fix. Each other mutant breaks one property
the change claims:

- the settle makes ONE transition: ``mark_completed`` accepts PAUSED as well as STARTED, and the
  pause gate is reopened after it (M01-M05; M02 restores the first version of this fix, which
  released the pause through RESUME and kept ``mark_completed`` STARTED-only);
- each of the five settles -- the post-fit stop, the completion, the interrupt, the failure and
  the monitor session end -- and each terminal session count is made only by the run that owns
  the lifecycle, and a run claims it as it enters ``_run_training``, before its START (M06-M20);
- the state machine makes each transition, and each read of more than one field, under its own
  lock, so a resume whose transition straddles the settle's cannot write STARTED over COMPLETED,
  and a fit that fails while paused cannot be settled inside another thread's transition (M21
  removes the lock altogether; M22-M31 each remove it from one of the ten methods that take it).

The detector is the new test module plus the state-machine suite, selected as CI selects them
(``-m "unit and not slow"``). Red = caught.

Hermetic, after ``2026-10-26_final_eval_accuracy_mutation_check.py``: each run copies ``src/``,
``conf/`` and ``pyproject.toml`` into a TemporaryDirectory (under ``$TMPDIR``) and mutates the
copy, so the working tree is never edited and no stale ``.pyc`` can answer for a mutant (a fresh
tree has no ``__pycache__``; ``PYTHONDONTWRITEBYTECODE=1`` besides). Every run also loads a
one-function pytest plugin written into the temporary directory, which prints the
``__file__`` pytest resolved for ``api.lifecycle.manager``; a run that did not import the
mutated copy counts as an error, never as a catch. The unmutated tree runs first as a control.
Exit codes are read from the subprocess directly, never through a pipe. Each mutant's edits
apply in order, and each edit's anchor must occur exactly once in the text it edits.

Usage:
    /opt/miniforge3/envs/JuniperCascor1/bin/python util/ad-hoc/2026-11-08_paused_fit_wedge_mutation_check.py [--base origin/main]
Exit: 0 when the control is green and every mutation is caught; 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess  # nosec B404 -- fixed-argv pytest and git runs
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANAGER = "src/api/lifecycle/manager.py"
STATE_MACHINE = "src/api/lifecycle/state_machine.py"
TESTS = ("src/tests/unit/api/test_lifecycle_paused_fit_return.py", "src/tests/unit/api/test_lifecycle_state_machine.py")
COPIED = ("src", "conf")
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "logs", "reports")
PLUGIN = "_paused_fit_provenance"
PLUGIN_SOURCE = """import sys


def pytest_sessionfinish(session, exitstatus):
    module = sys.modules.get("api.lifecycle.manager")
    print("\\nPROVENANCE " + str(getattr(module, "__file__", None)))
"""

# --- anchors in manager.py ----------------------------------------------------------------
_SETTLE = (
    "                    sm.mark_completed()\n"
    "                    # Reopen the gate a pause cleared. After the transition, not before: a pause\n"
    "                    # accepted in between would clear it again and leave it closed.\n"
    "                    self._pause_event.set()\n"
)
_GATE = '                    self._pause_event.set()\n                    state.update_state(status="Completed", phase="Idle")\n'
_COMPLETION_CHECK = '                if self._owns_lifecycle(run_token, "returned"):\n'
_CLAIM = "        self._lifecycle_owner = run_token\n"
_FIT_START = '            kwargs.setdefault("early_stopping", True)\n'
_MONITOR_END = "            if self._lifecycle_owner is run_token:\n                monitor.on_training_end()\n"
_FINAL_SCORE = "                final_scalar_metrics = self._compute_final_eval_metrics()\n"
# The first version of this fix: release the pause through RESUME, then mark_completed (STARTED only).
_ROUND_ONE_SETTLE = (
    "                    if self.state_machine.is_paused():\n"
    '                        self.logger.info("Training finished while paused; releasing it and settling Completed")\n'
    "                        self.state_machine.handle_command(Command.RESUME)\n"
    "                        self._pause_event.set()\n"
    "                    sm.mark_completed()\n"
)

# --- anchors in state_machine.py ----------------------------------------------------------
_ACCEPTS_PAUSED = '        if self._status in (TrainingStatus.STARTED, TrainingStatus.PAUSED):\n            prev = "Paused" if self._status == TrainingStatus.PAUSED else "Started"\n'
_STARTED_ONLY = '        if self._status == TrainingStatus.STARTED:\n            prev = "Started"\n'
_FSM_LOCK = "        self._lock = threading.RLock()\n"
_NO_LOCK = '        self._lock = __import__("contextlib").nullcontext()\n'
# Every method that holds the machine's lock (``@_atomic``), in the order M22-M31 unlock them.
_LOCKED_METHODS = ("handle_command", "mark_completed", "set_phase", "save_candidate_state", "get_candidate_state", "mark_failed", "mark_resume_ready", "mark_investigating", "mark_replaying", "get_state_summary")


def _counter_outside_guard(status: str, before: str, after: str) -> tuple:
    """Move one terminal ``inc_training_session_completed`` call out of its ownership guard."""
    old = f"{before}inc_training_session_completed({status})\n{after}"
    new = f"{before[:-4] if before.endswith('    ') else before}inc_training_session_completed({status})\n{after}"
    return (MANAGER, old, new)


def _unlocked(method: str) -> tuple:
    """Remove ``@_atomic`` from one state-machine method, so it runs without the machine's lock."""
    return (STATE_MACHINE, f"    @_atomic\n    def {method}(", f"    def {method}(")


# (name, [(file, old, new), ...]) -- the edits apply in order; each ``old`` must occur exactly once.
MUTATIONS = [
    ("mark_completed accepts STARTED only again (the FSM table reverted)", [(STATE_MACHINE, _ACCEPTS_PAUSED, _STARTED_ONLY)]),
    ("the first version of the settle: RESUME, then a STARTED-only mark_completed", [(MANAGER, _SETTLE, _ROUND_ONE_SETTLE), (STATE_MACHINE, _ACCEPTS_PAUSED, _STARTED_ONLY)]),
    ("the pause gate is not reopened after the settle", [(MANAGER, _GATE, '                    state.update_state(status="Completed", phase="Idle")\n')]),
    ("the pause gate is reopened before mark_completed, not after", [(MANAGER, _SETTLE, "                    self._pause_event.set()\n                    sm.mark_completed()\n")]),
    ("the pause gate is reopened before the ownership check", [(MANAGER, _COMPLETION_CHECK, "                self._pause_event.set()\n" + _COMPLETION_CHECK)]),
    ("the completion ignores ownership", [(MANAGER, 'if self._owns_lifecycle(run_token, "returned"):', "if True:")]),
    ("the post-fit stop ignores ownership", [(MANAGER, 'if self._owns_lifecycle(run_token, "returned with a stop requested"):', "if True:")]),
    ("the interrupt settle ignores ownership", [(MANAGER, 'if self._owns_lifecycle(run_token, "was interrupted"):', "if True:")]),
    ("the failure settle ignores ownership", [(MANAGER, 'if self._owns_lifecycle(run_token, "failed"):', "if True:")]),
    ("the monitor session end ignores ownership", [(MANAGER, _MONITOR_END, "            if True:\n                monitor.on_training_end()\n")]),
    ("_owns_lifecycle always answers yes", [(MANAGER, "        if self._lifecycle_owner is run_token:\n            return True\n", "        if True:\n            return True\n")]),
    ("every run shares one ownership token", [(MANAGER, "        run_token = object()\n", "        run_token = None\n")]),
    ("a run never takes the lifecycle", [(MANAGER, _CLAIM, "        pass\n")]),
    ("a run takes the lifecycle late, after its START, as its fit begins", [(MANAGER, _CLAIM, ""), (MANAGER, _FIT_START, "            self._lifecycle_owner = run_token\n" + _FIT_START)]),
    ("the final score is kept before the ownership check", [(MANAGER, _FINAL_SCORE, _FINAL_SCORE + "                self._final_scalar_metrics = final_scalar_metrics\n")]),
    ("every run clears the ownership token as it ends", [(MANAGER, _MONITOR_END, _MONITOR_END + "            self._lifecycle_owner = None\n")]),
    ("the completion's terminal count is outside the ownership guard", [_counter_outside_guard("TRAINING_SESSION_STATUS_SUCCESS", "                    self._broadcast_training_state(force=True)\n                    ", "        except TrainingInterrupted:\n")]),
    ("the post-fit stop's terminal count is outside the ownership guard", [_counter_outside_guard("TRAINING_SESSION_STATUS_CANCELLED", "                    self._broadcast_training_state(force=True)\n                    ", "            else:\n")]),
    ("the interrupt's terminal count is outside the ownership guard", [_counter_outside_guard("TRAINING_SESSION_STATUS_CANCELLED", "                self._broadcast_training_state(force=True)\n                ", "        except Exception as e:\n")]),
    ("the failure's terminal count is outside the ownership guard", [_counter_outside_guard("TRAINING_SESSION_STATUS_FAILURE", "                self._broadcast_training_state(force=True)\n                ", "            raise\n")]),
    ("the state machine takes no lock", [(STATE_MACHINE, _FSM_LOCK, _NO_LOCK)]),
    *[(f"{method} is not atomic", [_unlocked(method)]) for method in _LOCKED_METHODS],
]


def _apply(sources: dict, edits: list) -> tuple:
    """Apply one mutant's edits in order; return (mutated texts, error or None)."""
    texts = dict(sources)
    for relative, old, new in edits:
        count = texts[relative].count(old)
        if count != 1:
            return texts, f"anchor in {relative} occurs {count} times, expected 1: {old!r}"
        texts[relative] = texts[relative].replace(old, new)
    return texts, None


def _build_tree(dest: Path, replacements: dict) -> None:
    for name in COPIED:
        shutil.copytree(ROOT / name, dest / name, ignore=IGNORED)
    shutil.copy2(ROOT / "pyproject.toml", dest / "pyproject.toml")
    for relative, text in replacements.items():
        (dest / relative).write_text(text, encoding="utf-8")
    (dest / "_plugins").mkdir()
    (dest / "_plugins" / f"{PLUGIN}.py").write_text(PLUGIN_SOURCE, encoding="utf-8")


def _env(plugin_dir: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "JUNIPER_CASCOR_LOG_DIR"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(plugin_dir)
    return env


def suite(replacements: dict) -> tuple:
    """Run the detector on a mutated copy; return (exit code, failing test ids, provenance ok)."""
    with tempfile.TemporaryDirectory(prefix="paused-fit-wedge-") as tmp:
        tree = Path(tmp)
        _build_tree(tree, replacements)
        argv = [sys.executable, "-m", "pytest", "-m", "unit and not slow", *TESTS, "-p", "no:cacheprovider", "-p", PLUGIN, "-rf", "--timeout=120"]
        proc = subprocess.run(argv, cwd=tree, env=_env(tree / "_plugins"), capture_output=True, text=True, timeout=900, check=False)  # nosec B603 -- fixed argv
        expected = str(tree / MANAGER)
    failed = [line.split(" ", 1)[1].split(" - ", 1)[0] for line in proc.stdout.splitlines() if line.startswith("FAILED ")]
    provenance = [line.split(" ", 1)[1] for line in proc.stdout.splitlines() if line.startswith("PROVENANCE ")]
    return proc.returncode, failed, provenance == [expected]


def _base_blob(base: str, relative: str) -> str:
    proc = subprocess.run(["git", "-C", str(ROOT), "show", f"{base}:{relative}"], capture_output=True, text=True, check=True)  # nosec B603 B607 -- fixed argv
    return proc.stdout


def _report(label: str, name: str, code: int, failed: list, provenance_ok: bool) -> bool:
    caught = code != 0 and provenance_ok
    verdict = "CAUGHT  " if caught else ("NO-PROOF" if not provenance_ok else "SURVIVED")
    print(f"{label} {verdict} {name}  [exit {code}; {len(failed)} failed]", flush=True)
    for test_id in failed[:8]:
        print(f"      {test_id.rsplit('::', 1)[-1]}")
    if len(failed) > 8:
        print(f"      ... and {len(failed) - 8} more")
    return caught


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main", help="ref whose manager.py / state_machine.py M00 restores")
    args = parser.parse_args()

    sources = {MANAGER: (ROOT / MANAGER).read_text(encoding="utf-8"), STATE_MACHINE: (ROOT / STATE_MACHINE).read_text(encoding="utf-8")}
    mutants = []
    for name, edits in MUTATIONS:
        texts, error = _apply(sources, edits)
        if error is not None:
            print(f"ANCHOR ERROR: {name!r}: {error}")
            return 1
        mutants.append((name, {relative: text for relative, text in texts.items() if text != sources[relative]}))

    code, failed, provenance_ok = suite({})
    print(f"CONTROL  exit {code}, {len(failed)} failed, provenance {'ok' if provenance_ok else 'WRONG'}", flush=True)
    if code != 0 or not provenance_ok:
        print("CONTROL IS NOT GREEN -- no mutation result below would mean anything.")
        return 1

    caught = 0
    base = {MANAGER: _base_blob(args.base, MANAGER), STATE_MACHINE: _base_blob(args.base, STATE_MACHINE)}
    caught += _report("M00", f"the whole fix reverted ({args.base}'s manager.py and state_machine.py)", *suite(base))
    for index, (name, replacements) in enumerate(mutants, start=1):
        caught += _report(f"M{index:02d}", name, *suite(replacements))

    total = len(mutants) + 1
    print(f"\n{caught}/{total} mutations caught, {total - caught} not caught.")
    return 0 if caught == total else 1


if __name__ == "__main__":
    sys.exit(main())
