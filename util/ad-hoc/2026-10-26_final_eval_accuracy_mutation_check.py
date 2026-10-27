#!/usr/bin/env python3
"""Mutation-check the final evaluation's ``accuracy``: every deliberate break must be caught.

Project: juniper-cascor
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-10-26
Status: ad-hoc — investigation (verifies the tests added with ``compute_classification_accuracy``)
Retire when: the PR that adds ``eval_metrics.final.accuracy`` has merged and nobody needs to re-run the evidence
Related: juniper-ml notes/JUNIPER_2026-08-14_JUNIPER-ECOSYSTEM_DEFECT-REGISTER.md row APD-ML-002 (owner ruling
         2026-10-26: "Keep test_accuracy via cascor"); juniper-ml#2064;
         src/tests/unit/api/test_classification_metrics.py and src/tests/unit/api/test_final_eval_accuracy.py

Each mutant breaks one property the change claims: the definition shared with the training
history (``CascadeCorrelationNetwork._accuracy``), the guards shared with the sibling metrics,
the rows that the block's ``split`` label names, and the unchanged live block. The detector is
the two test modules, selected as CI selects them (``-m "unit and not slow"``). Red = caught.

Hermetic, after ``2026-09-23_d6_advisory_mutation_check.py``: each run copies ``src/``,
``conf/`` and ``pyproject.toml`` into a TemporaryDirectory (under ``$TMPDIR``) and mutates the
copy, so the working tree is never edited and no stale ``.pyc`` can answer for a mutant (a fresh
tree has no ``__pycache__``; ``PYTHONDONTWRITEBYTECODE=1`` besides). The unmutated tree runs
first as a control: a detector that fires without a mutation proves nothing when it fires with
one. Exit codes are read from the subprocess directly, never through a pipe.

Usage:
    /opt/miniforge3/envs/JuniperCascor1/bin/python util/ad-hoc/2026-10-26_final_eval_accuracy_mutation_check.py
Exit: 0 when the control is green and every mutation is caught; 1 otherwise.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404 -- fixed-argv pytest runs in a temp tree
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = "src/api/lifecycle/classification_metrics.py"
MANAGER = "src/api/lifecycle/manager.py"
TESTS = ("src/tests/unit/api/test_classification_metrics.py", "src/tests/unit/api/test_final_eval_accuracy.py")
COPIED = ("src", "conf")
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "logs", "reports")

# (name, file, old, new) -- ``old`` must occur exactly once in ``file``.
MUTATIONS = [
    ("output threshold off by one (> 0.5 becomes >= 0.5)", METRICS, "        y_pred = (output.reshape(-1) > 0.5)", "        y_pred = (output.reshape(-1) >= 0.5)"),
    ("target threshold off by one (> 0.5 becomes >= 0.5)", METRICS, "        y_true = (target.reshape(-1) > 0.5)", "        y_true = (target.reshape(-1) >= 0.5)"),
    ("prediction argmax on the wrong axis", METRICS, "        y_pred = torch.argmax(output, dim=1)\n        y_true", "        y_pred = torch.argmax(output, dim=0)\n        y_true"),
    ("target argmax on the wrong axis", METRICS, "        y_true = torch.argmax(target, dim=1)\n    correct", "        y_true = torch.argmax(target, dim=0)\n    correct"),
    ("one output column argmax-decoded (always class 0)", METRICS, "    if int(output.shape[1]) == 1:\n", "    if int(output.shape[1]) == 0:\n"),
    ("denominator off by one", METRICS, "    return correct / n_samples, None\n", "    return correct / (n_samples + 1), None\n"),
    ("empty batch returns NaN, as _accuracy does", METRICS, '        return None, "empty_batch"\n', '        return float("nan"), None\n'),
    ("non-finite batch scored instead of degraded", METRICS, '        return None, "invalid_output"\n', "        pass\n"),
    ("single-class target degraded like the siblings", METRICS, "    correct = int((y_pred == y_true).sum())\n", '    if int(torch.unique(y_true).numel()) < 2:\n        return None, "single_class"\n    correct = int((y_pred == y_true).sum())\n'),
    ("accuracy joins the live metric set", METRICS, 'METRIC_KEYS: Tuple[str, ...] = ("f1", "precision", "recall", "roc_auc")', 'METRIC_KEYS: Tuple[str, ...] = ("f1", "precision", "recall", "roc_auc", "accuracy")'),
    ("accuracy scored on the in-loop split, not the labelled one", MANAGER, "accuracy, accuracy_undefined = compute_classification_accuracy(output, y)", "accuracy, accuracy_undefined = compute_classification_accuracy(network.forward(self._eval_split()[0]), self._eval_split()[1])"),
    ("accuracy never attached to the final block", MANAGER, '            metrics["accuracy"] = accuracy\n', "            pass\n"),
    ("undefined reason not recorded", MANAGER, '                undefined["accuracy"] = accuracy_undefined\n', "                pass\n"),
    ("a top-level accuracy key leaks into /v1/metrics", MANAGER, '            "roc_auc": scalars.get("roc_auc"),\n', '            "roc_auc": scalars.get("roc_auc"),\n            "accuracy": None,\n'),
]


def _build_tree(dest: Path, replacements: dict) -> None:
    for name in COPIED:
        shutil.copytree(ROOT / name, dest / name, ignore=IGNORED)
    shutil.copy2(ROOT / "pyproject.toml", dest / "pyproject.toml")
    for relative, text in replacements.items():
        (dest / relative).write_text(text, encoding="utf-8")


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if k != "JUNIPER_CASCOR_LOG_DIR"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def suite(replacements: dict) -> tuple:
    """Run the two test modules on a mutated copy; return (exit code, failing test ids)."""
    with tempfile.TemporaryDirectory(prefix="final-eval-accuracy-") as tmp:
        tree = Path(tmp)
        _build_tree(tree, replacements)
        argv = [sys.executable, "-m", "pytest", "-m", "unit and not slow", *TESTS, "-p", "no:cacheprovider", "-rf", "--timeout=300"]
        proc = subprocess.run(argv, cwd=tree, env=_env(), capture_output=True, text=True, timeout=900, check=False)  # nosec B603 -- fixed argv
    failed = [line.split(" ", 1)[1].split(" - ", 1)[0] for line in proc.stdout.splitlines() if line.startswith("FAILED ")]
    return proc.returncode, failed


def main() -> int:
    sources = {METRICS: (ROOT / METRICS).read_text(encoding="utf-8"), MANAGER: (ROOT / MANAGER).read_text(encoding="utf-8")}
    for name, relative, old, _ in MUTATIONS:
        if sources[relative].count(old) != 1:
            print(f"ANCHOR ERROR: {name!r}: expected 1 occurrence in {relative}, found {sources[relative].count(old)}")
            return 1

    code, failed = suite({})
    print(f"CONTROL  exit {code}, {len(failed)} failed")
    if code != 0:
        print("CONTROL IS NOT GREEN -- no mutation result below would mean anything.")
        return 1

    survivors = 0
    for index, (name, relative, old, new) in enumerate(MUTATIONS, start=1):
        code, failed = suite({relative: sources[relative].replace(old, new)})
        caught = code != 0
        survivors += not caught
        print(f"M{index:02d} {'CAUGHT  ' if caught else 'SURVIVED'} {name}  [{relative.rsplit('/', 1)[-1]}; exit {code}; {len(failed)} failed]")
        for test_id in failed[:4]:
            print(f"      {test_id.rsplit('/', 1)[-1]}")
        if len(failed) > 4:
            print(f"      ... and {len(failed) - 4} more")

    print(f"\n{len(MUTATIONS) - survivors}/{len(MUTATIONS)} mutations caught, {survivors} survived.")
    return 0 if survivors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
