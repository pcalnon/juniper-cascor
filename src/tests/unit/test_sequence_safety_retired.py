"""Anti-resurrection guard for the packaged sequence-safety screens.

cascor's two compositional-loss screens (an AST symbol-loss screen + a markdown
deletion-magnitude screen) were ported inline from juniper-ml in cascor#482, then — in
the Wave-3 sequence-safety rollout — switched to the published ``juniper-ci-tools``
package (the ``juniper-symbol-loss-check`` / ``juniper-docs-additions-check`` console
scripts) and the inline ``util/sequence_safety/`` copy was deleted. The package is the
single source of truth now (ml canonical, consumers consume — the same shape the house
used for the doc-links validator and the dep-docs generator).

This is the always-on guard that the migration is not silently reversed:

 * ``util/sequence_safety/`` must not carry a Python module again (a resurrected inline
   copy is exactly the drift that packaging was meant to kill).
 * both advisory workflows (``sequence-safety.yml`` + ``main-verify.yml``) must pin
   ``juniper-ci-tools`` with a range that still admits the packaged version this
   migration targets (>= 0.8.0 — the release that introduced the two console scripts and
   the ``--scope`` knob), so a stale ceiling (e.g. ``<0.8.0``) that would silently stop
   the screens from installing is caught here rather than in a quiet green CI.
 * both workflows must still invoke the console scripts (proof the retrofit stayed wired).

Modeled on ``test_workflow_coverage_gate.py``: same ``REPO_ROOT`` walk-up, same
filesystem-only / hermetic style (no git, no network, no subprocess), same
``@pytest.mark.unit`` marker so it runs in the standard unit lane.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# .../src/tests/unit/this_file.py -> repo root is parent.parent.parent.parent
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
INLINE_DIR = REPO_ROOT / "util" / "sequence_safety"

# The juniper-ci-tools release that first shipped the two screens as console scripts
# (juniper-symbol-loss-check / juniper-docs-additions-check) plus the --scope knob cascor
# relies on. Every consumer pin must admit this version.
_CI_TOOLS_MIN = (0, 8, 0)

# The two advisory workflows that consume the packaged screens, and the console scripts
# they must invoke.
_SCREEN_WORKFLOWS = ("sequence-safety.yml", "main-verify.yml")
_CONSOLE_SCRIPTS = ("juniper-symbol-loss-check", "juniper-docs-additions-check")

# Matches a house ``juniper-ci-tools>=<lo>,<<hi>`` pin (the >=X,<Y minor window).
_PIN_RE = re.compile(r"juniper-ci-tools\s*>=\s*([0-9]+(?:\.[0-9]+)*)\s*,\s*<\s*([0-9]+(?:\.[0-9]+)*)")


def _ver(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _pad(parts: tuple[int, ...], width: int = 3) -> tuple[int, ...]:
    return parts + (0,) * max(0, width - len(parts))


def _inline_py_modules() -> list[Path]:
    if not INLINE_DIR.exists():
        return []
    return sorted(INLINE_DIR.rglob("*.py"))


def _ci_tools_pins(text: str) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return [(_pad(_ver(lo)), _pad(_ver(hi))) for lo, hi in _PIN_RE.findall(text)]


@pytest.mark.unit
class TestSequenceSafetyRetired:
    """Guard that the inline sequence_safety/ copy stays gone and the pins admit the package."""

    def test_workflows_directory_exists(self):
        assert WORKFLOWS_DIR.is_dir(), f"Expected {WORKFLOWS_DIR} to exist"

    def test_inline_copy_stays_deleted(self):
        """util/sequence_safety/ must never carry a Python module again."""
        stray = _inline_py_modules()
        assert not stray, "The inline sequence-safety copy has been resurrected under util/sequence_safety/: " + ", ".join(str(p.relative_to(REPO_ROOT)) for p in stray) + ". The screens are consumed from the juniper-ci-tools package now (juniper-symbol-loss-check / juniper-docs-additions-check) — do not re-add the inline copy. See this file's module docstring."

    def test_screen_workflow_pins_admit_packaged_version(self):
        """Each advisory workflow's juniper-ci-tools pin must still admit >= 0.8.0."""
        want = _pad(_CI_TOOLS_MIN)
        problems: list[str] = []
        for name in _SCREEN_WORKFLOWS:
            workflow = WORKFLOWS_DIR / name
            assert workflow.is_file(), f"missing screen workflow {name}"
            pins = _ci_tools_pins(workflow.read_text(encoding="utf-8"))
            if not any(lo <= want < hi for lo, hi in pins):
                problems.append(f"  - {name}: juniper-ci-tools pin(s) {pins or '[]'} do not admit {'.'.join(str(n) for n in _CI_TOOLS_MIN)}")
        assert not problems, "Sequence-safety workflow pin drift (a stale ceiling would silently stop the screens installing):\n" + "\n".join(problems)

    def test_screen_workflows_invoke_console_scripts(self):
        """Both workflows must still call the packaged console scripts (retrofit wiring)."""
        problems: list[str] = []
        for name in _SCREEN_WORKFLOWS:
            text = (WORKFLOWS_DIR / name).read_text(encoding="utf-8")
            for script in _CONSOLE_SCRIPTS:
                if script not in text:
                    problems.append(f"  - {name}: no longer invokes {script}")
        assert not problems, "Sequence-safety retrofit wiring lost:\n" + "\n".join(problems)
