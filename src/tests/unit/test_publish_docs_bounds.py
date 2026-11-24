"""Pin the publish docs' numbers to the publish workflows: the bounds the docs state are the bounds the steps run.

``docs/ci_cd/MANUAL.md`` and ``docs/ci_cd/REFERENCE.md`` give an operator the numbers that decide whether a failed
check was lag: how often and how far apart the JSON checks poll, how long each request may take, how many times the
install checks try and how long they wait. Each is read here from the workflows and required in the docs, so a change
to either side alone fails. A module of its own: ``test_publish_one_artifact.py`` reads the workflows alone, and its
non-vacuity check (``util/ad-hoc/2026-11-15_verify_publish_one_artifact_tests_are_not_vacuous.py``) copies only those.

Project: juniper-cascor
Author: Paul Calnon
Created: 2026-11-23
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# CI's unit lane runs ``pytest -m "unit and not slow" src/tests/unit``, so an unmarked test is collected and then
# silently DESELECTED. Module-level, so it covers tests added later.
pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO / ".github" / "workflows"
DOCS = REPO / "docs" / "ci_cd"
#: The JSON check every copy of the template runs; its first copy, in publish.yml's TestPyPI job, is read here.
INDEX_CHECK = "Verify TestPyPI serves exactly these files"
INSTALL = "verify-testpypi-install"
#: workflow -> its install check's step name
INSTALL_STEPS = {"publish.yml": "Verify TestPyPI install", "publish-cascor-model.yml": "Verify install from TestPyPI (no-deps, version-only import)", "publish-protocol.yml": "Verify install from TestPyPI"}


def _jobs(workflow: str) -> dict:
    return yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))["jobs"]


def _run(job: dict, name: str) -> str:
    found = [s for s in job.get("steps") or [] if s.get("name") == name]
    assert len(found) == 1, f"expected exactly one step named {name!r}"
    return str(found[0]["run"])


def _doc(name: str) -> str:
    """The doc's text with each run of whitespace made one space: a rewrapped paragraph states the same bounds."""
    return " ".join((DOCS / name).read_text(encoding="utf-8").split())


def _retry(workflow: str) -> tuple[int, list[int], list[int]]:
    """The install check's (tries, the waits inside its loop, the waits before it), read from its commands."""
    run = _run(_jobs(workflow)[INSTALL], INSTALL_STEPS[workflow])
    commands = [line.strip() for line in run.splitlines() if line.strip() and not line.strip().startswith("#")]
    loops = [i for i, c in enumerate(commands) if re.fullmatch(r"for attempt in [0-9 ]+; do", c)]
    assert len(loops) == 1, f"{workflow}: expected one retry loop"
    start, end = loops[0], commands.index("done", loops[0])
    tries = len(commands[start].split()[3:-1])
    waits = [int(c.split()[1]) for c in commands[start:end] if re.fullmatch(r"sleep [0-9]+", c)]
    before = [int(c.split()[1]) for c in commands[:start] if re.fullmatch(r"sleep [0-9]+", c)]
    return tries, waits, before


class TestPublishDocsStateTheBounds:
    def test_the_json_checks_bounds(self):
        run = _run(_jobs("publish.yml")["testpypi"], INDEX_CHECK)
        attempts = int(re.search(r'"HASH_CHECK_ATTEMPTS", "(\d+)"', run).group(1))
        interval = int(re.search(r'"HASH_CHECK_INTERVAL", "(\d+)"', run).group(1))
        timeout = int(re.search(r"urllib\.request\.urlopen\(request, timeout=(\d+)\)", run).group(1))
        slept = (attempts - 1) * interval
        # The worst case: every sleep, and every request running to its timeout.
        minutes = round((slept + attempts * timeout) / 60)
        reference = _doc("REFERENCE.md")
        assert f"The JSON checks poll up to {attempts} times, {interval} s apart" in reference
        assert f"{slept} s of sleep, plus up to {attempts} request timeouts of {timeout} s" in reference
        assert f"A JSON check gives up after {attempts} attempts | The index never listed the version or a file within the bound: {attempts - 1} sleeps of {interval} s ({slept} s) plus up to {attempts} request timeouts of {timeout} s, about {minutes} minutes at worst" in _doc("MANUAL.md")

    def test_the_install_checks_retry(self):
        found = {workflow: _retry(workflow) for workflow in INSTALL_STEPS}
        tries, waits, _ = found["publish.yml"]
        # One loop in all three: its tries, a wait between them and none after the last; publish.yml waits once first.
        assert {workflow: (n, w) for workflow, (n, w, _) in found.items()} == dict.fromkeys(INSTALL_STEPS, (tries, waits))
        assert len(waits) == 1
        assert {workflow: before for workflow, (_, _, before) in found.items()} == {"publish.yml": found["publish.yml"][2], "publish-cascor-model.yml": [], "publish-protocol.yml": []}
        assert len(found["publish.yml"][2]) == 1
        wait, first = waits[0], found["publish.yml"][2][0]
        retry = f"{tries} tries, {wait} s apart"
        reference = _doc("REFERENCE.md")
        assert reference.count(f"; {retry} |") == len(INSTALL_STEPS)
        assert f"after a {first} s wait; {retry} |" in reference
        assert f"Each install check makes {retry} (`publish.yml` waits {first} s first)" in _doc("MANUAL.md")
        # The error each check stops with after its last try, as each step prints it and as the docs quote it.
        error = f"is not installable from TestPyPI after {tries} tries (index lag?)"
        for workflow in INSTALL_STEPS:
            assert error in _run(_jobs(workflow)[INSTALL], INSTALL_STEPS[workflow])
        for name in ("REFERENCE.md", "MANUAL.md"):
            assert error in _doc(name)
