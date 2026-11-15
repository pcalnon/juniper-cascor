#!/usr/bin/env python3
"""Negative control for the one-artifact publish: prove the src/tests/unit/test_publish_one_artifact.py cases can FAIL.

Project: juniper-cascor
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-11-15
Status: ad-hoc — investigation (negative control for the one-artifact publish change, the owner's directive of 2026-11-15)
Retire when: the one-artifact template steps in the three publish workflows are replaced or removed.
Related: .github/workflows/publish.yml, publish-cascor-model.yml and publish-protocol.yml (the steps marked "one-artifact
         publish template"); src/tests/unit/test_publish_one_artifact.py

WHY THIS EXISTS
---------------
A test that passes against both the fixed and the broken workflow pins nothing. This copies the three publish
workflows, the test file and pyproject.toml into a scratch directory, checks that the copy passes, then applies each
mutation below: one exact-match edit, to one file, or to every copy of a template step (so that the byte-identity case
cannot be what catches it). It runs the test file under pytest after each, restores the files, and reports any mutation
the tests still pass. The repository itself is never written.

The test file needs only pytest and PyYAML, so it runs with ``--noconftest`` (src/tests/conftest.py imports torch and
the service's modules, which no case here uses).

    python3 util/ad-hoc/2026-11-15_verify_publish_one_artifact_tests_are_not_vacuous.py [--scratch DIR] [--python PY]

The scratch directory defaults to a new one under the repository's gitignored ``tmp/``. It is left in place.

Exit status: 0 when the baseline passes and every mutation fails the tests; 1 otherwise; 2 when the scratch copy could
not be made or does not pass unmutated.
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess  # nosec B404 -- runs pytest with a fixed argv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PUBLISH = ".github/workflows/publish.yml"
MODEL = ".github/workflows/publish-cascor-model.yml"
PROTOCOL = ".github/workflows/publish-protocol.yml"
WORKFLOWS = (PUBLISH, MODEL, PROTOCOL)
TEST = "src/tests/unit/test_publish_one_artifact.py"
COPIED = (*WORKFLOWS, TEST, "pyproject.toml")

#: (label, files, old, new, mode): mode "one" requires exactly one match in each file; "all" replaces every match (at
#: least one in each file); "last" replaces the last match only.
MUTATIONS = (
    ("TestPyPI upload skips an existing file", (PUBLISH,), "          repository-url: https://test.pypi.org/legacy/\n          packages-dir: ${{ env.PUBLISH_DIST_DIR }}/\n", "          repository-url: https://test.pypi.org/legacy/\n          packages-dir: ${{ env.PUBLISH_DIST_DIR }}/\n          skip-existing: true\n", "one"),
    ("the PyPI job builds its own files again", (PUBLISH,), "      - name: Publish to PyPI\n", "      - name: Build package\n        run: python -m build\n\n      - name: Publish to PyPI\n", "one"),
    ("one copy of the JSON check drifts", (PUBLISH,), '"HASH_CHECK_ATTEMPTS", "20"', '"HASH_CHECK_ATTEMPTS", "21"', "last"),
    ("a file with another sha256 is not refused", WORKFLOWS, "                  if wrong:\n", "                  if wrong and False:\n", "all"),
    ("a file the run did not build is ignored", WORKFLOWS, "if expected.get(name) != digest)", "if name in expected and expected.get(name) != digest)", "all"),
    ("the dist check ignores an extra file", WORKFLOWS, "if not recorded or present != recorded:", "if not recorded or {k: present.get(k) for k in recorded} != recorded:", "all"),
    ("the record step takes any file", WORKFLOWS, "          if not names or strays:\n", "          if not names:\n", "all"),
    ("the artifact expires before the gate", (PUBLISH,), "          retention-days: 90\n", "          retention-days: 7\n", "one"),
    ("the PyPI check reads TestPyPI", (PUBLISH,), "          INDEX_URL: https://pypi.org\n", "          INDEX_URL: https://test.pypi.org\n", "one"),
    ("no TestPyPI check before the PyPI upload", (PUBLISH,), "      - name: Verify TestPyPI still serves exactly these files\n", "      - name: Verify TestPyPI still serves these files\n", "one"),
    ("no TestPyPI check after a sub-package's TestPyPI upload", (MODEL,), "      - name: Verify TestPyPI serves exactly these files\n", "      - name: Verify TestPyPI serves these files\n", "one"),
    ("the PyPI job no longer waits for TestPyPI", (PUBLISH,), "    needs: testpypi\n", "    needs: build\n", "one"),
    ("the JSON check polls without bound", WORKFLOWS, '"HASH_CHECK_ATTEMPTS", "20"', '"HASH_CHECK_ATTEMPTS", "2000"', "all"),
    ("a 404 is not polled through", WORKFLOWS, "except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as exc:", "except (ValueError, KeyError, TypeError) as exc:", "all"),
    ("a template step reads a workflow expression", WORKFLOWS, "          # Hash every file the build wrote", "          # ${{ github.repository }}: hash every file the build wrote", "all"),
    ("another package's file is accepted", WORKFLOWS, "if normalized(project) != normalized(package) or not version", "if not version", "all"),
    ("publish.yml also fires on a tag push", (PUBLISH,), "on:\n  release:\n    types: [published]\n", "on:\n  release:\n    types: [published]\n  push:\n    tags: ['v*']\n", "one"),
    ("publish.yml's permissions change", (PUBLISH,), "permissions:\n  id-token: write\n", "permissions:\n  id-token: write\n  contents: read\n", "one"),
    ("the TestPyPI job is renamed", (PUBLISH,), "    name: Publish to TestPyPI\n", "    name: Publish to Test PyPI\n", "one"),
    ("the build runs inside the testpypi environment", (PUBLISH,), "    if: startsWith(github.event.release.tag_name, 'v')\n    runs-on: ubuntu-latest\n", "    if: startsWith(github.event.release.tag_name, 'v')\n    runs-on: ubuntu-latest\n    environment: testpypi\n", "one"),
    ("the TestPyPI upload reads another directory", (PUBLISH,), "          repository-url: https://test.pypi.org/legacy/\n          packages-dir: ${{ env.PUBLISH_DIST_DIR }}/\n", "          repository-url: https://test.pypi.org/legacy/\n          packages-dir: dist-other/\n", "one"),
    ("a checkout runs after the PyPI job's download", (PUBLISH,), "      # One-artifact publish template, step 4: before every upload.\n      - name: Verify the dist against SHA256SUMS\n", "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1\n\n      # One-artifact publish template, step 4: before every upload.\n      - name: Verify the dist against SHA256SUMS\n", "last"),
    ("the tag guard leaves the build job", (PUBLISH,), "    if: startsWith(github.event.release.tag_name, 'v')\n", "", "one"),
    ("the install check runs before the sha256 check", (PUBLISH,), "      # One-artifact publish template, step 5: after the TestPyPI upload, before the install check.\n", "      - name: Verify TestPyPI install early\n        run: pip install --no-deps --index-url https://test.pypi.org/simple/ juniper-cascor\n\n      # One-artifact publish template, step 5: after the TestPyPI upload, before the install check.\n", "one"),
)


def run_tests(python: str, scratch: Path) -> int:
    proc = subprocess.run([python, "-m", "pytest", TEST, "-p", "no:cacheprovider", "--noconftest", "-x", "-q"], cwd=scratch, capture_output=True, text=True, timeout=900, check=False)  # nosec B603 -- fixed argv
    return proc.returncode


def mutate(text: str, old: str, new: str, mode: str) -> "str | None":
    """Return the mutated text, or None when the edit does not apply as its mode requires."""
    count = text.count(old)
    if mode == "one":
        return text.replace(old, new) if count == 1 else None
    if mode == "all":
        return text.replace(old, new) if count >= 1 else None
    if count < 1:
        return None
    head, _, tail = text.rpartition(old)
    return head + new + tail


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--scratch", type=Path, default=REPO / "tmp" / f"verify_publish_one_artifact-{stamp}", help="an empty or absent directory to copy into")
    parser.add_argument("--python", default=sys.executable, help="an interpreter with pytest, pytest-asyncio and PyYAML")
    args = parser.parse_args(argv)
    if args.scratch.exists() and any(args.scratch.iterdir()):
        print(f"REFUSED: {args.scratch} is not empty")
        return 2
    for rel in COPIED:
        (args.scratch / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, args.scratch / rel)
    originals = {rel: (args.scratch / rel).read_text(encoding="utf-8") for rel in WORKFLOWS}
    baseline = run_tests(args.python, args.scratch)
    print(f"scratch: {args.scratch}")
    print(f"baseline: pytest exit {baseline}")
    if baseline != 0:
        print("REFUSED: the unmutated copy does not pass")
        return 2
    survivors = []
    for label, files, old, new, mode in MUTATIONS:
        mutated = {rel: mutate(originals[rel], old, new, mode) for rel in files}
        if any(text is None or text == originals[rel] for rel, text in mutated.items()):
            print(f"DID NOT APPLY: {label}")
            survivors.append(label)
            continue
        try:
            for rel, text in mutated.items():
                (args.scratch / rel).write_text(str(text), encoding="utf-8")
            code = run_tests(args.python, args.scratch)
        finally:
            for rel in mutated:
                (args.scratch / rel).write_text(originals[rel], encoding="utf-8")
        print(f"{'caught' if code != 0 else 'SURVIVED'}: {label} (pytest exit {code})")
        if code == 0:
            survivors.append(label)
    print(f"{len(MUTATIONS) - len(survivors)} of {len(MUTATIONS)} mutations caught")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
