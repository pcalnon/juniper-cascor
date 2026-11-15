"""Pin the one-artifact publish: PyPI takes exactly the files TestPyPI took, and the run checks it by sha256.

The owner's directive of 2026-11-15: "deploy to pypi should send the same artifact that was already sent to testpypi.
an identical hash check, of the artifacts after testpypi deploy and prior to pypi deploy, might also be helpful here to
prevent this from slipping through the cracks again." ``publish.yml`` ran ``python -m build`` in its TestPyPI job and
again in its PyPI job, so PyPI served a second build of each release: the juniper-cascor 0.13.0 wheel is
``eda26e26...`` on PyPI and ``f5dbc1b6...`` on TestPyPI, and the 0.14.0 wheel ``b70f3d39...`` against ``c5c9a5dc...``.

All three publishers now build once and hand both publish jobs one artifact, the dist directory plus ``SHA256SUMS``.
The steps that carry it are a template other Juniper publishers copy verbatim: they read only ``PUBLISH_PACKAGE`` and
``PUBLISH_DIST_DIR`` from the workflow ``env``, and a JSON check's ``INDEX_URL``.

The static cases pin the wiring: one build, in a job with no environment; the job chain and the two job names
juniper-ml's release-train ceremony reads; each upload preceded by the dist check; TestPyPI's JSON checked after the
TestPyPI upload and again before the PyPI one, and PyPI's after; no ``skip-existing``; an artifact that outlives the
30-day ``pypi`` gate; the triggers, permissions and tag guards as they were; and every copy of a template step
byte-identical. The rehearsals run each template step's own Python, extracted from ``publish.yml``, in a scratch
workspace, and run the JSON check against a stand-in index on 127.0.0.1: it passes on an exact match, polls through a
404, a missing file or an unreadable answer, fails at once on a file with another sha256 or one the run did not build,
refuses a ``SHA256SUMS`` of another package or of two versions before it asks, and gives up after its bounded
attempts.

Project: juniper-cascor
Author: Paul Calnon
Created: 2026-11-15
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import shutil
import subprocess  # nosec B404 - runs the workflow's OWN extracted step scripts with a fixed argv
import sys
import threading
from pathlib import Path

import pytest
import yaml

# CI's unit lane runs ``pytest -m "unit and not slow" src/tests/unit``, so an unmarked test is collected and then
# silently DESELECTED. Module-level, so it covers tests added later.
pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO / ".github" / "workflows"
#: workflow -> (PUBLISH_PACKAGE, PUBLISH_DIST_DIR, TestPyPI job, PyPI job); each also has a ``build`` job.
PUBLISHERS = {
    "publish.yml": ("juniper-cascor", "dist", "testpypi", "pypi"),
    "publish-cascor-model.yml": ("juniper-cascor-model", "juniper-cascor-model/dist", "publish-testpypi", "publish-pypi"),
    "publish-protocol.yml": ("juniper-cascor-protocol", "juniper-cascor-protocol/dist", "publish-testpypi", "publish-pypi"),
}
RECORD = "Record SHA256SUMS"
UPLOAD = "Upload dist artifacts"
DOWNLOAD = "Download dist artifacts"
VERIFY_DIST = "Verify the dist against SHA256SUMS"
TESTPYPI_AFTER = "Verify TestPyPI serves exactly these files"
TESTPYPI_BEFORE = "Verify TestPyPI still serves exactly these files"
PYPI_AFTER = "Verify PyPI serves exactly these files"
INDEX_CHECKS = {TESTPYPI_AFTER: "https://test.pypi.org", TESTPYPI_BEFORE: "https://test.pypi.org", PYPI_AFTER: "https://pypi.org"}
#: Everything a template step may read from its environment: the two parameters, the step's index, the runner's workspace.
TEMPLATE_ENV = {"GITHUB_WORKSPACE", "PUBLISH_PACKAGE", "PUBLISH_DIST_DIR", "INDEX_URL", "HASH_CHECK_ATTEMPTS", "HASH_CHECK_INTERVAL"}
PUBLISH_ACTION = "pypa/gh-action-pypi-publish@"
BUILD_COMMAND = re.compile(r"\bpython3?\s+-m\s+build\b|\bpip3?\s+wheel\b|\buv\s+build\b|\bsetup\.py\s+(?:sdist|bdist_wheel)\b")
#: GitHub lets a deployment wait this many days for its reviewer: the PyPI job may start that late, and needs the artifact.
GATE_APPROVAL_DAYS = 30
ARTIFACT = "${{ env.PUBLISH_PACKAGE }}-dist"
DIST_INPUT = "${{ env.PUBLISH_DIST_DIR }}/"
WHEEL = "juniper_cascor-9.9.9-py3-none-any.whl"
SDIST = "juniper_cascor-9.9.9.tar.gz"
BUILT = {WHEEL: b"wheel bytes", SDIST: b"sdist bytes"}


def _workflow(name: str) -> dict:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
    data["on"] = data.pop(True, data.get("on"))
    return data


def _names(job: dict) -> list[str]:
    return [str(s.get("name") or s.get("uses") or "") for s in job.get("steps") or []]


def _index(job: dict, name: str) -> int:
    names = _names(job)
    assert names.count(name) == 1, f"expected exactly one step named {name!r}, found {names.count(name)} in {names}"
    return names.index(name)


def _step(job: dict, name: str) -> dict:
    return job["steps"][_index(job, name)]


def _uploads(job: dict) -> list[int]:
    return [i for i, s in enumerate(job.get("steps") or []) if str(s.get("uses") or "").startswith(PUBLISH_ACTION)]


def _environment(job: dict) -> str:
    env = job.get("environment")
    return str(env.get("name") if isinstance(env, dict) else env)


def _assert_no_checkout_between(job: dict, start: int, end: int) -> None:
    """A checkout empties the workspace, so one between the download and the last template step would lose the artifact."""
    between = [str(s.get("uses") or "") for s in job["steps"][start:end]]
    assert not [u for u in between if u.startswith("actions/checkout@")], f"a checkout runs between steps {start} and {end}: {between}"


def _template_scripts() -> dict[str, str]:
    """The template steps' Python, as ``publish.yml`` runs it; every copy elsewhere must equal these."""
    jobs = _workflow("publish.yml")["jobs"]
    return {RECORD: _step(jobs["build"], RECORD)["run"], VERIFY_DIST: _step(jobs["testpypi"], VERIFY_DIST)["run"], "index": _step(jobs["testpypi"], TESTPYPI_AFTER)["run"]}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _script(kind: str, tmp_path: Path) -> Path:
    path = tmp_path / f"{re.sub(r'[^a-z]+', '_', kind.lower())}.py"
    path.write_text(_template_scripts()[kind], encoding="utf-8")
    return path


def _run(script: Path, workspace: Path, cwd: Path | None = None, **env: str) -> subprocess.CompletedProcess[str]:
    """Run an extracted step as the runner does (``python3 {0}``), with a minimal environment rather than a copy of ours."""
    child = {"PATH": os.environ.get("PATH", ""), "GITHUB_WORKSPACE": str(workspace), "PUBLISH_PACKAGE": "juniper-cascor", "PUBLISH_DIST_DIR": "dist"}
    child.update(env)
    return subprocess.run([sys.executable, "-I", str(script)], cwd=cwd or workspace, env=child, capture_output=True, text=True, timeout=120, check=False)  # nosec B603 - fixed argv: this interpreter and an extracted step


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "dist").mkdir(parents=True)
    for name, data in BUILT.items():
        (root / "dist" / name).write_bytes(data)
    return root


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Wiring: one build, one artifact, a check before every upload, and the names the release train reads
# ─────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
class TestWiring:
    def test_the_template_parameters_name_this_package(self, workflow):
        package, dist_dir, _, _ = PUBLISHERS[workflow]
        assert _workflow(workflow)["env"] == {"PUBLISH_PACKAGE": package, "PUBLISH_DIST_DIR": dist_dir}

    def test_only_the_build_job_builds_and_it_holds_no_environment(self, workflow):
        jobs = _workflow(workflow)["jobs"]
        builders = sorted(job_id for job_id, job in jobs.items() if any(BUILD_COMMAND.search(str(s.get("run") or "")) for s in job.get("steps") or []))
        assert builders == ["build"]
        assert jobs["build"]["name"] == "Build and Validate"
        assert "environment" not in jobs["build"]
        assert not _uploads(jobs["build"])

    def test_the_publish_jobs_chain_on_the_build_under_the_names_the_release_train_reads(self, workflow):
        _, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        assert sorted(jobs) == sorted(["build", test_id, prod_id])
        assert jobs[test_id]["needs"] == "build"
        assert jobs[prod_id]["needs"] == test_id
        # juniper-ml's util/release_train/ceremony.py tells the two jobs apart by name ("testpypi" / "pypi").
        assert jobs[test_id]["name"] == "Publish to TestPyPI"
        assert jobs[prod_id]["name"] == "Publish to PyPI"
        assert _environment(jobs[test_id]) == "testpypi"
        assert _environment(jobs[prod_id]) == "pypi"

    def test_the_testpypi_job_checks_the_dist_uploads_then_checks_testpypi_before_the_install_check(self, workflow):
        _, _, test_id, _ = PUBLISHERS[workflow]
        job = _workflow(workflow)["jobs"][test_id]
        uploads = _uploads(job)
        assert len(uploads) == 1
        assert job["steps"][uploads[0]]["with"]["repository-url"] == "https://test.pypi.org/legacy/"
        assert _index(job, DOWNLOAD) < _index(job, VERIFY_DIST) < uploads[0] < _index(job, TESTPYPI_AFTER)
        installs = [i for i, s in enumerate(job["steps"]) if "pip install" in str(s.get("run") or "")]
        assert installs, "the TestPyPI install check is gone"
        assert min(installs) > _index(job, TESTPYPI_AFTER), "the install check must come after the sha256 check that vouches for what it installs"
        _assert_no_checkout_between(job, _index(job, DOWNLOAD), _index(job, TESTPYPI_AFTER))

    def test_the_pypi_job_rechecks_testpypi_before_its_upload_and_pypi_after(self, workflow):
        _, _, _, prod_id = PUBLISHERS[workflow]
        job = _workflow(workflow)["jobs"][prod_id]
        uploads = _uploads(job)
        assert len(uploads) == 1
        assert "repository-url" not in (job["steps"][uploads[0]].get("with") or {})
        assert _index(job, DOWNLOAD) < _index(job, VERIFY_DIST) < _index(job, TESTPYPI_BEFORE) < uploads[0] < _index(job, PYPI_AFTER)
        _assert_no_checkout_between(job, _index(job, DOWNLOAD), _index(job, PYPI_AFTER))

    def test_every_upload_takes_the_downloaded_dist_and_none_skips_an_existing_file(self, workflow):
        for job in _workflow(workflow)["jobs"].values():
            for i in _uploads(job):
                inputs = job["steps"][i].get("with") or {}
                assert inputs.get("packages-dir") == DIST_INPUT
                # skip-existing would keep an earlier file under the same name, and the install check would test it.
                assert "skip-existing" not in inputs

    def test_one_artifact_carries_the_dist_and_sha256sums_and_outlives_the_pypi_gate(self, workflow):
        _, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        upload = _step(jobs["build"], UPLOAD)["with"]
        assert upload["name"] == ARTIFACT
        assert [p.strip() for p in upload["path"].splitlines() if p.strip()] == [DIST_INPUT, "SHA256SUMS"]
        assert upload["if-no-files-found"] == "error"
        assert int(upload["retention-days"]) > GATE_APPROVAL_DAYS
        assert _index(jobs["build"], RECORD) < _index(jobs["build"], UPLOAD)
        for job_id in (test_id, prod_id):
            assert _step(jobs[job_id], DOWNLOAD)["with"] == {"name": ARTIFACT, "path": "${{ github.workspace }}"}

    def test_each_json_check_reads_its_own_index_and_keeps_the_default_bounds(self, workflow):
        _, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        for job_id, name in ((test_id, TESTPYPI_AFTER), (prod_id, TESTPYPI_BEFORE), (prod_id, PYPI_AFTER)):
            assert _step(jobs[job_id], name)["env"] == {"INDEX_URL": INDEX_CHECKS[name]}


class TestUnchanged:
    def test_publish_yml_keeps_its_trigger_permissions_and_tag_guard(self):
        wf = _workflow("publish.yml")
        assert wf["on"] == {"release": {"types": ["published"]}}
        assert wf["permissions"] == {"id-token": "write"}
        assert "concurrency" not in wf
        assert wf["jobs"]["build"]["if"] == "startsWith(github.event.release.tag_name, 'v')"
        assert "if" not in wf["jobs"]["testpypi"]
        assert "if" not in wf["jobs"]["pypi"]

    def test_publish_yml_builds_with_the_commands_it_used_before(self):
        runs = [str(s.get("run") or "").strip() for s in _workflow("publish.yml")["jobs"]["build"]["steps"]]
        assert runs.index("pip install build twine") < runs.index("python -m build") < runs.index("twine check dist/*")

    @pytest.mark.parametrize("workflow", ["publish-cascor-model.yml", "publish-protocol.yml"])
    def test_the_sub_package_publishers_keep_their_triggers_permissions_concurrency_and_guard(self, workflow):
        package = PUBLISHERS[workflow][0]
        wf = _workflow(workflow)
        assert set(wf["on"]) == {"release", "workflow_dispatch"}
        assert wf["on"]["release"] == {"types": ["published"]}
        assert wf["permissions"] == {"id-token": "write", "contents": "read"}
        assert wf["concurrency"]["cancel-in-progress"] is False
        assert wf["jobs"]["build"]["if"] == f"github.event_name != 'release' || startsWith(github.event.release.tag_name, '{package}-v')"


class TestTemplate:
    def test_every_copy_of_a_template_step_is_byte_identical(self):
        copies: dict[str, list[str]] = {RECORD: [], VERIFY_DIST: [], "index": []}
        for workflow in PUBLISHERS:
            for job in _workflow(workflow)["jobs"].values():
                for step in job.get("steps") or []:
                    kind = "index" if step.get("name") in INDEX_CHECKS else step.get("name")
                    if kind in copies:
                        assert step.get("shell") == "python3 {0}"
                        copies[kind].append(step["run"])
        # One record per workflow, a dist check in each publish job, three JSON checks per workflow.
        assert {kind: len(runs) for kind, runs in copies.items()} == {RECORD: 3, VERIFY_DIST: 6, "index": 9}
        assert {kind: len(set(runs)) for kind, runs in copies.items()} == {RECORD: 1, VERIFY_DIST: 1, "index": 1}

    def test_the_template_reads_nothing_but_its_parameters(self):
        for kind, run in _template_scripts().items():
            assert "${{" not in run, f"{kind}: an expression would tie the step to one workflow"
            read = set(re.findall(r"os\.environ(?:\.get\(|\[)\"([A-Z_]+)\"", run))
            assert read, kind
            assert read <= TEMPLATE_ENV, f"{kind} reads {sorted(read - TEMPLATE_ENV)}"

    def test_the_json_check_polls_a_bounded_number_of_times(self):
        run = _template_scripts()["index"]
        attempts = re.search(r'"HASH_CHECK_ATTEMPTS", "(\d+)"', run)
        interval = re.search(r'"HASH_CHECK_INTERVAL", "(\d+(?:\.\d+)?)"', run)
        assert attempts and interval
        # Long enough for a CDN to catch up after an upload, and bounded.
        assert 60 <= int(attempts.group(1)) * float(interval.group(1)) <= 900
        for workflow in PUBLISHERS:
            for job in _workflow(workflow)["jobs"].values():
                for step in job.get("steps") or []:
                    assert not {"HASH_CHECK_ATTEMPTS", "HASH_CHECK_INTERVAL"} & set(step.get("env") or {}), f"{workflow}: {step.get('name')} overrides the bounds"


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Rehearsals: the template steps' own Python, run as the runner runs it
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestRecordSha256sums:
    def test_it_hashes_every_distribution_in_sha256sum_format(self, workspace, tmp_path):
        proc = _run(_script(RECORD, tmp_path), workspace)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (workspace / "SHA256SUMS").read_text(encoding="utf-8") == "".join(f"{_sha(data)}  {name}\n" for name, data in sorted(BUILT.items()))

    def test_sha256sum_checks_what_it_writes(self, workspace, tmp_path):
        binary = shutil.which("sha256sum")
        if binary is None:
            pytest.skip("no sha256sum on PATH")
        assert _run(_script(RECORD, tmp_path), workspace).returncode == 0
        check = subprocess.run([binary, "--check", "--strict", str(workspace / "SHA256SUMS")], cwd=workspace / "dist", capture_output=True, text=True, timeout=60, check=False)  # nosec B603 - fixed argv
        assert check.returncode == 0, check.stdout + check.stderr

    @pytest.mark.parametrize("stray", ["notes.txt", "juniper_cascor-9.9.9.zip"])
    def test_it_refuses_anything_but_wheels_and_sdists(self, workspace, tmp_path, stray):
        (workspace / "dist" / stray).write_bytes(b"x")
        proc = _run(_script(RECORD, tmp_path), workspace)
        assert proc.returncode == 1
        assert "::error::" in proc.stdout
        assert not (workspace / "SHA256SUMS").exists()

    def test_it_refuses_an_empty_or_a_missing_dist_directory(self, tmp_path):
        root = tmp_path / "empty"
        (root / "dist").mkdir(parents=True)
        script = _script(RECORD, tmp_path)
        assert _run(script, root).returncode == 1
        assert _run(script, root, PUBLISH_DIST_DIR="nowhere").returncode == 1

    def test_it_leaves_out_dotfiles_as_the_upload_does(self, workspace, tmp_path):
        (workspace / "dist" / ".gitignore").write_text("*\n", encoding="utf-8")
        assert _run(_script(RECORD, tmp_path), workspace).returncode == 0
        assert ".gitignore" not in (workspace / "SHA256SUMS").read_text(encoding="utf-8")

    def test_a_nested_dist_directory_resolves_from_the_workspace_not_the_working_directory(self, tmp_path):
        root = tmp_path / "workspace"
        package_dir = root / "juniper-cascor-model"
        (package_dir / "dist").mkdir(parents=True)
        (package_dir / "dist" / "juniper_cascor_model-1.0.0-py3-none-any.whl").write_bytes(b"model wheel")
        # The sub-package build jobs run every step in the package directory (defaults.run.working-directory).
        proc = _run(_script(RECORD, tmp_path), root, cwd=package_dir, PUBLISH_PACKAGE="juniper-cascor-model", PUBLISH_DIST_DIR="juniper-cascor-model/dist")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (root / "SHA256SUMS").read_text(encoding="utf-8") == f"{_sha(b'model wheel')}  juniper_cascor_model-1.0.0-py3-none-any.whl\n"


class TestVerifyDist:
    @staticmethod
    def _recorded(workspace: Path, tmp_path: Path) -> Path:
        assert _run(_script(RECORD, tmp_path), workspace).returncode == 0
        return _script(VERIFY_DIST, tmp_path)

    def test_it_passes_on_the_files_the_build_recorded(self, workspace, tmp_path):
        proc = _run(self._recorded(workspace, tmp_path), workspace)
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_it_fails_when_a_file_changed(self, workspace, tmp_path):
        script = self._recorded(workspace, tmp_path)
        (workspace / "dist" / WHEEL).write_bytes(b"another build")
        assert _run(script, workspace).returncode == 1

    def test_it_fails_on_a_file_the_build_did_not_record(self, workspace, tmp_path):
        script = self._recorded(workspace, tmp_path)
        (workspace / "dist" / "juniper_cascor-9.9.9-py2.py3-none-any.whl").write_bytes(b"x")
        assert _run(script, workspace).returncode == 1

    def test_it_fails_when_a_recorded_file_is_missing(self, workspace, tmp_path):
        script = self._recorded(workspace, tmp_path)
        (workspace / "dist" / SDIST).unlink()
        assert _run(script, workspace).returncode == 1

    @pytest.mark.parametrize(
        "sums",
        ["", f"nothex  {WHEEL}\n", f"{'a' * 64} {WHEEL}\n", f"{'a' * 64}  {WHEEL}\n{'a' * 64}  {WHEEL}\n"],
        ids=["empty", "not-a-sha256", "one-space", "listed-twice"],
    )
    def test_it_fails_on_a_sha256sums_it_cannot_read(self, workspace, tmp_path, sums):
        (workspace / "SHA256SUMS").write_text(sums, encoding="utf-8")
        assert _run(_script(VERIFY_DIST, tmp_path), workspace).returncode == 1


class _Index(http.server.ThreadingHTTPServer):
    """A stand-in for an index's JSON API: each GET gets the next scripted answer, and the last one repeats."""

    def __init__(self, answers: list[tuple[int, object]]) -> None:
        super().__init__(("127.0.0.1", 0), _IndexHandler)
        self.answers = list(answers)
        self.paths: list[str] = []


class _IndexHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        index = self.server
        index.paths.append(self.path)
        status, body = index.answers.pop(0) if len(index.answers) > 1 else index.answers[0]
        payload = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        """Quiet: the cases read the paths the check asked for, not the server's log."""
        return


@pytest.fixture
def index_server():
    started: list[_Index] = []

    def start(*answers: tuple[int, object]) -> _Index:
        server = _Index(list(answers))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return server

    yield start
    for server in started:
        server.shutdown()
        server.server_close()


def _release(files: dict[str, str]) -> dict:
    return {"info": {"name": "juniper-cascor", "version": "9.9.9"}, "urls": [{"filename": name, "digests": {"sha256": digest, "md5": "0" * 32}} for name, digest in files.items()]}


NOT_FOUND = (404, {"message": "Not Found"})
SERVED = {name: _sha(data) for name, data in BUILT.items()}


class TestIndexCheck:
    @staticmethod
    def _ready(workspace: Path, tmp_path: Path) -> Path:
        assert _run(_script(RECORD, tmp_path), workspace).returncode == 0
        return _script("index", tmp_path)

    @staticmethod
    def _check(script: Path, workspace: Path, server: _Index, attempts: int = 4) -> subprocess.CompletedProcess[str]:
        return _run(script, workspace, INDEX_URL=f"http://127.0.0.1:{server.server_address[1]}", HASH_CHECK_ATTEMPTS=str(attempts), HASH_CHECK_INTERVAL="0")

    def test_an_exact_match_passes_at_the_first_answer(self, workspace, tmp_path, index_server):
        server = index_server((200, _release(SERVED)))
        proc = self._check(self._ready(workspace, tmp_path), workspace, server)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert server.paths == ["/pypi/juniper-cascor/9.9.9/json"]

    def test_it_polls_through_a_version_the_index_does_not_serve_yet(self, workspace, tmp_path, index_server):
        server = index_server(NOT_FOUND, NOT_FOUND, (200, _release(SERVED)))
        proc = self._check(self._ready(workspace, tmp_path), workspace, server)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert len(server.paths) == 3

    def test_it_polls_through_a_file_the_index_does_not_list_yet(self, workspace, tmp_path, index_server):
        server = index_server((200, _release({WHEEL: SERVED[WHEEL]})), (200, _release(SERVED)))
        proc = self._check(self._ready(workspace, tmp_path), workspace, server)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert len(server.paths) == 2

    def test_it_polls_through_an_answer_it_cannot_read(self, workspace, tmp_path, index_server):
        server = index_server((200, b"<html>busy</html>"), (503, b""), (200, {"info": {}}), (200, _release(SERVED)))
        proc = self._check(self._ready(workspace, tmp_path), workspace, server, attempts=5)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert len(server.paths) == 4

    def test_a_file_with_another_sha256_fails_at_once(self, workspace, tmp_path, index_server):
        server = index_server((200, _release({**SERVED, WHEEL: _sha(b"another build")})))
        proc = self._check(self._ready(workspace, tmp_path), workspace, server)
        assert proc.returncode == 1
        assert "::error::" in proc.stdout
        assert len(server.paths) == 1

    def test_a_file_this_run_did_not_build_fails_at_once(self, workspace, tmp_path, index_server):
        server = index_server((200, _release({**SERVED, "juniper_cascor-9.9.9-py2.py3-none-any.whl": "b" * 64})))
        proc = self._check(self._ready(workspace, tmp_path), workspace, server)
        assert proc.returncode == 1
        assert len(server.paths) == 1

    def test_it_gives_up_after_its_bounded_attempts(self, workspace, tmp_path, index_server):
        server = index_server(NOT_FOUND)
        proc = self._check(self._ready(workspace, tmp_path), workspace, server, attempts=3)
        assert proc.returncode == 1
        assert "after 3 attempts" in proc.stdout
        assert len(server.paths) == 3

    @pytest.mark.parametrize("extra", ["juniper_cascor_model-9.9.9-py3-none-any.whl", "juniper_cascor-9.9.8.tar.gz"], ids=["another-package", "another-version"])
    def test_it_refuses_a_sha256sums_it_cannot_vouch_for_before_asking_the_index(self, workspace, tmp_path, index_server, extra):
        script = self._ready(workspace, tmp_path)
        with (workspace / "SHA256SUMS").open("a", encoding="utf-8") as sums:
            sums.write(f"{'c' * 64}  {extra}\n")
        server = index_server((200, _release(SERVED)))
        proc = self._check(script, workspace, server)
        assert proc.returncode == 1
        assert "::error::" in proc.stdout
        assert server.paths == []
