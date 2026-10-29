"""
Pin ``util/check_junit_executed.py`` and the scheduled lanes that rely on it.

The nightly ``Long-Running Correctness Tests`` job ran zero tests and passed (job 121983548401:
``1 skipped, 6214 deselected``): it lifted only the ``long`` skip, and its one pick also carries
the ``slow`` and ``integration`` skips. pytest exits 0 when every pick is skipped, so the exit
status could not see it. The script reads the lane's JUnit report instead, and fails when nothing
executed or anything failed.

The synthetic reports drive each verdict. Two real pytest runs pin the encoding the script relies
on: pytest writes an XFAIL as ``<skipped type="pytest.xfail">``, so a naive "tests minus skipped"
would call a lane whose one test XFAILs empty, and an all-skipped lane exits 0. The workflow tests
pin the wiring in ``scheduled-tests.yml``: the long lane lifts all three skips, the untimed
performance smoke runs the whole of ``src/tests/performance``, each is followed by the check, and
the smoke can fail the nightly.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess  # nosec B404 - runs a child pytest with a fixed argv, never a shell
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "util" / "check_junit_executed.py"
SCRIPT_REL = "util/check_junit_executed.py"
WORKFLOW = REPO / ".github" / "workflows" / "scheduled-tests.yml"
GUARDED_JOBS = ("long-running-tests", "performance-smoke")


def _load():
    spec = importlib.util.spec_from_file_location("check_junit_executed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # ``@dataclass`` resolves its module through sys.modules, so register before executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def guard():
    return _load()


PASSING = '<testcase classname="m.C" name="test_pass" time="0.1" />'
SKIP_SLOW = '<testcase classname="m.C" name="test_long" time="0.0"><skipped type="pytest.skip" message="need --slow option to run">m.py:41: need --slow option to run</skipped></testcase>'
SKIP_PERF = '<testcase classname="m.C" name="test_bench" time="0.0"><skipped type="pytest.skip" message="need --run-performance flag">m.py:9: need --run-performance flag</skipped></testcase>'
XFAIL = '<testcase classname="m.C" name="test_resume" time="1.0"><skipped type="pytest.xfail" message="not yet deterministic" /></testcase>'
FAIL = '<testcase classname="m.C" name="test_fail" time="0.1"><failure message="assert False">tb</failure></testcase>'
ERROR = '<testcase classname="m.C" name="test_error" time="0.0"><error message="fixture boom">tb</error></testcase>'
FAIL_AND_TEARDOWN_ERROR = '<testcase classname="m.C" name="test_both" time="0.1"><failure message="assert False">tb</failure><error message="teardown boom">tb</error></testcase>'


def _suite(*testcases: str) -> str:
    return f'<testsuite name="pytest">{"".join(testcases)}</testsuite>'


def _report(tmp_path: Path, *suites: str) -> Path:
    path = tmp_path / "junit.xml"
    path.write_text(f'<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">{"".join(suites)}</testsuites>', encoding="utf-8")
    return path


def _run(guard, capsys, *argv: str) -> tuple[int, str]:
    rc = guard.main([*argv])
    return rc, capsys.readouterr().out


class TestVerdict:
    def test_the_zero_test_lane_fails_and_names_the_skip_reason(self, guard, tmp_path, capsys):
        """The shape of job 121983548401's report: one pick, skipped for want of --slow."""
        rc, out = _run(guard, capsys, str(_report(tmp_path, _suite(SKIP_SLOW))), "--lane", "Long-Running Correctness Tests")
        assert rc == 1
        assert "executed 0 " in out
        assert "need --slow option to run" in out
        assert "::error title=Long-Running Correctness Tests::the lane executed 0 test(s), fewer than the 1 required" in out

    def test_an_xfail_counts_as_executed(self, guard, tmp_path, capsys):
        rc, out = _run(guard, capsys, str(_report(tmp_path, _suite(XFAIL))))
        assert rc == 0, out
        assert "executed 1 (passed 0, xfailed 1, failed 0), errors 0, skipped 0" in out
        assert "::error" not in out

    def test_a_passing_lane_passes(self, guard, tmp_path, capsys):
        rc, out = _run(guard, capsys, str(_report(tmp_path, _suite(PASSING, PASSING, SKIP_PERF))))
        assert rc == 0, out
        assert "3 testcase(s)" in out
        assert "executed 2 " in out

    def test_a_failure_fails_the_lane_even_though_tests_executed(self, guard, tmp_path, capsys):
        rc, out = _run(guard, capsys, str(_report(tmp_path, _suite(PASSING, FAIL))))
        assert rc == 1
        assert "1 test(s) failed and 0 errored" in out

    def test_an_error_fails_the_lane(self, guard, tmp_path, capsys):
        rc, out = _run(guard, capsys, str(_report(tmp_path, _suite(PASSING, ERROR))))
        assert rc == 1
        assert "0 test(s) failed and 1 errored" in out

    def test_skips_do_not_count_toward_the_minimum(self, guard, tmp_path, capsys):
        report = str(_report(tmp_path, _suite(PASSING, SKIP_PERF, SKIP_PERF, SKIP_PERF)))
        assert _run(guard, capsys, report, "--min-executed", "1")[0] == 0
        rc, out = _run(guard, capsys, report, "--min-executed", "2")
        assert rc == 1
        assert "the lane executed 1 test(s), fewer than the 2 required: of 4 reported, 3 were skipped" in out
        assert "    3  need --run-performance flag" in out

    def test_counts_span_every_testsuite(self, guard, tmp_path):
        root = guard.ET.parse(_report(tmp_path, _suite(PASSING, SKIP_SLOW), _suite(XFAIL, FAIL))).getroot()
        counts = guard.count(root)
        assert (counts.passed, counts.xfailed, counts.failed, counts.errors, counts.skipped) == (1, 1, 1, 0, 1)
        assert counts.total == 4
        assert counts.executed == 3

    def test_a_failure_with_a_teardown_error_counts_once_as_failed(self, guard, tmp_path):
        counts = guard.count(guard.ET.parse(_report(tmp_path, _suite(FAIL_AND_TEARDOWN_ERROR))).getroot())
        assert (counts.total, counts.failed, counts.errors) == (1, 1, 0)

    def test_a_report_with_no_testcase_fails(self, guard, tmp_path, capsys):
        rc, out = _run(guard, capsys, str(_report(tmp_path, _suite())))
        assert rc == 1
        assert "0 testcase(s)" in out

    def test_a_bare_testsuite_root_is_read(self, guard, tmp_path, capsys):
        path = tmp_path / "bare.xml"
        path.write_text(_suite(PASSING), encoding="utf-8")
        assert _run(guard, capsys, str(path))[0] == 0


class TestUnreadableReport:
    def test_a_missing_report_is_exit_2(self, guard, tmp_path, capsys):
        rc, out = _run(guard, capsys, str(tmp_path / "absent.xml"), "--lane", "Performance Smoke")
        assert rc == 2
        assert out.startswith("::error title=Performance Smoke::no readable JUnit report at ")

    def test_malformed_xml_is_exit_2(self, guard, tmp_path, capsys):
        path = tmp_path / "torn.xml"
        path.write_text('<testsuites><testsuite name="pytest"><testcase', encoding="utf-8")
        assert _run(guard, capsys, str(path))[0] == 2

    def test_a_non_junit_document_is_exit_2(self, guard, tmp_path, capsys):
        path = tmp_path / "page.xml"
        path.write_text("<html><body>no tests here</body></html>", encoding="utf-8")
        rc, out = _run(guard, capsys, str(path))
        assert rc == 2
        assert "its root element is <html>" in out

    def test_a_minimum_below_one_is_a_usage_error(self, guard, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            guard.main([str(_report(tmp_path, _suite(PASSING))), "--min-executed", "0"])
        assert excinfo.value.code == 2


class TestAnnotation:
    def test_title_and_message_are_escaped(self, guard):
        line = guard.annotate("lane: a, b", "50% done\nnext")
        assert line == "::error title=lane%3A a%2C b::50%25 done%0Anext"


def _child_pytest(tmp_path: Path, body: str) -> tuple[int, Path]:
    """Run a real, isolated pytest on ``body``: no repo config, no conftest, no autoloaded plugins."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_child.py").write_text(body, encoding="utf-8")
    report = tmp_path / "child.xml"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    argv = [sys.executable, "-m", "pytest", "-c", str(tmp_path / "pytest.ini"), "--rootdir", str(tmp_path), "--confcutdir", str(tmp_path), "-p", "no:cacheprovider", "-q", f"--junitxml={report}", str(tmp_path / "test_child.py")]
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=50)  # nosec B603 - fixed argv, no shell
    assert report.is_file(), result.stdout + result.stderr
    return result.returncode, report


CHILD_FOUR_OUTCOMES = textwrap.dedent(
    """\
    import pytest


    def test_pass():
        pass


    @pytest.mark.skip(reason="need --slow option to run")
    def test_skip():
        pass


    @pytest.mark.xfail(reason="known gap")
    def test_xfail():
        assert False


    @pytest.mark.xfail(reason="known gap")
    def test_xpass():
        pass
    """
)

CHILD_ALL_SKIPPED = textwrap.dedent(
    """\
    import pytest


    @pytest.mark.skip(reason="need --slow option to run")
    def test_long():
        pass
    """
)


class TestRealPytestReport:
    def test_pass_skip_xfail_and_xpass_are_counted_as_pytest_writes_them(self, guard, tmp_path, capsys):
        rc, report = _child_pytest(tmp_path, CHILD_FOUR_OUTCOMES)
        assert rc == 0
        counts = guard.count(guard.ET.parse(report).getroot())
        assert (counts.passed, counts.xfailed, counts.failed, counts.errors, counts.skipped) == (2, 1, 0, 0, 1)
        assert _run(guard, capsys, str(report))[0] == 0

    def test_an_all_skipped_lane_exits_0_from_pytest_and_1_from_the_check(self, guard, tmp_path, capsys):
        rc, report = _child_pytest(tmp_path, CHILD_ALL_SKIPPED)
        assert rc == 0, "pytest's own exit status cannot see a lane that skipped everything"
        assert _run(guard, capsys, str(report))[0] == 1


def _workflow() -> dict:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
    data["on"] = data.pop(True, data.get("on"))
    return data


def _pytest_step(job: dict) -> tuple[int, list[str]]:
    """The one step of ``job`` that runs pytest, and the arguments after ``python -m pytest`` (continuations joined)."""
    found = []
    for index, step in enumerate(job["steps"]):
        lines = str(step.get("run", "")).splitlines()
        for start, line in enumerate(lines):
            if "python -m pytest" not in line:
                continue
            parts = [line]
            cursor = start
            while parts[-1].rstrip().endswith("\\") and cursor + 1 < len(lines):
                cursor += 1
                parts.append(lines[cursor])
            tokens = shlex.split(" ".join(part.rstrip().rstrip("\\") for part in parts))
            found.append((index, tokens[tokens.index("pytest") + 1 :]))
    assert len(found) == 1, f"expected exactly one pytest invocation, found {len(found)}"
    return found[0]


def _junit_path(argv: list[str]) -> str:
    paths = [arg.split("=", 1)[1] for arg in argv if arg.startswith("--junitxml=")]
    assert len(paths) == 1, argv
    return paths[0]


class TestScheduledLaneWiring:
    def test_the_long_lane_lifts_every_skip_its_pick_carries(self):
        """``test_deterministic_training_resume`` carries three skips from ``src/tests/conftest.py``; each flag lifts one.

        ``long`` and ``slow`` are its markers. ``integration`` is not: the conftest tests ``"integration" in
        item.keywords``, and the keywords hold every parent node's name, ``src/tests/integration`` included.
        """
        _, argv = _pytest_step(_workflow()["jobs"]["long-running-tests"])
        assert argv[argv.index("-m") + 1] == "long"
        assert "--run-long" in argv
        assert "--slow" in argv
        assert "--integration" in argv

    @pytest.mark.parametrize("job_id", GUARDED_JOBS)
    def test_the_check_reads_the_lane_report_right_after_pytest(self, job_id):
        job = _workflow()["jobs"][job_id]
        index, argv = _pytest_step(job)
        check = job["steps"][index + 1]
        tokens = shlex.split(str(check.get("run", "")))
        assert tokens[:3] == ["python", SCRIPT_REL, _junit_path(argv)]
        assert "if" not in check, "the check must run whenever pytest passed"
        assert not check.get("continue-on-error"), "a zero-test lane must fail the job"

    def test_the_performance_smoke_runs_the_whole_suite_untimed(self):
        job = _workflow()["jobs"]["performance-smoke"]
        assert "if" not in job, "the smoke runs on every schedule and dispatch"
        _, argv = _pytest_step(job)
        assert "src/tests/performance" in argv
        assert "--run-performance" in argv
        assert "--benchmark-disable" in argv
        assert "-m" not in argv, "a marker filter would drop part of the suite"
        assert not [arg for arg in argv if arg.startswith("--benchmark-compare")], "timing is report-only: no compare, no compare-fail"

    def test_a_failed_performance_smoke_fails_the_nightly(self):
        summary = _workflow()["jobs"]["summary"]
        assert "performance-smoke" in summary["needs"]
        run = "\n".join(str(step.get("run", "")) for step in summary["steps"])
        gate = run.index('needs.performance-smoke.result }}" == "failure"')
        assert gate < run.index("exit 1")
