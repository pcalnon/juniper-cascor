"""Pin the open-PR budget alarm (``.github/workflows/pr-budget-alarm.yml``, #654).

The workflow is report-only on purpose: a breach is a green run, a ``gh`` failure is a
green run, and Slack fires only after a breach. Nothing else in the suite executes
that shell, so a change that treated a failed query as an empty queue (level OK,
zero PRs) or that posted the webhook URL inside the notice would ship green.

These tests run the workflow's own ``run:`` scripts under fake ``gh`` / ``curl``.
They do not call GitHub or Slack.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "pr-budget-alarm.yml"

_WEBHOOK = "https://hooks.example.test/services/T00/B00/SECRETTOKEN"
_RUN_URL = "https://github.com/pcalnon/juniper-cascor/actions/runs/99"


def _workflow() -> dict:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
    data["on"] = data.pop(True, data.get("on"))
    return data


def _job() -> dict:
    return _workflow()["jobs"]["budget-alarm"]


def _step(name_prefix: str) -> dict:
    matches = [s for s in _job()["steps"] if str(s.get("name", "")).startswith(name_prefix)]
    assert len(matches) == 1, f"expected one step named {name_prefix!r}, found {len(matches)}"
    return matches[0]


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _outputs(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[key] = value
    return parsed


def _run_count(tmp_path: Path, *, stdout: str, stderr: str = "", rc: int = 0, warn: str | None = None, alarm: str | None = None, unset_thresholds: bool = False) -> subprocess.CompletedProcess[str]:
    """Execute the count step. ``warn``/``alarm`` of None leave the variable unset."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "gh.stdout").write_text(stdout, encoding="utf-8")
    (tmp_path / "gh.stderr").write_text(stderr, encoding="utf-8")
    _write_executable(
        bin_dir / "gh",
        f"""#!/bin/bash
printf '%s\\n' "$@" > "{tmp_path}/gh.argv"
cat "{tmp_path}/gh.stderr" >&2
cat "{tmp_path}/gh.stdout"
exit {rc}
""",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["GH_TOKEN"] = "test-token-not-a-secret"  # nosec B105 - dummy token for the PATH-stubbed gh, never a real credential
    env["GH_REPO"] = "pcalnon/juniper-cascor"
    env["GITHUB_OUTPUT"] = str(tmp_path / "output.txt")
    env["GITHUB_STEP_SUMMARY"] = str(tmp_path / "summary.md")
    for name, value in (("PR_BUDGET_WARN", warn), ("PR_BUDGET_ALARM", alarm)):
        if unset_thresholds or value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return subprocess.run(  # nosec B603 B607 - fixed argv, the script under test is this repo's workflow
        ["bash", "-c", _step("Count open PRs")["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _prs(*names: str) -> str:
    return json.dumps([{"number": i + 1, "headRefName": name} for i, name in enumerate(names)])


def _run_slack(tmp_path: Path, *, level: str, total: str, cursor: str, warn: str, alarm: str, webhook: str, curl_rc: int = 0) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "curl",
        f"""#!/bin/bash
printf '%s\\0' "$@" > "{tmp_path}/curl.argv"
exit {curl_rc}
""",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["SLACK_WEBHOOK_URL"] = webhook
    env["RUN_URL"] = _RUN_URL
    env["LEVEL"] = level
    env["TOTAL"] = total
    env["CURSOR"] = cursor
    env["WARN"] = warn
    env["ALARM"] = alarm
    return subprocess.run(  # nosec B603 B607 - fixed argv, the script under test is this repo's workflow
        ["bash", "-c", _step("Slack notification")["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _curl_args(tmp_path: Path) -> list[str]:
    raw = (tmp_path / "curl.argv").read_bytes()
    return [part.decode() for part in raw.split(b"\0") if part]


class TestCountStep:
    @pytest.mark.parametrize("mode", ["unset", "empty"])
    def test_missing_thresholds_default_to_15_and_30(self, tmp_path: Path, mode: str) -> None:
        """GitHub sets an unset repo variable to an empty string, which ``:-`` must still default."""
        warn = "" if mode == "empty" else None
        alarm = "" if mode == "empty" else None
        result = _run_count(tmp_path, stdout="[]", warn=warn, alarm=alarm, unset_thresholds=(mode == "unset"))
        assert result.returncode == 0, result.stderr
        outputs = _outputs(tmp_path / "output.txt")
        assert outputs["warn"] == "15"
        assert outputs["alarm"] == "30"
        assert outputs["level"] == "OK"
        assert outputs["total"] == "0"
        assert outputs["cursor"] == "0"

    @pytest.mark.parametrize(
        ("count", "level"),
        [(0, "OK"), (14, "OK"), (15, "WARN"), (29, "WARN"), (30, "ALARM"), (31, "ALARM")],
    )
    def test_default_thresholds_are_greater_or_equal(self, tmp_path: Path, count: int, level: str) -> None:
        """14 is still OK; 15 warns; 30 alarms. A switch to ``-gt`` would move every boundary."""
        names = tuple(f"feature/{i}" for i in range(count))
        result = _run_count(tmp_path, stdout=_prs(*names))
        assert result.returncode == 0, result.stderr
        outputs = _outputs(tmp_path / "output.txt")
        assert outputs["level"] == level
        assert outputs["total"] == str(count)
        assert outputs["cursor"] == "0"
        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert f"| Open PRs (total) | {count} |" in summary
        assert f"| Status | **{level}** |" in summary
        assert result.stdout.strip().splitlines()[-1] == f"PR budget: total={count} cursor=0 warn=15 alarm=30 level={level}"

    def test_a_breach_stays_green(self, tmp_path: Path) -> None:
        """The alarm is not a merge gate. Thirty open PRs must not fail the run."""
        result = _run_count(tmp_path, stdout=_prs(*(f"feature/{i}" for i in range(30))))
        assert result.returncode == 0, result.stderr
        assert _outputs(tmp_path / "output.txt")["level"] == "ALARM"
        assert "this run stays green" in (tmp_path / "summary.md").read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        ("count", "level"),
        [(1, "OK"), (2, "WARN"), (3, "WARN"), (4, "ALARM")],
    )
    def test_custom_thresholds_are_greater_or_equal(self, tmp_path: Path, count: int, level: str) -> None:
        names = tuple(f"feature/{i}" for i in range(count))
        result = _run_count(tmp_path, stdout=_prs(*names), warn="2", alarm="4")
        assert result.returncode == 0, result.stderr
        outputs = _outputs(tmp_path / "output.txt")
        assert outputs["level"] == level
        assert outputs["warn"] == "2"
        assert outputs["alarm"] == "4"

    def test_cursor_prefix_is_counted_and_only_that_prefix(self, tmp_path: Path) -> None:
        """``cursor/`` counts. ``cursor``, ``Cursor/``, and ``cursor-foo`` do not.

        The cursor count cannot by itself outrank the total (it is a subset, and both
        use the same thresholds), so the regression is a wrong count in the report,
        not a different level.
        """
        names = (
            "cursor/missing-test-coverage-a09b",
            "cursor/",
            "cursor",
            "Cursor/case",
            "cursor-bot/x",
            "feature/not-cursor/child",
            "my-cursor/x",
        )
        result = _run_count(tmp_path, stdout=_prs(*names))
        assert result.returncode == 0, result.stderr
        outputs = _outputs(tmp_path / "output.txt")
        assert outputs["total"] == "7"
        assert outputs["cursor"] == "2"
        assert outputs["level"] == "OK"
        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "| Open `cursor/` PRs | 2 |" in summary

    def test_gh_is_asked_only_for_open_prs_capped_at_500(self, tmp_path: Path) -> None:
        result = _run_count(tmp_path, stdout="[]")
        assert result.returncode == 0, result.stderr
        argv = (tmp_path / "gh.argv").read_text(encoding="utf-8").splitlines()
        assert argv[:2] == ["pr", "list"]
        assert "--repo" in argv and argv[argv.index("--repo") + 1] == "pcalnon/juniper-cascor"
        assert "--state" in argv and argv[argv.index("--state") + 1] == "open"
        assert "--limit" in argv and argv[argv.index("--limit") + 1] == "500"
        assert "--json" in argv and argv[argv.index("--json") + 1] == "number,headRefName"

    def test_gh_failure_stays_green_and_is_not_an_empty_queue(self, tmp_path: Path) -> None:
        """A failed query must not look like zero open PRs. That reading would silence the alarm."""
        result = _run_count(tmp_path, stdout="", stderr="api down\ntry later\n", rc=1)
        assert result.returncode == 0, result.stderr
        outputs = _outputs(tmp_path / "output.txt")
        assert outputs == {"level": "OK"}
        assert "total" not in outputs
        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "Could not query open PRs" in summary
        assert "| Open PRs (total) |" not in summary
        assert "**OK**" not in summary
        assert "::warning title=pr-budget-alarm::Could not list open PRs: api down try later" in result.stdout

    @pytest.mark.parametrize("body", ["null", '{"message":"bad"}', "not-json", "[{"])
    def test_unusable_body_is_not_reported_as_an_empty_ok_queue(self, tmp_path: Path, body: str) -> None:
        """``gh`` exiting 0 with a body jq cannot count must not become total=0 level=OK."""
        result = _run_count(tmp_path, stdout=body, rc=0)
        assert result.returncode != 0, result.stdout
        summary = (tmp_path / "summary.md").read_text(encoding="utf-8") if (tmp_path / "summary.md").exists() else ""
        assert "| Open PRs (total) | 0 |" not in summary
        # The successful OK report always records a total. An unusable body must not.
        assert "total" not in _outputs(tmp_path / "output.txt")

    def test_a_pr_with_no_branch_name_is_not_an_empty_ok_queue(self, tmp_path: Path) -> None:
        """``startswith`` on a missing headRefName errors. Swallowing that as zero PRs hides the queue."""
        result = _run_count(tmp_path, stdout=json.dumps([{"number": 1}]))
        assert result.returncode != 0, result.stdout
        assert "total" not in _outputs(tmp_path / "output.txt")

    def test_a_non_numeric_threshold_does_not_report_ok(self, tmp_path: Path) -> None:
        """A garbage variable must not be read as 'within budget' for a queue that is over both defaults."""
        names = tuple(f"feature/{i}" for i in range(40))
        result = _run_count(tmp_path, stdout=_prs(*names), warn="abc", alarm="30")
        outputs = _outputs(tmp_path / "output.txt")
        assert not (result.returncode == 0 and outputs.get("level") == "OK")

    def test_warn_of_zero_warns_on_an_empty_queue(self, tmp_path: Path) -> None:
        """``-ge`` means a threshold of 0 matches a count of 0. ``-gt`` would call that OK."""
        result = _run_count(tmp_path, stdout="[]", warn="0", alarm="30")
        assert result.returncode == 0, result.stderr
        assert _outputs(tmp_path / "output.txt")["level"] == "WARN"

    def test_alarm_wins_when_the_thresholds_are_equal(self, tmp_path: Path) -> None:
        result = _run_count(tmp_path, stdout=_prs("only"), warn="1", alarm="1")
        assert result.returncode == 0, result.stderr
        assert _outputs(tmp_path / "output.txt")["level"] == "ALARM"


class TestSlackStep:
    def test_a_missing_webhook_exits_zero_and_does_not_post(self, tmp_path: Path) -> None:
        result = _run_slack(tmp_path, level="ALARM", total="40", cursor="12", warn="15", alarm="30", webhook="")
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / "curl.argv").exists()
        assert "::warning title=PR budget ALARM with no Slack webhook::40 open PR(s), 12 on cursor/ branches (warn=15 alarm=30)." in result.stdout
        assert "SLACK_WEBHOOK_URL is not set" in result.stdout
        assert _WEBHOOK not in result.stdout

    def test_the_payload_is_the_notice_and_does_not_contain_the_webhook(self, tmp_path: Path) -> None:
        result = _run_slack(tmp_path, level="WARN", total="16", cursor="3", warn="15", alarm="30", webhook=_WEBHOOK)
        assert result.returncode == 0, result.stderr
        args = _curl_args(tmp_path)
        assert args[0:3] == ["-fsS", "-X", "POST"]
        assert "Content-Type: application/json" in args
        payload = json.loads(args[args.index("-d") + 1])
        assert list(payload) == ["text"]
        assert payload["text"] == f"PR budget WARN: 16 open PR(s), 3 on cursor/ branches (thresholds warn=15 / alarm=30). Run: {_RUN_URL}"
        assert "SECRETTOKEN" not in payload["text"]
        assert args[-1] == _WEBHOOK
        assert "SECRETTOKEN" not in result.stdout
        assert result.stdout.strip() == "Slack notification posted."

    def test_a_failed_post_fails_the_script_and_the_step_does_not_fail_the_job(self, tmp_path: Path) -> None:
        """The script itself fails closed on curl. The workflow's continue-on-error is what keeps the job green."""
        result = _run_slack(tmp_path, level="ALARM", total="40", cursor="1", warn="15", alarm="30", webhook=_WEBHOOK, curl_rc=22)
        assert result.returncode != 0
        step = _step("Slack notification")
        assert step["continue-on-error"] is True
        assert step["if"] == "steps.count.outputs.level != 'OK'"


class TestWorkflowShape:
    def test_it_is_schedule_and_dispatch_only(self) -> None:
        triggers = _workflow()["on"]
        assert set(triggers) == {"schedule", "workflow_dispatch"}
        assert triggers["schedule"] == [{"cron": "0 14 * * *"}]
        assert "pull_request" not in triggers

    def test_permissions_are_read_only(self) -> None:
        permissions = _workflow()["permissions"]
        assert permissions == {"contents": "read", "pull-requests": "read"}

    def test_a_second_report_cancels_the_first(self) -> None:
        concurrency = _workflow()["concurrency"]
        assert concurrency["group"] == "pr-budget-alarm"
        assert concurrency["cancel-in-progress"] is True

    def test_the_count_step_does_not_receive_the_webhook(self) -> None:
        """The secret is in scope only for the step that posts. A count failure cannot see it."""
        env = _step("Count open PRs")["env"]
        assert "SLACK_WEBHOOK_URL" not in env
        assert env["GH_TOKEN"] == "${{ github.token }}"
        assert env["GH_REPO"] == "${{ github.repository }}"

    def test_slack_runs_only_after_a_breach(self) -> None:
        step = _step("Slack notification")
        assert step["if"] == "steps.count.outputs.level != 'OK'"
        assert step["continue-on-error"] is True
        assert step["env"]["SLACK_WEBHOOK_URL"] == "${{ secrets.SLACK_WEBHOOK_URL }}"
