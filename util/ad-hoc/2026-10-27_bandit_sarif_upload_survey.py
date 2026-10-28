#!/usr/bin/env python3
"""Survey the Bandit SARIF upload: did the step ever fail, and did every PR head get its required ``Bandit`` check?

Project: juniper-cascor
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-10-27
Status: ad-hoc — investigation (the flood-3 lead "juniper-cascor's Bandit upload step may fail while Bandit is a required check")
Retire when: that lead is closed in juniper-ml notes/JUNIPER_2026-10-08_JUNIPER-ECOSYSTEM_CURSOR-FLOOD-3-DISPOSITION.md §8 and nobody needs to re-run the evidence
Related: .github/workflows/ci.yml job ``security`` ("Security Scans"), step "Upload Bandit SARIF to GitHub Security";
         ruleset 15081045, whose required context ``Bandit`` carries integration_id 57789 (the github-advanced-security app)

Why the job result cannot answer this. The upload step carries ``continue-on-error: true``, so a failed
upload leaves "Security Scans" green. The step's own conclusion still records ``failure``. And the
required ``Bandit`` context is not an Actions job at all: it is a check run that GitHub Advanced Security
creates when it processes an uploaded Bandit analysis. A failed upload therefore shows up twice, as a
``failure`` step inside a green job and as a PR head with no ``Bandit`` check run.

For each of the newest ``--limit`` completed runs of ci.yml this reads the Security Scans job's step
conclusion and, for ``pull_request`` runs, the check runs named ``Bandit`` on the run's head SHA.
Read-only: ``gh api`` GETs only.

Usage:
    python3 util/ad-hoc/2026-10-27_bandit_sarif_upload_survey.py [--limit 200] [--repo pcalnon/juniper-cascor]
Exit: 0 when no upload step failed and every PR head whose upload succeeded carries a Bandit check run;
      1 when either is violated; 2 when the survey could not run or examined no upload at all
      (a survey that saw nothing is not a clean result).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # nosec B404 -- fixed-argv, read-only ``gh api`` GETs
import sys
import time
from collections import Counter

WORKFLOW = "ci.yml"
JOB = "Security Scans"
STEP = "Upload Bandit SARIF to GitHub Security"
CHECK = "Bandit"
APP = "github-advanced-security"
ATTEMPTS = 4  # a 5xx is retried with backoff; anything else fails at once
_SERVER_ERROR = re.compile(r"HTTP 5\d\d")


class SurveyError(RuntimeError):
    """The survey's own machinery failed; the result must not read as clean."""


def gh_api(path: str) -> dict:
    for attempt in range(1, ATTEMPTS + 1):
        proc = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=False)  # nosec B603 B607 -- fixed argv, read-only GET
        if proc.returncode == 0:
            return json.loads(proc.stdout)
        if attempt < ATTEMPTS and _SERVER_ERROR.search(proc.stderr):
            time.sleep(2**attempt)
            continue
        raise SurveyError(f"gh api {path} exited {proc.returncode} after {attempt} attempt(s): {proc.stderr.strip()[:300]}")
    raise SurveyError(f"gh api {path}: no attempt made")


def list_runs(repo: str, limit: int) -> list[dict]:
    runs: list[dict] = []
    page = 1
    while len(runs) < limit:
        batch = gh_api(f"repos/{repo}/actions/workflows/{WORKFLOW}/runs?status=completed&per_page=100&page={page}").get("workflow_runs", [])
        if not batch:
            break
        runs.extend(batch)
        page += 1
    return runs[:limit]


def upload_step(repo: str, run_id: int) -> tuple[str, str]:
    """Return (job conclusion, upload-step conclusion) for the run's Security Scans job."""
    for job in gh_api(f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100").get("jobs", []):
        if job.get("name") != JOB:
            continue
        for step in job.get("steps") or []:
            if step.get("name") == STEP:
                return str(job.get("conclusion")), str(step.get("conclusion") or step.get("status"))
        return str(job.get("conclusion")), "step-absent"
    return "job-absent", "job-absent"


def bandit_checks(repo: str, sha: str) -> list[str]:
    """Conclusions of the check runs named Bandit that the code-scanning app put on ``sha``."""
    data = gh_api(f"repos/{repo}/commits/{sha}/check-runs?check_name={CHECK}&per_page=100")
    return [str(c.get("conclusion")) for c in data.get("check_runs", []) if (c.get("app") or {}).get("slug") == APP]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="pcalnon/juniper-cascor")
    parser.add_argument("--limit", type=int, default=200, help="newest completed ci.yml runs to read (default 200)")
    args = parser.parse_args(argv)

    try:
        runs = list_runs(args.repo, args.limit)
        if not runs:
            print(f"::error::no completed {WORKFLOW} runs returned for {args.repo}")
            return 2
        tally: Counter[tuple[str, str, str]] = Counter()
        failed_uploads: list[str] = []
        pr_heads_ok = 0
        pr_heads_missing: list[str] = []
        executed = 0
        for run in runs:
            event = str(run.get("event"))
            job_conclusion, step_conclusion = upload_step(args.repo, run["id"])
            tally[(event, job_conclusion, step_conclusion)] += 1
            if step_conclusion in ("success", "failure"):
                executed += 1
            if step_conclusion == "failure":
                failed_uploads.append(f"run {run['id']} ({event}, {run.get('created_at')}): job {job_conclusion}, upload FAILED -- {run.get('html_url')}")
            if event == "pull_request" and step_conclusion == "success":
                conclusions = bandit_checks(args.repo, run["head_sha"])
                if conclusions:
                    pr_heads_ok += 1
                else:
                    pr_heads_missing.append(f"run {run['id']} head {run['head_sha'][:8]}: upload succeeded but no {CHECK} check run from {APP}")
    except SurveyError as exc:
        print(f"::error::{exc}")
        return 2

    print(f"repo={args.repo} workflow={WORKFLOW} runs_read={len(runs)} window={runs[-1].get('created_at')} .. {runs[0].get('created_at')}")
    print(f"upload steps that executed (success or failure): {executed}")
    print("event / Security Scans job conclusion / upload step conclusion -> runs")
    for (event, job_conclusion, step_conclusion), count in sorted(tally.items()):
        print(f"  {event:20s} {job_conclusion:10s} {step_conclusion:12s} {count}")
    print(f"pull_request runs whose upload succeeded and whose head carries a {CHECK} check run from {APP}: {pr_heads_ok}")
    for line in failed_uploads:
        print(f"  FAILED UPLOAD: {line}")
    for line in pr_heads_missing:
        print(f"  MISSING CHECK: {line}")

    if executed == 0:
        print("::error::no upload step executed in the window; this survey saw nothing and proves nothing")
        return 2
    if failed_uploads or pr_heads_missing:
        print(f"RESULT: {len(failed_uploads)} failed upload step(s), {len(pr_heads_missing)} PR head(s) missing the {CHECK} check")
        return 1
    print(f"RESULT: no upload step failed in {executed} executed upload(s); every surveyed PR head with a successful upload carries the {CHECK} check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
