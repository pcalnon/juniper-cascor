# CI/CD Reference

**Project**: Juniper Cascor  
**Workflow Files**: `.github/workflows/ci.yml`, `codeql.yml`, `security-scan.yml`, `claude.yml`, `golden-regression.yml`, `conformance.yml`, `ci-protocol.yml`, `ci-cascor-model.yml`, `lockfile-update.yml`, `publish.yml`, `publish-protocol.yml`, `publish-cascor-model.yml`

**Last Updated**: 2026-11-15

---

## Workflow Configuration Reference

### Trigger Events and Branches

| Event | Branches |
|-------|----------|
| `push` | `main`, `develop`, `feature/**`, `fix/**` |
| `pull_request` | `main`, `develop` |

### Jobs Overview

| Job | Purpose | Runs On | Dependencies |
|-----|---------|---------|--------------|
| `lint` | Code quality checks (Black, isort, Flake8, MyPy) | All triggers | None |
| `test` | Unit tests with coverage | All triggers | None |
| `integration` | Integration tests | PRs only | `test` |
| `quality-gate` | Aggregated pass/fail status | Always | `lint`, `test` |
| `notify` | Build status notification | Always | `quality-gate` |

---

## WS-6 Gate Workflows

Dedicated serial lanes (OUT-12 / OUT-13). Source of truth: the workflow YAML files.

| Workflow | Marker / flags | Test path | Python | Torch | Concurrency group |
|----------|----------------|-----------|--------|-------|-------------------|
| `golden-regression.yml` | `-m golden --golden --slow --integration` | `src/tests/integration` | 3.13 | 2.11.0 CPU | `golden-${{ github.ref }}` |
| `conformance.yml` | `-m conformance --conformance --slow --integration` | `src/tests/conformance` | 3.13 | 2.11.0 CPU | `conformance-${{ github.ref }}` |

Shared env (job-level, before interpreter start): `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`,
`NUMEXPR_NUM_THREADS=1`, `CASCOR_NUM_PROCESSES=1`, `JUNIPER_CASCOR_LOG_LEVEL=ERROR`.

| Setting | Value |
|---------|-------|
| Triggers | `push` → `main`; `pull_request` → `main`/`develop`; `workflow_dispatch` |
| `cancel-in-progress` | `true` |
| `permissions.contents` | `read` |
| pytest timeout | `300` |
| Artifacts | `golden-regression-results` / `conformance-results` (JUnit, 30 days) |
| Coverage | **Not** collected (dedicated lanes, not the 80% unit gate) |

Collection gates live in `src/tests/conftest.py` (`--golden` / `--conformance`).
Fixtures: `src/tests/fixtures/golden/`. Capture rewrite: `GOLDEN_CAPTURE=1`.

### Package CI Workflows

| Workflow | Paths filter | Python matrix | Coverage |
|----------|--------------|---------------|----------|
| `ci-protocol.yml` | `juniper-cascor-protocol/**` | 3.12, 3.13; 3.15 informational (never blocks) | `--cov-fail-under=95` + `juniper-coverage-gap-map --enforce` |
| `ci-cascor-model.yml` | `juniper-cascor-model/**` | 3.12; 3.15 informational (never blocks) | package + candidate/utils/log_config/constants; per-file enforce |

## CodeQL Analysis

Semantic SAST for Python. Source of truth: `.github/workflows/codeql.yml` (fleet template; header names `juniper-data/.github/workflows/codeql.yml` as the copy origin). Operator runbook: [CI Manual — CodeQL Analysis](MANUAL.md#codeql-analysis).

| Setting | Value |
|---------|-------|
| Job | `analyze` on `ubuntu-latest` |
| Language matrix | `python` (`fail-fast: false`) |
| Queries | `+security-and-quality` (default security pack **plus** quality queries) |
| Steps | `codeql-action/init` → `autobuild` → `analyze` (`category: /language:python`) |
| Permissions | `actions: read`, `contents: read`, `security-events: write` |
| Required check | **Yes** — the `main` ruleset (`juniper-cascor-rules`, id `15081045`) requires `Analyze (python)`, so a red CodeQL run blocks the merge. Its `code_scanning` rule also blocks on a CodeQL alert at `error` or a security alert at `high` or above. Standalone workflow, so not in the Quality Gate `needs:`. |
| `workflow_dispatch` | **None** — push, PR, or wait for the Monday cron |

### Triggers

| Event | Branches |
|-------|----------|
| `push` | `main`, `develop` |
| `pull_request` | `main` only (a PR targeting `develop` does **not** run CodeQL) |
| `schedule` | `0 6 * * 1` (Monday 06:00 UTC — same cron as `security-scan.yml`) |

### Action pins (Dependabot `codeql-action` group)

`.github/dependabot.yml` groups `github/codeql-action*` so these four SHA pins move together:

| Use | Workflow |
|-----|----------|
| `github/codeql-action/init` | `codeql.yml` |
| `github/codeql-action/autobuild` | `codeql.yml` |
| `github/codeql-action/analyze` | `codeql.yml` |
| `github/codeql-action/upload-sarif` | `ci.yml` **Security Scans** job (Bandit SARIF → GitHub code scanning) |

Do not document a specific SHA or `v4.x.y` patch — Dependabot refreshes the pin and the `# vX.Y.Z` comment. Partial hand-edits of one of the four uses leave CodeQL init/analyze and Bandit SARIF upload on different action versions.

### Contrast with the other security lanes

| Lane | Workflow | What it runs | GitHub code scanning |
|------|----------|--------------|----------------------|
| CodeQL | `codeql.yml` | CodeQL Python query pack | Native CodeQL upload from `analyze` |
| PR / push CI | `ci.yml` job `security` | Gitleaks (skip `repository_dispatch`; a pull request is scanned over its whole `base..head`, see [Secret Scan (Gitleaks)](#secret-scan-gitleaks)), Bandit medium+ (blocking) + SARIF (`--exit-zero` then upload), pip-audit | Bandit SARIF via `upload-sarif` (`if: always()`, `continue-on-error: ${{ github.event_name != 'pull_request' }}`: a failed upload fails Security Scans on a PR only) |
| Scheduled | `security-scan.yml` | Bandit medium+ + `pip-audit --strict`; artifact `security-reports` | **None** (no CodeQL, no Gitleaks, no `upload-sarif`) |

`security-scan.yml` has `workflow_dispatch`; `codeql.yml` does not.

## Secret Scan (Gitleaks)

Gitleaks runs in one place, the `security` job (`Security Scans`) of `.github/workflows/ci.yml`. `Security Scans` is a required check in ruleset `juniper-cascor-rules`, and so is `Quality Gate`, which fails when that job fails. The weekly `security-scan.yml` and pre-commit do not run Gitleaks. The engine is gitleaks **8.24.3** on every event that scans. It loads this repository's `.gitleaks.toml`, which extends the built-in ruleset (`[extend] useDefault = true`) and adds a top-level allowlist: the paths `conf/temp/`, `notes/setup_config_guides/` and `notes/FIX_RESOURCE_TRACKER_AND_WARNINGS.md`, `YOUR_API_KEY`-style placeholders, and conda build hashes. The job log says so: `using existing gitleaks config .gitleaks.toml from (--source)/.gitleaks.toml`, then `extending config with default config`. There is no `.gitleaksignore`.

| Event | Step | What is scanned |
|-------|------|-----------------|
| `push` to `main` / `develop` | `gitleaks/gitleaks-action` at `e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e` (v3.0.0) | `--no-merges --first-parent <first>^..<last>` across the push payload's commits, `git log -1` when the payload holds one, and nothing when it holds none |
| `workflow_dispatch` | The same action | No `--log-opts`: every commit reachable from every branch and tag of the `fetch-depth: 0` checkout |
| `pull_request` into `main` / `develop` | `Install Gitleaks (pull request)`, then `Run Gitleaks over the pull request (Secrets Detection)` | Every commit in `<base>..<head>`, from the event payload's `pull_request.base.sha` and `pull_request.head.sha`, with `--remerge-diff`, which adds what each two-parent merge commit changed itself |
| `repository_dispatch` (`data-client-updated`, `cascor-worker-updated`) | None | Nothing, as before: the action rejects the event, and a dispatch brings no new commit |

**Engine pin.** The action step sets `GITLEAKS_VERSION: "8.24.3"`. That is the action's own default at the pinned SHA (`src/index.js`) and the version its log named here before the variable was set (`gitleaks version: 8.24.3`, run 43662358600 of pull request #759), so setting it changes no engine. It is set so that a bump of the action SHA cannot move the engine on `push` while pull requests stay on the installed one. The install step carries the same version and `GITLEAKS_SHA256` (`9991e0b2903da4c8f6122b5c3186448b927a5da4deef1fe45271c3793f4ee29c`, the `linux_x64` line of the release's `gitleaks_8.24.3_checksums.txt`), and runs `sha256sum --check --strict` before it unpacks the tarball. A bump moves both versions and the digest together.

**Pull requests.** The two pull-request steps are juniper-data's (juniper-data#530), ported under ruling 10 in section 8 of juniper-ml's `notes/JUNIPER_2026-11-06_JUNIPER-ECOSYSTEM_CURSOR-FLOOD-4-DISPOSITION.md`; the scan step's shell is unchanged. gitleaks-action lists a pull request's commits with one unpaginated request (30 per page, oldest first) and scans `--no-merges --first-parent <page[0]>^..<page[-1]>`. Commit 31 onwards is never scanned, and from the page's last commit `--first-parent` drops every commit a merge brought into the branch. The second loss happened here. Of the 734 pull requests before this change (#1 to #760, read on 2026-11-15), none had more than 30 commits (the largest had 14), but a simulation of the action's window over each one's commit listing finds six whose own non-merge commits it never reached: #94, #178, #179, #302, #308 and #633. The run of #633 shows it: 9 commits, 4 of them merges; the action ran `--no-merges --first-parent aecd0bb8…^..0c1a0f84…`, logged `2 commits scanned.`, and never reached 3 of the pull request's 5 non-merge commits. Because the action passes `--no-merges`, no pull-request scan here saw what a merge commit changed itself either. The scan step:

- refuses a base or head SHA that is not a 40-hex commit in the checkout, before it scans
- logs `Scanning N commit(s) in <base>..<head>; the pull request reports M.`, and warns when N and M differ
- refuses a range that holds an octopus merge, for which git prints no `--remerge-diff` patch
- puts a `git` filter first on the engine's `PATH`, which deletes the `remerge ...` notes git writes into a merge's file headers and exits with `git`'s status; with the notes left in, gitleaks 8.24.3 can report a finding in a merge's own changes as `b/<path>`, or without its commit
- runs the action's detect flags, `detect --redact -v --exit-code=2 --report-format=sarif --log-level=debug`, with `--log-opts=--remerge-diff <base>..<head>` and the report at `reports/security/gitleaks.sarif`
- fails the way the action does: a leak (gitleaks exit 2) fails the step with 1, and any other non-zero gitleaks exit fails it with that code

The scan makes no API call, reads no token and posts no review comment (the action's comments needed `pull-requests: write`, which this job does not grant). A finding is in the step log, with its `Fingerprint:` line, and in the report. The job summary carries the range's commit count and gitleaks' exit code. Gitleaks' own `N commits scanned.` line counts only the commits for which `git log -p` printed a patch of a text file that is not a deletion (`detect/git.go` in 8.24.3), so it is smaller than N when the range holds clean merges: juniper-data#532 logged `Scanning 5 commit(s)` and `1 commits scanned.` for one commit and four clean merges of `main`.

**Reports.** On a pull request the redacted report is `gitleaks.sarif` in the `security-reports` artifact (30 days), beside Bandit's SARIF and pip-audit's `pip-freeze.txt`. The action's own `gitleaks-results.sarif` artifact is still uploaded on `push` and `workflow_dispatch`, and no longer on a pull request.

**Tests.** `src/tests/unit/test_gitleaks_pr_scan.py` has 25 cases. Six read `ci.yml`: the engine version on both paths, the digest and its check line, the action's `if`, the workflow's events, the scan's event range and step order, `fetch-depth: 0`, the unit lane's selection, and the wiring that makes a leak fail the required check. For that wiring, the job must keep the name `Security Scans` with no `if` and no `continue-on-error`; the install step, the scan step, the self-test and the step that installs its tools must carry no `continue-on-error`; the self-test and its tools must skip `repository_dispatch` and no other event; the self-test's `run:` must be exactly the command below, after pip-audit; and `Upload Security Reports` must run under `if: always()` on `reports/security/`, after the scan, so that a failed scan still uploads its report. Eleven run the scan step's own `run:` under `bash -e` with a stand-in `gitleaks`. Eight run it with the real engine in a scratch repository that carries this repository's `.gitleaks.toml`: a token in the 1st and the 35th commit of a 35-commit range, one a merge brought in, one that only a merge commit's own changes hold, a finding's commit and path under a conflict note that says `commit`, a clean range, and a clean merge of a base branch that holds a token. The file builds its tokens at run time, and the scratch repositories plant them outside the allowlisted paths, so a config that stopped extending the built-in rules, or allowed every path, turns the finding cases red. Two lanes run the file. The `unit-tests` job's `pytest -m "unit and not slow" src/tests/unit` selects it by its module-level `unit` mark on each of its four required legs (Python 3.12, 3.13 and 3.14 on Ubuntu, 3.12 on macOS), where the eight real-engine cases skip when no `gitleaks` is on `PATH`. The `security` job installs `pytest`, `pytest-asyncio`, `pytest-timeout` and `PyYAML` (`Install the Gitleaks self-test's tools`), then runs the file alone with `GITLEAKS_PARITY_REQUIRED=1`, which turns a missing binary into a failure, on every event but `repository_dispatch`:

```bash
GITLEAKS_PARITY_REQUIRED=1 python -m pytest src/tests/unit/test_gitleaks_pr_scan.py --noconftest -v -v --timeout=120
```

`--noconftest` keeps out `src/tests/conftest.py`, which imports torch and cascor's own modules; the file needs neither. The binary is the one the action put on `PATH`, or on a pull request the one the install step put there. Locally, set `GITLEAKS_BIN` to a gitleaks 8.24.3 binary. `util/ad-hoc/2026-11-14_verify_gitleaks_pr_scan_tests_are_not_vacuous.py` undoes one property at a time, nearly always by an edit of `ci.yml`, and shows the cases that pin it fail. CI does not call it. The action's window is rebuilt in a scratch repository by juniper-data's `util/ad-hoc/2026-11-08_reproduce_gitleaks_pr_commit_window.py` and juniper-data-client's `util/ad-hoc/2026-11-11_reproduce_gitleaks_pr_commit_window.py`.

**A finding that is a false positive.** For one finding, commit a `.gitleaksignore` at the repo root holding the finding's `Fingerprint:` line as the log prints it, `<40-hex commit>:<file>:<rule>:<line>`, under a comment that names the class. A fingerprint that does not start with a commit is file-scoped and would hide a later secret on that line: never add one. A class of documentation or build values belongs in `.gitleaks.toml`'s allowlist, as the conda build hashes do.

**Not covered**, as before this change: on a pull request, a commit that a force-push removed from the branch before any run scanned it (`cancel-in-progress` cancels the run that would have; juniper-data#536 tracks that gap), and whatever a `.gitleaks.toml` or `.gitleaksignore` changed or added by the pull request itself allows, because the scan reads both from the merge checkout. On `push`, the action's `--first-parent` still leaves out the commits a merge commit brought in; the pull-request scan covered them when they came through a pull request. A `repository_dispatch` run scans nothing.

**When a step fails.**

- `Install Gitleaks (pull request)` fails at `sha256sum`: the downloaded tarball does not match `GITLEAKS_SHA256`. After an engine bump, set `GITLEAKS_VERSION` on this step and on the action step, and take the digest from the `linux_x64` line of the new release's `gitleaks_<version>_checksums.txt`, never from the file the runner downloaded. The step also refuses a runner that is not `Linux/X64`.
- The scan fails with `is not a commit in this checkout`: `pull_request.base.sha` or `pull_request.head.sha` is not a commit the checkout holds. Check that the job's checkout still sets `fetch-depth: 0`.
- The scan fails with `octopus merge(s)`: redo the merge as two-parent merges, one branch at a time, and push.
- The self-test errors at setup with `GITLEAKS_PARITY_REQUIRED=1 but no gitleaks binary`: neither path put the binary on `PATH`. `ci.yml` runs on `push`, `pull_request`, `repository_dispatch` and `workflow_dispatch`; an event added to `on:` that the action rejects fails `test_the_action_no_longer_scans_a_pull_request` first.
- A token you planted stays green: on a pull request a commit already on the base branch is outside `<base>..<head>`, a `push` whose payload has no commits scans nothing, a file under an allowlisted path is skipped, and a `.gitleaks.toml` without `[extend] useDefault = true` loads no built-in rule.

Local commands, run at the repo root:

```bash
gitleaks git --redact .
gitleaks dir --redact .
gitleaks git --redact --log-opts="--remerge-diff origin/main..HEAD" .
```

`git` scans history and `dir` scans the working tree, gitignored files included. The third scans what a pull request from the current branch into an up-to-date `main` would, without the step's `git` filter.

## Claude Code Workflow

Interactive `@claude` assistant. Source of truth: `.github/workflows/claude.yml`. The file header names `juniper-ml/.github/workflows/claude.yml` as the copy origin and says `ANTHROPIC_API_KEY` is an org-level secret — confirm this repo can read that secret before relying on the job. Operator runbook: [CI Manual — Claude Code Workflow](MANUAL.md#claude-code-workflow).

| Setting | Value |
|---------|-------|
| Workflow / job | **Claude Code** / `claude` on `ubuntu-latest` |
| Required check | **No.** Comment, review, and issue events only. The job does not feed `ci.yml`'s Quality Gate. |
| `workflow_dispatch` | **None** |
| Checkout | SHA-pinned `actions/checkout`, `fetch-depth: 1` |
| Action input | `anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}` only |
| Permissions | `contents: write`, `pull-requests: write`, `issues: write`, `id-token: write`, `actions: read` |

This workflow sets no Bedrock, Vertex, Foundry, or workload-identity inputs, so the model authenticates with `ANTHROPIC_API_KEY`. `id-token: write` is **used** on every run that starts: no `github_token` input is passed, so the action trades the job's OIDC token for a Claude GitHub App installation token, which makes the GitHub API calls and pushes. The Claude GitHub App must be installed on this repo.

### Triggers

The job `if` is `contains(..., '@claude')`, which GitHub evaluates as a case-insensitive substring test.

| Event | Types | Text the `if` reads |
|-------|-------|---------------------|
| `issue_comment` | `created` | `github.event.comment.body` |
| `pull_request_review_comment` | `created` | `github.event.comment.body` |
| `pull_request_review` | `submitted` | `github.event.review.body` |
| `issues` | `opened`, `assigned` | `github.event.issue.body` or `github.event.issue.title` |

These events are absent from `on:`, so they never schedule the job: `pull_request` (title or body), `edited` on a comment or review, and `issues` `labeled`.

### Second gate inside the action

The workflow `if` only decides whether the job starts. The pinned `anthropics/claude-code-action` then applies its own checks. This workflow leaves the action defaults in place: `trigger_phrase` `@claude`, `assignee_trigger` unset, `allowed_bots` empty, `label_trigger` `claude` (unused here), `branch_prefix` `claude/`, `branch_name_template` empty.

Order in the action entrypoint:

1. **GitHub App token.** The OIDC exchange runs first. It fails red when the App is not installed or `id-token: write` is missing; a workflow-validation rejection (expected while the workflow file itself is changing) logs `Skipping action due to workflow validation:` and exits green.
2. **Write permission.** Collaborator permission must be `admin` or `write`. Otherwise the step throws `Actor does not have write permissions to the repository`. A login ending in `[bot]` passes without a lookup; an app account without that suffix fails here.
3. **Phrase match.** A miss logs `No trigger found, skipping remaining steps` and the step returns success, so the job stays green.
4. **Human actor**, only after a phrase match, inside tag-mode prepare. `allowed_bots` is empty, so a non-User throws `Workflow initiated by non-human actor:`.

The action's phrase test is case-insensitive. It requires the start of the text or whitespace before `@claude`, and whitespace, one of `.,!?;:`, or the end of the text after it. Both gates ignore case; they differ only on word boundaries:

| Text | Job | Action |
|------|-----|--------|
| `@claude` as its own word in a new comment, a submitted review, or a newly opened issue title/body | Starts | Runs when the actor is a human with write access |
| `@Claude` | Starts | Runs like `@claude` (both checks ignore case) |
| `@claudette`, `foo@claude`, or `@claude)` | Starts | Skips green (`No trigger found, skipping remaining steps`) for an actor with write access; anyone else fails the write check first |
| Issue **assigned**, title or body already contains `@claude` | Starts | Skips green when the assigner has write access. Body and title are scanned on `opened` only, and `assignee_trigger` is unset |
| Issue labeled `claude` | Stays idle | `labeled` is absent from `on:`. The default `label_trigger` never fires here |
| `@claude` in a pull request title or body | Stays idle | `pull_request` is absent from `on:` |
| Bot account whose comment contains `@claude` | Starts when the workflow `if` matches | A `[bot]` login passes the write check, then fails the human check on a bounded match or skips green without one. An app login without the `[bot]` suffix fails the write check |

### Where it writes

| Event | Branch |
|-------|--------|
| Comment or review on an **open** pull request | Checks out that PR's head branch (a fork PR is fetched as `refs/pull/<number>/head`). The workflow's `fetch-depth: 1` is shallow, so the action fetches again at `max(commit count, 20)`. Pushes go only to `origin` (this repo): a same-repo PR's head is updated, a fork PR's is not. |
| Issue, or a **closed / merged** pull request | Creates `claude/issue-<number>-<YYYYMMDD-HHmm>` or `claude/pr-<number>-<YYYYMMDD-HHmm>` from the default branch; it reaches GitHub only if Claude pushes. The timestamp comes from the runner's local clock (local `Date` getters). The name is lowercased and truncated to 50 characters. |

Pushes authenticate with the Claude GitHub App token: the action removes the checkout's credential, so the job's own `contents: write` does not authorize them. An empty `branch_name_template` is the fallback above; a template that collapses to an empty name uses the same fallback.

### Pins

Leave the action SHA and the `# vX.Y.Z` comment in the workflow file. Dependabot's `github-actions` group covers only `github/codeql-action*`, so `claude-code-action` and the `actions/checkout` pin in this file each move in their own PR. The second gate lives in the action's source, so any pin bump can change it (defaults, matcher, check order, messages, branch naming) even when `on:`, the job `if`, `permissions`, and `with:` are unchanged. Re-check it at the new SHA. Verified at `v1.0.240`; `v1.0.237` behaves the same.

A missing `ANTHROPIC_API_KEY` is checked only after a phrase match. The step then throws `Environment variable validation failed:` and lists `Either ANTHROPIC_API_KEY, CLAUDE_CODE_OAUTH_TOKEN, or workload identity federation (ANTHROPIC_FEDERATION_RULE_ID and ANTHROPIC_ORGANIZATION_ID) is required when using direct Anthropic API.` A phrase miss stays green without the secret.

## Publish Workflows

Trusted Publishing (OIDC) pipelines. Source of truth: the workflow YAML files under `.github/workflows/`.

| Workflow | Package | Trigger | Tag guard | Environments | Publish action |
|----------|---------|---------|-----------|--------------|----------------|
| `publish.yml` | `juniper-cascor` | `release: published` | `startsWith(tag, 'v')` | `testpypi` → `pypi` | `pypa/gh-action-pypi-publish` (SHA-pinned) |
| `publish-protocol.yml` | `juniper-cascor-protocol` | `release` + `workflow_dispatch` | `juniper-cascor-protocol-v*` | `testpypi` → `pypi` | same pin |
| `publish-cascor-model.yml` | `juniper-cascor-model` | `release` + `workflow_dispatch` | `juniper-cascor-model-v*` | `testpypi` → `pypi` | same pin |

### One-Artifact Contract

All three publishers build once and hand both publish jobs one artifact, `<PUBLISH_PACKAGE>-dist`: the dist directory and `SHA256SUMS`, kept 90 days so it outlives the `pypi` gate. The steps that carry it are a template: they read only the workflow `env` values `PUBLISH_PACKAGE` and `PUBLISH_DIST_DIR`, and a JSON check's `INDEX_URL` ([`MANUAL.md` § One Artifact, Checked Against TestPyPI](MANUAL.md#one-artifact-checked-against-testpypi)).

| Step | Job | What it checks |
|------|-----|----------------|
| `Record SHA256SUMS` | build | The dist directory holds only wheels and sdists; writes `<sha256>  <file>` lines, the format `sha256sum --check` reads |
| `Upload dist artifacts` | build | Uploads the dist directory and `SHA256SUMS` as one artifact (`if-no-files-found: error`, `retention-days: 90`) |
| `Download dist artifacts` | both publish jobs | Restores both into the workspace; no checkout may run after it before the job's last template step |
| `Verify the dist against SHA256SUMS` | both publish jobs, before the upload | The directory holds exactly the listed files, with their sha256 |
| `Verify TestPyPI serves exactly these files` | TestPyPI job, after the upload | `https://test.pypi.org/pypi/<package>/<version>/json` lists exactly those files and sha256 |
| `Verify TestPyPI still serves exactly these files` | PyPI job, before the upload | The same, again |
| `Verify PyPI serves exactly these files` | PyPI job, after the upload | The same, on `https://pypi.org` |

The JSON checks poll up to 20 times, 15 s apart (`HASH_CHECK_ATTEMPTS`, `HASH_CHECK_INTERVAL`), while the version is not served yet or a file is missing (CDN lag). A file listed with another sha256, or one the run did not build, fails at once. The version comes from the file names in `SHA256SUMS`, not from the tag. Pinned by `src/tests/unit/test_publish_one_artifact.py`.

### Permissions and Concurrency

| Setting | Value | Why |
|---------|-------|-----|
| `permissions.id-token` | `write` | OIDC token for Trusted Publishing |
| `permissions.contents` | `read` (protocol/model) | Checkout / sparse-checkout for version verify |
| `concurrency.group` | `publish-<pkg>-${{ github.ref_name }}` (protocol/model) | Serialize re-fires against immutable TestPyPI |
| `cancel-in-progress` | `false` | Never cancel a mid-upload publish |

### TestPyPI Verify Contract

| Package | Install flags | Success assertion |
|---------|---------------|-------------------|
| `juniper-cascor` | `--no-deps --index-url https://test.pypi.org/simple/` + 30s sleep | `from juniper_cascor import __version__` |
| `juniper-cascor-model` | same + 5×10s retry | `import juniper_cascor_model` + `__version__` |
| `juniper-cascor-protocol` | same + 5×10s retry | `importlib.metadata.version(...)` (no package import) |

Constraints:

- No `--extra-index-url https://pypi.org/simple/` on the verify step (anti-target-squatting; juniper-ml#384).
- Do not add `push: tags` next to `release: published` (double-fire race; juniper-ml#555).
- No `skip-existing` on a TestPyPI upload: an earlier file under the same name would be kept silently and the install check would test it. The sha256 check after the upload would fail the run instead.
- Dependabot bumps the `gh-action-pypi-publish` SHA and `# vX.Y.Z` comment in all three files together; keep them aligned.

### Twine Pin Surfaces

| Surface | Source of truth | Install / pin pattern |
|---------|-----------------|-----------------------|
| CI pip freeze | `conf/requirements_ci.txt` | Exact `twine==X.Y.Z` (Dependabot may bump; the artifact is also regenerated by `juniper-generate-dep-docs`) |
| CI conda freeze | `conf/conda_environment_ci.yaml` | Exact pin under `pip:` (can lag the pip freeze) |
| Package build validation | publish + `ci-protocol` / `ci-cascor-model` workflows | `pip install build twine` or `pip install --upgrade build twine`, then `twine check dist/*` |
| Release upload | `pypa/gh-action-pypi-publish` | Twine bundled in the SHA-pinned action — independent of the repo freezes |

Twine **7.0.0** rejects Metadata-Version 2.0 and requires `packaging >= 26.1`. Cascor packages use PEP 621 + setuptools; still re-run `twine check` under ≥ 7 after a major freeze bump. Full operator notes: [CI/CD Manual — Twine Pin Surfaces](MANUAL.md#twine-pin-surfaces).

### Step-by-Step Breakdown

#### Lint Job

1. **Checkout Code** - Shallow clone (`fetch-depth: 1`)
2. **Set up Python** - Python 3.14
3. **Install Linting Tools** - black, isort, mypy, flake8 with plugins
4. **Run Black** - Format check
5. **Run isort** - Import sort check
6. **Run Flake8** - Linting
7. **Run MyPy** - Type checking

#### Test Job

1. **Checkout Code** - Shallow clone
2. **Set up Conda** - Miniforge with mamba, `JuniperCascor` environment
3. **Verify Conda Environment** - List packages, verify Python
4. **Free Disk Space** - Clean conda cache, remove unused tools
5. **Install Test Dependencies** - pytest, pytest-cov, pytest-timeout, pytest-xdist
6. **Verify Test Files** - Create required directories
7. **Run Unit Tests** - Fast tests only (`-m "unit and not slow"`)
8. **Upload Coverage Report** - XML and HTML artifacts
9. **Upload Test Results** - JUnit XML artifact
10. **Enforce Coverage Gate** - 80% aggregate minimum (hard fail)
11. **Test Summary** - Report generation status

#### Integration Job

1. **Checkout Code** - Shallow clone
2. **Set up Conda** - Same as test job
3. **Install Test Dependencies** - pytest suite
4. **Create Required Directories** - logs, reports
5. **Run Integration Tests** - Fast tests only (`-m "integration and not slow"`)
6. **Integration Test Summary** - Completion status

---

## Coverage Gates

### Current Configuration

| Setting | Value |
|---------|-------|
| **Aggregate threshold** | 80% |
| **Enforcement** | Hard fail in the `unit-tests` job |
| **Command** | `python -m coverage report --fail-under=${COVERAGE_FAIL_UNDER}` |

### Failure Behavior

When aggregate coverage falls below the threshold:

- The standalone coverage report step fails.
- The `unit-tests` job fails and blocks merge when it is a required status check.
- Coverage artifacts are still uploaded with `if: always()` for inspection.

### How to Increase Thresholds

1. Edit `.github/workflows/ci.yml`
2. Locate the top-level `COVERAGE_FAIL_UNDER` value
3. Modify it to the desired aggregate threshold:

```yaml
env:
  COVERAGE_FAIL_UNDER: "85"
```

### Recommended Threshold Progression

| Phase | Threshold | Notes |
|-------|-----------|-------|
| Current aggregate gate | 80% | Enforced by `.github/workflows/ci.yml` |
| Comprehensive | 90%+ | Research codebase goal |

### Per-Module Thresholds

Per-source-file and pooled packaged-sub-module bars are tracked with `juniper-coverage-gap-map` during the rollout. They are advisory in `juniper-cascor` until the final gate PR adds the blocking `--enforce` step.

| Scope | Advisory bar | Measurement |
|-------|--------------|-------------|
| Source file | >=90% statement | Coverage JSON from `pytest -m "unit and not slow" src/tests/unit --cov=src` |
| Packaged sub-module | >=95% pooled statement | Covered statements divided by total statements for the sub-module |

Or configure in `pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = 80
```

---

## Artifact Reference

### Generated Artifacts

| Artifact | Path | Contents |
|----------|------|----------|
| Coverage XML | `src/tests/reports/coverage.xml` | Machine-readable coverage data |
| Coverage HTML | `src/tests/reports/htmlcov/` | Interactive HTML coverage report |
| JUnit XML | `src/tests/reports/junit.xml` | Test results for CI integrations |

### Artifact Uploads

| Artifact Name | Files Included | Retention |
|---------------|----------------|-----------|
| `coverage-report-{python-version}` | `coverage.xml`, `htmlcov/` | 30 days |
| `test-results-{python-version}` | `junit.xml` | 30 days |

### Accessing Artifacts

1. Navigate to the GitHub Actions run
2. Scroll to the "Artifacts" section at the bottom
3. Download the desired artifact ZIP file

---

## Test Marker Mapping

### CI Job to Marker Mapping

| Job | Marker Expression | Purpose |
|-----|-------------------|---------|
| `test` (Unit Tests) | `-m "unit and not slow"` | Fast unit tests only |
| `integration` | `-m "integration and not slow"` | Fast integration tests only |

### Available Markers

| Marker | Description | CI Inclusion |
|--------|-------------|--------------|
| `unit` | Unit tests for individual components | Included in `test` job |
| `integration` | Integration tests for full workflows | Included in `integration` job |
| `performance` | Performance and benchmarking tests | Not run in CI |
| `slow` | Tests that take a long time | **Excluded** from CI |
| `gpu` | Tests requiring GPU/CUDA | Not run in CI |
| `multiprocessing` | Tests using multiprocessing | Runs if marked `unit` |
| `spiral` | Spiral problem tests | Runs based on other markers |
| `correlation` | Correlation calculation tests | Runs based on other markers |
| `network_growth` | Network growth algorithm tests | Runs based on other markers |
| `candidate_training` | Candidate unit training tests | Runs based on other markers |
| `validation` | Input validation tests | Runs based on other markers |
| `accuracy` | Accuracy calculation tests | Runs based on other markers |
| `early_stopping` | Early stopping logic tests | Runs based on other markers |

### Running Slow Tests Locally

```bash
# Run slow tests with extended timeout
cd src/tests && python -m pytest -m slow --timeout=0

# Run all tests including slow
cd src/tests && python -m pytest --timeout=300
```

---

## Timeout Configuration

### Job-Level Timeouts

| Job | Timeout | Notes |
|-----|---------|-------|
| `test` | No explicit limit | GitHub default: 6 hours |
| `integration` | No explicit limit | GitHub default: 6 hours |

### Test-Level Timeouts

| Configuration | Value | Location |
|---------------|-------|----------|
| Global pytest timeout | 60 seconds | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Unit test timeout (CI) | 60 seconds | `ci.yml` (`--timeout=60`) |
| Integration test timeout (CI) | 120 seconds | `ci.yml` (`--timeout=120`) |
| Slow test timeout | 300 seconds | Per-test `@pytest.mark.timeout(300)` |

### Slow Test Handling

Tests marked with `@pytest.mark.slow` are:

- **Excluded** from CI runs by default
- Expected to have individual 300-second timeouts
- Run separately with `pytest -m slow --timeout=0`

To run slow tests with disabled global timeout:

```bash
pytest -m slow --timeout=0
```

### Timeout Method

```ini
timeout_method = signal
```

Uses POSIX signals for timeout enforcement (Linux/macOS compatible).

---

## Status Badges

### Badge URL Format

```markdown
![CI/CD Pipeline](https://github.com/pcalnon/juniper-cascor/actions/workflows/ci.yml/badge.svg)
```

For Juniper Cascor:

```markdown
![CI/CD Pipeline](https://github.com/pcalnon/juniper-cascor/actions/workflows/ci.yml/badge.svg)
```

### Branch-Specific Badges

```markdown
![CI/CD (main)](https://github.com/pcalnon/juniper-cascor/actions/workflows/ci.yml/badge.svg?branch=main)
![CI/CD (develop)](https://github.com/pcalnon/juniper-cascor/actions/workflows/ci.yml/badge.svg?branch=develop)
```

### How to Add to README

Add to the top of `README.md`:

```markdown
# Juniper Cascor

![CI/CD Pipeline](https://github.com/pcalnon/juniper-cascor/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.14-blue)
![License](https://img.shields.io/badge/license-MIT-green)

...
```

### Badge Status Meanings

| Badge | Meaning |
|-------|---------|
| ![passing](https://img.shields.io/badge/build-passing-brightgreen) | All required jobs passed |
| ![failing](https://img.shields.io/badge/build-failing-red) | One or more required jobs failed |
| ![pending](https://img.shields.io/badge/build-pending-yellow) | Workflow is currently running |
| ![no status](https://img.shields.io/badge/build-no%20status-lightgrey) | No workflow runs yet |

---

## Environment Configuration

### Python Version

- **CI Version**: Python 3.14
- **Matrix Testing**: `unit-tests` runs Python 3.12, 3.13 and 3.14 on Ubuntu and 3.12 on macOS (all four required), plus an informational Python 3.15 leg on Ubuntu that reports but never blocks; `pre-commit` runs 3.12, 3.13 and 3.14

### Conda Environment

- **Environment Name**: `JuniperCascor`
- **Environment File**: `conf/conda_environment.yaml`
- **Channels**: `conda-forge`, `pytorch`, `nvidia`
- **Package Manager**: mamba (faster than conda)

### Disk Space Optimization

The workflow includes disk space cleanup:

- Conda cache cleaning
- Removal of `/usr/share/dotnet`
- Removal of `/opt/ghc`

---

## Lockfile Update Workflow

Source of truth: `.github/workflows/lockfile-update.yml`. Companion gate: `lockfile-check` ("Lockfile Freshness") in `ci.yml`.

### Triggers

| Event | Filter | Purpose |
|-------|--------|---------|
| `push` | `dependabot/pip/**` and `github.actor == dependabot[bot]` | Auto-regen after Dependabot bumps |
| `pull_request` | paths include `pyproject.toml`, same-repo head only | Cover manual range edits; forks skipped |

### PAT availability gate

Checkout/push uses `secrets.CROSS_REPO_DISPATCH_TOKEN` (not `GITHUB_TOKEN`) so the lockfile commit re-triggers CI. Dependabot runs read the **Dependabot** secret store, so a PAT present only under Actions secrets is empty there.

| Condition | Result |
|-----------|--------|
| PAT non-empty | Full regen of both locks + one signed push (`[dependabot skip] Update requirements.lock and requirements-cpu.lock`) |
| PAT empty + Dependabot actor | Green no-op with `::notice::` — Lockfile Freshness still enforces |
| PAT empty + other actor | Hard fail (`::error::`) — secret misconfiguration |

Register the same PAT under **Settings → Secrets → Dependabot** to restore Dependabot auto-regen without editing the workflow (cascor #428; canopy #476).

### Compile flags

Regen (`.github/workflows/lockfile-update.yml`) compiles `pyproject.toml` to a fresh path, moves that file onto `requirements.lock`, then derives `requirements-cpu.lock` with `--constraint requirements.lock`.
The fresh path is the upgrade: uv treats an existing `-o requirements.lock` as extra constraints and keeps the old pins.
`--no-emit-package torch` leaves torch out of both locks.
The CPU step refuses when `ARG TORCH_VERSION` disagrees with the CPU lock header, and it splices that header back after uv overwrites it.
One `createCommitOnBranch` commit carries both files.

```bash
# Locally, from the repo root: the fresh path lives in the gitignored tmp/, never /tmp.
uv pip compile pyproject.toml \
  --extra ml --extra api --extra observability --extra juniper-data \
  --index-strategy unsafe-best-match --no-emit-package torch \
  -o tmp/requirements.lock.check
mv tmp/requirements.lock.check requirements.lock
```

Freshness (`ci.yml` `lockfile-check`) recompiles with `--constraint requirements.lock` and diffs `pkg==version` pin lines (ignores uv header / `-c` annotations). Newer PyPI versions alone do not fail that gate. The CPU half of the same job checks that every image dependency **name** is present in `requirements-cpu.lock`. It does not compare versions to `requirements.lock`, and neither half reads the conf freezes.

The skip commit can therefore move transitive pins past the conf lines Dependabot just wrote, in the same PR.
On [#701](https://github.com/pcalnon/juniper-cascor/pull/701), conf freezes landed `filelock==4.0.9` while both locks resolved `4.0.11`, and `websockets` stayed `17.1` in the conf freezes while both locks moved to `17.2`.
`torch==2.14.1` in the conf freezes left `ARG TORCH_VERSION=2.14.0` in place.
Full recipe and the three-row table: [Regenerating both locks](../../notes/DEPENDENCY_UPDATE_WORKFLOW.md#regenerating-both-locks).

> Operator narrative: [Dependency Update Workflow](../../notes/DEPENDENCY_UPDATE_WORKFLOW.md)

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Tests timing out | Missing `--timeout` flag | Ensure timeout is set in pytest command |
| Coverage not generating | Wrong working directory | Run from `src/tests/` |
| Conda env not found | Environment file path wrong | Check `conf/conda_environment.yaml` exists |
| Artifacts missing | Step failed before upload | Check for earlier step failures |
| Golden tests all skipped | Missing `--golden` (and/or `--slow --integration`) | Pass the three flags; marker alone is not enough |
| Conformance tests all skipped | Missing `--conformance` | Same pattern as golden; see `conftest.py` collection gates |
| Golden float mismatches locally | Wrong Python/torch or multi-thread BLAS / xdist | Use 3.13 + torch 2.11.0, single-thread env, serial pytest |
| Conformance fails after model-core bump | Interface drift in `CascorModel` adapter hooks | Fix production `CascorModel` / factory hooks — do not weaken the kit |
| Package CI did not run | Path filter missed the change | Edit under `juniper-cascor-protocol/` or `juniper-cascor-model/`, or use `workflow_dispatch` |
| Lockfile Freshness red on Dependabot PR | PAT gate green no-op (Dependabot secret store) | Register `CROSS_REPO_DISPATCH_TOKEN` under Dependabot secrets, or commit a signed regen of both locks |
| Lock moved a package Dependabot's table omitted | Regen re-resolves `pyproject.toml` onto a fresh file; conf freezes are a separate edit | Expected on a PAT-enabled run. Align conf freezes when they must match; conda yaml is a third freeze |
| Conf `torch` pin moved; image still on the old version | Torch is omitted from both locks. Image pin is `ARG TORCH_VERSION` plus the CPU lock header | Bump the Dockerfile ARG and the header together, then re-derive `requirements-cpu.lock` |
| CodeQL missing on a `develop` PR | `pull_request` filter is `main` only | Push the branch to `develop`, retarget the PR at `main`, or wait for the Monday cron on `main`/`develop` |
| CodeQL action pins drifted across files | Partial merge of a `codeql-action` group PR | Keep `init` / `autobuild` / `analyze` / `ci.yml` `upload-sarif` on the same SHA |
| Bandit findings missing from the Security tab | The `upload-sarif` step failed. It is `continue-on-error` on a push or dispatch, so **Security Scans** stayed green; on a PR it fails the job, since the required `Bandit` context would otherwise never post | Read the upload step log. The blocking Bandit step still judged the findings |
| No way to re-run CodeQL from Actions | `codeql.yml` has no `workflow_dispatch` | Push an empty commit, open/sync a PR against `main`, or wait for Monday 06:00 UTC |
| `@claude` comment, job green, no reply | Workflow `if` matched a substring; the action required a bounded phrase, or the event was `issues` `assigned` | Put `@claude` as its own word in a **new** comment, a **submitted** review, or a newly **opened** issue. See [Claude Code Workflow](#claude-code-workflow) |
| Claude Code job red: `Actor does not have write permissions to the repository` | Commenter is below `write`, or is an app account whose login lacks the `[bot]` suffix | A user with `admin` or `write` has to be the actor. This workflow sets no `allowed_non_write_users` |
| Claude Code job red: `Workflow initiated by non-human actor:` | `allowed_bots` is empty | A human has to comment. This workflow does not pass `allowed_bots` |
| Claude Code job red: `Environment variable validation failed:` and `ANTHROPIC_API_KEY` | Secret missing or blank on a real trigger | Repository must be able to read `ANTHROPIC_API_KEY` (the workflow header calls it an org-level secret) |
| Update Lockfile hard-fails on human PR | Actions PAT missing/expired | Restore Actions secret or commit lock manually |
| Publish skipped for wrong package | Tag prefix does not match workflow guard | Use `v*` / `juniper-cascor-protocol-v*` / `juniper-cascor-model-v*` |
| TestPyPI 400 already exists | Concurrent publish or retry of same version | Bump version; avoid dual `release`+`push: tags` triggers |
| OIDC publish auth failure | Trusted publisher not registered for workflow/env | Configure pending publisher on TestPyPI and PyPI |
| Action pin drift across publish workflows | Partial Dependabot merge | Keep the same `gh-action-pypi-publish` SHA in all three YAML files |
| A Twine freeze bump does not change upload behavior | Upload Twine is action-bundled; check jobs install Twine unpinned | See [Twine Pin Surfaces](#twine-pin-surfaces); only bumping the action SHA changes upload Twine |
| `twine check` rejects Metadata-Version 2.0 | Twine ≥ 7 removed the 2.0 monkeypatch | Rebuild with current setuptools / PEP 621; upgrade local `packaging` to ≥ 26.1 |

### Debugging Workflows

1. **View logs**: Click on failed job in Actions tab
2. **Re-run with debug**: Use "Re-run jobs" → "Enable debug logging"
3. **Local testing**: Use `act` to run workflows locally

```bash
# Install act
brew install act  # macOS
# or
curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run workflow locally
act -j test
```

---

## Related Documentation

- [Testing Guide](../testing/QUICK_START.md)
- [Installation Guide](../install/QUICK_START.md)
- [API Reference](../api/API_REFERENCE.md)
- [CodeQL Analysis](#codeql-analysis)
- [Secret Scan (Gitleaks)](#secret-scan-gitleaks)
- [Claude Code Workflow](#claude-code-workflow)
- Test Runner Scripts (`src/tests/scripts/`)
