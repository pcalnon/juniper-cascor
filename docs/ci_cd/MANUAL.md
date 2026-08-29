# CI/CD Manual

**Project**: Juniper Cascor  
**Version**: 0.3.17  
**Reference**: CASCOR-P1-007

---

## Pipeline Architecture

### Workflow File Location

```
.github/workflows/ci.yml
```

### Trigger Events

| Event           | Branches                                  |
| --------------- | ----------------------------------------- |
| `push`          | `main`, `develop`, `feature/**`, `fix/**` |
| `pull_request`  | `main`, `develop`                         |

### Job Dependency Graph

```
┌────────┐     ┌────────┐
│  lint  │     │  test  │
└────┬───┘     └────┬───┘
     │              │
     │              ├──────────────┐
     │              │              │
     │              ▼              │
     │       ┌─────────────┐       │
     │       │ integration │       │
     │       │  (PR only)  │       │
     │       └─────────────┘       │
     │              │              │
     ▼              ▼              │
┌──────────────────────────────────┘
│
▼
┌──────────────┐
│ quality-gate │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    notify    │
└──────────────┘
```

---

## Job Details

### Lint Job

**Name**: Code Quality Checks  
**Runs on**: `ubuntu-latest`  
**Python**: 3.14

| Tool    | Purpose           | Continue on Error |
| ------- | ----------------- | ----------------- |
| Black   | Format check      | Yes               |
| isort   | Import sort check | Yes               |
| Flake8  | Linting           | Yes               |
| MyPy    | Type checking     | Yes               |

**Flake8 Configuration**:

- Max line length: 512
- Max complexity: 15
- Ignored: E203, E266, E501, W503
- Exit zero (warnings only)

### Test Job

**Name**: Test Suite  
**Runs on**: `ubuntu-latest`  
**Python Matrix**: 3.14  
**Conda Environment**: JuniperCascor

| Setting             | Value                           |
| ------------------- | ------------------------------- |
| Timeout per test    | 60 seconds                      |
| Test markers        | `unit and not slow`             |
| Max failures        | 10                              |
| Coverage modules    | `cascade_correlation`, `candidate_unit` |

**Steps**:

1. Checkout code
2. Set up Conda environment
3. Install test dependencies (`pytest`, `pytest-cov`, `pytest-timeout`, `pytest-xdist`)
4. Verify test files and create directories
5. Run unit tests (fast only)
6. Upload coverage report
7. Upload test results
8. Check coverage thresholds
9. Generate test summary

### Integration Job

**Name**: Integration Tests  
**Runs on**: `ubuntu-latest`  
**Depends on**: `test`  
**Condition**: Pull request only (`github.event_name == 'pull_request'`)

| Setting          | Value                        |
| ---------------- | ---------------------------- |
| Timeout per test | 120 seconds                  |
| Test markers     | `integration and not slow`   |
| Max failures     | 5                            |

### Quality Gate

**Name**: Quality Gate  
**Depends on**: `lint`, `test`  
**Runs**: Always (`if: always()`)

**Enforcement Rules**:

| Condition      | Result                              |
| -------------- | ----------------------------------- |
| Test failure   | **FAIL** - Exit 1                   |
| Lint failure   | **WARN** - Pipeline continues       |
| Both pass      | **PASS** - Quality gate passed      |

### Notify Job

**Name**: Notification  
**Depends on**: `quality-gate`  
**Runs**: Always

Outputs build status including workflow name, branch, commit SHA, and actor.

---

## Coverage Handling

### Coverage Report Generation

Coverage is generated during the test job:

```bash
python -m pytest \
  -m "unit and not slow" \
  src/tests/unit \
  --verbose \
  --timeout=60 \
  --maxfail=5 \
  --cov=src \
  --cov-report=term-missing \
  --cov-report=xml:reports/coverage.xml \
  --cov-report=html:reports/htmlcov
```

**Output Formats**:

| Format        | Location                          |
| ------------- | --------------------------------- |
| Terminal      | Console output with missing lines |
| XML (Cobertura) | `src/tests/reports/coverage.xml`  |
| HTML          | `src/tests/reports/htmlcov/`      |

### Artifact Upload

| Artifact                 | Retention   | Contents                         |
| ------------------------ | ----------- | -------------------------------- |
| `coverage-report-{ver}`  | 30 days     | `coverage.xml`, `htmlcov/`       |
| `test-results-{ver}`     | 30 days     | `junit.xml`                      |

Artifacts are uploaded via `actions/upload-artifact@v4` and are available even if tests fail (`if: always()`).

### Coverage Threshold

**Threshold**: 80% aggregate (hard fail)  
**Reference**: P2-NEW-002

```bash
python -m coverage report --fail-under="${COVERAGE_FAIL_UNDER:-80}"
```

- If aggregate coverage falls below the threshold, the unit-tests job fails.
- Per-file >=90% and pooled sub-module >=95% statement bars are measured during the rollout with `juniper-coverage-gap-map`; the blocking `--enforce` step lands after all modules clear.
- Increase `COVERAGE_FAIL_UNDER` deliberately as aggregate coverage improves.

---

## Slow Test Handling

### Why Slow Tests Are Excluded

Slow tests (marked with `@pytest.mark.slow`) involve full neural network training cycles that can take 2-5+ minutes per test. Including them in the default CI run would:

- Exceed the 60-second timeout
- Significantly increase pipeline duration
- Risk GitHub Actions timeout limits

### Test Marker Exclusion

Both unit and integration test runs exclude slow tests:

```bash
# Unit tests
-m "unit and not slow"

# Integration tests
-m "integration and not slow"
```

### Running Slow Tests Separately

**Locally**:

```bash
cd src/tests

# Run only slow tests
python -m pytest -m "slow" --timeout=300 -v

# Run all tests including slow
python -m pytest --timeout=300 -v
```

**Using the test runner script**:

```bash
cd src/tests && bash scripts/run_tests.bash -m "slow"
```

### CASCOR-TIMEOUT-001 Resolution

The exclusion of slow tests is documented inline with reference `CASCOR-TIMEOUT-001`. Slow tests require:

- Extended timeout (300 seconds per test)
- Dedicated test runs outside the main CI pipeline
- Manual or scheduled execution for full coverage

---

## Modifying the Pipeline

### Adding New Jobs

1. Define the job in `.github/workflows/ci.yml`:

```yaml
new-job:
  name: New Job Name
  runs-on: ubuntu-latest
  needs: [test]  # Optional dependencies

  steps:
    - name: Checkout Code
      uses: actions/checkout@v6

    - name: Your Step
      run: |
        echo "Running new job..."
```

2. Add to `quality-gate` needs if it should block merges:

```yaml
quality-gate:
  needs: [lint, test, new-job]
```

### Changing Test Markers

Modify the `-m` flag in the pytest command:

```yaml
# Current (fast tests only)
-m "unit and not slow"

# Include slow tests
-m "unit"

# Run specific category
-m "unit and correlation"

# Exclude multiple markers
-m "unit and not slow and not gpu"
```

**Available markers** (from `src/tests/pytest.ini`):

- `unit`, `integration`, `performance`, `slow`
- `gpu`, `multiprocessing`, `spiral`
- `correlation`, `network_growth`, `candidate_training`
- `validation`, `accuracy`, `early_stopping`

### Adjusting Timeouts

**Per-test timeout** (in pytest command):

```yaml
# Unit tests: 60s → 120s
--timeout=120

# Integration tests: 120s → 300s
--timeout=300
```

**GitHub Actions job timeout** (add to job definition):

```yaml
test:
  name: Test Suite
  runs-on: ubuntu-latest
  timeout-minutes: 30  # Default is 360 minutes
```

### Adjusting Coverage Thresholds

**Increase the aggregate threshold** in `.github/workflows/ci.yml`:

```yaml
env:
  COVERAGE_FAIL_UNDER: "85"
```

The gate is already strict. Reproduce the same sequence locally from the repository root:

```bash
bash util/run_coverage.bash
```

**Add branch coverage requirement**:

```yaml
python -m pytest src/tests/unit \
  -m "unit and not slow" \
  --cov=src \
  --cov-branch \
  --cov-report=term-missing
```

---

## Lockfile Update Workflow

**Workflow:** `.github/workflows/lockfile-update.yml`  
**Enforcement:** `lockfile-check` / "Lockfile Freshness" in `ci.yml` (required by the quality gate)

### Intent

Keep Docker/CI pins (`requirements.lock`) aligned with `pyproject.toml` when Dependabot or a human bumps dependency ranges, without relying on `GITHUB_TOKEN` (which would not re-trigger CI on the lock commit).

### When it runs

1. **Push** to `dependabot/pip/**` by `dependabot[bot]`
2. **Pull request** that changes `pyproject.toml` on a branch in this repository (forks skipped — they cannot push with the PAT)

### PAT availability gate

Secrets are not readable in job `if:` expressions, so the first step exports `HAVE_PAT` from `secrets.CROSS_REPO_DISPATCH_TOKEN != ''` and branches in shell:

| Actor / secret | Outcome |
|----------------|---------|
| PAT present | Checkout with PAT → `uv pip compile` → commit/push if dirty |
| Dependabot + empty PAT | `::notice::` + skip remaining steps (exit 0) |
| Anyone else + empty PAT | `::error::` + exit 1 |

**Why Dependabot sees an empty PAT:** Dependabot-triggered workflow runs use the Dependabot secret store. Registering the token only under Actions secrets leaves Dependabot runs empty (historical hard-fail on run 30346680261 / #426; fixed by #428). Optional: duplicate the PAT under Dependabot secrets to restore auto-regen.

### Operator recovery when auto-regen no-ops

```bash
uv pip compile pyproject.toml \
  --extra ml --extra api --extra observability --extra juniper-data \
  --index-strategy unsafe-best-match --no-emit-package torch \
  --upgrade -o requirements.lock
git add requirements.lock
git commit -m "[dependabot skip] Update requirements.lock"
git push
```

> Narrative + troubleshooting matrix: [notes/DEPENDENCY_UPDATE_WORKFLOW.md](../../notes/DEPENDENCY_UPDATE_WORKFLOW.md)

---

## CodeQL Analysis

**Workflow:** `.github/workflows/codeql.yml`  
**Enforcement:** Soak / advisory — **not** a required status check (the same convention `sequence-safety.yml` documents). Findings land in the repository **Security → Code scanning** tab.

### Intent

Run GitHub CodeQL semantic SAST on the Python tree so query-pack findings (security **and** quality: `queries: +security-and-quality`) sit beside Bandit/Gitleaks/pip-audit rather than replacing them. The workflow file is the fleet Python template; its header names `juniper-data/.github/workflows/codeql.yml` as the copy origin. First-run comment in the YAML: treat the initial cycle as a shakedown, then an owner may promote it to a required context — this repo has **not** done that.

### When it runs

| Event | Filter |
|-------|--------|
| `push` | `main`, `develop` |
| `pull_request` | `main` **only** |
| `schedule` | Monday 06:00 UTC (`0 6 * * 1`) — same cron as `security-scan.yml` |

There is **no** `workflow_dispatch`. To force a run: push to `main`/`develop`, open or synchronize a PR targeting `main`, or wait for the weekly cron.

### What it does

1. Checkout (SHA-pinned `actions/checkout`)
2. `github/codeql-action/init` with `languages: python` and `queries: +security-and-quality`
3. `github/codeql-action/autobuild` (Python has no compile step; this is the template default)
4. `github/codeql-action/analyze` with `category: /language:python`

Job `analyze` uses `fail-fast: false` and `security-events: write`. Results do not feed `ci.yml`'s Quality Gate.

### Pins and Dependabot

Dependabot's `github-actions` ecosystem groups `github/codeql-action*` (`.github/dependabot.yml`). One PR bumps all four uses together:

- `codeql.yml`: `init`, `autobuild`, `analyze`
- `ci.yml` Security Scans: `upload-sarif` for Bandit (`reports/security/bandit.sarif`)

Keep those four SHAs aligned. Do not pin a patch version in docs — the group PR updates the SHA and the `# vX.Y.Z` comment.

The Bandit upload is **best-effort**: `if: always()` and `continue-on-error: true`. A failed SARIF upload does not fail Security Scans; the separate Bandit invocation with `--confidence-level medium --severity-level medium` is the blocking check.

### Contrast

| Workflow | Gitleaks | Bandit | pip-audit | CodeQL |
|----------|----------|--------|-----------|--------|
| `ci.yml` `security` | Yes (skipped on `repository_dispatch`) | Blocking medium+ + SARIF upload | `--strict` + documented torch `--ignore-vuln` list | No (only `upload-sarif`) |
| `security-scan.yml` | No | Blocking medium+ (artifact only) | `--strict` (no ignore list in this workflow) | No |
| `codeql.yml` | No | No | No | Yes |

### Operator pitfalls

- A PR whose base is `develop` never starts CodeQL. Retarget at `main` or push the commits onto `develop`/`main`.
- Monday 06:00 UTC also starts `security-scan.yml`. Overlapping Bandit + CodeQL load is expected, not a stuck runner.
- `security-scan.yml` can be dispatched by hand; CodeQL cannot. Do not look for a "Run workflow" button on **CodeQL Analysis**.
- Monday `pip-audit --strict` in `security-scan.yml` has **no** torch `--ignore-vuln` list. A red scheduled scan can be an advisory `ci.yml` already ignores — compare the two commands before treating it as a new CVE.

> Contract tables: [CI Reference — CodeQL Analysis](REFERENCE.md#codeql-analysis)

---

## Environment Setup

### Conda Environment

The pipeline uses Mamba for faster environment setup:

```yaml
- name: Set up Conda
  uses: conda-incubator/setup-miniconda@v3
  with:
    python-version: ${{ matrix.python-version }}
    channels: conda-forge,pytorch,nvidia
    miniforge-version: latest
    use-mamba: true
    activate-environment: JuniperCascor
    environment-file: conf/conda_environment.yaml
```

### Required Directories

The pipeline creates these directories before running tests:

```bash
mkdir -p logs src/logs reports/junit src/tests/reports
```

---

## WS-6 Gates (Golden + Conformance)

### Intent

Before (and during) the WS-6 refactor that repoints cascor onto
`juniper-service-core` / `juniper-model-core`, two **serial** CI lanes freeze
observable behavior and the GrowableModel interface contract. A refactor that
cannot keep both green without an intentional, reviewed golden update fails the
WS-6 kill-criterion.

| Half | Workflow | Roadmap | What it freezes |
|------|----------|---------|-----------------|
| OUT-12 | `golden-regression.yml` | Golden / snapshot regression | Training trajectory, predict-after-load, scrubbed API response shapes |
| OUT-13 | `conformance.yml` | model-core conformance | Shapes / keys / event order via `juniper_model_core.conformance.GrowableModelConformance` |

### Determinism contract (both lanes)

Mirrored in the workflow YAML job `env:` and required for local reproduction:

| Constraint | Value | Why |
|------------|-------|-----|
| Parallelism | **Serial only** — never `-n` / pytest-xdist | Parallel candidate reduction is not bit-stable |
| `CASCOR_NUM_PROCESSES` | `1` | Forces sequential candidate path |
| BLAS threads | `OMP` / `MKL` / `OPENBLAS` / `VECLIB` / `NUMEXPR` = `1` | Set at job level **before** Python loads native libs |
| Python / torch | **3.13** / **2.11.0** (CPU wheel) | Calibration environment; CPU kernels match `+cu` of the same torch version for these tensors |
| Interpreter | GIL required | `conftest` aborts under `Py_GIL_DISABLED` |
| Collection gates | `--golden` or `--conformance` **plus** `--slow --integration` | Markers alone are insufficient — opt-in flags prevent leakage into unit / integration / scheduled-slow |

Unlike goldens, conformance asserts **interface** (shapes / keys / event order),
not float values — so it carries no cross-build tolerance risk. Goldens use
tolerance for floats and exact compare for discrete/structural signals
(`src/tests/golden_support.py`; see `src/tests/fixtures/golden/README.md`).

### CI shape

```
push/PR/workflow_dispatch
        │
        ├─► golden-regression.yml
        │     pip install torch==2.11.0 (CPU) then -e ".[all]"
        │     pytest -m golden --golden --slow --integration src/tests/integration
        │     artifact: golden-regression-results (JUnit, 30d)
        │
        └─► conformance.yml
              pip install torch==2.11.0 (CPU) then -e ".[all]"
              pytest -m conformance --conformance --slow --integration src/tests/conformance
              artifact: conformance-results (JUnit, 30d)
```

Concurrency groups: `golden-${{ github.ref }}` / `conformance-${{ github.ref }}`
with `cancel-in-progress: true`. Permissions: `contents: read` only.

### Operator constraints

1. **Do not add xdist** to either workflow. Serial is the safety property.
2. **Do not drop the opt-in flags.** Without `--golden` / `--conformance`,
   `pytest_collection_modifyitems` skips those markers even when `--slow` and
   `--integration` are present (`src/tests/conftest.py`).
3. **Pin torch before editable install** so `pip install -e ".[all]"` does not
   upgrade away from `2.11.0`.
4. **Regenerate goldens only after intentional behavior change** — set
   `GOLDEN_CAPTURE=1`, review the diff, then re-run without capture. See
   `src/tests/fixtures/golden/README.md`.
5. Conformance skips the kit's bit-exact serialization check (D-C4); predict-
   after-load coverage stays in the golden lane at `allclose` tolerance.

### Package path-filtered CI

| Workflow | Working directory | Matrix | Notes |
|----------|-------------------|--------|-------|
| `ci-protocol.yml` | `juniper-cascor-protocol/` | Python 3.12, 3.13 | 95% package coverage + `juniper-coverage-gap-map --enforce`; then build/`twine check` |
| `ci-cascor-model.yml` | `juniper-cascor-model/` | Python 3.12 | CPU torch + `[test,full]`; coverage includes drift-guard; per-file gate via `juniper-ci-tools` |

Both trigger on `paths:` for their package tree (and their workflow file) plus
`workflow_dispatch`. They do **not** replace server `ci.yml`.

## PyPI Publishing

### Intent

Release packages from this monorepo via Trusted Publishing (OIDC) — no long-lived PyPI API tokens in GitHub secrets. Each package has its own workflow so a sub-package cut does not republish the main app (and vice versa).

### Workflows and Tag Guards

| Workflow | Package / working dir | Runs when release tag… | Manual re-fire |
|----------|----------------------|------------------------|----------------|
| `publish.yml` | `juniper-cascor` (repo root) | `startsWith(..., 'v')` — e.g. `v0.7.0` | No (`release` only) |
| `publish-protocol.yml` | `juniper-cascor-protocol/` | `juniper-cascor-protocol-v*` | Yes (`workflow_dispatch`) |
| `publish-cascor-model.yml` | `juniper-cascor-model/` | `juniper-cascor-model-v*` | Yes (`workflow_dispatch`) |

All three listen to `release: types: [published]`. A single Release fires every workflow file; the `if:` tag-prefix guards skip jobs that are not for that package. Sub-package workflows also set `concurrency.group: publish-<pkg>-${{ github.ref_name }}` with `cancel-in-progress: false` so a manual dispatch cannot race a live release onto immutable TestPyPI.

### Pipeline Shape

```
Release published (matching tag)
        │
        ▼
   build + twine check
        │
        ▼
   Publish to TestPyPI  (environment: testpypi, id-token: write)
        │
        ▼
   Verify install from TestPyPI ONLY
   (--no-deps, --index-url https://test.pypi.org/simple/)
        │
        ▼
   Publish to PyPI      (environment: pypi, needs: testpypi)
```

**Anti-target-squatting verify** (release-train Phase 0.2 / juniper-ml#384): never add `--extra-index-url https://pypi.org/simple/` to the TestPyPI install step. Under `--no-deps`, a pypi.org fallback could only resolve a squatted same-name *target* package. The main package and `juniper-cascor-model` assert via a torch-free `__version__` import; `juniper-cascor-protocol` asserts via `importlib.metadata` because importing the package pulls the numpy BinaryFrame codec (absent under `--no-deps`).

### Operator Constraints

1. **Publish by cutting a GitHub Release**, not `git push <tag>`. Sub-package workflows intentionally omit `push: tags` — dual triggers double-fired concurrent uploads that raced TestPyPI (juniper-ml#555; 400 "file already exists").
2. **Match the tag prefix** to the package. A `v*` Release skips protocol/model builds; a `juniper-cascor-protocol-v*` Release skips `publish.yml`'s `startsWith('v')` jobs.
3. **OIDC Trusted Publishing** must be configured on both TestPyPI and PyPI for each project (`permissions.id-token: write`; GitHub Environments `testpypi` / `pypi`).
4. **Keep `pypa/gh-action-pypi-publish` SHA-pinned** (trailing `# vX.Y.Z` comment). Dependabot opens bumps across all three workflows together. Prefer staying current: the action bundles Twine/sigstore for the upload step, and older pins have hit short-lived GitHub OIDC token lifetimes (~5 minutes) on large multi-wheel publishes.
5. **Treat Twine as three independent surfaces** (see below). A Dependabot bump of `conf/requirements_ci.txt` does **not** pin the publish jobs' `twine check` or the action's upload Twine.

### Twine Pin Surfaces

| Surface | Where | How Twine is selected | What a Dependabot Twine major bump changes |
|---------|-------|----------------------|---------------------------------------------|
| CI freeze (pip) | `conf/requirements_ci.txt` | Exact pin (currently `twine==7.0.0`) in the generated freeze | Yes — the Dependabot pip ecosystem can open a major bump here |
| CI freeze (conda) | `conf/conda_environment_ci.yaml` | Exact pin under `pip:` (currently `twine==6.2.0`) | Often lags the pip freeze until the next `juniper-generate-dep-docs` commit |
| Publish / package CI `twine check` | `publish.yml`, `publish-protocol.yml`, `publish-cascor-model.yml`, `ci-protocol.yml`, `ci-cascor-model.yml` | Unpinned `pip install build twine` or `pip install --upgrade build twine` at job time | No — those jobs resolve whatever Twine is current on PyPI |
| Upload Twine | `pypa/gh-action-pypi-publish` (SHA-pinned) | Bundled inside the action | No — only an action SHA bump changes the uploader |

**Intent:** keep local/CI metadata validation honest without implying that freezing Twine in `requirements_ci.txt` controls release uploads.

**Twine 7.0.0 operator notes** (verify against the [twine changelog](https://github.com/pypa/twine/blob/main/docs/changelog.rst) before merging a major bump):

- **Metadata 2.0 rejected.** Twine 7 drops the Metadata-Version 2.0 monkeypatch (never an official standard) while fixing Metadata 2.5 uploads. This repo's packages are PEP 621 + setuptools (`pyproject.toml` build backends) and emit modern metadata — still run `python -m build && twine check dist/*` under Twine ≥ 7 before the first post-bump Release.
- **`packaging >= 26.1`.** Twine 7 raises its packaging floor. Prefer a current `packaging` in any environment where you run a local `twine check` (old parsers can also mis-read `license-file`).
- **UTF-8 `.pypirc` reads / richer `--version` / non-standard HTTP status handling.** Relevant for manual uploads and verbose CI logs; Trusted Publishing paths do not use `.pypirc`.

**Review checklist for a `twine` major Dependabot PR:**

1. Confirm the diff is limited to the CI freeze (`conf/requirements_ci.txt`) and any lockfile collateral — Twine is not a runtime `pyproject.toml` dependency.
2. Expect `conf/conda_environment_ci.yaml` to still show the previous pin until the dependency docs are regenerated; do not treat that lag as a publish blocker.
3. Smoke-check metadata under Twine 7 locally (or rely on the next `ci-protocol` / `ci-cascor-model` / publish build job, which already installs current Twine unpinned).
4. Remember that upload Twine still tracks the SHA-pinned `gh-action-pypi-publish` action, not this freeze.

### Re-firing a Failed Sub-Package Publish

```bash
# After fixing a transient TestPyPI/PyPI failure for protocol or model:
gh workflow run publish-protocol.yml
# or
gh workflow run publish-cascor-model.yml
```

`publish.yml` (main package) has no `workflow_dispatch` — create a new Release or re-run the failed jobs from the Actions UI for that release run.

---

## Troubleshooting

### Test Timeout Failures

If tests timeout:

1. Check if slow tests are accidentally included
2. Increase `--timeout` value
3. Add `@pytest.mark.slow` to long-running tests

### WS-6 Golden / Conformance Failures

1. Confirm you reproduced with the calibration pins (Python 3.13, torch 2.11.0,
   single-thread BLAS, `CASCOR_NUM_PROCESSES=1`, GIL env) — not the main CI 3.14
   conda lane.
2. Confirm serial execution (no xdist worker count in the command or plugin).
3. For goldens: check whether an intentional behavior change needs
   `GOLDEN_CAPTURE=1` + reviewed fixture update (`src/tests/fixtures/golden/`).
4. For conformance: failures are interface/shape/order — compare against
   `GrowableModelConformance` expectations, not float noise.

### Coverage Report Missing

If coverage artifacts are empty:

1. Verify `--cov` paths are correct relative to test execution directory
2. Check that test files are found (`find src/tests -name "test_*.py"`)
3. Ensure `pytest-cov` is installed

### Conda Environment Failures

If environment setup fails:

1. Verify `conf/conda_environment.yaml` syntax
2. Check channel availability (conda-forge, pytorch, nvidia)
3. Review disk space (pipeline includes cleanup step)

### Lockfile Freshness / Update Lockfile

| Symptom | Cause | Fix |
|---------|-------|-----|
| Lockfile Freshness red; Update Lockfile green with no commit | Dependabot PAT gate no-op | Register `CROSS_REPO_DISPATCH_TOKEN` under Dependabot secrets, or push a local regen |
| Update Lockfile fails on a human `pyproject.toml` PR | Actions PAT missing | Restore Actions secret or commit the lock in the PR |
| Freshness fails after a range bump | Lock cannot satisfy new mins | Regen with `--upgrade` (see compile command above) |

### Publish Failures

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| TestPyPI `400 File already exists` on a fresh Release | Dual trigger (`release` + `push: tags`) or concurrent `workflow_dispatch` | Keep a single `release: published` trigger; rely on concurrency groups; bump version for a new upload |
| OIDC / trusted publisher errors | Missing PyPI pending publisher or wrong workflow/env name | Register the workflow + environment on TestPyPI/PyPI; ensure `id-token: write` |
| TestPyPI verify can't find the version | Index lag | Protocol/model retry 5× with 10s sleep; main package sleeps 30s once — re-run the verify step if lag exceeds that |
| Protocol verify import fails under `--no-deps` | Expecting `import juniper_cascor_protocol` | Use `importlib.metadata` version check (workflow already does) |
| Wrong package published / skipped | Tag prefix mismatch | Use `v*` for the app, `juniper-cascor-<pkg>-v*` for sub-packages |
| `twine check` fails after merging a Twine major in `requirements_ci.txt` | The local/CI freeze Twine is not what publish uses; or the Metadata-Version is too old for Twine 7 | Rebuild with current setuptools; run `twine check` under Twine ≥ 7; do not expect the freeze pin to change the action's upload Twine |
| CodeQL did not run on this PR | `codeql.yml` `pull_request` branches are `[main]` only | Retarget the PR at `main`, or push to `main`/`develop` |
| Cannot click **Run workflow** on CodeQL Analysis | No `workflow_dispatch` on `codeql.yml` | Push / PR-against-`main` / wait for Monday 06:00 UTC |
| Dependabot CodeQL PR touches `ci.yml` too | `upload-sarif` is in the same `codeql-action` group | Expected — review Bandit SARIF pin with the CodeQL pins |
| Security tab missing Bandit but Security Scans is green | `upload-sarif` `continue-on-error: true` | Check the upload step log; the blocking Bandit CLI still ran |
| `conda_environment_ci.yaml` still pins old Twine after a `requirements_ci.txt` bump | Generated freezes update on different cadences | Regenerate via the CI dependency-docs job (`juniper-generate-dep-docs`); publish jobs ignore both freezes |

---

## References

- **CASCOR-P1-007**: CI/CD Pipeline Setup
- **CASCOR-TIMEOUT-001**: Slow Test Exclusion
- **P2-NEW-002**: Coverage Thresholds in CI
- **juniper-ml#384 / #555**: TestPyPI verify policy and dual-trigger race
- **JuniperCanopy CI/CD**: Base workflow pattern
- **CodeQL soak**: `.github/workflows/codeql.yml` (not a required check)
