# CI/CD Reference

**Project**: Juniper Cascor  
**Workflow Files**: `.github/workflows/ci.yml`, `golden-regression.yml`, `conformance.yml`, `ci-protocol.yml`, `ci-cascor-model.yml`, `lockfile-update.yml`, `publish.yml`, `publish-protocol.yml`, `publish-cascor-model.yml`  
**Workflow Files**: `.github/workflows/ci.yml`

**Last Updated**: 2026-08-05

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
| `ci-protocol.yml` | `juniper-cascor-protocol/**` | 3.12, 3.13 | `--cov-fail-under=95` + `juniper-coverage-gap-map --enforce` |
| `ci-cascor-model.yml` | `juniper-cascor-model/**` | 3.12 | package + candidate/utils/log_config/constants; per-file enforce |

## Publish Workflows

Trusted Publishing (OIDC) pipelines. Source of truth: the workflow YAML files under `.github/workflows/`.

| Workflow | Package | Trigger | Tag guard | Environments | Publish action |
|----------|---------|---------|-----------|--------------|----------------|
| `publish.yml` | `juniper-cascor` | `release: published` | `startsWith(tag, 'v')` | `testpypi` → `pypi` | `pypa/gh-action-pypi-publish` (SHA-pinned) |
| `publish-protocol.yml` | `juniper-cascor-protocol` | `release` + `workflow_dispatch` | `juniper-cascor-protocol-v*` | `testpypi` → `pypi` | same pin |
| `publish-cascor-model.yml` | `juniper-cascor-model` | `release` + `workflow_dispatch` | `juniper-cascor-model-v*` | `testpypi` → `pypi` | same pin |

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
- **Matrix Testing**: Single version (expandable)

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
| PAT non-empty | Full regen + push (`[dependabot skip] Update requirements.lock`) |
| PAT empty + Dependabot actor | Green no-op with `::notice::` — Lockfile Freshness still enforces |
| PAT empty + other actor | Hard fail (`::error::`) — secret misconfiguration |

Register the same PAT under **Settings → Secrets → Dependabot** to restore Dependabot auto-regen without editing the workflow (cascor #428; canopy #476).

### Compile flags (regen and freshness)

```bash
uv pip compile pyproject.toml \
  --extra ml --extra api --extra observability --extra juniper-data \
  --index-strategy unsafe-best-match --no-emit-package torch \
  --upgrade -o requirements.lock
```

Freshness recompiles with `--constraint requirements.lock` and diffs `pkg==version` pin lines (ignores uv header / `-c` annotations). Newer PyPI versions alone do not fail the gate.

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
| Lockfile Freshness red on Dependabot PR | PAT gate green no-op (Dependabot secret store) | Register `CROSS_REPO_DISPATCH_TOKEN` under Dependabot secrets, or commit a local regen |
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
- Test Runner Scripts (`src/tests/scripts/`)
