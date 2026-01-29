# CI/CD Reference

**Project**: Juniper Cascor  
**Workflow File**: `.github/workflows/ci.yml`  
**Last Updated**: 2026-01-29

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

### Step-by-Step Breakdown

#### Lint Job

1. **Checkout Code** - Shallow clone (`fetch-depth: 1`)
2. **Set up Python** - Python 3.14
3. **Install Linting Tools** - black, isort, mypy, flake8 with plugins
4. **Run Black** - Format check (soft fail)
5. **Run isort** - Import sort check (soft fail)
6. **Run Flake8** - Linting with `--exit-zero` (soft fail)
7. **Run MyPy** - Type checking (soft fail)

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
10. **Check Coverage Thresholds** - 50% minimum (soft fail)
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
| **Threshold** | 50% |
| **Enforcement** | Soft fail (warning only) |
| **Command** | `python -m coverage report --fail-under=50` |

### Soft Fail Behavior

When coverage falls below the threshold:

- A GitHub Actions warning is emitted: `::warning::Coverage below 50% threshold`
- The step continues (`continue-on-error: true`)
- The quality gate does **not** fail the build
- Message: "Coverage gate: SOFT FAIL (warning only during initial phase)"

### How to Increase Thresholds

1. Edit `.github/workflows/ci.yml`
2. Locate the "Check Coverage Thresholds" step
3. Modify `--fail-under=50` to the desired value:

```yaml
python -m coverage report --fail-under=80
```

4. To enforce hard failures, remove `continue-on-error: true`

### Recommended Threshold Progression

| Phase | Threshold | Notes |
|-------|-----------|-------|
| Initial | 50% | Current setting |
| Stabilization | 70% | After core modules covered |
| Mature | 80% | Production-ready target |
| Comprehensive | 90%+ | Research codebase goal |

### Per-Module Thresholds

Per-module thresholds are not currently configured in CI. To add them, use a `.coveragerc` file:

```ini
[coverage:report]
fail_under = 80

[coverage:run]
source = cascade_correlation,candidate_unit
```

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
| Global pytest timeout | 60 seconds | `pytest.ini` |
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
![CI/CD Pipeline](https://github.com/{owner}/{repo}/actions/workflows/ci.yml/badge.svg)
```

For Juniper Cascor:

```markdown
![CI/CD Pipeline](https://github.com/pcalnon/juniper_cascor/actions/workflows/ci.yml/badge.svg)
```

### Branch-Specific Badges

```markdown
![CI/CD (main)](https://github.com/pcalnon/juniper_cascor/actions/workflows/ci.yml/badge.svg?branch=main)
![CI/CD (develop)](https://github.com/pcalnon/juniper_cascor/actions/workflows/ci.yml/badge.svg?branch=develop)
```

### How to Add to README

Add to the top of `README.md`:

```markdown
# Juniper Cascor

![CI/CD Pipeline](https://github.com/pcalnon/juniper_cascor/actions/workflows/ci.yml/badge.svg)
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

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Tests timing out | Missing `--timeout` flag | Ensure timeout is set in pytest command |
| Coverage not generating | Wrong working directory | Run from `src/tests/` |
| Conda env not found | Environment file path wrong | Check `conf/conda_environment.yaml` exists |
| Artifacts missing | Step failed before upload | Check for earlier step failures |

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

- [Testing Guide](../testing/README.md)
- [Installation Guide](../install/README.md)
- [API Reference](../api/README.md)
- [Test Runner Scripts](../../src/tests/scripts/README.md)
