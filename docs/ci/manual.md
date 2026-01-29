# CI/CD Manual

**Project**: Juniper Cascor  
**Version**: 0.3.16  
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

- Max line length: 120
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
python -m pytest unit/ \
  --cov=../cascade_correlation \
  --cov=../candidate_unit \
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

**Threshold**: 50% (soft fail)  
**Reference**: P2-NEW-002

```bash
python -m coverage report --fail-under=50
```

- If coverage falls below 50%, a warning is emitted
- Pipeline continues (soft fail during initial phase)
- Threshold to be increased as coverage improves

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

### Adding Coverage Thresholds

**Increase threshold** (in coverage check step):

```yaml
# Change from 50% to 70%
python -m coverage report --fail-under=70
```

**Make threshold enforcement strict**:

```yaml
- name: Check Coverage Thresholds
  shell: bash -el {0}
  run: |
    cd src/tests
    python -m coverage report --fail-under=70 || {
      echo "::error::Coverage below 70% threshold"
      exit 1  # Hard fail instead of warning
    }
  # Remove: continue-on-error: true
```

**Add branch coverage requirement**:

```yaml
python -m pytest unit/ \
  --cov-branch \
  --cov-fail-under=70
```

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

## Troubleshooting

### Test Timeout Failures

If tests timeout:

1. Check if slow tests are accidentally included
2. Increase `--timeout` value
3. Add `@pytest.mark.slow` to long-running tests

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

---

## References

- **CASCOR-P1-007**: CI/CD Pipeline Setup
- **CASCOR-TIMEOUT-001**: Slow Test Exclusion
- **P2-NEW-002**: Coverage Thresholds in CI
- **JuniperCanopy CI/CD**: Base workflow pattern
