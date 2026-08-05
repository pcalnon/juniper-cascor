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

---

## References

- **CASCOR-P1-007**: CI/CD Pipeline Setup
- **CASCOR-TIMEOUT-001**: Slow Test Exclusion
- **P2-NEW-002**: Coverage Thresholds in CI
- **JuniperCanopy CI/CD**: Base workflow pattern
