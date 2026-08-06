# Testing Reference

Complete reference documentation for the Juniper Cascor test suite.

---

## Marker Reference

### Complete Marker Table

| Marker               | Description                                | When to Use                                          | CI Behavior                                   |
|----------------------|--------------------------------------------|------------------------------------------------------|-----------------------------------------------|
| `unit`               | Unit tests for individual components       | Testing single functions/methods in isolation        | Runs on all pushes and PRs (excluding `slow`) |
| `integration`        | Integration tests for full workflows       | Testing component interactions and data flow         | Runs on PRs only (excluding `slow`)           |
| `performance`        | Performance and benchmarking tests         | Measuring execution time and resource usage          | Not run automatically in CI                   |
| `slow`               | Tests that take a long time to run         | Full training cycles, large datasets                 | **Excluded** from default CI runs             |
| `gpu`                | Tests that require GPU/CUDA                | GPU-accelerated operations, CUDA kernels             | Not run in CI (no GPU available)              |
| `multiprocessing`    | Tests that use multiprocessing             | Parallel candidate training, worker pools            | Runs with standard tests                      |
| `spiral`             | Spiral problem tests                       | Two-spiral classification validation                 | Runs with unit/integration tests              |
| `correlation`        | Correlation coefficient calculations       | Pearson correlation, covariance tests                | Runs with unit tests                          |
| `network_growth`     | Network growth algorithms                  | Hidden unit addition, architecture changes           | Runs with unit tests                          |
| `candidate_training` | Candidate unit training                    | Candidate weight updates, training loops             | Runs with unit tests                          |
| `validation`         | Input validation functions                 | Parameter checking, type validation                  | Runs with unit tests                          |
| `accuracy`           | Accuracy calculation methods               | Classification accuracy, metrics                     | Runs with unit tests                          |
| `early_stopping`     | Early stopping logic                       | Convergence detection, patience handling             | Runs with unit tests                          |

### Early-Stopping Regression Coverage (No Validation Data Path)

The no-validation branch in `CascadeCorrelationNetwork.validate_training()` is covered by targeted unit regressions in `src/tests/unit/test_cascade_correlation_coverage_extended.py`.

Key scenarios covered:

- Improvement path: `train_loss` improves by more than `convergence_threshold`, so `best_value_loss` is updated and `patience_counter` resets to `0`.
- Patience exhaustion path: non-improving `train_loss` increments `patience_counter` until `patience` is exhausted and `early_stop` becomes `True`.
- Cross-iteration propagation: `grow_network()` feeds updated `patience_counter` and `best_value_loss` from one `validate_training()` call into the next iteration's `ValidateTrainingInputs`.

Run only these regression tests:

```bash
cd src
python -m pytest tests/unit/test_cascade_correlation_coverage_extended.py -k "validate_training_without_validation_data or propagates_validation_state" -v
```

Why this matters:

- The no-validation path is used when `x_val`/`y_val` are omitted, which is common in lightweight training runs.
- Regressions here can silently disable or destabilize early stopping behavior even if validation-based tests still pass.

### Marker Combinations

```bash
# Run unit tests only
pytest -m "unit"

# Run unit tests excluding slow
pytest -m "unit and not slow"

# Run spiral-related unit tests
pytest -m "unit and spiral"

# Run correlation or accuracy tests
pytest -m "correlation or accuracy"

# Run everything except GPU and slow
pytest -m "not gpu and not slow"
```

---

## Report Locations

All test reports are generated in `src/tests/reports/`:

| Report Type      | Location                              | Description                              |
|------------------|---------------------------------------|------------------------------------------|
| HTML Coverage    | `reports/htmlcov/index.html`          | Interactive HTML coverage report         |
| XML Coverage     | `reports/coverage.xml`                | Cobertura-format XML for CI integration  |
| JUnit XML        | `reports/junit.xml`                   | JUnit-format test results for CI         |

### Accessing Reports

```bash
# Open HTML coverage report (Linux)
xdg-open src/tests/reports/htmlcov/index.html

# Open HTML coverage report (macOS)
open src/tests/reports/htmlcov/index.html

# View coverage summary in terminal
python -m coverage report
```

---

## Test Command Reference

### Direct pytest Commands

```bash
# Run all tests
cd src/tests && python -m pytest

# Run specific test file
python -m pytest unit/test_forward_pass.py -v

# Run tests matching name pattern
python -m pytest -k "test_accuracy" -v

# Run with specific markers
python -m pytest -m "unit and accuracy" -v

# Run with verbose output and short traceback
python -m pytest -v --tb=short

# Run with full traceback on failures
python -m pytest --tb=long

# Stop after first failure
python -m pytest -x

# Stop after N failures
python -m pytest --maxfail=3

# Run last failed tests only
python -m pytest --lf

# Run failed tests first, then rest
python -m pytest --ff

# Parallel execution (requires pytest-xdist)
python -m pytest -n auto

# Disable warnings
python -m pytest -p no:warnings
```

### run_tests.bash Script Options

```bash
cd src/tests/scripts

# Basic usage (unit tests with coverage)
bash run_tests.bash

# Script options
bash run_tests.bash -u              # Unit tests (default: true)
bash run_tests.bash -i              # Integration tests
bash run_tests.bash -p              # Performance tests
bash run_tests.bash -s              # Include slow tests
bash run_tests.bash -g              # Include GPU tests
bash run_tests.bash -v              # Verbose output
bash run_tests.bash -c              # With coverage (default: true)
bash run_tests.bash --no-coverage   # Disable coverage
bash run_tests.bash -j              # Parallel execution
bash run_tests.bash -f              # Re-run failed tests only
bash run_tests.bash -t FILE         # Run specific test file
bash run_tests.bash -m "MARKERS"    # Run with specific markers
bash run_tests.bash -o DIR          # Custom output directory
bash run_tests.bash -h              # Show help

# Combined options
bash run_tests.bash -v -c           # Verbose with coverage
bash run_tests.bash -i -s           # Integration and slow tests
bash run_tests.bash -u -j           # Unit tests in parallel
bash run_tests.bash -m "spiral"     # Spiral problem tests
```

### Coverage Commands

```bash
# Run the CI-parity aggregate gate from the repository root
bash util/run_coverage.bash

# Run with repository-wide coverage collection
python -m pytest -m "unit and not slow" src/tests/unit --cov=src

# Generate HTML report
python -m pytest -m "unit and not slow" src/tests/unit --cov=src --cov-report=html:src/tests/reports/htmlcov

# Generate XML report (for CI)
python -m pytest -m "unit and not slow" src/tests/unit --cov=src --cov-report=xml:src/tests/reports/coverage.xml

# Show missing lines in terminal
python -m pytest -m "unit and not slow" src/tests/unit --cov=src --cov-report=term-missing

# Multiple report formats
python -m pytest \
    -m "unit and not slow" \
    src/tests/unit \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html:src/tests/reports/htmlcov \
    --cov-report=xml:src/tests/reports/coverage.xml

# Check coverage threshold
python -m coverage report --fail-under=80

# View coverage for specific file
python -m coverage report --include="*/cascade_correlation.py"

# Produce per-file/sub-module gap-map input
python -m pytest -m "unit and not slow" src/tests/unit --cov=src --cov-report=json:src/tests/reports/coverage.json
juniper-coverage-gap-map --coverage-json src/tests/reports/coverage.json
```

---

## CI Test Matrix

### Trigger Events

| Event                      | Branches                               | Jobs Executed                           |
|----------------------------|----------------------------------------|----------------------------------------|
| Push                       | `main`, `develop`, `feature/**`, `fix/**` | lint, test, quality-gate, notify       |
| Pull Request               | `main`, `develop`                      | lint, test, **integration**, quality-gate, notify |

### Job Dependencies

```
lint ──────┐
           ├──> quality-gate ──> notify
test ──────┤
           │
           └──> integration (PR only)
```

### CI Marker Combinations

| Job             | Markers Used                      | Timeout | Notes                          |
|-----------------|-----------------------------------|---------|--------------------------------|
| Unit Tests      | `unit and not slow`               | 60s     | Runs on all pushes and PRs     |
| Integration     | `integration and not slow`        | 120s    | Runs on PRs only               |
| Slow Tests      | `slow`                            | 300s    | **Not run in CI** (manual)     |
| GPU Tests       | `gpu`                             | N/A     | **Not run in CI** (no GPU)     |

### CI Test Command (Unit Tests)

```bash
python -m pytest \
    -m "unit and not slow" \
    src/tests/unit \
    --verbose \
    --timeout=60 \
    --maxfail=5 \
    --junitxml=reports/junit/junit-unit.xml \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=xml:reports/coverage.xml \
    --cov-report=html:reports/htmlcov
```

### CI Test Command (Integration Tests)

```bash
python -m pytest integration/ \
    --verbose \
    --timeout=120 \
    -m "integration and not slow" \
    --maxfail=5
```

### Coverage Thresholds

| Threshold | Status | Action on Failure |
|-----------|--------|-------------------|
| 80% aggregate | Hard fail | `python -m coverage report --fail-under=${COVERAGE_FAIL_UNDER}` blocks the unit-tests job |
| >=90% per source file | Advisory until final gate PR | Measured with `juniper-coverage-gap-map` from coverage JSON |
| >=95% pooled per packaged sub-module | Advisory until final gate PR | Statement-weighted pooled coverage, not mean-of-files |

---

## Timeout Reference

### Default Timeouts

| Context              | Timeout  | Configuration Location            |
|----------------------|----------|-----------------------------------|
| Standard tests       | 60s      | `pyproject.toml` (`timeout = 60`) |
| CI unit tests        | 60s      | `.github/workflows/ci.yml`        |
| CI integration tests | 120s     | `.github/workflows/ci.yml`        |
| Slow tests           | 300s     | Per-test `@pytest.mark.timeout()` |

### Timeout Method

The project uses the `thread` timeout method (configured in
`pyproject.toml`). Thread-based timeout avoids SIGALRM interference
with forkserver IPC in multiprocessing-heavy performance tests:

```toml
timeout = 60
timeout_method = "thread"
```

### Overriding Timeouts

```bash
# Disable timeout for a single run
pytest --timeout=0

# Set custom timeout
pytest --timeout=120

# Timeout for slow tests (recommended approach)
pytest -m slow --timeout=300

# Disable global timeout when running slow tests
pytest -m slow --timeout=0
```

### Per-Test Timeout Decorator

```python
import pytest

# Set 300-second timeout for slow test
@pytest.mark.slow
@pytest.mark.timeout(300)
def test_full_training_cycle():
    ...

# Disable timeout for specific test
@pytest.mark.timeout(0)
def test_indefinite_operation():
    ...
```

---

## Exit Codes

### pytest Exit Codes

| Code | Name                  | Meaning                                              |
|------|-----------------------|------------------------------------------------------|
| 0    | `OK`                  | All tests passed                                     |
| 1    | `TESTS_FAILED`        | Some tests failed                                    |
| 2    | `INTERRUPTED`         | Test run interrupted by user (Ctrl+C)                |
| 3    | `INTERNAL_ERROR`      | Internal pytest error occurred                       |
| 4    | `USAGE_ERROR`         | pytest command line usage error                      |
| 5    | `NO_TESTS_COLLECTED`  | No tests were collected                              |

### Coverage Failure Conditions

| Condition                          | Exit Code | Trigger                                |
|------------------------------------|-----------|----------------------------------------|
| Coverage below threshold           | 2         | `--fail-under=N` with coverage < N%    |
| Missing coverage data              | 1         | `--cov` with no coverage collected     |
| Coverage file not found            | 1         | Specified source not found             |

### run_tests.bash Exit Codes

The script propagates pytest's exit code:

```bash
bash run_tests.bash
echo $?  # Shows pytest exit code
```

### Interpreting CI Results

| Job Result    | Meaning                                        | Action Required          |
|---------------|------------------------------------------------|--------------------------|
| ✓ Success     | All tests passed, quality gates met            | None                     |
| ✗ Failure     | Tests failed or quality gate failed            | Review failures          |
| ○ Skipped     | Job skipped (e.g., integration on push)        | None                     |
| ⊘ Cancelled   | Workflow cancelled                             | Re-run if needed         |

---

## Quick Reference Card

```bash
# Most common commands
cd src/tests/scripts
bash run_tests.bash              # Default: unit tests + coverage
bash run_tests.bash -v           # Verbose output
bash run_tests.bash -i           # Add integration tests
bash run_tests.bash -m "spiral"  # Run spiral tests only
bash run_tests.bash -f           # Re-run failed tests

# Direct pytest
cd src/tests
python -m pytest unit/ -v                    # Unit tests verbose
python -m pytest -m "unit and not slow" -v   # Fast unit tests
python -m pytest -k "accuracy" -v            # Tests matching "accuracy"
python -m pytest --lf                        # Last failed only

# Coverage from repository root
cd ../..
bash util/run_coverage.bash                 # CI-parity aggregate gate
python -m pytest src/tests/unit --cov=src --cov-report=term-missing
python -m coverage report --fail-under=80   # Current aggregate threshold
```

---

## Worker Round-Boundary Pending Clear (#471)

Regression coverage for the ISSUE-319-class early-unblock hole lands with open #471:

| Contract | Tests |
|----------|-------|
| `submit_tasks` clears prior-round `_pending_tasks` / `_unassigned_tasks` | `src/tests/unit/api/test_worker_coordinator.py` (`test_submit_tasks_clears_prior_round_pending`) |
| Late prior-round `submit_result` rejected after new round starts | `src/tests/unit/api/test_worker_coordinator.py` (`test_reject_stale_round_result_after_new_submit`) |
| Defense-in-depth `round_id` mismatch reject | `src/tests/unit/api/test_worker_coordinator.py` (`test_reject_stale_round_id_defense`) |

Operator contracts: [JUNIPER_CASCOR_API_REFERENCE.md — WS `/ws/v1/workers`](../api/JUNIPER_CASCOR_API_REFERENCE.md#ws-wsv1workers).

---

## Related Documentation

- [Testing Quick Start](QUICK_START.md) - Getting started with testing
- [AGENTS.md](../../AGENTS.md) - Project conventions and commands
- [pyproject.toml](../../pyproject.toml) - pytest configuration (`[tool.pytest.ini_options]`)
- [CI Workflow](../../.github/workflows/ci.yml) - GitHub Actions configuration
