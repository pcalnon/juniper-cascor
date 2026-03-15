# Testing Quick Start Guide

Get tests running in under 5 minutes.

## Test Categories Overview

| Category | Speed | Purpose | Marker |
|----------|-------|---------|--------|
| **Unit** | Fast (~seconds) | Isolated component tests | `unit` |
| **Integration** | Medium (~minutes) | Full workflow tests | `integration` |
| **Slow** | Long (~5 min timeout) | Training-intensive tests | `slow` |
| **GPU** | Varies | CUDA-dependent tests | `gpu` |

## Most Common Commands

### 1. Run All Fast Tests (Default)

```bash
cd src/tests && bash scripts/run_tests.bash
```

This excludes slow tests and is what CI runs for quick feedback.

### 2. Run Only Unit Tests

```bash
cd src/tests && pytest -m "unit and not slow" -v
```

### 3. Run Integration Tests

```bash
cd src/tests && pytest -m "integration and not slow" -v
```

### 4. Run a Specific Test File

```bash
cd src/tests && pytest unit/test_forward_pass.py -v
```

### 5. Run Tests with Coverage

```bash
cd src/tests && bash scripts/run_tests.bash -c
```

### 6. Run Tests by Marker

```bash
# Spiral problem tests
cd src/tests && bash scripts/run_tests.bash -m "spiral"

# Candidate training tests
cd src/tests && pytest -m "candidate_training" -v

# Network growth tests
cd src/tests && pytest -m "network_growth" -v
```

### 7. Run Slow Tests (Extended Training)

```bash
# Only slow tests with timeout disabled
cd src/tests && pytest -m slow --timeout=0

# All tests including slow
cd src/tests && pytest --timeout=0
```

## Understanding Test Output

### Passing Tests

```
tests/unit/test_forward_pass.py::TestForwardPassBasics::test_forward_pass_no_hidden_units PASSED
tests/unit/test_accuracy.py::TestAccuracyCalculation::test_perfect_accuracy PASSED
======================= 42 passed in 3.21s =======================
```

### Report Locations

| Report | Path |
|--------|------|
| HTML Coverage | `src/tests/reports/htmlcov/index.html` |
| XML Coverage | `src/tests/reports/coverage.xml` |
| JUnit XML | `src/tests/reports/junit.xml` |

Open the HTML coverage report in a browser for detailed line-by-line coverage.

## Quick Troubleshooting

### Tests Timing Out?

They may be `slow` tests. Either:

```bash
# Exclude slow tests (default CI behavior)
pytest -m "not slow"

# Or disable timeout for slow tests
pytest --timeout=0
```

### Multiprocessing Errors?

Check for forkserver issues. Try running without parallel execution:

```bash
pytest --forked -n0
```

### GPU Tests Failing?

If you don't have CUDA installed:

```bash
# Exclude GPU tests
pytest -m "not gpu"
```

If you have CUDA but tests still fail, verify installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Import Errors?

Ensure you're in the correct directory:

```bash
cd src/tests && pytest ...
```

### Random Test Failures?

Set a random seed for deterministic behavior using the `random_seed` config option.

## Available Test Markers

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests for individual components |
| `integration` | Integration tests for full workflows |
| `slow` | Tests that take a long time to run |
| `gpu` | Tests that require GPU/CUDA |
| `multiprocessing` | Tests using multiprocessing |
| `spiral` | Spiral problem tests |
| `correlation` | Correlation calculation tests |
| `network_growth` | Network growth algorithm tests |
| `candidate_training` | Candidate unit training tests |
| `validation` | Input validation tests |
| `accuracy` | Accuracy calculation tests |
| `early_stopping` | Early stopping logic tests |

## Next Steps

- **Full Test Documentation**: See [src/tests/README.md](../../src/tests/README.md)
- **Test Runner Options**: Run `bash scripts/run_tests.bash -h` for all options
- **Writing New Tests**: Follow patterns in existing `unit/` and `integration/` tests
- **CI Configuration**: Tests run with `-m "not slow"` for fast feedback
