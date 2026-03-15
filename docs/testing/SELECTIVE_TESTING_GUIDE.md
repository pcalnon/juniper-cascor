# Selective Testing Guide

This guide covers techniques for running specific subsets of tests in the Juniper Cascor project.

---

## 1. Running Specific Test Categories

The test suite uses pytest markers to categorize tests. Use the `-m` flag to select tests by marker.

### Unit Tests Only

```bash
cd src/tests && bash scripts/run_tests.bash -u
# Or directly with pytest:
cd src/tests && python -m pytest -m "unit" -v
```

### Integration Tests

```bash
cd src/tests && bash scripts/run_tests.bash -i
# Or directly:
cd src/tests && python -m pytest -m "integration" --integration -v
```

### Fast Tests (Exclude Slow)

```bash
cd src/tests && python -m pytest -m "not slow" -v
```

### GPU Tests

GPU tests require the `--gpu` flag to enable CUDA:

```bash
cd src/tests && bash scripts/run_tests.bash -g
# Or directly:
cd src/tests && python -m pytest -m "gpu" --gpu -v
```

### Combining Markers

```bash
# Unit AND accuracy tests
cd src/tests && python -m pytest -m "unit and accuracy" -v

# Unit OR integration tests
cd src/tests && python -m pytest -m "unit or integration" -v

# Unit tests but NOT slow
cd src/tests && python -m pytest -m "unit and not slow" -v
```

### Available Markers

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests for individual components |
| `integration` | Integration tests for full workflows |
| `performance` | Performance and benchmarking tests |
| `slow` | Tests that take a long time to run |
| `gpu` | Tests that require GPU/CUDA |
| `multiprocessing` | Tests that use multiprocessing |
| `spiral` | Spiral problem tests |
| `correlation` | Correlation calculation tests |
| `network_growth` | Network growth algorithm tests |
| `candidate_training` | Candidate unit training tests |
| `validation` | Input validation tests |
| `accuracy` | Accuracy calculation tests |
| `early_stopping` | Early stopping logic tests |

---

## 2. Running Tests by Module/File

### Single Test File

```bash
cd src/tests && python -m pytest unit/test_forward_pass.py -v
```

### Single Test Function

```bash
cd src/tests && python -m pytest unit/test_forward_pass.py::test_forward_basic -v
```

### Single Test Class

```bash
cd src/tests && python -m pytest unit/test_forward_pass.py::TestForwardPass -v
```

### Tests by Keyword

Use `-k` to filter tests by name pattern:

```bash
# All tests with "accuracy" in their name
cd src/tests && python -m pytest -k "accuracy" -v

# Tests matching "forward" but not "backward"
cd src/tests && python -m pytest -k "forward and not backward" -v

# Tests matching a specific pattern
cd src/tests && python -m pytest -k "test_train" -v
```

### Combining File and Keyword

```bash
cd src/tests && python -m pytest unit/ -k "correlation" -v
```

---

## 3. Running Slow Tests

### Why Slow Tests Are Separate

Slow tests are marked with `@pytest.mark.slow` and are skipped by default because:

- They involve full training cycles that can take minutes
- CI pipelines need fast feedback loops
- Local development should remain responsive

### Running Slow Tests

```bash
# Using the test runner script
cd src/tests && bash scripts/run_tests.bash -s

# Directly with pytest (disable global timeout)
cd src/tests && python -m pytest -m "slow" --slow --timeout=0 -v
```

### Expected Runtime

| Test Category | Typical Duration |
|---------------|------------------|
| Unit tests | 5-30 seconds |
| Integration tests | 1-5 minutes |
| Slow tests | 5-30+ minutes |
| Full suite with slow | 30-60 minutes |

### Fast-Slow Mode

For faster slow test execution with reduced training parameters:

```bash
cd src/tests && python -m pytest -m "slow" --slow --fast-slow -v
# Or via environment variable:
JUNIPER_FAST_SLOW=1 python -m pytest -m "slow" --slow -v
```

---

## 4. Running with Coverage

### Enable Coverage

Coverage is enabled by default in `run_tests.bash`. To explicitly enable:

```bash
cd src/tests && bash scripts/run_tests.bash -c
```

### Disable Coverage

For faster test runs without coverage overhead:

```bash
cd src/tests && bash scripts/run_tests.bash --no-coverage
```

### Specific Module Coverage

```bash
cd src/tests && python -m pytest \
    --cov=../cascade_correlation \
    --cov=../candidate_unit \
    --cov-report=term-missing \
    -v
```

### Coverage for a Subset of Tests

```bash
cd src/tests && python -m pytest unit/test_forward_pass.py \
    --cov=../cascade_correlation \
    --cov-report=html:reports/htmlcov \
    -v
```

### Viewing Coverage Reports

After running tests with coverage:

```bash
# HTML report (interactive, recommended)
open src/tests/reports/htmlcov/index.html

# Terminal summary is displayed automatically with --cov-report=term-missing

# XML report for CI integration
cat src/tests/reports/coverage.xml
```

---

## 5. Parallel Test Execution

### Using pytest-xdist

Enable parallel execution with the `-j` flag:

```bash
cd src/tests && bash scripts/run_tests.bash -j
```

Or directly with pytest:

```bash
cd src/tests && python -m pytest -n auto -v
# Specific number of workers:
cd src/tests && python -m pytest -n 4 -v
```

### Coverage with Parallel Tests

Coverage works with parallel execution but requires `pytest-cov` to aggregate results:

```bash
cd src/tests && python -m pytest -n auto \
    --cov=../cascade_correlation \
    --cov-report=html:reports/htmlcov \
    -v
```

### Thread Limiting

The test configuration automatically limits threads per worker to prevent CPU oversubscription:

```python
# From conftest.py - automatically applied when using pytest-xdist
if os.environ.get("PYTEST_XDIST_WORKER"):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
```

### When NOT to Use Parallel Execution

Avoid `-n` (parallel) when:

- **Debugging**: Output becomes interleaved and hard to follow
- **Shared resources**: Tests that write to the same files
- **GPU tests**: CUDA memory conflicts between workers
- **Multiprocessing tests**: Risk of process pool conflicts
- **Investigating flaky tests**: Need reproducible ordering

---

## 6. Re-running Failed Tests

### Last Failed (`--lf`)

Re-run only the tests that failed in the previous run:

```bash
cd src/tests && bash scripts/run_tests.bash -f
# Or directly:
cd src/tests && python -m pytest --lf -v
```

### Failed First (`--ff`)

Run failed tests first, then the rest:

```bash
cd src/tests && python -m pytest --ff -v
```

### Clear Failed Test Cache

```bash
cd src/tests && rm -rf .pytest_cache/v/cache/lastfailed
```

### CI Retry Strategies

For CI pipelines, consider:

```bash
# Retry failed tests once
cd src/tests && python -m pytest --lf --tb=short || python -m pytest --lf -v

# Use pytest-rerunfailures for automatic retries
cd src/tests && python -m pytest --reruns 2 --reruns-delay 1
```

---

## 7. Performance Testing

### Running Performance Markers

```bash
cd src/tests && bash scripts/run_tests.bash -p
# Or directly:
cd src/tests && python -m pytest -m "performance" -v
```

### Benchmark Tests

Run the dedicated benchmark script:

```bash
cd src/tests/scripts && bash run_benchmarks.bash           # All benchmarks
cd src/tests/scripts && bash run_benchmarks.bash -s        # Serialization only
cd src/tests/scripts && bash run_benchmarks.bash -q -n 10  # Quiet mode, 10 iterations
```

### Profiling During Tests

```bash
# CPU profiling with cProfile
cd src && python main.py --profile

# Memory profiling
cd src && python main.py --profile-memory

# Sampling profiler for training
./util/profile_training.bash
./util/profile_training.bash --svg  # Generate flame graph
```

---

## 8. Common Recipes

### Quick Validation Before Commit

Fast feedback for local development:

```bash
cd src/tests && python -m pytest -m "unit and not slow" --no-cov -q
```

### Full Local Test Run Matching CI

Replicate the CI test environment:

```bash
cd src/tests && bash scripts/run_tests.bash -v -c
```

### Debug a Specific Failing Test

```bash
# Run single test with verbose output and full traceback
cd src/tests && python -m pytest unit/test_forward_pass.py::test_failing_case \
    -v --tb=long -s

# With pdb on failure
cd src/tests && python -m pytest unit/test_forward_pass.py::test_failing_case \
    --pdb -v

# Show local variables in traceback
cd src/tests && python -m pytest unit/test_forward_pass.py::test_failing_case \
    -v --tb=long -l
```

### Run Tests Matching Multiple Patterns

```bash
cd src/tests && python -m pytest -k "forward or backward" -m "unit" -v
```

### Reduce Logging Noise During Tests

```bash
export CASCOR_LOG_LEVEL=WARNING
cd src/tests && python -m pytest -v
```

### Run All Tests Except GPU and Slow

```bash
cd src/tests && python -m pytest -m "not gpu and not slow" -v
```

### Generate JUnit XML for CI

```bash
cd src/tests && python -m pytest \
    --junitxml=reports/junit.xml \
    -v
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Unit tests only | `bash scripts/run_tests.bash -u` |
| Integration tests | `bash scripts/run_tests.bash -i` |
| Slow tests | `bash scripts/run_tests.bash -s` |
| GPU tests | `bash scripts/run_tests.bash -g` |
| Performance tests | `bash scripts/run_tests.bash -p` |
| With coverage | `bash scripts/run_tests.bash -c` |
| Without coverage | `bash scripts/run_tests.bash --no-coverage` |
| Parallel execution | `bash scripts/run_tests.bash -j` |
| Re-run failed | `bash scripts/run_tests.bash -f` |
| Verbose output | `bash scripts/run_tests.bash -v` |
| Specific test file | `bash scripts/run_tests.bash -t test_file.py` |
| Custom markers | `bash scripts/run_tests.bash -m "unit and accuracy"` |
