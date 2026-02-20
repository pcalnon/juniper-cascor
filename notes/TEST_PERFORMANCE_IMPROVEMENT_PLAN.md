# JuniperCascor Test Performance Improvement Plan

**Created:** 2026-02-19
**Target:** ≤ 5 seconds for full unit test suite (1,422 tests)
**Current:** 500+ seconds (stalling/deadlocking, effectively infinite)
**Constraint:** Do NOT delete or disable any tests or test functionality

---

## Table of Contents

- [JuniperCascor Test Performance Improvement Plan](#junipercascor-test-performance-improvement-plan)
  - [Table of Contents](#table-of-contents)
  - [Executive Summary](#executive-summary)
  - [Current State Analysis](#current-state-analysis)
    - [Test Suite Metrics](#test-suite-metrics)
    - [Timing Breakdown (Current)](#timing-breakdown-current)
    - [Baseline Without Coverage (non-training tests only)](#baseline-without-coverage-non-training-tests-only)
  - [Root Cause Analysis](#root-cause-analysis)
    - [CRITICAL: Multiprocessing Worker Deadlock](#critical-multiprocessing-worker-deadlock)
    - [HIGH: Training Overhead in Non-Training Tests](#high-training-overhead-in-non-training-tests)
    - [MEDIUM: Coverage Overhead in Default Test Runs](#medium-coverage-overhead-in-default-test-runs)
    - [LOW: Broken Test Collection](#low-broken-test-collection)
    - [LOW: Excessive Fixture Training Parameters](#low-excessive-fixture-training-parameters)
  - [Improvement Plan](#improvement-plan)
    - [Phase 1: Eliminate Deadlocks (CRITICAL)](#phase-1-eliminate-deadlocks-critical)
      - [Fix 1A: Force Sequential Training in Tests via conftest.py](#fix-1a-force-sequential-training-in-tests-via-conftestpy)
    - [Phase 2: Remove Coverage from Default Runs](#phase-2-remove-coverage-from-default-runs)
      - [Fix 2A: Remove coverage from pytest.ini addopts](#fix-2a-remove-coverage-from-pytestini-addopts)
      - [Fix 2B: Remove `--cov=correlation` (non-existent module)](#fix-2b-remove---covcorrelation-non-existent-module)
      - [Fix 2C: Add coverage run script/command](#fix-2c-add-coverage-run-scriptcommand)
    - [Phase 3: Fix Broken Test Collection](#phase-3-fix-broken-test-collection)
      - [Fix 3A: Fix test\_hdf5.py import path](#fix-3a-fix-test_hdf5py-import-path)
    - [Phase 4: Optimize Training Fixtures](#phase-4-optimize-training-fixtures)
      - [Fix 4A: Create lightweight `network_with_hidden_units` fixture](#fix-4a-create-lightweight-network_with_hidden_units-fixture)
      - [Fix 4B: Reduce conftest fixture training parameters](#fix-4b-reduce-conftest-fixture-training-parameters)
    - [Phase 5: Harden Worker Shutdown (Preventive)](#phase-5-harden-worker-shutdown-preventive)
      - [Fix 5A: Bound total shutdown time in `_stop_workers()`](#fix-5a-bound-total-shutdown-time-in-_stop_workers)
      - [Fix 5B: Add environment variable for process count override](#fix-5b-add-environment-variable-for-process-count-override)
  - [Expected Impact](#expected-impact)
    - [After Phase 1 (Force Sequential Training)](#after-phase-1-force-sequential-training)
    - [After Phase 2 (Remove Coverage from Default)](#after-phase-2-remove-coverage-from-default)
    - [After Phase 4 (Optimize Fixtures)](#after-phase-4-optimize-fixtures)
    - [Stretch Goal: ≤ 5 seconds](#stretch-goal--5-seconds)
  - [Risk Assessment](#risk-assessment)
  - [Verification Procedure](#verification-procedure)
  - [Phase 6: Logger Class Method and Torch Warmup Optimizations (2026-02-20)](#phase-6-logger-class-method-and-torch-warmup-optimizations-2026-02-20)
    - [Problem](#problem)
    - [Fixes Applied](#fixes-applied)
    - [Per-File Impact (Before → After)](#per-file-impact-before--after)
    - [Result](#result)
    - [Remaining Floor](#remaining-floor)
  - [References](#references)

---

## Executive Summary

The JuniperCascor unit test suite (1,422 tests) is effectively **stalling indefinitely** due to a critical
multiprocessing worker deadlock. When tests call `network.fit()`, the cascade correlation network spawns
real multiprocessing worker processes that fail with `BrokenPipeError` in the test environment. The
`_stop_workers()` method then blocks for **15 seconds per worker** during shutdown, and with multiple
workers per training call, this accumulates to 90-120+ seconds per occurrence. Combined with coverage
overhead and excessive training parameters in fixtures, the suite exceeds 500 seconds.

The fix requires 5 targeted changes, none of which delete or disable tests.

---

## Current State Analysis

### Test Suite Metrics

| Metric                       | Value                                     |
| ---------------------------- | ----------------------------------------- |
| Total unit tests             | 1,422                                     |
| Tests skipped (markers)      | ~15 (slow, integration)                   |
| Tests with collection errors | 1 (test_hdf5.py ImportError)              |
| Tests stalling (>30s each)   | 3 (snapshot_serializer_coverage fixtures) |
| Tests slow (>5s)             | 1 (cascade_correlation_coverage_90)       |
| Tests moderate (0.5-2s)      | ~25                                       |
| Remaining tests              | ~1,377 (<0.5s each)                       |

### Timing Breakdown (Current)

| Component                               | Time                 | Notes                                          |
| --------------------------------------- | -------------------- | ---------------------------------------------- |
| 3 × multiprocessing deadlock setups     | ~90s                 | `test_snapshot_serializer_coverage.py` fixture |
| 1 × `test_get_training_results_empty`   | ~9s                  | Triggers parallel training                     |
| 1 × `test_start_training` (lifecycle)   | ~2s                  | ThreadPoolExecutor training                    |
| ~25 × tests with training (0.5-2s each) | ~20s                 | Various serializer/training tests              |
| ~1,377 × fast tests (~0.04s each)       | ~55s                 | Includes Python startup overhead               |
| Coverage collection + report generation | ~2-3x multiplier     | Adds ~100-200s                                 |
| **Total observed**                      | **500-600+ seconds** | **Often stalls indefinitely**                  |

### Baseline Without Coverage (non-training tests only)

~657 non-training tests complete in ~28 seconds without coverage.

---

## Root Cause Analysis

### CRITICAL: Multiprocessing Worker Deadlock

**Impact:** Adds 90-120+ seconds per occurrence. Primary cause of stalling.

**Mechanism:**

1. Tests call `network.fit()` which triggers `grow_network()` → `train_candidates()`
2. `_calculate_optimal_process_count()` returns `cpu_count - 1` (e.g., 7 on an 8-core machine)
3. `_execute_candidate_training()` sees `process_count > 1` and calls `_execute_parallel_training()`
4. `_execute_parallel_training()` spawns real `multiprocessing.Process` workers via `self._mp_ctx.Process()`
5. Workers communicate via `multiprocessing.managers.BaseManager` queues
6. In the test environment, workers fail with `BrokenPipeError` / `EOFError` when trying to put results
7. `_stop_workers()` then executes a **serial** 4-phase shutdown:
   - Phase 1: Send sentinels (fails with BrokenPipeError)
   - Phase 2: `worker.join(timeout=15)` **per worker** — blocks 15s × N workers = 105+ seconds
   - Phase 3: `worker.terminate()` + `worker.join(timeout=2)`
   - Phase 4: SIGKILL
8. After timeout, `_execute_candidate_training()` falls back to sequential training (double work)

**Evidence:**

```bash
Stack trace from pytest-timeout:
  _stop_workers() → worker.join(timeout=15) → popen_fork.wait(timeout) → selector.poll(timeout)

Error logs:
  BrokenPipeError: [Errno 32] Broken pipe
  EOFError (from multiprocessing.connection.recv)
  CRITICAL: Worker critical get error: [Errno 32] Broken pipe
```

**Affected Tests:**

| Test File                                       | Test/Fixture                                  | Time Impact   |
| ----------------------------------------------- | --------------------------------------------- | ------------- |
| `test_snapshot_serializer_coverage.py`          | `network_with_hidden_units` fixture (3 tests) | 30s × 3 = 90s |
| `test_cascade_correlation_coverage_90.py`       | `test_get_training_results_empty`             | 9.27s         |
| `test_cascade_correlation_coverage_extended.py` | 5 tests calling `fit()`                       | Variable      |
| conftest.py                                     | `trained_simple_network` fixture              | Variable      |

**Key Code Locations:**

- `src/cascade_correlation/cascade_correlation.py:1492` — `_execute_parallel_training()` call
- `src/cascade_correlation/cascade_correlation.py:1537` — Process count calculation
- `src/cascade_correlation/cascade_correlation.py:1717` — `worker.join(timeout=15)` serial loop
- `src/cascade_correlation/cascade_correlation.py:1680` — `_collect_training_results()` 60s queue timeout

### HIGH: Training Overhead in Non-Training Tests

**Impact:** Adds 10-30 seconds (before deadlock fixes are applied).

Tests that exist to validate serialization, topology, or configuration create networks and call `fit()`
solely to produce a network with hidden units. This triggers the full training pipeline (candidate
generation, training, selection, installation) when all they need is a network object with hidden units
populated.

**Affected:**

- `test_snapshot_serializer_coverage.py:48` — `network_with_hidden_units` fixture calls `fit(x, y, epochs=5)`
- `test_snapshot_serializer.py:196` — `test_save_network_with_hidden_units` trains to get hidden units
- Various coverage tests that call `fit()` to test error handling paths

### MEDIUM: Coverage Overhead in Default Test Runs

**Impact:** 2-3x time multiplier on all tests.

`pytest.ini` includes coverage collection in `addopts`:

```ini
addopts =
    --cov=cascade_correlation
    --cov=candidate_unit
    --cov=correlation           # <-- Module never imported (CoverageWarning)
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
```

**Issues:**

1. `--cov=correlation` targets a module that doesn't exist/isn't imported → wasted overhead + warning
2. HTML report generation (`--cov-report=html:htmlcov`) adds file I/O overhead on every run
3. XML report generation (`--cov-report=xml`) adds overhead on every run
4. Coverage instrumentation adds ~2-3x runtime overhead to every test
5. `cascade_correlation.py` is 4,194 lines — expensive to instrument

### LOW: Broken Test Collection

**Impact:** 1 test file fails to collect (21 tests lost).

`test_hdf5.py:14` imports `cascade_correlation.snapshots.snapshot_utils.HDF5Utils` which doesn't exist.
The module path has changed to `snapshots.snapshot_utils`.

### LOW: Excessive Fixture Training Parameters

**Impact:** ~5-15 seconds (after deadlock fix).

Default conftest fixtures use training parameters that are larger than needed for unit tests:

- `candidate_pool_size`: min(8, ...) in conftest but **50** in standalone fixtures
- `candidate_epochs`: min(10, ...)
- `output_epochs`: min(10, ...)
- `epochs_max`: min(20, ...)

---

## Improvement Plan

### Phase 1: Eliminate Deadlocks (CRITICAL)

**Estimated time savings: 100-200+ seconds**
**Implementation effort: Small (< 1 hour)**

#### Fix 1A: Force Sequential Training in Tests via conftest.py

Add an autouse fixture to `src/tests/conftest.py` that prevents multiprocessing from being used
during tests. This eliminates the worker deadlock entirely.

**Approach:** Monkeypatch `_calculate_optimal_process_count` to always return `1` in tests.
When `process_count == 1`, `_execute_candidate_training()` uses `_execute_sequential_training()`
instead of spawning worker processes.

```python
# Add to src/tests/conftest.py

@pytest.fixture(autouse=True)
def force_sequential_training(monkeypatch):
    """Force sequential candidate training in tests to prevent multiprocessing deadlocks.

    The parallel training path spawns multiprocessing.Process workers that fail with
    BrokenPipeError in test environments. The _stop_workers() method then blocks for
    15 seconds per worker during shutdown. By forcing process_count=1, all training
    uses the sequential path, which is functionally identical but avoids multiprocessing
    overhead and deadlock risk.

    Tests that specifically need to test multiprocessing behavior should use
    @pytest.mark.multiprocessing and mock the multiprocessing components directly.
    """
    monkeypatch.setattr(
        CascadeCorrelationNetwork,
        "_calculate_optimal_process_count",
        lambda self: 1,
    )
```

**Why this works:**

- The decision point is in `_execute_candidate_training()` at line 1492:

  ```python
  if process_count > 1:
      results = self._execute_parallel_training(tasks, process_count)
  else:
      results = self._execute_sequential_training(tasks)
  ```

- Returning `1` from `_calculate_optimal_process_count()` guarantees the sequential path is taken
- Sequential training produces identical results (same `train_candidate_worker()` function)
- Tests marked `@pytest.mark.multiprocessing` already skip by default (need `--slow` flag)

**Verification:**

```bash
cd src/tests
pytest unit/test_snapshot_serializer_coverage.py --no-cov -v --timeout=10
# Should complete in < 5 seconds (was timing out at 30+ seconds per test)
```

### Phase 2: Remove Coverage from Default Runs

**Estimated time savings: 50-60% of remaining runtime (2-3x speedup)**
**Implementation effort: Small (< 30 minutes)**

#### Fix 2A: Remove coverage from pytest.ini addopts

Move coverage configuration out of `addopts` so it's not executed on every test run.
Create a separate coverage invocation for CI/explicit coverage runs.

**Before (`pytest.ini`):**

```ini
addopts =
    -ra
    -q
    -p no:warnings
    --strict-markers
    --strict-config
    --continue-on-collection-errors
    --ignore=src/tests
    --tb=short
    --cov=cascade_correlation
    --cov=candidate_unit
    --cov=correlation
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
```

**After (`pytest.ini`):**

```ini
addopts =
    -ra
    -q
    -p no:warnings
    --strict-markers
    --strict-config
    --continue-on-collection-errors
    --ignore=src/tests
    --tb=short
```

#### Fix 2B: Remove `--cov=correlation` (non-existent module)

The `correlation` module is never imported. Remove the `--cov=correlation` entry entirely.
This also eliminates the `CoverageWarning: Module correlation was never imported` message.

#### Fix 2C: Add coverage run script/command

Ensure coverage can still be run explicitly:

```bash
# In run_tests.bash or as a documented command:
pytest unit/ \
    --cov=cascade_correlation \
    --cov=candidate_unit \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --cov-report=xml:coverage.xml
```

### Phase 3: Fix Broken Test Collection

**Estimated time savings: Negligible (correctness fix)**
**Implementation effort: Small (< 15 minutes)**

#### Fix 3A: Fix test_hdf5.py import path

The import at line 14 references a non-existent module path:

```python
# Current (broken):
from cascade_correlation.snapshots.snapshot_utils import HDF5Utils

# Fix: Update to correct import path:
from snapshots.snapshot_utils import HDF5Utils
```

**Or** if `HDF5Utils` no longer exists, update the test to use the current API.

### Phase 4: Optimize Training Fixtures

**Estimated time savings: 5-15 seconds**
**Implementation effort: Medium (1-2 hours)**

#### Fix 4A: Create lightweight `network_with_hidden_units` fixture

Instead of calling `fit()` (which runs full training), directly construct a network
with hidden units by calling the internal `_install_candidate()` or `add_unit()` method.

**Current (slow — triggers full training pipeline):**

```python
@pytest.fixture
def network_with_hidden_units(simple_network, tmp_path):
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20, 1)).float()
    simple_network.fit(x, y, epochs=5)  # Triggers candidate training
    return simple_network
```

**Proposed (fast — directly adds a hidden unit):**

```python
@pytest.fixture
def network_with_hidden_units(simple_network):
    """Create a network with hidden units WITHOUT full training.

    Directly adds a candidate unit to the network, avoiding the full
    training pipeline (candidate generation, training, selection).
    Use this for tests that need hidden unit structure but don't test training.
    """
    # Create and install a candidate unit directly
    candidate = CandidateUnit(
        _CandidateUnit__input_size=simple_network.input_size + 1,  # +1 for bias
        _CandidateUnit__learning_rate=0.01,
        _CandidateUnit__epochs=1,
        _CandidateUnit__log_level_name="ERROR",
    )
    candidate.weights = torch.randn(simple_network.input_size + 1)
    simple_network.add_unit(candidate)
    return simple_network
```

**Note:** The exact API for adding units must be verified against the `add_unit()` method signature.
If `add_unit()` has prerequisites (e.g., needing correlation data), a minimal `fit()` with
`candidate_pool_size=1, candidate_epochs=1, output_epochs=1, epochs_max=1` can be used instead.

#### Fix 4B: Reduce conftest fixture training parameters

For the shared conftest fixtures, reduce default training parameters further:

```python
@pytest.fixture
def simple_config(fast_training_params) -> CascadeCorrelationConfig:
    return CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=2,
        learning_rate=0.1,
        candidate_learning_rate=0.1,
        max_hidden_units=2,           # Was min(5, ...)
        candidate_pool_size=2,        # Was min(8, ...) — key reduction
        correlation_threshold=0.01,   # Lower threshold = faster convergence
        patience=1,                   # Was min(3, ...)
        candidate_epochs=3,           # Was min(10, ...)
        output_epochs=3,              # Was min(10, ...)
        epochs_max=5,                 # Was min(20, ...)
    )
```

### Phase 5: Harden Worker Shutdown (Preventive)

**Estimated time savings: Prevents future stalls**
**Implementation effort: Medium (1-2 hours)**

#### Fix 5A: Bound total shutdown time in `_stop_workers()`

Change the serial `worker.join(timeout=15)` loop to use a **bounded total shutdown time**
instead of 15 seconds per worker. This prevents N workers × 15s = long stalls.

**Current (serial, unbounded):**

```python
# Phase 2: Wait gracefully with increased timeout
for worker in workers:
    worker.join(timeout=15)  # 15s PER WORKER = N × 15s total
```

**Proposed (bounded total):**

```python
# Phase 2: Wait with bounded total timeout
total_timeout = 10  # Total time for all workers, not per-worker
deadline = time.time() + total_timeout
for worker in workers:
    remaining = max(0.1, deadline - time.time())
    worker.join(timeout=remaining)
    if time.time() >= deadline:
        break
```

#### Fix 5B: Add environment variable for process count override

Add support for `CASCOR_NUM_PROCESSES` environment variable in `_calculate_optimal_process_count()`:

```python
def _calculate_optimal_process_count(self) -> int:
    # Allow environment override (useful for testing and CI)
    env_override = os.environ.get("CASCOR_NUM_PROCESSES")
    if env_override is not None:
        return max(1, int(env_override))
    # ... existing logic ...
```

This provides a configuration-based alternative to monkeypatching.

---

## Expected Impact

### After Phase 1 (Force Sequential Training)

| Metric                            | Before    | After        |
| --------------------------------- | --------- | ------------ |
| 3 × snapshot_serializer deadlocks | 90s       | < 3s         |
| test_get_training_results_empty   | 9.27s     | < 1s         |
| Parallel training fallbacks       | ~30s      | 0s           |
| **Total estimated**               | **500+s** | **~80-120s** |

### After Phase 2 (Remove Coverage from Default)

| Metric                            | Before       | After       |
| --------------------------------- | ------------ | ----------- |
| Coverage instrumentation overhead | 2-3x         | 1x          |
| HTML/XML report generation        | ~5-10s       | 0s          |
| **Total estimated**               | **~80-120s** | **~30-50s** |

### After Phase 4 (Optimize Fixtures)

| Metric                    | Before      | After       |
| ------------------------- | ----------- | ----------- |
| Training fixture overhead | ~20-30s     | ~5-10s      |
| **Total estimated**       | **~30-50s** | **~15-25s** |

### Stretch Goal: ≤ 5 seconds

Achieving ≤ 5s for 1,422 tests requires ~3.5ms per test average. This may require:

- `pytest-xdist` parallel execution across CPU cores
- Module-scope fixtures for expensive test setup (network creation)
- Lazy imports in test modules
- Reducing Python startup overhead

These are **not** included in this plan as they require architectural changes to the test suite.
A realistic target after Phases 1-4 is **15-25 seconds**, which is a **20-40x improvement** from
the current stalling state.

---

## Risk Assessment

| Risk                          | Mitigation                                                                                                                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Monkeypatch misses code path  | Patch targets `_calculate_optimal_process_count` only entry point for proc count decisions in `_execute_candidate_training()`. All training flows go through this method.                                     |
| Lose multiprocessing test cov | Multiproc tests marked `@pytest.mark.multiprocessing` mock components & verify logic, don't spawn procs (`test_cascade_correlation_coverage_90.py:286`, `test_cascade_correlation_coverage_extended.py:902`). |
| Fixture change break asserts  | The `network_with_hidden_units` fixture is only used by 3 tests in `test_snapshot_serializer_coverage.py` & test serialization format, not training behavior.                                                 |
| Default, not track Cov metric | Coverage moved to explicit invocation (`pytest --cov=...`), CI pipeline should be updated with this explicit command.                                                                                         |

---

## Verification Procedure

After implementing all phases, verify with:

```bash
# 1. Run full unit test suite without coverage
cd src/tests
time pytest unit/ --no-cov --timeout=30 -q

# Expected: < 30 seconds, 0 failures, 0 errors

# 2. Run with coverage (explicit)
time pytest unit/ --cov=cascade_correlation --cov=candidate_unit --cov-report=term-missing --timeout=30 -q

# Expected: < 60 seconds, 0 failures, 0 errors

# 3. Verify no test functionality was lost
pytest unit/ --no-cov --co -q 2>&1 | grep "<Function" | wc -l
# Expected: 1,422+ (same or more than before)

# 4. Verify multiprocessing tests still pass (with explicit flag)
pytest unit/ -m multiprocessing --no-cov -v --slow --timeout=60
# Expected: All multiprocessing-marked tests pass
```

---

## Phase 6: Logger Class Method and Torch Warmup Optimizations (2026-02-20)

### Problem

After Phases 1-5, the test suite ran in 36-64s. Profiling revealed two remaining bottlenecks:

1. **Logger._log_at_level → inspect.getouterframes()**: CandidateUnit.**init** sets `self.logger = Logger`
   (the class), then calls ~20+ class-level Logger methods (trace, debug, info, verbose, etc.) during
   construction. While most messages are filtered by log level, WARNING-level messages from
   `_seed_random_generator` still pass through to `_log_at_level`, which calls `_console_dict` →
   `_frame_info` → `getouterframes(frame)`. Each `getouterframes()` call triggers `inspect.getmodule()`
   which scans all loaded modules via `hasattr()` — ~800K `hasattr` calls per 20 CandidateUnit creations,
   costing **4.3s**. SpiralProblem.**init** has the same issue.

2. **First torch.optim/nn.Linear usage**: The first test to call `train_output_layer()` pays ~2s for lazy
   torch internal initialization (sympy, _dynamo, etc.).

3. **test_start_training**: Actually ran CasCor training for 3.81s.

4. **test_sequential_training_fallback**: Ran real candidate training for 2.58s with default epochs.

### Fixes Applied

| Fix                                                                 | File                                   | Impact                                                                       |
| ------------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------- |
| Patch `Logger._log_at_level` to no-op in session fixture            | `conftest.py`                          | Eliminates all `inspect.getouterframes()` overhead from Logger class methods |
| Add `_warmup_torch` session fixture                                 | `conftest.py`                          | Moves ~2s one-time torch init to session startup                             |
| Mock `fit()` in `test_start_training`                               | `test_lifecycle_manager.py`            | 3.81s → <0.1s                                                                |
| Reduce candidate epochs to 1 in `test_sequential_training_fallback` | `test_cascade_correlation_coverage.py` | 2.58s → <0.3s                                                                |

### Per-File Impact (Before → After)

| File                                    | Before (s) | After (s) | Speedup |
| --------------------------------------- | ---------- | --------- | ------- |
| test_spiral_problem_extended.py         | 9.64       | 0.37      | 26x     |
| test_candidate_unit_extended.py         | 5.28       | 0.17      | 31x     |
| test_cascade_correlation_coverage.py    | 5.04       | 0.88      | 5.7x    |
| test_lifecycle_manager.py               | 4.67       | 0.10      | 47x     |
| test_candidate_unit_coverage.py         | 3.90       | 0.16      | 24x     |
| test_cascade_correlation_coverage_90.py | 3.71       | 0.82      | 4.5x    |
| test_snapshot_serializer_extended.py    | 3.64       | 0.30      | 12x     |
| test_snapshot_serializer_coverage.py    | 3.45       | 0.33      | 10x     |

### Result

- **Before Phase 6**: 36-64s wall time, 43s measured test time
- **After Phase 6**: 12-24s wall time, ~10s measured test time
- **Total reduction from original**: 167s → 12-24s (86-93% reduction)
- **Tests**: 1408 passed, 15 skipped, 0 failed

### Remaining Floor

The ~10-12s minimum is dominated by:

- Python startup + torch/numpy/matplotlib imports (~8s collection time)
- pytest framework overhead (1408 × ~2ms ≈ 3s)
- Irreducible test execution (~5s of actual training/validation)

Further improvement to ≤5s would require `pytest-xdist` for parallel execution.

---

## References

- `src/cascade_correlation/cascade_correlation.py` — Lines 1476-1742 (training pipeline)
- `src/tests/conftest.py` — Test fixtures and configuration
- `src/tests/pytest.ini` — Pytest configuration with coverage settings
- `src/tests/unit/test_snapshot_serializer_coverage.py` — Affected fixture (line 48-53)
- `src/cascor_constants/constants_candidates/constants_candidates.py:214` — Default pool size = 50
