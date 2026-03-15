# Test API Mismatch Fix Plan — 2026-03-14

**Date**: 2026-03-14
**Branch**: `fix/test-api-mismatches`
**Status**: Complete
**Related**: `PRE_COMMIT_FIX_PLAN.md`, `PRE_COMMIT_FIX_PLAN_2026-03-14.md`

---

## Overview

Three test files contain assertions written against obsolete production APIs. The production code evolved (return types changed, methods split into backward-compatible and detailed variants) but these test files were not updated. All three failures are **test-code defects**, not production-code regressions.

**Test run**: 2201 passed, 3 failed, 1 skipped, 135 warnings (267.79s)

---

## Root Cause Analysis

### Common Theme

The production code underwent two API evolutions:

1. **`CandidateUnit.train()`** was split into two methods:
   - `train()` → returns `float` (backward-compatible, returns correlation value)
   - `train_detailed()` → returns `CandidateTrainingResult` dataclass (new detailed API)
   - `train()` internally calls `train_detailed()` and also stores the result in `self.last_training_result`

2. **`CascadeCorrelationNetwork.train_candidates()`** return type changed:
   - **Old**: returned a `tuple` containing candidate data
   - **New**: returns a `TrainingResults` dataclass with named fields

Three test files still assert against the old APIs.

---

## Issue 1: `test_cascor_fix.py::test_sequential_candidate_training`

### Symptom

```
AssertionError: Invalid training_stats format: TrainingResults(...)
assert isinstance(TrainingResults(...), tuple)  → False
```

### Root Cause

`_get_candidate_training_stats()` (line 79) asserts `isinstance(training_stats, tuple)`. The production `train_candidates()` now returns a `TrainingResults` dataclass.

Lines 79-82 expect:
```python
# Old: tuple-based structure
assert isinstance(training_stats, tuple) and len(training_stats) >= 1
candidates_data = training_stats[0]
assert isinstance(candidates_data, tuple) and len(candidates_data) == 4
candidate_ids, candidate_uuids, correlations, candidates = candidates_data
```

### Fix

Update `_get_candidate_training_stats()` to use `TrainingResults` dataclass fields:

```python
from cascade_correlation.cascade_correlation import TrainingResults

assert isinstance(training_stats, TrainingResults), f"Expected TrainingResults, got {type(training_stats)}"
_validate_candidates_correlations(training_stats)
```

Update `_validate_candidates_correlations()` to accept `TrainingResults` and access named fields:

```python
def _validate_candidates_correlations(training_stats):
    candidate_ids = training_stats.candidate_ids
    candidate_uuids = training_stats.candidate_uuids
    correlations = training_stats.correlations
    candidates = training_stats.candidate_objects
    ...
```

### Files Modified

- `src/tests/unit/test_cascor_fix.py`: lines 66-96

---

## Issue 2: `test_critical_fixes.py::test_3_candidate_training`

### Symptom

```
AssertionError: train() should return CandidateTrainingResult
isinstance(0.40637328407715523, CandidateTrainingResult) → False
```

### Root Cause

Line 73 calls `candidate.train()` which now returns `float`. The test asserts the return value is `CandidateTrainingResult` (line 75). The test needs to call `candidate.train_detailed()` instead.

### Fix

Change `candidate.train(...)` to `candidate.train_detailed(...)` on line 73:

```python
result = candidate.train_detailed(x=x, epochs=test_epochs, residual_error=residual_error, learning_rate=0.01)
```

Both methods accept identical parameters, so only the method name changes.

### Files Modified

- `src/tests/unit/test_critical_fixes.py`: line 73

---

## Issue 3: `test_p1_fixes.py::test_1_early_stopping`

### Symptom

```
AttributeError: 'float' object has no attribute 'epochs_completed'
```

### Root Cause

Line 48 calls `candidate.train()` which returns `float`. Line 51 then tries `result.epochs_completed`. The test needs `train_detailed()` to get the `CandidateTrainingResult` with `epochs_completed`.

### Fix

Change `candidate.train(...)` to `candidate.train_detailed(...)` on line 48:

```python
result = candidate.train_detailed(x=x, epochs=100, residual_error=residual_error, learning_rate=0.01)
```

### Files Modified

- `src/tests/unit/test_p1_fixes.py`: line 48

---

## Warnings Analysis (135 warnings)

The test output also shows 135 warnings. These are NOT failures but are noted for completeness:

| Count | Source | Warning | Severity |
|-------|--------|---------|----------|
| 16 | `starlette/testclient.py:439` | `DeprecationWarning: 'timeout' arg with TestClient` | Low — upstream Starlette |
| 1 | `cascor_plotter.py:126` | `UserWarning: FigureCanvasAgg is non-interactive` | Low — expected in headless |
| 23 | `sentry_sdk/integrations/starlette.py:427` | `DeprecationWarning: asyncio.iscoroutinefunction` → Python 3.16 | Low — upstream sentry-sdk |
| 95 | `sentry_sdk/integrations/fastapi.py:76` | Same `asyncio.iscoroutinefunction` deprecation | Low — upstream sentry-sdk |

All warnings originate from third-party libraries and are already filtered in `pyproject.toml` where possible. No action required.

---

## Implementation Order

1. Fix `test_cascor_fix.py` — update tuple assertions to `TrainingResults` dataclass
2. Fix `test_critical_fixes.py` — change `train()` to `train_detailed()`
3. Fix `test_p1_fixes.py` — change `train()` to `train_detailed()`
4. Run the three fixed tests individually to verify
5. Run full test suite to verify no regressions
6. Run `pre-commit run --all-files` to verify all hooks pass

## Regression Testing

After fixes:
- All 3 previously-failing tests must pass
- All 2201 previously-passing tests must still pass
- Coverage must remain ≥ 80%
- All pre-commit hooks must pass

---

## Verification Commands

```bash
# Step 1: Run the three fixed tests
cd src && python -m pytest tests/unit/test_cascor_fix.py::test_sequential_candidate_training \
    tests/unit/test_critical_fixes.py::test_3_candidate_training \
    tests/unit/test_p1_fixes.py::test_1_early_stopping -v --tb=short

# Step 2: Full test suite
cd src/tests && bash scripts/run_tests.bash

# Step 3: Pre-commit
pre-commit run --all-files
```
