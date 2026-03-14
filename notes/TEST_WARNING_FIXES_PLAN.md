# Test Warning Fixes Plan

**Date**: 2026-03-13
**Author**: Claude Code
**Status**: Complete

---

## Problem Statement

Running the juniper-cascor test suite with warnings enabled (`-W default` or without `-p no:warnings`) produces multiple categories of warnings. While all 2124 tests pass, the warnings indicate code quality issues and potential bugs that should be addressed.

## Root Cause Analysis

### Issue 1: PytestReturnNotNoneWarning (13 test functions, 4 files)

**Root Cause**: Legacy test functions use a `return True`/`return False` pattern designed for standalone script execution (`if __name__ == "__main__"`). Pytest expects test functions to return `None` — failures should propagate as exceptions, not boolean return values.

**Affected Files**:
- `src/tests/unit/test_cascor_fix.py` — 2 test functions + 2 helpers
- `src/tests/unit/test_critical_fixes.py` — 5 test functions + 1 helper
- `src/tests/unit/test_hdf5.py` — 1 test function + 1 helper
- `src/tests/unit/test_p1_fixes.py` — 5 test functions + 3 helpers

**Fix**: Remove `return True/False` from test functions and helpers. Replace `return False` with `assert` or `pytest.fail()`. Remove try/except wrappers that swallow exceptions (pytest handles exception reporting). Update `if __name__ == "__main__"` blocks to use try/except for standalone execution.

### Issue 2: RuntimeWarning — Unawaited Coroutine `WebSocketManager.broadcast`

**Root Cause**: In `test_websocket_manager.py::test_broadcast_from_thread_with_loop`, `asyncio.run_coroutine_threadsafe` is mocked. The `broadcast_from_thread()` method calls `self.broadcast(message)` creating a coroutine object, then passes it to the mocked function. Since the mock doesn't schedule the coroutine, it is never awaited. When Python's GC collects the orphaned coroutine, it emits the RuntimeWarning. The warning surfaces during a later test (`test_cascade_correlation_coverage_final.py`) when GC happens to run.

**Affected File**: `src/tests/unit/api/test_websocket_manager.py` (lines 118–127)

**Fix**: After the mock assertion, retrieve the captured coroutine argument and close it explicitly to prevent the GC warning.

### Issue 3: UserWarning — Mismatched Tensor Sizes in `train_output_layer()`

**Root Cause**: `CascadeCorrelationNetwork.train_output_layer()` does not validate that `x` and `y` have the same batch size (dimension 0). When mismatched tensors are passed, PyTorch's `MSELoss` emits a UserWarning about broadcasting before ultimately raising a RuntimeError. The existing test `test_train_with_mismatched_sizes` expects an error, but the warning is emitted before the error.

**Affected Files**:
- `src/cascade_correlation/cascade_correlation.py` (line ~1256, `train_output_layer()`)
- `src/tests/unit/test_training_workflow.py` (test passes, but emits warning)

**Fix**: Add batch size validation to `train_output_layer()` immediately after the None check. Raise `ValueError` with a descriptive message if `x.shape[0] != y.shape[0]`. This prevents the warning and provides a clearer error message.

### Issue 4: Third-Party Deprecation/User Warnings (Cannot Fix at Source)

These warnings originate in third-party library code, not in juniper-cascor:

| Warning | Source | Count |
|---------|--------|-------|
| Starlette TestClient `timeout` DeprecationWarning | `starlette/testclient.py` | 16 |
| sentry_sdk `asyncio.iscoroutinefunction` DeprecationWarning | `sentry_sdk/integrations/` | 118 |
| matplotlib `FigureCanvasAgg` non-interactive UserWarning | `cascor_plotter.py` (plt.show) | 1 |

**Fix**: Replace the blanket `-p no:warnings` in `pytest.ini` with targeted `filterwarnings` entries for these specific third-party warnings. This enables warnings for our own code while suppressing unavoidable upstream noise.

### Issue 5: `pytest.ini` Blanket Warning Suppression

**Root Cause**: `pytest.ini` contains `-p no:warnings` which disables the entire pytest warnings plugin, hiding all warnings including actionable ones from our own code.

**Fix**: Remove `-p no:warnings` from `addopts`. Add a `filterwarnings` section with targeted filters for third-party warnings only.

---

## Implementation Plan

### Step 1: Fix PytestReturnNotNoneWarning (4 files)

For each affected test function:
1. Remove `return True` / `return False` statements
2. Remove try/except wrappers that swallow exceptions — let pytest handle failures
3. Preserve `try/finally` blocks where cleanup is needed (e.g., `os.cpu_count` restoration)
4. Convert helper function `return False` to `assert False` or `pytest.fail()`
5. Update `if __name__ == "__main__"` blocks to use try/except for pass/fail detection

### Step 2: Fix Unawaited Coroutine Warning (1 file)

In `test_websocket_manager.py::test_broadcast_from_thread_with_loop`:
1. After `mock_submit.assert_called_once()`, retrieve the coroutine from `mock_submit.call_args`
2. Call `.close()` on the coroutine to prevent the GC warning

### Step 3: Add Batch Size Validation to `train_output_layer()` (1 file)

In `cascade_correlation.py::train_output_layer()`:
1. After the `if x is None or y is None` check (line 1254–1255)
2. Add: `if x.shape[0] != y.shape[0]: raise ValueError(...)`
3. This eliminates the torch UserWarning and provides a clearer error

### Step 4: Update `pytest.ini` Warning Configuration (1 file)

1. Remove `-p no:warnings` from `addopts`
2. Add `filterwarnings` section with targeted filters:
   - `ignore::DeprecationWarning:starlette.testclient`
   - `ignore::DeprecationWarning:sentry_sdk`
   - `ignore:FigureCanvasAgg is non-interactive:UserWarning`

### Step 5: Run Full Test Suite

1. Run `pytest src/tests/ -v` with warnings enabled
2. Verify all 2124+ tests pass
3. Verify zero warnings from juniper-cascor code
4. Confirm only filtered third-party warnings remain suppressed

---

## Files Modified

| File | Change Type |
|------|-------------|
| `src/tests/unit/test_cascor_fix.py` | Remove return values, refactor helpers |
| `src/tests/unit/test_critical_fixes.py` | Remove return values, refactor helpers |
| `src/tests/unit/test_hdf5.py` | Remove return values, refactor helper |
| `src/tests/unit/test_p1_fixes.py` | Remove return values, refactor helpers |
| `src/tests/unit/api/test_websocket_manager.py` | Close unawaited coroutine |
| `src/cascade_correlation/cascade_correlation.py` | Add batch size validation |
| `src/tests/pytest.ini` | Replace warning suppression with targeted filters |

## Validation Criteria

- All existing tests pass (0 failures)
- No PytestReturnNotNoneWarning warnings
- No RuntimeWarning about unawaited coroutines
- No UserWarning about mismatched tensor sizes
- Third-party warnings properly filtered
- `if __name__ == "__main__"` standalone execution still works
