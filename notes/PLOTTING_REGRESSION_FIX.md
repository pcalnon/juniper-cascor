# Plotting Regression Fix - Multiprocessing Pickle Error

**Date**: 2025-10-27  
**Issue**: AttributeError when plotting decision boundary  
**Status**: ✓ FIXED

---

## Problem

```bash
AttributeError: Can't get local object 'CascadeCorrelationNetwork.plot_decision_boundary.<locals>._plot_worker'
```

**Stack Trace Location**:

- File: `cascade_correlation.py`
- Method: `plot_decision_boundary()`
- Line: 3395 (`plot_process.start()`)

---

## Root Cause

The plotting methods (`plot_decision_boundary` and `plot_training_history`) defined local worker functions (`_plot_worker`) inside the method bodies. When using the forkserver multiprocessing context (set in `_init_multiprocessing()`), these local functions cannot be pickled for transfer to worker processes.

**Why It Failed**:

1. `_init_multiprocessing()` sets `self._mp_ctx = mp.get_context('forkserver')`
2. Plotting code used `self._mp_ctx.Process(target=_plot_worker, ...)`
3. Forkserver requires all targets to be picklable
4. Local/nested functions are NOT picklable in Python
5. Process start attempted to pickle `_plot_worker` → AttributeError

---

## Solution

**Two-Part Fix**:

### 1. Move Worker Functions to Module Level

Created two module-level functions that are picklable:

```python
def _plot_decision_boundary_worker(network, x_data, y_data, title_str):
    """
    Worker function to create decision boundary plot in separate process.
    
    This function must be at module level to be picklable for multiprocessing.
    """
    from cascor_plotter.cascor_plotter import CascadeCorrelationPlotter
    plotter = CascadeCorrelationPlotter()
    plotter.plot_decision_boundary(network, x_data, y_data, title_str)


def _plot_training_history_worker(history_data):
    """
    Worker function to create training history plot in separate process.
    
    This function must be at module level to be picklable for multiprocessing.
    """
    from cascor_plotter.cascor_plotter import CascadeCorrelationPlotter
    plotter = CascadeCorrelationPlotter()
    plotter.plot_training_history(history_data)
```

**Location**: Added after `CandidateTrainingManager` registration (line ~238)

### 2. Use Spawn Context for Plotting

Changed plotting code to explicitly use 'spawn' context instead of inheriting forkserver:

```python
# BEFORE (BROKEN):
plot_process = self._mp_ctx.Process(
    target=_plot_worker,  # Local function - not picklable!
    args=(self, x, y, title),
    daemon=True,
    name="PlotDecisionBoundary"
)

# AFTER (FIXED):
plot_ctx = mp.get_context('spawn')
plot_process = plot_ctx.Process(
    target=_plot_decision_boundary_worker,  # Module-level - picklable!
    args=(self, x, y, title),
    daemon=True,
    name="PlotDecisionBoundary"
)
```

---

## Why This Works

1. **Module-Level Functions Are Picklable**: Functions defined at module scope can be pickled because Python can reference them by module path (`cascade_correlation._plot_decision_boundary_worker`)

2. **Spawn Context Isolates Plotting**: Using a separate 'spawn' context for plotting avoids conflicts with the forkserver context used for candidate training

3. **No Shared State Issues**: Plotting processes don't need to share state with candidate training workers, so separate context is safe

---

## Alternative Solutions Considered

### Option A: Use Spawn for Everything

**Rejected**: Forkserver is better for candidate training (faster, cleaner)

### Option B: Make Methods Instead of Functions

**Rejected**: Methods still need pickling, and class instance must be pickled

### Option C: Use Threading Instead

**Rejected**: Matplotlib/plotting can have thread-safety issues, process isolation better

### Option D: Remove Async Plotting

**Rejected**: User wants non-blocking plotting during training

---

## Impact

### Fixed ✓

- Decision boundary plotting works
- Training history plotting works
- No more AttributeError on multiprocessing start
- Both sync and async plotting modes functional

### Not Changed

- Candidate training still uses forkserver context (optimal)
- Existing API unchanged (backward compatible)
- Synchronous plotting still works (fallback mode)

---

## Testing

### Manual Test

```bash
cd src/prototypes/cascor
python3 src/cascor.py
```

Expected: Network trains and generates decision boundary plot without error

### Verification

- Check that plot process starts without AttributeError
- Verify PID logged: "Started plotting process PID: XXXX"
- Confirm plot image generated

---

## Files Modified

- **cascade_correlation.py**:
  - Added module-level `_plot_decision_boundary_worker()`
  - Added module-level `_plot_training_history_worker()`
  - Modified `plot_decision_boundary()` to use spawn context + module function
  - Modified `plot_training_history()` to use spawn context + module function

---

## Code Pattern for Future

**When adding async processing with multiprocessing**:

✓ **DO**: Define worker functions at module level
✓ **DO**: Use descriptive function names (not nested lambdas)
✓ **DO**: Document why function is at module level
✓ **DO**: Use appropriate context (spawn vs forkserver)

✗ **DON'T**: Define worker functions as local/nested functions
✗ **DON'T**: Use lambda functions as multiprocessing targets
✗ **DON'T**: Assume all contexts handle pickling the same way

---

## Related Issues

This fix addresses one of the common pitfalls in Python multiprocessing:

**Python Multiprocessing Contexts**:

- **fork**: Fastest, not safe on macOS (deprecated), copies full state
- **spawn**: Safest, slowest, requires picklable targets
- **forkserver**: Middle ground, requires picklable targets, used for candidate training

**Pickling Requirements**:

- Module-level functions: ✓ Picklable
- Class methods: ✓ Picklable (with instance)
- Nested/local functions: ✗ NOT picklable
- Lambda functions: ✗ NOT picklable
- Closures: ✗ NOT picklable (capture local scope)

---

## Lessons Learned

1. **Multiprocessing contexts have different requirements** - forkserver and spawn both require picklable targets

2. **Local functions break pickling** - Always define worker functions at module or class level

3. **Use separate contexts for different purposes** - Candidate training (forkserver) vs plotting (spawn) can coexist

4. **Document pickling requirements** - Add comments explaining why functions are at module level

---

## References

- Python multiprocessing documentation: <https://docs.python.org/3/library/multiprocessing.html>
- Pickle limitations: <https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled>
- Forkserver context: <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>
