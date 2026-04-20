# Fix: Resource Tracker KeyErrors and Test Warnings

**Date**: 2026-04-02
**Branch**: `fix/resource-tracker-and-test-warnings`
**Author**: Paul Calnon (with Claude Code)

---

## Summary

Four issues were identified in the juniper-cascor test suite output: multiprocessing resource tracker `KeyError` tracebacks after test completion, and three pytest warnings. All tests pass (2926 passed), but the post-run noise obscures clean test output.

---

## Issue 1: Resource Tracker KeyError Tracebacks

### Symptoms

After the test suite completes, 18 `KeyError` tracebacks from `multiprocessing/resource_tracker.py:374`:

```
KeyError: 'juniper_train_d1d8429f'   (×9)
KeyError: 'juniper_train_14b1f0c7'   (×9)
```

### Root Cause

The `SharedTrainingMemory.reconstruct_tensors()` method (OPT-5) manually calls `resource_tracker.unregister(shm.name, "shared_memory")` to prevent worker processes from prematurely unlinking shared memory on exit. However, there is a **name mismatch**:

- `SharedMemory.__init__` registers the internal name `self._name` which has a POSIX `/` prefix: `/juniper_train_xyz`
- `shm.name` (the public property) strips the `/` prefix, returning `juniper_train_xyz`
- `unregister(shm.name, ...)` sends an UNREGISTER message with the **wrong name** (no `/` prefix)
- The resource tracker daemon receives the message, tries `cache["shared_memory"].remove("juniper_train_xyz")`, but the cache contains `/juniper_train_xyz` → `KeyError`
- The error occurs in the daemon process (separate from our code), so our `try/except` wrapper never catches it

With 9 candidates per training round and 2 rounds, this produces 18 KeyError tracebacks.

### Fix

**File**: `src/cascade_correlation/cascade_correlation.py` (line ~317)

Replaced the manual `unregister()` approach with Python 3.13+'s `track=False` parameter on `SharedMemory`:

```python
# Before:
shm = SharedMemory(name=metadata["shm_name"], create=False)
try:
    try:
        from multiprocessing.resource_tracker import unregister
        unregister(shm.name, "shared_memory")
    except Exception:
        pass

# After:
shm = SharedMemory(name=metadata["shm_name"], create=False, track=False)
```

`track=False` prevents the worker-side `SharedMemory` from registering with the resource tracker at all, which is semantically correct since the main process owns the shared memory lifecycle.

---

## Issue 2: PytestBenchmarkWarning

### Symptom

```
PytestBenchmarkWarning: Benchmark fixture was not used at all in this test!
```

### Root Cause

`test_pickle_dumps_vs_loads_asymmetry` in `test_concurrency_scaling.py` accepted a `benchmark` fixture parameter but used the custom `BenchmarkTimer` class instead.

### Fix

**File**: `src/tests/performance/test_concurrency_scaling.py` (line 451)

Removed the unused `benchmark` parameter from the method signature.

---

## Issue 3: TypedStorage Deprecation Warning

### Symptom

```
UserWarning: TypedStorage is deprecated. It will be removed in the future...
```

### Root Cause

`test_zero_copy_verification` in `test_shared_memory.py` called `tensors[0].storage()` which returns the deprecated `TypedStorage`.

### Fix

**File**: `src/tests/performance/test_shared_memory.py` (line 134)

Replaced `.storage()` with `.untyped_storage()`.

---

## Issue 4: Unawaited Coroutine Warning

### Symptom

```
RuntimeWarning: coroutine 'WebSocketManager.broadcast' was never awaited
```

### Root Cause

`WebSocketManager.broadcast_from_thread()` created the coroutine inline inside `asyncio.run_coroutine_threadsafe(self.broadcast(message), self._event_loop)`. When the event loop closes between the `is_closed()` check and the submit call (a race condition during test teardown), `RuntimeError` is raised and caught, but the coroutine object is garbage-collected without being awaited, triggering the warning.

### Fix

**File**: `src/api/websocket/manager.py` (line ~89)

Create the coroutine before the try block and explicitly `coro.close()` it in the exception handler:

```python
coro = self.broadcast(message)
try:
    asyncio.run_coroutine_threadsafe(coro, self._event_loop)
except RuntimeError:
    coro.close()  # Prevent "coroutine was never awaited" warning
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/cascade_correlation/cascade_correlation.py` | Use `track=False` for worker-side SharedMemory, remove manual unregister |
| `src/api/websocket/manager.py` | Close coroutine on failed `run_coroutine_threadsafe` submit |
| `src/tests/performance/test_concurrency_scaling.py` | Remove unused `benchmark` fixture parameter |
| `src/tests/performance/test_shared_memory.py` | Use `untyped_storage()` instead of deprecated `storage()` |

## Validation

- All 4 fixes validated by sub-agent code review
- Resource tracker fix confirmed: no other `SharedMemory(create=False)` callsites need `track=False`
- `TestResourceTrackerNoPrematureUnlink` test exercises the new `track=False` path
- Full test suite run pending
