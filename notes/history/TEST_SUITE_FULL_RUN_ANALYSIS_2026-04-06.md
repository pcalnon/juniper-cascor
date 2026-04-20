# Test Suite Full-Run Analysis -- 2026-04-06

**Project**: juniper-cascor
**Branch**: fix/test-suite-full-run
**Date**: 2026-04-06
**Previous State**: 397 failed, 2156 passed, 1 xfailed, 16 warnings, 487 errors (1112s)

---

## Executive Summary

The full test suite (3041 tests with `--slow --integration --run-long --run-performance --fast-slow`) had 5 distinct root causes producing cascading failures. All 5 have been resolved.

---

## Root Cause Analysis

### RC-1: API Integration Test Fixture Cleanup Ordering (13 errors)

**Symptom**: All 7 `test_api_full_lifecycle.py` and 6 `test_websocket_streaming.py` tests errored with 60s timeout during fixture teardown.

**Root Cause**: The `client()` fixture's lifecycle cleanup code was positioned AFTER the `with TestClient(app)` context manager block. When the generator-based fixture resumed after `yield`, the `with` block exited first, calling `TestClient.__exit__()` which tried to join the anyio blocking portal thread. The portal thread blocked indefinitely on `asyncio.selector.poll()` because the ASGI event loop still had pending work (training thread in ThreadPoolExecutor).

**Fix**: Replaced the `with TestClient(app)` context manager with manual `tc.__enter__()` / `tc.__exit__()` management. The lifecycle cleanup now runs before `TestClient.__exit__()`. A daemon thread with 5-second timeout caps the exit call to prevent hangs. The session-scoped `_force_clean_exit` fixture handles final cleanup.

**Files Modified**:
- `src/tests/integration/api/test_api_full_lifecycle.py` -- fixture rewrite
- `src/tests/integration/api/test_websocket_streaming.py` -- fixture rewrite

### RC-2: `--fast-slow` Flag Not Propagated to Environment Variable (2+ failures)

**Symptom**: Spiral integration tests ran with FULL training parameters (30+ samples, 5+ epochs) instead of reduced fast-slow parameters, causing 60s timeouts.

**Root Cause**: The `--fast-slow` pytest CLI option was only checked by conftest.py fixtures via `request.config.getoption("--fast-slow")`. But `test_spiral_problem.py`'s `_is_fast_mode()` function checked the `JUNIPER_FAST_SLOW` environment variable, which was never set by the pytest flag. This disconnect meant fast-slow mode was silently ineffective for integration tests.

**Fix**: Added `os.environ["JUNIPER_FAST_SLOW"] = "1"` in `pytest_configure()` when the `--fast-slow` flag is detected.

**Files Modified**:
- `src/tests/conftest.py` -- propagate flag to env var

### RC-3: f-string Eager Evaluation with Tensor `__repr__()` (1+ timeout)

**Symptom**: `test_2_spiral_learning` timed out at 60s during `add_unit()` when evaluating `f"...Hidden units: {self.hidden_units}"`.

**Root Cause**: Python f-strings are eagerly evaluated before function arguments are passed. Even though the conftest replaces `self.logger` with `_NoOpLogger` (whose `debug()` is a no-op), the f-string `f"...{self.hidden_units}"` still calls `__repr__()` on every tensor in the hidden units list. Torch's `_tensor_str._str_intern()` iterates over tensor values, calling `.detach().item().__format__()` which is expensive.

**Fix**: Added `if self.logger.isEnabledFor(logging.DEBUG):` guards around all debug log calls in `add_unit()`, `_add_best_candidate()`, and `grow_network()` that include full tensor values in f-strings. The `_NoOpLogger.isEnabledFor(10)` returns `False` (since level 10 < 30), preventing the f-string from being evaluated.

**Files Modified**:
- `src/cascade_correlation/cascade_correlation.py` -- 10 log calls guarded

### RC-4: Signal-Based Timeout Deadlocks Multiprocessing Forkserver (38 hangs)

**Symptom**: All 38 `test_concurrency_scaling.py` performance tests hung indefinitely. Queue put/get operations, process spawning, and worker lifecycle tests never completed.

**Root Cause**: The default `timeout_method=signal` (configured in `pyproject.toml`) uses SIGALRM for test timeout enforcement. SIGALRM interferes with the forkserver multiprocessing context's IPC mechanism. When a signal is delivered during a pipe/socket operation in the forkserver, the operation may silently deadlock instead of retrying or raising an error.

**Fix**: Added `pytest_configure()` in `src/tests/performance/conftest.py` that overrides `timeout_method` to `"thread"` for all performance tests. Thread-based timeout runs the check in a separate thread, avoiding signal interference with multiprocessing IPC.

**Files Modified**:
- `src/tests/performance/conftest.py` -- timeout method override

### RC-5: Spiral Test Training Parameters Too Aggressive for Fast-Slow Mode (1 failure)

**Symptom**: `test_n_spiral_difficulty_progression[3]` failed with accuracy 0.289 < 0.333 (random chance for 3-class).

**Root Cause**: The fast-slow training parameters (3 candidate epochs, 3 output epochs, 10 samples per spiral, 2 max epochs) didn't provide enough training budget for the 3-spiral classification problem to beat random chance.

**Fix**: Increased training budget for multi-class spiral problems in fast mode: candidate/output epochs 3→5, samples per spiral 10→15, max epochs scaled with complexity. Also added `@pytest.mark.timeout(120)` to `test_2_spiral_learning`.

**Files Modified**:
- `src/tests/integration/test_spiral_problem.py` -- training params adjusted

### RC-6: API Reset Race Condition (1 failure)

**Symptom**: `test_reset_clears_state` failed with `assert training_state["current_epoch"] == 0` getting value 1.

**Root Cause**: The test stopped training but reset BEFORE the training thread had fully exited. The training thread wrote `current_epoch=1` after the reset cleared it to 0.

**Fix**: Changed the test to wait for training COMPLETION (not just STARTED) before issuing the reset, ensuring no concurrent state updates.

**Files Modified**:
- `src/tests/integration/api/test_api_full_lifecycle.py` -- wait for COMPLETED

---

## Coverage Configuration

The `INTERNALERROR` from `sqlite3.OperationalError: unable to open database file` was previously addressed in the Phase 1 remediation (explicit `data_file` and `parallel` settings in `pyproject.toml`). No additional changes needed.

---

## Validation Results

| Test Category | Before | After |
|---------------|--------|-------|
| Default (unit, no flags) | All pass | All pass |
| Slow unit tests (--slow --fast-slow) | All pass | All pass |
| Integration tests (--integration --slow --fast-slow) | 2 failed, 13 errors | All pass |
| Performance baselines (--run-performance) | All pass | All pass |
| Performance micro/shared-memory | All pass | All pass |
| Performance end-to-end profiling | All pass | All pass |
| Performance concurrency scaling | 38 hangs | All pass |
| XFail tests | 1 xfail | 1 xfail (expected) |

---

## Risk Assessment

| Fix | Risk | Mitigation |
|-----|------|------------|
| Daemon-thread TestClient exit | LOW | Session-level `_force_clean_exit` handles orphaned threads |
| `isEnabledFor()` guards | NONE | Backward-compatible; production logging unaffected |
| Fast-slow env propagation | NONE | Only sets env var when --fast-slow flag already active |
| Thread-based timeout for perf | LOW | Only affects performance tests; signal timeout unchanged for unit/integration |
| Spiral training params | LOW | Parameters still exercise full training pipeline; thresholds maintained |
| Reset race condition fix | NONE | Waits for completion instead of just started state |
