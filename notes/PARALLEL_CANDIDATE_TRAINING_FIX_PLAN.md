# Parallel Candidate Training Fix Plan

**Project**: Juniper-Cascor
**Issue**: Candidate training executes serially despite multiprocessing architecture
**Created**: 2026-03-17
**Status**: Complete - All Phases (1, 2, 3) Implemented and Verified

---

## Problem Statement

The juniper-cascor application is designed to train candidate neural network nodes in parallel using a multiprocessing process server with worker processes. However, at runtime, candidate training and evaluation execute serially. This violates the critical project requirement for parallel operation.

---

## Root Causes Identified

### RC-1: PyTorch Internal Thread Pool Contention (PRIMARY, Critical)

**Severity**: Critical - independently sufficient to serialize execution

Worker processes do not call `torch.set_num_threads(1)` or set `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` environment variables. PyTorch defaults to using ALL available CPU cores for its internal thread pool. When N worker processes each spawn M threads, the resulting N*M threads massively oversubscribe the available cores, causing context switching that effectively serializes execution.

Thread limiting is present in `src/tests/conftest.py:100-105` (for pytest-xdist only) but is completely absent from all production code paths: `main.py`, `_worker_loop()`, `train_candidate_worker()`, `_init_multiprocessing()`, and `candidate_unit.py`.

**Files affected**:
- `src/cascade_correlation/cascade_correlation.py` (_worker_loop, _init_multiprocessing)
- `src/main.py`
- `src/cascor_constants/constants_model/constants_model.py`
- `src/cascade_correlation/cascade_correlation_config/cascade_correlation_config.py`

### RC-2: Manager-Proxied Queue Serialization Bottleneck (PRIMARY, Critical)

**Severity**: Critical - independently sufficient to serialize execution

The architecture uses `BaseManager`-proxied queues (via `CandidateTrainingManager`). Every `put()` and `get()` is an RPC call through a Unix socket to a single-threaded manager server process. All data must be pickled, transmitted over IPC, and unpickled for every operation. The manager server processes requests sequentially, creating a serial bottleneck for all data transfer.

**Files affected**:
- `src/cascade_correlation/cascade_correlation.py` (CandidateTrainingManager, _execute_parallel_training)

### RC-3: Redundant Data Duplication in Task Payloads (CONTRIBUTING, High)

**Severity**: High - amplifies RC-2

Every task tuple contains the same `training_inputs` tuple (including torch tensors for `candidate_input`, `y`, `residual_error`). Through manager proxy queues, each reference becomes an independent full serialization. With `candidate_pool_size=32`, the same tensors are serialized 32 times.

**Files affected**:
- `src/cascade_correlation/cascade_correlation.py` (_generate_candidate_tasks)

### RC-4: Manager and Worker Lifecycle Overhead (CONTRIBUTING, Medium)

**Severity**: Medium - adds per-round latency

The manager server is started/stopped and worker processes are spawned/terminated for every call to `_execute_parallel_training()`. Since `train_candidates()` is called once per epoch in `grow_network()`, this overhead recurs for potentially hundreds of iterations. Each round requires: manager process creation, queue proxy init, N worker process spawning (with forkserver), PyTorch loading in each worker, 4-phase shutdown.

**Files affected**:
- `src/cascade_correlation/cascade_correlation.py` (_execute_parallel_training, _start_manager, _stop_manager)

---

## Implementation Plan

### Phase 1: PyTorch Thread Pinning (RC-1 Fix)

**Priority**: Immediate
**Estimated Impact**: 5-15x throughput improvement
**Risk**: Very low

#### Task 1.1: Add thread pinning to `_worker_loop()` entry point
- **File**: `src/cascade_correlation/cascade_correlation.py`
- **Location**: Start of `_worker_loop()` static method (after logger init, line ~2448)
- **Change**: Add `torch.set_num_threads(1)` and set OMP/MKL/OPENBLAS env vars
- **Rationale**: Each worker must limit its own PyTorch thread pool to prevent N*M oversubscription

#### Task 1.2: Add thread pinning to `main.py` startup
- **File**: `src/main.py`
- **Location**: Early in file, before torch import or BLAS library loading
- **Change**: Set OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS env vars
- **Rationale**: Env vars must be set before BLAS libraries are loaded to take effect

#### Task 1.3: Add configurable worker thread count
- **File**: `src/cascor_constants/constants_model/constants_model.py`
- **File**: `src/cascade_correlation/cascade_correlation_config/cascade_correlation_config.py`
- **Change**: Add `_PROJECT_MODEL_WORKER_THREAD_COUNT = 1` constant and corresponding config parameter
- **Rationale**: Allow tuning thread count per worker rather than hardcoding

#### Task 1.4: Verify with existing tests
- **Command**: `cd src/tests && bash scripts/run_tests.bash -u`
- **Expected**: All existing tests pass; no regressions

### Phase 2: Replace Manager Queues with Direct Queues (RC-2 + RC-3 Fix)

**Priority**: After Phase 1
**Estimated Impact**: 3-10x additional throughput improvement
**Risk**: Medium - requires careful pickling validation

#### Task 2.1: Replace BaseManager-proxied queues with direct multiprocessing.Queue
- **File**: `src/cascade_correlation/cascade_correlation.py`
- **Change**: Replace `CandidateTrainingManager` usage in `_execute_parallel_training()` with direct `mp_ctx.Queue()` instances
- **Rationale**: Direct queues use pipes (not sockets through a server process), eliminating the single-threaded manager bottleneck
- **Scope**: `_execute_parallel_training()`, `_start_manager()`, `_stop_manager()`

#### Task 2.2: Separate shared training data from per-candidate task payloads
- **File**: `src/cascade_correlation/cascade_correlation.py`
- **Change**: Pass shared training data (tensors) once via a separate mechanism instead of duplicating in every task tuple
- **Approach**: Use `multiprocessing.Queue` for tasks containing only per-candidate metadata; pass shared tensors as separate args to `_worker_loop()`
- **Rationale**: Eliminates N-fold redundant serialization of identical training tensors

#### Task 2.3: Verify Phase 2 changes with test suite
- **Command**: `cd src/tests && bash scripts/run_tests.bash -u`

### Phase 3: Persistent Worker Pool (RC-4 Fix)

**Priority**: After Phase 2
**Estimated Impact**: 20-50% latency reduction per round
**Risk**: Medium - state management across rounds

#### Task 3.1: Convert _execute_parallel_training to reuse a persistent worker pool
- **File**: `src/cascade_correlation/cascade_correlation.py`
- **Change**: Maintain worker processes across training rounds instead of spawning/terminating per round
- **Approach**: Lazy-initialize worker pool in `_init_multiprocessing()`; send tasks per round; keep workers alive with sentinel-on-shutdown-only
- **Rationale**: Eliminates per-round overhead of process creation, PyTorch init, and 4-phase shutdown

#### Task 3.2: Verify Phase 3 changes with test suite
- **Command**: `cd src/tests && bash scripts/run_tests.bash -u`

---

## Change Log

| Date | Phase | Change | Files Modified |
|------|-------|--------|----------------|
| 2026-03-17 | Plan | Initial plan document created | notes/PARALLEL_CANDIDATE_TRAINING_FIX_PLAN.md |
| 2026-03-17 | Phase 1 | Added PyTorch thread pinning to `_worker_loop()` with configurable `worker_thread_count` param | `cascade_correlation.py` |
| 2026-03-17 | Phase 1 | Set OMP/MKL/OPENBLAS env vars early in `main.py` before BLAS library loading | `main.py` |
| 2026-03-17 | Phase 1 | Added `_PROJECT_MODEL_WORKER_THREAD_COUNT` constant and cascor-level alias | `constants_model.py`, `constants.py` |
| 2026-03-17 | Phase 1 | Added `worker_thread_count` to `CascadeCorrelationConfig` constructor | `cascade_correlation_config.py` |
| 2026-03-17 | Phase 1 | Set parent process thread count in `_init_multiprocessing()` | `cascade_correlation.py` |
| 2026-03-17 | Phase 1 | All unit tests pass (95% coverage, 0 failures) | — |
| 2026-03-17 | Phase 2 | Expanded Phase 2 implementation plan | notes/PARALLEL_CANDIDATE_TRAINING_FIX_PLAN.md |
| 2026-03-17 | Phase 2 | Replaced BaseManager-proxied queues with direct `mp_ctx.Queue()` in `_execute_parallel_training()` | `cascade_correlation.py` |
| 2026-03-17 | Phase 2 | Separated shared training data from per-candidate task payloads; shared tensors passed once to `_worker_loop()` | `cascade_correlation.py` |
| 2026-03-17 | Phase 2 | Updated `_worker_loop()` to accept `shared_training_inputs` and reconstruct full tasks from lightweight queue messages | `cascade_correlation.py` |
| 2026-03-17 | Phase 2 | All unit tests pass (95% coverage, 0 failures) | — |
| 2026-03-17 | Phase 3 | Added persistent worker pool: `_persistent_workers`, `_persistent_task_queue`, `_persistent_result_queue`, `_persistent_pool_size` attrs in `_init_multiprocessing()` | `cascade_correlation.py` |
| 2026-03-17 | Phase 3 | Added `_ensure_worker_pool()` for lazy pool creation/reuse across training rounds | `cascade_correlation.py` |
| 2026-03-17 | Phase 3 | Added `_shutdown_worker_pool()` with sentinel-based graceful shutdown, terminate fallback, and SIGKILL escalation | `cascade_correlation.py` |
| 2026-03-17 | Phase 3 | Refactored `_execute_parallel_training()` to use `_ensure_worker_pool()` instead of per-round spawn/terminate | `cascade_correlation.py` |
| 2026-03-17 | Phase 3 | Added result queue size check in wait loop for early exit optimization | `cascade_correlation.py` |
| 2026-03-17 | Phase 3 | Excluded persistent pool attributes from `__getstate__` for pickle safety | `cascade_correlation.py` |
| 2026-03-17 | Phase 3 | Note: RC-3 shared_training_inputs optimization deferred in persistent pool mode (training data changes each round); full tasks sent through direct queue | — |
| 2026-03-17 | Phase 3 | All unit tests pass (93% coverage, 0 failures, 10 skipped) | — |

---

## Verification Strategy

1. **py-spy profiling** (`util/profile_training.bash`): Multiple worker PIDs should show CPU time
2. **htop**: All cores should show utilization during candidate training
3. **Wall-clock timing**: Log time for `train_candidates()` before/after changes
4. **Test suite**: `cd src/tests && bash scripts/run_tests.bash -m "multiprocessing"`
