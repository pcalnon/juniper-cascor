# OPT-5: Shared Memory Training Tensors — Implementation Plan

**Author**: Claude (with Paul Calnon)
**Date**: 2026-04-01
**Status**: Plan — awaiting approval
**Risk Level**: Medium-High
**Estimated Effort**: Medium (3-5 days implementation + 1-2 days validation)

---

## 1. Executive Summary

OPT-5 eliminates redundant per-task tensor serialization in parallel candidate training
by sharing training tensors across worker processes via named POSIX shared memory blocks.
Currently, each of N candidates (8-32 per round) sends the same training tensors through
the queue — even though PyTorch's ForkingPickler already transmits handles (~340 bytes)
rather than raw data, the GET-side tensor reconstruction from shared memory handles costs
~320us (same-process) to ~9ms (cross-process) per task. For 16 candidates, this amounts
to ~100-145ms of queue overhead per training round.

The recommended approach uses `multiprocessing.shared_memory.SharedMemory` to create a
named memory block each round, then sends lightweight tasks (74 bytes vs 561 bytes)
containing only the block name and candidate-specific data. Workers attach to the block
by name, construct zero-copy tensor views, and process candidates with no tensor
reconstruction overhead. This eliminates 97% of GET-side queue overhead.

**Expected improvement**: 5-20% total round time reduction, scaling with candidate pool
size and inversely with training duration. Largest impact on short-training, large-pool
configurations.

---

## 2. Problem Analysis

### 2.1 Current Data Flow

```
train_candidates()
  → _prepare_candidate_input()        → candidate_input  [n_samples × features]
  → _generate_candidate_tasks()        → N tasks, each containing:
      (candidate_index, candidate_data, training_inputs, round_id)
      where training_inputs = (candidate_input, epochs, y, residual_error, lr, freq)
  → _execute_parallel_training()
      → task_queue.put(task) × N       ← BOTTLENECK: N redundant tensor transmissions
      → workers process tasks
      → result_queue.get() × N
```

Each of the N candidate tasks contains **identical** training tensors. The same
`candidate_input`, `y`, and `residual_error` are serialized N times.

### 2.2 Why It's Not as Bad as Expected (But Still Worth Fixing)

PyTorch registers custom reducers with Python's `ForkingPickler` (used by
`multiprocessing.Queue`). These reducers automatically move tensors to POSIX shared
memory (`/dev/shm`) and transmit only file descriptor handles. This means:

| Metric | Expected (naive pickle) | Actual (ForkingPickler) |
|--------|------------------------|------------------------|
| Per-task payload | 1.6 MB (xlarge) | 561 bytes (handle) |
| 16 tasks total | 25.6 MB | 8,976 bytes |
| PUT latency | High | 0.4 ms (16 puts) |

**The data copy problem is already solved.** However, the **GET-side reconstruction**
from shared memory handles remains expensive:

| Operation | Same-process | Cross-process (forkserver) |
|-----------|-------------|---------------------------|
| Full task GET (with tensors) | ~320 us | ~9 ms |
| Lightweight task GET (no tensors) | ~19 us | ~0.5 ms |
| 16 full gets | 10.1 ms | ~145 ms |
| 16 lightweight gets | 0.3 ms | ~8 ms |

### 2.3 Benchmark Data (Measured 2026-04-01)

**Pickle serialization overhead by dataset size:**

| Config | Samples | Features | Outputs | Raw Tensor MB | Pickle MB | Dumps ms | Loads ms | 16-task MB |
|--------|---------|----------|---------|---------------|-----------|----------|----------|------------|
| small | 100 | 10 | 2 | 0.006 | 0.007 | 0.84 | 0.94 | 0.12 |
| medium | 500 | 20 | 2 | 0.048 | 0.050 | 0.83 | 1.08 | 0.80 |
| large-2D | 1000 | 50 | 2 | 0.216 | 0.218 | 1.20 | 1.37 | 3.49 |
| xlarge-10D | 5000 | 60 | 10 | 1.600 | 1.602 | 3.70 | 2.08 | 25.63 |

Note: Above uses `pickle.dumps()`. The Queue's `ForkingPickler` sends only 340-561 bytes
regardless of tensor size.

**Queue PUT/GET timing (16 tasks, same process):**

| Metric | Full task (tensors) | Lightweight (no tensors) |
|--------|-------------------|--------------------------|
| 16× PUT total | 0.407 ms | 0.192 ms |
| 16× GET total | 10.071 ms | 0.301 ms |
| GET overhead from tensors | 9.770 ms (97.0%) | — |

**First PUT vs subsequent PUTs (same tensor reference):**
- First PUT: 390 us (ForkingPickler shm-move)
- Subsequent PUTs: ~1 us (handle reuse)

**Cross-process queue overhead (4 workers, forkserver):**

| Config | Full 16-task total | With share_memory_() |
|--------|-------------------|---------------------|
| small | 95 ms | 100 ms (worse — shm setup overhead > benefit) |
| xlarge-10D | 149 ms | 109 ms (27% reduction) |

### 2.4 Thread Safety Validation

**Training tensors are read-only during candidate training.** Validated across all code
paths:

- `train_detailed()`: Passes `x` and `residual_error` to `forward()` and
  `_calculate_correlation()` — both read-only
- `forward()`: `x.unsqueeze(0)` is a local rebinding, not in-place mutation
- `_calculate_correlation()`: `flatten()`, `mean()`, `torch.dot()`,
  `torch.linalg.norm()` — all return new tensors
- `_update_weights_and_bias()`: Only mutates `self.weights` and `self.bias` (private
  per-candidate); `requires_grad_(True)` is only called on cloned weight/bias copies
- `backward()`: Computes gradients only for `weights_param` and `bias_param`; shared
  tensors have no `requires_grad` flag
- No in-place operations (`add_()`, `mul_()`, etc.) found on shared tensors

**Conclusion**: Multiple workers can safely read the same shared tensors simultaneously.
No locks or copies required.

---

## 3. Approaches Evaluated

### 3.1 Approach A: Pre-share Tensors (Minimal Change)

Call `share_memory_()` on training tensors before `_generate_candidate_tasks()`. The
ForkingPickler detects they're already in shared memory and skips the shm-move step.

| Aspect | Assessment |
|--------|-----------|
| Complexity | Very low (3-5 lines) |
| Benefit | Negligible — ForkingPickler already sends handles regardless |
| Risk | None |
| Verdict | **Rejected** — measured 0% pickle size difference; shm-move is cached after first put |

### 3.2 Approach B: Round Header Protocol (Queue-Based Broadcast)

Send N_workers "round header" messages containing training data via the task queue, then
send N lightweight tasks. Workers cache the header's training data for the round.

| Aspect | Assessment |
|--------|-----------|
| Complexity | Medium |
| Benefit | Reduces GET ops with tensors from N to N_workers |
| Risk | **HIGH** |
| Verdict | **Rejected** — two fundamental problems |

**Fatal flaw 1 — Race condition**: A single FIFO queue cannot guarantee each worker gets
exactly one header before any lightweight task. A fast worker could consume two headers
while another gets none.

**Fatal flaw 2 — Worker death**: If a worker dies mid-round, its cached data is lost.
Replacement workers (from `_ensure_worker_pool` recreation) have no way to get the
current round's training data.

### 3.3 Approach C: Named SharedMemory + Lightweight Tasks (Recommended)

Use `multiprocessing.shared_memory.SharedMemory` to create a named memory block each
round. Workers attach by name and construct zero-copy tensor views via numpy.

| Aspect | Assessment |
|--------|-----------|
| Complexity | Medium |
| Benefit | Eliminates 97% of GET-side tensor reconstruction overhead |
| Risk | Medium (lifecycle management) |
| Verdict | **Recommended** |

**Why this avoids Approach B's problems:**
- No ordering concern — workers attach by name, not from queue
- No cache loss on worker death — workers re-attach by name on next task
- No stale cache — block name changes each round (includes round_id)
- Sequential fallback works unchanged (full tasks still self-contained)

---

## 4. Recommended Design

### 4.1 Architecture

```
MAIN PROCESS                          WORKER PROCESSES
─────────────                         ─────────────────

1. Create SharedMemory block          
   - Name: f"juniper_train_{round_id[:8]}"
   - Size: candidate_input + y + residual_error + metadata header
   - Copy tensor data into block

2. Submit N lightweight tasks:        
   task_queue.put((                   3. Worker receives task
     idx,                                - Extracts shm_name from task
     candidate_data,                     - Attaches: SharedMemory(name=shm_name)
     round_id,                           - Reconstructs tensors via numpy view:
     shm_metadata                          np.ndarray(shape, dtype, buffer=shm.buf[offset:])
   ))                                      torch.from_numpy(shared_np)  # zero-copy
                                         - Trains candidate with shared tensors
                                         - Detaches: shm.close()
                                         - Puts result in result_queue

4. Collect results                    

5. Cleanup:
   - shm.close()
   - shm.unlink()
```

### 4.2 SharedMemory Block Layout

```
┌──────────────────────────────────────────────────┐
│ Header (64 bytes, fixed)                         │
│  - magic: b"JNPR" (4 bytes)                     │
│  - version: uint8 (1 byte)                       │
│  - n_tensors: uint8 (1 byte)                     │
│  - reserved: (58 bytes)                          │
├──────────────────────────────────────────────────┤
│ Tensor Descriptor Table (32 bytes × n_tensors)   │
│  Per tensor:                                     │
│   - offset: uint64 (8 bytes)                     │
│   - nbytes: uint64 (8 bytes)                     │
│   - ndim: uint8 (1 byte)                         │
│   - dtype_code: uint8 (1 byte)                   │
│   - shape[0]: uint32 (4 bytes)                   │
│   - shape[1]: uint32 (4 bytes)                   │
│   - reserved: (6 bytes)                          │
├──────────────────────────────────────────────────┤
│ Tensor 0: candidate_input (contiguous float32)   │
├──────────────────────────────────────────────────┤
│ Tensor 1: y (contiguous float32)                 │
├──────────────────────────────────────────────────┤
│ Tensor 2: residual_error (contiguous float32)    │
└──────────────────────────────────────────────────┘
```

### 4.3 Metadata Passed in Lightweight Task

Instead of the full `training_inputs` tuple, each task carries:

```python
shm_metadata = {
    "shm_name": str,              # SharedMemory block name
    "candidate_epochs": int,       # Scalar — cheap to serialize
    "candidate_learning_rate": float,
    "candidate_display_frequency": int,
}
```

Total serialized size: ~150 bytes (vs 561 bytes for full task with ForkingPickler handles).

### 4.4 Task Format Changes

```python
# Current (4-tuple):
(candidate_index, candidate_data, training_inputs, round_id)
#                                 ^^^^^^^^^^^^^^^^
#                                 Contains 3 tensors + 3 scalars

# Proposed (4-tuple, same shape):
(candidate_index, candidate_data, shm_metadata, round_id)
#                                 ^^^^^^^^^^^^
#                                 Contains shm_name + 3 scalars (no tensors)
```

The 4-tuple structure is preserved for RC-5 compatibility. `_build_candidate_inputs()`
unpacks position [2] as training_inputs — with the new format, it unpacks
`shm_metadata` instead and reconstructs the tensors from SharedMemory.

### 4.5 Worker-Side Tensor Reconstruction

**CRITICAL**: `torch.from_numpy()` on a SharedMemory-backed numpy array creates a
zero-copy view. The `SharedMemory` handle MUST remain open for the entire duration of
tensor use. Closing the handle invalidates the memory mapping and causes segfaults.

The reconstruction function returns both the training inputs AND the SharedMemory handle.
The caller (`_process_worker_task`) must close the handle after training completes.

```python
def _reconstruct_training_tensors(shm_metadata: dict) -> tuple:
    """Reconstruct training tensors from SharedMemory block.

    Returns:
        (training_inputs_tuple, shm_handle). Caller MUST keep shm_handle alive
        until all tensor operations complete, then call shm_handle.close().
    """
    shm = SharedMemory(name=shm_metadata["shm_name"])

    header = shm.buf[:64]
    assert header[:4] == b"JNPR", "Invalid SharedMemory block header"

    # Read tensor descriptors and reconstruct as zero-copy views
    tensors = []
    for i in range(n_tensors):
        desc = _read_descriptor(shm.buf, 64 + i * 32)
        np_array = np.ndarray(
            shape=desc.shape, dtype=desc.dtype,
            buffer=shm.buf[desc.offset:desc.offset + desc.nbytes]
        )
        tensors.append(torch.from_numpy(np_array))  # Zero-copy view into shm

    candidate_input, y, residual_error = tensors
    training_inputs = (
        candidate_input,
        shm_metadata["candidate_epochs"],
        y,
        residual_error,
        shm_metadata["candidate_learning_rate"],
        shm_metadata["candidate_display_frequency"],
    )
    # DO NOT close shm here — tensor views reference its buffer.
    # Return handle so caller can close after training completes.
    return training_inputs, shm
```

The corresponding change in `_process_worker_task()`:

```python
@staticmethod
def _process_worker_task(task, shared_training_inputs, progress_queue, result_queue, parallel, logger):
    shm_handle = None  # OPT-5: track SharedMemory handle for cleanup
    try:
        # RC-3 / OPT-5: Reconstruct full task
        if shared_training_inputs is not None and len(task) == 2:
            full_task = (task[0], task[1], shared_training_inputs)
        else:
            full_task = task

        # ... existing training code ...
        result = CascadeCorrelationNetwork.train_candidate_worker(...)
        result_queue.put(result, timeout=30)
    finally:
        # OPT-5: Close SharedMemory handle after training completes
        if shm_handle is not None:
            try:
                shm_handle.close()
            except Exception:
                pass
```

Note: `shm_handle` is set by `_build_candidate_inputs()` when it detects dict-type
training inputs and calls `_reconstruct_training_tensors()`. The handle is passed up
through the call chain to `_process_worker_task()` for cleanup.

### 4.6 Lifecycle Management

```
Round N starts:
  1. Main creates SharedMemory block: shm_N
  2. Main copies training tensors into shm_N
  3. Main submits N lightweight tasks referencing shm_N
  4. Workers attach shm_N by name, hold handle open during training, close after
  5. Main collects all N results
  6. Main unlinks shm_N (in finally block — runs even on error/interrupt)

Round N+1 starts:
  1. Main creates new SharedMemory block: shm_N+1
  ... (same cycle)
```

**Worker handle lifecycle**: Workers open the SharedMemory handle in
`_build_candidate_inputs()` and close it in `_process_worker_task()`'s `finally` block.
The handle must remain open during the entire `train_candidate_worker()` call because
the zero-copy tensor views reference the mapped memory.

**Cleanup on error/shutdown**: `_shutdown_worker_pool()` must also unlink any
outstanding SharedMemory blocks. Track active blocks in
`self._active_shm_blocks: list[SharedTrainingMemory]`.

**Critical: Cleanup must be in `finally` block** of `_execute_parallel_training()`, not
just after result collection. This ensures cleanup runs even on `KeyboardInterrupt` or
exceptions during task submission.

**Signal handling**: Register `atexit` handler to unlink any leaked blocks:

```python
import atexit
atexit.register(self._cleanup_shared_memory)
```

**Python 3.12 resource tracker caveat**: On Python 3.12, when a worker opens
`SharedMemory(name=..., create=False)`, the child process's resource tracker registers
the block. If the worker exits cleanly, the tracker may call `shm_unlink()` — destroying
the block while other workers or the main process still need it. Mitigation:

```python
# In _reconstruct_training_tensors(), after opening:
from multiprocessing.resource_tracker import unregister
unregister(shm._name, "shared_memory")  # Prevent worker's tracker from unlinking
```

This is safe because the main process exclusively owns the unlink lifecycle.

### 4.7 Fallback Behavior

| Scenario | Behavior |
|----------|----------|
| Sequential training | Tasks still contain `shm_metadata` dict; `_build_candidate_inputs()` handles both formats transparently. No IPC benefit but no correctness issue. |
| Worker death mid-round | New worker attaches to SharedMemory by name — no data loss |
| SharedMemory creation fails | Fall back to full tasks with tensor tuple (current behavior) |
| Remote workers | Cannot access local `/dev/shm`; `_execute_candidate_training()` reconstructs full tensors for remote dispatch |
| Python < 3.8 | Not applicable (project requires >=3.12) |
| Docker /dev/shm too small | Training tensors are small (< 50 MB); 64 MB default is sufficient for most cases |
| Non-contiguous tensors | Assert `.is_contiguous()` or call `.contiguous()` before writing to SharedMemory block |

---

## 5. Strengths, Weaknesses, Risks, and Guardrails

### 5.1 Strengths

1. **Zero-copy reads**: Workers read training tensors directly from `/dev/shm` via
   `numpy.ndarray(buffer=shm.buf)` + `torch.from_numpy()`. No deserialization, no
   memory allocation per worker.
2. **No ordering constraints**: Workers attach by name — no race conditions, no
   barriers required. Any worker can attach at any time.
3. **Worker death resilient**: SharedMemory blocks persist in `/dev/shm` until
   explicitly unlinked. Replacement workers attach by name — no cache to lose.
4. **Compatible with RC-4 persistent pool**: No changes to pool creation or worker
   startup. SharedMemory is orthogonal to pool lifecycle.
5. **Compatible with RC-5 round tagging**: 4-tuple task structure preserved; round_id
   filtering works unchanged.
6. **Graceful degradation**: Falls back to full tasks if SharedMemory creation fails.
7. **Thread-safe**: Training tensors are read-only in all worker code paths (validated).

### 5.2 Weaknesses

1. **Platform-specific**: `multiprocessing.shared_memory` uses POSIX `shm_open()` on
   Linux. Works on all target platforms (Linux, macOS) but behavior differs.
2. **Manual lifecycle**: Must explicitly `close()` and `unlink()`. PyTorch's
   `share_memory_()` has automatic cleanup via `torch_shm_manager` daemon; stdlib
   SharedMemory does not.
3. **Data copy on creation**: Training tensors must be copied from process memory into
   the SharedMemory block once per round. For xlarge datasets this costs ~1-5 ms.
4. **Contiguous requirement**: Tensors must be contiguous for direct buffer mapping.
   Non-contiguous tensors need `.contiguous()` first (additional copy).
5. **No CUDA support**: SharedMemory is CPU-only. Not a concern for juniper-cascor
   (CPU training), but limits future GPU portability.

### 5.3 Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| SharedMemory leak (crash before unlink) | Medium | Low | `atexit` handler + `/dev/shm` monitoring in tests |
| Worker reads partially-written block | High | Very Low | Write block completely before submitting any tasks (sequential guarantee) |
| `/dev/shm` exhaustion | Medium | Very Low | Max block size < 50 MB; system has 47 GB `/dev/shm` |
| Non-contiguous tensor passed | Low | Low | Assert `.is_contiguous()` or call `.contiguous()` before copy |
| RestrictedUnpickler rejects new types | Low | Medium | `shm_metadata` is a plain dict — no new pickle types needed |
| Remote workers (Phase 1b) can't access local /dev/shm | High | Medium | Remote workers fall back to full tasks (separate code path) |
| Python resource tracker prematurely unlinks | Medium | Low | Use `track=False` on Python 3.13+ in workers; on 3.12, the creator process tracks |

### 5.4 Guardrails

1. **Magic number validation**: Workers verify `b"JNPR"` header before reading tensors.
   Prevents misinterpreting corrupted or wrong-round blocks.
2. **Round-scoped blocks**: Block names include `round_id[:8]` — stale references to
   old rounds will fail with `FileNotFoundError` (block already unlinked).
3. **Fallback on error**: Any `SharedMemory` error in task creation falls back to full
   tasks for that round. No training data loss.
4. **Active block tracking**: `self._active_shm_blocks` list tracks all created blocks.
   `_shutdown_worker_pool()` and `atexit` handler unlink all tracked blocks.
5. **Size validation**: Computed block size must match sum of tensor sizes + header.
   Mismatch raises `ValueError` before writing.
6. **Integration test**: New test `test_shared_memory_training_round()` validates
   end-to-end with actual workers and SharedMemory.

---

## 6. Expected Performance Improvement

### 6.1 Queue Overhead Reduction

| Metric | Current | With OPT-5 | Improvement |
|--------|---------|-----------|-------------|
| GET-side tensor reconstruction (16 tasks, same-proc) | 10.1 ms | 0.3 ms | 97% |
| GET-side (16 tasks, cross-proc forkserver) | ~145 ms | ~8 ms | 94% |
| PUT-side (16 tasks) | 0.4 ms | 0.2 ms | 50% |
| SharedMemory setup + copy (per round) | 0 ms | 1-5 ms | (new cost) |
| **Net queue overhead (cross-proc)** | **~145 ms** | **~13 ms** | **91%** |

### 6.2 Total Round Time Impact

Queue overhead as a fraction of total round time depends on training duration:

| Training Duration | Queue Overhead (current) | Queue Overhead (OPT-5) | Round Improvement |
|-------------------|--------------------------|------------------------|-------------------|
| 0.5 s (short, few epochs) | 29.0% | 2.6% | **~26%** |
| 2.0 s (typical) | 7.3% | 0.7% | **~6.6%** |
| 5.0 s (long, many epochs) | 2.9% | 0.3% | **~2.7%** |
| 10.0 s (very long) | 1.5% | 0.1% | **~1.3%** |

### 6.3 Scaling with Pool Size

| Pool Size | Current Queue Overhead | OPT-5 Queue Overhead | Savings |
|-----------|----------------------|---------------------|---------|
| 8 candidates | ~72 ms | ~9 ms | 63 ms |
| 16 candidates | ~145 ms | ~13 ms | 132 ms |
| 32 candidates | ~290 ms | ~21 ms | 269 ms |

### 6.4 Memory Impact

| Config | Current (per-worker copies) | OPT-5 (one shared block) | Reduction |
|--------|-----------------------------|--------------------------|-----------|
| small (100×10×2) | 4 × 6 KB = 24 KB | 6 KB | 75% |
| large-2D (1000×50×2) | 4 × 216 KB = 864 KB | 216 KB | 75% |
| xlarge-10D (5000×60×10) | 4 × 1.6 MB = 6.4 MB | 1.6 MB | 75% |

Note: "per-worker copies" reflects the ForkingPickler's shm storage — each tensor is
moved to `/dev/shm` once, but each worker's `get()` reconstructs a new Python tensor
object referencing the same shm region. Memory savings are modest; the main benefit is
reduced reconstruction overhead.

### 6.5 When OPT-5 Matters Most

```
High impact:  pool_size >= 16, candidate_epochs <= 30, samples >= 1000
              → 10-26% round time improvement

Medium impact: pool_size = 8, candidate_epochs = 50-100
              → 3-7% round time improvement

Low impact:   pool_size <= 4, candidate_epochs >= 200
              → < 2% round time improvement
```

---

## 7. Implementation Plan

### 7.1 Priority-Ordered Changes

| Priority | File | Change | Effort | Risk |
|----------|------|--------|--------|------|
| **P0** | `cascade_correlation.py` | Add `SharedTrainingMemory` helper class | Medium | Medium |
| **P1** | `cascade_correlation.py` | Modify `_generate_candidate_tasks()` to create SharedMemory block | Low | Low |
| **P1b** | `cascade_correlation.py` | Modify `_execute_candidate_training()` to handle dict-type training_inputs (lines 1768-1769) | Low | Medium |
| **P2** | `cascade_correlation.py` | Modify `_execute_parallel_training()` — add `finally` cleanup for SharedMemory blocks | Low | Medium |
| **P3** | `cascade_correlation.py` | Modify `_build_candidate_inputs()` to reconstruct from SharedMemory (returns shm_handle) | Medium | Medium |
| **P3b** | `cascade_correlation.py` | Modify `_process_worker_task()` to close shm_handle in `finally` block | Low | Medium |
| **P4** | `cascade_correlation.py` | Add cleanup in `_shutdown_worker_pool()` | Low | Medium |
| **P5** | `cascade_correlation.py` | Add `atexit` handler and `_active_shm_blocks` tracking in `__init__()` | Low | Low |
| **P6** | `cascade_correlation.py` | Update `RestrictedUnpickler.ALLOWED_CLASSES` if needed | Low | Low |
| **P7** | `tests/performance/` | Add SharedMemory integration, concurrent-read stress, and benchmark tests | Medium | Low |

### 7.2 Detailed Change Specifications

#### P0: SharedTrainingMemory Helper Class

**Location**: `cascade_correlation.py`, new class near top of file (after imports)

```python
class SharedTrainingMemory:
    """Manages a POSIX shared memory block for training tensor sharing.

    Creates a named /dev/shm block containing training tensors that worker
    processes can attach to by name for zero-copy reads.
    """

    MAGIC = b"JNPR"
    VERSION = 1
    HEADER_SIZE = 64
    DESCRIPTOR_SIZE = 32
    DTYPE_MAP = {torch.float32: 0, torch.float64: 1, torch.int32: 2, torch.int64: 3}
    DTYPE_RMAP = {v: k for k, v in DTYPE_MAP.items()}
    NUMPY_DTYPE_MAP = {0: np.float32, 1: np.float64, 2: np.int32, 3: np.int64}

    def __init__(self, tensors: list[torch.Tensor], name_suffix: str):
        """Create SharedMemory block and copy tensors into it."""
        ...

    def get_metadata(self) -> dict:
        """Return metadata dict for inclusion in lightweight tasks."""
        ...

    @staticmethod
    def reconstruct_tensors(metadata: dict) -> list[torch.Tensor]:
        """Attach to SharedMemory by name and return zero-copy tensor views."""
        ...

    def close_and_unlink(self):
        """Release the SharedMemory block."""
        ...
```

#### P1: Modify `_generate_candidate_tasks()`

**Location**: Lines 1655-1699

```python
def _generate_candidate_tasks(self, candidate_input, y, residual_error):
    input_size = candidate_input.shape[1]

    # OPT-5: Create shared memory block for training tensors
    try:
        shm = SharedTrainingMemory(
            tensors=[candidate_input, y, residual_error],
            name_suffix=str(uuid.uuid4())[:8],
        )
        self._active_shm_blocks.append(shm)
        shm_metadata = shm.get_metadata()
        shm_metadata["candidate_epochs"] = self.candidate_epochs
        shm_metadata["candidate_learning_rate"] = self.candidate_learning_rate
        shm_metadata["candidate_display_frequency"] = self.candidate_display_frequency
        training_inputs = shm_metadata  # Dict, not tensor tuple
    except Exception:
        self.logger.warning("OPT-5: SharedMemory creation failed, falling back to full tasks")
        training_inputs = (candidate_input, self.candidate_epochs, y, residual_error,
                          self.candidate_learning_rate, self.candidate_display_frequency)

    # ... rest unchanged (candidate_data generation, task list creation)
    tasks = [(i, candidate_data[i], training_inputs) for i in range(self.candidate_pool_size)]
    return tasks
```

#### P1b: Modify `_execute_candidate_training()`

**Location**: Lines 1768-1769

The remote worker dispatch path extracts tensors by position from `training_inputs`:

```python
# CURRENT (breaks when training_inputs is a dict):
training_inputs = tasks[0][2]
candidate_input, _, y, residual_error = training_inputs[0], training_inputs[1], training_inputs[2], training_inputs[3]
```

Must handle both dict (SharedMemory) and tuple (legacy) formats:

```python
# PROPOSED:
training_inputs = tasks[0][2]
if isinstance(training_inputs, dict):
    # OPT-5: Reconstruct tensors from SharedMemory for remote dispatch
    tensors = SharedTrainingMemory.reconstruct_tensors(training_inputs)
    candidate_input, y, residual_error = tensors
    # Note: remote workers receive full tensor copies (no local /dev/shm access)
else:
    candidate_input, _, y, residual_error = (
        training_inputs[0], training_inputs[1], training_inputs[2], training_inputs[3]
    )
```

Remote workers cannot access local `/dev/shm` blocks, so the remote dispatch path must
reconstruct full tensors before sending them over the network. This is the correct
fallback behavior — SharedMemory only benefits local workers.

#### P2: Modify `_execute_parallel_training()`

Task submission is unchanged (4-tuple structure preserved). Add cleanup in the
**`finally` block** — critical for handling `KeyboardInterrupt` and exceptions:

```python
# In _execute_parallel_training(), the existing try block (around line 1862):
try:
    # ... existing task submission and result collection ...
    results = self._collect_training_results(...)
finally:
    # OPT-5: Release SharedMemory blocks for this round (runs even on error)
    for shm in list(self._active_shm_blocks):
        try:
            shm.close_and_unlink()
            self._active_shm_blocks.remove(shm)
        except Exception as e:
            self.logger.warning(f"OPT-5: SharedMemory cleanup error: {e}")
```

#### P3: Modify `_build_candidate_inputs()`

**Location**: Lines 2616-2675

Add detection of `shm_metadata` dict vs legacy `training_inputs` tuple:

```python
# After unpacking training_inputs at line 2652-2660:
if isinstance(training_inputs, dict):
    # OPT-5: Reconstruct from SharedMemory
    tensors = SharedTrainingMemory.reconstruct_tensors(training_inputs)
    candidate_input, y, residual_error = tensors
    candidate_epochs = training_inputs["candidate_epochs"]
    candidate_learning_rate = training_inputs["candidate_learning_rate"]
    candidate_display_frequency = training_inputs["candidate_display_frequency"]
else:
    # Legacy path: full tensor tuple
    (candidate_input, candidate_epochs, y, residual_error,
     candidate_learning_rate, candidate_display_frequency) = training_inputs
```

#### P3b: Modify `_process_worker_task()` — SharedMemory Handle Cleanup

**Location**: Lines 3010-3046

The SharedMemory handle returned by `_build_candidate_inputs()` must be closed after
training completes. Add `shm_handle` tracking to `_process_worker_task()`:

```python
@staticmethod
def _process_worker_task(task, shared_training_inputs, progress_queue, result_queue, parallel, logger):
    """Process a single candidate training task from the work queue."""
    shm_handle = None  # OPT-5: track SharedMemory handle for deferred close
    try:
        # RC-3 path (unchanged)
        if shared_training_inputs is not None and len(task) == 2:
            full_task = (task[0], task[1], shared_training_inputs)
        else:
            full_task = task

        # Process task (may set shm_handle via _build_candidate_inputs)
        result = CascadeCorrelationNetwork.train_candidate_worker(
            task_data_input=full_task, parallel=parallel, progress_callback=_progress_cb
        )
        result_queue.put(result, timeout=30)
    finally:
        # OPT-5: Close SharedMemory handle after training completes
        if shm_handle is not None:
            try:
                shm_handle.close()
            except Exception:
                pass
```

**Design note**: `shm_handle` is set within `_build_candidate_inputs()` (called by
`train_candidate_worker()`) and needs to be accessible in the outer `finally` block.
Two approaches:
1. Pass a mutable container (e.g., `[None]`) into the call chain
2. Store on a thread-local or worker-local variable
3. Return it alongside the result from `train_candidate_worker()`

Option 3 is cleanest: modify `train_candidate_worker()` to return
`(result, shm_handle)` when SharedMemory is active, and `(result, None)` otherwise.

#### P4: Cleanup in `_shutdown_worker_pool()`

**Location**: Around line 2874

```python
def _shutdown_worker_pool(self) -> None:
    # ... existing shutdown logic ...

    # OPT-5: Clean up any outstanding SharedMemory blocks
    for shm in list(getattr(self, '_active_shm_blocks', [])):
        try:
            shm.close_and_unlink()
        except Exception:
            pass
    self._active_shm_blocks = []
```

#### P5: atexit Handler

**Location**: In `__init__()`, after `self._active_shm_blocks = []`

```python
import atexit
atexit.register(self._cleanup_shared_memory)

def _cleanup_shared_memory(self):
    """Emergency cleanup of SharedMemory blocks on process exit."""
    for shm in list(getattr(self, '_active_shm_blocks', [])):
        try:
            shm.close_and_unlink()
        except Exception:
            pass
```

#### P7: Tests

New test file: `tests/performance/test_shared_memory.py`

| Test | Purpose |
|------|---------|
| `test_shared_training_memory_create_reconstruct` | Unit test: create block, reconstruct tensors, verify values match |
| `test_shared_training_memory_cleanup` | Verify `close_and_unlink()` removes `/dev/shm` block |
| `test_shared_training_memory_fallback` | Verify graceful fallback when SharedMemory creation fails |
| `test_shared_training_memory_contiguity` | Verify non-contiguous tensors are handled (forced contiguous or rejected) |
| `test_lightweight_task_round_trip` | End-to-end: submit lightweight tasks to persistent pool, collect results |
| `test_concurrent_read_stress` | 4 workers simultaneously read from same SharedMemory block under load |
| `test_worker_death_recovery` | Kill worker mid-round, verify replacement can still train via SharedMemory |
| `test_resource_tracker_no_premature_unlink` | Verify worker exit doesn't unlink block prematurely (Python 3.12 tracker) |
| `test_shm_cleanup_on_interrupt` | Simulate exception during task submission, verify `finally` cleanup runs |
| `test_shared_memory_benchmark` | Performance comparison: full tasks vs SharedMemory tasks |

Existing tests in `test_concurrency_scaling.py` should continue to pass unchanged
(they use the full task path which remains as fallback).

---

## 8. Validation Plan

### 8.1 Pre-Implementation Validation (Complete)

- [x] Thread safety analysis: Training tensors are read-only in all worker paths
- [x] ForkingPickler behavior: Already sends handles (~340 bytes), not data
- [x] Cross-process `share_memory_()`: Confirmed zero-copy (`same data_ptr = True`)
- [x] GET-side bottleneck: 97% of queue overhead is tensor reconstruction
- [x] `/dev/shm` capacity: 47 GB available (68 MB used)
- [x] forkserver compatibility: SharedMemory works with all start methods
- [x] Approach B (round headers) rejected: Race conditions + worker death issues

### 8.2 Implementation Validation

| Step | Command | Expected |
|------|---------|----------|
| 1. Unit tests pass | `pytest tests/ -m "unit and not slow" --ignore=tests/performance -q` | 2425 passed |
| 2. SharedMemory tests pass | `pytest tests/performance/test_shared_memory.py --run-performance -v` | All pass |
| 3. Existing perf tests pass | `pytest tests/performance/ --run-performance --benchmark-disable -q` | 134 passed |
| 4. No /dev/shm leaks | `ls /dev/shm/juniper_*` after test suite | No files |
| 5. Benchmark improvement | Compare `test_shared_memory_benchmark` vs baseline | > 50% queue overhead reduction |

### 8.3 Post-Implementation Monitoring

- Monitor `/dev/shm/juniper_*` during long training runs for leaks
- Compare total `grow_network()` time before and after OPT-5
- Verify memory profile (RSS) is stable or improved

---

## 9. Appendix: Alternative Approach — torch.multiprocessing

An alternative to stdlib SharedMemory is using `torch.multiprocessing` instead of
`multiprocessing`:

```python
import torch.multiprocessing as mp  # Drop-in replacement
```

PyTorch's custom reducers would then automatically handle shm transfer. However, this
approach was not selected because:

1. **No control over lifecycle**: PyTorch's `torch_shm_manager` daemon handles cleanup,
   but it's less predictable than explicit `close()`/`unlink()`
2. **fd exhaustion risk**: The `file_descriptor` sharing strategy can leak fds over
   1000+ training rounds (PyTorch issue #973)
3. **Per-get overhead unchanged**: Even with `torch.multiprocessing`, each `queue.get()`
   still reconstructs a tensor from the handle — the exact bottleneck we're optimizing
4. **Broader import change**: Replacing `import multiprocessing` with
   `import torch.multiprocessing` could affect other multiprocessing usage in the file
   (plotting, remote workers)

The stdlib SharedMemory approach gives explicit control over the block lifecycle and
eliminates per-get reconstruction entirely (workers attach once and hold the view).

---

## 10. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-01 | Pre-share approach rejected | 0% pickle size difference via ForkingPickler; no measurable benefit |
| 2026-04-01 | Round header approach rejected | Race conditions (FIFO queue ordering) + worker death loses cache |
| 2026-04-01 | Named SharedMemory selected | No ordering constraints, worker-death resilient, explicit lifecycle |
| 2026-04-01 | Fallback to full tasks on error | Ensures training never fails due to SharedMemory issues |
| 2026-04-01 | Sequential path uses same code path | `_build_candidate_inputs()` handles dict transparently; no separate logic needed |

---

## 11. Plan Validation Review

This plan was reviewed by automated validation agents. The following issues were
identified and resolved:

### Critical Issues (Fixed)

| Issue | Description | Resolution |
|-------|-------------|------------|
| **Use-after-close** | Original design called `shm.close()` in `_reconstruct_training_tensors()`'s `finally` block, invalidating zero-copy tensor views before training begins | Changed to return `(training_inputs, shm_handle)` — caller closes after training completes (section 4.5) |
| **`_execute_candidate_training` breakage** | Lines 1768-1769 index into `training_inputs` by position (`[0]`, `[2]`, `[3]`), which fails on dict | Added P1b change specification with `isinstance(training_inputs, dict)` branch (section 7.2) |

### Medium Issues (Fixed)

| Issue | Description | Resolution |
|-------|-------------|------------|
| **Python 3.12 resource tracker** | Worker processes' resource tracker may prematurely unlink SharedMemory blocks on clean exit | Added `resource_tracker.unregister()` call after opening in workers (section 4.6) |
| **Cleanup not in `finally` block** | SharedMemory cleanup was placed after result collection (inside `try`), missing on `KeyboardInterrupt` or exceptions | Moved cleanup to `finally` block in `_execute_parallel_training()` (section 7.2, P2) |
| **Sequential path narrative** | Plan said "sequential path unchanged" but it receives dict-type training inputs | Corrected fallback table to note that `_build_candidate_inputs()` handles both formats (section 4.7) |

### Low Issues (Noted)

| Issue | Description | Resolution |
|-------|-------------|------------|
| **Missing concurrent-read test** | No test for multiple workers reading same block simultaneously | Added `test_concurrent_read_stress` to test plan (section 7.2, P7) |
| **Remote workers need concrete changes** | Risk table mentioned fallback but no code specification | Added P1b change specification for remote dispatch path |
| **Non-contiguous tensor handling** | No explicit check before writing to SharedMemory | Added to fallback table (section 4.7) and test plan |
| **Performance denominator** | Section 6.2 uses training duration (not total round time) as denominator | Acknowledged — estimates are upper bounds; actual improvement is slightly lower |
