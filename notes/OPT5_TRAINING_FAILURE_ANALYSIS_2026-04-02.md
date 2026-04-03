# OPT-5 Training Failure Analysis

**Date**: 2026-04-02
**Severity**: CRITICAL - Blocks all training execution
**Reference**: `/home/pcalnon/Development/python/Juniper/juniper-canopy/notes/analysis/CRITICAL_REGRESSION_ANALYSIS_2026-04-02.md`

---

## Summary

The OPT-5 SharedMemory optimization (commit `f603f1b`, 2026-04-01) introduced three defects in the parallel candidate training pipeline that cause training to fail shortly after the first epoch begins. A fourth pre-existing defect (walrus operator precedence) was found during investigation but had already been fixed.

## Root Causes and Fixes Applied

### RC-1 (CRITICAL): Non-Writable SharedMemory Tensor Views

**Problem**: `SharedTrainingMemory.reconstruct_tensors()` returns zero-copy numpy-backed tensors from the shared memory buffer. These are read-only. When candidate training attempts in-place operations (gradient accumulation, weight updates), PyTorch raises `RuntimeError`.

**Fix**: Added `.clone()` calls after tensor reconstruction in `_build_candidate_inputs()`:
```python
candidate_input, y, residual_error = tensors[0].clone(), tensors[1].clone(), tensors[2].clone()
```

**File**: `cascade_correlation.py`, `_build_candidate_inputs` method

### RC-2 (CRITICAL): SharedMemory Use-After-Free Race Condition

**Problem**: The `finally` block in `_execute_parallel_training` unlinks SharedMemory blocks while persistent workers may still be reading from them.

**Fix**: RC-1's clone fix eliminates the race condition. Workers clone tensors immediately upon reconstruction, so they no longer hold references to the SharedMemory buffer during training. The existing cleanup in the `finally` block is now safe.

### RC-3 (HIGH): Correlation Validation Rejects Valid Results

**Problem**: `_validate_training_result` strictly rejects `correlation > 1.0`, but floating-point arithmetic with the epsilon denominator in Pearson correlation can produce values marginally above 1.0. Valid parallel results are discarded, potentially causing premature training termination.

**Fix**: Added correlation clamping at the source in `candidate_unit.py::_calculate_correlation`:
```python
if correlation > 1.0:
    self.logger.debug(f"Clamping correlation {correlation:.10f} to 1.0 (floating-point overshoot)")
    correlation = 1.0
```

**File**: `candidate_unit.py`, `_calculate_correlation` method

### RC-4 (MODERATE): Walrus Operator Precedence Bug — Already Fixed

**Problem**: `if snapshot_path := self.create_snapshot() is not None:` assigns a boolean instead of the path.
**Status**: Already corrected in current codebase to `if (snapshot_path := self.create_snapshot()) is not None:`.

## Impact

With these fixes, the OPT-5 SharedMemory optimization is preserved with the following characteristics:
- Training tensors are still shared via `/dev/shm` (zero-copy read from shared block)
- Each worker clones tensors locally after reading (writable, safe for autograd)
- The performance benefit of avoiding N-fold serialization through the task queue is maintained
- The SharedMemory lifecycle is safe: workers don't hold references during training
