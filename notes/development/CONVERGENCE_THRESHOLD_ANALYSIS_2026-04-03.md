# Training Convergence Threshold Analysis

**Date**: 2026-04-03
**Version**: 1.0.0
**Status**: Active
**Scope**: juniper-cascor training convergence
**Author**: Claude Code

---

## Problem Statement

Training stalls before accuracy/loss reach target values because the patience-based early stopping mechanism cannot trigger. The root cause is that `check_patience()` resets its counter on any improvement, no matter how small (e.g., 1e-8 reduction in loss).

## Root Cause Analysis

### Output Training (grow_network loop)

**File**: `cascade_correlation.py:4449`

**Before** (broken):

```python
if value_loss < best_value_loss:
    best_value_loss = value_loss
    patience_counter = 0
```

**Problem**: During late training, loss decreases by infinitesimal amounts each epoch (e.g., 0.523411 -> 0.523410). This always satisfies `<`, resetting patience to 0. The patience limit is never reached, so training continues until `epochs_max`.

**After** (fixed):

```python
if value_loss < best_value_loss - self.convergence_threshold:
    best_value_loss = value_loss
    patience_counter = 0
```

With `convergence_threshold = 0.001`, improvements smaller than 0.001 are not considered meaningful. The patience counter accumulates, and training stops when `patience_counter >= patience`.

### Candidate Training

**File**: `candidate_unit.py:602`

**Before** (broken):

```python
if current_abs_correlation > abs(best_correlation_so_far):
```

**After** (fixed):

```python
if current_abs_correlation > abs(best_correlation_so_far) + self.convergence_threshold:
```

Same pattern: candidates now stop training when correlation improvement plateaus below the threshold.

## Parameter Propagation

The convergence threshold is configurable at runtime through the full stack:

```
constants_model.py (defaults)
  -> constants.py (propagation)
    -> cascade_correlation_config.py (config class)
      -> cascade_correlation.py (network attribute)
        -> api/lifecycle/manager.py (runtime update)
          -> api/models/training.py (API validation)
            -> juniper-canopy UI (patience/convergence inputs)
```

### Default Values

| Parameter | Default | Scope |
|-----------|---------|-------|
| `_PROJECT_MODEL_CONVERGENCE_THRESHOLD` | 0.001 | Output training patience |
| `_PROJECT_MODEL_CANDIDATE_CONVERGENCE_THRESHOLD` | 0.001 | Candidate training patience |
| `_PROJECT_MODEL_PATIENCE` | 50 | Output training patience epochs |
| `_PROJECT_MODEL_CANDIDATE_PATIENCE` | 30 | Candidate training patience epochs |

### Runtime Updateable Parameters

The following new parameters are exposed via `PATCH /v1/training/params`:

- `convergence_threshold`: Minimum loss improvement to reset output patience (float, 0 < x <= 1.0)
- `candidate_patience`: Candidate training early stopping patience (int, >= 1)
- `candidate_convergence_threshold`: Minimum correlation improvement to reset candidate patience (float, 0 < x <= 1.0)

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Threshold too high: premature stopping | Low | Medium | Default 0.001 is conservative; tunable at runtime |
| Threshold too low: stalling persists | Low | High | Monitor training runs; increase if needed |
| Breaking existing trained models | None | None | Convergence threshold is a training-time parameter only |

## Verification

1. Start training with default threshold (0.001)
2. Observe that patience counter accumulates during loss plateaus
3. Confirm training stops when patience is exhausted (patience_counter >= 50)
4. Verify accuracy at stop point is meaningfully higher than early epochs
5. Test runtime parameter update: `PATCH /v1/training/params {"convergence_threshold": 0.01}`

---

*Generated 2026-04-03.*
