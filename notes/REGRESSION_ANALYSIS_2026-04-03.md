# Juniper-CasCor Regression Analysis — 2026-04-03

**Scope**: Critical training stalling, parameter update failures, and cross-service integration issues
**Status**: Root cause analysis complete — fixes in development
**Affects**: juniper-cascor, juniper-canopy, juniper-cascor-client
**Author**: Claude Code (Principal Engineer analysis)

---

## Executive Summary

Training stalls before reaching target accuracy/loss values due to **four interconnected root causes** spanning the cascor backend, canopy frontend, and the client library that bridges them. The previously documented RC-001 through RC-007 bugs have all been resolved, but the following new critical issues remain:

1. **No convergence threshold in patience check** — patience resets on infinitesimal improvements
2. **Missing updatable parameters** in lifecycle manager — convergence/patience params silently dropped
3. **Incorrect parameter mapping** in canopy service adapter — convergence_threshold mapped to patience
4. **No training-loss-based early stopping** when validation data is absent

---

## Root Cause Analysis

### RC-NEW-001: Missing convergence_threshold in check_patience() — CRITICAL

**File**: `cascade_correlation.py:4440-4444`
**Impact**: Training never converges — runs to max_epochs without meaningful progress

**Current code:**
```python
def check_patience(self, patience_counter=0, value_loss=float("inf"), best_value_loss=float("inf")):
    if value_loss < best_value_loss:          # ANY improvement resets patience
        best_value_loss = value_loss
        patience_counter = 0
    else:
        patience_counter += 1
```

**Problem**: The patience counter resets on *any* improvement to validation loss, no matter how small (e.g., 0.000001). When training is in a near-plateau region:
- Loss decreases by tiny amounts each iteration
- Patience counter constantly resets to 0
- Patience is never exhausted
- Training runs until max_epochs without converging to target

**Expected behavior**: A convergence threshold should define the *minimum meaningful improvement*. Only improvements greater than this threshold should reset the patience counter.

**Fix**: Add `convergence_threshold` attribute to the network and use it in `check_patience()`:
```python
if value_loss < best_value_loss - self.convergence_threshold:  # Meaningful improvement
    best_value_loss = value_loss
    patience_counter = 0
else:
    patience_counter += 1
```

**Dependencies**: Requires adding `convergence_threshold` to:
- `CascadeCorrelationConfig` dataclass
- `CascadeCorrelationNetwork.__init__()` parameter initialization
- Constants module for default value
- Lifecycle manager's `updatable_keys` set

---

### RC-NEW-002: update_params Missing Critical Parameters — HIGH

**File**: `api/lifecycle/manager.py:709-717`
**Impact**: Parameter updates from canopy UI silently dropped

**Current updatable_keys:**
```python
updatable_keys = {
    "learning_rate",
    "candidate_learning_rate",
    "correlation_threshold",
    "candidate_pool_size",
    "max_hidden_units",
    "epochs_max",
    "patience",
}
```

**Missing from updatable_keys:**
- `convergence_threshold` — needed for RC-NEW-001 fix
- `candidate_patience` — candidate training early stopping
- `candidate_convergence_threshold` — candidate training convergence
- `candidate_epochs` — candidate training duration

**Note**: The REST API's `TrainingParamUpdateRequest` model (models/training.py:45-58) already defines `convergence_threshold`, `candidate_patience`, and `candidate_convergence_threshold` as valid request fields. They pass Pydantic validation but are silently ignored by the lifecycle manager.

---

### RC-NEW-003: Incorrect Canopy-to-CasCor Parameter Mapping — HIGH

**File**: `juniper-canopy/src/backend/cascor_service_adapter.py:426-434`
**Impact**: Convergence threshold updates go to wrong parameter; several params unmapped

**Current mapping:**
```python
_CANOPY_TO_CASCOR_PARAM_MAP = {
    "nn_learning_rate": "learning_rate",
    "nn_max_hidden_units": "max_hidden_units",
    "nn_max_total_epochs": "epochs_max",
    "nn_growth_convergence_threshold": "patience",       # ← WRONG! Should be "convergence_threshold"
    "cn_pool_size": "candidate_pool_size",
    "cn_correlation_threshold": "correlation_threshold",
    "cn_candidate_learning_rate": "candidate_learning_rate",
}
```

**Issues:**
1. `nn_growth_convergence_threshold` → `patience` is WRONG. This maps the convergence threshold value to the patience counter, corrupting both.
2. Missing mappings for: `nn_patience` → `patience`, `cn_training_convergence_threshold` → `candidate_convergence_threshold`

**Additional issue in canopy main.py**: The `cn_keys` list is missing `cn_candidate_learning_rate`, so even though the adapter has a mapping for it, the parameter is never collected from the incoming request.

---

### RC-NEW-004: No Early Stopping Without Validation Data — MEDIUM

**File**: `cascade_correlation.py:4285`
**Impact**: Training runs to max_epochs when no validation data provided

**Current code in validate_training():**
```python
if x_val is not None and y_val is not None:
    # ALL early stopping logic lives here
    ...
```

When `x_val` and `y_val` are None, `early_stop_flag` stays False (line 4276). The only exit conditions are:
- `max_epochs` reached
- No candidate meets `correlation_threshold`
- Candidate training fails

**Fix**: Add training-loss-based early stopping when no validation data:
```python
else:
    # No validation data — use training loss for early stopping
    if early_stopping:
        if train_loss < best_value_loss - self.convergence_threshold:
            best_value_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
        patience_exhausted = patience_counter >= self.patience
        max_units_reached = self.check_hidden_units_max()
        train_accuracy_reached = self.check_training_accuracy(train_accuracy, self.target_accuracy)
        early_stop_flag = patience_exhausted or max_units_reached or train_accuracy_reached
```

---

## Failure Cascade

The interaction of these root causes creates a cascading failure:

1. User sets convergence threshold in canopy UI (e.g., 0.001)
2. Canopy sends `nn_growth_convergence_threshold: 0.001`
3. Service adapter maps it to `patience: 0.001` (RC-NEW-003)
4. CasCor receives PATCH with `patience: 0.001`
5. Pydantic validates patience as `Optional[int]` with `ge=1` — **0.001 fails validation** → returns 400/422
6. Even if it passed, lifecycle manager would set `self.network.patience = 0.001` (invalid)
7. Meanwhile, `check_patience()` has no convergence_threshold check (RC-NEW-001)
8. Patience counter resets on trivial improvements
9. Training runs indefinitely without meaningful convergence

---

## Previously Fixed Issues (Verified 2026-04-03)

| RC ID | Issue | Status |
|-------|-------|--------|
| RC-001 | Walrus operator precedence | ✅ Fixed (line 1611) |
| RC-002 | WebSocket coroutine leak | ✅ Fixed (broad except) |
| RC-003 | Exception handling in _run_training | ✅ Fixed (state machine update) |
| RC-004 | Drain thread queue timing race | ✅ Fixed (deferred discovery) |
| RC-005 | SharedMemory lifecycle race | ✅ Fixed (atomic cleanup) |
| RC-006 | Undeclared global variable | ✅ Fixed (removed) |
| RC-007 | Duplicate ActivationWithDerivative | ✅ Fixed (extracted to utils/) |

---

## Impact Assessment

| Issue | Severity | User Impact |
|-------|----------|-------------|
| RC-NEW-001 | **CRITICAL** | Training never converges; project unusable |
| RC-NEW-002 | **HIGH** | Parameter tuning via UI has no effect on training |
| RC-NEW-003 | **HIGH** | Convergence threshold corrupts patience; params mismatch |
| RC-NEW-004 | **MEDIUM** | Training without val data can't stop early |

---

## Recommended Fix Priority

1. RC-NEW-001 + RC-NEW-002 (cascor core) — Add convergence_threshold, fix updatable_keys
2. RC-NEW-003 (canopy adapter) — Fix parameter mapping
3. RC-NEW-004 (cascor core) — Training-loss-based early stopping

These fixes are interdependent: RC-NEW-003 depends on RC-NEW-001 being complete (convergence_threshold must exist before it can be mapped).
