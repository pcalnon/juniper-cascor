# Juniper Cascor -- Regression Analysis

**Date**: 2026-04-03
**Version**: 1.0.0
**Status**: Current
**Scope**: Epoch/iteration semantic error in `grow_network()` and related dataclasses
**Branch**: `fix/regression-epoch-iteration-semantics`

---

## Summary Table

| # | Issue | Severity | Component(s) | Status |
|---|-------|----------|---------------|--------|
| 1 | Epoch/Iteration Semantic Error in `grow_network()` | **High** | `cascade_correlation.py`, `ValidateTrainingInputs` dataclass | Identified -- rename required |

---

## Issue 1: Epoch/Iteration Semantic Error in `grow_network()`

**Severity**: High
**Component**: `src/cascade_correlation/cascade_correlation.py`
**Status**: Identified -- internal rename required. No breaking API changes.

### Root Cause

The `grow_network()` method uses `epoch` and `max_epochs` as its loop variable and parameter names, but each loop iteration performs a **complete Cascade Correlation growth cycle**, not a single pass through the training data.

A single pass through the `grow_network()` loop (lines 3587-3682) performs:

1. Residual error calculation (`_calculate_residual_error_safe`)
2. Full candidate pool training (`_get_training_results` -- which internally trains each candidate for multiple true epochs)
3. Best candidate selection (correlation threshold check)
4. Grow iteration callback to lifecycle manager
5. Candidate installation into the network (`_add_best_candidate` or `_select_best_candidates`)
6. Output layer retraining (`train_output_layer` -- which internally runs multiple true epochs)
7. Validation (`validate_training`)
8. Early stopping evaluation

This is an **iteration** (network growth cycle), not an epoch.

### Correct vs. Incorrect Usage in the Codebase

| Method | Variable Name | Actual Semantic | Correct? |
|--------|---------------|-----------------|----------|
| `grow_network()` | `max_epochs` (param, line 3542) | Maximum growth iterations | **No** -- should be `max_iterations` |
| `grow_network()` | `epoch` (loop var, line 3587) | Current growth iteration index | **No** -- should be `iteration` |
| `grow_network()` | `epochs_completed` (line 3586) | Growth iterations completed | **No** -- should be `iterations_completed` |
| `grow_network()` | Log messages: "Epoch {epoch}" (lines 3594, 3647, 3677, etc.) | Growth iteration index | **No** -- should be "Iteration" |
| `ValidateTrainingInputs` | `epoch` (field, line 163) | Current growth iteration index | **No** -- should be `iteration` |
| `ValidateTrainingInputs` | `max_epochs` (field, line 164) | Maximum growth iterations | **No** -- should be `max_iterations` |
| `train_output_layer()` | `epoch` (loop var, line 1567) | Single pass through training data | **Yes** -- this IS a true epoch |
| `CandidateUnit.train()` | epoch loop | Single pass through training data | **Yes** -- this IS a true epoch |
| `fit()` | `max_epochs` (param, line 1354) | Maximum output training epochs | **Yes** -- passed to `train_output_layer` |
| Lifecycle manager | `_grow_iteration_callback` (line 353) | Growth iteration callback | **Yes** -- already uses correct terminology |
| Lifecycle manager callback | `iteration`, `max_iterations` (line 353) | Growth iteration parameters | **Yes** -- already uses correct terminology |
| Grow callback invocation | `iteration=epoch, max_iterations=max_epochs` (lines 3614-3615) | Maps wrong names to right names | Transitional -- rename source |

### Evidence

**`grow_network()` signature** (line 3538-3542):

```python
def grow_network(
    self,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    max_epochs: int = 1000,  # <-- should be max_iterations
    ...
```

**Loop variable and log messages** (lines 3586-3594):

```python
epochs_completed = 0                    # <-- should be iterations_completed
for epoch in range(max_epochs):         # <-- should be: for iteration in range(max_iterations)
    ...
    self.logger.debug(f"... Epoch {epoch}, Residual Error: ...")  # <-- should be "Iteration"
```

**Callback invocation already uses correct names** (lines 3613-3615):

```python
_grow_cb(
    iteration=epoch,           # maps misnamed 'epoch' to correctly-named 'iteration'
    max_iterations=max_epochs, # maps misnamed 'max_epochs' to correctly-named 'max_iterations'
    ...
)
```

This confirms the lifecycle manager (`src/api/lifecycle/manager.py`, line 353) already expects `iteration`/`max_iterations` parameter names. The callback invocation at lines 3614-3615 performs an implicit rename from the incorrect internal names to the correct external names.

**`ValidateTrainingInputs` dataclass** (lines 160-174):

```python
@dataclass
class ValidateTrainingInputs:
    epoch: int       # <-- should be iteration
    max_epochs: int  # <-- should be max_iterations
    patience_counter: int
    early_stopping: bool
    ...
```

**Correct usage in `train_output_layer()`** (lines 1567):

```python
for epoch in range(epochs):  # Correct: each loop pass IS a single pass through the data
```

**Correct usage in `fit()` method** (lines 1415-1416):

```python
max_epochs = (max_epochs, self.output_epochs)[max_epochs is None]
train_loss = self.train_output_layer(x_train, y_train, max_epochs)  # Correct: epochs for output training
```

### Impact Analysis

| Scope | Impact | Breaking? |
|-------|--------|-----------|
| Internal variable/parameter names | Rename only -- no functional behavior change | No |
| Log messages | "Epoch N" becomes "Iteration N" in grow_network context only | No |
| `ValidateTrainingInputs` dataclass | Field rename: `epoch` to `iteration`, `max_epochs` to `max_iterations` | Internal only -- not part of public API |
| `TrainingResults` dataclass | `epochs_completed` field at line 141 | Review needed -- may refer to candidate training epochs (correct) or growth iterations (incorrect) depending on context |
| API layer | Already uses `grow_iteration` correctly (lifecycle manager) | No change needed |
| `fit()` method | Uses `max_epochs` for output training epochs -- this usage IS correct | No change needed |
| Downstream (juniper-canopy) | Lifecycle manager already translates to `iteration`/`max_iterations` | No change needed |
| `on_grow_iteration_callback` parameter | Already correctly named (line 3548) | No change needed |

### Variables to Rename

The following table lists every instance that requires renaming, scoped exclusively to `grow_network()` and the `ValidateTrainingInputs` dataclass. No other methods should be affected.

| Current Name | New Name | Location | Line(s) |
|--------------|----------|----------|---------|
| `max_epochs` (parameter) | `max_iterations` | `grow_network()` signature | 3542 |
| `max_epochs` (docstring) | `max_iterations` | `grow_network()` docstring | 3556 |
| `epochs_completed` | `iterations_completed` | `grow_network()` body | 3586, 3682, 3684 |
| `epoch` (loop variable) | `iteration` | `grow_network()` body | 3587, 3594, 3614, 3646, 3647, 3651, 3670, 3677, 3679, 3681, 3682 |
| `max_epochs` (references) | `max_iterations` | `grow_network()` body | 3587, 3615, 3652, 3670, 3684 |
| Log string "Epoch" | "Iteration" | `grow_network()` log messages | 3594, 3647, 3670, 3677, 3679, 3681 |
| `epoch` (field) | `iteration` | `ValidateTrainingInputs` | 163 |
| `max_epochs` (field) | `max_iterations` | `ValidateTrainingInputs` | 164 |

### Callers of `grow_network()` to Update

Any caller passing `max_epochs=` as a keyword argument must be updated:

| Caller | File | Notes |
|--------|------|-------|
| `fit()` | `cascade_correlation.py` line 1445 | Passes `max_epochs=max_epochs` -- rename keyword |
| Unit tests | `src/tests/unit/test_training_workflow.py` | Verify keyword arguments |
| Unit tests | `src/tests/unit/test_cascade_correlation_coverage*.py` | Verify keyword arguments |
| Performance tests | `src/tests/performance/test_endtoend_profiling.py` | Verify keyword arguments |
| Monitoring hook tests | `src/tests/unit/api/test_monitoring_hooks.py` | Verify keyword arguments |

### Consumers of `ValidateTrainingInputs` to Update

Any code constructing or reading `ValidateTrainingInputs` fields must be updated:

| Consumer | Location | Notes |
|----------|----------|-------|
| `grow_network()` | Line 3651 (`epoch=epoch`) | Rename to `iteration=iteration` |
| `grow_network()` | Line 3652 (`max_epochs=max_epochs`) | Rename to `max_iterations=max_iterations` |
| `validate_training()` | Reads `.epoch` and `.max_epochs` fields | Rename field access |
| Log messages | Lines referencing `validate_training_inputs` fields | Update if fields are interpolated |

### Proposed Remediation

1. Rename `grow_network()` parameter `max_epochs` to `max_iterations` with a backward-compatible alias (matching the pattern already used in `fit()` at lines 1375-1379)
2. Rename loop variable `epoch` to `iteration` throughout `grow_network()` body
3. Rename `epochs_completed` to `iterations_completed`
4. Update all log messages within `grow_network()` from "Epoch" to "Iteration"
5. Rename `ValidateTrainingInputs` fields: `epoch` to `iteration`, `max_epochs` to `max_iterations`
6. Update all callers and consumers listed above
7. Run full test suite to verify no regressions: `cd src/tests && bash scripts/run_tests.bash`

**Do NOT rename** `epoch` in `train_output_layer()`, `CandidateUnit.train()`, or `fit()` -- those usages are semantically correct.
