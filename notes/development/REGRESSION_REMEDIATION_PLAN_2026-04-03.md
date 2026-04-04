# Juniper-Cascor Regression Remediation Plan

**Date**: 2026-04-03
**Branch**: `fix/regression-epoch-iteration-semantics`
**Author**: Paul Calnon
**Status**: In Progress

---

## Summary

This document outlines the remediation plan for the epoch/iteration semantic
naming issue in juniper-cascor's `grow_network()` method and related code.
See `notes/analysis/REGRESSION_ANALYSIS_2026-04-03.md` for the full root cause analysis.

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | `grow_network()` uses "epoch" for iteration semantics | High | Resolved |

---

## Phase 1: Semantic Rename -- grow_network()

### Background

The `grow_network()` method is the core of the Cascade Correlation algorithm.
Each loop pass performs a full CasCor cycle:

1. Calculate residual error
2. Train candidate pool
3. Select best candidate
4. Add candidate to network (grow)
5. Retrain output layer
6. Validate results

This is an **iteration** (network growth cycle), NOT an **epoch** (single pass
through training data). The existing code used `epoch` and `max_epochs` variable
names, creating semantic confusion with the correctly-named epochs in
`train_output_layer()` and `CandidateUnit.train()`.

### Changes Made

#### Core Source: `cascade_correlation.py`

| Location | Before | After |
|----------|--------|-------|
| `ValidateTrainingInputs` (line 160) | `epoch: int`, `max_epochs: int` | `iteration: int`, `max_iterations: int` |
| `grow_network()` parameter | `max_epochs: int = 1000` | `max_iterations: int = 1000` |
| `grow_network()` loop variable | `for epoch in range(max_epochs)` | `for iteration in range(max_iterations)` |
| `grow_network()` counter | `epochs_completed` | `iterations_completed` |
| `validate_training()` unpacking | `epoch`, `max_epochs` | `iteration`, `max_iterations` |
| `evaluate_early_stopping()` params | `epoch`, `max_epochs` | `iteration`, `max_iterations` |
| `_calculate_residual_error_safe()` params | `epoch`, `max_epochs` | `iteration`, `max_iterations` |
| `_get_training_results()` params | `epoch`, `max_epochs` | `iteration`, `max_iterations` |
| `_add_best_candidate()` params | `epoch`, `max_epochs` | `iteration`, `max_iterations` |
| All log messages in above methods | "Epoch N" | "Iteration N" |

#### API Layer: `api/lifecycle/manager.py`

| Location | Before | After |
|----------|--------|-------|
| Line 176 | `max_epochs=kwargs.get("epochs_max", 200)` | `max_iterations=kwargs.get("epochs_max", 200)` |

Note: The config key `"epochs_max"` is preserved for backward compatibility.

#### Test Files (4 files updated)

- `test_cascade_correlation_coverage_90.py` -- `_add_best_candidate(epoch=0)` -> `(iteration=0)`
- `test_cascade_correlation_coverage_deep.py` -- `ValidateTrainingInputs` constructor
- `test_cascade_correlation_coverage_extended.py` -- `ValidateTrainingInputs` and `evaluate_early_stopping` calls
- `test_network_methods_extended.py` -- `ValidateTrainingInputs` constructors

### Intentionally NOT Changed

| Item | Reason |
|------|--------|
| `train_output_layer()` "epoch" usage | Correct -- each loop pass IS an epoch |
| `CandidateUnit.train()` "epoch" usage | Correct -- each loop pass IS an epoch |
| `output_epochs` attribute/config | Correct semantic |
| `candidate_epochs` in constants | Correct semantic |
| `epochs_max` config key | Backward compatibility |
| `fit()` `max_epochs` parameter | Public API, separate concern |
| History dict keys | Data contract stability |

---

## Phase 2: Validation

### Test Results

| Suite | Tests | Passed | Failed | Skipped |
|-------|-------|--------|--------|---------|
| Unit tests | 914 | 914 | 0 | 3 |

All 3 skipped tests are pre-existing (require `--slow` marker) and unrelated to changes.

---

## Future Considerations

1. **`fit()` parameter naming**: The `fit()` method's `max_epochs` parameter is
   passed to both `train_output_layer()` (correct usage) and `grow_network()`
   (now `max_iterations`). Consider adding a separate `max_iterations` parameter
   to `fit()` in a future API revision.

2. **Config key rename**: The `epochs_max` config key could be aliased to
   `iterations_max` with deprecation warnings in a future version.

3. **API state field**: `TrainingMonitorState.max_epochs` in the API layer refers
   to grow_network iterations. Consider renaming to `max_iterations` in a future
   API version bump.
