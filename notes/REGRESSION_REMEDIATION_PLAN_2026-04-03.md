# Juniper Regression Remediation Plan — 2026-04-03

**Scope**: Fixes for training stalling, parameter mapping, UI regressions
**Status**: Implementation complete — testing in progress
**Affects**: juniper-cascor, juniper-canopy

---

## Phase 1: Training Convergence Fix (juniper-cascor) — CRITICAL

### Fix 1.1: Add convergence_threshold to CasCor network

**Root Cause**: RC-NEW-001 — check_patience() resets on any improvement
**Approach**: Add a convergence_threshold minimum-improvement gate

**Files Modified**:
| File | Change |
|------|--------|
| `cascor_constants/constants_model/constants_model.py` | Added `_PROJECT_MODEL_CONVERGENCE_THRESHOLD = 0.001`, `_PROJECT_MODEL_CANDIDATE_CONVERGENCE_THRESHOLD = 0.001`, `_PROJECT_MODEL_CANDIDATE_PATIENCE = 50` |
| `cascor_constants/constants.py` | Threaded new constants through alias chain |
| `cascade_correlation_config/cascade_correlation_config.py` | Added `convergence_threshold`, `candidate_convergence_threshold`, `candidate_patience` fields |
| `cascade_correlation/cascade_correlation.py` (init) | Initialize new attributes from config |
| `cascade_correlation/cascade_correlation.py` (check_patience) | Changed `value_loss < best_value_loss` to `value_loss < best_value_loss - self.convergence_threshold` |

**Strengths**: Follows existing config → constant → network attribute pattern. Non-breaking — existing behavior preserved when threshold is 0.
**Risks**: If convergence_threshold is set too high, training may stop prematurely. Mitigated by reasonable default (0.001).
**Guardrails**: Pydantic validation enforces `gt=0` on API input.

### Fix 1.2: Training-loss-based early stopping without validation data

**Root Cause**: RC-NEW-004 — no early stopping when x_val/y_val are None
**Approach**: Add else branch in validate_training() using train_loss instead of val_loss

**Files Modified**:
| File | Change |
|------|--------|
| `cascade_correlation/cascade_correlation.py` (validate_training) | Added else block using train_loss for patience evaluation |

**Strengths**: Uses identical convergence_threshold + patience logic as val-based stopping.
**Risks**: Training loss is a less reliable signal than validation loss (overfitting not detected). Acceptable because the alternative is no early stopping at all.

### Fix 1.3: Expand updatable parameters

**Root Cause**: RC-NEW-002 — convergence/patience params silently dropped
**Approach**: Add missing params to updatable_keys set

**Files Modified**:
| File | Change |
|------|--------|
| `api/lifecycle/manager.py` | Added `convergence_threshold`, `candidate_convergence_threshold`, `candidate_patience`, `candidate_epochs` to updatable_keys |
| `api/models/training.py` | Added `candidate_epochs` field to TrainingParamUpdateRequest |

---

## Phase 2: Parameter Mapping Fix (juniper-canopy) — HIGH

### Fix 2.1: Correct _CANOPY_TO_CASCOR_PARAM_MAP

**Root Cause**: RC-NEW-003 — convergence_threshold mapped to patience
**Approach**: Fix mapping and add missing entries

**Files Modified**:
| File | Change |
|------|--------|
| `backend/cascor_service_adapter.py` | Fixed `nn_growth_convergence_threshold` → `convergence_threshold`, added `nn_patience`, `cn_training_convergence_threshold`, `cn_training_iterations`, `cn_patience` |

### Fix 2.2: Add missing params to key lists

**Files Modified**:
| File | Change |
|------|--------|
| `main.py` | Added `nn_patience` to nn_keys, `cn_candidate_learning_rate` and `cn_patience` to cn_keys |

### Fix 2.3: Expand TrainingState updates after parameter changes

**Files Modified**:
| File | Change |
|------|--------|
| `main.py` | Added TrainingState updates for convergence_threshold, patience, candidate_pool_size, correlation_threshold, candidate_learning_rate |

---

## Phase 3: UI Improvements (juniper-canopy) — MEDIUM

### Fix 3.1: Increase plot card heights

**Files Modified**:
| File | Change |
|------|--------|
| `frontend/components/decision_boundary.py` | Height 600px → 800px |
| `frontend/components/dataset_plotter.py` | Height 600px → 800px |

### Fix 3.2: Add iteration tracking to demo mode

**Files Modified**:
| File | Change |
|------|--------|
| `demo_mode.py` | Added `current_iteration` counter, incremented on cascade install, included in metrics, reset on start/reset |

---

## Validation Checklist

- [ ] juniper-cascor unit tests pass
- [ ] juniper-cascor integration tests pass
- [ ] juniper-canopy unit tests pass
- [ ] juniper-canopy integration tests pass
- [ ] Parameter update round-trip: canopy → cascor → verify applied
- [ ] Training converges on spiral problem (demo mode)
- [ ] Training converges on spiral problem (service mode)
- [ ] Plot heights visually verified
- [ ] Iteration counter increments correctly on cascade add

---

## Risk Assessment

| Fix | Risk Level | Mitigation |
|-----|-----------|------------|
| convergence_threshold in check_patience | Low | Default 0.001 preserves near-original behavior |
| Training-loss early stopping | Medium | Only fires when no val data (fallback behavior) |
| Parameter mapping corrections | Low | Direct bug fix, no behavioral ambiguity |
| Plot height increase | Very Low | CSS-only change, aspect ratio maintained by Plotly |
| Iteration tracking | Low | Additive change, no existing behavior modified |
