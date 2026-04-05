# Leaf Tensor Autograd Fix — Remediation Plan

**Issue**: In-place slice assignment on leaf tensors with `requires_grad=True` in `add_unit()` and `add_units_as_layer()`
**Source**: PR #76 review comment ([discussion_r3036201967](https://github.com/pcalnon/juniper-cascor/pull/76#discussion_r3036201967))
**Date**: 2026-04-05

---

## Problem Analysis

PR #76 introduced the `init_output_weights` parameter (default `"zero"`) which changes output weight
initialization during unit addition from `torch.randn(...) * 0.1` to `torch.zeros(...)`.

When called with `requires_grad=True`, `torch.zeros()` creates a **leaf tensor**. PyTorch prohibits
in-place modifications (`[:, :] =`) on leaf tensors that require gradients, raising:

```
RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
```

The existing `torch.randn(..., requires_grad=True) * 0.1` path happened to work because `* 0.1`
produces a non-leaf tensor, but this is version-dependent and can corrupt the autograd computation
graph.

### Affected Code Paths

| Method | File | Lines |
|--------|------|-------|
| `add_unit()` | `cascade_correlation.py` | ~3458-3476 |
| `add_units_as_layer()` | `cascade_correlation.py` | ~3590-3601 |

---

## Remediation Plan

### Phase 1: Code Fix

#### Step 1: Fix tensor initialization pattern

**Task 1.1**: In both `add_unit()` and `add_units_as_layer()`, replace:
```python
self.output_weights = torch.zeros(size, requires_grad=True)  # leaf — breaks on in-place
```
with:
```python
self.output_weights = torch.zeros(size)         # no grad → safe for in-place
self.output_weights[:old, :] = old_weights      # copy old weights
self.output_weights.requires_grad_(True)        # enable grad after all mutations
```

Apply the same pattern to the `"random"` branch for consistency and safety.

### Phase 2: Regression Tests

#### Step 2: Add regression tests for `add_unit()`

**Task 2.1**: `test_add_unit_zero_init_no_autograd_error` — verifies no `RuntimeError` with `"zero"` mode
**Task 2.2**: `test_add_unit_random_init_no_autograd_error` — same for `"random"` mode
**Task 2.3**: `test_add_unit_zero_init_preserves_old_weights` — verifies old weights copied correctly

#### Step 3: Add regression tests for `add_units_as_layer()`

**Task 3.1**: `test_add_units_as_layer_zero_init_no_autograd_error` — verifies no `RuntimeError`
**Task 3.2**: `test_add_units_as_layer_random_init_no_autograd_error` — same for `"random"` mode
**Task 3.3**: `test_add_units_as_layer_zero_init_preserves_old_weights` — verifies weight preservation

### Phase 3: Validation

#### Step 4: Verify

**Task 4.1**: Run targeted unit tests for both methods
**Task 4.2**: Run broader test suite to ensure no regressions

---

## Summary of Changes

| File | Change |
|------|--------|
| `src/cascade_correlation/cascade_correlation.py` | Defer `requires_grad_(True)` until after in-place slice assignment in both `add_unit()` and `add_units_as_layer()` |
| `src/tests/unit/test_cascade_correlation_coverage_deep.py` | Add 3 regression tests to `TestAddUnit` |
| `src/tests/unit/test_cascade_correlation_coverage_90.py` | Add 3 regression tests to `TestAddUnitsAsLayer` |
| `notes/LEAF_TENSOR_AUTOGRAD_FIX_PLAN.md` | This plan document |
