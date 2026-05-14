# Phase 2 P2-1d Design — Live Dataset Swap: Resize + Pad

**Status**: Design lock-in (2026-05-13)
**Supersedes**: `juniper-canopy/notes/ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md` §3.6 (the prepend-adapter-layer approach)
**Scope**: cascor PR `phase2/p2-1d-resize-and-pad` + a follow-on canopy PR patching the parent spec to reference this doc
**Companion artifacts**: `src/api/lifecycle/manager.py`, `src/cascade_correlation/cascade_correlation.py`, `src/api/lifecycle/architecture_adapter.py` (deleted), and their associated tests

---

## 1. Why this design replaces §3.6

The original §3.6 approach — *prepend an input-side adapter layer on shrink, append an output-side adapter layer on shrink, expand the outermost adapter on grow* — accumulated complexity at every implementation step while drifting from the long-term cascade-correlation modelling goals:

- Adapter layers create a new layer type that the existing forward pass, snapshot serializer, replay reconstructor, and history persistence layers must all learn. Each becomes an entangled vector for future bugs.
- The zero-init contract from §3.6 ("connections from the new adapter into the next layer are zero-initialized") produces literal-zero outputs across all dims on day 1 after shrink, requiring `output_training_first` to relearn everything from scratch. The structural preservation buys nothing concrete.
- The sequential-composition rule (chains of input + output adapters, with grow re-targeting the outermost layer) is several hundred extra LOC of bookkeeping that exists only to maintain the adapter framing.

The new design eliminates all of that. The network is **monotonically non-decreasing**: it never shrinks. Dataset-side dimensions that fall below the network's current capacity are **zero-padded** to match. Dataset dimensions that exceed it cause the network to **grow with random-init new connections** — the same initialization pattern the network used at construction.

The cost is small: padded input slots inject a bounded perturbation into hidden activations via existing bias terms (which the network can readily learn to ignore), and padded output slots require a small loss-masking step to avoid training the network to predict zero on dead targets. Both costs are well-understood ML primitives. Nothing about the cascade architecture has to learn a new concept.

## 2. Behavioural contract

| Dataset dim relative to network dim | Action                                                                                   | Initialization                                                                                                                 |
|-------------------------------------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `dataset.input == network.input`    | No-op                                                                                    | —                                                                                                                              |
| `dataset.input > network.input`     | **Grow network input dim**                                                               | Random × `random_value_scale` for new weight rows in `output_weights` and inserted entries in each hidden unit's weight vector |
| `dataset.input < network.input`     | **Zero-pad input tensors** to `network.input_size`                                       | All-zero columns appended to `_train_x`, `_val_x`                                                                              |
| `dataset.output == network.output`  | No-op                                                                                    | —                                                                                                                              |
| `dataset.output > network.output`   | **Grow network output dim**                                                              | Random × `random_value_scale` for new columns in `output_weights` and new elements in `output_bias`                            |
| `dataset.output < network.output`   | **Zero-pad target tensors** to `network.output_size` AND **mask loss** to the active dim | All-zero columns appended to `_train_y`, `_val_y`; `criterion` slices to `output[:, :active_output_dim]`                       |

**Mixed** (e.g., input grows AND output shrinks) is supported by composition: grow the network's input side; pad the dataset's output side. Sides are independent.

**Invariants**:

- The network's `input_size` and `output_size` only ever grow. There is no "shrink the network" path.
- Tensor mutations on `output_weights`, `output_bias`, and `hidden_units[i]["weights"]` are atomic per swap (all succeed or rollback fires).
- `hidden_units[i]["bias"]` is a per-unit scalar — **never touched** on resize. The cascade-correlation contract continues to freeze hidden-unit parameters after promotion, even across dataset swaps.
- The §3.7 rollback contract (snapshot + restore on any failure between snapshot and topology rebroadcast) continues to hold. P2-1b's cancel mechanism continues to apply.

## 3. Implementation: where each piece lives

### 3.1 Network class (`cascade_correlation.py`)

The user's prototype (lines 734–800 of the current working tree) sketched the right API surface. The production version cleans up six bugs:

| Bug | Symptom | Fix |
| :--- | :--- | :--- |
| **B1**: reads from `self.config.input_size` | stale after the first grow; subsequent swaps compute wrong deltas | use `self.input_size` / `self.output_size` (live values) |
| **B2**: `_resize_network_hidden_for_dataset` passes `tensor=` and `dim0_size_*` kwargs to `_grow_network_for_dataset` whose signature is `weights_tensor=` and `input_size_*` | `TypeError` on first iteration | re-shape the helper signatures so call sites match |
| **B3**: `padding_size = (current - new, ...)` | non-positive on a grow → `cat` never fires | invert to `(new - current, ...)` |
| **B4**: `_grow_network_tensor` calls `tensor.size(1)` | `IndexError` on 1-D bias | branch on `tensor.ndim`; 1-D path uses single cat on dim 0 |
| **B5**: hidden-unit weight grow uses `torch.cat([w, zeros], dim=0)` | appends new raw-input weights AFTER prior-hidden weights, breaking forward-pass alignment at `cascade_correlation.py:1541-1544` | **row-insertion at index `self.input_size`** — same pattern that the P2-1c adapter used (the one piece of P2-1c carrying forward) |
| **B6**: `dims=(1, 0)` parameter passed inconsistently across call sites | latent: only works for one tensor shape | eliminate the `dims` parameter; let each method know its own tensor layout |

The cleaned-up shape on the network class:

```python
def _resize_network_for_dataset(self, input_size_new: int, output_size_new: int) -> dict:
    """Grow input and/or output dimensions to match a new dataset.

    Returns {input_delta, output_delta, hidden_preserved}. No-op on equal-dim.
    Raises ValueError on any attempted shrink — the network is monotonically
    non-decreasing; shrink is handled by zero-padding the dataset (see the
    lifecycle's _pad_dataset_for_network helper).
    """

def _grow_output_layer(self, input_delta: int, output_delta: int) -> None:
    """Row-insert input_delta new random-init rows into output_weights at
    index self.input_size (between raw-input rows and hidden-unit rows).
    Column-append output_delta random-init columns at the end. Update
    output_bias with output_delta new random-init elements. Re-enable
    requires_grad on the new tensors."""

def _grow_hidden_units_for_input(self, input_delta: int) -> None:
    """For each hidden unit, insert input_delta random-init entries into its
    weight vector at index self.input_size. Hidden-unit biases are scalars
    and untouched. Hidden-unit weights remain detached (cascade-correlation
    freezes them after promotion)."""
```

Bookkeeping: `self.input_size` and `self.output_size` are updated **after** all tensor mutations succeed, so a partial failure leaves the network self-consistent for the lifecycle's rollback path.

Initialization scale: `torch.randn(...) * self.random_value_scale` — matches the construction-time initialization at `cascade_correlation.py:714-715`.

### 3.2 Dataset padding helper (lifecycle layer)

New helper on `TrainingLifecycleManager`:

```python
def _pad_dataset_for_network(
    self, x, y, val_x, val_y
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int, int]:
    """Zero-pad dataset tensors up to the network's current input/output dims.

    Returns (x_padded, y_padded, val_x_padded, val_y_padded,
             active_input_dim, active_output_dim) where active_*_dim is the
    pre-padding dataset dim. The lifecycle then passes active_output_dim
    into the training methods so the loss is masked to active dims only.

    Raises ValueError if dataset dim exceeds network dim (the caller must
    have resized the network first).
    """
```

Called from **two places** (per the O1 design decision):

- `start_training` — for cold-swap parity. If the user starts training on a dataset smaller than the network's previous capacity, padding kicks in.
- `swap_dataset_live` — after `_reload_dataset` and after any grow, before `_check_swap_cancel`.

The helper does not mutate stored tensors directly; it returns padded copies. The lifecycle assigns the returned tensors to `self._train_x`, `self._train_y`, etc.

### 3.3 Loss masking (cascade_correlation.py training methods)

`train_output_layer` and `train_candidates` accept a new optional `active_output_dim: Optional[int] = None` keyword. Default `None` → no masking (preserves current behaviour for non-padded datasets).

When non-`None`, the loss computation is sliced:

```python
# Before (line 1733):
loss = criterion(output, y)
# After:
y_active = y if active_output_dim is None else y[:, :active_output_dim]
output_active = output if active_output_dim is None else output[:, :active_output_dim]
loss = criterion(output_active, y_active)
```

The same slicing is applied at:

- `train_output_layer` line 1733 (epoch loss)
- `train_output_layer` line 1759 (final-loss reporting)
- `train_candidates` residual-error and metric computations (any site where target shape is consumed)

The lifecycle passes `active_output_dim` whenever it calls `_run_training` / `network.fit` after a pad/swap. The active dim is stored on the lifecycle: `self._active_output_dim: int = self.network.output_size` by default, updated whenever `_pad_dataset_for_network` runs.

### 3.4 Lifecycle integration (`manager.py swap_dataset_live` step 8)

The P2-1c step 8 (which called `architecture_adapter.adapt_for_dataset_swap`) is replaced with:

```python
# Step 8 (P2-1d): resize network to fit grow side, then pad dataset to fit shrink side.
new_input = self._train_x.shape[1]
new_output = self._train_y.shape[1]
arch_changes = self._apply_dataset_dim_change(
    new_input, new_output, pre.input_size, pre.output_size
)
```

Where `_apply_dataset_dim_change` orchestrates:

1. Compute `input_delta = max(new_input - pre.input_size, 0)` and `output_delta = max(new_output - pre.output_size, 0)`.
2. If either delta > 0: call `self.network._resize_network_for_dataset(input_size_new, output_size_new)` where the `_new` values are `max(network, dataset)`.
3. Call `self._pad_dataset_for_network(...)` — handles the shrink-side padding and updates `self._active_output_dim`.
4. Build the §3.3 `arch_changes` block from the resize result + the active-dim info.

### 3.5 Old P2-1c artifacts

- **DELETE**: `src/api/lifecycle/architecture_adapter.py` (~250 lines). Its only consumer was `swap_dataset_live`, which now goes direct to the network method.
- **DELETE**: `src/tests/unit/api/test_architecture_adapter.py` (~330 lines). Replaced by network-level + lifecycle-level unit tests in their natural homes.
- **UPDATE**: `src/tests/integration/api/test_swap_dataset_live.py` — P2-1c grow tests need adjustment for random-init (assert delta+shape, not bit-exact zero-init); add new tests for shrink-via-padding and loss-masking.

The §3.3 response shape stays stable for canopy: `arch_changes` still has `{input_delta, output_delta, hidden_preserved, appended_nodes, prepended_layers, abandoned_candidate_pool_size}`. `prepended_layers` is always `[]` (we never prepend); it is preserved as a forward-compatible no-op field rather than removed, since canopy P2-5/P2-6 may already read it.

## 4. Test plan

### 4.1 Network-level unit tests (`tests/unit/cascade_correlation/test_resize_for_dataset.py`)

| Test                                                             | Pin                                                                                                                                                   |
|------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `equal_dim_is_noop`                                              | No tensor mutation when called with current dims                                                                                                      |
| `grow_input_only_expands_output_weights_rows`                    | Shape `[I_new + H, O]`; rows 0..I_old preserved; rows I_old..I_new are random (non-zero, non-equal); rows I_new..I_new+H preserved (shifted from old) |
| `grow_input_only_expands_hidden_unit_weights_with_row_insertion` | Each unit's vector preserves `[0:I_old]`, inserts random at `[I_old:I_new]`, preserves prior-hidden entries at `[I_new:I_new+i]`                      |
| `grow_input_does_not_touch_hidden_unit_bias`                     | Hidden biases (scalars) unchanged across resize                                                                                                       |
| `grow_output_only_appends_columns`                               | `output_weights` shape `[N, O_new]`; `output_bias` shape `[O_new]`; new entries random                                                                |
| `grow_output_does_not_touch_hidden_units`                        | Hidden units bit-equivalent before/after                                                                                                              |
| `grow_both_input_and_output`                                     | Combined deltas; output_weights shape `[I_new+H, O_new]`                                                                                              |
| `shrink_input_raises_with_clear_message`                         | `ValueError` mentioning current/new dims                                                                                                              |
| `shrink_output_raises`                                           | `ValueError`                                                                                                                                          |
| `random_init_uses_random_value_scale`                            | New entries' magnitude scales with `self.random_value_scale` (statistical check across many cells)                                                    |
| `requires_grad_restored_on_growth`                               | `self.output_weights.requires_grad is True` after grow                                                                                                |
| `hidden_unit_weights_stay_detached`                              | `unit["weights"].requires_grad is False` after grow (cascade contract)                                                                                |
| `live_input_size_used_after_first_grow`                          | Two consecutive grows compute correct deltas using `self.input_size`, not `self.config.input_size` (B1 regression)                                    |

### 4.2 Lifecycle-level unit tests (`tests/unit/api/test_dataset_padding.py`)

| Test                                               | Pin                                                                         |
|----------------------------------------------------|-----------------------------------------------------------------------------|
| `pad_input_only_appends_zero_cols_to_x_and_val_x`  | Shape match, zero values, dtype/device preserved                            |
| `pad_output_only_appends_zero_cols_to_y_and_val_y` | Shape match, zero values                                                    |
| `pad_both_input_and_output`                        | Mixed                                                                       |
| `no_pad_when_dataset_matches_network`              | Returns originals (or identical-shape copies) + active dims = network dims  |
| `none_val_tensors_stay_none`                       | If `val_x is None`, return `None` (not padded zero tensor)                  |
| `dataset_larger_than_network_raises`               | Caller must resize first; helper does not auto-grow                         |
| `active_dims_reflect_pre_pad_shape`                | Returned `active_input_dim` / `active_output_dim` = dataset's pre-pad shape |

### 4.3 Training-method loss-masking tests

| Test                                                    | Pin                                                                                                                               |
|---------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `train_output_layer_no_mask_when_active_dim_is_none`    | Existing behaviour preserved (backward compat)                                                                                    |
| `train_output_layer_masks_loss_to_active_output_dim`    | Loss value matches `criterion(output[:, :K], y[:, :K])` for active_output_dim=K                                                   |
| `train_output_layer_does_not_learn_zero_on_padded_dims` | After a few epochs with padded y, network's output on the dead dims stays roughly at its random-init magnitude (no zero-collapse) |
| `train_candidates_masks_residual_to_active_dim`         | Symmetric to output-layer masking                                                                                                 |

### 4.4 Integration tests (`tests/integration/api/test_swap_dataset_live.py`)

P2-1c grow-success tests updated:

| Test                                  | Old expectation                    | New expectation                                     |
|---------------------------------------|------------------------------------|-----------------------------------------------------|
| `grow_input_success`                  | input_delta=2, new rows zero       | input_delta=2, new rows random × random_value_scale |
| `grow_output_success`                 | output_delta=2, new cols zero      | output_delta=2, new cols random                     |
| `grow_both`                           | combined zero                      | combined random                                     |
| `grow_log_line_reports_actual_deltas` | (unchanged — log format unchanged) | (unchanged)                                         |

P2-1c "rejects shrink" tests **DELETED** (no longer applicable — shrink is now supported via padding).

New shrink + pad tests:

| Test                                                           | Pin                                                                                                   |
|----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `swap_input_shrink_pads_dataset_x`                             | `_train_x` shape after swap = `[batch, network.input_size]` with zero columns appended                |
| `swap_output_shrink_pads_dataset_y_and_sets_active_output_dim` | `_train_y` padded; `mgr._active_output_dim` = pre-pad output count                                    |
| `swap_mixed_input_grow_output_shrink`                          | Compositional: network grows on input side, dataset pads on output side                               |
| `swap_response_shape_stable_for_canopy`                        | `arch_changes` keys unchanged; `prepended_layers == []`                                               |
| `swap_emits_completion_log_with_active_dim`                    | (optional) log includes the active output dim                                                         |
| `cancel_mid_swap_still_works`                                  | P2-1b cancel + rollback regression — pre-swap state restored on cancel during dataset fetch or resize |
| `forward_after_resize_consumes_padded_input_correctly`         | Real network, swap with input shrink, `forward(padded_x)` produces valid output (no shape error)      |

### 4.5 Regression coverage

- Full pause/stop interrupt tests (`test_pause_stop_actually_interrupts.py`) — P2-PRE-1 must remain healthy.
- P2-1b cancel mechanism integration tests — must continue to pass.
- Broader `tests/unit/` and adjacent `tests/integration/api/` — no regressions.

## 5. Out of scope

| Item                                                                                       | Where it lands                                                                                                                                                   |
|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Snapshot/replay serialization of post-resize network                                       | P2-2 / P2-3 (the new layout is simpler than §3.6's adapter chain — snapshot just dumps `output_weights`, `output_bias`, `hidden_units` as before; no new fields) |
| History `dataset_swap` event                                                               | P2-2                                                                                                                                                             |
| Canopy Experimental Functions toggle UI                                                    | P2-4                                                                                                                                                             |
| Canopy Live Dataset Switch button                                                          | P2-5                                                                                                                                                             |
| Adaptive `random_value_scale` per swap (e.g. inversely proportional to new-dim size)       | Future work — current PR uses a single scale matching construction                                                                                               |
| Auto-shrink the network when the user explicitly asks (separate "Compact Network" feature) | Out of Phase 2 entirely                                                                                                                                          |

## 6. Migration / risk

**Breaking changes**: none externally. The §3.3 response shape is preserved (`prepended_layers` is now a stable empty list rather than reserved-for-future-use).

**Internal removals**: `architecture_adapter.py` and its test file. No other module imports from `architecture_adapter`; verified by grep before delete.

**P2-1c tests that will need updates**:

- `tests/unit/api/test_architecture_adapter.py` — DELETED.
- `tests/integration/api/test_swap_dataset_live.py::test_swap_dataset_live_rejects_input_shrink` — DELETED (shrink now supported).
- `tests/integration/api/test_swap_dataset_live.py::test_swap_dataset_live_rejects_output_shrink` — DELETED.
- Grow-success assertions: change "new rows == 0" to "new rows finite, not equal to original rows" (probabilistic but strong).

**Cross-repo references** to retire: `juniper-canopy/notes/ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md` §3.6 and §7 P2-1d row mention the adapter approach. The companion canopy PR patches both to reference this doc.

## 7. PR shape (single PR, per O4)

The cascor PR `phase2/p2-1d-resize-and-pad` lands the whole change as one coherent unit:

1. Polish `_resize_network_for_dataset` (+ helpers) in `cascade_correlation.py` — B1–B6 fixes.
2. Add `_pad_dataset_for_network` to `manager.py`.
3. Thread `active_output_dim` through `train_output_layer` + `train_candidates`.
4. Rewire `swap_dataset_live` step 8.
5. Delete `architecture_adapter.py` + `test_architecture_adapter.py`.
6. Add new tests (§4.1–§4.4).
7. Update existing P2-1c swap tests for the new contract.

Approximate diff size: ~600–900 lines (≈250 net delete from old adapter; ≈400 net add from new tests + masking; ≈100 net add for production network methods replacing prototype).

A companion **canopy** PR (separate, doc-only) patches `juniper-canopy/notes/ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md` to retire §3.6's adapter approach and reference this design doc.
