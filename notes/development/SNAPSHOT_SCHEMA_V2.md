# Snapshot Schema V2 — Migration Notes & FAQ

**Created**: 2026-05-03
**Status**: Active (V2 schema shipped via CAN-015g)
**Project**: Juniper Cascade Correlation Neural Network
**Tracks**: CAN-015g (Replay V2 — per-epoch weight history)

---

## What this doc covers

V2 is the snapshot file format produced by the cascor training loop
once CAN-015g has landed. It extends the V1 format with a new
`history/weights/` group that holds per-sample weight tensors so
canopy's replay player can render decision-boundary playback as the
user scrubs through training.

This doc answers the questions reviewers, ops, and downstream
consumers will reasonably ask:

- What changed on disk?
- Will my V1 snapshots still load?
- Why does scrubbing inside an inter-sample window snap to the
  nearest sample instead of showing the exact epoch?
- How do I tune the on-disk size for a given replay fidelity?

For the implementation roadmap and design rationale see the parent
plan at
[`juniper-ml/notes/JUNIPER_2026-05-04_JUNIPER-ECOSYSTEM_PHASE-6E-DEFERRED-CAN-015GH-DESIGN.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-05-04_JUNIPER-ECOSYSTEM_PHASE-6E-DEFERRED-CAN-015GH-DESIGN.md).

---

## On-disk layout (additive over V1)

V2 adds **only** the `history/weights/` group. All other groups
(`config/`, `parameters/`, `architecture/`, `hidden_units/`,
`history/{train_loss,value_loss,...}/`, `history/hidden_units_added/`,
etc.) are unchanged. V1 readers that don't know about
`history/weights/` ignore it and load the rest of the file
identically to before.

```text
history/weights/
  meta/                                 (subgroup, attrs only)
    schema_version (int64)              always 2 for this writer
    sampling_strategy (str)             "adaptive" | "every_n" | "trigger"
    sampling_interval (int64)           N (epochs); 0 == trigger-only
    num_samples (int64)
  sample_indices (int64 dataset)        epoch numbers per sample
  output_weights/                       (per-sample subgroup)
    0000  (float32 [in + hid_at_sample_0, out])
    0001  (float32 [in + hid_at_sample_1, out])
    ...
  output_bias/                          (per-sample subgroup)
    0000  (float32 [out])
    ...
  hidden_units/
    0000/                               (one subgroup per unit)
      first_sample_index (int64 attr)   sample-list index, NOT epoch
      activation (str attr)
      weights/                          (per-sample subgroup)
        0050  (float32 [in + cascade_index])
      bias/                             (per-sample subgroup)
        0050  (float32 length-1 array)  scalar wrapped because h5py
                                        rejects compression on 0-d
```

**Why per-sample subgroups instead of single 3D arrays?** Each
cascade-grow event widens the output layer (`[in + hid_at_sample, out]`),
so there's no fixed shape across samples after the first growth event.
Per-sample datasets handle the variable shape natively without
zero-padding.

**`first_sample_index` is a sample-list index, not an epoch number.**
A unit added at sample 5 (which might be epoch 250 with `N=50`) has
`first_sample_index=5`. Slicing per-unit arrays uses
`local_idx = current_sample - first_sample_index` directly. This
convention is documented inline in the cascor lifecycle's
`_WeightCache._build_payload` and `_WeightHistoryRecorder` and is
critical to keep stable across all V2 consumers.

---

## V1 → V2 backward compatibility

V2 is **strictly additive**: V1 snapshots continue to load
identically to before, and V2 snapshots can be read by pre-CAN-015g
clients (they'll just ignore the `history/weights/` group).

| Scenario | Behaviour |
|---|---|
| V1 snapshot loaded by V2 cascor | Loads normally. `network.weight_history` is `None`. Replay session emits V1-shape events (no `is_sample_boundary`, no `weights`). |
| V2 snapshot loaded by V2 cascor | Loads normally. `network.weight_history` is populated. Replay session emits V2-shape events with `weights` blocks at sample-boundary epochs. |
| V2 snapshot loaded by pre-CAN-015g cascor (V1-only loader) | Loads normally. The unknown `history/weights/` group is ignored. |
| V2 snapshot with `schema_version != 2` | Loader logs a WARNING and degrades to V1 behaviour (`weight_history = None`). Forward-compatibility for an eventual V3. |
| V1 canopy client connects to a V2 cascor | Existing metric events flow normally. The new `weights` block on sample-boundary `epoch_end` events is silently ignored by the older client. |

The strict-additive guarantee is verified by:

- `test_snapshot_weight_history.py::TestV1BackwardCompat` — V1
  snapshots still save/load with no `weights/` group; loader leaves
  `weight_history` `None`.
- `test_snapshot_weight_history.py::test_unsupported_schema_version_degrades_gracefully`
  — future schema versions don't crash the loader.

---

## Storage strategy and sampling decisions

Naive per-epoch persistence would produce a ~400 GB snapshot for a
10000-epoch run with a 50-unit network (5×10⁶ values × 8 bytes ×
10000 epochs). V2 uses **adaptive subsampling** to keep snapshots
practical:

1. **Every Nth epoch** (default `N=50`, configurable via
   `CascadeCorrelationConfig.weight_history_sampling_interval`).
2. **Every cascade-grow event** — always sampled regardless of N
   because growth events are the visually-meaningful moments.
3. **Final epoch** — captured at training completion so the last
   sample reflects the truly-final weights.

A 10000-epoch run with default `N=50` and ~10 cascade-grow events
produces ~210 samples (200 periodic + 10 cascade) instead of
10000 — a 50× reduction in tensor count, with the cascade narrative
arc preserved exactly.

### Memory ceiling

The lifecycle's `_WeightHistoryRecorder` enforces a soft cap of
**1000 samples** by default
(`CascadeCorrelationConfig.weight_history_max_samples`). On
overflow it decimates inter-cascade samples by 2× while always
retaining cascade-add and terminal samples. The recorded
`sampling_interval` doubles to reflect the post-decimation gap so
loaders / canopy can interpret what they're seeing.

---

## "Snap to sample" UX behaviour (FAQ)

### Q: I scrubbed the player to epoch 137, but the decision-boundary view shows epoch 100. Why?

V2 captures weights at sample-boundary epochs only — epoch 100, 150,
200, … with default `N=50`, plus every cascade-grow event. Epoch
137 is between samples 100 and 150 (and not a cascade boundary), so
no exact tensors exist for it.

The player's behaviour is **snap to nearest sample**: scrubbing
inside an inter-sample window shows the most-recent sampled state
rather than interpolating. This is a deliberate design choice:
linear interpolation between weight tensors is mathematically
meaningless (the model state at epoch 137 is **not** a 0.74 / 0.26
mix of epochs 100 and 150) and would mislead users into thinking
they were seeing a real intermediate state.

### Q: Can I get exact-epoch playback?

Yes — set `weight_history_sampling_interval=1` at network creation
time (or via PATCH `/v1/training/params` mid-run). Every epoch is
captured. This is functionally Option A in the design's
storage-strategy table; expect the snapshot file size to grow
roughly proportionally with the number of epochs.

### Q: What if I only care about cascade-add events?

Set `weight_history_sampling_interval=0`. The periodic trigger is
disabled and only cascade-grow events sample. This is Option D in
the storage-strategy table — smallest file, no inter-growth
playback fidelity.

### Q: What does the player UI show when scrubbing inside an inter-sample window?

Per the canopy g-4 design, the player's `last-sample-readout`
displays `last sample: epoch <epoch> (<N> buffered)` and the
decision-boundary view (when g-4-V2 / g-7 lands) renders against
that last sample. The scrubber position itself moves freely through
the metric-event timeline; only the weight-dependent renders snap.

### Q: How do I see which exact epochs were sampled?

The `/v1/snapshots/{id}/replay` response includes a
`weight_sampling.sample_epochs` array with the epoch numbers that
correspond to each sample-boundary. Canopy uses this to position
sample markers on the scrubber.

---

## Configuration cheat sheet

Two new tunables on `CascadeCorrelationConfig` (defaults shown):

| Field | Default | Effect |
|---|---|---|
| `weight_history_sampling_interval` | `50` | Capture every Nth epoch. `N=1` = every epoch (Option A). `N=0` = disable periodic, fall back to cascade-add only (Option D). |
| `weight_history_max_samples` | `1000` | Soft cap before decimation. `0` = unbounded (use with care on long runs). |

Both are runtime-mutable via `PATCH /v1/training/params` — changes
take effect at the next training-monitor `on_epoch_end` event.

### Snapshot-size budget by configuration

Approximate snapshot size for a 10000-epoch run with a 50-unit
network on the spiral problem (input=2, output=1):

| Config | Approx. samples | Approx. weights size | Total file size |
|---|---|---|---|
| V1 (no weights) | 0 | 0 | ~50 KB |
| `N=50, max=1000` (default) | 210 | ~3 MB | ~3 MB |
| `N=10, max=1000` | 1000 (decimation kicks in once) | ~14 MB | ~14 MB |
| `N=1, max=0` (Option A, unbounded) | 10010 | ~140 MB | ~140 MB |
| `N=0` (Option D, cascade-only) | 10 | ~140 KB | ~200 KB |

These are order-of-magnitude estimates from the toy fixtures —
production-sized networks (1000+ hidden units) will be 20×–100× larger.

---

## How to verify a snapshot's V2 status

### Via h5py

```python
import h5py
with h5py.File("snapshot.h5", "r") as f:
    if "history/weights" not in f:
        print("V1 snapshot — no per-sample weight history")
    else:
        meta = f["history/weights/meta"]
        print(f"V2 snapshot, schema_version={int(meta.attrs['schema_version'])}")
        print(f"  strategy={meta.attrs['sampling_strategy'].decode('utf-8')}")
        print(f"  interval={int(meta.attrs['sampling_interval'])}")
        print(f"  num_samples={int(meta.attrs['num_samples'])}")
```

### Via the cascor REST API

```bash
curl -s http://localhost:8200/v1/snapshots/<id>/replay -X POST \
  | jq '.data.weights_available, .data.weight_sampling'
```

The response's `data.weights_available` is `true` iff the loaded
snapshot has a usable V2 `history/weights/` group; false otherwise
(V1 snapshot, or V2 with `num_samples == 0`, or unknown schema
version).

---

## When to expect snapshot-size growth

V2 snapshots are noticeably larger than V1 only when training
produces many sample-boundary epochs **and** the network is
large. Two regimes:

- **Small networks, default config**: V2 is ~1.05× to ~1.5× the V1
  size. The new HDF5 group adds ~10–50 KB of metadata + tiny
  tensors. The g-1 size-regression test asserts a 1.05×
  ceiling on the empty-weight-history fast path.
- **Production networks, default config**: V2 size scales with
  `num_samples × tensor_size`. For a 50-unit network with default
  `N=50`, a 10000-epoch run produces ~210 samples. Each sample
  carries the full output-layer weight tensor (~50×1 floats) plus
  per-unit weights (~10–60 floats each) = a few MB of weight data
  per snapshot. Negligible compared to typical ML model sizes.

If file size becomes a concern, three knobs to turn:

1. **Increase `weight_history_sampling_interval`** — fewer samples,
   coarser playback granularity, smaller file.
2. **Lower `weight_history_max_samples`** — cap decimation kicks
   in earlier; growth-event samples stay; inter-growth fidelity
   drops.
3. **Set `weight_history_sampling_interval=0`** — cascade-only
   capture; smallest possible V2 size.

---

## Glossary

| Term | Meaning |
|---|---|
| **V1 snapshot** | Pre-CAN-015g schema. Metric arrays + topology metadata only. No `history/weights/` group. |
| **V2 snapshot** | Post-CAN-015g schema. Strictly additive — V1 readers ignore the new group. |
| **Sample-boundary epoch** | An epoch at which `_WeightHistoryRecorder` captured weight tensors. Determined by the periodic trigger (every Nth) plus cascade-grow events plus the terminal-epoch capture. |
| **`first_sample_index`** | Per-hidden-unit field in the V2 layout. The 0-based index into `sample_indices` at which the unit first appeared. **Not** an epoch number. |
| **Snap to sample** | The player's behaviour when the scrubber is between two sample boundaries — the weight-dependent views render against the most-recent sampled state, not an interpolation. |
| **Decimation** | The memory-ceiling enforcement that drops every other inter-cascade sample when `weight_history_max_samples` is exceeded. Cascade-add and terminal samples are always retained. |

---

## Related work

- **g-1** (#180): serializer — adds the V2 layout above. Schema docs
  in this file are anchored on g-1's implementation.
- **g-2** (#189 — retarget of #184): replay session weight cache +
  extended `state_summary` (`weights_available`,
  `weight_sampling`).
- **g-3** (#190 — retarget of #187): synthetic-event emission of
  base64-encoded weight tensors on sample-boundary epochs.
- **g-6** (#191): the recorder that actually populates
  `network.weight_history` during a training run. Without g-6, V2
  replay only works against ad-hoc fixtures — production training
  produces V1 snapshots.
- **canopy g-4** (canopy #220): WS bridge + player-panel
  `replay-weight-buffer` Store + V2 indicator badge.
- **deferred g-4 V2 / g-7**: the actual decision-boundary playback
  rendering. The infrastructure shipped by g-4 makes this a pure
  rendering refactor of `decision_boundary.py` and
  `network_evolution.py`.
