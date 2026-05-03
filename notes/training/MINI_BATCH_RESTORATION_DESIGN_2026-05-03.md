# juniper-cascor — Mini-Batch Training Restoration: Design

**Date:** 2026-05-03
**Status:** Design draft — pending senior-engineer review and sub-track entry-plan
**Sub-track (proposed):** `TRAIN-ARCH-01: Cascor mini-batch training restoration`
**Related:**

- juniper-ml#189 — mini-batch design doc that surfaced the gap (parent program)
- juniper-cascor#188 — METRICS-MON R5.4-pre (training counters, train-step
  histogram, worker→Prometheus bridge) — already merged on `main`
  (commit `9348a26`)
- juniper-deploy#48 — METRICS-MON R5.1 SLO catalog
- juniper-deploy#49 — R5.1 SLO catalog fixup PR
- `notes/observability/HISTOGRAM_BUCKETS_RATIONALE_2026-05-02.md` §6 —
  train-step histogram boundary discussion (R5.4-pre)

---

## 1. Executive summary

While drafting a metrics-instrumentation design (juniper-ml#189), an
investigating agent found that cascor's two trainers are **full-batch
end-to-end**:

- `src/cascade_correlation/cascade_correlation.py:1638` — output-layer
  trainer: `for epoch in range(epochs):` with full-tensor forward +
  `loss.backward()` + `optimizer.step()` per epoch.
- `src/candidate_unit/candidate_unit.py:564` — candidate-unit trainer:
  same pattern (full-batch correlation calculation; manual gradient
  descent via `loss.backward()` and an explicit `weights -= lr * grad`).
- No `DataLoader`, no `TensorDataset`, no mini-batch idioms anywhere
  outside test code (production code paths only contain `batch_size` as
  a tensor-shape label, not as a hyperparameter).

The user's framing was **"this is a regression — mini-batch training was
once a capability and has been lost."** §2 below disconfirms that
framing on the basis of git archaeology and a legacy-archive sweep:
mini-batching was **never** present in any cascor lineage in this
project, including the consolidated `juniper-legacy/` archive. The
correct framing is therefore **absence, not regression** — a capability
that is missing and is now being added for the first time.

The proposed restoration adds two new config knobs (`use_mini_batch`,
default `True`; `mini_batch_size`, sane default to be ratified — see
§5) backed by constants in `src/cascor_constants/`, plus a guarded
mini-batch iteration path in the **output-layer trainer**. The
**candidate-unit trainer is intentionally NOT mini-batched** (see §4.5
— Pearson correlation needs full-batch statistics). Estimated scope:
**3–4 PRs** (constants + config + output trainer + test suite). Effort:
**Tier-2** (algorithmic edit to a hot path with reproducibility and
metrics-throttle implications). Cross-cutting impact: the existing
R5.4-pre per-epoch histogram throttle (every 25th epoch) becomes
inadequate when each epoch contains N mini-batch steps; §8 proposes a
follow-up.

## 2. Investigation: was mini-batch ever there?

**This is the load-bearing section.** The user explicitly framed the
work as a regression, but a regression vs. a never-implemented gap
leads to very different doc framings, sequencing, and risk profiles.

### 2.1 Method

Searches performed at the worktree off `origin/main` (HEAD `9348a26`):

```bash
git log --all -S "DataLoader"     -- src/   # who ever added DataLoader
git log --all -S "TensorDataset"  -- src/
git log --all -S "mini_batch"     -- src/
git log --all -S "minibatch"      -- src/
git log --all -S "batch_size"     -- src/cascade_correlation/
git log --all --oneline -- src/cascade_correlation/cascade_correlation.py | head -50
git show 2076d21 -- src/cascade_correlation/cascade_correlation.py   # initial commit
```

Plus a sweep of the `juniper-legacy/` consolidated archive at
`/home/pcalnon/Development/python/Juniper/juniper-legacy/`:

- `JuniperCascor/juniper_cascor/src/cascade_correlation/cascade_correlation.py`
- `JuniperCascor/juniper_cascor/src/candidate_unit/candidate_unit.py`
- `JuniperCascor/juniper_cascor/src/cascade_correlation/backups/cascade_correlation-ORIG.py`
- `JuniperLegacy/src/prototypes/juniper_cascor/src/cascade_correlation/cascade_correlation.py`
- `JuniperLegacy/src/prototypes/cascor_spiral/src/cascade_correlation/cascade_correlation.py`

### 2.2 Findings

#### 2.2.1 The current repo

- `git log --all -S "DataLoader" -- src/` returns **zero hits** in
  cascor source. The five hits in `git log --all -S "TensorDataset"`
  are all in `src/tests/` (test fixtures), not in production trainer
  code.
- `git log --all -S "mini_batch" -- src/` returns **zero hits**.
- `git log --all -S "minibatch" -- src/` returns hits only in comments
  (e.g. "Calculate the correlation … for the entire minibatch" in
  `_calculate_correlation`, and trace-log strings) — no code that ever
  iterated mini-batches.
- The **initial commit** (`2076d21`, 2025-12-08, "initial commit of
  standalone JuniperCascor") already contained `for epoch in
  range(epochs):` with the same full-batch shape:

  ```python
  for epoch in range(epochs):
      hidden_outputs = []
      for unit in self.hidden_units:
          ...
      output_input = (torch.cat([x] + hidden_outputs, dim=1) ...)
      output = output_layer(output_input)
      loss = criterion(output, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  ```

  **The trainer was born full-batch.** No prior commit on any branch
  introduced mini-batch iteration and no prior commit removed it.

#### 2.2.2 The legacy archive

Every cascor variant in `juniper-legacy/` — the pre-polyrepo
`JuniperCascor` source, the `JuniperLegacy/src/prototypes/cascor_spiral`
prototype, the `JuniperLegacy/src/prototypes/juniper_cascor` prototype,
and the `cascade_correlation-ORIG.py` backup — uses the identical
full-batch `for epoch in range(epochs):` pattern at the analogous loci.
**None** contains `DataLoader`, `TensorDataset`, `mini_batch`, or any
mini-batch iteration construct.

#### 2.2.3 The DataLoader hits in `juniper-legacy/`

`grep -l "DataLoader"` against `juniper-legacy/` does return five
files:

```text
JuniperLegacy/refs/mnist_dynamic_classifier.py
JuniperLegacy/refs/dynamic_classifier.py
JuniperLegacy/refs/mnist_classifier.py
JuniperLegacy/refs/full_dynamic_classifier.py
JuniperLegacy/prompts/info/dynamic_layer_nn-analysis.py
```

These are **not cascade-correlation lineage**. They are reference
implementations of a generic dynamic-classifier MLP / MNIST trainer
under `JuniperLegacy/refs/`, used as exemplar code for a different
algorithmic family. None mention `cascade`, `correlation`, or any
candidate-unit / cascade-correlation construct. The DataLoader
appearance there does **not** support a "mini-batch was once in cascor"
claim.

### 2.3 Verdict — absence, not regression

**Mini-batch training was never present in cascor in this project.**
The user's "regression" framing is not supported by the code history
or the legacy archive. The doc therefore frames this work as an
**absence-fill / feature addition**, not a regression repair. This
distinction matters for:

- **Risk framing**: there is no prior reference behaviour to recover
  bit-exactly; we are introducing new behaviour for the first time.
- **Sequencing**: no urgency from a "we lost something users depended
  on" angle; the urgency, if any, comes from §4's memory and
  convergence arguments and from the §8 metrics intersection.
- **Default**: the user requested `use_mini_batch=True` as the default,
  which means the absence-fill **also flips the default behaviour** of
  every existing call site. This is a semantic change to the
  public-facing training contract and warrants explicit release-note
  treatment and a validation-vs-baseline gate (see §7, §9).

**Recommendation to the senior engineer:** consider whether the default
should be `False` for one minor version (opt-in mini-batch) before
flipping to `True`, given there is no prior mini-batch reference to
match. Listed as Open Question Q1 in §9.

## 3. Current trainer architecture (concrete)

### 3.1 Output-layer trainer

`src/cascade_correlation/cascade_correlation.py:1556..1678`,
`CascadeCorrelationNetwork.train_output_layer`:

```python
def train_output_layer(self, x, y, epochs=None, on_epoch_callback=None):
    epochs = (epochs, _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS)[epochs is None]
    ...
    criterion = nn.MSELoss()
    output_layer = nn.Linear(input_size, self.output_size)
    # weights/bias seeded from self.output_weights / self.output_bias
    self.output_optimizer = self._create_optimizer(output_layer.parameters())
    optimizer = self.output_optimizer

    # CR-060: Hoist hidden-output computation above the epoch loop
    output_input = self._compute_hidden_outputs(x)        # full-batch, shape (N, F+H)

    for epoch in range(epochs):
        output = output_layer(output_input)               # full-batch fwd
        loss = criterion(output, y)                       # full-batch MSE
        optimizer.zero_grad()
        loss.backward()                                   # full-batch grad
        optimizer.step()
        if self._network_display_progress(epoch):
            self.logger.info(...)
        # Throttled callback for real-time metrics emission
        _cb = on_epoch_callback or getattr(self, "_output_epoch_callback", None)
        if _cb is not None and (epoch % 25 == 0 or epoch == epochs - 1):
            _cb(epoch=epoch + 1, epochs=epochs, loss=loss.item())
```

Tensors `x`, `y`, `output_input`, `output`, `loss` are full-batch.
Optimizer is constructed by `_create_optimizer` (line 2589) from
`OptimizerConfig`; the active default is `torch.optim.Adam` per the
factory default in `cascade_correlation_config.py`. No device transfer
inside the loop (PyTorch infers from the parameters' device). The
metrics throttle "every 25th epoch" lives at line **1655** in the
trainer and corresponds to the per-epoch `_output_training_callback`
in `src/api/lifecycle/manager.py:681..711` (R5.4-pre).

### 3.2 Candidate-unit trainer

`src/candidate_unit/candidate_unit.py:509..680`,
`CandidateUnit.train_detailed`:

```python
for epoch in range(epochs):
    output = self.forward(x)                              # full-batch fwd
    candidate_training_result = self._get_correlations(output, residual_error)
    candidate_parameters_update = CandidateParametersUpdate(...)
    self._update_weights_and_bias(candidate_parameters_update)
    ...
    if self.early_stopping:
        # patience tracking
        ...
    _pcb = progress_callback or getattr(self, "_progress_callback", None)
    if _pcb is not None and (epoch % 50 == 0 or epoch == epochs - 1):
        _pcb(...)
```

`_update_weights_and_bias` (line 958..1050) does **NOT** use a
PyTorch `optim.Optimizer`. Instead it:

1. Clones `weights`/`bias` and calls `requires_grad_(True)` on the
   clone.
2. Computes correlation on the **full batch** —
   `output_mean = output.mean()`, `error_mean = error_slice.mean()`,
   `output_centered = output - output_mean`, etc.
3. Calls `loss.backward()` to populate `weights_param.grad` and
   `bias_param.grad`.
4. Applies the update manually: `self.weights -= lr * grad_w`.

The relevant correlation primitive is `_calculate_correlation`
(line 878..950), which computes Pearson correlation by mean-centering
the **entire** `output` and `residual_error` tensors and calling
`torch.dot` / `torch.linalg.norm` on the full vectors.

### 3.3 Other trainer entry points

- `src/cascade_correlation/cascade_correlation.py:2775` —
  `for epoch in range(max_epochs):` is the **outer growth loop**, not a
  per-epoch SGD loop. It calls `train_output_layer` and
  `train_candidates` per outer iteration; it is not in scope for this
  design.
- `src/profiling/logging_utils.py:16, 110` — synthetic test loops in a
  profiling helper; not production trainers.

### 3.4 Where data enters

- Output trainer: `train_output_layer(x, y, epochs)` is called from
  `cascade_correlation.fit(x_train, y_train, ...)`
  (`cascade_correlation.py:1396..`) with `x` and `y` arriving as
  already-fully-resident tensors at the configured device.
- Candidate trainer: `CandidateUnit.train_detailed(x, residual_error, ...)`
  is called via `_execute_candidate_training` after
  `_prepare_candidate_input(x)` (full-batch) and
  `_generate_candidate_tasks(...)` (also full-batch — see §4.5).

There is **no streaming / chunked input** anywhere; the trainer
assumes the entire training tensor fits in memory.

## 4. Why mini-batch matters

### 4.1 Memory

For a dataset with N samples × F input features in float32, the input
tensor is `4 N F` bytes. With H hidden units, `_compute_hidden_outputs`
materializes a `(N, F + H)` tensor — `4 N (F+H)` bytes — and the
forward activation, loss, and gradient buffers add roughly the same
again per layer. For N=10 000, F=100, H=50, this is ~6 MB per buffer,
trivial. For N=1 000 000 (a common dataset size juniper-data is
designed to scale to per its data-contract), this becomes ~600 MB per
buffer × ~3 buffers = ~1.8 GB, which is the OOM threshold on a 4 GB
device. The point at which full-batch breaks is dataset-dependent and
algorithm-internal (cascade-correlation grows H over time, so the
buffer *grows* across the run).

### 4.2 Convergence

Standard result: SGD-style mini-batch updates introduce gradient noise
that helps escape sharp minima and tends to generalise better than
full-batch GD. For cascor's **output-layer** training (a linear
regression-style fit minimising MSE over `nn.Linear`), this is a clean
benefit. For the **candidate phase**, the picture is different — see
§4.5.

### 4.3 GPU utilization

Full-batch on small datasets underutilizes the GPU; full-batch on
large datasets overruns device memory. A mini-batch sized to match the
device's L2 / SM budget keeps utilization high. cascor today does not
use `pin_memory` or `non_blocking` transfers — see §6.5.

### 4.4 cascor-specific motivations

- The R5.4-pre `juniper_cascor_training_step_duration_seconds`
  histogram measures one full-batch epoch as one "step". For large
  datasets this can exceed the SLO 3.4 target of `p95 < 5 s` simply
  because the step itself is the entire dataset.  Mini-batching
  decouples the histogram step from the dataset size and makes the SLO
  achievable on a wider range of datasets.
- The candidate phase trains `candidate_pool_size` (default 32) units
  in parallel via the manager-proxied process pool; smaller per-task
  memory footprints would relax the
  `PARALLEL_CANDIDATE_TRAINING_FIX_PLAN` data-duplication pressure
  (`notes/development/PARALLEL_CANDIDATE_TRAINING_FIX_PLAN.md` RC-3).

### 4.5 The cascade-correlation full-batch constraint (CRITICAL)

**The candidate phase trains weights to maximise the Pearson
correlation between candidate output and the network's residual
error.** The Pearson correlation is, by definition, a **statistic over
the full sample**:

```text
corr = |Σ_i (o_i - ō)(e_i - ē)| / √( Σ_i (o_i - ō)² × Σ_i (e_i - ē)² )
```

`ō` and `ē` are the **batch means**.
`_calculate_correlation` computes them as `output.mean()` and
`error_slice.mean()` over the full vectors
(`candidate_unit.py:1013..1014`).

**If we mini-batch the candidate phase, the per-step `output.mean()`
and `error.mean()` are *biased* estimators of the population means.**
This makes the per-step correlation gradient a biased estimator of the
true correlation gradient — not just noisy (the way SGD is noisy
relative to GD), but pointing the wrong direction in expectation when
batch means drift from population means. The resulting candidate
selection (which compares correlations across candidates) becomes
unreliable: which candidate wins depends on which mini-batch happened
to be sampled.

There are mitigations in the literature (Welford-style running
statistics across mini-batches, or computing the correlation on the
full batch but applying the gradient mini-batch-wise), but these are
**non-trivial algorithmic changes** to cascade-correlation, not a
straightforward DataLoader insertion.

**Design decision (this doc):**

- **Output-layer trainer:** mini-batch is straightforward and beneficial.
  Restore here.
- **Candidate-unit trainer:** keep full-batch by default. Expose a
  separate config knob (`use_mini_batch_candidate`, default `False`)
  so a future research track can experiment, but treat that as
  out-of-scope for this restoration.

## 5. Proposed config schema

### 5.1 Constants

Proposed file: `src/cascor_constants/constants_model/constants_model.py`
(co-located with the existing `_PROJECT_MODEL_OUTPUT_EPOCHS`,
`_PROJECT_MODEL_EPOCHS_MAX`).

```python
# Mini-batch toggle and size for the output-layer trainer.
# Cascor was full-batch from the initial commit (2076d21, 2025-12-08).
# These constants are introduced by TRAIN-ARCH-01 (2026-05-03) per
# notes/training/MINI_BATCH_RESTORATION_DESIGN_2026-05-03.md.
_PROJECT_MODEL_USE_MINI_BATCH = True
_PROJECT_MODEL_MINI_BATCH_SIZE = 256

# Candidate-phase default is FULL-BATCH due to the Pearson-correlation
# population-statistic constraint (§4.5 of the design doc). Exposed as
# a research toggle only; do not flip without algorithmic work.
_PROJECT_MODEL_USE_MINI_BATCH_CANDIDATE = False
```

Then chained through the existing `constants_problem` /
`constants.py` ladder so the public-facing
`_CASCADE_CORRELATION_NETWORK_*` constants (line 859 region) get
analogues:

```python
_CASCADE_CORRELATION_NETWORK_USE_MINI_BATCH         = _SPIRAL_PROBLEM_USE_MINI_BATCH
_CASCADE_CORRELATION_NETWORK_MINI_BATCH_SIZE        = _SPIRAL_PROBLEM_MINI_BATCH_SIZE
_CASCADE_CORRELATION_NETWORK_USE_MINI_BATCH_CANDIDATE = _SPIRAL_PROBLEM_USE_MINI_BATCH_CANDIDATE
```

(matching the existing constants-cascade pattern of
`_PROJECT_MODEL_X → _CASCOR_X → _SPIRAL_PROBLEM_X →
_CASCADE_CORRELATION_NETWORK_X`).

### 5.2 Default values — justification

`_PROJECT_MODEL_MINI_BATCH_SIZE = 256` is proposed because:

- The two-spiral canonical task uses 194 samples (per
  `src/spiral_problem/spiral_problem.py` defaults). 256 falls back to
  full-batch via §5.3 validation, preserving current behaviour for the
  smoke-test workload.
- For larger workloads (e.g. a 10k-sample synthetic dataset), 256 is a
  PyTorch convention sweet spot: enough to give clean gradient
  estimates, small enough to keep the per-step memory footprint <100
  MB on typical configurations, divisible by common GPU SM counts.
- Non-power-of-two values (e.g. 200) hurt GPU kernel selection;
  power-of-two values are the convention.

**Alternatives considered:** 32, 64, 128 (too small for cascor's
typical N-sample workload; produces too many gradient updates per
epoch and amplifies the metrics-throttle problem in §8); 512 (too
close to the spiral dataset size and to typical small-workload N
values, undermining the §5.3 fallback).

**Open question Q2 (§9):** ratify 256 vs. 128.

### 5.3 Validation

In `CascadeCorrelationConfig.__init__`:

```python
if self.mini_batch_size <= 0:
    raise ValueError(
        f"mini_batch_size must be positive, got {self.mini_batch_size}"
    )
# mini_batch_size > dataset_size case is RUNTIME, not config-time —
# it is handled in the trainer (§5.4) because dataset_size isn't
# known until fit() is called.
```

### 5.4 Runtime fallback

When `use_mini_batch=True` and `mini_batch_size >= x.shape[0]`, the
trainer logs an INFO line and **falls back to full-batch** silently for
that fit() call. Rationale: erroring would be hostile to the spiral
smoke-test workload (N=194 < default 256). Logging makes the fallback
auditable.

### 5.5 Per-stage independence

Two independent toggles (`use_mini_batch` for output,
`use_mini_batch_candidate` for candidate) per §4.5. They share a
single `mini_batch_size` field; if the candidate-phase ever flips its
toggle on, it uses the same mini-batch size as the output trainer.
(Open question Q3 (§9): do we need a separate
`mini_batch_size_candidate`? Likely yes, but defer until
candidate-phase research begins.)

### 5.6 CascadeCorrelationConfig dataclass additions

`src/cascade_correlation/cascade_correlation_config/cascade_correlation_config.py:131`:

```python
class CascadeCorrelationConfig:
    def __init__(
        self,
        ...
        use_mini_batch: bool = _CASCADE_CORRELATION_NETWORK_USE_MINI_BATCH,
        mini_batch_size: int = _CASCADE_CORRELATION_NETWORK_MINI_BATCH_SIZE,
        use_mini_batch_candidate: bool = _CASCADE_CORRELATION_NETWORK_USE_MINI_BATCH_CANDIDATE,
        ...
    ):
        ...
        self.use_mini_batch = use_mini_batch
        self.mini_batch_size = mini_batch_size
        self.use_mini_batch_candidate = use_mini_batch_candidate
```

(Mirrors the existing pattern at lines 144–227 — explicit constructor
args defaulting to constants, assigned to `self.X` for serialization
parity.)

## 6. Implementation approach

### 6.1 Output-layer trainer — mini-batch path

In `train_output_layer`, after the CR-060 hoist of `output_input`:

```python
use_mb = self.config.use_mini_batch and (output_input.shape[0] > self.config.mini_batch_size)
if use_mb:
    bsz = self.config.mini_batch_size
    n = output_input.shape[0]
    # Reproducibility: shuffle indices with the network's RNG generator
    generator = torch.Generator(device=output_input.device).manual_seed(self.random_seed)
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        n_steps = 0
        for start in range(0, n, bsz):
            idx = perm[start:start+bsz]
            xb = output_input[idx]
            yb = y[idx]
            output = output_layer(xb)
            loss = criterion(output, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * idx.numel()
            n_steps += 1
        epoch_loss /= n
        # Throttled callback uses the epoch-mean loss for per-epoch parity
        _cb = on_epoch_callback or getattr(self, "_output_epoch_callback", None)
        if _cb is not None and (epoch % 25 == 0 or epoch == epochs - 1):
            _cb(epoch=epoch+1, epochs=epochs, loss=epoch_loss)
else:
    # EXISTING full-batch path (unchanged) — guards bit-exact reproducibility
    for epoch in range(epochs):
        ...
```

### 6.2 Iterator choice — TensorDataset+DataLoader vs. index-permutation

Two viable approaches:

1. **`torch.utils.data.TensorDataset` + `DataLoader`** —
   PyTorch-idiomatic, supports `pin_memory`, `num_workers`, gives
   "free" batch shuffling and drop_last semantics.
2. **Index-permutation loop** (shown in §6.1) — no DataLoader
   construction overhead per `fit()`; smaller blast radius for
   debugging; doesn't introduce a `num_workers` worker-pool that would
   collide with the existing candidate-phase
   `multiprocessing` pool (see
   `PARALLEL_CANDIDATE_TRAINING_FIX_PLAN.md` for the existing
   process-server architecture).

**Recommendation:** index-permutation in v1, defer DataLoader to v2.
The DataLoader is overkill for in-memory training tensors and would
re-introduce a worker-pool concern this codebase has invested heavily
in disciplining.

### 6.3 Reproducibility

Cascor seeds via `_initialize_randomness` (`cascade_correlation.py:979`
calling `torch.manual_seed`, `np.random.seed`, `random.seed`, and the
torch CUDA seed). The mini-batch shuffle MUST use the same seed
plumbing — propose a dedicated `torch.Generator` constructed with
`self.random_seed`. **Critical:** for users who run the legacy
full-batch path, results MUST be bit-exact unchanged (regression
guard, §7.1).

### 6.4 Metrics callback adaptation

Two reasonable choices:

- **(a) Per-epoch callback (current behaviour preserved)** — callback
  fires once per epoch with `epoch_loss` as the mean over mini-batch
  steps. Histogram throttle stays at "every 25th epoch". Pro: minimal
  blast radius. Con: histogram observes **one** sample per 25 epochs,
  but each epoch now contains N steps, so the histogram resolution
  drops sharply.
- **(b) Per-step callback (new behaviour)** — callback fires once per
  mini-batch step. Histogram observes **per step**. Throttle drops to
  "every 25th step" or removed entirely. Pro: high-resolution training
  curves, useful for SLO 3.4 enforcement. Con: callback rate × 50–500
  on typical workloads, more pressure on the WS broadcast layer
  (R5.1b WS-sub-ms histograms become more relevant).

**Recommendation:** v1 ships (a) — preserves existing behaviour,
stays out of the metrics-throttle redesign. v2 (a follow-up sub-track,
likely **METRICS-MON R5.5**) introduces (b) with adaptive throttling.
Open question Q4 (§9).

### 6.5 Device and pinning

Existing trainer is device-implicit. Mini-batch path adds:

```python
generator = torch.Generator(device=output_input.device).manual_seed(self.random_seed)
```

so the shuffle stays on the data's device. **No** `pin_memory` or
`non_blocking=True` in v1; defer to a perf-tier follow-up.

### 6.6 Backward-compatibility / fall-through

The full-batch `for epoch in range(epochs)` path is preserved
**verbatim** as the `else:` branch. Setting `use_mini_batch=False` (or
running with `mini_batch_size >= dataset_size` per §5.4) yields
bit-exact pre-restoration behaviour. This is the regression guard for
existing users and is enforced by §7.1.

## 7. Test strategy

### 7.1 Bit-exact regression guard (CRITICAL)

```python
def test_full_batch_path_bit_exact_against_baseline(snapshot_baseline):
    """
    With use_mini_batch=False, output trainer must reproduce the
    pre-restoration loss trajectory and final weights to within
    torch.testing.assert_close default tolerance.
    """
    # baseline checked in as a small fixture or recomputed from a
    # frozen pre-restoration commit
```

This is the strongest argument that mini-batch is opt-in safe and is
the gate on flipping the default to `True`.

### 7.2 Iteration count

```python
def test_mini_batch_iterates_correct_step_count():
    # dataset size 1024, batch 256, epochs 3 → 12 steps total
    cfg = CascadeCorrelationConfig(use_mini_batch=True, mini_batch_size=256)
    ...
    assert step_counter == 12
```

### 7.3 Convergence equivalence on canonical task

```python
@pytest.mark.spiral
def test_two_spiral_converges_with_and_without_mini_batch():
    # Train with use_mini_batch=False vs True, mini_batch_size=64
    # Both must reach the same convergence threshold within
    # comparable epoch budgets (allow 2x slop on epoch count).
```

Uses the existing `src/spiral_problem/` fixtures.

### 7.4 Validation behaviour

```python
def test_mini_batch_size_zero_or_negative_raises():
    with pytest.raises(ValueError):
        CascadeCorrelationConfig(mini_batch_size=0)

def test_mini_batch_size_exceeds_dataset_falls_back_to_full_batch(caplog):
    # mini_batch_size=1024, dataset=194 → fall back, log INFO line
    ...
    assert "falling back to full-batch" in caplog.text
```

### 7.5 Reproducibility

```python
def test_mini_batch_deterministic_with_seed():
    # Same random_seed → same final weights across two runs
```

## 8. Performance / convergence implications

### 8.1 Memory

Mini-batch slashes peak memory in the output trainer by `dataset_size /
mini_batch_size`. For the spiral fallback case this is 1× (no change);
for a 100k-sample workload at `mini_batch_size=256` it is ~390×.

### 8.2 Speed per epoch

Per-epoch wall-clock typically slows 1.5–3× due to Python / kernel
launch overhead (more `optimizer.step()` calls). On GPU the slowdown
is partially recovered by better SM occupancy. On CPU it is a clean
slowdown; users running the spiral smoke test on CPU may see longer
fit() times. **This is the primary convergence-risk knob** to
communicate in the release notes.

### 8.3 Convergence — epoch budget

Mini-batch SGD usually converges in **fewer wall-clock seconds** but
**more or comparable epochs** than full-batch GD. The existing
`_CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS = 10000` budget is
overprovisioned for the spiral task and unlikely to bind. Open
question Q5 (§9): expose a recommended `output_epochs` reduction
when `use_mini_batch=True`?

### 8.4 Metrics intersection (R5.4-pre throttle)

`src/cascade_correlation/cascade_correlation.py:1655` and
`src/api/lifecycle/manager.py:681..711`:

The R5.4-pre histogram emits one observation per output-phase epoch
(modulo a 25-epoch throttle).  Under v1 (per-epoch callback,
recommended above), the histogram still emits per epoch — but each
epoch is now slower and contains N mini-batch steps. The histogram
**resolution** stays the same, but each sample now represents a
larger compute unit. The SLO 3.4 target ("p95 < 5 s") needs
re-validation against the new step definition.

Under v2 (per-step callback, Q4), the throttle (`% 25`) becomes
inadequate — for 100 mini-batches per epoch, a throttle of 25 means
one observation every quarter-epoch, which is acceptable, but the
throttle SHOULD be re-derived from "samples per minute target" rather
than "epochs per emission". This is the seed of a follow-up
**METRICS-MON R5.5** sub-track.

## 9. Sequencing & sub-track proposal

### 9.1 Proposed sub-track

**`TRAIN-ARCH-01: Cascor mini-batch training restoration`**

(Naming follows the pattern of `METRICS-MON R5.4-pre` /
`PARALLEL_CANDIDATE_TRAINING_FIX_PLAN`. `TRAIN-ARCH-01` distinguishes
training-architecture work from the metrics work but admits a sibling
`TRAIN-ARCH-02` later, e.g. for the candidate-phase research.)

### 9.2 Estimated PR count

**3–4 PRs**, in dependency order:

1. **PR-1** — constants (`constants_model.py`, `constants.py` ladder
   updates, comment headers). No behaviour change.
2. **PR-2** — `CascadeCorrelationConfig` dataclass fields + validation
   + serialization parity. No trainer change.
3. **PR-3** — output-trainer mini-batch branch + tests §7.1, §7.2,
   §7.4, §7.5. **This is where default `use_mini_batch=True` flips.**
   Gated on §7.1 bit-exact regression guard.
4. **PR-4** *(optional, deferrable)* — convergence equivalence test
   §7.3 (slow, marker=`spiral`) + release-notes fragment.

### 9.3 Dependencies

- **Does NOT block on R5.4** (already merged).
- **Should sequence after R5.1 SLO catalog ratification**
  (juniper-deploy#48/#49) so the convergence-equivalence threshold in
  §7.3 can be expressed in SLO terms.
- **Should sequence before R5.5 metrics throttle redesign** —
  R5.5's design depends on whether v1 ships per-epoch (a) or per-step
  (b) callback semantics (§6.4).

### 9.4 Risks (explicit)

| Risk | Severity | Mitigation |
|------|----------|------------|
| Default flip changes user-visible convergence behaviour | **High** | §7.1 bit-exact guard for `use_mini_batch=False`; consider Q1 (default=False for one minor) |
| Spiral smoke-test perf regression on CPU | Medium | §5.4 fallback when `mini_batch_size >= dataset_size`; spiral N=194 < default 256 → fallback fires |
| Reproducibility regression for users relying on global seed | Medium | §6.3 dedicated `torch.Generator` keyed to `self.random_seed`; §7.5 test |
| Candidate phase mistakenly mini-batched | **High** (algorithmic correctness) | §4.5 + §5.5 — separate `use_mini_batch_candidate`, default `False`, gated against §7 candidate-phase test |
| Metrics histogram resolution loss under v1 (per-epoch) | Low | §8.4 — accept v1, queue R5.5 follow-up |

### 9.5 Open questions for entry-plan resolution

In the R3/R4/R5 entry-plan Q-style:

- **Q1.** Should `_PROJECT_MODEL_USE_MINI_BATCH` default to `False`
  for one minor version (opt-in), then flip to `True` after a
  validation cycle? Or flip to `True` immediately per the user's
  stated requirement? **The user's requirement is explicit (default
  `True`), but absence-not-regression framing argues for the cautious
  path.**
- **Q2.** Ratify `_PROJECT_MODEL_MINI_BATCH_SIZE = 256` vs. 128 vs. 64.
  Decision should be informed by canonical-workload benchmarking on
  spiral and one larger synthetic dataset.
- **Q3.** Add a separate `mini_batch_size_candidate` field now (for
  forward-compat with future research) or defer until needed?
  Recommendation: defer.
- **Q4.** Per-epoch (v1, recommended) or per-step (v2) metrics
  callback semantics? **R5.5 sequencing depends on this answer.**
- **Q5.** Expose a recommended-reduction `output_epochs` value when
  `use_mini_batch=True` (e.g. via a config-level "auto-scale epochs
  by mini-batch ratio" toggle)? Or document in the release notes only?

## 10. References

### Cross-repo (annotated as such)

- juniper-ml#189 — *cross-repo* — mini-batch design doc that surfaced
  the gap (parent program tracking)
- juniper-deploy#48 — *cross-repo* — METRICS-MON R5.1 SLO catalog
- juniper-deploy#49 — *cross-repo* — R5.1 SLO catalog fixup PR

### Within juniper-cascor

- juniper-cascor#188 — METRICS-MON R5.4-pre (training counters,
  train-step histogram, worker→Prometheus bridge), merged
- `src/cascade_correlation/cascade_correlation.py:1556..1678` —
  `train_output_layer`
- `src/cascade_correlation/cascade_correlation.py:1638` — output-trainer epoch loop
- `src/cascade_correlation/cascade_correlation.py:2589..` —
  `_create_optimizer` factory
- `src/candidate_unit/candidate_unit.py:509..680` —
  `train_detailed`
- `src/candidate_unit/candidate_unit.py:564` — candidate-trainer epoch loop
- `src/candidate_unit/candidate_unit.py:878..950` —
  `_calculate_correlation` (full-batch Pearson, §4.5 constraint)
- `src/candidate_unit/candidate_unit.py:958..1050` —
  `_update_weights_and_bias` (manual gradient over full-batch
  correlation)
- `src/api/lifecycle/manager.py:681..711` —
  `_output_training_callback` and the per-epoch train-step histogram
  emission (R5.4-pre)
- `src/cascor_constants/constants_model/constants_model.py:206, 300` —
  `_PROJECT_MODEL_EPOCHS_MAX`, `_PROJECT_MODEL_OUTPUT_EPOCHS`
- `src/cascor_constants/constants_candidates/constants_candidates.py:130` —
  `_PROJECT_MODEL_CANDIDATE_EPOCHS = 400`
- `src/cascade_correlation/cascade_correlation_config/cascade_correlation_config.py:131..298` —
  `CascadeCorrelationConfig` dataclass
- `notes/observability/HISTOGRAM_BUCKETS_RATIONALE_2026-05-02.md` §6 —
  train-step histogram boundary discussion
- `notes/development/PARALLEL_CANDIDATE_TRAINING_FIX_PLAN.md` —
  candidate-phase parallelism architecture (relevant to §4.4)

### Historical commits reviewed (§2)

- `2076d21` — initial commit of standalone JuniperCascor (2025-12-08)
  — already full-batch
- `a854e65` — perf: fused tensor ops (OPT-2) and pre-allocated forward
  buffer (OPT-1) — full-batch unchanged
- `dc95dd6`, `3f55e17`, `ce2664d`, `f75b4b5`, `b6ccb8f`, `4139e2a`,
  `3d1a19d`, `5a7c710` — touched `cascade_correlation.py` /
  `candidate_unit.py` / tests; none introduced or removed mini-batch
  iteration
- `juniper-legacy/JuniperCascor/...` — full-batch
- `juniper-legacy/JuniperLegacy/src/prototypes/{juniper_cascor,cascor_spiral}/...` —
  full-batch
- `juniper-legacy/JuniperCascor/.../backups/cascade_correlation-ORIG.py` —
  full-batch

---

**End of design.** Sub-track entry-plan and PR-1 may proceed pending
senior-engineer ratification of Q1–Q5 in §9.5.
