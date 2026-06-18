# Golden / Snapshot Regression Fixtures (OUT-12 / WS-6 gate)

Checked-in reference artifacts for the golden regression suite. These pin the
cascade-correlation network's observable behavior **before** the WS-6 refactor
(repointing cascor onto `juniper-service-core` / `juniper-model-core`), so any
behavior change introduced by that refactor is caught.

Build plan: `juniper-ml/notes/JUNIPER_CASCOR_GOLDEN_REGRESSION_SUITE_BUILD_PLAN_2026-06-17.md`
(roadmap **OUT-12**). Comparison strategy resolved by evidence (OQ-13):
**tolerance for floats, exact for discrete/structural signals.**

## Artifacts

| File | Produced by | Compared how |
|------|-------------|--------------|
| `two_spiral_seed42.npz` | frozen `SpiralDataGenerator.generate_2_spiral(n_per_spiral=30, noise=0.05, seed=42)` | input only (frozen so a generator change can't silently move the goldens) |
| `golden_trajectory_seed42.json` | `test_golden_trajectory.py` | losses / accuracies / correlations → tolerance; growth sequence, `weight_shape`, `unit_index`, `completion_reason`, counts → exact |
| `golden_predict_seed42.json` | `test_golden_serialization_roundtrip.py` | trained `predict` output → tolerance |
| `api_snapshots/*.json` | `test_golden_api_snapshots.py` | scrubbed JSON bodies; floats → tolerance, structure/status → exact |
| `VERSIONS.json` | capture run | provenance (python / torch / numpy / config / tolerances) |

## Tolerances

| Signal | rtol | atol |
|--------|------|------|
| Trajectory floats, correlations, API weights, predict | `1e-3` | `1e-4` |
| Predict-after-load round-trip (`torch.allclose`) | — | `1e-6` |

Discrete signals (growth-count sequence, `unit_index` ordering, `weight_shape`,
`_completion_reason`, hidden-unit counts, epoch counts, route status codes,
JSON key sets) are compared **exactly** by the recursive comparator in
`src/tests/golden_support.py` (int / str / bool / None leaves are exact; only
float leaves flex).

## Calibration evidence (2026-06-17)

Captured and verified on `JuniperCascor1` (Python 3.13.13, torch 2.11.0+cu130,
numpy 2.4.4). Under the pinned config (`CASCOR_NUM_PROCESSES=1`, single-thread
BLAS, `torch.set_num_threads(1)`, `use_deterministic_algorithms(True)`,
`seed=42`) the trajectory was **bit-identical across repeated runs**, and the
full lane passed repeatedly in assert mode. The `1e-3 / 1e-4` tolerance
therefore carries generous headroom for cross-machine / cross-build float noise.

The CI lane (`.github/workflows/golden-regression.yml`) pins **Python 3.13 +
torch 2.11.0 (CPU)** to match this calibration; the CPU kernels of the `+cpu`
and `+cu` builds of the same torch version are numerically equivalent for the
CPU tensors used here.

## Regenerating (after an intentional, reviewed behavior change ONLY)

```bash
# From the juniper-cascor repo root, on the GIL env (JuniperCascor1):
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CASCOR_NUM_PROCESSES=1 \
GOLDEN_CAPTURE=1 \
python -m pytest -m golden --golden --slow --integration src/tests/integration
```

Then review the diff (the whole point of the gate is that goldens change only
when behavior is intended to change) and re-run **without** `GOLDEN_CAPTURE=1`
to confirm they lock in. If a refactor cannot keep these green without changing
observable behavior, the WS-6 kill-criterion fires (see the build plan §6).
