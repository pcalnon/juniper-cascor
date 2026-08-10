"""
F-P4-1 reproduction: in-process replay of the SERVICE path's spiral training termination.

Project: juniper-cascor
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-08-09
Status: ad-hoc — investigation
Retire when: F-P4-1 is fixed and covered by a regression test
Related: juniper-ml notes/JUNIPER_2026-08-09_JUNIPER-ECOSYSTEM_CLI-EXPERIMENTATION-P4-STUDIES-EVIDENCE.md §4;
         E-A suite e-a-cascor-budget-sweep-20260809T085929Z

Replays what api/routes/training.py + api/lifecycle/manager.py do for a spiral
/v1/training/start:
  1. dataset from the route's _generate_spiral_data with the DRIVER's params dict
     (whose 'n_points_per_spiral' key misses the route's 'n_per_spiral' lookup
     -> 100/spiral defaults -> 200 points, one-hot 2-column targets);
  2. network from CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=2);
  3. the E-A TrainingParams applied via setattr with the _apply_params_unlocked
     hasattr filter (skips reported);
  4. fit(max_epochs=2000, max_iterations=12, early_stopping=True).

Arm B trains the same data on a pure-defaults network/fit for contrast.
"""

import logging
import os
import sys
import time

REPO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, os.path.abspath(REPO_SRC))

logging.basicConfig(level=logging.WARNING)
for name in list(logging.Logger.manager.loggerDict):
    logging.getLogger(name).setLevel(logging.WARNING)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(20260729)
np.random.seed(20260729)

# --------------------------------------------------------------------------- #
# 1. The route's _generate_spiral_data, byte-faithful, fed the DRIVER's params
# --------------------------------------------------------------------------- #

DRIVER_DATASET_PARAMS = {
    "n_spirals": 2,
    "n_points_per_spiral": 500,  # route reads 'n_per_spiral' -> MISSES -> default 100
    "n_rotations": 3.0,
    "noise": 0.05,
    "algorithm": "modern",
    "train_ratio": 0.8,
    "test_ratio": 0.2,
    "seed": 20260729,
}

E_A_TRAINING_PARAMS = {
    "max_epochs": 2000,
    "max_iterations": 12,
    "early_stopping": True,
    "learning_rate": 0.05,
    "candidate_learning_rate": 0.05,
    "correlation_threshold": 0.2,
    "candidate_pool_size": 8,
    "max_hidden_units": 32,
    "patience": 200,
    "convergence_threshold": 1.0e-5,
    "candidate_patience": 100,
    "candidate_epochs": 500,
}

FIT_KWARGS = {"max_epochs", "epochs", "max_iterations", "early_stopping"}


def generate_route_spiral(params: dict):
    """Copy of api/routes/training.py::_generate_spiral_data."""
    n_per_spiral = params.get("n_per_spiral", 100)
    n_spirals = params.get("n_spirals", 2)
    x_data = []
    y_data = []
    for i in range(n_spirals):
        t = np.linspace(0, 4 * np.pi, n_per_spiral)
        angle_offset = 2 * np.pi * i / n_spirals
        x_spiral = t * np.cos(t + angle_offset) / (4 * np.pi)
        y_spiral = t * np.sin(t + angle_offset) / (4 * np.pi)
        x_data.append(np.stack([x_spiral, y_spiral], axis=1))
        y_one_hot = np.zeros((n_per_spiral, n_spirals))
        y_one_hot[:, i] = 1
        y_data.append(y_one_hot)
    x = torch.tensor(np.concatenate(x_data, axis=0), dtype=torch.float32)
    y = torch.tensor(np.concatenate(y_data, axis=0), dtype=torch.float32)
    return x, y


def build_network(apply_params: bool):
    from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
    from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

    config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=2)
    net = CascadeCorrelationNetwork(config=config)

    applied, skipped = [], []
    if apply_params:
        simple = {k: v for k, v in E_A_TRAINING_PARAMS.items() if k not in FIT_KWARGS}
        for k, v in simple.items():
            if hasattr(net, k):
                setattr(net, k, v)
                applied.append(k)
            else:
                skipped.append(k)
    return net, applied, skipped


def report_effective(net):
    keys = [
        "learning_rate",
        "candidate_learning_rate",
        "correlation_threshold",
        "candidate_pool_size",
        "max_hidden_units",
        "max_iterations",
        "patience",
        "convergence_threshold",
        "candidate_convergence_threshold",
        "candidate_patience",
        "candidate_epochs",
        "output_epochs",
        "num_processes",
    ]
    out = {}
    for k in keys:
        out[k] = getattr(net, k, "<absent>")
    return out


def run_arm(label: str, apply_params: bool, fit_kwargs: dict, x_scale: float = 1.0):
    print(f"\n===== ARM {label} =====")
    x, y = generate_route_spiral(DRIVER_DATASET_PARAMS)
    if x_scale != 1.0:
        x = x * x_scale
    print(f"data: x={tuple(x.shape)} y={tuple(y.shape)} x_scale={x_scale} |x|max={float(x.abs().max()):.3f}")

    net, applied, skipped = build_network(apply_params)
    print(f"params applied: {applied}")
    print(f"params skipped (no-such-attribute): {skipped}")
    print(f"effective: {report_effective(net)}")
    try:
        print(f"optimal process count: {net._calculate_optimal_process_count()}")
    except Exception as exc:  # noqa: BLE001
        print(f"optimal process count: <error {exc}>")

    correlations = []
    orig_emit = getattr(net, "_emit_candidate_correlation", None)
    if orig_emit is not None:
        def record_emit(value, _orig=orig_emit):
            correlations.append(float(value))
            return _orig(value)

        net._emit_candidate_correlation = record_emit

    t0 = time.time()
    try:
        net.fit(x, y, **fit_kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"fit RAISED: {type(exc).__name__}: {exc}")
    wall = time.time() - t0

    acc = None
    try:
        acc = net.calculate_accuracy(x, y)
    except Exception as exc:  # noqa: BLE001
        acc = f"<error {exc}>"
    print(f"wall: {wall:.1f}s")
    print(f"hidden units: {len(net.hidden_units)}")
    print(f"completion reason: {getattr(net, '_completion_reason', '<absent>')}")
    print(f"train accuracy: {acc}")
    print(f"best-candidate correlations per grow iteration: {correlations}")
    hist = getattr(net, "history", {})
    print(f"history lens: " + ", ".join(f"{k}={len(v)}" for k, v in hist.items() if isinstance(v, list)))


if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv) > 1 else "both"
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    if arm in ("a", "both"):
        run_arm(
            "A (service replay: E-A params)",
            apply_params=True,
            fit_kwargs={"max_epochs": 2000, "max_iterations": 12, "early_stopping": True},
            x_scale=scale,
        )
    if arm in ("b", "both"):
        run_arm("B (pure defaults)", apply_params=False, fit_kwargs={}, x_scale=scale)
