"""Shared support for the golden / snapshot regression suite (OUT-12 / WS-6 gate).

This module is the single home for the determinism hardening, frozen-dataset
helpers, the recursive volatile-field scrubber, and the recursive
tolerance-aware comparator used by the three golden test modules:

  * integration/test_golden_trajectory.py
  * integration/test_golden_serialization_roundtrip.py
  * integration/api/test_golden_api_snapshots.py

Design (see notes/JUNIPER_CASCOR_GOLDEN_REGRESSION_SUITE_BUILD_PLAN_2026-06-17.md
in juniper-ml, OQ-13 resolution):

  * Comparison is **tolerance-based for floats, exact for discrete signals**.
    A single recursive comparator achieves this: float leaves are compared with
    ``rtol``/``atol``; ints, bools, strings and ``None`` are compared exactly.
    So per-epoch losses / correlations / weights flex within tolerance, while
    growth-count sequences, ``unit_index`` ordering, ``weight_shape``,
    ``_completion_reason`` and route status codes are pinned exactly.
  * Capture-first: set ``GOLDEN_CAPTURE=1`` to (re)write the goldens instead of
    asserting. Eyeball the written artifacts, then run without the flag to lock
    them in.
  * Determinism is hardened to the sequential candidate path
    (``CASCOR_NUM_PROCESSES=1``), single-thread BLAS/torch, and ``seed=42``.

It is intentionally a plain importable module (``import golden_support``), since
``src/tests`` is on ``sys.path`` (see conftest.py) and the test directories are
namespace packages.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

# This file lives at src/tests/golden_support.py, so the golden artifacts live
# under src/tests/fixtures/golden/.
GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
TWO_SPIRAL_NPZ = GOLDEN_DIR / "two_spiral_seed42.npz"

# The fixed network/train config used by every golden capture. ``epochs_max``
# and ``max_iterations`` are the real CascadeCorrelationConfig kwargs (the
# attribute is ``epochs_max``, not ``max_epochs``); ``early_stopping`` is a
# ``fit()`` argument, NOT a config kwarg, so it is passed at train time.
GOLDEN_NET_CONFIG = {
    "random_seed": 42,
    "max_hidden_units": 3,
    "candidate_pool_size": 2,
    "candidate_epochs": 3,
    "output_epochs": 3,
    "epochs_max": 5,
    "max_iterations": 3,
}

# Default tolerances (calibrated empirically — bit-identical across runs on the
# JuniperCascor1 env; these leave generous cross-machine / docker headroom).
RTOL = 1e-3
ATOL = 1e-4
# Predict round-trip (save/load) must match much more tightly: same machine,
# same weights, only HDF5 (de)serialization in between.
PREDICT_ROUNDTRIP_ATOL = 1e-6


# ---------------------------------------------------------------------------
# Volatile fields (scrubbed before any snapshot comparison)
# ---------------------------------------------------------------------------

# Exact key names stripped anywhere they appear in a response (recursive).
# ``timestamp`` covers meta.timestamp, state_machine.timestamp, and per-element
# metric timestamps in one shot. See build-plan §4.
VOLATILE_KEYS = frozenset(
    {
        "timestamp",
        "git_sha",
        "build_date",
        "version",
        "duration_ms",
        "uptime_seconds",
        "uuid",
        "server_instance_id",
        "snapshot_seq",
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
        "elapsed_seconds",
        "wall_time",
    }
)

# Any key ending in one of these suffixes is treated as a volatile counter.
VOLATILE_SUFFIXES = ("_total",)


def is_capture() -> bool:
    """True when running in capture mode (``GOLDEN_CAPTURE=1``)."""
    return os.environ.get("GOLDEN_CAPTURE") == "1"


# ---------------------------------------------------------------------------
# Determinism hardening
# ---------------------------------------------------------------------------


def harden_determinism() -> None:
    """Pin the process to the deterministic, sequential candidate path.

    NOTE: the ``*_NUM_THREADS`` env vars are read by BLAS at load time, so when
    torch/numpy are already imported (they are, via conftest) setting them here
    is best-effort only — the serial golden CI lane also exports them in the job
    environment *before* pytest starts. ``torch.set_num_threads(1)`` and
    ``CASCOR_NUM_PROCESSES=1`` ARE honored at runtime and are the load-bearing
    knobs here.
    """
    os.environ["CASCOR_NUM_PROCESSES"] = "1"
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")

    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    try:  # belt-and-suspenders; warn_only so unsupported ops don't hard-fail
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001 - older torch / unsupported backend
        pass


# ---------------------------------------------------------------------------
# Frozen dataset
# ---------------------------------------------------------------------------


def freeze_two_spiral() -> Path:
    """Generate and freeze the two-spiral dataset to ``two_spiral_seed42.npz``.

    Frozen deliberately so a future change to ``SpiralDataGenerator`` cannot
    silently move the goldens. Called in capture mode; idempotent.
    """
    import numpy as np
    from unit.test_data.generators import SpiralDataGenerator  # test-tree generator

    x, y, _info = SpiralDataGenerator.generate_2_spiral(n_per_spiral=30, noise=0.05, seed=42)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(TWO_SPIRAL_NPZ, x=x.numpy(), y=y.numpy())
    return TWO_SPIRAL_NPZ


def load_two_spiral():
    """Load the frozen two-spiral tensors ``(x, y)`` (float32)."""
    import numpy as np
    import torch

    if not TWO_SPIRAL_NPZ.exists():
        raise FileNotFoundError(f"{TWO_SPIRAL_NPZ} is missing. Freeze it first: " f"run the golden suite with GOLDEN_CAPTURE=1, or call " f"golden_support.freeze_two_spiral().")
    data = np.load(TWO_SPIRAL_NPZ)
    x = torch.from_numpy(data["x"]).float()
    y = torch.from_numpy(data["y"]).float()
    return x, y


def two_spiral_inline():
    """Return the frozen two-spiral as plain nested lists for API inline_data."""
    x, y = load_two_spiral()
    return x.tolist(), y.tolist()


def build_and_train_golden_network(x, y):
    """Construct + train the canonical golden network on ``(x, y)``.

    Mirrors the validated probe sequence: seed, construct, reseed, fit. Returns
    the trained ``CascadeCorrelationNetwork``.
    """
    import numpy as np
    import torch

    from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

    harden_determinism()
    net = CascadeCorrelationNetwork(**GOLDEN_NET_CONFIG)
    # Reseed post-construction so fit() starts from a known RNG state
    # regardless of any RNG draws during construction.
    torch.manual_seed(42)
    np.random.seed(42)
    net.fit(x, y, x_val=x, y_val=y, early_stopping=False)
    return net


# ---------------------------------------------------------------------------
# JSON normalization + scrubbing
# ---------------------------------------------------------------------------


def _json_default(obj):
    import numpy as np
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def jsonify(obj):
    """Round-trip ``obj`` through JSON so tuples->lists, tensors->lists, etc.

    Makes a live Python object structurally comparable to a golden loaded from
    disk (apples-to-apples).
    """
    return json.loads(json.dumps(obj, default=_json_default))


def scrub(obj, *, extra_keys=(), drop_array_values_under=()):
    """Recursively strip volatile keys from a (already json-ified) structure.

    Args:
        obj: nested dict/list/scalar structure.
        extra_keys: additional exact key names to strip.
        drop_array_values_under: key names whose *value*, when it is a list of
            numbers (a weight/topology array), is replaced by a structural
            descriptor ``{"__array_len__": N}`` instead of the raw floats. Use
            only if a weight array proves cross-machine unstable beyond
            tolerance; by default weights are kept and tolerance-compared.
    """
    strip_keys = set(VOLATILE_KEYS) | set(extra_keys)
    drop_under = set(drop_array_values_under)

    def _is_number_list(value):
        return isinstance(value, list) and bool(value) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)

    def _strip(node):
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                if key in strip_keys or (isinstance(key, str) and key.endswith(VOLATILE_SUFFIXES)):
                    continue
                if key in drop_under and _is_number_list(value):
                    out[key] = {"__array_len__": len(value)}
                    continue
                out[key] = _strip(value)
            return out
        if isinstance(node, list):
            return [_strip(v) for v in node]
        return node

    return _strip(obj)


# ---------------------------------------------------------------------------
# Recursive tolerance-aware comparison
# ---------------------------------------------------------------------------


def _isclose(a: float, b: float, rtol: float, atol: float) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def compare(golden, actual, *, rtol=RTOL, atol=ATOL, path="$"):
    """Recursively compare ``golden`` vs ``actual``; return list of mismatches.

    Float leaves use ``rtol``/``atol``; bools, ints, strings and ``None`` are
    exact; containers must match structurally (lengths / keys). An empty list
    means the structures match.
    """
    out = []

    # bool is a subclass of int — handle first, exact.
    if isinstance(golden, bool) or isinstance(actual, bool):
        if golden != actual or type(golden) is not type(actual):
            out.append(f"{path}: bool {golden!r} != {actual!r}")
        return out

    if isinstance(golden, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(golden, int) and isinstance(actual, int):
            if golden != actual:
                out.append(f"{path}: int {golden} != {actual}")
        else:
            gf, af = float(golden), float(actual)
            if not _isclose(gf, af, rtol, atol):
                out.append(f"{path}: float {gf!r} != {af!r} (rtol={rtol}, atol={atol}, |Δ|={abs(gf - af):.3e})")
        return out

    if golden is None or actual is None:
        if golden is not actual:
            out.append(f"{path}: {golden!r} != {actual!r}")
        return out

    if isinstance(golden, str) and isinstance(actual, str):
        if golden != actual:
            out.append(f"{path}: str {golden!r} != {actual!r}")
        return out

    if isinstance(golden, list) and isinstance(actual, list):
        if len(golden) != len(actual):
            out.append(f"{path}: list length {len(golden)} != {len(actual)}")
            return out
        for i, (g, a) in enumerate(zip(golden, actual)):
            out += compare(g, a, rtol=rtol, atol=atol, path=f"{path}[{i}]")
        return out

    if isinstance(golden, dict) and isinstance(actual, dict):
        gkeys, akeys = set(golden), set(actual)
        if gkeys != akeys:
            missing = sorted(gkeys - akeys)
            extra = sorted(akeys - gkeys)
            if missing:
                out.append(f"{path}: missing keys {missing}")
            if extra:
                out.append(f"{path}: unexpected keys {extra}")
        for key in sorted(gkeys & akeys):
            out += compare(golden[key], actual[key], rtol=rtol, atol=atol, path=f"{path}.{key}")
        return out

    out.append(f"{path}: type mismatch {type(golden).__name__} != {type(actual).__name__}")
    return out


# ---------------------------------------------------------------------------
# Capture / assert entry point
# ---------------------------------------------------------------------------


def assert_or_capture(name, actual, *, rtol=RTOL, atol=ATOL, extra_scrub_keys=(), drop_array_values_under=()):
    """Capture (write) or assert ``actual`` against golden ``name``.

    In capture mode (``GOLDEN_CAPTURE=1``) the scrubbed, json-ified ``actual``
    is written to ``GOLDEN_DIR/name`` (pretty, key-sorted) and no assertion is
    made. Otherwise the golden is loaded and compared with tolerance.
    """
    actual_json = scrub(
        jsonify(actual),
        extra_keys=extra_scrub_keys,
        drop_array_values_under=drop_array_values_under,
    )
    target = GOLDEN_DIR / name

    if is_capture():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(actual_json, indent=2, sort_keys=True) + "\n")
        return

    if not target.exists():
        raise AssertionError(f"Golden file missing: {target}\n" f"Run the golden suite with GOLDEN_CAPTURE=1 to create it.")
    golden_json = json.loads(target.read_text())
    mismatches = compare(golden_json, actual_json, rtol=rtol, atol=atol)
    assert not mismatches, "Golden mismatch for {} ({} difference(s)):\n  {}".format(name, len(mismatches), "\n  ".join(mismatches[:40]))


def write_versions(extra=None):
    """In capture mode, record torch/numpy/python versions next to the goldens."""
    if not is_capture():
        return
    import sys

    import numpy as np
    import torch

    payload = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "net_config": GOLDEN_NET_CONFIG,
        "tolerances": {"rtol": RTOL, "atol": ATOL, "predict_roundtrip_atol": PREDICT_ROUNDTRIP_ATOL},
    }
    if extra:
        payload.update(extra)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    (GOLDEN_DIR / "VERSIONS.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
