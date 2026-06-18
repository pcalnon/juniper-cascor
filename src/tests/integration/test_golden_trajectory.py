"""Golden trajectory regression (OUT-12 / WS-6 pre-refactor baseline).

Trains the cascade-correlation network on the frozen two-spiral dataset under a
fully-pinned, sequential, single-thread configuration and compares the resulting
``network.history`` against a checked-in golden:

  * per-epoch ``train_loss`` / ``value_loss`` / ``train_accuracy`` /
    ``value_accuracy`` -> tolerance (rtol=1e-3, atol=1e-4),
  * per-unit ``correlation`` -> tolerance,
  * the hidden-unit growth sequence (``unit_index`` ordering + ``weight_shape``),
    the growth count, and ``_completion_reason`` -> exact.

The exact-vs-tolerance split is enforced automatically by the recursive
comparator in ``golden_support`` (float leaves flex; ints / strings / shapes are
pinned). See the build plan in juniper-ml,
notes/JUNIPER_CASCOR_GOLDEN_REGRESSION_SUITE_BUILD_PLAN_2026-06-17.md.

Run (serial, GIL env, deterministic):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    CASCOR_NUM_PROCESSES=1 \
    python -m pytest -m golden --golden --slow --integration \
        src/tests/integration/test_golden_trajectory.py

Re-capture goldens (after an intentional, reviewed behavior change only):
    GOLDEN_CAPTURE=1 <same command>
"""

import golden_support as gs
import pytest

_KNOWN_COMPLETION_REASONS = {
    "residual_collapsed",
    "no_candidate",
    "below_threshold",
    "early_stopped",
    "max_iterations",
}


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.golden
def test_golden_trajectory():
    """Fixed-seed two-spiral training trajectory matches the checked-in golden."""
    if gs.is_capture():
        gs.freeze_two_spiral()
        gs.write_versions()

    x, y = gs.load_two_spiral()
    net = gs.build_and_train_golden_network(x, y)
    history = net.history

    # Build the comparable trajectory payload. The final element of
    # ``hidden_units_added`` is the sentinel {corr:0.0, weight_shape:(), idx:-1}
    # appended by fit(); growth_count excludes it.
    trajectory = {
        "train_loss": history["train_loss"],
        "value_loss": history["value_loss"],
        "train_accuracy": history["train_accuracy"],
        "value_accuracy": history["value_accuracy"],
        "hidden_units_added": history["hidden_units_added"],
        "dataset_swaps": history["dataset_swaps"],
        "completion_reason": net._completion_reason,
        "growth_count": len(history["hidden_units_added"]) - 1,
        "num_hidden_units": len(net.hidden_units),
    }

    # Mode-independent sanity (guards capture runs from freezing nonsense).
    assert net._completion_reason in _KNOWN_COMPLETION_REASONS, net._completion_reason
    assert trajectory["growth_count"] >= 1
    assert trajectory["num_hidden_units"] == trajectory["growth_count"]
    assert len(history["train_loss"]) >= 1

    gs.assert_or_capture("golden_trajectory_seed42.json", trajectory, rtol=gs.RTOL, atol=gs.ATOL)
