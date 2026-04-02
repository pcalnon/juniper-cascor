"""
Project:       Juniper
Sub-Project:   JuniperCascor
File Name:     test_endtoend_profiling.py
File Path:     src/tests/performance/

Author:        Paul Calnon
Version:       0.1.0

Date Created:  2026-04-01
Last Modified: 2026-04-01

License:       MIT License
Copyright:     Copyright (c) 2024-2026 Paul Calnon

Description:
    Phase 4: End-to-end profiling tests for the Cascade Correlation training loop.

    Step 4.1 — Full training run profiling with cProfile/ProfileContext
    Step 4.2 — Training phase time distribution via monkey-patched timing
    Step 4.3 — Memory growth profiling across growth epochs
    Step 4.4 — Convergence vs performance tradeoff parametrization

    Run with: pytest tests/performance/test_endtoend_profiling.py --run-performance -v -s
"""

import resource
import time
from typing import Dict, List

import pytest

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from profiling.deterministic import ProfileContext

from .conftest import BenchmarkTimer, _make_benchmark_config

# ===================================================================
# HELPERS
# ===================================================================


def _make_small_training_config(**overrides) -> CascadeCorrelationConfig:
    """Create a small, fast config for end-to-end profiling tests."""
    defaults = {
        "candidate_pool_size": 4,
        "candidate_epochs": 20,
        "max_hidden_units": 5,
        "output_epochs": 25,
        "patience": 3,
        "correlation_threshold": 0.001,
    }
    defaults.update(overrides)
    return _make_benchmark_config(**defaults)


def _get_rss_kb() -> int:
    """Return current process RSS in KB (Linux: ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _get_rss_from_proc_mb() -> float:
    """Return current process RSS in MB via /proc/self/status (more accurate snapshot)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # KB -> MB
    except (OSError, ValueError):
        pass
    # Fallback to resource module
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ===================================================================
# STEP 4.1: Full Training Run Profiling
# ===================================================================


@pytest.mark.performance
class TestFullTrainingRunProfiling:
    """Profile a complete training run using cProfile/ProfileContext.

    Uses a small configuration (spiral data, pool_size=4, candidate_epochs=20,
    max_hidden=5) to keep runtime short while capturing meaningful function-level
    profiling data.
    """

    @pytest.mark.timeout(180)
    def test_full_training_cprofile(self, small_spiral_data):
        """Profile a complete fit() call with cProfile and capture top functions."""
        x_train, y_train = small_spiral_data

        config = _make_small_training_config(max_hidden_units=3)
        net = CascadeCorrelationNetwork(config=config)

        with ProfileContext("full_training_run") as profile:
            net.fit(x_train, y_train, max_epochs=3)

        # Verify profiling captured data
        profile_data = profile.to_dict(top_n=20)
        assert profile_data.get("total_calls", 0) > 0, "Profiler should capture function calls"
        assert profile_data.get("elapsed_seconds", 0) > 0, "Elapsed time should be positive"
        assert len(profile_data.get("top_functions", [])) > 0, "Should capture top functions"

        # Print structured results for data collection
        print("\n" + "=" * 70)
        print("STEP 4.1: Full Training Run cProfile Results")
        print("=" * 70)
        print(f"  Total function calls:  {profile_data['total_calls']}")
        print(f"  Wall-clock time:       {profile_data['elapsed_seconds']:.3f}s")
        print(f"  Total internal time:   {profile_data['total_tt']:.3f}s")
        print(f"  Hidden units added:    {len(net.hidden_units)}")
        print("\n  Top 15 functions by cumulative time:")
        print(f"  {'Function':<45} {'Calls':>8} {'TotTime':>10} {'CumTime':>10}")
        print(f"  {'-'*45} {'-'*8} {'-'*10} {'-'*10}")
        for fn in profile_data["top_functions"][:15]:
            print(f"  {fn['function']:<45} {fn['calls']:>8} {fn['total_time']:>10.4f} {fn['cumulative_time']:>10.4f}")

        # Print full stats to stdout for manual review
        profile.print_stats(top_n=20)

    @pytest.mark.timeout(180)
    def test_grow_network_cprofile(self, small_spiral_data):
        """Profile grow_network() in isolation (after initial output training)."""
        x_train, y_train = small_spiral_data

        config = _make_small_training_config(max_hidden_units=3)
        net = CascadeCorrelationNetwork(config=config)

        # Pre-train output layer so grow_network starts from a reasonable state
        net.train_output_layer(x_train, y_train, epochs=25)

        with ProfileContext("grow_network_only") as profile:
            net.grow_network(x_train=x_train, y_train=y_train, max_epochs=3)

        profile_data = profile.to_dict(top_n=20)
        assert profile_data.get("total_calls", 0) > 0

        print("\n" + "=" * 70)
        print("STEP 4.1: grow_network() Isolated cProfile Results")
        print("=" * 70)
        print(f"  Total function calls:  {profile_data['total_calls']}")
        print(f"  Wall-clock time:       {profile_data['elapsed_seconds']:.3f}s")
        print(f"  Hidden units added:    {len(net.hidden_units)}")
        print("\n  Top 10 functions by cumulative time:")
        for fn in profile_data["top_functions"][:10]:
            print(f"    {fn['function']:<45} calls={fn['calls']:<6} cum={fn['cumulative_time']:.4f}s")

        profile.print_stats(top_n=15)


# ===================================================================
# STEP 4.2: Training Phase Time Distribution
# ===================================================================


@pytest.mark.performance
class TestTrainingPhaseTimeDistribution:
    """Instrument a grow_network cycle to measure time in each training phase.

    Uses monkey-patching to wrap key methods with timing instrumentation,
    then reports the percentage of total time spent in each phase.
    """

    @pytest.mark.timeout(180)
    def test_phase_time_distribution(self, small_spiral_data):
        """Measure time distribution across training phases in grow_network."""
        x_train, y_train = small_spiral_data

        config = _make_small_training_config(max_hidden_units=3)
        net = CascadeCorrelationNetwork(config=config)

        # Pre-train output layer
        net.train_output_layer(x_train, y_train, epochs=25)

        # Methods to instrument (these are the key phases in grow_network)
        methods_to_time = [
            "_calculate_residual_error_safe",
            "_get_training_results",
            "add_unit",
            "train_output_layer",
            "validate_training",
            "calculate_accuracy",
        ]

        timings: Dict[str, float] = {}
        call_counts: Dict[str, int] = {}
        originals: Dict[str, object] = {}

        # Monkey-patch each method with timing wrapper
        for method_name in methods_to_time:
            if not hasattr(net, method_name):
                continue
            original_method = getattr(net, method_name)
            originals[method_name] = original_method
            timings[method_name] = 0.0
            call_counts[method_name] = 0

            def make_timed_wrapper(name, orig):
                def timed_method(*args, **kwargs):
                    t0 = time.perf_counter()
                    result = orig(*args, **kwargs)
                    elapsed = time.perf_counter() - t0
                    timings[name] = timings.get(name, 0.0) + elapsed
                    call_counts[name] = call_counts.get(name, 0) + 1
                    return result

                return timed_method

            setattr(net, method_name, make_timed_wrapper(method_name, original_method))

        # Run grow_network
        wall_start = time.perf_counter()
        net.grow_network(x_train=x_train, y_train=y_train, max_epochs=3)
        wall_total = time.perf_counter() - wall_start

        # Restore original methods
        for method_name, original in originals.items():
            setattr(net, method_name, original)

        # Calculate percentages and report
        instrumented_total = sum(timings.values())
        overhead = wall_total - instrumented_total

        print(f"\n{'='*70}")
        print("STEP 4.2: Training Phase Time Distribution")
        print(f"{'='*70}")
        print(f"  Total wall-clock time:     {wall_total:.4f}s")
        print(f"  Instrumented total:        {instrumented_total:.4f}s")
        print(f"  Uninstrumented overhead:   {overhead:.4f}s ({100 * overhead / wall_total:.1f}%)")
        print(f"  Hidden units added:        {len(net.hidden_units)}")
        print(f"\n  {'Phase':<40} {'Time (s)':>10} {'Calls':>8} {'% of Total':>12}")
        print(f"  {'-'*40} {'-'*10} {'-'*8} {'-'*12}")

        for method_name in sorted(timings, key=lambda k: timings[k], reverse=True):
            t = timings[method_name]
            pct = 100 * t / wall_total if wall_total > 0 else 0
            print(f"  {method_name:<40} {t:>10.4f} {call_counts[method_name]:>8} {pct:>11.1f}%")

        print(f"  {'(uninstrumented overhead)':<40} {overhead:>10.4f} {'':>8} {100 * overhead / wall_total:>11.1f}%")

        # Assertions: total instrumented time should account for most of the wall clock
        assert wall_total > 0, "Training should take measurable time"
        assert len(timings) > 0, "At least some methods should have been timed"

        # The instrumented methods should account for a substantial fraction of time
        # (some overhead from logging, loop control, etc. is expected)
        if wall_total > 0.01:  # Only assert ratio for non-trivially-short runs
            instrumented_ratio = instrumented_total / wall_total
            assert instrumented_ratio > 0.5, f"Instrumented methods should account for >50% of wall time, " f"got {instrumented_ratio:.1%}"

    @pytest.mark.timeout(180)
    def test_phase_distribution_with_more_epochs(self, small_spiral_data):
        """Measure phase distribution with max_hidden=5 for more growth iterations."""
        x_train, y_train = small_spiral_data

        config = _make_small_training_config(max_hidden_units=5)
        net = CascadeCorrelationNetwork(config=config)
        net.train_output_layer(x_train, y_train, epochs=25)

        methods_to_time = [
            "_calculate_residual_error_safe",
            "_get_training_results",
            "add_unit",
            "train_output_layer",
            "validate_training",
            "calculate_accuracy",
        ]

        timings: Dict[str, float] = {}
        call_counts: Dict[str, int] = {}
        originals: Dict[str, object] = {}

        for method_name in methods_to_time:
            if not hasattr(net, method_name):
                continue
            original_method = getattr(net, method_name)
            originals[method_name] = original_method
            timings[method_name] = 0.0
            call_counts[method_name] = 0

            def make_timed_wrapper(name, orig):
                def timed_method(*args, **kwargs):
                    t0 = time.perf_counter()
                    result = orig(*args, **kwargs)
                    elapsed = time.perf_counter() - t0
                    timings[name] = timings.get(name, 0.0) + elapsed
                    call_counts[name] = call_counts.get(name, 0) + 1
                    return result

                return timed_method

            setattr(net, method_name, make_timed_wrapper(method_name, original_method))

        wall_start = time.perf_counter()
        net.grow_network(x_train=x_train, y_train=y_train, max_epochs=5)
        wall_total = time.perf_counter() - wall_start

        for method_name, original in originals.items():
            setattr(net, method_name, original)

        print(f"\n{'='*70}")
        print("STEP 4.2: Phase Distribution (max_hidden=5)")
        print(f"{'='*70}")
        print(f"  Wall-clock: {wall_total:.4f}s | Hidden units: {len(net.hidden_units)}")
        for method_name in sorted(timings, key=lambda k: timings[k], reverse=True):
            t = timings[method_name]
            pct = 100 * t / wall_total if wall_total > 0 else 0
            print(f"  {method_name:<40} {t:.4f}s ({pct:.1f}%) x{call_counts[method_name]}")

        assert wall_total > 0


# ===================================================================
# STEP 4.3: Memory Growth Profiling
# ===================================================================


@pytest.mark.performance
class TestMemoryGrowthProfiling:
    """Track RSS memory at each growth epoch.

    Measures base memory before training, per-hidden-unit memory increment,
    and peak memory during training using the resource module and /proc.
    """

    @pytest.mark.timeout(180)
    def test_memory_growth_per_epoch(self, small_spiral_data):
        """Track RSS memory at each growth epoch via monkey-patched grow_network."""
        x_train, y_train = small_spiral_data

        config = _make_small_training_config(max_hidden_units=5)
        net = CascadeCorrelationNetwork(config=config)

        # Measure baseline memory before any training
        base_rss_mb = _get_rss_from_proc_mb()

        # Pre-train output layer
        net.train_output_layer(x_train, y_train, epochs=25)
        post_output_rss_mb = _get_rss_from_proc_mb()

        # Track memory at each epoch by wrapping add_unit
        memory_snapshots: List[Dict] = []
        original_add_unit = net.add_unit

        def tracking_add_unit(*args, **kwargs):
            result = original_add_unit(*args, **kwargs)
            rss_mb = _get_rss_from_proc_mb()
            memory_snapshots.append(
                {
                    "hidden_units": len(net.hidden_units),
                    "rss_mb": rss_mb,
                }
            )
            return result

        net.add_unit = tracking_add_unit

        # Run grow_network
        net.grow_network(x_train=x_train, y_train=y_train, max_epochs=5)

        # Restore
        net.add_unit = original_add_unit

        # Final measurement
        final_rss_mb = _get_rss_from_proc_mb()

        print(f"\n{'='*70}")
        print("STEP 4.3: Memory Growth Per Epoch")
        print(f"{'='*70}")
        print(f"  Base RSS (before training):    {base_rss_mb:.2f} MB")
        print(f"  Post-output-training RSS:      {post_output_rss_mb:.2f} MB")

        if memory_snapshots:
            print(f"\n  {'Epoch':<8} {'Hidden Units':>14} {'RSS (MB)':>12} {'Delta (MB)':>12}")
            print(f"  {'-'*8} {'-'*14} {'-'*12} {'-'*12}")
            prev_rss = post_output_rss_mb
            for snap in memory_snapshots:
                delta = snap["rss_mb"] - prev_rss
                print(f"  {snap['hidden_units']:<8} {snap['hidden_units']:>14} {snap['rss_mb']:>12.2f} {delta:>+12.2f}")
                prev_rss = snap["rss_mb"]

            # Calculate per-unit increment
            if len(memory_snapshots) >= 2:
                first_rss = memory_snapshots[0]["rss_mb"]
                last_rss = memory_snapshots[-1]["rss_mb"]
                n_units = memory_snapshots[-1]["hidden_units"] - memory_snapshots[0]["hidden_units"]
                if n_units > 0:
                    per_unit_mb = (last_rss - first_rss) / n_units
                    print(f"\n  Avg memory per hidden unit:    {per_unit_mb:+.4f} MB")

        print(f"  Final RSS:                     {final_rss_mb:.2f} MB")
        print(f"  Total growth:                  {final_rss_mb - base_rss_mb:+.2f} MB")
        print(f"  Hidden units added:            {len(net.hidden_units)}")

        # Basic assertions
        assert base_rss_mb > 0, "Base RSS should be measurable"
        assert final_rss_mb >= base_rss_mb, "RSS should not decrease (monotonic max)"

    @pytest.mark.timeout(180)
    def test_peak_memory_during_training(self, small_spiral_data):
        """Measure peak RSS during candidate training phases."""
        x_train, y_train = small_spiral_data

        config = _make_small_training_config(max_hidden_units=3)
        net = CascadeCorrelationNetwork(config=config)

        # Reset peak via resource module
        # Note: ru_maxrss tracks peak RSS since process start on Linux;
        # we record before/after to get delta
        rss_before_kb = _get_rss_kb()

        net.train_output_layer(x_train, y_train, epochs=25)
        rss_after_output_kb = _get_rss_kb()

        net.grow_network(x_train=x_train, y_train=y_train, max_epochs=3)
        rss_after_grow_kb = _get_rss_kb()

        peak_during_grow_mb = (rss_after_grow_kb - rss_before_kb) / 1024.0

        print(f"\n{'='*70}")
        print("STEP 4.3: Peak Memory During Training")
        print(f"{'='*70}")
        print(f"  RSS before training:      {rss_before_kb / 1024:.2f} MB")
        print(f"  RSS after output train:   {rss_after_output_kb / 1024:.2f} MB")
        print(f"  RSS after grow_network:   {rss_after_grow_kb / 1024:.2f} MB")
        print(f"  Peak growth delta:        {peak_during_grow_mb:+.2f} MB")
        print(f"  Hidden units:             {len(net.hidden_units)}")

        # Sanity: peak RSS should be non-negative
        assert rss_after_grow_kb >= rss_before_kb

    @pytest.mark.timeout(180)
    def test_memory_growth_medium_data(self, medium_spiral_data):
        """Track memory growth with a larger dataset (400 samples)."""
        x_train, y_train = medium_spiral_data

        config = _make_small_training_config(max_hidden_units=3)
        net = CascadeCorrelationNetwork(config=config)

        base_rss_mb = _get_rss_from_proc_mb()
        net.fit(x_train, y_train, max_epochs=3)
        final_rss_mb = _get_rss_from_proc_mb()

        print(f"\n{'='*70}")
        print("STEP 4.3: Memory Growth (Medium Dataset, 400 samples)")
        print(f"{'='*70}")
        print(f"  Base RSS:      {base_rss_mb:.2f} MB")
        print(f"  Final RSS:     {final_rss_mb:.2f} MB")
        print(f"  Growth:        {final_rss_mb - base_rss_mb:+.2f} MB")
        print(f"  Hidden units:  {len(net.hidden_units)}")

        assert final_rss_mb >= base_rss_mb


# ===================================================================
# STEP 4.4: Training Convergence vs Performance Tradeoffs
# ===================================================================


@pytest.mark.performance
class TestConvergenceVsPerformanceTradeoffs:
    """Parametrized tests measuring time-to-accuracy under different hyperparameters.

    Explores the tradeoff space between candidate_pool_size, candidate_epochs,
    and patience to identify configurations that balance training speed and
    final accuracy.
    """

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize(
        "pool_size,candidate_epochs",
        [
            (2, 10),
            (4, 20),
            (4, 30),
            (8, 20),
        ],
        ids=["pool2_ep10", "pool4_ep20", "pool4_ep30", "pool8_ep20"],
    )
    def test_pool_size_vs_epochs(self, small_spiral_data, pool_size, candidate_epochs):
        """Measure time and accuracy for different pool_size x candidate_epochs combos."""
        x_train, y_train = small_spiral_data

        config = _make_small_training_config(
            candidate_pool_size=pool_size,
            candidate_epochs=candidate_epochs,
            max_hidden_units=3,
            patience=3,
        )
        net = CascadeCorrelationNetwork(config=config)

        timer = BenchmarkTimer()
        with timer:
            net.fit(x_train, y_train, max_epochs=3)

        final_accuracy = net.calculate_accuracy(x_train, y_train)
        elapsed_ms = timer.times_ms[0]

        print(f"\n{'='*70}")
        print(f"STEP 4.4: pool_size={pool_size}, candidate_epochs={candidate_epochs}")
        print(f"{'='*70}")
        print(f"  Training time:     {elapsed_ms:.1f} ms")
        print(f"  Final accuracy:    {final_accuracy:.4f}")
        print(f"  Hidden units:      {len(net.hidden_units)}")
        print(f"  Time per unit:     {elapsed_ms / max(len(net.hidden_units), 1):.1f} ms")

        # Verify training produced a valid network
        assert final_accuracy >= 0.0
        assert final_accuracy <= 1.0
        assert elapsed_ms > 0

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize(
        "patience",
        [1, 3, 5],
        ids=["patience1", "patience3", "patience5"],
    )
    def test_patience_tradeoff(self, small_spiral_data, patience):
        """Measure how patience affects training time and final accuracy."""
        x_train, y_train = small_spiral_data

        config = _make_small_training_config(
            candidate_pool_size=4,
            candidate_epochs=20,
            max_hidden_units=3,
            patience=patience,
        )
        net = CascadeCorrelationNetwork(config=config)

        timer = BenchmarkTimer()
        with timer:
            net.fit(x_train, y_train, max_epochs=5)

        final_accuracy = net.calculate_accuracy(x_train, y_train)
        elapsed_ms = timer.times_ms[0]

        print(f"\n{'='*70}")
        print(f"STEP 4.4: patience={patience}")
        print(f"{'='*70}")
        print(f"  Training time:     {elapsed_ms:.1f} ms")
        print(f"  Final accuracy:    {final_accuracy:.4f}")
        print(f"  Hidden units:      {len(net.hidden_units)}")

        assert final_accuracy >= 0.0
        assert elapsed_ms > 0

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize(
        "candidate_epochs,output_epochs",
        [
            (10, 10),
            (20, 25),
            (30, 25),
            (20, 50),
        ],
        ids=["cand10_out10", "cand20_out25", "cand30_out25", "cand20_out50"],
    )
    def test_epoch_budget_allocation(self, small_spiral_data, candidate_epochs, output_epochs):
        """Measure how shifting epochs between candidate and output training affects outcome."""
        x_train, y_train = small_spiral_data

        config = _make_small_training_config(
            candidate_pool_size=4,
            candidate_epochs=candidate_epochs,
            output_epochs=output_epochs,
            max_hidden_units=3,
            patience=3,
        )
        net = CascadeCorrelationNetwork(config=config)

        timer = BenchmarkTimer()
        with timer:
            net.fit(x_train, y_train, max_epochs=3)

        final_accuracy = net.calculate_accuracy(x_train, y_train)
        elapsed_ms = timer.times_ms[0]
        total_epoch_budget = candidate_epochs + output_epochs

        print(f"\n{'='*70}")
        print(f"STEP 4.4: candidate_epochs={candidate_epochs}, output_epochs={output_epochs}")
        print(f"{'='*70}")
        print(f"  Total epoch budget:  {total_epoch_budget}")
        print(f"  Training time:       {elapsed_ms:.1f} ms")
        print(f"  Final accuracy:      {final_accuracy:.4f}")
        print(f"  Hidden units:        {len(net.hidden_units)}")
        print(f"  Efficiency:          {final_accuracy / (elapsed_ms / 1000):.4f} acc/sec")

        assert final_accuracy >= 0.0
        assert elapsed_ms > 0

    @pytest.mark.timeout(300)
    def test_convergence_summary_table(self, small_spiral_data):
        """Run a grid of configurations and print a summary comparison table."""
        x_train, y_train = small_spiral_data

        configs = [
            {"label": "minimal", "candidate_pool_size": 2, "candidate_epochs": 10, "output_epochs": 10, "max_hidden_units": 3, "patience": 1},
            {"label": "balanced", "candidate_pool_size": 4, "candidate_epochs": 20, "output_epochs": 25, "max_hidden_units": 3, "patience": 3},
            {"label": "thorough", "candidate_pool_size": 4, "candidate_epochs": 30, "output_epochs": 25, "max_hidden_units": 5, "patience": 5},
            {"label": "wide_pool", "candidate_pool_size": 8, "candidate_epochs": 20, "output_epochs": 25, "max_hidden_units": 3, "patience": 3},
        ]

        results: List[Dict] = []

        for cfg in configs:
            label = cfg.pop("label")
            config = _make_small_training_config(**cfg)
            net = CascadeCorrelationNetwork(config=config)

            timer = BenchmarkTimer()
            with timer:
                net.fit(x_train, y_train, max_epochs=cfg.get("max_hidden_units", 3))

            accuracy = net.calculate_accuracy(x_train, y_train)
            elapsed_ms = timer.times_ms[0]

            results.append(
                {
                    "label": label,
                    "elapsed_ms": elapsed_ms,
                    "accuracy": accuracy,
                    "hidden_units": len(net.hidden_units),
                    "pool_size": cfg["candidate_pool_size"],
                    "cand_epochs": cfg["candidate_epochs"],
                    "out_epochs": cfg["output_epochs"],
                    "patience": cfg["patience"],
                }
            )

        # Print summary table
        print(f"\n{'='*70}")
        print("STEP 4.4: Convergence vs Performance Summary")
        print(f"{'='*70}")
        print(f"  {'Config':<12} {'Pool':>5} {'CEp':>5} {'OEp':>5} {'Pat':>5} {'Time(ms)':>10} {'Acc':>8} {'Units':>6}")
        print(f"  {'-'*12} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*10} {'-'*8} {'-'*6}")
        for r in results:
            print(f"  {r['label']:<12} {r['pool_size']:>5} {r['cand_epochs']:>5} " f"{r['out_epochs']:>5} {r['patience']:>5} {r['elapsed_ms']:>10.1f} " f"{r['accuracy']:>8.4f} {r['hidden_units']:>6}")

        # Verify all configs produced valid results
        for r in results:
            assert r["accuracy"] >= 0.0
            assert r["elapsed_ms"] > 0
