#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# File Name:     run_benchmarks.bash
# Author:        Paul Calnon
# Version:       0.3.16
#
# Date Created:  2026-01-24
# Last Modified: 2026-01-24
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Performance benchmark harness for Juniper Cascor.
#    Measures serialization, training, and forward pass performance.
#
# Usage:
#    bash run_benchmarks.bash [OPTIONS]
#
# Options:
#    -s, --serialization    Run serialization benchmarks
#    -t, --training         Run training benchmarks
#    -f, --forward          Run forward pass benchmarks
#    -a, --all              Run all benchmarks (default)
#    -o, --output FILE      Output results to file (default: stdout)
#    -n, --iterations N     Number of iterations per benchmark (default: 5)
#    -q, --quiet            Quiet mode (minimal log output)
#    -h, --help             Show this help
#
# References:
#    - CASCOR-P3-004: Performance Benchmark Harness
#####################################################################################################################################################################################################

set -e

# Get script directory
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
TESTS_DIR="$(dirname "${SCRIPT_DIR}")"
SRC_DIR="$(dirname "${TESTS_DIR}")"
PROJECT_ROOT="$(dirname "${SRC_DIR}")"

# Default values
RUN_SERIALIZATION=false
RUN_TRAINING=false
RUN_FORWARD=false
RUN_ALL=true
OUTPUT_FILE=""
ITERATIONS=5
QUIET=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

show_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Performance benchmark harness for Juniper Cascor.

OPTIONS:
    -s, --serialization    Run serialization benchmarks
    -t, --training         Run training benchmarks (short)
    -f, --forward          Run forward pass benchmarks
    -a, --all              Run all benchmarks (default)
    -o, --output FILE      Output results to file
    -n, --iterations N     Number of iterations (default: 5)
    -q, --quiet            Quiet mode (set CASCOR_LOG_LEVEL=WARNING)
    -h, --help             Show this help

EXAMPLES:
    $0                     # Run all benchmarks
    $0 -s -n 10            # Run serialization benchmark 10 times
    $0 -t -q               # Run training benchmark quietly
    $0 -a -o results.txt   # Run all and save to file

EOF
}

# Parse arguments
while (($# > 0)); do
    case $1 in
    -s | --serialization)
        RUN_SERIALIZATION=true
        RUN_ALL=false
        shift
        ;;
    -t | --training)
        RUN_TRAINING=true
        RUN_ALL=false
        shift
        ;;
    -f | --forward)
        RUN_FORWARD=true
        RUN_ALL=false
        shift
        ;;
    -a | --all)
        RUN_ALL=true
        shift
        ;;
    -o | --output)
        OUTPUT_FILE="$2"
        shift 2
        ;;
    -n | --iterations)
        ITERATIONS="$2"
        shift 2
        ;;
    -q | --quiet)
        QUIET=true
        shift
        ;;
    -h | --help)
        show_usage
        exit 0
        ;;
    *)
        print_color "${RED}" "Unknown option: $1"
        show_usage
        exit 1
        ;;
    esac
done

# Set quiet mode if requested
if [[ "${QUIET}" == "true" ]]; then
    export CASCOR_LOG_LEVEL=WARNING
fi

# Change to source directory
cd "${SRC_DIR}" || exit 1

print_color "${BLUE}" "╔════════════════════════════════════════════════════════════╗"
print_color "${BLUE}" "║       Juniper Cascor Performance Benchmark Harness         ║"
print_color "${BLUE}" "╚════════════════════════════════════════════════════════════╝"
print_color "${CYAN}" "Source Directory: ${SRC_DIR}"
print_color "${CYAN}" "Iterations: ${ITERATIONS}"
print_color "${CYAN}" "Quiet Mode: ${QUIET}"
echo ""

# Create Python benchmark script inline
BENCHMARK_SCRIPT=$(cat <<'PYTHON_SCRIPT'
#!/usr/bin/env python
"""
Performance Benchmark Suite for Juniper Cascor
CASCOR-P3-004: Performance Benchmark Harness
"""
import os
import sys
import time
import tempfile
import statistics
from typing import Callable, List, Tuple

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np


def time_function(func: Callable, iterations: int = 5, warmup: int = 1) -> Tuple[float, float, float]:
    """
    Time a function over multiple iterations.
    Returns: (mean_time, std_dev, min_time) in seconds
    """
    # Warmup runs
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    mean_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
    min_time = min(times)
    return mean_time, std_dev, min_time


def benchmark_serialization(iterations: int = 5) -> dict:
    """Benchmark HDF5 serialization performance."""
    from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
    from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

    results = {"save": {}, "load": {}, "verify": {}}

    # Create network with various sizes
    for hidden_units in [0, 10, 50]:
        config = CascadeCorrelationConfig(
            input_size=2,
            output_size=2,
            random_seed=42,
        )
        network = CascadeCorrelationNetwork(config=config)

        # Add hidden units
        for i in range(hidden_units):
            network.hidden_units.append({
                "weights": torch.randn(network.input_size + i),
                "bias": torch.randn(1),
                "activation": network.activation_fn,
            })

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Benchmark save
            def save_network():
                network.save_to_hdf5(filepath)

            mean, std, min_t = time_function(save_network, iterations)
            results["save"][f"{hidden_units}_units"] = {"mean": mean, "std": std, "min": min_t}

            # Benchmark load
            def load_network():
                CascadeCorrelationNetwork.load_from_hdf5(filepath)

            mean, std, min_t = time_function(load_network, iterations)
            results["load"][f"{hidden_units}_units"] = {"mean": mean, "std": std, "min": min_t}

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    return results


def benchmark_forward_pass(iterations: int = 5) -> dict:
    """Benchmark forward pass performance."""
    from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
    from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

    results = {}

    # Test different batch sizes and hidden unit counts
    for batch_size in [1, 32, 128]:
        for hidden_units in [0, 10, 50]:
            config = CascadeCorrelationConfig(
                input_size=2,
                output_size=2,
                random_seed=42,
            )
            network = CascadeCorrelationNetwork(config=config)

            # Add hidden units
            for i in range(hidden_units):
                network.hidden_units.append({
                    "weights": torch.randn(network.input_size + i),
                    "bias": torch.randn(1),
                    "activation": network.activation_fn,
                })

            # Create input batch
            x = torch.randn(batch_size, network.input_size)

            def forward():
                network.forward(x)

            mean, std, min_t = time_function(forward, iterations)
            key = f"batch{batch_size}_units{hidden_units}"
            results[key] = {"mean": mean, "std": std, "min": min_t}

    return results


def benchmark_training(iterations: int = 3) -> dict:
    """Benchmark short training sessions."""
    from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
    from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

    results = {}

    # Generate simple dataset
    np.random.seed(42)
    n_samples = 100
    x = torch.randn(n_samples, 2)
    y = torch.zeros(n_samples, 2)
    y[x[:, 0] > 0, 0] = 1.0
    y[x[:, 0] <= 0, 1] = 1.0

    # Benchmark output layer training
    for epochs in [10, 50]:
        config = CascadeCorrelationConfig(
            input_size=2,
            output_size=2,
            random_seed=42,
        )
        network = CascadeCorrelationNetwork(config=config)

        def train_output():
            network.train_output_layer(x, y, epochs=epochs, display_frequency=1000)

        mean, std, min_t = time_function(train_output, iterations)
        results[f"output_epochs{epochs}"] = {"mean": mean, "std": std, "min": min_t}

    return results


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def print_results(results: dict, title: str):
    """Print benchmark results in formatted table."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")
    print(f"{'Benchmark':<30} {'Mean':>10} {'Std':>10} {'Min':>10}")
    print(f"{'-' * 60}")

    for name, timing in results.items():
        mean = format_time(timing["mean"])
        std = format_time(timing["std"])
        min_t = format_time(timing["min"])
        print(f"{name:<30} {mean:>10} {std:>10} {min_t:>10}")

    print(f"{'=' * 60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Juniper Cascor Performance Benchmarks")
    parser.add_argument("-s", "--serialization", action="store_true", help="Run serialization benchmarks")
    parser.add_argument("-t", "--training", action="store_true", help="Run training benchmarks")
    parser.add_argument("-f", "--forward", action="store_true", help="Run forward pass benchmarks")
    parser.add_argument("-a", "--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("-n", "--iterations", type=int, default=5, help="Iterations per benchmark")
    args = parser.parse_args()

    # Default to all if nothing specified
    if not (args.serialization or args.training or args.forward):
        args.all = True

    print("╔════════════════════════════════════════════════════════════╗")
    print("║       Juniper Cascor Performance Benchmarks                ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"Iterations per benchmark: {args.iterations}")

    if args.all or args.serialization:
        print("\nRunning serialization benchmarks...")
        results = benchmark_serialization(args.iterations)
        print_results(results["save"], "Serialization: Save to HDF5")
        print_results(results["load"], "Serialization: Load from HDF5")

    if args.all or args.forward:
        print("\nRunning forward pass benchmarks...")
        results = benchmark_forward_pass(args.iterations)
        print_results(results, "Forward Pass Performance")

    if args.all or args.training:
        print("\nRunning training benchmarks...")
        results = benchmark_training(min(args.iterations, 3))  # Limit training iterations
        print_results(results, "Output Layer Training Performance")

    print("\n✓ Benchmark suite complete")


if __name__ == "__main__":
    main()
PYTHON_SCRIPT
)

# Write and run benchmark script
BENCHMARK_FILE="${SRC_DIR}/_benchmark_runner.py"
echo "${BENCHMARK_SCRIPT}" > "${BENCHMARK_FILE}"

# Build command
CMD="python ${BENCHMARK_FILE} -n ${ITERATIONS}"

if [[ "${RUN_ALL}" == "true" ]]; then
    CMD="${CMD} -a"
else
    [[ "${RUN_SERIALIZATION}" == "true" ]] && CMD="${CMD} -s"
    [[ "${RUN_TRAINING}" == "true" ]] && CMD="${CMD} -t"
    [[ "${RUN_FORWARD}" == "true" ]] && CMD="${CMD} -f"
fi

# Run benchmarks
if [[ -n "${OUTPUT_FILE}" ]]; then
    print_color "${CYAN}" "Output will be saved to: ${OUTPUT_FILE}"
    eval "${CMD}" | tee "${OUTPUT_FILE}"
else
    eval "${CMD}"
fi

# Cleanup
rm -f "${BENCHMARK_FILE}"

print_color "${GREEN}" "✓ Benchmark harness complete"
