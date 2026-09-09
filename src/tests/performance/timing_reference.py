#!/usr/bin/env python
"""
Project:       Juniper
Sub-Project:   JuniperCascor
File Name:     timing_reference.py
File Path:     src/tests/performance/

Author:        Paul Calnon

Date Created:  2026-09-08
Last Modified: 2026-09-08

License:       MIT License
Copyright:     Copyright (c) 2024-2026 Paul Calnon

Description:
    Helpers behind the micro-tier TIMING REFERENCE (juniper-ml perf lane, P2 item 2.4 / PF-4).

    Until 2026-09-08 no cascor baseline file had ever carried a timing figure: ``save_baseline``
    promised ``mean_ms`` / ``stddev_ms`` / ... in its docstring, every caller passed only the
    test's parameters, and every historical ``baseline_*.json`` (2026-03-31 .. 2026-05-26) holds
    ``hidden_units`` and memory keys and nothing else. This module supplies what a reference needs
    and the tests never had:

      * ``benchmark_stats_ms`` -- the pytest-benchmark statistics of a completed ``benchmark``
        fixture run, in milliseconds, so a baseline entry records what it measured.
      * ``collect_environment`` -- host identity (``cpu_model``, thread pins, ``git_sha``) and the
        load the host carried when the figure was taken (``loadavg_1m``). A timing figure without
        the condition it was taken under is a number, not a reference.
      * ``stable_machine_fields`` / ``run_condition_fields`` -- the same split for pytest-benchmark's
        saved JSON: identity goes into ``machine_info`` (compared on ``--benchmark-compare``; a
        difference WARNS), the load condition goes beside it (recorded, never compared -- putting a
        load average into ``machine_info`` would make every comparison warn).

    REPORT-ONLY BY DECISION. The owner ruled 2026-09-07 (P2 item 2.5) that PF-4 stays report-only
    on timing: the quiet-host noise band on the reference workstation is 20.5% and six competing
    processes cost +19.9%, so any percentage tolerance is either blind to regressions or fires on
    an ordinary loaded host. Nothing here asserts on a timing; the memory gate in
    ``test_baselines.py`` is unchanged.

    Kept separate from ``conftest.py`` so ``src/tests/unit`` can import and pin it: pytest loads a
    conftest as a plugin, not as a module a unit test can reach.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

#: Every benchmark runs under this torch / BLAS thread pin (the autouse fixture in ``conftest.py``
#: applies it per test). Recorded as identity rather than read back at save time, because by then
#: the fixture has restored the interpreter's original thread count and a read would report that.
BENCHMARK_THREAD_PIN = 1

UNKNOWN = "unknown"

_MS_PER_S = 1000.0


def _to_ms(seconds: Any) -> float:
    return float(seconds) * _MS_PER_S


def benchmark_stats_ms(benchmark: Any) -> Dict[str, Any]:
    """Timing summary of a completed ``benchmark`` fixture run, in milliseconds.

    pytest-benchmark keeps its statistics in SECONDS on ``benchmark.stats.stats`` (a ``Metadata``
    wrapping a ``Stats``) once ``benchmark(...)`` or ``benchmark.pedantic(...)`` has run. This is
    the summary ``save_baseline``'s docstring has always promised and no caller ever supplied.

    Returns ``{}`` when there are no statistics -- ``--benchmark-disable`` runs the function once
    without timing it, and a fixture that was never invoked has none -- so a caller can splat the
    result into its ``results`` dict unconditionally.
    """
    meta = getattr(benchmark, "stats", None)
    stats = getattr(meta, "stats", None)
    if stats is None or not getattr(stats, "rounds", 0):
        return {}
    return {
        "mean_ms": _to_ms(stats.mean),
        "median_ms": _to_ms(stats.median),
        "stddev_ms": _to_ms(stats.stddev),
        "min_ms": _to_ms(stats.min),
        "max_ms": _to_ms(stats.max),
        "rounds": int(stats.rounds),
        "iterations": int(getattr(meta, "iterations", 1) or 1),
    }


def cpu_model() -> str:
    """The CPU model string -- ``/proc/cpuinfo`` on Linux, where ``platform.processor()`` is empty."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip() or UNKNOWN
    except OSError:
        pass
    return platform.processor() or UNKNOWN


def git_sha(cwd: Optional[Path] = None) -> str:
    """HEAD sha of the tree the benchmarks ran from, or ``"unknown"``. Never raises."""
    try:
        completed = subprocess.run(  # nosec B603 B607 -- fixed argv, no shell, 5 s cap
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    sha = completed.stdout.strip()
    return sha if completed.returncode == 0 and sha else UNKNOWN


def load_average() -> Optional[Dict[str, float]]:
    """1 / 5 / 15-minute load averages, or ``None`` where the platform has none."""
    getter = getattr(os, "getloadavg", None)
    if getter is None:
        return None
    try:
        one, five, fifteen = getter()
    except OSError:
        return None
    return {"1m": float(one), "5m": float(five), "15m": float(fifteen)}


def collect_environment() -> Dict[str, Any]:
    """Environment, host identity and load condition for a ``save_baseline`` entry.

    Called from INSIDE a test, so ``torch_num_threads`` reports the pinned value; the three BLAS
    variables are read back rather than assumed so a run that escaped the pin is visible.
    """
    load = load_average()
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model(),
        "torch_num_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "unset"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "loadavg_1m": None if load is None else load["1m"],
        "git_sha": git_sha(),
    }


def stable_machine_fields() -> Dict[str, Any]:
    """Identity fields for pytest-benchmark's ``machine_info`` -- a change WARNS on compare.

    Mirrors the run tier's ``HOST.json`` (``cpu_model``, ``cpu_count``, thread budget): a figure
    from another host, another torch build or another thread pin is not comparable. Deliberately
    contains nothing volatile -- see ``run_condition_fields``.
    """
    return {
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "thread_pin": BENCHMARK_THREAD_PIN,
    }


def run_condition_fields() -> Dict[str, Any]:
    """The condition a saved run was taken under -- recorded beside the JSON, never compared."""
    return {
        "loadavg": load_average(),
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
    }
