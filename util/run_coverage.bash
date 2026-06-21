#!/usr/bin/env bash
#####################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# File Name:     util/run_coverage.bash
# Author:        Paul Calnon
# Version:       0.1.0
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Reproduce the CI coverage gate locally (full suite). Mirrors the coverage
#    invocation enforced in .github/workflows/ci.yml (parallel coverage with a
#    custom data_file) so a developer can verify the aggregate gate before pushing.
#    Runs the FULL gated suite by design; use plain pytest for a subset.
#
# Usage:
#    bash util/run_coverage.bash                          # full suite + gate
#    make coverage                                        # equivalent wrapper
#    COVERAGE_FAIL_UNDER=90 bash util/run_coverage.bash   # override the gate
#
# References:
#    - https://pytest-cov.readthedocs.io/
#####################################################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

COVERAGE_FAIL_UNDER="${COVERAGE_FAIL_UNDER:-80}"

echo "==> Coverage (reproduces CI gate: ${COVERAGE_FAIL_UNDER}% aggregate) — ${REPO_ROOT}"

# ── Reproduce the CI coverage sequence (keep in sync with .github/workflows/ci.yml) ──
# Step 1 — Create required directories. pyproject.toml's
# [tool.coverage.run] data_file = "src/tests/reports/.coverage" means pytest-cov
# silently fails to write its parallel .coverage.* files if the parent dir does
# not exist (gitignored, so absent on a fresh checkout), which then leaves the
# coverage-report step with "No data to report".
mkdir -p logs src/logs reports/junit reports/htmlcov src/tests/reports

# Step 2 — Run the gated unit suite under coverage. Same selection/markers as CI;
# the xml/html report flags are CI artifacts only and are dropped locally.
python -m pytest \
  -m "unit and not slow" \
  src/tests/unit \
  --verbose \
  --timeout=60 \
  --maxfail=5 \
  --cov=src \
  --cov-report=term-missing

# Step 3 — Enforce the aggregate coverage gate (CI does NOT pass --cov-fail-under
# inline; it runs a standalone coverage report against the parallel data_file).
python -m coverage report --fail-under="${COVERAGE_FAIL_UNDER}"
# ─────────────────────────────────────────────────────────────────────────────────────
