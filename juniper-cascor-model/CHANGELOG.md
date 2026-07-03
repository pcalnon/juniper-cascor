# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **First CI coverage gate (per-file coverage rollout C-5).** `ci-cascor-model.yml` now
  measures statement coverage across all five shipped packages (`juniper_cascor_model`,
  `candidate_unit`, `utils`, `log_config`, `cascor_constants`) and runs the blocking
  `juniper-coverage-gap-map --enforce` step — failing the build when any source file is
  below 90% statement coverage or any sub-module's pooled (statement-weighted) coverage is
  below 95%. Previously the package ran a bare `pytest -v` with no coverage gate (the
  ecosystem's last no-gate outlier). See juniper-ml
  `notes/JUNIPER_ECOSYSTEM_PER_FILE_COVERAGE_ROLLOUT_SCOPING_2026-06-30.md`.

### Tests

- **Lifted the package to the ratified coverage bars** (overall 72% → 99%). Added
  real-instantiation coverage for the `Logger`/`LogConfig` bootstrap paths (custom-level
  registration, YAML `dictConfig` success + best-effort fallback, the already-configured
  short-circuit), TRACE-level candidate-training tests that exercise the level-gated
  diagnostic branches, `dill`/`columnar` optional-dependency import guards, and ported the
  drift-identical `candidate_unit` / `logger` / `utils` coverage suites from
  `juniper-cascor/src`. The CI test job now installs the `[full]` extra so the `utils`
  debug-helper paths are measured. No source files changed; existing tests untouched.

### Security

- **Raised the `torch` floor to `>=2.10.0`** (was `>=2.0`) — the minimal pin that clears
  every *fixable* torch CVE affecting `>=2.0`, up to and including
  [CVE-2025-3001](https://github.com/advisories/GHSA-qfhq-4f3w-5fph) (`lstm_cell` memory
  corruption, fixed in torch 2.10.0). Verified against OSV + GHSA on 2026-06-14. Four torch
  CVEs remain unfixed upstream at every version (CVE-2025-2148, CVE-2025-2149, CVE-2025-2998,
  CVE-2025-3000) and cannot be addressed by any version pin. The deployed `JuniperCascor1`
  runtime already runs torch 2.11.0, which satisfies the new floor.

## [0.1.0] - 2026-06-04

### Added

- **Initial extraction (CW-05 Wave 0).** `juniper-cascor-model` packages the CasCor
  candidate-training core — `candidate_unit/`, `utils/` (utils + activation registry),
  `log_config/`, and candidate-relevant `cascor_constants/` — extracted verbatim from
  `juniper-cascor/src`. Shipped under the same top-level package names cascor uses
  (migration plan §3.1 option (i)) so consumers' imports resolve unchanged. Zero coupling
  to the cascor server/training stack (no FastAPI, `cascade_correlation`, or `api`).
  Enables `juniper-cascor-worker` to execute remote candidates via a single PyPI dependency
  instead of `--cascor-path` + a cascor source mount
  ([juniper-cascor-worker#97](https://github.com/pcalnon/juniper-cascor-worker/issues/97)).
- **Deployment-agnostic logging.** The shared logger now honors `JUNIPER_CASCOR_LOG_DIR`
  for the log-file directory and degrades to console-only (rather than raising) when the
  directory is missing or unwritable — fixes the worker's
  `[Errno 2] '/logs/juniper_cascor.log'` candidate-training crash (CW-05 gap #3). This is
  the one intentional divergence from cascor src (to be backported in Wave 2; tracked by the
  drift-guard allowlist).

### Notes

- Runtime deps: `numpy`, `torch`, `PyYAML`. Optional `[full]` extra (`dill`, `columnar`)
  for the lazily-imported dev/debug helpers in `utils.py`.
