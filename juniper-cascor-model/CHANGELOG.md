# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
