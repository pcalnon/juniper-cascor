# Juniper-CasCor Consolidated Development Record

**Project**: Juniper-CasCor — Cascade Correlation Neural Network
**Version**: 0.4.0
**Generated**: 2026-04-17
**Status**: Validated against current codebase
**Author**: Consolidated from 12 development documents by automated analysis
**Source Documents**: `notes/development/` (see Section 6)

---

## Executive Summary

This document consolidates ALL documented development work for the juniper-cascor application from 12 source documents spanning October 2025 through April 2026. Each item has been validated against the current codebase as of 2026-04-17.

### Statistics

| Category                               | Count |
|----------------------------------------|-------|
| Total documented work items            | ~120  |
| Verified COMPLETE (codebase-validated) | ~80   |
| Still Open (confirmed in codebase)     | 10    |
| Deferred / Future Work                 | ~30   |

### Validation Methodology

Four specialized validation agents independently verified items against the codebase:

- **P0 Critical Bugs Agent**: Validated 8 critical/high bugs → 5 fixed, 3 still present
- **Performance Agent**: Validated 12 optimization items → all 12 implemented
- **Architecture Agent**: Validated 16 architecture/quality items → 12 resolved, 4 still open
- **Cross-reference**: Results reconciled and de-duplicated

---

## 1. Completed Work

### 1.1 Architecture & Migration (Polyrepo)

All 7 phases of the polyrepo migration are **COMPLETE** (verified 2026-03-02).

| Phase | Description                     | Status                    | Key Deliverables                                                                           |
|-------|---------------------------------|---------------------------|--------------------------------------------------------------------------------------------|
| 0     | Stabilize baseline              | ✅ COMPLETE (2026-02-19)  | 4 repos stabilized, pre-migration tags, zero merge conflicts                               |
| 1     | Extract/publish client packages | ✅ COMPLETE               | `juniper-data-client` v0.3.1 on PyPI; vendored copies removed                              |
| 2     | Build CasCor Service API        | ✅ COMPLETE               | FastAPI server, 19 REST + 2 WS endpoints, lifecycle management, Pydantic BaseSettings      |
| 3     | Create cascor-client + worker   | ✅ COMPLETE (2026-02-24)  | `juniper-cascor-client` v0.1.0 + `juniper-cascor-worker` v0.1.0 on PyPI                    |
| 4     | Decouple Canopy from CasCor     | ✅ COMPLETE (2026-02-25)  | `CascorServiceAdapter` (306 lines, 52 tests); `CascorIntegration` deleted (1,601 lines)    |
| 5     | Split into separate repos       | ✅ COMPLETE (2026-02-25)  | 8 repos, SSH deploy keys, per-repo CI/CD; juniper-ml meta-package                          |
| 6     | Post-migration hardening        | ✅ COMPLETE (2026-02-25)  | Docker Compose (juniper-deploy), health check standardization (/v1/health), version matrix |
| 7     | Production readiness            | ✅ COMPLETE (2026-03-02)  | Security hardening, observability, dependency management                                   |

**Microservices Architecture Phases 1-9**: All complete across all 8 active repos (repo creation, PyPI publishing, microservice conversions, health checks, Pydantic BaseSettings configuration standardization).

**Ecosystem compatibility matrix** (verified baseline):

| juniper-canopy | juniper-cascor | juniper-data | data-client | cascor-client | cascor-worker |
|----------------|----------------|--------------|-------------|---------------|---------------|
| 0.2.x          | 0.3.x          | 0.4.x        | ≥0.3.1      | ≥0.1.0        | ≥0.1.0        |

---

### 1.2 Bug Fixes (Verified Fixed)

| ID         | Description                                               | Resolution                                                                        | File(s)                                                                           |
|------------|-----------------------------------------------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| INT-P0-001 | Walrus operator precedence bug                            | ✅ Parentheses added: `if (snapshot_path := self.create_snapshot()) is not None:` | cascade_correlation.py:1664                                                       |
| INT-P0-002 | ActivationWithDerivative class duplicated                 | ✅ Extracted to shared module                                                     | src/utils/activation.py (removed from cascade_correlation.py & candidate_unit.py) |
| INT-P0-003 | Invalid CandidateUnit constructor params in fit()         | ✅ Correct `CandidateUnit__` prefix used                                          | cascade_correlation.py:883-884, 1307-1314                                         |
| INT-P2-002 | `import datetime as pd` misleading alias                  | ✅ Fixed — alias removed                                                          | cascade_correlation.py                                                            |
| INT-P2-001 | Undeclared `global shared_object_dict`                    | ✅ Resolved — no longer exists in codebase                                        | cascade_correlation.py                                                            |
| INT-P2-003 | `validate_training_results` uninitialized variable        | ✅ Resolved — fallback construction at line 3790                                  | cascade_correlation.py:3668-3790                                                  |
| INT-P2-007 | conftest fast-slow mode logic inverted                    | ✅ Resolved — logic cleaned up                                                    | src/tests/conftest.py                                                             |
| INT-P2-008 | `_roll_sequence_number` memory issue                      | ✅ Resolved — loop + MAX_ROLL_COUNT cap                                           | cascade_correlation.py:1083, candidate_unit.py:367                                |
| INT-P2-010 | `os._exit()` used instead of `sys.exit()`                 | ✅ Resolved — removed from main.py                                                | src/main.py                                                                       |
| INT-P1-002 | `requests` as undeclared dependency                       | ✅ Resolved — transitive via juniper-data-client                                  | pyproject.toml                                                                    |
| INT-P1-007 | No connection retry logic for JuniperData                 | ✅ Resolved via CAS-INT-008                                                       | JuniperDataClient._request() with MAX_RETRIES=3                                   |
| INT-P2-013 | `check_object_pickleability` depends on undeclared `dill` | ✅ Resolved — `dill>=0.3.6` in pyproject.toml                                     | pyproject.toml                                                                    |
| BUG-001    | Test random state restoration failures                    | ✅ Fixed — module-specific RNG calls                                              | src/tests/integration/test_serialization.py                                       |
| BUG-002    | Logger pickling error in multiprocessing                  | ✅ Fixed — `__getstate__`/`__setstate__` implemented                              | cascade_correlation.py, candidate_unit.py                                         |
| —          | Convergence threshold not triggering                      | ✅ Fixed — `convergence_threshold` (default 0.001)                                | cascade_correlation.py:4431,4564; candidate_unit.py:601                           |
| —          | Leaf tensor autograd RuntimeError                         | ✅ Fixed — create without grad, mutate, then `requires_grad_(True)`               | cascade_correlation.py: add_unit(), add_units_as_layer()                          |
| P4-NEW-006 | Module naming collision                                   | ✅ Fixed — `constants/` → `cascor_constants/`                                     | 9 files updated                                                                   |
| INT-P1-001 | Duplicated JuniperDataClient                              | ✅ Resolved by polyrepo Phase 1                                                   | juniper-data-client v0.3.0 on PyPI                                                |
| INT-P1-004 | No IPC architecture                                       | ✅ Substantially resolved by Phases 2-3                                           | CasCor Service API + juniper-cascor-client                                        |
| C.1        | Async wrapper for synchronous fit()                       | ✅ Resolved by Phase 2                                                            | TrainingLifecycleManager with ThreadPoolExecutor                                  |
| C.2        | Expose RemoteWorkerClient                                 | ✅ Resolved by Phase 3                                                            | juniper-cascor-worker v0.1.0 on PyPI                                              |
| CAS-004    | Extract remote worker to separate package                 | ✅ Resolved by Phase 3                                                            | juniper-cascor-worker published                                                   |

---

### 1.3 Performance Optimizations (All Verified Implemented)

| ID    | Optimization                      | Measured Impact                             | Implementation                                                                                |
|-------|-----------------------------------|---------------------------------------------|-----------------------------------------------------------------------------------------------|
| RC-1  | PyTorch thread pinning            | 5-15x throughput improvement                | `torch.set_num_threads(1)` in `_worker_loop()` + OMP/MKL/OPENBLAS env vars in main.py         |
| RC-2  | Direct multiprocessing.Queue      | 3-10x additional improvement                | Replaced BaseManager-proxied queues with `mp_ctx.Queue()`                                     |
| RC-3  | Shared training data              | Reduces N-fold tensor serialization         | `_worker_loop()` accepts `shared_training_inputs`; lightweight task reconstruction            |
| RC-4  | Persistent worker pool            | 20-50% latency reduction per round          | `_ensure_worker_pool()`, `_shutdown_worker_pool()` with sentinel/terminate/SIGKILL escalation |
| OPT-1 | Pre-allocated forward buffer      | Eliminates N+1 torch.cat per forward        | `_compute_hidden_outputs()` pre-allocates buffer, fills incrementally                         |
| OPT-2 | Batch correlation computation     | 5-10% per correlation                       | `torch.dot()` + `torch.linalg.norm()` replacing multi-kernel patterns                         |
| OPT-4 | Cached forward pass               | 22x-1607x on isolated call; 5-15% total     | Cache in `forward()` keyed by `data_ptr()`, consumed by `_prepare_candidate_input()`          |
| OPT-5 | SharedMemory training tensors     | 5-20% total round time reduction            | `SharedTrainingMemory` class with POSIX shared memory blocks, full lifecycle management       |
| OPT-6 | Single-output correlation fix     | 37x speedup (18.24ms → 0.49ms)              | Removed tensor value formatting from 15 hot-path log calls                                    |
| —     | Convergence threshold in patience | Training stops when improvement < threshold | Both output training and candidate training patience checks                                   |

**Performance Benchmark Data** (collected 2026-03-31):

- Forward pass: Sub-linear scaling (0→50 hidden units = 1.86x)
- Autograd overhead: 6-8.5x pure forward computation
- Candidate training: Linear epoch scaling, sub-linear sample/input scaling
- Output training: Sub-linear hidden unit scaling (0→50 = 1.60x)
- Activation functions: < 10% difference between tanh/sigmoid/relu

---

### 1.4 Code Quality & Compliance

| Item                   | Status                    | Details                                                                             |
|------------------------|---------------------------|-------------------------------------------------------------------------------------|
| Pre-commit compliance  | ✅ All 20 hooks pass      | 9 violations fixed: F401 (×2), F402, C401, B007, B404 config + B105/B110/B107 nosec |
| Lint compliance tests  | ✅ 162 parametrized tests | `test_lint_compliance.py` for future detection                                      |
| Coverage threshold     | ✅ Configured             | `fail_under = 80` in pyproject.toml                                                 |
| CI pipeline            | ✅ Green                  | Pre-commit, unit tests (3.11/3.12/3.13), security, integration, build, quality gate |
| CPU-only PyTorch in CI | ✅ Configured             | `--index-url https://download.pytorch.org/whl/cpu`                                  |

---

### 1.5 Testing & Profiling Infrastructure

| Item                           | Status         | Details                                                                                                          |
|--------------------------------|----------------|------------------------------------------------------------------------------------------------------------------|
| cProfile integration           | ✅             | `src/profiling/deterministic.py` — ProfileContext, profile_function decorator                                    |
| Memory profiling               | ✅             | `src/profiling/memory.py` — MemoryTracker, `--profile-memory` CLI flag                                           |
| py-spy integration             | ✅             | `util/profile_training.bash` — SVG flame graphs, Speedscope JSON                                                 |
| Hot-path logging utilities     | ✅             | SampledLogger, BatchLogger, log_if_enabled, LogFrequencyTracker                                                  |
| Performance micro-benchmarks   | ✅             | Forward pass, autograd, correlation, candidate training, output training, concurrency, shared memory, end-to-end |
| Benchmark harness              | ✅             | `src/tests/scripts/run_benchmarks.bash` + pytest-benchmark integration                                           |
| JuniperData integration        | ✅ All 9 items | CAS-INT-001-009: API path, deprecation warnings, auth, NPZ validation, contract tests, retry/backoff             |
| Async training boundary        | ✅             | ThreadPoolExecutor, `fit_async()`, `start_training_background()`                                                 |
| RemoteWorkerClient integration | ✅             | REST endpoints for remote worker management                                                                      |
| Test suite CI/CD phases 0-4    | ✅             | MED-014 (line length) deferred                                                                                   |

---

### 1.6 Feature Implementations

| Feature                                    | Status | Key Files                                                                                                |
|--------------------------------------------|--------|----------------------------------------------------------------------------------------------------------|
| CasCor Service API                         | ✅     | `src/api/app.py`, `src/server.py`, 8 route modules                                                       |
| Training lifecycle management              | ✅     | `src/api/lifecycle/manager.py`, `monitor.py`, `state_machine.py`                                         |
| API security (auth + rate limiting)        | ✅     | `src/api/security.py`, `src/api/middleware.py`                                                           |
| WebSocket channels                         | ✅     | Control, training metrics, worker protocol streams                                                       |
| Remote worker system                       | ✅     | Registry, coordinator, binary protocol, security, audit                                                  |
| Decision boundary visualization            | ✅     | `src/api/routes/decision_boundary.py`                                                                    |
| Snapshot management                        | ✅     | `src/api/routes/snapshots.py`                                                                            |
| Convergence threshold (runtime-updateable) | ✅     | `PATCH /v1/training/params` — convergence_threshold, candidate_patience, candidate_convergence_threshold |
| Output weight initialization option        | ✅     | `init_output_weights: "zero"/"random"` in `add_unit()`, `add_units_as_layer()`                           |
| Pydantic-based configuration               | ✅     | `src/api/settings.py` with BaseSettings                                                                  |
| Constants API defaults                     | ✅     | `src/cascor_constants/constants_api/constants_api_defaults.py` — 49 constants                            |

---

## 2. Open / Remaining Work (Confirmed by Codebase Validation)

### 2.1 Active Issues (Confirmed Still Present)

| ID             | Description                                             | Severity | File(s)                                                                                  | Recommended Action                                         |
|----------------|---------------------------------------------------------|----------|------------------------------------------------------------------------------------------|------------------------------------------------------------|
| INT-P0-004     | `remote_client_0.py` has hardcoded path to old monorepo | Low      | `src/remote_client/remote_client_0.py`                                                   | Delete file — superseded by juniper-cascor-worker          |
| INT-P0-005     | Hardcoded paths in test file                            | Low      | `src/tests/unit/test_candidate_training_manager.py:10-12`                                | Remove stale `sys.path.append` lines                       |
| INT-P1-008     | `check.py` is stale duplicate of `spiral_problem.py`    | Medium   | `src/spiral_problem/check.py`                                                            | Delete file                                                |
| INT-P2-005/006 | `or` fallback bugs for falsy values                     | Medium   | `spiral_problem.py:604,1256,1415` (clockwise); `cascade_correlation.py` (numeric params) | Replace with `if x is not None` pattern                    |
| INT-P2-014     | Local `import traceback` inside exception handlers      | Low      | `cascade_correlation.py` (9 instances)                                                   | Uncomment top-level import (line 64), remove local imports |
| INT-P3-009     | Version strings inconsistent across file headers        | Low      | main.py says 0.3.1, cascade_correlation.py says 0.3.2; pyproject.toml says 0.4.0         | Update all file headers to match pyproject.toml            |
| —              | Legacy `remote_client/` directory still exists          | Low      | `src/remote_client/` (remote_client.py + remote_client_0.py)                             | Remove or archive — superseded by juniper-cascor-worker    |

**Estimated effort for all active issues**: 2-4 hours total

### 2.2 Not Started / Needs Work

| ID            | Description                                                 | Priority | Effort | Source                      |
|---------------|-------------------------------------------------------------|----------|--------|-----------------------------|
| CAS-REF-002   | CI/CD coverage gates enforcement (per-module thresholds)    | P2       | S      | PRE-DEPLOYMENT_ROADMAP-2.md |
| CAS-REF-003   | Fix critical type errors (mypy strict mode)                 | P2       | M      | Post-release roadmap        |
| CAS-007       | Optimize slow tests (target ≤ 5 min suite)                  | P2       | M      | PRE-DEPLOYMENT_ROADMAP-2.md |
| CAS-REF-004   | Legacy spiral code removal (16 deprecated methods)          | P2       | M      | Post-release roadmap        |
| INT-P3-003    | Docker Compose end-to-end validation                        | P3       | S      | Post-release roadmap        |
| INT-P3-008    | .pytest.ini.swp and coverage files in .gitignore            | P3       | S      | Post-release roadmap        |
| INT-P3-010    | `cascor_snapshots/` vs `snapshots/` directory confusion     | P3       | S      | Post-release roadmap        |
| Shell scripts | Oracle Scripts Analysis items 1-6 (path resolution, naming) | P3       | M      | ORACLE_ANALYSIS_SCRIPTS.md  |
| OPT-3         | Persistent output layer (nn.Linear reuse)                   | P3       | M      | PERFORMANCE_TESTING_PLAN.md |
| INT-P3-005    | Test WebSocket responsiveness during training               | P3       | M      | Post-release roadmap        |
| INT-P3-006    | Baseline performance profiles for regression detection      | P3       | M      | Post-release roadmap        |
| INT-P3-007    | Profiling in CI/CD pipeline                                 | P3       | L      | Post-release roadmap        |

### 2.3 Oracle Analysis Recommendations (Unvalidated)

#### Python Analysis (5 Items)

| # | Description                                                                     | Status              |
|---|---------------------------------------------------------------------------------|---------------------|
| 1 | `CandidateUnit.train()` return type mismatch; needs `train_detailed()` refactor | ❓ NEEDS VALIDATION |
| 2 | `CandidateTrainingManager.start()` does not accept `method` parameter           | ❓ NEEDS VALIDATION |
| 3 | `ValidationError` not a subclass of `ValueError`                                | ❓ NEEDS VALIDATION |
| 4 | Residual error validation rejects empty tensors                                 | ❓ NEEDS VALIDATION |
| 5 | `fit()` method uses `max_epochs` but tests call with `epochs`                   | ❓ NEEDS VALIDATION |

#### Scripts Analysis (6 Items)

| # | Description                                                    | Status         |
|---|----------------------------------------------------------------|----------------|
| 1 | Fix helper script resolution in `juniper_cascor.bash`          | 🔴 NOT STARTED |
| 2 | Verify helper filenames match actual files in `util/`          | 🔴 NOT STARTED |
| 3 | Confirm BASE_DIR and SOURCE_DIR resolve correctly              | 🔴 NOT STARTED |
| 4 | Fix CURRENT_OS and date helper sourcing; verify permissions    | 🔴 NOT STARTED |
| 5 | Fix naming inconsistencies in `script_util.cfg`                | 🔴 NOT STARTED |
| 6 | Grep for `GET_PROJECT_SCRIPT` pattern and fix in other scripts | 🔴 NOT STARTED |

---

## 3. Planned Enhancements (Future Work)

### 3.1 Training Control

| ID      | Description                                                               | Status         |
|---------|---------------------------------------------------------------------------|----------------|
| CAS-002 | Separate epoch limits for full network and candidate nodes                | 🔴 NOT STARTED |
| CAS-003 | Max train session iterations meta parameter                               | 🔴 NOT STARTED |
| CAS-006 | Auto-snap best network when new best accuracy achieved (accuracy ratchet) | 🔴 NOT STARTED |

### 3.2 Algorithm Enhancements

| ID      | Description                                                            | Status         |
|---------|------------------------------------------------------------------------|----------------|
| ENH-006 | Flexible optimizer system (OptimizerConfig: Adam, SGD, RMSprop, AdamW) | 🔴 NOT STARTED |
| ENH-007 | N-best candidate layer selection (config has placeholders)             | 🔴 NOT STARTED |

### 3.3 Network Architecture

| ID      | Description                                              | Status         |
|---------|----------------------------------------------------------|----------------|
| CAS-008 | Network hierarchy management (multi-hierarchical CasCor) | 🔴 NOT STARTED |
| CAS-009 | Network population management (ensemble approaches)      | 🔴 NOT STARTED |

### 3.4 Storage & Serialization

| ID      | Description                                               | Status         |
|---------|-----------------------------------------------------------|----------------|
| CAS-010 | Snapshot vector DB storage (indexed by UUID)              | 🔴 NOT STARTED |
| ENH-011 | Backward compatibility testing for older snapshot formats | 🔴 NOT STARTED |

### 3.5 Infrastructure

| ID         | Description                                   | Status         | Effort         |
|------------|-----------------------------------------------|----------------|----------------|
| P3-NEW-003 | GPU/CUDA support for training                 | 🔴 NOT STARTED | XL (2-4 weeks) |
| P3-NEW-004 | Continuous profiling with Grafana Pyroscope   | 🔵 DEFERRED    | L              |
| —          | Large file refactoring (no file > 2000 lines) | 🔴 NOT STARTED | L              |
| —          | Auto-generated API docs (MkDocs/Sphinx)       | 🔴 NOT STARTED | M              |
| —          | Documentation link checking in CI             | 🔴 NOT STARTED | S              |
| —          | Documentation search functionality            | 🔴 NOT STARTED | M              |

### 3.6 Code Cleanup (Deferred)

| ID            | Description                                                                  | Status      |
|---------------|------------------------------------------------------------------------------|-------------|
| CASCOR-P1-008 | Remove "Roll" concept in CandidateUnit                                       | 🔵 DEFERRED |
| P3-001        | Candidate factory refactor (all creation through `_create_candidate_unit()`) | 🔵 DEFERRED |
| MED-014       | Line length reduction to 120 characters                                      | 🔵 DEFERRED |
| INT-P4-012    | `LogConfig.__init__` parameter naming cleanup                                | 🔵 DEFERRED |
| INT-P4-013    | Logger TODO cleanup                                                          | 🔵 DEFERRED |
| INT-P4-014    | Remove commented-out code blocks                                             | 🔵 DEFERRED |
| INT-P4-015    | Clean up "Original corrupted line" comments in spiral_problem.py             | 🔵 DEFERRED |
| INT-P4-016    | Remove `uuid as uuid` redundant import alias                                 | 🔵 DEFERRED |
| ENH-009       | Per-instance queue management                                                | 🔵 DEFERRED |
| ENH-010       | Process-based plotting                                                       | 🔵 DEFERRED |
| INT-P4-010    | Add metrics for multiprocessing fallback frequency                           | 🔵 DEFERRED |
| INT-P4-011    | Test fallback under various failure modes                                    | 🔵 DEFERRED |

### 3.7 Canopy Enhancements (CAN-000 through CAN-021)

| ID       | Module            | Description                                                                |
|----------|-------------------|----------------------------------------------------------------------------|
| CAN-000  | Meta Param Menu   | Periodic updates pause when Apply Parameters button active                 |
| CAN-001  | Training Metrics  | Training Loss time window toggle/dropdown                                  |
| CAN-002  | Training Metrics  | Custom rolling time window for Training Loss graph                         |
| CAN-003  | Training Metrics  | Retain candidate pool data per node addition; expandable "Previous Pools"  |
| CAN-004  | Meta Param Tuning | New Tab for all exposed meta parameters                                    |
| CAN-005  | Meta Param Tuning | Pin/Unpin meta params from Tuning Tab to left side menu                    |
| CAN-006  | Meta Param Tuning | Network train epoch count parameter                                        |
| CAN-007  | Meta Param Tuning | Candidate pool training epoch count parameter                              |
| CAN-008  | Meta Param Tuning | Candidate pool node count parameter                                        |
| CAN-009  | Meta Param Tuning | Correlation threshold parameter                                            |
| CAN-010  | Meta Param Tuning | Optimizer type meta parameter                                              |
| CAN-011  | Meta Param Tuning | Activation function meta parameter                                         |
| CAN-012  | Meta Param Tuning | Number of top candidate nodes to select                                    |
| CAN-013  | Meta Param Tuning | Candidate node integration mode (Input Only, All Hidden, etc.)             |
| CAN-014  | Training Metrics  | Snapshot captures tuning values throughout training session                |
| CAN-015  | Training Metrics  | Snapshot replay with live tuning → new training session                    |
| CAN-016a | All               | Save/Load dashboard layout state                                           |
| CAN-016b | Dataset           | Import/Generate new dataset (local file, remote URL, Juniper Data REST)    |
| CAN-017  | All               | Tooltips on all dashboard controls                                         |
| CAN-018  | All               | Right-click tutorial descriptions with doc links                           |
| CAN-019  | All               | Walk-through style tutorial with highlighted steps                         |
| CAN-020  | All               | Show network at specific hierarchy level (dropdown/slider with thumbnails) |
| CAN-021  | All               | Show network in population (dropdown/slider for population > 1)            |

### 3.8 Cascor Feature Enhancements (CAS-001 through CAS-010)

| ID      | Description                               | Status                              |
|---------|-------------------------------------------|-------------------------------------|
| CAS-001 | Extract spiral generator to JuniperData   | ✅ COMPLETE                         |
| CAS-002 | Separate epoch limits                     | 🔴 NOT STARTED                      |
| CAS-003 | Max train session iterations              | 🔴 NOT STARTED                      |
| CAS-004 | Extract remote worker code                | ✅ COMPLETE (juniper-cascor-worker) |
| CAS-005 | Extract common dependencies to modules    | 🔴 NOT STARTED                      |
| CAS-006 | Auto-snap best network (accuracy ratchet) | 🔴 NOT STARTED                      |
| CAS-007 | Optimize slow tests (≤ 5 min)             | 🔴 NOT STARTED                      |
| CAS-008 | Network hierarchy management              | 🔴 NOT STARTED                      |
| CAS-009 | Network population management             | 🔴 NOT STARTED                      |
| CAS-010 | Snapshot vector DB storage                | 🔴 NOT STARTED                      |

---

## 4. Design Decisions Record

| #  | Decision                                     | Status                | Chosen Option                                                               | Source               |
|----|----------------------------------------------|-----------------------|-----------------------------------------------------------------------------|----------------------|
| 1  | ActivationWithDerivative extraction location | ✅ IMPLEMENTED        | Option A: New shared module (`src/utils/activation.py`)                     | Post-release roadmap |
| 2  | CandidateUnit constructor fix approach       | ✅ IMPLEMENTED        | Option A + C: Fix factory pattern + remove `**kwargs`                       | Post-release roadmap |
| 3  | `or` fallback pattern systematic fix         | DECIDED               | Option A: `if x is not None` pattern                                        | Post-release roadmap |
| 4  | Shared client package architecture           | ✅ IMPLEMENTED        | Option A: PyPI packages (juniper-data-client, cascor-client, cascor-worker) | Post-release roadmap |
| 5  | Async training wrapper approach              | ✅ IMPLEMENTED        | Option A: `loop.run_in_executor()` with ThreadPoolExecutor                  | Post-release roadmap |
| 6  | Large file refactoring strategy              | DECIDED (not started) | Option A Phase 1: Mixin-based decomposition                                 | Post-release roadmap |
| 7  | Legacy spiral code removal                   | DECIDED (gated)       | Option A: Hard removal after E2E test gate                                  | Post-release roadmap |
| 8  | Optimizer state serialization                | DECIDED               | Remove for MVP; can be added later                                          | Enhancements roadmap |
| 9  | Multiprocessing state restoration            | 🔵 DEFERRED           | Partial restore: save config, don't auto-restart                            | Enhancements roadmap |
| 10 | Shared memory approach for training tensors  | ✅ IMPLEMENTED        | Approach C: Named SharedMemory with lightweight tasks                       | OPT-5 plan           |

---

## 5. Verification Items

| Item       | Status           | Description                                                                                         |
|------------|------------------|-----------------------------------------------------------------------------------------------------|
| P4-NEW-001 | ❓ NOT VERIFIED  | Execute main.py end-to-end with plotting                                                            |
| P4-NEW-002 | ❓ NOT VERIFIED  | Test ./try script launch                                                                            |
| P4-NEW-005 | ❓ NEEDS TESTING | Verify parallel processing working (DEBUG logging)                                                  |
| —          | ❓ NOT VERIFIED  | CasCor Service API end-to-end (network creation, training, metrics WS, topology, decision boundary) |

---

## 6. Source Document Cross-Reference

| Document                                                        | Date       | Key Content                                   | Disposition                                            |
|-----------------------------------------------------------------|------------|-----------------------------------------------|--------------------------------------------------------|
| CASCOR_ENHANCEMENTS_ROADMAP.md                                  | 2025-10-28 | BUG-001/002, ENH-001-011, design decisions    | Phase 1 Complete; items consolidated here              |
| CONVERGENCE_THRESHOLD_ANALYSIS_2026-04-03.md                    | 2026-04-03 | Patience convergence fix analysis             | ✅ Implemented; consolidated here                      |
| JUNIPER_MICROSERVICES-ARCHITECTURE_DEVELOPMENT-ROADMAP_AUDIT.md | 2026-03    | Phases 1-9 cross-repo audit                   | All Complete; consolidated here                        |
| JUNIPER-CASCOR_POST-RELEASE_DEVELOPMENT-ROADMAP_2026-02-24.md   | 2026-02-24 | 89 items, pre-polyrepo audit                  | Superseded by post-polyrepo version                    |
| JUNIPER-CASCOR_POST-RELEASE_DEVELOPMENT-ROADMAP.md              | 2026-02-24 | 83 items, post-polyrepo reconciled            | Authoritative roadmap; now consolidated here           |
| LEAF_TENSOR_AUTOGRAD_FIX_PLAN.md                                | 2026-04-05 | Autograd fix for add_unit/add_units_as_layer  | ✅ Implemented; consolidated here                      |
| OPT5_SHARED_MEMORY_PLAN.md                                      | 2026-04-01 | SharedMemory for training tensors             | ✅ Implemented; consolidated here                      |
| PARALLEL_CANDIDATE_TRAINING_FIX_PLAN.md                         | 2026-03-17 | RC-1 through RC-4 parallel fixes              | All 3 Phases Complete; consolidated here               |
| PERFORMANCE_TESTING_PLAN.md                                     | 2026-03-31 | 5-phase profiling plan, OPT-1/2/4/5/6         | OPT-1/2/4/5/6 Done; Phases 3-5 data collection pending |
| POLYREPO_MIGRATION_PLAN.md                                      | 2026-03-02 | 7-phase polyrepo migration                    | All Phases Complete; consolidated here                 |
| PRE-COMMIT_COMPLIANCE_REMEDIATION_PLAN.md                       | 2026-03-21 | 9 lint violations remediation                 | ✅ All Fixed; consolidated here                        |
| PRE-DEPLOYMENT_ROADMAP-2.md                                     | 2026-01-25 | 19 pre-deployment tasks, CAS/CAN enhancements | 14/19 Complete; remaining items consolidated here      |

---

## Document History

| Date       | Author                   | Changes                                                                                  |
|------------|--------------------------|------------------------------------------------------------------------------------------|
| 2026-04-17 | Automated Analysis (Amp) | Initial creation: consolidated 12 source documents, validated all items against codebase |
