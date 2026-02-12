# Juniper Integration Development Plan

**Project**: Juniper Cascade Correlation Neural Network
**Document Version**: 1.0.0
**Created**: 2026-02-05
**Last Updated**: 2026-02-05
**Authors**: Claude Opus 4.5 (AI-assisted analysis)
**Status**: Active - Initial Compilation
**Branch**: `subproject.juniper_cascor.feature.frontend_integration.pre_deploy.spiral_gen_extract`

---

## Executive Summary

This document consolidates **all outstanding work** required to complete the integration of three Juniper applications:

- **JuniperCascor** - Backend neural network training engine (PyTorch)
- **JuniperData** - Dataset generation REST API microservice (FastAPI + NumPy)
- **JuniperCanopy** - Frontend monitoring dashboard (FastAPI + Dash + WebSocket)

Sources include four existing roadmap documents from the JuniperData `notes/` directory, two independent test suite audits, and a rigorous source code review performed on 2026-02-05.

### Issue Summary

| Source                               | Critical | High   | Medium | Low    | Total  |
| ------------------------------------ | -------- | ------ | ------ | ------ | ------ |
| Existing Documentation (Outstanding) | 2        | 8      | 12     | 10     | 32     |
| Source Code Review (New)             | 5        | 12     | 16     | 14     | 47     |
| **Combined (De-duplicated)**         | **5**    | **16** | **22** | **18** | **61** |

### Phase Summary

| Phase     | Description                   | Issues | Effort            | Priority       |
| --------- | ----------------------------- | ------ | ----------------- | -------------- |
| Phase 0   | Critical Bugs & Blockers      | 5      | 8-16 hours        | P0 - Immediate |
| Phase 1   | Integration Architecture      | 8      | 24-40 hours       | P1 - High      |
| Phase 2   | Code Quality & Test Integrity | 14     | 32-48 hours       | P2 - Medium    |
| Phase 3   | Infrastructure & CI/CD        | 10     | 20-32 hours       | P3 - Standard  |
| Phase 4   | Enhancements & Future Work    | 24     | 40-80 hours       | P4 - Deferred  |
| **Total** |                               | **61** | **124-216 hours** |                |

---

## Table of Contents

1. [Source Documents](#1-source-documents)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 0: Critical Bugs & Blockers](#3-phase-0-critical-bugs--blockers)
4. [Phase 1: Integration Architecture](#4-phase-1-integration-architecture)
5. [Phase 2: Code Quality & Test Integrity](#5-phase-2-code-quality--test-integrity)
6. [Phase 3: Infrastructure & CI/CD](#6-phase-3-infrastructure--cicd)
7. [Phase 4: Enhancements & Future Work](#7-phase-4-enhancements--future-work)
8. [Dependencies Matrix](#8-dependencies-matrix)
9. [Risk Assessment](#9-risk-assessment)
10. [Appendix: Source Code Review Details](#10-appendix-source-code-review-details)

---

## 1. Source Documents

### 1.1 Evaluated Documentation

| Document                                        | Location             | Version | Status                                |
| ----------------------------------------------- | -------------------- | ------- | ------------------------------------- |
| JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md | JuniperData `notes/` | 1.7.0   | Phases 0-4 Complete, Phase 5 Deferred |
| INTEGRATION_ROADMAP.md                          | JuniperData `notes/` | 2.1.0   | Active - Most issues resolved         |
| PRE-DEPLOYMENT_ROADMAP-2.md                     | JuniperData `notes/` | 2.3.0   | Active - 14/19 tasks complete (74%)   |
| PRE-DEPLOYMENT_ROADMAP.md                       | JuniperData `notes/` | 1.4.0   | Superseded by Roadmap-2               |

### 1.2 Additional References

| Document                                        | Location               | Purpose                                                      |
| ----------------------------------------------- | ---------------------- | ------------------------------------------------------------ |
| TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md | JuniperCascor `notes/` | Test/CI improvements (Phases 0-4 complete, MED-014 deferred) |
| TEST_SUITE_AUDIT_CASCOR_AMP.md                  | JuniperCascor `notes/` | Independent test audit (Amp)                                 |
| TEST_SUITE_AUDIT_CASCOR_CLAUDE.md               | JuniperCascor `notes/` | Independent test audit (Claude)                              |
| INTEGRATION_ROADMAP.md                          | JuniperCascor `notes/` | Cascor-Canopy integration tracker                            |

---

## 2. Architecture Overview

```bash
                    JUNIPER_DATA_URL
    +----------------------------------------------+
    |                                              |
    v                                              v
+------------------+    REST API   +-------------------+
| JuniperData      |<--------------| JuniperCascor     |
| (Port 8100)      |    /v1/...    | (Training Engine) |
|                  |               |                   |
| FastAPI + NumPy  |    NPZ files  | PyTorch + NumPy   |
| SpiralGenerator  |-------------->| CCN, Candidates   |
| Dataset Storage  |               | HDF5 Snapshots    |
+------------------+               +-------------------+
    ^                                    |
    |  JUNIPER_DATA_URL                  | CASCOR_BACKEND_PATH
    |                                    | (sys.path import)
    v                                    v
+---------------------------------------------------+
|              JuniperCanopy (Port 8050)            |
|                                                   |
| FastAPI + Dash + WebSocket                        |
| CascorIntegration (direct import from Cascor)     |
| DataAdapter (metric normalization)                |
| DemoMode (standalone demo)                        |
| WebSocket Manager (real-time broadcast)           |
+---------------------------------------------------+
```

### Integration Points (All Implemented)

| Interface          | Cascor Export                   | Canopy Import            | Mechanism                        | Status   |
| ------------------ | ------------------------------- | ------------------------ | -------------------------------- | -------- |
| Dataset Generation | Legacy local / JuniperData REST | SpiralDataProvider       | `JUNIPER_DATA_URL` env var       | Complete |
| Network Topology   | HDF5 snapshots                  | CascorIntegration        | `CASCOR_BACKEND_PATH` + sys.path | Complete |
| Training Metrics   | Real-time state                 | Data Adapter + WebSocket | normalize_metrics()              | Complete |
| Control Commands   | Multiprocess queues             | REST/WebSocket           | FastAPI endpoints                | Complete |

---

## 3. Phase 0: Critical Bugs & Blockers

**Priority**: P0 - Immediate
**Effort**: 8-16 hours
**Rationale**: These issues produce incorrect behavior, can crash the application, or block further development.

### INT-P0-001: Walrus Operator Precedence Bug in `train_output_layer`

| Field        | Value                                                 |
| ------------ | ----------------------------------------------------- |
| **Source**   | Source Code Review (NEW)                              |
| **File**     | `src/cascade_correlation/cascade_correlation.py:1322` |
| **Severity** | Critical                                              |
| **Type**     | Bug                                                   |
| **Blocks**   | Training workflow, snapshot creation                  |

**Description**: The line `if snapshot_path := self.create_snapshot() is not None:` is parsed as `snapshot_path := (self.create_snapshot() is not None)` due to Python operator precedence. `snapshot_path` is always assigned `True` or `False` (a boolean), never the actual snapshot path. The subsequent log message `f"Created network snapshot at: {snapshot_path}"` always prints `True`.

**Fix**: Add parentheses: `if (snapshot_path := self.create_snapshot()) is not None:`

---

### INT-P0-002: `ActivationWithDerivative` Class Duplicated in Two Files

| Field        | Value                                                                                            |
| ------------ | ------------------------------------------------------------------------------------------------ |
| **Source**   | Source Code Review (NEW)                                                                         |
| **Files**    | `src/cascade_correlation/cascade_correlation.py:291`, `src/candidate_unit/candidate_unit.py:138` |
| **Severity** | Critical                                                                                         |
| **Type**     | Bug / Refactor                                                                                   |
| **Blocks**   | Maintenance, multiprocessing correctness                                                         |

**Description**: The `ActivationWithDerivative` class is defined identically in both files. The `__setstate__` method in each copy reconstructs from its own `ACTIVATION_MAP`. If the maps diverge, deserialized objects will behave differently depending on which module unpickled them.

**Fix**: Extract `ActivationWithDerivative` to a shared module (e.g., `src/utils/activation.py`) and import from both files.

---

### INT-P0-003: Invalid `CandidateUnit` Constructor Parameters in `fit` Method

| Field        | Value                                                      |
| ------------ | ---------------------------------------------------------- |
| **Source**   | Source Code Review (NEW)                                   |
| **File**     | `src/cascade_correlation/cascade_correlation.py:1154-1166` |
| **Severity** | Critical                                                   |
| **Type**     | Bug                                                        |
| **Blocks**   | Candidate training configuration                           |

**Description**: The `fit` method passes several parameters to `CandidateUnit` that do not exist in its constructor signature: `_CandidateUnit__candidate_pool_size`, `_CandidateUnit__log_file_name`, `_CandidateUnit__log_file_path`, `_CandidateUnit__epochs_max`, `_CandidateUnit__output_size`. These are silently absorbed by `**kwargs` without being applied.

**Fix**: Remove invalid parameters or add them to `CandidateUnit.__init__`.

---

### INT-P0-004: Hardcoded Path in `remote_client_0.py`

| Field        | Value                                     |
| ------------ | ----------------------------------------- |
| **Source**   | Source Code Review (NEW)                  |
| **File**     | `src/remote_client/remote_client_0.py:16` |
| **Severity** | Critical                                  |
| **Type**     | Bug / Integration                         |
| **Blocks**   | Remote worker deployment, portability     |

**Description**: Contains hardcoded absolute path: `sys.path.append("/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src")` pointing to the old prototype location. This file will fail on any other machine.

**Fix**: Replace with relative path resolution or remove `remote_client_0.py` (it uses incompatible queue names from the current manager).

---

### INT-P0-005: Hardcoded Paths in Test File

| Field        | Value                                                     |
| ------------ | --------------------------------------------------------- |
| **Source**   | Source Code Review (NEW)                                  |
| **File**     | `src/tests/unit/test_candidate_training_manager.py:10,12` |
| **Severity** | Critical                                                  |
| **Type**     | Bug                                                       |
| **Blocks**   | Test portability                                          |

**Description**: Contains platform-specific hardcoded paths to old prototype directories. Will cause import failures on other machines and raises `EnvironmentError` on Windows.

**Fix**: Replace with dynamic path resolution using `os.path.dirname(__file__)`.

---

## 4. Phase 1: Integration Architecture

**Priority**: P1 - High
**Effort**: 24-40 hours
**Rationale**: These issues directly affect the integration between the three applications.

### INT-P1-001: Duplicated `JuniperDataClient` in Cascor and Canopy

| Field        | Value                                                                          |
| ------------ | ------------------------------------------------------------------------------ |
| **Source**   | Existing Documentation (SPIRAL_DATA_GEN_REFACTOR_PLAN, Remaining Work Item #2) |
| **Severity** | High                                                                           |
| **Type**     | Integration / Refactor                                                         |
| **Blocks**   | Client feature parity, maintenance                                             |

**Description**: Both Cascor and Canopy maintain separate copies of `JuniperDataClient`. Canopy's version has an additional `get_preview()` method not present in Cascor's copy. Changes to one are not reflected in the other.

**Fix**: Consolidate into a shared `juniper_data_client` package installable by both applications.

---

### INT-P1-002: `requests` as Undeclared Dependency

| Field        | Value                                     |
| ------------ | ----------------------------------------- |
| **Source**   | Source Code Review (NEW)                  |
| **File**     | `src/juniper_data_client/client.py`       |
| **Severity** | High                                      |
| **Type**     | Integration                               |
| **Blocks**   | JuniperData integration on fresh installs |

**Description**: The `JuniperDataClient` imports `requests`, but this library is not listed in CLAUDE.md as a dependency or in `pyproject.toml`.

**Fix**: Add `requests` to `pyproject.toml` dependencies and document in CLAUDE.md.

---

### INT-P1-003: No Shared Protocol/Interface Package

| Field        | Value                      |
| ------------ | -------------------------- |
| **Source**   | Integration Analysis (NEW) |
| **Severity** | High                       |
| **Type**     | Integration / Architecture |
| **Blocks**   | API contract enforcement   |

**Description**: There is no shared Python package defining the data contracts (Pydantic models, type aliases) between the three applications. Each app independently defines its expectations. The NPZ array key names (`X_train`, `y_train`, etc.) are documented but not enforced by a shared schema.

**Fix**: Create a shared `juniper_contracts` or similar package defining API schemas.

---

### INT-P1-004: Full IPC Architecture (Deferred)

| Field        | Value                                                         |
| ------------ | ------------------------------------------------------------- |
| **Source**   | Existing Documentation (PRE-DEPLOYMENT_ROADMAP-2, P1-NEW-001) |
| **Severity** | High                                                          |
| **Type**     | Architecture                                                  |
| **Status**   | DEFERRED                                                      |
| **Blocks**   | Production deployment, scalability                            |

**Description**: Cascor is currently embedded in the Canopy process via `sys.path` import. A true IPC architecture (separate processes communicating via sockets/REST) is needed for production deployment.

**Sub-tasks**:

- [ ] Design IPC protocol specification
- [ ] Implement Cascor server mode
- [ ] Update Canopy to connect to external Cascor
- [ ] Add connection management and health checks

---

### INT-P1-005: `main.py` Passes Invalid Parameters to `SpiralProblem`

| Field        | Value                           |
| ------------ | ------------------------------- |
| **Source**   | Source Code Review (NEW)        |
| **File**     | `src/main.py:208-210`           |
| **Severity** | High                            |
| **Type**     | Bug / Integration               |
| **Blocks**   | Application startup correctness |

**Description**: `main.py` passes `_SpiralProblem__spiral_config=logging.config` (the logging module, not a config object), `_SpiralProblem__dataset_tensors=None`, and `_SpiralProblem__dataset_file_info=None` to the `SpiralProblem` constructor. These are absorbed by `**kwargs` silently.

**Fix**: Remove invalid parameters or implement the commented-out `SpiralConfig` class.

---

### INT-P1-006: Missing Import Guard for `SpiralDataProvider`

| Field        | Value                                             |
| ------------ | ------------------------------------------------- |
| **Source**   | Source Code Review (NEW)                          |
| **File**     | `src/spiral_problem/spiral_problem.py:505`        |
| **Severity** | High                                              |
| **Type**     | Integration                                       |
| **Blocks**   | Graceful degradation when JuniperData unavailable |

**Description**: The lazy import `from spiral_problem.data_provider import SpiralDataProvider` imports `requests` transitively. If `requests` is not installed, this fails with `ModuleNotFoundError` even when `JUNIPER_DATA_URL` is set.

**Fix**: Add try/except guard around the import with a clear error message.

---

### INT-P1-007: No Connection Retry Logic for JuniperData Client

| Field        | Value                               |
| ------------ | ----------------------------------- |
| **Source**   | Source Code Review (NEW)            |
| **File**     | `src/juniper_data_client/client.py` |
| **Severity** | High                                |
| **Type**     | Integration                         |
| **Blocks**   | Resilient integration               |

**Description**: The REST client has no retry logic, no connection pooling, and no circuit breaker. A transient network error during training would crash the entire pipeline.

**Fix**: Add retry with exponential backoff (e.g., `urllib3.Retry` or `tenacity`).

---

### INT-P1-008: `check.py` Is a Stale Duplicate of `spiral_problem.py`

| Field        | Value                         |
| ------------ | ----------------------------- |
| **Source**   | Source Code Review (NEW)      |
| **File**     | `src/spiral_problem/check.py` |
| **Severity** | High                          |
| **Type**     | Refactor / Integration        |
| **Blocks**   | Code clarity, maintenance     |

**Description**: `check.py` contains a complete but outdated copy of the `SpiralProblem` class using old-style constructor parameters. This is likely dead code that creates confusion.

**Fix**: Remove or archive `check.py`.

---

## 5. Phase 2: Code Quality & Test Integrity

**Priority**: P2 - Medium
**Effort**: 32-48 hours
**Rationale**: These issues affect code reliability, test accuracy, and maintainability.

### INT-P2-001: Undeclared Global Variable `shared_object_dict`

| Field        | Value                                                 |
| ------------ | ----------------------------------------------------- |
| **Source**   | Source Code Review (NEW)                              |
| **File**     | `src/cascade_correlation/cascade_correlation.py:2300` |
| **Severity** | High                                                  |
| **Type**     | Bug                                                   |

**Description**: `global shared_object_dict` is declared inside `_train_candidate_unit` but never defined anywhere. Accessing it would raise `NameError`. Remnant from earlier design.

**Fix**: Remove the `global` declaration and related dead code.

---

### INT-P2-002: `import datetime as pd` - Misleading Alias

| Field        | Value                                               |
| ------------ | --------------------------------------------------- |
| **Source**   | Source Code Review (NEW)                            |
| **File**     | `src/cascade_correlation/cascade_correlation.py:38` |
| **Severity** | High                                                |
| **Type**     | Bug / Refactor                                      |

**Description**: `import datetime as pd` aliases `datetime` as `pd` (universally associated with `pandas`). The `pd` alias is never used. Line 37 already has `import datetime`.

**Fix**: Remove the `import datetime as pd` line.

---

### INT-P2-003: `validate_training_results` Bug (Documented TODO)

| Field        | Value                                                 |
| ------------ | ----------------------------------------------------- |
| **Source**   | Source Code Review (NEW)                              |
| **File**     | `src/cascade_correlation/cascade_correlation.py:2750` |
| **Severity** | High                                                  |
| **Type**     | Bug                                                   |

**Description**: TODO comment: `validate_training_results bug: needs to be fixed`. The variable is initialized to `None` and only set inside the for loop. If `max_epochs=0`, the post-loop check references unbound `epoch` variable.

**Fix**: Initialize `epoch` and `validate_training_results` before the loop.

---

### INT-P2-004: `snapshot_counter` Initialized Twice

| Field        | Value                                                    |
| ------------ | -------------------------------------------------------- |
| **Source**   | Source Code Review (NEW)                                 |
| **File**     | `src/cascade_correlation/cascade_correlation.py:530,548` |
| **Severity** | High                                                     |
| **Type**     | Bug                                                      |

**Description**: `self.snapshot_counter = 0` appears twice in `_init_network_parameters`. The second initialization is redundant.

**Fix**: Remove the duplicate initialization.

---

### INT-P2-005: `or` Fallback Chain Bug for Falsy Values

| Field        | Value                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------------- |
| **Source**   | Source Code Review (NEW)                                                                        |
| **File**     | `src/cascade_correlation/cascade_correlation.py` (multiple lines in `_init_network_parameters`) |
| **Severity** | Medium                                                                                          |
| **Type**     | Bug                                                                                             |

**Description**: Patterns like `self.learning_rate = self.config.learning_rate or _CONSTANT` fall through to the constant when the config value is `0` or `0.0` (falsy). This is incorrect for parameters where `0` is a valid value.

**Fix**: Use explicit `if self.config.learning_rate is not None:` checks.

---

### INT-P2-006: `_initialize_spiral_problem_params` Uses `or` for Boolean Parameter

| Field        | Value                                      |
| ------------ | ------------------------------------------ |
| **Source**   | Source Code Review (NEW)                   |
| **File**     | `src/spiral_problem/spiral_problem.py:671` |
| **Severity** | Medium                                     |
| **Type**     | Bug                                        |

**Description**: `self.clockwise = clockwise or self.clockwise or _SPIRAL_PROBLEM_CLOCKWISE` will never be `False` if any previous value is truthy. Same issue for `noise` at line 695 when `0.0` is valid.

**Fix**: Use explicit `None` checks.

---

### INT-P2-007: `conftest.py` Fast-Slow Mode Logic Inverted

| Field        | Value                      |
| ------------ | -------------------------- |
| **Source**   | Source Code Review (NEW)   |
| **File**     | `src/tests/conftest.py:83` |
| **Severity** | Medium                     |
| **Type**     | Bug                        |

**Description**: `os.environ.get("JUNIPER_FAST_SLOW") == "0"` triggers "fast-slow" mode when the env var is `"0"`, semantically the opposite of fast-slow being enabled. The `_is_fast_mode()` in `test_spiral_problem.py` checks `== "1"`.

**Fix**: Align the condition to use consistent semantics.

---

### INT-P2-008: `_roll_sequence_number` Memory Issue in `CascadeCorrelationNetwork`

| Field        | Value                                                |
| ------------ | ---------------------------------------------------- |
| **Source**   | Source Code Review (NEW)                             |
| **File**     | `src/cascade_correlation/cascade_correlation.py:775` |
| **Severity** | High                                                 |
| **Type**     | Bug / Performance                                    |

**Description**: The `CascadeCorrelationNetwork._roll_sequence_number` uses a list comprehension storing all discarded values. Unlike the `CandidateUnit` version (fixed in CASCOR-P1-008), this version still has potential OOM risk.

**Fix**: Apply the same fix as CASCOR-P1-008: use a simple for-loop with `MAX_ROLL_COUNT` cap.

---

### INT-P2-009: Inconsistent Queue Method Names Between Remote Clients

| Field        | Value                                                                        |
| ------------ | ---------------------------------------------------------------------------- |
| **Source**   | Source Code Review (NEW)                                                     |
| **Files**    | `src/remote_client/remote_client.py`, `src/remote_client/remote_client_0.py` |
| **Severity** | High                                                                         |
| **Type**     | Integration                                                                  |

**Description**: The two remote client implementations register different queue names (`get_task_queue`/`get_result_queue` vs `get_tasks_queue`/`get_done_queue`). The old client will fail to connect to the current manager.

**Fix**: Remove `remote_client_0.py` or align its queue names.

---

### INT-P2-010: `os._exit()` Used Instead of `sys.exit()` in `main.py`

| Field        | Value                    |
| ------------ | ------------------------ |
| **Source**   | Source Code Review (NEW) |
| **File**     | `src/main.py:142,145`    |
| **Severity** | Medium                   |
| **Type**     | Enhancement              |

**Description**: `os._exit(1)` bypasses cleanup handlers, finally blocks, and atexit functions. `sys.exit()` should be preferred.

**Fix**: Replace `os._exit()` with `sys.exit()`.

---

### INT-P2-011: CI/CD Coverage Gates Not Enforced

| Field        | Value                                                         |
| ------------ | ------------------------------------------------------------- |
| **Source**   | Existing Documentation (PRE-DEPLOYMENT_ROADMAP-2, P2-NEW-002) |
| **Severity** | Medium                                                        |
| **Type**     | CI/CD                                                         |
| **Status**   | NOT STARTED                                                   |

**Description**: Need to add `coverage report --fail-under=80` to CI, configure per-module thresholds, and add coverage badge to README.

---

### INT-P2-012: Type Errors - Gradual Fix

| Field        | Value                                                         |
| ------------ | ------------------------------------------------------------- |
| **Source**   | Existing Documentation (PRE-DEPLOYMENT_ROADMAP-2, P2-NEW-006) |
| **Severity** | Medium                                                        |
| **Type**     | Code Quality                                                  |
| **Status**   | IN PROGRESS                                                   |

**Sub-tasks**:

- [ ] Run mypy and categorize errors
- [ ] Fix critical type errors
- [ ] Gradually increase strictness
- [ ] Remove `continue-on-error` from CI

---

### INT-P2-013: `check_object_pickleability` Depends on `dill` (Not Listed)

| Field        | Value                    |
| ------------ | ------------------------ |
| **Source**   | Source Code Review (NEW) |
| **File**     | `src/utils/utils.py:248` |
| **Severity** | Medium                   |
| **Type**     | Integration              |

**Description**: The TODO-marked function imports `dill` which is not listed as a runtime dependency. Will crash with `ModuleNotFoundError` if called.

**Fix**: Move `dill` to test-only dependencies or add proper import guard.

---

### INT-P2-014: Multiple `import traceback` Inside Exception Handlers

| Field        | Value                                                                 |
| ------------ | --------------------------------------------------------------------- |
| **Source**   | Source Code Review (NEW)                                              |
| **File**     | `src/cascade_correlation/cascade_correlation.py` (multiple locations) |
| **Severity** | Medium                                                                |
| **Type**     | Refactor                                                              |

**Description**: `import traceback` is repeated inside many `except` blocks rather than at the top of the file.

**Fix**: Move `import traceback` to the file-level imports.

---

## 6. Phase 3: Infrastructure & CI/CD

**Priority**: P3 - Standard
**Effort**: 20-32 hours
**Rationale**: Infrastructure improvements that enhance integration reliability and developer experience.

### INT-P3-001: Legacy Code Removal (Spiral Generator)

| Field        | Value                                                                          |
| ------------ | ------------------------------------------------------------------------------ |
| **Source**   | Existing Documentation (SPIRAL_DATA_GEN_REFACTOR_PLAN, Remaining Work Item #1) |
| **Severity** | Medium                                                                         |
| **Type**     | Refactor                                                                       |
| **Status**   | NOT STARTED (awaiting JuniperData stability)                                   |

**Description**: Once JuniperData is stable, remove the legacy spiral generator code from `spiral_problem.py`. The dual-path (legacy + JuniperData) creates maintenance burden.

---

### INT-P3-002: E2E Live Service Integration Tests

| Field        | Value                                                                          |
| ------------ | ------------------------------------------------------------------------------ |
| **Source**   | Existing Documentation (SPIRAL_DATA_GEN_REFACTOR_PLAN, Remaining Work Item #3) |
| **Severity** | Medium                                                                         |
| **Type**     | Testing                                                                        |
| **Status**   | NOT STARTED                                                                    |

**Description**: No automated integration tests spin up JuniperData and verify the full pipeline (Cascor -> JuniperData -> artifact -> tensor conversion -> training). All current tests use mocks.

---

### INT-P3-003: Docker Compose Validation

| Field        | Value                      |
| ------------ | -------------------------- |
| **Source**   | Integration Analysis (NEW) |
| **Severity** | Medium                     |
| **Type**     | Infrastructure             |
| **Status**   | NOT STARTED                |

**Description**: The docker-compose configuration in the refactor plan shows a 3-service deployment but has not been tested end-to-end.

---

### INT-P3-004: `sys.path` Mutation Architectural Concern

| Field        | Value                                        |
| ------------ | -------------------------------------------- |
| **Source**   | Existing Documentation (INTEGRATION_ROADMAP) |
| **Severity** | Medium                                       |
| **Type**     | Architecture                                 |
| **Status**   | DOCUMENTED (workaround in place)             |

**Description**: Canopy uses `sys.path.insert(0, cascor_src)` to import Cascor modules directly. This is fragile and creates import order dependencies. The module naming collision was resolved (renamed to `cascor_constants/`), but the `sys.path` mutation pattern remains.

**Long-term fix**: Implement proper IPC (INT-P1-004) or make Cascor installable as a package.

---

### INT-P3-005: Test WebSocket Responsiveness During Training

| Field        | Value                                                                  |
| ------------ | ---------------------------------------------------------------------- |
| **Source**   | Existing Documentation (PRE-DEPLOYMENT_ROADMAP-2, P1-NEW-003 sub-task) |
| **Severity** | Medium                                                                 |
| **Type**     | Testing                                                                |
| **Status**   | NEEDS MANUAL VERIFICATION                                              |

**Description**: When Cascor training runs via `asyncio.run_in_executor()` in FastAPI, WebSocket responsiveness should be verified under load.

---

### INT-P3-006: Create Baseline Performance Profiles

| Field        | Value                                                                  |
| ------------ | ---------------------------------------------------------------------- |
| **Source**   | Existing Documentation (PRE-DEPLOYMENT_ROADMAP-2, P3-NEW-002 sub-task) |
| **Severity** | Low                                                                    |
| **Type**     | Performance                                                            |
| **Status**   | DEFERRED                                                               |

**Description**: Create baseline py-spy profiles for key operations to enable performance regression detection.

---

### INT-P3-007: Add Profiling to CI/CD for Regression Detection

| Field        | Value                                                                  |
| ------------ | ---------------------------------------------------------------------- |
| **Source**   | Existing Documentation (PRE-DEPLOYMENT_ROADMAP-2, P3-NEW-002 sub-task) |
| **Severity** | Low                                                                    |
| **Type**     | CI/CD                                                                  |
| **Status**   | DEFERRED                                                               |

---

### INT-P3-008: `.pytest.ini.swp` and Coverage Files in Git

| Field        | Value                                                    |
| ------------ | -------------------------------------------------------- |
| **Source**   | Source Code Review (NEW)                                 |
| **Files**    | `src/tests/.pytest.ini.swp`, `.coverage`, `coverage.xml` |
| **Severity** | Low                                                      |
| **Type**     | Enhancement                                              |

**Description**: Vim swap file and coverage artifacts appear in git status. Should be gitignored.

**Fix**: Add to `.gitignore`.

---

### INT-P3-009: Version Strings Inconsistent Across Files

| Field        | Value                    |
| ------------ | ------------------------ |
| **Source**   | Source Code Review (NEW) |
| **Severity** | Low                      |
| **Type**     | Enhancement              |

**Description**: File headers show various versions: `0.3.1 (0.7.3)`, `0.3.2 (0.7.3)`, `1.0.1`, `0.1.0`. CLAUDE.md states `0.6.2 (0.7.3)`.

---

### INT-P3-010: `cascor_snapshots` vs `snapshots` Directory Confusion

| Field        | Value                                       |
| ------------ | ------------------------------------------- |
| **Source**   | Source Code Review (NEW)                    |
| **File**     | `src/cascor_snapshots/` vs `src/snapshots/` |
| **Severity** | Low                                         |
| **Type**     | Enhancement                                 |

**Description**: Both directories exist under `src/`. Purpose distinction is unclear.

---

## 7. Phase 4: Enhancements & Future Work

**Priority**: P4 - Deferred
**Effort**: 40-80 hours
**Rationale**: Lower-priority enhancements, architectural improvements, and future features.

### 7.1 Architecture & Infrastructure (Deferred)

| ID         | Description                                           | Source                        | Status      |
| ---------- | ----------------------------------------------------- | ----------------------------- | ----------- |
| INT-P4-001 | GPU support (P3-NEW-003)                              | PRE-DEPLOYMENT_ROADMAP-2      | NOT STARTED |
| INT-P4-002 | Continuous profiling / Grafana Pyroscope (P3-NEW-004) | PRE-DEPLOYMENT_ROADMAP-2      | NOT STARTED |
| INT-P4-003 | PyTorch profiling (torch.profiler)                    | PRE-DEPLOYMENT_ROADMAP        | NOT STARTED |
| INT-P4-004 | Design minimal JuniperBranch worker package           | PRE-DEPLOYMENT_ROADMAP-2      | FUTURE      |
| INT-P4-005 | Document worker deployment procedure                  | PRE-DEPLOYMENT_ROADMAP-2      | FUTURE      |
| INT-P4-006 | Test distributed training across platforms            | PRE-DEPLOYMENT_ROADMAP-2      | FUTURE      |
| INT-P4-007 | Phase 5: Extended Data Sources (S3, DB, HuggingFace)  | SPIRAL_DATA_GEN_REFACTOR_PLAN | DEFERRED    |

### 7.2 Code Quality (Deferred)

| ID         | Description                                                      | Source                   | Status   |
| ---------- | ---------------------------------------------------------------- | ------------------------ | -------- |
| INT-P4-008 | Candidate factory refactor (P3-001)                              | PRE-DEPLOYMENT_ROADMAP   | DEFERRED |
| INT-P4-009 | Remove "roll" concept in CandidateUnit                           | PRE-DEPLOYMENT_ROADMAP-2 | DEFERRED |
| INT-P4-010 | Add metrics for multiprocessing fallback frequency               | PRE-DEPLOYMENT_ROADMAP-2 | DEFERRED |
| INT-P4-011 | Test fallback under various failure modes                        | PRE-DEPLOYMENT_ROADMAP-2 | DEFERRED |
| INT-P4-012 | `LogConfig.__init__` "crazy" parameter naming cleanup            | Source Code Review       | LOW      |
| INT-P4-013 | Logger "steaming pile" TODO cleanup                              | Source Code Review       | LOW      |
| INT-P4-014 | Remove commented-out code blocks                                 | Source Code Review       | LOW      |
| INT-P4-015 | Clean up "Original corrupted line" comments in spiral_problem.py | Source Code Review       | LOW      |
| INT-P4-016 | Remove `uuid as uuid` redundant import alias                     | Source Code Review       | LOW      |
| INT-P4-017 | MED-014: Line length reduction (full codebase reformat)          | TEST_SUITE_CICD_PLAN     | DEFERRED |

### 7.3 Canopy Enhancements (CAN-001 through CAN-021)

All 21 items from PRE-DEPLOYMENT_ROADMAP-2 are **NOT STARTED**:

| ID                | Description                             |
| ----------------- | --------------------------------------- |
| CAN-001 - CAN-005 | Training Metrics dashboard improvements |
| CAN-006 - CAN-010 | Meta Parameter Tuning tab               |
| CAN-011 - CAN-015 | Tooltips and tutorials                  |
| CAN-016 - CAN-021 | Network hierarchy/population display    |

### 7.4 Cascor Enhancements (CAS-001 through CAS-010)

All 10 items from PRE-DEPLOYMENT_ROADMAP-2 are **NOT STARTED**:

| ID      | Description                              | Notes                              |
| ------- | ---------------------------------------- | ---------------------------------- |
| CAS-001 | Spiral generator extraction              | Partially addressed by JuniperData |
| CAS-002 | Epoch separation                         |                                    |
| CAS-003 | Max iterations                           |                                    |
| CAS-004 | Remote worker extraction (JuniperBranch) |                                    |
| CAS-005 | Common module extraction                 |                                    |
| CAS-006 | Auto-snap best network                   |                                    |
| CAS-007 | Test optimization                        |                                    |
| CAS-008 | Network hierarchy                        |                                    |
| CAS-009 | Population management                    |                                    |
| CAS-010 | Vector DB snapshots                      |                                    |

---

## 8. Dependencies Matrix

```bash
INT-P0-001 (Walrus bug)
    └── No dependencies, fix immediately

INT-P0-002 (ActivationWithDerivative duplication)
    └── No dependencies, fix immediately

INT-P0-003 (Invalid CandidateUnit params)
    └── No dependencies, fix immediately

INT-P0-004 (Hardcoded path in remote_client_0)
    └── INT-P2-009 (Inconsistent queue names) - should be fixed together

INT-P1-001 (Duplicated JuniperDataClient)
    ├── INT-P1-002 (requests dependency) - should be fixed together
    └── INT-P1-003 (No shared protocol package)

INT-P1-004 (Full IPC)
    ├── INT-P3-004 (sys.path mutation)
    └── INT-P1-001 (shared client package)

INT-P2-011 (CI coverage gates)
    └── INT-P2-012 (Type errors) - coverage gates should wait for type fixes

INT-P3-001 (Legacy code removal)
    └── INT-P3-002 (E2E integration tests) - validate before removing legacy

INT-P4-001 (GPU support)
    └── INT-P4-002 (Continuous profiling) - profile before optimizing
```

---

## 9. Risk Assessment

| Risk                                                                        | Probability | Impact | Mitigation                                                    |
| --------------------------------------------------------------------------- | ----------- | ------ | ------------------------------------------------------------- |
| Walrus operator bug (INT-P0-001) causes silent data corruption in snapshots | High        | High   | Fix immediately in Phase 0                                    |
| `ActivationWithDerivative` ACTIVATION_MAP divergence between files          | Medium      | High   | Extract to shared module                                      |
| JuniperData service downtime crashes training                               | Medium      | High   | Add retry logic (INT-P1-007) and fallback to local generation |
| `sys.path` mutation causes import conflicts in production                   | Medium      | Medium | Document workaround; long-term fix via IPC                    |
| Coverage regression without enforced gates                                  | High        | Medium | Implement CI coverage gates (INT-P2-011)                      |
| Hardcoded paths break on deployment to other machines                       | High        | Medium | Fix all hardcoded paths in Phase 0                            |

---

## 10. Appendix: Source Code Review Details

### 10.1 Files Reviewed

| File                                                | Lines | Issues Found |
| --------------------------------------------------- | ----- | ------------ |
| `src/cascade_correlation/cascade_correlation.py`    | ~4500 | 18           |
| `src/candidate_unit/candidate_unit.py`              | ~800  | 4            |
| `src/spiral_problem/spiral_problem.py`              | ~1600 | 7            |
| `src/spiral_problem/check.py`                       | ~400  | 1            |
| `src/spiral_problem/data_provider.py`               | ~150  | 1            |
| `src/main.py`                                       | ~300  | 4            |
| `src/juniper_data_client/client.py`                 | ~100  | 2            |
| `src/remote_client/remote_client.py`                | ~100  | 1            |
| `src/remote_client/remote_client_0.py`              | ~80   | 2            |
| `src/utils/utils.py`                                | ~300  | 1            |
| `src/log_config/log_config.py`                      | ~200  | 1            |
| `src/log_config/logger/logger.py`                   | ~500  | 1            |
| `src/snapshots/snapshot_serializer.py`              | ~400  | 0            |
| `src/tests/conftest.py`                             | ~150  | 1            |
| `src/tests/unit/test_candidate_training_manager.py` | ~100  | 1            |
| Codebase-wide (TODOs, comments)                     | -     | 2            |
| **Total**                                           |       | **47**       |

### 10.2 Issue Distribution

| Severity | Count | Percentage |
| -------- | ----- | ---------- |
| Critical | 5     | 10.6%      |
| High     | 12    | 25.5%      |
| Medium   | 16    | 34.0%      |
| Low      | 14    | 29.8%      |

### 10.3 Issue Types

| Type            | Count |
| --------------- | ----- |
| Bug             | 18    |
| Integration     | 10    |
| Refactor        | 8     |
| Enhancement     | 7     |
| Bug/Refactor    | 2     |
| Bug/Integration | 1     |
| Bug/Performance | 1     |

---

## Document History

| Date       | Version | Author          | Changes                                                           |
| ---------- | ------- | --------------- | ----------------------------------------------------------------- |
| 2026-02-05 | 1.0.0   | Claude Opus 4.5 | Initial compilation from 4 roadmap documents + source code review |

---

**Next Review**: After Phase 0 completion
**Owner**: Development Team
**Document Status**: Active
