# JuniperCascor Post-Release Development Roadmap

**Project**: JuniperCascor - Cascade Correlation Neural Network Backend
**Created**: 2026-02-17
**Last Updated**: 2026-02-24 (polyrepo migration impact analysis)
**Author**: Paul Calnon
**Status**: Active - Post-Migration Reconciliation
**Source**: Exhaustive audit of all JuniperCascor `notes/` files (2026-02-18), reconciled against polyrepo migration (2026-02-24)

---

## Overview

This document is the **authoritative, consolidated roadmap** for all JuniperCascor updates, changes, fixes, and enhancements. It was originally compiled by exhaustive audit of every markdown file in the `notes/` directory on 2026-02-18. Items were de-duplicated across 25+ source documents and organized by phase.

**2026-02-24 Update**: This roadmap has been reconciled against the polyrepo migration documented in `POLYREPO_MIGRATION_PLAN.md` (v1.5.0) and `DECOUPLE_CANOPY_FROM_CASCOR_PLAN.md`. The migration (Phases 0-3 complete, Phase 4-5 in progress) has resolved, superseded, or changed the scope of many items. Each affected item now carries a **Migration Impact** annotation.

### Polyrepo Migration Summary (as of 2026-02-24)

| Migration Phase | Status | Key Impact on This Roadmap |
| --- | --- | --- |
| Phase 0 — Stabilize baseline | **COMPLETE** | Clean baseline established for all subprojects |
| Phase 1 — Publish `juniper-data-client` to PyPI | **COMPLETE** | Resolves INT-P1-001 (duplicated client), INT-P1-002 (requests dep — pre-migration resolved, further addressed by Phase 1) |
| Phase 2 — Build CasCor Service API | **COMPLETE** | Resolves C.1 (async wrapper); substantially resolves INT-P1-004 (IPC architecture, with Phase 3) |
| Phase 3 — Create `juniper-cascor-client` + `juniper-cascor-worker` | **COMPLETE** | Resolves CAS-004 (extract remote worker), supersedes C.2 |
| Phase 4 — Decouple Canopy from CasCor | **IN PROGRESS** | Removes `CascorIntegration`; changes scope of CAS-CANOPY-* items |
| Phase 5 — Split into separate repos | **IN PROGRESS** | CasCor extracted to `pcalnon/juniper-cascor`; Canopy pending |
| Phase 6 — Post-migration hardening | **NOT STARTED** | Adds new items: version matrix, integration test suite, Docker Compose |

### Source Documents Evaluated

| Document                                        | Date       | Items Extracted                                           | Status                                  |
| ----------------------------------------------- | ---------- | --------------------------------------------------------- | --------------------------------------- |
| PRE-DEPLOYMENT_ROADMAP.md                       | 2026-01-25 | Integration analysis, profiling roadmap, coverage roadmap | Superseded by Roadmap-2                 |
| PRE-DEPLOYMENT_ROADMAP-2.md                     | 2026-02-03 | 5 remaining tasks, CAS-001–CAS-010, deferred items        | Active (14/19 complete)                 |
| INTEGRATION_ROADMAP.md                          | 2026-02-05 | Cascor-Canopy integration fixes                           | All critical blockers resolved          |
| INTEGRATION_ROADMAP-01.md                       | 2026-02-05 | 61 issues (INT-P0 through INT-P4)                         | Active - most comprehensive             |
| CASCOR_JUNIPER_DATA_INTEGRATION_PLAN.md         | 2026-02-07 | CAS-INT-001 through CAS-INT-009                           | ALL 9 COMPLETE                          |
| JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md | 2026-02-07 | Phase 5 deferred, legacy code removal                     | Phases 0-4 Complete                     |
| TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md | 2026-02-05 | MED-014 deferred, qualitative metrics                     | Phases 0-4 Complete                     |
| CRITICAL_FIXES_REQUIRED.md                      | 2025-10-15 | P0/P1/P2 items                                            | P0 all fixed (per COMPLETE_FIX_SUMMARY) |
| CODE_REVIEW_SUMMARY.md                          | 2025-10-15 | P1 #11-14, P2 #15-18                                      | P0 fixed, P1/P2 pending                 |
| COMPLETE_FIX_SUMMARY.md                         | 2025-10-16 | 4 optional P2 enhancements                                | System production-ready                 |
| DEVELOPMENT_ROADMAP.md                          | 2025-10-28 | P0-P4 phases, in-code TODOs                               | Many items superseded                   |
| IMPLEMENTATION_PROGRESS.md                      | 2025-10-28 | ENH-006/007/008 pending                                   | Phase 1 in progress                     |
| NEXT_STEPS.md                                   | 2025-10-16 | Serialization enhancements                                | Many already implemented                |
| ORACLE_ANALYSIS_PYTHON.md                       | 2026-01-26 | 5 recommended Python fixes                                | Status unknown                          |
| ORACLE_ANALYSIS_SCRIPTS.md                      | 2026-01-26 | Shell script path fixes                                   | Status unknown                          |
| Oracle_analysis_2026-01-26.md                   | 2026-01-26 | C.1/C.2/C.3 integration architecture                      | C.1 resolved, C.2 superseded, C.3 substantially resolved (via INT-P1-004) |
| DOCUMENTATION_AUDIT.md                          | 2026-01-29 | 5 future doc enhancements                                 | Future work                             |
| FINAL_STATUS.md                                 | 2025-10-16 | Remaining work items                                      | Many superseded                         |
| FEATURES_GUIDE.md                               | 2025-01-12 | Feature reference                                         | No action items                         |
| API_REFERENCE.md                                | 2025-01-12 | API reference                                             | No action items                         |
| ARCHITECTURE_GUIDE.md                           | 2025-01-12 | Architecture reference                                    | No action items                         |
| IMPLEMENTATION_CHECKLIST.md                     | 2025-10-28 | Testing/verification checklists                           | Many superseded                         |
| IMPLEMENTATION_SUMMARY.md                       | 2025-10-28 | ENH status tracking                                       | Many already resolved                   |
| CHANGES_FOR_REVIEW.md                           | Various    | Change reviews                                            | All complete                            |
| POLYREPO_MIGRATION_PLAN.md                      | 2026-02-24 | Migration phases, architecture changes                    | **NEW** — Phases 0-3 complete           |
| DECOUPLE_CANOPY_FROM_CASCOR_PLAN.md             | 2026-02-21 | Canopy decoupling, CascorServiceAdapter design            | **NEW** — Phase 4 in progress           |

### Consolidated Statistics

| Category                         | Count | Change from 2026-02-18 |
| -------------------------------- | ----- | ---------------------- |
| Total unique non-completed items | 83    | -6 (was 89)            |
| Resolved/superseded by migration | 6     | +6 (see Section 10)    |
| Pre-migration resolved           | 2     | INT-P1-002, INT-P2-013 |
| Scope changed by migration       | 6     | (reclassified)         |
| New items from migration         | 2     | +2                     |
| Critical (P0)                    | 3     | -2 (was 5)            |
| High (P1)                        | 8     | -8 (was 16)           |
| Medium (P2)                      | 22    | -3 (was 25)           |
| Low / Deferred (P3-P4)           | 50    | +7 (was 43)           |

*Total = 89 - 6 (migration resolved/superseded) - 2 (pre-migration resolved) + 2 (new items) = 83. Priority sum: 3 + 8 + 22 + 50 = 83. P2 decrease reflects INT-P2-013 resolved pre-migration, INT-P2-004 downgraded to Low, partially offset by INT-P1-006 and INT-P2-009 entering P2 from High (downgraded to Medium); net -3 includes recount of items by effective severity vs. original section placement. P3-P4 increase reflects items downgraded from P0 (INT-P0-004, INT-P0-005) and from High/P1 severity (INT-P1-003, INT-P1-005, INT-P2-004 — note: INT-P2-004's "P2" is its integration phase, not priority; its original severity was High) plus 2 new items (Phase 1 #10: clean up `src/remote_client/` directory; Phase 5 #10: implement deferred API endpoints).*

### Codebase Validation Summary (2026-02-18)

| Result                   | Count     | Items                                                                                                        |
| ------------------------ | --------- | ------------------------------------------------------------------------------------------------------------ |
| CONFIRMED (bug exists)   | 19        | INT-P0-001 through P0-005, P2-001 through P2-010, P2-014, CAS-REF-004, INT-P1-005, INT-P1-008 (INT-P1-005 and INT-P2-004 also severity-adjusted) |
| RESOLVED (already fixed) | 3         | INT-P1-002 (requests dep), INT-P1-007 (retry logic), INT-P2-013 (dill dep)                                   |
| SEVERITY ADJUSTED        | 3         | INT-P2-004 (High→Low), INT-P1-006 (High→Medium), INT-P1-005 (High→Low)                                       |
| NOT YET VALIDATED        | Remaining | Architecture items (C.1, C.2 — now resolved/superseded by migration), deferred items, Oracle analysis items   |

---

## Section 1: Critical Bugs & Blockers (P0)

**Priority**: IMMEDIATE
**Estimated Effort**: 4-8 hours for P0 items only (reduced from 8-16: two P0 items downgraded by migration; see Phase 0 table for execution plan which adds P2 quick wins)
**Source**: INTEGRATION_ROADMAP-01.md (2026-02-05), Source Code Review

These items were identified during the 2026-02-05 source code review. They represent actual bugs in the current codebase that could cause incorrect behavior or crashes.

### INT-P0-001: Walrus Operator Precedence Bug

**Status**: NOT STARTED
**Severity**: Critical
**Source**: INTEGRATION_ROADMAP-01.md
**File**: `src/cascade_correlation/cascade_correlation.py` (line 1322)

**Description**: The `train_output_layer` method contains a walrus operator (`:=`) with incorrect precedence that could cause silent data corruption in snapshots.

**Codebase Validation (2026-02-18)**: **CONFIRMED**. Line 1322: `if snapshot_path := self.create_snapshot() is not None:` — The `:=` operator has lower precedence than `is not`, so `snapshot_path` is assigned the boolean result of `self.create_snapshot() is not None` (i.e., `True`/`False`) instead of the actual snapshot path. The subsequent log message at line 1323 would print `True` instead of the file path. Fix: add parentheses `if (snapshot_path := self.create_snapshot()) is not None:`. Note: 8 additional walrus expressions in the file (lines 1740, 2765, 2780, 3154, 3540, 3544, 3595, 3648) appear syntactically correct.

---

### INT-P0-002: ActivationWithDerivative Class Duplicated

**Status**: NOT STARTED
**Severity**: Critical
**Source**: INTEGRATION_ROADMAP-01.md
**File**: `src/cascade_correlation/cascade_correlation.py` (line 291), `src/candidate_unit/candidate_unit.py` (line 138)

**Description**: The `ActivationWithDerivative` class is duplicated in two files. If the ACTIVATION_MAP diverges between the two copies, incorrect activation functions could be used. Should be extracted to a shared module.

**Codebase Validation (2026-02-18)**: **CONFIRMED**. Exact duplicate at `cascade_correlation.py:291` and `candidate_unit.py:138`. Both define `class ActivationWithDerivative`. Fix: extract to a shared module (e.g., `src/activation/activation_with_derivative.py`) and import in both files.

---

### INT-P0-003: Invalid CandidateUnit Constructor Parameters in `fit`

**Status**: NOT STARTED
**Severity**: Critical
**Source**: INTEGRATION_ROADMAP-01.md
**File**: `src/cascade_correlation/cascade_correlation.py` (lines 1154-1166)

**Description**: The `fit` method passes invalid parameters to the `CandidateUnit` constructor.

**Codebase Validation (2026-02-18)**: **CONFIRMED — WORSE THAN DOCUMENTED**. Lines 1154-1166: All 11 parameters use the `_CandidateUnit__` prefix (with leading underscore, Python name-mangling style) instead of the correct `CandidateUnit__` prefix used by the constructor. Since CandidateUnit's `__init__` accepts `**kwargs`, ALL parameters are silently absorbed and the CandidateUnit is created with ALL default values. This means training in the `fit()` path silently ignores the user's activation function, input size, output size, learning rate, etc. Additionally, `_CandidateUnit__candidate_pool_size`, `_CandidateUnit__log_file_name`, and `_CandidateUnit__log_file_path` don't exist as CandidateUnit constructor parameters even with the correct prefix. Note: The other two CandidateUnit creation sites (line 989 `_create_candidate_unit` and line 2158 `train_candidate_worker`) use the correct `CandidateUnit__` prefix and appear correct.

---

### INT-P0-004: Hardcoded Path in remote_client_0.py

**Status**: NOT STARTED → **SCOPE CHANGED (consider deletion)**
**Severity**: ~~Critical~~ **Low** (legacy file, replaced by `juniper-cascor-worker`; 15-min deletion)
**Source**: INTEGRATION_ROADMAP-01.md
**File**: `src/remote_client/remote_client_0.py` (line 16)

**Description**: Contains a hardcoded absolute path that will break on any machine other than the development environment.

**Codebase Validation (2026-02-18)**: **CONFIRMED**. Line 16: `sys.path.append("/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src")`. Points to the old prototypes directory, not even the current project structure.

**Migration Impact (2026-02-24)**: `remote_client_0.py` is a legacy predecessor to `remote_client.py`, which itself is now superseded by the standalone `juniper-cascor-worker` package (published to PyPI as v0.1.0). The hardcoded path references the old monorepo prototype directory that no longer exists in the polyrepo layout. **Recommended action**: Delete `remote_client_0.py` entirely rather than fixing the path. If backward compatibility is needed during transition, fix the path as originally planned. Severity reduced from Critical to Low because this file is no longer in the active execution path.

**Related**: INT-P2-009 (inconsistent queue names between remote clients)

---

### INT-P0-005: Hardcoded Paths in Test File

**Status**: NOT STARTED → **SCOPE CHANGED**
**Severity**: ~~Critical~~ **Low** (paths are stale monorepo references; 15-min removal)
**Source**: INTEGRATION_ROADMAP-01.md
**File**: `src/tests/unit/test_candidate_training_manager.py` (lines 10-12)

**Description**: Test file contains hardcoded absolute paths.

**Codebase Validation (2026-02-18)**: **CONFIRMED**. Line 10: `sys.path.append("/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src")` (Linux), Line 12: `sys.path.append("/Users/pcalnon/Development/python/Juniper/src/prototypes/cascor/src")` (macOS). Both point to obsolete prototype paths.

**Migration Impact (2026-02-24)**: These paths reference the old monorepo prototype directory. In the polyrepo layout, `juniper-cascor` is installed via `pip install -e ".[all]"` and imports resolve through standard Python packaging — no `sys.path` manipulation is needed. **Recommended action**: Remove both `sys.path.append()` lines entirely. The CI workflow already uses editable install, so these paths are dead code. Severity reduced from Critical to Low because the paths don't break CI (they just fail silently on machines where the path doesn't exist).

---

## Section 2: Integration Architecture (P1)

**Priority**: HIGH
**Estimated Effort**: 8-16 hours (reduced from 24-40: many items resolved by migration)
**Source**: INTEGRATION_ROADMAP-01.md, Oracle_analysis_2026-01-26.md, PRE-DEPLOYMENT_ROADMAP-2.md

### INT-P1-001: Duplicated JuniperDataClient

**Status**: ~~NOT STARTED~~ **RESOLVED (by migration Phase 1)**
**Severity**: ~~High~~ N/A
**Source**: INTEGRATION_ROADMAP-01.md

**Description**: `JuniperDataClient` is duplicated in both JuniperCascor (`src/juniper_data_client/client.py`) and JuniperCanopy. Changes to the client API must be synchronized manually.

**Migration Impact (2026-02-24)**: **RESOLVED**. `juniper-data-client` v0.3.0 has been published to PyPI as the single source of truth. All vendored copies have been removed from CasCor, Canopy, and JuniperData. CasCor's `pyproject.toml` declares `juniper-data-client>=0.3.0` under `[project.optional-dependencies].juniper-data`. The `src/juniper_data_client/` directory in CasCor has been entirely removed.

---

### INT-P1-002: `requests` as Undeclared Dependency

**Status**: ~~NOT STARTED~~ **RESOLVED**
**Severity**: ~~High~~ N/A
**Source**: INTEGRATION_ROADMAP-01.md

**Description**: The `requests` library is used by `JuniperDataClient` but is not declared in the project's dependency files.

**Codebase Validation (2026-02-18)**: **RESOLVED**. Originally noted as resolved because `requests` was present in `conf/requirements-pip.txt` (line 95: `requests==2.32.5`). Post-migration update: `requests` is no longer a direct dependency of `juniper-cascor` — it is not declared in `pyproject.toml`. The vendored `JuniperDataClient` that required `requests` was removed (see INT-P1-001); `requests` is now a transitive dependency of the external `juniper-data-client` package.

---

### INT-P1-003: No Shared Protocol Package

**Status**: ~~NOT STARTED~~ **PARTIALLY RESOLVED (by migration Phases 1-3)**
**Severity**: ~~High~~ **Low**
**Source**: INTEGRATION_ROADMAP-01.md

**Description**: Three Juniper applications share API contracts, data formats, and client code but have no shared protocol/interface package. Each duplicates validation logic.

**Migration Impact (2026-02-24)**: **PARTIALLY RESOLVED**. The migration created three published client packages (`juniper-data-client`, `juniper-cascor-client`, `juniper-cascor-worker`) that define the inter-service API contracts. The CasCor service API uses a formal response envelope (`{status, data, meta}`) documented in `POLYREPO_MIGRATION_PLAN.md` Appendix A. However, there is still no formal `juniper-common` package for shared constants, base exceptions, or data format definitions (e.g., NPZ key contracts). This is a lower-priority concern now that the primary coupling has been eliminated.

**Remaining work**: Consider creating a `juniper-common` package if shared schema definitions accumulate across clients. Currently the client packages each define their own minimal schemas, which is acceptable at this scale.

---

### INT-P1-004: Full IPC Architecture

**Status**: ~~DEFERRED~~ **SUBSTANTIALLY RESOLVED (by migration Phases 2-3)**
**Severity**: ~~High~~ N/A
**Source**: INTEGRATION_ROADMAP-01.md, Oracle_analysis_2026-01-26.md (C.3), PRE-DEPLOYMENT_ROADMAP-2.md (P1-NEW-001)

**Description**: JuniperCascor is currently embedded in JuniperCanopy's process via `sys.path.insert()`. A proper IPC architecture (separate backend process with protocol-based communication) would enable independent deployment and scaling.

**Migration Impact (2026-02-24)**: **SUBSTANTIALLY RESOLVED**. The polyrepo migration implemented this as a first-class architecture:

- **CasCor Service API** (Phase 2): FastAPI server on port 8200 with 19 REST endpoints + 2 WebSocket endpoints
- **juniper-cascor-client** (Phase 3): Published HTTP/WebSocket client library for consuming the service
- **CascorServiceAdapter** (Phase 4, in progress): Replaces `CascorIntegration` in Canopy, delegates to the client library
- **`sys.path.insert()` elimination**: Phase 4 Step 5.8 will remove all `sys.path` manipulation and direct CasCor imports from Canopy

All four sub-tasks are addressed:

| Sub-task | Status |
| --- | --- |
| Design IPC protocol specification | **COMPLETE** — REST + WebSocket, documented in Appendix A of migration plan |
| Implement Cascor server mode | **COMPLETE** — `src/server.py` + `src/api/` module |
| Update Canopy to connect to external Cascor | **IN PROGRESS** — `CascorServiceAdapter` committed, integration testing pending |
| Connection management and health checks | **COMPLETE** — `/v1/health`, `/v1/health/live`, `/v1/health/ready` endpoints; `wait_for_ready()` in client |

---

### INT-P1-005: main.py Passes Invalid Parameters to SpiralProblem

**Status**: NOT STARTED
**Severity**: ~~High~~ **Low** (confirmed low impact 2026-02-18; silently absorbed by `**kwargs`)
**Source**: INTEGRATION_ROADMAP-01.md

**Description**: `main.py` passes parameters that `SpiralProblem` does not accept or uses incorrect parameter names.

**Codebase Validation (2026-02-18)**: **CONFIRMED — LOW IMPACT**. `main.py` line 208 passes `_SpiralProblem__spiral_config=logging.config` but SpiralProblem's constructor has no `spiral_config` parameter. Since `SpiralProblem.__init__` accepts `**kwargs` (line 177), this is silently absorbed. The config object is never used. Other parameters (lines 209-238) use the correct `_SpiralProblem__` prefix matching the constructor and appear valid.

---

### INT-P1-006: Missing Import Guard for SpiralDataProvider

**Status**: NOT STARTED
**Severity**: ~~High~~ **Medium** (partially resolved, see validation)
**Source**: INTEGRATION_ROADMAP-01.md

**Description**: Missing try/except import guard for `SpiralDataProvider` — if `requests` is not installed, the import fails silently or with a confusing error.

**Codebase Validation (2026-02-18)**: **PARTIALLY RESOLVED**. The import at `spiral_problem.py:509` is a lazy import inside `generate_n_spiral_dataset()`, so it only fails when actually calling the data provider. However, `requests` is now a declared dependency (see INT-P1-002 RESOLVED), so the "not installed" scenario is unlikely. The import still lacks a try/except guard for more descriptive error messages if the `data_provider` module itself has issues. **Severity reduced to MEDIUM**.

---

### INT-P1-007: No Connection Retry Logic for JuniperData Client

**Status**: COMPLETE (via CAS-INT-008)
**Resolution**: Implemented in `JuniperDataClient._request()` with `MAX_RETRIES = 3` and exponential backoff.

---

### INT-P1-008: check.py is a Stale Duplicate

**Status**: NOT STARTED
**Severity**: High
**Source**: INTEGRATION_ROADMAP-01.md
**File**: `src/spiral_problem/check.py`

**Description**: `check.py` is a stale duplicate of `spiral_problem.py` that should be removed.

**Codebase Validation (2026-02-18)**: **CONFIRMED**. File `src/spiral_problem/check.py` still exists.

---

### C.1: Async Wrapper for Synchronous fit()

**Status**: ~~NOT STARTED~~ **RESOLVED (by migration Phase 2)**
**Severity**: ~~High~~ N/A
**Effort**: ~~Medium (2-4 days)~~ N/A
**Source**: Oracle_analysis_2026-01-26.md

**Description**: Add async wrapper around synchronous `fit()` using `loop.run_in_executor()` for FastAPI endpoints. Includes:

- Add `ThreadPoolExecutor` and training task state to `CascorIntegration`
- Add `monitored_fit_async()` method
- Update FastAPI endpoints to call async wrapper
- Implement cancellation strategy ("stop requested" flag)
- Make broadcasting thread-safe (`websocket_manager.broadcast_sync`)

**Migration Impact (2026-02-24)**: **RESOLVED**. All sub-items have been implemented in the CasCor Service API (Phase 2):

| Sub-item | Implementation |
| --- | --- |
| ThreadPoolExecutor | `TrainingLifecycleManager` (579 lines) uses `ThreadPoolExecutor` for async training |
| Async training method | `POST /v1/training/start` triggers async training via lifecycle manager |
| Cancellation strategy | `TrainingStateMachine` formal FSM: STOPPED ↔ STARTED ↔ PAUSED → COMPLETED/FAILED |
| Thread-safe broadcasting | `WebSocketManager.broadcast_from_thread()` provides async/sync bridge |

The implementation lives in `src/api/lifecycle/manager.py`, `src/api/lifecycle/state_machine.py`, and `src/api/websocket/manager.py`. This is now part of the CasCor service, not `CascorIntegration`.

---

### C.2: Expose RemoteWorkerClient Through CascorIntegration

**Status**: ~~NOT STARTED~~ **SUPERSEDED (by migration Phase 3)**
**Severity**: ~~High~~ N/A
**Effort**: ~~Large (1-2 weeks)~~ N/A
**Source**: Oracle_analysis_2026-01-26.md

**Description**: Expose `RemoteWorkerClient` through `CascorIntegration` with config, API endpoints, and packaging. Includes:

- Import/packaging sanity for `RemoteWorkerClient`
- Expose configuration in Canopy (`app_config.yaml`)
- Add minimal API surface (connect/start/stop/disconnect)
- Connect remote workers to training (inject task/result queues)
- FastAPI endpoints for remote worker admin (POST /api/remote/*)
- Package as `juniper_branch` (shared Python package)

**Migration Impact (2026-02-24)**: **SUPERSEDED**. The polyrepo migration took a fundamentally different approach:

1. **`juniper-cascor-worker`** (published to PyPI as v0.1.0) replaces the planned `juniper_branch` package. It is a standalone CLI worker that connects to the CasCor service.
2. **Worker management is server-side**: The CasCor service manages workers directly. The migration plan lists `/v1/workers/*` endpoints as deferred, to be implemented when worker coordination is needed.
3. **`CascorIntegration` is being removed**: Phase 4 replaces it with `CascorServiceAdapter`, which exposes stub no-ops for all worker methods (`connect_remote_workers`, `start_remote_workers`, etc.) because workers are managed server-side.
4. **No Canopy-side worker config needed**: `app_config.yaml` worker section is no longer needed; workers connect to the CasCor service independently.

**Remaining work**: Implement the deferred `/v1/workers/*` REST endpoints in the CasCor service when remote worker coordination features are needed.

---

## Section 3: Code Quality & Testing (P2)

**Priority**: MEDIUM
**Estimated Effort**: 28-44 hours (reduced from 32-48: some items partially addressed by migration CI changes)
**Source**: INTEGRATION_ROADMAP-01.md, PRE-DEPLOYMENT_ROADMAP-2.md, TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md

### 3.1 Source Code Bugs

#### INT-P2-001: Undeclared Global Variable `shared_object_dict`

**Status**: NOT STARTED
**Severity**: High
**File**: `src/cascade_correlation/cascade_correlation.py` (line 2300)

**Codebase Validation (2026-02-18)**: **CONFIRMED**. Line 2300: `global shared_object_dict` — used in `_train_candidate_unit` static method but never declared at module scope. It's expected to be injected by the multiprocessing module's shared memory setup. While functional in practice (the global is set before worker processes fork), it fails static analysis and is fragile.

---

#### INT-P2-002: Misleading `import datetime as pd`

**Status**: NOT STARTED
**Severity**: High
**File**: `src/cascade_correlation/cascade_correlation.py` (line 38)

**Codebase Validation (2026-02-18)**: **CONFIRMED**. Line 38: `import datetime as pd`. The alias `pd` is conventionally used for `pandas`. Using it for `datetime` will mislead any developer reading the code.

---

#### INT-P2-003: `validate_training_results` Bug

**Status**: NOT STARTED
**Severity**: High
**File**: `src/cascade_correlation/cascade_correlation.py` (lines 2750-2843)

**Description**: Uninitialized variable when `max_epochs=0`.

**Codebase Validation (2026-02-18)**: **CONFIRMED**. TODO comment at line 2750 acknowledges the bug. Variable initialized as `None` at line 2754. If the training loop at line ~2810 doesn't execute (e.g., `max_epochs=0`), `validate_training_results` remains `None`. The guard at line 2830 (`if not validate_training_results:`) creates a default `ValidateTrainingResults`, but the debug log at line 2825 (inside the loop) would crash with `AttributeError` on `.early_stop` if executed with a None result from a prior iteration.

---

#### INT-P2-004: `snapshot_counter` Initialized Twice

**Status**: NOT STARTED
**Severity**: ~~High~~ **Low** (cosmetic)

**Codebase Validation (2026-02-18)**: **CONFIRMED — LOW IMPACT**. `self.snapshot_counter = 0` at both line 530 and line 548. Both set to 0, so the duplication is harmless. Just dead code to clean up.

---

#### INT-P2-005: `or` Fallback Chain Bug for Falsy Values

**Status**: NOT STARTED
**Severity**: Medium

**Description**: `learning_rate = param or default` treats `0` as falsy, using default instead of the intended value.

**Codebase Validation (2026-02-18)**: **CONFIRMED**. Two instances in `cascade_correlation.py` at lines 3015 and 3096: `pl.Path(self.cascade_correlation_network_snapshots_dir) or pl.Path(_CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR)` — A `Path` object is always truthy (even for empty string paths), so the fallback will never trigger. Also, numerous `or` fallback patterns across the file for numeric parameters (e.g., line 471 `self.config.log_file_path or ...`). Fix: use `if x is None` checks instead of `or`.

---

#### INT-P2-006: Boolean Parameter `clockwise` Uses `or` Pattern

**Status**: NOT STARTED
**Severity**: Medium
**File**: `src/spiral_problem/spiral_problem.py` (lines 597, 1249, 1408)

**Codebase Validation (2026-02-18)**: **CONFIRMED**. Three instances: `self.clockwise = clockwise or self.clockwise or _SPIRAL_PROBLEM_CLOCKWISE`. If `clockwise=False` is explicitly passed (meaning counter-clockwise), the `or` operator treats it as falsy and falls through to `self.clockwise` or the constant default. Fix: use `clockwise if clockwise is not None else ...`.

---

#### INT-P2-007: conftest.py Fast-Slow Mode Logic Inverted

**Status**: NOT STARTED
**Severity**: Medium
**File**: `src/tests/conftest.py` (lines 82-87)

**Codebase Validation (2026-02-18)**: **CONFIRMED — TWO BUGS**. (1) Line 82 (commented out) checks `== "1"` but the active line 83 checks `== "0"` — the env var check is inverted. (2) Both branches (fast-slow enabled and disabled) set `CASCOR_LOG_LEVEL` to `"WARNING"`, making the entire conditional dead code. The `_is_fast_mode` fixture at line 154 correctly checks `== "1"`, contradicting line 83.

---

#### INT-P2-008: `_roll_sequence_number` Memory Issue

**Status**: NOT STARTED
**Severity**: High
**File**: `src/cascade_correlation/cascade_correlation.py` (line 759)

**Codebase Validation (2026-02-18)**: **CONFIRMED**. Method at line 759 generates a discard list of random numbers at line 776 to "roll" the sequence number. For large sequence values, this creates a large in-memory list of discarded random numbers. The discard list size is bounded by `max_value` parameter (from `sequence_max_value`).

---

#### INT-P2-009: Inconsistent Queue Method Names Between Remote Clients

**Status**: NOT STARTED → **SCOPE CHANGED**
**Severity**: ~~High~~ **Medium**

**Codebase Validation (2026-02-18)**: **CONFIRMED**. `remote_client_0.py` registers and uses `get_tasks_queue()`/`get_done_queue()` (lines 35-36, 56-57). `remote_client.py` uses `get_task_queue()`/`get_result_queue()` (lines 72-73). Incompatible naming means they cannot connect to the same manager.

**Migration Impact (2026-02-24)**: `remote_client_0.py` is a legacy file superseded by `juniper-cascor-worker`. If `remote_client_0.py` is deleted (see INT-P0-004), this inconsistency disappears. If retained, the in-tree `remote_client.py` should be standardized with `juniper-cascor-worker`'s queue naming conventions. Severity reduced to Medium.

---

#### INT-P2-010: `os._exit()` Used Instead of `sys.exit()` in main.py

**Status**: NOT STARTED
**Severity**: Medium
**File**: `src/main.py` (lines 174, 177 — shifted from 142, 145 at time of 2026-02-18 audit)

**Codebase Validation (2026-02-18)**: **CONFIRMED**. `os._exit(1)` and `os._exit(2)` (originally lines 142/145; now lines 174/177). `os._exit()` bypasses all cleanup: finally blocks, atexit handlers, open file flushing. Should use `sys.exit()` unless intentionally bypassing cleanup.

**Migration Impact (2026-02-24)**: Note that `src/main.py` is the standalone CLI entry point (spiral problem evaluation), not the service entry point. The service uses `src/server.py` which starts uvicorn normally. This bug only affects CLI usage.

---

#### INT-P2-013: `check_object_pickleability` Depends on Undeclared `dill`

**Status**: ~~NOT STARTED~~ **RESOLVED**
**Severity**: ~~Medium~~ N/A

**Codebase Validation (2026-02-18)**: **RESOLVED**. `dill>=0.3.6` is declared in `pyproject.toml` (line 53). Used in `src/utils/utils.py:248` and test files. This issue has been fixed since the original audit.

---

#### INT-P2-014: Multiple `import traceback` Inside Exception Handlers

**Status**: NOT STARTED
**Severity**: Medium
**Description**: Should be top-level imports.

**Codebase Validation (2026-02-18)**: **CONFIRMED**. 21 instances of `import traceback` inside exception handlers in `cascade_correlation.py` (line numbers from 2026-02-18 audit: 1369, 1505, 1690, 2175, 2198, 2327, 2442, 2466, 2472, 2494, 2499, 2819, 2879, 2917, 2947, 3078, 3172, 3264, 3310, 3336, 3356). The top-level import at line 60 is commented out: `# import traceback`. Fix: uncomment the top-level import and remove the 21 local imports.

---

### 3.2 Code Coverage

#### CAS-REF-001: Code Coverage Below 90%

**Status**: IN PROGRESS
**Current**: ~67% overall (improved from ~50%)
**Target**: 90%
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md (P2-NEW-001)

**Migration Impact (2026-02-24)**: The CasCor Service API (Phase 2) added 213 unit tests + 13 integration tests for the new `src/api/` module. The `pyproject.toml` now enforces `fail_under = 80` for coverage. Overall coverage may have improved with the new API tests, but the core algorithm modules still need attention.

---

#### CAS-REF-002 / INT-P2-011: CI/CD Coverage Gates Not Enforced

**Status**: ~~NOT STARTED~~ **PARTIALLY RESOLVED**
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md (P2-NEW-002), INTEGRATION_ROADMAP-01.md

**Description**: Add `coverage report --fail-under=80` to CI. Configure per-module thresholds.

**Migration Impact (2026-02-24)**: **PARTIALLY RESOLVED**. The migrated CI workflow (`.github/workflows/ci.yml`) runs pytest with coverage, and `pyproject.toml` sets `fail_under = 80`. However, per-module thresholds have not been configured. The CI workflow uses `pip install -e ".[all]"` with CPU-only PyTorch.

**Remaining work**: Configure per-module coverage thresholds (e.g., higher bar for `cascade_correlation/` core, lower bar for `api/` endpoints).

---

### 3.3 Type Safety

#### CAS-REF-003 / INT-P2-012: Type Errors Gradual Fix

**Status**: IN PROGRESS
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md (P2-NEW-006), INTEGRATION_ROADMAP-01.md

**Description**: Run mypy, categorize errors, fix critical type errors in core modules. Gradually increase type strictness. Remove `continue-on-error` once stable.

**Sub-tasks** (all unchecked):

- Run mypy and categorize errors
- Fix critical type errors
- Gradually increase strictness
- Remove `continue-on-error` from CI

**Migration Impact (2026-02-24)**: The `pyproject.toml` configures mypy with `ignore_missing_imports = true` and excludes tests. The new `src/api/` module uses Pydantic models with type annotations, which should pass mypy cleanly. The core algorithm code remains the primary area needing type fixes.

---

### 3.4 Test Optimization

#### CAS-007: Optimize Slow Tests

**Status**: NOT STARTED
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md (Section 7.2)

**Description**: Numerous JuniperCascor tests are extremely slow (~45 minutes for complete run). Optimize for <= 5 minute test suite runtime.

**Migration Impact (2026-02-24)**: The CI now uses `scheduled-tests.yml` for long-running tests (with `--run-long` flag) separate from the main `ci.yml` pipeline. This separates fast feedback from thorough testing, partially addressing the CI bottleneck. However, the underlying slow tests still need optimization.

---

### 3.5 Legacy Code

#### CAS-REF-004 / INT-P3-001: Legacy Spiral Code Removal

**Status**: NOT STARTED
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md, INTEGRATION_ROADMAP-01.md

**Description**: Remove legacy spiral data generation code from JuniperCascor now that JuniperData provides this functionality via REST API. 16 methods currently carry `DeprecationWarning` (applied in CAS-INT-002).

**Dependencies**: JuniperData deployment and integration testing complete.

**Codebase Validation (2026-02-18)**: **CONFIRMED**. 16 `DeprecationWarning` instances found in `spiral_problem.py` at lines: 648, 684, 714, 733, 787, 813, 848, 869, 898, 941, 974, 1001, 1050, 1095, 1166, 1197. These mark the legacy methods scheduled for removal.

**Migration Impact (2026-02-24)**: JuniperData has been extracted to its own repo (`pcalnon/juniper-data`, 595 commits, CI green) and is running stably. `juniper-data-client` v0.3.0 is published to PyPI and declared as a dependency. The gate condition (JuniperData stability) is closer to being met, but formal E2E live-service integration tests (INT-P3-002) are still NOT STARTED.

---

## Section 4: Feature Enhancements

**Priority**: MEDIUM to LOW
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md (Section 7.2), IMPLEMENTATION_PROGRESS.md, CODE_REVIEW_SUMMARY.md

### 4.1 Training Control

#### CAS-002: Separate Epoch Limits

**Status**: NOT STARTED
**Module**: Cascor: Epoch Definition

**Description**: Separate epoch limits and epoch counts for full network and candidate nodes. Allows independent tuning of network training duration vs candidate pool training duration.

**Migration Impact (2026-02-24)**: The CasCor Service API's `POST /v1/training/start` endpoint accepts `epochs` and `params` parameters. When implementing separate epoch limits, the API contract will need to be extended to accept `network_epochs` and `candidate_epochs` (or similar). The `juniper-cascor-client`'s `start_training()` method will need a corresponding update.

---

#### CAS-003: Max Train Session Iterations

**Status**: NOT STARTED
**Module**: Cascor: Training Iterations

**Description**: Add max iterations (per train session) meta parameter that limits the total number of iterations for a given training session. Provides an upper bound on training duration, preventing runaway sessions.

**Migration Impact (2026-02-24)**: The `TrainingLifecycleManager` already tracks training state and could enforce iteration limits. This feature should be implemented in the lifecycle manager rather than in the core algorithm directly, as the service API provides the session boundary.

---

#### CAS-006: Auto-Snap Best Network (Accuracy Ratchet)

**Status**: NOT STARTED
**Module**: Cascor: Auto-Snap Best Network

**Description**: Automatically snapshot a network when a new best accuracy is achieved, after an initial count of full network epochs or training session iterations completed.

**Migration Impact (2026-02-24)**: The `TrainingMonitor` (in `src/api/lifecycle/monitor.py`) already has event-driven callbacks for `epoch_end` events. Auto-snap logic could be implemented as a monitor callback that triggers `snapshot_serializer.py` when accuracy improves. The deferred `/v1/snapshots/*` API endpoints would expose these snapshots to clients.

---

### 4.2 Algorithm Enhancements

#### ENH-006: Flexible Optimizer Management System

**Status**: NOT STARTED
**Source**: IMPLEMENTATION_PROGRESS.md, DEVELOPMENT_ROADMAP.md (P2-002)

**Description**: Implement flexible optimizer system with `OptimizerConfig` dataclass. Support Adam, SGD, RMSprop, AdamW. Includes serialization support.

---

#### ENH-007: N-Best Candidate Layer Selection

**Status**: NOT STARTED
**Source**: IMPLEMENTATION_PROGRESS.md, DEVELOPMENT_ROADMAP.md (P2-005)

**Description**: Config already has placeholders (`candidates_per_layer`, `layer_selection_strategy`). Implementation design completed in roadmap. High-value feature for architecture exploration.

---

### 4.3 Network Architecture

#### CAS-008: Network Hierarchy Management

**Status**: NOT STARTED
**Module**: Cascor: Network Hierarchy

**Description**: Add "Increment Network Hierarchy level" functionality and training/inference of multi-hierarchical CasCor networks.

---

#### CAS-009: Network Population Management

**Status**: NOT STARTED
**Module**: Cascor: Network Population

**Description**: Add "Network Population Initialization and Update" functionality to create a new population of CasCor neural networks at a given hierarchical level. Enables ensemble-style approaches with population-based training.

---

### 4.4 Storage & Serialization

#### CAS-010: Snapshot Vector DB Storage

**Status**: NOT STARTED
**Module**: Cascor: Snapshot Vector DB

**Description**: Store CasCor snapshots in a vector database, indexed by network UUID. Enables semantic search and comparison of network states across training runs.

---

### 4.5 Multiprocessing & Workers

#### CAS-004: Extract Remote Worker to JuniperBranch

**Status**: ~~NOT STARTED~~ **RESOLVED (by migration Phase 3, different name)**
**Module**: ~~Cascor: Candidate Remote Workers~~ N/A

**Description**: Refactor to extract the Remote Worker node and all its dependencies into a new application: JuniperBranch. Enables lightweight distributed training workers on heterogeneous hardware.

**Migration Impact (2026-02-24)**: **RESOLVED**. The polyrepo migration extracted the remote worker as `juniper-cascor-worker` (not `JuniperBranch` as originally planned). The package is published to PyPI as v0.1.0, has 44 tests (99% coverage), and provides a CLI entry point for standalone worker execution. The in-tree `src/remote_client/remote_client.py` still exists but is superseded.

**Remaining work**: Remove or archive the in-tree `src/remote_client/` directory since the standalone `juniper-cascor-worker` package now serves this role.

---

#### CAS-005: Extract Common Dependencies to Modules

**Status**: ~~NOT STARTED~~ **PARTIALLY RESOLVED**
**Module**: Cascor: Common Class Modules
**Dependencies**: ~~CAS-004 (JuniperBranch extraction must be planned first)~~ CAS-004 is resolved

**Description**: Refactor to extract all classes that are dependencies of both JuniperCascor and JuniperBranch into importable modules.

**Migration Impact (2026-02-24)**: The `juniper-cascor-worker` package includes its own copies of shared types. No formal shared module exists between `juniper-cascor` and `juniper-cascor-worker` yet. If these packages need to share code (e.g., `CandidateUnit` definition, activation function types), a shared dependency package should be created. This is a lower priority since the worker currently bundles everything it needs.

---

#### ENH-008: Worker Cleanup Improvements

**Status**: NOT STARTED
**Source**: IMPLEMENTATION_PROGRESS.md, CODE_REVIEW_SUMMARY.md (P2 #17)

**Description**: Better termination handling, SIGKILL fallback, improved logging, zombie process prevention.

**Migration Impact (2026-02-24)**: This applies to both the in-tree `remote_client.py` and the standalone `juniper-cascor-worker`. Any cleanup improvements should be made in the `juniper-cascor-worker` package, which is now the primary worker implementation.

---

## Section 5: Cross-Project Dependencies

### 5.1 JuniperCanopy Dependencies

#### CAS-CANOPY-001: Prediction Grid API for Decision Boundary

**Status**: ~~NOT STARTED~~ **RESOLVED (by migration Phase 2)**
**Priority**: ~~HIGH~~ N/A
**Source**: JuniperCanopy CAN-CRIT-001

**Description**: JuniperCanopy's decision boundary visualization requires the real CasCor backend to accept a grid of input points and return predictions. Currently only has a demo mode implementation. The CasCor backend must expose a prediction method that accepts arbitrary input grids.

**Migration Impact (2026-02-24)**: **RESOLVED**. The CasCor Service API includes `GET /v1/decision-boundary` (with `resolution` parameter) implemented in `src/api/routes/decision_boundary.py`. The `juniper-cascor-client` exposes `client.get_decision_boundary(resolution=50)`. The `CascorServiceAdapter` wraps this as `get_decision_boundary()`. This unblocks CAN-CRIT-001 in JuniperCanopy.

**Impact**: Blocks CAN-CRIT-001 in JuniperCanopy. → **UNBLOCKED**.

---

#### CAS-CANOPY-002: Serialization API for Training Snapshots

**Status**: NOT STARTED → **PARTIALLY ADDRESSED, SCOPE CHANGED**
**Priority**: HIGH
**Source**: JuniperCanopy CAN-CRIT-002

**Description**: JuniperCanopy needs `save_snapshot()` and `load_snapshot()` methods in `CascorIntegration`. These require CasCor to expose a serialization API (e.g., PyTorch `state_dict()` export/import). The API must capture full training state including network weights, optimizer state, and training metadata.

**Migration Impact (2026-02-24)**: **SCOPE CHANGED**. `CascorIntegration` is being removed. Snapshot functionality must now be exposed as REST endpoints on the CasCor service:

- The migration plan lists `/v1/snapshots/*` (4 endpoints) as **deferred** — not yet implemented
- The existing `src/snapshots/snapshot_serializer.py` provides the underlying HDF5 save/load functionality
- The implementation path is: create REST routes in `src/api/routes/snapshots.py` that wrap `snapshot_serializer.py`, then add `save_snapshot()`/`load_snapshot()` methods to `juniper-cascor-client`, then add corresponding methods to `CascorServiceAdapter`

**Impact**: Still blocks CAN-CRIT-002 and downstream CAN-014/CAN-015 (snapshot replay features). This is now a CasCor service API feature rather than a `CascorIntegration` method.

---

#### CAS-CANOPY-003: Profiling Roadmap Tasks

**Status**: NOT STARTED
**Priority**: MEDIUM
**Source**: PRE-DEPLOYMENT_ROADMAP.md (Profiling Roadmap section)

**Description**: 16 profiling tasks from the pre-deployment roadmap remain unchecked:

- Profile full training loop (CPU, memory, I/O)
- Profile candidate pool generation
- Profile network forward/backward passes
- Profile serialization/deserialization
- Identify and optimize hot paths
- Memory leak detection in long-running training sessions
- Threading overhead analysis
- Benchmark comparison with reference implementations

**Migration Impact (2026-02-24)**: Profiling is now more important given the service architecture. Network latency between Canopy and CasCor adds overhead that didn't exist with in-process calls. Profiling should include:
- HTTP request/response overhead for training control operations
- WebSocket streaming latency for metrics relay
- `TrainingLifecycleManager` thread pool overhead
- Memory usage in long-running service mode (vs. short-lived CLI mode)

---

### 5.2 JuniperData Dependencies

#### Phase 5: Extended Data Sources

**Status**: DEFERRED
**Source**: JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md

**Description**: Extended data source support (S3, database, HuggingFace). Deferred until JuniperData core is stable.

**Migration Impact (2026-02-24)**: JuniperData is now a standalone repo (`pcalnon/juniper-data`, 595 commits, CI green, v0.5.0). Core stability is established. However, the extended data sources are still deferred pending use-case prioritization.

---

#### E2E Live Service Integration Tests

**Status**: NOT STARTED (INT-P3-002)
**Source**: INTEGRATION_ROADMAP-01.md

**Description**: No automated tests currently spin up a live JuniperData service. All current E2E tests use in-process `TestClient`.

**Migration Impact (2026-02-24)**: **SCOPE EXPANDED**. With the polyrepo architecture, E2E testing now requires spinning up multiple services (Data on 8100, CasCor on 8200). The migration Phase 6 plans a full-stack integration test suite and Docker Compose orchestration. This item should be addressed as part of Phase 6 rather than in isolation.

---

## Section 6: Infrastructure & DevOps (P3)

**Priority**: STANDARD
**Estimated Effort**: 16-28 hours (reduced from 20-32: some items partially addressed by migration)
**Source**: INTEGRATION_ROADMAP-01.md, PRE-DEPLOYMENT_ROADMAP-2.md, ORACLE_ANALYSIS_SCRIPTS.md

### 6.1 Shell Scripts

#### Shell Script Path Resolution Fixes

**Status**: NOT STARTED
**Source**: ORACLE_ANALYSIS_SCRIPTS.md, DEVELOPMENT_ROADMAP.md (P0-005)

**Description**: Multiple shell script issues:

- `juniper_cascor.bash` (lines 61-63) overrides absolute paths with bare filenames
- `script_util.cfg` naming inconsistencies (`DATE_FUNCTIONS_NAME` vs actual filenames)
- Helper script resolution broken
- `GET_PROJECT_SCRIPT` pattern needs fixing across all scripts

**Migration Impact (2026-02-24)**: Shell scripts in `util/` may contain references to the old monorepo directory structure. After the polyrepo migration, `BASE_DIR` and `SOURCE_DIR` variables need to resolve correctly within the standalone `juniper-cascor` repo. Validate all path references assume the polyrepo layout (`juniper-cascor/` as root, not `Juniper/JuniperCascor/juniper_cascor/`).

---

### 6.2 Docker & Deployment

#### INT-P3-003: Docker Compose Validation

**Status**: NOT STARTED → **SCOPE CHANGED**
**Source**: INTEGRATION_ROADMAP-01.md

**Description**: The docker-compose configuration shows a 3-service deployment but has not been tested end-to-end.

**Migration Impact (2026-02-24)**: **SCOPE CHANGED**. The migration Phase 6 plans a `juniper-deploy` or `juniper-infra` repo with a full `docker-compose.yml` orchestrating three independent services with health check dependency ordering (CasCor depends on Data; Canopy depends on CasCor). The existing `conf/docker-compose.yaml` in the CasCor repo likely references the old monorepo structure and needs to be updated or replaced. This item should be coordinated with Phase 6 rather than addressed independently.

---

### 6.3 Git Hygiene

#### INT-P3-008: .pytest.ini.swp and Coverage Files in Git

**Status**: NOT STARTED → **LIKELY RESOLVED**
**Source**: INTEGRATION_ROADMAP-01.md

**Description**: Vim swap file and coverage artifacts appear in git status. Add to `.gitignore`.

**Migration Impact (2026-02-24)**: The polyrepo migration Phase 0 updated `.gitignore` and removed tracked build artifacts (`.coverage`, `coverage.xml`, `results.xml`, `egg-info/`). Verify that the `.gitignore` in the standalone `juniper-cascor` repo covers these patterns.

---

#### INT-P3-009: Version Strings Inconsistent Across Files

**Status**: NOT STARTED → **SCOPE CHANGED**
**Source**: INTEGRATION_ROADMAP-01.md

**Description**: File headers show various versions: `0.3.1 (0.7.3)`, `0.3.2 (0.7.3)`, `0.3.12`, `0.3.16`, `0.4.1`, `0.7.3` (bare), `1.0.1`, `0.1.0`. CLAUDE.md states `0.6.6 (0.7.3)`. API module files (`src/api/`) have no version headers.

**Migration Impact (2026-02-24)**: The canonical version is now in `pyproject.toml` (currently `0.3.17`). The CasCor Service API uses `0.4.0` in its response envelope metadata. File header versions should be reconciled to match `pyproject.toml`. Consider using a single-source-of-truth version (e.g., `importlib.metadata.version("juniper-cascor")`) instead of file header strings.

---

#### INT-P3-010: `cascor_snapshots` vs `snapshots` Directory Confusion

**Status**: NOT STARTED
**Source**: INTEGRATION_ROADMAP-01.md

**Description**: Both `src/cascor_snapshots/` and `src/snapshots/` directories exist under `src/`. Purpose distinction is unclear.

---

### 6.4 Profiling Infrastructure

#### Continuous Profiling (Grafana Pyroscope)

**Status**: NOT STARTED
**Effort**: Large (1-2 weeks)
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md (P3-NEW-004)

**Description**: Deploy Grafana Pyroscope, integrate SDK, create dashboards.

---

#### Baseline Performance Profiles

**Status**: DEFERRED (INT-P3-006)
**Source**: INTEGRATION_ROADMAP-01.md

**Description**: Create baseline py-spy profiles for key operations to enable performance regression detection.

---

#### Profiling in CI/CD

**Status**: DEFERRED (INT-P3-007)
**Source**: INTEGRATION_ROADMAP-01.md

**Description**: Add profiling to CI/CD pipeline for regression detection.

---

### 6.5 WebSocket Testing

#### INT-P3-005: Test WebSocket Responsiveness During Training

**Status**: ~~NEEDS MANUAL VERIFICATION~~ **PARTIALLY ADDRESSED**
**Source**: INTEGRATION_ROADMAP-01.md

**Description**: When Cascor training runs via `asyncio.run_in_executor()` in FastAPI, WebSocket responsiveness should be verified under load.

**Migration Impact (2026-02-24)**: The CasCor Service API test suite includes dedicated WebSocket test files: `test_websocket_control.py`, `test_websocket_manager.py`, `test_websocket_messages.py`, `test_websocket_training_stream.py`, and `test_websocket_streaming.py`. However, these may not test responsiveness under actual training load. Integration testing with a real training session is still needed as part of migration Phase 4 validation.

---

## Section 7: Deferred & Future Work (P4)

### 7.1 Major Architectural Changes

#### GPU/CUDA Support

**Status**: NOT STARTED
**Effort**: XL (2-4 weeks)
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md (P3-NEW-003)

**Description**: Add CUDA/GPU acceleration for training. Requires significant refactoring.

**Migration Impact (2026-02-24)**: The CI now uses CPU-only PyTorch (`--index-url https://download.pytorch.org/whl/cpu`). GPU support would need a separate CI matrix or scheduled workflow with GPU runners. The `juniper-cascor-worker` package already depends on full PyTorch and could benefit from GPU acceleration for distributed candidate training.

---

#### Large File Refactoring

**Status**: NOT STARTED
**Source**: DEVELOPMENT_ROADMAP.md (P3-002, P3-003)

**Description**: Phase 1: Extract multiprocessing, training, validation helpers (no file > 2000 lines). Phase 2: Deeper restructuring (no file > 1500 lines).

---

### 7.2 Code Cleanup

#### Remove "Roll" Concept in CandidateUnit

**Status**: DEFERRED
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md (CASCOR-P1-008)

**Description**: Consider removing the roll concept entirely from CandidateUnit. Currently capped at `MAX_ROLL_COUNT = 10000`.

---

#### Candidate Factory Refactor

**Status**: DEFERRED
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md (P3-001)

**Description**: Ensure all candidate creation routes through `_create_candidate_unit()` factory.

---

#### MED-014: Line Length Reduction

**Status**: DEFERRED
**Source**: TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md

**Description**: Full codebase reformat to 120-character line length.

---

#### Code Cleanup Items (Low Priority)

**Source**: INTEGRATION_ROADMAP-01.md (INT-P4-012 through INT-P4-016)

| ID         | Description                                                      | Status |
| ---------- | ---------------------------------------------------------------- | ------ |
| INT-P4-012 | `LogConfig.__init__` parameter naming cleanup                    | LOW    |
| INT-P4-013 | Logger TODO cleanup                                              | LOW    |
| INT-P4-014 | Remove commented-out code blocks                                 | LOW    |
| INT-P4-015 | Clean up "Original corrupted line" comments in spiral_problem.py | LOW    |
| INT-P4-016 | Remove `uuid as uuid` redundant import alias                     | LOW    |

---

### 7.3 In-Code TODOs

**Source**: DEVELOPMENT_ROADMAP.md

| File                   | Line  | Description                     | Priority            |
| ---------------------- | ----- | ------------------------------- | ------------------- |
| cascade_correlation.py | ~577  | CUDA random seeding             | P3                  |
| cascade_correlation.py | ~925  | Refactor repeated code          | P3                  |
| cascade_correlation.py | ~1381 | Convert to proper constants     | P3                  |
| cascade_correlation.py | ~2314 | validate_training_results bug   | P2 (see INT-P2-003) |
| snapshot_serializer.py | ~756  | Extend optimizer support        | P2                  |
| spiral_problem.py      | ~482  | Restore scaling functionality   | P3                  |
| log_config.py          | ~78   | Clean up logging initialization | P3                  |
| utils.py               | ~232  | Fix broken function             | P2                  |

---

### 7.4 Documentation Enhancements

**Source**: DOCUMENTATION_AUDIT.md

| Item            | Description                                | Status |
| --------------- | ------------------------------------------ | ------ |
| Auto-generation | Consider MkDocs/Sphinx for API docs        | Future |
| Link checking   | Add CI step to verify documentation links  | Future |
| Versioning      | Add version selector for multiple versions | Future |
| Search          | Add search functionality to docs           | Future |
| Examples        | Add more code examples and tutorials       | Future |

**Migration Impact (2026-02-24)**: The CasCor service API could benefit from auto-generated API documentation (FastAPI provides built-in OpenAPI/Swagger at `/docs`). This may partially fulfill the auto-generation item for the API surface.

---

### 7.5 Multiprocessing Deferred Items

**Source**: PRE-DEPLOYMENT_ROADMAP-2.md, INTEGRATION_ROADMAP-01.md

| ID         | Description                                              | Status      |
| ---------- | -------------------------------------------------------- | ----------- |
| INT-P4-010 | Add metrics for multiprocessing fallback frequency       | DEFERRED    |
| INT-P4-011 | Test fallback under various failure modes                | DEFERRED    |
| ENH-009    | Per-instance queue management (complex refactor)         | NOT STARTED |
| ENH-010    | Process-based plotting (depends on BUG-002 verification) | NOT STARTED |

---

## Section 8: Verification Items

These items require manual testing to confirm proper operation.

### P4-NEW-001: Execute main.py End-to-End with Plotting

**Status**: NOT VERIFIED
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md

**Description**: Run `python main.py` with default settings, verify plotting subprocess.

---

### P4-NEW-002: Test ./try Script Launch

**Status**: NOT VERIFIED
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md

**Description**: Test `./try` script successfully launches `main.py`.

---

### P4-NEW-005: Verify Parallel Processing Working

*Note: P4-NEW-003 and P4-NEW-004 from PRE-DEPLOYMENT_ROADMAP-2.md were completed prior to the 2026-02-18 audit and are not included in this roadmap.*

**Status**: NEEDS TESTING
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md

**Description**: Run with DEBUG logging, check for parallel vs sequential execution.

---

### NEW: Verify CasCor Service API End-to-End

**Status**: NOT VERIFIED
**Source**: POLYREPO_MIGRATION_PLAN.md (Phase 4 remaining work)
**Priority**: HIGH

**Description**: Start CasCor service (`python -m server`), connect via `juniper-cascor-client`, and verify: network creation, training start/stop, metrics WebSocket streaming, topology retrieval, decision boundary computation. This is a prerequisite for Phase 4 integration testing.

---

### NEW: Verify Three-Mode Activation in Canopy

**Status**: NOT VERIFIED
**Source**: DECOUPLE_CANOPY_FROM_CASCOR_PLAN.md (Section 4)
**Priority**: HIGH

**Description**: Verify Canopy's three-mode activation:
1. `CASCOR_DEMO_MODE=1` → Demo mode works identically to pre-migration
2. `CASCOR_SERVICE_URL=http://localhost:8200` → Service mode connects via `CascorServiceAdapter`
3. `CASCOR_BACKEND_PATH=...` → Legacy mode works during transition

---

## Section 9: Oracle Analysis Recommendations

Items from the 2026-01-26 Oracle analyses that may or may not have been addressed.

### Oracle Python Analysis (5 Items)

**Source**: ORACLE_ANALYSIS_PYTHON.md

| #   | Description                                                                     | Status           |
| --- | ------------------------------------------------------------------------------- | ---------------- |
| 1   | `CandidateUnit.train()` return type mismatch; needs `train_detailed()` refactor | NEEDS VALIDATION |
| 2   | `CandidateTrainingManager.start()` does not accept `method` parameter           | NEEDS VALIDATION |
| 3   | `ValidationError` not a subclass of `ValueError`                                | NEEDS VALIDATION |
| 4   | Residual error validation rejects empty tensors                                 | NEEDS VALIDATION |
| 5   | `fit()` method uses `max_epochs` but tests call with `epochs`                   | NEEDS VALIDATION |

### Oracle Scripts Analysis (6 Items)

**Source**: ORACLE_ANALYSIS_SCRIPTS.md

| #   | Description                                                    | Status      |
| --- | -------------------------------------------------------------- | ----------- |
| 1   | Fix helper script resolution in `juniper_cascor.bash`          | NOT STARTED |
| 2   | Verify helper filenames match actual files in `util/`          | NOT STARTED |
| 3   | Confirm BASE_DIR and SOURCE_DIR resolve correctly              | NOT STARTED |
| 4   | Fix CURRENT_OS and date helper sourcing; verify permissions    | NOT STARTED |
| 5   | Fix naming inconsistencies in `script_util.cfg`                | NOT STARTED |
| 6   | Grep for `GET_PROJECT_SCRIPT` pattern and fix in other scripts | NOT STARTED |

**Migration Impact (2026-02-24)**: Oracle Scripts items 3 and 6 are particularly important post-migration. `BASE_DIR` and `SOURCE_DIR` must resolve within the standalone `juniper-cascor` repo, not assume the old monorepo layout.

---

## Section 10: Completed Items (Reference)

These items are documented as COMPLETE and included for reference only.

### CAS-001: Extract Spiral Generator to JuniperData

**Status**: COMPLETE
**Resolution**: JuniperData v0.4.2 provides 8 generators including spiral via REST API.

### CAS-INT-001 through CAS-INT-009: JuniperData Integration

**Status**: ALL 9 COMPLETE
**Resolution**: Mandatory API path, deprecation warnings, API key auth, NPZ validation, test updates, contract tests, E2E integration tests, retry/backoff, config validation.

### CAS-REF-005: RemoteWorkerClient Integration

**Status**: COMPLETE (but **NOTE**: the integration target `CascorIntegration` is being removed)
**Resolution**: Added to `CascorIntegration` class with REST API endpoints.

**Migration Impact (2026-02-24)**: This completed work will be superseded when `CascorIntegration` is deleted in Phase 4 Step 5.8. The functionality is replaced by server-side worker management in the CasCor service and the standalone `juniper-cascor-worker` package.

### INT-P1-007: Connection Retry Logic

**Status**: COMPLETE (via CAS-INT-008)

### All INTEGRATION_ROADMAP.md Critical Blockers

**Status**: ALL RESOLVED

### TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN Phases 0-4

**Status**: COMPLETE (MED-014 deferred)

### PRE-DEPLOYMENT_ROADMAP P0 Critical Issues (6 items)

**Status**: ALL FIXED

### INT-P1-002: `requests` as Undeclared Dependency

**Status**: RESOLVED (pre-migration, confirmed 2026-02-18; further resolved by INT-P1-001)
**Resolution**: Originally resolved by declaring `requests` in `conf/requirements-pip.txt`. Fully resolved by migration: the vendored `JuniperDataClient` that required `requests` was removed (INT-P1-001); `requests` is now a transitive dependency of the external `juniper-data-client` package and no longer needed directly by `juniper-cascor`.

### INT-P2-013: `check_object_pickleability` Depends on Undeclared `dill`

**Status**: RESOLVED (pre-migration, confirmed 2026-02-18)
**Resolution**: `dill>=0.3.6` declared in `pyproject.toml`. Used in `src/utils/utils.py` and test files.

### COMPLETE_FIX_SUMMARY P0/P1/P2 (19 items)

**Status**: ALL RESOLVED

### Items Resolved by Polyrepo Migration

*5 resolved work items + 1 substantially resolved work item (INT-P1-004) + 2 design decisions. Design decisions are not counted in the 83 non-completed work items statistic.*

| Item | Resolution | Migration Phase |
| --- | --- | --- |
| INT-P1-001: Duplicated JuniperDataClient | `juniper-data-client` v0.3.0 on PyPI | Phase 1 |
| INT-P1-004: Full IPC Architecture | CasCor Service API + `juniper-cascor-client` (substantially resolved; Canopy integration testing pending) | Phases 2-3 |
| C.1: Async Wrapper for fit() | `TrainingLifecycleManager` with ThreadPoolExecutor | Phase 2 |
| C.2: Expose RemoteWorkerClient | `juniper-cascor-worker` on PyPI; server-side management | Phase 3 |
| CAS-004: Extract Remote Worker | Published as `juniper-cascor-worker` v0.1.0 | Phase 3 |
| CAS-CANOPY-001: Prediction Grid API | `GET /v1/decision-boundary` endpoint | Phase 2 |
| Design Decision 4: Shared Client Arch | PyPI packages (Option A implemented directly) | Phase 1 |
| Design Decision 5: Async Training | `TrainingLifecycleManager` (Option A implemented) | Phase 2 |

---

## Development Phases (Proposed — Updated for Post-Migration)

Based on codebase validation results, dependency analysis, effort estimates, and polyrepo migration impact.

### Phase 0: Critical Bug Fixes (1-2 days)

**Goal**: Fix all confirmed P0 bugs that cause silent incorrect behavior.
**No external dependencies.**
**Note**: Items 4-5 (INT-P0-004, INT-P0-005) were downgraded from Critical to Low after the migration made them trivial deletions, but remain in Phase 0 as quick wins (15 min each) best done alongside the actual critical fixes.

| #   | Item                                                            | Effort  | Notes                                                                |
| --- | --------------------------------------------------------------- | ------- | -------------------------------------------------------------------- |
| 1   | INT-P0-001: Fix walrus operator precedence (line 1322)          | 15 min  | Add parentheses                                                      |
| 2   | INT-P0-003: Fix CandidateUnit params in `fit` (lines 1154-1166) | 1 hr    | Change `_CandidateUnit__` → `CandidateUnit__`, remove invalid params |
| 3   | INT-P0-002: Extract `ActivationWithDerivative` to shared module | 2-3 hrs | Create `src/activation/`, update imports in both files               |
| 4   | INT-P0-004: Delete `remote_client_0.py` (legacy)               | 15 min  | Superseded by `juniper-cascor-worker` (**updated**: delete, don't fix) |
| 5   | INT-P0-005: Remove `sys.path` lines in test file               | 15 min  | Dead code in polyrepo layout (**updated**: remove, don't fix)        |
| 6   | INT-P2-002: Fix `import datetime as pd` alias                   | 15 min  | Rename to `dt` or `datetime`                                         |
| 7   | INT-P2-004: Remove duplicate `snapshot_counter` init            | 5 min   | Delete line 548                                                      |
| 8   | INT-P2-014: Move `import traceback` to top-level                | 30 min  | Uncomment line 60, remove 21 local imports                           |
| 9   | INT-P2-010: Replace `os._exit()` with `sys.exit()` in main.py   | 15 min  | Lines 174, 177                                                       |

**Estimated Total**: 4-6 hours

### Phase 1: Logic Bug Fixes (2-3 days)

**Goal**: Fix all confirmed logic bugs that cause incorrect behavior in specific conditions.
**Depends on**: Phase 0 complete.

| #   | Item                                                               | Effort  | Notes                                                |
| --- | ------------------------------------------------------------------ | ------- | ---------------------------------------------------- |
| 1   | INT-P2-005: Fix `or` fallback patterns for falsy values            | 2-3 hrs | Audit all `or` patterns, replace with `if x is None` |
| 2   | INT-P2-006: Fix boolean `clockwise` or-pattern                     | 1 hr    | 3 instances in spiral_problem.py                     |
| 3   | INT-P2-003: Fix `validate_training_results` uninitialized variable | 1-2 hrs | Handle `max_epochs=0` edge case                      |
| 4   | INT-P2-007: Fix conftest fast-slow mode logic                      | 30 min  | Fix inverted check, differentiate branches           |
| 5   | INT-P2-008: Fix `_roll_sequence_number` memory issue               | 1-2 hrs | Use generator instead of list for discards           |
| 6   | INT-P2-009: Standardize queue names (if remote_client.py retained) | 1 hr    | Align with `juniper-cascor-worker` conventions       |
| 7   | INT-P1-005: Fix `main.py` unused `spiral_config` parameter         | 15 min  | Remove the dead parameter                            |
| 8   | INT-P1-008: Remove stale `check.py` duplicate                      | 15 min  | Delete file, verify no references                    |
| 9   | INT-P2-001: Properly declare `shared_object_dict` at module scope  | 30 min  | Add module-level declaration                         |
| 10  | Clean up `src/remote_client/` directory (CAS-004 follow-up)       | 30 min  | **NEW**: Archive or remove in-tree remote client code superseded by `juniper-cascor-worker` |

**Estimated Total**: 8-12 hours

### Phase 2: Code Quality & Testing (1-2 weeks)

**Goal**: Improve code quality, test coverage, and type safety.
**Depends on**: Phase 1 complete.

| #   | Item                                          | Effort   | Notes                                           |
| --- | --------------------------------------------- | -------- | ----------------------------------------------- |
| 1   | CAS-REF-001: Increase code coverage to 90%    | 3-5 days | Currently ~67%; new API tests may help           |
| 2   | CAS-REF-003: Fix critical type errors (mypy)  | 2-3 days | API module is well-typed; focus on core modules  |
| 3   | CAS-REF-002: Add per-module CI coverage gates  | 1 day    | Global 80% is enforced; add per-module thresholds |
| 4   | CAS-007: Optimize slow tests (target ≤ 5 min) | 2-3 days | Profile and optimize; scheduled tests handle long runs |
| 5   | CAS-REF-004: Remove 16 legacy spiral methods  | 1 day    | JuniperData stable on PyPI; full completion gated on Phase 3 #3 (INT-P3-002 E2E tests) |

**Estimated Total**: 10-15 days (item sum: 9-13 days + coordination/integration overhead)

### Phase 3: Integration Architecture (1-2 weeks) — REDUCED SCOPE

**Goal**: Address remaining cross-project dependencies and integration concerns.
**Depends on**: Phase 2 complete. Coordination with JuniperData and JuniperCanopy teams.

**Note**: This phase was originally estimated at 2-4 weeks. The polyrepo migration resolved the two largest items (INT-P1-001 shared client, C.1 async wrapper). The remaining items focus on API completeness for Canopy consumption.

| #   | Item                                                    | Effort   | Notes                                             |
| --- | ------------------------------------------------------- | -------- | ------------------------------------------------- |
| 1   | CAS-CANOPY-002: Snapshot REST API endpoints             | 3-5 days | Create `/v1/snapshots/*` routes wrapping serializer (core 4 endpoints; broader API set in Phase 5 #10) |
| 2   | INT-P1-006: Add import guard for SpiralDataProvider     | 1 hr     | Low effort, include with other work               |
| 3   | INT-P3-002: E2E live-service integration tests          | 2-3 days | Coordinate with Phase 6 Docker Compose            |
| 4   | CAS-005: Evaluate shared types with `juniper-cascor-worker` | 1-2 days | Determine if shared package needed |

**Estimated Total**: 7-12 days (item sum: ~6-11 days + coordination/integration overhead)

### Phase 4: Feature Enhancements (4-8 weeks)

**Goal**: Implement new features and algorithm improvements.
**Depends on**: Phase 3 complete. Can be parallelized.

| #   | Item                                          | Effort    | Notes                                                    |
| --- | --------------------------------------------- | --------- | -------------------------------------------------------- |
| 1   | CAS-002: Separate epoch limits                | 2-3 days  | Extend API contract for `network_epochs`/`candidate_epochs` |
| 2   | CAS-003: Max train session iterations         | 1-2 days  | Implement in `TrainingLifecycleManager`                  |
| 3   | CAS-006: Auto-snap best network               | 2-3 days  | Implement as `TrainingMonitor` callback                  |
| 4   | ENH-006: Flexible optimizer management        | 3-5 days  |                                                          |
| 5   | ENH-007: N-best candidate layer selection     | 3-5 days  |                                                          |
| 6   | ENH-008: Worker cleanup improvements          | 2-3 days  | Apply to `juniper-cascor-worker` package                 |

**Estimated Total**: 15-25 days (item sum: 13-21 days + integration overhead; reduced from 20-35: C.2 resolved)

### Phase 5: Infrastructure & Future (Ongoing)

**Goal**: Infrastructure improvements, shell script fixes, deferred items.
**No hard dependencies. Can proceed in parallel with Phase 4.**

| #   | Item                                        | Effort    | Notes                                                         |
| --- | ------------------------------------------- | --------- | ------------------------------------------------------------- |
| 1   | Shell script path fixes (6 Oracle items)    | 2-3 days  | Validate paths in polyrepo layout                             |
| 2   | INT-P3-003: Docker Compose validation       | 1-2 days  | Coordinate with migration Phase 6 `juniper-deploy` repo       |
| 3   | INT-P3-008: Git hygiene (.gitignore)        | 1 hr      | **LIKELY RESOLVED** — verify polyrepo .gitignore covers artifacts |
| 4   | INT-P3-009: Version string consistency      | 1-2 hrs   | Reconcile to pyproject.toml `0.3.17` / API `0.4.0`           |
| 5   | INT-P3-010: Snapshot directory confusion    | 1 hr      |                                                               |
| 6   | Large file refactoring                      | 1-2 weeks |                                                               |
| 7   | CAS-008/009: Network hierarchy & population | 4-8 weeks |                                                               |
| 8   | CAS-010: Snapshot Vector DB                 | 2-4 weeks |                                                               |
| 9   | GPU/CUDA support                            | 2-4 weeks | CI needs GPU runner or separate workflow                      |
| 10  | Implement deferred API endpoints (CAS-CANOPY-002, C.2 follow-up) | 1-2 weeks | **NEW**: `/v1/workers/*` (5), `PUT /v1/training/params`, plus any snapshot endpoints beyond Phase 3 core |

**Estimated Total**: Not aggregated — Phase 5 is ongoing with items spanning days to weeks; items are independent and can be prioritized individually.

---

## High-Level Design Analysis

### Design Decision 1: ActivationWithDerivative Extraction (INT-P0-002)

**Context**: The `ActivationWithDerivative` class is duplicated in `cascade_correlation.py` (line 291) and `candidate_unit.py` (line 138). Both files define the same class with the same `ACTIVATION_MAP`. The class wraps activation functions to also compute derivatives, and it must be picklable for multiprocessing.

**Option A: New shared module** (Recommended)

- Create `src/activation/activation_with_derivative.py`
- Import in both `cascade_correlation.py` and `candidate_unit.py`
- Pros: Clean separation, single source of truth, easy to test independently
- Cons: New directory/module to maintain

**Option B: Keep in `candidate_unit.py`, import in `cascade_correlation.py`**

- Since CandidateUnit is the primary consumer
- Pros: No new directory
- Cons: Creates circular dependency risk (cascade_correlation imports candidate_unit which is used by cascade_correlation)

**Option C: Move to `cascor_constants/` with activation constants**

- Co-locate with existing `constants_activation/`
- Pros: Groups activation-related code
- Cons: Constants module isn't meant for classes with behavior

**Recommendation**: Option A. Clean module boundary, no circular dependency risk.

**Migration Impact (2026-02-24)**: No change. This is internal to the CasCor codebase and unaffected by repo structure. However, if the `ActivationWithDerivative` class is needed by `juniper-cascor-worker` for candidate training, it should be exported as part of a shared types package (see CAS-005).

---

### Design Decision 2: Constructor Parameter Pattern Fix (INT-P0-003)

**Context**: The `fit()` method at line 1154 uses `_CandidateUnit__` prefix (Python name-mangling convention for private access from outside the class) instead of `CandidateUnit__` (the actual parameter names). The `_create_candidate_unit` factory at line 989 uses the correct `CandidateUnit__` prefix.

**Option A: Fix prefix in `fit()` + use factory** (Recommended)

- Route the `fit()` CandidateUnit creation through `_create_candidate_unit()` factory
- Ensures consistent parameter passing across all call sites
- Also fixes the 3 invalid params (`candidate_pool_size`, `log_file_name`, `log_file_path`)
- Pros: DRY, consistent, factory already exists
- Cons: May need to extend factory to support `fit()`-specific params

**Option B: Just fix the prefix:**

- Change `_CandidateUnit__` → `CandidateUnit__` at lines 1154-1166
- Remove the 3 invalid parameters
- Pros: Minimal change
- Cons: Doesn't address the duplication of constructor call patterns

**Option C: Refactor CandidateUnit to not use `**kwargs`**

- Remove `**kwargs` from `CandidateUnit.__init__` so invalid params raise `TypeError`
- Pros: Prevents silent parameter absorption in the future
- Cons: Potentially breaks existing call sites that rely on kwargs absorption

**Recommendation**: Option A + Option C together. Fix the factory pattern AND remove `**kwargs` to prevent future silent failures.

---

### Design Decision 3: `or` Fallback Pattern Systematic Fix (INT-P2-005, INT-P2-006)

**Context**: The codebase uses `value = param or default` extensively. This fails for falsy values: `0`, `False`, `""`, `0.0`. The `clockwise` boolean and numeric parameters like `learning_rate` are affected.

**Option A: `if x is None` pattern** (Recommended)

- Replace each `x or default` with `x if x is not None else default`
- Pros: Explicit, handles all falsy values correctly
- Cons: More verbose; need to audit every `or` pattern

**Option B: Sentinel value pattern:**

- Use `_SENTINEL = object()` as default, check `if param is not _SENTINEL`
- Pros: Distinguishes "not provided" from "explicitly None"
- Cons: More complex, overkill for this codebase

**Option C: Optional type with `get_or_default()` helper**

- Create a utility function `def get_or_default(value, default): return value if value is not None else default`
- Pros: Consistent, reusable
- Cons: Adds abstraction for a simple pattern

**Recommendation**: Option A for simplicity. Do a global audit of `or` patterns in parameter initialization. Estimated 30-40 instances to review.

---

### Design Decision 4: Shared Client Package Architecture (INT-P1-001, INT-P1-003)

**Status**: **IMPLEMENTED (Option A)**

**Context**: `JuniperDataClient` was duplicated in JuniperCascor and JuniperCanopy. Three applications shared API contracts with no shared package.

**Migration Resolution (2026-02-24)**: The polyrepo migration implemented **Option A (PyPI packages)** directly, bypassing the recommended short-term Option D (manual sync + CI drift check):

- `juniper-data-client` v0.3.0 published to PyPI
- `juniper-cascor-client` v0.1.0 published to PyPI
- `juniper-cascor-worker` v0.1.0 published to PyPI

All vendored copies removed. This design decision is resolved.

---

### Design Decision 5: Async Training Wrapper (C.1)

**Status**: **IMPLEMENTED (Option A)**

**Context**: `CascadeCorrelationNetwork.fit()` is synchronous and blocks the FastAPI event loop when called from JuniperCanopy's web endpoints.

**Migration Resolution (2026-02-24)**: The polyrepo migration implemented **Option A (`ThreadPoolExecutor`)** in the CasCor Service API:

- `TrainingLifecycleManager` (579 lines) uses `ThreadPoolExecutor` with `max_workers=1`
- `TrainingStateMachine` provides formal FSM state management
- `WebSocketManager.broadcast_from_thread()` handles the async/sync bridge
- Cancellation via state machine transitions (STARTED → STOPPED)

This design decision is resolved.

---

### Design Decision 6: Large File Refactoring Strategy

**Context**: `cascade_correlation.py` has 100+ methods in a single class (`CascadeCorrelationNetwork`) spanning ~4100 lines. This makes navigation, testing, and maintenance difficult.

**Option A: Mixin-based decomposition** (Recommended)

- Split into logical mixins: `TrainingMixin`, `SerializationMixin`, `MultiprocessingMixin`, `ValidationMixin`, `VisualizationMixin`
- `CascadeCorrelationNetwork` inherits from all mixins
- Pros: Preserves API compatibility, logical grouping, independent testing
- Cons: Python mixins can be confusing, method resolution order complexity

**Option B: Composition with delegate objects:**

- Extract `TrainingManager`, `SnapshotManager`, `CandidateManager` as separate classes
- `CascadeCorrelationNetwork` delegates to these via composition
- Pros: Cleaner OOP, no inheritance complexity
- Cons: Requires passing state/context to delegates, more refactoring

**Option C: Module-level functions:**

- Extract stateless operations as module-level functions
- Keep stateful operations in the class
- Pros: Simple, testable functions
- Cons: Many methods need `self` state, limited extraction possible

**Recommendation**: Option A for Phase 1 (achieves file size targets with minimal API changes), evolve toward Option B in Phase 2 as the architecture matures.

**Migration Impact (2026-02-24)**: Note that the CasCor Service API already implements a form of Option B at the service layer: `TrainingLifecycleManager` (training coordination), `TrainingMonitor` (metrics), `TrainingStateMachine` (state), and `WebSocketManager` (communication) are delegate objects. The core `CascadeCorrelationNetwork` class remains monolithic, but the service layer demonstrates the delegate pattern. Consider aligning the core class refactoring with the service layer's architectural patterns.

---

### Design Decision 7: Legacy Spiral Code Removal Strategy (CAS-REF-004)

**Context**: 16 deprecated methods in `spiral_problem.py` carry `DeprecationWarning`. JuniperData now provides the functionality via REST API.

**Option A: Hard removal** (Recommended, with gate)

- Delete all 16 deprecated methods
- Gate: Only proceed after JuniperData E2E integration tests pass (INT-P3-002)
- Pros: Clean codebase, no maintenance burden
- Cons: No fallback if JuniperData is unavailable

**Option B: Soft removal with feature flag:**

- Add `CASCOR_USE_LOCAL_SPIRAL_GEN=1` env var fallback
- Remove methods only after 2 release cycles
- Pros: Graceful degradation
- Cons: Maintains two code paths

**Option C: Move to separate `legacy/` module**

- Don't delete, just move to `src/spiral_problem/legacy.py`
- Import only if needed
- Pros: Code still available if needed
- Cons: Still exists in codebase, still maintained

**Recommendation**: Option A with the gate. The deprecation warnings have been in place since CAS-INT-002. Once E2E tests confirm JuniperData stability, remove cleanly.

**Migration Impact (2026-02-24)**: The gate condition is closer to being met. JuniperData is running as a standalone service (`pcalnon/juniper-data`, CI green), and `juniper-data-client` v0.3.0 is a published dependency. The remaining gate is INT-P3-002 (E2E live-service tests).

---

## Dependencies Matrix (Updated for Post-Migration)

```
INT-P0-001 (Walrus bug)
    └── No dependencies, fix immediately

INT-P0-002 (ActivationWithDerivative duplication)
    └── No dependencies, fix immediately
    └── [NOTE] If shared with juniper-cascor-worker, coordinate with CAS-005

INT-P0-003 (Invalid CandidateUnit params)
    └── No dependencies, fix immediately

INT-P0-004 (Hardcoded path in remote_client_0)
    └── DELETE FILE (superseded by juniper-cascor-worker)

INT-P1-001 (Duplicated JuniperDataClient)
    └── RESOLVED (juniper-data-client on PyPI)

INT-P1-004 (Full IPC)
    └── SUBSTANTIALLY RESOLVED (CasCor Service API + juniper-cascor-client; Canopy integration testing pending)

C.1 (Async wrapper)
    └── RESOLVED (TrainingLifecycleManager)

C.2 (RemoteWorkerClient in CascorIntegration)
    └── SUPERSEDED (juniper-cascor-worker + server-side management)

CAS-004 (JuniperBranch extraction)
    └── RESOLVED (juniper-cascor-worker on PyPI)

CAS-CANOPY-001 (Prediction Grid API)
    └── RESOLVED (GET /v1/decision-boundary)

CAS-CANOPY-002 (Serialization API)
    └── Requires: /v1/snapshots/* API endpoints (deferred)
    └── Requires: juniper-cascor-client snapshot methods
    └── Blocks: CAN-CRIT-002, CAN-014, CAN-015

CAS-REF-002 (CI coverage gates)
    └── PARTIALLY RESOLVED (80% global; per-module pending)
    └── CAS-REF-003 (Type errors) - gates should wait for type fixes

CAS-REF-004 (Legacy code removal)
    └── INT-P3-002 (E2E integration tests) - validate before removing

CAS-005 (Common dependencies)
    └── CAS-004 RESOLVED; evaluate if shared types package needed

INT-P3-003 (Docker Compose)
    └── Coordinate with migration Phase 6 (juniper-deploy repo)
```

---

## Post-Migration Risk Assessment (Updated)

| Risk                                                                  | Probability | Impact | Mitigation                                                      |
| --------------------------------------------------------------------- | ----------- | ------ | --------------------------------------------------------------- |
| Walrus operator bug (INT-P0-001) causes silent data corruption        | High        | High   | Fix immediately in Phase 0                                      |
| `ActivationWithDerivative` ACTIVATION_MAP divergence                  | Medium      | High   | Extract to shared module                                        |
| JuniperData service downtime crashes training                         | Medium      | High   | Retry logic implemented (CAS-INT-008); service now independently deployed |
| Coverage regression without enforced gates                            | Medium      | Medium | 80% global gate enforced; per-module thresholds needed          |
| Hardcoded paths break on other machines                               | ~~High~~ Low | ~~Medium~~ Low | Delete legacy files; polyrepo uses pip install |
| Slow test suite blocks CI pipeline                                    | Medium      | Medium | Scheduled tests separate from main CI                           |
| **NEW**: CasCor API missing endpoints block Canopy features           | Medium      | High   | Deferred endpoints (`/v1/snapshots/*`, `/v1/workers/*`) need prioritization |
| **NEW**: Version drift between 6 independent repos                    | Medium      | Medium | Phase 6 version compatibility matrix; consider dependabot/renovate |
| **NEW**: HTTP latency vs in-process call performance regression       | Medium      | Medium | WebSocket streaming for metrics; profile critical paths         |
| **NEW**: `CascorIntegration` removal breaks unidentified Canopy code  | Low         | High   | Three-mode activation provides fallback; thorough interface compatibility tests |
| **NEW**: WebSocket message format mismatch (service vs legacy format) | Medium      | Medium | Document format contract; adapter relay can transform if needed |
| ~~`sys.path` mutation causes import conflicts in production~~         | ~~Medium~~  | ~~Medium~~ | **RESOLVED** by polyrepo migration; `sys.path` injection being eliminated |

---

## Document History

| Date       | Author   | Changes                                                                                                                      |
| ---------- | -------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 2026-02-17 | AI Agent | Initial creation from JuniperData codebase audit                                                                             |
| 2026-02-17 | AI Agent | Added Section 5: Cross-references from JuniperCanopy comprehensive audit                                                     |
| 2026-02-18 | AI Agent | Complete rewrite: Exhaustive audit of all 25+ notes files, de-duplicated 89 unique items                                     |
| 2026-02-18 | AI Agent | Codebase validation pass: Validated 23 items against source code, confirmed 17 bugs, resolved 3 items, adjusted 3 severities |
| 2026-02-18 | AI Agent | Added development phases (0-5), high-level design analysis (7 architectural decisions with options/recommendations)          |
| 2026-02-24 | AI Agent | **Polyrepo migration reconciliation**: Analyzed impact of `POLYREPO_MIGRATION_PLAN.md` (v1.5.0) and `DECOUPLE_CANOPY_FROM_CASCOR_PLAN.md` against all 89 roadmap items. 6 items resolved/superseded by migration, 6 scope-changed, 2 new items added. Added Migration Impact annotations to 27 items across all sections. Updated Development Phases 0-5 with post-migration actions. Added 2 new verification items (CasCor Service API E2E, Three-Mode Activation). Updated Dependencies Matrix with resolved/superseded items. Revised Risk Assessment with 5 new migration-specific risks and 2 mitigated risks. Updated Design Decisions 4 and 5 as IMPLEMENTED. Pre-update version archived to `history/JUNIPER-CASCOR_POST-RELEASE_DEVELOPMENT-ROADMAP_2026-02-24.md`. |
| 2026-02-24 | AI Agent | **Post-reconciliation validation**: Fixed 10 errors introduced during migration reconciliation — corrected consolidated statistics (resolved: 10→6, superseded: 5→removed, scope-changed: 12→6, total: 72→83, High: 9→8, Low: 38→50), fixed INT-P1-001 factual error (`src/juniper_data_client/` directory entirely removed, not partially), corrected INT-P1-004 status in resolved table to SUBSTANTIALLY RESOLVED, fixed In-Code TODO priority for validate_training_results (P0→P2), fixed INT-P4-012–017 header to INT-P4-012–016, annotated INT-P3-008 as LIKELY RESOLVED in Phase 5 table, added INT-P1-002 and INT-P2-013 to Section 10 completed items, expanded INT-P3-009 version list, added traceable references to new Phase items. |
| 2026-02-25 | AI Agent | **Second validation pass**: Fixed 4 remaining moderate issues — corrected INT-P0-004/INT-P0-005 severity from "Medium" to "Low" (aligning section entries with P3-P4 statistical bucket), corrected INT-P1-004 status in Dependencies Matrix from "RESOLVED" to "SUBSTANTIALLY RESOLVED", corrected INT-P1-002 validation text (removed false claim that `requests` is in `pyproject.toml`; clarified resolution via vendored client removal), added C.3 status to Oracle analysis source table row. |
| 2026-02-25 | AI Agent | **Minor issue cleanup**: 7 fixes — clarified INT-P2-004 priority origin in statistics footnote, added overhead notes to Phase 2/3/4 effort estimates, fixed Codebase Validation CONFIRMED count (17→18 with overlap note for INT-P1-005), added Phase 0 note explaining retained Low-severity items, standardized INT-P1-005 severity notation, added P4-NEW-003/004 gap explanation, added design decision clarification to resolved table. |
| 2026-02-25 | AI Agent | **Third validation pass**: Fixed 3 moderate issues — updated INT-P0-004/INT-P0-005 migration impact prose from "Medium" to "Low" (labels were corrected earlier but prose paragraphs were missed), corrected INT-P2-014 `import traceback` count from 22 to 21 (verified against codebase: 21 local imports + 1 commented-out top-level). |
| 2026-02-25 | AI Agent | **Final minor cleanup**: 5 fixes — standardized severity notation to `~~Old~~ **New**` pattern across 6 items (INT-P0-004, INT-P0-005, INT-P1-003, INT-P1-006, INT-P2-004, INT-P2-009) plus SEVERITY ADJUSTED summary row, clarified INT-P1-004 attribution in Migration Summary to span Phases 2-3, added C.1/C.2 resolution note to NOT YET VALIDATED row, clarified CAS-CANOPY-002 Phase 3/5 scope overlap, updated INT-P2-010 stale line references from 142/145 to 174/177 (verified against codebase). |
| 2026-02-25 | AI Agent | **Fifth validation pass**: Fixed 4 moderate issues — corrected INT-P1-004 Migration Impact paragraph from "RESOLVED" to "SUBSTANTIALLY RESOLVED" (aligning with status field, Dependencies Matrix, and Section 10), added Phase 3 dependency gate note to CAS-REF-004 in Phase 2 table, corrected Codebase Validation CONFIRMED count from 18 to 19 (INT-P2-004 now included with overlap note, matching INT-P1-005 treatment), added P2 bucket change explanation to consolidated statistics footnote. |
| 2026-02-25 | AI Agent | **Fifth validation minor cleanup**: 5 fixes — named the 2 new P3-P4 items in statistics footnote (Phase 1 #10, Phase 5 #10), added Phase 5 aggregate effort note explaining why total is not aggregated, clarified INT-P1-002 attribution in Migration Summary as pre-migration resolved + further addressed by Phase 1, added Section 1 cross-reference to Phase 0 execution plan, clarified Section 10 resolved table header to distinguish 5 resolved + 1 substantially resolved work items. |
