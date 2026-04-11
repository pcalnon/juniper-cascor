# Hardcoded Values Analysis — juniper-cascor

**Version**: 0.4.0
**Analysis Date**: 2026-04-08
**Analyst**: Claude Code (Automated Code Review)
**Status**: PLANNING ONLY — No source code modifications

---

## Executive Summary

The juniper-cascor codebase contains **76 identified hardcoded values** across its source files. Of these, **20 are already covered** by existing constants in the `cascor_constants/` module hierarchy, while **56 require extraction** into constants. The core neural network constants (`cascor_constants/`) are well-organized with 6 dedicated modules. The primary gaps are in the API layer — models, routes, middleware, observability, and service launcher components.

---

## 1. Existing Constants Infrastructure

### Constants Module Hierarchy

| Module | Path | Purpose | Constants Count |
|--------|------|---------|-----------------|
| `constants.py` | `cascor_constants/constants.py` | Master re-exports | N/A (re-exports) |
| `constants_model` | `cascor_constants/constants_model/constants_model.py` | Network hyperparameters, training config | 24+ |
| `constants_activation` | `cascor_constants/constants_activation/constants_activation.py` | Activation function definitions | 10+ |
| `constants_candidates` | `cascor_constants/constants_candidates/constants_candidates.py` | Candidate unit training params | 5+ |
| `constants_logging` | `cascor_constants/constants_logging/constants_logging.py` | Logger format strings, field names | 10+ |
| `constants_hdf5` | `cascor_constants/constants_hdf5/constants_hdf5.py` | File paths for snapshots | 5+ |
| `constants_problem` | `cascor_constants/constants_problem/constants_problem.py` | Spiral dataset configuration | 15+ |

### API Settings (`api/settings.py`)

All 16 settings values are properly defined as Pydantic BaseSettings fields with environment variable overrides using the `JUNIPER_CASCOR_` prefix. These are **fully covered**.

---

## 2. Hardcoded Values Inventory

### 2.1 API Network Model Defaults (`api/models/network.py`) — NOT COVERED

| Line | Value | Type | Context | Proposed Constant Name |
|------|-------|------|---------|----------------------|
| 11 | `2` | int | Input size Field default | `API_NETWORK_INPUT_SIZE_DEFAULT` |
| 12 | `2` | int | Output size Field default | `API_NETWORK_OUTPUT_SIZE_DEFAULT` |
| 13 | `0.01` | float | Learning rate Field default | `API_NETWORK_LEARNING_RATE_DEFAULT` |
| 14 | `0.005` | float | Candidate learning rate Field default | `API_NETWORK_CANDIDATE_LEARNING_RATE_DEFAULT` |
| 15 | `10` | int | Max hidden units Field default | `API_NETWORK_MAX_HIDDEN_UNITS_DEFAULT` |
| 16 | `8` | int | Candidate pool size Field default | `API_NETWORK_CANDIDATE_POOL_SIZE_DEFAULT` |
| 17 | `0.1` | float | Correlation threshold Field default | `API_NETWORK_CORRELATION_THRESHOLD_DEFAULT` |
| 18 | `5` | int | Patience Field default | `API_NETWORK_PATIENCE_DEFAULT` |
| 19 | `50` | int | Candidate epochs Field default | `API_NETWORK_CANDIDATE_EPOCHS_DEFAULT` |
| 20 | `25` | int | Output epochs Field default | `API_NETWORK_OUTPUT_EPOCHS_DEFAULT` |
| 21 | `200` | int | Epochs max Field default | `API_NETWORK_EPOCHS_MAX_DEFAULT` |
| 22 | `1000` | int | Max iterations Field default | `API_NETWORK_MAX_ITERATIONS_DEFAULT` |
| 23 | `"zero"` | str | Init output weights mode | `API_NETWORK_INIT_OUTPUT_WEIGHTS_DEFAULT` |

**Files requiring import**: `api/models/network.py`
**Target location**: New module `cascor_constants/constants_api/constants_api_defaults.py` or extend `api/settings.py`

### 2.2 API Training Model Defaults (`api/models/training.py`) — NOT COVERED

| Line | Value | Type | Context | Proposed Constant Name |
|------|-------|------|---------|----------------------|
| 11 | `"inline"` | str | Dataset source default | `API_DATASET_SOURCE_DEFAULT` |
| 24 | `100000` | int | Max training samples | `API_MAX_DATASET_SAMPLES` |
| 25 | `100000` | int | Max training targets | `API_MAX_DATASET_TARGETS` |

**Files requiring import**: `api/models/training.py`
**Target location**: `cascor_constants/constants_api/constants_api_defaults.py`

### 2.3 Lifecycle Manager (`api/lifecycle/manager.py`) — NOT COVERED

| Line | Value | Type | Context | Proposed Constant Name |
|------|-------|------|---------|----------------------|
| 175 | `10` | int | Default max hidden units | `LIFECYCLE_DEFAULT_MAX_HIDDEN_UNITS` |
| 176 | `200` | int | Default epochs max | `LIFECYCLE_DEFAULT_EPOCHS_MAX` |
| 177 | `1000` | int | Default max iterations | `LIFECYCLE_DEFAULT_MAX_ITERATIONS` |
| 243 | `100` | int | Recent metrics buffer size | `METRICS_BUFFER_SIZE` |
| 327 | `0.1` | float | Progress queue wait timeout | `PROGRESS_QUEUE_WAIT_TIMEOUT` |
| 332 | `0.25` | float | Progress queue get timeout | `PROGRESS_QUEUE_GET_TIMEOUT` |
| 402 | `2.0` | float | Drain thread join timeout | `DRAIN_THREAD_JOIN_TIMEOUT` |
| 680 | `30` | int | Default candidate patience | `LIFECYCLE_DEFAULT_CANDIDATE_PATIENCE` |

**Files requiring import**: `api/lifecycle/manager.py`
**Target location**: `cascor_constants/constants_api/constants_api_defaults.py`

### 2.4 Observability (`api/observability.py`) — NOT COVERED

| Line | Value | Type | Context | Proposed Constant Name |
|------|-------|------|---------|----------------------|
| 150 | `10 * 1024 * 1024` | int | Log file max bytes (10 MB) | `LOG_FILE_MAX_BYTES` |
| 151 | `5` | int | Log file backup count | `LOG_FILE_BACKUP_COUNT` |
| 176 | `1.0` | float | Sentry traces sample rate | `SENTRY_TRACES_SAMPLE_RATE` |
| 255 | `(0.001, 0.005, ...)` | tuple | Prometheus latency buckets | `PROMETHEUS_LATENCY_BUCKETS` |

**Files requiring import**: `api/observability.py`
**Target location**: `cascor_constants/constants_logging/constants_logging.py` (extend existing)

### 2.5 Service Launcher (`api/service_launcher.py`) — NOT COVERED

| Line | Value | Type | Context | Proposed Constant Name |
|------|-------|------|---------|----------------------|
| 55 | `5` | int | Process termination timeout (sec) | `PROCESS_TERMINATION_TIMEOUT` |
| 72 | `5` | int | Service termination timeout (sec) | `SERVICE_TERMINATION_TIMEOUT` |
| 102 | `5` | int | Health check HTTP timeout (sec) | `HEALTH_CHECK_HTTP_TIMEOUT` |

**Files requiring import**: `api/service_launcher.py`
**Target location**: `cascor_constants/constants_api/constants_api_defaults.py`

### 2.6 Middleware & Routes — NOT COVERED

| File | Line | Value | Type | Context | Proposed Constant Name |
|------|------|-------|------|---------|----------------------|
| `api/middleware.py` | 67 | `413` | int | HTTP 413 status code | `HTTP_413_PAYLOAD_TOO_LARGE` |
| `api/middleware.py` | 71 | `413` | int | HTTP 413 status code | (same) |
| `api/routes/decision_boundary.py` | 22 | `50` | int | Resolution default | `DECISION_BOUNDARY_RESOLUTION_DEFAULT` |
| `api/routes/decision_boundary.py` | 22 | `5` | int | Min resolution | `DECISION_BOUNDARY_RESOLUTION_MIN` |
| `api/routes/decision_boundary.py` | 22 | `200` | int | Max resolution | `DECISION_BOUNDARY_RESOLUTION_MAX` |
| `api/routes/decision_boundary.py` | 15,32,34,37 | `503,404,404,500` | int | HTTP status codes | `HTTP_503_*`, `HTTP_404_*`, `HTTP_500_*` |

**Files requiring import**: `api/middleware.py`, `api/routes/decision_boundary.py`
**Target location**: `cascor_constants/constants_api/constants_api_defaults.py`

### 2.7 Application URLs & Endpoints (`api/app.py`) — NOT COVERED

| Line | Value | Type | Context | Proposed Constant Name |
|------|-------|------|---------|----------------------|
| 112 | `"http://localhost:8100"` | str | JuniperData default URL | `JUNIPER_DATA_URL_DEFAULT` |
| 176 | `"http://localhost:8100"` | str | JuniperData env default | (same constant) |
| 183 | `60` | int | JuniperData ready timeout (sec) | `JUNIPER_DATA_READY_TIMEOUT` |
| 237 | `f"http://localhost:{port}/v1/health"` | str | Self health check URL template | `SELF_HEALTH_CHECK_URL_TEMPLATE` |
| 239 | `30.0` | float | Canopy startup wait timeout | `CANOPY_STARTUP_WAIT_TIMEOUT` |
| 239 | `1.0` | float | Canopy startup check interval | `CANOPY_STARTUP_CHECK_INTERVAL` |
| 247 | `"false"` | str | Canopy demo mode disabled | `CANOPY_DEMO_MODE_DISABLED` |
| 254 | `"http://localhost:8050/v1/health"` | str | Canopy health check URL | `CANOPY_HEALTH_CHECK_URL` |

**Files requiring import**: `api/app.py`
**Target location**: `cascor_constants/constants_api/constants_api_defaults.py`

### 2.8 Worker Security (`api/workers/security.py`) — PARTIALLY COVERED

| Line | Value | Type | Context | Proposed Constant Name | Status |
|------|-------|------|---------|----------------------|--------|
| 36 | `"TLSv1.3"` | str | Min TLS version | `TLS_MIN_VERSION_DEFAULT` | NOT_COVERED |
| 101 | `300.0` | float | Cleanup interval (5 min) | `RATE_LIMITER_CLEANUP_INTERVAL` | NOT_COVERED |
| 200 | `0.001` | float | Stale correlation threshold | `ANOMALY_STALE_CORR_THRESHOLD` | NOT_COVERED |
| 200 | `10` | int | Duplicate correlation window | `ANOMALY_DUPLICATE_CORR_WINDOW` | NOT_COVERED |

**Files requiring import**: `api/workers/security.py`
**Target location**: `cascor_constants/constants_api/constants_api_defaults.py`

### 2.9 Core Model & Candidate Unit — PARTIALLY COVERED

| File | Line | Value | Type | Context | Proposed Constant Name | Status |
|------|------|-------|------|---------|----------------------|--------|
| `candidate_unit/candidate_unit.py` | 296 | `10` | int | Random max value for seed | `RANDOM_MAX_VALUE_FOR_SEED` | NOT_COVERED |
| `candidate_unit/candidate_unit.py` | 370 | `10000` | int | Max roll count for randomness | `MAX_ROLL_COUNT` | NOT_COVERED |

**Files requiring import**: `candidate_unit/candidate_unit.py`
**Target location**: `cascor_constants/constants_candidates/constants_candidates.py` (extend existing)

### 2.10 Spiral Data Provider — NOT COVERED

| File | Line | Value | Type | Context | Proposed Constant Name |
|------|------|-------|------|---------|----------------------|
| `spiral_problem/data_provider.py` | 77-99 | (multiple) | str/URL | JuniperData URLs & messages | Various |
| `spiral_problem/spiral_problem.py` | 512 | `"http://localhost:8100"` | str | JuniperData default URL | `JUNIPER_DATA_URL_DEFAULT` |

**Files requiring import**: `spiral_problem/data_provider.py`, `spiral_problem/spiral_problem.py`
**Target location**: `cascor_constants/constants_api/constants_api_defaults.py`

### 2.11 Snapshots & Profiling — MINOR

| File | Line | Value | Type | Context | Proposed Constant Name | Status |
|------|------|-------|------|---------|----------------------|--------|
| `snapshots/snapshot_serializer.py` | 1077 | `"cascor_hdf5_v1"` | str | Legacy format name | `HDF5_FORMAT_NAME_LEGACY` | NOT_COVERED |
| `profiling/logging_utils.py` | 115 | `1000` | int | Max buffer size | `BATCH_LOGGER_MAX_BUFFER_DEFAULT` | NOT_COVERED |
| `profiling/memory.py` | 95 | `60` | int | Separator width | `SEPARATOR_WIDTH` | NOT_COVERED |

---

## 3. Coverage Summary

| Category | Total | Covered | Not Covered | Priority |
|----------|-------|---------|-------------|----------|
| API Network Defaults | 13 | 0 | 13 | **HIGH** |
| Lifecycle/Manager | 8 | 0 | 8 | **HIGH** |
| URLs/Endpoints | 6 | 0 | 6 | **HIGH** |
| Timeouts/Delays | 12 | 2 | 10 | **MEDIUM** |
| Observability | 4 | 0 | 4 | **MEDIUM** |
| Security/TLS | 4 | 2 | 2 | **MEDIUM** |
| HTTP Status Codes | 8 | 0 | 8 | **LOW** |
| Core Model | 2 | 0 | 2 | **LOW** |
| Snapshots/Profiling | 3 | 0 | 3 | **LOW** |
| API Settings | 16 | 16 | 0 | — |
| **TOTAL** | **76** | **20** | **56** | — |

---

## 4. Remediation Approaches

### Approach A: Extend Existing `cascor_constants/` Hierarchy (RECOMMENDED)

**Description**: Create a new `cascor_constants/constants_api/` submodule for all API-layer constants and extend existing modules for core constants.

**New files**:
- `cascor_constants/constants_api/__init__.py`
- `cascor_constants/constants_api/constants_api_defaults.py` — API model defaults, lifecycle defaults, URLs, timeouts, observability

**Extended files**:
- `cascor_constants/constants_candidates/constants_candidates.py` — Add `RANDOM_MAX_VALUE_FOR_SEED`, `MAX_ROLL_COUNT`
- `cascor_constants/constants_logging/constants_logging.py` — Add log rotation config
- `cascor_constants/constants.py` — Re-export new API constants

**Strengths**:
- Follows existing project patterns
- Clean separation of concerns (API vs. core)
- Minimal import path changes
- Consistent with established naming conventions

**Weaknesses**:
- Adds another submodule to the hierarchy
- May need to coordinate defaults between `settings.py` and `constants_api`

**Risks**:
- Potential for settings/constants value drift — mitigate by referencing constants from settings defaults
- Import cycle risk — mitigate by keeping constants leaf-level (no imports from application modules)

**Guardrails**:
- All new constants must use the `_PROJECT_*` naming prefix consistent with `constants_model`
- Add validation tests ensuring constants match Field defaults in Pydantic models
- Document the constants/settings hierarchy in AGENTS.md

### Approach B: Centralize All API Defaults in `api/settings.py`

**Description**: Add all API-layer defaults as module-level constants in `api/settings.py`, reference them from Pydantic models and lifecycle manager.

**Strengths**:
- Single source of truth for API configuration
- Environment variable overrides available for all values
- Simpler import structure

**Weaknesses**:
- `settings.py` grows significantly (would add ~50 constants)
- Mixes infrastructure config with domain defaults
- Not aligned with existing `cascor_constants/` pattern

**Risks**:
- Settings file becomes unwieldy
- Harder to distinguish environment-configurable vs. fixed constants

### Approach C: Hybrid — Settings for Configurable, Constants for Fixed

**Description**: Environment-configurable values (URLs, timeouts, ports) go in `api/settings.py`. Fixed domain constants (model defaults, thresholds, buffer sizes) go in `cascor_constants/constants_api/`.

**Strengths**:
- Clear semantic distinction
- Right tool for the right job
- Prevents unnecessary environment variable proliferation

**Weaknesses**:
- Requires judgment calls on each value
- Two places to look for defaults

### Recommended Approach: **A** (Extend `cascor_constants/` Hierarchy)

Rationale: The existing `cascor_constants/` infrastructure is mature and well-organized. The API layer should follow the same pattern. Values that need environment overrides can reference these constants as defaults in `settings.py`.

---

## 5. Files Requiring Modification (Implementation Phase)

| File | Changes | Imports Added |
|------|---------|---------------|
| `cascor_constants/constants_api/constants_api_defaults.py` | **NEW FILE** — 43 constants | N/A |
| `cascor_constants/constants_api/__init__.py` | **NEW FILE** — re-exports | N/A |
| `cascor_constants/constants.py` | Add re-export of API constants | `constants_api` |
| `api/models/network.py` | Replace 13 Field defaults with constants | `constants_api_defaults` |
| `api/models/training.py` | Replace 3 Field defaults with constants | `constants_api_defaults` |
| `api/lifecycle/manager.py` | Replace 8 hardcoded values | `constants_api_defaults` |
| `api/observability.py` | Replace 4 values | `constants_api_defaults` or `constants_logging` |
| `api/service_launcher.py` | Replace 3 timeout values | `constants_api_defaults` |
| `api/middleware.py` | Replace HTTP status codes | `constants_api_defaults` |
| `api/routes/decision_boundary.py` | Replace 6 values | `constants_api_defaults` |
| `api/app.py` | Replace 8 URL/timeout values | `constants_api_defaults` |
| `api/workers/security.py` | Replace 4 values | `constants_api_defaults` |
| `candidate_unit/candidate_unit.py` | Replace 2 values | `constants_candidates` |
| `spiral_problem/data_provider.py` | Replace URL values | `constants_api_defaults` |
| `spiral_problem/spiral_problem.py` | Replace URL value | `constants_api_defaults` |
| `snapshots/snapshot_serializer.py` | Replace 1 format string | `constants_hdf5` |
| `profiling/logging_utils.py` | Replace 1 buffer size | `constants_logging` |

---

## 6. Test Impact Assessment

All existing tests should continue to pass since constants will hold the same values. However:

- Unit tests that assert specific default values should be updated to reference constants
- Integration tests that depend on hardcoded URLs may need fixture updates
- New tests should be added to validate:
  - All constants are importable
  - Constants match Pydantic Field defaults
  - No orphaned constants (unused after refactor)

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Import cycle between constants and application modules | Low | High | Keep constants as leaf modules with no app-layer imports |
| Value drift between constants and settings defaults | Medium | Medium | Reference constants from settings; add validation test |
| Test failures from incorrect constant values | Low | Low | Run full test suite after each file change |
| Missing constant in re-export | Low | Medium | Add import validation test |
| Breaking change in API behavior | Very Low | High | Constants preserve existing literal values exactly |
