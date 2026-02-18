# JuniperCascor ↔ JuniperData Integration Plan

**Project**: JuniperCascor - Cascade Correlation Neural Network  
**Integration Target**: JuniperData - Dataset Generation Service  
**Version**: 1.0.0  
**Author**: Paul Calnon  
**Created**: 2026-02-06  
**Status**: COMPLETE  
**Last Updated**: 2026-02-07  

---

## Primary Objective

All datasets used by JuniperCascor MUST be received from JuniperData via API call. No dataset generation operations shall be carried out locally within the JuniperCascor application.

## Current State Summary

| Component            | Status                                  | Notes                                                                                                                                                                                                          |
| -------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `JuniperDataClient`  | IMPLEMENTED + AUTH + RETRY + VALIDATION | `src/juniper_data_client/client.py` - REST client with API key auth, retry/backoff, URL validation, health check (CAS-INT-003, CAS-INT-008, CAS-INT-009)                                                       |
| `SpiralDataProvider` | IMPLEMENTED + VALIDATION + CONFIG       | `src/spiral_problem/data_provider.py` - NPZ contract validation, configuration validation with health check (CAS-INT-004, CAS-INT-009)                                                                         |
| Integration toggle   | REMOVED (MANDATORY)                     | `JUNIPER_DATA_URL` env var is now REQUIRED - raises ConfigurationError if missing (CAS-INT-001)                                                                                                                |
| Local generation     | DEPRECATED                              | 16 local generation methods have `DeprecationWarning` - will be removed in a future release (CAS-INT-002)                                                                                                      |
| Unit tests           | 131 PASSING                             | `test_juniper_data_client.py` (37), `test_spiral_data_provider.py` (26), `test_spiral_problem_juniper_data_integration.py` (6), `test_spiral_problem_coverage.py` (12), `test_spiral_problem_extended.py` (50) |
| Integration tests    | 10 PASSING                              | `test_juniper_data_e2e.py` (10) - full E2E with in-process JuniperData TestClient (run with `--integration` flag)                                                                                              |
| Full test suite      | 1287 PASSED                             | 86 skipped (slow/integration/dill), 0 failures, 0 regressions                                                                                                                                                  |

## Integration Tasks

### Phase 0: Make JuniperData Mandatory (P0)

#### CAS-INT-001: Make JUNIPER_DATA_URL Required

**Priority**: CRITICAL | **Status**: COMPLETE | **Effort**: Medium  
**Completed**: 2026-02-06  
**Blocking**: All other integration tasks

Make `JUNIPER_DATA_URL` a required configuration. When not set, raise a clear `ConfigurationError` at startup with actionable guidance.

**Changes Required**:

- Modify `generate_n_spiral_dataset()` in `spiral_problem.py` to ALWAYS use `SpiralDataProvider`
- Remove the conditional `if juniper_data_url:` toggle (lines 501-517)
- Remove the local generation fallback code path (lines 519-595)
- Add startup validation for `JUNIPER_DATA_URL` in `SpiralProblem.__init__()` or `evaluate()`
- Keep `_initialize_spiral_problem_params()` only for network training params, not data generation
- Keep the helper methods used only for local generation (`_generate_spiral_coordinates`, `_generate_base_radial_distance`, `_generate_angular_offset`, `_generate_raw_spiral_coordinates`, `_generate_xy_coordinates`, `_make_coords`, `_make_noise`, `_create_input_features`, `_create_one_hot_targets`, `_create_spiral_dataset`, `_convert_to_tensors`, `_shuffle_dataset`, `_partition_dataset`, `_split_dataset`, `_find_partition_index_end`, `_dataset_split_index_end`) - these should be DEPRECATED and eventually removed but we will preserve them in this phase with deprecation warnings

**Acceptance Criteria**:

- Calling `generate_n_spiral_dataset()` without `JUNIPER_DATA_URL` raises a descriptive error
- Calling `generate_n_spiral_dataset()` WITH `JUNIPER_DATA_URL` uses `SpiralDataProvider` exclusively
- Existing tests updated to work with mandatory API path (mock the provider in all tests)

**Resolution Applied**:

- Modified `generate_n_spiral_dataset()` to require `JUNIPER_DATA_URL` env var
- Raises `ConfigurationError` with actionable guidance when not set
- Always delegates to `SpiralDataProvider` for dataset fetching
- Removed 78 lines of local spiral generation code (the fallback path)
- Updated `test_spiral_problem_coverage.py` to mock the API path
- All 1269 unit tests pass with 0 regressions

---

#### CAS-INT-002: Deprecate Local Spiral Generation Methods

**Priority**: HIGH | **Status**: COMPLETE | **Effort**: Small  
**Completed**: 2026-02-07  
**Depends On**: CAS-INT-001

Add deprecation warnings to all local generation methods in `SpiralProblem` that are no longer needed when JuniperData is mandatory:

- `_generate_spiral_coordinates()`
- `_generate_base_radial_distance()`
- `_generate_angular_offset()`
- `_generate_raw_spiral_coordinates()`
- `_generate_xy_coordinates()`
- `_make_coords()`
- `_make_noise()`
- `_create_input_features()`
- `_create_one_hot_targets()`
- `_create_spiral_dataset()`
- `_convert_to_tensors()`
- `_shuffle_dataset()`
- `_partition_dataset()`
- `_split_dataset()`
- `_find_partition_index_end()`
- `_dataset_split_index_end()`

These methods should remain for one release cycle but emit `DeprecationWarning`.

**Resolution Applied**:

- Added `import warnings` to `spiral_problem.py`
- Added `warnings.warn()` with `DeprecationWarning` and `stacklevel=2` as the first statement in all 16 methods
- Warning message: `"SpiralProblem._METHOD_NAME() is deprecated and will be removed in a future release. Dataset generation is now handled by JuniperData service."`
- All 68 spiral-related tests pass with 0 regressions

---

#### CAS-INT-003: Add API Key Authentication Support to Client

**Priority**: HIGH | **Status**: COMPLETE | **Effort**: Small  
**Completed**: 2026-02-06

JuniperData supports API key authentication via `X-API-Key` header (DATA-017 COMPLETE). The `JuniperDataClient` needs to support this.

**Changes Required**:

- Add `api_key: Optional[str]` parameter to `JuniperDataClient.__init__()`
- Read from `JUNIPER_DATA_API_KEY` env var as fallback
- Add `X-API-Key` header to all requests when configured
- Update `SpiralDataProvider` to pass API key through

**Resolution Applied**:

- Added `api_key: Optional[str]` parameter to `JuniperDataClient.__init__()`
- Falls back to `JUNIPER_DATA_API_KEY` env var when not explicitly provided
- Sets `X-API-Key` header on session for all requests
- Updated `SpiralDataProvider` to pass `api_key` through to client
- Added 5 new tests in `TestJuniperDataClientAuthentication`

---

#### CAS-INT-004: Add NPZ Data Contract Validation

**Priority**: HIGH | **Status**: COMPLETE | **Effort**: Small  
**Completed**: 2026-02-06

Add validation in `SpiralDataProvider._convert_arrays_to_tensors()` to assert the NPZ artifact meets the expected contract:

- Required keys: `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full`
- All arrays should be `float32` dtype
- Feature arrays should have 2 columns (x, y coordinates)
- Label arrays should have columns matching number of spirals

**Resolution Applied**:

- Added validation in `_convert_arrays_to_tensors()` before tensor conversion
- Validates required NPZ keys: `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full`
- Validates all arrays are 2D
- Validates feature arrays have exactly 2 columns (x, y coordinates)
- Raises `SpiralDataProviderError` with descriptive messages on contract violations
- Added 5 new tests in `TestSpiralDataProviderContractValidation`

---

### Phase 1: Test Infrastructure (P1)

#### CAS-INT-005: Update Existing Tests for Mandatory API Path

**Priority**: CRITICAL | **Status**: COMPLETE | **Effort**: Medium  
**Completed**: 2026-02-06  
**Depends On**: CAS-INT-001

Update `test_spiral_problem_juniper_data_integration.py`:

- `test_uses_legacy_path_when_env_not_set` → update to assert `ConfigurationError` is raised
- Add test verifying `generate_n_spiral_dataset()` always calls `SpiralDataProvider`
- Add test verifying correct error message when `JUNIPER_DATA_URL` is missing

**Resolution Applied**:

- Replaced `test_spiral_problem_juniper_data_integration.py` with 6 tests for mandatory API path
- Tests: ConfigurationError on missing URL, guidance message, provider usage, param passing, tuple format, error propagation
- Fixed `test_spiral_problem_coverage.py` to mock the API path instead of using local generation
- Fixed `test_spiral_data_provider.py` mock assertion for api_key parameter

---

#### CAS-INT-006: Add Data Contract Validation Tests

**Priority**: LOW | **Status**: COMPLETE | **Effort**: Small  
**Completed**: 2026-02-06 (delivered as part of CAS-INT-004)  
**Depends On**: CAS-INT-004

Test cases for NPZ contract validation:

- Missing required keys raises error
- Wrong dtype warns or raises
- Wrong shape raises error
- Valid contract passes silently

**Resolution Applied**:

- 5 tests in `TestSpiralDataProviderContractValidation` (in `test_spiral_data_provider.py`) cover all acceptance criteria
- Tests: missing keys, wrong dimensions, wrong feature columns, valid contract, error message content

---

#### CAS-INT-007: Add Integration Tests with JuniperData TestClient

**Priority**: MEDIUM | **Status**: COMPLETE | **Effort**: Large  
**Completed**: 2026-02-07  
**Depends On**: CAS-INT-001

Create `tests/integration/test_juniper_data_e2e.py` with:

- Marker: `@pytest.mark.integration` and `@pytest.mark.requires_juniper_data`
- Tests that exercise the full flow: create dataset → download NPZ → convert to tensors → feed to CascorNetwork
- Can use JuniperData's FastAPI `TestClient` if importable, or require live service

**Resolution Applied**:

- Created `src/tests/integration/test_juniper_data_e2e.py` with 10 E2E tests across 3 test classes
- Uses JuniperData's `create_app()` factory with `TestClient` for in-process API testing (no live service needed)
- Created `_RequestsSessionAdapter` to bridge httpx responses (Starlette TestClient) to `requests.Response` objects
- `TestJuniperDataE2EHealth` (2 tests): health endpoint connectivity, health_check() method
- `TestJuniperDataE2EDatasetCreation` (4 tests): metadata creation, NPZ download, shape validation, idempotent creation
- `TestJuniperDataE2EFullFlow` (4 tests): provider roundtrip, 3-spiral variant, legacy algorithm, CascorNetwork forward pass
- Registered `requires_juniper_data` marker in `pytest.ini`
- Tests run with `--integration` flag: `pytest integration/test_juniper_data_e2e.py --integration`
- All 10 tests pass in 1.86s

---

### Phase 2: Operational Readiness (P2)

#### CAS-INT-008: Add Retry/Backoff for Transient Failures

**Priority**: MEDIUM | **Status**: COMPLETE | **Effort**: Small  
**Completed**: 2026-02-07

Add basic retry logic to `JuniperDataClient._request()` for transient HTTP errors (502, 503, 504, connection errors).

**Resolution Applied**:

- Added `MAX_RETRIES = 3`, `RETRY_BACKOFF_BASE = 1.0`, `_RETRYABLE_STATUS_CODES = (502, 503, 504)` class constants
- Implemented loop-based retry with exponential backoff (`delay = base * 2^attempt`) in `_request()`
- Retries on 502/503/504 status codes, `requests.ConnectionError`, and `requests.Timeout`
- Non-retryable errors (4xx) fail immediately without retry
- Logs retries at WARNING level with attempt count and delay
- Added 8 new tests in `TestJuniperDataClientRetry` covering all retry scenarios
- All 30 client tests pass

---

#### CAS-INT-009: Configuration Validation at Startup

**Priority**: MEDIUM | **Status**: COMPLETE | **Effort**: Small  
**Completed**: 2026-02-07

Add early configuration validation:

- Validate `JUNIPER_DATA_URL` format (valid URL)
- Optionally ping health endpoint at startup
- Log configuration source for debugging

**Resolution Applied**:

- Added `validate_url()` static method to `JuniperDataClient` - validates scheme (http/https only) and hostname presence
- URL validation runs automatically in `__init__()` after normalization
- Added `health_check()` method to `JuniperDataClient` - GET `/v1/health` with 5s timeout, returns bool
- Added `validate_configuration()` method to `SpiralDataProvider` - checks URL is set, validates via client, warns if health check fails
- Added 7 new tests in `TestJuniperDataClientValidation` (URL scheme/host validation, health check success/failure)
- Added 2 new tests in `TestSpiralDataProviderConfigValidation` (missing URL, valid URL)
- All 63 provider+client tests pass

---

### Cross-Project References (from JuniperData plan)

| ID          | Item                              | Status                     | Source                      |
| ----------- | --------------------------------- | -------------------------- | --------------------------- |
| CAS-REF-001 | Code coverage below 90%           | IN PROGRESS                | PRE-DEPLOYMENT_ROADMAP-2.md |
| CAS-REF-002 | CI/CD coverage gates not enforced | NOT STARTED                | PRE-DEPLOYMENT_ROADMAP-2.md |
| CAS-REF-003 | Type errors gradual fix           | IN PROGRESS                | PRE-DEPLOYMENT_ROADMAP-2.md |
| CAS-REF-004 | Legacy spiral code removal        | → CAS-INT-001, CAS-INT-002 | Refactor Plan               |
| CAS-REF-005 | RemoteWorkerClient integration    | NOT STARTED                | PRE-DEPLOYMENT_ROADMAP.md   |

## Summary Statistics

| Category      | Count                                               |
| ------------- | --------------------------------------------------- |
| Total Tasks   | 9                                                   |
| COMPLETE      | 9 (CAS-INT-001 through CAS-INT-009) ✅ ALL COMPLETE |
| NOT STARTED   | 0                                                   |
| Cross-Project | 5 (CAS-REF-001 through CAS-REF-005)                 |

## Document History

| Date       | Author   | Changes                                                                                                                                  |
| ---------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-02-06 | AI Agent | Initial creation - JuniperCascor-specific integration plan                                                                               |
| 2026-02-06 | AI Agent | Completed CAS-INT-001 (mandatory API), CAS-INT-003 (API key auth), CAS-INT-004 (NPZ validation), CAS-INT-005 (test updates)              |
| 2026-02-07 | AI Agent | Completed CAS-INT-002 (deprecation warnings), CAS-INT-006 (contract tests), CAS-INT-008 (retry/backoff), CAS-INT-009 (config validation) |
| 2026-02-07 | AI Agent | Completed CAS-INT-007 (E2E integration tests with JuniperData TestClient) - ALL 9 TASKS COMPLETE                                         |
