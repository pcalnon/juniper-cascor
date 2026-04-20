# Hardcoded Values Refactor Plan — juniper-cascor

**Version**: 0.4.0
**Created**: 2026-04-08
**Status**: PLANNING — No source code modifications
**Companion Document**: `HARDCODED_VALUES_ANALYSIS.md`

---

## Phase 1: Constants Infrastructure (Priority: HIGH)

### Step 1.1: Create API Constants Module

**Task**: Create `cascor_constants/constants_api/__init__.py` and `cascor_constants/constants_api/constants_api_defaults.py`

**Details**:
- Define 43 constants organized into logical groups:
  - Network model defaults (13 constants)
  - Training model defaults (3 constants)
  - Lifecycle manager defaults (8 constants)
  - Service URLs and endpoints (8 constants)
  - Timeout values (6 constants)
  - Security defaults (4 constants)
  - Observability config (4 constants)
- Use `_PROJECT_API_*` naming prefix consistent with existing `_PROJECT_MODEL_*` pattern
- All constants as module-level variables with type annotations

### Step 1.2: Update Master Re-exports

**Task**: Update `cascor_constants/constants.py` to re-export the new API constants module

### Step 1.3: Extend Existing Constants Modules

**Task**: Add missing constants to existing modules:
- `constants_candidates.py`: Add `RANDOM_MAX_VALUE_FOR_SEED`, `MAX_ROLL_COUNT`
- `constants_logging.py`: Add `LOG_FILE_MAX_BYTES`, `LOG_FILE_BACKUP_COUNT`
- `constants_hdf5.py`: Add `HDF5_FORMAT_NAME_LEGACY`

---

## Phase 2: API Layer Refactor (Priority: HIGH)

### Step 2.1: Refactor Pydantic Model Defaults

**Files**: `api/models/network.py`, `api/models/training.py`
**Changes**: Replace 16 inline Field defaults with imported constants
**Risk**: Low — values unchanged, only import source changes

### Step 2.2: Refactor Lifecycle Manager

**File**: `api/lifecycle/manager.py`
**Changes**: Replace 8 hardcoded values with imported constants
**Risk**: Low — timeout values preserved exactly

### Step 2.3: Refactor Application URLs and Timeouts

**File**: `api/app.py`
**Changes**: Replace 8 hardcoded URLs, timeouts, and config strings
**Risk**: Medium — URL defaults affect service discovery; verify integration tests

### Step 2.4: Refactor Middleware and Routes

**Files**: `api/middleware.py`, `api/routes/decision_boundary.py`
**Changes**: Replace HTTP status codes and resolution limits
**Risk**: Low

### Step 2.5: Refactor Service Launcher

**File**: `api/service_launcher.py`
**Changes**: Replace 3 timeout values
**Risk**: Low

### Step 2.6: Refactor Observability

**File**: `api/observability.py`
**Changes**: Replace log rotation config, Sentry rate, Prometheus buckets
**Risk**: Low

---

## Phase 3: Core Layer Refactor (Priority: MEDIUM)

### Step 3.1: Refactor Worker Security

**File**: `api/workers/security.py`
**Changes**: Replace 4 security-related defaults
**Risk**: Low

### Step 3.2: Refactor Candidate Unit

**File**: `candidate_unit/candidate_unit.py`
**Changes**: Replace 2 randomization constants
**Risk**: Very Low

### Step 3.3: Refactor Spiral Data Provider

**Files**: `spiral_problem/data_provider.py`, `spiral_problem/spiral_problem.py`
**Changes**: Replace hardcoded JuniperData URLs
**Risk**: Low — existing environment variable overrides should be preserved

### Step 3.4: Refactor Snapshots

**File**: `snapshots/snapshot_serializer.py`
**Changes**: Replace 1 legacy format name string
**Risk**: Very Low

---

## Phase 4: Validation (Priority: HIGH)

### Step 4.1: Run Full Test Suite

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-cascor/src
pytest tests/ -v --tb=short
```

### Step 4.2: Run Pre-commit Hooks

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-cascor
pre-commit run --all-files
```

### Step 4.3: Validate Constants Importability

**Task**: Add test that imports all constants and verifies they are non-None

### Step 4.4: Validate Settings/Constants Alignment

**Task**: Add test that verifies Pydantic model Field defaults match constants values

---

## Phase 5: Documentation & Release (Priority: MEDIUM)

### Step 5.1: Update AGENTS.md

Add section documenting:
- New `constants_api` module
- Constants/settings hierarchy
- How to add new constants

### Step 5.2: Update CHANGELOG.md

Add entry for constants refactor under appropriate version

### Step 5.3: Create Release Description

Document all changes for the release, referencing the analysis document

---

## Execution Order

```
Phase 1 (Infrastructure) → Phase 2 (API Layer) → Phase 3 (Core Layer) → Phase 4 (Validation) → Phase 5 (Documentation)
```

All phases are sequential. Each step within a phase can be verified independently before proceeding to the next.
