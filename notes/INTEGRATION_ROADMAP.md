# Juniper Cascor ↔ Juniper Canopy Integration Roadmap

**Created**: 2026-01-20  
**Last Updated**: 2026-01-20 08:40 CST  
**Version**: 1.2.0  
**Status**: Active - Implementation Phase  
**Author**: Development Team

---

## Executive Summary

This document tracks the integration of **Juniper Cascor** (Cascade Correlation Neural Network backend) with **Juniper Canopy** (Web-based monitoring frontend). The analysis identifies **critical blocking issues**, **test failures**, **environment dependencies**, and **architectural concerns** that must be resolved before successful integration.

### Integration Readiness Status

| Component      | Test Status                                    | Launch Status                 | Integration Ready |
| -------------- | ---------------------------------------------- | ----------------------------- | ----------------- |
| Juniper Cascor | ✅ 152 tests collected (0 errors)              | ⚠️ Shell script path issues   | 🟡 Partial        |
| Juniper Canopy | ⚠️ 84 failures, 32 errors                      | ✅ Demo mode functional       | 🟡 Partial        |

### Critical Blockers Summary (Updated 2026-01-20 08:40)

| Priority | Issue                                               | Application | Impact                             | Status        |
| -------- | --------------------------------------------------- | ----------- | ---------------------------------- | ------------- |
| P0       | ~~Missing `scipy` dependency~~                      | Canopy      | ~~Blocks all backend integration~~ | ✅ RESOLVED   |
| P0       | ~~Missing `fastapi`, `uvicorn`, etc.~~              | Canopy      | ~~216 test collection errors~~     | ✅ RESOLVED   |
| P0       | ~~Missing `a2wsgi` dependency~~                     | Canopy      | ~~228 test collection errors~~     | ✅ RESOLVED   |
| P0       | ~~Test import errors (cascade_correlation_config)~~ | Cascor      | ~~2 test collection errors~~       | ✅ RESOLVED   |
| P0       | Shell script path resolution failures               | Cascor      | Cannot launch via `./try` script   | 🔴 BLOCKING   |
| P1       | Multiprocessing manager port conflicts              | Cascor      | Parallel candidate training fails  | ⚠️ DEGRADED   |
| P1       | ~~Environment mismatch~~                            | Both        | ~~Different conda environments~~   | ✅ RESOLVED   |
| P1       | Dashboard manager test errors                       | Canopy      | 32 test errors                     | ⚠️ NEEDS FIX  |

### Progress Summary

| Metric                   | Initial (03:00) | After Env Fix (05:00) | After Code Fix (08:40) | Total Change  |
| ------------------------ | --------------- | --------------------- | ---------------------- | ------------- |
| Canopy Tests Passed      | 536             | 2223                  | 2791                   | +2255 (+421%) |
| Canopy Tests Failed      | 241             | 103                   | 84                     | -157 (-65%)   |
| Canopy Collection Errors | 216             | 228                   | 32                     | -184 (-85%)   |
| Cascor Collection Errors | 2               | 2                     | 0                      | -2 (-100%)    |
| Cascor Tests Collected   | 127             | 129                   | 152                    | +25 (+20%)    |

---

## Table of Contents

1. [Environment Analysis](#1-environment-analysis)
2. [Juniper Cascor Issues](#2-juniper-cascor-issues)
3. [Juniper Canopy Issues](#3-juniper-canopy-issues)
4. [Integration-Specific Issues](#4-integration-specific-issues)
5. [Proposed Solutions](#5-proposed-solutions)
6. [Implementation Order](#6-implementation-order)
7. [Verification Procedures](#7-verification-procedures)
8. [Appendix: Full Test Output Analysis](#appendix-full-test-output-analysis)

---

## 1. Environment Analysis

### 1.1 Conda Environment Configuration

**JuniperCascor Environment** (`/opt/miniforge3/envs/JuniperCascor`):

- Python: 3.14.2
- Key packages: pytorch, numpy, matplotlib, h5py, pytest
- Status: Active environment for Cascor development

**JuniperCanopy Environment** (`/opt/miniforge3/envs/JuniperCanopy`):

- Python: 3.14.0
- Key packages: fastapi, uvicorn, dash, dash-bootstrap-components, scipy
- Status: Configured but has missing runtime dependencies when tests run under JuniperCascor environment

### 1.2 Environment Mismatch Issue

**Status**: ✅ RESOLVED (2026-01-20)

**Problem**: Tests were being run from the JuniperCascor environment, which lacked Canopy-specific packages.

**Resolution**: The following packages were installed in JuniperCascor environment:

- fastapi
- uvicorn
- dash
- dash-bootstrap-components
- scipy

**Remaining Issue**: `a2wsgi` package is still missing - see CANOPY-P0-003 below.

### 1.3 Python Version Discrepancy

**Status**: ✅ RESOLVED (2026-01-20)

| Application | Environment   | Python Version    |
| ----------- | ------------- | ----------------- |
| Cascor      | JuniperCascor | 3.14.2            |
| Canopy      | JuniperCanopy | 3.14.2 (upgraded) |

Both environments now use Python 3.14.2.

---

## 2. Juniper Cascor Issues

### 2.1 Critical Issues (P0)

#### CASCOR-P0-001: Shell Script Path Resolution Failure

**Location**: `util/juniper_cascor.bash`, `try` script  
**Status**: 🔴 BLOCKING

**Problem**: The `./try` convenience script fails to launch the application.

**Root Cause Analysis**:

1. Helper scripts are assigned bare filenames instead of full paths:

   ```bash
   GET_OS_SCRIPT="__get_os_name.bash"
   GET_PROJECT_SCRIPT="__get_project_dir.bash"
   ```

2. `BASE_DIR` becomes empty when `__get_project_dir.bash` fails
3. Python script path resolves to `/src/./main.py` (invalid)

**Error Output**:

```bash
juniper_cascor.bash: line 67: __get_project_dir.bash: command not found
Base Dir: 
Python Script: /src/./main.py
python3: can't open file '/src/./main.py': [Errno 2] No such file or directory
```

**Proposed Solution**:

```bash
# Fix helper script paths - derive from BASH_SOURCE[0]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GET_OS_SCRIPT="${SCRIPT_DIR}/__get_os_name.bash"
GET_PROJECT_SCRIPT="${SCRIPT_DIR}/__get_project_dir.bash"
```

**Files to Modify**:

- `util/juniper_cascor.bash`
- `conf/script_util.cfg`

---

#### CASCOR-P0-002: Test Suite Timeout/Hang Issues

**Location**: Test execution  
**Status**: 🔴 BLOCKING

**Problem**: Tests timeout after 180 seconds, never completing the full suite.

**Evidence**:

```bash
collected 129 items / 2 errors
src/tests/integration/test_comprehensive_serialization.py F...EXIT_CODE: 124
```

**Root Cause Analysis**:

1. Multiprocessing tests may hang on manager operations
2. Sequential fallback in candidate training is slower
3. Integration tests with large networks take excessive time

**Proposed Solution**:

- Add explicit timeouts to test fixtures
- Mark slow tests with `@pytest.mark.slow`
- Add `--timeout=60` default for individual tests
- Configure pytest-timeout in pytest.ini

---

#### CASCOR-P0-003: Test Collection Errors - Incorrect Module Imports

**Location**: `src/tests/`  
**Status**: ✅ RESOLVED (2026-01-20)

**Problem**: 2 test files fail to collect due to incorrect import paths.

**Affected Files**:

1. `src/tests/unit/test_hdf5.py` (line 10)
2. `src/tests/integration/test_serialization.py` (line 34)

**Error**:

```python
ModuleNotFoundError: No module named 'cascade_correlation_config'
```

**Root Cause Analysis**:
The `cascade_correlation_config` module is a submodule of `cascade_correlation`, but some test files import it incorrectly as a top-level module.

**Incorrect Import** (in test files):

```python
from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
```

**Correct Import** (from conftest.py):

```python
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
```

**Files Requiring Fixes**:

| File                                          | Line         | Current Import                       | Required Change                   |
| --------------------------------------------- | ------------ | ------------------------------------ | --------------------------------- |
| `src/tests/unit/test_hdf5.py`                 | 10           | `from cascade_correlation_config...` | Add `cascade_correlation.` prefix |
| `src/tests/unit/test_hdf5.py`                 | 24           | `from cascade_correlation_config...` | Add `cascade_correlation.` prefix |
| `src/tests/integration/test_serialization.py` | 34           | `from cascade_correlation_config...` | Add `cascade_correlation.` prefix |
| `src/tests/unit/test_p1_fixes.py`             | 73, 124, 195 | `from cascade_correlation_config...` | Add `cascade_correlation.` prefix |
| `src/tests/unit/test_critical_fixes.py`       | 47, 102      | `from cascade_correlation_config...` | Add `cascade_correlation.` prefix |

**Proposed Solution**:
Change all incorrect imports from:

```python
from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
```

to:

```python
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
```

---

### 2.2 High Priority Issues (P1)

#### CASCOR-P1-001: Multiprocessing Manager Port Conflicts

**Location**: `src/cascade_correlation/cascade_correlation.py`  
**Status**: ⚠️ Known issue, sequential fallback active

**Problem**: BaseManager with fixed port causes "Address already in use" errors.

**Current Workaround**: Sequential training fallback is active.

**Impact**:

- Parallel candidate training disabled
- Training is slower but functional
- Integration with Canopy will work but performance is degraded

**Proposed Solution** (from existing DEVELOPMENT_ROADMAP.md):

- Port already changed to 0 (dynamic allocation)
- Context changed from `forkserver` to `spawn`
- Sequential fallback implemented

---

#### CASCOR-P1-002: Missing PyYAML in Original Environment Spec

**Location**: `conf/conda_environment.yaml`  
**Status**: ⚠️ Previously identified

**Problem**: `pyyaml` was not listed in environment spec.

**Proposed Solution**: Add to conda environment:

```yaml
dependencies:
  - pyyaml=6.0.3
```

---

### 2.3 Cascor Component Status Summary

| Component                 | Status        | Notes                         |
| ------------------------- | ------------- | ----------------------------- |
| CascadeCorrelationNetwork | ✅ Functional | Core algorithm works          |
| CandidateUnit             | ✅ Functional | Training works                |
| Serialization (HDF5)      | ✅ Functional | Save/load implemented         |
| Multiprocessing           | ⚠️ Degraded   | Sequential fallback active    |
| Shell Scripts             | 🔴 Broken     | Path resolution fails         |
| Test Suite                | ⚠️ Partial    | Timeouts, 2 collection errors |

---

## 3. Juniper Canopy Issues

### 3.1 Critical Issues (P0)

#### CANOPY-P0-001: Missing `scipy` Dependency at Runtime

**Location**: `src/backend/statistics.py`  
**Status**: ✅ RESOLVED (2026-01-20)

**Problem**: Backend integration module failed to import due to missing scipy.

**Resolution**: scipy installed in JuniperCascor environment via conda.

---

#### CANOPY-P0-002: Missing Core Web Framework Dependencies

**Location**: Test execution environment  
**Status**: ✅ RESOLVED (2026-01-20)

**Problem**: Core packages missing during test execution.

**Resolution**: The following packages were installed in JuniperCascor environment:

- fastapi
- uvicorn
- dash
- dash-bootstrap-components

---

#### CANOPY-P0-003: Missing `a2wsgi` Dependency

**Location**: Test execution environment  
**Status**: ✅ RESOLVED (2026-01-20)

**Problem**: 228 tests fail to collect due to missing `a2wsgi` package.

**Evidence**:

```bash
ERROR src/tests/integration/test_demo_endpoints.py::* - ModuleNotFoundError: No module named 'a2wsgi'
ERROR src/tests/integration/test_mvp.py::* - ModuleNotFoundError: No module named 'a2wsgi'
ERROR src/tests/unit/test_main_coverage.py::* - ModuleNotFoundError: No module named 'a2wsgi'
```

**Impact**: 228 test collection errors across:

- Integration tests (most affected)
- Unit tests for main.py coverage
- WebSocket control tests

**Affected Test Files**:

- `src/tests/integration/test_demo_endpoints.py`
- `src/tests/integration/test_metrics_layouts_api.py`
- `src/tests/integration/test_mvp.py`
- `src/tests/integration/test_network_stats_endpoint.py`
- `src/tests/integration/test_websocket_control.py`
- `src/tests/integration/test_websocket_message_schema.py`
- `src/tests/integration/test_websocket_state.py`
- `src/tests/unit/test_main_coverage.py`
- `src/tests/unit/test_main_coverage_extended.py`

**Root Cause**: `a2wsgi` is a WSGI-to-ASGI adapter required by the test infrastructure but not installed.

**Proposed Solution**:
Install `a2wsgi` in the JuniperCascor environment:

```bash
conda install -c conda-forge a2wsgi
# or
pip install a2wsgi
```

---

### 3.2 High Priority Issues (P1)

#### CANOPY-P1-001: Test Failures (103 Remaining)

**Location**: `src/tests/`  
**Status**: ⚠️ IN PROGRESS

**Current Test Results** (2026-01-20 05:00):

- **Passed**: 2223
- **Failed**: 103
- **Skipped**: 26
- **Collection Errors**: 228 (due to missing `a2wsgi`)

**Categories of Remaining Failures**:

| Category                        | Count | Root Cause                   |
| ------------------------------- | ----- | ---------------------------- |
| Dashboard Manager Handler Tests | ~30   | Mock/fixture issues          |
| Frontend Component Tests        | ~50   | Test infrastructure          |
| WebSocket Tests                 | ~15   | Collection blocked by a2wsgi |
| Other                           | ~8    | Various                      |

**Note**: After environment fixes, most test failures are now due to test infrastructure issues, not missing dependencies.

---

#### CANOPY-P1-002: Backend Module Import Chain Failure

**Location**: `src/backend/`  
**Status**: ✅ RESOLVED (2026-01-20)

**Resolution**: scipy installed in JuniperCascor environment. Import chain now functional.

---

### 3.3 Canopy Component Status Summary

| Component           | Status        | Notes                        |
| ------------------- | ------------- | ---------------------------- |
| FastAPI Backend     | ✅ Functional | Tests passing                |
| Dash Dashboard      | ✅ Functional | Tests passing                |
| Demo Mode           | ✅ Functional | 2223 tests pass              |
| WebSocket Manager   | ⚠️ Partial    | Some tests blocked by a2wsgi |
| CascorIntegration   | ✅ Functional | scipy now available          |
| Frontend Components | ⚠️ Partial    | 103 test failures remaining  |

---

## 4. Integration-Specific Issues

### 4.1 Path Configuration for Backend Integration

**Issue**: Canopy needs to locate Cascor backend.

**Configuration** (`conf/app_config.yaml`):

```yaml
backend:
  cascor_integration:
    backend_path: "../cascor"
```

**Problem**: Relative path assumes specific directory layout:

```bash
Juniper/
├── JuniperCascor/juniper_cascor/  ← Cascor location
└── JuniperCanopy/juniper_canopy/  ← Canopy location
```

**Proposed Solution**:

```yaml
backend:
  cascor_integration:
    backend_path: "${CASCOR_BACKEND_PATH:../JuniperCascor/juniper_cascor}"
```

Or use environment variable:

```bash
export CASCOR_BACKEND_PATH="/home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor"
```

---

### 4.2 API/Protocol Compatibility

**Status**: ⚠️ NEEDS VERIFICATION

**Integration Points**:

| Interface        | Cascor Export       | Canopy Import       | Status     |
| ---------------- | ------------------- | ------------------- | ---------- |
| Network Topology | HDF5 snapshots      | CascorIntegration   | ❓ Unknown |
| Training Metrics | Real-time state     | WebSocket broadcast | ❓ Unknown |
| Control Commands | Multiprocess queues | REST/WebSocket      | ❓ Unknown |

**Required Verification**:

1. Test HDF5 snapshot compatibility
2. Verify metrics data format alignment
3. Test control command protocol

---

### 4.3 Data Format Alignment

**Cascor Output Format** (from training):

```python
{
    'epoch': int,
    'loss': float,
    'accuracy': float,
    'hidden_units': int,
    'phase': str,  # 'output' | 'candidate'
    'correlation': float,
}
```

**Canopy Expected Format**:

```python
{
    'epoch': int,
    'train_loss': float,      # Note: different key name
    'train_accuracy': float,  # Note: different key name
    'hidden_units': int,
    'phase': str,
}
```

**Issue**: Key naming mismatch between applications.

**Proposed Solution**: Implement data adapter in Canopy (`src/backend/data_adapter.py`) to normalize keys.

---

## 5. Proposed Solutions

### 5.1 Immediate Fixes (Required for Testing)

#### FIX-001: Environment Activation in Test Scripts

**Priority**: P0  
**Effort**: S (< 1 hour)

**Changes Required**:

**Cascor** (`tests` script):

```bash
# Add after line 43
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate JuniperCascor
```

**Canopy** (`tests` script):

```bash
# Add after line 43
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate JuniperCanopy
```

---

#### FIX-002: Cascor Shell Script Path Resolution

**Priority**: P0  
**Effort**: M (1-3 hours)

**File**: `util/juniper_cascor.bash`

**Changes**:

```bash
# Replace lines 61-63 with:
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GET_OS_SCRIPT="${SCRIPT_DIR}/__get_os_name.bash"
GET_PROJECT_SCRIPT="${SCRIPT_DIR}/__get_project_dir.bash"
DATE_FUNCTIONS_SCRIPT="${SCRIPT_DIR}/__git_log_weeks.bash"
```

---

#### FIX-003: Test Timeout Configuration

**Priority**: P1  
**Effort**: S (< 1 hour)

**File**: `src/tests/pytest.ini` (both applications)

**Add**:

```ini
[pytest]
timeout = 60
timeout_method = signal
```

**Install**: `pip install pytest-timeout`

---

### 5.2 Integration Fixes

#### FIX-004: Backend Path Configuration

**Priority**: P1  
**Effort**: S (< 1 hour)

**File**: Canopy `conf/app_config.yaml`

**Change**:

```yaml
backend:
  cascor_integration:
    backend_path: "${CASCOR_BACKEND_PATH}"
    default_path: "../JuniperCascor/juniper_cascor"
```

---

#### FIX-005: Data Adapter Key Normalization

**Priority**: P2  
**Effort**: M (1-3 hours)

**File**: Canopy `src/backend/data_adapter.py`

**Add normalization function**:

```python
def normalize_metrics(raw_metrics: dict) -> dict:
    """Normalize Cascor metrics to Canopy format."""
    return {
        'epoch': raw_metrics.get('epoch'),
        'train_loss': raw_metrics.get('loss'),
        'train_accuracy': raw_metrics.get('accuracy'),
        'val_loss': raw_metrics.get('val_loss'),
        'val_accuracy': raw_metrics.get('val_accuracy'),
        'hidden_units': raw_metrics.get('hidden_units'),
        'phase': raw_metrics.get('phase'),
    }
```

---

## 6. Implementation Order

### Phase 1: Environment Fixes (Day 1)

| Order | Task                                          | Application | Effort |
| ----- | --------------------------------------------- | ----------- | ------ |
| 1.1   | FIX-001: Add conda activation to test scripts | Both        | S      |
| 1.2   | FIX-002: Fix shell script path resolution     | Cascor      | M      |
| 1.3   | FIX-003: Add test timeout configuration       | Both        | S      |

**Exit Criteria**:

- Both test suites execute completely
- `./try` script launches Cascor application
- No test collection errors

### Phase 2: Test Suite Stabilization (Day 2-3)

| Order | Task                             | Application | Effort |
| ----- | -------------------------------- | ----------- | ------ |
| 2.1   | Run full Cascor test suite       | Cascor      | -      |
| 2.2   | Run full Canopy test suite       | Canopy      | -      |
| 2.3   | Document remaining test failures | Both        | M      |
| 2.4   | Fix import-related test failures | Both        | M      |

**Exit Criteria**:

- All tests complete without timeout
- Collection errors eliminated
- Test failure count documented

### Phase 3: Integration Testing (Day 4-5)

| Order | Task                                | Application | Effort |
| ----- | ----------------------------------- | ----------- | ------ |
| 3.1   | FIX-004: Backend path configuration | Canopy      | S      |
| 3.2   | Launch Cascor, connect Canopy       | Both        | -      |
| 3.3   | Verify HDF5 snapshot compatibility  | Both        | M      |
| 3.4   | FIX-005: Data adapter normalization | Canopy      | M      |
| 3.5   | End-to-end integration test         | Both        | L      |

**Exit Criteria**:

- Canopy successfully connects to running Cascor
- Metrics displayed in dashboard
- Network topology visualization works

---

## 7. Verification Procedures

### 7.1 Cascor Verification

```bash
# 1. Activate environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate JuniperCascor

# 2. Run tests
cd /home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor
./tests

# 3. Launch application
./try

# 4. Verify main.py directly
cd src && python main.py
```

### 7.2 Canopy Verification

```bash
# 1. Activate environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate JuniperCanopy

# 2. Run tests
cd /home/pcalnon/Development/python/Juniper/JuniperCanopy/juniper_canopy
./tests

# 3. Launch demo mode
./demo

# 4. Launch with backend (requires running Cascor)
./try
```

### 7.3 Integration Verification

```bash
# Terminal 1: Start Cascor
cd /home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor
conda activate JuniperCascor
cd src && python main.py

# Terminal 2: Start Canopy
cd /home/pcalnon/Development/python/Juniper/JuniperCanopy/juniper_canopy
conda activate JuniperCanopy
export CASCOR_BACKEND_PATH="/home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor"
unset CASCOR_DEMO_MODE
cd src && uvicorn main:app --host 0.0.0.0 --port 8050

# Terminal 3: Verify connection
curl http://localhost:8050/api/health
curl http://localhost:8050/api/network/topology
```

---

## Appendix: Full Test Output Analysis

### A.1 Cascor Test Summary

```bash
Platform: linux -- Python 3.14.2, pytest-9.0.1
Collected: 129 items / 2 errors
Status: Timeout after 180 seconds
```

### A.2 Canopy Test Summary

**Initial Run (2026-01-20 03:00)**:

```bash
Platform: linux -- Python 3.14.2, pytest-9.0.1
Results: 241 failed, 536 passed, 2 skipped, 33 warnings, 216 errors
Duration: 50.18s

Error Categories:
- ModuleNotFoundError: 'scipy' (backend imports)
- ModuleNotFoundError: 'fastapi' (API tests)
- ModuleNotFoundError: 'uvicorn' (WebSocket tests)
- ModuleNotFoundError: 'dash_bootstrap_components' (frontend tests)
```

**After Environment Fixes (2026-01-20 05:00)**:

```bash
Platform: linux -- Python 3.14.2, pytest-9.0.1
Results: 103 failed, 2223 passed, 26 skipped, 87 warnings, 228 errors
Duration: 111.41s

Remaining Error Categories:
- ModuleNotFoundError: 'a2wsgi' (228 collection errors)
- Dashboard Manager handler test failures (~30)
- Frontend component test failures (~50)
```

### A.3 Key Error Patterns

**Current (After Environment Fixes)**:

| Pattern                                            | Count | Root Cause           | Status       |
| -------------------------------------------------- | ----- | -------------------- | ------------ |
| `ModuleNotFoundError: No module named 'a2wsgi'`    | 228   | Missing dependency   | 🔴 BLOCKING  |
| Dashboard handler mock/fixture issues              | ~30   | Test infrastructure  | ⚠️ NEEDS FIX |
| Frontend component test failures                   | ~50   | Test infrastructure  | ⚠️ NEEDS FIX |

**Resolved Patterns**:

| Pattern                                                                | Count | Resolution             |
| ---------------------------------------------------------------------- | ----- | ---------------------- |
| ~~`ModuleNotFoundError: No module named 'scipy'`~~                     | ~40   | ✅ Installed via conda |
| ~~`ModuleNotFoundError: No module named 'fastapi'`~~                   | ~50   | ✅ Installed via conda |
| ~~`ModuleNotFoundError: No module named 'dash_bootstrap_components'`~~ | ~80   | ✅ Installed via conda |

---

## Document History

| Date       | Version | Author           | Changes                                                       |
| ---------- | ------- | ---------------- | ------------------------------------------------------------- |
| 2026-01-20 | 1.0.0   | Development Team | Initial integration analysis                                  |
| 2026-01-20 | 1.1.0   | Development Team | Updated after environment fixes; documented new issues        |
| 2026-01-20 | 1.2.0   | Development Team | Fixed Cascor import errors; a2wsgi installed; major progress  |

---

**Next Review**: After remaining test failures addressed  
**Owner**: Development Team  
**Document Status**: Active
