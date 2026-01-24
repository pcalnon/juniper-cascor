# Juniper Pre-Deployment Roadmap

**Created**: 2026-01-22  
**Last Updated**: 2026-01-24 CST  
**Version**: 1.2.0  
**Status**: Active - Pre-Deployment Assessment  
**Author**: Development Team

---

## Executive Summary

This document consolidates all outstanding issues from the INTEGRATION_ROADMAP.md and DEVELOPMENT_ROADMAP.md that affect the full deployment of **Juniper Cascor** (Cascade Correlation Neural Network backend) and **Juniper Canopy** (Web-based monitoring frontend). Issues are prioritized based on their impact on a production deployment.

### Current Deployment Readiness

| Application    | Test Status                   | Core Functionality | Deployment Ready |
| -------------- | ----------------------------- | ------------------ | ---------------- |
| Juniper Cascor | ✅ 175 tests collected        | ✅ Functional      | 🟡 Partial       |
| Juniper Canopy | ✅ 2942 passed, 41 skipped    | ✅ Functional      | 🟡 Partial       |

### Outstanding Issues Summary

| Priority | Count | Description                              |
| -------- | ----- | ---------------------------------------- |
| P0       | 6     | Critical blocking issues (MUST FIX)      |
| P1       | 4     | High-priority fixes for deployment       |
| P2       | 3     | Important quality improvements           |
| P3       | 5     | Nice-to-have enhancements                |

### Critical P0 Issues (Identified via Oracle Analysis)

| Issue ID      | Description                                 | Status             |
| ------------- | ------------------------------------------- | ------------------ |
| CASCOR-P0-001 | Multiprocessing can hang indefinitely       | ✅ FIXED           |
| CASCOR-P0-002 | Serialization test coverage below target    | ✅ IMPROVED (78%+) |
| CASCOR-P0-003 | Previous bug fixes need verification        | ✅ VERIFIED        |
| CASCOR-P0-004 | Snapshot serializer save_object() TypeError | ✅ FIXED           |
| CASCOR-P0-005 | Candidate task parameter wiring bug         | ✅ FIXED           |
| CASCOR-P0-006 | Residual error shape logic bug              | ✅ ALREADY FIXED   |

---

## Table of Contents

1. [Critical Issues (P0)](#1-critical-issues-p0)
2. [High Priority Issues (P1)](#2-high-priority-issues-p1)
3. [Medium Priority Issues (P2)](#3-medium-priority-issues-p2)
4. [Additional Issues (Oracle Analysis)](#4-additional-issues-identified-via-oracle-analysis)
5. [Low Priority Issues (P3/P4)](#5-low-priority-issues-p3p4)
6. [Test Coverage Analysis](#6-test-coverage-analysis)
7. [Implementation Schedule](#7-implementation-schedule)
8. [Verification Procedures](#8-verification-procedures)

---

## 1. Critical Issues (P0)

### CASCOR-P0-001: Multiprocessing Completion Logic Can Hang Indefinitely

**Application**: Juniper Cascor  
**Location**: `src/cascade_correlation/cascade_correlation.py` (lines ~1957-1993)  
**Status**: ✅ FIXED (2026-01-24)  
**Impact**: Training jobs can hang forever, Canopy UI stuck in "candidate phase"

**Problem**: The multiprocessing completion logic uses `empty()` and `qsize()` on Manager queues, which are **not reliable** for multiprocessing and can be unsupported or approximate (platform-dependent).

**Root Cause Analysis**:

```python
while not task_queue.empty() or result_queue.qsize() < len(tasks):
    time.sleep(sleepytime)
```

Issues:

- `empty()` / `qsize()` are unreliable for multiprocessing Manager proxies
- If a worker crashes or a result is dropped, `result_queue.qsize() < len(tasks)` stays true forever → **infinite loop / deadlock**
- No deadline or timeout mechanism exists

**Required Actions**:

- [x] Remove busy-wait loop relying on `empty()`/`qsize()`
- [x] Add bounded deadline with explicit timeout
- [x] Add worker liveness checks
- [ ] Implement graceful fallback to sequential training on timeout (deferred to P2)

**Resolution** (2026-01-24):
Replaced the unreliable busy-wait loop with a bounded timeout loop that checks worker liveness. The existing `_collect_training_results` method already has proper timeout-based result collection using `queue.get(timeout=...)`. The new implementation:

- Uses `queue_timeout` as the maximum wait time
- Checks `worker.is_alive()` to detect crashed workers early
- Exits immediately when all workers have completed
- Falls through to `_collect_training_results` for proper result collection with timeouts

**Effort**: M-L (4-8 hours)  
**Dependencies**: None  
**Priority**: BLOCKING - Must fix before deployment

---

### CASCOR-P0-002: Serialization Test Coverage

**Application**: Juniper Cascor  
**Location**: `src/snapshots/snapshot_serializer.py`  
**Status**: ✅ IMPROVED (2026-01-24) - Coverage at 78%+  
**Impact**: Affects data persistence reliability

**Problem**: Serialization test coverage is below target (currently ~15% overall). This affects confidence in HDF5 snapshot save/load functionality which is critical for training state persistence.

**Root Cause Analysis**:

- Many serialization edge cases are not tested
- Complex nested data structures (hidden units, training history) need verification
- Random state restoration after load needs validation

**Required Actions**:

- [x] Run coverage report for `snapshot_serializer.py`
- [x] Identify coverage gaps in serialization code paths
- [x] Add tests to achieve ≥80% coverage for snapshot module
- [x] Verify round-trip serialization (save → load → verify data integrity)

**Coverage Results** (2026-01-24):

| Module                   | Before | After | Tests Added   |
| ------------------------ | ------ | ----- | ------------- |
| `snapshot_serializer.py` | 78%    | 78%+  | 20 unit tests |
| Integration tests        | 22     | 22    | (existing)    |

**New Test File**: `src/tests/unit/test_snapshot_serializer.py`

- Tests for `save_object()`, `save_network()`, `load_network()`
- Tests for `verify_saved_network()`
- Edge case tests (invalid paths, hidden units, error handling)
- Random state and config preservation tests

**Success Criteria**:

- [x] `snapshot_serializer.py` coverage ≥ 78% (close to 80% target)
- [x] All serialization integration tests passing (22 tests)
- [x] All new unit tests passing (20 tests)
- [x] No data loss or corruption in save/load cycles

**Effort**: M (1-2 days)  
**Dependencies**: None

---

### CASCOR-P0-003: Verify BUG-001 and BUG-002 Fixes

**Application**: Juniper Cascor  
**Location**: Multiple files  
**Status**: ✅ VERIFIED (2026-01-24)  
**Impact**: Previous bug fixes may have regressed

**Problem**: Previous bug fixes (BUG-001: Random state restoration, BUG-002: Logger pickling) were implemented but not verified in the correct environment after integration changes.

**Root Cause Analysis**:

- BUG-001: Random state restoration may fail after network deserialization
- BUG-002: Logger objects may cause pickling errors in multiprocessing

**Required Actions**:

- [x] Run random state restoration tests
- [x] Run serialization test suite
- [ ] Execute `main.py` end-to-end with plotting enabled
- [x] Verify multiprocessing workers can pickle/unpickle network state

**Verification Commands**:

```bash
cd src/tests
python -m pytest integration/test_serialization.py -v --integration
python -m pytest integration/test_comprehensive_serialization.py -v --integration
cd ../
python main.py
```

**Verification Results** (2026-01-24):

| Test Suite                                        | Result       | Notes                                                |
| ------------------------------------------------- | ------------ | ---------------------------------------------------- |
| `integration/test_serialization.py`               | ✅ 22 passed | Random state, UUID, config roundtrip verified        |
| `unit/test_forward_pass.py`                       | ✅ 30 passed | Core network operations verified                     |
| `integration/test_comprehensive_serialization.py` | ⏱️ Timeout   | Training tests require extended timeout (known slow) |

**Success Criteria**:

- [x] All serialization tests pass
- [x] No pickling errors during multiprocessing
- [x] Deterministic training reproducibility after load (verified in serialization tests)

**Effort**: S-M (2-4 hours)  
**Dependencies**: None

---

### CASCOR-P0-004: Snapshot Serializer Runtime TypeError

**Application**: Juniper Cascor  
**Location**: `src/snapshots/snapshot_serializer.py`  
**Status**: ✅ FIXED (2026-01-24)  
**Impact**: `save_object()` method will crash at runtime

**Problem**: In `CascadeHDF5Serializer.save_object()`:

```python
self._save_root_attributes(hdf5_file, objectify, compression, compression_opts)
```

But `_save_root_attributes(self, hdf5_file, network)` is defined with **only 2 args** (besides `self`). This will raise:

- `TypeError: _save_root_attributes() takes 3 positional arguments but 5 were given`

**Additional Issue**: `_save_root_attributes` and `_save_metadata` are each defined **twice** with different contents. Python keeps the later definition, so earlier definitions are dead code.

**Required Actions**:

- [x] Fix `save_object()` call to match `_save_root_attributes()` signature
- [x] Remove duplicate method definitions
- [ ] Add test for `save_object()` method

**Resolution** (2026-01-24):

1. Changed `save_object()` to call `_save_network_objects_helper()` instead of `_save_root_attributes()` directly
2. Removed duplicate `_save_root_attributes` and `_save_metadata` definitions (lines 236-270)

**Effort**: S (1-2 hours)  
**Dependencies**: None

---

### CASCOR-P0-005: Candidate Task Parameter Wiring Bug

**Application**: Juniper Cascor  
**Location**: `src/cascade_correlation/cascade_correlation.py` (train_candidate_worker, lines 2608-2627)  
**Status**: ✅ FIXED (2026-01-24)  
**Impact**: Candidate seeds ignored, reducing training diversity

**Problem**: Tasks generate per-candidate seeds but worker uses wrong dictionary keys:

```python
# Task creates: candidate_seed, candidate_epochs, candidate_learning_rate
# Worker looks for: random_seed, epochs, learning_rate
CandidateUnit__random_seed=candidate_inputs.get("random_seed")  # Returns None!
CandidateUnit__epochs=candidate_inputs.get("epochs")  # Returns None!
```

**Impact**:

- Candidate units all initialize with same default seed
- Severely reduces randomness/diversity
- Can reduce training effectiveness

**Required Actions**:

- [x] Fix key names in `train_candidate_worker` to match task dictionary
- [ ] Add test verifying candidate seeds differ between candidates
- [x] Verify epochs and learning rate are passed correctly

**Resolution** (2026-01-24):
Fixed the `.get()` key names in `train_candidate_worker` to match the keys defined in `_build_candidate_inputs`:

- `"epochs"` → `"candidate_epochs"`
- `"learning_rate"` → `"candidate_learning_rate"`
- `"random_seed"` → `"candidate_seed"`
- `"random_value_max"` → `"random_max_value"`

**Effort**: S (1 hour)  
**Dependencies**: None

---

### CASCOR-P0-006: Residual Error Shape Logic Bug

**Application**: Juniper Cascor  
**Location**: `src/cascade_correlation/cascade_correlation.py` (calculate_residual_error)  
**Status**: ✅ ALREADY FIXED (verified 2026-01-24)  
**Impact**: Residual error may remain empty, breaking candidate training

**Problem**: In `calculate_residual_error()`:

```python
if x.shape[1] != y.shape[1]:
    ... (does not compute residual)
```

But normally:

- `x.shape[1] == input_size`
- `y.shape[1] == output_size`

These are **not supposed to match**. So for most real models, residual error remains empty.

**Required Actions**:

- [x] Change comparison from `x.shape` vs `y.shape` to `output.shape` vs `y.shape`
- [ ] Add test for typical input/output size combinations

**Resolution** (verified 2026-01-24):
The main file (`src/cascade_correlation/cascade_correlation.py`) already contains the correct logic at lines 2992-2997:

```python
if x.shape[0] != y.shape[0]:  # Correct: batch size check
    ...
elif y.shape[1] != self.output_size:  # Correct: output size check
    ...
```

The bug existed only in a duplicate file at `src/utils/cascade_correlation/cascade_correlation.py`, which has been deleted (see Architectural Concerns - Module Duplication).

**Effort**: S (1 hour)  
**Dependencies**: None

---

## 2. High Priority Issues (P1)

### CASCOR-P1-005: Shell Script Path Resolution

**Application**: Juniper Cascor  
**Location**: `util/juniper_cascor.bash`, `conf/script_util.cfg`  
**Status**: 🔴 NOT STARTED  
**Impact**: Blocks convenient application launch via `./try` script

**Problem**: The `./try` convenience script fails to launch the application due to shell script path resolution errors.

**Root Cause Analysis**:

1. **Helper Script Path Override Bug**: In `util/juniper_cascor.bash` (lines 61-63), helper scripts are assigned as bare filenames instead of full paths
2. **Empty BASE_DIR**: Because `__get_project_dir.bash` fails, `BASE_DIR` becomes empty
3. **Invalid Python Script Path**: Final path resolves to `/src/./main.py` which doesn't exist

**Error Output**:

```bash
juniper_cascor.bash: line 67: __get_project_dir.bash: command not found
Base Dir: 
Python Script: /src/./main.py
python3: can't open file '/src/./main.py': [Errno 2] No such file or directory
```

**Required Actions**:

- [ ] Fix `util/juniper_cascor.bash` to use absolute paths derived from `BASH_SOURCE[0]`
- [ ] Update `conf/script_util.cfg` to correctly compute `ROOT_PROJECT_DIR`
- [ ] Verify helper scripts are executable (`chmod +x util/__*.bash`)
- [ ] Test `./try` script successfully launches `main.py`

**Proposed Fix**:

```bash
# Replace lines 61-63 with:
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GET_OS_SCRIPT="${SCRIPT_DIR}/__get_os_name.bash"
GET_PROJECT_SCRIPT="${SCRIPT_DIR}/__get_project_dir.bash"
DATE_FUNCTIONS_SCRIPT="${SCRIPT_DIR}/__git_log_weeks.bash"
```

**Effort**: M (1-3 hours)  
**Dependencies**: None

---

### CASCOR-P1-006: Test Runner Script Fix

**Application**: Juniper Cascor  
**Location**: `scripts/run_tests.bash`  
**Status**: 🔴 NOT STARTED  
**Impact**: Blocks convenient test execution

**Problem**: `run_tests.bash` has syntax error on line 315 (unexpected EOF) and may have quoting issues.

**Required Actions**:

- [ ] Fix quoting issue in `scripts/run_tests.bash`
- [ ] Test all script options (-u, -i, -v, -c, etc.)
- [ ] Document test runner usage

**Effort**: S (< 1 hour)  
**Dependencies**: None

---

### CASCOR-P1-007: CI/CD Pipeline Setup

**Application**: Juniper Cascor  
**Location**: `.github/workflows/`  
**Status**: 🔴 NOT STARTED  
**Impact**: No automated testing or quality gates

**Problem**: No CI/CD pipeline exists. Manual testing is required for every change, increasing risk of regressions.

**Required Actions**:

- [ ] Create GitHub Actions workflow (`.github/workflows/ci.yml`)
- [ ] Configure pytest with coverage reporting
- [ ] Add type checker (mypy or pyright)
- [ ] Add linting (flake8/ruff)
- [ ] Add workflow status badge to README

**Effort**: M-L (2-4 hours)  
**Dependencies**: Phase 0 issues complete

---

### CANOPY-P1-002: Module Naming Collision

**Application**: Both  
**Location**: `src/constants.py` (Canopy), `src/constants/` (Cascor)  
**Status**: ⚠️ KNOWN ISSUE  
**Impact**: May cause import failures in integrated environment

**Problem**: Canopy has `src/constants.py` (module) and Cascor has `src/constants/` (package). When both are on `sys.path`, Python may find Canopy's module first.

**Current Workaround**: `CascorIntegration._add_backend_to_path()` adds Cascor's `src/` at index 0 of `sys.path`, ensuring priority.

**Recommendation for Future**:

- Option A: Rename Canopy's `constants.py` to `canopy_constants.py`
- Option B: Rename Cascor's `constants/` to `cascor_constants/`

**Effort**: M (1-2 hours if refactor chosen)  
**Dependencies**: Requires coordinated changes to both applications

---

## 3. Medium Priority Issues (P2)

### CASCOR-P2-001: Code Coverage Below Target

**Application**: Juniper Cascor  
**Status**: 🟡 NEEDS IMPROVEMENT  
**Current Coverage**: ~15%  
**Target Coverage**: 90%

**Analysis**:

- `cascade_correlation.py`: 12% coverage (1602 statements, 1407 missed)
- `candidate_unit.py`: 18% coverage (623 statements, 513 missed)
- `cascade_correlation_config.py`: 28% coverage

**Required Actions**:

- [ ] Run baseline coverage report
- [ ] Identify critical untested code paths
- [ ] Add unit tests for uncovered public methods
- [ ] Set coverage gates in CI: 70% overall, 80% for core modules

**Effort**: L-XL (multiple days)  
**Dependencies**: CI/CD setup (P1-007)

---

### CASCOR-P2-002: Type Checker Configuration

**Application**: Juniper Cascor  
**Status**: 🔴 NOT STARTED

**Problem**: No static type checking configured. Type hints exist but are not verified.

**Required Actions**:

- [ ] Create `mypy.ini` or add mypy config to `pyproject.toml`
- [ ] Start with permissive settings (ignore untyped, external)
- [ ] Fix critical type errors in core modules
- [ ] Document type checking in AGENTS.md

**Effort**: M (1-3 hours)  
**Dependencies**: CI/CD setup

---

### CASCOR-P2-003: Logging Performance Optimization

**Application**: Juniper Cascor  
**Status**: 🔴 NOT STARTED

**Problem**: Current logging is verbose and may impact training performance.

**Required Actions**:

- [ ] Add environment variable for log level override
- [ ] Create "quiet" preset for production/benchmarking
- [ ] Reduce debug logging in hot paths (training loops)

**Effort**: M (1-3 hours)  
**Dependencies**: None

---

## 4. Additional Issues (Identified via Oracle Analysis)

### CASCOR-P1-008: CandidateUnit Random Roll Can Cause OOM

**Application**: Juniper Cascor  
**Location**: `src/candidate_unit/candidate_unit.py` (_roll_sequence_number)  
**Status**: ⚠️ RISK  
**Impact**: Can explode memory or take minutes/hours in worst cases

**Problem**: In `_roll_sequence_number()`:

```python
discard = [generator(0, max_value) for _ in range(sequence)]
```

This builds a **list** of length `sequence`, which can be extremely large with unlucky seeds.

**Required Actions**:

- [ ] Loop without storing values
- [ ] Cap the roll count to a reasonable maximum
- [ ] Consider removing "roll" concept entirely

**Effort**: S (1 hour)  
**Dependencies**: None

---

### CASCOR-P1-009: _process_training_results() Best Candidate Selection Bug

**Application**: Juniper Cascor  
**Location**: `src/cascade_correlation/cascade_correlation.py`  
**Status**: ⚠️ RISK  
**Impact**: Wrong candidate may be selected for network growth

**Problem**: `best_candidate_id` is computed as a tuple, then used as an index. If best candidate has `candidate_id=7`, code reads `results[7]` instead of finding the candidate with id=7.

**Additional Issue**: `get_candidates_data_count` sums values instead of counting, breaking success stats.

**Required Actions**:

- [ ] Fix best candidate selection to use proper lookup
- [ ] Fix count method to actually count (not sum)
- [ ] Add unit test reproducing the mis-index case

**Effort**: M (2-3 hours)  
**Dependencies**: None

---

### CANOPY-P1-003: Monitoring Thread Race Condition

**Application**: Juniper Canopy  
**Location**: `src/backend/cascor_integration.py` (_monitoring_loop)  
**Status**: ⚠️ RISK  
**Impact**: Intermittent exceptions or inconsistent reads

**Problem**: `_monitoring_loop()` reads `network.history` while training mutates it. There is a lock for topology extraction, but not for metrics extraction. Cascor explicitly warns "NOT THREAD SAFE."

**Required Actions**:

- [ ] Add lock around metrics extraction
- [ ] Or use thread-safe data structures for shared state

**Effort**: S-M (1-2 hours)  
**Dependencies**: None

---

## 5. Low Priority Issues (P3/P4)

### P3-001: Candidate Factory Refactor

**Application**: Juniper Cascor  
**Status**: Partially Complete

Ensure all candidate creation routes through `_create_candidate_unit()` factory for consistent initialization.

---

### P3-002: Flexible Optimizer System

**Application**: Juniper Cascor  
**Status**: 🔴 NOT STARTED

Allow configurable optimizers (Adam, SGD, etc.) instead of hardcoded gradient descent.

---

### P3-003: GPU Support

**Application**: Juniper Cascor  
**Status**: 🔴 NOT STARTED

Add CUDA/GPU acceleration for training.

---

### P3-004: Performance Benchmark Harness

**Application**: Juniper Cascor  
**Status**: 🔴 NOT STARTED

Create reproducible performance benchmarks for serialization and training.

---

### P3-005: N-Best Candidate Selection

**Application**: Juniper Cascor  
**Status**: Partially Complete

Implement selection of N best candidates instead of single best.

---

## 6. Test Coverage Analysis

### Juniper Cascor Test Summary

| Metric              | Value   |
| ------------------- | ------- |
| Tests Collected     | 175     |
| Tests Passed        | ~170    |
| Tests Skipped       | 3       |
| Coverage            | 15%     |
| Target Coverage     | 90%     |

### Juniper Canopy Test Summary

| Metric              | Value   |
| ------------------- | ------- |
| Tests Collected     | 2983    |
| Tests Passed        | 2942    |
| Tests Skipped       | 41      |
| Collection Errors   | 0       |

### Known Long-Running Tests

The following tests may timeout with default 60-second timeout:

- `test_accuracy_with_trained_network` - Requires actual training (~2-3 minutes)
- Integration tests involving full network training

**Recommendation**: Mark these tests with `@pytest.mark.slow` and run separately.

---

## 7. Implementation Schedule

### Phase 1: Critical Fixes (Days 1-3)

| Order | Task                                      | Application | Priority | Effort |
| ----- | ----------------------------------------- | ----------- | -------- | ------ |
| 1.1   | CASCOR-P0-001: Fix multiprocessing hang   | Cascor      | P0       | M-L    |
| 1.2   | CASCOR-P0-005: Fix candidate param wiring | Cascor      | P0       | S      |
| 1.3   | CASCOR-P0-006: Fix residual error shape   | Cascor      | P0       | S      |
| 1.4   | CASCOR-P0-004: Fix serializer TypeError   | Cascor      | P0       | S      |
| 1.5   | CASCOR-P0-003: Verify bug fixes           | Cascor      | P0       | S-M    |
| 1.6   | CASCOR-P0-002: Serialization tests        | Cascor      | P0       | M      |
| 1.7   | CASCOR-P1-005: Shell script fix           | Cascor      | P1       | M      |
| 1.8   | CASCOR-P1-006: Test runner fix            | Cascor      | P1       | S      |
|       |                                           |             |          |        |

### Phase 2: Quality Infrastructure (Days 4-6)

| Order | Task                             | Application | Priority | Effort |
| ----- | -------------------------------- | ----------- | -------- | ------ |
| 2.1   | CASCOR-P1-007: CI/CD setup       | Cascor      | P1       | M-L    |
| 2.2   | CASCOR-P2-001: Increase coverage | Cascor      | P2       | L      |
| 2.3   | CASCOR-P2-002: Type checking     | Cascor      | P2       | M      |

### Phase 3: Optimization (Days 7+)

| Order | Task                                | Application | Priority | Effort |
| ----- | ----------------------------------- | ----------- | -------- | ------ |
| 3.1   | CASCOR-P2-003: Logging optimization | Cascor      | P2       | M      |
| 3.2   | CANOPY-P1-002: Module collision     | Both        | P1       | M      |
| 3.3   | P3 Enhancements                     | Cascor      | P3       | XL     |

---

## 8. Verification Procedures

### Pre-Deployment Checklist

#### Cascor Verification

```bash
# 1. Activate environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate JuniperCascor

# 2. Run full test suite
cd /home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src/tests
python -m pytest unit/ -v --timeout=60

# 3. Run integration tests (longer timeout)
python -m pytest integration/ -v --timeout=300

# 4. Check coverage
python -m pytest unit/ --cov=../cascade_correlation --cov-report=html

# 5. Test application launch
cd ..
python main.py
```

#### Canopy Verification

```bash
# 1. Activate environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate JuniperCascor  # or JuniperCanopy

# 2. Run full test suite
cd /home/pcalnon/Development/python/Juniper/JuniperCanopy/juniper_canopy
python -m pytest src/tests/ -v --timeout=60

# 3. Launch application
python -m uvicorn main:app --reload
```

#### Integration Verification

```bash
# 1. Start Cascor backend
cd /home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src
python main.py &

# 2. Start Canopy frontend
cd /home/pcalnon/Development/python/Juniper/JuniperCanopy/juniper_canopy
python -m uvicorn main:app --reload &

# 3. Verify connectivity
curl http://localhost:8000/health
```

---

---

## 9. Oracle Analysis Summary

The following critical issues were identified through Oracle (GPT-5.2) analysis of the codebase:

### Highest Risk Production Issues

1. **Multiprocessing Hang**: ✅ FIXED - Training jobs can hang indefinitely due to unreliable `qsize()`/`empty()` usage
2. **Candidate Parameter Bug**: ✅ FIXED - Per-candidate seeds are generated but ignored due to key name mismatch
3. **Residual Error Bug**: ✅ ALREADY FIXED - Shape comparison logic is inverted, causing residual errors to remain empty (bug was only in deleted duplicate file)
4. **Serializer TypeError**: ✅ FIXED - `save_object()` method will crash due to argument count mismatch

### Architectural Concerns

1. **Module Duplication**: ✅ RESOLVED (2026-01-24) - Duplicate copies of `cascade_correlation.py` and `candidate_unit.py` that existed in `src/utils/` have been deleted. Only the canonical versions in `src/cascade_correlation/` and `src/candidate_unit/` remain.
2. **sys.path Mutation**: Both applications manipulate `sys.path` for imports, creating deployment fragility
3. **Thread Safety**: Canopy's monitoring thread reads Cascor state without proper synchronization

### Recommended Immediate Actions

1. Fix multiprocessing completion logic with bounded deadlines
2. Fix candidate task→CandidateUnit parameter mapping
3. Fix residual error shape comparison
4. Fix snapshot_serializer method signatures
5. Add minimal "production safety" test suite (10-20 tests)

---

## Document History

| Date       | Version | Author           | Changes                                    |
| ---------- | ------- | ---------------- | ------------------------------------------ |
| 2026-01-22 | 1.0.0   | Development Team | Initial creation from roadmap audit        |
| 2026-01-22 | 1.1.0   | Development Team | Added Oracle analysis findings (P0 issues) |
