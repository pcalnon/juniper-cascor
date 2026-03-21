# Pre-Commit Compliance Remediation Plan

**Project**: juniper-cascor
**Date**: 2026-03-21
**Branch**: `fix/pre-commit-compliance`
**Status**: Implemented and Validated

---

## Executive Summary

The juniper-cascor pre-commit suite has 3 failing hooks with 9 total violations across
5 source files and 1 configuration file. All violations are low-severity, zero-risk
code hygiene issues. This plan addresses every failure with validated fixes organized
into a streamlined two-phase implementation.

---

## Root Cause Analysis

### Failing Hooks

| Hook | Exit Code | Violations | Root Cause Category |
|------|-----------|------------|---------------------|
| Flake8 (source) | 1 | 5 | Dead imports, variable naming, code style |
| Bandit (source) | 1 | 3 | False positives (already resolved in current code) |
| Bandit (tests) | 1 | 1 | Missing rule exemption in config |

### Detailed Root Causes

#### RC-1: Unused Import `MessageType` (F401)

- **File**: `src/api/workers/coordinator.py:20`
- **Code**: `from api.workers.protocol import BinaryFrame, MessageType, WorkerProtocol`
- **Root Cause**: `MessageType` was imported during Phase 4 security hardening when the
  worker protocol system was designed. The coordinator uses `WorkerProtocol` builder
  methods that internally reference `MessageType`, but never needs it directly.
- **Validation**: Confirmed `api/workers/__init__.py` imports `MessageType` from
  `protocol.py` directly, NOT through `coordinator.py`. Removing this import has
  zero cascading impact.

#### RC-2: Import Shadowing `field` (F402)

- **File**: `src/api/workers/protocol.py:249`
- **Code**: `for field in required:` shadows `from dataclasses import dataclass, field` (line 24)
- **Root Cause**: Loop variable named `field` shadows the `dataclasses.field` import.
  The import IS actively used at lines 351, 381, 385 for dataclass field factories.
  The shadowing is scoped to the `validate_task_result` method and does not cause a
  runtime bug, but violates Flake8's F402 rule.

#### RC-3: Unnecessary Generator (C401)

- **File**: `src/api/workers/security.py:260`
- **Code**: `unique = len(set(f"{c:.6f}" for c in recent_corrs))`
- **Root Cause**: Generator expression passed to `set()` instead of using a set
  comprehension. Flagged by `flake8-comprehensions` (C401).

#### RC-4: Unused Import Constant (F401)

- **File**: `src/cascade_correlation/cascade_correlation.py:125`
- **Code**: `_CASCADE_CORRELATION_NETWORK_WORKER_THREAD_COUNT` imported but never used
- **Root Cause**: Imported during Phase 1c parallelism improvements. The config system
  handles worker thread count via `CascadeCorrelationConfig.worker_thread_count`. The
  raw constant is used in `cascade_correlation_config.py` (which has its own import)
  but not in `cascade_correlation.py`.

#### RC-5: Unused Loop Variable (B007)

- **File**: `src/cascade_correlation/cascade_correlation.py:769`
- **Code**: `for task_idx, candidate_data_tuple, training_inputs in tasks:`
- **Root Cause**: `task_idx` is destructured from the task tuple but never referenced
  in the loop body. The loop only uses `candidate_data_tuple` and `training_inputs`.

#### RC-6: Missing B404 Exemption in Test Bandit Config

- **File**: `.pre-commit-config.yaml:251` and `src/tests/unit/test_cascade_correlation_security.py:57`
- **Code**: `import subprocess` inside `test_blocks_subprocess()` flagged as B404
- **Root Cause**: The test deliberately imports `subprocess` to construct a malicious
  pickle payload that tries to call `subprocess.call()`, verifying that
  `RestrictedUnpickler` blocks it. B404 is missing from the test bandit skip list,
  which already exempts B101, B104, B108, B110, B301, B311, B403, B614.

#### RC-7/8/9: Bandit Source False Positives (B311, B108)

- **Status**: NOT present in current code state
- **Analysis**: These violations (B311 at lines 248/190, B108 at line 172) were
  reported by the initial pre-commit run but are not reproducible. The B311 rule is
  already in the source bandit skip list. These may have been from a prior code state
  that was committed (see commit `bc64e4e`).

---

## Solution Analysis

### Issue 1: F401 `MessageType` (coordinator.py)

| Solution | Approach | Pros | Cons | Recommendation |
|----------|----------|------|------|----------------|
| A | Remove from import | Clean, zero-risk | None | **SELECTED** |
| B | Add `# noqa: F401` | Preserves for future use | Masks warning, bad precedent | Rejected |
| C | Find missing usage | N/A - no usage exists | N/A | Not applicable |

### Issue 2: F402 `field` Shadowing (protocol.py)

| Solution | Approach | Pros | Cons | Recommendation |
|----------|----------|------|------|----------------|
| A | Rename to `field_name` | Clear, descriptive, eliminates shadow | Minor diff | **SELECTED** |
| B | Remove `field` import | Not viable - import is used at lines 351, 381, 385 | N/A | Not viable |

### Issue 3: C401 Generator (security.py)

| Solution | Approach | Pros | Cons | Recommendation |
|----------|----------|------|------|----------------|
| A | Set comprehension `{...}` | Idiomatic, marginally faster | None | **SELECTED** |

### Issue 4: F401 Unused Constant (cascade_correlation.py)

| Solution | Approach | Pros | Cons | Recommendation |
|----------|----------|------|------|----------------|
| A | Remove from import | Consistent with existing TODO pattern | None | **SELECTED** |
| B | Add `# noqa: F401` | Inconsistent with established convention | Bad precedent | Rejected |

### Issue 5: B007 Unused Variable (cascade_correlation.py)

| Solution | Approach | Pros | Cons | Recommendation |
|----------|----------|------|------|----------------|
| A | Rename to `_task_idx` | Preserves semantic meaning | Slightly verbose | **SELECTED** |
| B | Replace with `_` | Most concise | Loses semantic info | Rejected |

### Issue 6+7: B404 Test Exemption (.pre-commit-config.yaml)

| Solution | Approach | Pros | Cons | Recommendation |
|----------|----------|------|------|----------------|
| A | Inline `# nosec B404` | Precise targeting | Breaks test convention (all-config) | Rejected |
| B | Add B404 to skip list | Consistent with all other test exemptions | Slightly broad | **SELECTED** |
| C | Both A and B | Defense in depth | Redundant | Over-engineering |

---

## Implementation Plan

### Phase 1: Configuration Fix

**Commit**: `chore: add B404 to test bandit skip list`

**Rationale**: Configuration first ensures the pre-commit environment is correct
before validating code changes.

**Changes**:

1. `.pre-commit-config.yaml` line 251:
   - FROM: `--skip=B101,B104,B108,B110,B301,B311,B403,B614`
   - TO: `--skip=B101,B104,B108,B110,B301,B311,B403,B404,B614`
2. Add comment after line 258:
   - `# B404: import subprocess (security tests deliberately import subprocess)`

### Phase 2: Code Fixes

**Commit**: `fix: resolve flake8 lint violations for pre-commit compliance`

**Changes** (in order of application):

1. **coordinator.py:20** - Remove `MessageType` from import:
   - FROM: `from api.workers.protocol import BinaryFrame, MessageType, WorkerProtocol`
   - TO: `from api.workers.protocol import BinaryFrame, WorkerProtocol`

2. **protocol.py:249-251** - Rename loop variable:
   - FROM: `for field in required:` / `if field not in msg:` / `f"...{field}"`
   - TO: `for field_name in required:` / `if field_name not in msg:` / `f"...{field_name}"`

3. **security.py:260** - Convert to set comprehension:
   - FROM: `unique = len(set(f"{c:.6f}" for c in recent_corrs))`
   - TO: `unique = len({f"{c:.6f}" for c in recent_corrs})`

4. **cascade_correlation.py:125** - Remove unused constant from import:
   - Remove `_CASCADE_CORRELATION_NETWORK_WORKER_THREAD_COUNT,`

5. **cascade_correlation.py:769** - Rename unused loop variable:
   - FROM: `for task_idx, candidate_data_tuple, training_inputs in tasks:`
   - TO: `for _task_idx, candidate_data_tuple, training_inputs in tasks:`

### Phase 3: Validation

1. Run `pre-commit run --all-files` - all hooks must pass
2. Run targeted unit tests for affected modules
3. Run full test suite via `bash scripts/run_tests.bash`
4. Verify import chains with smoke tests

### Phase 4: PR and Merge

1. Push branch to remote
2. Create PR against `main`
3. Verify CI passes
4. Merge

---

## Test Strategy

### Existing Test Coverage (Sufficient)

| Fix | Test File | Coverage |
|-----|-----------|----------|
| 1 (coordinator F401) | `test_worker_coordinator.py` | 6 test classes |
| 2 (protocol F402) | `test_worker_protocol.py` | `TestValidateTaskResult` (6 tests) |
| 3 (security C401) | `test_worker_security.py` | `TestAnomalyDetector` (7 tests) |
| 4 (cascor F401) | `test_cascade_correlation_*.py` | 6 test files |
| 5 (cascor B007) | (indirect) | Via coordinator test pipeline |
| 6+7 (config) | N/A | Pre-commit validation |

### New Tests to Add

1. **Regression test for dataclass field import integrity** (Priority 1):
   Test that `TaskAssignment` and `TaskResultMessage` dataclass fields with
   `default_factory` still work correctly after the `field` variable rename.

2. **Pre-commit compliance smoke test** (Priority 2):
   A test that runs `flake8` on the affected source files to catch lint regressions
   early in the test suite, before the full pre-commit run.

### Validation Sequence

```bash
# Step 1: Pre-commit (all hooks)
cd /path/to/worktree && pre-commit run --all-files

# Step 2: Targeted tests
cd src/tests && python -m pytest unit/api/test_worker_coordinator.py -v
cd src/tests && python -m pytest unit/api/test_worker_protocol.py -v
cd src/tests && python -m pytest unit/api/test_worker_security.py -v
cd src/tests && python -m pytest unit/test_cascade_correlation_security.py -v

# Step 3: Full suite
cd src/tests && bash scripts/run_tests.bash

# Step 4: Import smoke test
cd src && python -c "from api.workers.coordinator import WorkerCoordinator; print('OK')"
cd src && python -c "from api.workers.protocol import WorkerProtocol, TaskAssignment; print('OK')"
cd src && python -c "from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork; print('OK')"
```

---

## Risk Assessment

| Fix | Risk | Behavioral Change | Regression Potential |
|-----|------|-------------------|---------------------|
| 1 | None | None | Zero |
| 2 | None | None (error messages unchanged) | Zero |
| 3 | None | Semantically identical | Zero |
| 4 | None | None | Zero |
| 5 | None | None (variable was unused) | Zero |
| 6+7 | None | None (config-only) | Zero |

**Overall Risk**: **Negligible**. All fixes are code hygiene changes with no
behavioral impact.

---

## Prioritization

| Priority | Fix | Rationale |
|----------|-----|-----------|
| P1 | Fix 6+7 (B404 config) | Configuration must be correct before validating code |
| P2 | Fix 1 (F401 coordinator) | Simple import removal, zero risk |
| P2 | Fix 4 (F401 cascor) | Simple import removal, zero risk |
| P3 | Fix 5 (B007 cascor) | Naming convention, zero risk |
| P3 | Fix 3 (C401 security) | Mechanical syntax change |
| P4 | Fix 2 (F402 protocol) | Highest-complexity fix (variable rename in loop) |

---

## Future Prevention

All detection mechanisms are already in place through pre-commit hooks:
- **F401/F402/B007/C401**: Caught by source Flake8 hook with bugbear and comprehensions plugins
- **B404**: Now covered by updated test Bandit skip list
- **Import shadowing**: Caught by Flake8 F402/F811 rules

No additional tooling or CI changes are needed. The current pre-commit configuration
is comprehensive and well-tailored for the project's ML development context.

---

## Additional Fixes Discovered During Implementation

Three additional Bandit source false positives were discovered during validation
that were pre-existing in the codebase:

| File | Rule | Issue | Fix |
|------|------|-------|-----|
| `protocol.py:38` | B105 | `TOKEN_REFRESH = "token_refresh"` flagged as hardcoded password | `# nosec B105` — enum value |
| `cascade_correlation.py:2873` | B110 | `except Exception: pass` in process kill cleanup | `# nosec B110` — intentional resilience |
| `cascade_correlation_config.py:163` | B107 | `ws_worker_token_secret: str = ""` default parameter | `# nosec B107` — set from env at runtime |

### New Test Files Added

| File | Purpose | Tests |
|------|---------|-------|
| `test_lint_compliance.py` | AST-based lint checks for future detection | 162 parametrized tests |
| `test_worker_protocol.py` (additions) | Regression tests for dataclass field integrity | 5 new tests |

---

## Validation Results

- **Pre-commit**: All 20 hooks pass (0 failures)
- **Targeted unit tests**: 109 passed (coordinator, protocol, security, cascor security)
- **Lint compliance tests**: 162 passed
- **Import smoke tests**: All 4 modules verified
- **Full test suite**: Validated via `run_tests.bash`
