# Test Suite and CI/CD Enhancement Development Plan

**Project**: Juniper Cascade Correlation Neural Network  
**Document Version**: 1.0  
**Created**: 2026-02-03  
**Author**: AI Engineering Assistant (Amp)  
**Status**: Planning Document (No Code Changes Made)

---

## Executive Summary

This development plan consolidates findings from two independent test suite and CI/CD audits:

- [TEST_SUITE_AUDIT_CASCOR_AMP.md](TEST_SUITE_AUDIT_CASCOR_AMP.md)
- [TEST_SUITE_AUDIT_CASCOR_CLAUDE.md](TEST_SUITE_AUDIT_CASCOR_CLAUDE.md)

The plan prioritizes 23 identified issues by **severity**, **impact**, and **risk**, providing:

- Validated findings with supporting evidence
- Prioritized fixes organized into execution phases
- Effort estimates and expected durations
- Dependencies and risk mitigations

### Priority Distribution

| Priority | Count | Description |
|----------|-------|-------------|
| P0 (Critical) | 5 | CI correctness & test-signal integrity |
| P1 (High) | 5 | Tests not testing real code / portability |
| P2 (Medium) | 5 | Tooling hygiene improvements |
| P3 (Low) | 3 | Housekeeping & enhancements |

---

## Table of Contents

1. [Validated Findings](#1-validated-findings)
2. [Prioritized Issue List](#2-prioritized-issue-list)
3. [Execution Plan](#3-execution-plan)
4. [Issue Details and Remediation](#4-issue-details-and-remediation)
5. [Dependencies Matrix](#5-dependencies-matrix)
6. [Risk Assessment and Mitigations](#6-risk-assessment-and-mitigations)
7. [Success Metrics](#7-success-metrics)
8. [Appendix A: Files Requiring Changes](#appendix-a-files-requiring-changes)
9. [Appendix B: Effort Estimation Guide](#appendix-b-effort-estimation-guide)

---

## 1. Validated Findings

All findings from both audit reports were independently verified against the current codebase. The following sections detail each validated issue.

### 1.1 Test Suite Issues

| ID | Issue | Validation Status | Source File(s) |
|----|-------|------------------|----------------|
| TST-001 | Always-passing tests (assert True) | ✅ Confirmed | `test_training_workflow.py:186-204` |
| TST-002 | Mock-only tests not exercising source | ✅ Confirmed | `test_log_config_coverage.py` (67+ tests) |
| TST-003 | Hardcoded absolute paths | ✅ Confirmed | `test_quick.py:9`, `test_final.py:9`, `test_cascor_fix.py:9`, `test_p1_fixes.py:171` |
| TST-004 | Skipped critical deterministic test | ✅ Confirmed | `test_comprehensive_serialization.py:41-42` |
| TST-005 | Test file without pytest functions | ✅ Confirmed | `test_quick.py` (only `main()`) |
| TST-006 | Tests in lastfailed cache | ✅ Confirmed | 34 tests in `.pytest_cache/v/cache/lastfailed` |
| TST-007 | Tests skipped due to missing dill | ✅ Confirmed | 26+ tests in `test_utils_extended.py`, `test_utils_coverage.py` |
| TST-008 | Test with empty code block | ✅ Confirmed | `test_residual_error.py:43-46` (`pass` instead of test logic) |
| TST-009 | Test returns boolean without asserting | ✅ Confirmed | `test_final.py:79-86` |
| TST-010 | Weak OR logic assertions | ✅ Confirmed | `test_training_workflow.py:224` |

### 1.2 CI/CD Configuration Issues

| ID | Issue | Validation Status | Source File(s) |
|----|-------|------------------|----------------|
| CI-001 | Python 3.14 specified (unreleased) | ✅ Confirmed | `ci.yml:55`, `.pre-commit-config.yaml:34`, `pyproject.toml:134` |
| CI-002 | Slow tests never run in CI | ✅ Confirmed | `ci.yml:175,271` (`-m "not slow"`) |
| CI-003 | pip-audit doesn't fail on vulnerabilities | ✅ Confirmed | `ci.yml:385` (uses `|| echo "::warning::"`) |
| CI-004 | Bandit SARIF upload continue-on-error | ✅ Confirmed | `ci.yml:375` |
| CI-005 | Integration tests only on PR/main/develop | ✅ Confirmed | `ci.yml:214` |

### 1.3 Pre-commit and Linting Issues

| ID | Issue | Validation Status | Source File(s) |
|----|-------|------------------|----------------|
| LINT-001 | MyPy has 15 error codes disabled | ✅ Confirmed | `.pre-commit-config.yaml:163-177` |
| LINT-002 | Flake8 ignores important codes | ✅ Confirmed | `.pre-commit-config.yaml:141` |
| LINT-003 | Line length set to 512 | ✅ Confirmed | `.pre-commit-config.yaml:115,128,140`, `pyproject.toml:43,64` |
| LINT-004 | Tests excluded from all linting | ✅ Confirmed | `.pre-commit-config.yaml:149,179,200` |
| LINT-005 | Shellcheck severity = error only | ✅ Confirmed | `.pre-commit-config.yaml:225` |

### 1.4 Coverage Configuration Issues

| ID | Issue | Validation Status | Source File(s) |
|----|-------|------------------|----------------|
| COV-001 | Coverage sources missing directories | ✅ Confirmed | `pyproject.toml:105` (missing: spiral_problem, log_config, utils, profiling, remote_client, cascor_plotter) |
| COV-002 | pytest warnings suppressed | ✅ Confirmed | `pyproject.toml:80` (`-p no:warnings`) |

---

## 2. Prioritized Issue List

### P0 — Critical (CI Correctness & Test-Signal Integrity)

| ID | Issue | Impact | Risk | Effort |
|----|-------|--------|------|--------|
| CI-001 | Python 3.14 specified | CI failures/inconsistency | High | S |
| TST-001 | Always-passing tests | False confidence in tests | Critical | S |
| TST-007 | 26+ tests skipped (missing dill) | Coverage overstated | High | S |
| COV-001 | Coverage sources missing | Unmeasured code | High | M |
| CI-003 | pip-audit doesn't fail | Security not enforced | High | S |

### P1 — High (Tests Not Testing Real Code / Portability)

| ID | Issue | Impact | Risk | Effort |
|----|-------|--------|------|--------|
| TST-002 | Mock-only LogConfig tests | Zero real coverage | High | M |
| TST-003 | Hardcoded absolute paths | CI/portability broken | Medium | M |
| TST-004 | Skipped deterministic test | Critical behavior untested | High | M-L |
| TST-005 | test_quick.py no pytest functions | Tests not collected | Medium | S |
| TST-009 | Test returns without asserting | Test never fails | Medium | S |

### P2 — Medium (Tooling Hygiene)

| ID | Issue | Impact | Risk | Effort |
|----|-------|--------|------|--------|
| COV-002 | pytest warnings suppressed | Hidden deprecations | Medium | S-M |
| LINT-004 | Tests excluded from linting | Test code quality not enforced | Medium | M |
| LINT-002 | Flake8 ignores dangerous codes | Bugs allowed | Medium | L |
| LINT-001 | MyPy 15 codes disabled | Type checking ineffective | Medium | L-XL |
| LINT-003 | Line length 512 | Readability harmed | Low | M-L |

### P3 — Low (Housekeeping & Enhancements)

| ID | Issue | Impact | Risk | Effort |
|----|-------|--------|------|--------|
| TST-006 | 34 tests in lastfailed cache | Local state (not critical) | Low | S |
| CI-002 | Slow tests never run | Regressions may slip | Medium | M |
| TST-008/TST-010 | Weak/empty test logic | Minor false positives | Low | S |

---

## 3. Execution Plan

### Phase 0: Stabilize CI Baseline

**Goal**: Ensure CI runs correctly and produces meaningful results  
**Duration**: 1 day  
**Total Effort**: Small (2-4 hours)

| Task | Issue ID | Description | Effort | Owner |
|------|----------|-------------|--------|-------|
| 0.1 | CI-001 | Update Python version from 3.14 to 3.13 in all configs | S | TBD |
| 0.2 | TST-007 | Add `dill` to test dependencies in conda spec / CI | S | TBD |
| 0.3 | TST-006 | Add `.pytest_cache/` to `.gitignore`, remove if committed | S | TBD |

**Exit Criteria**:

- [ ] CI pipeline runs successfully on all configured Python versions
- [ ] All 26+ previously-skipped dill tests now execute
- [ ] `.pytest_cache/` is gitignored

---

### Phase 1: Make Test Results Meaningful

**Goal**: Fix tests that produce false positives  
**Duration**: 1-2 days  
**Total Effort**: Medium (6-10 hours)

| Task | Issue ID | Description | Effort | Owner |
|------|----------|-------------|--------|-------|
| 1.1 | TST-001 | Fix always-passing tests in `test_training_workflow.py` | S | TBD |
| 1.2 | TST-005 | Convert `test_quick.py` to proper pytest format or delete | S | TBD |
| 1.3 | TST-009 | Add proper assertions to `test_final.py` | S | TBD |
| 1.4 | TST-010 | Fix weak OR logic in gradient tests | S | TBD |
| 1.5 | TST-003 | Replace hardcoded paths with `tmp_path` fixtures | M | TBD |
| 1.6 | TST-008 | Fix empty test block in `test_residual_error.py` | S | TBD |

**Exit Criteria**:

- [ ] No tests use `assert True` in both branches of try/except
- [ ] All test files have proper `test_*` functions
- [ ] No hardcoded absolute paths in test files
- [ ] Tests that check conditions properly assert results

---

### Phase 2: Improve Test Realism & Coverage

**Goal**: Ensure tests exercise real code and coverage is accurately measured  
**Duration**: 2-4 days  
**Total Effort**: Medium-Large (12-24 hours)

| Task | Issue ID | Description | Effort | Owner |
|------|----------|-------------|--------|-------|
| 2.1 | TST-002 | Refactor `test_log_config_coverage.py` to use real LogConfig | M | TBD |
| 2.2 | COV-001 | Expand coverage sources to include all src directories | M | TBD |
| 2.3 | TST-004 | Stabilize and enable deterministic resume test | M-L | TBD |
| 2.4 | CI-003 | Configure pip-audit to fail on high/critical vulnerabilities | S | TBD |

**Exit Criteria**:

- [ ] LogConfig tests instantiate real objects
- [ ] Coverage includes `spiral_problem`, `log_config`, `utils`, `profiling`, `remote_client`, `cascor_plotter`
- [ ] Deterministic resume test runs (or has `xfail(strict=True)` with tracking issue)
- [ ] pip-audit fails CI on high/critical vulnerabilities

---

### Phase 3: Improve Tooling Quality Gates

**Goal**: Strengthen linting and type checking enforcement  
**Duration**: 3-5 days (can be incremental)  
**Total Effort**: Large (24-40 hours)

| Task | Issue ID | Description | Effort | Owner |
|------|----------|-------------|--------|-------|
| 3.1 | COV-002 | Remove `-p no:warnings`, add targeted filterwarnings | S-M | TBD |
| 3.2 | LINT-004 | Include test files in flake8/mypy with relaxed initial rules | M | TBD |
| 3.3 | LINT-002 | Re-enable dangerous flake8 codes (E722, F401) | M | TBD |
| 3.4 | CI-002 | Add scheduled nightly/weekly workflow for slow tests | M | TBD |
| 3.5 | LINT-001 | Gradually re-enable mypy error codes (staged per module) | L-XL | TBD |
| 3.6 | LINT-003 | Reduce line length to 100-120 (enforce on new/touched files) | M-L | TBD |

**Exit Criteria**:

- [ ] pytest warnings surface (with targeted suppressions for known issues)
- [ ] Test files are linted
- [ ] E722 and F401 no longer ignored
- [ ] Slow tests run on schedule
- [ ] At least 5 mypy error codes re-enabled
- [ ] Line length policy documented and enforced on new code

---

## 4. Issue Details and Remediation

### 4.1 P0 Issues (Critical)

#### CI-001: Python 3.14 Specified (Unreleased)

**Files Affected**:

- `.github/workflows/ci.yml` (line 55): `PYTHON_TEST_VERSION: "3.14"`
- `.pre-commit-config.yaml` (line 34): `python: python3.14`
- `pyproject.toml` (line 134): `python_version = "3.14"`
- `.github/workflows/ci.yml` (line 68): matrix includes "3.14"

**Current State**:

```yaml
# ci.yml
PYTHON_TEST_VERSION: "3.14"
```

**Remediation**:

```yaml
# ci.yml
PYTHON_TEST_VERSION: "3.13"

# Matrix should be:
python-version: ["3.11", "3.12", "3.13"]
```

```yaml
# .pre-commit-config.yaml
default_language_version:
  python: python3.13
```

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.13"
```

**Effort**: S (< 1 hour)  
**Risk**: Low (straightforward version change)

---

#### TST-001: Always-Passing Tests

**Files Affected**: `src/tests/unit/test_training_workflow.py`

**Current State** (lines 186-204):

```python
def test_forward_with_wrong_input_size(self, simple_network):
    wrong_input = torch.randn(10, simple_network.input_size + 1)
    try:
        output = simple_network.forward(wrong_input)
        assert True  # ALWAYS PASSES
    except (RuntimeError, ValueError):
        assert True  # ALWAYS PASSES
```

**Remediation**:

```python
def test_forward_with_wrong_input_size(self, simple_network):
    """Test forward pass with wrong input size raises appropriate error."""
    wrong_input = torch.randn(10, simple_network.input_size + 1)
    with pytest.raises((RuntimeError, ValueError)):
        simple_network.forward(wrong_input)
```

**Additional instances to fix**:

- `test_train_with_mismatched_sizes` (lines 192-204): Same pattern
- `test_gradients_exist_after_backward` (line 224): OR logic `assert ... or loss is not None`

**Effort**: S (30-60 minutes)  
**Risk**: Low (clear fix pattern)

---

#### TST-007: Tests Skipped Due to Missing dill

**Files Affected**:

- `src/tests/unit/test_utils_extended.py`: 11+ skipif decorators
- `src/tests/unit/test_utils_coverage.py`: 15+ skipif decorators

**Current State**:

```python
try:
    import dill
    HAS_DILL = True
except ImportError:
    HAS_DILL = False

@pytest.mark.skipif(not HAS_DILL, reason="Requires dill package")
def test_check_object_pickleability_none(self):
    ...
```

**Remediation**:

Add `dill` to test dependencies:

```yaml
# conf/conda_environment.yaml (add to dependencies)
- dill>=0.3.6

# Or in CI workflow:
pip install dill
```

**Effort**: S (< 30 minutes)  
**Risk**: Low (adding dependency only affects test environment)

---

#### COV-001: Coverage Sources Missing Directories

**Files Affected**: `pyproject.toml`

**Current State** (line 105):

```toml
[tool.coverage.run]
source = ["src/cascade_correlation", "src/candidate_unit", "src/snapshots"]
```

**Missing Directories**:

- `src/spiral_problem/`
- `src/log_config/`
- `src/cascor_plotter/`
- `src/profiling/`
- `src/utils/`
- `src/remote_client/`

**Remediation**:

```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/data/*",
    "*/logs/*",
]
```

**Note**: Coverage percentage may initially drop when measuring all code. Consider temporarily lowering `fail_under` threshold.

**Effort**: M (1-3 hours including test adjustments)  
**Risk**: Medium (may expose coverage gaps requiring additional tests)

---

#### CI-003: pip-audit Doesn't Fail on Vulnerabilities

**Files Affected**: `.github/workflows/ci.yml`

**Current State** (line 385):

```yaml
pip-audit -r reports/security/pip-freeze.txt || echo "::warning::Vulnerabilities found in dependencies"
```

**Remediation**:

```yaml
- name: Run pip-audit (Dependency Vulnerabilities)
  run: |
    # Fail on high and critical severity vulnerabilities
    pip-audit --require-hashes=false \
      --strict \
      --vulnerability-service osv \
      --desc on \
      || (echo "::error::Critical/High vulnerabilities found" && exit 1)
```

**Alternative with allowlist**:

```yaml
# Create .pip-audit-ignore.txt for known acceptable vulnerabilities
pip-audit --ignore-vuln PYSEC-XXXX-XXXX || exit 1
```

**Effort**: S (< 1 hour)  
**Risk**: Low (may require immediate dependency updates if vulnerabilities exist)

---

### 4.2 P1 Issues (High)

#### TST-002: Mock-Only LogConfig Tests

**Files Affected**: `src/tests/unit/test_log_config_coverage.py`

**Current State** (67+ tests):

```python
@pytest.fixture
def mock_log_config(self):
    mock = MagicMock()  # NOT TESTING REAL CLASS
    mock.uuid = "test-uuid-12345"
    # ... 20+ mock attributes
    return mock

def test_get_uuid_returns_uuid(self, mock_log_config):
    result = LogConfig.get_uuid(mock_log_config)
    assert result == "test-uuid-12345"  # Tests MagicMock, not LogConfig
```

**Impact**: Zero actual coverage of `LogConfig` class.

**Remediation Strategy**:

1. Create a real `LogConfig` fixture with minimal configuration
2. Use `tmp_path` for file-based operations
3. Mock only external dependencies (filesystem handlers, network)

```python
@pytest.fixture
def real_log_config(tmp_path):
    """Create a real LogConfig instance for testing."""
    log_file = tmp_path / "test.log"
    config_file = tmp_path / "logging_config.yaml"

    # Create minimal config file
    config_file.write_text("version: 1\nhandlers: {}")

    return LogConfig(
        log_file_name=str(log_file),
        log_config_file_path=str(config_file.parent),
        log_level_name="DEBUG",
    )

def test_get_uuid_returns_valid_uuid(self, real_log_config):
    result = real_log_config.get_uuid()
    assert isinstance(result, str)
    assert len(result) > 0
    # Validate UUID format
    import uuid
    uuid.UUID(result)  # Raises if invalid
```

**Effort**: M (4-8 hours to refactor 67+ tests)  
**Risk**: Medium (may uncover actual LogConfig bugs)

---

#### TST-003: Hardcoded Absolute Paths

**Files Affected**:

| File | Line | Hardcoded Path |
|------|------|----------------|
| `test_quick.py` | 9 | `/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src` |
| `test_final.py` | 9 | `/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src` |
| `test_cascor_fix.py` | 9 | `/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src` |
| `test_p1_fixes.py` | 171 | `/home/pcalnon/.../cascade_correlation.py` (file open) |

**Remediation**:

```python
# Remove hardcoded paths - imports should work via conftest.py path setup
# Instead of:
sys.path.append("/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src")

# Use relative path from __file__:
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

For file reading in `test_p1_fixes.py`:

```python
# Instead of:
with open("/home/pcalnon/.../cascade_correlation.py", "r") as f:

# Use:
import pathlib
source_file = pathlib.Path(__file__).parent.parent.parent / "cascade_correlation" / "cascade_correlation.py"
with open(source_file, "r") as f:
```

**Effort**: M (2-4 hours)  
**Risk**: Low (path resolution is straightforward)

---

#### TST-004: Skipped Critical Deterministic Test

**Files Affected**: `src/tests/integration/test_comprehensive_serialization.py`

**Current State** (lines 41-43):

```python
@pytest.mark.slow
@pytest.mark.skip(reason="Long-running deterministic correctness test - run manually with --run-long")
@pytest.mark.timeout(600)
def test_deterministic_training_resume(self):
    """
    Critical test: Train → Save → Load → Resume should be identical to continuous training.
    This is the most important test for deterministic reproducibility.
    """
```

**Problem**: The `--run-long` flag mentioned in the skip reason doesn't exist in `conftest.py`.

**Remediation Options**:

**Option A**: Add `--run-long` flag and remove skip

```python
# conftest.py
def pytest_addoption(parser):
    parser.addoption("--run-long", action="store_true", default=False, help="Run long-running tests")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-long"):
        skip_long = pytest.mark.skip(reason="need --run-long option to run")
        for item in items:
            if "long" in item.keywords:
                item.add_marker(skip_long)

# test file
@pytest.mark.slow
@pytest.mark.long
@pytest.mark.timeout(600)
def test_deterministic_training_resume(self):
```

**Option B**: Convert to xfail with tracking

```python
@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="Tracking issue: #XXX - determinism needs stabilization")
@pytest.mark.timeout(600)
def test_deterministic_training_resume(self):
```

**Option C**: Run in scheduled CI only (recommended)

```python
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_deterministic_training_resume(self):
```

Then add scheduled workflow (see CI-002).

**Effort**: M-L (4-16 hours depending on root cause of instability)  
**Risk**: Medium (may require fixing underlying determinism issues)

---

#### TST-005: Test File Without pytest Functions

**Files Affected**: `src/tests/unit/test_quick.py`

**Current State**:

```python
def main():
    print("Quick test of CascadeCorrelationNetwork fixes...")
    # ... 60+ lines of manual testing
    return success

if __name__ == "__main__":
    success = main()
    print(f"Test {'PASSED' if success else 'FAILED'}")
```

**Problem**: No `test_*` functions, so pytest discovers the file but runs zero tests.

**Remediation Options**:

**Option A**: Convert to proper pytest test

```python
import pytest

@pytest.mark.unit
def test_cascor_candidate_training_quick(fast_training_params):
    """Quick test of CascadeCorrelationNetwork fixes."""
    torch.manual_seed(42)
    x = torch.randn(8, 2)
    y = torch.randint(0, 2, (8, 1)).float()

    network = CascadeCorrelationNetwork(
        input_size=2,
        output_size=1,
        candidate_pool_size=2,
        # ... other params
    )

    results = network.train_candidates(x, y, network.calculate_residual_error(x, y))

    # Proper assertions
    assert results is not None
    # ... more specific assertions
```

**Option B**: Delete file if redundant

If the functionality is covered by other tests (e.g., `test_cascor_fix.py`), delete `test_quick.py`.

**Effort**: S (30-60 minutes)  
**Risk**: Low

---

#### TST-009: Test Returns Boolean Without Asserting

**Files Affected**: `src/tests/unit/test_final.py`

**Current State** (lines 79-86):

```python
success = different_correlations and non_zero_correlations and consistent_correlations

print(f"\n✅ SUCCESS: {success}")
print(f"   - Different correlations: {different_correlations}")
# ...
return success  # Returns but doesn't assert!
```

**Remediation**:

```python
# Replace return with assertions
assert different_correlations, "All candidates have identical correlations"
assert non_zero_correlations, "Some candidates have zero correlation"
assert consistent_correlations, "Returned correlation doesn't match instance correlation"
```

**Effort**: S (15 minutes)  
**Risk**: Low

---

### 4.3 P2 Issues (Medium)

#### COV-002: pytest Warnings Suppressed

**Files Affected**: `pyproject.toml`

**Current State** (line 80):

```toml
addopts = [
    "-p", "no:warnings",
    # ...
]
```

**Remediation**:

```toml
addopts = [
    "-ra",
    "-q",
    "--strict-markers",
    "--strict-config",
    "--continue-on-collection-errors",
    "--tb=short",
]

filterwarnings = [
    "ignore::DeprecationWarning:torch.*",
    "ignore::DeprecationWarning:numpy.*",
    "ignore::PendingDeprecationWarning:h5py.*",
    # Add specific suppressions as needed
]
```

**Effort**: S-M (1-3 hours to identify and configure filters)  
**Risk**: Low (may surface many warnings initially)

---

#### LINT-004: Tests Excluded from Linting

**Files Affected**: `.pre-commit-config.yaml`

**Current State**:

```yaml
# Flake8 (line 149)
files: ^src/
exclude: ^src/tests/

# MyPy (line 179)
exclude: ^src/tests/

# Bandit (line 200)
exclude: ^src/tests/
```

**Remediation**:

**Phase 1**: Include with relaxed rules

```yaml
# Flake8
- id: flake8
  name: Lint with Flake8
  files: ^src/
  # Remove: exclude: ^src/tests/

# Add separate flake8 for tests with relaxed rules
- id: flake8
  name: Lint tests with Flake8 (relaxed)
  files: ^src/tests/
  args:
    - --max-line-length=512
    - --extend-ignore=E203,E265,E266,E501,W503,E402,C901,B008
    # Keep some important checks: E722, F401
```

**Phase 2**: Gradually tighten test linting rules

**Effort**: M (2-6 hours for initial inclusion + fixing issues)  
**Risk**: Medium (will surface existing issues in test code)

---

#### LINT-002: Flake8 Ignores Dangerous Codes

**Files Affected**: `.pre-commit-config.yaml` (line 141)

**Current State**:

```yaml
--extend-ignore=E203,E265,E266,E501,W503,E722,E402,E226,C409,C901,B008,B904,B905,B907,F401
```

**Most Dangerous Ignores**:

| Code | Description | Risk |
|------|-------------|------|
| E722 | Bare `except:` clause | Security/reliability risk |
| F401 | Module imported but unused | Dead code accumulation |
| C901 | Function too complex | Maintainability |

**Remediation** (staged approach):

**Phase 1**: Remove highest-risk ignores

```yaml
--extend-ignore=E203,E265,E266,E501,W503,E402,E226,C409,C901,B008,B904,B905,B907
# Removed: E722, F401
```

Fix resulting errors in codebase.

**Phase 2**: Remove complexity ignore

```yaml
--extend-ignore=E203,E265,E266,E501,W503,E402,E226,C409,B008,B904,B905,B907
# Removed: C901
```

Refactor complex functions or add `# noqa: C901` with justification.

**Effort**: L (8-16 hours to fix violations)  
**Risk**: Medium (requires code changes)

---

#### LINT-001: MyPy Has 15 Error Codes Disabled

**Files Affected**: `.pre-commit-config.yaml` (lines 163-177)

**Current State**:

```yaml
args:
  - --disable-error-code=attr-defined
  - --disable-error-code=return-value
  - --disable-error-code=arg-type
  - --disable-error-code=assignment
  - --disable-error-code=no-redef
  - --disable-error-code=override
  - --disable-error-code=var-annotated
  - --disable-error-code=index
  - --disable-error-code=misc
  - --disable-error-code=call-arg
  - --disable-error-code=func-returns-value
  - --disable-error-code=has-type
  - --disable-error-code=str-bytes-safe
  - --disable-error-code=call-overload
  - --disable-error-code=return
```

**Remediation** (staged approach):

**Phase 1**: Re-enable critical codes

```yaml
# Remove from disabled list:
# - return-value (catches wrong return types)
# - arg-type (catches wrong argument types)
```

**Phase 2**: Per-module enforcement

Use mypy configuration with per-module overrides:

```toml
# pyproject.toml
[[tool.mypy.overrides]]
module = "cascade_correlation.*"
# Enable stricter checking for core module
disable_error_codes = ["attr-defined", "misc"]

[[tool.mypy.overrides]]
module = "spiral_problem.*"
# More relaxed for problem-specific code
disable_error_codes = ["attr-defined", "misc", "assignment"]
```

**Effort**: L-XL (16-40+ hours depending on type annotation state)  
**Risk**: High (may require significant refactoring)

---

#### LINT-003: Line Length Set to 512

**Files Affected**:

- `.pre-commit-config.yaml` (lines 115, 128, 140)
- `pyproject.toml` (lines 43, 64)

**Remediation** (non-breaking approach):

**Option A**: Enforce on new/modified files only

Use `git diff` based approach or configure Black to only format changed files.

**Option B**: Reduce limit with auto-formatting

1. Set reasonable limit (100 or 120)
2. Run Black auto-formatter on entire codebase
3. Review changes in single commit
4. Commit as "formatting: standardize line length to 120"

```yaml
# .pre-commit-config.yaml
- id: black
  args:
    - --line-length=120

- id: isort
  args:
    - --line-length=120

- id: flake8
  args:
    - --max-line-length=120
```

**Effort**: M-L (4-8 hours for codebase-wide formatting)  
**Risk**: Low (formatting only, no logic changes)

---

### 4.4 P3 Issues (Low)

#### TST-006: Tests in lastfailed Cache

**Files Affected**: `src/tests/.pytest_cache/v/cache/lastfailed`

**Current State**: 34 tests listed as failed.

**Remediation**:

1. Add to `.gitignore` (if not already):

```gitignore
.pytest_cache/
```

2. Remove from repo if committed:

```bash
git rm -r --cached src/tests/.pytest_cache/
```

3. Investigate and fix the 34 failing tests (separate effort)

**Effort**: S (15 minutes for gitignore, separate effort for fixing tests)  
**Risk**: Low

---

#### CI-002: Slow Tests Never Run in CI

**Files Affected**: `.github/workflows/ci.yml`

**Current State** (lines 175, 271):

```yaml
python -m pytest \
  -m "unit and not slow" \
```

**Remediation**: Add scheduled workflow

```yaml
# Add to ci.yml or create new file: slow-tests.yml
name: Slow Tests (Nightly)

on:
  schedule:
    - cron: '0 3 * * *'  # 3 AM UTC daily
  workflow_dispatch:

jobs:
  slow-tests:
    name: Slow Tests
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      # ... environment setup ...

      - name: Run Slow Tests
        run: |
          python -m pytest \
            -m "slow" \
            src/tests/ \
            --verbose \
            --timeout=600 \
            --junitxml=reports/junit/junit-slow.xml
```

**Effort**: M (2-4 hours)  
**Risk**: Low

---

## 5. Dependencies Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEPENDENCY FLOW                                  │
│                                                                     │
│  Phase 0 (Baseline)                                                 │
│  ┌─────────┐     ┌───────────┐     ┌─────────────┐                 │
│  │ CI-001  │────▶│ TST-007   │────▶│ All other   │                 │
│  │ Fix Py  │     │ Add dill  │     │ tasks       │                 │
│  │ version │     │           │     │             │                 │
│  └─────────┘     └───────────┘     └─────────────┘                 │
│                                                                     │
│  Phase 1 (Test Integrity)                                          │
│  ┌─────────┐     ┌───────────┐                                     │
│  │ TST-001 │────▶│ LINT-004  │  (Fix tests before linting them)    │
│  │ Fix     │     │ Include   │                                     │
│  │ asserts │     │ tests     │                                     │
│  └─────────┘     └───────────┘                                     │
│                                                                     │
│  Phase 2 (Coverage)                                                │
│  ┌─────────┐     ┌───────────┐                                     │
│  │ TST-002 │────▶│ COV-001   │  (Real tests before measuring)      │
│  │ Real    │     │ Expand    │                                     │
│  │ objects │     │ sources   │                                     │
│  └─────────┘     └───────────┘                                     │
│                                                                     │
│  Phase 3 (Tooling)                                                 │
│  ┌─────────┐     ┌───────────┐     ┌─────────────┐                 │
│  │ LINT-002│────▶│ LINT-001  │────▶│ LINT-003    │                 │
│  │ Flake8  │     │ MyPy      │     │ Line length │                 │
│  │ codes   │     │ codes     │     │             │                 │
│  └─────────┘     └───────────┘     └─────────────┘                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Dependencies**:

1. **CI-001 → All**: Python version must be fixed before CI is reliable
2. **TST-007 → COV-001**: dill tests must run before coverage measurement is accurate
3. **TST-001 → LINT-004**: Fix false-positive tests before including tests in linting
4. **TST-002 → COV-001**: Real object tests before expanding coverage sources
5. **LINT-002 → LINT-001**: Flake8 before MyPy (Flake8 is faster feedback loop)

---

## 6. Risk Assessment and Mitigations

### 6.1 High-Risk Items

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Coverage drops when expanding sources | High | Medium | Temporarily lower `fail_under`; track improvement plan |
| Enabling dill increases test runtime | Low | Low | dill is lightweight; monitor runtime metrics |
| Deterministic test remains flaky | Medium | High | Use `xfail(strict=True)` with tracking issue |
| Linting tests surfaces many issues | High | Medium | Phase approach: relaxed rules first |
| Line length change creates large diff | High | Low | Single atomic commit; no logic changes |

### 6.2 Rollback Strategy

Each phase should be implemented in separate PRs:

1. **Phase 0**: Single PR, easy rollback (version numbers)
2. **Phase 1**: One PR per test file, granular rollback
3. **Phase 2**: Separate PRs for mock refactor and coverage config
4. **Phase 3**: One PR per lint rule change

---

## 7. Success Metrics

### 7.1 Quantitative Metrics

| Metric | Current | Target (Phase 2) | Target (Phase 3) |
|--------|---------|------------------|------------------|
| Tests always passing incorrectly | 4+ | 0 | 0 |
| Tests using hardcoded paths | 4 | 0 | 0 |
| Tests skipped (dill) | 26+ | 0 | 0 |
| Mock-only test files | 1 | 0 | 0 |
| Coverage sources | 3 dirs | All dirs | All dirs |
| MyPy disabled codes | 15 | 15 | ≤ 10 |
| Flake8 ignored codes | 15 | 13 | ≤ 10 |
| Failed tests in cache | 34 | 0 | 0 |

### 7.2 Qualitative Metrics

- [ ] All CI runs on released Python versions
- [ ] Test failures reflect actual code issues
- [ ] Coverage accurately measures tested code
- [ ] Security scans have enforcement
- [ ] Slow tests run on schedule
- [ ] Type checking provides value

---

## Appendix A: Files Requiring Changes

### Test Files

| File | Changes Required |
|------|------------------|
| `src/tests/unit/test_training_workflow.py` | Fix assert True patterns (lines 186-204, 224) |
| `src/tests/unit/test_log_config_coverage.py` | Replace MagicMock with real LogConfig (entire file) |
| `src/tests/unit/test_quick.py` | Convert to pytest format or delete |
| `src/tests/unit/test_final.py` | Add assertions, fix path (lines 9, 79-86) |
| `src/tests/unit/test_cascor_fix.py` | Fix hardcoded path (line 9) |
| `src/tests/unit/test_p1_fixes.py` | Fix hardcoded paths (lines 9, 171) |
| `src/tests/unit/test_residual_error.py` | Fix empty code block (lines 43-46) |
| `src/tests/integration/test_comprehensive_serialization.py` | Enable or properly xfail deterministic test |
| `src/tests/conftest.py` | Add `--run-long` option if needed |

### Configuration Files

| File | Changes Required |
|------|------------------|
| `.github/workflows/ci.yml` | Fix Python version, add pip-audit enforcement, add slow test schedule |
| `.pre-commit-config.yaml` | Fix Python version, reduce ignores, include tests in linting |
| `pyproject.toml` | Fix mypy version, expand coverage, remove warning suppression |
| `conf/conda_environment.yaml` | Add dill dependency |
| `.gitignore` | Add/verify `.pytest_cache/` |

---

## Appendix B: Effort Estimation Guide

| Size | Hours | Description |
|------|-------|-------------|
| S | < 2 | Simple config change, single-file fix |
| M | 2-8 | Multiple files, moderate refactoring |
| L | 8-24 | Significant refactoring, multiple systems |
| XL | 24+ | Major architectural changes |

### Total Effort by Phase

| Phase | Effort Range | Calendar Time |
|-------|--------------|---------------|
| Phase 0 | 2-4 hours | 1 day |
| Phase 1 | 6-10 hours | 1-2 days |
| Phase 2 | 12-24 hours | 2-4 days |
| Phase 3 | 24-40 hours | 3-5 days |
| **Total** | **44-78 hours** | **7-12 days** |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | Amp | Initial development plan based on audit consolidation |

---

*This document was created as a planning artifact. No code changes have been made. All changes require explicit approval before implementation.*
