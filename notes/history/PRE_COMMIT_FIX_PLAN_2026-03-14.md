# Pre-Commit Fix Plan — 2026-03-14

**Date**: 2026-03-14
**Branch**: `fix/pre-commit-errors`
**Status**: Complete
**Prior Fix Plans**: `PRE_COMMIT_FIX_PLAN.md` (2026-03-11), `FIX_PLAN_flake8_test_lint_2026-03-13.md`

---

## Overview

Running `pre-commit run --all-files` on juniper-cascor produces **2 failing hooks** (bandit source scan, pytest-unit) out of 23 total hooks. All other hooks pass (including shellcheck, yamllint, markdownlint, flake8, mypy, black, isort, coverage gate).

## Passing Hooks (21/23)

- check-yaml, check-toml, check-json, end-of-file-fixer, trailing-whitespace
- check-merge-conflict, check-added-large-files, check-case-conflict, check-ast
- debug-statements, detect-private-key
- black, isort, flake8 (source), flake8 (tests)
- mypy, bandit (tests), markdownlint, shellcheck, yamllint
- coverage-gate (80% threshold met — 93.79%)

---

## Issue 1: Bandit Source Scan — 7 Findings in `api/service_launcher.py` and `api/app.py`

### Findings

| # | Check | File | Line | Description |
|---|-------|------|------|-------------|
| 1 | B104 | `api/app.py` | 69 | `"0.0.0.0"` in env_overrides dict |
| 2 | B404 | `api/service_launcher.py` | 17 | `import subprocess` |
| 3 | B110 | `api/service_launcher.py` | 63 | try/except/pass in `_close_log()` |
| 4 | B110 | `api/service_launcher.py` | 73 | try/except/pass in `_cleanup_at_exit()` |
| 5 | B110 | `api/service_launcher.py` | 105 | try/except/pass in `wait_for_health()` |
| 6 | B310 | `api/service_launcher.py` | 102 | `urllib.request.urlopen()` call |
| 7 | B603 | `api/service_launcher.py` | 155 | `subprocess.Popen()` call |

### Root Cause

`api/service_launcher.py` is a new file (companion service subprocess launcher for local dev) that was added after the previous bandit fixes. Its patterns are all intentional:

- **B104**: `"0.0.0.0"` is passed as an environment variable override for juniper-data's listen address in local dev mode, not a socket bind.
- **B404/B603**: `subprocess` is the core purpose of this module — it manages child processes.
- **B110**: try/except/pass in cleanup/teardown is standard defensive code — failure during cleanup should not propagate.
- **B310**: `urllib.request.urlopen` is used for health polling an internal service URL constructed from configuration, not user input.

### Fix

Add inline `# nosec` annotations to each finding with explanatory comments:

| File | Line | Annotation |
|------|------|------------|
| `api/app.py` | 69 | `# nosec B104 — env override for local dev, not a socket bind` |
| `api/service_launcher.py` | 17 | `# nosec B404 — subprocess is the core purpose of this module` |
| `api/service_launcher.py` | 63-64 | `# nosec B110 — cleanup must not propagate exceptions` |
| `api/service_launcher.py` | 73-74 | `# nosec B110 — cleanup must not propagate exceptions` |
| `api/service_launcher.py` | 102 | `# nosec B310 — internal health check URL from configuration` |
| `api/service_launcher.py` | 105-106 | `# nosec B110 — health poll retries on any exception` |
| `api/service_launcher.py` | 155 | `# nosec B603 — command is from settings, not user input` |

### Files Modified

- `src/api/app.py` (line 69)
- `src/api/service_launcher.py` (lines 17, 63, 73, 102, 105, 155)

---

## Issue 2: pytest-unit — 2 Test Failures (UUID Double-Set Tests)

### Failing Tests

1. `test_candidate_unit_coverage_deep.py::TestUUIDMethods::test_set_uuid_double_set_error`
2. `test_spiral_problem_coverage_deep.py::TestUUIDMethods::test_set_uuid_double_set_calls_exit`

### Root Cause

Both tests mock `os._exit` expecting the `set_uuid()` method to call `os._exit(1)` on double-set. However, the production code was previously changed from `os._exit(1)` to `sys.exit(1)`. The `sys.exit(1)` call raises `SystemExit`, which is not caught by the `os._exit` mock, causing the test to fail with an uncaught `SystemExit` exception.

### Fix

Update both tests to use `pytest.raises(SystemExit)` instead of mocking `os._exit`:

**Before:**
```python
with patch("os._exit") as mock_exit:
    candidate.set_uuid("new-uuid")
    mock_exit.assert_called_once_with(1)
```

**After:**
```python
with pytest.raises(SystemExit) as exc_info:
    candidate.set_uuid("new-uuid")
assert exc_info.value.code == 1
```

### Files Modified

- `src/tests/unit/test_candidate_unit_coverage_deep.py` (lines 669-671)
- `src/tests/unit/test_spiral_problem_coverage_deep.py` (lines 1312-1314)

---

## Implementation Order

1. Fix bandit nosec annotations (config/annotation changes, no behavioral risk)
2. Fix UUID double-set tests (test changes, no production code changes)
3. Verify all hooks pass with `pre-commit run --all-files`

## Verification

```bash
cd /path/to/worktree
pre-commit run --all-files
# Expected: All 23 hooks pass (including Skipped)
```
