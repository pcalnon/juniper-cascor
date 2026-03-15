# Pre-Commit Hook Failure Fix Plan

**Date**: 2026-03-11
**Branch**: (to be created)
**Status**: Complete

---

## Overview

Four pre-commit hooks fail when running `pre-commit run --all` on juniper-cascor. Two failures are configuration issues, one is a test defect, and one is a downstream consequence.

## Issue 1: Bandit "Unknown test B104" (exit code 2)

### Symptom

```
[main] ERROR Unknown test found in profile: B104
```

The "Security scan tests with Bandit (relaxed)" hook fails.

### Root Cause

The `.pre-commit-config.yaml` pins bandit at `rev: 1.7.9`. Pre-commit installs bandit from this git tag into an isolated virtualenv using the system Python (3.14). Bandit 1.7.9's B104 plugin (`general_bind_all_interfaces.py`) is decorated with `@test.checks("Str")`, which calls `getattr(ast, "Str")`. Python 3.14 removed `ast.Str` (deprecated since 3.8), causing the plugin to fail registration. When the `--skip=B104` argument is processed, bandit raises `ValueError: Unknown test found in profile: B104` because B104 was never registered.

### Fix

Upgrade bandit from `1.7.9` to `1.9.4` in both bandit hooks (lines 227 and 245 of `.pre-commit-config.yaml`). Bandit 1.9.x is compatible with Python 3.14.

### Files Modified

- `.pre-commit-config.yaml` (lines 227, 245): `rev: 1.7.9` → `rev: 1.9.4`

---

## Issue 2: Markdownlint MD024/MD040 Errors (exit code 1)

### Symptom

```
AGENTS.md:714 MD024/no-duplicate-heading [Context: "Quick Reference"]
AGENTS.md:758 MD024/no-duplicate-heading [Context: "What This Is"]
AGENTS.md:798 MD024/no-duplicate-heading [Context: "Rules"]
README.md:24 MD040/fenced-code-language [Context: "```"]
```

### Root Cause (MD024)

AGENTS.md contains template sections ("Worktree Procedures" and "Thread Handoff") that reuse sub-heading names ("Quick Reference", "What This Is", "Rules") under different parent sections. The markdownlint config (`.markdownlint.yaml`) does not configure MD024, so it uses the default behavior of flagging any duplicate heading text in the entire document regardless of nesting context.

### Root Cause (MD040)

README.md line 24 has a fenced code block containing a Unicode box-drawing architecture diagram with no language specifier.

### Fix (MD024)

Add `"MD024": { "siblings_only": true }` to `.markdownlint.json`. This allows headings with the same text under different parent sections, which correctly models the template structure.

Additionally, change the pre-commit hook from `--config=./.markdownlint.yaml` to `--config=./.markdownlint.json`. The YAML file uses a markdownlint-cli2 `config:` wrapper format that markdownlint-cli v1 does not correctly parse for nested rule options.

### Fix (MD040)

Change ` ``` ` to ` ```text ` on README.md line 24.

### Files Modified

- `.markdownlint.json`: Add `MD024` config with `siblings_only: true`
- `.pre-commit-config.yaml`: Change `--config` to use `.markdownlint.json`
- `README.md` (line 24): Add `text` language specifier

---

## Issue 3: pytest-unit Process Killed (exit code 3)

### Symptom

Tests run to ~55% completion, then the pytest process terminates abruptly with exit code 3. No test summary is printed.

### Root Cause

Five tests in `src/tests/unit/test_main_coverage.py` call `main()` with mocked `Logger`/`LogConfig`/`SpiralProblem` but do NOT mock:
- `os.environ.get("JUNIPER_DATA_URL")` — so the real env var lookup returns `None`
- `os._exit` — so the real `os._exit(3)` on line 238 of `main.py` is called

`os._exit()` performs immediate C-level process termination, bypassing all Python cleanup including pytest's test runner, atexit handlers, and coverage data flushing. This kills the entire pytest process.

### Affected Tests

1. `TestMainFunction::test_main_happy_path` (line 157)
2. `TestMainFunction::test_main_logs_startup_messages` (line 206)
3. `TestMainFunction::test_main_creates_spiral_problem_with_correct_params` (line 217)
4. `TestMainFunction::test_main_calls_evaluate_with_correct_params` (line 229)
5. `TestMainFunctionLogging::test_main_logs_log_config_creation_success` (line 247)

### Fix

Update the `mock_dependencies` fixture (used by the first 4 tests) to additionally mock:
- `main.os._exit` with `side_effect=SystemExit` as a safety net
- `os.environ` via `patch.dict(os.environ, {"JUNIPER_DATA_URL": "http://localhost:8100"})`
- `urllib.request.urlopen` to prevent real HTTP calls (note: `urllib.request` is imported inside `main()`, so the mock target is `urllib.request.urlopen`, not `main.urllib.request.urlopen`)

Update `test_main_logs_log_config_creation_success` (which sets up its own mocks) with the same additional mocks.

### Files Modified

- `src/tests/unit/test_main_coverage.py`: Update `mock_dependencies` fixture and `test_main_logs_log_config_creation_success`

---

## Issue 4: Coverage Gate "No data to report" (exit code 1)

### Symptom

```
No data to report.
```

### Root Cause

This is a downstream consequence of Issue 3. When `os._exit(3)` kills the pytest process, the `.coverage` data file is never written. The coverage-gate hook then runs `coverage report --fail-under=80` but finds no data.

### Fix

Resolving Issue 3 fixes this automatically. No separate change needed.

---

## Additional Issue: New Bandit Checks in 1.9.4

Upgrading bandit from 1.7.9 to 1.9.4 introduced new security checks that flag existing intentional code patterns:

| Check | File | Resolution |
|-------|------|------------|
| B110 (try_except_pass) | `src/api/routes/health.py:52` | Added inline `# nosec B110` — intentional for health check resilience |
| B310 (urllib_urlopen) | `src/main.py:246` | Added inline `# nosec B310` — internal health check URL from env var |
| B614 (pytorch_load) | `src/tests/unit/test_utils_coverage.py:38` | Added B614 to test bandit skip list — `torch.load` expected in tests |

---

## Implementation Order

1. Fix Bandit version (config change, no code risk)
2. Fix Markdownlint config + README (config/doc change, no code risk)
3. Fix test mocks (code change, requires careful verification)
4. Address new bandit findings from version upgrade
5. Verify all hooks pass with `pre-commit run --all`

## Verification

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-cascor
pre-commit run --all
```

All 23 hooks pass (including Skipped hooks).
