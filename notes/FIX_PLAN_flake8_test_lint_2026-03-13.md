# Fix Plan: Flake8 Test Lint Failures

**Date**: 2026-03-13
**Trigger**: `pre-commit run --all-files` — "Lint tests with Flake8 (relaxed)" hook fails with 5 violations
**Scope**: `src/tests/unit/` — 5 files, 5 violations

---

## Root Cause Analysis

All 5 failures are flake8 lint violations in test files caught by the `flake8-bugbear` and `flake8-comprehensions` plugins. No functional test failures — unit tests and coverage gate both pass.

## Issues and Fixes

### 1. C408 — `dict()` call should be dict literal (3 occurrences)

**Rule**: `flake8-comprehensions` C408 flags `dict(key=val)` calls that can be rewritten as `{"key": val}` literals.

| File | Line | Current | Fix |
|------|------|---------|-----|
| `test_cascade_correlation_coverage_deep.py` | 46 | `defaults = dict(input_size=2, ...)` | `defaults = {"input_size": 2, ...}` |
| `test_cascade_correlation_coverage_final.py` | 41 | `defaults = dict(input_size=2, ...)` | `defaults = {"input_size": 2, ...}` |
| `test_snapshot_serializer_coverage_final.py` | 32 | `defaults = dict(input_size=2, ...)` | `defaults = {"input_size": 2, ...}` |

### 2. B017 — `pytest.raises(Exception)` too broad (1 occurrence)

**Rule**: `flake8-bugbear` B017 flags `pytest.raises(Exception)` as dangerous because it can mask unrelated exceptions (e.g., typos causing `NameError`). A specific exception type should be used.

| File | Line | Current | Fix |
|------|------|---------|-----|
| `test_remaining_coverage_deep.py` | 149 | `with pytest.raises(Exception):` | `with pytest.raises(WebSocketDisconnect):` |

**Analysis**: The test verifies that a WebSocket connection fails when `ws_manager` is `None`. The handler at `src/api/websocket/training_stream.py:33-35` calls `websocket.close(code=1011)`, which causes Starlette's TestClient to raise `WebSocketDisconnect`. This exception type is already used in `src/tests/unit/api/test_websocket_auth.py`.

**Additional change**: Add `from starlette.websockets import WebSocketDisconnect` import.

### 3. B007 — Unused loop control variable (1 occurrence)

**Rule**: `flake8-bugbear` B007 flags loop variables not used in the loop body. Convention is to prefix with `_`.

| File | Line | Current | Fix |
|------|------|---------|-----|
| `test_snapshot_serializer_coverage_deep.py` | 207 | `for i, (orig, loaded_unit) in enumerate(...)` | `for _, (orig, loaded_unit) in enumerate(...)` |

## Other Potential Issues

No other issues identified. All other pre-commit hooks pass:
- Black formatting: Passed
- isort: Passed
- Source flake8: Passed
- MyPy: Passed
- Bandit security: Passed
- Unit tests: Passed
- Coverage gate (80%): Passed

## Verification

After applying all fixes, run:
```bash
cd /home/pcalnon/Development/python/Juniper/juniper-cascor
pre-commit run flake8 --all-files
```

All 5 violations should be resolved with zero functional impact.
