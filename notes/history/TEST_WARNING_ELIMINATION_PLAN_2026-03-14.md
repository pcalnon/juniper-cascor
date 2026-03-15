# Test Warning Elimination Plan — 2026-03-14

**Date**: 2026-03-14
**Branch**: `fix/test-warnings`
**Status**: Complete
**Baseline**: 2204 passed, 1 skipped, **135 warnings**, 3 subtests passed (267.45s)
**Target**: 0 warnings (excluding third-party log output)

---

## Overview

The test suite produces 135 pytest-collected warnings across 4 categories. All originate from either third-party libraries or test-code patterns that can be fixed at source. One additional log-level warning (`grow_network`) appears after the test run but is not a pytest warning — it's intentional logger output from tests that exercise early-exit paths.

---

## Root Cause: Config File Precedence

The `pytest.ini` at `src/tests/pytest.ini` already contains filters for starlette, sentry_sdk, and matplotlib warnings. However, **`pyproject.toml` at the repo root takes precedence** over `pytest.ini` in a subdirectory — pytest finds `pyproject.toml` first when searching upward from the test directory and uses its `[tool.pytest.ini_options]` section. The `pyproject.toml` filterwarnings list is **missing** the starlette, sentry_sdk, and matplotlib filters.

This is the root cause for 135 of the 135 warnings.

---

## Warning Category 1: Starlette TestClient `timeout` (16 warnings)

### Warning

```
starlette/testclient.py:439: DeprecationWarning: You should not use the 'timeout' argument with the TestClient.
```

### Source

`src/tests/integration/test_juniper_data_e2e.py` lines 64-71: `_RequestsSessionAdapter.request()`, `.get()`, `.post()` forward `**kwargs` including `timeout=30` (set by JuniperDataClient) to Starlette TestClient.

### Fix (source-level)

Strip `timeout` from kwargs before forwarding to TestClient:

```python
def request(self, method, url, **kwargs):
    kwargs.pop("timeout", None)
    return self._convert(self._tc.request(method, url, **kwargs))

def get(self, url, **kwargs):
    kwargs.pop("timeout", None)
    return self._convert(self._tc.get(url, **kwargs))

def post(self, url, **kwargs):
    kwargs.pop("timeout", None)
    return self._convert(self._tc.post(url, **kwargs))
```

### Fix (belt-and-suspenders filter)

Add to `pyproject.toml` filterwarnings:

```
"ignore::DeprecationWarning:starlette.*",
```

### Files Modified

- `src/tests/integration/test_juniper_data_e2e.py` (lines 64-71)
- `pyproject.toml` (filterwarnings)

---

## Warning Category 2: sentry-sdk `asyncio.iscoroutinefunction` (118 warnings)

### Warning

```
sentry_sdk/integrations/starlette.py:427: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated
sentry_sdk/integrations/fastapi.py:76: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated
```

### Source

Python 3.14 deprecated `asyncio.iscoroutinefunction()` in favor of `inspect.iscoroutinefunction()`. sentry-sdk's FastAPI and Starlette integrations still use the deprecated API. No upstream fix is available yet.

### Fix

Add to `pyproject.toml` filterwarnings:

```
"ignore::DeprecationWarning:sentry_sdk.*",
```

This cannot be fixed at source — it requires an upstream sentry-sdk release.

### Files Modified

- `pyproject.toml` (filterwarnings)

---

## Warning Category 3: matplotlib `FigureCanvasAgg is non-interactive` (1 warning)

### Warning

```
cascor_plotter/cascor_plotter.py:126: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
```

### Source

`test_plot_dataset_static` in `test_cascade_correlation_coverage_extended.py:797-814` calls `matplotlib.use("Agg")` at line 803, then calls `CascadeCorrelationNetwork.plot_dataset()` which calls `plt.show()` at `cascor_plotter.py:126`. The `matplotlib.use("Agg")` call is too late — matplotlib is already initialized by the time this test runs.

### Fix (source-level)

1. Set matplotlib backend globally in `conftest.py` `pytest_configure()` before any matplotlib imports
2. Mock `plt.show()` in the test to prevent the production code from calling it

### Fix (belt-and-suspenders filter)

Add to `pyproject.toml` filterwarnings:

```
"ignore:FigureCanvasAgg is non-interactive:UserWarning",
```

### Files Modified

- `src/tests/conftest.py` (add `matplotlib.use("Agg")` in `pytest_configure()`)
- `src/tests/unit/test_cascade_correlation_coverage_extended.py` (mock `plt.show()`)
- `pyproject.toml` (filterwarnings)

---

## Non-Warning: `grow_network` Log Message

### Message

```
[WARNING] CascadeCorrelationNetwork: grow_network: No validation was performed (training loop exited early or did not execute). Epochs completed: 0/
```

### Source

This is `logger.warning()` output, not a `warnings.warn()` call. It appears after the pytest summary because it's printed to stderr by tests that intentionally exercise the `grow_network` early-exit path (e.g., `test_grow_network_no_validation_fallback`, `test_grow_network_below_correlation_threshold`).

### Fix

No fix needed — this is expected behavior. The log message confirms the tests are correctly exercising the fallback code path. It is not a pytest warning and does not appear in the warnings count.

---

## Implementation Plan

### Step 1: Update `pyproject.toml` filterwarnings

Add the three missing filters from `pytest.ini` to align both configs:

```python
filterwarnings = [
    "ignore::DeprecationWarning:torch.*",
    "ignore::DeprecationWarning:numpy.*",
    "ignore::PendingDeprecationWarning:h5py.*",
    "ignore::DeprecationWarning:pkg_resources.*",
    "ignore::DeprecationWarning:google.*",
    "ignore::pytest.PytestUnraisableExceptionWarning",
    # Third-party: Starlette TestClient timeout deprecation
    "ignore::DeprecationWarning:starlette.*",
    # Third-party: sentry-sdk uses deprecated asyncio.iscoroutinefunction (Python 3.14+)
    "ignore::DeprecationWarning:sentry_sdk.*",
    # matplotlib non-interactive backend warning in headless test environments
    "ignore:FigureCanvasAgg is non-interactive:UserWarning",
]
```

### Step 2: Fix `_RequestsSessionAdapter` (source-level fix for starlette warnings)

Strip `timeout` from kwargs in all three adapter methods.

### Step 3: Fix matplotlib backend and test

1. Add `matplotlib.use("Agg")` to `conftest.py` `pytest_configure()`
2. Update `test_plot_dataset_static` to mock `plt.show()` instead of relying on `matplotlib.use()`

### Step 4: Verify

1. Run full test suite — expect 0 warnings
2. Run pre-commit — expect all hooks pass
3. Confirm all 2204 tests still pass

---

## Verification Commands

```bash
# Run tests and confirm 0 warnings
cd src && python -m pytest tests/ -m "unit and not slow" --tb=short -p no:warnings-are-errors -W error::UserWarning

# Full pre-commit
pre-commit run --all-files
```
