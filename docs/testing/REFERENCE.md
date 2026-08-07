# Testing Reference

Complete reference documentation for the Juniper Cascor test suite.

---

## Marker Reference

### Complete Marker Table

| Marker               | Description                                | When to Use                                          | CI Behavior                                   |
|----------------------|--------------------------------------------|------------------------------------------------------|-----------------------------------------------|
| `unit`               | Unit tests for individual components       | Testing single functions/methods in isolation        | Runs on all pushes and PRs (excluding `slow`) |
| `integration`        | Integration tests for full workflows       | Testing component interactions and data flow         | Runs on PRs only (excluding `slow`)           |
| `performance`        | Performance and benchmarking tests         | Measuring execution time and resource usage          | Not run automatically in CI                   |
| `slow`               | Tests that take a long time to run         | Full training cycles, large datasets                 | **Excluded** from default CI runs             |
| `gpu`                | Tests that require GPU/CUDA                | GPU-accelerated operations, CUDA kernels             | Not run in CI (no GPU available)              |
| `multiprocessing`    | Tests that use multiprocessing             | Parallel candidate training, worker pools            | Runs with standard tests                      |
| `spiral`             | Spiral problem tests                       | Two-spiral classification validation                 | Runs with unit/integration tests              |
| `correlation`        | Correlation coefficient calculations       | Pearson correlation, covariance tests                | Runs with unit tests                          |
| `network_growth`     | Network growth algorithms                  | Hidden unit addition, architecture changes           | Runs with unit tests                          |
| `candidate_training` | Candidate unit training                    | Candidate weight updates, training loops             | Runs with unit tests                          |
| `validation`         | Input validation functions                 | Parameter checking, type validation                  | Runs with unit tests                          |
| `accuracy`           | Accuracy calculation methods               | Classification accuracy, metrics                     | Runs with unit tests                          |
| `early_stopping`     | Early stopping logic                       | Convergence detection, patience handling             | Runs with unit tests                          |
| `golden`             | Golden / snapshot regression (OUT-12)      | Trajectory, predict-after-load, API snapshot goldens | Dedicated `golden-regression.yml` only        |
| `conformance`        | model-core GrowableModel contract (OUT-13) | Interface shapes / keys / event order                | Dedicated `conformance.yml` only              |

**Opt-in flags (required):** `--golden` and `--conformance` are separate from
markers. Without the matching flag, `pytest_collection_modifyitems` skips those
tests even when `--slow` and `--integration` are set — so they never leak into
the unit, integration, or scheduled-slow lanes. Both WS-6 workflows also pass
`--slow --integration` because the tests carry those markers too.

### API Security and Admission Coverage

Operator-facing middleware, admission, and secrets contracts pinned in the API unit suite:

| Area | Source | Test pin |
|------|--------|----------|
| CR-024 body-limit cap and stream enforcement | `api.middleware.RequestBodyLimitMiddleware` | `tests/unit/api/test_api_middleware.py` — `TestRequestBodyLimitMiddleware` |
| REST auth-first / rate-limit keying | `api.middleware.SecurityMiddleware` + `api.security.RateLimiter` | `tests/unit/api/test_api_middleware.py` — `TestSecurityMiddlewareAuthRateLimitInterplay` |
| Always-on security headers + conditional HSTS | `api.middleware.SecurityHeadersMiddleware` | `tests/unit/api/test_api_middleware.py` — `TestSecurityHeadersMiddleware` |
| `ws_identity_key` blank/whitespace | `api.websocket.manager` | `tests/unit/api/test_ws_connection_caps.py` — `TestWsIdentityKey` |
| Origins parser fail-soft | `api.settings` | `tests/unit/api/test_api_settings.py` — `TestWsControlAllowedOriginsParser` |

Key scenarios:

- **Body limit:** an oversized declared `Content-Length` → early **413**; an invalid header → **400**; chunked/streaming bodies over the cap → **413** with an early abort (not full buffering); an under-declared `Content-Length` (`N <= max`, stream larger than max) → **413** (CR-024; the stream-read must not be gated on `content_length is None`); a truthful under-limit `Content-Length` caches the body on `request._body` for downstream handlers (BUG-CC-15).
- **Auth ↔ rate limit:** a missing/invalid `X-API-Key` returns 401 **before** `RateLimiter.check` (forged keys cannot burn budgets); distinct authenticated keys have independent fixed-window counters, while open auth (`api_keys=None`/`[]`) keys as `ip:…`; 429 responses preserve `Retry-After` and `X-RateLimit-*` after `SecurityMiddleware` rebuilds the `JSONResponse`; exempt paths (health/docs/`/metrics`) remain reachable after a saturated non-exempt client.
- **Identity key:** an empty / whitespace-only `X-API-Key` → `None` (anonymous); real keys hash to a 16-char digest.

```bash
cd src
PYTHONPATH=. python -m pytest tests/unit/api/test_api_middleware.py -v
PYTHONPATH=. python -m pytest \
  tests/unit/api/test_ws_connection_caps.py::TestWsIdentityKey \
  tests/unit/api/test_api_settings.py -k "WsControlAllowedOrigins" -v
```

Why this matters:

- Body-size enforcement is a DoS / memory-exhaustion control on every mutating HTTP request; a present-header-only gate looks correct in happy-path tests but silently reopens the under-declared bypass.
- HSTS only fires on `X-Forwarded-Proto: https` — a misconfigured TLS terminator silently drops it.
- Whitespace-only API-key headers are truthy strings; without the strip they would mint one shared per-identity digest and self-DoS under `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_PER_IDENTITY`.

### WebSocket `_numeric_setting` Defensive Reads

`/ws/training` and `/ws/control` load heartbeat (and control idle) timeouts via `_numeric_setting(obj, name, fallback)` before passing them into `asyncio.sleep` / `asyncio.wait_for`.

**Why it exists:** `unittest.mock.MagicMock` (and similar doubles) invent attribute stubs for any name. Feeding those stubs into asyncio timing APIs raises `TypeError` and kills the heartbeat/idle loops even though production `Settings` would have been fine.

**Contract under test:**

| Input | Result |
|-------|--------|
| A real `int` / `float` on the object | Returned unchanged |
| `obj is None`, a missing attribute, a string value, or a `MagicMock` stub | The hardcoded `fallback` |

Handler fallbacks: heartbeat interval `30`, pong timeout `10`, control idle → `Settings.ws_control_idle_timeout_sec` (default `120`). Source: `src/api/websocket/control_stream.py` and `training_stream.py`.

**Pitfall when writing API WS tests:**

- Prefer a real `Settings(...)` (or a `SimpleNamespace` with numeric fields) on `app.state.settings`.
- Do not rely on a bare `MagicMock()` for settings if the handler under test reaches the heartbeat/idle path — `_numeric_setting` will fall back, masking whether your override was applied.
- To assert the helper itself, call `control_stream._numeric_setting` / `training_stream._numeric_setting` directly (see `TestNumericSetting`).

```bash
cd src
python -m pytest \
  tests/unit/api/test_control_stream_coverage.py \
  tests/unit/api/test_training_stream_coverage.py \
  -k numeric_setting -v
```

### Inline / Reload Dataset Alignment Coverage

Request-boundary and staged-reload split alignment lives in:

- `src/tests/unit/api/test_inline_dataset_validation.py` — `InlineDataset` length / half-specified val → a model `ValueError` and HTTP `422`
- `src/tests/unit/api/test_lifecycle_manager_swap.py` — `_reload_dataset` rejects non-2-D trains, sample-count mismatches, and partial `X_test`/`y_test`

```bash
cd src && PYTHONPATH=. python -m pytest \
  tests/unit/api/test_inline_dataset_validation.py \
  tests/unit/api/test_lifecycle_manager_swap.py -k "reload or mismatch or 2d or partial" \
  -v
```

Operator contract: [POST `/v1/training/start`](../api/JUNIPER_CASCOR_API_REFERENCE.md#post-v1trainingstart) (`InlineDataset` alignment + staged reload notes).

### Worker ID Admission and Staged Dialect Coverage

| Area | Source | Test pin |
|------|--------|----------|
| Worker `register` ID regex + `TaskResultMessage.from_dict` | `api.workers.protocol` | `tests/unit/api/test_worker_protocol.py` — `TestValidateRegister`, `TestTaskResultMessageFromDict` |
| Canopy → juniper-data dialect (`moons`/`spirals`, zero clamps, strip) | `TrainingLifecycleManager._translate_staged_config` | `tests/unit/api/test_lifecycle_manager_swap.py` — `TestTranslateStagedConfig` |

```bash
cd src && python -m pytest \
  tests/unit/api/test_worker_protocol.py \
  tests/unit/api/test_lifecycle_manager_swap.py -v
```

Why this matters:

- Invalid worker IDs close with `4008` before the registry insert; the typed `task_result` parse rejects missing / out-of-bounds fields and JSON bools masquerading as ints.
- Without moons/spirals aliasing, every canopy-staged reload fails at juniper-data with an unknown-generator error.

### Early-Stopping Regression Coverage (No Validation Data Path)

The no-validation branch in `CascadeCorrelationNetwork.validate_training()` is covered by targeted unit regressions in `src/tests/unit/test_cascade_correlation_coverage_extended.py`.

Key scenarios covered:

- Improvement path: `train_loss` improves by more than `convergence_threshold`, so `best_value_loss` is updated and `patience_counter` resets to `0`.
- Patience exhaustion path: non-improving `train_loss` increments `patience_counter` until `patience` is exhausted and `early_stop` becomes `True`.
- Cross-iteration propagation: `grow_network()` feeds updated `patience_counter` and `best_value_loss` from one `validate_training()` call into the next iteration's `ValidateTrainingInputs`.

Run only these regression tests:

```bash
cd src
python -m pytest tests/unit/test_cascade_correlation_coverage_extended.py -k "validate_training_without_validation_data or propagates_validation_state" -v
```

Why this matters:

- The no-validation path is used when `x_val`/`y_val` are omitted, which is common in lightweight training runs.
- Regressions here can silently disable or destabilize early stopping behavior even if validation-based tests still pass.

### API Version Assertions (BUG-CC-04)

Canonical runtime version for the API process:

| Symbol / path | Source | Used by |
|---------------|--------|---------|
| `api.app._API_VERSION` | `importlib.metadata.version("juniper-cascor")` (fallback `"0.0.0-dev"`) | FastAPI `app.version`, Sentry, `set_build_info` |
| `api.routes.health._API_VERSION` | Same `importlib.metadata` read (separate module fallback literal) | `/v1/health`, readiness `version` field |
| `api.models.common._API_VERSION` | Module literal (may lag releases) | `ResponseEnvelope.meta.version` |

**Rule for wiring tests:** import `api.app._API_VERSION` and assert equality against it. Do not pin `"0.x.y"` in tests that validate app metadata, health/readiness version fields, or build-info calls.

```python
from api.app import _API_VERSION, create_app

assert create_app(...).version == _API_VERSION
# health: assert response.json()["version"] == _API_VERSION
```

**Allowed literals:** synthetic model construction in fixtures (for example
`ReadinessResponse(version="0.6.0", ...)`) and shape-only checks that only
require `isinstance(..., str)`.

**Companion check:** `TestBugCC04VersionSingleSource` in
`src/tests/unit/test_phase_2e_topology_correlation_phase.py` asserts
installed metadata equals `pyproject.toml` `version`.

Full narrative and pitfalls: [Testing Manual — API Version Assertions](MANUAL.md#api-version-assertions-bug-cc-04).

### Marker Combinations

```bash
# Run unit tests only
pytest -m "unit"

# Run unit tests excluding slow
pytest -m "unit and not slow"

# Run spiral-related unit tests
pytest -m "unit and spiral"

# Run correlation or accuracy tests
pytest -m "correlation or accuracy"

# Run everything except GPU and slow
pytest -m "not gpu and not slow"
```

---

## Report Locations

All test reports are generated in `src/tests/reports/`:

| Report Type      | Location                              | Description                              |
|------------------|---------------------------------------|------------------------------------------|
| HTML Coverage    | `reports/htmlcov/index.html`          | Interactive HTML coverage report         |
| XML Coverage     | `reports/coverage.xml`                | Cobertura-format XML for CI integration  |
| JUnit XML        | `reports/junit.xml`                   | JUnit-format test results for CI         |

### Accessing Reports

```bash
# Open HTML coverage report (Linux)
xdg-open src/tests/reports/htmlcov/index.html

# Open HTML coverage report (macOS)
open src/tests/reports/htmlcov/index.html

# View coverage summary in terminal
python -m coverage report
```

---

## Test Command Reference

### Direct pytest Commands

```bash
# Run all tests
cd src/tests && python -m pytest

# Run specific test file
python -m pytest unit/test_forward_pass.py -v

# Run tests matching name pattern
python -m pytest -k "test_accuracy" -v

# Run with specific markers
python -m pytest -m "unit and accuracy" -v

# Run with verbose output and short traceback
python -m pytest -v --tb=short

# Run with full traceback on failures
python -m pytest --tb=long

# Stop after first failure
python -m pytest -x

# Stop after N failures
python -m pytest --maxfail=3

# Run last failed tests only
python -m pytest --lf

# Run failed tests first, then rest
python -m pytest --ff

# Parallel execution (requires pytest-xdist)
python -m pytest -n auto

# Disable warnings
python -m pytest -p no:warnings
```

### run_tests.bash Script Options

```bash
cd src/tests/scripts

# Basic usage (unit tests with coverage)
bash run_tests.bash

# Script options
bash run_tests.bash -u              # Unit tests (default: true)
bash run_tests.bash -i              # Integration tests
bash run_tests.bash -p              # Performance tests
bash run_tests.bash -s              # Include slow tests
bash run_tests.bash -g              # Include GPU tests
bash run_tests.bash -v              # Verbose output
bash run_tests.bash -c              # With coverage (default: true)
bash run_tests.bash --no-coverage   # Disable coverage
bash run_tests.bash -j              # Parallel execution
bash run_tests.bash -f              # Re-run failed tests only
bash run_tests.bash -t FILE         # Run specific test file
bash run_tests.bash -m "MARKERS"    # Run with specific markers
bash run_tests.bash -o DIR          # Custom output directory
bash run_tests.bash -h              # Show help

# Combined options
bash run_tests.bash -v -c           # Verbose with coverage
bash run_tests.bash -i -s           # Integration and slow tests
bash run_tests.bash -u -j           # Unit tests in parallel
bash run_tests.bash -m "spiral"     # Spiral problem tests
```

### Coverage Commands

```bash
# Run the CI-parity aggregate gate from the repository root
bash util/run_coverage.bash

# Run with repository-wide coverage collection
python -m pytest -m "unit and not slow" src/tests/unit --cov=src

# Generate HTML report
python -m pytest -m "unit and not slow" src/tests/unit --cov=src --cov-report=html:src/tests/reports/htmlcov

# Generate XML report (for CI)
python -m pytest -m "unit and not slow" src/tests/unit --cov=src --cov-report=xml:src/tests/reports/coverage.xml

# Show missing lines in terminal
python -m pytest -m "unit and not slow" src/tests/unit --cov=src --cov-report=term-missing

# Multiple report formats
python -m pytest \
    -m "unit and not slow" \
    src/tests/unit \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html:src/tests/reports/htmlcov \
    --cov-report=xml:src/tests/reports/coverage.xml

# Check coverage threshold
python -m coverage report --fail-under=80

# View coverage for specific file
python -m coverage report --include="*/cascade_correlation.py"

# Produce per-file/sub-module gap-map input
python -m pytest -m "unit and not slow" src/tests/unit --cov=src --cov-report=json:src/tests/reports/coverage.json
juniper-coverage-gap-map --coverage-json src/tests/reports/coverage.json
```

---

## CI Test Matrix

### Trigger Events

| Event                      | Branches                               | Jobs Executed                           |
|----------------------------|----------------------------------------|----------------------------------------|
| Push                       | `main`, `develop`, `feature/**`, `fix/**` | lint, test, quality-gate, notify       |
| Pull Request               | `main`, `develop`                      | lint, test, **integration**, quality-gate, notify |

### Job Dependencies

```
lint ──────┐
           ├──> quality-gate ──> notify
test ──────┤
           │
           └──> integration (PR only)
```

### CI Marker Combinations

| Job             | Markers Used                      | Timeout | Notes                          |
|-----------------|-----------------------------------|---------|--------------------------------|
| Unit Tests      | `unit and not slow`               | 60s     | Runs on all pushes and PRs     |
| Integration     | `integration and not slow`        | 120s    | Runs on PRs only               |
| Slow Tests      | `slow`                            | 300s    | **Not run in CI** (manual)     |
| GPU Tests       | `gpu`                             | N/A     | **Not run in CI** (no GPU)     |

### CI Test Command (Unit Tests)

```bash
python -m pytest \
    -m "unit and not slow" \
    src/tests/unit \
    --verbose \
    --timeout=60 \
    --maxfail=5 \
    --junitxml=reports/junit/junit-unit.xml \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=xml:reports/coverage.xml \
    --cov-report=html:reports/htmlcov
```

### CI Test Command (Integration Tests)

```bash
python -m pytest integration/ \
    --verbose \
    --timeout=120 \
    -m "integration and not slow" \
    --maxfail=5
```

### Coverage Thresholds

| Threshold | Status | Action on Failure |
|-----------|--------|-------------------|
| 80% aggregate | Hard fail | `python -m coverage report --fail-under=${COVERAGE_FAIL_UNDER}` blocks the unit-tests job |
| >=90% per source file | Advisory until final gate PR | Measured with `juniper-coverage-gap-map` from coverage JSON |
| >=95% pooled per packaged sub-module | Advisory until final gate PR | Statement-weighted pooled coverage, not mean-of-files |

---

## Timeout Reference

### Default Timeouts

| Context              | Timeout  | Configuration Location            |
|----------------------|----------|-----------------------------------|
| Standard tests       | 60s      | `pyproject.toml` (`timeout = 60`) |
| CI unit tests        | 60s      | `.github/workflows/ci.yml`        |
| CI integration tests | 120s     | `.github/workflows/ci.yml`        |
| Slow tests           | 300s     | Per-test `@pytest.mark.timeout()` |

### Timeout Method

The project uses the `thread` timeout method (configured in
`pyproject.toml`). Thread-based timeout avoids SIGALRM interference
with forkserver IPC in multiprocessing-heavy performance tests:

```toml
timeout = 60
timeout_method = "thread"
```

### Overriding Timeouts

```bash
# Disable timeout for a single run
pytest --timeout=0

# Set custom timeout
pytest --timeout=120

# Timeout for slow tests (recommended approach)
pytest -m slow --timeout=300

# Disable global timeout when running slow tests
pytest -m slow --timeout=0
```

### Per-Test Timeout Decorator

```python
import pytest

# Set 300-second timeout for slow test
@pytest.mark.slow
@pytest.mark.timeout(300)
def test_full_training_cycle():
    ...

# Disable timeout for specific test
@pytest.mark.timeout(0)
def test_indefinite_operation():
    ...
```

---

## Exit Codes

### pytest Exit Codes

| Code | Name                  | Meaning                                              |
|------|-----------------------|------------------------------------------------------|
| 0    | `OK`                  | All tests passed                                     |
| 1    | `TESTS_FAILED`        | Some tests failed                                    |
| 2    | `INTERRUPTED`         | Test run interrupted by user (Ctrl+C)                |
| 3    | `INTERNAL_ERROR`      | Internal pytest error occurred                       |
| 4    | `USAGE_ERROR`         | pytest command line usage error                      |
| 5    | `NO_TESTS_COLLECTED`  | No tests were collected                              |

### Coverage Failure Conditions

| Condition                          | Exit Code | Trigger                                |
|------------------------------------|-----------|----------------------------------------|
| Coverage below threshold           | 2         | `--fail-under=N` with coverage < N%    |
| Missing coverage data              | 1         | `--cov` with no coverage collected     |
| Coverage file not found            | 1         | Specified source not found             |

### run_tests.bash Exit Codes

The script propagates pytest's exit code:

```bash
bash run_tests.bash
echo $?  # Shows pytest exit code
```

### Interpreting CI Results

| Job Result    | Meaning                                        | Action Required          |
|---------------|------------------------------------------------|--------------------------|
| ✓ Success     | All tests passed, quality gates met            | None                     |
| ✗ Failure     | Tests failed or quality gate failed            | Review failures          |
| ○ Skipped     | Job skipped (e.g., integration on push)        | None                     |
| ⊘ Cancelled   | Workflow cancelled                             | Re-run if needed         |

---

## Quick Reference Card

```bash
# Most common commands
cd src/tests/scripts
bash run_tests.bash              # Default: unit tests + coverage
bash run_tests.bash -v           # Verbose output
bash run_tests.bash -i           # Add integration tests
bash run_tests.bash -m "spiral"  # Run spiral tests only
bash run_tests.bash -f           # Re-run failed tests

# Direct pytest
cd src/tests
python -m pytest unit/ -v                    # Unit tests verbose
python -m pytest -m "unit and not slow" -v   # Fast unit tests
python -m pytest -k "accuracy" -v            # Tests matching "accuracy"
python -m pytest --lf                        # Last failed only

# Coverage from repository root
cd ../..
bash util/run_coverage.bash                 # CI-parity aggregate gate
python -m pytest src/tests/unit --cov=src --cov-report=term-missing
python -m coverage report --fail-under=80   # Current aggregate threshold
```

---

## Worker In-Flight Recovery and Teardown Regressions

Coverage for all four immediate-requeue paths, result integrity, round cancellation, receive-site protocol guards, and anomaly-history teardown lives in the API unit suite:

| Area | Files / focus |
|------|----------------|
| Result ownership + tensor validation | `src/tests/unit/api/test_worker_coordinator.py` (`TestSubmitResult::test_reject_wrong_worker_ownership`); `test_worker_protocol.py` (`TestValidateTensors`, incl. `test_empty_weights_returns_error`) |
| Result integrity — `success=True` requires weights | `test_worker_coordinator.py` — `test_reject_success_true_with_missing_weights`, `test_reject_success_true_with_empty_weights`, `test_accept_success_false_without_weights` |
| Requeue path 1 — schema / tensor reject | `test_worker_coordinator.py` — `_reject_and_requeue_task` arms under `TestSubmitResult` |
| Requeue path 2 — soft binary-frame abort | `test_worker_coordinator.py` / `test_worker_stream.py` — text / oversized / decode paths call `abort_in_flight_result` |
| Requeue path 3 — clean WebSocket disconnect | `test_worker_coordinator.py` — `TestHandleWorkerDisconnect` (requeue + deregister + send-callback drop under one lock) |
| Requeue path 4 — dispatch send failure | `test_worker_coordinator.py` — `TestRequeueAfterDispatchFailure`; `test_worker_stream.py` — `test_send_json_failure_requeues_assigned_task`, `test_send_bytes_failure_requeues_after_partial_send` |
| Round cancellation frees registry busy state | `test_worker_coordinator.py` — `TestCancelRound::test_cancel_frees_registry_active_task` |
| Non-object JSON + `tensor_manifest` guards | `test_worker_stream.py` / `test_worker_protocol.py` — object-only messages; manifest type + ≤ 32 entries; UTF-8 dtype `ValueError` |
| Anomaly history on deregister | `test_worker_security_integration.py` — `test_disconnect_clears_anomaly_history`, `test_disconnect_without_anomaly_detector_still_cleans_up` |

```bash
cd src
python -m pytest tests/unit/api/test_worker_coordinator.py tests/unit/api/test_worker_stream.py tests/unit/api/test_worker_protocol.py -v
python -m pytest tests/unit/api/test_worker_security_integration.py -k "anomaly" -v
```

Why this matters:

- Each requeue path has a **distinct** trigger and log line; conflating them hides which one regressed. A stall with a *still-heartbeating* worker narrows to the soft-abort or dispatch-send-failure path, because CONC-10 stale-worker reaping cannot fire while heartbeats continue.
- The success-without-weights guard must stay ahead of `validate_tensors`, or an empty / absent `tensor_manifest` skips it and an untrained `CandidateUnit` wins N-best selection with a claimed correlation.
- `cancel_round` must free the registry, not just the coordinator maps: once pending tracking is cleared, `_check_task_timeouts` can no longer reclaim a worker left marked busy.

Operator contracts: [JUNIPER_CASCOR_API_REFERENCE.md — WS `/ws/v1/workers`](../api/JUNIPER_CASCOR_API_REFERENCE.md#ws-wsv1workers).

---

## Related Documentation

- [Testing Quick Start](QUICK_START.md) - Getting started with testing
- [AGENTS.md](../../AGENTS.md) - Project conventions and commands
- [pyproject.toml](../../pyproject.toml) - pytest configuration (`[tool.pytest.ini_options]`)
- [CI Workflow](../../.github/workflows/ci.yml) - GitHub Actions configuration
