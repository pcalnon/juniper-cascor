# AGENTS.md - Juniper Cascor Project Guide

**Project**: Juniper Cascade Correlation Neural Network
**Repository**: pcalnon/juniper-cascor
**Author**: Paul Calnon
**License**: MIT License
**Version**: 0.10.0
**Last Updated**: 2026-09-03

---

## Hazards (resident — do not relocate)

Directives whose **non-application destroys work**. Everything else in this file may be demoted to
`docs/REFERENCE.md` under the memory budget; these may not, because a pointer only helps an agent
that already knows to look. Adding a new hazard here is legitimate — ratchet space out of a
reference section in the same PR rather than waiving the budget gate.

- **`CORSMiddleware` must stay OUTERMOST — it is added last for exactly that reason.** A browser
  preflight carries no `X-API-Key` (the browser generates it; author-defined headers ride only on
  the request that follows), so any ordering that puts `SecurityMiddleware` outside CORS answers
  every preflight to a non-exempt path with **401**, and no browser client on a configured origin
  can reach a protected endpoint. Running outermost also attaches CORS headers to 401/429 responses,
  so a browser surfaces the real status instead of an opaque CORS failure. Pinned by
  `tests/unit/api/test_api_app.py::TestCorsPreflight`. Full rationale: § Middleware Stack.
- **`RequestBodyLimitMiddleware` must always STREAM-READ with a cumulative byte cap.**
  `Content-Length` is an early-reject fast path only — for `POST`/`PUT`/`PATCH`, trusting it lets an
  **under-declared header bypass the limit entirely** (CR-024). Cap: `_PROJECT_API_MAX_REQUEST_BODY_BYTES`.
  Pinned by `tests/unit/api/test_api_middleware.py::TestRequestBodyLimitMiddleware`.
- **`/ws/*` paths skip the HTTP middleware stack.** WebSocket upgrades are **not** intercepted by
  `BaseHTTPMiddleware`, so the body-limit and security middlewares never run for them; WebSocket
  auth and message validation are the only controls there. A guard added to the HTTP stack does
  **not** cover the WebSocket surface, and nothing will tell you so.
- **`/tmp/` is prohibited** as the home for any script that produces, modifies or analyzes
  repository content — it is reaped when sessions/containers end and the scripts are irrecoverable.
  Scratch *data* there is fine; source files are not. Full rule and the motivating incident:
  § Script Placement.
- **`max_epochs` alone silently diverges the service from the CLI, and a non-positive
  `output_epochs` silently no-ops training.** `fit()` applies `max_epochs` to the **initial** output
  pass only; `grow_network`'s per-round passes read `self.output_epochs` directly (three sites —
  `cascade_correlation.py:4894`, `:5085`, `:5137`). That split is **intended**: do **not** "fix" it
  by forwarding `max_epochs`, which changes service behaviour and is golden-suite-visible (finding
  L-2, settled 2026-08-21). But `max_epochs` *is* in the service's `_FIT_KWARGS`
  (`src/api/lifecycle/manager.py:2094`), so a config that sets it without `output_epochs` runs the
  service at N-then-**10000** (the `output_epochs` default) while the direct CLI aliases the two —
  the arms are not like-for-like and the service is quietly better-trained and slower than the
  config appears to ask for. **Any CLI-vs-service comparison must set BOTH knobs, to the same
  value.** Separately (BUG-CC-09), a non-positive resolved `output_epochs` reaching
  `train_output_layer` makes `range(0)` never execute — weights stay wherever the previous iteration
  left them and the returned `final_loss` reflects an unchanged forward pass — which is why the
  value is re-validated *after* the fall-back at `cascade_correlation.py:1952`. juniper-ml#1159
  detects the split config and records a `validation_warnings` entry on the run manifest.
- **Never serialize a training counter the network does not carry — absence must look like
  absence.** `_save_metadata` writes `snapshot_counter` / `current_epoch` / `patience_counter` /
  `best_value_loss` only under `hasattr` (`snapshots/snapshot_serializer.py:290`). Until
  2026-08-23 these were `getattr(network, ..., 0)`, and the default is what made a whole defect
  class invisible: an attribute the model never assigns serializes as `0` / `inf`,
  **indistinguishable from a measured zero**. Read literally, the live archive then said all
  27,908 snapshots sat at epoch 0 having never trained — a reading that **nearly justified
  deleting 27,005 real models**, and came apart only on checking a network known to have grown to
  260 hidden units. A missing key is a question a reader can ask; a fabricated default is an
  answer they cannot check. `_restore_training_state_helper`, the index and the classification
  tools all tolerate missing keys, so re-adding a default buys nothing and destroys the evidence.
- **Never let failed candidate training look like a converged network — and `success_count`, not
  `failed_count`, is the test.** Three guards enforce this and each covers a case the others miss
  (BUG-CC-18 / ROBUST-01). **(a)** Both training paths raise → `CandidateTrainingError`
  (`cascade_correlation.py:2541`). **(b)** Empty result list after a non-exception return →
  `CandidateTrainingError` (`:2549`). **(c)** *Every candidate errored individually* — the observed
  case is a full GPU (`CandidateUnit.__init__` dying on `CUDA error: out of memory`). This one
  **slips past (a) and (b)**: the per-candidate handlers catch and **return**
  `CandidateTrainingResult(success=False, candidate=None)`, so a full, non-empty result list comes
  back with no exception, and the run reported `succeeded` / `no_candidate` / 1 hidden unit **while
  having trained nothing — silently corrupting downstream experiment campaigns**.
  `_raise_if_candidate_training_failed` (`:4425`) closes it by consulting `success_count`.
  **`failed_count` cannot be used for this test** — it is `len(results) - successful_candidates`,
  i.e. candidates that missed the *correlation threshold*, not candidates that errored.
  Related: `_get_dummy_results` (`:3057`) still exists and is still referenced at `:3005`, where it
  would fabricate zero-correlation candidates. That branch is unreachable **only because guard (b)
  raises first** — weaken (b) and the fabrication path is live again.

## Quick Reference

### Conda Environment

> **Required:** Activate the live `JuniperCascor1` conda environment before running any commands. The env name is **versioned** — rebuilds increment the suffix and rename the old env `*-DEPRECATED` (never activate those). Discover yours with `conda env list | grep JuniperCascor`.

```bash
conda activate JuniperCascor1   # live env; see the note above
```

### Essential Commands

```bash
# --- Server (primary operational mode) ---
cd src && python server.py                                        # Development server
uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8200  # Production server

# --- CLI Training (standalone mode) ---
cd src && python main.py                                          # Train on two-spiral problem

# --- Docker ---
docker build -t juniper-cascor:latest .                           # Build container
docker run -p 8200:8200 juniper-cascor:latest                     # Run container

# --- Testing ---
bash src/tests/scripts/run_tests.bash                # Run all tests
cd src && python -m pytest tests/ -v                 # Verbose test output
cd src && python -m pytest tests/unit/ -v            # Unit tests only
cd src && python -m pytest tests/unit/api/ -v        # API unit tests
cd src && python -m pytest tests/integration/ -v     # Integration tests
cd src && python -m pytest tests/performance/ -v     # Performance benchmarks
cd src && python -m pytest tests/ -m "unit" -v       # By marker
cd src && python -m pytest tests/ -k "spiral" -v     # By keyword
cd src && python -m pytest tests/ --run-long         # Include long-running tests
cd src && python -m pytest tests/ --cov=. --cov-report=html:tests/reports/htmlcov  # Coverage

# --- Performance Benchmarks ---
bash src/tests/scripts/run_benchmarks.bash           # Run performance benchmarks

# --- Profiling ---
cd src && python -m cProfile -o profile.out main.py  # cProfile deterministic profiler
cd src && python -m profiling.memory                 # Memory profiling
bash util/profile_training.bash                      # py-spy sampling profiler

# --- Type Checking & Linting ---
cd src && python -m mypy .                           # Type checking
cd src && python -m flake8 .                         # Linting
cd src && python -m black . --check                  # Format check
cd src && python -m isort . --check-only             # Import sort check
cd src && python -m bandit -r . -c ../pyproject.toml # Security scan

# --- Pre-commit ---
pre-commit run --all-files                           # Run all hooks
pre-commit install                                   # Install hooks
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| **Server Configuration** | | |
| `JUNIPER_CASCOR_HOST` | API listen address | `127.0.0.1` |
| `JUNIPER_CASCOR_PORT` | API listen port | `8200` |
| `JUNIPER_CASCOR_LOG_LEVEL` | Log level (TRACE, DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `JUNIPER_CASCOR_LOG_FORMAT` | Log output format (`text` or `json`) | `text` |
| `JUNIPER_CASCOR_CORS_ORIGINS` | CORS allowed origins (JSON list) | `[]` (none) |
| **Authentication & Security** | | |
| `JUNIPER_CASCOR_API_KEYS` | Comma-separated API keys for authentication | `None` (auth disabled) |
| `JUNIPER_CASCOR_API_KEYS_FILE` | Docker-secrets path for API keys; an existing empty/whitespace file yields open auth in the compose `_FILE`-only pattern (`get_secret()` does not fall back to the plain env var) | `None` |
| `JUNIPER_CASCOR_REQUIRE_AUTH` | SEC-F01 intended auth posture: `false` = WARN and run open when keys missing/blank; `true` = refuse boot with `AuthPostureError`. Set `true` wherever secrets are provisioned (composed juniper-deploy). Bypass with `JUNIPER_SKIP_AUTH_POSTURE_CHECK=1` (logged loudly). | `false` |
| `JUNIPER_CASCOR_RATE_LIMIT_ENABLED` | Enable rate limiting | `false` |
| `JUNIPER_CASCOR_RATE_LIMIT_REQUESTS_PER_MINUTE` | Fixed-window budget; keyed `key:<api_key>` when REST auth succeeds, `ip:<client>` when auth is disabled | `60` |
| **WebSocket** | | |
| `JUNIPER_CASCOR_WS_MAX_CONNECTIONS` | Maximum WebSocket connections | `50` |
| `JUNIPER_WS_HEARTBEAT_INTERVAL_SEC` | WebSocket heartbeat interval (`Settings` `AliasChoices` name -- **not** `JUNIPER_CASCOR_`-prefixed) | `30` |
| `JUNIPER_WS_HEARTBEAT_PONG_TIMEOUT_SEC` | WebSocket heartbeat pong/liveness window (same `AliasChoices` binding) | `10` |
| **Observability** | | |
| `JUNIPER_CASCOR_SENTRY_DSN` | Sentry DSN for error tracking | `None` (disabled) |
| `JUNIPER_CASCOR_METRICS_ENABLED` | Enable Prometheus metrics | `false` |
| `JUNIPER_CASCOR_EVAL_METRICS_ENABLED` | Compute C7 scalar evaluation metrics (F1/precision/recall/ROC-AUC) per training step over the eval split; surfaced on `/v1/metrics`, `/v1/metrics/history`, and the WS `metrics` frames. Distinct from `JUNIPER_CASCOR_METRICS_ENABLED` (Prometheus). Set `0`/`false` to disable. | `true` |
| **Auto-Start** | | |
| `JUNIPER_CASCOR_AUTO_START` | Auto-start training on server startup | `true` |
| `JUNIPER_CASCOR_AUTO_DATASET` | Default dataset generator | `spiral` |
| `JUNIPER_CASCOR_AUTO_DATASET_PARAMS` | Dataset generator parameters (JSON) | `{}` |
| `JUNIPER_CASCOR_AUTO_NETWORK` | Network creation parameters (JSON) | `{}` |
| `JUNIPER_CASCOR_AUTO_TRAIN_EPOCHS` | Auto-start training max epochs | `200` |
| `JUNIPER_CASCOR_AUTO_START_DATA_SERVICE` | Auto-start juniper-data companion | `false` |
| `JUNIPER_CASCOR_AUTO_START_DATA_SERVICE_COMMAND` | Command to launch juniper-data service | `python -m juniper_data` |
| `JUNIPER_CASCOR_AUTO_START_CANOPY` | Auto-start juniper-canopy companion | `false` |
| `JUNIPER_CASCOR_AUTO_START_CANOPY_COMMAND` | Command to launch juniper-canopy service | `python -m juniper_canopy` |
| **Remote Workers** | | |
| `JUNIPER_CASCOR_REMOTE_WORKERS_HEARTBEAT_TIMEOUT` | Worker heartbeat stale timeout, seconds (CONC-10 reap) | `30.0` |
| `JUNIPER_CASCOR_REMOTE_WORKERS_TASK_REASSIGNMENT_TIMEOUT` | Fallback reassignment for orphaned in-flight tasks, seconds. All four immediate-requeue paths (reject, soft abort, clean disconnect, dispatch send failure) bypass this timeout -- reaching it means none of them fired. | `120.0` |
| **Legacy / Integration** | | |
| `CASCOR_LOG_LEVEL` | Override log level at runtime | `INFO` |
| `JUNIPER_DATA_URL` | JuniperData service URL | `http://localhost:8100` |
| `JUNIPER_DATA_API_KEY` | API key for JuniperData authentication | (none) |

### Key Entry Points

| File | Purpose |
|------|---------|
| `src/server.py` | **FastAPI server entry point** (primary production mode) |
| `src/main.py` | CLI entry point for standalone two-spiral training |
| `src/api/app.py` | FastAPI application factory with lifespan management |
| `src/api/settings.py` | Pydantic-based application configuration |
| `src/cascade_correlation/cascade_correlation.py` | Core neural network implementation |
| `src/candidate_unit/candidate_unit.py` | Candidate unit for network growth |
| `src/spiral_problem/spiral_problem.py` | Two-spiral problem solver |
| `src/parallelism/task_distributor.py` | Distributed task scheduling (local + remote workers) |
| `src/profiling/` | Profiling infrastructure (memory, deterministic) |
| `src/tests/scripts/run_tests.bash` | Test runner script |
| `src/tests/conftest.py` | Test configuration and fixtures |
| `util/profile_training.bash` | py-spy sampling profiler |

---

## Project Overview

Juniper Cascor is an AI/ML research platform implementing the **Cascade Correlation Neural Network** algorithm from foundational research (Fahlman & Lebiere, 1990). The project provides both a **REST/WebSocket service** for integrated deployment and a **standalone CLI** for direct experimentation.

**Research Philosophy**:

- Transparency over convenience: Algorithms implemented from first principles
- Understanding over abstraction: Full visibility into network behavior
- Modularity and scalability: Designed for research flexibility

**Operational Modes**:

| Mode | Entry Point | Purpose |
|------|-------------|---------|
| **Service** | `src/server.py` | FastAPI REST/WebSocket server for production and integration use |
| **CLI** | `src/main.py` | Standalone training on the two-spiral problem |

---

## Directory Structure

The annotated source tree, with the purpose of every package and key module. Moved to [`docs/REFERENCE.md` § Directory Structure Reference](docs/REFERENCE.md#directory-structure-reference) — read it when working on this area.

## REST API

Every REST route, its auth posture, and its request/response shape. Moved to [`docs/REFERENCE.md` § REST API Reference](docs/REFERENCE.md#rest-api-reference) — read it when working on this area.

## WebSocket Protocol

The three WebSocket channels, their message envelopes, and the admission rules. Moved to [`docs/REFERENCE.md` § WebSocket Protocol Reference](docs/REFERENCE.md#websocket-protocol-reference) — read it when working on this area.

## Training Lifecycle Management

The lifecycle system coordinates network training through deterministic state transitions.

| Component | Module | Purpose |
|-----------|--------|---------|
| `TrainingLifecycleManager` | `api.lifecycle.manager` | Central orchestrator (thread-safe via locks) |
| `TrainingStateMachine` | `api.lifecycle.state_machine` | Deterministic state transitions (idle, training, paused, etc.) |
| `TrainingMonitor` | `api.lifecycle.monitor` | Callback-based metrics collection |

**Training Events** (emitted via callbacks):

- `training_start` / `training_end`
- `epoch_end`
- `cascade_add`
- `candidate_progress`

The lifecycle manager wraps the network's training methods non-intrusively to emit events without modifying the core algorithm.

**Staged dataset dialect** -- Canopy stages `dataset_type` values in its own dialect (`spirals` / `moons` / `xor`). `TrainingLifecycleManager._translate_staged_config` aliases them to juniper-data generator keys (`spiral` / `moon`) only at the `_reload_dataset` / live-swap fetch boundary, remaps sample counts (`n_points_per_spiral`, `n_points_per_quadrant`), clamps zero divisors, and strips spiral-only fields for non-spiral generators. Stored configs keep the canopy names.

**C2b progress pairs** -- `output_epoch`/`output_total_epochs` and `candidate_epoch`/`candidate_total_epochs` are zeroed at run start (`_run_training`) and at growth-phase exit (the `training_end` handler after a grow), so UI bars never keep the previous pass's terminal values across those boundaries.

---

## Remote Worker System

Worker registry/coordinator components, the connect-to-deregister lifecycle, and the task-result wire contract. Moved to [`docs/REFERENCE.md` § Remote Worker System Reference](docs/REFERENCE.md#remote-worker-system-reference) — read it when working on this area.

## Middleware Stack

Registered in `src/api/app.py` via successive `app.add_middleware(...)` calls. Starlette/FastAPI middleware runs **LIFO** (last added = first executed), so the outer-to-inner request order when all layers are enabled is:

| Order (outer → inner) | Middleware | Module | Purpose |
|-----------------------|-----------|--------|---------|
| 1 | `CORSMiddleware` | FastAPI/Starlette | CORS headers + preflight short-circuit (only if origins are configured) |
| 2 | `RequestIdMiddleware` | `api.observability` | X-Request-ID propagation |
| 3 | `PrometheusMiddleware` | `api.observability` | Metrics (only when `metrics_enabled`) |
| 4 | `SecurityMiddleware` | `api.middleware` | API key auth + rate limiting (exempt paths) |
| 5 | `SecurityHeadersMiddleware` | `api.middleware` | CSP, HSTS, X-Frame-Options, etc. |
| 6 | `RequestBodyLimitMiddleware` | `api.middleware` | 10 MiB request body limit (CR-024 stream cap) |

**`CORSMiddleware` must stay outermost — it is added last for exactly that reason.**
A browser preflight carries no `X-API-Key`: the browser generates the preflight itself, and
author-defined headers ride only on the actual request that follows. So any ordering that puts
`SecurityMiddleware` outside CORS answers every preflight to a non-exempt path with 401, and no
browser client on a configured origin can reach a protected endpoint. Running outermost also
attaches the CORS headers to error responses (401/429), so a browser surfaces the real status
instead of an opaque CORS failure.

This is a stronger guarantee than an `OPTIONS` bypass in `SecurityMiddleware._is_exempt` would
give: CORS short-circuits only a *genuine* preflight (one carrying
`Access-Control-Request-Method`), so a plain `OPTIONS` request still authenticates. Pinned by
`tests/unit/api/test_api_app.py::TestCorsPreflight`.

WebSocket upgrade requests are **not** intercepted by `BaseHTTPMiddleware`, so `/ws/*` paths skip the body-limit and security-middleware HTTP paths entirely; they use WebSocket auth and message validation instead.

`RequestBodyLimitMiddleware` uses `Content-Length` only as an early-reject fast path. For `POST`/`PUT`/`PATCH` it must always stream-read with a cumulative byte cap so under-declared headers cannot bypass the limit (CR-024). Cap constant: `_PROJECT_API_MAX_REQUEST_BODY_BYTES`. Tests: `tests/unit/api/test_api_middleware.py::TestRequestBodyLimitMiddleware`.

---

## Security

### API Key Authentication

- Header: `X-API-Key`
- Comparison: HMAC-based (timing-safe)
- When `JUNIPER_CASCOR_API_KEYS` is unset/blank, authentication is disabled
- Docker secrets support: `JUNIPER_CASCOR_API_KEYS_FILE` for container deployments — `get_secret()` returns an existing file's stripped contents with no env fallback; an empty file leaves keys unset in the compose `_FILE`-only pattern

### Boot-Time Auth Posture (SEC-F01)

- Lifespan calls `juniper_service_core.enforce_auth_posture` after the bind guard, before serving (`src/api/app.py`)
- `JUNIPER_CASCOR_REQUIRE_AUTH=false` (default): missing/blank keys → loud WARNING, service continues (bare/dev)
- `JUNIPER_CASCOR_REQUIRE_AUTH=true`: missing/blank keys → CRITICAL + `AuthPostureError` (fail-closed for provisioned stacks)
- Escape hatch: `JUNIPER_SKIP_AUTH_POSTURE_CHECK=1` (logged loudly)

### Rate Limiting

- Fixed-window (thread-safe); optional via `JUNIPER_CASCOR_RATE_LIMIT_ENABLED` (default off)
- Default: 60 requests/minute when enabled (`JUNIPER_CASCOR_RATE_LIMIT_REQUESTS_PER_MINUTE`)
- Keying: `key:<api_key>` when REST API-key auth succeeds; `ip:<client>` when auth is disabled (`API_KEYS` unset / `[]`)
- Ordering: `SecurityMiddleware` authenticates **before** rate limiting -- a 401 never burns a budget slot
- 429 responses carry `Retry-After` plus `X-RateLimit-Limit` / `-Remaining` / `-Reset`, preserved through the middleware's `JSONResponse` rebuild
- Exempt paths (`EXEMPT_PATHS`): `/v1/health`, `/v1/health/live`, `/v1/health/ready`, `/docs`, `/openapi.json`, `/redoc`, `/metrics`, `/metrics/`

### Request Body Limits (CR-024)

- Default cap: 10 MiB (`_PROJECT_API_MAX_REQUEST_BODY_BYTES`)
- Mutating methods only (`POST` / `PUT` / `PATCH`)
- Oversized declared `Content-Length` → HTTP 413; invalid header → HTTP 400
- Stream path always enforces the cumulative cap (do **not** gate on absent `Content-Length` -- that reopens the under-declared bypass)
- Under-limit bodies are cached on `request._body` for downstream handlers (BUG-CC-15)
- WebSocket upgrades are not covered by this middleware (`BaseHTTPMiddleware` skip)

### Security Headers

`SecurityHeadersMiddleware` (`api.middleware`) injects on every HTTP response, including health probes:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: camera=(), microphone=(), geolocation=()`
- `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'` (constructor override supported)

HSTS (`Strict-Transport-Security: max-age=31536000; includeSubDomains`) is added **only** when the request carries `X-Forwarded-Proto: https`. A TLS terminator that does not forward that header will silently omit HSTS on an otherwise-HTTPS public URL.

### TLS

Certificate generation script: `scripts/tls/generate_certs.bash`

---

## Observability

### Logging

- **Text format** (development): Colored console output with custom formatters
- **JSON format** (production): Structured JSON logging for log aggregation
- **Request IDs**: Propagated via `ContextVar` and `X-Request-ID` header
- **Custom levels**: TRACE, VERBOSE, DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL

### Prometheus Metrics

When `JUNIPER_CASCOR_METRICS_ENABLED=true`:

- `http_requests_total` -- Request counter by method, path, status
- `http_request_duration_seconds` -- Request latency histogram
- Build info labels
- Endpoint: `GET /v1/metrics`

### Sentry Integration

When `JUNIPER_CASCOR_SENTRY_DSN` is set:

- Automatic error reporting
- Release tracking with API version

### Health Probes

- **Liveness** (`/v1/health/live`): Process is running
- **Readiness** (`/v1/health/ready`): Dependencies available, training system initialized

---

## Core Components

The cascade-correlation network, candidate units, and the configuration objects. Moved to [`docs/REFERENCE.md` § Core Components Reference](docs/REFERENCE.md#core-components-reference) — read it when working on this area.

## Parallelism and Distribution

### TaskDistributor

Central task scheduler (`src/parallelism/task_distributor.py`):

- **Local-first scheduling**: Assigns tasks to local multiprocessing workers as primary capacity
- **Remote overflow**: Routes excess tasks to registered WebSocket workers
- **Fallback**: Reverts to local execution if remote workers fail or timeout

### Local Multiprocessing

- `multiprocessing.Pool` with forkserver context (safer for CUDA, avoids GIL clone issues)
- Queue-based task distribution via custom manager
- Shared memory support (numpy arrays via `multiprocessing.shared_memory`)
- BLAS thread limits set before imports to prevent contention with workers

### Pickling

Classes implement `__getstate__` and `__setstate__` to handle non-picklable objects (loggers, closures). `ActivationWithDerivative` stores function type by name and reconstructs on unpickling.

---

## Serialization System

### HDF5 Snapshots

Save/load network state including architecture, weights, activation functions, training history, random state, UUID, and checksums.

**CLI Tools**:

```bash
cd src
python -m snapshots.snapshot_cli save network.pkl snapshot.h5
python -m snapshots.snapshot_cli load snapshot.h5
python -m snapshots.snapshot_cli verify snapshot.h5
python -m snapshots.snapshot_cli list ./snapshots/
```

---

## Programming Conventions

**Naming**:

- Constants: Uppercase with underscores, component-prefixed (e.g., `_CASCADE_CORRELATION_NETWORK_*`)
- Classes: PascalCase
- Methods/Functions: snake_case with private methods prefixed by underscore
- Constructor Parameters: Name-mangled style

**File Headers**: Standardized headers with Project, Sub-Project, Application, Author, Version, File Path, Dates, License, Description, References.

**Imports**: Standard library, Third-party, Local application (enforced by isort with `profile = "black"`).

**Type Hints**: Extensive use throughout (e.g., `def forward(self, x: torch.Tensor = None) -> torch.Tensor:`).

**Logging**: Custom system with levels: TRACE, VERBOSE, DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL.

**Documentation**: Structured docstrings with Description, Args, Returns, Raises, Notes sections.

**Line Length**: 512 for all linters (black, isort, flake8).

**Prometheus Collectors**: Use the canonical helpers from `juniper-observability` (`>=0.2.0`) for any new `Counter` / `Gauge` / `Histogram` / `Summary` / `Info` / `Enum`:

- `register_or_reuse(factory, name, *args, **kwargs)` — adopt-existing on duplicate (the default for almost every call site; preserves accumulated samples across in-process re-init).
- `register_fresh(...)` — drop-and-recreate (only when args genuinely differ; the legacy local `_register_or_reuse` shape).
- `register_info_or_update(name, description, **labels)` — sugar for the `Info` two-step register-then-`.info({...})` pattern.
- `lazy_register_or_reuse(...)` — for the lazy-init-with-`None`-sentinel pattern.

Tests touching these collectors should use `juniper_observability.testing.reset_prometheus_registry`. Existing examples: `src/api/observability.py:_ensure_training_metrics` / `_ensure_ws_metrics` (22+ call sites via the `_register_or_reuse` alias) and `src/api/websocket/control_stream.py:_get_command_counter`. See [the design doc in juniper-ml](https://github.com/pcalnon/juniper-ml/blob/main/notes/observability/JUNIPER_2026-05-05_JUNIPER-ML_REGISTER-OR-REUSE-HELPER-DESIGN.md) for the rationale.

---

## Testing Infrastructure

**Framework**: pytest
**Location**: `src/tests/`

### Test Categories (Markers)

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests for individual components |
| `integration` | Integration tests for full workflows |
| `performance` | Performance and benchmarking tests |
| `slow` | Long-running tests |
| `long` | Long-running correctness tests (use `--run-long`) |
| `gpu` | GPU/CUDA tests |
| `multiprocessing` | Multiprocessing tests |
| `spiral` | Spiral problem tests |
| `correlation` | Correlation calculation tests |
| `network_growth` | Network growth algorithm tests |
| `candidate_training` | Candidate unit training tests |
| `validation` | Input validation tests |
| `accuracy` | Accuracy calculation tests |
| `early_stopping` | Early stopping logic tests |
| `golden` | Golden / snapshot regression (OUT-12; needs `--golden`, serial WS-6 lane) |
| `conformance` | model-core GrowableModel conformance (OUT-13; needs `--conformance`, serial WS-6 lane) |
| `requires_juniper_data` | Tests requiring juniper-data package |

### Test Directory Structure

```text
src/tests/
├── conftest.py                  # Global fixtures
├── pytest.ini                   # pytest configuration
├── scripts/
│   ├── run_tests.bash           # Test runner
│   └── run_benchmarks.bash      # Benchmark runner
├── conformance/                 # WS-6 OUT-13 model-core conformance suite
├── fixtures/golden/             # WS-6 OUT-12 checked-in golden artifacts
├── helpers/
│   ├── assertions.py            # Custom assertion helpers
│   └── utilities.py             # Test utility functions
├── mocks/
│   └── mock_candidate.py        # Mock objects
├── unit/                        # Unit tests (70+ files)
│   ├── test_cascade_correlation_*.py   # CasCor network tests
│   ├── test_candidate_unit_*.py        # Candidate unit tests
│   ├── test_spiral_*.py                # Spiral problem tests
│   ├── test_snapshot_*.py              # Serialization tests
│   ├── test_log_*.py                   # Logging tests
│   ├── test_server_coverage.py         # Server entry point tests
│   ├── test_main_*.py                  # CLI entry point tests
│   └── api/                            # API layer tests (30+ files)
│       ├── test_api_app.py             #   App factory tests
│       ├── test_api_health.py          #   Health endpoint tests
│       ├── test_api_middleware.py       #   Middleware tests
│       ├── test_api_security.py        #   Security tests
│       ├── test_api_settings.py        #   Settings tests
│       ├── test_lifecycle_*.py         #   Lifecycle tests
│       ├── test_websocket_*.py         #   WebSocket tests
│       ├── test_worker_*.py            #   Worker system tests
│       ├── test_*_route*.py            #   Route handler tests
│       └── test_monitoring_hooks.py    #   Monitoring tests
├── integration/                 # Integration tests
│   ├── test_serialization.py
│   ├── test_spiral_problem.py
│   ├── test_comprehensive_serialization.py
│   ├── test_juniper_data_e2e.py
│   └── api/                     # API integration tests
│       ├── conftest.py
│       ├── test_api_full_lifecycle.py
│       └── test_websocket_streaming.py
└── performance/                 # Performance benchmarks
    ├── conftest.py
    ├── test_baselines.py
    ├── test_concurrency_scaling.py
    ├── test_endtoend_profiling.py
    ├── test_micro_autograd.py
    ├── test_micro_candidate.py
    ├── test_micro_correlation.py
    ├── test_micro_forward_pass.py
    ├── test_micro_output_training.py
    └── test_shared_memory.py
```

### Test Output

- HTML Coverage: `src/tests/reports/htmlcov/index.html`
- XML Coverage: `src/tests/reports/coverage.xml`
- JUnit XML: `src/tests/reports/junit.xml`

### Coverage

Reproduce the CI coverage gate locally (full suite):

```bash
make coverage                 # convenience wrapper
bash util/run_coverage.bash   # source of truth (mirrors .github/workflows/ci.yml)
```

Gate: 80% aggregate (override with `COVERAGE_FAIL_UNDER=<n>`). Coverage runs in parallel mode with a custom data_file (see pyproject `[tool.coverage.run]`); the script reproduces the CI sequence exactly. Full suite by design; for a narrower run use plain `pytest`.

---

## CI/CD Pipelines

### GitHub Actions Workflows

| Workflow | File | Triggers | Purpose |
|----------|------|----------|---------|
| CI/CD Pipeline | `.github/workflows/ci.yml` | Push (main, develop, feature/**, fix/**), PR, dispatch | Pre-commit, unit tests, integration tests, security scanning |
| Golden Regression (WS-6) | `.github/workflows/golden-regression.yml` | Push `main`, PR `main`/`develop`, dispatch | Serial OUT-12 golden / snapshot regression (Python 3.13 + torch 2.11.0) |
| Conformance (WS-6) | `.github/workflows/conformance.yml` | Push `main`, PR `main`/`develop`, dispatch | Serial OUT-13 model-core GrowableModel conformance |
| CI — protocol | `.github/workflows/ci-protocol.yml` | Path-filtered on `juniper-cascor-protocol/**`, dispatch | Package tests + build/`twine check` |
| CI — cascor-model | `.github/workflows/ci-cascor-model.yml` | Path-filtered on `juniper-cascor-model/**`, dispatch | Package tests (incl. drift-guard) + build/`twine check` |
| Scheduled Long Tests | `.github/workflows/scheduled-tests.yml` | Cron schedule (nightly), dispatch | Slow and long-running correctness tests |
| Publish | `.github/workflows/publish.yml` | Release (`v*`) | PyPI publish for `juniper-cascor` (TestPyPI → verify → PyPI) |
| Publish protocol | `.github/workflows/publish-protocol.yml` | Release (`juniper-cascor-protocol-v*`) + `workflow_dispatch` | PyPI publish for `juniper-cascor-protocol` |
| Publish model | `.github/workflows/publish-cascor-model.yml` | Release (`juniper-cascor-model-v*`) + `workflow_dispatch` | PyPI publish for `juniper-cascor-model` |
| Lockfile Update | `.github/workflows/lockfile-update.yml` | Push to dependabot/** branches | Dependency lockfile refresh |
| CodeQL Analysis | `.github/workflows/codeql.yml` | Push `main`/`develop`, PR `main`, weekly Monday 06:00 UTC | Python CodeQL SAST (`+security-and-quality`; soak, not a required check) |
| Security Scan | `.github/workflows/security-scan.yml` | Schedule/dispatch | Bandit + pip-audit `--strict` (no CodeQL, no Gitleaks) |
| Sequence Safety (Advisory) | `.github/workflows/sequence-safety.yml` | PR (`main`/`develop`) | Per-PR symbol-loss + docs-deletion screens over base..HEAD (ADVISORY, standalone, never a required check) |
| Post-Merge Main Verification | `.github/workflows/main-verify.yml` | Push `main`, dispatch | Bypass-proof post-merge compositional-loss net (catch-up base; stable-title tracking issue on failure) |

### Lockfile Update PAT Gate

`lockfile-update.yml` checks out and pushes with `CROSS_REPO_DISPATCH_TOKEN` so the lock commit re-triggers CI. Dependabot runs use the Dependabot secret store — a PAT registered only under Actions secrets is empty there.

| Condition | Behavior |
|-----------|----------|
| PAT present | Auto-regen + `[dependabot skip]` push |
| PAT absent + `dependabot[bot]` | Green no-op (`::notice::`); **Lockfile Freshness** in `ci.yml` still blocks stale locks |
| PAT absent + other actor | Hard fail (secret misconfiguration) |

Optional: register the same PAT under **Settings → Secrets → Dependabot** to restore Dependabot auto-regen. Operator narrative: [`notes/DEPENDENCY_UPDATE_WORKFLOW.md`](notes/DEPENDENCY_UPDATE_WORKFLOW.md).

### CodeQL Analysis

`.github/workflows/codeql.yml` runs GitHub CodeQL on Python (`queries: +security-and-quality`).
Push `main`/`develop`, PR **`main` only**, weekly Monday 06:00 UTC (same cron as `security-scan.yml`).
No `workflow_dispatch`. Soak / not a required check — findings go to **Security → Code scanning**, not the Quality Gate.
Dependabot groups `github/codeql-action*` so `init` / `autobuild` / `analyze` and `ci.yml`'s Bandit `upload-sarif` bump together.
Bandit SARIF upload is `continue-on-error: true`; the blocking Bandit step is the separate medium+ CLI invocation.
Details: [`docs/ci_cd/MANUAL.md`](docs/ci_cd/MANUAL.md#codeql-analysis) / [`docs/ci_cd/REFERENCE.md`](docs/ci_cd/REFERENCE.md#codeql-analysis).

### CI Pipeline Jobs (ci.yml)

- Pre-commit hooks (black, isort, flake8, mypy, bandit)
- Unit tests with coverage enforcement
- Integration tests
- Security scanning (Gitleaks, Bandit SARIF, pip-audit)
- Lockfile Freshness (`lockfile-check`) — required quality-gate input
- Dependency caching for performance
- Concurrency: one pipeline per branch, cancel-in-progress

### Additional GitHub Configuration

- `.github/CODEOWNERS` -- Code ownership rules
- `.github/dependabot.yml` -- Automated dependency updates (`codeql-action` group covers `github/codeql-action*`)

### Sequence Safety (Compositional-Loss Net)

Ported from juniper-ml (ml#873 / #880 / #928; the flood-remediation program) after the 2026-08-05 storm triage found *compositional loss* — a silently deleted def/class/method, a gutted body, or a net docs-section deletion that no per-PR check sees because a deleted test cannot fail — to be cascor's one remaining gap.

The two pure-stdlib git-diff screens are now **consumed from the published `juniper-ci-tools` package** (pinned `>=0.8.0,<0.9.0`) via two console scripts; the inline `util/sequence_safety/` copy that cascor#482 first ported was deleted in the Wave-3 retrofit (ml canonical, consumers consume — the same shape as the doc-tools / dep-docs migrations; see juniper-ml `notes/JUNIPER_2026-08-07_JUNIPER-ECOSYSTEM_SEQUENCE-SAFETY-ROLLOUT-PLAN.md`).

- `juniper-symbol-loss-check --scope 'src/**/*.py'` — AST symbol inventory of BASE vs HEAD over `src/**/*.py` (includes `src/tests/**`); FAIL on a LOST / WEAKENED / DUPLICATED def, with a qualified-name / body-similarity relocation downgrade. Escape hatch: an `Allow-Symbol-Loss: <qualified.symbol>` commit trailer.
  cascor passes `--scope 'src/**/*.py'` explicitly because the package's built-in default scope is juniper-ml's (`tests/*.py` + `util/**`); the `@property`/`@x.setter` accessor-pair disambiguation (once a cascor-local adaptation) is now upstreamed into the package.
- `juniper-docs-additions-check` — markdown deletion-magnitude screen over the package's universal default docs cluster (`AGENTS.md` + `docs/**` + `notes/**`, so no `--scope` is needed); FAIL on a deleted heading or a run of ≥ N consecutive deleted lines, WARN on small in-place swaps. Escape hatch: an `Allow-Docs-Rewrite: <path>` commit trailer.

Everything is **ADVISORY** — neither workflow is a required status check and this makes **no branch-ruleset change**.
`sequence-safety.yml` surfaces findings per-PR at review (with WARN-only `allow-symbol-loss` / `docs-rewrite` label hatches); `main-verify.yml` is the bypass-proof post-merge net that fires on every merge to `main` (catch-up base sweeps any `[skip ci]` window; a stable-title tracking issue is upserted on failure). Both `pip install "juniper-ci-tools>=0.8.0,<0.9.0"` then invoke the console scripts.
v1 defers the post-merge regression battery (cascor's suite is heavy) and Slack notify (no webhook secret) — see the workflow header comments.
The screens' canonical regression suite lives in the `juniper-ci-tools` package; `src/tests/unit/test_sequence_safety_retired.py` is cascor's local guard that the inline copy stays deleted and the workflow pins keep admitting the packaged version.

---

### PR base-branch guard (required check)

`.github/workflows/pr-base-branch-guard.yml` fails any PR whose base branch is not the
default branch. Its job name -- **`Guard PR base branch`** -- is a **required status check**
in this repo's ruleset, so renaming the job or deleting the file makes `main` unmergeable
until the context is un-required first.

**What it protects against.** A PR based on another feature branch can squash-merge into
that branch, stranding its content off `main` behind a green **MERGED** badge. It has
happened three times in this ecosystem (`juniper-recurrence#7`/`#8`, `juniper-canopy#365`).

**Why it matters more than it looks.** Both rulesets here are scoped to `~DEFAULT_BRANCH`, so
a PR whose base is a feature branch is governed by **no ruleset at all** -- it has zero
required status checks and merges clean with nothing having run:

```bash
gh api repos/pcalnon/<repo>/rules/branches/feature%2Fanything --jq length   # -> 0
gh api repos/pcalnon/<repo>/rules/branches/main               --jq length   # -> 9
```

This workflow carries no `branches:` filter, so it is the **only** check that runs on such a
PR. It cannot block the merge there -- no ruleset applies -- but it turns a silent merge into
a visibly red one.

**If it fails.** Re-open the work against the default branch. The house practice is
**close and re-open** a fresh PR titled `[retarget #NNN]`. Retargeting in place is *not*
sufficient on its own: every `ci*.yml` here uses the default `pull_request` types
`[opened, synchronize, reopened]`, which exclude `edited`, so a retarget re-runs this guard
and nothing else -- the PR stays blocked on its other required contexts until a push or a
close/re-open.

**`stacked-pr` label.** Silences this guard for a deliberate stack. It does **not** make the
PR mergeable into `main`, and it does **not** re-land the stack -- do that separately.

Rollout and rationale: [juniper-ml#434](https://github.com/pcalnon/juniper-ml/issues/434).

## Deployment

### Docker

Multi-stage Dockerfile (`Dockerfile`):

```bash
# Build
docker build -t juniper-cascor:latest .

# Run
docker run -p 8200:8200 \
  -e JUNIPER_DATA_URL=http://localhost:8100 \
  -e JUNIPER_CASCOR_API_KEYS=your-key-here \
  juniper-cascor:latest
```

- **Base image**: `python:3.14-slim`
- **CPU-only PyTorch** (avoids ~4GB CUDA dependency)
- **Non-root user** in production stage

### Docker Compose

Per-service compose file in `conf/docker-compose.yaml` is **deprecated**. Use the unified orchestration in `juniper-deploy`:

```bash
cd ../juniper-deploy && make up
```

### Service Ports

| Service | Port | Health Endpoint |
|---------|------|-----------------|
| juniper-cascor | 8200 | `/v1/health` |

### Service Launcher

When running outside containers, juniper-cascor can auto-start companion services (`src/api/service_launcher.py`):

- `JUNIPER_CASCOR_AUTO_START_DATA_SERVICE=true` -- Start juniper-data
- `JUNIPER_CASCOR_AUTO_START_CANOPY=true` -- Start juniper-canopy

The launcher probes health endpoints before declaring readiness.

**Failed-start cleanup** (fail-closed registry / FD hygiene):

- `ManagedService.terminate` always closes the companion log handle in a `finally` (even when wait/kill raises).
- Health-probe exceptions are treated as unhealthy; the subprocess is terminated and removed from `_active_services`.
- If `terminate()` itself raises after a failed health check, the launcher still drops the stale `_active_services` entry (the removal is in a `finally`), so atexit / shutdown cannot chase an orphaned companion or leave port conflicts.

---

## Key Dependencies

The runtime and development dependencies, and what each one is relied on for. Moved to [`docs/REFERENCE.md` § Key Dependencies Reference](docs/REFERENCE.md#key-dependencies-reference) — read it when working on this area.

## Constants Configuration

The constants packages, what each owns, and the alignment requirements between them. Moved to [`docs/REFERENCE.md` § Constants Configuration Reference](docs/REFERENCE.md#constants-configuration-reference) — read it when working on this area.

## Documentation Files

Every documentation file, what belongs in it, and where it lives. Moved to [`docs/REFERENCE.md` § Documentation Files Reference](docs/REFERENCE.md#documentation-files-reference) — read it when working on this area.

## MCP Server Availability

### Serena

The project includes Serena MCP server configuration in `.serena/memories/`:

| File | Purpose |
|------|---------|
| `project_overview.md` | Project context for Serena |
| `code_style_conventions.md` | Coding standards context |
| `suggested_commands.md` | Suggested development commands |
| `task_completion_checklist.md` | Task completion guidelines |

Serena provides semantic code analysis tools for navigating the codebase, finding symbols, and understanding architecture through symbolic tools rather than raw file reads.

---

## Known Issues and Workarounds

- **Logger Pickling**: Loggers excluded from `__getstate__` for multiprocessing
- **GPU Support**: Tests disable GPU by default; use `--gpu` flag for GPU tests
- **Long-Running Tests**: Skipped by default; use `--run-long` to run them
- **Random Reproducibility**: Set `random_seed` in config for deterministic training
- **BLAS Thread Contention**: BLAS thread limits (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`) set before imports to prevent contention with multiprocessing workers

---

## Development Workflow

**Adding Features**:

1. Create feature in appropriate module
2. Add constants to `src/cascor_constants/`
3. Add tests in `src/tests/unit/` (or `src/tests/unit/api/` for API features)
4. Update documentation in `notes/` or `docs/`
5. Run tests: `cd src && python -m pytest tests/ -v`
6. Run linting: `pre-commit run --all-files`

**Adding Tests**:

1. Create test file following `test_<feature>.py` naming
2. Use appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
3. Use fixtures from `conftest.py`
4. Follow Arrange-Act-Assert pattern
5. For API tests, use `httpx.AsyncClient` with the `TestClient` pattern

---

## Performance Considerations

**Training Tips**:

- Optimize `candidate_pool_size` for CPU core count
- Use N-best candidate selection for faster convergence
- Tune `patience` for speed vs. accuracy tradeoff
- Configure BLAS thread limits for multiprocessing environments

**Benchmarks**:

Performance micro-benchmarks in `src/tests/performance/` cover:

- Forward pass latency
- Candidate training throughput
- Correlation calculation
- Output training
- Concurrency scaling
- Shared memory performance
- Autograd overhead
- End-to-end profiling

---

## Security Notes

- No secrets or API keys in codebase
- Sensitive files excluded via `.gitignore`
- `.env.example` provided as template; `.env` is gitignored
- Docker secrets supported for container deployments
- SOPS encryption available (`.sops.yaml`, `.env.enc`)
- Log files may contain training data -- handle appropriately
- Pre-commit hooks include `bandit` security scanner
- CI pipeline includes Gitleaks and pip-audit scans

---

## Script Placement

**Permanent utilities** live in `util/`. **Single-use / temporary / unfinished scripts** go in `util/ad-hoc/` (create on first use). See [`util/ad-hoc/README.md`](util/ad-hoc/README.md) for the per-script header / lifecycle conventions.

`/tmp/` is **prohibited** as the home for any script that produces, modifies, or analyzes repository content. `/tmp/` is reaped when sessions / sandboxes / containers end, and scripts placed there are lost (irrecoverable). `/tmp/` remains fine as a scratch *workspace* for intermediate artifacts the script itself creates and reads — the prohibition is on script *source files*.

This is an ecosystem-wide rule restated in the parent `Juniper/AGENTS.md` "Cross-Project Conventions" section. Motivating incident: irrecoverable loss of `phase4_consolidate.py` and `v2_citation_validate.py` from the juniper-ml requirements-snapshot effort.

---

## Worktree Procedures (Mandatory -- Task Isolation)

Git worktrees allow multiple branches to be checked out simultaneously in separate directories. For the Juniper ecosystem, all worktrees are centralized in **`/home/pcalnon/Development/python/Juniper/worktrees/`** using standardized naming convention: `<repo-name>--<branch-name>--<YYYYMMDD-HHMM>--<short-hash>`

**Full procedures**:

- **`notes/WORKTREE_SETUP_PROCEDURE.md`** -- Creating a worktree
- **`notes/WORKTREE_CLEANUP_PROCEDURE_V2.md`** -- Merging, removing, pushing (V2 fixes CWD-trap bug)

**Rules**:

- Centralized location only
- Clean before starting
- Push before merging
- Prune after cleanup
- No stale worktrees

---

## Thread Handoff (Mandatory -- Replaces Thread Compaction)

Critical instruction: Thread handoff MUST replace thread compaction when context limits approach.

**Trigger threshold**: Initiate handoff at **95% to 99%** of compaction threshold (within 1-5% of compaction threshold).

**Additional triggers**:

| Condition | Indicator |
|-----------|-----------|
| Context saturation | 15+ tool calls or 5+ file edits |
| Phase boundary | Logical work phase complete |
| Degraded recall | Re-reading files or re-asking questions |
| Multi-module transition | Moving between major components |
| User request | User says "hand off", "new thread", or similar |

**Do NOT handoff** when:

- Task nearly complete (< 2 remaining steps)
- Current thread still sharp
- Work tightly coupled and splitting loses critical state

**How to execute**:

1. Checkpoint work done, remains, discovered, files in play
2. Compose concise, actionable handoff summary
3. Present to user and recommend new thread
4. Include verification commands
5. State git status

**Rules**:

- Not optional -- every Claude Code instance must follow
- Handoff early, not late
- Don't duplicate CLAUDE.md content
- Be specific: include file paths, decisions, test status

---

## Contact

For questions, refer to:

- User documentation in `docs/`
- Development documentation in `notes/`
- Test examples in `src/tests/`
- Constants definitions in `src/cascor_constants/`
- API reference in `docs/api/API_REFERENCE.md`
