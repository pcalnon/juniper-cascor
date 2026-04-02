# AGENTS.md Drift Analysis

**Project**: Juniper Cascor
**Document Type**: Audit / Analysis
**Date**: 2026-04-02
**Auditor**: Claude Code (automated)
**Scope**: Full comparison of AGENTS.md (v0.3.17, dated 2026-02-05) against current codebase state (v0.4.0)

---

## Executive Summary

The juniper-cascor AGENTS.md file has experienced **significant drift** from the actual codebase. Since the last AGENTS.md update (2026-02-05), the project has undergone a major architectural expansion from a CLI-centric neural network trainer to a full **FastAPI REST/WebSocket service** with distributed worker support, lifecycle management, observability infrastructure, and security hardening. The AGENTS.md version header references 0.3.17 while `pyproject.toml` declares version 0.4.0.

**Severity**: High -- The current AGENTS.md omits the entire service layer (REST API, WebSocket infrastructure, lifecycle management, worker system, security, observability) which now constitutes approximately 40% of the source code.

---

## 1. Version and Metadata Drift

| Field | AGENTS.md Value | Actual Value | Status |
|-------|----------------|--------------|--------|
| Version | 0.3.17 | 0.4.0 | OUTDATED |
| Last Updated | 2026-02-05 | N/A (needs update) | STALE |
| Python Requirement | Not stated (implied >=3.12) | >=3.12 (pyproject.toml) | IMPLICIT |

---

## 2. Missing Major Components

### 2.1 FastAPI REST API Layer (`src/api/`)

**Severity**: Critical

The entire `src/api/` module is undocumented in AGENTS.md. This module contains:

| Component | Path | Purpose |
|-----------|------|---------|
| `app.py` | `src/api/app.py` | FastAPI application factory with lifespan management |
| `settings.py` | `src/api/settings.py` | Pydantic-based settings (environment variable configuration) |
| `security.py` | `src/api/security.py` | API key authentication and rate limiting |
| `middleware.py` | `src/api/middleware.py` | Security headers, body limits, auth middleware |
| `observability.py` | `src/api/observability.py` | JSON/text logging, Prometheus, Sentry, request IDs |
| `service_launcher.py` | `src/api/service_launcher.py` | Auto-start companion services (juniper-data, juniper-canopy) |
| `secrets.py` | `src/api/secrets.py` | Docker secrets file loader |

### 2.2 REST API Routes (`src/api/routes/`)

**Severity**: Critical

Eight route modules are completely undocumented:

| Route Module | Endpoints | Purpose |
|-------------|-----------|---------|
| `health.py` | `GET /v1/health`, `/v1/health/live`, `/v1/health/ready` | Health and readiness probes |
| `network.py` | `POST/GET/DELETE /v1/network`, `/v1/network/topology`, `/v1/network/stats` | Network CRUD and inspection |
| `training.py` | `POST /v1/training/{start,stop,pause,resume,reset}`, `GET /v1/training/{status,params}`, `PATCH /v1/training/params` | Training control |
| `dataset.py` | `GET /v1/dataset`, `/v1/dataset/data` | Dataset metadata and arrays |
| `decision_boundary.py` | `GET /v1/decision_boundary` | Decision boundary visualization |
| `metrics.py` | `GET /v1/metrics` | Prometheus metrics |
| `snapshots.py` | `GET/POST /v1/snapshots` | Network snapshot management |
| `workers.py` | `GET /v1/workers` | Remote worker registry |

### 2.3 WebSocket Infrastructure (`src/api/websocket/`)

**Severity**: Critical

The WebSocket subsystem is entirely absent from AGENTS.md:

| Component | Path | Purpose |
|-----------|------|---------|
| `manager.py` | `src/api/websocket/manager.py` | WebSocketManager (connection lifecycle, broadcast) |
| `control_stream.py` | `src/api/websocket/control_stream.py` | `/ws/control` -- training command channel |
| `training_stream.py` | `src/api/websocket/training_stream.py` | `/ws/training` -- real-time metrics stream |
| `worker_stream.py` | `src/api/websocket/worker_stream.py` | `/ws/v1/workers` -- remote worker protocol |
| `messages.py` | `src/api/websocket/messages.py` | Message builders (metrics, events, state) |

### 2.4 Lifecycle Management (`src/api/lifecycle/`)

**Severity**: Critical

Training lifecycle infrastructure is undocumented:

| Component | Path | Purpose |
|-----------|------|---------|
| `manager.py` | `src/api/lifecycle/manager.py` | TrainingLifecycleManager (thread-safe orchestration) |
| `monitor.py` | `src/api/lifecycle/monitor.py` | TrainingMonitor (metrics collection, callbacks) |
| `state_machine.py` | `src/api/lifecycle/state_machine.py` | TrainingStateMachine (deterministic state transitions) |

### 2.5 Remote Worker System (`src/api/workers/`)

**Severity**: Critical

Distributed worker infrastructure is undocumented:

| Component | Path | Purpose |
|-----------|------|---------|
| `registry.py` | `src/api/workers/registry.py` | WorkerRegistry (worker pool management) |
| `coordinator.py` | `src/api/workers/coordinator.py` | WorkerCoordinator (task dispatch, monitoring) |
| `protocol.py` | `src/api/workers/protocol.py` | Wire protocol (JSON + binary frames) |
| `security.py` | `src/api/workers/security.py` | Worker authentication and token management |
| `audit.py` | `src/api/workers/audit.py` | Worker audit logging |

### 2.6 API Request/Response Models (`src/api/models/`)

**Severity**: High

Pydantic models undocumented:

| Component | Path | Purpose |
|-----------|------|---------|
| `common.py` | `src/api/models/common.py` | ResponseEnvelope, ErrorResponse |
| `health.py` | `src/api/models/health.py` | ReadinessResponse with dependencies |
| `network.py` | `src/api/models/network.py` | NetworkCreateRequest |
| `training.py` | `src/api/models/training.py` | TrainingStartRequest, TrainingParamUpdateRequest |

### 2.7 Server Entry Point (`src/server.py`)

**Severity**: High

`src/server.py` is the FastAPI server launcher (uvicorn) and is not listed in the Key Entry Points table. This is now the primary production entry point, while `src/main.py` serves as a CLI-only entry point.

### 2.8 Parallelism Module (`src/parallelism/`)

**Severity**: Medium

The `TaskDistributor` class (`src/parallelism/task_distributor.py`) implements local-first scheduling across multiprocessing and WebSocket workers. This is distinct from the older multiprocessing description in AGENTS.md.

---

## 3. Outdated Sections

### 3.1 Key Entry Points Table

**Current AGENTS.md**:
```
| src/main.py | Main application entry point |
| src/cascade_correlation/cascade_correlation.py | Core neural network implementation |
| src/spiral_problem/spiral_problem.py | Two-spiral problem solver |
| src/candidate_unit/candidate_unit.py | Candidate unit for network growth |
| src/profiling/ | Profiling infrastructure |
| src/tests/run_tests.bash | Test runner script |
| src/tests/conftest.py | Test configuration and fixtures |
| util/profile_training.bash | py-spy sampling profiler |
```

**Missing entries**:
- `src/server.py` -- FastAPI server launcher (primary production entry point)
- `src/api/app.py` -- FastAPI application factory
- `src/api/settings.py` -- Application configuration
- `src/parallelism/task_distributor.py` -- Distributed task scheduling

### 3.2 Directory Structure

**AGENTS.md describes** a flat structure: `src/`, `conf/`, `util/`, `notes/`, `data/`, `logs/`, `images/`, `reports/`

**Missing directories**:
- `src/api/` (with 6 subdirectories: `lifecycle/`, `routes/`, `websocket/`, `workers/`, `models/`, and root-level modules)
- `src/parallelism/`
- `docs/` (with subdirectories: `api/`, `ci_cd/`, `install/`, `overview/`, `source/`, `testing/`)
- `scripts/` (with `tls/` subdirectory)
- `.github/workflows/` (5 workflow files)
- `dist/` (build artifacts)

### 3.3 Environment Variables Table

**Current AGENTS.md lists 4 variables**:
- `CASCOR_LOG_LEVEL`
- `JUNIPER_DATA_URL`
- `CASCOR_BACKEND_PATH`
- `JUNIPER_DATA_API_KEY`

**Missing variables** (from `src/api/settings.py`, all prefixed `JUNIPER_CASCOR_`):
- `JUNIPER_CASCOR_HOST` -- API host (default: 127.0.0.1)
- `JUNIPER_CASCOR_PORT` -- API port (default: 8200)
- `JUNIPER_CASCOR_LOG_LEVEL` -- API log level
- `JUNIPER_CASCOR_CORS_ORIGINS` -- CORS allowed origins
- `JUNIPER_CASCOR_WS_MAX_CONNECTIONS` -- WebSocket connection limit (default: 50)
- `JUNIPER_CASCOR_WS_HEARTBEAT_INTERVAL_SEC` -- WS heartbeat interval (default: 30)
- `JUNIPER_CASCOR_API_KEYS` -- API key list for authentication
- `JUNIPER_CASCOR_RATE_LIMIT_ENABLED` -- Rate limiting toggle
- `JUNIPER_CASCOR_RATE_LIMIT_REQUESTS_PER_MINUTE` -- Rate limit (default: 60)
- `JUNIPER_CASCOR_LOG_FORMAT` -- Log format (text/json)
- `JUNIPER_CASCOR_SENTRY_DSN` -- Sentry integration DSN
- `JUNIPER_CASCOR_METRICS_ENABLED` -- Prometheus metrics toggle
- `JUNIPER_CASCOR_AUTO_START` -- Auto-start training on startup
- `JUNIPER_CASCOR_AUTO_START_DATA_SERVICE` -- Auto-start juniper-data
- `JUNIPER_CASCOR_AUTO_START_CANOPY` -- Auto-start juniper-canopy
- `JUNIPER_CASCOR_AUTO_DATASET` -- Auto dataset type (default: spiral)
- `JUNIPER_CASCOR_AUTO_TRAIN_EPOCHS` -- Auto-start training epochs
- `JUNIPER_CASCOR_REMOTE_WORKERS_HEARTBEAT_TIMEOUT` -- Worker heartbeat timeout (default: 30s)
- `JUNIPER_CASCOR_REMOTE_WORKERS_TASK_REASSIGNMENT_TIMEOUT` -- Task reassignment timeout (default: 120s)

### 3.4 Essential Commands Section

**Missing commands**:
- `cd src && python server.py` -- Start the FastAPI server
- `uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8200` -- Production server startup
- Docker build/run commands
- API testing with curl/httpie examples

### 3.5 Key Dependencies

**Missing from AGENTS.md**:
- `fastapi` -- REST API framework
- `uvicorn` -- ASGI server
- `pydantic` / `pydantic-settings` -- Configuration and validation
- `sentry-sdk` -- Error tracking
- `python-dotenv` -- Environment file loading
- `prometheus-client` -- Metrics (optional)

### 3.6 Testing Infrastructure

**Missing test categories**:
- `src/tests/unit/api/` -- 30+ API-specific unit tests
- `src/tests/integration/api/` -- API integration tests (full lifecycle, WebSocket streaming)
- `src/tests/performance/` -- Performance benchmarks (micro-benchmarks, concurrency scaling, shared memory)
- Test helpers (`src/tests/helpers/`) and mocks (`src/tests/mocks/`)

**Missing test markers**:
- API-related markers (if any)
- Performance test markers for micro-benchmarks

### 3.7 Documentation Files Table

**Missing entries**:
- `docs/INDEX.md`
- `docs/DOCUMENTATION_OVERVIEW.md`
- `docs/DEVELOPER_CHEATSHEET.md`
- `docs/USER_MANUAL.md`
- `docs/api/API_REFERENCE.md`
- `docs/api/API_SCHEMAS.md`
- `docs/ci_cd/` (5 files)
- `docs/install/` (4 files)
- `docs/testing/` (5 files)
- `docs/source/` (4 files)
- `docs/overview/CONSTANTS_GUIDE.md`
- Many notes/ documents (roadmaps, plans, setup guides)

### 3.8 CI/CD Workflows

**Completely absent from AGENTS.md**:
- `.github/workflows/ci.yml` -- Main CI pipeline (pre-commit, unit tests, integration tests, security scanning)
- `.github/workflows/scheduled-tests.yml` -- Scheduled test runs
- `.github/workflows/publish.yml` -- Package publishing
- `.github/workflows/lockfile-update.yml` -- Dependency lockfile updates
- `.github/workflows/security-scan.yml` -- Security scanning
- `.github/CODEOWNERS` -- Code ownership rules
- `.github/dependabot.yml` -- Dependency update automation

### 3.9 Serialization System

**AGENTS.md describes HDF5 serialization** but the codebase appears to have shifted toward **NPZ (numpy compressed)** format for snapshot storage. The snapshot module still exists but the primary format may have evolved.

### 3.10 Multiprocessing Section

**AGENTS.md describes basic multiprocessing** with custom manager for task/result queues. The actual implementation now includes:
- `TaskDistributor` with local-first scheduling policy
- Remote WebSocket worker integration
- Fallback from remote to local on failure/timeout
- Shared memory support (numpy arrays)
- BLAS thread limit management

### 3.11 Configuration Section

**AGENTS.md describes `CascadeCorrelationConfig`** for network configuration but omits:
- `Settings` class (Pydantic BaseSettings) for application/API configuration
- Docker secrets support
- `.env` file loading
- Environment variable prefix system (`JUNIPER_CASCOR_*`)

---

## 4. Accuracy Issues in Existing Content

### 4.1 Application Execution Command

AGENTS.md states: `cd src && python main.py`

This is still valid for CLI-mode training but is no longer the primary execution path. The service is now started via:
- `cd src && python server.py` (development)
- `uvicorn api.app:create_app --factory` (production)

### 4.2 Serialization Performance Claims

AGENTS.md claims:
- Save (100 units) < 2 seconds
- Load (100 units) < 3 seconds
- Checksum < 200ms

These benchmarks may need re-validation against the current implementation.

---

## 5. Structural Gaps

### 5.1 No REST API Reference in AGENTS.md

There is no section documenting the REST API endpoints, request/response schemas, authentication requirements, or error handling patterns. This is now a primary interface for the application.

### 5.2 No WebSocket Protocol Documentation

The WebSocket channels (`/ws/control`, `/ws/training`, `/ws/v1/workers`) are undocumented, including message formats, authentication, and connection lifecycle.

### 5.3 No Security Documentation

API key authentication, rate limiting, security headers, CORS configuration, worker security, and TLS certificate generation are all undocumented.

### 5.4 No Observability Documentation

Logging formats (JSON/text), Prometheus metrics, Sentry integration, and request ID propagation are undocumented.

### 5.5 No Deployment Documentation Reference

Dockerfile, docker-compose.yaml, CI/CD workflows, and environment configuration for deployment are not referenced.

### 5.6 No MCP Server Availability Documentation

The project has Serena MCP server configuration (`.serena/memories/`) but this is not documented in AGENTS.md.

---

## 6. Impact Assessment

| Category | Severity | Impact |
|----------|----------|--------|
| Missing API layer docs | Critical | New contributors cannot understand the service architecture |
| Missing WebSocket docs | Critical | Client integration impossible without reading source |
| Missing worker system docs | Critical | Distributed deployment undocumented |
| Missing security docs | High | Security configuration opaque |
| Outdated version | High | Misleading version reference |
| Outdated directory structure | High | Navigation guidance incorrect |
| Missing environment variables | High | Configuration incomplete for deployment |
| Missing CI/CD docs | Medium | Pipeline understanding requires reading YAML directly |
| Missing observability docs | Medium | Monitoring setup undocumented |
| Missing docs/ reference | Medium | Documentation discoverability impaired |
| Stale entry points | Medium | Primary server entry point not listed |
| Stale dependencies | Low | Core ML deps correct; service deps missing |

---

## 7. Recommendations

1. **Rewrite AGENTS.md** to reflect the application's evolution from CLI tool to REST/WebSocket service
2. **Add dedicated sections** for: REST API, WebSocket Protocol, Security, Observability, Deployment, CI/CD
3. **Update all tables** (entry points, environment variables, dependencies, documentation files, directory structure)
4. **Update version** to 0.4.0 and set appropriate last-updated date
5. **Restructure the document** to lead with the service architecture (since that's the primary operational mode) while preserving CLI documentation
6. **Add MCP server availability** section documenting Serena integration

---

*End of Analysis*
