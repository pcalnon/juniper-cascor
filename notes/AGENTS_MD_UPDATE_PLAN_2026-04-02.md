# AGENTS.md Update Plan

**Project**: Juniper Cascor
**Document Type**: Planning
**Date**: 2026-04-02
**Reference**: `notes/AGENTS_MD_DRIFT_ANALYSIS_2026-04-02.md`
**Scope**: Phased plan to bring AGENTS.md into alignment with codebase v0.4.0

---

## Objective

Update the juniper-cascor AGENTS.md file to accurately reflect the current application state, including the full service architecture (REST API, WebSocket, lifecycle management, distributed workers), security infrastructure, observability, CI/CD, and deployment configuration.

---

## Phase 1: Metadata and Structural Updates

**Goal**: Correct version, date, and document organization.

### Steps

1.1. Update version from 0.3.17 to 0.4.0
1.2. Update "Last Updated" to 2026-04-02
1.3. Restructure document to lead with service architecture (primary operational mode)
1.4. Preserve CLI-mode documentation as secondary usage path

### Affected Sections
- Header / metadata block
- Document structure / table of contents

---

## Phase 2: Quick Reference Updates

**Goal**: Ensure the Quick Reference section provides accurate, actionable commands and pointers.

### Steps

2.1. **Essential Commands**: Add server startup commands alongside existing CLI commands
   - `cd src && python server.py` (development server)
   - `uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8200` (production)
   - Docker build/run commands
   - Keep existing CLI, test, profiling, linting commands

2.2. **Environment Variables**: Expand table with all `JUNIPER_CASCOR_*` settings from `src/api/settings.py`
   - Host, port, log level, CORS
   - WebSocket configuration
   - API keys, rate limiting
   - Observability (log format, Sentry DSN, metrics)
   - Auto-start configuration
   - Remote worker timeouts
   - Retain existing `CASCOR_LOG_LEVEL`, `JUNIPER_DATA_URL`, `JUNIPER_DATA_API_KEY`
   - Remove `CASCOR_BACKEND_PATH` if no longer applicable (verify first)

2.3. **Key Entry Points**: Add missing entries
   - `src/server.py` -- FastAPI server launcher
   - `src/api/app.py` -- Application factory
   - `src/api/settings.py` -- Configuration
   - `src/parallelism/task_distributor.py` -- Task distribution
   - Retain all existing entries

---

## Phase 3: Architecture Documentation

**Goal**: Document the service architecture that constitutes ~40% of the codebase.

### Steps

3.1. **Add REST API section**
   - Endpoint inventory (all routes with methods, paths, purposes)
   - Request/response model descriptions
   - Authentication requirements (which endpoints require API keys)
   - Error handling patterns (ResponseEnvelope, ErrorResponse)

3.2. **Add WebSocket Protocol section**
   - Three channels: `/ws/control`, `/ws/training`, `/ws/v1/workers`
   - Message formats and command vocabularies
   - Authentication mechanisms
   - Connection lifecycle and limits

3.3. **Add Lifecycle Management section**
   - TrainingLifecycleManager, TrainingStateMachine, TrainingMonitor
   - State transitions and valid commands
   - Callback system for training events

3.4. **Add Remote Worker System section**
   - WorkerRegistry, WorkerCoordinator
   - Worker protocol (JSON + binary frames)
   - Heartbeat, task assignment, result collection
   - Security (authentication, token management)
   - Audit logging

3.5. **Add Middleware Stack section**
   - Execution order (LIFO)
   - SecurityHeadersMiddleware, RequestBodyLimitMiddleware, SecurityMiddleware
   - PrometheusMiddleware, RequestIdMiddleware, CORSMiddleware

3.6. **Add API Models section**
   - Pydantic models: ResponseEnvelope, NetworkCreateRequest, TrainingStartRequest, etc.

---

## Phase 4: Security and Observability

**Goal**: Document security infrastructure and monitoring capabilities.

### Steps

4.1. **Add Security section**
   - API key authentication (X-API-Key header, HMAC comparison)
   - Rate limiting (fixed-window per IP, thread-safe)
   - Security headers (CSP, HSTS, X-Frame-Options, etc.)
   - Request body size limits (10MB)
   - WebSocket Origin rejection (worker endpoints)
   - Docker secrets support
   - TLS certificate generation (`scripts/tls/generate_certs.bash`)

4.2. **Add Observability section**
   - Logging: JSON format (production) and text format (development)
   - Request ID propagation via ContextVar
   - Prometheus metrics (http_requests_total, http_request_duration_seconds)
   - Sentry integration for error tracking
   - Health check probes (liveness, readiness)

---

## Phase 5: Infrastructure Documentation

**Goal**: Document CI/CD, deployment, and tooling.

### Steps

5.1. **Add CI/CD Workflows section**
   - `.github/workflows/ci.yml` -- Main pipeline (pre-commit, tests, security)
   - `.github/workflows/scheduled-tests.yml` -- Periodic test runs
   - `.github/workflows/publish.yml` -- Package publishing
   - `.github/workflows/lockfile-update.yml` -- Dependency management
   - `.github/workflows/security-scan.yml` -- Security scanning
   - `.github/CODEOWNERS` and `.github/dependabot.yml`

5.2. **Add Deployment section**
   - Dockerfile reference
   - `conf/docker-compose.yaml` reference
   - Kubernetes readiness (health probes, environment configuration)
   - Service ports and networking

5.3. **Update Configuration section**
   - Add Pydantic Settings documentation
   - Add `.env` file documentation
   - Add Docker secrets support
   - Retain CascadeCorrelationConfig documentation

---

## Phase 6: Existing Section Updates

**Goal**: Correct and expand existing sections.

### Steps

6.1. **Update Directory Structure**
   - Add `src/api/` with all subdirectories
   - Add `src/parallelism/`
   - Add `docs/` with subdirectories
   - Add `scripts/` with `tls/`
   - Add `.github/workflows/`
   - Add `dist/`

6.2. **Update Key Dependencies**
   - Add: fastapi, uvicorn, pydantic, pydantic-settings, sentry-sdk, python-dotenv
   - Add optional: prometheus-client
   - Retain existing dependencies

6.3. **Update Testing Infrastructure**
   - Document API unit tests (`src/tests/unit/api/`, 30+ files)
   - Document API integration tests (`src/tests/integration/api/`)
   - Document performance tests (`src/tests/performance/`)
   - Document test helpers and mocks
   - Verify/update test marker list

6.4. **Update Multiprocessing section**
   - Rename to "Parallelism and Distribution" or similar
   - Add TaskDistributor (local-first scheduling)
   - Add remote worker integration
   - Add shared memory support
   - Add BLAS thread limit management
   - Retain existing pickling documentation

6.5. **Update Serialization System**
   - Verify HDF5 vs NPZ format status
   - Update if format has changed
   - Keep CLI tools documentation

6.6. **Update Documentation Files table**
   - Add all `docs/` subdirectory entries
   - Add new `notes/` entries (roadmaps, plans, setup guides)
   - Reference `notes/history/` for archived documents

---

## Phase 7: New Sections

**Goal**: Add sections for previously undocumented areas.

### Steps

7.1. **Add Service Launcher section**
   - Auto-start companion services (juniper-data, juniper-canopy)
   - Health probe configuration
   - Environment variable overrides

7.2. **Add MCP Server Availability section**
   - Document Serena MCP server configuration
   - Reference `.serena/memories/` directory
   - Note available MCP tools for development

---

## Dependencies and Ordering

```
Phase 1 (metadata) ──> Phase 2 (quick ref) ──> Phase 3 (architecture) ──> Phase 4 (security/obs)
                                                                      ──> Phase 5 (infrastructure)
                                                                      ──> Phase 6 (existing updates)
                                                                      ──> Phase 7 (new sections)
```

Phases 3-7 can be executed in any order after Phases 1-2, but are listed in priority order.

---

## Validation Criteria

- [ ] All source files in `src/api/` are referenced in appropriate sections
- [ ] All environment variables from `src/api/settings.py` are documented
- [ ] All REST endpoints are listed with methods, paths, and purposes
- [ ] All WebSocket channels are documented with message formats
- [ ] Version matches `pyproject.toml`
- [ ] Directory structure matches actual filesystem
- [ ] Key entry points table includes server.py and API modules
- [ ] CI/CD workflows are listed
- [ ] Dependencies match `pyproject.toml`
- [ ] Test categories match actual test directory structure
- [ ] Documentation files table matches actual docs/ and notes/ contents

---

*End of Plan*
