# AGENTS.md Update Development Roadmap

**Project**: Juniper Cascor
**Document Type**: Development Roadmap
**Date**: 2026-04-02
**Reference**: `notes/AGENTS_MD_DRIFT_ANALYSIS_2026-04-02.md`, `notes/AGENTS_MD_UPDATE_PLAN_2026-04-02.md`
**Priority Legend**: P0 = Critical, P1 = High, P2 = Medium, P3 = Low

---

## Roadmap Summary

| Phase | Description | Priority | Tasks | Status |
|-------|------------|----------|-------|--------|
| 1 | Metadata and Structure | P0 | 4 | Pending |
| 2 | Quick Reference | P0 | 3 | Pending |
| 3 | Service Architecture | P0 | 6 | Pending |
| 4 | Security and Observability | P1 | 2 | Pending |
| 5 | Infrastructure | P1 | 3 | Pending |
| 6 | Existing Section Updates | P1 | 6 | Pending |
| 7 | New Sections | P2 | 2 | Pending |
| 8 | Validation | P0 | 3 | Pending |

---

## Phase 1: Metadata and Structure (P0)

| # | Task | Description | Status |
|---|------|-------------|--------|
| 1.1 | Update version | Change 0.3.17 to 0.4.0 in header | Pending |
| 1.2 | Update date | Set "Last Updated" to 2026-04-02 | Pending |
| 1.3 | Restructure document | Reorder to lead with service architecture as primary mode | Pending |
| 1.4 | Preserve CLI docs | Retain CLI-mode documentation as secondary path | Pending |

---

## Phase 2: Quick Reference (P0)

| # | Task | Description | Status |
|---|------|-------------|--------|
| 2.1 | Add server commands | Add `python server.py`, `uvicorn`, Docker commands to Essential Commands | Pending |
| 2.2 | Expand env vars table | Add all 19 `JUNIPER_CASCOR_*` settings from `src/api/settings.py` | Pending |
| 2.3 | Update entry points | Add `server.py`, `api/app.py`, `api/settings.py`, `parallelism/task_distributor.py` | Pending |

---

## Phase 3: Service Architecture (P0)

| # | Task | Description | Status |
|---|------|-------------|--------|
| 3.1 | REST API section | Document all routes, methods, request/response models, auth requirements | Pending |
| 3.2 | WebSocket section | Document 3 channels, message formats, auth, connection lifecycle | Pending |
| 3.3 | Lifecycle section | Document TrainingLifecycleManager, state machine, monitor, callbacks | Pending |
| 3.4 | Worker system section | Document registry, coordinator, protocol, security, audit | Pending |
| 3.5 | Middleware section | Document stack order and each middleware component | Pending |
| 3.6 | API models section | Document Pydantic request/response models | Pending |

---

## Phase 4: Security and Observability (P1)

| # | Task | Description | Status |
|---|------|-------------|--------|
| 4.1 | Security section | API keys, rate limiting, headers, body limits, TLS, Docker secrets | Pending |
| 4.2 | Observability section | Logging formats, Prometheus, Sentry, request IDs, health probes | Pending |

---

## Phase 5: Infrastructure (P1)

| # | Task | Description | Status |
|---|------|-------------|--------|
| 5.1 | CI/CD section | Document 5 workflows, CODEOWNERS, dependabot | Pending |
| 5.2 | Deployment section | Dockerfile, docker-compose, K8s readiness, ports | Pending |
| 5.3 | Configuration section | Pydantic Settings, .env files, Docker secrets, env prefix | Pending |

---

## Phase 6: Existing Section Updates (P1)

| # | Task | Description | Status |
|---|------|-------------|--------|
| 6.1 | Directory structure | Add `src/api/`, `src/parallelism/`, `docs/`, `scripts/`, `.github/` | Pending |
| 6.2 | Dependencies | Add fastapi, uvicorn, pydantic, pydantic-settings, sentry-sdk, etc. | Pending |
| 6.3 | Testing infrastructure | Add API tests, performance tests, helpers, mocks | Pending |
| 6.4 | Parallelism | Expand to cover TaskDistributor, remote workers, shared memory | Pending |
| 6.5 | Serialization | Verify format (HDF5 vs NPZ) and update accordingly | Pending |
| 6.6 | Documentation table | Add docs/ entries, new notes/ entries, reference history/ | Pending |

---

## Phase 7: New Sections (P2)

| # | Task | Description | Status |
|---|------|-------------|--------|
| 7.1 | Service launcher | Auto-start companions, health probes, env overrides | Pending |
| 7.2 | MCP availability | Document Serena configuration and `.serena/memories/` | Pending |

---

## Phase 8: Validation (P0)

| # | Task | Description | Status |
|---|------|-------------|--------|
| 8.1 | Cross-reference audit | Verify every AGENTS.md claim against actual codebase | Pending |
| 8.2 | Completeness check | Confirm all src/api/ files, env vars, routes are documented | Pending |
| 8.3 | Accuracy check | Verify commands, paths, defaults, and examples are correct | Pending |

---

## Execution Notes

- Phases 1-3 are **blocking** and must be completed first (P0)
- Phases 4-7 can be parallelized after Phase 3
- Phase 8 (validation) must run after all other phases
- Total estimated tasks: 29
- All work targets a single file: `AGENTS.md` (with deliverable documents in `notes/`)

---

*End of Roadmap*
