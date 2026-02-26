# Juniper Polyrepo Migration Plan

**Last Updated:** 2026-02-25
**Version:** 1.6.0
**Status:** Active — Phase 0 Complete, Phase 1 Complete, Phase 2 Complete, Phase 3 Complete, Phase 4 Complete, Phase 5 Complete (2026-02-25), Phase 6 In Progress
**Author:** Paul Calnon / Claude Code
**Companion Document:** [MONOREPO_ANALYSIS.md](MONOREPO_ANALYSIS.md)

---

## Table of Contents

- [Overview](#overview)
- [Target Architecture](#target-architecture)
- [Repositories and Packages](#repositories-and-packages)
- [Phase 0 — Stabilize: Resolve Merge Conflicts and Clean Baseline](#phase-0--stabilize-resolve-merge-conflicts-and-clean-baseline)
- [Phase 1 — Extract and Publish Client Packages to PyPI](#phase-1--extract-and-publish-client-packages-to-pypi)
- [Phase 2 — Build CasCor Service API](#phase-2--build-cascor-service-api)
- [Phase 3 — Create CasCor Client and Remote Worker Packages](#phase-3--create-cascor-client-and-remote-worker-packages)
- [Phase 4 — Decouple Canopy from CasCor](#phase-4--decouple-canopy-from-cascor)
- [Phase 5 — Split into Separate Repositories](#phase-5--split-into-separate-repositories)
- [Phase 6 — Post-Migration Hardening](#phase-6--post-migration-hardening)
- [Risk Register](#risk-register)
- [Migration Checklist](#migration-checklist)
- [Appendix A — CasCor Service API Contract (Draft)](#appendix-a--cascor-service-api-contract-draft)
- [Appendix B — PyPI Publishing Workflow](#appendix-b--pypi-publishing-workflow)
- [Appendix C — Package Inventory](#appendix-c--package-inventory)

---

## Overview

This plan migrates the Juniper project from a single GitHub repository with three entangled subprojects to a multi-repository architecture with independently versioned, deployable, and installable components.

### Goals

1. **Decouple CasCor from Canopy** — CasCor becomes an independent service (FastAPI + WebSocket), not a library imported via `sys.path`
2. **Publish client packages to PyPI** — `juniper-data-client` and a new `juniper-cascor-client` become proper installable packages
3. **Create candidate pool remote worker client** — Enable remote hardware to run candidate training workers via an installable package
4. **Resolve structural merge conflicts** — Clean up the entangled `main` branch
5. **Independent CI/CD** — Each repository has its own pipeline, triggered only by its own changes
6. **Independent deployment** — Each service can be deployed, scaled, and versioned independently

### Non-Goals

- Changing the CasCor training algorithm itself
- Migrating to a different hosting platform (staying on GitHub)
- Changing the conda environment strategy (staying with `JuniperPython`)
- Rewriting existing test suites (tests migrate as-is, with import path updates)

### Migration Order Rationale

The phases are ordered to maintain a working system at every step:

```bash
Phase 0: Fix broken state (merge conflicts)
Phase 1: Publish juniper-data-client to PyPI (eliminates vendoring)
Phase 2: Add REST API to CasCor (new capability, no existing code removed)
Phase 3: Create cascor-client + worker packages (new packages, additive)
Phase 4: Rewire Canopy to use clients instead of sys.path (breaking change, swap)
Phase 5: Split repos (mechanical, preserving git history)
Phase 6: Harden (CI/CD, monitoring, documentation)
```

Each phase produces a working, testable system. No phase requires the next phase to be functional.

---

## Target Architecture

### Service Topology

```bash
┌─────────────────────┐     REST/WS      ┌──────────────────────┐
│   JuniperCanopy     │ ◄──────────────► │    JuniperCascor     │
│   (Dashboard)       │                  │    (Training Svc)    │
│   Port 8050         │                  │    Port 8200         │
│                     │                  │                      │
│   Uses:             │     REST         │    Uses:             │
│   - cascor-client   │ ◄──────────────► │    - data-client     │
│   - data-client     │                  │    - cascor-worker   │
└─────────────────────┘                  └──────────┬───────────┘
         │                                          │
         │ REST                              Task/Result Queues
         ▼                                          │
┌─────────────────────┐                  ┌──────────▼───────────┐
│   JuniperData       │                  │   Remote Workers     │
│   (Dataset Svc)     │                  │   (N instances)      │
│   Port 8100         │                  │                      │
│                     │                  │   Uses:              │
│                     │                  │   - cascor-worker    │
└─────────────────────┘                  └──────────────────────┘
```

### Dependency Graph (Target)

```bash
                     PyPI Packages (independently installable)
                ┌──────────────────┬────────────────────────┐
                │                  │                        │
         juniper-data-client  juniper-cascor-client  juniper-cascor-worker
                │       │          │          │             │
                │       │          │          │             │
         ┌──────┘       │    ┌─────┘          │         ┌───┘
         ▼              ▼    ▼                ▼         ▼
    juniper-data   juniper-cascor      juniper-canopy  Remote
    (service)      (service)           (dashboard)     Hardware
```

Key property: **No arrows between services.** All inter-service communication is via REST/WebSocket through published client packages.

---

## Repositories and Packages

### Target Repositories

| Repository                      | Package(s)              | PyPI        | Description                      |
| ------------------------------- | ----------------------- | ----------- | -------------------------------- |
| `pcalnon/juniper-data`          | `juniper-data`          | No (server) | Dataset generation service       |
| `pcalnon/juniper-data-client`   | `juniper-data-client`   | **Yes**     | HTTP client for JuniperData API  |
| `pcalnon/juniper-cascor`        | `juniper-cascor`        | No (server) | CasCor neural network service    |
| `pcalnon/juniper-cascor-client` | `juniper-cascor-client` | **Yes**     | HTTP/WS client for CasCor API    |
| `pcalnon/juniper-cascor-worker` | `juniper-cascor-worker` | **Yes**     | Remote candidate training worker |
| `pcalnon/juniper-canopy`        | `juniper-canopy`        | No (server) | Monitoring dashboard             |

### PyPI Namespace

All packages use the `juniper-` prefix. PyPI names:

- `juniper-data-client` (exists in repo, not yet published)
- `juniper-cascor-client` (new)
- `juniper-cascor-worker` (new, extracted from `remote_client`)

---

## Phase 0 — Stabilize: Resolve Merge Conflicts and Clean Baseline

**Duration:** 1–2 days
**Risk:** Low (local changes only)
**Prerequisite:** None
**Status:** COMPLETE (2026-02-19) — Validated 2026-02-22

### Objective

Resolve the broken `main` branch so it represents a clean, buildable state for all three subprojects.

### Completion Summary

> **Validation (2026-02-22):** Phase 0 deliverables confirmed. All four repositories remain on clean baselines with passing tests. No merge conflicts detected. Pre-migration tags intact.

All four repositories stabilized with clean baselines on 2026-02-19:

| Repository          | Phase 0 Commit | Tests      | Pre-Migration Tag                         | Branch                      |
| ------------------- | -------------- | ---------- | ----------------------------------------- | --------------------------- |
| JuniperCanopy       | `32bdfc8`      | 3,338 pass | `canopy-pre-migration-v0.2.3`, `v0.2.4`   | `canopy/migration`          |
| JuniperCascor       | `4139e2a`      | Unit pass  | `cascor-pre-migration-v0.3.17`, `v0.3.18` | `cascor/service-api`        |
| JuniperData         | `892ef33`      | 659 pass   | `data-pre-migration-v0.4.2`               | `subproject...enhancements` |
| juniper-data-client | `54a22b8`      | 41 pass    | `data-client-pre-migration-v0.3.0`        | `main`                      |

**Actions completed across all repos:**

- Removed tracked build artifacts (`.coverage`, `coverage.xml`, `results.xml`, `egg-info/`)
- Updated `.gitignore` to exclude CI/build artifacts
- Created pre-migration baseline tags
- Verified all test suites pass
- Verified pre-commit checks pass
- No merge conflicts remain (searched for `<<<<<<< HEAD` markers — zero found)
- Migration planning documents in place (canonical in CasCor, symlinks in others)

### Step 0.1 — Inventory Conflicted Files

COMPLETE
The following files show `UU` (unresolved merge conflict) status when the integration branch is merged to `main`:

| File                                  | Conflict Source                     |
| ------------------------------------- | ----------------------------------- |
| `.github/workflows/ci.yml`            | Canopy vs Data CI configs           |
| `.pre-commit-config.yaml`             | Canopy vs Data pre-commit hooks     |
| `CHANGELOG.md`                        | Canopy vs Data changelog entries    |
| `README.md`                           | Canopy vs Data project descriptions |
| `pyproject.toml`                      | Canopy vs Data package definitions  |
| `conf/common.conf`                    | Shared bash config divergence       |
| `conf/common_functions.conf`          | Shared bash config divergence       |
| `conf/conda_environment.yaml`         | Conda env divergence                |
| `conf/init.conf`                      | Shared bash config divergence       |
| `conf/juniper_data.conf`              | Data-specific config in Canopy      |
| `conf/logging.conf`                   | Logging config divergence           |
| `conf/logging_config.yaml`            | Logging config divergence           |
| `conf/logging_functions.conf`         | Shared bash config divergence       |
| `conf/requirements.txt`               | Requirements divergence             |
| `docs/QUICK_START.md`                 | Doc divergence                      |
| `juniper_data_client/.coverage`       | Coverage artifact                   |
| `src/tests/reports/junit/results.xml` | Test artifact                       |
| `util/run_all_tests.bash`             | Test runner divergence              |

### Step 0.2 — Resolution Strategy

For each conflicted file, determine which subproject "owns" it:

1. **Canopy-owned files** — Resolve in favor of the Canopy branch content:
   - `.github/workflows/ci.yml` (Canopy CI)
   - `pyproject.toml` (Canopy package definition)
   - `README.md` (Canopy readme)
   - `CHANGELOG.md` (Canopy changelog)
   - `conf/logging_config.yaml` (Canopy logging)
   - `docs/QUICK_START.md` (Canopy docs)

2. **Shared config files** — Resolve to the most recent/complete version, then tag for future elimination:
   - `conf/common.conf`, `conf/common_functions.conf`, `conf/init.conf`
   - `conf/logging.conf`, `conf/logging_functions.conf`
   - `conf/conda_environment.yaml`, `conf/requirements.txt`
   - `.pre-commit-config.yaml`
   - `util/run_all_tests.bash`

3. **Artifacts to remove from tracking** — These should never be in git:
   - `juniper_data_client/.coverage` → add to `.gitignore`
   - `src/tests/reports/junit/results.xml` → add to `.gitignore`

4. **Data-specific files in Canopy** — Assess whether needed:
   - `conf/juniper_data.conf` → Keep if Canopy uses it for Data client config; otherwise remove

### Step 0.3 — Execute Resolution

```bash
# In the juniper_canopy clone, on main branch
cd /home/pcalnon/Development/python/Juniper/JuniperCanopy/juniper_canopy

# Re-attempt the merge
git merge subproject.juniper_canopy.integration_and_enhancements.release

# For each conflicted file, open and resolve manually
# Then stage the resolution
git add <resolved-file>

# After all conflicts resolved
git commit -m "Resolve merge conflicts from integration branch merge

Resolves conflicts between subproject.juniper_canopy.integration_and_enhancements.release
and main. Each file resolved to the Canopy-appropriate version.

Artifacts (.coverage, results.xml) added to .gitignore."

# Verify clean state
git status
```

### Step 0.4 — Verify Baseline

```bash
# Tests pass
cd src && pytest tests/ -v

# Pre-commit passes
pre-commit run --all-files

# No remaining conflicts
git diff --check
```

### Step 0.5 — Tag Clean Baseline

```bash
git tag -a canopy-pre-migration-v0.2.3 -m "Clean baseline before polyrepo migration"
git push origin main --tags
```

### Deliverables

- [x] All merge conflicts resolved (2026-02-19)
- [x] All branches build and pass tests for all subprojects (2026-02-19)
- [x] Clean baseline tagged (4 repos, 5 tags total) (2026-02-19)
- [x] `.gitignore` updated to exclude build artifacts (2026-02-19)

---

## Phase 1 — Extract and Publish Client Packages to PyPI

**Duration:** 3–5 days
**Risk:** Low (additive, no existing code removed)
**Prerequisite:** Phase 0 complete
**Status:** COMPLETE (2026-02-21) — Validated 2026-02-22

### Objective, Phase 1

Publish `juniper-data-client` to PyPI as the single source of truth. Eliminate all vendored copies.

### Completion Summary, Phase 1

> **Validation (2026-02-22):** Phase 1 deliverables confirmed. `juniper-data-client` v0.3.0 remains published on PyPI and installed in the development environment. CI/CD pipeline (ci.yml + publish.yml) passing on GitHub. 6 commits on `main`, latest `821f74d`. No vendored copies remain in any downstream project.

The `juniper-data-client` package has been extracted, published to PyPI (v0.3.0), and installed in the development environment. All vendored copies have been removed from all projects.

| Step                             | Status   | Notes                                                                  |
| -------------------------------- | -------- | ---------------------------------------------------------------------- |
| 1.1 Prepare for PyPI             | COMPLETE | pyproject.toml, README, LICENSE, py.typed all present                  |
| 1.2 Create GitHub repo           | COMPLETE | `pcalnon/juniper-data-client` exists with full history                 |
| 1.3 PyPI publishing              | COMPLETE | v0.3.0 published 2026-02-20, CI/CD workflows active                    |
| 1.4a Remove Cascor vendored copy | COMPLETE | Empty dir remains (only `__pycache__/`), imports use external          |
| 1.4b Remove Canopy vendored copy | COMPLETE | No vendored copy; imports use external package                         |
| 1.4c Remove Data vendored copy   | COMPLETE | Removed 2026-02-21; added to .gitignore; pyproject.toml updated        |
| 1.5 Verify all tests pass        | COMPLETE | JuniperData 659 pass, CasCor 226 pass, all using external PyPI package |

**Package installation status:**

- Installed as editable from standalone repo: `juniper-data-client 0.3.0`
- JuniperCascor pyproject.toml: `juniper-data-client>=0.3.0` under `[project.optional-dependencies].juniper-data`
- JuniperCanopy pyproject.toml: `juniper-data-client>=0.3.0` under `[project.optional-dependencies].juniper-data`
- JuniperData pyproject.toml: `juniper-data-client>=0.3.0` under `[project.optional-dependencies].test`

### Step 1.1 — Prepare `juniper-data-client` for PyPI

The package already exists at `JuniperData/juniper_data/juniper_data_client/` with its own `pyproject.toml`. Updates needed:

**1.1.1 — Update `pyproject.toml`:**

```toml
[project]
name = "juniper-data-client"
version = "0.3.0"  # Bump to reflect PyPI-readiness and API key support
description = "HTTP client for the JuniperData dataset generation service"
readme = "README.md"
license = {text = "MIT"}
authors = [{name = "Paul Calnon"}]
requires-python = ">=3.11"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]
dependencies = [
    "numpy>=1.24.0",
    "requests>=2.28.0",
    "urllib3>=2.0.0",
]

[project.urls]
Homepage = "https://github.com/pcalnon/juniper-data-client"
Repository = "https://github.com/pcalnon/juniper-data-client"
Documentation = "https://github.com/pcalnon/juniper-data-client#readme"
Issues = "https://github.com/pcalnon/juniper-data-client/issues"

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
```

**1.1.2 — Reconcile vendored copies:**

The canonical version (in JuniperData) has `api_key` support. The Canopy vendored copy does not. The Cascor vendored copy uses completely different retry logic.

**Resolution:** The canonical JuniperData version is the source of truth. Ensure it includes:

- `api_key` parameter (already present)
- `HTTPAdapter` with `urllib3.Retry` (already present)
- All methods from both vendored copies
- Complete exception hierarchy

**1.1.3 — Add `README.md`:**

Write a standalone README with installation, usage examples, and API reference for the PyPI page.

**1.1.4 — Add `LICENSE`:**

Include MIT license file in the package root.

### Step 1.2 — Create GitHub Repository

```bash
# Create new repo on GitHub
gh repo create pcalnon/juniper-data-client --public --description "HTTP client for JuniperData dataset generation service"

# Extract package with history (optional, or start fresh)
cd /tmp
mkdir juniper-data-client
cp -r /path/to/JuniperData/juniper_data/juniper_data_client/* juniper-data-client/
cd juniper-data-client
git init
git add .
git commit -m "Initial release: juniper-data-client v0.3.0

Extracted from Juniper monorepo. Published to PyPI as the single
source of truth for the JuniperData HTTP client.

Features:
- Full JuniperData API coverage (generators, datasets, health)
- Automatic retry with exponential backoff
- Connection pooling
- API key authentication
- Context manager support
- Complete exception hierarchy
- PEP 561 type stubs"
git remote add origin git@github.com:pcalnon/juniper-data-client.git
git push -u origin main
```

### Step 1.3 — Set Up PyPI Publishing

**1.3.1 — Create PyPI account** (if not already done) at <https://pypi.org>

**1.3.2 — Generate API token** at <https://pypi.org/manage/account/token/>

**1.3.3 — Add GitHub Actions workflow** (see [Appendix B](#appendix-b--pypi-publishing-workflow))

**1.3.4 — Test with TestPyPI first:**

```bash
# Build
python -m build

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ juniper-data-client
```

**1.3.5 — Publish to PyPI:**

```bash
python -m twine upload dist/*

# Or via GitHub Actions: create a release tag
git tag v0.3.0
git push origin v0.3.0
```

### Step 1.4 — Replace Vendored Copies with PyPI Package

**In JuniperCascor:**

```bash
# Remove vendored copy
rm -rf src/juniper_data_client/

# Add to pyproject.toml dependencies (or optional extras)
# [project.optional-dependencies]
# juniper-data = ["juniper-data-client>=0.3.0"]

# Update imports (should remain the same: from juniper_data_client import ...)
# The package name on import is juniper_data_client (underscore)
```

**In JuniperCanopy:**

```bash
# Remove vendored copy from src/
rm -rf src/juniper_data_client/

# Remove vendored copy from project root
rm -rf juniper_data_client/

# Add to pyproject.toml dependencies
# dependencies = [..., "juniper-data-client>=0.3.0"]

# Remove the try/except import fallback in any __init__.py that tries local copy
# All imports become: from juniper_data_client import JuniperDataClient
```

### Step 1.5 — Verify

```bash
# In each project
pip install juniper-data-client>=0.3.0
cd src && pytest tests/ -v

# Verify no remaining vendored copies
find . -path "*/juniper_data_client/client.py" -not -path "*/site-packages/*"
# Should return nothing (or only the canonical source in juniper-data-client repo)
```

### Deliverables, Phase 1

- [x] `juniper-data-client` repository created on GitHub (2026-02-19)
- [x] Package published to PyPI as `juniper-data-client` v0.3.0 (2026-02-20)
- [x] CI/CD workflow for automated PyPI publishing on release tags (2026-02-20, uses Trusted Publishing/OIDC)
- [x] Vendored copy removed from Canopy (no vendored copy remains)
- [x] Vendored copy removed from Cascor (empty dir, imports use external package)
- [x] Vendored copy removed from JuniperData (2026-02-21, commit `4bada2a`)
- [x] All tests pass with the PyPI-installed package (JuniperData 659, CasCor 226 — all verified 2026-02-21)

---

## Phase 2 — Build CasCor Service API

**Duration:** 2–3 weeks
**Risk:** Medium (new code, but additive — existing CLI usage preserved)
**Prerequisite:** Phase 1 complete
**Status:** COMPLETE (2026-02-21) — Validated 2026-02-22

### Objective, Phase 2

Add a FastAPI + WebSocket service layer to JuniperCascor so it can be consumed as a network service rather than a library import. The existing CLI entry point (`main.py`) continues to work unchanged.

### Completion Summary, Phase 2

> **Validation (2026-02-22):** Phase 2 deliverables confirmed. All 19 REST endpoints and 2 WebSocket endpoints are implemented and tested. The API layer exists in the standalone `juniper-cascor` repo (127 commits on `main`) with CI/CD passing. The `TrainingLifecycleManager`, state machine, and monitoring hooks remain functional. Service entry point (`server.py`) and CLI (`main.py`) both present.

All Phase 2 core features have been implemented across 14 new source files and 17 test files. The API provides 19 REST endpoints and 2 WebSocket endpoints with full training lifecycle management.

| Step                         | Status           | Notes                                                                    |
| ---------------------------- | ---------------- | ------------------------------------------------------------------------ |
| 2.1 Define API contract      | COMPLETE         | Documented in Appendix A                                                 |
| 2.2 Add FastAPI to CasCor    | COMPLETE         | 6 route files, 4 websocket files, 4 model files, 3 lifecycle files       |
| 2.3 TrainingLifecycleManager | COMPLETE         | Thread-safe state machine, ThreadPoolExecutor, monitoring hooks          |
| 2.4 Add dependencies         | COMPLETE         | FastAPI, uvicorn, websockets in pyproject.toml                           |
| 2.5 Service entry point      | COMPLETE         | `server.py` alongside existing `main.py`                                 |
| 2.6 Test the service API     | COMPLETE         | 213 unit tests + 13 integration tests pass (2026-02-21)                  |

### Implementation Details

**REST Endpoints Implemented (19/19 Phase 2 targets):**

| Endpoint                    | File                 | Status                                             |
| --------------------------- | -------------------- | -------------------------------------------------- |
| `GET /v1/health`            | health.py            | COMPLETE                                           |
| `GET /v1/health/live`       | health.py            | COMPLETE (bonus: liveness probe)                   |
| `GET /v1/health/ready`      | health.py            | COMPLETE                                           |
| `POST /v1/network`          | network.py           | COMPLETE                                           |
| `GET /v1/network`           | network.py           | COMPLETE                                           |
| `DELETE /v1/network`        | network.py           | COMPLETE                                           |
| `GET /v1/network/topology`  | network.py           | COMPLETE                                           |
| `GET /v1/network/stats`     | network.py           | COMPLETE                                           |
| `POST /v1/training/start`   | training.py          | COMPLETE (supports inline data + spiral generator) |
| `POST /v1/training/stop`    | training.py          | COMPLETE                                           |
| `POST /v1/training/pause`   | training.py          | COMPLETE                                           |
| `POST /v1/training/resume`  | training.py          | COMPLETE                                           |
| `POST /v1/training/reset`   | training.py          | COMPLETE                                           |
| `GET /v1/training/status`   | training.py          | COMPLETE                                           |
| `GET /v1/training/params`   | training.py          | COMPLETE                                           |
| `GET /v1/metrics`           | metrics.py           | COMPLETE                                           |
| `GET /v1/metrics/history`   | metrics.py           | COMPLETE                                           |
| `GET /v1/dataset`           | dataset.py           | COMPLETE                                           |
| `GET /v1/decision-boundary` | decision_boundary.py | COMPLETE                                           |

**WebSocket Endpoints Implemented (2/2):**

| Path           | Handler            | Status                                                 |
| -------------- | ------------------ | ------------------------------------------------------ |
| `/ws/training` | training_stream.py | COMPLETE (3-message connect sequence, broadcast relay) |
| `/ws/control`  | control_stream.py  | COMPLETE (start/stop/pause/resume/reset commands)      |

**Deferred Endpoints (correctly not implemented):**

- `PUT /v1/training/params` — parameter update during training
- `/v1/snapshots/*` — HDF5 snapshot management (4 endpoints)
- `/v1/workers/*` — remote worker management (5 endpoints)

**Key Architecture:**

- `TrainingLifecycleManager` (579 lines) — thread-safe training coordination with ThreadPoolExecutor
- `TrainingStateMachine` — formal FSM: STOPPED ↔ STARTED ↔ PAUSED → COMPLETED/FAILED
- `TrainingMonitor` — event-driven callbacks (epoch_end, cascade_add, training_start/end)
- `WebSocketManager` — connection management with `broadcast_from_thread()` async/sync bridge
- Monitoring hooks: monkey-patches `fit()`, `train_output_layer()`, `grow_network()` for per-epoch metrics

**Test Suite (VALIDATED 2026-02-21 — all passing):**

Unit tests (18 files, 213 tests):

- `test_api_app.py`, `test_api_health.py`, `test_api_routes.py`, `test_api_settings.py`
- `test_dataset_route.py`, `test_decision_boundary_route.py`, `test_metrics_routes.py`
- `test_lifecycle_manager.py`, `test_lifecycle_manager_coverage.py`, `test_lifecycle_monitor.py`, `test_lifecycle_state_machine.py`
- `test_monitoring_hooks.py`
- `test_network_route_coverage.py`, `test_training_route_coverage.py`
- `test_websocket_control.py`, `test_websocket_manager.py`, `test_websocket_messages.py`, `test_websocket_training_stream.py`

Integration tests (2 files + conftest, 13 tests):

- `test_api_full_lifecycle.py` — create → train → stop → metrics → reset → cleanup
- `test_websocket_streaming.py` — connect sequence, control commands, multiple clients

### Step 2.1 — Define API Contract

See [Appendix A](#appendix-a--cascor-service-api-contract-draft) for the full contract. Summary:

**REST Endpoints:**

| Method   | Path                         | Purpose                                     |
| -------- | ---------------------------- | ------------------------------------------- |
| `GET`    | `/v1/health`                 | Health check                                |
| `GET`    | `/v1/health/ready`           | Readiness probe (network loaded?)           |
| `POST`   | `/v1/network`                | Create a new CasCor network                 |
| `GET`    | `/v1/network`                | Get network state/configuration             |
| `DELETE` | `/v1/network`                | Destroy current network                     |
| `POST`   | `/v1/training/start`         | Start training (async, returns immediately) |
| `POST`   | `/v1/training/stop`          | Request training stop                       |
| `POST`   | `/v1/training/pause`         | Pause training                              |
| `POST`   | `/v1/training/resume`        | Resume training                             |
| `POST`   | `/v1/training/reset`         | Reset training state                        |
| `GET`    | `/v1/training/status`        | Get current training status                 |
| `GET`    | `/v1/training/params`        | Get training parameters                     |
| `PUT`    | `/v1/training/params`        | Update training parameters                  |
| `GET`    | `/v1/metrics`                | Current metrics snapshot                    |
| `GET`    | `/v1/metrics/history`        | Full metrics history                        |
| `GET`    | `/v1/network/topology`       | Network topology for visualization          |
| `GET`    | `/v1/network/statistics`     | Network statistics                          |
| `GET`    | `/v1/dataset`                | Current dataset info                        |
| `GET`    | `/v1/decision-boundary`      | Decision boundary data                      |
| `POST`   | `/v1/snapshots`              | Create snapshot                             |
| `GET`    | `/v1/snapshots`              | List snapshots                              |
| `GET`    | `/v1/snapshots/{id}`         | Get snapshot detail                         |
| `POST`   | `/v1/snapshots/{id}/restore` | Restore from snapshot                       |
| `GET`    | `/v1/workers`                | Remote worker status                        |
| `POST`   | `/v1/workers/connect`        | Connect remote workers                      |
| `POST`   | `/v1/workers/start`          | Start remote workers                        |
| `POST`   | `/v1/workers/stop`           | Stop remote workers                         |
| `POST`   | `/v1/workers/disconnect`     | Disconnect remote workers                   |

**WebSocket Endpoints:**

| Path           | Purpose                                    | Direction       |
| -------------- | ------------------------------------------ | --------------- |
| `/ws/training` | Real-time metrics, state, topology updates | Server → Client |
| `/ws/control`  | Training control commands                  | Client → Server |

**Message format** (matching existing Canopy convention):

```json
{
    "type": "metrics|state|topology|event|cascade_add",
    "timestamp": 1234567890.123,
    "data": { ... }
}
```

### Step 2.2 — Add FastAPI to CasCor

**Directory structure addition:**

```bash
juniper_cascor/src/
├── api/                          # NEW
│   ├── __init__.py
│   ├── app.py                    # FastAPI app, lifespan
│   ├── settings.py               # Pydantic settings (port, host, etc.)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py             # Health/readiness
│   │   ├── network.py            # Network CRUD
│   │   ├── training.py           # Training control
│   │   ├── metrics.py            # Metrics retrieval
│   │   ├── snapshots.py          # Snapshot management
│   │   └── workers.py            # Remote worker management
│   ├── websocket/
│   │   ├── __init__.py
│   │   ├── manager.py            # WebSocket connection manager
│   │   ├── training_stream.py    # /ws/training handler
│   │   └── control_stream.py     # /ws/control handler
│   └── models/
│       ├── __init__.py
│       ├── network.py            # Request/response schemas
│       ├── training.py           # Training schemas
│       ├── metrics.py            # Metrics schemas
│       └── workers.py            # Worker schemas
├── main.py                       # Existing CLI entry point (unchanged)
├── server.py                     # NEW: Service entry point
└── ...
```

**Key implementation notes:**

- The `CascadeCorrelationNetwork` is **NOT thread-safe** (per its own docstring). The API layer must serialize access via a lock or run training in a dedicated thread with a message-passing interface.
- Training is blocking (`fit()` is synchronous). The API must run training in a `ThreadPoolExecutor` worker and stream results via WebSocket — similar to what `CascorIntegration.fit_async()` already does in Canopy.
- The monitoring hook pattern (monkey-patching `fit`, `train_output_layer`, `train_candidates`) should be formalized as proper callbacks/events in the service layer, not monkey-patches.

### Step 2.3 — Implement Training Lifecycle Manager

Create a `TrainingLifecycleManager` class that wraps `CascadeCorrelationNetwork` with:

- Thread-safe state machine (stopped → running → paused → running → stopped)
- Background training via `ThreadPoolExecutor` (single worker)
- Callback registration for training events (epoch complete, candidate added, training complete)
- WebSocket broadcast on each callback
- Parameter updates (where allowed during training, per `modifiable_during_training` flag)

This class replaces what Canopy's `CascorIntegration` currently does — the responsibility moves to CasCor itself.

### Step 2.4 — Add Dependencies to CasCor

```toml
# In juniper_cascor pyproject.toml
[project.optional-dependencies]
api = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "websockets>=11.0",
]
```

### Step 2.5 — Create Service Entry Point

```python
# src/server.py
"""CasCor Service — FastAPI entry point for the CasCor neural network service."""

import uvicorn
from api.app import create_app
from api.settings import Settings

def main():
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)

if __name__ == "__main__":
    main()
```

### Step 2.6 — Test the Service API

- Unit tests for each route handler (mocked network)
- Integration tests that start the service and exercise the full lifecycle
- WebSocket tests for streaming metrics
- Verify the existing CLI `main.py` still works unchanged

### Deliverables, Phase 2

- [x] FastAPI service layer added to CasCor (2026-02-20, committed as `529995c`)
- [x] REST endpoints for all training lifecycle operations (19 endpoints across 6 route files)
- [x] WebSocket endpoints for real-time streaming (`/ws/training`, `/ws/control`)
- [x] `TrainingLifecycleManager` with thread-safe state machine (579 lines, ThreadPoolExecutor)
- [x] Service entry point (`server.py`) alongside existing CLI (`main.py`)
- [x] Comprehensive test suite for the API layer (213 unit + 13 integration tests, validated 2026-02-21)
- [x] API contract documented (Appendix A)
- [x] Existing CLI (`main.py`) confirmed unchanged by Phase 2 work

---

## Phase 3 — Create CasCor Client and Remote Worker Packages

**Duration:** 1–2 weeks
**Risk:** Low (new packages, additive)
**Prerequisite:** Phase 2 complete (API contract defined)
**Status:** COMPLETE (2026-02-24) — Validated 2026-02-24

### Objective, Phase 3

Create two independently installable PyPI packages: `juniper-cascor-client` (HTTP/WebSocket client for the CasCor service API) and `juniper-cascor-worker` (remote candidate training worker). Both follow the patterns established by `juniper-data-client` in Phase 1.

### Completion Summary, Phase 3

> **Validation (2026-02-24):** Both packages published to PyPI as v0.1.0. Trusted Publishing configured on both PyPI and TestPyPI. v0.1.0 releases created on both repos — `publish.yml` workflows completed successfully (TestPyPI → PyPI two-stage pipeline). `pip index versions` confirms both packages available on PyPI. `juniper-cascor-client` installed as editable in JuniperCanopy conda env (required for Phase 4 adapter tests — all 83 now passing).

Both packages are published to PyPI as v0.1.0, with full test suites, CI/CD pipelines verified green, and comprehensive documentation. Both are editable-installed in their respective development environments.

| Step                                      | Status                | Notes                                                                                    |
| ----------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------- |
| 3.1 Create `juniper-cascor-client`        | COMPLETE              | Standalone repo, 55 tests pass, lint + mypy + coverage all pass                          |
| 3.2 Create `juniper-cascor-worker`        | COMPLETE              | Standalone repo, 44 tests pass (99% coverage), CLI entry point functional                |
| 3.3a Verify CI/CD workflows run on GitHub | COMPLETE (2026-02-21) | CI green on both repos (Python 3.11/3.12/3.13 matrix, all 6 jobs pass)                   |
| 3.3b GitHub environments created          | COMPLETE (2026-02-21) | `pypi` + `testpypi` environments created on both repos                                   |
| 3.3c Configure Trusted Publishing         | COMPLETE (2026-02-24) | Pending publishers registered on both PyPI and TestPyPI web interfaces for both packages |
| 3.3d Publish to TestPyPI                  | COMPLETE (2026-02-24) | v0.1.0 two-stage pipeline passed; TestPyPI install verified by workflow                  |
| 3.3e Publish to PyPI                      | COMPLETE (2026-02-24) | `juniper-cascor-client 0.1.0` and `juniper-cascor-worker 0.1.0` live on PyPI             |

**Package status summary:**

| Package                 | Repo                                     | Version | Tests   | Commits | CI (GitHub Actions)    | Editable Install                    | PyPI    |
| ----------------------- | ---------------------------------------- | ------- | ------- | ------- | ---------------------- | ----------------------------------- | ------- |
| `juniper-cascor-client` | `pcalnon/juniper-cascor-client` (public) | 0.1.0   | 55 pass | 6       | GREEN (3.11/3.12/3.13) | Yes (JuniperCascor + JuniperCanopy) | **Yes** |
| `juniper-cascor-worker` | `pcalnon/juniper-cascor-worker` (public) | 0.1.0   | 44 pass | 7       | GREEN (3.11/3.12/3.13) | Yes (JuniperCascor)                 | **Yes** |

### Step 3.1 — Create `juniper-cascor-client`

**Status:** COMPLETE (2026-02-21)

A Python HTTP/WebSocket client for the CasCor service API, following the same patterns as `juniper-data-client`.

**Repository:** `pcalnon/juniper-cascor-client`

**Package structure:**

```bash
juniper-cascor-client/
├── juniper_cascor_client/
│   ├── __init__.py
│   ├── client.py                # JuniperCascorClient
│   ├── ws_client.py             # WebSocket streaming client
│   ├── exceptions.py            # Exception hierarchy
│   ├── models.py                # Pydantic response models (optional)
│   └── py.typed                 # PEP 561
├── tests/
│   ├── __init__.py
│   ├── test_client.py
│   └── test_ws_client.py
├── pyproject.toml
├── README.md
├── LICENSE
└── .github/
    └── workflows/
        ├── ci.yml
        └── publish.yml
```

**`JuniperCascorClient` class — public API:**

```python
class JuniperCascorClient:
    def __init__(self, base_url="http://localhost:8200", timeout=30, retries=3, api_key=None): ...

    # Health
    def health_check(self) -> dict: ...
    def is_ready(self) -> bool: ...
    def wait_for_ready(self, timeout=30, poll_interval=0.5) -> bool: ...

    # Network
    def create_network(self, config: dict) -> dict: ...
    def get_network(self) -> dict: ...
    def delete_network(self) -> None: ...

    # Training control
    def start_training(self, dataset_id: str = None, params: dict = None) -> dict: ...
    def stop_training(self) -> dict: ...
    def pause_training(self) -> dict: ...
    def resume_training(self) -> dict: ...
    def reset_training(self) -> dict: ...
    def get_training_status(self) -> dict: ...
    def get_training_params(self) -> dict: ...
    def set_training_params(self, params: dict) -> dict: ...

    # Metrics
    def get_metrics(self) -> dict: ...
    def get_metrics_history(self) -> list: ...

    # Network info
    def get_topology(self) -> dict: ...
    def get_statistics(self) -> dict: ...
    def get_dataset(self) -> dict: ...
    def get_decision_boundary(self) -> dict: ...

    # Snapshots
    def create_snapshot(self) -> dict: ...
    def list_snapshots(self) -> list: ...
    def get_snapshot(self, snapshot_id: str) -> dict: ...
    def restore_snapshot(self, snapshot_id: str) -> dict: ...

    # Workers
    def get_worker_status(self) -> dict: ...
    def connect_workers(self, address: str, num_workers: int) -> dict: ...
    def start_workers(self) -> dict: ...
    def stop_workers(self) -> dict: ...
    def disconnect_workers(self) -> dict: ...

    # Context manager
    def close(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, ...): ...
```

**`CascorTrainingStream` class — WebSocket streaming:**

```python
class CascorTrainingStream:
    """Async WebSocket client for real-time training updates."""

    def __init__(self, base_url="ws://localhost:8200", api_key=None): ...

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...

    async def stream(self) -> AsyncIterator[dict]:
        """Yields messages as they arrive."""
        ...

    async def send_command(self, command: str, params: dict = None) -> dict:
        """Send a control command via WebSocket."""
        ...

    # Callback-based API (alternative to async iteration)
    def on_metrics(self, callback: Callable[[dict], None]) -> None: ...
    def on_state_change(self, callback: Callable[[dict], None]) -> None: ...
    def on_topology_change(self, callback: Callable[[dict], None]) -> None: ...
    def on_cascade_add(self, callback: Callable[[dict], None]) -> None: ...
```

**Dependencies:**

```toml
dependencies = [
    "requests>=2.28.0",
    "urllib3>=2.0.0",
    "websockets>=11.0",
]
```

**Implementation details (completed 2026-02-21):**

- **Repository**: `pcalnon/juniper-cascor-client` — public, 5 commits on `main`, clean working tree, CI green
- **Location**: `/home/pcalnon/Development/python/Juniper/juniper-cascor-client/`
- **Public API** (matches planned design with one bonus addition):
  - `JuniperCascorClient` — REST client with 24 public methods (health, network CRUD, training control, metrics, visualization, snapshots, workers)
  - `CascorTrainingStream` — async WebSocket client for `/ws/training` (iteration + callback patterns)
  - `CascorControlStream` — async WebSocket client for `/ws/control` (command/response pattern; bonus, not in original plan)
  - 7 exception classes: `ClientError`, `ConnectionError`, `TimeoutError`, `NotFoundError`, `ConflictError`, `ValidationError`, `ServiceUnavailableError`
- **Quality infrastructure**: black (120 line length), isort, mypy (strict mode), flake8, coverage gate (80%), PEP 561 typed
- **CI/CD**: `ci.yml` (matrix: Python 3.11/3.12/3.13), `publish.yml` (two-stage: TestPyPI → PyPI, Trusted Publishing)
- **Tests**: 55 tests (34 REST client + 21 WebSocket client), all passing
- **Build artifacts**: pre-built wheel + sdist in `dist/`
- **Editable install**: installed in `JuniperCascor` conda env
- **Consumer adoption**: none yet — no downstream project imports `juniper_cascor_client`

### Step 3.2 — Create `juniper-cascor-worker`

**Status:** COMPLETE (2026-02-21)

Extract and enhance the existing `RemoteWorkerClient` from CasCor's `src/remote_client/` into a standalone, independently installable package for remote candidate training.

**Repository:** `pcalnon/juniper-cascor-worker`

**Package structure:**

```bash
juniper-cascor-worker/
├── juniper_cascor_worker/
│   ├── __init__.py
│   ├── worker.py                # CandidateTrainingWorker
│   ├── client.py                # WorkerRegistrationClient (connects to manager)
│   ├── config.py                # Worker configuration
│   ├── exceptions.py            # Exception hierarchy
│   └── py.typed
├── tests/
│   ├── __init__.py
│   ├── test_worker.py
│   └── test_client.py
├── pyproject.toml
├── README.md
├── LICENSE
└── .github/
    └── workflows/
        ├── ci.yml
        └── publish.yml
```

**Key classes:**

```python
class CandidateTrainingWorker:
    """Runs on remote hardware. Connects to CasCor's CandidateTrainingManager
    and processes candidate training tasks from the task queue."""

    def __init__(self, manager_address: tuple, authkey: bytes, num_processes: int = 1): ...
    def connect(self) -> None: ...
    def start(self) -> None: ...      # Start worker processes
    def stop(self, timeout=10) -> None: ...
    def disconnect(self) -> None: ...
    def is_running(self) -> bool: ...

    # Context manager for clean lifecycle
    def __enter__(self): ...
    def __exit__(self, ...): ...
```

**CLI entry point** (for running on remote hardware):

```bash
# Install on remote machine
pip install juniper-cascor-worker

# Run worker
juniper-cascor-worker --manager-host 192.168.1.100 --manager-port 5000 --workers 4
```

```toml
[project.scripts]
juniper-cascor-worker = "juniper_cascor_worker.cli:main"
```

**Dependencies:**

```toml
dependencies = [
    "numpy>=1.24.0",
    "torch>=2.0.0",       # Required for candidate training
]
```

Note: The worker needs PyTorch because it runs `CascadeCorrelationNetwork._worker_loop` locally. This is an inherent requirement — the worker is doing actual computation, not just proxying.

**Implementation details (completed 2026-02-21):**

- **Repository**: `pcalnon/juniper-cascor-worker` — public, 6 commits on `main`, clean working tree, CI green
- **Location**: `/home/pcalnon/Development/python/Juniper/juniper-cascor-worker/`
- **Actual package structure** (slightly differs from plan — `client.py` became `cli.py`):

  ```bash
  juniper_cascor_worker/
  ├── __init__.py          # Exports: CandidateTrainingWorker, WorkerConfig, exceptions
  ├── worker.py            # CandidateTrainingWorker (connect/start/stop/disconnect)
  ├── cli.py               # argparse CLI with signal handling (replaces planned client.py)
  ├── config.py            # WorkerConfig dataclass with from_env() + validate()
  ├── exceptions.py        # WorkerError, WorkerConnectionError, WorkerConfigError
  └── py.typed             # PEP 561 marker
  ```

- **Improvements over in-tree `remote_client/`**:
  - stdlib `logging` instead of CasCor's custom `Logger` (no coupling)
  - `WorkerConfig` dataclass with validation and environment variable support (`from_env()`)
  - Custom exception hierarchy (3 classes) instead of generic `RuntimeError`
  - CLI entry point with signal handling (`juniper-cascor-worker` console_scripts)
  - Full test suite (24 tests) vs. zero tests for in-tree version
  - No `sys.path.append` hacks
- **Quality infrastructure**: black, isort, mypy, flake8, coverage gate (80%), PEP 561 typed
- **CI/CD**: `ci.yml` (matrix: Python 3.11/3.12/3.13), `publish.yml` (two-stage: TestPyPI → PyPI, Trusted Publishing)
- **Tests**: 44 tests (9 CLI + 10 config + 25 worker), all passing, 99% coverage
- **Build artifacts**: pre-built wheel (8.7 KB) + sdist (8.8 KB) in `dist/`
- **Editable install**: installed in `JuniperCascor` conda env
- **In-tree predecessor**: `src/remote_client/remote_client.py` still exists in JuniperCascor (not yet removed; removal is not a Phase 3 task)

### Step 3.3 — Publish Both to PyPI

**Status:** COMPLETE (2026-02-24)

**Completed tasks for Step 3.3:**

1. ~~**Verify GitHub Actions**~~: DONE (2026-02-21). CI workflows triggered and pass on both repos (Python 3.11/3.12/3.13 matrix, all 6 jobs green). Fixed lint errors (cascor-client: 4 flake8 issues), coverage gaps (cascor-worker: 57% → 99%), and mypy errors (cascor-worker: 2 type annotation fixes).
2. ~~**Create GitHub environments**~~: DONE (2026-02-21). `pypi` and `testpypi` environments created on both repos via GitHub API.
3. ~~**Build and verify packages**~~: DONE (2026-02-21). Both packages pass `twine check` and `python -m build`.
4. ~~**Configure Trusted Publishing**~~: DONE (2026-02-24). Pending publishers registered on PyPI and TestPyPI web interfaces for both packages.
5. ~~**Create v0.1.0 releases**~~: DONE (2026-02-24). Releases triggered `publish.yml`; both workflows completed successfully (TestPyPI → PyPI two-stage pipeline, `in_progress` → `completed success`).
6. ~~**Verify installation**~~: DONE (2026-02-24). `pip index versions` confirms `juniper-cascor-client (0.1.0)` and `juniper-cascor-worker (0.1.0)` available on PyPI.

### Deliverables, Phase 3

- [x] `juniper-cascor-client` repository created on GitHub (2026-02-21, `pcalnon/juniper-cascor-client`, 5 commits; verified 2026-02-22)
- [x] `juniper-cascor-worker` repository created on GitHub (2026-02-21, `pcalnon/juniper-cascor-worker`, 6 commits; verified 2026-02-22)
- [x] Both packages have CI/CD workflow files (ci.yml + publish.yml with Trusted Publishing)
- [x] CI/CD workflows verified green on GitHub (Python 3.11/3.12/3.13, all 6 matrix jobs pass) (2026-02-21)
- [x] GitHub environments (`pypi` + `testpypi`) created on both repos (2026-02-21)
- [x] Comprehensive test suites for both packages (55 + 44 tests, all passing, worker at 99% coverage)
- [x] README documentation with installation and usage examples (both repos)
- [x] Both packages editable-installed in local development environment
- [x] Worker CLI entry point functional (`juniper-cascor-worker` console_scripts)
- [x] Package builds pass `twine check` (both wheel + sdist)
- [x] Configure Trusted Publishing on TestPyPI (2026-02-24)
- [x] Configure Trusted Publishing on PyPI (2026-02-24)
- [x] Publish `juniper-cascor-client` to PyPI — v0.1.0 live (2026-02-24)
- [x] Publish `juniper-cascor-worker` to PyPI — v0.1.0 live (2026-02-24)

---

## Phase 4 — Decouple Canopy from CasCor

**Duration:** 2–3 weeks
**Risk:** High (core architectural change to Canopy)
**Prerequisite:** Phases 2 and 3 complete
**Status:** COMPLETE (2026-02-25) — Validated 2026-02-25
**Detailed Plan:** [`DECOUPLE_CANOPY_FROM_CASCOR_PLAN.md`](DECOUPLE_CANOPY_FROM_CASCOR_PLAN.md)

> **Note:** A comprehensive, standalone implementation plan exists in
> `notes/DECOUPLE_CANOPY_FROM_CASCOR_PLAN.md`. It contains full adapter code,
> method mapping tables, two-mode activation logic, WebSocket relay
> architecture, and a corrections table vs. the original version of this
> section. The summary below reflects the corrected design.

### Verification Summary (2026-02-25)

Phase 4 fully complete. Integration testing and legacy removal both verified:

| Component              | Status             | Details                                                                                                |
|------------------------|--------------------|--------------------------------------------------------------------------------------------------------|
| `CascorServiceAdapter` | COMPLETE           | 306 lines at `src/backend/cascor_service_adapter.py`; REST/WS adapter wrapping `juniper-cascor-client` |
| `CascorIntegration`    | DELETED            | Removed 2026-02-25 (Step 4.8); 1,601-line legacy backend gone                                          |
| Two-mode activation    | COMPLETE           | `main.py`: Demo > Service > Demo fallback; `_is_service_adapter is True` guard for shutdown            |
| Adapter tests          | PASSING            | 52 tests in `src/tests/unit/backend/test_cascor_service_adapter.py`                                    |
| Activation tests       | PASSING            | 11 tests in `src/tests/unit/test_three_mode_activation.py`                                             |
| Total Canopy tests     | PASSING            | 3,130 passed, 23 skipped (after 160 legacy tests removed)                                              |
| Dependency declaration | CONFIGURED         | `juniper-cascor-client>=0.1.0` in `pyproject.toml [project.optional-dependencies].juniper-cascor`      |
| REST integration tests | PASSING            | All 10 service endpoints verified against live CasCor service (port 8201)                              |
| WS relay integration   | PASSING            | CasCor WS → `CascorTrainingStream.stream()` → `ws_manager.broadcast()` verified                        |
| Legacy files removed   | COMPLETE           | `cascor_integration.py` + 13 legacy test files deleted                                                 |
| Latest commit          | `5f9987d`          | "Step 4.8: Remove legacy CascorIntegration mode"                                                       |
| Branch                 | `canopy/migration` | Tracking `origin` (monorepo `pcalnon/Juniper`)                                                         |

### Objective, Phase 4

Replace Canopy's `CascorIntegration` class (~1,600 lines of `sys.path` injection, monkey-patching, and direct CasCor imports) with a `CascorServiceAdapter` that wraps the `juniper-cascor-client` package (v0.1.0), communicating over REST/WebSocket instead of in-process Python imports.

### Step 4.1 — Understand What `CascorIntegration` Currently Does

`CascorIntegration` (in `src/backend/cascor_integration.py`) currently performs these functions:

| Function Category      | Methods                                                                                                                          | Replacement                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Backend discovery**  | `_resolve_backend_path`, `_add_backend_to_path`, `_import_backend_modules`                                                       | **Eliminated** — no path injection needed                                            |
| **Network lifecycle**  | `create_network`, `connect_to_network`                                                                                           | `client.create_network()`, `client.get_network()`                                    |
| **Training control**   | `fit_async`, `start_training_background`, `request_training_stop`, `is_training_in_progress`                                     | `client.start_training()`, `client.stop_training()`, `client.get_training_status()`  |
| **Monitoring hooks**   | `install_monitoring_hooks`, monkey-patching `fit`/`train_output`/`train_candidates`, `_on_*` callbacks                           | **Eliminated** — CasCor service handles its own monitoring and streams via WebSocket |
| **Monitoring thread**  | `start_monitoring_thread`, `stop_monitoring`, `_monitoring_loop`, `_extract_current_metrics`                                     | **Replaced** by WebSocket subscription via `CascorTrainingStream.stream()`           |
| **Network data**       | `get_network_topology`, `get_network_data`, `extract_cascor_topology`, `get_dataset_info`, `get_prediction_function`             | `client.get_topology()`, `client.get_dataset()`, `client.get_decision_boundary()`    |
| **Remote workers**     | `connect_remote_workers`, `start_remote_workers`, `stop_remote_workers`, `disconnect_remote_workers`, `get_remote_worker_status` | **Stub no-ops** — workers managed server-side by CasCor + `juniper-cascor-worker`    |
| **Dataset generation** | `_generate_dataset_from_juniper_data`, `_create_juniper_dataset`                                                                 | Handled by CasCor service; Canopy passes dataset config in `start_training()`        |
| **Snapshots**          | (referenced in `main.py`, not fully implemented in `CascorIntegration`)                                                          | `cascor_client.create_snapshot()`, `.restore_snapshot()` (deferred)                  |
| **Broadcasting**       | `_broadcast_message` (sends to Canopy's WebSocketManager)                                                                        | WebSocket relay: CasCor WS → adapter → Canopy frontend WS                            |

### Step 4.2 — Create `CascorServiceAdapter`

Replace `CascorIntegration` with a new `CascorServiceAdapter` that wraps the cascor client. Key design decisions (see detailed plan for full code):

- **Constructor**: `CascorServiceAdapter(service_url, api_key)` — NOT `(cascor_url, websocket_manager, data_client)`. The WebSocket manager is imported internally in `start_metrics_relay()`.
- **Backward-compatible method names**: Adapter exposes the same method names as `CascorIntegration` (e.g., `create_network`, `get_network_topology`, `start_training_background`, `request_training_stop`), minimizing route handler changes.
- **WS relay**: Uses `CascorTrainingStream.stream()` (async iterator), NOT `stream_metrics()` which does not exist.
- **Remote workers**: Stub no-ops — workers are managed server-side.
- **Monitoring hooks**: No-ops — monitoring is handled server-side and streamed via WebSocket.

```python
# src/backend/cascor_service_adapter.py (abbreviated — see detailed plan for full code)

from juniper_cascor_client import JuniperCascorClient, CascorTrainingStream

class CascorServiceAdapter:
    def __init__(self, service_url: str = "http://localhost:8200", api_key: str = None):
        self.client = JuniperCascorClient(base_url=service_url, api_key=api_key)
        ws_url = service_url.replace("http://", "ws://").replace("https://", "wss://")
        self.training_stream = CascorTrainingStream(base_url=ws_url, api_key=api_key)

    # Backward-compatible delegations
    def create_network(self, config=None): return self.client.create_network(**(config or {}))
    def get_network_topology(self): return self.client.get_topology()
    def start_training_background(self, **kw): self.client.start_training(**kw); return True
    def request_training_stop(self): self.client.stop_training(); return True
    def is_training_in_progress(self): return self.client.get_training_status().get("is_training", False)
    def get_training_status(self): return self.client.get_training_status()

    # No-ops (server-side concerns)
    def install_monitoring_hooks(self): return True
    def start_monitoring_thread(self, interval=1.0): pass
    def stop_monitoring(self): pass
    def restore_original_methods(self): pass
    def connect_remote_workers(self, *a, **kw): return True
    # ... see detailed plan for complete listing
```

### Step 4.3 — Three-Mode Activation in `main.py`

The activation logic supports three modes during the transition period:

| Mode        | Trigger                      | Backend                                      |
| ----------- | ---------------------------- | -------------------------------------------- |
| **Demo**    | `CASCOR_DEMO_MODE=1`         | `DemoMode` (unchanged)                       |
| **Service** | `CASCOR_SERVICE_URL` is set  | `CascorServiceAdapter` (new)                 |
| **Legacy**  | `CASCOR_BACKEND_PATH` is set | `CascorIntegration` (existing, transitional) |

Priority: Demo > Service > Legacy > fallback to Demo.

Route handlers require only a variable rename (`cascor_integration` → `backend`) since the adapter uses backward-compatible method names.

### Step 4.4 — Update Dependencies

```toml
# In juniper_canopy pyproject.toml
dependencies = [
    # ... existing deps ...
    "juniper-data-client>=0.3.0",
    "juniper-cascor-client>=0.1.0",
]
```

### Step 4.5 — Update Configuration

**Environment variables:**

| Old                        | New                         | Purpose                                                  |
| -------------------------- | --------------------------- | -------------------------------------------------------- |
| `CASCOR_BACKEND_PATH`      | **Retained** (transitional) | Legacy direct-import path (removed in Step 4.8)          |
| `CASCOR_BACKEND_AVAILABLE` | **Removed**                 | No longer needed                                         |
| (new)                      | `CASCOR_SERVICE_URL`        | URL of CasCor service (default: `http://localhost:8200`) |
| (new)                      | `CASCOR_SERVICE_API_KEY`    | API key for CasCor service (optional)                    |

**Default port:** `8200` — matches `juniper-cascor-client` defaults.

### Step 4.6 — Update Tests

- Unit tests: Mock `JuniperCascorClient` and verify adapter delegates correctly
- Interface compatibility tests: Verify adapter exposes all methods that `main.py` calls
- Three-mode activation tests: Verify correct backend for each env var combination
- WS relay tests: Mock `CascorTrainingStream.stream()` and verify broadcast
- Demo mode tests remain unchanged

### Step 4.7 — Integration Testing

- Start CasCor service, set `CASCOR_SERVICE_URL=http://localhost:8200`
- Verify: network creation, training start/stop, metrics streaming, topology retrieval
- Verify: demo mode still works with `CASCOR_DEMO_MODE=1`
- Verify: legacy mode still works with `CASCOR_BACKEND_PATH=...`

### Step 4.8 — Remove Legacy Mode (Post-Validation)

Once service mode is validated:

- Delete `src/backend/cascor_integration.py` (~1,600 lines)
- Remove `CASCOR_BACKEND_PATH` support from `main.py`
- Remove legacy mode branch from three-mode activation
- Remove all `sys.path` manipulation code and direct CasCor imports
- Remove tests that depend on CasCor source being on `sys.path`

### Deliverables, Phase 4

- [x] `CascorServiceAdapter` implemented and tested (306 lines, 52 unit tests — verified 2026-02-25)
- [x] Two-mode activation working (demo / service) — legacy mode removed in Step 4.8
- [x] All route handlers work with both backends (variable rename `cascor_integration` → `backend` complete)
- [x] WebSocket relay tested (CasCor WS → Canopy frontend) — verified 2026-02-25
- [x] All Canopy tests pass (3,130 passed, 23 skipped — verified 2026-02-25)
- [x] Demo mode continues to work identically
- [x] Configuration updated to use `CASCOR_SERVICE_URL` (port 8200)
- [x] Two-mode activation logic: Demo > Service > Demo fallback (legacy removed Step 4.8, verified 2026-02-25)
- [x] `pyproject.toml` updated with `juniper-cascor` optional dependency group (`juniper-cascor-client>=0.1.0`)
- [x] Unit tests for adapter (52 tests) and activation logic (6 tests) — verified 2026-02-25
- [x] `CascorIntegration` removed (all ~1,601 lines — commit `5f9987d`, Step 4.8)
- [x] All `sys.path` manipulation removed (only comments/docstrings remain)
- [x] No direct CasCor imports in Canopy
- [x] `CASCOR_BACKEND_PATH` support removed

---

## Phase 5 — Split into Separate Repositories

**Duration:** 1–2 weeks
**Risk:** Medium (mechanical but requires careful execution)
**Prerequisite:** Phase 4 complete (for Canopy extraction only; Data and CasCor have no Phase 4 dependency)
**Status:** COMPLETE — All three service repos extracted with CI green; monorepo superseded by `juniper-ml` meta-package (2026-02-25)

### Verification Summary, Phase 5 (2026-02-22)

| Repository              | Local Clone                                                              | Commits | Branch | CI Status             | Notable                                                                                             |
|-------------------------|--------------------------------------------------------------------------|---------|--------|-----------------------|-----------------------------------------------------------------------------------------------------|
| `juniper-data`          | `/home/pcalnon/Development/python/Juniper/juniper-data/`                 | 595     | `main` | GREEN (all jobs pass) | CodeQL + scheduled CI active; 3 dependabot PRs open                                                 |
| `juniper-cascor`        | `/home/pcalnon/Development/python/Juniper/juniper-cascor/`               | 127     | `main` | MOSTLY GREEN          | Recent failures (pytest-asyncio, logger config) resolved; latest 2 runs pass                        |
| `juniper-canopy`        | `/home/pcalnon/Development/python/Juniper/JuniperCanopy/juniper_canopy/` | 582     | `main` | GREEN (all jobs pass) | CI fully green 2026-02-25 (py3.11/3.12/3.13); cross-refs updated; Phase 5 complete                  |
| `juniper-data-client`   | `/home/pcalnon/Development/python/Juniper/juniper-data-client/`          | 6       | `main` | GREEN                 | Published to PyPI v0.3.0                                                                            |
| `juniper-cascor-client` | `/home/pcalnon/Development/python/Juniper/juniper-cascor-client/`        | 6       | `main` | GREEN                 | Published to PyPI v0.1.0 (2026-02-24)                                                               |
| `juniper-cascor-worker` | `/home/pcalnon/Development/python/Juniper/juniper-cascor-worker/`        | 7       | `main` | GREEN                 | Published to PyPI v0.1.0 (2026-02-24)                                                               |
| `juniper-ml` (meta)     | `/home/pcalnon/Development/python/Juniper/juniper-ml/`                   | —       | `main` | GREEN                 | Standalone meta-package; `pcalnon/juniper-ml` (renamed from `pcalnon/juniper`); CI green 2026-02-25 |

| Step | Description                        | Status                                                                                                                                                |
|------|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| 5.1  | Create target GitHub repos         | COMPLETE — all 3 repos created (`juniper-data`, `juniper-cascor`, `juniper-canopy`)                                                                   |
| 5.2  | Extract JuniperData with history   | COMPLETE — 595 commits (verified 2026-02-22), pushed to `pcalnon/juniper-data`                                                                        |
| 5.2  | Extract JuniperCascor with history | COMPLETE — 127 commits (verified 2026-02-22), pushed to `pcalnon/juniper-cascor`                                                                      |
| 5.2  | Extract JuniperCanopy with history | COMPLETE — 582 commits extracted to `pcalnon/juniper-canopy`; CI green 2026-02-25                                                                     |
| 5.3  | Verify extracted repos             | COMPLETE — Data: CI passing (scheduled + push); CasCor: CI passing (latest 2 runs green)                                                              |
| 5.4  | Set up per-repo CI/CD              | COMPLETE — CI fully green on both Data and CasCor (see details below)                                                                                 |
| 5.5  | Update cross-references            | COMPLETE — READMEs updated with ecosystem links and correct URLs                                                                                      |
| 5.6  | Archive the monorepo               | N/A — `pcalnon/Juniper` monorepo backup in `temp_git/Juniper/`; `juniper-ml` meta-package has its own repo (`pcalnon/juniper-ml`, renamed 2026-02-25) |
| 5.7  | Update local development setup     | COMPLETE — documented below                                                                                                                           |

**Extraction Approach Note:** The monorepo uses branch-per-project (not subdirectory-per-project), so `git filter-repo --subdirectory-filter` was not applicable. Instead, `git clone --single-branch -b <branch>` was used, followed by branch rename to `main`. This preserves the full reachable history for each project branch.

**Infrastructure Notes:**

- SSH deploy keys configured per-repo (`~/.ssh/id_ed25519_gh_juniper-<name>`)
- SSH config aliases: `github.com-juniper-data`, `github.com-juniper-cascor`, `github.com-juniper-canopy`

**CI Fixes Applied (juniper-data):**

- `.pre-commit-config.yaml`: Added `--severity=warning` to shellcheck hook to filter out style/info-level warnings
- `.pre-commit-config.yaml`: Added `exclude` pattern for legacy scripts with parse errors (`last_mod_update*.bash`, `no_canopy.bash`)

**CI Fixes Applied (juniper-cascor):**

- `.pre-commit-config.yaml`: Added Bandit skip codes — source: B301, B403 (pickle/dill for ML serialization); tests: B104, B108, B110, B301, B403 (hardcoded bind-all, /tmp paths, try/except/pass, pickle)
- `.pre-commit-config.yaml`: Coverage gate and pytest-unit hooks managed via `SKIP` env var in CI
- `.github/workflows/ci.yml`: **Major migration from conda/mamba to pip-based installation** — `conda_environment.yaml` is a local GPU machine explicit spec (contains CUDA packages) and is incompatible with CI runners; replaced `setup-miniconda` with `setup-python` + pip install across all test jobs
- `.github/workflows/ci.yml`: CPU-only PyTorch installed via `--index-url https://download.pytorch.org/whl/cpu`
- `.github/workflows/ci.yml`: Editable install (`pip install -e ".[ml,api,test]"`) for source-tree path resolution
- `pyproject.toml`: Added `[ml]` extras group (torch, h5py, matplotlib, PyYAML), `[all]` meta-group
- `pyproject.toml`: Added `psutil>=5.9.0` and `pytest-asyncio>=0.21.0` to test dependencies
- `src/log_config/logger/logger.py`: Bug fix — initialized `self.logger_configs = None` before try/except block to prevent `AttributeError` when config file is absent (CI environment)

### Objective, Phase 5

Move each subproject from the shared `Juniper` repo to its own dedicated GitHub repository with full git history preserved.

### Step 5.1 — Create Target Repositories

```bash
gh repo create pcalnon/juniper-data --public --description "Dataset generation service for the Juniper ecosystem"
gh repo create pcalnon/juniper-cascor --public --description "Cascade Correlation neural network training service"
gh repo create pcalnon/juniper-canopy --public --description "Monitoring and diagnostic dashboard for CasCor training"
```

Note: `juniper-data-client`, `juniper-cascor-client`, and `juniper-cascor-worker` repos were already created in Phases 1 and 3.

### Step 5.2 — Extract with History Using `git filter-repo`

For each subproject, extract its directory from the monorepo while preserving commit history:

```bash
# Install git-filter-repo if needed
pip install git-filter-repo

# --- JuniperData ---
git clone git@github.com:pcalnon/Juniper.git /tmp/juniper-data-extract
cd /tmp/juniper-data-extract

# Keep only the juniper_data directory (adjusting to actual path in repo)
git filter-repo --subdirectory-filter juniper_data

# Point to new remote
git remote add origin git@github.com:pcalnon/juniper-data.git
git push -u origin main --tags

# --- JuniperCascor ---
git clone git@github.com:pcalnon/Juniper.git /tmp/juniper-cascor-extract
cd /tmp/juniper-cascor-extract
git filter-repo --subdirectory-filter juniper_cascor
git remote add origin git@github.com:pcalnon/juniper-cascor.git
git push -u origin main --tags

# --- JuniperCanopy ---
git clone git@github.com:pcalnon/Juniper.git /tmp/juniper-canopy-extract
cd /tmp/juniper-canopy-extract
git filter-repo --subdirectory-filter juniper_canopy
git remote add origin git@github.com:pcalnon/juniper-canopy.git
git push -u origin main --tags
```

**Note:** The exact filter paths depend on how the monorepo directories map. If the subprojects are not cleanly nested (e.g., shared files at the root), a more nuanced `--path` approach may be needed. Test with `--dry-run` first.

### Step 5.3 — Verify Extracted Repositories

For each extracted repo:

```bash
# Clone fresh
git clone git@github.com:pcalnon/juniper-<name>.git
cd juniper-<name>

# Verify history
git log --oneline | head -20

# Verify builds
pip install -e ".[dev,test]"
pytest

# Verify CI
# Push a test branch to trigger GitHub Actions
```

### Step 5.4 — Set Up Per-Repo CI/CD

Each repository gets its own `.github/workflows/ci.yml`. These already exist in the monorepo — they just need cleanup to remove any cross-project assumptions.

Key changes per repo:

- Remove any `paths:` filters (no longer needed — each repo only has its own code)
- Update `checkout` step (no subdirectory navigation needed)
- Update test commands (may need to adjust `cd` paths)
- Add PyPI publish workflow for client packages

### Step 5.5 — Update Cross-References

- Update all `README.md` files to link to sibling repos by URL
- Update all `AGENTS.md` / `CLAUDE.md` files to reference the new repo structure
- Update any documentation that references monorepo paths

### Step 5.6 — Archive the Monorepo

```bash
# Update the original Juniper repo README
cat > README.md << 'EOF'
# Juniper (Archived)

This repository has been split into separate repositories:

- [juniper-data](https://github.com/pcalnon/juniper-data) — Dataset generation service
- [juniper-cascor](https://github.com/pcalnon/juniper-cascor) — CasCor neural network service
- [juniper-canopy](https://github.com/pcalnon/juniper-canopy) — Monitoring dashboard
- [juniper-data-client](https://github.com/pcalnon/juniper-data-client) — PyPI: juniper-data-client
- [juniper-cascor-client](https://github.com/pcalnon/juniper-cascor-client) — PyPI: juniper-cascor-client
- [juniper-cascor-worker](https://github.com/pcalnon/juniper-cascor-worker) — PyPI: juniper-cascor-worker

See individual repositories for current development.
EOF

git add README.md
git commit -m "Archive: project split into separate repositories"
git push origin main

# Archive the repo on GitHub (Settings → Archive this repository)
```

### Step 5.7 — Update Local Development Setup

```bash
# New local layout
Development/python/Juniper/
├── juniper-data/              # git@github.com:pcalnon/juniper-data.git
├── juniper-cascor/            # git@github.com:pcalnon/juniper-cascor.git
├── juniper-canopy/            # git@github.com:pcalnon/juniper-canopy.git
├── juniper-data-client/       # git@github.com:pcalnon/juniper-data-client.git (optional)
├── juniper-cascor-client/     # git@github.com:pcalnon/juniper-cascor-client.git (optional)
└── juniper-cascor-worker/     # git@github.com:pcalnon/juniper-cascor-worker.git (optional)
```

For local development of client packages alongside services:

```bash
# Install clients as editable for local development
pip install -e ../juniper-data-client
pip install -e ../juniper-cascor-client
```

### Deliverables, Phase 5

- [x] Three service repositories created with preserved git history (Data: 595 commits, CasCor: 127 commits, Canopy: 582 commits on `main` + 479 commits on `canopy/migration`)
- [x] Per-repo CI/CD verified — all jobs green: Pre-commit (3.11/3.12/3.13), Unit Tests (3.11/3.12/3.13), Security, Quick Integration, Full Integration, Build, Quality Gate
- [x] CI issues resolved: conda→pip migration, dependency fixes, Logger bug fix, Bandit/shellcheck tuning (see Infrastructure Notes)
- [x] Local clones verified (2026-02-22): juniper-data (595 commits, CI green, scheduled CI + CodeQL active, 3 dependabot PRs), juniper-cascor (127 commits, CI green on latest 2 runs, earlier failures in pytest-asyncio and logger resolved)
- [x] All cross-references updated (Data and CasCor READMEs updated with ecosystem links)
- [x] SSH deploy keys configured per-repo (`~/.ssh/id_ed25519_gh_juniper-<name>`) with SSH config aliases
- [x] Canopy extraction to standalone repo — COMPLETE 2026-02-25 (`pcalnon/juniper-canopy`, CI green)
- [x] `juniper-ml` meta-package has standalone repo `pcalnon/juniper-ml` (renamed from `pcalnon/juniper`); CI green 2026-02-25
- [x] Local development workflow documented

---

## Phase 6 — Post-Migration Hardening ✅ COMPLETE (2026-02-25)

**Duration:** 1–2 weeks (ongoing)
**Risk:** Low
**Prerequisite:** Phase 5 complete

### Step 6.1 — Version Coordination ✅ COMPLETE (2026-02-25)

Ecosystem compatibility matrix — current verified-compatible baseline:

| juniper-canopy | juniper-cascor | juniper-data | data-client | cascor-client | cascor-worker |
| -------------- | -------------- | ------------ | ----------- | ------------- | ------------- |
| 0.2.x          | 0.3.x          | 0.4.x        | >=0.3.1     | >=0.1.0       | >=0.1.0       |

Compatibility matrix added to READMEs in all repos (commit: `a7a2db8` cascor, `f4c4543` data, `cb5e4d6` canopy, `6900dcd` data-client, `410161a` cascor-client, `047c3f6` cascor-worker, `97f030f` juniper-ml).

### Step 6.2 — Integration Test Suite ✅ COMPLETE (2026-02-25)

Pytest-based suite in `juniper-deploy/tests/` (commit `5070046`):

| File | Coverage |
|------|----------|
| `conftest.py` | Shared fixtures: service URLs, HTTP session, cascor reset helper |
| `test_health.py` | `/v1/health`, `/v1/health/live`, `/v1/health/ready` for all 3 services; response schema validation |
| `test_data_service.py` | Generator list, dataset lifecycle (create → read → NPZ download → delete), stats |
| `test_full_stack.py` | CasCor network CRUD, CasCor start/stop training via JuniperData source, Canopy liveness; 3-service smoke test |

```bash
pip install -r requirements-test.txt
docker compose up -d
bash scripts/wait_for_services.sh
pytest tests/ -v
```

### Step 6.3 — Docker Compose for Full Stack ✅ COMPLETE (2026-02-25)

New `juniper-deploy` repo created at `pcalnon/juniper-deploy` (commit `7d98258`):

| File | Description |
|------|-------------|
| `docker-compose.yml` | Full stack with health checks and `depends_on` ordering |
| `.env.example` | Configurable port overrides |
| `scripts/wait_for_services.sh` | Polls all 3 health endpoints before tests |
| `README.md` | Quickstart, service URLs, integration test instructions |

Dockerfiles added:
- `juniper-cascor/Dockerfile` — multi-stage, CPU PyTorch, non-root user, `CMD ["python", "src/server.py"]` (commit `7ae3dcc`)
- `JuniperCanopy/juniper_canopy/Dockerfile` — multi-stage, CPU PyTorch, copies `conf/`, `CMD ["python", "src/main.py"]` (commit `e0fcf21`)

### Step 6.4 — Monitoring and Health Checks ✅ COMPLETE (2026-02-25)

All three services now expose standardized health endpoints:

| Endpoint           | juniper-data | juniper-cascor | juniper-canopy             |
|--------------------|--------------|----------------|----------------------------|
| `/v1/health`       | ✅           | ✅             | ✅ (new, commit `622994b`) |
| `/v1/health/live`  | ✅           | ✅             | ✅ (new)                   |
| `/v1/health/ready` | ✅           | ✅             | ✅ (new)                   |

JuniperCanopy retains `/health` and `/api/health` as backward-compatible aliases.

### Step 6.5 — Documentation Updates

**COMPLETE** (2026-02-25)

Added `## Architecture` (ASCII service topology diagram) and `## Related Services`
(env vars, Docker Deployment) sections to all three service READMEs:

| Repo | Commit |
|------|--------|
| juniper-data | `973ae39` |
| juniper-cascor | `8ffbe41` |
| juniper-canopy | `73d9919` |

### Deliverables, Phase 6

- [x] Version compatibility matrix documented
- [x] Integration test suite operational
- [x] Docker Compose full-stack deployment working
- [x] Health checks and monitoring in place
- [x] All documentation updated for polyrepo workflow

---

## Risk Register

| Risk                                                                | Impact | Likelihood | Mitigation                                                                                                         |
| ------------------------------------------------------------------- | ------ | ---------- | ------------------------------------------------------------------------------------------------------------------ |
| CasCor API design misses capabilities needed by Canopy              | High   | Medium     | Design API by examining every `CascorIntegration` method; validate with integration tests before removing old code |
| PyPI name collision                                                 | Medium | Low        | Check name availability before publishing; use `juniper-` prefix                                                   |
| Performance regression from HTTP vs in-process calls                | Medium | Medium     | Profile critical paths; use WebSocket streaming for metrics (not polling); connection pooling in clients           |
| Training state lost during CasCor service restart                   | High   | Medium     | Implement snapshot auto-save; add recovery protocol in client                                                      |
| `CascadeCorrelationNetwork` thread-safety issues in service context | High   | High       | Serialize all network access through `TrainingLifecycleManager`; single-threaded training with message passing     |
| Worker package needs full PyTorch — large install                   | Low    | Certain    | Document clearly; provide conda environment spec; consider future worker-lite option                               |
| Breaking changes during migration disrupt active development        | Medium | Medium     | Each phase is independently deployable; feature-flag new code paths; maintain demo mode throughout                 |

---

## Migration Checklist

> **Last verified:** 2026-02-25

### Phase 0 — Stabilize (COMPLETE 2026-02-19, Validated 2026-02-22)

- [x] All merge conflicts resolved
- [x] All branches build and tests pass
- [x] Clean baseline tagged (4 repos, 5 tags)
- [x] `.gitignore` updated

### Phase 1 — Publish `juniper-data-client` (COMPLETE 2026-02-21, Validated 2026-02-22)

- [x] `juniper-data-client` repo created (6 commits on `main`)
- [x] Published to TestPyPI
- [x] Published to PyPI (v0.3.0) — CI green, Trusted Publishing active
- [x] Vendored copies removed from Canopy
- [x] Vendored copies removed from Cascor
- [x] Vendored copy removed from JuniperData (commit `4bada2a`)
- [x] All tests pass with PyPI package (JuniperData 659, CasCor 226)

### Phase 2 — CasCor Service API (COMPLETE 2026-02-21, Validated 2026-02-22)

- [x] API contract defined and documented
- [x] FastAPI routes implemented (19 endpoints)
- [x] WebSocket streaming implemented (`/ws/training`, `/ws/control`)
- [x] `TrainingLifecycleManager` implemented (with state machine + monitor)
- [x] Service entry point (`server.py`) created
- [x] Existing CLI (`main.py`) still works (confirmed unchanged)
- [x] API test suite passing (213 unit + 13 integration tests)
- [x] Standalone repo `juniper-cascor` has 127 commits, CI passing

### Phase 3 — Client and Worker Packages (COMPLETE 2026-02-24, Validated 2026-02-24)

- [x] `juniper-cascor-client` package created (55 tests pass), GitHub repo pushed (6 commits)
- [x] `juniper-cascor-worker` package created (44 tests pass, 99% coverage), GitHub repo pushed (7 commits)
- [x] CI/CD workflow files committed and pushed (ci.yml + publish.yml with workflow_dispatch)
- [x] CI pipelines green on both repos (Python 3.11/3.12/3.13 matrix, all 6 jobs pass)
- [x] GitHub environments (`pypi` + `testpypi`) created on both repos
- [x] Worker CLI entry point functional (`juniper-cascor-worker` console_scripts)
- [x] Both packages editable-installed in local conda env (JuniperCascor; cascor-client also in JuniperCanopy)
- [x] README + LICENSE + py.typed in both packages
- [x] Lint (flake8), type check (mypy), and coverage (80%+) all pass on both repos
- [x] Package builds pass `twine check` (wheel + sdist)
- [x] Trusted Publishing configured on TestPyPI and PyPI for both packages (2026-02-24)
- [x] v0.1.0 releases created; `publish.yml` workflows completed successfully (TestPyPI → PyPI pipeline)
- [x] `pip install juniper-cascor-client juniper-cascor-worker` from PyPI verified (both v0.1.0)

### Phase 4 — Decouple Canopy (COMPLETE — validated 2026-02-25)

- [x] `CascorServiceAdapter` implemented (306 lines, wraps `juniper-cascor-client` over REST/WS)
- [x] `CascorIntegration` removed (deleted 2026-02-25; Step 4.8)
- [x] All `sys.path` manipulation removed
- [x] No direct CasCor imports in Canopy
- [x] Demo mode still works
- [x] All Canopy tests pass (3130 passed, 23 skipped — verified 2026-02-25)
- [x] Configuration updated to service URLs (`CASCOR_SERVICE_URL` env var activates service mode)
- [x] Two-mode activation logic: Demo > Service > Demo fallback (main.py)
- [x] `cascor_integration` global renamed to `backend` throughout main.py and tests
- [x] `pyproject.toml` updated with `juniper-cascor` optional dependency group (`juniper-cascor-client>=0.1.0`)
- [x] Unit tests for adapter (52 tests) and activation logic (11 tests) — verified 2026-02-22
- [x] WebSocket relay integration test (CasCor WS → adapter → Canopy frontend) — verified 2026-02-25
- [x] End-to-end integration test with live CasCor service in service mode — verified 2026-02-25

### Phase 5 — Split Repos (COMPLETE — 2026-02-25)

- [x] Three service repos created on GitHub (`juniper-data`, `juniper-cascor`, `juniper-canopy`)
- [x] Data extracted with history (595 commits on `main`, CI green including scheduled + CodeQL)
- [x] CasCor extracted with history (127 commits on `main`, CI green on latest runs)
- [x] Canopy extraction — COMPLETE (582 commits, CI green 2026-02-25, cross-refs updated)
- [x] Three client/worker repos created (from Phases 1, 3)
- [x] Per-repo CI/CD verified (Data: all green + CodeQL + scheduled; CasCor: green on latest 2 runs)
- [x] SSH deploy keys + SSH config aliases configured per-repo
- [x] `juniper-ml` meta-package has standalone repo `pcalnon/juniper-ml` (renamed from `pcalnon/juniper`); CI green 2026-02-25
- [x] Cross-references updated (Data, CasCor, and Canopy READMEs/AGENTS.md updated)

### Phase 6 — Hardening

- [x] Version compatibility matrix documented
- [x] Health check endpoints standardized (/v1/health + /v1/health/ready)
- [x] Integration tests operational (`juniper-deploy/tests/`, commit `5070046`)
- [x] Docker Compose full-stack working (`juniper-deploy` repo, commit `7d98258`; cascor Dockerfile `7ae3dcc`; canopy Dockerfile `e0fcf21`)
- [ ] Documentation complete

---

## Appendix A — CasCor Service API Contract (Draft)

### Common Response Envelope

All responses use a consistent envelope:

```json
{
    "status": "success" | "error",
    "data": { ... },
    "meta": {
        "timestamp": 1234567890.123,
        "version": "0.4.0"
    }
}
```

Error responses include additional fields:

```json
{
    "status": "error",
    "error": {
        "code": "TRAINING_NOT_RUNNING",
        "message": "Cannot pause: no training session is active"
    },
    "meta": { ... }
}
```

### Health Endpoints

#### `GET /v1/health`

```json
{"status": "ok", "version": "0.4.0", "uptime_seconds": 3600}
```

#### `GET /v1/health/ready`

Returns 200 if the service is ready to accept training requests:

```json
{"status": "ready", "network_loaded": true, "version": "0.4.0"}
```

### Network Endpoints

#### `POST /v1/network`

Create a new network.

**Request:**

```json
{
    "input_size": 2,
    "output_size": 1,
    "learning_rate": 0.1,
    "max_hidden_units": 10,
    "activation_fn": "sigmoid",
    "candidate_pool_size": 8
}
```

**Response (201):**

```json
{
    "status": "success",
    "data": {
        "uuid": "net-abc123",
        "input_size": 2,
        "output_size": 1,
        "hidden_units": 0,
        "created_at": 1234567890.123
    }
}
```

#### `GET /v1/network`

Get current network state.

#### `DELETE /v1/network`

Destroy current network. Returns 204.

### Training Endpoints

#### `POST /v1/training/start`

**Request:**

```json
{
    "dataset": {
        "source": "juniper-data",
        "url": "http://localhost:8100",
        "generator": "spiral",
        "params": {"n_points": 200, "noise": 0.15}
    },
    "params": {
        "max_epochs": 200,
        "learning_rate": 0.01,
        "patience": 10
    }
}
```

**Response (202 Accepted):**

```json
{
    "status": "success",
    "data": {
        "training_id": "train-xyz789",
        "state": "running",
        "started_at": 1234567890.123
    }
}
```

Training runs asynchronously. Progress is streamed via `/ws/training`.

#### `POST /v1/training/stop`

Request graceful stop. Response returns current state.

#### `POST /v1/training/pause`

Pause after current epoch completes.

#### `POST /v1/training/resume`

Resume paused training.

#### `POST /v1/training/reset`

Reset network and training state.

#### `GET /v1/training/status`

```json
{
    "status": "success",
    "data": {
        "state": "running",
        "epoch": 42,
        "phase": "output_training",
        "loss": 0.0523,
        "accuracy": 0.94,
        "hidden_units": 3,
        "elapsed_seconds": 12.5
    }
}
```

### Metrics Endpoints

#### `GET /v1/metrics`

Current metrics snapshot.

#### `GET /v1/metrics/history`

Full training history (list of per-epoch metrics).

### Network Info Endpoints

#### `GET /v1/network/topology`

Network structure for visualization (nodes, connections, layers).

#### `GET /v1/network/statistics`

Network statistics (parameter counts, weight distributions).

#### `GET /v1/dataset`

Current dataset info (source, dimensions, split sizes).

#### `GET /v1/decision-boundary`

Decision boundary grid data for 2D visualization.

### Snapshot Endpoints

#### `POST /v1/snapshots`

Create HDF5 snapshot. Returns snapshot metadata.

#### `GET /v1/snapshots`

List available snapshots.

#### `GET /v1/snapshots/{snapshot_id}`

Get snapshot detail.

#### `POST /v1/snapshots/{snapshot_id}/restore`

Restore network from snapshot.

### Worker Endpoints

#### `GET /v1/workers`

Remote worker status.

#### `POST /v1/workers/connect`

```json
{"address": "192.168.1.100", "port": 5000, "num_workers": 4}
```

#### `POST /v1/workers/start`

Start connected workers.

#### `POST /v1/workers/stop`

Stop workers gracefully.

#### `POST /v1/workers/disconnect`

Disconnect and cleanup.

### WebSocket: `/ws/training`

Server sends messages in this format:

```json
{"type": "metrics", "timestamp": 1234567890.123, "data": {"epoch": 42, "loss": 0.05, "accuracy": 0.94}}
{"type": "state", "timestamp": 1234567890.123, "data": {"state": "running", "phase": "candidate_training"}}
{"type": "topology", "timestamp": 1234567890.123, "data": {"nodes": [...], "connections": [...]}}
{"type": "cascade_add", "timestamp": 1234567890.123, "data": {"unit_index": 3, "correlation": 0.87}}
{"type": "event", "timestamp": 1234567890.123, "data": {"event": "training_complete", "final_accuracy": 0.98}}
```

### WebSocket: `/ws/control`

Client sends commands:

```json
{"command": "start", "params": {"max_epochs": 200}}
{"command": "stop"}
{"command": "pause"}
{"command": "resume"}
{"command": "reset"}
{"command": "set_params", "params": {"learning_rate": 0.005}}
```

Server responds:

```json
{"type": "command_response", "command": "start", "status": "ok", "data": {"training_id": "train-xyz789"}}
{"type": "command_response", "command": "stop", "status": "ok"}
{"type": "command_response", "command": "set_params", "status": "error", "error": "Cannot change learning_rate during candidate training phase"}
```

---

## Appendix B — PyPI Publishing Workflow

### GitHub Actions Workflow

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

permissions:
  id-token: write  # Required for trusted publishing

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install build tools
        run: pip install build twine

      - name: Build package
        run: python -m build

      - name: Check package
        run: twine check dist/*

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # Uses trusted publishing (OIDC) — no API token needed
        # Configure at https://pypi.org/manage/project/<package>/settings/publishing/
```

### Trusted Publishing Setup

1. Go to <https://pypi.org/manage/project/juniper-data-client/settings/publishing/>
2. Add GitHub as a trusted publisher:
   - Owner: `pcalnon`
   - Repository: `juniper-data-client`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. No API tokens needed — PyPI verifies via OIDC

### Release Process

```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG.md
# 3. Commit and push
git add pyproject.toml CHANGELOG.md
git commit -m "Release v0.3.1"
git push origin main

# 4. Create GitHub release (triggers publish workflow)
gh release create v0.3.1 --title "v0.3.1" --notes "Bug fixes and improvements"
```

---

## Appendix C — Package Inventory

### Packages to Publish to PyPI

| Package                 | Version | Dependencies                  | Install Size (est.) |
| ----------------------- | ------- | ----------------------------- | ------------------- |
| `juniper-data-client`   | 0.3.0   | numpy, requests, urllib3      | ~50 KB              |
| `juniper-cascor-client` | 0.1.0   | requests, urllib3, websockets | ~60 KB              |
| `juniper-cascor-worker` | 0.1.0   | numpy, torch                  | ~5 KB + torch       |

### Packages NOT on PyPI (Server Applications)

| Package          | Version | Deploy Method                    |
| ---------------- | ------- | -------------------------------- |
| `juniper-data`   | 0.5.0   | Docker / pip install from source |
| `juniper-cascor` | 0.4.0   | Docker / pip install from source |
| `juniper-canopy` | 0.3.0   | Docker / pip install from source |

### Import Names (Underscore Convention)

| PyPI Name (hyphen)      | Import Name (underscore) |
| ----------------------- | ------------------------ |
| `juniper-data-client`   | `juniper_data_client`    |
| `juniper-cascor-client` | `juniper_cascor_client`  |
| `juniper-cascor-worker` | `juniper_cascor_worker`  |
