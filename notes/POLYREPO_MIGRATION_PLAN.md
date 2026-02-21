# Juniper Polyrepo Migration Plan

**Last Updated:** 2026-02-21
**Version:** 1.2.0
**Status:** Active — Phase 0 Complete, Phase 1 Complete, Phase 2 Complete
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
│   Port 8050         │                  │    Port 8060         │
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
         ┌──────┘       │    ┌─────┘          │        ┌────┘
         ▼              ▼    ▼                ▼        ▼
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
**Status:** COMPLETE (2026-02-19)

### Objective

Resolve the broken `main` branch so it represents a clean, buildable state for all three subprojects.

### Completion Summary

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
**Status:** COMPLETE (2026-02-21)

### Objective, Phase 1

Publish `juniper-data-client` to PyPI as the single source of truth. Eliminate all vendored copies.

### Completion Summary, Phase 1

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
**Status:** COMPLETE (2026-02-21)

### Objective, Phase 2

Add a FastAPI + WebSocket service layer to JuniperCascor so it can be consumed as a network service rather than a library import. The existing CLI entry point (`main.py`) continues to work unchanged.

### Completion Summary, Phase 2

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

### Step 3.1 — Create `juniper-cascor-client`

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
    def __init__(self, base_url="http://localhost:8060", timeout=30, retries=3, api_key=None): ...

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

    def __init__(self, base_url="ws://localhost:8060", api_key=None): ...

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...

    async def stream_metrics(self) -> AsyncIterator[dict]:
        """Yields metrics updates as they arrive."""
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

### Step 3.2 — Create `juniper-cascor-worker`

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

### Step 3.3 — Publish Both to PyPI

Follow the same process as Phase 1 (TestPyPI first, then PyPI, with GitHub Actions automated publishing).

### Deliverables, Phase 3

- [ ] `juniper-cascor-client` repository created and published to PyPI
- [ ] `juniper-cascor-worker` repository created and published to PyPI
- [ ] Both packages have CI/CD workflows for automated testing and publishing
- [ ] Comprehensive test suites for both packages
- [ ] README documentation with installation and usage examples

---

## Phase 4 — Decouple Canopy from CasCor

**Duration:** 2–3 weeks
**Risk:** High (core architectural change to Canopy)
**Prerequisite:** Phases 2 and 3 complete

### Objective, Phase 4

Replace Canopy's `CascorIntegration` class (1,600 lines of `sys.path` injection, monkey-patching, and direct CasCor imports) with the `juniper-cascor-client` package communicating over REST/WebSocket.

### Step 4.1 — Understand What `CascorIntegration` Currently Does

`CascorIntegration` (in `src/backend/cascor_integration.py`) currently performs these functions:

| Function Category      | Methods                                                                                                                          | Replacement                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Backend discovery**  | `_resolve_backend_path`, `_add_backend_to_path`, `_import_backend_modules`                                                       | **Eliminated** — no path injection needed                                            |
| **Network lifecycle**  | `create_network`, `connect_to_network`                                                                                           | `cascor_client.create_network()`, `.get_network()`                                   |
| **Training control**   | `fit_async`, `start_training_background`, `request_training_stop`, `is_training_in_progress`                                     | `cascor_client.start_training()`, `.stop_training()`, `.get_training_status()`       |
| **Monitoring hooks**   | `install_monitoring_hooks`, monkey-patching `fit`/`train_output`/`train_candidates`, `_on_*` callbacks                           | **Eliminated** — CasCor service handles its own monitoring and streams via WebSocket |
| **Monitoring thread**  | `start_monitoring_thread`, `stop_monitoring`, `_monitoring_loop`, `_extract_current_metrics`                                     | **Replaced** by WebSocket subscription via `CascorTrainingStream`                    |
| **Network data**       | `get_network_topology`, `get_network_data`, `extract_cascor_topology`, `get_dataset_info`, `get_prediction_function`             | `cascor_client.get_topology()`, `.get_dataset()`, `.get_decision_boundary()`         |
| **Remote workers**     | `connect_remote_workers`, `start_remote_workers`, `stop_remote_workers`, `disconnect_remote_workers`, `get_remote_worker_status` | `cascor_client.connect_workers()`, `.start_workers()`, etc.                          |
| **Dataset generation** | `_generate_dataset_from_juniper_data`, `_create_juniper_dataset`, `_generate_dataset_local`                                      | `data_client.create_dataset()` — already using the client                            |
| **Snapshots**          | (referenced in `main.py`, not fully implemented in `CascorIntegration`)                                                          | `cascor_client.create_snapshot()`, `.restore_snapshot()`                             |
| **Broadcasting**       | `_broadcast_message` (sends to Canopy's WebSocketManager)                                                                        | WebSocket stream from CasCor, relayed to Canopy's frontend                           |

### Step 4.2 — Create `CascorServiceAdapter`

Replace `CascorIntegration` with a new `CascorServiceAdapter` that wraps the cascor client:

```python
# src/backend/cascor_service_adapter.py

from juniper_cascor_client import JuniperCascorClient, CascorTrainingStream

class CascorServiceAdapter:
    """Adapter between Canopy's internal interfaces and the CasCor service.

    Replaces CascorIntegration. Communicates with CasCor via REST API
    and WebSocket rather than direct Python imports.
    """

    def __init__(self, cascor_url: str, websocket_manager, data_client=None):
        self.client = JuniperCascorClient(base_url=cascor_url)
        self.stream = CascorTrainingStream(base_url=cascor_url.replace("http", "ws"))
        self.ws_manager = websocket_manager
        self.data_client = data_client
        self._stream_task = None

    async def start_metrics_relay(self):
        """Connect to CasCor WebSocket and relay metrics to Canopy frontend."""
        await self.stream.connect()
        self._stream_task = asyncio.create_task(self._relay_loop())

    async def _relay_loop(self):
        """Relay messages from CasCor WS to Canopy's frontend WebSocket."""
        async for message in self.stream.stream_metrics():
            await self.ws_manager.broadcast(message)

    # Delegate to client
    def create_network(self, config): return self.client.create_network(config)
    def start_training(self, **kwargs): return self.client.start_training(**kwargs)
    def stop_training(self): return self.client.stop_training()
    def pause_training(self): return self.client.pause_training()
    def resume_training(self): return self.client.resume_training()
    def reset_training(self): return self.client.reset_training()
    def get_training_status(self): return self.client.get_training_status()
    def get_topology(self): return self.client.get_topology()
    def get_metrics(self): return self.client.get_metrics()
    def get_metrics_history(self): return self.client.get_metrics_history()
    # ... etc.
```

### Step 4.3 — Update `main.py` Routes

Replace all references to `cascor_integration` in `main.py` with `cascor_service_adapter`:

- `get_topology()` → `adapter.get_topology()`
- `get_metrics()` → `adapter.get_metrics()`
- `api_train_start()` → `adapter.start_training()`
- etc.

The demo mode path remains unchanged — `DemoMode` continues to work as-is for local development.

### Step 4.4 — Update Dependencies

```toml
# In juniper_canopy pyproject.toml
dependencies = [
    # ... existing deps ...
    "juniper-data-client>=0.3.0",
    "juniper-cascor-client>=0.1.0",
]
```

### Step 4.5 — Remove `CascorIntegration` and `sys.path` Code

Once `CascorServiceAdapter` is validated:

```bash
# Remove the old integration
rm src/backend/cascor_integration.py

# Remove sys.path manipulation code from main.py
# Remove CASCOR_BACKEND_PATH resolution
# Remove _import_backend_modules calls
# Remove all direct CasCor imports
```

### Step 4.6 — Update Configuration

**Environment variables:**

| Old                        | New                      | Purpose                                                  |
| -------------------------- | ------------------------ | -------------------------------------------------------- |
| `CASCOR_BACKEND_PATH`      | **Removed**              | No longer needed                                         |
| `CASCOR_BACKEND_AVAILABLE` | **Removed**              | No longer needed                                         |
| (new)                      | `CASCOR_SERVICE_URL`     | URL of CasCor service (default: `http://localhost:8060`) |
| (new)                      | `CASCOR_SERVICE_API_KEY` | API key for CasCor service (optional)                    |

**`conf/app_config.yaml` updates:**

```yaml
backend:
  cascor_service:
    url: "http://localhost:8060"
    timeout: 30
    retries: 3
    api_key: null  # Or set via CASCOR_SERVICE_API_KEY env var
```

### Step 4.7 — Update Tests

- Update integration tests to mock `JuniperCascorClient` instead of `CascadeCorrelationNetwork`
- Remove tests that depend on CasCor source being on `sys.path`
- Add integration tests that exercise the full Canopy → CasCor service flow (using a test CasCor server or mock)
- Demo mode tests remain unchanged

### Step 4.8 — Update Demo Mode

`DemoMode` already simulates the CasCor backend. No changes needed to DemoMode itself. The switching logic in `main.py` changes from:

```python
# Old
if demo_mode_active:
    demo_mode_instance = get_demo_mode()
else:
    cascor_integration = CascorIntegration(backend_path=cascor_backend_path)
```

To:

```python
# New
if demo_mode_active:
    demo_mode_instance = get_demo_mode()
else:
    cascor_adapter = CascorServiceAdapter(
        cascor_url=os.getenv("CASCOR_SERVICE_URL", "http://localhost:8060"),
        websocket_manager=websocket_manager,
    )
```

### Deliverables, Phase 4

- [ ] `CascorServiceAdapter` implemented and tested
- [ ] `CascorIntegration` removed (all 1,600 lines)
- [ ] All `sys.path` manipulation code removed
- [ ] No direct imports of CasCor modules anywhere in Canopy
- [ ] All Canopy tests pass
- [ ] Demo mode continues to work identically
- [ ] Configuration updated to use service URLs instead of backend paths

---

## Phase 5 — Split into Separate Repositories

**Duration:** 1–2 weeks
**Risk:** Medium (mechanical but requires careful execution)
**Prerequisite:** Phase 4 complete

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

- [ ] Three service repositories created with preserved git history
- [ ] Per-repo CI/CD verified and passing
- [ ] All cross-references updated
- [ ] Original monorepo archived
- [ ] Local development workflow documented

---

## Phase 6 — Post-Migration Hardening

**Duration:** 1–2 weeks (ongoing)
**Risk:** Low
**Prerequisite:** Phase 5 complete

### Step 6.1 — Version Coordination

Establish a version compatibility matrix:

| juniper-canopy | juniper-cascor | juniper-data | data-client | cascor-client | cascor-worker |
| -------------- | -------------- | ------------ | ----------- | ------------- | ------------- |
| 0.3.0          | 0.4.0          | 0.5.0        | >=0.3.0     | >=0.1.0       | >=0.1.0       |

Document minimum compatible versions in each service's README.

### Step 6.2 — Integration Test Suite

Create a lightweight integration test repository or script that:

1. Starts JuniperData on port 8100
2. Starts JuniperCascor on port 8060
3. Starts JuniperCanopy on port 8050
4. Runs end-to-end tests through the full stack
5. Can be run in CI via docker-compose

### Step 6.3 — Docker Compose for Full Stack

```yaml
# docker-compose.yml (in a juniper-deploy or juniper-infra repo)
services:
  juniper-data:
    build: ../juniper-data
    ports: ["8100:8100"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8100/v1/health"]

  juniper-cascor:
    build: ../juniper-cascor
    ports: ["8060:8060"]
    environment:
      JUNIPER_DATA_URL: http://juniper-data:8100
    depends_on:
      juniper-data:
        condition: service_healthy

  juniper-canopy:
    build: ../juniper-canopy
    ports: ["8050:8050"]
    environment:
      CASCOR_SERVICE_URL: http://juniper-cascor:8060
      JUNIPER_DATA_URL: http://juniper-data:8100
    depends_on:
      juniper-cascor:
        condition: service_healthy
```

### Step 6.4 — Monitoring and Health Checks

- Each service exposes `/v1/health` and `/v1/health/ready`
- Canopy dashboard shows connection status to CasCor and Data services
- Alerts when downstream services are unreachable

### Step 6.5 — Documentation Updates

Update all documentation across all repositories:

- Architecture diagrams showing service topology
- Developer setup guides for the polyrepo workflow
- Contribution guides for each repo
- API documentation for each service

### Deliverables, Phase 6

- [ ] Version compatibility matrix documented
- [ ] Integration test suite operational
- [ ] Docker Compose full-stack deployment working
- [ ] Health checks and monitoring in place
- [ ] All documentation updated for polyrepo workflow

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

### Phase 0 — Stabilize (COMPLETE 2026-02-19)

- [x] All merge conflicts resolved
- [x] All branches build and tests pass
- [x] Clean baseline tagged (4 repos, 5 tags)
- [x] `.gitignore` updated

### Phase 1 — Publish `juniper-data-client` (COMPLETE 2026-02-21)

- [x] `juniper-data-client` repo created
- [x] Published to TestPyPI
- [x] Published to PyPI (v0.3.0)
- [x] Vendored copies removed from Canopy
- [x] Vendored copies removed from Cascor
- [x] Vendored copy removed from JuniperData (commit `4bada2a`)
- [x] All tests pass with PyPI package (JuniperData 659, CasCor 226)

### Phase 2 — CasCor Service API (COMPLETE 2026-02-21)

- [x] API contract defined and documented
- [x] FastAPI routes implemented (19 endpoints)
- [x] WebSocket streaming implemented (`/ws/training`, `/ws/control`)
- [x] `TrainingLifecycleManager` implemented (with state machine + monitor)
- [x] Service entry point (`server.py`) created
- [x] Existing CLI (`main.py`) still works (confirmed unchanged)
- [x] API test suite passing (213 unit + 13 integration tests)

### Phase 3 — Client and Worker Packages (IN PROGRESS — packages created, PyPI pending)

- [x] `juniper-cascor-client` package created (55 tests pass), GitHub repo created and pushed
- [x] `juniper-cascor-worker` package created (24 tests pass), GitHub repo created and pushed
- [x] Both have CI/CD workflow files (ci.yml, publish.yml) — pending push (PAT needs `workflow` scope)
- [x] Worker CLI entry point functional (`juniper-cascor-worker` console_scripts entry point)
- [ ] Push CI/CD workflow commits (requires PAT `workflow` scope)
- [ ] Publish both to TestPyPI then PyPI (requires `.pypirc` with API tokens)
- [ ] Set up Trusted Publishing on PyPI for both packages

### Phase 4 — Decouple Canopy

- [ ] `CascorServiceAdapter` implemented
- [ ] `CascorIntegration` removed
- [ ] All `sys.path` manipulation removed
- [ ] No direct CasCor imports in Canopy
- [ ] Demo mode still works
- [ ] All Canopy tests pass
- [ ] Configuration updated to service URLs

### Phase 5 — Split Repos

- [ ] Three service repos created with history
- [ ] Three client/worker repos created (from Phases 1, 3)
- [ ] Per-repo CI/CD verified
- [ ] Monorepo archived
- [ ] Cross-references updated

### Phase 6 — Hardening

- [ ] Version compatibility matrix documented
- [ ] Integration tests operational
- [ ] Docker Compose full-stack working
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
