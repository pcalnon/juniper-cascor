# Juniper Monorepo Analysis

**Last Updated:** 2026-02-18
**Version:** 1.0.0
**Status:** Current
**Author:** Paul Calnon / Claude Code

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Current Architecture](#current-architecture)
- [Assessment by Dimension](#assessment-by-dimension)
  - [Best Practices Alignment](#1-best-practices-alignment)
  - [Maintainability](#2-maintainability)
  - [Potential Problems](#3-potential-problems)
  - [Scalability](#4-scalability)
- [Recommendations](#recommendations)
- [Concerns](#concerns)
- [Strategy Comparison](#strategy-comparison)
- [Decision](#decision)

---

## Executive Summary

The Juniper project uses a **single GitHub repository** (`pcalnon/Juniper.git`) containing three subprojects, each with its own application. This analysis evaluates the current approach against best practices and identifies structural risks that will worsen as the project grows.

**Key Finding:** The current approach is an informal monorepo without monorepo tooling. It carries the complexity costs of a monorepo (merge conflicts, entangled history, shared CI triggers) without the benefits that proper monorepo tooling provides (dependency graphs, selective builds, workspace management). The three subprojects are developed on long-lived branches and maintained as three independent full git clones of the same remote, each tracking a different branch.

**Recommendation:** Migrate to a **multi-repository (polyrepo) architecture** with proper package distribution via PyPI. This is the correct long-term strategy given the project's direction toward service-oriented architecture, where CasCor will run as an independent service communicating via REST API and WebSockets rather than via `sys.path` injection and direct Python imports.

---

## Current Architecture

### Repository Layout

| Subproject    | Application      | Version | Role                                               |
| ------------- | ---------------- | ------- | -------------------------------------------------- |
| JuniperData   | `juniper_data`   | 0.4.2   | Dataset generation service (FastAPI, port 8100)    |
| JuniperCascor | `juniper_cascor` | 0.3.17  | CasCor neural network backend (library, no server) |
| JuniperCanopy | `juniper_canopy` | 0.2.3   | Monitoring dashboard (FastAPI + Dash, port 8050)   |

### How It Works Today

- **Single GitHub remote:** All three subprojects share `git@github.com:pcalnon/Juniper.git`
- **Branch-per-subproject development:** Long-lived branches like `subproject.juniper_cascor.feature.frontend_integration.pre_deploy.spiral_gen_extract.release`
- **Three full clones locally:** Each subproject is cloned separately, each tracking a different branch
- **No monorepo tooling:** No Nx, Turborepo, Pants, Bazel, uv workspaces, or hatch workspaces
- **Per-subproject configuration:** Each has its own `pyproject.toml`, `.github/workflows/`, `.pre-commit-config.yaml`, and `conf/` directory
- **Cross-project integration via vendoring and sys.path:**
  - `juniper_data_client` is vendored (copied) into Canopy and Cascor
  - Canopy imports CasCor classes via runtime `sys.path` manipulation

### Dependency Graph (Current)

```bash
juniper-data-client  (vendored copies in each consumer - no single source of truth)
       ↑                    ↑
juniper-data          juniper-cascor  (library, no API)
       ↑                    ↑
       └──── juniper-canopy ┘  (imports CasCor via sys.path injection)
```

---

## Assessment by Dimension

### 1. Best Practices Alignment

**What the current approach gets right:**

- **Unified visibility** — All code is discoverable in one place
- **Atomic cross-project changes** — A single commit can theoretically touch all three subprojects
- **Consistent tooling conventions** — All three use the same linter settings (line-length 512), test framework (pytest), conda environment (`JuniperPython`), and code style
- **Hierarchical branch naming** — `subproject.<name>.<feature>.<phase>` provides clear organization

**Where it diverges from monorepo best practices:**

- **No workspace/build orchestration** — True monorepos use tools like Nx, Turborepo, Pants, or Bazel to manage inter-project dependencies, selective builds, and caching. Juniper has none of this.
- **No dependency graph** — There is no declarative way to express that Canopy depends on CasCor and Data, or that changes to Data's client should trigger Canopy and Cascor CI.
- **Three clones instead of one** — Running three independent `git clone` operations of the same repo, each on different branches, negates the primary benefit of a monorepo (one working tree, one source of truth).
- **Duplicated configuration** — `pre-commit-config.yaml`, CI workflows, bash utility scripts (`conf/common.conf`, `conf/logging_functions.conf`), and conda configs are maintained as independent copies rather than shared.

### 2. Maintainability

**Current pain points observed in the repository:**

| Issue                | Evidence                                                                                                                                                                                                           | Severity   |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| Merge conflicts      | `pyproject.toml`, `ci.yml`, `.pre-commit-config.yaml`, `CHANGELOG.md`, `README.md` all show `UU` (unmerged) status                                                                                                 | **High**   |
| Vendored code drift  | `juniper_data_client` exists in 3+ locations with different implementations (Canopy has HTTPAdapter/retry; Cascor has manual retry; Canopy's vendored copy lacks `api_key` parameter present in canonical version) | **High**   |
| Config duplication   | `conf/common.conf`, `conf/logging_functions.conf`, etc. are copied across all three projects                                                                                                                       | **Medium** |
| CI/CD divergence     | Python matrix differs: Canopy/Data use 3.12–3.14, Cascor uses 3.11–3.13                                                                                                                                            | **Medium** |
| Branch proliferation | Deep branch hierarchies like `subproject.juniper_cascor.feature.frontend_integration.pre_deploy.spiral_gen_extract.release`                                                                                        | **Low**    |

**The merge conflict problem is structural**, not incidental. When all three subprojects share `main` and each has files at the same relative paths (`pyproject.toml`, `README.md`, `CHANGELOG.md`), merging any subproject branch into `main` will conflict with the others. This will get worse, not better, as the project grows.

### 3. Potential Problems

#### Problem 1: Identity Crisis — Three Projects Sharing One `pyproject.toml` Path

Each subproject has its own `pyproject.toml` at its root, but they all live on `main`. When `main` is checked out, which project's `pyproject.toml` is "the" project? The current merge conflict in `pyproject.toml` shows JuniperData and JuniperCanopy headers colliding — this is a fundamental structural issue, not a one-time conflict.

#### Problem 2: CI/CD Cannot Target Changed Subprojects

All three subprojects have `.github/workflows/ci.yml`. When a push to `main` triggers CI, GitHub Actions will attempt to run all workflows. There is no path-based filtering (`paths:` in workflow triggers) to run only the affected subproject's tests. This means:

- Every push runs all three CI pipelines (wasteful)
- Or CI runs the wrong pipeline for the branch (incorrect)
- Or CI only works correctly on subproject branches, not `main` (fragile)

#### Problem 3: Vendored Dependencies Create Silent Divergence

The `juniper_data_client` is vendored (copied) into Canopy and Cascor rather than installed as a package. The copies have already diverged:

- Canopy's copy uses `requests.adapters.HTTPAdapter` with `urllib3.util.retry.Retry`
- Cascor's copy uses manual retry loops with `time.sleep`
- The canonical version in JuniperData has an `api_key` constructor parameter; Canopy's vendored copy does not

When a bug is fixed or feature added in Data's client, it must be manually synchronized to two other locations. This is the classic "copy-paste inheritance" anti-pattern.

#### Problem 4: `sys.path` Manipulation Is Fragile

Canopy imports CasCor classes by injecting `CASCOR_BACKEND_PATH` into `sys.path` at runtime. This:

- Breaks IDE type checking and autocompletion
- Makes dependency relationships invisible to tooling
- Can silently import wrong versions if paths are misconfigured
- Cannot be validated at build time
- Creates tight coupling between two projects that should be independently deployable

#### Problem 5: Git History Entanglement

All three subprojects share a single git history on `main`. Running `git log` on `main` shows an interleaved history of unrelated changes across all three projects. This makes:

- `git bisect` unreliable (changes to Data should not affect Cascor tests)
- Release tagging ambiguous (does `v0.4.2` refer to Data, Cascor, or Canopy?)
- `git blame` noisy with cross-project merge commits

### 4. Scalability

**Current state (3 subprojects):** Manageable with discipline, but already showing strain (merge conflicts, config drift).

**At 5–7 subprojects:** The approach will become untenable:

- Merge conflicts multiply combinatorially (each new project conflicts with all others on shared-path files)
- CI run time grows linearly (all pipelines trigger on every push)
- Vendored code copies become impossible to synchronize
- Branch namespace becomes unwieldy
- New contributors face a steep learning curve understanding which clone/branch combination to use

**At 10+ subprojects:** The system will effectively be unmaintainable without monorepo tooling.

---

## Recommendations

### Primary Recommendation: Multi-Repository (Polyrepo) Architecture

Given Juniper's architectural direction — CasCor becoming an independent service, communication via REST/WebSocket, PyPI-published client packages — separate repositories are the correct strategy.

**Target architecture:**

```bash
github.com/pcalnon/juniper-data             # Dataset generation service
github.com/pcalnon/juniper-cascor           # CasCor neural network service
github.com/pcalnon/juniper-canopy           # Monitoring dashboard
github.com/pcalnon/juniper-data-client      # PyPI: juniper-data-client
github.com/pcalnon/juniper-cascor-client    # PyPI: juniper-cascor-client (new)
```

**Dependency graph (target):**

```bash
juniper-data-client (PyPI)     juniper-cascor-client (PyPI, new)
       ↑         ↑                      ↑           ↑
juniper-data   juniper-cascor     juniper-canopy   remote workers
               (service, FastAPI)  (REST/WS client)
```

**See:** [POLYREPO_MIGRATION_PLAN.md](POLYREPO_MIGRATION_PLAN.md) for the detailed migration plan.

### Immediate Guard-Rails (While Migration Is In Progress)

1. **Resolve the active merge conflicts** — The `UU` files in `juniper_canopy` will block all further merges
2. **Stop vendoring `juniper_data_client`** — Install from source as an editable package immediately
3. **Flatten branch names** — Use short-lived feature branches (`cascor/add-api`, not `subproject.juniper_cascor.feature.frontend_integration.pre_deploy.spiral_gen_extract.release`)

---

## Concerns

1. **The merge conflict backlog is a ticking clock.** The `UU` status on `pyproject.toml`, `ci.yml`, `.pre-commit-config.yaml`, `README.md`, `CHANGELOG.md`, and 10+ other files means the current `main` branch is in a broken state. Every day this persists makes resolution harder.

2. **Vendored client divergence is a bug factory.** If Canopy's `juniper_data_client` has retry logic that Cascor's does not, and one has `api_key` support while the other does not, bugs will manifest differently in each consumer. This is invisible until production.

3. **The three-clone workflow is non-obvious.** A new contributor would need to understand that they must clone the same repo three times, check out different branches in each, and mentally track which clone they are working in. This is high cognitive overhead with no tooling support.

4. **Release versioning is ambiguous.** With three packages at versions 0.2.3, 0.3.17, and 0.4.2 all in one repo, git tags like `v0.4.2` are ambiguous. Prefixed tags (`data-v0.4.2`, `cascor-v0.3.17`) add complexity that separate repos eliminate.

5. **CasCor's `sys.path` coupling blocks independent deployment.** Canopy cannot be deployed without a local copy of the CasCor source tree, even though in a service architecture they should be independently deployable to different hosts.

---

## Strategy Comparison

| Dimension              | Current (Informal Monorepo) | True Monorepo (uv workspaces) | Polyrepo (Recommended)       |
| ---------------------- | --------------------------- | ----------------------------- | ---------------------------- |
| Merge conflicts        | Frequent, structural        | Rare (separate paths)         | None (separate repos)        |
| Cross-project changes  | Possible but painful        | Easy (one commit)             | Coordinated releases         |
| CI efficiency          | All pipelines always run    | Path-filtered                 | Independent                  |
| Dependency management  | Vendored copies             | Workspace installs            | Published packages (PyPI)    |
| Contributor onboarding | 3 clones, branch juggling   | 1 clone, standard PRs         | Multiple repos, standard PRs |
| Tooling complexity     | None (manual)               | Moderate (workspace setup)    | Low (standard Python)        |
| Scalability            | Poor (>5 projects breaks)   | Good (proven pattern)         | Good (proven pattern)        |
| Independent deployment | Not possible                | Possible with effort          | Natural                      |
| Migration effort       | N/A                         | Medium                        | Medium-High                  |
| Service-oriented fit   | Poor                        | Moderate                      | Excellent                    |

---

## Decision

**Adopt the polyrepo architecture** with PyPI-published client packages.

This decision is driven by:

1. The project's architectural direction toward independent services (CasCor as a service, not a library import)
2. The need for independently deployable and versionable components
3. The current structural problems (merge conflicts, vendored code drift) that will only worsen
4. The desire to publish client packages to PyPI for external consumption
5. The addition of remote worker clients that must be independently installable on worker hardware

See [POLYREPO_MIGRATION_PLAN.md](POLYREPO_MIGRATION_PLAN.md) for the concrete implementation plan.
