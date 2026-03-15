# CI Coverage Gate Plan — Separate Coverage Check from Unit Tests

**Created:** 2026-02-22
**Status:** COMPLETE — Both repos CI fully green (2026-02-22)
**Author:** Paul Calnon / Claude Code
**Related:** POLYREPO_MIGRATION_PLAN.md Phase 5, Step 5.4 (Per-Repo CI/CD)

---

## Objective

Move the code coverage enforcement from the unit test step into its own, separate step during pre-commit. This applies to both `pcalnon/juniper-cascor` and `pcalnon/juniper-data`.

---

## Current State

### juniper-cascor

**Pre-commit (`.pre-commit-config.yaml`):**
- `pytest-coverage` local hook runs tests AND enforces 80% coverage in a single command
- Hardcodes `/opt/miniforge3/envs/JuniperPython/bin/python` — works locally, fails in CI (exit 127)
- Bandit hooks fail on B301 (pickle) — intentional pickle usage not excluded

**CI (`ci.yml`):**
- Pre-commit job: runs `pre-commit run --all-files` — fails because of hardcoded path + bandit
- Unit-tests job: runs pytest with `--cov-fail-under=80` embedded in the test command (coverage check combined with test execution)
- Unit-tests depends on pre-commit — blocked because pre-commit fails
- All downstream jobs blocked

### juniper-data

**Pre-commit (`.pre-commit-config.yaml`):**
- Shellcheck hook runs on all shell files with default settings — fails on 15 findings across 8 `util/` scripts
- No coverage gate hook in pre-commit (coverage only enforced in CI unit-tests job)

**CI (`ci.yml`):**
- Pre-commit job: fails on shellcheck
- Unit-tests job: runs pytest with `--cov-fail-under=80` embedded in the test command
- All downstream jobs blocked

---

## Plan

### 1. juniper-cascor — Fix pre-commit and separate coverage gate

#### 1a. Fix Bandit B301

Add `B301` (pickle deserialization) to `--skip` for both bandit hooks in `.pre-commit-config.yaml`. Pickle is used intentionally for random state serialization and multiprocessing pickling tests.

#### 1b. Split pre-commit coverage hook into two hooks

Current single hook (runs tests + enforces coverage):
```yaml
- id: pytest-coverage
  entry: bash -c 'cd src && /opt/miniforge3/envs/JuniperPython/bin/python -m pytest ... --cov-fail-under=80'
```

Split into:
```yaml
# Hook 1: Run unit tests with coverage data collection
- id: pytest-unit
  entry: bash -c 'cd src && python -m pytest -m "unit and not slow" tests/unit -q --tb=short --no-header -p no:warnings --cov=. --cov-config=../pyproject.toml --cov-report=term-missing:skip-covered'

# Hook 2: Enforce coverage threshold (separate step)
- id: coverage-gate
  entry: bash -c 'cd src && python -m coverage report --fail-under=80'
```

Key changes:
- Removes hardcoded conda path — uses `python` (works with any activated environment)
- Separates test execution from coverage enforcement
- Each appears as its own named step in pre-commit output

#### 1c. Skip local hooks in CI pre-commit, add dedicated CI coverage step

The CI pre-commit job only installs `pre-commit` (no torch/numpy/etc.), so local hooks that run tests can't work. Skip them via `SKIP` env var. Instead, separate the CI unit-tests step:

- **"Run Unit Tests"**: pytest with `--cov` flags but WITHOUT `--cov-fail-under`
- **"Enforce Coverage Gate (80%)"**: `python -m coverage report --fail-under=${COVERAGE_FAIL_UNDER}` as its own step

### 2. juniper-data — Fix shellcheck and separate coverage gate

#### 2a. Fix shellcheck failures

- Exclude non-Data scripts from shellcheck: `no_canopy.bash`, `last_mod_update-ORIG.bash`, `last_mod_update-local.bash`
- Add `--severity=warning` to shellcheck args (matching CasCor convention)
- Fix remaining issues: `last_mod_update.bash` sed syntax (SC1073), unquoted variables (SC2086) in `impatient_testing.bash` and `update_weekly.bash`

#### 2b. Separate CI coverage gate step

Same pattern as CasCor: split "Run Unit Tests with Coverage Gate" into two CI steps.

---

## Deliverables

| Item | Repo | Status |
|------|------|--------|
| Bandit B301 added to skip list | juniper-cascor | COMPLETE (commit f346047) |
| Pre-commit coverage split into two hooks | juniper-cascor | COMPLETE (pytest-unit + coverage-gate) |
| CI coverage gate as separate step | juniper-cascor | COMPLETE (commit f346047) |
| CI pre-commit skips local test hooks | juniper-cascor | COMPLETE (SKIP: pytest-unit,coverage-gate) |
| CI migrated from conda to pip | juniper-cascor | COMPLETE (commit 99d3c71) |
| Add [api] extras to pip install | juniper-cascor | COMPLETE (commit 5509766) |
| Add psutil to test deps | juniper-cascor | COMPLETE (commit 6587113) |
| Editable install for path resolution | juniper-cascor | COMPLETE (commit 724da05) |
| Fix Logger logger_configs AttributeError | juniper-cascor | COMPLETE (commit 5c001cc) |
| Add pytest-asyncio to test deps | juniper-cascor | COMPLETE (commit 6f331d6) |
| CI coverage gate re-separated into own step | juniper-cascor | COMPLETE (commit 5509766) |
| Shellcheck exclusions + severity filter | juniper-data | COMPLETE (commit 7e0b706) |
| Shell script fixes (sed, quoting) | juniper-data | COMPLETE (commit 7e0b706) |
| CI coverage gate as separate step | juniper-data | COMPLETE (commit 7e0b706) |
| CI passes — juniper-data | juniper-data | COMPLETE (CI green 2026-02-22) |
| Install [all] extras (includes juniper-data-client) | juniper-cascor | COMPLETE (commit e5c7502) |
| CI passes — juniper-cascor | juniper-cascor | COMPLETE (CI green 2026-02-22, all jobs passing) |

### Additional Fixes Discovered During Execution

1. **Bandit B403/B104/B108/B110** — Additional bandit codes needed for tests (added by parallel thread, commits 9537a86, 4fb2a56)
2. **Conda → pip migration** — Parallel thread migrated CI from conda to pip for portability (commit 99d3c71), which introduced several dependency issues:
   - Missing `[api]` extras (fastapi not installed) — fixed by adding `api` to pip install
   - Missing `psutil` from test deps — added to pyproject.toml
   - Path resolution broken by non-editable install — switched to `pip install -e`
   - Logger `logger_configs` AttributeError — `self.logger_configs` not initialized before try/except (fix: commit 5c001cc)
   - Missing `pytest-asyncio` for async WebSocket tests — added to pyproject.toml

---

## Alignment with POLYREPO_MIGRATION_PLAN.md

This work falls under **Phase 5, Step 5.4 — Set Up Per-Repo CI/CD**:

> Each repository gets its own `.github/workflows/ci.yml`. These already exist in the monorepo — they just need cleanup to remove any cross-project assumptions.

The coverage gate fix and separation directly addresses the "cleanup to remove cross-project assumptions" objective — the hardcoded conda path was a monorepo-specific assumption that breaks in the standalone repo CI.
