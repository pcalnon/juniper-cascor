# CI Pipeline Development Plan — juniper-cascor

**Created:** 2026-02-22
**Status:** IN PROGRESS (Phases 1-3 COMPLETE, Phase 4 CONDITIONAL)
**Author:** Paul Calnon / Claude Code
**Source of Truth:** `Juniper/JuniperCascor/juniper_cascor/` CI pipeline
**Target:** `Juniper/juniper-cascor/` CI pipeline
**Related:** `notes/CI_COVERAGE_GATE_PLAN.md`, `notes/POLYREPO_MIGRATION_PLAN.md` (Phase 5)

---

## 1. Objective

Audit and align the `juniper-cascor` (polyrepo) CI pipeline against the reference CI pipeline in `JuniperCascor/juniper_cascor/` (monorepo). Ensure feature parity, document gaps, and plan required changes. Where the polyrepo has already evolved beyond the monorepo (e.g., conda-to-pip migration), preserve those improvements.

---

## 2. Current State Summary

### 2.1 Reference Pipeline (`JuniperCascor/juniper_cascor/`)

| File                                    | Version | Description                                         |
| --------------------------------------- | ------- | --------------------------------------------------- |
| `.github/workflows/ci.yml`              | 0.4.1   | Main CI/CD pipeline — 8 jobs                        |
| `.github/workflows/scheduled-tests.yml` | 0.1.0   | Nightly slow/long/performance tests — 5 jobs        |
| `.pre-commit-config.yaml`               | 0.4.2   | 14 hooks + 1 local pytest-coverage hook             |
| `pyproject.toml`                        | 0.3.17  | Tool configs (black, isort, pytest, coverage, mypy) |
| `src/tests/pytest.ini`                  | —       | Pytest markers and timeout config                   |
| `src/tests/conftest.py`                 | —       | Custom CLI options, fixtures, env setup             |
| `src/tests/scripts/run_tests.bash`      | —       | Local test runner                                   |
| `src/tests/scripts/run_benchmarks.bash` | —       | Performance benchmark harness                       |

**Key Features:**

- Python version matrix: 3.11, 3.12, 3.13
- Concurrency control (`cancel-in-progress: true`)
- **Dependency management: conda/mamba** (explicit spec from `conf/conda_environment.yaml`)
- Pre-commit: all hooks run in CI (no SKIP), including hardcoded conda path in pytest-coverage local hook
- Unit tests with combined `--cov-fail-under=80` in pytest command
- Quick + full integration test split
- Build verification (`python -m build --sdist --wheel`)
- Security scanning: Gitleaks, Bandit SARIF, pip-audit (strict)
- Quality gate aggregator job
- Build notification job
- Scheduled nightly: slow unit, slow integration, long-running correctness, performance benchmarks
- Artifact retention: 30 days (tests/coverage), 90 days (benchmarks)

### 2.2 Target Pipeline (`juniper-cascor/`)

| File                                    | Version | Description                                                |
| --------------------------------------- | ------- | ---------------------------------------------------------- |
| `.github/workflows/ci.yml`              | 0.4.1   | Main CI/CD pipeline — 8 jobs (pip-based)                   |
| `.github/workflows/scheduled-tests.yml` | 0.1.0   | Nightly slow/long/performance tests — 5 jobs (conda-based) |
| `.pre-commit-config.yaml`               | 0.4.2   | 14 hooks + 2 local hooks (pytest-unit + coverage-gate)     |
| `pyproject.toml`                        | 0.3.17  | Tool configs + `[ml]`, `[all]` dependency groups           |
| `src/tests/pytest.ini`                  | —       | Pytest markers and timeout config                          |
| `src/tests/conftest.py`                 | —       | Custom CLI options, fixtures, env setup                    |
| `src/tests/scripts/run_tests.bash`      | —       | Local test runner                                          |
| `src/tests/scripts/run_benchmarks.bash` | —       | Performance benchmark harness                              |

**Key Features (differences from reference):**

- ci.yml: **Migrated from conda to pip** for CI portability
- ci.yml: CPU-only PyTorch via `--index-url https://download.pytorch.org/whl/cpu`
- ci.yml: Editable pip install (`pip install -e ".[all]"`)
- ci.yml: Pre-commit skips local hooks in CI (`SKIP: pytest-unit,coverage-gate`)
- ci.yml: Separate "Enforce Coverage Gate (80%)" step (not combined with pytest)
- Pre-commit: Split `pytest-coverage` into `pytest-unit` + `coverage-gate`
- Pre-commit: Uses `python3` (not `python3.13`) as default language
- Pre-commit: Additional Bandit skip codes for pickle/dill serialization
- pyproject.toml: Has `[ml]`, `[all]` optional dependency groups
- pyproject.toml: Has `pytest-asyncio`, `psutil` in test deps
- **scheduled-tests.yml: Still uses conda/mamba** (NOT migrated)

---

## 3. Gap Analysis

### 3.1 Features Present in Both (Parity Achieved)

| Feature                                               | ci.yml | scheduled-tests.yml | Pre-commit | pyproject.toml |
| ----------------------------------------------------- | ------ | ------------------- | ---------- | -------------- |
| Python 3.11/3.12/3.13 matrix (ci.yml)                 | MATCH  | —                   | —          | —              |
| Concurrency control                                   | MATCH  | —                   | —          | —              |
| Pre-commit job (black, isort, flake8, mypy, bandit)   | MATCH  | —                   | MATCH      | —              |
| Unit tests with coverage                              | MATCH  | —                   | —          | —              |
| Coverage threshold 80%                                | MATCH  | —                   | —          | MATCH          |
| Quick integration tests (all branches)                | MATCH  | —                   | —          | —              |
| Full integration tests (PR/main/develop only)         | MATCH  | —                   | —          | —              |
| Build verification (sdist + wheel)                    | MATCH  | —                   | —          | —              |
| Security scanning (Gitleaks, Bandit SARIF, pip-audit) | MATCH  | —                   | —          | —              |
| Quality gate aggregator                               | MATCH  | —                   | —          | —              |
| Build notification                                    | MATCH  | —                   | —          | —              |
| Slow unit tests (nightly)                             | —      | MATCH               | —          | —              |
| Slow integration tests (nightly)                      | —      | MATCH               | —          | —              |
| Long-running correctness tests                        | —      | MATCH               | —          | —              |
| Performance benchmarks                                | —      | MATCH               | —          | —              |
| Test summary aggregator                               | —      | MATCH               | —          | —              |
| General file checks (yaml, toml, json, ast, etc.)     | —      | —                   | MATCH      | —              |
| Flake8 source + tests (separate configs)              | —      | —                   | MATCH      | —              |
| MyPy Phase 1 (incremental error codes)                | —      | —                   | MATCH      | —              |
| Markdown linting                                      | —      | —                   | MATCH      | —              |
| Shell script linting (shellcheck)                     | —      | —                   | MATCH      | —              |
| YAML linting (yamllint)                               | —      | —                   | MATCH      | —              |
| Black/isort tool configs                              | —      | —                   | —          | MATCH          |
| Pytest markers and timeout                            | —      | —                   | —          | MATCH          |
| Coverage report config                                | —      | —                   | —          | MATCH          |
| MyPy overrides for torch/numpy/h5py                   | —      | —                   | —          | MATCH          |

### 3.2 juniper-cascor Improvements Over Reference (Keep)

These are improvements the polyrepo already made that should be preserved:

| Improvement                                 | File                    | Rationale                                                 |
| ------------------------------------------- | ----------------------- | --------------------------------------------------------- |
| **Conda-to-pip migration** (ci.yml)         | ci.yml                  | Polyrepo portability — no conda dependency in CI          |
| **CPU-only PyTorch**                        | ci.yml                  | CI doesn't need CUDA; saves minutes and disk              |
| **Editable pip install**                    | ci.yml                  | Preserves `__file__`-based path resolution for constants  |
| **SKIP local hooks in CI**                  | ci.yml                  | Local hooks require conda env; CI has dedicated jobs      |
| **Separate coverage gate step**             | ci.yml                  | Cleaner failure reporting; decoupled from test execution  |
| **Split pytest-unit + coverage-gate hooks** | .pre-commit-config.yaml | Cleaner local workflow; each step visible in output       |
| **Bandit B301/B403 skip** (source)          | .pre-commit-config.yaml | Pickle/dill used intentionally for ML model serialization |
| **Bandit B104/B108/B110 skip** (tests)      | .pre-commit-config.yaml | Tests bind 0.0.0.0, use /tmp, have try/except pass        |
| **`[ml]` dependency group**                 | pyproject.toml          | Clean separation of ML deps for pip install               |
| **`[all]` dependency group**                | pyproject.toml          | Single `pip install -e ".[all]"` for full environment     |
| **pytest-asyncio in test deps**             | pyproject.toml          | Required for async WebSocket API tests                    |
| **psutil in test deps**                     | pyproject.toml          | Required for multiprocessing test fixtures                |

### 3.3 Gaps Requiring Changes in juniper-cascor

| #         | Gap                                                    | Severity   | File(s) Affected          | Description                                                                                                                                                     |
| --------- | ------------------------------------------------------ | ---------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GAP-1** | scheduled-tests.yml still uses conda/mamba             | **HIGH**   | `scheduled-tests.yml`     | All 4 test jobs use `conda-incubator/setup-miniconda@v3` and `mamba create`. Should be migrated to pip (matching ci.yml).                                       |
| **GAP-2** | Pre-commit default_language_version mismatch           | **LOW**    | `.pre-commit-config.yaml` | Reference uses `python3.13`, target uses `python3`. Not a functional issue but inconsistent.                                                                    |
| **GAP-3** | Duplicate pytest-asyncio entry                         | **LOW**    | `pyproject.toml`          | `pytest-asyncio>=0.21.0` appears twice in `[project.optional-dependencies.test]`.                                                                               |
| **GAP-4** | MyPy files pattern may need API expansion              | **MEDIUM** | `.pre-commit-config.yaml` | MyPy `files`: `^src/(cascade_correlation\|candidate_unit\|spiral_problem\|snapshots)/` — does not include `api/`, `cascor_constants/`, `log_config/`, `utils/`. |
| **GAP-5** | No `conftest.py` `--slow` / `--integration` flag audit | **LOW**    | `src/tests/conftest.py`   | Same conftest both repos. Verify CLI options registered & wired (`--gpu`, `--slow`, `--integration`, `--fast-slow`, `--run-long`)                               |

### 3.4 Backport Recommendations (Reference Should Adopt from Target)

These changes should be applied back to `JuniperCascor/juniper_cascor/` to bring it in line:

| #        | Change                                                      | Severity   | File(s)                   |
| -------- | ----------------------------------------------------------- | ---------- | ------------------------- |
| **BP-1** | Migrate ci.yml from conda to pip                            | **HIGH**   | `ci.yml`                  |
| **BP-2** | Add SKIP env var for local hooks in CI pre-commit           | **HIGH**   | `ci.yml`                  |
| **BP-3** | Separate coverage gate into own CI step                     | **MEDIUM** | `ci.yml`                  |
| **BP-4** | Add Bandit B301/B403 to source skip list                    | **MEDIUM** | `.pre-commit-config.yaml` |
| **BP-5** | Add Bandit B104/B108/B110 to test skip list                 | **MEDIUM** | `.pre-commit-config.yaml` |
| **BP-6** | Split pytest-coverage hook into pytest-unit + coverage-gate | **MEDIUM** | `.pre-commit-config.yaml` |
| **BP-7** | Add `[ml]` and `[all]` optional dependency groups           | **MEDIUM** | `pyproject.toml`          |
| **BP-8** | Add pytest-asyncio and psutil to test deps                  | **LOW**    | `pyproject.toml`          |
| **BP-9** | Migrate scheduled-tests.yml from conda to pip               | **HIGH**   | `scheduled-tests.yml`     |

---

## 4. Implementation Plan

### Phase 1: Fix scheduled-tests.yml (GAP-1) — COMPLETE

**Status:** COMPLETE (2026-02-22)

**Goal:** Migrate `scheduled-tests.yml` from conda/mamba to pip, matching the pattern already established in `ci.yml`.

**Scope:** All 4 test jobs in `scheduled-tests.yml`:

1. `slow-unit-tests`
2. `slow-integration-tests`
3. `long-running-tests`
4. `performance-benchmarks`

**Changes applied per job:**

| Before (conda)                                                        | After (pip)                                                                                     |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `conda-incubator/setup-miniconda@v3`                                  | `actions/setup-python@v5`                                                                       |
| `mamba create -y -n "${ENV_NAME}" --file conf/conda_environment.yaml` | `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[all]"` |
| `conda activate "${ENV_NAME}"`                                        | (not needed — Python already on PATH)                                                           |
| `shell: bash -l {0}`                                                  | `run:` (default shell)                                                                          |
| Conda + pip cache blocks                                              | Single pip cache via `actions/setup-python` `cache: pip`                                        |
| `ENV_NAME` env var                                                    | Removed                                                                                         |
| `set -euxo pipefail`                                                  | Removed (GitHub Actions defaults)                                                               |

**Changes implemented:**

1. **File header:** Updated version `0.1.0` → `0.2.0`, Last Modified `2026-02-04` → `2026-02-22`, added migration reference to description
2. **Global env:** Removed `ENV_NAME: JuniperCascor` (no longer needed without conda)
3. **slow-unit-tests:** Replaced Miniforge + conda cache with `actions/setup-python@v5` + pip install pattern. Added `--slow` flag to pytest command. Removed `shell: bash -l {0}` from all steps.
4. **slow-integration-tests:** Same migration pattern. Added `--slow` flag. Simplified install step.
5. **long-running-tests:** Same migration pattern. Kept `--run-long` flag and 600s timeout. Preserved conditional execution logic.
6. **performance-benchmarks:** Same migration pattern. Verified `run_benchmarks.bash` has no conda references. Preserved 90-day artifact retention.
7. **summary job:** No changes (pure status aggregation).
8. **Added migration comments** to each job section header for traceability.

**Verification performed:**

- YAML syntax validated via `yaml.safe_load()`
- Grep confirmed zero functional conda/mamba/miniforge references (only in comments)
- All 4 jobs use `actions/setup-python@v5`, `actions/checkout@v4`, `actions/upload-artifact@v4`
- Benchmark script (`run_benchmarks.bash`) confirmed free of conda references

**Testing (pending push):**

- Trigger via `workflow_dispatch` after push
- Verify all 4 jobs complete successfully
- Verify benchmark output is produced

### Phase 2: Fix Minor Gaps (GAP-2, GAP-3) — COMPLETE

**Status:** COMPLETE (2026-02-22)

**GAP-2: Pre-commit default_language_version:**

**Decision:** Keep as `python3`. The `python3` value is more portable across CI environments where Python 3.13 may not be available. The specific Python version is controlled by `actions/setup-python@v5` in CI and the user's environment locally. No change needed.

**GAP-3: Duplicate pytest-asyncio entry:**

Removed duplicate `"pytest-asyncio>=0.21.0"` from `[project.optional-dependencies.test]` in `pyproject.toml` (line 68).

### Phase 3: Expand MyPy Coverage (GAP-4) — COMPLETE

**Status:** COMPLETE (2026-02-22)

**Goal:** Expand MyPy file pattern to include new modules (especially the Phase 2 API).

**Previous pattern:**

```yaml
files: ^src/(cascade_correlation|candidate_unit|spiral_problem|snapshots)/
```

**New pattern:**

```yaml
files: ^src/(cascade_correlation|candidate_unit|spiral_problem|snapshots|api|cascor_constants|remote_client)/
```

**Audit results (mypy with all pre-commit flags, run from project root, backups excluded):**

| Directory           | Files | Errors | Error Codes                                                 | Decision                     |
| ------------------- | ----- | ------ | ----------------------------------------------------------- | ---------------------------- |
| `api/`              | 23    | **0**  | — (notes only: annotation-unchecked)                        | **Added**                    |
| `cascor_constants/` | 14    | **0**  | —                                                           | **Added**                    |
| `remote_client/`    | 3     | **0**  | —                                                           | **Added**                    |
| `utils/`            | 2     | 1      | `valid-type` (utils.py:197 — `[str]` should be `list[str]`) | Documented for future fix    |
| `cascor_plotter/`   | 2     | 1      | `valid-type` (cascor_plotter.py:252 — `plt` module as type) | Documented for future fix    |
| `profiling/`        | 4     | 1      | `operator` (memory.py:170 — `None` subtraction)             | Documented for future fix    |
| `log_config/`       | 4     | 26     | `valid-type` x22, `misc` x1, `operator` x2, `call-arg` x1   | Deferred (major refactoring) |

**Combined validation:** 57 source files across all 7 covered directories — **0 errors** (exit code 0).

**Changes made:**

1. `.pre-commit-config.yaml`: Updated mypy `files` pattern to add `api`, `cascor_constants`, `remote_client`
2. `.pre-commit-config.yaml`: Added directory coverage documentation in mypy hook comments, listing covered, pending, and deferred directories with specific error details

### Phase 4: Backport to Reference Repo — CONDITIONAL

These changes should be applied to `JuniperCascor/juniper_cascor/` only if it remains actively maintained alongside the polyrepo. If the monorepo is being deprecated in favor of the polyrepo (per the POLYREPO_MIGRATION_PLAN.md), these backports are unnecessary.

**Decision required:** Is `JuniperCascor/juniper_cascor/` still actively maintained, or is `juniper-cascor` the canonical repo going forward?

If backporting:

- Apply BP-1 through BP-9 from Section 3.4
- Priority: BP-1 (conda-to-pip), BP-2 (SKIP env var), BP-9 (scheduled-tests pip) first

---

## 5. File Change Matrix

| File                                    | Phase | Action       | Description                                                         |
| --------------------------------------- | ----- | ------------ | ------------------------------------------------------------------- |
| `.github/workflows/scheduled-tests.yml` | 1     | **COMPLETE** | Migrated all 4 jobs from conda to pip                               |
| `pyproject.toml`                        | 2     | **COMPLETE** | Removed duplicate pytest-asyncio                                    |
| `.pre-commit-config.yaml`               | 3     | **COMPLETE** | Expanded MyPy files pattern (+api, cascor_constants, remote_client) |

No new files are created. No files are deleted.

---

## 6. Verification Checklist

### Phase 1 Verification

- [ ] `scheduled-tests.yml` syntax validates (`act` or push to branch)
- [ ] `slow-unit-tests` job passes with pip-based install
- [ ] `slow-integration-tests` job passes
- [ ] `long-running-tests` job passes (manual dispatch with `run_long_tests: true`)
- [ ] `performance-benchmarks` job runs benchmarks and uploads artifact
- [ ] `summary` job correctly aggregates results
- [ ] No conda references remain in `scheduled-tests.yml`
- [ ] Artifact retention periods preserved (30 days tests, 90 days benchmarks)

### Phase 2 Verification

- [ ] `pip install -e ".[test]"` succeeds without warnings about duplicate dep
- [ ] Pre-commit runs locally without warnings

### Phase 3 Verification

- [ ] MyPy passes on all newly-included directories
- [ ] No new CI failures introduced
- [ ] Phase tracking comments updated in `.pre-commit-config.yaml`

### Overall CI Parity Verification

- [ ] ci.yml: All 8 jobs match or exceed reference features
- [ ] scheduled-tests.yml: All 5 jobs match or exceed reference features
- [ ] Pre-commit: All hooks present and configured correctly
- [ ] pyproject.toml: All tool configs aligned
- [ ] Coverage threshold: 80% enforced in pre-commit AND CI
- [ ] Security scanning: Gitleaks + Bandit SARIF + pip-audit all present
- [ ] Build verification: sdist + wheel packaging confirmed
- [ ] Python matrix: 3.11, 3.12, 3.13 tested

---

## 7. Risk Assessment

| Risk                                                 | Probability | Impact | Mitigation                                                         |
| ---------------------------------------------------- | ----------- | ------ | ------------------------------------------------------------------ |
| pip install fails for scheduled-tests (missing deps) | Medium      | High   | Follow exact pattern from ci.yml; test via workflow_dispatch first |
| Benchmark script assumes conda environment           | Low         | Medium | Verify `run_benchmarks.bash` doesn't reference conda directly      |
| MyPy expansion introduces many violations            | Medium      | Low    | Audit each directory before adding; phase incrementally            |
| torch CPU wheel unavailable for Python 3.11          | Low         | High   | PyTorch supports 3.11+; verify in test workflow                    |

---

## 8. Timeline and Priority

| Phase                                      | Priority    | Estimated Effort      | Dependencies                     |
| ------------------------------------------ | ----------- | --------------------- | -------------------------------- |
| Phase 1: scheduled-tests.yml pip migration | HIGH        | COMPLETE (2026-02-22) | None                             |
| Phase 2: pyproject.toml cleanup            | LOW         | COMPLETE (2026-02-22) | None                             |
| Phase 3: MyPy expansion                    | MEDIUM      | COMPLETE (2026-02-22) | MyPy audit results               |
| Phase 4: Backport to reference             | CONDITIONAL | 2-3 hours             | Decision on monorepo maintenance |

---

## Appendix A: Complete ci.yml Job Dependency Graph

```bash

                      ┌───────────────────┐
                      │    pre-commit     │
                      │  (Py 3.11/12/13)  │
                      └─────────┬─────────┘
                                │
              ┌─────────────────┼────────────────┐
              │                 │                │
              ▼                 │                ▼
       ┌──────────────────┐     │      ┌────────────────────┐
       │    unit-tests    │     │      │      security      │
       │  (Py 3.11/12/13) │     │      │  (Gitleaks/Bandit/ │
       └─────────┬────────┘     │      │     pip-audit)     │
                 │              │      └─────────┬──────────┘
    ┌────────────┼──────────┐   │                │
    │            │          │   │                │
    ▼            ▼          ▼   │                │
┌────────┐   ┌────────┐  ┌──────┴─────┐          │
│ quick  │   │  full  │  │   build    │          │
│ integ. │   │ integ. │  │            │          │
└───┬────┘   └───┬────┘  └──────┬─────┘          │
    │            │              │                │
    └────────────┼──────────────┼────────────────┘
                 │              │
                 ▼              ▼
            ┌───────────────────────┐
            │    required-checks    │
            └───────────┬───────────┘
                        │
                        ▼
                 ┌────────────┐
                 │   notify   │
                 └────────────┘
```

---

## Appendix B: Complete scheduled-tests.yml Job Dependency Graph

```bash
┌──────────────────┐  ┌─────────────────────┐  ┌───────────────────┐  ┌─────────────────────┐
│ slow-unit-tests  │  │ slow-integ.-tests   │  │  long-running     │  │ perf. benchmarks    │
│                  │  │                     │  │ (schedule/manual) │  │                     │
└────────┬─────────┘  └──────────┬──────────┘  └─────────┬─────────┘  └──────────┬──────────┘
         │                       │                       │                       │
         └───────────────────────┼───────────────────────┴───────────────────────┘
                                 │
                                 ▼
                          ┌────────────┐
                          │   summary  │
                          └────────────┘
```

---

## Appendix C: Pre-commit Hook Inventory (juniper-cascor)

| #   | Hook ID                 | Source           | Version   | Files                      | Scope      |
| --- | ----------------------- | ---------------- | --------- | -------------------------- | ---------- |
| 1   | check-yaml              | pre-commit-hooks | v4.6.0    | all yaml                   | General    |
| 2   | check-toml              | pre-commit-hooks | v4.6.0    | all toml                   | General    |
| 3   | check-json              | pre-commit-hooks | v4.6.0    | all json                   | General    |
| 4   | end-of-file-fixer       | pre-commit-hooks | v4.6.0    | all                        | General    |
| 5   | trailing-whitespace     | pre-commit-hooks | v4.6.0    | all                        | General    |
| 6   | check-merge-conflict    | pre-commit-hooks | v4.6.0    | all                        | General    |
| 7   | check-added-large-files | pre-commit-hooks | v4.6.0    | all (<1MB)                 | General    |
| 8   | check-case-conflict     | pre-commit-hooks | v4.6.0    | all                        | General    |
| 9   | check-ast               | pre-commit-hooks | v4.6.0    | python                     | General    |
| 10  | debug-statements        | pre-commit-hooks | v4.6.0    | python                     | General    |
| 11  | detect-private-key      | pre-commit-hooks | v4.6.0    | all                        | Security   |
| 12  | black                   | psf/black        | 25.1.0    | `^src/`                    | Formatting |
| 13  | isort                   | pycqa/isort      | 5.13.2    | `^src/`                    | Formatting |
| 14  | flake8 (source)         | pycqa/flake8     | 7.1.1     | `^src/` excl tests         | Linting    |
| 15  | flake8 (tests)          | pycqa/flake8     | 7.1.1     | `^src/tests/`              | Linting    |
| 16  | mypy                    | mirrors-mypy     | v1.13.0   | `^src/(cc\|cu\|sp\|snap)/` | Types      |
| 17  | bandit (source)         | PyCQA/bandit     | 1.7.9     | `^src/` excl tests         | Security   |
| 18  | bandit (tests)          | PyCQA/bandit     | 1.7.9     | `^src/tests/`              | Security   |
| 19  | markdownlint            | markdownlint-cli | v0.42.0   | `*.md` excl notes/docs     | Docs       |
| 20  | shellcheck              | shellcheck-py    | v0.10.0.1 | `*.sh`, `*.bash`           | Scripts    |
| 21  | yamllint                | yamllint         | v1.35.1   | yaml files                 | Config     |
| 22  | pytest-unit             | local            | —         | all python                 | Testing    |
| 23  | coverage-gate           | local            | —         | all python                 | Coverage   |

---

## Appendix D: Environment Variables

| Variable              | ci.yml                      | scheduled-tests.yml | Purpose                                               |
| --------------------- | --------------------------- | ------------------- | ----------------------------------------------------- |
| `ENV_NAME`            | —                           | Removed             | Was conda env name (removed in Phase 1 pip migration) |
| `PYTHON_TEST_VERSION` | `"3.13"`                    | `"3.13"`            | Default Python for non-matrix jobs                    |
| `COVERAGE_FAIL_UNDER` | `"80"`                      | —                   | Coverage enforcement threshold                        |
| `CASCOR_LOG_LEVEL`    | —                           | —                   | Set to WARNING in conftest.py                         |
| `SKIP`                | `pytest-unit,coverage-gate` | —                   | Skip local pre-commit hooks in CI                     |
| `GITHUB_TOKEN`        | via `secrets`               | —                   | Gitleaks authentication                               |
