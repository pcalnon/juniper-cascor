# Dependency Update Workflow — juniper-cascor

**Last Updated:** 2026-10-05
**Version:** 1.1.2
**Status:** Current

---

## Overview

This document describes how dependency updates flow through juniper-cascor, from Dependabot PR to merged locks. `pyproject.toml` uses `>=` ranges. `requirements.lock` pins the GPU/dev resolution. `requirements-cpu.lock` is that lock minus the CUDA stack and is what the service image installs. Torch itself stays out of both files; the image pin is `ARG TORCH_VERSION` in the `Dockerfile`.

Primary workflow: `.github/workflows/lockfile-update.yml`  
Enforcement gate: `lockfile-check` ("Lockfile Freshness") in `.github/workflows/ci.yml`

## Automated Flow (Dependabot)

When Dependabot opens a PR to update a dependency:

```
1. Dependabot pushes to dependabot/pip/<package-or-group> branch
2. lockfile-update.yml triggers on push to dependabot/pip/**
   - Job guard: github.actor == 'dependabot[bot]' (push path)
   - PAT gate (see below) decides whether to auto-regen
   - When proceeding: recompile both locks (recipe below) and land one GitHub-signed commit `[dependabot skip] Update requirements.lock and requirements-cpu.lock`
   - The commit is `createCommitOnBranch` with `CROSS_REPO_DISPATCH_TOKEN`, so CI re-triggers and the commit is signed. A local unsigned commit on the same branch blocks merge; squash does not remove it (`notes/JUNIPER_2026-08-12_JUNIPER-CASCOR_BRANCH-PROTECTION-VALIDATION.md`).
3. CI runs on the updated branch
   - Lockfile Freshness verifies requirements.lock still satisfies pyproject.toml
   - Other quality gates run normally
4. Review and merge the Dependabot PR
```

The same workflow also runs on `pull_request` when `pyproject.toml` changes on a same-repo branch (manual min-version bumps). Fork PRs are skipped (they cannot push back with the PAT).

### PAT availability gate (`CROSS_REPO_DISPATCH_TOKEN`)

Dependabot-triggered runs use the **Dependabot secret store** (`Secret source: Dependabot`), not the Actions repository secrets. A repo-scoped `CROSS_REPO_DISPATCH_TOKEN` that exists under **Settings → Secrets and variables → Actions** is therefore **empty** on Dependabot runs unless the same PAT is also registered under **Settings → Secrets → Dependabot**.

| Condition | Gate result | Operator meaning |
|-----------|-------------|------------------|
| PAT present (non-empty) | Proceed — checkout with PAT, regen, push | Full auto-regen (unchanged happy path) |
| PAT absent **and** `github.actor == dependabot[bot]` | Loud **green no-op** (`::notice::`, `proceed=false`) | Auto-regen skipped; Lockfile Freshness still blocks stale locks |
| PAT absent **and** non-Dependabot actor | Hard fail (`::error::`, exit 1) | Secret misconfiguration — fix before merge |

**Optional restore of Dependabot auto-regen:** copy/register `CROSS_REPO_DISPATCH_TOKEN` under Dependabot secrets. No workflow change required.

Source: gate step in `.github/workflows/lockfile-update.yml` (ported from juniper-canopy #476; cascor #428).

### First CI run / green no-op

- **With PAT available to the run:** the first CI push may still race a stale lock for a few seconds; the lockfile-update commit cancels in-progress CI and the follow-up run passes.
- **Without PAT on Dependabot:** the Update Lockfile job stays green and leaves the locks untouched. Expect **Lockfile Freshness** to fail until someone regenerates both locks locally (or registers the Dependabot PAT and re-runs / rebases).

## Manual Flow (Editing pyproject.toml)

When you manually edit dependency ranges in `pyproject.toml`, regenerate **both** locks with the recipe in [Regenerating both locks](#regenerating-both-locks), then commit `pyproject.toml`, `requirements.lock`, and `requirements-cpu.lock` together.

Same-repo PRs that touch `pyproject.toml` also trigger `lockfile-update.yml` (subject to the PAT gate). Prefer committing a fresh pair with the pyproject change so CI is green even if the auto-regen arm no-ops. Sign the commit; an unsigned commit anywhere on the branch blocks merge.

Verify freshness the way `ci.yml` does (constraint mode, pin lines only):

```bash
uv pip compile pyproject.toml \
  --extra ml --extra api --extra observability --extra juniper-data \
  --index-strategy unsafe-best-match --no-emit-package torch \
  --constraint requirements.lock \
  -o /tmp/check.lock
diff <(grep '^[^[:space:]#]' requirements.lock | sort) \
     <(grep '^[^[:space:]#]' /tmp/check.lock | sort)
```

## Regenerating both locks

Source of truth: `.github/workflows/lockfile-update.yml` (the two compile steps). The compile input is `pyproject.toml`. `conf/requirements.txt`, `conf/requirements-pip.txt`, `conf/requirements_ci.txt`, and `conf/conda_environment_ci.yaml` stay as Dependabot and the conda export left them.

```bash
# GPU / dev lock. A fresh -o path is required: uv treats an existing -o file
# as extra constraints and keeps the old pins (workflow comment on the CPU step).
uv pip compile pyproject.toml \
  --extra ml \
  --extra api \
  --extra observability \
  --extra juniper-data \
  --index-strategy unsafe-best-match \
  --no-emit-package torch \
  -o /tmp/requirements.lock.check
mv /tmp/requirements.lock.check requirements.lock

# CPU / image lock, derived from the GPU lock. Refuse when the header torch
# pin disagrees with the Dockerfile, then splice the hand-written header back
# (uv replaces it with the command line).
TORCH_VERSION="$(sed -n 's/^ARG TORCH_VERSION=//p' Dockerfile)"
HEADER_VERSION="$(sed -n 's/.*torch==\([0-9][0-9.]*\)+cpu.*/\1/p' requirements-cpu.lock | head -1)"
test -n "${TORCH_VERSION}" && test "${TORCH_VERSION}" = "${HEADER_VERSION}"
echo "torch==${TORCH_VERSION}+cpu" > /tmp/torch-cpu-override
awk '/^[^[:space:]#]/{exit} {print}' requirements-cpu.lock > /tmp/cpu-lock-header
uv pip compile pyproject.toml \
  --extra ml \
  --extra api \
  --extra observability \
  --extra juniper-data \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --no-emit-package torch \
  --override /tmp/torch-cpu-override \
  --constraint requirements.lock \
  --python-version 3.14 \
  -o /tmp/requirements-cpu.lock.check
awk 'f || /^[^[:space:]#]/{f = 1; print}' /tmp/requirements-cpu.lock.check > /tmp/cpu-lock-body
cat /tmp/cpu-lock-header /tmp/cpu-lock-body > requirements-cpu.lock
```

| Flag | Purpose |
|------|---------|
| `--extra ml` / `api` / `observability` / `juniper-data` | The four extras both locks and the image freshness check include |
| `--index-strategy unsafe-best-match` | Allow the PyTorch index alongside PyPI |
| `--no-emit-package torch` | Leave torch out of both locks. The image installs it from `ARG TORCH_VERSION` |
| Fresh `-o /tmp/requirements.lock.check` | Resolve current versions that satisfy `pyproject.toml`. An existing `requirements.lock` as `-o` freezes the previous pins |
| `--constraint requirements.lock` (CPU step only) | Shared pins stay identical to the GPU lock |
| `--override /tmp/torch-cpu-override` | `torch==${TORCH_VERSION}+cpu` so the CPU index resolves the CPU wheel |
| `--extra-index-url https://download.pytorch.org/whl/cpu` | CPU wheel index for the image lock |
| `--python-version 3.14` | Match the image interpreter (`python:3.14-slim`) |

**What one Dependabot PR then contains.** Dependabot edits the conf freeze it was configured to touch. The skip commit, when the PAT is present, re-resolves every package the ranges allow and writes that resolution into both locks. The two files can disagree inside the same PR. Observed on [#701](https://github.com/pcalnon/juniper-cascor/pull/701) (`python-minor`, commits `f2a0cf3` then `1ba0c03`):

| Package | Conf freeze after Dependabot | Both locks after the skip commit |
|---------|------------------------------|----------------------------------|
| `filelock` | `4.0.9` (`conf/requirements.txt`, `conf/requirements-pip.txt`, `conf/requirements_ci.txt`; was `4.0.7`) | `4.0.11` (was `4.0.9` on `main`) |
| `websockets` | `17.1` unchanged in `conf/requirements-pip.txt` and `conf/requirements_ci.txt`. Not in the eight-package table | `17.2` (was `17.1`) |
| `torch` | `2.14.1` in those three conf files (was `2.14.0`) | Absent (`--no-emit-package torch`). Image pin stays `ARG TORCH_VERSION=2.14.0`, matching the `requirements-cpu.lock` header |

`conf/conda_environment_ci.yaml` is a third freeze (on `main`: `filelock==3.29.0`, `websockets==16.0`). This job does not update it.

**Image torch pin.** Bump `ARG TORCH_VERSION` in the `Dockerfile` and the `torch==X.Y.Z+cpu` line in the CPU lock header together, then re-run the recipe. The regen step exits 1 when they differ (`src/tests/unit/test_dockerfile_cpu_torch_pin.py` pins the same pair). A torch line in a conf freeze does not move the image.

**What CI checks afterwards.**

| Gate | What it compares | What still passes while files disagree |
|------|------------------|----------------------------------------|
| Lockfile Freshness (`ci.yml`) | `requirements.lock` pin lines vs a `--constraint requirements.lock` recompile | Newer PyPI releases. `requirements-cpu.lock` versions. Conf freezes |
| CPU name check (same job) | Every image dependency name from `pyproject.toml` (torch omitted) is present in `requirements-cpu.lock` | Version skew between the two locks. Conf freezes |

The freshness error string names only `uv pip compile ... --upgrade -o requirements.lock`. That refreshes the GPU lock when `--upgrade` is honored, and it leaves `requirements-cpu.lock` on the previous resolution. The paired recipe above is the one the workflow runs.

## Troubleshooting

### Lockfile check fails in CI

**Symptom:** `Lockfile Freshness` fails with "requirements.lock no longer satisfies pyproject.toml"

**Cause:** `pyproject.toml` (or Dependabot range edits) drifted without a matching lock regen — including the Dependabot green no-op when the PAT is missing from the Dependabot secret store.

**Fix:** Run [Regenerating both locks](#regenerating-both-locks) and commit both lockfiles (signed). Or register `CROSS_REPO_DISPATCH_TOKEN` under Dependabot secrets and `@dependabot rebase`.

### Lockfile-update workflow green but no auto-commit

**Symptom:** Dependabot PR has a green "Update requirements.lock" job, but no `[dependabot skip]` commit and Lockfile Freshness is red.

**Cause:** PAT gate took the Dependabot no-op path (`HAVE_PAT=false`).

**Fix:**
1. Confirm the notice in the gate step log about Dependabot secret store
2. Register `CROSS_REPO_DISPATCH_TOKEN` under **Settings → Secrets → Dependabot**, **or**
3. Regenerate both locks with the recipe above and push a signed commit to the Dependabot branch

```bash
gh secret list -R pcalnon/juniper-cascor | grep CROSS_REPO_DISPATCH_TOKEN
gh run list --workflow=lockfile-update.yml -R pcalnon/juniper-cascor
```

### Lockfile-update hard-fails on a human PR

**Symptom:** `::error::CROSS_REPO_DISPATCH_TOKEN is missing for a non-Dependabot run`

**Cause:** Actions secret missing/expired while a same-repo PR touched `pyproject.toml`.

**Fix:** Restore the Actions repository secret (not only the Dependabot store), or commit a manually regenerated lock pair and temporarily avoid relying on auto-push.

### Lockfile-update workflow doesn't trigger

**Symptom:** No Update Lockfile run at all

**Possible causes:**
1. Branch name doesn't match `dependabot/pip/**` (push path)
2. Event is a fork PR (skipped by design)
3. Change did not touch `pyproject.toml` and was not a Dependabot push
4. Workflow file has a syntax error

### Merge conflict in requirements.lock

**Symptom:** Dependabot PR shows merge conflict in `requirements.lock`

**Fix:** Check out the Dependabot branch and regenerate both locks with [the recipe above](#regenerating-both-locks). Lockfiles should be regenerated, not hand-merged. Commit `requirements.lock` and `requirements-cpu.lock` together, signed. The workflow's own message is `[dependabot skip] Update requirements.lock and requirements-cpu.lock`.

## ASGI / WebSocket transport reviews (`websockets`, `uvicorn`)

`websockets` is **not** a direct `pyproject.toml` dependency. It is installed by `uvicorn[standard]` (API extra) and appears in `requirements.lock` as `# via uvicorn`. Application handlers under `src/api/websocket/` use FastAPI/Starlette `WebSocket` only — there is no `import websockets` in `src/`.

When Dependabot opens a PR that bumps `websockets` (including major lines such as 16.1.x → 17.x):

| Check | Why |
|-------|-----|
| Python floor still ≥ 3.12 | `websockets` 17 requires ≥ 3.11; this repo is already stricter (`requires-python`) |
| No new direct `websockets` imports in `src/` | Keep the transport boundary at uvicorn |
| `requirements.lock`, `requirements-cpu.lock`, and the conf freezes | The skip commit re-resolves from `pyproject.toml` and can move a transitive pin (for example `websockets`) that Dependabot's table did not list, past the conf pin. See [#701](https://github.com/pcalnon/juniper-cascor/pull/701) |
| `conf/conda_environment_ci.yaml` noted if it still pins an older line | Conda freeze is maintained separately and can lag |
| WebSocket suites green | `tests/unit/api/test_websocket_*.py`, `test_ws_heartbeat.py`, `tests/integration/api/test_websocket_streaming.py` |

Operator-facing detail: [ASGI WebSocket transport](../docs/api/JUNIPER_CASCOR_API_REFERENCE.md#asgi-websocket-transport).

## GitHub Actions Dependabot (`codeql-action` group)

Pip lockfiles are only half of Dependabot. `.github/dependabot.yml` also has a `github-actions` ecosystem (weekly Monday, `open-pull-requests-limit: 3`, labels `dependencies` / `ci`, commit prefix `ci`).

The only named group is `codeql-action`, matching `github/codeql-action*`. That is why a CodeQL bump PR also edits `ci.yml`: Bandit SARIF upload uses `github/codeql-action/upload-sarif` and must stay on the same SHA as `codeql.yml`'s `init` / `autobuild` / `analyze`.

| Constraint | Why |
|------------|-----|
| Review all four `codeql-action` uses in one PR | Split pins leave CodeQL analyze and Bandit SARIF on different action versions |
| Do not expect `workflow_dispatch` on CodeQL | `codeql.yml` has none; `security-scan.yml` does |
| A PR targeting `develop` will not run CodeQL | `pull_request.branches` is `[main]` only |

Operator-facing detail: [CI Manual — CodeQL Analysis](../docs/ci_cd/MANUAL.md#codeql-analysis).

## Related Documentation

- [CI/CD Quick Start — Dependabot lockfile](../docs/ci_cd/QUICK_START.md#dependabot-lockfile-updates)
- [CI/CD Quick Start — CodeQL](../docs/ci_cd/QUICK_START.md#codeql-and-github-actions-dependabot)
- [CI/CD Manual — Lockfile Update](../docs/ci_cd/MANUAL.md#lockfile-update-workflow)
- [CI/CD Manual — CodeQL Analysis](../docs/ci_cd/MANUAL.md#codeql-analysis)
- [CI/CD Reference — Lockfile Update](../docs/ci_cd/REFERENCE.md#lockfile-update-workflow)
- [CI/CD Reference — CodeQL Analysis](../docs/ci_cd/REFERENCE.md#codeql-analysis)
- Workflow source: `.github/workflows/lockfile-update.yml`, `.github/workflows/codeql.yml`
- Freshness gate: `.github/workflows/ci.yml` job `lockfile-check`
