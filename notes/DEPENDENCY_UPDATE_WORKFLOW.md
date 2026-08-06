# Dependency Update Workflow — juniper-cascor

**Last Updated:** 2026-08-05
**Version:** 1.1.0
**Status:** Current

---

## Overview

This document describes how dependency updates flow through juniper-cascor, from Dependabot PR to merged lockfile. The lockfile (`requirements.lock`) pins exact versions for Docker builds while `pyproject.toml` uses `>=` ranges for library compatibility.

Primary workflow: `.github/workflows/lockfile-update.yml`  
Enforcement gate: `lockfile-check` ("Lockfile Freshness") in `.github/workflows/ci.yml`

## Automated Flow (Dependabot)

When Dependabot opens a PR to update a dependency:

```
1. Dependabot pushes to dependabot/pip/<package-or-group> branch
2. lockfile-update.yml triggers on push to dependabot/pip/**
   - Job guard: github.actor == 'dependabot[bot]' (push path)
   - PAT gate (see below) decides whether to auto-regen
   - When proceeding: uv pip compile → commit "[dependabot skip] Update requirements.lock"
   - Push uses CROSS_REPO_DISPATCH_TOKEN so CI re-triggers
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
- **Without PAT on Dependabot:** the Update Lockfile job stays green but does not commit. Expect **Lockfile Freshness** to fail until someone regenerates `requirements.lock` locally (or registers the Dependabot PAT and re-runs / rebases).

## Manual Flow (Editing pyproject.toml)

When you manually edit dependency ranges in `pyproject.toml`:

```bash
# 1. Edit pyproject.toml with your changes

# 2. Regenerate the lockfile
uv pip compile pyproject.toml \
  --extra ml \
  --extra api \
  --extra observability \
  --extra juniper-data \
  --index-strategy unsafe-best-match \
  --no-emit-package torch \
  --upgrade \
  -o requirements.lock

# 3. Verify the lockfile is fresh (same constraint check CI uses)
uv pip compile pyproject.toml \
  --extra ml --extra api --extra observability --extra juniper-data \
  --index-strategy unsafe-best-match --no-emit-package torch \
  --constraint requirements.lock \
  -o /tmp/check.lock
# Compare pin lines only (ignore uv header / -c annotations)
diff <(grep '^[^[:space:]#]' requirements.lock | sort) \
     <(grep '^[^[:space:]#]' /tmp/check.lock | sort)

# 4. Commit both files together
git add pyproject.toml requirements.lock
git commit -m "Update <package> to <version>"
```

Same-repo PRs that touch `pyproject.toml` also trigger `lockfile-update.yml` (subject to the PAT gate). Prefer committing a fresh lock with the pyproject change so CI is green even if the auto-regen arm no-ops.

## Compile Command Reference

```bash
uv pip compile pyproject.toml \
  --extra ml \
  --extra api \
  --extra observability \
  --extra juniper-data \
  --index-strategy unsafe-best-match \
  --no-emit-package torch \
  --upgrade \
  -o requirements.lock
```

| Flag | Purpose |
|------|---------|
| `--extra ml` | Include numpy, h5py, matplotlib, and ML dependencies |
| `--extra api` | Include FastAPI, uvicorn, and API dependencies |
| `--extra observability` | Include Prometheus and structured logging dependencies |
| `--extra juniper-data` | Include juniper-data-client dependency |
| `--index-strategy unsafe-best-match` | Allow PyTorch index alongside PyPI |
| `--no-emit-package torch` | Exclude torch from lockfile (installed separately via CPU index in CI/Docker) |
| `--upgrade` | Refresh pins when regenerating after a range bump |
| `-o requirements.lock` | Output file |

**Lockfile Freshness model:** CI recompiles with `--constraint requirements.lock` and diffs resolved `pkg==version` pin lines. It does **not** fail merely because newer versions exist on PyPI — only when `pyproject.toml` drifted past what the lock can satisfy.

## Troubleshooting

### Lockfile check fails in CI

**Symptom:** `Lockfile Freshness` fails with "requirements.lock no longer satisfies pyproject.toml"

**Cause:** `pyproject.toml` (or Dependabot range edits) drifted without a matching lock regen — including the Dependabot green no-op when the PAT is missing from the Dependabot secret store.

**Fix:** Run the compile command above (with `--upgrade`) and commit the updated lockfile. Or register `CROSS_REPO_DISPATCH_TOKEN` under Dependabot secrets and `@dependabot rebase`.

### Lockfile-update workflow green but no auto-commit

**Symptom:** Dependabot PR has a green "Update requirements.lock" job, but no `[dependabot skip]` commit and Lockfile Freshness is red.

**Cause:** PAT gate took the Dependabot no-op path (`HAVE_PAT=false`).

**Fix:**
1. Confirm the notice in the gate step log about Dependabot secret store
2. Register `CROSS_REPO_DISPATCH_TOKEN` under **Settings → Secrets → Dependabot**, **or**
3. Regenerate `requirements.lock` locally and push to the Dependabot branch

```bash
gh secret list -R pcalnon/juniper-cascor | grep CROSS_REPO_DISPATCH_TOKEN
gh run list --workflow=lockfile-update.yml -R pcalnon/juniper-cascor
```

### Lockfile-update hard-fails on a human PR

**Symptom:** `::error::CROSS_REPO_DISPATCH_TOKEN is missing for a non-Dependabot run`

**Cause:** Actions secret missing/expired while a same-repo PR touched `pyproject.toml`.

**Fix:** Restore the Actions repository secret (not only the Dependabot store), or commit a manually regenerated lock and temporarily avoid relying on auto-push.

### Lockfile-update workflow doesn't trigger

**Symptom:** No Update Lockfile run at all

**Possible causes:**
1. Branch name doesn't match `dependabot/pip/**` (push path)
2. Event is a fork PR (skipped by design)
3. Change did not touch `pyproject.toml` and was not a Dependabot push
4. Workflow file has a syntax error

### Merge conflict in requirements.lock

**Symptom:** Dependabot PR shows merge conflict in `requirements.lock`

**Fix:** Regenerate from scratch — lockfiles should never be manually merged:
```bash
git checkout dependabot/pip/<branch>
uv pip compile pyproject.toml \
  --extra ml --extra api --extra observability --extra juniper-data \
  --index-strategy unsafe-best-match --no-emit-package torch \
  --upgrade \
  -o requirements.lock
git add requirements.lock
git commit -m "[dependabot skip] Regenerate requirements.lock"
git push
```

## Related Documentation

- [CI/CD Quick Start — Dependabot lockfile](../docs/ci_cd/QUICK_START.md#dependabot-lockfile-updates)
- [CI/CD Manual — Lockfile Update](../docs/ci_cd/MANUAL.md#lockfile-update-workflow)
- [CI/CD Reference — Lockfile Update](../docs/ci_cd/REFERENCE.md#lockfile-update-workflow)
- Workflow source: `.github/workflows/lockfile-update.yml`
- Freshness gate: `.github/workflows/ci.yml` job `lockfile-check`
