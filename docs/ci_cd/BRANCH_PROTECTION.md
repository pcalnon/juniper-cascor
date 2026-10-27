# Branch Protection Rules - Juniper Cascor

**Version**: 0.4.3
**Last Updated**: 2026-10-26
**Author**: Paul Calnon

---

## Overview

This document records the branch protection in force on the Juniper Cascor repository. It is implemented as two repository **rulesets** on the default branch, not as classic branch protection: `gh api repos/pcalnon/juniper-cascor/branches/main/protection` answers `404 Branch not protected`. Every value below was read from the live rulesets on **2026-10-26**. The rulesets are authoritative; where this page and the API disagree, this page is stale. Re-derive it with:

```bash
# Every ruleset, then each one's full definition
gh api repos/pcalnon/juniper-cascor/rulesets --jq '.[] | "\(.id) \(.name) \(.target) \(.enforcement)"'
gh api repos/pcalnon/juniper-cascor/rulesets/15081045
gh api repos/pcalnon/juniper-cascor/rulesets/21019063

# The rules in force on main, aggregated from every ruleset
gh api repos/pcalnon/juniper-cascor/rules/branches/main --jq '.[] | "\(.type) \(.ruleset_id)"'
```

| Ruleset                  | ID         | Applies to                 | Rules |
| ------------------------ | ---------- | -------------------------- | ----- |
| `juniper-cascor-rules`   | `15081045` | `~DEFAULT_BRANCH` (`main`) | `required_signatures`, `pull_request`, `required_status_checks`, `code_scanning`, `code_quality`, `deletion`, `non_fast_forward`, `creation` |
| `juniper-no-direct-push` | `21019063` | `~DEFAULT_BRANCH` (`main`) | `pull_request` |

When rulesets overlap, GitHub aggregates them and the most restrictive version of each rule applies; classic branch protection, if any were added, would layer on top in the same way ([About rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets#about-rule-layering)).

---

## Protected Branches

Protection applies to one branch:

| Branch      | Protection Level                 | Description                                   |
| ----------- | -------------------------------- | --------------------------------------------- |
| `main`      | **Strict** (both rulesets)       | Default branch; production-ready code         |

There is no `develop` branch (`gh api repos/pcalnon/juniper-cascor/branches/develop` answered `404 Branch not found` on 2026-10-26), and neither ruleset matches any branch but the default one. A pull request whose base is any other branch is governed by **no ruleset**: `gh api repos/pcalnon/juniper-cascor/rules/branches/<branch> --jq length` (a `/` in the name URL-encoded as `%2F`) returns `0`, so it has no required checks and merges with nothing having run. `Guard PR base branch`, a required check on `main`, runs on every PR and fails such a PR, so that it is at least visibly red (`AGENTS.md`, "PR base-branch guard (required check)").

---

## Required Status Checks

### Must Pass Before Merge

`juniper-cascor-rules` requires these **24** contexts (read 2026-10-26). A context is matched by its exact check-run name, so renaming a job, or a matrix leg, silently stops it satisfying the requirement and leaves every PR waiting on a check that will never report:

| Context (exact name)                                   | Workflow                   | Job id                    | In Quality Gate `needs:` |
| ------------------------------------------------------ | -------------------------- | ------------------------- | ------------------------ |
| `Pre-commit (Python 3.12)`                             | `ci.yml`                   | `pre-commit`              | Yes                      |
| `Pre-commit (Python 3.13)`                             | `ci.yml`                   | `pre-commit`              | Yes                      |
| `Pre-commit (Python 3.14)`                             | `ci.yml`                   | `pre-commit`              | Yes                      |
| `Documentation Links`                                  | `ci.yml`                   | `docs`                    | Yes                      |
| `Unit Tests + Coverage (Python 3.12 on ubuntu-latest)` | `ci.yml`                   | `unit-tests`              | Yes                      |
| `Unit Tests + Coverage (Python 3.13 on ubuntu-latest)` | `ci.yml`                   | `unit-tests`              | Yes                      |
| `Unit Tests + Coverage (Python 3.14 on ubuntu-latest)` | `ci.yml`                   | `unit-tests`              | Yes                      |
| `Unit Tests + Coverage (Python 3.12 on macos-latest)`  | `ci.yml`                   | `unit-tests`              | Yes                      |
| `Quick Integration Tests`                              | `ci.yml`                   | `quick-integration-tests` | Yes                      |
| `Full Integration Tests`                               | `ci.yml`                   | `integration-tests`       | Yes                      |
| `Build Package`                                        | `ci.yml`                   | `build`                   | Yes                      |
| `Dependency Documentation`                             | `ci.yml`                   | `dependency-docs`         | Yes                      |
| `Security Scans`                                       | `ci.yml`                   | `security`                | Yes                      |
| `Lockfile Freshness`                                   | `ci.yml`                   | `lockfile-check`          | Yes                      |
| `Docker Build & Smoke Test`                            | `ci.yml`                   | `docker-build`            | Yes                      |
| `Quality Gate`                                         | `ci.yml`                   | `required-checks`         | (the gate itself)        |
| `Async-route audit (BUG-JD-10 class)`                  | `ci.yml`                   | `async-route-audit`       | No                       |
| `Memory Budget`                                        | `ci.yml`                   | `memory-budget`           | No                       |
| `Golden / Snapshot Regression`                         | `golden-regression.yml`    | `golden-regression`       | No                       |
| `model-core Conformance`                               | `conformance.yml`          | `conformance`             | No                       |
| `Analyze (python)`                                     | `codeql.yml`               | `analyze`                 | No                       |
| `Sequence Safety`                                      | `sequence-safety.yml`      | `sequence-safety`         | No                       |
| `Guard PR base branch`                                 | `pr-base-branch-guard.yml` | `guard-base-branch`       | No                       |
| `Bandit`                                               | code scanning (see below)  | —                         | No                       |

**A green Quality Gate does not mean mergeable.** Eight of the 24 are outside its `needs:`: two standalone `ci.yml` jobs, five jobs in other workflows (a `needs:` entry can only name a job in the same workflow), and `Bandit`. Each context is also pinned to the app that must post it (`integration_id`): GitHub Actions (`15368`) for 23 of them, and GitHub Advanced Security (`57789`) for `Bandit`, which is the code-scanning result of the Bandit SARIF that `ci.yml`'s `security` job uploads. Re-derive the list:

```bash
gh api repos/pcalnon/juniper-cascor/rules/branches/main --jq '.[] | select(.type=="required_status_checks") | .parameters.required_status_checks[] | "\(.context) \(.integration_id)"'
```

### Status Check Configuration

The `required_status_checks` rule's other parameters (read 2026-10-26):

| Setting                                          | Value                                    | API parameter                          |
| ------------------------------------------------ | ---------------------------------------- | -------------------------------------- |
| Require status checks to pass before merging     | ✅ Yes (the 24 contexts above)           | `required_status_checks`               |
| Require branches to be up to date before merging | ✅ Yes                                   | `strict_required_status_checks_policy` |
| Skip the status checks when creating the branch  | ✅ Yes                                   | `do_not_enforce_on_create`             |

The up-to-date policy means a PR that falls behind `main` must be updated, and its checks re-run, before it can merge.

---

## Pull Request Requirements

### For `main` Branch

Both rulesets carry a `pull_request` rule, and the more restrictive value of each parameter is the one in force. Read 2026-10-26; the last column is `juniper-cascor-rules` / `juniper-no-direct-push`:

| Setting                                      | Value          | API parameter                                                     |
| -------------------------------------------- | -------------- | ----------------------------------------------------------------- |
| Require a pull request before merging        | ✅ Yes         | a `pull_request` rule in both                                     |
| Required approving reviews                   | **0**          | `required_approving_review_count`: `0` / `0`                      |
| Dismiss stale pull request approvals         | ✅ Yes         | `dismiss_stale_reviews_on_push`: `true` / `false`                 |
| Require review from Code Owners              | ⚪ No          | `require_code_owner_review`: `false` / `false`                    |
| Require approval of the most recent push     | ⚪ No          | `require_last_push_approval`: `false` / `false`                   |
| Require conversation resolution              | ✅ Yes         | `required_review_thread_resolution`: `true` / `false`             |
| Allowed merge methods                        | merge, squash  | `allowed_merge_methods`: `merge`, `squash` / `merge`, `squash`, `rebase` |
| Require signed commits                       | ✅ Yes         | `required_signatures` rule (`juniper-cascor-rules`)               |

No approving review, and no code-owner review, is required to merge. Every review conversation must be resolved, and every commit must be signed.

### For `develop` Branch

Nothing applies: there is no `develop` branch, and both rulesets target only `~DEFAULT_BRANCH`. A `develop` branch created later would be unprotected until a ruleset names it.

---

## Additional Protections

### Enforce for Administrators

Bypass lists are per ruleset (read 2026-10-26):

| Ruleset                  | Bypass actors                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `juniper-cascor-rules`   | the Admin repository role and deploy keys (`always`); the release-train GitHub App, integration `4362741` (`pull_request` only) |
| `juniper-no-direct-push` | **none**                                                                                                            |

So "Do not allow bypassing the above settings" is **not** in force for `juniper-cascor-rules`: an admin can merge a PR whose required checks have not passed. Nobody can bypass `juniper-no-direct-push`, an admin included, so the pull request itself cannot be skipped.

### Restrict Who Can Push

| Setting                                          | `main`                       | Rule                                                     |
| ------------------------------------------------ | ---------------------------- | -------------------------------------------------------- |
| Push to `main` other than by merging a PR        | ❌ No, for everyone          | `pull_request` in `juniper-no-direct-push` (no bypass)   |
| Allow force pushes                               | ❌ No                        | `non_fast_forward`                                       |
| Allow deletions                                  | ❌ No                        | `deletion`                                               |
| Restrict creations                               | ✅ Yes                       | `creation`                                               |

---

## Coverage Enforcement

Coverage is enforced in CI by the `unit-tests` job. Pytest collects repository-wide coverage with `--cov=src`, then a standalone coverage step enforces the aggregate threshold:

| Threshold | Current | Target  | Enforcement      |
| --------- | ------- | ------- | ---------------- |
| Overall aggregate | 80% | 90%+ | Hard fail in CI |
| Per source file | >=90% | >=90% | Advisory until the final per-file gate PR |
| Per packaged sub-module | >=95% pooled | >=95% pooled | Advisory until the final per-file gate PR |

### Increasing Coverage Thresholds

As aggregate coverage improves, update the threshold in `.github/workflows/ci.yml`:

```yaml
env:
  COVERAGE_FAIL_UNDER: "80"  # Increase deliberately as coverage improves
```

The per-file rollout is tracked with `juniper-coverage-gap-map` from coverage JSON. Add the blocking `--enforce` step only after every source file and packaged sub-module clears the bars.

---

## Security Scanning

### Automated Security Checks

| Tool       | Purpose                        | Enforcement     |
| ---------- | ------------------------------ | --------------- |
| Gitleaks   | Secrets detection              | Hard fail (`ci.yml` Security Scans; skipped on `repository_dispatch`) |
| Bandit     | Python SAST (security issues)  | Hard fail on medium+; SARIF upload is best-effort (`continue-on-error`). Its code-scanning result is the **required** `Bandit` context, and the ruleset's `code_scanning` rule blocks on a Bandit `error` alert or a `high`+ security alert |
| pip-audit  | Dependency vulnerabilities     | Both `ci.yml` and `security-scan.yml` use `--strict`. `ci.yml` also passes documented torch `--ignore-vuln` IDs; `security-scan.yml` does not. |
| CodeQL     | Semantic SAST (Python)         | **Required** context `Analyze (python)`; the ruleset's `code_scanning` rule also blocks on an `error` alert or a `high`+ security alert |

CodeQL lives in `.github/workflows/codeql.yml` (push `main`/`develop`, PR `main`, Monday 06:00 UTC). The live ruleset requires its `Analyze (python)` context (listed under [Required Status Checks](#required-status-checks)). Findings go to **Security → Code scanning**; the workflow is standalone, so it is not in the Quality Gate `needs:`.

### Secrets Management

- **Never** commit secrets, API keys, or credentials
- Use GitHub Secrets for sensitive values
- Gitleaks runs on every push and PR

---

## Setting Up Branch Protection (Step-by-Step)

The protection already exists, as the two rulesets above. This section is how to inspect or change it; after any change, re-read the rulesets and update this page with the date you read them.

### Via GitHub Web UI

1. Go to **Repository** → **Settings** → **Rules** → **Rulesets**
2. Open `juniper-cascor-rules` or `juniper-no-direct-push`
3. Edit the rule (the required contexts are under **Require status checks to pass**)
4. Click **Save changes**
5. Re-derive the values with the commands in [Overview](#overview), and update this page

Do not add a rule under **Settings** → **Branches**: classic branch protection is not used here, and a classic rule would layer its own requirements on top of both rulesets.

### Via GitHub CLI

```bash
# Install GitHub CLI if needed
# https://cli.github.com/

# Read: one ruleset in full, and the rules in force on main
gh api repos/pcalnon/juniper-cascor/rulesets/15081045
gh api repos/pcalnon/juniper-cascor/rules/branches/main
```

A ruleset is changed through `PUT /repos/{owner}/{repo}/rulesets/{ruleset_id}`; the web UI above is the simpler path. Do **not** use `PUT /repos/{owner}/{repo}/branches/main/protection`. Until 2026-10-26 this page gave that command, with six status checks, one required approving review and a required code-owner review: run now, it would add a classic rule requiring that review and that code-owner review on top of both rulesets.

---

## Workflow for Contributors

### Feature Development

```bash
# 1. Create feature branch from main (there is no develop branch)
git checkout main
git pull origin main
git checkout -b feature/my-feature

# 2. Make changes and run pre-commit locally
pre-commit run --all-files

# 3. Commit and push (every commit must be signed: required_signatures)
git add .
git commit -S -m "feat: add my feature"
git push origin feature/my-feature

# 4. Create PR to main (Guard PR base branch fails any other base)
# - Wait for every required check to pass, on a branch up to date with main
# - Resolve every review conversation (no approving review is required)
# - Merge (merge commit or squash) when ready
```

### Release Process

```bash
# 1. Land the version bump and CHANGELOG on main through a PR, like any other change
# 2. Cut a GitHub Release tagged v<version>: merging to main publishes nothing
# 3. The Release fires publish.yml: TestPyPI, then PyPI once a reviewer approves the pypi environment
```

---

## Troubleshooting

### Common Issues

| Issue                           | Solution                                           |
| ------------------------------- | -------------------------------------------------- |
| Pre-commit fails locally        | Run `pre-commit run --all-files` and fix issues    |
| Coverage below threshold        | Add more tests or temporarily lower threshold      |
| Security scan blocks merge      | Review and fix security issues before merging      |
| Stale approval dismissed        | Nothing blocks: no approval is required to merge   |

### Emergency Bypass (Use Sparingly)

If you absolutely must bypass protections (emergencies only):

1. Open a pull request: nobody can bypass `juniper-no-direct-push`, so not even an admin can push to `main` directly
2. As an admin, merge it using the ruleset bypass: the Admin role bypasses `juniper-cascor-rules`, so its required checks need not have passed
3. Nothing needs re-enabling afterwards: no setting was changed
4. Document the bypass in the PR description

---

## References

- [GitHub Rulesets Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)
- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [GitHub CODEOWNERS Documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [GitHub Actions Status Checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks)
