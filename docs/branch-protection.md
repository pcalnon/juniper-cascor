# Branch Protection Rules - Juniper Cascor

**Version**: 0.4.0
**Last Updated**: 2026-01-29
**Author**: Paul Calnon

---

## Overview

This document describes the recommended branch protection rules for the Juniper Cascor repository to enforce code quality and prevent accidental breaks to protected branches.

---

## Protected Branches

Apply protection rules to the following branches:

| Branch      | Protection Level | Description                          |
| ----------- | ---------------- | ------------------------------------ |
| `main`      | **Strict**       | Production-ready code                |
| `develop`   | **Standard**     | Integration branch for features      |

---

## Required Status Checks

### Must Pass Before Merge

The following GitHub Actions checks **MUST** pass before a pull request can be merged:

| Check Name                           | Job                | Critical |
| ------------------------------------ | ------------------ | -------- |
| Pre-commit (Python 3.12)             | `pre-commit`       | ✅ Yes   |
| Pre-commit (Python 3.13)             | `pre-commit`       | ✅ Yes   |
| Unit Tests + Coverage                | `unit-tests`       | ✅ Yes   |
| Integration Tests                    | `integration-tests`| ✅ Yes   |
| Security Scans                       | `security`         | ✅ Yes   |
| Quality Gate                         | `required-checks`  | ✅ Yes   |

### Status Check Configuration

In GitHub repository settings → Branches → Add rule:

1. **Branch name pattern**: `main` (or `develop`)
2. **Require status checks to pass before merging**: ✅ Enabled
3. **Require branches to be up to date before merging**: ✅ Enabled
4. **Status checks that are required**:
   - `Pre-commit (Python 3.12)`
   - `Pre-commit (Python 3.13)`
   - `Unit Tests + Coverage (Python 3.14)`
   - `Integration Tests`
   - `Security Scans`
   - `Quality Gate`

---

## Pull Request Requirements

### For `main` Branch

| Setting                                      | Value     |
| -------------------------------------------- | --------- |
| Require a pull request before merging        | ✅ Yes    |
| Required approving reviews                   | 1         |
| Dismiss stale pull request approvals         | ✅ Yes    |
| Require review from Code Owners              | ✅ Yes    |
| Require approval of the most recent push     | ✅ Yes    |
| Require conversation resolution              | ✅ Yes    |

### For `develop` Branch

| Setting                                      | Value     |
| -------------------------------------------- | --------- |
| Require a pull request before merging        | ✅ Yes    |
| Required approving reviews                   | 1         |
| Dismiss stale pull request approvals         | ✅ Yes    |
| Require conversation resolution              | ✅ Yes    |

---

## Additional Protections

### Enforce for Administrators

| Setting                                      | `main`    | `develop` |
| -------------------------------------------- | --------- | --------- |
| Do not allow bypassing the above settings    | ✅ Yes    | ⚪ No     |

### Restrict Who Can Push

| Setting                                      | `main`    | `develop` |
| -------------------------------------------- | --------- | --------- |
| Restrict who can push to matching branches   | ✅ Yes    | ⚪ No     |
| Allow force pushes                           | ❌ No     | ❌ No     |
| Allow deletions                              | ❌ No     | ❌ No     |

---

## Coverage Enforcement

Coverage is enforced in CI with `pytest --cov-fail-under`:

| Threshold | Current | Target  | Enforcement      |
| --------- | ------- | ------- | ---------------- |
| Overall   | 50%     | 90%     | Hard fail in CI  |

### Increasing Coverage Thresholds

As coverage improves, update the threshold in `.github/workflows/ci.yml`:

```yaml
env:
  COVERAGE_FAIL_UNDER: "50"  # Increase to 60, 70, 80, 90 over time
```

---

## Security Scanning

### Automated Security Checks

| Tool       | Purpose                        | Enforcement     |
| ---------- | ------------------------------ | --------------- |
| Gitleaks   | Secrets detection              | Hard fail       |
| Bandit     | Python SAST (security issues)  | SARIF upload    |
| pip-audit  | Dependency vulnerabilities     | Warning         |

### Secrets Management

- **Never** commit secrets, API keys, or credentials
- Use GitHub Secrets for sensitive values
- Gitleaks runs on every push and PR

---

## Setting Up Branch Protection (Step-by-Step)

### Via GitHub Web UI

1. Go to **Repository** → **Settings** → **Branches**
2. Click **Add branch protection rule**
3. Enter **Branch name pattern**: `main`
4. Configure settings as described above
5. Click **Create** or **Save changes**
6. Repeat for `develop` branch

### Via GitHub CLI

```bash
# Install GitHub CLI if needed
# https://cli.github.com/

# Set up branch protection for main
gh api repos/{owner}/{repo}/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["Pre-commit (Python 3.12)","Pre-commit (Python 3.13)","Unit Tests + Coverage (Python 3.14)","Integration Tests","Security Scans","Quality Gate"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true,"require_code_owner_reviews":true}' \
  --field restrictions=null \
  --field allow_force_pushes=false \
  --field allow_deletions=false
```

---

## Workflow for Contributors

### Feature Development

```bash
# 1. Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/my-feature

# 2. Make changes and run pre-commit locally
pre-commit run --all-files

# 3. Commit and push
git add .
git commit -m "feat: add my feature"
git push origin feature/my-feature

# 4. Create PR to develop
# - Wait for all checks to pass
# - Get approval from code owner
# - Merge when ready
```

### Release Process

```bash
# 1. Create PR from develop to main
# 2. All checks must pass
# 3. Requires code owner approval
# 4. Merge creates production release
```

---

## Troubleshooting

### Common Issues

| Issue                           | Solution                                           |
| ------------------------------- | -------------------------------------------------- |
| Pre-commit fails locally        | Run `pre-commit run --all-files` and fix issues    |
| Coverage below threshold        | Add more tests or temporarily lower threshold      |
| Security scan blocks merge      | Review and fix security issues before merging      |
| Stale approval dismissed        | Request new review after pushing changes           |

### Emergency Bypass (Use Sparingly)

If you absolutely must bypass protections (emergencies only):

1. Temporarily disable "Do not allow bypassing" for administrators
2. Merge the emergency fix
3. **Immediately re-enable** the protection
4. Document the bypass in the PR description

---

## References

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [GitHub CODEOWNERS Documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [GitHub Actions Status Checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks)
