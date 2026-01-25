# Release Notes Template

**Purpose:** This template defines the required structure and formatting for all JuniperCanopy standard release notes.

**Usage:** Copy this template and replace placeholder text (indicated by `[PLACEHOLDER]`) with actual release information. For security patch releases, use `TEMPLATE_SECURITY_RELEASE_NOTES.md` instead.

**Naming Convention:** `RELEASE_NOTES_v[VERSION].md` (e.g., `RELEASE_NOTES_v0.23.0.md`, `RELEASE_NOTES_v0.24.0-alpha.md`)

**Location:** Save release notes to `notes/releases/` directory.

**Related Templates:**

- `TEMPLATE_SECURITY_RELEASE_NOTES.md` – For security patch releases
- `TEMPLATE_PULL_REQUEST_DESCRIPTION.md` – For accompanying PR descriptions
- `TEMPLATE_ISSUE_TRACKING.md` – For bug/issue documentation

---

# Juniper Canopy v[VERSION] Release Notes

**Release Date:** [YYYY-MM-DD]  
**Version:** [X.Y.Z] or [X.Y.Z-STAGE] (e.g., 0.23.0, 0.24.0-alpha)  
**Codename:** [OPTIONAL_CODENAME] (e.g., Phase 3 Complete, Refactor Verified)  
**Release Type:** [PATCH|MINOR|MAJOR|ALPHA|BETA]

---

## Overview

[ONE_TO_THREE_SENTENCES_DESCRIBING_THIS_RELEASE_AND_ITS_PRIMARY_GOALS]

> **Status:** [ALPHA|BETA|STABLE] – [BRIEF_STATUS_DESCRIPTION]

---

## Release Summary

- **Release type:** [PATCH|MINOR|MAJOR]
- **Primary focus:** [E.G._BUG_FIXES|NEW_FEATURES|PERFORMANCE|STABILITY|VERIFICATION|DOCUMENTATION]
- **Breaking changes:** [YES|NO]
- **Priority summary:** [E.G._P0_BUGS_FIXED|KEY_P1_FEATURES_DELIVERED|ALL_PHASES_COMPLETE]

---

## Features Summary

<!-- Use for releases with multiple features. Include cumulative phase status for milestone releases. -->

| ID     | Feature                | Status   | Version | Phase |
| ------ | ---------------------- | -------- | ------- | ----- |
| [P#-N] | [FEATURE_NAME]         | ✅ Done  | [X.Y.Z] | [0-3] |
| [P#-N] | [FEATURE_NAME]         | Planned  | -       | [0-3] |

<!-- Optional: Cumulative Phase Status (for milestone releases) -->

**Cumulative Phase Status:**

| Phase | Items | Status |
| ----- | ----- | ------ |
| Phase 0: [PHASE_TITLE] | [N] items | ✅ Complete |
| Phase 1: [PHASE_TITLE] | [N] items | ✅ Complete |
| Phase 2: [PHASE_TITLE] | [N] items | 🔄 In Progress |
| Phase 3: [PHASE_TITLE] | [N] items | 📋 Planned |

---

## What's New

### [FEATURE_CATEGORY_1] (e.g., HDF5 Snapshot Management)

#### [FEATURE_NAME] ([FEATURE_ID])

[DETAILED_DESCRIPTION_OF_THE_FEATURE]

**Backend:**

- [BACKEND_CHANGE_1]
- [BACKEND_CHANGE_2]

**Frontend:**

- [FRONTEND_CHANGE_1]
- [FRONTEND_CHANGE_2]

**API Endpoint(s):**

- `[METHOD] [/api/v1/endpoint]` – [DESCRIPTION]

<!-- Repeat for additional features -->

---

## Bug Fixes

### [BUG_TITLE] (v[VERSION])

**Problem:** [DESCRIPTION_OF_THE_BUG]

**Root Cause:** [EXPLANATION_OF_WHY_THE_BUG_OCCURRED]

**Solution:** [DESCRIPTION_OF_THE_FIX]

**Files:** [FILES_CHANGED_WITH_LINE_NUMBERS]

<!-- Repeat for additional bug fixes -->

---

## Improvements

### [IMPROVEMENT_CATEGORY] (e.g., Test Coverage, Performance)

[DESCRIPTION_OF_THE_IMPROVEMENT]

| Component | Before | After | Change |
| --------- | ------ | ----- | ------ |
| [FILE]    | [N]%   | [N]%  | +[N]%  |

<!-- Optional: Test count growth for multi-version releases -->

### Test Count Growth

| Version | Tests | Change |
| ------- | ----- | ------ |
| [X.Y.Z] | [N]   | —      |
| [X.Y.Z] | [N]   | +[N]   |

**Total new tests since v[BASELINE]:** [N] tests

---

## API Changes

### New Endpoints

| Method   | Endpoint                    | Description           |
| -------- | --------------------------- | --------------------- |
| `[POST]` | `/api/v1/[endpoint]`        | [DESCRIPTION]         |

### Changed Endpoints

<!-- Optional: Include if existing endpoints changed -->

| Method   | Endpoint                    | Change Type | Description | Breaking? |
| -------- | --------------------------- | ----------- | ----------- | --------- |
| `[GET]`  | `/api/v1/[endpoint]`        | [CHANGED]   | [DETAILS]   | [YES/NO]  |

### Response Codes

**[METHOD] [ENDPOINT]:**

- `[CODE] [STATUS]` – [DESCRIPTION]
- `[CODE] [STATUS]` – [DESCRIPTION]

---

## Test Results

### Test Suite

| Metric            | Result                    |
| ----------------- | ------------------------- |
| **Tests passed**  | [N]                       |
| **Tests skipped** | [N]                       |
| **Tests failed**  | [N]                       |
| **Runtime**       | [N] seconds               |
| **Coverage**      | [N]% overall              |

### Coverage Details

| Component | Coverage | Target | Status           |
| --------- | -------- | ------ | ---------------- |
| [FILE]    | [N]%     | 95%    | ✅ Exceeded      |
| [FILE]    | [N]%     | 95%    | ✅ Met           |
| [FILE]    | [N]%     | 95%    | ⚠️ Near target   |

---

## Upgrade Notes

<!-- Required for MINOR/MAJOR releases; Optional for PATCH -->

This is a backward-compatible release. [NO_MIGRATION_STEPS_REQUIRED | MIGRATION_STEPS_BELOW]

```bash
# Update and verify
git pull origin main
./demo

# Run test suite
cd src && pytest tests/ -v
```

### Migration Steps

<!-- If migration is required -->

1. [MIGRATION_STEP_1]
2. [MIGRATION_STEP_2]
3. [MIGRATION_STEP_3]

### Rollback Instructions

<!-- If rollback may be needed -->

```bash
[ROLLBACK_COMMANDS]
```

---

## Known Issues

<!-- Required section. If none, state explicitly. -->

- [ISSUE_SUMMARY] – [WORKAROUND_OR_N/A] (see [ISSUE-XXX])
- [ISSUE_SUMMARY] – Expected fix in v[VERSION]

<!-- Common patterns for known issues -->

- **[COMPONENT] coverage at [N]%:** [EXPLANATION_WHY_NOT_HIGHER]. Not a functional issue.
- **Documentation drift:** [BRIEF_DESCRIPTION]. To be addressed in future maintenance.

<!-- If no known issues -->
None known at time of release.

---

## What's Next

### Planned for v[NEXT_VERSION]

- [UPCOMING_FEATURE_1] – [BRIEF_DESCRIPTION]
- [UPCOMING_FEATURE_2] – [BRIEF_DESCRIPTION]

### Coverage Goals

- [FILE] currently at [N]%, target [N]%

### Roadmap

See [Development Roadmap](../../DEVELOPMENT_ROADMAP.md) for full details.

---

## Contributors

- [CONTRIBUTOR_NAME_1]
- [CONTRIBUTOR_NAME_2]

---

## Version History

| Version    | Date       | Description               |
| ---------- | ---------- | ------------------------- |
| [X.Y.Z]    | [DATE]     | [BRIEF_DESCRIPTION]       |
| [X.Y.Z-1]  | [DATE]     | [BRIEF_DESCRIPTION]       |

---

## Links

- [Full Changelog](../../CHANGELOG.md)
- [Development Roadmap](../../DEVELOPMENT_ROADMAP.md)
- [Phase Documentation](../../docs/phase[N]/README.md)
- [Verification Report](../development/[VERIFICATION_REPORT_FILE]) <!-- Optional: for milestone releases -->
- [Pull Request Details](../pull_requests/[PR_FILE])
- [Previous Release](RELEASE_NOTES_v[PREVIOUS_VERSION].md)

---

<!-- Template Maintenance Notes:
     - Last Updated: 2026-01-11
     - Added: Cumulative Phase Status table, Test Count Growth section
     - Added: Common Known Issues patterns, Related Templates section
     - Added: Location guidance, Verification Report link
     - Fixed: Relative paths to match actual project structure
-->
