# Issue Tracking Template

**Purpose:** This template defines the required structure and formatting for all JuniperCanopy issue and bug tracking reports.

**Usage:** Copy this template and replace placeholder text (indicated by `[PLACEHOLDER]`) with actual issue information. Update sections as the issue progresses from Open to Resolved.

**Naming Convention:** `[FIX|BUG|ISSUE]_[SHORT_NAME]_[YYYY-MM-DD].md` (e.g., `FIX_TESTING_ISSUES_2026-01-05.md`)

---

# [Issue|Bug|Fix]: [SHORT_DESCRIPTIVE_TITLE]

**Date:** [YYYY-MM-DD]  
**Author:** [REPORTER_NAME]  
**Version:** [AFFECTED_VERSION_OR_RANGE]  
**Status:** [OPEN|IN_PROGRESS|IN_REVIEW|RESOLVED|CLOSED]

---

## Summary

[ONE_OR_TWO_SENTENCES_DESCRIBING_THE_ISSUE]

---

## Environment

- **Environment:** [LOCAL|DEV|STAGING|PROD|DEMO_MODE]
- **App version:** [X.Y.Z]
- **Python version:** [X.Y.Z]
- **OS / Platform:** [E.G._UBUNTU_25.10_X64]
- **Browser (if UI):** [E.G._CHROME_XXX_OR_N/A]
- **Configuration / Flags:** [RELEVANT_SETTINGS_OR_ENV_VARS]

---

## Severity & Priority

- **Severity:** [BLOCKER|CRITICAL|MAJOR|MINOR|TRIVIAL]
- **Priority:** [P0|P1|P2|P3]
- **Impact scope:** [ESTIMATE_OF_AFFECTED_USERS_OR_SYSTEMS]

### Severity Definitions

- **BLOCKER:** System unusable, no workaround, blocks release
- **CRITICAL:** Major functionality broken, limited workaround
- **MAJOR:** Significant impact on functionality or user experience
- **MINOR:** Minor impact, easy workaround available
- **TRIVIAL:** Cosmetic issue or minor inconvenience

### Priority Legend

- **P0:** Critical - Must fix immediately, blocks deployment
- **P1:** High - Must fix before release, high user impact
- **P2:** Medium - Should fix, medium priority
- **P3:** Low - Nice to fix, low priority

---

## Problem Description

[DETAILED_DESCRIPTION_OF_THE_PROBLEM_AND_CONTEXT]

### Observed Behavior

[WHAT_CURRENTLY_HAPPENS]

### Expected Behavior

[WHAT_SHOULD_HAPPEN]

---

## Steps to Reproduce

1. [STEP_1]
2. [STEP_2]
3. [STEP_3]
4. [OBSERVE_ISSUE]

**Reproducibility:** [ALWAYS|OFTEN|SOMETIMES|RARELY|UNKNOWN]

**Minimal reproduction case:** [CODE_SNIPPET_OR_COMMANDS_IF_APPLICABLE]

```bash
# Example commands to reproduce
[REPRODUCTION_COMMANDS]
```

---

## Error Output

<!-- Include relevant error messages, logs, or stack traces -->

```bash
[ERROR_MESSAGE_OR_STACK_TRACE]
```

---

## Attachments / Logs

<!-- Optional: Include supporting materials -->

- Screenshots: [LINK_OR_N/A]
- Log files: [LINK_OR_INLINE_SNIPPETS]
- Related dashboards / traces: [LINKS]

---

## Root Cause Analysis

<!-- Complete this section once the root cause is identified -->

### Root Cause

[DESCRIPTION_OF_THE_UNDERLYING_CAUSE]

### Contributing Factors

- [FACTOR_1]
- [FACTOR_2]

### Why Not Caught Earlier?

[GAPS_IN_TESTS_MONITORING_OR_REVIEW_PROCESS]

---

## Solution / Fix Details

<!-- Complete this section once the fix is implemented -->

### Fix Approach

[HIGH_LEVEL_DESCRIPTION_OF_THE_FIX_STRATEGY]

### Code Changes

**Before:**

```python
# Old problematic code
[OLD_CODE]
```

**After:**

```python
# Fixed code
[NEW_CODE]
```

### Files Changed

| File              | Lines     | Description           |
| ----------------- | --------- | --------------------- |
| [PATH/TO/FILE]    | [L#-L#]   | [BRIEF_DESCRIPTION]   |

### Config / Data Changes

- [CONFIG_CHANGE_1_OR_N/A]
- [CONFIG_CHANGE_2_OR_N/A]

### Backport Required?

- **Backport needed:** [YES|NO]
- **Target versions:** [VERSIONS_OR_N/A]

---

## Verification & Tests

### Test Summary

| Test Type   | Result         | Notes                      |
| ----------- | -------------- | -------------------------- |
| Unit        | [PASS/FAIL/N/A]| [NOTES]                    |
| Integration | [PASS/FAIL/N/A]| [NOTES]                    |
| E2E         | [PASS/FAIL/N/A]| [NOTES]                    |
| Manual      | [PASS/FAIL/N/A]| [SCENARIOS_TESTED]         |

### Test Results After Fix

```bash
[TEST_EXECUTION_OUTPUT]
```

### Tests Added

- [TEST_FILE_1]: [NUMBER] tests for [DESCRIPTION]
- [TEST_FILE_2]: [NUMBER] tests for [DESCRIPTION]

### Verification Checklist

- [ ] Issue reproduces on affected version before fix
- [ ] Issue no longer reproduces with fix
- [ ] No regression in [RELATED_AREA_1]
- [ ] No regression in [RELATED_AREA_2]
- [ ] Tests added to prevent recurrence
- [ ] Monitoring/alerts updated if needed
- [ ] Documentation updated if needed
- [ ] Release notes updated (if user-visible)

---

## Regression Risk & Mitigation

<!-- Optional: Include for high-risk fixes -->

- **Risk level:** [LOW|MEDIUM|HIGH]
- **Affected components:** [COMPONENTS_OR_MODULES]
- **Mitigations:**
  - [ADDITIONAL_TESTS]
  - [FEATURE_FLAGS]
  - [MONITORING]
  - [ROLLBACK_PLAN]

---

## Linked Work

### Related Issues

### Pull Requests

### Release Information

- **Affected versions:** [VERSION_RANGE]
- **Fixed in version:** [X.Y.Z]
- **Released on:** [YYYY-MM-DD]

---

## Lessons Learned

<!-- Optional: Include insights that may prevent similar issues -->

1. **[LESSON_CATEGORY_1]:** [DESCRIPTION_OF_LESSON]

2. **[LESSON_CATEGORY_2]:** [DESCRIPTION_OF_LESSON]

---

## Timeline

<!-- Optional: Track issue lifecycle -->

| Date       | Event                    | Notes                      |
| ---------- | ------------------------ | -------------------------- |
| [DATE]     | Issue reported           | [BY_WHOM]                  |
| [DATE]     | Root cause identified    | [BRIEF_NOTES]              |
| [DATE]     | Fix implemented          | [PR_REFERENCE]             |
| [DATE]     | Fix verified             | [TEST_RESULTS]             |
| [DATE]     | Released in v[VERSION]   | [RELEASE_NOTES_LINK]       |

---

## Notes for Release

<!-- Optional: Pre-formatted snippet for release notes -->

### Fixed

- [USER_FRIENDLY_DESCRIPTION_OF_FIX] (fixes [ISSUE-XXX])
