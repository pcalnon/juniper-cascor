# Worktree Cleanup Procedure

**Purpose**: Standardized procedure for merging, cleaning up, and pushing after completing a task in a worktree
**Project**: juniper-cascor
**Last Updated**: 2026-02-25

---

## Why Proper Cleanup Matters

Stale worktrees consume disk space and clutter the centralized `worktrees/` directory. Orphan branches pollute the branch namespace locally and on the remote. Proper cleanup after each task ensures a clean, predictable development environment.

---

## Prerequisites

- All work in the worktree is committed (no uncommitted changes)
- Tests pass in the worktree
- You know the worktree directory path and branch name

### Pre-Merge Verification

Before beginning cleanup, run the project's test suite in the worktree:

```bash
cd <worktree-dir>
conda activate JuniperPython
cd src/tests && bash scripts/run_tests.bash
```

**GATE**: All tests must pass before proceeding with merge.

---

## Cleanup Protocol

### Step 1: Ensure All Work Is Committed

```bash
cd "$WORKTREE_DIR"
git status
```

**GATE**: Working tree must be clean. Commit any remaining changes:
```bash
git add -A
git commit -m "final changes for <task>"
```

### Step 2: Push Working Branch to Remote

```bash
git push origin "$BRANCH_NAME"
```

This serves as a backup before merging.

### Step 3: Switch to the Main Repo Directory

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-cascor
```

### Step 4: Update the Merge Target

```bash
MERGE_TARGET="main"    # Default; change if merging into a different branch
git checkout "$MERGE_TARGET"
git pull origin "$MERGE_TARGET"
```

### Step 5: Merge the Working Branch

```bash
git merge "$BRANCH_NAME"
```

If merge conflicts arise, see the [Merge Conflicts](#merge-conflicts) section below.

### Step 6: Push the Merged Branch to Remote

```bash
git push origin "$MERGE_TARGET"
```

### Step 7: Remove the Worktree

```bash
git worktree remove "$WORKTREE_DIR"
```

### Step 8: Delete the Working Branch (Local)

```bash
git branch -d "$BRANCH_NAME"
```

The `-d` flag is safe: it refuses to delete if the branch is not fully merged. Only use `-D` if you intentionally want to discard unmerged work.

### Step 9: Delete the Working Branch (Remote)

```bash
git push origin --delete "$BRANCH_NAME"
```

### Step 10: Prune and Verify

```bash
git worktree prune
git worktree list          # Should show only the main working directory
git branch                 # Working branch should be gone
ls "$WORKTREE_DIR" 2>/dev/null && echo "WARNING: dir still exists" || echo "OK: removed"
```

---

## Alternate Merge Targets

By default, the working branch merges into `main`. To merge into a different branch, simply change the `MERGE_TARGET` variable:

```bash
MERGE_TARGET="develop"
# or
MERGE_TARGET="release/v0.7.0"
```

The procedure is identical; only the target variable changes. Use this when:
- The task is part of a larger feature branch
- The project uses a branching model with `develop` or `release` branches
- The task was branched from something other than `main`

---

## Edge Cases

### Merge Conflicts

After `git merge "$BRANCH_NAME"` fails with conflicts:

**Option A: Resolve in-place**
```bash
git status                          # See conflicted files
# Edit each conflicted file, remove conflict markers
git add <resolved-files>
git merge --continue
```

**Option B: Abort and create a pull request instead**
```bash
git merge --abort
gh pr create --base "$MERGE_TARGET" --head "$BRANCH_NAME" \
  --title "Merge $BRANCH_NAME into $MERGE_TARGET"
```

### Worktree Removal Fails

If the worktree is dirty or locked:
```bash
# Check for lock files:
git worktree list --porcelain

# Force remove if confident data is safe:
git worktree remove --force "$WORKTREE_DIR"

# If the directory persists after worktree removal:
rm -rf "$WORKTREE_DIR"
git worktree prune
```

### Branch Deletion Fails

If `git branch -d` refuses because git doesn't recognize the merge:
```bash
# Verify the merge actually completed:
git log --oneline "$MERGE_TARGET" | head -5

# If merged but git doesn't see it (common with squash merges):
git branch -D "$BRANCH_NAME"    # Force delete (use with caution)

# Remote delete may fail if branch protection is enabled:
# Handle via GitHub UI or adjust branch protection settings.
```

### Squash Merge Instead of Regular Merge

If the project prefers squash merges:
```bash
git merge --squash "$BRANCH_NAME"
git commit -m "feat: <description of squashed changes>"
```

Note: After squash merge, `git branch -d` will fail (git doesn't see the branch as merged). Use `git branch -D` instead.

---

## Quick Reference (Copy-Paste)

For experienced users who know the procedure:

```bash
# In the worktree: commit and push
cd "$WORKTREE_DIR"
git push origin "$BRANCH_NAME"

# In the main repo: merge, push, cleanup
cd /home/pcalnon/Development/python/Juniper/juniper-cascor
git checkout main && git pull origin main
git merge "$BRANCH_NAME"
git push origin main
git worktree remove "$WORKTREE_DIR"
git branch -d "$BRANCH_NAME"
git push origin --delete "$BRANCH_NAME"
git worktree prune
```
