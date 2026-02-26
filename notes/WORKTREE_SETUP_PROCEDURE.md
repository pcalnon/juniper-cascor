# Worktree Setup Procedure

**Purpose**: Standardized procedure for creating a git worktree when beginning a new task
**Project**: juniper-cascor
**Last Updated**: 2026-02-25

---

## Why Worktrees

Git worktrees provide task isolation by allowing multiple branches to be checked out simultaneously in separate directories. This keeps the main working directory on the default branch while task work proceeds in a dedicated checkout, preventing accidental commits to `main` and enabling easy context switching between tasks.

For the Juniper ecosystem, all worktrees are centralized in `/home/pcalnon/Development/python/Juniper/worktrees/`.

---

## Prerequisites

- You must be in the repo's primary working directory (not already in a worktree)
- The working tree must be clean (no uncommitted changes)
- The target branch must not already be checked out in another worktree

---

## Setup Protocol

### Step 1: Ensure Clean State

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-cascor
git status
```

**GATE**: Working tree must be clean. If dirty, stash or commit before proceeding:
```bash
git stash push -m "WIP before worktree setup"
```

### Step 2: Fetch and Update Parent Branch

```bash
git fetch origin
git checkout main
git pull origin main
```

### Step 3: Create the Working Branch

```bash
BRANCH_NAME="feature/my-task-name"    # Set to your branch name
PARENT_BRANCH="main"                   # Usually main; change if needed

git branch "$BRANCH_NAME" "$PARENT_BRANCH"
```

**Important**: Do NOT use `git checkout -b`. The branch is created but the main working directory stays on the parent branch.

### Step 4: Compute the Worktree Directory Name

```bash
REPO_NAME=$(basename "$(pwd)")
SAFE_BRANCH=$(echo "$BRANCH_NAME" | sed 's|/|--|g')
TIMESTAMP=$(date +%Y%m%d-%H%M)
SHORT_HASH=$(git rev-parse --short=8 HEAD)
WORKTREE_BASE="/home/pcalnon/Development/python/Juniper/worktrees"
WORKTREE_DIR="${WORKTREE_BASE}/${REPO_NAME}--${SAFE_BRANCH}--${TIMESTAMP}--${SHORT_HASH}"
```

### Step 5: Create the Worktree

```bash
git worktree add "$WORKTREE_DIR" "$BRANCH_NAME"
```

### Step 6: Verify and Begin Work

```bash
git worktree list
cd "$WORKTREE_DIR"
git branch --show-current
git log --oneline -3
```

---

## Naming Convention

**Format**: `<repo-name>--<branch-name>--<YYYYMMDD-HHMM>--<short-hash>`

| Component | Description | Example |
|-----------|-------------|---------|
| `<repo-name>` | Git repo directory name | `juniper-cascor` |
| `<branch-name>` | Branch name, slashes replaced with `--` | `feature--add-validation` |
| `<YYYYMMDD-HHMM>` | Creation timestamp (local time) | `20260225-1430` |
| `<short-hash>` | First 8 chars of HEAD SHA | `fb530aa1` |

**Full example**: `juniper-cascor--feature--add-validation--20260225-1430--fb530aa1`

**Location**: All worktrees reside in `/home/pcalnon/Development/python/Juniper/worktrees/`

---

## Edge Cases

### Branch Already Exists

```bash
# Check first:
git branch --list "$BRANCH_NAME"

# Option A: Use the existing branch (skip Step 3, proceed to Step 4)
# Option B: Delete and recreate from current parent
git branch -d "$BRANCH_NAME"    # safe delete (fails if unmerged)
git branch "$BRANCH_NAME" "$PARENT_BRANCH"
```

### Worktree Directory Already Exists

The timestamp + hash combination makes collisions extremely rare. If one occurs:
```bash
# Append a counter:
WORKTREE_DIR="${WORKTREE_DIR}-2"
```

### Working from a Remote Branch

If the branch exists on remote but not locally:
```bash
git checkout -b "$BRANCH_NAME" "origin/$BRANCH_NAME"
# Skip Step 3, proceed to Step 4.
```

---

## Project-Specific Notes

After creating the worktree and navigating to it, verify the environment:

```bash
cd "$WORKTREE_DIR"
conda activate JuniperPython

# Verify tests pass in the worktree
cd src/tests && bash scripts/run_tests.bash
```

---

## Quick Reference (Copy-Paste)

For experienced users who know the procedure:

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-cascor
git fetch origin && git checkout main && git pull origin main
BRANCH_NAME="feature/my-task"
git branch "$BRANCH_NAME" main
REPO_NAME=$(basename "$(pwd)")
SAFE_BRANCH=$(echo "$BRANCH_NAME" | sed 's|/|--|g')
WORKTREE_DIR="/home/pcalnon/Development/python/Juniper/worktrees/${REPO_NAME}--${SAFE_BRANCH}--$(date +%Y%m%d-%H%M)--$(git rev-parse --short=8 HEAD)"
git worktree add "$WORKTREE_DIR" "$BRANCH_NAME"
cd "$WORKTREE_DIR"
```
