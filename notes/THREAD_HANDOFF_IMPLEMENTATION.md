# Thread Handoff Implementation Guide

**Purpose**: Document how the Thread Handoff procedure was integrated into Claude Code's operating parameters, and provide step-by-step instructions for replicating this on any machine or project.

**Author**: Paul Calnon
**Last Updated**: 2026-02-20

---

## Overview

This document describes the implementation of a **Thread Handoff** process that replaces Claude Code's default thread compaction behavior. Thread compaction summarizes prior context to free token capacity, but introduces information loss. Thread handoff instead transfers a curated summary to a fresh thread with full context capacity, preserving critical details.

The handoff is triggered **automatically** when token utilization is within 1–5% of the threshold at which compaction would normally occur, ensuring the handoff completes before any lossy compaction happens.

---

## Architecture

Claude Code's behavior is controlled through a hierarchy of instruction files:

```
~/.claude/CLAUDE.md          ← Global: applies to ALL Claude Code sessions on this machine
<project>/CLAUDE.md           ← Project: applies to sessions in this project
<project>/notes/THREAD_HANDOFF_PROCEDURE.md  ← Protocol: templates and detailed handoff rules
```

The implementation uses **two layers**:

1. **Global layer** (`~/.claude/CLAUDE.md`): Defines the universal policy — all Claude Code instances must prefer handoff over compaction, with the trigger threshold and execution protocol.
2. **Project layer** (`<project>/CLAUDE.md`): Reinforces the policy for the specific project and references the project-specific `THREAD_HANDOFF_PROCEDURE.md` for templates and protocol details.

---

## What Was Modified

### 1. Project CLAUDE.md — New Section Added

**File**: `<project>/CLAUDE.md`
**Change**: Appended a new `## Thread Handoff (Mandatory — Replaces Thread Compaction)` section before the `## Contact` section.

**Content added**:
- Declares thread handoff as mandatory, overriding default compaction
- Defines the 95–99% pre-compaction trigger threshold (within 1–5% of compaction)
- Specifies the self-assessment rule: at each turn during multi-step work, assess proximity to compaction
- Lists additional triggers (15+ tool calls, phase boundaries, degraded recall, module transitions, user request)
- Lists exclusions (nearly complete task, sharp thread, tightly coupled work)
- Defines the 5-step execution protocol (checkpoint, compose, present, verify, git status)
- References `notes/THREAD_HANDOFF_PROCEDURE.md` for templates

### 2. Global ~/.claude/CLAUDE.md — Created

**File**: `~/.claude/CLAUDE.md`
**Change**: Created new file (did not previously exist).

**Content**:
- Declares the same mandatory handoff-over-compaction policy at global scope
- Defines the same trigger threshold and self-assessment rule
- Defines the same execution protocol
- Notes that project-specific `THREAD_HANDOFF_PROCEDURE.md` files should be followed when present
- Kept concise (~80 lines) to avoid bloating every session's context

### 3. THREAD_HANDOFF_PROCEDURE.md — Pre-existing (Unchanged)

**File**: `<project>/notes/THREAD_HANDOFF_PROCEDURE.md`
**Change**: None — this file already contained the full protocol, templates, and examples.

**Content** (for reference):
- Rationale for handoff over compaction
- Trigger conditions table
- 3-step handoff protocol (checkpoint, compose goal, execute)
- Goal composition rules and templates (implementation in progress, debugging, multi-phase)
- Best practices
- Full examples

---

## Replication Guide — How to Apply This on Another Machine or Project

### Prerequisites

- Claude Code installed and configured
- A project with a `CLAUDE.md` file (or willingness to create one)

### Step 1: Create the Thread Handoff Procedure Document

Create `notes/THREAD_HANDOFF_PROCEDURE.md` (or equivalent location) in your project. You can copy the one from the Juniper Cascor project, or write your own following this structure:

```markdown
# Thread Handoff Procedure

## Why Handoff Over Compaction
[Explain the information loss problem]

## When to Initiate a Handoff
[Table of trigger conditions: context saturation, phase boundary, degraded recall, etc.]

## Handoff Protocol
### Step 1: Checkpoint Current State
### Step 2: Compose the Handoff Goal
### Step 3: Execute the Handoff

## Handoff Goal Templates
[Project-specific templates for common handoff scenarios]

## Best Practices
[Guidelines for effective handoffs]
```

### Step 2: Add the Thread Handoff Section to Your Project's CLAUDE.md

Append the following section to your project's `CLAUDE.md` (before any closing sections like Contact):

```markdown
## Thread Handoff (Mandatory — Replaces Thread Compaction)

> **CRITICAL OPERATING INSTRUCTION**: Thread handoff MUST be performed instead of
> thread compaction. This section defines mandatory behavior that overrides Claude
> Code's default compaction process.

### What This Is

When a long-running conversation approaches context limits, Claude Code normally
performs **thread compaction** — summarizing prior context to free token capacity.
This introduces information loss. Instead, Claude Code instances working on this
project MUST perform a **proactive thread handoff**: transferring a curated,
high-signal summary to a fresh thread with full context capacity.

The full handoff protocol is defined in **`notes/THREAD_HANDOFF_PROCEDURE.md`**.
Read that file when a handoff is triggered.

### When to Trigger a Handoff

**Automatic trigger (pre-compaction threshold):** Initiate a thread handoff when
token utilization reaches **95% to 99%** of the level at which thread compaction
would normally be triggered. This means the handoff fires when you are within
**1% to 5%** of the compaction threshold, ensuring the handoff completes before
compaction would occur.

Concretely:
- If compaction would trigger at N% context utilization, begin handoff at
  (N − 5)% to (N − 1)%.
- **Self-assessment rule**: At each turn where you are performing multi-step work,
  assess whether you are approaching the compaction threshold. If you estimate you
  are within 5% of it, begin the handoff protocol immediately.
- When the system compresses prior messages or you receive a context compression
  notification, treat this as a signal that handoff should have already occurred —
  immediately initiate one.

**Additional triggers** (from `notes/THREAD_HANDOFF_PROCEDURE.md`):

| Condition                   | Indicator                                            |
| --------------------------- | ---------------------------------------------------- |
| **Context saturation**      | 15+ tool calls or 5+ files edited                    |
| **Phase boundary**          | Logical phase of work is complete                    |
| **Degraded recall**         | Re-reading files or re-asking resolved questions     |
| **Multi-module transition** | Moving between major components                      |
| **User request**            | User says "hand off", "new thread", or similar       |

**Do NOT handoff** when:
- Task is nearly complete (< 2 remaining steps)
- Current thread is still sharp and producing correct output
- Work is tightly coupled and splitting would lose in-flight state

### How to Execute a Handoff

1. **Checkpoint**: Inventory what was done, what remains, what was discovered,
   and what files are in play
2. **Compose the handoff goal**: Write a concise, actionable summary
   (see templates in `notes/THREAD_HANDOFF_PROCEDURE.md`)
3. **Present to user**: Output the handoff goal and recommend starting a new
   thread with that goal as the initial prompt
4. **Include verification commands**: Specify how the new thread should verify
   its starting state
5. **State git status**: Mention branch, staged files, and uncommitted work

### Rules

- **This is not optional.** Every Claude Code instance on this project must
  follow these rules.
- **Handoff early, not late.** A handoff at 70% context is better than
  compaction at 95%.
- **Do not duplicate CLAUDE.md content** in the handoff goal.
- **Be specific**: Include file paths, decisions made, and test status.
```

### Step 3: Create or Update the Global ~/.claude/CLAUDE.md

Create `~/.claude/CLAUDE.md` to apply the handoff policy to **all** Claude Code sessions on the machine, regardless of project:

```markdown
# Global Claude Code Instructions

## Thread Handoff (Mandatory — Replaces Thread Compaction)

> **CRITICAL OPERATING INSTRUCTION**: Thread handoff MUST be performed instead of
> thread compaction across ALL projects.

### Policy

When approaching context limits, perform a **proactive thread handoff** instead
of allowing thread compaction. Handoff transfers a curated summary to a fresh
thread with full context capacity.

### When to Trigger

Initiate when token utilization reaches **95% to 99%** of the compaction
threshold (within **1% to 5%** of compaction).

**Self-assessment rule**: At each turn during multi-step work, assess proximity
to compaction. If within 5%, begin handoff immediately.

**Additional triggers**:
- 15+ tool calls or 5+ files edited
- Logical phase of work complete
- Degraded recall (re-reading files, re-asking questions)
- Moving between major codebase components
- User request

**Do NOT handoff** when task is nearly complete, thread is sharp, or work is
tightly coupled.

### How to Execute

1. Checkpoint: what was done, what remains, what was discovered, files in play
2. Compose handoff goal (concise, actionable, <500 words)
3. Present to user and recommend new thread
4. Include verification commands
5. State git status

If the project has `notes/THREAD_HANDOFF_PROCEDURE.md`, follow its templates.
```

### Step 4: Verify the Integration

After creating/modifying the files:

1. **Start a new Claude Code session** in the project
2. **Confirm CLAUDE.md is loaded**: Claude Code automatically reads CLAUDE.md at session start. You can verify by asking "What are the thread handoff rules?"
3. **Test with a long session**: Perform a multi-step task and verify that Claude Code proactively offers a handoff before context gets compressed

---

## File Inventory

| File | Scope | Purpose |
| ---- | ----- | ------- |
| `~/.claude/CLAUDE.md` | Global (all projects) | Universal handoff-over-compaction policy |
| `<project>/CLAUDE.md` | Project | Project-specific handoff instructions with procedure reference |
| `<project>/notes/THREAD_HANDOFF_PROCEDURE.md` | Project | Full protocol, templates, examples |
| `<project>/notes/THREAD_HANDOFF_IMPLEMENTATION.md` | Project | This document — implementation record and replication guide |

---

## Design Decisions

### Why 1–5% Pre-Compaction Threshold?

The handoff must complete **before** compaction fires. Composing a handoff goal requires several tool calls (reading recent state, composing the summary, presenting it). Starting at 5% before compaction gives enough room to:
- Assess current state (1-2 tool calls)
- Compose the handoff goal (text generation)
- Present it to the user (1 output)

Starting at 1% is the minimum — it risks compaction firing mid-handoff if the assessment takes too many tokens.

### Why Both Global and Project-Level Instructions?

- **Global** ensures the policy applies even to projects without their own THREAD_HANDOFF_PROCEDURE.md
- **Project-level** adds project-specific templates and references the detailed procedure document
- The two layers are complementary, not contradictory — both say the same thing, with the project level adding specificity

### Why Not a Hook or Plugin?

Claude Code hooks execute shell commands in response to tool events. They can't observe token utilization or trigger behavioral changes like "compose a handoff goal." The CLAUDE.md instruction approach works because:
- It's read at session start and influences all subsequent behavior
- It defines behavioral rules that the model follows as part of its system prompt
- It requires no external tooling or runtime dependencies

### Why Present to User Instead of Auto-Spawning?

Claude Code's `handoff()` tool (referenced in the procedure document) may not be available in all environments. The fallback — presenting the handoff goal as text and recommending the user start a new thread — works universally. If `handoff()` becomes available, the procedure document can be updated to use it.

---

## Changelog

| Date | Change |
| ---- | ------ |
| 2026-02-20 | Initial implementation: added section to project CLAUDE.md, created global ~/.claude/CLAUDE.md, created this document |
