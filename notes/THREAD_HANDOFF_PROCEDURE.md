# Thread Handoff Procedure

**Purpose**: Preserve context fidelity during long-running Amp sessions by proactively handing off to a new thread before context compaction degrades output quality.

**Last Updated**: 2026-02-20

---

## Why Handoff Over Compaction

Thread compaction summarizes prior context to free token capacity. This introduces **information loss** — subtle details about decisions made, edge cases discovered, partial progress, and the reasoning behind specific implementation choices get compressed or dropped. For complex, multi-step work on JuniperCascor (e.g., multi-file refactors, new feature implementation across modules, debugging cascading issues), this degradation can cause:

- Repeated mistakes the thread already resolved
- Inconsistent code style mid-task
- Loss of discovered constraints or gotchas
- Re-reading files that were already understood

A **proactive handoff** transfers a curated, high-signal summary to a fresh thread with full context capacity, preserving the critical information while discarding the noise.

---

## When to Initiate a Handoff

Trigger a handoff when **any** of the following conditions are met:

| Condition                   | Indicator                                                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Context saturation**      | Thread has performed 15+ tool calls or edited 5+ files                                                                      |
| **Phase boundary**          | A logical phase of work is complete (e.g., planning done → implementation starting; implementation done → testing starting) |
| **Degraded recall**         | The agent re-reads a file it already read, or asks a question it already resolved                                           |
| **Multi-module transition** | Moving from one major component to another (e.g., `cascade_correlation/` → `candidate_unit/` → `tests/`)                    |
| **User request**            | User says "hand off", "new thread", "continue in a fresh thread", or similar                                                |

**Do NOT handoff** when:

- The task is nearly complete (< 2 remaining steps)
- The current thread is still sharp and producing correct output
- The work is tightly coupled and splitting would lose critical in-flight state

---

## Handoff Protocol

### Step 1: Checkpoint Current State

Before initiating the handoff, mentally inventory:

1. **What was the original task?** (user's request, verbatim or paraphrased)
2. **What has been completed?** (files created, files edited, tests passed/failed)
3. **What remains?** (specific next steps, not vague summaries)
4. **What was discovered?** (gotchas, constraints, decisions, rejected approaches)
5. **What files are in play?** (paths of files read, modified, or relevant)

### Step 2: Compose the Handoff Goal

Write a **concise, actionable** goal for the new thread. Structure it as:

```bash
Continue [TASK DESCRIPTION].

Completed so far:
- [Concrete item 1]
- [Concrete item 2]

Remaining work:
- [Specific next step 1]
- [Specific next step 2]

Key context:
- [Important discovery or constraint]
- [File X was modified to do Y]
- [Approach Z was rejected because...]
```

**Rules for the goal**:

- **Be specific**: "Add validation to `CandidateUnit.train()` for empty tensors" not "finish the validation work"
- **Include file paths**: The new thread doesn't know what you've been looking at
- **State decisions made**: So the new thread doesn't re-litigate them
- **Mention test status**: If tests were run, state pass/fail counts
- **Keep it under ~500 words**: Dense signal, no filler

### Step 3: Execute the Handoff

Use the `handoff` tool:

```bash
handoff(
    goal="<composed goal from Step 2>",
    follow=true
)
```

- Set `follow=true` when the current thread should stop and work continues in the new thread (the common case).
- Set `follow=false` only if the current thread has independent remaining work (rare).

---

## Handoff Goal Templates

### Template: Implementation In Progress

```bash
Continue implementing [FEATURE] in JuniperCascor.

Completed:
- Created [file1] with [description]
- Modified [file2] to [change description]
- Tests in [test_file] pass (X/Y passing)

Remaining:
- Implement [specific method/class]
- Add tests for [specific behavior]
- Update constants in cascor_constants/ if needed

Key context:
- Using [pattern/approach] because [reason]
- [File X] has a constraint: [detail]
- Run tests with: cd src/tests && bash scripts/run_tests.bash
```

### Template: Debugging Session

```bash
Continue debugging [ISSUE DESCRIPTION] in JuniperCascor.

Findings so far:
- Root cause is likely in [file:line] because [evidence]
- Ruled out: [rejected hypothesis 1], [rejected hypothesis 2]
- Reproduced with: [command or test]

Next steps:
- Verify hypothesis by [specific action]
- Apply fix in [file]
- Run [specific test] to confirm

Key context:
- The bug manifests as [symptom]
- Related code path: [file1] → [file2] → [file3]
```

### Template: Multi-Phase Task (Phase Transition)

```bash
Continue [OVERALL TASK] — starting Phase [N]: [PHASE NAME].

Phase [N-1] ([PREV PHASE NAME]) completed:
- [Deliverable 1]
- [Deliverable 2]
- All tests passing: cd src/tests && bash scripts/run_tests.bash

Phase [N] scope:
- [Step 1]
- [Step 2]
- [Step 3]

Key context from prior phases:
- [Decision or discovery that affects this phase]
- [File modified in prior phase that this phase depends on]
```

---

## Best Practices

1. **Handoff early, not late** — A handoff at 70% context usage is better than compaction at 95%
2. **One handoff per phase boundary** — Don't chain 5 handoffs for one task; batch related work
3. **Include the verification command** — Always tell the new thread how to check its work (`bash scripts/run_tests.bash`, `mypy`, etc.)
4. **Reference AGENTS.md** — The new thread will read it automatically, but call out any project-specific conventions relevant to the remaining work
5. **Don't duplicate AGENTS.md content** — The new thread already has it; only include task-specific context
6. **State the git status** — If files are staged, modified, or if a branch is in use, mention it

---

## Integration with Project Workflow

This procedure complements the existing development workflow in AGENTS.md. When a thread handoff occurs during feature development:

- The new thread should verify it can run tests before making changes
- The new thread should re-read any file it plans to edit (fresh context, no assumptions)
- If the handoff crosses a major boundary (e.g., code → documentation), the new thread should check the relevant `notes/` files for consistency

---

## Examples

### Example 1: Simple Handoff

> *Thread has implemented a new method in `cascade_correlation.py` and its tests, but still needs to add the corresponding constants and update the plotter.*

```bash
Continue adding the new `get_training_summary()` feature to JuniperCascor.

Completed:
- Added `get_training_summary()` method to CascadeCorrelationNetwork in
  src/cascade_correlation/cascade_correlation.py (returns TrainingSummary dataclass)
- Added TrainingSummary dataclass to same file
- Added unit tests in src/tests/unit/test_training_summary.py (6/6 passing)

Remaining:
- Add TRAINING_SUMMARY_DEFAULT_FORMAT constant to src/cascor_constants/constants_model/
- Add summary visualization to src/cascor_plotter/cascor_plotter.py (new plot_training_summary method)
- Add integration test exercising summary through full fit() cycle

Key context:
- TrainingSummary uses @dataclass(frozen=True) for immutability
- Summary includes per-phase metrics (output training + each candidate round)
- Verify with: cd src/tests && bash scripts/run_tests.bash -v
```

### Example 2: Handoff After Discovery

> *Thread was asked to optimize candidate training but discovered a deeper issue.*

```bash
Continue optimizing candidate training in JuniperCascor. Initial investigation
revealed the bottleneck is not in training but in correlation calculation.

Findings:
- Profiled with py-spy (util/profile_training.bash --svg)
- 68% of candidate training time is in _calculate_correlation() in
  src/candidate_unit/candidate_unit.py, specifically the tensor operations
  on lines 145-162
- The current implementation recomputes mean/std on every call instead
  of caching across the epoch

Remaining:
- Refactor _calculate_correlation() to cache running mean/std
- Benchmark before/after with: cd src/tests/scripts && bash run_benchmarks.bash
- Ensure all correlation tests still pass: pytest -m correlation -v

Key context:
- Do NOT change the public API of CandidateUnit
- The correlation values must remain numerically identical (use torch.allclose in tests)
- Multiprocessing pickling means cached state must be in __getstate__/__setstate__
```
