# Juniper Pre-Deployment Roadmap - Phase 2

**Created**: 2026-01-25  
**Last Updated**: 2026-01-25 CST  
**Version**: 2.0.0  
**Status**: Active - Remaining Pre-Deployment Work  
**Author**: Development Team

---

## Executive Summary

This document consolidates all **remaining incomplete issues** from the original PRE-DEPLOYMENT_ROADMAP.md. Issues have been re-prioritized and organized into a new phased implementation approach focused on production deployment readiness.

### Original Roadmap Status

| Priority | Original Count | Complete | Remaining |
| -------- | -------------- | -------- | --------- |
| P0       | 6              | 6        | 0         |
| P1       | 11             | 11       | 0         |
| P2       | 3              | 2        | 1         |
| P3       | 5              | 3        | 2         |
| INTEG    | 5              | 2        | 3         |
| Other    | -              | -        | 15+       |

### Remaining Issues Summary

| New Priority | Count | Category                        |
| ------------ | ----- | ------------------------------- |
| P1-NEW       | 3     | Integration Architecture Issues |
| P2-NEW       | 6     | Code Quality & Coverage         |
| P3-NEW       | 4     | Profiling & Performance         |
| P4-NEW       | 6     | Documentation & Verification    |

---

## Table of Contents

1. [P1-NEW: Integration Architecture Issues](#1-p1-new-integration-architecture-issues)
2. [P2-NEW: Code Quality & Coverage](#2-p2-new-code-quality--coverage)
3. [P3-NEW: Profiling & Performance](#3-p3-new-profiling--performance)
4. [P4-NEW: Documentation & Verification](#4-p4-new-documentation--verification)
5. [Implementation Schedule](#5-implementation-schedule)
6. [Deferred Items](#6-deferred-items)

---

## 1. P1-NEW: Integration Architecture Issues

These issues affect the fundamental architecture of the Cascor-Canopy integration and should be addressed before production deployment.

### P1-NEW-001: No True IPC - Cascor Embedded in Canopy Process

**Original ID**: INTEG-001  
**Severity**: 🔶 HIGH  
**Status**: 📋 NOT STARTED  
**Effort**: XL (2-4 weeks)

**Problem**: The current architecture embeds Cascor within the Canopy process. There is no actual inter-process communication (IPC) between separately running Cascor and Canopy instances.

**Impact**:

- Cannot run Cascor training independently of Canopy frontend
- Cannot scale Cascor training on a different machine
- Cannot have multiple Canopy frontends observe a single Cascor training session
- Training failures crash the entire Canopy application

**Design Requirements**:

- Cascor and Canopy should be separate and distinct processes
- WebSocket implementation for real-time communication
- REST APIs for non-realtime communication

**Remediation Options**:

1. **Option A**: Add gRPC/REST API layer to Cascor for remote training control
2. **Option B**: Use Redis pub/sub for training state broadcasting
3. **Option C**: Implement shared memory or socket-based IPC

**Required Actions**:

- [ ] Consult Oracle for detailed architecture analysis
- [ ] Design IPC protocol specification
- [ ] Implement Cascor server mode
- [ ] Update Canopy to connect to external Cascor
- [ ] Add connection management and failover

---

### P1-NEW-002: RemoteWorkerClient Not Integrated with Canopy

**Original ID**: INTEG-002  
**Severity**: 🔶 HIGH  
**Status**: 📋 NOT STARTED  
**Effort**: L (1-2 weeks)

**Problem**: Cascor has a `RemoteWorkerClient` class for distributed training, but it is not used by Canopy. The distributed training capability exists but is not exposed.

**Design Requirements**:

- External hosts and processes must participate in candidate training queue
- RemoteWorkerClient should be OS-agnostic with minimal dependencies
- Consider packaging as sub-project: JuniperBranch / juniper_branch

**Target Platforms**:

- Ubuntu 25.10 (and recent versions)
- Raspberry Pi OS / Raspbian
- Debian
- Fedora/RockyLinux/AlmaLinux

**Required Actions**:

- [ ] Consult Oracle for RemoteWorkerClient analysis
- [ ] Design minimal worker package (juniper_branch)
- [ ] Document worker deployment procedure
- [ ] Integrate worker monitoring with Canopy
- [ ] Test distributed training across platforms

---

### P1-NEW-003: Blocking Training in FastAPI Async Context

**Original ID**: INTEG-004  
**Severity**: 🔶 MEDIUM  
**Status**: 📋 NOT STARTED  
**Effort**: M (2-4 days)

**Problem**: The Cascor `fit()` method is synchronous and blocking. When called from FastAPI/Uvicorn, it blocks the event loop, causing:

- Unresponsive WebSocket connections
- API timeout errors
- UI freeze during training

**Remediation Options**:

1. Run training in background thread (current demo_mode pattern)
2. Use `asyncio.run_in_executor()` to offload to thread pool
3. Implement async training loop with `asyncio.sleep()` yields

**Required Actions**:

- [ ] Evaluate current threading approach
- [ ] Implement proper async/thread boundary
- [ ] Add training progress callbacks
- [ ] Test WebSocket responsiveness during training

---

## 2. P2-NEW: Code Quality & Coverage

### P2-NEW-001: Cascor Code Coverage Below 90% Target

**Original ID**: CASCOR-P2-001  
**Status**: 🟡 IN PROGRESS  
**Current**: ~15-78% (varies by module)  
**Target**: 90%

**Coverage by Module**:

| Module                   | Current | Target | Gap  | Tests Needed |
|--------------------------|---------|--------|------|--------------|
| `cascade_correlation.py` | ~20%    | 85%    | 65%  | ~100 tests   |
| `candidate_unit.py`      | ~30%    | 90%    | 60%  | ~30 tests    |
| `snapshot_serializer.py` | 78%     | 90%    | 12%  | ~15 tests    |
| `log_config/`            | ~40%    | 80%    | 40%  | ~20 tests    |
| `utils/`                 | ~50%    | 80%    | 30%  | ~15 tests    |

**Required Actions**:

- [ ] Identify critical untested code paths
- [ ] Add unit tests for uncovered public methods
- [ ] Set coverage gates in CI: 70% overall, 80% for core modules
- [ ] Add tests for edge cases and error paths

---

### P2-NEW-002: CI/CD Coverage Gates Not Enforced

**Status**: 📋 NOT STARTED  
**Effort**: S (1-2 hours)

**Problem**: No coverage threshold enforcement in CI/CD pipeline.

**Required Actions**:

- [ ] Add `coverage report --fail-under=80` to CI
- [ ] Configure coverage thresholds per module
- [ ] Add coverage badge to README

---

### P2-NEW-003: Multiprocessing Timeout Hardening

**Original ID**: Section 13 Phase 2  
**Status**: 📋 DOCUMENTED  
**Effort**: M (1-2 days)

**Context**: Manager context is required for RemoteWorkerClient. Cannot replace with SimpleQueue.

**Approach** (within existing code):

1. Add process termination on timeout in `_execute_parallel_training`
2. Ensure workers always emit results (success or error)
3. Add bounded result collection with timeout fallback

**Required Actions**:

- [ ] Add try/finally wrapper with worker termination
- [ ] Add worker result guarantee pattern
- [ ] Add maximum wait time for result collection
- [ ] Test timeout behavior

---

### P2-NEW-004: Graceful Fallback to Sequential Training

**Original ID**: CASCOR-P0-001 deferred action  
**Status**: 📋 NOT STARTED  
**Effort**: S-M (2-4 hours)

**Problem**: When parallel training times out or fails, fallback to sequential should be automatic and graceful.

**Required Actions**:

- [ ] Implement automatic sequential fallback on MP timeout
- [ ] Add logging for fallback events
- [ ] Add metrics for fallback frequency
- [ ] Test fallback behavior under various failure modes

---

### P2-NEW-005: Add Missing Unit Tests for Specific Features

**Status**: 📋 NOT STARTED  

| Feature                        | Test Needed    | Original Reference |
| ------------------------------ | -------------- | ------------------ |
| `save_object()` method         | Add unit test  | CASCOR-P0-004      |
| Candidate seeds differ         | Verify seeds   | CASCOR-P0-005      |
| Input/output size combinations | Typical combos | CASCOR-P0-006      |
| Best candidate mis-index       | Reproduce case | CASCOR-P1-009      |

**Required Actions**:

- [ ] Add `test_save_object_method.py`
- [ ] Add `test_candidate_seed_diversity.py`
- [ ] Add `test_residual_error_shapes.py`
- [ ] Add `test_best_candidate_selection.py`

---

### P2-NEW-006: Type Errors - Gradual Fix

**Original ID**: CASCOR-P2-002 ongoing  
**Status**: 🟡 IN PROGRESS  

**Problem**: Type checking enabled but many errors exist. Currently using `continue-on-error` in CI.

**Required Actions**:

- [ ] Run mypy and categorize errors
- [ ] Fix critical type errors in core modules
- [ ] Gradually increase type strictness
- [ ] Remove `continue-on-error` once stable

---

## 3. P3-NEW: Profiling & Performance

### P3-NEW-001: Development Profiling Infrastructure

**Original ID**: Section 11 Phase 1  
**Status**: 📋 NOT STARTED  
**Effort**: M (3-5 days)

**Deliverables**:

- [ ] Add `--profile` flag to `main.py` for cProfile integration
- [ ] Create `src/profiling/__init__.py`
- [ ] Create `src/profiling/deterministic.py` (cProfile wrappers)
- [ ] Create `src/profiling/memory.py` (tracemalloc wrappers)
- [ ] Document profiling commands in AGENTS.md

---

### P3-NEW-002: Sampling Profiling Infrastructure

**Original ID**: Section 11 Phase 2  
**Status**: 📋 NOT STARTED  
**Effort**: M (2-3 days)

**Deliverables**:

- [ ] Install and test py-spy on development environment
- [ ] Create `util/profile_training.bash`
- [ ] Create baseline profiles for key operations
- [ ] Add profiling to CI/CD for performance regression detection

---

### P3-NEW-003: GPU Support

**Original ID**: P3-003  
**Status**: 🔴 NOT STARTED  
**Effort**: XL (2-4 weeks)

**Problem**: No CUDA/GPU acceleration for training.

**Notes**: Requires significant refactoring to move tensors to GPU devices. PyTorch CUDA support already available in environment.

**Required Actions**:

- [ ] Assess GPU memory requirements
- [ ] Add device configuration to CascadeCorrelationConfig
- [ ] Refactor tensor operations for GPU compatibility
- [ ] Add GPU tests (marked with `@pytest.mark.gpu`)
- [ ] Benchmark GPU vs CPU performance

---

### P3-NEW-004: Continuous Profiling (Grafana Pyroscope)

**Original ID**: Section 11 Phase 3-4  
**Status**: 📋 NOT STARTED  
**Effort**: L (1-2 weeks)

**Deliverables**:

- [ ] Deploy Grafana Pyroscope (Docker)
- [ ] Integrate pyroscope-io SDK into Cascor
- [ ] Create Grafana dashboards
- [ ] Set up performance regression alerting
- [ ] Integrate torch.profiler for GPU operations

---

## 4. P4-NEW: Documentation & Verification

### P4-NEW-001: Execute main.py End-to-End with Plotting

**Original ID**: CASCOR-P0-003 action item  
**Status**: 📋 NOT VERIFIED  
**Effort**: S (30 minutes)

**Required Actions**:

- [ ] Run `python main.py` with default settings
- [ ] Verify plotting subprocess works correctly
- [ ] Document any issues encountered
- [ ] Update roadmap with verification status

---

### P4-NEW-002: Test ./try Script Launch

**Original ID**: CASCOR-P1-005 action item  
**Status**: 📋 NOT VERIFIED  
**Effort**: S (30 minutes)

**Required Actions**:

- [ ] Test `./try` script successfully launches `main.py`
- [ ] Verify all environment variables set correctly
- [ ] Document any path issues

---

### P4-NEW-003: Add Workflow Status Badge to README

**Original ID**: CASCOR-P1-007 deferred  
**Status**: 📋 NOT STARTED  
**Effort**: S (15 minutes)

**Required Actions**:

- [ ] Generate GitHub Actions badge URL
- [ ] Add badge to README.md
- [ ] Verify badge displays correct status

---

### P4-NEW-004: Reduce Debug Logging in Hot Paths

**Original ID**: CASCOR-P2-003 deferred  
**Status**: 📋 NOT STARTED  
**Effort**: M (2-4 hours)

**Problem**: Excessive debug logging in training loops impacts performance.

**Required Actions**:

- [ ] Profile logging overhead in training loops
- [ ] Identify hot path log statements
- [ ] Move detailed logs to TRACE level
- [ ] Add conditional logging for batch operations

---

### P4-NEW-005: Verify Parallel Processing Working

**Original ID**: Section 10 - Parallel Processing Analysis  
**Status**: 📋 NEEDS TESTING  
**Effort**: S (1 hour)

**Verification Procedure**:

```bash
export CASCOR_LOG_LEVEL=DEBUG
cd src && python main.py 2>&1 | tee training_log.txt
grep "execute_parallel\|execute_sequential" training_log.txt
```

**Required Actions**:

- [ ] Run verification procedure
- [ ] Document parallel vs sequential behavior
- [ ] Monitor worker process spawning
- [ ] Verify result collection in logs

---

### P4-NEW-006: Module Naming Collision Resolution

**Original ID**: CANOPY-P1-002  
**Status**: 🟡 MITIGATED (workaround in place)  
**Effort**: M (1-2 hours) if refactor chosen

**Current Mitigation**: `sys.path.insert(0, ...)` ensures Cascor modules take priority.

**Future Resolution Options**:

- Option A: Rename Canopy's `constants.py` to `canopy_constants.py`
- Option B: Rename Cascor's `constants/` to `cascor_constants/`

**Required Actions**:

- [ ] Evaluate impact of current workaround
- [ ] Decide on rename strategy
- [ ] Implement rename if needed (post-deployment)

---

## 5. Implementation Schedule

### Phase A: Pre-Deployment Critical (Week 1-2)

| Order | Task                                   | Priority | Effort | Dependencies |
| ----- | -------------------------------------- | -------- | ------ | ------------ |
| A.1   | P4-NEW-001: Verify main.py end-to-end  | P4       | S      | None         |
| A.2   | P4-NEW-002: Verify ./try script        | P4       | S      | None         |
| A.3   | P4-NEW-005: Verify parallel processing | P4       | S      | None         |
| A.4   | P2-NEW-002: CI coverage gates          | P2       | S      | None         |
| A.5   | P4-NEW-003: README badge               | P4       | S      | CI working   |

### Phase B: Code Quality (Week 2-3)

| Order | Task                             | Priority | Effort | Dependencies |
| ----- | -------------------------------- | -------- | ------ | ------------ |
| B.1   | P2-NEW-005: Missing unit tests   | P2       | M      | None         |
| B.2   | P2-NEW-003: MP timeout hardening | P2       | M      | None         |
| B.3   | P2-NEW-004: Sequential fallback  | P2       | S-M    | B.2          |
| B.4   | P2-NEW-001: Coverage improvement | P2       | L      | B.1          |

### Phase C: Integration Architecture (Week 3-6)

| Order | Task                                | Priority | Effort | Dependencies |
| ----- | ----------------------------------- | -------- | ------ | ------------ |
| C.1   | P1-NEW-003: Async training boundary | P1       | M      | None         |
| C.2   | P1-NEW-002: RemoteWorkerClient      | P1       | L      | C.1          |
| C.3   | P1-NEW-001: IPC architecture        | P1       | XL     | C.1, C.2     |

### Phase D: Performance & Profiling (Week 6+)

| Order | Task                              | Priority | Effort | Dependencies |
| ----- | --------------------------------- | -------- | ------ | ------------ |
| D.1   | P3-NEW-001: Development profiling | P3       | M      | None         |
| D.2   | P3-NEW-002: Sampling profiling    | P3       | M      | D.1          |
| D.3   | P4-NEW-004: Reduce logging        | P4       | M      | D.1          |
| D.4   | P3-NEW-003: GPU support           | P3       | XL     | None         |
| D.5   | P3-NEW-004: Continuous profiling  | P3       | L      | D.1, D.2     |

---

## 6. Deferred Items

The following items are explicitly deferred to post-deployment:

| Item                                   | Reason                | Original Reference |
| -------------------------------------- | --------------------- | ------------------ |
| Module naming collision rename         | Workaround sufficient | CANOPY-P1-002      |
| Remove "roll" concept in CandidateUnit | Low priority, capped  | CASCOR-P1-008      |
| Candidate factory refactor             | Design decision       | P3-001             |
| Continuous profiling (Pyroscope)       | Infrastructure needed | Section 11 Phase 3 |

---

## Document History

| Date       | Version | Author           | Changes                        |
| ---------- | ------- | ---------------- | ------------------------------ |
| 2026-01-25 | 2.0.0   | Development Team | Initial extraction from v1.6.0 |

---
