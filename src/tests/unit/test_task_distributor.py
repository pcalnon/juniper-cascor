"""Tests for the Phase 3 TaskDistributor.

Covers:
- Local-only distribution (no remote workers)
- Mixed local + remote distribution
- Task redistribution on remote failure
- Local-first scheduling priority
- Unified result collection from both tiers
"""

import logging
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from parallelism.task_distributor import TaskDistributor


@dataclass
class MockResult:
    """Minimal stand-in for CandidateTrainingResult."""

    candidate_id: int
    success: bool = True
    correlation: float = 0.5


def _make_tasks(n: int) -> list:
    """Create n fake task tuples matching _generate_candidate_tasks format.

    Each task is (candidate_index, candidate_data_tuple, training_inputs).
    """
    return [(i, (i, 4, "sigmoid", 1.0, f"uuid-{i}", i, 1e6, 1e6), (None, 100, None, None, 0.01, 20)) for i in range(n)]


class TestTaskDistributorLocalOnly:
    """When no remote workers are available, all tasks go to local."""

    def test_all_tasks_go_to_local_fn(self):
        td = TaskDistributor()
        tasks = _make_tasks(4)
        local_calls = []

        def local_fn(t, pc):
            local_calls.append((len(t), pc))
            return [MockResult(candidate_id=task[0]) for task in t]

        results = td.distribute_and_collect(tasks=tasks, local_capacity=4, local_fn=local_fn, remote_fn=lambda t: [])
        assert len(results) == 4
        assert local_calls == [(4, 4)]

    def test_no_remote_fn_called(self):
        td = TaskDistributor()
        tasks = _make_tasks(3)
        remote_called = []

        def remote_fn(t):
            remote_called.append(True)
            return []

        td.distribute_and_collect(tasks=tasks, local_capacity=3, local_fn=lambda t, pc: [MockResult(candidate_id=i) for i in range(len(t))], remote_fn=remote_fn)
        assert remote_called == []


class TestTaskDistributorMixed:
    """Tasks split between local and remote when remote workers available."""

    def test_overflow_goes_to_remote(self):
        td = TaskDistributor()
        registry_mock = MagicMock()
        registry_mock.available_worker_count = 2
        coordinator_mock = MagicMock()
        coordinator_mock._registry = registry_mock
        td.set_coordinator(coordinator_mock)

        tasks = _make_tasks(6)
        local_tasks_received = []
        remote_tasks_received = []

        def local_fn(t, pc):
            local_tasks_received.extend(t)
            return [MockResult(candidate_id=task[0]) for task in t]

        def remote_fn(t):
            remote_tasks_received.extend(t)
            return [MockResult(candidate_id=task[0]) for task in t]

        results = td.distribute_and_collect(tasks=tasks, local_capacity=4, local_fn=local_fn, remote_fn=remote_fn)
        assert len(results) == 6
        assert len(local_tasks_received) == 4
        assert len(remote_tasks_received) == 2

    def test_remote_capacity_caps_remote_share(self):
        """Remote gets at most remote_worker_count tasks."""
        td = TaskDistributor()
        registry_mock = MagicMock()
        registry_mock.available_worker_count = 1
        coordinator_mock = MagicMock()
        coordinator_mock._registry = registry_mock
        td.set_coordinator(coordinator_mock)

        tasks = _make_tasks(10)
        local_tasks_count = []
        remote_tasks_count = []

        def local_fn(t, pc):
            local_tasks_count.append(len(t))
            return [MockResult(candidate_id=task[0]) for task in t]

        def remote_fn(t):
            remote_tasks_count.append(len(t))
            return [MockResult(candidate_id=task[0]) for task in t]

        td.distribute_and_collect(tasks=tasks, local_capacity=4, local_fn=local_fn, remote_fn=remote_fn)
        # Remote should get at most 1 task (remote_worker_count=1)
        assert remote_tasks_count == [1]
        # Local should get the remaining 9 (4 initial + 5 excess)
        assert local_tasks_count == [9]


class TestTaskRedistribution:
    """Failed remote tasks are reassigned to local."""

    def test_remote_exception_retries_locally(self):
        td = TaskDistributor()
        registry_mock = MagicMock()
        registry_mock.available_worker_count = 2
        coordinator_mock = MagicMock()
        coordinator_mock._registry = registry_mock
        td.set_coordinator(coordinator_mock)

        tasks = _make_tasks(6)

        def local_fn(t, pc):
            return [MockResult(candidate_id=task[0]) for task in t]

        def remote_fn(t):
            raise ConnectionError("Worker disconnected")

        results = td.distribute_and_collect(tasks=tasks, local_capacity=4, local_fn=local_fn, remote_fn=remote_fn)
        # All 6 tasks should complete (4 local + 2 retried locally)
        assert len(results) == 6

    def test_incomplete_remote_results_retried(self):
        td = TaskDistributor()
        registry_mock = MagicMock()
        registry_mock.available_worker_count = 3
        coordinator_mock = MagicMock()
        coordinator_mock._registry = registry_mock
        td.set_coordinator(coordinator_mock)

        tasks = _make_tasks(7)
        retry_calls = []

        def local_fn(t, pc):
            return [MockResult(candidate_id=task[0]) for task in t]

        def remote_fn(t):
            # Only return results for first task, skip the rest
            return [MockResult(candidate_id=t[0][0])]

        def retry_fn(t, pc):
            retry_calls.extend(t)
            return [MockResult(candidate_id=task[0]) for task in t]

        results = td.distribute_and_collect(tasks=tasks, local_capacity=4, local_fn=local_fn, remote_fn=remote_fn, remote_retry_fn=retry_fn)
        assert len(results) == 4 + 1 + 2  # 4 local + 1 remote ok + 2 retried
        assert len(retry_calls) == 2

    def test_failed_remote_results_retried(self):
        td = TaskDistributor()
        registry_mock = MagicMock()
        registry_mock.available_worker_count = 2
        coordinator_mock = MagicMock()
        coordinator_mock._registry = registry_mock
        td.set_coordinator(coordinator_mock)

        tasks = _make_tasks(5)

        def local_fn(t, pc):
            return [MockResult(candidate_id=task[0]) for task in t]

        def remote_fn(t):
            # Return failure for one task
            return [MockResult(candidate_id=t[0][0], success=False)]

        results = td.distribute_and_collect(tasks=tasks, local_capacity=3, local_fn=local_fn, remote_fn=remote_fn)
        # 3 local + 1 retried (the failed one) = 4, but remote only got 1 task
        # Actually: local_capacity=3, remote_count=2, tasks=5
        # Local gets 3, remote gets 2 but only returns 1 (the failed one)
        # The failed result gets retried locally + the missing result also retried
        assert len(results) >= 4


class TestLocalFirstScheduling:
    """Local workers always get priority over remote."""

    def test_local_fills_before_remote(self):
        td = TaskDistributor()
        registry_mock = MagicMock()
        registry_mock.available_worker_count = 10
        coordinator_mock = MagicMock()
        coordinator_mock._registry = registry_mock
        td.set_coordinator(coordinator_mock)

        tasks = _make_tasks(8)
        local_indices = []
        remote_indices = []

        def local_fn(t, pc):
            local_indices.extend([task[0] for task in t])
            return [MockResult(candidate_id=task[0]) for task in t]

        def remote_fn(t):
            remote_indices.extend([task[0] for task in t])
            return [MockResult(candidate_id=task[0]) for task in t]

        td.distribute_and_collect(tasks=tasks, local_capacity=4, local_fn=local_fn, remote_fn=remote_fn)
        # First 4 tasks should be local
        assert local_indices == [0, 1, 2, 3]
        # Remaining 4 go to remote
        assert remote_indices == [4, 5, 6, 7]

    def test_all_local_when_tasks_fit(self):
        td = TaskDistributor()
        registry_mock = MagicMock()
        registry_mock.available_worker_count = 5
        coordinator_mock = MagicMock()
        coordinator_mock._registry = registry_mock
        td.set_coordinator(coordinator_mock)

        tasks = _make_tasks(3)
        remote_called = []

        def local_fn(t, pc):
            return [MockResult(candidate_id=task[0]) for task in t]

        def remote_fn(t):
            remote_called.append(True)
            return []

        td.distribute_and_collect(tasks=tasks, local_capacity=4, local_fn=local_fn, remote_fn=remote_fn)
        # Tasks (3) <= local_capacity (4), so no remote dispatch
        assert remote_called == []


class TestUnifiedResultCollection:
    """Results from both tiers collected correctly."""

    def test_results_from_both_tiers_merged(self):
        td = TaskDistributor()
        registry_mock = MagicMock()
        registry_mock.available_worker_count = 2
        coordinator_mock = MagicMock()
        coordinator_mock._registry = registry_mock
        td.set_coordinator(coordinator_mock)

        tasks = _make_tasks(6)

        def local_fn(t, pc):
            return [MockResult(candidate_id=task[0], correlation=0.8) for task in t]

        def remote_fn(t):
            return [MockResult(candidate_id=task[0], correlation=0.9) for task in t]

        results = td.distribute_and_collect(tasks=tasks, local_capacity=4, local_fn=local_fn, remote_fn=remote_fn)
        assert len(results) == 6
        # Local results have correlation 0.8, remote have 0.9
        correlations = sorted([r.correlation for r in results])
        assert correlations.count(0.8) == 4
        assert correlations.count(0.9) == 2

    def test_remote_worker_count_property(self):
        td = TaskDistributor()
        assert td.remote_worker_count == 0

        registry_mock = MagicMock()
        registry_mock.available_worker_count = 3
        coordinator_mock = MagicMock()
        coordinator_mock._registry = registry_mock
        td.set_coordinator(coordinator_mock)
        assert td.remote_worker_count == 3
