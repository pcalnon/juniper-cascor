"""Tests for the Phase 3 TaskDistributor.

Covers:
- Local-only distribution (no remote workers)
- Mixed local + remote distribution
- Task redistribution on remote failure
- Local-first scheduling priority
- Unified result collection from both tiers
- All-remote distribution path
- Split-tasks edge cases (zero capacity, zero remote count)
- Remote fallback: incomplete results, failed results, no-matching-task branch
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_remote_worker_count_property(self):
        td = TaskDistributor()
        assert td.remote_worker_count == 0

        registry_mock = MagicMock()
        registry_mock.available_worker_count = 3
        coordinator_mock = MagicMock()
        coordinator_mock._registry = registry_mock
        td.set_coordinator(coordinator_mock)
        assert td.remote_worker_count == 3


# =====================================================================
# NEW TESTS: Cover branches missing from the 84.26% baseline
# =====================================================================


class TestDistributeAndCollectAllRemote:
    """Cover the all-remote path (lines 98-101): no local tasks, only remote."""

    @pytest.mark.unit
    def test_all_remote_when_local_capacity_zero(self):
        """When local_capacity=0, _split_tasks returns (tasks, []) — local path.

        This is a degenerate edge case; local_capacity <= 0 triggers the
        short-circuit in _split_tasks that sends everything local.
        """
        td = TaskDistributor()
        registry_mock = MagicMock()
        registry_mock.available_worker_count = 4
        coordinator_mock = MagicMock()
        coordinator_mock._registry = registry_mock
        td.set_coordinator(coordinator_mock)

        tasks = _make_tasks(4)
        local_fn_calls = []

        def local_fn(t, pc):
            local_fn_calls.append(len(t))
            return [MockResult(candidate_id=task[0]) for task in t]

        def remote_fn(t):
            return [MockResult(candidate_id=task[0]) for task in t]

        results = td.distribute_and_collect(tasks=tasks, local_capacity=0, local_fn=local_fn, remote_fn=remote_fn)
        # local_capacity <= 0 means _split_tasks returns (tasks, [])
        assert len(results) == 4
        assert local_fn_calls == [4]

    @pytest.mark.unit
    def test_all_remote_path_direct(self):
        """Directly exercise the all-remote branch by manually splitting.

        We mock _split_tasks to force ([], remote_tasks) to cover lines 98-101.
        """
        td = TaskDistributor()
        registry_mock = MagicMock()
        registry_mock.available_worker_count = 5
        coordinator_mock = MagicMock()
        coordinator_mock._registry = registry_mock
        td.set_coordinator(coordinator_mock)

        tasks = _make_tasks(3)
        remote_fn_calls = []

        def local_fn(t, pc):
            # Should not be called as primary (only as retry_fn default)
            return [MockResult(candidate_id=task[0]) for task in t]

        def remote_fn(t):
            remote_fn_calls.append(len(t))
            return [MockResult(candidate_id=task[0]) for task in t]

        # Patch _split_tasks to return no local tasks, all remote
        with patch.object(td, "_split_tasks", return_value=([], tasks)):
            results = td.distribute_and_collect(tasks=tasks, local_capacity=4, local_fn=local_fn, remote_fn=remote_fn)

        assert len(results) == 3
        assert remote_fn_calls == [3]

    @pytest.mark.unit
    def test_all_remote_with_exception_falls_back_to_retry(self):
        """All-remote path where remote_fn raises an exception triggers retry."""
        td = TaskDistributor()
        tasks = _make_tasks(2)
        retry_calls = []

        def remote_fn(t):
            raise TimeoutError("Remote worker timed out")

        def retry_fn(t, pc):
            retry_calls.extend(t)
            return [MockResult(candidate_id=task[0]) for task in t]

        with patch.object(td, "_split_tasks", return_value=([], tasks)):
            results = td.distribute_and_collect(tasks=tasks, local_capacity=2, local_fn=retry_fn, remote_fn=remote_fn, remote_retry_fn=retry_fn)

        assert len(results) == 2
        assert len(retry_calls) == 2


class TestSplitTasksEdgeCases:
    """Cover edge cases in _split_tasks (lines 122-140)."""

    @pytest.mark.unit
    def test_zero_remote_count_returns_all_local(self):
        td = TaskDistributor()
        tasks = _make_tasks(5)
        local, remote = td._split_tasks(tasks, local_capacity=3, remote_count=0)
        assert local == tasks
        assert remote == []

    @pytest.mark.unit
    def test_zero_local_capacity_returns_all_local(self):
        """When local_capacity <= 0, all tasks go local (degenerate case)."""
        td = TaskDistributor()
        tasks = _make_tasks(5)
        local, remote = td._split_tasks(tasks, local_capacity=0, remote_count=3)
        assert local == tasks
        assert remote == []

    @pytest.mark.unit
    def test_negative_remote_count_returns_all_local(self):
        td = TaskDistributor()
        tasks = _make_tasks(4)
        local, remote = td._split_tasks(tasks, local_capacity=2, remote_count=-1)
        assert local == tasks
        assert remote == []

    @pytest.mark.unit
    def test_tasks_fewer_than_local_capacity(self):
        td = TaskDistributor()
        tasks = _make_tasks(2)
        local, remote = td._split_tasks(tasks, local_capacity=5, remote_count=3)
        assert local == tasks
        assert remote == []

    @pytest.mark.unit
    def test_tasks_equal_local_capacity(self):
        td = TaskDistributor()
        tasks = _make_tasks(4)
        local, remote = td._split_tasks(tasks, local_capacity=4, remote_count=2)
        assert local == tasks
        assert remote == []

    @pytest.mark.unit
    def test_overflow_split_within_remote_capacity(self):
        """Tasks exceed local_capacity but overflow fits remote capacity."""
        td = TaskDistributor()
        tasks = _make_tasks(6)
        local, remote = td._split_tasks(tasks, local_capacity=4, remote_count=5)
        assert len(local) == 4
        assert len(remote) == 2
        assert [t[0] for t in local] == [0, 1, 2, 3]
        assert [t[0] for t in remote] == [4, 5]

    @pytest.mark.unit
    def test_overflow_exceeds_remote_capacity_excess_moves_back(self):
        """Overflow exceeds remote capacity; excess is redistributed to local."""
        td = TaskDistributor()
        tasks = _make_tasks(10)
        local, remote = td._split_tasks(tasks, local_capacity=3, remote_count=2)
        # local gets first 3 + 5 excess = 8
        # remote gets 2 (capped)
        assert len(remote) == 2
        assert len(local) == 8
        # Remote tasks should be indices 3,4 (first 2 of overflow)
        assert [t[0] for t in remote] == [3, 4]
        # Local = [0,1,2] + [5,6,7,8,9]
        assert [t[0] for t in local] == [0, 1, 2, 5, 6, 7, 8, 9]

    @pytest.mark.unit
    def test_empty_tasks(self):
        td = TaskDistributor()
        local, remote = td._split_tasks([], local_capacity=4, remote_count=2)
        assert local == []
        assert remote == []

    @pytest.mark.unit
    def test_single_task_goes_local(self):
        td = TaskDistributor()
        tasks = _make_tasks(1)
        local, remote = td._split_tasks(tasks, local_capacity=1, remote_count=5)
        assert local == tasks
        assert remote == []


class TestExecuteRemoteWithFallback:
    """Cover _execute_remote_with_fallback (lines 142-227) edge cases."""

    @pytest.mark.unit
    def test_all_remote_succeed(self):
        """All remote results are successful — no retries needed."""
        td = TaskDistributor()
        tasks = _make_tasks(3)

        def remote_fn(t):
            return [MockResult(candidate_id=task[0], success=True) for task in t]

        def retry_fn(t, pc):
            raise AssertionError("retry_fn should not be called")

        results = td._execute_remote_with_fallback(tasks, remote_fn, retry_fn, local_capacity=2, timeout=10.0)
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.unit
    def test_remote_exception_triggers_full_retry(self):
        """Remote function raises — all tasks retry locally."""
        td = TaskDistributor()
        tasks = _make_tasks(3)
        retry_calls = []

        def remote_fn(t):
            raise RuntimeError("connection lost")

        def retry_fn(t, pc):
            retry_calls.extend(t)
            return [MockResult(candidate_id=task[0]) for task in t]

        results = td._execute_remote_with_fallback(tasks, remote_fn, retry_fn, local_capacity=4, timeout=10.0)
        assert len(results) == 3
        assert len(retry_calls) == 3

    @pytest.mark.unit
    def test_incomplete_results_retries_missing(self):
        """Remote returns fewer results than tasks — missing tasks retried."""
        td = TaskDistributor()
        tasks = _make_tasks(4)

        def remote_fn(t):
            # Only complete task 0 and 1, skip 2 and 3
            return [MockResult(candidate_id=0), MockResult(candidate_id=1)]

        retry_calls = []

        def retry_fn(t, pc):
            retry_calls.extend(t)
            return [MockResult(candidate_id=task[0]) for task in t]

        results = td._execute_remote_with_fallback(tasks, remote_fn, retry_fn, local_capacity=4, timeout=10.0)
        # 2 remote ok + 2 retried
        assert len(results) == 4
        # Retried tasks should be indices 2 and 3
        retried_ids = [t[0] for t in retry_calls]
        assert sorted(retried_ids) == [2, 3]

    @pytest.mark.unit
    def test_incomplete_results_with_no_candidate_id_on_result(self):
        """Remote returns result without candidate_id — completed_indices stays empty."""
        td = TaskDistributor()
        tasks = _make_tasks(3)

        class PlainResult:
            """Result without candidate_id attribute."""

            pass

        def remote_fn(t):
            # Return 1 result without candidate_id (fewer than tasks)
            return [PlainResult()]

        retry_calls = []

        def retry_fn(t, pc):
            retry_calls.extend(t)
            return [MockResult(candidate_id=task[0]) for task in t]

        results = td._execute_remote_with_fallback(tasks, remote_fn, retry_fn, local_capacity=4, timeout=10.0)
        # 1 remote result + 3 retried (all tasks since none had matching candidate_id)
        assert len(results) == 4
        assert len(retry_calls) == 3

    @pytest.mark.unit
    def test_failed_results_with_matching_task_retried(self):
        """Remote returns success=False results with matching task — retried locally."""
        td = TaskDistributor()
        tasks = _make_tasks(3)

        def remote_fn(t):
            return [
                MockResult(candidate_id=0, success=True),
                MockResult(candidate_id=1, success=False),
                MockResult(candidate_id=2, success=True),
            ]

        retry_calls = []

        def retry_fn(t, pc):
            retry_calls.extend(t)
            return [MockResult(candidate_id=task[0]) for task in t]

        results = td._execute_remote_with_fallback(tasks, remote_fn, retry_fn, local_capacity=4, timeout=10.0)
        # 2 successful + 1 retried
        assert len(results) == 3
        retried_ids = [t[0] for t in retry_calls]
        assert retried_ids == [1]

    @pytest.mark.unit
    def test_failed_result_no_matching_task_treated_as_successful(self):
        """Remote returns success=False with candidate_id not in tasks — treated as successful.

        This covers lines 207-212: the else branch where no matching task is found
        for a failed result, so it gets appended to successful instead.
        """
        td = TaskDistributor()
        tasks = _make_tasks(2)  # task ids 0 and 1

        def remote_fn(t):
            return [
                MockResult(candidate_id=0, success=True),
                # candidate_id=99 doesn't match any task
                MockResult(candidate_id=99, success=False),
            ]

        retry_calls = []

        def retry_fn(t, pc):
            retry_calls.extend(t)
            return [MockResult(candidate_id=task[0]) for task in t]

        results = td._execute_remote_with_fallback(tasks, remote_fn, retry_fn, local_capacity=4, timeout=10.0)
        # Both results should be in successful (id=99 has no matching task)
        assert len(results) == 2
        # No retries should have happened
        assert retry_calls == []

    @pytest.mark.unit
    def test_multiple_failed_results_retried(self):
        """Multiple results with success=False — all matching tasks retried."""
        td = TaskDistributor()
        tasks = _make_tasks(4)

        def remote_fn(t):
            return [
                MockResult(candidate_id=0, success=False),
                MockResult(candidate_id=1, success=True),
                MockResult(candidate_id=2, success=False),
                MockResult(candidate_id=3, success=True),
            ]

        retry_calls = []

        def retry_fn(t, pc):
            retry_calls.extend(t)
            return [MockResult(candidate_id=task[0]) for task in t]

        results = td._execute_remote_with_fallback(tasks, remote_fn, retry_fn, local_capacity=4, timeout=10.0)
        # 2 successful + 2 retried
        assert len(results) == 4
        retried_ids = sorted([t[0] for t in retry_calls])
        assert retried_ids == [0, 2]


class TestDistributorInit:
    """Cover initialization and coordinator setup."""

    @pytest.mark.unit
    def test_default_logger(self):
        td = TaskDistributor()
        assert td._logger.name == "juniper_cascor.parallelism.task_distributor"

    @pytest.mark.unit
    def test_custom_logger(self):
        custom = logging.getLogger("test.custom")
        td = TaskDistributor(dist_logger=custom)
        assert td._logger is custom

    @pytest.mark.unit
    def test_set_coordinator_stores_reference(self):
        td = TaskDistributor()
        assert td._coordinator is None
        coordinator = MagicMock()
        td.set_coordinator(coordinator)
        assert td._coordinator is coordinator

    @pytest.mark.unit
    def test_remote_worker_count_no_coordinator(self):
        td = TaskDistributor()
        assert td.remote_worker_count == 0


class TestDistributeAndCollectEdgeCases:
    """Cover edge cases in distribute_and_collect orchestration."""

    @pytest.mark.unit
    def test_empty_tasks_returns_empty(self):
        td = TaskDistributor()
        results = td.distribute_and_collect(tasks=[], local_capacity=4, local_fn=lambda t, pc: [], remote_fn=lambda t: [])
        assert results == []

    @pytest.mark.unit
    def test_remote_retry_fn_defaults_to_local_fn(self):
        """When remote_retry_fn is None, local_fn is used for retries."""
        td = TaskDistributor()
        tasks = _make_tasks(2)
        local_fn_calls = []

        def local_fn(t, pc):
            local_fn_calls.append(("local_fn", len(t)))
            return [MockResult(candidate_id=task[0]) for task in t]

        def remote_fn(t):
            raise ConnectionError("down")

        # Force all-remote path to exercise retry
        with patch.object(td, "_split_tasks", return_value=([], tasks)):
            results = td.distribute_and_collect(tasks=tasks, local_capacity=2, local_fn=local_fn, remote_fn=remote_fn)

        # local_fn should have been used as the retry function
        assert len(results) == 2
        assert ("local_fn", 2) in local_fn_calls

    @pytest.mark.unit
    def test_single_task_local_only(self):
        td = TaskDistributor()
        tasks = _make_tasks(1)

        def local_fn(t, pc):
            return [MockResult(candidate_id=task[0]) for task in t]

        results = td.distribute_and_collect(tasks=tasks, local_capacity=1, local_fn=local_fn, remote_fn=lambda t: [])
        assert len(results) == 1
        assert results[0].candidate_id == 0

    @pytest.mark.unit
    def test_dual_path_returns_combined_local_and_remote(self):
        """Dual-path execution returns local_results + remote_results in order."""
        td = TaskDistributor()
        registry_mock = MagicMock()
        registry_mock.available_worker_count = 3
        coordinator_mock = MagicMock()
        coordinator_mock._registry = registry_mock
        td.set_coordinator(coordinator_mock)

        tasks = _make_tasks(5)

        def local_fn(t, pc):
            return [MockResult(candidate_id=task[0], correlation=0.1) for task in t]

        def remote_fn(t):
            return [MockResult(candidate_id=task[0], correlation=0.9) for task in t]

        results = td.distribute_and_collect(tasks=tasks, local_capacity=2, local_fn=local_fn, remote_fn=remote_fn)
        assert len(results) == 5
        # First 2 results should be local (correlation 0.1), rest remote (0.9)
        assert results[0].correlation == 0.1
        assert results[1].correlation == 0.1
        assert results[2].correlation == 0.9

    @pytest.mark.unit
    def test_all_remote_success_returns_original_results(self):
        """Verify the success path returns original remote_results unchanged."""
        td = TaskDistributor()
        tasks = _make_tasks(2)

        expected_results = [MockResult(candidate_id=0, success=True, correlation=0.77), MockResult(candidate_id=1, success=True, correlation=0.88)]

        def remote_fn(t):
            return list(expected_results)

        def retry_fn(t, pc):
            raise AssertionError("retry_fn should not be called on full success")

        results = td._execute_remote_with_fallback(tasks, remote_fn, retry_fn, local_capacity=2, timeout=10.0)
        # Should return the exact remote results, not modified
        assert results == expected_results
        assert len(results) == 2
        assert results[0].correlation == 0.77
        assert results[1].correlation == 0.88
