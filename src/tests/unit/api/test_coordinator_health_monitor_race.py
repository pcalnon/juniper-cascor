"""Phase 3D regression tests for the health-monitor / assignment race.

CONC-10 — `WorkerCoordinator._check_stale_workers` previously did its
re-check + active-task reassignment under `self._lock` but called
`self._registry.deregister(worker_id)` *outside* the lock. That left a
window in which `get_next_assignment(worker_id)` (which holds `self._lock`
for its entire critical section) could land a task on the worker between
the active-task handling and the deregister. The newly-assigned task then
waited up to `_task_reassignment_timeout` (default 120s) before being
picked back up.

The fix moves the entire re-check / reassignment / deregister sequence
under `self._lock`, so any concurrent `get_next_assignment(...)` either
runs before the worker is considered dead (and assigns normally) or after
it is removed (and finds no eligible worker / sees an empty registry).

The source-level checks below run unconditionally; the behavioural tests
defer the `from api.workers.coordinator import WorkerCoordinator` import via
a fixture so the JuniperCascor env's broken torch C-extension import does
not gate them.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

_COORDINATOR_SRC = Path(__file__).resolve().parents[3] / "api" / "workers" / "coordinator.py"


@pytest.mark.unit
class TestCheckStaleWorkersSourceLevel:
    """Source-level guard for the CONC-10 fix.

    Uses Python's ast module rather than regex so the indentation
    structure (which is the whole point of "is X inside the with-block?")
    is checked properly.
    """

    @classmethod
    def setup_class(cls) -> None:
        import ast

        cls.source = _COORDINATOR_SRC.read_text(encoding="utf-8")
        tree = ast.parse(cls.source)
        cls._tree = tree

    def _find_method(self, class_name: str, method_name: str):
        import ast

        for node in ast.walk(self._tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        return item
        return None

    def _collect_calls(self, body) -> list:
        """Return every Call node in the given AST body, recursively."""
        import ast

        calls = []
        for node in body:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    calls.append(sub)
        return calls

    @staticmethod
    def _call_matches(call, attr_chain: list[str]) -> bool:
        """Match `call.func` against an attribute chain like ['self', '_registry', 'deregister']."""
        import ast

        node = call.func
        chain = []
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            chain.append(node.id)
        chain.reverse()
        return chain == attr_chain

    def test_check_stale_workers_holds_lock_across_deregister(self):
        """The `with self._lock:` block must contain BOTH the active-task reassignment AND the deregister."""
        import ast

        method = self._find_method("WorkerCoordinator", "_check_stale_workers")
        assert method is not None, "could not locate WorkerCoordinator._check_stale_workers"

        # Find every `with self._lock:` block inside the method.
        with_blocks = []
        for node in ast.walk(method):
            if isinstance(node, ast.With):
                for item in node.items:
                    ctx = item.context_expr
                    if isinstance(ctx, ast.Attribute) and ctx.attr == "_lock" and isinstance(ctx.value, ast.Name) and ctx.value.id == "self":
                        with_blocks.append(node)

        assert with_blocks, "_check_stale_workers no longer uses `with self._lock:`"

        # The fix consolidates everything into a single lock block. Find at
        # least one block that contains both the deregister and the
        # unassigned-tasks append.
        any_holds_both = False
        for block in with_blocks:
            calls = self._collect_calls(block.body)
            has_deregister = any(self._call_matches(c, ["self", "_registry", "deregister"]) for c in calls)
            has_unassigned_append = any(self._call_matches(c, ["self", "_unassigned_tasks", "append"]) for c in calls)
            if has_deregister and has_unassigned_append:
                any_holds_both = True
                break

        assert any_holds_both, "CONC-10 regressed: no `with self._lock:` block in _check_stale_workers contains BOTH `self._registry.deregister(...)` AND `self._unassigned_tasks.append(...)` — the deregister is racing with assignment again"

    def test_unregister_send_callback_stays_outside_lock(self):
        """Per the lock-order rule documented in the fix, the send-callback unregister stays outside `_lock`."""
        import ast

        method = self._find_method("WorkerCoordinator", "_check_stale_workers")
        assert method is not None

        # Walk the for-loop body in the method and find calls at the
        # outer level (siblings of the `with` block).
        for node in ast.walk(method):
            if isinstance(node, ast.For):
                for stmt in node.body:
                    # Skip the `with self._lock:` block; look at sibling
                    # expression statements for the unregister_send_callback.
                    if isinstance(stmt, ast.With):
                        continue
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        call = stmt.value
                        if self._call_matches(call, ["self", "unregister_send_callback"]):
                            return  # found at the right scope
        # Either the call moved or got removed; both warrant a failure.
        pytest.fail("expected `self.unregister_send_callback(...)` as a sibling of the `with self._lock:` block in _check_stale_workers")


@pytest.fixture(scope="module")
def _coord_module():
    pytest.importorskip("torch", exc_type=ImportError)
    from api.workers import coordinator as coordinator_module
    from api.workers import registry as registry_module

    return coordinator_module, registry_module


@pytest.mark.unit
class TestCheckStaleWorkersBehaviour:
    """Behavioural cover: deregister-then-no-reassignment is atomic."""

    def test_active_task_returned_to_queue_when_worker_deregistered(self, _coord_module):
        """A worker with an active task that times out must have the task requeued."""
        coordinator_module, _ = _coord_module
        WorkerCoordinator = coordinator_module.WorkerCoordinator
        from api.workers.registry import WorkerRegistry

        registry = WorkerRegistry(heartbeat_timeout=0.01)
        coord = WorkerCoordinator(registry=registry, task_reassignment_timeout=60.0, health_check_interval=60.0)
        try:
            registry.register("w-stale", {})
            registry.assign_task("w-stale", "task-42")
            # Plant a pending task that w-stale was working on.
            coord._pending_tasks["task-42"] = coordinator_module.PendingTask(
                task_id="task-42",
                round_id="round-1",
                candidate_index=0,
                candidate_data={},
                training_params={},
                tensors={},
                assigned_worker_id="w-stale",
                dispatched_at=0.0,
            )

            # Force the heartbeat to be stale relative to the timeout.
            import time as _time

            _time.sleep(0.02)

            coord._check_stale_workers()

            # Worker is gone; active task is back on the unassigned queue.
            assert registry.get("w-stale") is None
            assert "task-42" in coord._unassigned_tasks
            assert coord._pending_tasks["task-42"].assigned_worker_id is None
        finally:
            coord.shutdown()

    def test_concurrent_assignment_and_dereg_no_orphan(self, _coord_module):
        """Hammering get_next_assignment while _check_stale_workers fires must not orphan a task."""
        coordinator_module, _ = _coord_module
        WorkerCoordinator = coordinator_module.WorkerCoordinator
        from api.workers.registry import WorkerRegistry

        registry = WorkerRegistry(heartbeat_timeout=60.0)  # workers are healthy initially
        coord = WorkerCoordinator(registry=registry, task_reassignment_timeout=60.0, health_check_interval=60.0)
        try:
            # Two workers; one will go stale during the run.
            registry.register("w-victim", {})
            registry.register("w-survivor", {})

            # Plant several unassigned tasks.
            import numpy as np

            specs = [{"candidate_index": i, "candidate_data": {"input_size": 4, "activation_name": "sigmoid"}, "training_params": {"epochs": 1, "learning_rate": 0.01}} for i in range(8)]
            tensors = {"candidate_input": np.zeros((4, 4), dtype=np.float32), "y": np.zeros((4, 1), dtype=np.float32), "residual_error": np.zeros((4, 1), dtype=np.float32)}
            coord.submit_tasks("round-r", specs, tensors)

            barrier = threading.Barrier(2)

            def assigner() -> None:
                barrier.wait()
                # Hammer assignments to both workers.
                for _ in range(50):
                    coord.get_next_assignment("w-survivor")
                    coord.get_next_assignment("w-victim")

            def reaper() -> None:
                barrier.wait()
                # Force-stale w-victim and run the reaper.
                registry._heartbeat_timeout = 0.0
                victim = registry.get("w-victim")
                if victim is not None:
                    victim.last_heartbeat = 0.0
                coord._check_stale_workers()

            t1 = threading.Thread(target=assigner)
            t2 = threading.Thread(target=reaper)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # Invariant: every PendingTask is either completed, owned by a
            # *currently registered* worker, or sitting in the unassigned
            # queue. No task may be assigned to a worker that no longer
            # exists in the registry.
            registered_ids = set()
            registry_lock = registry._lock
            with registry_lock:
                registered_ids = set(registry._workers.keys())

            orphans = []
            for task_id, task in coord._pending_tasks.items():
                if task.completed:
                    continue
                if task.assigned_worker_id is None:
                    assert task_id in coord._unassigned_tasks
                    continue
                if task.assigned_worker_id not in registered_ids:
                    orphans.append((task_id, task.assigned_worker_id))

            assert not orphans, f"CONC-10 race: tasks assigned to deregistered workers: {orphans!r}"
        finally:
            coord.shutdown()
