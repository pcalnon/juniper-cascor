"""Unit tests for parallelism/rc4_ring_buffer.py to raise code coverage.

The module is **disabled by default** (its ``ENABLED`` flag is read from
``CASCOR_RC4_RING_BUFFER`` at import time), so in a normal test run every
public function short-circuits on ``if not ENABLED`` and the real bodies are
never measured. These tests drive the enabled paths by monkeypatching the
module-level ``ENABLED`` flag (never the environment, so the conftest RC-4
fixtures — which key off the env var — stay inert) and isolate the module
globals per test so the suite stays order-independent.

Part of the per-file coverage rollout (Phase C-5); see juniper-ml
``notes/JUNIPER_ECOSYSTEM_PER_FILE_COVERAGE_ROLLOUT_SCOPING_2026-06-30.md``.
"""

import os
import queue as _queue
import sys
from collections import deque

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from parallelism import rc4_ring_buffer as rc4

pytestmark = pytest.mark.unit


class _FakeQueue:
    """Deterministic stand-in for a ``multiprocessing.Queue``.

    Avoids the real queue's background feeder thread so ``put_nowait`` /
    ``get_nowait`` behave synchronously and tests stay deterministic.
    ``force_full=True`` makes every ``put_nowait`` raise ``queue.Full`` to
    exercise ``emit``'s drop-on-overflow guard.
    """

    def __init__(self, maxsize: int = 0, force_full: bool = False):
        self._items: list = []
        self._maxsize = maxsize
        self._force_full = force_full

    def put_nowait(self, item) -> None:
        if self._force_full or (self._maxsize and len(self._items) >= self._maxsize):
            raise _queue.Full()
        self._items.append(item)

    def get_nowait(self):
        if not self._items:
            raise _queue.Empty()
        return self._items.pop(0)


class _FakeCtx:
    """Minimal ``multiprocessing.context``-like object for ``init_parent_queue``."""

    def __init__(self):
        self.queue_calls = 0

    def Queue(self, maxsize: int = 0) -> _FakeQueue:  # noqa: N802 - mirrors mp API
        self.queue_calls += 1
        return _FakeQueue(maxsize=maxsize)


@pytest.fixture(autouse=True)
def _isolate_module_state(monkeypatch):
    """Reset the module globals per test and restore them afterwards.

    Defaults to ``ENABLED=False``; tests that exercise the active paths flip
    it to ``True`` with their own ``monkeypatch.setattr``.
    """
    monkeypatch.setattr(rc4, "_BUFFER", deque(maxlen=10_000))
    monkeypatch.setattr(rc4, "_INSTRUMENTATION_QUEUE", None)
    monkeypatch.setattr(rc4, "ENABLED", False)


def _enable(monkeypatch):
    monkeypatch.setattr(rc4, "ENABLED", True)


class TestIsEnabled:
    def test_returns_flag_value(self, monkeypatch):
        assert rc4.is_enabled() is False
        _enable(monkeypatch)
        assert rc4.is_enabled() is True


class TestInitParentQueue:
    def test_disabled_returns_none_and_leaves_global_unset(self):
        ctx = _FakeCtx()
        assert rc4.init_parent_queue(ctx) is None
        assert ctx.queue_calls == 0
        assert rc4._INSTRUMENTATION_QUEUE is None

    def test_creates_queue_once_and_is_idempotent(self, monkeypatch):
        _enable(monkeypatch)
        ctx = _FakeCtx()

        first = rc4.init_parent_queue(ctx)
        assert first is not None
        assert rc4._INSTRUMENTATION_QUEUE is first
        assert ctx.queue_calls == 1

        # Second call is a no-op: same queue, no new Queue() construction.
        second = rc4.init_parent_queue(ctx)
        assert second is first
        assert ctx.queue_calls == 1


class TestSetWorkerQueue:
    def test_installs_queue_into_module_global(self):
        sentinel = _FakeQueue()
        rc4.set_worker_queue(sentinel)
        assert rc4._INSTRUMENTATION_QUEUE is sentinel


class TestEmit:
    def test_disabled_is_a_noop(self):
        rc4.emit("ignored", k="v")
        assert len(rc4._BUFFER) == 0

    def test_enabled_posts_to_queue_when_present(self, monkeypatch):
        _enable(monkeypatch)
        fake = _FakeQueue()
        rc4.set_worker_queue(fake)

        rc4.emit("worker_event", candidate=3)

        record = fake.get_nowait()
        ts, pid, event, payload = record
        assert isinstance(ts, int)
        assert pid == os.getpid()
        assert event == "worker_event"
        assert payload == {"candidate": 3}
        # Nothing leaked into the parent deque.
        assert len(rc4._BUFFER) == 0

    def test_enabled_full_queue_is_swallowed(self, monkeypatch):
        _enable(monkeypatch)
        rc4.set_worker_queue(_FakeQueue(force_full=True))

        # Must not raise even though put_nowait raises queue.Full.
        rc4.emit("dropped", n=1)
        assert len(rc4._BUFFER) == 0

    def test_enabled_appends_to_buffer_when_no_queue(self, monkeypatch):
        _enable(monkeypatch)
        rc4.emit("parent_event", value=42)

        assert len(rc4._BUFFER) == 1
        ts, pid, event, payload = rc4._BUFFER[0]
        assert event == "parent_event"
        assert payload == {"value": 42}


class TestDrainToBuffer:
    def test_disabled_returns_zero(self):
        assert rc4.drain_to_buffer() == 0

    def test_enabled_without_queue_returns_zero(self, monkeypatch):
        _enable(monkeypatch)
        assert rc4.drain_to_buffer() == 0

    def test_enabled_drains_queue_into_buffer(self, monkeypatch):
        _enable(monkeypatch)
        fake = _FakeQueue()
        fake.put_nowait((1, 111, "a", {}))
        fake.put_nowait((2, 222, "b", {}))
        rc4.set_worker_queue(fake)

        drained = rc4.drain_to_buffer()
        assert drained == 2
        assert len(rc4._BUFFER) == 2
        # Queue emptied.
        assert rc4.drain_to_buffer() == 0


class TestDumpSorted:
    def test_empty_returns_sentinel(self, monkeypatch):
        _enable(monkeypatch)
        assert rc4.dump_sorted() == "<empty>"

    def test_orders_by_timestamp_and_formats(self, monkeypatch):
        _enable(monkeypatch)
        # Insert out of order; dump must sort ascending by the ns timestamp.
        rc4._BUFFER.append((2_000, 222, "second", {"b": 2}))
        rc4._BUFFER.append((1_000, 111, "first", {"a": 1}))

        out = rc4.dump_sorted()
        lines = out.splitlines()
        assert len(lines) == 2
        # First (earliest) line is offset 0 and carries its pid/event/payload.
        assert "first" in lines[0]
        assert "pid=" in lines[0]
        assert "a=1" in lines[0]
        assert "second" in lines[1]
        assert "b=2" in lines[1]
        # Earliest event sorts first.
        assert lines[0].index("first") >= 0 and lines[1].index("second") >= 0


class TestReset:
    def test_disabled_is_a_noop(self):
        rc4._BUFFER.append((1, 1, "keep", {}))
        rc4.reset()  # ENABLED is False -> buffer untouched
        assert len(rc4._BUFFER) == 1

    def test_enabled_clears_buffer_without_queue(self, monkeypatch):
        _enable(monkeypatch)
        rc4._BUFFER.append((1, 1, "gone", {}))
        rc4.reset()
        assert len(rc4._BUFFER) == 0

    def test_enabled_clears_buffer_and_drains_pending_queue(self, monkeypatch):
        _enable(monkeypatch)
        rc4._BUFFER.append((1, 1, "gone", {}))
        fake = _FakeQueue()
        fake.put_nowait((9, 9, "pending", {}))
        rc4.set_worker_queue(fake)

        rc4.reset()

        assert len(rc4._BUFFER) == 0
        # The lingering queue event was drained (and discarded) during reset.
        assert fake.get_nowait.__self__ is fake  # queue object still usable
        with pytest.raises(_queue.Empty):
            fake.get_nowait()
