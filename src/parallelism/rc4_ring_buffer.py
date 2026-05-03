"""P-1 RC-4 multiprocessing-race ring buffer.

Lightweight, gated event capture for diagnosing the ``best_candidate=None``
race in the candidate-training distributor (V38 Phase A.2).

Intended properties:

- **Cheap-enough not to mask the race.** ``print(flush=True)`` masked the
  race because of stdout-flush + disk-sync timing. Here every event is an
  in-process ``deque.append`` (parent) or a non-blocking ``put_nowait``
  on a multiprocessing.Queue (worker). No I/O sync.
- **Disabled by default.** Imports are no-ops unless
  ``CASCOR_RC4_RING_BUFFER=1`` is set. Production code paths pay the cost
  of one cheap function-call per hook (which itself is gated on the same
  flag) and nothing else.
- **Cross-process.** Workers are separate processes so they can't share a
  Python deque with the parent. They post events through a dedicated
  ``multiprocessing.Queue`` (``set_worker_queue``); the parent drains it
  on demand (``drain_to_buffer``). The queue is bounded; ``put_nowait``
  drops on overflow rather than blocking.
- **Chronological dump.** ``dump_sorted`` returns a single string with
  events sorted by ``time.monotonic_ns()`` across all processes,
  formatted with µs offsets relative to the first event so signatures are
  easy to read.

Refs:

- ``notes/P1_RC4_INVESTIGATION_PLAN_2026-05-03.md`` (juniper-ml) §3.2.
- ``notes/V38_GROW_NETWORK_INVESTIGATION_PLAN_2026-05-02.md`` (juniper-ml) §7.

Hooks live in ``cascade_correlation.cascade_correlation`` at the
points enumerated in the plan's §3.2.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Optional

# Module-level flag so callers can short-circuit before constructing
# payload dicts.
ENABLED: bool = os.environ.get("CASCOR_RC4_RING_BUFFER", "") == "1"

# Parent-side ring buffer. Bounded so a runaway test doesn't OOM. 10 000
# events is enough for several pytest-repeat iterations of the V38 tests
# at the rate the plan expects (~50 events per round).
_BUFFER: deque = deque(maxlen=10_000)

# Cross-process pump. The parent creates the queue (``init_parent_queue``)
# and seeds workers via ``set_worker_queue`` when spawning them. None
# until the parent initializes.
_INSTRUMENTATION_QUEUE: Optional[Any] = None


def is_enabled() -> bool:
    """Return whether the ring buffer is active for this process."""
    return ENABLED


def init_parent_queue(mp_ctx) -> Any:
    """Parent creates the cross-process queue once, lazily.

    Args:
        mp_ctx: a ``multiprocessing.context`` (e.g.
            ``multiprocessing.get_context("spawn")``) so the queue is
            spawn-compatible. ``cascade_correlation`` already keeps a
            reference to the active context for worker creation.

    Returns:
        The queue, suitable for passing to workers via
        ``set_worker_queue``. The same queue is also stored in the
        module global so the parent's ``emit`` can find it. Calling
        twice is idempotent — returns the existing queue.
    """
    global _INSTRUMENTATION_QUEUE
    if not ENABLED:
        return None
    if _INSTRUMENTATION_QUEUE is None:
        # Bounded so ``put_nowait`` overflow is observable and bounded.
        # Workers each emit ~10 events per task; a 4-worker, 4-task
        # round produces ~160 events. 4 096 covers ~25 rounds of
        # buffered events without dropping.
        _INSTRUMENTATION_QUEUE = mp_ctx.Queue(maxsize=4096)
    return _INSTRUMENTATION_QUEUE


def set_worker_queue(queue: Any) -> None:
    """Worker (child process) installs the queue handed to it by parent.

    Called once at the start of ``_worker_loop`` if the queue is
    non-None. From then on, ``emit`` in this worker process posts to
    the queue instead of the parent-only deque.
    """
    global _INSTRUMENTATION_QUEUE
    _INSTRUMENTATION_QUEUE = queue


def emit(event: str, **payload: Any) -> None:
    """Record an event with a monotonic-ns timestamp.

    O(1) common case: one ``time.monotonic_ns`` call, one tuple
    construction, one ``deque.append`` (parent) or one ``put_nowait``
    (worker). No locks, no I/O sync.

    Workers' ``put_nowait`` drops on a full queue rather than blocking,
    so an overrun is visible (gap in the dump) but never deadlocks the
    code we're trying to observe.
    """
    if not ENABLED:
        return
    record = (time.monotonic_ns(), os.getpid(), event, payload)
    if _INSTRUMENTATION_QUEUE is not None:
        try:
            _INSTRUMENTATION_QUEUE.put_nowait(record)
        except Exception:  # nosec B110 - drop on overflow, never block
            pass
    else:
        _BUFFER.append(record)


def drain_to_buffer() -> int:
    """Parent drains the cross-process queue into its in-process deque.

    Called from the pytest fixture's teardown so the dump includes
    every worker event the queue managed to receive. Returns the
    number of events drained for logging.
    """
    if not ENABLED or _INSTRUMENTATION_QUEUE is None:
        return 0
    from queue import Empty

    drained = 0
    while True:
        try:
            record = _INSTRUMENTATION_QUEUE.get_nowait()
            _BUFFER.append(record)
            drained += 1
        except Empty:
            return drained


def dump_sorted() -> str:
    """Format all captured events chronologically, with µs offsets.

    Sort key is the monotonic-ns timestamp, which is a single global
    clock the parent and all workers share (``time.monotonic_ns`` is
    process-independent on Linux/macOS as long as the host clock isn't
    rewound).

    The output's first event is offset 0; subsequent events show
    elapsed µs since that first event. Format:

        [+    123.4µs pid=12345] event_name                     k1=v1 k2=v2
    """
    drain_to_buffer()
    if not _BUFFER:
        return "<empty>"
    sorted_records = sorted(_BUFFER, key=lambda r: r[0])
    t0 = sorted_records[0][0]
    lines = []
    for ts, pid, event, payload in sorted_records:
        delta_us = (ts - t0) / 1000.0
        payload_str = " ".join(f"{k}={v}" for k, v in payload.items())
        lines.append(f"[+{delta_us:>10.1f}µs pid={pid:>6}] {event:<42} {payload_str}")
    return "\n".join(lines)


def reset() -> None:
    """Clear the parent's deque AND drain any pending worker events.

    Called from the pytest fixture's setup so each test starts with a
    fresh buffer. Worker queue isn't drained at setup (there shouldn't
    be lingering events from a prior test if teardown ran), but a
    drain is still cheap.
    """
    if not ENABLED:
        return
    _BUFFER.clear()
    if _INSTRUMENTATION_QUEUE is not None:
        from queue import Empty

        while True:
            try:
                _INSTRUMENTATION_QUEUE.get_nowait()
            except Empty:
                break
