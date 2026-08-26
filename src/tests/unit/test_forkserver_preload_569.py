#!/usr/bin/env python
"""Issue #569 (F3): the forkserver preload list carries the trainer, and every entry is importable.

Preloading a module in the forkserver runs its import once, in the forkserver process, and every
candidate worker then forks with it already in ``sys.modules`` instead of re-importing it after
fork to unpickle its Process target. Two things can silently go wrong with that list:

1. The trainer entry is dropped (a refactor of ``_init_multiprocessing``) -- workers quietly go
   back to importing ~70 modules each after fork (post-#588 size; measured at the #570 census).
2. An entry names a module that does not resolve. CPython's ``multiprocessing.forkserver.main``
   wraps each preload in ``try: __import__(modname) except ImportError: pass`` -- a mistyped
   entry is a **no-op with no error anywhere**. The only runtime evidence is a forkserver module
   census that fails to rise; this guard catches it at test time instead.

GUARDS (fail if the list regresses): the trainer-entry check and the importability check.
Also pinned: the misleading "fork context" comment #569 called out is gone.

The instance is built bare (``object.__new__``) with a stub config and a recording context, so
the test never creates a real pool, never registers real atexit handlers on a half-built object
and never changes this process's torch thread count.
"""

import importlib.util
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import cascade_correlation.cascade_correlation as cc_mod
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

pytestmark = pytest.mark.unit

TRAINER_MODULE = "cascade_correlation.cascade_correlation"


class _RecordingContext:
    """Stands in for ``mp.get_context("forkserver")``: records the preload list, does nothing else."""

    def __init__(self) -> None:
        self.preload = None

    def set_forkserver_preload(self, modules) -> None:
        self.preload = list(modules)


def _init_multiprocessing_with_recording_context(monkeypatch) -> _RecordingContext:
    ctx = _RecordingContext()
    monkeypatch.setattr(cc_mod.mp, "get_context", lambda *_a, **_k: ctx)
    monkeypatch.setattr(cc_mod.atexit, "register", MagicMock(name="atexit.register"))
    monkeypatch.setattr(cc_mod.torch, "set_num_threads", MagicMock(name="torch.set_num_threads"))

    net = object.__new__(CascadeCorrelationNetwork)
    net.logger = MagicMock(name="logger")
    net.config = SimpleNamespace(
        candidate_training_context_type="forkserver",
        enable_remote_workers=False,
        candidate_training_queue_authkey=b"test-authkey",
        candidate_training_queue_address=("127.0.0.1", 0),
        candidate_training_task_queue_timeout=None,
        candidate_training_shutdown_timeout=None,
        worker_thread_count=1,
    )
    net._init_multiprocessing()
    assert ctx.preload is not None, "set_forkserver_preload was never called for the forkserver context"
    return ctx


def test_preload_list_carries_the_trainer(monkeypatch):
    """GUARD: the trainer module is preloaded so workers fork with it already imported."""
    ctx = _init_multiprocessing_with_recording_context(monkeypatch)
    assert TRAINER_MODULE in ctx.preload, f"preload list lost the trainer entry: {ctx.preload}"
    # torch/numpy stay ahead of the trainer so the heavy preloads import first (ledger readability).
    assert ctx.preload.index("torch") < ctx.preload.index(TRAINER_MODULE)
    assert ctx.preload.index("numpy") < ctx.preload.index(TRAINER_MODULE)


def test_every_preload_entry_resolves(monkeypatch):
    """GUARD: no entry may be a silent no-op -- the forkserver swallows preload ImportErrors."""
    ctx = _init_multiprocessing_with_recording_context(monkeypatch)
    unresolved = [m for m in ctx.preload if importlib.util.find_spec(m) is None]
    assert not unresolved, f"preload entries that would be silently ignored by the forkserver: {unresolved}"


def test_preload_is_skipped_outside_forkserver(monkeypatch):
    """The list is only set for the forkserver start method (spawn/fork have no preload API)."""
    ctx = _RecordingContext()
    monkeypatch.setattr(cc_mod.mp, "get_context", lambda *_a, **_k: ctx)
    monkeypatch.setattr(cc_mod.atexit, "register", MagicMock())
    monkeypatch.setattr(cc_mod.torch, "set_num_threads", MagicMock())
    net = object.__new__(CascadeCorrelationNetwork)
    net.logger = MagicMock()
    net.config = SimpleNamespace(
        candidate_training_context_type="spawn",
        enable_remote_workers=False,
        candidate_training_queue_authkey=b"test-authkey",
        candidate_training_queue_address=("127.0.0.1", 0),
        candidate_training_task_queue_timeout=None,
        candidate_training_shutdown_timeout=None,
        worker_thread_count=1,
    )
    net._init_multiprocessing()
    assert ctx.preload is None


def test_misleading_fork_context_comment_is_gone():
    """#569 also asked for the garbled 'fork context' comment above the context creation to go."""
    source = inspect.getsource(CascadeCorrelationNetwork._init_multiprocessing)
    assert "did not corrUse" not in source
    assert "Use 'fork' context" not in source
