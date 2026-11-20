#!/usr/bin/env python
"""Pin that a pause holds only the run it was accepted for, and that a pause or resume that loses its race is refused.

The pause gate is one ``threading.Event`` that every run's interrupt checkpoints wait on. Before this
fix two routes left it closed under a run the pause was never meant for, and that run was held at
its first checkpoint while ``state_machine.status`` read STARTED, so ``resume`` was refused and only
a ``pause`` then a ``resume`` freed it:

- ``pause_training`` read the FSM, then cleared the gate, then issued PAUSE and ignored its answer,
  with no lock. A ``stop`` and a ``start`` landing between the check and the clear left the gate
  closed after the start had reopened it, and the run the start had queued was held. The pause still
  answered ``paused``. Over REST alone the three cannot overlap (the routes run the lifecycle on the
  event loop); a pause sent over ``/ws/control``, which runs each command on a worker thread, can.
- Only the completion settle reopened the gate. After a live swap whose join on the old fit timed
  out, the FSM still read STARTED, a pause sent then (over REST too) was accepted for the old fit,
  and once that fit stopped, the closed gate held the swapped run.

Now ``pause_training`` and ``resume_training`` make their check, their transition and their gate
write one step under ``_pause_lock``, and a start and a live swap open the gate under the same lock
before they submit. A pause whose transition is refused reopens the gate and is refused, and a resume
whose transition is refused is refused and writes nothing. A pause is also refused while the run that
owns the lifecycle has its stop raised, which is the window a live swap opens: after a timed-out join
the pause is refused and the swapped run trains on. Every settle a run owns reopens the gate after
its transition. juniper-cascor#756's description lists the first route under "What this does not
close". The tests drive the real manager, executor, FSM and (for some) the app, with a gated stub
``model.fit``: events and gates, no sleep decides an outcome.
"""

import json
import sys
import threading
import time
from concurrent.futures import Future
from typing import Callable, List, Optional, Tuple

import pytest
import torch
from fastapi.testclient import TestClient
from juniper_model_core.events import TrainingEvent

from api.app import create_app
from api.lifecycle import manager as manager_module
from api.lifecycle.state_machine import Command
from api.settings import Settings

# Bounded, so a failed assertion can never leave a fit thread parked for good.
_GATE_TIMEOUT = 10.0
_WAIT = 5.0
_CANCELLED = manager_module.TRAINING_SESSION_STATUS_CANCELLED
_SUCCESS = manager_module.TRAINING_SESSION_STATUS_SUCCESS
_FAILURE = manager_module.TRAINING_SESSION_STATUS_FAILURE
# The default /ws/control Origin allowlist is canopy's own origin, which canopy sends.
_CANOPY_ORIGIN = "http://localhost:8050"
_STRESS_TRIALS = 500
_REFUSED = "refused: Training is not active"
_RESUME_REFUSED = "refused: Training is not paused"


def _event(kind: str, seq: int) -> TrainingEvent:
    payloads = {
        "training_start": {"n_samples": 8},
        "epoch_end": {"epoch": 25, "epochs": 50, "metrics": {"loss": 0.5}},
        "training_end": {"metrics": {}},
    }
    return TrainingEvent(kind, payloads[kind], seq)


class _Run:
    """The gates of one call of the stub fit, which is one training run."""

    def __init__(self) -> None:
        self.after_release: Tuple[TrainingEvent, ...] = (_event("epoch_end", 1),)  # its next checkpoint
        self.raises = None  # raised, once released, in place of ``after_release``
        self.parked = threading.Event()
        self.release = threading.Event()
        self.ran_on = threading.Event()  # set once the fit got past ``after_release``
        # When set to an Event, the fit waits on it after its checkpoint, before it returns: a test that
        # resumes the run uses it so the run's settle cannot race the resume's own ``Started`` write.
        self.hold_after_checkpoint: Optional[threading.Event] = None


class _GatedFit:
    """Stands in for ``CascorModel.fit``: training_start, park on its gate, then its checkpoint and training_end."""

    def __init__(self) -> None:
        self.calls = 0
        self._runs: List[_Run] = []
        self._lock = threading.Lock()
        self._released = False

    def run(self, index: int) -> _Run:
        with self._lock:
            while len(self._runs) <= index:
                run = _Run()
                if self._released:  # a run entering after teardown must not park and outlive the test
                    run.release.set()
                self._runs.append(run)
            return self._runs[index]

    def __call__(self, x, y, *, X_val=None, y_val=None, on_event=None, **kwargs):
        with self._lock:
            index = self.calls
            self.calls += 1
        run = self.run(index)
        on_event(_event("training_start", 0))
        run.parked.set()
        run.release.wait(timeout=_GATE_TIMEOUT)
        if run.raises is not None:
            raise run.raises
        for event in run.after_release:
            on_event(event)
        run.ran_on.set()
        if run.hold_after_checkpoint is not None:
            run.hold_after_checkpoint.wait(timeout=_GATE_TIMEOUT)
        on_event(_event("training_end", 2))

    def release_all(self) -> None:
        with self._lock:
            self._released = True
            runs = list(self._runs)
        for run in runs:
            run.release.set()
            if run.hold_after_checkpoint is not None:
                run.hold_after_checkpoint.set()


class _WatchedGate(threading.Event):
    """The manager's pause gate, reporting when a fit waits on it while it is closed."""

    def __init__(self) -> None:
        self.held_a_fit = threading.Event()
        super().__init__()
        self.set()

    def wait(self, timeout=None):
        if not self.is_set():
            self.held_a_fit.set()
        return super().wait(timeout)


class _Hold:
    """Hold one request at a seam until the test lets it go, or until another thread waits on ``_pause_lock``.

    The second condition is what lets one test run on both trees: with the fix, a start or a resume
    that the held request's lock makes wait releases the hold instead of deadlocking it; without
    the fix there is no such lock, and the test lets the request go once the competing step returned.
    """

    def __init__(self, mgr) -> None:
        self.reached = threading.Event()
        self.go = threading.Event()
        self.contended = threading.Event()
        self._mgr = mgr
        self._real_lock = getattr(mgr, "_pause_lock", None)
        if self._real_lock is not None:
            mgr._pause_lock = _SignallingLock(self._real_lock, self)

    def wait_here(self) -> None:
        if not self.reached.is_set():
            self.reached.set()
            self.go.wait(timeout=_GATE_TIMEOUT)

    def remove(self) -> None:
        self.go.set()
        if self._real_lock is not None:
            self._mgr._pause_lock = self._real_lock


class _SignallingLock:
    """Wraps ``_pause_lock``: a thread that has to wait for it releases the hold first."""

    def __init__(self, real, hold: _Hold) -> None:
        self._real = real
        self._hold = hold

    def __enter__(self):
        if not self._real.acquire(blocking=False):
            self._hold.contended.set()
            self._hold.go.set()
            self._real.acquire()
        return self

    def __exit__(self, *exc_info):
        self._real.release()
        return False


class _Call:
    """Run one lifecycle request on its own thread, as ``/ws/control`` does; ``result`` is its answer."""

    def __init__(self, request: Callable[[], object]) -> None:
        self.done = threading.Event()
        self._answer = "NO ANSWER"
        self._thread = threading.Thread(target=self._run, args=(request,), daemon=True)
        self._thread.start()

    def _run(self, request) -> None:
        try:
            self._answer = _answer(request)
        finally:
            self.done.set()

    def result(self) -> str:
        assert self.done.wait(_WAIT), "the request never returned"
        return self._answer


def _answer(request: Callable[[], object]) -> str:
    """Make a lifecycle request: "ok", or "refused: <reason>" for what a route maps to 409."""
    try:
        request()
        return "ok"
    except (RuntimeError, ValueError) as exc:
        return f"refused: {exc}"


def _either(first: threading.Event, second: threading.Event) -> bool:
    """Wait, bounded, until either event is set. The events decide; the bound only ends a failed test."""
    deadline = time.monotonic() + _WAIT
    while time.monotonic() < deadline:
        if first.is_set() or second.is_set():
            return True
        first.wait(timeout=0.01)
    return False


def _xy() -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(8, 2)
    y = torch.zeros(8, 2)
    y[:4, 0] = 1
    y[4:, 1] = 1
    return x, y


def _start(mgr) -> Future:
    x, y = _xy()
    mgr.start_training(X=x, y=y)
    return mgr._training_future


def _parked(fit: _GatedFit, index: int) -> _Run:
    run = fit.run(index)
    assert run.parked.wait(timeout=_WAIT), f"run {index}'s fit never reached its gate"
    return run


def _passes_its_first_checkpoint(fit: _GatedFit, gate: _WatchedGate, index: int) -> None:
    """Release run ``index`` once it has entered, and require it to pass its checkpoint, not wait on the gate."""
    run = _parked(fit, index)
    run.release.set()
    assert _either(run.ran_on, gate.held_a_fit), f"run {index} neither passed its checkpoint nor waited on the gate"
    assert run.ran_on.is_set(), f"run {index} is held at its first checkpoint by a pause it was never sent"


def _fields(mgr) -> Tuple[str, str]:
    status = mgr.get_status()
    return status["state_machine"]["status"], status["training_state"]["status"]


def _hold_after_the_pause_check(mgr, monkeypatch) -> _Hold:
    """Hold ``pause_training`` right after its FSM check answers, before anything it does next."""
    hold = _Hold(mgr)
    sm = mgr.state_machine
    real_is_started = sm.is_started

    def _is_started() -> bool:
        answer = real_is_started()
        if sys._getframe(1).f_code.co_name == "pause_training":
            hold.wait_here()
        return answer

    monkeypatch.setattr(sm, "is_started", _is_started)
    return hold


def _hold_after_the_resume_check(mgr, monkeypatch) -> _Hold:
    """Hold ``resume_training`` right after its FSM check answers, before anything it does next."""
    hold = _Hold(mgr)
    sm = mgr.state_machine
    real_is_paused = sm.is_paused

    def _is_paused() -> bool:
        answer = real_is_paused()
        if sys._getframe(1).f_code.co_name == "resume_training":
            hold.wait_here()
        return answer

    monkeypatch.setattr(sm, "is_paused", _is_paused)
    return hold


class _RecordingLock:
    """Wraps ``_pause_lock``: the first thread that has to wait records the current future, then releases the hold."""

    def __init__(self, mgr, real, go: threading.Event) -> None:
        self._mgr = mgr
        self.real = real
        self._go = go
        self.contended = threading.Event()
        self.future_at_contention = None

    def __enter__(self):
        if not self.real.acquire(blocking=False):
            if not self.contended.is_set():
                self.future_at_contention = self._mgr._training_future
                self.contended.set()
            self._go.set()
            self.real.acquire()
        return self

    def __exit__(self, *exc_info):
        self.real.release()
        return False


@pytest.fixture
def lifecycle(monkeypatch):
    """A manager whose fit is gated and whose pause gate is watched, and its terminal session counter."""
    built: List[Tuple[object, _GatedFit]] = []
    counted: List[str] = []
    monkeypatch.setattr(manager_module, "inc_training_session_completed", counted.append)

    def _make():
        mgr = manager_module.TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        fit = _GatedFit()
        mgr.model.fit = fit
        gate = _WatchedGate()
        mgr._pause_event = gate
        built.append((mgr, fit))
        return mgr, fit, gate, counted

    yield _make
    for mgr, fit in built:
        fit.release_all()
        mgr._pause_event.set()
        mgr.shutdown()


# ---------------------------------------------------------------------------
# Route 1: a pause whose check passes just before a stop (or reset) and a start land
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("stopping", ["stop_training", "reset"])
def test_a_pause_overlapped_by_a_stop_and_a_start_does_not_hold_the_new_run(lifecycle, monkeypatch, stopping):
    """The measured overlap: the stopped fit stops, and the run the start queued trains."""
    mgr, fit, gate, counted = lifecycle()
    old_future = _start(mgr)
    old = _parked(fit, 0)
    hold = _hold_after_the_pause_check(mgr, monkeypatch)
    pause = _Call(mgr.pause_training)
    assert hold.reached.wait(_WAIT), "the pause never reached its FSM check"
    getattr(mgr, stopping)()
    start = _Call(lambda: _start(mgr))  # with the fix it waits for the pause, which the hold then lets go
    assert _either(start.done, hold.contended)
    hold.remove()

    # Before the fix: "ok" (answered paused), the gate closed and ``training_state`` Paused beside STOPPED.
    assert pause.result() == _REFUSED
    assert start.result() == "ok"
    assert hold.contended.is_set(), "the start submitted its run while the pause was between its check and its act"
    assert mgr._pause_event.is_set(), "the pause left the gate closed under the run the start queued"
    assert _fields(mgr) == ("STOPPED", "Stopped")
    new_future = mgr._training_future

    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    assert not old.ran_on.is_set(), "the stopped fit trained on"
    _passes_its_first_checkpoint(fit, gate, 1)
    assert new_future.result(timeout=_WAIT) is None
    assert _fields(mgr) == ("COMPLETED", "Completed")
    assert counted == [_CANCELLED, _SUCCESS]


def _donor_snapshot() -> str:
    donor = manager_module.TrainingLifecycleManager()
    try:
        donor.create_network(input_size=2, output_size=2)
        return donor.save_snapshot(description="pause check-then-act donor")["id"]
    finally:
        donor.shutdown()


@pytest.mark.unit
@pytest.mark.parametrize("interposer", ["stop", "reset", "stop then restore", "completion"])
def test_a_pause_that_loses_its_transition_is_refused_and_leaves_the_gate_open(lifecycle, monkeypatch, tmp_path, interposer):
    """Between the pause's check and its PAUSE, the FSM leaves STARTED: the pause changes nothing and is refused."""
    monkeypatch.setenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", str(tmp_path))
    snapshot_id = _donor_snapshot() if interposer == "stop then restore" else ""
    mgr, fit, _gate, counted = lifecycle()
    old_future = _start(mgr)
    old = _parked(fit, 0)
    hold = _hold_after_the_pause_check(mgr, monkeypatch)
    pause = _Call(mgr.pause_training)
    assert hold.reached.wait(_WAIT), "the pause never reached its FSM check"
    if interposer == "stop":
        mgr.stop_training()
        expected = ("STOPPED", "Stopped")
    elif interposer == "reset":
        mgr.reset()
        expected = ("STOPPED", "Stopped")
    elif interposer == "stop then restore":
        mgr.stop_training()
        assert mgr.load_snapshot(snapshot_id)["loaded"], "the restore was refused"
        expected = ("INVESTIGATING", "Stopped")
    else:
        old.release.set()  # the gate is still open: the fit passes its checkpoint and completes
        assert old_future.result(timeout=_WAIT) is None
        expected = ("COMPLETED", "Completed")
    hold.remove()

    # Before the fix: "ok", the gate left closed, and ``training_state`` Paused beside the new state.
    assert pause.result() == _REFUSED
    assert mgr._pause_event.is_set(), "a refused pause left the gate closed"
    assert _fields(mgr) == expected
    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    assert counted == ([_SUCCESS] if interposer == "completion" else [_CANCELLED])


@pytest.mark.unit
@pytest.mark.parametrize("interposer", ["stop", "reset", "completion"])
def test_a_resume_that_loses_its_transition_is_refused_and_writes_nothing(lifecycle, monkeypatch, interposer):
    """Between the resume's check and its RESUME, the FSM leaves PAUSED: the resume is refused, as one sent a moment later is."""
    mgr, fit, _gate, counted = lifecycle()
    run = fit.run(0)
    if interposer == "completion":
        run.after_release = ()  # no checkpoint left: the paused run's fit returns
    future = _start(mgr)
    _parked(fit, 0)
    assert _answer(mgr.pause_training) == "ok"
    hold = _hold_after_the_resume_check(mgr, monkeypatch)
    resume = _Call(mgr.resume_training)
    assert hold.reached.wait(_WAIT), "the resume never reached its FSM check"
    if interposer == "stop":
        mgr.stop_training()
        expected = ("STOPPED", "Stopped")
    elif interposer == "reset":
        mgr.reset()
        expected = ("STOPPED", "Stopped")
    else:
        run.release.set()  # the fit returns while PAUSED and settles COMPLETED
        assert future.result(timeout=_WAIT) is None
        expected = ("COMPLETED", "Completed")
    hold.remove()

    # Before this change: "ok" (answered resumed), and ``training_state`` Started beside the new state.
    assert resume.result() == _RESUME_REFUSED
    assert _fields(mgr) == expected
    run.release.set()
    assert future.result(timeout=_WAIT) is None
    assert not run.ran_on.is_set() or interposer == "completion", "the stopped fit trained on"
    assert counted == ([_SUCCESS] if interposer == "completion" else [_CANCELLED])


@pytest.mark.unit
@pytest.mark.parametrize("verb", ["start", "swap"])
def test_a_run_is_never_submitted_while_a_pause_is_between_its_check_and_its_act(lifecycle, monkeypatch, verb):
    """The start's and the swap's gate open precedes their submit, so the submit waits for a pause in progress."""
    mgr, fit, _gate, _counted = lifecycle()
    if verb == "swap":
        mgr.set_experimental_functions(True)
    old_future = _start(mgr)
    old = _parked(fit, 0)
    reached = threading.Event()
    go = threading.Event()
    recording = _RecordingLock(mgr, getattr(mgr, "_pause_lock", None), go)
    if recording.real is not None:  # without the lock nothing orders the submit, and the wait below fails
        mgr._pause_lock = recording
    sm = mgr.state_machine
    real_is_started = sm.is_started

    def _is_started() -> bool:
        answer = real_is_started()
        if sys._getframe(1).f_code.co_name == "pause_training" and not reached.is_set():
            reached.set()
            go.wait(timeout=_GATE_TIMEOUT)
        return answer

    monkeypatch.setattr(sm, "is_started", _is_started)
    pause = _Call(mgr.pause_training)
    assert reached.wait(_WAIT), "the pause never reached its FSM check"
    real_result = old_future.result
    try:
        if verb == "start":
            mgr.stop_training()
            other = _Call(lambda: _start(mgr))
        else:
            joins: List[object] = []

            def _join_times_out(timeout=None):
                if not joins:
                    joins.append(timeout)
                    raise TimeoutError("forced: the swap's join on the old fit timed out")
                return real_result(timeout)

            def _fetch(**_cfg) -> None:
                mgr._train_x, mgr._train_y = _xy()

            monkeypatch.setattr(old_future, "result", _join_times_out)
            monkeypatch.setattr(mgr, "_reload_dataset", _fetch)
            monkeypatch.setattr(mgr, "save_snapshot", lambda **_kwargs: {"id": "snap"})
            other = _Call(lambda: mgr.swap_dataset_live(dataset_type="spiral"))
        assert recording.contended.wait(_WAIT), f"the {verb} never waited for the pause"
    finally:
        go.set()
    assert pause.result() == _REFUSED, "the old fit's stop was raised before the pause's act"
    assert other.result() == "ok"
    # When the verb had to wait for the pause, the run it submits had not been submitted yet.
    assert recording.future_at_contention is not mgr._training_future, f"the {verb} submitted its run while the pause was between its check and its act"
    old.release.set()
    assert real_result(timeout=_WAIT) is None
    _parked(fit, 1).release.set()
    assert mgr._training_future.result(timeout=_WAIT) is None


# ---------------------------------------------------------------------------
# Route 2: a pause in the window a live swap opens, and the settles that end a run
# ---------------------------------------------------------------------------


def _swap_whose_join_times_out(mgr, fit: _GatedFit, monkeypatch, during_join: Callable[[], None] = lambda: None) -> Tuple[_Run, Future, Future]:
    """Start run 0, then live-swap while its fit is between checkpoints, forcing the swap's 10 s join to time out.

    ``during_join`` runs inside that join, on the swap's thread: the swap has raised the old fit's stop
    (step 5) and not yet submitted the swapped run (step 12).
    """
    mgr.set_experimental_functions(True)
    old_future = _start(mgr)
    old = _parked(fit, 0)
    real_result = old_future.result
    joins: List[object] = []

    def _join_times_out(timeout=None):
        if not joins:
            joins.append(timeout)
            during_join()
            raise TimeoutError("forced: the swap's join on the old fit timed out")
        return real_result(timeout)

    def _fetch(**_cfg) -> None:
        mgr._train_x, mgr._train_y = _xy()

    monkeypatch.setattr(old_future, "result", _join_times_out)
    monkeypatch.setattr(mgr, "_reload_dataset", _fetch)
    monkeypatch.setattr(mgr, "save_snapshot", lambda **_kwargs: {"id": "snap"})
    assert mgr.swap_dataset_live(dataset_type="spiral")["status"] == "swapped"
    assert joins == [10], "the swap never joined the old fit"
    return old, old_future, mgr._training_future


@pytest.mark.unit
@pytest.mark.parametrize("when", ["while the swap joins the old fit", "after the swap's join timed out"])
def test_a_pause_while_a_live_swap_stops_the_old_fit_is_refused_and_the_swapped_run_trains(lifecycle, monkeypatch, when):
    """The FSM still reads STARTED for the old fit, whose stop the swap has raised: a pause is refused, as one sent after a stop is."""
    mgr, fit, gate, counted = lifecycle()
    in_join: List[object] = []

    def _pause_in_the_join() -> None:
        in_join.append((_fields(mgr), _answer(mgr.pause_training), mgr._pause_event.is_set()))

    during_join = _pause_in_the_join if when == "while the swap joins the old fit" else (lambda: None)
    old, old_future, swapped = _swap_whose_join_times_out(mgr, fit, monkeypatch, during_join)
    if when == "while the swap joins the old fit":
        # Before this change: "ok", the FSM PAUSED, and the gate closed for a fit the swap was stopping.
        assert in_join == [(("STARTED", "Started"), _REFUSED, True)]
        assert _fields(mgr) == ("STARTED", "Started"), "the old fit has not settled"
    else:
        assert _fields(mgr) == ("STARTED", "Started"), "the old fit has not settled"
        # Before this change: "ok" and PAUSED; once the old fit stopped, 0.14.0 held the swapped run at
        # its first checkpoint under STARTED.
        assert _answer(mgr.pause_training) == _REFUSED
        assert _fields(mgr) == ("STARTED", "Started")
        assert mgr._pause_event.is_set(), "the refused pause closed the gate"

    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    assert not old.ran_on.is_set(), "the stopped old fit trained on"
    _passes_its_first_checkpoint(fit, gate, 1)
    assert swapped.result(timeout=_WAIT) is None
    assert _fields(mgr) == ("COMPLETED", "Completed")
    assert counted == [_CANCELLED, _SUCCESS]


@pytest.mark.unit
def test_a_pause_in_progress_when_a_live_swap_submits_its_run_does_not_pause_the_swapped_run(lifecycle, monkeypatch):
    """The swap opens the gate under the pause's lock, so a pause checked against the old fit cannot land on the swapped run."""
    mgr, fit, gate, counted = lifecycle()
    mgr.set_experimental_functions(True)
    old_future = _start(mgr)
    old = _parked(fit, 0)

    def _fetch(**_cfg) -> None:
        mgr._train_x, mgr._train_y = _xy()

    real_result = old_future.result
    joining = threading.Event()

    def _join(timeout=None):
        joining.set()
        return real_result(timeout)

    monkeypatch.setattr(old_future, "result", _join)
    monkeypatch.setattr(mgr, "_reload_dataset", _fetch)
    monkeypatch.setattr(mgr, "save_snapshot", lambda **_kwargs: {"id": "snap"})
    hold = _hold_after_the_pause_check(mgr, monkeypatch)
    pause = _Call(mgr.pause_training)
    assert hold.reached.wait(_WAIT), "the pause never reached its FSM check"
    swap = _Call(lambda: mgr.swap_dataset_live(dataset_type="spiral"))
    assert joining.wait(_WAIT), "the swap never joined the old fit"
    old.release.set()  # the swap has raised the old fit's stop: it stops at this checkpoint
    # With the fix the swap's gate open waits for the pause; without it the swapped run enters first.
    assert _either(hold.contended, fit.run(1).parked)
    hold.remove()

    # Before the fix: "ok", the PAUSE having landed on the swapped run's STARTED, which it then held.
    assert pause.result() == _REFUSED
    assert swap.result() == "ok"
    assert old_future.result(timeout=_WAIT) is None
    assert not old.ran_on.is_set()
    _passes_its_first_checkpoint(fit, gate, 1)
    assert mgr._training_future.result(timeout=_WAIT) is None
    assert counted == [_CANCELLED, _SUCCESS]


@pytest.mark.unit
@pytest.mark.parametrize("settle", ["interrupted at its checkpoint", "stopped after its last checkpoint", "failed"])
def test_a_paused_run_that_settles_reopens_the_gate(lifecycle, monkeypatch, settle):
    """Each settle a run owns reopens the gate after its transition, as the completion always did.

    The gate's state is recorded as the settle issues its transition: still closed there, so the reopen
    comes after it. A pause landing between a reopen and the transition would otherwise be accepted and
    leave the gate closed under the run queued next.
    """
    mgr, fit, gate, counted = lifecycle()
    run = fit.run(0)
    if settle == "stopped after its last checkpoint":
        run.after_release = ()
    elif settle == "failed":
        run.raises = RuntimeError("the fit failed")
    sm = mgr.state_machine
    gate_at_the_transition: List[bool] = []
    if settle == "failed":
        real_mark_failed = sm.mark_failed

        def _recording_mark_failed(reason="Unknown error"):
            gate_at_the_transition.append(mgr._pause_event.is_set())
            return real_mark_failed(reason)

        monkeypatch.setattr(sm, "mark_failed", _recording_mark_failed)
    else:
        real_handle = sm.handle_command

        def _recording_handle(command):
            if command is Command.STOP and sys._getframe(1).f_code.co_name == "_run_training":
                gate_at_the_transition.append(mgr._pause_event.is_set())
            return real_handle(command)

        monkeypatch.setattr(sm, "handle_command", _recording_handle)
    future = _start(mgr)
    _parked(fit, 0)
    assert _answer(mgr.pause_training) == "ok"
    if settle == "interrupted at its checkpoint":
        run.release.set()
        assert gate.held_a_fit.wait(_WAIT), "the paused fit was not held at its checkpoint"
        mgr.stop_training()
    elif settle == "stopped after its last checkpoint":
        mgr.stop_training()
        run.release.set()
    else:
        run.release.set()

    if settle == "failed":
        with pytest.raises(RuntimeError, match="the fit failed"):
            future.result(timeout=_WAIT)
        assert _fields(mgr) == ("FAILED", "Failed")
        assert counted == [_FAILURE]
    else:
        assert future.result(timeout=_WAIT) is None
        assert _fields(mgr) == ("STOPPED", "Stopped")
        assert counted == [_CANCELLED]
    assert gate_at_the_transition == [False], "the settle reopened the gate before its transition"
    # Before the fix: closed, so a run already queued behind this one would be held at its first checkpoint.
    assert mgr._pause_event.is_set(), "the run settled and left the gate closed"


@pytest.mark.unit
def test_a_pause_inside_a_failing_runs_settle_does_not_hold_the_run_queued_behind_it(lifecycle, monkeypatch):
    """A run whose stop is not raised fails; a pause lands in its settle just before ``mark_failed``: the next run trains.

    Two starts after a stop queue two runs that share one stop signal, so the first is a run a pause may
    rightly hold. Its failure settle reopens the gate after ``mark_failed``; reopened before it, the
    pause accepted in between would leave the gate closed under the second run.
    """
    mgr, fit, gate, counted = lifecycle()
    old_future = _start(mgr)
    old = _parked(fit, 0)
    mgr.stop_training()
    first = _start(mgr)  # queued behind the stopped fit
    second = _start(mgr)  # the FSM still reads STOPPED: queued behind the first
    fit.run(1).raises = RuntimeError("the first queued run failed")
    sm = mgr.state_machine
    real_mark_failed = sm.mark_failed
    landed: List[str] = []

    def _pause_lands_first(reason="Unknown error"):
        if not landed:
            landed.append(_answer(mgr.pause_training))  # STARTED for the failing run, its stop not raised
        return real_mark_failed(reason)

    monkeypatch.setattr(sm, "mark_failed", _pause_lands_first)
    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    _parked(fit, 1).release.set()
    with pytest.raises(RuntimeError, match="the first queued run failed"):
        first.result(timeout=_WAIT)
    assert landed == ["ok"], "the pause did not land inside the failing run's settle"
    _passes_its_first_checkpoint(fit, gate, 2)
    assert second.result(timeout=_WAIT) is None
    assert _fields(mgr) == ("COMPLETED", "Completed")
    assert counted == [_CANCELLED, _FAILURE, _SUCCESS]


@pytest.mark.unit
def test_a_pause_that_lands_as_the_stopped_old_fit_settles_is_refused_and_the_swapped_run_trains(lifecycle, monkeypatch):
    """The old fit's stop is raised (the swap raised it): a pause landing in its settle is refused, and the swapped run trains."""
    mgr, fit, gate, counted = lifecycle()
    old, old_future, swapped = _swap_whose_join_times_out(mgr, fit, monkeypatch)
    sm = mgr.state_machine
    real_handle = sm.handle_command
    landed: List[str] = []

    def _pause_lands_first(command):
        if command is Command.STOP and sys._getframe(1).f_code.co_name == "_run_training" and not landed:
            landed.append(_answer(mgr.pause_training))  # the FSM still reads STARTED for the old fit
        return real_handle(command)

    monkeypatch.setattr(sm, "handle_command", _pause_lands_first)
    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    # Before this change: "ok", a pause accepted for a fit that was settling STOPPED, which the swapped run never saw.
    assert landed == [_REFUSED], "the pause did not land inside the old fit's settle, or was accepted there"
    _passes_its_first_checkpoint(fit, gate, 1)
    assert swapped.result(timeout=_WAIT) is None
    assert counted == [_CANCELLED, _SUCCESS]


@pytest.mark.unit
@pytest.mark.parametrize("ending", ["interrupted", "failed"])
def test_a_stale_run_that_settles_leaves_the_newer_runs_pause_alone(lifecycle, monkeypatch, ending):
    """The reopen sits inside each settle's Overtoadship check, so a fit outliving ``shutdown()`` cannot release a newer pause."""
    mgr, fit, _gate, counted = lifecycle()
    stale = fit.run(0)
    if ending == "failed":
        stale.raises = RuntimeError("the stale fit failed")
    stale_future = _start(mgr)
    _parked(fit, 0)
    mgr.stop_training()
    monkeypatch.setattr(manager_module, "_SHUTDOWN_TRAINING_JOIN_TIMEOUT_SECONDS", 0.05)
    mgr.shutdown()
    assert not stale_future.done(), "the stale fit unwound during shutdown -- the overlap was not built"
    newer_future = _start(mgr)
    newer = _parked(fit, 1)
    assert _answer(mgr.pause_training) == "ok"

    stale.release.set()
    if ending == "failed":
        with pytest.raises(RuntimeError, match="the stale fit failed"):
            stale_future.result(timeout=_WAIT)
    else:
        assert stale_future.result(timeout=_WAIT) is None
    assert not mgr._pause_event.is_set(), "the stale run's settle released the newer run's pause"
    assert _fields(mgr) == ("PAUSED", "Paused")
    assert _answer(mgr.resume_training) == "ok"
    newer.release.set()
    assert newer_future.result(timeout=_WAIT) is None
    assert counted == [_SUCCESS]


# ---------------------------------------------------------------------------
# What must still hold
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_pause_of_a_running_fit_holds_it_at_its_next_checkpoint_until_resume(lifecycle):
    """A legitimate pause still pauses: the fit waits at its next checkpoint, and resume lets it go on."""
    mgr, fit, gate, counted = lifecycle()
    future = _start(mgr)
    run = _parked(fit, 0)
    run.hold_after_checkpoint = threading.Event()
    assert _answer(mgr.pause_training) == "ok"
    assert _fields(mgr) == ("PAUSED", "Paused")
    run.release.set()
    assert gate.held_a_fit.wait(_WAIT), "the paused fit was not held at its checkpoint"
    assert not run.ran_on.is_set()
    assert _answer(mgr.resume_training) == "ok"
    assert run.ran_on.wait(_WAIT), "resume did not let the fit go on"
    run.hold_after_checkpoint.set()  # the run settles only now, after the resume's own Started write
    assert future.result(timeout=_WAIT) is None
    assert _fields(mgr) == ("COMPLETED", "Completed")
    assert counted == [_SUCCESS]


@pytest.mark.unit
def test_a_pause_that_finds_a_new_run_started_is_not_refused_for_the_stopped_runs_stop(lifecycle, monkeypatch):
    """A run records its stop signal before its START, so a pause landing right after that START holds the new run."""
    mgr, fit, gate, counted = lifecycle()
    old_future = _start(mgr)
    old = _parked(fit, 0)
    mgr.stop_training()  # the stopped run's signal stays raised; the next start gets a fresh one
    new_future = _start(mgr)
    sm = mgr.state_machine
    real_handle = sm.handle_command
    landed: List[str] = []

    def _pause_right_after_the_start(command):
        accepted = real_handle(command)
        if command is Command.START and sys._getframe(1).f_code.co_name == "_run_training" and not landed:
            landed.append(_answer(mgr.pause_training))  # on the new run's thread, before its fit begins
        return accepted

    monkeypatch.setattr(sm, "handle_command", _pause_right_after_the_start)
    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    new = _parked(fit, 1)
    new.hold_after_checkpoint = threading.Event()
    assert landed == ["ok"], "a pause of the new run was refused for the stopped run's stop"
    assert sm.status.name == "PAUSED"
    new.release.set()
    assert gate.held_a_fit.wait(_WAIT), "the paused new run was not held at its checkpoint"
    assert _answer(mgr.resume_training) == "ok"
    assert new.ran_on.wait(_WAIT), "resume did not let the new run go on"
    new.hold_after_checkpoint.set()  # the run settles only now, after the resume's own Started write
    assert new_future.result(timeout=_WAIT) is None
    assert _fields(mgr) == ("COMPLETED", "Completed")
    assert counted == [_CANCELLED, _SUCCESS]


@pytest.mark.unit
def test_a_resume_cannot_land_between_a_pauses_transition_and_its_paused_write(lifecycle, monkeypatch):
    """Before the fix, such a resume left ``training_state`` reading Paused beside a STARTED run that trains."""
    mgr, fit, _gate, counted = lifecycle()
    future = _start(mgr)
    run = _parked(fit, 0)
    hold = _Hold(mgr)
    real_update = mgr.training_state.update_state

    def _update_state(**fields):
        if fields.get("status") == "Paused" and sys._getframe(1).f_code.co_name == "pause_training":
            hold.wait_here()
        return real_update(**fields)

    monkeypatch.setattr(mgr.training_state, "update_state", _update_state)
    pause = _Call(mgr.pause_training)
    assert hold.reached.wait(_WAIT), "the pause never reached its Paused write"
    resume = _Call(mgr.resume_training)
    assert _either(resume.done, hold.contended)
    hold.remove()

    assert pause.result() == "ok"
    assert resume.result() == "ok"
    # Before the fix: ("STARTED", "Paused"), the resume having run whole between the two.
    assert _fields(mgr) == ("STARTED", "Started")
    assert hold.contended.is_set(), "the resume ran inside the pause's step"
    assert mgr._pause_event.is_set()
    run.release.set()
    assert future.result(timeout=_WAIT) is None
    assert counted == [_SUCCESS]


# ---------------------------------------------------------------------------
# Over the routes: the pause over /ws/control, the stop and the start over REST
# ---------------------------------------------------------------------------


@pytest.fixture
def http_stack(monkeypatch):
    """The real app and lifecycle, with the network's fit gated and the pause gate watched."""
    counted: List[str] = []
    monkeypatch.setattr(manager_module, "inc_training_session_completed", counted.append)
    app = create_app(Settings(auto_start=False))
    with TestClient(app) as client:
        assert client.post("/v1/network", json={"input_size": 2, "output_size": 2}).status_code == 200
        lifecycle = client.app.state.lifecycle
        fit = _GatedFit()
        lifecycle.model.fit = fit
        gate = _WatchedGate()
        lifecycle._pause_event = gate
        try:
            yield client, lifecycle, fit, gate, counted
        finally:
            fit.release_all()
            lifecycle._pause_event.set()


@pytest.mark.unit
def test_a_ws_control_pause_overlapped_by_a_rest_stop_and_start_is_refused_and_the_new_run_trains(http_stack, monkeypatch):
    """What a client sees: the pause's ack is an error, the status reads STOPPED, and the new run trains."""
    client, lifecycle, fit, gate, counted = http_stack
    x, y = _xy()
    payload = {"inline_data": {"train_x": x.tolist(), "train_y": y.tolist()}}
    assert client.post("/v1/training/start", json=payload).status_code == 200
    old_future = lifecycle._training_future
    old = _parked(fit, 0)
    hold = _hold_after_the_pause_check(lifecycle, monkeypatch)
    with client.websocket_connect("/ws/control", headers={"origin": _CANOPY_ORIGIN}) as ws:
        assert ws.receive_json()["type"] == "connection_established"
        ws.send_text(json.dumps({"command": "pause", "command_id": "overlap"}))
        assert hold.reached.wait(_WAIT), "the /ws/control pause never reached its FSM check"
        assert client.post("/v1/training/stop").status_code == 200
        # The REST start runs on the event loop; with the fix it waits for the held pause, which then goes on.
        start = client.post("/v1/training/start", json=payload)
        hold.remove()
        ack = ws.receive_json()
    assert start.status_code == 200, start.text
    # Before the fix: "success" with ``{"status": "paused"}``, and the new run then held under STARTED.
    assert ack["data"]["command"] == "pause"
    assert ack["data"]["status"] == "error", ack
    status = client.get("/v1/training/status").json()["data"]
    assert (status["state_machine"]["status"], status["training_state"]["status"], status["training_active"]) == ("STOPPED", "Stopped", False)
    new_future = lifecycle._training_future

    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    _passes_its_first_checkpoint(fit, gate, 1)
    assert new_future.result(timeout=_WAIT) is None
    status = client.get("/v1/training/status").json()["data"]
    assert (status["state_machine"]["status"], status["training_state"]["status"]) == ("COMPLETED", "Completed")
    assert counted == [_CANCELLED, _SUCCESS]


def _rest_status(client) -> Tuple[str, str]:
    status = client.get("/v1/training/status").json()["data"]
    return status["state_machine"]["status"], status["training_state"]["status"]


def _held_request(client, path: str) -> Tuple[threading.Thread, List[object]]:
    """POST ``path`` from its own thread; the route runs on the app's event loop, where the test's hold parks it."""
    responses: List[object] = []
    sender = threading.Thread(target=lambda: responses.append(client.post(path)), daemon=True)
    sender.start()
    return sender, responses


@pytest.mark.unit
def test_a_rest_pause_that_loses_its_transition_answers_409(http_stack, monkeypatch):
    """The run settles between the REST pause's check and its PAUSE: the route answers 409, not 200."""
    client, lifecycle, fit, _gate, counted = http_stack
    x, y = _xy()
    payload = {"inline_data": {"train_x": x.tolist(), "train_y": y.tolist()}}
    assert client.post("/v1/training/start", json=payload).status_code == 200
    old_future = lifecycle._training_future
    old = _parked(fit, 0)
    hold = _hold_after_the_pause_check(lifecycle, monkeypatch)
    sender, responses = _held_request(client, "/v1/training/pause")
    assert hold.reached.wait(_WAIT), "the REST pause never reached its FSM check"
    old.release.set()  # the gate is still open: the fit completes and settles COMPLETED
    assert old_future.result(timeout=_WAIT) is None
    hold.remove()
    sender.join(_WAIT)
    assert responses, "the REST pause never answered"
    # Before the fix: 200 "paused", with nothing paused and ``training_state`` Paused beside COMPLETED.
    assert responses[0].status_code == 409, responses[0].text
    assert "Training cannot be paused in the current state" in responses[0].text
    assert _rest_status(client) == ("COMPLETED", "Completed")
    assert counted == [_SUCCESS]


@pytest.mark.unit
def test_a_rest_pause_after_a_timed_out_live_swap_answers_409_and_the_swapped_run_trains(http_stack, monkeypatch):
    """Over REST alone: the swap's join on the old fit times out, then a pause is refused, as one sent after a stop is."""
    client, lifecycle, fit, gate, counted = http_stack
    x, y = _xy()
    payload = {"inline_data": {"train_x": x.tolist(), "train_y": y.tolist()}}
    assert client.post("/v1/admin/experimental_functions", json={"enabled": True}).status_code == 200
    assert client.post("/v1/training/start", json=payload).status_code == 200
    old_future = lifecycle._training_future
    old = _parked(fit, 0)
    real_result = old_future.result
    joins: List[object] = []

    def _join_times_out(timeout=None):
        if not joins:
            joins.append(timeout)
            raise TimeoutError("forced: the swap's join on the old fit timed out")
        return real_result(timeout)

    def _fetch(**_cfg) -> None:
        lifecycle._train_x, lifecycle._train_y = _xy()

    monkeypatch.setattr(old_future, "result", _join_times_out)
    monkeypatch.setattr(lifecycle, "_reload_dataset", _fetch)
    monkeypatch.setattr(lifecycle, "save_snapshot", lambda **_kwargs: {"id": "snap"})
    swap = client.post("/v1/training/dataset/live", json={"dataset_type": "spirals"})
    assert swap.status_code == 200, swap.text
    assert joins == [10], "the swap never joined the old fit"
    swapped = lifecycle._training_future
    assert _rest_status(client) == ("STARTED", "Started"), "the old fit has not settled"

    pause = client.post("/v1/training/pause")
    # Before this change: 200 "paused"; once the old fit stopped, 0.14.0 held the swapped run at its first checkpoint.
    assert pause.status_code == 409, pause.text
    assert "Training cannot be paused in the current state" in pause.text
    assert _rest_status(client) == ("STARTED", "Started")

    old.release.set()
    assert real_result(timeout=_WAIT) is None
    assert not old.ran_on.is_set(), "the stopped old fit trained on"
    _passes_its_first_checkpoint(fit, gate, 1)
    assert swapped.result(timeout=_WAIT) is None
    assert _rest_status(client) == ("COMPLETED", "Completed")
    assert counted == [_CANCELLED, _SUCCESS]


@pytest.mark.unit
def test_a_rest_resume_that_loses_its_transition_answers_409(http_stack, monkeypatch):
    """The paused run's fit returns and settles between the REST resume's check and its RESUME: 409, not 200."""
    client, lifecycle, fit, _gate, counted = http_stack
    x, y = _xy()
    payload = {"inline_data": {"train_x": x.tolist(), "train_y": y.tolist()}}
    run = fit.run(0)
    run.after_release = ()  # no checkpoint left: the paused run's fit returns
    assert client.post("/v1/training/start", json=payload).status_code == 200
    old_future = lifecycle._training_future
    _parked(fit, 0)
    assert client.post("/v1/training/pause").status_code == 200
    hold = _hold_after_the_resume_check(lifecycle, monkeypatch)
    sender, responses = _held_request(client, "/v1/training/resume")
    assert hold.reached.wait(_WAIT), "the REST resume never reached its FSM check"
    run.release.set()  # the fit returns while PAUSED and settles COMPLETED
    assert old_future.result(timeout=_WAIT) is None
    hold.remove()
    sender.join(_WAIT)
    assert responses, "the REST resume never answered"
    # Before this change: 200 "resumed", and ``training_state`` Started beside COMPLETED.
    assert responses[0].status_code == 409, responses[0].text
    assert "Training cannot be resumed in the current state" in responses[0].text
    assert _rest_status(client) == ("COMPLETED", "Completed")
    assert counted == [_SUCCESS]


# ---------------------------------------------------------------------------
# Stress: the same race with no hook, at a 1 us thread switch interval
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_pause_racing_a_stop_and_a_start_never_misreports_nor_leaves_the_gate_closed(lifecycle, monkeypatch):
    """A pause on one thread, a stop then a start on another, released together, many times.

    The run the start queues is not executed (``_run_training`` is stubbed): only the answers and
    the gate are judged. Each trial's START stands for a run entering ``_run_training``, so the trial
    records the newest stop signal as the lifecycle Overtoad's, as ``_run_training`` does; without that
    every pause after the first trial's stop would be refused at its check and the race never run.
    Before the fix about one trial in thirty answered ``paused`` although its PAUSE had been refused,
    and about one in fifty thousand left the gate closed under the start.
    """
    mgr, _fit, _gate, counted = lifecycle()
    monkeypatch.setattr(mgr, "_run_training", lambda *args, **kwargs: None)
    sm = mgr.state_machine
    real_handle = sm.handle_command
    pause_accepted: List[bool] = []

    def _recording_handle(command):
        accepted = real_handle(command)
        if command is Command.PAUSE:
            pause_accepted.append(accepted)
        return accepted

    monkeypatch.setattr(sm, "handle_command", _recording_handle)
    x, y = _xy()

    def _pauser(barrier: threading.Barrier, answers: dict) -> None:
        barrier.wait()
        answers["pause"] = _answer(mgr.pause_training)

    def _stop_then_start(barrier: threading.Barrier, answers: dict) -> None:
        barrier.wait()
        mgr.stop_training()
        answers["start"] = _answer(lambda: mgr.start_training(X=x, y=y))

    findings: List[Tuple[int, str]] = []
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for trial in range(_STRESS_TRIALS):
            real_handle(Command.RESET)
            mgr._lifecycle_stop_event = mgr._stop_event
            assert real_handle(Command.START)
            mgr._pause_event.set()
            pause_accepted.clear()
            barrier = threading.Barrier(2)
            answers: dict = {}
            threads = [threading.Thread(target=_pauser, args=(barrier, answers)), threading.Thread(target=_stop_then_start, args=(barrier, answers))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(_WAIT)
            if answers.get("pause") == "ok" and not (pause_accepted and pause_accepted[0]):
                findings.append((trial, "the pause answered paused although its PAUSE was refused"))
            if answers.get("start") == "ok" and not mgr._pause_event.is_set() and sm.status.name == "STOPPED":
                findings.append((trial, "the gate was left closed under the run the start queued"))
    finally:
        sys.setswitchinterval(previous)
    assert findings == []
    assert counted == []
