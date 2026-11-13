#!/usr/bin/env python
"""Pin how ``TrainingLifecycleManager`` settles a fit that returns while its run is PAUSED.

Pause is cooperative. ``pause_training`` moves the FSM to PAUSED and clears ``_pause_event``,
but the fit waits on that event only at an interrupt checkpoint (``_check_for_interrupt``),
which ``_handle_event`` runs on ``epoch_end`` and ``phase_change`` alone. Everything after a
fit's last checkpoint runs without one: the rest of the final output pass after its
last-epoch callback, ``CascorModel.fit``'s post-hoc ``unit_added`` / ``training_end`` events,
and ``_run_training``'s own metrics drain, terminal weight capture and final evaluation. A
pause that lands there cannot hold the fit, so the fit returns while the FSM is PAUSED.

Before this fix ``_run_training`` then called ``mark_completed()``, which accepted only STARTED,
ignored the refusal and wrote ``training_state`` ``Completed``. The FSM stayed PAUSED with no
thread behind it. A start answered 409 ("Cannot start training while paused"), a resume moved
the FSM to STARTED (still no thread), and every later start answered 409 ("Training already in
progress") until a stop or reset. This is the "paused-fit wedge". juniper-ml's flood-4 record,
``notes/JUNIPER_2026-11-06_JUNIPER-ECOSYSTEM_CURSOR-FLOOD-4-DISPOSITION.md``, lists this cascor
copy under §8 ruling 11 (owner ruling "Fix all four now", 2026-11-08); its §4 records the same
wedge in juniper-service-core, whose lifecycle mirrors this one.

The run now settles COMPLETED exactly as an unpaused run does: ``mark_completed`` accepts PAUSED
as well as STARTED, and the pause gate is reopened after it. That is one FSM transition, made
under the state machine's own lock, so a pause or resume that races the settle either lands
wholly first and is settled with the run, or finds COMPLETED and is refused. Releasing the pause
through RESUME and then calling ``mark_completed`` left a STARTED step between two unlocked
transitions, and a second pause landing there re-created the wedge; without the lock, a resume
whose own transition straddled the settle's could still write STARTED over COMPLETED. The
forced-pause and forced-resume tests pin that neither can happen now. A stop that lands before
the fit returns still settles STOPPED, a resume that lands before it still completes the same
run, and a fit that raises while paused still settles FAILED.

Only the run that owns the lifecycle settles it. A run takes ownership as it enters
``_run_training``, before it issues ``Command.START``, so a fit that returns, is interrupted or
fails after a newer run has entered leaves the FSM, the training state, the final score, the
terminal counter and the monitor session to that newer run. juniper-service-core's own fix for
the same wedge first released the pause without such a check, and its validation showed a stale
fit then resuming or completing a newer paused run. cascor's single-worker executor serializes
runs, so a newer run can enter while an older fit is still alive only once a run is started
after ``shutdown()`` dropped the executor, its bounded join having timed out. The ownership tests
build the overlap that way.
"""

import threading
from concurrent.futures import Future
from typing import Callable, List, Optional, Tuple

import pytest
import torch
from fastapi.testclient import TestClient
from juniper_model_core.events import TrainingEvent

from api.app import create_app
from api.lifecycle import manager as manager_module
from api.lifecycle.manager import TrainingInterrupted, TrainingLifecycleManager
from api.settings import Settings

# Bounded, so a failed assertion can never leave a fit thread parked for good.
_GATE_TIMEOUT = 10.0
_WAIT = 5.0
# How long a run must stay held to count as held. Shorter than ``_check_for_interrupt``'s
# 0.5 s poll would prove nothing about a stop; it is used only for pause holds.
_HELD = 0.5

# What the FSM reports after a stub fit that never grew settles normally (no pause involved):
# START put it in OUTPUT and nothing moved it, and COMPLETED keeps the phase.
_COMPLETED = {"status": "COMPLETED", "phase": "OUTPUT", "paused_phase": None, "has_candidate_state": False}


def _training_start(seq: int = 0) -> TrainingEvent:
    return TrainingEvent("training_start", {"n_samples": 8}, seq)


def _epoch_end(seq: int = 1) -> TrainingEvent:
    """An output-pass ``epoch_end``: one of the two event types ``_handle_event`` interrupt-checks."""
    return TrainingEvent("epoch_end", {"epoch": 25, "epochs": 50, "metrics": {"loss": 0.5}}, seq)


def _training_end(seq: int = 2) -> TrainingEvent:
    """``CascorModel.fit``'s last event. ``_handle_event`` runs no interrupt checkpoint on it."""
    return TrainingEvent("training_end", {"metrics": {}}, seq)


# The events a run's fit emits before it parks, per variant:
#   after-last-checkpoint -- a checkpointed epoch_end it passes, then training_end: where a real
#                            fit is once its final output pass's last-epoch callback has run;
#   no-checkpoint         -- only the events no checkpoint is run on.
_BEFORE_PARK = {
    "after-last-checkpoint": (_training_start(0), _epoch_end(1), _training_end(2)),
    "no-checkpoint": (_training_start(0), _training_end(1)),
}


class _Run:
    """The gates of one call of the stub fit, which is one training run."""

    def __init__(self) -> None:
        self.parked = threading.Event()  # set once the fit has emitted its events and parked
        self.release = threading.Event()  # the test sets it to let the fit go on
        self.after_release: Tuple[TrainingEvent, ...] = ()  # emitted once released, e.g. to reach a checkpoint
        self.raises: Optional[BaseException] = None  # raised once released, after ``after_release``


class _GatedFit:
    """Stands in for ``CascorModel.fit``, installed as ``mgr.model.fit`` (the house pattern).

    Each call is one run. It sends its events through the manager's real ``on_event`` sink
    (``_handle_event``), parks on its own gate, then -- once the test releases it -- sends any
    ``after_release`` events, and returns or raises.
    """

    def __init__(self, before_park: Tuple[TrainingEvent, ...] = _BEFORE_PARK["after-last-checkpoint"]) -> None:
        self.before_park = before_park
        self.calls = 0
        self._runs: List[_Run] = []
        self._lock = threading.Lock()

    def run(self, index: int) -> _Run:
        with self._lock:
            while len(self._runs) <= index:
                self._runs.append(_Run())
            return self._runs[index]

    def __call__(self, x, y, *, X_val=None, y_val=None, on_event=None, **kwargs):
        with self._lock:
            index = self.calls
            self.calls += 1
        run = self.run(index)
        for event in self.before_park:
            on_event(event)
        run.parked.set()
        run.release.wait(timeout=_GATE_TIMEOUT)
        for event in run.after_release:
            on_event(event)
        if run.raises is not None:
            raise run.raises

    def release_all(self) -> None:
        with self._lock:
            runs = list(self._runs)
        for run in runs:
            run.release.set()


def _parked(fit: _GatedFit, index: int) -> _Run:
    run = fit.run(index)
    assert run.parked.wait(timeout=_WAIT), f"run {index}'s fit never reached its gate"
    return run


def _xy() -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(8, 2)
    y = torch.zeros(8, 2)
    y[:4, 0] = 1
    y[4:, 1] = 1
    return x, y


def _settle(mgr: TrainingLifecycleManager) -> None:
    """Wait for the current run's ``_run_training`` to return."""
    mgr._training_future.result(timeout=_WAIT)


def _checkpoint_returns(mgr: TrainingLifecycleManager) -> bool:
    """Whether an interrupt checkpoint passes now, i.e. no pause is left pending.

    ``_check_for_interrupt`` is what every run's next ``epoch_end`` / ``phase_change`` meets. It
    runs on a helper thread with a deadline because a pending pause blocks it; the thread is
    released by the stop signal the fixture's ``shutdown()`` raises.
    """
    done = threading.Event()

    def _probe() -> None:
        try:
            mgr._check_for_interrupt()
            done.set()
        except TrainingInterrupted:
            # A pending pause held this probe until the fixture's shutdown() raised the stop
            # signal; the caller has already reported the hold as a failure.
            return

    threading.Thread(target=_probe, daemon=True, name="checkpoint-probe").start()
    return done.wait(timeout=_HELD * 4)


@pytest.fixture
def lifecycle():
    """A manager with a network whose fit is gated, cleaned up whatever the test did."""
    managers: List[Tuple[TrainingLifecycleManager, _GatedFit]] = []

    def _make(before_park: Tuple[TrainingEvent, ...] = _BEFORE_PARK["after-last-checkpoint"]):
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        fit = _GatedFit(before_park)
        mgr.model.fit = fit
        managers.append((mgr, fit))
        return mgr, fit

    yield _make
    for mgr, fit in managers:
        fit.release_all()
        mgr.shutdown()


def _pause_once_past_the_last_checkpoint(mgr: TrainingLifecycleManager, fit: _GatedFit) -> _Run:
    """Start a run and pause it once its fit has emitted its last event and parked."""
    x, y = _xy()
    mgr.start_training(X=x, y=y)
    run = _parked(fit, 0)
    mgr.pause_training()
    assert mgr.state_machine.get_state_summary()["status"] == "PAUSED"
    return run


# ---------------------------------------------------------------------------
# The wedge: a fit that returns while its run is paused
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("variant", sorted(_BEFORE_PARK))
def test_fit_that_returns_while_paused_settles_completed(lifecycle, variant):
    mgr, fit = lifecycle(_BEFORE_PARK[variant])
    run = _pause_once_past_the_last_checkpoint(mgr, fit)
    run.release.set()  # the fit returns; no checkpoint is left for the pause to hold it at
    _settle(mgr)

    status = mgr.get_status()
    # Before the fix: {"status": "PAUSED", "phase": "OUTPUT", "paused_phase": "OUTPUT", ...}
    # beside training_state "Completed".
    assert status["state_machine"] == _COMPLETED
    assert status["training_active"] is False
    assert status["training_state"]["status"] == "Completed"
    assert status["training_state"]["phase"] == "Idle"
    # The pause is released: the next run's first checkpoint passes instead of waiting.
    assert _checkpoint_returns(mgr), "a pause is still pending after the run settled"

    # A plain start, with no resume, runs a fresh fit, as after any completed run.
    x, y = _xy()
    mgr.start_training(X=x, y=y)
    _parked(fit, 1).release.set()
    _settle(mgr)
    assert fit.calls == 2
    assert mgr.get_status()["state_machine"] == _COMPLETED


@pytest.fixture
def http_stack():
    """The real app and lifecycle over HTTP, with the network's fit gated."""
    app = create_app(Settings(auto_start=False))
    with TestClient(app) as client:
        assert client.post("/v1/network", json={"input_size": 2, "output_size": 2}).status_code == 200
        lifecycle = client.app.state.lifecycle
        fit = _GatedFit()
        lifecycle.model.fit = fit
        try:
            yield client, lifecycle, fit
        finally:
            fit.release_all()


@pytest.mark.unit
def test_resume_then_start_after_a_paused_fit_returned_over_http(http_stack):
    """The defect's own sequence: pause, let the fit return, resume, then start."""
    client, lifecycle, fit = http_stack
    x, y = _xy()
    payload = {"inline_data": {"train_x": x.tolist(), "train_y": y.tolist()}}

    assert client.post("/v1/training/start", json=payload).status_code == 200
    run = _parked(fit, 0)
    paused = client.post("/v1/training/pause")
    assert paused.status_code == 200, paused.text
    assert client.get("/v1/training/status").json()["data"]["state_machine"]["status"] == "PAUSED"
    run.release.set()
    _settle(lifecycle)

    resume = client.post("/v1/training/resume")
    start = client.post("/v1/training/start", json=payload)
    # Before the fix the resume answered 200 (STARTED, with no thread behind it) and this start
    # answered 409, "Training cannot be started: Training already in progress".
    assert start.status_code == 200, start.text
    assert resume.status_code == 409, resume.text
    assert resume.json()["error"]["message"] == "Training cannot be resumed in the current state"

    _parked(fit, 1).release.set()
    _settle(lifecycle)
    status = client.get("/v1/training/status").json()["data"]
    assert status["state_machine"]["status"] == "COMPLETED"
    assert status["training_active"] is False
    assert fit.calls == 2


def _answer(request: Callable[[], object]) -> str:
    """Make a lifecycle request: "ok", or "refused: <reason>" for the RuntimeError a route maps to 409."""
    try:
        request()
        return "ok"
    except RuntimeError as exc:
        return f"refused: {exc}"


def _pause_from_another_thread(mgr: TrainingLifecycleManager) -> str:
    """Send ``pause_training`` from its own thread, as a request does, and wait for its answer."""
    answers: List[str] = []
    thread = threading.Thread(target=lambda: answers.append(_answer(mgr.pause_training)), daemon=True, name="pause-request")
    thread.start()
    thread.join(timeout=_WAIT)
    assert answers, "the pause request never answered"
    return answers[0]


@pytest.mark.unit
@pytest.mark.parametrize("paused_before", [True, False], ids=["after-a-pause-the-fit-outlived", "with-no-earlier-pause"])
def test_a_pause_that_lands_as_the_run_settles_cannot_wedge_it(lifecycle, monkeypatch, paused_before):
    """Force a pause request into the settle, just before its ``mark_completed`` transition.

    ``_run_training`` and ``pause_training`` share no lock, so a pause can land between any two of
    the settle's steps. One that lands before ``mark_completed`` must be settled with the run. A
    RESUME that released the earlier pause first left the FSM STARTED there, so such a pause was
    accepted and ``mark_completed``, which then accepted STARTED only, left the run PAUSED: the
    wedge again.
    """
    mgr, fit = lifecycle()
    x, y = _xy()
    mgr.start_training(X=x, y=y)
    run = _parked(fit, 0)
    if paused_before:
        mgr.pause_training()
    sm = mgr.state_machine
    mark_completed = sm.mark_completed
    answers: List[str] = []

    def _pause_lands_first() -> bool:
        if not answers:  # the first settle only
            answers.append(_pause_from_another_thread(mgr))
        return mark_completed()

    monkeypatch.setattr(sm, "mark_completed", _pause_lands_first)
    run.release.set()
    _settle(mgr)

    status = mgr.get_status()
    # Before: {"status": "PAUSED", ...} beside training_state "Completed", in both variants.
    assert status["state_machine"] == _COMPLETED
    assert status["training_active"] is False
    assert status["training_state"]["status"] == "Completed"
    assert _checkpoint_returns(mgr), "the pause gate was left closed after the run settled"
    # A run already PAUSED refuses the second pause; a STARTED one takes it into the settle.
    assert answers == (["refused: Training is not active"] if paused_before else ["ok"])

    x, y = _xy()
    mgr.start_training(X=x, y=y)
    _parked(fit, 1).release.set()
    _settle(mgr)
    assert fit.calls == 2
    assert mgr.get_status()["state_machine"] == _COMPLETED


@pytest.mark.unit
def test_a_resume_that_races_the_settle_cannot_strand_the_run(lifecycle, monkeypatch):
    """Force a resume's own transition to straddle the settle's ``mark_completed``.

    ``_handle_resume`` checks PAUSED and then calls ``_restore_from_paused``, which writes STARTED.
    The resume here is held at that call until the settle has tried its transition. Without the
    state machine's lock the settle completed the run inside that gap, and the resume then wrote
    STARTED over COMPLETED with no thread behind it: every later start was refused ("Training
    already in progress") until a stop or reset. With the lock the settle waits for the resume's
    transition to finish, then completes the run from STARTED.
    """
    mgr, fit = lifecycle()
    run = _pause_once_past_the_last_checkpoint(mgr, fit)
    sm = mgr.state_machine
    restore, mark_completed = sm._restore_from_paused, sm.mark_completed
    in_gap, settling, settled = threading.Event(), threading.Event(), threading.Event()

    def _restore_once_the_settle_has_tried() -> bool:
        in_gap.set()
        if settling.wait(timeout=_WAIT):
            settled.wait(timeout=_HELD)  # unlocked, the settle's transition completes here
        return restore()

    def _settle_transition() -> bool:
        settling.set()
        try:
            return mark_completed()
        finally:
            settled.set()

    monkeypatch.setattr(sm, "_restore_from_paused", _restore_once_the_settle_has_tried)
    monkeypatch.setattr(sm, "mark_completed", _settle_transition)
    answers: List[str] = []
    resume = threading.Thread(target=lambda: answers.append(_answer(mgr.resume_training)), daemon=True, name="resume-request")
    resume.start()
    try:
        assert in_gap.wait(timeout=_WAIT), "the resume never reached its STARTED write"
        run.release.set()
        _settle(mgr)
    finally:
        settling.set()
        resume.join(timeout=_WAIT)

    # Before: {"status": "STARTED", ...} with training_active true and no thread behind it.
    status = mgr.get_status()
    assert status["state_machine"] == _COMPLETED
    assert status["training_active"] is False
    assert answers == ["ok"], "the resume reached the machine first, so it was accepted"

    x, y = _xy()
    mgr.start_training(X=x, y=y)
    _parked(fit, 1).release.set()
    _settle(mgr)
    assert fit.calls == 2
    assert mgr.get_status()["state_machine"]["status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# Controls: what already worked keeps working
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resume_before_the_fit_returns_completes_the_same_run(lifecycle):
    mgr, fit = lifecycle()
    run = _pause_once_past_the_last_checkpoint(mgr, fit)
    mgr.resume_training()
    assert mgr.state_machine.get_state_summary()["status"] == "STARTED"
    run.release.set()
    _settle(mgr)

    assert mgr.get_status()["state_machine"] == _COMPLETED
    assert fit.calls == 1


@pytest.mark.unit
def test_stop_after_a_pause_the_fit_outlived_settles_stopped(lifecycle):
    mgr, fit = lifecycle()
    run = _pause_once_past_the_last_checkpoint(mgr, fit)
    mgr.stop_training()
    run.release.set()
    _settle(mgr)

    status = mgr.get_status()
    assert status["state_machine"] == {"status": "STOPPED", "phase": "IDLE", "paused_phase": None, "has_candidate_state": False}
    assert status["training_active"] is False
    assert status["training_state"]["status"] == "Stopped"
    assert mgr.get_metrics()["eval_metrics"]["final"] is None, "a stopped run reports no final score"


@pytest.mark.unit
def test_stop_while_paused_at_a_checkpoint_unwinds_the_fit(lifecycle):
    """A pause that lands BEFORE the fit's next checkpoint holds it there, and a stop unwinds it."""
    mgr, fit = lifecycle((_training_start(0),))
    x, y = _xy()
    mgr.start_training(X=x, y=y)
    run = _parked(fit, 0)
    run.after_release = (_epoch_end(1),)
    mgr.pause_training()
    run.release.set()  # the fit meets its next checkpoint while paused, and waits there

    with pytest.raises(TimeoutError):
        mgr._training_future.result(timeout=_HELD)
    mgr.stop_training()
    _settle(mgr)  # the checkpoint's 0.5 s poll observes the stop and unwinds the fit

    status = mgr.get_status()
    assert status["state_machine"]["status"] == "STOPPED"
    assert status["training_state"]["status"] == "Stopped"
    assert status["training_active"] is False


@pytest.mark.unit
def test_fit_that_raises_while_paused_settles_failed(lifecycle):
    mgr, fit = lifecycle()
    run = _pause_once_past_the_last_checkpoint(mgr, fit)
    run.raises = RuntimeError("fit failed after its last checkpoint")
    run.release.set()

    with pytest.raises(RuntimeError, match="after its last checkpoint"):
        _settle(mgr)
    status = mgr.get_status()
    assert status["state_machine"]["status"] == "FAILED"
    assert status["training_state"]["status"] == "Failed"
    assert status["training_active"] is False

    x, y = _xy()
    mgr.start_training(X=x, y=y)
    _parked(fit, 1).release.set()
    _settle(mgr)
    assert mgr.get_status()["state_machine"] == _COMPLETED


# ---------------------------------------------------------------------------
# Ownership: a stale fit leaves a newer run alone
# ---------------------------------------------------------------------------


def _leave_a_stale_fit_alive(mgr: TrainingLifecycleManager, fit: _GatedFit, monkeypatch) -> Tuple[_Run, Future, List[str]]:
    """Leave run 0's fit alive past its last event, with no executor behind it.

    Run 0 is stopped, then ``shutdown()``'s bounded join times out on its parked fit and drops
    the executor, so the next start gets a new executor and a new thread. Returns run 0's gate,
    its future, and the list the terminal session counter (``inc_training_session_completed``)
    now records into, so a test can tell which run counted a terminal status.

    Run 0's stop signal stays raised: since the stop-then-start fix a later start gives its own
    run a fresh signal and never clears this one. So a stale fit that returns here takes the
    post-fit STOP branch, and it is that branch's ownership check these tests exercise. The
    COMPLETION branch's check needs a stale run already past its stop check when the stop comes:
    ``test_lifecycle_stop_then_start.py``'s
    ``test_a_stale_run_past_its_stop_check_leaves_the_newer_paused_run_alone`` builds that.
    """
    counted: List[str] = []
    monkeypatch.setattr(manager_module, "inc_training_session_completed", counted.append)
    x, y = _xy()
    x_test, y_test = _xy()
    mgr.start_training(X=x, y=y, X_test=x_test, y_test=y_test)  # a test split, so a final score is computable
    stale = _parked(fit, 0)
    stale_future = mgr._training_future
    mgr.stop_training()
    monkeypatch.setattr(manager_module, "_SHUTDOWN_TRAINING_JOIN_TIMEOUT_SECONDS", 0.05)
    mgr.shutdown()
    assert not stale_future.done(), "run 0 unwound during shutdown -- the overlap was not built"
    return stale, stale_future, counted


def _overlap(mgr: TrainingLifecycleManager, fit: _GatedFit, monkeypatch, *, pause_newer: bool) -> Tuple[_Run, Future, List[str]]:
    """Leave run 0's fit alive past its last event while run 1 owns the lifecycle.

    Run 1 enters ``_run_training`` on the new executor's thread, takes the lifecycle and parks in
    its fit (paused, if asked). Returns what :func:`_leave_a_stale_fit_alive` returns.
    """
    stale, stale_future, counted = _leave_a_stale_fit_alive(mgr, fit, monkeypatch)
    x, y = _xy()
    mgr.start_training(X=x, y=y)
    _parked(fit, 1)
    if pause_newer:
        mgr.pause_training()
    assert mgr.state_machine.get_state_summary()["status"] == ("PAUSED" if pause_newer else "STARTED")
    return stale, stale_future, counted


def _assert_newer_run_untouched(mgr: TrainingLifecycleManager, counted: List[str], *, paused: bool) -> None:
    status = mgr.get_status()
    assert status["state_machine"]["status"] == ("PAUSED" if paused else "STARTED")
    assert status["training_active"] is (not paused)
    assert status["training_state"]["status"] == ("Paused" if paused else "Started")
    assert mgr.get_metrics()["eval_metrics"]["final"] is None, "the stale fit wrote a final score for the newer run"
    assert mgr.monitor.get_current_state()["is_training"] is True, "the stale fit ended the newer run's monitor session"
    assert counted == [], "the stale fit counted a terminal session"


@pytest.mark.unit
def test_a_stale_fit_that_returns_leaves_the_newer_paused_run_alone(lifecycle, monkeypatch):
    """The stale fit must not release the newer run's pause or settle it.

    Its stop is still raised (see ``_leave_a_stale_fit_alive``), so the settle it must not make is
    the stop's: STOPPED and a cancelled session for the newer, paused run.
    """
    mgr, fit = lifecycle()
    stale, stale_future, counted = _overlap(mgr, fit, monkeypatch, pause_newer=True)
    stale.release.set()
    stale_future.result(timeout=_WAIT)

    _assert_newer_run_untouched(mgr, counted, paused=True)
    with pytest.raises(RuntimeError, match="paused"):
        mgr.start_training(X=_xy()[0], y=_xy()[1])

    # The newer run is still held at its next checkpoint, and continues once resumed.
    newer = fit.run(1)
    newer.after_release = (_epoch_end(3),)
    newer.release.set()
    with pytest.raises(TimeoutError):
        mgr._training_future.result(timeout=_HELD)
    mgr.resume_training()
    _settle(mgr)
    assert mgr.get_status()["state_machine"] == _COMPLETED
    assert fit.calls == 2
    assert counted == [manager_module.TRAINING_SESSION_STATUS_SUCCESS], "the newer run must count exactly one terminal session"


@pytest.mark.unit
@pytest.mark.parametrize("reaches_checkpoint", [True, False], ids=["interrupted-at-a-checkpoint", "returned-with-the-stop-signal-up"])
def test_a_stale_fit_that_sees_a_stop_signal_leaves_the_newer_run_alone(lifecycle, monkeypatch, reaches_checkpoint):
    """A stop signal with no FSM transition -- what ``shutdown()`` and a live swap raise -- reaches
    the stale fit too. Its STOP must not land on the newer, running run."""
    mgr, fit = lifecycle()
    stale, stale_future, counted = _overlap(mgr, fit, monkeypatch, pause_newer=False)
    if reaches_checkpoint:
        stale.after_release = (_epoch_end(3),)
    mgr._stop_event.set()
    stale.release.set()
    stale_future.result(timeout=_WAIT)

    _assert_newer_run_untouched(mgr, counted, paused=False)
    mgr.stop_training()
    fit.run(1).release.set()
    _settle(mgr)
    assert mgr.get_status()["state_machine"]["status"] == "STOPPED"
    assert counted == [manager_module.TRAINING_SESSION_STATUS_CANCELLED], "the newer run must count exactly one terminal session"


@pytest.mark.unit
def test_a_stale_fit_that_raises_leaves_the_newer_paused_run_alone(lifecycle, monkeypatch):
    mgr, fit = lifecycle()
    stale, stale_future, counted = _overlap(mgr, fit, monkeypatch, pause_newer=True)
    stale.raises = RuntimeError("stale fit failed")
    stale.release.set()
    with pytest.raises(RuntimeError, match="stale fit failed"):
        stale_future.result(timeout=_WAIT)

    _assert_newer_run_untouched(mgr, counted, paused=True)
    mgr.resume_training()
    fit.run(1).release.set()
    _settle(mgr)
    assert mgr.get_status()["state_machine"] == _COMPLETED
    assert counted == [manager_module.TRAINING_SESSION_STATUS_SUCCESS], "the newer run must count exactly one terminal session"


@pytest.mark.unit
def test_a_stale_fit_that_returns_as_the_newer_run_starts_leaves_it_alone(lifecycle, monkeypatch):
    """Ownership passes as the newer run enters ``_run_training``, before its START, not at its fit.

    The newer run is held just after its ``Command.START``, inside the ``monitor.on_phase_change``
    call ``_run_training`` makes next, and the stale fit returns there. Were the lifecycle claimed
    any later, at the fit for instance, the stale run would still own it and would settle the
    newer run: STOPPED and a cancelled session counted while its fit runs (the stale fit returns
    with its stop still raised, see ``_leave_a_stale_fit_alive``), and its monitor session ended.
    """
    mgr, fit = lifecycle()
    stale, stale_future, counted = _leave_a_stale_fit_alive(mgr, fit, monkeypatch)
    held, hold = threading.Event(), threading.Event()
    on_phase_change = mgr.monitor.on_phase_change

    def _hold_after_start(phase: str) -> None:
        on_phase_change(phase)
        if not held.is_set():  # the newer run's call right after its START only
            held.set()
            hold.wait(timeout=_GATE_TIMEOUT)

    monkeypatch.setattr(mgr.monitor, "on_phase_change", _hold_after_start)
    x, y = _xy()
    try:
        mgr.start_training(X=x, y=y)
        assert held.wait(timeout=_WAIT), "the newer run never reached its START"
        assert mgr.state_machine.get_state_summary()["status"] == "STARTED"
        stale.release.set()
        stale_future.result(timeout=_WAIT)
    finally:
        hold.set()
    _parked(fit, 1)

    _assert_newer_run_untouched(mgr, counted, paused=False)
    mgr.stop_training()
    fit.run(1).release.set()
    _settle(mgr)
    assert mgr.get_status()["state_machine"]["status"] == "STOPPED"
    assert counted == [manager_module.TRAINING_SESSION_STATUS_CANCELLED], "the newer run must count exactly one terminal session"
