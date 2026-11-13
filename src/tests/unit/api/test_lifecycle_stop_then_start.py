#!/usr/bin/env python
"""Pin that a start sent after a stop leaves the stopped fit stopped (the "stop-then-start race").

A stop is cooperative. ``stop_training`` raises the run's stop signal and moves the FSM to STOPPED
at once, but the fit sees the signal only at its next interrupt checkpoint (``_check_for_interrupt``,
which ``_handle_event`` runs on ``epoch_end`` and ``phase_change``), or after it returns.
``start_training`` accepts a start from STOPPED, so a start can arrive while the stopped fit is
still running; its run then queues behind that fit on the single-worker executor.

Before this fix every run shared ONE stop event, and ``start_training`` cleared it. A start in that
window therefore cancelled the stop: the old fit passed its next checkpoint and trained on to its
natural end while the FSM read STOPPED, then, still owning the lifecycle (the new run had not
entered ``_run_training``), wrote ``training_state`` Completed beside the STOPPED FSM and counted a
successful session, for a run the operator had stopped. A paused fit was released the same way, and
``swap_dataset_live`` un-stopped the old fit after its join timed out. canopy's Restart (stop, wait
until ``training_active`` is false -- the FSM, so at once -- then start) takes exactly this path.

Now a start never clears a raised stop signal: it gives its own run a fresh one, and each run
watches the signal it was accepted with (``_stop_event_for_new_run_locked``). Every run accepted
since the last stop shares the newest signal, so a stop still reaches a running fit and every run
queued behind it, and never a run accepted after it. The tests drive the real manager, executor,
FSM and (for one) HTTP routes, with a gated stub ``model.fit``: events and gates, no sleeps.

juniper-ml's flood-4 record, ``notes/JUNIPER_2026-11-06_JUNIPER-ECOSYSTEM_CURSOR-FLOOD-4-DISPOSITION.md``,
lists the paused-fit wedge under §8 ruling 11; the PR that fixed it (juniper-cascor#753) reported
this race, and the owner's ruling 17 (2026-11-09; the record's next update adds it to its §8),
"cascor: stop-then-start", gave it a fix of its own, in its own PR. That ruling named no design;
the one above is this change's, and the owner ruled on 2026-11-12 to keep it.
"""

import threading
from concurrent.futures import Future
from typing import Callable, List, Tuple

import pytest
import torch
from fastapi.testclient import TestClient
from juniper_model_core.events import TrainingEvent

from api.app import create_app
from api.lifecycle import manager as manager_module
from api.lifecycle.state_machine import Command
from api.models.training import TrainingParams
from api.settings import Settings

# Bounded, so a failed assertion can never leave a fit thread parked for good.
_GATE_TIMEOUT = 10.0
_WAIT = 5.0
# The session counter's three terminal statuses, as ``_run_training`` reports them.
_CANCELLED = manager_module.TRAINING_SESSION_STATUS_CANCELLED
_SUCCESS = manager_module.TRAINING_SESSION_STATUS_SUCCESS


def _training_start(seq: int = 0) -> TrainingEvent:
    return TrainingEvent("training_start", {"n_samples": 8}, seq)


def _epoch_end(seq: int = 1) -> TrainingEvent:
    """An output-pass ``epoch_end``: ``_handle_event`` runs an interrupt checkpoint on it."""
    return TrainingEvent("epoch_end", {"epoch": 25, "epochs": 50, "metrics": {"loss": 0.5}}, seq)


def _phase_change(seq: int = 1) -> TrainingEvent:
    """A grow-iteration ``phase_change``: the other event ``_handle_event`` interrupt-checks."""
    return TrainingEvent("phase_change", {"detail": {"grow_iteration": 1, "max_iterations": 2, "best_correlation": 0.1, "candidates_trained": 1}}, seq)


def _training_end(seq: int = 3) -> TrainingEvent:
    return TrainingEvent("training_end", {"metrics": {}}, seq)


_CHECKPOINTS = {"epoch_end": _epoch_end, "phase_change": _phase_change}


class _Run:
    """The gates of one call of the stub fit, which is one training run."""

    def __init__(self) -> None:
        self.before_park: Tuple[TrainingEvent, ...] = (_training_start(0),)
        self.after_release: Tuple[TrainingEvent, ...] = (_epoch_end(1),)  # its next checkpoint
        self.parked = threading.Event()  # set once the fit has sent ``before_park`` and parked
        self.release = threading.Event()  # the test sets it to let the fit go on
        self.ran_on = threading.Event()  # set once the fit got past ``after_release``: no checkpoint stopped it


class _GatedFit:
    """Stands in for ``CascorModel.fit``, installed as ``mgr.model.fit`` (the suite's house pattern).

    Each call is one run: it sends ``before_park`` through the manager's real ``on_event`` sink,
    parks on its gate, then -- once released -- sends ``after_release`` (by default one
    checkpointed ``epoch_end``), marks ``ran_on`` and returns after ``training_end``. A checkpoint
    that raises ``TrainingInterrupted`` ends the call before ``ran_on``.
    """

    def __init__(self) -> None:
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
        for event in run.before_park:
            on_event(event)
        run.parked.set()
        run.release.wait(timeout=_GATE_TIMEOUT)
        for event in run.after_release:
            on_event(event)
        run.ran_on.set()
        on_event(_training_end())

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


def _start(mgr) -> Future:
    """Start a run on fresh inline data and return its training future."""
    x, y = _xy()
    mgr.start_training(X=x, y=y)
    return mgr._training_future


def _answer(request: Callable[[], object]) -> str:
    """Make a lifecycle request: "ok", or "refused: <reason>" for what a route maps to 409."""
    try:
        request()
        return "ok"
    except (RuntimeError, ValueError) as exc:
        return f"refused: {exc}"


@pytest.fixture
def lifecycle(monkeypatch):
    """A manager whose fit is gated, and the list its terminal session counter records into."""
    built: List[Tuple[object, _GatedFit]] = []
    counted: List[str] = []
    monkeypatch.setattr(manager_module, "inc_training_session_completed", counted.append)

    def _make():
        mgr = manager_module.TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        fit = _GatedFit()
        mgr.model.fit = fit
        built.append((mgr, fit))
        return mgr, fit, counted

    yield _make
    for mgr, fit in built:
        fit.release_all()
        mgr.shutdown()


def _record_mark_completed(mgr, monkeypatch) -> List[str]:
    """Record the FSM status each ``mark_completed`` call found. A stopped run must make none."""
    sm = mgr.state_machine
    mark_completed = sm.mark_completed
    found: List[str] = []

    def _recorded() -> bool:
        found.append(sm.status.name)
        return mark_completed()

    monkeypatch.setattr(sm, "mark_completed", _recorded)
    return found


def _finish_newer_run(mgr, fit: _GatedFit, index: int, future: Future) -> None:
    """Release a queued run once it parks, wait for it, and check it was neither stopped nor failed."""
    _parked(fit, index).release.set()
    assert future.result(timeout=_WAIT) is None
    assert fit.run(index).ran_on.is_set(), f"run {index} was stopped by a stop sent before it was started"
    assert mgr.state_machine.get_state_summary()["status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# The race: a start sent after a stop, before the stopped fit's next checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("stopping, checkpoint", [("stop_training", "epoch_end"), ("stop_training", "phase_change"), ("reset", "epoch_end")])
def test_a_start_after_a_stop_leaves_the_stopped_fit_stopped(lifecycle, monkeypatch, stopping, checkpoint):
    """The race itself: the stopped fit must stop at its next checkpoint, and the new run must run.

    ``reset`` raises the same signal and moves the FSM to STOPPED the same way, so a start after a
    reset raced it too. Both checkpointed event types are covered.
    """
    mgr, fit, counted = lifecycle()
    fit.run(0).after_release = (_CHECKPOINTS[checkpoint](1),)
    old_future = _start(mgr)
    old = _parked(fit, 0)  # between checkpoints
    getattr(mgr, stopping)()
    assert mgr.state_machine.get_state_summary()["status"] == "STOPPED"
    found = _record_mark_completed(mgr, monkeypatch)

    assert _answer(lambda: _start(mgr)) == "ok", "a start after a stop is accepted, as before"
    new_future = mgr._training_future
    assert not new_future.done(), "the new run must wait behind the stopped fit"

    old.release.set()
    assert old_future.result(timeout=_WAIT) is None  # TrainingInterrupted is a clean cancellation
    # Before the fix: True -- the start cleared the stop signal, so the checkpoint passed.
    assert not old.ran_on.is_set(), "the stopped fit passed its next checkpoint: the start cancelled the stop"
    # Before the fix: ["STOPPED"] (refused) beside training_state "Completed", and "success" counted.
    assert found == [], "the stopped run tried to settle COMPLETED"
    assert counted == [_CANCELLED], "the stopped run must count as cancelled, and only it so far"

    _finish_newer_run(mgr, fit, 1, new_future)
    assert counted == [_CANCELLED, _SUCCESS]
    assert fit.calls == 2


@pytest.mark.unit
def test_a_start_after_a_stop_past_the_last_checkpoint_settles_the_stopped_run_stopped(lifecycle, monkeypatch):
    """A fit that has passed its last checkpoint returns anyway; its run must still settle as stopped."""
    mgr, fit, counted = lifecycle()
    fit.run(0).after_release = ()  # no checkpoint left
    old_future = _start(mgr)
    old = _parked(fit, 0)
    mgr.stop_training()
    found = _record_mark_completed(mgr, monkeypatch)
    states: List[str] = []
    update_state = mgr.training_state.update_state

    def _recorded_update(**fields):
        if "status" in fields:
            states.append(fields["status"])
        return update_state(**fields)

    monkeypatch.setattr(mgr.training_state, "update_state", _recorded_update)
    new_future = _start(mgr)

    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    # Before the fix: mark_completed refused from STOPPED, then training_state "Completed" and "success".
    assert found == []
    assert states[:1] == ["Stopped"], f"the stopped run's settle wrote {states[:1]}"
    assert counted == [_CANCELLED]

    _finish_newer_run(mgr, fit, 1, new_future)
    assert counted == [_CANCELLED, _SUCCESS]


@pytest.mark.unit
def test_a_stopped_fit_held_by_a_pause_stays_stopped_when_a_start_reopens_the_gate(lifecycle, monkeypatch):
    """A fit paused at a checkpoint, then stopped: the start reopens the pause gate, and must not release it.

    The checkpoint's pause loop wakes when the gate opens and re-reads the stop signal. While the
    start cleared that signal, the loop let the fit go on, and it trained to its end. The loop's
    first wait is held here until the stop and the start have both landed, so its 0.5 s poll
    cannot observe the stop first.
    """
    mgr, fit, counted = lifecycle()
    old_future = _start(mgr)
    old = _parked(fit, 0)
    pause_gate = mgr._pause_event
    real_wait = pause_gate.wait
    waiting, started = threading.Event(), threading.Event()

    def _wait(timeout=None):
        waiting.set()
        started.wait(timeout=_GATE_TIMEOUT)
        return real_wait(timeout)

    monkeypatch.setattr(pause_gate, "wait", _wait)
    mgr.pause_training()
    old.release.set()  # the fit meets its next checkpoint while paused, and waits there
    try:
        assert waiting.wait(timeout=_WAIT), "the fit never waited at its checkpoint"
        mgr.stop_training()
        new_future = _start(mgr)  # reopens the pause gate
        assert pause_gate.is_set()
    finally:
        started.set()

    assert old_future.result(timeout=_WAIT) is None
    assert not old.ran_on.is_set(), "the start released the stopped fit from its pause"
    assert counted == [_CANCELLED]

    _finish_newer_run(mgr, fit, 1, new_future)
    assert counted == [_CANCELLED, _SUCCESS]


@pytest.mark.unit
def test_a_start_refused_after_a_stop_leaves_the_stopped_fit_stopped(lifecycle, monkeypatch):
    """A start the lifecycle refuses queues no run, and must not cancel the stop it follows either.

    ``start_training`` checks the request's params after it has bound the data, and refuses a
    candidate-pool triple that cannot hold (the route answers 409). The start used to clear the
    shared stop signal BEFORE that check, so even a refused start cancelled the stop: the stopped
    fit trained on to its end, with no new run behind it. A run's signal is now chosen only as the
    run is submitted, after the last check that can refuse, and the stopped fit keeps its own.

    The params are one candidate more than the pool holds, built through the route's own request
    model: it admits them, so over HTTP this start reaches the lifecycle and is refused there. (A
    count the model itself refuses, 0 for one, never reaches the lifecycle: the route's 422.)
    """
    mgr, fit, counted = lifecycle()
    old_future = _start(mgr)
    old = _parked(fit, 0)
    mgr.stop_training()
    found = _record_mark_completed(mgr, monkeypatch)

    x, y = _xy()
    params = TrainingParams(selected_candidates=mgr.network.candidate_pool_size + 1).model_dump(exclude_none=True)
    answer = _answer(lambda: mgr.start_training(X=x, y=y, **params))
    assert answer.startswith("refused: selected_candidates") and "not in [1, candidate_pool_size=" in answer, answer
    assert mgr._training_future is old_future, "a refused start queued a run"

    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    # Before the fix: True -- the refused start had already cleared the stop signal.
    assert not old.ran_on.is_set(), "a refused start cancelled the stop"
    assert found == [], "the stopped run tried to settle COMPLETED"
    assert counted == [_CANCELLED]
    assert fit.calls == 1

    _finish_newer_run(mgr, fit, 1, _start(mgr))
    assert counted == [_CANCELLED, _SUCCESS]


# ---------------------------------------------------------------------------
# Which runs a stop reaches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_stop_reaches_every_run_queued_behind_a_stopped_fit(lifecycle):
    """Two starts queue behind a stopped fit, then a second stop: all three runs must stop.

    The queued runs share the newest signal, so the second stop reaches both. Were each start to
    get a fresh signal of its own, that stop would raise only the second queued run's, and the
    first would train. Each queued run here meets a checkpoint as soon as its fit begins.
    """
    mgr, fit, counted = lifecycle()
    for index in (1, 2):
        fit.run(index).before_park = (_training_start(0), _epoch_end(1))
    old_future = _start(mgr)
    old = _parked(fit, 0)
    mgr.stop_training()
    first_queued = _start(mgr)
    second_queued = _start(mgr)
    mgr.stop_training()

    old.release.set()
    for future in (old_future, first_queued, second_queued):
        assert future.result(timeout=_WAIT) is None
    assert not any(fit.run(index).ran_on.is_set() for index in (0, 1, 2))
    assert not fit.run(1).parked.is_set(), "the first queued run passed a checkpoint: the second stop missed it"
    assert not fit.run(2).parked.is_set()
    assert counted == [_CANCELLED, _CANCELLED, _CANCELLED]
    assert mgr.state_machine.get_state_summary()["status"] == "STOPPED"


@pytest.mark.unit
def test_a_queued_run_stopped_before_it_began_stays_stopped_when_another_start_follows(lifecycle):
    """Stop, start, stop, start: the first queued run was stopped before it began, the second was not.

    The second start finds the queued run's signal raised and gives its own run a fresh one. Each
    run must watch the signal it was accepted with: one that read the newest signal as it entered
    would read the second start's, and the run the operator stopped would train.
    """
    mgr, fit, counted = lifecycle()
    fit.run(1).before_park = (_training_start(0), _epoch_end(1))
    old_future = _start(mgr)
    old = _parked(fit, 0)
    mgr.stop_training()
    stopped_queued = _start(mgr)
    mgr.stop_training()
    newest = _start(mgr)

    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    assert stopped_queued.result(timeout=_WAIT) is None
    assert not old.ran_on.is_set()
    assert not fit.run(1).parked.is_set(), "a run stopped while it was queued trained once it began"
    assert counted == [_CANCELLED, _CANCELLED]
    _finish_newer_run(mgr, fit, 2, newest)
    assert counted == [_CANCELLED, _CANCELLED, _SUCCESS]


@pytest.mark.unit
def test_a_start_that_lands_inside_a_stop_is_not_stopped_by_it(lifecycle, monkeypatch):
    """Force a start in between the stop's FSM transition and its return: the new run must run.

    The FSM accepts a start once it reads STOPPED. ``stop_training`` raises the signal BEFORE that
    transition, so the start finds it raised and gives its run a fresh one. Raised after it, the
    start would share the old run's signal, and the stop would then stop the new run too.
    """
    mgr, fit, counted = lifecycle()
    old_future = _start(mgr)
    old = _parked(fit, 0)
    sm = mgr.state_machine
    handle_command = sm.handle_command
    answers: List[str] = []

    def _start_lands_inside_the_stop(command):
        accepted = handle_command(command)
        if command is Command.STOP and not answers:
            answers.append(_answer(lambda: _start(mgr)))
        return accepted

    monkeypatch.setattr(sm, "handle_command", _start_lands_inside_the_stop)
    mgr.stop_training()
    assert answers == ["ok"]
    new_future = mgr._training_future

    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    assert not old.ran_on.is_set()
    _finish_newer_run(mgr, fit, 1, new_future)
    assert counted == [_CANCELLED, _SUCCESS]


def _swap_whose_join_times_out(mgr, fit: _GatedFit, monkeypatch) -> Tuple[_Run, Future, Future]:
    """Start run 0, then live-swap while its fit is between checkpoints, forcing the swap's join to time out.

    ``swap_dataset_live`` raises the stop signal and joins the old fit for at most 10 s; on a
    timeout it carries on and submits the swapped run, which queues behind the old fit. The
    timeout is forced by the old future's ``result``. Returns run 0's gate, its future and the
    swapped run's future.
    """
    mgr.set_experimental_functions(True)
    old_future = _start(mgr)
    old = _parked(fit, 0)
    real_result = old_future.result
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
    assert mgr.swap_dataset_live(dataset_type="spiral")["status"] == "swapped"
    assert joins == [10], "the swap never joined the old fit"
    swapped = mgr._training_future
    assert swapped is not old_future and not swapped.done()
    return old, old_future, swapped


@pytest.mark.unit
def test_a_live_swap_whose_join_timed_out_leaves_the_old_fit_stopped(lifecycle, monkeypatch):
    """The swap used to clear the stop signal before it submitted, so the old fit, never interrupted, trained on to its end."""
    mgr, fit, counted = lifecycle()
    old, old_future, swapped = _swap_whose_join_times_out(mgr, fit, monkeypatch)

    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    # Before the fix: True, then COMPLETED and "success" for the run the swap stopped.
    assert not old.ran_on.is_set(), "the swap released the old fit it had stopped"
    assert counted == [_CANCELLED]
    _finish_newer_run(mgr, fit, 1, swapped)
    assert counted == [_CANCELLED, _SUCCESS]


@pytest.mark.unit
def test_a_swapped_run_stopped_while_queued_stays_stopped_when_a_start_follows(lifecycle, monkeypatch):
    """The swapped run, too, watches the signal it was submitted with.

    A stop while it is queued raises that signal, and the start that follows gives its own run a
    fresh one. A swapped run that read the newest signal as it entered would train.
    """
    mgr, fit, counted = lifecycle()
    fit.run(1).before_park = (_training_start(0), _epoch_end(1))
    old, old_future, swapped = _swap_whose_join_times_out(mgr, fit, monkeypatch)
    mgr.stop_training()
    newest = _start(mgr)

    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    assert swapped.result(timeout=_WAIT) is None
    assert not old.ran_on.is_set()
    assert not fit.run(1).parked.is_set(), "the swapped run, stopped while queued, trained once it began"
    assert counted == [_CANCELLED, _CANCELLED]
    _finish_newer_run(mgr, fit, 2, newest)
    assert counted == [_CANCELLED, _CANCELLED, _SUCCESS]


# ---------------------------------------------------------------------------
# Ownership: the completion guard still holds a stale run that is past its stop check
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_stale_run_past_its_stop_check_leaves_the_newer_paused_run_alone(lifecycle, monkeypatch):
    """A stale run in the shutdown overlap that has already passed its stop check settles nothing.

    ``shutdown()`` raises the stop signal before its bounded join, and a start no longer clears it,
    so in the overlap juniper-cascor#753's ownership tests build, a stale fit now takes the stop
    path. A stale run takes the completion path only if it was past its stop check -- in its final
    evaluation here -- when the stop came. The completion's ownership guard must still hold it: no
    final score, no COMPLETED, no pause gate reopened, no session counted for the newer run.
    """
    mgr, fit, counted = lifecycle()
    fit.run(0).after_release = ()
    real_final = mgr._compute_final_eval_metrics
    in_final, final_release = threading.Event(), threading.Event()

    def _final_eval_parks_once():
        if not in_final.is_set():
            in_final.set()
            final_release.wait(timeout=_GATE_TIMEOUT)
        return real_final()

    monkeypatch.setattr(mgr, "_compute_final_eval_metrics", _final_eval_parks_once)
    x, y = _xy()
    x_test, y_test = _xy()
    mgr.start_training(X=x, y=y, X_test=x_test, y_test=y_test)  # a test split, so a final score is computable
    stale_future = mgr._training_future
    _parked(fit, 0).release.set()
    try:
        assert in_final.wait(timeout=_WAIT), "the stale run never reached its final evaluation"
        mgr.stop_training()
        monkeypatch.setattr(manager_module, "_SHUTDOWN_TRAINING_JOIN_TIMEOUT_SECONDS", 0.05)
        mgr.shutdown()
        assert not stale_future.done(), "the stale run finished during shutdown -- the overlap was not built"
        newer_future = _start(mgr)
        _parked(fit, 1)
        mgr.pause_training()
    finally:
        final_release.set()
    assert stale_future.result(timeout=_WAIT) is None

    status = mgr.get_status()
    assert status["state_machine"]["status"] == "PAUSED"
    assert status["training_state"]["status"] == "Paused"
    assert mgr.get_metrics()["eval_metrics"]["final"] is None, "the stale run kept a final score"
    assert mgr.monitor.get_current_state()["is_training"] is True, "the stale run ended the newer run's monitor session"
    assert not mgr._pause_event.is_set(), "the stale run reopened the newer run's pause gate"
    assert counted == [], "the stale run counted a terminal session"

    mgr.resume_training()
    _finish_newer_run(mgr, fit, 1, newer_future)
    assert counted == [_SUCCESS]


# ---------------------------------------------------------------------------
# Over HTTP: canopy's Restart sequence
# ---------------------------------------------------------------------------


@pytest.fixture
def http_stack(monkeypatch):
    """The real app and lifecycle over HTTP, with the network's fit gated."""
    counted: List[str] = []
    monkeypatch.setattr(manager_module, "inc_training_session_completed", counted.append)
    app = create_app(Settings(auto_start=False))
    with TestClient(app) as client:
        assert client.post("/v1/network", json={"input_size": 2, "output_size": 2}).status_code == 200
        lifecycle = client.app.state.lifecycle
        fit = _GatedFit()
        lifecycle.model.fit = fit
        try:
            yield client, lifecycle, fit, counted
        finally:
            fit.release_all()


@pytest.mark.unit
def test_stop_then_start_over_http_answers_200_and_the_stopped_fit_stays_stopped(http_stack):
    """canopy's Restart: stop, wait until ``training_active`` is false, start. The answers are unchanged."""
    client, lifecycle, fit, counted = http_stack
    x, y = _xy()
    payload = {"inline_data": {"train_x": x.tolist(), "train_y": y.tolist()}}

    assert client.post("/v1/training/start", json=payload).status_code == 200
    old_future = lifecycle._training_future
    old = _parked(fit, 0)
    stop = client.post("/v1/training/stop")
    assert stop.status_code == 200, stop.text
    status = client.get("/v1/training/status").json()["data"]
    assert status["training_active"] is False, "canopy's Restart waits on this before it starts"
    assert status["state_machine"]["status"] == "STOPPED"
    start = client.post("/v1/training/start", json=payload)
    assert start.status_code == 200, start.text
    new_future = lifecycle._training_future

    old.release.set()
    assert old_future.result(timeout=_WAIT) is None
    assert not old.ran_on.is_set(), "the start cancelled the stop"
    assert counted == [_CANCELLED]
    _finish_newer_run(lifecycle, fit, 1, new_future)
    status = client.get("/v1/training/status").json()["data"]
    assert status["state_machine"]["status"] == "COMPLETED"
    assert status["training_state"]["status"] == "Completed"
    assert counted == [_CANCELLED, _SUCCESS]
