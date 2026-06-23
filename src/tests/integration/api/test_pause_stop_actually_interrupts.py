#!/usr/bin/env python
"""Regression tests for P2-PRE-1: pause/stop actually interrupt training.

Pre-fix (HEAD 2069930), ``pause_training`` and ``stop_training`` REST endpoints
set ``_pause_event`` / ``_stop_event`` and transitioned the FSM, but the
flags were never observed inside ``cascade_correlation.fit()`` or any inner
training loop. ``_stop_event.is_set()`` was checked only *after* fit returned
naturally — by which point fit had already run to completion. **Result**: pause
and stop were observably no-ops at the training-loop level.

The fix wires ``_check_for_interrupt()`` into the training-loop event path. It
raises ``TrainingInterrupted`` when ``_stop_event`` is set, and blocks on
``_pause_event.wait(timeout=0.5)`` when paused (re-checking ``_stop_event`` every
0.5 s so a Stop-during-Pause is observed promptly).

WS-6 PR-B3.3 moved monitoring off the network monkey-patch onto ``CascorModel.fit``'s
``on_event`` sink: the live output-epoch and grow-iteration callbacks now emit
``epoch_end`` / ``phase_change`` events that the manager's ``_handle_event`` projects —
and ``_handle_event`` is where ``_check_for_interrupt()`` now runs. Because ``on_event``
is dispatched synchronously from CCN's bare callback sites, a ``TrainingInterrupted``
raised in ``_handle_event`` propagates straight out of ``fit``; ``_run_training`` catches
it as a clean cancellation (same FSM/state/gauge transitions as a post-fit stop). These
tests pin that contract by driving ``_handle_event`` directly (exercising the exact
signal-checking path without starting a real fit) and by driving ``_run_training`` with a
stubbed ``model.fit``.

See ``ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md`` §3.4 for the audit record.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest
from juniper_model_core.events import TrainingEvent

from api.lifecycle.manager import TrainingInterrupted, TrainingLifecycleManager
from api.lifecycle.state_machine import Command


def _epoch_event(loss=0.5):
    """An output-training ``epoch_end`` event (the boundary _handle_event interrupt-checks)."""
    return TrainingEvent("epoch_end", {"epoch": 1, "epochs": 10, "metrics": {"loss": loss}}, 0)


def _grow_event():
    """A cascade grow-iteration ``phase_change`` event (the other interrupt boundary)."""
    return TrainingEvent(
        "phase_change",
        {
            "phase": "candidate",
            "detail": {
                "grow_iteration": 1,
                "max_iterations": 10,
                "best_correlation": 0.42,
                "candidates_trained": 8,
                "candidates_total": 8,
                "phase_detail": "growing",
            },
        },
        0,
    )


@pytest.fixture
def manager():
    """A manager with a network. WS-6 PR-B3.3: no monitoring hooks are installed at create
    time — ``_handle_event`` is the live event sink and can be driven directly."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    yield mgr
    mgr.shutdown()


# ---------------------------------------------------------------------------
# Output-training (epoch_end) event: stop & pause signal checks
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_epoch_event_raises_on_stop_signal(manager):
    """When _stop_event is set, handling an epoch_end event raises TrainingInterrupted."""
    manager._stop_event.set()
    with pytest.raises(TrainingInterrupted, match="stop_requested"):
        manager._handle_event(_epoch_event())


@pytest.mark.integration
def test_epoch_event_returns_normally_when_not_paused_or_stopped(manager):
    """Default state (pause set, stop clear) — handling the event returns without raising."""
    # __init__ sets _pause_event and clears _stop_event; double-check.
    assert manager._pause_event.is_set()
    assert not manager._stop_event.is_set()
    # Should not raise; just project metrics/state + return None.
    result = manager._handle_event(_epoch_event())
    assert result is None


@pytest.mark.integration
def test_epoch_event_blocks_on_pause_then_resumes(manager):
    """Clear pause → _handle_event blocks; setting pause from another thread → it returns."""
    manager._pause_event.clear()  # paused
    callback_done = threading.Event()

    def _run():
        manager._handle_event(_epoch_event())
        callback_done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Should NOT complete while paused.
    assert not callback_done.wait(timeout=1.0), "event handled early — pause was not observed"

    # Resume: setting pause_event unblocks the wait loop.
    manager._pause_event.set()
    assert callback_done.wait(timeout=2.0), "event did not complete within 2s of resume"
    t.join(timeout=1.0)


@pytest.mark.integration
def test_epoch_event_raises_on_stop_during_pause(manager):
    """Paused; setting _stop_event wakes the wait loop and raises TrainingInterrupted.

    The wait loop's 0.5s timeout ensures a stop after pause is observed promptly —
    without it the handler would block forever waiting for a resume that never comes.
    """
    manager._pause_event.clear()  # paused
    raised: list[Exception] = []
    callback_done = threading.Event()

    def _run():
        try:
            manager._handle_event(_epoch_event())
        except Exception as e:
            raised.append(e)
        finally:
            callback_done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(0.1)  # let the handler enter the wait loop

    manager._stop_event.set()
    assert callback_done.wait(timeout=2.0), "handler did not exit within 2s of stop"
    t.join(timeout=1.0)
    assert len(raised) == 1, f"expected exactly one exception, got {raised!r}"
    assert isinstance(raised[0], TrainingInterrupted)
    assert "stop_requested_during_pause" in str(raised[0])


# ---------------------------------------------------------------------------
# Grow-iteration (phase_change) event: same contract at the cascade boundary
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_grow_event_raises_on_stop_signal(manager):
    """A grow-iteration event also honours stop. Pause inside multiprocessing candidate
    training is intentionally out of scope; the iteration boundary is the natural pause
    point for the cascade-growth loop."""
    manager._stop_event.set()
    with pytest.raises(TrainingInterrupted, match="stop_requested"):
        manager._handle_event(_grow_event())


@pytest.mark.integration
def test_grow_event_returns_normally_when_not_stopped(manager):
    """Not stopped — the grow event projects grow state and returns without raising."""
    assert not manager._stop_event.is_set()
    manager.state_machine.handle_command(Command.START)
    result = manager._handle_event(_grow_event())
    assert result is None


# ---------------------------------------------------------------------------
# _run_training: TrainingInterrupted is a clean cancellation, not a failure
# ---------------------------------------------------------------------------


def _mgr_with_mocked_model(side_effect):
    """A manager whose ``model.fit`` is mocked to raise ``side_effect``. WS-6 PR-B3.3:
    ``_run_training`` drives ``self.model.fit`` and owns the terminal FSM, so the
    cancellation/failure contract is exercised by stubbing ``model.fit``."""
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mock_fit = MagicMock(side_effect=side_effect)
    mgr.model.fit = mock_fit
    mgr._stop_event.clear()
    return mgr, mock_fit


@pytest.mark.integration
def test_run_training_treats_interrupt_as_clean_cancellation():
    """When model.fit raises TrainingInterrupted, _run_training must:
    - transition the FSM via Command.STOP (not mark_failed)
    - update training_state to status=Stopped, phase=Idle
    - NOT re-raise (the user requested stop; not an API error)
    """
    mgr, mock_fit = _mgr_with_mocked_model(TrainingInterrupted("stop_requested"))
    try:
        # Must not raise — clean cancellation is swallowed by _run_training.
        mgr._run_training(MagicMock(), MagicMock(), None, None)
        assert mock_fit.call_count == 1, "model.fit was invoked once"

        state = mgr.training_state.get_state()
        assert state["status"] == "Stopped", f"expected Stopped, got {state['status']!r}"
        assert state["phase"] == "Idle", f"expected Idle, got {state['phase']!r}"
    finally:
        mgr.shutdown()


@pytest.mark.integration
def test_run_training_treats_other_exceptions_as_failure():
    """Sanity: NON-TrainingInterrupted exceptions still flow through the failure path
    (mark_failed, status=Failed, exception re-raised so the future surfaces it). This pins
    the boundary between the two except-clauses so a future refactor doesn't accidentally
    widen the cancellation catch."""
    mgr, _ = _mgr_with_mocked_model(RuntimeError("boom"))
    try:
        with pytest.raises(RuntimeError, match="boom"):
            mgr._run_training(MagicMock(), MagicMock(), None, None)
        state = mgr.training_state.get_state()
        assert state["status"] == "Failed"
    finally:
        mgr.shutdown()
