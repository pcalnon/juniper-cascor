#!/usr/bin/env python
"""Regression tests for P2-PRE-1: pause/stop actually interrupt training.

Pre-fix (HEAD 2069930), ``pause_training`` and ``stop_training`` REST endpoints
set ``_pause_event`` / ``_stop_event`` and transitioned the FSM, but the
flags were never observed inside ``cascade_correlation.fit()`` or any inner
training loop — there are zero references to ``Event``/``wait``/``pause``/
``threading`` in ``cascade_correlation.py``. The two callbacks wired into the
training loop (``_output_training_callback``, ``_grow_iteration_callback``)
were pure metric-emission sinks. ``_stop_event.is_set()`` was checked only
*after* ``original_fit()`` returned naturally — by which point fit had already
run to completion. **Result**: pause and stop were observably no-ops at the
training-loop level; training ran to natural completion regardless.

The fix wires ``_check_for_interrupt()`` into both callbacks. It raises
``TrainingInterrupted`` when ``_stop_event`` is set, and blocks on
``_pause_event.wait(timeout=0.5)`` when paused (re-checking ``_stop_event``
every 0.5 s so a Stop-during-Pause is observed promptly). ``monitored_fit``
catches ``TrainingInterrupted`` as a clean cancellation: same FSM/state/gauge
transitions as the post-fit stop-event path, no exception propagated.

These tests pin the contract by invoking the installed callbacks directly via
``network._output_epoch_callback`` and ``network._grow_iteration_callback``
(the attribute-fallback hook points at ``cascade_correlation.py:1668`` and
``cascade_correlation.py:3942``). Driving the callbacks directly avoids the
need to start a real fit() while still exercising the exact closure-captured
signal-checking code path.

See ``ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md`` §3.4 for the audit
record and the full P2-PRE-1 / Phase 2 plan.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from api.lifecycle.manager import TrainingInterrupted, TrainingLifecycleManager
from api.lifecycle.state_machine import Command


@pytest.fixture
def mgr_with_hooks():
    """Manager with a network and monitoring hooks installed.

    ``_install_monitoring_hooks`` builds the ``_check_for_interrupt`` closure
    and assigns ``_output_training_callback`` / ``_grow_iteration_callback``
    to the network as attribute-fallback hooks. After this fixture the
    callbacks can be invoked directly via ``mgr.network._output_epoch_callback``
    and ``mgr.network._grow_iteration_callback`` without starting fit().
    """
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr._install_monitoring_hooks()
    yield mgr
    mgr.shutdown()


# ---------------------------------------------------------------------------
# Output-training callback: stop & pause signal checks
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_output_callback_raises_on_stop_signal(mgr_with_hooks):
    """When _stop_event is set, the output callback raises TrainingInterrupted."""
    mgr = mgr_with_hooks
    mgr._stop_event.set()
    with pytest.raises(TrainingInterrupted, match="stop_requested"):
        mgr.network._output_epoch_callback(epoch=1, epochs=10, loss=0.5)


@pytest.mark.integration
def test_output_callback_returns_normally_when_not_paused_or_stopped(mgr_with_hooks):
    """Default state (pause set, stop clear) — callback returns without raising."""
    mgr = mgr_with_hooks
    # __init__ sets _pause_event and clears _stop_event; double-check.
    assert mgr._pause_event.is_set()
    assert not mgr._stop_event.is_set()
    # Should not raise; just emit metrics + return None.
    result = mgr.network._output_epoch_callback(epoch=1, epochs=10, loss=0.5)
    assert result is None


@pytest.mark.integration
def test_output_callback_blocks_on_pause_then_resumes(mgr_with_hooks):
    """Clear pause → callback blocks; setting pause from another thread → callback returns."""
    mgr = mgr_with_hooks
    mgr._pause_event.clear()  # paused
    callback_done = threading.Event()

    def _run_callback():
        mgr.network._output_epoch_callback(epoch=1, epochs=10, loss=0.5)
        callback_done.set()

    t = threading.Thread(target=_run_callback, daemon=True)
    t.start()

    # Should NOT complete while paused.
    assert not callback_done.wait(timeout=1.0), "callback returned early — pause was not observed"

    # Resume: setting pause_event unblocks the wait loop.
    mgr._pause_event.set()
    assert callback_done.wait(timeout=2.0), "callback did not return within 2s of resume"
    t.join(timeout=1.0)


@pytest.mark.integration
def test_output_callback_raises_on_stop_during_pause(mgr_with_hooks):
    """Paused; setting _stop_event wakes the wait loop and raises TrainingInterrupted.

    Critical: pre-fix this scenario was impossible (callbacks ignored signals).
    Post-fix the wait loop's 0.5s timeout ensures a stop after pause is observed
    promptly — without the timeout the callback would block forever waiting for
    a resume that's never coming.
    """
    mgr = mgr_with_hooks
    mgr._pause_event.clear()  # paused
    raised: list[Exception] = []
    callback_done = threading.Event()

    def _run_callback():
        try:
            mgr.network._output_epoch_callback(epoch=1, epochs=10, loss=0.5)
        except Exception as e:
            raised.append(e)
        finally:
            callback_done.set()

    t = threading.Thread(target=_run_callback, daemon=True)
    t.start()
    time.sleep(0.1)  # let the callback enter the wait loop

    mgr._stop_event.set()
    assert callback_done.wait(timeout=2.0), "callback did not exit within 2s of stop"
    t.join(timeout=1.0)
    assert len(raised) == 1, f"expected exactly one exception, got {raised!r}"
    assert isinstance(raised[0], TrainingInterrupted)
    assert "stop_requested_during_pause" in str(raised[0])


# ---------------------------------------------------------------------------
# Grow-iteration callback: same contract at the cascade-iteration boundary
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_grow_callback_raises_on_stop_signal(mgr_with_hooks):
    """Grow-iteration callback also honours stop. Pause inside multiprocessing
    candidate training is intentionally out of scope (would require
    multiprocessing-aware signal threading); the iteration boundary is the
    natural pause point for the cascade-growth loop."""
    mgr = mgr_with_hooks
    mgr._stop_event.set()
    with pytest.raises(TrainingInterrupted, match="stop_requested"):
        mgr.network._grow_iteration_callback(
            iteration=1,
            max_iterations=10,
            best_correlation=0.42,
            candidates_trained=8,
            candidates_total=8,
            phase_detail="growing",
        )


@pytest.mark.integration
def test_grow_callback_returns_normally_when_not_stopped(mgr_with_hooks):
    mgr = mgr_with_hooks
    assert not mgr._stop_event.is_set()
    # Returns None; updates state. No exception.
    mgr.network._grow_iteration_callback(
        iteration=1,
        max_iterations=10,
        best_correlation=0.42,
        candidates_trained=8,
        candidates_total=8,
        phase_detail="growing",
    )


# ---------------------------------------------------------------------------
# monitored_fit: TrainingInterrupted is a clean cancellation, not a failure
# ---------------------------------------------------------------------------


def _mgr_with_mocked_fit(side_effect):
    """Build a manager whose network.fit is mocked, with monitoring hooks
    re-installed so ``monitored_fit``'s closure captures the mock as
    ``original_fit``.

    Required because:
      1. ``create_network`` already installs hooks (capturing the REAL fit
         as ``original_fit``), so the second ``_install_monitoring_hooks``
         call is a no-op (gated by ``_monitoring_active``).
      2. After hook install, ``self.network.fit`` IS ``monitored_fit``;
         simply replacing it with a mock detaches monitored_fit entirely.

    Solution: restore the real fit, replace it with the mock, then force a
    re-install by clearing ``_monitoring_active``. The re-installed
    monitored_fit closure captures the mock.
    """
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    # Restore real fit, mock it, then force re-install so closure rebinds.
    mgr.network.fit = mgr._original_methods["fit"]
    mock_fit = MagicMock(side_effect=side_effect)
    mgr.network.fit = mock_fit
    mgr._monitoring_active = False
    mgr._install_monitoring_hooks()
    # Drive FSM to Started so Command.STOP is a valid transition.
    mgr._stop_event.clear()
    assert mgr.state_machine.handle_command(Command.START)
    return mgr, mock_fit


@pytest.mark.integration
def test_monitored_fit_treats_interrupt_as_clean_cancellation():
    """When original_fit raises TrainingInterrupted, monitored_fit must:
    - transition FSM via Command.STOP (not mark_failed)
    - update training_state to status=Stopped, phase=Idle
    - NOT re-raise the exception (user requested stop; not an API error)
    - return None
    """
    mgr, mock_fit = _mgr_with_mocked_fit(TrainingInterrupted("stop_requested"))
    try:
        # Must not raise — clean cancellation is swallowed by monitored_fit.
        result = mgr.network.fit(
            x=MagicMock(),
            y=MagicMock(),
            x_val=None,
            y_val=None,
        )
        assert result is None, "monitored_fit returns None on clean cancellation"
        assert mock_fit.call_count == 1, "original_fit was invoked once"

        state = mgr.training_state.get_state()
        assert state["status"] == "Stopped", f"expected Stopped, got {state['status']!r}"
        assert state["phase"] == "Idle", f"expected Idle, got {state['phase']!r}"
    finally:
        mgr.shutdown()


@pytest.mark.integration
def test_monitored_fit_treats_other_exceptions_as_failure():
    """Sanity: NON-TrainingInterrupted exceptions still flow through the
    failure path (mark_failed, status=Failed, exception re-raised). This
    pins the boundary between the two except-clauses so a future refactor
    doesn't accidentally widen the cancellation catch."""
    mgr, _ = _mgr_with_mocked_fit(RuntimeError("boom"))
    try:
        with pytest.raises(RuntimeError, match="boom"):
            mgr.network.fit(x=MagicMock(), y=MagicMock(), x_val=None, y_val=None)
        state = mgr.training_state.get_state()
        assert state["status"] == "Failed"
    finally:
        mgr.shutdown()
