"""Regression tests for BUG-CC-#5: control-event normalisation in reset().

Before the fix, ``reset()`` set ``_stop_event`` but did not re-set
``_pause_event``.  If the user paused before stopping, the next
``start_training()`` call would inherit a stale ``_pause_event.clear()``
and the training loop would synthetically pause after a single iteration.

The fix introduces ``TrainingLifecycleManager._reset_event_state()`` as the
single source of truth for control-event normalisation, called from
``reset()``.  These tests pin the contract.
"""

import itertools

import pytest

from api.lifecycle.manager import TrainingLifecycleManager
from api.lifecycle.state_machine import Command


@pytest.fixture
def mgr():
    """Fresh lifecycle manager (no network, no executor, no training thread).

    Sufficient for testing the event-state contract because ``reset()`` and
    ``_reset_event_state()`` touch only ``_pause_event``, ``_stop_event``,
    the FSM, the training-state object, and the broadcast hook — all of which
    are live after ``__init__``.
    """
    return TrainingLifecycleManager()


# ---------------------------------------------------------------------------
# Helper-contract: _reset_event_state() normalises from any prior state.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("pause_set,stop_set", list(itertools.product([False, True], repeat=2)))
def test_reset_event_state_normalises_from_any_prior_state(mgr, pause_set, stop_set):
    """For every combination of prior event states, post-condition holds."""
    if pause_set:
        mgr._pause_event.set()
    else:
        mgr._pause_event.clear()
    if stop_set:
        mgr._stop_event.set()
    else:
        mgr._stop_event.clear()

    mgr._reset_event_state()

    assert mgr._pause_event.is_set(), "post: _pause_event must be set"
    assert mgr._stop_event.is_set(), "post: _stop_event must be set"


# ---------------------------------------------------------------------------
# Focused regression: the exact user-reported sequence.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_reset_after_pause_leaves_pause_event_set(mgr):
    """User scenario: pause -> stop -> reset must leave _pause_event set.

    Pre-fix this asserted False (the bug).  Drives the FSM directly to avoid
    needing a real training thread / network for the test to be meaningful.
    """
    # Mirror the start path's event manipulations + advance FSM.
    mgr._stop_event.clear()
    mgr._pause_event.set()
    assert mgr.state_machine.handle_command(Command.START)

    # User pauses — pause_training() clears _pause_event.
    mgr.pause_training()
    assert not mgr._pause_event.is_set(), "guard: pause must clear _pause_event"

    # User stops — stop_training() sets _stop_event but does not touch pause.
    mgr.stop_training()
    assert not mgr._pause_event.is_set(), "guard: stop must not re-set _pause_event"

    # User resets — fix is here.
    mgr.reset()

    assert mgr._pause_event.is_set(), "BUG-CC-#5 regression: reset() must normalise _pause_event so the next " "start_training() does not inherit a stale clear()"
    assert mgr._stop_event.is_set()


# ---------------------------------------------------------------------------
# Broader coverage: every legal sequence ending in reset() leaves events set.
# ---------------------------------------------------------------------------


_COMMANDS = ["pause", "resume", "stop", "reset"]


def _apply(mgr, cmd):
    """Apply a command; swallow RuntimeError raised for FSM-illegal transitions."""
    method = {
        "pause": mgr.pause_training,
        "resume": mgr.resume_training,
        "stop": mgr.stop_training,
        "reset": mgr.reset,
    }[cmd]
    try:
        method()
    except RuntimeError:
        # Invalid transition (e.g. resume when not paused) — fine for invariant testing.
        pass


@pytest.mark.integration
@pytest.mark.parametrize(
    "seq",
    [seq for n in range(1, 4) for seq in itertools.product(_COMMANDS, repeat=n) if seq[-1] == "reset"],
)
def test_after_reset_pause_event_is_set(mgr, seq):
    """No matter what command sequence preceded a final reset(), _pause_event is set.

    Pre-stages the FSM into STARTED so pause/resume are not all rejected at the
    front door; mirrors what start_training() does to the event pair.
    """
    mgr._stop_event.clear()
    mgr._pause_event.set()
    assert mgr.state_machine.handle_command(Command.START)

    for cmd in seq:
        _apply(mgr, cmd)

    assert mgr._pause_event.is_set(), f"_pause_event left cleared after sequence {seq}"
    assert mgr._stop_event.is_set(), f"_stop_event left cleared after sequence {seq}"
