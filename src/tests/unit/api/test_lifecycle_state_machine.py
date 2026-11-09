"""Tests for training state machine."""

import threading

import pytest

from api.lifecycle.state_machine import Command, TrainingPhase, TrainingStateMachine, TrainingStatus


@pytest.mark.unit
class TestTrainingStateMachine:
    """Test state machine transitions."""

    def test_initial_state(self):
        """State machine starts in Stopped/Idle."""
        sm = TrainingStateMachine()
        assert sm.status == TrainingStatus.STOPPED
        assert sm.phase == TrainingPhase.IDLE
        assert sm.is_stopped()
        assert not sm.is_started()
        assert not sm.is_paused()

    def test_start_from_stopped(self):
        """Start transitions Stopped -> Started."""
        sm = TrainingStateMachine()
        result = sm.handle_command(Command.START)
        assert result is True
        assert sm.is_started()
        assert sm.phase == TrainingPhase.OUTPUT

    def test_stop_from_started(self):
        """Stop transitions Started -> Stopped."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        result = sm.handle_command(Command.STOP)
        assert result is True
        assert sm.is_stopped()
        assert sm.phase == TrainingPhase.IDLE

    def test_pause_from_started(self):
        """Pause transitions Started -> Paused."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        result = sm.handle_command(Command.PAUSE)
        assert result is True
        assert sm.is_paused()
        assert sm.paused_phase == TrainingPhase.OUTPUT

    def test_resume_from_paused(self):
        """Resume transitions Paused -> Started, restores phase."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.set_phase(TrainingPhase.CANDIDATE)
        sm.handle_command(Command.PAUSE)
        result = sm.handle_command(Command.RESUME)
        assert result is True
        assert sm.is_started()
        assert sm.phase == TrainingPhase.CANDIDATE

    def test_reset_from_any_state(self):
        """Reset always goes to Stopped."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        result = sm.handle_command(Command.RESET)
        assert result is True
        assert sm.is_stopped()

    def test_invalid_start_while_started(self):
        """Cannot start when already started."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        result = sm.handle_command(Command.START)
        assert result is False
        assert sm.is_started()

    def test_invalid_pause_while_stopped(self):
        """Cannot pause when stopped."""
        sm = TrainingStateMachine()
        result = sm.handle_command(Command.PAUSE)
        assert result is False

    def test_invalid_resume_while_stopped(self):
        """Cannot resume when stopped."""
        sm = TrainingStateMachine()
        result = sm.handle_command(Command.RESUME)
        assert result is False

    def test_invalid_stop_while_stopped(self):
        """Stop while already stopped returns False."""
        sm = TrainingStateMachine()
        result = sm.handle_command(Command.STOP)
        assert result is False

    def test_mark_completed(self):
        """Mark completed transitions Started -> Completed."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        result = sm.mark_completed()
        assert result is True
        assert sm.is_completed()

    def test_mark_failed(self):
        """Mark failed transitions Started -> Failed."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        result = sm.mark_failed("test error")
        assert result is True
        assert sm.is_failed()

    def test_mark_completed_when_stopped(self):
        """Cannot mark completed when not started."""
        sm = TrainingStateMachine()
        result = sm.mark_completed()
        assert result is False

    def test_mark_failed_when_stopped(self):
        """Cannot mark failed when stopped."""
        sm = TrainingStateMachine()
        result = sm.mark_failed("test")
        assert result is False

    def test_set_phase_when_started(self):
        """Can set phase when started."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.set_phase(TrainingPhase.CANDIDATE)
        assert sm.phase == TrainingPhase.CANDIDATE

    def test_set_phase_when_stopped_ignored(self):
        """Setting phase when stopped is ignored."""
        sm = TrainingStateMachine()
        sm.set_phase(TrainingPhase.CANDIDATE)
        assert sm.phase == TrainingPhase.IDLE

    def test_save_and_get_candidate_state(self):
        """Save and retrieve candidate sub-state."""
        sm = TrainingStateMachine()
        state = {"epoch": 5, "best_correlation": 0.8}
        sm.save_candidate_state(state)
        retrieved = sm.get_candidate_state()
        assert retrieved == state
        assert retrieved is not state  # Should be a copy

    def test_get_state_summary(self):
        """State summary returns expected dict."""
        sm = TrainingStateMachine()
        summary = sm.get_state_summary()
        assert summary["status"] == "STOPPED"
        assert summary["phase"] == "IDLE"
        assert summary["paused_phase"] is None
        assert summary["has_candidate_state"] is False

    def test_start_from_paused(self):
        """Start command also works from paused state."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.handle_command(Command.PAUSE)
        result = sm.handle_command(Command.START)
        assert result is True
        assert sm.is_started()

    def test_stop_from_paused(self):
        """Can stop from paused state."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.handle_command(Command.PAUSE)
        result = sm.handle_command(Command.STOP)
        assert result is True
        assert sm.is_stopped()

    def test_stop_while_resume_ready_is_rejected(self):
        """STOP is legal only from STARTED or PAUSED. RESUME_READY still owns the snapshotted run.

        Accepting STOP here would drop the FSM to Stopped, and the next start would take the fresh-run path: the auto-snap ratchet would clear and the loaded history would be retained as if it were a new run.
        """
        sm = TrainingStateMachine()
        assert sm.mark_resume_ready() is True
        assert sm.handle_command(Command.STOP) is False
        assert sm.is_resume_ready()
        assert sm.phase == TrainingPhase.IDLE

    def test_mark_failed_from_paused(self):
        """Can mark failed from paused state."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.handle_command(Command.PAUSE)
        result = sm.mark_failed("error during pause")
        assert result is True
        assert sm.is_failed()

    def test_mark_completed_from_paused(self):
        """A fit can return while its run is paused, so Paused -> Completed is one transition.

        The lifecycle manager settles such a run with ``mark_completed`` alone. Accepting STARTED
        only left the run PAUSED with no training thread behind it (the paused-fit wedge).
        """
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.set_phase(TrainingPhase.CANDIDATE)
        sm.save_candidate_state({"epoch": 5})
        sm.handle_command(Command.PAUSE)
        result = sm.mark_completed()
        assert result is True
        assert sm.is_completed()
        assert sm.paused_phase is None
        assert sm.get_candidate_state() is None
        assert sm.get_state_summary() == {"status": "COMPLETED", "phase": "CANDIDATE", "paused_phase": None, "has_candidate_state": False}

    @pytest.mark.parametrize(
        "enter",
        [
            pytest.param(lambda sm: None, id="stopped"),
            pytest.param(lambda sm: (sm.handle_command(Command.START), sm.mark_completed()), id="completed"),
            pytest.param(lambda sm: (sm.handle_command(Command.START), sm.mark_failed("x")), id="failed"),
            pytest.param(lambda sm: sm.mark_resume_ready(), id="resume_ready"),
            pytest.param(lambda sm: sm.mark_investigating(), id="investigating"),
            pytest.param(lambda sm: sm.mark_replaying(), id="replaying"),
        ],
    )
    def test_mark_completed_accepts_only_started_or_paused(self, enter):
        """``mark_completed`` refuses every state but STARTED and PAUSED, and leaves it as it was."""
        sm = TrainingStateMachine()
        enter(sm)
        before = sm.get_state_summary()
        assert sm.mark_completed() is False
        assert sm.get_state_summary() == before

    def test_a_transition_is_not_split_by_another_threads_transition(self):
        """A transition holds the machine's lock from its check to its write.

        ``_handle_resume`` checks PAUSED and then calls ``_restore_from_paused``, which writes
        STARTED. A ``mark_completed`` sent from another thread while a resume sits at that call
        must wait for the resume and then complete the run from STARTED. Run in between, it would
        complete the run and the resume would then write STARTED over COMPLETED.
        """
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.handle_command(Command.PAUSE)
        restore = sm._restore_from_paused
        in_gap, release = threading.Event(), threading.Event()

        def _held_restore() -> bool:
            in_gap.set()
            release.wait(timeout=5.0)
            return restore()

        sm._restore_from_paused = _held_restore
        resumed: list = []
        completed: list = []
        resumer = threading.Thread(target=lambda: resumed.append(sm.handle_command(Command.RESUME)), daemon=True)
        completer = threading.Thread(target=lambda: completed.append(sm.mark_completed()), daemon=True)
        try:
            resumer.start()
            assert in_gap.wait(timeout=5.0), "the resume never reached its STARTED write"
            completer.start()
            completer.join(timeout=0.5)
            assert completed == [], "mark_completed ran inside the resume's transition"
        finally:
            release.set()
            resumer.join(timeout=5.0)
            completer.join(timeout=5.0)
        assert resumed == [True]
        assert completed == [True]
        assert sm.is_completed()

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda sm: sm.handle_command(Command.PAUSE), id="handle_command"),
            pytest.param(lambda sm: sm.set_phase(TrainingPhase.CANDIDATE), id="set_phase"),
            pytest.param(lambda sm: sm.save_candidate_state({"epoch": 1}), id="save_candidate_state"),
            pytest.param(lambda sm: sm.get_candidate_state(), id="get_candidate_state"),
            pytest.param(lambda sm: sm.mark_completed(), id="mark_completed"),
            pytest.param(lambda sm: sm.mark_failed("x"), id="mark_failed"),
            pytest.param(lambda sm: sm.mark_resume_ready(), id="mark_resume_ready"),
            pytest.param(lambda sm: sm.mark_investigating(), id="mark_investigating"),
            pytest.param(lambda sm: sm.mark_replaying(), id="mark_replaying"),
            pytest.param(lambda sm: sm.get_state_summary(), id="get_state_summary"),
        ],
    )
    def test_every_transition_and_multi_field_read_waits_for_the_lock(self, call):
        """Every ``@_atomic`` method waits while another thread holds the machine's lock.

        ``mark_failed`` matters as much as ``mark_completed``: a fit that raises while its run is
        paused settles through it, and a resume held between its PAUSED check and its STARTED write
        would otherwise strand the run STARTED with no training thread behind it.
        """
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        finished = threading.Event()
        worker = threading.Thread(target=lambda: (call(sm), finished.set()), daemon=True)
        with sm._lock:
            worker.start()
            assert not finished.wait(timeout=0.2), "ran while another thread held the machine's lock"
        assert finished.wait(timeout=5.0)
        worker.join(timeout=5.0)

    def test_reset_clears_candidate_state(self):
        """Reset clears saved candidate state."""
        sm = TrainingStateMachine()
        sm.save_candidate_state({"epoch": 5})
        sm.handle_command(Command.RESET)
        assert sm.get_candidate_state() is None

    def test_start_auto_resets_from_failed(self):
        """Start command auto-resets from FAILED terminal state (CR-007)."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.mark_failed("test error")
        assert sm.is_failed()
        result = sm.handle_command(Command.START)
        assert result is True
        assert sm.is_started()

    def test_start_auto_resets_from_completed(self):
        """Start command auto-resets from COMPLETED terminal state (CR-007)."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.mark_completed()
        assert sm.is_completed()
        result = sm.handle_command(Command.START)
        assert result is True
        assert sm.is_started()


@pytest.mark.unit
class TestResumeReadyState:
    """CAN-015b (Phase 6E Sprint B B-2): tests for the new
    ``RESUME_READY`` state and ``mark_resume_ready`` method."""

    def test_initial_state_is_not_resume_ready(self):
        """Fresh state machine is Stopped, not ResumeReady."""
        sm = TrainingStateMachine()
        assert not sm.is_resume_ready()

    def test_mark_resume_ready_from_stopped(self):
        """Stopped -> ResumeReady transition succeeds."""
        sm = TrainingStateMachine()
        result = sm.mark_resume_ready()
        assert result is True
        assert sm.is_resume_ready()
        assert sm.status == TrainingStatus.RESUME_READY
        # Phase resets to IDLE on entry — same as Stopped.
        assert sm.phase == TrainingPhase.IDLE

    def test_mark_resume_ready_from_completed(self):
        """Completed -> ResumeReady is permitted (resume after a finished run)."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.mark_completed()
        assert sm.is_completed()
        result = sm.mark_resume_ready()
        assert result is True
        assert sm.is_resume_ready()

    def test_mark_resume_ready_from_failed(self):
        """Failed -> ResumeReady is permitted (resume after a failed run)."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.mark_failed("test failure")
        assert sm.is_failed()
        result = sm.mark_resume_ready()
        assert result is True
        assert sm.is_resume_ready()

    def test_mark_resume_ready_idempotent(self):
        """ResumeReady -> ResumeReady is permitted (re-resume on a different snapshot)."""
        sm = TrainingStateMachine()
        sm.mark_resume_ready()
        assert sm.is_resume_ready()
        # Calling again still succeeds.
        result = sm.mark_resume_ready()
        assert result is True
        assert sm.is_resume_ready()

    def test_mark_resume_ready_rejected_when_started(self):
        """Started -> ResumeReady is REJECTED — must stop training first."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        assert sm.is_started()
        result = sm.mark_resume_ready()
        assert result is False
        # FSM unchanged.
        assert sm.is_started()
        assert not sm.is_resume_ready()

    def test_mark_resume_ready_rejected_when_paused(self):
        """Paused -> ResumeReady is REJECTED — must stop training first."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.handle_command(Command.PAUSE)
        assert sm.is_paused()
        result = sm.mark_resume_ready()
        assert result is False
        # FSM unchanged.
        assert sm.is_paused()
        assert not sm.is_resume_ready()

    def test_start_from_resume_ready_transitions_to_started(self):
        """ResumeReady + START command -> Started (Output phase) — same as
        Stopped + START, but the lifecycle wraps this with history-preserving
        logic. The FSM transition itself looks identical."""
        sm = TrainingStateMachine()
        sm.mark_resume_ready()
        assert sm.is_resume_ready()
        result = sm.handle_command(Command.START)
        assert result is True
        assert sm.is_started()
        assert sm.phase == TrainingPhase.OUTPUT

    def test_reset_from_resume_ready(self):
        """ResumeReady + RESET command -> Stopped (any state -> Stopped)."""
        sm = TrainingStateMachine()
        sm.mark_resume_ready()
        result = sm.handle_command(Command.RESET)
        assert result is True
        assert sm.is_stopped()
        assert not sm.is_resume_ready()

    def test_get_state_summary_reports_resume_ready(self):
        """get_state_summary surfaces the new state name."""
        sm = TrainingStateMachine()
        sm.mark_resume_ready()
        summary = sm.get_state_summary()
        assert summary["status"] == "RESUME_READY"
        assert summary["phase"] == "IDLE"


@pytest.mark.unit
class TestInvestigatingState:
    """CAN-015d (Phase 6E Sprint B B-4): tests for the new
    ``INVESTIGATING`` state and ``mark_investigating`` method.

    Investigating is the inspection / modification mode loaded by
    ``/restore`` — the user can edit meta-params and re-snapshot but
    cannot start training directly. They must invoke ``/retrain`` or
    ``/resume`` to transition out of Investigating before training can
    begin.
    """

    def test_initial_state_is_not_investigating(self):
        """Fresh state machine is Stopped, not Investigating."""
        sm = TrainingStateMachine()
        assert not sm.is_investigating()

    def test_mark_investigating_from_stopped(self):
        """Stopped -> Investigating transition succeeds."""
        sm = TrainingStateMachine()
        result = sm.mark_investigating()
        assert result is True
        assert sm.is_investigating()
        assert sm.status == TrainingStatus.INVESTIGATING
        assert sm.phase == TrainingPhase.IDLE

    def test_mark_investigating_from_completed(self):
        """Completed -> Investigating is permitted (load a snapshot after a finished run)."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.mark_completed()
        assert sm.is_completed()
        assert sm.mark_investigating() is True
        assert sm.is_investigating()

    def test_mark_investigating_from_failed(self):
        """Failed -> Investigating is permitted."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.mark_failed("test failure")
        assert sm.is_failed()
        assert sm.mark_investigating() is True
        assert sm.is_investigating()

    def test_mark_investigating_from_resume_ready(self):
        """ResumeReady -> Investigating is permitted (replace a Resume target with a Restore target)."""
        sm = TrainingStateMachine()
        sm.mark_resume_ready()
        assert sm.is_resume_ready()
        assert sm.mark_investigating() is True
        assert sm.is_investigating()

    def test_mark_investigating_idempotent(self):
        """Investigating -> Investigating is permitted (replace one inspected snapshot with another)."""
        sm = TrainingStateMachine()
        sm.mark_investigating()
        assert sm.mark_investigating() is True
        assert sm.is_investigating()

    def test_mark_investigating_rejected_when_started(self):
        """Started -> Investigating is REJECTED — must stop training first."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        result = sm.mark_investigating()
        assert result is False
        assert sm.is_started()
        assert not sm.is_investigating()

    def test_mark_investigating_rejected_when_paused(self):
        """Paused -> Investigating is REJECTED."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.handle_command(Command.PAUSE)
        result = sm.mark_investigating()
        assert result is False
        assert sm.is_paused()

    def test_start_rejected_from_investigating(self):
        """START command is REJECTED from Investigating — the whole
        contract: user must invoke /retrain or /resume to enter a
        training state."""
        sm = TrainingStateMachine()
        sm.mark_investigating()
        assert sm.is_investigating()
        result = sm.handle_command(Command.START)
        assert result is False
        # FSM still in Investigating — not silently auto-transitioned.
        assert sm.is_investigating()
        assert not sm.is_started()

    def test_resume_command_rejected_from_investigating(self):
        """RESUME command (used to resume from Paused) is REJECTED from
        Investigating — same logic as START."""
        sm = TrainingStateMachine()
        sm.mark_investigating()
        result = sm.handle_command(Command.RESUME)
        assert result is False
        assert sm.is_investigating()

    def test_pause_rejected_from_investigating(self):
        """PAUSE command is REJECTED from Investigating (no training to pause)."""
        sm = TrainingStateMachine()
        sm.mark_investigating()
        result = sm.handle_command(Command.PAUSE)
        assert result is False
        assert sm.is_investigating()

    def test_reset_from_investigating(self):
        """RESET command transitions Investigating -> Stopped — escape hatch
        for clearing the state if the user wants a fresh start without
        going through /retrain or /resume."""
        sm = TrainingStateMachine()
        sm.mark_investigating()
        result = sm.handle_command(Command.RESET)
        assert result is True
        assert sm.is_stopped()
        assert not sm.is_investigating()

    def test_mark_resume_ready_from_investigating(self):
        """Investigating -> ResumeReady is permitted (the user explicitly
        invokes /resume after first inspecting via /restore)."""
        sm = TrainingStateMachine()
        sm.mark_investigating()
        result = sm.mark_resume_ready()
        assert result is True
        assert sm.is_resume_ready()
        assert not sm.is_investigating()

    def test_get_state_summary_reports_investigating(self):
        """get_state_summary surfaces the new state name."""
        sm = TrainingStateMachine()
        sm.mark_investigating()
        summary = sm.get_state_summary()
        assert summary["status"] == "INVESTIGATING"
        assert summary["phase"] == "IDLE"


@pytest.mark.unit
class TestReplayingState:
    """CAN-015c (Phase 6E Sprint B B-3): tests for the new ``REPLAYING``
    state and ``mark_replaying`` method.

    Replaying is the read-only-playback mode used by ``/replay``.
    Training commands are rejected; only ``/replay/control`` is
    permitted. ``RESET`` exits the state.
    """

    def test_initial_state_is_not_replaying(self):
        sm = TrainingStateMachine()
        assert not sm.is_replaying()

    def test_mark_replaying_from_stopped(self):
        sm = TrainingStateMachine()
        assert sm.mark_replaying() is True
        assert sm.is_replaying()
        assert sm.status == TrainingStatus.REPLAYING
        assert sm.phase == TrainingPhase.IDLE

    def test_mark_replaying_from_investigating(self):
        """Replay over a prior Restore is permitted."""
        sm = TrainingStateMachine()
        sm.mark_investigating()
        assert sm.mark_replaying() is True
        assert sm.is_replaying()

    def test_mark_replaying_from_resume_ready(self):
        sm = TrainingStateMachine()
        sm.mark_resume_ready()
        assert sm.mark_replaying() is True

    def test_mark_replaying_idempotent(self):
        """Replacing one replay session with another is permitted."""
        sm = TrainingStateMachine()
        sm.mark_replaying()
        assert sm.mark_replaying() is True
        assert sm.is_replaying()

    def test_mark_replaying_from_completed(self):
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.mark_completed()
        assert sm.mark_replaying() is True

    def test_mark_replaying_from_failed(self):
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.mark_failed("test")
        assert sm.mark_replaying() is True

    def test_mark_replaying_rejected_when_started(self):
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        assert sm.mark_replaying() is False
        assert sm.is_started()

    def test_mark_replaying_rejected_when_paused(self):
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.handle_command(Command.PAUSE)
        assert sm.mark_replaying() is False
        assert sm.is_paused()

    def test_start_rejected_from_replaying(self):
        sm = TrainingStateMachine()
        sm.mark_replaying()
        assert sm.handle_command(Command.START) is False
        assert sm.is_replaying()

    def test_pause_rejected_from_replaying(self):
        sm = TrainingStateMachine()
        sm.mark_replaying()
        assert sm.handle_command(Command.PAUSE) is False
        assert sm.is_replaying()

    def test_resume_command_rejected_from_replaying(self):
        sm = TrainingStateMachine()
        sm.mark_replaying()
        assert sm.handle_command(Command.RESUME) is False
        assert sm.is_replaying()

    def test_reset_from_replaying(self):
        """RESET transitions Replaying -> Stopped (escape hatch alongside /replay/control stop)."""
        sm = TrainingStateMachine()
        sm.mark_replaying()
        assert sm.handle_command(Command.RESET) is True
        assert sm.is_stopped()

    def test_get_state_summary_reports_replaying(self):
        sm = TrainingStateMachine()
        sm.mark_replaying()
        summary = sm.get_state_summary()
        assert summary["status"] == "REPLAYING"
        assert summary["phase"] == "IDLE"

    def test_mark_investigating_rejected_from_replaying(self):
        """A Restore can't pre-empt a running replay — user must /replay/control stop first."""
        sm = TrainingStateMachine()
        sm.mark_replaying()
        assert sm.mark_investigating() is False
        assert sm.is_replaying()

    def test_mark_resume_ready_rejected_from_replaying(self):
        """Same for Resume."""
        sm = TrainingStateMachine()
        sm.mark_replaying()
        assert sm.mark_resume_ready() is False
        assert sm.is_replaying()
