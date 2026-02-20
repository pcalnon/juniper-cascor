"""Tests for training state machine."""

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

    def test_mark_failed_from_paused(self):
        """Can mark failed from paused state."""
        sm = TrainingStateMachine()
        sm.handle_command(Command.START)
        sm.handle_command(Command.PAUSE)
        result = sm.mark_failed("error during pause")
        assert result is True
        assert sm.is_failed()

    def test_reset_clears_candidate_state(self):
        """Reset clears saved candidate state."""
        sm = TrainingStateMachine()
        sm.save_candidate_state({"epoch": 5})
        sm.handle_command(Command.RESET)
        assert sm.get_candidate_state() is None
