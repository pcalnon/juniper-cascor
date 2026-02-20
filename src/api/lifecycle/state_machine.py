"""Formal finite state machine for training control state management.

Ported from JuniperCanopy backend/training_state_machine.py.
Ensures deterministic state transitions and prevents invalid operations.
"""

import logging
from enum import Enum, auto
from typing import Optional


class TrainingPhase(Enum):
    """Training phase enumeration."""

    IDLE = auto()
    OUTPUT = auto()
    CANDIDATE = auto()
    INFERENCE = auto()


class TrainingStatus(Enum):
    """Training status enumeration."""

    STOPPED = auto()
    STARTED = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()


class Command(Enum):
    """Control command enumeration."""

    START = auto()
    STOP = auto()
    PAUSE = auto()
    RESUME = auto()
    RESET = auto()


class TrainingStateMachine:
    """Formal finite state machine for training control.

    States:
    - Stopped: Training is not active
    - Started: Training is active (with sub-states: Output, Candidate, Inference)
    - Paused: Training is paused (remembers previous sub-state)

    Transitions:
    - Stopped -> Started (on start command)
    - Started -> Paused (on pause command, remember sub-state)
    - Paused -> Started (on resume or start command, restore sub-state)
    - Started -> Stopped (on stop command)
    - Any -> Stopped (on reset command)
    """

    def __init__(self):
        """Initialize state machine in Stopped state."""
        self.logger = logging.getLogger(__name__)
        self._status = TrainingStatus.STOPPED
        self._phase = TrainingPhase.IDLE
        self._paused_phase: Optional[TrainingPhase] = None
        self._candidate_sub_state: Optional[dict] = None

    @property
    def status(self) -> TrainingStatus:
        """Get current training status."""
        return self._status

    @property
    def phase(self) -> TrainingPhase:
        """Get current training phase."""
        return self._phase

    @property
    def paused_phase(self) -> Optional[TrainingPhase]:
        """Get phase that was active when paused."""
        return self._paused_phase

    def is_stopped(self) -> bool:
        return self._status == TrainingStatus.STOPPED

    def is_started(self) -> bool:
        return self._status == TrainingStatus.STARTED

    def is_paused(self) -> bool:
        return self._status == TrainingStatus.PAUSED

    def is_completed(self) -> bool:
        return self._status == TrainingStatus.COMPLETED

    def is_failed(self) -> bool:
        return self._status == TrainingStatus.FAILED

    def handle_command(self, command: Command) -> bool:
        """Handle control command and perform state transition.

        Returns:
            True if transition successful, False if invalid
        """
        handler = {
            Command.START: self._handle_start,
            Command.STOP: self._handle_stop,
            Command.PAUSE: self._handle_pause,
            Command.RESUME: self._handle_resume,
            Command.RESET: self._handle_reset,
        }.get(command)
        if handler is None:
            self.logger.error(f"Unknown command: {command}")
            return False
        return handler()

    def _handle_start(self) -> bool:
        if self._status == TrainingStatus.STOPPED:
            self._status = TrainingStatus.STARTED
            self._phase = TrainingPhase.OUTPUT
            self._paused_phase = None
            self._candidate_sub_state = None
            self.logger.info("State transition: Stopped -> Started (Output)")
            return True
        if self._status == TrainingStatus.PAUSED:
            return self._restore_from_paused()
        self.logger.warning("Invalid transition: START while already Started")
        return False

    def _handle_stop(self) -> bool:
        if self._status in (TrainingStatus.STARTED, TrainingStatus.PAUSED):
            prev = self._status.name
            self._reset_to_stopped()
            self.logger.info(f"State transition: {prev} -> Stopped")
            return True
        self.logger.warning("Invalid transition: STOP while already Stopped")
        return False

    def _handle_pause(self) -> bool:
        if self._status == TrainingStatus.STARTED:
            self._status = TrainingStatus.PAUSED
            self._paused_phase = self._phase
            self.logger.info(f"State transition: Started -> Paused (saved phase: {self._phase.name})")
            return True
        self.logger.warning(f"Invalid transition: PAUSE while {self._status.name}")
        return False

    def _handle_resume(self) -> bool:
        if self._status == TrainingStatus.PAUSED:
            return self._restore_from_paused()
        self.logger.warning(f"Invalid transition: RESUME while {self._status.name}")
        return False

    def _handle_reset(self) -> bool:
        prev = self._status.name
        self._reset_to_stopped()
        self.logger.info(f"State transition: {prev} -> Stopped (RESET)")
        return True

    def _restore_from_paused(self) -> bool:
        self._status = TrainingStatus.STARTED
        if self._paused_phase:
            self._phase = self._paused_phase
            self.logger.info(f"State transition: Paused -> Started (restored phase: {self._phase.name})")
        else:
            self._phase = TrainingPhase.OUTPUT
            self.logger.info("State transition: Paused -> Started (Output, no saved phase)")
        self._paused_phase = None
        return True

    def _reset_to_stopped(self) -> None:
        self._status = TrainingStatus.STOPPED
        self._phase = TrainingPhase.IDLE
        self._paused_phase = None
        self._candidate_sub_state = None

    def set_phase(self, phase: TrainingPhase) -> None:
        """Set current training phase (only when Started)."""
        if self._status != TrainingStatus.STARTED:
            self.logger.warning(f"Cannot set phase to {phase.name} while status is {self._status.name}")
            return
        prev = self._phase
        self._phase = phase
        self.logger.debug(f"Phase change: {prev.name} -> {phase.name}")

    def save_candidate_state(self, state: dict) -> None:
        """Save candidate phase sub-state for resume."""
        self._candidate_sub_state = state.copy()

    def get_candidate_state(self) -> Optional[dict]:
        """Get saved candidate phase sub-state."""
        return self._candidate_sub_state

    def mark_completed(self) -> bool:
        """Mark training as completed (terminal state)."""
        if self._status == TrainingStatus.STARTED:
            self._status = TrainingStatus.COMPLETED
            self._paused_phase = None
            self._candidate_sub_state = None
            self.logger.info("State transition: Started -> Completed")
            return True
        self.logger.warning(f"Invalid transition: mark_completed while {self._status.name}")
        return False

    def mark_failed(self, reason: str = "Unknown error") -> bool:
        """Mark training as failed (terminal state)."""
        if self._status in (TrainingStatus.STARTED, TrainingStatus.PAUSED):
            prev = self._status.name
            self._status = TrainingStatus.FAILED
            self._paused_phase = None
            self._candidate_sub_state = None
            self.logger.info(f"State transition: {prev} -> Failed ({reason})")
            return True
        self.logger.warning(f"Invalid transition: mark_failed while {self._status.name}")
        return False

    def get_state_summary(self) -> dict:
        """Get current state as dictionary."""
        return {
            "status": self._status.name,
            "phase": self._phase.name,
            "paused_phase": self._paused_phase.name if self._paused_phase else None,
            "has_candidate_state": self._candidate_sub_state is not None,
        }
