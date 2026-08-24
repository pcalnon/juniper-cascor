#!/usr/bin/env python
"""
Unit tests for the opt-in candidate-worker profiling dispatcher.

The candidate workers are FORKED, so nothing that profiles the parent sees their time --
and the candidate phase is where the cost is. `main.py --profile` instruments the parent
only, and the service has no equivalent flag, so there was no way to look inside a worker
from either entry point. `JUNIPER_CASCOR_WORKER_PROFILE=<dir>` makes each worker dump a
.prof there.

The load-bearing property is that this is INERT by default: a measurement run must not be
perturbed by the existence of the hook. The second property is that profiling must never
break training -- an unwritable profile directory is a lost profile, not a failed run.

Tests focus on:
- unset env -> the implementation is called directly, nothing is written
- set env -> a .prof lands in the directory
- a worker that raises still leaves its profile behind (the `finally`)
- an unwritable profile directory does not break training
"""

import os
import sys
from unittest.mock import patch

import pytest

# Add parent directories for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit

ENV_VAR = "JUNIPER_CASCOR_WORKER_PROFILE"


@pytest.fixture(autouse=True)
def _clear_env():
    """The dispatcher reads process env; never leak it between tests."""
    previous = os.environ.pop(ENV_VAR, None)
    yield
    os.environ.pop(ENV_VAR, None)
    if previous is not None:
        os.environ[ENV_VAR] = previous


class TestWorkerProfileDispatcher:
    """JUNIPER_CASCOR_WORKER_PROFILE is opt-in and never load-bearing."""

    def test_unset_env_calls_through_without_profiling(self, tmp_path):
        """Default path: delegate straight to the implementation, write nothing."""
        with patch.object(CascadeCorrelationNetwork, "_train_candidate_worker_impl", return_value="sentinel") as impl:
            result = CascadeCorrelationNetwork.train_candidate_worker(task_data_input=("task",), parallel=False)

        assert result == "sentinel"
        impl.assert_called_once_with(("task",), False, None)
        assert list(tmp_path.iterdir()) == []

    def test_empty_env_is_treated_as_unset(self):
        """Whitespace-only is not a directory name."""
        os.environ[ENV_VAR] = "   "
        with patch.object(CascadeCorrelationNetwork, "_train_candidate_worker_impl", return_value="sentinel") as impl:
            result = CascadeCorrelationNetwork.train_candidate_worker(task_data_input=None, parallel=False)

        assert result == "sentinel"
        impl.assert_called_once()

    def test_set_env_writes_a_profile(self, tmp_path):
        """Opt-in path: a .prof lands in the requested directory."""
        target = tmp_path / "profiles"
        os.environ[ENV_VAR] = str(target)

        with patch.object(CascadeCorrelationNetwork, "_train_candidate_worker_impl", return_value="sentinel") as impl:
            result = CascadeCorrelationNetwork.train_candidate_worker(task_data_input=None, parallel=False)

        assert result == "sentinel", "profiling must not change the return value"
        impl.assert_called_once()
        profiles = list(target.glob("*.prof"))
        assert len(profiles) == 1, f"expected exactly one profile, got {profiles}"
        assert str(os.getpid()) in profiles[0].name, profiles[0].name
        assert profiles[0].stat().st_size > 0

    def test_profile_is_written_even_when_the_worker_raises(self, tmp_path):
        """The `finally` -- an exception path is when the timing is most worth seeing."""
        target = tmp_path / "profiles"
        os.environ[ENV_VAR] = str(target)

        with patch.object(CascadeCorrelationNetwork, "_train_candidate_worker_impl", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                CascadeCorrelationNetwork.train_candidate_worker(task_data_input=None, parallel=False)

        assert len(list(target.glob("*.prof"))) == 1, "a failing worker left no profile behind"

    def test_unwritable_profile_dir_does_not_break_training(self, tmp_path):
        """A lost profile is acceptable; a failed training run is not."""
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("this is a file, so mkdir() beneath it must fail")
        os.environ[ENV_VAR] = str(blocker / "profiles")

        with patch.object(CascadeCorrelationNetwork, "_train_candidate_worker_impl", return_value="sentinel") as impl:
            result = CascadeCorrelationNetwork.train_candidate_worker(task_data_input=None, parallel=False)

        assert result == "sentinel"
        impl.assert_called_once()
