"""The --allow-truncated-datasets opt-in: default, surfaces, and run failure.

Project:       Juniper
Sub-Project:   JuniperCascor
Application:   juniper_cascor
File Name:     test_allow_truncated_datasets.py
Author:        Paul Calnon
License:       MIT License

juniper-data refuses (422) a dataset it cannot produce in full unless the caller
opts in. This pins the cascor side of that contract: the flag is OFF by default,
reaches the same setting from three surfaces, and -- when unset -- turns the
producer's refusal into a run failure whose message names the knob to turn.

The failure path is the one that matters. A run that quietly trains on a partial
dataset reports a score for data nobody chose, and nothing downstream can tell.
"""

from __future__ import annotations

import logging
import os
import sys
from unittest.mock import patch

import pytest

from api.lifecycle.manager import TrainingLifecycleManager
from api.settings import Settings
from cascor_constants.constants_api.constants_api_defaults import _PROJECT_API_ALLOW_TRUNCATED_DATASETS_DEFAULT

pytestmark = pytest.mark.unit


class TestDefaultAndSurfaces:
    """OFF by default, and settable three ways."""

    def test_the_constant_is_false(self) -> None:
        """The default lives in the constants class and is False.

        Stated as its own arm because every other surface derives from it: if
        this flips, every run silently starts accepting partial data.
        """
        assert _PROJECT_API_ALLOW_TRUNCATED_DATASETS_DEFAULT is False

    def test_settings_default_is_the_constant(self) -> None:
        """Settings must not restate the default -- it sources it."""
        assert Settings().allow_truncated_datasets is _PROJECT_API_ALLOW_TRUNCATED_DATASETS_DEFAULT

    def test_environment_variable_turns_it_on(self) -> None:
        """JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS, the CLI-less surface."""
        with patch.dict(os.environ, {"JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS": "true"}):
            assert Settings().allow_truncated_datasets is True

    def test_cli_flag_parses_and_defaults_off(self) -> None:
        import main

        with patch.object(sys, "argv", ["main.py"]):
            assert main.parse_args().allow_truncated_datasets is False
        with patch.object(sys, "argv", ["main.py", "--allow-truncated-datasets"]):
            assert main.parse_args().allow_truncated_datasets is True


class TestRunFailureMessage:
    """An unmet shortfall must fail the run, and say what to do about it."""

    def test_a_shortfall_refusal_names_the_remedy(self) -> None:
        """The 422 case gets the actionable message, not a generic fetch error."""
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(
            Exception("HTTP 422: ... Re-submit with allow_truncation=true ..."),
            allow_truncated=False,
        )
        assert "--allow-truncated-datasets" in message
        assert "JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS" in message
        assert "allow_truncated_datasets:" in message
        assert "FAILING" in message
        # The producer's own detail is quoted, not replaced -- it names the
        # affected symbols and row counts, which cascor cannot know.
        assert "422" in message

    def test_an_ordinary_outage_is_not_dressed_up_as_a_shortfall(self) -> None:
        """A connection failure must not tell the operator to set a truncation flag."""
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(Exception("connection refused"), allow_truncated=False)
        assert message == "juniper-data fetch failed: connection refused"
        assert "allow-truncated" not in message

    def test_an_already_opted_in_run_gets_the_plain_message(self) -> None:
        """If the flag is already set, the shortfall was not the reason -- do not misdirect."""
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(Exception("HTTP 422 allow_truncation"), allow_truncated=True)
        assert message.startswith("juniper-data fetch failed:")


class TestShortfallLogging:
    """An accepted shortfall has to be visible in THIS run's log."""

    @staticmethod
    def _manager() -> TrainingLifecycleManager:
        manager = TrainingLifecycleManager.__new__(TrainingLifecycleManager)
        manager.logger = logging.getLogger("test.shortfall")
        return manager

    def test_a_clean_dataset_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            self._manager()._log_dataset_shortfall({}, allow_truncated=False)
        assert caplog.records == []

    def test_truncation_is_reported_with_its_numbers(self, caplog: pytest.LogCaptureFixture) -> None:
        meta = {"truncation": {"unit": "symbols", "cap": 14, "requested": 503, "imported": 14}}
        with caplog.at_level(logging.WARNING):
            self._manager()._log_dataset_shortfall(meta, allow_truncated=True)
        text = caplog.text
        assert "DATASET IS PARTIAL" in text
        assert "503" in text and "14" in text

    def test_unrescued_and_degraded_are_reported_separately(self, caplog: pytest.LogCaptureFixture) -> None:
        """They are different problems and must not be collapsed into one line.

        `unrescued` means a value is absent; `degraded` means it was recovered
        from a weaker source, so quantities derived from it are not comparable
        with the rest. An operator needs to be able to tell those apart.
        """
        meta = {
            "data_quality": {
                "unrescued": {"STZ": "no shares concept"},
                "degraded": {"META": "period_average"},
                "rows_affected": 1510,
                "policy": "accept",
            }
        }
        with caplog.at_level(logging.WARNING):
            self._manager()._log_dataset_shortfall(meta, allow_truncated=True)
        text = caplog.text
        assert "UNRESOLVABLE" in text and "STZ" in text and "1510" in text
        assert "DEGRADED" in text and "META=period_average" in text
        assert "NOT directly comparable" in text

    def test_drop_policy_says_dropped_not_filled(self, caplog: pytest.LogCaptureFixture) -> None:
        meta = {"data_quality": {"unrescued": {"STZ": "x"}, "degraded": {}, "rows_affected": 0, "policy": "drop"}}
        with caplog.at_level(logging.WARNING):
            self._manager()._log_dataset_shortfall(meta, allow_truncated=True)
        assert "were dropped" in caplog.text
