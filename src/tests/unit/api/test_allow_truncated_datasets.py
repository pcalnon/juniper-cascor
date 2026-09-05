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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


class TestCallerStanceIsNotOverridden:
    """A deployment default must not silently replace what the caller asked for.

    The partial-data contract gives an operator three options, and the third --
    "fail the data load completely" -- is expressed by sending NEITHER parameter,
    so juniper-data answers 422. The forwarding used to be an unconditional
    ``{**jd_params, "allow_truncation": True}``, with the literal key LAST in the
    merge. On any deployment with the flag on, that turned "send neither" into
    "accept", making option 3 unreachable and defeating the point of asking.
    """

    @staticmethod
    def _manager() -> TrainingLifecycleManager:
        manager = TrainingLifecycleManager.__new__(TrainingLifecycleManager)
        manager.logger = logging.getLogger("test.stance")
        manager._dataset_shortfall = None
        return manager

    @classmethod
    def _params_on_the_wire(cls, caller_params: dict, *, deployment_flag: bool) -> dict:
        """Run ``_reload_dataset`` far enough to see what reached the client.

        The fake client raises as soon as it has recorded the request, because
        the request IS the assertion -- everything after it is tensor plumbing
        this test has no opinion about.
        """
        sent: dict = {}

        class _FakeClient:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def create_dataset(self, *, generator: str, params: dict, persist: bool) -> dict:
                sent["generator"] = generator
                sent["params"] = dict(params)
                raise RuntimeError("recorded")

        settings = SimpleNamespace(juniper_data_url="http://juniper-data:8100", allow_truncated_datasets=deployment_flag)
        with (
            patch("juniper_data_client.JuniperDataClient", _FakeClient),
            patch("api.settings.Settings", lambda: settings),
            patch("api.secrets.get_secret", lambda _name: "key"),
            pytest.raises(RuntimeError),
        ):
            cls._manager()._reload_dataset(dataset_type="equities", params=dict(caller_params))
        return sent["params"]

    def test_an_explicit_false_survives_the_deployment_default(self) -> None:
        """THE REGRESSION. Option 3 must stay reachable when the flag is on."""
        params = self._params_on_the_wire({"allow_truncation": False}, deployment_flag=True)
        assert params["allow_truncation"] is False, "the deployment default overrode an explicit caller refusal -- option 3 is unreachable"

    def test_an_explicit_true_is_preserved(self) -> None:
        """The other polarity, so the fix is not merely 'False is special'."""
        params = self._params_on_the_wire({"allow_truncation": True}, deployment_flag=True)
        assert params["allow_truncation"] is True

    def test_the_deployment_default_still_applies_when_the_caller_is_silent(self) -> None:
        """The flag must keep working -- this is what it is FOR."""
        params = self._params_on_the_wire({}, deployment_flag=True)
        assert params["allow_truncation"] is True

    def test_nothing_is_forwarded_when_the_flag_is_off(self) -> None:
        """Unset means unset: juniper-data must see no opt-in and refuse with 422."""
        params = self._params_on_the_wire({}, deployment_flag=False)
        assert "allow_truncation" not in params

    def test_incomplete_rows_reaches_the_producer_untouched(self) -> None:
        """Option 2 ("drop") is expressed with this, and cascor must not strip it."""
        params = self._params_on_the_wire({"allow_truncation": True, "incomplete_rows": "drop"}, deployment_flag=False)
        assert params["incomplete_rows"] == "drop"


class TestShortfallIsPollable:
    """A log line is not a surface. Canopy has to be able to READ the shortfall."""

    def test_a_clean_dataset_annotates_nothing(self) -> None:
        """None, not a dict of empties -- a consumer branches on presence alone."""
        assert TrainingLifecycleManager._build_dataset_shortfall({}, dataset_id="d1", allow_truncated=False) is None

    def test_the_annotation_names_the_dataset_it_describes(self) -> None:
        """An annotation that does not identify its artifact is a claim about nothing.

        cascor issues its OWN create_dataset, and the deployment default can
        change the params -- so its content-addressed id need not equal the
        driver's. Recording the id beside the annotation is what stops the two
        being silently attributed to each other.
        """
        meta = {"truncation": {"unit": "symbols", "cap": 14, "requested": 503, "imported": 14}}
        built = TrainingLifecycleManager._build_dataset_shortfall(meta, dataset_id="abc123", allow_truncated=True)
        assert built is not None
        assert built["dataset_id"] == "abc123"
        assert built["accepted_via_allow_truncated_datasets"] is True
        assert built["truncation"] == meta["truncation"]
        assert "14" in built["summary"] and "503" in built["summary"]

    def test_unrescued_and_degraded_stay_distinct_in_the_annotation(self) -> None:
        """Same reason the log keeps them apart: absent is not recovered-from-weaker."""
        meta = {
            "data_quality": {
                "unrescued": {"STZ": "no shares concept"},
                "degraded": {"META": "period_average"},
                "rows_affected": 1510,
                "policy": "accept",
            }
        }
        built = TrainingLifecycleManager._build_dataset_shortfall(meta, dataset_id="d2", allow_truncated=True)
        assert built is not None
        assert built["data_quality"]["unrescued"] == {"STZ": "no shares concept"}
        assert built["data_quality"]["degraded"] == {"META": "period_average"}
        assert "unresolvable" in built["summary"] and "weaker source" in built["summary"]

    def test_get_status_carries_it(self) -> None:
        """The single field canopy needs -- and it reaches the WS stream for free."""
        manager = TrainingLifecycleManager.__new__(TrainingLifecycleManager)
        manager.network = None
        manager.state_machine = MagicMock()
        manager.state_machine.get_state_summary.return_value = {}
        manager.state_machine.is_started.return_value = False
        manager.monitor = MagicMock()
        manager.monitor.get_current_state.return_value = {}
        manager.training_state = MagicMock()
        manager.training_state.get_state.return_value = {}
        manager.get_pending_dataset_config = lambda: None
        manager._metrics_undo_available = lambda: False

        manager._dataset_shortfall = None
        assert manager.get_status()["dataset_shortfall"] is None

        manager._dataset_shortfall = {"dataset_id": "d3", "summary": "14 of 503 symbols imported (cap 14)"}
        assert manager.get_status()["dataset_shortfall"]["dataset_id"] == "d3"
