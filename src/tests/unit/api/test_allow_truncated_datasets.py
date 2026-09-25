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

import inspect
import itertools
import logging
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from juniper_data_client import JuniperDataValidationError

from api.lifecycle.manager import _FETCH_PATH_AUTO_START, _FETCH_PATH_LIVE_SWAP, _FETCH_PATH_STAGED_START, _OPT_IN_SKIPPED_CALLER_DEFERRED, _OPT_IN_SKIPPED_LIST_UNREADABLE, _TRUNCATABLE_GENERATORS, TrainingLifecycleManager
from api.settings import Settings
from cascor_constants.constants_api.constants_api_defaults import _PROJECT_API_ALLOW_TRUNCATED_DATASETS_DEFAULT, _PROJECT_API_SHORTFALL_ACCEPTED_BY_DEPLOYMENT, _PROJECT_API_SHORTFALL_ACCEPTED_BY_PRODUCER, _PROJECT_API_SHORTFALL_ACCEPTED_BY_REQUEST, _PROJECT_API_SHORTFALL_REFUSAL_TOKEN

pytestmark = pytest.mark.unit


def _canopy_producer_detail(detail: str) -> str:
    """juniper-canopy's ``_producer_detail_from_refusal``, VERBATIM -- the other half of a cross-repo contract.

    Copied from juniper-canopy ``src/frontend/dashboard_manager.py`` (origin/main
    ``7ab994e5``, lines 8404-8413; the body is unchanged since ``e9053227``, where it
    sat at 8332-8341), because canopy is not installed where these tests run. canopy shows the operator what this returns as juniper-data's own
    words, so every cascor refusal must put ``" To accept it,"`` straight after the
    producer's detail: text placed before it is shown as if the producer wrote it.
    If canopy changes its cut, change this copy with it.
    """
    marker = "Producer detail: "
    if marker not in detail:
        return ""
    tail = detail.split(marker, 1)[1]
    for stop in (" To accept it,", " The resulting dataset"):
        if stop in tail:
            tail = tail.split(stop, 1)[0]
    return tail.strip()


# Every input combination that reaches a refusal branch of
# ``_describe_dataset_fetch_failure``: one per ``stance``, and the withheld branch
# once per fetch path, since each path appends its own retry to that remedy.
_EVERY_REFUSAL_BRANCH: Dict[str, Dict[str, Any]] = {
    "flag off, caller silent": {},
    "caller refused": {"caller_refused": True},
    "caller deferred, flag off": {"opt_in_skipped": _OPT_IN_SKIPPED_CALLER_DEFERRED},
    "caller deferred, flag on": {"opt_in_skipped": _OPT_IN_SKIPPED_CALLER_DEFERRED, "deployment_flag_on": True},
    "withheld, staged start": {"opt_in_skipped": _OPT_IN_SKIPPED_LIST_UNREADABLE, "deployment_flag_on": True, "fetch_path": _FETCH_PATH_STAGED_START},
    "withheld, live swap": {"opt_in_skipped": _OPT_IN_SKIPPED_LIST_UNREADABLE, "deployment_flag_on": True, "fetch_path": _FETCH_PATH_LIVE_SWAP},
    "withheld, auto-start": {"opt_in_skipped": _OPT_IN_SKIPPED_LIST_UNREADABLE, "deployment_flag_on": True, "fetch_path": _FETCH_PATH_AUTO_START},
    "withheld, no path named": {"opt_in_skipped": _OPT_IN_SKIPPED_LIST_UNREADABLE, "deployment_flag_on": True},
    "flag on, no reason given": {"deployment_flag_on": True},
}
# juniper-data's OWN words, as juniper-data-client 0.5.0 renders them -- ``Validation
# error (<status>): <detail>``, with the status on ``status_code``. #688's validation
# found the refusal check passing on a stand-in that merely NAMED a truncation field,
# while a real juniper-data 400 names the field it rejects too. So the refusal texts
# below are real, and so are the 400s that must NOT read as refusals. Captured from a
# juniper-data main server (0f0f7e0) by juniper-ml
# ``util/ad-hoc/2026-09-24_cascor688_realjd_error_texts.py``, except the
# ``IncompleteDataError`` one, which needs a symbol whose shares cannot be resolved:
# that one is juniper-data's own class rendering the equities generator's detail.
_REFUSAL_INPUT_TOO_LARGE = "Validation error (422): The requested universe is 3 symbols, over the 2 symbols cap. Re-submit with allow_truncation=true (or set JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION=true) to import the first 2 symbols. The resulting dataset will be permanently annotated as truncated."
_REFUSAL_INCOMPLETE_DATA = "Validation error (422): Shares outstanding could not be resolved for part of the requested universe, so total_shares and market_cap would be fabricated for those rows. Affected (3): BF.B, BRK.B, STZ. 1,510 row(s) would carry fabricated values. Re-submit with allow_truncation=true (or set JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION=true) to accept them, or with incomplete_rows='drop' to exclude them. Either choice is recorded permanently in the dataset's metadata."
_PARAMETER_ERRORS: Dict[str, str] = {
    "n_spirals=1": "Validation error (400): Invalid parameters: 1 validation error for SpiralParams\nn_spirals\n  Input should be greater than or equal to 2 [type=greater_than_equal, input_value=1, input_type=int]\n    For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal",
    "incomplete_rows='keep'": "Validation error (400): Invalid parameters: 1 validation error for EquitiesParams\nincomplete_rows\n  Input should be 'accept' or 'drop' [type=literal_error, input_value='keep', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/literal_error",
    "allow_truncation=''": "Validation error (400): Invalid parameters: 1 validation error for CsvImportParams\nallow_truncation\n  Input should be a valid boolean, unable to interpret input [type=bool_parsing, input_value='', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/bool_parsing",
    "allow_truncation='  '": "Validation error (400): Invalid parameters: 1 validation error for CsvImportParams\nallow_truncation\n  Input should be a valid boolean, unable to interpret input [type=bool_parsing, input_value='  ', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/bool_parsing",
    # A 400 ECHOES the value it rejects, so a value that quotes the remedy sentence puts
    # the sentence in a 400. Only the status tells this one apart from a refusal.
    "incomplete_rows quoting the remedy": "Validation error (400): Invalid parameters: 1 validation error for EquitiesParams\nincomplete_rows\n  Input should be 'accept' or 'drop' [type=literal_error, input_value='Re-submit with allow_truncation=true', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/literal_error",
}
# Each 400 as juniper-data-client 0.5.0 raises it (with ``status_code``) and as an older
# client does (none), except the one only a status can separate from a refusal.
_PARAMETER_ERROR_CASES = [(case, status) for status in (True, False) for case in sorted(_PARAMETER_ERRORS) if status or "remedy" not in case]
# The other kind of 422: a request that breaks juniper-data's DECLARED request schema,
# rejected by FastAPI at the boundary. It carries no remedy sentence.
_REQUEST_SCHEMA_422 = "Validation error (422): body.params: Input should be a valid dictionary"
_PRODUCER_DETAIL = _REFUSAL_INCOMPLETE_DATA


def _refusal(detail: str = _REFUSAL_INCOMPLETE_DATA) -> JuniperDataValidationError:
    """A shortfall refusal exactly as the live path receives it: juniper-data-client's error for a 422."""
    return JuniperDataValidationError(detail, status_code=422)


# What juniper-data's ``GET /v1/generators`` says, reduced to the one fact the
# stance resolver reads: whether the param schema declares ``allow_truncation``.
# Since APD-CASCOR-008 the truncatable set is derived from this rather than from a
# cascor constant, so every fake producer below has to answer it -- a fake that
# cannot is an UNREADABLE list, and the deployment default is then withheld.
_GENERATOR_LISTING = [
    {"name": "equities", "schema": {"properties": {"allow_truncation": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}}}},
    {"name": "spiral", "schema": {"properties": {"n_spirals": {"type": "integer"}}}},
]


@pytest.fixture(autouse=True)
def _fresh_truncatable_memo():
    """The derived set is memoised per process; one test's success must not serve the next."""
    _TRUNCATABLE_GENERATORS.reset()
    yield
    _TRUNCATABLE_GENERATORS.reset()


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
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(_refusal(), allow_truncated=False)
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
        """If the flag is already set, the shortfall was not the reason -- do not misdirect.

        A REAL refusal, so it is the opt-in that makes this plain. The stand-in this
        used ("HTTP 422 allow_truncation") no longer reads as a refusal at all, so the
        arm would have passed whatever the opt-in did.
        """
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(_refusal(), allow_truncated=True)
        assert message == f"juniper-data fetch failed: {_REFUSAL_INCOMPLETE_DATA}"

    def test_the_refusal_opens_with_a_machine_readable_token(self) -> None:
        """A consumer (canopy's three-way prompt) must recognise the class without matching prose.

        The token is the contract; the sentence after it is free to change. An
        ordinary outage must NOT carry it, or the prompt fires on a dead service.
        """
        refusal = TrainingLifecycleManager._describe_dataset_fetch_failure(_refusal(), allow_truncated=False)
        assert refusal.startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN + " ")
        outage = TrainingLifecycleManager._describe_dataset_fetch_failure(Exception("connection refused"), allow_truncated=False)
        assert _PROJECT_API_SHORTFALL_REFUSAL_TOKEN not in outage

    def test_a_caller_that_refused_is_told_to_resend_not_to_flip_the_setting(self) -> None:
        """An explicit allow_truncation=false wins over the service setting (cascor#624).

        Pointing that caller at --allow-truncated-datasets would send them to a knob
        that cannot change the outcome. The remedy is the request's own two
        parameters, and the message must say which stance was actually taken.
        """
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(_refusal(), allow_truncated=False, caller_refused=True)
        assert message.startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN)
        assert "explicitly refused" in message
        assert "allow_truncation=true" in message
        assert "incomplete_rows=accept" in message and "incomplete_rows=drop" in message
        assert "--allow-truncated-datasets" not in message

    @pytest.mark.parametrize("deployment_flag_on", [True, False], ids=["flag-on", "flag-off"])
    def test_an_ordinary_422_is_not_a_shortfall_refusal(self, deployment_flag_on: bool) -> None:
        """#686's validation: any 422 used to read as a shortfall.

        juniper-data answers 422 for two things: a shortfall refusal, and a request
        that breaks its DECLARED request schema -- here ``params`` not a mapping,
        rejected by FastAPI at the boundary. The second carries no remedy sentence, so
        it is a plain failure in both flag positions. This arm used a ``spiral``
        ``n_spirals=1`` detail rendered as a 422 until #688's validation; juniper-data
        answers that one 400, which is the next arm's subject.
        """
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(JuniperDataValidationError(_REQUEST_SCHEMA_422, status_code=422), allow_truncated=False, deployment_flag_on=deployment_flag_on)
        assert message == f"juniper-data fetch failed: {_REQUEST_SCHEMA_422}"

    @pytest.mark.parametrize("deployment_flag_on", [True, False], ids=["flag-on", "flag-off"])
    @pytest.mark.parametrize(("case", "carries_a_status"), _PARAMETER_ERROR_CASES, ids=[f"{case}-{'client-0.5.0' if status else 'older-client'}" for case, status in _PARAMETER_ERROR_CASES])
    def test_a_parameter_error_is_not_a_refusal_even_when_it_names_a_truncation_field(self, case: str, carries_a_status: bool, deployment_flag_on: bool) -> None:
        """#688's validation, MEDIUM: a REAL juniper-data 400 that names ``allow_truncation`` or ``incomplete_rows``.

        juniper-data answers a parameter its generator rejects with 400, and names
        the field. The check matched those names, so ``incomplete_rows='keep'`` and a
        blank ``allow_truncation`` carried the refusal token -- which opens canopy's
        partial-data prompt -- and, with the flag off, told the operator to set the
        flag, which cannot fix a bad parameter. Measured against a real juniper-data
        main server. Each is also raised as a juniper-data-client older than 0.5.0
        raises it, with no status, where the remedy sentence alone must decide.

        The last case puts the sentence itself in a 400, as the echoed input value.
        Only the status tells that one apart from a refusal, so it has no
        older-client form: without a status it is indistinguishable, and reads as one.
        """
        detail = _PARAMETER_ERRORS[case]
        exc = JuniperDataValidationError(detail, status_code=400) if carries_a_status else Exception(detail)
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(exc, allow_truncated=False, deployment_flag_on=deployment_flag_on)
        assert message == f"juniper-data fetch failed: {detail}"

    @pytest.mark.parametrize("carries_a_status", [True, False], ids=["client-0.5.0", "older-client"])
    @pytest.mark.parametrize("detail", [_REFUSAL_INPUT_TOO_LARGE, _REFUSAL_INCOMPLETE_DATA], ids=["InputTooLargeError", "IncompleteDataError"])
    def test_both_of_the_producers_refusals_are_still_recognised(self, detail: str, carries_a_status: bool) -> None:
        """The guard on the other side: a check that rejects the 400s must not lose a real refusal.

        Worded as juniper-data's ``InputTooLargeError`` and ``IncompleteDataError``
        render them (``juniper_data/core/limits.py``), after juniper-data-client
        prefixes ``Validation error (422):``. juniper-data-client 0.5.0 also puts the
        422 on ``status_code``; the releases before it -- this service floors the
        dependency at 0.3.0 -- raise the same text with no status, and the remedy
        sentence alone must still be enough.
        """
        exc = JuniperDataValidationError(detail, status_code=422) if carries_a_status else Exception(detail)
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(exc, allow_truncated=False)
        assert message.startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN + " ")
        assert "--allow-truncated-datasets" in message
        assert f"Producer detail: {detail} To accept it," in message

    @pytest.mark.parametrize("branch", sorted(_EVERY_REFUSAL_BRANCH))
    def test_every_refusal_branch_leaves_canopy_the_producers_own_text(self, branch: str) -> None:
        """cascor#678 follow-up, item 2. canopy cuts juniper-data's sentence out at ``" To accept it,"``.

        #678's WITHHELD remedy opened with "Retry once ...", so canopy showed about
        400 characters of cascor's retry advice as if juniper-data had written them
        (measured in #678's post-merge validation by running canopy's function on
        every branch). Every remedy now BEGINS with the phrase, so the cut leaves
        exactly the producer's text on every branch.
        """
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(Exception(_PRODUCER_DETAIL), allow_truncated=False, **_EVERY_REFUSAL_BRANCH[branch])
        assert message.startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN + " ")
        assert " To accept it," in message
        assert _canopy_producer_detail(message) == _PRODUCER_DETAIL

    def test_the_branch_table_above_covers_every_refusal_the_describer_can_emit(self) -> None:
        """Enumerate the CALLER, not a declared set: a new ``stance`` branch must join ``_EVERY_REFUSAL_BRANCH``.

        Counted from the describer's own source, so a branch added there and not
        here fails this arm instead of shipping unchecked against canopy's cut.
        """
        stances = set()
        for kwargs in _EVERY_REFUSAL_BRANCH.values():
            message = TrainingLifecycleManager._describe_dataset_fetch_failure(Exception(_PRODUCER_DETAIL), allow_truncated=False, **kwargs)
            stances.add(message.split(" in full, ", 1)[1].split(", so the run is FAILING", 1)[0])
        source = inspect.getsource(TrainingLifecycleManager._describe_dataset_fetch_failure)
        assert len(stances) == source.count("stance = ")


class TestShortfallLogging:
    """An accepted shortfall has to be visible in THIS run's log."""

    @staticmethod
    def _manager() -> TrainingLifecycleManager:
        manager = TrainingLifecycleManager.__new__(TrainingLifecycleManager)
        manager.logger = logging.getLogger("test.shortfall")
        return manager

    def test_a_clean_dataset_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            self._manager()._log_dataset_shortfall({}, acceptance_source=None)
        assert caplog.records == []

    def test_truncation_is_reported_with_its_numbers(self, caplog: pytest.LogCaptureFixture) -> None:
        meta = {"truncation": {"unit": "symbols", "cap": 14, "requested": 503, "imported": 14}}
        with caplog.at_level(logging.WARNING):
            self._manager()._log_dataset_shortfall(meta, acceptance_source=_PROJECT_API_SHORTFALL_ACCEPTED_BY_DEPLOYMENT)
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
            self._manager()._log_dataset_shortfall(meta, acceptance_source=_PROJECT_API_SHORTFALL_ACCEPTED_BY_DEPLOYMENT)
        text = caplog.text
        assert "UNRESOLVABLE" in text and "STZ" in text and "1510" in text
        assert "DEGRADED" in text and "META=period_average" in text
        assert "NOT directly comparable" in text

    def test_drop_policy_says_dropped_not_filled(self, caplog: pytest.LogCaptureFixture) -> None:
        meta = {"data_quality": {"unrescued": {"STZ": "x"}, "degraded": {}, "rows_affected": 0, "policy": "drop"}}
        with caplog.at_level(logging.WARNING):
            self._manager()._log_dataset_shortfall(meta, acceptance_source=_PROJECT_API_SHORTFALL_ACCEPTED_BY_DEPLOYMENT)
        assert "were dropped" in caplog.text

    def test_the_log_names_who_accepted_and_the_producer_when_nobody_here_did(self, caplog: pytest.LogCaptureFixture) -> None:
        """juniper-data applies its own deployment opt-in to a request that sends no stance.

        A partial dataset that arrives with no opt-in sent from this side was accepted
        by the PRODUCER, and the log must say so rather than restate a setting that
        was off -- the old line read "accepted it via allow_truncated_datasets=False".

        This docstring said "a client cannot opt out" until juniper-data APD-DATA-052
        made ``allow_truncation`` a tri-state. One CAN now, by sending ``false``; this
        test covers the case where nobody did, which is still reachable and still the
        case the PRODUCER value exists for.
        """
        meta = {"truncation": {"unit": "symbols", "cap": 14, "requested": 503, "imported": 14}}
        with caplog.at_level(logging.WARNING):
            self._manager()._log_dataset_shortfall(meta, acceptance_source=None)
        assert "DATASET SHORTFALL" in caplog.text
        assert "producer's own deployment default" in caplog.text
        assert "allow_truncated_datasets=False" not in caplog.text

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            self._manager()._log_dataset_shortfall(meta, acceptance_source=_PROJECT_API_SHORTFALL_ACCEPTED_BY_REQUEST)
        assert "dataset request itself" in caplog.text


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

            def list_generators(self) -> list:
                return _GENERATOR_LISTING

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

    def test_a_refusal_after_an_explicit_false_still_names_a_remedy(self) -> None:
        """The failure message must key off the WIRE stance, not the setting.

        Flag ON, caller sends allow_truncation=false: cascor withholds its default
        (correct), the producer refuses, and the message used to consult the
        SETTING -- so it returned the bare "fetch failed" line, with no remedy, in
        exactly the case the remedy exists for. Found by round-37 validation.
        """
        sent: dict = {}

        class _RefusingClient:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def list_generators(self) -> list:
                return _GENERATOR_LISTING

            def create_dataset(self, *, generator: str, params: dict, persist: bool) -> dict:
                sent["params"] = dict(params)
                raise RuntimeError("HTTP 422: Shares outstanding could not be resolved for part of the requested universe. Re-submit with allow_truncation=true")

        settings = SimpleNamespace(juniper_data_url="http://juniper-data:8100", allow_truncated_datasets=True)
        with (
            patch("juniper_data_client.JuniperDataClient", _RefusingClient),
            patch("api.settings.Settings", lambda: settings),
            patch("api.secrets.get_secret", lambda _name: "key"),
            pytest.raises(RuntimeError) as excinfo,
        ):
            self._manager()._reload_dataset(dataset_type="equities", params={"allow_truncation": False})
        assert sent["params"]["allow_truncation"] is False, "the caller's refusal must reach the producer unchanged"
        message = str(excinfo.value)
        assert message.startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN)
        assert "explicitly refused" in message and "allow_truncation=true" in message


# What juniper-data makes of a caller's ``allow_truncation``, MEASURED rather than restated.
# Every value here was validated through the REAL ``EquitiesParams``, ``EquitiesSeqParams``
# and ``CsvImportParams`` of juniper-data main (0f0f7e0, pydantic 2.12.5), which agree on
# all of them, by juniper-ml
# ``util/ad-hoc/2026-09-24_cascor690_bool_stance_producer_table.py``. Each class types the
# field ``bool | None`` under pydantic's lax coercion. A REJECTED value is a 400:
# juniper-data refuses the request as a bad parameter.
_PRODUCER_READS_AS_TRUE: List[Any] = [True, 1, 1.0, "1", "ON", "On", "oN", "on", "T", "t", "TRUE", "True", "tRUE", "true", "Y", "y", "YES", "Yes", "yES", "yes"]
_PRODUCER_READS_AS_FALSE: List[Any] = [False, 0, 0.0, -0.0, "0", "OFF", "Off", "oFF", "off", "F", "f", "FALSE", "False", "fALSE", "false", "N", "n", "NO", "No", "nO", "no"]
_PRODUCER_REJECTS: List[Any] = [
    *(2, -1, 10, 0.5, 1.5, 2.0, -1.0, float("nan"), float("inf"), float("-inf")),
    *(" true", "true ", "\ttrue", "true\n", " 1", "0 ", " f", "no ", "", "  "),
    *("maybe", "2", "-1", "1.0", "0.0", "tru", "truee", "yes!", "none", "null", "None", "nil", "enable", "disabled", "ok"),
    *("ｔｒｕｅ", "ｆ", "trüe", "ＹＥＳ", [], [True], {}, {"a": 1}),
]
# pydantic-core's ``str_as_bool`` (``src/input/shared.rs``): the twelve strings, by polarity.
_PYDANTIC_TRUE_STRINGS = {"1", "on", "t", "true", "y", "yes"}
_PYDANTIC_FALSE_STRINGS = {"0", "off", "f", "false", "n", "no"}


class TestTheStanceIsReadAsJuniperDataReadsIt:
    """A caller's ``allow_truncation`` means here exactly what it means to juniper-data (#690's fixup).

    ``_as_bool_stance`` used to fall back to truthiness for anything it did not list, and to
    strip whitespace. So "f" and "n", which juniper-data reads as False, read as an opt-in.
    A caller refusing a partial dataset that way was recorded as accepting it, and the
    refusal that followed came out as a plain fetch failure with no remedy. Every value
    juniper-data rejects ("maybe", ``2``, a padded " true") also read as a stance, for a
    request that can only fail as a bad parameter. They are no stance now, as a blank string
    is.
    """

    @pytest.mark.parametrize("value", _PRODUCER_READS_AS_TRUE, ids=repr)
    def test_every_spelling_juniper_data_reads_as_true_is_an_opt_in(self, value: Any) -> None:
        assert TrainingLifecycleManager._as_bool_stance(value) is True

    @pytest.mark.parametrize("value", _PRODUCER_READS_AS_FALSE, ids=repr)
    def test_every_spelling_juniper_data_reads_as_false_is_a_refusal(self, value: Any) -> None:
        assert TrainingLifecycleManager._as_bool_stance(value) is False

    @pytest.mark.parametrize("value", [None, *_PRODUCER_REJECTS], ids=repr)
    def test_null_and_every_value_juniper_data_rejects_are_no_stance(self, value: Any) -> None:
        assert TrainingLifecycleManager._as_bool_stance(value) is None

    def test_the_table_holds_every_spelling_the_producer_accepts_in_its_own_polarity(self) -> None:
        """Enumerate the producer's rule, not the table: a spelling missing from the table would go untested."""
        assert {v.lower() for v in _PRODUCER_READS_AS_TRUE if isinstance(v, str)} == _PYDANTIC_TRUE_STRINGS
        assert {v.lower() for v in _PRODUCER_READS_AS_FALSE if isinstance(v, str)} == _PYDANTIC_FALSE_STRINGS

    def test_it_agrees_with_pydantics_own_lax_bool_on_every_casing_and_padding(self) -> None:
        """The other direction, and more of it: never read a stance into a value pydantic rejects, nor miss one it accepts.

        Differential against pydantic's own lax ``bool | None`` -- the producer's rule by
        construction -- over every casing of every spelling, each spelling padded with each
        whitespace character on either side, and every printable ASCII character. The table
        above is the producer's measured answer. This is the breadth check it cannot be.
        """
        from pydantic import TypeAdapter, ValidationError

        adapter = TypeAdapter(Optional[bool])
        values: List[Any] = []
        for word in sorted(_PYDANTIC_TRUE_STRINGS | _PYDANTIC_FALSE_STRINGS):
            values += ["".join(cased) for cased in itertools.product(*[(ch.lower(), ch.upper()) for ch in word])]
            values += [f"{pad}{word}" for pad in (" ", "\t", "\n", "\r", "\x0b", "\x0c", " ")]
            values += [f"{word}{pad}" for pad in (" ", "\t", "\n", "\r", "\x0b", "\x0c", " ")]
        values += [chr(code) for code in range(32, 127)]
        mismatches = []
        for value in values:
            try:
                expected = adapter.validate_python(value)
            except ValidationError:
                expected = None
            if TrainingLifecycleManager._as_bool_stance(value) is not expected:
                mismatches.append((value, expected))
        assert mismatches == [], f"{len(mismatches)} of {len(values)} disagree with pydantic: {mismatches[:10]}"


class TestShortfallIsPollable:
    """A log line is not a surface. Canopy has to be able to READ the shortfall."""

    def test_a_clean_dataset_annotates_nothing(self) -> None:
        """None, not a dict of empties -- a consumer branches on presence alone."""
        assert TrainingLifecycleManager._build_dataset_shortfall({}, dataset_id="d1", acceptance_source=None) is None

    def test_the_annotation_names_the_dataset_it_describes(self) -> None:
        """An annotation that does not identify its artifact is a claim about nothing.

        cascor issues its OWN create_dataset, and the deployment default can
        change the params -- so its content-addressed id need not equal the
        driver's. Recording the id beside the annotation is what stops the two
        being silently attributed to each other.
        """
        meta = {"truncation": {"unit": "symbols", "cap": 14, "requested": 503, "imported": 14}}
        built = TrainingLifecycleManager._build_dataset_shortfall(meta, dataset_id="abc123", acceptance_source=_PROJECT_API_SHORTFALL_ACCEPTED_BY_DEPLOYMENT)
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
        built = TrainingLifecycleManager._build_dataset_shortfall(meta, dataset_id="d2", acceptance_source=_PROJECT_API_SHORTFALL_ACCEPTED_BY_DEPLOYMENT)
        assert built is not None
        assert built["data_quality"]["unrescued"] == {"STZ": "no shares concept"}
        assert built["data_quality"]["degraded"] == {"META": "period_average"}
        assert "unresolvable" in built["summary"] and "weaker source" in built["summary"]

    _PARTIAL_META = {"truncation": {"unit": "symbols", "cap": 14, "requested": 503, "imported": 14}}

    @staticmethod
    def _annotation_after_reload(caller_params: dict, *, deployment_flag: bool, meta: dict) -> dict:
        """Run ``_reload_dataset`` to completion and return the annotation it bound.

        The fake client delivers ``meta`` and a real three-partition artifact, and
        the reload runs to the end. It used to stop at tensor conversion, because
        the annotation was written BEFORE it -- which was itself the defect: a
        status poll during conversion saw a shortfall for data never loaded.
        Since APD-CASCOR-013 the annotation is set only when the data is bound, so
        it can only be observed after a completed reload.
        """

        class _PartialClient:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def list_generators(self) -> list:
                return _GENERATOR_LISTING

            def create_dataset(self, *, generator: str, params: dict, persist: bool) -> dict:
                return {"dataset_id": "partial-1", "meta": meta}

            def download_artifact_npz(self, dataset_id: str) -> dict:
                rng = np.random.default_rng(20260923)
                return {key: rng.standard_normal((rows, 2)).astype("float32") for key, rows in (("X_train", 20), ("y_train", 20), ("X_val", 6), ("y_val", 6), ("X_test", 4), ("y_test", 4))}

        manager = TrainingLifecycleManager.__new__(TrainingLifecycleManager)
        manager.logger = logging.getLogger("test.annotation")
        manager._dataset_shortfall = None
        settings = SimpleNamespace(juniper_data_url="http://juniper-data:8100", allow_truncated_datasets=deployment_flag)
        with (
            patch("juniper_data_client.JuniperDataClient", _PartialClient),
            patch("api.settings.Settings", lambda: settings),
            patch("api.secrets.get_secret", lambda _name: "key"),
        ):
            manager._reload_dataset(dataset_type="equities", params=dict(caller_params))
        assert manager._train_x is not None, "the reload must have bound the data the annotation describes"
        assert manager._dataset_shortfall is not None
        return manager._dataset_shortfall

    def test_a_caller_opt_in_is_recorded_as_the_request(self) -> None:
        """Options 1 and 2 of the partial-data contract arrive as request params, with the service flag off."""
        built = self._annotation_after_reload({"allow_truncation": True}, deployment_flag=False, meta=self._PARTIAL_META)
        assert built["accepted_by_this_run"] is True
        assert built["acceptance_source"] == _PROJECT_API_SHORTFALL_ACCEPTED_BY_REQUEST
        # The original field means exactly what its name says: the SETTING did not supply this opt-in.
        assert built["accepted_via_allow_truncated_datasets"] is False
        assert "dataset request itself" in built["summary"]

    def test_the_service_setting_is_recorded_as_the_deployment(self) -> None:
        built = self._annotation_after_reload({}, deployment_flag=True, meta=self._PARTIAL_META)
        assert built["accepted_by_this_run"] is True
        assert built["acceptance_source"] == _PROJECT_API_SHORTFALL_ACCEPTED_BY_DEPLOYMENT
        assert built["accepted_via_allow_truncated_datasets"] is True
        assert "allow_truncated_datasets setting" in built["summary"]

    def test_a_partial_dataset_nobody_here_asked_for_names_the_producer(self) -> None:
        """THE REGRESSION (round-37 handoff §0.13).

        Flag off, caller SILENT (not refusing -- since juniper-data APD-DATA-052 an
        explicit ``false`` would refuse, and this test deliberately sends neither), and
        the producer delivered a partial dataset anyway on its own deployment opt-in.
        The annotation
        used to read ``accepted_via_allow_truncated_datasets: false``: the truth
        about the setting, and a denial of the acceptance it was annotating. It now
        says who accepted, and that this run did not.
        """
        built = self._annotation_after_reload({}, deployment_flag=False, meta=self._PARTIAL_META)
        assert built["accepted_by_this_run"] is False
        assert built["acceptance_source"] == _PROJECT_API_SHORTFALL_ACCEPTED_BY_PRODUCER
        assert built["accepted_via_allow_truncated_datasets"] is False
        assert "producer" in built["summary"]

    def test_a_string_stance_is_read_as_a_bool(self) -> None:
        """The staged params cross a JSON boundary; ``bool("false")`` is ``True``."""
        assert TrainingLifecycleManager._as_bool_stance("false") is False
        assert TrainingLifecycleManager._as_bool_stance("True") is True
        assert TrainingLifecycleManager._as_bool_stance(False) is False
        assert TrainingLifecycleManager._as_bool_stance(None) is None
        assert TrainingLifecycleManager._as_bool_stance("") is None

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
        manager._auto_start_failure = None
        # ``current_dataset`` reads these two (nothing loaded -> None).
        manager._train_x = None
        manager._current_dataset_config = None

        manager._dataset_shortfall = None
        assert manager.get_status()["dataset_shortfall"] is None

        manager._dataset_shortfall = {"dataset_id": "d3", "summary": "14 of 503 symbols imported (cap 14)"}
        assert manager.get_status()["dataset_shortfall"]["dataset_id"] == "d3"

    def test_get_metrics_carries_it_too(self) -> None:
        """`/v1/metrics` is where the numbers are, so it is where the caveat has to be.

        A consumer reading the metrics route alone -- canopy's metric panels do --
        otherwise gets an accuracy with no mark of the data behind it and no
        reason to go looking for one. The partial-data contract requires the
        "accept" and "drop" options to annotate progress, metrics AND results.

        One field read from two surfaces, so status and metrics cannot disagree.
        ``get_metrics_history`` deliberately does NOT carry it: its rows are
        per-epoch samples, and the shortfall is a property of the run's dataset,
        not of any epoch in it -- stamping it on every row would imply it could
        vary between them.
        """
        manager = TrainingLifecycleManager()
        manager.create_network(input_size=2, output_size=2)
        manager.network.history = {"train_loss": [0.5], "train_accuracy": [0.6], "value_loss": [0.55], "value_accuracy": [0.55]}
        manager.network.hidden_units = []

        assert manager.get_metrics()["dataset_shortfall"] is None

        manager._dataset_shortfall = {"dataset_id": "d4", "summary": "14 of 503 symbols imported (cap 14)"}
        assert manager.get_metrics()["dataset_shortfall"]["dataset_id"] == "d4"
