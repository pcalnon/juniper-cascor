"""Auto-start's half of the partial-data contract: forward, annotate, and admit failure.

Project:       Juniper
Sub-Project:   JuniperCascor
Application:   juniper_cascor
File Name:     test_auto_start_shortfall.py
Author:        Paul Calnon
License:       MIT License

``_reload_dataset`` (the staged / live-swap route) has carried the truncation
stance and the ``dataset_shortfall`` annotation since cascor#624 / cascor#633.
``_auto_start_training`` -- the other live dataset-fetch path -- carried neither:
it sent no opt-in on a deployment whose flag was on, and set no annotation
afterwards, so an auto-started run training on a partial dataset reported
``dataset_shortfall: null``. It also swallowed every failure into a log line
nothing could poll, leaving a service that came up green with no training and no
queryable reason.

These are the round-38 follow-ups filed in cascor#624's PR body, confirmed
against source on 2026-09-08 and fixed 2026-09-09. The two paths now share one
stance resolver (``TrainingLifecycleManager._resolve_truncation_stance``), so
the arms below are also the guard that they cannot drift apart again.

The cascor#678 follow-ups add three things here: the listing client auto-start
builds is pinned (its URL, key, timeout and retries -- item 6); a withheld opt-in's
refusal gives auto-start's own retry and no other path's (item 7); and the
shortfall reaches the training log only once ``start_training`` binds its data, so a
refused artifact leaves no line saying a run is training on it (item 8).

#688's validation adds the other fetch order -- a clean auto-start fetch after a
partial one reports ``null`` -- and replaces the refusal stand-ins with juniper-data's
own text.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from contextlib import ExitStack, redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from juniper_data_client import JuniperDataValidationError

from api.app import _auto_start_training
from api.lifecycle.manager import _TRUNCATABLE_GENERATORS, TrainingLifecycleManager
from cascor_constants.constants_api.constants_api_defaults import _PROJECT_API_SHORTFALL_ACCEPTED_BY_DEPLOYMENT, _PROJECT_API_SHORTFALL_ACCEPTED_BY_PRODUCER, _PROJECT_API_SHORTFALL_ACCEPTED_BY_REQUEST, _PROJECT_API_SHORTFALL_REFUSAL_TOKEN

pytestmark = pytest.mark.unit

_PARTIAL_META: Dict[str, Any] = {"truncation": {"unit": "symbols", "cap": 14, "requested": 503, "imported": 14}}

# juniper-data's own refusal, as juniper-data-client 0.5.0 raises it (captured from a
# juniper-data main server by juniper-ml
# ``util/ad-hoc/2026-09-24_cascor688_realjd_error_texts.py``). It replaces the stand-in
# "HTTP 422 allow_truncation", which no longer reads as a refusal: the check keys on
# juniper-data's remedy sentence since #688's validation, not on a field's name.
_REFUSAL_TEXT = "Validation error (422): The requested universe is 3 symbols, over the 2 symbols cap. Re-submit with allow_truncation=true (or set JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION=true) to import the first 2 symbols. The resulting dataset will be permanently annotated as truncated."


def _refusal() -> JuniperDataValidationError:
    """juniper-data's shortfall refusal, as auto-start's ``create_dataset`` receives it."""
    return JuniperDataValidationError(_REFUSAL_TEXT, status_code=422)


# juniper-data's ``GET /v1/generators``, reduced to what the stance resolver reads.
# Since APD-CASCOR-008 the truncatable set is DERIVED from this -- a generator is
# truncatable iff its param schema declares ``allow_truncation`` -- so the fake
# producer has to answer it; one that cannot is an unreadable list, on which the
# deployment default is withheld (ruled 2026-09-22).
_GENERATOR_LISTING: List[Dict[str, Any]] = [
    {"name": "equities", "schema": {"properties": {"allow_truncation": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}}}},
    {"name": "spiral", "schema": {"properties": {"n_spirals": {"type": "integer"}}}},
]


@pytest.fixture(autouse=True)
def _fresh_truncatable_memo():
    """The derived set is memoised per process; one test's success must not serve the next."""
    _TRUNCATABLE_GENERATORS.reset()
    yield
    _TRUNCATABLE_GENERATORS.reset()


class _StopAfterAnnotation(Exception):
    """Raised in place of tensor conversion: everything after the annotation is built is plumbing."""


def _lifecycle() -> TrainingLifecycleManager:
    """A manager carrying exactly the state this path touches, and nothing else.

    ``__new__`` rather than a full construction, matching
    ``TestCallerStanceIsNotOverridden._manager`` in
    ``test_allow_truncated_datasets.py``: the real ``__init__`` builds a state
    machine, a monitor and a metrics thread that have no bearing on whether the
    dataset request was correct.
    """
    manager = TrainingLifecycleManager.__new__(TrainingLifecycleManager)
    manager.logger = logging.getLogger("test.autostart")
    manager._dataset_shortfall = None
    manager._auto_start_failure = None
    return manager


def _artifact() -> Dict[str, Any]:
    """A three-partition artifact, with distinct row counts so a mis-bound split cannot pass."""
    rng = np.random.default_rng(20260909)
    return {
        "X_train": rng.standard_normal((20, 2)).astype("float32"),
        "y_train": rng.standard_normal((20, 2)).astype("float32"),
        "X_val": rng.standard_normal((6, 2)).astype("float32"),
        "y_val": rng.standard_normal((6, 2)).astype("float32"),
        "X_test": rng.standard_normal((4, 2)).astype("float32"),
        "y_test": rng.standard_normal((4, 2)).astype("float32"),
    }


async def _run_auto_start(
    caller_params: Dict[str, Any],
    *,
    deployment_flag: bool,
    generator: str = "equities",
    meta: Optional[Dict[str, Any]] = None,
    ready: bool = True,
    create_error: Optional[Exception] = None,
    arrays: Optional[Dict[str, Any]] = None,
    lifecycle: Optional[TrainingLifecycleManager] = None,
    listing: Any = None,
) -> Tuple[TrainingLifecycleManager, Dict[str, Any]]:
    """Drive ``_auto_start_training`` against a fake producer and report what it sent.

    ``arrays is None`` stops the run at tensor conversion, which is immediately
    after the annotation is BUILT -- the same device
    ``TestShortfallIsPollable._annotation_after_reload`` uses on the staged path.
    Since APD-CASCOR-013 an auto-start that stops there leaves NO annotation: it is
    handed to ``start_training`` with the run's tensors, and that start never
    happens. Pass a real artifact to exercise the whole sequence through
    ``start_training`` (see ``_annotation_handed_to_the_run``).

    ``listing`` is what the producer's ``GET /v1/generators`` returns --
    ``_GENERATOR_LISTING`` when omitted; an exception instance makes it raise.

    Returns ``(lifecycle, sent)``. ``sent`` carries the generator and the params
    that reached ``create_dataset``, because for half of these arms the request
    IS the assertion, plus ``listing_calls`` -- and, once the list is read,
    ``listing_client``: the constructor kwargs of the client that read it. The
    double used to ignore those kwargs, so nothing pinned the URL or the key the
    listing client is built with (cascor#678 follow-up, item 6).
    """
    sent: Dict[str, Any] = {"listing_calls": 0}
    manager = lifecycle if lifecycle is not None else _lifecycle()

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            self._kwargs = dict(kwargs)

        def wait_for_ready(self, timeout: Optional[float] = None) -> bool:
            return ready

        def list_generators(self) -> Any:
            sent["listing_calls"] += 1
            sent["listing_client"] = dict(self._kwargs)
            answer = _GENERATOR_LISTING if listing is None else listing
            if isinstance(answer, BaseException):
                raise answer
            return answer

        def create_dataset(self, *, generator: str, params: dict, persist: bool) -> dict:
            sent["generator"] = generator
            sent["params"] = dict(params)
            if create_error is not None:
                raise create_error
            return {"dataset_id": "auto-1", "meta": meta or {}}

        def download_artifact_npz(self, dataset_id: str) -> dict:
            sent["dataset_id"] = dataset_id
            return arrays if arrays is not None else {}

    app = SimpleNamespace(state=SimpleNamespace(lifecycle=manager))
    settings = SimpleNamespace(
        juniper_data_url="http://juniper-data:8100",
        auto_dataset=generator,
        auto_dataset_params=json.dumps(caller_params),
        auto_network=json.dumps({}),
        allow_truncated_datasets=deployment_flag,
    )

    with ExitStack() as stack:
        stack.enter_context(patch("juniper_data_client.JuniperDataClient", _FakeClient))
        stack.enter_context(patch("api.app.get_secret", lambda _name: "key"))
        if arrays is None:
            stack.enter_context(patch.object(TrainingLifecycleManager, "_artifact_to_tensors", side_effect=_StopAfterAnnotation("stop")))
        await _auto_start_training(app, settings)
    return manager, sent


async def _annotation_handed_to_the_run(caller_params: Dict[str, Any], *, deployment_flag: bool, meta: Dict[str, Any]) -> Tuple[TrainingLifecycleManager, Optional[Dict[str, Any]]]:
    """Run the WHOLE sequence and return the annotation auto-start hands to ``start_training``.

    APD-CASCOR-013 binds the annotation together with the tensors it describes,
    so auto-start no longer writes it onto the manager ahead of its own start
    (binding the tensors would replace it); it passes it in with the tensors it
    fetched. The annotation is therefore observed where it now travels: the start
    call.
    ``start_training`` and ``create_network`` are doubles because what they do
    with it is ``test_shortfall_lifecycle.py``'s subject, not this file's.
    """
    manager = _lifecycle()
    manager.create_network = MagicMock(return_value={"input_size": 2, "output_size": 2})
    manager.start_training = MagicMock(return_value={"status": "training_started"})
    await _run_auto_start(caller_params, deployment_flag=deployment_flag, meta=meta, arrays=_artifact(), lifecycle=manager)
    assert manager._auto_start_failure is None, manager._auto_start_failure
    manager.start_training.assert_called_once()
    return manager, manager.start_training.call_args.kwargs["dataset_shortfall"]


def _flag_help() -> str:
    """The ``--allow-truncated-datasets`` help text as an operator sees it.

    Read out of the real parser rather than restated here, and whitespace-folded
    because argparse rewraps to the terminal width -- an assertion on a raw
    substring would pass or fail on ``COLUMNS``.
    """
    import main

    buffer = io.StringIO()
    with patch.object(sys, "argv", ["main.py", "--help"]), redirect_stdout(buffer), pytest.raises(SystemExit):
        main.parse_args()
    return " ".join(buffer.getvalue().split())


class TestAutoStartForwardsTheStance:
    """The deployment default has to reach the producer -- and must not overrule the operator."""

    async def test_the_deployment_default_applies_when_auto_dataset_params_is_silent(self) -> None:
        """THE REGRESSION. A flag-on deployment auto-starting on equities used to send NOTHING.

        juniper-data then answered 422, ``_auto_start_training`` swallowed the
        exception, and the service came up healthy with no training -- while the
        very knob that would have allowed the fetch was already set.
        """
        _, sent = await _run_auto_start({}, deployment_flag=True)
        assert sent["params"]["allow_truncation"] is True

    async def test_an_explicit_false_in_auto_dataset_params_survives(self) -> None:
        """A DEFAULT, not an override -- the rule the staged path was fixed to obey in cascor#624.

        ``JUNIPER_CASCOR_AUTO_DATASET_PARAMS='{"allow_truncation": false}'`` is
        how an operator expresses the contract's third option ("fail the data
        load completely") on a deployment whose flag is on. Overriding it would
        make that option unreachable here exactly as it once was there.
        """
        _, sent = await _run_auto_start({"allow_truncation": False}, deployment_flag=True)
        assert sent["params"]["allow_truncation"] is False

    async def test_an_explicit_true_is_preserved(self) -> None:
        """The other polarity, so the rule is not merely 'False is special'."""
        _, sent = await _run_auto_start({"allow_truncation": True}, deployment_flag=False)
        assert sent["params"]["allow_truncation"] is True

    async def test_nothing_is_forwarded_when_the_flag_is_off(self) -> None:
        """Unset means unset: the producer must see no opt-in and refuse with 422."""
        _, sent = await _run_auto_start({}, deployment_flag=False)
        assert "allow_truncation" not in sent["params"]

    async def test_a_generator_that_cannot_be_partial_is_never_sent_the_flag(self) -> None:
        """``spiral`` synthesises its data and always delivers in full.

        Sending it a parameter it ignores would imply the knob does something
        there -- the same misreading the CLI warning exists to prevent.
        """
        _, sent = await _run_auto_start({}, deployment_flag=True, generator="spiral")
        assert sent["generator"] == "spiral"
        assert "allow_truncation" not in sent["params"]

    async def test_other_params_reach_the_producer_untouched(self) -> None:
        """Option 2 ("drop") is expressed with ``incomplete_rows``; auto-start must not strip it."""
        _, sent = await _run_auto_start({"allow_truncation": True, "incomplete_rows": "drop", "tickers": ["AAPL"]}, deployment_flag=False)
        assert sent["params"]["incomplete_rows"] == "drop"
        assert sent["params"]["tickers"] == ["AAPL"]

    async def test_the_truncatable_set_comes_from_the_producer(self) -> None:
        """APD-CASCOR-008. A producer that lists ``equities`` WITHOUT the field is not sent the flag.

        The deleted constant named ``equities`` unconditionally; only the
        producer's own schema may say whether the parameter is accepted.
        """
        listing = [{"name": "equities", "schema": {"properties": {"tickers": {"type": "array"}}}}]
        _, sent = await _run_auto_start({}, deployment_flag=True, listing=listing)
        assert "allow_truncation" not in sent["params"]

    async def test_an_unreadable_list_withholds_the_default(self) -> None:
        """RULED 2026-09-22: the list cannot be read, so the opt-in is WITHHELD -- not guessed."""
        _, sent = await _run_auto_start({}, deployment_flag=True, listing=ConnectionError("connection refused"))
        assert sent["listing_calls"] == 1
        assert "allow_truncation" not in sent["params"]

    async def test_a_refusal_after_a_withheld_default_says_to_retry(self) -> None:
        """The knob is ON; telling the operator to turn it on would be a remedy that changes nothing.

        And the retry is AUTO-START's (cascor#678 follow-up, item 7). It runs once
        and never again on its own, so it is told both operator retries -- stage
        the dataset and start, or restart -- and none of the other paths'. It used
        to get one string for all three, including "a failed start leaves its
        dataset staged", which is false here: auto-start stages nothing.
        """
        manager, _ = await _run_auto_start({}, deployment_flag=True, listing=ConnectionError("connection refused"), create_error=_refusal())
        failure = manager._auto_start_failure
        assert failure is not None
        assert failure.startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN)
        assert "WITHHELD" in failure
        assert "--allow-truncated-datasets" not in failure
        assert "stage the dataset and start training" in failure and "restart the service to run auto-start again" in failure
        assert "starting training again retries it" not in failure and "re-issue the swap" not in failure

    async def test_a_refusal_for_a_generator_the_list_does_not_declare_does_not_name_the_knob(self) -> None:
        """Flag ON, list READ, ``equities`` not declared, and the producer 422'd anyway.

        The opt-in was correctly not sent, so the old default remedy -- "set
        JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS=true" -- pointed at a knob that was
        already on. Since the cascor#678 follow-up (item 4) it is not dressed as a
        shortfall refusal at all: a generator whose schema has no such parameter
        cannot be short, so the token -- which opens canopy's partial-data prompt --
        would offer options that re-send a request that fails the same way. The text
        is a REAL refusal, so it is the NOT_TRUNCATABLE reason that makes it plain.
        """
        listing = [{"name": "equities", "schema": {"properties": {"tickers": {"type": "array"}}}}]
        manager, sent = await _run_auto_start({}, deployment_flag=True, listing=listing, create_error=_refusal())
        assert sent["listing_calls"] == 1
        assert manager._auto_start_failure == f"juniper-data fetch failed: {_REFUSAL_TEXT}"

    async def test_the_listing_client_carries_the_auto_start_url_and_key(self) -> None:
        """cascor#678 follow-up, item 6. Kills M22 (``api_key=None``) and M23 (a hard-coded URL).

        juniper-data exempts only ``/v1/health*`` and ``/metrics`` from auth, so
        ``GET /v1/generators`` needs the key wherever auth is configured. Without it
        every such auto-start's read fails and its opt-in is silently withheld; a
        hard-coded URL reads some other juniper-data's list. The code was right, but
        the double ignored its constructor, so neither mutant failed a test.
        """
        _, sent = await _run_auto_start({}, deployment_flag=True)
        assert sent["listing_calls"] == 1
        assert sent["listing_client"] == {"base_url": "http://juniper-data:8100", "api_key": "key", "timeout": 5, "retries": 0}

    async def test_with_the_flag_off_the_list_is_never_read(self) -> None:
        """Lazy: no default can apply, so there is nothing to look up.

        Not a claim that auto-start has no startup dependency -- it is itself a
        boot-time request that waits for the producer and runs once. The claim is
        narrower: the list adds no read, and no failure mode, where it cannot
        change the outcome.
        """
        _, sent = await _run_auto_start({}, deployment_flag=False)
        assert sent["listing_calls"] == 0

    @pytest.mark.parametrize("caller_value", [True, False])
    async def test_a_caller_stance_is_never_worth_a_fetch(self, caller_value: bool) -> None:
        """Flag ON, but ``JUNIPER_CASCOR_AUTO_DATASET_PARAMS`` decided: the set cannot change the outcome."""
        _, sent = await _run_auto_start({"allow_truncation": caller_value}, deployment_flag=True)
        assert sent["listing_calls"] == 0
        assert sent["params"]["allow_truncation"] is caller_value

    @pytest.mark.parametrize("caller_value", [True, False])
    async def test_an_unreadable_list_never_touches_the_operators_value(self, caller_value: bool) -> None:
        """Withholding drops only THIS service's default -- "a DEFAULT, never an OVERRIDE", both polarities."""
        _, sent = await _run_auto_start({"allow_truncation": caller_value}, deployment_flag=True, listing=ConnectionError("connection refused"))
        assert sent["params"]["allow_truncation"] is caller_value


class TestAutoStartAnnotatesTheRun:
    """An auto-started run on a partial dataset must be distinguishable from a clean one."""

    async def test_a_caller_opt_in_is_recorded_as_the_request(self) -> None:
        """The opt-in came from ``JUNIPER_CASCOR_AUTO_DATASET_PARAMS``, with the service flag off."""
        _, annotation = await _annotation_handed_to_the_run({"allow_truncation": True}, deployment_flag=False, meta=_PARTIAL_META)
        assert annotation is not None
        assert annotation["acceptance_source"] == _PROJECT_API_SHORTFALL_ACCEPTED_BY_REQUEST
        assert annotation["accepted_by_this_run"] is True
        assert annotation["accepted_via_allow_truncated_datasets"] is False

    async def test_the_service_setting_is_recorded_as_the_deployment(self) -> None:
        _, annotation = await _annotation_handed_to_the_run({}, deployment_flag=True, meta=_PARTIAL_META)
        assert annotation is not None
        assert annotation["acceptance_source"] == _PROJECT_API_SHORTFALL_ACCEPTED_BY_DEPLOYMENT
        assert annotation["accepted_via_allow_truncated_datasets"] is True

    async def test_a_partial_dataset_nobody_here_asked_for_names_the_producer(self) -> None:
        """Flag off, params silent, and juniper-data delivered a partial dataset anyway.

        Its own deployment opt-in, which it applies to a request that sends no
        stance. (Before juniper-data APD-DATA-052 it ORed on top of an explicit
        ``false`` too, and a client could not refuse; now it does not.) The
        annotation must record that this run did not accept it -- not deny that
        anyone did.
        """
        _, annotation = await _annotation_handed_to_the_run({}, deployment_flag=False, meta=_PARTIAL_META)
        assert annotation is not None
        assert annotation["acceptance_source"] == _PROJECT_API_SHORTFALL_ACCEPTED_BY_PRODUCER
        assert annotation["accepted_by_this_run"] is False

    async def test_the_annotation_names_the_dataset_it_describes(self) -> None:
        """auto-start issues its OWN create_dataset, and the default above changes the params.

        So its content-addressed id need not equal anything the operator wrote
        down, and an annotation that does not identify its artifact is a claim
        about an unidentified one.
        """
        _, annotation = await _annotation_handed_to_the_run({}, deployment_flag=True, meta=_PARTIAL_META)
        assert annotation is not None
        assert annotation["dataset_id"] == "auto-1"
        assert "14" in annotation["summary"] and "503" in annotation["summary"]

    async def test_a_clean_dataset_annotates_nothing(self) -> None:
        """None, not a dict of empties -- a consumer branches on presence alone."""
        _, annotation = await _annotation_handed_to_the_run({}, deployment_flag=True, meta={})
        assert annotation is None

    async def test_a_failure_after_the_fetch_leaves_no_annotation(self) -> None:
        """APD-CASCOR-013. The sequence fails between the fetch and the start: there is no run.

        It used to write the annotation onto the manager before converting the
        artifact, so a refused or malformed artifact left ``dataset_shortfall``
        describing a dataset no run was training on, beside an
        ``auto_start_failure`` saying no run had started.
        """
        manager, _ = await _run_auto_start({}, deployment_flag=True, meta=_PARTIAL_META)
        assert manager._auto_start_failure is not None and "_StopAfterAnnotation" in manager._auto_start_failure
        assert manager._dataset_shortfall is None

    async def test_the_shortfall_also_reaches_the_training_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """The pollable annotation and the log line say the same thing, from one source -- once the data is bound.

        Through the REAL ``start_training``, which logs the annotation it is handed
        once it has bound the tensors and no staged dataset has replaced them
        (cascor#678 follow-up, item 8; #688's validation). auto-start used to log it
        itself, before converting the artifact.
        """
        manager = TrainingLifecycleManager()
        try:
            with caplog.at_level(logging.WARNING), patch.object(manager, "_run_training"):
                await _run_auto_start({}, deployment_flag=True, meta=_PARTIAL_META, arrays=_artifact(), lifecycle=manager)
                if manager._training_future is not None:
                    manager._training_future.result(timeout=10)
            assert manager._auto_start_failure is None, manager._auto_start_failure
            assert "DATASET SHORTFALL" in caplog.text
            assert "allow_truncated_datasets setting" in caplog.text
        finally:
            manager.shutdown()

    async def test_auto_starts_fetch_replaces_every_split_so_log_and_status_agree(self, caplog: pytest.LogCaptureFixture) -> None:
        """Auto-start IS a fetch, so ``start_training`` binds it wholesale (``as_fetch``), as it binds a staged one.

        The first fetch leaves all three splits loaded, clean. The second is partial
        and has no test split. Handed in as inline tensors, it kept the first
        fetch's test -- and with it the first fetch's record -- so the status said
        ``dataset_shortfall: null`` while the log said this run was on a partial
        dataset (#686's validation, ``probe_s9_log.py``). The test split this fetch
        lacks is cleared instead (cascor#582), and the record is this fetch's.
        """
        manager = TrainingLifecycleManager()
        try:
            train_val = {key: value for key, value in _artifact().items() if key not in ("X_test", "y_test")}
            with patch.object(manager, "_run_training"):
                await _run_auto_start({}, deployment_flag=False, meta={}, arrays=_artifact(), lifecycle=manager)
                assert manager._test_x is not None and manager.get_status()["dataset_shortfall"] is None
                with caplog.at_level(logging.WARNING):
                    await _run_auto_start({}, deployment_flag=False, meta=_PARTIAL_META, arrays=train_val, lifecycle=manager)
                if manager._training_future is not None:
                    manager._training_future.result(timeout=10)
            assert manager._auto_start_failure is None, manager._auto_start_failure
            assert manager._test_x is None, "the first fetch's test split is still in the run"
            annotation = manager.get_status()["dataset_shortfall"]
            assert annotation is not None and annotation["dataset_id"] == "auto-1"
            assert "DATASET SHORTFALL" in caplog.text
        finally:
            manager.shutdown()

    async def test_a_clean_auto_start_fetch_clears_the_annotation_a_partial_one_left(self) -> None:
        """#688's validation (mutant NV3): the other order -- a PARTIAL fetch first, then a CLEAN one.

        A fetch is bound wholesale, annotation included, so a clean fetch binds
        ``dataset_shortfall: null``. A bind that wrote the annotation only when it
        had one left the partial fetch's shortfall describing the clean fetch's
        data, and nothing tested that order.
        """
        manager = TrainingLifecycleManager()
        try:
            with patch.object(manager, "_run_training"):
                await _run_auto_start({}, deployment_flag=False, meta=_PARTIAL_META, arrays=_artifact(), lifecycle=manager)
                if manager._training_future is not None:
                    manager._training_future.result(timeout=10)
                assert manager.get_status()["dataset_shortfall"] is not None, "the partial fetch left no annotation -- the arm proves nothing"
                await _run_auto_start({}, deployment_flag=False, meta={}, arrays=_artifact(), lifecycle=manager)
                if manager._training_future is not None:
                    manager._training_future.result(timeout=10)
            assert manager._auto_start_failure is None, manager._auto_start_failure
            assert manager.get_status()["dataset_shortfall"] is None
            assert manager.get_metrics()["dataset_shortfall"] is None
        finally:
            manager.shutdown()

    async def test_a_refused_artifact_leaves_no_shortfall_in_the_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """Item 8. The producer answered with a partial dataset, and section 6.1 then refused its artifact.

        auto-start logged "this run is training on a partial dataset" before it
        converted the artifact, so the line stayed in the log for data that was
        never loaded, beside an ``auto_start_failure`` saying no run had started.
        """
        train_only = {key: value for key, value in _artifact().items() if key in ("X_train", "y_train")}
        with caplog.at_level(logging.WARNING):
            manager, _ = await _run_auto_start({}, deployment_flag=True, meta=_PARTIAL_META, arrays=train_only)
        assert manager._auto_start_failure is not None and "NEITHER a validation split" in manager._auto_start_failure
        assert "DATASET SHORTFALL" not in caplog.text
        assert "DATASET IS PARTIAL" not in caplog.text

    async def test_a_full_run_annotates_and_still_trains(self) -> None:
        """The annotation must not stand between the fetch and the training it annotates."""
        manager, annotation = await _annotation_handed_to_the_run({}, deployment_flag=True, meta=_PARTIAL_META)
        assert annotation is not None
        assert manager._auto_start_failure is None
        # The tensors and their annotation travel in ONE call, so they cannot be split.
        kwargs = manager.start_training.call_args.kwargs
        assert kwargs["X"] is not None and kwargs["X"].shape[0] == 20

    async def test_the_annotation_survives_a_real_start(self) -> None:
        """End to end through the REAL ``start_training``, which binds it together with the tensors.

        The failure mode this pins: writing the annotation onto the manager and
        then starting the run on inline tensors -- which is what auto-start did --
        is replaced the moment the start binds those tensors. Only the fit is
        replaced.
        """
        manager = TrainingLifecycleManager()
        try:
            with patch.object(manager, "_run_training"):
                await _run_auto_start({}, deployment_flag=True, meta=_PARTIAL_META, arrays=_artifact(), lifecycle=manager)
                if manager._training_future is not None:
                    manager._training_future.result(timeout=10)
            assert manager._auto_start_failure is None, manager._auto_start_failure
            status = manager.get_status()["dataset_shortfall"]
            assert status is not None and status["dataset_id"] == "auto-1"
        finally:
            manager.shutdown()


class TestAutoStartFailureIsQueryable:
    """A swallowed failure that leaves no trace is indistinguishable from an idle service."""

    async def test_a_shortfall_refusal_is_recorded_with_its_remedy(self) -> None:
        """The 422 case: logged at ERROR, recorded on the run, and still swallowed."""
        manager, _ = await _run_auto_start({}, deployment_flag=False, create_error=_refusal())
        assert manager._auto_start_failure is not None
        assert manager._auto_start_failure.startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN)
        assert "--allow-truncated-datasets" in manager._auto_start_failure
        # Nothing was fetched, so nothing may be annotated.
        assert manager._dataset_shortfall is None

    async def test_a_caller_that_refused_is_told_to_re_send_not_to_flip_the_setting(self) -> None:
        """The message keys off the WIRE stance, not the setting -- as it does on the staged path.

        Flag ON and ``auto_dataset_params`` sending ``allow_truncation: false``:
        auto-start correctly withholds its default, so pointing the operator at
        the service knob would send them to one their own value overrides.
        """
        manager, sent = await _run_auto_start({"allow_truncation": False}, deployment_flag=True, create_error=_refusal())
        assert sent["params"]["allow_truncation"] is False
        assert manager._auto_start_failure is not None
        assert "explicitly refused" in manager._auto_start_failure
        assert "--allow-truncated-datasets" not in manager._auto_start_failure

    async def test_an_ordinary_outage_is_not_dressed_up_as_a_shortfall(self) -> None:
        """A dead producer must not tell the operator to set a truncation flag."""
        manager, _ = await _run_auto_start({}, deployment_flag=False, create_error=ConnectionError("connection refused"))
        assert manager._auto_start_failure == "juniper-data fetch failed: connection refused"

    async def test_a_producer_that_never_becomes_ready_is_recorded(self) -> None:
        """The other early return. It used to log and vanish."""
        manager, sent = await _run_auto_start({}, deployment_flag=False, ready=False)
        assert sent == {"listing_calls": 0}, "nothing may be requested from a producer that never became ready"
        assert manager._auto_start_failure is not None
        assert "not ready" in manager._auto_start_failure

    async def test_any_other_failure_is_recorded_too(self) -> None:
        """The catch-all handler. ``_StopAfterAnnotation`` stands in for anything later in the sequence.

        The exception TYPE is part of the record because the message alone is
        frequently useless -- a bare ``KeyError('meta')`` renders as ``'meta'``.
        """
        manager, _ = await _run_auto_start({}, deployment_flag=False, meta={})
        assert manager._auto_start_failure is not None
        assert "_StopAfterAnnotation" in manager._auto_start_failure

    async def test_a_failure_never_escapes_the_task(self) -> None:
        """Still swallowed: the service must come up healthy whether or not the demo run starts.

        Reaching the assertion at all is the assertion -- an exception out of
        ``_auto_start_training`` would fail this test at the await.
        """
        manager, _ = await _run_auto_start({}, deployment_flag=False, create_error=ConnectionError("boom"))
        assert manager._auto_start_failure is not None

    async def test_get_status_publishes_it(self) -> None:
        """The whole point of recording it: an operator can poll for the reason."""
        manager = _lifecycle()
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
        # ``current_dataset`` reads these two (nothing loaded -> None).
        manager._train_x = None
        manager._current_dataset_config = None

        assert manager.get_status()["auto_start_failure"] is None
        manager._auto_start_failure = "Auto-start failed: JuniperData not ready"
        assert manager.get_status()["auto_start_failure"] == "Auto-start failed: JuniperData not ready"


class TestTheCliFlagSaysItIsInert:
    """``--allow-truncated-datasets`` on ``main.py`` configures a SERVICE, not this run."""

    def test_the_warning_fires_when_the_flag_is_passed(self) -> None:
        """An accepted flag that silently does nothing is worse than a rejected one.

        The operator concludes the shortfall was allowed; nothing ever asked.
        """
        import main

        with patch.dict(os.environ, {}, clear=False), patch.object(main.Logger, "warning") as warn:
            os.environ.pop("JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS", None)
            assert main.apply_allow_truncated_datasets(True) is True
            assert os.environ["JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS"] == "true"
        warn.assert_called_once()
        message = warn.call_args.args[0]
        assert "NO EFFECT" in message
        # The warning must name the REASON, not just the verdict. `spiral` is what makes
        # the flag inert here -- juniper-data always delivers it in full and it is not in
        # the truncatable set. An earlier version said the data was generated locally and
        # that main.py asks juniper-data for nothing, which is false: main.py health-checks
        # the service and refuses to start without it.
        assert "spiral" in message
        assert "in full" in message
        assert "locally" not in message

    def test_nothing_happens_when_the_flag_is_absent(self) -> None:
        """Only ever SET, never cleared: omitting it leaves an operator's env choice standing."""
        import main

        with patch.dict(os.environ, {"JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS": "true"}), patch.object(main.Logger, "warning") as warn:
            assert main.apply_allow_truncated_datasets(False) is False
            assert os.environ["JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS"] == "true"
        warn.assert_not_called()

    def test_the_help_text_admits_the_flag_is_inert_here(self) -> None:
        """An operator must not have to run it to discover it did nothing.

        The flag is deliberately NOT removed -- exporting the variable is how it
        reaches a service this process launches -- so the ``--help`` text is the
        only place that can say where it does and does not apply.
        """
        help_text = _flag_help()
        assert "NO EFFECT ON THIS ENTRY POINT'S OWN RUN" in help_text
        assert "src/server.py" in help_text
