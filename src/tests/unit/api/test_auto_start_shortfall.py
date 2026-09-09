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
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from contextlib import ExitStack, redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.app import _auto_start_training
from api.lifecycle.manager import TrainingLifecycleManager
from cascor_constants.constants_api.constants_api_defaults import _PROJECT_API_SHORTFALL_ACCEPTED_BY_DEPLOYMENT, _PROJECT_API_SHORTFALL_ACCEPTED_BY_PRODUCER, _PROJECT_API_SHORTFALL_ACCEPTED_BY_REQUEST, _PROJECT_API_SHORTFALL_REFUSAL_TOKEN

pytestmark = pytest.mark.unit

_PARTIAL_META: Dict[str, Any] = {"truncation": {"unit": "symbols", "cap": 14, "requested": 503, "imported": 14}}


class _StopAfterAnnotation(Exception):
    """Raised in place of tensor conversion: everything after the annotation is plumbing."""


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
) -> Tuple[TrainingLifecycleManager, Dict[str, Any]]:
    """Drive ``_auto_start_training`` against a fake producer and report what it sent.

    ``arrays is None`` stops the run at tensor conversion, which is immediately
    after the annotation is written -- the same device
    ``TestShortfallIsPollable._annotation_after_reload`` uses on the staged path.
    Pass a real artifact to exercise the whole sequence through ``start_training``.

    Returns ``(lifecycle, sent)``. ``sent`` carries the generator and the params
    that reached ``create_dataset``, because for half of these arms the request
    IS the assertion.
    """
    sent: Dict[str, Any] = {}
    manager = lifecycle if lifecycle is not None else _lifecycle()

    class _FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def wait_for_ready(self, timeout: Optional[float] = None) -> bool:
            return ready

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


class TestAutoStartAnnotatesTheRun:
    """An auto-started run on a partial dataset must be distinguishable from a clean one."""

    async def test_a_caller_opt_in_is_recorded_as_the_request(self) -> None:
        """The opt-in came from ``JUNIPER_CASCOR_AUTO_DATASET_PARAMS``, with the service flag off."""
        manager, _ = await _run_auto_start({"allow_truncation": True}, deployment_flag=False, meta=_PARTIAL_META)
        assert manager._dataset_shortfall is not None
        assert manager._dataset_shortfall["acceptance_source"] == _PROJECT_API_SHORTFALL_ACCEPTED_BY_REQUEST
        assert manager._dataset_shortfall["accepted_by_this_run"] is True
        assert manager._dataset_shortfall["accepted_via_allow_truncated_datasets"] is False

    async def test_the_service_setting_is_recorded_as_the_deployment(self) -> None:
        manager, _ = await _run_auto_start({}, deployment_flag=True, meta=_PARTIAL_META)
        assert manager._dataset_shortfall is not None
        assert manager._dataset_shortfall["acceptance_source"] == _PROJECT_API_SHORTFALL_ACCEPTED_BY_DEPLOYMENT
        assert manager._dataset_shortfall["accepted_via_allow_truncated_datasets"] is True

    async def test_a_partial_dataset_nobody_here_asked_for_names_the_producer(self) -> None:
        """Flag off, params silent, and juniper-data delivered a partial dataset anyway.

        Its own deployment opt-in, which it ORs with the request and a client
        cannot refuse. The annotation must record that this run did not accept
        it -- not deny that anyone did.
        """
        manager, _ = await _run_auto_start({}, deployment_flag=False, meta=_PARTIAL_META)
        assert manager._dataset_shortfall is not None
        assert manager._dataset_shortfall["acceptance_source"] == _PROJECT_API_SHORTFALL_ACCEPTED_BY_PRODUCER
        assert manager._dataset_shortfall["accepted_by_this_run"] is False

    async def test_the_annotation_names_the_dataset_it_describes(self) -> None:
        """auto-start issues its OWN create_dataset, and the default above changes the params.

        So its content-addressed id need not equal anything the operator wrote
        down, and an annotation that does not identify its artifact is a claim
        about an unidentified one.
        """
        manager, _ = await _run_auto_start({}, deployment_flag=True, meta=_PARTIAL_META)
        assert manager._dataset_shortfall is not None
        assert manager._dataset_shortfall["dataset_id"] == "auto-1"
        assert "14" in manager._dataset_shortfall["summary"] and "503" in manager._dataset_shortfall["summary"]

    async def test_a_clean_dataset_annotates_nothing(self) -> None:
        """None, not a dict of empties -- a consumer branches on presence alone."""
        manager, _ = await _run_auto_start({}, deployment_flag=True, meta={})
        assert manager._dataset_shortfall is None

    async def test_the_shortfall_also_reaches_the_training_log(self) -> None:
        """The pollable annotation and the log line say the same thing, from one source."""
        manager = _lifecycle()
        manager.logger = MagicMock()
        await _run_auto_start({}, deployment_flag=True, meta=_PARTIAL_META, lifecycle=manager)
        emitted = " ".join(str(arg) for call in manager.logger.warning.call_args_list for arg in call.args)
        assert "DATASET SHORTFALL" in emitted
        assert "allow_truncated_datasets setting" in emitted

    async def test_a_full_run_annotates_and_still_trains(self) -> None:
        """The annotation must not stand between the fetch and the training it annotates."""
        manager = _lifecycle()
        manager.create_network = MagicMock(return_value={"input_size": 2, "output_size": 2})
        manager.start_training = MagicMock(return_value={"status": "training_started"})
        await _run_auto_start({}, deployment_flag=True, meta=_PARTIAL_META, arrays=_artifact(), lifecycle=manager)
        assert manager._dataset_shortfall is not None
        assert manager._auto_start_failure is None
        manager.start_training.assert_called_once()


class TestAutoStartFailureIsQueryable:
    """A swallowed failure that leaves no trace is indistinguishable from an idle service."""

    async def test_a_shortfall_refusal_is_recorded_with_its_remedy(self) -> None:
        """The 422 case: logged at ERROR, recorded on the run, and still swallowed."""
        manager, _ = await _run_auto_start(
            {},
            deployment_flag=False,
            create_error=RuntimeError("HTTP 422: shares outstanding could not be resolved. Re-submit with allow_truncation=true"),
        )
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
        manager, sent = await _run_auto_start(
            {"allow_truncation": False},
            deployment_flag=True,
            create_error=RuntimeError("HTTP 422 allow_truncation"),
        )
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
        assert sent == {}, "nothing may be requested from a producer that never became ready"
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
        assert "two-spiral" in message

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
