"""APD-CASCOR-008: the truncatable-generator set is read from juniper-data, not restated here.

Project:       Juniper
Sub-Project:   JuniperCascor
Application:   juniper_cascor
File Name:     test_truncatable_generators.py
Author:        Paul Calnon
License:       MIT License

Two owner rulings (juniper-ml
``notes/JUNIPER_2026-08-14_JUNIPER-ECOSYSTEM_DEFECT-REGISTER.md`` section 4.9):

* **2026-09-09** (juniper-ml#1864) -- derive the set from juniper-data's
  ``GET /v1/generators``: one source of truth, no drift. Rejected: widening
  cascor's ``dataset_type`` Literal, and narrowing the old constant to its one
  reachable member.
* **2026-09-22** -- when the list cannot be read: **withhold the opt-in, and
  retry.** Fetch lazily -- only when the resolver consults the set -- memoise only
  a success, and on a failure send no deployment opt-in for that request so
  juniper-data's own default governs. Rejected: refusing to start, a built-in
  fallback copy of the set, and a last-known set cached on disk.

Implementation constraints pinned below, each re-derived in source: the list is
read only inside the resolver's default branch (flag on, caller silent); the memo
is keyed by the juniper-data URL; the listing client is bounded (short timeout, no
retries) because the staged path holds the manager lock; a withheld opt-in gets
its own remedy on both paths; and a caller's own value passes through untouched
in either polarity while the list is unreadable.

The constant this replaces, ``_PROJECT_API_TRUNCATABLE_GENERATORS``, restated
knowledge juniper-data owns; a generator that gained an input bound had to be
added to it by hand, or every cascor run's shortfall on it was refused with no
way to opt in.

The cascor#678 follow-ups pinned here, from its post-merge validation: a listing in
which ANY entry lacks a schema is a failed read, not a smaller set cached for life
(item 5); a caller's ``allow_truncation: null`` is recorded as deferring whatever the
flag, so its refusal never names a knob that cannot help (item 3); a 422 for a
generator the list does not declare is a plain fetch failure, not a shortfall
refusal (item 4); and a withheld opt-in's refusal gives the retry of the path it came
from and no other (item 7).

From #688's validation, against a real juniper-data: a blank ``allow_truncation`` is
not a deferral (juniper-data answers it 400), and a 400 that names a truncation field
is a plain fetch failure on the live path, in both flag positions. The refusals and
the 400s these arms use are juniper-data's own texts.
"""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
from juniper_data_client import JuniperDataValidationError
from pydantic import BaseModel

import cascor_constants.constants_api as constants_api
from api.lifecycle.manager import _FETCH_PATH_AUTO_START, _FETCH_PATH_LIVE_SWAP, _FETCH_PATH_STAGED_START, _OPT_IN_SKIPPED_CALLER_DEFERRED, _OPT_IN_SKIPPED_LIST_UNREADABLE, _OPT_IN_SKIPPED_NOT_TRUNCATABLE, _TRUNCATABLE_GENERATORS, TrainingLifecycleManager, _TruncatableGenerators
from api.models.training import StageDatasetRequest, SwapDatasetLiveRequest
from cascor_constants.constants_api.constants_api_defaults import _PROJECT_API_SHORTFALL_ACCEPTED_BY_DEPLOYMENT, _PROJECT_API_SHORTFALL_ACCEPTED_BY_REQUEST, _PROJECT_API_SHORTFALL_REFUSAL_TOKEN

pytestmark = pytest.mark.unit


def _entry(name: str, *, truncatable: bool) -> Dict[str, Any]:
    """One ``GET /v1/generators`` entry, in the shape juniper-data serialises (``schema`` by alias)."""
    properties: Dict[str, Any] = {"seed": {"type": "integer"}}
    if truncatable:
        properties["allow_truncation"] = {"anyOf": [{"type": "boolean"}, {"type": "null"}], "default": None}
    return {"name": name, "version": "3.0.0", "description": name, "available": True, "install_hint": None, "schema": {"type": "object", "properties": properties}}


# Mirrors juniper-data's live registry as of 2026-09-22: exactly these three
# declare ``allow_truncation`` (measured by generating every params class's JSON
# schema, and again through the real app's ``GET /v1/generators``).
LISTING: List[Dict[str, Any]] = [
    _entry("spiral", truncatable=False),
    _entry("xor", truncatable=False),
    _entry("csv_import", truncatable=True),
    _entry("equities", truncatable=True),
    _entry("equities_seq", truncatable=True),
    _entry("mnist", truncatable=False),
]

# juniper-data's OWN words, as juniper-data-client 0.5.0 raises them, captured from a
# juniper-data main server (0f0f7e0) by juniper-ml
# ``util/ad-hoc/2026-09-24_cascor688_realjd_error_texts.py``. The refusal is an
# ``InputTooLargeError``; the 400s are parameter errors that NAME a truncation field,
# which the refusal check matched on until #688's validation. These replace the
# stand-in "HTTP 422 allow_truncation": that string no longer reads as a refusal, so
# an arm built on it would pass without reaching the branch it is about.
_REFUSAL_TEXT = "Validation error (422): The requested universe is 3 symbols, over the 2 symbols cap. Re-submit with allow_truncation=true (or set JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION=true) to import the first 2 symbols. The resulting dataset will be permanently annotated as truncated."
_INCOMPLETE_ROWS_400 = "Validation error (400): Invalid parameters: 1 validation error for EquitiesParams\nincomplete_rows\n  Input should be 'accept' or 'drop' [type=literal_error, input_value='keep', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/literal_error"
_BLANK_ALLOW_TRUNCATION_400 = "Validation error (400): Invalid parameters: 1 validation error for CsvImportParams\nallow_truncation\n  Input should be a valid boolean, unable to interpret input [type=bool_parsing, input_value='', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/bool_parsing"
_N_SPIRALS_400 = "Validation error (400): Invalid parameters: 1 validation error for SpiralParams\nn_spirals\n  Input should be greater than or equal to 2 [type=greater_than_equal, input_value=1, input_type=int]\n    For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal"


def _refusal() -> JuniperDataValidationError:
    """juniper-data's shortfall refusal, as the live path receives it."""
    return JuniperDataValidationError(_REFUSAL_TEXT, status_code=422)


class _ListingClient:
    """Answers ``list_generators()`` from a script of results, counting the calls."""

    def __init__(self, *results: Any) -> None:
        self._results = list(results)
        self.calls = 0

    def list_generators(self) -> Any:
        self.calls += 1
        result = self._results.pop(0) if len(self._results) > 1 else self._results[0]
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture(autouse=True)
def _fresh_memo():
    """The process-wide memo must not leak a success from one test into the next."""
    _TRUNCATABLE_GENERATORS.reset()
    yield
    _TRUNCATABLE_GENERATORS.reset()


class TestDerivation:
    """A generator is truncatable iff its listed param schema declares ``allow_truncation``."""

    def test_the_set_is_exactly_the_generators_that_declare_the_field(self) -> None:
        assert _TruncatableGenerators.derive(LISTING) == frozenset({"csv_import", "equities", "equities_seq"})

    def test_an_inherited_field_counts(self) -> None:
        """``equities_seq`` declares nothing of its own; it lists the field through ``EquitiesParams``.

        Built from a REAL pydantic hierarchy rather than a hand-written schema, so
        the arm proves what pydantic emits for an inherited field -- not what this
        file assumes it emits.
        """

        class _Parent(BaseModel):
            allow_truncation: Optional[bool] = None

        class _Child(_Parent):
            window: int = 5

        listing = [{"name": "child", "schema": _Child.model_json_schema()}]
        assert _TruncatableGenerators.derive(listing) == frozenset({"child"})

    def test_a_generator_without_the_field_is_excluded(self) -> None:
        """A synthetic generator that ignores the knob must not be sent it."""
        assert _TruncatableGenerators.derive([_entry("new_synthetic", truncatable=False)]) == frozenset()

    @pytest.mark.parametrize("schemaless", [{"name": "odd", "schema": None}, {"name": "odd"}], ids=["schema-null", "schema-absent"])
    def test_any_entry_without_a_usable_schema_fails_the_read(self, schemaless: Dict[str, Any]) -> None:
        """cascor#678 follow-up, item 5. Beside entries that DO carry a schema, a schema-less one is UNKNOWN.

        It used to be skipped, so the listing derived a smaller set -- and a set is
        memoised for the life of the process, so the skipped generator was never
        sent the opt-in again, even after its producer listed it properly. An entry
        that cannot say whether it accepts the parameter must not be read as saying
        it does not.
        """
        listing = [schemaless, _entry("spiral", truncatable=False), _entry("equities", truncatable=True)]
        with pytest.raises(ValueError, match="'odd' carries no parameter schema"):
            _TruncatableGenerators.derive(listing)

    @pytest.mark.parametrize("listing", [{"generators": LISTING}, [{"schema": {}}], ["spiral"], None])
    def test_a_malformed_listing_is_a_failed_read_not_an_empty_set(self, listing: Any) -> None:
        """An empty set would read as "nothing is truncatable" and be memoised for the process."""
        with pytest.raises(ValueError):
            _TruncatableGenerators.derive(listing)

    @pytest.mark.parametrize(
        "listing",
        [
            [{"name": "spiral", "parameters": ["n_spirals"]}, {"name": "equities", "parameters": ["tickers"]}],
            [{"name": "odd", "schema": None}, {"name": "odder"}],
            [],
        ],
        ids=["parameters-not-schema", "no-usable-schema", "empty"],
    )
    def test_a_listing_in_which_no_entry_carries_a_schema_is_a_failed_read(self, listing: Any) -> None:
        """It cannot say which generators accept the parameter, so it must not be read as saying none do.

        The first shape is ``juniper_data_client.testing``'s fake catalog, which
        lists ``parameters`` rather than a ``schema``: read as a success, it
        derived ``frozenset()``, was memoised, and sent no opt-in to ``equities``
        for the life of the process.
        """
        with pytest.raises(ValueError, match="parameter schema"):
            _TruncatableGenerators.derive(listing)


class TestLazyResolution:
    """Memoise a success; never a failure; retry on the next request."""

    def test_a_success_is_memoised(self) -> None:
        client = _ListingClient(LISTING)
        first = _TRUNCATABLE_GENERATORS.resolve(lambda: client, source="http://juniper-data:8100")
        second = _TRUNCATABLE_GENERATORS.resolve(lambda: client, source="http://juniper-data:8100")
        assert first == second == frozenset({"csv_import", "equities", "equities_seq"})
        assert client.calls == 1

    def test_a_failure_is_unknown_and_the_next_request_retries(self, caplog: pytest.LogCaptureFixture) -> None:
        """THE 2026-09-22 RULING: withhold now, read again next time."""
        client = _ListingClient(ConnectionError("connection refused"), LISTING)
        with caplog.at_level(logging.WARNING, logger="api.lifecycle.manager"):
            assert _TRUNCATABLE_GENERATORS.resolve(lambda: client, source="http://juniper-data:8100") is None
        assert "WITHHELD" in caplog.text and "ConnectionError" in caplog.text
        assert _TRUNCATABLE_GENERATORS.resolve(lambda: client, source="http://juniper-data:8100") == frozenset({"csv_import", "equities", "equities_seq"})
        assert client.calls == 2

    def test_a_malformed_payload_is_a_failure_too_and_is_not_memoised(self) -> None:
        client = _ListingClient({"detail": "not a list"}, LISTING)
        assert _TRUNCATABLE_GENERATORS.resolve(lambda: client, source="http://juniper-data:8100") is None
        assert _TRUNCATABLE_GENERATORS.resolve(lambda: client, source="http://juniper-data:8100") == frozenset({"csv_import", "equities", "equities_seq"})

    def test_a_schemaless_listing_is_not_memoised_as_an_empty_set(self) -> None:
        """The empty-set cache bug: a schema-less listing must stay UNKNOWN and be read again."""
        client = _ListingClient([{"name": "equities", "parameters": ["tickers"]}], LISTING)
        assert _TRUNCATABLE_GENERATORS.resolve(lambda: client, source="http://juniper-data:8100") is None
        assert _TRUNCATABLE_GENERATORS.resolve(lambda: client, source="http://juniper-data:8100") == frozenset({"csv_import", "equities", "equities_seq"})
        assert client.calls == 2

    def test_a_partly_schemad_listing_is_not_memoised_as_a_smaller_set(self) -> None:
        """Item 5, as #678's post-merge validation measured it: ``equities`` lost its schema, the rest kept theirs.

        The read derived ``{csv_import, equities_seq}``, memoised it, and still
        answered that after the producer recovered -- one listing call for the life
        of the process, and ``equities`` never sent the opt-in again.
        """
        partial = [dict(entry) for entry in LISTING]
        for entry in partial:
            if entry["name"] == "equities":
                del entry["schema"]
        client = _ListingClient(partial, LISTING)
        assert _TRUNCATABLE_GENERATORS.resolve(lambda: client, source="http://juniper-data:8100") is None
        assert _TRUNCATABLE_GENERATORS.resolve(lambda: client, source="http://juniper-data:8100") == frozenset({"csv_import", "equities", "equities_seq"})
        assert client.calls == 2

    def test_the_memo_is_keyed_by_the_juniper_data_it_was_read_from(self) -> None:
        a, b = _ListingClient(LISTING), _ListingClient([_entry("csv_import", truncatable=True)])
        assert _TRUNCATABLE_GENERATORS.resolve(lambda: a, source="http://one:8100") == frozenset({"csv_import", "equities", "equities_seq"})
        assert _TRUNCATABLE_GENERATORS.resolve(lambda: b, source="http://two:8100") == frozenset({"csv_import"})

    def test_reset_forgets(self) -> None:
        client = _ListingClient(LISTING)
        _TRUNCATABLE_GENERATORS.resolve(lambda: client, source="http://juniper-data:8100")
        _TRUNCATABLE_GENERATORS.reset()
        _TRUNCATABLE_GENERATORS.resolve(lambda: client, source="http://juniper-data:8100")
        assert client.calls == 2


class _Reader:
    """A ``truncatable_generators`` reader that records whether the resolver consulted it."""

    def __init__(self, answer: Optional[frozenset]) -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self) -> Optional[frozenset]:
        self.calls += 1
        return self.answer


class TestTheResolverConsultsTheDerivedSet:
    """``_resolve_truncation_stance`` with a reader passed in -- pure apart from it, so each arm is one call."""

    _SET = frozenset({"csv_import", "equities", "equities_seq"})

    def _resolve(self, params: Dict[str, Any], *, generator: str = "equities", allow_truncated: bool = True, answer: Optional[frozenset] = _SET) -> Tuple[tuple, int]:
        """The resolver's own 5-tuple, and how many times it consulted the reader -- kept apart.

        Returned as a PAIR rather than one flattened tuple: CodeQL's
        ``py/mismatched-multiple-assignment`` cannot see through a starred
        re-pack, and error-level alerts block this repo's merges.
        """
        reader = _Reader(answer)
        result = TrainingLifecycleManager._resolve_truncation_stance(params, generator=generator, allow_truncated=allow_truncated, truncatable_generators=reader)
        return result, reader.calls

    def test_a_truncatable_generator_gets_the_deployment_default(self) -> None:
        (params, source, wire, refused, skipped), reads = self._resolve({}, generator="equities_seq")
        assert params == {"allow_truncation": True}
        assert (source, wire, refused, skipped, reads) == (_PROJECT_API_SHORTFALL_ACCEPTED_BY_DEPLOYMENT, True, False, None, 1)

    def test_a_generator_outside_the_set_is_not_sent_the_flag_and_says_why(self) -> None:
        """Not sent, and recorded as NOT_TRUNCATABLE so a later refusal is not told to turn on the knob."""
        (params, source, wire, _, skipped), _ = self._resolve({}, generator="spiral")
        assert params == {}
        assert (source, wire, skipped) == (None, False, _OPT_IN_SKIPPED_NOT_TRUNCATABLE)

    def test_an_unknown_set_withholds_the_default(self) -> None:
        """THE 2026-09-22 RULING. Nothing is sent, and the result says so."""
        (params, source, wire, refused, skipped), _ = self._resolve({}, answer=None)
        assert params == {}
        assert (source, wire, refused, skipped) == (None, False, False, _OPT_IN_SKIPPED_LIST_UNREADABLE)

    @pytest.mark.parametrize("caller_value", [True, False])
    def test_an_unknown_set_leaves_the_callers_own_value_alone(self, caller_value: bool) -> None:
        """CONSTRAINT 6 -- "a DEFAULT, never an OVERRIDE", in both polarities, with the list unreadable.

        Withholding drops only this service's default. The caller's value never
        needed the set, so the reader is not even consulted.
        """
        (params, source, wire, refused, skipped), reads = self._resolve({"allow_truncation": caller_value}, answer=None)
        assert params == {"allow_truncation": caller_value}
        assert (wire, refused, skipped, reads) == (caller_value, not caller_value, None, 0)
        assert source == (_PROJECT_API_SHORTFALL_ACCEPTED_BY_REQUEST if caller_value else None)

    def test_with_the_flag_off_the_reader_is_never_called(self) -> None:
        """CONSTRAINT 1 -- no default can apply, so nothing is fetched and nothing skipped."""
        (params, _, _, _, skipped), reads = self._resolve({}, allow_truncated=False, answer=None)
        assert params == {} and skipped is None and reads == 0

    @pytest.mark.parametrize("allow_truncated", [True, False], ids=["flag-on", "flag-off"])
    def test_a_caller_that_sent_null_deferred_whatever_the_flag(self, allow_truncated: bool) -> None:
        """cascor#678 follow-up, item 3 (register constraint 4). The KEY is present, so the default never applies.

        The request carried ``allow_truncation: null``: it deferred to the producer.
        This service never overrides a key the request carries -- the test is the
        key's PRESENCE, not its value -- so the reader is not consulted and the value
        goes on the wire untouched, flag on or off. That is recorded as
        ``CALLER_DEFERRED`` in BOTH positions: with the flag off it used to be
        ``None``, and the refusal then named a knob that cannot change the outcome.
        """
        (params, source, wire, refused, skipped), reads = self._resolve({"allow_truncation": None}, allow_truncated=allow_truncated)
        assert params == {"allow_truncation": None}
        assert (source, wire, refused, skipped, reads) == (None, False, False, _OPT_IN_SKIPPED_CALLER_DEFERRED, 0)

    @pytest.mark.parametrize("allow_truncated", [True, False], ids=["flag-on", "flag-off"])
    @pytest.mark.parametrize("blank", ["", "  "], ids=["empty", "spaces"])
    def test_a_blank_string_is_not_a_deferral(self, allow_truncated: bool, blank: str) -> None:
        """#688's validation: juniper-data does not defer on a blank string -- it answers 400.

        ``_as_bool_stance`` reads a blank as no stance, and the resolver used to
        record it as ``CALLER_DEFERRED``, so a remedy spoke of a request that
        "deferred to the producer" -- for one the producer rejects as a bad
        parameter ("Input should be a valid boolean"). The key is still present, so
        the default still never applies and the value still goes on the wire
        untouched; what changes is that no reason is recorded for it.
        """
        (params, source, wire, refused, skipped), reads = self._resolve({"allow_truncation": blank}, allow_truncated=allow_truncated)
        assert params == {"allow_truncation": blank}
        assert (source, wire, refused, skipped, reads) == (None, False, False, None, 0)

    @pytest.mark.parametrize("allow_truncated", [True, False], ids=["flag-on", "flag-off"])
    @pytest.mark.parametrize("spelling", ["f", "F", "n", "N", "no", "off", "0", 0, 0.0], ids=repr)
    def test_a_refusal_spelled_as_juniper_data_reads_it_is_a_refusal(self, allow_truncated: bool, spelling: Any) -> None:
        """#690's fixup: juniper-data reads each of these as ``false``, so this service must as well.

        ``bool("f")`` is ``True``, and the stance reader used to fall back to it. So a caller
        that refused with "f" or "n" was recorded as opting in -- the request's own
        acceptance, on the wire -- while the producer refused. The value still goes on the wire
        untouched.
        """
        (params, source, wire, refused, skipped), reads = self._resolve({"allow_truncation": spelling}, allow_truncated=allow_truncated)
        assert params == {"allow_truncation": spelling}
        assert (source, wire, refused, skipped, reads) == (None, False, True, None, 0)

    @pytest.mark.parametrize("allow_truncated", [True, False], ids=["flag-on", "flag-off"])
    @pytest.mark.parametrize("rejected", ["maybe", " true", "yes!", 2, 0.5, []], ids=repr)
    def test_a_value_juniper_data_rejects_is_read_as_a_blank_string_is(self, allow_truncated: bool, rejected: Any) -> None:
        """No stance and no deferral: juniper-data answers each of these 400, before any stance matters.

        They used to read as ``True`` by truthiness (``[]`` as ``False``), and ``" true"``
        after a strip juniper-data does not do. So the resolver recorded an opt-in, or a
        refusal, for a request the producer rejects outright.
        """
        (params, source, wire, refused, skipped), reads = self._resolve({"allow_truncation": rejected}, allow_truncated=allow_truncated)
        assert params == {"allow_truncation": rejected}
        assert (source, wire, refused, skipped, reads) == (None, False, False, None, 0)


class _RecordingClientClass:
    """Stands in for ``JuniperDataClient``: records every construction's kwargs."""

    def __init__(self, listing: Any) -> None:
        self.listing = listing
        self.constructions: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> "_RecordingClientClass":
        self.constructions.append(kwargs)
        return self

    def list_generators(self) -> Any:
        if isinstance(self.listing, BaseException):
            raise self.listing
        return self.listing


class TestTheReaderIsLazyAndBounded:
    """CONSTRAINTS 1-3: nothing built until consulted; one short, retry-free client; keyed by URL."""

    def test_building_the_reader_fetches_nothing(self) -> None:
        client_class = _RecordingClientClass(LISTING)
        _TRUNCATABLE_GENERATORS.reader(client_class, source="http://juniper-data:8100", api_key="k")
        assert client_class.constructions == []

    def test_the_listing_client_is_bounded_and_does_not_retry(self) -> None:
        """The staged path reads the list under ``_lock``; the client defaults are 30 s x 3 retries."""
        from api.lifecycle.manager import _GENERATOR_LIST_RETRIES, _GENERATOR_LIST_TIMEOUT_SECONDS

        client_class = _RecordingClientClass(LISTING)
        reader = _TRUNCATABLE_GENERATORS.reader(client_class, source="http://juniper-data:8100", api_key="k")
        assert reader() == frozenset({"csv_import", "equities", "equities_seq"})
        assert client_class.constructions == [{"base_url": "http://juniper-data:8100", "api_key": "k", "timeout": _GENERATOR_LIST_TIMEOUT_SECONDS, "retries": _GENERATOR_LIST_RETRIES}]
        assert _GENERATOR_LIST_RETRIES == 0 and 0 < _GENERATOR_LIST_TIMEOUT_SECONDS <= 5

    def test_a_memo_hit_builds_no_client(self) -> None:
        client_class = _RecordingClientClass(LISTING)
        reader = _TRUNCATABLE_GENERATORS.reader(client_class, source="http://juniper-data:8100", api_key=None)
        reader()
        reader()
        assert len(client_class.constructions) == 1


class TestTheWithheldRemedy:
    """An operator whose knob is ON must not be told to turn it on."""

    _EXC = _refusal()

    @staticmethod
    def _names_the_knob(message: str) -> bool:
        return "--allow-truncated-datasets" in message or "JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS" in message

    # Each path's retry, as its refusal words it (cascor#678 follow-up, item 7).
    # Auto-start's names BOTH operator retries -- stage and start, or restart --
    # because it never runs again on its own; "retried only by restarting" was
    # the old wording, which contradicted ``_TruncatableGenerators``' docstring.
    _RETRY_BY_PATH = {
        _FETCH_PATH_STAGED_START: ("starting training again retries it",),
        _FETCH_PATH_LIVE_SWAP: ("re-issue the swap to retry it",),
        _FETCH_PATH_AUTO_START: ("stage the dataset and start training", "restart the service to run auto-start again"),
    }

    @pytest.mark.parametrize("fetch_path", [_FETCH_PATH_STAGED_START, _FETCH_PATH_LIVE_SWAP, _FETCH_PATH_AUTO_START])
    def test_a_withheld_opt_in_is_told_how_to_retry_on_each_path(self, fetch_path: str) -> None:
        """Path-accurate, and ONLY its own path: one string listing all three left the operator to pick.

        A failed start keeps its dataset staged; a swap stages nothing; auto-start
        runs once. Auto-start used to be told "a failed start leaves its dataset
        staged", which is another path's retry.
        """
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False, opt_in_skipped=_OPT_IN_SKIPPED_LIST_UNREADABLE, deployment_flag_on=True, fetch_path=fetch_path)
        assert message.startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN + " ")
        assert "WITHHELD" in message and "GET /v1/generators" in message
        for path, phrases in self._RETRY_BY_PATH.items():
            for phrase in phrases:
                assert (phrase in message) is (path == fetch_path), f"{fetch_path}: {phrase!r}"
        assert "retried only by restarting" not in message
        # The staged path does NOT need a re-stage: the failed start left the config staged.
        assert "re-stage" not in message
        assert not self._names_the_knob(message)

    def test_a_withheld_opt_in_from_no_named_path_claims_no_path(self) -> None:
        """A direct call names no path, so it gets the retry every path shares and no path's own."""
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False, opt_in_skipped=_OPT_IN_SKIPPED_LIST_UNREADABLE, deployment_flag_on=True)
        assert "retry once juniper-data answers GET /v1/generators" in message
        assert not any(phrase in message for phrases in self._RETRY_BY_PATH.values() for phrase in phrases)

    def test_a_generator_the_list_does_not_declare_is_not_told_to_turn_on_the_knob(self) -> None:
        """cascor#678 follow-up, item 4: flag ON, list READ, generator not truncatable -- so not a shortfall at all.

        Such a generator cannot be short by construction, so a refusal from it is
        not one this service can act on. Dressed as a refusal it carried the token,
        which opens canopy's three-way partial-data prompt -- every option of which
        re-sends a request that fails the same way -- and called the producer
        inconsistent. The text is a REAL refusal, so it is the NOT_TRUNCATABLE reason
        that makes this plain, not the wording.
        """
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False, opt_in_skipped=_OPT_IN_SKIPPED_NOT_TRUNCATABLE, deployment_flag_on=True)
        assert message == f"juniper-data fetch failed: {_REFUSAL_TEXT}"

    @pytest.mark.parametrize("deployment_flag_on", [True, False], ids=["flag-on", "flag-off"])
    def test_a_request_that_deferred_with_a_null_is_not_told_to_turn_on_the_knob(self, deployment_flag_on: bool) -> None:
        """Item 3: the request carried ``allow_truncation: null``, which no setting of this service overrides.

        With the flag OFF this used to get the knob remedy -- set
        JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS=true -- which cannot help: the
        default never applies to a request that carries the key.
        """
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False, opt_in_skipped=_OPT_IN_SKIPPED_CALLER_DEFERRED, deployment_flag_on=deployment_flag_on)
        assert message.startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN + " ")
        assert "carried allow_truncation with no value" in message
        assert "cannot change this outcome, on or off" in message
        assert not self._names_the_knob(message)

    def test_a_silent_caller_with_the_knob_off_still_gets_the_knob(self) -> None:
        """Guard: the pre-existing remedy is unchanged when the knob really is off."""
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False)
        assert self._names_the_knob(message) and "WITHHELD" not in message

    def test_the_remedies_differ(self) -> None:
        """CONSTRAINT 5 -- the wrong-remedy class cascor#640 removed, stated as its own inequality."""
        withheld = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False, opt_in_skipped=_OPT_IN_SKIPPED_LIST_UNREADABLE, deployment_flag_on=True)
        undeclared = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False, opt_in_skipped=_OPT_IN_SKIPPED_NOT_TRUNCATABLE, deployment_flag_on=True)
        deferred = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False, opt_in_skipped=_OPT_IN_SKIPPED_CALLER_DEFERRED)
        silent = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False)
        assert len({withheld, undeclared, deferred, silent}) == 4


class _StagedClient:
    """A juniper-data double for ``_reload_dataset``: records the request, then stops the reload."""

    sent: Dict[str, Any] = {}

    def __init__(self, listing: Any, create_error: Optional[Exception] = None) -> None:
        self._listing = listing
        self._create_error = create_error
        self.listing_calls = 0

    def list_generators(self) -> Any:
        self.listing_calls += 1
        if isinstance(self._listing, BaseException):
            raise self._listing
        return self._listing

    def create_dataset(self, *, generator: str, params: dict, persist: bool) -> dict:
        _StagedClient.sent = {"generator": generator, "params": dict(params)}
        raise self._create_error or RuntimeError("recorded")


def _reload(client: _StagedClient, *, deployment_flag: bool, generator: str = "equities", params: Optional[Dict[str, Any]] = None, constructions: Optional[List[Dict[str, Any]]] = None, url: str = "http://juniper-data:8100") -> RuntimeError:
    """``_reload_dataset`` against ``client``; every ``JuniperDataClient(...)`` returns it.

    ``constructions``, when given, collects the kwargs of every construction, so an
    arm can tell the dataset client from the bounded listing client.
    """
    manager = TrainingLifecycleManager.__new__(TrainingLifecycleManager)
    manager.logger = logging.getLogger("test.truncatable")
    manager._dataset_shortfall = None
    settings = SimpleNamespace(juniper_data_url=url, allow_truncated_datasets=deployment_flag)

    def _construct(**kwargs: Any) -> _StagedClient:
        if constructions is not None:
            constructions.append(kwargs)
        return client

    with (
        patch("juniper_data_client.JuniperDataClient", _construct),
        patch("api.settings.Settings", lambda: settings),
        patch("api.secrets.get_secret", lambda _name: "key"),
        pytest.raises(RuntimeError) as excinfo,
    ):
        manager._reload_dataset(dataset_type=generator, params=dict(params or {}))
    return excinfo.value


class TestTheStagedPathReadsTheList:
    """``_reload_dataset`` end to end, up to the request that reaches the producer."""

    def test_the_default_reaches_a_generator_the_list_confirms(self) -> None:
        _reload(_StagedClient(LISTING), deployment_flag=True)
        assert _StagedClient.sent["params"]["allow_truncation"] is True

    def test_a_generator_the_list_does_not_confirm_is_sent_nothing(self) -> None:
        """``equities`` missing from juniper-data's list means cascor must not assume it."""
        _reload(_StagedClient([_entry("equities", truncatable=False)]), deployment_flag=True)
        assert "allow_truncation" not in _StagedClient.sent["params"]

    def test_an_unreadable_list_withholds_and_the_refusal_says_why(self) -> None:
        client = _StagedClient(ConnectionError("connection refused"), create_error=_refusal())
        error = _reload(client, deployment_flag=True)
        assert "allow_truncation" not in _StagedClient.sent["params"]
        assert str(error).startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN)
        assert "WITHHELD" in str(error) and "--allow-truncated-datasets" not in str(error)

    def test_a_schemaless_listing_withholds_instead_of_caching_an_empty_set(self) -> None:
        """Item 3, on the live path: the listing answers, carries no schemas, and must count as UNREAD.

        Read as a success it derived ``frozenset()``, so ``equities`` was sent no
        opt-in and the 422 that followed told the operator to set a flag that was
        already on. It must be withheld-and-retried instead, and say so.
        """
        client = _StagedClient([{"name": "equities", "parameters": ["tickers"]}], create_error=_refusal())
        error = _reload(client, deployment_flag=True)
        assert "allow_truncation" not in _StagedClient.sent["params"]
        assert "WITHHELD" in str(error) and "JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS" not in str(error)
        # Not memoised: the next request reads again.
        _reload(client, deployment_flag=True)
        assert client.listing_calls == 2

    @pytest.mark.parametrize(
        ("generator", "listing", "params", "detail", "status"),
        [
            ("equities", [_entry("equities", truncatable=False)], None, _REFUSAL_TEXT, 422),
            ("spiral", LISTING, {"n_spirals": 1}, _N_SPIRALS_400, 400),
        ],
        ids=["equities-undeclared", "spiral-param-error"],
    )
    def test_a_refusal_for_a_generator_the_list_does_not_declare_does_not_name_the_knob(self, generator: str, listing: Any, params: Optional[Dict[str, Any]], detail: str, status: int) -> None:
        """cascor#678 follow-up, item 4, on the live path: flag ON, list READ, generator not declared.

        The first case is a REAL refusal from a generator the list does not declare,
        so the NOT_TRUNCATABLE reason is what makes it plain. The second is the one
        #678's post-merge validation reproduced, an ordinary parameter error on
        ``spiral``: it used to carry the refusal token -- which opens canopy's
        three-way partial-data prompt -- and a sentence calling the producer
        inconsistent. It was written as a 422 until #688's validation; juniper-data
        answers it 400, as here.
        """
        client = _StagedClient(listing, create_error=JuniperDataValidationError(detail, status_code=status))
        error = _reload(client, deployment_flag=True, generator=generator, params=params)
        assert client.listing_calls == 1, "the list was not read, so NOT_TRUNCATABLE was never the reason -- the arm proves nothing"
        assert str(error) == f"juniper-data fetch failed: {detail}"

    @pytest.mark.parametrize("deployment_flag", [True, False], ids=["flag-on", "flag-off"])
    @pytest.mark.parametrize(
        ("generator", "params", "detail"),
        [
            ("equities", {"incomplete_rows": "keep"}, _INCOMPLETE_ROWS_400),
            ("csv_import", {"file_path": "big.csv", "allow_truncation": ""}, _BLANK_ALLOW_TRUNCATION_400),
            ("spiral", {"n_spirals": 1}, _N_SPIRALS_400),
        ],
        ids=["incomplete-rows-keep", "blank-allow-truncation", "n-spirals-1"],
    )
    def test_a_parameter_error_is_a_plain_fetch_failure_on_the_live_path(self, generator: str, params: Dict[str, Any], detail: str, deployment_flag: bool) -> None:
        """#688's validation, MEDIUM, end to end: a real 400 that NAMES a truncation field is not a refusal.

        juniper-data answers a parameter its generator rejects with 400 and names the
        field. The check matched the names ``allow_truncation`` / ``incomplete_rows``,
        so with the flag off ``incomplete_rows='keep'`` got the token and the knob
        remedy, and a blank ``allow_truncation`` -- in EITHER position, because the
        resolver also called it a deferral -- got the token and a remedy saying the
        request "deferred to the producer". Both measured against a real juniper-data
        main server. The caller's value still reaches the producer untouched.
        """
        client = _StagedClient(LISTING, create_error=JuniperDataValidationError(detail, status_code=400))
        error = _reload(client, deployment_flag=deployment_flag, generator=generator, params=params)
        for key, value in params.items():
            assert _StagedClient.sent["params"][key] == value
        assert str(error) == f"juniper-data fetch failed: {detail}"

    @pytest.mark.parametrize("deployment_flag", [True, False], ids=["flag-on", "flag-off"])
    def test_a_null_stance_reaches_the_producer_as_sent_and_its_refusal_names_no_knob(self, deployment_flag: bool) -> None:
        """cascor#678 follow-up, item 3, on the live path. The flag-on arm kills mutant M21.

        M21 tested the VALUE (``params.get("allow_truncation") is None``) where the
        resolver tests the key's ABSENCE. It survived every suite, because no test
        sent a null down this path: with the flag on it overrides the caller's null
        with this service's default -- ``True`` on the wire. With the flag off, the
        refusal used to name JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS, which cannot
        change the outcome for a request that carries the key.
        """
        client = _StagedClient(LISTING, create_error=_refusal())
        error = _reload(client, deployment_flag=deployment_flag, params={"allow_truncation": None})
        assert _StagedClient.sent["params"] == {"allow_truncation": None}
        assert client.listing_calls == 0
        message = str(error)
        assert message.startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN)
        assert "carried allow_truncation with no value" in message
        assert "--allow-truncated-datasets" not in message and "JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS" not in message

    @pytest.mark.parametrize("deployment_flag", [True, False], ids=["flag-on", "flag-off"])
    @pytest.mark.parametrize("spelling", ["f", "n"])
    def test_a_refusal_spelled_f_or_n_gets_the_refusal_remedy_on_the_live_path(self, deployment_flag: bool, spelling: str) -> None:
        """#690's fixup, end to end: "f" is juniper-data's ``false``, so its refusal is the caller's own refusal.

        The stance reader made "f" and "n" an opt-in, so ``allow_truncated`` reached the
        describer as ``True``. The refusal that followed came out as a plain ``juniper-data
        fetch failed`` with no remedy at all, where the caller is owed "explicitly refused"
        and how to re-send.
        """
        client = _StagedClient(LISTING, create_error=_refusal())
        error = _reload(client, deployment_flag=deployment_flag, params={"allow_truncation": spelling})
        assert _StagedClient.sent["params"] == {"allow_truncation": spelling}
        message = str(error)
        assert message.startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN)
        assert "explicitly refused" in message
        assert "--allow-truncated-datasets" not in message

    @pytest.mark.parametrize("model", [StageDatasetRequest, SwapDatasetLiveRequest])
    def test_a_null_stance_survives_both_routes_request_models(self, model: Any) -> None:
        """Why item 3 is reachable at all: ``exclude_none`` drops a top-level ``None``, never one inside ``params``."""
        body = model.model_validate({"dataset_type": "equities", "params": {"allow_truncation": None, "tickers": ["AAPL"]}})
        assert body.model_dump(exclude_none=True)["params"] == {"allow_truncation": None, "tickers": ["AAPL"]}

    def test_with_the_flag_off_the_list_is_never_read(self) -> None:
        """CONSTRAINT 1 -- the default deployment (flag off) never asks juniper-data for the list."""
        client = _StagedClient(LISTING)
        _reload(client, deployment_flag=False)
        assert client.listing_calls == 0

    @pytest.mark.parametrize("caller_value", [True, False])
    def test_a_caller_stance_is_never_worth_a_fetch(self, caller_value: bool) -> None:
        """CONSTRAINT 1 -- flag ON, but the caller decided: the set cannot change the outcome."""
        client = _StagedClient(LISTING)
        _reload(client, deployment_flag=True, params={"allow_truncation": caller_value})
        assert client.listing_calls == 0
        assert _StagedClient.sent["params"]["allow_truncation"] is caller_value

    @pytest.mark.parametrize("caller_value", [True, False])
    def test_an_unreadable_list_never_touches_the_callers_value(self, caller_value: bool) -> None:
        """CONSTRAINT 6 on the live path -- withholding drops only this service's default."""
        _reload(_StagedClient(ConnectionError("connection refused")), deployment_flag=True, params={"allow_truncation": caller_value})
        assert _StagedClient.sent["params"]["allow_truncation"] is caller_value

    def test_the_listing_client_is_a_separate_bounded_one(self) -> None:
        """CONSTRAINT 3 -- under ``_lock``, the list is read with a short timeout and no retries.

        The dataset client keeps the defaults; only the listing read is bounded.
        """
        from api.lifecycle.manager import _GENERATOR_LIST_RETRIES, _GENERATOR_LIST_TIMEOUT_SECONDS

        constructions: List[Dict[str, Any]] = []
        _reload(_StagedClient(LISTING), deployment_flag=True, constructions=constructions)
        assert {"base_url": "http://juniper-data:8100", "api_key": "key"} in constructions
        assert {"base_url": "http://juniper-data:8100", "api_key": "key", "timeout": _GENERATOR_LIST_TIMEOUT_SECONDS, "retries": _GENERATOR_LIST_RETRIES} in constructions
        assert len(constructions) == 2

    def test_the_memo_follows_the_url_the_reload_reads(self) -> None:
        """CONSTRAINT 2 -- ``_reload_dataset`` re-reads the URL each call; the memo must follow it.

        A memo that ignored the URL would pin whichever juniper-data answered first:
        here, the second producer's own list would never be read.
        """
        _reload(_StagedClient([_entry("equities", truncatable=False)]), deployment_flag=True, url="http://first:8100")
        assert "allow_truncation" not in _StagedClient.sent["params"]
        _reload(_StagedClient(LISTING), deployment_flag=True, url="http://second:8100")
        assert _StagedClient.sent["params"]["allow_truncation"] is True


def _swap_network() -> types.SimpleNamespace:
    """A live-swap-capable fake network (equal dims), built as ``test_shortfall_lifecycle.py`` builds it."""
    net = types.SimpleNamespace(input_size=2, output_size=2, active_output_dim=2, output_weights=torch.zeros(2, 2), output_bias=torch.zeros(2), hidden_units=[{"weights": torch.zeros(3)}], candidate_pool_size=8)
    net._resize_network_for_dataset = MagicMock(return_value={"hidden_preserved": 1, "input_delta": 0, "output_delta": 0})
    net.record_dataset_swap_event = MagicMock(return_value={"event": "dataset_swap", "id": 1})
    return net


class TestEachPathNamesItsOwnRetry:
    """Item 7's WIRING: each live path tells the describer which path it is, so the retry it prints is true.

    The describer's texts are pinned in ``TestTheWithheldRemedy``; these arms prove
    ``start_training`` and ``swap_dataset_live`` pass their own path, through the
    REAL ``_reload_dataset``. Flag ON, the list unreadable (so the opt-in is
    withheld), and the producer refuses. Each arm also checks that the retry it was
    told is actually available -- a staged start stays staged; a swap stages nothing.
    auto-start's arm is in ``test_auto_start_shortfall.py``.
    """

    @classmethod
    def _producer(cls) -> Any:
        """Patches for a juniper-data whose list cannot be read and whose create refuses."""

        class _Client:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            def list_generators(self) -> Any:
                raise ConnectionError("connection refused")

            def create_dataset(self, *, generator: str, params: dict, persist: bool) -> dict:
                raise _refusal()

        settings = SimpleNamespace(juniper_data_url="http://juniper-data:8100", allow_truncated_datasets=True)
        return (patch("juniper_data_client.JuniperDataClient", _Client), patch("api.settings.Settings", lambda: settings), patch("api.secrets.get_secret", lambda _name: "key"))

    @staticmethod
    def _told(message: str) -> set:
        """Which paths' retries ``message`` gives."""
        return {path for path, phrases in TestTheWithheldRemedy._RETRY_BY_PATH.items() if any(phrase in message for phrase in phrases)}

    def test_a_staged_start_is_told_to_start_again(self) -> None:
        manager = TrainingLifecycleManager()
        try:
            manager.stage_dataset_config(dataset_type="equities")
            client_patch, settings_patch, secret_patch = self._producer()
            with client_patch, settings_patch, secret_patch, patch.object(manager, "_run_training"), pytest.raises(RuntimeError) as excinfo:
                manager.start_training()
            message = str(excinfo.value)
            assert "WITHHELD" in message
            assert self._told(message) == {_FETCH_PATH_STAGED_START}
            # ...and it is true: the failed start left the dataset staged, so starting again retries it.
            assert manager.get_pending_dataset_config() == {"dataset_type": "equities"}
        finally:
            manager.shutdown()

    def test_a_live_swap_is_told_to_re_issue_the_swap(self) -> None:
        manager = TrainingLifecycleManager()
        try:
            manager.model = types.SimpleNamespace(network=_swap_network())
            manager._experimental_functions_enabled = True
            manager._train_x, manager._train_y = torch.zeros(8, 2), torch.zeros(8, 2)
            client_patch, settings_patch, secret_patch = self._producer()
            with (
                client_patch,
                settings_patch,
                secret_patch,
                patch.object(manager.state_machine, "is_started", return_value=True),
                patch.object(manager, "save_snapshot", return_value={"id": "snap"}),
                patch.object(manager, "_run_training"),
                pytest.raises(RuntimeError) as excinfo,
            ):
                manager.swap_dataset_live(dataset_type="equities")
            message = str(excinfo.value)
            assert "WITHHELD" in message
            assert self._told(message) == {_FETCH_PATH_LIVE_SWAP}
            # ...and it is true: a swap stages nothing, so re-issuing it is the only retry.
            assert manager.get_pending_dataset_config() is None
        finally:
            manager.shutdown()


class TestTheConstantIsGone:
    """A built-in fallback copy was offered and rejected; it must not come back."""

    def test_no_truncatable_generator_constant_is_exported(self) -> None:
        assert not hasattr(constants_api, "_PROJECT_API_TRUNCATABLE_GENERATORS")
        defaults = sys.modules["cascor_constants.constants_api.constants_api_defaults"]
        assert not hasattr(defaults, "_PROJECT_API_TRUNCATABLE_GENERATORS")
