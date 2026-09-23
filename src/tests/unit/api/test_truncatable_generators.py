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
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
from pydantic import BaseModel

import cascor_constants.constants_api as constants_api
from api.lifecycle.manager import _OPT_IN_SKIPPED_LIST_UNREADABLE, _OPT_IN_SKIPPED_NOT_TRUNCATABLE, _TRUNCATABLE_GENERATORS, TrainingLifecycleManager, _TruncatableGenerators
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

    def test_an_entry_with_no_usable_schema_is_not_truncatable(self) -> None:
        """Beside at least one entry that DOES carry a schema, a schema-less entry is simply excluded."""
        listing = [{"name": "odd", "schema": None}, {"name": "odder"}, _entry("spiral", truncatable=False)]
        assert _TruncatableGenerators.derive(listing) == frozenset()

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

    _EXC = Exception("HTTP 422 allow_truncation")

    @staticmethod
    def _names_the_knob(message: str) -> bool:
        return "--allow-truncated-datasets" in message or "JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS" in message

    def test_a_withheld_opt_in_is_told_how_to_retry_on_each_path(self) -> None:
        """Path-accurate: a failed start keeps its dataset staged; a swap stages nothing; auto-start runs once."""
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False, opt_in_skipped=_OPT_IN_SKIPPED_LIST_UNREADABLE, deployment_flag_on=True)
        assert message.startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN + " ")
        assert "WITHHELD" in message and "GET /v1/generators" in message
        assert "starting training again retries it" in message
        assert "re-issue the swap" in message
        assert "retried only by restarting the service" in message
        # The staged path does NOT need a re-stage: the failed start left the config staged.
        assert "re-stage" not in message
        assert not self._names_the_knob(message)

    def test_a_generator_the_list_does_not_declare_is_not_told_to_turn_on_the_knob(self) -> None:
        """Item 3's second half: flag ON, list READ, generator not truncatable, and the producer refused anyway."""
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False, opt_in_skipped=_OPT_IN_SKIPPED_NOT_TRUNCATABLE, deployment_flag_on=True)
        assert message.startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN + " ")
        assert "does not declare allow_truncation" in message and "allow_truncation=true" in message
        assert not self._names_the_knob(message)

    def test_a_request_that_deferred_with_a_null_is_not_told_to_turn_on_the_knob(self) -> None:
        """The same invariant for the remaining flag-on case: the request itself carried allow_truncation: null."""
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False, deployment_flag_on=True)
        assert "carried allow_truncation with no value" in message
        assert not self._names_the_knob(message)

    def test_a_silent_caller_with_the_knob_off_still_gets_the_knob(self) -> None:
        """Guard: the pre-existing remedy is unchanged when the knob really is off."""
        message = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False)
        assert self._names_the_knob(message) and "WITHHELD" not in message

    def test_the_remedies_differ(self) -> None:
        """CONSTRAINT 5 -- the wrong-remedy class cascor#640 removed, stated as its own inequality."""
        withheld = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False, opt_in_skipped=_OPT_IN_SKIPPED_LIST_UNREADABLE, deployment_flag_on=True)
        undeclared = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False, opt_in_skipped=_OPT_IN_SKIPPED_NOT_TRUNCATABLE, deployment_flag_on=True)
        silent = TrainingLifecycleManager._describe_dataset_fetch_failure(self._EXC, allow_truncated=False)
        assert len({withheld, undeclared, silent}) == 3


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
        client = _StagedClient(ConnectionError("connection refused"), create_error=RuntimeError("HTTP 422 allow_truncation"))
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
        client = _StagedClient([{"name": "equities", "parameters": ["tickers"]}], create_error=RuntimeError("HTTP 422 allow_truncation"))
        error = _reload(client, deployment_flag=True)
        assert "allow_truncation" not in _StagedClient.sent["params"]
        assert "WITHHELD" in str(error) and "JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS" not in str(error)
        # Not memoised: the next request reads again.
        _reload(client, deployment_flag=True)
        assert client.listing_calls == 2

    def test_a_refusal_for_a_generator_the_list_does_not_declare_does_not_name_the_knob(self) -> None:
        """Item 3's second half, on the live path: flag ON, list read, generator not declared, producer 422s."""
        client = _StagedClient([_entry("equities", truncatable=False)], create_error=RuntimeError("HTTP 422 allow_truncation"))
        error = _reload(client, deployment_flag=True)
        assert str(error).startswith(_PROJECT_API_SHORTFALL_REFUSAL_TOKEN)
        assert "does not declare allow_truncation" in str(error)
        assert "--allow-truncated-datasets" not in str(error) and "JUNIPER_CASCOR_ALLOW_TRUNCATED_DATASETS" not in str(error)

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


class TestTheConstantIsGone:
    """A built-in fallback copy was offered and rejected; it must not come back."""

    def test_no_truncatable_generator_constant_is_exported(self) -> None:
        assert not hasattr(constants_api, "_PROJECT_API_TRUNCATABLE_GENERATORS")
        defaults = sys.modules["cascor_constants.constants_api.constants_api_defaults"]
        assert not hasattr(defaults, "_PROJECT_API_TRUNCATABLE_GENERATORS")
