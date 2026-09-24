"""APD-CASCOR-013: ``dataset_shortfall`` describes the data a run trains on -- it moves with the data.

Project:       Juniper
Sub-Project:   JuniperCascor
Application:   juniper_cascor
File Name:     test_shortfall_lifecycle.py
Author:        Paul Calnon
License:       MIT License

The ruling (juniper-ml ``notes/JUNIPER_2026-08-14_JUNIPER-ECOSYSTEM_DEFECT-REGISTER.md``
section 4.9): the field "is named for what THIS run trained on, so it describes this
run or it is null". Rejected: clearing only on a new fetch, and documenting the old
keep-until-reset behaviour.

**Owner ruling 2026-09-23 -- "follow the loaded data".** A start on data RETAINED
from an earlier fetch (Stop -> Start, nothing new fetched) carries that data's
annotation, so a run on partial data always says so, consistent with
``current_dataset``. Rejected: reporting ``None`` for such a start because it fetched
nothing -- the "annotation denies the partial data it trains on" shape APD-CASCOR-007
fixed. A run on inline data reports ``None``; a run on a new fetch reports that fetch's
annotation.

**Owner ruling 2026-09-24 -- "keep while fetched splits stay"** (extends
APD-CASCOR-013). ``X_val`` / ``X_test`` are retain-on-omit (cascor#582), so inline
tensors can replace SOME of a fetch's partitions: a train-only start after a partial
fetch early-stops on that fetch's val and reports on its test. The fetch's annotation,
and ``current_dataset``, stay while ANY partition of that fetch is still loaded, and
clear only once train, val and test have all been replaced. Before the ruling, binding
``X`` alone set both to ``None`` while the run still selected on, and reported from,
the partial fetch's rows (found by #678's post-merge validation, over real HTTP).

Before, ``_dataset_shortfall`` was written at one line and never cleared, so a run
started on inline tensors kept the previous run's annotation and named a
``dataset_id`` it was not training on.

What each class proves, and against what:

* ``TestTheAnnotationMovesWithTheData`` -- the behaviour. Several arms FAIL against
  the pre-fix manager (the stale annotation on inline data); the retained-data and
  reset arms FAIL against the first cut of this change, which cleared the
  annotation on every start that fetched nothing.
* ``TestWhatMustSurvive`` -- over-correction guards. "Follow the data" must never
  become "keep the last annotation forever": new inline data after a partial run
  still reports ``None``, and a start that never happens changes nothing.
* ``TestKeepWhileFetchedSplitsStay`` -- the 2026-09-24 ruling, through the REAL
  fetch: each partial replacement keeps the record, replacing all three clears it
  (in one start or across several), and a new fetch replaces it outright.
* ``TestTheShortfallIsLoggedOnceItsDataIsBound`` -- the training log says a run is on
  a partial dataset only once that data is bound, never for an artifact refused
  after the producer answered.
* ``TestTheLiveSwap`` -- a swap's rollback restores the annotation AND which
  partitions it stands on.
"""

from __future__ import annotations

import contextlib
import logging
import types
from types import SimpleNamespace
from typing import Any, Dict, Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from api.lifecycle.manager import SwapCancelledError, TrainingLifecycleManager

pytestmark = pytest.mark.unit

_PRIOR: Dict[str, Any] = {"dataset_id": "an-earlier-run", "summary": "14 of 503 symbols imported (cap 14)"}
_OWN: Dict[str, Any] = {"dataset_id": "this-run", "summary": "20 of 503 symbols imported (cap 20)"}
_PARTIAL_META: Dict[str, Any] = {"truncation": {"unit": "symbols", "cap": 14, "requested": 503, "imported": 14}}
_TRAIN_ONLY_ARTIFACT: Dict[str, Any] = {"X_train": np.zeros((5, 2), dtype=np.float32), "y_train": np.zeros((5, 2), dtype=np.float32)}


def _three_partition_artifact() -> Dict[str, Any]:
    rng = np.random.default_rng(20260923)
    return {
        "X_train": rng.standard_normal((20, 2)).astype("float32"),
        "y_train": rng.standard_normal((20, 2)).astype("float32"),
        "X_val": rng.standard_normal((6, 2)).astype("float32"),
        "y_val": rng.standard_normal((6, 2)).astype("float32"),
        "X_test": rng.standard_normal((4, 2)).astype("float32"),
        "y_test": rng.standard_normal((4, 2)).astype("float32"),
    }


@contextlib.contextmanager
def _producer(*, meta: Dict[str, Any], arrays: Dict[str, Any], dataset_id: str = "partial-1") -> Iterator[None]:
    """A juniper-data double at the client seam, so the REAL ``_reload_dataset`` runs.

    The deployment flag is off, so the generator list is never read.
    """

    class _Client:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def create_dataset(self, *, generator: str, params: dict, persist: bool) -> dict:
            return {"dataset_id": dataset_id, "meta": meta}

        def download_artifact_npz(self, dataset_id: str) -> dict:
            return arrays

    settings = SimpleNamespace(juniper_data_url="http://juniper-data:8100", allow_truncated_datasets=False)
    with patch("juniper_data_client.JuniperDataClient", _Client), patch("api.settings.Settings", lambda: settings), patch("api.secrets.get_secret", lambda _name: "key"):
        yield


@pytest.fixture
def mgr():
    m = TrainingLifecycleManager()
    try:
        yield m
    finally:
        m.shutdown()


def _tensors() -> tuple:
    return torch.zeros(8, 2), torch.zeros(8, 2)


def _start(m: TrainingLifecycleManager, **kwargs: Any) -> Dict[str, Any]:
    """``start_training`` with the fit replaced, so only its synchronous half runs."""
    with patch.object(m, "_run_training"):
        result = m.start_training(**kwargs)
        if m._training_future is not None:
            m._training_future.result(timeout=10)
    return result


def _start_on_a_staged_partial_fetch(m: TrainingLifecycleManager, annotation: Dict[str, Any]) -> None:
    """A run whose data comes from a staged fetch the producer could not deliver in full.

    ``_reload_dataset`` is replaced by what it does to the manager: bind the fetched
    tensors, the annotation describing them, and which partitions that annotation
    stands on, together. This double fills the train split alone -- the manager
    holds no val or test -- so replacing train replaces all of it;
    ``TestKeepWhileFetchedSplitsStay`` drives the real fetch of all three.
    """
    x, y = _tensors()

    def _fetch(**_cfg: Any) -> None:
        m._train_x, m._train_y = x, y
        m._current_dataset_config = {"dataset_type": "equities"}
        m._dataset_shortfall = dict(annotation)
        m._described_partitions = frozenset({"train"})

    m._pending_dataset_config = {"dataset_type": "equities"}
    with patch.object(m, "_reload_dataset", side_effect=_fetch):
        _start(m)


class TestTheAnnotationMovesWithTheData:
    """The status names the shortfall of exactly the data being trained on."""

    def test_an_inline_run_after_a_partial_run_reports_none(self, mgr):
        """THE ROW'S CASE. New inline tensors after a partial run: nothing was fetched for them.

        Pre-fix, the earlier run's annotation stood and named an artifact this run
        was not training on.
        """
        _start_on_a_staged_partial_fetch(mgr, _PRIOR)
        x, y = _tensors()
        _start(mgr, X=x, y=y)
        assert mgr.get_status()["dataset_shortfall"] is None

    def test_a_run_on_retained_data_carries_that_datas_annotation(self, mgr):
        """OWNER RULING 2026-09-23. Stop -> Start on retained partial data must say so.

        The run fetched nothing, but it trains on data that was fetched partial. A
        ``None`` here is the annotation denying the partial data it trains on.
        """
        _start_on_a_staged_partial_fetch(mgr, _OWN)
        _start(mgr)  # no X, nothing staged: the data is retained
        status = mgr.get_status()
        assert status["dataset_shortfall"] == _OWN
        # ...consistent with current_dataset, which names the same retained data.
        assert status["current_dataset"] == {"dataset_type": "equities"}

    def test_reset_keeps_it_because_the_data_stays_loaded(self, mgr):
        """Reset discards the run's metrics, never its data, so the next plain start trains on it."""
        _start_on_a_staged_partial_fetch(mgr, _OWN)
        mgr.reset()
        assert mgr.get_status()["dataset_shortfall"] == _OWN
        _start(mgr)
        assert mgr.get_status()["dataset_shortfall"] == _OWN

    def test_a_start_that_binds_inline_data_and_then_fails_moves_it_anyway(self, mgr):
        """Bound WITH the data, not at submit: what is loaded is what the status describes.

        The network is smaller than the new tensors, so the pad step refuses AFTER
        the inline tensors are bound. The loaded data is now that inline data --
        ``current_dataset`` says so -- and an annotation left over from the earlier
        partial fetch would describe a dataset no longer loaded, which a later
        retained-data start would then train under.
        """
        _start_on_a_staged_partial_fetch(mgr, _PRIOR)
        with pytest.raises(ValueError, match="exceeds network capacity"):
            _start(mgr, X=torch.zeros(8, 5), y=torch.zeros(8, 2))
        status = mgr.get_status()
        assert status["current_dataset"] == {"dataset_type": None}
        assert status["dataset_shortfall"] is None

    def test_a_caller_that_fetched_its_own_tensors_hands_the_annotation_in(self, mgr):
        """auto-start's shape: it fetches, then starts the run on those tensors inline."""
        _start_on_a_staged_partial_fetch(mgr, _PRIOR)
        x, y = _tensors()
        _start(mgr, X=x, y=y, dataset_shortfall=dict(_OWN))
        assert mgr.get_status()["dataset_shortfall"] == _OWN

    def test_an_annotation_without_tensors_is_refused(self, mgr):
        """It would be a claim about data this call did not supply."""
        mgr._dataset_shortfall = dict(_PRIOR)
        with pytest.raises(ValueError, match="dataset_shortfall"):
            _start(mgr, dataset_shortfall=dict(_OWN))
        assert mgr._dataset_shortfall == _PRIOR

    def test_a_refused_artifact_binds_neither_data_nor_annotation(self, mgr):
        """The REAL fetch path: the producer answers with a partial dataset, then the artifact is refused.

        A train-only artifact has nothing held out, so section 6.1 refuses it after
        the producer has answered -- and after the annotation has been BUILT. Nothing
        is bound: the previous data stays loaded, so the previous annotation must
        still describe it, and the config stays staged for a retry.
        """
        x, y = _tensors()
        mgr._train_x, mgr._train_y = x, y
        mgr._dataset_shortfall = dict(_PRIOR)
        mgr._pending_dataset_config = {"dataset_type": "equities"}
        with _producer(meta=_PARTIAL_META, arrays=_TRAIN_ONLY_ARTIFACT), pytest.raises(RuntimeError, match="NEITHER a validation split"):
            _start(mgr)
        assert mgr._dataset_shortfall == _PRIOR
        assert mgr._train_x is x
        assert mgr._pending_dataset_config == {"dataset_type": "equities"}

    def test_no_status_poll_sees_the_annotation_before_its_data(self, mgr):
        """Item 2 of the validation: the annotation is SET WHEN THE DATA IS BOUND, never before.

        ``get_status()`` reads the field without the lock, so a poll during the
        artifact's conversion used to show the new fetch's shortfall beside data
        that was not loaded yet. Captured here at exactly that moment, through the
        real ``_reload_dataset``.
        """
        mgr._dataset_shortfall = dict(_PRIOR)
        mgr._pending_dataset_config = {"dataset_type": "equities"}
        seen_during_conversion: Dict[str, Any] = {}
        real_convert = TrainingLifecycleManager._artifact_to_tensors

        def _convert_and_poll(arrays: Any) -> Any:
            seen_during_conversion["shortfall"] = mgr.get_status()["dataset_shortfall"]
            return real_convert(arrays)

        with _producer(meta=_PARTIAL_META, arrays=_three_partition_artifact()), patch.object(TrainingLifecycleManager, "_artifact_to_tensors", side_effect=_convert_and_poll):
            _start(mgr)
        assert seen_during_conversion["shortfall"] == _PRIOR, "a poll mid-conversion saw the new annotation before its data was bound"
        after = mgr.get_status()["dataset_shortfall"]
        assert after is not None and after["dataset_id"] == "partial-1"
        assert mgr._train_x.shape[0] == 20


class TestWhatMustSurvive:
    """Over-correction guards: following the data must not become keeping a stale annotation."""

    def test_retained_inline_data_stays_unannotated(self, mgr):
        """Partial fetch, then new inline data, then a retained start: that run trains on the INLINE data.

        The annotation it carries is the inline data's (``None``), never the older
        partial fetch's -- "follow the data" is not "keep the last annotation".
        """
        _start_on_a_staged_partial_fetch(mgr, _PRIOR)
        x, y = _tensors()
        _start(mgr, X=x, y=y)
        _start(mgr)
        assert mgr.get_status()["dataset_shortfall"] is None

    def test_a_staged_run_keeps_the_annotation_its_own_fetch_wrote(self, mgr):
        """The run's own fetch is short: the annotation is this run's, and must reach the status."""
        mgr._dataset_shortfall = dict(_PRIOR)
        _start_on_a_staged_partial_fetch(mgr, _OWN)
        assert mgr.get_status()["dataset_shortfall"] == _OWN
        assert mgr._pending_dataset_config is None

    def test_a_rejected_start_leaves_the_running_run_annotated(self, mgr):
        """A start refused because a run is in progress binds nothing and changes nothing."""
        mgr._dataset_shortfall = dict(_PRIOR)
        x, y = _tensors()
        with patch.object(mgr.state_machine, "is_started", return_value=True), pytest.raises(RuntimeError, match="already in progress"):
            mgr.start_training(X=x, y=y)
        assert mgr._dataset_shortfall == _PRIOR

    def test_a_start_with_no_data_leaves_it(self, mgr):
        """No run began and nothing was bound, so nothing about the loaded data changed."""
        mgr._dataset_shortfall = dict(_PRIOR)
        with pytest.raises(ValueError, match="Training data not provided"):
            _start(mgr)
        assert mgr._dataset_shortfall == _PRIOR


# ``start_training``'s keyword for each partition's tensors, and the slot it binds.
_INLINE_KWARGS = {"train": ("X", "y"), "val": ("X_val", "y_val"), "test": ("X_test", "y_test")}
_SLOT = {"train": "_train_x", "val": "_val_x", "test": "_test_x"}
_FETCHED_CONFIG: Dict[str, Any] = {"dataset_type": "equities"}


def _inline(*partitions: str) -> Dict[str, Any]:
    """``start_training`` kwargs binding fresh inline tensors to exactly ``partitions``."""
    kwargs: Dict[str, Any] = {}
    for name in partitions:
        x_key, y_key = _INLINE_KWARGS[name]
        kwargs[x_key], kwargs[y_key] = _tensors()
    return kwargs


def _fetch_partial(m: TrainingLifecycleManager, *, dataset_id: str = "partial-1") -> Dict[str, Any]:
    """A REAL staged fetch of a partial, three-partition dataset. Returns the annotation and the tensors it bound."""
    m._pending_dataset_config = dict(_FETCHED_CONFIG)
    with _producer(meta=_PARTIAL_META, arrays=_three_partition_artifact(), dataset_id=dataset_id):
        _start(m)
    annotation = m.get_status()["dataset_shortfall"]
    assert annotation is not None and annotation["dataset_id"] == dataset_id, "the fixture fetch must leave its own annotation"
    return {"annotation": annotation, "train": m._train_x, "val": m._val_x, "test": m._test_x}


class TestKeepWhileFetchedSplitsStay:
    """OWNER RULING 2026-09-24: the fetch's record stays while ANY of its partitions is loaded.

    ``X_val`` / ``X_test`` are retain-on-omit (cascor#582), so an inline start can
    replace some of a fetch's partitions and keep the rest. The record -- the
    ``dataset_shortfall`` annotation AND ``current_dataset`` -- clears only once train,
    val and test have all been replaced; a new fetch replaces it outright; ``reset()``
    and a start on retained data keep it. Every arm starts from the REAL staged fetch
    of a partial, three-partition dataset.
    """

    def test_a_train_only_inline_start_keeps_the_fetchs_record(self, mgr):
        """THE VALIDATION'S CASE. The run selects on, and reports from, the fetch's rows.

        #678's post-merge validation measured this over real HTTP: ``shortfall None``
        beside ``in-loop val IS the partial fetch's: True | reported test IS the
        partial fetch's: True``, and ``GET /v1/metrics`` said ``None`` too.
        """
        fetched = _fetch_partial(mgr)
        with patch.object(mgr, "_run_training") as run:
            mgr.start_training(**_inline("train"))  # what POST /v1/training/start sends for inline_data {train_x, train_y}
            mgr._training_future.result(timeout=10)
        _, _, in_loop_val, _ = run.call_args.args[:4]
        assert in_loop_val is fetched["val"], "the run no longer early-stops on the fetch's val -- the arm proves nothing"
        assert mgr._test_x is fetched["test"]
        status = mgr.get_status()
        assert status["dataset_shortfall"] == fetched["annotation"]
        assert status["current_dataset"] == _FETCHED_CONFIG
        assert mgr.get_metrics()["dataset_shortfall"] == fetched["annotation"]

    @pytest.mark.parametrize(
        ("replaced", "still_fetched"),
        [(("train",), ("val", "test")), (("train", "val"), ("test",)), (("train", "test"), ("val",))],
        ids=["train-only", "train+val", "train+test"],
    )
    def test_the_record_stays_while_any_fetched_split_is_loaded(self, mgr, replaced, still_fetched):
        """Each partial replacement leaves at least one of the fetch's splits in the run."""
        fetched = _fetch_partial(mgr)
        _start(mgr, **_inline(*replaced))
        for name in still_fetched:
            assert getattr(mgr, _SLOT[name]) is fetched[name], f"the fetch's {name} split is no longer loaded -- the arm proves nothing"
        status = mgr.get_status()
        assert status["dataset_shortfall"] == fetched["annotation"]
        assert status["current_dataset"] == _FETCHED_CONFIG

    def test_a_clean_fetch_keeps_its_name_too(self, mgr):
        """``current_dataset`` follows the same rule when the fetch was delivered in full (no annotation to keep)."""
        mgr._pending_dataset_config = dict(_FETCHED_CONFIG)
        with _producer(meta={}, arrays=_three_partition_artifact()):
            _start(mgr)
        _start(mgr, **_inline("train"))
        status = mgr.get_status()
        assert status["current_dataset"] == _FETCHED_CONFIG
        assert status["dataset_shortfall"] is None

    @pytest.mark.parametrize("annotation", [_OWN, None], ids=["partial", "clean"])
    def test_tensors_a_caller_fetched_itself_follow_the_same_rule(self, mgr, annotation):
        """auto-start's shape: it fetches, then hands ``start_training`` all three partitions with their record.

        That is a fetch too, so a later train-only inline start leaves its val and
        test -- and its record -- in place.
        """
        config = {"dataset_type": "equities", "tickers": ["AAPL"]}
        _start(mgr, **_inline("train", "val", "test"), dataset_config=dict(config), dataset_shortfall=dict(annotation) if annotation else None)
        _start(mgr, **_inline("train"))
        status = mgr.get_status()
        assert status["current_dataset"] == config
        assert status["dataset_shortfall"] == annotation

    def test_replacing_all_three_clears_it(self, mgr):
        """Nothing of the fetch is left: raw inline tensors, so ``null`` and an unknown identity."""
        _fetch_partial(mgr)
        _start(mgr, **_inline("train", "val", "test"))
        status = mgr.get_status()
        assert status["dataset_shortfall"] is None
        assert status["current_dataset"] == {"dataset_type": None}
        assert mgr.get_metrics()["dataset_shortfall"] is None

    def test_it_clears_once_every_fetched_split_is_replaced_across_starts(self, mgr):
        """Per partition, not per call: three starts, one split each, and only the last clears it."""
        fetched = _fetch_partial(mgr)
        _start(mgr, **_inline("train"))
        assert mgr.get_status()["dataset_shortfall"] == fetched["annotation"]
        _start(mgr, **_inline("val"))
        assert mgr.get_status()["dataset_shortfall"] == fetched["annotation"], "the fetch's test split is still loaded"
        _start(mgr, **_inline("test"))
        status = mgr.get_status()
        assert status["dataset_shortfall"] is None
        assert status["current_dataset"] == {"dataset_type": None}

    def test_a_new_fetch_replaces_everything(self, mgr):
        """The second fetch's record describes all three of ITS partitions, whatever the first had left."""
        _fetch_partial(mgr, dataset_id="partial-1")
        _start(mgr, **_inline("train"))  # the first fetch's record now stands on its val and test alone
        second = _fetch_partial(mgr, dataset_id="partial-2")
        assert mgr.get_status()["dataset_shortfall"] == second["annotation"]
        # Replacing val and test leaves the second fetch's train split, so its record stays.
        _start(mgr, **_inline("val", "test"))
        assert mgr.get_status()["dataset_shortfall"] == second["annotation"]
        assert mgr.get_status()["current_dataset"] == _FETCHED_CONFIG

    def test_reset_and_a_retained_start_keep_a_partly_replaced_record(self, mgr):
        """As before the ruling: neither binds anything, so neither changes what the record stands on."""
        fetched = _fetch_partial(mgr)
        _start(mgr, **_inline("train"))
        mgr.reset()
        assert mgr.get_status()["dataset_shortfall"] == fetched["annotation"]
        _start(mgr)  # no tensors, nothing staged: the data is retained
        assert mgr.get_status()["dataset_shortfall"] == fetched["annotation"]
        # ...and the fetch's val and test are still what it stands on: another train-only start keeps it.
        _start(mgr, **_inline("train"))
        assert mgr.get_status()["dataset_shortfall"] == fetched["annotation"]


class TestTheShortfallIsLoggedOnceItsDataIsBound:
    """cascor#678 follow-up, item 8: the log said a run was on a partial dataset before the artifact was converted."""

    def test_a_refused_artifact_leaves_no_shortfall_in_the_log(self, mgr, caplog: pytest.LogCaptureFixture):
        """The producer answered with a partial dataset, and section 6.1 then refused the artifact.

        The shortfall used to be logged as accepted the moment the producer
        answered, so the log said a run was training on data that was never loaded.
        """
        mgr._pending_dataset_config = dict(_FETCHED_CONFIG)
        with caplog.at_level(logging.WARNING), _producer(meta=_PARTIAL_META, arrays=_TRAIN_ONLY_ARTIFACT), pytest.raises(RuntimeError, match="NEITHER a validation split"):
            _start(mgr)
        assert "DATASET SHORTFALL" not in caplog.text
        assert "DATASET IS PARTIAL" not in caplog.text

    def test_a_bound_fetch_logs_its_shortfall_after_its_data(self, mgr):
        """Logged, and at that moment the tensors and the annotation are already the fetch's."""
        seen: Dict[str, Any] = {}
        real_log = mgr._log_dataset_shortfall

        def _log_and_look(meta: Dict[str, Any], *, acceptance_source: Any) -> None:
            seen["train_rows"] = None if mgr._train_x is None else mgr._train_x.shape[0]
            seen["dataset_id"] = (mgr._dataset_shortfall or {}).get("dataset_id")
            real_log(meta, acceptance_source=acceptance_source)

        mgr._pending_dataset_config = dict(_FETCHED_CONFIG)
        with patch.object(mgr, "_log_dataset_shortfall", side_effect=_log_and_look), _producer(meta=_PARTIAL_META, arrays=_three_partition_artifact()):
            _start(mgr)
        assert seen == {"train_rows": 20, "dataset_id": "partial-1"}

    def test_a_descriptor_the_log_cannot_format_does_not_undo_the_load(self, mgr, caplog: pytest.LogCaptureFixture):
        """The log runs after the data is bound, so it must not raise out of a completed load.

        ``_build_dataset_shortfall`` only counts ``degraded``; the log iterates its
        items. A list there builds an annotation and fails the log line, which used
        to run first and fail the start before anything was bound. Now the data and
        the annotation are loaded, the start completes, and the log says why it
        could not restate the shortfall.
        """
        meta = {"data_quality": {"unrescued": {}, "degraded": ["META"], "rows_affected": 3, "policy": "accept"}}
        mgr._pending_dataset_config = dict(_FETCHED_CONFIG)
        with caplog.at_level(logging.ERROR), _producer(meta=meta, arrays=_three_partition_artifact()):
            _start(mgr)
        assert mgr._pending_dataset_config is None, "the start did not complete"
        assert mgr._train_x.shape[0] == 20
        assert mgr.get_status()["dataset_shortfall"]["dataset_id"] == "partial-1"
        assert "Could not restate the dataset shortfall" in caplog.text


def _swap_network() -> types.SimpleNamespace:
    """A live-swap-capable fake network (equal-dim), as ``test_lifecycle_manager_swap`` builds it."""
    net = types.SimpleNamespace(
        input_size=2,
        output_size=2,
        active_output_dim=2,
        output_weights=torch.zeros(2, 2),
        output_bias=torch.zeros(2),
        hidden_units=[{"weights": torch.zeros(3)}],
        candidate_pool_size=8,
    )
    net._resize_network_for_dataset = MagicMock(return_value={"hidden_preserved": 1, "input_delta": 0, "output_delta": 0})
    net.record_dataset_swap_event = MagicMock(return_value={"event": "dataset_swap", "id": 1})
    return net


class TestTheLiveSwap:
    """The swap re-submits the run on new data, so it is a data change too."""

    def _swap(self, mgr, reload_effect):
        with patch.object(mgr.state_machine, "is_started", return_value=True), patch.object(mgr, "save_snapshot", return_value={"id": "snap"}), patch.object(mgr, "_run_training"), patch.object(mgr, "_reload_dataset", side_effect=reload_effect):
            return mgr.swap_dataset_live(dataset_type="equities")

    def test_a_cancelled_swap_restores_the_annotation_of_the_data_it_restores(self, mgr):
        """The swap's fetch rewrites the annotation, then the cancel rolls the DATA back.

        Without the snapshot slot the status kept naming the abandoned dataset
        while the restored one was loaded.
        """
        mgr.model = types.SimpleNamespace(network=_swap_network())
        mgr._experimental_functions_enabled = True
        mgr._train_x, mgr._train_y = _tensors()
        mgr._dataset_shortfall = dict(_PRIOR)

        def _fetch_then_cancel(**_cfg: Any) -> None:
            mgr._train_x, mgr._train_y = torch.zeros(6, 2), torch.zeros(6, 2)
            mgr._dataset_shortfall = dict(_OWN)
            mgr._swap_cancel_requested.set()  # trips the post-fetch checkpoint

        with pytest.raises(SwapCancelledError):
            self._swap(mgr, _fetch_then_cancel)
        assert mgr._dataset_shortfall == _PRIOR
        assert mgr._train_x.shape[0] == 8

    def test_a_cancelled_swap_restores_which_splits_the_record_stands_on(self, mgr):
        """OWNER RULING 2026-09-24, on the rollback path. The swap's REAL fetch rewrites the set too.

        The pre-swap state is what a partial fetch and a train-only inline start
        leave (``TestKeepWhileFetchedSplitsStay``): the record stands on the fetch's
        val and test. Restored without the set, a later start would decide what to
        keep from the ABANDONED fetch's partitions while the restored ones are loaded.
        """
        mgr.model = types.SimpleNamespace(network=_swap_network())
        mgr._experimental_functions_enabled = True
        mgr._train_x, mgr._train_y = _tensors()
        mgr._val_x, mgr._val_y = _tensors()
        mgr._test_x, mgr._test_y = _tensors()
        mgr._current_dataset_config = dict(_FETCHED_CONFIG)
        mgr._dataset_shortfall = dict(_PRIOR)
        mgr._described_partitions = frozenset({"val", "test"})
        real_reload = mgr._reload_dataset

        def _fetch_then_cancel(**cfg: Any) -> None:
            with _producer(meta=_PARTIAL_META, arrays=_three_partition_artifact(), dataset_id="swap-1"):
                real_reload(**cfg)
            assert mgr._described_partitions == {"train", "val", "test"}, "the swap's fetch must rewrite the set for this arm to mean anything"
            mgr._swap_cancel_requested.set()  # trips the post-fetch checkpoint

        with pytest.raises(SwapCancelledError):
            self._swap(mgr, _fetch_then_cancel)
        assert mgr._dataset_shortfall == _PRIOR
        assert mgr._described_partitions == {"val", "test"}
        # Behaviour, not just the slot: replacing the restored val and test now leaves nothing of it.
        mgr.model = None
        _start(mgr, **_inline("val", "test"))
        assert mgr.get_status()["dataset_shortfall"] is None

    def test_a_completed_swap_reports_the_swap_fetch(self, mgr):
        """Guard: a swap that goes through trains on the new data, so it carries that fetch's annotation."""
        mgr.model = types.SimpleNamespace(network=_swap_network())
        mgr._experimental_functions_enabled = True
        mgr._dataset_shortfall = dict(_PRIOR)

        def _fetch(**_cfg: Any) -> None:
            mgr._train_x, mgr._train_y = torch.zeros(6, 2), torch.zeros(6, 2)
            mgr._current_dataset_config = {"dataset_type": "equities"}
            mgr._dataset_shortfall = dict(_OWN)

        result = self._swap(mgr, _fetch)
        assert result["status"] == "swapped"
        assert mgr._dataset_shortfall == _OWN
