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
"""

from __future__ import annotations

import contextlib
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
def _producer(*, meta: Dict[str, Any], arrays: Dict[str, Any]) -> Iterator[None]:
    """A juniper-data double at the client seam, so the REAL ``_reload_dataset`` runs.

    The deployment flag is off, so the generator list is never read.
    """

    class _Client:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def create_dataset(self, *, generator: str, params: dict, persist: bool) -> dict:
            return {"dataset_id": "partial-1", "meta": meta}

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
    tensors, and the annotation describing them, together.
    """
    x, y = _tensors()

    def _fetch(**_cfg: Any) -> None:
        m._train_x, m._train_y = x, y
        m._current_dataset_config = {"dataset_type": "equities"}
        m._dataset_shortfall = dict(annotation)

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
