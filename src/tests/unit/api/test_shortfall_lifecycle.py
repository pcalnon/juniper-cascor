"""APD-CASCOR-013: ``dataset_shortfall`` describes what THIS run fetched, or it is None.

Project:       Juniper
Sub-Project:   JuniperCascor
Application:   juniper_cascor
File Name:     test_shortfall_lifecycle.py
Author:        Paul Calnon
License:       MIT License

The ruling (juniper-ml ``notes/JUNIPER_2026-08-14_JUNIPER-ECOSYSTEM_DEFECT-REGISTER.md``
section 4.9): **clear the annotation at the start of every run.** "The field is
named for what THIS run trained on, so it describes this run or it is null; a run
that fetched nothing correctly reports nothing." Rejected: clearing only on a new
fetch, and documenting the old keep-until-reset behaviour.

Before, ``_dataset_shortfall`` was written at one line and never cleared -- not on
a new run, not on reset -- so a run started on inline tensors kept the previous
run's annotation and named a ``dataset_id`` it was not training on. Same family as
APD-CASCOR-007: an annotation describing data this run is not training on.

The arms below are split by what they prove. The first class FAILS against the
pre-fix manager (the defect). The second holds before and after: they are the
over-correction guards -- a start that never happens must not erase the annotation
of the run that did, and a run whose OWN fetch is short must keep what it wrote.
"""

from __future__ import annotations

import types
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch

from api.lifecycle.manager import SwapCancelledError, TrainingLifecycleManager

pytestmark = pytest.mark.unit

_PRIOR: Dict[str, Any] = {"dataset_id": "an-earlier-run", "summary": "14 of 503 symbols imported (cap 14)"}
_OWN: Dict[str, Any] = {"dataset_id": "this-run", "summary": "20 of 503 symbols imported (cap 20)"}


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


class TestEveryRunStartsWithItsOwnAnnotation:
    """Each of these fails against the pre-fix manager, which never cleared the field."""

    def test_an_inline_run_after_a_partial_run_reports_none(self, mgr):
        """THE ROW'S CASE. Inline tensors, no staged dataset, a previous run's annotation standing.

        The run trains on the caller's tensors; the annotation named another
        run's artifact. It must read None, because nothing was fetched for this run.
        """
        mgr._dataset_shortfall = dict(_PRIOR)
        x, y = _tensors()
        _start(mgr, X=x, y=y)
        assert mgr.get_status()["dataset_shortfall"] is None

    def test_a_run_on_retained_data_reports_none(self, mgr):
        """The ruling's own sentence: a run that fetched nothing correctly reports nothing.

        No ``X``, nothing staged: the run continues on the tensors an earlier run
        loaded. It fetched nothing, so it reports nothing -- stated as its own arm
        because it is the consequence of the ruling most likely to surprise, and a
        future change to it should have to delete a test that names it.
        """
        x, y = _tensors()
        mgr._train_x, mgr._train_y = x, y
        mgr._dataset_shortfall = dict(_PRIOR)
        _start(mgr)
        assert mgr.get_status()["dataset_shortfall"] is None

    def test_a_caller_that_fetched_its_own_tensors_hands_the_annotation_in(self, mgr):
        """auto-start's shape: it fetches, then starts the run on those tensors inline.

        Written onto the manager BEFORE the start, as auto-start used to, the
        start would now erase it -- so the annotation travels with the tensors.
        """
        mgr._dataset_shortfall = dict(_PRIOR)
        x, y = _tensors()
        _start(mgr, X=x, y=y, dataset_shortfall=dict(_OWN))
        assert mgr.get_status()["dataset_shortfall"] == _OWN

    def test_an_annotation_without_tensors_is_refused(self, mgr):
        """It would be a claim about data this call did not supply."""
        mgr._dataset_shortfall = dict(_PRIOR)
        with pytest.raises(ValueError, match="dataset_shortfall"):
            _start(mgr, dataset_shortfall=dict(_OWN))
        assert mgr._dataset_shortfall == _PRIOR

    def test_a_refused_staged_fetch_restores_the_previous_annotation(self, mgr):
        """``_reload_dataset`` writes the annotation BEFORE it converts the artifact.

        A refusal after that write (here a val-less artifact under the section 6.1
        rules) binds no tensors, so the data still loaded is the previous data --
        and the annotation must still describe it, not the refused artifact.
        """
        x, y = _tensors()
        mgr._train_x, mgr._train_y = x, y
        mgr._dataset_shortfall = dict(_PRIOR)
        mgr._pending_dataset_config = {"dataset_type": "equities"}

        def _write_then_refuse(**_cfg: Any) -> None:
            mgr._dataset_shortfall = dict(_OWN)
            raise ValueError("artifact carries neither X_val nor X_test")

        with patch.object(mgr, "_reload_dataset", side_effect=_write_then_refuse), pytest.raises(ValueError):
            _start(mgr)
        assert mgr._dataset_shortfall == _PRIOR
        # Still staged, so the user can fix the upstream issue and retry.
        assert mgr._pending_dataset_config == {"dataset_type": "equities"}

    def test_reset_clears_it(self, mgr):
        """Reset discards the run -- its metrics and counters -- so its annotation goes too."""
        mgr._dataset_shortfall = dict(_PRIOR)
        mgr.reset()
        assert mgr.get_status()["dataset_shortfall"] is None


class TestWhatMustSurvive:
    """Over-correction guards. These pass before AND after the fix, by design."""

    def test_a_staged_run_keeps_the_annotation_its_own_fetch_wrote(self, mgr):
        """The run's own fetch is short: the annotation is this run's, and must reach the status."""
        x, y = _tensors()
        mgr._dataset_shortfall = dict(_PRIOR)
        mgr._pending_dataset_config = {"dataset_type": "equities"}

        def _fetch(**_cfg: Any) -> None:
            mgr._train_x, mgr._train_y = x, y
            mgr._dataset_shortfall = dict(_OWN)

        with patch.object(mgr, "_reload_dataset", side_effect=_fetch):
            _start(mgr)
        assert mgr.get_status()["dataset_shortfall"] == _OWN
        assert mgr._pending_dataset_config is None

    def test_a_rejected_start_leaves_the_running_run_annotated(self, mgr):
        """A start refused because a run is in progress must not touch that run's annotation."""
        mgr._dataset_shortfall = dict(_PRIOR)
        x, y = _tensors()
        with patch.object(mgr.state_machine, "is_started", return_value=True), pytest.raises(RuntimeError, match="already in progress"):
            mgr.start_training(X=x, y=y)
        assert mgr._dataset_shortfall == _PRIOR

    def test_a_start_with_no_data_leaves_it(self, mgr):
        """No run began, so the previous run's annotation still describes the previous run."""
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
    """The swap re-submits the run on new data, so it is a run start too."""

    def _swap(self, mgr, reload_effect):
        with patch.object(mgr.state_machine, "is_started", return_value=True), patch.object(mgr, "save_snapshot", return_value={"id": "snap"}), patch.object(mgr, "_run_training"), patch.object(mgr, "_reload_dataset", side_effect=reload_effect):
            return mgr.swap_dataset_live(dataset_type="equities")

    def test_a_cancelled_swap_restores_the_annotation_of_the_data_it_restores(self, mgr):
        """FAILS pre-fix. The swap's fetch rewrites the annotation, then the cancel rolls the DATA back.

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
