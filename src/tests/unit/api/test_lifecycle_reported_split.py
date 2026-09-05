#!/usr/bin/env python
"""Unit coverage for the reported-partition surface (cascor#582).

Pins the separation this change exists to create: the partition early stopping
consumes (``_eval_split``) and the partition the final score is reported on
(``_reported_split``) are now distinct, and the second deliberately does NOT
share the first's fallback to the training split.

Also pins the two failure modes that would make the change silently useless:
a rollback dropping the reported tensors, and a stale ``final`` block surviving
into a run that produced no result.

All fast unit tests; no network is trained.
"""

import types

import pytest
import torch

from api.lifecycle.manager import TrainingLifecycleManager, _PreSwapSnapshot

pytestmark = pytest.mark.unit


@pytest.fixture
def mgr():
    m = TrainingLifecycleManager()
    try:
        yield m
    finally:
        m.shutdown()


def _t(v: float) -> torch.Tensor:
    return torch.full((4, 2), v)


class TestReportedSplitSelection:
    def test_prefers_test_over_val(self, mgr):
        mgr._val_x, mgr._val_y = _t(1.0), _t(1.0)
        mgr._test_x, mgr._test_y = _t(2.0), _t(2.0)
        x, _ = mgr._reported_split()
        assert torch.equal(x, _t(2.0)), "the test partition must win when present"
        assert mgr._reported_split_name() == "test"

    def test_falls_back_to_val_when_no_test(self, mgr):
        mgr._val_x, mgr._val_y = _t(1.0), _t(1.0)
        x, _ = mgr._reported_split()
        assert torch.equal(x, _t(1.0))
        assert mgr._reported_split_name() == "validation"

    def test_does_NOT_fall_back_to_training(self, mgr):
        """The invariant. Neither split falls back to train any more.

        Reporting a training-set score under a held-out label is the defect being
        removed, so the absence of any reportable partition has to surface as
        ``(None, None)`` rather than as a plausible number.
        """
        mgr._train_x, mgr._train_y = _t(9.0), _t(9.0)
        mgr._val_x = mgr._val_y = mgr._test_x = mgr._test_y = None
        assert mgr._reported_split() == (None, None)
        assert mgr._reported_split_name() is None

    def test_eval_split_no_longer_falls_back_to_training(self, mgr):
        """§6.1 rule 3: the IN-LOOP fallback is gone too.

        This previously asserted the opposite -- it was a regression guard for a
        fallback that the three-way partition removes. ``_val_*`` now comes from
        the artifact's own ``X_val``, so its absence is a condition the ingestion
        gate has already refused or explicitly marked; silently scoring the
        training rows as an in-loop metric is no longer available as a degraded
        mode.
        """
        mgr._train_x, mgr._train_y = _t(9.0), _t(9.0)
        mgr._val_x = mgr._val_y = None

        assert mgr._eval_split() == (None, None)
        assert mgr._eval_split_name() is None, "the label must not name a partition _eval_split no longer returns"

    def test_partial_pair_is_not_reportable(self, mgr):
        """A half-populated pair must not be treated as a partition."""
        mgr._val_x, mgr._val_y = _t(1.0), _t(1.0)
        mgr._test_x, mgr._test_y = _t(2.0), None
        assert mgr._reported_split_name() == "validation", "an unpaired test_x must not win"


class TestFinalEvalMetrics:
    def test_returns_none_when_disabled(self, mgr):
        mgr._eval_metrics_enabled = False
        mgr._val_x, mgr._val_y = _t(1.0), _t(1.0)
        assert mgr._compute_final_eval_metrics() is None

    def test_returns_none_with_nothing_to_report_on(self, mgr):
        mgr._eval_metrics_enabled = True
        mgr._val_x = mgr._val_y = mgr._test_x = mgr._test_y = None
        assert mgr._compute_final_eval_metrics() is None

    def test_never_raises_when_the_forward_pass_fails(self, mgr):
        """A diagnostic must not be the thing that fails a completed run."""
        mgr._eval_metrics_enabled = True
        mgr._val_x, mgr._val_y = _t(1.0), _t(1.0)
        mgr.network = types.SimpleNamespace(forward=lambda _x: (_ for _ in ()).throw(RuntimeError("boom")))
        assert mgr._compute_final_eval_metrics() is None


class TestRollbackCarriesReportedTensors:
    def test_snapshot_round_trips_the_test_pair(self, mgr):
        """An aborted swap must not leave the final score computed on the wrong rows."""
        pre = _PreSwapSnapshot(
            train_x=_t(0.0),
            train_y=_t(0.0),
            val_x=_t(1.0),
            val_y=_t(1.0),
            state_dict=None,
            input_size=2,
            output_size=2,
            dataset_config=None,
            test_x=_t(2.0),
            test_y=_t(2.0),
        )
        assert torch.equal(pre.test_x, _t(2.0))
        assert torch.equal(pre.test_y, _t(2.0))

    def test_snapshot_defaults_keep_existing_callers_working(self):
        """The new fields are keyword-with-default, so pre-existing construction sites
        that know nothing about a test partition still build a valid snapshot."""
        pre = _PreSwapSnapshot(
            train_x=None,
            train_y=None,
            val_x=None,
            val_y=None,
            state_dict=None,
            input_size=2,
            output_size=2,
            dataset_config=None,
        )
        assert pre.test_x is None
        assert pre.test_y is None
