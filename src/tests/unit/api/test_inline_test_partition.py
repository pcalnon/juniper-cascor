#!/usr/bin/env python
"""Unit coverage for the inline test-partition ingress (cascor#582).

cascor#614 built the reported-partition machinery (``_test_x``/``_test_y``,
``_reported_split``, ``_compute_final_eval_metrics``) but left the slot with no
way to be populated. This pins the first ingress that fills it, plus the three
ways filling it would otherwise have been silently useless:

* a request field that is accepted and discarded (``InlineDataset`` had no
  ``extra="forbid"``, so ``test_x`` 200'd and vanished — ml#1523 §2.5);
* a reported tensor that is never zero-padded, so the end-of-training forward
  pass raises into a deliberate bare ``except`` and the final score is simply
  ABSENT rather than wrong — the failure reads as "nothing to report";
* a live dataset swap that leaves the PREVIOUS dataset's held-out rows in the
  slot, scoring the new run against them under the label ``"test"``.

All fast unit tests; no network is trained.
"""

import pytest
import torch
from pydantic import ValidationError

from api.lifecycle.manager import TrainingLifecycleManager
from api.models.training import InlineDataset

pytestmark = pytest.mark.unit


@pytest.fixture
def mgr():
    m = TrainingLifecycleManager()
    try:
        yield m
    finally:
        m.shutdown()


def _rows(n: int, width: int, v: float) -> list:
    return [[v] * width for _ in range(n)]


class TestInlineDatasetAcceptsTestPartition:
    def test_test_pair_round_trips(self):
        d = InlineDataset(train_x=_rows(4, 2, 1.0), train_y=_rows(4, 1, 1.0), test_x=_rows(2, 2, 9.0), test_y=_rows(2, 1, 9.0))
        assert d.test_x == _rows(2, 2, 9.0)
        assert d.test_y == _rows(2, 1, 9.0)

    def test_test_pair_is_optional(self):
        d = InlineDataset(train_x=_rows(4, 2, 1.0), train_y=_rows(4, 1, 1.0))
        assert d.test_x is None and d.test_y is None

    @pytest.mark.parametrize("supplied,missing", [("test_x", "test_y"), ("test_y", "test_x")])
    def test_half_specified_test_split_is_rejected(self, supplied, missing):
        kwargs = {"train_x": _rows(4, 2, 1.0), "train_y": _rows(4, 1, 1.0), supplied: _rows(2, 2 if supplied == "test_x" else 1, 9.0)}
        with pytest.raises(ValidationError, match=missing):
            InlineDataset(**kwargs)

    def test_test_pair_length_mismatch_is_rejected(self):
        with pytest.raises(ValidationError, match="length mismatch"):
            InlineDataset(train_x=_rows(4, 2, 1.0), train_y=_rows(4, 1, 1.0), test_x=_rows(3, 2, 9.0), test_y=_rows(2, 1, 9.0))

    def test_val_split_rules_are_unchanged(self):
        """Regression guard: rewriting the validator as a loop must not relax val."""
        with pytest.raises(ValidationError, match="val_y"):
            InlineDataset(train_x=_rows(4, 2, 1.0), train_y=_rows(4, 1, 1.0), val_x=_rows(2, 2, 5.0))


class TestInlineDatasetForbidsUnknownKeys:
    def test_unknown_key_is_now_rejected(self):
        """THE point of ``extra="forbid"``: before it, this 200'd and dropped the key."""
        with pytest.raises(ValidationError):
            InlineDataset(train_x=_rows(4, 2, 1.0), train_y=_rows(4, 1, 1.0), tset_x=_rows(2, 2, 9.0))


class TestInlineDatasetWidthAlignment:
    """A wider-than-train split has no legal padding; catch it at the boundary."""

    def test_wider_test_x_is_rejected(self):
        with pytest.raises(ValidationError, match="feature count"):
            InlineDataset(train_x=_rows(4, 2, 1.0), train_y=_rows(4, 1, 1.0), test_x=_rows(2, 5, 9.0), test_y=_rows(2, 1, 9.0))

    def test_wider_test_y_is_rejected(self):
        with pytest.raises(ValidationError, match="target count"):
            InlineDataset(train_x=_rows(4, 2, 1.0), train_y=_rows(4, 1, 1.0), test_x=_rows(2, 2, 9.0), test_y=_rows(2, 4, 9.0))

    def test_narrower_split_is_allowed(self):
        """Narrower is legal — it is zero-padded up to the network dims downstream."""
        d = InlineDataset(train_x=_rows(4, 3, 1.0), train_y=_rows(4, 1, 1.0), test_x=_rows(2, 3, 9.0), test_y=_rows(2, 1, 9.0))
        assert d.test_x is not None


class TestStartTrainingPopulatesTheSlot:
    """Exercises the real ``start_training`` ingress without launching a run.

    The tensor assignments happen before the "Training data not provided" guard,
    so calling with no training data reaches the assignments and then raises —
    which is exactly the seam this needs, and keeps the fast unit lane free of
    training futures.
    """

    def test_x_test_reaches_the_manager(self, mgr):
        mgr._test_x = mgr._test_y = None
        x_test, y_test = torch.full((2, 2), 7.0), torch.full((2, 1), 7.0)
        with pytest.raises(ValueError, match="Training data not provided"):
            mgr.start_training(X=None, y=None, X_test=x_test, y_test=y_test)
        assert mgr._test_x is x_test, "X_test must land in the reported slot"
        assert mgr._test_y is y_test
        assert mgr._reported_split_name() == "test"

    def test_omitting_test_retains_the_current_pair(self, mgr):
        """Retain-on-omit, matching ``X_val``. A continue-training run keeps its partition."""
        keep_x, keep_y = torch.full((2, 2), 7.0), torch.full((2, 1), 7.0)
        mgr._test_x, mgr._test_y = keep_x, keep_y
        with pytest.raises(ValueError, match="Training data not provided"):
            mgr.start_training(X=None, y=None)
        assert mgr._test_x is keep_x, "omitting X_test must not clear the slot"
        assert mgr._test_y is keep_y

    def test_val_and_test_are_independent(self, mgr):
        """Supplying only one of the two must not disturb the other."""
        keep_x, keep_y = torch.full((2, 2), 7.0), torch.full((2, 1), 7.0)
        mgr._test_x, mgr._test_y = keep_x, keep_y
        val_x, val_y = torch.full((2, 2), 3.0), torch.full((2, 1), 3.0)
        with pytest.raises(ValueError, match="Training data not provided"):
            mgr.start_training(X=None, y=None, X_val=val_x, y_val=val_y)
        assert mgr._val_x is val_x
        assert mgr._test_x is keep_x


class TestReportedPairIsPadded:
    """The silent-absence failure: an unpadded test_x raises inside a swallowing except."""

    def _mgr_with_net(self, mgr, input_size, output_size):
        import types

        mgr.network = types.SimpleNamespace(input_size=input_size, output_size=output_size)
        return mgr

    def test_narrow_test_x_is_padded_up_to_network_input(self, mgr):
        self._mgr_with_net(mgr, 5, 2)
        mgr._test_x = torch.full((3, 3), 1.0)
        mgr._test_y = torch.full((3, 2), 1.0)
        mgr._pad_test_split_for_network()
        assert mgr._test_x.shape == (3, 5)
        assert torch.equal(mgr._test_x[:, :3], torch.full((3, 3), 1.0)), "original columns preserved"
        assert torch.equal(mgr._test_x[:, 3:], torch.zeros(3, 2)), "pad columns are zero"

    def test_narrow_test_y_is_padded_up_to_network_output(self, mgr):
        self._mgr_with_net(mgr, 2, 4)
        mgr._test_x = torch.full((3, 2), 1.0)
        mgr._test_y = torch.full((3, 1), 1.0)
        mgr._pad_test_split_for_network()
        assert mgr._test_y.shape == (3, 4)

    def test_matching_widths_pass_through_untouched(self, mgr):
        self._mgr_with_net(mgr, 2, 1)
        original = torch.full((3, 2), 1.0)
        mgr._test_x, mgr._test_y = original, torch.full((3, 1), 1.0)
        mgr._pad_test_split_for_network()
        assert mgr._test_x is original, "no-op must not copy"

    def test_unset_pair_is_a_no_op(self, mgr):
        self._mgr_with_net(mgr, 5, 2)
        mgr._test_x = mgr._test_y = None
        mgr._pad_test_split_for_network()
        assert mgr._test_x is None

    def test_no_network_is_a_no_op(self, mgr):
        mgr.network = None
        mgr._test_x = torch.full((3, 3), 1.0)
        mgr._pad_test_split_for_network()
        assert mgr._test_x.shape == (3, 3)

    def test_padded_pair_survives_the_final_forward(self, mgr):
        """End-to-end of the failure this exists to prevent.

        With the pad call REMOVED, ``forward`` raises on the shape mismatch,
        ``_compute_final_eval_metrics`` swallows it, and the assertion below sees
        ``None`` — a final score that is absent with no error anywhere.
        """
        import types

        net = types.SimpleNamespace(input_size=5, output_size=2)
        net.forward = lambda t: torch.zeros(t.shape[0], 2) if t.shape[1] == 5 else (_ for _ in ()).throw(RuntimeError(f"width {t.shape[1]} != 5"))
        mgr.network = net
        mgr._eval_metrics_enabled = True
        mgr._test_x = torch.full((4, 3), 1.0)
        mgr._test_y = torch.zeros(4, 2)
        mgr._pad_test_split_for_network()
        result = mgr._compute_final_eval_metrics()
        assert result is not None, "the final score must survive a padded network"
        assert result["split"] == "test"
        assert result["final"] is True


class TestDatasetSwapClearsTheReportedPair:
    """Drives the real ``_reload_dataset`` with a stubbed juniper-data client.

    Without the clear, the previous dataset's rows stay in ``_test_x`` and
    ``_reported_split_name()`` keeps returning ``"test"`` — the new run's final
    score computed on the OLD dataset's held-out rows, under an honest-looking
    label. That is worse than no score.
    """

    @staticmethod
    def _stub_client(monkeypatch, arrays):
        import sys
        import types as _types

        class _FakeClient:
            def __init__(self, *a, **kw):
                pass

            def create_dataset(self, **kw):
                return {"dataset_id": "stub-1"}

            def download_artifact_npz(self, dataset_id):
                return arrays

        monkeypatch.setitem(sys.modules, "juniper_data_client", _types.SimpleNamespace(JuniperDataClient=_FakeClient))

    @staticmethod
    def _three_way_arrays():
        import numpy as np

        return {
            "X_train": np.ones((4, 2), dtype=np.float32),
            "y_train": np.ones((4, 1), dtype=np.float32),
            "X_val": np.full((3, 2), 3.0, dtype=np.float32),
            "y_val": np.full((3, 1), 3.0, dtype=np.float32),
            "X_test": np.full((2, 2), 5.0, dtype=np.float32),
            "y_test": np.full((2, 1), 5.0, dtype=np.float32),
        }

    def test_reload_clears_a_stale_reported_pair(self, mgr, monkeypatch):
        """The stale pair must not survive a swap, and the new one must be the artifact's.

        Rewritten for the three-way partition: the artifact now carries its own
        ``X_val``, so ``_test_x`` is REPLACED by the new dataset's held-out rows
        rather than cleared to None. The staleness invariant is unchanged and is
        what the 7.0-valued rows check.
        """
        self._stub_client(monkeypatch, self._three_way_arrays())
        monkeypatch.setattr(mgr, "_translate_staged_config", lambda dt, cfg: ("spiral", {}))

        mgr._test_x, mgr._test_y = torch.full((9, 2), 7.0), torch.full((9, 1), 7.0)
        assert mgr._reported_split_name() == "test"

        mgr._reload_dataset(dataset_type="spiral")

        assert mgr._test_x.shape == (2, 2), "the artifact's X_test binds to the REPORTED slot"
        assert not torch.equal(mgr._test_x, torch.full((9, 2), 7.0)), "the old dataset's held-out rows must not survive"
        assert mgr._val_x.shape == (3, 2), "the artifact's X_val binds to the IN-LOOP slot"
        assert mgr._reported_split_name() == "test"
        assert mgr._eval_split_name() == "validation"
        assert not torch.equal(mgr._val_x, mgr._test_x), "the in-loop signal and the reported score must be different rows"
        assert mgr._validation_warning is None, "a three-way artifact carries no caveat"

    def test_reload_refuses_a_legacy_artifact_without_a_validation_split(self, mgr, monkeypatch):
        """§6.1 rule 2: no ``X_val`` must not silently promote ``X_test``.

        This is the defect the whole arc removes, so the default has to be a
        refusal rather than a warning nobody reads.
        """
        import numpy as np

        legacy = {
            "X_train": np.ones((4, 2), dtype=np.float32),
            "y_train": np.ones((4, 1), dtype=np.float32),
            "X_test": np.full((2, 2), 5.0, dtype=np.float32),
            "y_test": np.full((2, 1), 5.0, dtype=np.float32),
        }
        self._stub_client(monkeypatch, legacy)
        monkeypatch.setattr(mgr, "_translate_staged_config", lambda dt, cfg: ("spiral", {}))

        with pytest.raises(RuntimeError, match="no validation split"):
            mgr._reload_dataset(dataset_type="spiral")

    def test_reload_proceeds_with_a_caveat_when_explicitly_overridden(self, mgr, monkeypatch):
        """§6.4 option 1: the override proceeds, and MARKS the run."""
        import numpy as np

        legacy = {
            "X_train": np.ones((4, 2), dtype=np.float32),
            "y_train": np.ones((4, 1), dtype=np.float32),
            "X_test": np.full((2, 2), 5.0, dtype=np.float32),
            "y_test": np.full((2, 1), 5.0, dtype=np.float32),
        }
        self._stub_client(monkeypatch, legacy)
        monkeypatch.setattr(mgr, "_translate_staged_config", lambda dt, cfg: ("spiral", {}))
        monkeypatch.setenv("JUNIPER_CASCOR_ALLOW_MISSING_VALIDATION_SPLIT", "true")

        mgr._reload_dataset(dataset_type="spiral")

        assert mgr._val_x.shape == (2, 2), "the override promotes X_test to the in-loop slot"
        assert torch.equal(mgr._val_x, mgr._test_x), "which is exactly the selected-on condition the caveat describes"
        assert mgr._validation_warning is not None, "an overridden run MUST carry the caveat"
        assert "SELECTED-ON" in mgr._validation_warning


class TestEvalSplitNameExtraction:
    """Behaviour-preserving extraction of ``get_metrics``' inline conditional."""

    def test_validation_when_val_pair_present(self, mgr):
        mgr._val_x, mgr._val_y = torch.zeros(2, 2), torch.zeros(2, 1)
        assert mgr._eval_split_name() == "validation"

    def test_none_when_only_train_present(self, mgr):
        """§6.1 rule 3: a training-only state has no in-loop partition to name.

        This previously asserted ``"training"``. The label was honest about which
        rows it named, but naming training rows truthfully does not make them a
        usable in-loop signal, and the label must agree with ``_eval_split``,
        which no longer returns them.
        """
        mgr._val_x = mgr._val_y = None
        mgr._train_x, mgr._train_y = torch.zeros(2, 2), torch.zeros(2, 1)
        assert mgr._eval_split_name() is None

    def test_none_when_nothing_loaded(self, mgr):
        mgr._val_x = mgr._val_y = mgr._train_x = mgr._train_y = None
        assert mgr._eval_split_name() is None

    def test_a_test_partition_does_NOT_change_the_in_loop_label(self, mgr):
        """The invariant, from the label side: the reported partition is invisible
        to the in-loop signal. If populating ``_test_x`` moved this label, the two
        splits would not actually be separate."""
        mgr._val_x, mgr._val_y = torch.zeros(2, 2), torch.zeros(2, 1)
        mgr._test_x, mgr._test_y = torch.ones(2, 2), torch.ones(2, 1)
        assert mgr._eval_split_name() == "validation"
        assert mgr._reported_split_name() == "test"
