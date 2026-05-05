#!/usr/bin/env python
"""Unit tests for remove_hidden_unit_manual (CAN-015h-3).

Cascade-rebuild semantics are subtle — the test surface specifically
asserts:

- Removing the **tail** unit is structurally identical to truncating
  output_weights at the last row.
- Removing a **middle** unit drops the corresponding column from
  ``output_weights`` AND drops the corresponding weight from each
  subsequent unit's weight vector. After the surgery, every unit's
  weight vector length matches its new cascade position.
- The forward-pass shape invariant holds post-delete: a fresh call
  to ``_compute_hidden_outputs`` succeeds without raising.
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.lifecycle.manager import TrainingLifecycleManager  # noqa: E402
from api.lifecycle.state_machine import TrainingStatus  # noqa: E402
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork  # noqa: E402
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import (  # noqa: E402
    CascadeCorrelationConfig,
)

pytestmark = pytest.mark.unit


def _make_lifecycle(num_hidden=3, in_size=2, out_size=1):
    config = CascadeCorrelationConfig.create_simple_config(
        input_size=in_size,
        output_size=out_size,
        learning_rate=0.1,
        max_hidden_units=10,
        random_seed=42,
        init_output_weights="zero",
    )
    network = CascadeCorrelationNetwork(config=config)
    for i in range(num_hidden):
        prev_in = network.output_weights.shape[0]
        # Deterministic weights so we can verify the post-delete
        # state by inspection.
        weights = torch.arange(in_size + i, dtype=torch.float32) * 0.1 + i
        network._install_hidden_unit(
            weights=weights,
            bias=torch.tensor([float(i)], dtype=torch.float32),
            activation_fn=network.activation_fn,
            correlation=0.5 * (i + 1),
        )
        network._resize_output_layer_for_new_units(num_added=1, prev_input_size=prev_in)
    lifecycle = TrainingLifecycleManager()
    lifecycle.network = network
    lifecycle.state_machine._status = TrainingStatus.INVESTIGATING
    return lifecycle


# =============================================================================
# FSM gate + range
# =============================================================================


class TestRemoveFSMGate:
    @pytest.mark.parametrize(
        "status",
        [
            TrainingStatus.STOPPED,
            TrainingStatus.STARTED,
            TrainingStatus.PAUSED,
            TrainingStatus.COMPLETED,
            TrainingStatus.FAILED,
            TrainingStatus.RESUME_READY,
            TrainingStatus.REPLAYING,
        ],
    )
    def test_non_investigating_rejected(self, status):
        lc = _make_lifecycle(num_hidden=2)
        lc.state_machine._status = status
        result = lc.remove_hidden_unit_manual(idx=0)
        assert result["status"] == lc._REMOVE_FSM_REJECTED


class TestRemoveValidation:
    def test_no_network_rejected(self):
        lc = TrainingLifecycleManager()
        lc.state_machine._status = TrainingStatus.INVESTIGATING
        result = lc.remove_hidden_unit_manual(idx=0)
        assert result["status"] == lc._REMOVE_NO_NETWORK

    def test_negative_idx_rejected(self):
        lc = _make_lifecycle(num_hidden=2)
        result = lc.remove_hidden_unit_manual(idx=-1)
        assert result["status"] == lc._REMOVE_OUT_OF_RANGE

    def test_idx_at_count_rejected(self):
        lc = _make_lifecycle(num_hidden=2)
        result = lc.remove_hidden_unit_manual(idx=2)
        assert result["status"] == lc._REMOVE_OUT_OF_RANGE

    def test_remove_when_empty_rejected(self):
        lc = _make_lifecycle(num_hidden=0)
        result = lc.remove_hidden_unit_manual(idx=0)
        assert result["status"] == lc._REMOVE_OUT_OF_RANGE


# =============================================================================
# Tail removal
# =============================================================================


class TestRemoveTail:
    def test_remove_tail_succeeds(self):
        lc = _make_lifecycle(num_hidden=3, in_size=2)
        result = lc.remove_hidden_unit_manual(idx=2)  # last unit
        assert result["status"] == lc._REMOVE_OK
        assert result["removed_index"] == 2
        assert result["num_hidden_units"] == 2
        assert len(lc.network.hidden_units) == 2
        # output_weights was [in+3, out=1] = [5, 1]; now [4, 1].
        assert lc.network.output_weights.shape == (4, 1)

    def test_tail_remove_preserves_remaining_units_unchanged(self):
        lc = _make_lifecycle(num_hidden=3, in_size=2)
        prev_unit_0_w = lc.network.hidden_units[0]["weights"].detach().clone()
        prev_unit_1_w = lc.network.hidden_units[1]["weights"].detach().clone()
        lc.remove_hidden_unit_manual(idx=2)
        # Unit 0 weights unchanged (it never referenced the removed unit).
        np.testing.assert_array_equal(
            lc.network.hidden_units[0]["weights"].detach().numpy(),
            prev_unit_0_w.numpy(),
        )
        # Unit 1 weights unchanged (referenced the removed unit's INPUT,
        # which was at index 2+1=3 — but unit 1's weight vector has
        # length 3, not 4, so it never referenced the removed unit).
        np.testing.assert_array_equal(
            lc.network.hidden_units[1]["weights"].detach().numpy(),
            prev_unit_1_w.numpy(),
        )


# =============================================================================
# Middle removal — cascade rebuild
# =============================================================================


class TestRemoveMiddle:
    def test_remove_middle_drops_subsequent_unit_weights(self):
        # 3 units, in_size=2:
        #   unit 0 weights: length 2  -> [in_0, in_1]
        #   unit 1 weights: length 3  -> [in_0, in_1, h_0]
        #   unit 2 weights: length 4  -> [in_0, in_1, h_0, h_1]
        # Remove unit 1 (idx=1):
        #   col_to_drop = input_size + idx = 2 + 1 = 3
        #   unit 0 unchanged (length 2; index 3 is out of bounds — skipped)
        #   unit 2 (now becomes new unit 1): drop weight at index 3,
        #     so its weights become length 3. But length 3 = input_size + 1
        #     which matches its new cascade position. Good.
        lc = _make_lifecycle(num_hidden=3, in_size=2)
        prev_unit_2_w = lc.network.hidden_units[2]["weights"].detach().clone()
        result = lc.remove_hidden_unit_manual(idx=1)
        assert result["status"] == lc._REMOVE_OK
        assert len(lc.network.hidden_units) == 2

        # The "new unit 1" is what was unit 2. Its weights should be
        # the original unit 2's weights with index 3 dropped (i.e.,
        # entries [0, 1, 2] preserved, entry 3 dropped).
        new_unit_1_w = lc.network.hidden_units[1]["weights"].detach()
        expected_w = torch.cat([prev_unit_2_w[:3], prev_unit_2_w[4:]])  # only 4 entries; drop idx 3
        np.testing.assert_array_almost_equal(new_unit_1_w.numpy(), expected_w.numpy())
        # Length matches new cascade position: input_size(2) + 1 = 3.
        assert new_unit_1_w.shape == (3,)

    def test_remove_first_drops_correct_column(self):
        # 3 units. Remove unit 0 (col_to_drop = 2):
        #   unit 0 deleted.
        #   unit 1 (new unit 0): had weights [in_0, in_1, h_0]; drop h_0
        #     (index 2). New weights: length 2 = input_size + 0. Good.
        #   unit 2 (new unit 1): had weights [in_0, in_1, h_0, h_1]; drop h_0
        #     (index 2). New weights: length 3 = input_size + 1. Good.
        lc = _make_lifecycle(num_hidden=3, in_size=2)
        prev_unit_1_w = lc.network.hidden_units[1]["weights"].detach().clone()
        prev_unit_2_w = lc.network.hidden_units[2]["weights"].detach().clone()
        result = lc.remove_hidden_unit_manual(idx=0)
        assert result["status"] == lc._REMOVE_OK
        assert len(lc.network.hidden_units) == 2

        new_unit_0_w = lc.network.hidden_units[0]["weights"].detach()
        # Was prev_unit_1_w with index 2 dropped → length 2.
        np.testing.assert_array_almost_equal(new_unit_0_w.numpy(), prev_unit_1_w[[0, 1]].numpy())
        new_unit_1_w = lc.network.hidden_units[1]["weights"].detach()
        # Was prev_unit_2_w with index 2 dropped → length 3.
        np.testing.assert_array_almost_equal(
            new_unit_1_w.numpy(),
            torch.cat([prev_unit_2_w[:2], prev_unit_2_w[3:]]).numpy(),
        )

    def test_post_delete_forward_pass_succeeds(self):
        # The cascade-input shape invariant must hold post-delete:
        # _compute_hidden_outputs() should run without raising.
        lc = _make_lifecycle(num_hidden=3, in_size=2)
        lc.remove_hidden_unit_manual(idx=1)
        x = torch.randn(4, 2)
        # Smoke test: must not raise.
        outputs = lc.network._compute_hidden_outputs(x)
        # Shape: [batch, input_size + num_hidden_after] = [4, 4]
        assert outputs.shape == (4, 2 + len(lc.network.hidden_units))


# =============================================================================
# Output weights surgery
# =============================================================================


class TestRemoveOutputWeights:
    def test_output_weights_column_dropped(self):
        lc = _make_lifecycle(num_hidden=3, in_size=2)
        # Stamp a recognizable pattern: row k has value k+10.
        with torch.no_grad():
            for k in range(lc.network.output_weights.shape[0]):
                lc.network.output_weights[k, :] = k + 10
        lc.remove_hidden_unit_manual(idx=1)  # col_to_drop = 3
        rows = lc.network.output_weights.detach().numpy().flatten()
        # Should be values 10, 11, 12, 14 (skip 13 which was the dropped unit).
        np.testing.assert_array_almost_equal(rows, [10.0, 11.0, 12.0, 14.0])

    def test_optimizer_dropped(self):
        lc = _make_lifecycle(num_hidden=2)
        lc.network.output_optimizer = MagicMock()
        result = lc.remove_hidden_unit_manual(idx=0)
        assert result["status"] == lc._REMOVE_OK
        assert lc.network.output_optimizer is None

    def test_requires_grad_preserved(self):
        lc = _make_lifecycle(num_hidden=2)
        # output_weights starts with requires_grad=True.
        assert lc.network.output_weights.requires_grad is True
        lc.remove_hidden_unit_manual(idx=0)
        assert lc.network.output_weights.requires_grad is True
