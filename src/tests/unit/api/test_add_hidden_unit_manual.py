#!/usr/bin/env python
"""Unit tests for add_hidden_unit_manual (CAN-015h-2)."""

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


def _make_lifecycle(num_hidden=1, max_hidden_units=5):
    """Build a network in INVESTIGATING with ``num_hidden`` units already installed."""
    config = CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=1,
        learning_rate=0.1,
        max_hidden_units=max_hidden_units,
        random_seed=42,
        init_output_weights="random",  # exercises the zero-init override
    )
    network = CascadeCorrelationNetwork(config=config)
    for i in range(num_hidden):
        prev_in = network.output_weights.shape[0]
        network._install_hidden_unit(
            weights=torch.randn(2 + i, dtype=torch.float32),
            bias=torch.tensor([0.0], dtype=torch.float32),
            activation_fn=network.activation_fn,
            correlation=0.5,
        )
        network._resize_output_layer_for_new_units(num_added=1, prev_input_size=prev_in)
    lifecycle = TrainingLifecycleManager()
    lifecycle.network = network
    lifecycle.state_machine._status = TrainingStatus.INVESTIGATING
    return lifecycle


# =============================================================================
# FSM gate
# =============================================================================


class TestAddHiddenUnitFSMGate:
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
        lc = _make_lifecycle(num_hidden=1)
        lc.state_machine._status = status
        prev_in = lc.network.output_weights.shape[0]
        result = lc.add_hidden_unit_manual(weights=[0.0] * prev_in)
        assert result["status"] == lc._ADD_FSM_REJECTED

    def test_investigating_allowed(self):
        lc = _make_lifecycle(num_hidden=1)
        prev_in = lc.network.output_weights.shape[0]
        result = lc.add_hidden_unit_manual(weights=[0.0] * prev_in)
        assert result["status"] == lc._ADD_OK


# =============================================================================
# Validation
# =============================================================================


class TestAddHiddenUnitValidation:
    def test_no_network_rejected(self):
        lc = TrainingLifecycleManager()
        lc.state_machine._status = TrainingStatus.INVESTIGATING
        result = lc.add_hidden_unit_manual(weights=[0.0, 0.0])
        assert result["status"] == lc._ADD_NO_NETWORK

    def test_at_cap_rejected(self):
        lc = _make_lifecycle(num_hidden=2, max_hidden_units=2)
        prev_in = lc.network.output_weights.shape[0]
        result = lc.add_hidden_unit_manual(weights=[0.0] * prev_in)
        assert result["status"] == lc._ADD_AT_CAP

    def test_unknown_activation_rejected(self):
        lc = _make_lifecycle()
        prev_in = lc.network.output_weights.shape[0]
        result = lc.add_hidden_unit_manual(weights=[0.0] * prev_in, activation="exotic")
        assert result["status"] == lc._ADD_BAD_ACTIVATION

    def test_nan_weights_rejected(self):
        lc = _make_lifecycle()
        prev_in = lc.network.output_weights.shape[0]
        result = lc.add_hidden_unit_manual(weights=[float("nan")] * prev_in)
        assert result["status"] == lc._ADD_NAN_INF

    def test_inf_bias_rejected(self):
        lc = _make_lifecycle()
        prev_in = lc.network.output_weights.shape[0]
        result = lc.add_hidden_unit_manual(weights=[0.0] * prev_in, bias=float("inf"))
        assert result["status"] == lc._ADD_NAN_INF

    def test_shape_mismatch_rejected(self):
        lc = _make_lifecycle(num_hidden=1)
        # Expected length is 3 (in=2 + 1 hidden); pass length 5 → mismatch.
        result = lc.add_hidden_unit_manual(weights=[0.0, 0.0, 0.0, 0.0, 0.0])
        assert result["status"] == lc._ADD_BAD_SHAPE

    def test_2d_weights_rejected(self):
        lc = _make_lifecycle()
        prev_in = lc.network.output_weights.shape[0]
        result = lc.add_hidden_unit_manual(weights=[[0.0] * prev_in])
        assert result["status"] == lc._ADD_BAD_SHAPE


# =============================================================================
# Successful append
# =============================================================================


class TestAddHiddenUnitSuccess:
    def test_appends_unit_and_resizes(self):
        lc = _make_lifecycle(num_hidden=1)
        n_before = len(lc.network.hidden_units)
        prev_in = lc.network.output_weights.shape[0]
        result = lc.add_hidden_unit_manual(weights=[1.0] * prev_in, bias=0.5, activation="Tanh")
        assert result["status"] == lc._ADD_OK
        assert result["unit_index"] == n_before
        assert result["num_hidden_units"] == n_before + 1
        assert len(lc.network.hidden_units) == n_before + 1
        # Output layer widened by exactly one row.
        assert lc.network.output_weights.shape == (prev_in + 1, lc.network.output_size)

    def test_new_output_column_is_zero(self):
        # Even though network was constructed with init_output_weights="random",
        # the manual-append path must force zero-init for the new column.
        lc = _make_lifecycle(num_hidden=0)
        prev_in = lc.network.output_weights.shape[0]
        # Stamp a recognizable pattern into the existing output weights.
        with torch.no_grad():
            lc.network.output_weights.copy_(torch.full_like(lc.network.output_weights, 7.0))
        result = lc.add_hidden_unit_manual(weights=[0.5, -0.5])
        assert result["status"] == lc._ADD_OK
        # New row (index prev_in) must be zero.
        new_row = lc.network.output_weights[prev_in:, :].detach().numpy()
        np.testing.assert_array_equal(new_row, np.zeros_like(new_row))
        # Old rows must be preserved at the stamped pattern.
        old_rows = lc.network.output_weights[:prev_in, :].detach().numpy()
        np.testing.assert_array_almost_equal(old_rows, np.full_like(old_rows, 7.0))

    def test_init_output_weights_config_preserved(self):
        # The temporary "zero" override must restore the original setting.
        lc = _make_lifecycle()
        original = lc.network.init_output_weights
        prev_in = lc.network.output_weights.shape[0]
        lc.add_hidden_unit_manual(weights=[0.0] * prev_in)
        assert lc.network.init_output_weights == original

    def test_history_records_zero_correlation(self):
        # Manual inserts have undefined correlation; sentinel 0.0.
        lc = _make_lifecycle()
        n_history_before = len(lc.network.history["hidden_units_added"])
        prev_in = lc.network.output_weights.shape[0]
        lc.add_hidden_unit_manual(weights=[0.0] * prev_in)
        assert len(lc.network.history["hidden_units_added"]) == n_history_before + 1
        assert lc.network.history["hidden_units_added"][-1]["correlation"] == 0.0

    def test_optimizer_dropped_after_append(self):
        lc = _make_lifecycle()
        lc.network.output_optimizer = MagicMock()
        prev_in = lc.network.output_weights.shape[0]
        result = lc.add_hidden_unit_manual(weights=[0.0] * prev_in)
        assert result["status"] == lc._ADD_OK
        assert lc.network.output_optimizer is None

    def test_two_sequential_appends(self):
        lc = _make_lifecycle(num_hidden=0)
        # First append: weights of length 2 (in_size).
        r1 = lc.add_hidden_unit_manual(weights=[0.5, -0.5])
        assert r1["status"] == lc._ADD_OK
        assert r1["unit_index"] == 0
        # Second append: weights of length 3 (in_size + 1).
        r2 = lc.add_hidden_unit_manual(weights=[0.1, 0.2, 0.3])
        assert r2["status"] == lc._ADD_OK
        assert r2["unit_index"] == 1
        assert lc.network.output_weights.shape == (4, lc.network.output_size)
