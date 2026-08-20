#!/usr/bin/env python
"""Unit tests for the lifecycle's patch_weights method (CAN-015h-1).

Covers:
- FSM gate: any non-Investigating state returns _PATCH_FSM_REJECTED.
- Target/field validation: unknown target or unknown field rejected.
- Output-layer patching: weights and bias both rewritable; shape
  mismatch rejected; NaN/Inf rejected; requires_grad preserved.
- Hidden-unit patching: weights and bias both rewritable;
  out-of-range index rejected; shape mismatch rejected.
- Optimizer-state zero-out: when an Adam state exists for the
  parameter, momentum/variance buffers are reset.
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


def _make_lifecycle_with_network(num_hidden=2):
    """Build a lifecycle + network and force-enter Investigating state.

    The patch endpoint requires INVESTIGATING (entered via /restore in
    production); for unit tests we set ``state_machine._status``
    directly so we don't have to round-trip through a snapshot.
    """
    config = CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=1,
        learning_rate=0.1,
        max_hidden_units=5,
        random_seed=42,
        init_output_weights="zero",
    )
    network = CascadeCorrelationNetwork(config=config)
    # Add hidden units via the helper from h-0 so output_weights is widened.
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


class TestPatchWeightsFSMGate:
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
    def test_non_investigating_states_rejected(self, status):
        lc = _make_lifecycle_with_network(num_hidden=1)
        lc.state_machine._status = status
        result = lc.patch_weights(target="output", field="weights", values=lc.network.output_weights.detach().tolist())
        assert result["status"] == lc._PATCH_FSM_REJECTED

    def test_investigating_state_allowed(self):
        lc = _make_lifecycle_with_network(num_hidden=1)
        # Already in INVESTIGATING from fixture
        result = lc.patch_weights(target="output", field="weights", values=lc.network.output_weights.detach().tolist())
        assert result["status"] == lc._PATCH_OK


# =============================================================================
# Target / field / index validation
# =============================================================================


class TestPatchWeightsValidation:
    def test_no_network_rejected(self):
        lc = TrainingLifecycleManager()
        lc.state_machine._status = TrainingStatus.INVESTIGATING
        result = lc.patch_weights(target="output", field="weights", values=[[0.0]])
        assert result["status"] == lc._PATCH_NO_NETWORK

    def test_unknown_target_rejected(self):
        lc = _make_lifecycle_with_network()
        result = lc.patch_weights(target="invalid", field="weights", values=[[0.0]])
        assert result["status"] == lc._PATCH_BAD_TARGET

    def test_unknown_field_rejected(self):
        lc = _make_lifecycle_with_network()
        result = lc.patch_weights(target="output", field="invalid", values=[[0.0]])
        assert result["status"] == lc._PATCH_BAD_TARGET

    def test_hidden_unit_index_required_for_hidden_target(self):
        lc = _make_lifecycle_with_network(num_hidden=2)
        unit_w = lc.network.hidden_units[0]["weights"].detach().tolist()
        result = lc.patch_weights(target="hidden_unit", field="weights", values=unit_w, hidden_unit_index=None)
        assert result["status"] == lc._PATCH_HIDDEN_UNIT_OUT_OF_RANGE

    def test_hidden_unit_index_out_of_range_rejected(self):
        lc = _make_lifecycle_with_network(num_hidden=2)
        unit_w = lc.network.hidden_units[0]["weights"].detach().tolist()
        result = lc.patch_weights(target="hidden_unit", field="weights", values=unit_w, hidden_unit_index=99)
        assert result["status"] == lc._PATCH_HIDDEN_UNIT_OUT_OF_RANGE

    def test_shape_mismatch_rejected(self):
        lc = _make_lifecycle_with_network()
        wrong_shape = [[1.0, 2.0, 3.0]]  # output_weights is [in+hid, out=1] not [1, 3]
        result = lc.patch_weights(target="output", field="weights", values=wrong_shape)
        assert result["status"] == lc._PATCH_SHAPE_MISMATCH

    def test_nan_value_rejected(self):
        lc = _make_lifecycle_with_network()
        shape = lc.network.output_weights.shape
        bad_values = torch.full(shape, float("nan")).tolist()
        result = lc.patch_weights(target="output", field="weights", values=bad_values)
        assert result["status"] == lc._PATCH_NAN_INF

    def test_inf_value_rejected(self):
        lc = _make_lifecycle_with_network()
        shape = lc.network.output_weights.shape
        bad_values = torch.full(shape, float("inf")).tolist()
        result = lc.patch_weights(target="output", field="weights", values=bad_values)
        assert result["status"] == lc._PATCH_NAN_INF

    def test_unparsable_values_rejected(self):
        lc = _make_lifecycle_with_network()
        # Ragged nested list — can't form a tensor.
        result = lc.patch_weights(target="output", field="weights", values=[[1, 2], [3]])
        assert result["status"] == lc._PATCH_NAN_INF


# =============================================================================
# Output-layer patches
# =============================================================================


class TestPatchOutputLayer:
    def test_output_weights_rewrite(self):
        lc = _make_lifecycle_with_network(num_hidden=1)
        shape = lc.network.output_weights.shape
        new_values = torch.full(shape, 0.42).tolist()
        result = lc.patch_weights(target="output", field="weights", values=new_values)
        assert result["status"] == lc._PATCH_OK
        np.testing.assert_array_almost_equal(
            lc.network.output_weights.detach().numpy(),
            torch.full(shape, 0.42).numpy(),
        )

    def test_output_weights_preserves_requires_grad(self):
        lc = _make_lifecycle_with_network(num_hidden=1)
        # output_weights starts with requires_grad=True after _install_hidden_unit.
        assert lc.network.output_weights.requires_grad is True
        new_values = torch.zeros_like(lc.network.output_weights).tolist()
        result = lc.patch_weights(target="output", field="weights", values=new_values)
        assert result["status"] == lc._PATCH_OK
        assert lc.network.output_weights.requires_grad is True

    def test_output_bias_rewrite(self):
        lc = _make_lifecycle_with_network()
        new_bias = torch.tensor([1.234], dtype=torch.float32).tolist()
        result = lc.patch_weights(target="output", field="bias", values=new_bias)
        assert result["status"] == lc._PATCH_OK
        np.testing.assert_array_almost_equal(lc.network.output_bias.detach().numpy(), [1.234])


# =============================================================================
# Hidden-unit patches
# =============================================================================


class TestPatchHiddenUnit:
    def test_hidden_unit_weights_rewrite(self):
        lc = _make_lifecycle_with_network(num_hidden=2)
        original_shape = lc.network.hidden_units[1]["weights"].shape
        new_values = torch.full(original_shape, 0.5).tolist()
        result = lc.patch_weights(target="hidden_unit", field="weights", values=new_values, hidden_unit_index=1)
        assert result["status"] == lc._PATCH_OK
        np.testing.assert_array_almost_equal(
            lc.network.hidden_units[1]["weights"].detach().numpy(),
            torch.full(original_shape, 0.5).numpy(),
        )

    def test_hidden_unit_bias_rewrite(self):
        lc = _make_lifecycle_with_network(num_hidden=2)
        original_shape = lc.network.hidden_units[0]["bias"].shape
        new_bias = torch.full(original_shape, 0.7).tolist()
        result = lc.patch_weights(target="hidden_unit", field="bias", values=new_bias, hidden_unit_index=0)
        assert result["status"] == lc._PATCH_OK
        np.testing.assert_array_almost_equal(
            lc.network.hidden_units[0]["bias"].detach().numpy(),
            torch.full(original_shape, 0.7).numpy(),
        )

    def test_hidden_unit_shape_mismatch(self):
        lc = _make_lifecycle_with_network(num_hidden=2)
        # Unit 1's weights are [in+1] = [3] but pass [5] → mismatch.
        result = lc.patch_weights(
            target="hidden_unit",
            field="weights",
            values=[1.0, 2.0, 3.0, 4.0, 5.0],
            hidden_unit_index=1,
        )
        assert result["status"] == lc._PATCH_SHAPE_MISMATCH


# =============================================================================
# Optimizer state zero-out
# =============================================================================


class TestOptimizerStateZeroOut:
    def test_zeros_adam_state_for_patched_parameter(self):
        lc = _make_lifecycle_with_network(num_hidden=1)
        # Build a fake Adam-shaped optimizer state keyed by the
        # CURRENT output_weights tensor (the one that's about to be
        # replaced). Zero-out runs against this key BEFORE the
        # reassignment per the lifecycle implementation.
        param = lc.network.output_weights
        fake_optimizer = MagicMock()
        fake_optimizer.state = {
            param: {
                "step": torch.tensor(42),
                "exp_avg": torch.full_like(param, 0.5),
                "exp_avg_sq": torch.full_like(param, 0.25),
            }
        }
        lc.network.output_optimizer = fake_optimizer

        new_values = torch.full_like(param, 0.1).tolist()
        result = lc.patch_weights(target="output", field="weights", values=new_values)
        assert result["status"] == lc._PATCH_OK

        # ``param`` is still a Python reference to the old tensor.
        zero_state = fake_optimizer.state[param]
        assert torch.all(zero_state["exp_avg"] == 0)
        assert torch.all(zero_state["exp_avg_sq"] == 0)
        # Step counter is preserved (only running statistics are bias-affected).
        assert int(zero_state["step"]) == 42

    def test_zeros_adam_state_for_hidden_unit(self):
        lc = _make_lifecycle_with_network(num_hidden=2)
        # State keyed by the hidden-unit weight tensor.
        unit_w = lc.network.hidden_units[1]["weights"]
        fake_optimizer = MagicMock()
        fake_optimizer.state = {
            unit_w: {
                "exp_avg": torch.full_like(unit_w, 0.3),
                "exp_avg_sq": torch.full_like(unit_w, 0.1),
            }
        }
        lc.network.output_optimizer = fake_optimizer

        new_values = torch.zeros_like(unit_w).tolist()
        result = lc.patch_weights(target="hidden_unit", field="weights", values=new_values, hidden_unit_index=1)
        assert result["status"] == lc._PATCH_OK
        zero_state = fake_optimizer.state[unit_w]
        assert torch.all(zero_state["exp_avg"] == 0)
        assert torch.all(zero_state["exp_avg_sq"] == 0)

    def test_no_optimizer_attribute_is_noop(self):
        lc = _make_lifecycle_with_network()
        # No output_optimizer attribute set — patch must still succeed.
        if hasattr(lc.network, "output_optimizer"):
            delattr(lc.network, "output_optimizer")
        new_values = torch.zeros_like(lc.network.output_weights).tolist()
        result = lc.patch_weights(target="output", field="weights", values=new_values)
        assert result["status"] == lc._PATCH_OK

    def test_optimizer_state_missing_for_param_is_noop(self):
        lc = _make_lifecycle_with_network()
        # Optimizer exists but doesn't track this parameter — must
        # not raise. (Real-world: parameter was added after
        # optimizer was built, e.g. cascade-grow without re-init.)
        fake_optimizer = MagicMock()
        fake_optimizer.state = {}
        lc.network.output_optimizer = fake_optimizer
        new_values = torch.zeros_like(lc.network.output_weights).tolist()
        result = lc.patch_weights(target="output", field="weights", values=new_values)
        assert result["status"] == lc._PATCH_OK


class TestZeroOptimizerStateWithRealOptimizer:
    """The same zero-out, against the optimizer production actually builds.

    Every other test in this file keys a mock optimizer's ``state`` by the
    network-level tensor (``network.output_weights``). Production never does:
    ``train_output_layer`` builds an ``nn.Linear`` and hands ITS ``Parameter``\\ s to
    the optimizer, so the identity lookup could never match and the zero-out was a
    permanent no-op that the mocks reported as working. That was harmless only
    while the optimizer was thrown away on every call; now that a resume carries
    the moments forward (R3), an edit stepped with pre-edit moments is a real
    defect. These tests use a REAL optimizer so the correspondence is exercised.
    """

    @staticmethod
    def _warm_real_optimizer(lc):
        """Give the network a genuine, stepped ``output_optimizer``."""
        x = torch.randn(8, 2, dtype=torch.float32)
        y = torch.randn(8, 1, dtype=torch.float32)
        lc.network.train_output_layer(x, y, epochs=3)
        optimizer = lc.network.output_optimizer
        # Fill the running buffers with a recognizably non-zero pattern.
        for entry in optimizer.state.values():
            for key, val in entry.items():
                if isinstance(val, torch.Tensor) and val.dim() > 0:
                    entry[key] = torch.full_like(val, 0.5)
        return optimizer

    def _slot_state(self, optimizer, slot):
        params = [p for group in optimizer.param_groups for p in group["params"]]
        return optimizer.state[params[slot]]

    def test_patching_output_weights_zeroes_the_real_slot(self):
        lc = _make_lifecycle_with_network()
        optimizer = self._warm_real_optimizer(lc)
        assert torch.count_nonzero(self._slot_state(optimizer, 0)["exp_avg"]) > 0

        new_values = torch.zeros_like(lc.network.output_weights).tolist()
        result = lc.patch_weights(target="output", field="weights", values=new_values)

        assert result["status"] == lc._PATCH_OK
        weight_state = self._slot_state(optimizer, 0)
        assert torch.all(weight_state["exp_avg"] == 0), "pre-edit moments survived an output-weight patch"
        assert torch.all(weight_state["exp_avg_sq"] == 0)
        # The bias slot was not edited, so its moments must be left alone.
        assert torch.count_nonzero(self._slot_state(optimizer, 1)["exp_avg"]) > 0, "patching weights must not clear the bias slot"

    def test_patching_output_bias_zeroes_only_the_bias_slot(self):
        lc = _make_lifecycle_with_network()
        optimizer = self._warm_real_optimizer(lc)

        new_values = torch.zeros_like(lc.network.output_bias).tolist()
        result = lc.patch_weights(target="output", field="bias", values=new_values)

        assert result["status"] == lc._PATCH_OK
        assert torch.all(self._slot_state(optimizer, 1)["exp_avg"] == 0), "pre-edit moments survived an output-bias patch"
        assert torch.count_nonzero(self._slot_state(optimizer, 0)["exp_avg"]) > 0, "patching bias must not clear the weight slot"

    def test_step_counter_is_preserved_across_a_patch(self):
        """Only the running statistics carry pre-edit bias; the step count does not."""
        lc = _make_lifecycle_with_network()
        optimizer = self._warm_real_optimizer(lc)
        before = float(self._slot_state(optimizer, 0)["step"])

        result = lc.patch_weights(target="output", field="weights", values=torch.zeros_like(lc.network.output_weights).tolist())

        assert result["status"] == lc._PATCH_OK
        assert float(self._slot_state(optimizer, 0)["step"]) == before
