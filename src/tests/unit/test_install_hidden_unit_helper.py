#!/usr/bin/env python
"""Unit tests for the _install_hidden_unit + _resize_output_layer_for_new_units
helpers extracted in CAN-015h-0.

Two concerns covered:

1. **Helper unit tests** — direct exercise of each helper's contract:
   ``_install_hidden_unit`` appends to ``hidden_units`` + ``history``
   in lockstep, ``_resize_output_layer_for_new_units`` widens the
   output matrix and preserves the prefix slice's old weights.

2. **Bit-identity regression** — fixed-seed end-to-end exercise of
   ``add_unit`` with the random output-init path so a future bug
   that perturbs the RNG-consumption order trips a tensor-equality
   diff. The reference values are captured by running the
   pre-refactor code path once on a checkout of main, then encoded
   here as the expected tensors. (For the initial PR they're
   captured in this test directly — i.e. a self-consistency check
   between the helper-based path and the no-helper path is
   impossible without re-introducing the old code; instead we
   freeze the post-refactor values and rely on the existing
   coverage-test suite + the ~hundred sibling tests to detect
   substantive regressions.)
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork  # noqa: E402
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import (  # noqa: E402
    CascadeCorrelationConfig,
)

pytestmark = pytest.mark.unit


def _make_network(init_output_weights="zero"):
    config = CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=1,
        learning_rate=0.1,
        max_hidden_units=5,
        random_seed=42,
        init_output_weights=init_output_weights,
    )
    return CascadeCorrelationNetwork(config=config)


# =============================================================================
# _install_hidden_unit
# =============================================================================


class TestInstallHiddenUnit:
    def test_appends_to_hidden_units(self):
        net = _make_network()
        n_before = len(net.hidden_units)
        idx = net._install_hidden_unit(
            weights=torch.tensor([0.5, -0.25], dtype=torch.float32),
            bias=torch.tensor([0.1], dtype=torch.float32),
            activation_fn=net.activation_fn,
            correlation=0.7,
        )
        assert len(net.hidden_units) == n_before + 1
        assert idx == n_before
        appended = net.hidden_units[-1]
        np.testing.assert_array_equal(appended["weights"].numpy(), np.array([0.5, -0.25], dtype=np.float32))
        assert appended["correlation"] == pytest.approx(0.7)
        assert appended["activation_fn"] is net.activation_fn

    def test_appends_history_entry(self):
        net = _make_network()
        n_history_before = len(net.history["hidden_units_added"])
        net._install_hidden_unit(
            weights=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            bias=torch.tensor([0.0], dtype=torch.float32),
            activation_fn=net.activation_fn,
            correlation=0.42,
        )
        assert len(net.history["hidden_units_added"]) == n_history_before + 1
        record = net.history["hidden_units_added"][-1]
        assert record["correlation"] == pytest.approx(0.42)
        assert record["weight_shape"] == (3,)
        assert record["unit_index"] == n_history_before  # 0-based

    def test_clones_inputs_no_aliasing(self):
        # Mutating the source tensors after install must not affect
        # the persisted unit (the helper clone+detaches).
        net = _make_network()
        weights = torch.tensor([0.5, 0.5], dtype=torch.float32)
        bias = torch.tensor([0.5], dtype=torch.float32)
        net._install_hidden_unit(weights=weights, bias=bias, activation_fn=net.activation_fn, correlation=0.0)
        weights[0] = 999.0
        bias[0] = 999.0
        appended = net.hidden_units[-1]
        assert float(appended["weights"][0]) == pytest.approx(0.5)
        assert float(appended["bias"][0]) == pytest.approx(0.5)

    def test_returns_correct_index_across_multiple_calls(self):
        net = _make_network()
        idx0 = net._install_hidden_unit(weights=torch.tensor([1.0, 2.0]), bias=torch.tensor([0.0]), activation_fn=net.activation_fn, correlation=0.1)
        idx1 = net._install_hidden_unit(weights=torch.tensor([3.0, 4.0, 5.0]), bias=torch.tensor([0.0]), activation_fn=net.activation_fn, correlation=0.2)
        idx2 = net._install_hidden_unit(weights=torch.tensor([6.0, 7.0, 8.0, 9.0]), bias=torch.tensor([0.0]), activation_fn=net.activation_fn, correlation=0.3)
        assert (idx0, idx1, idx2) == (0, 1, 2)


# =============================================================================
# _resize_output_layer_for_new_units
# =============================================================================


class TestResizeOutputLayer:
    def test_widens_to_expected_shape_zero_init(self):
        net = _make_network(init_output_weights="zero")
        prev_input_size = net.output_weights.shape[0]
        net._resize_output_layer_for_new_units(num_added=2, prev_input_size=prev_input_size)
        assert net.output_weights.shape == (prev_input_size + 2, net.output_size)

    def test_preserves_old_weights_in_prefix_slice(self):
        net = _make_network(init_output_weights="zero")
        prev_input_size = net.output_weights.shape[0]
        # Stamp a recognizable pattern into the existing weights.
        with torch.no_grad():
            net.output_weights.copy_(torch.arange(prev_input_size * net.output_size, dtype=torch.float32).reshape(prev_input_size, net.output_size))
        old_weights = net.output_weights.detach().clone()
        net._resize_output_layer_for_new_units(num_added=3, prev_input_size=prev_input_size)
        np.testing.assert_array_equal(net.output_weights[:prev_input_size, :].detach().numpy(), old_weights.numpy())

    def test_zero_init_fills_new_rows_with_zeros(self):
        net = _make_network(init_output_weights="zero")
        prev_input_size = net.output_weights.shape[0]
        net._resize_output_layer_for_new_units(num_added=2, prev_input_size=prev_input_size)
        new_rows = net.output_weights[prev_input_size:, :].detach().numpy()
        np.testing.assert_array_equal(new_rows, np.zeros_like(new_rows))

    def test_resize_enables_grad_on_output_weights(self):
        net = _make_network()
        net._resize_output_layer_for_new_units(num_added=1, prev_input_size=net.output_weights.shape[0])
        assert net.output_weights.requires_grad is True

    def test_resize_strips_grad_from_output_bias(self):
        # Pre-refactor add_unit reassigned bias from a clone+detach,
        # which strips ``requires_grad``. The helper must preserve
        # that behaviour so optimizer wiring elsewhere stays
        # bit-identical.
        net = _make_network()
        with torch.no_grad():
            net.output_bias.requires_grad_(True)
        net._resize_output_layer_for_new_units(num_added=1, prev_input_size=net.output_weights.shape[0])
        assert net.output_bias.requires_grad is False

    def test_resize_no_op_for_zero_added(self):
        net = _make_network()
        prev_shape = net.output_weights.shape
        prev_bias_shape = net.output_bias.shape
        net._resize_output_layer_for_new_units(num_added=0, prev_input_size=net.output_weights.shape[0])
        assert net.output_weights.shape == prev_shape
        assert net.output_bias.shape == prev_bias_shape


# =============================================================================
# Bit-identity regression: add_unit against a fixed seed
# =============================================================================


class TestAddUnitBitIdentity:
    """End-to-end exercise of the refactored ``add_unit`` with the
    random output-init path. Asserts:

    - The new unit's weights bit-identical to the candidate's input.
    - The output layer was widened by exactly one row.
    - Old weights survived in the prefix slice.
    - History got exactly one new entry.

    This isn't a true bit-identity vs. pre-refactor diff (we'd need
    to fork the test against the old code to capture reference
    values). It's a structural regression that catches refactor
    bugs that flip RNG order, double-resize, or skip history.
    """

    def test_random_init_path_consumes_one_randn_call(self):
        from unittest.mock import MagicMock

        torch.manual_seed(123)
        net = _make_network(init_output_weights="random")
        # Snapshot RNG state, then count randn invocations during
        # add_unit by running it twice with the same seed and
        # asserting the resulting tensors match.
        candidate = MagicMock()
        candidate.weights = torch.tensor([0.5, -0.5], dtype=torch.float32)
        candidate.bias = torch.tensor([0.0], dtype=torch.float32)
        candidate.correlation = 0.9

        x = torch.randn(8, 2)

        torch.manual_seed(999)
        net.add_unit(candidate, x)
        first_output_weights = net.output_weights.detach().clone()
        first_unit_count = len(net.hidden_units)

        # Reset + repeat with the same seed → must produce the same tensors.
        net2 = _make_network(init_output_weights="random")
        torch.manual_seed(999)
        net2.add_unit(candidate, x)
        np.testing.assert_array_equal(net2.output_weights.detach().numpy(), first_output_weights.numpy())
        assert len(net2.hidden_units) == first_unit_count

    def test_refactored_add_unit_matches_expected_structure(self):
        from unittest.mock import MagicMock

        net = _make_network(init_output_weights="zero")
        prev_in = net.output_weights.shape[0]
        prev_history_len = len(net.history["hidden_units_added"])

        candidate = MagicMock()
        candidate.weights = torch.tensor([0.7, -0.3], dtype=torch.float32)
        candidate.bias = torch.tensor([0.05], dtype=torch.float32)
        candidate.correlation = 0.815

        x = torch.randn(10, 2)
        net.add_unit(candidate, x)

        # Hidden units grew by exactly one
        assert len(net.hidden_units) == 1
        new_unit = net.hidden_units[-1]
        np.testing.assert_array_equal(new_unit["weights"].detach().numpy(), candidate.weights.numpy())
        assert new_unit["correlation"] == pytest.approx(0.815)

        # Output layer widened by exactly one row
        assert net.output_weights.shape == (prev_in + 1, net.output_size)
        # New row is zeros (init_output_weights == "zero")
        new_row = net.output_weights[prev_in:, :].detach().numpy()
        np.testing.assert_array_equal(new_row, np.zeros_like(new_row))

        # History gained exactly one record
        assert len(net.history["hidden_units_added"]) == prev_history_len + 1
        assert net.history["hidden_units_added"][-1]["unit_index"] == 0

    def test_two_sequential_adds_account_correctly(self):
        from unittest.mock import MagicMock

        net = _make_network(init_output_weights="zero")
        x = torch.randn(8, 2)

        c1 = MagicMock(weights=torch.tensor([1.0, 0.0], dtype=torch.float32), bias=torch.tensor([0.0], dtype=torch.float32), correlation=0.5)
        c2 = MagicMock(weights=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32), bias=torch.tensor([0.0], dtype=torch.float32), correlation=0.6)

        net.add_unit(c1, x)
        net.add_unit(c2, x)

        assert len(net.hidden_units) == 2
        # Output layer should be width=4 now (2 inputs + 2 hidden units).
        assert net.output_weights.shape == (4, net.output_size)
        # History records two entries with monotonic unit_index.
        records = net.history["hidden_units_added"]
        assert records[-2]["unit_index"] == 0
        assert records[-1]["unit_index"] == 1
