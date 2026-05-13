"""P2-1d (Issue #3) — ``CascadeCorrelationNetwork._resize_network_for_dataset``.

Covers the contract documented in
``juniper-cascor/notes/PHASE_2_P2_1D_DESIGN_2026-05-13.md``:

  * The network is **monotonically non-decreasing** on both ``input_size`` and
    ``output_size``. Shrink raises ``ValueError`` (handled at the dataset
    layer via ``_pad_dataset_for_network``).
  * Grow uses **random-init × ``self.random_value_scale``** for new entries
    (matching the construction-time pattern at
    ``CascadeCorrelationNetwork.__init__``).
  * Grow-input on ``output_weights`` is **row-insertion at index
    ``self.input_size``**, not append — preserves the
    ``[raw_inputs | hidden_outputs]`` layout the forward pass expects.
  * Each hidden unit's weight vector mirrors the same row-insertion.
  * Hidden-unit biases (per-unit scalars) are never touched.
  * Bookkeeping (``input_size``, ``output_size``, ``active_output_dim``) is
    updated AFTER all tensor mutations succeed.
"""

from __future__ import annotations

import pytest
import torch

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures — small networks with known weight patterns
# ---------------------------------------------------------------------------


def _make_network(input_size: int = 2, output_size: int = 2, hidden_widths: list | None = None) -> CascadeCorrelationNetwork:
    """Build a CascadeCorrelationNetwork with deterministic weight patterns.

    ``hidden_widths`` names which hidden-unit indices to install. Each unit
    ``i`` gets weights shape ``[input_size + i]`` with values ``(i+1)*10 + j``
    so the test can verify row-insertion preserves the right entries.
    """
    cfg = CascadeCorrelationConfig(input_size=input_size, output_size=output_size, random_value_scale=1.0, random_seed=42)
    net = CascadeCorrelationNetwork(cfg)

    # Overwrite construction-time random init with a known pattern.
    # output_weights row j = [j+1, j+1, ...] (constant across columns).
    net.output_weights = torch.arange(1, input_size + 1, dtype=torch.float32).unsqueeze(1).repeat(1, output_size).requires_grad_(True)
    # output_bias element k = 0.1 * (k+1).
    net.output_bias = (torch.arange(1, output_size + 1, dtype=torch.float32) * 0.1).requires_grad_(True)

    # Install hidden units with deterministic weight patterns.
    hidden_widths = hidden_widths or []
    for i in hidden_widths:
        weight_size = input_size + i
        weights = torch.tensor([(i + 1) * 10 + j for j in range(weight_size)], dtype=torch.float32)
        bias = torch.tensor((i + 1) * 100.0, dtype=torch.float32)
        net.hidden_units.append(
            {
                "weights": weights,
                "bias": bias,
                "activation_fn": torch.tanh,
                "correlation": 0.0,
            }
        )
        # After installing a unit, output_weights needs an extra row for it.
        # Match the production "added unit" pattern: pad output_weights with
        # a deterministic row that the test can detect later.
        old_w = net.output_weights.detach()
        new_row = torch.full((1, output_size), float(input_size + i + 1))
        net.output_weights = torch.cat([old_w, new_row], dim=0).requires_grad_(True)

    return net


# ---------------------------------------------------------------------------
# Equal-dim no-op
# ---------------------------------------------------------------------------


class TestEqualDimNoOp:
    def test_equal_dim_returns_zero_deltas(self):
        net = _make_network(input_size=3, output_size=2)
        before_w = net.output_weights.clone()
        before_b = net.output_bias.clone()
        result = net._resize_network_for_dataset(input_size_new=3, output_size_new=2)
        assert result == {"input_delta": 0, "output_delta": 0, "hidden_preserved": 0}
        # No tensor mutation (avoids unnecessary requires_grad churn).
        assert torch.equal(net.output_weights, before_w)
        assert torch.equal(net.output_bias, before_b)


# ---------------------------------------------------------------------------
# Shrink rejection
# ---------------------------------------------------------------------------


class TestShrinkRejection:
    def test_input_shrink_raises(self):
        net = _make_network(input_size=5, output_size=2)
        with pytest.raises(ValueError, match=r"input_size cannot shrink \(5 → 2\)"):
            net._resize_network_for_dataset(input_size_new=2, output_size_new=2)

    def test_output_shrink_raises(self):
        net = _make_network(input_size=2, output_size=3)
        with pytest.raises(ValueError, match=r"output_size cannot shrink \(3 → 2\)"):
            net._resize_network_for_dataset(input_size_new=2, output_size_new=2)

    def test_partial_shrink_rejects_without_mutating(self):
        """A mixed-direction request (input grows, output shrinks) is still a
        shrink overall — raises, network unchanged."""
        net = _make_network(input_size=2, output_size=3)
        before_w = net.output_weights.clone()
        before_b = net.output_bias.clone()
        with pytest.raises(ValueError, match="output_size cannot shrink"):
            net._resize_network_for_dataset(input_size_new=5, output_size_new=2)
        # Output grow happens before hidden-unit grow in the implementation;
        # since the size check fires first, output_weights MUST be untouched.
        assert torch.equal(net.output_weights, before_w)
        assert torch.equal(net.output_bias, before_b)
        assert net.input_size == 2  # bookkeeping unchanged on rejection


# ---------------------------------------------------------------------------
# Grow-input
# ---------------------------------------------------------------------------


class TestGrowInput:
    def test_grow_input_no_hidden_units_appends_rows(self):
        """No hidden units → output_weights is [input_size, output_size];
        grow-input row-insertion at index ``self.input_size`` reduces to a
        simple append of random rows at the end."""
        net = _make_network(input_size=2, output_size=2, hidden_widths=[])
        # Pre: output_weights = [[1,1],[2,2]], output_bias = [0.1, 0.2]
        result = net._resize_network_for_dataset(input_size_new=4, output_size_new=2)
        assert result["input_delta"] == 2
        assert result["output_delta"] == 0
        assert result["hidden_preserved"] == 0
        assert net.input_size == 4
        assert net.output_weights.shape == (4, 2)
        # Preserved rows.
        assert torch.equal(net.output_weights[0], torch.tensor([1.0, 1.0]))
        assert torch.equal(net.output_weights[1], torch.tensor([2.0, 2.0]))
        # New rows are random, not zero — distinguishes P2-1d from P2-1c.
        assert not torch.equal(net.output_weights[2:], torch.zeros(2, 2)), "new rows must be random-init, not zero"
        # Grad re-enabled on the new tensor.
        assert net.output_weights.requires_grad is True

    def test_grow_input_with_hidden_units_inserts_rows_at_input_size(self):
        """Critical row-insertion test: hidden-unit rows must shift down so
        the new input rows occupy the middle slice. If the implementation
        appends instead of inserting, the network's forward pass at
        cascade_correlation.py line 1577 will produce wrong output.
        """
        # input_size=2, output_size=1, 3 hidden units.
        # output_weights pre-grow: [[1],[2],[3],[4],[5]]
        #   - rows 0-1: raw-input rows (pattern j+1)
        #   - rows 2-4: hidden-unit rows (pattern input_size+i+1 = 3, 4, 5)
        net = _make_network(input_size=2, output_size=1, hidden_widths=[0, 1, 2])

        result = net._resize_network_for_dataset(input_size_new=5, output_size_new=1)
        assert result["input_delta"] == 3
        assert result["hidden_preserved"] == 3
        assert net.input_size == 5
        # New shape: [5 + 3, 1] = [8, 1]
        assert net.output_weights.shape == (8, 1)
        flat = net.output_weights[:, 0].tolist()
        # Rows 0-1 preserved (raw inputs); rows 2-4 random new inputs;
        # rows 5-7 preserved from old rows 2-4 (hidden, shifted down by 3).
        assert flat[0] == pytest.approx(1.0)
        assert flat[1] == pytest.approx(2.0)
        # New input rows are random (probabilistic check: not equal to any
        # of the preserved values, and unlikely to all be zero).
        new_rows = flat[2:5]
        assert all(v != 0.0 for v in new_rows), "row-insertion must use random init, not zero"
        # Hidden-unit rows shifted to the bottom in their original order.
        assert flat[5] == pytest.approx(3.0)
        assert flat[6] == pytest.approx(4.0)
        assert flat[7] == pytest.approx(5.0)

    def test_grow_input_expands_each_hidden_unit_weight_vector(self):
        """Each hidden unit's weight vector mirrors the row-insertion: preserve
        raw-input entries, insert random in the middle, shift prior-hidden
        entries down. Without this, forward()'s hidden-unit computation at
        line 1544 (``unit_input * unit["weights"]``) would align wrong
        positionally."""
        # Unit 0: weights [10, 11]            (i=0, no prior hidden)
        # Unit 1: weights [20, 21, 22]        (i=1, one prior hidden)
        net = _make_network(input_size=2, output_size=1, hidden_widths=[0, 1])

        net._resize_network_for_dataset(input_size_new=4, output_size_new=1)

        # Unit 0: new shape [4]. Pattern: [10, 11, RAND, RAND].
        u0 = net.hidden_units[0]["weights"].tolist()
        assert u0[0] == pytest.approx(10.0)
        assert u0[1] == pytest.approx(11.0)
        assert len(u0) == 4
        # Unit 1: new shape [5]. Pattern: [20, 21, RAND, RAND, 22] —
        # raw-input prefix preserved, two random inserted at index input_size=2,
        # prior-hidden entry shifted from index 2 to index 4.
        u1 = net.hidden_units[1]["weights"].tolist()
        assert u1[0] == pytest.approx(20.0)
        assert u1[1] == pytest.approx(21.0)
        assert u1[4] == pytest.approx(22.0), "prior-hidden entry must shift to new tail position"
        assert len(u1) == 5

    def test_grow_input_does_not_touch_hidden_unit_bias(self):
        """Hidden-unit biases (per-unit scalars) carry no input-dim coupling.
        The §3.6 zero-init contract from the old design didn't cover this
        either, but the prototype tried to pass them through the resize
        helpers. P2-1d's contract: bias is untouched."""
        net = _make_network(input_size=2, output_size=1, hidden_widths=[0, 1])
        u0_bias_before = net.hidden_units[0]["bias"].clone()
        u1_bias_before = net.hidden_units[1]["bias"].clone()

        net._resize_network_for_dataset(input_size_new=4, output_size_new=1)

        assert torch.equal(net.hidden_units[0]["bias"], u0_bias_before)
        assert torch.equal(net.hidden_units[1]["bias"], u1_bias_before)


# ---------------------------------------------------------------------------
# Grow-output
# ---------------------------------------------------------------------------


class TestGrowOutput:
    def test_grow_output_appends_columns_and_bias(self):
        """Output grow = column-append on output_weights + element-append on
        output_bias. Hidden units untouched (no output-dim coupling)."""
        net = _make_network(input_size=2, output_size=2, hidden_widths=[0])
        u0_weights_before = net.hidden_units[0]["weights"].clone()
        # output_bias = [0.1, 0.2]
        result = net._resize_network_for_dataset(input_size_new=2, output_size_new=4)
        assert result["output_delta"] == 2
        assert net.output_size == 4
        # output_weights shape: [3, 4]; first 2 cols preserved.
        assert net.output_weights.shape == (3, 4)
        # Pre cols: col 0 = col 1 = [1, 2, 3]
        assert net.output_weights[:, 0].tolist() == pytest.approx([1.0, 2.0, 3.0])
        assert net.output_weights[:, 1].tolist() == pytest.approx([1.0, 2.0, 3.0])
        # New cols are random (not zero).
        new_cols = net.output_weights[:, 2:].abs().sum().item()
        assert new_cols > 1e-6, "new output_weights cols must be random-init, not zero"
        # output_bias preserves [0.1, 0.2]; new elements random.
        assert net.output_bias[:2].tolist() == pytest.approx([0.1, 0.2], abs=1e-6)
        new_bias_mag = net.output_bias[2:].abs().sum().item()
        assert new_bias_mag > 1e-6, "new output_bias elements must be random-init, not zero"
        # Hidden unit weights untouched on grow-output.
        assert torch.equal(net.hidden_units[0]["weights"], u0_weights_before)


# ---------------------------------------------------------------------------
# Mixed grow
# ---------------------------------------------------------------------------


class TestGrowMixed:
    def test_grow_both_input_and_output(self):
        net = _make_network(input_size=2, output_size=2, hidden_widths=[0])
        result = net._resize_network_for_dataset(input_size_new=4, output_size_new=3)
        assert result["input_delta"] == 2
        assert result["output_delta"] == 1
        assert net.input_size == 4
        assert net.output_size == 3
        # output_weights: input grow [3,2]→[5,2], then output grow [5,2]→[5,3].
        assert net.output_weights.shape == (5, 3)
        # Hidden unit 0 weights expanded for input grow.
        assert net.hidden_units[0]["weights"].shape == (4,)
        # output_bias: [2] → [3]
        assert net.output_bias.shape == (3,)


# ---------------------------------------------------------------------------
# Active-output-dim bookkeeping
# ---------------------------------------------------------------------------


class TestActiveOutputDimReset:
    def test_resize_resets_active_output_dim_to_new_output_size(self):
        """A grow always pairs with a dataset whose output dim equals the new
        ``output_size``; the resize method resets ``active_output_dim`` so a
        prior shrink-via-padding doesn't leave stale loss masking in place
        after a grow."""
        net = _make_network(input_size=2, output_size=2)
        # Simulate a prior shrink: active dim < output_size.
        net.active_output_dim = 1
        net._resize_network_for_dataset(input_size_new=2, output_size_new=4)
        assert net.active_output_dim == 4, "resize should reset active_output_dim to the new output_size"

    def test_equal_dim_resize_does_not_touch_active_output_dim(self):
        """Equal-dim is a no-op; ``active_output_dim`` stays as-is (the
        lifecycle's pad helper owns it in the no-grow case)."""
        net = _make_network(input_size=2, output_size=3)
        net.active_output_dim = 2  # simulate prior shrink
        net._resize_network_for_dataset(input_size_new=2, output_size_new=3)
        assert net.active_output_dim == 2, "equal-dim resize must not clobber active_output_dim"


# ---------------------------------------------------------------------------
# Live size attributes (B1 regression)
# ---------------------------------------------------------------------------


class TestLiveSizeAttributes:
    def test_two_consecutive_grows_use_live_input_size(self):
        """Prototype bug B1: reading ``self.config.input_size`` (the original
        construction value) would compute wrong deltas after the first grow.
        Production code MUST read ``self.input_size`` (live)."""
        net = _make_network(input_size=2, output_size=2)
        # First grow: 2 → 4
        net._resize_network_for_dataset(input_size_new=4, output_size_new=2)
        assert net.input_size == 4
        # Second grow: 4 → 6 (delta should be 2, not 4 which would happen
        # if the implementation read config.input_size=2 instead).
        first_shape = net.output_weights.shape
        result = net._resize_network_for_dataset(input_size_new=6, output_size_new=2)
        assert result["input_delta"] == 2
        # Output weights gained 2 rows since the first grow.
        assert net.output_weights.shape[0] == first_shape[0] + 2
        assert net.input_size == 6
