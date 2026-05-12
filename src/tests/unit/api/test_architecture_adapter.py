"""Unit tests for ``api.lifecycle.architecture_adapter`` (Phase 2 P2-1c).

Covers the additive-only architecture adapter defined by
``ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md`` §3.6:

* Equal-dim → no-op (returns zero-delta ``ArchChanges``).
* Grow-only (input and/or output) → in-place weight-tensor expansion with
  zero-initialised new connections.
* Any shrink → ``ValueError`` starting with ``shrink_unsupported`` (route
  → HTTP 422). P2-1d will lift this.

The most subtle invariant is the **zero-init weight preservation**: after a
grow-input swap, padding the original input with zeros for the new slots
must reproduce the pre-swap forward pass exactly across the original output
dims. The integration regression test
``test_grow_input_preserves_forward_pass_on_padded_input`` pins this using
a real ``CascadeCorrelationNetwork`` with a few hidden units installed.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.lifecycle.architecture_adapter import ArchChanges, adapt_for_dataset_swap

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stub network helpers — most unit tests don't need a real
# CascadeCorrelationNetwork, only an object with the four attributes the
# adapter touches. The forward-pass regression test below builds a real one.
# ---------------------------------------------------------------------------


def _stub_network(input_size: int, output_size: int, hidden_widths=None):
    """Build a SimpleNamespace that quacks like a CascadeCorrelationNetwork
    well enough for the adapter to touch its weight tensors.

    ``hidden_widths`` is an iterable of integers naming the hidden-unit
    indices to install — each unit i gets a weight vector of size
    ``input_size + i`` and a scalar bias. This lets a test pin exactly
    how many cascaded units exist without spinning up the real class.
    """
    hidden_widths = list(hidden_widths or [])
    n_hidden = len(hidden_widths)
    return SimpleNamespace(
        input_size=input_size,
        output_size=output_size,
        # output_weights: rows = inputs + hidden, cols = outputs. Use a
        # known pattern so we can assert preservation: row i is filled
        # with the value ``i + 1`` (avoids confusion with zero-init padding).
        output_weights=torch.arange(1, input_size + n_hidden + 1, dtype=torch.float32).unsqueeze(1).repeat(1, output_size),
        output_bias=torch.arange(1, output_size + 1, dtype=torch.float32) * 0.1,
        hidden_units=[
            {
                # Each hidden unit's weights: vector of length input_size + i.
                # Pattern: row j is value ``(i+1) * 10 + j``.
                "weights": torch.tensor([(i + 1) * 10 + j for j in range(input_size + i)], dtype=torch.float32),
                "bias": torch.tensor((i + 1) * 100.0),
                "activation_fn": torch.tanh,
                "correlation": 0.0,
            }
            for i in hidden_widths
        ],
    )


# ---------------------------------------------------------------------------
# ArchChanges shape + ``to_response_dict``
# ---------------------------------------------------------------------------


class TestArchChanges:
    def test_default_shape(self):
        """Frozen dataclass with the §3.3 fields in the order callers expect."""
        a = ArchChanges(input_delta=2, output_delta=0, hidden_preserved=5)
        assert a.input_delta == 2
        assert a.output_delta == 0
        assert a.hidden_preserved == 5
        assert a.appended_nodes == {"input": 0, "output": 0}
        assert a.prepended_layers == []

    def test_to_response_dict_matches_3_3_response_keys(self):
        """``to_response_dict`` renders the §3.3 ``arch_changes`` shape exactly.
        Lifecycle code merges the swap-runtime ``abandoned_candidate_pool_size``
        field onto the result — the adapter itself does not own that field."""
        a = ArchChanges(
            input_delta=1,
            output_delta=2,
            hidden_preserved=3,
            appended_nodes={"input": 1, "output": 2},
            prepended_layers=[],
        )
        d = a.to_response_dict()
        assert d == {
            "input_delta": 1,
            "output_delta": 2,
            "hidden_preserved": 3,
            "appended_nodes": {"input": 1, "output": 2},
            "prepended_layers": [],
        }
        # Mutating the returned dict must not affect the dataclass — pins
        # the dict-copy guarantee. The frozen dataclass enforces this for
        # scalars but ``appended_nodes`` would alias without the copy.
        d["appended_nodes"]["input"] = 999
        assert a.appended_nodes == {"input": 1, "output": 2}


# ---------------------------------------------------------------------------
# Equal-dim no-op
# ---------------------------------------------------------------------------


class TestEqualDimNoOp:
    def test_equal_dim_returns_zero_delta(self):
        net = _stub_network(input_size=3, output_size=2, hidden_widths=[0, 1])
        before_w = net.output_weights.clone()
        before_b = net.output_bias.clone()

        result = adapt_for_dataset_swap(net, before=(3, 2), after=(3, 2))

        assert result.input_delta == 0
        assert result.output_delta == 0
        assert result.appended_nodes == {"input": 0, "output": 0}
        assert result.prepended_layers == []
        # No mutation on equal-dim — the adapter should not touch tensors
        # when there's no work to do (avoids unnecessary requires_grad
        # churn and optimizer-state invalidation downstream).
        assert torch.equal(net.output_weights, before_w)
        assert torch.equal(net.output_bias, before_b)


# ---------------------------------------------------------------------------
# Shrink rejection
# ---------------------------------------------------------------------------


class TestShrinkRejection:
    def test_input_shrink_raises_with_shrink_unsupported_prefix(self):
        """Route translation depends on the message starting with
        ``shrink_unsupported``; the route's bare ValueError → 422 path
        passes the message through unchanged. Pin both the prefix and
        the dim deltas being named (UX for the canopy error toast)."""
        net = _stub_network(input_size=5, output_size=2)
        with pytest.raises(ValueError, match=r"^shrink_unsupported \(P2-1c\): input 5→2"):
            adapt_for_dataset_swap(net, before=(5, 2), after=(2, 2))

    def test_output_shrink_raises(self):
        net = _stub_network(input_size=2, output_size=3)
        with pytest.raises(ValueError, match=r"shrink_unsupported.*output 3→2"):
            adapt_for_dataset_swap(net, before=(2, 3), after=(2, 2))

    def test_mixed_shrink_grow_rejected_as_shrink(self):
        """Mixed (input grows, output shrinks) is still a shrink overall — P2-1d
        territory. Network must NOT be mutated when the adapter rejects."""
        net = _stub_network(input_size=2, output_size=3)
        before_w = net.output_weights.clone()
        before_b = net.output_bias.clone()
        with pytest.raises(ValueError, match="shrink_unsupported"):
            adapt_for_dataset_swap(net, before=(2, 3), after=(5, 2))
        assert torch.equal(net.output_weights, before_w)
        assert torch.equal(net.output_bias, before_b)

    def test_non_positive_new_dim_rejected(self):
        net = _stub_network(input_size=2, output_size=2)
        with pytest.raises(ValueError, match="non-positive new dim"):
            adapt_for_dataset_swap(net, before=(2, 2), after=(0, 2))
        with pytest.raises(ValueError, match="non-positive new dim"):
            adapt_for_dataset_swap(net, before=(2, 2), after=(2, -1))


# ---------------------------------------------------------------------------
# Grow-input
# ---------------------------------------------------------------------------


class TestGrowInput:
    def test_grow_input_only_no_hidden_units(self):
        """Simplest case: no hidden units. ``output_weights`` is just
        ``[input_size, output_size]`` — grow appends zero-rows. Hidden
        units list stays empty."""
        net = _stub_network(input_size=2, output_size=2, hidden_widths=[])
        # Output rows pattern: row j = (j+1) on both cols
        # → row 0 = [1, 1], row 1 = [2, 2]
        result = adapt_for_dataset_swap(net, before=(2, 2), after=(4, 2))
        assert result.input_delta == 2
        assert result.appended_nodes == {"input": 2, "output": 0}
        assert net.input_size == 4
        assert net.output_weights.shape == (4, 2)
        # Preserved input rows
        assert torch.equal(net.output_weights[0], torch.tensor([1.0, 1.0]))
        assert torch.equal(net.output_weights[1], torch.tensor([2.0, 2.0]))
        # Zero-init new input rows
        assert torch.equal(net.output_weights[2], torch.zeros(2))
        assert torch.equal(net.output_weights[3], torch.zeros(2))
        # requires_grad restored so the next train_output_layer call can
        # build an optimizer on the expanded tensor.
        assert net.output_weights.requires_grad is True

    def test_grow_input_with_hidden_units_inserts_rows_in_middle(self):
        """Row-insertion (NOT append): hidden-unit rows must shift down so
        the new input rows occupy the middle slice ``[i_old:i_new, :]``."""
        # input_size=2, output_size=1, 3 hidden units → output_weights shape
        # [2 + 3, 1] = [5, 1] with row pattern [[1],[2],[3],[4],[5]].
        # After grow to input=5: shape [5 + 3, 1] = [8, 1]
        # Expected: rows 0-1 preserved = [1, 2]
        #           rows 2-4 zero-init  = [0, 0, 0]   (new input slots)
        #           rows 5-7 preserved  = [3, 4, 5]   (hidden rows shifted)
        net = _stub_network(input_size=2, output_size=1, hidden_widths=[0, 1, 2])

        adapt_for_dataset_swap(net, before=(2, 1), after=(5, 1))

        assert net.input_size == 5
        assert net.output_weights.shape == (8, 1)
        flat = net.output_weights[:, 0].tolist()
        assert flat == [1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0]

    def test_grow_input_expands_each_hidden_unit_weights(self):
        """Every hidden unit's weight vector must mirror the row-insertion:
        preserve input entries, zero-fill the gap, shift prior-hidden
        entries down. Without this, hidden activations would change
        post-swap and the zero-init invariant would not hold."""
        # 2 hidden units, input=2: unit 0 has weights shape [2]; unit 1 has
        # shape [3] (input + 1 prior hidden).
        # Unit 0 pattern: [10, 11]            (value (i+1)*10 + j with i=0)
        # Unit 1 pattern: [20, 21, 22]        (i=1)
        net = _stub_network(input_size=2, output_size=1, hidden_widths=[0, 1])

        adapt_for_dataset_swap(net, before=(2, 1), after=(4, 1))

        # Unit 0: new shape [4], pattern [10, 11, 0, 0] (no prior hidden,
        # so the post-input section is empty in both old + new — i==0).
        u0 = net.hidden_units[0]["weights"].tolist()
        assert u0 == [10.0, 11.0, 0.0, 0.0]
        # Unit 1: new shape [4 + 1] = [5]
        # Original [20, 21, 22] = [in0, in1, prior_h0]
        # New      [20, 21, 0, 0, 22]
        u1 = net.hidden_units[1]["weights"].tolist()
        assert u1 == [20.0, 21.0, 0.0, 0.0, 22.0]
        # Biases untouched — they're per-unit scalars.
        assert net.hidden_units[0]["bias"].item() == 100.0
        assert net.hidden_units[1]["bias"].item() == 200.0


# ---------------------------------------------------------------------------
# Grow-output
# ---------------------------------------------------------------------------


class TestGrowOutput:
    def test_grow_output_appends_columns(self):
        """Output dim grows → output_weights gains zero-init columns and
        output_bias gains zero-init elements. Hidden units are untouched."""
        net = _stub_network(input_size=2, output_size=2, hidden_widths=[0])
        # Pre: output_weights shape [3, 2]
        #   row 0: [1, 1]
        #   row 1: [2, 2]
        #   row 2: [3, 3]
        # output_bias: [0.1, 0.2]
        # hidden_units[0]: weights=[10, 11], bias=100
        before_u0_weights = net.hidden_units[0]["weights"].clone()
        before_u0_bias = net.hidden_units[0]["bias"].clone()

        result = adapt_for_dataset_swap(net, before=(2, 2), after=(2, 4))

        assert result.output_delta == 2
        assert result.appended_nodes == {"input": 0, "output": 2}
        # output_weights: shape [3, 4]; first 2 cols preserved; last 2 zero.
        assert net.output_weights.shape == (3, 4)
        assert torch.equal(net.output_weights[:, 0], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(net.output_weights[:, 1], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(net.output_weights[:, 2], torch.zeros(3))
        assert torch.equal(net.output_weights[:, 3], torch.zeros(3))
        assert net.output_weights.requires_grad is True
        # output_bias: shape [4]; first 2 preserved; last 2 zero. The
        # tensor's float32 representation of 0.1 is 0.10000000149..., not
        # the Python float64 ``0.1`` — compare via ``pytest.approx``.
        assert net.output_bias.tolist() == pytest.approx([0.1, 0.2, 0.0, 0.0], abs=1e-6)
        # Hidden units: untouched (no output-dim coupling).
        assert torch.equal(net.hidden_units[0]["weights"], before_u0_weights)
        assert torch.equal(net.hidden_units[0]["bias"], before_u0_bias)
        assert net.output_size == 4


# ---------------------------------------------------------------------------
# Mixed grow (input + output both grow)
# ---------------------------------------------------------------------------


class TestGrowMixed:
    def test_grow_both_input_and_output(self):
        """Both dims grow → input grow applied first, then output grow.
        End state: rows inserted in middle, columns appended on right."""
        # input=2, output=2, 1 hidden unit → output_weights shape [3, 2].
        # row 0=[1,1], row 1=[2,2], row 2=[3,3]
        net = _stub_network(input_size=2, output_size=2, hidden_widths=[0])

        result = adapt_for_dataset_swap(net, before=(2, 2), after=(4, 3))

        assert result.input_delta == 2
        assert result.output_delta == 1
        assert result.appended_nodes == {"input": 2, "output": 1}
        # After grow-input to 4: output_weights shape [4+1, 2] = [5, 2].
        #   rows 0-1 preserved [1,1] [2,2]
        #   rows 2-3 zero-init
        #   row 4 hidden-row shifted from old row 2: [3, 3]
        # Then grow-output to 3: shape [5, 3]. Last col zero-init.
        assert net.output_weights.shape == (5, 3)
        cols = net.output_weights.tolist()
        assert cols[0] == [1.0, 1.0, 0.0]
        assert cols[1] == [2.0, 2.0, 0.0]
        assert cols[2] == [0.0, 0.0, 0.0]
        assert cols[3] == [0.0, 0.0, 0.0]
        assert cols[4] == [3.0, 3.0, 0.0]
        # output_bias: [0.1, 0.2, 0.0] within float32 representation noise.
        assert net.output_bias.tolist() == pytest.approx([0.1, 0.2, 0.0], abs=1e-6)
        # Hidden unit 0: weights [10, 11] → [10, 11, 0, 0] (no prior hidden).
        assert net.hidden_units[0]["weights"].tolist() == [10.0, 11.0, 0.0, 0.0]
        assert net.input_size == 4
        assert net.output_size == 3


# ---------------------------------------------------------------------------
# Weight-preservation regression — the §3.6 zero-init invariant
# ---------------------------------------------------------------------------


class TestZeroInitForwardInvariant:
    """The most subtle invariant of P2-1c: after a grow-input swap, feeding
    the post-swap network ``[x_old, 0, 0, ...]`` (the original input padded
    with zeros for the new slots) reproduces the pre-swap forward pass
    exactly across the original output dims. New output dims after a
    grow-output swap are exactly zero on the first forward call.

    Uses a real ``CascadeCorrelationNetwork`` via the lifecycle manager so
    the forward pass really runs the cascade construction (not the stub's
    static tensors)."""

    def test_grow_input_preserves_forward_pass_on_padded_input(self):
        """Build a small network with cascaded hidden units, capture
        ``forward(x_old)``, swap to a larger input dim, verify that
        ``forward([x_old | 0]) == forward(x_old)`` (FP-tolerant)."""
        from api.lifecycle.manager import TrainingLifecycleManager

        torch.manual_seed(2026)
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=3, output_size=2)
        net = mgr.network

        # Install two hidden units with deterministic weights so the
        # forward pass produces a non-trivial signal we can check against.
        # Unit 0: input_size=3 incoming weights.
        # Unit 1: input_size + 1 = 4 incoming weights (input + prior hidden).
        net.hidden_units.append(
            {
                "weights": torch.tensor([0.5, -0.3, 0.2], dtype=torch.float32),
                "bias": torch.tensor(0.1, dtype=torch.float32),
                "activation_fn": torch.tanh,
                "correlation": 0.0,
            }
        )
        net.hidden_units.append(
            {
                "weights": torch.tensor([0.1, 0.4, -0.2, 0.6], dtype=torch.float32),
                "bias": torch.tensor(-0.05, dtype=torch.float32),
                "activation_fn": torch.tanh,
                "correlation": 0.0,
            }
        )
        # Grow output_weights to match: shape [3 + 2, 2] = [5, 2].
        net.output_weights = torch.randn(5, 2)
        net.output_bias = torch.randn(2)

        x_old = torch.randn(4, 3)  # batch of 4, 3 features
        pre_output = net.forward(x_old).detach().clone()

        # Grow input from 3 → 5.
        adapt_for_dataset_swap(net, before=(3, 2), after=(5, 2))

        # Pad x_old with zero columns for the new input slots.
        x_padded = torch.cat([x_old, torch.zeros(4, 2)], dim=1)
        post_output = net.forward(x_padded).detach()

        # Zero-init invariant: outputs should be bit-identical (we're
        # multiplying by zero on the new connections; no FP error
        # accumulates differently). Use a tight FP tolerance anyway
        # since matmul column ordering can introduce one-ULP noise.
        assert torch.allclose(pre_output, post_output, atol=1e-6, rtol=0), f"forward(padded) differs from pre-swap forward by max={torch.max(torch.abs(pre_output - post_output)).item():g}"
        mgr.shutdown()

    def test_grow_output_yields_zero_on_new_output_dims(self):
        """After a grow-output swap, the new output columns are pure zero
        on the first forward pass (zero weights + zero bias). The
        original output columns are unchanged."""
        from api.lifecycle.manager import TrainingLifecycleManager

        torch.manual_seed(2026)
        mgr = TrainingLifecycleManager()
        mgr.create_network(input_size=2, output_size=2)
        net = mgr.network
        net.output_weights = torch.randn(2, 2)
        net.output_bias = torch.randn(2)
        x = torch.randn(3, 2)
        pre_output = net.forward(x).detach().clone()

        adapt_for_dataset_swap(net, before=(2, 2), after=(2, 4))

        post_output = net.forward(x).detach()
        # Original output columns preserved bit-for-bit.
        assert torch.allclose(post_output[:, :2], pre_output, atol=1e-6, rtol=0)
        # New output columns are zero (matmul with zero weights + zero bias).
        assert torch.equal(post_output[:, 2:], torch.zeros(3, 2))
        mgr.shutdown()
