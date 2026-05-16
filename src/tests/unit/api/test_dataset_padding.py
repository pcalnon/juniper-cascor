"""P2-1d (Issue #3) — ``TrainingLifecycleManager._pad_dataset_for_network``.

Covers the contract documented in
``juniper-cascor/notes/PHASE_2_P2_1D_DESIGN_2026-05-13.md``:

  * The helper zero-pads dataset tensors up to ``network.input_size`` /
    ``network.output_size`` and returns the pre-pad active dims so the
    caller can update ``network.active_output_dim`` for loss masking.
  * ``None`` validation tensors stay ``None``.
  * Tensors that already match network dims pass through unchanged.
  * Datasets that exceed network capacity raise ``ValueError`` (the caller
    must resize the network first; only ``swap_dataset_live`` owns grow).
  * The helper does NOT mutate ``self.network.active_output_dim`` directly —
    that's the caller's responsibility, keeping the helper safe to use in
    snapshot-load-validation paths that don't yet own the live training
    network.
"""

from __future__ import annotations

import pytest
import torch

from api.lifecycle.manager import TrainingLifecycleManager

pytestmark = pytest.mark.unit


def _make_mgr(input_size: int = 5, output_size: int = 3) -> TrainingLifecycleManager:
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=input_size, output_size=output_size)
    return mgr


class TestPadDatasetForNetwork:
    def test_no_pad_when_dataset_matches_network(self):
        mgr = _make_mgr(input_size=3, output_size=2)
        x = torch.randn(8, 3)
        y = torch.zeros(8, 2)
        val_x = torch.randn(4, 3)
        val_y = torch.zeros(4, 2)
        x_p, y_p, vx_p, vy_p, active_in, active_out = mgr._pad_dataset_for_network(x, y, val_x, val_y)
        # Shape unchanged (passthrough), dims reflect dataset's pre-pad dim.
        assert x_p.shape == (8, 3)
        assert y_p.shape == (8, 2)
        assert vx_p.shape == (4, 3)
        assert vy_p.shape == (4, 2)
        assert active_in == 3
        assert active_out == 2
        mgr.shutdown()

    def test_input_only_pad_appends_zero_cols(self):
        mgr = _make_mgr(input_size=5, output_size=2)
        x = torch.randn(8, 3)
        y = torch.zeros(8, 2)
        x_p, y_p, vx_p, vy_p, active_in, active_out = mgr._pad_dataset_for_network(x, y, None, None)
        assert x_p.shape == (8, 5)
        # First 3 cols preserved (passthrough).
        assert torch.equal(x_p[:, :3], x)
        # Last 2 cols zero.
        assert torch.equal(x_p[:, 3:], torch.zeros(8, 2))
        # y unchanged (network output matches).
        assert torch.equal(y_p, y)
        assert active_in == 3
        assert active_out == 2
        mgr.shutdown()

    def test_output_only_pad_appends_zero_target_cols(self):
        mgr = _make_mgr(input_size=2, output_size=4)
        x = torch.randn(8, 2)
        y = torch.ones(8, 2)  # use ones so we can detect the pad zeros
        x_p, y_p, vx_p, vy_p, active_in, active_out = mgr._pad_dataset_for_network(x, y, None, None)
        assert y_p.shape == (8, 4)
        assert torch.equal(y_p[:, :2], y)
        assert torch.equal(y_p[:, 2:], torch.zeros(8, 2))
        assert active_out == 2
        mgr.shutdown()

    def test_both_input_and_output_pad(self):
        mgr = _make_mgr(input_size=5, output_size=4)
        x = torch.randn(8, 2)
        y = torch.ones(8, 2)
        x_p, y_p, _, _, active_in, active_out = mgr._pad_dataset_for_network(x, y, None, None)
        assert x_p.shape == (8, 5)
        assert y_p.shape == (8, 4)
        assert active_in == 2
        assert active_out == 2
        mgr.shutdown()

    def test_none_val_tensors_stay_none(self):
        """Don't fabricate a zero validation set when the caller passes None
        — the absence of validation data is a meaningful signal that
        downstream training paths consume."""
        mgr = _make_mgr(input_size=5, output_size=2)
        x = torch.randn(8, 3)
        y = torch.zeros(8, 2)
        x_p, y_p, vx_p, vy_p, _, _ = mgr._pad_dataset_for_network(x, y, None, None)
        assert vx_p is None
        assert vy_p is None
        mgr.shutdown()

    def test_val_tensors_padded_alongside_train(self):
        mgr = _make_mgr(input_size=5, output_size=3)
        x = torch.randn(8, 3)
        y = torch.ones(8, 2)
        val_x = torch.randn(4, 3)
        val_y = torch.ones(4, 2)
        _, _, vx_p, vy_p, _, _ = mgr._pad_dataset_for_network(x, y, val_x, val_y)
        assert vx_p.shape == (4, 5)
        assert vy_p.shape == (4, 3)
        # First 3 input cols preserved.
        assert torch.equal(vx_p[:, :3], val_x)
        # First 2 output cols preserved.
        assert torch.equal(vy_p[:, :2], val_y)
        mgr.shutdown()

    def test_dataset_larger_than_network_raises(self):
        """Caller (swap_dataset_live) must resize the network FIRST. The
        helper does not auto-grow — it's a one-way operation (only pads up,
        never down)."""
        mgr = _make_mgr(input_size=2, output_size=2)
        x = torch.randn(8, 5)  # larger than network
        y = torch.zeros(8, 2)
        with pytest.raises(ValueError, match="exceeds network capacity"):
            mgr._pad_dataset_for_network(x, y, None, None)
        mgr.shutdown()

    def test_pad_preserves_tensor_dtype_and_device(self):
        """Pad columns must inherit dtype and device from the source tensor
        so a non-default-dtype training run doesn't get its dtype changed
        mid-stream."""
        mgr = _make_mgr(input_size=5, output_size=2)
        x = torch.randn(8, 3, dtype=torch.float64)
        y = torch.zeros(8, 2, dtype=torch.float64)
        x_p, y_p, _, _, _, _ = mgr._pad_dataset_for_network(x, y, None, None)
        assert x_p.dtype == torch.float64
        assert y_p.dtype == torch.float64
        mgr.shutdown()

    def test_helper_does_not_mutate_network_active_output_dim(self):
        """The helper returns the pre-pad active output dim; the caller is
        responsible for assigning it to ``network.active_output_dim``. This
        keeps the helper safe to use in snapshot-load-validation contexts
        that don't yet own the live training network."""
        mgr = _make_mgr(input_size=2, output_size=4)
        before = mgr.network.active_output_dim
        x = torch.randn(8, 2)
        y = torch.zeros(8, 2)
        _, _, _, _, _, active_out = mgr._pad_dataset_for_network(x, y, None, None)
        assert mgr.network.active_output_dim == before, "helper must not mutate network state"
        assert active_out == 2  # caller can act on the returned value
        mgr.shutdown()
