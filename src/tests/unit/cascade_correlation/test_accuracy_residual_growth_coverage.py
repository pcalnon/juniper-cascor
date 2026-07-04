#!/usr/bin/env python
"""Coverage for accuracy, residual-error, metric-emission, and network-growth
edges (per-file coverage lift 5, C-5).

Drives the previously-uncovered validation / defensive / debug-gated arms of
``CascadeCorrelationNetwork`` on small deterministic networks:

* ``calculate_residual_error`` — the non-tensor-input reset and the P2-1d
  active-output-dim residual masking.
* ``_emit_candidate_correlation`` — the swallowed metric-emission failure.
* ``calculate_accuracy`` — the output-not-a-tensor and output-batch-mismatch
  rejections (via a patched ``forward``).
* ``_accuracy`` — the sample-count mismatch rejection.
* ``add_unit`` — the stale-candidate weight-dimension mismatch + the
  debug-gated logging branches.
* ``add_units_as_layer`` — the per-candidate weight-mismatch skip.
* ``_add_best_candidate`` — the ``INFO``-gated entry log with a ``None``
  candidate.

The debug/info-gated branches are reached by swapping in a ``MagicMock``
logger (whose ``isEnabledFor`` is truthy) — the suite's default no-op logger
filters below WARNING.
"""

import logging
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork  # noqa: F401  (import parity / readability)
from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError

pytestmark = pytest.mark.unit


class TestCalculateResidualError:
    def test_non_tensor_inputs_reset_to_empty(self, simple_network):
        # Non-None, non-tensor inputs survive the None-defaulting and hit the
        # type-reset branch.
        residual = simple_network.calculate_residual_error("not-a-tensor", "also-not")
        assert isinstance(residual, torch.Tensor)
        assert residual.shape[0] == 0

    def test_active_output_dim_masks_residual_tail(self, simple_network):
        net = simple_network
        net.active_output_dim = 1  # output_size is 2 -> mask column 1
        x = torch.zeros(4, 2)
        y = torch.ones(4, 2)
        residual = net.calculate_residual_error(x, y)
        assert residual.shape == (4, 2)
        # The inactive tail column was zeroed.
        assert torch.count_nonzero(residual[:, 1:]) == 0


class TestEmitCandidateCorrelation:
    def test_emission_failure_swallowed(self, simple_network):
        with patch("api.observability.set_candidate_correlation", side_effect=RuntimeError("gauge down")):
            # Must not raise — metric emission is best-effort.
            simple_network._emit_candidate_correlation(0.42)


class TestCalculateAccuracyValidation:
    def test_non_tensor_output_rejected(self, simple_network):
        net = simple_network
        # A numpy array has ``.shape`` (so the pre-check debug log survives) but
        # is not a ``torch.Tensor`` -> the type-rejection arm.
        with patch.object(net, "forward", return_value=np.zeros((4, 2))):
            with pytest.raises(ValueError, match="Output tensor must be of type torch.Tensor"):
                net.calculate_accuracy(torch.zeros(4, 2), torch.zeros(4, 2))

    def test_output_batch_mismatch_rejected(self, simple_network):
        net = simple_network
        # Same feature dim, different sample count -> the shape[0] mismatch arm.
        with patch.object(net, "forward", return_value=torch.zeros(3, 2)):
            with pytest.raises(ValueError, match="compatible shapes"):
                net.calculate_accuracy(torch.zeros(4, 2), torch.zeros(4, 2))


class TestAccuracyValidation:
    def test_sample_count_mismatch_rejected(self, simple_network):
        with pytest.raises(ValueError, match="same number of samples"):
            simple_network._accuracy(y=torch.zeros(4, 2), output=torch.zeros(3, 2))


class TestAddUnit:
    def test_stale_candidate_weight_mismatch_raises(self, simple_network):
        net = simple_network
        x = torch.zeros(4, 2)
        # Fresh network: candidate_input width == input_size == 2; give the
        # candidate a mismatched (size-3) weight vector.
        candidate = types.SimpleNamespace(weights=torch.zeros(3), bias=torch.zeros(1), correlation=0.5)
        with pytest.raises(ValidationError, match="weight dimension mismatch"):
            net.add_unit(candidate, x)

    def test_debug_gated_logging_branches(self, simple_network):
        net = simple_network
        net.logger = MagicMock()
        net.logger.isEnabledFor.return_value = True  # force the debug branches
        x = torch.zeros(4, 2)
        candidate = types.SimpleNamespace(weights=torch.zeros(2), bias=torch.zeros(1), correlation=0.5)
        before = len(net.hidden_units)
        net.add_unit(candidate, x)
        assert len(net.hidden_units) == before + 1
        # The debug-gated new-unit output computation ran.
        assert net.logger.isEnabledFor.call_args_list  # isEnabledFor was consulted
        assert net.logger.isEnabledFor.call_args_list[0].args == (logging.DEBUG,)


class TestAddUnitsAsLayer:
    def test_mismatched_candidate_skipped(self, simple_network):
        net = simple_network
        x = torch.zeros(4, 2)
        before = len(net.hidden_units)
        mismatched = types.SimpleNamespace(candidate=types.SimpleNamespace(weights=torch.zeros(3)))
        net.add_units_as_layer([mismatched], x)
        # The single mismatched candidate was skipped -> no unit added.
        assert len(net.hidden_units) == before


class TestAddBestCandidate:
    def test_info_gated_log_with_none_candidate(self, simple_network):
        net = simple_network
        net.logger = MagicMock()
        net.logger.isEnabledFor.return_value = True  # force the INFO entry log
        train_loss, train_accuracy = net._add_best_candidate(best_candidate=None)
        assert train_loss is None and train_accuracy is None
        net.logger.isEnabledFor.assert_any_call(logging.INFO)
