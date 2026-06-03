#!/usr/bin/env python
"""Unit tests for the grow_network completion-reason diagnostic (Issue #3 follow-up).

``grow_network`` records *which* exit fired on the most recent growth run in
``self._completion_reason`` (surfaced by ``manager.get_status()`` as
``completion_reason``), so canopy can distinguish a genuine convergence from a
0-unit stall instead of both showing a bare "Completed". These tests pin each
break path to its reason string. Producer side only — the canopy consumer is a
separate follow-up.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig


def _make_network():
    config = CascadeCorrelationConfig(input_size=2, output_size=2, candidate_pool_size=2, candidate_epochs=1)
    return CascadeCorrelationNetwork(config=config)


def _xy():
    torch.manual_seed(0)
    return torch.randn(8, 2), torch.randn(8, 2)


@pytest.mark.unit
class TestCompletionReason:
    """Each grow_network exit sets the expected completion_reason."""

    def test_default_is_none_before_training(self):
        """A freshly constructed network has no completion reason yet."""
        assert _make_network()._completion_reason is None

    def test_max_iterations(self):
        """max_iterations=0 → the for-loop body never runs → for/else fires.

        Exercises the cap path with zero training work (range(0) is empty, so
        the ``else`` clause runs immediately).
        """
        net = _make_network()
        x, y = _xy()
        net.grow_network(x_train=x, y_train=y, max_iterations=0, early_stopping=False)
        assert net._completion_reason == "max_iterations"

    def test_residual_collapsed(self):
        """Residual error None on the first iteration → residual_collapsed."""
        net = _make_network()
        x, y = _xy()
        with patch.object(net, "_calculate_residual_error_safe", return_value=None):
            net.grow_network(x_train=x, y_train=y, max_iterations=3, early_stopping=False)
        assert net._completion_reason == "residual_collapsed"

    def test_no_candidate(self):
        """No training results / no best candidate → no_candidate (the stall signature)."""
        net = _make_network()
        x, y = _xy()
        with patch.object(net, "_calculate_residual_error_safe", return_value=torch.ones(8, 2)), patch.object(net, "_get_training_results", return_value=None):
            net.grow_network(x_train=x, y_train=y, max_iterations=3, early_stopping=False)
        assert net._completion_reason == "no_candidate"

    def test_below_threshold(self):
        """Best candidate correlation below the adaptive threshold → below_threshold."""
        net = _make_network()
        x, y = _xy()
        results = MagicMock()
        results.best_candidate.get_correlation.return_value = 1e-9  # far below any adaptive threshold
        with patch.object(net, "_calculate_residual_error_safe", return_value=torch.ones(8, 2)), patch.object(net, "_get_training_results", return_value=results):
            net.grow_network(x_train=x, y_train=y, max_iterations=3, early_stopping=False)
        assert net._completion_reason == "below_threshold"

    def test_early_stopped(self):
        """Validation early-stop on an otherwise healthy iteration → early_stopped."""
        net = _make_network()
        x, y = _xy()
        results = MagicMock()
        results.best_candidate.get_correlation.return_value = 10.0  # well above threshold
        validation = MagicMock(early_stop=True, patience_counter=0, best_value_loss=0.0)
        with patch.object(net, "_calculate_residual_error_safe", return_value=torch.ones(8, 2)), patch.object(net, "_get_training_results", return_value=results), patch.object(net, "_effective_candidate_count", return_value=1), patch.object(net, "_add_best_candidate", return_value=(0.1, 0.9)), patch.object(net, "validate_training", return_value=validation):
            net.grow_network(x_train=x, y_train=y, max_iterations=3, early_stopping=True)
        assert net._completion_reason == "early_stopped"
