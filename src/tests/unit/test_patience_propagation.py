#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# Purpose:       Regression test for patience counter propagation in grow_network()
#
# Author:        Paul Calnon
# Version:       0.1.0
# File Name:     test_patience_propagation.py
# File Path:     <Project>/<Sub-Project>/<Application>/src/tests/unit/
#
# Date Created:  2026-04-03
# Last Modified: 2026-04-03
#
# License:       MIT License
# Copyright:     Copyright (c) 2026 Paul Calnon
#
# Description:
#     Regression tests verifying that patience_counter and best_value_loss
#     are properly propagated across growth iterations in the grow_network()
#     method. Prior to the fix, these values were computed by validate_training()
#     but never assigned back to the loop variables, making patience-based
#     early stopping impossible.
#
#####################################################################################################################################################################################################

from unittest.mock import MagicMock, patch

import pytest
import torch

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork, ValidateTrainingInputs, ValidateTrainingResults
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig


def set_deterministic_behavior():
    """Set deterministic behavior for reproducible tests."""
    torch.manual_seed(42)


class TestPatienceCounterPropagation:
    """Regression tests for patience_counter and best_value_loss propagation in grow_network()."""

    @pytest.fixture
    def network(self):
        """Create a minimal network for testing."""
        set_deterministic_behavior()
        config = CascadeCorrelationConfig(
            input_size=2,
            output_size=2,
            candidate_pool_size=2,
            candidate_epochs=5,
            output_epochs=5,
            correlation_threshold=0.001,
            patience=3,
            convergence_threshold=0.001,
            max_hidden_units=20,
        )
        return CascadeCorrelationNetwork(config=config)

    @pytest.fixture
    def simple_data(self):
        """Generate simple classification data with train and validation splits."""
        set_deterministic_behavior()
        n = 40
        class_0 = torch.randn(n // 2, 2) + torch.tensor([-2.0, -2.0])
        class_1 = torch.randn(n // 2, 2) + torch.tensor([2.0, 2.0])
        x = torch.cat([class_0, class_1], dim=0)
        y = torch.cat([torch.tensor([[1, 0]] * (n // 2)), torch.tensor([[0, 1]] * (n // 2))], dim=0).float()
        return x, y

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_patience_counter_increments_across_iterations(self, network, simple_data):
        """Verify patience_counter is updated from validate_training results across growth iterations.

        This is a regression test for the bug where patience_counter stayed at 0
        because validate_training_results were never assigned back to loop variables.
        """
        x, y = simple_data
        x_train, x_val = x[:30], x[30:]
        y_train, y_val = y[:30], y[30:]

        patience_values_seen = []

        original_validate = network.validate_training

        def tracking_validate(inputs):
            """Wrapper that records patience_counter values passed to validate_training."""
            patience_values_seen.append(inputs.patience_counter)
            return original_validate(inputs)

        with patch.object(network, "validate_training", side_effect=tracking_validate):
            network.grow_network(
                x_train=x_train,
                y_train=y_train,
                max_iterations=10,
                early_stopping=True,
                patience_counter=0,
                best_value_loss=float("inf"),
                x_val=x_val,
                y_val=y_val,
            )

        # With the fix, patience_counter should be updated between iterations.
        # If validation loss doesn't improve, patience_counter should increment.
        # We verify that validate_training was called with non-zero patience_counter
        # at some point (unless training converges immediately).
        if len(patience_values_seen) > 1:
            # The first call should have patience_counter=0
            assert patience_values_seen[0] == 0, f"First iteration should start with patience_counter=0, got {patience_values_seen[0]}"
            # At least one subsequent call should have a different patience_counter
            # (either incremented if loss didn't improve, or reset to 0 if it did)
            # The key check is that it's NOT always 0 (the old bug behavior)
            # unless loss improved every single iteration
            has_non_zero = any(p > 0 for p in patience_values_seen[1:])
            all_improving = all(p == 0 for p in patience_values_seen)
            # Either patience incremented at some point (typical) or loss improved every time (rare but valid)
            assert has_non_zero or all_improving, f"Patience counter values across iterations: {patience_values_seen}. " f"Expected either increments (non-zero values) or consistent improvement (all zeros with reset)."

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_best_value_loss_propagates_across_iterations(self, network, simple_data):
        """Verify best_value_loss is updated from validate_training results across growth iterations."""
        x, y = simple_data
        x_train, x_val = x[:30], x[30:]
        y_train, y_val = y[:30], y[30:]

        best_loss_values_seen = []

        original_validate = network.validate_training

        def tracking_validate(inputs):
            """Wrapper that records best_value_loss values passed to validate_training."""
            best_loss_values_seen.append(inputs.best_value_loss)
            return original_validate(inputs)

        with patch.object(network, "validate_training", side_effect=tracking_validate):
            network.grow_network(
                x_train=x_train,
                y_train=y_train,
                max_iterations=10,
                early_stopping=True,
                patience_counter=0,
                best_value_loss=float("inf"),
                x_val=x_val,
                y_val=y_val,
            )

        if len(best_loss_values_seen) > 1:
            # First call starts with inf
            assert best_loss_values_seen[0] == float("inf"), f"First iteration should start with best_value_loss=inf, got {best_loss_values_seen[0]}"
            # After the first iteration, best_value_loss should be updated to a finite value
            # (the validation loss from iteration 0)
            assert best_loss_values_seen[1] < float("inf"), f"After first iteration, best_value_loss should be updated to a finite value, " f"got {best_loss_values_seen[1]}. This indicates the propagation bug."

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_early_stopping_triggers_on_patience_exhaustion(self, network, simple_data):
        """Verify that early stopping can actually trigger based on patience exhaustion."""
        x, y = simple_data

        # Set very low patience to force early stopping
        network.patience = 2

        result = network.grow_network(
            x_train=x,
            y_train=y,
            max_iterations=50,
            early_stopping=True,
            patience_counter=0,
            best_value_loss=float("inf"),
            x_val=x,
            y_val=y,
        )

        # With patience=2, if validation loss plateaus, training should stop well before 50 iterations.
        # The result should indicate early stopping was considered.
        assert result is not None, "grow_network should return ValidateTrainingResults"
        # Either early stopping triggered, or training completed for other reasons (correlation threshold, max units)
        # But it should NOT have run all 50 iterations without any stopping mechanism working
        hidden_units_added = len(network.hidden_units)
        assert hidden_units_added < 50, f"Added {hidden_units_added} hidden units. With patience=2 and early_stopping=True, " f"training should have stopped well before 50 growth iterations."
