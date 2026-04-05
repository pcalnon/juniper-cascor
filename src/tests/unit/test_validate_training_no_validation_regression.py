#!/usr/bin/env python
"""
Focused regression tests for validate_training() without validation data.

These tests cover the no-validation early-stopping branch restored in PR 72.
"""

import pytest

from cascade_correlation.cascade_correlation import ValidateTrainingInputs


class TestValidateTrainingNoValidationRegression:
    """Regression coverage for no-validation early-stopping behavior."""

    @pytest.mark.unit
    def test_improving_train_loss_updates_best_loss_and_resets_patience(self, simple_network, simple_2d_data):
        """Improving train loss should refresh best loss and clear patience counter."""
        x, y = simple_2d_data
        simple_network.patience = 3
        simple_network.convergence_threshold = 0.001
        simple_network.target_accuracy = 0.99

        inputs = ValidateTrainingInputs(
            iteration=2,
            max_iterations=10,
            patience_counter=2,
            early_stopping=True,
            train_accuracy=0.25,
            train_loss=0.10,
            best_value_loss=0.50,
            x_train=x,
            y_train=y,
            x_val=None,
            y_val=None,
        )

        result = simple_network.validate_training(inputs)

        assert result.early_stop is False
        assert result.best_value_loss == pytest.approx(0.10)
        assert result.patience_counter == 0
        assert result.value_loss == float("inf")
        assert result.value_accuracy == 0.0

    @pytest.mark.unit
    def test_non_improving_train_loss_exhausts_patience_and_stops(self, simple_network, simple_2d_data):
        """When loss does not improve and patience is exhausted, early stop should trigger."""
        x, y = simple_2d_data
        simple_network.patience = 1
        simple_network.convergence_threshold = 0.001
        simple_network.target_accuracy = 0.99

        inputs = ValidateTrainingInputs(
            iteration=3,
            max_iterations=10,
            patience_counter=0,
            early_stopping=True,
            train_accuracy=0.25,
            train_loss=0.60,
            best_value_loss=0.50,
            x_train=x,
            y_train=y,
            x_val=None,
            y_val=None,
        )

        result = simple_network.validate_training(inputs)

        assert result.early_stop is True
        assert result.best_value_loss == pytest.approx(0.50)
        assert result.patience_counter == 1
        assert result.value_loss == float("inf")

    @pytest.mark.unit
    def test_target_train_accuracy_triggers_early_stop_without_validation_data(self, simple_network, simple_2d_data):
        """High training accuracy alone should trigger stop in no-validation mode."""
        x, y = simple_2d_data
        simple_network.patience = 10
        simple_network.target_accuracy = 0.90
        simple_network.convergence_threshold = 0.001

        inputs = ValidateTrainingInputs(
            iteration=1,
            max_iterations=10,
            patience_counter=0,
            early_stopping=True,
            train_accuracy=0.95,
            train_loss=0.40,
            best_value_loss=0.50,
            x_train=x,
            y_train=y,
            x_val=None,
            y_val=None,
        )

        result = simple_network.validate_training(inputs)

        assert result.early_stop is True
        assert result.best_value_loss == pytest.approx(0.40)
        assert result.patience_counter == 0
