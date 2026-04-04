#!/usr/bin/env python
"""Regression tests for callback hooks added to core training loops."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from cascade_correlation.cascade_correlation import ValidateTrainingResults


@pytest.mark.unit
class TestOutputLayerEpochCallback:
    """Test throttled output-layer epoch callback behavior."""

    def test_train_output_layer_calls_explicit_callback_with_throttle(self, simple_network, simple_2d_data):
        """Explicit callback should fire at epoch 1, every 25 epochs, and final epoch."""
        x, y = simple_2d_data
        callback = MagicMock()

        with patch.object(simple_network, "create_snapshot", return_value=None):
            simple_network.train_output_layer(x, y, epochs=27, on_epoch_callback=callback)

        emitted_epochs = [call.kwargs["epoch"] for call in callback.call_args_list]
        assert emitted_epochs == [1, 26, 27]
        assert all(call.kwargs["epochs"] == 27 for call in callback.call_args_list)

    def test_train_output_layer_uses_attribute_callback_fallback(self, simple_network, simple_2d_data):
        """Fallback callback attribute should be used when explicit callback is not passed."""
        x, y = simple_2d_data
        fallback_callback = MagicMock()
        simple_network._output_epoch_callback = fallback_callback

        with patch.object(simple_network, "create_snapshot", return_value=None):
            simple_network.train_output_layer(x, y, epochs=27)

        emitted_epochs = [call.kwargs["epoch"] for call in fallback_callback.call_args_list]
        assert emitted_epochs == [1, 26, 27]

    def test_train_output_layer_prefers_explicit_callback_over_attribute(self, simple_network, simple_2d_data):
        """Explicit callback should take precedence over fallback attribute callback."""
        x, y = simple_2d_data
        explicit_callback = MagicMock()
        fallback_callback = MagicMock()
        simple_network._output_epoch_callback = fallback_callback

        with patch.object(simple_network, "create_snapshot", return_value=None):
            simple_network.train_output_layer(x, y, epochs=27, on_epoch_callback=explicit_callback)

        assert len(explicit_callback.call_args_list) == 3
        assert len(fallback_callback.call_args_list) == 0


@pytest.mark.unit
class TestGrowIterationCallback:
    """Test grow-network iteration callback behavior."""

    def test_grow_network_calls_explicit_iteration_callback(self, simple_network, simple_2d_data):
        """Explicit grow callback should receive live iteration metadata."""
        x, y = simple_2d_data
        grow_callback = MagicMock()

        mock_candidate = MagicMock()
        mock_candidate.get_correlation.return_value = 0.91
        mock_results = MagicMock()
        mock_results.best_candidate = mock_candidate
        mock_results.candidate_objects = [object(), object(), object()]

        validate_result = ValidateTrainingResults(
            early_stop=True,
            patience_counter=0,
            best_value_loss=0.1,
            value_output=None,
            value_loss=0.1,
            value_accuracy=0.8,
        )

        with patch.object(simple_network, "_calculate_residual_error_safe", return_value=torch.ones_like(y)):
            with patch.object(simple_network, "_get_training_results", return_value=mock_results):
                with patch.object(simple_network, "_add_best_candidate", return_value=(0.2, 0.8)):
                    with patch.object(simple_network, "validate_training", return_value=validate_result):
                        simple_network.grow_network(x, y, max_epochs=5, on_grow_iteration_callback=grow_callback)

        grow_callback.assert_called_once()
        callback_kwargs = grow_callback.call_args.kwargs
        assert callback_kwargs["iteration"] == 0
        assert callback_kwargs["max_iterations"] == simple_network.max_hidden_units
        assert callback_kwargs["best_correlation"] == 0.91
        assert callback_kwargs["candidates_trained"] == 3
        assert callback_kwargs["candidates_total"] == simple_network.candidate_pool_size
        assert callback_kwargs["phase_detail"] == "adding_candidate"

    def test_grow_network_uses_attribute_callback_fallback(self, simple_network, simple_2d_data):
        """Fallback grow callback attribute should be used when explicit callback is omitted."""
        x, y = simple_2d_data
        fallback_callback = MagicMock()
        simple_network._grow_iteration_callback = fallback_callback

        mock_candidate = MagicMock()
        mock_candidate.get_correlation.return_value = 0.91
        mock_results = MagicMock()
        mock_results.best_candidate = mock_candidate
        mock_results.candidate_objects = [object()]

        validate_result = ValidateTrainingResults(
            early_stop=True,
            patience_counter=0,
            best_value_loss=0.1,
            value_output=None,
            value_loss=0.1,
            value_accuracy=0.8,
        )

        with patch.object(simple_network, "_calculate_residual_error_safe", return_value=torch.ones_like(y)):
            with patch.object(simple_network, "_get_training_results", return_value=mock_results):
                with patch.object(simple_network, "_add_best_candidate", return_value=(0.2, 0.8)):
                    with patch.object(simple_network, "validate_training", return_value=validate_result):
                        simple_network.grow_network(x, y, max_epochs=1)

        fallback_callback.assert_called_once()
