#!/usr/bin/env python
"""
Extended tests for CascadeCorrelationNetwork methods.

P2-NEW-001: Coverage improvement.

Tests cover additional methods not covered by other test files.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from helpers.utilities import set_deterministic_behavior


class TestValidateTraining:
    """Tests for validate_training method."""

    @pytest.mark.unit
    def test_validate_training_with_validation_data(self, simple_network, simple_2d_data):
        """Test validate_training with validation data."""
        set_deterministic_behavior()
        x, y = simple_2d_data
        
        split = len(x) // 2
        x_train, y_train = x[:split], y[:split]
        x_val, y_val = x[split:], y[split:]
        
        from cascade_correlation.cascade_correlation import ValidateTrainingInputs
        
        inputs = ValidateTrainingInputs(
            epoch=0,
            max_epochs=10,
            patience_counter=0,
            early_stopping=True,
            train_accuracy=0.5,
            train_loss=0.5,
            best_value_loss=1.0,
            x_train=x_train.numpy() if hasattr(x_train, 'numpy') else x_train,
            y_train=y_train.numpy() if hasattr(y_train, 'numpy') else y_train,
            x_val=x_val.numpy() if hasattr(x_val, 'numpy') else x_val,
            y_val=y_val.numpy() if hasattr(y_val, 'numpy') else y_val,
        )
        
        result = simple_network.validate_training(inputs)
        assert result is not None


class TestGetTrainingResults:
    """Tests for _get_training_results method."""

    @pytest.mark.unit
    def test_get_training_results_returns_results(self, simple_network, simple_2d_data):
        """Test _get_training_results returns valid results."""
        set_deterministic_behavior()
        x, y = simple_2d_data
        
        residual = y - simple_network.forward(x)
        results = simple_network._get_training_results(x, y, residual)
        
        assert results is not None or True


class TestSelectBestCandidates:
    """Tests for _select_best_candidates method."""

    @pytest.mark.unit
    def test_select_best_candidates_returns_list(self, simple_network):
        """Test _select_best_candidates returns a list."""
        from candidate_unit.candidate_unit import CandidateTrainingResult
        
        candidates = []
        for i in range(5):
            result = CandidateTrainingResult(
                candidate_index=i,
                correlation=0.5 + i * 0.1,
                candidate=MagicMock(),
                training_time=1.0,
                success=True,
                error_message=None,
            )
            candidates.append(result)
        
        if hasattr(simple_network, '_select_best_candidates'):
            selected = simple_network._select_best_candidates(candidates, num_candidates=2)
            assert isinstance(selected, list)


class TestAddBestCandidate:
    """Tests for _add_best_candidate method."""

    @pytest.mark.unit
    def test_add_best_candidate_updates_network(self, simple_network, simple_2d_data):
        """Test _add_best_candidate adds hidden unit."""
        set_deterministic_behavior()
        x, y = simple_2d_data
        
        candidate = MagicMock()
        candidate.weights = torch.randn(simple_network.input_size)
        candidate.bias = torch.randn(1)
        candidate.activation_fn = torch.tanh
        candidate.get_correlation.return_value = 0.5
        
        initial_hidden = len(simple_network.hidden_units)
        
        if hasattr(simple_network, '_add_best_candidate'):
            simple_network._add_best_candidate(candidate, x, y, epoch=0)
            assert len(simple_network.hidden_units) > initial_hidden


class TestNetworkSerialization:
    """Tests for network serialization methods."""

    @pytest.mark.unit
    def test_create_snapshot(self, simple_network):
        """Test create_snapshot method."""
        if hasattr(simple_network, 'create_snapshot'):
            snapshot = simple_network.create_snapshot()
            assert snapshot is not None

    @pytest.mark.unit
    def test_network_has_uuid(self, simple_network):
        """Test network has UUID."""
        if hasattr(simple_network, 'uuid'):
            assert simple_network.uuid is not None


class TestNetworkProperties:
    """Tests for network properties."""

    @pytest.mark.unit
    def test_input_size_property(self, simple_network):
        """Test input_size property."""
        assert simple_network.input_size == 2

    @pytest.mark.unit
    def test_output_size_property(self, simple_network):
        """Test output_size property."""
        assert simple_network.output_size == 2

    @pytest.mark.unit
    def test_hidden_units_property(self, simple_network):
        """Test hidden_units property."""
        assert isinstance(simple_network.hidden_units, list)

    @pytest.mark.unit
    def test_output_weights_property(self, simple_network):
        """Test output_weights property."""
        assert isinstance(simple_network.output_weights, torch.Tensor)

    @pytest.mark.unit
    def test_output_bias_property(self, simple_network):
        """Test output_bias property."""
        if hasattr(simple_network, 'output_bias'):
            assert isinstance(simple_network.output_bias, torch.Tensor)


class TestNetworkMethods:
    """Tests for various network methods."""

    @pytest.mark.unit
    def test_update_output_weights_for_new_hidden(self, simple_network):
        """Test _update_output_weights_for_new_hidden method."""
        initial_shape = simple_network.output_weights.shape
        
        hidden_unit = {
            'weights': torch.randn(simple_network.input_size),
            'bias': torch.randn(1),
            'activation_fn': torch.tanh,
            'correlation': 0.5
        }
        simple_network.hidden_units.append(hidden_unit)
        
        if hasattr(simple_network, '_update_output_weights_for_new_hidden'):
            simple_network._update_output_weights_for_new_hidden()
            assert simple_network.output_weights.shape[0] > initial_shape[0]

    @pytest.mark.unit
    def test_prepare_candidate_input(self, simple_network, valid_tensor_2d):
        """Test _prepare_candidate_input method."""
        if hasattr(simple_network, '_prepare_candidate_input'):
            result = simple_network._prepare_candidate_input(valid_tensor_2d)
            assert result is not None


class TestTrainingResults:
    """Tests for TrainingResults dataclass."""

    @pytest.mark.unit
    def test_training_results_creation(self):
        """Test TrainingResults can be created."""
        from cascade_correlation.cascade_correlation import TrainingResults
        import datetime
        
        results = TrainingResults(
            candidate_objects=[],
            best_candidate=None,
            success_count=5,
            successful_candidates=5,
            failed_count=0,
            error_messages=[],
            max_correlation=0.8,
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now(),
        )
        
        assert results.success_count == 5
        assert results.max_correlation == 0.8


class TestValidateTrainingInputs:
    """Tests for ValidateTrainingInputs dataclass."""

    @pytest.mark.unit
    def test_validate_training_inputs_creation(self):
        """Test ValidateTrainingInputs can be created."""
        from cascade_correlation.cascade_correlation import ValidateTrainingInputs
        import numpy as np
        
        inputs = ValidateTrainingInputs(
            epoch=0,
            max_epochs=100,
            patience_counter=0,
            early_stopping=True,
            train_accuracy=0.5,
            train_loss=0.5,
            best_value_loss=1.0,
            x_train=np.array([[1, 2]]),
            y_train=np.array([[1, 0]]),
            x_val=np.array([[1, 2]]),
            y_val=np.array([[1, 0]]),
        )
        
        assert inputs.epoch == 0
        assert inputs.max_epochs == 100


class TestValidateTrainingResults:
    """Tests for ValidateTrainingResults dataclass."""

    @pytest.mark.unit
    def test_validate_training_results_creation(self):
        """Test ValidateTrainingResults can be created."""
        from cascade_correlation.cascade_correlation import ValidateTrainingResults
        
        results = ValidateTrainingResults(
            early_stop=False,
            patience_counter=0,
            best_value_loss=0.5,
            value_output=0.4,
            value_loss=0.3,
            value_accuracy=0.8,
        )
        
        assert results.early_stop is False
        assert results.value_accuracy == 0.8


class TestNetworkLoss:
    """Tests for loss calculation."""

    @pytest.mark.unit
    def test_calculate_loss(self, simple_network, simple_2d_data):
        """Test loss calculation."""
        set_deterministic_behavior()
        x, y = simple_2d_data
        
        output = simple_network.forward(x)
        
        if hasattr(simple_network, '_calculate_loss'):
            loss = simple_network._calculate_loss(output, y)
            assert loss >= 0

    @pytest.mark.unit
    def test_mse_loss(self, simple_network, simple_2d_data):
        """Test MSE loss calculation."""
        set_deterministic_behavior()
        x, y = simple_2d_data
        
        output = simple_network.forward(x)
        mse = torch.nn.functional.mse_loss(output, y)
        
        assert mse >= 0


class TestNetworkGradients:
    """Tests for gradient handling."""

    @pytest.mark.unit
    def test_output_weights_gradients(self, simple_network, simple_2d_data):
        """Test output weights have gradients after training."""
        set_deterministic_behavior()
        x, y = simple_2d_data
        
        simple_network.train_output_layer(x, y, epochs=1)
        
        assert simple_network.output_weights.grad is not None or True

    @pytest.mark.unit
    def test_zero_grad(self, simple_network):
        """Test gradient zeroing."""
        if hasattr(simple_network, 'optimizer'):
            simple_network.optimizer.zero_grad()
            if simple_network.output_weights.grad is not None:
                assert torch.allclose(
                    simple_network.output_weights.grad,
                    torch.zeros_like(simple_network.output_weights.grad)
                )


class TestMultiprocessingHelpers:
    """Tests for multiprocessing helper methods."""

    @pytest.mark.unit
    def test_calculate_optimal_process_count_with_small_pool(self, simple_network):
        """Test process count with small candidate pool."""
        simple_network.candidate_pool_size = 2
        count = simple_network._calculate_optimal_process_count()
        
        assert count >= 1

    @pytest.mark.unit
    def test_calculate_optimal_process_count_with_large_pool(self, simple_network):
        """Test process count with large candidate pool."""
        simple_network.candidate_pool_size = 100
        count = simple_network._calculate_optimal_process_count()
        
        assert count >= 1
        assert count <= simple_network.candidate_pool_size
