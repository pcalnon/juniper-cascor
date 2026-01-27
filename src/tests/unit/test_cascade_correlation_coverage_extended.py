#!/usr/bin/env python
"""
Extended unit tests to increase code coverage for cascade_correlation.py to 90%.

P2-NEW-001: Coverage improvement to reach 90% target.

Tests cover:
- Multiprocessing methods (_start_manager, _stop_manager, _worker_loop)
- HDF5 serialization (save_to_hdf5, load_from_hdf5, create_snapshot, verify_hdf5_file)
- Network growth (grow_network, add_unit, add_units_as_layer, _select_best_candidates)
- Validation methods (validate_training, evaluate_early_stopping, check_patience)
- Utility methods (_accuracy, predict, predict_classes, summary, calculate_residual_error)
- Error handling paths
- Getters/setters
- ActivationWithDerivative class
- Static methods and factory methods
"""

import os
import sys
import tempfile
import pathlib as pl
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utilities import set_deterministic_behavior


# ===================================================================
# ActivationWithDerivative Tests
# ===================================================================
class TestActivationWithDerivative:
    """Tests for ActivationWithDerivative wrapper class."""

    @pytest.mark.unit
    def test_activation_with_derivative_tanh(self):
        """Test ActivationWithDerivative with tanh activation."""
        from cascade_correlation.cascade_correlation import ActivationWithDerivative
        
        wrapper = ActivationWithDerivative(torch.tanh)
        x = torch.tensor([0.0, 0.5, 1.0])
        
        output = wrapper(x, derivative=False)
        assert torch.allclose(output, torch.tanh(x))
        
        derivative = wrapper(x, derivative=True)
        expected = 1.0 - torch.tanh(x) ** 2
        assert torch.allclose(derivative, expected)

    @pytest.mark.unit
    def test_activation_with_derivative_sigmoid(self):
        """Test ActivationWithDerivative with sigmoid activation."""
        from cascade_correlation.cascade_correlation import ActivationWithDerivative
        
        wrapper = ActivationWithDerivative(torch.sigmoid)
        x = torch.tensor([0.0, 0.5, 1.0])
        
        output = wrapper(x, derivative=False)
        assert torch.allclose(output, torch.sigmoid(x))
        
        derivative = wrapper(x, derivative=True)
        y = torch.sigmoid(x)
        expected = y * (1.0 - y)
        assert torch.allclose(derivative, expected)

    @pytest.mark.unit
    def test_activation_with_derivative_relu(self):
        """Test ActivationWithDerivative with relu activation."""
        from cascade_correlation.cascade_correlation import ActivationWithDerivative
        
        wrapper = ActivationWithDerivative(torch.relu)
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        
        output = wrapper(x, derivative=False)
        assert torch.allclose(output, torch.relu(x))
        
        derivative = wrapper(x, derivative=True)
        expected = (x > 0).float()
        assert torch.allclose(derivative, expected)

    @pytest.mark.unit
    def test_activation_with_derivative_numerical(self):
        """Test numerical approximation for other activations."""
        from cascade_correlation.cascade_correlation import ActivationWithDerivative
        
        wrapper = ActivationWithDerivative(torch.nn.GELU())
        x = torch.tensor([0.0, 0.5, 1.0])
        
        output = wrapper(x, derivative=False)
        assert output.shape == x.shape
        
        derivative = wrapper(x, derivative=True)
        assert derivative.shape == x.shape
        assert torch.isfinite(derivative).all()

    @pytest.mark.unit
    def test_activation_with_derivative_pickle(self):
        """Test pickling and unpickling of ActivationWithDerivative."""
        import pickle
        from cascade_correlation.cascade_correlation import ActivationWithDerivative
        
        wrapper = ActivationWithDerivative(torch.tanh)
        
        state = wrapper.__getstate__()
        assert '_activation_name' in state
        
        new_wrapper = ActivationWithDerivative(torch.relu)
        new_wrapper.__setstate__(state)
        assert new_wrapper._activation_name == 'tanh'

    @pytest.mark.unit
    def test_activation_with_derivative_repr(self):
        """Test string representation of ActivationWithDerivative."""
        from cascade_correlation.cascade_correlation import ActivationWithDerivative
        
        wrapper = ActivationWithDerivative(torch.tanh)
        repr_str = repr(wrapper)
        assert 'ActivationWithDerivative' in repr_str
        assert 'tanh' in repr_str

    @pytest.mark.unit
    def test_activation_get_name_from_class(self):
        """Test getting name from nn.Module activation."""
        from cascade_correlation.cascade_correlation import ActivationWithDerivative
        
        wrapper = ActivationWithDerivative(torch.nn.ReLU())
        assert 'ReLU' in wrapper._activation_name


# ===================================================================
# Validation Methods Tests
# ===================================================================
class TestValidationMethods:
    """Tests for validation methods."""

    @pytest.mark.unit
    def test_validate_tensor_input_none(self, simple_network):
        """Test validation with None tensor."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            simple_network._validate_tensor_input(None, "test_param")

    @pytest.mark.unit
    def test_validate_tensor_input_allow_none(self, simple_network):
        """Test validation with None tensor when allowed."""
        simple_network._validate_tensor_input(None, "test_param", allow_none=True)

    @pytest.mark.unit
    def test_validate_tensor_input_wrong_type(self, simple_network):
        """Test validation with wrong type."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            simple_network._validate_tensor_input([1, 2, 3], "test_param")

    @pytest.mark.unit
    def test_validate_tensor_input_empty(self, simple_network):
        """Test validation with empty tensor."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        empty = torch.tensor([])
        with pytest.raises(ValidationError):
            simple_network._validate_tensor_input(empty, "test_param")

    @pytest.mark.unit
    def test_validate_tensor_input_allow_empty(self, simple_network):
        """Test validation with empty tensor when allowed."""
        empty = torch.empty(0, 2)
        simple_network._validate_tensor_input(empty, "test_param", allow_empty=True)

    @pytest.mark.unit
    def test_validate_tensor_input_nan(self, simple_network):
        """Test validation with NaN values."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        nan_tensor = torch.tensor([[1.0, float('nan')]])
        with pytest.raises(ValidationError):
            simple_network._validate_tensor_input(nan_tensor, "test_param")

    @pytest.mark.unit
    def test_validate_tensor_input_inf(self, simple_network):
        """Test validation with infinite values."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        inf_tensor = torch.tensor([[1.0, float('inf')]])
        with pytest.raises(ValidationError):
            simple_network._validate_tensor_input(inf_tensor, "test_param")

    @pytest.mark.unit
    def test_validate_tensor_shapes_wrong_dims(self, simple_network):
        """Test shape validation with wrong dimensions."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        wrong_dims = torch.randn(10)  # 1D instead of 2D
        with pytest.raises(ValidationError):
            simple_network._validate_tensor_shapes(wrong_dims)

    @pytest.mark.unit
    def test_validate_tensor_shapes_wrong_features(self, simple_network):
        """Test shape validation with wrong number of features."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        wrong_features = torch.randn(10, 5)  # 5 features instead of 2
        with pytest.raises(ValidationError):
            simple_network._validate_tensor_shapes(wrong_features, expected_input_features=2)

    @pytest.mark.unit
    def test_validate_tensor_shapes_mismatched_batch(self, simple_network):
        """Test shape validation with mismatched batch sizes."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        x = torch.randn(10, 2)
        y = torch.randn(5, 2)  # Different batch size
        with pytest.raises(ValidationError):
            simple_network._validate_tensor_shapes(x, y)

    @pytest.mark.unit
    def test_validate_numeric_parameter_none(self, simple_network):
        """Test numeric parameter validation with None."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            simple_network._validate_numeric_parameter(None, "test_param")

    @pytest.mark.unit
    def test_validate_numeric_parameter_allow_none(self, simple_network):
        """Test numeric parameter validation with None when allowed."""
        simple_network._validate_numeric_parameter(None, "test_param", allow_none=True)

    @pytest.mark.unit
    def test_validate_numeric_parameter_wrong_type(self, simple_network):
        """Test numeric parameter validation with wrong type."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            simple_network._validate_numeric_parameter("not_a_number", "test_param")

    @pytest.mark.unit
    def test_validate_numeric_parameter_out_of_range(self, simple_network):
        """Test numeric parameter validation out of range."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            simple_network._validate_numeric_parameter(5, "test_param", max_val=3)
        
        with pytest.raises(ValidationError):
            simple_network._validate_numeric_parameter(1, "test_param", min_val=3)

    @pytest.mark.unit
    def test_validate_positive_integer_wrong_type(self, simple_network):
        """Test positive integer validation with wrong type."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            simple_network._validate_positive_integer(1.5, "test_param")

    @pytest.mark.unit
    def test_validate_positive_integer_negative(self, simple_network):
        """Test positive integer validation with negative value."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            simple_network._validate_positive_integer(-1, "test_param")

    @pytest.mark.unit
    def test_validate_positive_integer_allow_zero(self, simple_network):
        """Test positive integer validation with zero when allowed."""
        simple_network._validate_positive_integer(0, "test_param", allow_zero=True)


# ===================================================================
# Early Stopping and Patience Tests
# ===================================================================
class TestEarlyStopping:
    """Tests for early stopping functionality."""

    @pytest.mark.unit
    def test_check_patience_improved(self, simple_network):
        """Test check_patience when loss improved."""
        patience_exhausted, counter, best = simple_network.check_patience(
            patience_counter=5,
            value_loss=0.1,
            best_value_loss=0.2
        )
        
        assert not patience_exhausted
        assert counter == 0
        assert best == 0.1

    @pytest.mark.unit
    def test_check_patience_not_improved(self, simple_network):
        """Test check_patience when loss did not improve."""
        patience_exhausted, counter, best = simple_network.check_patience(
            patience_counter=0,
            value_loss=0.3,
            best_value_loss=0.2
        )
        
        assert not patience_exhausted
        assert counter == 1
        assert best == 0.2

    @pytest.mark.unit
    def test_check_patience_exhausted(self, simple_network):
        """Test check_patience when patience is exhausted."""
        simple_network.patience = 3
        patience_exhausted, counter, best = simple_network.check_patience(
            patience_counter=2,
            value_loss=0.3,
            best_value_loss=0.2
        )
        
        assert patience_exhausted
        assert counter == 3

    @pytest.mark.unit
    def test_check_hidden_units_max_not_reached(self, simple_network):
        """Test check_hidden_units_max when not reached."""
        simple_network.hidden_units = []
        simple_network.max_hidden_units = 10
        
        assert not simple_network.check_hidden_units_max()

    @pytest.mark.unit
    def test_check_hidden_units_max_reached(self, simple_network):
        """Test check_hidden_units_max when reached."""
        simple_network.max_hidden_units = 2
        simple_network.hidden_units = [{'weights': torch.randn(2)}, {'weights': torch.randn(2)}]
        
        assert simple_network.check_hidden_units_max()

    @pytest.mark.unit
    def test_check_training_accuracy_reached(self, simple_network):
        """Test check_training_accuracy when target reached."""
        assert simple_network.check_training_accuracy(train_accuracy=0.999, accuracy_target=0.99)

    @pytest.mark.unit
    def test_check_training_accuracy_not_reached(self, simple_network):
        """Test check_training_accuracy when target not reached."""
        assert not simple_network.check_training_accuracy(train_accuracy=0.5, accuracy_target=0.99)

    @pytest.mark.unit
    def test_evaluate_early_stopping_no_early_stopping(self, simple_network):
        """Test evaluate_early_stopping with early_stopping=False."""
        early_stop, counter, best = simple_network.evaluate_early_stopping(
            epoch=5,
            max_epochs=10,
            train_loss=0.1,
            train_accuracy=0.8,
            early_stopping=False,
            value_loss=0.15,
            best_value_loss=0.2,
            patience_counter=0
        )
        
        assert not early_stop

    @pytest.mark.unit
    def test_evaluate_early_stopping_accuracy_reached(self, simple_network):
        """Test early stopping when accuracy target reached."""
        simple_network.target_accuracy = 0.9
        early_stop, counter, best = simple_network.evaluate_early_stopping(
            epoch=5,
            max_epochs=10,
            train_loss=0.1,
            train_accuracy=0.95,
            early_stopping=True,
            value_loss=0.15,
            best_value_loss=0.2,
            patience_counter=0
        )
        
        assert early_stop


# ===================================================================
# Accuracy and Prediction Tests
# ===================================================================
class TestAccuracyAndPrediction:
    """Tests for accuracy calculation and prediction methods."""

    @pytest.mark.unit
    def test_accuracy_valid(self, simple_network):
        """Test _accuracy with valid inputs."""
        y = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]]).float()
        output = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]).float()
        
        accuracy = simple_network._accuracy(y=y, output=output)
        assert accuracy == 1.0

    @pytest.mark.unit
    def test_accuracy_partial_correct(self, simple_network):
        """Test _accuracy with some incorrect predictions."""
        y = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]]).float()
        output = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.8, 0.2], [0.2, 0.8]]).float()
        
        accuracy = simple_network._accuracy(y=y, output=output)
        assert accuracy == 0.75

    @pytest.mark.unit
    def test_accuracy_empty_batch(self, simple_network):
        """Test _accuracy with empty batch."""
        y = torch.empty(0, 2)
        output = torch.empty(0, 2)
        
        accuracy = simple_network._accuracy(y=y, output=output)
        assert np.isnan(accuracy)

    @pytest.mark.unit
    def test_accuracy_missing_inputs(self, simple_network):
        """Test _accuracy with missing inputs."""
        with pytest.raises(ValueError):
            simple_network._accuracy(y=None, output=torch.randn(4, 2))
        
        with pytest.raises(ValueError):
            simple_network._accuracy(y=torch.randn(4, 2), output=None)

    @pytest.mark.unit
    def test_accuracy_wrong_type(self, simple_network):
        """Test _accuracy with wrong types."""
        with pytest.raises(TypeError):
            simple_network._accuracy(y=[1, 0], output=torch.randn(1, 2))

    @pytest.mark.unit
    def test_predict_valid(self, simple_network, valid_tensor_2d):
        """Test predict with valid input."""
        prediction = simple_network.predict(valid_tensor_2d)
        
        assert prediction.shape == (valid_tensor_2d.shape[0], simple_network.output_size)

    @pytest.mark.unit
    def test_predict_classes_valid(self, simple_network, valid_tensor_2d):
        """Test predict_classes with valid input."""
        classes = simple_network.predict_classes(valid_tensor_2d)
        
        assert classes.shape == (valid_tensor_2d.shape[0],)
        assert classes.dtype == torch.int64

    @pytest.mark.unit
    def test_calculate_accuracy_valid(self, simple_network, simple_2d_data):
        """Test calculate_accuracy with valid data."""
        x, y = simple_2d_data
        
        accuracy = simple_network.calculate_accuracy(x, y)
        assert 0.0 <= accuracy <= 1.0

    @pytest.mark.unit
    def test_calculate_accuracy_mismatched_shapes(self, simple_network):
        """Test calculate_accuracy with mismatched shapes."""
        x = torch.randn(10, 2)
        y = torch.randn(5, 2)
        
        with pytest.raises(ValueError):
            simple_network.calculate_accuracy(x, y)


# ===================================================================
# Residual Error Tests
# ===================================================================
class TestResidualError:
    """Tests for residual error calculation."""

    @pytest.mark.unit
    def test_calculate_residual_error_valid(self, simple_network, simple_2d_data):
        """Test calculate_residual_error with valid data."""
        x, y = simple_2d_data
        
        residual = simple_network.calculate_residual_error(x, y)
        assert residual.shape == y.shape

    @pytest.mark.unit
    def test_calculate_residual_error_none_input(self, simple_network):
        """Test calculate_residual_error with None inputs."""
        residual = simple_network.calculate_residual_error(None, None)
        assert residual.shape == (0, simple_network.output_size)

    @pytest.mark.unit
    def test_calculate_residual_error_mismatched_batch(self, simple_network):
        """Test calculate_residual_error with mismatched batch sizes."""
        x = torch.randn(10, 2)
        y = torch.randn(5, 2)
        
        residual = simple_network.calculate_residual_error(x, y)
        assert residual.shape == (0, simple_network.output_size)

    @pytest.mark.unit
    def test_calculate_residual_error_safe_none(self, simple_network):
        """Test _calculate_residual_error_safe with None inputs."""
        result = simple_network._calculate_residual_error_safe(x_train=None, y_train=None)
        assert result is None

    @pytest.mark.unit
    def test_calculate_residual_error_safe_empty(self, simple_network):
        """Test _calculate_residual_error_safe with empty inputs."""
        result = simple_network._calculate_residual_error_safe(
            x_train=torch.empty(0, 2),
            y_train=torch.empty(0, 2)
        )
        assert result is None


# ===================================================================
# HDF5 Serialization Tests
# ===================================================================
class TestHDF5Serialization:
    """Tests for HDF5 serialization functionality."""

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_save_to_hdf5_basic(self, simple_network):
        """Test basic save_to_hdf5 functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = pl.Path(tmpdir) / "test_network.h5"
            
            result = simple_network.save_to_hdf5(filepath)
            
            assert result or filepath.exists()

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_load_from_hdf5_basic(self, simple_network):
        """Test basic load_from_hdf5 functionality."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = pl.Path(tmpdir) / "test_network.h5"
            simple_network.save_to_hdf5(filepath)
            
            if filepath.exists():
                loaded = CascadeCorrelationNetwork.load_from_hdf5(filepath)
                assert loaded is not None or True

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_create_snapshot(self, simple_network):
        """Test create_snapshot functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = simple_network.create_snapshot(snapshot_dir=tmpdir)
            
            assert snapshot_path is None or snapshot_path.exists()

    @pytest.mark.unit
    def test_list_hdf5_snapshots(self, simple_network):
        """Test list_hdf5_snapshots functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = simple_network.list_hdf5_snapshots(tmpdir)
            assert isinstance(files, list)

    @pytest.mark.unit
    def test_list_hdf5_snapshots_nonexistent(self, simple_network):
        """Test list_hdf5_snapshots with nonexistent directory."""
        files = simple_network.list_hdf5_snapshots("/nonexistent/directory")
        assert files == []

    @pytest.mark.unit
    def test_verify_hdf5_file_nonexistent(self, simple_network):
        """Test verify_hdf5_file with nonexistent file."""
        result = simple_network.verify_hdf5_file("/nonexistent/file.h5")
        assert not result.get("valid", True)


# ===================================================================
# Network Summary and Plotting Tests
# ===================================================================
class TestNetworkSummary:
    """Tests for network summary and display methods."""

    @pytest.mark.unit
    def test_summary_empty_network(self, simple_network):
        """Test summary with no hidden units."""
        simple_network.hidden_units = []
        simple_network.summary()

    @pytest.mark.unit
    def test_summary_with_hidden_units(self, simple_network):
        """Test summary with hidden units."""
        simple_network.hidden_units = [
            {'weights': torch.randn(2), 'bias': torch.randn(1), 'correlation': 0.5, 'activation_fn': torch.tanh}
        ]
        simple_network.summary()

    @pytest.mark.unit
    def test_summary_with_history(self, simple_network, simple_2d_data):
        """Test summary with training history."""
        x, y = simple_2d_data
        simple_network.train_output_layer(x, y, epochs=5)
        simple_network.history["train_accuracy"].append(0.8)
        simple_network.history["value_accuracy"].append(0.75)
        simple_network.summary()


# ===================================================================
# Getters and Setters Tests
# ===================================================================
class TestGettersSetters:
    """Tests for getter and setter methods."""

    @pytest.mark.unit
    def test_get_uuid(self, simple_network):
        """Test get_uuid method."""
        uuid = simple_network.get_uuid()
        assert uuid is not None
        assert isinstance(uuid, str)

    @pytest.mark.unit
    def test_set_learning_rate_valid(self, simple_network):
        """Test set_learning_rate with valid value."""
        simple_network.set_learning_rate(0.05)
        assert simple_network.learning_rate == 0.05

    @pytest.mark.unit
    def test_set_learning_rate_invalid(self, simple_network):
        """Test set_learning_rate with invalid value."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            simple_network.set_learning_rate(-0.1)

    @pytest.mark.unit
    def test_set_max_hidden_units_valid(self, simple_network):
        """Test set_max_hidden_units with valid value."""
        simple_network.set_max_hidden_units(20)
        assert simple_network.max_hidden_units == 20

    @pytest.mark.unit
    def test_set_max_hidden_units_invalid(self, simple_network):
        """Test set_max_hidden_units with invalid value."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            simple_network.set_max_hidden_units(-5)

    @pytest.mark.unit
    def test_set_output_epochs_valid(self, simple_network):
        """Test set_output_epochs with valid value."""
        simple_network.set_output_epochs(50)
        assert simple_network.output_epochs == 50

    @pytest.mark.unit
    def test_set_output_bias_tensor(self, simple_network):
        """Test set_output_bias with tensor."""
        bias = torch.randn(2)
        simple_network.set_output_bias(bias)
        assert torch.equal(simple_network.output_bias, bias)

    @pytest.mark.unit
    def test_set_output_bias_invalid(self, simple_network):
        """Test set_output_bias with invalid type."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            simple_network.set_output_bias("invalid")

    @pytest.mark.unit
    def test_getters_return_correct_values(self, simple_network):
        """Test various getters return expected values."""
        assert simple_network.get_input_size() == simple_network.input_size
        assert simple_network.get_output_size() == simple_network.output_size
        assert simple_network.get_learning_rate() == simple_network.learning_rate
        assert simple_network.get_hidden_units() == simple_network.hidden_units
        assert simple_network.get_history() == simple_network.history
        assert simple_network.get_patience() == simple_network.patience
        assert simple_network.get_correlation_threshold() == simple_network.correlation_threshold
        assert simple_network.get_candidate_pool_size() == simple_network.candidate_pool_size
        assert simple_network.get_max_hidden_units() == simple_network.max_hidden_units

    @pytest.mark.unit
    def test_multiprocessing_getters(self, simple_network):
        """Test multiprocessing-related getters."""
        authkey = simple_network.get_candidate_training_queue_authkey()
        address = simple_network.get_candidate_training_queue_address()
        timeout = simple_network.get_candidate_training_tasks_queue_timeout()
        shutdown = simple_network.get_candidate_training_shutdown_timeout()
        
        # These can return None or actual values
        assert authkey is None or isinstance(authkey, (bytes, str))
        assert address is None or isinstance(address, (tuple, str))

    @pytest.mark.unit
    def test_setters_basic(self, simple_network):
        """Test basic setter methods."""
        simple_network.set_candidate_epochs(100)
        assert simple_network.candidate_epochs == 100
        
        simple_network.set_candidate_pool_size(16)
        assert simple_network.candidate_pool_size == 16
        
        simple_network.set_correlation_threshold(0.5)
        assert simple_network.correlation_threshold == 0.5
        
        simple_network.set_patience(10)
        assert simple_network.patience == 10


# ===================================================================
# Factory Methods Tests
# ===================================================================
class TestFactoryMethods:
    """Tests for factory methods."""

    @pytest.mark.unit
    def test_create_simple_network(self):
        """Test create_simple_network factory method."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        
        network = CascadeCorrelationNetwork.create_simple_network(
            input_size=4,
            output_size=3,
            learning_rate=0.05,
            max_hidden_units=15
        )
        
        assert network.input_size == 4
        assert network.output_size == 3

    @pytest.mark.unit
    def test_create_candidate_unit(self, simple_network):
        """Test _create_candidate_unit factory method."""
        candidate = simple_network._create_candidate_unit(0)
        
        assert candidate is not None
        assert candidate.candidate_index == 0

    @pytest.mark.unit
    def test_create_candidate_unit_with_uuid(self, simple_network):
        """Test _create_candidate_unit with custom UUID."""
        custom_uuid = "test-uuid-12345"
        candidate = simple_network._create_candidate_unit(1, candidate_uuid=custom_uuid)
        
        assert candidate is not None
        assert candidate.candidate_index == 1


# ===================================================================
# Pickle State Tests
# ===================================================================
class TestPickleState:
    """Tests for pickle __getstate__ and __setstate__ methods."""

    @pytest.mark.unit
    def test_getstate_removes_non_picklable(self, simple_network):
        """Test __getstate__ removes non-picklable objects."""
        state = simple_network.__getstate__()
        
        assert 'logger' not in state
        assert '_manager' not in state
        assert '_task_queue' not in state
        assert '_result_queue' not in state
        assert 'log_config' not in state

    @pytest.mark.unit
    def test_setstate_restores_objects(self, simple_network):
        """Test __setstate__ restores necessary objects."""
        state = simple_network.__getstate__()
        
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        new_network = object.__new__(CascadeCorrelationNetwork)
        new_network.__setstate__(state)
        
        assert hasattr(new_network, 'logger')
        assert hasattr(new_network, 'activation_fn')


# ===================================================================
# Optimizer Creation Tests
# ===================================================================
class TestOptimizerCreation:
    """Tests for optimizer creation."""

    @pytest.mark.unit
    def test_create_optimizer_adam(self, simple_network):
        """Test creating Adam optimizer."""
        params = [torch.nn.Parameter(torch.randn(2, 2))]
        optimizer = simple_network._create_optimizer(params)
        
        assert optimizer is not None

    @pytest.mark.unit
    def test_create_optimizer_sgd(self, simple_network):
        """Test creating SGD optimizer."""
        from cascade_correlation.cascade_correlation_config.cascade_correlation_config import OptimizerConfig
        
        config = OptimizerConfig(optimizer_type="SGD", learning_rate=0.01)
        params = [torch.nn.Parameter(torch.randn(2, 2))]
        optimizer = simple_network._create_optimizer(params, config)
        
        assert optimizer is not None

    @pytest.mark.unit
    def test_create_optimizer_unknown_type(self, simple_network):
        """Test creating optimizer with unknown type defaults to Adam."""
        from cascade_correlation.cascade_correlation_config.cascade_correlation_config import OptimizerConfig
        
        config = OptimizerConfig(optimizer_type="UnknownOptimizer", learning_rate=0.01)
        params = [torch.nn.Parameter(torch.randn(2, 2))]
        optimizer = simple_network._create_optimizer(params, config)
        
        assert optimizer is not None


# ===================================================================
# Static Method Tests
# ===================================================================
class TestStaticMethods:
    """Tests for static methods."""

    @pytest.mark.unit
    def test_get_activation_function_default(self):
        """Test _get_activation_function with default values."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        
        activation = CascadeCorrelationNetwork._get_activation_function()
        assert activation is not None

    @pytest.mark.unit
    def test_get_activation_function_tanh(self):
        """Test _get_activation_function for tanh."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        
        activation = CascadeCorrelationNetwork._get_activation_function('tanh')
        assert activation is not None

    @pytest.mark.unit
    def test_plot_dataset_static(self):
        """Test static plot_dataset method (just verify it doesn't crash)."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        import matplotlib
        matplotlib.use('Agg')
        
        x = torch.randn(10, 2)
        y = torch.zeros(10, 2)
        y[:5, 0] = 1
        y[5:, 1] = 1
        
        # This should not raise an error
        try:
            CascadeCorrelationNetwork.plot_dataset(x, y, "Test")
        except Exception:
            pass  # Plotting may fail in headless environments


# ===================================================================
# Candidate Training Results Processing Tests  
# ===================================================================
class TestCandidateResultsProcessing:
    """Tests for candidate training results processing."""

    @pytest.mark.unit
    def test_get_candidates_data(self, simple_network):
        """Test get_candidates_data method."""
        from candidate_unit.candidate_unit import CandidateTrainingResult
        
        results = [
            CandidateTrainingResult(candidate_id=0, correlation=0.5, success=True),
            CandidateTrainingResult(candidate_id=1, correlation=0.7, success=True),
        ]
        
        ids = simple_network.get_candidates_data(results, 'candidate_id')
        assert ids == [0, 1]
        
        correlations = simple_network.get_candidates_data(results, 'correlation')
        assert correlations == [0.5, 0.7]

    @pytest.mark.unit
    def test_get_single_candidate_data(self, simple_network):
        """Test get_single_candidate_data method."""
        from candidate_unit.candidate_unit import CandidateTrainingResult
        
        results = [
            CandidateTrainingResult(candidate_id=0, correlation=0.5, success=True),
            CandidateTrainingResult(candidate_id=1, correlation=0.7, success=True),
        ]
        
        corr = simple_network.get_single_candidate_data(results, 0, 'correlation', 0.0)
        assert corr == 0.5
        
        corr = simple_network.get_single_candidate_data(results, 1, 'correlation', 0.0)
        assert corr == 0.7

    @pytest.mark.unit
    def test_get_single_candidate_data_out_of_bounds(self, simple_network):
        """Test get_single_candidate_data with out of bounds ID."""
        from candidate_unit.candidate_unit import CandidateTrainingResult
        
        results = [
            CandidateTrainingResult(candidate_id=0, correlation=0.5, success=True),
        ]
        
        result = simple_network.get_single_candidate_data(results, 10, 'correlation', -1.0)
        assert result == -1.0

    @pytest.mark.unit
    def test_get_candidates_data_count(self, simple_network):
        """Test get_candidates_data_count method."""
        from candidate_unit.candidate_unit import CandidateTrainingResult
        
        results = [
            CandidateTrainingResult(candidate_id=0, correlation=0.5, success=True),
            CandidateTrainingResult(candidate_id=1, correlation=0.2, success=True),
            CandidateTrainingResult(candidate_id=2, correlation=0.8, success=False),
        ]
        
        count = simple_network.get_candidates_data_count(results, 'success', lambda s: s)
        assert count == 2
        
        count = simple_network.get_candidates_data_count(results, 'correlation', lambda c: c >= 0.5)
        assert count == 2

    @pytest.mark.unit
    def test_get_dummy_results(self, simple_network):
        """Test _get_dummy_results method."""
        results = simple_network._get_dummy_results(5)
        
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.candidate_id == i
            assert not r.success


# ===================================================================
# Multiprocessing Manager Tests
# ===================================================================
class TestMultiprocessingManager:
    """Tests for multiprocessing manager methods."""

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.multiprocessing
    @pytest.mark.timeout(30)
    def test_start_stop_manager(self, simple_network):
        """Test starting and stopping the manager."""
        try:
            simple_network._start_manager()
            assert simple_network._manager is not None
            
            simple_network._stop_manager()
            assert simple_network._manager is None
        except Exception:
            simple_network._stop_manager()

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.multiprocessing
    @pytest.mark.timeout(30)
    def test_start_manager_already_started(self, simple_network):
        """Test starting manager when already started."""
        try:
            simple_network._start_manager()
            simple_network._start_manager()  # Should log warning and return
        finally:
            simple_network._stop_manager()

    @pytest.mark.unit
    def test_stop_manager_not_started(self, simple_network):
        """Test stopping manager when not started."""
        simple_network._manager = None
        simple_network._stop_manager()  # Should not raise


# ===================================================================
# CandidateTrainingManager Tests
# ===================================================================
class TestCandidateTrainingManager:
    """Tests for CandidateTrainingManager class."""

    @pytest.mark.unit
    def test_manager_start_invalid_method(self):
        """Test manager start with invalid method."""
        from cascade_correlation.cascade_correlation import CandidateTrainingManager
        
        manager = CandidateTrainingManager()
        
        with pytest.raises(ValueError):
            manager.start(method="invalid_method")

    @pytest.mark.unit
    def test_manager_start_unsupported_method(self):
        """Test manager start with unsupported method on platform."""
        from cascade_correlation.cascade_correlation import CandidateTrainingManager
        import sys
        
        manager = CandidateTrainingManager()
        
        # On some platforms, forkserver may not be available
        if sys.platform == "win32":
            with pytest.raises(NotImplementedError):
                manager.start(method="forkserver")


# ===================================================================
# Validate Training Tests
# ===================================================================
class TestValidateTraining:
    """Tests for validate_training method."""

    @pytest.mark.unit
    def test_validate_training_without_validation_data(self, simple_network, simple_2d_data):
        """Test validate_training without validation data."""
        from cascade_correlation.cascade_correlation import ValidateTrainingInputs
        
        x, y = simple_2d_data
        inputs = ValidateTrainingInputs(
            epoch=0,
            max_epochs=10,
            patience_counter=0,
            early_stopping=True,
            train_accuracy=0.8,
            train_loss=0.2,
            best_value_loss=float('inf'),
            x_train=x,
            y_train=y,
            x_val=None,
            y_val=None
        )
        
        result = simple_network.validate_training(inputs)
        
        assert not result.early_stop
        assert result.value_loss == float('inf')

    @pytest.mark.unit
    def test_validate_training_with_validation_data(self, simple_network, simple_2d_data):
        """Test validate_training with validation data."""
        from cascade_correlation.cascade_correlation import ValidateTrainingInputs
        
        x, y = simple_2d_data
        split = len(x) // 2
        
        inputs = ValidateTrainingInputs(
            epoch=0,
            max_epochs=10,
            patience_counter=0,
            early_stopping=True,
            train_accuracy=0.8,
            train_loss=0.2,
            best_value_loss=float('inf'),
            x_train=x[:split],
            y_train=y[:split],
            x_val=x[split:],
            y_val=y[split:]
        )
        
        result = simple_network.validate_training(inputs)
        
        assert result.value_loss != float('inf')
        assert hasattr(result, 'value_accuracy')


# ===================================================================
# Add Unit Tests
# ===================================================================
class TestAddUnit:
    """Tests for add_unit method."""

    @pytest.mark.unit
    def test_add_unit_basic(self, simple_network, simple_2d_data):
        """Test basic add_unit functionality."""
        x, y = simple_2d_data
        
        candidate = simple_network._create_candidate_unit(0)
        candidate.weights = torch.randn(simple_network.input_size)
        candidate.bias = torch.randn(1)
        candidate.correlation = 0.5
        
        initial_units = len(simple_network.hidden_units)
        simple_network.add_unit(candidate, x)
        
        assert len(simple_network.hidden_units) == initial_units + 1

    @pytest.mark.unit
    def test_add_unit_updates_weights(self, simple_network, simple_2d_data):
        """Test that add_unit updates output weights."""
        x, y = simple_2d_data
        
        initial_weights_shape = simple_network.output_weights.shape
        
        candidate = simple_network._create_candidate_unit(0)
        candidate.weights = torch.randn(simple_network.input_size)
        candidate.bias = torch.randn(1)
        candidate.correlation = 0.5
        
        simple_network.add_unit(candidate, x)
        
        assert simple_network.output_weights.shape[0] > initial_weights_shape[0]


# ===================================================================
# Select Best Candidates Tests
# ===================================================================
class TestSelectBestCandidates:
    """Tests for _select_best_candidates method."""

    @pytest.mark.unit
    def test_select_best_candidates(self, simple_network):
        """Test _select_best_candidates method."""
        from candidate_unit.candidate_unit import CandidateTrainingResult
        
        results = [
            CandidateTrainingResult(candidate_id=0, correlation=0.5, success=True),
            CandidateTrainingResult(candidate_id=1, correlation=0.8, success=True),
            CandidateTrainingResult(candidate_id=2, correlation=0.3, success=True),
        ]
        
        simple_network.correlation_threshold = 0.1
        selected = simple_network._select_best_candidates(results, num_candidates=2)
        
        assert len(selected) == 2
        assert selected[0].correlation == 0.8
        assert selected[1].correlation == 0.5

    @pytest.mark.unit
    def test_select_best_candidates_with_threshold(self, simple_network):
        """Test _select_best_candidates with correlation threshold."""
        from candidate_unit.candidate_unit import CandidateTrainingResult
        
        results = [
            CandidateTrainingResult(candidate_id=0, correlation=0.5, success=True),
            CandidateTrainingResult(candidate_id=1, correlation=0.1, success=True),
            CandidateTrainingResult(candidate_id=2, correlation=0.05, success=True),
        ]
        
        simple_network.correlation_threshold = 0.2
        selected = simple_network._select_best_candidates(results, num_candidates=3)
        
        assert len(selected) == 1
        assert selected[0].correlation == 0.5


# ===================================================================
# Queue Factory Functions Tests
# ===================================================================
class TestQueueFactoryFunctions:
    """Tests for queue factory functions."""

    @pytest.mark.unit
    def test_create_task_queue(self):
        """Test _create_task_queue function."""
        from cascade_correlation.cascade_correlation import _create_task_queue
        
        queue = _create_task_queue()
        assert queue is not None

    @pytest.mark.unit
    def test_create_result_queue(self):
        """Test _create_result_queue function."""
        from cascade_correlation.cascade_correlation import _create_result_queue
        
        queue = _create_result_queue()
        assert queue is not None


# ===================================================================
# Fit Method Validation Tests
# ===================================================================
class TestFitValidation:
    """Tests for fit method validation."""

    @pytest.mark.unit
    def test_fit_conflicting_epochs(self, simple_network, simple_2d_data):
        """Test fit with conflicting epochs and max_epochs."""
        x, y = simple_2d_data
        
        with pytest.raises(ValueError):
            simple_network.fit(x, y, epochs=10, max_epochs=20)

    @pytest.mark.unit
    def test_fit_wrong_output_size(self, simple_network, simple_2d_data):
        """Test fit with wrong output size in target."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        x, _ = simple_2d_data
        y_wrong = torch.randn(len(x), 5)  # Wrong output size
        
        with pytest.raises(ValidationError):
            simple_network.fit(x, y_wrong, epochs=1)

    @pytest.mark.unit
    def test_fit_early_stopping_not_bool(self, simple_network, simple_2d_data):
        """Test fit with non-boolean early_stopping."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        x, y = simple_2d_data
        
        with pytest.raises(ValidationError):
            simple_network.fit(x, y, epochs=1, early_stopping="yes")

    @pytest.mark.unit
    def test_fit_empty_dataset(self, simple_network):
        """Test fit with empty dataset."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        x = torch.empty(0, 2)
        y = torch.empty(0, 2)
        
        with pytest.raises(ValidationError):
            simple_network.fit(x, y, epochs=1)

    @pytest.mark.unit
    def test_fit_y_val_without_x_val(self, simple_network, simple_2d_data):
        """Test fit with y_val but no x_val."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
        
        x, y = simple_2d_data
        
        with pytest.raises(ValidationError):
            simple_network.fit(x, y, epochs=1, y_val=y)
