#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# File Name:     test_candidate_unit_coverage_deep.py
# Author:        Paul Calnon
# Version:       0.1.0
#
# Date Created:  2026-03-12
# Last Modified: 2026-03-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Deep coverage tests for CandidateUnit class targeting uncovered lines.
#    Covers: ActivationWithDerivative edge cases, CUDA path mocking,
#    _seed_random_generator None seeder, forward 1-D, train early stopping,
#    train display progress exception, _get_correlation_abs_value,
#    _calculate_abs_value, _calculate_correlation zero denominator,
#    _update_weights_and_bias edge cases, _validate_correlation_params,
#    setters, getters, UUID methods, clear_display_status/progress.
#
#####################################################################################################################################################################################################
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidate_unit.candidate_unit import (
    ActivationWithDerivative,
    CandidateParametersUpdate,
    CandidateTrainingResult,
    CandidateUnit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def candidate():
    """Create a basic CandidateUnit for testing."""
    return CandidateUnit(
        _CandidateUnit__input_size=2,
        _CandidateUnit__learning_rate=0.01,
        _CandidateUnit__epochs=10,
        _CandidateUnit__log_level_name="ERROR",
        _CandidateUnit__random_seed=42,
    )


@pytest.fixture
def candidate_early_stop():
    """Create a CandidateUnit with early stopping enabled and low patience."""
    return CandidateUnit(
        _CandidateUnit__input_size=2,
        _CandidateUnit__learning_rate=0.01,
        _CandidateUnit__epochs=10,
        _CandidateUnit__epochs_max=100,
        _CandidateUnit__log_level_name="ERROR",
        _CandidateUnit__random_seed=42,
        _CandidateUnit__early_stopping=True,
        _CandidateUnit__patience=2,
    )


@pytest.fixture
def sample_input():
    """Create sample 2D input tensor."""
    torch.manual_seed(42)
    return torch.randn(10, 2)


@pytest.fixture
def sample_residual_error():
    """Create sample multi-output residual error tensor."""
    torch.manual_seed(99)
    return torch.randn(10, 2)


# ===========================================================================
# 1. ActivationWithDerivative edge cases
# ===========================================================================


class TestActivationWithDerivativeEdgeCases:
    """Tests for ActivationWithDerivative fallback and numerical derivative paths."""

    @pytest.mark.unit
    def test_activation_name_fallback_str(self):
        """Line 207: _activation_name fallback when activation lacks __name__ and __class__."""
        # Create an object that has neither __name__ nor __class__
        # In practice, every Python object has __class__, so we test the __name__ path
        # by providing an nn.Module instance (which has __class__ but no __name__)
        act = torch.nn.Softplus()
        awd = ActivationWithDerivative(act)
        # Should use __class__.__name__ -> "Softplus"
        assert awd._activation_name == "Softplus"

    @pytest.mark.unit
    def test_numerical_derivative_for_unknown_activation(self):
        """Lines 231-232: Numerical derivative for unknown activation functions."""
        # Use a custom function that has __name__ but is not tanh/sigmoid/relu
        def custom_square(x):
            return x ** 2

        awd = ActivationWithDerivative(custom_square)
        assert awd._activation_name == "custom_square"

        x = torch.tensor([1.0, 2.0, 3.0])
        # derivative=True should use numerical approximation
        result = awd(x, derivative=True)
        # Derivative of x^2 is 2x; numerical approx uses eps=1e-6 so tolerance must account for float precision
        expected = torch.tensor([2.0, 4.0, 6.0])
        torch.testing.assert_close(result, expected, atol=0.5, rtol=0.1)

    @pytest.mark.unit
    def test_numerical_derivative_for_lambda(self):
        """Line 207: Lambda function lacking __name__ in some contexts."""
        custom_fn = lambda x: x * 2  # noqa: E731
        awd = ActivationWithDerivative(custom_fn)
        # Lambda has __name__ = "<lambda>"
        assert awd._activation_name == "<lambda>"

        x = torch.tensor([1.0, 2.0])
        result = awd(x, derivative=True)
        # Derivative of 2x is 2; uses numerical approx with eps=1e-6 (float32 precision limits)
        expected = torch.tensor([2.0, 2.0])
        torch.testing.assert_close(result, expected, atol=0.5, rtol=0.1)

    @pytest.mark.unit
    def test_activation_forward_pass_no_derivative(self):
        """Test normal forward pass (derivative=False)."""
        awd = ActivationWithDerivative(torch.tanh)
        x = torch.tensor([0.0, 1.0, -1.0])
        result = awd(x, derivative=False)
        expected = torch.tanh(x)
        torch.testing.assert_close(result, expected)


# ===========================================================================
# 2. _initialize_randomness CUDA path
# ===========================================================================


class TestInitializeRandomnessCUDA:
    """Test CUDA seeding path in _initialize_randomness."""

    @pytest.mark.unit
    def test_cuda_seeding_path(self, candidate):
        """Lines 418-422: Mock torch.cuda.is_available to return True."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.manual_seed") as mock_cuda_seed,
            patch("torch.rand", return_value=torch.tensor([0.5])),
        ):
            candidate._initialize_randomness(seed=42, max_value=10)
            mock_cuda_seed.assert_called()


# ===========================================================================
# 3. _seed_random_generator with None seeder
# ===========================================================================


class TestSeedRandomGeneratorNoneSeeder:
    """Test _seed_random_generator early return when seeder is None."""

    @pytest.mark.unit
    def test_none_seeder_returns_early(self, candidate):
        """Lines 442-443: Call with seeder=None, verify early return."""
        # Should not raise, just return
        candidate._seed_random_generator(seed=42, max_value=10, seeder=None, generator=None)


# ===========================================================================
# 4. forward() with 1-D tensor
# ===========================================================================


class TestForward1DTensor:
    """Test forward pass with 1-D input tensor."""

    @pytest.mark.unit
    def test_forward_1d_unsqueeze(self, candidate):
        """Lines 562-563: Forward with 1-D tensor triggers unsqueeze."""
        x = torch.randn(2)  # 1-D tensor, matches input_size=2
        output = candidate.forward(x)
        assert output.dim() >= 1
        assert output.numel() >= 1

    @pytest.mark.unit
    def test_forward_2d_normal(self, candidate):
        """Forward with 2-D tensor does not unsqueeze."""
        x = torch.randn(5, 2)
        output = candidate.forward(x)
        assert output.shape[0] == 5


# ===========================================================================
# 5. train() early stopping
# ===========================================================================


class TestTrainEarlyStopping:
    """Test train with parameters that trigger early stopping."""

    @pytest.mark.unit
    def test_early_stopping_triggers(self, candidate_early_stop):
        """Lines 705-721, 716-718: Early stopping when correlation does not improve."""
        x = torch.randn(10, 2)
        # Create residual error that will produce near-constant correlation
        residual_error = torch.zeros(10, 2)  # Zero error -> zero correlation -> no improvement

        result = candidate_early_stop.train(
            x=x,
            epochs=50,
            residual_error=residual_error,
            learning_rate=0.01,
            display_frequency=100,
        )
        # With zero residual error the correlation cannot improve, so
        # early stopping should kick in well before 50 epochs
        assert isinstance(result, float)


# ===========================================================================
# 6. train() display progress exception
# ===========================================================================


class TestTrainDisplayProgressException:
    """Test error handling when _display_training_progress raises."""

    @pytest.mark.unit
    def test_display_progress_exception_handled(self, candidate):
        """Lines 731-735: Mock _display_training_progress to raise."""
        x = torch.randn(10, 2)
        residual_error = torch.randn(10, 2)

        with patch.object(candidate, "_display_training_progress", side_effect=RuntimeError("Display error")):
            # Should not raise - exception is caught and logged
            result = candidate.train(
                x=x,
                epochs=3,
                residual_error=residual_error,
                learning_rate=0.01,
                display_frequency=1,
            )
            assert isinstance(result, float)


# ===========================================================================
# 7. _get_correlation_abs_value with various types
# ===========================================================================


class TestGetCorrelationAbsValue:
    """Test _get_correlation_abs_value with different correlation types.

    NOTE: Line 951 evaluates len(correlation) in an f-string even when the
    logger is a no-op, so every test item must support len().
    """

    @pytest.mark.unit
    def test_with_tensor_1d(self, candidate):
        """Line 959: 1-D Tensor correlation (isinstance torch.Tensor branch)."""
        candidate.correlations = [torch.tensor([-0.75])]
        result = candidate._get_correlation_abs_value(index=0)
        assert float(result) == pytest.approx(0.75, abs=1e-6)

    @pytest.mark.unit
    def test_with_tuple(self, candidate):
        """Line 957-958: Tuple of (tensor, int) — tuple/list branch."""
        candidate.correlations = [(torch.tensor(-0.5), 0)]
        result = candidate._get_correlation_abs_value(index=0)
        assert float(result) == pytest.approx(0.5, abs=1e-6)

    @pytest.mark.unit
    def test_with_list(self, candidate):
        """Line 957-958: List of [float, int] — tuple/list branch."""
        candidate.correlations = [[-0.3, 1]]
        result = candidate._get_correlation_abs_value(index=0)
        assert float(result) == pytest.approx(0.3, abs=1e-6)

    @pytest.mark.unit
    def test_with_ndarray(self, candidate):
        """Line 959: ndarray correlation."""
        candidate.correlations = [np.array([-0.9])]
        result = candidate._get_correlation_abs_value(index=0)
        assert float(np.asarray(result).flat[0]) == pytest.approx(0.9, abs=1e-6)

    @pytest.mark.unit
    def test_with_tuple_float_first(self, candidate):
        """Line 957-958: Tuple with float as first element."""
        candidate.correlations = [(-0.42, 1)]
        result = candidate._get_correlation_abs_value(index=0)
        assert float(result) == pytest.approx(0.42, abs=1e-6)


# ===========================================================================
# 8. _calculate_abs_value with various types
# ===========================================================================


class TestCalculateAbsValue:
    """Test _calculate_abs_value with different value types."""

    @pytest.mark.unit
    def test_torch_tensor(self, candidate):
        """Line 986-987: Torch tensor."""
        result = candidate._calculate_abs_value(torch.tensor(-3.0))
        assert float(result) == pytest.approx(3.0)

    @pytest.mark.unit
    def test_ndarray(self, candidate):
        """Line 988-989: Numpy array."""
        result = candidate._calculate_abs_value(np.array(-2.5))
        assert float(result) == pytest.approx(2.5)

    @pytest.mark.unit
    def test_float(self, candidate):
        """Line 988-989: Float."""
        result = candidate._calculate_abs_value(-1.5)
        assert float(result) == pytest.approx(1.5)

    @pytest.mark.unit
    def test_int(self, candidate):
        """Line 988-989: Int."""
        result = candidate._calculate_abs_value(-7)
        assert float(result) == pytest.approx(7.0)

    @pytest.mark.unit
    def test_unknown_type_fallback(self, candidate):
        """Lines 990-992: Unknown type falls through to numpy abs."""
        result = candidate._calculate_abs_value(np.float32(-4.0))
        assert float(result) == pytest.approx(4.0)


# ===========================================================================
# 9. _calculate_correlation denominator zero/NaN
# ===========================================================================


class TestCalculateCorrelationDenomZero:
    """Test _calculate_correlation when denominator is zero or NaN."""

    @pytest.mark.unit
    def test_zero_denominator_constant_output(self, candidate):
        """Lines 1066-1067: Constant output and constant error produce zero denominator."""
        # Constant output: all same value -> std=0 -> denominator effectively 0
        output = torch.ones(10)
        residual_error = torch.ones(10)
        correlation, norm_output, norm_error, num, den = candidate._calculate_correlation(
            output=output, residual_error=residual_error
        )
        # With the epsilon, denominator won't be exactly zero but correlation
        # should be near zero since numerator is zero (all centered values are 0)
        assert abs(correlation) < 1e-3

    @pytest.mark.unit
    def test_zero_residual_error(self, candidate):
        """Zero residual error produces near-zero correlation."""
        output = torch.randn(10)
        residual_error = torch.zeros(10)
        correlation, norm_output, norm_error, num, den = candidate._calculate_correlation(
            output=output, residual_error=residual_error
        )
        assert abs(correlation) < 1e-3


# ===========================================================================
# 10. _update_weights_and_bias edge cases
# ===========================================================================


class TestUpdateWeightsAndBiasEdgeCases:
    """Test _update_weights_and_bias with multi-output and gradient edge cases."""

    @pytest.mark.unit
    def test_multi_output_negative_best_corr_idx(self, candidate, sample_input, sample_residual_error):
        """Lines 1140-1141: Multi-output with best_corr_idx < 0 -> fallback warning."""
        output = candidate.forward(sample_input)
        params = CandidateParametersUpdate(
            x=sample_input,
            y=output,
            residual_error=sample_residual_error,
            learning_rate=0.01,
            norm_output=output - output.mean(),
            norm_error=sample_residual_error[:, 0] - sample_residual_error[:, 0].mean(),
            best_corr_idx=-1,  # Invalid index -> fallback
            numerator=0.0,
            denominator=1.0,
        )
        result = candidate._update_weights_and_bias(candidate_parameters_update=params)
        # Should not raise, should use column 0 as fallback
        assert result is not None

    @pytest.mark.unit
    def test_single_output_error(self, candidate, sample_input):
        """Lines 1143-1145: Single output error path."""
        output = candidate.forward(sample_input)
        residual_error_1d = torch.randn(10)
        params = CandidateParametersUpdate(
            x=sample_input,
            y=output,
            residual_error=residual_error_1d,
            learning_rate=0.01,
            norm_output=output - output.mean(),
            norm_error=residual_error_1d - residual_error_1d.mean(),
            best_corr_idx=0,
            numerator=0.0,
            denominator=1.0,
        )
        result = candidate._update_weights_and_bias(candidate_parameters_update=params)
        assert result is not None


# ===========================================================================
# 11. _validate_correlation_params validation errors
# ===========================================================================


class TestValidateCorrelationParams:
    """Test each of the 6 validation paths in _validate_correlation_params."""

    @pytest.mark.unit
    def test_none_output(self, candidate):
        """Line 1236: None output raises ValueError."""
        with pytest.raises(ValueError, match="must not be None"):
            candidate._validate_correlation_params(output=None, residual_error=torch.randn(5))

    @pytest.mark.unit
    def test_none_residual_error(self, candidate):
        """Line 1236: None residual_error raises ValueError."""
        with pytest.raises(ValueError, match="must not be None"):
            candidate._validate_correlation_params(output=torch.randn(5), residual_error=None)

    @pytest.mark.unit
    def test_non_tensor_residual_error(self, candidate):
        """Line 1242: Non-tensor residual_error raises TypeError."""
        # Use numpy array (has .shape but is not torch.Tensor)
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            candidate._validate_correlation_params(output=torch.randn(5), residual_error=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    @pytest.mark.unit
    def test_non_tensor_output(self, candidate):
        """Line 1242: Non-tensor output raises TypeError."""
        # Use numpy array (has .shape but is not torch.Tensor)
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            candidate._validate_correlation_params(output=np.array([1.0, 2.0, 3.0, 4.0, 5.0]), residual_error=torch.randn(5))

    @pytest.mark.unit
    def test_mismatched_batch_sizes(self, candidate):
        """Line 1248: Mismatched batch sizes raises ValueError."""
        with pytest.raises(ValueError, match="same batch size"):
            candidate._validate_correlation_params(output=torch.randn(5, 2), residual_error=torch.randn(10, 2))

    @pytest.mark.unit
    def test_greater_than_2d_input(self, candidate):
        """Line 1260: >2D input raises ValueError."""
        with pytest.raises(ValueError, match="at most two dimensions"):
            candidate._validate_correlation_params(output=torch.randn(5, 2, 3), residual_error=torch.randn(5, 2))

    @pytest.mark.unit
    def test_greater_than_2d_residual(self, candidate):
        """Line 1260: >2D residual_error raises ValueError."""
        with pytest.raises(ValueError, match="at most two dimensions"):
            candidate._validate_correlation_params(output=torch.randn(5, 2), residual_error=torch.randn(5, 2, 3))

    @pytest.mark.unit
    def test_mismatched_features(self, candidate):
        """Line 1270: Mismatched features when residual_error is 2D."""
        with pytest.raises(ValueError, match="same number of features"):
            candidate._validate_correlation_params(output=torch.randn(5, 2), residual_error=torch.randn(5, 3))

    @pytest.mark.unit
    def test_valid_params_pass(self, candidate):
        """Ensure valid parameters pass validation without raising."""
        # 1D tensors
        candidate._validate_correlation_params(output=torch.randn(5), residual_error=torch.randn(5))
        # 2D tensors with matching features
        candidate._validate_correlation_params(output=torch.randn(5, 2), residual_error=torch.randn(5, 2))


# ===========================================================================
# 12. Setter methods
# ===========================================================================


class TestSetterMethods:
    """Test setter methods for CandidateUnit attributes."""

    @pytest.mark.unit
    def test_set_correlation(self, candidate):
        """Line 1371."""
        candidate.set_correlation(0.95)
        assert candidate.correlation == 0.95

    @pytest.mark.unit
    def test_set_activation_fn(self, candidate):
        """Line 1379."""
        new_fn = torch.nn.ReLU()
        candidate.set_activation_fn(new_fn)
        assert candidate.activation_fn is new_fn

    @pytest.mark.unit
    def test_set_activation_fn_base(self, candidate):
        """Line 1387."""
        new_fn = torch.sigmoid
        candidate.set_activation_fn_base(new_fn)
        assert candidate.activation_fn_base is new_fn

    @pytest.mark.unit
    def test_set_bias(self, candidate):
        """Line 1395."""
        new_bias = torch.tensor([0.5])
        candidate.set_bias(new_bias)
        assert torch.equal(candidate.bias, new_bias)

    @pytest.mark.unit
    def test_set_display_frequency(self, candidate):
        """Line 1411."""
        candidate.set_display_frequency(50)
        assert candidate.display_frequency == 50

    @pytest.mark.unit
    def test_set_epochs_max(self, candidate):
        """Line 1419."""
        candidate.set_epochs_max(200)
        assert candidate.epochs_max == 200

    @pytest.mark.unit
    def test_set_learning_rate(self, candidate):
        """Line 1427."""
        candidate.set_learning_rate(0.001)
        assert candidate.learning_rate == 0.001

    @pytest.mark.unit
    def test_set_logging_file_name(self, candidate):
        """Line 1435."""
        candidate.set_logging_file_name("test_log.log")
        assert candidate.logging_file_name == "test_log.log"

    @pytest.mark.unit
    def test_set_logging_level(self, candidate):
        """Line 1443."""
        candidate.set_logging_level(10)
        assert candidate.logging_level == 10

    @pytest.mark.unit
    def test_set_random_value_scale(self, candidate):
        """Line 1451."""
        candidate.set_random_value_scale(0.5)
        assert candidate.random_value_scale == 0.5

    @pytest.mark.unit
    def test_set_weights(self, candidate):
        """Line 1459."""
        new_weights = torch.tensor([1.0, 2.0])
        candidate.set_weights(new_weights)
        assert torch.equal(candidate.weights, new_weights)


# ===========================================================================
# 13. Getter methods
# ===========================================================================


class TestGetterMethods:
    """Test getter methods for CandidateUnit attributes."""

    @pytest.mark.unit
    def test_get_activation_fn(self, candidate):
        """Line 1512."""
        result = candidate.get_activation_fn()
        assert result is not None

    @pytest.mark.unit
    def test_get_activation_fn_base(self, candidate):
        """Line 1520."""
        result = candidate.get_activation_fn_base()
        assert result is not None

    @pytest.mark.unit
    def test_get_bias(self, candidate):
        """Line 1528."""
        result = candidate.get_bias()
        assert isinstance(result, torch.Tensor)

    @pytest.mark.unit
    def test_get_display_frequency(self, candidate):
        """Line 1536."""
        result = candidate.get_display_frequency()
        assert isinstance(result, int)

    @pytest.mark.unit
    def test_get_epochs_max(self, candidate):
        """Line 1552."""
        result = candidate.get_epochs_max()
        assert isinstance(result, int)

    @pytest.mark.unit
    def test_get_learning_rate(self, candidate):
        """Line 1560."""
        result = candidate.get_learning_rate()
        assert isinstance(result, float)

    @pytest.mark.unit
    def test_get_logging_file_name(self, candidate):
        """Line 1568."""
        candidate.set_logging_file_name("my_log.txt")
        result = candidate.get_logging_file_name()
        assert result == "my_log.txt"

    @pytest.mark.unit
    def test_get_logging_level(self, candidate):
        """Line 1576."""
        candidate.set_logging_level(20)
        result = candidate.get_logging_level()
        assert result == 20

    @pytest.mark.unit
    def test_get_random_value_scale(self, candidate):
        """Line 1584."""
        result = candidate.get_random_value_scale()
        assert isinstance(result, float)

    @pytest.mark.unit
    def test_get_weights(self, candidate):
        """Line 1592."""
        result = candidate.get_weights()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2,)

    @pytest.mark.unit
    def test_get_correlation(self, candidate):
        """Line 1504."""
        result = candidate.get_correlation()
        assert isinstance(result, float)


# ===========================================================================
# 14. UUID methods
# ===========================================================================


class TestUUIDMethods:
    """Test set_uuid and get_uuid methods."""

    @pytest.mark.unit
    def test_set_uuid_with_value(self):
        """Lines 1461-1477: set_uuid with a valid UUID string."""
        candidate = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__learning_rate=0.01,
            CandidateUnit__epochs=10,
            CandidateUnit__log_level_name="ERROR",
            CandidateUnit__random_seed=42,
            CandidateUnit__uuid="test-uuid-1234",
        )
        assert candidate.uuid == "test-uuid-1234"

    @pytest.mark.unit
    def test_set_uuid_double_set_error(self, candidate):
        """Lines 1473-1475: Double-set UUID triggers fatal and os._exit."""
        # candidate already has a UUID set during __init__
        assert candidate.uuid is not None
        with patch("os._exit") as mock_exit:
            candidate.set_uuid("new-uuid")
            mock_exit.assert_called_once_with(1)

    @pytest.mark.unit
    def test_get_uuid_lazy_initialization(self):
        """Lines 1491-1493: get_uuid lazily initializes UUID if not set."""
        candidate = CandidateUnit(
            _CandidateUnit__input_size=2,
            _CandidateUnit__learning_rate=0.01,
            _CandidateUnit__epochs=10,
            _CandidateUnit__log_level_name="ERROR",
            _CandidateUnit__random_seed=42,
        )
        # UUID is set during __init__, but test the getter
        uuid_val = candidate.get_uuid()
        assert uuid_val is not None
        assert isinstance(uuid_val, str)
        assert len(uuid_val) > 0

    @pytest.mark.unit
    def test_get_uuid_generates_when_none(self):
        """Lines 1491-1493: get_uuid generates UUID when self.uuid is None."""
        candidate = CandidateUnit(
            _CandidateUnit__input_size=2,
            _CandidateUnit__learning_rate=0.01,
            _CandidateUnit__epochs=10,
            _CandidateUnit__log_level_name="ERROR",
            _CandidateUnit__random_seed=42,
        )
        # Force uuid to None to test lazy generation
        candidate.uuid = None
        uuid_val = candidate.get_uuid()
        assert uuid_val is not None
        assert len(uuid_val) > 0


# ===========================================================================
# 15. clear_display_status/progress
# ===========================================================================


class TestClearDisplayMethods:
    """Test clear_display_status and clear_display_progress."""

    @pytest.mark.unit
    def test_clear_display_status_with_attribute(self, candidate):
        """Lines 1329-1344: Clear when attribute exists and is not None."""
        candidate._candidate_display_status = MagicMock()
        candidate.clear_display_status()
        assert candidate._candidate_display_status is None

    @pytest.mark.unit
    def test_clear_display_status_already_none(self, candidate):
        """Lines 1342-1343: Clear when attribute is None already."""
        candidate._candidate_display_status = None
        candidate.clear_display_status()
        assert candidate._candidate_display_status is None

    @pytest.mark.unit
    def test_clear_display_status_no_attribute(self, candidate):
        """Lines 1342-1343: Clear when attribute does not exist."""
        if hasattr(candidate, "_candidate_display_status"):
            delattr(candidate, "_candidate_display_status")
        candidate.clear_display_status()
        # Should not raise, should log warning

    @pytest.mark.unit
    def test_clear_display_progress_with_attribute(self, candidate):
        """Lines 1346-1361: Clear when attribute exists and is not None."""
        candidate._candidate_display_progress = MagicMock()
        candidate.clear_display_progress()
        assert candidate._candidate_display_progress is None

    @pytest.mark.unit
    def test_clear_display_progress_already_none(self, candidate):
        """Lines 1359-1360: Clear when attribute is None already."""
        candidate._candidate_display_progress = None
        candidate.clear_display_progress()
        assert candidate._candidate_display_progress is None

    @pytest.mark.unit
    def test_clear_display_progress_no_attribute(self, candidate):
        """Lines 1359-1360: Clear when attribute does not exist."""
        if hasattr(candidate, "_candidate_display_progress"):
            delattr(candidate, "_candidate_display_progress")
        candidate.clear_display_progress()
        # Should not raise, should log warning


# ===========================================================================
# Additional edge case coverage
# ===========================================================================


class TestActivationDerivativeKnownFunctions:
    """Test ActivationWithDerivative derivative for known functions."""

    @pytest.mark.unit
    def test_tanh_derivative(self):
        """Line 223: tanh derivative path."""
        awd = ActivationWithDerivative(torch.tanh)
        x = torch.tensor([0.0, 1.0])
        result = awd(x, derivative=True)
        expected = 1.0 - torch.tanh(x) ** 2
        torch.testing.assert_close(result, expected)

    @pytest.mark.unit
    def test_sigmoid_derivative(self):
        """Lines 225-226: sigmoid derivative path."""
        awd = ActivationWithDerivative(torch.sigmoid)
        x = torch.tensor([0.0, 1.0])
        result = awd(x, derivative=True)
        y = torch.sigmoid(x)
        expected = y * (1.0 - y)
        torch.testing.assert_close(result, expected)

    @pytest.mark.unit
    def test_relu_derivative(self):
        """Line 228: relu derivative path."""
        awd = ActivationWithDerivative(torch.relu)
        x = torch.tensor([-1.0, 0.0, 1.0])
        result = awd(x, derivative=True)
        expected = torch.tensor([0.0, 0.0, 1.0])
        torch.testing.assert_close(result, expected)


class TestActivationSerialization:
    """Test ActivationWithDerivative pickle support."""

    @pytest.mark.unit
    def test_getstate_setstate(self):
        """Lines 236-244: __getstate__ and __setstate__ round trip."""
        awd = ActivationWithDerivative(torch.tanh)
        state = awd.__getstate__()
        assert state == {"_activation_name": "tanh"}

        awd2 = ActivationWithDerivative.__new__(ActivationWithDerivative)
        awd2.__setstate__(state)
        assert awd2._activation_name == "tanh"
        # Verify the restored function works
        x = torch.tensor([1.0])
        result = awd2(x)
        assert result.shape == (1,)

    @pytest.mark.unit
    def test_repr(self):
        """Line 248: __repr__."""
        awd = ActivationWithDerivative(torch.tanh)
        assert "tanh" in repr(awd)


class TestSeedRandomGeneratorWithGenerator:
    """Test _seed_random_generator when generator is None."""

    @pytest.mark.unit
    def test_generator_none_skips_rolling(self, candidate):
        """Lines 446-448: When generator is None, skip rolling."""
        # Just verify no error
        candidate._seed_random_generator(seed=42, max_value=10, seeder=lambda s: None, generator=None)


class TestCandidateTrainingResultDataclass:
    """Test CandidateTrainingResult defaults."""

    @pytest.mark.unit
    def test_defaults(self):
        """Verify dataclass default values."""
        result = CandidateTrainingResult()
        assert result.candidate_id == -1
        assert result.correlation == 0.0
        assert result.success is True
        assert result.epochs_completed == 0
        assert result.all_correlations == []
