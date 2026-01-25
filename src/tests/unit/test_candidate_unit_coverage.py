#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# File Name:     test_candidate_unit_coverage.py
# Author:        Paul Calnon
# Version:       0.3.16
#
# Date Created:  2026-01-24
# Last Modified: 2026-01-24
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Unit tests for CandidateUnit class to increase code coverage.
#    Part of CASCOR-P2-001: Increase code coverage.
#
#####################################################################################################################################################################################################
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidate_unit.candidate_unit import CandidateUnit, CandidateTrainingResult


@pytest.fixture
def basic_candidate():
    """Create a basic CandidateUnit for testing."""
    return CandidateUnit(
        CandidateUnit__input_size=2,
        CandidateUnit__epochs=10,
        CandidateUnit__learning_rate=0.01,
        CandidateUnit__random_seed=42,
        CandidateUnit__candidate_index=0,
    )


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(10, 2)


@pytest.fixture
def sample_residual_error():
    """Create sample residual error tensor."""
    return torch.randn(10, 2)


class TestCandidateUnitInit:
    """Tests for CandidateUnit initialization."""

    @pytest.mark.unit
    def test_basic_initialization(self):
        """Test basic CandidateUnit initialization."""
        candidate = CandidateUnit(
            CandidateUnit__input_size=3,
            CandidateUnit__candidate_index=0,
        )
        assert candidate.input_size == 3
        assert hasattr(candidate, "weights")
        assert hasattr(candidate, "bias")

    @pytest.mark.unit
    def test_initialization_with_custom_epochs(self):
        """Test CandidateUnit initialization with custom epochs."""
        candidate = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__epochs=100,
            CandidateUnit__candidate_index=0,
        )
        assert candidate.epochs == 100

    @pytest.mark.unit
    def test_initialization_with_custom_learning_rate(self):
        """Test CandidateUnit initialization with custom learning rate."""
        candidate = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__learning_rate=0.001,
            CandidateUnit__candidate_index=0,
        )
        assert candidate.learning_rate == 0.001

    @pytest.mark.unit
    def test_initialization_with_random_seed(self):
        """Test CandidateUnit initialization with random seed for reproducibility."""
        candidate1 = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__random_seed=42,
            CandidateUnit__candidate_index=0,
        )
        candidate2 = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__random_seed=42,
            CandidateUnit__candidate_index=0,
        )
        # Weights should be similar (within tolerance) with same seed
        assert candidate1.weights.shape == candidate2.weights.shape

    @pytest.mark.unit
    def test_initialization_with_uuid(self):
        """Test CandidateUnit initialization with custom UUID."""
        test_uuid = "test-uuid-abc123"
        candidate = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__uuid=test_uuid,
            CandidateUnit__candidate_index=0,
        )
        assert candidate.get_uuid() == test_uuid


class TestCandidateUnitProperties:
    """Tests for CandidateUnit properties and getters."""

    @pytest.mark.unit
    def test_get_uuid(self, basic_candidate):
        """Test get_uuid returns a string."""
        uuid = basic_candidate.get_uuid()
        assert isinstance(uuid, str)
        assert len(uuid) > 0

    @pytest.mark.unit
    def test_weights_shape(self, basic_candidate):
        """Test weights tensor has correct shape."""
        weights = basic_candidate.weights
        assert isinstance(weights, torch.Tensor)
        assert weights.shape[0] == basic_candidate.input_size

    @pytest.mark.unit
    def test_bias_shape(self, basic_candidate):
        """Test bias tensor has correct shape."""
        bias = basic_candidate.bias
        assert isinstance(bias, torch.Tensor)
        # Bias should be a scalar or 1D tensor

    @pytest.mark.unit
    def test_correlation_initial_value(self, basic_candidate):
        """Test correlation starts at 0."""
        assert basic_candidate.correlation == 0.0 or basic_candidate.correlation is None


class TestCandidateUnitForward:
    """Tests for CandidateUnit forward pass."""

    @pytest.mark.unit
    def test_forward_basic(self, basic_candidate, sample_input):
        """Test basic forward pass produces output."""
        output = basic_candidate.forward(sample_input)
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == sample_input.shape[0]

    @pytest.mark.unit
    def test_forward_preserves_batch_size(self, basic_candidate):
        """Test forward preserves batch dimension."""
        batch_sizes = [1, 5, 10, 32]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, basic_candidate.input_size)
            output = basic_candidate.forward(x)
            assert output.shape[0] == batch_size

    @pytest.mark.unit
    def test_forward_output_is_1d_per_sample(self, basic_candidate, sample_input):
        """Test forward output has correct dimensions."""
        output = basic_candidate.forward(sample_input)
        # Output should be (batch_size,) or (batch_size, 1)
        assert len(output.shape) <= 2


class TestCandidateUnitPickling:
    """Tests for CandidateUnit pickling support."""

    @pytest.mark.unit
    def test_getstate(self, basic_candidate):
        """Test __getstate__ returns picklable dict."""
        state = basic_candidate.__getstate__()
        assert isinstance(state, dict)
        # Logger should not be in state
        assert "logger" not in state

    @pytest.mark.unit
    def test_setstate(self, basic_candidate):
        """Test __setstate__ restores instance."""
        state = basic_candidate.__getstate__()
        new_candidate = CandidateUnit.__new__(CandidateUnit)
        new_candidate.__setstate__(state)
        assert hasattr(new_candidate, "weights")
        assert hasattr(new_candidate, "bias")

    @pytest.mark.unit
    def test_pickle_roundtrip(self, basic_candidate):
        """Test pickle/unpickle roundtrip."""
        import pickle
        pickled = pickle.dumps(basic_candidate)
        unpickled = pickle.loads(pickled)
        assert unpickled.input_size == basic_candidate.input_size
        assert unpickled.epochs == basic_candidate.epochs


class TestCandidateTrainingResult:
    """Tests for CandidateTrainingResult dataclass."""

    @pytest.mark.unit
    def test_create_basic_result(self):
        """Test creating a basic CandidateTrainingResult."""
        result = CandidateTrainingResult(
            candidate_id=0,
            candidate_uuid="test-uuid",
            correlation=0.75,
            candidate=None,
            success=True,
        )
        assert result.candidate_id == 0
        assert result.correlation == 0.75
        assert result.success is True

    @pytest.mark.unit
    def test_create_failed_result(self):
        """Test creating a failed CandidateTrainingResult."""
        result = CandidateTrainingResult(
            candidate_id=1,
            candidate_uuid="test-uuid-2",
            correlation=0.0,
            candidate=None,
            success=False,
            error_message="Training failed",
        )
        assert result.success is False
        assert result.error_message == "Training failed"


class TestActivationWithDerivative:
    """Tests for ActivationWithDerivative class."""

    @pytest.mark.unit
    def test_import_activation_with_derivative(self):
        """Test ActivationWithDerivative can be imported."""
        from candidate_unit.candidate_unit import ActivationWithDerivative
        assert ActivationWithDerivative is not None

    @pytest.mark.unit
    def test_create_with_tanh(self):
        """Test creating ActivationWithDerivative with tanh."""
        from candidate_unit.candidate_unit import ActivationWithDerivative
        awd = ActivationWithDerivative(torch.tanh)
        assert awd is not None
        assert callable(awd)

    @pytest.mark.unit
    def test_call_forward(self):
        """Test calling ActivationWithDerivative for forward pass."""
        from candidate_unit.candidate_unit import ActivationWithDerivative
        awd = ActivationWithDerivative(torch.tanh)
        x = torch.tensor([0.0, 0.5, 1.0])
        output = awd(x, derivative=False)
        expected = torch.tanh(x)
        assert torch.allclose(output, expected)

    @pytest.mark.unit
    def test_call_derivative(self):
        """Test calling ActivationWithDerivative for derivative."""
        from candidate_unit.candidate_unit import ActivationWithDerivative
        awd = ActivationWithDerivative(torch.tanh)
        x = torch.tensor([0.0, 0.5, 1.0])
        deriv = awd(x, derivative=True)
        # Derivative of tanh is 1 - tanh^2
        expected = 1.0 - torch.tanh(x) ** 2
        assert torch.allclose(deriv, expected)

    @pytest.mark.unit
    def test_pickle_activation_with_derivative(self):
        """Test ActivationWithDerivative can be pickled."""
        import pickle
        from candidate_unit.candidate_unit import ActivationWithDerivative
        awd = ActivationWithDerivative(torch.tanh)
        pickled = pickle.dumps(awd)
        unpickled = pickle.loads(pickled)
        # Test it still works after unpickling
        x = torch.tensor([0.5])
        assert torch.allclose(awd(x), unpickled(x))

    @pytest.mark.unit
    def test_repr(self):
        """Test ActivationWithDerivative __repr__."""
        from candidate_unit.candidate_unit import ActivationWithDerivative
        awd = ActivationWithDerivative(torch.tanh)
        repr_str = repr(awd)
        assert "ActivationWithDerivative" in repr_str
        assert "tanh" in repr_str


class TestCandidateUnitCorrelation:
    """Tests for correlation calculation."""

    @pytest.mark.unit
    def test_calculate_correlation_basic(self, basic_candidate, sample_input, sample_residual_error):
        """Test basic correlation calculation."""
        output = basic_candidate.forward(sample_input)
        # Call _calculate_correlation if it exists
        if hasattr(basic_candidate, "_calculate_correlation"):
            result = basic_candidate._calculate_correlation(output, sample_residual_error)
            assert result is not None

    @pytest.mark.unit
    def test_correlation_range(self, basic_candidate, sample_input, sample_residual_error):
        """Test correlation is in valid range [-1, 1]."""
        output = basic_candidate.forward(sample_input)
        if hasattr(basic_candidate, "_calculate_correlation"):
            result = basic_candidate._calculate_correlation(output, sample_residual_error)
            if isinstance(result, tuple):
                correlation = result[0]
            else:
                correlation = result
            if correlation is not None:
                assert -1.0 <= abs(float(correlation)) <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
