#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# File Name:     test_cascor_getters_setters.py
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
#    Unit tests for CascadeCorrelationNetwork getter/setter methods and utility functions.
#    Part of CASCOR-P2-001: Increase code coverage.
#
#####################################################################################################################################################################################################
import os
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig


@pytest.fixture
def basic_network():
    """Create a basic network for testing."""
    config = CascadeCorrelationConfig(
        input_size=2,
        output_size=2,
        random_seed=42,
    )
    return CascadeCorrelationNetwork(config=config)


@pytest.fixture
def mock_candidate_result():
    """Create a mock CandidateTrainingResult for testing."""

    @dataclass
    class MockResult:
        candidate_id: int
        correlation: float
        success: bool
        error_message: str = ""

    return MockResult


class TestCascorGetters:
    """Tests for CascadeCorrelationNetwork getter methods."""

    @pytest.mark.unit
    def test_get_uuid_returns_string(self, basic_network):
        """Test get_uuid returns a valid UUID string."""
        uuid = basic_network.get_uuid()
        assert isinstance(uuid, str)
        assert len(uuid) > 0

    @pytest.mark.unit
    def test_get_activation_fn(self, basic_network):
        """Test get_activation_fn returns the activation function."""
        activation = basic_network.get_activation_fn()
        assert activation is not None
        assert callable(activation)

    @pytest.mark.unit
    def test_get_candidate_training_queue_authkey(self, basic_network):
        """Test getting candidate training queue authkey."""
        authkey = basic_network.get_candidate_training_queue_authkey()
        # May be None, bytes, or string
        assert authkey is None or isinstance(authkey, (bytes, str))

    @pytest.mark.unit
    def test_get_candidate_training_queue_address(self, basic_network):
        """Test getting candidate training queue address."""
        address = basic_network.get_candidate_training_queue_address()
        # May be None, string, or tuple
        assert address is None or isinstance(address, (str, tuple))

    @pytest.mark.unit
    def test_get_candidate_training_tasks_queue_timeout(self, basic_network):
        """Test getting candidate training tasks queue timeout."""
        timeout = basic_network.get_candidate_training_tasks_queue_timeout()
        # May be None or an integer
        assert timeout is None or isinstance(timeout, (int, float))

    @pytest.mark.unit
    def test_get_candidate_training_shutdown_timeout(self, basic_network):
        """Test getting candidate training shutdown timeout."""
        timeout = basic_network.get_candidate_training_shutdown_timeout()
        # May be None or an integer
        assert timeout is None or isinstance(timeout, (int, float))


class TestCascorSetters:
    """Tests for CascadeCorrelationNetwork setter methods."""

    @pytest.mark.unit
    def test_set_activation_fn(self, basic_network):
        """Test set_activation_fn updates the activation function."""
        basic_network.set_activation_fn(torch.relu)
        assert basic_network.activation_fn == torch.relu

    @pytest.mark.unit
    def test_set_activation_fn_no_diff(self, basic_network):
        """Test set_activation_fn_no_diff updates the activation function."""
        basic_network.set_activation_fn_no_diff(torch.sigmoid)
        assert basic_network.activation_fn_no_diff == torch.sigmoid

    @pytest.mark.unit
    def test_set_candidate_training_queue_authkey(self, basic_network):
        """Test setting candidate training queue authkey."""
        authkey = b"test_authkey"
        basic_network.set_candidate_training_queue_authkey(authkey)
        assert basic_network.candidate_training_queue_authkey == authkey

    @pytest.mark.unit
    def test_set_candidate_training_queue_address(self, basic_network):
        """Test setting candidate training queue address."""
        address = "localhost:5000"
        basic_network.set_candidate_training_queue_address(address)
        assert basic_network.candidate_training_queue_address == address

    @pytest.mark.unit
    def test_set_candidate_training_tasks_queue_timeout(self, basic_network):
        """Test setting candidate training tasks queue timeout."""
        timeout = 30
        basic_network.set_candidate_training_tasks_queue_timeout(timeout)
        assert basic_network.candidate_training_tasks_queue_timeout == timeout

    @pytest.mark.unit
    def test_set_candidate_training_shutdown_timeout(self, basic_network):
        """Test setting candidate training shutdown timeout."""
        timeout = 60
        basic_network.set_candidate_training_shutdown_timeout(timeout)
        assert basic_network.candidate_training_shutdown_timeout == timeout


class TestCandidateDataHelpers:
    """Tests for candidate data extraction helper methods."""

    @pytest.mark.unit
    def test_get_candidates_data_empty_list(self, basic_network):
        """Test get_candidates_data with empty results list."""
        results = []
        data = basic_network.get_candidates_data(results, "correlation")
        assert data == []

    @pytest.mark.unit
    def test_get_candidates_data_with_results(self, basic_network, mock_candidate_result):
        """Test get_candidates_data extracts field values."""
        MockResult = mock_candidate_result
        results = [
            MockResult(candidate_id=0, correlation=0.5, success=True),
            MockResult(candidate_id=1, correlation=0.7, success=True),
            MockResult(candidate_id=2, correlation=0.3, success=False),
        ]
        correlations = basic_network.get_candidates_data(results, "correlation")
        assert correlations == [0.5, 0.7, 0.3]

    @pytest.mark.unit
    def test_get_single_candidate_data(self, basic_network, mock_candidate_result):
        """Test get_single_candidate_data retrieves correct candidate."""
        MockResult = mock_candidate_result
        results = [
            MockResult(candidate_id=0, correlation=0.5, success=True),
            MockResult(candidate_id=1, correlation=0.7, success=True),
        ]
        correlation = basic_network.get_single_candidate_data(results, candidate_id=1, field="correlation", default=0.0)
        assert correlation == 0.7

    @pytest.mark.unit
    def test_get_single_candidate_data_not_found(self, basic_network, mock_candidate_result):
        """Test get_single_candidate_data returns default when not found."""
        MockResult = mock_candidate_result
        results = [
            MockResult(candidate_id=0, correlation=0.5, success=True),
        ]
        correlation = basic_network.get_single_candidate_data(results, candidate_id=99, field="correlation", default=-1.0)
        assert correlation == -1.0

    @pytest.mark.unit
    def test_get_candidates_data_count_all_success(self, basic_network, mock_candidate_result):
        """Test get_candidates_data_count counts items matching constraint."""
        MockResult = mock_candidate_result
        results = [
            MockResult(candidate_id=0, correlation=0.5, success=True),
            MockResult(candidate_id=1, correlation=0.7, success=True),
            MockResult(candidate_id=2, correlation=0.3, success=False),
        ]
        count = basic_network.get_candidates_data_count(results, field="success", constraint=lambda x: x is True)
        assert count == 2

    @pytest.mark.unit
    def test_get_candidates_data_count_none_match(self, basic_network, mock_candidate_result):
        """Test get_candidates_data_count returns 0 when no matches."""
        MockResult = mock_candidate_result
        results = [
            MockResult(candidate_id=0, correlation=0.5, success=False),
            MockResult(candidate_id=1, correlation=0.7, success=False),
        ]
        count = basic_network.get_candidates_data_count(results, field="success", constraint=lambda x: x is True)
        assert count == 0

    @pytest.mark.unit
    def test_get_candidates_error_messages(self, basic_network, mock_candidate_result):
        """Test get_candidates_error_messages extracts error messages."""

        @dataclass
        class MockResultWithUuid:
            candidate_id: int
            candidate_uuid: str
            correlation: float
            success: bool
            error_message: str = ""

        results = [
            MockResultWithUuid(candidate_id=0, candidate_uuid="uuid-0", correlation=0.5, success=True),
            MockResultWithUuid(candidate_id=1, candidate_uuid="uuid-1", correlation=0.0, success=False, error_message="Training failed"),
            MockResultWithUuid(candidate_id=2, candidate_uuid="uuid-2", correlation=0.7, success=True),
        ]
        # valid_candidates should be a list of booleans indicating if each result is valid
        valid_candidates = [True, False, True]
        error_messages = basic_network.get_candidates_error_messages(results, valid_candidates)
        assert isinstance(error_messages, dict)


class TestNetworkProperties:
    """Tests for network property accessors."""

    @pytest.mark.unit
    def test_input_size_property(self, basic_network):
        """Test input_size property returns correct value."""
        assert basic_network.input_size == 2

    @pytest.mark.unit
    def test_output_size_property(self, basic_network):
        """Test output_size property returns correct value."""
        assert basic_network.output_size == 2

    @pytest.mark.unit
    def test_hidden_units_starts_empty(self, basic_network):
        """Test hidden_units list starts empty."""
        assert isinstance(basic_network.hidden_units, list)
        assert len(basic_network.hidden_units) == 0

    @pytest.mark.unit
    def test_history_initialized(self, basic_network):
        """Test history dictionary is initialized."""
        assert isinstance(basic_network.history, dict)
        assert "train_loss" in basic_network.history or hasattr(basic_network, "history")

    @pytest.mark.unit
    def test_learning_rate_property(self, basic_network):
        """Test learning_rate property returns a float."""
        lr = basic_network.learning_rate
        assert isinstance(lr, float)
        assert lr > 0

    @pytest.mark.unit
    def test_output_weights_shape(self, basic_network):
        """Test output_weights has correct shape."""
        weights = basic_network.output_weights
        assert isinstance(weights, torch.Tensor)
        assert weights.shape[0] == basic_network.output_size

    @pytest.mark.unit
    def test_output_bias_shape(self, basic_network):
        """Test output_bias has correct shape."""
        bias = basic_network.output_bias
        assert isinstance(bias, torch.Tensor)
        assert bias.shape[0] == basic_network.output_size


class TestCreateCandidateUnit:
    """Tests for _create_candidate_unit factory method."""

    @pytest.mark.unit
    def test_create_candidate_unit_basic(self, basic_network):
        """Test _create_candidate_unit creates a valid CandidateUnit."""
        candidate = basic_network._create_candidate_unit(candidate_index=0)
        assert candidate is not None
        assert hasattr(candidate, "get_uuid")

    @pytest.mark.unit
    def test_create_candidate_unit_with_uuid(self, basic_network):
        """Test _create_candidate_unit respects provided UUID."""
        test_uuid = "test-uuid-12345"
        candidate = basic_network._create_candidate_unit(
            candidate_index=0,
            candidate_uuid=test_uuid,
        )
        assert candidate.get_uuid() == test_uuid

    @pytest.mark.unit
    def test_create_candidate_unit_with_custom_input_size(self, basic_network):
        """Test _create_candidate_unit respects custom input_size."""
        candidate = basic_network._create_candidate_unit(
            candidate_index=0,
            input_size=5,
        )
        assert candidate.input_size == 5


class TestSelectBestCandidates:
    """Tests for _select_best_candidates method."""

    @pytest.mark.unit
    def test_select_best_candidates_empty(self, basic_network):
        """Test _select_best_candidates with empty list."""
        selected = basic_network._select_best_candidates([], num_candidates=1)
        assert selected == []

    @pytest.mark.unit
    def test_select_best_candidates_single(self, basic_network, mock_candidate_result):
        """Test _select_best_candidates selects highest correlation."""
        MockResult = mock_candidate_result
        results = [
            MockResult(candidate_id=0, correlation=0.5, success=True),
            MockResult(candidate_id=1, correlation=0.9, success=True),
            MockResult(candidate_id=2, correlation=0.3, success=True),
        ]
        selected = basic_network._select_best_candidates(results, num_candidates=1)
        assert len(selected) == 1
        assert selected[0].correlation == 0.9

    @pytest.mark.unit
    def test_select_best_candidates_multiple(self, basic_network, mock_candidate_result):
        """Test _select_best_candidates selects top N."""
        MockResult = mock_candidate_result
        results = [
            MockResult(candidate_id=0, correlation=0.5, success=True),
            MockResult(candidate_id=1, correlation=0.9, success=True),
            MockResult(candidate_id=2, correlation=0.7, success=True),
        ]
        selected = basic_network._select_best_candidates(results, num_candidates=2)
        assert len(selected) == 2
        # Should be sorted by correlation descending
        assert selected[0].correlation == 0.9
        assert selected[1].correlation == 0.7

    @pytest.mark.unit
    def test_select_best_candidates_negative_correlation(self, basic_network, mock_candidate_result):
        """Test _select_best_candidates handles negative correlations (uses abs)."""
        MockResult = mock_candidate_result
        results = [
            MockResult(candidate_id=0, correlation=-0.9, success=True),
            MockResult(candidate_id=1, correlation=0.5, success=True),
        ]
        selected = basic_network._select_best_candidates(results, num_candidates=1)
        assert len(selected) == 1
        # -0.9 has higher absolute value
        assert abs(selected[0].correlation) == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
