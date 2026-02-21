#!/usr/bin/env python
"""
Additional unit tests for SpiralDataProvider to improve code coverage.

Covers:
- validate_configuration: successful health check, JuniperDataConnectionError path
- _get_client: client caching behavior, api_key forwarding
- _convert_arrays_to_tensors: direct invocation for validation paths
- _build_spiral_dataset: client interaction flow
- get_spiral_dataset: end-to-end with mocked client

Note: The juniper_data_client package may not be fully importable in all environments
(e.g., when a local shadow directory exists). This module patches sys.modules
to ensure reliable test execution.
"""

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock juniper_data_client if it can't be imported properly
_jdc_mock_needed = False
try:
    from juniper_data_client import JuniperDataClient, JuniperDataConnectionError
except (ImportError, AttributeError):
    _jdc_mock_needed = True
    # Create mock module with the required classes
    _mock_jdc = ModuleType("juniper_data_client")
    JuniperDataConnectionError = type("JuniperDataConnectionError", (ConnectionError,), {})
    JuniperDataClient = MagicMock
    _mock_jdc.JuniperDataClient = JuniperDataClient
    _mock_jdc.JuniperDataConnectionError = JuniperDataConnectionError
    sys.modules["juniper_data_client"] = _mock_jdc

from spiral_problem.data_provider import SpiralDataProvider, SpiralDataProviderError

pytestmark = pytest.mark.unit


def _make_valid_arrays():
    """Helper to create a valid set of NPZ arrays."""
    return {
        "X_train": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "y_train": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "X_test": np.array([[5.0, 6.0]], dtype=np.float32),
        "y_test": np.array([[1.0, 0.0]], dtype=np.float32),
        "X_full": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "y_full": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    }


def _default_dataset_kwargs():
    """Default kwargs for get_spiral_dataset calls."""
    return {
        "n_spirals": 2,
        "n_points": 50,
        "n_rotations": 1.0,
        "noise_level": 0.05,
        "clockwise": False,
        "train_ratio": 0.8,
        "test_ratio": 0.1,
    }


class TestValidateConfigurationPaths:
    """Tests for validate_configuration health check paths."""

    def test_validate_configuration_healthy_service(self):
        """validate_configuration should succeed when health check passes."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.health_check.return_value = True
            MockClient.return_value = mock_client

            provider.validate_configuration()

            mock_client.health_check.assert_called_once()

    def test_validate_configuration_logs_warning_on_connection_error(self):
        """validate_configuration should not raise when service is unreachable."""
        from spiral_problem.data_provider import JuniperDataConnectionError as JDCError

        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.health_check.side_effect = JDCError("unreachable")
            MockClient.return_value = mock_client

            # Should not raise, just log warning
            provider.validate_configuration()

    def test_validate_configuration_raises_when_url_missing(self):
        """validate_configuration should raise SpiralDataProviderError when URL is not set."""
        env_without_url = {k: v for k, v in os.environ.items() if k != "JUNIPER_DATA_URL"}
        with patch.dict(os.environ, env_without_url, clear=True):
            provider = SpiralDataProvider()

            with pytest.raises(SpiralDataProviderError, match="not configured"):
                provider.validate_configuration()


class TestGetClientBehavior:
    """Tests for _get_client caching and initialization."""

    def test_get_client_creates_client_once(self):
        """_get_client should create a JuniperDataClient on first call and cache it."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            client1 = provider._get_client()
            client2 = provider._get_client()

            MockClient.assert_called_once()
            assert client1 is client2

    def test_get_client_passes_api_key(self):
        """_get_client should pass api_key to JuniperDataClient."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100", api_key="test-key-123")

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            MockClient.return_value = MagicMock()

            provider._get_client()

            MockClient.assert_called_once_with(base_url="http://localhost:8100", api_key="test-key-123")

    def test_get_client_raises_when_no_url(self):
        """_get_client should raise SpiralDataProviderError when URL not configured."""
        env_without_url = {k: v for k, v in os.environ.items() if k != "JUNIPER_DATA_URL"}
        with patch.dict(os.environ, env_without_url, clear=True):
            provider = SpiralDataProvider()

            with pytest.raises(SpiralDataProviderError, match="not configured"):
                provider._get_client()


class TestBuildSpiralDataset:
    """Tests for _build_spiral_dataset method."""

    def test_build_spiral_dataset_creates_and_downloads(self):
        """_build_spiral_dataset should create dataset and download artifact."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")
        valid_arrays = _make_valid_arrays()

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.return_value = {"dataset_id": "ds-001"}
            mock_client.download_artifact_npz.return_value = valid_arrays
            MockClient.return_value = mock_client

            params = {"n_spirals": 2, "n_points_per_spiral": 50}
            result = provider._build_spiral_dataset(params)

            mock_client.create_dataset.assert_called_once_with(generator="spiral", params=params)
            mock_client.download_artifact_npz.assert_called_once_with("ds-001")
            assert len(result) == 3


class TestConvertArraysToTensorsDirect:
    """Tests for _convert_arrays_to_tensors called directly."""

    def test_valid_arrays_produce_correct_tensors(self):
        """_convert_arrays_to_tensors should return properly structured tensor tuples."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")
        arrays = _make_valid_arrays()

        (x_train, y_train), (x_test, y_test), (x_full, y_full) = provider._convert_arrays_to_tensors(arrays)

        assert isinstance(x_train, torch.Tensor)
        assert x_train.dtype == torch.float32
        assert x_train.shape == (2, 2)
        assert y_train.shape == (2, 2)
        assert x_test.shape == (1, 2)
        assert y_test.shape == (1, 2)
        assert x_full.shape == (3, 2)
        assert y_full.shape == (3, 2)

    def test_missing_keys_raises_error(self):
        """_convert_arrays_to_tensors should raise when required keys are missing."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")
        arrays = {
            "X_train": np.array([[1.0, 2.0]]),
            "y_train": np.array([[1.0, 0.0]]),
        }

        with pytest.raises(SpiralDataProviderError, match="missing required keys"):
            provider._convert_arrays_to_tensors(arrays)

    def test_wrong_dimensions_raises_error(self):
        """_convert_arrays_to_tensors should raise when arrays are not 2D."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")
        arrays = _make_valid_arrays()
        arrays["X_train"] = np.array([1.0, 2.0])  # 1D

        with pytest.raises(SpiralDataProviderError, match="dimensions"):
            provider._convert_arrays_to_tensors(arrays)

    def test_wrong_feature_columns_raises_error(self):
        """_convert_arrays_to_tensors should raise when feature arrays don't have 2 columns."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")
        arrays = _make_valid_arrays()
        arrays["X_train"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 3 cols
        arrays["X_test"] = np.array([[1.0, 2.0, 3.0]])
        arrays["X_full"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        with pytest.raises(SpiralDataProviderError, match="columns"):
            provider._convert_arrays_to_tensors(arrays)

    def test_y_target_dimension_check(self):
        """_convert_arrays_to_tensors should raise if y_train is not 2D."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")
        arrays = _make_valid_arrays()
        arrays["y_train"] = np.array([1.0, 0.0])  # 1D

        with pytest.raises(SpiralDataProviderError, match="dimensions"):
            provider._convert_arrays_to_tensors(arrays)


class TestGetSpiralDatasetIntegration:
    """Tests for get_spiral_dataset end-to-end with mocked client."""

    def test_successful_dataset_retrieval(self):
        """get_spiral_dataset should return tensor tuples on success."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")
        valid_arrays = _make_valid_arrays()

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.return_value = {"dataset_id": "test-id"}
            mock_client.download_artifact_npz.return_value = valid_arrays
            MockClient.return_value = mock_client

            result = provider.get_spiral_dataset(**_default_dataset_kwargs())

            (x_train, y_train), (x_test, y_test), (x_full, y_full) = result
            assert isinstance(x_train, torch.Tensor)
            assert len(result) == 3

    def test_client_exception_wrapped_in_provider_error(self):
        """get_spiral_dataset should wrap client exceptions in SpiralDataProviderError."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.side_effect = RuntimeError("network error")
            MockClient.return_value = mock_client

            with pytest.raises(SpiralDataProviderError, match="Failed to fetch"):
                provider.get_spiral_dataset(**_default_dataset_kwargs())

    def test_seed_included_when_provided(self):
        """get_spiral_dataset should include seed in params when provided."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")
        valid_arrays = _make_valid_arrays()

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.return_value = {"dataset_id": "test-id"}
            mock_client.download_artifact_npz.return_value = valid_arrays
            MockClient.return_value = mock_client

            provider.get_spiral_dataset(**_default_dataset_kwargs(), seed=42)

            params = mock_client.create_dataset.call_args[1]["params"]
            assert params["seed"] == 42

    def test_algorithm_included_when_provided(self):
        """get_spiral_dataset should include algorithm in params when provided."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")
        valid_arrays = _make_valid_arrays()

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.return_value = {"dataset_id": "test-id"}
            mock_client.download_artifact_npz.return_value = valid_arrays
            MockClient.return_value = mock_client

            provider.get_spiral_dataset(**_default_dataset_kwargs(), algorithm="legacy_cascor")

            params = mock_client.create_dataset.call_args[1]["params"]
            assert params["algorithm"] == "legacy_cascor"

    def test_raises_when_url_not_configured(self):
        """get_spiral_dataset should raise immediately when URL not configured."""
        env_without_url = {k: v for k, v in os.environ.items() if k != "JUNIPER_DATA_URL"}
        with patch.dict(os.environ, env_without_url, clear=True):
            provider = SpiralDataProvider()

            with pytest.raises(SpiralDataProviderError, match="not configured"):
                provider.get_spiral_dataset(**_default_dataset_kwargs())
