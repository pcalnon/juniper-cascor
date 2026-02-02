#!/usr/bin/env python
"""
Unit tests for SpiralDataProvider.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spiral_problem.data_provider import SpiralDataProvider, SpiralDataProviderError


@pytest.mark.unit
class TestSpiralDataProviderUseJuniperData:
    """Tests for use_juniper_data property."""

    def test_use_juniper_data_true_when_url_provided(self):
        """use_juniper_data should be True when URL is provided."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")
        assert provider.use_juniper_data is True

    def test_use_juniper_data_false_when_url_not_provided(self):
        """use_juniper_data should be False when URL is not provided."""
        with patch.dict(os.environ, {}, clear=True):
            if "JUNIPER_DATA_URL" in os.environ:
                del os.environ["JUNIPER_DATA_URL"]
            provider = SpiralDataProvider(juniper_data_url=None)
            assert provider.use_juniper_data is False

    def test_use_juniper_data_true_from_env_var(self):
        """use_juniper_data should be True when JUNIPER_DATA_URL env var is set."""
        with patch.dict(os.environ, {"JUNIPER_DATA_URL": "http://api.example.com:8100"}):
            provider = SpiralDataProvider()
            assert provider.use_juniper_data is True

    def test_use_juniper_data_false_when_env_var_not_set(self):
        """use_juniper_data should be False when no URL configured."""
        env_without_url = {k: v for k, v in os.environ.items() if k != "JUNIPER_DATA_URL"}
        with patch.dict(os.environ, env_without_url, clear=True):
            provider = SpiralDataProvider()
            assert provider.use_juniper_data is False

    def test_use_juniper_data_false_for_empty_string(self):
        """use_juniper_data should be False for empty string URL."""
        provider = SpiralDataProvider(juniper_data_url="")
        assert provider.use_juniper_data is False


@pytest.mark.unit
class TestSpiralDataProviderGetSpiralDataset:
    """Tests for get_spiral_dataset method with mocked JuniperDataClient."""

    def test_get_spiral_dataset_calls_client_with_correct_params(self):
        """get_spiral_dataset should map parameters correctly to client."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")

        mock_arrays = {
            "X_train": np.array([[1.0, 2.0]]),
            "y_train": np.array([[1.0, 0.0]]),
            "X_test": np.array([[3.0, 4.0]]),
            "y_test": np.array([[0.0, 1.0]]),
            "X_full": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "y_full": np.array([[1.0, 0.0], [0.0, 1.0]]),
        }

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.return_value = {"dataset_id": "test-dataset-id"}
            mock_client.download_artifact_npz.return_value = mock_arrays
            MockClient.return_value = mock_client

            provider.get_spiral_dataset(
                n_spirals=3,
                n_points=100,
                n_rotations=1.5,
                noise_level=0.1,
                clockwise=True,
                train_ratio=0.7,
                test_ratio=0.2,
                seed=42,
            )

            mock_client.create_dataset.assert_called_once()
            call_kwargs = mock_client.create_dataset.call_args[1]
            assert call_kwargs["generator"] == "spiral"
            params = call_kwargs["params"]
            assert params["n_spirals"] == 3
            assert params["n_points_per_spiral"] == 100
            assert params["n_rotations"] == 1.5
            assert params["noise"] == 0.1
            assert params["clockwise"] is True
            assert params["train_ratio"] == 0.7
            assert params["test_ratio"] == 0.2
            assert params["seed"] == 42

    def test_get_spiral_dataset_omits_seed_when_none(self):
        """get_spiral_dataset should not include seed param when None."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")

        mock_arrays = {
            "X_train": np.array([[1.0, 2.0]]),
            "y_train": np.array([[1.0, 0.0]]),
            "X_test": np.array([[3.0, 4.0]]),
            "y_test": np.array([[0.0, 1.0]]),
            "X_full": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "y_full": np.array([[1.0, 0.0], [0.0, 1.0]]),
        }

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.return_value = {"dataset_id": "test-id"}
            mock_client.download_artifact_npz.return_value = mock_arrays
            MockClient.return_value = mock_client

            provider.get_spiral_dataset(
                n_spirals=2,
                n_points=50,
                n_rotations=1.0,
                noise_level=0.05,
                clockwise=False,
                train_ratio=0.8,
                test_ratio=0.1,
                seed=None,
            )

            params = mock_client.create_dataset.call_args[1]["params"]
            assert "seed" not in params

    def test_get_spiral_dataset_downloads_artifact_with_dataset_id(self):
        """get_spiral_dataset should download artifact using returned dataset_id."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")

        mock_arrays = {
            "X_train": np.array([[1.0, 2.0]]),
            "y_train": np.array([[1.0, 0.0]]),
            "X_test": np.array([[3.0, 4.0]]),
            "y_test": np.array([[0.0, 1.0]]),
            "X_full": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "y_full": np.array([[1.0, 0.0], [0.0, 1.0]]),
        }

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.return_value = {"dataset_id": "specific-dataset-123"}
            mock_client.download_artifact_npz.return_value = mock_arrays
            MockClient.return_value = mock_client

            provider.get_spiral_dataset(
                n_spirals=2,
                n_points=50,
                n_rotations=1.0,
                noise_level=0.05,
                clockwise=False,
                train_ratio=0.8,
                test_ratio=0.1,
            )

            mock_client.download_artifact_npz.assert_called_once_with("specific-dataset-123")

    def test_get_spiral_dataset_passes_algorithm_parameter(self):
        """get_spiral_dataset should pass algorithm parameter to client."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")

        mock_arrays = {
            "X_train": np.array([[1.0, 2.0]]),
            "y_train": np.array([[1.0, 0.0]]),
            "X_test": np.array([[3.0, 4.0]]),
            "y_test": np.array([[0.0, 1.0]]),
            "X_full": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "y_full": np.array([[1.0, 0.0], [0.0, 1.0]]),
        }

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.return_value = {"dataset_id": "test-id"}
            mock_client.download_artifact_npz.return_value = mock_arrays
            MockClient.return_value = mock_client

            provider.get_spiral_dataset(
                n_spirals=2,
                n_points=50,
                n_rotations=1.0,
                noise_level=0.05,
                clockwise=False,
                train_ratio=0.8,
                test_ratio=0.1,
                algorithm="legacy_cascor",
            )

            mock_client.create_dataset.assert_called_once()
            call_kwargs = mock_client.create_dataset.call_args[1]
            params = call_kwargs["params"]
            assert params["algorithm"] == "legacy_cascor"


@pytest.mark.unit
class TestSpiralDataProviderTensorConversion:
    """Tests for numpy to torch tensor conversion."""

    def test_converts_arrays_to_float32_tensors(self):
        """Should convert numpy arrays to float32 torch tensors."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")

        mock_arrays = {
            "X_train": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            "y_train": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            "X_test": np.array([[5.0, 6.0]], dtype=np.float64),
            "y_test": np.array([[1.0, 0.0]], dtype=np.float64),
            "X_full": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64),
            "y_full": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        }

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.return_value = {"dataset_id": "test-id"}
            mock_client.download_artifact_npz.return_value = mock_arrays
            MockClient.return_value = mock_client

            result = provider.get_spiral_dataset(
                n_spirals=2,
                n_points=50,
                n_rotations=1.0,
                noise_level=0.05,
                clockwise=False,
                train_ratio=0.8,
                test_ratio=0.1,
            )

            (x_train, y_train), (x_test, y_test), (x_full, y_full) = result

            assert x_train.dtype == torch.float32
            assert y_train.dtype == torch.float32
            assert x_test.dtype == torch.float32
            assert y_test.dtype == torch.float32
            assert x_full.dtype == torch.float32
            assert y_full.dtype == torch.float32

    def test_preserves_array_shape(self):
        """Should preserve array shapes during conversion."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")

        mock_arrays = {
            "X_train": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "y_train": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            "X_test": np.array([[7.0, 8.0]]),
            "y_test": np.array([[0.0, 1.0]]),
            "X_full": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            "y_full": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        }

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.return_value = {"dataset_id": "test-id"}
            mock_client.download_artifact_npz.return_value = mock_arrays
            MockClient.return_value = mock_client

            result = provider.get_spiral_dataset(
                n_spirals=2,
                n_points=50,
                n_rotations=1.0,
                noise_level=0.05,
                clockwise=False,
                train_ratio=0.8,
                test_ratio=0.1,
            )

            (x_train, y_train), (x_test, y_test), (x_full, y_full) = result

            assert x_train.shape == (3, 2)
            assert y_train.shape == (3, 2)
            assert x_test.shape == (1, 2)
            assert y_test.shape == (1, 2)
            assert x_full.shape == (4, 2)
            assert y_full.shape == (4, 2)

    def test_preserves_array_values(self):
        """Should preserve array values during conversion."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")

        expected_x_train = np.array([[1.5, 2.5], [3.5, 4.5]])
        expected_y_train = np.array([[1.0, 0.0], [0.0, 1.0]])

        mock_arrays = {
            "X_train": expected_x_train,
            "y_train": expected_y_train,
            "X_test": np.array([[5.0, 6.0]]),
            "y_test": np.array([[1.0, 0.0]]),
            "X_full": np.array([[1.5, 2.5], [3.5, 4.5], [5.0, 6.0]]),
            "y_full": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        }

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.return_value = {"dataset_id": "test-id"}
            mock_client.download_artifact_npz.return_value = mock_arrays
            MockClient.return_value = mock_client

            result = provider.get_spiral_dataset(
                n_spirals=2,
                n_points=50,
                n_rotations=1.0,
                noise_level=0.05,
                clockwise=False,
                train_ratio=0.8,
                test_ratio=0.1,
            )

            (x_train, y_train), _, _ = result

            np.testing.assert_array_almost_equal(x_train.numpy(), expected_x_train)
            np.testing.assert_array_almost_equal(y_train.numpy(), expected_y_train)


@pytest.mark.unit
class TestSpiralDataProviderErrorHandling:
    """Tests for error handling in SpiralDataProvider."""

    def test_raises_error_when_url_not_configured(self):
        """Should raise SpiralDataProviderError when URL not configured."""
        env_without_url = {k: v for k, v in os.environ.items() if k != "JUNIPER_DATA_URL"}
        with patch.dict(os.environ, env_without_url, clear=True):
            provider = SpiralDataProvider()

            with pytest.raises(SpiralDataProviderError) as exc_info:
                provider.get_spiral_dataset(
                    n_spirals=2,
                    n_points=50,
                    n_rotations=1.0,
                    noise_level=0.05,
                    clockwise=False,
                    train_ratio=0.8,
                    test_ratio=0.1,
                )

            assert "not configured" in str(exc_info.value)

    def test_wraps_client_exception_in_provider_error(self):
        """Should wrap client exceptions in SpiralDataProviderError."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.side_effect = Exception("Connection refused")
            MockClient.return_value = mock_client

            with pytest.raises(SpiralDataProviderError) as exc_info:
                provider.get_spiral_dataset(
                    n_spirals=2,
                    n_points=50,
                    n_rotations=1.0,
                    noise_level=0.05,
                    clockwise=False,
                    train_ratio=0.8,
                    test_ratio=0.1,
                )

            assert "Failed to fetch spiral dataset" in str(exc_info.value)
            assert "Connection refused" in str(exc_info.value)

    def test_wraps_download_exception_in_provider_error(self):
        """Should wrap download exceptions in SpiralDataProviderError."""
        provider = SpiralDataProvider(juniper_data_url="http://localhost:8100")

        with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
            mock_client = MagicMock()
            mock_client.create_dataset.return_value = {"dataset_id": "test-id"}
            mock_client.download_artifact_npz.side_effect = Exception("Download failed")
            MockClient.return_value = mock_client

            with pytest.raises(SpiralDataProviderError) as exc_info:
                provider.get_spiral_dataset(
                    n_spirals=2,
                    n_points=50,
                    n_rotations=1.0,
                    noise_level=0.05,
                    clockwise=False,
                    train_ratio=0.8,
                    test_ratio=0.1,
                )

            assert "Failed to fetch spiral dataset" in str(exc_info.value)
            assert "Download failed" in str(exc_info.value)

    def test_get_client_raises_error_when_url_not_set(self):
        """_get_client should raise SpiralDataProviderError when URL not configured."""
        env_without_url = {k: v for k, v in os.environ.items() if k != "JUNIPER_DATA_URL"}
        with patch.dict(os.environ, env_without_url, clear=True):
            provider = SpiralDataProvider()

            with pytest.raises(SpiralDataProviderError) as exc_info:
                provider._get_client()

            assert "not configured" in str(exc_info.value)


@pytest.mark.unit
class TestSpiralDataProviderEnvironmentVariable:
    """Tests for environment variable handling."""

    def test_env_var_takes_precedence_over_none(self):
        """JUNIPER_DATA_URL env var should be used when no URL provided."""
        with patch.dict(os.environ, {"JUNIPER_DATA_URL": "http://env-url:8100"}):
            provider = SpiralDataProvider()
            assert provider._juniper_data_url == "http://env-url:8100"

    def test_explicit_url_takes_precedence_over_env_var(self):
        """Explicit URL should take precedence over env var."""
        with patch.dict(os.environ, {"JUNIPER_DATA_URL": "http://env-url:8100"}):
            provider = SpiralDataProvider(juniper_data_url="http://explicit-url:8100")
            assert provider._juniper_data_url == "http://explicit-url:8100"

    def test_client_created_with_correct_url(self):
        """Client should be created with the configured URL."""
        with patch.dict(os.environ, {"JUNIPER_DATA_URL": "http://test-server:8100"}):
            with patch("spiral_problem.data_provider.JuniperDataClient") as MockClient:
                mock_client = MagicMock()
                MockClient.return_value = mock_client

                provider = SpiralDataProvider()
                provider._get_client()

                MockClient.assert_called_once_with(base_url="http://test-server:8100")
