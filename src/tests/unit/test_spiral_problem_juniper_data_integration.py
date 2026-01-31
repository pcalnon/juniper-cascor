#!/usr/bin/env python
"""
Unit tests for JUNIPER_DATA_URL feature flag integration in SpiralProblem.generate_n_spiral_dataset().
"""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spiral_problem.spiral_problem import SpiralProblem


@pytest.mark.unit
@pytest.mark.spiral
class TestSpiralProblemJuniperDataIntegration:
    """Tests for JUNIPER_DATA_URL feature flag integration in generate_n_spiral_dataset."""

    def test_uses_legacy_path_when_env_not_set(self):
        """Verify that when JUNIPER_DATA_URL is NOT set, the legacy code path is used (provider is not called)."""
        env_without_url = {k: v for k, v in os.environ.items() if k != "JUNIPER_DATA_URL"}
        with patch.dict(os.environ, env_without_url, clear=True):
            with patch("spiral_problem.data_provider.SpiralDataProvider") as MockProvider:
                sp = SpiralProblem(
                    _SpiralProblem__n_points=20,
                    _SpiralProblem__n_spirals=2,
                )
                result = sp.generate_n_spiral_dataset()

                MockProvider.assert_not_called()

                assert isinstance(result, tuple)
                assert len(result) == 3
                (x_train, y_train), (x_test, y_test), (x_full, y_full) = result
                assert isinstance(x_train, torch.Tensor)
                assert isinstance(y_train, torch.Tensor)

    def test_uses_juniper_data_when_env_set(self):
        """Verify that when JUNIPER_DATA_URL IS set, SpiralDataProvider is called."""
        mock_arrays = {
            "X_train": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "y_train": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "X_test": np.array([[5.0, 6.0]], dtype=np.float32),
            "y_test": np.array([[1.0, 0.0]], dtype=np.float32),
            "X_full": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            "y_full": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        }

        mock_result = (
            (torch.from_numpy(mock_arrays["X_train"]), torch.from_numpy(mock_arrays["y_train"])),
            (torch.from_numpy(mock_arrays["X_test"]), torch.from_numpy(mock_arrays["y_test"])),
            (torch.from_numpy(mock_arrays["X_full"]), torch.from_numpy(mock_arrays["y_full"])),
        )

        with patch.dict(os.environ, {"JUNIPER_DATA_URL": "http://localhost:8100"}):
            with patch("spiral_problem.data_provider.SpiralDataProvider") as MockProvider:
                mock_provider_instance = MagicMock()
                mock_provider_instance.get_spiral_dataset.return_value = mock_result
                MockProvider.return_value = mock_provider_instance

                sp = SpiralProblem(
                    _SpiralProblem__n_points=20,
                    _SpiralProblem__n_spirals=2,
                )
                result = sp.generate_n_spiral_dataset()

                MockProvider.assert_called_once_with("http://localhost:8100")
                mock_provider_instance.get_spiral_dataset.assert_called_once()

    def test_returns_correct_format_from_juniper_data(self):
        """Verify the return format matches ((x_train, y_train), (x_test, y_test), (x_full, y_full))."""
        mock_x_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        mock_y_train = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        mock_x_test = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
        mock_y_test = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        mock_x_full = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
        mock_y_full = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

        mock_result = (
            (mock_x_train, mock_y_train),
            (mock_x_test, mock_y_test),
            (mock_x_full, mock_y_full),
        )

        with patch.dict(os.environ, {"JUNIPER_DATA_URL": "http://localhost:8100"}):
            with patch("spiral_problem.data_provider.SpiralDataProvider") as MockProvider:
                mock_provider_instance = MagicMock()
                mock_provider_instance.get_spiral_dataset.return_value = mock_result
                MockProvider.return_value = mock_provider_instance

                sp = SpiralProblem(
                    _SpiralProblem__n_points=20,
                    _SpiralProblem__n_spirals=2,
                )
                result = sp.generate_n_spiral_dataset()

                assert isinstance(result, tuple)
                assert len(result) == 3

                (x_train, y_train), (x_test, y_test), (x_full, y_full) = result

                assert isinstance(x_train, torch.Tensor)
                assert isinstance(y_train, torch.Tensor)
                assert isinstance(x_test, torch.Tensor)
                assert isinstance(y_test, torch.Tensor)
                assert isinstance(x_full, torch.Tensor)
                assert isinstance(y_full, torch.Tensor)

                assert x_train.shape == (2, 2)
                assert y_train.shape == (2, 2)
                assert x_test.shape == (1, 2)
                assert y_test.shape == (1, 2)
                assert x_full.shape == (3, 2)
                assert y_full.shape == (3, 2)

                torch.testing.assert_close(x_train, mock_x_train)
                torch.testing.assert_close(y_train, mock_y_train)
