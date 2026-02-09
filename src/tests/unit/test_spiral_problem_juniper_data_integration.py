#!/usr/bin/env python
"""
Unit tests for JuniperData mandatory integration in SpiralProblem.generate_n_spiral_dataset().

Since CAS-INT-001, JUNIPER_DATA_URL is REQUIRED. There is no local generation fallback.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ConfigurationError
from spiral_problem.spiral_problem import SpiralProblem


def _make_mock_result(n_train=2, n_test=1, n_spirals=2):
    """Create mock tensor result matching SpiralDataProvider return format."""
    n_full = n_train + n_test
    return (
        (torch.randn(n_train, 2), torch.zeros(n_train, n_spirals)),
        (torch.randn(n_test, 2), torch.zeros(n_test, n_spirals)),
        (torch.randn(n_full, 2), torch.zeros(n_full, n_spirals)),
    )


@pytest.mark.unit
@pytest.mark.spiral
class TestMandatoryJuniperDataIntegration:
    """Tests for mandatory JuniperData integration (CAS-INT-001)."""

    def test_raises_configuration_error_when_url_not_set(self):
        """Must raise ConfigurationError when JUNIPER_DATA_URL is not set."""
        env_without_url = {k: v for k, v in os.environ.items() if k != "JUNIPER_DATA_URL"}
        with patch.dict(os.environ, env_without_url, clear=True):
            sp = SpiralProblem(
                _SpiralProblem__n_points=20,
                _SpiralProblem__n_spirals=2,
            )
            with pytest.raises(ConfigurationError) as exc_info:
                sp.generate_n_spiral_dataset()

            assert "JUNIPER_DATA_URL" in str(exc_info.value)

    def test_error_message_includes_guidance(self):
        """Error message should include configuration guidance."""
        env_without_url = {k: v for k, v in os.environ.items() if k != "JUNIPER_DATA_URL"}
        with patch.dict(os.environ, env_without_url, clear=True):
            sp = SpiralProblem(
                _SpiralProblem__n_points=20,
                _SpiralProblem__n_spirals=2,
            )
            with pytest.raises(ConfigurationError) as exc_info:
                sp.generate_n_spiral_dataset()

            error_msg = str(exc_info.value)
            assert "http://localhost:8100" in error_msg
            assert "AGENTS.md" in error_msg

    def test_always_uses_spiral_data_provider(self):
        """Must always use SpiralDataProvider when JUNIPER_DATA_URL is set."""
        mock_result = _make_mock_result()

        with patch.dict(os.environ, {"JUNIPER_DATA_URL": "http://localhost:8100"}):
            with patch("spiral_problem.data_provider.SpiralDataProvider") as MockProvider:
                mock_instance = MagicMock()
                mock_instance.get_spiral_dataset.return_value = mock_result
                MockProvider.return_value = mock_instance

                sp = SpiralProblem(
                    _SpiralProblem__n_points=20,
                    _SpiralProblem__n_spirals=2,
                )
                result = sp.generate_n_spiral_dataset()
                assert result
                assert isinstance(result, tuple)

                MockProvider.assert_called_once_with("http://localhost:8100")
                mock_instance.get_spiral_dataset.assert_called_once()

    def test_passes_correct_params_to_provider(self):
        """Must pass all generation parameters to SpiralDataProvider."""
        mock_result = _make_mock_result(n_train=14, n_test=6, n_spirals=2)

        with patch.dict(os.environ, {"JUNIPER_DATA_URL": "http://test-server:8100"}):
            with patch("spiral_problem.data_provider.SpiralDataProvider") as MockProvider:
                mock_instance = MagicMock()
                mock_instance.get_spiral_dataset.return_value = mock_result
                MockProvider.return_value = mock_instance

                sp = SpiralProblem(
                    _SpiralProblem__n_points=20,
                    _SpiralProblem__n_spirals=2,
                    _SpiralProblem__random_seed=42,
                )
                sp.generate_n_spiral_dataset(
                    n_spirals=2,
                    n_points=20,
                    n_rotations=1.5,
                    noise_level=0.1,
                    clockwise=True,
                    train_ratio=0.7,
                    test_ratio=0.3,
                )

                call_kwargs = mock_instance.get_spiral_dataset.call_args[1]
                assert call_kwargs["n_spirals"] == 2
                assert call_kwargs["n_points"] == 20
                assert call_kwargs["n_rotations"] == 1.5
                assert call_kwargs["noise_level"] == 0.1
                assert call_kwargs["clockwise"] is True
                assert call_kwargs["train_ratio"] == 0.7
                assert call_kwargs["test_ratio"] == 0.3
                assert call_kwargs["seed"] == 42

    def test_returns_correct_tuple_format(self):
        """Must return ((x_train, y_train), (x_test, y_test), (x_full, y_full))."""
        mock_result = _make_mock_result()

        with patch.dict(os.environ, {"JUNIPER_DATA_URL": "http://localhost:8100"}):
            with patch("spiral_problem.data_provider.SpiralDataProvider") as MockProvider:
                mock_instance = MagicMock()
                mock_instance.get_spiral_dataset.return_value = mock_result
                MockProvider.return_value = mock_instance

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

    def test_provider_error_propagates(self):
        """SpiralDataProviderError should propagate to caller."""
        from spiral_problem.data_provider import SpiralDataProviderError

        with patch.dict(os.environ, {"JUNIPER_DATA_URL": "http://localhost:8100"}):
            with patch("spiral_problem.data_provider.SpiralDataProvider") as MockProvider:
                mock_instance = MagicMock()
                mock_instance.get_spiral_dataset.side_effect = SpiralDataProviderError("Service unavailable")
                MockProvider.return_value = mock_instance

                sp = SpiralProblem(
                    _SpiralProblem__n_points=20,
                    _SpiralProblem__n_spirals=2,
                )
                with pytest.raises(SpiralDataProviderError):
                    sp.generate_n_spiral_dataset()
