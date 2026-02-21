#!/usr/bin/env python
"""
Unit tests for CascadeCorrelationPlotter to improve code coverage.

Covers:
- __getstate__ / __setstate__ pickling cycle
- plot_dataset input validation (non-tensor, wrong shape)
- plot_decision_boundary validation (None inputs, non-tensor, wrong features)
- plot_training_history validation (non-dict, empty history)
- _plot_headings validation (invalid title, xlabel, ylabel types)
"""

import os
import pickle
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
from cascor_plotter.cascor_plotter import CascadeCorrelationPlotter

pytestmark = pytest.mark.unit


class TestPlotterPickling:
    """Tests for __getstate__ / __setstate__."""

    def test_pickle_roundtrip(self):
        """Plotter should be picklable and restorable."""
        plotter = CascadeCorrelationPlotter()
        data = pickle.dumps(plotter)
        restored = pickle.loads(data)
        assert isinstance(restored, CascadeCorrelationPlotter)
        assert hasattr(restored, "logger")

    def test_getstate_excludes_logger(self):
        """__getstate__ should exclude logger."""
        plotter = CascadeCorrelationPlotter()
        state = plotter.__getstate__()
        assert "logger" not in state

    def test_setstate_restores_logger(self):
        """__setstate__ should restore logger."""
        plotter = CascadeCorrelationPlotter()
        state = plotter.__getstate__()
        new_plotter = CascadeCorrelationPlotter.__new__(CascadeCorrelationPlotter)
        new_plotter.__setstate__(state)
        assert hasattr(new_plotter, "logger")


class TestPlotDatasetValidation:
    """Tests for plot_dataset input validation."""

    def test_non_tensor_input_raises_validation_error(self):
        """plot_dataset should raise ValidationError for non-tensor inputs."""
        with pytest.raises(ValidationError, match="torch.Tensor"):
            CascadeCorrelationPlotter.plot_dataset(
                x=[[1, 2], [3, 4]],
                y=torch.tensor([[1, 0], [0, 1]]),
            )

    def test_non_tensor_target_raises_validation_error(self):
        """plot_dataset should raise ValidationError for non-tensor target."""
        with pytest.raises(ValidationError, match="torch.Tensor"):
            CascadeCorrelationPlotter.plot_dataset(
                x=torch.tensor([[1.0, 2.0]]),
                y=[[1, 0]],
            )

    def test_wrong_feature_count_raises_validation_error(self):
        """plot_dataset should raise ValidationError for non-2D features."""
        with pytest.raises(ValidationError, match="2 features"):
            CascadeCorrelationPlotter.plot_dataset(
                x=torch.tensor([[1.0, 2.0, 3.0]]),
                y=torch.tensor([[1, 0]]),
            )


class TestPlotDecisionBoundaryValidation:
    """Tests for plot_decision_boundary input validation."""

    def test_plots_disabled_returns_early(self):
        """plot_decision_boundary should return early when plots are disabled."""
        plotter = CascadeCorrelationPlotter()
        mock_network = MagicMock()
        mock_network.get_generate_plots.return_value = False

        # Should return early without error
        plotter.plot_decision_boundary(network=mock_network, x=None, y=None)

    def test_none_inputs_raise_validation_error(self):
        """plot_decision_boundary should raise ValidationError for None inputs."""
        plotter = CascadeCorrelationPlotter()
        mock_network = MagicMock()
        mock_network.get_generate_plots.return_value = True

        with pytest.raises(ValidationError, match="must be provided"):
            plotter.plot_decision_boundary(network=mock_network, x=None, y=None)

    def test_non_tensor_inputs_raise_validation_error(self):
        """plot_decision_boundary should raise ValidationError for non-tensor inputs."""
        plotter = CascadeCorrelationPlotter()
        mock_network = MagicMock()
        mock_network.get_generate_plots.return_value = True

        with pytest.raises(ValidationError, match="torch.Tensor"):
            plotter.plot_decision_boundary(network=mock_network, x=[[1, 2]], y=[[1, 0]])

    def test_wrong_features_raise_validation_error(self):
        """plot_decision_boundary should raise ValidationError for non-2D features."""
        plotter = CascadeCorrelationPlotter()
        mock_network = MagicMock()
        mock_network.get_generate_plots.return_value = True

        with pytest.raises(ValidationError, match="2 features"):
            plotter.plot_decision_boundary(
                network=mock_network,
                x=torch.tensor([[1.0, 2.0, 3.0]]),
                y=torch.tensor([[1, 0]]),
            )


class TestPlotTrainingHistoryValidation:
    """Tests for plot_training_history input validation."""

    def test_non_dict_raises_validation_error(self):
        """plot_training_history should raise ValidationError for non-dict input."""
        plotter = CascadeCorrelationPlotter()
        with pytest.raises(ValidationError, match="dictionary"):
            plotter.plot_training_history(history="not a dict")

    def test_empty_history_raises_validation_error(self):
        """plot_training_history should raise ValidationError for empty loss."""
        plotter = CascadeCorrelationPlotter()
        with pytest.raises(ValidationError, match="empty"):
            plotter.plot_training_history(history={"train_loss": []})

    def test_missing_train_loss_raises_validation_error(self):
        """plot_training_history should raise ValidationError for missing train_loss."""
        plotter = CascadeCorrelationPlotter()
        with pytest.raises(ValidationError, match="empty"):
            plotter.plot_training_history(history={"train_accuracy": [0.5]})


class TestPlotHeadingsValidation:
    """Tests for _plot_headings input validation."""

    def test_invalid_title_type_raises_validation_error(self):
        """_plot_headings should raise ValidationError for non-string title."""
        plotter = CascadeCorrelationPlotter()
        with pytest.raises(ValidationError, match="title"):
            plotter._plot_headings(title=123)

    def test_invalid_xlabel_type_raises_validation_error(self):
        """_plot_headings should raise ValidationError for non-string x_label."""
        plotter = CascadeCorrelationPlotter()
        with pytest.raises(ValidationError, match="x_label"):
            plotter._plot_headings(x_label=123)

    def test_invalid_ylabel_type_raises_validation_error(self):
        """_plot_headings should raise ValidationError for non-string y_label."""
        plotter = CascadeCorrelationPlotter()
        with pytest.raises(ValidationError, match="y_label"):
            plotter._plot_headings(y_label=123)

    def test_invalid_plot_object_raises_validation_error(self):
        """_plot_headings should raise ValidationError for invalid plot object."""
        plotter = CascadeCorrelationPlotter()
        with pytest.raises(ValidationError, match="matplotlib"):
            plotter._plot_headings(plot="not_a_plot")

    def test_plot_without_required_methods_raises_validation_error(self):
        """_plot_headings should raise ValidationError for plot without title/xlabel/ylabel."""
        plotter = CascadeCorrelationPlotter()
        mock_plot = MagicMock(spec=[])  # No title/xlabel/ylabel methods
        with pytest.raises(ValidationError):
            plotter._plot_headings(plot=mock_plot)
