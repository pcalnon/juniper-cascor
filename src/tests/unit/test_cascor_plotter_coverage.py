#!/usr/bin/env python
"""
Tests for cascor_plotter/cascor_plotter.py to increase code coverage.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ValidationError
from cascor_plotter.cascor_plotter import CascadeCorrelationPlotter


class TestCascadeCorrelationPlotterInit:
    """Tests for CascadeCorrelationPlotter initialization."""

    def test_init_default_logger(self):
        """Test initialization with default logger."""
        plotter = CascadeCorrelationPlotter()
        assert plotter.logger is not None

    def test_init_custom_logger(self):
        """Test initialization with custom logger."""
        mock_logger = MagicMock()
        plotter = CascadeCorrelationPlotter(logger=mock_logger)
        assert plotter.logger is mock_logger


class TestCascadeCorrelationPlotterPickle:
    """Tests for pickle support (__getstate__ and __setstate__)."""

    def test_getstate_removes_logger(self):
        """Test that __getstate__ removes logger."""
        plotter = CascadeCorrelationPlotter()
        state = plotter.__getstate__()
        assert "logger" not in state

    def test_setstate_restores_logger(self):
        """Test that __setstate__ restores logger."""
        plotter = CascadeCorrelationPlotter()
        state = plotter.__getstate__()

        new_plotter = CascadeCorrelationPlotter.__new__(CascadeCorrelationPlotter)
        new_plotter.__setstate__(state)

        assert new_plotter.logger is not None


class TestPlotDataset:
    """Tests for plot_dataset static method."""

    def test_plot_dataset_invalid_type(self):
        """Test plot_dataset with invalid input types."""
        with pytest.raises(ValidationError, match="must be torch.Tensor"):
            CascadeCorrelationPlotter.plot_dataset([1, 2, 3], [1, 0])

    def test_plot_dataset_wrong_features(self):
        """Test plot_dataset with wrong number of features."""
        x = torch.randn(10, 3)  # 3 features instead of 2
        y = torch.zeros(10, 2)
        y[:, 0] = 1

        with pytest.raises(ValidationError, match="exactly 2 features"):
            CascadeCorrelationPlotter.plot_dataset(x, y)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_dataset_valid(self, mock_figure, mock_show):
        """Test plot_dataset with valid inputs."""
        x = torch.randn(10, 2)
        y = torch.zeros(10, 2)
        y[:5, 0] = 1
        y[5:, 1] = 1

        CascadeCorrelationPlotter.plot_dataset(x, y, title="Test Dataset")
        mock_show.assert_called_once()


class TestPlotDecisionBoundary:
    """Tests for plot_decision_boundary method."""

    def test_plot_decision_boundary_plotting_disabled(self):
        """Test plot_decision_boundary when plotting is disabled."""
        plotter = CascadeCorrelationPlotter()
        mock_network = MagicMock()
        mock_network.get_generate_plots.return_value = False

        x = torch.randn(10, 2)
        y = torch.zeros(10, 2)

        # Should return without raising
        plotter.plot_decision_boundary(mock_network, x, y)
        mock_network.get_generate_plots.assert_called_once()

    def test_plot_decision_boundary_none_inputs(self):
        """Test plot_decision_boundary with None inputs."""
        plotter = CascadeCorrelationPlotter()
        mock_network = MagicMock()
        mock_network.get_generate_plots.return_value = True

        with pytest.raises(ValidationError, match="must be provided"):
            plotter.plot_decision_boundary(mock_network, None, None)

    def test_plot_decision_boundary_invalid_types(self):
        """Test plot_decision_boundary with invalid input types."""
        plotter = CascadeCorrelationPlotter()
        mock_network = MagicMock()
        mock_network.get_generate_plots.return_value = True

        with pytest.raises(ValidationError, match="must be torch.Tensor"):
            plotter.plot_decision_boundary(mock_network, [1, 2], [1, 0])

    def test_plot_decision_boundary_wrong_features(self):
        """Test plot_decision_boundary with wrong number of features."""
        plotter = CascadeCorrelationPlotter()
        mock_network = MagicMock()
        mock_network.get_generate_plots.return_value = True

        x = torch.randn(10, 3)  # 3 features instead of 2
        y = torch.zeros(10, 2)

        with pytest.raises(ValidationError, match="must have 2 features"):
            plotter.plot_decision_boundary(mock_network, x, y)


class TestPlotTrainingHistory:
    """Tests for plot_training_history method."""

    def test_plot_training_history_invalid_type(self):
        """Test plot_training_history with invalid type."""
        plotter = CascadeCorrelationPlotter()

        with pytest.raises(ValidationError, match="must be a dictionary"):
            plotter.plot_training_history("not a dict")

    def test_plot_training_history_empty(self):
        """Test plot_training_history with empty history."""
        plotter = CascadeCorrelationPlotter()

        with pytest.raises(ValidationError, match="empty or missing"):
            plotter.plot_training_history({})

    def test_plot_training_history_missing_train_loss(self):
        """Test plot_training_history with missing train_loss."""
        plotter = CascadeCorrelationPlotter()

        with pytest.raises(ValidationError, match="empty or missing"):
            plotter.plot_training_history({"train_accuracy": [0.5]})

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplot")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.tight_layout")
    def test_plot_training_history_valid(self, mock_tight, mock_plot, mock_subplot, mock_figure, mock_show):
        """Test plot_training_history with valid history."""
        plotter = CascadeCorrelationPlotter()

        history = {
            "train_loss": [0.5, 0.4, 0.3],
            "train_accuracy": [0.6, 0.7, 0.8],
            "value_loss": [0.55, 0.45, 0.35],
            "value_accuracy": [0.55, 0.65, 0.75],
            "hidden_units_added": [
                {"correlation": 0.5},
                {"correlation": 0.6},
            ],
        }

        plotter.plot_training_history(history)
        mock_show.assert_called_once()


class TestPlotHeadings:
    """Tests for _plot_headings method."""

    def test_plot_headings_invalid_plot(self):
        """Test _plot_headings with invalid plot object."""
        plotter = CascadeCorrelationPlotter()

        with pytest.raises(ValidationError, match="must be a matplotlib"):
            plotter._plot_headings(plot="not a plot")

    def test_plot_headings_invalid_title(self):
        """Test _plot_headings with invalid title."""
        plotter = CascadeCorrelationPlotter()

        with pytest.raises(ValidationError, match="title must be a string"):
            plotter._plot_headings(title=123)

    def test_plot_headings_invalid_x_label(self):
        """Test _plot_headings with invalid x_label."""
        plotter = CascadeCorrelationPlotter()

        with pytest.raises(ValidationError, match="x_label must be a string"):
            plotter._plot_headings(x_label=123)

    def test_plot_headings_invalid_y_label(self):
        """Test _plot_headings with invalid y_label."""
        plotter = CascadeCorrelationPlotter()

        with pytest.raises(ValidationError, match="y_label must be a string"):
            plotter._plot_headings(y_label=123)

    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.xlabel")
    @patch("matplotlib.pyplot.ylabel")
    @patch("matplotlib.pyplot.legend")
    def test_plot_headings_valid_with_legend(self, mock_legend, mock_ylabel, mock_xlabel, mock_title):
        """Test _plot_headings with valid inputs and legend."""
        plotter = CascadeCorrelationPlotter()

        plotter._plot_headings(plot=None, title="Test Title", x_label="X", y_label="Y", legend=True)  # Uses default plt

        mock_title.assert_called_once_with("Test Title")
        mock_xlabel.assert_called_once_with("X")
        mock_ylabel.assert_called_once_with("Y")
        mock_legend.assert_called_once()

    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.xlabel")
    @patch("matplotlib.pyplot.ylabel")
    @patch("matplotlib.pyplot.legend")
    def test_plot_headings_valid_no_legend(self, mock_legend, mock_ylabel, mock_xlabel, mock_title):
        """Test _plot_headings without legend."""
        plotter = CascadeCorrelationPlotter()

        plotter._plot_headings(plot=None, title="Test", x_label="X", y_label="Y", legend=False)

        mock_legend.assert_not_called()
