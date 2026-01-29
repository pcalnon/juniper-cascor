#!/usr/bin/env python
"""
Extended tests for cascor_plotter/cascor_plotter.py to improve code coverage.

Targets uncovered lines 164-196 (plot_decision_boundary full path) and line 287
(_plot_headings validation for missing methods).
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


class TestPlotDecisionBoundaryFullPath:
    """Tests for plot_decision_boundary covering lines 164-196."""

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.contourf")
    @patch("matplotlib.pyplot.scatter")
    def test_plot_decision_boundary_full_execution(self, mock_scatter, mock_contourf, mock_figure, mock_show):
        """Test plot_decision_boundary with valid inputs for full execution path (lines 164-196)."""
        plotter = CascadeCorrelationPlotter()

        mock_network = MagicMock()
        mock_network.get_generate_plots.return_value = True

        x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        y = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float32)

        def predict_classes_side_effect(input_tensor):
            return torch.zeros(input_tensor.shape[0], dtype=torch.long)

        mock_network.predict_classes.side_effect = predict_classes_side_effect

        with patch.object(plotter, "_plot_headings"):
            plotter.plot_decision_boundary(mock_network, x, y, title="Test Decision Boundary")

        mock_network.get_generate_plots.assert_called_once()
        mock_network.predict_classes.assert_called_once()
        mock_figure.assert_called_once()
        mock_contourf.assert_called_once()
        mock_show.assert_called_once()

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.contourf")
    @patch("matplotlib.pyplot.scatter")
    def test_plot_decision_boundary_multiple_classes(self, mock_scatter, mock_contourf, mock_figure, mock_show):
        """Test plot_decision_boundary with multiple classes."""
        plotter = CascadeCorrelationPlotter()

        mock_network = MagicMock()
        mock_network.get_generate_plots.return_value = True

        x = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]],
            dtype=torch.float32,
        )
        y = torch.tensor(
            [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
            dtype=torch.float32,
        )

        def predict_classes_side_effect(input_tensor):
            return torch.zeros(input_tensor.shape[0], dtype=torch.long)

        mock_network.predict_classes.side_effect = predict_classes_side_effect

        with patch.object(plotter, "_plot_headings"):
            plotter.plot_decision_boundary(mock_network, x, y, title="Multi-class Boundary")

        assert mock_scatter.call_count == 3

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.contourf")
    @patch("matplotlib.pyplot.scatter")
    def test_plot_decision_boundary_calls_plot_headings(self, mock_scatter, mock_contourf, mock_figure, mock_show):
        """Test that plot_decision_boundary calls _plot_headings with correct args."""
        plotter = CascadeCorrelationPlotter()

        mock_network = MagicMock()
        mock_network.get_generate_plots.return_value = True

        x = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        y = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

        def predict_classes_side_effect(input_tensor):
            return torch.zeros(input_tensor.shape[0], dtype=torch.long)

        mock_network.predict_classes.side_effect = predict_classes_side_effect

        with patch.object(plotter, "_plot_headings") as mock_headings:
            plotter.plot_decision_boundary(mock_network, x, y, title="Custom Title")
            mock_headings.assert_called_once()
            call_kwargs = mock_headings.call_args
            assert call_kwargs[1]["title"] == "Custom Title"
            assert call_kwargs[1]["x_label"] == "X1"
            assert call_kwargs[1]["y_label"] == "Y1"


class TestPlotHeadingsMissingMethods:
    """Tests for _plot_headings covering line 287 (missing methods validation).

    Note: Line 287 can only be reached if the plot object passes the isinstance check
    on line 278 (isinstance(plot, type(plt))). This is challenging to test directly
    since matplotlib.pyplot is a module, not a class. These tests verify related
    validation behavior.
    """

    @pytest.mark.unit
    def test_plot_headings_invalid_plot_type(self):
        """Test _plot_headings with non-matplotlib plot object."""
        plotter = CascadeCorrelationPlotter()

        mock_plot = MagicMock(spec=["xlabel", "ylabel", "title"])

        with pytest.raises(ValidationError, match="must be a matplotlib.pyplot object"):
            plotter._plot_headings(plot=mock_plot, title="Test", x_label="X", y_label="Y")

    @pytest.mark.unit
    def test_plot_headings_string_plot(self):
        """Test _plot_headings with string instead of plot."""
        plotter = CascadeCorrelationPlotter()

        with pytest.raises(ValidationError, match="must be a matplotlib.pyplot object"):
            plotter._plot_headings(plot="not_a_plot", title="Test", x_label="X", y_label="Y")

    @pytest.mark.unit
    def test_plot_headings_number_plot(self):
        """Test _plot_headings with number instead of plot."""
        plotter = CascadeCorrelationPlotter()

        with pytest.raises(ValidationError, match="must be a matplotlib.pyplot object"):
            plotter._plot_headings(plot=123, title="Test", x_label="X", y_label="Y")

    @pytest.mark.unit
    def test_plot_headings_list_plot(self):
        """Test _plot_headings with list instead of plot."""
        plotter = CascadeCorrelationPlotter()

        with pytest.raises(ValidationError, match="must be a matplotlib.pyplot object"):
            plotter._plot_headings(plot=[1, 2, 3], title="Test", x_label="X", y_label="Y")


class TestPlotTrainingHistoryExtended:
    """Extended tests for plot_training_history edge cases."""

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplot")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.tight_layout")
    def test_plot_training_history_no_validation_data(self, mock_tight, mock_plot, mock_subplot, mock_figure, mock_show):
        """Test plot_training_history without validation loss/accuracy."""
        plotter = CascadeCorrelationPlotter()

        history = {
            "train_loss": [0.5, 0.4, 0.3],
            "train_accuracy": [0.6, 0.7, 0.8],
            "hidden_units_added": [{"correlation": 0.5}],
        }

        plotter.plot_training_history(history)
        mock_show.assert_called_once()

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplot")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.tight_layout")
    def test_plot_training_history_empty_hidden_units(self, mock_tight, mock_plot, mock_subplot, mock_figure, mock_show):
        """Test plot_training_history with empty hidden_units_added."""
        plotter = CascadeCorrelationPlotter()

        history = {
            "train_loss": [0.5, 0.4],
            "train_accuracy": [0.6, 0.7],
            "hidden_units_added": [],
        }

        plotter.plot_training_history(history)
        mock_show.assert_called_once()

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplot")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.tight_layout")
    def test_plot_training_history_empty_validation_lists(self, mock_tight, mock_plot, mock_subplot, mock_figure, mock_show):
        """Test plot_training_history with empty validation lists."""
        plotter = CascadeCorrelationPlotter()

        history = {
            "train_loss": [0.5, 0.4],
            "train_accuracy": [0.6, 0.7],
            "value_loss": [],
            "value_accuracy": [],
            "hidden_units_added": [],
        }

        plotter.plot_training_history(history)
        mock_show.assert_called_once()


class TestPlotDatasetExtended:
    """Extended tests for plot_dataset edge cases."""

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.scatter")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.xlabel")
    @patch("matplotlib.pyplot.ylabel")
    @patch("matplotlib.pyplot.legend")
    def test_plot_dataset_single_class(self, mock_legend, mock_ylabel, mock_xlabel, mock_title, mock_scatter, mock_figure, mock_show):
        """Test plot_dataset with single class."""
        x = torch.randn(10, 2)
        y = torch.zeros(10, 2)
        y[:, 0] = 1

        CascadeCorrelationPlotter.plot_dataset(x, y, title="Single Class")
        mock_show.assert_called_once()
        mock_scatter.assert_called_once()

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.scatter")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.xlabel")
    @patch("matplotlib.pyplot.ylabel")
    @patch("matplotlib.pyplot.legend")
    def test_plot_dataset_many_classes(self, mock_legend, mock_ylabel, mock_xlabel, mock_title, mock_scatter, mock_figure, mock_show):
        """Test plot_dataset with many classes."""
        x = torch.randn(20, 2)
        y = torch.zeros(20, 5)
        for i in range(5):
            y[i * 4 : (i + 1) * 4, i] = 1

        CascadeCorrelationPlotter.plot_dataset(x, y, title="Many Classes")
        mock_show.assert_called_once()
        assert mock_scatter.call_count == 5
