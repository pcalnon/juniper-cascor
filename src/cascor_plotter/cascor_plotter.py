#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     plotter.py
# Author:        Paul Calnon
# Version:       0.3.2 (0.7.3)
#
# Date Created:  2025-09-26
# Last Modified: 2026-01-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This file contains the class used to plot datasets, classification boundaries, and training history for the Cascade Correlation Neural Network.
#
#####################################################################################################################################################################################################
# Notes:
#   - The Cascade Correlation Neural Network is designed to incrementally add hidden units to the network.
#   - The network uses a correlation-based approach to determine the relevance of each candidate unit.
#   - The network is trained using a combination of supervised and unsupervised learning techniques.
#   - The network is designed to handle large-scale and high-dimensional datasets efficiently.
# 
#####################################################################################################################################################################################################
# References:
#
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#
#####################################################################################################################################################################################################
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import current_process

from log_config.logger.logger import Logger

from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import (
    ValidationError
)

#####################################################################################################################################################################################################
# Plotting functionality for Cascade Correlation Network
#####################################################################################################################################################################################################
class CascadeCorrelationPlotter:
    """Handles all plotting functionality for the Cascade Correlation Network."""

    def __init__(self, logger=None):
        """
        Initialize the plotter.

        Args:
            logger: Logger instance for debugging output
        """
        self.logger = logger or Logger

    def __getstate__(self):
        """Remove non-picklable logger for multiprocessing."""
        state = self.__dict__.copy()
        state.pop('logger', None)
        return state

    def __setstate__(self, state):
        """Restore state and reinitialize logger."""
        self.__dict__.update(state)
        from log_config.logger.logger import Logger
        self.logger = Logger

    @staticmethod
    def plot_dataset(
        x: torch.Tensor,
        y: torch.Tensor,
        title: str = "Training Dataset",
    ) -> None:
        """
        Plot the training dataset.

        Args:
            x: Input tensor (must have 2 features for 2D plotting)
            y: Target tensor (one-hot encoded labels)
            title: Plot title

        Raises:
            ValidationError: If input tensors are not valid for plotting
        """
        logger = Logger
        logger.set_level("INFO")
        logger.trace("CascadeCorrelationPlotter: plot_dataset: Starting to plot the dataset.")

        process_info = current_process()
        process = (process_info.pid, process_info.name,)
        logger.debug(f"CascadeCorrelationPlotter: plot_dataset: Process ID: {os.getpid()}, Process ID: {process[0]}, Process Name: {process[1]}")

        # Validate inputs
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValidationError("Input x and target y must be torch.Tensor objects")

        if x.shape[1] != 2:
            raise ValidationError("Input tensor must have exactly 2 features for 2D plotting")

        # Convert to numpy for plotting
        logger.debug(f"CascadeCorrelationPlotter: plot_dataset: process {process[0]}: Converting input and target tensors to numpy arrays for plotting.")
        logger.debug(f"CascadeCorrelationPlotter: plot_dataset: process {process[0]}: Input shape: {x.shape}, Target shape: {y.shape},\nY Value:\n{y}")
        x_np = x.numpy()
        y_np = torch.argmax(y, dim=1).numpy()

        # Plot the figure and labels
        logger.info(f"CascadeCorrelationPlotter: plot_dataset: process {process[0]}: Plotting dataset with title: {title}")
        plt.figure(figsize=(10, 8))
        for i in range(len(np.unique(y_np))):
            plt.scatter(x_np[y_np == i, 0], x_np[y_np == i, 1], label=f'Class {i}')
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()
        logger.trace(f"CascadeCorrelationPlotter: plot_dataset: process {process[0]}: Completed plotting the dataset.")

    def plot_decision_boundary(
        self,
        network,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        title: str = "Decision Boundary",
    ) -> None:
        """
        Plot the decision boundary of the network.
        
        Args:
            network: CascadeCorrelationNetwork instance
            x: Input tensor (must have 2 features for 2D plotting)
            y: Target tensor (one-hot encoded labels)
            title: Plot title
            
        Raises:
            ValidationError: If input tensors are not valid for plotting
        """
        self.logger.trace("CascadeCorrelationPlotter: plot_decision_boundary: Starting to plot the decision boundary.")

        # Validate that plotting is enabled
        if not network.get_generate_plots():
            self.logger.warning("CascadeCorrelationPlotter: plot_decision_boundary: Plotting solutions is disabled. Set 'generate_plots' to True to enable plotting.")
            return
        self.logger.debug("CascadeCorrelationPlotter: plot_decision_boundary: Plotting Solutions is enabled. Proceeding to plot decision boundary.")

        # Validate input and target tensors
        self.logger.debug("CascadeCorrelationPlotter: plot_decision_boundary: Verifying input and target tensors for plotting decision boundary.")
        if x is None or y is None:
            raise ValidationError("Input (x) and target (y) tensors must be provided for plotting the decision boundary.")
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValidationError("Input (x) and target (y) must be torch.Tensor objects.")
        if x.shape[1] != 2:
            raise ValidationError("Input tensor must have 2 features.")
        self.logger.debug("CascadeCorrelationPlotter: plot_decision_boundary: Successfully Verified input and target tensors for plotting decision boundary.")
        self.logger.debug(f"CascadeCorrelationPlotter: plot_decision_boundary: Plotting decision boundary for input shape: {x.shape}, target shape: {y.shape}")

        # Convert to numpy for plotting
        self.logger.debug("CascadeCorrelationPlotter: plot_decision_boundary: Converting input and target tensors to numpy arrays for plotting.")
        x_np = x.numpy()
        y_np = torch.argmax(y, dim=1).numpy()

        # Create a mesh grid
        self.logger.debug("CascadeCorrelationPlotter: plot_decision_boundary: Creating mesh grid for plotting decision boundary.")
        h = 0.02  # step size in the mesh
        x_min, x_max = x_np[:, 0].min() - 1, x_np[:, 0].max() + 1
        y_min, y_max = x_np[:, 1].min() - 1, x_np[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict class labels for each point in the mesh
        self.logger.debug("CascadeCorrelationPlotter: plot_decision_boundary: Predicting class labels for each point in the mesh grid.")
        self.logger.debug(f"CascadeCorrelationPlotter: plot_decision_boundary: Mesh grid shape: {xx.shape}, {yy.shape}")
        Z = network.predict_classes(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).numpy()
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        self.logger.debug("CascadeCorrelationPlotter: plot_decision_boundary: Plotting the decision boundary.")
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8)

        # Plot the training points
        self.logger.debug("CascadeCorrelationPlotter: plot_decision_boundary: Plotting the training points.")
        for i in range(len(np.unique(y_np))):
            plt.scatter(x_np[y_np == i, 0], x_np[y_np == i, 1], label=f'Class {i}')
        self._plot_headings(plot=plt, title=title, x_label="X1", y_label="Y1")
        plt.show()
        self.logger.trace("CascadeCorrelationPlotter: plot_decision_boundary: Completed plotting the decision boundary.")

    def plot_training_history(self, history: dict):
        """
        Plot the training history of the network.
        
        Args:
            history: Dictionary containing training history data
            
        Raises:
            ValidationError: If training history is empty or invalid
        """
        self.logger.trace("CascadeCorrelationPlotter: plot_training_history: Starting to plot the training history.")
        
        if not isinstance(history, dict):
            raise ValidationError("Training history must be a dictionary.")
            
        if not history.get('train_loss'):
            raise ValidationError("Training history is empty or missing required data.")
            
        plt.figure(figsize=(15, 10))
        self.logger.debug(f"CascadeCorrelationPlotter: plot_training_history: History: {history}")

        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        if 'value_loss' in history and history['value_loss']:
            plt.plot(history['value_loss'], label='Validation Loss')
        self._plot_headings(plot=plt, title="Loss During Training", x_label="Epochs", y_label="Loss")

        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        if 'value_accuracy' in history and history['value_accuracy']:
            plt.plot(history['value_accuracy'], label='Validation Accuracy')
        self._plot_headings(plot=plt, title="Accuracy During Training", x_label="Epochs", y_label="Accuracy")

        # Plot number of hidden units
        plt.subplot(2, 2, 3)
        plt.plot(range(len(history['hidden_units_added']) + 1), [0] + [i+1 for i in range(len(history['hidden_units_added']))])
        self._plot_headings(plot=plt, title="Number of Hidden Units", x_label="Epochs", y_label="Number of Units", legend=False)

        # Plot correlation of added units
        if history['hidden_units_added']:
            plt.subplot(2, 2, 4)
            correlations = [unit['correlation'] for unit in history['hidden_units_added']]
            plt.plot(correlations)
            self._plot_headings(plot=plt, title="Correlation of Added Units", x_label="Unit Number", y_label="Correlation", legend=False)

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()
        self.logger.trace("CascadeCorrelationPlotter: plot_training_history: Completed plotting the training history.")

    def _plot_headings(
        self,
        plot: plt = None,
        title: str = "Headings",
        x_label: str = "X axis",
        y_label: str = "Y axis",
        legend: bool = True,
    ) -> None:
        """
        Set headings for the plot.
        
        Args:
            plot: Matplotlib plot object
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            legend: Whether to display legend
            
        Raises:
            ValidationError: If plot object or parameters are invalid
        """
        self.logger.trace("CascadeCorrelationPlotter: _plot_headings: Starting to set plot headings.")
        self.logger.debug(f"CascadeCorrelationPlotter: _plot_headings: Setting plot headings: Title: {title}, X Label: {x_label}, Y Label: {y_label}, Legend: {legend}")
        self.logger.debug(f"CascadeCorrelationPlotter: _plot_headings: Plot type: {type(plot)}, Plot: {plot}")

        # Validate inputs
        if plot is None:
            plot = plt
        if not isinstance(plot, type(plt)):
            raise ValidationError("plot must be a matplotlib.pyplot object")
        if not isinstance(title, str):
            raise ValidationError("title must be a string")
        if not isinstance(x_label, str):
            raise ValidationError("x_label must be a string")
        if not isinstance(y_label, str):
            raise ValidationError("y_label must be a string")
        if not hasattr(plot, 'title') or not hasattr(plot, 'xlabel') or not hasattr(plot, 'ylabel'):
            raise ValidationError("plot must have title, xlabel, and ylabel methods")

        # Set plot headings
        plot.title(title)
        plot.xlabel(x_label)
        plot.ylabel(y_label)

        # Display legend if requested
        if legend:
            plot.legend()
        self.logger.debug(f"CascadeCorrelationPlotter: _plot_headings: Plot headings set: Title: {title}, X Label: {x_label}, Y Label: {y_label}, Legend: {legend}")
        self.logger.trace("CascadeCorrelationPlotter: _plot_headings: Completed setting plot headings.")

