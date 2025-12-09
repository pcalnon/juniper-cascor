#!/usr/bin/env python3
#####################################################################################################################################################################################################
# Project:       Cascade Correlation Neural Network
# File Name:     cascor.py
# Author:        Paul Calnon
# Version:       1.0.1
# Date:          2025-06-11
# Last Modified: 2025-06-11
# License:       MIT License
#
# Description:
#    This file contains the functions and code needed to solve the two spiral problem using a Cascade Correlation Neural Network.
#
#
#####################################################################################################################################################################################################
# References:
#
#
#
#####################################################################################################################################################################################################
# TODO:
#
#
#
#####################################################################################################################################################################################################

import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from typing import List, Tuple, Optional, Dict, Any
# import logging
import random
# import math
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

from constants.constants import (
    _CASCOR_INPUT_SIZE,
    _CASCOR_OUTPUT_SIZE,
    _CASCOR_CANDIDATE_POOL_SIZE,
    _CASCOR_ACTIVATION_FUNCTION,
    _CASCOR_LEARNING_RATE,
    _CASCOR_MAX_HIDDEN_UNITS,
    _CASCOR_CORRELATION_THRESHOLD,
    _CASCOR_PATIENCE,
    _CASCOR_CANDIDATE_EPOCHS,
    _CASCOR_OUTPUT_EPOCHS,
)


#####################################################################################################################################################################################################
# Define function to generate the two spiral problem dataset.
# TODO: Convert this to use spiral problem in Project Data Dir.
def generate_two_spiral_data(n_points=100, noise=0.0):
    """
    Generate the two spiral problem dataset.
    
    Args:
        n_points: Number of points per spiral
        noise: Amount of noise to add
        
    Returns:
        x: Input features
        y: One-hot encoded targets
    """
    n = np.sqrt(np.random.rand(n_points)) * 780 * (2 * np.pi) / 360
    
    d1x = -np.cos(n) * n + np.random.rand(n_points) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points) * noise
    
    d2x = np.cos(n) * n + np.random.rand(n_points) * noise
    d2y = -np.sin(n) * n + np.random.rand(n_points) * noise
    
    # Create input features
    x = np.vstack([
        np.hstack([d1x, d2x]),
        np.hstack([d1y, d2y])
    ]).T
    
    # Create targets (one-hot encoded)
    y = np.zeros((2 * n_points, 2))
    y[:n_points, 0] = 1
    y[n_points:, 1] = 1
    
    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return x_tensor, y_tensor


#####################################################################################################################################################################################################
# Define function to solve the two spiral problem using Cascade Correlation Network.
def solve_two_spiral_problem(n_points=100, noise=0.05, plot=True):
    """
    Solve the two spiral problem using Cascade Correlation.
    
    Args:
        n_points: Number of points per spiral
        noise: Amount of noise to add
        plot: Whether to plot the results
        
    Returns:
        Trained CascadeCorrelationNetwork
    """
    # Generate the two spiral dataset
    x, y = generate_two_spiral_data(n_points, noise)
    
    # Create and train the network
    network = CascadeCorrelationNetwork(
        input_size=_CASCOR_INPUT_SIZE,
        output_size=_CASCOR_OUTPUT_SIZE,
        candidate_pool_size=_CASCOR_CANDIDATE_POOL_SIZE,
        learning_rate=_CASCOR_LEARNING_RATE,
        max_hidden_units=_CASCOR_MAX_HIDDEN_UNITS,
        correlation_threshold=_CASCOR_CORRELATION_THRESHOLD,
        patience=_CASCOR_PATIENCE,
        candidate_epochs=_CASCOR_CANDIDATE_EPOCHS,
        output_epochs=_CASCOR_OUTPUT_EPOCHS,
        activation_function=_CASCOR_ACTIVATION_FUNCTION,
    )

    # Train the network
    history = network.fit(x, y, max_epochs=50)
    print(f"Training history: {history}")
    
    # Print summary
    network.summary()
    
    # Plot results
    if plot:
        network.plot_decision_boundary(x, y, "Two Spiral Problem - Decision Boundary")
        network.plot_training_history()
    
    return network


#####################################################################################################################################################################################################
# Main function to run the two spiral problem solution
# This is the entry point for the script.
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Solve the two spiral problem
    print("Solving the two spiral problem with Cascade Correlation...")
    network = solve_two_spiral_problem(n_points=100, noise=0.0)
    
    # Evaluate the final accuracy
    x, y = generate_two_spiral_data(n_points=100, noise=0.0)
    accuracy = network.calculate_accuracy(x, y)
    print(f"Final accuracy on the two spiral problem: {accuracy:.4f}")
