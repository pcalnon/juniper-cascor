#!/usr/bin/env python3
#####################################################################################################################################################################################################
# Project:       Cascade Correlation Neural Network
# File Name:     cascade_correlation.py
# Author:        Paul Calnon
# Version:       1.0.1
# Date Created:  2025-06-11
# Last Modified: 2026-01-12
# License:       MIT License
#
# Description:
#    This file contains the implementation of the Cascade Correlation Neural Network.
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
import matplotlib
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# from typing import List, Tuple, Optional, Dict, Any
from typing import List, Optional, Dict
import logging
# import random
# import math

from candidate_unit.candidate_unit import CandidateUnit

from constants.constants import (
    _CASCADE_CORRELATION_NETWORK_INPUT_SIZE,
    _CASCADE_CORRELATION_NETWORK_OUTPUT_SIZE,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_POOL_SIZE,
    _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION,
    _CASCADE_CORRELATION_NETWORK_LEARNING_RATE,
    _CASCADE_CORRELATION_NETWORK_MAX_HIDDEN_UNITS,
    _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD,
    _CASCADE_CORRELATION_NETWORK_PATIENCE,
    _CASCADE_CORRELATION_NETWORK_CANDIDATE_EPOCHS,
    _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS,
    _CASCADE_CORRELATION_NETWORK_LOGLEVEL_DEFAULT,
)


# # Define constants for the Cascade Correlation Network
# _CASCADE_CORRELATION_NETWORK_INPUT_SIZE = 2
# _CASCADE_CORRELATION_NETWORK_OUTPUT_SIZE = 2
# _CASCADE_CORRELATION_NETWORK_CANDIDATE_POOL_SIZE = 20
# _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION = torch.tanh
# _CASCADE_CORRELATION_NETWORK_LEARNING_RATE = 0.1
# # _CASCADE_CORRELATION_NETWORK_LEARNING_RATE = 0.05
# _CASCADE_CORRELATION_NETWORK_MAX_HIDDEN_UNITS = 40
# # _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD = 0.4
# # _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD = 0.125
# _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD = 0.05
# # _CASCADE_CORRELATION_NETWORK_PATIENCE = 5
# _CASCADE_CORRELATION_NETWORK_PATIENCE = 10
# _CASCADE_CORRELATION_NETWORK_CANDIDATE_EPOCHS = 400
# _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS = 500
# # _CASCADE_CORRELATION_NETWORK_LOGLEVEL_DEFAULT = logging.DEBUG
# _CASCADE_CORRELATION_NETWORK_LOGLEVEL_DEFAULT = logging.INFO


#####################################################################################################################################################################################################
# Class definition for the Cascade Correlation Network
class CascadeCorrelationNetwork:
    """
    Description:
        Cascade Correlation Network class.
        This class implements the Cascade Correlation Neural Network algorithm.
        It allows for dynamic growth of the network by adding hidden units based on candidate units
        trained on the residual error of the network.
    Attributes:
        logger (logging.Logger): Logger for the network.
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        candidate_pool_size (int): Number of candidate units to train.
        activation_fn (torch): Activation function to use.
        learning_rate (float): Learning rate for the network.
        max_hidden_units (int): Maximum number of hidden units to allow.
        correlation_threshold (float): Threshold for candidate unit correlation.
        patience (int): Patience for early stopping.
        candidate_epochs (int): Number of epochs for candidate unit training.
        output_epochs (int): Number of epochs for output layer training.
        hidden_units (List[Dict]): List of hidden units in the network.
        output_weights (torch.Tensor): Weights for the output layer.
        output_bias (torch.Tensor): Bias for the output layer.
        history (Dict[str, List]): Dictionary to store training history.
    Args:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        candidate_pool_size (int): Number of candidate units to train.
        activation_function (torch): Activation function to use.
        learning_rate (float): Learning rate for the network.
        max_hidden_units (int): Maximum number of hidden units to allow.
        correlation_threshold (float): Threshold for candidate unit correlation.
        patience (int): Patience for early stopping.
        candidate_epochs (int): Number of epochs for candidate unit training.
        output_epochs (int): Number of epochs for output layer training.
        logging_level (logging.LEVEL): Logging level for the network.
    Methods:
        add_hidden_units(self, input_size: int, output_size: int, hidden_units: int) -> None:  Adds hidden units to the network.
        forward(self, x: torch.Tensor) -> torch.Tensor:  Performs a forward pass through the network.
        update_candidate_units(self, x: torch.Tensor, y: torch.Tensor) -> None:  Updates the candidate units based on the residual error.
        train_output_layer(self, x: torch.Tensor, y: torch.Tensor, epochs: int) -> float:  Trains the output layer of the network.
        train_candidates(self, x: torch.Tensor, residual_error: torch.Tensor) -> List[CandidateUnit]:  Trains the candidate units based on the residual error.
        add_unit(self, candidate: CandidateUnit, x: torch.Tensor) -> None:  Adds a candidate unit to the network.
        calculate_residual_error(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  Calculates the residual error between the predicted and actual outputs.
        fit(self, x_train: torch.Tensor, y_train: torch.Tensor, x_val: Optional[torch.Tensor] = None, y_val: Optional[torch.Tensor] = None, max_epochs: int = 1000, early_stopping: bool = True) -> Dict[str, List]:  
            Trains the Cascade Correlation Network.  Returns a dictionary containing the training history.
    """


    #################################################################################################################################################################################################
    # Constructor for the Cascade Correlation Network
    def __init__(
        self,
        input_size: int = _CASCADE_CORRELATION_NETWORK_INPUT_SIZE,
        output_size: int = _CASCADE_CORRELATION_NETWORK_OUTPUT_SIZE,
        candidate_pool_size: int = _CASCADE_CORRELATION_NETWORK_CANDIDATE_POOL_SIZE,
        activation_function: torch = _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION,
        learning_rate: float = _CASCADE_CORRELATION_NETWORK_LEARNING_RATE,
        max_hidden_units: int = _CASCADE_CORRELATION_NETWORK_MAX_HIDDEN_UNITS,
        correlation_threshold: float = _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD,
        patience: int = _CASCADE_CORRELATION_NETWORK_PATIENCE,
        candidate_epochs: int = _CASCADE_CORRELATION_NETWORK_CANDIDATE_EPOCHS,
        output_epochs: int = _CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS,
        logging_level: logging = _CASCADE_CORRELATION_NETWORK_LOGLEVEL_DEFAULT,
    ):
        """
        Initialize the Cascade Correlation Network.
        Args:
            input_size (int): Size of the input layer.
            output_size (int): Size of the output layer.
            candidate_pool_size (int): Number of candidate units to train.
            activation_fn (torch): Activation function to use.
            learning_rate (float): Learning rate for the network.
            max_hidden_units (int): Maximum number of hidden units to allow.
            correlation_threshold (float): Threshold for candidate unit correlation.
            patience (int): Patience for early stopping.
            candidate_epochs (int): Number of epochs for candidate unit training.
            output_epochs (int): Number of epochs for output layer training.
            logging_level (logging.LEVEL): Logging level for the network.
        """
        super().__init__()

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(fmt="[%(filename)s:%(lineno)d] (%(asctime)s) [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("Initializing Cascade Correlation Network with parameters:")
        self.input_size = input_size
        self.output_size = output_size
        self.candidate_pool_size = candidate_pool_size
        self.activation_fn = self._get_activation_with_derivative(activation_function)
        self.learning_rate = learning_rate
        self.max_hidden_units = max_hidden_units
        self.correlation_threshold = correlation_threshold
        self.patience = patience
        self.candidate_epochs = candidate_epochs
        self.output_epochs = output_epochs
        
        # Initialize network Model Parameters

        self.hidden_units = []
        self.output_weights = torch.randn(input_size, output_size, requires_grad=True) * 0.1
        self.output_bias = torch.randn(output_size, requires_grad=True) * 0.1
        
        self.history = {
            'train_loss': [],
            'value_loss': [],
            'train_accuracy': [],
            'value_accuracy': [],
            'hidden_units_added': []
        }
        self.logger.info("Cascade Correlation Network initialized with parameters")


    #################################################################################################################################################################################################
    # Helper method to add hidden units to the network
    def _get_activation_with_derivative(self, activation_fn):
        """
        Wrap activation function to also provide its derivative.
        
        Args:
            activation_fn: Base activation function
            
        Returns:
            Function that can compute both activation and its derivative
        """
        def wrapped_activation(x, derivative=False):
            if derivative:
                # For tanh, derivative is 1 - tanh^2(x)
                if activation_fn == torch.tanh:
                    return 1.0 - activation_fn(x)**2
                # For sigmoid, derivative is sigmoid(x) * (1 - sigmoid(x))
                elif activation_fn == torch.sigmoid:
                    y = activation_fn(x)
                    return y * (1.0 - y)
                # For ReLU, derivative is 1 for x > 0, 0 otherwise
                elif activation_fn == torch.relu:
                    return (x > 0).float()
                else:
                    # Numerical approximation for other functions
                    eps = 1e-6
                    return (activation_fn(x + eps) - activation_fn(x - eps)) / (2 * eps)
            else:
                return activation_fn(x)
        return wrapped_activation


    #################################################################################################################################################################################################
    # Public Method that Performs a Forward pass through the network
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Network output
        """
        # Start with the input features
        features = x
        print(f"Input shape: {features.shape}")

        # Pass through each hidden unit
        hidden_outputs = []
        for unit in self.hidden_units:
            # Concatenate all previous outputs with the input
            unit_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
            # Get output from this unit
            unit_output = unit['activation_fn'](torch.sum(unit_input * unit['weights'], dim=1) + unit['bias']).unsqueeze(1)
            hidden_outputs.append(unit_output)

        # Prepare input for the output layer
        output_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
        # Output layer (linear combination)
        output = torch.matmul(output_input, self.output_weights) + self.output_bias
        print(f"Output shape: {output.shape}")

        return output


    #################################################################################################################################################################################################
    # Public Method to train the output layer of the network
    def train_output_layer(self, x: torch.Tensor, y: torch.Tensor, epochs: int) -> float:
        """
        Train only the output layer of the network.
        
        Args:
            x: Input tensor
            y: Target tensor
            epochs: Number of training epochs
            
        Returns:
            Final loss value
        """
        criterion = nn.MSELoss()

        # # Ensure output weights and bias have requires_grad=True
        # # if not self.output_weights.requires_grad:
        # #     self.output_weights = self.output_weights.detach().clone().requires_grad_(True)
        # # if not self.output_bias.requires_grad:
        # #     self.output_bias = self.output_bias.detach().clone().requires_grad_(True)

        # # Create proper leaf tensors for optimization
        # # Instead of modifying existing tensors, create new Parameter objects
        # output_weights = nn.Parameter(self.output_weights.detach().clone())
        # output_bias = nn.Parameter(self.output_bias.detach().clone())


        # Create a simple linear layer for the output
        input_size = x.shape[1]
        self.logger.debug(f"Input size for output layer: {input_size}, Output size: {self.output_size}")
        if self.hidden_units:
            input_size += len(self.hidden_units)
        self.logger.debug(f"Adjusted input size for output layer with hidden units: {input_size}")
        
        # Create a temporary linear layer with the same weights as our current output layer
        output_layer = nn.Linear(input_size, self.output_size)
        with torch.no_grad():
            output_layer.weight.copy_(self.output_weights.t())  # Transpose because nn.Linear expects (out_features, in_features)
            self.logger.debug(f"Output weights shape: {self.output_weights.shape}, Transposed weights shape: {output_layer.weight.shape}")
            output_layer.bias.copy_(self.output_bias)
            self.logger.debug(f"Output bias shape: {self.output_bias.shape}, Bias: {output_layer.bias}")
        
        # Use this layer for optimization
        optimizer = optim.Adam(output_layer.parameters(), lr=self.learning_rate)
        self.logger.debug(f"Learning Rate: {self.learning_rate}, Optimizer:\n{optimizer}")
        self.logger.debug(f"Output layer initialized with weights shape: {output_layer.weight.shape}, Bias shape: {output_layer.bias.shape}")


        # # Use these parameters for optimization
        # # optimizer = optim.Adam([self.output_weights, self.output_bias], lr=self.learning_rate)
        # optimizer = optim.Adam([output_weights, output_bias], lr=self.learning_rate)
        
        for epoch in range(epochs):
            # Forward pass - use the parameters directly instead of the original tensors

            # output_input = torch.cat([x] + [unit['activation_fn'](
            #     torch.sum(torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x * unit['weights'], dim=1) + unit['bias']
            # ).unsqueeze(1) for unit, hidden_outputs in self._get_hidden_outputs_for_units(x)], dim=1) if self.hidden_units else x
        

            # # Forward pass
            # # output = self.forward(x)
            # output = torch.matmul(output_input, output_weights) + output_bias
            # loss = criterion(output, y)

            # Get the input for the output layer (original input + hidden unit outputs)
            hidden_outputs = []
            for unit in self.hidden_units:
                unit_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
                unit_output = unit['activation_fn'](torch.sum(unit_input * unit['weights'], dim=1) + unit['bias']).unsqueeze(1)
                hidden_outputs.append(unit_output)
            
            output_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
            output = output_layer(output_input)
            loss = criterion(output, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                self.logger.debug(f"Output Layer Training - Epoch {epoch}, Loss: {loss.item():.6f}")
        
        # # Update the model parameters with the trained values
        # self.output_weights = output_weights.detach().clone()
        # self.output_bias = output_bias.detach().clone()

        # Update our model's weights with the trained values
        with torch.no_grad():
            self.output_weights = output_layer.weight.t().clone()  # Transpose back
            self.logger.debug(f"Output weights shape: {self.output_weights.shape}, Weights:\n{self.output_weights}")
            self.output_bias = output_layer.bias.clone()
            self.logger.debug(f"Output bias shape: {self.output_bias.shape}, Bias:\n{self.output_bias}")

        # Final loss
        with torch.no_grad():
            output = self.forward(x)
            self.logger.debug(f"Final output shape: {output.shape}, Output: {output}")
            final_loss = criterion(output, y).item()
            self.logger.info(f"Final output layer training loss: {final_loss:.6f}")

        return final_loss


    #################################################################################################################################################################################################
    # def _get_hidden_outputs_for_units(self, x):
    #     """
    #     Helper method to get hidden outputs for each unit.
    #
    #     Args:
    #         x: Input tensor
    #
    #     Returns:
    #         List of (unit, hidden_outputs) tuples
    #     """
    #     result = []
    #     hidden_outputs = []
    #     for unit in self.hidden_units:
    #         result.append((unit, hidden_outputs.copy()))
    #         unit_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
    #         unit_output = unit['activation_fn'](torch.sum(unit_input * unit['weights'], dim=1) + unit['bias']).unsqueeze(1)
    #         hidden_outputs.append(unit_output)
    #     return result


    #################################################################################################################################################################################################
    # Public Method to update candidate units based on the residual error
    def train_candidates(
        self,
        x: torch.Tensor,
        residual_error: torch.Tensor,
    ) -> List[CandidateUnit]:
        """
        Train a pool of candidate units and select the best one.

        Args:
            x: Input tensor
            residual_error: Residual error from the network

        Returns:
            List of trained candidate units
        """
        # Prepare input for candidates (includes outputs from existing hidden units)
        hidden_outputs = []
        for unit in self.hidden_units:
            unit_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
            # Concatenate input with outputs from existing hidden units
            unit_output = unit['activation_fn'](torch.sum(unit_input * unit['weights'], dim=1) + unit['bias']).unsqueeze(1)
            hidden_outputs.append(unit_output)
        self.logger.debug(f"Hidden outputs shape: {[h.shape for h in hidden_outputs]}")
        self.logger.debug(f"Hidden outputs: {hidden_outputs}")

        candidate_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
        input_size = candidate_input.shape[1]
        self.logger.debug(f"Candidate input shape: {candidate_input.shape}, Input size: {input_size}")

        # Create and train candidate units
        candidates = []
        for candidate_id in range(self.candidate_pool_size):
            candidate = CandidateUnit(input_size, self.activation_fn)
            self.logger.debug(f"Training candidate {candidate_id} with input size: {input_size}.\n{candidate}")
            # Train the candidate unit
            correlation = candidate.train(
                candidate_input,
                residual_error, 
                learning_rate=self.learning_rate, 
                epochs=self.candidate_epochs,
            )
            candidates.append(candidate)
            self.logger.debug(f"Candidate: {candidate_id + 1} trained with correlation: {correlation:.6f}")

        # Sort candidates by correlation
        candidates.sort(key=lambda c: c.correlation, reverse=True)
        return candidates


    #################################################################################################################################################################################################
    # Public Method to add a new hidden unit based on the correlation
    def add_unit(
        self,
        candidate: CandidateUnit,
        x: torch.Tensor,
    ) -> None:
        """
        Add a new hidden unit to the network.

        Args:
            candidate: Candidate unit to add
            x: Input tensor to calculate the unit's output
        """
        # Prepare input for the new unit (includes outputs from existing hidden units)
        hidden_outputs = []
        for unit in self.hidden_units:
            unit_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
            unit_output = unit['activation_fn'](torch.sum(unit_input * unit['weights'], dim=1) + unit['bias']).unsqueeze(1)
            self.logger.debug(f"Unit output shape: {unit_output.shape}, Unit output: {unit_output}")
            hidden_outputs.append(unit_output)
        self.logger.debug(f"Hidden outputs shape: {[h.shape for h in hidden_outputs]}")

        candidate_input = torch.cat([x] + hidden_outputs, dim=1) if hidden_outputs else x
        self.logger.debug(f"Candidate input shape: {candidate_input.shape}, Input size: {candidate_input.shape[1]}, Candidate Input:\n{candidate_input}")

        # Create a new hidden unit
        new_unit = {
            'weights': candidate.weights.clone().detach(),
            'bias': candidate.bias.clone().detach(),
            'activation_fn': self.activation_fn,
            'correlation': candidate.correlation
        }
        self.logger.debug(f"Adding new hidden unit with weights: {new_unit['weights']}, bias: {new_unit['bias']}, correlation: {new_unit['correlation']:.6f}, Unit: {new_unit}")
        # Add the new unit to the network
        self.hidden_units.append(new_unit)
        self.logger.debug(f"Current number of hidden units: {len(self.hidden_units)}, Hidden units: {self.hidden_units}")

        # Update output layer weights to include the new unit
        old_output_weights = self.output_weights.clone().detach()
        self.logger.debug(f"Old output weights shape: {old_output_weights.shape}, Weights: {old_output_weights}")
        old_output_bias = self.output_bias.clone().detach()
        self.logger.debug(f"Old output bias shape: {old_output_bias.shape}, Bias: {old_output_bias}")

        # Calculate the output of the new unit
        unit_output = self.activation_fn(torch.sum(candidate_input * new_unit['weights'], dim=1) + new_unit['bias']).unsqueeze(1)
        self.logger.debug(f"New unit output shape: {unit_output.shape}, New unit output: {unit_output}")

        # Create new output weights with an additional row for the new unit
        if hidden_outputs:
            new_input_size = x.shape[1] + len(hidden_outputs) + 1
        else:
            new_input_size = x.shape[1] + 1
        self.logger.debug(f"New input size for output weights: {new_input_size}, Old input size: {old_output_weights.shape[0]}")

        # Ensure new weights have requires_grad=True
        self.output_weights = torch.randn(new_input_size, self.output_size, requires_grad=True) * 0.1
        self.logger.debug(f"New output weights shape: {self.output_weights.shape}, Weights: {self.output_weights}")

        # Copy old weights
        if hidden_outputs:
            input_size_before = x.shape[1] + len(hidden_outputs)
        else:
            input_size_before = x.shape[1]
        self.logger.debug(f"Input size before adding new unit: {input_size_before}")

        self.output_weights[:input_size_before, :] = old_output_weights
        self.logger.debug(f"Updated output weights after copying old weights: {self.output_weights}")
        self.output_bias = old_output_bias
        self.logger.debug(f"Updated output bias after copying old bias: {self.output_bias}")
        self.logger.info(f"Added hidden unit with correlation: {candidate.correlation:.6f}")
        self.history['hidden_units_added'].append({
            'correlation': candidate.correlation,
            'weights': candidate.weights.clone().detach().numpy(),
            'bias': candidate.bias.clone().detach().numpy()
        })
        self.logger.info(f"Current number of hidden units: {len(self.hidden_units)}")
        self.logger.debug(f"Updated history with new hidden unit:\n{self.history['hidden_units_added'][-1]}\nHistory\n{self.history}")


    #################################################################################################################################################################################################
    # Public Method to calculate the residual error of the network
    def calculate_residual_error(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the residual error of the network.

        Args:
            x: Input tensor
            y: Target tensor

        Returns:
            Residual error tensor
        """
        with torch.no_grad():
            output = self.forward(x)
            residual = y - output
        return residual


    #################################################################################################################################################################################################
    # Public Method to calculate the accuracy of the network
    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        max_epochs: int = 1000,
        early_stopping: bool = True
    ) -> Dict[str, List]:
        """
        Train the network using the cascade correlation algorithm.

        Args:
            x_train: Training input tensor
            y_train: Training target tensor
            x_val: Validation input tensor
            y_val: Validation target tensor
            max_epochs: Maximum number of epochs to train
            early_stopping: Whether to use early stopping

        Returns:
            Training history
        """
        # Initial training of the output layer
        self.logger.info("Initial training of output layer")
        train_loss = self.train_output_layer(x_train, y_train, self.output_epochs)
        self.history['train_loss'].append(train_loss)

        if x_val is not None and y_val is not None:
            with torch.no_grad():
                value_output = self.forward(x_val)
                value_loss = nn.MSELoss()(value_output, y_val).item()
            self.history['value_loss'].append(value_loss)
            self.logger.info(f"Initial - Train Loss: {train_loss:.6f}, Val Loss: {value_loss:.6f}")
        else:
            self.logger.info(f"Initial - Train Loss: {train_loss:.6f}")

        # Calculate initial accuracy
        train_accuracy = self.calculate_accuracy(x_train, y_train)
        self.history['train_accuracy'].append(train_accuracy)

        if x_val is not None and y_val is not None:
            value_accuracy = self.calculate_accuracy(x_val, y_val)
            self.history['value_accuracy'].append(value_accuracy)
            self.logger.info(f"Initial - Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {value_accuracy:.4f}")
        else:
            self.logger.info(f"Initial - Train Accuracy: {train_accuracy:.4f}")

        # Main training loop
        patience_counter = 0
        best_value_loss = float('inf') if x_val is not None else None
        self.logger.info(f"Starting main training loop with max epochs: {max_epochs}, early stopping: {early_stopping}")
        self.grow_network(
            x_train=x_train,
            y_train=y_train,
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            patience_counter=patience_counter,
            candidate=CandidateUnit(self.input_size, self.activation_fn),
            best_value_loss=best_value_loss,
            x_val=x_val,
            y_val=y_val,
        )
        self.history['hidden_units_added'].append({'correlation': 0.0, 'weights': [], 'bias': []})
        return self.history


    #################################################################################################################################################################################################
    # Public Method to grow the network by adding hidden units
    # This method is the core of the Cascade Correlation algorithm
    # It iteratively adds hidden units based on the residual error until stopping criteria are met
    def grow_network(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        max_epochs: int = 1000,
        early_stopping: bool = True,
        patience_counter: int = 0,
        candidate: CandidateUnit = None,
        best_value_loss: float = float('inf'),
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
    ) -> (bool, int, float, torch.Tensor, float, float):
        """
        Grow the network by adding hidden units until stopping criteria are met.
        Args:
            x_train: Training input tensor
            y_train: Training target tensor
            max_epochs: Maximum number of epochs to train
            early_stopping: Whether to use early stopping
            patience_counter: Counter for early stopping patience
            candidate: Candidate unit for training
            best_value_loss: Best validation loss seen so far
            x_val: Validation input tensor
            y_val: Validation target tensor
        """
        for epoch in range(max_epochs):
            # Calculate residual error
            residual_error = self.calculate_residual_error(x_train, y_train)
            self.logger.debug(f"Epoch {epoch}, Residual Error: {residual_error.mean().item():.6f}")

            # Train candidate units
            candidates = self.train_candidates(x_train, residual_error)
            self.logger.debug(f"Epoch {epoch}, Number of Candidates: {len(candidates)}\n{candidates}")
            best_candidate = candidates[0]
            self.logger.debug(f"Best Candidate: {best_candidate})")
            self.logger.info(f"Best Candidate Correlation: {best_candidate.correlation:.6f}, Weights: {best_candidate.weights}, Bias: {best_candidate.bias}")

            # Check if best candidate meets correlation threshold
            if best_candidate.correlation < self.correlation_threshold:
                self.logger.info(f"No candidate met correlation threshold: {self.correlation_threshold}, Best Correlation Achieved: {best_candidate.correlation:.6f}")
                break
            self.logger.info(f"Best Candidate: {best_candidate.correlation}, Met correlation threshold: {self.correlation_threshold}")

            # Add best candidate to the network
            self.add_unit(best_candidate, x_train)
            self.logger.info("Added best candidate to the network")

            # Retrain output layer
            train_loss = self.train_output_layer(x_train, y_train, self.output_epochs)
            self.logger.info(f"Epoch {epoch}, Train Loss: {train_loss:.6f}")
            self.history['train_loss'].append(train_loss)
            self.logger.debug(f"For Current Epoch: {epoch}, Post-Trained History:\n{self.history}")

            # Calculate accuracy
            train_accuracy = self.calculate_accuracy(x_train, y_train)
            self.logger.debug(f"For Current Epoch {epoch}, Train Accuracy: {train_accuracy:.4f}")
            self.history['train_accuracy'].append(train_accuracy)
            self.logger.debug(f"For Current Epoch {epoch}, Post-Train Accuracy History:\n{self.history}")

            # Validation
            (early_stop, patience_counter, best_value_loss, value_output, value_loss, value_accuracy) = self.validate_training(
                epoch=epoch,
                max_epochs=max_epochs,
                patience_counter=patience_counter,
                early_stopping=early_stopping,
                train_accuracy=train_accuracy,
                train_loss=train_loss,
                best_value_loss=best_value_loss,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            )
            self.logger.debug(f"Epoch {epoch}, Early Stop: {early_stop}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}, Value Output: {value_output} Value Loss: {value_loss:.6f}, Value Accuracy: {value_accuracy:.4f}")
            if early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}, Early stop: {early_stop}")
        self.logger.info(f"Finished training after {epoch + 1} epochs. Total hidden units: {len(self.hidden_units)}")
        return (early_stop, patience_counter, best_value_loss, value_output, value_loss, value_accuracy)


    #################################################################################################################################################################################################
    # Public Method to validate the training process
    def validate_training(
        self,
        epoch: int = 0,
        max_epochs: int = 0,
        patience_counter: int = 0,
        early_stopping: bool = True,
        train_accuracy: float = 0.0,
        train_loss: float = float('inf'),
        best_value_loss: float = 9999999.9,
        x_train: torch.Tensor = None,
        y_train: torch.Tensor = None,
        x_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
    ) -> ():
        early_stop_flag = False
        value_output = 0
        value_loss = float('inf')
        value_accuracy = 0.0
        best_value_loss = best_value_loss if best_value_loss is not None else 9999999.9
        # self.logger.debug(f"Epoch {epoch}, Max Epochs: {max_epochs}, Early Stopping: {early_stopping}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}")
        self.logger.debug(f"Epoch {epoch}, Max Epochs: {max_epochs}, Early Stopping: {early_stopping}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}, Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}")
        # self.logger.debug(f"Epoch {epoch}, Max Epochs: {max_epochs}, Early Stopping: {early_stopping}, Patience Counter: {patience_counter}, Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}")
        # self.logger.debug(f"Best Value Loss: {best_value_loss}")

        self.logger.debug(f"Validating training: X Train: {x_train}, Y Train: {y_train}, X Val: {x_val}, Y Val: {y_val}")
        if x_val is not None and y_val is not None:
            # Validate the model on the validation set
            with torch.no_grad():
                value_output = self.forward(x_val)
                value_loss = nn.MSELoss()(value_output, y_val).item()
            self.history['value_loss'].append(value_loss)
            # Calculate validation accuracy
            value_accuracy = self.calculate_accuracy(x_val, y_val)
            self.history['value_accuracy'].append(value_accuracy)
            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_loss:.6f}, Val Loss: {value_loss:.6f}, "
                f"Train Acc: {train_accuracy:.4f}, Val Acc: {value_accuracy:.4f}, "
                f"Units: {len(self.hidden_units)}"
            )
            # Check for early stopping conditions
            (early_stop, patience_counter, best_value_loss) = self.evaluate_early_stopping(
                epoch=epoch,
                max_epochs=max_epochs,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                early_stopping=early_stopping,
                value_loss=value_loss,
                best_value_loss=best_value_loss,
                patience_counter=patience_counter,
            )
            self.logger.debug(f"Early Stopping: {early_stopping}")
            self.logger.debug(f"Early Stop: {early_stop}")
            self.logger.debug(f"Epoch: {epoch}, Early Stop: {early_stop}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}")
            # early_stop_flag = True if early_stopping and early_stop else False
            early_stop_flag = early_stopping and early_stop
            self.logger.info(f"Stop Training Early: {early_stop} and Early Stopping: {early_stopping}: {early_stopping and early_stop}")
            self.logger.info(f"Early Stopping: {early_stop_flag}, Patience Counter: {patience_counter}, Best Val Loss: {best_value_loss:.6f}")
            self.logger.debug(f"Value Output: {value_output}, Value Loss: {value_loss:.6f}, Value Accuracy: {value_accuracy:.4f} ")
        self.logger.debug(f"Epoch {epoch}, Early Stop: {early_stop_flag}, Patience Counter: {patience_counter}, Best Value Loss: {best_value_loss:.6f}, Value Output: {value_output}, Value Loss: {value_loss:.6f}, Value Accuracy: {value_accuracy:.4f}")
        return (early_stop_flag, patience_counter, best_value_loss, value_output, value_loss, value_accuracy,)


    #################################################################################################################################################################################################
    # Public Method to evaluate early stopping conditions
    # This method checks if the training should stop early based on validation loss, patience, and other criteria
    def evaluate_early_stopping(
        self,
        epoch: int = 0,
        max_epochs: int = 0,
        train_loss: float = float('inf'),
        train_accuracy: float = 0.0,
        early_stopping: bool = True,
        value_loss: float = float('inf'),
        best_value_loss: float = float('inf'),
        patience_counter: int = 0,
    ) -> (bool, int, float):
        """
        Check if early stopping should be triggered.
        Args:
            value_loss: Validation loss
            best_value_loss: Best validation loss
            patience_counter: Patience counter
            max_epochs: Maximum number of epochs
            early_stopping: Whether to use early stopping
            max_hidden_units: Maximum number of hidden units
        Returns:
            bool: Whether early stopping should be triggered
            int: Updated patience counter
            float: Updated best validation loss
        """
        # Early stopping
        if early_stopping:
            # Check if we've reached the end of our patience
            (patience_exhausted, patience_counter, best_value_loss) = self.check_patience(
                patience_counter=patience_counter,
                value_loss=value_loss,
                best_value_loss=best_value_loss,
            )
            self.logger.info(f"Epoch {epoch} - Patience Counter: {patience_counter}, Value Loss: {value_loss}, Best Val Loss: {best_value_loss:.6f}")
            if patience_exhausted:
                self.logger.info(f"Patience Exhausted: {patience_exhausted}, Early stopping triggered after {epoch} epochs")
            else:
                self.logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.6f}, " f"Train Acc: {train_accuracy:.4f}, Units: {len(self.hidden_units)}")
            # Check if we've reached the maximum number of hidden units
            if (max_units_reached := self.check_hidden_units_max()):
                self.logger.info(f"Reached maximum number of hidden units: {max_units_reached}, stopping training")
            # Check if we've achieved perfect accuracy
            if (train_accuracy_reached := self.check_training_accuracy(
                train_accuracy=train_accuracy,
                accuracy_target=0.999,
            )):
                self.logger.info(f"Training accuracy reached target: {train_accuracy:.4f} >= 0.999")
        early_stop = (early_stopping and (train_accuracy_reached or max_units_reached or patience_exhausted))
        self.logger.info(f"Early Stopping: {early_stop}, Patience Counter: {patience_counter}, Best Val Loss: {best_value_loss:.6f}")
        return (early_stop, patience_counter, best_value_loss)


    #################################################################################################################################################################################################
    # Public Method to check patience limit
    # This method checks if the patience limit is reached based on validation loss
    def check_patience(
        self,
        patience_counter: int = 0,
        value_loss: float = float('inf'),
        best_value_loss: float = float('inf'),
    ) -> (bool, int, float):
        """
        Check if patience limit is reached.
        Args:
            value_loss: Validation loss
            best_value_loss: Best validation loss
            patience_counter: Patience counter
        Returns:
            bool: Whether patience limit is reached
            int: Updated patience counter
            float: Best validation loss
        """
        # Check if validation loss improved
        if value_loss < best_value_loss:
            best_value_loss = value_loss
            patience_counter = 0
        else:
            patience_counter += 1
        self.logger.info(f"Patience counter: {patience_counter}, Best Validation Loss: {best_value_loss:.6f}")
        # Check if patience limit is reached
        if (patience_exhausted := (patience_counter >= self.patience)):
            self.logger.info(f"Patience limit reached: {patience_counter} >= {self.patience}")
        return (patience_exhausted, patience_counter, best_value_loss)


    #################################################################################################################################################################################################
    # Public Methods to check conditions for training
    def check_hidden_units_max(self) -> bool:
        """
        Check if we've reached max hidden units

        Returns:
            bool: Whether we've reached max hidden units
        """
        # Check if we've reached max hidden units
        max_units_reached = len(self.hidden_units) >= self.max_hidden_units
        self.logger.info(f"Current hidden units: {max_units_reached}, Max allowed: {self.max_hidden_units}")
        if max_units_reached:
            self.logger.info(f"Reached maximum number of hidden units: {self.max_hidden_units}")
        return max_units_reached


    #################################################################################################################################################################################################
    # Public Method to check if training accuracy has reached the target
    # This method checks if the training accuracy has reached the target accuracy
    def check_training_accuracy(
        self,
        train_accuracy: float = 0.0,
        accuracy_target: float = 0.999,
    ) -> bool:
        """
        Check if training accuracy has reached the target.

        Args:
            train_accuracy: Current training accuracy
            accuracy_target: Target accuracy to reach

        Returns:
            bool: Whether target accuracy has been reached
        """
        if (train_accuracy_reached := (train_accuracy >= accuracy_target)):
            self.logger.info(f"Reached target training accuracy: {train_accuracy:.4f} >= {accuracy_target:.4f}")
        return train_accuracy_reached


    #################################################################################################################################################################################################
    # Public Method to calculate classification accuracy
    # This method calculates the classification accuracy of the network
    # It compares the predicted output with the target output
    def calculate_accuracy(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        """
        Calculate classification accuracy.

        Args:
            x: Input tensor
            y: Target tensor

        Returns:
            Classification accuracy
        """
        self.logger.debug(f"Calculating accuracy for input shape: {x.shape}, target shape: {y.shape}")
        with torch.no_grad():
            output = self.forward(x)
            self.logger.debug(f"Output shape: {output.shape}, Output: {output}")
            predicted = torch.argmax(output, dim=1)
            self.logger.debug(f"Predicted shape: {predicted.shape}, Predicted: {predicted}")
            target = torch.argmax(y, dim=1)
            self.logger.debug(f"Target shape: {target.shape}, Target: {target}")
            correct = (predicted == target).sum().item()
            self.logger.debug(f"Number of correct predictions: {correct}, Total samples: {len(y)}")
            accuracy = correct / len(y)
            self.logger.info(f"Calculated accuracy: {accuracy:.4f}")
        return accuracy


    #################################################################################################################################################################################################
    # Public Method to make predictions
    # This method uses the forward method to get the output of the network
    # It is used to make predictions on new data
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the network.

        Args:
            x: Input tensor

        Returns:
            Predicted output
        """
        with torch.no_grad():
            return self.forward(x)


    #################################################################################################################################################################################################
    # Public Method to predict class labels
    # This method predicts the class labels for the input tensor
    # It uses the forward method to get the output and then applies argmax to get the class labels
    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Args:
            x: Input tensor

        Returns:
            Predicted class labels
        """
        with torch.no_grad():
            output = self.forward(x)
            return torch.argmax(output, dim=1)


    #################################################################################################################################################################################################
    # Public Method to plot the decision boundary of the network
    # This method visualizes the decision boundary of the network in a 2D space
    # It uses matplotlib to create a contour plot of the decision boundary
    def plot_decision_boundary(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        title: str = "Decision Boundary",
    ):
        """
        Plot the decision boundary of the network.

        Args:
            x: Input tensor
            y: Target tensor
            title: Plot title
        """
        # Convert to numpy for plotting
        x_np = x.numpy()
        y_np = torch.argmax(y, dim=1).numpy()

        # Create a mesh grid
        h = 0.02  # step size in the mesh
        x_min, x_max = x_np[:, 0].min() - 1, x_np[:, 0].max() + 1
        y_min, y_max = x_np[:, 1].min() - 1, x_np[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict class labels for each point in the mesh
        Z = self.predict_classes(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).numpy()
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8)

        # Plot the training points
        for i in range(len(np.unique(y_np))):
            plt.scatter(x_np[y_np == i, 0], x_np[y_np == i, 1], label=f'Class {i}')
        self.plot_headings(plot=plt, title=title, x_label="X1", y_label="Y1")
        # plt.title("Decision Boundary")
        # plt.xlabel("X1")
        # plt.ylabel("X2")
        plt.show()


    #################################################################################################################################################################################################
    # Public Method to plot the training history of the network
    # This method visualizes the training history including loss, accuracy, number of hidden units, and correlation of added units
    # It uses matplotlib to create subplots for each metric
    def plot_training_history(self):
        """
        Plot the training history of the network.
        """
        plt.figure(figsize=(15, 10))

        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'value_loss' in self.history and self.history['value_loss']:
            plt.plot(self.history['value_loss'], label='Validation Loss')
        self.plot_headings(plot=plt, title="Loss During Training", x_label="Epochs", y_label="Loss")
        # plt.title('Loss During Training')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()

        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.history['train_accuracy'], label='Train Accuracy')
        if 'value_accuracy' in self.history and self.history['value_accuracy']:
            plt.plot(self.history['value_accuracy'], label='Validation Accuracy')
        self.plot_headings(plot=plt, title="Accuracy During Training", x_label="Epochs", y_label="Accuracy")
        # plt.title('Accuracy During Training')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()

        # Plot number of hidden units
        plt.subplot(2, 2, 3)
        plt.plot(range(len(self.history['hidden_units_added']) + 1), [0] + [i+1 for i in range(len(self.history['hidden_units_added']))])
        self.plot_headings(plot=plt, title="Number of Hidden Units", x_label="Epochs", y_label="Number of Units", legend=False)
        # plt.title('Number of Hidden Units')
        # plt.xlabel('Epochs')
        # plt.ylabel('Number of Units')

        # Plot correlation of added units
        if self.history['hidden_units_added']:
            plt.subplot(2, 2, 4)
            correlations = [unit['correlation'] for unit in self.history['hidden_units_added']]
            plt.plot(correlations)
            self.plot_headings(plot=plt, title="Correlation of Added Units", x_label="Unit Number", y_label="Correlation", legend=False)
            # plt.title('Correlation of Added Units')
            # plt.xlabel('Unit Number')
            # plt.ylabel('Correlation')

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()


    #################################################################################################################################################################################################
    # Public Method to set headings for the plot
    # This method sets the title, x-label, y-label, and legend for the plot
    def plot_headings(
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
        """
        self.logger.debug(f"Setting plot headings: Title: {title}, X Label: {x_label}, Y Label: {y_label}, Legend: {legend}")
        self.logger.debug(f"Plot type: {type(plot)}, Plot: {plot}")

        if plot is None:
            plot = plt
        if not isinstance(plot, type(matplotlib.pyplot)):
            raise ValueError("plot must be a matplotlib.pyplot object")
        if not isinstance(title, str):
            raise ValueError("title must be a string")
        if not isinstance(x_label, str):
            raise ValueError("x_label must be a string")
        if not isinstance(y_label, str):
            raise ValueError("y_label must be a string")
        if not hasattr(plot, 'title') or not hasattr(plot, 'xlabel') or not hasattr(plot, 'ylabel'):
            raise ValueError("plot must have title, xlabel, and ylabel methods")
        plot.title(title)
        plot.xlabel(x_label)
        plot.ylabel(y_label)
        if legend:
            plot.legend()


    #################################################################################################################################################################################################
    # Public Method to print a summary of the network architecture
    # This method prints the input size, output size, number of hidden units, and training parameters
    # It also prints the details of each hidden unit including its weights, bias, and correlation
    def summary(self):
        """
        Print a summary of the network architecture.
        """
        print("Cascade Correlation Network Summary:")
        print(f"Input size: {self.input_size}")
        print(f"Output size: {self.output_size}")
        print(f"Number of hidden units: {len(self.hidden_units)}")

        if self.hidden_units:
            print("\nHidden Units:")
            for i, unit in enumerate(self.hidden_units):
                print(f"  Unit {i+1}:")
                print(f"    Input size: {len(unit['weights'])}")
                print(f"    Correlation: {unit['correlation']:.6f}")

        print("\nTraining Parameters:")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Candidate pool size: {self.candidate_pool_size}")
        print(f"  Correlation threshold: {self.correlation_threshold}")

        if self.history['train_accuracy']:
            print(f"\nFinal training accuracy: {self.history['train_accuracy'][-1]:.6f}")
        if 'value_accuracy' in self.history and self.history['value_accuracy']:
            print(f"Final validation accuracy: {self.history['value_accuracy'][-1]:.6f}")
