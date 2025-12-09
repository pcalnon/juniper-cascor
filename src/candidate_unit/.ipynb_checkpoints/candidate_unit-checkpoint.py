#!/usr/bin/env python3
#####################################################################################################################################################################################################
# Project:       Cascade Correlation Neural Network
# File Name:     candidate_unit.py
# Author:        Paul Calnon
# Version:       1.0.1
# Date:          2025-06-11
# Last Modified: 2025-06-11
# License:       MIT License
#
# Description:
#    This module implements a candidate unit for the Cascade Correlation Neural Network.
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

import torch
from typing import Optional
import logging

from constants.constants import (
    _CANDIDATE_UNIT_INPUT_SIZE,
    _CANDIDATE_UNIT_ACTIVATION_FUNCTION,
    _CANDIDATE_UNIT_LOGLEVEL_DEFAULT,
)

# # Define constants for the Cascade Correlation Network
# _CANDIDATE_UNIT_INPUT_SIZE = 2
# _CANDIDATE_UNIT_ACTIVATION_FUNCTION = torch.tanh
# _CANDIDATE_UNIT_LOGLEVEL_DEFAULT = logging.INFO


#####################################################################################################################################################################################################
class CandidateUnit:
    """
    Candidate Unit for Cascade Correlation Neural Network.
    This class represents a candidate unit in the Cascade Correlation Neural Network.
    It initializes the weights and bias, sets the activation function, and provides methods for forward pass and training.
    Attributes:
        weights: Weights of the candidate unit
        bias: Bias of the candidate unit
        activation_fn: Activation function used in the candidate unit
        correlation: Correlation between the candidate unit output and residual error
        logger: Logger for debugging and information messages
    Methods:
        __init__: Initializes the candidate unit with weights, bias, and activation function.
        forward: Performs a forward pass through the candidate unit.
        train: Trains the candidate unit to maximize correlation with residual error.
        _update_weights_and_bias: Updates weights and bias based on correlation with residual error.
        _final_correlation: Calculates final correlation after training.
        _single_output_correlation: Calculates correlation for single-output networks.
        _multi_output_correlation: Calculates correlation for multi-output networks.
        _calculate_correlation: Calculates correlation between output and residual error.
    Args:
        input_size: Number of inputs to this unit (default: 2)
        activation_function: Activation function to use (default: torch.tanh)
        logging_level: Logging level for the logger (default: logging.INFO)
    Example:
        candidate_unit = CandidateUnit(input_size=2, activation_function=torch.tanh, logging_level=logging.DEBUG)
        x = torch.randn(10, 2)  # Example input tensor
        residual_error = torch.randn(10, 1)  # Example residual error tensor
        correlation = candidate_unit.train(x, residual_error, learning_rate=0.1, epochs=100)
        print(f"Final correlation: {correlation}")
    """

    #################################################################################################################################################################################################
    # Initialize a candidate unit.
    # This method initializes the weights and bias of the candidate unit, sets the activation function, and configures the logger.
    def __init__(
        self,
        input_size: int = _CANDIDATE_UNIT_INPUT_SIZE,
        activation_function: torch = _CANDIDATE_UNIT_ACTIVATION_FUNCTION,
        logging_level = _CANDIDATE_UNIT_LOGLEVEL_DEFAULT,
    ):
        """
        Initialize a candidate unit.
        Args:
            input_size: Number of inputs to this unit
            activation_fn: Activation function to use
        """
        self.weights = torch.randn(input_size) * 0.1
        self.bias = torch.randn(1) * 0.1
        self.activation_fn = activation_function
        self.correlation = 0.0
        # Add logger
        self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.INFO)
        self.logger.setLevel(logging_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            formatter = logging.Formatter(fmt="[%(filename)s:%(lineno)d] (%(asctime)s) [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


    #################################################################################################################################################################################################
    # Calculate the correlation between the candidate unit output and the residual error.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the candidate unit.
        Args:
            x: Input tensor
        Returns:
            Output of the candidate unit
        """
        return self.activation_fn(torch.sum(x * self.weights, dim=1) + self.bias)


    #################################################################################################################################################################################################
    # Train the candidate unit to maximize correlation with residual error.
    # This method takes the input tensor, residual error, learning rate, and number of epochs.
    # It performs a forward pass, calculates the correlation with the residual error, and updates the weights and bias accordingly.
    # It handles both single-output and multi-output networks.
    # The method returns the final correlation value after training.
    def train(
        self,
        x: torch.Tensor,
        residual_error: torch.Tensor,
        learning_rate: float = 0.1,
        epochs: int = 100,
    ) -> float:
        """
        Train the candidate unit to maximize correlation with residual error.
        Args:
            x: Input tensor
            residual_error: Residual error from the network
            learning_rate: Learning rate for training
            epochs: Number of training epochs
        Returns:
            Final correlation value
        """
        # Debug shapes
        self.logger.debug(f"Input shape: {x.shape}, Residual error shape: {residual_error.shape}")
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(x)
            # Debug output shape
            self.logger.debug(f"For Epoch {epoch}: Output shape: {output.shape}")
            self.logger.debug(f"For Epoch {epoch}: Residual error shape: {len(residual_error.shape)}, Residual_error_shape: {residual_error.shape[1]}")
            # Ensure residual_error has the right shape for correlation calculation.  If residual_error is [batch_size, output_size], we need to handle it properly
            if len(residual_error.shape) > 1 and residual_error.shape[1] > 1:
                # For multi-output networks, we'll compute correlation with each output and use the maximum absolute correlation
                (correlation, correlations, norm_output, norm_error, numerator, denominator) = self._multi_output_correlation(residual_error=residual_error, output=output)
                self.logger.debug(f"For Epoch {epoch}: Multi-output correlations len: {len(correlations)}, Correlations: {correlations}")
                self.logger.debug(f"Correlations Shape:\n1.\t{correlations[0].shape}")
                self.logger.debug(f"Correlations Dims:\n1.\t{correlations[0].dim()}")
                self.logger.debug(f"Correlations Value:\n1.\t{correlations[0].item()}")
                self.logger.debug(f"Correlations Shape:\n2.\t{correlations[1].shape}")
                self.logger.debug(f"Correlations Dims:\n2.\t{correlations[1].dim()}")
                self.logger.debug(f"Correlations Value:\n2.\t{correlations[1].item()}")
                # Use the error component with the highest absolute correlation

                # best_corr_idx = max(range(len(correlations)), key=lambda i: abs(correlations[i][0]))
                # range(len(correlations)), key=lambda i: if correlations[i].dim() < 1 and (correlations[i].items()) is not None else abs(correlations[i][0])
                # b=[random.randint(1,1000) for i in range(a)]
                # new_list = [true_expr if conditional else false_expr for member in iterable]
                # new_list = [abs(correlations[i].items()) if correlations[i].dim() < 1 else abs(correlations[i][0]) for i in range(a)]
                # l2 = list(map(lambda v: v ** 2, l1))
                # range(len(correlations)), key=lambda i: correlations[i].dim() < 1 and correlations[i] = abs(correlations[i].items()) or abs(correlations[i][0])
                # best_corr_idx = max(range(len(correlations)), key=lambda i: abs(correlations[i][0]))
                # if (correlations[i].dim() < 1 and (correlations[i] := correlations[i].items()) is not None for i in range(len(correlations))):
                # best_corr_idx = max(range(len(correlations)), key=lambda i: abs(correlations[i][0]))
                best_corr_idx = max(range(len(correlations)), key=lambda i: abs(correlations[i].item()) if correlations[i].dim() < 1 else abs(correlations[i][0]))
                self.logger.debug(f"Correlation Array: \n{correlation}")
                self.logger.debug(f"Best correlation Index: {best_corr_idx}, Value: {correlations[best_corr_idx]}, List: {correlations}")
                self.logger.debug(f"Best correlations List length: {len(correlations)}, Value: {correlations[best_corr_idx].item()}")

                if isinstance(correlations[best_corr_idx], tuple):
                    # If the best correlation is a tuple, unpack it
                    self.logger.debug(f"Best correlation is a tuple: {correlations[best_corr_idx]}")
                    # TODO: Fix this.  correlations[best_corr_idx] is a single value (i.e., can't be iterated over).
                    (correlation, error_idx, norm_output, norm_error, numerator, denominator) = correlations[best_corr_idx]
                    self.logger.debug(f"Best correlation with output {error_idx}: {correlation}")
                else:
                    # If the best correlation is a single value, use it directly
                    self.logger.debug(f"Best correlation is a single value: {correlations[best_corr_idx]}")
                    correlation = correlations[best_corr_idx]
                    norm_output = output
                    norm_error = residual_error[:, best_corr_idx]
                    numerator = torch.sum(norm_output * norm_error)
                    denominator = torch.sqrt(torch.sum(norm_output**2) * torch.sum(norm_error**2) + 1e-8)
                    self.logger.debug(f"Best correlation with output {best_corr_idx}: {correlation}")
            else:
                # For single-output networks, compute correlation directly
                (correlation, norm_output, norm_error, numerator, denominator) = self._single_output_correlation(output=output, residual_error=residual_error)
                self.logger.debug(f"Single correlation with output {norm_error}: {correlation}")
            self.logger.debug(f"Normalized output:\n{norm_output}\nNormalized error:\n{norm_error}")
            self.logger.debug(f"Numerator: {numerator}, Denominator: {denominator}")
            self.logger.info(f"Correlation: {correlation}")
            (grad_corr, grad_output) = self._update_weights_and_bias(
                x=x,
                residual_error=residual_error,
                learning_rate=learning_rate,
                norm_output=norm_output,
                norm_error=norm_error,
                numerator=numerator,
                denominator=denominator,
            )
            self.logger.debug(f"For Epoch: {epoch}: Gradient of correlation: {grad_corr}")
            self.logger.debug(f"For Epoch: {epoch}: Gradient of output: {grad_output}")
        # Calculate final correlation
        output = self.forward(x)
        self.correlation = self._final_correlation(output=output, residual_error=residual_error)
        self.logger.info(f"For Final Epoch: {epoch}: Final correlation after training: {self.correlation}")
        return self.correlation


    #################################################################################################################################################################################################
    # Update weights and bias of the candidate unit based on correlation with residual error.
    # This method computes the gradient of the correlation with respect to the output and updates the weights and bias accordingly.
    # It handles both single-output and multi-output networks.
    # The method takes the input tensor, residual error, learning rate, and optionally normalized output and error tensors, as well as numerator and denominator for correlation.
    # It returns the gradients of correlation and output.
    def _update_weights_and_bias(
        self,
        x: torch.Tensor,
        residual_error: torch.Tensor,
        learning_rate: float = 0.1,
        norm_output: Optional[torch.Tensor] = None,
        norm_error: Optional[torch.Tensor] = None,
        numerator: Optional[float] = None,
        denominator: Optional[float] = None,
    ) -> None:
        """
        Update weights and bias of the candidate unit based on correlation with residual error.
        Args:
            x: Input tensor
            residual_error: Residual error from the network
            learning_rate: Learning rate for weight updates
            norm_output: Normalized output tensor (optional)
            norm_error: Normalized error tensor (optional)
            numerator: Numerator of the correlation (optional)
            denominator: Denominator of the correlation (optional)
        """
        # Gradient of correlation with respect to output
        grad_corr = norm_error / denominator - numerator * norm_output * torch.sum(norm_error**2) / (denominator**3)
        # Gradient of output with respect to weights
        grad_output = self.activation_fn(torch.sum(x * self.weights, dim=1) + self.bias, derivative=True)
        # Update weights
        for i in range(len(self.weights)):
            grad_w = torch.sum(grad_corr * grad_output * x[:, i])
            self.weights[i] += learning_rate * grad_w
        # Update bias
        grad_b = torch.sum(grad_corr * grad_output)
        self.bias += learning_rate * grad_b
        return (grad_corr, grad_output)


    #################################################################################################################################################################################################
    # Final correlation after training.
    # This method calculates the final correlation between the output of the candidate unit and the residual error from the network.
    # It handles both single-output and multi-output networks.
    def _final_correlation(
        self,
        output: torch.Tensor,
        residual_error: torch.Tensor
    ) -> float:
        """
        Calculate final correlation after training.
        Args:
            output: Output of the candidate unit
            residual_error: Residual error from the network
        Returns:
            Final correlation value
        """
        # Calculate correlation between output and residual error
        if len(residual_error.shape) > 1 and residual_error.shape[1] > 1:
            (correlation, _, _, _, _, _,) = self._multi_output_correlation(residual_error=residual_error, output=output)
        else:
            (correlation, _, _, _, _,) = self._single_output_correlation(output=output, residual_error=residual_error)
        self.logger.info(f"Final correlation: {correlation}")
        return correlation


    #################################################################################################################################################################################################
    # Calculate correlation for single-output and multi-output networks.
    # This method checks the shape of the residual error tensor to determine if it is single-output or multi-output.
    def _single_output_correlation(
        self,
        output: torch.Tensor = None,
        residual_error: torch.Tensor = None
    ) -> (
        float,
        torch.Tensor,
        torch.Tensor,
        float,
        float,
    ): 
        """
        Calculate correlation for single-output networks.
        Args:
            output: Output of the candidate unit
            residual_error: Residual error from the network
        Returns:
            Correlation between output and residual error, normalized output, normalized error, numerator, denominator
        """
        # Calculate correlation between output and this error component
        (correlation, norm_output, norm_error, numerator, denominator) = self._calculate_correlation(output=output, residual_error=residual_error)
        self.correlation = correlation
        self.logger.info(f"Single output correlation: {self.correlation}")
        self.logger.info(f"Numerator: {numerator}, Denominator: {denominator}")
        self.logger.info(f"Normalized output:\n{norm_output}\nNormalized error:\n{norm_error}")
        return (self.correlation, norm_output, norm_error, numerator, denominator)


    #################################################################################################################################################################################################
    # Calculate correlation for multi-output networks.
    # This method iterates over each output and calculates the correlation with the residual error.
    def _multi_output_correlation(
        self,
        residual_error: torch.Tensor = None,
        output: torch.Tensor = None
    ) -> (
        float,
        list,
        torch.Tensor,
        torch.Tensor,
        float,
        float,
    ):
        """
        Calculate correlation for multi-output networks.
        Args:
            residual_error: Residual error from the network
            output: Output of the candidate unit
        Returns:
            Max correlation, list of correlations for each output, normalized output, normalized error, numerator, denominator
        """
        # Debug shapes
        self.logger.debug(f"Output shape: {output.shape}, Residual error shape: {residual_error.shape}")
        # For multi-output networks, compute correlation with each output
        max_correlation = 0.0
        correlations = []
        for i in range(residual_error.shape[1]):
            error_i = residual_error[:, i]
            # Calculate correlation between output and this error component
            (correlation, norm_output, norm_error, numerator, denominator) = self._calculate_correlation(output=output, residual_error=error_i)
            if correlation > max_correlation:
                max_correlation = correlation
            correlations.append(correlation)
        self.logger.debug(f"Correlations: {correlations}")
        self.logger.debug(f"Max correlation: {max_correlation}")
        self.logger.debug(f"Numerator: {numerator}, Denominator: {denominator}")
        self.logger.debug(f"Normalized output:\n{norm_output}\nNormalized error:\n{norm_error}")
        self.correlation = max_correlation
        self.logger.info(f"Multi-output correlation: {self.correlation}")
        return (self.correlation, correlations, norm_output, norm_error, numerator, denominator)


    #################################################################################################################################################################################################
    # Calculate correlation between output and residual error.
    # This method computes the correlation coefficient between the output and the residual error.
    def _calculate_correlation(
        self,
        output: torch.Tensor,
        residual_error: torch.Tensor,
    ) -> float:
        """
        Calculate correlation between output and residual error.
        Args:
            output: Output tensor
            residual_error: Residual error tensor
        Returns:
            Correlation between output and residual error
        """
        # Debug shapes
        self.logger.debug(f"Output shape: {output.shape}, Residual error shape: {residual_error.shape}")
        # Calculate correlation between output and residual error
        output_mean = torch.mean(output)
        error_mean = torch.mean(residual_error)
        # Normalized output and error
        norm_output = output - output_mean
        norm_error = residual_error - error_mean
        # Correlation
        numerator = torch.sum(norm_output * norm_error)
        denominator = torch.sqrt(torch.sum(norm_output**2) * torch.sum(norm_error**2) + 1e-8)
        # correlation = numerator / denominator
        correlation = abs(numerator / denominator)
        self.logger.info(f"Numerator: {numerator}, Denominator: {denominator}")
        self.logger.info(f"Correlation: {correlation}")
        return (correlation, norm_output, norm_error, numerator, denominator)
