#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     cascade_correlation_exceptions.py
# Author:        Paul Calnon
# Version:       0.3.2 (0.7.3)
#
# Date:          2025-09-26
# Last Modified: 2025-09-26
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This file contains the custom exception classes for the Cascade Correlation Neural Network.
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


#####################################################################################################################################################################################################
# Define Custom Exceptions for Cascade Correlation Network
#####################################################################################################################################################################################################
class CascadeCorrelationError(Exception):
    """Base exception for Cascade Correlation Network errors."""
    pass


class NetworkInitializationError(CascadeCorrelationError):
    """Raised when network initialization fails."""
    pass


class TrainingError(CascadeCorrelationError):
    """Raised when training encounters a critical error."""
    pass


class ValidationError(CascadeCorrelationError):
    """Raised when input validation fails."""
    pass


class ConfigurationError(CascadeCorrelationError):
    """Raised when configuration parameters are invalid."""
    pass

