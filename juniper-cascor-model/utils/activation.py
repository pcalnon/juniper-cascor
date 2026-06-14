#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     activation.py
# Author:        Paul Calnon
#
# Date Created:  2026-04-03
# Last Modified: 2026-04-03
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Shared picklable activation function wrapper for multiprocessing support.
#    Extracted from cascade_correlation.py and candidate_unit.py to eliminate
#    duplication and divergence risk (CASCOR-P1-003).
#
#####################################################################################################################################################################################################

import torch


#####################################################################################################################################################################################################
# Picklable Activation Function Wrapper
# CASCOR-P1-003: Multiprocessing Pickling Error Fix
# This class replaces the local function 'wrapped_activation', which cannot be pickled for multiprocessing.
# The local function defined inside _init_activation_with_derivative() created a closure that Python's
# pickle module cannot serialize, causing multiprocessing workers to fail when sending results back.
#####################################################################################################################################################################################################
class ActivationWithDerivative:
    """
    Picklable wrapper for activation functions that also provides derivatives.

    This class solves the multiprocessing pickling issue where local functions
    cannot be serialized. It stores the activation function type by name and
    reconstructs the function on unpickling.

    Supports: All standard PyTorch activation functions (tanh, sigmoid, relu, etc.)
    """

    # Mapping of activation names to functions for reconstruction after unpickling
    ACTIVATION_MAP = {
        "identity": torch.nn.Identity(),
        "elu": torch.nn.functional.elu,
        "hardshrink": torch.nn.functional.hardshrink,
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "softmax": torch.nn.Softmax(dim=1),
        "tanh": torch.tanh,
        "Identity": torch.nn.Identity(),
        "ELU": torch.nn.ELU(),
        "Hardshrink": torch.nn.Hardshrink(),
        "Hardsigmoid": torch.nn.Hardsigmoid(),
        "Hardtanh": torch.nn.Hardtanh(),
        "Hardswish": torch.nn.Hardswish(),
        "LeakyReLU": torch.nn.LeakyReLU(),
        "LogSigmoid": torch.nn.LogSigmoid(),
        "PReLU": torch.nn.PReLU(),
        "ReLU": torch.nn.ReLU(),
        "ReLU6": torch.nn.ReLU6(),
        "RReLU": torch.nn.RReLU(),
        "SELU": torch.nn.SELU(),
        "CELU": torch.nn.CELU(),
        "GELU": torch.nn.GELU(),
        "Sigmoid": torch.nn.Sigmoid(),
        "SiLU": torch.nn.SiLU(),
        "Mish": torch.nn.Mish(),
        "Softmax": torch.nn.Softmax(dim=1),
        "Softplus": torch.nn.Softplus(),
        "Softshrink": torch.nn.Softshrink(),
        "Softsign": torch.nn.Softsign(),
        "Tanh": torch.nn.Tanh(),
        "Tanhshrink": torch.nn.Tanhshrink(),
        "Threshold": torch.nn.Threshold(0.1, 0.0),  # Default threshold=0.1, value=0.0
        "GLU": torch.nn.GLU(),
    }

    def __init__(self, activation_fn):
        """
        Initialize with an activation function.

        Args:
            activation_fn: A PyTorch activation function (e.g., torch.tanh, torch.nn.Tanh())
                or the legacy (activation, derivative) tuple returned by juniper-cascor-worker.
        """
        self.activation_fn = self._normalize_activation_fn(activation_fn)
        self._activation_name = self._get_activation_name(activation_fn)

    def _normalize_activation_fn(self, activation_fn):
        """Accept the worker's legacy (activation, derivative) tuple without keeping lambdas."""
        if isinstance(activation_fn, tuple):
            if not activation_fn or not callable(activation_fn[0]):
                raise TypeError("activation tuple must start with a callable activation function")
            return activation_fn[0]
        return activation_fn

    def _get_activation_name(self, activation_fn) -> str:
        """
        Extract a string name from the activation function for serialization.

        Args:
            activation_fn: The activation function to get the name from

        Returns:
            String name of the activation function
        """
        activation_fn = self._normalize_activation_fn(activation_fn)
        if hasattr(activation_fn, "__name__"):
            return activation_fn.__name__
        elif hasattr(activation_fn, "__class__"):
            return activation_fn.__class__.__name__
        else:
            return str(activation_fn)

    def _apply_activation(self, x):
        """
        Apply activation with compatibility guards for known module edge cases.

        Softmax modules are often configured with dim=1, but this project invokes
        activation wrappers on 1-D pre-activation vectors in multiple training paths.
        Guard invalid dimensions to avoid runtime crashes during forward/derivative use.
        """
        if self._activation_name.lower() == "softmax":
            if x.dim() == 0:
                return torch.ones_like(x)
            configured_dim = getattr(self.activation_fn, "dim", -1)
            dim = configured_dim if isinstance(configured_dim, int) else -1
            if dim < -x.dim() or dim >= x.dim():
                dim = -1
            return torch.softmax(x, dim=dim)
        return self.activation_fn(x)

    def __call__(self, x, derivative: bool = False):
        """
        Apply activation function or compute its derivative.

        Args:
            x: Input tensor
            derivative: If True, compute the derivative instead of the activation

        Returns:
            Activation output or derivative value
        """
        if not derivative:
            return self._apply_activation(x)
        name = self._activation_name.lower()
        if name == "tanh":
            return 1.0 - torch.tanh(x) ** 2
        elif name == "sigmoid":
            y = torch.sigmoid(x)
            return y * (1.0 - y)
        elif name == "relu":
            return (x > 0).float()
        else:
            # Numerical approximation for other activation functions
            eps = 1e-6
            return (self._apply_activation(x + eps) - self._apply_activation(x - eps)) / (2 * eps)

    def __getstate__(self):
        """Serialize by storing activation name instead of function (for pickle/multiprocessing)."""
        return {"_activation_name": self._activation_name}

    def __setstate__(self, state):
        """Reconstruct activation function from name after unpickling."""
        self._activation_name = state["_activation_name"]
        activation = self.ACTIVATION_MAP.get(self._activation_name) or self.ACTIVATION_MAP.get(self._activation_name.lower())
        if activation is None:
            raise ValueError(f"Unrecognized activation function name during deserialization: '{self._activation_name}'. Known activations: {list(self.ACTIVATION_MAP.keys())}")
        self.activation_fn = activation

    def __repr__(self):
        """String representation for debugging."""
        return f"ActivationWithDerivative({self._activation_name})"
