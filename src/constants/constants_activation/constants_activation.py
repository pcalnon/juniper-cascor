#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     constants_activation.py
# Author:        Paul Calnon
# Version:       0.3.1 (0.7.3)
#
# Date Created:  2025-09-14
# Last Modified: 2026-01-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This file contains constants used in the Cascade Correlation Neural Network implementation.
#
#####################################################################################################################################################################################################
# Notes:
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
# import numpy as np
import torch


#####################################################################################################################################################################################################
# Define the constants for the Cascade Correlation Network Model, Activation Function
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_IDENTITY = torch.nn.Identity()
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_TANH = torch.nn.Tanh()                   # This value was used with 100% run
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SIGMOID = torch.nn.Sigmoid()
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_RELU = torch.nn.ReLU()
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_LEAKY_RELU = torch.nn.LeakyReLU()
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_ELU = torch.nn.ELU()
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SELU = torch.nn.SELU()
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_GELU = torch.nn.GELU()
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SOFTMAX = torch.nn.Softmax(dim=1)
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SOFTPLUS = torch.nn.Softplus()
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_HARDTANH = torch.nn.Hardtanh()
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SOFTSHRINK = torch.nn.Softshrink()
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_TANHSHRINK = torch.nn.Tanhshrink()
_PROJECT_MODEL_ACTIVATION_FUNCTION_TANH = torch.tanh                      # This is the default value used in the original implementation.
_PROJECT_MODEL_ACTIVATION_FUNCTION_SIGMOID = torch.sigmoid
_PROJECT_MODEL_ACTIVATION_FUNCTION_RELU = torch.relu

_PROJECT_MODEL_ACTIVATION_FUNCTION = _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_TANH
_PROJECT_MODEL_ACTIVATION_FUNCTION_DEFAULT = _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_TANH


_PROJECT_MODEL_ACTIVATION_FUNCTIONS_LIST = [
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_IDENTITY,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_TANH,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SIGMOID,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_RELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_LEAKY_RELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_ELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_GELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SOFTMAX,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SOFTPLUS,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_HARDTANH,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_SOFTSHRINK,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_TANHSHRINK,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_TANH,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_SIGMOID,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_RELU,
]

_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_IDENTITY = "Identity"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_TANH = "Tanh"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SIGMOID = "Sigmoid"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_RELU = "ReLU"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_LEAKY_RELU = "LeakyReLU"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_ELU = "ELU"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SELU = "SELU"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_GELU = "GELU"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SOFTMAX = "Softmax"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SOFTPLUS = "Softplus"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_HARDTANH = "Hardtanh"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SOFTSHRINK = "Softshrink"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_TANHSHRINK = "Tanhshrink"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_TANH = "tanh"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_SIGMOID = "sigmoid"
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_RELU = "relu"

_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME = _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_TANH


_PROJECT_MODEL_ACTIVATION_FUNCTIONS_NAME_LIST = [
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_IDENTITY,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_TANH,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SIGMOID,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_RELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_LEAKY_RELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_ELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_GELU,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SOFTMAX,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SOFTPLUS,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_HARDTANH,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_SOFTSHRINK,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NN_TANHSHRINK,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_TANH,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_SIGMOID,
    _PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_RELU,
]

_PROJECT_MODEL_ACTIVATION_FUNCTIONS_DICT = dict(zip(_PROJECT_MODEL_ACTIVATION_FUNCTIONS_NAME_LIST, _PROJECT_MODEL_ACTIVATION_FUNCTIONS_LIST, strict=False))
