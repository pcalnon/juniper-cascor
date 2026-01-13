#!/usr/bin/env python3
#####################################################################################################################################################################################################
# Project:       Cascade Correlation Neural Network
# File Name:     constants.py
# Author:        Paul Calnon
# Version:       1.0.1
# Date Created:  2025-06-11
# Last Modified: 2026-01-12
# License:       MIT License
#
# Description:
#    This file contains constants used in the Cascade Correlation Neural Network implementation.
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

# import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from typing import List, Tuple, Optional, Dict, Any
import logging
# import random
# import math
# from cascade_correlation import CascadeCorrelationNetwork


#####################################################################################################################################################################################################
# Define the default logging level, name, and configuration

# _LOG_LEVEL_MAIN = "FATAL"
# _LOG_LEVEL_MAIN = "CRITICAL"
# _LOG_LEVEL_MAIN = "ERROR"
# _LOG_LEVEL_MAIN = "WARNING"
# _LOG_LEVEL_MAIN = "INFO"
_LOG_LEVEL_MAIN = "DEBUG"


#####################################################################################################################################################################################################
# Define constants for the Cascade Correlation Network, cascor.py file
_CASCOR_INPUT_SIZE = 2
_CASCOR_OUTPUT_SIZE = 2
_CASCOR_CANDIDATE_POOL_SIZE = 20
_CASCOR_ACTIVATION_FUNCTION = torch.tanh
_CASCOR_LEARNING_RATE = 0.1
# _CASCOR_LEARNING_RATE = 0.05
_CASCOR_MAX_HIDDEN_UNITS = 40
# _CASCOR_CORRELATION_THRESHOLD = 0.4
# _CASCOR_CORRELATION_THRESHOLD = 0.125
_CASCOR_CORRELATION_THRESHOLD = 0.05
# _CASCOR_PATIENCE = 5
_CASCOR_PATIENCE = 10
_CASCOR_CANDIDATE_EPOCHS = 400
_CASCOR_OUTPUT_EPOCHS = 500


#####################################################################################################################################################################################################
# Define constants for the Cascade Correlation Network
_CASCADE_CORRELATION_NETWORK_INPUT_SIZE = 2
_CASCADE_CORRELATION_NETWORK_OUTPUT_SIZE = 2
_CASCADE_CORRELATION_NETWORK_CANDIDATE_POOL_SIZE = 20
_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION = torch.tanh
_CASCADE_CORRELATION_NETWORK_LEARNING_RATE = 0.1
# _CASCADE_CORRELATION_NETWORK_LEARNING_RATE = 0.05
_CASCADE_CORRELATION_NETWORK_MAX_HIDDEN_UNITS = 40
# _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD = 0.4
# _CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD = 0.125
_CASCADE_CORRELATION_NETWORK_NODE_CORRELATION_THRESHOLD = 0.05
# _CASCADE_CORRELATION_NETWORK_PATIENCE = 5
_CASCADE_CORRELATION_NETWORK_PATIENCE = 10
_CASCADE_CORRELATION_NETWORK_CANDIDATE_EPOCHS = 400
_CASCADE_CORRELATION_NETWORK_OUTPUT_EPOCHS = 500
# _CASCADE_CORRELATION_NETWORK_LOGLEVEL_DEFAULT = logging.DEBUG
_CASCADE_CORRELATION_NETWORK_LOGLEVEL_DEFAULT = logging.INFO


#####################################################################################################################################################################################################
# Define constants for the Cascade Correlation Network
_CANDIDATE_UNIT_INPUT_SIZE = 2
_CANDIDATE_UNIT_ACTIVATION_FUNCTION = torch.tanh
_CANDIDATE_UNIT_LOGLEVEL_DEFAULT = logging.INFO