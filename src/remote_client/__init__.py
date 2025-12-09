#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network  
# File Name:     __init__.py
# Author:        Paul Calnon
# Version:       0.3.2 (0.7.3)
#
# Date:          2025-09-27
# Last Modified: 2025-09-27
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    Remote client module for Cascade Correlation multiprocessing.
#
#####################################################################################################################################################################################################

from .remote_client import RemoteWorkerClient

__all__ = ['RemoteWorkerClient']
