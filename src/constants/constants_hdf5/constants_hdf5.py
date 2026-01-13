#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     constants_hdf5.py
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
#    This file contains hdf5 constants used in the Cascade Correlation Neural Network implementation.
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
import pathlib
# import torch
# import hd5py
# import numpy as np


#####################################################################################################################################################################################################
_HDF5_PROJECT_HDF5_CONSTANTS_DIR = pathlib.Path(__file__).parent.resolve()
_HDF5_PROJECT_CONSTANTS_DIR = _HDF5_PROJECT_HDF5_CONSTANTS_DIR.parent.resolve()
_HDF5_PROJECT_SOURCE_DIR = _HDF5_PROJECT_CONSTANTS_DIR.parent.resolve()
_HDF5_PROJECT_DIR = _HDF5_PROJECT_SOURCE_DIR.parent.resolve()

_HDF5_PROJECT_SNAPSHOTS_DIR_NAME="cascor_snapshots"
_HDF5_PROJECT_SNAPSHOTS_DIR=pathlib.Path(_HDF5_PROJECT_SOURCE_DIR).joinpath(_HDF5_PROJECT_SNAPSHOTS_DIR_NAME)


# Define HDF5 Storage class Constants to provide reasonable defaults

# Define HDF5Storage Constants for hdf5 file and dataset structure
# _HDF5_STORAGE_HOME_DIR = _GENERATED_DATASETS_HOME_DIR
# _HDF5_STORAGE_ROOT_DIR = _GENERATED_DATASETS_ROOT_DIR
# _HDF5_STORAGE_PARENT_DIR_NAME = _GENERATED_DATASETS_PARENT_DIR_NAME
# _HDF5_STORAGE_PARENT_DIR = _GENERATED_DATASETS_PARENT_DIR
# _HDF5_STORAGE_APPLICATION_DIR_NAME = _GENERATED_DATASETS_APPLICATION_DIR_NAME
# _HDF5_STORAGE_APPLICATION_DIR = _GENERATED_DATASETS_APPLICATION_DIR
# _HDF5_STORAGE_PROJ_DIR_NAME = _GENERATED_DATASETS_PROJ_DIR_NAME
# _HDF5_STORAGE_PROJ_DIR = _GENERATED_DATASETS_PROJ_DIR
# _HDF5_STORAGE_DATA_DIR_NAME = _GENERATED_DATASETS_DATA_DIR_NAME
# _HDF5_STORAGE_DATA_DIR = _GENERATED_DATASETS_DATA_DIR
# _HDF5_STORAGE_IMAGE_DIR_NAME = _GENERATED_DATASETS_IMAGE_DIR_NAME
# _HDF5_STORAGE_IMAGE_DIR = _GENERATED_DATASETS_IMAGE_DIR


# Define HDF5Storage Constants for Logging
# _HDF5_STORAGE_LOG_NAME = _CASCOR_SPIRAL_DATASET_LOG_NAME
# _HDF5_STORAGE_LOG_DATE_FORMAT = _CASCOR_SPIRAL_DATASET_LOG_DATE_FORMAT
# _HDF5_STORAGE_LOG_FORMATTER_STRING = _CASCOR_SPIRAL_DATASET_LOG_FORMATTER_STRING
# _HDF5_STORAGE_LOG_DIR = _CASCOR_SPIRAL_DATASET_LOG_DIR_DEFAULT
# _HDF5_STORAGE_LOG_FILE_PATH = _CASCOR_SPIRAL_DATASET_LOG_FILE_PATH_DEFAULT
# _HDF5_STORAGE_LOG_LEVEL = _CASCOR_SPIRAL_DATASET_LOG_LEVEL_DEFAULT

