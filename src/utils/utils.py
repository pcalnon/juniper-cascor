#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network with N-Spiral Dataset Generator
# File Name:     utils.py
# Author:        Paul Calnon
# Version:       0.3.2 (0.7.3)
#
# Date Created:  2025-06-01
# Last Modified: 2026-01-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#     This module provides utility functions for saving, loading, and processing datasets.
#
#####################################################################################################################################################################################################
# Notes:
#    - This file contains a group of utility and convenience functions that can be imported and used to simplify methods
#       in the Cascade Correlation Neural Network with N-Spiral Dataset Generator.
#    - These functions include saving and loading datasets, converting between PyTorch tensors and NumPy arrays,
#       and generating N-Spiral datasets.
#    - The save_dataset function saves the input features and target tensors as a dictionary to the specified file path using PyTorch's torch.save function.
#    - The load_dataset function loads the input features and target tensors from the specified file path using PyTorch's torch.load function.
#    - The convert_to_numpy function converts PyTorch tensors to NumPy arrays, while the convert_to_tensor function converts NumPy arrays to PyTorch tensors.
#    - The display_progress function checks if the current epoch matches a specified display frequency for logging progress during training.
#    - The get_class_distribution function calculates the distribution of classes in a one-hot encoded target tensor and returns it as a dictionary.
#    - The display_object_attributes function returns the attributes of a specified object in a columnar format using the columnar library.
#
#####################################################################################################################################################################################################
# References:
#
#
#
#####################################################################################################################################################################################################
# TODO :
#    - incorporate this code into the SpiralDataset class by converting these functions into an inherited abstract base class
#
#####################################################################################################################################################################################################
# COMPLETED:
#    - Integration of util code from cascor_spiral/src/utils/utils.py
#
#####################################################################################################################################################################################################

# from typing import Tuple, Dict, Any
from typing import Dict, Tuple
try:
    import columnar as col
    HAS_COLUMNAR = True
except ImportError:
    HAS_COLUMNAR = False
    col = None
import numpy as np
import torch
import yaml

from log_config.logger.logger import Logger

def save_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    file_path: str,
) -> None:
    """
    Description:
        Save a dataset to a file.
    Args:
        x: Input features tensor
        y: Target tensor
        file_path: Path to save the dataset
    """
    torch.save({"x": x, "y": y}, file_path)


def load_dataset(file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
        Load a dataset from a file.
    Args:
        file_path: Path to the saved dataset
    Returns:
        Tuple containing:
            - x: Input features tensor
            - y: Target tensor
    """
    data = yaml.safe_load(file_path.read())
    # data = torch.load(file_path)
    return (data["x"], data["y"])


def display_progress(display_frequency: int,) -> bool:
    """
    Description:
        Determine if epoch matches display frequency for logging progress of training.
    Args:
        epoch: Current epoch number
    Returns:
        bool: True if epoch matches display frequency, False otherwise
    Notes:
        This method checks if the current epoch is a multiple of the display frequency.
        If it is, it returns True, indicating that progress should be displayed.
        Otherwise, it returns False.
        This method is not called directly, but rather returns a lambda function 
        that can be stored as a class attribute and used to check the condition based on the class frequency.
    """
    # if epoch <= 0:
    #     raise ValueError("Epoch must be a positive integer.")
    logger = Logger
    logger.debug(f"Utils: display_progress: Display Frequency: Type: {type(display_frequency)} Value: {display_frequency}")
    return lambda epoch: ((epoch + 1) % display_frequency == 0 if display_frequency > 0 else False) if epoch >= 0 else lambda_raise_(ValueError("Epoch must be a positive integer."))
    # return lambda display_frequency : (epoch + 1) % display_frequency == 0 if display_frequency > 0 else False


def lambda_raise_(ex: Exception) -> None:
    raise ex


def get_class_distribution(y: torch.Tensor) -> Dict[int, int]:
    """
    Description:
        Get the distribution of classes in the dataset.
        
    Args:
        y: Target tensor (one-hot encoded)
    Returns:
        Dictionary mapping class indices to counts
    """
    if len(y.shape) > 1 and y.shape[1] > 1:
        y_indices = torch.argmax(y, dim=1).numpy()   # One-hot encoded
    else:
        y_indices = y.numpy()                        # Class indices
    unique, counts = np.unique(y_indices, return_counts=True)
    return dict(zip(unique, counts, strict=False))


def convert_to_numpy(
    x: torch.Tensor,
    y: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        Convert PyTorch tensors to NumPy arrays.
    Args:
        x: Input features tensor
        y: Target tensor
    Returns:
        Tuple containing:
            - x_np: Input features as NumPy array
            - y_np: Targets as NumPy array
    """
    x_np = x.numpy() if isinstance(x, torch.Tensor) else x
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y
    return (x_np, y_np)

def convert_to_tensor(x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
        Convert NumPy arrays to PyTorch tensors.
    Args:
        x: Input features as NumPy array
        y: Targets as NumPy array
    Returns:
        Tuple containing:
            - x_tensor: Input features as PyTorch tensor
            - y_tensor: Targets as PyTorch tensor
    """
    x_tensor = torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x
    y_tensor = torch.tensor(y, dtype=torch.float32) if isinstance(y, np.ndarray) else y
    return (x_tensor, y_tensor)


def display_object_attributes(object_name: str = None, private_attrs: bool = False) -> list[str]:
    """
    Description:
        Return the attributes of a specified object in a columnar format.
    Args:
        object_name: Name of the object to display attributes for (default is None, which uses the logging module)
        private_attrs: If True, include private attributes (those starting with an underscore)
    Returns:
        List of strings representing the object's attributes and their values in a columnar format.
    Notes:
        This function uses the `columnar` library to format and display the attributes of the specified object.
        It filters out private attributes (those starting with an underscore) if specified.
    Example:
        display_object_attributes(object_name="logging", private_attrs=False)
    """
    obj = __import__(object_name) if object_name is not None and isinstance(object_name, str) and hasattr( __import__(object_name), '__dict__') and isinstance(private_attrs, bool) else None
    obj_dict = obj.__dict__ if obj is not None else None
    keys = obj_dict.keys() if obj_dict is not None else None
    return _object_attributes_to_table(obj_dict=obj_dict, keys=keys, private_attrs=private_attrs)


def _object_attributes_to_table(obj_dict: dict = None, keys: [str] = None, private_attrs: bool = False) -> str:
    """
    Description:
        Convert the attributes of a specified object to a columnar format.
    Args:
        obj_dict: Dictionary of the object's attributes (default is None)
        keys: List of attribute names (default is None)
        private_attrs: If True, include private attributes (those starting with an underscore)
    Returns:
        String representing the object's attributes and their values in a columnar format.
    """
    if (content := _init_content_list( obj_dict is not None and keys is not None and private_attrs is not None) is not None):
        for key in keys:                                     # parse attributes from target object
            if key.startswith("_") and not private_attrs:    # Ignore private attributes
                continue
            content.append([key, obj_dict.get(key)],)
        headers = ['Attribute','Attribute Value']
        # Generate the columnar data table containing the attributes and their values for the provided object
        if HAS_COLUMNAR and col is not None:
            no_borders = True
            content = col.columnar(data=content, headers=headers, no_borders=no_borders) if content else None
        else:
            # Fallback to simple string formatting if columnar not available
            content = '\n'.join([f"{headers[0]}: {row[0]}, {headers[1]}: {row[1]}" for row in content]) if content else None
    return content                                           # return the columnar table data

def _init_content_list(validity_check: bool = False) -> list[str]:
    """
    Description:
        This function is used to handle the scenario where the object's attributes are not valid or not available.
    Args:
        validity_check: If True, return an empty list; otherwise, return None
    Returns:
        List of strings representing the content (or None)
    """
    return [] if validity_check else None


# TODO: This function is borked
def check_object_pickleability(instance: object = None) -> bool:
    """
    Description:
        Check if the given object instance is pickleable.
    Args:
        instance: The object instance to check (default is None)
    Returns:
        bool: True if the object is pickleable, False otherwise
    """
    import dill # trunk-ignore(bandit/B403)

    if instance is None or not hasattr(instance, '__dict__'):
        return False

    logger = Logger

    is_a_pickle = True
    logger.debug(f"Utils: check_object_pickleability: Candidate Training Manager instance attributes dict: {instance.__dict__}")
    for i in instance.__dict__.keys():
        logger.debug(f"Utils: _create_multiprocessing_manager: Candidate Training Manager instance attribute: {i}: \t\t{instance.__dict__[i]}")
        logger.debug(f"Utils: _create_multiprocessing_manager: Checking instance attribute for Pickleability: {i}, \t\t{dill.pickles(instance.__dict__[i])}")
        if not dill.pickles(instance.__dict__[i]):
            is_a_pickle = False
            logger.debug(f"Utils: _create_multiprocessing_manager: Instance attribute NOT Pickleable: {i}, \t\t{dill.pickles(instance.__dict__[i])}")
    return is_a_pickle


