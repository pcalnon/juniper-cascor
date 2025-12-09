#!/usr/bin/env python
"""
HDF5 serialization package for CascadeCorrelationNetwork.
Provides comprehensive state capture and restoration capabilities.
"""

from .snapshot_serializer import CascadeHDF5Serializer
from .snapshot_utils import HDF5Utils
from .snapshot_common import (
    write_str_attr, read_str_attr, write_str_dataset, read_str_dataset,
    save_tensor, load_tensor, save_numpy_array, load_numpy_array
)

__version__ = "2.0.0"
__all__ = [
    'CascadeHDF5Serializer',
    'HDF5Utils',
    'write_str_attr',
    'read_str_attr', 
    'write_str_dataset',
    'read_str_dataset',
    'save_tensor',
    'load_tensor',
    'save_numpy_array',
    'load_numpy_array'
]
