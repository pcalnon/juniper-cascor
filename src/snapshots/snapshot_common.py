#!/usr/bin/env python
"""
Common utilities for HDF5 serialization operations.
Provides robust string and tensor I/O helpers.
"""


import contextlib
import hashlib
import numpy as np
import torch
import h5py
from typing import Any, Optional, Union


def write_str_attr(obj: Union[h5py.Group, h5py.Dataset], key: str, value: Any) -> None:
    """
    Safely write a string attribute to HDF5 object.
    
    Args:
        obj: HDF5 group or dataset
        key: Attribute key
        value: Value to store as string
    """
    if value is None:
        return
    # Use np.bytes_ for NumPy 2.0+ compatibility (np.string_ was removed)
    obj.attrs[key] = np.bytes_(str(value))


def read_str_attr(obj: Union[h5py.Group, h5py.Dataset], key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Safely read a string attribute from HDF5 object.
    
    Args:
        obj: HDF5 group or dataset
        key: Attribute key
        default: Default value if key doesn't exist
        
    Returns:
        String value or default
    """
    if key not in obj.attrs:
        return default
    
    val = obj.attrs[key]
    
    # Handle bytes-like objects
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode('utf-8')
    
    # Handle objects with decode method
    if hasattr(val, 'decode'):
        try:
            return val.decode('utf-8')
        except (UnicodeDecodeError, AttributeError):
            return str(val)
    
    return str(val)


def write_str_dataset(group: h5py.Group, name: str, value: Any, **kwargs) -> h5py.Dataset:
    """
    Write a string dataset with proper UTF-8 encoding.
    
    Args:
        group: HDF5 group
        name: Dataset name
        value: String value to store
        **kwargs: Additional dataset creation arguments (compression/chunks excluded for scalar strings)
        
    Returns:
        Created dataset
    """
    # Use UTF-8 string dtype
    dt = h5py.string_dtype('utf-8')
    
    # Remove existing dataset if present
    if name in group:
        del group[name]
    
    # Remove compression/chunk options that don't work with scalar string datasets
    kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ['compression', 'compression_opts', 'chunks']}
    
    return group.create_dataset(name, data=str(value), dtype=dt, **kwargs_filtered)


def read_str_dataset(group: h5py.Group, name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Read a string dataset with proper decoding.
    
    Args:
        group: HDF5 group
        name: Dataset name
        default: Default value if dataset doesn't exist
        
    Returns:
        String value or default
    """
    if name not in group:
        return default
    
    dataset = group[name]
    
    try:
        # Try to use asstr() for string datasets
        return dataset.asstr()[()]
    except (AttributeError, TypeError):
        # Fallback to manual decoding
        data = dataset[()]
        if isinstance(data, (bytes, np.bytes_)):
            return data.decode('utf-8')
        return str(data)


def save_tensor(
    group: h5py.Group, 
    name: str, 
    tensor: torch.Tensor, 
    compression: str = 'gzip',
    compression_opts: int = 4
) -> h5py.Dataset:
    """
    Save a PyTorch tensor to HDF5 with metadata preservation.
    
    Args:
        group: HDF5 group
        name: Dataset name
        tensor: PyTorch tensor to save
        compression: Compression method
        compression_opts: Compression level
        
    Returns:
        Created dataset
    """
    # Convert to numpy on CPU
    arr = tensor.detach().cpu().numpy()
    
    # Create dataset with compression
    dataset = group.create_dataset(
        name, 
        data=arr, 
        compression=compression,
        compression_opts=compression_opts
    )
    
    # Store tensor metadata
    write_str_attr(dataset, 'tensor_type', 'torch.Tensor')
    write_str_attr(dataset, 'dtype', str(tensor.dtype))
    write_str_attr(dataset, 'device', str(tensor.device))
    dataset.attrs['requires_grad'] = bool(getattr(tensor, 'requires_grad', False))
    
    # Store shape for verification
    dataset.attrs['shape'] = tensor.shape
    
    return dataset


def load_tensor(dataset: h5py.Dataset) -> torch.Tensor:
    """
    Load a PyTorch tensor from HDF5 dataset with metadata restoration.
    
    Args:
        dataset: HDF5 dataset containing tensor data
        
    Returns:
        Reconstructed PyTorch tensor
    """
    # Load numpy array
    data = dataset[:]

    # Convert to tensor
    tensor = torch.from_numpy(data)

    # Restore requires_grad
    if 'requires_grad' in dataset.attrs:
        tensor.requires_grad_(bool(dataset.attrs['requires_grad']))

    # Restore device (if not CPU and CUDA available)
    device = read_str_attr(dataset, 'device', 'cpu')
    if device != 'cpu' and torch.cuda.is_available():
        with contextlib.suppress(RuntimeError):
            tensor = tensor.to(device)
    return tensor


def save_numpy_array(
    group: h5py.Group, 
    name: str, 
    array: np.ndarray, 
    compression: str = 'gzip',
    compression_opts: int = 4
) -> h5py.Dataset:
    """
    Save a numpy array to HDF5 with metadata.
    
    Args:
        group: HDF5 group
        name: Dataset name
        array: Numpy array to save
        compression: Compression method
        compression_opts: Compression level
        
    Returns:
        Created dataset
    """
    dataset = group.create_dataset(
        name, 
        data=array, 
        compression=compression,
        compression_opts=compression_opts
    )
    
    # Store metadata
    write_str_attr(dataset, 'array_type', 'numpy.ndarray')
    write_str_attr(dataset, 'dtype', str(array.dtype))
    dataset.attrs['shape'] = array.shape
    
    return dataset


def load_numpy_array(dataset: h5py.Dataset) -> np.ndarray:
    """
    Load a numpy array from HDF5 dataset.
    
    Args:
        dataset: HDF5 dataset containing array data
        
    Returns:
        Numpy array
    """
    return dataset[:]


def validate_tensor_dataset(dataset: h5py.Dataset) -> bool:
    """
    Validate that a dataset contains valid tensor data.

    Args:
        dataset: HDF5 dataset to validate

    Returns:
        True if valid tensor dataset
    """
    try:
        # Check for required attributes
        tensor_type = read_str_attr(dataset, 'tensor_type')
        if tensor_type != 'torch.Tensor':
            return False

        # Check data shape consistency
        if 'shape' in dataset.attrs:
            expected_shape = tuple(dataset.attrs['shape'])
            actual_shape = dataset.shape
            if expected_shape != actual_shape:
                return False

        return True

    except Exception:
        return False


def calculate_tensor_checksum(tensor: torch.Tensor) -> str:
    """
    Calculate SHA256 checksum of tensor data.

    Args:
        tensor: PyTorch tensor to checksum

    Returns:
        Hexadecimal checksum string
    """
    tensor_bytes = tensor.detach().cpu().numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


def verify_tensor_checksum(tensor: torch.Tensor, expected_checksum: str) -> bool:
    """
    Verify tensor checksum matches expected value.

    Args:
        tensor: PyTorch tensor to verify
        expected_checksum: Expected checksum value

    Returns:
        True if checksums match
    """
    actual_checksum = calculate_tensor_checksum(tensor)
    return actual_checksum == expected_checksum
