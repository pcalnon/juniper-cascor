#!/usr/bin/env python
"""
Extended unit tests for snapshots/snapshot_common.py to improve test coverage.

Tests cover:
- read_str_attr when decode() fails - should fall back to str() (lines 55-58)
- read_str_dataset when asstr() fails - should fall back to manual decode (lines 109-114)
- load_tensor with CUDA device that fails - should suppress RuntimeError and return CPU tensor (lines 172-173)
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, PropertyMock, patch

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from snapshots.snapshot_common import load_tensor, read_str_attr, read_str_dataset, write_str_attr

pytestmark = pytest.mark.unit


@pytest.fixture
def temp_h5_file():
    """Create a temporary HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        filepath = f.name
    yield filepath
    if os.path.exists(filepath):
        os.unlink(filepath)


class TestReadStrAttrDecodeFailure:
    """Tests for read_str_attr when decode() fails (lines 55-58)."""

    def test_decode_unicode_error_falls_back_to_str(self):
        """Test that UnicodeDecodeError during decode falls back to str()."""

        class BadDecoder:
            def decode(self, encoding):
                raise UnicodeDecodeError("utf-8", b"\xff\xfe", 0, 1, "invalid")

            def __str__(self):
                return "str_fallback_value"

        mock_obj = MagicMock()
        mock_obj.attrs = {"key": BadDecoder()}

        result = read_str_attr(mock_obj, "key")
        assert result == "str_fallback_value"

    def test_decode_attribute_error_falls_back_to_str(self):
        """Test that AttributeError during decode falls back to str()."""

        class BadDecoder:
            def decode(self, encoding):
                raise AttributeError("no decode method")

            def __str__(self):
                return "attr_error_fallback"

        mock_obj = MagicMock()
        mock_obj.attrs = {"key": BadDecoder()}

        result = read_str_attr(mock_obj, "key")
        assert result == "attr_error_fallback"

    def test_object_with_decode_that_raises(self):
        """Test object that has decode method but raises exception."""

        class BadDecoder:
            def decode(self, encoding):
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")

            def __str__(self):
                return "fallback_from_str"

        mock_obj = MagicMock()
        mock_obj.attrs = {"key": BadDecoder()}

        result = read_str_attr(mock_obj, "key")
        assert result == "fallback_from_str"


class TestReadStrDatasetAsstrFailure:
    """Tests for read_str_dataset when asstr() fails (lines 109-114)."""

    def test_asstr_attribute_error_falls_back_to_manual_decode(self):
        """Test AttributeError from asstr() falls back to manual bytes decode."""
        mock_dataset = MagicMock()
        mock_dataset.asstr.side_effect = AttributeError("no asstr")
        mock_dataset.__getitem__ = MagicMock(return_value=b"manual_decode_value")

        mock_group = MagicMock()
        mock_group.__contains__ = MagicMock(return_value=True)
        mock_group.__getitem__ = MagicMock(return_value=mock_dataset)

        result = read_str_dataset(mock_group, "bytes_data")
        assert result == "manual_decode_value"

    def test_asstr_type_error_falls_back_to_manual_decode(self):
        """Test TypeError from asstr() falls back to manual bytes decode."""
        mock_dataset = MagicMock()
        mock_dataset.asstr.side_effect = TypeError("type error")
        mock_dataset.__getitem__ = MagicMock(return_value=b"type_error_fallback")

        mock_group = MagicMock()
        mock_group.__contains__ = MagicMock(return_value=True)
        mock_group.__getitem__ = MagicMock(return_value=mock_dataset)

        result = read_str_dataset(mock_group, "bytes_data")
        assert result == "type_error_fallback"

    def test_fallback_with_np_bytes_value(self):
        """Test fallback correctly decodes np.bytes_ value."""
        mock_dataset = MagicMock()
        mock_dataset.asstr.side_effect = AttributeError()
        mock_dataset.__getitem__ = MagicMock(return_value=np.bytes_("numpy_bytes_test"))

        mock_group = MagicMock()
        mock_group.__contains__ = MagicMock(return_value=True)
        mock_group.__getitem__ = MagicMock(return_value=mock_dataset)

        result = read_str_dataset(mock_group, "np_bytes")
        assert result == "numpy_bytes_test"

    def test_fallback_with_non_bytes_value(self):
        """Test fallback uses str() for non-bytes value."""
        mock_dataset = MagicMock()
        mock_dataset.asstr.side_effect = AttributeError()
        mock_dataset.__getitem__ = MagicMock(return_value=12345)

        mock_group = MagicMock()
        mock_group.__contains__ = MagicMock(return_value=True)
        mock_group.__getitem__ = MagicMock(return_value=mock_dataset)

        result = read_str_dataset(mock_group, "numeric")
        assert result == "12345"


class TestLoadTensorCudaFailure:
    """Tests for load_tensor when CUDA device fails (lines 172-173)."""

    def test_cuda_runtime_error_suppressed_returns_cpu_tensor(self, temp_h5_file):
        """Test that RuntimeError from CUDA .to() is suppressed and CPU tensor returned."""
        original = torch.randn(3, 3)

        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("tensors")
            dataset = group.create_dataset("data", data=original.numpy())
            write_str_attr(dataset, "device", "cuda:0")

        with h5py.File(temp_h5_file, "r") as f:
            with patch("torch.cuda.is_available", return_value=True):
                with patch.object(torch.Tensor, "to", side_effect=RuntimeError("CUDA error")):
                    loaded = load_tensor(f["tensors/data"])

                    assert isinstance(loaded, torch.Tensor)
                    assert loaded.device.type == "cpu"
                    assert torch.allclose(loaded, original)

    def test_cuda_available_but_device_transfer_fails(self, temp_h5_file):
        """Test tensor stays on CPU when CUDA transfer fails."""
        original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("tensors")
            dataset = group.create_dataset("weights", data=original.numpy())
            write_str_attr(dataset, "device", "cuda:1")
            dataset.attrs["requires_grad"] = False

        with h5py.File(temp_h5_file, "r") as f:
            with patch("torch.cuda.is_available", return_value=True):
                with patch.object(torch.Tensor, "to", side_effect=RuntimeError("device not found")):
                    loaded = load_tensor(f["tensors/weights"])

                    assert loaded.device.type == "cpu"
                    assert torch.equal(loaded, original)

    def test_cpu_device_no_transfer_attempted(self, temp_h5_file):
        """Test that CPU device skips .to() call entirely."""
        original = torch.randn(2, 2)

        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("tensors")
            dataset = group.create_dataset("cpu_tensor", data=original.numpy())
            write_str_attr(dataset, "device", "cpu")

        with h5py.File(temp_h5_file, "r") as f:
            loaded = load_tensor(f["tensors/cpu_tensor"])

            assert loaded.device.type == "cpu"

    def test_cuda_not_available_no_transfer_attempted(self, temp_h5_file):
        """Test that CUDA unavailable skips .to() call."""
        original = torch.randn(2, 2)

        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("tensors")
            dataset = group.create_dataset("tensor", data=original.numpy())
            write_str_attr(dataset, "device", "cuda:0")

        with h5py.File(temp_h5_file, "r") as f:
            with patch("torch.cuda.is_available", return_value=False):
                loaded = load_tensor(f["tensors/tensor"])

                assert loaded.device.type == "cpu"

    def test_no_device_attr_defaults_to_cpu(self, temp_h5_file):
        """Test tensor without device attr defaults to CPU."""
        original = torch.randn(3, 3)

        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("tensors")
            group.create_dataset("no_device", data=original.numpy())

        with h5py.File(temp_h5_file, "r") as f:
            loaded = load_tensor(f["tensors/no_device"])

            assert loaded.device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
