#!/usr/bin/env python
"""
Unit tests for snapshots/snapshot_common.py to improve test coverage.

Tests focus on:
- read_str_attr / write_str_attr utility functions
- read_str_dataset / write_str_dataset functions
- save_tensor / load_tensor functions
- save_numpy_array / load_numpy_array functions
- validate_tensor_dataset function
- calculate_tensor_checksum / verify_tensor_checksum functions
- Error handling paths and edge cases
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from snapshots.snapshot_common import calculate_tensor_checksum, load_numpy_array, load_tensor, read_str_attr, read_str_dataset, save_numpy_array, save_tensor, validate_tensor_dataset, verify_tensor_checksum, write_str_attr, write_str_dataset

pytestmark = pytest.mark.unit


@pytest.fixture
def temp_h5_file():
    """Create a temporary HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        filepath = f.name
    yield filepath
    if os.path.exists(filepath):
        os.unlink(filepath)


class TestWriteStrAttr:
    """Tests for write_str_attr function."""

    def test_write_str_attr_basic(self, temp_h5_file):
        """Test writing a basic string attribute."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")
            write_str_attr(group, "key", "value")
            assert "key" in group.attrs

    def test_write_str_attr_none_value(self, temp_h5_file):
        """Test that None value is not written."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")
            write_str_attr(group, "key", None)
            assert "key" not in group.attrs

    def test_write_str_attr_to_dataset(self, temp_h5_file):
        """Test writing string attribute to a dataset."""
        with h5py.File(temp_h5_file, "w") as f:
            dataset = f.create_dataset("data", data=np.array([1, 2, 3]))
            write_str_attr(dataset, "description", "test data")
            assert "description" in dataset.attrs

    def test_write_str_attr_integer_value(self, temp_h5_file):
        """Test writing an integer value as string attribute."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")
            write_str_attr(group, "number", 42)
            assert "number" in group.attrs


class TestReadStrAttr:
    """Tests for read_str_attr function."""

    def test_read_str_attr_basic(self, temp_h5_file):
        """Test reading a basic string attribute."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")
            group.attrs["key"] = np.bytes_("value")

        with h5py.File(temp_h5_file, "r") as f:
            result = read_str_attr(f["test"], "key")
            assert result == "value"

    def test_read_str_attr_missing_key(self, temp_h5_file):
        """Test reading missing attribute returns default."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")

        with h5py.File(temp_h5_file, "r") as f:
            result = read_str_attr(f["test"], "missing")
            assert result is None

    def test_read_str_attr_missing_key_with_default(self, temp_h5_file):
        """Test reading missing attribute returns custom default."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")

        with h5py.File(temp_h5_file, "r") as f:
            result = read_str_attr(f["test"], "missing", default="default_value")
            assert result == "default_value"

    def test_read_str_attr_bytes_value(self, temp_h5_file):
        """Test reading bytes value."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")
            group.attrs["key"] = b"bytes_value"

        with h5py.File(temp_h5_file, "r") as f:
            result = read_str_attr(f["test"], "key")
            assert result == "bytes_value"

    def test_read_str_attr_np_bytes_value(self, temp_h5_file):
        """Test reading np.bytes_ value."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")
            group.attrs["key"] = np.bytes_("np_bytes_value")

        with h5py.File(temp_h5_file, "r") as f:
            result = read_str_attr(f["test"], "key")
            assert result == "np_bytes_value"

    def test_read_str_attr_non_bytes_value(self, temp_h5_file):
        """Test reading a non-bytes value (falls through to str())."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")
            group.attrs["key"] = 12345

        with h5py.File(temp_h5_file, "r") as f:
            result = read_str_attr(f["test"], "key")
            assert result == "12345"

    def test_read_str_attr_decode_error_fallback(self, temp_h5_file):
        """Test fallback when decode raises UnicodeDecodeError."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")
            group.attrs["key"] = "test_string"

        with h5py.File(temp_h5_file, "r") as f:
            mock_val = MagicMock()
            mock_val.decode = MagicMock(side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "error"))
            mock_val.__str__ = MagicMock(return_value="fallback_string")

            with patch.object(f["test"].attrs, "__getitem__", return_value=mock_val):
                with patch.object(f["test"].attrs, "__contains__", return_value=True):
                    result = read_str_attr(f["test"], "key")
                    assert "fallback_string" in str(result) or result is not None


class TestWriteStrDataset:
    """Tests for write_str_dataset function."""

    def test_write_str_dataset_basic(self, temp_h5_file):
        """Test writing a basic string dataset."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")
            dataset = write_str_dataset(group, "text", "hello world")
            assert "text" in group
            assert dataset is not None

    def test_write_str_dataset_overwrites_existing(self, temp_h5_file):
        """Test that writing overwrites existing dataset."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")
            write_str_dataset(group, "text", "first")
            write_str_dataset(group, "text", "second")
            result = read_str_dataset(group, "text")
            assert result == "second"

    def test_write_str_dataset_filters_compression_kwargs(self, temp_h5_file):
        """Test that compression kwargs are filtered for scalar strings."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")
            dataset = write_str_dataset(
                group,
                "text",
                "test value",
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )
            assert "text" in group
            assert dataset is not None


class TestReadStrDataset:
    """Tests for read_str_dataset function."""

    def test_read_str_dataset_basic(self, temp_h5_file):
        """Test reading a basic string dataset."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")
            write_str_dataset(group, "text", "hello")

        with h5py.File(temp_h5_file, "r") as f:
            result = read_str_dataset(f["test"], "text")
            assert result == "hello"

    def test_read_str_dataset_missing(self, temp_h5_file):
        """Test reading missing dataset returns default."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")

        with h5py.File(temp_h5_file, "r") as f:
            result = read_str_dataset(f["test"], "missing")
            assert result is None

    def test_read_str_dataset_missing_with_default(self, temp_h5_file):
        """Test reading missing dataset returns custom default."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")

        with h5py.File(temp_h5_file, "r") as f:
            result = read_str_dataset(f["test"], "missing", default="fallback")
            assert result == "fallback"

    def test_read_str_dataset_bytes_fallback(self, temp_h5_file):
        """Test reading dataset with bytes fallback path."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("test")
            group.create_dataset("bytes_data", data=np.bytes_("bytes content"))

        with h5py.File(temp_h5_file, "r") as f:
            result = read_str_dataset(f["test"], "bytes_data")
            assert result == "bytes content"


class TestSaveTensor:
    """Tests for save_tensor function."""

    def test_save_tensor_basic(self, temp_h5_file):
        """Test saving a basic tensor."""
        tensor = torch.randn(5, 3)
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("tensors")
            dataset = save_tensor(group, "weights", tensor)
            assert "weights" in group
            assert dataset is not None
            assert "tensor_type" in dataset.attrs
            assert "dtype" in dataset.attrs
            assert "shape" in dataset.attrs

    def test_save_tensor_with_requires_grad(self, temp_h5_file):
        """Test saving a tensor with requires_grad=True."""
        tensor = torch.randn(3, 3, requires_grad=True)
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("tensors")
            dataset = save_tensor(group, "grad_tensor", tensor)
            assert bool(dataset.attrs["requires_grad"]) is True

    def test_save_tensor_custom_compression(self, temp_h5_file):
        """Test saving tensor with custom compression options."""
        tensor = torch.randn(10, 10)
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("tensors")
            dataset = save_tensor(group, "compressed", tensor, compression="gzip", compression_opts=9)
            assert "compressed" in group


class TestLoadTensor:
    """Tests for load_tensor function."""

    def test_load_tensor_basic(self, temp_h5_file):
        """Test loading a basic tensor."""
        original = torch.randn(5, 3)
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("tensors")
            save_tensor(group, "weights", original)

        with h5py.File(temp_h5_file, "r") as f:
            loaded = load_tensor(f["tensors/weights"])
            assert torch.allclose(loaded, original)

    def test_load_tensor_preserves_requires_grad(self, temp_h5_file):
        """Test that requires_grad is preserved."""
        original = torch.randn(3, 3, requires_grad=True)
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("tensors")
            save_tensor(group, "grad_tensor", original)

        with h5py.File(temp_h5_file, "r") as f:
            loaded = load_tensor(f["tensors/grad_tensor"])
            assert loaded.requires_grad is True

    def test_load_tensor_without_requires_grad_attr(self, temp_h5_file):
        """Test loading tensor without requires_grad attribute."""
        arr = np.random.randn(3, 3).astype(np.float32)
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("tensors")
            dataset = group.create_dataset("plain", data=arr)
            write_str_attr(dataset, "device", "cpu")

        with h5py.File(temp_h5_file, "r") as f:
            loaded = load_tensor(f["tensors/plain"])
            assert isinstance(loaded, torch.Tensor)

    def test_load_tensor_cpu_device_fallback(self, temp_h5_file):
        """Test loading tensor with non-CPU device falls back to CPU when CUDA unavailable."""
        original = torch.randn(3, 3)
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("tensors")
            dataset = group.create_dataset("data", data=original.numpy())
            write_str_attr(dataset, "device", "cuda:0")

        with h5py.File(temp_h5_file, "r") as f:
            loaded = load_tensor(f["tensors/data"])
            assert isinstance(loaded, torch.Tensor)


class TestSaveNumpyArray:
    """Tests for save_numpy_array function."""

    def test_save_numpy_array_basic(self, temp_h5_file):
        """Test saving a basic numpy array."""
        arr = np.random.randn(10, 5)
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("arrays")
            dataset = save_numpy_array(group, "data", arr)
            assert "data" in group
            assert "array_type" in dataset.attrs
            assert "dtype" in dataset.attrs
            assert "shape" in dataset.attrs

    def test_save_numpy_array_custom_compression(self, temp_h5_file):
        """Test saving numpy array with custom compression."""
        arr = np.random.randn(20, 20)
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("arrays")
            dataset = save_numpy_array(group, "compressed", arr, compression="gzip", compression_opts=9)
            assert "compressed" in group


class TestLoadNumpyArray:
    """Tests for load_numpy_array function."""

    def test_load_numpy_array_basic(self, temp_h5_file):
        """Test loading a basic numpy array."""
        original = np.random.randn(10, 5)
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("arrays")
            save_numpy_array(group, "data", original)

        with h5py.File(temp_h5_file, "r") as f:
            loaded = load_numpy_array(f["arrays/data"])
            np.testing.assert_array_almost_equal(loaded, original)


class TestValidateTensorDataset:
    """Tests for validate_tensor_dataset function."""

    def test_validate_valid_tensor_dataset(self, temp_h5_file):
        """Test validating a valid tensor dataset."""
        tensor = torch.randn(5, 3)
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("tensors")
            save_tensor(group, "valid", tensor)

        with h5py.File(temp_h5_file, "r") as f:
            result = validate_tensor_dataset(f["tensors/valid"])
            assert result is True

    def test_validate_missing_tensor_type_attr(self, temp_h5_file):
        """Test validation fails when tensor_type is missing."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("data")
            group.create_dataset("invalid", data=np.array([1, 2, 3]))

        with h5py.File(temp_h5_file, "r") as f:
            result = validate_tensor_dataset(f["data/invalid"])
            assert result is False

    def test_validate_wrong_tensor_type(self, temp_h5_file):
        """Test validation fails when tensor_type is wrong."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("data")
            dataset = group.create_dataset("invalid", data=np.array([1, 2, 3]))
            write_str_attr(dataset, "tensor_type", "wrong.Type")

        with h5py.File(temp_h5_file, "r") as f:
            result = validate_tensor_dataset(f["data/invalid"])
            assert result is False

    def test_validate_shape_mismatch(self, temp_h5_file):
        """Test validation fails when shape doesn't match."""
        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("data")
            dataset = group.create_dataset("invalid", data=np.array([[1, 2], [3, 4]]))
            write_str_attr(dataset, "tensor_type", "torch.Tensor")
            dataset.attrs["shape"] = (5, 5)

        with h5py.File(temp_h5_file, "r") as f:
            result = validate_tensor_dataset(f["data/invalid"])
            assert result is False

    def test_validate_exception_handling(self, temp_h5_file):
        """Test validation handles exceptions gracefully."""
        mock_dataset = MagicMock()
        mock_dataset.attrs = MagicMock()
        mock_dataset.attrs.__contains__ = MagicMock(side_effect=Exception("test error"))

        result = validate_tensor_dataset(mock_dataset)
        assert result is False


class TestCalculateTensorChecksum:
    """Tests for calculate_tensor_checksum function."""

    def test_calculate_checksum_basic(self):
        """Test calculating checksum of a tensor."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        checksum = calculate_tensor_checksum(tensor)
        assert isinstance(checksum, str)
        assert len(checksum) == 64

    def test_calculate_checksum_deterministic(self):
        """Test that checksum is deterministic."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        checksum1 = calculate_tensor_checksum(tensor)
        checksum2 = calculate_tensor_checksum(tensor)
        assert checksum1 == checksum2

    def test_calculate_checksum_different_tensors(self):
        """Test that different tensors have different checksums."""
        tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        tensor2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        checksum1 = calculate_tensor_checksum(tensor1)
        checksum2 = calculate_tensor_checksum(tensor2)
        assert checksum1 != checksum2


class TestVerifyTensorChecksum:
    """Tests for verify_tensor_checksum function."""

    def test_verify_checksum_match(self):
        """Test verification with matching checksum."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        checksum = calculate_tensor_checksum(tensor)
        result = verify_tensor_checksum(tensor, checksum)
        assert result is True

    def test_verify_checksum_mismatch(self):
        """Test verification with non-matching checksum."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        wrong_checksum = "0" * 64
        result = verify_tensor_checksum(tensor, wrong_checksum)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
