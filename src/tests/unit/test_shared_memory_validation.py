#!/usr/bin/env python
"""
Project:       Juniper
Prototype:     Cascade Correlation Neural Network
File Name:     test_shared_memory_validation.py
Author:        Paul Calnon

Date:          2026-04-05
Last Modified: 2026-04-05

License:       MIT License
Copyright:     Copyright (c) 2024-2026 Paul Calnon

Description:
    Unit tests for SharedTrainingMemory ndim validation (CR-047).
    Verifies that tensors with ndim > 2 are rejected with a clear error message,
    and that 0D, 1D, and 2D tensors continue to work correctly.
"""

import pytest
import torch

from cascade_correlation.cascade_correlation import SharedTrainingMemory


class TestSharedTrainingMemoryNdimValidation:
    """Test ndim validation in SharedTrainingMemory (CR-047)."""

    @pytest.mark.unit
    @pytest.mark.validation
    def test_rejects_3d_tensor(self):
        """SharedTrainingMemory should reject 3D tensors with ValueError."""
        tensor_3d = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match=r"only supports tensors up to 2 dimensions.*got tensor with 3 dimensions"):
            SharedTrainingMemory([tensor_3d], name_suffix="test_3d")

    @pytest.mark.unit
    @pytest.mark.validation
    def test_rejects_4d_tensor(self):
        """SharedTrainingMemory should reject 4D tensors with ValueError."""
        tensor_4d = torch.randn(2, 3, 4, 5)
        with pytest.raises(ValueError, match=r"only supports tensors up to 2 dimensions.*got tensor with 4 dimensions"):
            SharedTrainingMemory([tensor_4d], name_suffix="test_4d")

    @pytest.mark.unit
    @pytest.mark.validation
    def test_rejects_5d_tensor(self):
        """SharedTrainingMemory should reject 5D tensors with ValueError."""
        tensor_5d = torch.randn(2, 3, 4, 5, 6)
        with pytest.raises(ValueError, match=r"only supports tensors up to 2 dimensions.*got tensor with 5 dimensions"):
            SharedTrainingMemory([tensor_5d], name_suffix="test_5d")

    @pytest.mark.unit
    @pytest.mark.validation
    def test_error_message_includes_shape(self):
        """Error message should include the actual tensor shape."""
        tensor_3d = torch.randn(8, 16, 32)
        with pytest.raises(ValueError, match=r"\(8, 16, 32\)"):
            SharedTrainingMemory([tensor_3d], name_suffix="test_shape_msg")

    @pytest.mark.unit
    @pytest.mark.validation
    def test_rejects_3d_tensor_in_mixed_list(self):
        """A 3D tensor among valid tensors should still be rejected."""
        tensor_2d = torch.randn(4, 5)
        tensor_3d = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match=r"only supports tensors up to 2 dimensions"):
            SharedTrainingMemory([tensor_2d, tensor_3d], name_suffix="test_mixed")

    @pytest.mark.unit
    @pytest.mark.validation
    def test_accepts_1d_tensor(self):
        """SharedTrainingMemory should accept 1D tensors."""
        tensor_1d = torch.randn(10)
        shm = SharedTrainingMemory([tensor_1d], name_suffix="test_1d_ok")
        try:
            tensors, handle = SharedTrainingMemory.reconstruct_tensors(shm.get_metadata())
            assert len(tensors) == 1  # trunk-ignore(bandit/B101)
            assert tensors[0].shape == (10,)  # trunk-ignore(bandit/B101)
            handle.close()
        finally:
            shm.close()
            shm.unlink()

    @pytest.mark.unit
    @pytest.mark.validation
    def test_accepts_2d_tensor(self):
        """SharedTrainingMemory should accept 2D tensors."""
        tensor_2d = torch.randn(4, 5)
        shm = SharedTrainingMemory([tensor_2d], name_suffix="test_2d_ok")
        try:
            tensors, handle = SharedTrainingMemory.reconstruct_tensors(shm.get_metadata())
            assert len(tensors) == 1  # trunk-ignore(bandit/B101)
            assert tensors[0].shape == (4, 5)  # trunk-ignore(bandit/B101)
            handle.close()
        finally:
            shm.close()
            shm.unlink()
