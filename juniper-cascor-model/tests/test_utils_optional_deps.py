#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     test_utils_optional_deps.py
# Author:        Paul Calnon
#
# Date Created:  2026-07-03
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#     Coverage for the optional-dependency edges of the candidate-core utils layer:
#     the ``columnar`` import-guard fallback, the ``check_object_pickleability`` dill
#     ImportError path, and the 0-dimensional Softmax activation guard. These exercise
#     the branches that only run when an optional helper is absent (or a scalar pre-
#     activation is passed), which the happy-path suites cannot reach. Part of the
#     juniper-cascor-model per-file coverage rollout (C-5).
#
#####################################################################################################################################################################################################

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")


class TestColumnarImportGuard:
    """The module-level columnar import must degrade to the string-formatting fallback."""

    def test_reload_without_columnar_sets_fallback_flags(self):
        import utils.utils as utils_module

        # Force the ``import columnar`` at module import time to fail, then reload so the
        # except branch (HAS_COLUMNAR = False; col = None) executes. Always reload back to
        # the real module state so downstream tests keep the genuine columnar binding.
        try:
            with patch.dict(sys.modules, {"columnar": None}):
                importlib.reload(utils_module)
                assert utils_module.HAS_COLUMNAR is False
                assert utils_module.col is None
        finally:
            importlib.reload(utils_module)

        assert utils_module.HAS_COLUMNAR is True


class TestCheckObjectPickleabilityDillGuard:
    """check_object_pickleability must raise a clear ImportError when dill is unavailable."""

    def test_missing_dill_raises_actionable_import_error(self):
        from utils.utils import check_object_pickleability

        class _Holder:
            def __init__(self):
                self.value = 1

        # Masking ``dill`` in sys.modules makes the in-function ``import dill`` fail.
        with patch.dict(sys.modules, {"dill": None}):
            with pytest.raises(ImportError, match="requires the 'dill' package"):
                check_object_pickleability(_Holder())


class TestSoftmaxScalarGuard:
    """The Softmax activation wrapper must tolerate a 0-dimensional pre-activation."""

    def test_zero_dim_softmax_returns_ones_like(self):
        from utils.activation import ActivationWithDerivative

        wrapper = ActivationWithDerivative(torch.nn.Softmax(dim=1))
        scalar = torch.tensor(2.0)

        result = wrapper(scalar)

        assert result.dim() == 0
        assert torch.equal(result, torch.ones_like(scalar))
