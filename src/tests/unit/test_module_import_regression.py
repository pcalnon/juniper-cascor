#!/usr/bin/env python
"""Regression tests for module-level import bugs.

These tests guard against two specific regressions that previously broke
~1,354 tests across the suite:

  1. ``cascade_correlation.cascade_correlation`` referenced
     ``_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT`` inside
     ``_init_activation_function`` but the symbol was missing from the
     ``cascor_constants.constants`` import block, so every
     ``CascadeCorrelationNetwork.__init__`` raised ``NameError`` once the
     activation-name fallback path executed.

  2. ``snapshots.snapshot_serializer._create_network_from_file`` imported
     ``CascadeCorrelationConfig`` from the stale top-level path
     ``cascade_correlation_config.cascade_correlation_config`` instead of
     ``cascade_correlation.cascade_correlation_config.cascade_correlation_config``.
     The ``ImportError`` was swallowed by the surrounding ``try/except``,
     so ``load_network`` silently returned ``None`` on every legacy-format
     load.  The same stale path was also present in
     ``CascadeCorrelationNetwork._get_optimizer`` for ``OptimizerConfig``.

Both defects are invisible to static linters because the bad symbols live
inside ``try/except`` blocks or fallback branches that are not exercised
on the happy path of most callers.
"""

import h5py
import numpy as np
import pytest

from cascade_correlation import cascade_correlation as cc_mod
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import (
    CascadeCorrelationConfig,
    OptimizerConfig,
)
from snapshots.snapshot_serializer import CascadeHDF5Serializer

pytestmark = pytest.mark.unit


def test_activation_function_default_is_imported_into_module() -> None:
    """The default activation constant must be in cascade_correlation's namespace.

    Guards regression 1: prevents ``_init_activation_function`` from
    raising ``NameError`` when the user-supplied activation function name
    is falsy and the implementation falls back to the module-level default.
    """
    assert hasattr(cc_mod, "_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT")
    assert cc_mod._CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT is not None


def test_network_constructs_with_default_activation() -> None:
    """Constructing a network must execute the activation-default fallback.

    Guards regression 1 end-to-end: if the import is missing again,
    ``__init__`` raises ``NameError`` from ``_init_activation_function``.
    """
    config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
    network = CascadeCorrelationNetwork(config=config)
    assert network.activation_function_name


def test_create_network_from_file_legacy_format_returns_network(tmp_path) -> None:
    """``_create_network_from_file`` must successfully import its deps.

    Guards regression 2: writes a minimal legacy-format HDF5 (no
    ``config_json``, just attrs) and asserts that ``load_network``
    returns a real network rather than swallowing an ``ImportError`` and
    returning ``None``.
    """
    filepath = tmp_path / "legacy_format.h5"
    with h5py.File(filepath, "w") as f:
        f.attrs["format"] = "juniper.cascor"
        f.attrs["format_version"] = "1"
        meta = f.create_group("meta")
        meta.attrs["uuid"] = np.bytes_("regression-uuid")
        config_group = f.create_group("config")
        config_group.attrs["input_size"] = 2
        config_group.attrs["output_size"] = 1
        config_group.attrs["learning_rate"] = 0.1
        f.create_group("arch")
        f.create_group("random")
        params = f.create_group("params")
        output_layer = params.create_group("output_layer")
        output_layer.create_dataset("weights", data=np.random.randn(2, 1).astype(np.float32))
        output_layer.create_dataset("bias", data=np.random.randn(1).astype(np.float32))

    serializer = CascadeHDF5Serializer()
    loaded = serializer.load_network(str(filepath), CascadeCorrelationNetwork)
    assert loaded is not None
    assert isinstance(loaded, CascadeCorrelationNetwork)


def test_create_optimizer_uses_valid_optimizer_config_path() -> None:
    """``_create_optimizer`` must import ``OptimizerConfig`` from the real path.

    Guards regression 2 (sibling site): the function-local import in
    ``CascadeCorrelationNetwork._create_optimizer`` previously pointed at
    the stale top-level module path.  This test forces execution of
    that import by calling the method directly.
    """
    import torch

    config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=1)
    network = CascadeCorrelationNetwork(config=config)
    params = [torch.nn.Parameter(torch.zeros(2, 1))]
    optimizer = network._create_optimizer(params, OptimizerConfig(learning_rate=0.1, optimizer_type="SGD"))
    assert optimizer is not None
