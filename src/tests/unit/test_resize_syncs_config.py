"""
A resized network must not carry a stale config object.

``input_size`` / ``output_size`` live in TWO places: on the network (authoritative at
runtime, and what the tensors are built to) and on ``self.config`` (what
``_save_configuration`` serializes into ``config_json``). They are read from the config once
at construction and, until 2026-08-23, never written again anywhere in the tree --
``_resize_network_for_dataset`` assigned ``self.output_size`` and left ``self.config``
untouched.

So a resized network carried a permanently stale config, and every snapshot taken
afterwards recorded the contradiction: ``arch`` and the tensors at the NEW width,
``config_json`` at the old one. Because the loader rebuilds from ``config_json``, it
produced a network narrower than the tensors about to be loaded into it and the arch
integrity gate refused the file.

Measured 2026-08-22 (juniper-ml#1254): **239 of 27,908** archive snapshots, all intact and
all unloadable. juniper-cascor#560 taught the LOADER to recover them by preferring ``arch``
where the tensors corroborate it; this is the WRITER-side half, so newly-written snapshots
never need that recovery.

THE LOAD-BEARING TEST IS ``TestTheArchiveDefectCannotRecur``. The unit-level assertions
above it would all pass against a fix that synced the config but broke the save path; only
the full resize -> save -> load round-trip proves the defect is actually gone.
"""

from __future__ import annotations

import json
import os
import sys

import h5py
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork  # noqa: E402
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig  # noqa: E402
from snapshots.snapshot_load_status import SNAPSHOT_OK  # noqa: E402
from snapshots.snapshot_serializer import CascadeHDF5Serializer  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def network():
    torch.manual_seed(19)
    config = CascadeCorrelationConfig(input_size=2, output_size=2, random_seed=19)
    return CascadeCorrelationNetwork(config=config)


def _resize(net, input_size, output_size):
    """Drive the real resize path — the operation that staled the config."""
    return net._resize_network_for_dataset(input_size_new=input_size, output_size_new=output_size)


def _stored_config_json(path):
    with h5py.File(path, "r") as handle:
        raw = handle["config"]["config_json"][()]
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


class TestResizeKeepsConfigInStep:
    def test_output_size_reaches_the_config(self, network):
        _resize(network, 2, 3)
        assert network.output_size == 3
        assert network.config.output_size == 3, "the config object must not lag the live network"

    def test_input_size_reaches_the_config(self, network):
        _resize(network, 4, 2)
        assert network.input_size == 4
        assert network.config.input_size == 4

    def test_both_dimensions_at_once(self, network):
        _resize(network, 5, 3)
        assert (network.config.input_size, network.config.output_size) == (5, 3)

    def test_a_no_op_resize_leaves_everything_alone(self, network):
        before = (network.config.input_size, network.config.output_size)
        _resize(network, 2, 2)
        assert (network.config.input_size, network.config.output_size) == before

    def test_successive_resizes_track(self, network):
        _resize(network, 3, 3)
        _resize(network, 6, 4)
        assert (network.config.input_size, network.config.output_size) == (6, 4)

    def test_a_config_without_the_attributes_does_not_break_the_resize(self, network):
        """Defensive: the sync must never turn a resize -- which runs on the live training
        path -- into a crash because a stubbed or unusual config lacks a field."""

        class Bare:
            pass

        network.config = Bare()
        _resize(network, 3, 3)  # must not raise
        assert network.output_size == 3
        assert not hasattr(network.config, "output_size"), "the sync must not invent attributes on a foreign config"


class TestTheArchiveDefectCannotRecur:
    """The full round-trip. This is the test that actually pins the defect.

    Reproduces the archive's 239-snapshot shape end to end: resize the network, save it,
    and confirm the snapshot no longer disagrees with itself.
    """

    def test_a_snapshot_written_after_a_resize_agrees_with_itself(self, network, tmp_path):
        _resize(network, 2, 3)
        path = tmp_path / "resized.h5"
        assert CascadeHDF5Serializer().save_network(network, str(path))

        stored = _stored_config_json(path)
        with h5py.File(path, "r") as handle:
            arch_output = int(handle["arch"].attrs["output_size"])
            weights_shape = tuple(handle["params"]["output_layer"]["weights"].shape)

        assert stored["output_size"] == arch_output == 3, f"config_json says {stored['output_size']}, arch says {arch_output}"
        assert weights_shape[1] == 3, "the tensors must agree too"

    def test_it_loads_without_needing_the_arch_recovery_path(self, network, tmp_path):
        """Before the writer fix this returned ``snapshot_arch_mismatch``. It must now load
        cleanly on its own, not by leaning on juniper-cascor#560's reconciliation."""
        _resize(network, 2, 3)
        path = tmp_path / "resized.h5"
        CascadeHDF5Serializer().save_network(network, str(path))

        serializer = CascadeHDF5Serializer()
        reconciled = []
        original_warning = serializer.logger.warning

        def _spy(message, *args, **kwargs):
            captured = str(message)
            if "rebuilding from arch" in captured:
                reconciled.append(captured)
            return original_warning(message, *args, **kwargs)

        serializer.logger.warning = _spy
        result = serializer.load_network_result(str(path), restore_multiprocessing=False)

        assert result.status == SNAPSHOT_OK, f"expected a clean load, got {result.status}: {result.detail}"
        assert result.network.output_size == 3
        assert reconciled == [], "a freshly-written snapshot must not need the loader's stale-config recovery"

    def test_the_recovered_network_infers_at_the_new_width(self, network, tmp_path):
        _resize(network, 2, 3)
        path = tmp_path / "resized.h5"
        CascadeHDF5Serializer().save_network(network, str(path))
        loaded = CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False)
        output = loaded.forward(torch.zeros(4, loaded.input_size))
        assert tuple(output.shape) == (4, 3)
        assert bool(torch.isfinite(output).all())

    def test_an_input_resize_round_trips_too(self, network, tmp_path):
        _resize(network, 5, 2)
        path = tmp_path / "wider.h5"
        CascadeHDF5Serializer().save_network(network, str(path))
        result = CascadeHDF5Serializer().load_network_result(str(path), restore_multiprocessing=False)
        assert result.status == SNAPSHOT_OK, f"{result.status}: {result.detail}"
        assert result.network.input_size == 5
