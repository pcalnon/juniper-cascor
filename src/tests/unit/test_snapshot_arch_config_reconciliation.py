"""
When ``config_json`` and ``arch`` disagree about a structural dimension, ``arch`` wins.

A snapshot records ``output_size`` in FOUR places from TWO sources:

    arch.attrs           <- the LIVE network
    config.attrs         <- the LIVE network
    params/output_layer  <- the LIVE tensors
    config/config_json   <- the CONFIG OBJECT

``_resize_network_for_dataset`` (the live dataset swap) assigns ``self.output_size`` and
never touches ``self.config``; nothing in the cascor tree ever assigns
``config.output_size``. So once a network is resized its config object is permanently
stale, and every snapshot taken afterwards carries the contradiction.

``_create_network_from_file`` rebuilt from ``config_json`` -- the one stale copy -- so the
network came out NARROWER than the tensors about to be loaded into it, and the arch gate
correctly refused the whole snapshot. Measured 2026-08-22 (juniper-ml#1254): **239**
archive snapshots, every one intact; loaded permissively, 5/5 sampled produced a
self-consistent network that infers.

WHY THE RECONCILIATION IS CONDITIONAL
    "arch wins" is only justified because on these snapshots the TENSORS agree with arch
    and only ``config_json`` dissents. So the loader verifies that rather than assuming it:
    it adopts arch only when ``params/output_layer`` actually has the geometry arch claims.

    An UNCONDITIONAL preference was written first, and it was wrong. Building the network
    from arch means it can never disagree with arch, which makes ``SNAPSHOT_ARCH_MISMATCH``
    unreachable -- retiring a status D-E added deliberately, that four API routes map to
    their own 422 code, and that exists to tell an operator an arch disagreement is a
    different investigation from damage. A deliberately corrupted arch was reported as
    generic shape damage instead. Six tests in
    ``tests/unit/api/test_snapshot_integrity_gates.py`` caught it.

    ``TestArchIsStillCheckedAgainstTheTensors`` is the load-bearing class in this file, and
    it asserts the specific STATUS rather than merely "not ok" -- the weaker assertion would
    have passed against the broken design.
"""

import json
import os
import sys

import h5py
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork  # noqa: E402
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig  # noqa: E402
from snapshots.snapshot_load_status import SNAPSHOT_ARCH_MISMATCH, SNAPSHOT_OK  # noqa: E402
from snapshots.snapshot_serializer import CascadeHDF5Serializer  # noqa: E402

pytestmark = pytest.mark.unit


def _save(tmp_path, output_size=2, name="recon"):
    """Write a healthy snapshot at ``output_size``; arch, config attrs and tensors agree."""
    torch.manual_seed(13)
    config = CascadeCorrelationConfig(input_size=2, output_size=output_size, random_seed=13)
    network = CascadeCorrelationNetwork(config=config)
    path = tmp_path / f"{name}.h5"
    assert CascadeHDF5Serializer().save_network(network, str(path))
    return path


def _stale_config_json(path, field, stale_value):
    """Rewrite only ``config_json``, leaving arch / config attrs / tensors alone.

    This reproduces the archive shape exactly: the live network really was resized, so
    everything written from it is at the NEW width and only the config object lagged.
    """
    with h5py.File(path, "r") as handle:
        raw = handle["config"]["config_json"][()]
    config = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    config[field] = stale_value
    with h5py.File(path, "a") as handle:
        del handle["config"]["config_json"]
        handle["config"].create_dataset("config_json", data=json.dumps(config))


def _set_arch(path, field, value):
    with h5py.File(path, "a") as handle:
        handle["arch"].attrs[field] = value


def _record_warnings(serializer):
    """cascor's ``Logger`` does not propagate to root, so ``caplog`` sees nothing from it
    and an assertion through caplog would pass whatever the code did."""
    captured = []
    original = serializer.logger.warning

    def _spy(message, *args, **kwargs):
        captured.append(str(message))
        return original(message, *args, **kwargs)

    serializer.logger.warning = _spy
    return captured


class TestStaleConfigJsonIsReconciled:
    def test_the_archive_case_loads(self, tmp_path):
        """arch/tensors at 3, config_json stale at 2 -- the exact 239-snapshot shape."""
        path = _save(tmp_path, output_size=3)
        _stale_config_json(path, "output_size", 2)
        result = CascadeHDF5Serializer().load_network_result(str(path), restore_multiprocessing=False)
        assert result.status == SNAPSHOT_OK, f"expected a clean load, got {result.status}: {result.detail}"
        assert result.network.output_size == 3, "the network must be rebuilt at the width its tensors actually have"

    def test_the_recovered_network_matches_its_tensors(self, tmp_path):
        path = _save(tmp_path, output_size=3)
        _stale_config_json(path, "output_size", 2)
        network = CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False)
        expected_rows = network.input_size + len(network.hidden_units)
        assert tuple(network.output_weights.shape) == (expected_rows, 3)
        assert tuple(network.output_bias.shape) == (3,)

    def test_the_recovered_network_can_infer(self, tmp_path):
        """The point of recovering them: 5/5 sampled archive specimens produced finite output."""
        path = _save(tmp_path, output_size=3)
        _stale_config_json(path, "output_size", 2)
        network = CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False)
        output = network.forward(torch.zeros(4, network.input_size))
        assert tuple(output.shape) == (4, 3)
        assert bool(torch.isfinite(output).all())

    def test_input_size_is_reconciled_too(self, tmp_path):
        """``_resize_network_for_dataset`` assigns both dimensions and stales both."""
        path = _save(tmp_path)
        _stale_config_json(path, "input_size", 1)
        network = CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False)
        assert network is not None
        assert network.input_size == 2

    def test_the_reconciliation_is_reported(self, tmp_path):
        """Silently rebuilding a network at a different width than its file says is how the
        NEXT investigation starts from nothing."""
        path = _save(tmp_path, output_size=3)
        _stale_config_json(path, "output_size", 2)
        serializer = CascadeHDF5Serializer()
        warnings = _record_warnings(serializer)
        serializer.load_network(str(path), restore_multiprocessing=False)
        assert any("output_size" in message and "arch" in message for message in warnings), f"the reconciliation must name the field and the source; got {warnings}"


class TestHealthySnapshotsAreUntouched:
    def test_agreeing_snapshot_logs_no_reconciliation(self, tmp_path):
        """The warning has to mean something when it appears -- 27,382 of 27,908 archive
        snapshots load cleanly and must stay silent."""
        path = _save(tmp_path)
        serializer = CascadeHDF5Serializer()
        warnings = _record_warnings(serializer)
        serializer.load_network(str(path), restore_multiprocessing=False)
        assert not any("disagrees with the arch group" in message for message in warnings)

    def test_agreeing_snapshot_still_loads_at_its_own_width(self, tmp_path):
        path = _save(tmp_path, output_size=2)
        network = CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False)
        assert network.output_size == 2


class TestArchIsStillCheckedAgainstTheTensors:
    """THE load-bearing class, and the reason reconciliation is CONDITIONAL.

    An unconditional "prefer arch" would build the network FROM arch, so it could never
    disagree with arch -- making ``SNAPSHOT_ARCH_MISMATCH`` unreachable. That status is not
    decoration: D-E added it deliberately, four API routes map it to their own 422 code, and
    it exists to tell an operator that an arch disagreement is a different investigation
    from damage. Reconciling unconditionally retires it and reports a corrupted arch as
    generic shape damage instead.

    Caught by ``tests/unit/api/test_snapshot_integrity_gates.py`` (6 failures) before this
    class existed. The gate below is what keeps that suite passing.
    """

    def test_arch_disagreeing_with_the_tensors_still_reports_arch_mismatch(self, tmp_path):
        """The status must survive, not just the refusal.

        Asserting only ``!= SNAPSHOT_OK`` would pass even if the reconciliation had
        swallowed the distinction and downgraded this to generic corruption -- which is
        exactly the regression this class exists to prevent.
        """
        path = _save(tmp_path, output_size=2)
        _set_arch(path, "output_size", 5)
        result = CascadeHDF5Serializer().load_network_result(str(path), restore_multiprocessing=False)
        assert result.status == SNAPSHOT_ARCH_MISMATCH, f"an arch/tensor disagreement must stay its own status, got {result.status}: {result.detail}"
        assert result.network is None
        assert "output_size" in result.detail

    def test_a_bogus_input_size_in_arch_reports_arch_mismatch(self, tmp_path):
        path = _save(tmp_path, output_size=2)
        _set_arch(path, "input_size", 9)
        result = CascadeHDF5Serializer().load_network_result(str(path), restore_multiprocessing=False)
        assert result.status == SNAPSHOT_ARCH_MISMATCH

    def test_an_uncorroborated_arch_is_not_reconciled_into_the_config(self, tmp_path):
        """The mechanism, pinned directly: when the tensors do not back arch, the stale
        config_json is left alone and the gates see the disagreement they exist to catch."""
        path = _save(tmp_path, output_size=2)
        _set_arch(path, "output_size", 5)
        serializer = CascadeHDF5Serializer()
        warnings = _record_warnings(serializer)
        serializer.load_network(str(path), restore_multiprocessing=False)
        assert not any("rebuilding from arch" in message for message in warnings), "arch must not be adopted without tensor corroboration"

    def test_a_non_numeric_arch_attr_does_not_crash_the_load(self, tmp_path):
        """A corrupt arch attr is a different fault; it must fall through to the gates
        rather than raising out of the config-reconciliation step."""
        path = _save(tmp_path, output_size=2)
        with h5py.File(path, "a") as handle:
            del handle["arch"].attrs["output_size"]
            handle["arch"].attrs["output_size"] = b"not-a-number"
        result = CascadeHDF5Serializer().load_network_result(str(path), restore_multiprocessing=False)
        assert result is not None, "a non-numeric arch attr must produce a verdict, not an exception"
