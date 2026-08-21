"""
D-E: the snapshot loader's integrity gates must actually STOP a bad load.

``load_network`` ran six checks and enforced none of them: it logged (two at ERROR),
returned the network anyway, and then logged ``Successfully loaded network`` on the
next line. The network was installed on the live lifecycle and the operator was told
the restore succeeded.

Why that is worse than a loud failure: three shape-violation classes raise later, but a
hidden-unit weight vector of length 1 is *broadcast-compatible* with the slice it
multiplies -- so the network computes a different answer with no error anywhere, trains,
reports a plausible loss, and can be re-snapshotted, propagating the corruption.

Design of record:
``juniper-ml/notes/JUNIPER_2026-08-20_JUNIPER-CASCOR_SNAPSHOT-INTEGRITY-GATES-DESIGN.md``.
Owner decisions (2026-08-21): all gates fail-closed including checksums; the forensic
escape hatch is a serializer parameter only (no API surface); an arch disagreement gets
its own reason code at the same 422; existing archive files are rejected rather than
grandfathered.
"""

import json
import os
import sys

import h5py
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.app import create_app  # noqa: E402
from api.settings import Settings  # noqa: E402
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork  # noqa: E402
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig  # noqa: E402
from snapshots.snapshot_load_status import SNAPSHOT_ARCH_MISMATCH, SNAPSHOT_CORRUPT, SNAPSHOT_OK  # noqa: E402
from snapshots.snapshot_serializer import CascadeHDF5Serializer  # noqa: E402

pytestmark = pytest.mark.unit

VERBS = ("restore", "retrain", "resume", "replay")
INPUT_SIZE = 3
OUTPUT_SIZE = 2


@pytest.fixture
def client():
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def snapshots_dir(client, tmp_path, monkeypatch):
    monkeypatch.setattr(client.app.state.lifecycle, "_get_snapshots_dir", lambda: tmp_path, raising=True)
    return tmp_path


def write_valid(directory, snapshot_id="good", n_hidden=2):
    """A real snapshot, grown through the same helpers the cascade path uses."""
    torch.manual_seed(42)
    config = CascadeCorrelationConfig(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, random_seed=42)
    network = CascadeCorrelationNetwork(config=config)
    for _ in range(n_hidden):
        prev_in = network.output_weights.shape[0]
        network._install_hidden_unit(
            weights=torch.randn(prev_in, dtype=torch.float32),
            bias=torch.tensor([0.0], dtype=torch.float32),
            activation_fn=network.activation_fn,
            correlation=0.5,
        )
        network._resize_output_layer_for_new_units(num_added=1, prev_input_size=prev_in)
    network.train_output_layer(torch.randn(8, INPUT_SIZE), torch.randn(8, OUTPUT_SIZE), epochs=2)
    path = directory / f"{snapshot_id}.h5"
    assert CascadeHDF5Serializer().save_network(network, str(path))
    return path


def _replace(path, dataset, array):
    with h5py.File(path, "a") as hf:
        del hf[dataset]
        hf.create_dataset(dataset, data=array)


def break_output_shape(path):
    """Gate 5, loud class: output_weights loses a row. Raises later if it loads."""
    with h5py.File(path, "r") as hf:
        rows, cols = hf["params/output_layer/weights"].shape
    _replace(path, "params/output_layer/weights", np.zeros((rows - 1, cols), dtype=np.float32))


def break_hidden_shape_broadcast(path):
    """Gate 5, SILENT class: a length-1 hidden weight vector broadcasts cleanly.

    This is the one that trains on garbage without raising, so it is the arm that
    proves the gate does something a later exception would not.
    """
    _replace(path, "hidden_units/unit_1/weights", np.ones((1,), dtype=np.float32))


def break_arch(path, output_size=OUTPUT_SIZE + 1):
    """Gates 1-2: the arch group disagrees with the config the network is built from."""
    with h5py.File(path, "a") as hf:
        hf["arch"].attrs["output_size"] = np.int64(output_size)


def break_checksum(path):
    """Gates 3-4: same-shape different bytes, with the original checksum left intact."""
    with h5py.File(path, "r") as hf:
        shape = hf["params/output_layer/weights"].shape
        assert "checksums" in hf["params/output_layer"], "fixture needs a snapshot that stores checksums"
    _replace(path, "params/output_layer/weights", np.full(shape, 7.5, dtype=np.float32))


class TestGatesRejectAtLoad:
    """Every gate is fail-closed now; none of them were before."""

    def test_valid_snapshot_still_loads(self, tmp_path):
        """Negative control. Without it, a change that rejected EVERYTHING would pass
        every other arm in this file."""
        path = write_valid(tmp_path)
        result = CascadeHDF5Serializer().load_network_result(path, restore_multiprocessing=False)
        assert result.status == SNAPSHOT_OK
        assert result.network is not None

    def test_output_shape_violation_is_refused(self, tmp_path):
        path = write_valid(tmp_path)
        break_output_shape(path)
        assert CascadeHDF5Serializer().load_network(path, restore_multiprocessing=False) is None

    def test_broadcast_compatible_violation_is_refused(self, tmp_path):
        """THE arm that matters: this one used to load, train, and report a plausible
        loss while computing a different answer."""
        path = write_valid(tmp_path)
        break_hidden_shape_broadcast(path)

        serializer = CascadeHDF5Serializer()
        assert serializer.load_network(path, restore_multiprocessing=False) is None, "a silently-wrong network must not reach the caller"
        assert serializer.load_network_result(path, restore_multiprocessing=False).status == SNAPSHOT_CORRUPT

    def test_checksum_mismatch_is_refused(self, tmp_path):
        """A checksum mismatch is positive evidence of corruption. It was logged at
        ERROR and then ignored."""
        path = write_valid(tmp_path)
        break_checksum(path)

        serializer = CascadeHDF5Serializer()
        assert serializer.load_network(path, restore_multiprocessing=False) is None
        result = serializer.load_network_result(path, restore_multiprocessing=False)
        assert result.status == SNAPSHOT_CORRUPT
        assert "checksum" in result.detail.lower()

    def test_arch_mismatch_is_refused_with_its_own_reason(self, tmp_path):
        path = write_valid(tmp_path)
        break_arch(path)

        serializer = CascadeHDF5Serializer()
        assert serializer.load_network(path, restore_multiprocessing=False) is None
        result = serializer.load_network_result(path, restore_multiprocessing=False)
        assert result.status == SNAPSHOT_ARCH_MISMATCH, "an arch disagreement is a different investigation from damage"
        assert "output_size" in result.detail


class TestForensicOptIn:
    """The escape hatch is library-only, by decision — no API surface reaches it."""

    @pytest.mark.parametrize("break_it", [break_output_shape, break_hidden_shape_broadcast, break_arch, break_checksum])
    def test_allow_invalid_loads_anyway(self, tmp_path, break_it):
        path = write_valid(tmp_path)
        break_it(path)
        network = CascadeHDF5Serializer().load_network(path, restore_multiprocessing=False, allow_invalid=True)
        assert network is not None, "forensic inspection of a rejected snapshot must remain possible"

    def test_allow_invalid_is_not_the_default(self, tmp_path):
        """A caller has to ask for it explicitly — the whole point of fail-closed."""
        path = write_valid(tmp_path)
        break_arch(path)
        assert CascadeHDF5Serializer().load_network(path, restore_multiprocessing=False) is None

    def test_no_api_route_exposes_the_hatch(self, client, snapshots_dir):
        """Decision: the hatch stays out of the service tier, so a knowingly-broken
        network can never be put on the live lifecycle from a URL."""
        path = write_valid(snapshots_dir, "broken-snap")
        break_arch(path)

        for query in ("", "?allow_invalid=true", "?allow_invalid=1"):
            response = client.post(f"/v1/snapshots/broken-snap/restore{query}")
            assert response.status_code == 422, f"query {query!r} must not unlock a rejected snapshot"


class TestRouteMapping:
    """All four verbs, because the load path is shared but the mapping is per-route."""

    @pytest.mark.parametrize("verb", VERBS)
    def test_shape_violation_is_422_corrupt(self, client, snapshots_dir, verb):
        path = write_valid(snapshots_dir, "shape-broken")
        break_hidden_shape_broadcast(path)

        response = client.post(f"/v1/snapshots/shape-broken/{verb}")

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "SNAPSHOT_CORRUPT"

    @pytest.mark.parametrize("verb", VERBS)
    def test_arch_mismatch_is_422_with_its_own_code(self, client, snapshots_dir, verb):
        path = write_valid(snapshots_dir, "arch-broken")
        break_arch(path)

        response = client.post(f"/v1/snapshots/arch-broken/{verb}")

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "SNAPSHOT_ARCH_MISMATCH"
        assert "describes a different network" in response.json()["error"]["message"]

    @pytest.mark.parametrize("verb", VERBS)
    def test_valid_snapshot_still_succeeds(self, client, snapshots_dir, verb):
        """Negative control at the route layer."""
        write_valid(snapshots_dir, "good-snap")
        response = client.post(f"/v1/snapshots/good-snap/{verb}")
        assert response.status_code == 200, response.text

    def test_absent_is_still_404(self, client, snapshots_dir):
        """D-E must not regress D-B: absent stays distinguishable from unusable."""
        response = client.post("/v1/snapshots/nope/restore")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "SNAPSHOT_ABSENT"


class TestCheckIntegrityUnit:
    """The gate collector itself, so a future edit cannot quietly drop a gate."""

    def test_clean_snapshot_yields_no_findings(self, tmp_path):
        path = write_valid(tmp_path)
        serializer = CascadeHDF5Serializer()
        network = serializer.load_network(path, restore_multiprocessing=False)
        with h5py.File(path, "r") as hf:
            assert serializer._check_integrity(hf, network) == []

    def test_every_gate_is_represented(self, tmp_path):
        """One finding per broken gate, each reachable independently."""
        serializer = CascadeHDF5Serializer()
        for break_it, expected in (
            (break_arch, SNAPSHOT_ARCH_MISMATCH),
            (break_checksum, SNAPSHOT_CORRUPT),
            (break_hidden_shape_broadcast, SNAPSHOT_CORRUPT),
        ):
            path = write_valid(tmp_path, f"g-{break_it.__name__}")
            break_it(path)
            network = serializer.load_network(path, restore_multiprocessing=False, allow_invalid=True)
            with h5py.File(path, "r") as hf:
                findings = serializer._check_integrity(hf, network)
            assert findings, f"{break_it.__name__} produced no finding"
            assert expected in {status for status, _ in findings}, f"{break_it.__name__} -> {findings}"

    def test_checksums_are_verified_once(self, tmp_path):
        """The verification moved out of ``_load_parameters`` so a successful load does
        not re-hash the output tensors. Guards against it being restored there."""
        source = (CascadeHDF5Serializer.__module__, "_load_parameters")
        import inspect

        body = inspect.getsource(CascadeHDF5Serializer._load_parameters)
        assert "verify_tensor_checksum" not in body, f"checksum verification is back in {source[1]} — it now lives in _check_integrity"


class TestSnapshotHistoryStillWritten:
    """A rejected load must not look like a successful one anywhere."""

    def test_rejected_load_reports_not_loaded(self, client, snapshots_dir):
        path = write_valid(snapshots_dir, "reject-me")
        break_checksum(path)
        result = client.app.state.lifecycle.load_snapshot("reject-me")
        assert result["loaded"] is False
        assert result["reason_code"] == SNAPSHOT_CORRUPT
        assert json.dumps(result), "result must stay JSON-serialisable for the route payload"
