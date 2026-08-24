"""
D-B: a CORRUPT snapshot must be distinguishable from an ABSENT one.

Both used to surface as ``404 "not found or failed to load"``, fusing two opposite
operator situations — *pick a different snapshot* and *investigate data loss* — across
all four verbs (restore / retrain / resume / replay).

Design of record:
``juniper-ml/notes/JUNIPER_2026-08-20_JUNIPER-CASCOR_SNAPSHOT-ERROR-TAXONOMY-DESIGN.md``.

The arms that matter:
- absent -> 404 / ``SNAPSHOT_ABSENT``
- corrupt -> 422 / ``SNAPSHOT_CORRUPT``   (the arm that fails against the old code)
- a VALID snapshot still loads on all four verbs (negative control — without it, a
  change that classified everything as corrupt would pass every other arm)
- the classification is per-verb, because the fusion was duplicated four times and a
  fix applied to three is the likeliest way this ships half-done.
"""

import os
import sys

import h5py
import pytest
import torch
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.app import create_app  # noqa: E402
from api.settings import Settings  # noqa: E402
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork  # noqa: E402
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig  # noqa: E402
from snapshots.snapshot_load_status import SNAPSHOT_ABSENT, SNAPSHOT_CORRUPT, SNAPSHOT_OK  # noqa: E402
from snapshots.snapshot_serializer import CascadeHDF5Serializer  # noqa: E402

pytestmark = pytest.mark.unit

VERBS = ("restore", "retrain", "resume", "replay")


@pytest.fixture
def client():
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def snapshots_dir(client, tmp_path, monkeypatch):
    """Point the lifecycle at an empty temp snapshots directory."""
    monkeypatch.setattr(client.app.state.lifecycle, "_get_snapshots_dir", lambda: tmp_path, raising=True)
    return tmp_path


def _write_valid_snapshot(directory, snapshot_id):
    """A real, loadable snapshot written by the production serializer."""
    config = CascadeCorrelationConfig(input_size=2, output_size=2, random_seed=42)
    network = CascadeCorrelationNetwork(config=config)
    network.train_output_layer(torch.randn(8, 2), torch.randn(8, 2), epochs=2)
    path = directory / f"{snapshot_id}.h5"
    assert CascadeHDF5Serializer().save_network(network, str(path))
    return path


def _write_corrupt_snapshot(directory, snapshot_id):
    """A file that EXISTS but cannot be deserialized.

    Valid HDF5 with a wrong ``format`` attribute, so it fails at the format gate rather
    than at the file-open — the case most likely to be mistaken for "absent".
    """
    path = directory / f"{snapshot_id}.h5"
    with h5py.File(path, "w") as hf:
        hf.attrs["format"] = "not-a-juniper-snapshot"
        hf.attrs["format_version"] = "1"
    return path


def _write_unreadable_snapshot(directory, snapshot_id):
    """A file that is not HDF5 at all — fails inside the open, not at the format gate."""
    path = directory / f"{snapshot_id}.h5"
    path.write_bytes(b"this is not an HDF5 file")
    return path


class TestSerializerClassification:
    """``load_network_result`` is where the reason still exists."""

    def test_absent_is_classified_absent(self, tmp_path):
        result = CascadeHDF5Serializer().load_network_result(tmp_path / "nope.h5")
        assert result.status == SNAPSHOT_ABSENT
        assert not result

    def test_bad_format_is_classified_corrupt(self, tmp_path):
        path = _write_corrupt_snapshot(tmp_path, "bad-format")
        result = CascadeHDF5Serializer().load_network_result(path)
        assert result.status == SNAPSHOT_CORRUPT
        assert not result
        # The reason names the failing check rather than "failed to load".
        assert "format" in result.detail.lower()

    def test_unreadable_file_is_classified_corrupt(self, tmp_path):
        path = _write_unreadable_snapshot(tmp_path, "not-hdf5")
        result = CascadeHDF5Serializer().load_network_result(path)
        assert result.status == SNAPSHOT_CORRUPT
        assert not result

    def test_valid_snapshot_loads(self, tmp_path):
        path = _write_valid_snapshot(tmp_path, "good")
        result = CascadeHDF5Serializer().load_network_result(path)
        assert result.status == SNAPSHOT_OK
        assert result
        assert result.network is not None


class TestVerifySavedNetworkNamesTheFailingCheck:
    """``verify_saved_network`` blamed the format string for every structural failure."""

    def test_missing_group_is_not_reported_as_invalid_format(self, tmp_path):
        path = tmp_path / "missing-groups.h5"
        with h5py.File(path, "w") as hf:
            # Correct format identifier; the model payload is what's missing.
            hf.attrs["format"] = "juniper.cascor"
            hf.attrs["format_version"] = "2"

        result = CascadeHDF5Serializer().verify_saved_network(str(path))

        assert result["valid"] is False
        assert "Missing required group" in result["error"], "the operator was pointed at the format string, which was fine"

    def test_genuinely_bad_format_still_says_so(self, tmp_path):
        path = _write_corrupt_snapshot(tmp_path, "bad-format")
        result = CascadeHDF5Serializer().verify_saved_network(str(path))
        assert result["valid"] is False
        assert "Invalid format" in result["error"]

    def test_absent_format_attribute_is_not_reported_as_invalid(self, tmp_path):
        """An ABSENT ``format`` is a different failure from a WRONG one.

        ``read_str_attr`` returns None when the attribute does not exist, and the
        rejection branch rendered that straight into the message -- so the six
        emptiest files in the archive were each reported as ``Invalid format: None``,
        naming a format that does not exist rather than the attribute that is missing.
        The distinction matters because the two have different causes: a wrong value
        means some other writer stamped it, an absent one means the write died before
        stamping anything.
        """
        path = tmp_path / "no-format-attr.h5"
        with h5py.File(path, "w") as hf:
            # Deliberately stamp NOTHING -- this is the truncated-write signature.
            hf.attrs["format_version"] = "1"

        result = CascadeHDF5Serializer().verify_saved_network(str(path))

        assert result["valid"] is False
        assert "Missing required attribute: format" in result["error"]
        assert "None" not in result["error"], "the absent attribute must be named, not rendered as a phantom format value"


class TestRouteStatusMapping:
    """The four verbs must agree, because the fusion was duplicated four times."""

    @pytest.mark.parametrize("verb", VERBS)
    def test_absent_snapshot_is_404(self, client, snapshots_dir, verb):
        response = client.post(f"/v1/snapshots/nonexistent/{verb}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "SNAPSHOT_ABSENT"

    @pytest.mark.parametrize("verb", VERBS)
    def test_corrupt_snapshot_is_422(self, client, snapshots_dir, verb):
        """The arm that fails against the pre-D-B code, which returned 404 here."""
        _write_corrupt_snapshot(snapshots_dir, "corrupt-snap")

        response = client.post(f"/v1/snapshots/corrupt-snap/{verb}")

        assert response.status_code == 422, "a snapshot that EXISTS but cannot be read must not be reported as missing"
        assert response.json()["error"]["code"] == "SNAPSHOT_CORRUPT"
        # The prose names the failing check, not just "failed to load".
        assert "could not be read" in response.json()["error"]["message"]

    @pytest.mark.parametrize("verb", VERBS)
    def test_unreadable_snapshot_is_422(self, client, snapshots_dir, verb):
        """Classification must not be keyed to one specific failure mode: this one
        fails inside the file open rather than at the format gate."""
        _write_unreadable_snapshot(snapshots_dir, "unreadable-snap")

        response = client.post(f"/v1/snapshots/unreadable-snap/{verb}")

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "SNAPSHOT_CORRUPT"

    @pytest.mark.parametrize("verb", VERBS)
    def test_valid_snapshot_still_succeeds(self, client, snapshots_dir, verb):
        """Negative control. Without it, a change that classified EVERYTHING as
        corrupt would pass every other arm in this file."""
        _write_valid_snapshot(snapshots_dir, "good-snap")

        response = client.post(f"/v1/snapshots/good-snap/{verb}")

        assert response.status_code == 200, response.text
        assert response.json()["data"]["snapshot_id"] == "good-snap"


class TestStartReplayConverged:
    """``start_replay`` returned a bare bool, so replay was the one verb that could
    not carry a reason even in principle (design §6)."""

    def test_returns_the_snapshot_result_shape(self, client, snapshots_dir):
        result = client.app.state.lifecycle.start_replay("nonexistent")
        assert isinstance(result, dict), "start_replay must return the same dict as its sibling verbs"
        assert result["loaded"] is False
        assert result["operation"] == "replay"
        assert result["reason_code"] == SNAPSHOT_ABSENT

    def test_carries_the_corrupt_reason(self, client, snapshots_dir):
        _write_corrupt_snapshot(snapshots_dir, "corrupt-snap")
        result = client.app.state.lifecycle.start_replay("corrupt-snap")
        assert result["loaded"] is False
        assert result["reason_code"] == SNAPSHOT_CORRUPT

    def test_fsm_rejection_is_not_a_load_failure(self, client, snapshots_dir):
        """An FSM conflict still has no ``reason_code`` — it is a 409 at the route,
        and must not be mistaken for a missing snapshot."""
        from api.lifecycle.state_machine import Command

        lifecycle = client.app.state.lifecycle
        lifecycle.create_network(input_size=2, output_size=2)
        lifecycle.state_machine.handle_command(Command.START)
        try:
            result = lifecycle.start_replay("anything")
            assert result["loaded"] is False
            assert result["reason_code"] is None
            assert "rejected" in result["reason"]
        finally:
            lifecycle.state_machine.handle_command(Command.RESET)
