"""
D-C: a snapshot records WHICH RUN produced it.

Before this, a snapshot said what it was (architecture, parameters, library versions)
but never where it came from -- ``meta.uuid`` identifies the file, not the work. So
"find the model from the E-I cap-128 cell" was unanswerable across ~27.9k files, and
identity has to precede retention: a deletion rule over anonymous artifacts is guesswork.

Owner decisions (2026-08-21): transport is process env (mirroring
``JUNIPER_CASCOR_SNAPSHOTS_DIR``); the data lives in its own top-level ``provenance``
group; existing snapshots are left alone with absence meaning *unknown*; the field set is
run_id + experiment + cell_id + dataset_id + git_sha.

The load-bearing arm in this file is
``TestAbsentProvenanceStaysLegal::test_snapshot_without_provenance_still_loads`` -- if
``provenance`` ever became a required group, every one of the ~27.9k archived snapshots
would stop loading at once.
"""

import os
import sys

import h5py
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork  # noqa: E402
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig  # noqa: E402
from snapshots.snapshot_provenance import (  # noqa: E402
    PROVENANCE_ENV,
    PROVENANCE_FIELDS,
    PROVENANCE_GROUP,
    PROVENANCE_SCHEMA_VERSION,
    capture_from_env,
    read_provenance,
    write_provenance,
)
from snapshots.snapshot_serializer import CascadeHDF5Serializer  # noqa: E402

pytestmark = pytest.mark.unit

FULL_ENV = {
    "JUNIPER_CASCOR_RUN_ID": "run-20260821-0001",
    "JUNIPER_CASCOR_EXPERIMENT": "e-i-cap-sweep",
    "JUNIPER_CASCOR_CELL_ID": "c007-9f3ab12c",
    "JUNIPER_CASCOR_DATASET_ID": "spiral-2d-seed42",
    "JUNIPER_CASCOR_GIT_SHA": "3d6aa81e",
}


def _network():
    torch.manual_seed(7)
    config = CascadeCorrelationConfig(input_size=2, output_size=2, random_seed=7)
    network = CascadeCorrelationNetwork(config=config)
    network.train_output_layer(torch.randn(8, 2), torch.randn(8, 2), epochs=2)
    return network


def _save(directory, name="snap", env=None, monkeypatch=None):
    if env is not None:
        for key in PROVENANCE_ENV.values():
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
    path = directory / f"{name}.h5"
    assert CascadeHDF5Serializer().save_network(_network(), str(path))
    return path


class TestCaptureFromEnv:
    """Capture is injectable and treats blank as unset."""

    def test_captures_every_field(self):
        captured = capture_from_env(FULL_ENV)
        assert set(captured) == set(PROVENANCE_FIELDS)
        assert captured["cell_id"] == "c007-9f3ab12c"

    def test_partial_run_records_what_it_knows(self):
        captured = capture_from_env({"JUNIPER_CASCOR_RUN_ID": "solo"})
        assert captured == {"run_id": "solo"}, "a partially-identified run must not emit empty strings for the rest"

    def test_blank_is_treated_as_unset(self):
        captured = capture_from_env({"JUNIPER_CASCOR_RUN_ID": "   ", "JUNIPER_CASCOR_EXPERIMENT": ""})
        assert captured == {}

    def test_unidentified_run_captures_nothing(self):
        assert capture_from_env({}) == {}

    def test_reads_at_call_time_not_import_time(self, monkeypatch):
        """A long-lived process must see a value exported after it started."""
        monkeypatch.delenv("JUNIPER_CASCOR_RUN_ID", raising=False)
        assert "run_id" not in capture_from_env()
        monkeypatch.setenv("JUNIPER_CASCOR_RUN_ID", "late-arrival")
        assert capture_from_env()["run_id"] == "late-arrival"


class TestRoundTrip:
    """Provenance survives a real save -> load through the production serializer."""

    def test_full_provenance_round_trips(self, tmp_path, monkeypatch):
        path = _save(tmp_path, env=FULL_ENV, monkeypatch=monkeypatch)

        loaded = CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False)

        assert loaded is not None
        assert loaded.provenance is not None
        for field, env_var in PROVENANCE_ENV.items():
            assert loaded.provenance[field] == FULL_ENV[env_var]
        assert loaded.provenance["schema_version"] == PROVENANCE_SCHEMA_VERSION

    def test_answers_the_question_the_census_could_not(self, tmp_path, monkeypatch):
        """'Find the model from the E-I cap-128 cell' -- the concrete D-C use case."""
        path = _save(tmp_path, env={**FULL_ENV, "JUNIPER_CASCOR_CELL_ID": "c128-capsweep"}, monkeypatch=monkeypatch)

        with h5py.File(path, "r") as hf:
            recovered = read_provenance(hf)

        assert recovered["cell_id"] == "c128-capsweep"
        assert recovered["experiment"] == "e-i-cap-sweep"

    def test_partial_provenance_round_trips(self, tmp_path, monkeypatch):
        path = _save(tmp_path, env={"JUNIPER_CASCOR_RUN_ID": "only-run"}, monkeypatch=monkeypatch)

        loaded = CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False)

        assert loaded.provenance["run_id"] == "only-run"
        assert "experiment" not in loaded.provenance


class TestAbsentProvenanceStaysLegal:
    """The ~27.9k existing snapshots have no provenance group. They must still load."""

    def test_unidentified_run_writes_no_group(self, tmp_path, monkeypatch):
        """An empty group would make 'unknown' indistinguishable from 'failed to write'."""
        path = _save(tmp_path, env={}, monkeypatch=monkeypatch)

        with h5py.File(path, "r") as hf:
            assert PROVENANCE_GROUP not in hf

    def test_snapshot_without_provenance_still_loads(self, tmp_path, monkeypatch):
        """THE arm that matters: if ``provenance`` ever became a required group, the
        entire archive would stop loading at once."""
        path = _save(tmp_path, env={}, monkeypatch=monkeypatch)

        serializer = CascadeHDF5Serializer()
        loaded = serializer.load_network(str(path), restore_multiprocessing=False)

        assert loaded is not None, "a pre-D-C snapshot must still load"
        assert loaded.provenance is None, "absence must read as unknown, not as an error"
        assert serializer.verify_saved_network(str(path))["valid"] is True

    def test_provenance_is_not_a_required_group(self):
        """Anti-resurrection: guards the format contract directly, so a future edit to
        ``_validate_format`` cannot quietly strand the archive."""
        import inspect

        body = inspect.getsource(CascadeHDF5Serializer._validate_format_detail)
        assert PROVENANCE_GROUP not in body, "provenance must never be a required group — every pre-D-C snapshot lacks it"

    def test_absent_provenance_is_not_an_integrity_finding(self, tmp_path, monkeypatch):
        """Missing identity is not corruption; it must not trip the D-E gates."""
        path = _save(tmp_path, env={}, monkeypatch=monkeypatch)

        serializer = CascadeHDF5Serializer()
        network = serializer.load_network(str(path), restore_multiprocessing=False)
        with h5py.File(path, "r") as hf:
            assert serializer._check_integrity(hf, network) == []


class TestWriteProvenanceUnit:
    """The writer, exercised directly."""

    def test_explicit_provenance_overrides_env(self, tmp_path, monkeypatch):
        for key, value in FULL_ENV.items():
            monkeypatch.setenv(key, value)
        path = tmp_path / "explicit.h5"
        with h5py.File(path, "w") as hf:
            assert write_provenance(hf, {"run_id": "explicit-wins"}) is True
        with h5py.File(path, "r") as hf:
            recovered = read_provenance(hf)
        assert recovered["run_id"] == "explicit-wins"
        assert "experiment" not in recovered

    def test_empty_provenance_writes_nothing(self, tmp_path):
        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as hf:
            assert write_provenance(hf, {}) is False
            assert PROVENANCE_GROUP not in hf

    def test_read_of_absent_group_is_none(self, tmp_path):
        path = tmp_path / "bare.h5"
        with h5py.File(path, "w") as hf:
            hf.attrs["format"] = "juniper.cascor"
        with h5py.File(path, "r") as hf:
            assert read_provenance(hf) is None
