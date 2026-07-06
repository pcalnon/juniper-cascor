#!/usr/bin/env python
"""Regression tests for the 2026-07 latent-defect repairs (defects verified on main @ 0a44938).

Pins the REAL success behavior that was previously unreachable:

1. ``restore_snapshot`` — used to execute ``cls.__dict__.update(...)`` on the
   read-only class ``mappingproxy``, so every load-then-restore raised
   ``AttributeError``, fell into the ``except``, and returned ``False``; the
   success path was dead code. It now returns the restored network. The
   round-trip test below fails against the pre-fix code (it gets ``False``).
2. ``list_hdf5_snapshots`` — used to call the undefined
   ``HDF5Utils.list_hdf5_files``, so every existing-directory call raised
   ``AttributeError`` and returned ``[]`` via the ``except`` fallback. The
   helper now exists; the listing tests below fail against the pre-fix code.
3. ``calculate_accuracy`` — the dead ``if x is None or y is None:`` guard
   (unreachable after the tuple-index defaulting above it) was removed; the
   valid-tensor behavior is pinned unchanged here.
4. ``_init_logging_system`` — the real body was monkey-patched away for the
   entire unit run by the autouse ``_cache_logging_system`` fixture; the test
   below executes the genuine method via the ``real_init_logging_system``
   conftest fixture so regressions in it are visible again, while the
   suite-wide fast-logging fixture stays in place for every other test.
"""
import pathlib

import pytest
import torch

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from snapshots.snapshot_utils import HDF5Utils

pytestmark = pytest.mark.unit


def _make_config(**overrides):
    defaults = {
        "input_size": 2,
        "output_size": 2,
        "random_seed": 42,
        "candidate_pool_size": 2,
        "candidate_epochs": 3,
        "output_epochs": 3,
        "max_hidden_units": 2,
        "patience": 1,
    }
    defaults.update(overrides)
    return CascadeCorrelationConfig(**defaults)


def _make_network(**overrides):
    return CascadeCorrelationNetwork(config=_make_config(**overrides))


# ---------------------------------------------------------------------------
# Defect 1: restore_snapshot must actually deliver a restored network
# ---------------------------------------------------------------------------
class TestRestoreSnapshotRoundTrip:
    """restore_snapshot success path: save to tmp_path, restore, compare state."""

    def test_restore_snapshot_round_trip_returns_restored_network(self, tmp_path):
        """A saved snapshot restores to a network whose state matches the original."""
        network = _make_network()
        snapshot_path = network.create_snapshot(snapshot_dir=tmp_path)
        assert snapshot_path is not None
        assert snapshot_path.exists()

        restored = CascadeCorrelationNetwork.restore_snapshot(snapshot_path=snapshot_path, restore_multiprocessing=False)

        assert restored is not False, "restore_snapshot returned False for a valid snapshot (the pre-fix dead success path)"
        assert isinstance(restored, CascadeCorrelationNetwork)
        assert restored is not network
        assert restored.input_size == network.input_size
        assert restored.output_size == network.output_size
        assert len(restored.hidden_units) == len(network.hidden_units)
        assert torch.equal(restored.output_weights.detach(), network.output_weights.detach())
        if network.output_bias is not None:
            assert torch.equal(restored.output_bias.detach(), network.output_bias.detach())

    def test_restore_snapshot_failure_paths_still_return_false(self, tmp_path):
        """The pinned False-on-error contract is preserved by the fix."""
        assert CascadeCorrelationNetwork.restore_snapshot(snapshot_path=None) is False
        assert CascadeCorrelationNetwork.restore_snapshot(snapshot_path=tmp_path / "missing.h5") is False


# ---------------------------------------------------------------------------
# Defect 2: list_hdf5_snapshots / HDF5Utils.list_hdf5_files must actually list
# ---------------------------------------------------------------------------
class TestListHdf5Snapshots:
    """Success path: HDF5 files in an existing directory are returned."""

    def test_list_hdf5_files_returns_hdf5_paths(self, tmp_path):
        """HDF5Utils.list_hdf5_files returns the .h5/.hdf5 paths, sorted, and nothing else."""
        (tmp_path / "a.h5").touch()
        (tmp_path / "b.hdf5").touch()
        (tmp_path / "not_a_snapshot.txt").touch()

        files = HDF5Utils.list_hdf5_files(tmp_path)

        assert [p.name for p in files] == ["a.h5", "b.hdf5"]
        assert all(isinstance(p, pathlib.Path) for p in files)
        assert all(p.parent == tmp_path for p in files)

    def test_list_hdf5_files_missing_directory_returns_empty(self, tmp_path):
        """A nonexistent directory yields an empty list, not an exception."""
        assert HDF5Utils.list_hdf5_files(tmp_path / "does_not_exist") == []

    def test_list_hdf5_snapshots_returns_files_in_existing_directory(self, tmp_path):
        """list_hdf5_snapshots reaches its success path (pre-fix: except-fallback to [])."""
        network = _make_network()
        (tmp_path / "snap_one.h5").touch()
        (tmp_path / "snap_two.hdf5").touch()
        (tmp_path / "readme.txt").touch()

        files = network.list_hdf5_snapshots(tmp_path)

        assert sorted(p.name for p in files) == ["snap_one.h5", "snap_two.hdf5"], "list_hdf5_snapshots returned a wrong/empty list (the pre-fix except fallback)"

    def test_list_hdf5_snapshots_nonexistent_directory_still_returns_empty(self):
        """The pinned empty-list contract for a missing directory is preserved."""
        network = _make_network()
        assert network.list_hdf5_snapshots("/nonexistent/directory/for/defect2") == []


# ---------------------------------------------------------------------------
# Defect 3: dead None guard removed from calculate_accuracy
# ---------------------------------------------------------------------------
class TestCalculateAccuracyValidPath:
    """Valid-tensor behavior must be unchanged by the dead-guard removal."""

    def test_calculate_accuracy_valid_tensors(self):
        """calculate_accuracy returns a float in [0, 1] for well-formed input."""
        network = _make_network()
        torch.manual_seed(42)
        x = torch.randn(8, network.input_size)
        y = torch.zeros(8, network.output_size)
        y[torch.arange(8), torch.randint(0, network.output_size, (8,))] = 1.0

        accuracy = network.calculate_accuracy(x, y)

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_calculate_accuracy_mismatched_batch_still_raises(self):
        """The shape-mismatch ValueError (which also serves the None-x case) is preserved."""
        network = _make_network()
        x = torch.randn(10, network.input_size)
        y = torch.randn(5, network.output_size)
        with pytest.raises(ValueError):
            network.calculate_accuracy(x, y)


# ---------------------------------------------------------------------------
# Defect 4: the genuine _init_logging_system body must run under pytest
# ---------------------------------------------------------------------------
class TestRealInitLoggingSystem:
    """Execute the real (un-patched) _init_logging_system against a tmp_path config."""

    def test_real_init_logging_system_builds_logger_from_config(self, real_init_logging_system, tmp_path):
        """The real body builds a fresh LogConfig from THIS network's config and wires its logger in.

        The suite-wide fast fixture would instead attach the session-cached
        LogConfig and a no-op logger, so every assertion below distinguishes
        the genuine body from the patched one.
        """
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        network = _make_network(log_file_name="latent_defect_d4", log_file_path=str(log_dir), log_level_name="WARNING")

        real_init_logging_system(network)

        assert network.log_file_name == "latent_defect_d4"
        assert network.log_file_path == str(log_dir)
        assert network.log_level_name == "WARNING"
        assert network.log_config is not None
        assert network.log_config.get_log_file_path() == str(log_dir), "log_config was not built from this network's config (fast-fixture cached LogConfig?)"
        assert network.logger is network.log_config.get_logger(), "logger was not wired from the freshly built LogConfig (no-op fast-fixture logger?)"
        assert type(network.logger).__name__ != "_NoOpLogger"
        assert network.logger.level == network.log_config.get_log_level()
